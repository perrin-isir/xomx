import numpy as np
from sklearn.utils.random import check_random_state
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Tuple
from xomx.tools.utils import _to_dense


class Encoder(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray]:
        x = nn.Dense(features=self.hidden_dims[0])(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.hidden_dims[1])(x)
        x = nn.gelu(x)
        mu = nn.Dense(features=self.latent_dim)(x)
        log_var = nn.Dense(features=self.latent_dim)(x)
        return mu, log_var


class Decoder(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        z = nn.Dense(features=self.hidden_dims[0])(z)
        z = nn.gelu(z)
        z = nn.Dense(features=self.hidden_dims[1])(z)
        z = nn.gelu(z)
        x = nn.Dense(features=self.output_dim)(z)
        return x


class VAE(nn.Module):
    latents: int = 20
    output_dim: int = 100
    hidden_dims: tuple = (100, 100)

    def setup(self):
        hidden_dims_encoder = list(self.hidden_dims)
        hidden_dims_decoder = list(self.hidden_dims)
        self.encoder = Encoder(hidden_dims_encoder, self.latents, name="encoder")
        self.decoder = Decoder(hidden_dims_decoder, self.output_dim, name="decoder")

    def __call__(self, x, z_rng):
        mean, log_var = self.encoder(x)
        z = reparameterize(z_rng, mean, log_var)
        recon_x = self.decoder(z)
        return recon_x, mean, log_var

    def encode(self, params, x):
        mean, log_var = self.bind({"params": params}).encoder(x)
        return mean

    def generate(self, params, z):
        return nn.sigmoid(self.bind({"params": params}).decoder(z))


def reparameterize(rngen, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rngen, logvar.shape)
    return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))


@partial(jax.jit, static_argnames=["latents", "output_dim", "hidden_dims", "beta"])
def train_step(latents, output_dim, hidden_dims, beta, state, batch, z_rng):
    def loss_fn(params):
        recon_x, mean, logvar = VAE(
            latents=latents, output_dim=output_dim, hidden_dims=hidden_dims
        ).apply({"params": params}, batch, z_rng)
        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + beta * kld_loss
        return loss, {"loss": loss}

    grads, aux = jax.grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), aux


class BetaVAE:
    def __init__(
        self,
        adata,
        n_components=2,
        beta=1.0,
        hidden_dims=(100, 100),
        learning_rate=1e-3,
        batch_size=80,
        epsilon=1e-9,
        random_state=None,
    ):
        self.adata = adata
        self.random_state = random_state
        self.n_components = n_components
        self.beta = beta
        self.hidden_dims = hidden_dims
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.output_dim = adata.n_vars
        self.learning_rate = learning_rate
        max_val = adata.X.max()
        min_val = adata.X.min()
        self.data = np.asarray(
            (_to_dense(adata.X) - min_val + self.epsilon)
            / (max_val - min_val + self.epsilon)
        ).copy()
        self.vae = VAE(
            latents=self.n_components,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
        )
        self.np_random_state = check_random_state(random_state)
        self.rng = random.PRNGKey(self.np_random_state.randint(1_000_000))
        self.state = self.init_state()

    def init_state(self):
        self.rng, key = random.split(self.rng)
        return train_state.TrainState.create(
            apply_fn=self.vae.apply,
            params=self.vae.init(
                key, jnp.ones((self.batch_size, self.output_dim)), self.rng
            )["params"],
            tx=optax.adam(learning_rate=self.learning_rate),
        )

    def fit_transform(self, iterations=10_000):
        print_iter = max(int(iterations / 40.0), 1)
        for i in range(iterations):
            batch_idxs = self.np_random_state.choice(
                self.adata.uns["train_indices"]
                if "train_indices" in self.adata.uns
                else np.arange(self.adata.n_obs),
                size=self.batch_size,
                replace=True,
            )
            batch = self.data[batch_idxs]
            self.rng, key = random.split(self.rng)
            self.state, info = train_step(
                self.n_components,
                self.output_dim,
                self.hidden_dims,
                self.beta,
                self.state,
                batch,
                key,
            )
            if not i % print_iter:
                print(f"iteration {i}/{iterations}, loss: {info['loss']}")
        return np.array(
            self.vae.encode(self.state.params, self.data)[:, 0 : self.n_components]
        )

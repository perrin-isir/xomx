import sys
import xomx
import anndata
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

xd = anndata.AnnData(data.data, dtype=np.float32)
xd.obs["labels"] = np.where(data.target, "benign", "malignant")
xd.var_names = data["feature_names"]
xd.obs_names = ["obs" + s for s in xd.obs_names]
xd.uns["all_labels"] = xomx.tl.all_labels(xd.obs["labels"])
xd.uns["var_indices"] = xomx.tl.var_indices(xd)
xd.uns["obs_indices"] = xomx.tl.obs_indices(xd)
xd.uns["obs_indices_per_label"] = xomx.tl.indices_per_label(xd.obs["labels"])
only_benign = xd[xd.uns["obs_indices_per_label"]["benign"]]
xd.var["mean_values"] = xomx.tl.var_mean_values(only_benign)
xd.var["standard_deviations"] = xomx.tl.var_standard_deviations(only_benign)
# normalize data
xd.X = (xd.X - np.array(xd.var["mean_values"])) / np.array(
    xd.var["standard_deviations"]
)

xd.obs["deviation_index"] = np.copy((np.abs(xd.X)).mean(axis=1))

if len(sys.argv) > 1 and sys.argv[1] in ["bokeh", "matplotlib"]:
    xomx.pl.force_extension(sys.argv[1])
else:
    backend = ""
    while backend not in ("bokeh", "matplotlib"):
        backend = input(
            "Please choose a backend for the plots (type 'bokeh' or 'matplotlib):\n"
        )
        backend = backend.strip("'")
        backend = backend.strip('"')
    xomx.pl.force_extension(backend)

xomx.pl.colormap()

xomx.pl.scatter(
    xd,
    lambda idx: xd.var["mean_values"][idx],
    lambda idx: xd.var["standard_deviations"][idx],
    obs_or_var="var",
    pointsize=6,
    xlog_scale=True,
    ylog_scale=True,
    xlabel="mean values",
    ylabel="standard deviations",
    title="standard deviations vs. mean values",
    subset_indices=np.random.choice(
        xd.n_vars, size=int(xd.n_vars * 0.8), replace=False
    ),
    width=900,
    height=600,
)

xomx.pl.plot(
    xd,
    lambda idx: xd.obs["deviation_index"][idx],
    obs_or_var="obs",
    pointsize=4,
    xlabel="samples",
    ylabel="deviation index",
    title="",
    subset_indices=np.random.choice(xd.n_obs, size=int(xd.n_obs * 0.8), replace=False),
    width=900,
    height=600,
)

threshold = 0.2
scores_raw = xd.obs["deviation_index"] / xd.obs["deviation_index"].max()
indices = (
    xd.uns["obs_indices_per_label"]["benign"]
    + xd.uns["obs_indices_per_label"]["malignant"]
)
scores = scores_raw[indices]
xomx.pl.plot_scores(
    xd,
    scores=scores,
    score_threshold=threshold,
    indices=indices,
    label=None,
    pointsize=5,
    text_complements=[
        " | above threshold" if scores_raw[i] > threshold else " | below threshold"
        for i in indices
    ],
    lines=False,
    yticks=None,
    ylabel="",
    title="",
    width=900,
    height=600,
)

xomx.pl.plot_var(
    xd,
    features=None,
    pointsize=5,
    xlabel="",
    ylabel="",
    title="",
    subset_indices=None,
    equal_size=False,
    width=900,
    height=600,
)

xomx.pl.plot_var(
    xd,
    features="area error",
    pointsize=5,
    xlabel="",
    ylabel="",
    title="area error",
    subset_indices=None,
    equal_size=False,
    width=900,
    height=600,
)

xd.obsm["TSNE"] = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=3
).fit_transform(xd.X)

xomx.pl.plot_2d_obsm(
    xd,
    "TSNE",
    color_var=None,
    pointsize=4,
    xlabel="",
    ylabel="",
    title="",
    subset_indices=None,
    height=750,
)

xd.obs["colors"] = np.copy(xd[:, "area error"].X.clip(-np.infty, 15.0).reshape(-1))

xomx.pl.plot_2d_obsm(
    xd,
    "TSNE",
    color_var=None,
    pointsize=4,
    xlabel="",
    ylabel="",
    title="",
    subset_indices=None,
    height=750,
)

xd.obsm["TSNE3D"] = TSNE(
    n_components=3, learning_rate="auto", init="random", perplexity=3
).fit_transform(xd.X)

xomx.pl.plot_3d_obsm(
    xd,
    "TSNE3D",
    color_var=None,
    pointsize=4,
    xlabel="",
    ylabel="",
    title="",
    subset_indices=None,
    height=750,
)

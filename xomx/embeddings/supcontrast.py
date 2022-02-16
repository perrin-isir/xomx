# This file is derived from SupContrast, a PyTorch implementation of "Supervised
# Contrastive Learning" (https://arxiv.org/abs/2004.11362) available
# at: https://github.com/HobbitLong/SupContrast.
# It was released under the following licence:
#
# BSD 2-Clause License
#
# Copyright (c) 2020, Yonglong Tian
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import numpy as np
import sys


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = 1
        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class DataloadBalanced:
    def __init__(self, labels, indices_per_label, batchsize):
        self.indices_per_label = indices_per_label
        self.batchsize = batchsize
        self.idxs = {}
        self.sizes = {}
        self.total_length = 0
        self.labels = labels
        if self.labels is None:
            self.keys = list(self.indices_per_label.keys())
        else:
            self.keys = self.labels
        for key in self.keys:
            self.idxs[key] = 0
            np.random.shuffle(self.indices_per_label[key])
            self.sizes[key] = len(self.indices_per_label[key])
            self.total_length += self.sizes[key]
        self.length = self.total_length // self.batchsize
        self.index = 0

    def len(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.total_length:
            halfbatch = np.random.choice(self.keys,
                                         size=self.batchsize // 2,
                                         replace=True)
            fullbatch = np.hstack((halfbatch, halfbatch))
            batch = []
            for key in fullbatch:
                batch.append(self.indices_per_label[key][self.idxs[key]])
                self.idxs[key] = (self.idxs[key] + 1) % self.sizes[key]
                self.index += 1
                if self.idxs[key] == 0:
                    np.random.shuffle(self.indices_per_label[key])
            return batch
        self.index = 0
        raise StopIteration


def fanin_init(tensor: torch.Tensor) -> torch.Tensor:
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def initialize_hidden_layer(layer: torch.nn.Module, b_init_value: float = 0.1):
    fanin_init(layer.weight)
    layer.bias.data.fill_(b_init_value)


def initialize_last_layer(layer: torch.nn.Module, init_w: float = 1e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_size = 128

        self.dp1 = nn.Dropout(p=0.8)
        self.dp2 = nn.Dropout(p=0.8)

        # encoder
        self.l1 = nn.Linear(440, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, self.embedding_size)

        # head
        self.l4 = nn.Linear(self.embedding_size, 64)
        self.l5 = nn.Linear(64, 32)

        initialize_hidden_layer(self.l1)
        initialize_hidden_layer(self.l2)
        initialize_last_layer(self.l3)
        initialize_hidden_layer(self.l4)
        initialize_hidden_layer(self.l5)

    def forward(self, x):
        x = x.float()

        # encode
        x = F.relu(self.l1(x))
        x = self.dp1(x)
        x = F.relu(self.l2(x))
        x = self.dp2(x)
        x = self.l3(x)

        # head
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x

    def encode(self, x):
        x = x.float()

        # encode
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class SupContrast:
    def __init__(
            self,
            adata: sc.AnnData,
            labels=None,
            batch_size=256,
            random_state=None,
    ):
        self.adata = adata
        assert (
                "all_labels" in adata.uns
                and "train_indices" in adata.uns
                and "test_indices" in adata.uns
                and "train_indices_per_label" in adata.uns
                and "test_indices_per_label" in adata.uns
        )
        self.int_labels = {}
        for lbl in self.adata.uns['all_labels']:
            self.int_labels[lbl] = (self.adata.uns['all_labels'] == lbl).argmax()
        self.labels = labels
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loader = DataloadBalanced(self.labels,
                                       adata.uns['train_indices_per_label'],
                                       batch_size)
        self.model = Model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-3,
                                          weight_decay=1e-5)
        self.criterion = SupConLoss(temperature=0.2).to(self.device)

    @staticmethod
    def save_model(model, optimizer, save_file):
        print('==> Saving...')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, save_file)
        del state

    def save(self, filename):
        self.save_model(self.model, self.optimizer, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    @staticmethod
    def train_(loader, adata, int_labels, model, criterion, optimizer, device, epoch):
        """one epoch training"""
        model.train()
        loss = None

        for idx, indices in enumerate(loader):
            inputs = torch.tensor(adata.X[indices]).to(device)
            l_labels = adata.obs['labels'][indices]
            labels = torch.tensor([int_labels[elt] for elt in l_labels]).to(device)

            # compute loss
            features = model(inputs)
            loss = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train: [epoch {0}]({1} gradient steps)]\t'
              'loss {loss:.3f}'.format(epoch,
                                       loader.len(),
                                       loss=loss.item()))
        sys.stdout.flush()

    def train(self, n_epochs):
        for epoch in range(1, n_epochs + 1):
            self.train_(self.loader,
                        self.adata,
                        self.int_labels,
                        self.model,
                        self.criterion,
                        self.optimizer,
                        self.device,
                        epoch)

            # if epoch % opt.save_freq == 0:
            #     save_file = os.path.join(
            #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            #     # save_model(model, optimizer, opt, epoch, save_file)
            #     save_model(my_model, my_optimizer, opt, epoch, save_file)

    def encode(self, inputs):
        return self.model.encode(
            torch.tensor(inputs).to(self.device)).detach().cpu().numpy()

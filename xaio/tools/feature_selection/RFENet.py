import os
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from xaio.tools.basic_tools import (
    confusion_matrix,
    naive_feature_selection,
    plot_scores,
)
from joblib import dump, load
from IPython import embed as e

assert e


class RFENet:
    def __init__(
        self,
        data,
        annotation,
        init_selection_size=4000,
        batch_size=128,
    ):
        self.data = data
        self.annotation = annotation
        self.init_selection_size = init_selection_size
        (
            self.current_feature_indices,
            self.data_train,
            self.target_train,
            self.data_test,
            self.target_test,
        ) = naive_feature_selection(
            self.data, self.annotation, self.init_selection_size
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.net = None
        self.confusion_matrix = None
        self.eval_batch_inputs = None
        self.eval_batch_targets = None
        self.log = []

    def init(self):
        self.fit(self.data_train, self.target_train)
        self.confusion_matrix = confusion_matrix(
            self.net, self.data_test, self.target_test
        )
        self.log.append(
            {
                "feature_indices": self.current_feature_indices,
                "confusion_matrix": self.confusion_matrix,
            }
        )

    def select_features(self, n):
        assert n <= self.data_train.shape[1]
        if n > 50:
            reduced_feats = (
                torch.argsort(torch.linalg.norm(self.net.fc1.weight, dim=0))[-n:]
                .clone()
                .detach()
                .numpy()
            )
        else:
            fiv = self.compute_permutation_importance()
            reduced_feats = np.argsort(fiv)[-n:]
        self.current_feature_indices = np.take(
            self.current_feature_indices, reduced_feats, axis=0
        )
        self.data_train = np.take(
            self.data_train.transpose(), reduced_feats, axis=0
        ).transpose()
        self.data_test = np.take(
            self.data_test.transpose(), reduced_feats, axis=0
        ).transpose()
        self.fit(self.data_train, self.target_train)
        self.confusion_matrix = confusion_matrix(
            self.net, self.data_test, self.target_test
        )
        self.log.append(
            {
                "feature_indices": self.current_feature_indices,
                "confusion_matrix": self.confusion_matrix,
            }
        )
        return self.confusion_matrix

    def fit(self, data_train, target_train, epochs=50, learning_rate=1e-4):
        input_dim = data_train.shape[1]
        self.net = NNet(self.device, input_dim)
        self.net = self.net.to(self.device)
        class0 = np.where(target_train == 0)[0]
        class1 = np.where(target_train == 1)[0]
        l0 = len(class0)
        l1 = len(class1)
        if l1 < l0 // 2:
            class1 = np.tile(class1, l0 // l1)
        elif l0 < l1 // 2:
            class0 = np.tile(class0, l1 // l0)
        class0and1 = np.hstack((class0, class1))
        inputs = torch.tensor(data_train[class0and1], dtype=torch.float32)
        inputs = inputs.to(self.device)
        targets = torch.tensor(target_train[class0and1], dtype=torch.long)
        targets = targets.to(self.device)
        eval_batch_indices = torch.randint(0, inputs.shape[0], (10 * self.batch_size,))
        self.eval_batch_inputs = inputs[eval_batch_indices, :]
        self.eval_batch_targets = targets[eval_batch_indices]
        train = TensorDataset(inputs, targets)
        dl = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        for j in range(epochs):
            for i, batch in enumerate(dl):
                loss = criterion(self.net(batch[0]), batch[1])
                loss.backward()
                optimizer.step()
            # print(j, confusion_matrix(self.net, self.data_test, self.target_test))
            print(f"\rEpoch {j+1}/{epochs}", end="", flush=True)
        print("\n")

    def predict(self, x):
        return self.net.predict(x)

    def score(self, x):
        return self.net.score(x)

    def save(self, fpath):
        sdir = fpath + "/" + self.__class__.__name__
        os.makedirs(sdir, exist_ok=True)
        self.save_nets([self.net], sdir)
        dump(self.log, sdir + "/log.joblib")

    def load(self, fpath):
        sdir = fpath + "/" + self.__class__.__name__
        if os.path.isfile(sdir + "/net_1.pth") and os.path.isfile(sdir + "/log.joblib"):
            self.log = load(sdir + "/log.joblib")
            feat_indices = np.copy(self.log[-1]["feature_indices"])
            featpos = {
                self.current_feature_indices[i]: i
                for i in range(len(self.current_feature_indices))
            }
            reduced_feats = np.array([featpos[i] for i in feat_indices])
            self.data_train = np.take(
                self.data_train.transpose(), reduced_feats, axis=0
            ).transpose()
            self.data_test = np.take(
                self.data_test.transpose(), reduced_feats, axis=0
            ).transpose()
            self.current_feature_indices = feat_indices
            self.net = NNet(self.device, self.data_train.shape[1])
            self.load_nets([self.net], sdir)
            return True
        else:
            return False

    @staticmethod
    def save_nets(nets, directory):
        for i, net in enumerate(nets):
            filename = "net_" + str(i + 1)
            os.makedirs(directory, exist_ok=True)
            torch.save(net.state_dict(), "%s/%s.pth" % (directory, filename))

    def load_nets(self, nets, directory):
        for i, net in enumerate(nets):
            filename = "net_" + str(i + 1)
            net.load_state_dict(
                torch.load(
                    "%s/%s.pth" % (directory, filename), map_location=self.device
                )
            )

    def compute_permutation_importance(self):
        input_dim = self.eval_batch_inputs.shape[1]
        feature_importance_vec = np.zeros(input_dim)
        criterion = nn.CrossEntropyLoss()
        for feature_to_eval in range(input_dim):
            r = torch.randperm(self.eval_batch_inputs.shape[0])
            self.eval_batch_inputs[:, feature_to_eval] = self.eval_batch_inputs[
                r, feature_to_eval
            ]
            feature_importance_vec[feature_to_eval] = criterion(
                self.net(self.eval_batch_inputs), self.eval_batch_targets
            ).item()
            self.eval_batch_inputs[:, feature_to_eval] = self.eval_batch_inputs[
                torch.argsort(r), feature_to_eval
            ]
        return feature_importance_vec

    def plot(self, annotation=None, save_dir=None):
        res = self.score(self.data_test)
        plot_scores(self.data, res, 0.0, self.data.test_indices, annotation, save_dir)


class NNet(nn.Module):
    def __init__(self, device, input_dim):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        h = functional.relu(self.fc1(x))
        h = functional.relu(self.fc2(h))
        h = self.fc3(h)
        return h

    def predict(self, x):
        h = torch.tensor(x)
        h = h.to(self.device)
        h = self.forward(h)
        return np.array(torch.argmax(h, dim=1))

    def score(self, x):
        h = torch.tensor(x)
        h = h.to(self.device)
        h = self.forward(h)
        hloss = nn.CrossEntropyLoss()
        tmph = [
            -hloss(h[i : i + 1], torch.ones(1, dtype=torch.int64)).item()
            + hloss(h[i : i + 1], torch.zeros(1, dtype=torch.int64)).item()
            for i in range(len(h))
        ]
        return np.array(tmph)

import os
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import xaio
import scanpy as sc

# from xaio.tools.basic_tools import (
#     confusion_matrix,
#     naive_feature_selection,
#     plot_scores,
# )
from joblib import dump, load


class RFEExtraTrees:
    def __init__(
        self,
        adata: sc.AnnData,
        label,
        n_estimators=450,
        random_state=None,
    ):
        self.adata = adata
        assert (
            "train_indices" in adata.uns
            and "test_indices" in adata.uns
            and "train_indices_per_label" in adata.uns
            and "test_indices_per_label" in adata.uns
        )
        self.label = label
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.current_feature_indices = range(adata.n_vars)
        self.data_train = adata[adata.uns["train_indices"], :].X.copy()
        self.data_test = adata[adata.uns["test_indices"], :].X.copy()
        self.target_train = np.zeros(adata.n_obs)
        self.target_train[adata.uns["train_indices_per_label"][label]] = 1.0
        self.target_train = np.take(
            self.target_train, adata.uns["train_indices"], axis=0
        )
        self.target_test = np.zeros(adata.n_obs)
        self.target_test[adata.uns["test_indices_per_label"][label]] = 1.0
        self.target_test = np.take(self.target_test, adata.uns["test_indices"], axis=0)
        self.forest = None
        self.confusion_matrix = None
        self.log = []

    def init(self):
        self.forest = ExtraTreesClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.forest.fit(self.data_train, self.target_train)
        self.confusion_matrix = xaio.tl.confusion_matrix(
            self.forest, self.data_test, self.target_test
        )
        self.log.append(
            {
                "feature_indices": self.current_feature_indices,
                "confusion_matrix": self.confusion_matrix,
            }
        )

    def select_features(self, n):
        assert n <= self.data_train.shape[1]
        sorted_feats = np.argsort(self.forest.feature_importances_)[::-1]
        reduced_feats = list(sorted_feats[:n])
        self.current_feature_indices = np.take(
            self.current_feature_indices, reduced_feats, axis=0
        )
        self.data_train = np.take(
            self.data_train.transpose(), reduced_feats, axis=0
        ).transpose()
        self.data_test = np.take(
            self.data_test.transpose(), reduced_feats, axis=0
        ).transpose()
        self.forest = ExtraTreesClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.forest.fit(self.data_train, self.target_train)
        self.confusion_matrix = xaio.tl.confusion_matrix(
            self.forest, self.data_test, self.target_test
        )
        self.log.append(
            {
                "feature_indices": self.current_feature_indices,
                "confusion_matrix": self.confusion_matrix,
            }
        )
        return self.confusion_matrix

    def predict(self, x):
        if len(x.shape) > 0:
            if x.shape[1] == self.adata.n_vars:
                x_tmp = np.take(
                    x.transpose(), self.current_feature_indices, axis=0
                ).transpose()
                return self.forest.predict(x_tmp)
        return self.forest.predict(x)

    def score(self, x):
        if len(x.shape) < 2:
            x_tmp = np.expand_dims(x, axis=0)
        else:
            x_tmp = x
        if x_tmp.shape[1] == self.adata.n_vars:
            x_tmp = np.take(
                x_tmp.transpose(), self.current_feature_indices, axis=0
            ).transpose()
        return (
            np.array(
                sum(
                    self.forest.estimators_[i].predict(x_tmp)
                    for i in range(self.forest.n_estimators)
                )
            )
            / self.forest.n_estimators
        )

    def save(self, fpath):
        sdir = fpath
        os.makedirs(sdir, exist_ok=True)
        dump(self.forest, os.path.join(sdir, "model.joblib"))
        dump(self.log, os.path.join(sdir, "log.joblib"))

    def load(self, fpath):
        # The initialization before load() must be the same as the initialization of
        # the RFEExtraTrees object that was saved (but init() does not need to be
        # executed).
        sdir = fpath
        if os.path.isfile(os.path.join(sdir, "model.joblib")) and os.path.isfile(
            os.path.join(sdir, "log.joblib")
        ):
            self.log = load(os.path.join(sdir, "log.joblib"))
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
            self.forest = load(os.path.join(sdir, "model.joblib"))
            return True
        else:
            return False

    def plot(self, label=None, save_dir=None):
        res = self.score(self.data_test)
        xaio.pl.plot_scores(
            self.adata, res, 0.5, self.adata.uns["test_indices"], label, save_dir
        )


def load_RFEExtraTrees(
    fpath, adata: sc.AnnData, label, n_estimators=450, random_state=None
) -> RFEExtraTrees:
    rfeet = RFEExtraTrees(adata, label, n_estimators, random_state)
    rfeet.load(fpath)
    return rfeet

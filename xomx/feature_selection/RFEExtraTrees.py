import os
import copy
import numpy as np
from xomx.tools.utils import _to_dense, confusion_matrix
from xomx.plotting.basic_plot import plot_scores
from joblib import dump, load
from typing import List, Union, Optional


class RFEExtraTrees:
    def __init__(
        self,
        adata,
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
        self.init_selection_size = None
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.current_feature_indices = np.arange(adata.n_vars)
        self.data_train = np.asarray(
            _to_dense(adata[adata.uns["train_indices"], :].X).copy()
        )
        self.data_test = np.asarray(
            _to_dense(adata[adata.uns["test_indices"], :].X).copy()
        )
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

    def init(
        self, init_selection_size=None, rank: Union[np.ndarray, List, None] = None
    ):
        self.init_selection_size = init_selection_size
        if self.init_selection_size is not None:
            list_features = rank[: self.init_selection_size]
            assert (
                "var_indices" in self.adata.uns
            ), 'self.adata.uns["var_indices"] must exist.'
            selected_feats = np.array(
                [self.adata.uns["var_indices"][feat] for feat in list_features]
            )
            self.select_features(self.init_selection_size, selected_feats)
        else:
            self.select_features(
                self.data_train.shape[1], np.arange(self.data_train.shape[1])
            )

    def select_features(self, n, selected_feats=None):
        from sklearn.ensemble import ExtraTreesClassifier  # lazy import

        if n < self.data_train.shape[1]:
            if selected_feats is not None:
                assert len(selected_feats) == n
                reduced_feats = selected_feats
            else:
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
        self.confusion_matrix = confusion_matrix(
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
        if len(x.shape) < 2:
            x_tmp = np.expand_dims(x, axis=0)
        else:
            x_tmp = x
        if x_tmp.shape[1] == self.adata.n_vars:
            x_tmp = np.take(
                x_tmp.transpose(), self.current_feature_indices, axis=0
            ).transpose()
        return self.forest.predict(x_tmp)

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
        dump(self.forest, os.path.join(sdir, "forest.joblib"))
        dump(self.log, os.path.join(sdir, "log.joblib"))
        dump(self.label, os.path.join(sdir, "label.joblib"))
        dump(self.init_selection_size, os.path.join(sdir, "init_selection_size.joblib"))
        dump(self.n_estimators, os.path.join(sdir, "n_estimators.joblib"))
        dump(self.random_state, os.path.join(sdir, "random_state.joblib"))

    def _load(self, fpath):
        # _load() does not load self.adata and self.label,
        # so they must be given at __init__
        sdir = fpath
        if os.path.isfile(os.path.join(sdir, "forest.joblib")):
            self.init_selection_size = load(
                os.path.join(sdir, "init_selection_size.joblib")
            )
            self.n_estimators = load(os.path.join(sdir, "n_estimators.joblib"))
            self.random_state = load(os.path.join(sdir, "random_state.joblib"))
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
            self.forest = load(os.path.join(sdir, "forest.joblib"))
            return True
        else:
            return False

    def copy(self):
        return copy.deepcopy(self)

    def plot(
        self,
        label=None,
        *,
        pointsize: int = 5,
        output_file: Optional[str] = None,
        title: str = "",
        random_subset_size: Optional[int] = None,
        rng=None,
        width: int = 900,
        height: int = 600,
    ):
        from sklearn.utils.validation import check_random_state  # lazy import

        if random_subset_size is None:
            res = self.score(self.data_test)
            indices = self.adata.uns["test_indices"]
        else:
            tmp_rng = check_random_state(rng)
            idxs = sorted(
                tmp_rng.choice(
                    len(self.adata.uns["test_indices"]),
                    random_subset_size,
                    replace=False,
                )
            )
            res = self.score(self.data_test[idxs])
            indices = self.adata.uns["test_indices"][idxs]
        plot_scores(
            self.adata,
            res,
            0.5,
            indices,
            label,
            pointsize=pointsize,
            output_file=output_file,
            title=title,
            ylabel="scores",
            width=width,
            height=height,
        )


def load_RFEExtraTrees(
    fpath,
    adata,
) -> RFEExtraTrees:
    label = load(os.path.join(fpath, "label.joblib"))
    rfeet = RFEExtraTrees(adata, label)
    rfeet._load(fpath)
    return rfeet

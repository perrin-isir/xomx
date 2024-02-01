import os
import copy
import numpy as np
from xomx.tools.utils import _to_dense, confusion_matrix
from xomx.plotting.basic_plot import plot_scores
from joblib import dump, load
from typing import Optional


class ExtraTrees:
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
        self.n_estimators = n_estimators
        self.random_state = random_state
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

    def train(self):
        from sklearn.ensemble import ExtraTreesClassifier  # lazy import

        self.forest = ExtraTreesClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.forest.fit(self.data_train, self.target_train)
        self.confusion_matrix = confusion_matrix(
            self.forest, self.data_test, self.target_test
        )
        return self.confusion_matrix

    def predict(self, x):
        if len(x.shape) < 2:
            x_tmp = np.expand_dims(x, axis=0)
        else:
            x_tmp = x
        return self.forest.predict(x_tmp)

    def score(self, x):
        if len(x.shape) < 2:
            x_tmp = np.expand_dims(x, axis=0)
        else:
            x_tmp = x
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
        dump(self.label, os.path.join(sdir, "label.joblib"))
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


def load_ExtraTrees(
    fpath,
    adata,
) -> ExtraTrees:
    label = load(os.path.join(fpath, "label.joblib"))
    et = ExtraTrees(adata, label)
    et._load(fpath)
    return et

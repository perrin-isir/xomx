import numpy as np
import xaio
from scipy.sparse import issparse


def _to_dense(x):
    if issparse(x):
        return x.todense()
    else:
        return x


class ScoreBasedMulticlass:
    def __init__(self, adata, annotations, binary_classifiers):
        self.adata = adata
        self.annotations = annotations
        self.binary_classifiers = binary_classifiers

    def predict(self, x):
        scores = {}
        for annot in self.annotations:
            scores[annot] = self.binary_classifiers[annot].score(x)
        predictions = np.argmax([scores[annot] for annot in self.annotations], axis=0)
        return np.array([self.annotations[i] for i in predictions])

    def plot(self, label=None, save_dir=None):
        res = self.predict(_to_dense(self.adata[self.adata.uns["test_indices"], :].X))
        xaio.pl.plot_scores(
            self.adata,
            res.astype(np.float),
            -0.5,
            self.adata.uns["test_indices"],
            label,
            save_dir,
        )

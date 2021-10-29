import numpy as np
import xaio
import scipy


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

    def score(self, x):
        scores = {}
        for annot in self.annotations:
            scores[annot] = self.binary_classifiers[annot].score(x)
        scores_list = [scores[annot] for annot in self.annotations]
        predictions = np.argmax(scores_list, axis=0).astype(np.float)
        add_score = np.max(scipy.special.softmax(scores_list, axis=0), axis=0).astype(
            np.float
        )
        maxas = max(add_score)
        minas = min(add_score)
        add_score = (add_score - minas) / (maxas - minas)
        return predictions + add_score

    def plot(self, label=None, save_dir=None):
        # res = self.predict(
        #     xaio.tl._to_dense_to_dense(
        #     self.adata[self.adata.uns["test_indices"], :].X)
        # )
        res = self.score(
            xaio.tl._to_dense(self.adata[self.adata.uns["test_indices"], :].X)
        )
        xaio.pl.plot_scores(
            self.adata,
            res.astype(np.float),
            -0.5,
            self.adata.uns["test_indices"],
            label,
            save_dir,
        )

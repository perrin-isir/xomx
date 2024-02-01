import numpy as np
from xomx.tools.utils import _to_dense
from xomx.plotting.basic_plot import plot_scores
from scipy.special import softmax
from typing import Optional


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

    def _pred_score(self, x):
        scores = {}
        for annot in self.annotations:
            scores[annot] = self.binary_classifiers[annot].score(x)
        scores_list = [scores[annot] for annot in self.annotations]
        predictions = np.argmax(scores_list, axis=0)
        add_score = np.max(softmax(scores_list, axis=0), axis=0).astype(float)
        maxas = max(add_score)
        minas = min(add_score)
        add_score = ((add_score - minas) / (maxas - minas),)
        return predictions, predictions.astype(float) + add_score

    def plot(
        self,
        label=None,
        *,
        pointsize: int = 5,
        output_file: Optional[str] = None,
        title: str = "",
        random_subset_size=None,
        rng=None,
        width: int = 900,
        height: int = 600,
    ):
        if random_subset_size is None:
            indices = self.adata.uns["test_indices"]
        else:
            idxs = sorted(
                rng.choice(
                    len(self.adata.uns["test_indices"]),
                    random_subset_size,
                    replace=False,
                )
            )
            indices = self.adata.uns["test_indices"][idxs]
        predictions, res = self._pred_score(
            np.asarray(_to_dense(self.adata[indices, :].X))
        )
        plot_scores(
            self.adata,
            res.astype(float),
            None,
            indices,
            label,
            pointsize=pointsize,
            output_file=output_file,
            text_complements=[
                " | prediction: " + str(att_) for att_ in self.annotations[predictions]
            ],
            lines=True,
            yticks=self.annotations,
            ylabel="predictions",
            title=title,
            width=width,
            height=height,
        )

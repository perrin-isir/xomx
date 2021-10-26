import numpy as np

# import os
from IPython import embed as e

assert e


class ScoreBasedMulticlass:
    def __init__(self, annotations, binary_classifiers):
        self.annotations = annotations
        self.binary_classifiers = binary_classifiers

    def predict(self, x):
        scores = {}
        for i, annot in enumerate(self.annotations):
            scores[annot] = self.binary_classifiers[i].score(x)
        predictions = np.argmax([scores[annot] for annot in self.annotations], axis=0)
        return np.array([self.annotations[i] for i in predictions])

    # def save(self, fpath):
    #     np.save(
    #         os.path.join(fpath, "annotations.npy"),
    #         self.annotations,
    #     )
    #     for i, annot in enumerate(self.annotations):
    #         assert annot in self.binary_classifiers
    #         sdir = os.path.join(fpath, "binary_classifier_" + str(i))
    #         self.binary_classifiers[annot].save(sdir)
    #
    # def load(self, fpath):
    #     self.annotations = np.load(os.path.join(fpath, "annotations.npy"),
    #                                allow_pickle=True
    #                                ).item()
    #     self.binary_classifiers = {}
    #     for i, annot in enumerate(self.annotations):
    #         sdir = os.path.join(fpath, "binary_classifier_" + str(i))
    #         assert self.binary_classifiers[annot].load(sdir)
    #     return True

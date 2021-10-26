from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xaio.tools.basic_tools import (
    confusion_matrix,
    feature_selection_from_list,
    plot_scores,
)
import numpy as np
from IPython import embed as e

assert e


class LinearSGD:
    def __init__(self, data, max_iter=1000, tol=1e-3):
        self.data = data
        self.clf = make_pipeline(
            StandardScaler(), SGDClassifier(max_iter=max_iter, tol=tol)
        )

    def fit(self, x, y, x_test, y_test):
        self.clf.fit(x, y)
        return confusion_matrix(self.clf, x_test, y_test)

    def predict(self, x):
        return self.clf.predict(x)

    def fit_list(self, transcripts_list, annotation):
        (
            feature_indices,
            train_indices,
            test_indices,
            data_train,
            target_train,
            data_test,
            target_test,
        ) = feature_selection_from_list(self.data, annotation, transcripts_list)
        return self.fit(data_train, target_train, data_test, target_test)

    def plot(self, x, indices, annotation=None, save_dir=None):
        res = self.clf.decision_function(x)
        plot_scores(self.data, res, 0.0, indices, annotation, save_dir)

    def plot_list(self, transcripts_list, annotation=None, save_dir=None):
        transcripts_indices = np.copy(transcripts_list)
        for i in range(len(transcripts_indices)):
            if (
                type(transcripts_indices[i]) == str
                or type(transcripts_indices[i]) == np.str_
            ):
                transcripts_indices[i] = self.data.feature_shortnames_ref[
                    transcripts_indices[i]
                ]
        transcripts_indices = np.array(transcripts_indices).astype(int)
        test_indices = sum(self.data.test_indices_per_annotation.values(), [])
        data_test = np.take(
            np.take(self.data.data.transpose(), transcripts_indices, axis=0),
            test_indices,
            axis=1,
        ).transpose()
        self.plot(data_test, test_indices, annotation, save_dir)

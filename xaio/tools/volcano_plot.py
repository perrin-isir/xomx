import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from IPython import embed as e

assert e

# , chi2, f_classif
# from sklearn.feature_selection import f_classif

# from IPython import embed as e
# from sklearn import linear_model
# from regressors import stats
# import scipy.stats as stat

#
#
# class LinearRegression(linear_model.LinearRegression):
#     """
#     LinearRegression class after sklearn's, but calculate t-statistics
#     and p-values for model coefficients (betas).
#     Additional attributes available after .fit()
#     are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
#     which is (n_features, n_coefs)
#     This class sets the intercept to 0 by default, since usually we include it
#     in X.
#     """
#
#     # nothing changes in __init__
#     def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
#                  n_jobs=1):
#         self.fit_intercept = fit_intercept
#         self.normalize = normalize
#         self.copy_X = copy_X
#         self.n_jobs = n_jobs
#
#     def fit(self, X, y, n_jobs=1):
#         self = super(LinearRegression, self).fit(X, y, n_jobs)
#
#         # Calculate SSE (sum of squared errors)
#         # and SE (standard error)
#         sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(
#             X.shape[0] - X.shape[1])
#         se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
#
#         # compute the t-statistic for each feature
#         self.t = self.coef_ / se
#         # find the p-value for each feature
#         self.p = np.squeeze(
#             2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
#         return self


class VolcanoPlot:
    def __init__(self, data, annotation, threshold=1e-5):
        self.data = data
        self.annotation = annotation
        self.threshold = threshold
        self.log2_foldchange = None
        self.log10_pvalues = None
        self.ok_data = None
        self.ok_target = None
        self.ok_indices = None

    def init(self, feature_indices=None):
        reference_values = self.data.feature_mean_values
        on_annotation_values = (
            self.data.std_values_on_training_sets[self.annotation]
            * self.data.feature_standard_deviations
            + self.data.feature_mean_values
        )
        if feature_indices is not None:
            reference_values = reference_values[feature_indices]
            on_annotation_values = on_annotation_values[feature_indices]
        ok1_indices = np.where(reference_values > self.threshold)[0]
        if feature_indices is not None:
            self.ok_indices = feature_indices[ok1_indices]
        else:
            self.ok_indices = ok1_indices
        reference_values = reference_values[ok1_indices]
        on_annotation_values = on_annotation_values[ok1_indices]
        ok2_indices = np.where(on_annotation_values > self.threshold)[0]
        reference_values = reference_values[ok2_indices]
        on_annotation_values = on_annotation_values[ok2_indices]
        self.ok_indices = self.ok_indices[ok2_indices]
        self.log2_foldchange = np.log2(on_annotation_values / reference_values)
        self.ok_data = np.take(
            self.data.data.transpose(), self.ok_indices, axis=0
        ).transpose()
        self.ok_target = np.zeros(self.data.nr_samples)
        self.ok_target[self.data.train_indices_per_annotation[self.annotation]] = 1.0
        self.ok_target[self.data.test_indices_per_annotation[self.annotation]] = 1.0
        fscores, pvalues = f_regression(self.ok_data, self.ok_target)
        self.log10_pvalues = -np.log10(pvalues + 1e-45)

    def plot(self, feature_list=[], save_dir=None):
        fig, ax = plt.subplots()
        idxs = np.sort(feature_list)
        k = 0
        args_ok = np.argsort(self.ok_indices)
        colors = np.zeros(len(self.ok_indices))
        sizes = np.ones(len(self.ok_indices)) * 5
        for i in range(len(self.ok_indices)):
            if idxs[k] < self.ok_indices[args_ok[i]]:
                if k < len(idxs) - 1:
                    k = k + 1
            if idxs[k] == self.ok_indices[args_ok[i]]:
                colors[args_ok[i]] = 1
                sizes[args_ok[i]] = 35
        sc = ax.scatter(
            self.log2_foldchange, self.log10_pvalues, c=colors, cmap="coolwarm", s=sizes
        )
        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(-100, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        ann.set_visible(False)

        def update_annot(ind, sc):
            pos = sc.get_offsets()[ind["ind"][0]]
            ann.xy = pos
            # text = "{}".format(self.data.transcripts[self.ok_indices[["ind"][0]]])
            # text = "{}".format(self.data.transcripts[["ind"][0]])
            text = "{}".format(self.data.feature_names[self.ok_indices[ind["ind"][0]]])
            ann.set_text(text)

        def hover(event):
            vis = ann.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind, sc)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        ann.set_visible(False)
                        fig.canvas.draw_idle()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind, sc)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        ann.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir + "/volcano_plot.png", dpi=200)
        else:
            plt.show()

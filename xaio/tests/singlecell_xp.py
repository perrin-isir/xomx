import numpy as np
from xaio.xaio_config import output_dir, xaio_tag
from xaio.tools.basic_tools import (
    XAIOData,
    confusion_matrix,
    matthews_coef,
)
from sklearn.cluster import KMeans
from xaio.tools.feature_selection.RFEExtraTrees import RFEExtraTrees

# from tools.basic_tools import (
#     FeatureTools,
#     confusion_matrix,
#     matthews_coef,
#     umap_plot,
# )

# from tools.feature_selection.RFEExtraTrees import RFEExtraTrees

# from tools.feature_selection.RFENet import RFENet

# from tools.classifiers.LinearSGD import LinearSGD
# import os

from IPython import embed as e

assert e

# _ = RFEExtraTrees, RFENet

data = XAIOData()
data.save_dir = output_dir + "/dataset/scRNASeq/"
data.load(["raw", "std", "log"])

# data = loadscRNASeq("log")
# data = loadscRNASeq("raw")
# data = loadscRNASeq()

# data.reduce_features(np.where(data.nr_non_zero_samples > 2)[0])
#
mitochondrial_genes = data.regex_search(r"\|MT\-")
# mt_percents = np.array(
#     [
#         data.percentage_feature_set(mitochondrial_genes, i)
#         for i in range(data.nr_samples)
#     ]
# )
#
# data.reduce_samples(np.where(mt_percents < 0.05)[0])
# data.reduce_samples(np.where(data.nr_non_zero_features < 2500)[0])


def tsums(i):
    return data.total_feature_sums[i]


def mt_p(i):
    return data.feature_values_ratio(mitochondrial_genes, i)


def nzfeats(i):
    return data.nr_non_zero_features[i]


def stdval(i):
    return data.feature_standard_deviations[i]


def mval(i):
    return data.feature_mean_values[i]


# data.function_plot(tsums, "samples")
#
# data.function_scatter(tsums, nzfeats, "samples")
#
# data.function_scatter(mval, stdval, "features")

# data.function_plot(lambda i: data.data[i, data.feature_shortnames_ref['MALAT1']],
#                    "samples", violinplot_=False)

data.reduce_features(np.argsort(data.feature_standard_deviations)[-4000:])

n_clusters = 6

kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data.data_array["log"])
data.sample_annotations = kmeans.labels_
data.compute_all_annotations()
data.compute_sample_indices_per_annotation()
data.compute_train_and_test_indices_per_annotation()
data.compute_std_values_on_training_sets()
data.compute_std_values_on_training_sets_argsort()

data.save_dir = output_dir + "/dataset/scRNASeqKMEANS/"
data.save()

e()
quit()

# list_genes = ["FTL", "LYZ"]
#
#
# def feature_func(x, annot):
#     return data.log_data[x, data.feature_shortnames_ref[annot]]
#
#
# fun_list = [lambda x: feature_func(x, 0), lambda x: feature_func(x, 1)]

# e()
# quit()

gene_list = []

feature_selector = np.empty(n_clusters, dtype=object)

for annotation in range(n_clusters):
    feature_selector[annotation] = RFEExtraTrees(
        data, annotation, init_selection_size=4000
    )

    print("Initialization...")
    feature_selector[annotation].init()
    for siz in [100, 30, 20, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selector[annotation].select_features(siz)
        cm = confusion_matrix(
            feature_selector[annotation],
            feature_selector[annotation].data_test,
            feature_selector[annotation].target_test,
        )
        print("MCC score:", matthews_coef(cm))
    # feature_selector.save(save_dir)
    print("Done.")
    feature_selector[annotation].plot()
    # print("MCC score:", matthews_coef(cm))

    print(feature_selector[annotation].current_feature_indices)
    gene_list = gene_list + [
        data.feature_names[
            feature_selector[annotation].current_feature_indices[i]
        ].split("|")[1]
        for i in range(len(feature_selector[annotation].current_feature_indices))
    ]
    print(gene_list)

e()

data.feature_plot(gene_list, "log")

e()

# ft = FeatureTools(data1)

save_dir = output_dir + "/results/scRNASeq/" + xaio_tag + "/"

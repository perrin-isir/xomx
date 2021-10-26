from xaio.xaio_config import output_dir

# from RNASeq_preprocessing.load import loadRNASeq
from xaio.tools.basic_tools import (
    XAIOData,
    confusion_matrix,
    matthews_coef,
    umap_plot,
)
from xaio.tools.feature_selection.RFEExtraTrees import RFEExtraTrees

from xaio.tools.feature_selection.RFENet import RFENet

# from tools.classifiers.LinearSGD import LinearSGD
# import os

from IPython import embed as e

assert e

_ = RFEExtraTrees, RFENet

xaio_tag = "xaio_tag_1"

# data = loadRNASeq("log")
# data = loadRNASeq("raw")
# ft = FeatureTools(data)

data = XAIOData()
data.save_dir = output_dir + "/dataset/RNASeq/"
data.load(["std"])

# annotation = "Acute myeloid leukemia"
annotation = "Diffuse large B-cell lymphoma"
# annotation = "Glioblastoma multiforme"
# annotation = "Lung adenocarcinoma"
# annotation = "Lung squamous cell carcinoma"
# annotation = "Pheochromocytoma and paraganglioma"
# annotation = "Small cell lung cancer"
# annotation = "Uveal melanoma"
# annotation = "Skin cutaneous melanoma"
# annotation = "Brain lower grade glioma"
# annotation = "TCGA-LGG_Primary Tumor"
# annotation = "Breast invasive carcinoma"

save_dir = (
    output_dir + "/results/RNASeq/" + xaio_tag + "/" + annotation.replace(" ", "_")
)

# feature_selector = RFENet(data, annotation, init_selection_size=4000)
feature_selector = RFEExtraTrees(data, annotation, init_selection_size=4000)
if not feature_selector.load(save_dir):
    print("Initialization...")
    feature_selector.init()
    for siz in [100, 30, 20, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selector.select_features(siz)
        cm = confusion_matrix(
            feature_selector, feature_selector.data_test, feature_selector.target_test
        )
        print("MCC score:", matthews_coef(cm))
    feature_selector.save(save_dir)
print("Done.")
feature_selector.plot()
# print("MCC score:", matthews_coef(cm))

print(feature_selector.current_feature_indices)
print(
    [
        data.feature_names[feature_selector.current_feature_indices[i]]
        for i in range(len(feature_selector.current_feature_indices))
    ]
)

if False:
    umap_plot(data, feature_selector.data_test, feature_selector.test_indices)

# linear_clf = LinearSGD(data)
# cm_linear = linear_clf.fit(
#     feature_selector.data_train,
#     feature_selector.target_train,
#     feature_selector.data_test,
#     feature_selector.target_test,
# )
# print("MCC score (linear fit):", matthews_coef(cm_linear))
# linear_clf.plot(
#     feature_selector.data_test,
#     feature_selector.test_indices,
#     annotation,
# )

# gene_list = [
#     "PTPRZ1",
#     "RP11-977G19.5",
#     "BCAN",
#     "MIR497HG",
#     "MYO16",
#     "RP4-592A1.2",
#     "RPSAP58",
#     "HNRNPA3P6",
#     "HMGN2P5",
#     "RP11-535M15.2",
# ]
# # gene_list = [3301, 47704, 6692, 54294, 583, 16343, 20741, 23090, 34358, 33734]
# cm_linear = linear_clf.fit_list(gene_list, annotation)
# print("MCC score (linear fit):", matthews_coef(cm_linear))
# linear_clf.plot_list(gene_list, None)

e()

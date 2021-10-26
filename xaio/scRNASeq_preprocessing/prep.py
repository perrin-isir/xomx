from xaio.xaio_config import output_dir
from xaio.tools.basic_tools import XAIOData
from xaio.scRNASeq_preprocessing.config import (
    scRNASeq_data,
    scRNASeq_features,
    scRNASeq_barcodes,
)
import csv
import scipy.io
import numpy as np

# from IPython import embed as e

mat = scipy.io.mmread(scRNASeq_data)
feature_ids = [
    row[0] + "|" + row[1] for row in csv.reader(open(scRNASeq_features), delimiter="\t")
]
barcodes = [row[0] for row in csv.reader(open(scRNASeq_barcodes), delimiter="\t")]

data = XAIOData()
data.save_dir = output_dir + "/dataset/scRNASeq/"
data.data_array["raw"] = mat.todense().transpose()
data.nr_features = len(feature_ids)
data.nr_samples = mat.shape[1]
data.feature_names = np.empty((data.nr_features,), dtype=object)
for i in range(data.nr_features):
    data.feature_names[i] = feature_ids[i]
data.sample_ids = np.empty((data.nr_samples,), dtype=object)
for i in range(data.nr_samples):
    data.sample_ids[i] = barcodes[i]
data.compute_sample_indices()
data.compute_feature_mean_values()
data.compute_feature_standard_deviations()
data.compute_feature_shortnames_ref()
data.compute_normalization("std")
data.compute_normalization("log")
data.compute_nr_non_zero_features()
data.compute_nr_non_zero_samples()
data.compute_total_feature_sums()

mitochondrial_genes = data.regex_search(r"\|MT\-")
mt_percents = np.array(
    [data.feature_values_ratio(mitochondrial_genes, i) for i in range(data.nr_samples)]
)

data.reduce_samples(np.where(mt_percents < 0.05)[0])
data.reduce_samples(np.where(data.nr_non_zero_features < 2500)[0])
data.reduce_features(np.where(data.nr_non_zero_samples > 2)[0])

data.save()

# data.mean_expressions = [np.mean(data.raw_data[:, i])
#                                for i in range(data.nr_features)]
# data.std_expressions = [np.std(data.raw_data[:, i]) for i in range(data.nr_features)]
# data.std_data = np.copy(data.raw_data)
# for i in range(data.nr_features):
#     for j in range(data.nr_samples):
#         if data.std_expressions[i] == 0.0:
#             data.std_data[j, i] = 0.0
#         else:
#             data.std_data[j, i] = (
#                 data.std_data[j, i] - data.mean_expressions[i]
#             ) / data.std_expressions[i]
# data.log_data = np.copy(data.raw_data)
# data.epsilon_shift = 1.0
# for i in range(data.nr_features):
#     data.log_data[:, i] = np.log(data.log_data[:, i] + data.epsilon_shift)
# data.maxlog = np.max(data.log_data)
# for i in range(data.nr_features):
#     data.log_data[:, i] = (data.log_data[:, i] - np.log(data.epsilon_shift)) / (
#         data.maxlog - np.log(data.epsilon_shift)
#     )
# data.feature_shortnames_ref = {}
# for i, elt in enumerate(data.feature_names):
#     data.feature_shortnames_ref[elt.split("|")[1]] = i
# data.save()

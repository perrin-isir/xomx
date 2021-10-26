# from xaio import gdc_create_manifest, gdc_create_data_matrix
from xaio import XAIOData, confusion_matrix, matthews_coef, anndata_interface

# from xaio import RFEExtraTrees
# import numpy as np

# from xaio.tools.basic_tools import (
#     XAIOData,
#     confusion_matrix,
#     matthews_coef,
# )
from sklearn.cluster import KMeans
from xaio.tools.feature_selection.RFEExtraTrees import RFEExtraTrees
import scipy.io
import csv
import argparse

# import pandas as pd
import numpy as np
import scanpy as sc
import os
import requests
import tarfile

# import shutil
from IPython import embed as e

assert e

"""
TUTORIAL: PERIPHERAL BLOOD MONONUCLEAR CELLS ANALYSIS

This tutorial is similar to the following tutorial with the R package Seurat:
https://satijalab.org/seurat/articles/pbmc3k_tutorial.html

The objective is to analyze a dataset of Peripheral Blood Mononuclear Cells (PBMC)
freely available from 10X Genomics, composed of 2,700 single cells that were
sequenced on the Illumina NextSeq 500.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", metavar="S", type=int, nargs="?", default=None, help="execute step S"
    )
    parser.add_argument(
        "--savedir",
        default=os.path.join(os.path.expanduser("~"), "results", "xaio", "pbmc"),
        help="directory in which data and outputs will be stored",
    )
    args_ = parser.parse_args()
    return args_


# Unless specified otherwise, the data and outputs will be saved in the
# directory: ~/results/xaio/pbmc
args = get_args()
savedir = args.savedir
os.makedirs(savedir, exist_ok=True)

# We use the file next_step.txt to know which step to execute next. 7 consecutive
# executions of the code complete the 7 steps of the tutorial.
# A specific step can also be chosen using an integer in argument
# (e.g. `python kidney_classif.py 1` to execute step 1).
if args.step is not None:
    assert 1 <= args.step <= 7
    step = args.step
elif not os.path.exists(os.path.join(savedir, "next_step.txt")):
    step = 1
else:
    step = np.loadtxt(os.path.join(savedir, "next_step.txt"), dtype="int")
print("STEP", step)

"""
STEP 1: Load data from the 10X Genomics website, and save it as a XAIOData object.
"""
if step == 1:
    url = (
        "https://cf.10xgenomics.com/samples/cell/pbmc3k/"
        + "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    )
    r = requests.get(url, allow_redirects=True)
    open(os.path.join(savedir, "pbmc3k.tar.gz"), "wb").write(r.content)
    tar = tarfile.open(os.path.join(savedir, "pbmc3k.tar.gz"), "r:gz")
    downloaded_content = {}
    for member in tar.getmembers():
        file = tar.extractfile(member)
        if file is not None:
            downloaded_content[member.name] = file

    # The data matrix is loaded as a SciPy sparse matrix.
    sparse_data_matrix = scipy.io.mmread(
        downloaded_content["filtered_gene_bc_matrices/hg19/matrix.mtx"]
    )

    # List of feature names (genes):
    featurenames = [
        "|".join(row)
        for row in list(
            csv.reader(
                downloaded_content["filtered_gene_bc_matrices/hg19/genes.tsv"]
                .read()
                .decode("utf-8")
                .splitlines(),
                delimiter="\t",
            )
        )
    ]
    # We use gene identifiers of this form:
    #                       <Ensembl gene ID>|<HGNC official gene symbol>
    # Example:
    #                       'ENSG00000243485|MIR1302-10'

    # List of the sample IDs (barcodes):
    sampleids = list(
        csv.reader(
            downloaded_content["filtered_gene_bc_matrices/hg19/barcodes.tsv"]
            .read()
            .decode("utf-8")
            .splitlines(),
            delimiter="\t",
        )
    )

    # Creation of the XAIOData object:
    xd = XAIOData()
    xd.save_dir = savedir

    # Sparse matrices are not supported yet in XAIO, so we convert the data matrix into
    # a dense matrix before copying it to the XAIOData object. It is transposed because
    # XAIO uses the convention: columns=samples and rows=features.
    xd.data_array["raw"] = sparse_data_matrix.todense().transpose()
    xd.feature_names = featurenames
    xd.sample_ids = sampleids
    xd.save(["raw"])

"""
STEP 2: Pre-processing workflow
"""
if step == 2:
    xd = XAIOData()
    xd.save_dir = savedir
    xd.load(["raw"])

    # Look for mitochondrial genes (HGNC official symbol starting with "MT-") with
    # regex_search:
    mitochondrial_genes = xd.regex_search(r"\|MT\-")

    # The following function, for the i-th cell, returns the percentage of
    # mitochondrial counts:
    def mt_ratio(i):
        return xd.feature_values_ratio(mitochondrial_genes, i)

    # To plot mitochondrial count percentages, we use function_plot:
    xd.function_plot(
        mt_ratio,
        samples_or_features="samples",
        violinplot=True,
        ylog_scale=False,
        xlabel="cells",
        ylabel="mitochondrial count percentage",
    )

    # Compute the number of non-zero features for each cell, and the number of
    # cells with non-zero counts for each feature. The values are saved in
    # the xd object.
    xd.compute_nr_non_zero_features()
    xd.compute_nr_non_zero_samples()
    # Compute the total sum of counts for each cell:
    xd.compute_total_feature_sums()

    def nr_non_zero_features(i):
        return xd.nr_non_zero_features[i]

    # Plot the number of non-zero features for each cell:
    xd.function_plot(
        nr_non_zero_features,
        "samples",
        True,
        False,
        xlabel="cells",
        ylabel="number of non-zero features",
    )

    def total_feature_sums(i):
        return xd.total_feature_sums[i]

    # Plot the total number of counts for each sample:
    xd.function_plot(
        total_feature_sums,
        "samples",
        True,
        False,
        xlabel="cells",
        ylabel="total number of counts",
    )

    # For the i-th gene, the following array is the array of fractions of counts in
    # each single cell, across all cells:
    # xd.data_array["raw"][:, i]/xd.total_feature_sums
    # So this function returns the mean fraction of counts across all cells for the
    def mean_count_fractions(i):
        return np.mean(xd.data_array["raw"][:, i] / xd.total_feature_sums)

    # Plot the mean fractions of counts across all cells, for each gene:
    xd.function_plot(
        mean_count_fractions,
        samples_or_features="features",
        violinplot=False,
        xlabel="genes",
        ylabel="mean fraction of counts across all cells",
    )

    # Plot mitochondrial count percentages vs. total number number of counts:
    xd.function_scatter(
        total_feature_sums,
        mt_ratio,
        samples_or_features="samples",
        violinplot=False,
        xlog_scale=False,
        ylog_scale=False,
        xlabel="total number number of counts",
        ylabel="mitochondrial count percentages",
    )

    # Plot the number of non-zero features vs. total number number of counts:
    xd.function_scatter(
        total_feature_sums,
        nr_non_zero_features,
        xlabel="total number number of counts",
        ylabel="number of non-zero features",
    )

    # Filter cells that have 2500 unique feature counts or more:
    xd.reduce_samples(np.where(xd.nr_non_zero_features < 2500)[0])

    # Filter cells that have 200 feature counts or less:
    xd.reduce_samples(np.where(xd.nr_non_zero_features > 200)[0])

    # Filter cells that have >=5% mitochondrial counts:
    xd.reduce_samples(np.where(np.vectorize(mt_ratio)(range(xd.nr_samples)) < 0.05)[0])

    xd.save(["raw"])

"""
STEP 3: Normalizing the data
"""
if step == 3:
    xd = XAIOData()
    xd.save_dir = savedir
    xd.load(["raw"])

    # Normalize total counts to 10000 reads per cell.
    xd.normalize_feature_sums(1e4)

    # Logarithmize the data, and store it in xd.data_array["log1p"]:
    xd.data_array["log1p"] = np.log(1 + xd.data_array["raw"])
    # The log-normalized data is stored in xd.data_array["log"]
    # Log-normalized values are all between 0 and 1.

    # Create an AnnData object that shares the same data as xd.data_array["log1p"].
    ad = anndata_interface(xd.data_array["log1p"])
    # WARNING: changes in ad affect xd.data_array["log1p"], and vice versa.

    # Use scanpy to compute the top 8000 highly variable genes:
    sc.pp.highly_variable_genes(ad, n_top_genes=8000)

    # Array of highly variable genes:
    hv_genes = np.where(ad.var.highly_variable == 1)[0]

    # Keep only highly variable genes.
    xd.reduce_features(hv_genes)

    # Save "raw" and "log1p" data.
    xd.save(["raw", "log1p"], os.path.join(savedir, "xd_small"))

"""
STEP 4:
"""
if step == 4:
    xd = XAIOData()
    xd.save_dir = os.path.join(savedir, "xd_small")

    # Load the "log1p" data:
    xd.load(["log1p"])

    ad = anndata_interface(xd.data_array["log1p"])
    sc.tl.pca(ad, svd_solver="arpack")
    sc.pp.neighbors(ad, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(ad)
    xd.sample_annotations = ad.obs.leiden.to_numpy()

    # n_clusters = 8
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(xd.data_array["log"])
    # xd.sample_annotations = kmeans.labels_

    xd.compute_all_annotations()
    xd.compute_feature_indices()
    xd.compute_sample_indices()
    xd.compute_sample_indices_per_annotation()
    xd.compute_train_and_test_indices(test_train_ratio=0.25)
    xd.save()

    feature_selector = np.empty(len(xd.all_annotations), dtype=object)
    gene_set = set()
    for i in range(len(xd.all_annotations)):
        annotation = xd.all_annotations[i]
        feature_selector[i] = RFEExtraTrees(xd, annotation)
        print("Annotation: " + str(annotation))
        feature_selector[i].init()
        for siz in [100, 30, 20, 12]:
            print("Selecting", siz, "features...")
            feature_selector[i].select_features(siz)
            cm = confusion_matrix(
                feature_selector[i],
                feature_selector[i].data_test,
                feature_selector[i].target_test,
            )
            print("MCC score:", matthews_coef(cm))
        feature_selector[i].save(
            os.path.join(savedir, "xd_small", "feature_selectors", annotation)
        )
        print("Done.")
        selected_gene_list = [
            xd.feature_names[idx_]
            for idx_ in feature_selector[i].current_feature_indices
        ]
        print("Selected genes: ", selected_gene_list)
        gene_set = gene_set.union(selected_gene_list)

"""
STEP 5:
"""
if step == 5:
    xd = XAIOData()
    xd.save_dir = os.path.join(savedir, "xd_small")

    # Load the "log1p" and raw data:
    xd.load(["log1p", "raw"])

    feature_selector = np.empty(len(xd.all_annotations), dtype=object)
    gene_set = set()
    for i in range(len(feature_selector)):
        annotation = xd.all_annotations[i]
        feature_selector[i] = RFEExtraTrees(xd, annotation)
        feature_selector[i].load(
            os.path.join(
                savedir, "xd_small", "feature_selectors", xd.all_annotations[i]
            )
        )
        selected_gene_list = [
            xd.feature_names[idx_]
            for idx_ in feature_selector[i].current_feature_indices
        ]
        gene_set = gene_set.union(selected_gene_list)

    xd.feature_plot(list(gene_set), "log1p")

    # xd.umap_plot(n_neighbors=30)
    # xd.compute_sample_indices_per_annotation()
    # data.compute_train_and_test_indices_per_annotation()
    # data.compute_std_values_on_training_sets()
    # data.compute_std_values_on_training_sets_argsort()

    biomarkers = [
        "ENSG00000168685|IL7R",
        "ENSG00000170458|CD14",
        "ENSG00000090382|LYZ",
        "ENSG00000156738|MS4A1",
        "ENSG00000153563|CD8A",
        "ENSG00000115523|GNLY",
        "ENSG00000105374|NKG7",
        "ENSG00000203747|FCGR3A",
        "ENSG00000166927|MS4A7",
        "ENSG00000179639|FCER1A",
        "ENSG00000101439|CST3",
        "ENSG00000163736|PPBP",
    ]

    e()
    quit()

    # We compute the mean value and standard deviation of gene counts (in raw data):
    xd.compute_feature_mean_values()
    xd.compute_feature_standard_deviations()

    def feature_standard_deviations(i):
        return xd.feature_standard_deviations[i]

    def feature_mean_values(i):
        return xd.feature_mean_values[i]

    # xd.function_scatter(
    #     feature_mean_values,
    #     feature_standard_deviations,
    #     samples_or_features="features",
    #     violinplot=False,
    #     xlog_scale=False,
    #     ylog_scale=False
    # )

    cvals = np.column_stack((xd.feature_standard_deviations, xd.feature_mean_values))

    # from sklearn.preprocessing import power_transform
    from sklearn.preprocessing import PowerTransformer

    pt = PowerTransformer(method="yeo-johnson")
    data = [[1], [3], [5]]
    print(pt.fit(cvals))
    res = pt.transform(cvals)

    # res = cvals

    def fsd(i):
        return res[i][1]

    def fmv(i):
        return res[i][0]

    xd.function_scatter(
        fsd,
        fmv,
        samples_or_features="features",
        violinplot=False,
        xlog_scale=False,
        ylog_scale=False,
    )

    # data = [[1, 2], [3, 2], [4, 5]]
    # print(power_transform(data, method='box-cox'))

    # n_clusters = 6
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(xd.data_array["log"])
    e()

quit()
# data = XAIOData()
# data.save_dir = output_dir + "/dataset/scRNASeq/"
# data.load(["raw", "std", "log"])

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

# data.save_dir = output_dir + "/dataset/scRNASeqKMEANS/"
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
    gene_list += [
        xd.feature_names[idx_] for idx_ in feature_selector[i].current_feature_indices
    ]

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

"""
INCREMENTING next_step.txt
"""
# noinspection PyTypeChecker
np.savetxt(os.path.join(savedir, "next_step.txt"), [min(step + 1, 7)], fmt="%u")

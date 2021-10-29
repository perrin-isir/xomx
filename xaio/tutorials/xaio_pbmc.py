import xaio
import argparse
import numpy as np
import scanpy as sc
import os
import requests
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
# (e.g. `python xaio_pmbc.py 1` to execute step 1).
if args.step is not None:
    assert 1 <= args.step <= 7
    step = args.step
elif not os.path.exists(os.path.join(savedir, "next_step.txt")):
    step = 1
else:
    step = np.loadtxt(os.path.join(savedir, "next_step.txt"), dtype="int")
print("STEP", step)

"""
STEP 1: Load the data from the 10X Genomics website, store it as an AnnData object,
and apply the preprocessing workflow and clustering.
"""
if step == 1:
    url = (
        "https://cf.10xgenomics.com/samples/cell/pbmc3k/"
        + "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    )
    r = requests.get(url, allow_redirects=True)
    open(os.path.join(savedir, "pbmc3k.tar.gz"), "wb").write(r.content)
    os.popen(
        "tar -xzf " + os.path.join(savedir, "pbmc3k.tar.gz") + " -C " + savedir
    ).read()

    xd = sc.read_10x_mtx(
        os.path.join(savedir, "filtered_gene_bc_matrices", "hg19"),
        var_names="gene_symbols",
    )
    xd.var_names_make_unique()

    sc.pp.filter_cells(xd, min_genes=200)
    sc.pp.filter_genes(xd, min_cells=3)
    xd.var["mt"] = xd.var_names.str.startswith(
        "MT-"
    )  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(
        xd, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # The k-th element of the following array is the mean fraction of counts of the
    # k-th gene in each single cell, across all cells:
    mean_count_fractions = np.squeeze(
        np.asarray(
            np.mean(
                xd.X / np.array(xd.obs["total_counts"]).reshape((xd.n_obs, 1)), axis=0
            )
        )
    )

    xaio.pl.function_plot(
        xd,
        lambda idx: mean_count_fractions[idx],
        obs_or_var="var",
        violinplot=False,
        ylog_scale=False,
        xlabel="genes",
        ylabel="mean fractions of counts across all cells",
    )

    xaio.pl.function_plot(
        xd,
        lambda idx: xd.obs["total_counts"][idx],
        obs_or_var="obs",
        violinplot=True,
        ylog_scale=False,
        xlabel="cells",
        ylabel="total number of counts",
    )

    # Plot mitochondrial count percentages vs. total number number of counts:
    xaio.pl.function_scatter(
        xd,
        lambda idx: xd.obs["total_counts"][idx],
        lambda idx: xd.obs["pct_counts_mt"][idx],
        obs_or_var="obs",
        violinplot=False,
        xlog_scale=False,
        ylog_scale=False,
        xlabel="total number number of counts",
        ylabel="mitochondrial count percentages",
    )

    xd = xd[xd.obs.n_genes_by_counts < 2500, :]
    xd = xd[xd.obs.pct_counts_mt < 5, :]
    sc.pp.normalize_total(xd, target_sum=1e4)
    sc.pp.log1p(xd)
    sc.pp.highly_variable_genes(xd, min_mean=0.0125, max_mean=3, min_disp=0.5)
    xd.raw = xd
    xd = xd[:, xd.var.highly_variable]
    sc.pp.regress_out(xd, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(xd, max_value=10)
    sc.tl.pca(xd, svd_solver="arpack")
    sc.pp.neighbors(xd, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(xd)

    # We now retrieve the unfiltered data, and recover the .obsp annotations:
    obsp = xd.obsp.copy()
    xd = xd.raw.to_adata()
    xd.obsp = obsp

    # Compute the dictionary of feature (var) indices:
    xd.uns["var_indices"] = xaio.tl.var_indices(xd)

    # The "leiden" clusters define labels, and XAIO uses labels stored in obs["labels"]:
    xd.obs["labels"] = xd.obs["leiden"]

    # Several plotting functions require the list of all labels and the
    # dictionary of sample indices per label:
    xd.uns["all_labels"] = xaio.tl.all_labels(xd.obs["labels"])
    xd.uns["obs_indices_per_label"] = xaio.tl.indices_per_label(xd.obs["labels"])

    # Plot the expression of the gene NKG7, and group samples by clusters:
    xaio.pl.var_plot(xd, "NKG7")

    # Compute training and test sets:
    xaio.tl.train_and_test_indices(xd, "obs_indices_per_label", test_train_ratio=0.25)

    # Rank the genes for each cluster with t-test:
    sc.tl.rank_genes_groups(xd, "leiden", method="t-test")

    # Saving the AnnData object to the disk:
    xd.write(os.path.join(savedir, "xaio_pbmc.h5ad"))
    print("STEP 1: done")

"""
STEP 2: For every label, train a binary classifier with recursive feature
elimination to determine a discriminative list of 10 features.
"""
if step == 2:
    # Loading the AnnData object:
    xd = sc.read(os.path.join(savedir, "xaio_pbmc.h5ad"), cache=True)

    feature_selector = {}
    # gene_lists = {}
    for label in xd.uns["all_labels"]:
        print("Annotation: " + label)
        feature_selector[label] = xaio.fs.RFEExtraTrees(
            xd,
            label,
            init_selection_size=8000,
            n_estimators=450,
            random_state=0,
        )
        feature_selector[label].init()
        for siz in [100, 30, 20, 15, 10]:
            print("Selecting", siz, "features...")
            feature_selector[label].select_features(siz)
            cm = xaio.tl.confusion_matrix(
                feature_selector[label],
                feature_selector[label].data_test,
                feature_selector[label].target_test,
            )
            print("MCC score:", xaio.tl.matthews_coef(cm))
        feature_selector[label].save(os.path.join(savedir, "feature_selectors", label))
        # gene_lists[label] = [
        #     xd.var_names[idx_]
        #     for idx_ in feature_selector[label].current_feature_indices
        # ]
        print("Done.")

    print("STEP 2: done")

"""
STEP 3: Normalizing the data
"""
if step == 3:
    # Loading the AnnData object:
    xd = sc.read(os.path.join(savedir, "xaio_pbmc.h5ad"), cache=True)

    feature_selector = {}
    gene_list = {}
    for label in xd.uns["all_labels"]:
        feature_selector[label] = xaio.fs.load_RFEExtraTrees(
            os.path.join(savedir, "feature_selectors", label),
            xd,
        )
        gene_list[label] = [
            xd.var_names[idx_]
            for idx_ in feature_selector[label].current_feature_indices
        ]
    sbm = xaio.cl.ScoreBasedMulticlass(xd, xd.uns["all_labels"], feature_selector)

    # new_cluster_names = [
    #     "CD4 T",
    #     "CD14 Monocytes",
    #     "B",
    #     "CD8 T",
    #     "NK",
    #     "FCGR3A Monocytes",
    #     "Dendritic",
    #     "Megakaryocytes",
    # ]
    # xd.rename_categories("labels", new_cluster_names)

    biomarkers = [
        "IL7R",
        "CD14",
        "LYZ",
        "MS4A1",
        "CD8A",
        "GNLY",
        "NKG7",
        "FCGR3A",
        "MS4A7",
        "FCER1A",
        "CST3",
        "PPBP",
    ]

    sc.tl.umap(xd)

    e()
    quit()
#
#     # Look for mitochondrial genes (HGNC official symbol starting with "MT-") with
#     # regex_search:
#     mitochondrial_genes = xd.regex_search(r"\|MT\-")
#
#     # The following function, for the i-th cell, returns the percentage of
#     # mitochondrial counts:
#     def mt_ratio(i):
#         return xd.feature_values_ratio(mitochondrial_genes, i)
#
#     # To plot mitochondrial count percentages, we use function_plot:
#     xd.function_plot(
#         mt_ratio,
#         samples_or_features="samples",
#         violinplot=True,
#         ylog_scale=False,
#         xlabel="cells",
#         ylabel="mitochondrial count percentage",
#     )
#
#     # Compute the number of non-zero features for each cell, and the number of
#     # cells with non-zero counts for each feature. The values are saved in
#     # the xd object.
#     xd.compute_nr_non_zero_features()
#     xd.compute_nr_non_zero_samples()
#     # Compute the total sum of counts for each cell:
#     xd.compute_total_feature_sums()
#
#     def nr_non_zero_features(i):
#         return xd.nr_non_zero_features[i]
#
#     # Plot the number of non-zero features for each cell:
#     xd.function_plot(
#         nr_non_zero_features,
#         "samples",
#         True,
#         False,
#         xlabel="cells",
#         ylabel="number of non-zero features",
#     )
#
#     def total_feature_sums(i):
#         return xd.total_feature_sums[i]
#
#     # Plot the total number of counts for each sample:
#     xd.function_plot(
#         total_feature_sums,
#         "samples",
#         True,
#         False,
#         xlabel="cells",
#         ylabel="total number of counts",
#     )
#
#     # For the i-th gene, the following array is the array of fractions of counts in
#     # each single cell, across all cells:
#     # xd.data_array["raw"][:, i]/xd.total_feature_sums
#     # So this function returns the mean fraction of counts across all cells for the
#     def mean_count_fractions(i):
#         return np.mean(xd.data_array["raw"][:, i] / xd.total_feature_sums)
#
#     # Plot the mean fractions of counts across all cells, for each gene:
#     xd.function_plot(
#         mean_count_fractions,
#         samples_or_features="features",
#         violinplot=False,
#         xlabel="genes",
#         ylabel="mean fraction of counts across all cells",
#     )
#
#     # Plot mitochondrial count percentages vs. total number number of counts:
#     xd.function_scatter(
#         total_feature_sums,
#         mt_ratio,
#         samples_or_features="samples",
#         violinplot=False,
#         xlog_scale=False,
#         ylog_scale=False,
#         xlabel="total number number of counts",
#         ylabel="mitochondrial count percentages",
#     )
#
#     # Plot the number of non-zero features vs. total number number of counts:
#     xd.function_scatter(
#         total_feature_sums,
#         nr_non_zero_features,
#         xlabel="total number number of counts",
#         ylabel="number of non-zero features",
#     )
#
#     # Filter cells that have 2500 unique feature counts or more:
#     xd.reduce_samples(np.where(xd.nr_non_zero_features < 2500)[0])
#
#     # Filter cells that have 200 feature counts or less:
#     xd.reduce_samples(np.where(xd.nr_non_zero_features > 200)[0])
#
#     # Filter cells that have >=5% mitochondrial counts:
#     xd.reduce_samples(
#     np.where(np.vectorize(mt_ratio)(range(xd.nr_samples)) < 0.05)[0])
#
#     xd.save(["raw"])
#
# """
# STEP 3: Normalizing the data
# """
# if step == 3:
#     xd = XAIOData()
#     xd.save_dir = savedir
#     xd.load(["raw"])
#
#     # Normalize total counts to 10000 reads per cell.
#     xd.normalize_feature_sums(1e4)
#
#     # Logarithmize the data, and store it in xd.data_array["log1p"]:
#     xd.data_array["log1p"] = np.log(1 + xd.data_array["raw"])
#     # The log-normalized data is stored in xd.data_array["log"]
#     # Log-normalized values are all between 0 and 1.
#
#     # Create an AnnData object that shares the same data as xd.data_array["log1p"].
#     ad = anndata_interface(xd.data_array["log1p"])
#     # WARNING: changes in ad affect xd.data_array["log1p"], and vice versa.
#
#     # Use scanpy to compute the top 8000 highly variable genes:
#     sc.pp.highly_variable_genes(ad, n_top_genes=8000)
#
#     # Array of highly variable genes:
#     hv_genes = np.where(ad.var.highly_variable == 1)[0]
#
#     # Keep only highly variable genes.
#     xd.reduce_features(hv_genes)
#
#     # Save "raw" and "log1p" data.
#     xd.save(["raw", "log1p"], os.path.join(savedir, "xd_small"))
#
# """
# STEP 4:
# """
# if step == 4:
#     xd = XAIOData()
#     xd.save_dir = os.path.join(savedir, "xd_small")
#
#     # Load the "log1p" data:
#     xd.load(["log1p"])
#
#     ad = anndata_interface(xd.data_array["log1p"])
#     sc.tl.pca(ad, svd_solver="arpack")
#     sc.pp.neighbors(ad, n_neighbors=10, n_pcs=40)
#     sc.tl.leiden(ad)
#     xd.sample_annotations = ad.obs.leiden.to_numpy()
#
#     # n_clusters = 8
#     # kmeans = KMeans(
#     n_clusters=n_clusters, random_state=42).fit(xd.data_array["log"])
#     # xd.sample_annotations = kmeans.labels_
#
#     xd.compute_all_annotations()
#     xd.compute_feature_indices()
#     xd.compute_sample_indices()
#     xd.compute_sample_indices_per_annotation()
#     xd.compute_train_and_test_indices(test_train_ratio=0.25)
#     xd.save()
#
#     feature_selector = np.empty(len(xd.all_annotations), dtype=object)
#     gene_set = set()
#     for i in range(len(xd.all_annotations)):
#         annotation = xd.all_annotations[i]
#         feature_selector[i] = RFEExtraTrees(xd, annotation)
#         print("Annotation: " + str(annotation))
#         feature_selector[i].init()
#         for siz in [100, 30, 20, 12]:
#             print("Selecting", siz, "features...")
#             feature_selector[i].select_features(siz)
#             cm = confusion_matrix(
#                 feature_selector[i],
#                 feature_selector[i].data_test,
#                 feature_selector[i].target_test,
#             )
#             print("MCC score:", matthews_coef(cm))
#         feature_selector[i].save(
#             os.path.join(savedir, "xd_small", "feature_selectors", annotation)
#         )
#         print("Done.")
#         selected_gene_list = [
#             xd.feature_names[idx_]
#             for idx_ in feature_selector[i].current_feature_indices
#         ]
#         print("Selected genes: ", selected_gene_list)
#         gene_set = gene_set.union(selected_gene_list)
#
# """
# STEP 5:
# """
# if step == 5:
#     xd = XAIOData()
#     xd.save_dir = os.path.join(savedir, "xd_small")
#
#     # Load the "log1p" and raw data:
#     xd.load(["log1p", "raw"])
#
#     feature_selector = np.empty(len(xd.all_annotations), dtype=object)
#     gene_set = set()
#     for i in range(len(feature_selector)):
#         annotation = xd.all_annotations[i]
#         feature_selector[i] = RFEExtraTrees(xd, annotation)
#         feature_selector[i].load(
#             os.path.join(
#                 savedir, "xd_small", "feature_selectors", xd.all_annotations[i]
#             )
#         )
#         selected_gene_list = [
#             xd.feature_names[idx_]
#             for idx_ in feature_selector[i].current_feature_indices
#         ]
#         gene_set = gene_set.union(selected_gene_list)
#
#     xd.feature_plot(list(gene_set), "log1p")
#
#     # xd.umap_plot(n_neighbors=30)
#     # xd.compute_sample_indices_per_annotation()
#     # data.compute_train_and_test_indices_per_annotation()
#     # data.compute_std_values_on_training_sets()
#     # data.compute_std_values_on_training_sets_argsort()
#
#     biomarkers = [
#         "ENSG00000168685|IL7R",
#         "ENSG00000170458|CD14",
#         "ENSG00000090382|LYZ",
#         "ENSG00000156738|MS4A1",
#         "ENSG00000153563|CD8A",
#         "ENSG00000115523|GNLY",
#         "ENSG00000105374|NKG7",
#         "ENSG00000203747|FCGR3A",
#         "ENSG00000166927|MS4A7",
#         "ENSG00000179639|FCER1A",
#         "ENSG00000101439|CST3",
#         "ENSG00000163736|PPBP",
#     ]
#
#     e()
#     quit()
#
#     # We compute the mean value and standard deviation of gene counts (in raw data):
#     xd.compute_feature_mean_values()
#     xd.compute_feature_standard_deviations()
#
#     def feature_standard_deviations(i):
#         return xd.feature_standard_deviations[i]
#
#     def feature_mean_values(i):
#         return xd.feature_mean_values[i]
#
#     # xd.function_scatter(
#     #     feature_mean_values,
#     #     feature_standard_deviations,
#     #     samples_or_features="features",
#     #     violinplot=False,
#     #     xlog_scale=False,
#     #     ylog_scale=False
#     # )
#
#     cvals = np.column_stack((xd.feature_standard_deviations, xd.feature_mean_values))
#
#     # from sklearn.preprocessing import power_transform
#     from sklearn.preprocessing import PowerTransformer
#
#     pt = PowerTransformer(method="yeo-johnson")
#     data = [[1], [3], [5]]
#     print(pt.fit(cvals))
#     res = pt.transform(cvals)
#
#     # res = cvals
#
#     def fsd(i):
#         return res[i][1]
#
#     def fmv(i):
#         return res[i][0]
#
#     xd.function_scatter(
#         fsd,
#         fmv,
#         samples_or_features="features",
#         violinplot=False,
#         xlog_scale=False,
#         ylog_scale=False,
#     )
#
#     # data = [[1, 2], [3, 2], [4, 5]]
#     # print(power_transform(data, method='box-cox'))
#
#     # n_clusters = 6
#     # kmeans = KMeans(
#     n_clusters=n_clusters, random_state=42).fit(xd.data_array["log"])
#     e()
#
# quit()
# # data = XAIOData()
# # data.save_dir = output_dir + "/dataset/scRNASeq/"
# # data.load(["raw", "std", "log"])
#
# # data = loadscRNASeq("log")
# # data = loadscRNASeq("raw")
# # data = loadscRNASeq()
#
# # data.reduce_features(np.where(data.nr_non_zero_samples > 2)[0])
# #
# mitochondrial_genes = data.regex_search(r"\|MT\-")
#
#
# # mt_percents = np.array(
# #     [
# #         data.percentage_feature_set(mitochondrial_genes, i)
# #         for i in range(data.nr_samples)
# #     ]
# # )
# #
# # data.reduce_samples(np.where(mt_percents < 0.05)[0])
# # data.reduce_samples(np.where(data.nr_non_zero_features < 2500)[0])
#
#
# def tsums(i):
#     return data.total_feature_sums[i]
#
#
# def mt_p(i):
#     return data.feature_values_ratio(mitochondrial_genes, i)
#
#
# def nzfeats(i):
#     return data.nr_non_zero_features[i]
#
#
# def stdval(i):
#     return data.feature_standard_deviations[i]
#
#
# def mval(i):
#     return data.feature_mean_values[i]
#
#
# # data.function_plot(tsums, "samples")
# #
# # data.function_scatter(tsums, nzfeats, "samples")
# #
# # data.function_scatter(mval, stdval, "features")
#
# # data.function_plot(lambda i: data.data[i, data.feature_shortnames_ref['MALAT1']],
# #                    "samples", violinplot_=False)
#
# data.reduce_features(np.argsort(data.feature_standard_deviations)[-4000:])
#
# n_clusters = 6
#
# kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data.data_array["log"])
# data.sample_annotations = kmeans.labels_
# data.compute_all_annotations()
# data.compute_sample_indices_per_annotation()
# data.compute_train_and_test_indices_per_annotation()
# data.compute_std_values_on_training_sets()
# data.compute_std_values_on_training_sets_argsort()
#
# # data.save_dir = output_dir + "/dataset/scRNASeqKMEANS/"
# data.save()
#
# e()
# quit()
#
# # list_genes = ["FTL", "LYZ"]
# #
# #
# # def feature_func(x, annot):
# #     return data.log_data[x, data.feature_shortnames_ref[annot]]
# #
# #
# # fun_list = [lambda x: feature_func(x, 0), lambda x: feature_func(x, 1)]
#
# # e()
# # quit()
#
# gene_list = []
#
# feature_selector = np.empty(n_clusters, dtype=object)
#
# for annotation in range(n_clusters):
#     feature_selector[annotation] = RFEExtraTrees(
#         data, annotation, init_selection_size=4000
#     )
#
#     print("Initialization...")
#     feature_selector[annotation].init()
#     for siz in [100, 30, 20, 15, 10]:
#         print("Selecting", siz, "features...")
#         feature_selector[annotation].select_features(siz)
#         cm = confusion_matrix(
#             feature_selector[annotation],
#             feature_selector[annotation].data_test,
#             feature_selector[annotation].target_test,
#         )
#         print("MCC score:", matthews_coef(cm))
#     # feature_selector.save(save_dir)
#     print("Done.")
#     feature_selector[annotation].plot()
#     # print("MCC score:", matthews_coef(cm))
#
#     print(feature_selector[annotation].current_feature_indices)
#     gene_list += [
#         xd.feature_names[idx_] for idx_ in feature_selector[i].current_feature_indices
#     ]
#
#     gene_list = gene_list + [
#         data.feature_names[
#             feature_selector[annotation].current_feature_indices[i]
#         ].split("|")[1]
#         for i in range(len(feature_selector[annotation].current_feature_indices))
#     ]
#     print(gene_list)
#
# e()
#
# data.feature_plot(gene_list, "log")
#
# e()
#
# # ft = FeatureTools(data1)

"""
INCREMENTING next_step.txt
"""
# noinspection PyTypeChecker
np.savetxt(os.path.join(savedir, "next_step.txt"), [min(step + 1, 7)], fmt="%u")

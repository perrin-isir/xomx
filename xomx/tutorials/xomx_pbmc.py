import xomx
import numpy as np
import scanpy as sc
import os
import requests

"""
xomx tutorial: preprocessing and clustering 3k PBMCs

This tutorial follows the single cell RNA-seq Scanpy tutorial on 3k PBMCs:
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html.

The objective is to analyze a dataset of Peripheral Blood Mononuclear Cells (PBMC)
freely available from 10X Genomics, composed of 2,700 single cells that were
sequenced on the Illumina NextSeq 500.
We replace some Scanpy plots by interactive xomx plots, and modify the
computation of marker genes. Instead of using a t-test, Wilcoxon-Mann-Whitney test
or logistic regression, we perform recursive feature elimination with
the Extra-Trees algorithm.
"""


# Unless specified otherwise, the data and outputs will be saved in the
# directory: ~/results/xomx/pbmc
args = xomx.tt.get_args("pbmc")
savedir = args.savedir
os.makedirs(savedir, exist_ok=True)

# We use the file next_step.txt to know which step to execute next. 3 consecutive
# executions of the code complete the 3 steps of the tutorial.
# A specific step can also be chosen using an integer in argument
# (e.g. `python xomx_pmbc.py 1` to execute step 1).
step = xomx.tt.step_init(args, 3)

# Setting the pseudo-random number generator
rng = np.random.RandomState(0)

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
    # k-th gene in each single cell, across all cells
    mean_count_fractions = np.squeeze(
        np.asarray(
            np.mean(
                xd.X / np.array(xd.obs["total_counts"]).reshape((xd.n_obs, 1)), axis=0
            )
        )
    )

    # Plot, for all genes, the mean fraction
    # of counts in single cells, across all cells
    xomx.pl.plot(
        xd,
        lambda idx: mean_count_fractions[idx],
        obs_or_var="var",
        violinplot=False,
        ylog_scale=False,
        xlabel="genes",
        ylabel="mean fractions of counts across all cells",
    )

    # Plot the total counts per cell
    xomx.pl.plot(
        xd,
        lambda idx: xd.obs["total_counts"][idx],
        obs_or_var="obs",
        ylog_scale=False,
        xlabel="cells",
        ylabel="total number of counts",
    )

    # Plot mitochondrial count percentages vs total number of counts
    xomx.pl.scatter(
        xd,
        lambda idx: xd.obs["total_counts"][idx],
        lambda idx: xd.obs["pct_counts_mt"][idx],
        obs_or_var="obs",
        xlog_scale=False,
        ylog_scale=False,
        xlabel="total number number of counts",
        ylabel="mitochondrial count percentages",
    )

    # Preprocessing and clustering following the steps of the Scanpy tutorial
    xd = xd[xd.obs.n_genes_by_counts < 2500, :]
    xd = xd[xd.obs.pct_counts_mt < 5, :]
    sc.pp.normalize_total(xd, target_sum=1e4)
    sc.pp.log1p(xd)
    sc.pp.highly_variable_genes(xd, min_mean=0.0125, max_mean=3, min_disp=0.5)
    xd.raw = xd
    xd = xd[:, xd.var.highly_variable]
    sc.pp.regress_out(xd, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(xd, max_value=10)
    sc.tl.pca(xd, svd_solver="arpack", random_state=rng)
    sc.pp.neighbors(xd, n_neighbors=10, n_pcs=40, random_state=rng)
    sc.tl.leiden(xd, random_state=rng)

    # Rename the "leiden" clusters
    new_cluster_names = [
        "CD4 T",
        "CD14 Monocytes",
        "B",
        "CD8 T",
        "NK",
        "FCGR3A Monocytes",
        "Dendritic",
        "Megakaryocytes",
    ]
    xd.rename_categories("leiden", new_cluster_names)

    # We now retrieve the unfiltered data, and recover the .obsp annotations
    obsp = xd.obsp.copy()
    xd = xd.raw.to_adata()
    xd.obsp = obsp

    # Compute the dictionary of feature (var) indices
    xd.uns["var_indices"] = xomx.tl.var_indices(xd)

    # The "leiden" clusters define labels, and xomx uses labels stored in obs["labels"]
    xd.obs["labels"] = xd.obs["leiden"]

    # Several xomx functions require the list of all labels and the
    # dictionary of sample indices per label
    xd.uns["all_labels"] = xomx.tl.all_labels(xd.obs["labels"])
    xd.uns["obs_indices_per_label"] = xomx.tl.indices_per_label(xd.obs["labels"])

    # Compute training and test sets
    xomx.tl.train_and_test_indices(
        xd, "obs_indices_per_label", test_train_ratio=0.25, rng=rng
    )

    # Rank the genes for each cluster with t-test
    sc.tl.rank_genes_groups(xd, "leiden", method="t-test")

    # Saving the AnnData object to the disk
    xd.write(os.path.join(savedir, "xomx_pbmc.h5ad"))
    print("STEP 1: done")

"""
STEP 2: For every label, train a binary classifier with recursive feature
elimination to determine a discriminative list of 10 features.
"""
if step == 2:
    # Loading the AnnData object
    xd = sc.read(os.path.join(savedir, "xomx_pbmc.h5ad"), cache=True)

    # Training feature selectors
    feature_selectors = {}
    for label in xd.uns["all_labels"]:
        print("Label: " + label)
        feature_selectors[label] = xomx.fs.RFEExtraTrees(
            xd,
            label,
            init_selection_size=8000,
            n_estimators=450,
            random_state=rng,
        )
        feature_selectors[label].init()
        for siz in [100, 30, 20, 15, 10]:
            print("Selecting", siz, "features...")
            feature_selectors[label].select_features(siz)
            print(
                "MCC score:",
                xomx.tl.matthews_coef(feature_selectors[label].confusion_matrix),
            )
        feature_selectors[label].save(os.path.join(savedir, "feature_selectors", label))
        print("Done.")

    print("STEP 2: done")

"""
STEP 3: Visualizing results
"""
if step == 3:
    # Loading the AnnData object
    xd = sc.read(os.path.join(savedir, "xomx_pbmc.h5ad"), cache=True)

    # Load feature selectors
    feature_selectors = {}
    gene_dict = {}
    for label in xd.uns["all_labels"]:
        feature_selectors[label] = xomx.fs.load_RFEExtraTrees(
            os.path.join(savedir, "feature_selectors", label),
            xd,
        )
        gene_dict[label] = [
            xd.var_names[idx_]
            for idx_ in feature_selectors[label].current_feature_indices
        ]

    # Multiclass classifier
    sbm = xomx.cl.ScoreBasedMulticlass(xd, xd.uns["all_labels"], feature_selectors)
    sbm.plot()

    # Visualizing 10-gene signatures for CD14 and FCGR3A Monocytes
    sc.pl.dotplot(
        xd,
        gene_dict["CD14 Monocytes"] + gene_dict["FCGR3A Monocytes"],
        groupby="labels",
    )

    # Selected genes in a single list
    all_selected_genes = np.asarray(list(gene_dict.values())).flatten()

    # Known biomarkers
    biomarkers = {
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
    }

    # Interaction between selected genes and known biomarkers
    print(biomarkers.intersection(all_selected_genes))

    # Compute UMAP embedding
    sc.tl.umap(xd, random_state=rng)

    # Interactive UMAP plots
    xomx.pl.plot_2d_obsm(xd, "X_umap")
    xomx.pl.plot_2d_obsm(xd, "X_umap", "CST3")

    # Interactive PCA plots
    xomx.pl.plot_2d_obsm(xd, "X_pca")
    xomx.pl.plot_2d_obsm(xd, "X_pca", "CST3")

    print("STEP 3: done")

"""
INCREMENTING next_step.txt
"""
xomx.tt.step_increment(step, args)

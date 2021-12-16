import xomx
import scanpy as sc
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import shutil

"""
xomx tutorial: constructing diagnostic biomarker signatures.

The objective of this tutorial is to use a recursive feature elimination method on
RNA-seq data from the Cancer Genome Atlas (TCGA) to identify gene biomarker signatures
for the differential diagnosis of three types of kidney cancer: kidney renal clear cell
carcinoma (KIRC), kidney renal papillary cell carcinoma (KIRP), and chromophobe
renal cell carcinoma (KICH).
See xomx_kidney_classif.md for detailed explanations.
"""


# Unless specified otherwise, the data and outputs will be saved in the
# directory: ~/results/xomx/kidney_classif
args = xomx.tt.get_args("kidney_classif")
savedir = args.savedir
os.makedirs(savedir, exist_ok=True)

# We use the file next_step.txt to know which step to execute next. 7 consecutive
# executions of the code complete the 7 steps of the tutorial.
# A specific step can also be chosen using an integer in argument
# (e.g. `python xomx_kidney_classif.py 1` to execute step 1).
step = xomx.tt.step_init(args, 7)

# Setting the pseudo-random number generator
rng = np.random.RandomState(0)

"""
STEP 1: Use the gdc_create_manifest function (from xomx/data_importation/gdc.py)
to create a manifest.txt file that will be used to import data with the GDC
Data Transfer Tool (gdc-client). 10 types of cancers are considered, with
for each of them 150 samples corresponding to cases of adenocarcinomas.
"""
if step == 1:
    disease_type = "Adenomas and Adenocarcinomas"
    # The 3 categories of cancers studied in this tutorial correspond to the following
    # TCGA projects, which are different types of adenocarcinomas
    project_list = ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]
    # Fetch 200 cases of KIRC, 200 cases of KIRP, and 65 cases of KICH from the
    # GDC database
    case_numbers = [200, 200, 65]
    df_list = xomx.di.gdc_create_manifest(
        disease_type,
        project_list,
        case_numbers,
    )
    df = pd.concat(df_list)
    # noinspection PyTypeChecker
    df.to_csv(
        os.path.join(savedir, "manifest.txt"),
        header=True,
        index=False,
        sep="\t",
        mode="w",
    )
    print("STEP 1: done")


"""
STEP 2: Collect the data with gdc-client (which can be downloaded at
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).
If all downloads succeed, 465 directories are created in a temporary directory
named tmpdir_GDCsamples.
"""
if step == 2:
    tmpdir = "tmpdir_GDCsamples"
    os.makedirs(tmpdir, exist_ok=True)
    commandstring = (
        "gdc-client download -d "
        + tmpdir
        + " -m "
        + os.path.join(savedir, "manifest.txt")
    )
    try:
        subprocess.check_call(commandstring, shell=True)
    except subprocess.CalledProcessError:
       print("ERROR: make sure you have downloaded gdc-client" + \
             " and that it is accessible (see xomx_kidney_classif.md)")
       sys.exit()
    print("STEP 2: done")


"""
STEP 3: Gather all individual cases to create the data matrix, and save it
as an AnnData object.
After that, all the individual files imported with gdc-client are erased.
"""
if step == 3:
    tmpdir = "tmpdir_GDCsamples"
    df = xomx.di.gdc_create_data_matrix(
        tmpdir,
        os.path.join(savedir, "manifest.txt"),
    )
    # Drop the last 5 rows containing special information which we will not be used
    df = df.drop(index=df.index[-5:])

    # The dataframe df does not follow the AnnData convention (rows = samples,
    # columns = features), so we transpose it when creating the AnnData object
    xd = sc.AnnData(df.transpose())

    # Make sure that the variable (feature) names are unique
    xd.var_names_make_unique()

    # In order to improve cross-sample comparisons, we normalize the sequencing
    # depth to 1 million.
    # WARNING: basic pre-processing is used here for simplicity, but for more advanced
    # applications, a more sophisticated preprocessing may be required.
    sc.pp.normalize_total(xd, target_sum=1e6)

    # Saving the AnnData object to the disk
    xd.write(os.path.join(savedir, "xomx_kidney_classif.h5ad"))

    # Erase the individual sample directories downloaded with gdc-client
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("STEP 3: done")


"""
STEP 4: Labelling samples.
Labels (annotations) are fetched from the previously created file manifest.txt.
"""
if step == 4:
    # Loading the AnnData object
    xd = sc.read(os.path.join(savedir, "xomx_kidney_classif.h5ad"))
    # Loading the manifest
    manifest = pd.read_table(os.path.join(savedir, "manifest.txt"), header=0)

    # Create a dictionary of labels
    label_dict = {}
    for i in range(xd.n_obs):
        label_dict[manifest["id"][i]] = manifest["annotation"][i]

    label_array = np.array([label_dict[xd.obs_names[i]] for i in range(xd.n_obs)])
    xd.obs["labels"] = label_array

    # Compute the list of different labels
    xd.uns["all_labels"] = xomx.tl.all_labels(xd.obs["labels"])

    # Computing the list of sample (obs) indices for every label
    xd.uns["obs_indices_per_label"] = xomx.tl.indices_per_label(xd.obs["labels"])

    # Saving the AnnData object to the disk
    xd.write(os.path.join(savedir, "xomx_kidney_classif.h5ad"))
    print("STEP 4: done")


"""
STEP 5: Keep only the top 8000 highly variable features,
and randomly separate samples in training and test datasets.
"""
if step == 5:
    xd = sc.read(os.path.join(savedir, "xomx_kidney_classif.h5ad"))

    # Compute the mean and standard deviation (across samples) for all the features
    xd.var["mean_values"] = xomx.tl.var_mean_values(xd)
    xd.var["standard_deviations"] = xomx.tl.var_standard_deviations(xd)

    # Logarithmize the data
    sc.pp.log1p(xd)

    # Compute the top 8000 highly variable features
    sc.pp.highly_variable_genes(xd, n_top_genes=8000)

    # Filter the data to keep only the 8000 highly variable features
    xd = xd[:, xd.var.highly_variable]

    # Compute the dictionary of feature (var) indices
    xd.uns["var_indices"] = xomx.tl.var_indices(xd)

    # Randomly separate samples into training and test sets.
    xomx.tl.train_and_test_indices(xd,
                                   "obs_indices_per_label",
                                   test_train_ratio=0.25,
                                   rng=rng)
    # New annotations after this call:
    # xd.uns["train_indices_per_label"]
    # xd.uns["test_indices_per_label"]
    # xd.uns["train_indices"]
    # xd.uns["test_indices"]

    # Saving the data to a new file
    xd.write(os.path.join(savedir, "xomx_k_c_small.h5ad"))
    print("STEP 5: done")


"""
STEP 6: For every label, train a binary classifier with recursive feature
elimination to determine a discriminative list of 10 features.
"""
if step == 6:
    # Loading the AnnData object
    xd = sc.read(os.path.join(savedir, "xomx_k_c_small.h5ad"))

    # Training feature selectors
    feature_selectors = {}
    for label in xd.uns["all_labels"]:
        print("Label: " + label)
        feature_selectors[label] = xomx.fs.RFEExtraTrees(
            xd,
            label,
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

    print("STEP 6: done")


"""
STEP 7: Visualizing results.
"""
if step == 7:
    # Loading the AnnData object
    xd = sc.read(os.path.join(savedir, "xomx_k_c_small.h5ad"))

    # Plot standard deviation vs mean value for all features
    xomx.pl.function_scatter(
        xd,
        lambda idx: xd.var["mean_values"][idx],
        lambda idx: xd.var["standard_deviations"][idx],
        obs_or_var="var",
        xlog_scale=True,
        ylog_scale=True,
        xlabel="mean values",
        ylabel="standard deviations",
    )

    # Load feature selectors (binary classifiers on selected features)
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

    # Plot results of feature_selectors["TCGA-KIRP"] on the test set
    feature_selectors["TCGA-KIRP"].plot()

    # Create a multiclass classifier based on the 3 binary classifiers
    sbm = xomx.cl.ScoreBasedMulticlass(xd, xd.uns["all_labels"], feature_selectors)
    sbm.plot()

    # Selected genes in a single list
    all_selected_genes = np.asarray(list(gene_dict.values())).flatten()

    # Visualizing all genes
    xomx.pl.var_plot(xd, all_selected_genes)

    # Visualizing the 10-gene signature for "TCGA-KIRP"
    xomx.pl.var_plot(xd, gene_dict["TCGA-KIRP"])

    # Stacked violin plot (using Scanpy)
    sc.pl.stacked_violin(xd, gene_dict["TCGA-KIRP"], groupby="labels", rotation=90)

    # Visualizing the 10-gene signature for "TCGA-KICH"
    xomx.pl.var_plot(xd, gene_dict["TCGA-KICH"])

    # A single feature
    xomx.pl.var_plot(xd, "ENSG00000168269.8")

    # Visualizing the 10-gene signature for "TCGA-KIRC"
    xomx.pl.var_plot(xd, gene_dict["TCGA-KIRC"])

    # Computing and plotting a 2D UMAP embedding
    xd = xd[:, all_selected_genes]
    xd.var_names_make_unique()
    sc.pp.neighbors(xd, n_neighbors=10, n_pcs=40, random_state=rng)
    sc.tl.umap(xd, random_state=rng)
    xomx.pl.plot2d(xd, "X_umap")

    print("STEP 7: done")

"""
INCREMENTING next_step.txt
"""
xomx.tt.step_increment(step, args)

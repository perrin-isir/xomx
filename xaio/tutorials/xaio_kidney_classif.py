import xaio
import scanpy as sc
import argparse
import pandas as pd
import numpy as np
import os
import shutil

"""
TUTORIAL: KIDNEY CANCER CLASSIFICATION

The objective of this tutorial is to create and test a classifier of different
types of kidney cancers based on RNA-Seq data from the Cancer Genome Atlas (TCGA).
See kidney_classif.md for detailed explanations.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", metavar="S", type=int, nargs="?", default=None, help="execute step S"
    )
    parser.add_argument(
        "--savedir",
        default=os.path.join(
            os.path.expanduser("~"), "results", "xaio", "kidney_classif"
        ),
        help="directory in which data and outputs will be stored",
    )
    args_ = parser.parse_args()
    return args_


# Unless specified otherwise, the data and outputs will be saved in the
# directory: ~/results/xaio/kidney_classif
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
STEP 1: Use the gdc_create_manifest function (from xaio/data_importation/gdc.py)
to create a manifest.txt file that will be used to import data with the GDC
Data Transfer Tool (gdc-client). 10 types of cancers are considered, with
for each of them 150 samples corresponding to cases of adenocarcinomas.
"""
if step == 1:
    disease_type = "Adenomas and Adenocarcinomas"
    # The 3 categories of cancers studied in this tutorial correspond to the following
    # TCGA projects, which are different types of adenocarcinomas:
    project_list = ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]
    # We fetch 200 cases of KIRC, 200 cases of KIRP, and 65 cases of KICH from the
    # GDC database:
    case_numbers = [200, 200, 65]
    df_list = xaio.di.gdc_create_manifest(
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
tmpdir = "tmpdir_GDCsamples"
if step == 2:
    os.makedirs(tmpdir, exist_ok=True)
    commandstring = (
        "gdc-client download -d "
        + tmpdir
        + " -m "
        + os.path.join(savedir, "manifest.txt")
    )
    os.system(commandstring)
    print("STEP 2: done")


"""
STEP 3: Gather all individual cases to create the data matrix, and save it in
a folder named "xd".
After that, all the individual files imported with gdc-client are erased.
"""
if step == 3:
    df = xaio.di.gdc_create_data_matrix(
        tmpdir,
        os.path.join(savedir, "manifest.txt"),
    )
    # We drop the last 5 rows containing special information which we will not use:
    df = df.drop(index=df.index[-5:])

    # This dataframe does not follow the AnnData convention (rows = samples,
    # columns = features), so we transpose it when creating the AnnData object:
    xd = sc.AnnData(df.transpose())

    # In order to improve cross-sample comparisons, we normalize the sequencing
    # depth to 1 million.
    # WARNING: basic pre-processing is used here for simplicity, but for more advanced
    # applications, a more sophisticated pre-processing may be required.
    sc.pp.normalize_total(xd, target_sum=1e6)

    # We compute the mean and standard deviation (across samples) for all the features:
    xd.var["mean_values"] = xaio.tl.var_mean_values(xd)
    xd.var["standard_deviations"] = xaio.tl.var_standard_deviations(xd)

    # Saving the AnnData object to the disk:
    xd.write(os.path.join(savedir, "xaio_kidney_classif.h5ad"))

    # We erase the individual sample directories downloaded with gdc-client:
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("STEP 3: done")


"""
STEP 4: Labelling samples.
Labels (annotations) are fetched from the previously created file manifest.txt.
"""
if step == 4:
    # Loading the AnnData object:
    xd = sc.read(os.path.join(savedir, "xaio_kidney_classif.h5ad"))

    manifest = pd.read_table(os.path.join(savedir, "manifest.txt"), header=0)

    # Create a dictionary of labels:
    label_dict = {}
    for i in range(xd.n_obs):
        label_dict[manifest["id"][i]] = manifest["annotation"][i]

    label_array = np.array([label_dict[xd.obs_names[i]] for i in range(xd.n_obs)])
    xd.obs["labels"] = label_array

    # Compute the list of different labels:
    xd.uns["all_labels"] = xaio.tl.all_labels(xd.obs["labels"])

    # Computing the list of sample (obs) indices for every label:
    xd.uns["obs_indices_per_label"] = xaio.tl.indices_per_label(xd.obs["labels"])

    # Saving the AnnData object to the disk:
    xd.write(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
    print("STEP 4: done")


"""
STEP 5: Keep only the top 4000 highly variable features,
and randomly separate samples in training and test datasets.
"""
if step == 5:
    xd = sc.read(os.path.join(savedir, "xaio_kidney_classif.h5ad"))

    # Logarithmize the data
    sc.pp.log1p(xd)
    # Compute the top 4000 highly variable features
    sc.pp.highly_variable_genes(xd, n_top_genes=4000)
    # Filter the data to keep only the 4000 highly variable features
    xd = xd[:, xd.var.highly_variable]

    # Randomly separate samples into train and test sets.
    xaio.tl.train_and_test_indices(xd, "obs_indices_per_label", test_train_ratio=0.25)
    # New annotations after this call:
    # xd.uns["train_indices_per_label"]
    # xd.uns["test_indices_per_label"]
    # xd.uns["train_indices"]
    # xd.uns["test_indices"]

    # Saving the filtered data to a new file:
    xd.write(os.path.join(savedir, "xaio_k_c_small.h5ad"))
    print("STEP 5: done")


"""
STEP 6: For every label, train a binary classifier with recursive feature
elimination to determine a discriminative list of 10 features.
"""
if step == 6:
    xd = sc.read(os.path.join(savedir, "xaio_k_c_small.h5ad"))
    feature_selector = {}
    for label in xd.uns["all_labels"]:
        print("Annotation: " + label)
        feature_selector[label] = xaio.fs.RFEExtraTrees(
            xd,
            label,
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
        feature_selector[label].save(
            os.path.join(savedir, "xd_small", "feature_selectors", label)
        )
        print("Done.")

    print("STEP 6: done")


"""
STEP 7: Visualizing results.
"""
if step == 7:
    xd = sc.read(os.path.join(savedir, "xaio_k_c_small.h5ad"))

    xaio.pl.function_scatter(
        xd,
        lambda idx: xd.var["mean_values"][idx],
        lambda idx: xd.var["standard_deviations"][idx],
        "var",
        xlog_scale=True,
        ylog_scale=True,
    )

    feature_selector = {}
    gene_list = []
    for label in xd.uns["all_labels"]:
        feature_selector[label] = xaio.fs.load_RFEExtraTrees(
            os.path.join(savedir, "xd_small", "feature_selectors", label),
            xd,
            label,
            n_estimators=450,
            random_state=0,
        )
        gene_list += [
            xd.var_names[idx_]
            for idx_ in feature_selector[label].current_feature_indices
        ]

    feature_selector["TCGA-KIRC"].plot()

    xd = xd[:, gene_list]

    xaio.pl.umap_plot(xd)

    # Compute the set of feature (var) indices:
    xd.uns["var_indices"] = xaio.tl.var_indices(xd)
    xaio.pl.var_plot(xd, gene_list)

    xaio.pl.var_plot(xd, "ENSG00000168269.8")  # FOXI1
    # xaio.pl.var_plot(xd, "ENSG00000163435.14")  # ELF3
    xaio.pl.var_plot(xd, "ENSG00000125872.7")  # ELF3
    xaio.pl.var_plot(xd, "ENSG00000185633.9")  # NDUFA4L2

    # Some of the most remarkable genes on this plot:
    # ENSG00000185633.9
    # ENSG00000168269.8 for KICH: FOXI1, known in
    # "Cell-Type-Specific Gene Programs of the Normal Human
    # Nephron Define Kidney Cancer Subtypes"

    # For KIRP: ELF3 ENSG00000163435.14
    # Diagnostic
    # biomarkers
    # for renal cell carcinoma: selection
    # using
    # novel
    # bioinformatics
    # systems
    # for microarray data analysis

    # The Gene ENSG00000185633.9 (NDUFA4L2) seems associated to KIRC
    # (Kidney Renal Clear Cell Carcinoma).
    # This is confirmed by the publication:
    # Role of NADH Dehydrogenase (Ubiquinone) 1 alpha subcomplex 4-like 2 in clear cell
    # renal cell carcinoma
    # xd.feature_plot("ENSG00000185633.9", "raw")


"""
INCREMENTING next_step.txt
"""
# noinspection PyTypeChecker
np.savetxt(os.path.join(savedir, "next_step.txt"), [min(step + 1, 7)], fmt="%u")
# *XAIO Tutorial:* constructing diagnostic biomarker signatures.

-----

The objective of this tutorial is to use a recursive feature elimination method on 
RNA-Seq data to identify gene biomarker signatures for the differential diagnosis of three 
types of kidney cancer: kidney renal clear cell carcinoma (**KIRC**), kidney renal 
papillary cell carcinoma (**KIRP**), and chromophobe renal cell carcinoma (**KICH**).

The recursive feature elimination method is based on 
the [Extra-Trees algorithm](https://link.springer.com/article/10.1007/s10994-006-6226-1)
(and its implementation in 
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)).

### Running the tutorial:
- **Repeated executions of the [xaio_kidney_classif.py](xaio/tutorials/xaio_kidney_classif.py) 
file perform each of the 7 steps of 
the tutorial, one by one.**

- A specific step can also be chosen using an integer
argument. For instance, `python xaio_kidney_classif.py 1` executes the step 1.

### Table of Contents:
+ [Step 1: Preparing the manifest](#s1)
+ [Step 2: Importing the data](#s2)
+ [Step 3: Creating and saving the AnnData object](#s3)
+ [Step 4: Labelling the samples](#s4)
+ [Step 5: Basic preprocessing](#5)
+ [Step 6: Training binary classifiers and performing recursive feature elimination](#s6)
+ [Step 7: Visualizing results](#s7)

### Saving results:

In [xaio_kidney_classif.py](xaio/tutorials/xaio_kidney_classif.py), after the imports, the 
following lines define the string variable `savedir`: the folder 
in which data and outputs will be stored.
```python
args = get_args()
savedir = args.savedir
```
By default, `savedir` is `~/results/xaio/kidney_classif`, but it can be modified using a 
`--savedir` argument in input (e.g. `python xaio_kidney_classif.py --savedir /tmp`).

<a name="s1"></a>
## Step 1: Preparing the manifest

We use the 
[GDC Data Transfer Tool](
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
)
to import data from the Cancer Genome Atlas (TCGA). 
This involves creating a `manifest.txt` file that describes the files to be imported.

The `gdc_create_manifest()` function
facilitates the creation of this manifest. It is designed to import files of gene 
expression counts obtained with [HTSeq](https://github.com/simon-anders/htseq). 
You can have a look at its implementation in 
[xaio/data_importation/gdc.py](xaio/data_importation/gdc.py) to adapt it to your own
needs if you want to import other types of data.

`gdc_create_manifest()` takes in input the disease type (in our case "Adenomas and 
Adenocarcinomas"), the list of project names ("TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"), 
and the numbers of samples desired for each of these projects (remark: for "TCGA-KICH", 
there are only 66 samples available). It returns a list of Pandas dataframes, one for 
each project.

More information on GDC data can be found on the [GDC Data Portal](
https://portal.gdc.cancer.gov/
).


```python
disease_type = "Adenomas and Adenocarcinomas"
project_list = ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]
case_numbers = [200, 200, 66]
df_list = xaio.di.gdc_create_manifest(disease_type, project_list, case_numbers)
```

The Pandas library (imported as `pd`) is used to write the concatenation of the
output dataframes to the file `manifest.txt`:

```python
df = pd.concat(df_list)
df.to_csv(
    os.path.join(savedir, "manifest.txt"),
    header=True,
    index=False,
    sep="\t",
    mode="w",
)
```
<a name="s2"></a>
## Step 2: Importing the data

Once the manifest is written, individual samples are downloaded to a local
temporary folder (`tmpdir_GDCsamples/`) with the following command:

`gdc-client download -d tmpdir_GDCsamples -m /path/to/manifest.txt`

This requires the `gdc-client`, which can be downloaded at: 
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

On linux, the command `export PATH=$PATH:/path/to/gdc-client/folder` can be useful to make
sure that the `gdc-client` is found during the execution of `xaio_kidney_classif.py`.

Remark: the execution of this step, i.e. the import of all samples,
may take some time.

<a name="s3"></a>
## Step 3: Creating and saving the AnnData object

```
df = xaio.di.gdc_create_data_matrix(
    tmpdir,
    os.path.join(savedir, "manifest.txt"),
)
```
First, the `gdc_create_data_matrix()` function (implemented in
[xaio/data_importation/gdc.py](xaio/data_importation/gdc.py)
) is used to create a Pandas dataframe with all the individual samples.

The content of the dataframe `df` looks like this:
```
In [1]: df
Out[1]: 
                    3fd3d1ba-67f4-4380-800b-af1d8f39b40c  8439d23c-d1ec-4871-8bcf-1f6064dbd3bc  ...  ec467273-4e66-403e-b166-c7f6c22be922  e9c9df49-2d63-4c80-861c-1feedd598fe5
ENSG00000000003.13                                  3242                                  4853  ...                                   995                                  1739
ENSG00000000005.5                                     13                                    46  ...                                     8                                    12
ENSG00000000419.11                                  1610                                  2130  ...                                  1178
...
```
Here, every column represents a sample (with a unique identifier), 
and the rows correspond to different genes, identified by their 
Ensembl gene ID with a version number after the dot (see
[https://www.ensembl.org/info/genome/stable_ids/index.html](https://www.ensembl.org/info/genome/stable_ids/index.html)).
The integer values are the raw gene expression level measurements for all genes 
and all samples.  
Since the last 5 rows contain special information that we will not use, we drop them
with the following command:
```
df = df.drop(index=df.index[-5:])
```

In the convention used by Scanpy (and various other tools), samples are stored as raws of the
data matrix, therefore we transpose the dataframe when creating the AnnData object:

```
xd = sc.AnnData(df.transpose())
```
See this documentation for details on AnnData objects: 
[https://anndata.readthedocs.io](https://anndata.readthedocs.io).

`xd.X[0, :]`, the first row, contains the expression levels of all genes for the 
first sample.  
`xd.X[:, 0]`, the first column, contains the expression levels of
the first gene for all samples.

The feature names (gene IDs) are stored in `xd.var_names`, and the sample
identifiers are stored in `xd.obs_names`.  
In order to improve cross-sample comparisons, we normalize the sequencing
depth to 1 million, with the following Scanpy command:
```
sc.pp.normalize_total(xd, target_sum=1e6)
``` 
`normalize_total()` performs a linear normalization for each sample 
so that the sum of the feature values becomes equal to `target_sum`.  
It is a very basic normalization that we use for simplicity in this tutorial, 
but for more advanced applications, a more sophisticated preprocessing may be 
required.  
`normalize_total()` is an in-place modification of the data, so after its 
application, `xd.X` contains the modified data.

For each feature, we compute both its mean value accross all samples, and its
standard deviation accross all samples. We save the results as annotations of 
variables/features (var):
```
xd.var["mean_values"] = xaio.tl.var_mean_values(xd)
xd.var["standard_deviations"] = xaio.tl.var_standard_deviations(xd)
```

Then, we save `xd` as the file **xaio_kidney_classif.h5ad**
in the `savedir` directory:
```
xd.write(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
```

At the end of Step 3, we delete the individual sample files that were downloaded in
Step 2:
```
shutil.rmtree(tmpdir, ignore_errors=True)
```

<a name="s4"></a>
## Step 4: Labelling the samples

We load the AnnData object and the manifest (which will be used to assign labels to 
samples):
```
xd = sc.read(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
manifest = pd.read_table(os.path.join(savedir, "manifest.txt"), header=0)
```
The manifest contains the labels (`"TCGA-KIRC"`, `"TCGA-KIRP"` or `"TCGA-KICH"`) of 
every sample.  
We use it create a dictionary of labels: `label_dict`.
```
label_dict = {}
for i in range(xd.n_obs):
    label_dict[manifest["id"][i]] = manifest["annotation"][i]
```
Example: `label_dict['80c9e71b-7f2f-48cf-b3ef-f037660a4903']` is equal to `"TCGA-KICH"`.

Then we create the array of labels (considering samples in the same order as 
`xd.obs_names`), and assign it to `xd.obs["labels"]`.

```
label_array = np.array([label_dict[xd.obs_names[i]] for i in range(xd.n_obs)])
xd.obs["labels"] = label_array
```

We compute the list of distinct labels, and assign it, as an unstructured annotation,
to `xd.uns["all_labels"]`.
```
xd.uns["all_labels"] = xaio.tl.all_labels(xd.obs["labels"])
```
We also compute the list of sample indices for every label:
```
xd.uns["obs_indices_per_label"] = xaio.tl.indices_per_label(xd.obs["labels"])
```
Example: `xd.uns["obs_indices_per_label"]["TCGA-KIRC"]` is the list of indices
of the samples that are labelled as "TCGA-KIRC".

We then save the modifications:
```
xd.write(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
```

<a name="s5"></a>
## Step 5: Basic preprocessing
Loading the AnnData object: 
```
xd = sc.read(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
```
First, we logarithmize the data  with the following Scanpy function that applies
the equation X = log(1 + X):
```
sc.pp.log1p(xd)
```
We follow the Scanpy procedure to select the top 4000 highly variable genes:
```
sc.pp.highly_variable_genes(xd, n_top_genes=4000)
```
And we perform the filtering to actually remove the non-highly variable genes:
```
xd = xd[:, xd.var.highly_variable]
```

We then randomly split the samples into training and test sets:
```
xaio.tl.train_and_test_indices(xd, "obs_indices_per_label", test_train_ratio=0.25)
```
The function `train_and_test_indices()` requires `xd.uns["obs_indices_per_label"]`, which was computed in 
the previous step. With `test_train_ratio=0.25`, for every label 
("TCGA-KIRC", "TCGA-KIRP" or "TCGA-KICH"), 25% of the samples are assigned to 
the test set, and 75% to the train set. It creates the following unstructured 
annotations:
- `xd.uns["train_indices"]`: the array of the indices of all samples that belong 
to the training set.
- `xd.uns["test_indices"]`: the array of the indices of all samples that belong 
to the test set.
- `xd.uns["train_indices_per_label"]`: the dictionary of sample indices in the 
training set, per label. For instance, `xd.uns["train_indices_per_label"]["TCGA-KIRP"]` is the array
of indices of all the samples labelled as "TCGA-KIRP" that belong to the training set.
- `xd.uns["test_indices_per_label"]`: the dictionary of sample indices in the 
test set, per label.

We save the preprocessed and filtered data to a new file:
```
xd.write(os.path.join(savedir, "xaio_k_c_small.h5ad"))
```

<a name="s6"></a>
## Step 6: Training binary classifiers and performing recursive feature elimination

```
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
            os.path.join(
                savedir, "xd_small", "feature_selectors", label
            )
        )
        print("Done.")
```

<a name="s7"></a>
## Step 7: Visualizing results

+ Standard deviation vs. mean value for all features:

```python
xdata.function_scatter(
    lambda idx: xdata.feature_mean_values[idx],
    lambda idx: xdata.feature_standard_deviations[idx],
    "features",
    xlog_scale=True,
    ylog_scale=True,
)
```
![alt text](imgs/tuto1_mean_vs_std_deviation.png 
"Standard deviation vs. mean value for all features")

+ Scores on the test dataset for the "TCGA-KIRC" binary classifier 
(positive samples are above the y=0.5 line):
```python
feature_selector[0].plot()
```
![alt text](imgs/tuto1_KIRC_scores.png 
"Scores on the test dataset for the 'TCGA-KIRC' binary classifier")


+ 2D UMAP projection of the log-normalized data limited to the 30 selected features
(10 for each type of cancer):

```python
xdata.reduce_features(gene_list)
xdata.compute_normalization("log")
xdata.umap_plot("log")
```
![alt text](imgs/tuto1_UMAP.png 
"2D UMAP plot")

We observe 3 distinct clusters corresponding to the three categories
KIRC, KIRP and KICH. Remark: it may be possible that some of the 
samples have been miscategorized.

+ Log-normalized values accross all samples, for the 30 genes that have been 
selected:
```python
xdata.feature_plot(gene_list, "log")
```

![alt text](imgs/tuto1_30features.png 
"Log-normalized values accross all samples for the 30 selected features")

The recursive feature elimination procedure returned 30 features whose combined values 
allow us to distinguish the 3 categories of cancers. A strong contrast can also be 
observed for some individual features. For example, in the figure above, 
the features ENSG00000185633.9 (for KIRC), ENSG00000168269.8 (for KICH) and
ENSG00000163435.14 (for KIRP) stand out.

Let us plot the read counts accross all samples for each of these 3 features.

+ ENSG00000185633.9 (NDUFA4L2 gene):
```python
xdata.feature_plot("ENSG00000185633.9", "raw")
```
![alt text](imgs/tuto1_NDUFA4L2_KIRC.png 
"Read counts for ENSG00000185633.9")

+ ENSG00000163435.14 (ELF3 gene):
```python
xdata.feature_plot("ENSG00000163435.14", "raw")
```
![alt text](imgs/tuto1_ELF3_KIRP.png 
"Read counts for ENSG00000163435.14")

+ ENSG00000168269.8 (FOXI1 gene):
```python
xdata.feature_plot("ENSG00000168269.8", "raw")
```
![alt text](imgs/tuto1_FOXI1_KICH.png 
"Read counts for ENSG00000168269.8")

Studies on the role of these genes in kidney cancers can be found in the literature:
+ In the following publication, the gene NDUFA4L2 (ENSG00000185633.9) is analyzed as a 
biomarker for KIRC:
[D. R. Minton et al., *Role of NADH Dehydrogenase (Ubiquinone) 1 alpha subcomplex 4-like 
2 in clear cell renal cell carcinoma*, 
Clin Cancer Res. 2016 Jun 1;22(11):2791-801. doi: [10.1158/1078-0432.CCR-15-1511](
https://doi.org/10.1158/1078-0432.CCR-15-1511
)].
+ In [A. O. Osunkoya et al., *Diagnostic biomarkers for renal cell carcinoma: selection 
using novel bioinformatics systems for microarray data analysis*, 
Hum Pathol. 2009 Dec; 40(12): 1671–1678. doi: [10.1016/j.humpath.2009.05.006](
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2783948/
)], the gene ELF3 (ENSG00000163435.14) is verified as a biomarker for KIRP.
+ Finally, [D. Lindgren et al., *Cell-Type-Specific Gene Programs of the Normal Human 
Nephron Define Kidney Cancer Subtypes*, Cell Reports 2017 Aug; 20(6): 1476-1489. 
doi: [10.1016/j.celrep.2017.07.043](
https://doi.org/10.1016/j.celrep.2017.07.043
)] identifies the transcription factor FOXI1 (ENSG00000168269.8) to be drastically 
overexpressed in KICH.


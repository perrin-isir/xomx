# *XAIO Tutorial:* constructing diagnostic biomarker signatures

-----

The objective of this tutorial is to use a recursive feature elimination method on 
RNA-seq data from the Cancer Genome Atlas (TCGA) to identify gene biomarker signatures 
for the differential diagnosis of three types of kidney cancer: kidney renal clear cell
carcinoma (**KIRC**), kidney renal papillary cell carcinoma (**KIRP**), and chromophobe
renal cell carcinoma (**KICH**).

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

```python
tmpdir = "tmpdir_GDCsamples"
```
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

```python
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
```python
df = df.drop(index=df.index[-5:])
```

In the convention used by Scanpy (and various other tools), samples are stored as raws of the
data matrix, therefore we transpose the dataframe when creating the AnnData object:

```python
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
We make sure that the feature names are unique with the
following command:
```python
xd.var_names_make_unique()
```
In order to improve cross-sample comparisons, we normalize the sequencing
depth to 1 million, with the following Scanpy command:
```python
sc.pp.normalize_total(xd, target_sum=1e6)
``` 
`normalize_total()` performs a linear normalization for each sample 
so that the sum of the feature values becomes equal to `target_sum`.  
It is a very basic normalization that we use for simplicity in this tutorial, 
but for more advanced applications, a more sophisticated preprocessing may be 
required.  
`normalize_total()` is an in-place modification of the data, so after its 
application, `xd.X` contains the modified data.

We save `xd` as the file **xaio_kidney_classif.h5ad**
in the `savedir` directory:
```python
xd.write(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
```

At the end of Step 3, we delete the individual sample files that were downloaded in
Step 2:
```python
shutil.rmtree(tmpdir, ignore_errors=True)
```

<a name="s4"></a>
## Step 4: Labelling the samples

We load the AnnData object and the manifest:
```python
xd = sc.read(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
manifest = pd.read_table(os.path.join(savedir, "manifest.txt"), header=0)
```
The manifest contains the labels (`"TCGA-KIRC"`, `"TCGA-KIRP"` or `"TCGA-KICH"`) of 
every sample.  
We use it create a dictionary of labels: `label_dict`.
```python
label_dict = {}
for i in range(xd.n_obs):
    label_dict[manifest["id"][i]] = manifest["annotation"][i]
```
Example: `label_dict['80c9e71b-7f2f-48cf-b3ef-f037660a4903']` is equal to `"TCGA-KICH"`.

Then we create the array of labels, considering samples in the same order as 
`xd.obs_names`, and assign it to `xd.obs["labels"]`.

```python
label_array = np.array([label_dict[xd.obs_names[i]] for i in range(xd.n_obs)])
xd.obs["labels"] = label_array
```

We compute the list of distinct labels, and assign it, as an unstructured annotation,
to `xd.uns["all_labels"]`.
```python
xd.uns["all_labels"] = xaio.tl.all_labels(xd.obs["labels"])
```
We also compute the list of sample indices for every label:
```python
xd.uns["obs_indices_per_label"] = xaio.tl.indices_per_label(xd.obs["labels"])
```
Example: `xd.uns["obs_indices_per_label"]["TCGA-KIRC"]` is the list of indices
of the samples that are labelled as `"TCGA-KIRC"`.

It is important to use the keys `"labels"`,
`"all_labels"` and `"obs_indices_per_label"` as they
are expected by some XAIO functions.

We then save the modifications:
```python
xd.write(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
```

<a name="s5"></a>
## Step 5: Basic preprocessing
Loading the AnnData object: 
```python
xd = sc.read(os.path.join(savedir, "xaio_kidney_classif.h5ad"))
```
First, we compute the mean and standard deviation (across samples) for all the features:
```python
xd.var["mean_values"] = xaio.tl.var_mean_values(xd)
xd.var["standard_deviations"] = xaio.tl.var_standard_deviations(xd)
```
Remark: `xd.var["mean_values"]` and 
`xd.var["standard_deviations"]` will be used only in Step 7.

We logarithmize the data with the following Scanpy function that applies
the transformation X = log(1 + X):
```python
sc.pp.log1p(xd)
```
We then follow the Scanpy procedure to select the top 8000 highly variable genes:
```python
sc.pp.highly_variable_genes(xd, n_top_genes=8000)
```
We perform the filtering to actually remove the other features:
```python
xd = xd[:, xd.var.highly_variable]
```
The reason why we reduce the number of features
is to speed up the process of feature elimination
in Step 6, which can be relatively slow if it begins 
with tens of thousands of features. Keeping 
highly variable features is one possibility,
but there are other options for the
initial selection of features, see for instance 
the [xaio_pbmc.md](xaio_pbmc.md) tutorial (Step 2).

We compute the dictionary of feature indices,
which is required by some XAIO functions:
```python
xd.uns["var_indices"] = xaio.tl.var_indices(xd)
```
Example:  `xd.uns["var_indices"]['ENSG00000281918.1']`
is equal to 7999 because ENSG00000281918.1 is now
the last of the 8000 features in `xd.var_names`.

We then randomly split the samples into training and test sets:
```python
xaio.tl.train_and_test_indices(xd, "obs_indices_per_label", test_train_ratio=0.25)
```
The function `train_and_test_indices()` requires `xd.uns["obs_indices_per_label"]`, which was computed in 
the previous step. With `test_train_ratio=0.25`, for every label 
(`"TCGA-KIRC"`, `"TCGA-KIRP"` or `"TCGA-KICH"`), 25% of the samples are assigned to 
the test set, and 75% to the train set. It creates the following unstructured 
annotations:
- `xd.uns["train_indices"]`: the array of indices of all samples that belong 
to the training set.
- `xd.uns["test_indices"]`: the array of indices of all samples that belong 
to the test set.
- `xd.uns["train_indices_per_label"]`: the dictionary of sample indices in the 
training set, per label. For instance, `xd.uns["train_indices_per_label"]["TCGA-KIRP"]` is the array
of indices of all the samples labelled as `"TCGA-KIRP"` that belong to the training set.
- `xd.uns["test_indices_per_label"]`: the dictionary of sample indices in the 
test set, per label.

We save the logarithmized and filtered data to a new file:
```python
xd.write(os.path.join(savedir, "xaio_k_c_small.h5ad"))
```

<a name="s6"></a>
## Step 6: Training binary classifiers and performing recursive feature elimination

Loading: 
```python
xd = sc.read(os.path.join(savedir, "xaio_k_c_small.h5ad"))
```

We initialize an empty dictionary 
of "feature selectors":
```python
feature_selectors = {}
```
There will be one feature selector per label.
What we call feature selector here is a binary classifier
trained with the Extra-Trees algorithm to
distinguish samples with a given label from
other types of samples. After training, features are
ranked by a measure of importance known as the Gini importance, 
and the 100 most important features are kept. 
Then, the Extra-Trees algorithm is run again on the training
data filtered to the 100 selected features, which leads to a 
new measure of importance of the features. We repeat the 
procedure to progressively select 30, then 20, 15 and finally 10
features. At each iteration, we evaluate on the test set the 
Matthews correlation coefficient (MCC score) of the 
classifier to observe how the performance changes 
when the number of features decreases.  
The progression 100-30-20-15-10 is arbitrary, but 
the most efficient strategies start by aggressively 
reducing the number of features, and then slow down
when the number of features becomes small.

Here is the loop that trains all the classifiers and ends up 
selecting 10 features for every label. 
Each classifier is saved in the folder `feature_selectors/` in the
`savedir` directory.

```python
for label in xd.uns["all_labels"]:
    print("Annotation: " + label)
    feature_selectors[label] = xaio.fs.RFEExtraTrees(
        xd,
        label,
        n_estimators=450,
        random_state=0,
    )
    feature_selectors[label].init()
    for siz in [100, 30, 20, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selectors[label].select_features(siz)
        print("MCC score:", xaio.tl.matthews_coef(
            feature_selectors[label].confusion_matrix
        ))
    feature_selectors[label].save(os.path.join(savedir, "feature_selectors", label))
    print("Done.")
```
<a name="s7"></a>
## Step 7: Visualizing results

Loading:
```python
xd = sc.read(os.path.join(savedir, "xaio_k_c_small.h5ad"))
```

Using the plotting function `function_scatter()`,
we plot the standard deviation vs mean value for all the 
genes (which were computed before logarithmizing the data).
`function_scatter()` takes in input two functions, one for 
the x-axis, and one for the y-axis. Each of these functions
must take in input the feature index. By changing the 
`obs_or_var` option to "obs" instead of "var", we can use
`function_scatter()` to make a scatter plot over the samples
instead of over the features.

```python
xaio.pl.function_scatter(
    xd,
    lambda idx: xd.var["mean_values"][idx],
    lambda idx: xd.var["standard_deviations"][idx],
    obs_or_var="var",
    xlog_scale=True,
    ylog_scale=True,
)
```
![alt text](imgs/tuto1_mean_vs_std.gif 
"Standard deviation vs. mean value for all features")

This plot shows the 8000 highly variable genes selected
in Step 5, and we can observe the frontier that was defined by 
`sc.pp.highly_variable_genes()` to remove genes considered 
less variable.
Hovering over points with the cursor shows the identifiers 
of the corresponding genes.

We then load the feature selectors trained in Step 6,
and create `gene_dict`, a dictionary of the 10-gene signatures for each label.
```python
feature_selectors = {}
gene_dict = {}
for label in xd.uns["all_labels"]:
    feature_selectors[label] = xaio.fs.load_RFEExtraTrees(
        os.path.join(savedir, "feature_selectors", label),
        xd,
    )
    gene_dict[label] = [
        xd.var_names[idx_]
        for idx_ in feature_selectors[label].current_feature_indices
    ]
```
Example: `gene_dict['TCGA-KICH']` is equal to `['ENSG00000162399.6',
 'ENSG00000168269.8',
...
 'ENSG00000173253.13',
 'ENSG00000156284.5']`, the list of 10 genes that have been selected 
by the feature selection process on `"TCGA-KICH"`.

For a given feature selector, for example `feature_selectors["TCGA-KIRP"]`,
`plot()` displays results on the test set. The classifier uses only the selected 
features, here the 10 features selected for the label `"TCGA-KIRP"`.
Points above the horizontal red line (score > 0.5) are classified as positives (the 
prediction is: `"TCGA-KIRP"`), and points below the horizontal line (score < 0.5)
are classified as negatives (the prediction is: `not "TCGA-KIRP"`).
```python
feature_selectors["TCGA-KIRP"].plot()
```
![alt text](imgs/tuto1_KIRP.gif 
"10-gene classifier for the TCGA-KIRP label")

Hovering over a point with the cursor shows the identifier of the corresponding 
sample and its true label.

We can construct a multiclass classifier based on the 3 binary classifiers:
```python
sbm = xaio.cl.ScoreBasedMulticlass(xd, xd.uns["all_labels"], feature_selectors)
```
This multiclass classifier bases its predictions on 30 features (at most): the 
union of the three 10-gene signatures (one per label). It simply computes the 3 
scores of each of the binary classifiers, and returns the label that corresponds 
to the highest score.  
`plot()` displays results on the test set:
```python
sbm.plot()
```
![alt text](imgs/tuto1_multiclass.gif 
"Multiclass classifier")

Hovering over a point with the cursor shows the identifier of the corresponding 
sample, its true label, and the predicted label.  
For each of the 3 labels, points that are 
higher in the horizontal band correspond to a 
higher confidence in the prediction (but
the very top of the band does not mean 100% 
confidence).

We gather the selected genes in a single list:
```python
all_selected_genes = np.asarray(list(gene_dict.values())).flatten()
```
We can visualize these marker genes with `xaio.pl.var_plot()`:
```python
xaio.pl.var_plot(xd, all_selected_genes)
```
![alt text](imgs/tuto1_markers.png
"Marker gene expressions")

Interestingly, we observe here that, for the label `"TCGA-KIRP"`,
the selected marker genes are mostly downregulated
(which does not mean that upregulated marker genes cannot 
lead to similarly good results).  
Let us zoom on the marker genes for KIRP:
```python
xaio.pl.var_plot(xd, gene_dict["TCGA-KIRP"])
```
![alt text](imgs/tuto1_KIRPmarkers.png
"Downregulated marker genes for TCGA-KIRP")

Or, using the Scanpy function `stacked_violin()`:
```python
sc.pl.stacked_violin(xd, gene_dict["TCGA-KIRP"], groupby="labels")
```
![alt text](imgs/tuto1_stacked_violin.png
"Mainly downregulated marker genes for TCGA-KIRP")

We observe 3 significantly downregulated genes: 
EBF2 (ENSG00000221818), PTGER3 (ENSG00000050628)
and C6orf223 (ENSG00000181577).

KICH markers:
```python
xaio.pl.var_plot(xd, gene_dict["TCGA-KICH"])
```
![alt text](imgs/tuto1_KICHmarkers.png
"Upregulated marker genes for TCGA-KIRC")
We can also use `var_plot()` with a single gene:
```python
xaio.pl.var_plot(xd, "ENSG00000168269.8")
```
![alt text](imgs/tuto1_FOXI1_KICH.png
"Upregulated marker genes for TCGA-KICH")
The FOXI1 (ENSG00000168269) transcription factor is known to 
be drastically overexpressed in KICH. In fact, it has been argued that 
the FOXI1-driven transcriptome that defines renal intercalated cells is retained 
in KICH and implicates the intercalated cell type as the cell of origin 
for KICH; see: 
[D. Lindgren et al., *Cell-Type-Specific Gene Programs of the Normal Human 
Nephron Define Kidney Cancer Subtypes*, Cell Reports 2017 Aug; 20(6): 1476-1489. 
doi: [10.1016/j.celrep.2017.07.043](
https://doi.org/10.1016/j.celrep.2017.07.043
)]

KIRC markers:
```python
xaio.pl.var_plot(xd, gene_dict["TCGA-KIRC"])
```
![alt text](imgs/tuto1_KIRCmarkers.png
"Upregulated marker genes for TCGA-KIRC")

We can notice in particular the upregulation of NDUFA4L2 (ENSG00000185633.9),
a gene that has been analyzed as a biomarker for KIRC in
[D. R. Minton et al., *Role of NADH Dehydrogenase (Ubiquinone) 1 alpha subcomplex 4-like 
2 in clear cell renal cell carcinoma*, 
Clin Cancer Res. 2016 Jun 1;22(11):2791-801. doi: [10.1158/1078-0432.CCR-15-1511](
https://doi.org/10.1158/1078-0432.CCR-15-1511
)].

Finally, we filter and restrict the data to the selected genes, and follow 
the Scanpy procedure to compute a 2D UMAP embedding:
```python
xd = xd[:, all_selected_genes]
xd.var_names_make_unique()
sc.pp.neighbors(xd, n_neighbors=10, n_pcs=40)
sc.tl.umap(xd)
```
`sc.tl.umap()` stores the embedding in `xd.obsm["X_umap"]`.  
We use `xaio.pl.plot2d()` to display an interactive plot:
```python
xaio.pl.plot2d(xd, "X_umap")
```
![alt text](imgs/tuto1_UMAP.gif
"Interactive UMAP plot")

Hovering the cursor over points shows sample identifiers and labels,
which can be useful to find unusual or possibly mislabelled samples.


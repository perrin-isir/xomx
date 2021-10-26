# XAIO - Biomarker Discovery Tutorial

-----

The objective of this tutorial is to use a recursive feature elimination method on 
RNA-Seq data to identify gene biomarkers for the differential diagnosis of three 
types of kidney cancer: Kidney Renal Clear Cell Carcinoma (**KIRC**), Kidney Renal 
Papillary Cell Carcinoma (**KIRP**), and Kidney Renal Clear Cell Carcinoma (**KICH**).

### Running the tutorial:
- **Repeated executions of the [kidney_classif.py](xaio/tutorials/kidney_classif.py) 
file perform each of the 7 steps of 
the tutorial, one by one.**

- A specific step can also be chosen using an integer
argument. For instance, `python kidney_classif.py 1` executes the step 1.

### Table of Contents:
+ [Step 1: Preparing the manifest](#s1)
+ [Step 2: Importing the data](#s2)
+ [Step 3: Creating and saving the XAIOData object](#s3)
+ [Step 4: Annotating the samples](#s4)
+ [Step 5: Basic pre-processing](#5)
+ [Step 6: Training binary classifiers and performing recursive feature elimination](#s6)
+ [Step 7: Visualizing results](#s7)

### Saving results:

In [kidney_classif.py](xaio/tutorials/kidney_classif.py), after the imports, the 
following lines define the string variable `savedir`,  which determines the folder 
in which data and outputs will be stored:
```python
args = get_args()
savedir = args.savedir
```
By default, `savedir` is `~/results/xaio/kidney_classif`, but it can be modified using a 
`--savedir` argument in input (e.g. `python kidney_classif.py --savedir /tmp`).

<a name="s1"></a>
## Step 1: Preparing the manifest

We use the 
[GDC Data Transfer Tool](
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
)
to import data from the Cancer Genome Atlas (TCGA). 
This involves creating a `manifest.txt` file that describes the files to be imported.

The `gdc_create_manifest()` function (`from xaio import gdc_create_manifest`) 
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
df_list = gdc_create_manifest(disease_type, project_list, case_numbers)
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

On linux, the command `export PATH=$PATH:/path/to/gdc-client` can be useful to make
sure that the `gdc-client` is found during the execution of `kidney_classif.py`.

Remark: the execution of this step, i.e. the import of all samples,
may take some time.

<a name="s3"></a>
## Step 3: Creating and saving the XAIOData object

```
df = gdc_create_data_matrix(
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
Every column represents a sample (with a unique identifier), 
and the rows correspond to different genes, identified by their 
Ensembl gene ID with a version number after the dot (see
[http://www.ensembl.org/info/genome/stable_ids/index.html](http://www.ensembl.org/info/genome/stable_ids/index.html)).
The integer values are the raw gene expression level measurements for all genes 
and samples.

Since the last 5 rows contain special information that we will not use, we drop them
with the following command:
```
df = df.drop(index=df.index[-5:])
```
Then we create a XAIOData object, and import the dataframe in it:
```
xd = XAIOData()
xd.import_pandas(df)
```
The XAIOData class is the most important data structure of the XAIO library.
Its implementation can be found in 
[xaio/tools/basic_tools.py](xaio/tools/basic_tools.py).
The objects of this class contain 2D data arrays and give access to various 
functionalities to process them.

After `xd.import_pandas(df)`, `xd.data_array["raw"]` is a NumPy array of raw data equal
to the transpose of the dataframe df. So, `xd.data_array["raw"][0, :]`, the first row,
contains the expression levels of all genes for the first sample. 
And `xd.data_array["raw"][:, 0]`, the first column, contains the expression levels of
the first gene for all samples.

The feature names (gene IDs) are stored in `xd.feature_names`, and the sample
identifiers are stored in `xd.sample_ids`. For most applications, XAIOData 
objects require both feature names and sample IDs.

In order to improve cross-sample comparisons, we normalize the sequencing
depth to 1 million, with the following command:
```
xd.normalize_feature_sums(1e6)
```
`normalize_feature_sums(X)` performs a linear normalization for each sample 
so that the sum of the feature values is equal to `X`.  It is a very basic 
normalization that we use for simplicity in this tutorial, but for more advanced
applications, a more sophisticated pre-processing may be required.
Unlike most other normalizations, `normalize_feature_sums()` is an 
in-place modification of the raw data, so after its application, 
the modified data is still `xd.data_array["raw"]`.

For each feature, we compute both its mean value accross all samples, and its
standard deviation accross all samples:
```
xd.compute_feature_mean_values()
xd.compute_feature_standard_deviations()
```
The mean values are stored in the list `xd.feature_mean_values`, 
and the standard deviations are stored in the list `xd.feature_standard_deviations`.

Then, we save `xd` to the disk (in the `savedir` directory, in an `xd` subfolder):
```
xd.save(["raw"], os.path.join(savedir, "xd"))
```
The list `["raw"]` in input specifies that `xd.data_array["raw"]` is saved to the disk.

`xd.save([], os.path.join(savedir, "xd"))`, would save other elements of the `xd` 
object, such as `xd.feature_mean_values` or `xd.feature_standard_deviations`, but 
not the data array `xd.data_array["raw"]`. This is particularly useful when the data 
array has not changed and does not need to be overwritten.

At the end of Step 3, we delete the individual sample files that were downloaded in
Step 2:
```
shutil.rmtree(tmpdir, ignore_errors=True)
```

<a name="s4"></a>
## Step 4: Annotating the samples

<a name="s5"></a>
## Step 5: Basic pre-processing

<a name="s6"></a>
## Step 6: Training binary classifiers and performing recursive feature elimination

```python
nr_annotations = len(xdata.all_annotations)
feature_selector = np.empty(nr_annotations, dtype=object)
for i in range(nr_annotations):
    print("Annotation: " + xdata.all_annotations[i])
    feature_selector[i] = RFEExtraTrees(
        xdata,
        xdata.all_annotations[i],
        init_selection_size=4000,
        n_estimators=450,
        random_state=0,
    )
    feature_selector[i].init()
    for siz in [100, 30, 20, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selector[i].select_features(siz)
    feature_selector[i].save(
        os.path.join(
            savedir, "xdata_small", "feature_selectors", xdata.all_annotations[i]
        )
    )
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


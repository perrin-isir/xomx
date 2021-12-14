# *xomx tutorial:* preprocessing and clustering 3k PBMCs

-----

This tutorial follows the single cell RNA-seq [Scanpy tutorial on 3k PBMCs](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html).

The objective is to analyze a dataset of Peripheral Blood Mononuclear Cells (PBMC)
freely available from 10X Genomics, composed of 2,700 single cells that were
sequenced on the Illumina NextSeq 500.  
We replace some Scanpy plots by interactive *xomx* plots, and modify the
computation of marker genes. Instead of using a t-test, Wilcoxon-Mann-Whitney test 
or logistic regression, we perform recursive feature elimination with 
the [Extra-Trees algorithm](
https://link.springer.com/article/10.1007/s10994-006-6226-1).

### Running the tutorial:
- **Repeated executions of the [xomx_pbmc.py](xomx_pbmc.py) 
file perform each of the 3 steps of 
the tutorial, one by one.**
- A specific step can also be chosen using an integer
argument. For instance, `python xomx_kidney_classif.py 1` executes the step 1.

### Table of Contents:
+ [Step 1: Data importation, preprocessing and clustering](#s1)
+ [Step 2: Training binary classifiers and performing recursive feature elimination](#s2)
+ [Step 3: Visualizing results](#s3)

### Saving results:
In [xomx_pbmc.py](xomx_pbmc.py), after the imports, the 
following lines define the string variable `savedir`: the folder 
in which data and outputs will be stored.
```python
args = xomx.tt.get_args("pbmc")
savedir = args.savedir
```
By default, `savedir` is `~/results/xomx/pbmc`, but it can be modified using a 
`--savedir` argument in input (e.g. `python xomx_pbmc.py --savedir /tmp`).

<a name="s1"></a>
## Step 1: Data importation, preprocessing and clustering

We start by downloading scRNA-seq data freely available from 10x Genomics, 
and read it into an  
[AnnData](https://anndata.readthedocs.io) object with the Scanpy function 
`read_10x_mtx()`:
```python
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
```

We apply basic filtering, annotate the group of mitochondrial genes and compute various
metrics, as it is done in the [Scanpy tutorial](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html):
```python
sc.pp.filter_cells(xd, min_genes=200)
sc.pp.filter_genes(xd, min_cells=3)
xd.var["mt"] = xd.var_names.str.startswith(
    "MT-"
)  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(
    xd, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)
```

We compute the following NumPy array:
```python
mean_count_fractions = np.squeeze(
    np.asarray(
        np.mean(
            xd.X / np.array(xd.obs["total_counts"]).reshape((xd.n_obs, 1)), axis=0
        )
    )
)
```
The k-th element of `mean_count_fractions` is the mean fraction of counts of the k-th 
gene in each single cell, across all cells.

3 interactive plots with *xomx* functions:

+ Plot, for all genes, the mean fraction of counts in single cells, across all cells.

We use `xomx.pl.function_plot()`. Besides the AnnData object, it takes in input a 
function (here `lambda idx: mean_count_fractions[idx]`) which itself takes as input 
the index of a feature (if `obs_or_var="var"`) or a sample (if `obs_or_var="obs"`). 
```python
xomx.pl.function_plot(
    xd,
    lambda idx: mean_count_fractions[idx],
    obs_or_var="var",
    violinplot=False,
    ylog_scale=False,
    xlabel="genes",
    ylabel="mean fractions of counts across all cells",
)
```
![alt text](imgs/tuto2_mean_fraction_counts.gif 
"Mean fraction of counts")

Hovering over points with the cursor shows the names 
of the corresponding genes.

+ Plot the total counts per cell.
```python
xomx.pl.function_plot(
    xd,
    lambda idx: xd.obs["total_counts"][idx],
    obs_or_var="obs",
    violinplot=True,
    ylog_scale=False,
    xlabel="cells",
    ylabel="total number of counts",
)
```
![alt text](imgs/tuto2_total_counts.gif 
"Total counts per cell")

Hovering over points with the cursor shows the identifiers 
of the corresponding cells.

+ Plot mitochondrial count percentages vs total number of counts.

We use `xomx.pl.function_scatter()` which takes in input two functions, one for 
the x-axis, and one for the y-axis.
```python
xomx.pl.function_scatter(
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
```
![alt text](imgs/tuto2_mitochondrial.gif 
"Mitochondrial count percentages vs total number of counts")

We follow the steps of the [Scanpy tutorial](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html) for the preprocessing
and clustering of the data:
```python
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
```

We rename the clusters as it is done in the [Scanpy tutorial](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html):
```python
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
```
The data was filtered to compute neighborhood graph and clusters, now we retrieve
the unfiltered data, with all the features, as follows:
```python
obsp = xd.obsp.copy()
xd = xd.raw.to_adata()
xd.obsp = obsp
```
The copy of `xd.obsp` is necessary as it is not restored by `xd.raw.to_adata()`.

We compute the dictionary of feature indices, which is required by some *xomx* functions:
```python
xd.uns["var_indices"] = xomx.tl.var_indices(xd)
```
Example:  `xd.uns["var_indices"]["MALAT1"]` is 7854 and `xd.var_names[7854]` is 
`"MALAT1"`.

The "leiden" clusters define labels, but *xomx* uses labels stored in `.obs["labels"]`, so
we make the following copy:
```python
xd.obs["labels"] = xd.obs["leiden"]
```

Several *xomx* functions require the list of all labels and the 
dictionary of sample indices per label:
```python
xd.uns["all_labels"] = xomx.tl.all_labels(xd.obs["labels"])
xd.uns["obs_indices_per_label"] = xomx.tl.indices_per_label(xd.obs["labels"])
```
Example: `xd.uns["obs_indices_per_label"]["Megakaryocytes"]` is the list of indices
of the samples that are labelled as `"Megakaryocytes"`.

We then randomly split the samples into training and test sets:
```python
xomx.tl.train_and_test_indices(xd, "obs_indices_per_label", test_train_ratio=0.25)
```
With `test_train_ratio=0.25`, for every label, 25% of the samples are assigned to 
the test set, and 75% to the train set. It creates the following unstructured 
annotations:
- `xd.uns["train_indices"]`: the array of indices of all samples that belong 
to the training set.
- `xd.uns["test_indices"]`: the array of indices of all samples that belong 
to the test set.
- `xd.uns["train_indices_per_label"]`: the dictionary of sample indices in the 
training set, per label. For instance,
`xd.uns["train_indices_per_label"]["Megakaryocytes]` is the array
of indices of all the samples labelled as `"Megakaryocytes"` that belong to the
training set.
- `xd.uns["test_indices_per_label"]`: the dictionary of sample indices in the 
test set, per label.

We use the Scanpy function `rank_genes_groups()` to rank the genes for each 
cluster with a t-test:
```python
sc.tl.rank_genes_groups(xd, "leiden", method="t-test")
```
After that, the ranking information is contained in 
`xd.uns["rank_genes_groups"]`. For instance, 
`xd.uns["rank_genes_groups"]["names"]["Megakaryocytes"]` is the list of genes 
ordered from highest to lowest rank for the label `"Megakaryocytes"`.

We save `xd` as the file **xomx_pbmc.h5ad**
in the `savedir` directory:
```python
xd.write(os.path.join(savedir, "xomx_pbmc.h5ad"))
```

<a name="s2"></a>
## Step 2:  Training binary classifiers and performing recursive feature elimination
Loading the AnnData object:
```python
xd = sc.read(os.path.join(savedir, "xomx_pbmc.h5ad"), cache=True)
```

Just like in the [Step 6 of the xomx_kidney_classif.md tutorial](
xomx_kidney_classif.md#s6),
we use the Extra-Trees algorithms and run it several times per label to select
100, then 30, 20, 15 and finally 10 marker genes for each label.  
The only difference with the [Step 6 in xomx_kidney_classif.md](
xomx_kidney_classif.md#s6)
is here the use of the option `init_selection_size=8000`. 
This option speeds up the process of feature elimination by starting with an
initial selection of features of size 8000, different for each label (while 
in [xomx_kidney_classif.md](xomx_kidney_classif.md), a global filtering was applied
to start with a common initial selection of 8000 highly variable genes). 
With the `init_selection_size` option, initial selections rely 
on `xd.uns["rank_genes_groups"]`, which must have been computed before.
For each label, the initial selection coincides with the
highest ranked features in `xd.uns["rank_genes_groups"]`.
After the training, for each label, `feature_selectors[label]` is a
binary classifier using only 10 features to discriminate samples with the label 
from other samples.


```python
feature_selectors = {}
for label in xd.uns["all_labels"]:
    print("Label: " + label)
    feature_selectors[label] = xomx.fs.RFEExtraTrees(
        xd,
        label,
        init_selection_size=8000,
        n_estimators=450,
        random_state=0,
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
```

<a name="s3"></a>
## Step 3: Visualizing results
Loading the AnnData object:
```python
xd = sc.read(os.path.join(savedir, "xomx_pbmc.h5ad"), cache=True)
```

Loading the binary classifiers, and creating `gene_dict`, a dictionary of the 10-gene
signatures for each label:
```python
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
```

We construct a multiclass classifier based on the binary classifiers:
```python
sbm = xomx.cl.ScoreBasedMulticlass(xd, xd.uns["all_labels"], feature_selectors)
```
This multiclass classifier bases its predictions on the union of the 10-gene 
signatures for each label. It simply computes the scores of each of the binary
classifiers, and returns the label that corresponds to the highest score.  
`plot()` displays results on the test set:
```python
sbm.plot()
```
![alt text](imgs/tuto2_multiclass.gif 
"Multiclass classifier")

With the Scanpy function `dotplot()`, we visualize the 10-gene signatures 
of CD14 Monocytes and FCGR3A Monocytes:
```python
sc.pl.dotplot(xd, gene_dict["CD14 Monocytes"] + gene_dict["FCGR3A Monocytes"], groupby="labels")
```
![alt text](imgs/tuto2_Monocytes.png
"10-gene signatures for CD14 Monocytes and FCGR3A Monocytes")

We gather all the selected genes in a single list:
```python
all_selected_genes = np.asarray(list(gene_dict.values())).flatten()
```

For comparison, we define a list of known biomarkers as suggested in the 
[Scanpy tutorial](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html):
```python
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
```

By computing the intersection with the selected genes, we observe that the 
known biomarkers are all present in the union of the 10-gene signatures 
obtained with the Extra-Trees + Recursive Feature Elimination approach:
```
In [1]: print(biomarkers.intersection(all_selected_genes))
{'FCER1A', 'MS4A7', 'FCGR3A', 'GNLY', 'CD14', 'PPBP', 'LYZ', 'CST3', 'MS4A1', 'NKG7', 'CD8A', 'IL7R'}
```

We use Scanpy to create a UMAP embedding, stored in `.obsm["X_umap"]`: 
```python
sc.tl.umap(xd)
```

Using `xomx.pl.plot2d()`, we get an interactive plot of this embedding:
```python
xomx.pl.plot2d(xd, "X_umap")
```
![alt text](imgs/tuto2_UMAP.gif 
"Interactive UMAP plot")

By default, different colors correspond to the different labels, but 
we can also specify a feature:
```python
xomx.pl.plot2d(xd, "X_umap", "CST3")
```
![alt text](imgs/tuto2_UMAP_CST3.gif 
"Interactive UMAP plot")

We can also use `xomx.pl.plot2d()` to get an interactive plot of the 
first 2 PCA components of the data (computed in Step 1):
```python
xomx.pl.plot2d(xd, "X_pca")
```
![alt text](imgs/tuto2_PCA.png 
"First 2 PCA components")

```python
xomx.pl.plot2d(xd, "X_pca", "CST3")
```
![alt text](imgs/tuto2_PCA_CST3.gif 
"First 2 PCA components")

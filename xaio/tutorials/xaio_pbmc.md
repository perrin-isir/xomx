# *XAIO Tutorial:* preprocessing and clustering 3k PBMCs

-----

This tutorial follows the single cell RNA-seq [Scanpy tutorial on 3k PBMCs](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)  
The objective is to analyze a dataset of Peripheral Blood Mononuclear Cells (PBMC)
freely available from 10X Genomics, composed of 2,700 single cells that were
sequenced on the Illumina NextSeq 500.  
We replace some Scanpy plots by interactive XAIO plots, and modify the
computation of marker genes. Instead of using a t-test, Wilcoxon-Mann-Whitney test 
or logistic regression, we perform recursive feature elimination with 
the [Extra-Trees algorithm](
https://link.springer.com/article/10.1007/s10994-006-6226-1).


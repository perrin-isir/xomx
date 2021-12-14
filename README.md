# XOMX

![version](https://img.shields.io/badge/version-0.1.0-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


XOMX is a python library providing data processing and 
machine learning tools for computational omics, with a 
particular emphasis on explainability.

It relies on [AnnData](https://anndata.readthedocs.io) objects, which makes it
fully compatible with [Scanpy](https://scanpy.readthedocs.io).

*XOMX is currently in beta version.*

-----



## Install

Recommended installation steps (with conda): 
```
git clone git@github.com:perrin-isir/XOMX.git
cd xomx
conda env create -f environment.yaml
conda activate xomxv
```
Then, use the following command to install the xaio library within the xaiov virtual
environment: 
```
pip install -e .
```
-----
## Tutorials

Tutorials (in [xomx/tutorials/](xomx/tutorials/)) are the best way to learn to use
the XOMX library.

*List of tutorials:*
* [xomx_kidney_classif.md](xomx/tutorials/xomx_kidney_classif.md) (*goal:*  use a 
recursive feature elimination method on RNA-seq data to identify gene 
biomarkers for the differential diagnosis of three types of kidney cancer)
* [xomx_pbmc.md](xomx/tutorials/xomx_pbmc.md) (*goal:* 
follow the single cell RNA-seq [Scanpy tutorial on 3k PBMCs](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html), except
for the computation of biomarkers for which recursive feature elimination is used
)

-----
## Citing the project
To cite this repository in publications:

```bibtex
@misc{xomx,
  author = {Perrin-Gilbert, Nicolas and Vibert, Julien and Vandenbogaert, Mathias and Waterfall, Joshua J.},
  title = {XOMX},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/perrin-isir/xomx}},
}
```

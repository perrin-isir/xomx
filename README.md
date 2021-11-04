# XAIO

![version](https://img.shields.io/badge/version-0.1.0-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


XAIO is a python library providing data processing and 
machine learning tools for computational omics, with a 
particular emphasis on explainability.

It relies on [AnnData](https://anndata.readthedocs.io) objects, which makes it
fully compatible with [Scanpy](https://scanpy.readthedocs.io).

*XAIO is currently in beta version.*

-----



## Install

Recommended installation steps (with conda): 
```
git clone git@github.com:perrin-isir/xaio.git
cd xaio
conda env create -f environment.yaml
conda activate xaiov
```
Then, use the following command to install the xaio library within the xaiov virtual
environment: 
```
pip install -e .
```
-----
## Tutorials

Tutorials (in [xaio/tutorials/](xaio/tutorials/)) are the best way to learn to use
the XAIO library.

*List of tutorials:*
* [xaio_kidney_classif.md](xaio/tutorials/xaio_kidney_classif.md) (*goal:*  use a 
recursive feature elimination method on RNA-seq data to identify gene 
biomarkers for the differential diagnosis of three types of kidney cancer)
* [xaio_pbmc.md](xaio/tutorials/xaio_pbmc.md) (*goal:* 
follow the single cell RNA-seq [Scanpy tutorial on 3k PBMCs](
https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html), except
for the computation of biomarkers for which recursive feature elimination is used
)

-----
## Citing the project
To cite this repository in publications:

```bibtex
@misc{xaio,
  author = {Perrin-Gilbert, Nicolas and Vibert, Julien and Vandenbogaert, Mathias and Waterfall, Joshua J.},
  title = {XAIO},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/perrin-isir/xaio}},
}
```
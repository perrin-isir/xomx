# XAIO

![version](https://img.shields.io/badge/version-0.1.0-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


XAIO is a python library providing data processing and 
machine learning tools for computational omics, with a 
particular emphasis on explainability.

*It is currently in beta version.*

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

Here is the list of tutorials:
* [kidney_classif.md](xaio/tutorials/kidney_classif.md) (*goal:*  use a recursive feature 
elimination method on RNA-Seq data to identify gene biomarkers for the differential 
diagnosis of three types of kidney cancer)

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
# ![alt text](https://raw.githubusercontent.com/perrin-isir/xomx/master/logo.png "xomx logo")

![version](https://img.shields.io/badge/version-0.1.9-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/github/actions/workflow/status/perrin-isir/xomx/docs.yml?branch=master&label=docs)](https://perrin-isir.github.io/xomx/)
[![PyPI version](https://img.shields.io/pypi/v/xomx)](https://pypi.org/project/xomx/)

*xomx* is an open-source python library providing data processing and 
machine learning tools for computational omics, with a 
particular emphasis on explainability.

It relies on [AnnData](https://anndata.readthedocs.io) objects, which makes it
fully compatible with [Scanpy](https://scanpy.readthedocs.io).

*xomx is currently in beta version.*

-----



## Install

<details><summary>Option 1: conda (preferred option)</summary>
<p>

This option is preferred because it relies mainly on conda-forge  (which among other things simplifies the installation of JAX).


    git clone https://github.com/perrin-isir/xomx.git
    cd xomx
    conda update conda
    
Install micromamba if you don't already have it (you can also simply use conda, by replacing below `micromamba create`, `micromamba update` and `micromamba activate` respectively by `conda env create`, `conda env update` and `conda activate`, but this will lead to a significantly slower installation):

    conda install -c conda-forge micromamba

Choose a conda environment name, for instance `xomxenv`.  
The following command creates the `xomxenv` environment with the requirements listed in [environment.yaml](environment.yaml):

    micromamba create --name xomxenv --file environment.yaml

If you prefer to update an existing environment (`existing_env`):

    micromamba update --name existing_env --file environment.yml

Then, activate the `xomxenv` environment:

    micromamba activate xomxenv

Finally, to install the *xomx* library in the activated environment:

    pip install -e .

</p>
</details>

<details><summary>Option 2: pip</summary>
<p>

For the pip install, you need to properly install JAX yourself. Otherwise, if JAX is installed automatically as a pip dependency of *xomx*, it will probably not work as desired (e.g. it will not be GPU-compatible). So you should install it beforehand, following these guidelines: 

[https://github.com/google/jax#installation](https://github.com/google/jax#installation) 

Then, install *xomx* with:

    pip install xomx

</p>
</details>

<details><summary>JAX</summary>
<p>

The neural network-based machine learning algorithms in *xomx* are written in JAX (and flax), so it needs to be installed properly for them to work.

To verify that the JAX installation went well, check the backend used by JAX with the following command:
```
python -c "import jax; print(jax.lib.xla_bridge.get_backend().platform)"
```
It will print "cpu", "gpu" or "tpu" depending on the platform JAX is using.

</p>
</details>

-----
## Tutorials

Tutorials are the best way to learn how to use
*xomx*.

The xomx-tutorials repository contains a list of tutorials (colab notebooks) for xomx:  
https://github.com/perrin-isir/xomx-tutorials

-----
## Acknowledgements

Maintainer and main contributor:
- Nicolas Perrin-Gilbert (CNRS, ISIR)

Other people who contributed to *xomx*:
- Joshua J. Waterfall (Curie Institute)
- Julien Vibert (Curie Institute)
- Mathias Vandenbogaert (Curie Institute)
- Paul Klein (Curie Institute)

-----
## Citing the project
To cite this repository in publications:

```bibtex
@misc{xomx,
  author = {Perrin-Gilbert, Nicolas and Vibert, Julien and Vandenbogaert, Mathias and Waterfall, Joshua J.},
  title = {xomx},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/perrin-isir/xomx}},
}
```

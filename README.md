# ![alt text](logo.png "logo")

![version](https://img.shields.io/badge/version-0.1.0-blue)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


*xomx* is an open-source python library providing data processing and 
machine learning tools for computational omics, with a 
particular emphasis on explainability.

It relies on [AnnData](https://anndata.readthedocs.io) objects, which makes it
fully compatible with [Scanpy](https://scanpy.readthedocs.io).

*xomx is currently in beta version.*

-----



## Install

<details><summary>Option 1: pip</summary>
<p>

    pip install git+https://github.com/perrin-isir/xomx

</p>
</details>

<details><summary>Option 2: conda</summary>
<p>

    git clone https://github.com/perrin-isir/xomx.git
    cd xomx

Choose a conda environmnent name, for instance `xomxv`.  
The following command creates the `xomxv` environment with the requirements listed in [environment.yaml](environment.yaml):

    conda env create --name xomxv --file environment.yaml

If you prefer to update an existing environment (`existing_env`):

    conda env update --name existing_env --file environment.yml

To activate the `xomxv` environment:

    conda activate xomxv

Finally, to install the *xomx* library in the activated virtual environment:

    pip install -e .

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

Here is the list of people who have contributed to *xomx*:
- Nicolas Perrin-Gilbert (ISIR)
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

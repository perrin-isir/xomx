from setuptools import setup, find_packages

# Install with 'pip install -e .'

setup(
    name="xomx",
    version="0.1.0",
    author="Nicolas Perrin-Gilbert",
    description="xomx: a python library for computational omics",
    url="https://github.com/perrin-isir/xomx",
    packages=find_packages(),
    install_requires=[
        "argparse>=1.1",
        "pandas>=1.3.0",
        "numpy>=1.21.1",
        "anndata>=0.7.6",
        "scipy>=1.4.1",
        "matplotlib>=3.1.3",
        "torch>=1.7.1",
        "scikit-learn>=0.24.2",
        "joblib>=1.0.1",
        "requests>=2.23.0",
        "leidenalg>=0.8.8",
        "holoviews>=1.15.0",
        "bokeh>=2.4.3",
    ],
    license="LICENSE",
)

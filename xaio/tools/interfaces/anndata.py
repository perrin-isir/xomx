import anndata
import numpy as np


def anndata_interface(data_matrix: np.ndarray):
    ad = anndata.AnnData(shape=data_matrix.shape)
    ad.X = data_matrix
    return ad

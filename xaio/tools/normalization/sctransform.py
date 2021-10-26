from xaio.tools.basic_tools import XAIOData
from SCTransform import SCTransform
from anndata import AnnData
from scipy.sparse import csr_matrix
import numpy as np
from IPython import embed as e

assert e


def compute_sctransform(data: XAIOData):
    assert data.data_array["raw"] is not None
    tmp_csr = AnnData(csr_matrix(data.data_array["raw"]))

    sct_data = SCTransform(
        tmp_csr,
        min_cells=5,
        gmean_eps=1,
        n_genes=2000,
        n_cells=None,  # use all cells
        bin_size=500,
        bw_adjust=3,
        inplace=False,
    )

    data.data_array["sct"] = sct_data.X.toarray()


def compute_logsctransform(data: XAIOData):
    assert data.data_array["sct"] is not None and data.nr_features is not None
    data.data_array["logsct"] = np.copy(data.data_array["sct"])
    if data.params is None:
        data.params = {}
    data.params["logsct_epsilon_shift"] = 1.0
    for i in range(data.nr_features):
        data.data_array["logsct"][:, i] = np.log(
            data.data_array["logsct"][:, i] + data.params["logsct_epsilon_shift"]
        )
    data.params["logsct_maxlog"] = np.max(data.data_array["logsct"])
    for i in range(data.nr_features):
        data.data_array["logsct"][:, i] = (
            data.data_array["logsct"][:, i]
            - np.log(data.params["logsct_epsilon_shift"])
        ) / (data.params["logsct_maxlog"] - np.log(data.params["logsct_epsilon_shift"]))

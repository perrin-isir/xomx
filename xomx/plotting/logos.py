import pandas as pd
import numpy as np
import scanpy as sc
import xomx
import logomaker
import matplotlib.pyplot as plt
from scipy.stats import entropy
# from xomx.tools import aminoacids


def compute_logo_df(adata: sc.AnnData, indices):
    """
    The sample names (adata.obs_names) must be strings made of amino acid characters.
    The list of allowed characters is xomx.tl.aminoacids.
    """
    pos_list = np.arange(max([len(adata.obs_names[idx]) for idx in indices]))
    probability_matrix = np.zeros((len(pos_list), len(xomx.tl.aminoacids)))
    for position in pos_list:
        counts = {}
        for aa in xomx.tl.aminoacids:
            counts[aa] = 0
        for idx in indices:
            if position < len(adata.obs_names[idx]):
                counts[adata.obs_names[idx][position]] += 1
        for k in range(len(xomx.tl.aminoacids)):
            probability_matrix[position, k] = counts[
                                                  xomx.tl.aminoacids[k]] / len(indices)
    # from probabilities to bits:
    max_entropy = -np.log2(1/len(xomx.tl.aminoacids))
    for position in pos_list:
        pos_entropy = max_entropy - entropy(1e-10 + probability_matrix[position, :],
                                            base=2)
        probability_matrix[position, :] *= pos_entropy
    dico = {'pos': pos_list}
    for k in range(len(xomx.tl.aminoacids)):
        dico[xomx.tl.aminoacids[k]] = probability_matrix[:, k]
    df_ = pd.DataFrame(dico)
    df_ = df_.set_index('pos')
    return df_


def plot_logo(adata: sc.AnnData, indices):
    """
    The sample names (adata.obs_names) must be strings made of amino acid characters.
    The list of allowed characters is xomx.tl.aminoacids.
    """
    df = compute_logo_df(adata, indices)
    fig, ax = plt.subplots(1, 1, figsize=[4, 2])
    logo = logomaker.Logo(df,
                          ax=ax,
                          baseline_width=0,
                          show_spines=False,
                          vsep=.005,
                          width=.95)
    logo.fig.tight_layout()
    plt.show()

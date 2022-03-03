# import numpy as np
# from pandas import DataFrame
# from xomx.tools.bio import aminoacids
# # import matplotlib.pyplot as plt
# from scipy.stats import entropy


# def plot_logo(adata, indices, fixed_length: int = None):
#     """
#     The sample names (adata.obs_names) must be strings made of amino acid characters.
#     The list of allowed characters is aminoacids.
#     """
#     df = compute_logo_df(adata, indices, fixed_length)
#     fig, ax = plt.subplots(1, 1, figsize=[4, 2])
#     logo = Logo(df,
#                 ax=ax,
#                 baseline_width=0,
#                 show_spines=False,
#                 vsep=.005,
#                 width=.95)
#     logo.fig.tight_layout()
#     plt.show()

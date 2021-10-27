import os
import numpy as np
import scanpy as sc

# import re
import matplotlib.pyplot as plt

# import umap


def plot_scores(
    adata: sc.AnnData, scores, score_threshold, indices, annotation=None, save_dir=None
):
    annot_colors = {}
    assert "all_labels" in adata.uns and "labels" in adata.obs
    denom = len(adata.uns["all_labels"])
    for i, val in enumerate(adata.uns["all_labels"]):
        if annotation:
            if val == annotation:
                annot_colors[val] = 0.0 / denom
            else:
                annot_colors[val] = (denom + i) / denom
        else:
            annot_colors[val] = i / denom

    samples_color = np.zeros(len(indices))
    for i in range(len(indices)):
        samples_color[i] = annot_colors[adata.obs["labels"][indices[i]]]

    fig, ax = plt.subplots()
    if annotation:
        cm = "winter"
    else:
        cm = "nipy_spectral"
    sctr = ax.scatter(np.arange(len(indices)), scores, c=samples_color, cmap=cm, s=5)
    ax.axhline(y=score_threshold, xmin=0, xmax=1, lw=1, ls="--", c="red")
    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(-100, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def update_annot(ind, sctr_):
        pos = sctr_.get_offsets()[ind["ind"][0]]
        ann.xy = pos
        text = "{}: {}".format(
            adata.obs_names[indices[ind["ind"][0]]],
            adata.obs["labels"][indices[ind["ind"][0]]],
        )
        ann.set_text(text)

    def hover(event):
        vis = ann.get_visible()
        if event.inaxes == ax:
            cont, ind = sctr.contains(event)
            if cont:
                update_annot(ind, sctr)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()
        if event.inaxes == ax:
            cont, ind = sctr.contains(event)
            if cont:
                update_annot(ind, sctr)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "plot.png"), dpi=200)
    else:
        plt.show()

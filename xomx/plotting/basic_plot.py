import os
import numpy as np
import xaio
import scanpy as sc
import umap
import matplotlib.pyplot as plt


def _hover(event, fig, ax, ann, sctr, update_annot):
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


def plot_scores(
    adata: sc.AnnData,
    scores,
    score_threshold,
    indices,
    label=None,
    save_dir=None,
    text_complements=None,
    lines=False,
    yticks=None,
    ylabel="",
):
    annot_colors = {}
    assert "all_labels" in adata.uns and "labels" in adata.obs
    denom = len(adata.uns["all_labels"])
    for i, val in enumerate(adata.uns["all_labels"]):
        if label:
            if val == label:
                annot_colors[val] = 0.0 / denom
            else:
                annot_colors[val] = (denom + i) / denom
        else:
            annot_colors[val] = i / denom

    samples_color = np.zeros(len(indices))
    for i in range(len(indices)):
        samples_color[i] = annot_colors[adata.obs["labels"][indices[i]]]

    fig, ax = plt.subplots()
    if label:
        cm = "winter"
    else:
        cm = "nipy_spectral"
    sctr = ax.scatter(np.arange(len(indices)), scores, c=samples_color, cmap=cm, s=5)
    if score_threshold is not None:
        ax.axhline(y=score_threshold, xmin=0, xmax=1, lw=1, ls="--", c="red")
    if lines:
        for k_ in range(denom + 1):
            ax.axhline(y=k_, xmin=0, xmax=1, lw=0.2, ls="-", c="gray")

    if yticks is not None:
        plt.yticks(
            np.arange(len(yticks)) + 0.5,
            yticks,
        )

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
        if text_complements is not None:
            text += text_complements[ind["ind"][0]]
        ann.set_text(text)

    # def hover(event):
    #     vis = ann.get_visible()
    #     if event.inaxes == ax:
    #         cont, ind = sctr.contains(event)
    #         if cont:
    #             update_annot(ind, sctr)
    #             ann.set_visible(True)
    #             fig.canvas.draw_idle()
    #         else:
    #             if vis:
    #                 ann.set_visible(False)
    #                 fig.canvas.draw_idle()

    # fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect(
        "motion_notify_event",
        lambda event: _hover(event, fig, ax, ann, sctr, update_annot),
    )

    plt.ylabel(ylabel)
    plt.xlabel("samples (test set)")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "plot.png"), dpi=200)
    else:
        plt.show()


def _samples_by_labels(adata: sc.AnnData, sort_annot=False):
    assert "obs_indices_per_label" in adata.uns and "all_labels" in adata.uns
    if sort_annot:
        argsort_labels = np.argsort(
            [
                len(adata.uns["obs_indices_per_label"][i])
                for i in adata.uns["all_labels"]
            ]
        )[::-1]
    else:
        argsort_labels = np.arange(len(adata.uns["all_labels"]))
    list_samples = np.concatenate(
        [
            adata.uns["obs_indices_per_label"][adata.uns["all_labels"][i]]
            for i in argsort_labels
        ]
    )
    boundaries = np.cumsum(
        [0]
        + [
            len(adata.uns["obs_indices_per_label"][adata.uns["all_labels"][i]])
            for i in argsort_labels
        ]
    )
    set_xticks2 = (boundaries[1:] + boundaries[:-1]) // 2
    set_xticks = list(np.sort(np.concatenate((boundaries, set_xticks2))))
    set_xticks_text = ["|"] + list(
        np.concatenate([[str(adata.uns["all_labels"][i]), "|"] for i in argsort_labels])
    )
    return list_samples, set_xticks, set_xticks_text, boundaries


def _identity_func(x):
    return x


def function_scatter(
    adata: sc.AnnData,
    func1_=_identity_func,
    func2_=_identity_func,
    obs_or_var="obs",
    violinplot=False,
    xlog_scale=False,
    ylog_scale=False,
    xlabel="",
    ylabel="",
    function_plot_=False,
):
    """Displays a scatter plot, with coordinates computed by applying two
    functions (func1_ and func2_) to every sample or every feature, depending
    on the value of obs_or_var which must be either "obs" or "var"
    (both functions must take indices in input)
    """
    assert obs_or_var == "obs" or obs_or_var == "var"
    set_xticks = None
    set_xticks_text = None
    violinplots_done = False
    fig, ax = plt.subplots()
    if obs_or_var == "obs":
        if "all_labels" in adata.uns and function_plot_:
            (
                list_samples,
                set_xticks,
                set_xticks_text,
                boundaries,
            ) = _samples_by_labels(adata, sort_annot=True)
            y = [func2_(i) for i in list_samples]
            x = [i for i in range(adata.n_obs)]
            if violinplot:
                for i in range(len(boundaries) - 1):
                    parts = ax.violinplot(
                        y[boundaries[i] : boundaries[i + 1]],
                        [(boundaries[i + 1] + boundaries[i]) / 2.0],
                        points=60,
                        widths=(boundaries[i + 1] - boundaries[i]) * 0.8,
                        showmeans=False,
                        showextrema=False,
                        showmedians=False,
                        bw_method=0.5,
                    )
                    for pc in parts["bodies"]:
                        pc.set_facecolor("#D43F3A")
                        pc.set_edgecolor("grey")
                        pc.set_alpha(0.5)
            violinplots_done = True
        else:
            y = [func2_(i) for i in range(adata.n_obs)]
            x = [func1_(i) for i in range(adata.n_obs)]
    else:
        y = [func2_(i) for i in range(adata.n_vars)]
        x = [func1_(i) for i in range(adata.n_vars)]
    xmax = np.max(x)
    xmin = np.min(x)
    if violinplot and not violinplots_done:
        parts = ax.violinplot(
            y,
            [(xmax + xmin) / 2.0],
            points=60,
            widths=xmax - xmin,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            bw_method=0.5,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_edgecolor("grey")
            pc.set_alpha(0.5)
    scax = ax.scatter(x, y, s=1)
    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(-100, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def update_annot(ind, scax_):
        pos = scax_.get_offsets()[ind["ind"][0]]
        ann.xy = pos
        if obs_or_var == "obs":
            if "all_labels" in adata.uns and function_plot_:
                text = "{}".format(adata.obs_names[list_samples[ind["ind"][0]]])
            else:
                text = "{}".format(adata.obs_names[ind["ind"][0]])

        else:
            text = "{}".format(adata.var_names[ind["ind"][0]])
        ann.set_text(text)

    def hover(event):
        vis = ann.get_visible()
        if event.inaxes == ax:
            cont, ind = scax.contains(event)
            if cont:
                update_annot(ind, scax)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if set_xticks is not None:
        plt.xticks(set_xticks, set_xticks_text)
    if xlog_scale:
        plt.xscale("log")
    if ylog_scale:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def function_plot(
    adata: sc.AnnData,
    func=_identity_func,
    obs_or_var="obs",
    violinplot=True,
    ylog_scale=False,
    xlabel="",
    ylabel="",
):
    """Plots the value of a function on every sample or every feature, depending
    on the value of obs_or_var which must be either "obs" or "var"
    (the function must take indices in input)"""
    function_scatter(
        adata,
        _identity_func,
        func,
        obs_or_var,
        violinplot,
        xlog_scale=False,
        ylog_scale=ylog_scale,
        xlabel=xlabel,
        ylabel=ylabel,
        function_plot_=True,
    )


def var_plot(adata: sc.AnnData, features=None, ylog_scale=False):
    """ """
    if type(features) == str or type(features) == np.str_ or type(features) == int:
        idx = features
        if type(idx) == str or type(idx) == np.str_:
            assert "var_indices" in adata.uns
            idx = adata.uns["var_indices"][idx]
        function_plot(adata, lambda i: adata.X[i, idx], "obs", ylog_scale=ylog_scale)
    else:
        xsize = adata.n_obs
        ysize = len(features)
        set_xticks = None
        set_xticks_text = None
        plot_array = np.empty((len(features), xsize))
        feature_indices_list_ = []
        for idx in features:
            if type(idx) == str or type(idx) == np.str_:
                assert "var_indices" in adata.uns
                idx = adata.uns["var_indices"][idx]
            feature_indices_list_.append(idx)
        if "all_labels" not in adata.uns:
            for k, idx in enumerate(feature_indices_list_):
                plot_array[k, :] = [adata.X[i, idx] for i in range(adata.n_obs)]
        else:
            (
                list_samples,
                set_xticks,
                set_xticks_text,
                boundaries,
            ) = _samples_by_labels(adata, sort_annot=False)
            for k, idx in enumerate(feature_indices_list_):
                plot_array[k, :] = [adata.X[i, idx] for i in list_samples]

        fig, ax = plt.subplots()
        im = ax.imshow(plot_array, extent=[0, xsize, 0, ysize], aspect="auto")
        if set_xticks is not None:
            plt.xticks(set_xticks, set_xticks_text)
        plt.yticks(
            np.arange(ysize) + 0.5,
            [adata.var_names[i] for i in feature_indices_list_][::-1],
        )
        plt.tick_params(axis=u"both", which=u"both", length=0)
        plt.colorbar(im)
        plt.show()


def umap_plot(
    adata: sc.AnnData,
    save_dir=None,
    metric="cosine",
    min_dist=0.0,
    n_neighbors=30,
    random_state=None,
):
    assert "labels" in adata.obs and "all_labels" in adata.uns
    reducer = umap.UMAP(
        metric=metric,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    print("Starting UMAP reduction...")
    reducer.fit(adata.X)
    embedding = reducer.transform(adata.X)
    print("Done.")

    def hover_function(id_):
        return "{}".format(adata.obs_names[id_] + ": " + str(adata.obs["labels"][id_]))

    annot_idxs = {}
    for i, annot_ in enumerate(adata.uns["all_labels"]):
        annot_idxs[annot_] = i

    samples_color = np.empty(adata.n_obs)
    for i in range(adata.n_obs):
        samples_color[i] = annot_idxs[adata.obs["labels"][i]]

    fig, ax = plt.subplots()

    sctr = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=samples_color, cmap="nipy_spectral", s=5
    )
    plt.gca().set_aspect("equal", "datalim")

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def update_annot(ind):
        pos = sctr.get_offsets()[ind["ind"][0]]
        ann.xy = pos
        text = hover_function(ind["ind"][0])
        ann.set_text(text)

    def hover(event):
        vis = ann.get_visible()
        if event.inaxes == ax:
            cont, ind = sctr.contains(event)
            if cont:
                update_annot(ind)
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


def plot2d(
    adata: sc.AnnData,
    obsm_key,
    var_key=None,
    save_dir=None,
):
    if "labels" in adata.obs:

        def hover_function(id_):
            return "{}".format(
                adata.obs_names[id_] + ": " + str(adata.obs["labels"][id_])
            )

    else:

        def hover_function(id_):
            return "{}".format(adata.obs_names[id_])

    if var_key is not None:
        colorbar = True
        cmap = "viridis"
        samples_color = np.squeeze(np.asarray(xaio.tl._to_dense(adata[:, var_key].X)))
    else:
        if "labels" in adata.obs and "all_labels" in adata.uns:
            colorbar = False
            cmap = "nipy_spectral"
            annot_idxs = {}
            for i, annot_ in enumerate(adata.uns["all_labels"]):
                annot_idxs[annot_] = i

            samples_color = np.empty(adata.n_obs)
            for i in range(adata.n_obs):
                samples_color[i] = annot_idxs[adata.obs["labels"][i]]

        else:
            colorbar = False
            cmap = "viridis"
            samples_color = np.zeros(adata.n_obs)

    fig, ax = plt.subplots()

    sctr = plt.scatter(
        adata.obsm[obsm_key][:, 0],
        adata.obsm[obsm_key][:, 1],
        c=samples_color,
        cmap=cmap,
        s=5,
    )
    plt.gca().set_aspect("equal", "datalim")
    if colorbar:
        plt.colorbar(location="left", aspect=50)

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def update_annot(ind):
        pos = sctr.get_offsets()[ind["ind"][0]]
        ann.xy = pos
        text = hover_function(ind["ind"][0])
        ann.set_text(text)

    def hover(event):
        vis = ann.get_visible()
        if event.inaxes == ax:
            cont, ind = sctr.contains(event)
            if cont:
                update_annot(ind)
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

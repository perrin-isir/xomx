import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import rcParams
from typing import Optional
import string
from xomx.tools.utils import _to_dense


global_xomx_force_extension = False
global_xomx_extension_bokeh_or_matplotlib = "bokeh"


def force_extension(bokeh_or_matplotlib: str):
    global global_xomx_force_extension, global_xomx_extension_bokeh_or_matplotlib
    assert bokeh_or_matplotlib in [
        "bokeh",
        "matplotlib",
    ], 'Input must be "bokeh" or "matplotlib".'
    global_xomx_force_extension = True
    global_xomx_extension_bokeh_or_matplotlib = bokeh_or_matplotlib


def extension(bokeh_or_matplotlib: str):
    global global_xomx_force_extension, global_xomx_extension_bokeh_or_matplotlib
    assert bokeh_or_matplotlib in [
        "bokeh",
        "matplotlib",
    ], 'Input must be "bokeh" or "matplotlib".'
    if not global_xomx_force_extension:
        global_xomx_extension_bokeh_or_matplotlib = bokeh_or_matplotlib
    else:
        print(
            f"Warning: a call to xomx.pl.force_extension() was previously made. "
            f"The extension remains: {global_xomx_extension_bokeh_or_matplotlib}"
        )


def colormap(
    *,
    width: int = 900,
    height: int = 150,
):
    if global_xomx_extension_bokeh_or_matplotlib == "matplotlib":
        rcParams["figure.dpi"] = 100
        rcParams["figure.figsize"] = (
            900 / rcParams["figure.dpi"],
            350 / rcParams["figure.dpi"],
        )
        fig, axes = plt.subplots(2, 1)
        cm = ["nipy_spectral", "viridis"]
        for i in range(2):
            axes[i].imshow(
                np.vstack((np.linspace(0, 1, 100), np.linspace(0, 1, 100))),
                extent=[0, 1, 0, 1],
                aspect="auto",
                cmap=cm[i],
            )
            axes[i].get_yaxis().set_visible(False)
            axes[i].locator_params(axis="x", nbins=20)
        plt.show()
    elif global_xomx_extension_bokeh_or_matplotlib == "bokeh":
        import holoviews as hv  # lazy import
        import bokeh.models  # lazy import
        import bokeh.io  # lazy import
        import bokeh.layouts  # lazy import

        hv.extension("bokeh")
        cbar_nipy_spectral = hv.Image(
            np.linspace(0, 1, 100)[np.newaxis], ydensity=1, bounds=(0, 0, 1, 1)
        ).opts(
            cmap="nipy_spectral",
            xticks=20,
            xlabel="",
            height=height,
            width=width,
            yaxis=None,
        )
        cbar_viridis = hv.Image(
            np.linspace(0, 1, 100)[np.newaxis], ydensity=1, bounds=(0, 0, 1, 1)
        ).opts(
            cmap="viridis",
            xticks=20,
            xlabel="",
            height=height,
            width=width,
            yaxis=None,
        )
        bokeh.io.show(
            bokeh.layouts.column(hv.render(cbar_nipy_spectral), hv.render(cbar_viridis))
        )
    else:
        raise ValueError(
            'Execute xomx.pl.extension("bokeh") or xomx.pl.extension("matplotlib")'
        )


def _custom_legend(bokeh_plot):
    from bokeh.models import Legend

    if len(bokeh_plot.legend) > 0:
        bokeh_plot.legend[0].items[0].visible = False
        glyph = bokeh_plot.legend[0].items[0].renderers[0].glyph
        factors = glyph.fill_color["transform"].factors
        palette = glyph.fill_color["transform"].palette
        size = glyph.size
        height = 24
        margin = 0
        spacing = 0
        padding = 5
        max_nr = (bokeh_plot.height - 2 * margin - 2 * padding - height) // (
            height + spacing
        )
        full_length = len(factors)
        cuts = list(np.arange(0, full_length, max_nr)) + [full_length]
        list_intervals = [np.arange(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        for itvl in list_intervals:
            items_list = [
                (factors[i], [bokeh_plot.scatter(size=size, color=palette[i])])
                for i in itvl
            ]
            legend = Legend(
                items=items_list,
                label_height=height,
                glyph_height=height,
                spacing=spacing,
                padding=padding,
                margin=margin,
            )
            bokeh_plot.add_layout(legend, "right")
    return bokeh_plot


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
    adata,
    scores: np.ndarray,
    score_threshold,
    indices,
    label=None,
    *,
    pointsize: int = 5,
    output_file: Optional[str] = None,
    text_complements=None,
    lines: bool = False,
    yticks=None,
    ylabel: str = "",
    title: str = "",
    width: int = 900,
    height: int = 600,
):
    global global_xomx_extension_bokeh_or_matplotlib
    annot_colors = {}
    assert "all_labels" in adata.uns and "labels" in adata.obs, (
        "plot_scores() requires data with labels (even if there is only one label), "
        'so adata.obs["labels"] and adata.uns["all_labels"] must exist.'
    )
    denom = max(len(adata.uns["all_labels"]) - 1, 1)
    for i, val in enumerate(adata.uns["all_labels"]):
        if "label_colors" in adata.uns:
            annot_colors[val] = adata.uns["label_colors"][i]
        elif label:
            if val == label:
                annot_colors[val] = 0.0 / denom
            else:
                annot_colors[val] = (denom + i) / denom
        else:
            annot_colors[val] = i / denom

    sample_colors = np.zeros(len(indices))
    for i in range(len(indices)):
        sample_colors[i] = annot_colors[adata.obs["labels"][indices[i]]]

    xlabel = "samples"

    if global_xomx_extension_bokeh_or_matplotlib == "matplotlib":
        rcParams["figure.dpi"] = 100
        rcParams["figure.figsize"] = (
            width / rcParams["figure.dpi"],
            height / rcParams["figure.dpi"],
        )
        fig, ax = plt.subplots()
        if label:
            cm = "winter"
        else:
            cm = "nipy_spectral"
        sctr = ax.scatter(
            np.arange(len(indices)),
            scores,
            c=sample_colors,
            cmap=cm,
            norm=matplotlib.colors.NoNorm(),
            s=pointsize,
        )
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
            print(text)

        fig.canvas.mpl_connect(
            "motion_notify_event",
            lambda event: _hover(event, fig, ax, ann, sctr, update_annot),
        )

        def lp(i_):
            val_ = adata.uns["all_labels"][i_]
            return plt.plot(
                [],
                color=sctr.cmap(sctr.norm(annot_colors[val_])),
                ms=5,
                mec="none",
                label=val_,
                ls="",
                marker="o",
            )[0]

        handles = [lp(i) for i in range(len(adata.uns["all_labels"]))]
        plt.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.0, 0.0))
        plt.subplots_adjust(right=0.75)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if output_file:
            plt.savefig(output_file, dpi=200)
        else:
            plt.show()

    ####################################################################################
    # Bokeh
    elif global_xomx_extension_bokeh_or_matplotlib == "bokeh":
        import holoviews as hv  # lazy import
        import bokeh.models  # lazy import
        import bokeh.io  # lazy import
        import bokeh.layouts  # lazy import
        from bokeh.models import HoverTool  # lazy import

        tmp_df = adata.obs.iloc[indices].copy()

        random_id = "".join(np.random.choice(list(string.ascii_letters), 10)) + "_"
        tooltips = [("name", "@{" + random_id + "name}")]
        if text_complements is not None:
            tooltips += [("info", "@{" + random_id + "info}")]
        tooltips += [(key, "@{" + key + "}") for key in tmp_df.keys()]
        tmp_df[random_id + "x_" + xlabel] = np.arange(len(indices))
        tmp_df[random_id + "y_" + ylabel] = scores.reshape(len(indices))
        tooltips += [
            ("x:" + xlabel, "@{" + random_id + "x_" + xlabel + "}"),
            ("y:" + ylabel, "@{" + random_id + "y_" + ylabel + "}"),
        ]

        tmp_df[random_id + "name"] = adata.obs_names[indices]
        if text_complements is not None:
            tmp_df[random_id + "info"] = text_complements
        tmp_df[random_id + "colors"] = sample_colors
        tmp_cmap = "nipy_spectral"
        new_tmp_cmap = {}
        for i, lbl in enumerate(adata.uns["all_labels"]):
            new_tmp_cmap[lbl] = matplotlib.colors.rgb2hex(
                plt.get_cmap(tmp_cmap)(
                    adata.uns["label_colors"][i]
                    if "label_colors" in adata.uns
                    else annot_colors[lbl]
                )
            )
        tmp_cmap = new_tmp_cmap
        hv.extension("bokeh")
        points = hv.Points(
            tmp_df,
            [random_id + "x_" + xlabel, random_id + "y_" + ylabel],
            list(tmp_df.keys()),
        )
        bokeh_data = bokeh.models.ColumnDataSource(
            tmp_df.loc[:, random_id + "name"].to_frame()
        )
        hover = HoverTool(tooltips=tooltips)
        div = bokeh.models.Div(width=width, height=50, height_policy="fixed")
        cb = bokeh.models.CustomJS(
            args=dict(
                hvr=hover, div=div, source=bokeh_data.data, col_name=random_id + "name"
            ),
            code="""
                       if (cb_data['index'].indices.length > 0) {
                           const line_list = [];
                           for (let i = 0; i<cb_data['index'].indices.length; i++) {
                               var line = "<b>"
                               line += source[col_name][cb_data['index'].indices[i]]
                               line += "</b>"
                               line_list.push(line)
                           }
                           div.text = line_list.join(" --- ")
                       }
                   """,
        )
        hover.callback = cb  # callback whenever the HoverTool function is called

        points.opts(
            tools=[hover],
            color="labels"
            if ("all_labels" in adata.uns and "labels" in adata.obs)
            else random_id + "colors",
            cmap=tmp_cmap,
            size=pointsize,
            width=width,
            height=height,
            show_grid=False,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            logx=False,
            logy=False,
        )
        if yticks is not None:
            points.opts(
                yticks=list(
                    zip([float(y) for y in np.arange(len(yticks)) + 0.5], yticks)
                )
            )
        hline = []
        if score_threshold is not None:
            hline.append(hv.HLine(score_threshold))
            hline[-1].opts(
                color="red",
                line_dash="dashed",
                line_width=1.0,
            )
        if lines:
            for k_ in range(denom + 1):
                hline.append(hv.HLine(k_))
                hline[-1].opts(
                    color="gray",
                    line_width=0.5,
                )
        hv_plot = hv.Overlay([points] + hline)
        if output_file:
            hv.save(hv_plot, output_file, fmt="html")
        else:
            bokeh.io.show(bokeh.layouts.column(_custom_legend(hv.render(hv_plot)), div))
        del tmp_df[random_id + "name"]
        if text_complements is not None:
            del tmp_df[random_id + "info"]
        del tmp_df[random_id + "colors"]
        del tmp_df[random_id + "x_" + xlabel]
        del tmp_df[random_id + "y_" + ylabel]
    ####################################################################################
    else:
        raise ValueError(
            'Execute xomx.pl.extension("bokeh") or xomx.pl.extension("matplotlib")'
        )


def _samples_by_labels(adata, sort_annot=False, subset_indices=None, equal_size=False):
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

    if subset_indices is None:
        if not equal_size:
            i_per_label = adata.uns["obs_indices_per_label"]
        else:
            i_per_label = adata.uns["obs_indices_per_label"].copy()
    else:
        i_per_label = {}
        for lbl in adata.uns["all_labels"]:
            i_per_label[lbl] = []
        for i, annot in enumerate(adata.obs["labels"][subset_indices]):
            i_per_label[annot].append(int(subset_indices[i]))
        for lbl in adata.uns["all_labels"]:
            i_per_label[lbl] = np.array(i_per_label[lbl])

    if equal_size:
        max_size = 0
        for lbl in adata.uns["all_labels"]:
            len_lbl = len(i_per_label[lbl])
            if len_lbl > max_size:
                max_size = len_lbl
        for lbl in adata.uns["all_labels"]:
            tmp_array = np.repeat(
                i_per_label[lbl], int(np.ceil(max_size / len(i_per_label[lbl])))
            )
            rng = np.random.default_rng()
            rng.shuffle(tmp_array)
            i_per_label[lbl] = tmp_array[:max_size]

    list_samples = np.concatenate(
        [i_per_label[adata.uns["all_labels"][i]] for i in argsort_labels]
    ).astype("int")

    boundaries = np.cumsum(
        [0] + [len(i_per_label[adata.uns["all_labels"][i]]) for i in argsort_labels]
    )

    set_xticks2 = (boundaries[1:] + boundaries[:-1]) // 2
    set_xticks = list(np.sort(np.concatenate((boundaries, set_xticks2))))
    set_xticks_text = ["|"] + list(
        np.concatenate([[str(adata.uns["all_labels"][i]), "|"] for i in argsort_labels])
    )
    return list_samples, set_xticks, set_xticks_text, boundaries


def _identity_func(x):
    return x


def scatter2d_and_3d(
    adata,
    func1=_identity_func,
    func2=_identity_func,
    func3_=None,
    obs_or_var: str = "obs",
    *,
    pointsize: int = 3,
    xlog_scale: bool = False,
    ylog_scale: bool = False,
    zlog_scale: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    title: str = "",
    subset_indices=None,
    equal_size=False,
    output_file: Optional[str] = None,
    width=900,
    height=600,
):
    """Displays a scatter plot, with coordinates computed by applying two
    functions (`func1` and `func2`) to every sample or every feature, depending
    on the value of obs_or_var which must be either "obs" or "var"
    (both functions must take indices in input)
    """
    global global_xomx_extension_bokeh_or_matplotlib
    point_size = pointsize
    if func1 == _identity_func:
        function_plot_ = True
        mode3d = False
    else:
        function_plot_ = False
        if func3_ is None:
            mode3d = False
        else:
            mode3d = True
    assert obs_or_var == "obs" or obs_or_var == "var"
    set_xticks = None
    set_xticks_text = None
    sample_colors = None
    colormap = None
    color_min = 0.0
    color_max = 0.0
    if global_xomx_extension_bokeh_or_matplotlib == "matplotlib":
        rcParams["figure.dpi"] = 100
        rcParams["figure.figsize"] = (
            width / rcParams["figure.dpi"],
            height / rcParams["figure.dpi"],
        )
        if not mode3d:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
    if obs_or_var == "obs":
        if "all_labels" in adata.uns and "labels" in adata.obs and function_plot_:
            (
                list_samples,
                set_xticks,
                set_xticks_text,
                boundaries,
            ) = _samples_by_labels(
                adata,
                sort_annot=True,
                subset_indices=subset_indices,
                equal_size=equal_size,
            )
            y = [func2(i) for i in list_samples]
            x = [i for i in range(len(y))]
            subset_indices = list_samples
            if global_xomx_extension_bokeh_or_matplotlib == "matplotlib":
                # violin plots
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
        else:
            if subset_indices is None:
                y = [func2(i) for i in range(adata.n_obs)]
                x = [func1(i) for i in range(adata.n_obs)]
                if mode3d:
                    z = [func3_(i) for i in range(adata.n_obs)]
                else:
                    z = None
            else:
                y = [func2(i) for i in subset_indices]
                x = [func1(i) for i in subset_indices]
                if mode3d:
                    z = [func3_(i) for i in subset_indices]
                else:
                    z = None
        if "colors" in adata.obs:
            colormap = "viridis"
            if subset_indices is None:
                color_min = adata.obs["colors"].min()
                color_max = adata.obs["colors"].max()
                if color_max - color_min < 1e-7:
                    color_max += 1e-7
                sample_colors = adata.obs["colors"]
            else:
                color_min = adata.obs["colors"][subset_indices].min()
                color_max = adata.obs["colors"][subset_indices].max()
                if color_max - color_min < 1e-7:
                    color_max += 1e-7
                sample_colors = adata.obs["colors"][subset_indices]
        elif "all_labels" in adata.uns and "labels" in adata.obs:
            annot_colors = {}
            denom = max(len(adata.uns["all_labels"]) - 1, 1)
            for i, val in enumerate(adata.uns["all_labels"]):
                if "label_colors" in adata.uns:
                    annot_colors[val] = adata.uns["label_colors"][i]
                else:
                    annot_colors[val] = i / denom
            sample_colors = np.zeros_like(x, dtype=float)
            if subset_indices is None:
                for i in range(adata.n_obs):
                    sample_colors[i] = annot_colors[adata.obs["labels"][i]]
            else:
                for i in range(len(subset_indices)):
                    sample_colors[i] = annot_colors[
                        adata.obs["labels"][subset_indices[i]]
                    ]
    else:
        if subset_indices is None:
            y = [func2(i) for i in range(adata.n_vars)]
            x = [func1(i) for i in range(adata.n_vars)]
            if mode3d:
                z = [func3_(i) for i in range(adata.n_vars)]
            else:
                z = None
        else:
            y = [func2(i) for i in subset_indices]
            x = [func1(i) for i in subset_indices]
            if mode3d:
                z = [func3_(i) for i in subset_indices]
            else:
                z = None
        if "colors" in adata.var:
            colormap = "viridis"
            if subset_indices is None:
                color_min = adata.var["colors"].min()
                color_max = adata.var["colors"].max()
                if color_max - color_min < 1e-7:
                    color_max += 1e-7
                sample_colors = adata.var["colors"]
            else:
                color_min = adata.var["colors"].min()
                color_max = adata.var["colors"].max()
                if color_max - color_min < 1e-7:
                    color_max += 1e-7
                sample_colors = adata.var["colors"][subset_indices]

    if colormap is None:
        colormap = "nipy_spectral"
    if color_min >= 0.0 and color_max <= 1.0:
        color_min = 0.0
        color_max = 1.0

    if global_xomx_extension_bokeh_or_matplotlib == "matplotlib":
        if sample_colors is None:
            if not mode3d:
                scax = ax.scatter(
                    x,
                    y,
                    s=point_size,
                    c="#69add5" if obs_or_var == "obs" else "#fa694a",
                )
            else:
                scax = ax.scatter(
                    x,
                    y,
                    z,
                    s=point_size,
                    c="#69add5" if obs_or_var == "obs" else "#fa694a",
                )
        else:
            if not mode3d:
                scax = ax.scatter(
                    x,
                    y,
                    s=point_size,
                    c=sample_colors,
                    cmap=colormap,
                    norm=matplotlib.colors.Normalize(vmin=color_min, vmax=color_max),
                )
            else:
                scax = ax.scatter(
                    x,
                    y,
                    z,
                    s=point_size,
                    c=sample_colors,
                    cmap=colormap,
                    norm=matplotlib.colors.Normalize(vmin=color_min, vmax=color_max),
                )
            if colormap == "viridis":
                cbar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(
                        norm=matplotlib.colors.Normalize(
                            vmin=color_min, vmax=color_max
                        ),
                        cmap=colormap,
                    ),
                    ax=ax,
                )
                cbar.formatter.set_scientific(False)
                cbar.formatter.set_useOffset(False)

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
            idx_from_ind = ind["ind"][0]
            if obs_or_var == "obs":
                if (
                    "all_labels" in adata.uns
                    and "labels" in adata.obs
                    and function_plot_
                ):
                    text = "{}".format(adata.obs_names[subset_indices[idx_from_ind]])
                else:
                    if sample_colors is None:
                        if subset_indices is None:
                            text = "{}".format(adata.obs_names[idx_from_ind])
                        else:
                            text = "{}".format(
                                adata.obs_names[subset_indices[idx_from_ind]]
                            )
                    else:
                        if subset_indices is None:
                            text = "{}: {}".format(
                                adata.obs_names[idx_from_ind],
                                str(adata.obs["labels"][idx_from_ind]),
                            )
                        else:
                            text = "{}: {}".format(
                                adata.obs_names[subset_indices[idx_from_ind]],
                                str(adata.obs["labels"][subset_indices[idx_from_ind]]),
                            )
            else:
                if subset_indices is None:
                    text = "{}".format(adata.var_names[idx_from_ind])
                else:
                    text = "{}".format(adata.var_names[subset_indices[idx_from_ind]])
            ann.set_text(text)
            print(text)

        fig.canvas.mpl_connect(
            "motion_notify_event",
            lambda event: _hover(event, fig, ax, ann, scax, update_annot),
        )

        if (
            obs_or_var == "obs"
            and "all_labels" in adata.uns
            and "labels" in adata.obs
            and not colormap == "viridis"
        ):

            def lp(i):
                val_ = adata.uns["all_labels"][i]
                if not mode3d:
                    return plt.plot(
                        [],
                        color=scax.cmap(scax.norm(annot_colors[val_])),
                        ms=5,
                        mec="none",
                        label=val_,
                        ls="",
                        marker="o",
                    )[0]
                else:
                    return plt.plot(
                        [],
                        [],
                        color=scax.cmap(scax.norm(annot_colors[val_])),
                        ms=5,
                        mec="none",
                        label=val_,
                        ls="",
                        marker="o",
                    )[0]

            handles = [lp(i) for i in range(len(adata.uns["all_labels"]))]
            plt.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.0, 0.0))
            plt.subplots_adjust(right=0.75)

        if set_xticks is not None:
            plt.xticks(set_xticks, set_xticks_text)
        if not mode3d:
            if xlog_scale:
                plt.xscale("log")
            if ylog_scale:
                plt.yscale("log")
        else:
            if xlog_scale:
                ax.set_xscale("log")
            if ylog_scale:
                ax.set_yscale("log")
            if zlog_scale:
                ax.set_zscale("log")
        plt.title(title)
        if not mode3d:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        if output_file:
            plt.savefig(output_file, dpi=200)
        else:
            plt.show()

    ####################################################################################
    # Bokeh
    elif global_xomx_extension_bokeh_or_matplotlib == "bokeh":
        assert not mode3d, "The extension must be matplotlib for 3d plots."
        # import holoviews as hv  # lazy import
        import bokeh.models  # lazy import
        import bokeh.io  # lazy import
        import bokeh.layouts  # lazy import
        from bokeh.models import HoverTool  # lazy import
        import bokeh.plotting as bkp  # lazy import

        if subset_indices is None:
            tmp_df = adata.obs if obs_or_var == "obs" else adata.var
        else:
            tmp_df = (
                adata.obs.iloc[subset_indices].copy()
                if obs_or_var == "obs"
                else adata.var.iloc[subset_indices].copy()
            )

        random_id = "".join(np.random.choice(list(string.ascii_letters), 10)) + "_"
        tooltips = [("name", "@{" + random_id + "name}")] + [
            (key, "@{" + key + "}") for key in tmp_df.keys()
        ]
        tmp_df[random_id + "x_" + xlabel] = x
        tmp_df[random_id + "y_" + ylabel] = y
        tooltips += [
            ("x:" + xlabel, "@{" + random_id + "x_" + xlabel + "}"),
            ("y:" + ylabel, "@{" + random_id + "y_" + ylabel + "}"),
        ]

        if subset_indices is None:
            tmp_df[random_id + "name"] = (
                adata.obs_names if obs_or_var == "obs" else adata.var_names
            )
        else:
            tmp_df[random_id + "name"] = (
                adata.obs_names[subset_indices]
                if obs_or_var == "obs"
                else adata.var_names[subset_indices]
            )
        if sample_colors is not None:
            tmp_df[random_id + "colors"] = sample_colors
            tmp_cmap = colormap
        else:
            tmp_df[random_id + "colors"] = 0.5
            tmp_cmap = "blues" if obs_or_var == "obs" else "reds"
        if (
            sample_colors is not None
            and obs_or_var == "obs"
            and "all_labels" in adata.uns
            and "labels" in adata.obs
            and not tmp_cmap == "viridis"
        ):
            # Is this all and only the nipy_spectral cases?
            new_tmp_cmap = {}
            for i, lbl in enumerate(adata.uns["all_labels"]):
                new_tmp_cmap[lbl] = matplotlib.colors.rgb2hex(
                    plt.get_cmap(tmp_cmap)(
                        adata.uns["label_colors"][i]
                        if "label_colors" in adata.uns
                        else annot_colors[lbl]
                    )
                )
            tmp_cmap = new_tmp_cmap

        # hv.extension("bokeh")
        # points = hv.Points(
        #     tmp_df,
        #     [random_id + "x_" + xlabel, random_id + "y_" + ylabel],
        #     list(tmp_df.keys()),
        # )

        bokeh_data = bokeh.models.ColumnDataSource(
            tmp_df.loc[:, random_id + "name"].to_frame()
        )

        hover = HoverTool(tooltips=tooltips)
        div = bokeh.models.Div(width=width, height=50, height_policy="fixed")
        cb = bokeh.models.CustomJS(
            args=dict(
                hvr=hover, div=div, source=bokeh_data.data, col_name=random_id + "name"
            ),
            code="""
                if (cb_data['index'].indices.length > 0) {
                    const line_list = [];
                    for (let i = 0; i<cb_data['index'].indices.length; i++) {
                        var line = "<b>"
                        line += source[col_name][cb_data['index'].indices[i]] + "</b>"
                        line_list.push(line)
                    }
                    div.text = line_list.join(" --- ");
                }
            """,
        )
        hover.callback = cb  # callback whenever the HoverTool function is called

        # points.opts(
        #     tools=[hover],
        #     color="labels"
        #     if (
        #         obs_or_var == "obs"
        #         and "all_labels" in adata.uns
        #         and "labels" in adata.obs
        #         and not tmp_cmap == "viridis"
        #     )
        #     else random_id + "colors",
        #     cmap=tmp_cmap,
        #     size=point_size,
        #     width=width,
        #     height=height,
        #     show_grid=False,
        #     title=title,
        #     xlabel=xlabel,
        #     ylabel=ylabel,
        #     logx=xlog_scale,
        #     logy=ylog_scale,
        #     colorbar=True if tmp_cmap == "viridis" else False,
        #     clim=(color_min, color_max),
        # )
        # if set_xticks is not None:
        #     points.opts(
        #         xticks=list(zip([float(x) for x in set_xticks], set_xticks_text))
        #     )
        # hv_plot = points

        data_dict = {}
        for k in tmp_df.keys():
            data_dict[k] = [None]
        data_dict[random_id + "x_" + xlabel] = [np.inf]
        data_dict[random_id + "y_" + ylabel] = [np.inf]
        new_source = bokeh.models.ColumnDataSource(data=data_dict)
        points_bokeh_plot = bkp.figure(
            width=width,
            height=height,
            title=title,
            x_axis_type="log" if xlog_scale else "linear",
            y_axis_type="log" if ylog_scale else "linear",
        )
        points_bokeh_plot.xgrid.grid_line_color = None
        points_bokeh_plot.ygrid.grid_line_color = None
        points_bokeh_plot.xaxis.axis_label = xlabel
        points_bokeh_plot.yaxis.axis_label = ylabel
        if set_xticks is not None:
            # we should use float(..) instead of int(..) but with float(..) the labels
            # are not overridden
            points_bokeh_plot.xaxis.ticker = [int(x) for x in set_xticks]
            points_bokeh_plot.xaxis.major_label_overrides = {
                int(set_xticks[i]): set_xticks_text[i] for i in range(len(set_xticks))
            }

        if isinstance(tmp_cmap, dict):
            color_mapper = bokeh.models.CategoricalColorMapper(
                factors=list(tmp_cmap.keys()), palette=list(tmp_cmap.values())
            )
        if tmp_cmap == "blues":
            color_arg = "#69add5"
        elif tmp_cmap == "reds":
            color_arg = "#fa694a"
        elif tmp_cmap == "viridis":
            color_mapper = bokeh.models.LinearColorMapper(
                palette="Viridis256", low=color_min, high=color_max
            )
            color_arg = {"field": random_id + "colors", "transform": color_mapper}
        else:
            assert isinstance(tmp_cmap, dict)
            color_mapper = bokeh.models.CategoricalColorMapper(
                factors=list(tmp_cmap.keys()), palette=list(tmp_cmap.values())
            )
            color_arg = {
                "field": "labels"
                if (
                    obs_or_var == "obs"
                    and "all_labels" in adata.uns
                    and "labels" in adata.obs
                )
                else random_id + "colors",
                "transform": color_mapper,
            }

        if (
            obs_or_var == "obs"
            and "all_labels" in adata.uns
            and "labels" in adata.obs
            and not tmp_cmap == "viridis"
        ):
            points_bokeh_plot.scatter(
                random_id + "x_" + xlabel,
                random_id + "y_" + ylabel,
                source=tmp_df,
                fill_alpha=1.0,
                size=point_size,
                color=color_arg,
                legend_field="labels",
            )
        else:
            points_bokeh_plot.scatter(
                random_id + "x_" + xlabel,
                random_id + "y_" + ylabel,
                source=tmp_df,
                fill_alpha=1.0,
                size=point_size,
                color=color_arg,
            )
        if tmp_cmap == "viridis":
            color_bar = bokeh.models.ColorBar(
                color_mapper=color_mapper,
                # label_standoff=12,
                # border_line_color=None,
                location=(0, 0),
            )
            points_bokeh_plot.add_layout(color_bar, "right")
        points_bokeh_plot.scatter(
            random_id + "x_" + xlabel,
            random_id + "y_" + ylabel,
            source=new_source,
            size=max(point_size * 2, 10),
            marker="circle",
            color="orange",
            line_color="black",
            line_width=2,
        )
        points_bokeh_plot.add_tools(hover)
        offset_text = bokeh.models.TextInput(
            value="", title="Search:", name="texty", width=width
        )
        thecallback = bokeh.models.CustomJS(
            args=dict(
                source=new_source,
                main_source=bokeh.models.ColumnDataSource(tmp_df),
                dict_index={k: v for v, k in enumerate(tmp_df.index)},
                ot=offset_text,
            ),
            # os=offset_slider),
            code="""
                const data = source.data;
                // everything below here is unaltered
                for (const key in data) {
                    const valref = data[key];
                    valref[0] = main_source.data[key][dict_index[ot.value]];
                }
                source.change.emit();
            """,
        )
        offset_text.js_on_change("value", thecallback)

        if output_file:
            bkp.output_file(output_file)
            bokeh.io.save(
                bokeh.layouts.column(
                    _custom_legend(points_bokeh_plot), offset_text, div
                )
            )
            # hv.save(hv_plot, output_file, fmt="html")
        else:
            bokeh.io.show(
                bokeh.layouts.column(
                    _custom_legend(points_bokeh_plot), offset_text, div
                )
            )
            # bokeh.io.show(
            #     bokeh.layouts.column(_custom_legend(hv.render(hv_plot)), div))
        del tmp_df[random_id + "name"]
        del tmp_df[random_id + "colors"]
        del tmp_df[random_id + "x_" + xlabel]
        del tmp_df[random_id + "y_" + ylabel]
    ####################################################################################
    else:
        raise ValueError(
            'Execute xomx.pl.extension("bokeh") or xomx.pl.extension("matplotlib")'
        )


def scatter(
    adata,
    func1=_identity_func,
    func2=_identity_func,
    obs_or_var: str = "obs",
    *,
    pointsize: int = 3,
    xlog_scale: bool = False,
    ylog_scale: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    subset_indices=None,
    equal_size=False,
    output_file: Optional[str] = None,
    width=900,
    height=600,
):
    scatter2d_and_3d(
        adata,
        func1,
        func2,
        None,
        obs_or_var,
        pointsize=pointsize,
        xlog_scale=xlog_scale,
        ylog_scale=ylog_scale,
        zlog_scale=False,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel="",
        title=title,
        subset_indices=subset_indices,
        equal_size=equal_size,
        output_file=output_file,
        width=width,
        height=height,
    )


def plot(
    adata,
    func=_identity_func,
    obs_or_var="obs",
    *,
    pointsize: int = 5,
    ylog_scale=False,
    xlabel="",
    ylabel="",
    title="",
    subset_indices=None,
    equal_size: bool = False,
    output_file: Optional[str] = None,
    width: int = 900,
    height: int = 600,
):
    """Plots the value of a function on every sample or every feature, depending
    on the value of obs_or_var which must be either "obs" or "var"
    (the function must take indices in input)"""
    return scatter(
        adata,
        _identity_func,
        func,
        obs_or_var,
        pointsize=pointsize,
        xlog_scale=False,
        ylog_scale=ylog_scale,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        subset_indices=subset_indices,
        equal_size=equal_size,
        output_file=output_file,
        width=width,
        height=height,
    )


def plot_var(
    adata,
    features=None,
    *,
    pointsize: int = 5,
    xlabel="",
    ylabel="",
    title="",
    subset_indices=None,
    equal_size=False,
    output_file: Optional[str] = None,
    width=900,
    height=600,
):
    """ """
    global global_xomx_extension_bokeh_or_matplotlib
    if type(features) == str or type(features) == np.str_ or type(features) == int:
        idx = features
        if type(idx) == str or type(idx) == np.str_:
            assert "var_indices" in adata.uns, (
                "Please compute and store the dictionary of feature indices with: "
                'adata.uns["var_indices"] = xomx.tl.var_indices(adata)'
            )
            idx = adata.uns["var_indices"][idx]
        plot(
            adata,
            lambda i: adata.X[i, idx],
            "obs",
            pointsize=pointsize,
            ylog_scale=False,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            subset_indices=subset_indices,
            equal_size=equal_size,
            output_file=output_file,
            width=width,
            height=height,
        )
    else:
        if features is None:
            features = np.arange(adata.n_vars)
        if subset_indices is None:
            xsize = adata.n_obs
        else:
            xsize = len(subset_indices)
        ysize = len(features)
        set_xticks = None
        set_xticks_text = None
        feature_indices_list_ = []
        for idx in features:
            if type(idx) == str or type(idx) == np.str_:
                assert "var_indices" in adata.uns
                idx = adata.uns["var_indices"][idx]
            feature_indices_list_.append(idx)
        if "all_labels" not in adata.uns:
            plot_array = np.empty((len(features), xsize))
            for k, idx in enumerate(feature_indices_list_):
                plot_array[k, :] = [adata.X[i, idx] for i in range(adata.n_obs)]
        else:
            (
                list_samples,
                set_xticks,
                set_xticks_text,
                boundaries,
            ) = _samples_by_labels(
                adata,
                sort_annot=True,
                subset_indices=subset_indices,
                equal_size=equal_size,
            )
            plot_array = np.empty((len(features), len(list_samples)))
            for k, idx in enumerate(feature_indices_list_):
                plot_array[k, :] = [adata.X[i, idx] for i in list_samples]
        xsize = plot_array.shape[1]

        if global_xomx_extension_bokeh_or_matplotlib == "matplotlib":
            rcParams["figure.dpi"] = 100
            rcParams["figure.figsize"] = (
                width / rcParams["figure.dpi"],
                height / rcParams["figure.dpi"],
            )
            fig, ax = plt.subplots()
            im = ax.imshow(plot_array, extent=[0, xsize, 0, ysize], aspect="auto")
            if set_xticks is not None:
                plt.xticks(set_xticks, set_xticks_text)
            plt.yticks(
                np.arange(ysize) + 0.5,
                [adata.var_names[i] for i in feature_indices_list_][::-1],
            )
            plt.tick_params(axis="both", which="both", length=0)
            plt.colorbar(im)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if output_file:
                plt.savefig(output_file, dpi=200)
            else:
                plt.show()

        ################################################################################
        # Bokeh
        elif global_xomx_extension_bokeh_or_matplotlib == "bokeh":
            import holoviews as hv  # lazy import
            import bokeh.models  # lazy import
            import bokeh.io  # lazy import
            import bokeh.layouts  # lazy import

            # import bokeh.plotting as bkp  # lazy import

            hv.extension("bokeh")
            bounds = (0, 0, xsize, ysize)  # Coord system: (left, bottom, right, top)
            img = hv.Image(plot_array, bounds=bounds)
            img.opts(
                cmap="viridis",
                width=width,
                height=height,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                colorbar=True,
            )
            if set_xticks is not None:
                img.opts(
                    xticks=list(zip([float(x) for x in set_xticks], set_xticks_text))
                )
            img.opts(
                yticks=list(
                    zip(
                        [float(y) for y in np.arange(ysize) + 0.5],
                        [adata.var_names[i] for i in feature_indices_list_][::-1],
                    )
                )
            )
            hv_plot = img
            if output_file:
                hv.save(hv_plot, output_file, fmt="html")
            else:
                bokeh.io.show(_custom_legend(hv.render(hv_plot)))

        ################################################################################
        else:
            raise ValueError(
                'Execute xomx.pl.extension("bokeh") or xomx.pl.extension("matplotlib")'
            )


def plot_2d_obsm(
    adata,
    obsm_key,
    color_var=None,
    *,
    pointsize: int = 3,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    subset_indices=None,
    output_file: Optional[str] = None,
    width: int = 900,
    height: int = 600,
):
    def embedding_x(j):
        return adata.obsm[obsm_key][j, 0]

    def embedding_y(j):
        return adata.obsm[obsm_key][j, 1]

    if color_var is not None:
        if "colors" in adata.obs:
            adata.obs.rename(
                {"colors": "xomx_temporary_colors"}, axis="columns", inplace=True
            )
        assert (
            color_var in adata.var_names
        ), "the color_var input must be in adata.var_names"
        adata.obs["colors"] = np.array(_to_dense(adata[:, color_var].X))

    scatter(
        adata,
        embedding_x,
        embedding_y,
        "obs",
        pointsize=pointsize,
        xlog_scale=False,
        ylog_scale=False,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        subset_indices=subset_indices,
        output_file=output_file,
        width=width,
        height=height,
    )

    if color_var is not None:
        del adata.obs["colors"]
        if "xomx_temporary_colors" in adata.obs:
            adata.obs.rename(
                {"xomx_temporary_colors": "colors"}, axis="columns", inplace=True
            )


def plot_3d_obsm(
    adata,
    obsm_key,
    color_var=None,
    *,
    pointsize: int = 3,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    title: str = "",
    subset_indices=None,
    output_file: Optional[str] = None,
    width: int = 900,
    height: int = 600,
):
    def embedding_x(j):
        return adata.obsm[obsm_key][j, 0]

    def embedding_y(j):
        return adata.obsm[obsm_key][j, 1]

    def embedding_z(j):
        return adata.obsm[obsm_key][j, 2]

    if color_var is not None:
        if "colors" in adata.obs:
            adata.obs.rename(
                {"colors": "xomx_temporary_colors"}, axis="columns", inplace=True
            )
        assert (
            color_var in adata.var_names
        ), "the color_var input must be in adata.var_names"
        adata.obs["colors"] = np.array(_to_dense(adata[:, color_var].X))

    scatter2d_and_3d(
        adata,
        embedding_x,
        embedding_y,
        embedding_z,
        "obs",
        pointsize=pointsize,
        xlog_scale=False,
        ylog_scale=False,
        zlog_scale=False,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        title=title,
        subset_indices=subset_indices,
        output_file=output_file,
        width=width,
        height=height,
    )

    if color_var is not None:
        del adata.obs["colors"]
        if "xomx_temporary_colors" in adata.obs:
            adata.obs.rename(
                {"xomx_temporary_colors": "colors"}, axis="columns", inplace=True
            )


def plot_2d_varm(
    adata,
    varm_key,
    color_obs=None,
    *,
    pointsize: int = 3,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    subset_indices=None,
    output_file: Optional[str] = None,
    width: int = 900,
    height: int = 600,
):
    def embedding_x(j):
        return adata.varm[varm_key][j, 0]

    def embedding_y(j):
        return adata.varm[varm_key][j, 1]

    if color_obs is not None:
        if "colors" in adata.var:
            adata.var.rename(
                {"colors": "xomx_temporary_colors"}, axis="columns", inplace=True
            )
        assert (
            color_obs in adata.obs_names
        ), "the color_obs input must be in adata.obs_names"
        adata.var["colors"] = np.array(_to_dense(adata[color_obs, :].X))

    scatter(
        adata,
        embedding_x,
        embedding_y,
        "var",
        pointsize=pointsize,
        xlog_scale=False,
        ylog_scale=False,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        subset_indices=subset_indices,
        output_file=output_file,
        width=width,
        height=height,
    )

    if color_obs is not None:
        del adata.var["colors"]
        if "xomx_temporary_colors" in adata.var:
            adata.var.rename(
                {"xomx_temporary_colors": "colors"}, axis="columns", inplace=True
            )


def plot_3d_varm(
    adata,
    varm_key,
    color_obs=None,
    *,
    pointsize: int = 3,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    title: str = "",
    subset_indices=None,
    output_file: Optional[str] = None,
    width: int = 900,
    height: int = 600,
):
    def embedding_x(j):
        return adata.varm[varm_key][j, 0]

    def embedding_y(j):
        return adata.varm[varm_key][j, 1]

    def embedding_z(j):
        return adata.varm[varm_key][j, 2]

    if color_obs is not None:
        if "colors" in adata.var:
            adata.var.rename(
                {"colors": "xomx_temporary_colors"}, axis="columns", inplace=True
            )
        assert (
            color_obs in adata.obs_names
        ), "the color_obs input must be in adata.obs_names"
        adata.var["colors"] = np.array(_to_dense(adata[color_obs, :].X))

    scatter2d_and_3d(
        adata,
        embedding_x,
        embedding_y,
        embedding_z,
        "var",
        pointsize=pointsize,
        xlog_scale=False,
        ylog_scale=False,
        zlog_scale=False,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        title=title,
        subset_indices=subset_indices,
        output_file=output_file,
        width=width,
        height=height,
    )

    if color_obs is not None:
        del adata.var["colors"]
        if "xomx_temporary_colors" in adata.var:
            adata.var.rename(
                {"xomx_temporary_colors": "colors"}, axis="columns", inplace=True
            )

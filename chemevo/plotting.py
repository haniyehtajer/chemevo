"""Shared plotting primitives for chemevo model output."""
import numpy as np
import matplotlib.pyplot as plt
from chemevo import plotstyle

plotstyle.use()

from chemevo.mn_fe_lines import mn_fe_model_line


def _normalize_models(models):
    """
    Turn `models` into a list of (label, dataframe, scatter_kwargs) triples,
    no matter whether the caller passed a dict or a list.
    """
    normalized = []

    if isinstance(models, dict):
        for label, df in models.items():
            normalized.append((label, df, {}))
        return normalized

    for entry in models:
        if len(entry) == 2:
            label, df = entry
            kwargs = {}
        else:
            label, df, kwargs = entry
        normalized.append((label, df, kwargs))

    return normalized


def _make_axes_grid(n_bins, ncols, figsize):
    """Create a new figure with one Axes per bin, laid out in a grid."""
    if ncols is None:
        ncols = n_bins
    nrows = int(np.ceil(n_bins / ncols))

    if figsize is None:
        # Matches the panel size already used for multi-panel plots in
        # VICE/examples (figsize=(25, 8) for 5 panels) - a shorter default
        # crams the same point-sized tick/axis labels into less room and
        # looks small by comparison, even though the font size is identical.
        panel_width = 5
        panel_height = 8
        figsize = (panel_width * ncols, panel_height * nrows)

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = axes_grid.flatten()
    return fig, axes_flat


def plot_mn_fe_vs_fe_mg(models, lines_df, fe_mg_col="fe_mg", mn_fe_col="mn_fe",
                         mg_h_bin_col="mg_h_bin", line_x_range=(-0.3, 0.0),
                         xlim=(-0.6, 0.25), ylim=(-0.8, 0.3),
                         ncols=None, figsize=None, axes=None,
                         legend_on = 0, **scatter_kwargs):
    """
    Scatter [Fe/Mg] vs [Mn/Fe] for one or more models, one panel per [Mg/H]
    bin (taken from `lines_df`), with the data-derived reference line
    (chemevo.mn_fe_lines) overlaid in each panel.

    Parameters
    ----------
    models : dict[str, pandas.DataFrame] or list of (label, df) / (label, df, scatter_kwargs)
        Which model(s) to plot and what to label them. Each DataFrame needs
        `fe_mg_col`, `mn_fe_col`, and `mg_h_bin_col` columns. Pass an empty
        dict/list to draw just the reference lines. A model's own
        scatter_kwargs (3rd tuple element) override `color`/`edgecolor` and
        `**scatter_kwargs` below, for that model only.
    lines_df : pandas.DataFrame
        From `chemevo.mn_fe_lines.load_mn_fe_lines()` / `compute_mn_fe_lines()`.
        Defines which [Mg/H] bins get a panel and the line drawn in each.
    line_x_range : tuple
        x-range to draw the reference line over, in each panel.
    xlim, ylim : tuple or None
        Axis limits applied to every panel. Pass None to leave axes
        auto-scaled instead.
    ncols : int, optional
        Panels per row (default: all bins in a single row).
    figsize : tuple, optional
        Overall figure size in inches (default scales with panel count).
    axes : array-like of Axes, optional
        Draw into existing axes instead of creating a new figure/grid.
    legend_panel : {"first", "last", "all", None}
        Which panel(s) get a legend.
    color, edgecolor : optional
        Default point/edge color applied to every model's scatter, unless a
        model overrides it via its own scatter_kwargs.
    **scatter_kwargs
        Any other keyword forwarded to every `Axes.scatter()` call (e.g.
        `s=`, `alpha=`, `marker=`), as a default for every model.

    Returns
    -------
    fig, axes : the figure and the (n_bins,) array of Axes actually used.
    """
    models = _normalize_models(models)
    mg_h_bins = lines_df["mg_h_bin_center"].to_numpy()
    n_bins = len(mg_h_bins)

    if axes is None:
        fig, axes_flat = _make_axes_grid(n_bins, ncols, figsize)
    else:
        axes_flat = np.atleast_1d(axes).flatten()
        fig = axes_flat[0].get_figure()

    # Points along the x-axis at which to draw the reference line.
    line_x_low, line_x_high = line_x_range
    line_x = np.linspace(line_x_low, line_x_high, 100)

    # Scatter settings shared by every model, unless a model overrides them.
    default_kwargs = {"s": 15, "zorder": 5}
    default_kwargs.update(scatter_kwargs)

    for panel_index, mg_h_bin in enumerate(mg_h_bins):
        ax = axes_flat[panel_index]

        line_y = mn_fe_model_line(lines_df, mg_h_bin, line_x)
        ax.plot(line_x, line_y, color="black", zorder=20, linewidth=3)

        for label, df, model_kwargs in models:
            in_this_bin = np.isclose(df[mg_h_bin_col], mg_h_bin)
            subset = df[in_this_bin]

            kwargs = default_kwargs.copy()
            kwargs.update(model_kwargs)
            ax.scatter(subset[fe_mg_col], subset[mn_fe_col], label=label, **kwargs)

        ax.set_xlabel("[Fe/Mg]")
        ax.set_title(f"[Mg/H] = {mg_h_bin:.1f}")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        if panel_index == 0:
            ax.set_ylabel("[Mn/Fe]")

        if legend_on == 1:
            if panel_index == 4:
                # loc="upper left" + bbox_to_anchor=(1.02, 1) means: put the
                # legend's upper-left corner just past this axes' right edge
                # (x=1.02, in this axes' own 0-1 fraction coordinates) at
                # its top (y=1) - i.e. outside the plot, on the right side.
                ax.legend(fontsize=18, markerscale=2, loc="upper left", bbox_to_anchor=(1.02, 1))

    # Hide any leftover panels (e.g. axes was passed in with more slots
    # than there are bins).
    unused_axes = axes_flat[n_bins:]
    for ax in unused_axes:
        ax.set_visible(False)

    fig.suptitle(label, fontsize=30)
    fig.tight_layout()
    return fig, axes_flat[:n_bins]

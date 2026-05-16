"""
Helper functions for matplotlib plotting defaults.

This module sets default rcParams for fontsize and grid styling
to ensure consistent appearance across all plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from utils.helpers.fitting_helper import rxr


def set_plotting_defaults():
    """
    Set matplotlib rcParams for fontsize and grid defaults.

    This function configures:
    - Font sizes for labels, ticks, legend, and titles
    - Grid appearance (alpha, linestyle, linewidth)
    - General figure aesthetics

    Examples
    --------
    >>> from src.utils.helpers.plotting_helper import set_plotting_defaults
    >>> set_plotting_defaults()
    >>> plt.plot([1, 2, 3], [1, 4, 9])
    >>> plt.show()
    """
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 10,
            "font.family": "sans-serif",  # non serif font for all text
            "mathtext.fontset": "dejavusans",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
        }
    )


def reset_plotting_defaults():
    """
    Reset matplotlib rcParams to default values.

    Examples
    --------
    >>> from src.utils.helpers.plotting_helper import reset_plotting_defaults
    >>> reset_plotting_defaults()
    """
    plt.rcParams.update(plt.rcParamsDefault())  # type: ignore


#  Wrapper function for the model.plot function to let you get just beta


def plot_vertically_offset_energies(
    ax,
    energies,
    objectives_list,
    loaded_data,
    title,
    offset_scale=-1.5,
    line_styles=None,
    line_colors=None,
):
    """
    Helper function to plot multiple energies with vertical offsets.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    energies : list
        List of energies to plot.
    objectives_list : list or GlobalObjective
        A list of objective sets to plot, or a single objective set.
        Each objective set should have an `objectives` attribute.
    loaded_data : dict
        Dictionary of loaded data keyed by energy string.
    title : str
        Title for the plot.
    offset_scale : float
        Scale factor for vertical offsets (powers of 10).
    line_styles : list, optional
        List of line styles to cycle through. Defaults to ["-", "--", "-.", ":"].
    line_colors : list, optional
        List of colors to cycle through. Defaults to ["k", "grey", "darkgrey",
        "lightgrey"].
    """

    # Handle single objective or list of objectives
    if not isinstance(objectives_list, list):
        objectives_list = [objectives_list]

    # Default line styles and colors
    if line_styles is None:
        line_styles = ["-", "--", "-.", ":"]
    if line_colors is None:
        # Start with black, then shift to greys
        line_colors = ["k", "grey", "darkgrey", "lightgrey"]

    # Build objective dictionaries for each objective set
    objectives_dicts = []
    for obj_set in objectives_list:
        obj_dict = {obj.model.energy: obj for obj in obj_set.objectives}
        objectives_dicts.append(obj_dict)

    # Create color map for data points
    cmap = plt.colormaps.get_cmap("tab20")
    xdata = np.linspace(0.0005, 0.27, 1000)

    for idx, E in enumerate(energies):
        # Check if energy exists in at least one objective set
        if not any(E in obj_dict for obj_dict in objectives_dicts):
            continue

        data = loaded_data[str(E)]

        # Vertical offset (powers of 10)
        offset = 10 ** (offset_scale * idx)
        color = cmap(idx)

        # Plot data points (smaller markers)
        for pol, marker_style in [("s", "o"), ("p", "s")]:
            pol_data = getattr(data, pol)
            ax.errorbar(
                pol_data.x,
                pol_data.y * offset,
                yerr=pol_data.y_err * offset,
                lw=0,
                color=color,
                capsize=0.5,
                marker="o",
                elinewidth=0.5,
                markersize=0.5,
                markerfacecolor=color,
                markevery=2,
                errorevery=2,
            )

        # Plot fits for each objective set
        for obj_idx, obj_dict in enumerate(objectives_dicts):
            if E not in obj_dict:
                continue

            obj = obj_dict[E]
            ls = line_styles[obj_idx % len(line_styles)]
            lc = line_colors[obj_idx % len(line_colors)]

            for pol in ["s", "p"]:
                y_fit = rxr(xdata, obj.model, pol)
                ax.plot(xdata, y_fit * offset, color=lc, lw=0.5, ls=ls, zorder=-10)

    # Configure axes
    ax.set_yscale("log")
    ax.set_ylabel(r"Reflectivity (Vertically Offset)")
    ax.set_xlim(0.002, 0.07)
    ax.minorticks_on()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(offset * pol_data.y.min(), 1)
    ax.set_xticks([0.03, 0.06])
    ax.set_xticklabels([r"$3$", r"$6$"])


def plot_all_results(
    objectives_list,
    loaded_data,
    save_path: Path | str | None = None,
    line_styles=None,
    line_colors=None,
    legend_labels=None,
):
    """
    Plot all results in a single figure matching the target layout:
    - Top: 250 eV and 283.7 eV spanning two columns
    - Bottom left: non-resonant energies (between 250 and 283.7 eV)
    - Bottom right: resonant energies (greater than 283.7 eV)

    Parameters
    ----------
    objectives_list : list or GlobalObjective
        A list of objective sets to plot, or a single objective set.
        Each objective set should have an `objectives` attribute.
    loaded_data : dict
        Dictionary of loaded data keyed by energy string.
    save_path : Path or str, optional
        Path to save the figure. If None, displays the figure.
    line_styles : list, optional
        List of line styles to cycle through. Defaults to ["-", "--", "-.", ":"].
    line_colors : list, optional
        List of colors to cycle through. Defaults to ["k", "grey",
        "darkgrey", "lightgrey"].
    legend_labels : list, optional
        Labels for each objective set in the legend.
    """
    import matplotlib.gridspec as gs

    # Handle single objective or list of objectives
    if not isinstance(objectives_list, list):
        objectives_list = [objectives_list]

    # Default line styles and colors
    if line_styles is None:
        line_styles = ["-", "--", "-.", ":"]
    if line_colors is None:
        line_colors = ["k", "grey", "darkgrey", "lightgrey"]
    if legend_labels is None:
        legend_labels = [f"Model {i + 1}" for i in range(len(objectives_list))]

    # Build objective dictionaries for each objective set
    objectives_dicts = []
    for obj_set in objectives_list:
        obj_dict = {obj.model.energy: obj for obj in obj_set.objectives}
        objectives_dicts.append(obj_dict)

    # Create figure and gridspec (single column)
    fig = plt.figure(figsize=(2.5, 3.5))
    grid = gs.GridSpec(
        2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.05, wspace=0.01
    )

    # Top panel spans both columns
    ax_top = fig.add_subplot(grid[0, :])

    # Bottom panels
    ax_bottom_left = fig.add_subplot(grid[1, 0])
    ax_bottom_right = fig.add_subplot(grid[1, 1])

    # ===== TOP PANEL: 250 eV and 283.7 eV =====
    comparison_energies = np.array([250.0, 283.7])

    cols = ["C0", "C1"]
    for i, (E, col) in enumerate(zip(comparison_energies, cols)):
        data = loaded_data[str(E)]
        scale = 10 ** (-3 * i)

        def _clip_low(arr):
            return np.clip(arr, 1e-12, None)

        # Plot data points with error bands
        for pol, marker_style in [("s", "o"), ("p", "s")]:
            pol_data = getattr(data, pol)
            ax_top.errorbar(
                pol_data.x,
                pol_data.y * scale,
                yerr=pol_data.y_err * scale,
                lw=0,
                color=col,
                capsize=0.5,
                marker="o",
                elinewidth=0.5,
                markersize=0.5,
                markerfacecolor=col,
            )

        # Plot fits for each objective set
        for obj_idx, obj_dict in enumerate(objectives_dicts):
            if E not in obj_dict:
                continue

            obj = obj_dict[E]
            ls = line_styles[obj_idx % len(line_styles)]
            lc = line_colors[obj_idx % len(line_colors)]

            for pol in ["s", "p"]:
                pol_data = getattr(data, pol)
                y_fit = rxr(pol_data.x, obj.model, pol)
                ax_top.plot(
                    pol_data.x, y_fit * scale, color=lc, lw=0.5, ls=ls, zorder=10
                )

    # Configure top panel
    ax_top.set_xlim(0.002, 0.27)
    ax_top.set_xticklabels([])
    ax_top.set_yscale("log")
    ax_top.set_yticklabels([])
    ax_top.set_xticks([0.002, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24])
    ax_top.minorticks_on()
    ax_top.grid(True, alpha=0.3)
    ax_top.set_ylim(scale * pol_data.y.min(), 1)

    # Consolidated legend for top panel
    from matplotlib.lines import Line2D

    # Create legend handles for model lines
    legend_handles = []
    for obj_idx, label in enumerate(legend_labels):
        ls = line_styles[obj_idx % len(line_styles)]
        lc = line_colors[obj_idx % len(line_colors)]
        legend_handles.append(Line2D([], [], color=lc, lw=1, ls=ls, label=label))

    # Add energy color indicators
    for E, col in zip(comparison_energies, cols):
        legend_handles.append(
            Line2D(
                [],
                [],
                color=col,
                marker="o",
                markersize=1,
                markerfacecolor=col,
                linestyle="None",
                label=f"{E:.1f} eV",
            )
        )

    # Place consolidated legend
    ax_top.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize="x-small",
        ncols=2,
        handlelength=1.2,
        columnspacing=1.0,
        framealpha=0.9,
    )

    # ===== BOTTOM PANELS: Non-resonant and Resonant Energies =====
    # Get all energies from the first objective set
    all_energies = sorted([obj.model.energy for obj in objectives_list[0].objectives])

    # Split energies
    non_resonant = [e for e in all_energies if 250.0 < e < 283.7]
    resonant = [e for e in all_energies if e > 283.7]

    # Plot non-resonant energies (bottom left)
    plot_vertically_offset_energies(
        ax_bottom_left,
        non_resonant,
        objectives_list,
        loaded_data,
        "Non-Resonant Energies",
        offset_scale=-0.5,
        line_styles=line_styles,
        line_colors=line_colors,
    )

    # Plot resonant energies (bottom right)
    plot_vertically_offset_energies(
        ax_bottom_right,
        resonant,
        objectives_list,
        loaded_data,
        "Resonant Energies",
        offset_scale=-2.5,
        line_styles=line_styles,
        line_colors=line_colors,
    )

    ax_bottom_left.set_ylabel("")
    ax_bottom_left.set_yticklabels([])
    ax_bottom_right.set_ylabel("")
    ax_bottom_right.set_yticklabels([])

    #  Add figure common y-axis label
    fig.text(0.05, 0.5, r"Reflectivity (log scale)", va="center", rotation="vertical")
    #  Add a figure common x-axis label
    fig.text(0.5, 0.05, r"$Q$ ($\AA^{-1}\times 10^3$)", ha="center", va="center")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig

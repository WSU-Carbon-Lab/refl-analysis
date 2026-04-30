import copy

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyref.fitting as fit
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# -------------------------------------------------
# 1.  Helper functions  (as defined above)
# -------------------------------------------------
def chi2(obj):
    # data = np.concatenate([obj.data.s.y, obj.data.p.y])
    # sigma = np.concatenate([obj.data.s.y_err, obj.data.p.y_err])
    # pols = ["s", "p"]
    # model = []
    # model_func = copy.deepcopy(obj.model)
    # for p in pols:
    #     model_func.pol = p
    #     model.append(model_func(obj.data.__getattribute__(p).x))
    # model = np.concatenate(model)
    return obj.chisqr()


def reduced_chi2(obj):
    return chi2(obj) / (obj.npoints - len(obj.varying_parameters()))


def pure_loglikelihood(obj):
    obj_copy = copy.deepcopy(obj)
    obj_copy.anisotropy_weight = 0
    return obj_copy.logl()


def rmsd(obj):
    return np.sqrt(obj.chisqr() / obj.npoints)


def aic(obj):
    return 2 * len(obj.varying_parameters()) - 2 * pure_loglikelihood(obj)


def bic(obj):
    return len(obj.varying_parameters()) * np.log(obj.npoints) - 2 * pure_loglikelihood(
        obj
    )


def plot(  # type: ignore
    obj,
    samples=0,
    model=None,
    ax=None,
    ax_anisotropy=None,
    color_err=("C0", "C1", "C2"),
    color_fit=("C0", "C1", "C2"),
    data_kwargs=None,
    model_kwargs=None,
    show_s=True,
    show_p=True,
    show_anisotropy=True,
) -> tuple[Axes | None, Axes | None]:
    """
    Plot function that includes anisotropy information.

    Parameters
    ----------
    samples : int, optional
        Number of sample curves to plot from MCMC chain
    model : array-like, optional
        Model data to plot
    ax : matplotlib.Axes, optional
        Axes for reflectivity plot
    ax_anisotropy : matplotlib.Axes, optional
        Axes for anisotropy plot
    data_kwargs : dict, optional
        Keyword arguments for data plotting
    model_kwargs : dict, optional
        Keyword arguments for model plotting
    show_s : bool, optional
        Whether to show s-polarization data
    show_p : bool, optional
        Whether to show p-polarization data
    show_anisotropy : bool, optional
        Whether to show anisotropy plot

    Returns
    -------
    tuple
        (ax, ax_anisotropy) - matplotlib axes objects
    """
    import matplotlib.pyplot as plt

    if data_kwargs is None:
        data_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}

    # Set up axes
    if ax is None:
        if show_anisotropy:
            fig, axs = plt.subplots(
                nrows=2,
                sharex=False,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax = axs[0]
            ax_anisotropy = axs[1]
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax_anisotropy = None
    elif ax_anisotropy is None and show_anisotropy:
        # Get the figure from the provided axis
        fig = ax.figure
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
        ax_anisotropy = fig.add_subplot(gs[1], sharex=ax)

    # Check if we have separate s and p polarization data
    has_separate_pol = hasattr(obj.data, "s") and hasattr(obj.data, "p")

    # Plot data
    if has_separate_pol:
        # Plot s-polarization if requested
        if show_s:
            ax.errorbar(
                obj.data.s.x,  # type: ignore
                obj.data.s.y,  # type: ignore
                obj.data.s.y_err,  # type: ignore
                label=f"{obj.data.name} s-pol" if obj.data.name else "s-pol",
                marker="o",
                color=color_err[0],
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
                **data_kwargs,
            )

            # Calculate s-polarization model
            original_pol = obj.model.pol
            obj.model.pol = "s"
            s_model = obj.model(obj.data.s.x)  # type: ignore
            ax.plot(
                obj.data.s.x,  # type: ignore
                s_model,
                color=color_fit[0],
                label="s-pol fit",
                zorder=20,
                **model_kwargs,
            )
            obj.model.pol = original_pol

        # Plot p-polarization if requested
        if show_p:
            ax.errorbar(
                obj.data.p.x,  # type: ignore
                obj.data.p.y,  # type: ignore
                obj.data.p.y_err,  # type: ignore
                label=f"{obj.data.name} p-pol" if obj.data.name else "p-pol",
                marker="o",
                color=color_err[1],
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
                **data_kwargs,
            )

            # Calculate p-polarization model
            original_pol = obj.model.pol
            obj.model.pol = "p"
            p_model = obj.model(obj.data.p.x)  # type: ignore
            ax.plot(
                obj.data.p.x,  # type: ignore
                p_model,
                color=color_fit[1],
                label="p-pol fit",
                zorder=20,
                **model_kwargs,
            )
            obj.model.pol = original_pol
    else:
        # Handle combined data case
        ax.errorbar(
            obj.data.x,
            obj.data.y,
            obj.data.y_err,
            label=obj.data.name,
            marker="o",
            color=color_err[0],
            ms=3,
            lw=0,
            elinewidth=1,
            capsize=1,
            ecolor="k",
            **data_kwargs,
        )

        # Plot combined model
        model = obj.generative()
        _, _, model_transformed = obj._data_transform(model=model)

        if samples > 0:
            # Get sample curves from MCMC chain
            models = []
            for curve in obj._generate_generative_mcmc(ngen=samples):
                _, _, model_t = obj._data_transform(model=curve)
                models.append(model_t)
            models = np.array(models)

            # Show 1-sigma and 2-sigma confidence intervals
            ax.fill_between(
                obj.data.x,
                np.percentile(models, 16, axis=0),  # type: ignore
                np.percentile(models, 84, axis=0),  # type: ignore
                color=color_fit[1],
                alpha=0.5,
            )
            ax.fill_between(
                obj.data.x,
                np.percentile(models, 2.5, axis=0),  # type: ignore
                np.percentile(models, 97.5, axis=0),  # type: ignore
                color=color_fit[1],
                alpha=0.2,
            )

        # Plot the fit
        ax.plot(
            obj.data.x,
            model_transformed,  # type: ignore
            color=color_fit[1],
            label="fit",
            zorder=20,
            **model_kwargs,
        )

    # Plot anisotropy if enabled
    if (
        ax_anisotropy is not None
        and show_anisotropy
        and hasattr(obj.data, "anisotropy")
    ):
        ax_anisotropy.set_ylabel("Anisotropy")

        # Plot anisotropy model
        ax_anisotropy.plot(
            obj.data.anisotropy.x,  # type: ignore
            obj.model.anisotropy(obj.data.anisotropy.x),  # type: ignore
            color=color_fit[2],
            zorder=20,
            label="model",
        )

        # Plot anisotropy data
        ax_anisotropy.plot(
            obj.data.anisotropy.x,  # type: ignore
            obj.data.anisotropy.y,  # type: ignore
            color=color_err[2],
            marker="o",
            markersize=3,
            linestyle="None",
            label="data",
        )

        ax_anisotropy.legend()
        ax_anisotropy.axhline(0, color="k", ls="-", lw=plt.rcParams["axes.linewidth"])
        ax_anisotropy.set_xlabel(r"$q (\AA^{-1})$")

    # Finalize styling
    ax.set_ylabel("Reflectivity")
    ax.set_yscale("log")
    ax.legend()

    return ax, ax_anisotropy


def plot_reflectivity_and_structure(
    global_obj, figsize=(12, 10)
) -> tuple[Figure, Axes | None]:
    """
    Plot reflectivity data and structure profiles for a global objective.

    Parameters:
    -----------
    global_obj : GlobalObjective
        The global objective containing multiple objectives
    stacks : dict
        Dictionary of structure stacks with energy keys
    energy_labels : list, optional
        List of energy labels for plotting. If None, uses stack keys.
    figsize : tuple, optional
        Figure size (width, height)

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    objectives = global_obj.objectives
    energy_labels = [o.model.energy for o in objectives]
    stacks = {str(o.model.energy): o.model.structure for o in objectives}
    n_objectives = len(objectives)

    # Create figure and axes
    fig, ax = plt.subplots(
        nrows=n_objectives,
        ncols=2,
        figsize=figsize,
        gridspec_kw={
            "hspace": 0.25,
            "wspace": 0.15,
            "width_ratios": [2.5, 1],
        },
    )

    # Handle single objective case
    if n_objectives == 1:
        ax = ax.reshape(1, -1)

    # Define colors for consistency
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Plot objectives and structures
    for i in range(n_objectives):
        if i >= len(objectives):
            break

        o = objectives[i]
        color = colors[i % len(colors)]

        # Plot reflectivity data
        plot(
            o,
            ax=ax[i][0],
            show_anisotropy=False,
            color_fit=("k", "k", "k"),
            model_kwargs={"lw": 0.5},
        )
        ax[i][0].set_ylabel(None, fontsize=12, fontweight="bold")
        ax[i][0].set_yscale("log")
        ax[i][0].grid(True, alpha=0.3, linestyle="--")
        ax[i][0].tick_params(direction="in", labelsize=10)

        # Add energy label as text box
        ax[i][0].text(
            0.55,
            0.95,
            energy_labels[i],
            transform=ax[i][0].transAxes,
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
            ),
            verticalalignment="top",
            zorder=10,
        )

        # Custom legend for reflectivity
        if i == 0:
            ax[i][0].legend(
                ["s-pol", "p-pol"],
                loc="upper right",
                frameon=True,
                fancybox=False,
                fontsize=10,
            )
        else:
            if ax[i][0].get_legend():
                ax[i][0].get_legend().remove()

        # Plot structure (SLD profile)
        stack_key = list(stacks.keys())[i]
        stack = stacks[stack_key]
        stack.plot(ax=ax[i][1])
        ax[i][1].grid(True, alpha=0.3, linestyle="--")
        ax[i][1].tick_params(direction="in", labelsize=10)

        # Format y-axis for SLD to show in scientific notation
        ax[i][1].ticklabel_format(
            style="scientific", axis="y", scilimits=(0, 0), useMathText=True
        )

        # Remove individual legends except for the top structure plot
        if i == 0:
            ax[i][1].legend(
                loc="upper center",
                ncol=2,
                fontsize=9,
                frameon=True,
                fancybox=False,
            )
        else:
            if ax[i][1].get_legend():
                ax[i][1].get_legend().remove()

    # Set up x-axis labels
    ax[n_objectives - 1][0].set_xlabel("Q (Å⁻¹)", fontsize=12, fontweight="bold")

    # Set y-axis labels for structure plots
    for i in range(n_objectives):
        ax[i][1].set_ylabel(None, fontsize=12, fontweight="bold")
        if i < n_objectives - 1:
            ax[i][1].set_xlabel("")
        else:
            ax[i][1].set_xlabel("Distance (Å)", fontsize=12, fontweight="bold")

    # Set consistent x-axis limits for better comparison
    # for i in range(n_objectives):
    # ax[i][1].set_xlim(-10, 200)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()

    return fig, ax


def load_reflectivity_dataset(filename):
    """
    Load reflectivity dataset from a parquet file.

    Parameters
    ----------
    filename : str
        Input filename (should be a .parquet file)

    Returns
    -------
    dict
        Dictionary containing XrayReflectDataset objects with energy keys
    """
    # Load the DataFrame
    df_load = pl.read_parquet(filename)

    # Reconstruct the data dictionary
    data_reconstructed = {}

    # Group by energy
    for group_key, group_data in df_load.group_by("energy"):
        energy_val = group_key[0]  # Extract energy value from the group key tuple

        # Extract arrays
        Q = group_data["Q"].to_numpy()
        R = group_data["R"].to_numpy()
        dR = group_data["dR"].to_numpy()

        # Create XrayReflectDataset
        dataset = fit.XrayReflectDataset(data=(Q, R, dR))
        data_reconstructed[str(energy_val)] = dataset

    print(f"Dataset loaded from {filename}")
    return data_reconstructed


def ooc_function(energy, ooc, theta=0.0, density=1.0):
    """Get optical constants for a given energy."""
    #  return the interpolated values for a given energy
    n_xx = np.interp(energy, ooc["energy"], ooc["n_xx"]) * density
    n_zz = np.interp(energy, ooc["energy"], ooc["n_zz"]) * density
    n_ixx = np.interp(energy, ooc["energy"], ooc["n_ixx"]) * density
    n_izz = np.interp(energy, ooc["energy"], ooc["n_izz"]) * density
    #  Rotate by theta
    if theta != 0.0:
        n_xx, n_zz = (
            0.5 * (n_xx * (1 + np.cos(theta) ** 2) + n_zz * np.sin(theta) ** 2),
            n_xx * np.sin(theta) ** 2 + n_zz * np.cos(theta) ** 2,
        )
        n_ixx, n_izz = (
            0.5 * (n_ixx * (1 + np.cos(theta) ** 2) + n_izz * np.sin(theta) ** 2),
            n_ixx * np.sin(theta) ** 2 + n_izz * np.cos(theta) ** 2,
        )
    return n_xx, n_zz, n_ixx, n_izz


def plot_optical_constants_with_energies(
    ooc,
    energy_batches,
    en_shift=-0.0,
    label=True,
    theta=0.0,
    density=1.0,
    show_theta_range=False,
):
    """
    Plot optical constants with energy markers and return figure and axes.

    Parameters:
    -----------
    ooc : pandas.DataFrame
        DataFrame containing optical constants data
    energy_batches : list
        List of energy arrays to mark on the plot
    """

    ooc_func = lambda e: ooc_function(e, ooc=ooc, theta=theta, density=density)  # noqa: E731
    if show_theta_range:
        ooc_func_low = lambda e: ooc_function(e, ooc=ooc, theta=0, density=density)  # noqa: E731
        ooc_func_high = lambda e: ooc_function(  # noqa: E731
            e, ooc=ooc, theta=np.pi / 2, density=density
        )

    # Create a cleaner plot with better layout - stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    emin = min(energy_batches) + en_shift - 10
    emax = max(energy_batches) + en_shift + 10
    # Create energy array for plotting
    energy_plot = np.linspace(emin, emax, 1000)

    # Calculate optical constants using ooc_func
    n_xx_plot = []
    n_zz_plot = []
    n_ixx_plot = []
    n_izz_plot = []

    for energy in energy_plot:
        n_xx, n_zz, n_ixx, n_izz = ooc_func(energy)
        n_xx_plot.append(n_xx)
        n_zz_plot.append(n_zz)
        n_ixx_plot.append(n_ixx)
        n_izz_plot.append(n_izz)

    # Plot the main curves
    ax1.plot(energy_plot, n_xx_plot, "b-", linewidth=1.5, label="δ_xx")
    ax1.plot(energy_plot, n_zz_plot, "r-", linewidth=1.5, label="δ_zz")
    ax1.set_ylabel("δ (Real part)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(direction="in")

    ax2.plot(energy_plot, n_ixx_plot, "b-", linewidth=1.5, label="β_xx")
    ax2.plot(energy_plot, n_izz_plot, "r-", linewidth=1.5, label="β_zz")

    #  fill between for theta range if specified
    if show_theta_range:
        n_xx_low_plot = []
        n_xx_high_plot = []
        n_ixx_low_plot = []
        n_ixx_high_plot = []

        for energy in energy_plot:
            n_xx_low, _, n_ixx_low, _ = ooc_func_low(energy)
            n_xx_high, _, n_ixx_high, _ = ooc_func_high(energy)
            n_xx_low_plot.append(n_xx_low)
            n_xx_high_plot.append(n_xx_high)
            n_ixx_low_plot.append(n_ixx_low)
            n_ixx_high_plot.append(n_ixx_high)

        ax1.fill_between(
            energy_plot,
            n_xx_low_plot,
            n_xx_high_plot,
            color="blue",
            alpha=0.1,
        )

        ax2.fill_between(
            energy_plot,
            n_ixx_low_plot,
            n_ixx_high_plot,
            color="blue",
            alpha=0.1,
        )

    ax2.set_ylabel("β (Imaginary part)", fontsize=12)
    ax2.set_xlabel("Energy (eV)", fontsize=12)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(direction="in")
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Draw vertical lines at the measurement energies
    if not label:
        plt.tight_layout()
        return fig, (ax1, ax2)

    for i, e in enumerate(energy_batches):
        if e < 283.7:
            continue
        n_xx, n_zz, n_ixx, n_izz = ooc_func(e)
        n_xx_shifted, n_zz_shifted, n_ixx_shifted, n_izz_shifted = ooc_func(
            e + en_shift
        )
        ax1.vlines(
            e,
            ymin=min(n_xx, n_zz),
            ymax=max(n_xx, n_zz),
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=0,
        )
        ax2.vlines(
            e,
            ymin=min(n_ixx, n_izz),
            ymax=max(n_ixx, n_izz),
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=0,
        )

        # Energy-shifted energies (solid red lines)
        ax1.vlines(
            e + en_shift,
            ymin=min(n_xx_shifted, n_zz_shifted),
            ymax=max(n_xx_shifted, n_zz_shifted),
            color="k",
            linestyle="-",
            linewidth=1.2,
            zorder=0,
        )
        ax2.vlines(
            e + en_shift,
            ymin=min(n_ixx_shifted, n_izz_shifted),
            ymax=max(n_ixx_shifted, n_izz_shifted),
            color="k",
            linestyle="-",
            linewidth=1.2,
            zorder=0,
        )

        ax1.scatter(
            e + en_shift,
            n_xx_shifted,
            color="blue",
            marker="o",
            s=50,
            label=f"δ_xx at {e + en_shift} eV",
        )
        ax1.scatter(
            e + en_shift,
            n_zz_shifted,
            color="red",
            marker="o",
            s=50,
            label=f"δ_zz at {e} eV",
        )
        ax2.scatter(
            e + en_shift,
            n_ixx_shifted,
            color="blue",
            marker="o",
            s=50,
            label=f"β_xx at {e} eV",
        )
        ax2.scatter(
            e + en_shift,
            n_izz_shifted,
            color="red",
            marker="o",
            s=50,
            label=f"β_zz at {e + en_shift} eV",
        )
        # if the energy is >= 283.7 eV annotate the point with a box and bold text
        if e >= 283.7:
            ax1.annotate(
                f"{e + en_shift:.1f} eV",
                (e + en_shift, n_xx_shifted),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
                fontweight="bold",
            )
            ax2.annotate(
                f"{e + en_shift:.1f} eV",
                (e + en_shift, n_ixx_shifted),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
                fontweight="bold",
            )
    # Adjust layout
    plt.tight_layout()

    return fig, (ax1, ax2)

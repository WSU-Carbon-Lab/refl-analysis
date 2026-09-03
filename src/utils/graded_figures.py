"""Matplotlib helpers for graded book-ended fit and validation figures.

Build publication-style panels from objectives produced by
:mod:`~utils.graded_objective` (upsampled meshes, multi-energy rebuilds).
Uses pyref's native ``AnisotropyObjective.plot`` for reflectivity so s/p
cross terms stay on the correct Jones ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from refloxide.pxr.energy.bookended import EnergyBookendedOrientationDensityProfile

from utils.graded_objective import (
    bookended_film_from_structure,
    extract_graded_fit_context,
)

if TYPE_CHECKING:
    from utils.models import AnisotropyObjective


def bookended_profile_arrays(
    film: EnergyBookendedOrientationDensityProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample depth, orientation angle, and mass density along the graded film.

    Parameters
    ----------
    film
        Book-ended film component.

    Returns
    -------
    depth, alpha, density
        1-D arrays at microslab midpoints (angstrom, rad, g/cm^3).
    """
    depth = np.asarray(film.mid_points, dtype=np.float64)
    alpha = np.asarray(film.orientation(depth), dtype=np.float64)
    density = np.asarray(film.local_density(depth), dtype=np.float64)
    return depth, alpha, density


def plot_bookended_profiles(
    films: Sequence[EnergyBookendedOrientationDensityProfile],
    *,
    labels: Sequence[str] | None = None,
    ax_alpha: Axes | None = None,
    ax_density: Axes | None = None,
    alpha_deg: bool = True,
) -> tuple[Axes, Axes]:
    """Plot orientation and density profiles for one or more book-ended films.

    Parameters
    ----------
    films
        Films to overlay (e.g. fit mesh vs upsampled plot mesh).
    labels
        Legend labels; length must match ``films`` when provided.
    ax_alpha, ax_density
        Optional axes. When omitted, a new 1x2 figure is created.
    alpha_deg
        When ``True``, plot orientation in degrees; otherwise radians.

    Returns
    -------
    ax_alpha, ax_density
        Axes with profiles drawn.
    """
    if ax_alpha is None or ax_density is None:
        _, (ax_alpha, ax_density) = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=False)
    if labels is not None and len(labels) != len(films):
        msg = f"labels length {len(labels)} != films length {len(films)}"
        raise ValueError(msg)

    for idx, film in enumerate(films):
        depth, alpha, density = bookended_profile_arrays(film)
        if alpha_deg:
            alpha = np.degrees(alpha)
        label = None if labels is None else labels[idx]
        ax_alpha.plot(depth, alpha, label=label)
        ax_density.plot(depth, density, label=label)

    ylabel = r"$\alpha$ (deg)" if alpha_deg else r"$\alpha$ (rad)"
    ax_alpha.set_xlabel("depth (A)")
    ax_alpha.set_ylabel(ylabel)
    ax_density.set_xlabel("depth (A)")
    ax_density.set_ylabel(r"$\rho$ (g/cm$^3$)")
    if labels is not None:
        ax_alpha.legend(frameon=False, fontsize="small")
    return ax_alpha, ax_density


def plot_reflectivity_objective(
    objective: AnisotropyObjective,
    ax: Axes | None = None,
    *,
    show_anisotropy: bool = True,
    **plot_kwargs: Any,
) -> Axes:
    """Plot data and model reflectivity for a graded anisotropy objective.

    Parameters
    ----------
    objective
        Single-energy objective (fit or upsampled rebuild).
    ax
        Target axes. When ``None``, uses ``plt.gca()``.
    show_anisotropy
        Forwarded to ``objective.plot``.
    **plot_kwargs
        Additional keyword arguments for ``objective.plot``.

    Returns
    -------
    Axes
        Axes with reflectivity curves.
    """
    target = ax if ax is not None else plt.gca()
    objective.plot(ax=target, show_anisotropy=show_anisotropy, **plot_kwargs)
    energy = float(objective.model.energy)
    target.set_title(f"{energy:.1f} eV")
    return target


def plot_structure_sld(
    structure: Any,
    ax: Axes | None = None,
    *,
    title: str | None = None,
    **plot_kwargs: Any,
) -> Axes:
    """Plot SLD / tensor stack diagram via pyref ``Structure.plot``.

    Parameters
    ----------
    structure
        Pyref ``Structure`` (graded or DFT template).
    ax
        Optional axes.
    title
        Optional panel title after plotting.
    **plot_kwargs
        Forwarded to ``structure.plot``.

    Returns
    -------
    Axes
        Axes with the structure diagram.
    """
    target = ax if ax is not None else plt.gca()
    structure.plot(ax=target, **plot_kwargs)
    if title is not None:
        target.set_title(title)
    return target


def figure_fit_vs_upsampled(
    fit_objective: AnisotropyObjective,
    plot_objective: AnisotropyObjective,
    *,
    dft_objective: AnisotropyObjective | None = None,
    figsize: tuple[float, float] = (10.0, 7.0),
) -> Figure:
    """Two-row figure: reflectivity fit vs upsampled model; profiles and SLD.

    Parameters
    ----------
    fit_objective
        Objective used during fitting (coarse microslab mesh).
    plot_objective
        Upsampled objective from
        :func:`~utils.graded_objective.upsample_graded_objective`.
    dft_objective
        Optional DFT template objective for a third SLD panel.
    figsize
        Figure size in inches.

    Returns
    -------
    Figure
        Matplotlib figure with four panels.
    """
    ctx_fit = extract_graded_fit_context(fit_objective)
    ctx_plot = extract_graded_fit_context(plot_objective)
    n_cols = 3 if dft_objective is not None else 2
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)

    plot_reflectivity_objective(
        plot_objective,
        ax=axes[0, 0],
        show_anisotropy=False,
    )
    axes[0, 0].set_title(f"reflectivity ({ctx_plot.film.num_slabs} slabs)")

    plot_bookended_profiles(
        [ctx_fit.film, ctx_plot.film],
        labels=[f"fit ({ctx_fit.film.num_slabs})", f"plot ({ctx_plot.film.num_slabs})"],
        ax_alpha=axes[0, 1],
        ax_density=axes[1, 1],
    )
    axes[0, 1].set_title("orientation")
    axes[1, 1].set_title("density")

    plot_structure_sld(
        ctx_plot.structure,
        ax=axes[1, 0],
        title=f"graded SLD ({ctx_plot.film.num_slabs} slabs)",
    )

    if dft_objective is not None:
        plot_structure_sld(
            dft_objective.model.structure,
            ax=axes[0, 2],
            title="DFT slab template",
        )
        axes[1, 2].set_visible(False)

    fig.tight_layout()
    return fig


def figure_multi_energy_reflectivity(
    objectives: dict[float, AnisotropyObjective],
    *,
    energies: Sequence[float] | None = None,
    ncol: int = 3,
    figsize: tuple[float, float] | None = None,
    show_anisotropy: bool = False,
) -> Figure:
    """Grid of reflectivity panels keyed by photon energy.

    Parameters
    ----------
    objectives
        Mapping from energy (eV) to objectives (e.g. from
        :func:`~utils.graded_objective.graded_objectives_for_energies`).
    energies
        Subset and order of energies to plot. Defaults to sorted keys.
    ncol
        Number of columns in the subplot grid.
    figsize
        Figure size. Defaults to ``(3.5 * ncol, 2.8 * nrows)``.
    show_anisotropy
        Forwarded to each ``objective.plot`` call.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    ordered = sorted(objectives) if energies is None else [float(e) for e in energies]
    n = len(ordered)
    nrows = int(np.ceil(n / ncol))
    if figsize is None:
        figsize = (3.5 * ncol, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncol, figsize=figsize, squeeze=False)
    for idx, energy in enumerate(ordered):
        row, col = divmod(idx, ncol)
        plot_reflectivity_objective(
            objectives[energy],
            ax=axes[row, col],
            show_anisotropy=show_anisotropy,
        )
    for idx in range(n, nrows * ncol):
        row, col = divmod(idx, ncol)
        axes[row, col].set_visible(False)
    fig.tight_layout()
    return fig


def film_from_objective(
    objective: AnisotropyObjective,
) -> EnergyBookendedOrientationDensityProfile:
    """Return the book-ended film inside ``objective.model.structure``."""
    return bookended_film_from_structure(objective.model.structure)


__all__ = [
    "bookended_profile_arrays",
    "figure_fit_vs_upsampled",
    "figure_multi_energy_reflectivity",
    "film_from_objective",
    "plot_bookended_profiles",
    "plot_reflectivity_objective",
    "plot_structure_sld",
]

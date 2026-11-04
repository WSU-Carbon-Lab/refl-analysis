"""Depth-profile comparison: bookended vs DFT-slab E_RES logY polish fits.

Samples orientation and mass density along the ZnPc film (vacuum interface at
``z = 0``) for the multi-energy bookended polish and the slab L-BFGS result,
using the same colors as ``_plot_bookended_vs_dft_slab_logY.py``.
"""

from __future__ import annotations

import json
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from utils import models_fitting_results
from utils.graded_objective import (
    extract_graded_fit_context,
    load_bookended_multienergy_logY_polish,
)
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

BOOKENDED_DIR = models_fitting_results / "bookended_multienergy_logY_polish"
SLAB_DIR = models_fitting_results / "dft_slab_eres_logY_polish"
OUT_DIR = models_fitting_results / "bookended_vs_dft_slab_eres_logY"

COLOR_BOOK = "#C0392B"
COLOR_SLAB = "#2471A3"
FILM_SLAB_IDX = (1, 2, 3)


def log(msg: str) -> None:
    print(msg, flush=True)


def nevot_croce_density(film, *, n_points: int = 2000, n_sigma: int = 5):
    """Convolve book-ended density with the surface Névot-Croce roughness."""
    total = float(film.total_thick.value)
    sigma = float(film.surface_roughness.value)
    depth = np.linspace(0.0, total, n_points)
    rho_sharp = np.asarray(film.local_density(depth), dtype=float)
    if sigma <= 0.0:
        return depth, rho_sharp, rho_sharp.copy()

    dz = depth[1] - depth[0]
    pad = int(np.ceil(n_sigma * sigma / dz))
    z_ext = np.concatenate(
        [
            depth[0] + dz * np.arange(-pad, 0),
            depth,
            depth[-1] + dz * np.arange(1, pad + 1),
        ]
    )
    rho_ext = np.empty_like(z_ext)
    below, above = z_ext < 0.0, z_ext > total
    inside = ~(below | above)
    rho_ext[below] = 0.0
    rho_ext[inside] = np.asarray(film.local_density(z_ext[inside]), dtype=float)
    rho_ext[above] = float(film.local_density(total))

    kernel = np.exp(-0.5 * (dz * np.arange(-pad, pad + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    rho_rough = np.convolve(rho_ext, kernel, mode="same")[pad : pad + n_points]
    return depth, rho_sharp, rho_rough


def slab_film_profiles(model, idx: tuple[int, ...] = FILM_SLAB_IDX, n: int = 600):
    """Erf-smoothed orientation (deg) and density for the three ZnPc slabs.

    ``component[i].rough`` is the Névot-Croce width at the interface above
    slab ``i``; includes the vacuum/film step at ``z = 0`` via ``slabs[0].rough``.
    """
    slabs = [model.structure[i] for i in idx]
    thick = np.array([float(c.thick.value) for c in slabs])
    alpha = np.array([float(c.sld.rotation.value) for c in slabs])
    rho = np.array([float(c.sld.density.value) for c in slabs])
    r0 = max(float(slabs[0].rough.value), 1e-6)
    r1 = max(float(slabs[1].rough.value), 1e-6)
    r2 = max(float(slabs[2].rough.value), 1e-6)
    total = float(thick.sum())
    depth = np.linspace(0.0, total, n)
    z1, z2 = thick[0], thick[0] + thick[1]
    step0 = 0.5 * (1.0 + erf((depth - 0.0) / (np.sqrt(2.0) * r0)))
    step1 = 0.5 * (1.0 + erf((depth - z1) / (np.sqrt(2.0) * r1)))
    step2 = 0.5 * (1.0 + erf((depth - z2) / (np.sqrt(2.0) * r2)))

    def blend(values: np.ndarray, *, vac: float) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        return (
            vac + (v[0] - vac) * step0 + (v[1] - v[0]) * step1 + (v[2] - v[1]) * step2
        )

    return depth, np.degrees(blend(alpha, vac=0.0)), blend(rho, vac=0.0)


def bookended_profiles_at(energy: float = 283.7):
    """Return depth, orientation (deg), and NC-smeared density for bookended film."""
    polish = load_bookended_multienergy_logY_polish(BOOKENDED_DIR / "fit.pkl")
    film = extract_graded_fit_context(polish.at(energy)).film
    depth, _rho_sharp, rho_rough = nevot_croce_density(film)
    alpha_deg = np.degrees(np.asarray(film.orientation(depth), dtype=float))
    return depth, alpha_deg, rho_rough, float(film.total_thick.value)


def slab_profiles():
    """Return depth, orientation (deg), density, and total film thickness for slab."""
    with (SLAB_DIR / "fit.pkl").open("rb") as fh:
        objective = pickle.load(fh)["objective"]
    depth, alpha_deg, rho = slab_film_profiles(objective.model)
    total = float(sum(objective.model.structure[i].thick.value for i in FILM_SLAB_IDX))
    return depth, alpha_deg, rho, total


def plot_comparison(path_stem) -> dict:
    """Draw orientation and density overlays; return sampled arrays for metadata."""
    z_book, alpha_book, rho_book, l_book = bookended_profiles_at()
    z_slab, alpha_slab, rho_slab, l_slab = slab_profiles()

    book_summary = json.loads((BOOKENDED_DIR / "summary.json").read_text())
    slab_summary = json.loads((SLAB_DIR / "summary.json").read_text())
    ox = book_summary["oxide"]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_alpha, ax_rho) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        constrained_layout=True,
        sharex=False,
    )

    s_book = z_book / l_book
    s_slab = z_slab / l_slab

    ax_alpha.plot(s_book, alpha_book, color=COLOR_BOOK, lw=1.4, label="bookended")
    ax_alpha.plot(s_slab, alpha_slab, color=COLOR_SLAB, lw=1.4, label="DFT slab")
    ax_alpha.axvline(1.0, color="0.45", ls=":", lw=0.9, alpha=0.7)
    ax_alpha.set_xlabel(r"$z / L$")
    ax_alpha.set_ylabel(r"$\gamma$ (deg)")
    ax_alpha.set_title("orientation")

    ax_rho.plot(s_book, rho_book, color=COLOR_BOOK, lw=1.4, label="bookended")
    ax_rho.plot(s_slab, rho_slab, color=COLOR_SLAB, lw=1.4, label="DFT slab")
    ax_rho.axvline(1.0, color="0.45", ls=":", lw=0.9, alpha=0.7)
    ax_rho.set_xlabel(r"$z / L$")
    ax_rho.set_ylabel(r"$\rho$ (g/cm$^3$)")
    ax_rho.set_title("density (NC / erf smoothed)")

    for ax in (ax_alpha, ax_rho):
        ax.set_xlim(0.0, 1.0)
        ax.legend(frameon=False, loc="best", fontsize=8)

    fig.suptitle(
        r"ZnPc depth profiles — bookended vs DFT slab (E$_{\mathrm{RES}}$ logY polish)"
        f"\n$L_{{\\mathrm{{book}}}}={l_book:.1f}$ Å  ·  "
        f"$L_{{\\mathrm{{slab}}}}={l_slab:.1f}$ Å  ·  "
        f"oxide={ox['thick']:.2f}/{ox['rough']:.2f} Å  ·  "
        f"RMS book {book_summary['after']['rms']:.2f}  "
        f"slab {slab_summary['after']['rms']:.2f}",
        fontsize=10,
    )

    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "normalization": {
            "axis": "z_over_L_model",
            "L_book_A": l_book,
            "L_slab_A": l_slab,
            "L_slab_over_L_book": l_slab / l_book,
        },
        "bookended": {
            "film_thick_A": l_book,
            "film": book_summary["film"],
        },
        "dft_slab": {
            "film_thick_A": l_slab,
            "film": slab_summary["film"],
        },
        "oxide": ox,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "demo_bookended_vs_dft_slab_profiles"
    meta = plot_comparison(stem)
    meta_path = OUT_DIR / "profiles_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"wrote {stem.with_suffix('.png')}")
    log(f"wrote {stem.with_suffix('.pdf')}")
    log(
        f"film thickness  book={meta['bookended']['film_thick_A']:.1f} Å  "
        f"slab={meta['dft_slab']['film_thick_A']:.1f} Å"
    )


if __name__ == "__main__":
    main()

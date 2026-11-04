"""Overlay bookended vs DFT-slab logY polish demos on ``E_RES``.

Writes a multi-energy panel figure in the same style as
``demo_multienergy_logY.png``, with both fits on shared data.
"""

from __future__ import annotations

import json
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import Transform
from utils import models_fitting_results
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

E_RES = (283.7, 284.2, 285.1, 287.0, 289.0)
BOOKENDED_DIR = models_fitting_results / "bookended_multienergy_logY_polish"
SLAB_DIR = models_fitting_results / "dft_slab_eres_logY_polish"
OUT_DIR = models_fitting_results / "bookended_vs_dft_slab_eres_logY"
TRANSFORM = Transform("logY")

COLOR_DATA = "0.25"
COLOR_BOOK = "#C0392B"
COLOR_SLAB = "#2471A3"


def log(msg: str) -> None:
    print(msg, flush=True)


def reflectivity(model, q: np.ndarray, energy: float, pol: str) -> np.ndarray:
    """Evaluate one polarization channel of a refloxide ``ReflectModel``."""
    result = model(np.asarray(q, dtype=float), float(energy))
    return np.asarray(result.s if pol == "s" else result.p, dtype=float)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (BOOKENDED_DIR / "fit.pkl").open("rb") as fh:
        book_bundle = pickle.load(fh)
    with (SLAB_DIR / "fit.pkl").open("rb") as fh:
        slab_bundle = pickle.load(fh)

    book_obj = book_bundle["gpu_objective"]
    slab_obj = slab_bundle["objective"]
    book_summary = book_bundle["summary"]
    slab_summary = slab_bundle["summary"]

    book_obj.model.device = "cpu"
    slab_obj.model.device = "cpu"
    book_obj.model.parallel = True
    slab_obj.model.parallel = True

    ds = book_obj._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e_arr = np.asarray(ds.energy, dtype=float)

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
    nrows = len(E_RES)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.0, 1.55 * nrows + 0.7),
        sharex="col",
        constrained_layout=True,
    )
    data_kw = dict(
        fmt="o",
        ms=2.2,
        lw=0.45,
        color=COLOR_DATA,
        ecolor="0.65",
        elinewidth=0.4,
        capsize=0,
        zorder=1,
        rasterized=True,
    )
    letters = "abcdefghij"

    for i, energy in enumerate(E_RES):
        book_rms = next(
            row["rms"]
            for row in book_summary["per_energy"]
            if np.isclose(row["energy"], energy)
        )
        slab_rms = next(
            row["rms"]
            for row in slab_summary["per_energy"]
            if np.isclose(row["energy"], energy)
        )
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq, rr, ee = q[m], r[m], r_err[m]
            order = np.argsort(qq)
            qq, rr, ee = qq[order], rr[order], ee[order]

            book_r = reflectivity(book_obj.model, qq, energy, pol_name)
            slab_r = reflectivity(slab_obj.model, qq, energy, pol_name)

            yd, ed = TRANSFORM(qq, rr, ee)
            y_book, _ = TRANSFORM(qq, book_r)
            y_slab, _ = TRANSFORM(qq, slab_r)

            ax.errorbar(qq, yd, yerr=ed, **data_kw)
            ax.plot(qq, y_book, color=COLOR_BOOK, lw=1.2, zorder=3)
            ax.plot(qq, y_slab, color=COLOR_SLAB, lw=1.2, zorder=2)
            ax.text(
                0.02,
                0.94,
                f"({letters[i * 2 + col]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=9,
            )
            ax.text(
                0.98,
                0.94,
                f"{energy:g} eV  {pol_name}\nbook {book_rms:.2f}  slab {slab_rms:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="0.2",
                linespacing=1.25,
            )
            if col == 0:
                ax.set_ylabel(r"$\log_{10} R$")
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")

    axes[0, 0].plot([], [], "o", ms=3.5, color=COLOR_DATA, label="data")
    axes[0, 0].plot([], [], color=COLOR_BOOK, lw=1.4, label="bookended")
    axes[0, 0].plot([], [], color=COLOR_SLAB, lw=1.4, label="DFT slab")
    axes[0, 0].legend(loc="lower left", frameon=False, handlelength=1.6)

    ox = book_summary["oxide"]
    fig.suptitle(
        r"ZnPc E_RES log$Y$ polish — bookended vs DFT slab"
        f"\nshared oxide={ox['thick']:.2f}/{ox['rough']:.2f} Å, "
        r"$\rho_{\mathrm{SiO_2}}=2.2$, $\rho_{\mathrm{Si}}=2.33$, "
        r"$r_{\mathrm{Si}}=1.2$ Å"
        f"  ·  book RMS={book_summary['after']['rms']:.2f}"
        f"  ·  slab RMS={slab_summary['after']['rms']:.2f}",
        fontsize=10,
    )
    stem = OUT_DIR / "demo_bookended_vs_dft_slab_logY"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    meta = {
        "energies": list(E_RES),
        "bookended": {
            "path": str(BOOKENDED_DIR.resolve()),
            "rms": book_summary["after"]["rms"],
            "nll": book_summary["after"]["nll"],
            "per_energy": book_summary["per_energy"],
        },
        "dft_slab": {
            "path": str(SLAB_DIR.resolve()),
            "rms": slab_summary["after"]["rms"],
            "nll": slab_summary["after"]["nll"],
            "per_energy": slab_summary["per_energy"],
        },
        "oxide": ox,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {stem.with_suffix('.png')}")
    log(f"wrote {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

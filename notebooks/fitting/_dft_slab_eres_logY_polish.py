"""logYX4 L-BFGS-B from the as-fit DFT-slab warm start (no DE).

Builds the E_RES slab objective from ``refit-refloxide.pkl`` (as-fit slab
MLE) with bookended oxide/Si locks, locks per-energy ``bkg`` to the
bookended polish values, frees film + scales/thetas, and runs
bound-normalized L-BFGS-B via ``refloxide.objective.minimize_lbfgsb``
under ``LogYX4``.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from refloxide.objective import minimize_lbfgsb
from refnx.analysis import Transform

_HERE = Path(__file__).resolve().parent
_STAGE1 = _HERE / "_dft_slab_eres_logyx4.py"
_spec = importlib.util.spec_from_file_location("_dft_slab_eres_logyx4", _STAGE1)
s1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = s1
assert _spec.loader is not None
_spec.loader.exec_module(s1)

E_RES = s1.E_RES
RHO_SIO2 = s1.RHO_SIO2
RHO_SI = s1.RHO_SI
SUBSTRATE_ROUGH = s1.SUBSTRATE_ROUGH
FILM_SLABS = s1.FILM_SLABS
WARM_PATH = s1.WARM_PATH
BOOKENDED_POLISH = s1.BOOKENDED_POLISH
OUT_DIR = s1.models_fitting_results / "dft_slab_eres_logY_polish"
TRANSFORM = s1.TRANSFORM
DISPLAY = Transform("logY")


def log(msg: str) -> None:
    print(msg, flush=True)


def polish(objective) -> list[dict]:
    history: list[dict] = []
    objective.model.device = "cpu"
    objective.model.parallel = True
    nll0 = float(objective.nll())
    history.append({"stage": "start", "nll": nll0, "wall_s": 0.0})
    log(f"logYX4 start nll={nll0:.6f}  nvary={len(objective.varying_parameters())}")

    t0 = time.perf_counter()
    result = minimize_lbfgsb(
        objective,
        options={"maxiter": 500, "ftol": 1e-14},
    )
    nll1 = float(objective.nll())
    history.append(
        {
            "stage": "cpu_minimize_lbfgsb_logYX4",
            "nll": nll1,
            "delta_nll": nll0 - nll1,
            "nfev": int(getattr(result, "nfev", -1)),
            "nit": int(getattr(result, "nit", -1)),
            "wall_s": time.perf_counter() - t0,
            "message": str(getattr(result, "message", "")),
            "success": bool(getattr(result, "success", False)),
        }
    )
    log(
        f"  cpu L-BFGS-B (nll_and_grad)  nll={nll1:.6f}  Δnll={nll0 - nll1:+.6f}  "
        f"nfev={history[-1]['nfev']}  nit={history[-1]['nit']}  "
        f"wall={history[-1]['wall_s']:.2f}s  msg={history[-1]['message']}"
    )
    return history


def plot_demo(objective, summary: dict, path: Path) -> None:
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e_arr = np.asarray(ds.energy)
    energies = [float(e) for e in summary["energies"]]

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
    nrows = len(energies)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.0, 1.55 * nrows + 0.55),
        sharex="col",
        constrained_layout=True,
    )
    data_kw = dict(
        fmt="o",
        ms=2.2,
        lw=0.45,
        color="0.25",
        ecolor="0.65",
        elinewidth=0.4,
        capsize=0,
        zorder=1,
        rasterized=True,
    )
    fit_kw = dict(color="#2471A3", lw=1.15, zorder=2)
    letters = "abcdefghij"
    for i, energy in enumerate(energies):
        rms = next(
            row["rms"]
            for row in summary["per_energy"]
            if np.isclose(row["energy"], energy)
        )
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq, rr, ee, ff = q[m], r[m], r_err[m], y_fit[m]
            order = np.argsort(qq)
            qq, rr, ee, ff = qq[order], rr[order], ee[order], ff[order]
            yd, ed = DISPLAY(qq, rr, ee)
            yf, _ = DISPLAY(qq, ff)
            ax.errorbar(qq, yd, yerr=ed, **data_kw)
            ax.plot(qq, yf, **fit_kw)
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
                f"{energy:g} eV  {pol_name}   RMS {rms:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.2",
            )
            ax.set_ylabel(r"$\log_{10} R$")
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")

    after = summary["after"]
    ox = summary["oxide"]
    fig.suptitle(
        f"DFT slab E_RES logYX4 L-BFGS  |  nll={after['nll']:.3f}  "
        f"RMS={after['rms']:.3f}  |  oxide={ox['thick']:.2f}/{ox['rough']:.2f}",
        fontsize=10,
    )
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    log(f"wrote {path.with_suffix('.png')}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oxide_mle, energy_offset, bkg_lock = s1.load_bookended_locks()

    warm = s1.load_refloxide_fit(WARM_PATH)
    objective, comps = s1.build_objective(
        warm,
        oxide_mle,
        energy_offset,
        bkg_lock=bkg_lock,
        transform=TRANSFORM,
        device="cpu",
    )
    objective.refresh_parameters()
    _vac, surface, bulk, interface, oxide, _sub = comps

    before = s1.metrics(objective)
    log(f"before (logYX4, slab start): {before}")
    history = polish(objective)
    after = s1.metrics(objective)
    per = s1.per_energy_rms(objective, transform=DISPLAY)

    summary = {
        "energies": list(E_RES),
        "transform": "logYX4",
        "rho_sio2": RHO_SIO2,
        "rho_si": RHO_SI,
        "substrate_rough": SUBSTRATE_ROUGH,
        "energy_offset": energy_offset,
        "before": before,
        "after": after,
        "history": history,
        "per_energy": per,
        "film": s1.film_snapshot(surface, bulk, interface),
        "oxide": {
            "thick": float(oxide.thick.value),
            "rough": float(oxide.rough.value),
            "density": float(oxide.sld.density.value),
        },
        "instruments": s1.instrument_snapshot(objective.model),
        "protocol": "cpu_minimize_lbfgsb_dft_slab_eres_logYX4_from_slab_mle_bkg_bookended",
        "warm_from": str(WARM_PATH.resolve()),
        "bookended_locks": str(BOOKENDED_POLISH.resolve()),
    }

    fit_path = OUT_DIR / "fit.pkl"
    with fit_path.open("wb") as fh:
        pickle.dump({"objective": objective, "summary": summary}, fh)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_demo(objective, summary, OUT_DIR / "demo_dft_slab_eres_logY")
    log(f"wrote {fit_path}")
    log(f"after: {after}")
    for row in per:
        log(f"  {row['energy']:g} eV  RMS={row['rms']:.3f}  n={row['n']}")


if __name__ == "__main__":
    main()

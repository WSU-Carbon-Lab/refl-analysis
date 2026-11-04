"""Re-polish FIG4 cores on the full resonant set including 283.7 eV.

Seeds
-----
* Graded bookended: ``bookended_multienergy_logY_polish/fit.pkl``
* DFT-slab: ``dft_slab_eres_logY_polish/fit.pkl`` (s/p labels corrected)

Energies: ``(283.7, 284.2, 285.1, 287.0, 289.0)``. Free sets match the FIG4
polishes (bookended: ``tau_vac`` / ``alpha_vac`` / ``surface_roughness`` +
per-energy scale/theta; slab: ZnPc film + per-energy scale/theta). Likelihood
contract: ``Transform('logY')``, ``anisotropy_weight=0``,
``normalization='auto'``, ``nc_constraint=True``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from refloxide.data import ReflectDataset
from refloxide.objective import _build_kernel_batches
from refnx.analysis import Transform
from scipy.optimize import minimize
from utils import models_fitting_results
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

_HERE = Path(__file__).resolve().parent
OUT_ROOT = models_fitting_results / "eres_full_logY_polish"
BOOK_SEED = models_fitting_results / "bookended_multienergy_logY_polish" / "fit.pkl"
SLAB_SEED = models_fitting_results / "dft_slab_eres_logY_polish" / "fit.pkl"
E_RES = (283.7, 284.2, 285.1, 287.0, 289.0)
TRANSFORM = Transform("logY")
COLOR_BOOK = "#C0392B"
COLOR_SLAB = "#1F618D"
COLOR_DATA = "0.25"
E_TAGS = ("283p7eV", "284p2eV", "285p1eV", "287p0eV", "289p0eV")


def log(msg: str) -> None:
    print(msg, flush=True)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def prep(objective) -> None:
    objective.model.device = "cpu"
    objective.model.parallel = False
    objective.transform = TRANSFORM
    objective.anisotropy_weight = 0.0
    if hasattr(objective, "normalization"):
        objective.normalization = "auto"
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    if hasattr(objective, "anisotropy_targets"):
        objective.anisotropy_targets = None
    if hasattr(objective, "_anisotropy_pairs"):
        objective._anisotropy_pairs = ()
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()


def restrict_energies(objective, energies: tuple[float, ...] = E_RES) -> int:
    ds = objective._dataset
    e = np.asarray(ds.energy, dtype=float)
    keep = np.zeros(e.shape, dtype=bool)
    for energy in energies:
        keep |= np.isclose(e, float(energy))
    n = int(keep.sum())
    if n == 0:
        raise RuntimeError(f"no rows in energies={energies}")
    subset = ReflectDataset(
        q=np.asarray(ds.q, dtype=float)[keep].copy(),
        energy=e[keep].copy(),
        pol=np.asarray(ds.pol)[keep].copy(),
        r=np.asarray(ds.r, dtype=float)[keep].copy(),
        r_err=np.asarray(ds.r_err, dtype=float)[keep].copy(),
    )
    objective._dataset = subset
    objective._groups = list(subset.groups())
    objective._batches = _build_kernel_batches(subset)
    prep(objective)
    return n


def keep_fig4_free_set(objective, *, kind: str) -> None:
    """Open the FIG4 free set on all E_RES channels; freeze everything else."""
    film_book = {"tau_vac", "alpha_vac", "surface_roughness"}
    for p in objective.parameters.flattened():
        if getattr(p, "constraint", None) is not None:
            continue
        name = str(p.name)
        is_inst = name.startswith("scale_") or name.startswith("theta_offset_")
        is_bkg = name.startswith("bkg")
        keep = False
        if kind == "bookended":
            if name in film_book:
                keep = True
            elif is_inst and any(tag in name for tag in E_TAGS):
                keep = True
        else:
            if name.startswith("ZnPc_") and "energy_offset" not in name:
                keep = True
            elif is_inst and any(tag in name for tag in E_TAGS):
                keep = True
        if is_bkg:
            keep = False
        try:
            p.vary = bool(keep)
        except RuntimeError:
            pass
    objective.refresh_parameters()


def varying_state(objective):
    names, x0, bounds = [], [], []
    for p in objective.varying_parameters():
        names.append(str(p.name))
        x0.append(float(p.value))
        bounds.append((float(p.bounds.lb), float(p.bounds.ub)))
    return names, np.asarray(x0, dtype=float), bounds


def set_varying(objective, names, x) -> None:
    by = {str(p.name): p for p in objective.varying_parameters()}
    for name, val in zip(names, x, strict=True):
        by[name].value = float(val)


def overall_rms(objective) -> float:
    ds = objective._dataset
    y = np.asarray(objective.generative(), dtype=float)
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    yd, ed = TRANSFORM(q, r, r_err)
    ym, _ = TRANSFORM(q, y)
    ed = np.asarray(ed, dtype=float)
    m = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
    return float(np.sqrt(np.mean(((yd[m] - ym[m]) / ed[m]) ** 2)))


def per_energy_rms(objective) -> list[dict]:
    ds = objective._dataset
    y = np.asarray(objective.generative(), dtype=float)
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e = np.asarray(ds.energy, dtype=float)
    rows = []
    for energy in E_RES:
        row: dict = {"energy": float(energy)}
        for pn in ("s", "p"):
            m = (pol == pn) & np.isclose(e, energy)
            yd, ed = TRANSFORM(q[m], r[m], r_err[m])
            ym, _ = TRANSFORM(q[m], y[m])
            ed = np.asarray(ed, dtype=float)
            ok = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
            row[f"rms_{pn}"] = float(
                np.sqrt(np.mean(((yd[ok] - ym[ok]) / ed[ok]) ** 2))
            )
            row[f"n_{pn}"] = int(ok.sum())
        rows.append(row)
    return rows


def multi_start_lbfgs(objective, *, n_extra: int = 5, seed: int = 3) -> dict:
    names, x_seed, bounds = varying_state(objective)
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    if not np.all(np.isfinite(lo) & np.isfinite(hi)):
        bad = [
            n
            for n, (a, b) in zip(names, bounds, strict=True)
            if not (np.isfinite(a) and np.isfinite(b) and b > a)
        ]
        raise RuntimeError(f"non-finite free bounds: {bad}")
    x_seed = np.clip(x_seed, lo, hi)
    set_varying(objective, names, x_seed)
    prep(objective)
    nll_seed = float(objective.nll())

    def nll_fn(x):
        try:
            set_varying(objective, names, x)
            objective.refresh_parameters()
            val = float(objective.nll())
            return val if np.isfinite(val) else 1.0e12
        except (ValueError, RuntimeError, FloatingPointError):
            return 1.0e12

    rng = np.random.default_rng(seed)
    candidates = [x_seed.copy()]
    for _ in range(n_extra):
        u = rng.random(len(x_seed))
        xt = lo + u * (hi - lo)
        candidates.append(np.clip(0.8 * x_seed + 0.2 * xt, lo, hi))

    best_fun = nll_seed
    best_x = x_seed.copy()
    best_nit = 0
    best_success = False
    for i, xt in enumerate(candidates):
        res = minimize(
            nll_fn,
            xt,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-5},
        )
        fun = float(res.fun)
        log(
            f"  start[{i}]: nll {nll_fn(xt):.2f} -> {fun:.2f}  "
            f"success={res.success} nit={getattr(res, 'nit', '?')}"
        )
        if fun < best_fun:
            best_fun = fun
            best_x = np.asarray(res.x, dtype=float).copy()
            best_nit = int(getattr(res, "nit", -1))
            best_success = bool(res.success)

    set_varying(objective, names, best_x)
    prep(objective)
    return {
        "names": names,
        "bounds": bounds,
        "seed_values": {n: float(v) for n, v in zip(names, x_seed, strict=True)},
        "fit_values": {n: float(v) for n, v in zip(names, best_x, strict=True)},
        "nll_seed": nll_seed,
        "nll_fit": float(objective.nll()),
        "rms_fit": overall_rms(objective),
        "nvary": len(names),
        "success": best_success,
        "nit": best_nit,
    }


def _rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_fit(objective, *, title: str, color: str, stem: Path, meta: dict) -> None:
    _rc()
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e = np.asarray(ds.energy, dtype=float)
    y = np.asarray(objective.generative(), dtype=float)
    nrows = len(E_RES)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.0, 1.55 * nrows + 0.9),
        sharex="col",
        constrained_layout=True,
    )
    per = per_energy_rms(objective)
    for i, energy in enumerate(E_RES):
        row = per[i]
        for col, pn in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol == pn) & np.isclose(e, energy)
            qq, rr, ee, yy = q[m], r[m], r_err[m], y[m]
            order = np.argsort(qq)
            qq, rr, ee, yy = qq[order], rr[order], ee[order], yy[order]
            yd, ed = TRANSFORM(qq, rr, ee)
            ym, _ = TRANSFORM(qq, yy)
            ax.errorbar(
                qq,
                yd,
                yerr=ed,
                fmt="o",
                ms=2.2,
                lw=0.45,
                color=COLOR_DATA,
                ecolor="0.65",
                elinewidth=0.4,
                capsize=0,
                rasterized=True,
                zorder=1,
            )
            ax.plot(qq, ym, color=color, lw=1.2, zorder=2)
            ax.text(
                0.98,
                0.94,
                f"{energy:g} eV {pn}\nRMS={row[f'rms_{pn}']:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
            )
            if col == 0:
                ax.set_ylabel(r"$\log_{10} R$")
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    fig.suptitle(
        f"{title}\nnll={meta['nll_fit']:.1f}  RMS={meta['rms_fit']:.2f}  "
        f"nvary={meta['nvary']}  energies={list(E_RES)}",
        fontsize=9.5,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {stem.with_suffix('.png')}")


def plot_overlay(book, slab, *, stem: Path, book_meta: dict, slab_meta: dict) -> None:
    _rc()
    ds = slab._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e = np.asarray(ds.energy, dtype=float)
    nrows = len(E_RES)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.0, 1.55 * nrows + 1.0),
        sharex="col",
        constrained_layout=True,
    )
    sp = _load("sp_full", _HERE / "_plot_slab_dynesty_exports.py")
    for i, energy in enumerate(E_RES):
        for col, pn in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol == pn) & np.isclose(e, energy)
            qq, rr, ee = q[m], r[m], r_err[m]
            order = np.argsort(qq)
            qq, rr, ee = qq[order], rr[order], ee[order]
            yb_m = sp.reflectivity(book.model, qq, energy, pn)
            ys_m = sp.reflectivity(slab.model, qq, energy, pn)
            yd, ed = TRANSFORM(qq, rr, ee)
            yb1, _ = TRANSFORM(qq, yb_m)
            ys1, _ = TRANSFORM(qq, ys_m)
            rb = sp.rms_logy(qq, rr, ee, yb_m)
            rs = sp.rms_logy(qq, rr, ee, ys_m)
            ax.errorbar(
                qq,
                yd,
                yerr=ed,
                fmt="o",
                ms=2.0,
                lw=0.4,
                color=COLOR_DATA,
                ecolor="0.65",
                elinewidth=0.35,
                capsize=0,
                rasterized=True,
                zorder=1,
            )
            ax.plot(qq, yb1, color=COLOR_BOOK, lw=1.2, zorder=3)
            ax.plot(qq, ys1, color=COLOR_SLAB, lw=1.1, ls="--", zorder=4)
            ax.text(
                0.98,
                0.94,
                f"{energy:g} eV {pn}\nbook={rb:.2f}\nslab={rs:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.6,
            )
            if col == 0:
                ax.set_ylabel(r"$\log_{10} R$")
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    axes[0, 0].plot([], [], color=COLOR_BOOK, lw=1.2, label="bookended (FIG4 re-polish)")
    axes[0, 0].plot(
        [], [], color=COLOR_SLAB, lw=1.1, ls="--", label="DFT-slab (FIG4 re-polish)"
    )
    axes[0, 0].legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle(
        "FIG4 cores re-polished on full E_RES including 283.7 eV (logY)\n"
        f"book nll={book_meta['nll_fit']:.1f} RMS={book_meta['rms_fit']:.2f}  |  "
        f"slab nll={slab_meta['nll_fit']:.1f} RMS={slab_meta['rms_fit']:.2f}",
        fontsize=9.5,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {stem.with_suffix('.png')}")


def load_bookended():
    with BOOK_SEED.open("rb") as fh:
        return pickle.load(fh)["gpu_objective"]


def load_slab():
    ed = _load("ed_full", _HERE / "_eres_dynesty_gpu.py")
    return ed.load_slab(contract="polish")


def run_one(label: str) -> dict:
    out = OUT_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== {label} full E_RES logY polish (incl. 283.7) ===")
    if label == "bookended":
        objective = load_bookended()
        kind = "bookended"
        color = COLOR_BOOK
        seed_path = BOOK_SEED
    else:
        objective = load_slab()
        kind = "dft_slab"
        color = COLOR_SLAB
        seed_path = SLAB_SEED

    n = restrict_energies(objective, E_RES)
    keep_fig4_free_set(objective, kind=kind)
    prep(objective)
    log(
        f"{label}: n={n}  free={len(objective.varying_parameters())}  "
        f"seed={seed_path.name}"
    )
    for p in objective.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"[{float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g}]"
        )

    fit = multi_start_lbfgs(objective, n_extra=6, seed=5 if label == "bookended" else 9)
    per = per_energy_rms(objective)
    plot_fit(
        objective,
        title=f"{label} FIG4 re-polish on full E_RES (incl. 283.7 eV)",
        color=color,
        stem=out / "fit_quality_logY",
        meta=fit,
    )

    summary = {
        "label": label,
        "energies": list(E_RES),
        "transform": "logY",
        "seed": str(seed_path.resolve()),
        "fit": fit,
        "per_energy": per,
        "protocol": "fig4_cores_full_E_RES_logY_lbfgs",
    }
    with (out / "fit.pkl").open("wb") as fh:
        pickle.dump({"objective": objective, "summary": summary}, fh)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(
        f"{label}: nll {fit['nll_seed']:.2f} -> {fit['nll_fit']:.2f}  "
        f"rms={fit['rms_fit']:.3f}"
    )
    return {"objective": objective, "summary": summary, "fit": fit}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=["bookended", "dft_slab"],
        choices=["bookended", "dft_slab"],
    )
    p.add_argument("--parallel", action="store_true")
    return p.parse_args(argv)


def _spawn_parallel(args) -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    cwd = script.parents[2]
    procs = []
    for label in args.models:
        log_path = OUT_ROOT / f"{label}.log"
        cmd = [sys.executable, str(script), "--models", label]
        log(f"parallel spawn: {' '.join(cmd)} -> {log_path}")
        fh = log_path.open("w")
        procs.append(
            (
                label,
                subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ),
            )
        )
    rc = 0
    for label, proc in procs:
        code = proc.wait()
        log(f"parallel {label}: exit={code}")
        rc = rc or int(code)

    # Overlay + root summary after both finish.
    if rc == 0 and set(args.models) >= {"bookended", "dft_slab"}:
        with (OUT_ROOT / "bookended" / "fit.pkl").open("rb") as fh:
            book = pickle.load(fh)["objective"]
        with (OUT_ROOT / "dft_slab" / "fit.pkl").open("rb") as fh:
            slab = pickle.load(fh)["objective"]
        book_sum = json.loads((OUT_ROOT / "bookended" / "summary.json").read_text())
        slab_sum = json.loads((OUT_ROOT / "dft_slab" / "summary.json").read_text())
        prep(book)
        prep(slab)
        plot_overlay(
            book,
            slab,
            stem=OUT_ROOT / "fit_quality_overlay_logY",
            book_meta=book_sum["fit"],
            slab_meta=slab_sum["fit"],
        )
        root = {
            "energies": list(E_RES),
            "models": {
                "bookended": {
                    "nll": book_sum["fit"]["nll_fit"],
                    "rms": book_sum["fit"]["rms_fit"],
                    "nvary": book_sum["fit"]["nvary"],
                    "per_energy": book_sum["per_energy"],
                },
                "dft_slab": {
                    "nll": slab_sum["fit"]["nll_fit"],
                    "rms": slab_sum["fit"]["rms_fit"],
                    "nvary": slab_sum["fit"]["nvary"],
                    "per_energy": slab_sum["per_energy"],
                },
            },
        }
        (OUT_ROOT / "summary.json").write_text(json.dumps(root, indent=2))
        log(f"wrote {OUT_ROOT / 'summary.json'}")
    return rc


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.parallel and len(args.models) > 1:
        return _spawn_parallel(args)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for label in args.models:
        results[label] = run_one(label)

    if "bookended" in results and "dft_slab" in results:
        plot_overlay(
            results["bookended"]["objective"],
            results["dft_slab"]["objective"],
            stem=OUT_ROOT / "fit_quality_overlay_logY",
            book_meta=results["bookended"]["fit"],
            slab_meta=results["dft_slab"]["fit"],
        )
        root_summary = {
            "energies": list(E_RES),
            "models": {
                k: {
                    "nll": v["fit"]["nll_fit"],
                    "rms": v["fit"]["rms_fit"],
                    "nvary": v["fit"]["nvary"],
                    "per_energy": v["summary"]["per_energy"],
                }
                for k, v in results.items()
            },
        }
        (OUT_ROOT / "summary.json").write_text(json.dumps(root_summary, indent=2))
        log(f"wrote {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

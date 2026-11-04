"""283.7 eV-only open overfit: bookended vs DFT-slab head-to-head.

Same single-energy protocol as ``_bookended_2837_nc_refit.py`` / surface+oxide
refits, wired into the profile-contract eres objectives: subset the flat
``ReflectDataset`` to ``E≈283.7``, rebuild ``_groups``/``_batches``, free film +
oxide/Si + that energy's instruments only, L-BFGS multi-start. Then score
transfer on the held-out resonant energies.
"""

from __future__ import annotations

import importlib.util
import json
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

ROOT = models_fitting_results / "eres_dynesty_gpu" / "full_profile_contract"
OUT = ROOT / "e283p7_only_overfit"
_HERE = Path(__file__).resolve().parent

ENERGY = 283.7
E_RES = (283.7, 284.2, 285.1, 287.0, 289.0)
TRANSFORM = Transform("logY")
COLOR_BOOK = "#117A65"
COLOR_SLAB = "#C0392B"
COLOR_DATA = "0.25"
BKG_BOUNDS = (0.0, 5e-3)
ANCHOR_TAG = "283p7eV"


def log(msg: str) -> None:
    print(msg, flush=True)


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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def prep_objective(objective) -> None:
    objective.model.device = "cpu"
    objective.model.parallel = False
    objective.transform = TRANSFORM
    objective.anisotropy_weight = 0.0
    if hasattr(objective, "normalization"):
        objective.normalization = "auto"
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()


def restrict_to_energy(objective, energy: float = ENERGY) -> int:
    """Keep only rows at ``energy``; rebuild Objective group/batch caches."""
    ds = objective._dataset
    e = np.asarray(ds.energy, dtype=float)
    keep = np.isclose(e, float(energy))
    n = int(keep.sum())
    if n == 0:
        raise RuntimeError(f"no rows at energy={energy}")
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
    # Drop multi-energy anisotropy targets left over from the polish pickle.
    objective.anisotropy_weight = 0.0
    if hasattr(objective, "anisotropy_targets"):
        objective.anisotropy_targets = None
    if hasattr(objective, "_anisotropy_pairs"):
        objective._anisotropy_pairs = ()
    if hasattr(objective, "normalization"):
        objective.normalization = "auto"
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()
    return n


def freeze_non_anchor_instruments(objective, *, tag: str = ANCHOR_TAG) -> int:
    """Freeze instrument params that are not the anchor-energy channel."""
    n_frozen = 0
    for p in objective.parameters.flattened():
        name = str(p.name)
        is_inst = (
            name.startswith("scale_")
            or name.startswith("theta_offset_")
            or name.startswith("bkg")
        )
        if not is_inst:
            continue
        if tag in name:
            continue
        if p.vary:
            p.vary = False
            n_frozen += 1
    objective.refresh_parameters()
    return n_frozen


def varying_state(objective) -> tuple[list[str], np.ndarray, list[tuple[float, float]]]:
    names: list[str] = []
    x0: list[float] = []
    bounds: list[tuple[float, float]] = []
    for p in objective.varying_parameters():
        names.append(str(p.name))
        x0.append(float(p.value))
        bounds.append((float(p.bounds.lb), float(p.bounds.ub)))
    return names, np.asarray(x0, dtype=float), bounds


def set_varying(objective, names: list[str], x: np.ndarray) -> None:
    by_name = {str(p.name): p for p in objective.varying_parameters()}
    for name, val in zip(names, x, strict=True):
        by_name[name].value = float(val)


def overall_rms(objective) -> float:
    ds = objective._dataset
    y = np.asarray(objective.generative(), dtype=float)
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    yd, ed_arr = TRANSFORM(q, r, r_err)
    ym, _ = TRANSFORM(q, y)
    ed = np.asarray(ed_arr, dtype=float)
    m = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
    return float(np.sqrt(np.mean(((yd[m] - ym[m]) / ed[m]) ** 2)))


def rms_logy(q, r, r_err, y) -> float:
    yd, ed = TRANSFORM(q, r, r_err)
    ym, _ = TRANSFORM(q, y)
    ed = np.asarray(ed, dtype=float)
    m = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
    return float(np.sqrt(np.mean(((yd[m] - ym[m]) / ed[m]) ** 2)))


def reflectivity(model, q, energy, pol) -> np.ndarray:
    result = model(np.asarray(q, dtype=float), float(energy))
    return np.asarray(result.s if pol == "s" else result.p, dtype=float)


def multi_start_lbfgs(objective, *, n_extra: int = 6, seed: int = 1) -> dict:
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
    prep_objective(objective)
    nll_seed = float(objective.nll())

    def nll_fn(x):
        try:
            set_varying(objective, names, x)
            objective.refresh_parameters()
            val = float(objective.nll())
            if not np.isfinite(val):
                return 1.0e12
            return val
        except (ValueError, RuntimeError, FloatingPointError):
            return 1.0e12

    rng = np.random.default_rng(seed)
    candidates = [x_seed.copy()]
    for _ in range(n_extra):
        u = rng.random(len(x_seed))
        xt = lo + u * (hi - lo)
        candidates.append(np.clip(0.75 * x_seed + 0.25 * xt, lo, hi))

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
            options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-5},
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
    prep_objective(objective)
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


def open_slab_anchor(ed, objective, slab_med: dict[str, float]) -> None:
    """Free film slabs + anchor-energy instruments for 283.7-only slab overfit."""
    for p in objective.parameters.flattened():
        name = str(p.name)
        if name in slab_med:
            p.value = float(slab_med[name])
    for p in objective.parameters.flattened():
        name = str(p.name)
        is_film = any(
            name.startswith(f"ZnPc_{layer}.")
            for layer in ("surface", "bulk", "interface")
        )
        if is_film and p.constraint is None:
            if not p.vary:
                p.vary = True
        is_inst = (
            name.startswith("scale_")
            or name.startswith("theta_offset_")
            or name.startswith("bkg")
        )
        if is_inst and ANCHOR_TAG in name:
            if name.startswith("bkg"):
                ed._setp(
                    p,
                    value=max(0.0, float(p.value)),
                    vary=True,
                    bounds=BKG_BOUNDS,
                )
            else:
                if not p.vary:
                    p.vary = True
        elif is_inst:
            p.vary = False
    ed._widen_instrument_bounds(objective)
    # Re-freeze non-anchor after widen (widen may re-open).
    freeze_non_anchor_instruments(objective)
    if hasattr(objective.model, "energy_offset"):
        objective.model.energy_offset.setp(vary=False)
    objective.refresh_parameters()
    log(f"slab 283.7-only open: free={len(objective.varying_parameters())}")


def per_energy_scores(objective, energies=E_RES) -> list[dict]:
    """Score objective on each energy using direct model calls (full dataset)."""
    # Caller must restore a full multi-energy dataset before transfer scoring.
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e_arr = np.asarray(ds.energy, dtype=float)
    rows = []
    for energy in energies:
        row: dict = {"energy": float(energy)}
        for pn in ("s", "p"):
            m = (pol == pn) & np.isclose(e_arr, energy)
            qq, rr, ee = q[m], r[m], r_err[m]
            if qq.size == 0:
                row[f"rms_{pn}"] = None
                row[f"n_{pn}"] = 0
                continue
            order = np.argsort(qq)
            qq, rr, ee = qq[order], rr[order], ee[order]
            y = reflectivity(objective.model, qq, energy, pn)
            row[f"rms_{pn}"] = rms_logy(qq, rr, ee, y)
            row[f"n_{pn}"] = int(qq.size)
        rows.append(row)
    return rows


def plot_2837(
    book,
    slab,
    *,
    stem: Path,
    title: str,
    book_meta: dict,
    slab_meta: dict,
) -> None:
    _rc()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.1), sharey=False, constrained_layout=True
    )
    for ax, pn in zip(axes, ("s", "p"), strict=True):
        ds = book._dataset
        q = np.asarray(ds.q, dtype=float)
        r = np.asarray(ds.r, dtype=float)
        r_err = np.asarray(ds.r_err, dtype=float)
        pol = np.asarray(ds.pol)
        m = pol == pn
        qq, rr, ee = q[m], r[m], r_err[m]
        order = np.argsort(qq)
        qq, rr, ee = qq[order], rr[order], ee[order]
        yb = reflectivity(book.model, qq, ENERGY, pn)
        ys = reflectivity(slab.model, qq, ENERGY, pn)
        yd, ed = TRANSFORM(qq, rr, ee)
        yb1, _ = TRANSFORM(qq, yb)
        ys1, _ = TRANSFORM(qq, ys)
        ax.errorbar(
            qq,
            yd,
            yerr=ed,
            fmt="o",
            ms=2.4,
            lw=0.45,
            color=COLOR_DATA,
            ecolor="0.65",
            elinewidth=0.4,
            capsize=0,
            zorder=1,
            rasterized=True,
            label="data",
        )
        ax.plot(
            qq,
            yb1,
            color=COLOR_BOOK,
            lw=1.35,
            label=f"bookended 283.7-only (RMS={rms_logy(qq, rr, ee, yb):.2f})",
        )
        ax.plot(
            qq,
            ys1,
            color=COLOR_SLAB,
            lw=1.2,
            ls="--",
            label=f"slab 283.7-only (RMS={rms_logy(qq, rr, ee, ys):.2f})",
        )
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(f"{ENERGY:g} eV {pn}-pol")
        if pn == "s":
            ax.set_ylabel(r"$\log_{10} R$")
            ax.legend(loc="lower left", fontsize=7, frameon=False)
    fig.suptitle(
        f"{title}\n"
        f"book nll={book_meta['nll_fit']:.1f} RMS={book_meta['rms_fit']:.2f}  |  "
        f"slab nll={slab_meta['nll_fit']:.1f} RMS={slab_meta['rms_fit']:.2f}",
        fontsize=9.5,
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {stem.with_suffix('.png')}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ed = _load_module("ed_2837", _HERE / "_eres_dynesty_gpu.py")
    br = _load_module("br_2837", _HERE / "_bookended_open_erefits.py")

    book_post = json.loads((ROOT / "bookended" / "posterior_summary.json").read_text())
    slab_post = json.loads((ROOT / "dft_slab" / "posterior_summary.json").read_text())
    book_med = br.medians(book_post)
    slab_med = br.medians(slab_post)

    # --- Bookended: open from profile best, restrict to 283.7, freeze other E ---
    book = ed.load_bookended(contract="profile")
    br.open_all_from_profile_best(ed, book, book_med)
    n_keep = restrict_to_energy(book, ENERGY)
    n_fr = freeze_non_anchor_instruments(book)
    prep_objective(book)
    log(
        f"bookended 283.7-only: n={n_keep}  frozen_other_E_inst={n_fr}  "
        f"free={len(book.varying_parameters())}  seed_nll={float(book.nll()):.2f}"
    )
    log("bookended free params:")
    for p in book.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"[{float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g}]"
        )
    book_fit = multi_start_lbfgs(book, n_extra=8, seed=7)
    log(
        f"bookended fit: nll {book_fit['nll_seed']:.2f} -> {book_fit['nll_fit']:.2f}  "
        f"rms={book_fit['rms_fit']:.3f}  nit={book_fit['nit']}"
    )

    # --- Slab: profile median seed, open film + 283.7 instruments, same subset ---
    slab = ed.load_slab(contract="profile")
    prep_objective(slab)
    open_slab_anchor(ed, slab, slab_med)
    n_keep_s = restrict_to_energy(slab, ENERGY)
    prep_objective(slab)
    log(
        f"slab 283.7-only: n={n_keep_s}  free={len(slab.varying_parameters())}  "
        f"seed_nll={float(slab.nll()):.2f}"
    )
    log("slab free params:")
    for p in slab.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"[{float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g}]"
        )
    slab_fit = multi_start_lbfgs(slab, n_extra=8, seed=11)
    log(
        f"slab fit: nll {slab_fit['nll_seed']:.2f} -> {slab_fit['nll_fit']:.2f}  "
        f"rms={slab_fit['rms_fit']:.3f}  nit={slab_fit['nit']}"
    )

    plot_2837(
        book,
        slab,
        stem=OUT / "fit_quality_2837_only",
        title="283.7 eV-only open overfit (bookended vs DFT-slab)",
        book_meta=book_fit,
        slab_meta=slab_fit,
    )

    # Transfer: restore full datasets, keep fitted geometry/instruments.
    book_full = ed.load_bookended(contract="profile")
    br.open_all_from_profile_best(ed, book_full, book_med)
    prep_objective(book_full)
    # Apply fitted values by name where present.
    by_name = {str(p.name): p for p in book_full.parameters.flattened()}
    for name, val in book_fit["fit_values"].items():
        if name in by_name:
            by_name[name].value = float(val)
    prep_objective(book_full)

    slab_full = ed.load_slab(contract="profile")
    prep_objective(slab_full)
    by_name_s = {str(p.name): p for p in slab_full.parameters.flattened()}
    for name, val in slab_fit["fit_values"].items():
        if name in by_name_s:
            by_name_s[name].value = float(val)
    prep_objective(slab_full)

    transfer_book = per_energy_scores(book_full)
    transfer_slab = per_energy_scores(slab_full)
    log("transfer RMS (bookended fit frozen):")
    for row in transfer_book:
        log(
            f"  {row['energy']:g}  s={row['rms_s']:.3f}  p={row['rms_p']:.3f}"
            if row["rms_s"] is not None
            else f"  {row['energy']:g}  missing"
        )
    log("transfer RMS (slab fit frozen):")
    for row in transfer_slab:
        log(
            f"  {row['energy']:g}  s={row['rms_s']:.3f}  p={row['rms_p']:.3f}"
            if row["rms_s"] is not None
            else f"  {row['energy']:g}  missing"
        )

    summary = {
        "energy": ENERGY,
        "protocol": "2837_only_open_lbfgs_multistart",
        "bookended": book_fit,
        "slab": slab_fit,
        "delta_nll_book_minus_slab": book_fit["nll_fit"] - slab_fit["nll_fit"],
        "winner_2837": (
            "bookended" if book_fit["nll_fit"] < slab_fit["nll_fit"] else "slab"
        ),
        "transfer_bookended": transfer_book,
        "transfer_slab": transfer_slab,
        "paths": {
            "fit_quality": str((OUT / "fit_quality_2837_only.png").resolve()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {OUT / 'summary.json'}")
    log(
        f"283.7 winner={summary['winner_2837']}  "
        f"Δnll(book-slab)={summary['delta_nll_book_minus_slab']:+.2f}"
    )


if __name__ == "__main__":
    main()

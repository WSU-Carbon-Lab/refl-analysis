"""GPU Dynesty on open 12-param + instrument q-truncation cases.

Samples the full open free set (``nvary=32``) for three cuts ``q_hi``, mid,
and ``q_lo``, warm-started from the open-DE best seed. Reuses Dynesty plumbing
from ``_eres_dynesty_gpu.py`` and the open truncation objective builder.

Writes under
``@models/fitting_results/bookended_q_truncation_diffraction_de_open/dynesty/<cut>/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dynesty.utils import resample_equal
from matplotlib.patches import Patch, Rectangle
from scipy.stats import gaussian_kde

from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

_HERE = Path(__file__).resolve().parent
_OPEN_DE = _HERE / "_bookended_q_truncation_diffraction_de_open.py"
_DY = _HERE / "_eres_dynesty_gpu.py"

_open_spec = importlib.util.spec_from_file_location(
    "_bookended_q_truncation_diffraction_de_open", _OPEN_DE
)
if _open_spec is None or _open_spec.loader is None:
    raise ImportError(f"cannot load open DE module from {_OPEN_DE}")
open_de = importlib.util.module_from_spec(_open_spec)
sys.modules[_open_spec.name] = open_de
_open_spec.loader.exec_module(open_de)

_dy_spec = importlib.util.spec_from_file_location("_eres_dynesty_gpu", _DY)
if _dy_spec is None or _dy_spec.loader is None:
    raise ImportError(f"cannot load dynesty module from {_DY}")
dy = importlib.util.module_from_spec(_dy_spec)
sys.modules[_dy_spec.name] = dy
_dy_spec.loader.exec_module(dy)

base = open_de.base
cv = open_de.cv
ENERGIES = open_de.ENERGIES
ANCHOR = open_de.ANCHOR
FILM_FREE = open_de.FILM_FREE
DE_OPEN_DIR = open_de.OUT_DIR
OUT_ROOT = DE_OPEN_DIR / "dynesty"

CUT_NAMES = ("q_hi", "mid", "q_lo")
CUT_COLORS = {"q_hi": "#1f77b4", "mid": "#ff7f0e", "q_lo": "#2ca02c"}
SI_KEYS = ("alpha_si", "density_si", "tau_si")
SI_LABELS = (
    r"$\alpha_{\mathrm{si}}$ (deg)",
    r"$\rho_{\mathrm{si}}$ (g/cm$^3$)",
    r"$\tau_{\mathrm{si}}$ (Å)",
)
FILM_WARM_KEYS = (
    "total_thick",
    "surface_roughness",
    "tau_vac",
    "tau_si",
    "alpha_vac",
    "alpha_si",
    "alpha_bulk",
    "density_vac",
    "density_si",
    "density_bulk",
)


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def resolve_cuts(meta: dict[str, Any]) -> dict[str, float]:
    """Map cut labels to ``q_cut`` values from the open-DE scan summary."""
    q_hi = float(meta["q_hi"])
    q_lo = float(meta["q_lo"])
    return {
        "q_hi": q_hi,
        "mid": 0.5 * (q_hi + q_lo),
        "q_lo": q_lo,
    }


def nearest_agg_row(agg: pd.DataFrame, q_target: float) -> pd.Series:
    i = int((agg["q_cut"] - q_target).abs().to_numpy().argmin())
    return agg.iloc[i]


def best_seed_row(rep: pd.DataFrame, arow: pd.Series) -> pd.Series:
    g = rep.loc[np.isclose(rep["q_cut"].to_numpy(dtype=float), float(arow["q_cut"]))]
    return g.loc[g["seed"] == int(arow["best_seed"])].iloc[0]


def warm_structure_from_row(obj_map: dict[float, Any], brow: pd.Series) -> None:
    """Overwrite film/oxide values with the open-DE best-seed MLE."""
    _vac, film, oxide, _sub = obj_map[ANCHOR].model.structure.components
    for name in FILM_WARM_KEYS:
        getattr(film, name).value = float(brow[name])
    oxide.thick.value = float(brow["oxide_thick"])
    oxide.rough.value = float(brow["oxide_rough"])


def build_cut_objective(
    *,
    q_cut: float,
    summary: dict[str, Any],
    warm_row: pd.Series | None,
) -> Any:
    """Build truncated open ``logY`` objective at ``q_cut``."""
    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    open_de.apply_open_seed(obj_map, summary)
    if warm_row is not None:
        warm_structure_from_row(obj_map, warm_row)
    train = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=float(q_cut))
    train.model.device = "gpu"
    return train


def configure_truncation_objective(objective: Any, *, label: str):
    """GPU dynesty contract keeping truncation ``normalization='energy'``."""
    objective.model.device = "gpu"
    objective.model.parallel = False
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()
    lo, hi, names = dy._finite_bounds(objective)
    x0 = np.array([float(p.value) for p in objective.varying_parameters()], dtype=float)
    nll = float(objective.nll())
    logl0 = float(objective.logl(x0))
    logp0 = float(objective.logp(x0))
    log(
        f"{label}: nvary={len(names)}  nll={nll:.6f}  logl0={logl0:.6f}  "
        f"logp0={logp0:.6f}  device={objective.model.device}  "
        f"norm={getattr(objective, 'normalization', None)}  "
        f"nc={getattr(objective, 'nc_constraint', None)}"
    )
    return lo, hi, names, x0, logl0


def run_probe_cut(
    label: str,
    objective: Any,
    *,
    probe_s: float,
    nlive: int,
    walks: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Short wall-clock Dynesty probe for throughput / nvary checks."""
    lo, hi, names, x0, logl0 = configure_truncation_objective(objective, label=label)
    ms = dy.timed_logl(objective) * 1000.0
    ndim = len(names)
    log(f"{label}: GPU logl ~ {ms:.2f} ms/call  nlive={nlive}  walks={walks}")

    out = out_dir / label
    out.mkdir(parents=True, exist_ok=True)
    prior_transform = dy.make_prior_transform(lo, hi)
    loglike = dy.make_loglike(objective)
    sampler = dy.make_sampler(loglike, prior_transform, ndim, nlive=nlive, walks=walks)

    t0 = time.perf_counter()
    ncall = 0
    nit = 0
    status: dict[str, Any] = {
        "mode": "probe",
        "label": label,
        "ndim": ndim,
        "names": names,
        "nlive": nlive,
        "walks": walks,
        "logl_ms": ms,
        "logl0_mle": logl0,
        "probe_s": probe_s,
        "ncall": 0,
        "nit": 0,
        "logz": None,
        "logzerr": None,
        "stopped": "budget",
        "x0": x0.tolist(),
    }

    for results in sampler.sample_initial(nlive=nlive, dlogz=1.0e-3):
        ncall += int(results.nc)
        nit = int(results.worst_it) + 1
        status["ncall"] = ncall
        status["nit"] = nit
        status["loglstar"] = float(results.loglstar)
        if hasattr(results, "logz"):
            status["logz"] = float(results.logz)
            status["logzerr"] = float(np.sqrt(max(results.logzvar, 0.0)))
        elapsed = time.perf_counter() - t0
        if elapsed >= probe_s:
            status["wall_s"] = elapsed
            status["ncall_per_s"] = ncall / max(elapsed, 1e-9)
            break
    else:
        elapsed = time.perf_counter() - t0
        status["wall_s"] = elapsed
        status["ncall_per_s"] = ncall / max(elapsed, 1e-9)
        status["stopped"] = "sample_initial_exhausted"

    dy._atomic_json(out / "probe_status.json", status)
    log(
        f"{label}: probe done wall={elapsed:.1f}s  ncall={ncall}  "
        f"({status['ncall_per_s']:.0f}/s)  nvary={ndim}"
    )
    return status


def run_full_cut(
    label: str,
    objective: Any,
    *,
    nlive: int,
    walks: int,
    dlogz_init: float,
    n_effective: int,
    out_dir: Path,
    meta_extra: dict[str, Any],
) -> dict[str, Any]:
    """Production DynamicNestedSampler run with checkpoint resume."""
    lo, hi, names, x0, logl0 = configure_truncation_objective(objective, label=label)
    ms = dy.timed_logl(objective) * 1000.0
    ndim = len(names)
    log(
        f"{label}: GPU logl ~ {ms:.2f} ms/call  "
        f"nlive={nlive}  walks={walks}  dlogz_init={dlogz_init}  "
        f"n_effective={n_effective}"
    )

    out = out_dir / label
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pkl"
    status_path = out / "status.json"

    prior_transform = dy.make_prior_transform(lo, hi)
    loglike = dy.make_loglike(objective)
    sampler = dy.make_sampler(loglike, prior_transform, ndim, nlive=nlive, walks=walks)

    status: dict[str, Any] = {
        "mode": "full",
        "label": label,
        "ndim": ndim,
        "names": names,
        "nlive": nlive,
        "walks": walks,
        "sample": dy.SAMPLE,
        "bound": dy.BOUND,
        "dlogz_init": dlogz_init,
        "n_effective": n_effective,
        "logl_ms": ms,
        "logl0_mle": logl0,
        "ncall": 0,
        "nit": 0,
        "logz": None,
        "logzerr": None,
        "phase": "starting",
        "wall_s": 0.0,
        "ncall_per_s": 0.0,
        "stopped": None,
        **meta_extra,
    }
    dy._atomic_json(status_path, status)

    t0 = time.perf_counter()
    last_write = 0.0

    def print_func(results, niter, ncall, nbatch=None, dlogz=None, **_kw):
        nonlocal last_write
        elapsed = time.perf_counter() - t0
        status["wall_s"] = elapsed
        status["ncall"] = int(ncall) if ncall is not None else status["ncall"]
        status["nit"] = int(niter) if niter is not None else status["nit"]
        status["ncall_per_s"] = status["ncall"] / max(elapsed, 1e-9)
        status["nbatch"] = None if nbatch is None else int(nbatch)
        status["dlogz"] = None if dlogz is None else float(dlogz)
        status["phase"] = "initial" if (nbatch is None or int(nbatch) == 0) else "batch"
        try:
            if hasattr(results, "logz") and np.ndim(results.logz) == 0:
                status["logz"] = float(results.logz)
                if hasattr(results, "logzvar"):
                    status["logzerr"] = float(np.sqrt(max(results.logzvar, 0.0)))
            elif hasattr(results, "logz"):
                status["logz"] = float(np.asarray(results.logz)[-1])
                if hasattr(results, "logzerr"):
                    status["logzerr"] = float(np.asarray(results.logzerr)[-1])
        except Exception:
            pass
        if (
            elapsed - last_write >= dy.STATUS_WRITE_S
            or status["nit"] % dy.STATUS_WRITE_NIT == 0
        ):
            last_write = elapsed
            dy._atomic_json(status_path, status)
            log(
                f"  {label} {status['phase']} nit={status['nit']} "
                f"ncall={status['ncall']} "
                f"logZ={status['logz']}±{status['logzerr']} "
                f"wall={elapsed / 60.0:.1f}m  "
                f"({status['ncall_per_s']:.0f}/s)"
            )

    resume = checkpoint.exists()
    if resume:
        log(f"{label}: resuming from {checkpoint}")

    sampler.run_nested(
        nlive_init=nlive,
        dlogz_init=dlogz_init,
        nlive_batch=nlive,
        n_effective=n_effective,
        print_progress=True,
        print_func=print_func,
        checkpoint_file=str(checkpoint),
        checkpoint_every=120,
        resume=resume,
    )

    elapsed = time.perf_counter() - t0
    results = sampler.results
    status["wall_s"] = elapsed
    if hasattr(results, "ncall"):
        status["ncall"] = int(np.sum(results.ncall))
    status["nit"] = int(results.niter)
    status["logz"] = float(results.logz[-1])
    logzerr = float(results.logzerr[-1])
    if not np.isfinite(logzerr):
        # dynesty can leave logzerr as nan when information is ill-conditioned
        logzerr = float("nan")
    status["logzerr"] = logzerr
    status["logl_max"] = float(np.max(results.logl))
    status["ncall_per_s"] = status["ncall"] / max(elapsed, 1e-9)
    status["phase"] = "done"
    status["stopped"] = "run_nested_complete"
    status["n_effective_est"] = float(getattr(results, "n_effective", np.nan))
    dy._atomic_json(status_path, status)

    post = dy.posterior_summary(results, names, x0)
    dy._atomic_json(out / "posterior_summary.json", post)

    summary = {
        **status,
        "posterior_ess": post["ess"],
        "map": post["map"],
        "structural_free": list(open_de.STRUCTURAL_FREE),
        "free_instrument": list(cv.FREE_INSTRUMENT),
    }
    dy._atomic_json(out / "summary.json", summary)

    with (out / "results.pkl").open("wb") as fh:
        pickle.dump(
            {
                "results": results,
                "status": status,
                "names": names,
                "x0_mle": x0,
                "posterior": post,
                "summary": summary,
            },
            fh,
        )

    log(
        f"{label}: DONE wall={elapsed / 3600.0:.2f}h  ncall={status['ncall']}  "
        f"({status['ncall_per_s']:.0f}/s)  logZ={status['logz']:.4f}±"
        f"{status['logzerr']:.4f}  ESS~{post['ess']:.0f}"
    )
    return status


def _disp_si(key: str, val: float) -> float:
    return float(np.rad2deg(val)) if key == "alpha_si" else float(val)


def si_prior_bounds_display() -> dict[str, tuple[float, float]]:
    """Open-contract prior boxes for α_si (deg), ρ_si, τ_si in plot units."""
    polish = base.load_polish_summary()
    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    open_de.apply_open_seed(obj_map, polish)
    train = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=0.27)
    train.refresh_parameters()
    raw = {
        str(p.name): (float(p.bounds.lb), float(p.bounds.ub))
        for p in train.varying_parameters()
    }
    return {k: (_disp_si(k, raw[k][0]), _disp_si(k, raw[k][1])) for k in SI_KEYS}


def load_si_equal_samples(cut_dir: Path, *, n_draw: int = 20_000) -> np.ndarray:
    """Equal-weight posterior draws of ``(alpha_si_deg, density_si, tau_si)``."""
    with (cut_dir / "results.pkl").open("rb") as fh:
        payload = pickle.load(fh)
    names: list[str] = list(payload["names"])
    results = payload["results"]
    idx = [names.index(k) for k in SI_KEYS]
    samples = np.asarray(results.samples, dtype=float)[:, idx]
    logwt = np.asarray(results.logwt, dtype=float)
    logz = float(results.logz[-1])
    weights = np.exp(logwt - logz)
    weights = weights / weights.sum()
    eq = resample_equal(samples, weights, rstate=np.random.default_rng(1))
    if eq.shape[0] > n_draw:
        eq = eq[np.random.default_rng(2).choice(eq.shape[0], n_draw, replace=False)]
    eq = np.asarray(eq, dtype=float)
    eq[:, 0] = np.rad2deg(eq[:, 0])
    return eq


def _contour_level_2d(density: np.ndarray, level: float = 0.68) -> float:
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return float(flat[int(np.searchsorted(csum, level))])


def plot_corner_si_dynesty(
    *,
    dynesty_root: Path | None = None,
    de_open_dir: Path | None = None,
) -> Path:
    """Dynesty-only α/ρ/τ corner with gray prior-bound shading for three cuts."""
    root = Path(dynesty_root) if dynesty_root is not None else OUT_ROOT
    de_dir = Path(de_open_dir) if de_open_dir is not None else DE_OPEN_DIR
    de_summary = json.loads((de_dir / "summary.json").read_text())
    cut_map = resolve_cuts(de_summary)
    prior = si_prior_bounds_display()

    meta: dict[str, dict[str, float]] = {}
    dynesty_samples: dict[str, np.ndarray] = {}
    for name in CUT_NAMES:
        cut_dir = root / name
        if not (cut_dir / "results.pkl").exists():
            raise FileNotFoundError(
                f"missing Dynesty results: {cut_dir / 'results.pkl'}"
            )
        summary = json.loads((cut_dir / "summary.json").read_text())
        meta[name] = {
            "q_cut": float(summary.get("q_cut", cut_map[name])),
            "d": float(summary.get("d_pi_over_q", np.pi / cut_map[name])),
        }
        dynesty_samples[name] = load_si_equal_samples(cut_dir)
        log(
            f"{name}: equal-weight draws={dynesty_samples[name].shape[0]}  "
            f"med={np.median(dynesty_samples[name], axis=0)}  "
            f"prior={ {k: prior[k] for k in SI_KEYS} }"
        )

    lims: dict[str, tuple[float, float]] = {}
    for i, k in enumerate(SI_KEYS):
        plo, phi = prior[k]
        vals = [plo, phi]
        for name in CUT_NAMES:
            col = dynesty_samples[name][:, i]
            vals.extend([float(np.percentile(col, 1)), float(np.percentile(col, 99))])
        lo, hi = float(min(vals)), float(max(vals))
        pad = 0.04 * (hi - lo + 1e-12)
        lims[k] = (lo - pad, hi + pad)

    n = 3
    fig, axes = plt.subplots(
        n, n, figsize=(6.4, 6.2), gridspec_kw={"hspace": 0.08, "wspace": 0.08}
    )
    grid_n = 120
    prior_face = "#b0b0b0"
    prior_alpha = 0.28
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ki, kj = SI_KEYS[i], SI_KEYS[j]
            if j > i:
                ax.axis("off")
                continue
            plo_i, phi_i = prior[ki]
            plo_j, phi_j = prior[kj]
            if i == j:
                ax.axvspan(
                    plo_i,
                    phi_i,
                    color=prior_face,
                    alpha=prior_alpha,
                    zorder=0,
                    lw=0,
                )
                x = np.linspace(lims[ki][0], lims[ki][1], 400)
                ymax = 0.0
                for name in CUT_NAMES:
                    col = dynesty_samples[name][:, i]
                    kde = gaussian_kde(col)
                    y = kde(x)
                    ax.plot(x, y, color=CUT_COLORS[name], lw=1.6, zorder=3)
                    lo68, hi68 = np.percentile(col, [16, 84])
                    m = (x >= lo68) & (x <= hi68)
                    ax.fill_between(
                        x[m],
                        0.0,
                        y[m],
                        color=CUT_COLORS[name],
                        alpha=0.22,
                        zorder=2,
                        lw=0,
                    )
                    ymax = max(ymax, float(y.max()))
                ax.set_xlim(lims[ki])
                ax.set_ylim(0.0, ymax * 1.15)
                ax.set_yticks([])
                if i < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(SI_LABELS[i])
            else:
                ax.add_patch(
                    Rectangle(
                        (plo_j, plo_i),
                        phi_j - plo_j,
                        phi_i - plo_i,
                        facecolor=prior_face,
                        edgecolor="none",
                        alpha=prior_alpha,
                        zorder=0,
                    )
                )
                for name in CUT_NAMES:
                    samp = dynesty_samples[name]
                    xx = samp[:, j]
                    yy = samp[:, i]
                    kde = gaussian_kde(np.vstack([xx, yy]))
                    xg = np.linspace(lims[kj][0], lims[kj][1], grid_n)
                    yg = np.linspace(lims[ki][0], lims[ki][1], grid_n)
                    X, Y = np.meshgrid(xg, yg)
                    dens = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    lev = _contour_level_2d(dens, 0.68)
                    ax.contourf(
                        X,
                        Y,
                        dens,
                        levels=[lev, dens.max()],
                        colors=[CUT_COLORS[name]],
                        alpha=0.22,
                        zorder=2,
                    )
                    ax.contour(
                        X,
                        Y,
                        dens,
                        levels=[lev],
                        colors=[CUT_COLORS[name]],
                        linewidths=1.5,
                        zorder=4,
                    )
                    ax.plot(
                        float(np.median(xx)),
                        float(np.median(yy)),
                        "o",
                        ms=3.0,
                        color=CUT_COLORS[name],
                        zorder=5,
                    )
                ax.set_xlim(lims[kj])
                ax.set_ylim(lims[ki])
                if i < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(SI_LABELS[j])
                if j > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(SI_LABELS[i])

    handles = [
        Patch(
            facecolor=prior_face,
            edgecolor="none",
            alpha=prior_alpha,
            label="prior bounds",
        ),
        *[
            Patch(
                facecolor=CUT_COLORS[name],
                edgecolor=CUT_COLORS[name],
                alpha=0.45,
                label=(
                    rf"{name}: $q_{{\mathrm{{cut}}}}$={meta[name]['q_cut']:.3f} "
                    rf"($d$={meta[name]['d']:.1f} Å)"
                ),
            )
            for name in CUT_NAMES
        ],
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        fontsize=8,
        title=r"Dynesty 68% (open nvary=32)",
        title_fontsize=8,
    )
    fig.suptitle(
        r"Substrate bookend: $\alpha_{\mathrm{si}}$, $\rho_{\mathrm{si}}$, $\tau_{\mathrm{si}}$"
        "\nmarginalized Dynesty posteriors; gray = open prior box",
        fontsize=10,
        y=1.02,
    )
    out_png = de_dir / "corner_si_dynesty.png"
    out_pdf = de_dir / "corner_si_dynesty.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out_png}")
    log(f"wrote {out_pdf}")
    return out_png


def plot_corner_si_dynesty_vs_toy(
    *,
    dynesty_root: Path | None = None,
    de_open_dir: Path | None = None,
) -> Path:
    """Backward-compatible alias: Dynesty-only corner with prior shading."""
    return plot_corner_si_dynesty(dynesty_root=dynesty_root, de_open_dir=de_open_dir)


def _apply_map_values(objective: Any, map_vals: dict[str, float]) -> None:
    for p in objective.varying_parameters():
        name = str(p.name)
        if name in map_vals:
            p.value = float(map_vals[name])


def plot_reflectivity_three_cuts(
    *,
    dynesty_root: Path | None = None,
    de_open_dir: Path | None = None,
) -> Path:
    """283.7 eV log10 R for Dynesty MAP models at q_hi / mid / q_lo."""
    root = Path(dynesty_root) if dynesty_root is not None else OUT_ROOT
    de_dir = Path(de_open_dir) if de_open_dir is not None else DE_OPEN_DIR
    de_summary = json.loads((de_dir / "summary.json").read_text())
    polish = base.load_polish_summary()
    cut_map = resolve_cuts(de_summary)
    agg = pd.read_parquet(de_dir / "scan.parquet")
    rep = pd.read_parquet(de_dir / "replicates.parquet")

    curves: dict[str, dict[str, Any]] = {}
    for name in CUT_NAMES:
        cut_dir = root / name
        summary = json.loads((cut_dir / "summary.json").read_text())
        q_cut = float(summary["q_cut"])
        arow = nearest_agg_row(agg, q_cut)
        brow = best_seed_row(rep, arow)
        objectives = cv.seed_objectives()
        obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
        open_de.apply_open_seed(obj_map, polish)
        warm_structure_from_row(obj_map, brow)
        train = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=q_cut)
        configure_truncation_objective(train, label=f"{name}:map")
        _apply_map_values(train, summary["map"])
        cur = base.full_q_curves(obj_map, train, energy=ANCHOR)
        curves[name] = {
            **cur,
            "q_cut": q_cut,
            "d": float(summary["d_pi_over_q"]),
        }
        log(f"{name}: reflectivity MAP at q_cut={q_cut:.4f}")

    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.4), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    ref = curves["q_hi"]
    for ax, pol_label, pol_code in zip(axes, ("s", "p"), ("s", "p"), strict=True):
        mref = np.asarray(ref["pol"]) == pol_code
        q0 = ref["q"][mref]
        order0 = np.argsort(q0)
        ax.errorbar(
            q0[order0],
            ref["logR"][mref][order0],
            yerr=ref["logR_err"][mref][order0],
            fmt="o",
            ms=2.0,
            lw=0.4,
            color="0.55",
            alpha=0.7,
            zorder=1,
            label="data",
        )
        for name in CUT_NAMES:
            cur = curves[name]
            m = np.asarray(cur["pol"]) == pol_code
            q = cur["q"][m]
            order = np.argsort(q)
            color = CUT_COLORS[name]
            ax.plot(
                q[order],
                cur["logR_fit"][m][order],
                color=color,
                lw=1.4,
                zorder=3,
                label=(
                    rf"{name}: $q_{{\mathrm{{cut}}}}$={cur['q_cut']:.3f} "
                    rf"($d$={cur['d']:.1f} Å)"
                ),
            )
            ax.axvline(cur["q_cut"], color=color, ls="--", lw=0.8, alpha=0.7, zorder=0)
        ax.set_xlabel(r"$q$ (Å$^{-1}$)")
        ax.set_title(rf"{pol_label}-pol")
        ax.set_xlim(left=0.0)
    axes[0].set_ylabel(r"$\log_{10} R$")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(
        r"Dynesty MAP fits at three $q_{\mathrm{cut}}$ (283.7 eV; open nvary=32)",
        fontsize=10,
        y=1.02,
    )
    out_png = de_dir / "reflectivity_q_cuts_dynesty.png"
    out_pdf = de_dir / "reflectivity_q_cuts_dynesty.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    cache = de_dir / "reflectivity_q_cuts_dynesty_curves.npz"
    np.savez_compressed(
        cache,
        cuts=np.asarray(list(CUT_NAMES)),
        q_cuts=np.asarray([curves[n]["q_cut"] for n in CUT_NAMES], dtype=float),
        **{f"q_{i}": curves[n]["q"] for i, n in enumerate(CUT_NAMES)},
        **{f"logR_{i}": curves[n]["logR"] for i, n in enumerate(CUT_NAMES)},
        **{f"logR_err_{i}": curves[n]["logR_err"] for i, n in enumerate(CUT_NAMES)},
        **{f"logR_fit_{i}": curves[n]["logR_fit"] for i, n in enumerate(CUT_NAMES)},
        **{f"pol_{i}": curves[n]["pol"] for i, n in enumerate(CUT_NAMES)},
    )
    log(f"wrote {out_png}")
    log(f"wrote {out_pdf}")
    log(f"wrote {cache}")
    return out_png


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynesty open q-truncation (12 structural + instruments)"
    )
    p.add_argument(
        "--cut",
        choices=[*CUT_NAMES, "all"],
        default="all",
        help="which q_cut case(s) to sample",
    )
    p.add_argument(
        "--probe-minutes",
        type=float,
        default=0.0,
        help="if >0, run a wall-clock probe instead of full nested sampling",
    )
    p.add_argument("--nlive", type=int, default=0, help="0 => ndim-scaled default")
    p.add_argument("--walks", type=int, default=0, help="0 => ndim-scaled default")
    p.add_argument("--dlogz-init", type=float, default=dy.DLOGZ_INIT)
    p.add_argument("--n-effective", type=int, default=dy.N_EFFECTIVE)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--plot-corner",
        action="store_true",
        help="rebuild Dynesty α/ρ/τ corner (prior-shaded) and exit",
    )
    p.add_argument(
        "--plot-reflectivity",
        action="store_true",
        help="rebuild three-cut Dynesty MAP reflectivity figure and exit",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.plot_corner:
        plot_corner_si_dynesty(dynesty_root=out_root)
        return
    if args.plot_reflectivity:
        plot_reflectivity_three_cuts(dynesty_root=out_root)
        return

    de_summary = json.loads((DE_OPEN_DIR / "summary.json").read_text())
    agg = pd.read_parquet(DE_OPEN_DIR / "scan.parquet")
    rep = pd.read_parquet(DE_OPEN_DIR / "replicates.parquet")
    polish = base.load_polish_summary()
    cut_map = resolve_cuts(de_summary)

    labels = list(CUT_NAMES) if args.cut == "all" else [args.cut]
    log(f"cuts={labels}  out={out_root}")
    log(f"q targets: { {k: f'{cut_map[k]:.6f}' for k in labels} }")

    for label in labels:
        q_target = cut_map[label]
        arow = nearest_agg_row(agg, q_target)
        brow = best_seed_row(rep, arow)
        q_cut = float(arow["q_cut"])
        d_cut = float(arow["d_pi_over_q"])
        log(
            f"=== {label}: q_cut={q_cut:.6f}  d={d_cut:.4f} A  "
            f"best_seed={int(arow['best_seed'])}  nll_de={float(brow['nll']):.4f}"
        )
        objective = build_cut_objective(q_cut=q_cut, summary=polish, warm_row=brow)
        lo, hi, names, x0, logl0 = configure_truncation_objective(
            objective, label=f"{label}:setup"
        )
        del lo, hi, x0, logl0
        ndim = len(names)
        if ndim != 32:
            raise RuntimeError(f"{label}: expected nvary=32, got {ndim}: {names}")
        nlive = dy.nlive_for_ndim(ndim, args.nlive if args.nlive > 0 else None)
        walks = dy.walks_for_ndim(ndim, args.walks if args.walks > 0 else None)

        meta_extra = {
            "q_cut": q_cut,
            "d_pi_over_q": d_cut,
            "best_seed": int(arow["best_seed"]),
            "de_nll": float(brow["nll"]),
            "cut_label": label,
            "protocol": "q_truncation_dynesty_open_12struct",
            "de_open_dir": str(DE_OPEN_DIR.resolve()),
        }

        if args.probe_minutes > 0:
            run_probe_cut(
                label,
                objective,
                probe_s=float(args.probe_minutes) * 60.0,
                nlive=nlive,
                walks=walks,
                out_dir=out_root,
            )
            continue

        run_full_cut(
            label,
            objective,
            nlive=nlive,
            walks=walks,
            dlogz_init=float(args.dlogz_init),
            n_effective=int(args.n_effective),
            out_dir=out_root,
            meta_extra=meta_extra,
        )

    log("all requested cuts finished")
    if args.cut == "all" and all(
        (out_root / name / "results.pkl").exists() for name in CUT_NAMES
    ):
        plot_corner_si_dynesty(dynesty_root=out_root)
        plot_reflectivity_three_cuts(dynesty_root=out_root)


if __name__ == "__main__":
    main()

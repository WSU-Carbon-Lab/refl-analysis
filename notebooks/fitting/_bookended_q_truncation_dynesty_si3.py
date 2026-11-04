"""3-parameter Dynesty on α_si, ρ_si, τ_si for three q-truncation cuts.

Freezes all other parameters at the Fig 4 open multi-energy ``logY`` polish
MLE (``bookended_multienergy_logY_polish_open``), not at open-DE seeds.
Likelihood matches the truncated multi-energy ``logY`` objective.

Writes under
``@models/fitting_results/bookended_q_truncation_diffraction_de_open/dynesty_si3/<cut>/``.
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

import numpy as np
from refnx.analysis import Transform
from scipy.special import erf

from utils import models_root
from utils.helpers.fitting_helper import rxr
from utils.models import configure_refloxide_fitting
from utils.refloxide_view import load_refloxide_fit, objective_view_at

configure_refloxide_fitting()

_HERE = Path(__file__).resolve().parent
_OPEN_DE = _HERE / "_bookended_q_truncation_diffraction_de_open.py"
_DY = _HERE / "_eres_dynesty_gpu.py"
_DYN_OPEN = _HERE / "_bookended_q_truncation_dynesty_open.py"


def _load_module(name: str, path: Path):
    """Load ``path`` once under ``name`` (stable pickles for dynesty checkpoints)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


open_de = _load_module("_bookended_q_truncation_diffraction_de_open", _OPEN_DE)
dy = _load_module("_eres_dynesty_gpu", _DY)

base = open_de.base
cv = open_de.cv
ENERGIES = open_de.ENERGIES
ANCHOR = open_de.ANCHOR
DE_OPEN_DIR = open_de.OUT_DIR
FIG4_POLISH_DIR = cv.models_fitting_results / "bookended_multienergy_logY_polish_open"
OUT_ROOT = DE_OPEN_DIR / "dynesty_si3"

CUT_NAMES = ("q_hi", "mid", "q_lo")
SI_KEYS = ("alpha_si", "density_si", "tau_si")
CUT_COLORS = {"q_hi": "#0072B2", "mid": "#E69F00", "q_lo": "#009E73"}
CUT_LABELS = {
    "q_hi": r"$q_{\mathrm{hi}}$",
    "mid": r"$q_{\mathrm{mid}}$",
    "q_lo": r"$q_{\mathrm{lo}}$",
}
SI_LABELS = (
    r"$\gamma_{\mathrm{si}}$ (deg)",
    r"$\rho_{\mathrm{si}}$ (g/cm$^3$)",
    r"$\tau_{\mathrm{si}}$ (Å)",
)
TAU_KEY = "tau_si"
C_SLAB = "0.15"
SLAB_LS = "--"
SLAB_LW = 1.0
SLAB_PKL = models_root / "xrr" / "znpc" / "refloxide" / "refit-refloxide.pkl"
FILM_SLAB_IDX = (1, 2, 3)

# Wide priors so the focused 3-param run is not clipped like the open local box.
SI3_BOUNDS = {
    "alpha_si": (0.0, float(np.pi / 2)),
    "density_si": (0.2, 3.5),
    "tau_si": (0.5, 60.0),
}
N_REFLECT_DRAWS = 180


def diffraction_length_A(q_cut: float) -> float:
    """Map a q-cut to the real-space diffraction period ``d = 2π/q`` (Å)."""
    return float(2.0 * np.pi / float(q_cut))


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def load_fig4_summary() -> dict[str, Any]:
    path = FIG4_POLISH_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Fig 4 open polish summary missing: {path}")
    return json.loads(path.read_text())


def apply_si3_seed(obj_map: dict[float, Any], summary: dict[str, Any]) -> Any:
    """Freeze at Fig 4 open polish; free only α_si, ρ_si, τ_si with wide priors."""
    film = open_de.apply_open_seed(obj_map, summary)
    for _energy, obj in obj_map.items():
        cv.freeze_all_parameters(obj)

    for name in SI_KEYS:
        cv.safely_setp_param(
            getattr(film, name),
            value=float(summary["film"][name]),
            vary=True,
            bounds=SI3_BOUNDS[name],
        )
    return film


def build_si3_objective(*, q_cut: float, summary: dict[str, Any]) -> Any:
    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    apply_si3_seed(obj_map, summary)
    train = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=float(q_cut))
    train.model.device = "gpu"
    return train


def configure_si3(objective: Any, *, label: str):
    objective.model.device = "gpu"
    objective.model.parallel = False
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()
    lo, hi, names = dy._finite_bounds(objective)
    if len(names) != 3 or set(names) != set(SI_KEYS):
        raise RuntimeError(f"{label}: expected free={SI_KEYS}, got {names}")
    x0 = np.array([float(p.value) for p in objective.varying_parameters()], dtype=float)
    nll = float(objective.nll())
    logl0 = float(objective.logl(x0))
    logp0 = float(objective.logp(x0))
    log(
        f"{label}: nvary={len(names)}  names={names}  nll={nll:.6f}  "
        f"logl0={logl0:.6f}  logp0={logp0:.6f}  device={objective.model.device}"
    )
    for name, a, b, x in zip(names, lo, hi, x0, strict=True):
        if name == "alpha_si":
            log(
                f"  {name}: value={x:.5g} ({np.rad2deg(x):.2f} deg)  "
                f"bounds=({a:.5g},{b:.5g}) deg=({np.rad2deg(a):.2f},{np.rad2deg(b):.2f})"
            )
        else:
            log(f"  {name}: value={x:.5g}  bounds=({a:.5g},{b:.5g})")
    return lo, hi, names, x0, logl0


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
    lo, hi, names, x0, logl0 = configure_si3(objective, label=label)
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
        "si3_bounds": {k: list(v) for k, v in SI3_BOUNDS.items()},
        "seed_polish": str(FIG4_POLISH_DIR.resolve()),
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
    status["logzerr"] = logzerr if np.isfinite(logzerr) else float("nan")
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
        "free": list(SI_KEYS),
        "protocol": "q_truncation_dynesty_si3_fig4_seed",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynesty SI3 (α_si, ρ_si, τ_si) from Fig 4 polish"
    )
    p.add_argument("--cut", choices=[*CUT_NAMES, "all"], default="all")
    p.add_argument("--nlive", type=int, default=0)
    p.add_argument("--walks", type=int, default=0)
    p.add_argument("--dlogz-init", type=float, default=dy.DLOGZ_INIT)
    p.add_argument("--n-effective", type=int, default=dy.N_EFFECTIVE)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--plot-corner", action="store_true")
    p.add_argument("--plot-depth", action="store_true")
    p.add_argument(
        "--depth-variant",
        choices=["v1", "v2", "v3", "v4", "all"],
        default="all",
        help="which depth-figure layout to write when --plot-depth is set",
    )
    p.add_argument(
        "--plot-si-edu",
        action="store_true",
        help="write educational SI posterior figures (separate / shared-tau / corner)",
    )
    return p.parse_args(argv)


def _disp_si(key: str, val: float) -> float:
    return float(np.rad2deg(val)) if key == "alpha_si" else float(val)


def resolve_cuts(meta: dict[str, Any]) -> dict[str, float]:
    q_hi = float(meta["q_hi"])
    q_lo = float(meta["q_lo"])
    return {"q_hi": q_hi, "mid": 0.5 * (q_hi + q_lo), "q_lo": q_lo}


def load_si_equal_samples(cut_dir: Path, *, n_draw: int = 20_000) -> np.ndarray:
    from dynesty.utils import resample_equal

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


def slab_film_profiles(
    model: Any, idx: tuple[int, ...] = FILM_SLAB_IDX, n: int = 600
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Erf-smoothed γ (deg) and ρ for the 3-slab ZnPc film from vacuum."""
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
    step0 = 0.5 * (1.0 + erf(depth / (np.sqrt(2.0) * r0)))
    step1 = 0.5 * (1.0 + erf((depth - z1) / (np.sqrt(2.0) * r1)))
    step2 = 0.5 * (1.0 + erf((depth - z2) / (np.sqrt(2.0) * r2)))

    def blend(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        return v[0] + (v[1] - v[0]) * step1 + (v[2] - v[1]) * step2

    return depth, np.degrees(blend(alpha)), step0 * blend(rho)


def load_dft_slab_objective() -> Any:
    """Load the FIG4 ML-DFT slab fit used for the static dashed overlay."""
    if not SLAB_PKL.exists():
        raise FileNotFoundError(f"missing DFT slab pickle: {SLAB_PKL}")
    return load_refloxide_fit(SLAB_PKL)


def _contour_level_2d(density: np.ndarray, level: float = 0.68) -> float:
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return float(flat[int(np.searchsorted(csum, level))])


def plot_si3_corner(*, dynesty_root: Path | None = None) -> Path:
    """Publishable SI3 Dynesty corner: 68% bands, median ×, diffraction marks on τ."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde

    from utils.helpers.plotting_helper import set_plotting_defaults

    set_plotting_defaults()
    plt.rcParams.update({"axes.grid": False})

    root = Path(dynesty_root) if dynesty_root is not None else OUT_ROOT
    de_dir = DE_OPEN_DIR
    de_summary = json.loads((de_dir / "summary.json").read_text())
    cut_map = resolve_cuts(de_summary)

    meta: dict[str, dict[str, float]] = {}
    samples: dict[str, np.ndarray] = {}
    for name in CUT_NAMES:
        cut_dir = root / name
        summary = json.loads((cut_dir / "summary.json").read_text())
        q_cut = float(summary.get("q_cut", cut_map[name]))
        meta[name] = {
            "q_cut": q_cut,
            "d": diffraction_length_A(q_cut),
        }
        samples[name] = load_si_equal_samples(cut_dir)
        log(
            f"{name}: draws={samples[name].shape[0]} med={np.median(samples[name], axis=0)}"
        )

    lims: dict[str, tuple[float, float]] = {}
    for i, k in enumerate(SI_KEYS):
        vals: list[float] = []
        for name in CUT_NAMES:
            col = samples[name][:, i]
            vals.extend([float(np.percentile(col, 1)), float(np.percentile(col, 99))])
        if k == TAU_KEY:
            vals.extend(float(meta[name]["d"]) for name in CUT_NAMES)
        lo, hi = float(min(vals)), float(max(vals))
        pad = 0.08 * (hi - lo + 1e-12)
        lims[k] = (lo - pad, hi + pad)

    n = 3
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(4.8, 4.7),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    grid_n = 140

    def _mark_diffraction(ax, *, vertical: bool) -> None:
        for name in CUT_NAMES:
            d = meta[name]["d"]
            kw = dict(
                color=CUT_COLORS[name],
                ls="--",
                lw=0.9,
                alpha=0.85,
                zorder=6,
            )
            if vertical:
                ax.axvline(d, **kw)
            else:
                ax.axhline(d, **kw)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ki, kj = SI_KEYS[i], SI_KEYS[j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                x = np.linspace(lims[ki][0], lims[ki][1], 400)
                ymax = 0.0
                for name in CUT_NAMES:
                    col = samples[name][:, i]
                    y = gaussian_kde(col)(x)
                    ax.plot(x, y, color=CUT_COLORS[name], lw=1.35, zorder=3)
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
                if ki == TAU_KEY:
                    _mark_diffraction(ax, vertical=True)
                ax.set_xlim(lims[ki])
                ax.set_ylim(0.0, ymax * 1.18)
                ax.set_yticks([])
                if i < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(SI_LABELS[i])
            else:
                for name in CUT_NAMES:
                    samp = samples[name]
                    xx, yy = samp[:, j], samp[:, i]
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
                        alpha=0.20,
                        zorder=2,
                    )
                    ax.contour(
                        X,
                        Y,
                        dens,
                        levels=[lev],
                        colors=[CUT_COLORS[name]],
                        linewidths=1.25,
                        zorder=4,
                    )
                    ax.plot(
                        float(np.median(xx)),
                        float(np.median(yy)),
                        marker="x",
                        ms=5.5,
                        mew=1.35,
                        color=CUT_COLORS[name],
                        linestyle="none",
                        zorder=5,
                    )
                if kj == TAU_KEY:
                    _mark_diffraction(ax, vertical=True)
                if ki == TAU_KEY:
                    _mark_diffraction(ax, vertical=False)
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
        *[
            Line2D(
                [],
                [],
                color=CUT_COLORS[name],
                marker="x",
                markersize=5.5,
                markeredgewidth=1.35,
                linestyle="-",
                linewidth=1.35,
                label=(
                    rf"{CUT_LABELS[name]}  "
                    rf"$d_{{2\pi/q}}={meta[name]['d']:.1f}$ Å"
                ),
            )
            for name in CUT_NAMES
        ],
        Line2D(
            [],
            [],
            color="0.35",
            ls="--",
            lw=0.9,
            label=r"$d=2\pi/q_{\mathrm{cut}}$",
        ),
    ]
    axes[0, -1].legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=7,
        handlelength=1.6,
        borderaxespad=0.0,
        labelspacing=0.45,
    )
    out_png = de_dir / "corner_si_dynesty_si3.png"
    out_pdf = de_dir / "corner_si_dynesty_si3.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out_png}")
    return out_png


def _si3_reflectivity_bands(
    *,
    root: Path,
    obj_map: dict[float, Any],
    energies: tuple[float, ...] | list[float],
    n_draw: int = N_REFLECT_DRAWS,
) -> dict[str, dict[str, Any]]:
    """Posterior predictive log10 R bands for each cut, energy, and polarization."""
    from dynesty.utils import resample_equal

    transform = Transform("logY")
    energies_f = [float(e) for e in energies]
    out: dict[str, dict[str, Any]] = {}

    for name in CUT_NAMES:
        cut_dir = root / name
        summary = json.loads((cut_dir / "summary.json").read_text())
        with (cut_dir / "results.pkl").open("rb") as fh:
            blob = pickle.load(fh)
        names = list(blob["names"])
        results = blob["results"]
        samples = np.asarray(results.samples, dtype=float)
        logwt = np.asarray(results.logwt, dtype=float)
        logz = float(results.logz[-1])
        wt = np.exp(logwt - logz)
        wt = wt / wt.sum()
        rng = np.random.default_rng(11 + hash(name) % 997)
        eq = resample_equal(samples, wt, rstate=rng)
        if eq.shape[0] > n_draw:
            eq = eq[rng.choice(eq.shape[0], n_draw, replace=False)]

        q_cut = float(summary.get("q_cut"))
        train = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=q_cut)
        train.model.device = "cpu"
        train.model.parallel = True
        if hasattr(train, "nc_constraint"):
            train.nc_constraint = True
        train._refresh_data_transform_cache()
        train.refresh_parameters()
        film = obj_map[ANCHOR].model.structure.components[1]
        idx = {n: names.index(n) for n in SI_KEYS}

        def apply_row(row: np.ndarray) -> None:
            film.alpha_si.value = float(row[idx["alpha_si"]])
            film.density_si.value = float(row[idx["density_si"]])
            film.tau_si.value = float(row[idx["tau_si"]])
            train.refresh_parameters()

        map_vals = summary.get("map") or {}
        map_row = np.array([float(map_vals[n]) for n in names], dtype=float)
        apply_row(map_row)

        full = base.build_logY_objective(obj_map, q_min=base.Q_MIN, model=train.model)
        y_fit0 = np.asarray(full.generative(), dtype=float)
        ds = full._dataset
        q_all = np.asarray(ds.q, dtype=float)
        r_all = np.asarray(ds.r, dtype=float)
        e_all = np.asarray(ds.r_err, dtype=float)
        pol_all = np.asarray(ds.pol)
        en_all = np.asarray(ds.energy, dtype=float)

        grids: dict[tuple[float, str], np.ndarray] = {}
        data: dict[tuple[float, str], dict[str, np.ndarray]] = {}
        for energy in energies_f:
            for pol in ("s", "p"):
                m = np.isclose(en_all, energy) & (pol_all == pol)
                order = np.argsort(q_all[m])
                qq = q_all[m][order]
                yd, ed = transform(qq, r_all[m][order], e_all[m][order])
                yf, _ = transform(qq, y_fit0[m][order])
                key = (energy, pol)
                grids[key] = qq
                data[key] = {
                    "q": qq,
                    "logR": np.asarray(yd, dtype=float),
                    "logR_err": np.asarray(ed, dtype=float),
                    "logR_map": np.asarray(yf, dtype=float),
                }

        band_stacks: dict[tuple[float, str], list[np.ndarray]] = {
            (energy, pol): [] for energy in energies_f for pol in ("s", "p")
        }
        for row in eq:
            apply_row(row)
            full_i = base.build_logY_objective(
                obj_map, q_min=base.Q_MIN, model=train.model
            )
            y_fit = np.asarray(full_i.generative(), dtype=float)
            for energy in energies_f:
                for pol in ("s", "p"):
                    key = (energy, pol)
                    m = np.isclose(en_all, energy) & (pol_all == pol)
                    order = np.argsort(q_all[m])
                    qq = q_all[m][order]
                    yf, _ = transform(qq, y_fit[m][order])
                    band_stacks[key].append(
                        np.interp(grids[key], qq, np.asarray(yf, dtype=float))
                    )

        bands: dict[tuple[float, str], dict[str, np.ndarray]] = {}
        for key, rows in band_stacks.items():
            stack = np.vstack(rows)
            bands[key] = {
                "q": grids[key],
                "p025": np.percentile(stack, 2.5, axis=0),
                "p16": np.percentile(stack, 16, axis=0),
                "p50": np.percentile(stack, 50, axis=0),
                "p84": np.percentile(stack, 84, axis=0),
                "p975": np.percentile(stack, 97.5, axis=0),
            }

        out[name] = {
            "q_cut": q_cut,
            "d": diffraction_length_A(q_cut),
            "data": data,
            "bands": bands,
        }
        log(
            f"{name}: reflectivity bands n_draw={eq.shape[0]}  "
            f"q_cut={q_cut:.4f}  n_energy={len(energies_f)}"
        )
    return out


def plot_si3_depth(
    *,
    dynesty_root: Path | None = None,
    n_draw: int = 800,
    n_depth: int = 400,
    n_reflect: int = N_REFLECT_DRAWS,
    energy_offset_decade: float = -4.5,
    pol_offset_decade: float = 0.35,
    data_every: int = 2,
    data_markersize: float = 3.0,
    variant: str = "v1",
) -> Path:
    """SI3 FIG4-ramp figure: R(q) + γ/ρ depth; layout variants match struct12 v2.

    Parameters
    ----------
    variant
        ``v1`` depth-panel SI insets; ``v2`` SI insets on the R free space;
        ``v3`` SI contours on the depth axes (τ_si on the real depth scale);
        ``v4`` clean R + depth with no SI insets or overlays.
    """
    import matplotlib.pyplot as plt
    from dynesty.utils import resample_equal
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde

    from utils.helpers.plotting_helper import set_plotting_defaults

    if variant not in {"v1", "v2", "v3", "v4"}:
        raise ValueError(f"unknown variant {variant!r}; expected v1|v2|v3|v4")

    dexp = _load_module(
        "_plot_bookended_dynesty_exports",
        _HERE / "_plot_bookended_dynesty_exports.py",
    )

    set_plotting_defaults()
    plt.rcParams.update({"axes.grid": False})

    root = Path(dynesty_root) if dynesty_root is not None else OUT_ROOT
    de_summary = json.loads((DE_OPEN_DIR / "summary.json").read_text())
    cut_map = resolve_cuts(de_summary)
    fig4 = load_fig4_summary()
    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    apply_si3_seed(obj_map, fig4)
    _vac, film, _oxide, _sub = obj_map[ANCHOR].model.structure.components
    energies = [float(e) for e in ENERGIES]

    A2NM = 0.1
    L_ref = float(film.total_thick.value)
    L_nm = L_ref * A2NM
    s = np.linspace(0.0, 1.0, n_depth)
    depth_nm = s * L_nm

    profiles: dict[str, dict[str, Any]] = {}
    corner: dict[str, np.ndarray] = {}
    for name in CUT_NAMES:
        cut_dir = root / name
        summary = json.loads((cut_dir / "summary.json").read_text())
        with (cut_dir / "results.pkl").open("rb") as fh:
            blob = pickle.load(fh)
        names = list(blob["names"])
        results = blob["results"]
        samples = np.asarray(results.samples, dtype=float)
        logwt = np.asarray(results.logwt, dtype=float)
        logz = float(results.logz[-1])
        wt = np.exp(logwt - logz)
        wt = wt / wt.sum()
        rng = np.random.default_rng(7 + hash(name) % 1000)
        eq = resample_equal(samples, wt, rstate=rng)
        if eq.shape[0] > n_draw:
            eq = eq[rng.choice(eq.shape[0], n_draw, replace=False)]
        idx = {n: names.index(n) for n in SI_KEYS}
        gamma_stack = np.empty((eq.shape[0], n_depth), dtype=float)
        rho_stack = np.empty_like(gamma_stack)
        for i, row in enumerate(eq):
            film.alpha_si.value = float(row[idx["alpha_si"]])
            film.density_si.value = float(row[idx["density_si"]])
            film.tau_si.value = float(row[idx["tau_si"]])
            z = s * L_ref
            gamma_stack[i] = np.degrees(np.asarray(film.orientation(z), dtype=float))
            rho_stack[i] = dexp.nevot_croce_density(film, z)
        q_cut = float(summary.get("q_cut", cut_map[name]))
        d = diffraction_length_A(q_cut)
        corner[name] = np.column_stack(
            [
                np.rad2deg(eq[:, idx["alpha_si"]]),
                eq[:, idx["density_si"]],
                eq[:, idx["tau_si"]],
            ]
        )
        profiles[name] = {
            "q_cut": q_cut,
            "d": d,
            "L": L_ref,
            "s_diff": float(1.0 - d / L_ref),
            "gamma_p025": np.percentile(gamma_stack, 2.5, axis=0),
            "gamma_p16": np.percentile(gamma_stack, 16, axis=0),
            "gamma_p50": np.percentile(gamma_stack, 50, axis=0),
            "gamma_p84": np.percentile(gamma_stack, 84, axis=0),
            "gamma_p975": np.percentile(gamma_stack, 97.5, axis=0),
            "rho_p025": np.percentile(rho_stack, 2.5, axis=0),
            "rho_p16": np.percentile(rho_stack, 16, axis=0),
            "rho_p50": np.percentile(rho_stack, 50, axis=0),
            "rho_p84": np.percentile(rho_stack, 84, axis=0),
            "rho_p975": np.percentile(rho_stack, 97.5, axis=0),
        }
        log(
            f"{name}: L={L_ref:.2f} A  d(2pi/q)={d:.2f} A  "
            f"s_diff(from substrate)={profiles[name]['s_diff']:.3f}"
        )

    reflect = _si3_reflectivity_bands(
        root=root, obj_map=obj_map, energies=energies, n_draw=n_reflect
    )

    draw_order = ("q_lo", "mid", "q_hi")
    fig = plt.figure(figsize=(8.6, 5.7), dpi=300)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.95, 2.05],
        wspace=0.12,
        left=0.09,
        right=0.93,
        top=0.95,
        bottom=0.12,
    )
    ax_R = fig.add_subplot(outer[0, 0])
    depth_gs = outer[0, 1].subgridspec(1, 2, wspace=0.16, width_ratios=[1.0, 1.0])
    ax_g = fig.add_subplot(depth_gs[0, 0])
    ax_rho = fig.add_subplot(depth_gs[0, 1], sharey=ax_g)

    dft_obj = load_dft_slab_objective()
    z_slab, g_slab, r_slab = slab_film_profiles(
        objective_view_at(dft_obj, float(ANCHOR), swap_pol=False).model
    )
    z_slab_nm = (z_slab / float(z_slab[-1])) * L_nm

    def _light_hue(color: str, amount: float = 0.70) -> tuple[float, float, float]:
        r, g, b = to_rgb(color)
        return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)

    def _decade_to_R(log10_vals: np.ndarray) -> np.ndarray:
        return np.power(10.0, np.asarray(log10_vals, dtype=float))

    def _opaque_band(
        ax,
        u,
        lo,
        med,
        hi,
        color: str,
        *,
        vertical: bool = False,
        z0: int,
        lw: float = 1.5,
    ) -> None:
        if vertical:
            ax.fill_betweenx(
                u,
                lo,
                hi,
                facecolor=_light_hue(color),
                edgecolor="none",
                linewidth=0.0,
                antialiased=True,
                alpha=1.0,
                zorder=z0,
            )
            x_med, y_med = med, u
        else:
            ax.fill_between(
                u,
                lo,
                hi,
                facecolor=_light_hue(color),
                edgecolor="none",
                linewidth=0.0,
                antialiased=True,
                alpha=1.0,
                zorder=z0,
            )
            x_med, y_med = u, med
        ax.plot(
            x_med,
            y_med,
            color="white",
            lw=lw + 1.5,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=z0 + 1,
        )
        ax.plot(
            x_med,
            y_med,
            color=color,
            lw=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=z0 + 2,
        )

    fs_ann = float(plt.rcParams["legend.fontsize"])

    def _annotate_lmin_q(
        ax, x: float, color: str, L_min: float, *, z0: int = 50
    ) -> None:
        ax.annotate(
            rf"$L_{{\mathrm{{min}}}}={L_min:.1f}$ Å",
            xy=(x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(-3.5, -2),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="right",
            fontsize=fs_ann,
            color=color,
            zorder=z0,
            clip_on=False,
        )

    def _annotate_lmin_depth(
        ax, z_nm: float, color: str, L_min: float, *, z0: int = 50
    ) -> None:
        ax.annotate(
            rf"$L_{{\mathrm{{min}}}}={L_min:.1f}$ Å",
            xy=(1.0, z_nm),
            xycoords=("axes fraction", "data"),
            xytext=(3.0, -3.0),
            textcoords="offset points",
            rotation=0,
            va="top",
            ha="left",
            fontsize=fs_ann,
            color=color,
            zorder=z0,
            clip_on=False,
        )

    pol_dy = {"s": float(pol_offset_decade), "p": -float(pol_offset_decade)}

    for i, name in enumerate(draw_order):
        ax_R.axvline(
            reflect[name]["q_cut"],
            color=CUT_COLORS[name],
            ls="--",
            lw=0.85,
            alpha=0.8,
            zorder=1,
        )
        _annotate_lmin_q(
            ax_R,
            reflect[name]["q_cut"],
            CUT_COLORS[name],
            reflect[name]["d"],
            z0=60 + i,
        )

    for ie, energy in enumerate(energies):
        dy_e = float(energy_offset_decade) * ie
        for i, name in enumerate(draw_order):
            color = CUT_COLORS[name]
            R = reflect[name]
            zf = 2 + 4 * i
            for pol in ("s", "p"):
                B = R["bands"][(energy, pol)]
                dy = dy_e + pol_dy[pol]
                _opaque_band(
                    ax_R,
                    B["q"],
                    _decade_to_R(B["p16"] + dy),
                    _decade_to_R(B["p50"] + dy),
                    _decade_to_R(B["p84"] + dy),
                    color,
                    vertical=False,
                    z0=zf,
                    lw=1.5,
                )
        Ds = reflect["q_hi"]["data"][(energy, "s")]
        n_lab = max(8, Ds["q"].size // 8)
        y_lab = _decade_to_R(
            float(np.nanmedian(Ds["logR"][:n_lab])) + dy_e + pol_dy["s"] + 0.45
        )
        ax_R.text(
            0.012,
            y_lab,
            rf"{energy:g} eV",
            fontsize=fs_ann,
            color="0.25",
            va="bottom",
            ha="left",
            zorder=230,
            clip_on=False,
            transform=ax_R.get_yaxis_transform(),
        )

    transform = Transform("logY")
    q_data_max = {
        float(e): max(
            float(np.max(reflect["q_hi"]["data"][(float(e), "s")]["q"])),
            float(np.max(reflect["q_hi"]["data"][(float(e), "p")]["q"])),
        )
        for e in energies
    }
    q_hi_max = max(q_data_max.values())
    for ie, energy in enumerate(energies):
        dy_e = float(energy_offset_decade) * ie
        view = objective_view_at(dft_obj, float(energy), swap_pol=False)
        qq_slab = np.linspace(0.002, q_data_max[float(energy)], 400)
        for pol in ("s", "p"):
            rr = np.asarray(rxr(qq_slab, view.model, pol), dtype=float)
            logy, _ = transform(qq_slab, rr)
            yy = np.asarray(logy, dtype=float) + dy_e + pol_dy[pol]
            keep = np.isfinite(yy)
            if not np.any(keep):
                continue
            ax_R.plot(
                qq_slab[keep],
                _decade_to_R(yy[keep]),
                color=C_SLAB,
                ls=SLAB_LS,
                lw=SLAB_LW,
                zorder=80,
                solid_capstyle="round",
            )

    ax_R.set_axisbelow(True)
    for ie, energy in enumerate(energies):
        dy_e = float(energy_offset_decade) * ie
        for pol, marker in (("s", "o"), ("p", "^")):
            D = reflect["q_hi"]["data"][(energy, pol)]
            sl = slice(None, None, int(data_every))
            qq = D["q"][sl]
            logy = D["logR"][sl] + dy_e + pol_dy[pol]
            loge = D["logR_err"][sl]
            yy = _decade_to_R(logy)
            y_lo = _decade_to_R(logy - loge)
            y_hi = _decade_to_R(logy + loge)
            eb = ax_R.errorbar(
                qq,
                yy,
                yerr=np.vstack([yy - y_lo, y_hi - yy]),
                fmt="none",
                ecolor="0.35",
                alpha=0.75,
                zorder=190,
                elinewidth=0.35,
                capsize=0,
            )
            for artist in (*eb[1], *eb[2]):
                artist.set_zorder(190)
            ax_R.plot(
                qq,
                yy,
                linestyle="none",
                marker=marker,
                markersize=float(data_markersize),
                markerfacecolor="0.15",
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=210,
                clip_on=True,
            )

    ax_R.set_yscale("log")
    ax_R.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    ax_R.set_ylabel("Reflectivity (offset)")
    ax_R.set_xlim(0.0, q_hi_max * 1.02)
    ax_R.minorticks_on()
    ax_R.tick_params(labelbottom=True)

    for i, name in enumerate(draw_order):
        color = CUT_COLORS[name]
        p = profiles[name]
        z0 = 2 + 4 * i
        _opaque_band(
            ax_g,
            depth_nm,
            p["gamma_p16"],
            p["gamma_p50"],
            p["gamma_p84"],
            color,
            vertical=True,
            z0=z0,
            lw=1.5,
        )
        _opaque_band(
            ax_rho,
            depth_nm,
            p["rho_p16"],
            p["rho_p50"],
            p["rho_p84"],
            color,
            vertical=True,
            z0=z0,
            lw=1.5,
        )

    ax_g.plot(
        g_slab,
        z_slab_nm,
        color=C_SLAB,
        ls=SLAB_LS,
        lw=SLAB_LW,
        zorder=55,
        solid_capstyle="round",
    )
    ax_rho.plot(
        r_slab,
        z_slab_nm,
        color=C_SLAB,
        ls=SLAB_LS,
        lw=SLAB_LW,
        zorder=55,
        solid_capstyle="round",
    )

    z_top_nm = -0.45
    ax_g.set_ylim(L_nm, z_top_nm)
    ax_rho.set_ylim(L_nm, z_top_nm)
    ax_g.set_xlabel(r"$\gamma$ (deg)")
    ax_rho.set_xlabel(r"$\rho$ (g/cm$^3$)")
    ax_g.set_ylabel("")
    ax_g.tick_params(labelleft=False)
    ax_rho.yaxis.tick_right()
    ax_rho.tick_params(labelleft=False, labelright=True)
    ax_rho.set_ylabel("depth (nm)")
    ax_rho.yaxis.set_label_position("right")

    lims: dict[str, tuple[float, float]] = {}
    for i, k in enumerate(SI_KEYS):
        vals: list[float] = []
        for name in CUT_NAMES:
            col = corner[name][:, i]
            vals.extend([float(np.percentile(col, 1)), float(np.percentile(col, 99))])
        if k == TAU_KEY:
            vals.extend(float(profiles[name]["d"]) for name in CUT_NAMES)
        lo, hi = float(min(vals)), float(max(vals))
        pad = 0.08 * (hi - lo + 1e-12)
        lims[k] = (lo - pad, hi + pad)

    g_vals = [
        float(np.min(g_slab)),
        float(np.max(g_slab)),
        *[float(np.min(profiles[n]["gamma_p16"])) for n in CUT_NAMES],
        *[float(np.max(profiles[n]["gamma_p84"])) for n in CUT_NAMES],
    ]
    g_lo, g_hi = min(g_vals), max(g_vals)
    g_pad = 0.10 * (g_hi - g_lo + 1e-12)
    ax_g.set_xlim(g_lo - g_pad, g_hi + g_pad)
    ax_rho.set_xlim(1.0, 2.0)

    def _mark_diffraction_tau(ax, *, z0: int = 1000) -> None:
        x0, x1 = ax.get_xlim()
        for i, name in enumerate(CUT_NAMES):
            d = float(profiles[name]["d"])
            ax.plot(
                [x0, x1],
                [d, d],
                color=CUT_COLORS[name],
                ls="--",
                lw=0.95,
                solid_capstyle="butt",
                zorder=z0 + i,
                clip_on=True,
            )
        ax.set_xlim(x0, x1)

    def _paint_joint_contours(
        ax, *, samples, keys, key_lims, x_idx, tau_key, y_vals_fn, z_base: int = 30
    ) -> None:
        grid_n = 100
        kj = keys[x_idx]
        xg = np.linspace(key_lims[kj][0], key_lims[kj][1], grid_n)
        tg = np.linspace(key_lims[tau_key][0], key_lims[tau_key][1], grid_n)
        medians: list[tuple[str, float, float]] = []
        for name in draw_order:
            samp = samples[name]
            xx = samp[:, x_idx]
            tau = samp[:, 2]
            kde = gaussian_kde(np.vstack([xx, tau]))
            X, T = np.meshgrid(xg, tg)
            dens = kde(np.vstack([X.ravel(), T.ravel()])).reshape(X.shape)
            Y = y_vals_fn(T)
            lev = _contour_level_2d(dens, 0.68)
            zf = z_base + 2 * list(draw_order).index(name)
            cf = ax.contourf(
                X,
                Y,
                dens,
                levels=[lev, dens.max()],
                colors=[_light_hue(CUT_COLORS[name])],
                zorder=zf,
            )
            for coll in getattr(cf, "collections", []) or []:
                coll.set_alpha(1.0)
                coll.set_zorder(zf)
            if hasattr(cf, "set_alpha"):
                cf.set_alpha(1.0)
            cs = ax.contour(
                X,
                Y,
                dens,
                levels=[lev],
                colors=[CUT_COLORS[name]],
                linewidths=1.1,
                zorder=zf + 1,
            )
            for coll in getattr(cs, "collections", []) or []:
                coll.set_alpha(1.0)
                coll.set_zorder(zf + 1)
            medians.append(
                (
                    name,
                    float(np.median(xx)),
                    float(y_vals_fn(np.asarray([np.median(tau)]))[0]),
                )
            )
        for i, (name, mx, my) in enumerate(reversed(medians)):
            zm = z_base + 40 + i
            ax.plot(
                mx,
                my,
                marker="x",
                ms=5.5,
                mew=2.2,
                color="white",
                linestyle="none",
                zorder=zm,
            )
            ax.plot(
                mx,
                my,
                marker="x",
                ms=5.0,
                mew=1.25,
                color=CUT_COLORS[name],
                linestyle="none",
                zorder=zm + 1,
            )

    def _draw_si_inset_axes(
        host,
        *,
        x_idx: int,
        bounds,
        transform=None,
        sharey=None,
        ylabel: bool = True,
        zorder: float = 70,
    ):
        kw = {}
        if transform is not None:
            kw["transform"] = transform
        if sharey is not None:
            kw["sharey"] = sharey
        ax = host.inset_axes(bounds, **kw)
        ax.set_facecolor("white")
        ax.patch.set_alpha(1.0)
        ax.set_zorder(zorder)
        host.set_zorder(0)
        ax.set_xlim(lims[SI_KEYS[x_idx]])
        ax.set_ylim(lims[TAU_KEY])
        _paint_joint_contours(
            ax,
            samples=corner,
            keys=SI_KEYS,
            key_lims=lims,
            x_idx=x_idx,
            tau_key=TAU_KEY,
            y_vals_fn=lambda t: t,
            z_base=30,
        )
        _mark_diffraction_tau(ax, z0=1000)
        ax.set_xlabel(SI_LABELS[x_idx], labelpad=1.0)
        if ylabel:
            ax.set_ylabel(SI_LABELS[2], labelpad=6.0)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(length=2.0, pad=1.5)
        ax.minorticks_on()
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(float(plt.rcParams["axes.linewidth"]))
            spine.set_zorder(zorder + 1)
        return ax

    if variant == "v1":
        _draw_si_inset_axes(ax_g, x_idx=0, bounds=[0.38, 0.76, 0.52, 0.18], zorder=120)
        _draw_si_inset_axes(
            ax_rho, x_idx=1, bounds=[0.32, 0.76, 0.58, 0.18], zorder=120
        )
    elif variant == "v2":
        q0, q1 = 0.085, 0.225
        r0, r1 = 1e-14, 3e-9
        mid = 0.5 * (q0 + q1)
        ax_ig = _draw_si_inset_axes(
            ax_R,
            x_idx=0,
            bounds=[q0, r0, mid - q0, r1 - r0],
            transform=ax_R.transData,
            ylabel=True,
            zorder=220,
        )
        _draw_si_inset_axes(
            ax_R,
            x_idx=1,
            bounds=[mid, r0, q1 - mid, r1 - r0],
            transform=ax_R.transData,
            sharey=ax_ig,
            ylabel=False,
            zorder=220,
        )
        for ax_in in ax_R.child_axes[-2:]:
            ax_in.patch.set_alpha(1.0)
            ax_in.set_facecolor("white")
            ax_in.set_zorder(220)
            for spine in ax_in.spines.values():
                spine.set_zorder(221)
    elif variant == "v3":

        def _tau_si_to_z(tau: np.ndarray) -> np.ndarray:
            return L_nm - np.asarray(tau, dtype=float) * A2NM

        for ax, x_idx in ((ax_g, 0), (ax_rho, 1)):
            _paint_joint_contours(
                ax,
                samples=corner,
                keys=SI_KEYS,
                key_lims=lims,
                x_idx=x_idx,
                tau_key=TAU_KEY,
                y_vals_fn=_tau_si_to_z,
                z_base=66,
            )
        z_si_lab = float(
            np.median([L_nm - np.median(corner[n][:, 2]) * A2NM for n in CUT_NAMES])
        )
        ax_g.text(
            0.02,
            z_si_lab,
            r"$\tau_{\mathrm{si}}$",
            transform=ax_g.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=fs_ann,
            color="0.35",
            zorder=95,
        )

    lmin_z0 = 50 if variant == "v1" else 1000
    for ax in (ax_g, ax_rho):
        ax.axhline(L_nm, color="0.55", ls=":", lw=0.8, zorder=1)
        x0, x1 = ax.get_xlim()
        for i, name in enumerate(draw_order):
            z_lmin = L_nm - float(profiles[name]["d"]) * A2NM
            ax.plot(
                [x0, x1],
                [z_lmin, z_lmin],
                color=CUT_COLORS[name],
                ls="--",
                lw=0.95,
                solid_capstyle="butt",
                zorder=lmin_z0 + i,
            )
        ax.set_xlim(x0, x1)
    for i, name in enumerate(draw_order):
        z_lmin = L_nm - float(profiles[name]["d"]) * A2NM
        _annotate_lmin_depth(
            ax_g, z_lmin, CUT_COLORS[name], profiles[name]["d"], z0=lmin_z0 + 10 + i
        )

    handles = [
        Line2D([], [], color=CUT_COLORS["q_hi"], lw=1.5, label="Gradient"),
        Line2D([], [], color=C_SLAB, ls=SLAB_LS, lw=SLAB_LW, label="Slab"),
        Line2D(
            [],
            [],
            color="0.15",
            marker="o",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            linestyle="none",
            label="s",
        ),
        Line2D(
            [],
            [],
            color="0.15",
            marker="^",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            linestyle="none",
            label="p",
        ),
    ]
    ax_R.legend(
        handles=handles,
        loc="upper right" if variant == "v2" else "lower right",
        ncols=2,
        frameon=False,
        handlelength=1.4,
        labelspacing=0.25,
        borderaxespad=0.35,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    stem = f"depth_profile_si3_dynesty_{variant}"
    out_png = root / f"{stem}.png"
    out_pdf = root / f"{stem}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    if variant == "v1":
        fig.savefig(root / "depth_profile_si3_dynesty.png", dpi=300)
        fig.savefig(root / "depth_profile_si3_dynesty.pdf")
        # Also keep the legacy DE_OPEN_DIR copy that notebooks may reference.
        fig.savefig(DE_OPEN_DIR / "depth_profile_si3_dynesty.png", dpi=300)
        fig.savefig(DE_OPEN_DIR / "depth_profile_si3_dynesty.pdf")
    np.savez_compressed(
        root / f"{stem}.npz",
        z_over_L=s,
        depth_nm=depth_nm,
        depth_A=s * L_ref,
        L_A=L_ref,
        L_nm=L_nm,
        diffraction="2pi/q",
        energies=np.asarray(energies, dtype=float),
        energy_offset_decade=float(energy_offset_decade),
        variant=variant,
        **{
            f"{name}_{key}": profiles[name][key]
            for name in CUT_NAMES
            for key in (
                "gamma_p025",
                "gamma_p16",
                "gamma_p50",
                "gamma_p84",
                "gamma_p975",
                "rho_p025",
                "rho_p16",
                "rho_p50",
                "rho_p84",
                "rho_p975",
                "d",
                "s_diff",
                "q_cut",
                "L",
            )
        },
    )
    plt.close(fig)
    log(f"wrote {out_png}")
    return out_png


def plot_si3_edu_distributions(*, dynesty_root: Path | None = None) -> list[Path]:
    """Educational SI3 posterior panels sized to match v1 depth-figure insets.

    Axes areas match the physical inset sizes on ``depth_profile_si3_dynesty_v1``.
    No legends — compose cut colors externally.

    Writes
    ------
    ``si_edu_tau_gamma`` / ``si_edu_tau_rho``
        Single panels at exact v1 inset axes size.
    ``si_edu_separate``
        Two panels side-by-side, each at inset size, independent τ axes.
    ``si_edu_shared_tau``
        Same two joints sharing τ, each panel at inset size.
    ``si_edu_corner``
        Full 3×3 α_si / ρ_si / τ_si corner (opaque 68% joints + 1D margins).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from scipy.stats import gaussian_kde

    from utils.helpers.plotting_helper import set_plotting_defaults

    set_plotting_defaults()
    plt.rcParams.update({"axes.grid": False})

    root = Path(dynesty_root) if dynesty_root is not None else OUT_ROOT
    de_summary = json.loads((DE_OPEN_DIR / "summary.json").read_text())
    cut_map = resolve_cuts(de_summary)

    _fig_w, _fig_h = 8.6, 5.7
    _usable_w = _fig_w * (0.93 - 0.09)
    _usable_h = _fig_h * (0.95 - 0.12)
    _depth_w = _usable_w * (2.05 / 4.0)
    _panel_w = _depth_w / (2.0 + 0.16)
    INSET_AX_WG = 0.52 * _panel_w
    INSET_AX_WR = 0.58 * _panel_w
    INSET_AX_H = 0.18 * _usable_h
    LABEL_L, LABEL_B, LABEL_T, LABEL_R = 0.42, 0.32, 0.06, 0.06
    GAP = 0.10

    meta: dict[str, dict[str, float]] = {}
    samples: dict[str, np.ndarray] = {}
    for name in CUT_NAMES:
        cut_dir = root / name
        if not (cut_dir / "results.pkl").exists():
            raise FileNotFoundError(
                f"missing Dynesty results: {cut_dir / 'results.pkl'}"
            )
        summary = json.loads((cut_dir / "summary.json").read_text())
        q_cut = float(summary.get("q_cut", cut_map[name]))
        meta[name] = {"q_cut": q_cut, "d": diffraction_length_A(q_cut)}
        samples[name] = load_si_equal_samples(cut_dir)

    lims: dict[str, tuple[float, float]] = {}
    for i, k in enumerate(SI_KEYS):
        vals: list[float] = []
        for name in CUT_NAMES:
            col = samples[name][:, i]
            vals.extend([float(np.percentile(col, 1)), float(np.percentile(col, 99))])
        if k == TAU_KEY:
            vals.extend(float(meta[name]["d"]) for name in CUT_NAMES)
        lo, hi = float(min(vals)), float(max(vals))
        pad = 0.08 * (hi - lo + 1e-12)
        lims[k] = (lo - pad, hi + pad)

    draw_order = ("q_lo", "mid", "q_hi")

    def _light_hue(color: str, amount: float = 0.70) -> tuple[float, float, float]:
        r, g, b = to_rgb(color)
        return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)

    def _mark_lmin(ax, *, vertical: bool, z0: int = 50) -> None:
        for i, name in enumerate(CUT_NAMES):
            d = meta[name]["d"]
            if vertical:
                ax.axvline(d, color=CUT_COLORS[name], ls="--", lw=0.95, zorder=z0 + i)
            else:
                ax.axhline(d, color=CUT_COLORS[name], ls="--", lw=0.95, zorder=z0 + i)

    def _paint_joint(ax, *, x_idx: int, y_idx: int, z_base: int = 10) -> None:
        grid_n = 120
        kj, ki = SI_KEYS[x_idx], SI_KEYS[y_idx]
        xg = np.linspace(lims[kj][0], lims[kj][1], grid_n)
        yg = np.linspace(lims[ki][0], lims[ki][1], grid_n)
        medians: list[tuple[str, float, float]] = []
        for name in draw_order:
            samp = samples[name]
            xx, yy = samp[:, x_idx], samp[:, y_idx]
            kde = gaussian_kde(np.vstack([xx, yy]))
            X, Y = np.meshgrid(xg, yg)
            dens = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            lev = _contour_level_2d(dens, 0.68)
            zf = z_base + 2 * list(draw_order).index(name)
            cf = ax.contourf(
                X,
                Y,
                dens,
                levels=[lev, dens.max()],
                colors=[_light_hue(CUT_COLORS[name])],
                zorder=zf,
            )
            for coll in getattr(cf, "collections", []) or []:
                coll.set_alpha(1.0)
                coll.set_zorder(zf)
            if hasattr(cf, "set_alpha"):
                cf.set_alpha(1.0)
            ax.contour(
                X,
                Y,
                dens,
                levels=[lev],
                colors=[CUT_COLORS[name]],
                linewidths=1.15,
                zorder=zf + 1,
            )
            medians.append((name, float(np.median(xx)), float(np.median(yy))))
        for i, (name, mx, my) in enumerate(reversed(medians)):
            zm = z_base + 40 + i
            ax.plot(
                mx,
                my,
                marker="x",
                ms=5.5,
                mew=2.2,
                color="white",
                linestyle="none",
                zorder=zm,
            )
            ax.plot(
                mx,
                my,
                marker="x",
                ms=5.0,
                mew=1.25,
                color=CUT_COLORS[name],
                linestyle="none",
                zorder=zm + 1,
            )
        ax.set_xlim(lims[kj])
        ax.set_ylim(lims[ki])
        if kj == TAU_KEY:
            _mark_lmin(ax, vertical=True, z0=80)
        if ki == TAU_KEY:
            _mark_lmin(ax, vertical=False, z0=80)

    def _style_inset_ax(ax, *, x_idx: int, ylabel: bool) -> None:
        ax.set_xlabel(SI_LABELS[x_idx], labelpad=1.0)
        if ylabel:
            ax.set_ylabel(SI_LABELS[2], labelpad=6.0)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(length=2.0, pad=1.5)
        ax.minorticks_on()
        ax.grid(False)

    def _single_inset_fig(x_idx: int, ax_w: float):
        fig_w = LABEL_L + ax_w + LABEL_R
        fig_h = LABEL_B + INSET_AX_H + LABEL_T
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
        ax = fig.add_axes(
            [LABEL_L / fig_w, LABEL_B / fig_h, ax_w / fig_w, INSET_AX_H / fig_h]
        )
        _paint_joint(ax, x_idx=x_idx, y_idx=2)
        _style_inset_ax(ax, x_idx=x_idx, ylabel=True)
        return fig

    def _pair_inset_fig(*, share_tau: bool):
        mid_pad = LABEL_L if not share_tau else GAP
        fig_w = LABEL_L + INSET_AX_WG + mid_pad + INSET_AX_WR + LABEL_R
        fig_h = LABEL_B + INSET_AX_H + LABEL_T
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
        ax0 = fig.add_axes(
            [
                LABEL_L / fig_w,
                LABEL_B / fig_h,
                INSET_AX_WG / fig_w,
                INSET_AX_H / fig_h,
            ]
        )
        ax1 = fig.add_axes(
            [
                (LABEL_L + INSET_AX_WG + mid_pad) / fig_w,
                LABEL_B / fig_h,
                INSET_AX_WR / fig_w,
                INSET_AX_H / fig_h,
            ],
            sharey=ax0 if share_tau else None,
        )
        for ax, x_idx in ((ax0, 0), (ax1, 1)):
            _paint_joint(ax, x_idx=x_idx, y_idx=2)
            _style_inset_ax(ax, x_idx=x_idx, ylabel=(not share_tau) or (x_idx == 0))
        return fig

    def _save(fig, stem: str) -> Path:
        png = root / f"{stem}.png"
        fig.savefig(png, dpi=300)
        fig.savefig(root / f"{stem}.pdf")
        plt.close(fig)
        log(f"wrote {png}")
        return png

    written: list[Path] = []
    written.append(_save(_single_inset_fig(0, INSET_AX_WG), "si_edu_tau_gamma"))
    written.append(_save(_single_inset_fig(1, INSET_AX_WR), "si_edu_tau_rho"))
    written.append(_save(_pair_inset_fig(share_tau=False), "si_edu_separate"))
    written.append(_save(_pair_inset_fig(share_tau=True), "si_edu_shared_tau"))

    n = 3
    fig3, axes3 = plt.subplots(
        n, n, figsize=(5.0, 4.9), dpi=300, gridspec_kw={"hspace": 0.06, "wspace": 0.06}
    )
    for i in range(n):
        for j in range(n):
            ax = axes3[i, j]
            ki, kj = SI_KEYS[i], SI_KEYS[j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                x = np.linspace(lims[ki][0], lims[ki][1], 400)
                ymax = 0.0
                for name in draw_order:
                    col = samples[name][:, i]
                    y = gaussian_kde(col)(x)
                    ax.fill_between(
                        x,
                        0.0,
                        y,
                        facecolor=_light_hue(CUT_COLORS[name]),
                        edgecolor="none",
                        alpha=1.0,
                        zorder=2 + list(draw_order).index(name),
                        lw=0,
                    )
                    ax.plot(x, y, color=CUT_COLORS[name], lw=1.25, zorder=6)
                    ymax = max(ymax, float(y.max()))
                if ki == TAU_KEY:
                    _mark_lmin(ax, vertical=True, z0=20)
                ax.set_xlim(lims[ki])
                ax.set_ylim(0.0, ymax * 1.18)
                ax.set_yticks([])
                if i < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(SI_LABELS[i])
            else:
                _paint_joint(ax, x_idx=j, y_idx=i)
                if i < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(SI_LABELS[j])
                if j > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(SI_LABELS[i])
    p3 = root / "si_edu_corner.png"
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    fig3.savefig(root / "si_edu_corner.pdf", bbox_inches="tight")
    plt.close(fig3)
    written.append(p3)
    log(f"wrote {p3}")
    return written


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.plot_corner or args.plot_depth or args.plot_si_edu:
        if args.plot_corner:
            plot_si3_corner(dynesty_root=out_root)
        if args.plot_depth:
            variants = (
                ("v1", "v2", "v3", "v4")
                if args.depth_variant == "all"
                else (args.depth_variant,)
            )
            for var in variants:
                plot_si3_depth(dynesty_root=out_root, variant=var)
        if args.plot_si_edu:
            plot_si3_edu_distributions(dynesty_root=out_root)
        return

    fig4 = load_fig4_summary()
    de_summary = json.loads((DE_OPEN_DIR / "summary.json").read_text())
    cut_map = resolve_cuts(de_summary)
    labels = list(CUT_NAMES) if args.cut == "all" else [args.cut]
    log(f"SI3 Dynesty cuts={labels}  seed={FIG4_POLISH_DIR.name}  out={out_root}")
    log(f"q targets: { {k: f'{cut_map[k]:.6f}' for k in labels} }")
    log(f"SI3 bounds: {SI3_BOUNDS}")

    for label in labels:
        q_cut = float(cut_map[label])
        d_cut = diffraction_length_A(q_cut)
        log(f"=== {label}: q_cut={q_cut:.6f}  d(2pi/q)={d_cut:.4f} A")
        objective = build_si3_objective(q_cut=q_cut, summary=fig4)
        lo, hi, names, x0, logl0 = configure_si3(objective, label=f"{label}:setup")
        del lo, hi, x0, logl0
        ndim = len(names)
        nlive = dy.nlive_for_ndim(ndim, args.nlive if args.nlive > 0 else None)
        walks = dy.walks_for_ndim(ndim, args.walks if args.walks > 0 else None)
        meta_extra = {
            "q_cut": q_cut,
            "d_2pi_over_q": d_cut,
            "d_pi_over_q": float(np.pi / q_cut),
            "cut_label": label,
            "protocol": "q_truncation_dynesty_si3_fig4_seed",
            "fig4_polish": str(FIG4_POLISH_DIR.resolve()),
        }
        objective = build_si3_objective(q_cut=q_cut, summary=fig4)
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

    log("all requested SI3 cuts finished")
    if args.cut == "all" and all(
        (out_root / name / "results.pkl").exists() for name in CUT_NAMES
    ):
        plot_si3_corner(dynesty_root=out_root)
        plot_si3_edu_distributions(dynesty_root=out_root)
        for var in ("v1", "v2", "v3", "v4"):
            plot_si3_depth(dynesty_root=out_root, variant=var)


if __name__ == "__main__":
    main()

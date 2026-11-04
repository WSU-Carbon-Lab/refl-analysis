"""Open-parameter multi-energy polish under ``Transform("logY")``.

Second-version companion to ``_bookended_multienergy_logY_polish.py``. Warms
from the locked logY polish MLE when available (falls back to the multi-energy
logYX4 warm), opens film bookends, bulk alpha/rho, oxide geometry/density, Si
density, and per-energy scales/thetas. Shared ``energy_offset`` stays pinned at
the locked-polish value. Widens ``tau_vac`` so the surface decay is not pinned
to the old ``(8, 55)`` wall. Writes ``bookended_multienergy_logY_polish_open/``
without touching the locked polish artifacts.
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
from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel, Slab
from refloxide.objective import Objective
from refnx.analysis import CurveFitter, Transform

_HERE = Path(__file__).resolve().parent
_MULTI = _HERE / "_bookended_multienergy_surface_oxide_logyx4.py"
_spec = importlib.util.spec_from_file_location(
    "_bookended_multienergy_surface_oxide_logyx4", _MULTI
)
me = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = me
assert _spec.loader is not None
_spec.loader.exec_module(me)

cv = me.cv
legacy = me.legacy
ENERGIES = me.ENERGIES
ANCHOR = me.ANCHOR
RHO_SIO2 = me.RHO_SIO2
RHO_SI = me.RHO_SI
INSTRUMENT_FIELDS = me.INSTRUMENT_FIELDS
MULTI_DIR = me.OUT_DIR
LOCKED_DIR = cv.models_fitting_results / "bookended_multienergy_logY_polish"
OUT_DIR = cv.models_fitting_results / "bookended_multienergy_logY_polish_open"
TRANSFORM = Transform("logY")

TAU_VAC_BOUNDS = (5.0, 100.0)
LOCKED_ENERGY_OFFSET = 0.0
FILM_FREE = (
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


def _material(slab, name: str) -> MaterialSLD:
    sld = slab.sld
    return MaterialSLD(
        sld._compound, density=sld.density, name=name, energy_offset=sld.energy_offset
    )


def resolve_warm_summary(*, prefer_open: bool = False) -> tuple[dict, Path]:
    """Prefer locked logY polish MLE; fall back to multi-energy logYX4 warm.

    When ``prefer_open`` is true and a prior open-polish summary exists, resume
    from that MLE so continued L-BFGS runs can push past maxfev stops.
    """
    candidates = []
    if prefer_open and (OUT_DIR / "summary.json").exists():
        candidates.append(OUT_DIR / "summary.json")
    candidates.extend((LOCKED_DIR / "summary.json", MULTI_DIR / "summary.json"))
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text()), path.parent
    locked = LOCKED_DIR / "summary.json"
    multi = MULTI_DIR / "summary.json"
    raise FileNotFoundError(f"no warm summary at {locked} or {multi}")


def film_bounds(film_mle: dict) -> dict[str, tuple[float, float]]:
    return {
        "total_thick": legacy.local_bounds(
            film_mle["total_thick"], lo=150.0, hi=220.0, rel=0.12
        ),
        "surface_roughness": legacy.local_bounds(
            film_mle["surface_roughness"], lo=0.0, hi=18.0, abs_pad=4.0, rel=None
        ),
        "tau_vac": TAU_VAC_BOUNDS,
        "tau_si": legacy.local_bounds(
            film_mle["tau_si"], lo=0.5, hi=60.0, rel=0.5, abs_pad=None
        ),
        "alpha_vac": legacy.local_bounds(
            film_mle["alpha_vac"], lo=0.0, hi=float(np.pi / 2), abs_pad=0.4, rel=None
        ),
        "alpha_si": legacy.local_bounds(
            film_mle["alpha_si"], lo=0.0, hi=float(np.pi / 2), abs_pad=0.4, rel=None
        ),
        "alpha_bulk": legacy.local_bounds(
            film_mle["alpha_bulk"], lo=0.0, hi=float(np.pi / 2), abs_pad=0.4, rel=None
        ),
        "density_vac": (0.5, 3.2),
        "density_si": (0.2, 3.5),
        "density_bulk": (0.8, 2.5),
    }


def nc_limit(rough: float) -> float:
    return float(np.sqrt(2.0 * np.pi) * rough / 2.0)


def open_instrument_bounds() -> dict[str, tuple[float, float]]:
    """Wide instrument boxes; avoid the warm-start wall clipping of local boxes."""
    return {
        "scale_s": (0.5, 2.2),
        "scale_p": (0.5, 2.2),
        "theta_offset_s": (-0.8, 0.8),
        "theta_offset_p": (-0.8, 0.8),
    }


def apply_open_mle(obj_map: dict, summary: dict) -> tuple:
    """Seed structure from warm summary and open the full free set."""
    vacuum, film, oxide, substrate = obj_map[ANCHOR].model.structure.components
    film_mle = summary["film"]
    oxide_mle = summary["oxide"]
    oxide_rho = float(oxide_mle.get("density", RHO_SIO2))
    si_rho = float(
        summary.get("substrate", {}).get("density", summary.get("rho_si", RHO_SI))
    )

    for _energy, obj in obj_map.items():
        cv.freeze_all_parameters(obj)

    bounds = film_bounds(film_mle)
    for name in FILM_FREE:
        cv.safely_setp_param(
            getattr(film, name),
            value=float(film_mle[name]),
            vary=True,
            bounds=bounds[name],
        )

    # Locked polish sits on the Nevot-Croce floor (thick ~= nc(rough)); FD
    # steps that raise rough or lower thick then return non-finite nll and
    # L-BFGS freezes. Only nudge when already on the floor; keep NC-safe bounds.
    ox_thick = float(oxide_mle["thick"])
    ox_rough = float(oxide_mle["rough"])
    nc0 = nc_limit(ox_rough)
    if ox_thick <= nc0 + 1e-3:
        ox_thick = nc0 + 0.75
        log(f"oxide NC nudge thick -> {ox_thick:.4f} (was on floor nc={nc0:.4f})")
    nc_scale = np.sqrt(2.0 * np.pi) / 2.0
    rough_hi = min(12.0, max(ox_rough + 1.0, ox_thick / nc_scale * 0.98))
    ox_rough = min(ox_rough, rough_hi - 1e-3)
    thick_lo = max(2.0, nc_limit(min(ox_rough, rough_hi)) + 0.02)
    oxide.thick.setp(value=ox_thick, vary=True, bounds=(thick_lo, 25.0))
    oxide.rough.setp(value=ox_rough, vary=True, bounds=(0.0, float(rough_hi)))
    oxide.sld.density.setp(value=oxide_rho, vary=True, bounds=(1.6, 2.6))
    substrate.sld.density.setp(value=si_rho, vary=True, bounds=(1.8, 2.8))

    instruments = {float(k): v for k, v in summary["instruments"].items()}
    for energy, obj in obj_map.items():
        model = obj.model
        vals = instruments[float(energy)]
        ib = open_instrument_bounds()
        for name in cv.FREE_INSTRUMENT:
            cv.safely_setp_param(
                getattr(model, name),
                value=float(vals[name]),
                vary=True,
                bounds=ib[name],
            )
        cv.safely_setp_param(
            model.energy_offset, value=LOCKED_ENERGY_OFFSET, vary=False
        )
    cv.tighten_theta_offsets(obj_map, ENERGIES)
    log(
        f"open free film={FILM_FREE}  tau_vac bounds={TAU_VAC_BOUNDS}  "
        f"oxide={ox_thick:.3f}/{ox_rough:.3f} rho={oxide_rho:.3f}  "
        f"Si_rho={si_rho:.3f}  nc0={nc0:.3f}  "
        f"E_off locked={LOCKED_ENERGY_OFFSET:+.4f}"
    )
    return vacuum, film, oxide, substrate


def build_logY_objective(obj_map: dict) -> Objective:  # noqa: N802
    comps = obj_map[ANCHOR].model.structure.components
    vacuum, film, oxide, substrate = comps
    structure = (
        Slab(
            float(vacuum.thick.value),
            _material(vacuum, "vacuum"),
            float(vacuum.rough.value),
            "vacuum",
        )
        | BookendedComponent(film)
        | Slab(oxide.thick, _material(oxide, "oxide"), oxide.rough, "oxide")
        | Slab(
            float(substrate.thick.value),
            _material(substrate, "substrate"),
            substrate.rough,
            "substrate",
        )
    )
    energies = list(ENERGIES)
    model = ReflectModel(structure, energies=energies, device="gpu")
    for par in model.corrections.parameters().flattened():
        if par.constraint is None:
            par.vary = False
    for energy in energies:
        lm = obj_map[energy].model
        for field in INSTRUMENT_FIELDS:
            src = getattr(lm, field)
            dst = getattr(model, field).at(energy)
            dst.setp(
                float(src.value),
                vary=bool(src.vary),
                bounds=(float(src.bounds.lb), float(src.bounds.ub)),
            )
    model.energy_offset.setp(LOCKED_ENERGY_OFFSET, vary=False)

    rows: dict[str, list] = {k: [] for k in ("q", "energy", "pol", "r", "r_err")}
    targets = {}
    for energy in energies:
        data = obj_map[energy].data
        for pol in ("s", "p"):
            sub = getattr(data, pol)
            q = np.asarray(sub.x, dtype=float)
            keep = q >= cv.Q_MIN
            rows["q"].append(q[keep])
            rows["energy"].append(np.full(int(keep.sum()), energy))
            rows["pol"].append(np.full(int(keep.sum()), pol, dtype=object))
            rows["r"].append(np.asarray(sub.y, dtype=float)[keep])
            rows["r_err"].append(np.asarray(sub.y_err, dtype=float)[keep])
        targets[energy] = (
            np.asarray(data.anisotropy.x, dtype=float),
            np.asarray(data.anisotropy.y, dtype=float),
        )
    dataset = ReflectDataset(**{k: np.concatenate(v) for k, v in rows.items()})
    weights = {float(obj_map[e].logp_anisotropy_weight) for e in energies}
    if len(weights) != 1:
        raise RuntimeError(f"per-energy anisotropy weights differ: {weights}")
    objective = Objective(
        model,
        dataset,
        transform=TRANSFORM,
        anisotropy_weight=weights.pop(),
        anisotropy_targets=targets,
        normalization="energy",
        nc_constraint=True,
    )
    objective.refresh_parameters()
    return objective


def polish(objective: Objective) -> list[dict]:
    history: list[dict] = []
    objective.model.device = "cpu"
    objective.model.parallel = True
    nll0 = float(objective.nll())
    history.append({"stage": "start", "nll": nll0, "wall_s": 0.0})
    log(f"logY open start nll={nll0:.6f}  nvary={len(objective.varying_parameters())}")

    t0 = time.perf_counter()
    result = CurveFitter(objective).fit(
        "L-BFGS-B",
        options={"maxiter": 800, "maxfun": 60000, "ftol": 1e-12, "eps": 1e-4},
        verbose=False,
    )
    nll1 = float(objective.nll())
    history.append(
        {
            "stage": "cpu_lbfgs_logY_fd_open",
            "nll": nll1,
            "delta_nll": nll0 - nll1,
            "nfev": int(getattr(result, "nfev", -1)),
            "wall_s": time.perf_counter() - t0,
            "message": str(getattr(result, "message", "")),
            "success": bool(getattr(result, "success", False)),
        }
    )
    log(
        f"  cpu L-BFGS (FD)  nll={nll1:.6f}  Δnll={nll0 - nll1:+.6f}  "
        f"nfev={history[-1]['nfev']}  wall={history[-1]['wall_s']:.2f}s  "
        f"msg={history[-1]['message']}"
    )
    return history


def metrics(objective: Objective) -> dict[str, float]:
    nll = float(objective.nll())
    chi2 = float(objective.chisqr())
    n = int(objective.npoints)
    nv = len(objective.varying_parameters())
    return {
        "nll": nll,
        "chi2": chi2,
        "n": n,
        "nvary": nv,
        "rms": float(np.sqrt(chi2 / max(n - nv, 1))),
    }


def per_energy_rms(objective: Objective) -> list[dict]:
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e = np.asarray(ds.energy)
    rows = []
    for energy in ENERGIES:
        m = np.isclose(e, energy)
        yd, ed = TRANSFORM(q[m], r[m], r_err[m])
        yf, _ = TRANSFORM(q[m], y_fit[m])
        resid = (yd - yf) / np.maximum(ed, 1e-30)
        n = int(resid.size)
        chi2 = float(np.sum(resid**2))
        rows.append(
            {
                "energy": energy,
                "n": n,
                "chi2": chi2,
                "rms": float(np.sqrt(chi2 / max(n, 1))),
                "n_s": int(np.sum(m & (pol == "s"))),
                "n_p": int(np.sum(m & (pol == "p"))),
            }
        )
    return rows


def bound_hits(objective: Objective, tol: float = 1e-4) -> list[str]:
    hits = []
    for p in objective.varying_parameters():
        v = float(p.value)
        lo = float(p.bounds.lb)
        hi = float(p.bounds.ub)
        if np.isfinite(lo) and abs(v - lo) <= tol:
            hits.append(f"{p.name} at lo={lo:g} (v={v:g})")
        if np.isfinite(hi) and abs(v - hi) <= tol:
            hits.append(f"{p.name} at hi={hi:g} (v={v:g})")
    return hits


def plot_demo(objective: Objective, summary: dict, path: Path) -> None:
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
    data_kw = {
        "fmt": "o",
        "ms": 2.2,
        "lw": 0.45,
        "color": "0.25",
        "ecolor": "0.65",
        "elinewidth": 0.4,
        "capsize": 0,
        "zorder": 1,
        "rasterized": True,
    }
    fit_kw = {"color": "#C0392B", "lw": 1.15, "zorder": 2}
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
            yd, ed = TRANSFORM(qq, rr, ee)
            yf, _ = TRANSFORM(qq, ff)
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
            if col == 0:
                ax.set_ylabel(r"$\log_{10} R$")
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")

    axes[0, 0].plot([], [], "o", ms=3.5, color="0.25", label="data")
    axes[0, 0].plot([], [], color="#C0392B", lw=1.4, label="fit")
    axes[0, 0].legend(loc="lower left", frameon=False, handlelength=1.6)

    film = summary["film"]
    ox = summary["oxide"]
    after = summary["after"]
    fig.suptitle(
        r"ZnPc book-ended — open multi-energy log$Y$ polish"
        f"\noxide={ox['thick']:.2f}/{ox['rough']:.2f} Å  "
        r"$\rho_{\mathrm{ox}}$"
        f"={ox['density']:.3f}  ·  nll={after['nll']:.2f}  "
        f"RMS={after['rms']:.2f}  ·  "
        r"$\tau_{\mathrm{vac}}$"
        f"={film['tau_vac']:.1f} Å",
        fontsize=10,
    )
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    log(f"wrote {path.with_suffix('.png')}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-open",
        action="store_true",
        help="warm from an existing open-polish summary if present",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_in, warm_dir = resolve_warm_summary(prefer_open=bool(args.resume_open))
    log(f"warm from {warm_dir}  protocol={summary_in.get('protocol')}")

    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    vacuum, film, oxide, substrate = apply_open_mle(obj_map, summary_in)
    _ = vacuum

    objective = build_logY_objective(obj_map)
    before = metrics(objective)
    log(f"before (logY open): {before}")
    log(f"varying ({len(objective.varying_parameters())}):")
    for p in objective.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
        )

    history = polish(objective)
    after = metrics(objective)
    per = per_energy_rms(objective)
    hits = bound_hits(objective)

    for energy in ENERGIES:
        lm = obj_map[energy].model
        for field in INSTRUMENT_FIELDS:
            getattr(lm, field).value = float(
                getattr(objective.model, field).at(energy).value
            )
        lm.energy_offset.value = float(objective.model.energy_offset.value)

    film_snap = {name: float(getattr(film, name).value) for name in FILM_FREE}
    oxide_snap = {
        "thick": float(oxide.thick.value),
        "rough": float(oxide.rough.value),
        "density": float(oxide.sld.density.value),
    }
    substrate_snap = {"density": float(substrate.sld.density.value)}
    instruments = {
        float(energy): {
            name: float(getattr(obj_map[energy].model, name).value)
            for name in INSTRUMENT_FIELDS
        }
        for energy in ENERGIES
    }
    summary = {
        "energies": list(ENERGIES),
        "transform": "logY",
        "before": before,
        "after": after,
        "history": history,
        "per_energy": per,
        "film": film_snap,
        "oxide": oxide_snap,
        "substrate": substrate_snap,
        "energy_offset": float(objective.model.energy_offset.value),
        "instruments": instruments,
        "bound_hits": hits,
        "tau_vac_bounds": list(TAU_VAC_BOUNDS),
        "protocol": "cpu_lbfgs_logY_polish_open_structure_instruments_Eoff_locked",
        "warm_from": str(warm_dir),
        "energy_offset_locked": True,
        "locked_reference_nll": float(
            summary_in.get("after", {}).get("nll", float("nan"))
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT_DIR / "fit.pkl").open("wb") as fh:
        pickle.dump(
            {"obj_map": obj_map, "gpu_objective": objective, "summary": summary},
            fh,
        )
    plot_demo(objective, summary, OUT_DIR / "demo_multienergy_logY_open")

    log(f"after: {after}")
    for row in per:
        log(f"  {row['energy']:g} eV  RMS={row['rms']:.3f}")
    if hits:
        log("bound hits:")
        for h in hits:
            log(f"  {h}")
    else:
        log("bound hits: none")
    log(
        f"tau_vac={film_snap['tau_vac']:.3f}  "
        f"density_vac={film_snap['density_vac']:.3f}  "
        f"oxide={oxide_snap['thick']:.3f}/{oxide_snap['rough']:.3f}/"
        f"{oxide_snap['density']:.3f}  "
        f"Si={substrate_snap['density']:.3f}  "
        f"E_off={summary['energy_offset']:+.4f}"
    )
    log(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

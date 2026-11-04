"""Final multi-energy polish under ``Transform("logY")``.

Loads the locked-oxide / locked-``density_vac`` multi-energy MLE, rebuilds the
GPU objective with straight ``log10(R)``, and L-BFGS-B-polishes the free set
(``tau_vac``, ``alpha_vac``, ``surface_roughness``, per-energy scales/thetas).
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
TAU_VAC_BOUNDS = me.TAU_VAC_BOUNDS
SURFACE_PARAMS = me.SURFACE_PARAMS
INSTRUMENT_FIELDS = me.INSTRUMENT_FIELDS
MULTI_DIR = me.OUT_DIR
OUT_DIR = cv.models_fitting_results / "bookended_multienergy_logY_polish"
TRANSFORM = Transform("logY")


def log(msg: str) -> None:
    print(msg, flush=True)


def _material(slab, name: str) -> MaterialSLD:
    sld = slab.sld
    return MaterialSLD(
        sld._compound, density=sld.density, name=name, energy_offset=sld.energy_offset
    )


def apply_multi_mle(obj_map: dict, summary: dict) -> tuple:
    """Pin locks and free set from the multi-energy logYX4 MLE."""
    vacuum, film, oxide, substrate = obj_map[ANCHOR].model.structure.components
    film_mle = summary["film"]
    oxide_mle = summary["oxide"]
    graded = me.load_single_warm()["start"]

    oxide.sld.density.setp(value=RHO_SIO2, vary=False)
    substrate.sld.density.setp(value=RHO_SI, vary=False)

    for energy, obj in obj_map.items():
        cv.freeze_all_parameters(obj)

    for name in ("tau_si", "density_si"):
        getattr(film, name).setp(value=graded[name], vary=False)
    cv.safely_setp_param(film.density_vac, value=film_mle["density_vac"], vary=False)
    oxide.thick.setp(value=oxide_mle["thick"], vary=False)
    oxide.rough.setp(value=oxide_mle["rough"], vary=False)
    log(
        f"locked density_vac={film_mle['density_vac']:.4f}  "
        f"oxide={oxide_mle['thick']:.4f}/{oxide_mle['rough']:.4f}"
    )

    for name in ("total_thick", "density_bulk", "alpha_bulk", "alpha_si"):
        cv.safely_setp_param(getattr(film, name), value=graded[name], vary=False)

    bounds = {
        "tau_vac": TAU_VAC_BOUNDS,
        "alpha_vac": legacy.local_bounds(
            film_mle["alpha_vac"], lo=0.0, hi=float(np.pi / 2), abs_pad=0.35, rel=None
        ),
        "surface_roughness": legacy.local_bounds(
            film_mle["surface_roughness"], lo=0.0, hi=15.0, abs_pad=3.0, rel=None
        ),
    }
    for name in SURFACE_PARAMS:
        cv.safely_setp_param(
            getattr(film, name),
            value=film_mle[name],
            vary=True,
            bounds=bounds[name],
        )

    instruments = {float(k): v for k, v in summary["instruments"].items()}
    for energy, obj in obj_map.items():
        model = obj.model
        vals = instruments[float(energy)]
        for name in cv.FREE_INSTRUMENT:
            cv.safely_setp_param(
                getattr(model, name),
                value=float(vals[name]),
                vary=True,
                bounds=legacy.instrument_bounds_for(model)[name],
            )
    cv.tighten_theta_offsets(obj_map, ENERGIES)
    return vacuum, film, oxide, substrate


def build_logY_objective(obj_map: dict) -> Objective:
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
    # Theta offsets remesh q; analytic nll_and_grad assumes fixed q layout.
    # CurveFitter L-BFGS-B uses finite-difference jacobians instead.
    objective.model.device = "cpu"
    objective.model.parallel = True
    nll0 = float(objective.nll())
    history.append({"stage": "start", "nll": nll0, "wall_s": 0.0})
    log(f"logY start nll={nll0:.6f}  nvary={len(objective.varying_parameters())}")

    t0 = time.perf_counter()
    result = CurveFitter(objective).fit(
        "L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-14},
        verbose=False,
    )
    nll1 = float(objective.nll())
    history.append(
        {
            "stage": "cpu_lbfgs_logY_fd",
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
    DATA_KW = dict(
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
    FIT_KW = dict(color="#C0392B", lw=1.15, zorder=2)
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
            ax.errorbar(qq, yd, yerr=ed, **DATA_KW)
            ax.plot(qq, yf, **FIT_KW)
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
        r"ZnPc book-ended profile — multi-energy log$Y$ polish"
        f"\nlocked oxide={ox['thick']:.2f}/{ox['rough']:.2f} Å, "
        r"$\rho_{\mathrm{vac}}$"
        f"={film['density_vac']:.3f}  ·  nll={after['nll']:.2f}  "
        f"RMS={after['rms']:.2f}  ·  "
        r"$\tau_{\mathrm{vac}}$"
        f"={film['tau_vac']:.1f} Å",
        fontsize=10,
    )
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    log(f"wrote {path.with_suffix('.png')}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_in = json.loads((MULTI_DIR / "summary.json").read_text())
    log(f"warm from {MULTI_DIR}  protocol={summary_in.get('protocol')}")

    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    vacuum, film, oxide, substrate = apply_multi_mle(obj_map, summary_in)
    _ = vacuum, substrate

    objective = build_logY_objective(obj_map)
    before = metrics(objective)
    log(f"before (logY): {before}")
    log(f"varying ({len(objective.varying_parameters())}):")
    for p in objective.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
        )

    history = polish(objective)
    after = metrics(objective)
    per = per_energy_rms(objective)

    for energy in ENERGIES:
        lm = obj_map[energy].model
        for field in INSTRUMENT_FIELDS:
            getattr(lm, field).value = float(
                getattr(objective.model, field).at(energy).value
            )

    film_snap = {
        name: float(getattr(film, name).value)
        for name in (
            "total_thick",
            "surface_roughness",
            "density_bulk",
            "density_si",
            "density_vac",
            "tau_si",
            "tau_vac",
            "alpha_bulk",
            "alpha_si",
            "alpha_vac",
        )
    }
    oxide_snap = {
        "thick": float(oxide.thick.value),
        "rough": float(oxide.rough.value),
        "density": float(oxide.sld.density.value),
    }
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
        "instruments": instruments,
        "protocol": "gpu_lbfgs_logY_polish_locked_oxide_density_vac",
        "warm_from": str(MULTI_DIR),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT_DIR / "fit.pkl").open("wb") as fh:
        pickle.dump(
            {"obj_map": obj_map, "gpu_objective": objective, "summary": summary},
            fh,
        )
    plot_demo(objective, summary, OUT_DIR / "demo_multienergy_logY")

    log(f"after: {after}")
    for row in per:
        log(f"  {row['energy']:g} eV  RMS={row['rms']:.3f}")
    log(
        f"tau_vac={film_snap['tau_vac']:.3f}  density_vac={film_snap['density_vac']:.3f}  "
        f"oxide={oxide_snap['thick']:.3f}/{oxide_snap['rough']:.3f}"
    )
    log(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

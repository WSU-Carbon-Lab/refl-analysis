"""283.7 eV surface + oxide refit with substrate side and tabulated Si/SiO2 fixed.

Freezes ``tau_si`` / ``density_si`` at the graded MLE start, pins SiO2 and Si
densities to tabulated values (2.2 / 2.33 g/cm^3), and refits surface film
knobs plus oxide thick/rough (and scales/thetas) with GPU differential
evolution under Nevot-Croce, scoring in ``log10(R q^4)``.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel, Slab
from refloxide.objective import Objective, gpu_batch
from refnx.analysis import CurveFitter

_HERE = Path(__file__).resolve().parent
_POLISH = _HERE / "_bookended_multienergy_polish.py"
_spec = importlib.util.spec_from_file_location("_bookended_multienergy_polish", _POLISH)
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
assert _spec.loader is not None
_spec.loader.exec_module(legacy)
cv = legacy.cv

ENERGY = float(cv.ANCHOR)
OUT_DIR = cv.models_fitting_results / "bookended_2837_surface_oxide_refit"
RHO_SIO2 = 2.2
RHO_SI = 2.33
TAU_VAC_BOUNDS = (8.0, 55.0)
SURFACE_PARAMS = ("tau_vac", "density_vac", "alpha_vac", "surface_roughness")
FIXED_SUBSTRATE_SIDE = ("tau_si", "density_si")
INSTRUMENT_FIELDS = ("scale_s", "scale_p", "theta_offset_s", "theta_offset_p", "bkg")
_LN10 = float(np.log(10.0))


class LogYX4:
    """``log10(R q^4)`` transform with first-order error propagation.

    refnx ships ``YX4`` and ``logY`` separately; this composes them so the
    objective scores in logarithmic Fresnel-compensated space.
    """

    form = "logYX4"

    def __repr__(self) -> str:
        return "LogYX4()"

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        yx4 = y * np.power(x, 4)
        # Floor tiny/nonpositive values so log stays finite under DE proposals.
        yx4 = np.maximum(yx4, 1e-30)
        yt = np.log10(yx4)
        if y_err is None:
            return yt, None
        et = np.asarray(y_err, dtype=float) * np.power(x, 4)
        et = np.maximum(et, 0.0) / (yx4 * _LN10)
        return yt, et


TRANSFORM = LogYX4()


def log(msg: str) -> None:
    print(msg, flush=True)


def nc_limit(rough: float) -> float:
    return float(np.sqrt(2.0 * np.pi) * rough / 2.0)


def _material(slab, name: str) -> MaterialSLD:
    sld = slab.sld
    return MaterialSLD(
        sld._compound, density=sld.density, name=name, energy_offset=sld.energy_offset
    )


def pin_tabulated_densities(oxide, substrate) -> dict[str, float]:
    oxide.sld.density.setp(value=RHO_SIO2, vary=False)
    substrate.sld.density.setp(value=RHO_SI, vary=False)
    info = {
        "sio2": RHO_SIO2,
        "si": RHO_SI,
        "oxide_rho": float(oxide.sld.density.value),
        "substrate_rho": float(substrate.sld.density.value),
    }
    log(f"tabulated densities: SiO2={RHO_SIO2:g}  Si={RHO_SI:g} g/cm^3 (frozen)")
    return info


def fix_substrate_side(film, start: dict[str, float]) -> None:
    for name in FIXED_SUBSTRATE_SIDE:
        getattr(film, name).setp(value=start[name], vary=False)
        log(f"fix {name}={start[name]:.6g} (graded start)")


def open_surface_and_oxide(obj, film, oxide, start: dict[str, float]) -> None:
    cv.freeze_all_parameters(obj)

    # Surface knobs only.
    bounds = {
        "tau_vac": TAU_VAC_BOUNDS,
        "density_vac": legacy.local_bounds(
            start["density_vac"], lo=0.5, hi=2.5, rel=0.25
        ),
        "alpha_vac": legacy.local_bounds(
            start["alpha_vac"], lo=0.0, hi=float(np.pi / 2), abs_pad=0.35, rel=None
        ),
        "surface_roughness": legacy.local_bounds(
            start["surface_roughness"], lo=0.0, hi=15.0, abs_pad=3.0, rel=None
        ),
    }
    for name in SURFACE_PARAMS:
        cv.safely_setp_param(
            getattr(film, name), value=start[name], vary=True, bounds=bounds[name]
        )

    # Bulk / substrate-side film geometry stay at graded start.
    for name in (
        "total_thick",
        "density_bulk",
        "alpha_bulk",
        "alpha_si",
        "tau_si",
        "density_si",
    ):
        cv.safely_setp_param(getattr(film, name), value=start[name], vary=False)

    # Oxide thick/rough free; start NC-valid for current rough.
    rough0 = float(oxide.rough.value)
    thick0 = float(oxide.thick.value)
    floor = nc_limit(rough0)
    thick_start = max(thick0, floor)
    oxide.thick.setp(
        value=thick_start,
        vary=True,
        bounds=(5.0, 30.0),
    )
    oxide.rough.setp(
        value=rough0,
        vary=True,
        bounds=(0.5, 15.0),
    )
    log(
        f"oxide free: thick={thick_start:.4f} (was {thick0:.4f}, NC floor={floor:.4f})  "
        f"rough={rough0:.4f}"
    )

    # Scales/thetas free so the tabulated Si/SiO2 density jump can be absorbed;
    # structure free set stays surface + oxide only.
    for name in cv.FREE_INSTRUMENT:
        cv.safely_setp_param(
            getattr(obj.model, name),
            vary=True,
            bounds=legacy.instrument_bounds_for(obj.model)[name],
        )
    cv.tighten_theta_offsets({ENERGY: obj}, (ENERGY,))


def build_gpu_objective(obj) -> Objective:
    comps = obj.model.structure.components
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
    model = ReflectModel(structure, energies=[ENERGY], device="gpu")
    for par in model.corrections.parameters().flattened():
        if par.constraint is None:
            par.vary = False

    lm = obj.model
    for field in INSTRUMENT_FIELDS:
        src = getattr(lm, field)
        dst = getattr(model, field).at(ENERGY)
        dst.setp(
            float(src.value),
            vary=bool(src.vary),
            bounds=(float(src.bounds.lb), float(src.bounds.ub)),
        )

    rows: dict[str, list] = {k: [] for k in ("q", "energy", "pol", "r", "r_err")}
    data = obj.data
    for pol in ("s", "p"):
        sub = getattr(data, pol)
        q = np.asarray(sub.x, dtype=float)
        keep = q >= cv.Q_MIN
        rows["q"].append(q[keep])
        rows["energy"].append(np.full(int(keep.sum()), ENERGY))
        rows["pol"].append(np.full(int(keep.sum()), pol, dtype=object))
        rows["r"].append(np.asarray(sub.y, dtype=float)[keep])
        rows["r_err"].append(np.asarray(sub.y_err, dtype=float)[keep])
    targets = {
        ENERGY: (
            np.asarray(data.anisotropy.x, dtype=float),
            np.asarray(data.anisotropy.y, dtype=float),
        )
    }
    dataset = ReflectDataset(**{k: np.concatenate(v) for k, v in rows.items()})
    objective = Objective(
        model,
        dataset,
        transform=TRANSFORM,
        anisotropy_weight=float(obj.logp_anisotropy_weight),
        anisotropy_targets=targets,
        normalization="energy",
        nc_constraint=True,
    )
    objective.refresh_parameters()
    return objective


DE_KW = {
    "popsize": 15,
    "maxiter": 80,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "seed": 1,
    "polish": False,
    "verbose": False,
}


def run_de(objective: Objective) -> dict:
    """GPU differential evolution over the free surface + oxide + instrument set."""
    objective.model.device = "gpu"
    nll0 = float(objective.nll())
    log(
        f"DE start nll={nll0:.6f}  "
        f"popsize={DE_KW['popsize']}  maxiter={DE_KW['maxiter']}  "
        f"nvary={len(objective.varying_parameters())}"
    )
    t0 = time.perf_counter()
    result = CurveFitter(objective).fit(
        "differential_evolution",
        workers=gpu_batch(objective),
        **DE_KW,
    )
    wall = time.perf_counter() - t0
    nll1 = float(objective.nll())
    stage = {
        "stage": "de_gpu",
        "nll_before": nll0,
        "nll_after": nll1,
        "delta_nll": nll0 - nll1,
        "wall_s": wall,
        "nit": int(getattr(result, "nit", -1)),
        "nfev": int(getattr(result, "nfev", -1)),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        **{k: DE_KW[k] for k in ("popsize", "maxiter", "recombination", "seed")},
    }
    log(
        f"DE done  nll={nll1:.6f}  Δnll={nll0 - nll1:+.6f}  "
        f"nit={stage['nit']}  nfev={stage['nfev']}  wall={wall:.1f}s"
    )
    return stage


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


def plot_overlay(objective: Objective, summary: dict, path: Path) -> None:
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e = np.asarray(ds.energy)
    yx4 = TRANSFORM
    energy = float(summary["energy"])

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), constrained_layout=True)
    for col, pol_name in enumerate(("s", "p")):
        m = (pol == pol_name) & np.isclose(e, energy)
        qq, rr, ee, ff = q[m], r[m], r_err[m], y_fit[m]
        order = np.argsort(qq)
        qq, rr, ee, ff = qq[order], rr[order], ee[order], ff[order]

        ax = axes[0, col]
        ax.errorbar(
            qq,
            rr,
            yerr=ee,
            fmt="o",
            ms=2.5,
            lw=0.6,
            color="0.35",
            ecolor="0.7",
            elinewidth=0.5,
            capsize=0,
            label="data",
            zorder=1,
        )
        ax.plot(qq, ff, color="C3", lw=1.3, label="fit", zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$R$")
        ax.set_title(f"{energy:g} eV  {pol_name}-pol  (log $R$)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)

        yd, ed = yx4(qq, rr, ee)
        yf, _ = yx4(qq, ff)
        ax = axes[1, col]
        ax.errorbar(
            qq,
            yd,
            yerr=ed,
            fmt="o",
            ms=2.5,
            lw=0.6,
            color="0.35",
            ecolor="0.7",
            elinewidth=0.5,
            capsize=0,
            label="data",
            zorder=1,
        )
        ax.plot(qq, yf, color="C3", lw=1.3, label="fit", zorder=2)
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$\log_{10}(R\,q^{4})$")
        ax.set_title(f"{energy:g} eV  {pol_name}-pol  (log YX4)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    film = summary["film"]
    ox = summary["oxide"]
    fig.suptitle(
        f"283.7 surface+oxide (log YX4)  |  nll={summary['after']['nll']:.3f}  "
        f"RMS={summary['after']['rms']:.3f}  |  "
        f"tau_vac={film['tau_vac']:.2f}  oxide={ox['thick']:.2f}/{ox['rough']:.2f} A",
        fontsize=10,
    )
    fig.savefig(path, dpi=180)
    log(f"wrote {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={OUT_DIR}")

    objectives = cv.seed_objectives()
    obj = objectives[ENERGY]
    vacuum, film, oxide, substrate = obj.model.structure.components
    _ = vacuum

    start = {
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

    density_info = pin_tabulated_densities(oxide, substrate)
    fix_substrate_side(film, start)
    open_surface_and_oxide(obj, film, oxide, start)

    log("varying:")
    for p in obj.parameters.flattened():
        if p.vary and p.constraint is None:
            log(
                f"  {p.name}={float(p.value):.6g} "
                f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
            )

    objective = build_gpu_objective(obj)
    objective.refresh_parameters()
    before = metrics(objective)
    log(f"before: {before}")
    log(f"gpu varying ({len(objective.varying_parameters())}):")
    for p in objective.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
        )

    history = [run_de(objective)]
    after = metrics(objective)

    for field in INSTRUMENT_FIELDS:
        getattr(obj.model, field).value = float(
            getattr(objective.model, field).at(ENERGY).value
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
        "nc_floor": nc_limit(float(oxide.rough.value)),
        "nc_ok": float(oxide.thick.value) >= nc_limit(float(oxide.rough.value)) - 1e-9,
    }
    inst_snap = {
        name: float(getattr(obj.model, name).value) for name in INSTRUMENT_FIELDS
    }
    summary = {
        "energy": ENERGY,
        "transform": "logYX4",
        "rho_sio2": RHO_SIO2,
        "rho_si": RHO_SI,
        "tau_vac_bounds": list(TAU_VAC_BOUNDS),
        "densities": density_info,
        "start": start,
        "before": before,
        "after": after,
        "history": history,
        "film": film_snap,
        "oxide": oxide_snap,
        "substrate_density": float(substrate.sld.density.value),
        "instrument": inst_snap,
        "free": list(SURFACE_PARAMS)
        + ["oxide.thick", "oxide.rough"]
        + list(cv.FREE_INSTRUMENT),
        "protocol": "gpu_de_surface_oxide_logYX4_tabulated_si_sio2",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT_DIR / "fit.pkl").open("wb") as fh:
        pickle.dump(
            {
                "legacy_objective": obj,
                "gpu_objective": objective,
                "summary": summary,
            },
            fh,
        )
    plot_overlay(objective, summary, OUT_DIR / "fit_2837_overlay.png")

    log(f"after: {after}")
    log(
        f"surface: tau_vac={film_snap['tau_vac']:.3f}  "
        f"density_vac={film_snap['density_vac']:.3f}  "
        f"alpha_vac={film_snap['alpha_vac']:.4f}  "
        f"rough={film_snap['surface_roughness']:.3f}"
    )
    log(
        f"fixed substrate-side: tau_si={film_snap['tau_si']:.3f}  "
        f"density_si={film_snap['density_si']:.3f}"
    )
    log(
        f"oxide: thick={oxide_snap['thick']:.3f}  rough={oxide_snap['rough']:.3f}  "
        f"rho={oxide_snap['density']:.3f}  NC_ok={oxide_snap['nc_ok']}"
    )
    log(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Multi-energy resonant DE: surface + instruments under logYX4.

Warm-starts from the single-energy 283.7 logYX4 surface+oxide MLE. Pins
``tau_si`` / ``density_si`` at the graded start, SiO2/Si densities to
tabulated 2.2 / 2.33, and locks ``density_vac`` plus oxide thick/rough at the
283.7 MLE. Frees ``tau_vac`` / ``alpha_vac`` / ``surface_roughness`` and
per-energy scales/thetas across ``E_RES``.
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

ENERGIES = tuple(float(e) for e in cv.E_RES)
ANCHOR = float(cv.ANCHOR)
OUT_DIR = cv.models_fitting_results / "bookended_multienergy_surface_oxide_logyx4"
SINGLE_DIR = cv.models_fitting_results / "bookended_2837_surface_oxide_refit"
RHO_SIO2 = 2.2
RHO_SI = 2.33
TAU_VAC_BOUNDS = (8.0, 55.0)
SURFACE_PARAMS = ("tau_vac", "alpha_vac", "surface_roughness")
LOCKED_FROM_2837 = ("density_vac", "oxide.thick", "oxide.rough")
INSTRUMENT_FIELDS = ("scale_s", "scale_p", "theta_offset_s", "theta_offset_p", "bkg")
_LN10 = float(np.log(10.0))

DE_KW = {
    "popsize": 15,
    "maxiter": 120,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "seed": 1,
    "polish": False,
    "verbose": False,
}


class LogYX4:
    """``log10(R q^4)`` transform with first-order error propagation."""

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
        yx4 = np.maximum(y * np.power(x, 4), 1e-30)
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


def load_single_warm() -> dict:
    path = SINGLE_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing single-energy warm start: {path}")
    return json.loads(path.read_text())


def setup_structure(obj_map: dict, warm: dict) -> tuple:
    anchor = obj_map[ANCHOR]
    vacuum, film, oxide, substrate = anchor.model.structure.components
    graded_start = warm["start"]
    film_warm = warm["film"]
    oxide_warm = warm["oxide"]

    oxide.sld.density.setp(value=RHO_SIO2, vary=False)
    substrate.sld.density.setp(value=RHO_SI, vary=False)
    log(f"tabulated densities: SiO2={RHO_SIO2:g}  Si={RHO_SI:g}")

    # Pin substrate-side film at graded start.
    for name in ("tau_si", "density_si"):
        getattr(film, name).setp(value=graded_start[name], vary=False)
        log(f"fix {name}={graded_start[name]:.6g}")

    for energy, obj in obj_map.items():
        cv.freeze_all_parameters(obj)

    # Lock density_vac + oxide thick/rough at the 283.7 MLE (joint DE was
    # driving them unphysically high / thick).
    cv.safely_setp_param(film.density_vac, value=film_warm["density_vac"], vary=False)
    log(f"fix density_vac={film_warm['density_vac']:.6g} (283.7 MLE)")

    # Remaining surface knobs free; warm from single-energy MLE.
    bounds = {
        "tau_vac": TAU_VAC_BOUNDS,
        "alpha_vac": legacy.local_bounds(
            film_warm["alpha_vac"], lo=0.0, hi=float(np.pi / 2), abs_pad=0.35, rel=None
        ),
        "surface_roughness": legacy.local_bounds(
            film_warm["surface_roughness"], lo=0.0, hi=15.0, abs_pad=3.0, rel=None
        ),
    }
    for name in SURFACE_PARAMS:
        cv.safely_setp_param(
            getattr(film, name),
            value=film_warm[name],
            vary=True,
            bounds=bounds[name],
        )

    for name in ("total_thick", "density_bulk", "alpha_bulk", "alpha_si"):
        cv.safely_setp_param(getattr(film, name), value=graded_start[name], vary=False)

    oxide.thick.setp(value=oxide_warm["thick"], vary=False)
    oxide.rough.setp(value=oxide_warm["rough"], vary=False)
    log(
        f"fix oxide thick={oxide_warm['thick']:.4f}  "
        f"rough={oxide_warm['rough']:.4f}  (283.7 MLE, NC_ok={oxide_warm['nc_ok']})"
    )

    # Per-energy instruments: 283.7 from single-energy MLE, others from seed.
    for energy, obj in obj_map.items():
        model = obj.model
        if np.isclose(float(energy), ANCHOR):
            for name, value in warm["instrument"].items():
                if name == "bkg":
                    continue
                cv.safely_setp_param(
                    getattr(model, name),
                    value=float(value),
                    vary=True,
                    bounds=legacy.instrument_bounds_for(model)[name],
                )
        else:
            for name in cv.FREE_INSTRUMENT:
                cv.safely_setp_param(
                    getattr(model, name),
                    vary=True,
                    bounds=legacy.instrument_bounds_for(model)[name],
                )
    cv.tighten_theta_offsets(obj_map, ENERGIES)
    return vacuum, film, oxide, substrate


def build_gpu_objective(obj_map: dict) -> Objective:
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


def run_de(objective: Objective) -> dict:
    objective.model.device = "gpu"
    nll0 = float(objective.nll())
    log(
        f"DE start nll={nll0:.6f}  popsize={DE_KW['popsize']}  "
        f"maxiter={DE_KW['maxiter']}  nvary={len(objective.varying_parameters())}"
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


def plot_all_energies(objective: Objective, summary: dict, path: Path) -> None:
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol_arr = np.asarray(ds.pol)
    e = np.asarray(ds.energy)

    nrows = len(ENERGIES)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(8.5, 2.2 * nrows), constrained_layout=True, sharex=True
    )
    if nrows == 1:
        axes = np.asarray([axes])

    for i, energy in enumerate(ENERGIES):
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol_arr == pol_name) & np.isclose(e, energy)
            qq, rr, ee, ff = q[m], r[m], r_err[m], y_fit[m]
            order = np.argsort(qq)
            qq, rr, ee, ff = qq[order], rr[order], ee[order], ff[order]
            yd, ed = TRANSFORM(qq, rr, ee)
            yf, _ = TRANSFORM(qq, ff)
            ax.errorbar(
                qq,
                yd,
                yerr=ed,
                fmt="o",
                ms=2.0,
                lw=0.5,
                color="0.35",
                ecolor="0.75",
                elinewidth=0.4,
                capsize=0,
                zorder=1,
            )
            ax.plot(qq, yf, color="C3", lw=1.1, zorder=2)
            ax.set_ylabel(r"$\log_{10}(Rq^{4})$")
            title = f"{energy:g} eV  {pol_name}"
            per = next(
                r for r in summary["per_energy"] if np.isclose(r["energy"], energy)
            )
            ax.set_title(f"{title}  RMS={per['rms']:.2f}", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")

    film = summary["film"]
    ox = summary["oxide"]
    after = summary["after"]
    fig.suptitle(
        f"resonant multi-energy (log YX4)  |  "
        f"nll={after['nll']:.3f}  RMS={after['rms']:.3f}  |  "
        f"locked oxide={ox['thick']:.2f}/{ox['rough']:.2f}  "
        f"density_vac={film['density_vac']:.3f}  |  "
        f"tau_vac={film['tau_vac']:.2f}",
        fontsize=10,
    )
    fig.savefig(path, dpi=160)
    log(f"wrote {path}")


def plot_logR(objective: Objective, summary: dict, path: Path) -> None:
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol_arr = np.asarray(ds.pol)
    e = np.asarray(ds.energy)

    nrows = len(ENERGIES)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(8.5, 2.2 * nrows), constrained_layout=True, sharex=True
    )
    if nrows == 1:
        axes = np.asarray([axes])

    for i, energy in enumerate(ENERGIES):
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol_arr == pol_name) & np.isclose(e, energy)
            qq, rr, ee, ff = q[m], r[m], r_err[m], y_fit[m]
            order = np.argsort(qq)
            qq, rr, ee, ff = qq[order], rr[order], ee[order], ff[order]
            ax.errorbar(
                qq,
                rr,
                yerr=ee,
                fmt="o",
                ms=2.0,
                lw=0.5,
                color="0.35",
                ecolor="0.75",
                elinewidth=0.4,
                capsize=0,
                zorder=1,
            )
            ax.plot(qq, ff, color="C3", lw=1.1, zorder=2)
            ax.set_yscale("log")
            ax.set_ylabel(r"$R$")
            ax.set_title(f"{energy:g} eV  {pol_name}", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")

    after = summary["after"]
    fig.suptitle(
        f"resonant multi-energy surface+oxide (log R view)  |  "
        f"nll={after['nll']:.3f}  RMS={after['rms']:.3f}",
        fontsize=10,
    )
    fig.savefig(path, dpi=160)
    log(f"wrote {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={OUT_DIR}")
    warm = load_single_warm()
    log(f"warm start from {SINGLE_DIR}")

    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    vacuum, film, oxide, substrate = setup_structure(obj_map, warm)
    _ = vacuum, substrate

    log("varying (legacy map):")
    for energy in ENERGIES:
        for p in obj_map[energy].parameters.flattened():
            if p.vary and p.constraint is None:
                log(
                    f"  [{energy:g}] {p.name}={float(p.value):.6g} "
                    f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
                )

    objective = build_gpu_objective(obj_map)
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
        "nc_floor": nc_limit(float(oxide.rough.value)),
        "nc_ok": float(oxide.thick.value) >= nc_limit(float(oxide.rough.value)) - 1e-9,
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
        "transform": "logYX4",
        "rho_sio2": RHO_SIO2,
        "rho_si": RHO_SI,
        "before": before,
        "after": after,
        "history": history,
        "per_energy": per,
        "film": film_snap,
        "oxide": oxide_snap,
        "instruments": instruments,
        "protocol": "gpu_de_multienergy_locked_oxide_density_vac_logYX4",
        "locked": list(LOCKED_FROM_2837)
        + ["tau_si", "density_si", "rho_sio2", "rho_si"],
        "warm_from": str(SINGLE_DIR),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT_DIR / "fit.pkl").open("wb") as fh:
        pickle.dump(
            {
                "obj_map": obj_map,
                "gpu_objective": objective,
                "summary": summary,
            },
            fh,
        )

    plot_all_energies(objective, summary, OUT_DIR / "fit_logyx4_overlay.png")
    plot_logR(objective, summary, OUT_DIR / "fit_logR_overlay.png")

    log(f"after: {after}")
    for row in per:
        log(f"  {row['energy']:g} eV  RMS={row['rms']:.3f}  chi2={row['chi2']:.1f}")
    log(
        f"film: tau_vac={film_snap['tau_vac']:.3f}  "
        f"density_vac={film_snap['density_vac']:.3f}  "
        f"tau_si={film_snap['tau_si']:.3f}  density_si={film_snap['density_si']:.3f}"
    )
    log(
        f"oxide: thick={oxide_snap['thick']:.3f}  rough={oxide_snap['rough']:.3f}  "
        f"NC_ok={oxide_snap['nc_ok']}"
    )
    log(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Single-energy 283.7 eV GPU L-BFGS from the graded MLE.

1. Fix SiO2 so thick/rough satisfy Nevot-Croce (raise thick to the NC floor).
2. Widen ``tau_vac`` past the graded ceiling of 25 A (SE ~1-2 A).
3. Refit geometry + instruments at 283.7 only on ``device=\"gpu\"``.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel, Slab
from refloxide.objective import Objective, minimize_lbfgsb

_HERE = Path(__file__).resolve().parent
_POLISH = _HERE / "_bookended_multienergy_polish.py"
_spec = importlib.util.spec_from_file_location("_bookended_multienergy_polish", _POLISH)
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
assert _spec.loader is not None
_spec.loader.exec_module(legacy)
cv = legacy.cv

ENERGY = float(cv.ANCHOR)
OUT_DIR = cv.models_fitting_results / "bookended_2837_nc_refit"
INSTRUMENT_FIELDS = ("scale_s", "scale_p", "theta_offset_s", "theta_offset_p", "bkg")
OPEN_GEOMETRY = legacy.OPEN_GEOMETRY
# Graded MLE sat at tau_vac=25 (ub) with stderr ~1-2 A; open a few A past that.
TAU_VAC_BOUNDS = (8.0, 55.0)


def log(msg: str) -> None:
    print(msg, flush=True)


def nc_limit(rough: float) -> float:
    return float(np.sqrt(2.0 * np.pi) * rough / 2.0)


def fix_oxide_nc(oxide) -> dict[str, float]:
    """Raise oxide thick to the NC floor for the current roughness; keep frozen."""
    rough = float(oxide.rough.value)
    floor = nc_limit(rough)
    thick0 = float(oxide.thick.value)
    thick = max(thick0, floor)
    oxide.thick.setp(value=thick, vary=False, bounds=(0.0, 40.0))
    oxide.rough.setp(value=rough, vary=False)
    info = {
        "thick_before": thick0,
        "thick_after": thick,
        "rough": rough,
        "nc_floor": floor,
        "was_valid": thick0 >= floor,
    }
    log(
        f"oxide NC fix: thick {thick0:.4f} -> {thick:.4f} A "
        f"(rough={rough:.4f}, NC floor={floor:.4f}); frozen"
    )
    return info


def geometry_bounds(film) -> dict[str, tuple[float, float]]:
    """Local film bounds; ``tau_vac`` explicitly opened past the graded ceiling."""
    bounds = legacy.geometry_bounds_for(film)
    bounds["tau_vac"] = TAU_VAC_BOUNDS
    return bounds


def open_single_energy(obj, film, bounds: dict[str, tuple[float, float]]) -> None:
    cv.freeze_all_parameters(obj)
    for name in OPEN_GEOMETRY:
        cv.safely_setp_param(getattr(film, name), vary=True, bounds=bounds[name])
    for name in cv.FREE_INSTRUMENT:
        cv.safely_setp_param(
            getattr(obj.model, name),
            vary=True,
            bounds=legacy.instrument_bounds_for(obj.model)[name],
        )
    cv.tighten_theta_offsets({ENERGY: obj}, (ENERGY,))


def _material(slab, name: str) -> MaterialSLD:
    sld = slab.sld
    return MaterialSLD(
        sld._compound, density=sld.density, name=name, energy_offset=sld.energy_offset
    )


def build_gpu_objective(obj) -> Objective:
    """Single-energy refloxide GPU objective sharing film Parameter objects."""
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
        transform=cv.TRANSFORM,
        anisotropy_weight=float(obj.logp_anisotropy_weight),
        anisotropy_targets=targets,
        normalization="energy",
        nc_constraint=True,
    )
    objective.refresh_parameters()
    return objective


def run_lbfgsb_chunked(
    objective: Objective,
    *,
    chunk_iters: int = 25,
    max_chunks: int = 20,
    slope_rel: float = 2e-3,
    slope_abs: float = 0.05,
) -> list[dict]:
    history: list[dict] = []
    objective.model.device = "gpu"
    nll_prev = float(objective.nll())
    history.append(
        {
            "chunk": 0,
            "nll": nll_prev,
            "delta_nll": 0.0,
            "nfev": 0,
            "wall_s": 0.0,
            "stage": "start",
        }
    )
    log(f"start nll={nll_prev:.6f}  device={objective.model.device}")

    for chunk in range(1, max_chunks + 1):
        t0 = time.perf_counter()
        result = minimize_lbfgsb(
            objective,
            polish=False,
            options={"maxiter": chunk_iters, "ftol": 1e-12},
        )
        nll = float(objective.nll())
        delta = nll_prev - nll
        wall = time.perf_counter() - t0
        thresh = max(slope_abs, slope_rel * max(abs(nll), 1.0))
        row = {
            "chunk": chunk,
            "nll": nll,
            "delta_nll": delta,
            "nfev": int(result.nfev),
            "wall_s": wall,
            "message": str(result.message),
            "stage": "gpu",
        }
        history.append(row)
        log(
            f"  chunk {chunk}  nll={nll:.6f}  Δnll={delta:+.6f}  "
            f"nfev={result.nfev}  wall={wall:.2f}s"
        )
        scale_s = float(objective.model.scale_s.at(ENERGY).value)
        scale_p = float(objective.model.scale_p.at(ENERGY).value)
        if scale_s < 0.05 or scale_p < 0.05:
            log("ABORT: scale collapsed")
            break
        if abs(delta) < thresh:
            log(f"early stop: |Δnll|={abs(delta):.6g} < {thresh:.4g}")
            break
        nll_prev = nll

    # CPU f64 polish
    t0 = time.perf_counter()
    device = objective.model.device
    parallel = objective.model.parallel
    try:
        objective.model.device = "cpu"
        objective.model.parallel = True
        result = minimize_lbfgsb(
            objective,
            polish=True,
            options={"maxiter": 200, "ftol": 1e-14},
        )
    finally:
        objective.model.device = device
        objective.model.parallel = parallel
    nll = float(objective.nll())
    history.append(
        {
            "chunk": len(history),
            "nll": nll,
            "delta_nll": nll_prev - nll,
            "nfev": int(result.nfev),
            "wall_s": time.perf_counter() - t0,
            "message": str(result.message),
            "stage": "cpu_polish",
        }
    )
    log(f"cpu polish  nll={nll:.6f}  wall={history[-1]['wall_s']:.2f}s")
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={OUT_DIR}")

    objectives = cv.seed_objectives()
    obj = objectives[ENERGY]
    comps = obj.model.structure.components
    vacuum, film, oxide, substrate = comps
    _ = vacuum, substrate

    oxide_info = fix_oxide_nc(oxide)
    bounds = geometry_bounds(film)
    open_single_energy(obj, film, bounds)

    log(
        f"tau_vac={float(film.tau_vac.value):.4f} "
        f"bounds=({float(film.tau_vac.bounds.lb):.1f}, {float(film.tau_vac.bounds.ub):.1f})"
    )
    log("varying (legacy):")
    for p in obj.parameters.flattened():
        if p.vary and p.constraint is None:
            log(
                f"  {p.name}={float(p.value):.6g} "
                f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
            )

    objective = build_gpu_objective(obj)
    # Film vary flags already set on shared Parameters; instruments copied with vary.
    objective.refresh_parameters()
    before = metrics(objective)
    log(f"before: {before}")
    log(f"gpu varying ({len(objective.varying_parameters())}):")
    for p in objective.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
        )

    history = run_lbfgsb_chunked(objective)
    after = metrics(objective)

    # Sync instruments back to legacy for pickle consumers.
    for field in INSTRUMENT_FIELDS:
        getattr(obj.model, field).value = float(
            getattr(objective.model, field).at(ENERGY).value
        )

    film_snap = {name: float(getattr(film, name).value) for name in OPEN_GEOMETRY}
    inst_snap = {
        name: float(getattr(obj.model, name).value) for name in INSTRUMENT_FIELDS
    }
    summary = {
        "energy": ENERGY,
        "transform": "YX4",
        "tau_vac_bounds": list(TAU_VAC_BOUNDS),
        "oxide": oxide_info,
        "before": before,
        "after": after,
        "history": history,
        "film": film_snap,
        "instrument": inst_snap,
        "protocol": "gpu_chunked_lbfgs_2837_nc_tau_refit",
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

    log(f"after: {after}")
    log(
        f"film: tau_vac={film_snap['tau_vac']:.3f}  "
        f"total_thick={film_snap['total_thick']:.3f}  "
        f"surface_roughness={film_snap['surface_roughness']:.3f}"
    )
    log(
        f"oxide: thick={float(oxide.thick.value):.3f}  "
        f"rough={float(oxide.rough.value):.3f}"
    )
    log(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

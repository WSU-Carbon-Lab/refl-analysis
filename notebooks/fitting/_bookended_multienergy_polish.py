"""Joint L-BFGS-B polish of all free model parameters (YX4).

Warm-starts from the graded 283.7 eV MLE geometry and instruments (not the
neutral scale=1 / theta=0 seed). Other energies keep their latest polished
instruments. Opens shared film geometry and per-energy ``scale_*`` /
``theta_offset_*``, keeps calibrated ``energy_offset`` / ``dq`` / ``q_offset`` /
``bkg`` fixed, and runs chunked GlobalObjective L-BFGS-B (10 steps at a time)
with local bounds, stopping early when the nll slope flattens.

Writes ``@models/fitting_results/bookended_multienergy_polish/``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from refnx.analysis import CurveFitter
from utils.pyref_patch import patch_pyref

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CV_PATH = Path(__file__).resolve().with_name("_bookended_resonant_cv.py")
_spec = importlib.util.spec_from_file_location("_bookended_resonant_cv", _CV_PATH)
cv = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cv
assert _spec.loader is not None
_spec.loader.exec_module(cv)

OUT_DIR = cv.models_fitting_results / "bookended_multienergy_polish"
OPEN_GEOMETRY = (
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
cv.FREE_INSTRUMENT = ("scale_s", "scale_p", "theta_offset_s", "theta_offset_p")
SNAPSHOT_INSTRUMENT = cv.FREE_INSTRUMENT
# Each chunk restarts L-BFGS-B (curvature memory is lost), so chunks are long.
CHUNK_ITERS = 50
MAX_CHUNKS = 30
# Premature stop: relative and absolute nll improvement over one chunk.
SLOPE_REL = 1e-6
SLOPE_ABS = 1e-3
N_THREADS = max(1, os.cpu_count() or 4)


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def local_bounds(
    value: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    rel: float | None = 0.15,
    abs_pad: float | None = None,
) -> tuple[float, float]:
    """Build a box around ``value``, then clip to optional physical ``lo``/``hi``."""
    v = float(value)
    if abs_pad is not None:
        a, b = v - abs_pad, v + abs_pad
    else:
        assert rel is not None
        span = max(abs(v) * float(rel), 1e-6)
        a, b = v - span, v + span
    if lo is not None:
        a = max(a, float(lo))
    if hi is not None:
        b = min(b, float(hi))
    if a >= b:
        mid = 0.5 * ((lo if lo is not None else a) + (hi if hi is not None else b))
        width = max(abs(mid) * 0.05, 1e-4)
        a, b = mid - width, mid + width
        if lo is not None:
            a = max(a, float(lo))
        if hi is not None:
            b = min(b, float(hi))
    return float(a), float(b)


def geometry_bounds_for(film) -> dict[str, tuple[float, float]]:
    return {
        "total_thick": local_bounds(
            film.total_thick.value, lo=150.0, hi=220.0, rel=0.12
        ),
        "surface_roughness": local_bounds(
            film.surface_roughness.value, lo=0.0, hi=15.0, abs_pad=2.0, rel=None
        ),
        "density_bulk": local_bounds(film.density_bulk.value, lo=0.5, hi=2.5, rel=0.15),
        "density_si": local_bounds(film.density_si.value, lo=0.3, hi=2.0, rel=0.15),
        "density_vac": local_bounds(film.density_vac.value, lo=0.5, hi=2.5, rel=0.15),
        "tau_si": local_bounds(film.tau_si.value, lo=0.5, hi=60.0, rel=0.35),
        "tau_vac": (8.0, 55.0),
        "alpha_bulk": local_bounds(
            film.alpha_bulk.value, lo=0.0, hi=float(np.pi / 2), abs_pad=0.25, rel=None
        ),
        "alpha_si": local_bounds(
            film.alpha_si.value, lo=0.0, hi=float(np.pi / 2), abs_pad=0.25, rel=None
        ),
        "alpha_vac": local_bounds(
            film.alpha_vac.value, lo=0.0, hi=float(np.pi / 2), abs_pad=0.25, rel=None
        ),
    }


def instrument_bounds_for(model) -> dict[str, tuple[float, float]]:
    return {
        "scale_s": local_bounds(model.scale_s.value, lo=0.5, hi=2.1, rel=0.25),
        "scale_p": local_bounds(model.scale_p.value, lo=0.5, hi=2.1, rel=0.25),
        "theta_offset_s": local_bounds(
            model.theta_offset_s.value, lo=-0.8, hi=0.8, abs_pad=0.35, rel=None
        ),
        "theta_offset_p": local_bounds(
            model.theta_offset_p.value, lo=-0.8, hi=0.8, abs_pad=0.35, rel=None
        ),
    }


def geometry_snapshot(obj_map: dict) -> dict[str, float]:
    film = cv.film_of(obj_map[cv.ANCHOR])
    return {name: float(getattr(film, name).value or 0.0) for name in OPEN_GEOMETRY}


def instrument_snapshot(obj_map: dict) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for energy, obj in obj_map.items():
        out[float(energy)] = {
            name: float(getattr(obj.model, name).value or 0.0)
            for name in SNAPSHOT_INSTRUMENT
        }
    return out


def raise_theta_floors_keep_ub(obj_map: dict) -> None:
    """Raise theta lower bounds for grazing safety without widening the local ub."""
    for energy in cv.E_RES:
        obj = obj_map[float(energy)]
        model = obj.model
        for pol, field in (("s", "theta_offset_s"), ("p", "theta_offset_p")):
            pol_data = getattr(obj.data, pol)
            q = np.asarray(pol_data.x, dtype=float)
            if q.size == 0:
                continue
            grazing = cv.min_data_grazing_angle_deg(q, float(energy))
            safe_lb = cv.MIN_INCIDENT_THETA_DEG + cv.THETA_OFFSET_MARGIN_DEG - grazing
            param = getattr(model, field)
            lb_old, ub = float(param.bounds.lb), float(param.bounds.ub)
            new_lb = max(lb_old, float(safe_lb))
            if new_lb > ub:
                new_lb = ub
            kwargs: dict = {"bounds": (new_lb, ub), "vary": bool(param.vary)}
            value = float(param.value or 0.0)
            if value < new_lb:
                kwargs["value"] = new_lb
            cv.safely_setp_param(param, **kwargs)
            log(
                f"  theta floor {float(energy):g} eV {field}: "
                f"lb={new_lb:.4f} (data_min={grazing:.4f} deg, value={value:.4f})"
            )


def open_all_with_local_bounds(obj_map: dict) -> dict[str, tuple[float, float]]:
    """Open shared geometry + per-energy instruments; return bound map for logging."""
    for energy in cv.E_RES:
        cv.freeze_all_parameters(obj_map[float(energy)])

    film = cv.film_of(obj_map[cv.ANCHOR])
    # density_bulk is not in CV GEOMETRY_PARAMS, so link it across energies here.
    for energy in cv.E_RES:
        if np.isclose(float(energy), cv.ANCHOR):
            continue
        film_b = cv.film_of(obj_map[float(energy)])
        film_b.density_bulk.constraint = film.density_bulk

    geom_bounds = geometry_bounds_for(film)
    for name in OPEN_GEOMETRY:
        cv.safely_setp_param(getattr(film, name), vary=True, bounds=geom_bounds[name])

    inst_bounds_log: dict[str, tuple[float, float]] = {
        f"geom.{k}": v for k, v in geom_bounds.items()
    }
    for energy in cv.E_RES:
        model = obj_map[float(energy)].model
        bounds = instrument_bounds_for(model)
        for name in cv.FREE_INSTRUMENT:
            cv.safely_setp_param(getattr(model, name), vary=True, bounds=bounds[name])
            inst_bounds_log[f"{energy:g}.{name}"] = bounds[name]

    raise_theta_floors_keep_ub(obj_map)
    for energy in cv.E_RES:
        model = obj_map[float(energy)].model
        for name in ("theta_offset_s", "theta_offset_p"):
            p = getattr(model, name)
            inst_bounds_log[f"{energy:g}.{name}"] = (
                float(p.bounds.lb),
                float(p.bounds.ub),
            )

    cv.assert_energy_offset_fixed(obj_map)
    return inst_bounds_log


def score_map(obj_map: dict) -> tuple[float, float, list[dict], dict, dict]:
    for energy in cv.E_RES:
        cv.freeze_all_parameters(obj_map[float(energy)])
    geom = geometry_snapshot(obj_map)
    inst = instrument_snapshot(obj_map)
    per = cv.per_energy_rows(obj_map, inst)
    glob = cv.eval_global(obj_map, cv.E_RES)
    return float(glob.chisqr()), float(glob.logl()), per, inst, geom


def enable_parallel_kernels() -> None:
    """Use all CPUs in the Rust reflectivity kernel for this process.

    The fused book-ended path skips transmission (``ReflectModel.model``
    discards it) so reflectance comes from refloxide's reflectance-only
    recursion: about 4x faster per ``logl`` and equal to the 4x4 result.
    """
    os.environ["RAYON_NUM_THREADS"] = str(N_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
    patch_pyref(use_rust=True, parallel=True, with_transmission=False)
    log(f"kernel parallel=True  reflectance-only  RAYON/OMP threads={N_THREADS}")


def load_warm_start() -> tuple[dict[str, float], dict[float, dict[str, float]]]:
    """Graded geometry + pre-joint instruments (283.7 = graded MLE, not 0/1).

    Prefers ``geometry_before_after.csv`` phase=before (graded MLE geometry)
    and ``instruments.csv`` phase=before (theta-only polish: graded 283.7,
    other energies scale=1 with polished theta). Falls back to geometry.csv
    and the before instrument rows.
    """
    geom_ba = OUT_DIR / "geometry_before_after.csv"
    geom_path = OUT_DIR / "geometry.csv"
    inst_path = OUT_DIR / "instruments.csv"
    if not inst_path.exists():
        raise FileNotFoundError(f"missing {inst_path}; run a prior polish first")

    if geom_ba.exists():
        gdf = pd.read_csv(geom_ba)
        before = gdf.query("phase == 'before'")
        geom_row = before.iloc[0] if not before.empty else gdf.iloc[0]
    elif geom_path.exists():
        geom_row = pd.read_csv(geom_path).iloc[0]
    else:
        raise FileNotFoundError(f"missing geometry CSVs in {OUT_DIR}")

    geometry = {name: float(geom_row[name]) for name in OPEN_GEOMETRY}
    inst_df = pd.read_csv(inst_path)
    # Explicitly the pre-joint / theta-only state — never the joint "after"
    # rows that drifted 283.7 away from the graded MLE.
    before_inst = inst_df.query("phase == 'before'")
    if before_inst.empty:
        raise RuntimeError(f"no phase=='before' rows in {inst_path}")
    instrument: dict[float, dict[str, float]] = {}
    for _, row in before_inst.iterrows():
        energy = float(row["energy"])
        instrument[energy] = {name: float(row[name]) for name in SNAPSHOT_INSTRUMENT}
    return geometry, instrument


def assert_anchor_not_neutral(instrument: dict[float, dict[str, float]]) -> None:
    """Fail fast if 283.7 was reset to scale=1 / theta=0."""
    anchor = instrument[float(cv.ANCHOR)]
    scale_neutral = (
        abs(anchor["scale_s"] - 1.0) < 1e-6 and abs(anchor["scale_p"] - 1.0) < 1e-6
    )
    theta_neutral = (
        abs(anchor["theta_offset_s"]) < 1e-6 and abs(anchor["theta_offset_p"]) < 1e-6
    )
    if scale_neutral and theta_neutral:
        raise RuntimeError(
            "283.7 eV instruments look neutral (scale=1, theta=0); "
            "expected graded MLE seed"
        )


def run_chunked_lbfgs(glob, *, fitter: CurveFitter) -> list[dict[str, float]]:
    """Run L-BFGS in CHUNK_ITERS bursts; stop when nll slope flattens."""
    history: list[dict[str, float]] = []
    nll_prev = float(-glob.logl())
    history.append(
        {
            "chunk": 0,
            "nll": nll_prev,
            "chi2": float(glob.chisqr()),
            "logl": float(glob.logl()),
            "delta_nll": 0.0,
        }
    )
    log(
        f"chunked L-BFGS: {CHUNK_ITERS} iters/chunk, max_chunks={MAX_CHUNKS}, "
        f"stop if |Δnll| < max({SLOPE_ABS}, {SLOPE_REL}*|nll|)"
    )
    log(f"  chunk 0  nll={nll_prev:.6f}  chi2={history[0]['chi2']:.3f}")

    for chunk in range(1, MAX_CHUNKS + 1):
        t_chunk = time.perf_counter()
        fitter.fit(
            "L-BFGS-B",
            target="nll",
            options={"maxiter": CHUNK_ITERS, "ftol": 1e-12},
        )
        nll = float(-glob.logl())
        chi2 = float(glob.chisqr())
        logl = float(glob.logl())
        delta = nll_prev - nll
        wall_c = time.perf_counter() - t_chunk
        history.append(
            {
                "chunk": float(chunk),
                "nll": nll,
                "chi2": chi2,
                "logl": logl,
                "delta_nll": delta,
                "wall_s": wall_c,
            }
        )
        thresh = max(SLOPE_ABS, SLOPE_REL * max(abs(nll), 1.0))
        log(
            f"  chunk {chunk}  nll={nll:.6f}  Δnll={delta:+.6f}  "
            f"chi2={chi2:.3f}  wall={wall_c:.1f}s  thresh={thresh:.4g}"
        )
        if chunk >= 1 and abs(delta) < thresh:
            log(
                f"early stop: |Δnll|={abs(delta):.6g} < {thresh:.4g} "
                f"(premature convergence)"
            )
            break
        nll_prev = nll
    return history


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enable_parallel_kernels()
    log(f"output: {OUT_DIR}")
    log("protocol: chunked joint L-BFGS-B, 283.7 seeded from graded MLE")
    log(f"open geometry: {OPEN_GEOMETRY}")
    log(f"open instrument/energy: {cv.FREE_INSTRUMENT}")
    log(f"calibrated fixed: {cv.CALIBRATED_INSTRUMENT}")

    warm_geom, warm_inst = load_warm_start()
    assert_anchor_not_neutral(warm_inst)
    log("warm-start geometry (graded MLE):")
    for k, v in warm_geom.items():
        log(f"  {k}={v:.6g}")
    log("warm-start 283.7 instruments (graded MLE):")
    for k, v in warm_inst[float(cv.ANCHOR)].items():
        log(f"  {k}={v:.6g}")

    objectives = cv.seed_objectives()
    # Capture graded 283.7 instruments from the seed before any snapshot apply.
    graded_anchor = {
        name: float(getattr(objectives[float(cv.ANCHOR)].model, name).value or 0.0)
        for name in SNAPSHOT_INSTRUMENT
    }
    obj_map = cv.build_yx4_objectives(objectives)
    cv.apply_geometry_snapshot(obj_map, warm_geom)
    cv.apply_instrument_snapshot(obj_map, warm_inst, cv.E_RES)
    # Hard-pin 283.7 to graded MLE instruments (overrides any CSV drift).
    for name, value in graded_anchor.items():
        cv.safely_setp_param(
            getattr(obj_map[float(cv.ANCHOR)].model, name), value=value
        )
    warm_inst[float(cv.ANCHOR)] = dict(graded_anchor)
    log("forced 283.7 instruments from seed_objectives graded MLE:")
    for k, v in graded_anchor.items():
        log(f"  {k}={v:.6g}")

    for _e, _obj in obj_map.items():
        cv.pin_calibrated_instrument(_obj)

    chi2_0, logl_0, per_0, inst_0, geom_0 = score_map(obj_map)
    log(f"\nbefore L-BFGS: chi2_all={chi2_0:.3f}  logl_all={logl_0:.3f}")
    for row in per_0:
        log(f"  {row['energy']:6.1f} eV  chi2={row['chi2']:.3f}  rms={row['rms']:.4g}")
    a283 = inst_0[float(cv.ANCHOR)]
    if abs(a283["scale_s"] - 1.0) < 1e-6 and abs(a283["theta_offset_s"]) < 1e-6:
        raise RuntimeError("283.7 still neutral after setup")

    bounds_log = open_all_with_local_bounds(obj_map)
    glob = cv.eval_global(obj_map, cv.E_RES)
    varying = list(glob.varying_parameters())
    n_geom = len(OPEN_GEOMETRY)
    n_inst = len(cv.E_RES) * len(cv.FREE_INSTRUMENT)
    expected = n_geom + n_inst
    if len(varying) != expected:
        raise RuntimeError(
            f"expected {expected} free params "
            f"({n_geom} geom + {n_inst} inst), got {len(varying)}: "
            f"{[p.name for p in varying]}"
        )
    log(f"\nL-BFGS-B DOF={len(varying)}")
    for name, (lo, hi) in bounds_log.items():
        log(f"  bounds {name}: ({lo:.5g}, {hi:.5g})")

    t0 = time.perf_counter()
    fitter = CurveFitter(glob)
    chunk_hist = run_chunked_lbfgs(glob, fitter=fitter)
    wall = time.perf_counter() - t0
    log(
        f"\nafter L-BFGS: chi2={glob.chisqr():.4f}  "
        f"logl={glob.logl():.4f}  wall={wall:.1f}s  chunks={len(chunk_hist) - 1}"
    )

    chi2_1, logl_1, per_1, inst_1, geom_1 = score_map(obj_map)
    log(f"\nafter score: chi2_all={chi2_1:.3f}  logl_all={logl_1:.3f}")
    for row in per_1:
        log(f"  {row['energy']:6.1f} eV  chi2={row['chi2']:.3f}  rms={row['rms']:.4g}")
    log("geometry after:")
    for k, v in geom_1.items():
        log(f"  {k}={v:.6g}  (was {geom_0[k]:.6g})")
    log("283.7 instruments after:")
    for k, v in inst_1[float(cv.ANCHOR)].items():
        log(f"  {k}={v:.6g}  (was {inst_0[float(cv.ANCHOR)][k]:.6g})")

    before_df = pd.DataFrame([{"phase": "before", **r} for r in per_0])
    after_df = pd.DataFrame([{"phase": "after", **r} for r in per_1])
    pd.concat([before_df, after_df], ignore_index=True).to_csv(
        OUT_DIR / "per_energy.csv", index=False
    )
    pd.DataFrame([geom_1]).to_csv(OUT_DIR / "geometry.csv", index=False)
    pd.DataFrame([{"phase": "before", **geom_0}, {"phase": "after", **geom_1}]).to_csv(
        OUT_DIR / "geometry_before_after.csv", index=False
    )

    inst_rows = []
    for phase, snap in (("before", inst_0), ("after", inst_1)):
        for energy, vals in snap.items():
            inst_rows.append({"phase": phase, "energy": energy, **vals})
    pd.DataFrame(inst_rows).to_csv(OUT_DIR / "instruments.csv", index=False)
    pd.DataFrame(chunk_hist).to_csv(OUT_DIR / "lbfgs_chunks.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "phase": "before",
                "chi2_all": chi2_0,
                "logl_all": logl_0,
                "rms_283p7": next(
                    r["rms"] for r in per_0 if np.isclose(r["energy"], 283.7)
                ),
            },
            {
                "phase": "after",
                "chi2_all": chi2_1,
                "logl_all": logl_1,
                "rms_283p7": next(
                    r["rms"] for r in per_1 if np.isclose(r["energy"], 283.7)
                ),
                "wall_s": wall,
                "nparams": expected,
                "n_chunks": len(chunk_hist) - 1,
            },
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    bundle = {
        "E_RES": cv.E_RES,
        "ANCHOR": cv.ANCHOR,
        "OPEN_GEOMETRY": OPEN_GEOMETRY,
        "FREE_INSTRUMENT": cv.FREE_INSTRUMENT,
        "geometry_before": geom_0,
        "geometry_after": geom_1,
        "instrument_before": inst_0,
        "instrument_after": inst_1,
        "per_energy_before": per_0,
        "per_energy_after": per_1,
        "obj_map": obj_map,
        "chi2_before": chi2_0,
        "chi2_after": chi2_1,
        "logl_before": logl_0,
        "logl_after": logl_1,
        "bounds": {k: list(v) for k, v in bounds_log.items()},
        "nparams": expected,
        "chunk_history": chunk_hist,
        "graded_anchor_seed": graded_anchor,
    }
    with (OUT_DIR / "bundle.pkl").open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    (OUT_DIR / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "protocol": "lbfgs_chunked_joint_graded_anchor_seed",
                "anchor": cv.ANCHOR,
                "energies": list(cv.E_RES),
                "open_geometry": list(OPEN_GEOMETRY),
                "free_instrument": list(cv.FREE_INSTRUMENT),
                "nparams": expected,
                "chunk_iters": CHUNK_ITERS,
                "n_chunks": len(chunk_hist) - 1,
                "transform": "YX4",
                "parallel": True,
                "n_threads": N_THREADS,
                "chi2_before": chi2_0,
                "chi2_after": chi2_1,
                "wall_s": wall,
                "graded_anchor_seed": graded_anchor,
            },
            indent=2,
        )
    )
    log(f"\nDONE. Bundle: {OUT_DIR / 'bundle.pkl'}")
    log("\nRMS before -> after:")
    for e in cv.E_RES:
        b = next(r["rms"] for r in per_0 if np.isclose(r["energy"], e))
        a = next(r["rms"] for r in per_1 if np.isclose(r["energy"], e))
        log(f"  {e:6.1f} eV  {b:.4g} -> {a:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

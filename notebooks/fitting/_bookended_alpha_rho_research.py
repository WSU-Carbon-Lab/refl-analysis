"""Research: multi-energy bookended fit with only regional alpha and rho free.

Keeps the 283.7 eV MLE for thickness, roughness, and both tau decay lengths.
Frees ``density_{bulk,si,vac}`` and ``alpha_{bulk,si,vac}`` inside strong boxes
around that MLE, with extra substrate-side physics:

- ``density_si < 1`` (underdense Si-side bookend)
- ``alpha_si`` near 0 and well below ``pi/2`` (not upright at the substrate)

Compares:
1. baseline — geometry fully frozen; polish scale/theta at every energy
2. alpha_rho — DE on the six free film params (all energies), then instrument polish

Outputs under ``@models/fitting_results/bookended_alpha_rho_research/``.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_CV_PATH = Path(__file__).resolve().with_name("_bookended_resonant_cv.py")
_spec = importlib.util.spec_from_file_location("_bookended_resonant_cv", _CV_PATH)
cv = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cv
assert _spec.loader is not None
_spec.loader.exec_module(cv)

OUT_DIR = cv.models_fitting_results / "bookended_alpha_rho_research"

# Regional orientation (alpha) and density (rho); taus / thick / rough stay fixed.
FREE_FILM = (
    "density_bulk",
    "density_si",
    "density_vac",
    "alpha_bulk",
    "alpha_si",
    "alpha_vac",
)
FIXED_FILM = (
    "total_thick",
    "surface_roughness",
    "tau_si",
    "tau_vac",
)

DENSITY_REL = 0.15
ALPHA_DELTA = 0.15
# Soft boxes around the MLE, then clipped by these hard physical caps.
DENSITY_HARD = {
    "density_bulk": (1.0, 2.5),
    "density_si": (0.5, 1.0),  # substrate bookend: underdense, never >= 1
    "density_vac": (0.5, 2.5),
}
ALPHA_HARD = {
    "alpha_bulk": (0.0, float(np.pi / 2)),
    "alpha_si": (0.0, 0.55),  # near 0; far from pi/2 (and below old pi/4 cap)
    "alpha_vac": (0.0, float(np.pi / 2)),
}

DE_KW = {
    "popsize": 12,
    "maxiter": 40,
    "mutation": 0.7,
    "recombination": 0.7,
    "workers": 1,
    "seed": 1,
}


@dataclass
class RunResult:
    """One research configuration and its per-energy diagnostics."""

    name: str
    geometry: dict[str, float]
    free_params: tuple[str, ...]
    instrument: dict[float, dict[str, float]]
    per_energy: list[dict[str, float]]
    chi2_all: float
    logl_all: float
    wall_s: float


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def strong_bounds(anchor: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Build MLE-local boxes, clipped by substrate physics on the Si bookend.

    ``density_si`` stays strictly below 1. ``alpha_si`` stays near 0 (hard cap
    0.55 rad), not near ``pi/2``.
    """
    out: dict[str, tuple[float, float]] = {}
    for name in ("density_bulk", "density_si", "density_vac"):
        v = float(anchor[name])
        lo_s, hi_s = v * (1.0 - DENSITY_REL), v * (1.0 + DENSITY_REL)
        lo_h, hi_h = DENSITY_HARD[name]
        lo, hi = max(lo_h, lo_s), min(hi_h, hi_s)
        if lo >= hi:
            raise ValueError(
                f"empty density box for {name}: soft=({lo_s}, {hi_s}) "
                f"hard=({lo_h}, {hi_h}) anchor={v}"
            )
        out[name] = (lo, hi)
    for name in ("alpha_bulk", "alpha_si", "alpha_vac"):
        v = float(anchor[name])
        lo_s, hi_s = v - ALPHA_DELTA, v + ALPHA_DELTA
        lo_h, hi_h = ALPHA_HARD[name]
        lo, hi = max(lo_h, lo_s), min(hi_h, hi_s)
        if lo >= hi:
            raise ValueError(
                f"empty alpha box for {name}: soft=({lo_s}, {hi_s}) "
                f"hard=({lo_h}, {hi_h}) anchor={v}"
            )
        out[name] = (lo, hi)
    return out


def full_geometry_snapshot(obj_map: dict) -> dict[str, float]:
    """Snapshot free + fixed film params (includes density_bulk)."""
    film = cv.film_of(obj_map[cv.ANCHOR])
    names = FREE_FILM + FIXED_FILM
    return {name: float(getattr(film, name).value or 0.0) for name in names}


def apply_full_geometry(obj_map: dict, snap: dict[str, float]) -> None:
    film = cv.film_of(obj_map[cv.ANCHOR])
    for name, value in snap.items():
        cv.safely_setp_param(getattr(film, name), value=value)


def open_free_film(
    obj_map: dict, bounds: dict[str, tuple[float, float]], *, vary: bool
) -> None:
    film = cv.film_of(obj_map[cv.ANCHOR])
    for name in FREE_FILM + FIXED_FILM:
        p = getattr(film, name)
        if name in FREE_FILM and vary:
            cv.safely_setp_param(p, vary=True, bounds=bounds[name])
        else:
            cv.safely_setp_param(p, vary=False)


def link_all_film(obj_map: dict) -> None:
    """Constrain every non-anchor film parameter (free + fixed) to the anchor."""
    film_a = cv.film_of(obj_map[cv.ANCHOR])
    names = FREE_FILM + FIXED_FILM
    for energy, obj in obj_map.items():
        if np.isclose(float(energy), cv.ANCHOR):
            continue
        film_b = cv.film_of(obj)
        for name in names:
            getattr(film_b, name).constraint = getattr(film_a, name)


def build_objectives(source: dict) -> dict:
    out = cv.build_yx4_objectives(source)
    link_all_film(out)
    return out


def per_energy_table(obj_map: dict, instrument: dict) -> list[dict[str, float]]:
    return cv.per_energy_rows(obj_map, instrument)


def score_all(obj_map: dict) -> tuple[float, float, list[dict[str, float]], dict]:
    for energy in cv.E_RES:
        cv.freeze_all_parameters(obj_map[float(energy)])
        cv.open_instrument(obj_map[float(energy)].model, vary=False)
    open_free_film(obj_map, strong_bounds(full_geometry_snapshot(obj_map)), vary=False)
    inst = cv.instrument_snapshot(obj_map)
    per = per_energy_table(obj_map, inst)
    glob = cv.eval_global(obj_map, cv.E_RES)
    return float(glob.chisqr()), float(glob.logl()), per, inst


def run_baseline(objectives: dict, anchor: dict[str, float]) -> RunResult:
    """Frozen geometry at 283.7 MLE; polish instruments at every energy."""
    t0 = time.perf_counter()
    log("\n======== BASELINE: frozen geometry, instrument polish all E ========")
    obj_map = build_objectives(objectives)
    apply_full_geometry(obj_map, anchor)
    for _e, _obj in obj_map.items():
        cv.pin_calibrated_instrument(_obj)
    open_free_film(obj_map, strong_bounds(anchor), vary=False)
    cv.polish_instruments(obj_map, cv.E_RES, label="all-energies")
    chi2, logl, per, inst = score_all(obj_map)
    geom = full_geometry_snapshot(obj_map)
    result = RunResult(
        name="baseline_frozen_geom",
        geometry=geom,
        free_params=(),
        instrument=inst,
        per_energy=per,
        chi2_all=chi2,
        logl_all=logl,
        wall_s=time.perf_counter() - t0,
    )
    _log_result(result)
    return result


def run_alpha_rho(objectives: dict, anchor: dict[str, float]) -> RunResult:
    """DE on regional alpha/rho inside strong boxes; thick/rough/tau fixed."""
    t0 = time.perf_counter()
    bounds = strong_bounds(anchor)
    log(
        "\n======== ALPHA+RHO: DE free densities/alphas, "
        "tau/thick/rough fixed ========"
    )
    log(f"free: {FREE_FILM}")
    log(f"fixed: {FIXED_FILM}")
    for name, (lo, hi) in bounds.items():
        log(f"  bound {name}: [{lo:.5g}, {hi:.5g}]  anchor={anchor[name]:.5g}")

    obj_map = build_objectives(objectives)
    apply_full_geometry(obj_map, anchor)
    for _e, _obj in obj_map.items():
        cv.pin_calibrated_instrument(_obj)

    for energy in cv.E_RES:
        cv.freeze_all_parameters(obj_map[float(energy)])
        cv.open_instrument(obj_map[float(energy)].model, vary=False)
    open_free_film(obj_map, bounds, vary=True)

    train_obj = cv.train_objective(obj_map, cv.E_RES)
    varying = list(train_obj.varying_parameters())
    if len(varying) != len(FREE_FILM):
        raise RuntimeError(
            f"expected {len(FREE_FILM)} free film DOF, got {len(varying)}: "
            f"{[p.name for p in varying]}"
        )
    x0 = np.asarray([float(p.value) for p in varying], dtype=float)
    log(f"DE DOF={len(x0)}: {[p.name for p in varying]}")
    log(f"DE_KW={DE_KW}")

    de_state = {"ngen": 0, "t0": time.perf_counter()}

    def _cb(_xk, convergence):
        de_state["ngen"] += 1
        if de_state["ngen"] == 1 or de_state["ngen"] % 5 == 0:
            log(
                f"  DE gen={de_state['ngen']}  "
                f"convergence={float(convergence):.4g}  "
                f"elapsed={time.perf_counter() - de_state['t0']:.1f}s"
            )
        return False

    fitter = cv.CurveFitter(train_obj)
    fitter.fit(
        "differential_evolution",
        target="nll",
        x0=x0,
        polish=False,
        updating="deferred",
        callback=_cb,
        **DE_KW,
    )
    log(
        f"after DE: chi2={train_obj.chisqr():.4f}  "
        f"logl={train_obj.logl():.4f}  gens={de_state['ngen']}"
    )

    open_free_film(obj_map, bounds, vary=False)
    cv.polish_instruments(obj_map, cv.E_RES, label="all-energies")
    chi2, logl, per, inst = score_all(obj_map)
    geom = full_geometry_snapshot(obj_map)
    result = RunResult(
        name="alpha_rho_strong_bounds",
        geometry=geom,
        free_params=FREE_FILM,
        instrument=inst,
        per_energy=per,
        chi2_all=chi2,
        logl_all=logl,
        wall_s=time.perf_counter() - t0,
    )
    _log_result(result)
    _log_geometry_delta(anchor, geom)
    return result


def _log_result(result: RunResult) -> None:
    log(
        f"{result.name}: chi2_all={result.chi2_all:.3f}  "
        f"logl_all={result.logl_all:.3f}  wall={result.wall_s:.1f}s"
    )
    for row in result.per_energy:
        log(
            f"  {row['energy']:6.1f} eV  chi2={row['chi2']:.3f}  "
            f"rms={row['rms']:.4g}  logl={row['logl']:.3f}"
        )


def _log_geometry_delta(anchor: dict[str, float], geom: dict[str, float]) -> None:
    log("geometry delta vs 283.7 MLE:")
    for name in FREE_FILM + FIXED_FILM:
        a, b = float(anchor[name]), float(geom[name])
        log(f"  {name}: {a:.5g} -> {b:.5g}  (d={b - a:+.5g})")


def save_results(
    anchor: dict[str, float],
    baseline: RunResult,
    alpha_rho: RunResult,
    prior_stage5: pd.DataFrame | None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in (baseline, alpha_rho):
        for row in result.per_energy:
            rows.append({"run": result.name, **row})
    if prior_stage5 is not None:
        for _, row in prior_stage5.iterrows():
            rows.append(
                {
                    "run": "prior_cv_stage5",
                    "energy": float(row["energy"]),
                    "ndata": float(row["ndata"]),
                    "chi2": float(row["chi2"]),
                    "rms": float(row["rms"]),
                    "logl": float(row["logl"]),
                }
            )
    per_df = pd.DataFrame(rows)
    per_df.to_csv(OUT_DIR / "per_energy_compare.csv", index=False)

    geom_rows = [{"run": "anchor_283p7", **anchor}]
    for result in (baseline, alpha_rho):
        geom_rows.append({"run": result.name, **result.geometry})
    pd.DataFrame(geom_rows).to_csv(OUT_DIR / "geometry_compare.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "run": r.name,
                "chi2_all": r.chi2_all,
                "logl_all": r.logl_all,
                "rms_283p7": next(
                    x["rms"] for x in r.per_energy if np.isclose(x["energy"], 283.7)
                ),
                "rms_285p1": next(
                    x["rms"] for x in r.per_energy if np.isclose(x["energy"], 285.1)
                ),
                "wall_s": r.wall_s,
                "n_free_film": len(r.free_params),
            }
            for r in (baseline, alpha_rho)
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    bundle = {
        "anchor": anchor,
        "bounds": strong_bounds(anchor),
        "FREE_FILM": FREE_FILM,
        "FIXED_FILM": FIXED_FILM,
        "baseline": baseline,
        "alpha_rho": alpha_rho,
        "DE_KW": DE_KW,
    }
    with (OUT_DIR / "bundle.pkl").open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    (OUT_DIR / "status.json").write_text(
        json.dumps(
            {
                "complete": True,
                "runs": ["baseline_frozen_geom", "alpha_rho_strong_bounds"],
                "DENSITY_REL": DENSITY_REL,
                "ALPHA_DELTA": ALPHA_DELTA,
                "DENSITY_HARD": {k: list(v) for k, v in DENSITY_HARD.items()},
                "ALPHA_HARD": {k: list(v) for k, v in ALPHA_HARD.items()},
            },
            indent=2,
        )
    )
    log(f"\nWrote {OUT_DIR}")
    log("\nSummary:")
    log(summary.round(4).to_string(index=False))
    log("\nPer-energy RMS comparison:")
    piv = per_df.pivot_table(index="energy", columns="run", values="rms")
    log(piv.round(4).to_string())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"output: {OUT_DIR}")

    objectives = cv.seed_objectives()
    yx4 = build_objectives(objectives)
    # Full film snapshot from seeded graded@283.7 (includes density_bulk).
    anchor = full_geometry_snapshot(yx4)
    log("anchor geometry (283.7 MLE):")
    for k, v in anchor.items():
        log(f"  {k}={v:.6g}")

    prior_path = cv.OUT_DIR / "per_energy.csv"
    prior_stage5 = None
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        prior_stage5 = prior.loc[prior["stage"] == prior["stage"].max()].copy()

    baseline = run_baseline(objectives, anchor)
    alpha_rho = run_alpha_rho(objectives, anchor)
    save_results(anchor, baseline, alpha_rho, prior_stage5)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""GPU joint polish of the book-ended multi-energy fit on the refloxide stack.

Same warm start as ``_bookended_multienergy_polish.py``: graded 283.7 eV MLE
geometry and instruments (not scale=1 / θ=0), other energies from the latest
theta polish. Free parameters and local bounds match that script. Evaluation
goes through ``refloxide.model`` / ``refloxide.objective`` on ``device="gpu"``:

* one dispatch per objective evaluation across energies / polarizations
* L-BFGS-B via ``minimize_lbfgsb`` with exact ``nll_and_grad`` (chunked:
  10 iterations at a time, early stop on flat nll slope, optional f64 CPU polish)
* optional DE / MCMC via ``gpu_batch``

YX4 transform, anisotropy blend, and per-energy normalization match the
legacy GlobalObjective. Results are also written in the legacy CSV/bundle
layout for ``bookended_resonant_init.ipynb``.

Writes ``@models/fitting_results/bookended_multienergy_polish_gpu/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel, Slab
from refloxide.objective import Objective, gpu_batch, minimize_lbfgsb
from refnx.analysis import CurveFitter

_POLISH_PATH = Path(__file__).resolve().with_name("_bookended_multienergy_polish.py")
_spec = importlib.util.spec_from_file_location(
    "_bookended_multienergy_polish", _POLISH_PATH
)
legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = legacy
assert _spec.loader is not None
_spec.loader.exec_module(legacy)
cv = legacy.cv

OUT_DIR = cv.models_fitting_results / "bookended_multienergy_polish_gpu"
INSTRUMENT_FIELDS = ("scale_s", "scale_p", "theta_offset_s", "theta_offset_p", "bkg")
EXPECTED_FREE = len(legacy.OPEN_GEOMETRY) + len(cv.E_RES) * len(cv.FREE_INSTRUMENT)


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def build_legacy_map() -> dict:
    """Legacy objectives with the polish script's warm start and local bounds."""
    warm_geom, warm_inst = legacy.load_warm_start()
    legacy.assert_anchor_not_neutral(warm_inst)
    objectives = cv.seed_objectives()
    graded_anchor = {
        name: float(getattr(objectives[float(cv.ANCHOR)].model, name).value or 0.0)
        for name in legacy.SNAPSHOT_INSTRUMENT
    }
    obj_map = cv.build_yx4_objectives(objectives)
    cv.apply_geometry_snapshot(obj_map, warm_geom)
    cv.apply_instrument_snapshot(obj_map, warm_inst, cv.E_RES)
    for name, value in graded_anchor.items():
        cv.safely_setp_param(
            getattr(obj_map[float(cv.ANCHOR)].model, name), value=value
        )
    for obj in obj_map.values():
        cv.pin_calibrated_instrument(obj)
    legacy.open_all_with_local_bounds(obj_map)
    return obj_map


def _material(slab, name: str) -> MaterialSLD:
    """``MaterialSLD`` sharing the legacy density and energy-offset Parameters."""
    sld = slab.sld
    return MaterialSLD(
        sld._compound, density=sld.density, name=name, energy_offset=sld.energy_offset
    )


def build_gpu_objective(obj_map: dict) -> Objective:
    """refloxide ``Objective`` equivalent to the legacy GlobalObjective.

    Geometry parameters are the anchor film's own ``Parameter`` objects, so
    their values and local bounds are shared with the legacy map.
    """
    comps = obj_map[float(cv.ANCHOR)].model.structure.components
    vacuum, film, oxide, substrate = comps
    # Slab(...) keeps Parameter vary flags; Scatterer.__call__ would force vary=True.
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
    energies = [float(e) for e in cv.E_RES]
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
        transform=cv.TRANSFORM,
        anisotropy_weight=weights.pop(),
        anisotropy_targets=targets,
        normalization="energy",
        nc_constraint=False,
    )
    objective.refresh_parameters()
    return objective


def copy_instruments_to_legacy(objective: Objective, obj_map: dict) -> None:
    """Write fitted per-energy instruments back onto the legacy models."""
    for energy in cv.E_RES:
        lm = obj_map[float(energy)].model
        for field in INSTRUMENT_FIELDS:
            value = float(getattr(objective.model, field).at(float(energy)).value)
            getattr(lm, field).value = value


def legacy_nll(obj_map: dict) -> float:
    return float(-cv.eval_global(obj_map, cv.E_RES).logl())


def snapshot(objective: Objective) -> dict[str, float]:
    return {p.name: float(p.value) for p in objective.varying_parameters()}


def run_lbfgsb_chunked(
    objective: Objective,
    *,
    chunk_iters: int = 10,
    max_chunks: int = 30,
    slope_rel: float = 2e-3,
    slope_abs: float = 0.05,
    final_polish: bool = True,
) -> list[dict]:
    """GPU L-BFGS-B in short bursts; stop when nll slope flattens.

    Each chunk is ``minimize_lbfgsb(..., polish=False, maxiter=chunk_iters)``
    on ``device="gpu"`` with exact ``nll_and_grad``. After early stop (or
    ``max_chunks``), an optional double-precision CPU polish finishes the
    search (``polish=True``).
    """
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
    log(
        f"chunked GPU L-BFGS: {chunk_iters} iters/chunk, max_chunks={max_chunks}, "
        f"stop if |Δnll| < max({slope_abs}, {slope_rel}*|nll|)"
    )
    log(f"  chunk 0  nll={nll_prev:.6f}  device={objective.model.device}")

    for chunk in range(1, max_chunks + 1):
        t = time.perf_counter()
        result = minimize_lbfgsb(
            objective,
            polish=False,
            options={"maxiter": chunk_iters, "ftol": 1e-12},
        )
        nll = float(objective.nll())
        delta = nll_prev - nll
        wall_c = time.perf_counter() - t
        thresh = max(slope_abs, slope_rel * max(abs(nll), 1.0))
        history.append(
            {
                "chunk": chunk,
                "nll": nll,
                "delta_nll": delta,
                "nfev": int(result.nfev),
                "wall_s": wall_c,
                "message": str(result.message),
                "stage": "gpu",
            }
        )
        log(
            f"  chunk {chunk}  nll={nll:.6f}  Δnll={delta:+.6f}  "
            f"nfev={result.nfev}  wall={wall_c:.2f}s  thresh={thresh:.4g}"
        )
        if abs(delta) < thresh:
            log(
                f"early stop: |Δnll|={abs(delta):.6g} < {thresh:.4g} "
                f"(premature convergence)"
            )
            break
        nll_prev = nll

    if final_polish:
        t = time.perf_counter()
        # One CPU f64 polish stage from the GPU solution (minimize_lbfgsb
        # with polish=True also re-runs a GPU stage first when device=gpu;
        # switch to cpu so only the double-precision finish runs).
        device = objective.model.device
        parallel = objective.model.parallel
        try:
            objective.model.device = "cpu"
            objective.model.parallel = True
            result = minimize_lbfgsb(
                objective,
                polish=False,
                options={"maxiter": 50, "ftol": 1e-12},
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
                "wall_s": time.perf_counter() - t,
                "message": str(result.message),
                "stage": "cpu_polish",
            }
        )
        log(
            f"  cpu polish  nll={nll:.6f}  Δnll={nll_prev - nll:+.6f}  "
            f"nfev={result.nfev}  wall={history[-1]['wall_s']:.2f}s"
        )
    return history


def write_legacy_compatible(
    objective: Objective,
    obj_map: dict,
    *,
    chi2_0: float,
    logl_0: float,
    per_0: list,
    inst_0: dict,
    geom_0: dict,
    chunk_hist: list[dict],
    wall: float,
) -> None:
    """Mirror the CPU polish CSV/bundle layout for the notebook overlays."""
    copy_instruments_to_legacy(objective, obj_map)
    chi2_1, logl_1, per_1, inst_1, geom_1 = legacy.score_map(obj_map)

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
    pd.DataFrame(
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
                "n_chunks": max(0, len(chunk_hist) - 1),
            },
        ]
    ).to_csv(OUT_DIR / "summary.csv", index=False)

    bundle = {
        "E_RES": cv.E_RES,
        "ANCHOR": cv.ANCHOR,
        "OPEN_GEOMETRY": legacy.OPEN_GEOMETRY,
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
        "chunk_history": chunk_hist,
        "protocol": "refloxide_gpu_chunked_lbfgs",
    }
    with (OUT_DIR / "bundle.pkl").open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log("\nRMS before -> after (legacy YX4 score):")
    for e in cv.E_RES:
        b = next(r["rms"] for r in per_0 if np.isclose(r["energy"], e))
        a = next(r["rms"] for r in per_1 if np.isclose(r["energy"], e))
        log(f"  {e:6.1f} eV  {b:.4g} -> {a:.4g}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--de", type=int, default=0, help="GPU DE generations first (0 = skip)"
    )
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument(
        "--mcmc", type=int, default=0, help="GPU MCMC steps last (0 = skip)"
    )
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument(
        "--chunk-iters",
        type=int,
        default=10,
        help="L-BFGS-B iterations per GPU chunk",
    )
    parser.add_argument("--max-chunks", type=int, default=30)
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="skip the final f64 CPU L-BFGS-B polish",
    )
    return parser.parse_args(argv)


def run_de(objective: Objective, generations: int, popsize: int) -> dict:
    """Differential evolution with one GPU dispatch per generation."""
    t = time.perf_counter()
    objective.model.device = "gpu"
    CurveFitter(objective).fit(
        "differential_evolution",
        workers=gpu_batch(objective),
        maxiter=generations,
        popsize=popsize,
        polish=False,
        seed=1,
        verbose=False,
    )
    stage = {"stage": "de_gpu", "wall_s": time.perf_counter() - t}
    stage["nll"] = objective.nll()
    log(f"DE ({generations} gen x {popsize}): {stage['wall_s']:.1f} s")
    log(f"  nll={stage['nll']:.6f}")
    return stage


def run_mcmc(objective: Objective, steps: int, walkers: int) -> dict:
    """MCMC with one GPU dispatch per ensemble step; restores the starting point."""
    x_opt = np.array(
        [float(p.value) for p in objective.varying_parameters()], dtype=float
    )
    t = time.perf_counter()
    objective.model.device = "gpu"
    fitter = CurveFitter(objective, nwalkers=walkers)
    fitter.sample(steps, pool=gpu_batch(objective), random_state=1, verbose=False)
    np.save(OUT_DIR / "chain.npy", fitter.chain)
    objective.setp(x_opt)
    stage = {"stage": "mcmc_gpu", "wall_s": time.perf_counter() - t}
    stage |= {"steps": steps, "walkers": walkers}
    log(f"MCMC ({steps} steps x {walkers} walkers): {stage['wall_s']:.1f} s")
    return stage


def write_outputs(objective: Objective, summary: dict, before: dict) -> None:
    after = snapshot(objective)
    log("parameters (before -> after):")
    names = list(dict.fromkeys([*before.keys(), *after.keys()]))
    for name in names:
        b = before.get(name)
        a = after.get(name)
        if b is None:
            log(f"  {name:<28} (new) -> {a:.6g}")
        elif a is None:
            log(f"  {name:<28} {b:.6g} -> (gone)")
        else:
            log(f"  {name:<28} {b:.6g} -> {a:.6g}")
    pd.DataFrame([{"phase": "before", **before}, {"phase": "after", **after}]).to_csv(
        OUT_DIR / "parameters.csv", index=False
    )
    if summary.get("stages"):
        pd.DataFrame(summary["stages"]).to_csv(OUT_DIR / "stages.csv", index=False)
    with (OUT_DIR / "objective.pkl").open("wb") as fh:
        pickle.dump(objective, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT_DIR / "status.json").write_text(json.dumps(summary, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"output: {OUT_DIR}")
    log("protocol: refloxide GPU chunked L-BFGS (graded 283.7 seed)")

    obj_map = build_legacy_map()
    # Score legacy warm-start before opening bounds mutate vary flags for GPU.
    chi2_0, logl_0, per_0, inst_0, geom_0 = legacy.score_map(obj_map)
    # Re-open with local bounds for the GPU objective (score_map freezes).
    legacy.open_all_with_local_bounds(obj_map)

    objective = build_gpu_objective(obj_map)
    varying = list(objective.varying_parameters())
    log(
        f"refloxide objective: {len(objective._dataset)} rows, "
        f"{len(varying)} free, device={objective.model.device}"
    )
    if len(varying) != EXPECTED_FREE:
        names = [p.name for p in varying]
        raise RuntimeError(f"expected {EXPECTED_FREE} free params, got {names}")

    a283 = inst_0[float(cv.ANCHOR)]
    log(
        f"283.7 seed: scale_s={a283['scale_s']:.4f}  scale_p={a283['scale_p']:.4f}  "
        f"ths={a283['theta_offset_s']:+.4f}  thp={a283['theta_offset_p']:+.4f}"
    )
    if abs(a283["scale_s"] - 1.0) < 1e-6 and abs(a283["theta_offset_s"]) < 1e-6:
        raise RuntimeError("283.7 instruments are neutral; expected graded MLE")

    summary: dict = {
        "complete": True,
        "protocol": "refloxide_gpu_chunked_lbfgs_graded_anchor",
        "energies": list(cv.E_RES),
        "nparams": len(varying),
        "chunk_iters": args.chunk_iters,
        "nll_before": float(objective.nll()),
        "nll_legacy_before": float(-logl_0),
        "chi2_legacy_before": chi2_0,
        "stages": [],
    }
    before = snapshot(objective)
    log(
        f"before: nll={summary['nll_before']:.6f}  "
        f"legacy nll={summary['nll_legacy_before']:.6f}  "
        f"legacy chi2={chi2_0:.3f}"
    )
    t0 = time.perf_counter()
    if args.de:
        summary["stages"].append(run_de(objective, args.de, args.popsize))

    chunk_hist = run_lbfgsb_chunked(
        objective,
        chunk_iters=args.chunk_iters,
        max_chunks=args.max_chunks,
        final_polish=not args.no_polish,
    )
    summary["stages"].extend(chunk_hist[1:])  # drop start row from stages table
    summary["chunk_history"] = chunk_hist

    if args.mcmc:
        summary["stages"].append(run_mcmc(objective, args.mcmc, args.walkers))

    wall = time.perf_counter() - t0
    summary["nll_after"] = float(objective.nll())
    summary["wall_s"] = wall
    log(f"after GPU fit: nll={summary['nll_after']:.6f}  wall={wall:.1f}s")

    write_legacy_compatible(
        objective,
        obj_map,
        chi2_0=chi2_0,
        logl_0=logl_0,
        per_0=per_0,
        inst_0=inst_0,
        geom_0=geom_0,
        chunk_hist=chunk_hist,
        wall=wall,
    )
    summary["nll_legacy_after"] = legacy_nll(obj_map)
    log(f"legacy nll (same params)={summary['nll_legacy_after']:.6f}")
    write_outputs(objective, summary, before)
    log(f"DONE. {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

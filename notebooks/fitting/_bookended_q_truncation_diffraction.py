"""Soft-curve q-truncation scan for diffraction-limit claims.

Re-polishes the locked multi-energy bookended ``logY`` model on
``q <= q_cut`` for ``q_cut`` from ``q_max(283.7)`` down to the smallest
``q_max`` among the other resonant energies, scores train vs held-out
high-``q`` points, and writes a scan table under
``@models/fitting_results/bookended_q_truncation_diffraction/``.
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
import pandas as pd
from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, MaterialSLD, ReflectModel, Slab
from refloxide.objective import Objective
from refnx.analysis import CurveFitter, Transform

_HERE = Path(__file__).resolve().parent
_POLISH = _HERE / "_bookended_multienergy_logY_polish.py"
_spec = importlib.util.spec_from_file_location(
    "_bookended_multienergy_logY_polish", _POLISH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load polish module from {_POLISH}")
polish_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = polish_mod
_spec.loader.exec_module(polish_mod)

cv = polish_mod.cv
legacy = polish_mod.legacy
ENERGIES = polish_mod.ENERGIES
ANCHOR = polish_mod.ANCHOR
RHO_SIO2 = polish_mod.RHO_SIO2
RHO_SI = polish_mod.RHO_SI
TAU_VAC_BOUNDS = polish_mod.TAU_VAC_BOUNDS
SURFACE_PARAMS = polish_mod.SURFACE_PARAMS
INSTRUMENT_FIELDS = polish_mod.INSTRUMENT_FIELDS
POLISH_DIR = polish_mod.OUT_DIR
TRANSFORM = Transform("logY")
Q_MIN = float(cv.Q_MIN)
OUT_DIR = cv.models_fitting_results / "bookended_q_truncation_diffraction"
FILM_NAMES = (
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
LOCKED_FILM = (
    "total_thick",
    "density_bulk",
    "density_si",
    "density_vac",
    "tau_si",
    "alpha_bulk",
    "alpha_si",
)
MIN_SCAN_STEPS = 2


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def _material(slab: Any, name: str) -> MaterialSLD:
    sld = slab.sld
    return MaterialSLD(
        sld._compound, density=sld.density, name=name, energy_offset=sld.energy_offset
    )


def q_max_of(obj: Any) -> float:
    """Return the union max measured ``q`` for s and p on ``obj.data``."""
    ds = obj.data
    qs = np.asarray(ds.s.x, dtype=float)
    qp = np.asarray(ds.p.x, dtype=float)
    vals = [float(a.max()) for a in (qs, qp) if a.size]
    if not vals:
        raise ValueError("empty s/p q arrays")
    return max(vals)


def scan_bounds(obj_map: dict[float, Any]) -> tuple[float, float, dict[float, float]]:
    """Compute ``q_hi`` (283.7) and ``q_lo`` (min other resonant ``q_max``)."""
    q_by_e = {float(e): q_max_of(obj_map[float(e)]) for e in ENERGIES}
    q_hi = float(q_by_e[float(ANCHOR)])
    others = [q for e, q in q_by_e.items() if not np.isclose(e, ANCHOR)]
    q_lo = float(min(others))
    return q_hi, q_lo, q_by_e


def q_cut_grid(q_hi: float, q_lo: float, n_steps: int) -> np.ndarray:
    """Build inclusive descending ``q_cut`` grid from ``q_hi`` to ``q_lo``."""
    if n_steps < MIN_SCAN_STEPS:
        raise ValueError(f"n_steps must be >= {MIN_SCAN_STEPS}")
    if q_hi <= q_lo:
        raise ValueError(f"q_hi ({q_hi}) must exceed q_lo ({q_lo})")
    return np.linspace(q_hi, q_lo, int(n_steps), dtype=float)


def apply_polish_seed(obj_map: dict[float, Any], summary: dict[str, Any]) -> Any:
    """Pin locks and free set from the locked multi-energy logY polish MLE."""
    _vacuum, film, oxide, substrate = obj_map[ANCHOR].model.structure.components
    film_mle = summary["film"]
    oxide_mle = summary["oxide"]

    oxide.sld.density.setp(value=RHO_SIO2, vary=False)
    substrate.sld.density.setp(value=RHO_SI, vary=False)

    for _energy, obj in obj_map.items():
        cv.freeze_all_parameters(obj)

    for name in LOCKED_FILM:
        cv.safely_setp_param(
            getattr(film, name), value=float(film_mle[name]), vary=False
        )
    oxide.thick.setp(value=float(oxide_mle["thick"]), vary=False)
    oxide.rough.setp(value=float(oxide_mle["rough"]), vary=False)

    bounds = {
        "tau_vac": TAU_VAC_BOUNDS,
        "alpha_vac": legacy.local_bounds(
            float(film_mle["alpha_vac"]),
            lo=0.0,
            hi=float(np.pi / 2),
            abs_pad=0.35,
            rel=None,
        ),
        "surface_roughness": legacy.local_bounds(
            float(film_mle["surface_roughness"]),
            lo=0.0,
            hi=15.0,
            abs_pad=3.0,
            rel=None,
        ),
    }
    for name in SURFACE_PARAMS:
        cv.safely_setp_param(
            getattr(film, name),
            value=float(film_mle[name]),
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
        cv.safely_setp_param(
            model.bkg,
            value=float(vals.get("bkg", 0.0)),
            vary=False,
        )
    cv.tighten_theta_offsets(obj_map, ENERGIES)
    return film


def _q_keep(
    q: np.ndarray,
    *,
    q_min: float,
    q_max: float | None,
    q_gt: float | None,
) -> np.ndarray:
    keep = q >= float(q_min)
    if q_max is not None:
        keep &= q <= float(q_max)
    if q_gt is not None:
        keep &= q > float(q_gt)
    return keep


def _make_reflect_model(obj_map: dict[float, Any]) -> ReflectModel:
    vacuum, film, oxide, substrate = obj_map[ANCHOR].model.structure.components
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
    return model


def _collect_windowed_data(
    obj_map: dict[float, Any],
    energies: list[float],
    *,
    q_min: float,
    q_max: float | None,
    q_gt: float | None,
) -> tuple[dict[str, list], dict[float, tuple[np.ndarray, np.ndarray]], set[float]]:
    rows: dict[str, list] = {k: [] for k in ("q", "energy", "pol", "r", "r_err")}
    targets: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    used_energies: set[float] = set()
    for energy in energies:
        data = obj_map[energy].data
        n_before = len(rows["q"])
        for pol in ("s", "p"):
            sub = getattr(data, pol)
            q = np.asarray(sub.x, dtype=float)
            keep = _q_keep(q, q_min=q_min, q_max=q_max, q_gt=q_gt)
            if not np.any(keep):
                continue
            rows["q"].append(q[keep])
            rows["energy"].append(np.full(int(keep.sum()), energy))
            rows["pol"].append(np.full(int(keep.sum()), pol, dtype=object))
            rows["r"].append(np.asarray(sub.y, dtype=float)[keep])
            rows["r_err"].append(np.asarray(sub.y_err, dtype=float)[keep])
        if len(rows["q"]) == n_before:
            continue
        used_energies.add(float(energy))
        q_ani = np.asarray(data.anisotropy.x, dtype=float)
        y_ani = np.asarray(data.anisotropy.y, dtype=float)
        m_ani = _q_keep(q_ani, q_min=q_min, q_max=q_max, q_gt=q_gt)
        if np.any(m_ani):
            targets[float(energy)] = (q_ani[m_ani], y_ani[m_ani])
    return rows, targets, used_energies


def build_logY_objective(  # noqa: N802
    obj_map: dict[float, Any],
    *,
    q_min: float = Q_MIN,
    q_max: float | None = None,
    q_gt: float | None = None,
    model: ReflectModel | None = None,
) -> Objective:
    """Build a multi-energy ``logY`` objective with optional q-window filters.

    Parameters
    ----------
    obj_map
        Per-energy pyref objectives supplying structure and datasets.
    q_min
        Inclusive lower q bound (Angstrom^-1).
    q_max
        Inclusive upper q bound; ``None`` keeps all q above ``q_min``.
    q_gt
        Exclusive lower bound used for holdout slices (``q > q_gt``).
    model
        Optional shared ``ReflectModel``; when omitted a new GPU model is built
        from the anchor structure and instrument values in ``obj_map``.

    Returns
    -------
    Objective
        refloxide multi-energy objective under ``Transform("logY")``.
    """
    energies = list(ENERGIES)
    if model is None:
        model = _make_reflect_model(obj_map)

    rows, targets, used_energies = _collect_windowed_data(
        obj_map, energies, q_min=q_min, q_max=q_max, q_gt=q_gt
    )
    if not rows["q"]:
        raise ValueError("q filter selected zero reflectivity points")

    dataset = ReflectDataset(**{k: np.concatenate(v) for k, v in rows.items()})
    weights = {float(obj_map[e].logp_anisotropy_weight) for e in used_energies}
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


def polish_cut(objective: Objective) -> dict[str, Any]:
    """L-BFGS-B polish free parameters on a truncated ``logY`` objective."""
    objective.model.device = "cpu"
    objective.model.parallel = True
    nll0 = float(objective.nll())
    t0 = time.perf_counter()
    result = CurveFitter(objective).fit(
        "L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-14},
        verbose=False,
    )
    nll1 = float(objective.nll())
    return {
        "nll_start": nll0,
        "nll": nll1,
        "delta_nll": nll0 - nll1,
        "nfev": int(getattr(result, "nfev", -1)),
        "wall_s": float(time.perf_counter() - t0),
        "message": str(getattr(result, "message", "")),
        "success": bool(getattr(result, "success", False)),
    }


def residual_metrics(objective: Objective) -> dict[str, float]:
    """Compute χ² and RMS on ``log10(R)`` residuals for ``objective``."""
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    yd, ed = TRANSFORM(q, r, r_err)
    yf, _ = TRANSFORM(q, y_fit)
    resid = (yd - yf) / np.maximum(ed, 1e-30)
    n = int(resid.size)
    chi2 = float(np.sum(resid**2))
    return {
        "n": float(n),
        "chi2": chi2,
        "rms_logR": float(np.sqrt(chi2 / max(n, 1))),
    }


def sync_obj_map_from_objective(
    obj_map: dict[float, Any], objective: Objective
) -> None:
    """Copy fitted instrument values from ``objective.model`` onto ``obj_map``."""
    for energy in ENERGIES:
        lm = obj_map[energy].model
        for field in INSTRUMENT_FIELDS:
            getattr(lm, field).value = float(
                getattr(objective.model, field).at(energy).value
            )


def _row_from_fit(
    *,
    film: Any,
    train: Objective,
    history: dict[str, Any],
    hold_m: dict[str, float],
    q_cut: float,
) -> dict[str, Any]:
    train_m = residual_metrics(train)
    nvary = len(train.varying_parameters())
    n_train = int(train_m["n"])
    chi2_train = float(train_m["chi2"])
    n_hold = int(hold_m["n"])
    chi2_hold = float(hold_m["chi2"])
    chi2_per_n_hold = float("nan") if n_hold == 0 else chi2_hold / float(n_hold)
    row: dict[str, Any] = {
        "q_cut": float(q_cut),
        "d_pi_over_q": float(np.pi / q_cut),
        "n_train": n_train,
        "n_holdout": n_hold,
        "nvary": int(nvary),
        "chi2_train": chi2_train,
        "redchi_train": float(chi2_train / max(n_train - nvary, 1)),
        "rms_logR_train": float(train_m["rms_logR"]),
        "chi2_holdout": chi2_hold,
        "chi2_per_n_holdout": chi2_per_n_hold,
        "redchi_holdout": chi2_per_n_hold,
        "rms_logR_holdout": float(hold_m["rms_logR"]),
        "nll": float(history["nll"]),
        "success": bool(history["success"]),
        "runtime_s": float(history["wall_s"]),
        "nfev": int(history["nfev"]),
        "message": str(history["message"]),
    }
    for name in FILM_NAMES:
        row[name] = float(getattr(film, name).value)
    return row


def fit_cut_objective(
    obj_map: dict[float, Any],
    summary: dict[str, Any],
    q_cut: float,
) -> tuple[dict[str, Any], Objective]:
    """Re-polish on ``q <= q_cut`` and return metrics plus the train objective.

    Parameters
    ----------
    obj_map
        Seeded per-energy objectives (geometry shared via structure).
    summary
        Locked multi-energy ``logY`` polish summary used as the warm start.
    q_cut
        Inclusive train upper bound in Angstrom^-1.

    Returns
    -------
    row
        Scan metrics for this cut.
    train
        Fitted train objective (model carries the truncated MLE).
    """
    film = apply_polish_seed(obj_map, summary)
    train = build_logY_objective(obj_map, q_min=Q_MIN, q_max=q_cut)
    history = polish_cut(train)
    sync_obj_map_from_objective(obj_map, train)
    try:
        hold = build_logY_objective(obj_map, q_min=Q_MIN, q_gt=q_cut, model=train.model)
        hold_m = residual_metrics(hold)
    except ValueError:
        hold_m = {"n": 0.0, "chi2": float("nan"), "rms_logR": float("nan")}
    return _row_from_fit(
        film=film, train=train, history=history, hold_m=hold_m, q_cut=q_cut
    ), train


def full_q_curves(
    obj_map: dict[float, Any],
    train: Objective,
    *,
    energy: float = ANCHOR,
) -> dict[str, np.ndarray]:
    """Evaluate the truncated-fit model on the full measured ``q`` for ``energy``.

    Parameters
    ----------
    obj_map
        Per-energy objectives used to assemble the full-``q`` dataset.
    train
        Fitted train objective whose ``model`` supplies the prediction.
    energy
        Photon energy (eV) whose s/p curves are returned.

    Returns
    -------
    dict
        Arrays ``q``, ``logR``, ``logR_err``, ``logR_fit``, ``pol`` for the
        requested energy over the full measured ``q`` range.
    """
    full = build_logY_objective(obj_map, q_min=Q_MIN, model=train.model)
    y_fit = np.asarray(full.generative(), dtype=float)
    ds = full._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e = np.asarray(ds.energy, dtype=float)
    m = np.isclose(e, float(energy))
    yd, ed = TRANSFORM(q[m], r[m], r_err[m])
    yf, _ = TRANSFORM(q[m], y_fit[m])
    return {
        "q": q[m],
        "logR": np.asarray(yd, dtype=float),
        "logR_err": np.asarray(ed, dtype=float),
        "logR_fit": np.asarray(yf, dtype=float),
        "pol": pol[m],
    }


def run_cut(
    obj_map: dict[float, Any],
    summary: dict[str, Any],
    q_cut: float,
) -> dict[str, Any]:
    """Warm-start from polish MLE, re-polish on ``q <= q_cut``, score holdout."""
    row, _train = fit_cut_objective(obj_map, summary, q_cut)
    return row


def load_polish_summary() -> dict[str, Any]:
    path = POLISH_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run _bookended_multienergy_logY_polish.py first"
        )
    return json.loads(path.read_text())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="q-truncation diffraction-limit soft-curve scan"
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=21,
        help="number of inclusive q_cut points from q_hi down to q_lo",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run only q_hi, midpoint, and q_lo (3 points)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="output directory under fitting_results",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_polish_summary()
    log(f"seed polish: {POLISH_DIR}")
    log(f"protocol: {summary.get('protocol')}")

    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    q_hi, q_lo, q_by_e = scan_bounds(obj_map)
    d_full = float(np.pi / q_hi)
    log(f"q_by_energy: { {f'{e:g}': round(q, 6) for e, q in q_by_e.items()} }")
    log(f"q_hi={q_hi:.6f}  q_lo={q_lo:.6f}  d_full=pi/q_hi={d_full:.4f} A")

    if args.smoke:
        cuts = np.array([q_hi, 0.5 * (q_hi + q_lo), q_lo], dtype=float)
        log("smoke grid: 3 cuts")
    else:
        cuts = q_cut_grid(q_hi, q_lo, int(args.n_steps))
        log(f"scan grid: {len(cuts)} cuts")

    rows: list[dict[str, Any]] = []
    for i, q_cut in enumerate(cuts):
        log(f"[{i + 1}/{len(cuts)}] q_cut={q_cut:.6f}  d=pi/q={np.pi / q_cut:.4f} A")
        row = run_cut(obj_map, summary, float(q_cut))
        rows.append(row)
        log(
            f"  train n={row['n_train']} redchi={row['redchi_train']:.4f}  "
            f"hold n={row['n_holdout']} rms={row['rms_logR_holdout']:.4f}  "
            f"tau_vac={row['tau_vac']:.3f}  success={row['success']}  "
            f"wall={row['runtime_s']:.1f}s"
        )
        ckpt = out_dir / f"cut_{i:03d}_q{q_cut:.6f}.json"
        ckpt.write_text(json.dumps(row, indent=2))

    df = pd.DataFrame(rows)
    parquet_path = out_dir / "scan.parquet"
    csv_path = out_dir / "scan.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    meta = {
        "energies": list(ENERGIES),
        "anchor": float(ANCHOR),
        "q_hi": q_hi,
        "q_lo": q_lo,
        "d_full": d_full,
        "q_by_energy": {f"{e:g}": float(q) for e, q in q_by_e.items()},
        "n_steps": len(cuts),
        "smoke": bool(args.smoke),
        "transform": "logY",
        "diffraction_length": "pi / q_cut",
        "free_params": list(SURFACE_PARAMS) + list(cv.FREE_INSTRUMENT),
        "seed_polish": str(POLISH_DIR.resolve()),
        "seed_protocol": summary.get("protocol"),
        "protocol": "q_truncation_diffraction_soft_curve_repollish",
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    with (out_dir / "scan.pkl").open("wb") as fh:
        pickle.dump({"rows": rows, "summary": meta, "dataframe": df}, fh)

    log(f"wrote {parquet_path}")
    log(f"wrote {csv_path}")
    log(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

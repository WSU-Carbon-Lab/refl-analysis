"""Multi-seed DE q-truncation scan for reflectivity diffraction-limit variance.

Same cut grid and free set as ``_bookended_q_truncation_diffraction.py``, but
each ``q_cut`` is fit with differential evolution under several RNG seeds
(optionally followed by L-BFGS-B polish) so holdout metrics and film parameters
carry empirical variance.

Writes replicate and aggregated tables under
``@models/fitting_results/bookended_q_truncation_diffraction_de/``.
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
from refloxide.objective import gpu_batch
from refnx.analysis import CurveFitter

_HERE = Path(__file__).resolve().parent
_BASE = _HERE / "_bookended_q_truncation_diffraction.py"
_spec = importlib.util.spec_from_file_location(
    "_bookended_q_truncation_diffraction", _BASE
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load base truncation module from {_BASE}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

cv = base.cv
ENERGIES = base.ENERGIES
ANCHOR = base.ANCHOR
SURFACE_PARAMS = base.SURFACE_PARAMS
FILM_NAMES = base.FILM_NAMES
POLISH_DIR = base.POLISH_DIR
OUT_DIR = cv.models_fitting_results / "bookended_q_truncation_diffraction_de"

DE_MUTATION: tuple[float, float] = (0.5, 1.0)
DE_KW: dict[str, Any] = {
    "popsize": 12,
    "maxiter": 60,
    "mutation": DE_MUTATION,
    "recombination": 0.7,
    "polish": False,
    "verbose": False,
}
METRIC_COLS = (
    "chi2_train",
    "redchi_train",
    "rms_logR_train",
    "chi2_holdout",
    "chi2_per_n_holdout",
    "redchi_holdout",
    "rms_logR_holdout",
    "nll",
    "tau_vac",
    "alpha_vac",
    "surface_roughness",
)


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def de_cut(objective: Any, *, seed: int, de_kw: dict[str, Any]) -> dict[str, Any]:
    """Run GPU differential evolution on a truncated ``logY`` objective."""
    kw = {**de_kw, "seed": int(seed)}
    objective.model.device = "gpu"
    nll0 = float(objective.nll())
    t0 = time.perf_counter()
    result = CurveFitter(objective).fit(
        "differential_evolution",
        workers=gpu_batch(objective),
        **kw,
    )
    nll1 = float(objective.nll())
    return {
        "stage": "de_gpu",
        "nll_start": nll0,
        "nll": nll1,
        "delta_nll": nll0 - nll1,
        "nfev": int(getattr(result, "nfev", -1)),
        "nit": int(getattr(result, "nit", -1)),
        "wall_s": float(time.perf_counter() - t0),
        "message": str(getattr(result, "message", "")),
        "success": bool(getattr(result, "success", False)),
        "seed": int(seed),
        "popsize": int(kw["popsize"]),
        "maxiter": int(kw["maxiter"]),
    }


def fit_cut_de(
    obj_map: dict[float, Any],
    summary: dict[str, Any],
    q_cut: float,
    *,
    opts: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """DE (+ optional L-BFGS) on ``q <= q_cut``; return metrics and train objective.

    Parameters
    ----------
    obj_map
        Seeded per-energy objectives.
    summary
        Locked multi-energy ``logY`` polish summary (warm-start locks / free set).
    q_cut
        Inclusive train upper bound in Angstrom^-1.
    opts
        Must include ``seed`` (int), ``de_kw`` (dict for CurveFitter DE), and
        ``lbfgs_polish`` (bool).

    Returns
    -------
    row
        Replicate metrics including ``seed`` and stage wall times.
    train
        Fitted train objective.
    """
    seed = int(opts["seed"])
    de_kw = dict(opts["de_kw"])
    lbfgs_polish = bool(opts["lbfgs_polish"])
    film = base.apply_polish_seed(obj_map, summary)
    train = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=q_cut)
    de_hist = de_cut(train, seed=seed, de_kw=de_kw)
    wall = float(de_hist["wall_s"])
    nfev = int(de_hist["nfev"])
    success = bool(de_hist["success"])
    message = str(de_hist["message"])

    if lbfgs_polish:
        lb_hist = base.polish_cut(train)
        wall += float(lb_hist["wall_s"])
        nfev += int(lb_hist["nfev"])
        success = bool(lb_hist["success"])
        message = f"de:{message} | lbfgs:{lb_hist['message']}"
        history = {
            "nll": float(lb_hist["nll"]),
            "nll_de": float(de_hist["nll"]),
            "wall_s": wall,
            "nfev": nfev,
            "success": success,
            "message": message,
        }
    else:
        history = {
            "nll": float(de_hist["nll"]),
            "nll_de": float(de_hist["nll"]),
            "wall_s": wall,
            "nfev": nfev,
            "success": bool(np.isfinite(de_hist["nll"])),
            "message": message,
        }

    base.sync_obj_map_from_objective(obj_map, train)
    try:
        hold = base.build_logY_objective(
            obj_map, q_min=base.Q_MIN, q_gt=q_cut, model=train.model
        )
        hold_m = base.residual_metrics(hold)
    except ValueError:
        hold_m = {"n": 0.0, "chi2": float("nan"), "rms_logR": float("nan")}

    row = base._row_from_fit(
        film=film, train=train, history=history, hold_m=hold_m, q_cut=q_cut
    )
    row["seed"] = int(seed)
    row["nll_de"] = float(history["nll_de"])
    row["de_nit"] = int(de_hist["nit"])
    row["lbfgs_polish"] = bool(lbfgs_polish)
    return row, train


def aggregate_replicates(rep_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed rows to mean/std summaries keyed by ``q_cut``."""
    rows: list[dict[str, Any]] = []
    for q_cut_key, g in rep_df.groupby("q_cut", sort=True):
        g = g.sort_values("seed")
        q_cut_f = float(np.asarray(q_cut_key, dtype=float))
        out: dict[str, Any] = {
            "q_cut": q_cut_f,
            "d_pi_over_q": float(np.pi / q_cut_f),
            "n_train": int(g["n_train"].iloc[0]),
            "n_holdout": int(g["n_holdout"].iloc[0]),
            "nvary": int(g["nvary"].iloc[0]),
            "n_seeds": len(g),
            "n_success": int(g["success"].sum()),
            "runtime_s_mean": float(g["runtime_s"].mean()),
            "runtime_s_sum": float(g["runtime_s"].sum()),
            "best_seed": int(g.loc[g["nll"].idxmin(), "seed"]),
            "best_nll": float(g["nll"].min()),
        }
        for col in METRIC_COLS:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            if np.all(~np.isfinite(vals)):
                out[f"{col}_mean"] = float("nan")
                out[f"{col}_std"] = float("nan")
                out[f"{col}_min"] = float("nan")
                out[f"{col}_max"] = float("nan")
                continue
            out[f"{col}_mean"] = float(np.nanmean(vals))
            n_ok = int(np.sum(np.isfinite(vals)))
            out[f"{col}_std"] = float(np.nanstd(vals, ddof=1)) if n_ok > 1 else 0.0
            out[f"{col}_min"] = float(np.nanmin(vals))
            out[f"{col}_max"] = float(np.nanmax(vals))
        rows.append(out)
    return (
        pd.DataFrame(rows).sort_values("q_cut", ascending=False).reset_index(drop=True)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="multi-seed DE q-truncation reflectivity diffraction-limit scan"
    )
    p.add_argument("--n-steps", type=int, default=21)
    p.add_argument("--n-seeds", type=int, default=5, help="DE RNG seeds per q_cut")
    p.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="first DE seed; uses seed_start .. seed_start+n_seeds-1",
    )
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument(
        "--lbfgs",
        action="store_true",
        help="run CPU L-BFGS-B after each DE (collapses seed variance; off by default)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="3 q_cuts and 2 seeds (fast variance probe)",
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = 2 if args.smoke else int(args.n_seeds)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + n_seeds))
    de_kw = {
        **DE_KW,
        "popsize": int(args.popsize),
        "maxiter": int(args.maxiter) if not args.smoke else min(int(args.maxiter), 25),
    }
    lbfgs_polish = bool(args.lbfgs)

    summary = base.load_polish_summary()
    log(f"seed polish: {POLISH_DIR}")
    log(
        f"DE variance scan: n_seeds={n_seeds} seeds={seeds}  "
        f"popsize={de_kw['popsize']} maxiter={de_kw['maxiter']}  "
        f"lbfgs_polish={lbfgs_polish}"
    )

    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    q_hi, q_lo, q_by_e = base.scan_bounds(obj_map)
    d_full = float(np.pi / q_hi)
    log(f"q_hi={q_hi:.6f}  q_lo={q_lo:.6f}  d_full=pi/q_hi={d_full:.4f} A")

    if args.smoke:
        cuts = np.array([q_hi, 0.5 * (q_hi + q_lo), q_lo], dtype=float)
        log("smoke grid: 3 cuts x 2 seeds")
    else:
        cuts = base.q_cut_grid(q_hi, q_lo, int(args.n_steps))
        log(f"scan grid: {len(cuts)} cuts x {n_seeds} seeds")

    replicates: list[dict[str, Any]] = []
    for i, q_cut in enumerate(cuts):
        log(f"[{i + 1}/{len(cuts)}] q_cut={q_cut:.6f}  d=pi/q={np.pi / q_cut:.4f} A")
        for seed in seeds:
            log(f"  seed={seed} ...")
            row, _train = fit_cut_de(
                obj_map,
                summary,
                float(q_cut),
                opts={
                    "seed": seed,
                    "de_kw": de_kw,
                    "lbfgs_polish": lbfgs_polish,
                },
            )
            row["cut_index"] = int(i)
            replicates.append(row)
            log(
                f"    train redchi={row['redchi_train']:.4f}  "
                f"hold rms={row['rms_logR_holdout']:.4f}  "
                f"tau_vac={row['tau_vac']:.3f}  "
                f"nll={row['nll']:.4f}  wall={row['runtime_s']:.1f}s  "
                f"ok={row['success']}"
            )
            ckpt = out_dir / f"cut_{i:03d}_q{q_cut:.6f}_seed{seed:02d}.json"
            ckpt.write_text(json.dumps(row, indent=2))

        cut_reps = [r for r in replicates if r["cut_index"] == i]
        hold_rms = [r["rms_logR_holdout"] for r in cut_reps]
        finite = [v for v in hold_rms if np.isfinite(v)]
        if finite:
            mu = float(np.mean(finite))
            sig = float(np.std(finite, ddof=1))
            log(f"  hold rms mean+/-std = {mu:.4f}+/-{sig:.4f} (n={len(finite)})")

    rep_df = pd.DataFrame(replicates)
    agg_df = aggregate_replicates(rep_df)
    rep_df.to_parquet(out_dir / "replicates.parquet", index=False)
    rep_df.to_csv(out_dir / "replicates.csv", index=False)
    agg_df.to_parquet(out_dir / "scan.parquet", index=False)
    agg_df.to_csv(out_dir / "scan.csv", index=False)

    meta = {
        "energies": list(ENERGIES),
        "anchor": float(ANCHOR),
        "q_hi": q_hi,
        "q_lo": q_lo,
        "d_full": d_full,
        "q_by_energy": {f"{e:g}": float(q) for e, q in q_by_e.items()},
        "n_steps": len(cuts),
        "n_seeds": n_seeds,
        "seeds": seeds,
        "smoke": bool(args.smoke),
        "transform": "logY",
        "diffraction_length": "pi / q_cut",
        "optimizer": "differential_evolution",
        "lbfgs_polish": lbfgs_polish,
        "de_kw": {k: de_kw[k] for k in ("popsize", "maxiter", "recombination")},
        "mutation": [DE_MUTATION[0], DE_MUTATION[1]],
        "free_params": list(SURFACE_PARAMS) + list(cv.FREE_INSTRUMENT),
        "seed_polish": str(POLISH_DIR.resolve()),
        "seed_protocol": summary.get("protocol"),
        "protocol": "q_truncation_diffraction_de_multiseed",
        "base_lbfgs_scan": str(
            (cv.models_fitting_results / "bookended_q_truncation_diffraction").resolve()
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    with (out_dir / "scan.pkl").open("wb") as fh:
        pickle.dump(
            {
                "replicates": replicates,
                "rep_df": rep_df,
                "agg_df": agg_df,
                "summary": meta,
            },
            fh,
        )

    log(f"wrote {out_dir / 'replicates.parquet'}")
    log(f"wrote {out_dir / 'scan.parquet'}")
    log(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

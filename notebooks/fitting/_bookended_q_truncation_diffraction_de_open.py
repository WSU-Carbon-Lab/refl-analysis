"""Open 12-parameter multi-seed DE q-truncation diffraction-limit scan.

Same cut grid and DE protocol as ``_bookended_q_truncation_diffraction_de.py``,
but opens the 10 film bookend/bulk parameters plus oxide thick/rough (12
structural free parameters). SiO2 and Si densities and ``energy_offset`` stay
pinned. Per-energy instruments remain free.

Writes under ``@models/fitting_results/bookended_q_truncation_diffraction_de_open/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DE = _HERE / "_bookended_q_truncation_diffraction_de.py"
_OPEN = _HERE / "_bookended_multienergy_logY_polish_open.py"

_de_spec = importlib.util.spec_from_file_location(
    "_bookended_q_truncation_diffraction_de", _DE
)
if _de_spec is None or _de_spec.loader is None:
    raise ImportError(f"cannot load DE truncation module from {_DE}")
de = importlib.util.module_from_spec(_de_spec)
sys.modules[_de_spec.name] = de
_de_spec.loader.exec_module(de)

_open_spec = importlib.util.spec_from_file_location(
    "_bookended_multienergy_logY_polish_open", _OPEN
)
if _open_spec is None or _open_spec.loader is None:
    raise ImportError(f"cannot load open polish module from {_OPEN}")
open_mod = importlib.util.module_from_spec(_open_spec)
sys.modules[_open_spec.name] = open_mod
_open_spec.loader.exec_module(open_mod)

base = de.base
cv = de.cv
ENERGIES = de.ENERGIES
ANCHOR = de.ANCHOR
FILM_NAMES = de.FILM_NAMES
POLISH_DIR = de.POLISH_DIR
DE_MUTATION = de.DE_MUTATION
DE_KW = de.DE_KW
RHO_SIO2 = base.RHO_SIO2
RHO_SI = base.RHO_SI

OUT_DIR = cv.models_fitting_results / "bookended_q_truncation_diffraction_de_open"
LOCKED_DE_DIR = cv.models_fitting_results / "bookended_q_truncation_diffraction_de"
LOCKED_ENERGY_OFFSET = float(open_mod.LOCKED_ENERGY_OFFSET)
FILM_FREE = open_mod.FILM_FREE
STRUCTURAL_FREE = (*FILM_FREE, "oxide.thick", "oxide.rough")
METRIC_COLS = (
    "chi2_train",
    "redchi_train",
    "rms_logR_train",
    "chi2_holdout",
    "chi2_per_n_holdout",
    "redchi_holdout",
    "rms_logR_holdout",
    "nll",
    *FILM_NAMES,
    "oxide_thick",
    "oxide_rough",
)


def log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def apply_open_seed(obj_map: dict[float, Any], summary: dict[str, Any]) -> Any:
    """Warm-start from locked polish MLE and open the 12 structural params.

    Opens ``FILM_FREE`` (10) plus oxide thick/rough. Pins oxide and Si densities
    and ``energy_offset``. Opens per-energy instruments.

    Parameters
    ----------
    obj_map
        Seeded per-energy objectives sharing film geometry.
    summary
        Locked multi-energy ``logY`` polish summary used as the warm start.

    Returns
    -------
    film
        Bookended film component with the open free set applied.
    """
    _vacuum, film, oxide, substrate = obj_map[ANCHOR].model.structure.components
    film_mle = summary["film"]
    oxide_mle = summary["oxide"]
    oxide_rho = float(oxide_mle.get("density", RHO_SIO2))
    si_rho = float(
        summary.get("substrate", {}).get("density", summary.get("rho_si", RHO_SI))
    )

    for _energy, obj in obj_map.items():
        cv.freeze_all_parameters(obj)

    bounds = open_mod.film_bounds(film_mle)
    for name in FILM_FREE:
        cv.safely_setp_param(
            getattr(film, name),
            value=float(film_mle[name]),
            vary=True,
            bounds=bounds[name],
        )

    ox_thick = float(oxide_mle["thick"])
    ox_rough = float(oxide_mle["rough"])
    nc0 = open_mod.nc_limit(ox_rough)
    if ox_thick <= nc0 + 1e-3:
        ox_thick = nc0 + 0.75
        log(f"oxide NC nudge thick -> {ox_thick:.4f} (was on floor nc={nc0:.4f})")
    nc_scale = np.sqrt(2.0 * np.pi) / 2.0
    rough_hi = min(12.0, max(ox_rough + 1.0, ox_thick / nc_scale * 0.98))
    ox_rough = min(ox_rough, rough_hi - 1e-3)
    thick_lo = max(2.0, open_mod.nc_limit(min(ox_rough, rough_hi)) + 0.02)
    oxide.thick.setp(value=ox_thick, vary=True, bounds=(thick_lo, 25.0))
    oxide.rough.setp(value=ox_rough, vary=True, bounds=(0.0, float(rough_hi)))
    oxide.sld.density.setp(value=oxide_rho, vary=False)
    substrate.sld.density.setp(value=si_rho, vary=False)

    instruments = {float(k): v for k, v in summary["instruments"].items()}
    ib = open_mod.open_instrument_bounds()
    for energy, obj in obj_map.items():
        model = obj.model
        vals = instruments[float(energy)]
        for name in cv.FREE_INSTRUMENT:
            cv.safely_setp_param(
                getattr(model, name),
                value=float(vals[name]),
                vary=True,
                bounds=ib[name],
            )
        cv.safely_setp_param(
            model.bkg,
            value=float(vals.get("bkg", 0.0)),
            vary=False,
        )
        cv.safely_setp_param(
            model.energy_offset, value=LOCKED_ENERGY_OFFSET, vary=False
        )
    cv.tighten_theta_offsets(obj_map, ENERGIES)
    return film


def fit_cut_de_open(
    obj_map: dict[float, Any],
    summary: dict[str, Any],
    q_cut: float,
    *,
    opts: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """DE on ``q <= q_cut`` with the open 12-parameter structural free set.

    Parameters
    ----------
    obj_map
        Seeded per-energy objectives.
    summary
        Locked multi-energy ``logY`` polish summary (warm start).
    q_cut
        Inclusive train upper bound in Angstrom^-1.
    opts
        Must include ``seed``, ``de_kw``, and ``lbfgs_polish``.

    Returns
    -------
    row
        Replicate metrics including film and oxide thick/rough.
    train
        Fitted train objective.
    """
    seed = int(opts["seed"])
    de_kw = dict(opts["de_kw"])
    lbfgs_polish = bool(opts["lbfgs_polish"])
    film = apply_open_seed(obj_map, summary)
    _vacuum, _film, oxide, _substrate = obj_map[ANCHOR].model.structure.components
    train = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=q_cut)
    de_hist = de.de_cut(train, seed=seed, de_kw=de_kw)
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
    row["oxide_thick"] = float(oxide.thick.value)
    row["oxide_rough"] = float(oxide.rough.value)
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
        description="open 12-param DE q-truncation diffraction-limit scan"
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


def _write_outputs(
    out_dir: Path,
    *,
    replicates: list[dict[str, Any]],
    meta: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_df = pd.DataFrame(replicates)
    agg_df = aggregate_replicates(rep_df)
    rep_df.to_parquet(out_dir / "replicates.parquet", index=False)
    rep_df.to_csv(out_dir / "replicates.csv", index=False)
    agg_df.to_parquet(out_dir / "scan.parquet", index=False)
    agg_df.to_csv(out_dir / "scan.csv", index=False)
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
    return rep_df, agg_df


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
        f"OPEN DE variance scan: structural={list(STRUCTURAL_FREE)}  "
        f"n_seeds={n_seeds} seeds={seeds}  "
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

    apply_open_seed(obj_map, summary)
    probe = base.build_logY_objective(obj_map, q_min=base.Q_MIN, q_max=float(cuts[0]))
    nvary0 = len(probe.varying_parameters())
    log(f"nvary probe={nvary0} (expect 32 = 12 structural + 20 instrument)")
    del probe

    replicates: list[dict[str, Any]] = []
    for i, q_cut in enumerate(cuts):
        log(f"[{i + 1}/{len(cuts)}] q_cut={q_cut:.6f}  d=pi/q={np.pi / q_cut:.4f} A")
        for seed in seeds:
            log(f"  seed={seed} ...")
            row, _train = fit_cut_de_open(
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
                f"oxide={row['oxide_thick']:.3f}/{row['oxide_rough']:.3f}  "
                f"nvary={row['nvary']}  nll={row['nll']:.4f}  "
                f"wall={row['runtime_s']:.1f}s  ok={row['success']}"
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
        "structural_free": list(STRUCTURAL_FREE),
        "n_structural_free": len(STRUCTURAL_FREE),
        "free_instrument": list(cv.FREE_INSTRUMENT),
        "locked": [
            "oxide.sld.density",
            "substrate.sld.density",
            "energy_offset",
            "bkg",
        ],
        "energy_offset_locked": LOCKED_ENERGY_OFFSET,
        "seed_polish": str(POLISH_DIR.resolve()),
        "seed_protocol": summary.get("protocol"),
        "protocol": "q_truncation_diffraction_de_open_12struct",
        "locked_de_scan": str(LOCKED_DE_DIR.resolve()),
        "nvary_probe": int(nvary0),
    }
    _write_outputs(out_dir, replicates=replicates, meta=meta)


if __name__ == "__main__":
    main()

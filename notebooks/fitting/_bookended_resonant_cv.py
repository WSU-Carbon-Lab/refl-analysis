"""Progressive resonant CV for the bookended graded ZnPc fit.

Stage 1 (283.7 eV only) keeps the published geometry MLE (no DE). Later stages
warm-start from the previous geometry, DE-refit shared film parameters on
``E_RES[:k]``, and L-BFGS-B-polish train scale/theta.

Holdout energies always get their own L-BFGS-B polish of scale/theta with
geometry frozen: laboratory corrections cannot be extrapolated from the train
set, so validation asks whether the shared geometry can describe left-out
energies once per-energy instruments are free. Calibrated terms
(``energy_offset``, ``dq``, ``q_offset``, ``bkg``) stay fixed.

Writes incremental checkpoints and a final bundle under
``@models/fitting_results/bookended_resonant_cv/`` for the notebook to load.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from refloxide.pxr.plugin.dft_fit import (
    MIN_INCIDENT_THETA_DEG,
    THETA_OFFSET_MARGIN_DEG,
    min_data_grazing_angle_deg,
)
from scipy import stats
from utils import (
    models_fitting_results,
    objectives_retarget_energies,
    read_fit,
    read_xrr,
)
from utils.graded_objective import (
    REFLECT_MODEL_INSTRUMENTATION,
    extract_graded_fit_context,
)
from utils.helpers.fitting_helper import ObjectiveInterface
from utils.models import (
    AnisotropyObjective,
    CurveFitter,
    GlobalObjective,
    Transform,
    configure_refloxide_fitting,
)
from utils.refloxide_view import load_refloxide_fit
from utils.slab_builders import safely_setp_param

configure_refloxide_fitting()

ENERGY_ANCHOR = 283.7
E_RES = (283.7, 284.2, 285.1, 287.0, 289.0)
ANCHOR = float(ENERGY_ANCHOR)
Q_MIN = 1e-4
TRANSFORM = Transform("YX4")

GEOMETRY_PARAMS = (
    "total_thick",
    "surface_roughness",
    "density_si",
    "density_vac",
    "tau_si",
    "tau_vac",
    "alpha_bulk",
    "alpha_si",
    "alpha_vac",
)
GEOMETRY_BOUNDS = {
    "total_thick": (160.0, 210.0),
    "surface_roughness": (0.0, 10.0),
    "density_si": (0.5, 2.5),
    "density_vac": (0.5, 2.5),
    "tau_si": (0.5, 25.0),
    "tau_vac": (0.0, 25.0),
    "alpha_bulk": (0.0, np.pi / 2),
    "alpha_si": (0.0, np.pi / 4),
    "alpha_vac": (0.0, np.pi / 2),
}

CALIBRATED_INSTRUMENT = ("energy_offset", "dq", "q_offset", "bkg")
FREE_INSTRUMENT = ("scale_s", "scale_p", "theta_offset_s", "theta_offset_p")
INSTRUMENT_BOUNDS = {
    "scale_s": (0.5, 2.1),
    "scale_p": (0.5, 2.1),
    # Keep theta within the same window used by slab/DFT notebooks; L-BFGS
    # with (-1.5, 1.5) can drive remapped q to grazing and singular Berreman.
    "theta_offset_s": (-0.8, 0.8),
    "theta_offset_p": (-0.8, 0.8),
}

DE_KW = {
    "popsize": 15,
    "maxiter": 60,
    "mutation": 0.7,
    "recombination": 0.7,
    "workers": 1,
    "seed": 1,
}
LBFGS_KW = {"options": {"maxiter": 400}}

OUT_DIR = models_fitting_results / "bookended_resonant_cv"
REFLOXIDE_DFT_PATH = (
    Path(__file__).resolve().parents[2]
    / "@models"
    / "xrr"
    / "znpc"
    / "refloxide"
    / "refit-refloxide.pkl"
)


def log(msg: str) -> None:
    """Flush progress lines so a babysitting process sees them immediately."""
    print(msg, flush=True)  # noqa: T201


@dataclass
class StageResult:
    """Summary metrics and snapshots for one progressive-CV train size."""

    stage: int
    train_energies: tuple[float, ...]
    holdout_energies: tuple[float, ...]
    nparams_geometry: int
    nparams_instrument: int
    nparams_total: int
    ndata_train: int
    chi2_train: float
    redchi_train: float
    logl_train: float
    aic_train: float
    bic_train: float
    ndata_holdout: int
    chi2_holdout: float
    rms_holdout: float
    logl_holdout: float
    geometry: dict[str, float]
    instrument: dict[float, dict[str, float]]
    per_energy: list[dict[str, float]]
    fitted_geometry: bool
    wall_s: float


def dft_instrument_values(dft_corrections, energy: float) -> dict[str, float]:
    """Per-energy DFT-slab corrections plus shared calibrated energy/q terms."""
    channel = dft_corrections.channel_at(float(energy))
    return {
        "scale_s": float(channel.scale_s.value),
        "scale_p": float(channel.scale_p.value),
        "theta_offset_s": float(channel.theta_offset_s.value),
        "theta_offset_p": float(channel.theta_offset_p.value),
        "bkg": float(channel.bkg.value),
        "dq": float(dft_corrections.dq.value or 0.0),
        "q_offset": float(dft_corrections.q_offset.value or 0.0),
        "energy_offset": float(dft_corrections.energy_offset.value or 0.0),
    }


def pin_calibrated_instrument(obj, values: dict[str, float] | None = None) -> None:
    """Freeze calibrated instrument terms (and film energy_offset) at input."""
    model = obj.model
    for name in CALIBRATED_INSTRUMENT:
        kwargs: dict = {"vary": False}
        if values is not None and name in values:
            kwargs["value"] = values[name]
        safely_setp_param(getattr(model, name), **kwargs)

    film = extract_graded_fit_context(obj).film
    e_off = (
        float(model.energy_offset.value or 0.0)
        if values is None
        else float(values.get("energy_offset", model.energy_offset.value or 0.0))
    )
    if film.energy_offset.constraint is None:
        safely_setp_param(film.energy_offset, value=e_off, vary=False)


def apply_fixed_instrumentation(model, values: dict[str, float]) -> None:
    """Write laboratory corrections onto ``model`` and freeze them."""
    for name in REFLECT_MODEL_INSTRUMENTATION:
        safely_setp_param(getattr(model, name), value=values[name], vary=False)


def mask_q_min(dataset, q_min: float = Q_MIN):
    """Mask q<=q_min on an XrayReflectDataset (required for finite YX4 errors)."""
    ds = copy.deepcopy(dataset)
    q_full = np.asarray(ds.unmasked_data[0], dtype=float)
    ds.mask = q_full > float(q_min)
    return ds


def film_of(obj: AnisotropyObjective):
    return extract_graded_fit_context(obj).film


def freeze_all_parameters(objective) -> None:
    for p in objective.parameters.flattened():
        if p.constraint is not None:
            continue
        p.vary = False


def open_geometry(obj_map: dict[float, AnisotropyObjective], *, vary: bool) -> None:
    """Open/close shared geometry on the anchor film only (others are constrained)."""
    film = film_of(obj_map[ANCHOR])
    for name in GEOMETRY_PARAMS:
        kwargs: dict = {"vary": vary}
        if vary:
            kwargs["bounds"] = GEOMETRY_BOUNDS[name]
        safely_setp_param(getattr(film, name), **kwargs)


def open_instrument(model, *, vary: bool) -> None:
    for name in FREE_INSTRUMENT:
        kwargs: dict = {"vary": vary}
        if vary:
            kwargs["bounds"] = INSTRUMENT_BOUNDS[name]
        safely_setp_param(getattr(model, name), **kwargs)


def tighten_theta_offsets(
    obj_map: dict[float, AnisotropyObjective],
    energies: tuple[float, ...] | list[float],
) -> None:
    """Raise ``theta_offset`` lower bounds so remapped q stays above grazing.

    Uses the same data-derived floor as
    :func:`refloxide.pxr.plugin.dft_fit.tighten_theta_offset_bounds_from_terms`
    but for pyref ``AnisotropyObjective`` models.
    """
    for energy in energies:
        obj = obj_map[float(energy)]
        model = obj.model
        for pol, field in (("s", "theta_offset_s"), ("p", "theta_offset_p")):
            pol_data = getattr(obj.data, pol)
            q = np.asarray(pol_data.x, dtype=float)
            if q.size == 0:
                continue
            grazing = min_data_grazing_angle_deg(q, float(energy))
            safe_lb = MIN_INCIDENT_THETA_DEG + THETA_OFFSET_MARGIN_DEG - grazing
            lb_old, ub = INSTRUMENT_BOUNDS[field]
            new_lb = max(float(lb_old), float(safe_lb))
            param = getattr(model, field)
            kwargs: dict = {"bounds": (new_lb, ub), "vary": bool(param.vary)}
            value = float(param.value or 0.0)
            if value < new_lb:
                kwargs["value"] = new_lb
            safely_setp_param(param, **kwargs)
            log(
                f"  theta floor {float(energy):g} eV {field}: "
                f"lb={new_lb:.4f} (data_min={grazing:.4f} deg, value={value:.4f})"
            )


def link_shared_geometry(obj_map: dict[float, AnisotropyObjective]) -> None:
    """Tie every non-anchor film geometry parameter to the anchor Parameter objects."""
    film_a = film_of(obj_map[ANCHOR])
    for energy, obj in obj_map.items():
        if np.isclose(float(energy), ANCHOR):
            continue
        film_b = film_of(obj)
        for name in GEOMETRY_PARAMS:
            getattr(film_b, name).constraint = getattr(film_a, name)


def geometry_snapshot(obj_map: dict[float, AnisotropyObjective]) -> dict[str, float]:
    film = film_of(obj_map[ANCHOR])
    return {name: float(getattr(film, name).value or 0.0) for name in GEOMETRY_PARAMS}


def apply_geometry_snapshot(
    obj_map: dict[float, AnisotropyObjective], snap: dict[str, float]
) -> None:
    film = film_of(obj_map[ANCHOR])
    for name, value in snap.items():
        safely_setp_param(getattr(film, name), value=value)


def instrument_snapshot(
    obj_map: dict[float, AnisotropyObjective],
) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for energy, obj in obj_map.items():
        out[float(energy)] = {
            name: float(getattr(obj.model, name).value or 0.0)
            for name in FREE_INSTRUMENT
        }
    return out


def apply_instrument_snapshot(
    obj_map: dict[float, AnisotropyObjective],
    snap: dict[float, dict[str, float]],
    energies: tuple[float, ...] | list[float],
) -> None:
    for energy in energies:
        model = obj_map[float(energy)].model
        for name, value in snap[float(energy)].items():
            safely_setp_param(getattr(model, name), value=value)


def build_yx4_objectives(
    source_objectives: dict[float, AnisotropyObjective],
) -> dict[float, AnisotropyObjective]:
    """Deep-copy models onto YX4 objectives, mask q, pin calibration, link geometry."""
    out: dict[float, AnisotropyObjective] = {}
    for energy, src in source_objectives.items():
        model = copy.deepcopy(src.model)
        data = mask_q_min(src.data, Q_MIN)
        obj = AnisotropyObjective(
            model,
            data,
            transform=TRANSFORM,
            logp_anisotropy_weight=float(src.logp_anisotropy_weight),
        )
        freeze_all_parameters(obj)
        pin_calibrated_instrument(obj)
        out[float(energy)] = obj
    link_shared_geometry(out)
    return out


def assert_energy_offset_fixed(obj_map: dict[float, AnisotropyObjective]) -> None:
    """Fail fast if any energy_offset slipped into the free set."""
    for energy, obj in obj_map.items():
        if obj.model.energy_offset.vary:
            raise RuntimeError(f"model.energy_offset vary=True at {energy} eV")
        film = film_of(obj)
        if film.energy_offset.constraint is None and film.energy_offset.vary:
            raise RuntimeError(f"film.energy_offset vary=True at {energy} eV")
        for p in obj.varying_parameters():
            if "energy_offset" in str(p.name):
                raise RuntimeError(
                    f"energy_offset in varying_parameters at {energy} eV: {p.name}"
                )


def train_objective(
    obj_map: dict[float, AnisotropyObjective], train_energies: tuple[float, ...]
):
    members = [obj_map[float(e)] for e in train_energies]
    return members[0] if len(members) == 1 else GlobalObjective(members)


def eval_global(
    obj_map: dict[float, AnisotropyObjective],
    energies: tuple[float, ...] | list[float],
):
    members = [obj_map[float(e)] for e in energies]
    if not members:
        return None
    return members[0] if len(members) == 1 else GlobalObjective(members)


def ndata_of(objective) -> int:
    return ObjectiveInterface(objective).ndata()


def gaussian_aic_bic(chi2: float, nparams: int, ndata: int) -> tuple[float, float]:
    """AIC/BIC for Gaussian least-squares on a fixed evaluation set."""
    n = int(ndata)
    k = int(nparams)
    rss = max(float(chi2), np.finfo(float).tiny)
    base = n * np.log(rss / n)
    return float(base + 2 * k), float(base + k * np.log(n))


def wilks_lrt(logl_full: float, logl_reduced: float, df: int) -> dict[str, float]:
    lr = float(2.0 * (logl_full - logl_reduced))
    df = int(df)
    p = float(stats.chi2.sf(max(lr, 0.0), df)) if df > 0 else float("nan")
    return {"lr_stat": lr, "df": float(df), "p_value": p}


def nested_f_test(
    chi2_reduced: float,
    chi2_full: float,
    ndata: int,
    df_extra: int,
    nparams_full: int,
) -> dict[str, float]:
    df_extra = int(df_extra)
    df_full = int(ndata) - int(nparams_full)
    if df_extra <= 0 or df_full <= 0:
        return {
            "F": float("nan"),
            "df1": float(df_extra),
            "df2": float(df_full),
            "p_value": float("nan"),
        }
    num = (float(chi2_reduced) - float(chi2_full)) / df_extra
    den = float(chi2_full) / df_full
    f_stat = float(num / den) if den > 0 else float("inf")
    p = float(stats.f.sf(max(f_stat, 0.0), df_extra, df_full))
    return {"F": f_stat, "df1": float(df_extra), "df2": float(df_full), "p_value": p}


def per_energy_rows(
    obj_map: dict[float, AnisotropyObjective],
    instrument: dict[float, dict[str, float]],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for energy in E_RES:
        obj = obj_map[float(energy)]
        nd = ndata_of(obj)
        chi2 = float(obj.chisqr())
        rows.append(
            {
                "energy": float(energy),
                "ndata": float(nd),
                "chi2": chi2,
                "rms": float(np.sqrt(chi2 / nd)),
                "logl": float(obj.logl()),
                **instrument[float(energy)],
            }
        )
    return rows


def polish_instruments(
    obj_map: dict[float, AnisotropyObjective],
    energies: tuple[float, ...] | list[float],
    *,
    label: str,
) -> None:
    """L-BFGS-B polish FREE_INSTRUMENT on ``energies`` with geometry frozen.

    Parameters
    ----------
    obj_map
        Linked multi-energy YX4 objectives.
    energies
        Energies whose scale/theta are opened; all other free params stay frozen.
    label
        Tag for progress logs (``train`` / ``holdout``).
    """
    energies = tuple(float(e) for e in energies)
    if not energies:
        return

    for energy in E_RES:
        freeze_all_parameters(obj_map[float(energy)])
        open_instrument(obj_map[float(energy)].model, vary=False)
    open_geometry(obj_map, vary=False)
    for energy in energies:
        open_instrument(obj_map[float(energy)].model, vary=True)
    tighten_theta_offsets(obj_map, energies)
    assert_energy_offset_fixed(obj_map)

    obj = train_objective(obj_map, energies)
    n_inst = len(obj.varying_parameters())
    expected = len(energies) * len(FREE_INSTRUMENT)
    if n_inst != expected:
        raise RuntimeError(
            f"{label}: expected {expected} instrument DOF, got {n_inst}: "
            f"{[p.name for p in obj.varying_parameters()]}"
        )
    log(
        f"L-BFGS-B {label} instrument DOF={n_inst}: "
        f"{[p.name for p in obj.varying_parameters()]}"
    )
    fitter = CurveFitter(obj)
    fitter.fit("L-BFGS-B", target="nll", **LBFGS_KW)
    log(
        f"after {label} instrument polish: "
        f"chi2={obj.chisqr():.4f}  logl={obj.logl():.4f}"
    )


def score_stage(
    obj_map: dict[float, AnisotropyObjective],
    train_energies: tuple[float, ...],
    *,
    fitted_geometry: bool,
    wall_s: float,
) -> StageResult:
    """Freeze free params, snapshot, and score train / holdout / per-energy GOF.

    Call after train (and holdout) instrument polish. Holdout metrics assume
    holdout scale/theta were already optimized with geometry fixed.
    """
    holdout_energies = tuple(e for e in E_RES if e not in train_energies)
    for energy in E_RES:
        freeze_all_parameters(obj_map[float(energy)])
        open_instrument(obj_map[float(energy)].model, vary=False)
    open_geometry(obj_map, vary=False)

    geom = geometry_snapshot(obj_map)
    inst = instrument_snapshot(obj_map)
    per_e = per_energy_rows(obj_map, inst)

    train_eval = train_objective(obj_map, train_energies)
    chi2_train = float(train_eval.chisqr())
    logl_train = float(train_eval.logl())
    nd_train = ndata_of(train_eval)
    npar_g = len(GEOMETRY_PARAMS)
    npar_i = len(train_energies) * len(FREE_INSTRUMENT)
    npar = npar_g + npar_i
    red_train = chi2_train / max(nd_train - npar, 1)
    aic_t, bic_t = gaussian_aic_bic(chi2_train, npar, nd_train)

    if holdout_energies:
        hold_eval = eval_global(obj_map, holdout_energies)
        chi2_hold = float(hold_eval.chisqr())
        logl_hold = float(hold_eval.logl())
        nd_hold = ndata_of(hold_eval)
        npar_hold = len(holdout_energies) * len(FREE_INSTRUMENT)
        rms_hold = float(np.sqrt(chi2_hold / nd_hold))
        red_hold = chi2_hold / max(nd_hold - npar_hold, 1)
        log(
            f"  holdout (instruments polished) redchi={red_hold:.4f}  "
            f"n_inst_hold={npar_hold}"
        )
    else:
        chi2_hold = float("nan")
        logl_hold = float("nan")
        nd_hold = 0
        rms_hold = float("nan")

    return StageResult(
        stage=len(train_energies),
        train_energies=train_energies,
        holdout_energies=holdout_energies,
        nparams_geometry=npar_g,
        nparams_instrument=npar_i,
        nparams_total=npar,
        ndata_train=nd_train,
        chi2_train=chi2_train,
        redchi_train=red_train,
        logl_train=logl_train,
        aic_train=aic_t,
        bic_train=bic_t,
        ndata_holdout=nd_hold,
        chi2_holdout=chi2_hold,
        rms_holdout=rms_hold,
        logl_holdout=logl_hold,
        geometry=geom,
        instrument=inst,
        per_energy=per_e,
        fitted_geometry=fitted_geometry,
        wall_s=wall_s,
    )


def log_stage_summary(result: StageResult) -> None:
    mode = "EVAL (seeded MLE)" if not result.fitted_geometry else "FIT (DE+LBFGS)"
    log(
        f"STAGE {result.stage} [{mode}] train={result.train_energies} "
        f"holdout={result.holdout_energies or '-'}"
    )
    log(
        f"  train chi2={result.chi2_train:.4f}  "
        f"redchi={result.redchi_train:.4f}  "
        f"logl={result.logl_train:.4f}  "
        f"AIC={result.aic_train:.3f}  BIC={result.bic_train:.3f}"
    )
    if result.holdout_energies:
        log(
            f"  holdout chi2={result.chi2_holdout:.4f}  "
            f"RMS={result.rms_holdout:.4g}  logl={result.logl_holdout:.4f}"
        )
    log("  per-energy GOF:")
    for row in result.per_energy:
        role = "train" if row["energy"] in result.train_energies else "hold"
        log(
            f"    {row['energy']:6.1f} eV [{role}]  "
            f"chi2={row['chi2']:.3f}  rms={row['rms']:.4g}  logl={row['logl']:.3f}"
        )
    log(
        "  geometry: "
        + ", ".join(f"{k}={result.geometry[k]:.5g}" for k in GEOMETRY_PARAMS)
    )
    log(f"  wall={result.wall_s:.1f}s")


def save_checkpoint(
    out_dir: Path,
    stage_results: list[StageResult],
    stage_maps: list[dict[float, AnisotropyObjective]],
    *,
    final: bool = False,
) -> None:
    """Write CSV tables plus a pickle bundle after each stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_rows = []
    for r in stage_results:
        cv_rows.append(
            {
                "stage": r.stage,
                "train": ",".join(f"{e:g}" for e in r.train_energies),
                "holdout": ",".join(f"{e:g}" for e in r.holdout_energies) or "-",
                "fitted_geometry": r.fitted_geometry,
                "nparams": r.nparams_total,
                "n_geom": r.nparams_geometry,
                "n_inst": r.nparams_instrument,
                "ndata_train": r.ndata_train,
                "chi2_train": r.chi2_train,
                "redchi_train": r.redchi_train,
                "logl_train": r.logl_train,
                "AIC_train": r.aic_train,
                "BIC_train": r.bic_train,
                "ndata_holdout": r.ndata_holdout,
                "chi2_holdout": r.chi2_holdout,
                "RMS_holdout": r.rms_holdout,
                "logl_holdout": r.logl_holdout,
                "wall_s": r.wall_s,
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(out_dir / "cv_table.csv", index=False)

    per_rows = []
    for r in stage_results:
        for row in r.per_energy:
            per_rows.append({"stage": r.stage, **row})
    pd.DataFrame(per_rows).to_csv(out_dir / "per_energy.csv", index=False)

    geom_rows = []
    for r in stage_results:
        geom_rows.append({"stage": r.stage, **r.geometry})
    pd.DataFrame(geom_rows).to_csv(out_dir / "geometry_by_stage.csv", index=False)

    status = {
        "n_stages_done": len(stage_results),
        "n_stages_total": len(E_RES),
        "last_stage": stage_results[-1].stage if stage_results else 0,
        "complete": final,
        "E_RES": list(E_RES),
    }
    (out_dir / "status.json").write_text(json.dumps(status, indent=2))

    bundle = {
        "E_RES": E_RES,
        "GEOMETRY_PARAMS": GEOMETRY_PARAMS,
        "FREE_INSTRUMENT": FREE_INSTRUMENT,
        "CALIBRATED_INSTRUMENT": CALIBRATED_INSTRUMENT,
        "stage_results": stage_results,
        "cv_table": cv_table,
        "complete": final,
    }
    if final and stage_maps:
        bundle["final_map"] = stage_maps[-1]
        bundle["final_result"] = stage_results[-1]

    with (out_dir / "bundle.pkl").open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log(f"  checkpoint -> {out_dir} (stages={len(stage_results)}, final={final})")


def evaluate_seed_stage(
    objectives: dict[float, AnisotropyObjective],
    seed_geometry: dict[str, float],
    seed_instrument: dict[float, dict[str, float]],
) -> tuple[dict[float, AnisotropyObjective], StageResult]:
    """Stage 1: keep published geometry; polish holdout instruments; score."""
    t0 = time.perf_counter()
    train_energies = (ANCHOR,)
    holdout_energies = tuple(e for e in E_RES if e not in train_energies)
    obj_map = build_yx4_objectives(objectives)
    apply_geometry_snapshot(obj_map, seed_geometry)
    apply_instrument_snapshot(obj_map, seed_instrument, E_RES)
    for _e, _obj in obj_map.items():
        pin_calibrated_instrument(_obj)
    assert_energy_offset_fixed(obj_map)

    for energy in E_RES:
        freeze_all_parameters(obj_map[float(energy)])
        open_instrument(obj_map[float(energy)].model, vary=False)
    open_geometry(obj_map, vary=False)
    n_vary = len(train_objective(obj_map, train_energies).varying_parameters())
    if n_vary != 0:
        raise RuntimeError(
            f"stage-1 evaluate expected 0 free params before holdout polish, "
            f"got {n_vary}"
        )

    polish_instruments(obj_map, holdout_energies, label="holdout")
    result = score_stage(
        obj_map,
        train_energies,
        fitted_geometry=False,
        wall_s=time.perf_counter() - t0,
    )
    return obj_map, result


def fit_stage(
    objectives: dict[float, AnisotropyObjective],
    train_energies: tuple[float, ...],
    warm_geometry: dict[str, float],
    warm_instrument: dict[float, dict[str, float]],
) -> tuple[dict[float, AnisotropyObjective], StageResult]:
    """DE shared geometry on train, polish train then holdout instruments."""
    t0 = time.perf_counter()
    holdout_energies = tuple(e for e in E_RES if e not in train_energies)
    obj_map = build_yx4_objectives(objectives)
    apply_geometry_snapshot(obj_map, warm_geometry)
    apply_instrument_snapshot(obj_map, warm_instrument, E_RES)
    for _e, _obj in obj_map.items():
        pin_calibrated_instrument(_obj)
    assert_energy_offset_fixed(obj_map)

    for energy in E_RES:
        freeze_all_parameters(obj_map[float(energy)])
        open_instrument(obj_map[float(energy)].model, vary=False)
    open_geometry(obj_map, vary=True)

    train_obj = train_objective(obj_map, train_energies)
    x0_geom = np.asarray(
        [float(p.value) for p in train_obj.varying_parameters()], dtype=float
    )
    if len(x0_geom) != len(GEOMETRY_PARAMS):
        raise RuntimeError(
            f"expected {len(GEOMETRY_PARAMS)} geometry DOF, got {len(x0_geom)}: "
            f"{[p.name for p in train_obj.varying_parameters()]}"
        )

    log(f"\n=== FIT stage train={train_energies}  holdout={holdout_energies} ===")
    log(
        f"DE geometry DOF={len(x0_geom)}: "
        f"{[p.name for p in train_obj.varying_parameters()]}"
    )
    log(f"DE_KW={DE_KW}")

    de_state = {"ngen": 0, "t0": time.perf_counter()}

    def _de_callback(_xk, convergence):
        de_state["ngen"] += 1
        if de_state["ngen"] == 1 or de_state["ngen"] % 5 == 0:
            elapsed = time.perf_counter() - de_state["t0"]
            log(
                f"  DE gen={de_state['ngen']}  "
                f"convergence={float(convergence):.4g}  "
                f"elapsed={elapsed:.1f}s"
            )
        return False

    fitter = CurveFitter(train_obj)
    fitter.fit(
        "differential_evolution",
        target="nll",
        x0=x0_geom,
        polish=False,
        updating="deferred",
        callback=_de_callback,
        **DE_KW,
    )
    log(
        f"after DE: train chi2={train_obj.chisqr():.4f}  "
        f"logl={train_obj.logl():.4f}  gens={de_state['ngen']}"
    )

    polish_instruments(obj_map, train_energies, label="train")
    polish_instruments(obj_map, holdout_energies, label="holdout")

    result = score_stage(
        obj_map,
        train_energies,
        fitted_geometry=True,
        wall_s=time.perf_counter() - t0,
    )
    return obj_map, result


def build_nested_table(
    objectives: dict[float, AnisotropyObjective],
    stage_results: list[StageResult],
    stage_maps: list[dict[float, AnisotropyObjective]],
) -> pd.DataFrame:
    """Likelihood-ratio / F tests of consecutive stages on the expanded train set."""
    nested_rows = []
    for k in range(len(stage_results) - 1):
        r_red = stage_results[k]
        r_full = stage_results[k + 1]
        train_full = r_full.train_energies
        new_energy = train_full[-1]
        df_extra = len(FREE_INSTRUMENT)

        map_red = build_yx4_objectives(objectives)
        apply_geometry_snapshot(map_red, r_red.geometry)
        apply_instrument_snapshot(map_red, r_red.instrument, E_RES)
        for e in E_RES:
            freeze_all_parameters(map_red[float(e)])
            open_instrument(map_red[float(e)].model, vary=False)
        open_geometry(map_red, vary=False)
        eval_red = train_objective(map_red, train_full)

        map_full = stage_maps[k + 1]
        eval_full = train_objective(map_full, train_full)

        chi2_red = float(eval_red.chisqr())
        chi2_full = float(eval_full.chisqr())
        logl_red = float(eval_red.logl())
        logl_full = float(eval_full.logl())
        nd = ndata_of(eval_full)
        lrt = wilks_lrt(logl_full, logl_red, df_extra)
        ftest = nested_f_test(chi2_red, chi2_full, nd, df_extra, r_full.nparams_total)

        nested_rows.append(
            {
                "compare": f"{r_red.stage}->{r_full.stage}",
                "new_energy": new_energy,
                "ndata_train_full": nd,
                "chi2_reduced": chi2_red,
                "chi2_full": chi2_full,
                "delta_chi2": chi2_red - chi2_full,
                "LR": lrt["lr_stat"],
                "LR_df": lrt["df"],
                "LR_p": lrt["p_value"],
                "F": ftest["F"],
                "F_df1": ftest["df1"],
                "F_df2": ftest["df2"],
                "F_p": ftest["p_value"],
                "RMS_holdout_prev": r_red.rms_holdout,
                "RMS_holdout_next": r_full.rms_holdout,
                "delta_RMS_holdout": r_full.rms_holdout - r_red.rms_holdout,
            }
        )
    return pd.DataFrame(nested_rows)


def seed_objectives() -> dict[float, AnisotropyObjective]:
    """Load graded + DFT pickles and retarget shared geometry across E_RES."""
    if not REFLOXIDE_DFT_PATH.exists():
        raise FileNotFoundError(f"missing {REFLOXIDE_DFT_PATH}")

    data = read_xrr("reflectivity_data", material="znpc", source="hub")
    graded_fit = read_fit("graded/graded_fit.pkl", material="znpc", source="local")
    graded_ctx = extract_graded_fit_context(graded_fit)
    dft_objective = load_refloxide_fit(REFLOXIDE_DFT_PATH)
    dft_corrections = dft_objective.model.corrections

    log(f"graded anchor energy: {graded_ctx.energy:g} eV")
    log(f"resonant series: {', '.join(f'{e:g}' for e in E_RES)} eV")
    log(f"DFT slab energies: {list(dft_corrections.energies)}")

    objectives = objectives_retarget_energies(graded_fit, list(E_RES), data=data)

    for energy in E_RES:
        obj = objectives[float(energy)]
        if np.isclose(energy, ENERGY_ANCHOR):
            values = {
                name: float(getattr(obj.model, name).value or 0.0)
                for name in REFLECT_MODEL_INSTRUMENTATION
            }
            source = "graded@283.7"
        else:
            values = dft_instrument_values(dft_corrections, energy)
            apply_fixed_instrumentation(obj.model, values)
            source = "dft-slab"
        for name in REFLECT_MODEL_INSTRUMENTATION:
            safely_setp_param(getattr(obj.model, name), vary=False)
        pin_calibrated_instrument(obj, values)
        log(
            f"  seed {energy:6.1f} eV [{source:11s}]  "
            f"scale_s={values['scale_s']:.4f}  "
            f"scale_p={values['scale_p']:.4f}  "
            f"ths={values['theta_offset_s']:+.4f}  "
            f"thp={values['theta_offset_p']:+.4f}  "
            f"E_off={values['energy_offset']:+.4f}"
        )
    return objectives


def rebuild_stage_from_csv(  # noqa: PLR0913
    objectives: dict[float, AnisotropyObjective],
    stage: int,
    geom_df: pd.DataFrame,
    per_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    *,
    polish_holdout: bool = False,
) -> tuple[dict[float, AnisotropyObjective], StageResult]:
    """Rebuild a completed stage from CSV snapshots.

    Parameters
    ----------
    polish_holdout
        When ``True``, keep train instruments from the CSV but L-BFGS-B-polish
        holdout scale/theta under the frozen geometry before scoring.
    """
    geom_row = geom_df.loc[geom_df["stage"] == stage].iloc[0]
    geometry = {name: float(geom_row[name]) for name in GEOMETRY_PARAMS}
    inst_rows = per_df.loc[per_df["stage"] == stage]
    instrument: dict[float, dict[str, float]] = {}
    for _, row in inst_rows.iterrows():
        instrument[float(row["energy"])] = {
            name: float(row[name]) for name in FREE_INSTRUMENT
        }

    obj_map = build_yx4_objectives(objectives)
    apply_geometry_snapshot(obj_map, geometry)
    apply_instrument_snapshot(obj_map, instrument, E_RES)
    for _e, _obj in obj_map.items():
        pin_calibrated_instrument(_obj)

    cv_row = cv_df.loc[cv_df["stage"] == stage].iloc[0]
    train = tuple(E_RES[:stage])
    holdouts = tuple(e for e in E_RES if e not in train)
    if polish_holdout and holdouts:
        log(f"  rescoring stage {stage}: polishing holdout instruments {holdouts}")
        polish_instruments(obj_map, holdouts, label=f"holdout-stage{stage}")

    result = score_stage(
        obj_map,
        train,
        fitted_geometry=bool(cv_row["fitted_geometry"]),
        wall_s=float(cv_row["wall_s"]),
    )
    return obj_map, result


def rescore_holdouts(
    objectives: dict[float, AnisotropyObjective],
) -> int:
    """Re-polish holdout instruments on existing stage geometries; rewrite CSVs."""
    cv_df = pd.read_csv(OUT_DIR / "cv_table.csv")
    geom_df = pd.read_csv(OUT_DIR / "geometry_by_stage.csv")
    per_df = pd.read_csv(OUT_DIR / "per_energy.csv")
    stages = sorted(int(s) for s in cv_df["stage"].tolist())
    if not stages:
        raise FileNotFoundError(f"no stages in {OUT_DIR / 'cv_table.csv'}")

    stage_maps: list[dict[float, AnisotropyObjective]] = []
    stage_results: list[StageResult] = []
    for stage in stages:
        log(f"\n######## RESCORE HOLDOUTS STAGE {stage} ########")
        obj_k, res_k = rebuild_stage_from_csv(
            objectives,
            stage,
            geom_df,
            per_df,
            cv_df,
            polish_holdout=True,
        )
        stage_maps.append(obj_k)
        stage_results.append(res_k)
        log_stage_summary(res_k)
        save_checkpoint(OUT_DIR, stage_results, stage_maps, final=False)

    nested_table = build_nested_table(objectives, stage_results, stage_maps)
    nested_table.to_csv(OUT_DIR / "nested_tests.csv", index=False)
    log("\nNested tests:")
    log(nested_table.round(4).to_string(index=False))
    save_checkpoint(
        OUT_DIR,
        stage_results,
        stage_maps,
        final=len(stages) == len(E_RES),
    )
    log(f"\nDONE rescore. Bundle: {OUT_DIR / 'bundle.pkl'}")
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-from",
        type=int,
        default=1,
        help="First stage to (re)fit; earlier stages reload from CSV checkpoints.",
    )
    parser.add_argument(
        "--rescore-holdouts",
        action="store_true",
        help=(
            "Keep saved geometries / train instruments; re-polish holdout "
            "scale/theta and rewrite holdout metrics (no DE)."
        ),
    )
    args = parser.parse_args(argv)
    resume_from = int(args.resume_from)
    if resume_from < 1 or resume_from > len(E_RES):
        raise SystemExit(f"--resume-from must be in 1..{len(E_RES)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"output directory: {OUT_DIR}")
    log(f"DE_KW={DE_KW}  LBFGS_KW={LBFGS_KW}")
    log(
        f"theta bounds={INSTRUMENT_BOUNDS['theta_offset_s']}  resume_from={resume_from}"
    )

    objectives = seed_objectives()

    if args.rescore_holdouts:
        return rescore_holdouts(objectives)

    yx4_seed = build_yx4_objectives(objectives)
    seed_geometry = geometry_snapshot(yx4_seed)
    seed_instrument = instrument_snapshot(yx4_seed)

    stage_maps: list[dict[float, AnisotropyObjective]] = []
    stage_results: list[StageResult] = []

    if resume_from > 1:
        cv_df = pd.read_csv(OUT_DIR / "cv_table.csv")
        geom_df = pd.read_csv(OUT_DIR / "geometry_by_stage.csv")
        per_df = pd.read_csv(OUT_DIR / "per_energy.csv")
        for stage in range(1, resume_from):
            if stage not in set(cv_df["stage"].astype(int)):
                raise FileNotFoundError(
                    f"cannot resume from {resume_from}: "
                    f"missing stage {stage} in cv_table.csv"
                )
            log(f"\n######## RELOAD STAGE {stage} from CSV checkpoint ########")
            obj_k, res_k = rebuild_stage_from_csv(
                objectives,
                stage,
                geom_df,
                per_df,
                cv_df,
                polish_holdout=True,
            )
            stage_maps.append(obj_k)
            stage_results.append(res_k)
            log_stage_summary(res_k)
        warm_geometry = stage_results[-1].geometry
        warm_instrument = stage_results[-1].instrument
        save_checkpoint(OUT_DIR, stage_results, stage_maps, final=False)
    else:
        log("\n######## STAGE 1: evaluate seeded 283.7 eV MLE (no DE) ########")
        obj1, res1 = evaluate_seed_stage(objectives, seed_geometry, seed_instrument)
        stage_maps.append(obj1)
        stage_results.append(res1)
        log_stage_summary(res1)
        if res1.fitted_geometry:
            raise RuntimeError("stage 1 must not refit geometry")
        save_checkpoint(OUT_DIR, stage_results, stage_maps, final=False)
        warm_geometry = res1.geometry
        warm_instrument = res1.instrument

    start_k = 2 if resume_from <= 1 else resume_from
    for k in range(start_k, len(E_RES) + 1):
        train = tuple(E_RES[:k])
        log(
            f"\n######## STAGE {k}: DE geometry on {train}, "
            f"holdout {tuple(e for e in E_RES if e not in train)} ########"
        )
        obj_k, res_k = fit_stage(objectives, train, warm_geometry, warm_instrument)
        stage_maps.append(obj_k)
        stage_results.append(res_k)
        log_stage_summary(res_k)
        if not res_k.fitted_geometry:
            raise RuntimeError(f"stage {k} must fit geometry")
        save_checkpoint(OUT_DIR, stage_results, stage_maps, final=False)
        warm_geometry = res_k.geometry
        warm_instrument = res_k.instrument

    nested_table = build_nested_table(objectives, stage_results, stage_maps)
    nested_table.to_csv(OUT_DIR / "nested_tests.csv", index=False)
    log("\nNested tests:")
    log(nested_table.round(4).to_string(index=False))

    save_checkpoint(OUT_DIR, stage_results, stage_maps, final=True)

    hold = pd.read_csv(OUT_DIR / "cv_table.csv").dropna(subset=["RMS_holdout"])
    if len(hold):
        best_rms = float(hold["RMS_holdout"].min())
        tol = 0.05 * best_rms if best_rms > 0 else 0.0
        pick = int(hold.loc[hold["RMS_holdout"] <= best_rms + tol, "stage"].iloc[0])
        log(
            f"\nHoldout RMS plateau: best={best_rms:.4g}; "
            f"first stage within 5% = {pick}"
        )

    log(f"\nDONE. Bundle: {OUT_DIR / 'bundle.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

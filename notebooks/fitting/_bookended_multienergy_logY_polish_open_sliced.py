"""Refit the open logY gradient polish on the FIG4 point mask.

Drops the circled low-Q outliers, keeps the 283.7 eV low-Q anchor, pins every
energy's background to zero, and writes
``bookended_multienergy_logY_polish_open_sliced/`` from the current open
polish MLE. The locked and open polish pickles are left unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from refloxide.data import ReflectDataset
from refloxide.model import BookendedComponent, ReflectModel, Slab
from refloxide.objective import Objective

_HERE = Path(__file__).resolve().parent
_OPEN = _HERE / "_bookended_multienergy_logY_polish_open.py"
_spec = importlib.util.spec_from_file_location(
    "_bookended_multienergy_logY_polish_open", _OPEN
)
op = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = op
assert _spec.loader is not None
_spec.loader.exec_module(op)

cv = op.cv
ENERGIES = op.ENERGIES
ANCHOR = op.ANCHOR
INSTRUMENT_FIELDS = op.INSTRUMENT_FIELDS
LOCKED_ENERGY_OFFSET = op.LOCKED_ENERGY_OFFSET
TRANSFORM = op.TRANSFORM
OUT_DIR = cv.models_fitting_results / "bookended_multienergy_logY_polish_open_sliced"

DROP_Q: dict[tuple[float, str], tuple[float, ...]] = {
    (284.2, "p"): (0.01361, 0.01465, 0.01570),
    (284.2, "s"): (0.03449, 0.03656, 0.03760, 0.03863),
    (285.1, "p"): (0.04188,),
    (287.0, "p"): (0.068578, 0.070815),
}
Q_ATOL = 5e-4


def log(msg: str) -> None:
    """Print a progress line immediately."""
    print(msg, flush=True)  # noqa: T201


def kept_q(energy: float, pol: str, q: np.ndarray) -> np.ndarray:
    """Return a boolean mask that drops the FIG4 outliers and keeps every other row.

    Parameters
    ----------
    energy
        Photon energy in eV.
    pol
        Polarization label, ``"s"`` or ``"p"``.
    q
        Momentum transfer in inverse angstroms.

    Returns
    -------
    numpy.ndarray
        Boolean mask aligned with ``q``. True rows stay in the objective.
    """
    keep = np.ones(q.shape, dtype=bool)
    for q0 in DROP_Q.get((float(energy), pol), ()):
        keep &= ~np.isclose(q, q0, rtol=0.0, atol=Q_ATOL)
    return keep


THETA_HI = 1.2
DENSITY_NAMES = ("density_vac", "density_si", "density_bulk")


def pin_backgrounds(obj_map: dict) -> None:
    """Set every energy background to zero and hold it fixed.

    Parameters
    ----------
    obj_map
        Energy-keyed pyref objectives whose models own the instrument fields
        copied into the joint logY objective.
    """
    for energy, obj in obj_map.items():
        obj.model.bkg.setp(0.0, vary=False)
        log(f"bkg {float(energy):g} eV = 0 (fixed)")


def retighten_theta(obj_map: dict, hi: float = THETA_HI) -> None:
    """Set theta bounds from positive-Q grazing angles and raise the ceiling.

    The exact ``Q = 0`` anchor is kept in the objective, but it is not used to
    set the lower bound. Treating that row as a zero-degree ray pins
    ``theta_offset_s`` at 283.7 eV and the polish then inflates the substrate
    density to chase the critical edge.

    Parameters
    ----------
    obj_map
        Energy-keyed objectives whose instrument parameters are copied into
        the joint model.
    hi
        Upper bound on both theta offsets, in degrees.
    """
    for energy, obj in obj_map.items():
        for pol, field in (("s", "theta_offset_s"), ("p", "theta_offset_p")):
            q = np.asarray(getattr(obj.data, pol).x, dtype=float)
            q = q[q > 0.0]
            grazing = cv.min_data_grazing_angle_deg(q, float(energy))
            safe_lb = cv.MIN_INCIDENT_THETA_DEG + cv.THETA_OFFSET_MARGIN_DEG - grazing
            lb_old, _ub = cv.INSTRUMENT_BOUNDS[field]
            new_lb = max(float(lb_old), float(safe_lb))
            param = getattr(obj.model, field)
            value = float(param.value)
            value = max(value, new_lb)
            value = min(value, hi)
            cv.safely_setp_param(param, value=value, vary=True, bounds=(new_lb, hi))
            log(
                f"  theta {float(energy):g} eV {field}: "
                f"[{new_lb:.4f}, {hi:.4f}] value={value:.4f}"
            )


def freeze_structure(film, oxide, substrate) -> None:
    """Hold film, oxide, and substrate geometry at the warm-start values.

    Parameters
    ----------
    film, oxide, substrate
        Structure components shared with the joint reflectivity model.
    """
    for name in op.FILM_FREE:
        getattr(film, name).vary = False
    oxide.thick.vary = False
    oxide.rough.vary = False
    oxide.sld.density.vary = False
    substrate.sld.density.vary = False


def open_nondensity_structure(film, oxide) -> None:
    """Reopen film shape and oxide geometry while densities stay at the warm start.

    Parameters
    ----------
    film, oxide
        Structure components shared with the joint reflectivity model.
    """
    bounds = op.film_bounds(
        {name: float(getattr(film, name).value) for name in op.FILM_FREE}
    )
    for name in op.FILM_FREE:
        if name in DENSITY_NAMES:
            getattr(film, name).vary = False
            continue
        cv.safely_setp_param(
            getattr(film, name),
            value=float(getattr(film, name).value),
            vary=True,
            bounds=bounds[name],
        )
    oxide.thick.vary = True
    oxide.rough.vary = True
    oxide.sld.density.vary = False


def sync_structure(obj_map: dict) -> None:
    """Copy the anchor structure onto every energy's pyref model.

    The joint polish updates the anchor film, oxide, and substrate. The
    manuscript figure evaluates each energy through its own model, so those
    copies have to carry the same geometry or the drawn curves are the
    warm-start stack with only the new instruments.

    Parameters
    ----------
    obj_map
        Energy-keyed objectives. The anchor entry is the one the joint model
        wraps.
    """
    anchor = obj_map[ANCHOR].model.structure.components
    _vacuum, film_a, oxide_a, sub_a = anchor
    for obj in obj_map.values():
        _v, film, oxide, sub = obj.model.structure.components
        if film is film_a:
            continue
        for name in op.FILM_FREE:
            getattr(film, name).value = float(getattr(film_a, name).value)
        oxide.thick.value = float(oxide_a.thick.value)
        oxide.rough.value = float(oxide_a.rough.value)
        oxide.sld.density.value = float(oxide_a.sld.density.value)
        sub.sld.density.value = float(sub_a.sld.density.value)


def linear_coverage(objective: Objective) -> tuple[float, float]:
    """Return the percent of points within 2 sigma, overall and for ``Q <= 0.08``.

    Parameters
    ----------
    objective
        Fitted joint objective. ``generative`` is physical reflectivity.

    Returns
    -------
    tuple of float
        ``(all_percent, low_q_percent)``.
    """
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    r = np.asarray(ds.r, dtype=float)
    err = np.asarray(ds.r_err, dtype=float)
    q = np.asarray(ds.q, dtype=float)
    chi = np.abs(r - y_fit) / np.maximum(err, 1e-30)
    low = q <= 0.08
    return 100.0 * float(np.mean(chi < 2.0)), 100.0 * float(np.mean(chi[low] < 2.0))


def build_sliced_objective(obj_map: dict) -> Objective:
    """Build the joint logY objective on the masked reflectivity rows.

    Parameters
    ----------
    obj_map
        Energy-keyed objectives after the open-polish MLE and zero backgrounds
        have been applied.

    Returns
    -------
    refloxide.objective.Objective
        Joint objective over ``ENERGIES`` with the FIG4 outliers removed and
        ``energy_offset`` locked.

    Raises
    ------
    RuntimeError
        When per-energy anisotropy weights are not identical.
    """
    comps = obj_map[ANCHOR].model.structure.components
    vacuum, film, oxide, substrate = comps
    structure = (
        Slab(
            float(vacuum.thick.value),
            op._material(vacuum, "vacuum"),
            float(vacuum.rough.value),
            "vacuum",
        )
        | BookendedComponent(film)
        | Slab(oxide.thick, op._material(oxide, "oxide"), oxide.rough, "oxide")
        | Slab(
            float(substrate.thick.value),
            op._material(substrate, "substrate"),
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
    model.energy_offset.setp(LOCKED_ENERGY_OFFSET, vary=False)

    rows: dict[str, list] = {k: [] for k in ("q", "energy", "pol", "r", "r_err")}
    targets = {}
    n_drop = 0
    for energy in energies:
        data = obj_map[energy].data
        dropped_q: list[float] = []
        for pol in ("s", "p"):
            sub = getattr(data, pol)
            q = np.asarray(sub.x, dtype=float)
            keep = ((q >= cv.Q_MIN) | (q == 0.0)) & kept_q(float(energy), pol, q)
            n_drop += int((~keep).sum())
            dropped_q.extend(q[~keep].tolist())
            rows["q"].append(q[keep])
            rows["energy"].append(np.full(int(keep.sum()), energy))
            rows["pol"].append(np.full(int(keep.sum()), pol, dtype=object))
            rows["r"].append(np.asarray(sub.y, dtype=float)[keep])
            rows["r_err"].append(np.asarray(sub.y_err, dtype=float)[keep])
        ax = np.asarray(data.anisotropy.x, dtype=float)
        ay = np.asarray(data.anisotropy.y, dtype=float)
        akeep = kept_q(float(energy), "s", ax) & kept_q(float(energy), "p", ax)
        targets[energy] = (ax[akeep], ay[akeep])
        log(
            f"drop {float(energy):g} eV  n={len(dropped_q)}  "
            f"q={np.array2string(np.asarray(dropped_q), precision=5)}"
        )
    log(f"dropped rows={n_drop}")
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


def main() -> None:
    """Warm from the unmasked open polish, then polish the sliced objective."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_in, warm_dir = op.resolve_warm_summary(prefer_open=True)
    log(f"warm from {warm_dir}  protocol={summary_in.get('protocol')}")

    objectives = cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in ENERGIES}
    _vacuum, film, oxide, substrate = op.apply_open_mle(obj_map, summary_in)
    pin_backgrounds(obj_map)
    log("theta bounds from Q>0, ceiling widened")
    retighten_theta(obj_map)
    freeze_structure(film, oxide, substrate)
    log(
        f"stage 1 instruments only  "
        f"Si_rho={float(substrate.sld.density.value):.3f} (fixed)  "
        f"density_si={float(film.density_si.value):.3f} (fixed)"
    )

    objective = build_sliced_objective(obj_map)
    before = op.metrics(objective)
    cov_all, cov_lo = linear_coverage(objective)
    log(f"before (sliced logY): {before}  within2={cov_all:.1f}%  lowQ={cov_lo:.1f}%")
    bkg_vary = [
        p.name for p in objective.varying_parameters() if p.name.startswith("bkg")
    ]
    if bkg_vary:
        raise RuntimeError(f"background still free: {bkg_vary}")

    history = op.polish(objective)
    cov_all, cov_lo = linear_coverage(objective)
    log(f"stage 1 within2={cov_all:.1f}%  lowQ={cov_lo:.1f}%")
    for hit in op.bound_hits(objective):
        log(f"  stage 1 bound {hit}")

    best_values = [
        (p, float(p.value))
        for p in objective.parameters.flattened()
        if p.constraint is None
    ]
    best_cov = cov_all
    open_nondensity_structure(film, oxide)
    objective.refresh_parameters()
    log("stage 2 reopen film shape and oxide geometry; densities stay fixed")
    history.extend(op.polish(objective))
    cov2, lo2 = linear_coverage(objective)
    log(f"stage 2 within2={cov2:.1f}%  lowQ={lo2:.1f}%")
    for hit in op.bound_hits(objective):
        log(f"  stage 2 bound {hit}")
    if cov2 + 0.05 < best_cov:
        log(f"stage 2 coverage dropped {best_cov:.1f} -> {cov2:.1f}; restoring stage 1")
        for par, value in best_values:
            par.value = value
        history.append({"stage": "restore_instruments_only", "within2": best_cov})
    else:
        best_cov = cov2

    after = op.metrics(objective)
    per = op.per_energy_rms(objective)
    hits = op.bound_hits(objective)
    cov_all, cov_lo = linear_coverage(objective)
    log(f"kept within2={cov_all:.1f}%  lowQ={cov_lo:.1f}%")

    for energy in ENERGIES:
        lm = obj_map[energy].model
        for field in INSTRUMENT_FIELDS:
            getattr(lm, field).value = float(
                getattr(objective.model, field).at(energy).value
            )
        lm.energy_offset.value = float(objective.model.energy_offset.value)
        if abs(float(lm.bkg.value)) > 0.0:
            raise RuntimeError(f"bkg drifted at {energy}: {float(lm.bkg.value)}")
    sync_structure(obj_map)

    film_snap = {name: float(getattr(film, name).value) for name in op.FILM_FREE}
    oxide_snap = {
        "thick": float(oxide.thick.value),
        "rough": float(oxide.rough.value),
        "density": float(oxide.sld.density.value),
    }
    substrate_snap = {"density": float(substrate.sld.density.value)}
    instruments = {
        float(energy): {
            name: float(getattr(obj_map[energy].model, name).value)
            for name in INSTRUMENT_FIELDS
        }
        for energy in ENERGIES
    }
    summary = {
        "energies": list(ENERGIES),
        "transform": "logY",
        "before": before,
        "after": after,
        "history": history,
        "per_energy": per,
        "film": film_snap,
        "oxide": oxide_snap,
        "substrate": substrate_snap,
        "energy_offset": float(objective.model.energy_offset.value),
        "instruments": instruments,
        "bound_hits": hits,
        "tau_vac_bounds": list(op.TAU_VAC_BOUNDS),
        "protocol": "cpu_lbfgs_logY_sliced_instruments_then_shape_densities_fixed",
        "warm_from": str(warm_dir),
        "energy_offset_locked": True,
        "background_locked": 0.0,
        "dropped_q": {
            f"{energy:g}_{pol}": list(qs) for (energy, pol), qs in DROP_Q.items()
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT_DIR / "fit.pkl").open("wb") as fh:
        pickle.dump(
            {"obj_map": obj_map, "gpu_objective": objective, "summary": summary},
            fh,
        )
    log(f"after: {after}")
    for row in per:
        log(f"  {row['energy']:g} eV  RMS={row['rms']:.3f}")
    if hits:
        log("bound hits:")
        for hit in hits:
            log(f"  {hit}")
    else:
        log("bound hits: none")
    log(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()

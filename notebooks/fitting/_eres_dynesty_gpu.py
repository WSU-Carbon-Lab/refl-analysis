"""GPU dynesty nested sampling for bookended vs DFT-slab on ``E_RES``.

Default contract matches the multi-energy logY L-BFGS polish:
``Transform('logY')``, ``anisotropy_weight=0``, ``normalization='auto'``
(point-weighted), ``nc_constraint=True`` folded into the dynesty score via
``logl + logp``, and the polish free set (no geometry unlock).

Modes
-----
* ``--probe-minutes N`` — wall-clock smoke probe (throughput check).
* ``--full`` — production evidence + posterior run with ndim-scaled
  ``nlive`` / ``walks``, ``dlogz_init``, and ``n_effective`` stopping.
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
from dynesty import DynamicNestedSampler
from refnx.analysis import Transform
from scipy.optimize import minimize
from utils import models_fitting_results
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

_HERE = Path(__file__).resolve().parent
_STAGE1 = _HERE / "_dft_slab_eres_logyx4.py"
_spec = importlib.util.spec_from_file_location("_dft_slab_eres_logyx4", _STAGE1)
s1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = s1
assert _spec.loader is not None
_spec.loader.exec_module(s1)

BOOK_PKL = models_fitting_results / "bookended_multienergy_logY_polish" / "fit.pkl"
SLAB_PKL = models_fitting_results / "dft_slab_eres_logY_polish" / "fit.pkl"
OUT_DIR = models_fitting_results / "eres_dynesty_gpu"
FULL_DIR = OUT_DIR / "full"
FULL_UNLOCKED_DIR = OUT_DIR / "full_unlocked"
# Honest polish-contract evidence (locked free set, point-weighted, aniso=0).
FULL_POLISH_DIR = OUT_DIR / "full_polish_contract"
# Same likelihood contract with film/oxide unlocked + widened tau_vac.
FULL_OPEN_DIR = OUT_DIR / "full_open_contract"
# Instruments + oxide frozen; film profile free (bookended + slab).
FULL_PROFILE_DIR = OUT_DIR / "full_profile_contract"
OPEN_CENTROID_CSV = FULL_OPEN_DIR / "bookended" / "snapshot_best" / "parameters.csv"
OPEN_SNAPSHOT_PARAMS = FULL_OPEN_DIR / "bookended" / "snapshot_best" / "parameters.csv"
TRANSFORM = Transform("logY")
ANISOTROPY_WEIGHT = 0.0
NORMALIZATION = "auto"
SAMPLE = "rwalk"
BOUND = "multi"
NLIVE_PER_DIM = 50
WALKS_PER_DIM = 3
DLOGZ_INIT = 0.01
N_EFFECTIVE = 25_000
SEED = 1
STATUS_WRITE_S = 30.0
STATUS_WRITE_NIT = 200
LOGP_REJECT = -1.0e300


def log(msg: str) -> None:
    print(msg, flush=True)


def nlive_for_ndim(ndim: int, override: int | None) -> int:
    if override is not None and override > 0:
        return int(override)
    return max(1000, int(NLIVE_PER_DIM * ndim))


def walks_for_ndim(ndim: int, override: int | None) -> int:
    if override is not None and override > 0:
        return int(override)
    return max(25, int(WALKS_PER_DIM * ndim))


def _finite_bounds(objective) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names: list[str] = []
    lo: list[float] = []
    hi: list[float] = []
    for p in objective.varying_parameters():
        lb, ub = float(p.bounds.lb), float(p.bounds.ub)
        if not np.isfinite(lb) or not np.isfinite(ub) or ub <= lb:
            raise ValueError(f"non-finite bounds for {p.name}: {(lb, ub)}")
        names.append(str(p.name))
        lo.append(lb)
        hi.append(ub)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), names


def make_prior_transform(lo: np.ndarray, hi: np.ndarray) -> UniformPriorTransform:
    return UniformPriorTransform(lo, hi)


def make_loglike(objective) -> ObjectiveLogPost:
    return ObjectiveLogPost(objective)


class UniformPriorTransform:
    """Picklable unit-cube → box prior for dynesty checkpoints."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray) -> None:
        self.lo = np.asarray(lo, dtype=float)
        self.span = np.asarray(hi, dtype=float) - self.lo

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.asarray(u, dtype=float) * self.span


class ObjectiveLogPost:
    """Dynesty score ``logl + logp`` (NC + bounds via ``nc_constraint``)."""

    def __init__(self, objective) -> None:
        self.objective = objective

    def __call__(self, x: np.ndarray) -> float:
        try:
            x_arr = np.asarray(x, dtype=float)
            logp = float(self.objective.logp(x_arr))
            if not np.isfinite(logp):
                return LOGP_REJECT
            logl = float(self.objective.logl(x_arr))
            if not np.isfinite(logl):
                return LOGP_REJECT
            return logl + logp
        except (ValueError, RuntimeError, FloatingPointError):
            return LOGP_REJECT


def configure_objective(objective, *, label: str):
    """GPU objective matching the logY L-BFGS polish likelihood contract."""
    objective.model.device = "gpu"
    objective.model.parallel = False
    objective.transform = TRANSFORM
    objective.anisotropy_weight = ANISOTROPY_WEIGHT
    if hasattr(objective, "normalization"):
        objective.normalization = NORMALIZATION
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()
    lo, hi, names = _finite_bounds(objective)
    x0 = np.array([float(p.value) for p in objective.varying_parameters()], dtype=float)
    nll = float(objective.nll())
    logl0 = float(objective.logl(x0))
    logp0 = float(objective.logp(x0))
    log(
        f"{label}: nvary={len(names)}  nll={nll:.6f}  logl0={logl0:.6f}  "
        f"logp0={logp0:.6f}  device={objective.model.device}  "
        f"aniso={objective.anisotropy_weight}  norm={NORMALIZATION}  "
        f"nc={getattr(objective, 'nc_constraint', None)}  "
        f"transform={TRANSFORM}"
    )
    return lo, hi, names, x0, logl0


def load_bookended(*, unlock: bool = False, contract: str = "polish"):
    with BOOK_PKL.open("rb") as fh:
        objective = pickle.load(fh)["gpu_objective"]
    mode = str(contract)
    if unlock and mode == "polish":
        mode = "open"
    if mode == "open":
        unlock_bookended(objective)
    elif mode == "profile":
        configure_bookended_profile_contract(objective)
    elif mode != "polish":
        raise ValueError(f"unknown bookended contract: {contract!r}")
    return objective


def correct_dft_slab_polarization(
    objective,
    *,
    swap_instruments: bool = True,
) -> bool:
    """Fix swapped s/p labels on a DFT-slab objective loaded from polish pickle.

    The warm-start / ``dft_slab_eres_logY_polish`` datasets label the 283.7 eV
    Brewster deep-min channel as ``s``; bookended (and the physical model
    channels) put it on ``p``. Mutating ``dataset.pol`` alone is not enough:
    ``Objective`` caches ``_groups`` and ``_batches`` at construction, so both
    must be rebuilt or the loss keeps scoring the old channel assignment.

    When ``swap_instruments`` is True, per-energy ``scale_*`` / ``theta_offset_*``
    values and bounds are exchanged so polish-tuned nuisances stay paired with
    the physical reflectivity they were fit against. Profile contract then
    overwrites those with the shared bookended centroid.

    Returns
    -------
    bool
        True when labels were flipped.
    """
    from refloxide.data import ReflectDataset
    from refloxide.objective import _build_kernel_batches

    ds = objective._dataset
    energy = np.asarray(ds.energy, dtype=float)
    pol = np.asarray(ds.pol)
    r = np.asarray(ds.r, dtype=float)
    if not s1.polarization_labels_swapped(energy, pol, r):
        return False

    corrected = ReflectDataset(
        q=np.asarray(ds.q, dtype=float).copy(),
        energy=energy.copy(),
        pol=s1.flip_pol_labels(pol),
        r=r.copy(),
        r_err=np.asarray(ds.r_err, dtype=float).copy(),
    )
    objective._dataset = corrected
    objective._groups = list(corrected.groups())
    objective._batches = _build_kernel_batches(corrected)
    objective._refresh_data_transform_cache()

    if swap_instruments:
        model = objective.model
        energies = sorted({float(e) for e, _p, _i in corrected.groups()})
        for e in energies:
            for field_s, field_p in (
                ("scale_s", "scale_p"),
                ("theta_offset_s", "theta_offset_p"),
            ):
                ps = getattr(model, field_s).at(e)
                pp = getattr(model, field_p).at(e)
                vs, vp = float(ps.value), float(pp.value)
                bs = (float(ps.bounds.lb), float(ps.bounds.ub))
                bp = (float(pp.bounds.lb), float(pp.bounds.ub))
                vary_s, vary_p = bool(ps.vary), bool(pp.vary)
                ps.setp(value=vp, bounds=bp, vary=vary_s)
                pp.setp(value=vs, bounds=bs, vary=vary_p)

    objective.refresh_parameters()
    r_s = r[np.isclose(energy, 283.7) & (corrected.pol == "s")]
    r_p = r[np.isclose(energy, 283.7) & (corrected.pol == "p")]
    log(
        "dft_slab: corrected swapped s/p labels "
        f"(283.7 eV Rmin s={float(np.min(r_s)):.3e}, p={float(np.min(r_p)):.3e}; "
        f"instruments_swapped={swap_instruments})"
    )
    return True


def load_slab(*, unlock: bool = False, contract: str = "polish"):
    with SLAB_PKL.open("rb") as fh:
        objective = pickle.load(fh)["objective"]
    correct_dft_slab_polarization(objective, swap_instruments=True)
    mode = str(contract)
    if unlock and mode == "polish":
        mode = "open"
    if mode == "open":
        unlock_slab(objective)
    elif mode == "profile":
        configure_slab_profile_contract(objective)
    elif mode != "polish":
        raise ValueError(f"unknown slab contract: {contract!r}")
    return objective


def timed_logl(objective, n: int = 30) -> float:
    x = np.array([float(p.value) for p in objective.varying_parameters()], dtype=float)
    for _ in range(3):
        objective.logl(x)
    t0 = time.perf_counter()
    for _ in range(n):
        objective.logl(x)
    return (time.perf_counter() - t0) / n


def _local_bounds(
    value: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    rel: float | None = 0.15,
    abs_pad: float | None = None,
) -> tuple[float, float]:
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


def _setp(
    param, *, value: float | None = None, vary: bool, bounds: tuple[float, float]
):
    kwargs: dict = {"vary": vary, "bounds": bounds}
    if value is not None:
        kwargs["value"] = float(value)
    param.setp(**kwargs)


# Film geometry unlocked for dynesty (oxide SiO2/Si densities stay tabulated).
BOOKENDED_OPEN = (
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
# Was (0.5, 32) and trapped the polish MLE; (8, 55) still clipped under
# locked runs at 32. Open the vacuum length scale with a generous ceiling.
TAU_VAC_BOUNDS = (5.0, 80.0)
TAU_SI_BOUNDS = (3.0, 45.0)
# Profile-contract: leave headroom below the open-contract left-skew (~14 Å).
PROFILE_TAU_VAC_BOUNDS = (1.0, 80.0)
PROFILE_TAU_SI_BOUNDS = (3.0, 45.0)
# Profile σ_surf piled at polish+3 Å (~11.36); open the ceiling.
PROFILE_SURFACE_ROUGH_BOUNDS = (3.0, 20.0)
# Open-contract piled rho_vac at ~2.35 and rho_si at ~1.03.
PROFILE_DENSITY_VAC_BOUNDS = (1.5, 5.0)
PROFILE_DENSITY_SI_BOUNDS = (0.5, 1.6)
BOUNDARY_EPS_FRAC = 0.02
BOUNDARY_MASS_WARN = 0.05


def _load_open_contract_medians(path: Path | None = None) -> dict[str, float]:
    """Read posterior medians from the aborted open-contract snapshot table."""
    import csv

    csv_path = path if path is not None else OPEN_CENTROID_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"open-contract centroid table missing: {csv_path}")
    out: dict[str, float] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            out[str(row["name"])] = float(row["median"])
    return out


def _freeze_instruments_at(objective, values: dict[str, float]) -> list[str]:
    """Pin scale/theta to ``values`` and set ``vary=False``."""
    frozen: list[str] = []
    for p in list(objective.parameters.flattened()):
        name = str(p.name)
        if not (name.startswith("scale_") or name.startswith("theta_offset_")):
            continue
        if name not in values:
            continue
        val = float(values[name])
        lb, ub = float(p.bounds.lb), float(p.bounds.ub)
        # Keep a tiny box so refnx still has finite bounds while frozen.
        pad = max(0.02 * abs(val), 1e-3)
        a, b = val - pad, val + pad
        a = min(a, lb) if np.isfinite(lb) else a
        b = max(b, ub) if np.isfinite(ub) else b
        if a >= b:
            a, b = val - pad, val + pad
        _setp(p, value=val, vary=False, bounds=(a, b))
        frozen.append(name)
    return frozen


def configure_bookended_profile_contract(objective) -> list[str]:
    """Freeze instruments + oxide; free film profile with wall-aware priors.

    Instruments take the open-contract posterior median. Oxide thick/rough and
    Si/SiO2 densities stay at the L-BFGS polish values with ``vary=False``.
    ``rho_vac`` / ``rho_si`` / ``tau_vac`` priors are opened past the walls seen
    in ``full_open_contract``.
    """
    centroid = _load_open_contract_medians()
    film = objective.model.structure[1].profile
    oxide = objective.model.structure[2]
    substrate = objective.model.structure[3]
    pi2 = float(np.pi / 2)

    frozen_instr = _freeze_instruments_at(objective, centroid)

    # Oxide / Si from polish (L-BFGS) — do not refine.
    ox_thick = float(oxide.thick.value)
    ox_rough = float(oxide.rough.value)
    _setp(
        oxide.thick,
        value=ox_thick,
        vary=False,
        bounds=(max(0.5, ox_thick - 1.0), ox_thick + 1.0),
    )
    _setp(
        oxide.rough,
        value=ox_rough,
        vary=False,
        bounds=(max(0.1, ox_rough - 1.0), ox_rough + 1.0),
    )
    oxide.sld.density.setp(vary=False)
    substrate.sld.density.setp(vary=False)

    # Seed film knobs from open-contract median when available.
    for name in BOOKENDED_OPEN:
        if name in centroid:
            getattr(film, name).value = float(centroid[name])

    bounds = {
        "total_thick": _local_bounds(
            float(film.total_thick.value), lo=170.0, hi=210.0, rel=0.08
        ),
        "surface_roughness": PROFILE_SURFACE_ROUGH_BOUNDS,
        "density_bulk": _local_bounds(
            float(film.density_bulk.value), lo=1.2, hi=2.1, rel=0.15
        ),
        "density_si": PROFILE_DENSITY_SI_BOUNDS,
        "density_vac": PROFILE_DENSITY_VAC_BOUNDS,
        "tau_si": PROFILE_TAU_SI_BOUNDS,
        "tau_vac": PROFILE_TAU_VAC_BOUNDS,
        "alpha_bulk": _local_bounds(
            float(film.alpha_bulk.value), lo=0.9, hi=pi2, abs_pad=0.20, rel=None
        ),
        "alpha_si": _local_bounds(
            float(film.alpha_si.value), lo=0.15, hi=0.95, abs_pad=0.25, rel=None
        ),
        "alpha_vac": _local_bounds(
            float(film.alpha_vac.value), lo=0.7, hi=pi2, abs_pad=0.20, rel=None
        ),
    }
    opened: list[str] = []
    for name in BOOKENDED_OPEN:
        _setp(getattr(film, name), vary=True, bounds=bounds[name])
        opened.append(name)

    objective.refresh_parameters()
    log(
        f"bookended profile_contract: free={opened}; "
        f"frozen_instruments={len(frozen_instr)}; "
        f"oxide=({ox_thick:.3g},{ox_rough:.3g}) fixed; "
        f"tau_vac={PROFILE_TAU_VAC_BOUNDS}; "
        f"sigma_surf={PROFILE_SURFACE_ROUGH_BOUNDS}; "
        f"rho_vac={PROFILE_DENSITY_VAC_BOUNDS}; "
        f"rho_si={PROFILE_DENSITY_SI_BOUNDS}; "
        f"centroid={OPEN_CENTROID_CSV.name}"
    )
    return opened


def _widen_instrument_bounds(objective) -> list[str]:
    """Relax scale/theta boxes that sit on polish ceilings so NS can move."""
    opened: list[str] = []
    for p in objective.varying_parameters():
        name = str(p.name)
        val = float(p.value)
        lb, ub = float(p.bounds.lb), float(p.bounds.ub)
        if name.startswith("scale_"):
            # Keep a physical floor; open the ceiling past piled MLEs.
            new_lo = max(0.4, min(lb, 0.6 * val))
            new_hi = max(ub, 2.5, 1.35 * val)
            if new_hi > ub + 1e-9 or new_lo < lb - 1e-9:
                _setp(p, vary=True, bounds=(new_lo, new_hi))
                opened.append(name)
        elif name.startswith("theta_offset_"):
            new_lo = min(lb, -1.0)
            new_hi = max(ub, 1.0)
            if new_hi > ub + 1e-9 or new_lo < lb - 1e-9:
                _setp(p, vary=True, bounds=(new_lo, new_hi))
                opened.append(name)
    return opened


def unlock_bookended(objective) -> list[str]:
    """Free film + oxide near the polished MLE; widen ``tau_vac`` and instruments.

    Orientations stay local: bulk near edge-on, substrate-side near face-on.
    Wall hits on those locals are acceptable; wide face/edge swaps are not.
    """
    film = objective.model.structure[1].profile
    oxide = objective.model.structure[2]
    pi2 = float(np.pi / 2)
    bounds = {
        "total_thick": _local_bounds(
            float(film.total_thick.value), lo=160.0, hi=220.0, rel=0.10
        ),
        "surface_roughness": _local_bounds(
            float(film.surface_roughness.value), lo=2.0, hi=16.0, abs_pad=3.5, rel=None
        ),
        "density_bulk": _local_bounds(
            float(film.density_bulk.value), lo=1.0, hi=2.3, rel=0.15
        ),
        "density_si": _local_bounds(
            float(film.density_si.value), lo=0.40, hi=1.5, rel=0.18
        ),
        "density_vac": _local_bounds(
            float(film.density_vac.value), lo=1.1, hi=2.6, rel=0.15
        ),
        "tau_si": TAU_SI_BOUNDS,
        "tau_vac": TAU_VAC_BOUNDS,
        "alpha_bulk": _local_bounds(
            float(film.alpha_bulk.value), lo=0.85, hi=pi2, abs_pad=0.25, rel=None
        ),
        "alpha_si": _local_bounds(
            float(film.alpha_si.value), lo=0.0, hi=0.95, abs_pad=0.25, rel=None
        ),
        "alpha_vac": _local_bounds(
            float(film.alpha_vac.value), lo=0.55, hi=pi2, abs_pad=0.25, rel=None
        ),
    }
    opened: list[str] = []
    for name in BOOKENDED_OPEN:
        _setp(getattr(film, name), vary=True, bounds=bounds[name])
        opened.append(name)
    _setp(
        oxide.thick,
        vary=True,
        bounds=_local_bounds(
            float(oxide.thick.value), lo=2.5, hi=16.0, abs_pad=3.0, rel=None
        ),
    )
    opened.append("oxide.thick")
    _setp(
        oxide.rough,
        vary=True,
        bounds=_local_bounds(
            float(oxide.rough.value), lo=0.5, hi=14.0, abs_pad=3.0, rel=None
        ),
    )
    opened.append("oxide.rough")
    oxide.sld.density.setp(vary=False)
    objective.model.structure[3].sld.density.setp(vary=False)
    objective.refresh_parameters()
    instr = _widen_instrument_bounds(objective)
    objective.refresh_parameters()
    log(
        f"bookended unlock (coherent local priors): film/oxide={opened}; "
        f"tau_vac={TAU_VAC_BOUNDS}; instruments_widened={instr}"
    )
    return opened + instr


def configure_slab_profile_contract(objective) -> list[str]:
    """Freeze shared bookended instruments + oxide; free ZnPc with open walls.

    Instrument scales/thetas are pinned to the bookended open-contract posterior
    medians (same nuisance centroid as the bookended profile contract). Oxide
    and tabulated Si/SiO2 densities stay at polish. ZnPc surface/bulk/interface
    thick, rough, density, and rotation vary under expanded boxes that clear the
    walls seen in the first slab profile run (surface thick/rough, bulk rough /
    rotation floor, interface thick/rough/rotation).
    """
    # Match the dynesty likelihood before the film L-BFGS seed.
    objective.model.device = "cpu"
    objective.model.parallel = False
    objective.transform = TRANSFORM
    objective.anisotropy_weight = ANISOTROPY_WEIGHT
    if hasattr(objective, "normalization"):
        objective.normalization = NORMALIZATION
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()

    centroid = _load_open_contract_medians()
    instr_vals = {
        k: float(v)
        for k, v in centroid.items()
        if str(k).startswith(("scale_", "theta_offset_"))
    }
    frozen_instr = _freeze_instruments_at(objective, instr_vals)
    structure = objective.model.structure
    pi2 = float(np.pi / 2)
    opened: list[str] = []

    for i, comp in enumerate(structure):
        name = str(getattr(comp, "name", i)).lower()
        if name in {"vacuum", "substrate"} or name == "si":
            dens = getattr(getattr(comp, "sld", None), "density", None)
            if dens is not None:
                dens.setp(vary=False)
            continue

        is_oxide = "oxide" in name or "sio" in name
        is_bulk = "bulk" in name
        is_interface = "interface" in name
        is_surface = "surface" in name

        if is_oxide:
            for attr in ("thick", "rough"):
                p = getattr(comp, attr, None)
                if p is None:
                    continue
                val = float(p.value)
                _setp(
                    p,
                    value=val,
                    vary=False,
                    bounds=(max(0.1, val - 1.0), val + 1.0),
                )
            dens = getattr(getattr(comp, "sld", None), "density", None)
            if dens is not None:
                dens.setp(vary=False)
            continue

        for attr in ("thick", "rough"):
            p = getattr(comp, attr, None)
            if p is None or not hasattr(p, "setp"):
                continue
            val = float(p.value)
            if attr == "thick":
                if is_bulk:
                    b = (140.0, 210.0)
                else:
                    # Surface/interface piled at ~polish+4 Å in the first run.
                    b = (4.0, 30.0)
            else:
                if is_bulk:
                    b = (0.5, 25.0)
                elif is_interface:
                    b = (0.3, 15.0)
                else:
                    b = (0.5, 20.0)
            _setp(p, vary=True, bounds=b)
            opened.append(f"{getattr(comp, 'name', i)}.{attr}")

        sld = getattr(comp, "sld", None)
        if sld is None:
            continue
        if hasattr(sld, "density") and getattr(sld.density, "constraint", None) is None:
            if is_surface:
                b = (1.0, 5.0)
            elif is_bulk:
                b = (1.0, 2.5)
            elif is_interface:
                b = (0.5, 3.5)
            else:
                b = (1.0, 3.0)
            _setp(sld.density, vary=True, bounds=b)
            opened.append(f"{getattr(comp, 'name', i)}.density")
        if (
            hasattr(sld, "rotation")
            and getattr(sld.rotation, "constraint", None) is None
        ):
            if is_bulk:
                # Old run glued to the lower edge-on floor (~0.86).
                b = (0.5, pi2)
            elif is_interface:
                b = (0.0, pi2)
            else:
                b = (0.3, pi2)
            _setp(sld.rotation, vary=True, bounds=b)
            opened.append(f"{getattr(comp, 'name', i)}.rotation")

    objective.refresh_parameters()

    # Seed film under shared instruments so NS does not start at nll~65k.
    x0 = np.array([float(p.value) for p in objective.varying_parameters()], dtype=float)
    bounds = [
        (float(p.bounds.lb), float(p.bounds.ub)) for p in objective.varying_parameters()
    ]
    nll0 = float(objective.nll(x0))
    res = minimize(
        lambda x: float(objective.nll(x)),
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 80, "ftol": 1e-8},
    )
    if res.success or float(res.fun) < nll0:
        objective.setp(res.x)
        objective.refresh_parameters()
    nll1 = float(objective.nll())
    log(
        f"dft_slab profile_contract: free={opened}; "
        f"frozen_instruments={len(frozen_instr)} (bookended open-contract centroid); "
        f"oxide/Si fixed at polish; film walls opened; "
        f"film seed nll {nll0:.1f} -> {nll1:.1f} (L-BFGS nit={getattr(res, 'nit', '?')})"
    )
    return opened


def unlock_slab(objective) -> list[str]:
    """Free ZnPc + oxide near the polished MLE; keep bulk edge-on / interface face-on."""
    opened: list[str] = []
    structure = objective.model.structure
    pi2 = float(np.pi / 2)
    for i, comp in enumerate(structure):
        name = str(getattr(comp, "name", i)).lower()
        if name in {"vacuum", "substrate"} or name == "si":
            continue
        is_oxide = "oxide" in name or "sio" in name
        is_bulk = "bulk" in name
        is_interface = "interface" in name
        for attr in ("thick", "rough"):
            p = getattr(comp, attr, None)
            if p is None or not hasattr(p, "setp"):
                continue
            val = float(p.value)
            if attr == "thick":
                if is_oxide:
                    b = _local_bounds(val, lo=3.0, hi=15.0, rel=0.25)
                elif is_bulk:
                    b = _local_bounds(val, lo=140.0, hi=210.0, rel=0.12)
                else:
                    b = _local_bounds(val, lo=5.0, hi=25.0, rel=0.25)
            else:
                b = _local_bounds(val, lo=0.5, hi=18.0, abs_pad=2.5, rel=None)
            _setp(p, vary=True, bounds=b)
            opened.append(f"{getattr(comp, 'name', i)}.{attr}")
        sld = getattr(comp, "sld", None)
        if sld is None:
            continue
        if hasattr(sld, "density") and not is_oxide:
            p = sld.density
            if getattr(p, "constraint", None) is None:
                if is_interface:
                    b = _local_bounds(float(p.value), lo=0.6, hi=1.9, rel=0.15)
                elif is_bulk:
                    b = _local_bounds(float(p.value), lo=1.1, hi=2.1, rel=0.12)
                else:
                    b = _local_bounds(float(p.value), lo=1.0, hi=2.3, rel=0.15)
                _setp(p, vary=True, bounds=b)
                opened.append(f"{getattr(comp, 'name', i)}.density")
        if hasattr(sld, "rotation"):
            p = sld.rotation
            if getattr(p, "constraint", None) is None:
                val = float(p.value)
                if is_bulk:
                    b = _local_bounds(val, lo=0.7, hi=pi2, abs_pad=0.20, rel=None)
                elif is_interface:
                    b = _local_bounds(val, lo=0.0, hi=0.9, abs_pad=0.20, rel=None)
                else:
                    b = _local_bounds(val, lo=0.2, hi=pi2, abs_pad=0.20, rel=None)
                _setp(p, vary=True, bounds=b)
                opened.append(f"{getattr(comp, 'name', i)}.rotation")
    objective.refresh_parameters()
    log(f"dft_slab unlock (local MLE priors): opened {len(opened)} geometry terms")
    return opened


def boundary_diagnostics(
    names: list[str],
    lo: np.ndarray,
    hi: np.ndarray,
    samples: np.ndarray,
    *,
    eps_frac: float = BOUNDARY_EPS_FRAC,
    mass_warn: float = BOUNDARY_MASS_WARN,
) -> dict:
    """Flag parameters whose posterior mass piles against a prior wall."""
    n = samples.shape[0]
    rows = []
    hits = []
    for i, name in enumerate(names):
        a, b = float(lo[i]), float(hi[i])
        span = max(b - a, 1e-12)
        xs = samples[:, i]
        near_lo = float(np.mean(xs <= a + eps_frac * span))
        near_hi = float(np.mean(xs >= b - eps_frac * span))
        med = float(np.median(xs))
        q16 = float(np.percentile(xs, 16))
        q84 = float(np.percentile(xs, 84))
        flag_lo = near_lo >= mass_warn or med <= a + eps_frac * span
        flag_hi = near_hi >= mass_warn or med >= b - eps_frac * span
        row = {
            "name": name,
            "lo": a,
            "hi": b,
            "median": med,
            "q16": q16,
            "q84": q84,
            "frac_near_lo": near_lo,
            "frac_near_hi": near_hi,
            "hit_lo": bool(flag_lo),
            "hit_hi": bool(flag_hi),
        }
        rows.append(row)
        if flag_lo or flag_hi:
            hits.append(row)
    return {
        "n_samples": int(n),
        "eps_frac": eps_frac,
        "mass_warn": mass_warn,
        "n_hits": len(hits),
        "hits": hits,
        "parameters": rows,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def posterior_summary(results, names: list[str], x0: np.ndarray) -> dict:
    """Weighted posterior moments from dynesty samples (fit-quality view)."""
    samples = np.asarray(results.samples, dtype=float)
    logwt = np.asarray(results.logwt, dtype=float)
    logz = float(results.logz[-1])
    wt = np.exp(logwt - logz)
    wt = wt / wt.sum()
    mean = np.average(samples, axis=0, weights=wt)
    # ESS from Kish formula
    ess = float(1.0 / np.sum(wt**2))
    qlo = np.zeros(samples.shape[1])
    qmed = np.zeros(samples.shape[1])
    qhi = np.zeros(samples.shape[1])
    for i in range(samples.shape[1]):
        order = np.argsort(samples[:, i])
        cdf = np.cumsum(wt[order])
        cdf /= cdf[-1]
        xs = samples[order, i]
        qlo[i] = float(np.interp(0.16, cdf, xs))
        qmed[i] = float(np.interp(0.50, cdf, xs))
        qhi[i] = float(np.interp(0.84, cdf, xs))
    params = []
    for i, name in enumerate(names):
        params.append(
            {
                "name": name,
                "mle": float(x0[i]),
                "polish_x0": float(x0[i]),
                "mean": float(mean[i]),
                "median": float(qmed[i]),
                "q16": float(qlo[i]),
                "q84": float(qhi[i]),
            }
        )
    i_map = int(np.argmax(np.asarray(results.logl, dtype=float)))
    map_vals = {names[i]: float(samples[i_map, i]) for i in range(len(names))}
    return {
        "ess": ess,
        "n_samples": int(samples.shape[0]),
        "logz": logz,
        "logzerr": float(results.logzerr[-1]),
        "logl_max": float(np.max(results.logl)),
        "map_index": i_map,
        "map": map_vals,
        "params": params,
    }


def make_sampler(
    loglike,
    prior_transform,
    ndim: int,
    *,
    nlive: int,
    walks: int,
):
    return DynamicNestedSampler(
        loglike,
        prior_transform,
        ndim,
        nlive=nlive,
        sample=SAMPLE,
        bound=BOUND,
        walks=walks,
        rstate=np.random.default_rng(SEED),
    )


def run_probe(
    label: str,
    objective,
    *,
    probe_s: float,
    nlive: int,
    walks: int,
    out_dir: Path | None = None,
) -> dict:
    lo, hi, names, _x0, _logl0 = configure_objective(objective, label=label)
    ms = timed_logl(objective) * 1000.0
    log(f"{label}: GPU logl ~ {ms:.2f} ms/call  nlive={nlive}  walks={walks}")
    log(f"{label}: bounds preview:")
    for name, a, b, x in zip(names, lo, hi, _x0, strict=True):
        if (
            name in ("tau_vac", "tau_si", "surface_roughness", "alpha_vac")
            or "tau" in name
        ):
            log(f"  {name}: value={x:.5g}  bounds=({a:.5g},{b:.5g})")

    prior_transform = make_prior_transform(lo, hi)
    loglike = make_loglike(objective)
    ndim = len(names)
    sampler = make_sampler(loglike, prior_transform, ndim, nlive=nlive, walks=walks)

    t0 = time.perf_counter()
    ncall = 0
    nit = 0
    status = {
        "mode": "probe",
        "label": label,
        "ndim": ndim,
        "names": names,
        "nlive": nlive,
        "walks": walks,
        "sample": SAMPLE,
        "logl_ms": ms,
        "probe_s": probe_s,
        "ncall": 0,
        "nit": 0,
        "logz": None,
        "logzerr": None,
        "loglstar": None,
        "stopped": "budget",
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "x0": _x0.tolist(),
    }

    def _update(results) -> None:
        nonlocal ncall, nit
        ncall += int(results.nc)
        nit = int(results.worst_it) + 1
        status["ncall"] = ncall
        status["nit"] = nit
        status["loglstar"] = float(results.loglstar)
        if hasattr(results, "logz"):
            status["logz"] = float(results.logz)
            status["logzerr"] = float(np.sqrt(max(results.logzvar, 0.0)))
        else:
            try:
                r = sampler.results
                status["logz"] = float(r.logz[-1])
                status["logzerr"] = float(r.logzerr[-1])
            except Exception:
                pass

    for results in sampler.sample_initial(nlive=nlive, dlogz=1.0e-3):
        _update(results)
        elapsed = time.perf_counter() - t0
        if elapsed >= probe_s:
            status["wall_s"] = elapsed
            status["stopped"] = "probe_during_initial"
            break
        if nit % 50 == 0:
            log(
                f"  {label} initial nit={nit} ncall={ncall} "
                f"logZ={status['logz']:.3f}±{status['logzerr']:.3f} "
                f"wall={elapsed:.1f}s"
            )
    else:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= probe_s:
                status["wall_s"] = elapsed
                status["stopped"] = "probe_after_initial"
                break
            for results in sampler.sample_batch():
                _update(results)
                elapsed = time.perf_counter() - t0
                if elapsed >= probe_s:
                    status["wall_s"] = elapsed
                    status["stopped"] = "probe_during_batch"
                    break
            else:
                continue
            break

    status["wall_s"] = float(status.get("wall_s", time.perf_counter() - t0))
    status["ncall_per_s"] = status["ncall"] / max(status["wall_s"], 1e-9)
    log(
        f"{label}: wall={status['wall_s']:.1f}s  ncall={status['ncall']}  "
        f"({status['ncall_per_s']:.1f}/s)  nit={status['nit']}  "
        f"logZ~{status['logz']}±{status['logzerr']}  "
        f"stopped={status['stopped']}"
    )

    out = out_dir if out_dir is not None else (OUT_DIR / label)
    out.mkdir(parents=True, exist_ok=True)
    bounds_report = None
    try:
        results = sampler.results
        samples = np.asarray(results.samples, dtype=float)
        bounds_report = boundary_diagnostics(names, lo, hi, samples)
        status["boundary_n_hits"] = bounds_report["n_hits"]
        status["boundary_hits"] = [
            {
                "name": h["name"],
                "hit_lo": h["hit_lo"],
                "hit_hi": h["hit_hi"],
                "median": h["median"],
                "lo": h["lo"],
                "hi": h["hi"],
                "frac_near_lo": h["frac_near_lo"],
                "frac_near_hi": h["frac_near_hi"],
            }
            for h in bounds_report["hits"]
        ]
        if bounds_report["n_hits"]:
            log(f"{label}: BOUNDARY HITS ({bounds_report['n_hits']}):")
            for h in bounds_report["hits"]:
                side = []
                if h["hit_lo"]:
                    side.append("lo")
                if h["hit_hi"]:
                    side.append("hi")
                log(
                    f"  {h['name']}: median={h['median']:.5g} "
                    f"bounds=({h['lo']:.5g},{h['hi']:.5g}) "
                    f"hits={'+'.join(side)}  "
                    f"frac_lo={h['frac_near_lo']:.3f} frac_hi={h['frac_near_hi']:.3f}"
                )
        else:
            log(f"{label}: no boundary pile-up at {BOUNDARY_EPS_FRAC:.0%} of span")
        with (out / "results.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "results": results,
                    "status": status,
                    "names": names,
                    "bounds_report": bounds_report,
                },
                fh,
            )
        _atomic_json(out / "bounds_report.json", bounds_report)
    except Exception as exc:
        log(f"{label}: could not snapshot results ({exc})")
    _atomic_json(out / "probe_status.json", status)
    return status


def run_full(
    label: str,
    objective,
    *,
    nlive: int,
    walks: int,
    dlogz_init: float,
    n_effective: int,
    out_dir: Path | None = None,
) -> dict:
    lo, hi, names, x0, logl0 = configure_objective(objective, label=label)
    ms = timed_logl(objective) * 1000.0
    ndim = len(names)
    log(
        f"{label}: GPU logl ~ {ms:.2f} ms/call  "
        f"nlive={nlive}  walks={walks}  dlogz_init={dlogz_init}  "
        f"n_effective={n_effective}"
    )

    out = (out_dir if out_dir is not None else FULL_DIR) / label
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pkl"
    status_path = out / "status.json"

    prior_transform = make_prior_transform(lo, hi)
    loglike = make_loglike(objective)
    sampler = make_sampler(loglike, prior_transform, ndim, nlive=nlive, walks=walks)

    status: dict = {
        "mode": "full",
        "label": label,
        "ndim": ndim,
        "names": names,
        "nlive": nlive,
        "walks": walks,
        "sample": SAMPLE,
        "bound": BOUND,
        "dlogz_init": dlogz_init,
        "n_effective": n_effective,
        "logl_ms": ms,
        "logl0_mle": logl0,
        "ncall": 0,
        "nit": 0,
        "logz": None,
        "logzerr": None,
        "phase": "starting",
        "wall_s": 0.0,
        "ncall_per_s": 0.0,
        "stopped": None,
    }
    _atomic_json(status_path, status)

    t0 = time.perf_counter()
    last_write = 0.0

    def print_func(results, niter, ncall, nbatch=None, dlogz=None, **_kw):
        nonlocal last_write
        elapsed = time.perf_counter() - t0
        status["wall_s"] = elapsed
        status["ncall"] = int(ncall) if ncall is not None else status["ncall"]
        status["nit"] = int(niter) if niter is not None else status["nit"]
        status["ncall_per_s"] = status["ncall"] / max(elapsed, 1e-9)
        status["nbatch"] = None if nbatch is None else int(nbatch)
        status["dlogz"] = None if dlogz is None else float(dlogz)
        status["phase"] = "initial" if (nbatch is None or int(nbatch) == 0) else "batch"
        try:
            if hasattr(results, "logz") and np.ndim(results.logz) == 0:
                status["logz"] = float(results.logz)
                if hasattr(results, "logzvar"):
                    status["logzerr"] = float(np.sqrt(max(results.logzvar, 0.0)))
            elif hasattr(results, "logz"):
                status["logz"] = float(np.asarray(results.logz)[-1])
                if hasattr(results, "logzerr"):
                    status["logzerr"] = float(np.asarray(results.logzerr)[-1])
        except Exception:
            pass
        if (
            elapsed - last_write >= STATUS_WRITE_S
            or status["nit"] % STATUS_WRITE_NIT == 0
        ):
            last_write = elapsed
            _atomic_json(status_path, status)
            log(
                f"  {label} {status['phase']} nit={status['nit']} "
                f"ncall={status['ncall']} "
                f"logZ={status['logz']}±{status['logzerr']} "
                f"wall={elapsed / 60.0:.1f}m  "
                f"({status['ncall_per_s']:.0f}/s)"
            )

    resume = checkpoint.exists()
    if resume:
        log(f"{label}: resuming from {checkpoint}")

    sampler.run_nested(
        nlive_init=nlive,
        dlogz_init=dlogz_init,
        nlive_batch=nlive,
        n_effective=n_effective,
        print_progress=True,
        print_func=print_func,
        checkpoint_file=str(checkpoint),
        checkpoint_every=120,
        resume=resume,
    )

    elapsed = time.perf_counter() - t0
    results = sampler.results
    status["wall_s"] = elapsed
    if hasattr(results, "ncall"):
        status["ncall"] = int(np.sum(results.ncall))
    status["nit"] = int(results.niter)
    status["logz"] = float(results.logz[-1])
    status["logzerr"] = float(results.logzerr[-1])
    status["logl_max"] = float(np.max(results.logl))
    status["ncall_per_s"] = status["ncall"] / max(elapsed, 1e-9)
    status["phase"] = "done"
    status["stopped"] = "run_nested_complete"
    status["n_effective_est"] = float(getattr(results, "n_effective", np.nan))
    _atomic_json(status_path, status)

    post = posterior_summary(results, names, x0)
    _atomic_json(out / "posterior_summary.json", post)

    with (out / "results.pkl").open("wb") as fh:
        pickle.dump(
            {
                "results": results,
                "status": status,
                "names": names,
                "x0_mle": x0,
                "posterior": post,
            },
            fh,
        )

    log(
        f"{label}: DONE wall={elapsed / 3600.0:.2f}h  ncall={status['ncall']}  "
        f"({status['ncall_per_s']:.0f}/s)  logZ={status['logz']:.4f}±"
        f"{status['logzerr']:.4f}  ESS~{post['ess']:.0f}  "
        f"logl_max={post['logl_max']:.3f} (MLE logl0={logl0:.3f})"
    )
    return status


def bayes_factor_summary(statuses: dict[str, dict]) -> dict | None:
    if "bookended" not in statuses or "dft_slab" not in statuses:
        return None
    b = statuses["bookended"]
    s = statuses["dft_slab"]
    if b.get("logz") is None or s.get("logz") is None:
        return None
    dln = float(b["logz"]) - float(s["logz"])
    dln_err = float(np.sqrt(float(b["logzerr"]) ** 2 + float(s["logzerr"]) ** 2))
    return {
        "delta_logz": dln,
        "delta_logz_err": dln_err,
        "log10_bf": dln / np.log(10.0),
        "two_ln_bf": 2.0 * dln,
        "bf": float(np.exp(dln)),
        "bookended_logz": float(b["logz"]),
        "dft_slab_logz": float(s["logz"]),
    }


def resolve_contract(args: argparse.Namespace) -> str:
    if args.contract is not None:
        return str(args.contract)
    return "open" if args.unlock else "polish"


def contract_out_dir(contract: str, *, full: bool) -> Path:
    if full:
        return {
            "polish": FULL_POLISH_DIR,
            "open": FULL_OPEN_DIR,
            "profile": FULL_PROFILE_DIR,
        }[contract]
    return (
        OUT_DIR
        / {
            "polish": "probe_polish_contract",
            "open": "probe_open_contract",
            "profile": "probe_profile_contract",
        }[contract]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--probe-minutes",
        type=float,
        default=None,
        help="Wall-clock budget per model for a smoke probe",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Production evidence + posterior run (no wall-clock cut)",
    )
    p.add_argument(
        "--nlive",
        type=int,
        default=0,
        help="Override live points (0 = max(1000, 50*ndim))",
    )
    p.add_argument(
        "--walks",
        type=int,
        default=0,
        help="Override rwalk steps (0 = max(25, 3*ndim))",
    )
    p.add_argument("--dlogz-init", type=float, default=DLOGZ_INIT)
    p.add_argument("--n-effective", type=int, default=N_EFFECTIVE)
    p.add_argument(
        "--models",
        nargs="+",
        default=["bookended", "dft_slab"],
        choices=["bookended", "dft_slab"],
    )
    p.add_argument(
        "--contract",
        choices=("polish", "open", "profile"),
        default=None,
        help=(
            "Bookended free-set contract: polish (locked), open (film+oxide+wide "
            "instruments), profile (instruments+oxide frozen, film profile free). "
            "Default: polish, or open when --unlock."
        ),
    )
    p.add_argument(
        "--unlock",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated alias for --contract open",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Run each --models entry as a concurrent subprocess",
    )
    return p.parse_args(argv)


def _spawn_parallel(args: argparse.Namespace) -> int:
    import subprocess

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    cwd = script.parents[2]
    full_mode = bool(args.full)
    contract = resolve_contract(args)
    log_root = contract_out_dir(contract, full=full_mode)
    log_root.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, object]] = []
    for label in args.models:
        log_path = log_root / f"{label}.log"
        cmd = [
            sys.executable,
            str(script),
            "--models",
            label,
            "--contract",
            contract,
        ]
        if full_mode:
            cmd.append("--full")
        else:
            cmd.extend(["--probe-minutes", str(args.probe_minutes)])
        if args.nlive > 0:
            cmd.extend(["--nlive", str(args.nlive)])
        if args.walks > 0:
            cmd.extend(["--walks", str(args.walks)])
        if full_mode:
            cmd.extend(
                [
                    "--dlogz-init",
                    str(args.dlogz_init),
                    "--n-effective",
                    str(args.n_effective),
                ]
            )
        log(f"parallel spawn: {' '.join(cmd)} -> {log_path}")
        fh = log_path.open("w")
        procs.append(
            (
                label,
                subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ),
            )
        )
    rc = 0
    for label, proc in procs:
        code = proc.wait()
        log(f"parallel {label}: exit={code}")
        rc = rc or int(code)
    if full_mode:
        summary: dict = {
            "mode": "full",
            "parallel": True,
            "contract": contract,
            "unlock": contract == "open",
            "models": {},
            "bayes_factor": None,
        }
        statuses: dict[str, dict] = {}
        for label in args.models:
            st_path = log_root / label / "status.json"
            if st_path.exists():
                st = json.loads(st_path.read_text())
                statuses[label] = st
                summary["models"][label] = {
                    k: st[k]
                    for k in (
                        "ndim",
                        "nlive",
                        "walks",
                        "wall_s",
                        "ncall",
                        "ncall_per_s",
                        "nit",
                        "logz",
                        "logzerr",
                        "logl_max",
                        "stopped",
                    )
                    if k in st
                }
        summary["bayes_factor"] = bayes_factor_summary(statuses)
        _atomic_json(log_root / "summary.json", summary)
        log(f"wrote {log_root / 'summary.json'}")
        if summary["bayes_factor"] is not None:
            bf = summary["bayes_factor"]
            log(
                f"Bayes factor bookended/slab: ΔlogZ={bf['delta_logz']:.4f}±"
                f"{bf['delta_logz_err']:.4f}  log10(BF)={bf['log10_bf']:.3f}"
            )
        return rc

    summary = {
        "probe_minutes": args.probe_minutes,
        "parallel": True,
        "contract": contract,
        "models": {},
    }
    for label in args.models:
        st_path = log_root / label / "probe_status.json"
        if st_path.exists():
            st = json.loads(st_path.read_text())
            summary["models"][label] = {
                k: st[k]
                for k in (
                    "ndim",
                    "nlive",
                    "walks",
                    "wall_s",
                    "ncall",
                    "ncall_per_s",
                    "logz",
                    "logzerr",
                    "stopped",
                    "boundary_n_hits",
                    "boundary_hits",
                )
                if k in st
            }
    _atomic_json(log_root / "probe_summary.json", summary)
    log(f"wrote {log_root / 'probe_summary.json'}")
    return rc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.full and args.probe_minutes is None:
        args.full = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.parallel and len(args.models) > 1:
        return _spawn_parallel(args)

    contract = resolve_contract(args)
    loaders = {
        "bookended": lambda: load_bookended(contract=contract),
        "dft_slab": lambda: load_slab(contract=contract),
    }
    nlive_override = None if args.nlive <= 0 else args.nlive
    walks_override = None if args.walks <= 0 else args.walks

    if args.full:
        full_root = contract_out_dir(contract, full=True)
        full_root.mkdir(parents=True, exist_ok=True)
        log(f"OUT_DIR={full_root}")
        log(
            f"FULL run  sample={SAMPLE}  bound={BOUND}  contract={contract}  "
            f"aniso={ANISOTROPY_WEIGHT}  norm={NORMALIZATION}  "
            f"dlogz_init={args.dlogz_init}  n_effective={args.n_effective}"
        )
        summary: dict = {
            "mode": "full",
            "sample": SAMPLE,
            "bound": BOUND,
            "contract": contract,
            "unlock": contract == "open",
            "dlogz_init": args.dlogz_init,
            "n_effective": args.n_effective,
            "nlive_per_dim": NLIVE_PER_DIM,
            "walks_per_dim": WALKS_PER_DIM,
            "models": {},
            "bayes_factor": None,
        }
        statuses: dict[str, dict] = {}
        for label in args.models:
            log(f"=== {label} (full) ===")
            objective = loaders[label]()
            n_peek = len(objective.varying_parameters())
            nlive = nlive_for_ndim(n_peek, nlive_override)
            walks = walks_for_ndim(n_peek, walks_override)
            status = run_full(
                label,
                objective,
                nlive=nlive,
                walks=walks,
                dlogz_init=float(args.dlogz_init),
                n_effective=int(args.n_effective),
                out_dir=full_root,
            )
            statuses[label] = status
            summary["models"][label] = {
                k: status[k]
                for k in (
                    "ndim",
                    "nlive",
                    "walks",
                    "logl_ms",
                    "wall_s",
                    "ncall",
                    "ncall_per_s",
                    "nit",
                    "logz",
                    "logzerr",
                    "logl_max",
                    "stopped",
                )
                if k in status
            }
            _atomic_json(full_root / "summary.json", summary)

        summary["bayes_factor"] = bayes_factor_summary(statuses)
        _atomic_json(full_root / "summary.json", summary)
        if summary["bayes_factor"] is not None:
            bf = summary["bayes_factor"]
            log(
                f"Bayes factor bookended/slab: ΔlogZ={bf['delta_logz']:.4f}±"
                f"{bf['delta_logz_err']:.4f}  log10(BF)={bf['log10_bf']:.3f}"
            )
        log(f"wrote {full_root / 'summary.json'}")
        return 0

    probe_s = float(args.probe_minutes) * 60.0
    probe_root = contract_out_dir(contract, full=False)
    probe_root.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={probe_root}")
    log(
        f"probe={args.probe_minutes:g} min/model  sample={SAMPLE}  "
        f"contract={contract}  aniso={ANISOTROPY_WEIGHT}"
    )
    summary = {
        "probe_minutes": args.probe_minutes,
        "contract": contract,
        "unlock": contract == "open",
        "models": {},
    }
    for label in args.models:
        log(f"=== {label} ===")
        objective = loaders[label]()
        n_peek = len(objective.varying_parameters())
        nlive = nlive_for_ndim(n_peek, nlive_override)
        walks = walks_for_ndim(n_peek, walks_override)
        status = run_probe(
            label,
            objective,
            probe_s=probe_s,
            nlive=nlive,
            walks=walks,
            out_dir=probe_root / label,
        )
        summary["models"][label] = {
            k: status[k]
            for k in (
                "ndim",
                "nlive",
                "walks",
                "logl_ms",
                "wall_s",
                "ncall",
                "ncall_per_s",
                "nit",
                "logz",
                "logzerr",
                "stopped",
                "boundary_n_hits",
                "boundary_hits",
            )
            if k in status
        }
    _atomic_json(probe_root / "probe_summary.json", summary)
    log(f"wrote {probe_root / 'probe_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

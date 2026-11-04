"""Multi-energy DFT-slab helpers under ``logYX4`` on ``E_RES``.

Warm-starts from the as-fit slab ``refit-refloxide.pkl``, locks Si/SiO2 to
the bookended polish (oxide thick/rough, tabulated densities, substrate
rough 1.2 A, shared ``energy_offset``), locks per-energy ``bkg`` to the
bookended polish values, and frees ZnPc film + scales/thetas.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from refloxide.data import ReflectDataset
from refloxide.model import ReflectModel
from refloxide.objective import Objective, gpu_batch
from refloxide.pxr.plugin.dft_fit import (
    MIN_INCIDENT_THETA_DEG,
    THETA_OFFSET_MARGIN_DEG,
    min_data_grazing_angle_deg,
)
from refnx.analysis import CurveFitter
from utils import models_fitting_results, models_root
from utils.models import configure_refloxide_fitting
from utils.refloxide_view import load_refloxide_fit

configure_refloxide_fitting()

E_RES = (283.7, 284.2, 285.1, 287.0, 289.0)
Q_MIN = 1e-4
RHO_SIO2 = 2.2
RHO_SI = 2.33
SUBSTRATE_ROUGH = 1.2
INSTRUMENT_FIELDS = (
    "scale_s",
    "scale_p",
    "theta_offset_s",
    "theta_offset_p",
    "bkg",
)
FILM_SLABS = ("ZnPc_surface", "ZnPc_bulk", "ZnPc_interface")
WARM_PATH = models_root / "xrr/znpc/refloxide/refit-refloxide.pkl"
BOOKENDED_POLISH = (
    models_fitting_results / "bookended_multienergy_logY_polish" / "fit.pkl"
)
OUT_DIR = models_fitting_results / "dft_slab_eres_logyx4"

DE_KW = {
    "popsize": 15,
    "maxiter": 120,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "seed": 1,
    "polish": False,
    "verbose": False,
}

_LN10 = float(np.log(10.0))


class LogYX4:
    """``log10(R q^4)`` transform with first-order error propagation."""

    form = "logYX4"

    def __repr__(self) -> str:
        return "LogYX4()"

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_err: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        yx4 = np.maximum(y * np.power(x, 4), 1e-30)
        yt = np.log10(yx4)
        if y_err is None:
            return yt, None
        et = np.asarray(y_err, dtype=float) * np.power(x, 4)
        et = np.maximum(et, 0.0) / (yx4 * _LN10)
        return yt, et


TRANSFORM = LogYX4()


def log(msg: str) -> None:
    print(msg, flush=True)


def load_bookended_locks() -> tuple[dict, float, dict[float, float]]:
    """Load oxide geometry, energy offset, and per-energy ``bkg`` from bookended polish."""
    with BOOKENDED_POLISH.open("rb") as fh:
        bundle = pickle.load(fh)
    oxide = dict(bundle["summary"]["oxide"])
    eo = float(bundle["obj_map"][283.7].model.energy_offset.value)
    instruments = bundle["summary"]["instruments"]
    bkg = {float(e): float(vals["bkg"]) for e, vals in instruments.items()}
    return oxide, eo, bkg


def polarization_labels_swapped(
    energy: np.ndarray,
    pol: np.ndarray,
    r: np.ndarray,
    *,
    probe_ev: float = 283.7,
) -> bool:
    """Detect s/p label swap via the 283.7 eV Brewster-like deep minimum.

    Physical p-pol carries the deep reflectivity dip near q~0.21 A^-1 at
    283.7 eV. The warm ``refit-refloxide.pkl`` dataset labels that channel
    as ``s``, so any subset that inherits those labels must be flipped.
    """
    m = np.isclose(np.asarray(energy, dtype=float), float(probe_ev))
    if not np.any(m):
        return False
    pol_arr = np.asarray(pol)
    r_arr = np.asarray(r, dtype=float)
    r_s = r_arr[m & (pol_arr == "s")]
    r_p = r_arr[m & (pol_arr == "p")]
    if r_s.size == 0 or r_p.size == 0:
        return False
    return float(np.min(r_s)) < 0.1 * float(np.min(r_p))


def flip_pol_labels(pol: np.ndarray) -> np.ndarray:
    """Return a copy of ``pol`` with every ``s``/``p`` label swapped."""
    out = np.asarray(pol, dtype=object).copy()
    out[:] = np.array(["p" if str(x) == "s" else "s" for x in out], dtype=object)
    return out


def subset_dataset(warm: Objective) -> ReflectDataset:
    """Subset warm-start rows onto ``E_RES`` with corrected s/p labels.

    The warm ``refit-refloxide.pkl`` packs swap polarization channel names
    relative to the bookended polish dataset. Flip labels when the Brewster
    deep minimum sits on rows labeled ``s`` so loss compares each channel
    to the matching model reflectivity.
    """
    ds = warm._dataset
    q = np.asarray(ds.q, dtype=float)
    energy = np.asarray(ds.energy, dtype=float)
    pol = np.asarray(ds.pol)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    mask = q >= Q_MIN
    em = np.zeros(q.shape, dtype=bool)
    for e in E_RES:
        em |= np.isclose(energy, e)
    mask &= em
    pol_sub = pol[mask]
    if polarization_labels_swapped(energy[mask], pol_sub, r[mask]):
        pol_sub = flip_pol_labels(pol_sub)
        log(
            "subset_dataset: flipped s/p labels "
            "(warm Brewster deep-min was on 's'; physical channel is 'p')"
        )
    return ReflectDataset(
        q=q[mask],
        energy=energy[mask],
        pol=pol_sub,
        r=r[mask],
        r_err=r_err[mask],
    )


def freeze_all(structure) -> None:
    for component in structure.components:
        for par in (component.thick, component.rough):
            if par.constraint is None:
                par.vary = False
        for par in component.sld.parameters.flattened():
            if par.constraint is None:
                par.vary = False


def apply_locks(structure, oxide_mle: dict) -> tuple:
    vacuum, surface, bulk, interface, oxide, substrate = structure.components
    freeze_all(structure)

    oxide.thick.setp(value=float(oxide_mle["thick"]), vary=False)
    oxide.rough.setp(value=float(oxide_mle["rough"]), vary=False)
    oxide.sld.density.setp(value=RHO_SIO2, vary=False)
    substrate.sld.density.setp(value=RHO_SI, vary=False)
    substrate.rough.setp(value=SUBSTRATE_ROUGH, vary=False)

    for slab in (surface, bulk, interface):
        for par in (slab.thick, slab.rough, slab.sld.density, slab.sld.rotation):
            par.vary = True
            lb, ub = float(par.bounds.lb), float(par.bounds.ub)
            if not np.isfinite(lb) or not np.isfinite(ub) or ub <= lb:
                val = float(par.value)
                span = max(0.05 * abs(val), 0.05)
                par.setp(bounds=(val - span, val + span), vary=True)
            elif float(par.value) >= ub - 1e-12 or float(par.value) <= lb + 1e-12:
                val = float(par.value)
                pad = max(0.02 * abs(ub - lb), 1e-3)
                par.setp(
                    bounds=(min(lb, val - pad), max(ub, val + pad)),
                    vary=True,
                )

    log(
        f"locked oxide={float(oxide.thick.value):.4f}/{float(oxide.rough.value):.4f}  "
        f"rho_SiO2={RHO_SIO2:g}  rho_Si={RHO_SI:g}  "
        f"substrate_rough={SUBSTRATE_ROUGH:g}"
    )
    return vacuum, surface, bulk, interface, oxide, substrate


def tighten_theta_offsets(model: ReflectModel, dataset: ReflectDataset) -> None:
    q = np.asarray(dataset.q, dtype=float)
    energy = np.asarray(dataset.energy, dtype=float)
    pol = np.asarray(dataset.pol)
    for e in E_RES:
        for pol_name, field in (("s", "theta_offset_s"), ("p", "theta_offset_p")):
            m = np.isclose(energy, e) & (pol == pol_name)
            qq = q[m]
            if qq.size == 0:
                continue
            grazing = min_data_grazing_angle_deg(qq, float(e))
            safe_lb = MIN_INCIDENT_THETA_DEG + THETA_OFFSET_MARGIN_DEG - grazing
            param = getattr(model, field).at(e)
            lb_old, ub = float(param.bounds.lb), float(param.bounds.ub)
            new_lb = max(lb_old, float(safe_lb))
            value = float(param.value or 0.0)
            kwargs: dict = {"bounds": (new_lb, ub), "vary": True}
            if value < new_lb:
                kwargs["value"] = new_lb
            param.setp(**kwargs)
            lb_f, ub_f = float(param.bounds.lb), float(param.bounds.ub)
            val_f = float(param.value or 0.0)
            if ub_f - lb_f < 1e-6:
                param.setp(bounds=(val_f - 0.05, val_f + 0.05), vary=True)
            elif val_f <= lb_f + 1e-12:
                param.setp(bounds=(lb_f, max(ub_f, lb_f + 0.05)), vary=True)
            elif val_f >= ub_f - 1e-12:
                param.setp(bounds=(min(lb_f, ub_f - 0.05), ub_f + 0.05), vary=True)
            log(
                f"  theta floor {e:g} eV {field}: "
                f"lb={float(param.bounds.lb):.4f} "
                f"(grazing={grazing:.4f} deg, value={float(param.value):.4f})"
            )


def build_objective(
    warm: Objective,
    oxide_mle: dict,
    energy_offset: float,
    *,
    bkg_lock: dict[float, float],
    transform=TRANSFORM,
    device: str = "gpu",
) -> tuple[Objective, tuple]:
    structure = warm.model.structure
    comps = apply_locks(structure, oxide_mle)
    dataset = subset_dataset(warm)

    model = ReflectModel(
        structure,
        energies=list(E_RES),
        device=device,
        energy_offset=float(energy_offset),
    )
    for par in model.corrections.parameters().flattened():
        if par.constraint is None:
            par.vary = False
    model.energy_offset.setp(value=float(energy_offset), vary=False)
    log(f"locked energy_offset={float(energy_offset):.6g} eV")

    for energy in E_RES:
        for field in INSTRUMENT_FIELDS:
            src = getattr(warm.model, field).at(energy)
            dst = getattr(model, field).at(energy)
            if field == "bkg":
                value = float(bkg_lock[float(energy)])
                dst.setp(value, vary=False)
                log(f"  bkg@{energy:g}eV locked={value:.6g} (bookended)")
                continue
            bounds = (float(src.bounds.lb), float(src.bounds.ub))
            dst.setp(float(src.value), vary=True, bounds=bounds)
    tighten_theta_offsets(model, dataset)

    objective = Objective(
        model,
        dataset,
        transform=transform,
        anisotropy_weight=0.0,
        normalization="energy",
        nc_constraint=True,
    )
    objective.refresh_parameters()
    return objective, comps


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


def per_energy_rms(objective: Objective, transform=TRANSFORM) -> list[dict]:
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e = np.asarray(ds.energy)
    rows = []
    for energy in E_RES:
        m = np.isclose(e, energy)
        yd, ed = transform(q[m], r[m], r_err[m])
        yf, _ = transform(q[m], y_fit[m])
        resid = (yd - yf) / np.maximum(ed, 1e-30)
        n = int(resid.size)
        chi2 = float(np.sum(resid**2))
        rows.append(
            {
                "energy": energy,
                "n": n,
                "chi2": chi2,
                "rms": float(np.sqrt(chi2 / max(n, 1))),
                "n_s": int(np.sum(m & (pol == "s"))),
                "n_p": int(np.sum(m & (pol == "p"))),
            }
        )
    return rows


def film_snapshot(surface, bulk, interface) -> dict:
    out: dict[str, float] = {}
    for slab, name in zip((surface, bulk, interface), FILM_SLABS, strict=True):
        out[f"{name}_thick"] = float(slab.thick.value)
        out[f"{name}_rough"] = float(slab.rough.value)
        out[f"{name}_density"] = float(slab.sld.density.value)
        out[f"{name}_rotation"] = float(slab.sld.rotation.value)
    return out


def instrument_snapshot(model: ReflectModel) -> dict[float, dict[str, float]]:
    return {
        float(energy): {
            name: float(getattr(model, name).at(energy).value)
            for name in INSTRUMENT_FIELDS
        }
        for energy in E_RES
    }


def run_de(objective: Objective) -> dict:
    objective.model.device = "gpu"
    nll0 = float(objective.nll())
    log(
        f"DE start nll={nll0:.6f}  popsize={DE_KW['popsize']}  "
        f"maxiter={DE_KW['maxiter']}  nvary={len(objective.varying_parameters())}"
    )
    t0 = time.perf_counter()
    result = CurveFitter(objective).fit(
        "differential_evolution",
        workers=gpu_batch(objective),
        **DE_KW,
    )
    wall = time.perf_counter() - t0
    nll1 = float(objective.nll())
    stage = {
        "stage": "de_gpu_logYX4",
        "nll_before": nll0,
        "nll_after": nll1,
        "delta_nll": nll0 - nll1,
        "wall_s": wall,
        "nit": int(getattr(result, "nit", -1)),
        "nfev": int(getattr(result, "nfev", -1)),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        **{k: DE_KW[k] for k in ("popsize", "maxiter", "recombination", "seed")},
    }
    log(
        f"DE done  nll={nll1:.6f}  Δnll={nll0 - nll1:+.6f}  "
        f"nit={stage['nit']}  nfev={stage['nfev']}  wall={wall:.1f}s"
    )
    return stage


def plot_all_energies(objective: Objective, summary: dict, path: Path) -> None:
    y_fit = np.asarray(objective.generative(), dtype=float)
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol_arr = np.asarray(ds.pol)
    e = np.asarray(ds.energy)

    nrows = len(E_RES)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(8.5, 2.2 * nrows), constrained_layout=True, sharex=True
    )
    for i, energy in enumerate(E_RES):
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol_arr == pol_name) & np.isclose(e, energy)
            qq, rr, ee, ff = q[m], r[m], r_err[m], y_fit[m]
            order = np.argsort(qq)
            qq, rr, ee, ff = qq[order], rr[order], ee[order], ff[order]
            yd, ed = TRANSFORM(qq, rr, ee)
            yf, _ = TRANSFORM(qq, ff)
            ax.errorbar(
                qq,
                yd,
                yerr=ed,
                fmt="o",
                ms=2.0,
                lw=0.5,
                color="0.35",
                ecolor="0.75",
                elinewidth=0.4,
                capsize=0,
                zorder=1,
            )
            ax.plot(qq, yf, color="C0", lw=1.1, zorder=2)
            ax.set_ylabel(r"$\log_{10}(Rq^{4})$")
            per = next(
                row
                for row in summary["per_energy"]
                if np.isclose(row["energy"], energy)
            )
            ax.set_title(f"{energy:g} eV  {pol_name}  RMS={per['rms']:.2f}", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")

    after = summary["after"]
    ox = summary["oxide"]
    fig.suptitle(
        f"DFT slab E_RES (log YX4 DE)  |  nll={after['nll']:.3f}  "
        f"RMS={after['rms']:.3f}  |  locked oxide={ox['thick']:.2f}/{ox['rough']:.2f}",
        fontsize=10,
    )
    fig.savefig(path, dpi=160)
    log(f"wrote {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={OUT_DIR}")
    oxide_mle, energy_offset, bkg_lock = load_bookended_locks()
    warm = load_refloxide_fit(WARM_PATH)
    objective, comps = build_objective(
        warm, oxide_mle, energy_offset, bkg_lock=bkg_lock, device="gpu"
    )
    _vac, surface, bulk, interface, oxide, _sub = comps

    before = metrics(objective)
    log(f"before: {before}")
    log(f"gpu varying ({len(objective.varying_parameters())}):")
    for p in objective.varying_parameters():
        log(
            f"  {p.name}={float(p.value):.6g} "
            f"bounds=({float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g})"
        )

    history = [run_de(objective)]
    after = metrics(objective)
    per = per_energy_rms(objective)

    summary = {
        "energies": list(E_RES),
        "transform": "logYX4",
        "rho_sio2": RHO_SIO2,
        "rho_si": RHO_SI,
        "substrate_rough": SUBSTRATE_ROUGH,
        "energy_offset": float(energy_offset),
        "before": before,
        "after": after,
        "history": history,
        "per_energy": per,
        "film": film_snapshot(surface, bulk, interface),
        "oxide": {
            "thick": float(oxide.thick.value),
            "rough": float(oxide.rough.value),
            "density": float(oxide.sld.density.value),
        },
        "instruments": instrument_snapshot(objective.model),
        "protocol": "gpu_de_dft_slab_eres_locked_oxide_bkg2837_logYX4",
        "warm_from": str(WARM_PATH.resolve()),
        "bookended_locks": str(BOOKENDED_POLISH.resolve()),
    }

    fit_path = OUT_DIR / "fit.pkl"
    with fit_path.open("wb") as fh:
        pickle.dump({"objective": objective, "summary": summary}, fh)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_all_energies(objective, summary, OUT_DIR / "demo_dft_slab_eres_logyx4.png")
    log(f"wrote {fit_path}")
    log(f"after: {after}")
    for row in per:
        log(f"  {row['energy']:g} eV  RMS={row['rms']:.3f}  n={row['n']}")


if __name__ == "__main__":
    main()

"""Fully open bookended L-BFGS refit on ``E_RES``, seeded from profile best fit.

Starts at the profile-contract posterior (film median + shared instrument
centroid + frozen SiO2/Si geometry/density), then frees film, oxide
(thick/rough/density), substrate (density/rough), and instrument
scale/theta/bkg. SiO2/Si start at the profile values (2.2 / 2.33).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import Transform
from scipy.optimize import minimize
from utils import models_fitting_results
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

ROOT = models_fitting_results / "eres_dynesty_gpu" / "full_profile_contract"
OUT = ROOT / "bookended_open_refit"
_HERE = Path(__file__).resolve().parent

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
E_RES = (283.7, 284.2, 285.1, 287.0, 289.0)
TRANSFORM = Transform("logY")
COLOR_BOOK = "#1F618D"
COLOR_OPEN = "#117A65"
COLOR_SLAB = "#C0392B"
COLOR_DATA = "0.25"
N_DEPTH = 700

# Physical boxes for previously frozen substrate optics.
OXIDE_DENSITY_BOUNDS = (1.8, 2.45)
SI_DENSITY_BOUNDS = (2.20, 2.45)
OXIDE_THICK_BOUNDS = (2.5, 16.0)
OXIDE_ROUGH_BOUNDS = (0.5, 14.0)
SI_ROUGH_BOUNDS = (0.3, 5.0)
BKG_BOUNDS = (0.0, 5e-3)


def log(msg: str) -> None:
    print(msg, flush=True)


def _rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def prep_objective(objective) -> None:
    objective.model.device = "cpu"
    objective.model.parallel = False
    objective.transform = TRANSFORM
    objective.anisotropy_weight = 0.0
    if hasattr(objective, "normalization"):
        objective.normalization = "auto"
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()


def medians(post: dict) -> dict[str, float]:
    return {p["name"]: float(p["median"]) for p in post["params"]}


def overall_rms(objective) -> float:
    ds = objective._dataset
    y = np.asarray(objective.generative(), dtype=float)
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    yd, ed_arr = TRANSFORM(q, r, r_err)
    ym, _ = TRANSFORM(q, y)
    ed = np.asarray(ed_arr, dtype=float)
    m = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
    return float(np.sqrt(np.mean(((yd[m] - ym[m]) / ed[m]) ** 2)))


def varying_state(objective) -> tuple[list[str], np.ndarray, list[tuple[float, float]]]:
    names: list[str] = []
    x0: list[float] = []
    bounds: list[tuple[float, float]] = []
    for p in objective.varying_parameters():
        names.append(str(p.name))
        x0.append(float(p.value))
        bounds.append((float(p.bounds.lb), float(p.bounds.ub)))
    return names, np.asarray(x0, dtype=float), bounds


def set_varying(objective, names: list[str], x: np.ndarray) -> None:
    by_name = {str(p.name): p for p in objective.varying_parameters()}
    for name, val in zip(names, x, strict=True):
        by_name[name].value = float(val)


def open_all_from_profile_best(ed, objective, book_med: dict[str, float]) -> list[str]:
    """Seed at profile best fit, then free film + oxide/Si + instruments + bkg."""
    film = objective.model.structure[1].profile
    oxide = objective.model.structure[2]
    substrate = objective.model.structure[3]
    pi2 = float(np.pi / 2)
    opened: list[str] = []

    # Film at profile posterior median with profile-width boxes.
    film_bounds = {
        "total_thick": ed._local_bounds(
            float(book_med["total_thick"]), lo=170.0, hi=210.0, rel=0.08
        ),
        "surface_roughness": ed.PROFILE_SURFACE_ROUGH_BOUNDS,
        "density_bulk": ed._local_bounds(
            float(book_med["density_bulk"]), lo=1.2, hi=2.1, rel=0.15
        ),
        "density_si": ed.PROFILE_DENSITY_SI_BOUNDS,
        "density_vac": ed.PROFILE_DENSITY_VAC_BOUNDS,
        "tau_si": ed.PROFILE_TAU_SI_BOUNDS,
        "tau_vac": ed.PROFILE_TAU_VAC_BOUNDS,
        "alpha_bulk": ed._local_bounds(
            float(book_med["alpha_bulk"]), lo=0.9, hi=pi2, abs_pad=0.20, rel=None
        ),
        "alpha_si": ed._local_bounds(
            float(book_med["alpha_si"]), lo=0.15, hi=0.95, abs_pad=0.25, rel=None
        ),
        "alpha_vac": ed._local_bounds(
            float(book_med["alpha_vac"]), lo=0.7, hi=pi2, abs_pad=0.20, rel=None
        ),
    }
    for name in FILM_NAMES:
        ed._setp(
            getattr(film, name),
            value=float(book_med[name]),
            vary=True,
            bounds=film_bounds[name],
        )
        opened.append(name)

    # Oxide / Si start at profile (same substrates), then free.
    ox_t = float(oxide.thick.value)
    ox_r = float(oxide.rough.value)
    ox_d = float(oxide.sld.density.value)
    si_d = float(substrate.sld.density.value)
    si_r = float(substrate.rough.value)
    ed._setp(oxide.thick, value=ox_t, vary=True, bounds=OXIDE_THICK_BOUNDS)
    ed._setp(oxide.rough, value=ox_r, vary=True, bounds=OXIDE_ROUGH_BOUNDS)
    ed._setp(oxide.sld.density, value=ox_d, vary=True, bounds=OXIDE_DENSITY_BOUNDS)
    ed._setp(substrate.sld.density, value=si_d, vary=True, bounds=SI_DENSITY_BOUNDS)
    ed._setp(substrate.rough, value=si_r, vary=True, bounds=SI_ROUGH_BOUNDS)
    opened.extend(
        [
            "oxide.thick",
            "oxide.rough",
            "oxide.density",
            "substrate.density",
            "substrate.rough",
        ]
    )

    # Instruments: restore open-contract centroid (profile freeze), then free.
    centroid = ed._load_open_contract_medians()
    for p in objective.parameters.flattened():
        name = str(p.name)
        if name in centroid and (
            name.startswith("scale_") or name.startswith("theta_offset_")
        ):
            p.value = float(centroid[name])
    ed._widen_instrument_bounds(objective)
    for p in objective.parameters.flattened():
        name = str(p.name)
        if name.startswith("scale_") or name.startswith("theta_offset_"):
            if not p.vary:
                p.vary = True
            opened.append(name)
        elif name.startswith("bkg"):
            val = max(0.0, float(p.value))
            # Profile/polish sometimes leaves bkg at 0 with infinite ub; give a
            # finite box so L-BFGS finite differences stay well-defined.
            ed._setp(p, value=val, vary=True, bounds=BKG_BOUNDS)
            opened.append(name)

    # Keep shared energy_offset locked at profile value.
    if hasattr(objective.model, "energy_offset"):
        objective.model.energy_offset.setp(vary=False)

    objective.refresh_parameters()
    log(
        f"fully open from profile best: free={len(objective.varying_parameters())}  "
        f"oxide0=({ox_t:.3g},{ox_r:.3g},rho={ox_d:.3g})  "
        f"Si0=(rho={si_d:.3g},rough={si_r:.3g})"
    )
    return opened


def plot_fit_quality(
    objective,
    *,
    title: str,
    stem: Path,
    secondary: np.ndarray | None = None,
    secondary_label: str = "profile slab median",
) -> dict:
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e_arr = np.asarray(ds.energy, dtype=float)
    y_pri = np.asarray(objective.generative(), dtype=float)
    nll = float(objective.nll())
    rms = overall_rms(objective)

    _rc()
    nrows = len(E_RES)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(7.0, 1.55 * nrows + 0.85),
        sharex="col",
        constrained_layout=True,
    )
    data_kw = dict(
        fmt="o",
        ms=2.2,
        lw=0.45,
        color=COLOR_DATA,
        ecolor="0.65",
        elinewidth=0.4,
        capsize=0,
        zorder=1,
        rasterized=True,
    )
    letters = "abcdefghij"
    per_energy = []
    for i, energy in enumerate(E_RES):
        row: dict = {"energy": float(energy)}
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq, rr, ee = q[m], r[m], r_err[m]
            yp = y_pri[m]
            order = np.argsort(qq)
            qq, rr, ee, yp = qq[order], rr[order], ee[order], yp[order]
            yd, ed = TRANSFORM(qq, rr, ee)
            y1, _ = TRANSFORM(qq, yp)
            ax.errorbar(qq, yd, yerr=ed, **data_kw)
            ax.plot(qq, y1, color=COLOR_OPEN, lw=1.25, zorder=3)
            if secondary is not None:
                ys = secondary[m][order]
                y2, _ = TRANSFORM(qq, ys)
                ax.plot(qq, y2, color=COLOR_SLAB, lw=1.05, ls="--", zorder=2)
            ed_a = np.asarray(ed, dtype=float)
            ok = np.isfinite(yd) & np.isfinite(y1) & (ed_a > 0)
            rms_ch = float(np.sqrt(np.mean(((yd[ok] - y1[ok]) / ed_a[ok]) ** 2)))
            row[f"rms_{pol_name}"] = rms_ch
            ax.text(
                0.02,
                0.94,
                f"({letters[i * 2 + col]})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=9,
            )
            ax.text(
                0.98,
                0.94,
                f"{energy:g} eV  {pol_name}\nRMS={rms_ch:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="0.2",
                linespacing=1.25,
            )
            if col == 0:
                ax.set_ylabel(r"$\log_{10} R$")
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        per_energy.append(row)
    axes[0, 0].plot([], [], "o", ms=3.5, color=COLOR_DATA, label="data")
    axes[0, 0].plot([], [], color=COLOR_OPEN, lw=1.4, label="fully open refit")
    if secondary is not None:
        axes[0, 0].plot(
            [], [], color=COLOR_SLAB, lw=1.2, ls="--", label=secondary_label
        )
    axes[0, 0].legend(loc="lower left", frameon=False, handlelength=1.8, fontsize=7.5)
    fig.suptitle(
        f"{title}\nnll={nll:.1f}  ·  overall RMS={rms:.2f}  ·  "
        f"nvary={len(objective.varying_parameters())}",
        fontsize=10,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"nll": nll, "overall_rms": rms, "per_energy": per_energy}


def plot_depth_compare(
    z: np.ndarray,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    stem: Path,
    meta: dict,
) -> None:
    _rc()
    styles = {
        "profile_book": (COLOR_BOOK, "--", 1.5, "profile bookended median"),
        "open_refit": (COLOR_OPEN, "-", 1.9, "fully open refit"),
        "profile_slab": (COLOR_SLAB, "-", 1.5, "profile slab median"),
    }
    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    for key, (rho, gam) in curves.items():
        color, ls, lw, label = styles[key]
        ax_a.plot(z, gam, color=color, ls=ls, lw=lw, label=label)
        ax_r.plot(z, rho, color=color, ls=ls, lw=lw, label=label)
    for ax in (ax_a, ax_r):
        ax.axvline(0.0, color="0.55", ls=":", lw=0.8)
        ax.set_xlabel(r"$z$ (Å)")
        ax.set_xlim(float(z[0]), float(z[-1]))
        ax.legend(frameon=False, fontsize=7.5, loc="best")
    ax_a.set_ylabel(r"$\gamma$ (deg)")
    ax_a.set_title("orientation")
    ax_r.set_ylabel(r"$\rho$ (g/cm$^3$)")
    ax_r.set_title("density")
    fig.suptitle(
        "Fully open bookended refit (from profile best fit)\n"
        f"nll open={meta['nll_open_refit']:.1f}  ·  "
        f"profile book={meta['nll_profile_book']:.1f}  ·  "
        f"profile slab={meta['nll_profile_slab']:.1f}  ·  "
        f"SiO2={meta['oxide_density_fit']:.3f}  Si={meta['si_density_fit']:.3f}",
        fontsize=10,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ed = _load_module("ed_full_open", _HERE / "_eres_dynesty_gpu.py")
    overlay = _load_module(
        "overlay_full_open", _HERE / "_plot_profile_contract_depth_overlay.py"
    )
    book_helpers = overlay._load_book_helpers()

    book_post = json.loads((ROOT / "bookended" / "posterior_summary.json").read_text())
    slab_post = json.loads((ROOT / "dft_slab" / "posterior_summary.json").read_text())
    book_med = medians(book_post)
    slab_med = medians(slab_post)

    # Start from profile-contract objective (best fit basin), then open all.
    book = ed.load_bookended(contract="profile")
    open_all_from_profile_best(ed, book, book_med)
    prep_objective(book)

    names, x_seed, bounds = varying_state(book)
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    if not np.all(np.isfinite(lo) & np.isfinite(hi)):
        bad = [
            n
            for n, (a, b) in zip(names, bounds, strict=True)
            if not (np.isfinite(a) and np.isfinite(b) and b > a)
        ]
        raise RuntimeError(f"non-finite free bounds: {bad}")
    x_seed = np.clip(x_seed, lo, hi)
    set_varying(book, names, x_seed)
    prep_objective(book)

    nll_seed = float(book.nll())
    rms_seed = overall_rms(book)
    seed_vals = {n: float(v) for n, v in zip(names, x_seed, strict=True)}
    log(
        f"seed (profile best, all free): nvary={len(names)}  nll={nll_seed:.2f}  rms={rms_seed:.3f}"
    )

    def nll_fn(x):
        try:
            set_varying(book, names, x)
            book.refresh_parameters()
            val = float(book.nll())
            if not np.isfinite(val):
                return 1.0e12
            return val
        except (ValueError, RuntimeError, FloatingPointError):
            return 1.0e12

    # Multi-start L-BFGS: seed + a few interior draws (singular TMM can abort a
    # single trajectory when instruments/oxide move far).
    rng = np.random.default_rng(1)
    candidates = [x_seed.copy()]
    for _ in range(6):
        u = rng.random(len(x_seed))
        xt = lo + u * (hi - lo)
        candidates.append(np.clip(0.75 * x_seed + 0.25 * xt, lo, hi))

    best_fun = nll_seed
    best_x = x_seed.copy()
    best_nit = 0
    best_success = False
    for i, xt in enumerate(candidates):
        res = minimize(
            nll_fn,
            xt,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-5},
        )
        fun = float(res.fun)
        log(
            f"  start[{i}]: nll {nll_fn(xt):.2f} -> {fun:.2f}  "
            f"success={res.success} nit={getattr(res, 'nit', '?')}"
        )
        if fun < best_fun:
            best_fun = fun
            best_x = np.asarray(res.x, dtype=float).copy()
            best_nit = int(getattr(res, "nit", -1))
            best_success = bool(res.success)

    x_fit = best_x
    set_varying(book, names, x_fit)
    prep_objective(book)
    nll_open = float(book.nll())
    rms_open = overall_rms(book)
    fit_vals = {n: float(v) for n, v in zip(names, x_fit, strict=True)}
    oxide = book.model.structure[2]
    substrate = book.model.structure[3]
    ox_d_fit = float(oxide.sld.density.value)
    si_d_fit = float(substrate.sld.density.value)
    y_open = np.asarray(book.generative(), dtype=float).copy()
    log(
        f"fully open polish: nll {nll_seed:.2f} -> {nll_open:.2f}  "
        f"rms {rms_seed:.3f} -> {rms_open:.3f}  "
        f"SiO2 {2.2:.2f}->{ox_d_fit:.3f}  Si {2.33:.2f}->{si_d_fit:.3f}  "
        f"success={best_success} nit={best_nit}"
    )

    # Profile references (frozen instruments) for the ladder.
    book_prof = ed.load_bookended(contract="profile")
    prep_objective(book_prof)
    film_p = book_prof.model.structure[1].profile
    for name in FILM_NAMES:
        getattr(film_p, name).value = float(book_med[name])
    prep_objective(book_prof)
    nll_prof_book = float(book_prof.nll())
    rms_prof_book = overall_rms(book_prof)

    slab = ed.load_slab(contract="profile")
    prep_objective(slab)
    for p in slab.varying_parameters():
        if p.name in slab_med:
            p.value = float(slab_med[p.name])
    prep_objective(slab)
    nll_prof_slab = float(slab.nll())
    rms_prof_slab = overall_rms(slab)
    y_slab = np.asarray(slab.generative(), dtype=float).copy()

    gap0 = nll_prof_book - nll_prof_slab
    gap1 = nll_open - nll_prof_slab
    interpretation = (
        "fully opening params closed most of the nll gap to slab"
        if gap1 < 0.35 * gap0
        else (
            "fully opening helped but slab still preferred"
            if gap1 < 0.85 * gap0
            else "fully opening did not close the gap; bookended limit persists"
        )
    )

    film = book.model.structure[1].profile
    sigma_pad = max(float(film.surface_roughness.value), 12.0)
    L_book = float(film.total_thick.value) + float(oxide.thick.value)
    _n, th_s, _rg, _d, _a = overlay.slab_layers(slab)
    L_slab = float(np.sum(th_s[:-1])) if th_s.size else L_book
    z = np.linspace(-3.0 * sigma_pad, max(L_book, L_slab) + sigma_pad, N_DEPTH)
    rho_open, gam_open = overlay.bookended_profiles_on_grid(
        film, oxide, z, book_helpers
    )
    rho_prof, gam_prof = overlay.bookended_profiles_on_grid(
        film_p, book_prof.model.structure[2], z, book_helpers
    )
    rho_slab, gam_slab = overlay.slab_profiles_on_grid(slab, z)

    meta = {
        "nvary": len(names),
        "names": names,
        "seed": "profile_contract_posterior_median_film_plus_open_centroid_instruments",
        "nll_seed": nll_seed,
        "nll_open_refit": nll_open,
        "nll_profile_book": nll_prof_book,
        "nll_profile_slab": nll_prof_slab,
        "rms_seed": rms_seed,
        "rms_open_refit": rms_open,
        "rms_profile_book": rms_prof_book,
        "rms_profile_slab": rms_prof_slab,
        "delta_nll_vs_profile_book": nll_open - nll_prof_book,
        "delta_nll_vs_profile_slab": nll_open - nll_prof_slab,
        "gap_profile_book_minus_slab": gap0,
        "gap_open_minus_slab": gap1,
        "oxide_density_seed": 2.2,
        "si_density_seed": 2.33,
        "oxide_density_fit": ox_d_fit,
        "si_density_fit": si_d_fit,
        "oxide_thick_fit": float(oxide.thick.value),
        "oxide_rough_fit": float(oxide.rough.value),
        "substrate_rough_fit": float(substrate.rough.value),
        "polish_success": bool(best_success),
        "polish_nit": int(best_nit),
        "interpretation": interpretation,
        "seed_values": seed_vals,
        "fit_values": fit_vals,
    }

    plot_depth_compare(
        z,
        {
            "profile_book": (rho_prof, gam_prof),
            "open_refit": (rho_open, gam_open),
            "profile_slab": (rho_slab, gam_slab),
        },
        OUT / "depth_profile",
        meta,
    )
    fq = plot_fit_quality(
        book,
        title="Bookended fully open L-BFGS (from profile best; SiO2/Si free)",
        stem=OUT / "fit_quality_logY",
        secondary=y_slab,
    )

    rows = []
    bound_map = {n: b for n, b in zip(names, bounds, strict=True)}
    for name in names:
        rows.append(
            {
                "name": name,
                "seed": seed_vals[name],
                "fit": fit_vals[name],
                "delta": fit_vals[name] - seed_vals[name],
                "lo": bound_map[name][0],
                "hi": bound_map[name][1],
            }
        )
    csv_path = OUT / "parameters.csv"
    with csv_path.open("w") as fh:
        fh.write("name,seed,fit,delta,lo,hi\n")
        for row in rows:
            fh.write(
                f"{row['name']},{row['seed']:.8g},{row['fit']:.8g},"
                f"{row['delta']:.8g},{row['lo']:.8g},{row['hi']:.8g}\n"
            )

    summary = {
        **meta,
        "fit_quality": fq,
        "parameters": rows,
        "paths": {
            "fit_quality": str((OUT / "fit_quality_logY.png").resolve()),
            "depth_profile": str((OUT / "depth_profile.png").resolve()),
            "parameters_csv": str(csv_path.resolve()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        OUT / "curves.npz",
        z_A=z,
        rho_open=rho_open,
        gam_open=gam_open,
        rho_profile_book=rho_prof,
        gam_profile_book=gam_prof,
        rho_profile_slab=rho_slab,
        gam_profile_slab=gam_slab,
        y_open=y_open,
        y_slab=y_slab,
    )
    log(f"wrote {OUT / 'summary.json'}")
    log(
        "nll ladder: "
        f"profile_book={nll_prof_book:.1f}  seed={nll_seed:.1f}  "
        f"fully_open={nll_open:.1f}  profile_slab={nll_prof_slab:.1f}"
    )
    log(f"interpretation: {interpretation}")


if __name__ == "__main__":
    main()

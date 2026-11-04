"""Match bookended film morphology to the DFT-slab posterior median, then polish.

1. Build slab median rho(z)/gamma(z) with erf interface roughness.
2. L-BFGS-match bookended film params to that morphology (ZnPc region only).
3. L-BFGS reflectivity polish from the matched seed under the profile contract
   (frozen shared instruments + oxide; logY + auto + NC).
4. Write comparison tables and overlays under ``morphology_match/``.
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
OUT = ROOT / "morphology_match"
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
W_RHO = 1.0
W_GAM = (1.0 / np.pi) ** 2
N_DEPTH = 700
COLOR_SLAB = "#C0392B"
COLOR_BOOK0 = "#7F8C8D"
COLOR_MATCH = "#1F618D"
COLOR_POLISH = "#117A65"
COLOR_DATA = "0.25"


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


def set_film(film, values: dict[str, float]) -> None:
    for name, val in values.items():
        if hasattr(film, name):
            getattr(film, name).value = float(val)


def film_bounds(film) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for name in FILM_NAMES:
        p = getattr(film, name)
        out.append((float(p.bounds.lb), float(p.bounds.ub)))
    return out


def film_vector(film) -> np.ndarray:
    return np.asarray([float(getattr(film, n).value) for n in FILM_NAMES], dtype=float)


def apply_film_vector(film, x: np.ndarray) -> None:
    for name, val in zip(FILM_NAMES, x, strict=True):
        getattr(film, name).value = float(val)


def overall_rms(objective) -> float:
    ds = objective._dataset
    y = np.asarray(objective.generative(), dtype=float)
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    yd, ed = TRANSFORM(q, r, r_err)
    ym, _ = TRANSFORM(q, y)
    ed = np.asarray(ed, dtype=float)
    m = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
    return float(np.sqrt(np.mean(((yd[m] - ym[m]) / ed[m]) ** 2)))


def morphology_loss(
    x: np.ndarray,
    *,
    film,
    oxide,
    overlay,
    book_helpers,
    z: np.ndarray,
    mask: np.ndarray,
    rho_t: np.ndarray,
    gam_t: np.ndarray,
) -> float:
    apply_film_vector(film, x)
    rho_b, gam_b = overlay.bookended_profiles_on_grid(film, oxide, z, book_helpers)
    dr = rho_b[mask] - rho_t[mask]
    dg = np.radians(gam_b[mask] - gam_t[mask])
    return float(W_RHO * np.mean(dr * dr) + W_GAM * np.mean(dg * dg))


def morphology_rmse(
    film, oxide, overlay, book_helpers, z, mask, rho_t, gam_t
) -> dict[str, float]:
    rho_b, gam_b = overlay.bookended_profiles_on_grid(film, oxide, z, book_helpers)
    return {
        "rmse_rho": float(np.sqrt(np.mean((rho_b[mask] - rho_t[mask]) ** 2))),
        "rmse_gamma_deg": float(np.sqrt(np.mean((gam_b[mask] - gam_t[mask]) ** 2))),
        "loss": float(
            W_RHO * np.mean((rho_b[mask] - rho_t[mask]) ** 2)
            + W_GAM * np.mean(np.radians(gam_b[mask] - gam_t[mask]) ** 2)
        ),
    }


def plot_morphology_overlay(
    z: np.ndarray,
    mask: np.ndarray,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    stem: Path,
    meta: dict,
) -> None:
    _rc()
    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    styles = {
        "slab_median": (COLOR_SLAB, "-", 2.0, "DFT-slab median"),
        "book_median": (COLOR_BOOK0, "--", 1.5, "bookended median"),
        "book_matched": (COLOR_MATCH, "-", 1.8, "bookended morphology-matched"),
        "book_polished": (COLOR_POLISH, "-", 1.6, "bookended refl-polished"),
    }
    for key, (rho, gam) in curves.items():
        color, ls, lw, label = styles[key]
        ax_a.plot(z, gam, color=color, ls=ls, lw=lw, label=label)
        ax_r.plot(z, rho, color=color, ls=ls, lw=lw, label=label)
    for ax in (ax_a, ax_r):
        ax.axvline(0.0, color="0.55", ls=":", lw=0.8)
        z_hi = float(z[mask][-1]) if np.any(mask) else float(z[-1])
        ax.axvline(z_hi, color="0.55", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlabel(r"$z$ (Å)")
        ax.set_xlim(float(z[0]), float(z[-1]))
        ax.legend(frameon=False, fontsize=7.5, loc="best")
    ax_a.set_ylabel(r"$\gamma$ (deg)")
    ax_a.set_title("orientation")
    ax_r.set_ylabel(r"$\rho$ (g/cm$^3$)")
    ax_r.set_title("density")
    fig.suptitle(
        "Morphology match: bookended → DFT-slab median (profile contract)\n"
        f"match RMSE ρ={meta['match_rmse_rho']:.3f} g/cm³, "
        f"γ={meta['match_rmse_gamma_deg']:.2f}°  ·  "
        f"nll book med={meta['nll_book_median']:.1f} -> "
        f"matched={meta['nll_matched']:.1f} -> "
        f"polished={meta['nll_polished']:.1f}  ·  "
        f"slab med={meta['nll_slab_median']:.1f}",
        fontsize=9.5,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_fit_quality(
    objective,
    *,
    title: str,
    stem: Path,
    secondary: dict[str, np.ndarray] | None = None,
) -> dict:
    """Plot data vs generative for the current objective state."""
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
            ax.plot(qq, y1, color=COLOR_MATCH, lw=1.25, zorder=3, label="primary")
            if secondary is not None:
                ys = secondary["y"][m][order]
                y2, _ = TRANSFORM(qq, ys)
                ax.plot(
                    qq,
                    y2,
                    color=COLOR_SLAB,
                    lw=1.05,
                    ls="--",
                    zorder=2,
                    label="slab median",
                )
            ed_arr = np.asarray(ed, dtype=float)
            ok = np.isfinite(yd) & np.isfinite(y1) & (ed_arr > 0)
            rms_ch = float(np.sqrt(np.mean(((yd[ok] - y1[ok]) / ed_arr[ok]) ** 2)))
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
    axes[0, 0].plot([], [], color=COLOR_MATCH, lw=1.4, label="bookended")
    if secondary is not None:
        axes[0, 0].plot([], [], color=COLOR_SLAB, lw=1.2, ls="--", label="slab median")
    axes[0, 0].legend(loc="lower left", frameon=False, handlelength=1.8, fontsize=7.5)
    fig.suptitle(f"{title}\nnll={nll:.1f}  ·  overall RMS={rms:.2f}", fontsize=10)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"nll": nll, "overall_rms": rms, "per_energy": per_energy}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ed = _load_module("ed_match", _HERE / "_eres_dynesty_gpu.py")
    overlay = _load_module(
        "overlay_match", _HERE / "_plot_profile_contract_depth_overlay.py"
    )
    book_helpers = overlay._load_book_helpers()

    book = ed.load_bookended(contract="profile")
    slab = ed.load_slab(contract="profile")
    prep_objective(book)
    prep_objective(slab)

    film = book.model.structure[1].profile
    oxide = book.model.structure[2]

    book_post = json.loads((ROOT / "bookended" / "posterior_summary.json").read_text())
    slab_post = json.loads((ROOT / "dft_slab" / "posterior_summary.json").read_text())
    book_med = medians(book_post)
    slab_med = medians(slab_post)

    # Set slab to posterior median for the morphology target.
    for p in slab.varying_parameters():
        if p.name in slab_med:
            p.value = float(slab_med[p.name])
    slab.refresh_parameters()

    set_film(film, {k: book_med[k] for k in FILM_NAMES})
    book.refresh_parameters()

    names_s, th_s, _rg, _d, _a = overlay.slab_layers(slab)
    # ZnPc ends before oxide (oxide then synthetic Si pad are last two).
    L_znpc = float(np.sum(th_s[:-2])) if th_s.size >= 2 else float(np.sum(th_s))
    sigma_pad = max(
        float(book_med["surface_roughness"]),
        float(slab_med["ZnPc_surface_rough"]),
        12.0,
    )
    z = np.linspace(-3.0 * sigma_pad, L_znpc + 8.0, N_DEPTH)
    # Loss mask: vacuum pad through ZnPc (stop before oxide jump).
    mask = (z >= -2.0 * sigma_pad) & (z <= L_znpc)

    rho_t, gam_t = overlay.slab_profiles_on_grid(slab, z)
    rho0, gam0 = overlay.bookended_profiles_on_grid(film, oxide, z, book_helpers)

    before = morphology_rmse(film, oxide, overlay, book_helpers, z, mask, rho_t, gam_t)
    log(
        f"morphology before: RMSE_rho={before['rmse_rho']:.4f}  "
        f"RMSE_gamma={before['rmse_gamma_deg']:.3f} deg  loss={before['loss']:.5f}"
    )

    x0 = film_vector(film)
    bounds = film_bounds(film)

    def loss_fn(x):
        return morphology_loss(
            x,
            film=film,
            oxide=oxide,
            overlay=overlay,
            book_helpers=book_helpers,
            z=z,
            mask=mask,
            rho_t=rho_t,
            gam_t=gam_t,
        )

    res_m = minimize(
        loss_fn,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-14},
    )
    apply_film_vector(film, res_m.x)
    book.refresh_parameters()
    matched_vals = {n: float(v) for n, v in zip(FILM_NAMES, res_m.x, strict=True)}
    after = morphology_rmse(film, oxide, overlay, book_helpers, z, mask, rho_t, gam_t)
    log(
        f"morphology after:  RMSE_rho={after['rmse_rho']:.4f}  "
        f"RMSE_gamma={after['rmse_gamma_deg']:.3f} deg  loss={after['loss']:.5f}  "
        f"success={res_m.success} nit={res_m.nit}"
    )

    # Reflectivity ladder -------------------------------------------------
    # 1) bookended posterior median
    set_film(film, {k: book_med[k] for k in FILM_NAMES})
    prep_objective(book)
    nll_book = float(book.nll())
    rms_book = overall_rms(book)
    y_book = np.asarray(book.generative(), dtype=float).copy()

    # 2) morphology-matched seed
    set_film(film, matched_vals)
    prep_objective(book)
    nll_matched = float(book.nll())
    rms_matched = overall_rms(book)
    y_matched = np.asarray(book.generative(), dtype=float).copy()
    rho_m, gam_m = overlay.bookended_profiles_on_grid(film, oxide, z, book_helpers)

    # 3) reflectivity polish from matched seed
    x_match = film_vector(film)

    def nll_fn(x):
        apply_film_vector(film, x)
        book.refresh_parameters()
        return float(book.nll())

    res_r = minimize(
        nll_fn,
        x_match,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-12},
    )
    apply_film_vector(
        film, res_r.x if (res_r.success or res_r.fun < nll_matched) else x_match
    )
    prep_objective(book)
    polished_vals = {n: float(getattr(film, n).value) for n in FILM_NAMES}
    nll_polished = float(book.nll())
    rms_polished = overall_rms(book)
    y_polished = np.asarray(book.generative(), dtype=float).copy()
    rho_p, gam_p = overlay.bookended_profiles_on_grid(film, oxide, z, book_helpers)
    log(
        f"refl polish: nll {nll_matched:.2f} -> {nll_polished:.2f}  "
        f"rms {rms_matched:.3f} -> {rms_polished:.3f}  "
        f"success={res_r.success} nit={getattr(res_r, 'nit', '?')}"
    )

    # 4) slab median reference on its own objective
    for p in slab.varying_parameters():
        if p.name in slab_med:
            p.value = float(slab_med[p.name])
    prep_objective(slab)
    nll_slab = float(slab.nll())
    rms_slab = overall_rms(slab)
    y_slab = np.asarray(slab.generative(), dtype=float).copy()

    meta = {
        "match_rmse_rho": after["rmse_rho"],
        "match_rmse_gamma_deg": after["rmse_gamma_deg"],
        "match_rmse_rho_before": before["rmse_rho"],
        "match_rmse_gamma_deg_before": before["rmse_gamma_deg"],
        "morphology_success": bool(res_m.success),
        "morphology_nit": int(getattr(res_m, "nit", -1)),
        "refl_polish_success": bool(res_r.success),
        "refl_polish_nit": int(getattr(res_r, "nit", -1)),
        "nll_book_median": nll_book,
        "nll_matched": nll_matched,
        "nll_polished": nll_polished,
        "nll_slab_median": nll_slab,
        "rms_book_median": rms_book,
        "rms_matched": rms_matched,
        "rms_polished": rms_polished,
        "rms_slab_median": rms_slab,
        "L_znpc_slab_A": L_znpc,
        "book_median": {k: book_med[k] for k in FILM_NAMES},
        "matched": matched_vals,
        "polished": polished_vals,
        "slab_median_film": slab_med,
    }

    # Exports -------------------------------------------------------------
    plot_morphology_overlay(
        z,
        mask,
        {
            "slab_median": (rho_t, gam_t),
            "book_median": (rho0, gam0),
            "book_matched": (rho_m, gam_m),
            "book_polished": (rho_p, gam_p),
        },
        OUT / "depth_profile_morphology_match",
        meta,
    )
    log(f"wrote {OUT / 'depth_profile_morphology_match.png'}")

    # Fit quality: polished primary, slab median dashed (same dataset indices).
    # Align slab generative onto book dataset order (identical points).
    set_film(film, polished_vals)
    prep_objective(book)
    fq_polished = plot_fit_quality(
        book,
        title="Bookended after morphology-match + reflectivity polish",
        stem=OUT / "fit_quality_polished",
        secondary={"y": y_slab},
    )
    set_film(film, matched_vals)
    prep_objective(book)
    fq_matched = plot_fit_quality(
        book,
        title="Bookended morphology-matched seed (no refl polish)",
        stem=OUT / "fit_quality_matched",
        secondary={"y": y_slab},
    )
    set_film(film, {k: book_med[k] for k in FILM_NAMES})
    prep_objective(book)
    fq_book = plot_fit_quality(
        book,
        title="Bookended posterior median (baseline)",
        stem=OUT / "fit_quality_book_median",
        secondary={"y": y_slab},
    )

    params_rows = []
    for name in FILM_NAMES:
        params_rows.append(
            {
                "name": name,
                "book_median": book_med[name],
                "matched": matched_vals[name],
                "polished": polished_vals[name],
                "delta_match": matched_vals[name] - book_med[name],
                "delta_polish": polished_vals[name] - matched_vals[name],
            }
        )
    csv_path = OUT / "parameters.csv"
    with csv_path.open("w") as fh:
        fh.write("name,book_median,matched,polished,delta_match,delta_polish\n")
        for row in params_rows:
            fh.write(
                f"{row['name']},{row['book_median']:.8g},{row['matched']:.8g},"
                f"{row['polished']:.8g},{row['delta_match']:.8g},"
                f"{row['delta_polish']:.8g}\n"
            )

    summary = {
        **meta,
        "fit_quality_book_median": fq_book,
        "fit_quality_matched": fq_matched,
        "fit_quality_polished": fq_polished,
        "parameters": params_rows,
        "paths": {
            "depth_profile": str(
                (OUT / "depth_profile_morphology_match.png").resolve()
            ),
            "fit_quality_polished": str((OUT / "fit_quality_polished.png").resolve()),
            "fit_quality_matched": str((OUT / "fit_quality_matched.png").resolve()),
            "fit_quality_book_median": str(
                (OUT / "fit_quality_book_median.png").resolve()
            ),
            "parameters_csv": str(csv_path.resolve()),
        },
        "interpretation": (
            "nll closed toward slab"
            if nll_polished < 0.5 * (nll_book + nll_slab)
            else "nll remained near bookended; parametrization limit likely"
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        OUT / "curves.npz",
        z_A=z,
        mask=mask,
        rho_slab=rho_t,
        gam_slab=gam_t,
        rho_book_median=rho0,
        gam_book_median=gam0,
        rho_matched=rho_m,
        gam_matched=gam_m,
        rho_polished=rho_p,
        gam_polished=gam_p,
    )
    log(f"wrote {OUT / 'summary.json'}")
    log(
        "nll ladder: "
        f"book_med={nll_book:.1f}  matched={nll_matched:.1f}  "
        f"polished={nll_polished:.1f}  slab_med={nll_slab:.1f}"
    )
    log(
        "rms ladder: "
        f"book_med={rms_book:.3f}  matched={rms_matched:.3f}  "
        f"polished={rms_polished:.3f}  slab_med={rms_slab:.3f}"
    )


if __name__ == "__main__":
    main()

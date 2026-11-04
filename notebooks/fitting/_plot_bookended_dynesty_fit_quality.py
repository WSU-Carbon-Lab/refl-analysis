"""Bookended fit-quality overlay from the dynesty posterior median.

Loads the polished bookended objective (optionally unlocked), sets varying
parameters to the dynesty posterior median (MLE dashed), and writes
multi-energy logY data/model + residual figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from refnx.analysis import Transform
from utils import models_fitting_results
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

E_RES = (283.7, 284.2, 285.1, 287.0, 289.0)
TRANSFORM = Transform("logY")
COLOR_DATA = "0.25"
COLOR_MED = "#C0392B"
COLOR_MAP = "#1A5276"
COLOR_MLE = "#7F8C8D"
N_BAND_DRAWS = 200
N_SPAGHETTI = 40

_HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(msg, flush=True)


RUN_DIRS = {
    "full": "full",
    "unlocked": "full_unlocked",
    "polish_contract": "full_polish_contract",
    "open_contract": "full_open_contract",
    "profile_contract": "full_profile_contract",
}
RUN_CONTRACT = {
    "full": "polish",
    "unlocked": "open",
    "polish_contract": "polish",
    "open_contract": "open",
    "profile_contract": "profile",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        choices=tuple(RUN_DIRS),
        default="polish_contract",
        help="Which eres_dynesty_gpu result tree to plot (default: polish_contract)",
    )
    p.add_argument(
        "--unlocked",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias: --unlocked -> --run unlocked",
    )
    p.add_argument(
        "--curve",
        choices=("median", "map", "polish"),
        default="median",
        help="Primary solid curve (default: posterior median); MAP always overplotted",
    )
    return p.parse_args()


def resolve_run(args: argparse.Namespace) -> str:
    if args.unlocked is None:
        return str(args.run)
    return "unlocked" if args.unlocked else "full"


def _load_dynesty_helpers():
    path = _HERE / "_eres_dynesty_gpu.py"
    spec = importlib.util.spec_from_file_location("_eres_dynesty_gpu", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def reflectivity(model, q: np.ndarray, energy: float, pol: str) -> np.ndarray:
    result = model(np.asarray(q, dtype=float), float(energy))
    return np.asarray(result.s if pol == "s" else result.p, dtype=float)


def set_varying(objective, values: dict[str, float]) -> None:
    for p in objective.varying_parameters():
        if p.name in values:
            p.value = float(values[p.name])


def rms_logy(
    q: np.ndarray,
    r: np.ndarray,
    r_err: np.ndarray,
    r_model: np.ndarray,
) -> float:
    yd, ed = TRANSFORM(q, r, r_err)
    ym, _ = TRANSFORM(q, r_model)
    ed = np.asarray(ed, dtype=float)
    m = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean(((yd[m] - ym[m]) / ed[m]) ** 2)))


def main() -> None:
    args = parse_args()
    run = resolve_run(args)
    tag = run
    contract = RUN_CONTRACT[run]
    full_dir = models_fitting_results / "eres_dynesty_gpu" / RUN_DIRS[run] / "bookended"
    out_dir = full_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={out_dir}  run={run}  contract={contract}  curve={args.curve}")

    helpers = _load_dynesty_helpers()
    obj = helpers.load_bookended(contract=contract)
    obj.model.device = "cpu"
    obj.model.parallel = True
    obj.transform = TRANSFORM
    obj.anisotropy_weight = 0.0
    if hasattr(obj, "normalization"):
        obj.normalization = "auto"
    if hasattr(obj, "nc_constraint"):
        obj.nc_constraint = True
    obj._refresh_data_transform_cache()
    obj.refresh_parameters()

    post = json.loads((full_dir / "posterior_summary.json").read_text())
    status = json.loads((full_dir / "status.json").read_text())
    polish = {p["name"]: float(p["mle"]) for p in post["params"]}
    med = {p["name"]: float(p["median"]) for p in post["params"]}
    with (full_dir / "results.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    samples = np.asarray(blob["results"].samples, dtype=float)
    logl = np.asarray(blob["results"].logl, dtype=float)
    logwt = np.asarray(blob["results"].logwt, dtype=float)
    names = list(blob["names"])
    if post.get("map"):
        map_vals = {k: float(v) for k, v in post["map"].items()}
    else:
        i_map = int(np.argmax(logl))
        map_vals = {n: float(samples[i_map, i]) for i, n in enumerate(names)}

    primary = med
    primary_label = "posterior median"
    if args.curve == "map":
        primary = map_vals
        primary_label = "NS MAP"
    elif args.curve == "polish":
        primary = polish
        primary_label = "polish x0"

    logz = float(blob["results"].logz[-1])
    wt = np.exp(logwt - logz)
    wt /= wt.sum()
    rng = np.random.default_rng(1)
    draw_idx = rng.choice(samples.shape[0], size=N_BAND_DRAWS, replace=True, p=wt)
    draws = samples[draw_idx]

    ds = obj._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e_arr = np.asarray(ds.energy, dtype=float)

    set_varying(obj, polish)
    obj.refresh_parameters()
    nll_polish = float(obj.nll())
    logl_polish = float(obj.logl())
    polish_curves: dict[tuple[float, str], np.ndarray] = {}
    for energy in E_RES:
        for pol_name in ("s", "p"):
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq = np.sort(q[m])
            polish_curves[(float(energy), pol_name)] = reflectivity(
                obj.model, qq, energy, pol_name
            )

    set_varying(obj, map_vals)
    obj.refresh_parameters()
    nll_map = float(obj.nll())
    map_curves: dict[tuple[float, str], np.ndarray] = {}
    for energy in E_RES:
        for pol_name in ("s", "p"):
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq = np.sort(q[m])
            map_curves[(float(energy), pol_name)] = reflectivity(
                obj.model, qq, energy, pol_name
            )

    set_varying(obj, med)
    obj.refresh_parameters()
    nll_med = float(obj.nll())
    med_curves: dict[tuple[float, str], np.ndarray] = {}
    for energy in E_RES:
        for pol_name in ("s", "p"):
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq = np.sort(q[m])
            med_curves[(float(energy), pol_name)] = reflectivity(
                obj.model, qq, energy, pol_name
            )

    # Posterior predictive bands in logY.
    band_stacks: dict[tuple[float, str], list[np.ndarray]] = {
        (float(e), p): [] for e in E_RES for p in ("s", "p")
    }
    q_grids: dict[tuple[float, str], np.ndarray] = {}
    for energy in E_RES:
        for pol_name in ("s", "p"):
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            q_grids[(float(energy), pol_name)] = np.sort(q[m])
    for row in draws:
        vals = {n: float(row[i]) for i, n in enumerate(names)}
        set_varying(obj, vals)
        obj.refresh_parameters()
        for key, qq in q_grids.items():
            energy, pol_name = key
            rr = reflectivity(obj.model, qq, energy, pol_name)
            yy, _ = TRANSFORM(qq, rr)
            band_stacks[key].append(np.asarray(yy, dtype=float))
    band_p16 = {k: np.percentile(np.vstack(v), 16, axis=0) for k, v in band_stacks.items()}
    band_p84 = {k: np.percentile(np.vstack(v), 84, axis=0) for k, v in band_stacks.items()}
    band_p025 = {
        k: np.percentile(np.vstack(v), 2.5, axis=0) for k, v in band_stacks.items()
    }
    band_p975 = {
        k: np.percentile(np.vstack(v), 97.5, axis=0) for k, v in band_stacks.items()
    }
    spaghetti_idx = rng.choice(len(draws), size=min(N_SPAGHETTI, len(draws)), replace=False)

    set_varying(obj, primary)
    obj.refresh_parameters()
    nll_curve = float(obj.nll())
    logl_curve = float(obj.logl())

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
    per_energy: list[dict] = []
    chi2_all: list[np.ndarray] = []

    for i, energy in enumerate(E_RES):
        row: dict[str, float] = {"energy": float(energy)}
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq, rr, ee = q[m], r[m], r_err[m]
            order = np.argsort(qq)
            qq, rr, ee = qq[order], rr[order], ee[order]
            key = (float(energy), pol_name)

            r_med = med_curves[key]
            r_map = map_curves[key]
            r_ref = polish_curves[key]
            yd, ed = TRANSFORM(qq, rr, ee)
            y_med, _ = TRANSFORM(qq, r_med)
            y_map, _ = TRANSFORM(qq, r_map)
            y_ref, _ = TRANSFORM(qq, r_ref)
            y_primary = y_med if args.curve == "median" else (
                y_map if args.curve == "map" else y_ref
            )
            r_primary = r_med if args.curve == "median" else (
                r_map if args.curve == "map" else r_ref
            )
            rms = rms_logy(qq, rr, ee, r_primary)
            row[f"rms_{pol_name}"] = rms

            ed_arr = np.asarray(ed, dtype=float)
            ok = np.isfinite(yd) & np.isfinite(y_primary) & (ed_arr > 0)
            chi2_all.append(((yd[ok] - y_primary[ok]) / ed_arr[ok]) ** 2)

            ax.errorbar(qq, yd, yerr=ed, **data_kw)
            ax.fill_between(
                qq,
                band_p025[key],
                band_p975[key],
                color=COLOR_MED,
                alpha=0.18,
                lw=0,
                zorder=2,
            )
            ax.fill_between(
                qq,
                band_p16[key],
                band_p84[key],
                color=COLOR_MED,
                alpha=0.32,
                lw=0,
                zorder=3,
            )
            for j in spaghetti_idx:
                ax.plot(
                    qq,
                    band_stacks[key][int(j)],
                    color=COLOR_MED,
                    lw=0.45,
                    alpha=0.18,
                    zorder=3,
                )
            ax.plot(qq, y_ref, color=COLOR_MLE, lw=1.0, ls=":", zorder=4)
            ax.plot(qq, y_map, color=COLOR_MAP, lw=1.15, ls="--", zorder=5)
            ax.plot(qq, y_med, color=COLOR_MED, lw=1.45, zorder=6)
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
                f"{energy:g} eV  {pol_name}\nRMS={rms:.2f}",
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
        row["rms"] = float(np.hypot(row["rms_s"], row["rms_p"]) / np.sqrt(2.0))
        per_energy.append(row)

    overall_rms = float(np.sqrt(np.mean(np.concatenate(chi2_all))))

    axes[0, 0].fill_between(
        [], [], [], color=COLOR_MED, alpha=0.32, label="16–84% / 2.5–97.5%"
    )
    axes[0, 0].plot([], [], color=COLOR_MED, lw=0.6, alpha=0.5, label="posterior draws")
    axes[0, 0].plot([], [], "o", ms=3.5, color=COLOR_DATA, label="data")
    axes[0, 0].plot([], [], color=COLOR_MED, lw=1.4, label="posterior median")
    axes[0, 0].plot([], [], color=COLOR_MAP, lw=1.2, ls="--", label="NS MAP (max logl)")
    axes[0, 0].plot([], [], color=COLOR_MLE, lw=1.0, ls=":", label="polish x0")
    axes[0, 0].legend(loc="lower left", frameon=False, handlelength=1.8, fontsize=7.5)

    fig.suptitle(
        rf"Bookended ZnPc — dynesty posterior ({tag}, log$Y$)"
        f"\nlogZ={status['logz']:.2f}±{status['logzerr']:.2f}"
        f"  ·  ESS={post['ess']:.0f}"
        f"  ·  overall RMS({primary_label})={overall_rms:.2f}"
        f"  ·  nll(med)={nll_med:.1f} / nll(MAP)={nll_map:.1f} / nll(polish)={nll_polish:.1f}",
        fontsize=10,
    )
    stem = out_dir / "fit_quality_logY"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig_r, axes_r = plt.subplots(
        nrows,
        2,
        figsize=(7.0, 1.35 * nrows + 0.7),
        sharex="col",
        sharey=True,
        constrained_layout=True,
    )
    for i, energy in enumerate(E_RES):
        for col, pol_name in enumerate(("s", "p")):
            ax = axes_r[i, col]
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq, rr, ee = q[m], r[m], r_err[m]
            order = np.argsort(qq)
            qq, rr, ee = qq[order], rr[order], ee[order]
            key = (float(energy), pol_name)
            r_med = med_curves[key]
            yd, ed = TRANSFORM(qq, rr, ee)
            y_med, _ = TRANSFORM(qq, r_med)
            ed_arr = np.asarray(ed, dtype=float)
            resid = (yd - y_med) / np.where(ed_arr > 0, ed_arr, np.nan)
            ax.axhline(0.0, color="0.75", lw=0.7)
            ax.axhline(2.0, color="0.85", lw=0.5, ls=":")
            ax.axhline(-2.0, color="0.85", lw=0.5, ls=":")
            ax.fill_between(
                qq,
                (band_p16[key] - y_med) / np.where(ed_arr > 0, ed_arr, np.nan),
                (band_p84[key] - y_med) / np.where(ed_arr > 0, ed_arr, np.nan),
                color=COLOR_MED,
                alpha=0.25,
                lw=0,
            )
            ax.plot(
                qq, resid, "o", ms=2.0, color=COLOR_MED, alpha=0.85, rasterized=True
            )
            ax.text(
                0.02,
                0.92,
                f"({letters[i * 2 + col]}) {energy:g} eV {pol_name}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            if col == 0:
                ax.set_ylabel(r"$(y-\hat y)/\sigma$")
            if i == nrows - 1:
                ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    axes_r[0, 0].set_ylim(-5.5, 5.5)
    fig_r.suptitle(
        rf"Bookended dynesty posterior median ({tag}) — normalized log$Y$ residuals"
        f"  ·  overall RMS={overall_rms:.2f}  ·  band = 16–84% predictive",
        fontsize=10,
    )
    stem_r = out_dir / "fit_quality_residuals"
    fig_r.savefig(stem_r.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig_r.savefig(stem_r.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_r)

    meta = {
        "tag": tag,
        "curve": args.curve,
        "curve_label": primary_label,
        "energies": list(E_RES),
        "overall_rms": overall_rms,
        "nll_curve": nll_curve,
        "nll_median": nll_med,
        "nll_map": nll_map,
        "nll_polish": nll_polish,
        "logl_curve": logl_curve,
        "logl_polish": logl_polish,
        "n_band_draws": N_BAND_DRAWS,
        "logz": status["logz"],
        "logzerr": status["logzerr"],
        "ess": post["ess"],
        "per_energy": per_energy,
        "figure_png": str(stem.with_suffix(".png").resolve()),
        "residuals_png": str(stem_r.with_suffix(".png").resolve()),
    }
    (out_dir / "fit_quality_summary.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {stem.with_suffix('.png')}")
    log(f"wrote {stem_r.with_suffix('.png')}")
    log(
        f"overall RMS={overall_rms:.3f}  nll(med)={nll_med:.3f}  "
        f"nll(MAP)={nll_map:.3f}  nll(polish)={nll_polish:.3f}"
    )


if __name__ == "__main__":
    main()

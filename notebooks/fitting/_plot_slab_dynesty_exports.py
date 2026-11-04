"""DFT-slab dynesty exports: depth profile, corners, parameter table, fit quality.

Reads ``eres_dynesty_gpu/full_unlocked/dft_slab`` by default.
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
COLOR = "#1F618D"
COLOR_DATA = "0.25"
COLOR_MED = "#1F618D"
COLOR_MLE = "#7F8C8D"
N_PROFILE_DRAWS = 600
_HERE = Path(__file__).resolve().parent

STRUCTURE_KEYS = (
    "ZnPc_surface_thick",
    "ZnPc_surface_rough",
    "ZnPc_surface_density",
    "ZnPc_surface_rotation",
    "ZnPc_bulk_thick",
    "ZnPc_bulk_rough",
    "ZnPc_bulk_density",
    "ZnPc_bulk_rotation",
    "ZnPc_interface_thick",
    "ZnPc_interface_rough",
    "ZnPc_interface_density",
    "ZnPc_interface_rotation",
    "oxide_thick",
    "oxide_rough",
)
STRUCTURE_LABELS = (
    r"$d_{\mathrm{surf}}$",
    r"$\sigma_{\mathrm{surf}}$",
    r"$\rho_{\mathrm{surf}}$",
    r"$\alpha_{\mathrm{surf}}$",
    r"$d_{\mathrm{bulk}}$",
    r"$\sigma_{\mathrm{bulk}}$",
    r"$\rho_{\mathrm{bulk}}$",
    r"$\alpha_{\mathrm{bulk}}$",
    r"$d_{\mathrm{int}}$",
    r"$\sigma_{\mathrm{int}}$",
    r"$\rho_{\mathrm{int}}$",
    r"$\alpha_{\mathrm{int}}$",
    r"$d_{\mathrm{ox}}$",
    r"$\sigma_{\mathrm{ox}}$",
)


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
        default="profile_contract",
        help="Which eres_dynesty_gpu result tree to export (default: profile_contract)",
    )
    p.add_argument(
        "--unlocked",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias: --unlocked -> --run unlocked",
    )
    return p.parse_args()


def resolve_run(args: argparse.Namespace) -> str:
    if args.unlocked is None:
        return str(args.run)
    return "unlocked" if args.unlocked else "full"


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


def _load_helpers():
    path = _HERE / "_eres_dynesty_gpu.py"
    spec = importlib.util.spec_from_file_location("_eres_dynesty_gpu_slab", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_posterior(full_dir: Path):
    with (full_dir / "results.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    results = blob["results"]
    names = list(blob["names"])
    post = json.loads((full_dir / "posterior_summary.json").read_text())
    status = json.loads((full_dir / "status.json").read_text())
    samples = np.asarray(results.samples, dtype=float)
    logwt = np.asarray(results.logwt, dtype=float)
    logz = float(results.logz[-1])
    wt = np.exp(logwt - logz)
    wt /= wt.sum()
    return results, names, samples, wt, post, status


def weighted_resample(samples, wt, n, rng):
    idx = rng.choice(samples.shape[0], size=n, replace=True, p=wt)
    return samples[idx]


def set_varying(objective, values: dict[str, float]) -> None:
    for p in objective.varying_parameters():
        if p.name in values:
            p.value = float(values[p.name])


def reflectivity(model, q, energy, pol):
    result = model(np.asarray(q, dtype=float), float(energy))
    return np.asarray(result.s if pol == "s" else result.p, dtype=float)


def rms_logy(q, r, r_err, r_model):
    yd, ed = TRANSFORM(q, r, r_err)
    ym, _ = TRANSFORM(q, r_model)
    ed = np.asarray(ed, dtype=float)
    m = np.isfinite(yd) & np.isfinite(ym) & np.isfinite(ed) & (ed > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean(((yd[m] - ym[m]) / ed[m]) ** 2)))


def slab_profile_from_objective(objective) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build piecewise depth, density, and orientation (deg) from ZnPc + oxide."""
    structure = objective.model.structure
    z_edges = [0.0]
    rho = []
    alpha = []
    labels = []
    for comp in structure:
        name = str(getattr(comp, "name", "")).lower()
        if name in {"vacuum", "substrate", "si"}:
            continue
        thick = float(comp.thick.value)
        if thick <= 0:
            continue
        z_edges.append(z_edges[-1] + thick)
        sld = getattr(comp, "sld", None)
        dens = (
            float(sld.density.value)
            if sld is not None and hasattr(sld, "density")
            else np.nan
        )
        rot = (
            float(np.degrees(sld.rotation.value))
            if sld is not None and hasattr(sld, "rotation")
            else np.nan
        )
        rho.append(dens)
        alpha.append(rot)
        labels.append(str(getattr(comp, "name", "")))
    z = np.asarray(z_edges, dtype=float)
    return z, np.asarray(rho, dtype=float), np.asarray(alpha, dtype=float)


def profile_quantiles(objective, draws, names, name_index):
    stacks_rho = []
    stacks_alpha = []
    stacks_thick = []
    for row in draws:
        vals = {n: float(row[i]) for n, i in name_index.items()}
        set_varying(objective, vals)
        objective.refresh_parameters()
        z, rho, alpha = slab_profile_from_objective(objective)
        stacks_rho.append(rho)
        stacks_alpha.append(alpha)
        stacks_thick.append(np.diff(z))
    n_layers = stacks_rho[0].size
    rho_mat = np.vstack(stacks_rho)
    alpha_mat = np.vstack(stacks_alpha)
    thick_mat = np.vstack(stacks_thick)
    med_thicks = np.median(thick_mat, axis=0)
    z_edges = np.concatenate([[0.0], np.cumsum(med_thicks)])
    return {
        "z_edges": z_edges,
        "L_A": float(z_edges[-1]),
        "rho_p16": np.percentile(rho_mat, 16, axis=0),
        "rho_p50": np.percentile(rho_mat, 50, axis=0),
        "rho_p84": np.percentile(rho_mat, 84, axis=0),
        "alpha_p16": np.percentile(alpha_mat, 16, axis=0),
        "alpha_p50": np.percentile(alpha_mat, 50, axis=0),
        "alpha_p84": np.percentile(alpha_mat, 84, axis=0),
        "n_draws": int(draws.shape[0]),
        "n_layers": int(n_layers),
    }


def plot_depth_profile(bands, stem, status, post):
    _rc()
    z = bands["z_edges"]

    # step plots: values constant on [z_i, z_{i+1}]
    def step_xy(edges, vals):
        xs, ys = [], []
        for i, v in enumerate(vals):
            xs.extend([edges[i], edges[i + 1]])
            ys.extend([v, v])
        return np.asarray(xs), np.asarray(ys)

    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    for p16, p50, p84, ax, ylab, title in (
        (
            bands["alpha_p16"],
            bands["alpha_p50"],
            bands["alpha_p84"],
            ax_a,
            r"$\gamma$ (deg)",
            "orientation",
        ),
        (
            bands["rho_p16"],
            bands["rho_p50"],
            bands["rho_p84"],
            ax_r,
            r"$\rho$ (g/cm$^3$)",
            "density",
        ),
    ):
        x50, y50 = step_xy(z, p50)
        x16, y16 = step_xy(z, p16)
        x84, y84 = step_xy(z, p84)
        ax.fill_between(x50, y16, y84, color=COLOR, alpha=0.35, lw=0, label="16–84%")
        ax.plot(x50, y16, color=COLOR, lw=0.7, alpha=0.85)
        ax.plot(x50, y84, color=COLOR, lw=0.7, alpha=0.85)
        ax.plot(x50, y50, color=COLOR, lw=1.8, label="posterior median")
        ax.set_xlabel(r"$z$ (Å)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xlim(0.0, bands["L_A"])
        ax.legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle(
        r"DFT-slab ZnPc — dynesty posterior depth profile (profile_contract)"
        f"\n$L={bands['L_A']:.1f}$ Å  ·  {bands['n_draws']} draws  ·  "
        f"band = 16–84%  ·  logZ={status['logz']:.2f}±{status['logzerr']:.2f}"
        f"  ·  ESS={post['ess']:.0f}",
        fontsize=10,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def export_corners(results, names, samples, wt, post, out_dir):
    import corner
    from dynesty import plotting as dyplot

    _rc()
    written = []
    name_index = {n: i for i, n in enumerate(names)}
    med_truths = [
        next(p["median"] for p in post["params"] if p["name"] == n) for n in names
    ]
    map_truths = None
    if post.get("map"):
        map_truths = [
            float(post["map"].get(n, med_truths[i])) for i, n in enumerate(names)
        ]
    note = (
        "solid = posterior median; dashed = NS MAP"
        if map_truths is not None
        else "solid = posterior median"
    )

    rot_names = [n for n in names if "rotation" in n]
    if rot_names:
        idx = [name_index[n] for n in rot_names]
        labels = [n.replace("ZnPc_", "").replace("_", "\n") for n in rot_names]
        fig = corner.corner(
            samples[:, idx],
            weights=wt,
            labels=labels,
            truths=[med_truths[i] for i in idx],
            truth_color="#1A5276",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g",
            color=COLOR,
            hist_kwargs={"density": True},
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            smooth=0.9,
        )
        if map_truths is not None:
            corner.overplot_lines(
                fig, [map_truths[i] for i in idx], color=COLOR, linestyle="--"
            )
        fig.suptitle(
            f"DFT-slab dynesty — rotations\n(shaded 68/95%; {note})",
            fontsize=10,
            y=1.02,
        )
        stem = out_dir / "corner_structure"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        written.append(stem.with_suffix(".png"))

    dens_names = [n for n in names if "density" in n]
    if dens_names:
        idx = [name_index[n] for n in dens_names]
        labels = [n.replace("ZnPc_", "").replace("_", "\n") for n in dens_names]
        fig = corner.corner(
            samples[:, idx],
            weights=wt,
            labels=labels,
            truths=[med_truths[i] for i in idx],
            truth_color="#1A5276",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g",
            color=COLOR,
            hist_kwargs={"density": True},
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            smooth=0.9,
        )
        if map_truths is not None:
            corner.overplot_lines(
                fig, [map_truths[i] for i in idx], color=COLOR, linestyle="--"
            )
        fig.suptitle(
            f"DFT-slab dynesty — densities\n(shaded 68/95%; {note})",
            fontsize=10,
            y=1.02,
        )
        stem = out_dir / "corner_orientation"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        written.append(stem.with_suffix(".png"))

    labels = [n.replace("@", "\n") for n in names]
    fig_all, _ = dyplot.cornerplot(
        results,
        labels=labels,
        truths=med_truths,
        show_titles=True,
        title_quantiles=(0.16, 0.5, 0.84),
        quantiles=(0.16, 0.84),
        color=COLOR,
        truth_color="#1A5276",
        max_n_ticks=3,
        label_kwargs={"fontsize": 6},
        title_kwargs={"fontsize": 6},
    )
    fig_all.suptitle(
        f"DFT-slab dynesty — all varying parameters\n({note})",
        fontsize=11,
        y=1.01,
    )
    stem_all = out_dir / "corner_all"
    fig_all.savefig(stem_all.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig_all.savefig(stem_all.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_all)
    written.append(stem_all.with_suffix(".png"))
    return written


def export_parameter_table(post, status, names, out_dir):
    rows = []
    for p in post["params"]:
        rows.append(
            {
                "name": p["name"],
                "mle": p["mle"],
                "mean": p["mean"],
                "median": p["median"],
                "q16": p["q16"],
                "q84": p["q84"],
                "err_minus": p["median"] - p["q16"],
                "err_plus": p["q84"] - p["median"],
            }
        )
    csv_path = out_dir / "parameters.csv"
    header = "name,mle,mean,median,q16,q84,err_minus,err_plus,median_pm"
    lines = [header]
    for r in rows:
        pretty = f"{r['median']:.6g} -{r['err_minus']:.6g} +{r['err_plus']:.6g}"
        lines.append(
            f"{r['name']},{r['mle']:.8g},{r['mean']:.8g},{r['median']:.8g},"
            f"{r['q16']:.8g},{r['q84']:.8g},{r['err_minus']:.8g},"
            f"{r['err_plus']:.8g},{pretty}"
        )
    csv_path.write_text("\n".join(lines) + "\n")
    md = [
        "# DFT-slab dynesty posterior parameters (unlocked)",
        "",
        f"- logZ = {status['logz']:.4f} ± {status['logzerr']:.4f}",
        f"- ESS ≈ {post['ess']:.0f}",
        f"- logL_max (NS) = {post['logl_max']:.4f}",
        f"- n_samples = {post['n_samples']}",
        "",
        "| Parameter | MLE | Median | −σ (16%) | +σ (84%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| `{r['name']}` | {r['mle']:.5g} | {r['median']:.5g} | "
            f"{r['err_minus']:.5g} | {r['err_plus']:.5g} |"
        )
    (out_dir / "parameters.md").write_text("\n".join(md) + "\n")
    (out_dir / "parameters.json").write_text(
        json.dumps(
            {
                "logz": status["logz"],
                "logzerr": status["logzerr"],
                "ess": post["ess"],
                "logl_max": post["logl_max"],
                "names": names,
                "parameters": rows,
            },
            indent=2,
        )
    )
    return csv_path


def export_fit_quality(objective, post, status, out_dir):
    mle = {p["name"]: float(p["mle"]) for p in post["params"]}
    med = {p["name"]: float(p["median"]) for p in post["params"]}
    ds = objective._dataset
    q = np.asarray(ds.q, dtype=float)
    r = np.asarray(ds.r, dtype=float)
    r_err = np.asarray(ds.r_err, dtype=float)
    pol = np.asarray(ds.pol)
    e_arr = np.asarray(ds.energy, dtype=float)

    set_varying(objective, mle)
    objective.refresh_parameters()
    nll_mle = float(objective.nll())
    mle_curves = {}
    for energy in E_RES:
        for pol_name in ("s", "p"):
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq = np.sort(q[m])
            mle_curves[(float(energy), pol_name)] = reflectivity(
                objective.model, qq, energy, pol_name
            )

    set_varying(objective, med)
    objective.refresh_parameters()
    nll_med = float(objective.nll())

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
    chi2_all = []
    for i, energy in enumerate(E_RES):
        row = {"energy": float(energy)}
        for col, pol_name in enumerate(("s", "p")):
            ax = axes[i, col]
            m = (pol == pol_name) & np.isclose(e_arr, energy)
            qq, rr, ee = q[m], r[m], r_err[m]
            order = np.argsort(qq)
            qq, rr, ee = qq[order], rr[order], ee[order]
            r_med = reflectivity(objective.model, qq, energy, pol_name)
            r_mle = mle_curves[(float(energy), pol_name)]
            yd, ed = TRANSFORM(qq, rr, ee)
            y_med, _ = TRANSFORM(qq, r_med)
            y_mle, _ = TRANSFORM(qq, r_mle)
            rms = rms_logy(qq, rr, ee, r_med)
            row[f"rms_{pol_name}"] = rms
            ed_arr = np.asarray(ed, dtype=float)
            ok = np.isfinite(yd) & np.isfinite(y_med) & (ed_arr > 0)
            chi2_all.append(((yd[ok] - y_med[ok]) / ed_arr[ok]) ** 2)
            ax.errorbar(qq, yd, yerr=ed, **data_kw)
            ax.plot(qq, y_mle, color=COLOR_MLE, lw=1.05, ls="--", zorder=2)
            ax.plot(qq, y_med, color=COLOR_MED, lw=1.25, zorder=3)
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
    axes[0, 0].plot([], [], "o", ms=3.5, color=COLOR_DATA, label="data")
    axes[0, 0].plot([], [], color=COLOR_MED, lw=1.4, label="posterior median")
    axes[0, 0].plot([], [], color=COLOR_MLE, lw=1.2, ls="--", label="polish MLE")
    axes[0, 0].legend(loc="lower left", frameon=False, handlelength=1.8)
    fig.suptitle(
        r"DFT-slab ZnPc — dynesty posterior median (unlocked, log$Y$)"
        f"\nlogZ={status['logz']:.2f}±{status['logzerr']:.2f}"
        f"  ·  ESS={post['ess']:.0f}"
        f"  ·  overall RMS={overall_rms:.2f}"
        f"  ·  nll(med)={nll_med:.1f} / nll(MLE)={nll_mle:.1f}",
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
            r_med = reflectivity(objective.model, qq, energy, pol_name)
            yd, ed = TRANSFORM(qq, rr, ee)
            y_med, _ = TRANSFORM(qq, r_med)
            ed_arr = np.asarray(ed, dtype=float)
            resid = (yd - y_med) / np.where(ed_arr > 0, ed_arr, np.nan)
            ax.axhline(0.0, color="0.75", lw=0.7)
            ax.axhline(2.0, color="0.85", lw=0.5, ls=":")
            ax.axhline(-2.0, color="0.85", lw=0.5, ls=":")
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
        rf"DFT-slab dynesty median — normalized log$Y$ residuals  ·  RMS={overall_rms:.2f}",
        fontsize=10,
    )
    stem_r = out_dir / "fit_quality_residuals"
    fig_r.savefig(stem_r.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig_r.savefig(stem_r.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_r)
    meta = {
        "overall_rms": overall_rms,
        "nll_median": nll_med,
        "nll_mle": nll_mle,
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
    return meta


def main() -> None:
    args = parse_args()
    run = resolve_run(args)
    contract = RUN_CONTRACT[run]
    full_dir = models_fitting_results / "eres_dynesty_gpu" / RUN_DIRS[run] / "dft_slab"
    out_dir = full_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={out_dir}  run={run}  contract={contract}")

    helpers = _load_helpers()
    objective = helpers.load_slab(contract=contract)
    objective.model.device = "cpu"
    objective.model.parallel = True
    objective.transform = TRANSFORM
    objective.anisotropy_weight = helpers.ANISOTROPY_WEIGHT
    if hasattr(objective, "normalization"):
        objective.normalization = helpers.NORMALIZATION
    if hasattr(objective, "nc_constraint"):
        objective.nc_constraint = True
    objective._refresh_data_transform_cache()
    objective.refresh_parameters()

    results, names, samples, wt, post, status = load_posterior(full_dir)
    name_index = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(1)
    draws = weighted_resample(samples, wt, N_PROFILE_DRAWS, rng)
    bands = profile_quantiles(objective, draws, names, name_index)
    stem_prof = out_dir / "depth_profile_posterior"
    plot_depth_profile(bands, stem_prof, status, post)
    np.savez_compressed(
        out_dir / "depth_profile_posterior.npz",
        z_edges=bands["z_edges"],
        rho_p16=bands["rho_p16"],
        rho_p50=bands["rho_p50"],
        rho_p84=bands["rho_p84"],
        alpha_p16=bands["alpha_p16"],
        alpha_p50=bands["alpha_p50"],
        alpha_p84=bands["alpha_p84"],
        L_A=bands["L_A"],
    )
    log(f"wrote {stem_prof.with_suffix('.png')}")

    for p in export_corners(results, names, samples, wt, post, out_dir):
        log(f"wrote {p}")
    csv_path = export_parameter_table(post, status, names, out_dir)
    log(f"wrote {csv_path}")
    fit_meta = export_fit_quality(objective, post, status, out_dir)

    med = {p["name"]: p["median"] for p in post["params"]}
    present = [n for n in STRUCTURE_KEYS if n in med]
    summary = {
        "run": run,
        "contract": contract,
        "depth_profile_png": str(stem_prof.with_suffix(".png").resolve()),
        "corner_structure_png": str((out_dir / "corner_structure.png").resolve()),
        "corner_all_png": str((out_dir / "corner_all.png").resolve()),
        "parameters_csv": str(csv_path.resolve()),
        "parameters_md": str((out_dir / "parameters.md").resolve()),
        "fit_quality": fit_meta,
        "L_A": bands["L_A"],
        "structure_median": {n: med[n] for n in present},
    }
    (out_dir / "exports_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_dir / 'exports_summary.json'}")


if __name__ == "__main__":
    main()

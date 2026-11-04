"""Bookended dynesty exports: depth-profile bands, corner plots, parameter table.

Uses weighted posterior samples from
``eres_dynesty_gpu/{full,full_unlocked,full_polish_contract}/bookended``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils import models_fitting_results
from utils.graded_objective import (
    extract_graded_fit_context,
    load_bookended_multienergy_logY_polish,
)
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

BOOKENDED_DIR = models_fitting_results / "bookended_multienergy_logY_polish"
ENERGY = 283.7
COLOR = "#C0392B"
N_PROFILE_DRAWS = 600
N_DEPTH = 400

RUN_DIRS = {
    "full": "full",
    "unlocked": "full_unlocked",
    "polish_contract": "full_polish_contract",
    "open_contract": "full_open_contract",
    "profile_contract": "full_profile_contract",
}

PROFILE_NAMES = (
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
PROFILE_LABELS = (
    r"$L$ (Å)",
    r"$\sigma_{\mathrm{surf}}$ (Å)",
    r"$\rho_{\mathrm{bulk}}$",
    r"$\rho_{\mathrm{si}}$",
    r"$\rho_{\mathrm{vac}}$",
    r"$\tau_{\mathrm{si}}$ (Å)",
    r"$\tau_{\mathrm{vac}}$ (Å)",
    r"$\alpha_{\mathrm{bulk}}$",
    r"$\alpha_{\mathrm{si}}$",
    r"$\alpha_{\mathrm{vac}}$",
)
OXIDE_NAMES = ("Oxide_283.7_thick", "Oxide_283.7_rough")
OXIDE_LABELS = (r"$d_{\mathrm{ox}}$ (Å)", r"$\sigma_{\mathrm{ox}}$ (Å)")
STRUCTURE_NAMES = PROFILE_NAMES + OXIDE_NAMES
STRUCTURE_LABELS = PROFILE_LABELS + OXIDE_LABELS
ORIENT_NAMES = ("alpha_bulk", "alpha_si", "alpha_vac")
ORIENT_LABELS = (
    r"$\alpha_{\mathrm{bulk}}$ (rad)",
    r"$\alpha_{\mathrm{si}}$ (rad)",
    r"$\alpha_{\mathrm{vac}}$ (rad)",
)
INSTRUMENT_SCALE_NAMES = tuple(
    f"scale_{pol}@{e}"
    for e in ("283p7eV", "284p2eV", "285p1eV", "287p0eV", "289p0eV")
    for pol in ("s", "p")
)
INSTRUMENT_THETA_NAMES = tuple(
    f"theta_offset_{pol}@{e}"
    for e in ("283p7eV", "284p2eV", "285p1eV", "287p0eV", "289p0eV")
    for pol in ("s", "p")
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        choices=tuple(RUN_DIRS),
        default="polish_contract",
        help="Which eres_dynesty_gpu result tree to export (default: polish_contract)",
    )
    p.add_argument(
        "--unlocked",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias: --unlocked -> --run unlocked; --no-unlocked -> --run full",
    )
    return p.parse_args()


def resolve_run(args: argparse.Namespace) -> str:
    if args.unlocked is None:
        return str(args.run)
    return "unlocked" if args.unlocked else "full"


def run_tag(run: str) -> str:
    return {
        "full": "locked",
        "unlocked": "unlocked",
        "polish_contract": "polish_contract",
        "open_contract": "open_contract",
        "profile_contract": "profile_contract",
    }[run]


def _rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
    wt = wt / wt.sum()
    return results, names, samples, wt, post, status


def weighted_resample(
    samples: np.ndarray, wt: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    idx = rng.choice(samples.shape[0], size=n, replace=True, p=wt)
    return samples[idx]


def nevot_croce_density(film, depth: np.ndarray, *, n_sigma: int = 5):
    total = float(film.total_thick.value)
    sigma = float(film.surface_roughness.value)
    if sigma <= 0.0:
        return np.asarray(film.local_density(depth), dtype=float)
    dz = float(depth[1] - depth[0])
    pad = int(np.ceil(n_sigma * sigma / dz))
    z_ext = np.concatenate(
        [
            depth[0] + dz * np.arange(-pad, 0),
            depth,
            depth[-1] + dz * np.arange(1, pad + 1),
        ]
    )
    rho_ext = np.empty_like(z_ext)
    below, above = z_ext < 0.0, z_ext > total
    inside = ~(below | above)
    rho_ext[below] = 0.0
    rho_ext[inside] = np.asarray(film.local_density(z_ext[inside]), dtype=float)
    rho_ext[above] = float(film.local_density(total))
    kernel = np.exp(-0.5 * (dz * np.arange(-pad, pad + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(rho_ext, kernel, mode="same")[pad : pad + depth.size]


def set_film_from_row(film, oxide, row: np.ndarray, name_index: dict[str, int]) -> None:
    for name in PROFILE_NAMES:
        if name in name_index:
            getattr(film, name).value = float(row[name_index[name]])
    if "Oxide_283.7_thick" in name_index:
        oxide.thick.value = float(row[name_index["Oxide_283.7_thick"]])
    if "Oxide_283.7_rough" in name_index:
        oxide.rough.value = float(row[name_index["Oxide_283.7_rough"]])


def profile_quantiles(
    film,
    oxide,
    samples_struct: np.ndarray,
    name_index: dict[str, int],
) -> dict:
    totals = []
    for row in samples_struct:
        set_film_from_row(film, oxide, row, name_index)
        totals.append(float(film.total_thick.value))
    l_med = float(np.median(totals))
    depth = np.linspace(0.0, l_med, N_DEPTH)
    s = depth / l_med
    alpha_stack = np.empty((samples_struct.shape[0], N_DEPTH), dtype=float)
    rho_stack = np.empty_like(alpha_stack)
    for i, row in enumerate(samples_struct):
        set_film_from_row(film, oxide, row, name_index)
        total = float(film.total_thick.value)
        z = s * total
        alpha_stack[i] = np.degrees(np.asarray(film.orientation(z), dtype=float))
        rho_stack[i] = nevot_croce_density(film, z)

    def qband(stack: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "p16": np.percentile(stack, 16, axis=0),
            "p50": np.percentile(stack, 50, axis=0),
            "p84": np.percentile(stack, 84, axis=0),
        }

    n_show = min(40, samples_struct.shape[0])
    return {
        "depth_A": depth,
        "z_over_L": s,
        "L_A": l_med,
        "orientation_deg": qband(alpha_stack),
        "density_g_cm3": qband(rho_stack),
        "orientation_draws": alpha_stack[:n_show],
        "density_draws": rho_stack[:n_show],
        "n_draws": int(samples_struct.shape[0]),
    }


def plot_depth_profile(
    bands: dict, stem: Path, status: dict, post: dict, *, tag: str
) -> None:
    _rc()
    s = bands["z_over_L"]
    a = bands["orientation_deg"]
    r = bands["density_g_cm3"]
    a_map = bands.get("orientation_map_deg")
    r_map = bands.get("density_map_g_cm3")
    a_draws = bands.get("orientation_draws")
    r_draws = bands.get("density_draws")
    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)

    def _panel(ax, p16, p50, p84, p_map, draws, ylab, title):
        if draws is not None:
            for row in draws:
                ax.plot(s, row, color=COLOR, lw=0.45, alpha=0.12, zorder=1)
        ax.fill_between(
            s, p16, p84, color=COLOR, alpha=0.40, lw=0, zorder=2, label="16–84%"
        )
        ax.plot(s, p16, color=COLOR, lw=0.8, alpha=0.9, zorder=3)
        ax.plot(s, p84, color=COLOR, lw=0.8, alpha=0.9, zorder=3)
        ax.plot(s, p50, color=COLOR, lw=1.9, zorder=4, label="posterior median")
        if p_map is not None:
            ax.plot(
                s,
                p_map,
                color="#1A5276",
                lw=1.4,
                ls="--",
                zorder=5,
                label="NS MAP (max logl)",
            )
        ax.axvline(1.0, color="0.45", ls=":", lw=0.9, alpha=0.7)
        ax.set_xlabel(r"$z / L$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.legend(frameon=False, loc="best", fontsize=7.5)

    _panel(
        ax_a,
        a["p16"],
        a["p50"],
        a["p84"],
        a_map,
        a_draws,
        r"$\gamma$ (deg)",
        "orientation",
    )
    _panel(
        ax_r,
        r["p16"],
        r["p50"],
        r["p84"],
        r_map,
        r_draws,
        r"$\rho$ (g/cm$^3$)",
        "density (NC-smoothed)",
    )

    w_a = float(np.nanmedian(a["p84"] - a["p16"]))
    w_r = float(np.nanmedian(r["p84"] - r["p16"]))
    fig.suptitle(
        rf"Bookended ZnPc — dynesty posterior depth profile ({tag})"
        f"\n$L={bands['L_A']:.1f}$ Å  ·  "
        f"{bands['n_draws']} weighted draws  ·  "
        f"band = 16–84% (med width $\\gamma$={w_a:.2f}°, $\\rho$={w_r:.3f})"
        f"  ·  logZ={status['logz']:.2f}±{status['logzerr']:.2f}"
        f"  ·  ESS={post['ess']:.0f}",
        fontsize=10,
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _corner_subset(
    samples: np.ndarray,
    wt: np.ndarray,
    names: list[str],
    labels: tuple[str, ...],
    truths: list[float],
    name_index: dict[str, int],
    stem: Path,
    title: str,
    *,
    map_truths: list[float] | None = None,
) -> Path:
    import corner

    idx = [name_index[n] for n in names if n in name_index]
    use_names = [n for n in names if n in name_index]
    if len(use_names) < 2:
        return stem.with_suffix(".png")
    use_labels = [labels[names.index(n)] for n in use_names]
    use_truths = [truths[name_index[n]] for n in use_names]
    fig = corner.corner(
        samples[:, idx],
        weights=wt,
        labels=use_labels,
        truths=use_truths,
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
        use_map = [map_truths[name_index[n]] for n in use_names]
        corner.overplot_lines(fig, use_map, color=COLOR, linestyle="--")
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def export_corners(
    results,
    names: list[str],
    samples: np.ndarray,
    wt: np.ndarray,
    post: dict,
    out_dir: Path,
    *,
    tag: str,
) -> list[Path]:
    from dynesty import plotting as dyplot

    _rc()
    written: list[Path] = []
    name_index = {n: i for i, n in enumerate(names)}
    med_truths = [
        next(p["median"] for p in post["params"] if p["name"] == n) for n in names
    ]
    map_truths = None
    if post.get("map"):
        map_truths = [
            float(post["map"].get(n, med_truths[i])) for i, n in enumerate(names)
        ]

    truth_note = (
        "solid = posterior median; dashed = NS MAP"
        if map_truths is not None
        else "solid = posterior median"
    )

    if sum(1 for n in INSTRUMENT_SCALE_NAMES if n in name_index) >= 2:
        scale_labels = tuple(n.replace("@", "\n") for n in INSTRUMENT_SCALE_NAMES)
        written.append(
            _corner_subset(
                samples,
                wt,
                list(INSTRUMENT_SCALE_NAMES),
                scale_labels,
                med_truths,
                name_index,
                out_dir / "corner_instruments_scales",
                f"Bookended dynesty ({tag}) — instrumental scales\n"
                f"(shaded 68/95%; {truth_note})",
                map_truths=map_truths,
            )
        )

    if sum(1 for n in INSTRUMENT_THETA_NAMES if n in name_index) >= 2:
        theta_labels = tuple(n.replace("@", "\n") for n in INSTRUMENT_THETA_NAMES)
        written.append(
            _corner_subset(
                samples,
                wt,
                list(INSTRUMENT_THETA_NAMES),
                theta_labels,
                med_truths,
                name_index,
                out_dir / "corner_instruments_theta",
                f"Bookended dynesty ({tag}) — instrumental θ offsets\n"
                f"(shaded 68/95%; {truth_note})",
                map_truths=map_truths,
            )
        )

    if sum(1 for n in PROFILE_NAMES if n in name_index) >= 2:
        written.append(
            _corner_subset(
                samples,
                wt,
                list(PROFILE_NAMES),
                PROFILE_LABELS,
                med_truths,
                name_index,
                out_dir / "corner_profile",
                f"Bookended dynesty ({tag}) — profile "
                r"($L$, $\sigma$, $\rho$, $\tau$, $\alpha$)"
                f"\n(shaded 68/95%; {truth_note})",
                map_truths=map_truths,
            )
        )

    if sum(1 for n in OXIDE_NAMES if n in name_index) >= 2:
        written.append(
            _corner_subset(
                samples,
                wt,
                list(OXIDE_NAMES),
                OXIDE_LABELS,
                med_truths,
                name_index,
                out_dir / "corner_oxide",
                f"Bookended dynesty ({tag}) — oxide thick/rough\n"
                f"(shaded 68/95%; {truth_note})",
                map_truths=map_truths,
            )
        )

    struct_present = [n for n in STRUCTURE_NAMES if n in name_index]
    if (
        2 <= len(struct_present) <= 4
        and sum(1 for n in PROFILE_NAMES if n in name_index) < 5
    ):
        written.append(
            _corner_subset(
                samples,
                wt,
                list(STRUCTURE_NAMES),
                STRUCTURE_LABELS,
                med_truths,
                name_index,
                out_dir / "corner_structure",
                f"Bookended dynesty ({tag}) — free structure\n"
                f"(shaded 68/95%; {truth_note})",
                map_truths=map_truths,
            )
        )

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
        f"Bookended dynesty ({tag}) — all varying parameters\n({truth_note})",
        fontsize=11,
        y=1.01,
    )
    stem_all = out_dir / "corner_all"
    fig_all.savefig(stem_all.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig_all.savefig(stem_all.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_all)
    written.append(stem_all.with_suffix(".png"))
    return written


def export_parameter_table(
    post: dict, status: dict, names: list[str], out_dir: Path, *, tag: str, run: str
) -> Path:
    rows = []
    for p in post["params"]:
        row = {
            "name": p["name"],
            "mle": p["mle"],
            "mean": p["mean"],
            "median": p["median"],
            "q16": p["q16"],
            "q84": p["q84"],
            "err_minus": p["median"] - p["q16"],
            "err_plus": p["q84"] - p["median"],
        }
        if post.get("map") and p["name"] in post["map"]:
            row["map"] = float(post["map"][p["name"]])
        rows.append(row)

    csv_path = out_dir / "parameters.csv"
    has_map = any("map" in r for r in rows)
    if has_map:
        header = (
            "name,mle,map,mean,median,q16,q84,err_minus,err_plus,"
            "median_minus_err_minus_plus_err_plus"
        )
    else:
        header = (
            "name,mle,mean,median,q16,q84,err_minus,err_plus,"
            "median_minus_err_minus_plus_err_plus"
        )
    lines = [header]
    for r in rows:
        pretty = f"{r['median']:.6g} -{r['err_minus']:.6g} +{r['err_plus']:.6g}"
        if has_map:
            map_v = r.get("map", float("nan"))
            lines.append(
                f"{r['name']},{r['mle']:.8g},{map_v:.8g},{r['mean']:.8g},"
                f"{r['median']:.8g},{r['q16']:.8g},{r['q84']:.8g},"
                f"{r['err_minus']:.8g},{r['err_plus']:.8g},{pretty}"
            )
        else:
            lines.append(
                f"{r['name']},{r['mle']:.8g},{r['mean']:.8g},{r['median']:.8g},"
                f"{r['q16']:.8g},{r['q84']:.8g},{r['err_minus']:.8g},"
                f"{r['err_plus']:.8g},{pretty}"
            )
    csv_path.write_text("\n".join(lines) + "\n")

    md_path = out_dir / "parameters.md"
    md = [
        f"# Bookended dynesty posterior parameters ({tag})",
        "",
        f"- run = `{run}`",
        f"- logZ = {status['logz']:.4f} ± {status['logzerr']:.4f}",
        f"- ESS ≈ {post['ess']:.0f}",
        f"- logL_max (NS) = {post['logl_max']:.4f}",
        f"- n_samples = {post['n_samples']}",
        "",
    ]
    if has_map:
        md.extend(
            [
                "| Parameter | Polish MLE | NS MAP | Median | −σ (16%) | +σ (84%) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for r in rows:
            md.append(
                f"| `{r['name']}` | {r['mle']:.5g} | {r.get('map', float('nan')):.5g} | "
                f"{r['median']:.5g} | {r['err_minus']:.5g} | {r['err_plus']:.5g} |"
            )
    else:
        md.extend(
            [
                "| Parameter | MLE | Median | −σ (16%) | +σ (84%) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for r in rows:
            md.append(
                f"| `{r['name']}` | {r['mle']:.5g} | {r['median']:.5g} | "
                f"{r['err_minus']:.5g} | {r['err_plus']:.5g} |"
            )
    md_path.write_text("\n".join(md) + "\n")

    json_path = out_dir / "parameters.json"
    json_path.write_text(
        json.dumps(
            {
                "run": run,
                "tag": tag,
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


def main() -> None:
    args = parse_args()
    run = resolve_run(args)
    tag = run_tag(run)
    full_dir = models_fitting_results / "eres_dynesty_gpu" / RUN_DIRS[run] / "bookended"
    out_dir = full_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={out_dir}  run={run}  tag={tag}")

    results, names, samples, wt, post, status = load_posterior(full_dir)
    name_index = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(1)

    polish = load_bookended_multienergy_logY_polish(BOOKENDED_DIR / "fit.pkl")
    ctx = extract_graded_fit_context(polish.at(ENERGY))
    film = ctx.film
    obj = polish.at(ENERGY)
    oxide = obj.model.structure[2]

    draws = weighted_resample(samples, wt, N_PROFILE_DRAWS, rng)
    bands = profile_quantiles(film, oxide, draws, name_index)
    med = {p["name"]: p["median"] for p in post["params"]}
    row = np.zeros(len(names), dtype=float)
    for n, i in name_index.items():
        if n in med:
            row[i] = float(med[n])
    set_film_from_row(film, oxide, row, name_index)

    if post.get("map"):
        map_row = np.zeros(len(names), dtype=float)
        for n, i in name_index.items():
            if n in post["map"]:
                map_row[i] = float(post["map"][n])
            elif n in med:
                map_row[i] = float(med[n])
        set_film_from_row(film, oxide, map_row, name_index)
        total = float(film.total_thick.value)
        z = bands["z_over_L"] * total
        bands["orientation_map_deg"] = np.degrees(
            np.asarray(film.orientation(z), dtype=float)
        )
        bands["density_map_g_cm3"] = nevot_croce_density(film, z)
        set_film_from_row(film, oxide, row, name_index)

    stem_prof = out_dir / "depth_profile_posterior"
    plot_depth_profile(bands, stem_prof, status, post, tag=tag)
    np.savez_compressed(
        out_dir / "depth_profile_posterior.npz",
        depth_A=bands["depth_A"],
        z_over_L=bands["z_over_L"],
        orientation_p16=bands["orientation_deg"]["p16"],
        orientation_p50=bands["orientation_deg"]["p50"],
        orientation_p84=bands["orientation_deg"]["p84"],
        density_p16=bands["density_g_cm3"]["p16"],
        density_p50=bands["density_g_cm3"]["p50"],
        density_p84=bands["density_g_cm3"]["p84"],
        L_A=bands["L_A"],
    )
    log(f"wrote {stem_prof.with_suffix('.png')}")

    corners = export_corners(results, names, samples, wt, post, out_dir, tag=tag)
    for p in corners:
        log(f"wrote {p}")

    csv_path = export_parameter_table(post, status, names, out_dir, tag=tag, run=run)
    log(f"wrote {csv_path}")
    log(f"wrote {out_dir / 'parameters.md'}")

    present = [n for n in STRUCTURE_NAMES if n in med]
    summary = {
        "run": run,
        "tag": tag,
        "depth_profile_png": str(stem_prof.with_suffix(".png").resolve()),
        "corners": [str(p.resolve()) for p in corners],
        "corner_structure_png": str((out_dir / "corner_structure.png").resolve())
        if (out_dir / "corner_structure.png").exists()
        else None,
        "corner_orientation_png": str((out_dir / "corner_orientation.png").resolve())
        if (out_dir / "corner_orientation.png").exists()
        else None,
        "corner_scales_png": str((out_dir / "corner_scales.png").resolve())
        if (out_dir / "corner_scales.png").exists()
        else None,
        "corner_theta_png": str((out_dir / "corner_theta.png").resolve())
        if (out_dir / "corner_theta.png").exists()
        else None,
        "corner_all_png": str((out_dir / "corner_all.png").resolve())
        if (out_dir / "corner_all.png").exists()
        else None,
        "parameters_csv": str(csv_path.resolve()),
        "parameters_md": str((out_dir / "parameters.md").resolve()),
        "L_A": bands["L_A"],
        "n_profile_draws": bands["n_draws"],
        "structure_median": {n: med[n] for n in present},
        "structure_q16": {
            n: next(p["q16"] for p in post["params"] if p["name"] == n) for n in present
        },
        "structure_q84": {
            n: next(p["q84"] for p in post["params"] if p["name"] == n) for n in present
        },
    }
    (out_dir / "exports_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_dir / 'exports_summary.json'}")


if __name__ == "__main__":
    main()

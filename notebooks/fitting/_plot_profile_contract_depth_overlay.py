"""Overlay bookended vs DFT-slab posterior depth profiles (profile_contract).

Bookended density uses the existing surface Nevot-Croce convolution.
Slab density/orientation use erf-smoothed interfaces so vacuum/surface and
ZnPc interlayer roughnesses are visible (refnx slab ``rough`` = top interface).
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from utils import models_fitting_results
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

ROOT = models_fitting_results / "eres_dynesty_gpu" / "full_profile_contract"
N_DEPTH = 900
N_DRAWS = 500
COLOR_BOOK = "#1F618D"
COLOR_SLAB = "#C0392B"
_HERE = Path(__file__).resolve().parent


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


def _load_ed():
    path = _HERE / "_eres_dynesty_gpu.py"
    spec = importlib.util.spec_from_file_location("_eres_dynesty_gpu_overlay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_book_helpers():
    path = _HERE / "_plot_bookended_dynesty_exports.py"
    spec = importlib.util.spec_from_file_location("_plot_book_overlay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_results(run_dir: Path):
    with (run_dir / "results.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    results = blob["results"]
    names = list(blob["names"])
    post = json.loads((run_dir / "posterior_summary.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    samples = np.asarray(results.samples, dtype=float)
    logwt = np.asarray(results.logwt, dtype=float)
    logz = float(results.logz[-1])
    wt = np.exp(logwt - logz)
    wt /= wt.sum()
    return names, samples, wt, post, status


def weighted_resample(samples, wt, n, rng):
    idx = rng.choice(samples.shape[0], size=n, replace=True, p=wt)
    return samples[idx]


def erf_interfaces(
    z: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    sigmas: np.ndarray,
    *,
    value_before: float,
) -> np.ndarray:
    """Piecewise-constant profile with Nevot-Croce / erf interface roughness.

    ``edges[i]`` is the top of layer ``i`` (vacuum interface at ``edges[0]=0``),
    ``values[i]`` is the plateau in that layer, and ``sigmas[i]`` is the roughness
    of the top interface of layer ``i`` (refnx slab ``rough`` convention).
    """
    out = np.full(z.shape, float(value_before), dtype=float)
    prev = float(value_before)
    for edge, val, sig in zip(edges, values, sigmas, strict=True):
        jump = float(val) - prev
        if abs(jump) < 1e-15:
            prev = float(val)
            continue
        s = max(float(sig), 1e-6)
        out = out + 0.5 * jump * (1.0 + erf((z - float(edge)) / (s * np.sqrt(2.0))))
        prev = float(val)
    return out


def slab_layers(
    objective,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return film+oxide(+Si pad) rows for the roughness-aware profile.

    Substrate has ``thick=0`` in the stack but still owns the Si density and the
    oxide/Si roughness, so a finite Si plateau is appended for plotting only.
    """
    names: list[str] = []
    thick: list[float] = []
    rough: list[float] = []
    dens: list[float] = []
    alpha: list[float] = []
    substrate = None
    for comp in objective.model.structure:
        name = str(getattr(comp, "name", "")).lower()
        if name == "vacuum":
            continue
        if (
            name in {"substrate", "si"}
            and float(getattr(comp.thick, "value", 0.0) or 0.0) <= 0.0
        ):
            substrate = comp
            continue
        t = float(comp.thick.value)
        if t <= 0.0:
            continue
        names.append(str(getattr(comp, "name", name)))
        thick.append(t)
        rough.append(float(comp.rough.value))
        sld = getattr(comp, "sld", None)
        dens.append(
            float(sld.density.value)
            if sld is not None and hasattr(sld, "density")
            else np.nan
        )
        if sld is not None and hasattr(sld, "rotation"):
            alpha.append(float(sld.rotation.value))
        else:
            alpha.append(np.nan)
    if substrate is not None:
        names.append(str(getattr(substrate, "name", "substrate")))
        thick.append(40.0)
        rough.append(float(substrate.rough.value))
        dens.append(float(substrate.sld.density.value))
        alpha.append(np.nan)
    return (
        names,
        np.asarray(thick, dtype=float),
        np.asarray(rough, dtype=float),
        np.asarray(dens, dtype=float),
        np.asarray(alpha, dtype=float),
    )


def slab_profiles_on_grid(objective, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _names, thick, rough, dens, alpha = slab_layers(objective)
    edges = np.concatenate([[0.0], np.cumsum(thick[:-1])])
    rho = erf_interfaces(z, edges, dens, rough, value_before=0.0)
    # Orientation: vacuum has no director; ramp from 0 at the rough surface, then
    # erf jumps between ZnPc layers. Oxide/substrate keep the last ZnPc angle.
    alpha_fill = alpha.copy()
    last = np.nan
    for i, a in enumerate(alpha_fill):
        if np.isfinite(a):
            last = float(a)
        elif np.isfinite(last):
            alpha_fill[i] = last
    alpha_fill = np.where(np.isfinite(alpha_fill), alpha_fill, 0.0)
    gam = np.degrees(erf_interfaces(z, edges, alpha_fill, rough, value_before=0.0))
    return rho, gam


def bookended_profiles_on_grid(
    film, oxide, z: np.ndarray, book_helpers
) -> tuple[np.ndarray, np.ndarray]:
    total = float(film.total_thick.value)
    sigma_s = float(film.surface_roughness.value)
    # Density: NC surface blur on the film grid (allows z<0 vacuum pad), then
    # replace the substrate side with erf film→oxide→Si.
    rho = book_helpers.nevot_croce_density(film, z)
    dens_at_film_bot = float(film.local_density(total))
    ox_t = float(oxide.thick.value)
    ox_r = float(oxide.rough.value)
    sub_r = 1.2
    above = z > total
    if np.any(above):
        edges = np.asarray([total, total + ox_t], dtype=float)
        vals = np.asarray([2.2, 2.33], dtype=float)
        sigs = np.asarray([ox_r, sub_r], dtype=float)
        rho[above] = erf_interfaces(
            z[above], edges, vals, sigs, value_before=dens_at_film_bot
        )

    # Orientation: same surface Gaussian blur as density, film-only then hold.
    alpha_raw = np.degrees(
        np.asarray(film.orientation(np.clip(z, 0.0, total)), dtype=float)
    )
    if sigma_s <= 0.0:
        gam = alpha_raw
    else:
        dz = float(z[1] - z[0])
        pad = int(np.ceil(5.0 * sigma_s / dz))
        z_ext = np.concatenate(
            [z[0] + dz * np.arange(-pad, 0), z, z[-1] + dz * np.arange(1, pad + 1)]
        )
        a_ext = np.degrees(
            np.asarray(film.orientation(np.clip(z_ext, 0.0, total)), dtype=float)
        )
        # Vacuum side: no orientation → 0 before surface.
        a_ext = np.where(z_ext < 0.0, 0.0, a_ext)
        kernel = np.exp(-0.5 * (dz * np.arange(-pad, pad + 1) / sigma_s) ** 2)
        kernel /= kernel.sum()
        gam = np.convolve(a_ext, kernel, mode="same")[pad : pad + z.size]
    return rho, gam


def qband(stack: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "p16": np.percentile(stack, 16, axis=0),
        "p50": np.percentile(stack, 50, axis=0),
        "p84": np.percentile(stack, 84, axis=0),
    }


def main() -> None:
    ed = _load_ed()
    book_helpers = _load_book_helpers()
    rng = np.random.default_rng(1)

    book_obj = ed.load_bookended(contract="profile")
    slab_obj = ed.load_slab(contract="profile")
    book_obj.model.device = "cpu"
    slab_obj.model.device = "cpu"
    book_obj.model.parallel = False
    slab_obj.model.parallel = False

    film = book_obj.model.structure[1].profile
    oxide_b = book_obj.model.structure[2]

    b_names, b_samples, b_wt, b_post, b_status = load_results(ROOT / "bookended")
    s_names, s_samples, s_wt, s_post, s_status = load_results(ROOT / "dft_slab")
    b_idx = {n: i for i, n in enumerate(b_names)}
    s_idx = {n: i for i, n in enumerate(s_names)}
    b_draws = weighted_resample(b_samples, b_wt, N_DRAWS, rng)
    s_draws = weighted_resample(s_samples, s_wt, N_DRAWS, rng)

    # Common z grid covering both films + oxide + a vacuum pad for surface σ.
    med_book = {p["name"]: float(p["median"]) for p in b_post["params"]}
    med_slab = {p["name"]: float(p["median"]) for p in s_post["params"]}
    for name, val in med_book.items():
        if hasattr(film, name):
            getattr(film, name).value = val
    for p in slab_obj.varying_parameters():
        if p.name in med_slab:
            p.value = float(med_slab[p.name])
    slab_obj.refresh_parameters()

    L_book = float(film.total_thick.value) + float(oxide_b.thick.value)
    _n, th, rg, _d, _a = slab_layers(slab_obj)
    L_slab = float(np.sum(th))
    sigma_pad = max(
        float(med_book.get("surface_roughness", 12.0)),
        float(med_slab.get("ZnPc_surface_rough", 8.0)),
        12.0,
    )
    z_max = max(L_book, L_slab) + 2.0 * sigma_pad
    z = np.linspace(-3.0 * sigma_pad, z_max, N_DEPTH)

    rho_b = np.empty((N_DRAWS, N_DEPTH), dtype=float)
    gam_b = np.empty_like(rho_b)
    rho_s = np.empty_like(rho_b)
    gam_s = np.empty_like(rho_b)
    L_book_draws = np.empty(N_DRAWS, dtype=float)
    L_slab_draws = np.empty(N_DRAWS, dtype=float)

    for i, row in enumerate(b_draws):
        book_helpers.set_film_from_row(film, oxide_b, row, b_idx)
        L_book_draws[i] = float(film.total_thick.value)
        rho_b[i], gam_b[i] = bookended_profiles_on_grid(film, oxide_b, z, book_helpers)

    for i, row in enumerate(s_draws):
        vals = {n: float(row[j]) for n, j in s_idx.items()}
        for p in slab_obj.varying_parameters():
            if p.name in vals:
                p.value = vals[p.name]
        slab_obj.refresh_parameters()
        _n, th, _rg, _d, _a = slab_layers(slab_obj)
        # thick layout: ZnPc layers + oxide + synthetic Si pad → ZnPc = all but last 2
        L_slab_draws[i] = float(np.sum(th[:-2])) if th.size >= 2 else float(np.sum(th))
        rho_s[i], gam_s[i] = slab_profiles_on_grid(slab_obj, z)

    bands = {
        "z_A": z,
        "book_rho": qband(rho_b),
        "book_gam": qband(gam_b),
        "slab_rho": qband(rho_s),
        "slab_gam": qband(gam_s),
        "L_book_film_A": float(np.median(L_book_draws)),
        "L_slab_znpc_A": float(np.median(L_slab_draws)),
        "oxide_A": float(oxide_b.thick.value),
        "n_draws": N_DRAWS,
    }

    # Interface markers at posterior median slab edges (roughness-aware labels).
    for p in slab_obj.varying_parameters():
        if p.name in med_slab:
            p.value = float(med_slab[p.name])
    slab_obj.refresh_parameters()
    names, thick, rough, dens, alpha = slab_layers(slab_obj)
    edges = np.concatenate([[0.0], np.cumsum(thick[:-1])])
    interface_notes = list(
        zip(
            names,
            edges,
            rough,
            dens,
            np.degrees(np.where(np.isfinite(alpha), alpha, np.nan)),
            strict=True,
        )
    )

    _rc()
    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

    def panel(ax, book, slab, ylab, title):
        ax.fill_between(
            z, book["p16"], book["p84"], color=COLOR_BOOK, alpha=0.22, lw=0, zorder=1
        )
        ax.fill_between(
            z, slab["p16"], slab["p84"], color=COLOR_SLAB, alpha=0.22, lw=0, zorder=2
        )
        ax.plot(z, book["p50"], color=COLOR_BOOK, lw=1.9, zorder=4, label="bookended")
        ax.plot(z, slab["p50"], color=COLOR_SLAB, lw=1.9, zorder=5, label="DFT-slab")
        ax.axvline(0.0, color="0.55", ls=":", lw=0.9, alpha=0.85)
        ax.axvline(
            bands["L_book_film_A"],
            color=COLOR_BOOK,
            ls="--",
            lw=0.9,
            alpha=0.7,
        )
        ax.axvline(
            bands["L_slab_znpc_A"],
            color=COLOR_SLAB,
            ls="--",
            lw=0.9,
            alpha=0.7,
        )
        ax.set_xlabel(r"$z$ (Å)  (0 = vacuum / film surface)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xlim(z[0], z[-1])
        ax.legend(frameon=False, loc="best", fontsize=8)

    panel(ax_a, bands["book_gam"], bands["slab_gam"], r"$\gamma$ (deg)", "orientation")
    panel(ax_r, bands["book_rho"], bands["slab_rho"], r"$\rho$ (g/cm$^3$)", "density")

    # Annotate ZnPc interlayer roughnesses on density panel.
    y0, y1 = ax_r.get_ylim()
    for name, edge, sig, dens_i, gam_i in interface_notes:
        low = name.lower()
        if "oxide" in low or "si" in low or "sub" in low:
            continue
        ax_r.axvline(edge, color=COLOR_SLAB, ls=":", lw=0.7, alpha=0.55, zorder=0)
        ax_r.text(
            edge + 0.6,
            y1 - 0.04 * (y1 - y0),
            rf"$\sigma$={sig:.1f}",
            color=COLOR_SLAB,
            fontsize=6.5,
            rotation=90,
            va="top",
            ha="left",
            alpha=0.9,
        )

    ax_r.text(
        0.02,
        0.04,
        (
            r"book $\sigma_{\mathrm{surf}}$="
            f"{med_book['surface_roughness']:.1f} Å\n"
            r"slab $\sigma_{\mathrm{surf}}$="
            f"{med_slab['ZnPc_surface_rough']:.1f} Å"
        ),
        transform=ax_r.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.2",
        linespacing=1.25,
    )

    fig.suptitle(
        "Profile-contract depth profiles — bookended vs DFT-slab "
        "(posterior median, 16–84%)\n"
        f"bookended logZ={b_status['logz']:.1f}±{b_status['logzerr']:.2f}  ·  "
        f"slab logZ={s_status['logz']:.1f}±{s_status['logzerr']:.2f}  ·  "
        f"$L_{{\\mathrm{{book}}}}$={bands['L_book_film_A']:.1f} Å  ·  "
        f"$L_{{\\mathrm{{slab}}}}$={bands['L_slab_znpc_A']:.1f} Å  ·  "
        "erf / Nevot-Croce roughness shown",
        fontsize=10,
    )

    out = ROOT / "depth_profile_overlay"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        ROOT / "depth_profile_overlay.npz",
        z_A=z,
        book_rho_p16=bands["book_rho"]["p16"],
        book_rho_p50=bands["book_rho"]["p50"],
        book_rho_p84=bands["book_rho"]["p84"],
        book_gam_p16=bands["book_gam"]["p16"],
        book_gam_p50=bands["book_gam"]["p50"],
        book_gam_p84=bands["book_gam"]["p84"],
        slab_rho_p16=bands["slab_rho"]["p16"],
        slab_rho_p50=bands["slab_rho"]["p50"],
        slab_rho_p84=bands["slab_rho"]["p84"],
        slab_gam_p16=bands["slab_gam"]["p16"],
        slab_gam_p50=bands["slab_gam"]["p50"],
        slab_gam_p84=bands["slab_gam"]["p84"],
        L_book_film_A=bands["L_book_film_A"],
        L_slab_znpc_A=bands["L_slab_znpc_A"],
    )
    meta = {
        "png": str(out.with_suffix(".png").resolve()),
        "pdf": str(out.with_suffix(".pdf").resolve()),
        "n_draws": N_DRAWS,
        "bookended_logz": b_status["logz"],
        "dft_slab_logz": s_status["logz"],
        "L_book_film_A": bands["L_book_film_A"],
        "L_slab_znpc_A": bands["L_slab_znpc_A"],
        "surface_rough_book_A": med_book["surface_roughness"],
        "surface_rough_slab_A": med_slab["ZnPc_surface_rough"],
        "slab_interfaces": [
            {
                "name": n,
                "z_top_A": float(e),
                "sigma_A": float(s),
                "density": float(d),
                "gamma_deg": float(g) if np.isfinite(g) else None,
            }
            for n, e, s, d, g in interface_notes
        ],
    }
    (ROOT / "depth_profile_overlay.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {out.with_suffix('.png')}")
    log(f"wrote {out.with_suffix('.pdf')}")
    log(json.dumps(meta["slab_interfaces"], indent=2))


if __name__ == "__main__":
    main()

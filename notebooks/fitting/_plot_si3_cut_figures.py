"""Per-cut SI3 Dynesty checkpoint figures: corner, R(q) with bands, depth profile."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import resample_equal
from matplotlib.patches import Patch, Rectangle
from refnx.analysis import Transform
from scipy.stats import gaussian_kde

from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

_HERE = Path(__file__).resolve().parent
TRANSFORM = Transform("logY")
N_BAND_DRAWS = 200
N_SPAGHETTI = 30
N_DEPTH = 400
N_PROFILE_DRAWS = 800
CUT_TEX = {
    "q_hi": r"q_{\mathrm{hi}}",
    "mid": r"\mathrm{mid}",
    "q_lo": r"q_{\mathrm{lo}}",
}


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


si3 = _load("si3_fig", _HERE / "_bookended_q_truncation_dynesty_si3.py")
dexp = _load("dexp_fig", _HERE / "_plot_bookended_dynesty_exports.py")


def log(msg: str) -> None:
    print(msg, flush=True)


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


def _contour_level_2d(density: np.ndarray, level: float) -> float:
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    return float(flat[int(np.searchsorted(csum, level))])


def load_cut_payload(cut: str) -> dict[str, Any]:
    cut_dir = si3.OUT_ROOT / cut
    with (cut_dir / "results.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    summary = json.loads((cut_dir / "summary.json").read_text())
    post = json.loads((cut_dir / "posterior_summary.json").read_text())
    names = list(blob["names"])
    results = blob["results"]
    samples = np.asarray(results.samples, dtype=float)
    logwt = np.asarray(results.logwt, dtype=float)
    logz = float(results.logz[-1])
    wt = np.exp(logwt - logz)
    wt = wt / wt.sum()
    return {
        "cut": cut,
        "cut_dir": cut_dir,
        "names": names,
        "samples": samples,
        "wt": wt,
        "summary": summary,
        "post": post,
        "map": {k: float(v) for k, v in summary["map"].items()},
    }


def equal_draws(
    samples: np.ndarray,
    wt: np.ndarray,
    n: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eq = resample_equal(samples, wt, rstate=rng)
    if eq.shape[0] > n:
        eq = eq[rng.choice(eq.shape[0], n, replace=False)]
    return np.asarray(eq, dtype=float)


def set_si3(film: Any, row: np.ndarray, names: list[str]) -> None:
    idx = {n: names.index(n) for n in si3.SI_KEYS}
    film.alpha_si.value = float(row[idx["alpha_si"]])
    film.density_si.value = float(row[idx["density_si"]])
    film.tau_si.value = float(row[idx["tau_si"]])


def plot_corner_one(payload: dict[str, Any], out_stem: Path) -> Path:
    _rc()
    cut = payload["cut"]
    color = si3.CUT_COLORS[cut]
    eq = equal_draws(payload["samples"], payload["wt"], 20_000, seed=1)
    name_idx = [payload["names"].index(k) for k in si3.SI_KEYS]
    samp = eq[:, name_idx].copy()
    samp[:, 0] = np.rad2deg(samp[:, 0])
    prior = {
        k: (si3._disp_si(k, lo), si3._disp_si(k, hi))
        for k, (lo, hi) in si3.SI3_BOUNDS.items()
    }
    lims: dict[str, tuple[float, float]] = {}
    for i, k in enumerate(si3.SI_KEYS):
        plo, phi = prior[k]
        col = samp[:, i]
        lo = min(plo, float(np.percentile(col, 1)))
        hi = max(phi, float(np.percentile(col, 99)))
        pad = 0.04 * (hi - lo + 1e-12)
        lims[k] = (lo - pad, hi + pad)

    med = np.median(samp, axis=0)
    q16 = np.percentile(samp, 16, axis=0)
    q84 = np.percentile(samp, 84, axis=0)
    n = 3
    fig, axes = plt.subplots(
        n, n, figsize=(6.4, 6.2), gridspec_kw={"hspace": 0.08, "wspace": 0.08}
    )
    grid_n = 120
    prior_face = "#b0b0b0"
    prior_alpha = 0.28
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ki, kj = si3.SI_KEYS[i], si3.SI_KEYS[j]
            if j > i:
                ax.axis("off")
                continue
            plo_i, phi_i = prior[ki]
            plo_j, phi_j = prior[kj]
            if i == j:
                ax.axvspan(
                    plo_i, phi_i, color=prior_face, alpha=prior_alpha, zorder=0, lw=0
                )
                x = np.linspace(lims[ki][0], lims[ki][1], 400)
                y = gaussian_kde(samp[:, i])(x)
                ax.plot(x, y, color=color, lw=1.6, zorder=3)
                lo68, hi68 = q16[i], q84[i]
                m = (x >= lo68) & (x <= hi68)
                ax.fill_between(
                    x[m], 0.0, y[m], color=color, alpha=0.28, zorder=2, lw=0
                )
                ax.axvline(med[i], color=color, ls="--", lw=1.0, zorder=4)
                ax.set_xlim(lims[ki])
                ax.set_ylim(0.0, float(y.max()) * 1.15)
                ax.set_yticks([])
                if i < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(si3.SI_LABELS[i])
            else:
                ax.add_patch(
                    Rectangle(
                        (plo_j, plo_i),
                        phi_j - plo_j,
                        phi_i - plo_i,
                        facecolor=prior_face,
                        edgecolor="none",
                        alpha=prior_alpha,
                        zorder=0,
                    )
                )
                xx, yy = samp[:, j], samp[:, i]
                kde = gaussian_kde(np.vstack([xx, yy]))
                xg = np.linspace(lims[kj][0], lims[kj][1], grid_n)
                yg = np.linspace(lims[ki][0], lims[ki][1], grid_n)
                X, Y = np.meshgrid(xg, yg)
                dens = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                lev68 = _contour_level_2d(dens, 0.68)
                lev95 = _contour_level_2d(dens, 0.95)
                ax.contourf(
                    X,
                    Y,
                    dens,
                    levels=[lev95, lev68, dens.max()],
                    colors=[color, color],
                    alpha=0.18,
                    zorder=2,
                )
                ax.contour(
                    X,
                    Y,
                    dens,
                    levels=[lev95, lev68],
                    colors=[color],
                    linewidths=[1.0, 1.5],
                    zorder=4,
                )
                ax.plot(med[j], med[i], "o", ms=3.0, color=color, zorder=5)
                ax.set_xlim(lims[kj])
                ax.set_ylim(lims[ki])
                if i < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(si3.SI_LABELS[j])
                if j > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(si3.SI_LABELS[i])

    q_cut = float(payload["summary"]["q_cut"])
    d = float(
        payload["summary"].get(
            "d_2pi_over_q",
            si3.diffraction_length_A(float(payload["summary"]["q_cut"])),
        )
    )
    handles = [
        Patch(
            facecolor=prior_face,
            edgecolor="none",
            alpha=prior_alpha,
            label="prior bounds",
        ),
        Patch(
            facecolor=color,
            edgecolor=color,
            alpha=0.45,
            label=rf"{cut}: $q_{{\mathrm{{cut}}}}$={q_cut:.3f} ($d$={d:.1f} Å)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        fontsize=8,
        title="SI3 Dynesty 68/95%",
        title_fontsize=8,
    )
    fig.suptitle(
        rf"SI3 ${CUT_TEX[cut]}$ checkpoint"
        f"\n"
        rf"$\alpha$={med[0]:.1f}$^\circ$ [{q16[0]:.1f}, {q84[0]:.1f}]  ·  "
        rf"$\rho$={med[1]:.2f} [{q16[1]:.2f}, {q84[1]:.2f}]  ·  "
        rf"$\tau$={med[2]:.1f} [{q16[2]:.1f}, {q84[2]:.1f}] Å",
        fontsize=10,
        y=1.02,
    )
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out_stem.with_suffix('.png')}")
    return out_stem.with_suffix(".png")


def plot_reflectivity(payload: dict[str, Any], out_stem: Path) -> Path:
    _rc()
    cut = payload["cut"]
    color = si3.CUT_COLORS[cut]
    q_cut = float(payload["summary"]["q_cut"])
    fig4 = si3.load_fig4_summary()
    objectives = si3.cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in si3.ENERGIES}
    si3.apply_si3_seed(obj_map, fig4)
    train = si3.base.build_logY_objective(obj_map, q_min=si3.base.Q_MIN, q_max=q_cut)
    train.model.device = "cpu"
    train.model.parallel = True
    if hasattr(train, "nc_constraint"):
        train.nc_constraint = True
    train._refresh_data_transform_cache()
    train.refresh_parameters()

    names = payload["names"]
    film = obj_map[si3.ANCHOR].model.structure.components[1]

    def apply_row(row: np.ndarray) -> None:
        set_si3(film, row, names)
        train.refresh_parameters()

    map_row = np.array([payload["map"][n] for n in names], dtype=float)
    apply_row(map_row)
    nll_map = float(train.nll())
    cur_map = si3.base.full_q_curves(obj_map, train, energy=si3.ANCHOR)

    draws = equal_draws(payload["samples"], payload["wt"], N_BAND_DRAWS, seed=11)
    band: dict[str, list[np.ndarray]] = {"s": [], "p": []}
    q_pol: dict[str, np.ndarray] = {}
    for pol in ("s", "p"):
        m = np.asarray(cur_map["pol"]) == pol
        q_pol[pol] = np.sort(cur_map["q"][m])

    for row in draws:
        apply_row(row)
        full = si3.base.build_logY_objective(
            obj_map, q_min=si3.base.Q_MIN, model=train.model
        )
        y_fit = np.asarray(full.generative(), dtype=float)
        ds = full._dataset
        q = np.asarray(ds.q, dtype=float)
        e = np.asarray(ds.energy, dtype=float)
        pol_arr = np.asarray(ds.pol)
        mE = np.isclose(e, float(si3.ANCHOR))
        for pol in ("s", "p"):
            m = mE & (pol_arr == pol)
            qq = q[m]
            order = np.argsort(qq)
            yf, _ = TRANSFORM(qq[order], y_fit[m][order])
            # interpolate onto fixed q grid from MAP data points
            band[pol].append(
                np.interp(q_pol[pol], qq[order], np.asarray(yf, dtype=float))
            )

    p16 = {p: np.percentile(np.vstack(v), 16, axis=0) for p, v in band.items()}
    p50 = {p: np.percentile(np.vstack(v), 50, axis=0) for p, v in band.items()}
    p84 = {p: np.percentile(np.vstack(v), 84, axis=0) for p, v in band.items()}
    p025 = {p: np.percentile(np.vstack(v), 2.5, axis=0) for p, v in band.items()}
    p975 = {p: np.percentile(np.vstack(v), 97.5, axis=0) for p, v in band.items()}
    rng = np.random.default_rng(3)
    spaghetti = rng.choice(len(draws), size=min(N_SPAGHETTI, len(draws)), replace=False)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.4), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    for ax, pol in zip(axes, ("s", "p"), strict=True):
        m = np.asarray(cur_map["pol"]) == pol
        qq = cur_map["q"][m]
        order = np.argsort(qq)
        ax.errorbar(
            qq[order],
            cur_map["logR"][m][order],
            yerr=cur_map["logR_err"][m][order],
            fmt="o",
            ms=2.0,
            lw=0.4,
            color="0.55",
            alpha=0.7,
            zorder=1,
            label="data",
        )
        qg = q_pol[pol]
        ax.fill_between(
            qg, p025[pol], p975[pol], color=color, alpha=0.14, lw=0, zorder=2
        )
        ax.fill_between(
            qg,
            p16[pol],
            p84[pol],
            color=color,
            alpha=0.32,
            lw=0,
            zorder=3,
            label="16–84%",
        )
        stack = np.vstack(band[pol])
        for j in spaghetti:
            ax.plot(qg, stack[int(j)], color=color, lw=0.4, alpha=0.15, zorder=3)
        ax.plot(qg, p50[pol], color=color, lw=1.2, ls=":", zorder=4, label="median")
        ax.plot(
            qq[order],
            cur_map["logR_fit"][m][order],
            color=color,
            lw=1.5,
            zorder=5,
            label=rf"MAP SI3 ($q \leq {q_cut:.3f}$)",
        )
        ax.axvline(q_cut, color=color, ls="--", lw=0.9, alpha=0.75, zorder=0)
        ax.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
        ax.set_title(rf"{pol}-pol")
        ax.set_xlim(left=0.0)
    axes[0].set_ylabel(r"$\log_{10} R$")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    logz = float(payload["summary"]["logz"])
    logzerr = float(payload["summary"]["logzerr"])
    fig.suptitle(
        rf"SI3 ${CUT_TEX[cut]}$ MAP + posterior @ 283.7 eV "
        rf"(nll={nll_map:.2f}, logZ={logz:.2f}$\pm${logzerr:.2f})"
        "\ngray data; bands = 16–84% / 2.5–97.5%; instruments frozen at Fig 4 polish",
        fontsize=10,
        y=1.03,
    )
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    np.savez_compressed(
        out_stem.with_suffix(".npz"),
        q_s=q_pol["s"],
        q_p=q_pol["p"],
        p16_s=p16["s"],
        p50_s=p50["s"],
        p84_s=p84["s"],
        p16_p=p16["p"],
        p50_p=p50["p"],
        p84_p=p84["p"],
        q_cut=q_cut,
        nll_map=nll_map,
    )
    plt.close(fig)
    log(f"wrote {out_stem.with_suffix('.png')}")
    return out_stem.with_suffix(".png")


def plot_depth(payload: dict[str, Any], out_stem: Path) -> Path:
    _rc()
    cut = payload["cut"]
    color = si3.CUT_COLORS[cut]
    fig4 = si3.load_fig4_summary()
    objectives = si3.cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in si3.ENERGIES}
    si3.apply_si3_seed(obj_map, fig4)
    _vac, film, _oxide, _sub = obj_map[si3.ANCHOR].model.structure.components

    L = float(film.total_thick.value)
    sigma_surf = float(film.surface_roughness.value)
    depth = np.linspace(0.0, L, N_DEPTH)
    s = depth / L
    eq = equal_draws(payload["samples"], payload["wt"], N_PROFILE_DRAWS, seed=7)
    names = payload["names"]
    alpha_stack = np.empty((eq.shape[0], N_DEPTH), dtype=float)
    rho_stack = np.empty_like(alpha_stack)
    for i, row in enumerate(eq):
        set_si3(film, row, names)
        alpha_stack[i] = np.degrees(np.asarray(film.orientation(depth), dtype=float))
        rho_stack[i] = dexp.nevot_croce_density(film, depth)

    set_si3(
        film,
        np.array([payload["map"][n] for n in names], dtype=float),
        names,
    )
    alpha_map = np.degrees(np.asarray(film.orientation(depth), dtype=float))
    rho_map = dexp.nevot_croce_density(film, depth)
    a16, a50, a84 = np.percentile(alpha_stack, [16, 50, 84], axis=0)
    r16, r50, r84 = np.percentile(rho_stack, [16, 50, 84], axis=0)

    fig, (ax_a, ax_r) = plt.subplots(
        1, 2, figsize=(7.4, 3.4), gridspec_kw={"wspace": 0.28}
    )
    step = max(1, N_PROFILE_DRAWS // 40)
    for row in alpha_stack[::step]:
        ax_a.plot(s, row, color=color, lw=0.4, alpha=0.10, zorder=1)
    for row in rho_stack[::step]:
        ax_r.plot(s, row, color=color, lw=0.4, alpha=0.10, zorder=1)
    ax_a.fill_between(
        s, a16, a84, color=color, alpha=0.38, lw=0, zorder=2, label="16–84%"
    )
    ax_a.plot(s, a16, color=color, lw=0.8, zorder=3)
    ax_a.plot(s, a84, color=color, lw=0.8, zorder=3)
    ax_a.plot(s, a50, color=color, lw=1.9, zorder=4, label="posterior median")
    ax_a.plot(s, alpha_map, color="#1A5276", lw=1.4, ls="--", zorder=5, label="MAP")
    ax_r.fill_between(
        s, r16, r84, color=color, alpha=0.38, lw=0, zorder=2, label="16–84%"
    )
    ax_r.plot(s, r16, color=color, lw=0.8, zorder=3)
    ax_r.plot(s, r84, color=color, lw=0.8, zorder=3)
    ax_r.plot(s, r50, color=color, lw=1.9, zorder=4, label="posterior median")
    ax_r.plot(s, rho_map, color="#1A5276", lw=1.4, ls="--", zorder=5, label="MAP")
    for ax in (ax_a, ax_r):
        ax.axvline(1.0, color="0.45", ls=":", lw=0.9, alpha=0.7)
        ax.set_xlabel(r"$z / L$")
        ax.set_xlim(0.0, 1.0)
        ax.legend(frameon=False, loc="best", fontsize=7.5)
    ax_a.set_ylabel(r"$\gamma$ (deg)")
    ax_a.set_title("orientation")
    ax_r.set_ylabel(r"$\rho$ (g/cm$^3$)")
    ax_r.set_title(
        rf"density (NC-smoothed, $\sigma_{{\mathrm{{surf}}}}$={sigma_surf:.2f} Å)"
    )
    w_a = float(np.nanmedian(a84 - a16))
    w_r = float(np.nanmedian(r84 - r16))
    summary = payload["summary"]
    fig.suptitle(
        rf"SI3 ${CUT_TEX[cut]}$ depth profile  "
        rf"($q_{{\mathrm{{cut}}}}$={summary['q_cut']:.3f}, $L$={L:.1f} Å)"
        f"\n{N_PROFILE_DRAWS} equal-weight draws; band = 16–84% "
        rf"(med width $\gamma$={w_a:.2f}°, $\rho$={w_r:.3f})"
        f"  ·  logZ={summary['logz']:.2f}±{summary['logzerr']:.2f}"
        "\nonly $\\alpha_{si}$, $\\rho_{si}$, $\\tau_{si}$ free; "
        "surface roughness frozen at Fig 4",
        fontsize=9.5,
        y=1.06,
    )
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    np.savez_compressed(
        out_stem.with_suffix(".npz"),
        z_over_L=s,
        depth_A=depth,
        L_A=L,
        sigma_surf_A=sigma_surf,
        alpha_p16=a16,
        alpha_p50=a50,
        alpha_p84=a84,
        alpha_map=alpha_map,
        rho_p16=r16,
        rho_p50=r50,
        rho_p84=r84,
        rho_map=rho_map,
    )
    plt.close(fig)
    log(f"wrote {out_stem.with_suffix('.png')}")
    return out_stem.with_suffix(".png")


def run_cut(cut: str, *, which: tuple[str, ...]) -> None:
    payload = load_cut_payload(cut)
    out = si3.DE_OPEN_DIR
    if "corner" in which:
        plot_corner_one(payload, out / f"corner_si_dynesty_si3_{cut}_checkpoint")
    if "reflectivity" in which:
        plot_reflectivity(payload, out / f"reflectivity_si3_{cut}_checkpoint")
    if "depth" in which:
        plot_depth(payload, out / f"depth_profile_si3_{cut}_checkpoint")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cut", choices=["q_hi", "mid", "q_lo", "all"], default="mid")
    p.add_argument(
        "--which",
        default="corner,reflectivity,depth",
        help="comma list: corner,reflectivity,depth",
    )
    args = p.parse_args(argv)
    cuts = list(si3.CUT_NAMES) if args.cut == "all" else [args.cut]
    which = tuple(x.strip() for x in args.which.split(",") if x.strip())
    for cut in cuts:
        log(f"=== figures for {cut}: {which}")
        run_cut(cut, which=which)


if __name__ == "__main__":
    main()

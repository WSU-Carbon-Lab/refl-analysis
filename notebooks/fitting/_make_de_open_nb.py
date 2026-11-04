"""One-shot helper to write bookended_q_truncation_diffraction_de_open.ipynb."""

from __future__ import annotations

import json
from pathlib import Path


def cell_md(text: str) -> dict:
    lines = text.splitlines()
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def cell_code(text: str) -> dict:
    lines = text.splitlines()
    source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [
        cell_md(
            """# Reflectivity diffraction-limit probe — open 12-param DE variance

Same $q$-truncation multi-seed DE protocol as the locked DE scan, but with **12 structural free parameters**: the 10 film bookend/bulk knobs plus oxide thick/rough. SiO$_2$ and Si densities and `energy_offset` stay fixed. Per-energy instruments remain free (`nvary = 32`).

Runner: `_bookended_q_truncation_diffraction_de_open.py` → `@models/fitting_results/bookended_q_truncation_diffraction_de_open/`."""
        ),
        cell_code(
            """from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import models_fitting_results

OUT = models_fitting_results / "bookended_q_truncation_diffraction_de_open"
LOCKED_DE = models_fitting_results / "bookended_q_truncation_diffraction_de"
rep = pd.read_parquet(OUT / "replicates.parquet")
agg = pd.read_parquet(OUT / "scan.parquet").sort_values("d_pi_over_q")
meta = json.loads((OUT / "summary.json").read_text())
d_full = float(meta["d_full"])
q_hi = float(meta["q_hi"])
q_lo = float(meta["q_lo"])
print("structural_free:", meta.get("structural_free"))
print(
    "nvary_probe:",
    meta.get("nvary_probe"),
    "unique nvary:",
    sorted(rep["nvary"].unique().tolist()),
)
agg"""
        ),
        cell_code(
            r"""plt.rcParams.update(
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

hold = agg["n_holdout"] > 0
fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.4), sharex=True, gridspec_kw={"hspace": 0.14})

ax = axes[0]
ax.errorbar(
    agg["d_pi_over_q"],
    agg["redchi_train_mean"],
    yerr=agg["redchi_train_std"],
    fmt="o-",
    ms=3.5,
    color="0.2",
    capsize=2,
    label="train (open DE mean±std)",
)
ax.errorbar(
    agg.loc[hold, "d_pi_over_q"],
    agg.loc[hold, "redchi_holdout_mean"],
    yerr=agg.loc[hold, "redchi_holdout_std"],
    fmt="s-",
    ms=3.5,
    color="#C0392B",
    capsize=2,
    label="holdout (open DE mean±std)",
)
if (LOCKED_DE / "scan.parquet").exists():
    lk = pd.read_parquet(LOCKED_DE / "scan.parquet").sort_values("d_pi_over_q")
    m = lk["n_holdout"] > 0
    ax.plot(
        lk.loc[m, "d_pi_over_q"],
        lk.loc[m, "redchi_holdout_mean"],
        ":",
        color="0.45",
        lw=1.0,
        label="locked DE holdout",
    )
ax.axvline(d_full, color="0.45", ls=":", lw=0.9, label=rf"$d_{{\mathrm{{lim}}}}$={d_full:.2f} Å")
ax.set_ylabel(r"red.$\chi^2$ / $\chi^2/n$")
ax.legend(frameon=False, fontsize=7)
ax.set_title("Open 12-param DE soft curves")

ax = axes[1]
ax.errorbar(
    agg["d_pi_over_q"],
    agg["rms_logR_train_mean"],
    yerr=agg["rms_logR_train_std"],
    fmt="o-",
    ms=3.5,
    color="0.2",
    capsize=2,
    label="train",
)
ax.errorbar(
    agg.loc[hold, "d_pi_over_q"],
    agg.loc[hold, "rms_logR_holdout_mean"],
    yerr=agg.loc[hold, "rms_logR_holdout_std"],
    fmt="s-",
    ms=3.5,
    color="#C0392B",
    capsize=2,
    label="holdout",
)
if (LOCKED_DE / "scan.parquet").exists():
    lk = pd.read_parquet(LOCKED_DE / "scan.parquet").sort_values("d_pi_over_q")
    m = lk["n_holdout"] > 0
    ax.plot(
        lk.loc[m, "d_pi_over_q"],
        lk.loc[m, "rms_logR_holdout_mean"],
        ":",
        color="0.45",
        lw=1.0,
        label="locked DE holdout",
    )
ax.axvline(d_full, color="0.45", ls=":", lw=0.9)
ax.set_xlabel(r"reflectivity diffraction length $d=\pi/q_{\mathrm{cut}}$ (Å)")
ax.set_ylabel(r"RMS $\log_{10} R$")
ax.legend(frameon=False, fontsize=7)
fig.savefig(OUT / "soft_curves_de.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "soft_curves_de.pdf", bbox_inches="tight")
fig"""
        ),
        cell_code(
            r"""fig2, axes = plt.subplots(2, 3, figsize=(8.4, 4.6), gridspec_kw={"hspace": 0.28, "wspace": 0.32})
panels = [
    ("tau_vac", r"$\tau_{\mathrm{vac}}$ (Å)", False),
    ("tau_si", r"$\tau_{\mathrm{si}}$ (Å)", False),
    ("total_thick", r"$t_{\mathrm{tot}}$ (Å)", False),
    ("surface_roughness", r"$\sigma_{\mathrm{surf}}$ (Å)", False),
    ("oxide_thick", r"$t_{\mathrm{SiO_2}}$ (Å)", False),
    ("oxide_rough", r"$\sigma_{\mathrm{SiO_2}}$ (Å)", False),
]
for ax, (col, ylab, _) in zip(axes.ravel(), panels, strict=True):
    ax.errorbar(
        agg["d_pi_over_q"],
        agg[f"{col}_mean"],
        yerr=agg[f"{col}_std"],
        fmt="o-",
        ms=3.0,
        color="0.2",
        capsize=2,
    )
    ax.axvline(d_full, color="0.45", ls=":", lw=0.9)
    ax.set_xlabel(r"$d=\pi/q_{\mathrm{cut}}$ (Å)")
    ax.set_ylabel(ylab)

fig2.suptitle("Open DE structural parameters mean ± std", fontsize=10, y=1.02)
fig2.savefig(OUT / "params_vs_d_de.png", dpi=300, bbox_inches="tight")
fig2.savefig(OUT / "params_vs_d_de.pdf", bbox_inches="tight")
fig2"""
        ),
        cell_code(
            r"""fig2b, axes = plt.subplots(2, 3, figsize=(8.4, 4.6), gridspec_kw={"hspace": 0.28, "wspace": 0.32})
panels = [
    ("density_vac", r"$\rho_{\mathrm{vac}}$", False),
    ("density_bulk", r"$\rho_{\mathrm{bulk}}$", False),
    ("density_si", r"$\rho_{\mathrm{si}}$ (film)", False),
    ("alpha_vac", r"$\alpha_{\mathrm{vac}}$ (deg)", True),
    ("alpha_bulk", r"$\alpha_{\mathrm{bulk}}$ (deg)", True),
    ("alpha_si", r"$\alpha_{\mathrm{si}}$ (deg)", True),
]
for ax, (col, ylab, as_deg) in zip(axes.ravel(), panels, strict=True):
    y = agg[f"{col}_mean"]
    ye = agg[f"{col}_std"]
    if as_deg:
        y = np.rad2deg(y)
        ye = np.rad2deg(ye)
    ax.errorbar(agg["d_pi_over_q"], y, yerr=ye, fmt="o-", ms=3.0, color="0.2", capsize=2)
    ax.axvline(d_full, color="0.45", ls=":", lw=0.9)
    ax.set_xlabel(r"$d=\pi/q_{\mathrm{cut}}$ (Å)")
    ax.set_ylabel(ylab)

fig2b.suptitle("Open DE densities / alphas mean ± std", fontsize=10, y=1.02)
fig2b.savefig(OUT / "params_density_alpha_de.png", dpi=300, bbox_inches="tight")
fig2b.savefig(OUT / "params_density_alpha_de.pdf", bbox_inches="tight")
fig2b"""
        ),
        cell_md(
            r"""## Reflectivity cascade over all $q_{\mathrm{cut}}$

Full 283.7 eV $\log_{10} R$ with truncated **open** DE fits colored by $q_{\mathrm{cut}}$ (best seed per cut). Prefers cached `reflectivity_q_cuts_curves.npz` if present."""
        ),
        cell_code(
            """import importlib.util
import sys

CACHE = OUT / "reflectivity_q_cuts_curves.npz"
REBUILD = False

agg_q = agg.sort_values("q_cut", ascending=False).reset_index(drop=True)
q_cuts = [float(q) for q in agg_q["q_cut"].to_numpy()]
best_seeds = [int(s) for s in agg_q["best_seed"].to_numpy()]
curve_by_cut: dict[float, dict] = {}

if CACHE.exists() and not REBUILD:
    z = np.load(CACHE, allow_pickle=True)
    q_cuts = [float(q) for q in np.asarray(z["q_cuts"], dtype=float)]
    if "best_seeds" in z.files:
        best_seeds = [int(s) for s in np.asarray(z["best_seeds"], dtype=int)]
    for i, q_cut in enumerate(q_cuts):
        curve_by_cut[q_cut] = {
            "q": np.asarray(z[f"q_{i}"], dtype=float),
            "logR": np.asarray(z[f"logR_{i}"], dtype=float),
            "logR_err": np.asarray(z[f"logR_err_{i}"], dtype=float),
            "logR_fit": np.asarray(z[f"logR_fit_{i}"], dtype=float),
            "pol": np.asarray(z[f"pol_{i}"]),
        }
    print(f"loaded cache {CACHE.name}: {len(q_cuts)} cuts")
else:
    _HERE = Path("notebooks/fitting")
    if not (_HERE / "_bookended_q_truncation_diffraction_de_open.py").exists():
        _HERE = Path(".")
    _path = (_HERE / "_bookended_q_truncation_diffraction_de_open.py").resolve()
    _spec = importlib.util.spec_from_file_location("_q_trunc_de_open", _path)
    assert _spec is not None and _spec.loader is not None
    deo = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = deo
    _spec.loader.exec_module(deo)

    de_kw = {
        **deo.DE_KW,
        "popsize": int(meta["de_kw"]["popsize"]),
        "maxiter": int(meta["de_kw"]["maxiter"]),
    }
    summary = deo.base.load_polish_summary()
    objectives = deo.cv.seed_objectives()
    obj_map = {float(e): objectives[float(e)] for e in deo.ENERGIES}
    for i, (q_cut, seed) in enumerate(zip(q_cuts, best_seeds, strict=True)):
        print(f"[{i + 1}/{len(q_cuts)}] q_cut={q_cut:.6f} best_seed={seed}", flush=True)
        _row, train = deo.fit_cut_de_open(
            obj_map,
            summary,
            q_cut,
            opts={
                "seed": seed,
                "de_kw": de_kw,
                "lbfgs_polish": bool(meta.get("lbfgs_polish", False)),
            },
        )
        curve_by_cut[q_cut] = deo.base.full_q_curves(obj_map, train, energy=283.7)
    np.savez_compressed(
        CACHE,
        q_cuts=np.asarray(q_cuts, dtype=float),
        best_seeds=np.asarray(best_seeds, dtype=int),
        **{f"q_{i}": curve_by_cut[q]["q"] for i, q in enumerate(q_cuts)},
        **{f"logR_{i}": curve_by_cut[q]["logR"] for i, q in enumerate(q_cuts)},
        **{f"logR_err_{i}": curve_by_cut[q]["logR_err"] for i, q in enumerate(q_cuts)},
        **{f"logR_fit_{i}": curve_by_cut[q]["logR_fit"] for i, q in enumerate(q_cuts)},
        **{f"pol_{i}": curve_by_cut[q]["pol"] for i, q in enumerate(q_cuts)},
    )
    print(f"wrote {CACHE}")

cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(q_cuts)))
print("cascade cuts:", len(q_cuts), "best_seeds:", best_seeds)"""
        ),
        cell_code(
            r"""fig3, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
DATA_KW = dict(
    fmt="o",
    ms=1.8,
    lw=0.35,
    color="0.45",
    ecolor="0.75",
    elinewidth=0.3,
    capsize=0,
    zorder=1,
    rasterized=True,
)

for col, pol_name in enumerate(("s", "p")):
    ax = axes[col]
    ref = curve_by_cut[q_cuts[0]]
    m = ref["pol"] == pol_name
    order = np.argsort(ref["q"][m])
    qq = ref["q"][m][order]
    ax.errorbar(qq, ref["logR"][m][order], yerr=ref["logR_err"][m][order], **DATA_KW)

    for q_cut, color in zip(q_cuts, cmap, strict=True):
        cur = curve_by_cut[q_cut]
        m = cur["pol"] == pol_name
        order = np.argsort(cur["q"][m])
        qq = cur["q"][m][order]
        yf = cur["logR_fit"][m][order]
        ax.plot(qq, yf, color=color, lw=0.95, zorder=2, alpha=0.95)
        ax.axvline(q_cut, color=color, ls="--", lw=0.55, alpha=0.55, zorder=0)

    ax.axvline(q_lo, color="0.15", ls=":", lw=1.0, alpha=0.9, zorder=3)
    ax.axvline(q_hi, color="0.15", ls="-", lw=0.7, alpha=0.5, zorder=3)
    ax.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax.set_title(f"283.7 eV  {pol_name}", fontsize=9)
    if col == 0:
        ax.set_ylabel(r"$\log_{10} R$")

sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis,
    norm=plt.Normalize(vmin=min(q_cuts), vmax=max(q_cuts)),
)
sm.set_array([])
cbar = fig3.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cbar.set_label(r"$q_{\mathrm{cut}}$ (Å$^{-1}$)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

fig3.suptitle(
    "Open DE cascade (best seed per cut) on full 283.7 eV reflectivity\n"
    r"dashed = each $q_{\mathrm{cut}}$; holdout is $q > q_{\mathrm{cut}}$; "
    rf"$q_{{\mathrm{{lo}}}}$={q_lo:.3f} (dotted), $q_{{\mathrm{{hi}}}}$={q_hi:.3f}",
    fontsize=9.5,
    y=1.08,
)
fig3.savefig(OUT / "reflectivity_q_cuts.png", dpi=300, bbox_inches="tight")
fig3.savefig(OUT / "reflectivity_q_cuts.pdf", bbox_inches="tight")
fig3"""
        ),
        cell_code(
            """print(f"d_lim={d_full:.3f} A  seeds={meta.get('seeds')}  lbfgs_polish={meta.get('lbfgs_polish')}")
print(f"DE: popsize={meta['de_kw']['popsize']}  maxiter={meta['de_kw']['maxiter']}")
print(f"structural n={meta.get('n_structural_free')}  nvary_probe={meta.get('nvary_probe')}")
h = agg[agg["n_holdout"] > 0]
print(
    "holdout RMS mean range: "
    f"{h['rms_logR_holdout_mean'].min():.3f} .. {h['rms_logR_holdout_mean'].max():.3f}"
)
print(
    "holdout RMS std range: "
    f"{h['rms_logR_holdout_std'].min():.3f} .. {h['rms_logR_holdout_std'].max():.3f}"
)
rep.head()"""
        ),
    ],
}

out = Path(__file__).with_name("bookended_q_truncation_diffraction_de_open.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out}")

"""283.7 eV-only dynesty (fully open) for bookended vs DFT-slab.

Bookended seeds from ``e283p7_only_overfit`` L-BFGS; DFT-slab seeds from the
FIG4 polish pickle ``dft_slab_eres_logY_polish/fit.pkl``. Both restrict data
to 283.7 s+p, free film + oxide/Si + anchor instruments, and run the same GPU
``DynamicNestedSampler`` path as ``_eres_dynesty_gpu.py`` (status.json +
checkpoint.pkl for live check-in).

Launch
------
::

    UV_NO_CONFIG=1 uv run python notebooks/fitting/_eres_dynesty_2837_only.py \\
        --full --parallel --models bookended dft_slab
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from utils import models_fitting_results
from utils.models import configure_refloxide_fitting

configure_refloxide_fitting()

_HERE = Path(__file__).resolve().parent
ROOT = models_fitting_results / "eres_dynesty_gpu"
OUT_DIR = ROOT / "e283p7_only_open"
OVERFIT_SUMMARY = (
    ROOT / "full_profile_contract" / "e283p7_only_overfit" / "summary.json"
)
ENERGY = 283.7
ANCHOR_TAG = "283p7eV"
BKG_BOUNDS = (0.0, 5e-3)
OXIDE_DENSITY_BOUNDS = (1.8, 2.45)
SI_DENSITY_BOUNDS = (2.20, 2.45)
OXIDE_THICK_BOUNDS = (2.5, 16.0)
OXIDE_ROUGH_BOUNDS = (0.5, 14.0)
SI_ROUGH_BOUNDS = (0.3, 5.0)


def log(msg: str) -> None:
    print(msg, flush=True)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def apply_fit_values(objective, fit_values: dict[str, float]) -> int:
    by_name = {str(p.name): p for p in objective.parameters.flattened()}
    n = 0
    for name, val in fit_values.items():
        if name in by_name:
            by_name[name].value = float(val)
            n += 1
    objective.refresh_parameters()
    return n


def open_slab_everything(ed, objective) -> list[str]:
    """Free ZnPc + oxide + Si + 283.7 instruments (everything open at anchor)."""
    opened: list[str] = []
    pi2 = float(np.pi / 2)
    structure = objective.model.structure
    for i, comp in enumerate(structure):
        name = str(getattr(comp, "name", i)).lower()
        if name in {"vacuum"}:
            continue
        is_oxide = "oxide" in name or "sio" in name
        is_substrate = name in {"substrate", "si"} or "substrate" in name
        is_bulk = "bulk" in name
        is_interface = "interface" in name
        is_surface = "surface" in name or "znpc" in name

        if is_substrate:
            dens = getattr(getattr(comp, "sld", None), "density", None)
            if dens is not None and getattr(dens, "constraint", None) is None:
                ed._setp(
                    dens,
                    value=float(dens.value),
                    vary=True,
                    bounds=SI_DENSITY_BOUNDS,
                )
                opened.append(f"{getattr(comp, 'name', i)}.density")
            rough = getattr(comp, "rough", None)
            if rough is not None and getattr(rough, "constraint", None) is None:
                ed._setp(
                    rough,
                    value=float(rough.value),
                    vary=True,
                    bounds=SI_ROUGH_BOUNDS,
                )
                opened.append(f"{getattr(comp, 'name', i)}.rough")
            continue

        for attr in ("thick", "rough"):
            p = getattr(comp, attr, None)
            if p is None or not hasattr(p, "setp"):
                continue
            if getattr(p, "constraint", None) is not None:
                continue
            val = float(p.value)
            if attr == "thick":
                if is_oxide:
                    b = OXIDE_THICK_BOUNDS
                elif is_bulk:
                    b = ed._local_bounds(val, lo=140.0, hi=210.0, rel=0.12)
                else:
                    b = ed._local_bounds(val, lo=4.0, hi=30.0, rel=0.25)
            else:
                if is_oxide:
                    b = OXIDE_ROUGH_BOUNDS
                else:
                    b = ed._local_bounds(val, lo=0.3, hi=25.0, abs_pad=3.0, rel=None)
            ed._setp(p, vary=True, bounds=b)
            opened.append(f"{getattr(comp, 'name', i)}.{attr}")

        sld = getattr(comp, "sld", None)
        if sld is None:
            continue
        dens = getattr(sld, "density", None)
        if dens is not None and getattr(dens, "constraint", None) is None:
            val = float(dens.value)
            if is_oxide:
                b = OXIDE_DENSITY_BOUNDS
            elif is_interface:
                b = ed._local_bounds(val, lo=0.5, hi=3.5, rel=0.20)
            elif is_bulk:
                b = ed._local_bounds(val, lo=1.0, hi=2.5, rel=0.15)
            else:
                b = ed._local_bounds(val, lo=1.0, hi=5.0, rel=0.20)
            ed._setp(dens, vary=True, bounds=b)
            opened.append(f"{getattr(comp, 'name', i)}.density")
        rot = getattr(sld, "rotation", None)
        if (
            rot is not None
            and getattr(rot, "constraint", None) is None
            and (is_surface or is_bulk or is_interface)
            and not is_oxide
        ):
            val = float(rot.value)
            b = ed._local_bounds(val, lo=0.0, hi=pi2, abs_pad=0.35, rel=None)
            ed._setp(rot, vary=True, bounds=b)
            opened.append(f"{getattr(comp, 'name', i)}.rotation")

    for p in objective.parameters.flattened():
        name = str(p.name)
        is_inst = (
            name.startswith("scale_")
            or name.startswith("theta_offset_")
            or name.startswith("bkg")
        )
        if not is_inst:
            continue
        if ANCHOR_TAG not in name:
            p.vary = False
            continue
        if name.startswith("bkg"):
            ed._setp(
                p,
                value=max(0.0, float(p.value)),
                vary=True,
                bounds=BKG_BOUNDS,
            )
        else:
            p.vary = True
        opened.append(name)

    ed._widen_instrument_bounds(objective)
    # Re-freeze non-anchor instruments after widen.
    for p in objective.parameters.flattened():
        name = str(p.name)
        is_inst = (
            name.startswith("scale_")
            or name.startswith("theta_offset_")
            or name.startswith("bkg")
        )
        if is_inst and ANCHOR_TAG not in name:
            p.vary = False
    if hasattr(objective.model, "energy_offset"):
        objective.model.energy_offset.setp(vary=False)
    objective.refresh_parameters()
    log(f"dft_slab 283.7-only open: free={len(objective.varying_parameters())}")
    return opened


def build_bookended(ed, br, fit_values: dict[str, float], book_med: dict[str, float]):
    of = _load("of_2837_ns", _HERE / "_bookended_2837_only_overfit.py")
    book = ed.load_bookended(contract="profile")
    br.open_all_from_profile_best(ed, book, book_med)
    of.restrict_to_energy(book, ENERGY)
    of.freeze_non_anchor_instruments(book)
    apply_fit_values(book, fit_values)
    ed._widen_instrument_bounds(book)
    of.freeze_non_anchor_instruments(book)
    if hasattr(book.model, "energy_offset"):
        book.model.energy_offset.setp(vary=False)
    book.refresh_parameters()
    return book


def build_slab(ed):
    """FIG4 DFT-slab seed: ``dft_slab_eres_logY_polish/fit.pkl`` (polish MLE).

    Same pickle FIG4 loads via ``utils.refloxide_view.load_refloxide_fit``.
    Polarization swap is corrected inside ``load_slab``. Then open film +
    oxide/Si + 283.7 instruments and restrict the dataset to 283.7 only.
    """
    of = _load("of_2837_ns", _HERE / "_bookended_2837_only_overfit.py")
    slab = ed.load_slab(contract="polish")
    open_slab_everything(ed, slab)
    of.restrict_to_energy(slab, ENERGY)
    ed._widen_instrument_bounds(slab)
    of.freeze_non_anchor_instruments(slab)
    if hasattr(slab.model, "energy_offset"):
        slab.model.energy_offset.setp(vary=False)
    slab.refresh_parameters()
    log(f"dft_slab seed=FIG4 polish pickle  path={ed.SLAB_PKL}")
    return slab


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true", default=True)
    p.add_argument("--parallel", action="store_true")
    p.add_argument(
        "--models",
        nargs="+",
        default=["bookended", "dft_slab"],
        choices=["bookended", "dft_slab"],
    )
    p.add_argument("--nlive", type=int, default=0)
    p.add_argument("--walks", type=int, default=0)
    p.add_argument("--dlogz-init", type=float, default=None)
    p.add_argument("--n-effective", type=int, default=None)
    return p.parse_args(argv)


def _spawn_parallel(args: argparse.Namespace) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    cwd = script.parents[2]
    procs: list[tuple[str, object]] = []
    for label in args.models:
        log_path = OUT_DIR / f"{label}.log"
        cmd = [
            sys.executable,
            str(script),
            "--full",
            "--models",
            label,
        ]
        if args.nlive > 0:
            cmd.extend(["--nlive", str(args.nlive)])
        if args.walks > 0:
            cmd.extend(["--walks", str(args.walks)])
        if args.dlogz_init is not None:
            cmd.extend(["--dlogz-init", str(args.dlogz_init)])
        if args.n_effective is not None:
            cmd.extend(["--n-effective", str(args.n_effective)])
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

    summary: dict = {
        "mode": "full",
        "energy": ENERGY,
        "contract": "2837_only_open",
        "parallel": True,
        "models": {},
        "bayes_factor": None,
    }
    statuses: dict[str, dict] = {}
    ed = _load("ed_bf", _HERE / "_eres_dynesty_gpu.py")
    for label in args.models:
        st_path = OUT_DIR / label / "status.json"
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
    summary["bayes_factor"] = ed.bayes_factor_summary(statuses)
    ed._atomic_json(OUT_DIR / "summary.json", summary)
    log(f"wrote {OUT_DIR / 'summary.json'}")
    if summary["bayes_factor"] is not None:
        bf = summary["bayes_factor"]
        log(
            f"Bayes factor bookended/slab: ΔlogZ={bf['delta_logz']:.4f}±"
            f"{bf['delta_logz_err']:.4f}  log10(BF)={bf['log10_bf']:.3f}"
        )
    return rc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.parallel and len(args.models) > 1:
        return _spawn_parallel(args)

    ed = _load("ed_2837_ns", _HERE / "_eres_dynesty_gpu.py")
    br = _load("br_2837_ns", _HERE / "_bookended_open_erefits.py")

    if not OVERFIT_SUMMARY.exists():
        raise FileNotFoundError(
            f"missing 283.7 overfit seed {OVERFIT_SUMMARY}; "
            "run _bookended_2837_only_overfit.py first"
        )
    overfit = json.loads(OVERFIT_SUMMARY.read_text())
    book_post = json.loads(
        (
            ROOT / "full_profile_contract" / "bookended" / "posterior_summary.json"
        ).read_text()
    )
    book_med = br.medians(book_post)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"OUT_DIR={OUT_DIR}")
    dlogz = float(args.dlogz_init) if args.dlogz_init is not None else ed.DLOGZ_INIT
    n_eff = int(args.n_effective) if args.n_effective is not None else ed.N_EFFECTIVE
    nlive_override = None if args.nlive <= 0 else args.nlive
    walks_override = None if args.walks <= 0 else args.walks

    summary: dict = {
        "mode": "full",
        "energy": ENERGY,
        "contract": "2837_only_open",
        "seed_bookended": str(OVERFIT_SUMMARY),
        "seed_dft_slab": "dft_slab_eres_logY_polish/fit.pkl (FIG4)",
        "dlogz_init": dlogz,
        "n_effective": n_eff,
        "models": {},
        "bayes_factor": None,
    }
    statuses: dict[str, dict] = {}

    for label in args.models:
        log(f"=== {label} (283.7-only open full) ===")
        if label == "bookended":
            objective = build_bookended(
                ed, br, overfit["bookended"]["fit_values"], book_med
            )
        else:
            objective = build_slab(ed)
        n_peek = len(objective.varying_parameters())
        nlive = ed.nlive_for_ndim(n_peek, nlive_override)
        walks = ed.walks_for_ndim(n_peek, walks_override)
        log(f"{label}: free params ({n_peek}):")
        for p in objective.varying_parameters():
            log(
                f"  {p.name}={float(p.value):.6g} "
                f"[{float(p.bounds.lb):.4g}, {float(p.bounds.ub):.4g}]"
            )
        status = ed.run_full(
            label,
            objective,
            nlive=nlive,
            walks=walks,
            dlogz_init=dlogz,
            n_effective=n_eff,
            out_dir=OUT_DIR,
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
        ed._atomic_json(OUT_DIR / "summary.json", summary)

    summary["bayes_factor"] = ed.bayes_factor_summary(statuses)
    ed._atomic_json(OUT_DIR / "summary.json", summary)
    log(f"wrote {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

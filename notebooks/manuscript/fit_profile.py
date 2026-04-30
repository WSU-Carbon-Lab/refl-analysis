#!/usr/bin/env python3
"""Profile fitting script. Run: uv run python notebooks/manuscript/fit_profile.py <name> [--quiet] (from project root)."""
import argparse
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyref.fitting as fit
from uncertainties import ufloat

from utils import models_fitting_results, read_ooc, read_xrr
from utils.helpers.fitting_helper import aic, bic, reduced_chi2
from utils.helpers.plotting_helper import plot_all_results, set_plotting_defaults
from utils.profile_slab import AdaptiveOrientationProfile
from utils.slab_builders import sio2, substrate, surface, vacuum


def model(
    energy: float,
    oocs: pd.DataFrame,
    total_thick: float = 198.2,
    surface_roughness: float = 2.8,
    density: float = 1.6153,
    characteristic_thickness: float = 8,
    max_angle: float = 0.638,
    initial_angle: float = 0,
    num_slabs: int = 20,
):
    energy = float(energy)
    return (
        vacuum(energy)  # type: ignore
        | surface(energy, oocs, thick=0, rough=0)
        | AdaptiveOrientationProfile(
            oocs,
            energy=energy,
            total_thick=total_thick,
            surface_roughness=surface_roughness,
            density=density,
            characteristic_thickness=characteristic_thickness,
            max_angle=max_angle,
            initial_angle=initial_angle,
            name=f"ZnPc_{energy:.1f}",
            num_slabs=num_slabs,
        )
        | sio2(energy)
        | substrate(energy)
    )


def _filt_struct(x: str) -> bool:
    if "get_" in x.lower() or "get" in x.lower():
        return False
    if "slab" in x.lower():
        return False
    return (
        "thick" in x.lower()
        or "thickness" in x.lower()
        or "rough" in x.lower()
        or "roughness" in x.lower()
    )


def _filt_sld(x: str) -> bool:
    if "get_" in x.lower() or "get" in x.lower():
        return False
    return (
        "rot" in x.lower()
        or "rotation" in x.lower()
        or "angle" in x.lower()
        or "density" in x.lower()
        or "rho" in x.lower()
    )


def _link_params(slab, ref) -> None:
    struct_params = list(filter(_filt_struct, dir(slab)))
    for pname in struct_params:
        p = getattr(slab, pname)
        try:
            p.setp(vary=None, constraint=getattr(ref, pname))
        except Exception:
            print(f"Failed to link {pname}")
            print(slab)
            print(ref)

    sld_params_t0 = list(filter(_filt_sld, dir(slab)))
    for pname in sld_params_t0:
        p = getattr(slab, pname)
        p.setp(vary=None, constraint=getattr(ref, pname))
    sld_params_t1 = list(filter(_filt_sld, dir(slab.sld)))
    for pname in sld_params_t1:
        p = getattr(slab.sld, pname)
        p.setp(vary=None, constraint=getattr(ref.sld, pname))


def link_params(slab, ref) -> None:
    for _slab, _ref in zip(slab.components, ref.components, strict=True):
        _link_params(_slab, _ref)


def package_results(fitter: fit.CurveFitter) -> pd.DataFrame:
    params = []
    for p in fitter.objective.varying_parameters():
        err = p.stderr if p.stderr is not None else 0
        params.append({"name": p.name, "value": ufloat(p.value, err)})
    params.append(
        {"name": "reduced_chi2", "value": reduced_chi2(fitter.objective)}
    )
    params.append({"name": "aic", "value": aic(fitter.objective)})
    params.append({"name": "bic", "value": bic(fitter.objective)})
    return pd.DataFrame(params).set_index("name")


def _append_main_log(
    name: str,
    version: str,
    run_time_seconds: float,
    fitter: fit.CurveFitter,
) -> None:
    logs_dir = models_fitting_results / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = logs_dir / f"{name}.log"
    chi2 = reduced_chi2(fitter.objective)
    aic_val = aic(fitter.objective)
    bic_val = bic(fitter.objective)
    with open(main_log_path, "a", encoding="utf-8") as f:
        f.write(f"version={version} ")
        f.write(f"run_time_sec={run_time_seconds:.2f} ")
        f.write(f"reduced_chi2={chi2:.6f} aic={aic_val:.2f} bic={bic_val:.2f}\n")


def main(name: str, version: str) -> None:
    out_prefix = models_fitting_results / f"{name}_{version}"

    oocs = read_ooc("dft.csv", material="znpc")
    data = read_xrr("reflectivity_data", material="znpc")

    models = {e: model(float(e), oocs) for e in data}
    template = models["283.7"]

    template.components[1].thick.setp(vary=True)
    template.components[1].rough.setp(vary=True)
    template.components[1].sld.density.setp(vary=True, bounds=(1.0, 1.7))

    template.components[2].total_thick.setp(vary=True, bounds=(160, 210))
    template.components[2].surface_roughness.setp(vary=True, bounds=(0, 25))
    template.components[2].density.setp(value=1.6153, vary=False, bounds=(1.5, 1.7))
    template.components[2].max_angle.setp(vary=True, bounds=(np.pi / 2, np.pi))
    template.components[2].initial_angle.setp(vary=True, bounds=(0, np.pi / 2))

    template.components[3].thick.setp(vary=False, bounds=(0, 25))
    template.components[3].rough.setp(vary=False, bounds=(0, 25))
    template.components[3].sld.density.setp(vary=False)

    for e in data:
        if e == "283.7":
            continue
        link_params(models[e], template)

    refl: dict[str, fit.ReflectModel] = {
        e: fit.ReflectModel(models[e], energy=float(e), pol="sp") for e in data
    }
    fig, ax = plt.subplots(nrows=2)
    for e, ref in refl.items():
        ref.scale_s.setp(vary=True, bounds=(0.5, 2))
        ref.scale_p.setp(vary=True, bounds=(0.5, 2))
        ref.theta_offset_s.setp(vary=True, bounds=(-1, 1))
        ref.theta_offset_p.setp(vary=True, bounds=(-1, 1))
        if e in ["250.0", "283.7"]:
            exp = data[e].p
            ref.bkg.setp(value=exp.y.min(), vary=False)
            i = ["250.0", "283.7"].index(e)
            ax[i].plot(exp.x, exp.y)
            ax[i].set_yscale("log")
            ax[i].axhline(ref.bkg.value, color="k", linestyle="--")
    plt.close(fig)

    objective: list[fit.AnisotropyObjective] = [
        fit.AnisotropyObjective(
            refl[e], data[e], logp_anisotropy_weight=0.5, transform=fit.Transform("logY")
        )
        for e in data
    ]
    global_objective = fit.GlobalObjective(objective)
    print(global_objective.varying_parameters())

    fitter = fit.CurveFitter(global_objective)
    df_sparse = package_results(fitter)
    df_sparse.to_csv(f"{out_prefix}_initial.csv")

    start = perf_counter()
    fitter.fit(method="differential_evolution", updating="deferred", workers=10)
    end = perf_counter()
    run_time = end - start
    print(f"Time taken: {run_time} seconds")

    df_output = package_results(fitter)
    df_output.to_csv(f"{out_prefix}_final.csv")

    set_plotting_defaults()
    fig = plot_all_results(fitter.objective, data)
    if fig is not None:
        fig.set_size_inches(5, 8)
        fig.savefig(f"{out_prefix}_final.png", dpi=300, bbox_inches="tight")

    _append_main_log(name, version, run_time, fitter)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile fitting (runs in background by default).")
    parser.add_argument("name", help="Run name; used for log and output filenames.")
    parser.add_argument("--quiet", action="store_true", help="Do not print PID or log path to terminal.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timestamp", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.child:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        main(args.name, args.timestamp)
        sys.exit(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_log_path = models_fitting_results / f"{args.name}_version_{ts}.log"
    models_fitting_results.mkdir(parents=True, exist_ok=True)

    with open(run_log_path, "w", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            [sys.executable, __file__, args.name, "--child", "--timestamp", ts],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if not args.quiet:
        print(f"Started PID={proc.pid} log={run_log_path}")

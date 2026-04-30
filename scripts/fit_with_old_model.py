#!/usr/bin/env python3
"""
Fit reflectivity data using a previously fitted model as a starting point.

This script:
1. Loads reflectivity data from parquet
2. Loads a previous fitting result to use as starting structure
3. Opens structural parameters for refitting
4. Applies constraints between energy-dependent structures
5. Fits the data using SLSQP with constraints or dynamic nested sampling
6. Exports results with comprehensive plots and statistics
"""

import argparse
import copy
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
import pyref.fitting as fit
from matplotlib.lines import Line2D
from scipy.optimize import NonlinearConstraint

# Default CPU count
CPU_COUNT = os.cpu_count() or 16

# ============================================================================
# Data Loading
# ============================================================================


def load_reflectivity_dataset(filename: str | Path) -> dict:
    """
    Load reflectivity dataset from a parquet file.

    Parameters
    ----------
    filename : str or Path
        Input filename (should be a .parquet file)

    Returns
    -------
    dict
        Dictionary containing XrayReflectDataset objects with energy keys
    """
    df_load = pl.read_parquet(filename)
    data_reconstructed = {}

    for group_key, group_data in df_load.group_by("energy"):
        energy_val = group_key[0]
        Q = group_data["Q"].to_numpy()
        R = group_data["R"].to_numpy()
        dR = group_data["dR"].to_numpy()
        dataset = fit.XrayReflectDataset(data=(Q, R, dR))
        data_reconstructed[str(energy_val)] = dataset

    print(f"Dataset loaded from {filename}")
    return data_reconstructed


# ============================================================================
# Structure Manipulation
# ============================================================================


def safely_setp(param, **kwargs):
    """Remove constraint before setting parameter properties."""
    if param.constraint is not None:
        param.constraint = None
    param.setp(**kwargs)


def open_structural_parameters(structure):
    """
    Open the thickness, roughness, and density constraints for a structure.

    Parameters
    ----------
    structure : refnx.reflect.Structure
        The structure to open parameters for

    Returns
    -------
    structure : refnx.reflect.Structure
        The modified structure with open parameters
    """
    for i, slab in enumerate(structure):
        if i == 0:  # Skip vacuum layer
            continue
        elif i == len(structure) - 1:  # Substrate: only vary density
            safely_setp(slab.sld.density, vary=True)
        else:  # Middle layers: vary all structural parameters
            safely_setp(slab.thick, vary=True)
            safely_setp(slab.rough, vary=True)
            safely_setp(slab.sld.density, vary=True)
            safely_setp(slab.sld.rotation, vary=True)
    return structure


def constrain_structures(structure, constraint):
    """
    Constrain one structure to match the parameters of another.

    Parameters
    ----------
    structure : refnx.reflect.Structure
        The structure to constrain
    constraint : refnx.reflect.Structure
        The reference structure to constrain to

    Returns
    -------
    structure : refnx.reflect.Structure
        The constrained structure
    """
    for slab, slab_constraint in zip(structure.components, constraint.components):
        slab.thick.setp(vary=None, constraint=slab_constraint.thick)
        slab.rough.setp(vary=None, constraint=slab_constraint.rough)
        slab.sld.density.setp(vary=None, constraint=slab_constraint.sld.density)
        slab.sld.rotation.setp(vary=None, constraint=slab_constraint.sld.rotation)
    return structure


# ============================================================================
# Model Building
# ============================================================================


def new_model(energy: float, structure) -> fit.ReflectModel:
    """
    Create a new ReflectModel with standard instrument parameters.

    Parameters
    ----------
    energy : float
        Photon energy in eV
    structure : refnx.reflect.Structure
        The structure to use for the model

    Returns
    -------
    model : fit.ReflectModel
        The configured model
    """
    mod = fit.ReflectModel(
        structure,
        pol="sp",
        name=f"Model {energy}",
        energy=energy,
        bkg=0,
        scale_p=1,
        scale_s=1,
        theta_offset_p=0,
        theta_offset_s=0,
    )
    mod.energy_offset.setp(vary=True, bounds=(-0.4, 0.4))
    mod.scale_p.setp(vary=True, bounds=(0.6, 1.4))
    mod.scale_s.setp(vary=True, bounds=(0.6, 1.4))
    mod.theta_offset_p.setp(vary=True, bounds=(-0.5, 0.5))
    mod.theta_offset_s.setp(vary=True, bounds=(-0.5, 0.5))
    return mod


def new_objective(model, data, logp_weight: float = 0.16):
    """
    Create an AnisotropyObjective with logY transform.

    Parameters
    ----------
    model : fit.ReflectModel
        The model to fit
    data : fit.XrayReflectDataset
        The data to fit to
    logp_weight : float, optional
        Weight for the anisotropy log-probability, by default 0.16

    Returns
    -------
    objective : fit.AnisotropyObjective
        The configured objective
    """
    obj = fit.AnisotropyObjective(model, data, logp_anisotropy_weight=logp_weight)
    obj.transform = fit.Transform("logY")
    return obj


# ============================================================================
# Constraints
# ============================================================================


class LogpExtra:
    """Log Prior Constraint for reflectometry data fitting."""

    def __init__(self, objective):
        self.objective = objective

    def __call__(self, model, data):
        """
        Apply custom log-prior constraint.

        Checks:
        1. Interface roughness is less than layer thickness
        2. Oxide density > substrate density
        """
        for slab in model.structure.components:
            interface_limit = np.sqrt(2 * np.pi) * slab.rough.value / 2
            if float(slab.thick.value - interface_limit) < 0:
                return -np.inf

        # Check oxide > substrate density
        if float(
            model.structure.components[-1].sld.density.value
            - model.structure.components[-2].sld.density.value
        ) < 0:
            return -np.inf

        return 0


def create_global_constraints(global_obj):
    """
    Create nonlinear constraints for use with SLSQP optimizer.

    Parameters
    ----------
    global_obj : fit.GlobalObjective
        The global objective to constrain

    Returns
    -------
    list
        List of NonlinearConstraint objects
    """

    class GlobalDensityConstraint:
        def __init__(self, objective):
            self.objective = objective
            try:
                self.oxide_density = next(
                    p
                    for p in objective.varying_parameters()
                    if "Oxide" in p.name and "rho" in p.name
                )
                self.substrate_density = next(
                    p
                    for p in objective.varying_parameters()
                    if "Substrate" in p.name and "rho" in p.name
                )
            except StopIteration:
                # Fallback: use last two components
                params = list(objective.varying_parameters())
                density_params = [p for p in params if "density" in p.name or "rho" in p.name]
                if len(density_params) >= 2:
                    self.oxide_density = density_params[-2]
                    self.substrate_density = density_params[-1]
                else:
                    raise ValueError(
                        f"Could not find density parameters. Available: "
                        f"{[p.name for p in objective.parameters]}"
                    )

        def __call__(self, x_global):
            self.objective.setp(x_global)
            return float(self.oxide_density.value - self.substrate_density.value)

    class GlobalRoughnessConstraint:
        def __init__(self, objective):
            self.objective = objective

        def __call__(self, x_global):
            self.objective.setp(x_global)
            min_diff = np.inf
            for o in self.objective.objectives:
                for slab in o.model.structure.components:
                    if slab.thick.value > 0:  # Skip vacuum
                        interface_limit = np.sqrt(2 * np.pi) * slab.rough.value / 2
                        diff = float(slab.thick.value - interface_limit)
                        min_diff = min(min_diff, diff)
            return min_diff

    return [
        NonlinearConstraint(GlobalDensityConstraint(global_obj), 0, np.inf),
        NonlinearConstraint(GlobalRoughnessConstraint(global_obj), 0, np.inf),
    ]


# ============================================================================
# Fitting
# ============================================================================

T = TypeVar("T", bound=fit.Objective)


def fit_dynamic(obj: T, cpu: int = CPU_COUNT) -> tuple[T, Any]:
    """
    Fit the model using dynamic nested sampling.

    Parameters
    ----------
    obj : fit.Objective or fit.GlobalObjective
        The objective to fit
    cpu : int, optional
        Number of CPUs to use, by default CPU_COUNT

    Returns
    -------
    objective : fit.Objective
        The fitted objective
    nested_sampler : dynesty.DynamicNestedSampler
        The sampler object with results
    """
    import dynesty
    from refnx.analysis import process_chain

    objective = copy.deepcopy(obj)
    ndim = len(objective.varying_parameters())
    nlive = max(1000, 2 * ndim + 1)

    with dynesty.pool.Pool(cpu, objective.logl, objective.prior_transform) as pool:
        nested_sampler = dynesty.DynamicNestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=ndim,
            nlive=nlive,
            pool=pool,
        )
        nested_sampler.run_nested()

    logZdynesty = nested_sampler.results.logz[-1]  # type: ignore
    weights = np.exp(nested_sampler.results.logwt - logZdynesty)  # type: ignore
    chain = dynesty.utils.resample_equal(nested_sampler.results.samples, weights)  # type: ignore
    process_chain(objective, chain[:, None, :])

    return objective, nested_sampler


# ============================================================================
# Analysis and Export
# ============================================================================


def reduced_chi2(objective):
    """Calculate reduced chi-squared."""
    if isinstance(objective, fit.GlobalObjective):
        return objective.chisqr() / (len(objective.objectives) * (len(objective.objectives[0].data.s.x) + len(objective.objectives[0].data.p.x) - len(objective.objectives[0].varying_parameters())))
    ndata = len(objective.data.s.x) + len(objective.data.p.x)
    nparams = len(objective.varying_parameters())
    return objective.chisqr() / (ndata - nparams)


def aic(objective):
    """Calculate Akaike Information Criterion."""
    nparams = len(objective.varying_parameters())
    return objective.chisqr() + 2 * nparams


def bic(objective):
    """Calculate Bayesian Information Criterion."""
    ndata = len(objective.data.s.x) + len(objective.data.p.x)
    nparams = len(objective.varying_parameters())
    return objective.chisqr() + nparams * np.log(ndata)


def rxr(x, model, pol):
    """Calculate reflectivity for a given polarization."""
    _pol = model.pol
    model.pol = pol
    y = model(x)
    model.pol = _pol
    return y


def anisotropy(x, model):
    """Calculate anisotropy from model."""
    r_s = rxr(x, model, "s")
    r_p = rxr(x, model, "p")
    return (r_p - r_s) / (r_p + r_s)


def plot_fit_results(
    objective,
    save_path: Path | str | None = None,
    comparitive_objective=None,
    comparitive_label=None,
):
    """
    Generate a comprehensive plot of fitting results.

    Parameters
    ----------
    objective : fit.Objective
        The fitted objective
    save_path : Path or str, optional
        Where to save the figure
    comparitive_objective : fit.Objective, optional
        A second objective for comparison
    comparitive_label : str, optional
        Label for the comparison
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.dpi": 300,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    data = objective.data
    data.name = ""
    mod = objective.model

    fig, ((ax_ref, ax_ani), (ax_res, ax_ani_res)) = plt.subplots(
        2, 2, figsize=(10, 3), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot data
    data.plot(ax=ax_ref, show_anisotropy=False)
    ax_ani.scatter(
        data.anisotropy.x,
        data.anisotropy.y,
        label="Data Anisotropy",
        color="C2",
        marker="o",
        s=10,
    )

    # Plot model fits
    ax_ref.plot(
        data.s.x,
        rxr(data.s.x, mod, "s"),
        label="Fit (s-pol)",
        color="C0",
        linestyle="-",
    )
    ax_ref.plot(
        data.p.x,
        rxr(data.p.x, mod, "p"),
        label="Fit (p-pol)",
        color="C1",
        linestyle="-",
    )
    ax_ani.plot(
        data.anisotropy.x,
        anisotropy(data.anisotropy.x, mod),
        label="Fit Anisotropy",
        color="C2",
        linestyle="-",
    )

    # Comparison if provided
    if comparitive_objective:
        mod2 = comparitive_objective.model
        ax_ref.plot(
            data.s.x, rxr(data.s.x, mod2, "s"), label="Fit2 (s-pol)", color="C0", linestyle="--"
        )
        ax_ref.plot(
            data.p.x, rxr(data.p.x, mod2, "p"), label="Fit2 (p-pol)", color="C1", linestyle="--"
        )
        ax_ani.plot(
            data.anisotropy.x,
            anisotropy(data.anisotropy.x, mod2),
            label="Fit2 Anisotropy",
            color="C2",
            linestyle="--",
        )

    # Background shading
    ymin = ax_ref.get_ylim()[0]
    ax_ref.fill_between(
        x=data.s.x,
        y1=ymin,
        y2=mod.bkg.value,
        color="none",
        alpha=0.3,
        hatch="//",
        edgecolor="gray",
    )
    ax_ref.set_ylim(ymin, None)

    # Residuals
    res_s = (data.s.y - rxr(data.s.x, mod, "s")) / data.s.y_err
    res_p = (data.p.y - rxr(data.p.x, mod, "p")) / data.p.y_err
    res_a = data.anisotropy.y - anisotropy(data.anisotropy.x, mod)

    for res, x_data, color, ax_target in [
        (res_s, data.s.x, "C0", ax_res),
        (res_p, data.p.x, "C1", ax_res),
        (res_a, data.anisotropy.x, "C2", ax_ani_res),
    ]:
        quant = np.quantile(res, [0.1, 0.25, 0.5, 0.75, 0.9])
        ax_target.plot(
            x_data,
            res,
            color=color,
            linestyle="-",
            marker="o",
            markersize=2,
            lw=0.5,
            markerfacecolor="white",
            markeredgecolor=color,
        )
        ax_target.fill_between(x=x_data, y1=quant[1], y2=quant[-2], color=color, alpha=0.1)
        ax_target.axhline(0, color="k", linestyle="-", lw=plt.rcParams["axes.linewidth"])

    # Labels and formatting
    ax_ref.set_yscale("log")
    ax_ref.set_ylabel("Reflectivity [abs.]")
    ax_ani.set_ylabel("Anisotropy [abs.]")
    ax_res.set_ylabel("Res.")
    ax_ani_res.set_ylabel("Res.")
    ax_res.set_xlabel(r"$Q$ [$\AA^{-1}$]")
    ax_ani_res.set_xlabel(r"$Q$ [$\AA^{-1}$]")

    ax_ref.legend(ncols=2, handlelength=0.5)
    ax_ani.legend(ncols=2, handlelength=0.5)
    ax_ani.set_xlim(ax_ref.get_xlim())
    ax_ani.set_ylim(-1, 1)

    # Annotation
    scales = (mod.scale_s.value, mod.scale_p.value)
    offsets = (mod.theta_offset_s.value, mod.theta_offset_p.value)
    text = (
        f"Scale (s, p): ({scales[0]:.2f}, {scales[1]:.2f})\n"
        f"Offset (s, p): ({offsets[0]:.2f}, {offsets[1]:.2f})\n"
        f"Background : {mod.bkg.value:.1e}"
    )

    ax_ref.text(
        0.05,
        0.3,
        s=text,
        transform=ax_ref.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
    )

    fig.align_ylabels()
    fig.suptitle(
        rf"Fit at {mod.energy} eV $\chi^2_{{\rm red}} = {reduced_chi2(objective):.2f}$"
        + (f" | {comparitive_label} in dashed" if comparitive_label else "")
    )

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def export_fit_results(
    objectives,
    save_folder: Path = Path("figures"),
    comparitive_objectives=None,
    comparitive_label=None,
):
    """Export comparison plots for all objectives."""
    folder = save_folder / datetime.now().strftime("%Y%m%d")
    folder.mkdir(parents=True, exist_ok=True)

    for obj in objectives:
        comp_obj = (
            next(
                (co for co in comparitive_objectives if co.model.energy == obj.model.energy),
                None,
            )
            if comparitive_objectives
            else None
        )
        energy = obj.model.energy
        save_path = folder / f"fit_comparison_{energy}eV.png"
        plot_fit_results(
            obj,
            save_path=save_path,
            comparitive_objective=comp_obj,
            comparitive_label=comparitive_label,
        )
    print(f"Fit results exported to {folder}")


def export_stats(objectives, save_path: Path):
    """Export fitting statistics to CSV and plot."""
    save_path = save_path.with_name(f"{save_path.stem}_{datetime.now().strftime('%Y%m%d')}.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    stats = []
    for obj in objectives:
        energy = obj.model.energy
        chi2_red = reduced_chi2(obj)
        n_params = len(obj.varying_parameters())
        n_data = len(obj.data.s.x) + len(obj.data.p.x)
        stats.append(
            {
                "Energy (eV)": energy,
                "Reduced Chi^2": chi2_red,
                "Number of Parameters": n_params,
                "Number of Data Points": n_data,
                "AIC": aic(obj),
                "BIC": bic(obj),
            }
        )

    # Global objective stats
    _glob = fit.GlobalObjective(objectives)
    global_stats = {
        "Energy (eV)": 0,
        "Reduced Chi^2": reduced_chi2(_glob),
        "Number of Parameters": len(_glob.varying_parameters()),
        "Number of Data Points": sum(len(obj.data.s.x) + len(obj.data.p.x) for obj in objectives),
        "AIC": aic(_glob),
        "BIC": bic(_glob),
    }
    stats.append(global_stats)

    df_stats = pl.DataFrame(stats).sort("Energy (eV)")
    df_stats.write_csv(save_path)
    print(f"Statistics exported to {save_path}")


# ============================================================================
# Main Workflow
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fit XRR data using a previously fitted model as starting point"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("reflectivity_data.parquet"),
        help="Path to reflectivity data parquet file",
    )
    parser.add_argument(
        "--old-model",
        type=Path,
        default=Path("fitting_results_fixed.pkl"),
        help="Path to previous fitting results pickle file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fitting_results_refit.pkl"),
        help="Path to save new fitting results",
    )
    parser.add_argument(
        "--method",
        choices=["slsqp", "dynamic", "mcmc"],
        default="slsqp",
        help="Fitting method to use",
    )
    parser.add_argument(
        "--reference-energy",
        type=float,
        default=283.7,
        help="Reference energy for constraining other energies",
    )
    parser.add_argument(
        "--figures", type=Path, default=Path("figures"), help="Directory for output figures"
    )
    parser.add_argument("--cpu", type=int, default=CPU_COUNT, help="Number of CPUs to use")
    parser.add_argument(
        "--logp-weight", type=float, default=0.16, help="Weight for anisotropy log-probability"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("XRR Fitting with Previous Model")
    print("=" * 80)

    # Load data
    print("\n1. Loading reflectivity data...")
    loaded_data = load_reflectivity_dataset(args.data)

    # Load previous results
    print(f"\n2. Loading previous fitting results from {args.old_model}...")
    with open(args.old_model, "rb") as f:
        fitting_results = pickle.load(f)

    _objectives = [o for o in fitting_results.objectives]
    structs = {str(o.model.energy): o.model.structure for o in _objectives}
    energy = [o.model.energy for o in _objectives]
    print(f"   Loaded {len(energy)} energy points")

    # Open structural parameters
    print("\n3. Opening structural parameters for refitting...")
    opened_structs = {str(e): open_structural_parameters(s) for e, s in structs.items()}

    # Apply constraints between energies
    print(f"\n4. Constraining structures to reference energy {args.reference_energy} eV...")
    ref_key = str(args.reference_energy)
    for e, s in opened_structs.items():
        if e == ref_key:
            continue
        constrain_structures(s, opened_structs[ref_key])

    # Build models
    print("\n5. Building models...")
    models = {str(e): new_model(e, opened_structs[str(round(e, 1))]) for e in energy}

    # Constrain energy offsets
    for e, m in models.items():
        if e == ref_key:
            continue
        m.energy_offset.setp(vary=None, constraint=models[ref_key].energy_offset)

    # Build objectives
    print("\n6. Building objectives...")
    objectives = [
        new_objective(models[str(e)], loaded_data[str(e)], logp_weight=args.logp_weight)
        for e in energy
    ]

    # Add logp_extra constraints
    print("\n7. Adding prior constraints...")
    for o in objectives:
        o.logp_extra = LogpExtra(o)

    # Create global objective
    print("\n8. Creating global objective...")
    global_objective = fit.GlobalObjective(objectives)
    print(f"   Total varying parameters: {len(global_objective.varying_parameters())}")
    print(f"   Initial chi^2_red: {reduced_chi2(global_objective):.2f}")

    # Fit
    print(f"\n9. Fitting using {args.method.upper()} method...")
    if args.method == "slsqp":
        constraints = create_global_constraints(global_objective)
        fitter = fit.CurveFitter(global_objective)
        result = fitter.fit("SLSQP", constraints=constraints)
        print(f"   Optimization success: {result.success}")
        fitted_objective = global_objective

    elif args.method == "dynamic":
        constraints = create_global_constraints(global_objective)
        fitter = fit.CurveFitter(global_objective)
        result = fitter.fit("SLSQP", constraints=constraints)
        fitted_objective, nested_sampler = fit_dynamic(global_objective, cpu=args.cpu)
        print(f"   Log evidence: {nested_sampler.results.logz[-1]:.2f}")

    elif args.method == "mcmc":
        fitter = fit.CurveFitter(global_objective)
        fitter.initialise("covar")
        chain = fitter.sample(20, 10, pool=args.cpu)
        fitted_objective = global_objective
        print(f"   MCMC sampling complete")

    print(f"   Final chi^2_red: {reduced_chi2(fitted_objective):.2f}")

    # Save results
    print(f"\n10. Saving fitting results to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(fitted_objective, f)

    # Export figures and statistics
    print(f"\n11. Exporting figures to {args.figures}...")
    export_fit_results(fitted_objective.objectives, save_folder=args.figures)
    export_stats(fitted_objective.objectives, save_path=args.figures / "fitting_stats.csv")

    print("\n" + "=" * 80)
    print("Fitting complete!")
    print("=" * 80)
    print(f"Results saved to: {args.output}")
    print(f"Figures saved to: {args.figures / datetime.now().strftime('%Y%m%d')}")
    print(f"Final reduced chi^2: {reduced_chi2(fitted_objective):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

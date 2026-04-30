#!/usr/bin/env python3
"""
Comprehensive analysis of fitting results comparing DFT-constrained vs free-model fits.

This script analyzes differences between:
- fitting_results_fixed.pkl / fitting_results_fixed_2.pkl (DFT-constrained)
- fitting_results_free_model.pkl / fitting_results_free_model_2.pkl (Free-model)

For three energy regions:
1. (250.0, 283.7): Long q range fits
2. (>250, <283.7): Pre-edge delta crossing energies
3. (>283.7): Resonant energies
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyref.fitting as fit
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Import helper functions
import sys
import polars as pl

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'fitting'))

# Import from fitting/helper.py (has rxr and anisotropy)
from helper import (
    aic,
    bic,
    reduced_chi2,
    rxr,
    anisotropy,
)

# Define additional helper functions
def chi2(obj):
    """Calculate chi-squared."""
    return obj.chisqr()

def rmsd(obj):
    """Calculate RMSD."""
    return np.sqrt(obj.chisqr() / obj.npoints)

def load_reflectivity_dataset(filename):
    """Load reflectivity dataset from parquet file."""
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


def load_fitting_results(pkl_path: Path):
    """Load fitting results from pickle file."""
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    return results


def classify_energy_regions(energies: List[float]) -> Dict[str, List[float]]:
    """
    Classify energies into three regions.

    Parameters
    ----------
    energies : List[float]
        List of energy values

    Returns
    -------
    Dict[str, List[float]]
        Dictionary with keys: 'long_q', 'pre_edge', 'resonant'
    """
    regions = {
        'long_q': [],      # (250.0, 283.7)
        'pre_edge': [],    # (>250, <283.7)
        'resonant': []     # (>283.7)
    }

    for e in energies:
        if e == 250.0:
            regions['long_q'].append(e)
        elif 250.0 < e < 283.7:
            regions['pre_edge'].append(e)
        elif e >= 283.7:
            regions['resonant'].append(e)

    return regions


def calculate_fit_metrics(objective) -> Dict[str, float]:
    """
    Calculate comprehensive fit metrics for an objective.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to analyze

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics
    """
    return {
        'chi2': chi2(objective),
        'reduced_chi2': reduced_chi2(objective),
        'aic': aic(objective),
        'bic': bic(objective),
        'rmsd': rmsd(objective),
        'npoints': objective.npoints,
        'nparams': len(objective.varying_parameters()),
    }


def calculate_residuals(objective, pol: str = 'both') -> Dict[str, np.ndarray]:
    """
    Calculate residuals for each polarization.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to analyze
    pol : str
        's', 'p', or 'both'

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'q', 'residual', 'normalized_residual' for each pol
    """
    residuals = {}

    for p in ['s', 'p'] if pol == 'both' else [pol]:
        pol_data = getattr(objective.data, p)
        q = pol_data.x
        y_data = pol_data.y
        y_err = pol_data.y_err

        # Get model prediction
        y_model = rxr(q, objective.model, p)

        # Calculate residuals
        residual = y_data - y_model
        normalized_residual = residual / y_err

        residuals[p] = {
            'q': q,
            'residual': residual,
            'normalized_residual': normalized_residual,
            'data': y_data,
            'model': y_model,
            'error': y_err,
        }

    return residuals


def calculate_cumulative_metrics(
    objective,
    pol: str = 'both',
    sort_by_q: bool = True
) -> Dict[str, np.ndarray]:
    """
    Calculate cumulative fit metrics as more q points are added.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to analyze
    pol : str
        's', 'p', or 'both'
    sort_by_q : bool
        If True, sort by q before calculating cumulative metrics

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with cumulative chi2, reduced_chi2, etc.
    """
    residuals_dict = calculate_residuals(objective, pol)

    cumulative = {}

    for p in ['s', 'p'] if pol == 'both' else [pol]:
        res = residuals_dict[p]
        q = res['q']
        normalized_residual = res['normalized_residual']

        if sort_by_q:
            sort_idx = np.argsort(q)
            q = q[sort_idx]
            normalized_residual = normalized_residual[sort_idx]

        # Cumulative chi2 (sum of squared normalized residuals)
        cumulative_chi2 = np.cumsum(normalized_residual**2)

        # Cumulative RMSD
        cumulative_rmsd = np.sqrt(cumulative_chi2 / np.arange(1, len(q) + 1))

        cumulative[p] = {
            'q': q,
            'cumulative_chi2': cumulative_chi2,
            'cumulative_rmsd': cumulative_rmsd,
            'npoints': np.arange(1, len(q) + 1),
        }

    return cumulative


def extrapolate_model(
    objective,
    q_extrapolated: np.ndarray,
    pol: str = 'both'
) -> Dict[str, np.ndarray]:
    """
    Calculate model predictions for q values outside the data range.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to analyze
    q_extrapolated : np.ndarray
        Q values to extrapolate to
    pol : str
        's', 'p', or 'both'

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with extrapolated reflectivity for each pol
    """
    extrapolated = {}

    for p in ['s', 'p'] if pol == 'both' else [pol]:
        r_extrapolated = rxr(q_extrapolated, objective.model, p)
        extrapolated[p] = {
            'q': q_extrapolated,
            'r': r_extrapolated,
        }

    return extrapolated


def plot_residuals_comparison(
    objectives_fixed: Dict[float, object],
    objectives_free: Dict[float, object],
    energies: List[float],
    region_name: str,
    ax: Axes = None,
) -> Tuple[Figure, Axes]:
    """
    Plot residuals comparison for fixed vs free models.

    Parameters
    ----------
    objectives_fixed : Dict[float, object]
        Dictionary of objectives keyed by energy (DFT-constrained)
    objectives_free : Dict[float, object]
        Dictionary of objectives keyed by energy (Free-model)
    energies : List[float]
        List of energies to plot
    region_name : str
        Name of the energy region
    ax : Axes, optional
        Axes to plot on

    Returns
    -------
    Tuple[Figure, Axes]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    colors = plt.cm.tab20(np.linspace(0, 1, len(energies)))

    for i, e in enumerate(energies):
        if e not in objectives_fixed or e not in objectives_free:
            continue

        obj_fixed = objectives_fixed[e]
        obj_free = objectives_free[e]

        # Get residuals for both polarizations
        res_fixed = calculate_residuals(obj_fixed, 'both')
        res_free = calculate_residuals(obj_free, 'both')

        color = colors[i]

        # Plot s-polarization
        for p, marker in [('s', 'o'), ('p', 's')]:
            q_fixed = res_fixed[p]['q']
            norm_res_fixed = res_fixed[p]['normalized_residual']
            q_free = res_free[p]['q']
            norm_res_free = res_free[p]['normalized_residual']

            # Plot fixed model residuals
            ax.scatter(
                q_fixed,
                norm_res_fixed,
                marker=marker,
                color=color,
                alpha=0.6,
                s=20,
                label=f'{e:.1f} eV {p}-pol (DFT)' if i == 0 and p == 's' else '',
            )

            # Plot free model residuals (different marker style)
            ax.scatter(
                q_free,
                norm_res_free,
                marker=marker,
                color=color,
                alpha=0.3,
                s=20,
                facecolors='none',
                edgecolors=color,
                linewidths=1,
                label=f'{e:.1f} eV {p}-pol (Free)' if i == 0 and p == 's' else '',
            )

    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(1, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axhline(-1, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r'$Q$ ($\AA^{-1}$)')
    ax.set_ylabel('Normalized Residuals')
    ax.set_title(f'Residuals Comparison: {region_name}')
    ax.legend(ncol=2, fontsize='small', loc='best')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_cumulative_metrics(
    objectives_fixed: Dict[float, object],
    objectives_free: Dict[float, object],
    energies: List[float],
    region_name: str,
    ax: Axes = None,
) -> Tuple[Figure, Axes]:
    """
    Plot cumulative metrics as function of q points.

    Parameters
    ----------
    objectives_fixed : Dict[float, object]
        Dictionary of objectives keyed by energy (DFT-constrained)
    objectives_free : Dict[float, object]
        Dictionary of objectives keyed by energy (Free-model)
    energies : List[float]
        List of energies to plot
    region_name : str
        Name of the energy region
    ax : Axes, optional
        Axes to plot on

    Returns
    -------
    Tuple[Figure, Axes]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    colors = plt.cm.tab20(np.linspace(0, 1, len(energies)))

    for i, e in enumerate(energies):
        if e not in objectives_fixed or e not in objectives_free:
            continue

        obj_fixed = objectives_fixed[e]
        obj_free = objectives_free[e]

        # Get cumulative metrics
        cum_fixed = calculate_cumulative_metrics(obj_fixed, 'both')
        cum_free = calculate_cumulative_metrics(obj_free, 'both')

        color = colors[i]

        # Plot for both polarizations combined
        for p in ['s', 'p']:
            npoints_fixed = cum_fixed[p]['npoints']
            cum_chi2_fixed = cum_fixed[p]['cumulative_chi2']
            npoints_free = cum_free[p]['npoints']
            cum_chi2_free = cum_free[p]['cumulative_chi2']

            # Plot fixed model
            ax.plot(
                npoints_fixed,
                cum_chi2_fixed,
                color=color,
                linestyle='-',
                alpha=0.7,
                linewidth=1,
                label=f'{e:.1f} eV {p}-pol (DFT)' if i == 0 else '',
            )

            # Plot free model
            ax.plot(
                npoints_free,
                cum_chi2_free,
                color=color,
                linestyle='--',
                alpha=0.7,
                linewidth=1,
                label=f'{e:.1f} eV {p}-pol (Free)' if i == 0 else '',
            )

    ax.set_xlabel('Number of Q Points')
    ax.set_ylabel('Cumulative χ²')
    ax.set_title(f'Cumulative Metrics: {region_name}')
    ax.legend(ncol=2, fontsize='small', loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    return fig, ax


def plot_extrapolation_comparison(
    objectives_fixed: Dict[float, object],
    objectives_free: Dict[float, object],
    energies: List[float],
    region_name: str,
    q_extrapolated: np.ndarray = None,
    ax: Axes = None,
) -> Tuple[Figure, Axes]:
    """
    Compare model extrapolations beyond data range.

    Parameters
    ----------
    objectives_fixed : Dict[float, object]
        Dictionary of objectives keyed by energy (DFT-constrained)
    objectives_free : Dict[float, object]
        Dictionary of objectives keyed by energy (Free-model)
    energies : List[float]
        List of energies to plot
    region_name : str
        Name of the energy region
    q_extrapolated : np.ndarray, optional
        Q values to extrapolate to. If None, extends beyond data range.
    ax : Axes, optional
        Axes to plot on

    Returns
    -------
    Tuple[Figure, Axes]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    colors = plt.cm.tab20(np.linspace(0, 1, len(energies)))

    for i, e in enumerate(energies):
        if e not in objectives_fixed or e not in objectives_free:
            continue

        obj_fixed = objectives_fixed[e]
        obj_free = objectives_free[e]

        # Get data range
        q_data_s = obj_fixed.data.s.x
        q_data_p = obj_fixed.data.p.x
        q_min = min(q_data_s.min(), q_data_p.min())
        q_max = max(q_data_s.max(), q_data_p.max())

        # Create extrapolation range if not provided
        if q_extrapolated is None:
            q_extrapolated = np.linspace(q_max, q_max * 1.5, 100)

        # Get extrapolated values
        ext_fixed = extrapolate_model(obj_fixed, q_extrapolated, 'both')
        ext_free = extrapolate_model(obj_free, q_extrapolated, 'both')

        color = colors[i]

        # Plot for both polarizations
        for p, linestyle in [('s', '-'), ('p', '--')]:
            # Plot fixed model extrapolation
            ax.plot(
                ext_fixed[p]['q'],
                ext_fixed[p]['r'],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.7,
                label=f'{e:.1f} eV {p}-pol (DFT)' if i == 0 else '',
            )

            # Plot free model extrapolation
            ax.plot(
                ext_free[p]['q'],
                ext_free[p]['r'],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.5,
                label=f'{e:.1f} eV {p}-pol (Free)' if i == 0 else '',
            )

        # Mark data range
        ax.axvline(q_max, color='k', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r'$Q$ ($\AA^{-1}$)')
    ax.set_ylabel('Reflectivity')
    ax.set_title(f'Extrapolation Comparison: {region_name}')
    ax.set_yscale('log')
    ax.legend(ncol=2, fontsize='small', loc='best')
    ax.grid(True, alpha=0.3)

    return fig, ax


def create_metrics_table(
    objectives_fixed: Dict[float, object],
    objectives_free: Dict[float, object],
    energies: List[float],
) -> pd.DataFrame:
    """
    Create a comprehensive metrics table comparing fixed vs free models.

    Parameters
    ----------
    objectives_fixed : Dict[float, object]
        Dictionary of objectives keyed by energy (DFT-constrained)
    objectives_free : Dict[float, object]
        Dictionary of objectives keyed by energy (Free-model)
    energies : List[float]
        List of energies to analyze

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each energy and model type
    """
    rows = []

    for e in energies:
        if e not in objectives_fixed or e not in objectives_free:
            continue

        obj_fixed = objectives_fixed[e]
        obj_free = objectives_free[e]

        metrics_fixed = calculate_fit_metrics(obj_fixed)
        metrics_free = calculate_fit_metrics(obj_free)

        row = {
            'energy': e,
            'model_type': 'DFT',
            **metrics_fixed,
        }
        rows.append(row)

        row = {
            'energy': e,
            'model_type': 'Free',
            **metrics_free,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    """Main analysis function."""
    # Paths
    base_path = Path(__file__).parent.parent / 'fitting'
    data_path = base_path / 'reflectivity_data.parquet'

    # Load data
    print("Loading reflectivity data...")
    loaded_data = load_reflectivity_dataset(data_path)

    # Load fitting results
    print("Loading fitting results...")
    results_fixed = load_fitting_results(base_path / 'fitting_results_fixed.pkl')
    results_fixed_2 = load_fitting_results(base_path / 'fitting_results_fixed_2.pkl')
    results_free = load_fitting_results(base_path / 'fitting_results_free_model.pkl')
    results_free_2 = load_fitting_results(base_path / 'fitting_results_free_model_2.pkl')

    # Organize objectives by energy
    def get_objectives_dict(results):
        return {obj.model.energy: obj for obj in results.objectives}

    objectives_fixed = get_objectives_dict(results_fixed)
    objectives_fixed_2 = get_objectives_dict(results_fixed_2)
    objectives_free = get_objectives_dict(results_free)
    objectives_free_2 = get_objectives_dict(results_free_2)

    # Get all energies
    all_energies = sorted(set(
        list(objectives_fixed.keys()) +
        list(objectives_fixed_2.keys()) +
        list(objectives_free.keys()) +
        list(objectives_free_2.keys())
    ))

    # Classify energy regions
    regions = classify_energy_regions(all_energies)

    print(f"\nEnergy regions:")
    print(f"  Long q range (250.0, 283.7): {regions['long_q']}")
    print(f"  Pre-edge (>250, <283.7): {regions['pre_edge']}")
    print(f"  Resonant (>283.7): {regions['resonant']}")

    # Create metrics tables for each file pair
    print("\nCreating metrics tables...")
    metrics_fixed = create_metrics_table(objectives_fixed, objectives_free, all_energies)
    metrics_fixed_2 = create_metrics_table(objectives_fixed_2, objectives_free_2, all_energies)

    print("\nMetrics for fixed/free pair 1:")
    print(metrics_fixed.groupby(['energy', 'model_type'])[['reduced_chi2', 'aic', 'bic']].mean())

    print("\nMetrics for fixed/free pair 2:")
    print(metrics_fixed_2.groupby(['energy', 'model_type'])[['reduced_chi2', 'aic', 'bic']].mean())

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

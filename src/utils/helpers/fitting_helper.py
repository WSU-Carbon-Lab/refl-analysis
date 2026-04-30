"""
Helper functions for reflectivity fitting analysis.

This module contains utility functions for calculating fitting statistics
and analyzing reflectivity data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import ufloat
from pyref.fitting import AnisotropyObjective, GlobalObjective, Objective, CurveFitter

type ObjectiveType = Objective | GlobalObjective | AnisotropyObjective

# Define a subtle interface into the objective class that ensures there is
# a way to get the ndata and nparams from the objective. This is most
# important for the global objective. that contains multiple objectives.


class ObjectiveInterface:
    def __init__(self, objective: ObjectiveType):
        self.objective: ObjectiveType = objective

    def ndata(self) -> int:
        if isinstance(self.objective, GlobalObjective):
            return sum(
                len(obj.data.s.x) + len(obj.data.p.x)
                for obj in self.objective.objectives
            )
        elif isinstance(self.objective, AnisotropyObjective):
            return len(self.objective.data.s.x) + len(self.objective.data.p.x)  # pyright: ignore[reportAttributeAccessIssue]
        elif isinstance(self.objective, Objective):
            return len(self.objective.data.s.x) + len(self.objective.data.p.x)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            raise ValueError(f"Unsupported objective type: {type(self.objective)}")

    def nparams(self) -> int:
        return len(self.objective.varying_parameters())

def reduced_chi2(objective: ObjectiveType) -> float:
    """
    Calculate reduced chi-squared statistic.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to calculate chi-squared for

    Returns
    -------
    float
        Reduced chi-squared value
    """
    ndata = ObjectiveInterface(objective).ndata()
    nparams = ObjectiveInterface(objective).nparams()
    return objective.chisqr() / (ndata - nparams)


def aic(objective):
    """
    Calculate Akaike Information Criterion.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to calculate AIC for

    Returns
    -------
    float
        AIC value
    """
    nparams = ObjectiveInterface(objective).nparams()
    return objective.chisqr() + 2 * nparams


def bic(objective):
    """
    Calculate Bayesian Information Criterion.

    Parameters
    ----------
    objective : refnx.analysis.Objective
        The objective to calculate BIC for

    Returns
    -------
    float
        BIC value
    """
    ndata = ObjectiveInterface(objective).ndata()
    nparams = ObjectiveInterface(objective).nparams()
    return objective.chisqr() + nparams * np.log(ndata)


def rxr(x, model, pol):
    """
    Calculate reflectivity for a given polarization.

    Parameters
    ----------
    x : array-like
        Q values
    model : refnx.reflect.ReflectModel
        The reflectivity model
    pol : str
        Polarization ('s' or 'p')

    Returns
    -------
    array-like
        Reflectivity values
    """
    _pol = model.pol
    model.pol = pol
    y = model(x)
    model.pol = _pol
    return y


def anisotropy(x, model):
    """
    Calculate anisotropy from model.

    Parameters
    ----------
    x : array-like
        Q values
    model : refnx.reflect.ReflectModel
        The reflectivity model

    Returns
    -------
    array-like
        Anisotropy values: (R_p - R_s) / (R_p + R_s)
    """
    r_s = rxr(x, model, "s")
    r_p = rxr(x, model, "p")
    return (r_p - r_s) / (r_p + r_s)

def package_results(
    *,
    fitter: CurveFitter | None = None,
    objective: ObjectiveType | None = None,
    include_statistics: bool = True,
    ) -> pd.DataFrame:
    params = []
    # Ensure that we have at least one source of an objective, either
    # through the fitter or through the objective itself.
    if objective is None and fitter is None:
        raise ValueError("No objective or fitter provided")
    elif objective is None and fitter is not None:
        objective = fitter.objective
    elif fitter is None and objective is not None:
        fitter = CurveFitter(objective)
    else:
        raise ValueError("Both fitter and objective provided")
    for p in objective.varying_parameters():  # pyright: ignore[reportOptionalMemberAccess]
        err = p.stderr if p.stderr is not None else 0
        params.append(
            {
                "name": p.name,
                "value": ufloat(p.value, err),
            }
        )
    # Add fit statistics
    if include_statistics:
        params.append(
            {
                "name": "reduced_chi2",
                "value": reduced_chi2(fitter.objective),
            }
        )
        params.append(
            {
                "name": "aic",
                "value": aic(fitter.objective),
            }
        )
        params.append(
            {
                "name": "bic",
                "value": bic(fitter.objective),
            }
        )
    return pd.DataFrame(params).set_index("name")

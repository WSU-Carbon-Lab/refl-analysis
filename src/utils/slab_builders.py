from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pyref.fitting as fit
from pyref.fitting.structure import Slab

DEFAULT_FIXED_PARAMS: tuple[str, ...] = ("thick", "rough", "density")

_RESOLVE_DIRECT: dict[str, str] = {"thick": "thick", "rough": "rough"}


def _resolve_param(slab: Any, name: str) -> Any:
    if name in _RESOLVE_DIRECT:
        return getattr(slab, _RESOLVE_DIRECT[name], None)
    if name == "density":
        sld = getattr(slab, "sld", None)
        if sld is None:
            return None
        return getattr(sld, "density", None) or getattr(sld, "rho", None)
    if name == "rotation":
        sld = getattr(slab, "sld", None)
        return None if sld is None else getattr(sld, "rotation", None)
    obj = slab
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def resolve_param(slab: Any, name: str) -> Any:
    """
    Resolve a logical parameter name to a refnx Parameter on a slab.

    Parameters
    ----------
    slab : object
        Slab or component (e.g. ReflectSlice, Slab)
    name : str
        One of "thick", "rough", "density", "rotation", or a dotted path
        (e.g. "sld.density")

    Returns
    -------
    Parameter or None
        The Parameter if found, else None
    """
    return _resolve_param(slab, name)


def safely_setp_param(param: Any, **kwargs: Any) -> None:
    """
    Set Parameter attributes safely. When setting a constraint, use vary=None
    to avoid conflict (refnx requirement).

    Parameters
    ----------
    param : refnx.analysis.Parameter
        The parameter to modify
    **kwargs
        Passed to param.setp (value, vary, bounds, constraint)
    """
    if kwargs.get("vary", False) and kwargs.get("constraint") is not None:
        kwargs = {**kwargs, "vary": None}
    param.setp(**kwargs)


def safely_setp(slab: Any, param: str, **kwargs: Any) -> None:
    """
    Safely set a parameter on a slab by logical name. Uses vary=None when
    both vary=True and constraint are given.

    Parameters
    ----------
    slab : object
        Slab or component
    param : str
        Logical name: "thick", "rough", "density", "rotation", or dotted path
    **kwargs
        Passed to Parameter.setp
    """
    p = _resolve_param(slab, param)
    if p is not None:
        safely_setp_param(p, **kwargs)


def select(
    objective: fit.AnisotropyObjective | fit.GlobalObjective | fit.Objective,
    slab_name: str,
    energy: float | None = None,
) -> Slab:
    """
    Return the slab component whose name starts with slab_name in the
    objective for the given energy.

    Parameters
    ----------
    objective : fit.AnisotropyObjective | fit.GlobalObjective | fit.Objective
        Anisotropy objective containing a model
    slab_name : str
        Prefix of slab name (e.g. "Surface", "ZnPc")
    energy : float | None
        Model energy in eV

    Returns
    -------
    component
        The matching slab component

    Raises
    ------
    ValueError
        If no matching energy or slab found
    """
    if isinstance(objective, (fit.AnisotropyObjective, fit.Objective)):
        for s in objective.model.structure.components:
            if s.name.startswith(slab_name):
                return s
        raise ValueError(f"No slab {slab_name!r} found in energy {energy}")
    elif isinstance(objective, fit.GlobalObjective):
        for o in objective.objectives:
            if o.model.energy == energy:
                for s in o.model.structure.components:
                    if s.name.startswith(slab_name):
                        return s
                raise ValueError(f"No slab {slab_name!r} found in energy {energy}")
    else:
        raise ValueError(f"Unsupported objective type: {type(objective)}")


def constrain_to_template(
    new_slab: Any,
    template_slab: Any,
    param_names: Sequence[str] = DEFAULT_FIXED_PARAMS,
) -> None:
    """
    Constrain parameters on new_slab to those on template_slab via
    setp(constraint=template_param, vary=None). Only parameters present
    on both slabs are constrained.

    Parameters
    ----------
    new_slab : object
        Slab to constrain (e.g. new energy)
    template_slab : object
        Reference slab to constrain to
    param_names : sequence of str
        Logical names to constrain: thick, rough, density, rotation, etc.
    """
    for name in param_names:
        p_new = _resolve_param(new_slab, name)
        p_tpl = _resolve_param(template_slab, name)
        if p_new is not None and p_tpl is not None:
            safely_setp_param(p_new, constraint=p_tpl, vary=None)


def slab_from_template(
    template_slab: Any,
    energy: float,
    builder: Callable[..., Any],
    param_names: Sequence[str] = DEFAULT_FIXED_PARAMS,
    **builder_kwargs: Any,
) -> Any:
    """
    Build a new slab for a given energy from a template using a builder,
    then fix thickness, roughness, and density (or a custom list of
    parameter names) between template and new slab via constraints.

    Parameters
    ----------
    template_slab : object
        Reference slab to constrain to
    energy : float
        Energy in eV for the new slab
    builder : callable
        Builder (energy, **kwargs) -> slab. Same material type as template.
    param_names : sequence of str
        Parameter names to constrain (default: thick, rough, density)
    **builder_kwargs
        Passed to builder(energy, **builder_kwargs)

    Returns
    -------
    slab
        New slab with specified parameters constrained to template
    """
    new_slab = builder(energy, **builder_kwargs)
    constrain_to_template(new_slab, template_slab, param_names)
    return new_slab

ZNPC = "C32H16N8Zn"
MA = np.asin(np.sqrt(2 / 3))


def vacuum(energy):
    """Vacuum."""
    slab = fit.MaterialSLD("", 0, name=f"Vacuum_{energy}")(0, 0)
    slab.thick.setp(vary=False)
    slab.rough.setp(vary=False)
    slab.sld.density.setp(vary=False)
    return slab


def substrate(energy, thick=0, rough=1.2, density=2.44):
    """Substrate."""
    slab = fit.MaterialSLD(
        "Si", density=density, energy=energy, name=f"Substrate_{energy}"
    )(thick, rough)
    slab.thick.setp(vary=False)
    slab.rough.setp(vary=False)
    slab.sld.density.setp(vary=False, bounds=(2, 3))
    return slab


def sio2(energy, thick=8.22, rough=6.153, density=2.15):
    """SiO2."""
    slab = fit.MaterialSLD(
        "SiO2", density=density, energy=energy, name=f"Oxide_{energy}"
    )(thick, rough)
    slab.thick.setp(vary=True, bounds=(0, 12))
    slab.rough.setp(vary=True, bounds=(0, 8))
    slab.sld.density.setp(vary=False, bounds=(1, 2.3))
    return slab


def contamination(energy, ooc, thick=4.4, rough=2, density=1.0):
    """Contamination."""
    name = f"Contamination_{energy}"
    slab = fit.UniTensorSLD(
        ooc, density=density, rotation=0.81, energy=energy, name=name
    )(thick, rough)
    slab.sld.density.setp(vary=True, bounds=(1, 1.8))
    slab.sld.rotation.setp(vary=True, bounds=(np.pi / 4, 7 * np.pi / 8))

    slab.thick.setp(vary=True, bounds=(0, 12))
    slab.rough.setp(vary=True, bounds=(0, 5))
    return slab


def surface(energy, ooc, thick=3.3, rough=1, density=1.0):
    """Surface."""
    name = f"Surface_{energy}"
    slab = fit.UniTensorSLD(
        ooc, density=density, rotation=0.8, energy=energy, name=name
    )(thick, rough)
    slab.sld.density.setp(vary=True, bounds=(1, 1.8))
    slab.sld.rotation.setp(vary=True, bounds=(0, np.pi / 4))

    slab.thick.setp(vary=True, bounds=(0, 12))
    slab.rough.setp(vary=True, bounds=(0, 5))
    return slab


def znpc(energy, ooc, thick=191, rough=8.8, density=1.61):
    """ZnPc."""
    name = f"ZnPc_{energy}"
    slab = fit.UniTensorSLD(
        ooc, density=density, rotation=1.35, energy=energy, name=name
    )(thick, rough)
    slab.sld.density.setp(vary=True, bounds=(1.2, 1.8))
    slab.sld.rotation.setp(vary=True, bounds=(MA, np.pi / 2))

    slab.thick.setp(vary=True, bounds=(180, 210))
    slab.rough.setp(vary=True, bounds=(2, 16))
    return slab

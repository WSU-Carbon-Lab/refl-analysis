"""Retarget an existing reflectivity objective to another photon energy in place.

Unlike :mod:`~utils.graded_objective` template rebuilds, these helpers keep the
same ``Structure`` instance and fitted parameters, update
:class:`~pyref.fitting.model.ReflectModel` energy, and broadcast dispersive
lookup on substrates (``MaterialSLD``), legacy ``UniTensorSLD`` slabs, and
:class:`~refloxide.pxr.energy.bookended.EnergyBookendedOrientationDensityProfile`
OOC caches. Use for prediction and multi-energy figures when geometry is fixed.

Examples
--------
>>> from utils.objective_energy import objective_retarget_energy
>>> obj_250 = objective_retarget_energy(fit_obj, 250.0, data=data["250.0"])
>>> objs = objectives_retarget_energies(fit_obj, [250.0, 283.7, 287.0], data=data)
"""

from __future__ import annotations

import copy
import numbers
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from refloxide.pxr.energy.fused import effective_energy_ev, find_bookended_profile
from refloxide.pxr.plugin.structure import Slab

from utils.graded_objective import resolve_data_key
from utils.models import AnisotropyObjective, ReflectModel, Transform

if TYPE_CHECKING:
    from refloxide.pxr.plugin.structure import Scatterer


def iter_slab_scatterers(structure: Any) -> Iterator[tuple[Slab, Scatterer]]:
    """Yield ``(slab, scatterer)`` pairs for every slab component in ``structure``.

    Parameters
    ----------
    structure
        Pyref or refloxide ``Structure``.

    Yields
    ------
    Slab, Scatterer
        Slab components only; book-ended profiles are not slabs.
    """
    for component in structure.components:
        if isinstance(component, Slab):
            yield component, component.sld


def _is_nominal_energy_attribute(sld: Any) -> bool:
    energy = getattr(sld, "energy", None)
    if energy is None:
        return False
    if hasattr(energy, "value"):
        return False
    return isinstance(energy, (numbers.Real, np.floating)) and not isinstance(
        energy, (bool, np.bool_)
    )


def set_scatterer_nominal_energy(sld: Any, base_energy_ev: float) -> bool:
    """Assign ``base_energy_ev`` to a legacy energy-tagged scatterer when applicable.

    Parameters
    ----------
    sld
        ``MaterialSLD``, ``UniTensorSLD``, or similar with a float ``energy`` field.
    base_energy_ev
        Photon energy in eV before per-scatterer ``energy_offset``.

    Returns
    -------
    bool
        ``True`` when ``energy`` was updated.
    """
    if not _is_nominal_energy_attribute(sld):
        return False
    sld.energy = float(base_energy_ev)
    return True


def broadcast_structure_probe_energy(
    structure: Any,
    base_energy_ev: float,
    *,
    structure_energy_offset_ev: float = 0.0,
) -> None:
    """Broadcast photon energy across substrate slabs and graded film OOC caches.

    Updates every ``MaterialSLD`` / ``UniTensorSLD`` nominal ``energy`` so
    ``Slab.slabs()`` and the fused Rust book-ended path see dispersive SiO2/Si
    (and vacuum) at ``base_energy_ev`` plus each scatterer's own
    ``energy_offset``. Refreshes
    :class:`~refloxide.pxr.energy.bookended.EnergyBookendedOrientationDensityProfile`
    OOC caches at the effective film energy.

    Parameters
    ----------
    structure
        Stack to update in place (vac | film | backing layers).
    base_energy_ev
        Experiment photon energy in eV (typically ``ReflectModel.energy``).
    structure_energy_offset_ev
        Global offset in eV from ``ReflectModel.energy_offset`` (or structure).
    """
    base = float(base_energy_ev)
    struct_off = float(structure_energy_offset_ev)
    for _slab, sld in iter_slab_scatterers(structure):
        set_scatterer_nominal_energy(sld, base)
    located = find_bookended_profile(structure)
    if located is not None:
        _idx, profile = located
        profile.clear_ooc_cache()
        query_ev = effective_energy_ev(profile, base, struct_off)
        profile.cache_ooc_at(query_ev)


def structure_energy_offset_ev(model: ReflectModel) -> float:
    """Return the global structure/model energy offset in eV."""
    off = model.energy_offset.value
    return float(off) if off is not None else 0.0


def reflect_model_retarget_energy(
    model: ReflectModel,
    energy_ev: float,
    *,
    broadcast_substrates: bool = True,
) -> ReflectModel:
    """Set ``model.energy`` and optionally broadcast dispersive materials.

    Parameters
    ----------
    model
        Reflectivity model whose structure is updated when
        ``broadcast_substrates`` is ``True``.
    energy_ev
        Target photon energy in eV.
    broadcast_substrates
        When ``True``, call :func:`broadcast_structure_probe_energy` on
        ``model.structure``.

    Returns
    -------
    ReflectModel
        The same ``model`` instance (mutated).
    """
    model.energy = float(energy_ev)
    if broadcast_substrates:
        broadcast_structure_probe_energy(
            model.structure,
            float(energy_ev),
            structure_energy_offset_ev=structure_energy_offset_ev(model),
        )
    return model


def _anisotropy_kwargs_from_reference(
    reference: AnisotropyObjective,
    *,
    logp_anisotropy_weight: float | None,
    transform: Transform | Literal["logY", "linear"] | None,
) -> dict[str, Any]:
    weight = (
        reference.logp_anisotropy_weight
        if logp_anisotropy_weight is None
        else logp_anisotropy_weight
    )
    if transform is None:
        transform = reference.transform
    elif isinstance(transform, str):
        transform = Transform(transform)
    return {
        "logp_anisotropy_weight": weight,
        "transform": transform,
    }


def objective_retarget_energy(  # noqa: PLR0913
    reference_objective: AnisotropyObjective,
    energy_ev: float,
    *,
    data: Any | None = None,
    deepcopy_objective: bool = True,
    broadcast_substrates: bool = True,
    logp_anisotropy_weight: float | None = None,
    transform: Transform | Literal["logY", "linear"] | None = None,
) -> AnisotropyObjective:
    """Clone or reuse an objective, changing only photon energy and dispersive lookup.

    Preserves the fitted ``Structure``, film geometry, instrumentation parameters,
    and parameter values. Substrate and film optical constants are re-evaluated at
    ``energy_ev`` via :func:`broadcast_structure_probe_energy`; no DFT template
    rebuild is performed.

    Parameters
    ----------
    reference_objective
        Source objective (e.g. graded fit at 283.7 eV).
    energy_ev
        Target photon energy in eV.
    data
        Dataset for the new objective. When omitted, uses ``reference_objective.data``
        only if ``energy_ev`` matches ``reference_objective.model.energy``; otherwise
        raises ``ValueError``.
    deepcopy_objective
        When ``True``, deep-copy the objective so the reference is untouched.
    broadcast_substrates
        When ``True``, update ``MaterialSLD`` / film OOC caches before return.
    logp_anisotropy_weight, transform
        Forwarded to :class:`~pyref.fitting.AnisotropyObjective` when building a
        new wrapper; ignored when reusing the copied objective instance.

    Returns
    -------
    AnisotropyObjective
        Objective ready for ``plot`` or ``model`` at ``energy_ev``.

    Raises
    ------
    ValueError
        When ``data`` is required but not supplied.
    KeyError
        When ``data`` is a dict and has no entry for ``energy_ev``.
    """
    ref_energy = float(reference_objective.model.energy)
    target = float(energy_ev)

    if deepcopy_objective:
        objective = copy.deepcopy(reference_objective)
    else:
        objective = reference_objective

    reflect_model_retarget_energy(
        objective.model,
        target,
        broadcast_substrates=broadcast_substrates,
    )

    if data is None:
        if not np.isclose(target, ref_energy):
            msg = (
                "objective_retarget_energy requires `data` when energy differs "
                f"from the reference ({ref_energy} eV vs {target} eV)"
            )
            raise ValueError(msg)
        dataset = reference_objective.data
    elif isinstance(data, dict):
        key = resolve_data_key(data, target)
        dataset = data[key]
    else:
        dataset = data

    if deepcopy_objective:
        kwargs = _anisotropy_kwargs_from_reference(
            reference_objective,
            logp_anisotropy_weight=logp_anisotropy_weight,
            transform=transform,
        )
        return AnisotropyObjective(objective.model, dataset, **kwargs)
    objective.data = dataset
    return objective


def objectives_retarget_energies(
    reference_objective: AnisotropyObjective,
    energies: list[float],
    data: dict[str, Any],
    *,
    deepcopy_objective: bool = True,
    broadcast_substrates: bool = True,
    **kwargs: Any,
) -> dict[float, AnisotropyObjective]:
    """Build one energy-retargeted objective per photon energy.

    Parameters
    ----------
    reference_objective
        Fitted objective at a reference energy.
    energies
        Photon energies in eV to evaluate.
    data
        Reflectivity datasets keyed by energy string (``read_xrr`` output).
    deepcopy_objective, broadcast_substrates
        Forwarded to :func:`objective_retarget_energy`.
    **kwargs
        Additional keyword arguments forwarded to :func:`objective_retarget_energy`.

    Returns
    -------
    dict[float, AnisotropyObjective]
        Objectives keyed by float energy in eV.
    """
    out: dict[float, AnisotropyObjective] = {}
    for energy in energies:
        energy_f = float(energy)
        out[energy_f] = objective_retarget_energy(
            reference_objective,
            energy_f,
            data=data,
            deepcopy_objective=deepcopy_objective,
            broadcast_substrates=broadcast_substrates,
            **kwargs,
        )
    return out


__all__ = [
    "broadcast_structure_probe_energy",
    "iter_slab_scatterers",
    "objective_retarget_energy",
    "objectives_retarget_energies",
    "reflect_model_retarget_energy",
    "set_scatterer_nominal_energy",
    "structure_energy_offset_ev",
]

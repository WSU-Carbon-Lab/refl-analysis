"""Rebuild graded book-ended reflectivity objectives for figures and multi-energy work.

Utilities copy fitted film geometry from a pickled or live
:class:`~pyref.fitting.AnisotropyObjective` onto fresh stacks with different
microslab counts or photon energies. Intended for publication plots, not refitting.

To keep the same stack and parameters while only changing photon energy and
dispersive substrate/film lookup (no DFT template rebuild), use
:mod:`~utils.objective_energy` instead.

Examples
--------
>>> from utils import read_ooc, read_xrr, read_fit
>>> from utils.models import configure_refloxide_fitting
>>> from utils.graded_objective import (
...     upsample_graded_objective,
...     graded_objective_at_energy,
... )
>>> configure_refloxide_fitting()
>>> oocs = read_ooc("dft.csv", material="znpc")
>>> data = read_xrr("reflectivity_data", material="znpc", source="hub")
>>> fit_bundle = read_fit("dft/dft_en_offset_new2.pkl", material="znpc", source="local")
>>> ref = read_fit("xrr/znpc/graded/graded_fit_latest.pkl", material="znpc")
>>> plot_obj = upsample_graded_objective(ref, num_slabs=100)
>>> obj_250 = graded_objective_at_energy(
...     ref,
...     250.0,
...     template_objective=next(
...         o for o in fit_bundle.objectives if o.model.energy == 250.0
...     ),
...     ooc=oocs,
...     data=data,
... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from refloxide.pxr.energy.bookended import EnergyBookendedOrientationDensityProfile
from refloxide.pxr.energy.fused import find_bookended_profile

from utils.models import AnisotropyObjective, ReflectModel, Transform
from utils.refloxide_profile import model_bookended_refloxide
from utils.slab_builders import safely_setp_param

if TYPE_CHECKING:
    import pandas as pd
    from refloxide.pxr.energy.ooc import OocAnchor

BOOKENDED_FILM_PARAM_NAMES: tuple[str, ...] = (
    "total_thick",
    "surface_roughness",
    "tau_si",
    "tau_vac",
    "alpha_bulk",
    "alpha_si",
    "alpha_vac",
    "density_bulk",
    "density_si",
    "density_vac",
    "energy_offset",
)

REFLECT_MODEL_INSTRUMENTATION: tuple[str, ...] = (
    "scale_s",
    "scale_p",
    "theta_offset_s",
    "theta_offset_p",
    "bkg",
    "dq",
    "q_offset",
    "energy_offset",
)


@dataclass(frozen=True, slots=True)
class GradedFitContext:
    """Book-ended graded fit handles extracted from a single-energy objective."""

    objective: AnisotropyObjective
    model: ReflectModel
    structure: Any
    film: EnergyBookendedOrientationDensityProfile
    energy: float


def resolve_data_key(data: dict[str, Any], energy: float) -> str:
    """Return the ``data`` dict key for ``energy`` (eV).

    Parameters
    ----------
    data
        Mapping from energy string labels to datasets (as returned by ``read_xrr``).
    energy
        Photon energy in eV.

    Returns
    -------
    str
        Matching key in ``data``.

    Raises
    ------
    KeyError
        When no tabulated dataset matches ``energy``.
    """
    candidates = (
        str(energy),
        f"{float(energy):.1f}",
        f"{float(energy)}",
    )
    for key in candidates:
        if key in data:
            return key
    sample = list(data)[:5]
    msg = f"No reflectivity dataset for energy {energy!r}; keys include {sample!r}..."
    raise KeyError(msg)


def bookended_film_from_structure(
    structure: Any,
) -> EnergyBookendedOrientationDensityProfile:
    """Locate the graded book-ended film in a collapsed vac | film | SiO2 | Si stack.

    Parameters
    ----------
    structure
        Pyref ``Structure`` containing one
        :class:`~refloxide.pxr.energy.bookended.EnergyBookendedOrientationDensityProfile`.

    Returns
    -------
    EnergyBookendedOrientationDensityProfile
        The film component.

    Raises
    ------
    ValueError
        When no book-ended profile is present.
    """
    located = find_bookended_profile(structure)
    if located is None:
        msg = "Structure has no EnergyBookendedOrientationDensityProfile component"
        raise ValueError(msg)
    return located[1]


def extract_graded_fit_context(objective: AnisotropyObjective) -> GradedFitContext:
    """Extract model, structure, film, and energy from a graded fit objective.

    Parameters
    ----------
    objective
        Single-energy graded :class:`~pyref.fitting.AnisotropyObjective`.

    Returns
    -------
    GradedFitContext
        Handles for rebuild helpers.
    """
    model = objective.model
    structure = model.structure
    film = bookended_film_from_structure(structure)
    return GradedFitContext(
        objective=objective,
        model=model,
        structure=structure,
        film=film,
        energy=float(model.energy),
    )


def copy_bookended_film_parameters(
    src: EnergyBookendedOrientationDensityProfile,
    dst: EnergyBookendedOrientationDensityProfile,
    *,
    vary: bool | None = False,
    copy_constraints: bool = True,
) -> None:
    """Copy book-ended film parameters from ``src`` onto ``dst``.

    Parameters
    ----------
    src, dst
        Source and destination film components.
    vary
        When not ``None``, force ``vary`` on every copied parameter. When ``None``,
        preserve each source ``vary`` flag and constraint.
    copy_constraints
        When ``True`` and ``vary is None``, copy constraints and bounds from ``src``.
    """
    for name in BOOKENDED_FILM_PARAM_NAMES:
        p_src = getattr(src, name)
        p_dst = getattr(dst, name)
        kwargs: dict[str, Any] = {"value": p_src.value}
        if vary is not None:
            kwargs["vary"] = vary
        elif copy_constraints:
            kwargs["vary"] = p_src.vary
            if p_src.constraint is not None:
                kwargs["constraint"] = p_src.constraint
                kwargs["vary"] = None
            if p_src.bounds is not None:
                kwargs["bounds"] = p_src.bounds
        safely_setp_param(p_dst, **kwargs)


def clone_bookended_film(
    profile: EnergyBookendedOrientationDensityProfile,
    *,
    num_slabs: int | None = None,
    energy: float | None = None,
    mesh_constant: float | None = None,
    name: str | None = None,
) -> EnergyBookendedOrientationDensityProfile:
    """Duplicate a book-ended film with optional mesh or nominal energy overrides.

    Parameters
    ----------
    profile
        Film to clone.
    num_slabs
        Microslab count for the new adaptive grid. Defaults to ``profile.num_slabs``.
    energy
        Nominal photon energy (eV) stored on the new component. Defaults to the
        source nominal energy.
    mesh_constant
        Mesh refinement constant. Defaults to the source value.
    name
        Component name. Defaults to the source name.

    Returns
    -------
    EnergyBookendedOrientationDensityProfile
        New film with the same anchor table; parameter values are not copied—call
        :func:`copy_bookended_film_parameters` when needed.
    """
    nominal = float(energy if energy is not None else profile._nominal_energy_ev)
    return EnergyBookendedOrientationDensityProfile(
        profile.anchor,
        total_thick=float(profile.total_thick.value or 0.0),
        surface_roughness=float(profile.surface_roughness.value or 0.0),
        density_bulk=float(profile.density_bulk.value or 1.0),
        density_si=float(profile.density_si.value or 1.0),
        density_vac=float(profile.density_vac.value or 1.0),
        tau_si=float(profile.tau_si.value or 1.0),
        tau_vac=float(profile.tau_vac.value or 1.0),
        alpha_bulk=float(profile.alpha_bulk.value or 0.0),
        alpha_si=float(profile.alpha_si.value or 0.0),
        alpha_vac=float(profile.alpha_vac.value or 0.0),
        energy=nominal,
        energy_offset=float(profile.energy_offset.value or 0.0),
        name=name if name is not None else profile.name,
        num_slabs=int(num_slabs if num_slabs is not None else profile.num_slabs),
        mesh_constant=float(
            mesh_constant if mesh_constant is not None else profile.mesh_constant
        ),
        interp=profile.anchor.interp,
    )


def structure_with_upsampled_film(
    structure: Any,
    *,
    num_slabs: int,
) -> Any:
    """Return a new stack identical to ``structure`` but with finer film microslabs.

    Parameters
    ----------
    structure
        Pyref ``Structure`` with one book-ended film component.
    num_slabs
        Target microslab count for the graded film.

    Returns
    -------
    Structure
        Rebuilt stack (vac | upsampled film | backing layers).

    Raises
    ------
    ValueError
        When no book-ended film is found.
    """
    located = find_bookended_profile(structure)
    if located is None:
        msg = "Structure has no EnergyBookendedOrientationDensityProfile component"
        raise ValueError(msg)
    film_index, film = located
    new_film = clone_bookended_film(film, num_slabs=num_slabs)
    copy_bookended_film_parameters(film, new_film, vary=None, copy_constraints=True)
    new_film.clear_ooc_cache()

    rebuilt = structure.components[0]
    for idx, component in enumerate(structure.components[1:], start=1):
        piece = new_film if idx == film_index else component
        rebuilt = rebuilt | piece
    return rebuilt


def copy_reflect_model_instrumentation(
    src: ReflectModel,
    dst: ReflectModel,
    *,
    vary: bool | None = False,
    copy_constraints: bool = True,
) -> None:
    """Copy instrument parameters (scales, offsets, background) from ``src`` to ``dst``.

    Parameters
    ----------
    src, dst
        Source and destination :class:`~pyref.fitting.model.ReflectModel` instances.
    vary
        When not ``None``, force ``vary`` on every copied parameter.
    copy_constraints
        When ``True`` and ``vary is None``, preserve source constraints and bounds.
    """
    for name in REFLECT_MODEL_INSTRUMENTATION:
        p_src = getattr(src, name)
        p_dst = getattr(dst, name)
        kwargs: dict[str, Any] = {"value": p_src.value}
        if vary is not None:
            kwargs["vary"] = vary
        elif copy_constraints:
            kwargs["vary"] = p_src.vary
            if p_src.constraint is not None:
                kwargs["constraint"] = p_src.constraint
                kwargs["vary"] = None
            if p_src.bounds is not None:
                kwargs["bounds"] = p_src.bounds
        safely_setp_param(p_dst, **kwargs)


def reflect_model_for_structure(
    reference_model: ReflectModel,
    structure: Any,
    *,
    energy: float | None = None,
    instrumentation_from: ReflectModel | None = None,
    vary: bool | None = False,
) -> ReflectModel:
    """Build a non-varying :class:`~pyref.fitting.model.ReflectModel` on ``structure``.

    Parameters
    ----------
    reference_model
        Supplies ``pol`` and default instrumentation when ``instrumentation_from``
        is omitted.
    structure
        Pyref ``Structure`` for the new model.
    energy
        Photon energy in eV. Defaults to ``reference_model.energy``.
    instrumentation_from
        Optional model whose scales, offsets, and background are copied.
    vary
        When not ``None``, force ``vary`` on all copied instrument parameters.

    Returns
    -------
    ReflectModel
        Model ready for plotting or predictive evaluation.
    """
    eval_energy = float(energy if energy is not None else reference_model.energy)
    model = ReflectModel(structure, energy=eval_energy, pol=reference_model.pol)
    src = instrumentation_from if instrumentation_from is not None else reference_model
    copy_reflect_model_instrumentation(
        src,
        model,
        vary=vary,
        copy_constraints=vary is None,
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


def upsample_graded_objective(
    reference_objective: AnisotropyObjective,
    *,
    num_slabs: int = 100,
    data: Any | None = None,
    logp_anisotropy_weight: float | None = None,
    transform: Transform | Literal["logY", "linear"] | None = None,
) -> AnisotropyObjective:
    """Rebuild a fitted objective with finer microslabs for smooth plots.

    Parameters
    ----------
    reference_objective
        Fitted single-energy graded objective (pickle or in-memory).
    num_slabs
        Microslab count for the plotting stack. Typical values: 80-120.
    data
        Dataset for the objective. Defaults to ``reference_objective.data``.
    logp_anisotropy_weight, transform
        Forwarded to :class:`~pyref.fitting.AnisotropyObjective`. Defaults match
        ``reference_objective``.

    Returns
    -------
    AnisotropyObjective
        Non-varying objective at the same energy with upsampled film mesh.
    """
    ctx = extract_graded_fit_context(reference_objective)
    structure_hd = structure_with_upsampled_film(ctx.structure, num_slabs=num_slabs)
    film_hd = bookended_film_from_structure(structure_hd)
    film_hd.cache_ooc_at(ctx.energy)
    model_hd = reflect_model_for_structure(ctx.model, structure_hd, vary=False)
    dataset = reference_objective.data if data is None else data
    kwargs = _anisotropy_kwargs_from_reference(
        reference_objective,
        logp_anisotropy_weight=logp_anisotropy_weight,
        transform=transform,
    )
    return AnisotropyObjective(model_hd, dataset, **kwargs)


def graded_structure_at_energy(  # noqa: PLR0913
    energy: float,
    ooc: pd.DataFrame | OocAnchor,
    template_objective: Any,
    *,
    num_slabs: int = 24,
    mesh_constant: float = 0.1,
    interp: str = "linear",
) -> Any:
    """Build a fresh graded stack at ``energy`` using a multi-slab DFT template.

    Parameters
    ----------
    energy
        Photon energy in eV.
    ooc
        Full optical-constant table (not energy-windowed).
    template_objective
        Objective whose ``model.structure`` has vacuum + three ZnPc slabs + SiO2 + Si.
    num_slabs, mesh_constant, interp
        Forwarded to :func:`~utils.refloxide_profile.model_bookended_refloxide`.

    Returns
    -------
    Structure
        Collapsed vac | book-ended film | SiO2 | Si stack at ``energy``.
    """
    return model_bookended_refloxide(
        template_objective,
        ooc,
        float(energy),
        num_slabs=num_slabs,
        mesh_constant=mesh_constant,
        interp=interp,
    )


def graded_objective_at_energy(  # noqa: PLR0913
    reference_objective: AnisotropyObjective,
    energy: float,
    *,
    template_objective: Any,
    ooc: pd.DataFrame | OocAnchor,
    data: dict[str, Any],
    num_slabs: int = 24,
    mesh_constant: float = 0.1,
    instrumentation: Literal["template", "reference", "none"] = "template",
    logp_anisotropy_weight: float | None = None,
    transform: Transform | Literal["logY", "linear"] | None = None,
) -> AnisotropyObjective:
    """Predict reflectivity at ``energy`` using fitted book-ended film geometry.

    Parameters
    ----------
    reference_objective
        Fitted objective (typically at 283.7 eV) supplying film geometry.
    energy
        Target photon energy in eV.
    template_objective
        DFT (or other) objective at ``energy`` for dispersive substrate layers and,
        when ``instrumentation='template'``, scales and angular offsets.
    ooc
        Full OOC table for the film anchor.
    data
        Reflectivity datasets keyed by energy string (``read_xrr`` output).
    num_slabs, mesh_constant
        Microslab mesh for the new stack.
    instrumentation
        ``'template'`` copies scales and offsets from ``template_objective.model``;
        ``'reference'`` copies from the fitted model; ``'none'`` keeps pyref defaults.
    logp_anisotropy_weight, transform
        Forwarded to :class:`~pyref.fitting.AnisotropyObjective`.

    Returns
    -------
    AnisotropyObjective
        Non-varying objective at ``energy`` with fitted film geometry.
    """
    ctx = extract_graded_fit_context(reference_objective)
    structure_e = graded_structure_at_energy(
        energy,
        ooc,
        template_objective,
        num_slabs=num_slabs,
        mesh_constant=mesh_constant,
    )
    film_e = bookended_film_from_structure(structure_e)
    copy_bookended_film_parameters(ctx.film, film_e, vary=False, copy_constraints=False)
    film_e.clear_ooc_cache()
    film_e.cache_ooc_at(float(energy))

    if instrumentation == "template":
        instr_src: ReflectModel | None = template_objective.model
    elif instrumentation == "reference":
        instr_src = ctx.model
    else:
        instr_src = None

    model_e = ReflectModel(structure_e, energy=float(energy), pol=ctx.model.pol)
    if instr_src is not None:
        copy_reflect_model_instrumentation(instr_src, model_e, vary=False)

    data_key = resolve_data_key(data, energy)
    kwargs = _anisotropy_kwargs_from_reference(
        reference_objective,
        logp_anisotropy_weight=logp_anisotropy_weight,
        transform=transform,
    )
    return AnisotropyObjective(model_e, data[data_key], **kwargs)


def link_bookended_film_to_reference(
    film: EnergyBookendedOrientationDensityProfile,
    reference_film: EnergyBookendedOrientationDensityProfile,
    *,
    param_names: tuple[str, ...] = BOOKENDED_FILM_PARAM_NAMES,
) -> None:
    """Constrain ``film`` parameters to ``reference_film`` for multi-energy global fits.

    Parameters
    ----------
    film
        Film at a non-reference energy.
    reference_film
        Template film (e.g. at 283.7 eV) whose parameters are shared.
    param_names
        Book-ended parameters to link.
    """
    for name in param_names:
        p_ref = getattr(reference_film, name)
        p_new = getattr(film, name)
        safely_setp_param(p_new, constraint=p_ref, vary=None)


def graded_objectives_for_energies(  # noqa: PLR0913
    reference_objective: AnisotropyObjective,
    energies: list[float],
    *,
    template_at_energy,
    ooc: pd.DataFrame | OocAnchor,
    data: dict[str, Any],
    num_slabs: int = 24,
    link_geometry_to_reference: bool = True,
    **kwargs: Any,
) -> dict[float, AnisotropyObjective]:
    """Build one non-varying objective per energy for multi-panel figures.

    Parameters
    ----------
    reference_objective
        Fitted graded objective supplying shared film geometry.
    energies
        Photon energies in eV to evaluate.
    template_at_energy
        Callable ``(energy_eV) -> objective`` returning a DFT template at each energy
        (e.g. ``lambda e: en_access(e, dft_fit)``).
    ooc
        Full OOC table.
    data
        Reflectivity datasets from ``read_xrr``.
    num_slabs
        Microslab count for each rebuilt stack.
    link_geometry_to_reference
        When ``True``, after the first energy, constrain film parameters to the
        reference film instead of copying values (for strict global-fit linking).
    **kwargs
        Forwarded to :func:`graded_objective_at_energy`.

    Returns
    -------
    dict[float, AnisotropyObjective]
        Objectives keyed by float energy in eV.
    """
    ctx = extract_graded_fit_context(reference_objective)
    out: dict[float, AnisotropyObjective] = {}
    ref_film = ctx.film
    for energy in energies:
        energy_f = float(energy)
        template = template_at_energy(energy_f)
        obj = graded_objective_at_energy(
            reference_objective,
            energy_f,
            template_objective=template,
            ooc=ooc,
            data=data,
            num_slabs=num_slabs,
            **kwargs,
        )
        if link_geometry_to_reference and energy_f != ctx.energy:
            link_bookended_film_to_reference(
                bookended_film_from_structure(obj.model.structure),
                ref_film,
            )
        out[energy_f] = obj
    return out


__all__ = [
    "BOOKENDED_FILM_PARAM_NAMES",
    "REFLECT_MODEL_INSTRUMENTATION",
    "GradedFitContext",
    "bookended_film_from_structure",
    "clone_bookended_film",
    "copy_bookended_film_parameters",
    "copy_reflect_model_instrumentation",
    "extract_graded_fit_context",
    "graded_objective_at_energy",
    "graded_objectives_for_energies",
    "graded_structure_at_energy",
    "link_bookended_film_to_reference",
    "reflect_model_for_structure",
    "resolve_data_key",
    "structure_with_upsampled_film",
    "upsample_graded_objective",
]

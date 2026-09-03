"""Convert pickled pyref global fits to refloxide batched objectives and compare kernels.

Wraps :mod:`refloxide.pxr.plugin.dft_fit` for refl-analysis notebooks. Evaluates
:class:`~refloxide.pxr.plugin.batched_global.BatchedGlobalObjective` log-likelihood
with selectable Rust or pure-Python refloxide uniaxial kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from utils.pyref_patch import reflectivity as refloxide_reflectivity
from refloxide.pxr.layout import reflectivity_for_pol
from refloxide.pxr.plugin.batched_global import (
    AnisotropyBatchTerm,
    BatchedGlobalObjective,
    ReflectivityBatchTerm,
    _gaussian_logl,
    _materialize_structure,
    _q_grid_for_pol,
)
from refloxide.pxr.energy.ooc import OocAnchor
from refloxide.pxr.plugin.dft_fit import (
    DIAGNOSTIC_ENERGY_EV,
    apply_dft_diagnostic_structure_constraints,
    batched_objective_from_fit_bundle,
    build_dft_dispersive_model,
    tighten_theta_offset_bounds_from_terms,
)
from refloxide.pxr.plugin.dispersive_model import DispersiveReflectModel, resolve_instrument, select
from refloxide.pxr.plugin.fitters import LogpExtra
from refnx.analysis import is_parameter

if TYPE_CHECKING:
    import pandas as pd
    from refnx.analysis import GlobalObjective

KernelKind = Literal["pyref_stock", "refloxide_python", "refloxide_rust"]


@dataclass(frozen=True, slots=True)
class KernelComparisonResult:
    """Log-likelihood and reflectivity parity for three evaluation paths."""

    pyref_stock_logl: float
    refloxide_python_logl: float
    refloxide_rust_logl: float
    python_rust_logl_delta: float
    diagnostic_energy_ev: float
    diagnostic_pol: str
    pyref_stock_reflectivity: np.ndarray
    refloxide_python_reflectivity: np.ndarray
    refloxide_rust_reflectivity: np.ndarray
    max_abs_reflectivity_delta_python_rust: float


def convert_dft_fit_bundle(
    bundle: GlobalObjective,
    ooc: OocAnchor | pd.DataFrame,
    *,
    attach_logp_extra: bool = True,
    tighten_theta_bounds: bool = True,
) -> tuple[DispersiveReflectModel, BatchedGlobalObjective]:
    """Migrate a pickled DFT ``GlobalObjective`` to refloxide batched form.

    Parameters
    ----------
    bundle
        Pickled multi-energy fit loaded via :func:`~utils.read_fit`.
    ooc
        DFT optical-constant table for tabulated tensor layers.
    attach_logp_extra
        When ``True``, attach interfacial-thickness
        :class:`~refloxide.pxr.plugin.fitters.LogpExtra` to the batched objective.
    tighten_theta_bounds
        When ``True``, narrow ``theta_offset`` lower bounds from data grazing
        angles before Rust evaluation.

    Returns
    -------
    tuple
        ``(model, objective)`` ready for :class:`~refloxide.pxr.plugin.batched_global.BatchedFitter`
        or kernel comparison helpers in this module.
    """
    model = build_dft_dispersive_model(bundle, ooc, apply_constraints=True)
    objective = batched_objective_from_fit_bundle(
        model,
        bundle,
        tighten_theta_bounds=tighten_theta_bounds,
    )
    if attach_logp_extra:
        objective.logp_extra = LogpExtra(objective)
    return model, objective


_TENSOR_LAYER_LABELS: tuple[str, ...] = ("Surface", "ZnPc", "Contamination")


def _objective_at_energy(bundle: GlobalObjective, energy_ev: float) -> Any:
    """Return the first bundle objective whose model energy matches ``energy_ev``."""
    for obj in bundle.objectives:
        model_energy = getattr(obj.model, "energy", None)
        if model_energy is not None and abs(float(model_energy) - energy_ev) < 1e-3:
            return obj
    msg = f"No objective at {energy_ev} eV in fit bundle"
    raise ValueError(msg)


def apply_free_substrate_layers(
    model: DispersiveReflectModel,
    free_bundle: GlobalObjective,
    *,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
) -> DispersiveReflectModel:
    """Copy oxide and substrate geometry from a free-tensor diagnostic fit.

    Parameters
    ----------
    model
        Dispersive model whose ``Oxide`` and ``Substrate`` slabs are updated in
        place.
    free_bundle
        Pickled free-tensor global fit supplying substrate layer values.
    diagnostic_energy
        Photon energy (eV) used to select the reference objective.

    Returns
    -------
    DispersiveReflectModel
        The same ``model`` instance for chaining.
    """
    free_diag = _objective_at_energy(free_bundle, diagnostic_energy)
    structure = free_diag.model.structure  # type: ignore[union-attr]
    oxide_src = structure.components[4]
    substrate_src = structure.components[5]
    oxide = select(model, "Oxide")
    substrate = select(model, "Substrate")
    oxide.thick.setp(value=float(oxide_src.thick.value))
    oxide.rough.setp(value=float(oxide_src.rough.value))
    oxide.sld.density.setp(value=float(oxide_src.sld.density.value))
    substrate.thick.setp(value=float(substrate_src.thick.value))
    substrate.rough.setp(value=float(substrate_src.rough.value))
    substrate.sld.density.setp(value=float(substrate_src.sld.density.value))
    return model


def fix_dispersive_instrumentation(model: DispersiveReflectModel) -> DispersiveReflectModel:
    """Hold scale and theta-offset instrument parameters fixed on every channel.

    Parameters
    ----------
    model
        Dispersive reflectivity model whose per-energy instrument blocks are
        updated in place.

    Returns
    -------
    DispersiveReflectModel
        The same ``model`` instance for chaining.
    """
    for energy in model.energies:
        instrument = model.instrument_at(energy)
        instrument.theta_offset_s.setp(vary=False)
        instrument.theta_offset_p.setp(vary=False)
        instrument.scale_s.setp(vary=False)
        instrument.scale_p.setp(vary=False)
    return model


def configure_dft_diagnostic_mcmc_params(
    model: DispersiveReflectModel,
    dft_bundle: GlobalObjective,
    *,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
    vary_substrate_density: bool = True,
) -> DispersiveReflectModel:
    """Expose diagnostic-energy tensor and substrate parameters for MCMC.

    Mirrors the 283.7 eV varying-parameter block in ``fit_dft_fix.ipynb`` while
    keeping vacuum, oxide, and energy offset fixed.

    Parameters
    ----------
    model
        Dispersive model built from the DFT bundle.
    dft_bundle
        Source fit supplying tensor bounds and nominal values.
    diagnostic_energy
        Photon energy (eV) whose tensor slabs may vary during sampling.
    vary_substrate_density
        When ``True``, allow substrate mass density to vary at the diagnostic
        energy.

    Returns
    -------
    DispersiveReflectModel
        The same ``model`` instance for chaining.
    """
    diag_obj = _objective_at_energy(dft_bundle, diagnostic_energy)
    old = diag_obj.model.structure  # type: ignore[union-attr]
    model.energy_offset.at(diagnostic_energy).setp(vary=False)
    select(model, "Vacuum").thick.setp(vary=False)
    select(model, "Vacuum").rough.setp(vary=False)
    select(model, "Oxide").thick.setp(vary=False)
    select(model, "Oxide").rough.setp(vary=False)
    select(model, "Oxide").sld.density.setp(vary=False)
    select(model, "Substrate").thick.setp(vary=False)
    select(model, "Substrate").rough.setp(vary=False)
    if vary_substrate_density:
        substrate = select(model, "Substrate")
        old_sub = old.components[5]
        substrate.sld.density.setp(
            value=old_sub.sld.density.value,
            bounds=(old_sub.sld.density.bounds.lb, old_sub.sld.density.bounds.ub),
            vary=True,
        )
    else:
        select(model, "Substrate").sld.density.setp(vary=False)
    for label in _TENSOR_LAYER_LABELS:
        new_slab = select(model, label)
        old_slab = next(c for c in old.components if str(c.name).startswith(label))
        for attr in ("thick", "rough"):
            src = getattr(old_slab, attr)
            getattr(new_slab, attr).setp(
                value=src.value,
                bounds=(src.bounds.lb, src.bounds.ub),
                vary=True,
                constraint=None,
            )
        for attr in ("density", "rotation"):
            src = getattr(old_slab.sld, attr)
            getattr(new_slab.sld, attr).setp(
                value=src.value,
                bounds=(src.bounds.lb, src.bounds.ub),
                vary=True,
                constraint=None,
            )
    return model


def build_dft_diagnostic_mcmc_bundle(
    dft_bundle: GlobalObjective,
    ooc: OocAnchor | pd.DataFrame,
    *,
    free_bundle: GlobalObjective | None = None,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
    vary_substrate_density: bool = True,
) -> tuple[DispersiveReflectModel, BatchedGlobalObjective]:
    """Build a low-dimensional refloxide objective for diagnostic-energy MCMC.

    Evaluates reflectivity through the Rust uniaxial kernel on a single shared
    :class:`~refloxide.pxr.energy.structure.DispersiveStructure`, with only the
    diagnostic-energy s/p datasets and anisotropy term included. This avoids the
    duplicated per-energy parameters and repeated full-stack Python reflectivity
    evaluations that make stock ``GlobalObjective`` emcee sampling slow.

    Parameters
    ----------
    dft_bundle
        Pickled DFT global fit (multi-energy).
    ooc
        DFT optical-constant table for tabulated tensor layers.
    free_bundle
        Optional free-tensor fit whose oxide/substrate values override the DFT
        diagnostic stack, matching ``fit_dft_fix.ipynb``.
    diagnostic_energy
        Photon energy (eV) retained in the batched likelihood.
    vary_substrate_density
        When ``True``, include substrate density among MCMC parameters.

    Returns
    -------
    tuple
        ``(model, objective)`` for :class:`~refloxide.pxr.plugin.batched_global.BatchedFitter`.
    """
    anchor = ooc if isinstance(ooc, OocAnchor) else OocAnchor.from_dataframe(ooc)
    diag_obj = _objective_at_energy(dft_bundle, diagnostic_energy)
    model = build_dft_dispersive_model(
        dft_bundle,
        anchor,
        diagnostic_energy=diagnostic_energy,
        apply_constraints=False,
    )
    model.pol = "sp"  # type: ignore[assignment]
    apply_dft_diagnostic_structure_constraints(
        model,
        diag_obj,
        diagnostic_energy=diagnostic_energy,
    )
    if free_bundle is not None:
        apply_free_substrate_layers(
            model,
            free_bundle,
            diagnostic_energy=diagnostic_energy,
        )
    configure_dft_diagnostic_mcmc_params(
        model,
        dft_bundle,
        diagnostic_energy=diagnostic_energy,
        vary_substrate_density=vary_substrate_density,
    )
    fix_dispersive_instrumentation(model)

    class _DiagnosticBundle:
        objectives = [diag_obj]

    objective = batched_objective_from_fit_bundle(
        model,
        _DiagnosticBundle(),  # type: ignore[arg-type]
        tighten_theta_bounds=False,
    )
    tighten_theta_offset_bounds_from_terms(model, objective.terms)
    objective.logp_extra = LogpExtra(objective)
    return model, objective


def build_dft_diagnostic_pyref_mcmc_objective(
    dft_bundle: GlobalObjective,
    ooc: OocAnchor | pd.DataFrame,
    *,
    free_bundle: GlobalObjective | None = None,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
    vary_substrate_density: bool = True,
) -> Any:
    """Build a single-energy pyref objective for fast diagnostic MCMC.

    Uses patched :mod:`pyref.fitting` reflectivity (Rust via refloxide) on three
    ``UniTensorSLD`` slabs so density and rotation remain visible to refnx.
    Call :func:`~utils.models.configure_refloxide_fitting` before constructing
    the fitter.

    Parameters
    ----------
    dft_bundle
        Pickled DFT global fit.
    ooc
        DFT optical-constant table.
    free_bundle
        Optional free-tensor fit supplying oxide/substrate geometry.
    diagnostic_energy
        Photon energy (eV) for the MCMC likelihood.
    vary_substrate_density
        When ``True``, include substrate mass density among MCMC parameters.

    Returns
    -------
    AnisotropyObjective
        Single-energy objective with instrumentation and substrate/oxide fixed
        like ``fit_dft_fix.ipynb`` at the diagnostic energy.
    """
    import pyref.fitting as fit

    ooc_table = ooc.to_dataframe() if isinstance(ooc, OocAnchor) else ooc
    diag = _objective_at_energy(dft_bundle, diagnostic_energy)
    free_diag = (
        _objective_at_energy(free_bundle, diagnostic_energy)
        if free_bundle is not None
        else diag
    )
    struct = diag.model.structure  # type: ignore[union-attr]
    struct_free = free_diag.model.structure  # type: ignore[union-attr]

    def rebuild(old_component: Any) -> Any:
        return fit.UniTensorSLD(
            ooc=ooc_table,
            rotation=float(old_component.sld.rotation.value),
            density=float(old_component.sld.density.value),
            name=old_component.name,
        )(float(old_component.thick.value), float(old_component.rough.value))

    new_structure = (
        fit.MaterialSLD("", 0, name=struct.components[0].name)(0, 0)
        | rebuild(struct.components[1])
        | rebuild(struct.components[2])
        | rebuild(struct.components[3])
        | fit.MaterialSLD(
            "SiO2",
            float(struct_free.components[4].sld.density.value),
            name=struct.components[4].name,
        )(
            float(struct_free.components[4].thick.value),
            float(struct_free.components[4].rough.value),
        )
        | fit.MaterialSLD(
            "Si",
            float(struct_free.components[5].sld.density.value),
            name=struct.components[5].name,
        )(
            float(struct_free.components[5].thick.value),
            float(struct_free.components[5].rough.value),
        )
    )
    model = diag.model  # type: ignore[union-attr]
    new_model = fit.ReflectModel(
        new_structure,
        energy=model.energy,
        pol=model.pol,
        name=model.name,
        scale_s=model.scale_s,
        scale_p=model.scale_p,
        theta_offset_s=model.theta_offset_s,
        theta_offset_p=model.theta_offset_p,
        bkg=model.bkg,
    )
    new_model.theta_offset_p.setp(vary=False)
    new_model.theta_offset_s.setp(vary=False)
    new_model.scale_p.setp(vary=False)
    new_model.scale_s.setp(vary=False)
    new_model.energy_offset.setp(value=model.energy_offset.value, vary=False)
    new_model.structure.components[0].thick.setp(vary=False)
    new_model.structure.components[0].rough.setp(vary=False)
    new_model.structure.components[0].sld.density.setp(vary=False)
    new_model.structure.components[-1].rough.setp(vary=False)
    new_model.structure.components[-1].thick.setp(vary=False)
    new_model.structure.components[-2].thick.setp(vary=False)
    new_model.structure.components[-2].rough.setp(vary=False)
    new_model.structure.components[-2].sld.density.setp(vary=False)
    for index in (1, 2, 3):
        struct_new = new_model.structure.components[index]
        struct_old = struct.components[index]
        struct_new.sld.density.setp(
            value=struct_old.sld.density.value,
            bounds=(struct_old.sld.density.bounds.lb, struct_old.sld.density.bounds.ub),
            vary=True,
            constraint=None,
        )
        struct_new.sld.rotation.setp(
            value=struct_old.sld.rotation.value,
            bounds=(struct_old.sld.rotation.bounds.lb, struct_old.sld.rotation.bounds.ub),
            vary=True,
            constraint=None,
        )
        struct_new.thick.setp(
            value=struct_old.thick.value,
            bounds=(struct_old.thick.bounds.lb, struct_old.thick.bounds.ub),
            vary=True,
        )
        struct_new.rough.setp(
            value=struct_old.rough.value,
            bounds=(struct_old.rough.bounds.lb, struct_old.rough.bounds.ub),
            vary=True,
            constraint=None,
        )
    substrate = new_model.structure.components[-1]
    if vary_substrate_density:
        old_sub = struct.components[-1]
        substrate.sld.density.setp(
            value=old_sub.sld.density.value,
            bounds=(old_sub.sld.density.bounds.lb, old_sub.sld.density.bounds.ub),
            vary=True,
        )
    else:
        substrate.sld.density.setp(vary=False)
    objective = fit.AnisotropyObjective(
        new_model,
        diag.data,
        logp_anisotropy_weight=diag.logp_anisotropy_weight,
        transform=fit.Transform("logY"),
    )
    objective.logp_extra = LogpExtra(objective)
    return objective


def _evaluate_term_reflectivity(
    model: DispersiveReflectModel,
    term: ReflectivityBatchTerm,
    *,
    use_rust: bool,
    parallel: bool,
    snapshot: Any | None = None,
) -> np.ndarray:
    energy = float(term.energy)
    instrument = resolve_instrument(model, energy)  # type: ignore[arg-type]
    structure = model.structure  # type: ignore[union-attr]
    if snapshot is None:
        snapshot = _materialize_structure(
            structure,
            energy,
            structure_offset_ev=instrument.energy_offset_ev,
        )
    slabs = snapshot.layers
    tensor = snapshot.tensors
    qvals, qvals_1, qvals_2 = _q_grid_for_pol(
        term.x,
        term.pol,
        energy,
        theta_offset_s=instrument.theta_offset_s,
        theta_offset_p=instrument.theta_offset_p,
    )
    dq_raw = term.x_err if term.x_err is not None else instrument.dq
    dq = float(np.asarray(dq_raw).flat[0])
    result = refloxide_reflectivity(
        qvals + instrument.q_offset,
        slabs,
        tensor,
        energy,
        scale_s=instrument.scale_s,
        scale_p=instrument.scale_p,
        bkg=instrument.bkg,
        dq=dq,
        backend="uni",
        use_rust=use_rust,
        parallel=parallel,
    )
    if result is None:
        msg = "reflectivity returned None; check dq / backend"
        raise RuntimeError(msg)
    refl, _tran, _components = result
    return reflectivity_for_pol(
        term.pol,
        refl,
        qvals,
        qvals_1,
        qvals_2,
    )


def batched_logl_with_kernel(
    objective: BatchedGlobalObjective,
    *,
    use_rust: bool,
    parallel: bool = False,
) -> float:
    """Evaluate ``BatchedGlobalObjective.logl`` with a chosen refloxide kernel.

    Parameters
    ----------
    objective
        Batched objective built from a pickled global fit.
    use_rust
        When ``True``, call :func:`refloxide.rust.uniaxial_reflectivity`; when
        ``False``, use the pure-Python :mod:`refloxide.python.tmm` port.
    parallel
        Forwarded to the Rust kernel as ``parallel=``. Ignored when
        ``use_rust=False``.

    Returns
    -------
    float
        Total log-likelihood including priors and anisotropy terms.
    """
    objective.setp(None)
    model = objective.model  # type: ignore[assignment]
    structure = model.structure  # type: ignore[union-attr]
    snapshots: dict[tuple[float, float], Any] = {}
    batch_begin = getattr(structure, "begin_materialization_batch", None)
    batch_end = getattr(structure, "end_materialization_batch", None)
    if hasattr(structure, "materialize"):
        if batch_begin is not None:
            batch_begin()
        try:
            seen: set[tuple[float, float]] = set()
            for term in objective.terms:
                instrument = resolve_instrument(model, float(term.energy))  # type: ignore[arg-type]
                key = (float(term.energy), instrument.energy_offset_ev)
                if key in seen:
                    continue
                seen.add(key)
                snapshots[key] = _materialize_structure(
                    structure,
                    key[0],
                    structure_offset_ev=key[1],
                )
        finally:
            if batch_end is not None:
                batch_end()

    lnsigma = (
        float(objective.lnsigma.value) if is_parameter(objective.lnsigma) else None  # ty: ignore[unresolved-attribute]
    )
    logl = 0.0
    for idx, term in enumerate(objective.terms):
        instrument = resolve_instrument(model, float(term.energy))  # type: ignore[arg-type]
        snap_key = (float(term.energy), instrument.energy_offset_ev)
        model_y = _evaluate_term_reflectivity(
            model,
            term,
            use_rust=use_rust,
            parallel=parallel,
            snapshot=snapshots.get(snap_key),
        )
        y, y_err, model_y = objective._transform_term(term, model_y)  # noqa: SLF001
        logl += float(objective.lambdas[idx]) * _gaussian_logl(
            y,
            y_err,
            model_y,
            weighted=objective.weighted,
            lnsigma=lnsigma,
        )
    extra = model.logp()
    if objective.logp_extra is not None:
        extra += objective.logp_extra(model, objective.data)
    logl += extra
    for aterm in objective.anisotropy_terms:
        saved = model.energy  # type: ignore[union-attr]
        try:
            model.energy = aterm.energy  # type: ignore[union-attr]
            model_a = model.anisotropy(aterm.x)  # type: ignore[union-attr]
        finally:
            model.energy = saved  # type: ignore[union-attr]
        resid = model_a - aterm.y
        if aterm.y_err is not None:
            resid = resid / aterm.y_err
        logl += (
            float(aterm.lambda_)
            * float(aterm.weight)
            * float(-0.5 * np.sum(resid * resid))
        )
    return logl


def _diagnostic_term(
    objective: BatchedGlobalObjective,
    *,
    diagnostic_energy: float,
    pol: Literal["s", "p"],
) -> ReflectivityBatchTerm:
    for term in objective.terms:
        if abs(float(term.energy) - diagnostic_energy) < 1e-3 and term.pol == pol:
            return term
    msg = (
        f"No reflectivity term at {diagnostic_energy} eV pol={pol!r} "
        f"in batched objective"
    )
    raise ValueError(msg)


def _reflectivity_for_term(
    objective: BatchedGlobalObjective,
    term: ReflectivityBatchTerm,
    *,
    use_rust: bool,
    parallel: bool = False,
) -> np.ndarray:
    model = objective.model  # type: ignore[assignment]
    objective.setp(None)
    return _evaluate_term_reflectivity(
        model,
        term,
        use_rust=use_rust,
        parallel=parallel,
    )


def compare_kernel_paths(
    bundle: GlobalObjective,
    batched: BatchedGlobalObjective,
    *,
    diagnostic_energy: float = DIAGNOSTIC_ENERGY_EV,
    diagnostic_pol: Literal["s", "p"] = "s",
    parallel: bool = False,
    logl_rtol: float = 1e-9,
) -> KernelComparisonResult:
    """Compare stock pyref and refloxide Python/Rust log-likelihood on one fit.

    Parameters
    ----------
    bundle
        Original pickled global objective (stock pyref kernel).
    batched
        Refloxide batched objective converted from ``bundle``.
    diagnostic_energy
        Photon energy (eV) for reflectivity slice comparison.
    diagnostic_pol
        Laboratory polarization channel for the reflectivity slice.
    parallel
        Forwarded to the Rust kernel only.
    logl_rtol
        Relative tolerance printed when Python and Rust log-likelihood differ.

    Returns
    -------
    KernelComparisonResult
        Log-likelihood values and a diagnostic reflectivity vector per path.

    Raises
    ------
    RuntimeError
        When Python and Rust refloxide log-likelihood disagree beyond ``logl_rtol``.
    """
    pyref_logl = float(bundle.logl())
    py_logl = batched_logl_with_kernel(batched, use_rust=False, parallel=False)
    rust_logl = batched_logl_with_kernel(batched, use_rust=True, parallel=parallel)
    delta = abs(py_logl - rust_logl)
    if delta > logl_rtol * max(1.0, abs(py_logl)):
        msg = (
            f"refloxide Python/Rust logl mismatch: python={py_logl}, "
            f"rust={rust_logl}, delta={delta}"
        )
        raise RuntimeError(msg)

    term = _diagnostic_term(
        batched,
        diagnostic_energy=diagnostic_energy,
        pol=diagnostic_pol,
    )
    model = batched.model  # type: ignore[assignment]
    saved_pol = model.pol  # type: ignore[union-attr]
    try:
        model.pol = diagnostic_pol  # type: ignore[union-attr]
        diag_obj = next(
            obj
            for obj in bundle.objectives
            if abs(float(obj.model.energy) - diagnostic_energy) < 1e-3  # type: ignore[union-attr]
        )
        pyref_refl = np.asarray(diag_obj.model(diag_obj.data.s.x if diagnostic_pol == "s" else diag_obj.data.p.x))  # type: ignore[union-attr]
    finally:
        model.pol = saved_pol  # type: ignore[union-attr]

    refl_py = _reflectivity_for_term(batched, term, use_rust=False)
    refl_rust = _reflectivity_for_term(batched, term, use_rust=True, parallel=parallel)
    max_delta = float(np.max(np.abs(refl_py - refl_rust)))

    return KernelComparisonResult(
        pyref_stock_logl=pyref_logl,
        refloxide_python_logl=py_logl,
        refloxide_rust_logl=rust_logl,
        python_rust_logl_delta=delta,
        diagnostic_energy_ev=diagnostic_energy,
        diagnostic_pol=diagnostic_pol,
        pyref_stock_reflectivity=np.asarray(pyref_refl, dtype=np.float64),
        refloxide_python_reflectivity=refl_py,
        refloxide_rust_reflectivity=refl_rust,
        max_abs_reflectivity_delta_python_rust=max_delta,
    )


__all__ = [
    "DIAGNOSTIC_ENERGY_EV",
    "KernelComparisonResult",
    "KernelKind",
    "apply_free_substrate_layers",
    "batched_logl_with_kernel",
    "build_dft_diagnostic_mcmc_bundle",
    "build_dft_diagnostic_pyref_mcmc_objective",
    "compare_kernel_paths",
    "configure_dft_diagnostic_mcmc_params",
    "convert_dft_fit_bundle",
    "fix_dispersive_instrumentation",
]

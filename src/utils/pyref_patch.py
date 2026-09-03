"""Drop-in uniaxial reflectivity hooks for ``pyref.fitting`` workflows.

Vendored from ``refloxide.integrations.pyref``, which upstream refloxide
dropped (see refloxide commit ``6f33120``, "drop pyref integration for
python.model and tmm"). refloxide's own ``ReflectModel`` no longer needs
patching, but refl-analysis's existing pickled fits
(:class:`pyref.fitting.AnisotropyObjective` / ``GlobalObjective`` /
``ReflectModel`` instances) still evaluate through stock ``pyref.fitting``,
so this module keeps routing their reflectivity through refloxide's Rust
kernel instead of pyref's bundled one.

Routes reflectivity through refloxide's native Jones layout (``R_ss`` at
``[:,0,0]``, ``R_pp`` at ``[:,1,1]``) and patches
:class:`pyref.fitting.model.ReflectModel` so polarization extraction matches
stock pyref (``pol='s'`` from ``[:,1,1]``, ``pol='p'`` from ``[:,0,0]``),
preserving ``pol='sp'`` / ``pol='ps'`` segment ordering for combined datasets.

Scope is **uniaxial only** (``phi = 0``, ``backend = "uni"``). Do not use
this adapter for biaxial or general-incidence claims.
"""

from __future__ import annotations

import contextvars
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.interpolate import splev, splrep

if TYPE_CHECKING:
    from numpy.typing import NDArray

from refloxide.pxr.layout import apply_laboratory_scales, reflectivity_for_pol
from refloxide.python.tmm import uniaxial_reflectivity as _python_uniaxial

_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_DEFAULT_PROBE_ENERGY_EV = 250.0
_eval_energy: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "refloxide_pyref_eval_energy",
    default=None,
)

SldClassesMode = Literal["legacy", "energy", "both"]


@dataclass(frozen=True, slots=True)
class PyrefPatchReport:
    """Snapshot of which refloxide hooks are active on ``pyref.fitting``."""

    kernels: bool
    reflectivity: bool
    model_layout: bool
    fused_path: bool
    structure_ior: bool
    structure_materialization: bool
    sld_classes: SldClassesMode
    sld_exports: tuple[str, ...]


def _swap_sp_block(matrix: NDArray[Any]) -> NDArray[Any]:
    """Legacy alias; prefer :func:`refloxide.pxr.layout.reflectmodel_layout`."""
    from refloxide.pxr.layout import reflectmodel_layout

    return reflectmodel_layout(matrix)


def _solve_refloxide(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    *,
    use_rust: bool,
    parallel: bool,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    if use_rust:
        from refloxide.rust import uniaxial_reflectivity as rust_uniaxial

        refl, tran = rust_uniaxial(
            q,
            layers,
            tensor,
            float(energy),
            parallel=parallel,
        )
        refl_arr = np.asarray(refl, dtype=np.float64)
        tran_arr = np.asarray(tran, dtype=np.complex128)
    else:
        refl_arr, tran_arr, *_ = _python_uniaxial(
            q,
            layers,
            tensor,
            float(energy),
            phi=0.0,
        )
        refl_arr = np.asarray(refl_arr, dtype=np.float64)
        tran_arr = np.asarray(tran_arr, dtype=np.complex128)

    return refl_arr, tran_arr


def uniaxial_reflectivity(
    q: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    *,
    use_rust: bool = True,
    parallel: bool = False,
    return_components: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], *tuple[Any, ...]]:
    """Evaluate uniaxial reflectivity using refloxide's laboratory Jones layout.

    Accepts the same positional arguments as
    ``pyref.fitting.uniaxial.uniaxial_reflectivity`` (``q``, ``layers``,
    ``tensor``, ``energy`` in eV). Returns ``refl`` with ``[:,0,0] = R_ss`` and
    ``[:,1,1] = R_pp``. Pair with :func:`patch_pyref` so
    :class:`pyref.fitting.model.ReflectModel` reads those channels correctly.

    Parameters
    ----------
    q
        Scattering wavevectors in inverse angstroms, shape ``(n_q,)``.
    layers
        Slab table ``(n_layers, 4)`` with rows ``[thickness, sld_real,
        sld_imag, roughness]`` in refnx/pyref units (SLD scaled by
        ``1e-6`` Angstrom**-2).
    tensor
        Per-layer dispersion tensor ``(n_layers, 3, 3)`` with diagonals
        carrying :math:`\\delta + i\\beta` per principal axis before the
        ``epsilon = conj(I - 2 * tensor)`` conversion inside the kernel.
    energy
        Photon energy in eV.
    use_rust
        When ``True`` (default), call :func:`refloxide.rust.uniaxial_reflectivity`.
        When ``False``, use the pure-Python port in
        :mod:`refloxide.python.tmm`.
    parallel
        Passed to the Rust kernel as ``parallel=``. Default ``False`` so
        refnx/emcee worker pools are not oversubscribed. Ignored when
        ``use_rust=False``.
    return_components
        When ``True`` and ``use_rust=False``, append the diagnostic arrays
        returned by the Python port (``kx``, ``ky``, ``kz``, polarization
        bases, transfer matrices). When ``True`` and ``use_rust=True``, the
        trailing tuple is empty because the extension returns only ``refl``
        and ``tran``.

    Returns
    -------
    refl
        Power reflectance ``(n_q, 2, 2)`` with ``[:,0,0] = R_ss``,
        ``[:,1,1] = R_pp``.
    tran
        Complex transmission amplitudes in the same Jones layout as ``refl``.
    components
        Optional trailing diagnostics when ``return_components=True``.

    Raises
    ------
    ValueError
        Propagated from the Rust kernel on malformed inputs.
    RuntimeError
        Propagated from the Rust kernel when the dynamic matrix is singular.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    layers_arr = np.asarray(layers, dtype=np.float64)
    tensor_arr = np.asarray(tensor, dtype=np.complex128)
    refl, tran = _solve_refloxide(
        q_arr,
        layers_arr,
        tensor_arr,
        float(energy),
        use_rust=use_rust,
        parallel=parallel,
    )
    if return_components and not use_rust:
        *_, components = _python_uniaxial(
            q_arr,
            layers_arr,
            tensor_arr,
            float(energy),
            phi=0.0,
        )
        return refl, tran, *components
    return refl, tran


def reflectivity(
    q: np.ndarray,
    slabs: np.ndarray,
    tensor: np.ndarray,
    energy: float = 250.0,
    phi: float = 0.0,
    scale_s: float = 1.0,
    scale_p: float = 1.0,
    bkg: float = 0.0,
    dq: float = 0.0,
    backend: Literal["uni"] = "uni",
    *,
    use_rust: bool = True,
    parallel: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[Any]] | None:
    """Mirror ``pyref.fitting.model.reflectivity`` for uniaxial refloxide kernels.

    Applies laboratory-index scaling, background, and optional constant ``dQ/Q``
    smearing using the same recipe as pyref, but evaluates the stratified stack
    with :func:`uniaxial_reflectivity`.

    Parameters
    ----------
    q
        Scattering wavevectors in inverse angstroms.
    slabs
        Slab table with shape ``(2 + N, 4)``; see
        ``pyref.fitting.model.reflectivity`` for column semantics.
    tensor
        Per-layer ``(2 + N, 3, 3)`` dispersion tensor.
    energy
        Photon energy in eV.
    phi
        Accepted for API compatibility with pyref; ignored (uniaxial path
        fixes ``phi = 0``).
    scale_s, scale_p
        Independent scale factors applied to ``refl[:,0,0]`` (``R_ss``) and
        ``refl[:,1,1]`` (``R_pp``) before adding ``bkg``.
    bkg
        Constant background added to every reflectivity element.
    dq
        Constant ``dQ/Q`` resolution smearing in percent. ``0`` disables
        smearing.
    backend
        Must be ``"uni"``; other backends raise ``ValueError``.
    use_rust
        Forwarded to :func:`uniaxial_reflectivity`.
    parallel
        Forwarded to :func:`uniaxial_reflectivity`.

    Returns
    -------
    refl
        Smear-adjusted, scaled reflectivity matrix plus ``bkg``.
    tran
        Transmission amplitudes from the final kernel call inside the smear
        branch (unchanged by scaling).
    components
        Empty list when smearing is disabled; otherwise the trailing items
        from the last internal kernel call.

    Raises
    ------
    ValueError
        When ``backend`` is not ``"uni"``.
    """
    del phi
    if backend != "uni":
        msg = (
            "utils.pyref_patch.reflectivity supports backend='uni' "
            f"only; got {backend!r}"
        )
        raise ValueError(msg)

    if not isinstance(dq, numbers.Real):
        return None

    if float(dq) == 0:
        refl, tran, *components = uniaxial_reflectivity(
            q,
            slabs,
            tensor,
            energy,
            use_rust=use_rust,
            parallel=parallel,
        )
        refl = np.asarray(refl, dtype=np.float64)
        apply_laboratory_scales(refl, scale_s, scale_p)
        return refl + bkg, tran, list(components)

    smear_refl, smear_tran, *components = _smeared_reflectivity(
        q,
        slabs,
        tensor,
        energy,
        float(dq),
        use_rust=use_rust,
        parallel=parallel,
    )
    apply_laboratory_scales(smear_refl, scale_s, scale_p)
    return smear_refl + bkg, smear_tran, list(components)


def _smeared_reflectivity(
    q: np.ndarray,
    slabs: np.ndarray,
    tensor: np.ndarray,
    energy: float,
    resolution: float,
    *,
    use_rust: bool,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    if resolution < 0.5:
        refl, tran, *components = uniaxial_reflectivity(
            q,
            slabs,
            tensor,
            energy,
            use_rust=use_rust,
            parallel=parallel,
        )
        return np.asarray(refl), tran, list(components)

    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1.0 / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2 / s / s)

    lowq = float(np.min(q))
    highq = float(np.max(q))
    if lowq <= 0:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * resolution / _FWHM
    finish = np.log10(highq * (1 + 6 * resolution / _FWHM))
    interpnum = int(
        np.round(
            np.abs(1 * (np.abs(start - finish)))
            / (1.7 * resolution / _FWHM / gaussgpoint)
        )
    )
    xtemp = np.linspace(start, finish, interpnum)
    xlin = np.power(10.0, xtemp)

    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)
    refl, tran, *components = uniaxial_reflectivity(
        xlin,
        slabs,
        tensor,
        energy,
        use_rust=use_rust,
        parallel=parallel,
    )
    step = gauss_x[1] - gauss_x[0]
    smeared_ss = np.convolve(refl[:, 0, 0], gauss_y, mode="same") * step
    smeared_pp = np.convolve(refl[:, 1, 1], gauss_y, mode="same") * step
    smeared_sp = np.convolve(refl[:, 0, 1], gauss_y, mode="same") * step
    smeared_ps = np.convolve(refl[:, 1, 0], gauss_y, mode="same") * step

    smeared_output_ss = splev(q, splrep(xlin, smeared_ss))
    smeared_output_sp = splev(q, splrep(xlin, smeared_sp))
    smeared_output_ps = splev(q, splrep(xlin, smeared_ps))
    smeared_output_pp = splev(q, splrep(xlin, smeared_pp))

    smeared_output = np.rollaxis(
        np.array(
            [
                [smeared_output_ss, smeared_output_sp],
                [smeared_output_ps, smeared_output_pp],
            ]
        ),
        2,
        0,
    )
    return smeared_output, tran, list(components)


def _structure_energy_offset_ev(structure: Any | None) -> float:
    """Read the structure-level energy offset (eV) when present."""
    if structure is None:
        return 0.0
    for attr in ("structure_energy_offset", "energy_offset"):
        candidate = getattr(structure, attr, None)
        if candidate is not None and hasattr(candidate, "value"):
            return float(candidate.value or 0.0)
    return 0.0


def _resolve_reflectmodel_energy(
    model: Any,
    *,
    default: float = _DEFAULT_PROBE_ENERGY_EV,
) -> float:
    """Resolve a finite photon energy (eV) for ``ReflectModel`` q/theta adjustments.

    Stock pyref allows ``ReflectModel.energy is None`` at construction; this helper
    falls back to scatterer ``get_energy()``/``.energy`` on the attached structure,
    then ``default``.
    """
    energy = getattr(model, "energy", None)
    if energy is not None:
        return float(energy)
    structure = getattr(model, "structure", None)
    if structure is not None:
        for component in getattr(structure, "components", []):
            sld = getattr(component, "sld", None)
            if sld is None:
                continue
            get_energy = getattr(sld, "get_energy", None)
            if callable(get_energy):
                return float(get_energy())
            scatterer_energy = getattr(sld, "energy", None)
            if scatterer_energy is not None:
                return float(scatterer_energy)
    return float(default)


def _slab_energy_probe(
    _slab: Any,
    energy: float | None,
    *,
    structure: Any | None = None,
) -> Any:
    """Build a :class:`~refloxide.pxr.energy.probe.Probe` for one slab."""
    from refloxide.pxr.energy.probe import Probe

    ctx_energy = _eval_energy.get()
    if energy is not None:
        base = float(energy)
    elif ctx_energy is not None:
        base = float(ctx_energy)
    else:
        base = _DEFAULT_PROBE_ENERGY_EV
    return Probe(
        base_energy_ev=base,
        structure_offset_ev=_structure_energy_offset_ev(structure),
    )


def _tensor_to_slab_row(
    thick: float,
    rough: float,
    tensor: NDArray[Any],
) -> NDArray[np.float64]:
    """Convert a diagonal tensor to refnx ``[d, delta, beta, sigma]`` layout."""
    n_avg = (tensor[0, 0] + tensor[1, 1] + tensor[2, 2]) / 3.0
    delta = float((1.0 - n_avg).real)
    beta = float((1.0 - n_avg).imag)
    return np.array([thick, delta, beta, rough], dtype=np.float64)


def _refloxide_sld_exports() -> dict[str, Any]:
    """Scatterer symbols registered on ``pyref.fitting.structure``."""
    from refloxide.pxr.energy import (
        BookendedOrientationProfile,
        DeferredScatterer,
        DispersiveMaterialSLD,
        DispersiveStructure,
        EnergyBookendedOrientationDensityProfile,
        EnergyDependentMaterialSLD,
        EnergyDependentScatterer,
        EnergyDependentStructure,
        EnergyDependentUniTensorSLD,
        EnergyProbe,
        OocUniTensorScatterer,
        Probe,
        RefloxideScatterer,
        TabulatedUniTensorSLD,
        upgrade_scatterer,
        upgrade_structure,
    )

    return {
        "BookendedOrientationProfile": BookendedOrientationProfile,
        "DeferredScatterer": DeferredScatterer,
        "DispersiveMaterialSLD": DispersiveMaterialSLD,
        "DispersiveStructure": DispersiveStructure,
        "EnergyBookendedOrientationDensityProfile": (
            EnergyBookendedOrientationDensityProfile
        ),
        "EnergyDependentMaterialSLD": EnergyDependentMaterialSLD,
        "EnergyDependentScatterer": EnergyDependentScatterer,
        "EnergyDependentStructure": EnergyDependentStructure,
        "EnergyDependentUniTensorSLD": EnergyDependentUniTensorSLD,
        "EnergyProbe": EnergyProbe,
        "OocUniTensorScatterer": OocUniTensorScatterer,
        "Probe": Probe,
        "RefloxideScatterer": RefloxideScatterer,
        "TabulatedUniTensorSLD": TabulatedUniTensorSLD,
        "upgrade_scatterer": upgrade_scatterer,
        "upgrade_structure": upgrade_structure,
    }


def _reflectmodel_q_grid(
    model: Any,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror pyref ``ReflectModel`` q/theta adjustments before the kernel."""
    wavelength = 12398.42 / _resolve_reflectmodel_energy(model)
    if model.pol in ("sp", "ps"):
        concat_loc = int(np.argmax(np.abs(np.diff(x))))
        qvals_1 = x[: concat_loc + 1]
        qvals_2 = x[concat_loc + 1 :]
        num_q = max(len(x), concat_loc + 50)
        theta_s = np.arcsin(qvals_1 * wavelength / (4 * np.pi)) * 180 / np.pi
        theta_p = np.arcsin(qvals_2 * wavelength / (4 * np.pi)) * 180 / np.pi
        theta_s += float(model.theta_offset_s.value or 0.0)
        theta_p += float(model.theta_offset_p.value or 0.0)
        qvals_1 = (4 * np.pi / wavelength) * np.sin(theta_s * np.pi / 180)
        qvals_2 = (4 * np.pi / wavelength) * np.sin(theta_p * np.pi / 180)
        x_out = np.concatenate([qvals_1, qvals_2])
        qvals = np.linspace(float(np.min(x_out)), float(np.max(x_out)), num_q)
        return qvals, qvals_1, qvals_2
    if model.pol == "s":
        theta = np.arcsin(x * wavelength / (4 * np.pi)) * 180 / np.pi
        theta += float(model.theta_offset_s.value or 0.0)
        qvals = (4 * np.pi / wavelength) * np.sin(theta * np.pi / 180)
        return qvals, qvals, qvals
    if model.pol == "p":
        theta = np.arcsin(x * wavelength / (4 * np.pi)) * 180 / np.pi
        theta += float(model.theta_offset_p.value or 0.0)
        qvals = (4 * np.pi / wavelength) * np.sin(theta * np.pi / 180)
        return qvals, qvals, qvals
    return x, x, x


def _patch_reflect_model_fused(model_mod: Any, *, parallel: bool) -> None:
    """Route book-ended stacks through the fused Rust evaluator when possible."""
    reflect_model = model_mod.ReflectModel
    if getattr(reflect_model, "__refloxide_fused_patched__", False):
        return
    if not hasattr(reflect_model, "_model"):
        return
    original_model = reflect_model._model

    def _model(self, x, p=None, x_err=None):
        if getattr(self, "energy", None) is None and not hasattr(self, "structure"):
            return original_model(self, x, p=p, x_err=x_err)
        if p is not None and hasattr(self, "parameters"):
            self.parameters.pvals = np.array(p)
        if x_err is None:
            x_err = float(getattr(self, "dq", 0.0))
        x_arr = np.asarray(x, dtype=np.float64)
        qvals, qvals_1, qvals_2 = _reflectmodel_q_grid(self, x_arr)
        backend = getattr(self, "backend", "uni")
        model_energy = _resolve_reflectmodel_energy(self)
        if backend == "uni" and float(x_err) == 0.0 and hasattr(self, "structure"):
            from refloxide.pxr.energy.fused import evaluate_fused_bookended_reflectivity

            structure_offset = 0.0
            if hasattr(self, "energy_offset"):
                structure_offset = float(self.energy_offset.value or 0.0)
            q_offset = float(
                getattr(self, "q_offset", type("Q", (), {"value": 0.0})()).value or 0.0
            )
            q_kernel = qvals + q_offset
            fused = evaluate_fused_bookended_reflectivity(
                q_kernel,
                self.structure,
                model_energy,
                structure_energy_offset=structure_offset,
                parallel=parallel,
            )
            if fused is not None:
                refl, tran = fused
                scale_s = float(
                    getattr(self, "scale_s", type("S", (), {"value": 1.0})()).value
                    or 1.0
                )
                scale_p = float(
                    getattr(self, "scale_p", type("S", (), {"value": 1.0})()).value
                    or 1.0
                )
                bkg = float(
                    getattr(self, "bkg", type("S", (), {"value": 0.0})()).value or 0.0
                )
                apply_laboratory_scales(refl, scale_s, scale_p)
                refl = refl + bkg
                return qvals, qvals_1, qvals_2, refl, tran, []
        return original_model(self, x, p=p, x_err=x_err)

    reflect_model._model = _model
    reflect_model.__refloxide_fused_patched__ = True


def _patch_reflect_model_model(model_mod: Any) -> None:
    """Read laboratory Jones channels in :meth:`ReflectModel.model`."""
    reflect_model = model_mod.ReflectModel
    if getattr(reflect_model, "__refloxide_model_patched__", False):
        return

    def model(self, x, p=None, x_err=None):
        qvals, qvals_1, qvals_2, refl, _tran, _components = self._model(x, p, x_err)
        output = reflectivity_for_pol(self.pol, refl, qvals, qvals_1, qvals_2)
        return output

    reflect_model.model = model
    reflect_model.__refloxide_model_patched__ = True


def _is_bookended_profile(other: Any) -> bool:
    from refloxide.pxr.energy.bookended import BookendedOrientationProfile

    return isinstance(other, BookendedOrientationProfile)


def _patch_pyref_structure_ior() -> None:
    """Allow refloxide book-ended profiles in pyref ``Structure`` stacks."""
    import importlib
    from collections import UserList

    try:
        st_mod = importlib.import_module("pyref.fitting.structure")
    except ModuleNotFoundError:
        return
    if getattr(st_mod.Structure, "__refloxide_ior_patched__", False):
        return
    original_ior = st_mod.Structure.__ior__
    original_append = st_mod.Structure.append

    def append(self, item):
        if _is_bookended_profile(item):
            UserList.append(self, item)
            return
        return original_append(self, item)

    def __ior__(self, other):
        if _is_bookended_profile(other):
            UserList.append(self, other)
            return self
        return original_ior(self, other)

    st_mod.Structure.append = append  # ty: ignore[invalid-assignment]
    st_mod.Structure.__ior__ = __ior__  # ty: ignore[invalid-assignment]
    st_mod.Structure.__refloxide_ior_patched__ = True  # ty: ignore[unresolved-attribute]


def _patch_pyref_slab_materialization(st_mod: Any) -> None:
    """Materialize refloxide energy scatterers inside stock pyref ``Slab`` rows."""
    from refloxide.pxr.energy.scatterers import EnergyDependentScatterer

    if getattr(st_mod.Slab, "__refloxide_slab_patched__", False):
        return
    if not hasattr(st_mod, "Slab") or not hasattr(st_mod.Slab, "tensor"):
        return
    original_tensor = st_mod.Slab.tensor
    original_slabs = st_mod.Slab.slabs

    def tensor(self, energy=None):
        if isinstance(self.sld, EnergyDependentScatterer):
            probe = _slab_energy_probe(self, energy, structure=None)
            layer = self.sld.tensor_at(probe)
            return np.asarray([layer], dtype=np.complex128)
        return original_tensor(self, energy=energy)

    def slabs(self, structure=None):
        if isinstance(self.sld, EnergyDependentScatterer):
            probe = _slab_energy_probe(self, None, structure=structure)
            layer = self.sld.tensor_at(probe)
            thick = float(self.thick.value or 0.0)
            rough = float(self.rough.value or 0.0)
            row = _tensor_to_slab_row(thick, rough, layer)
            return np.asarray([row], dtype=np.float64)
        return original_slabs(self, structure=structure)

    st_mod.Slab.tensor = tensor  # ty: ignore[invalid-assignment]
    st_mod.Slab.slabs = slabs  # ty: ignore[invalid-assignment]
    st_mod.Slab.__refloxide_slab_patched__ = True  # ty: ignore[unresolved-attribute]


def _patch_pyref_sld_exports(
    st_mod: Any,
    mode: SldClassesMode,
) -> tuple[str, ...]:
    """Expose refloxide energy-deferred scatterers on ``pyref.fitting.structure``."""
    if mode == "legacy":
        return ()
    if getattr(st_mod, "__refloxide_sld_exports__", False):
        return tuple(getattr(st_mod, "__refloxide_sld_export_names__", ()))
    exports = _refloxide_sld_exports()
    for name, symbol in exports.items():
        setattr(st_mod, name, symbol)
    st_mod.__refloxide_sld_exports__ = True  # ty: ignore[unresolved-attribute]
    names = tuple(exports)
    st_mod.__refloxide_sld_export_names__ = names  # ty: ignore[unresolved-attribute]
    if mode == "energy":
        st_mod.MaterialSLD = exports["EnergyDependentMaterialSLD"]  # ty: ignore[invalid-assignment]
        st_mod.UniTensorSLD = exports["OocUniTensorScatterer"]  # ty: ignore[invalid-assignment]
    return names


def _patch_reflect_model_eval_energy(model_mod: Any) -> None:
    """Set evaluation energy context around every ``ReflectModel._model`` call."""
    reflect_model = model_mod.ReflectModel
    if getattr(reflect_model, "__refloxide_eval_energy_patched__", False):
        return
    if not hasattr(reflect_model, "_model"):
        return
    inner_model = reflect_model._model

    def _model(self, x, p=None, x_err=None):
        token = _eval_energy.set(_resolve_reflectmodel_energy(self))
        try:
            return inner_model(self, x, p=p, x_err=x_err)
        finally:
            _eval_energy.reset(token)

    reflect_model._model = _model
    reflect_model.__refloxide_eval_energy_patched__ = True


def patch_pyref(
    *,
    use_rust: bool = True,
    parallel: bool = False,
    patch_reflectivity: bool = True,
    sld_classes: SldClassesMode = "both",
) -> PyrefPatchReport:
    """Install refloxide uniaxial kernels into an imported ``pyref.fitting`` stack.

    Replaces ``pyref.fitting.uniaxial.uniaxial_reflectivity`` with a partial
    application of :func:`uniaxial_reflectivity`. Optionally replaces
    ``pyref.fitting.model.reflectivity`` so smearing and scaling stay on the
    refloxide code path. Patches :meth:`pyref.fitting.model.ReflectModel.model`
    to extract polarization channels with the same indexing as stock pyref.

    Parameters
    ----------
    use_rust
        When ``True`` (default), patched calls use
        :func:`refloxide.rust.uniaxial_reflectivity`.
    parallel
        Forwarded as ``parallel=`` to the Rust kernel. Default ``False`` for
        fitting loops that already parallelize across walkers or datasets.
    patch_reflectivity
        When ``True`` (default), also patch
        ``pyref.fitting.model.reflectivity``.
    sld_classes
        Controls refloxide scatterer exposure on ``pyref.fitting.structure``:

        - ``"legacy"`` — stock ``MaterialSLD`` / ``SLD`` / ``UniTensorSLD`` only.
        - ``"both"`` (default) — keep stock classes and add ``EnergyDependent*``
          symbols plus ``upgrade_scatterer`` / ``upgrade_structure``.
        - ``"energy"`` — same exports as ``"both"``, and alias ``MaterialSLD`` /
          ``UniTensorSLD`` to the energy-deferred implementations.

    Returns
    -------
    PyrefPatchReport
        Idempotent snapshot of which hooks were installed.

    Raises
    ------
    ImportError
        When ``pyref.fitting`` is not installed in the active environment.

    Notes
    -----
    Call this once at process startup before constructing
    :class:`pyref.fitting.model.ReflectModel` instances. The patch is
    process-global and affects every subsequent pyref reflectivity evaluation.
    """
    import importlib
    from functools import partial

    try:
        uni_mod = importlib.import_module("pyref.fitting.uniaxial")
        model_mod = importlib.import_module("pyref.fitting.model")
        st_mod = importlib.import_module("pyref.fitting.structure")
    except ModuleNotFoundError as exc:
        msg = (
            "pyref.fitting is not importable; install pyref or add it to "
            "PYTHONPATH before calling patch_pyref"
        )
        raise ImportError(msg) from exc

    patched_uni = partial(
        uniaxial_reflectivity,
        use_rust=use_rust,
        parallel=parallel,
    )
    uni_mod.uniaxial_reflectivity = patched_uni  # ty: ignore[unresolved-attribute]
    uni_mod.__refloxide_patched__ = True  # ty: ignore[unresolved-attribute]
    if patch_reflectivity:
        patched_refl = partial(
            reflectivity,
            use_rust=use_rust,
            parallel=parallel,
        )
        model_mod.reflectivity = patched_refl  # ty: ignore[invalid-assignment, unresolved-attribute]
    _patch_reflect_model_model(model_mod)
    _patch_reflect_model_fused(model_mod, parallel=parallel)
    _patch_reflect_model_eval_energy(model_mod)
    _patch_pyref_structure_ior()
    _patch_pyref_slab_materialization(st_mod)
    export_names: tuple[str, ...] = ()
    if sld_classes != "legacy":
        export_names = _patch_pyref_sld_exports(st_mod, sld_classes)
    model_mod.__refloxide_patched__ = True  # ty: ignore[unresolved-attribute]
    model_mod.__refloxide_sld_classes__ = sld_classes  # ty: ignore[unresolved-attribute]
    return PyrefPatchReport(
        kernels=True,
        reflectivity=patch_reflectivity,
        model_layout=getattr(
            model_mod.ReflectModel, "__refloxide_model_patched__", False
        ),
        fused_path=getattr(
            model_mod.ReflectModel, "__refloxide_fused_patched__", False
        ),
        structure_ior=getattr(st_mod.Structure, "__refloxide_ior_patched__", False),
        structure_materialization=getattr(
            st_mod.Slab, "__refloxide_slab_patched__", False
        ),
        sld_classes=sld_classes,
        sld_exports=export_names,
    )


def patch_pyref_if_needed(
    *,
    use_rust: bool = True,
    parallel: bool = False,
    patch_reflectivity: bool = True,
    sld_classes: SldClassesMode = "both",
    force: bool = False,
) -> PyrefPatchReport:
    """Apply :func:`patch_pyref` once per process unless already configured.

    Parameters
    ----------
    use_rust, parallel, patch_reflectivity, sld_classes
        Forwarded to :func:`patch_pyref`.
    force
        When ``True``, re-run the patch even if :func:`pyref_patched` is ``True``.

    Returns
    -------
    PyrefPatchReport
        Status after this call (existing report when skipped, fresh report when
        applied).
    """
    if pyref_patched() and not force:
        return pyref_patch_report()
    return patch_pyref(
        use_rust=use_rust,
        parallel=parallel,
        patch_reflectivity=patch_reflectivity,
        sld_classes=sld_classes,
    )


def pyref_patch_report() -> PyrefPatchReport:
    """Return the current refloxide hook status on ``pyref.fitting``."""
    import importlib

    try:
        model_mod = importlib.import_module("pyref.fitting.model")
        st_mod = importlib.import_module("pyref.fitting.structure")
        uni_mod = importlib.import_module("pyref.fitting.uniaxial")
    except ModuleNotFoundError:
        return PyrefPatchReport(
            kernels=False,
            reflectivity=False,
            model_layout=False,
            fused_path=False,
            structure_ior=False,
            structure_materialization=False,
            sld_classes="legacy",
            sld_exports=(),
        )
    sld_classes = getattr(model_mod, "__refloxide_sld_classes__", "legacy")
    return PyrefPatchReport(
        kernels=bool(getattr(uni_mod, "__refloxide_patched__", False)),
        reflectivity=getattr(model_mod, "reflectivity", None) is not None
        and getattr(uni_mod, "__refloxide_patched__", False),
        model_layout=getattr(
            model_mod.ReflectModel, "__refloxide_model_patched__", False
        ),
        fused_path=getattr(
            model_mod.ReflectModel, "__refloxide_fused_patched__", False
        ),
        structure_ior=getattr(st_mod.Structure, "__refloxide_ior_patched__", False),
        structure_materialization=getattr(
            st_mod.Slab, "__refloxide_slab_patched__", False
        ),
        sld_classes=sld_classes,
        sld_exports=tuple(getattr(st_mod, "__refloxide_sld_export_names__", ())),
    )


def pyref_patched() -> bool:
    """Return whether :func:`patch_pyref` has installed refloxide kernels in pyref."""
    import importlib

    try:
        model_mod = importlib.import_module("pyref.fitting.model")
    except ModuleNotFoundError:
        return False
    return bool(getattr(model_mod, "__refloxide_patched__", False)) and bool(
        getattr(model_mod.ReflectModel, "__refloxide_model_patched__", False)
    )


def require_pyref_patched() -> None:
    """Raise when pyref is not configured for refloxide laboratory channels.

    Call :func:`patch_pyref` or ``import utils.models`` (refl-analysis) before
    constructing or evaluating :class:`pyref.fitting.model.ReflectModel`.
    """
    if pyref_patched():
        return
    msg = (
        "pyref.fitting is not patched for refloxide. Call patch_pyref() or "
        "import utils.models before fitting so ReflectModel reads laboratory "
        "pyref-compatible s/p channel extraction on the native Jones matrix."
    )
    raise RuntimeError(msg)


__all__ = [
    "PyrefPatchReport",
    "patch_pyref",
    "patch_pyref_if_needed",
    "pyref_patch_report",
    "pyref_patched",
    "reflectivity",
    "require_pyref_patched",
    "uniaxial_reflectivity",
]

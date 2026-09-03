"""Depth-resolved electric-field reconstruction for uniaxial reflectivity stacks.

Reconstructs the standing-wave electric field inside a stratified uniaxial film
using the same conventions as :func:`refloxide.python.tmm.uniaxial_reflectivity`
(optic axis along the surface normal, plane of incidence ``x-z``, ``phi = 0``).
The stack is described by the ``layers`` table ``(N, 4)`` and per-layer diagonal
``tensor`` ``(N, 3, 3)`` that the refloxide/pyref kernels consume, so a materialized
``Structure`` can be fed directly.

Two independent, decoupled polarizations are supported for the diagonal-tensor
uniaxial geometry:

- ``"s"`` (ordinary / TE): the electric field is a scalar ``E_y``; the layer
  dispersion is ``kz = sqrt(eps_o * k0**2 - kx**2)``.
- ``"p"`` (extraordinary / TM): the tangential magnetic field ``H_y`` is scalar
  and the electric field lies in the plane of incidence; the layer dispersion is
  ``kz = sqrt(eps_o * k0**2 - kx**2 * eps_o / eps_e)``.

Amplitudes are propagated with a per-polarization 2x2 Abeles transfer matrix that
carries the identical Nevot-Croce interface roughness factors as the reference
4x4 kernel, so the reflectance recovered from the field solution reproduces the
kernel's ``R_ss`` / ``R_pp`` (validated by :func:`reflectance_from_field`).

The module owns field reconstruction only; it does not build stacks, resolve
photon energy, or apply instrument scaling. Feed it materialized ``layers`` /
``tensor`` arrays and either a single ``q`` (:func:`uniaxial_field_profile` for
the tangential field and intensity, :func:`uniaxial_field_components` for the
complex ``(E_x, E_y, E_z)`` vector used to build real-space ``(x, z)`` maps) or a
``q`` grid (:func:`uniaxial_field_map`) for a depth-vs-``q`` field map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

HC_EV_ANGSTROM = 12398.4193

Polarization = Literal["s", "p"]


@dataclass(frozen=True, slots=True)
class FieldProfile:
    """Depth-resolved field solution for one polarization at one ``q``.

    Parameters
    ----------
    depth
        Sample depths in angstrom, measured from the fronting/first-layer
        interface (``depth < 0`` is the incident medium). Shape ``(n_z,)``.
    e_field
        Complex electric-field amplitude sampled at ``depth`` and normalized so
        the incident wave carries unit electric-field magnitude. For ``"s"``
        this is the scalar ``E_y``; for ``"p"`` it is the tangential ``E_x``.
        Shape ``(n_z,)``.
    intensity
        Squared magnitude of the full electric field ``|E|**2`` at each depth,
        normalized to the incident field. For ``"s"`` equals ``abs(e_field)**2``;
        for ``"p"`` includes the ``E_z`` component. Shape ``(n_z,)``.
    reflectance
        Power reflectance ``|r|**2`` recovered from the transfer matrix; matches
        the kernel ``R_ss`` (``"s"``) or ``R_pp`` (``"p"``).
    polarization
        ``"s"`` or ``"p"``.
    """

    depth: NDArray[np.float64]
    e_field: NDArray[np.complex128]
    intensity: NDArray[np.float64]
    reflectance: float
    polarization: Polarization


@dataclass(frozen=True, slots=True)
class FieldMap:
    """Depth-vs-``q`` field map for one polarization.

    Parameters
    ----------
    q
        Scattering vectors in inverse angstrom, shape ``(n_q,)``.
    depth
        Common depth grid in angstrom, shape ``(n_z,)``; ``0`` is the
        fronting/first-layer interface unless the caller shifts it.
    e_field
        Complex tangential electric field, shape ``(n_q, n_z)``, normalized so
        the incident wave carries unit field magnitude (``E_y`` for ``"s"``,
        ``E_x`` for ``"p"``).
    intensity
        Normalized ``|E|**2`` including ``E_z`` for ``"p"``, shape ``(n_q, n_z)``.
    reflectance
        Power reflectance per ``q``, shape ``(n_q,)``.
    polarization
        ``"s"`` or ``"p"``.
    """

    q: NDArray[np.float64]
    depth: NDArray[np.float64]
    e_field: NDArray[np.complex128]
    intensity: NDArray[np.float64]
    reflectance: NDArray[np.float64]
    polarization: Polarization


@dataclass(frozen=True, slots=True)
class FieldComponents:
    """Depth-resolved complex field vector for one polarization at one ``q``.

    Parameters
    ----------
    depth
        Sample depths in angstrom, measured from the fronting/first-layer
        interface (``depth < 0`` is the incident medium). Shape ``(n_z,)``.
    e_x, e_y, e_z
        Complex Cartesian electric-field components sampled at ``depth``,
        normalized to unit incident ``|E|``. For ``"s"`` only ``e_y`` is
        nonzero; for ``"p"`` only ``e_x`` and ``e_z`` are nonzero. Each has
        shape ``(n_z,)`` and excludes the lateral ``exp(i k_x x)`` factor.
    kx
        In-plane wavevector (inverse angstrom) shared by every layer; multiply
        by ``x`` to form the real-space lateral phase ``exp(i k_x x)``.
    reflectance
        Power reflectance ``|r|**2`` recovered from the transfer matrix.
    polarization
        ``"s"`` or ``"p"``.
    """

    depth: NDArray[np.float64]
    e_x: NDArray[np.complex128]
    e_y: NDArray[np.complex128]
    e_z: NDArray[np.complex128]
    kx: float
    reflectance: float
    polarization: Polarization


@dataclass(frozen=True, slots=True)
class _StackSolution:
    """Per-``q`` amplitude solution used by the depth samplers."""

    kz: NDArray[np.complex128]
    eps_o: NDArray[np.complex128]
    kx: float
    amplitudes: NDArray[np.complex128]
    reflectance: float
    top_z: NDArray[np.float64]
    finite_bottoms: NDArray[np.float64]


def _tensor_to_epsilon(tensor: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Apply the kernel conversion ``epsilon = conj(I - 2 * tensor)`` per layer."""
    eye = np.eye(3, dtype=np.complex128)
    return np.conj(eye[None, :, :] - 2.0 * np.asarray(tensor, dtype=np.complex128))


def _layer_kz(
    eps_o: NDArray[np.complex128],
    eps_e: NDArray[np.complex128],
    kx: float,
    k0: float,
    pol: Polarization,
) -> NDArray[np.complex128]:
    """Forward ``kz`` per layer using the tjf4x4 uniaxial dispersion relations."""
    if pol == "s":
        return np.sqrt(eps_o * k0**2 - kx**2)
    return np.sqrt(eps_o * k0**2 - kx**2 * eps_o / eps_e)


def _layer_admittance(
    kz: NDArray[np.complex128],
    eps_o: NDArray[np.complex128],
    pol: Polarization,
) -> NDArray[np.complex128]:
    """Tangential-field admittance: ``kz`` (TE) or ``kz / eps_o`` (TM)."""
    if pol == "s":
        return kz
    return kz / eps_o


def _interface_matrix(
    eta_top: complex,
    eta_bot: complex,
    kz_top: complex,
    kz_bot: complex,
    sigma: float,
) -> NDArray[np.complex128]:
    """Abeles interface matrix (top layer -> lower layer) with Nevot-Croce roughness.

    Mirrors the ``Di @ D`` element scaling by the tjf4x4 ``W`` matrix: diagonal
    (transmission) entries carry ``exp(-(kz_top - kz_bot)**2 sigma**2 / 2)`` and
    off-diagonal (reflection) entries carry ``exp(-(kz_top + kz_bot)**2 sigma**2 / 2)``.
    """
    e_minus = np.exp(-((kz_top - kz_bot) ** 2) * sigma**2 / 2.0)
    e_plus = np.exp(-((kz_top + kz_bot) ** 2) * sigma**2 / 2.0)
    inv = 1.0 / (2.0 * eta_top)
    return (
        np.array(
            [
                [(eta_top + eta_bot) * e_minus, (eta_top - eta_bot) * e_plus],
                [(eta_top - eta_bot) * e_plus, (eta_top + eta_bot) * e_minus],
            ],
            dtype=np.complex128,
        )
        * inv
    )


def _wavevector_x(q: float, k0: float) -> float:
    """In-plane wavevector ``kx = k0 sqrt(1 - (q / 2 k0)**2)`` (tjf4x4 convention)."""
    ratio = np.clip(q / (2.0 * k0), -1.0, 1.0)
    return float(k0 * np.sqrt(1.0 - ratio**2))


def _layer_amplitudes(
    kz: NDArray[np.complex128],
    eta: NDArray[np.complex128],
    thickness: NDArray[np.float64],
    roughness: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], complex, complex]:
    """Forward/backward field amplitudes per layer at each layer's top interface.

    Returns the ``(N, 2)`` amplitude array ``[a_forward, a_backward]`` referenced
    to the top interface of each layer (incident forward amplitude fixed to one),
    the reflection amplitude ``r``, and the transmission amplitude ``t``.
    """
    n_layers = kz.size
    interfaces = [
        _interface_matrix(eta[j - 1], eta[j], kz[j - 1], kz[j], float(roughness[j]))
        for j in range(1, n_layers)
    ]
    eye = np.eye(2, dtype=np.complex128)

    def propagate(j: int) -> NDArray[np.complex128]:
        """Top-to-bottom amplitude propagation across finite layer ``j``."""
        if j <= 0 or j >= n_layers - 1:
            return eye
        return np.array(
            [
                [np.exp(1j * kz[j] * thickness[j]), 0.0],
                [0.0, np.exp(-1j * kz[j] * thickness[j])],
            ],
            dtype=np.complex128,
        )

    total = np.eye(2, dtype=np.complex128)
    for j in range(1, n_layers - 1):
        total = total @ interfaces[j - 1] @ np.linalg.inv(propagate(j))
    total = total @ interfaces[n_layers - 2]

    r = total[1, 0] / total[0, 0]
    t = 1.0 / total[0, 0]

    amplitudes = np.zeros((n_layers, 2), dtype=np.complex128)
    amplitudes[0] = np.array([1.0, r], dtype=np.complex128)
    for j in range(1, n_layers):
        propagated = propagate(j - 1) @ amplitudes[j - 1]
        amplitudes[j] = np.linalg.solve(interfaces[j - 1], propagated)
    return amplitudes, complex(r), complex(t)


def _validate_stack(
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    pol: Polarization,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    if pol not in ("s", "p"):
        msg = f"pol must be 's' or 'p', got {pol!r}"
        raise ValueError(msg)
    layers = np.asarray(layers, dtype=np.float64)
    tensor = np.asarray(tensor, dtype=np.complex128)
    if layers.shape[0] != tensor.shape[0]:
        msg = (
            f"layers ({layers.shape[0]}) and tensor ({tensor.shape[0]}) layer mismatch"
        )
        raise ValueError(msg)
    return layers, tensor


def _solve_stack(
    q: float,
    layers: NDArray[np.float64],
    eps: NDArray[np.complex128],
    k0: float,
    pol: Polarization,
) -> _StackSolution:
    """Solve forward/backward amplitudes for a single ``q`` (validated inputs)."""
    kx = _wavevector_x(float(q), k0)
    eps_o = eps[:, 0, 0]
    eps_e = eps[:, 2, 2]
    kz = _layer_kz(eps_o, eps_e, kx, k0, pol)
    eta = _layer_admittance(kz, eps_o, pol)

    thickness = layers[:, 0]
    roughness = layers[:, 3]
    amplitudes, r, _t = _layer_amplitudes(kz, eta, thickness, roughness)

    n_layers = kz.size
    finite_bottoms = np.cumsum(thickness[1:-1])
    top_z = np.zeros(n_layers, dtype=np.float64)
    if finite_bottoms.size:
        top_z[2:] = finite_bottoms
    return _StackSolution(
        kz=kz,
        eps_o=eps_o,
        kx=kx,
        amplitudes=amplitudes,
        reflectance=float(np.abs(r) ** 2),
        top_z=top_z,
        finite_bottoms=finite_bottoms,
    )


def _sample_components(
    depth: NDArray[np.float64],
    solution: _StackSolution,
    pol: Polarization,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Sample the normalized complex ``(E_x, E_y, E_z)`` at arbitrary depths.

    Fields are normalized so the incident wave carries unit electric-field
    magnitude: for ``"s"`` only ``E_y`` is populated; for ``"p"`` only ``E_x``
    and ``E_z`` are populated (``E_y = 0``), divided by the incident ``|E|``
    implied by the tangential ``H_y`` amplitude one in the fronting medium.
    """
    n_layers = solution.kz.size
    layer_index = np.searchsorted(solution.finite_bottoms, depth, side="right") + 1
    layer_index[depth < 0.0] = 0
    layer_index = np.clip(layer_index, 0, n_layers - 1)

    z_ref = solution.top_z[layer_index]
    a_fwd = solution.amplitudes[layer_index, 0]
    a_bwd = solution.amplitudes[layer_index, 1]
    kz_local = solution.kz[layer_index]
    phase = kz_local * (depth - z_ref)
    forward = a_fwd * np.exp(1j * phase)
    backward = a_bwd * np.exp(-1j * phase)
    primary = forward + backward

    e_x = np.zeros_like(primary)
    e_y = np.zeros_like(primary)
    e_z = np.zeros_like(primary)
    if pol == "s":
        e_y = primary
        norm = 1.0
    else:
        eps_o_local = solution.eps_o[layer_index]
        e_x = (kz_local / eps_o_local) * (forward - backward)
        e_z = -(solution.kx / eps_o_local) * primary
        norm = np.sqrt(
            np.abs(solution.kz[0] / solution.eps_o[0]) ** 2
            + np.abs(solution.kx / solution.eps_o[0]) ** 2
        )

    inv = 1.0 / norm
    return (
        (e_x * inv).astype(np.complex128),
        (e_y * inv).astype(np.complex128),
        (e_z * inv).astype(np.complex128),
    )


def _sample_field(
    depth: NDArray[np.float64],
    solution: _StackSolution,
    pol: Polarization,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Sample the normalized tangential field and full ``|E|**2`` at depths."""
    e_x, e_y, e_z = _sample_components(depth, solution, pol)
    e_field = e_y if pol == "s" else e_x
    intensity = np.abs(e_x) ** 2 + np.abs(e_y) ** 2 + np.abs(e_z) ** 2
    return e_field, intensity.astype(np.float64)


def substrate_interface_depth(layers: NDArray[np.float64]) -> float:
    """Return the depth (angstrom) of the deepest interface (top of the backing).

    Sums every finite-layer thickness (all layers except the semi-infinite
    fronting and backing), i.e. the depth at which the backing medium begins.
    """
    layers = np.asarray(layers, dtype=np.float64)
    return float(np.sum(layers[1:-1, 0]))


def uniaxial_field_profile(  # noqa: PLR0913
    q: float,
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    pol: Polarization,
    *,
    n_points: int = 2000,
    depth_pad: float = 30.0,
    depth: NDArray[np.float64] | None = None,
) -> FieldProfile:
    """Reconstruct the depth-resolved electric field for one polarization at ``q``.

    Parameters
    ----------
    q
        Single scattering vector in inverse angstrom.
    layers
        Slab table ``(N, 4)`` with rows ``[thickness_A, delta, beta, roughness_A]``
        as consumed by :func:`refloxide.python.tmm.uniaxial_reflectivity`. Only the
        thickness (column 0) and roughness (column 3) are used here; optical
        constants are taken from ``tensor``.
    tensor
        Per-layer diagonal index tensor ``(N, 3, 3)`` carrying ``delta + i beta``
        per principal axis before the kernel conversion ``eps = conj(I - 2 tensor)``.
    energy
        Photon energy in eV.
    pol
        ``"s"`` (ordinary/TE) or ``"p"`` (extraordinary/TM).
    n_points
        Number of depth samples when ``depth`` is not supplied.
    depth_pad
        Extra depth (angstrom) sampled into the incident medium (``depth < 0``)
        and past the final interface into the backing medium. Ignored when
        ``depth`` is supplied.
    depth
        Explicit depth grid (angstrom) to sample. When ``None`` a uniform grid
        spanning ``[-depth_pad, total_thickness + depth_pad]`` is built.

    Returns
    -------
    FieldProfile
        Depth grid, complex field, normalized intensity, and recovered reflectance.

    Raises
    ------
    ValueError
        When ``layers`` and ``tensor`` disagree in layer count or ``pol`` is not
        ``"s"`` or ``"p"``.
    """
    layers, tensor = _validate_stack(layers, tensor, pol)
    k0 = 2.0 * np.pi * energy / HC_EV_ANGSTROM
    eps = _tensor_to_epsilon(tensor)
    solution = _solve_stack(float(q), layers, eps, k0, pol)

    if depth is None:
        total_thick = (
            float(solution.finite_bottoms[-1]) if solution.finite_bottoms.size else 0.0
        )
        depth = np.linspace(-depth_pad, total_thick + depth_pad, int(n_points))
    else:
        depth = np.asarray(depth, dtype=np.float64)

    e_field, intensity = _sample_field(depth, solution, pol)
    return FieldProfile(
        depth=depth,
        e_field=e_field,
        intensity=intensity,
        reflectance=solution.reflectance,
        polarization=pol,
    )


def uniaxial_field_components(  # noqa: PLR0913
    q: float,
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    pol: Polarization,
    *,
    n_points: int = 2000,
    depth_pad: float = 30.0,
    depth: NDArray[np.float64] | None = None,
) -> FieldComponents:
    """Reconstruct the complex field vector ``(E_x, E_y, E_z)`` for one ``q``.

    Unlike :func:`uniaxial_field_profile`, this returns the individual Cartesian
    components (and the shared ``k_x``) so callers can build real-space ``(x, z)``
    maps via the lateral phase ``exp(i k_x x)`` and render standing-wave fronts
    ``Re(E_a exp(i k_x x))`` or intensities ``|E|**2``.

    Parameters
    ----------
    q
        Single scattering vector in inverse angstrom.
    layers, tensor, energy, pol
        Stack description and polarization; see :func:`uniaxial_field_profile`.
    n_points, depth_pad, depth
        Depth-grid controls; identical semantics to :func:`uniaxial_field_profile`.

    Returns
    -------
    FieldComponents
        Depth grid, complex ``e_x`` / ``e_y`` / ``e_z``, shared ``kx``, and the
        recovered reflectance.

    Raises
    ------
    ValueError
        When ``layers`` and ``tensor`` disagree in layer count or ``pol`` is not
        ``"s"`` or ``"p"``.
    """
    layers, tensor = _validate_stack(layers, tensor, pol)
    k0 = 2.0 * np.pi * energy / HC_EV_ANGSTROM
    eps = _tensor_to_epsilon(tensor)
    solution = _solve_stack(float(q), layers, eps, k0, pol)

    if depth is None:
        total_thick = (
            float(solution.finite_bottoms[-1]) if solution.finite_bottoms.size else 0.0
        )
        depth = np.linspace(-depth_pad, total_thick + depth_pad, int(n_points))
    else:
        depth = np.asarray(depth, dtype=np.float64)

    e_x, e_y, e_z = _sample_components(depth, solution, pol)
    return FieldComponents(
        depth=depth,
        e_x=e_x,
        e_y=e_y,
        e_z=e_z,
        kx=solution.kx,
        reflectance=solution.reflectance,
        polarization=pol,
    )


def uniaxial_field_map(  # noqa: PLR0913
    q_values: NDArray[np.float64],
    layers: NDArray[np.float64],
    tensor: NDArray[np.complex128],
    energy: float,
    pol: Polarization,
    depth: NDArray[np.float64],
) -> FieldMap:
    """Reconstruct a depth-vs-``q`` field map on a fixed depth grid.

    Parameters
    ----------
    q_values
        Scattering vectors in inverse angstrom, shape ``(n_q,)``.
    layers, tensor, energy, pol
        Stack description and polarization; see :func:`uniaxial_field_profile`.
    depth
        Depth grid (angstrom) shared across all ``q``, shape ``(n_z,)``. Depths
        are absolute (measured from the fronting/first-layer interface); shift
        with :func:`substrate_interface_depth` to reference the substrate.

    Returns
    -------
    FieldMap
        Complex field, intensity, and reflectance stacked over ``q``.
    """
    layers, tensor = _validate_stack(layers, tensor, pol)
    k0 = 2.0 * np.pi * energy / HC_EV_ANGSTROM
    eps = _tensor_to_epsilon(tensor)
    q_values = np.asarray(q_values, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    e_field = np.empty((q_values.size, depth.size), dtype=np.complex128)
    intensity = np.empty((q_values.size, depth.size), dtype=np.float64)
    reflectance = np.empty(q_values.size, dtype=np.float64)
    for i, q in enumerate(q_values):
        solution = _solve_stack(float(q), layers, eps, k0, pol)
        e_field[i], intensity[i] = _sample_field(depth, solution, pol)
        reflectance[i] = solution.reflectance

    return FieldMap(
        q=q_values,
        depth=depth,
        e_field=e_field,
        intensity=intensity,
        reflectance=reflectance,
        polarization=pol,
    )


def reflectance_from_field(profile: FieldProfile) -> float:
    """Return the power reflectance recovered from the transfer matrix solution."""
    return profile.reflectance


__all__ = [
    "FieldComponents",
    "FieldMap",
    "FieldProfile",
    "Polarization",
    "reflectance_from_field",
    "substrate_interface_depth",
    "uniaxial_field_components",
    "uniaxial_field_map",
    "uniaxial_field_profile",
]

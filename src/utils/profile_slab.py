from typing import Any

import numpy as np
import pandas as pd
from numba import njit  # type: ignore
from numpy import complexfloating
from numpy._typing import _64Bit
from numpy._typing._array_like import NDArray
from pyref.fitting.structure import PXR_Component as Component
from pyref.fitting.structure import possibly_create_parameter
from scipy.interpolate import PchipInterpolator

_MIN_PCHIP_POINTS = 2


def slice_range(
    df: pd.DataFrame,
    col: str,
    center: float,
    bounds: float,
    min_length: int = 3,
) -> pd.DataFrame:
    """
    Select rows of ``df`` within ``center`` plus/minus ``bounds`` on ``col``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table containing ``col``.
    col : str
        Column used for windowing (typically ``"energy"`` in eV).
    center : float
        Central value; rows with ``col`` in ``[center - bounds, center + bounds]``
        are retained when enough points exist.
    bounds : float
        Half-width of the selection window on ``col``.
    min_length : int, optional
        Minimum number of rows returned. If the window is narrower, the closest
        rows to ``center`` along ``col`` are used instead.

    Returns
    -------
    pandas.DataFrame
        Subset of ``df`` sorted by ``col`` with unique ``col`` values.

    Raises
    ------
    ValueError
        If ``col`` is missing or fewer than two distinct points remain.
    """
    if col not in df.columns:
        msg = f"column {col!r} not in dataframe"
        raise ValueError(msg)

    lo = center - bounds
    hi = center + bounds
    in_window = df[(df[col] >= lo) & (df[col] <= hi)].sort_values(col)

    if len(in_window) >= min_length:
        selected = in_window
    else:
        need = max(min_length, _MIN_PCHIP_POINTS)
        nearest_idx = (df[col] - center).abs().nsmallest(need).index
        selected = df.loc[nearest_idx].sort_values(col)

    out = selected.drop_duplicates(subset=[col], keep="first").reset_index(drop=True)
    if len(out) < _MIN_PCHIP_POINTS:
        msg = (
            f"need at least {_MIN_PCHIP_POINTS} distinct points near {center} "
            f"on {col!r}; got {len(out)}"
        )
        raise ValueError(msg)
    return out


@njit(cache=True, fastmath=True)
def _orientation_profile_core(
    total_thick,
    depth,
    characteristic_thickness,
    max_angle,
    initial_angle,
):
    return max_angle * (1 - np.exp(-depth / characteristic_thickness)) + initial_angle


def _scaled_molecular_components(
    n_xx,
    n_ixx,
    n_zz,
    n_izz,
    energy: float,
    density: float,
) -> tuple[complex, complex]:
    n_mol_xx = density * complex(n_xx(energy) + 1j * n_ixx(energy))
    n_mol_zz = density * complex(n_zz(energy) + 1j * n_izz(energy))
    return n_mol_xx, n_mol_zz


def _lab_tensor_diagonals(
    n_mol_xx: complex,
    n_mol_zz: complex,
    orientation: NDArray[np.float64] | float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    c = np.square(np.cos(orientation))
    s = np.square(np.sin(orientation))
    n_o = (n_mol_xx * (1 + c) + n_mol_zz * s) / 2
    n_e = n_mol_xx * s + n_mol_zz * c
    return np.asarray(n_o, dtype=np.complex128), np.asarray(n_e, dtype=np.complex128)


def _assemble_diagonal_tensor(
    n_o: NDArray[np.complex128],
    n_e: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    tensor = np.zeros((n_o.size, 3, 3), dtype=np.complex128)
    tensor[:, 0, 0] = n_o
    tensor[:, 1, 1] = n_o
    tensor[:, 2, 2] = n_e
    return tensor


def orientation_profile(
    total_thick, depth, characteristic_thickness, max_angle, initial_angle
):
    if np.isscalar(depth):
        return _orientation_profile_core(
            total_thick,
            depth,
            characteristic_thickness,
            max_angle,
            initial_angle,
        )

    depth_arr = np.asarray(depth, dtype=np.float64)
    return _orientation_profile_core(
        total_thick,
        depth_arr,
        characteristic_thickness,
        max_angle,
        initial_angle,
    )


@njit(cache=True, fastmath=True)
def _orientation_profile_bookended_core(
    total_thick,
    depth,
    tau_si,
    tau_vac,
    alpha_bulk,
    alpha_si,
    alpha_vac,
):
    dist_from_surface = depth
    dist_from_substrate = total_thick - depth
    term_vac = (alpha_vac - alpha_bulk) * np.exp(-dist_from_surface / tau_vac)
    term_si = (alpha_si - alpha_bulk) * np.exp(-dist_from_substrate / tau_si)
    return alpha_bulk + term_vac + term_si


@njit(cache=True, fastmath=True)
def _average_orientation_bookended_core(
    total_thick,
    tau_si,
    tau_vac,
    alpha_bulk,
    alpha_si,
    alpha_vac,
):
    term_si = (
        (tau_si / total_thick)
        * (alpha_si - alpha_bulk)
        * (1.0 - np.exp(-total_thick / tau_si))
    )
    term_vac = (
        (tau_vac / total_thick)
        * (alpha_vac - alpha_bulk)
        * (1.0 - np.exp(-total_thick / tau_vac))
    )
    return alpha_bulk + term_si + term_vac


def orientation_profile_bookended(
    total_thick: float,
    depth: float | NDArray[np.float64],
    tau_si: float,
    tau_vac: float,
    alpha_bulk: float,
    alpha_si: float,
    alpha_vac: float,
) -> float | NDArray[np.float64]:
    """
    Book-ended tilt profile at surface and buried (substrate) interface.

    alpha(z) = alpha_bulk + (alpha_vac - alpha_bulk)*exp(-z/tau_vac)
               + (alpha_si - alpha_bulk)*exp(-(L-z)/tau_si)
    with z = depth from surface (0 = vacuum, L = substrate).

    Parameters
    ----------
    total_thick : float
        Film thickness L (same units as depth, tau_si, tau_vac).
    depth : float or ndarray
        Distance from surface (0 = vacuum side).
    tau_si : float
        Substrate-side decay length.
    tau_vac : float
        Vacuum-side decay length.
    alpha_bulk, alpha_si, alpha_vac : float
        Tilt angles (degrees if angles_degrees True, else radians).
    angles_degrees : bool, optional
        If True, input/output angles in degrees; core uses radians internally.

    Returns
    -------
    float or ndarray
        Tilt angle(s) in radians (for use with cos/sin in tensor).
    """
    alpha_bulk_rad = alpha_bulk
    alpha_si_rad = alpha_si
    alpha_vac_rad = alpha_vac
    if np.isscalar(depth):
        return _orientation_profile_bookended_core(
            total_thick,
            depth,
            tau_si,
            tau_vac,
            alpha_bulk_rad,
            alpha_si_rad,
            alpha_vac_rad,
        )
    depth_arr = np.asarray(depth, dtype=np.float64)
    return _orientation_profile_bookended_core(
        total_thick,
        depth_arr,
        tau_si,
        tau_vac,
        alpha_bulk_rad,
        alpha_si_rad,
        alpha_vac_rad,
    )


@njit(cache=True, fastmath=True)
def _density_profile_bookended_core(
    total_thick,
    depth,
    tau_si,
    tau_vac,
    rho_bulk,
    rho_si,
    rho_vac,
):
    dist_from_surface = depth
    dist_from_substrate = total_thick - depth
    term_vac = (rho_vac - rho_bulk) * np.exp(-dist_from_surface / tau_vac)
    term_si = (rho_si - rho_bulk) * np.exp(-dist_from_substrate / tau_si)
    return rho_bulk + term_vac + term_si


def density_profile_bookended(
    total_thick: float,
    depth: float | NDArray[np.float64],
    tau_si: float,
    tau_vac: float,
    rho_bulk: float,
    rho_si: float,
    rho_vac: float,
) -> float | NDArray[np.float64]:
    """
    Book-ended density profile at surface and buried (substrate) interface.

    rho(z) = rho_bulk + (rho_vac - rho_bulk)*exp(-z/tau_vac)
             + (rho_si - rho_bulk)*exp(-(L-z)/tau_si)
    with z = depth from surface (0 = vacuum, L = substrate).

    Parameters
    ----------
    total_thick : float
        Film thickness L (same units as depth and tau values).
    depth : float or ndarray
        Distance from surface (0 = vacuum side).
    tau_si : float
        Substrate-side decay length (same units as depth).
    tau_vac : float
        Vacuum-side decay length (same units as depth).
    rho_bulk, rho_si, rho_vac : float
        Density in bulk, at substrate interface, and at vacuum interface.

    Returns
    -------
    float or ndarray
        Density at given depth(s).
    """
    if np.isscalar(depth):
        return _density_profile_bookended_core(
            total_thick,
            depth,
            tau_si,
            tau_vac,
            rho_bulk,
            rho_si,
            rho_vac,
        )
    depth_arr = np.asarray(depth, dtype=np.float64)
    return _density_profile_bookended_core(
        total_thick,
        depth_arr,
        tau_si,
        tau_vac,
        rho_bulk,
        rho_si,
        rho_vac,
    )


def average_orientation_bookended(
    total_thick: float,
    tau_si: float,
    tau_vac: float,
    alpha_bulk: float,
    alpha_si: float,
    alpha_vac: float,
    angles_degrees: bool = True,
) -> float:
    """
    Thickness-averaged tilt for the book-ended profile (closed-form integral).

    Parameters
    ----------
    total_thick, tau_si, tau_vac, alpha_bulk, alpha_si, alpha_vac
        Same as orientation_profile_bookended.
    angles_degrees : bool, optional
        If True, input angles in degrees and return value in degrees.

    Returns
    -------
    float
        Average tilt over the film (degrees if angles_degrees True, else radians).
    """
    if angles_degrees:
        alpha_bulk_rad = np.deg2rad(alpha_bulk)
        alpha_si_rad = np.deg2rad(alpha_si)
        alpha_vac_rad = np.deg2rad(alpha_vac)
    else:
        alpha_bulk_rad = alpha_bulk
        alpha_si_rad = alpha_si
        alpha_vac_rad = alpha_vac
    result_rad = _average_orientation_bookended_core(
        total_thick,
        tau_si,
        tau_vac,
        alpha_bulk_rad,
        alpha_si_rad,
        alpha_vac_rad,
    )
    return np.rad2deg(result_rad) if angles_degrees else result_rad


class OrientationProfile(Component):
    def __init__(
        self,
        ooc: pd.DataFrame,
        total_thick,
        surface_roughness,
        density,
        characteristic_thickness,
        max_angle,
        initial_angle,
        energy,
        energy_offset: float = 0,
        name: str | None = None,
        microslab_max_thickness: float = 1,
    ):
        super(OrientationProfile, self).__init__(name=name)
        # Load the OOC data
        self.energy = energy
        self._load_ooc(ooc, energy)
        # Setup the parameters
        self.total_thick = possibly_create_parameter(total_thick, name="total_thick")
        self.surface_roughness = possibly_create_parameter(
            surface_roughness, name="surface_roughness"
        )
        self.density = possibly_create_parameter(density, name="density")
        self.characteristic_thickness = possibly_create_parameter(
            characteristic_thickness, name="characteristic_thickness"
        )
        self.max_angle = possibly_create_parameter(max_angle, name="max_angle")
        self.initial_angle = possibly_create_parameter(
            initial_angle, name="initial_angle"
        )
        self.energy_offset = possibly_create_parameter(
            energy_offset, name="energy_offset"
        )
        # Initialize the microslabs
        self.microslab_max_thickness = microslab_max_thickness
        # Load into the super class
        self._parameters = super().parameters
        self._parameters.extend(
            [
                self.total_thick,
                self.surface_roughness,
                self.density,
                self.characteristic_thickness,
                self.max_angle,
                self.initial_angle,
                self.energy_offset,
            ]
        )

    def _load_ooc(self, ooc: pd.DataFrame, energy: float):
        """Loac Optical Constants from a DataFrame."""

        # Validate the DataFrame
        required_columns = ["energy", "n_xx", "n_ixx", "n_zz", "n_izz"]
        if not all(col in ooc.columns for col in required_columns):
            missing = [col for col in required_columns if col not in ooc.columns]
            e = f"Optical constants dataframe missing required columns: {missing}"
            raise ValueError(e)
        cropped_tensor = slice_range(ooc, "energy", energy, 0.5)
        self.n_xx = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_xx"])
        self.n_ixx = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_ixx"]
        )
        self.n_zz = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_zz"])
        self.n_izz = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_izz"]
        )

    def varying_parameters(self):
        return [p for p in self._parameters if p.vary]

    @property
    def num_slabs(self) -> int:
        return int(
            np.ceil(float(self.total_thick.value) / self.microslab_max_thickness)
        )

    @property
    def slab_thick(self) -> float:
        return self.total_thick.value / self.num_slabs

    @property
    def dist(self) -> NDArray[np.float64]:
        return np.linspace(self.slab_thick, self.total_thick.value, self.num_slabs)

    @property
    def mid_points(self) -> NDArray[np.float64]:
        return np.linspace(
            self.slab_thick / 2,
            self.total_thick.value - self.slab_thick / 2,
            self.num_slabs,
        )

    @property
    def parameters(self) -> Any:
        return self._parameters

    def orientation(
        self,
        depth: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        depth_from_substrate = self.total_thick.value - depth
        return orientation_profile(
            self.total_thick.value,
            depth_from_substrate,
            self.characteristic_thickness.value,
            self.max_angle.value,
            self.initial_angle.value,
        )

    def get_energy(self) -> float:
        return float(self.energy) + float(self.energy_offset.value)

    def tensor(
        self,
        energy: float | None = None,
    ) -> NDArray[complexfloating[_64Bit, _64Bit]]:
        if energy is None:
            energy = self.get_energy()
        depth_arr = self.mid_points
        ori = self.orientation(depth_arr)
        n_mol_xx, n_mol_zz = _scaled_molecular_components(
            self.n_xx,
            self.n_ixx,
            self.n_zz,
            self.n_izz,
            energy,
            float(self.density.value),
        )
        n_o, n_e = _lab_tensor_diagonals(n_mol_xx, n_mol_zz, ori)
        return _assemble_diagonal_tensor(n_o, n_e)

    @property
    def iso(self) -> NDArray[np.float64]:
        return np.trace(self.tensor(), axis1=1, axis2=2)

    @property
    def delta(self) -> NDArray[np.float64]:
        return np.real(self.iso)

    @property
    def beta(self) -> NDArray[np.float64]:
        return np.imag(self.iso)

    def slabs(self, structure=None) -> NDArray[np.float64]:
        tens = self.tensor()
        iso = np.trace(tens, axis1=1, axis2=2)
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = self.slab_thick
        slabs[..., 1] = np.real(iso)
        slabs[..., 2] = np.imag(iso)
        slabs[0, 3] = self.surface_roughness.value
        return slabs


class AdaptiveOrientationProfile(Component):
    def __init__(
        self,
        ooc: pd.DataFrame,
        total_thick,
        surface_roughness,
        density,
        characteristic_thickness,
        max_angle,
        initial_angle,
        energy,
        energy_offset: float = 0,
        name: str | None = None,
        num_slabs: int = 20,
        mesh_constant: float = 0.1,
    ):
        super(AdaptiveOrientationProfile, self).__init__(name=name)
        self.mesh_constant = mesh_constant
        self.energy = energy
        self._load_ooc(ooc, energy)
        self.total_thick = possibly_create_parameter(total_thick, name="total_thick")
        self.surface_roughness = possibly_create_parameter(
            surface_roughness, name="surface_roughness"
        )
        self.density = possibly_create_parameter(density, name="density")
        self.characteristic_thickness = possibly_create_parameter(
            characteristic_thickness, name="characteristic_thickness"
        )
        self.max_angle = possibly_create_parameter(max_angle, name="max_angle")
        self.initial_angle = possibly_create_parameter(
            initial_angle, name="initial_angle"
        )
        self.energy_offset = possibly_create_parameter(
            energy_offset, name="energy_offset"
        )
        self.num_slabs = num_slabs
        self._parameters = super().parameters
        self._parameters.extend(
            [
                self.total_thick,
                self.surface_roughness,
                self.density,
                self.characteristic_thickness,
                self.max_angle,
                self.initial_angle,
                self.energy_offset,
            ]
        )

    def _load_ooc(self, ooc: pd.DataFrame, energy: float):
        """Loac Optical Constants from a DataFrame."""

        # Validate the DataFrame
        required_columns = ["energy", "n_xx", "n_ixx", "n_zz", "n_izz"]
        if not all(col in ooc.columns for col in required_columns):
            missing = [col for col in required_columns if col not in ooc.columns]
            e = f"Optical constants dataframe missing required columns: {missing}"
            raise ValueError(e)
        cropped_tensor = slice_range(ooc, "energy", energy, 0.5)
        self.n_xx = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_xx"])
        self.n_ixx = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_ixx"]
        )
        self.n_zz = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_zz"])
        self.n_izz = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_izz"]
        )

    def varying_parameters(self):
        return [p for p in self._parameters if p.vary]

    def _generate_adaptive_grid(self) -> NDArray[np.float64]:
        total_thick_val = float(self.total_thick.value)
        num_slabs_val = self.num_slabs

        r = self.mesh_constant ** (1 / (num_slabs_val / 2))
        a = total_thick_val * (1 - r) / (1 - r**num_slabs_val)
        mesh = a * r ** np.arange(num_slabs_val)
        remainder = total_thick_val - mesh.sum()
        mesh[0] += remainder
        return mesh

    @property
    def slab_thick(self) -> NDArray[np.float64]:
        return self._generate_adaptive_grid()

    @property
    def dist(self) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        cumulative = np.cumsum(thicknesses)
        return cumulative

    @property
    def mid_points(self) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        cumulative = np.cumsum(thicknesses)
        mid_points = cumulative - thicknesses / 2
        return mid_points

    @property
    def parameters(self) -> Any:
        return self._parameters

    def orientation(
        self,
        depth: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        depth_from_substrate = self.total_thick.value - depth
        return orientation_profile(
            self.total_thick.value,
            depth_from_substrate,
            self.characteristic_thickness.value,
            self.max_angle.value,
            self.initial_angle.value,
        )

    def get_energy(self) -> float:
        return float(self.energy) + float(self.energy_offset.value)

    def tensor(
        self,
        energy: float | None = None,
    ) -> NDArray[complexfloating[_64Bit, _64Bit]]:
        if energy is None:
            energy = self.get_energy()
        depth_arr = self.mid_points
        ori = self.orientation(depth_arr)
        n_mol_xx, n_mol_zz = _scaled_molecular_components(
            self.n_xx,
            self.n_ixx,
            self.n_zz,
            self.n_izz,
            energy,
            float(self.density.value),
        )
        n_o, n_e = _lab_tensor_diagonals(n_mol_xx, n_mol_zz, ori)
        return _assemble_diagonal_tensor(n_o, n_e)

    @property
    def iso(self) -> NDArray[np.float64]:
        return np.trace(self.tensor(), axis1=1, axis2=2)

    @property
    def delta(self) -> NDArray[np.float64]:
        return np.real(self.iso)

    @property
    def beta(self) -> NDArray[np.float64]:
        return np.imag(self.iso)

    def slabs(self, structure=None) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        tens = self.tensor()
        iso = np.trace(tens, axis1=1, axis2=2)
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = thicknesses
        slabs[..., 1] = np.real(iso)
        slabs[..., 2] = np.imag(iso)
        slabs[0, 3] = self.surface_roughness.value
        return slabs


# ======/ Orientation Profile With Matching Density ======


class AdaptiveOrientationDensityProfile(AdaptiveOrientationProfile):
    """
    Adaptive orientation profile with coupled density gradient.

    Uses the same total thickness and characteristic-thickness kernel as
    :meth:`orientation` to prescribe a slab density gradient. ``initial_density``
    is the buried-interface value and ``max_density`` is the asymptotic
    surface-side value.
    """

    def __init__(
        self,
        ooc: pd.DataFrame,
        total_thick,
        surface_roughness,
        max_density,
        initial_density,
        characteristic_thickness,
        max_angle,
        initial_angle,
        energy,
        energy_offset: float = 0,
        name: str | None = None,
        num_slabs: int = 20,
        mesh_constant: float = 0.1,
    ):
        super(AdaptiveOrientationDensityProfile, self).__init__(
            ooc=ooc,
            total_thick=total_thick,
            surface_roughness=surface_roughness,
            density=max_density,
            characteristic_thickness=characteristic_thickness,
            max_angle=max_angle,
            initial_angle=initial_angle,
            energy=energy,
            energy_offset=energy_offset,
            name=name,
            num_slabs=num_slabs,
            mesh_constant=mesh_constant,
        )
        self.max_density = possibly_create_parameter(max_density, name="max_density")
        density_idx = self._parameters.index(self.density)
        self._parameters[density_idx] = self.max_density
        self.density = self.max_density
        self.initial_density = possibly_create_parameter(
            initial_density, name="initial_density"
        )
        self._parameters.extend([self.initial_density])

    def local_density(
        self,
        depth: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        density_span = self.max_density.value - self.initial_density.value
        depth_from_substrate = self.total_thick.value - depth
        return orientation_profile(
            self.total_thick.value,
            depth_from_substrate,
            self.characteristic_thickness.value,
            density_span,
            self.initial_density.value,
        )

    def tensor(
        self,
        energy: float | None = None,
    ) -> NDArray[complexfloating[_64Bit, _64Bit]]:
        tensor = super(AdaptiveOrientationDensityProfile, self).tensor(energy=energy)
        rho_bulk = float(self.max_density.value)
        rho_local = np.asarray(self.local_density(self.mid_points), dtype=np.float64)
        scale = rho_local / rho_bulk
        tensor[:, 0, 0] *= scale
        tensor[:, 1, 1] *= scale
        tensor[:, 2, 2] *= scale
        return tensor


class AdaptiveBookendedOrientationProfile(Component):
    """
    Adaptive slab profile with book-ended orientation: surface and buried interface.

    Same adaptive grid and tensor logic as AdaptiveOrientationProfile; orientation
    follows the two-interface model with tau_si, tau_vac, alpha_bulk, alpha_si, alpha_vac.
    All angle parameters are in degrees. orientation() returns radians for tensor use.
    """

    def __init__(
        self,
        ooc: pd.DataFrame,
        total_thick,
        surface_roughness,
        density,
        tau_si,
        tau_vac,
        alpha_bulk,
        alpha_si,
        alpha_vac,
        energy,
        energy_offset: float = 0,
        name: str | None = None,
        num_slabs: int = 20,
        mesh_constant: float = 0.1,
    ):
        super(AdaptiveBookendedOrientationProfile, self).__init__(name=name)
        self.mesh_constant = mesh_constant
        self.energy = energy
        self._load_ooc(ooc, energy)
        self.total_thick = possibly_create_parameter(total_thick, name="total_thick")
        self.surface_roughness = possibly_create_parameter(
            surface_roughness, name="surface_roughness"
        )
        self.density = possibly_create_parameter(density, name="density")
        self.tau_si = possibly_create_parameter(tau_si, name="tau_si")
        self.tau_vac = possibly_create_parameter(tau_vac, name="tau_vac")
        self.alpha_bulk = possibly_create_parameter(alpha_bulk, name="alpha_bulk")
        self.alpha_si = possibly_create_parameter(alpha_si, name="alpha_si")
        self.alpha_vac = possibly_create_parameter(alpha_vac, name="alpha_vac")
        self.energy_offset = possibly_create_parameter(
            energy_offset, name="energy_offset"
        )
        self.num_slabs = num_slabs
        self._parameters = super().parameters
        self._parameters.extend(
            [
                self.total_thick,
                self.surface_roughness,
                self.density,
                self.tau_si,
                self.tau_vac,
                self.alpha_bulk,
                self.alpha_si,
                self.alpha_vac,
                self.energy_offset,
            ]
        )

    def _load_ooc(self, ooc: pd.DataFrame, energy: float):
        """Loac Optical Constants from a DataFrame."""

        # Validate the DataFrame
        required_columns = ["energy", "n_xx", "n_ixx", "n_zz", "n_izz"]
        if not all(col in ooc.columns for col in required_columns):
            missing = [col for col in required_columns if col not in ooc.columns]
            e = f"Optical constants dataframe missing required columns: {missing}"
            raise ValueError(e)
        cropped_tensor = slice_range(ooc, "energy", energy, 0.5)
        self.n_xx = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_xx"])
        self.n_ixx = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_ixx"]
        )
        self.n_zz = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_zz"])
        self.n_izz = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_izz"]
        )

    def varying_parameters(self):
        return [p for p in self._parameters if p.vary]

    def _generate_adaptive_grid(self) -> NDArray[np.float64]:
        total_thick_val = float(self.total_thick.value)
        num_slabs_val = self.num_slabs
        if num_slabs_val <= 1:
            return np.array([total_thick_val])
        n_half = num_slabs_val // 2
        half_thick = total_thick_val / 2.0
        r = self.mesh_constant ** (1 / n_half)
        if num_slabs_val % 2 == 0:
            a = half_thick * (r - 1) / (r**n_half - 1)
            mesh_half = a * r ** np.arange(n_half)
            mesh = np.concatenate([mesh_half[::-1], mesh_half])
        else:
            center_share = total_thick_val / num_slabs_val
            half_sum = (total_thick_val - center_share) / 2.0
            a = half_sum * (r - 1) / (r**n_half - 1)
            mesh_half = a * r ** np.arange(n_half)
            center = total_thick_val - 2 * mesh_half.sum()
            mesh = np.concatenate([mesh_half[::-1], [center], mesh_half])
        remainder = total_thick_val - mesh.sum()
        mesh[0] += remainder
        return mesh

    @property
    def slab_thick(self) -> NDArray[np.float64]:
        return self._generate_adaptive_grid()

    @property
    def dist(self) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        cumulative = np.cumsum(thicknesses)
        return cumulative

    @property
    def mid_points(self) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        cumulative = np.cumsum(thicknesses)
        mid_points = cumulative - thicknesses / 2
        return mid_points

    @property
    def parameters(self) -> Any:
        return self._parameters

    def orientation(
        self,
        depth: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        result_rad = orientation_profile_bookended(
            self.total_thick.value,
            depth,
            self.tau_si.value,
            self.tau_vac.value,
            self.alpha_bulk.value,
            self.alpha_si.value,
            self.alpha_vac.value,
        )
        return result_rad

    def average_orientation(self) -> float:
        return average_orientation_bookended(
            self.total_thick.value,
            self.tau_si.value,
            self.tau_vac.value,
            self.alpha_bulk.value,
            self.alpha_si.value,
            self.alpha_vac.value,
        )

    def get_energy(self) -> float:
        return float(self.energy) + float(self.energy_offset.value)

    def tensor(
        self,
        energy: float | None = None,
    ) -> NDArray[complexfloating[_64Bit, _64Bit]]:
        if energy is None:
            energy = self.get_energy()
        depth_arr = self.mid_points
        ori = self.orientation(depth_arr)
        n_mol_xx, n_mol_zz = _scaled_molecular_components(
            self.n_xx,
            self.n_ixx,
            self.n_zz,
            self.n_izz,
            energy,
            float(self.density.value),
        )
        n_o, n_e = _lab_tensor_diagonals(n_mol_xx, n_mol_zz, ori)
        return _assemble_diagonal_tensor(n_o, n_e)

    @property
    def iso(self) -> NDArray[np.float64]:
        return np.trace(self.tensor(), axis1=1, axis2=2)

    @property
    def delta(self) -> NDArray[np.float64]:
        return np.real(self.iso)

    @property
    def beta(self) -> NDArray[np.float64]:
        return np.imag(self.iso)

    def slabs(self, structure=None) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        tens = self.tensor()
        iso = np.trace(tens, axis1=1, axis2=2)
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = thicknesses
        slabs[..., 1] = np.real(iso)
        slabs[..., 2] = np.imag(iso)
        slabs[0, 3] = self.surface_roughness.value
        return slabs


class AdaptiveBookendedOrientationDensityProfile(AdaptiveBookendedOrientationProfile):
    """
    Book-ended orientation and book-ended density on an adaptive microslab grid.

    At each microslab midpoint, orientation uses
    :func:`orientation_profile_bookended` and density uses
    :func:`density_profile_bookended`. Both share ``tau_si`` and ``tau_vac`` as
    the substrate-side and vacuum-side decay lengths. Optical constants are
    scaled by ``rho(z) / density_bulk`` at each depth.
    """

    def __init__(
        self,
        ooc: pd.DataFrame,
        total_thick,
        surface_roughness,
        density_bulk: float,
        density_si: float,
        density_vac: float,
        tau_si,
        tau_vac,
        alpha_bulk,
        alpha_si,
        alpha_vac,
        energy,
        energy_offset: float = 0,
        name: str | None = None,
        num_slabs: int = 20,
        mesh_constant: float = 0.1,
    ):
        super(AdaptiveBookendedOrientationDensityProfile, self).__init__(
            ooc=ooc,
            total_thick=total_thick,
            surface_roughness=surface_roughness,
            density=density_bulk,
            tau_si=tau_si,
            tau_vac=tau_vac,
            alpha_bulk=alpha_bulk,
            alpha_si=alpha_si,
            alpha_vac=alpha_vac,
            energy=energy,
            energy_offset=energy_offset,
            name=name,
            num_slabs=num_slabs,
            mesh_constant=mesh_constant,
        )
        self.density_bulk = possibly_create_parameter(density_bulk, name="density_bulk")
        density_idx = self._parameters.index(self.density)
        self._parameters[density_idx] = self.density_bulk
        self.density = self.density_bulk
        self.density_si = possibly_create_parameter(density_si, name="density_si")
        self.density_vac = possibly_create_parameter(density_vac, name="density_vac")
        self._parameters.extend([self.density_si, self.density_vac])

    def local_density(
        self,
        depth: NDArray[np.float64] | float,
    ) -> float | NDArray[np.float64]:
        return density_profile_bookended(
            self.total_thick.value,
            depth,
            self.tau_si.value,
            self.tau_vac.value,
            self.density_bulk.value,
            self.density_si.value,
            self.density_vac.value,
        )

    def tensor(
        self,
        energy: float | None = None,
    ) -> NDArray[complexfloating[_64Bit, _64Bit]]:
        if energy is None:
            energy = self.get_energy()
        depth_arr = self.mid_points
        ori = self.orientation(depth_arr)
        rho_bulk_val = float(self.density_bulk.value)
        rho_local = self.local_density(depth_arr)
        scale = np.asarray(rho_local, dtype=np.float64) / rho_bulk_val
        n_mol_xx, n_mol_zz = _scaled_molecular_components(
            self.n_xx,
            self.n_ixx,
            self.n_zz,
            self.n_izz,
            energy,
            rho_bulk_val,
        )
        n_o, n_e = _lab_tensor_diagonals(n_mol_xx, n_mol_zz, ori)
        return _assemble_diagonal_tensor(n_o * scale, n_e * scale)


class AdaptiveBookendedProfile(Component):
    """
    Adaptive slab profile with book-ended orientation and density.

    Orientation and density share the same functional form: bulk value plus
    exponential decays from vacuum and substrate interfaces, using ``tau_si``
    and ``tau_vac`` for both angle and density profiles. Optical response is
    scaled by local density / rho_bulk. All angle parameters are in degrees;
    orientation() returns radians for tensor use.
    """

    def __init__(
        self,
        ooc: pd.DataFrame,
        total_thick,
        surface_roughness,
        rho_bulk: float,
        rho_si: float,
        rho_vac: float,
        tau_si: float,
        tau_vac: float,
        alpha_bulk: float,
        alpha_si: float,
        alpha_vac: float,
        energy: float,
        energy_offset: float = 0,
        name: str | None = None,
        num_slabs: int = 20,
        mesh_constant: float = 0.1,
    ):
        super(AdaptiveBookendedProfile, self).__init__(name=name)
        self.mesh_constant = mesh_constant
        self.energy = energy
        self._load_ooc(ooc, energy)
        self.total_thick = possibly_create_parameter(total_thick, name="total_thick")
        self.surface_roughness = possibly_create_parameter(
            surface_roughness, name="surface_roughness"
        )
        self.rho_bulk = possibly_create_parameter(rho_bulk, name="rho_bulk")
        self.rho_si = possibly_create_parameter(rho_si, name="rho_si")
        self.rho_vac = possibly_create_parameter(rho_vac, name="rho_vac")
        self.tau_si = possibly_create_parameter(tau_si, name="tau_si")
        self.tau_vac = possibly_create_parameter(tau_vac, name="tau_vac")
        self.alpha_bulk = possibly_create_parameter(alpha_bulk, name="alpha_bulk")
        self.alpha_si = possibly_create_parameter(alpha_si, name="alpha_si")
        self.alpha_vac = possibly_create_parameter(alpha_vac, name="alpha_vac")
        self.energy_offset = possibly_create_parameter(
            energy_offset, name="energy_offset"
        )
        self.num_slabs = num_slabs
        self._parameters = super().parameters
        self._parameters.extend(
            [
                self.total_thick,
                self.surface_roughness,
                self.rho_bulk,
                self.rho_si,
                self.rho_vac,
                self.tau_si,
                self.tau_vac,
                self.alpha_bulk,
                self.alpha_si,
                self.alpha_vac,
                self.energy_offset,
            ]
        )

    def _load_ooc(self, ooc: pd.DataFrame, energy: float):
        """Loac Optical Constants from a DataFrame."""

        # Validate the DataFrame
        required_columns = ["energy", "n_xx", "n_ixx", "n_zz", "n_izz"]
        if not all(col in ooc.columns for col in required_columns):
            missing = [col for col in required_columns if col not in ooc.columns]
            e = f"Optical constants dataframe missing required columns: {missing}"
            raise ValueError(e)
        cropped_tensor = slice_range(ooc, "energy", energy, 0.5)
        self.n_xx = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_xx"])
        self.n_ixx = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_ixx"]
        )
        self.n_zz = PchipInterpolator(cropped_tensor["energy"], cropped_tensor["n_zz"])
        self.n_izz = PchipInterpolator(
            cropped_tensor["energy"], cropped_tensor["n_izz"]
        )

    def varying_parameters(self):
        return [p for p in self._parameters if p.vary]

    def _generate_adaptive_grid(self) -> NDArray[np.float64]:
        total_thick_val = float(self.total_thick.value)
        num_slabs_val = self.num_slabs
        if num_slabs_val <= 1:
            return np.array([total_thick_val])
        n_half = num_slabs_val // 2
        half_thick = total_thick_val / 2.0
        r = self.mesh_constant ** (1 / n_half)
        if num_slabs_val % 2 == 0:
            a = half_thick * (r - 1) / (r**n_half - 1)
            mesh_half = a * r ** np.arange(n_half)
            mesh = np.concatenate([mesh_half[::-1], mesh_half])
        else:
            center_share = total_thick_val / num_slabs_val
            half_sum = (total_thick_val - center_share) / 2.0
            a = half_sum * (r - 1) / (r**n_half - 1)
            mesh_half = a * r ** np.arange(n_half)
            center = total_thick_val - 2 * mesh_half.sum()
            mesh = np.concatenate([mesh_half[::-1], [center], mesh_half])
        remainder = total_thick_val - mesh.sum()
        mesh[0] += remainder
        return mesh

    @property
    def slab_thick(self) -> NDArray[np.float64]:
        return self._generate_adaptive_grid()

    @property
    def dist(self) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        cumulative = np.cumsum(thicknesses)
        return cumulative

    @property
    def mid_points(self) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        cumulative = np.cumsum(thicknesses)
        mid_points = cumulative - thicknesses / 2
        return mid_points

    @property
    def parameters(self) -> Any:
        return self._parameters

    def density(
        self,
        depth: NDArray[np.float64] | float,
    ) -> float | NDArray[np.float64]:
        return density_profile_bookended(
            self.total_thick.value,
            depth,
            self.tau_si.value,
            self.tau_vac.value,
            self.rho_bulk.value,
            self.rho_si.value,
            self.rho_vac.value,
        )

    def orientation(
        self,
        depth: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        result_rad = orientation_profile_bookended(
            self.total_thick.value,
            depth,
            self.tau_si.value,
            self.tau_vac.value,
            self.alpha_bulk.value,
            self.alpha_si.value,
            self.alpha_vac.value,
        )
        return result_rad

    def average_orientation(self) -> float:
        return average_orientation_bookended(
            self.total_thick.value,
            self.tau_si.value,
            self.tau_vac.value,
            self.alpha_bulk.value,
            self.alpha_si.value,
            self.alpha_vac.value,
        )

    def get_energy(self) -> float:
        return float(self.energy) + float(self.energy_offset.value)

    def tensor(
        self,
        energy: float | None = None,
    ) -> NDArray[complexfloating[_64Bit, _64Bit]]:
        if energy is None:
            energy = self.get_energy()
        depth_arr = self.mid_points
        ori = self.orientation(depth_arr)
        rho_bulk_val = float(self.rho_bulk.value)
        rho_local = self.density(depth_arr)
        scale = np.asarray(rho_local, dtype=np.float64) / rho_bulk_val
        n_mol_xx, n_mol_zz = _scaled_molecular_components(
            self.n_xx,
            self.n_ixx,
            self.n_zz,
            self.n_izz,
            energy,
            rho_bulk_val,
        )
        n_o, n_e = _lab_tensor_diagonals(n_mol_xx, n_mol_zz, ori)
        return _assemble_diagonal_tensor(n_o * scale, n_e * scale)

    @property
    def iso(self) -> NDArray[np.float64]:
        return np.trace(self.tensor(), axis1=1, axis2=2)

    @property
    def delta(self) -> NDArray[np.float64]:
        return np.real(self.iso)

    @property
    def beta(self) -> NDArray[np.float64]:
        return np.imag(self.iso)

    def slabs(self, structure=None) -> NDArray[np.float64]:
        thicknesses = self.slab_thick
        tens = self.tensor()
        iso = np.trace(tens, axis1=1, axis2=2)
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = thicknesses
        slabs[..., 1] = np.real(iso)
        slabs[..., 2] = np.imag(iso)
        slabs[0, 3] = self.surface_roughness.value
        return slabs

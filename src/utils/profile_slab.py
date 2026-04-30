from typing import Any

import numpy as np
import pandas as pd
from numba import njit  # type: ignore
from numpy import complexfloating
from numpy._typing import _64Bit
from numpy._typing._array_like import NDArray
from pyref.fitting.structure import PXR_Component as Component
from pyref.fitting.structure import possibly_create_parameter, slice_range
from scipy.interpolate import PchipInterpolator


@njit(cache=True, fastmath=True)
def _orientation_profile_core(
    total_thick,
    depth,
    characteristic_thickness,
    max_angle,
    initial_angle,
):
    thick = depth  # total_thick - depth
    return max_angle * (1 - np.exp(-thick / characteristic_thickness)) + initial_angle


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
    decay_length_substrate,
    decay_length_vacuum,
    rho_bulk,
    rho_si,
    rho_vac,
):
    dist_from_surface = depth
    dist_from_substrate = total_thick - depth
    term_vac = (rho_vac - rho_bulk) * np.exp(-dist_from_surface / decay_length_vacuum)
    term_si = (rho_si - rho_bulk) * np.exp(
        -dist_from_substrate / decay_length_substrate
    )
    return rho_bulk + term_vac + term_si


def density_profile_bookended(
    total_thick: float,
    depth: float | NDArray[np.float64],
    decay_length_substrate: float,
    decay_length_vacuum: float,
    rho_bulk: float,
    rho_si: float,
    rho_vac: float,
) -> float | NDArray[np.float64]:
    """
    Book-ended density profile at surface and buried (substrate) interface.

    rho(z) = rho_bulk + (rho_vac - rho_bulk)*exp(-z/decay_length_vacuum)
             + (rho_si - rho_bulk)*exp(-(L-z)/decay_length_substrate)
    with z = depth from surface (0 = vacuum, L = substrate).

    Parameters
    ----------
    total_thick : float
        Film thickness L (same units as depth and decay lengths).
    depth : float or ndarray
        Distance from surface (0 = vacuum side).
    decay_length_substrate : float
        Substrate-side density decay length.
    decay_length_vacuum : float
        Vacuum-side density decay length.
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
            decay_length_substrate,
            decay_length_vacuum,
            rho_bulk,
            rho_si,
            rho_vac,
        )
    depth_arr = np.asarray(depth, dtype=np.float64)
    return _density_profile_bookended_core(
        total_thick,
        depth_arr,
        decay_length_substrate,
        decay_length_vacuum,
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
        t = self.total_thick.value - depth
        return orientation_profile(
            self.total_thick.value,
            t,
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

        n_xx = complex(
            self.n_xx(self.energy + self.energy_offset)
            + 1j * self.n_ixx(self.energy + self.energy_offset)
        )
        n_zz = complex(
            self.n_zz(self.energy + self.energy_offset)
            + 1j * self.n_izz(self.energy + self.energy_offset)
        )

        c = np.square(np.cos(ori))
        s = np.square(np.sin(ori))
        xx = (n_xx * (1 + c) + n_zz * s) / 2
        zz = n_xx * s + n_zz * c

        tensor = np.zeros((depth_arr.size, 3, 3), dtype=np.complex128)
        tensor[:, 0, 0] = xx
        tensor[:, 1, 1] = xx
        tensor[:, 2, 2] = zz
        return tensor

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
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = self.slab_thick
        slabs[..., 1] = self.delta
        slabs[..., 2] = self.beta
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
        cropped_tensor = slice_range(ooc, "energy", self.energy, 0.5)
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
        t = self.total_thick.value - depth
        return orientation_profile(
            self.total_thick.value,
            t,
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
        n_xx = complex(
            self.n_xx(self.energy + self.energy_offset)
            + 1j * self.n_ixx(self.energy + self.energy_offset)
        )
        n_zz = complex(
            self.n_zz(self.energy + self.energy_offset)
            + 1j * self.n_izz(self.energy + self.energy_offset)
        )

        c = np.square(np.cos(ori))
        s = np.square(np.sin(ori))
        xx = (n_xx * (1 + c) + n_zz * s) / 2
        zz = n_xx * s + n_zz * c

        tensor = np.zeros((depth_arr.size, 3, 3), dtype=np.complex128)
        tensor[:, 0, 0] = xx
        tensor[:, 1, 1] = xx
        tensor[:, 2, 2] = zz
        return tensor

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
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = thicknesses
        slabs[..., 1] = self.delta
        slabs[..., 2] = self.beta
        slabs[0, 3] = self.surface_roughness.value
        return slabs


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
        self.surface_roughness = possibly_create_parameter(0, name="surface_roughness")
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
        cropped_tensor = slice_range(ooc, "energy", self.energy, 0.5)
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
        n_xx = complex(
            self.n_xx(self.energy + self.energy_offset)
            + 1j * self.n_ixx(self.energy + self.energy_offset)
        )
        n_zz = complex(
            self.n_zz(self.energy + self.energy_offset)
            + 1j * self.n_izz(self.energy + self.energy_offset)
        )
        c = np.square(np.cos(ori))
        s = np.square(np.sin(ori))
        xx = (n_xx * (1 + c) + n_zz * s) / 2
        zz = n_xx * s + n_zz * c
        tensor = np.zeros((depth_arr.size, 3, 3), dtype=np.complex128)
        tensor[:, 0, 0] = xx
        tensor[:, 1, 1] = xx
        tensor[:, 2, 2] = zz
        return tensor

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
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = thicknesses
        slabs[..., 1] = self.delta
        slabs[..., 2] = self.beta
        slabs[0, 3] = self.surface_roughness.value
        return slabs


class AdaptiveBookendedProfile(Component):
    """
    Adaptive slab profile with book-ended orientation and density.

    Orientation and density share the same functional form: bulk value plus
    exponential decays from vacuum and substrate interfaces. Orientation uses
    tau_si, tau_vac (angle decay lengths). Density uses decay_length_substrate,
    decay_length_vacuum and rho_bulk, rho_si, rho_vac. Optical response is
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
        decay_length_substrate: float,
        decay_length_vacuum: float,
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
        self.surface_roughness = possibly_create_parameter(0, name="surface_roughness")
        self.rho_bulk = possibly_create_parameter(rho_bulk, name="rho_bulk")
        self.rho_si = possibly_create_parameter(rho_si, name="rho_si")
        self.rho_vac = possibly_create_parameter(rho_vac, name="rho_vac")
        self.decay_length_substrate = possibly_create_parameter(
            decay_length_substrate, name="decay_length_substrate"
        )
        self.decay_length_vacuum = possibly_create_parameter(
            decay_length_vacuum, name="decay_length_vacuum"
        )
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
                self.decay_length_substrate,
                self.decay_length_vacuum,
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
        cropped_tensor = slice_range(ooc, "energy", self.energy, 0.5)
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
            self.decay_length_substrate.value,
            self.decay_length_vacuum.value,
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
        n_xx = complex(
            self.n_xx(self.energy + self.energy_offset)
            + 1j * self.n_ixx(self.energy + self.energy_offset)
        )
        n_zz = complex(
            self.n_zz(self.energy + self.energy_offset)
            + 1j * self.n_izz(self.energy + self.energy_offset)
        )
        c = np.square(np.cos(ori))
        s = np.square(np.sin(ori))
        xx = (n_xx * (1 + c) + n_zz * s) / 2
        zz = n_xx * s + n_zz * c
        tensor = np.zeros((depth_arr.size, 3, 3), dtype=np.complex128)
        tensor[:, 0, 0] = xx * scale
        tensor[:, 1, 1] = xx * scale
        tensor[:, 2, 2] = zz * scale
        return tensor

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
        slabs = np.zeros((self.num_slabs, 4))
        slabs[..., 0] = thicknesses
        slabs[..., 1] = self.delta
        slabs[..., 2] = self.beta
        slabs[0, 3] = self.surface_roughness.value
        return slabs

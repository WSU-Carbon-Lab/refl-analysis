"""Notebook helpers: book-ended film models via refloxide energy + fused Rust path.

After ``patch_pyref()``, :class:`~refloxide.pxr.energy.bookended.EnergyBookendedOrientationDensityProfile`
stacks use the fused evaluator automatically (no Python ``slabs``/``tensor`` rebuild).
Set ``parallel=False`` on the patch when refnx or emcee already parallelizes walkers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from refloxide.pxr.energy.bookended import (
    EnergyBookendedOrientationDensityProfile,
    bookended_from_three_slabs,
)

if TYPE_CHECKING:
    import pandas as pd
    from refloxide.pxr.energy.ooc import OocAnchor


def model_bookended_refloxide(
    template_objective,
    ooc: pd.DataFrame | OocAnchor,
    energy: float,
    *,
    num_slabs: int = 24,
    mesh_constant: float = 0.1,
    interp: str = "linear",
):
    """Replace surf+bulk+inter UniTensor slabs with one Rust-backed book-ended profile.

    Parameters
    ----------
    template_objective
        A refnx/pyref objective whose ``model.structure`` has vacuum at index 0,
        three ZnPc ``UniTensorSLD`` slabs at 1-3, then substrate layers 4-5.
    ooc
        Optical constant table (full energy range; not windowed).
    energy
        Photon energy in eV for this single-energy fit.
    num_slabs
        Microslab count for the adaptive grid. Use 20-32 for fitting; more for plots.
    mesh_constant, interp
        Mesh refinement and OOC interpolation (``linear`` uses Rust).

    Returns
    -------
    Structure
        New stack: vacuum | book-ended ZnPc | SiO2 | Si (components copied from template).
    """
    structure = template_objective.model.structure
    film = bookended_from_three_slabs(
        structure.components[1],
        structure.components[2],
        structure.components[3],
        ooc,
        energy=float(energy),
        energy_offset=float(template_objective.model.energy_offset.value or 0.0),
        num_slabs=num_slabs,
        mesh_constant=mesh_constant,
        name=f"ZnPc_{float(energy):.1f}",
        interp=interp,
    )
    return (
        structure.components[0]
        | film
        | structure.components[4]
        | structure.components[5]
    )


__all__ = [
    "EnergyBookendedOrientationDensityProfile",
    "bookended_from_three_slabs",
    "model_bookended_refloxide",
]

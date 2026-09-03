"""Pyref reflectivity models backed by refloxide uniaxial kernels.

Import :class:`ReflectModel` and related pyref fitting symbols from this module
instead of ``pyref.fitting`` when fits should evaluate through refloxide's
lab-frame s-in/p-in TJF 4x4 paxth (Rust by default, ``parallel=False`` for
MCMC and refnx worker pools).

Examples
--------
>>> from utils.models import ReflectModel, configure_refloxide_fitting
>>> configure_refloxide_fitting()  # no-op after first call
>>> model = ReflectModel(structure, energy=283.7, pol="sp")
"""

from __future__ import annotations

from utils.pyref_patch import (
    patch_pyref,
    pyref_patched,
    require_pyref_patched,
    uniaxial_reflectivity,
)
from utils.pyref_patch import (
    reflectivity as refloxide_reflectivity,
)

_PATCHED = False


def configure_refloxide_fitting(
    *,
    use_rust: bool = True,
    parallel: bool = False,
    patch_reflectivity: bool = True,
    force: bool = False,
) -> None:
    """Route ``pyref.fitting`` uniaxial reflectivity through refloxide.

    Replaces ``pyref.fitting.uniaxial.uniaxial_reflectivity`` and, by default,
    ``pyref.fitting.model.reflectivity`` so :class:`ReflectModel` and refnx
    objectives use the refloxide kernel without further changes.

    Parameters
    ----------
    use_rust
        When ``True`` (default), call :func:`refloxide.rust.uniaxial_reflectivity`.
    parallel
        Forwarded to the Rust kernel. Default ``False`` to avoid nested Rayon
        when emcee or refnx pools already parallelize walkers.
    patch_reflectivity
        When ``True`` (default), also patch smearing/scaling in
        ``pyref.fitting.model.reflectivity``.
    force
        When ``True``, re-apply the patch even if this module already configured
        pyref in the current process.

    Notes
    -----
    The patch is process-global. Call once before constructing
    :class:`ReflectModel` instances, or rely on the automatic configuration
    performed at import time.
    """
    global _PATCHED
    if _PATCHED and not force:
        return
    patch_pyref(
        use_rust=use_rust,
        parallel=parallel,
        patch_reflectivity=patch_reflectivity,
    )
    _PATCHED = True


configure_refloxide_fitting()

from pyref.fitting import (
    AnisotropyObjective,
    CurveFitter,
    GlobalObjective,
    Objective,
    ReflectModel,
    Transform,
)

require_pyref_patched()

__all__ = [
    "AnisotropyObjective",
    "CurveFitter",
    "GlobalObjective",
    "Objective",
    "ReflectModel",
    "Transform",
    "configure_refloxide_fitting",
    "pyref_patched",
    "refloxide_reflectivity",
    "require_pyref_patched",
    "uniaxial_reflectivity",
]

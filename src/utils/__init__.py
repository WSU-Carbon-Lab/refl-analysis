"""Shared helpers for the ZnPc reflectivity analysis.

Note: refloxide is the intended library for reflectivity modeling and fitting
(``refloxide.model`` / ``refloxide.objective``; see ``demo/run_demo.py``). The
helpers in this package that wrap ``pyref.fitting`` (``models``, ``graded_objective``,
``slab_builders``, ``profile_slab``, ``io.read_fit`` for the original pickled fits)
are legacy and need the optional ``pyref`` dependency: ``uv sync --extra legacy``.
"""

try:
    import pyref  # noqa: F401
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "The `utils` package still wraps legacy `pyref.fitting` code. "
        "Install it with `uv sync --extra legacy` (refloxide is the intended interface; "
        "see demo/run_demo.py for a pyref-free example)."
    ) from err

from utils.graded_objective import (
    BookendedMultiEnergyPolish,
    graded_objective_at_energy,
    load_bookended_multienergy_logY_polish,
    upsample_graded_objective,
)
from utils.helpers.io import (
    data_external,
    data_interim,
    data_processed,
    data_raw,
    data_root,
    docs_root,
    hf_artifacts_config_path,
    models_checkpoints,
    models_fitting_results,
    models_optical,
    models_root,
    notebooks_data_collection,
    notebooks_dft,
    notebooks_exploration,
    notebooks_fitting,
    notebooks_giwaxs,
    notebooks_manuscript,
    notebooks_optical_models,
    notebooks_photoresist,
    notebooks_root,
    project_root,
    read_fit,
    read_ooc,
    read_xrr,
    references_root,
    reports_root,
    resolve_artifact_directory,
    resolve_hf_mapping,
    resolve_special_path,
    scripts_root,
    src_root,
)
from utils.objective_energy import (
    broadcast_structure_probe_energy,
    objective_retarget_energy,
    objectives_retarget_energies,
)
from utils.refloxide_fitting import (
    KernelComparisonResult,
    batched_logl_with_kernel,
    build_dft_diagnostic_mcmc_bundle,
    build_dft_diagnostic_pyref_mcmc_objective,
    compare_kernel_paths,
    convert_dft_fit_bundle,
)
from utils.slab_builders import (
    DEFAULT_FIXED_PARAMS,
    constrain_to_template,
    resolve_param,
    safely_setp,
    safely_setp_param,
    select,
    slab_from_template,
)

__all__ = [
    "DEFAULT_FIXED_PARAMS",
    "constrain_to_template",
    "broadcast_structure_probe_energy",
    "KernelComparisonResult",
    "batched_logl_with_kernel",
    "build_dft_diagnostic_mcmc_bundle",
    "build_dft_diagnostic_pyref_mcmc_objective",
    "compare_kernel_paths",
    "convert_dft_fit_bundle",
    "BookendedMultiEnergyPolish",
    "graded_objective_at_energy",
    "load_bookended_multienergy_logY_polish",
    "objective_retarget_energy",
    "objectives_retarget_energies",
    "upsample_graded_objective",
    "data_external",
    "data_interim",
    "data_processed",
    "data_raw",
    "data_root",
    "docs_root",
    "hf_artifacts_config_path",
    "models_checkpoints",
    "models_fitting_results",
    "models_optical",
    "models_root",
    "notebooks_data_collection",
    "notebooks_dft",
    "notebooks_exploration",
    "notebooks_fitting",
    "notebooks_giwaxs",
    "notebooks_manuscript",
    "notebooks_optical_models",
    "notebooks_photoresist",
    "notebooks_root",
    "project_root",
    "read_fit",
    "read_ooc",
    "read_xrr",
    "references_root",
    "reports_root",
    "resolve_artifact_directory",
    "resolve_hf_mapping",
    "resolve_param",
    "resolve_special_path",
    "safely_setp",
    "safely_setp_param",
    "scripts_root",
    "select",
    "slab_from_template",
    "src_root",
]

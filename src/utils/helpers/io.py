import os
import tomllib
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl
import pyref.fitting as fit
from huggingface_hub import snapshot_download

os.environ["POLARS_VERBOSE"] = "0"
os.environ["RUST_LOG"] = "error"

project_root = Path(__file__).parent.parent.parent.parent.resolve()
hf_artifacts_config_path = project_root / "configs" / "hf-artifacts.toml"

data_root = project_root / "@data"
data_raw = data_root / "raw"
data_interim = data_root / "interim"
data_processed = data_root / "processed"
data_external = data_root / "external"

notebooks_root = project_root / "notebooks"
notebooks_data_collection = notebooks_root / "data_collection"
notebooks_dft = notebooks_root / "dft"
notebooks_exploration = notebooks_root / "exploration"
notebooks_fitting = notebooks_root / "fitting"
notebooks_giwaxs = notebooks_root / "giwaxs"
notebooks_manuscript = notebooks_root / "manuscript"
notebooks_optical_models = notebooks_root / "optical_models"
notebooks_photoresist = notebooks_root / "photoresist"

models_root = project_root / "@models"
models_fitting_results = models_root / "fitting_results"
models_checkpoints = models_root / "checkpoints"
models_optical = models_root / "optical"

reports_root = project_root / "reports"
scripts_root = project_root / "scripts"
references_root = project_root / "references"
docs_root = project_root / "docs"
src_root = project_root / "src"


def _load_hf_artifact_config(config_path: Path = hf_artifacts_config_path) -> dict:
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_special_path(value: str | Path) -> Path:
    path = Path(value)
    path_str = path.as_posix()
    if path_str.startswith("@data/") or path_str == "@data":
        suffix = path_str.removeprefix("@data").lstrip("/")
        return (data_root / suffix).resolve()
    if path_str.startswith("@models/") or path_str == "@models":
        suffix = path_str.removeprefix("@models").lstrip("/")
        return (models_root / suffix).resolve()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def resolve_hf_mapping(
    kind: Literal["data", "models"],
    group: str,
    material: str,
    config_path: Path = hf_artifacts_config_path,
) -> dict[str, str]:
    config = _load_hf_artifact_config(config_path)
    rows = config.get(kind, [])
    group_key = "experiment_type" if kind == "data" else "model_type"
    for row in rows:
        if row[group_key] == group and row["material"] == material:
            return {
                "repo_id": row["repo_id"],
                "repo_type": row["repo_type"],
                "revision": row.get("revision", "main"),
                "local_path": str(resolve_special_path(row["local_path"])),
            }
    raise ValueError(
        "No Hugging Face mapping found for "
        f"kind={kind}, group={group}, material={material}"
    )


def resolve_artifact_directory(
    kind: Literal["data", "models"],
    group: str,
    material: str,
    source: Literal["local", "hub", "auto"] = "auto",
    config_path: Path = hf_artifacts_config_path,
) -> Path:
    mapping = resolve_hf_mapping(
        kind=kind, group=group, material=material, config_path=config_path
    )
    local_path = Path(mapping["local_path"])
    if source == "local":
        if not local_path.exists():
            raise FileNotFoundError(f"Local artifact directory not found: {local_path}")
        return local_path
    if source == "hub" or (source == "auto" and not local_path.exists()):
        local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=mapping["repo_id"],
            repo_type=mapping["repo_type"],
            revision=mapping["revision"],
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )
    if not local_path.exists():
        raise FileNotFoundError(
            f"Artifact directory not found after resolution: {local_path}"
        )
    return local_path


def read_xrr(  # noqa: PLR0912, PLR0915
    filename: str | Path | None = None,
    material: str | None = None,
    data_dir: Path | None = None,
    source: Literal["local", "hub", "auto"] = "auto",
) -> dict[str, fit.XrayReflectDataset]:
    """
    Load reflectivity dataset from a parquet file.

    Parameters
    ----------
    filename : str or Path, optional
        Input filename (without extension). If None, uses "refl"
    material : str, optional
        Material name (e.g., "znpc"). If None, uses data_processed / "fitting"
    data_dir : Path, optional
        Directory to search for data. If None and material is provided,
        uses data_root / "xrr" / material

    Returns
    -------
    dict[str, XrayReflectDataset]
        Dictionary containing XrayReflectDataset objects with energy keys

    Examples
    --------
    >>> data = read_xrr(material="znpc")
    >>> data = read_xrr("refl", material="znpc")
    >>> data = read_xrr("my_data", material="znpc")
    >>> data = read_xrr("my_data.parquet", data_dir=Path("/custom/path"))
    """
    if material is not None:
        if data_dir is None:
            data_dir = resolve_artifact_directory(
                kind="data",
                group="xrr",
                material=material,
                source=source,
            )
        if filename is None:
            filename = "refl"
        filename_path = Path(filename)
        if filename_path.suffix != ".parquet":
            filename = data_dir / f"{filename_path}.parquet"
        else:
            filename = data_dir / filename_path
    else:
        if data_dir is None:
            data_dir = data_processed / "fitting"
        if filename is None:
            filename = "reflectivity_data"
        filename_path = Path(filename)
        if filename_path.suffix != ".parquet":
            filename = data_dir / f"{filename_path}.parquet"
        else:
            filename = data_dir / filename_path

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Reflectivity data file not found: {filename}")

    old_verbose = os.environ.get("POLARS_VERBOSE")
    try:
        os.environ["POLARS_VERBOSE"] = "0"
        df_load = pl.read_parquet(filename)
    finally:
        if old_verbose is not None:
            os.environ["POLARS_VERBOSE"] = old_verbose
        elif "POLARS_VERBOSE" in os.environ:
            del os.environ["POLARS_VERBOSE"]

    columns = df_load.columns

    def find_column(possible_names: list[str], column_type: str) -> str:
        for col in possible_names:
            if col in columns:
                return col
        raise ValueError(
            f"Could not find {column_type} column. Available columns: {columns}"
        )

    energy_col = find_column(
        ["energy", "Beamline Energy [eV]", "Energy", "energy [eV]"], "energy"
    )
    q_col = find_column(["Q", "Q [Å⁻¹]", "q", "Q [A^-1]"], "Q")
    r_col = find_column(["R", "r", "r [a. u.]", "reflectivity"], "R")
    dr_col = find_column(["dR", "dr", "δr [a. u.]", "dR [a. u.]", "error"], "dR")

    data_reconstructed: dict[str, fit.XrayReflectDataset] = {}

    for group_key, group_data in df_load.group_by(energy_col):
        energy_val = group_key[0]
        diff = group_data[r_col] - group_data[dr_col]
        mask = diff > 0
        group_data = group_data.filter(mask)

        Q = group_data[q_col].to_numpy()  # noqa: N806
        R = group_data[r_col].to_numpy()  # noqa: N806
        dR = group_data[dr_col].to_numpy()  # noqa: N806

        dataset = fit.XrayReflectDataset(data=(Q, R, dR))
        data_reconstructed[str(energy_val)] = dataset

    return data_reconstructed


def read_ooc(
    filename: str | Path | None = None,
    material: str | None = None,
    data_dir: Path | None = None,
    source: Literal["local", "hub", "auto"] = "auto",
) -> pd.DataFrame:
    """
    Load optical constants from a CSV file.

    Parameters
    ----------
    filename : str or Path, optional
        Input filename. If None and material is provided,
        constructs path as models_optical / material / "dft.csv"
    material : str, optional
        Material name (e.g., "znpc"). If provided, constructs default path.
    data_dir : Path, optional
        Directory to search for optical constants. If None, uses models_optical

    Returns
    -------
    pd.DataFrame
        DataFrame containing optical constants with columns:
        energy, n_xx, n_zz, n_ixx, n_izz

    Examples
    --------
    >>> ooc = read_ooc(material="znpc")
    >>> ooc = read_ooc("dft_beta.csv", material="znpc")
    >>> ooc = read_ooc("custom_ooc.csv", data_dir=Path("/custom/path"))
    """
    if data_dir is None:
        data_dir = models_optical

    if filename is None:
        if material is None:
            raise ValueError("Either filename or material must be provided")
        material_dir = resolve_artifact_directory(
            kind="models",
            group="optical",
            material=material,
            source=source,
        )
        filename = material_dir / "dft.csv"
    elif material is not None:
        filename = data_dir / material / filename
    else:
        filename = Path(filename)
        if not filename.is_absolute():
            filename = data_dir / filename

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Optical constants file not found: {filename}")

    ooc = pd.read_csv(filename)

    required_columns = {"energy", "n_xx", "n_zz", "n_ixx", "n_izz"}
    if not required_columns.issubset(ooc.columns):
        missing = required_columns - set(ooc.columns)
        raise ValueError(f"Optical constants file missing required columns: {missing}")

    return ooc


def read_fit(
    filename: str | Path | None = None,
    data_dir: Path | None = None,
) -> fit.GlobalObjective:
    """
    Load a GlobalObjective from a pickle file.

    Parameters
    ----------
    filename : str or Path, optional
        Input filename. If None, raises ValueError
    data_dir : Path, optional
        Directory to search for fitting results. If None, uses models_fitting_results

    Returns
    -------
    GlobalObjective
        Loaded GlobalObjective object

    Examples
    --------
    >>> obj = read_fit("dft_en_offset_best.pkl")
    >>> obj = read_fit("my_fit.pkl", data_dir=Path("/custom/path"))
    """
    if filename is None:
        raise ValueError("filename must be provided")

    if data_dir is None:
        data_dir = models_fitting_results

    filename = Path(filename)
    if not filename.is_absolute():
        filename = data_dir / filename

    if not filename.exists():
        raise FileNotFoundError(f"Fitting results file not found: {filename}")

    import pickle

    with filename.open("rb") as f:
        result = pickle.load(f)

    if not isinstance(result, fit.GlobalObjective):
        raise TypeError(f"Loaded object is not a GlobalObjective, got {type(result)}")

    return result

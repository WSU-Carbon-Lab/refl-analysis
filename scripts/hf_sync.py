"""Synchronize local data/model folders with Hugging Face repositories."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

VALID_REPO_TYPES = {"dataset", "model", "space"}
DATA_ROOT = "@data"
MODELS_ROOT = "@models"
PUSH_IGNORE_PATTERNS = (
    ".cache/**",
    "**/.cache/**",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
)


@dataclass(frozen=True)
class Mapping:
    kind: str
    local_path: Path
    repo_id: str
    repo_type: str
    revision: str
    group: str
    material: str


def load_config(config_path: Path) -> dict:
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def mappings_from_config(config: dict, root: Path) -> list[Mapping]:
    items: list[Mapping] = []

    for row in config.get("data", []):
        items.append(
            Mapping(
                kind="data",
                local_path=root / row["local_path"],
                repo_id=row["repo_id"],
                repo_type=row["repo_type"],
                revision=row.get("revision", "main"),
                group=row["experiment_type"],
                material=row["material"],
            )
        )

    for row in config.get("models", []):
        items.append(
            Mapping(
                kind="models",
                local_path=root / row["local_path"],
                repo_id=row["repo_id"],
                repo_type=row["repo_type"],
                revision=row.get("revision", "main"),
                group=row["model_type"],
                material=row["material"],
            )
        )

    return items


def expected_repo_name(group: str, material: str) -> str:
    return f"{group}-{material}".lower()


def validate_mapping(mapping: Mapping) -> list[str]:
    failures: list[str] = []
    local_parts = mapping.local_path.parts
    if len(local_parts) < 3:
        failures.append(f"{mapping.local_path}: path must be <root>/<group>/<material>")
        return failures

    expected_local = (local_parts[-3], local_parts[-2], local_parts[-1])
    if mapping.kind == "data" and expected_local[0] != DATA_ROOT:
        failures.append(
            f"{mapping.local_path}: data mapping must live under {DATA_ROOT}/"
        )
    if mapping.kind == "models" and expected_local[0] != MODELS_ROOT:
        failures.append(
            f"{mapping.local_path}: models mapping must live under {MODELS_ROOT}/"
        )

    if expected_local[1] != mapping.group:
        failures.append(
            f"{mapping.local_path}: folder group '{expected_local[1]}' "
            f"does not match config '{mapping.group}'"
        )
    if expected_local[2] != mapping.material:
        failures.append(
            f"{mapping.local_path}: folder material '{expected_local[2]}' "
            f"does not match config '{mapping.material}'"
        )

    if mapping.repo_type not in VALID_REPO_TYPES:
        failures.append(f"{mapping.repo_id}: invalid repo_type '{mapping.repo_type}'")

    repo_suffix = mapping.repo_id.split("/", maxsplit=1)[-1]
    expected_suffix = expected_repo_name(mapping.group, mapping.material)
    if repo_suffix != expected_suffix:
        failures.append(
            f"{mapping.repo_id}: expected repo suffix '{expected_suffix}' "
            f"for {mapping.local_path}"
        )

    return failures


def resolve_target_path(target: str, root: Path) -> Path:
    """Resolve a CLI sync target to an absolute local artifact path.

    Parameters
    ----------
    target : str
        Path from ``--target``. Accepts absolute paths, repo-relative
        ``@data/...`` / ``@models/...`` forms from ``hf-artifacts.toml``,
        and bare ``data/...`` / ``models/...`` aliases used in some READMEs.
    root : Path
        Project root used to anchor relative targets.

    Returns
    -------
    Path
        Absolute resolved path to the local artifact directory.
    """
    target_path = Path(target)
    if target_path.is_absolute():
        return target_path.resolve()

    parts = target_path.parts
    if parts and parts[0] in {"data", "models"}:
        target_path = Path(f"@{target_path.as_posix()}")

    return (root / target_path).resolve()


def select_mappings(
    all_mappings: list[Mapping],
    target: str | None,
    all_flag: bool,
    *,
    root: Path,
) -> list[Mapping]:
    """Select artifact mappings for ``--all`` or a single ``--target``.

    Parameters
    ----------
    all_mappings : list of Mapping
        Mappings loaded from the artifact config.
    target : str or None
        Local path passed to ``--target``; ignored when ``all_flag`` is True.
    all_flag : bool
        When True, return every mapping.
    root : Path
        Project root used to resolve relative targets against absolute
        ``Mapping.local_path`` values.

    Returns
    -------
    list of Mapping
        Selected mappings.

    Raises
    ------
    ValueError
        If neither ``--all`` nor ``--target`` is provided, or if ``target``
        does not match any mapping.
    """
    if all_flag:
        return all_mappings
    if target is None:
        raise ValueError("provide --all or --target")

    target_resolved = resolve_target_path(target, root)
    resolved = [m for m in all_mappings if m.local_path.resolve() == target_resolved]
    if not resolved:
        raise ValueError(f"no mapping found for target: {target}")
    return resolved


def command_plan(mappings: list[Mapping]) -> None:
    payload = [
        {
            "kind": m.kind,
            "local_path": m.local_path.as_posix(),
            "repo_id": m.repo_id,
            "repo_type": m.repo_type,
            "revision": m.revision,
        }
        for m in mappings
    ]
    print(json.dumps(payload, indent=2))


def command_validate(mappings: list[Mapping]) -> None:
    failures: list[str] = []
    for mapping in mappings:
        failures.extend(validate_mapping(mapping))
    if failures:
        for failure in failures:
            print(f"INVALID: {failure}")
        raise SystemExit(1)
    print(f"Validated {len(mappings)} mappings.")


def command_pull(mappings: list[Mapping], dry_run: bool) -> None:
    for mapping in mappings:
        if dry_run:
            print(
                "DRY_RUN: snapshot_download "
                f"repo_id={mapping.repo_id} "
                f"repo_type={mapping.repo_type} "
                f"revision={mapping.revision} "
                f"local_dir={mapping.local_path}"
            )
            continue
        mapping.local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=mapping.repo_id,
            repo_type=mapping.repo_type,
            revision=mapping.revision,
            local_dir=str(mapping.local_path),
        )


def command_push(mappings: list[Mapping], dry_run: bool) -> None:
    """Upload each mapped local folder to its Hub repo via ``upload_folder``.

    Uses the same ``huggingface_hub`` path as ``hf upload``: multi-commit
    streaming for large trees, skip-already-uploaded resume, and repo-root
    placement. Local Hub upload caches under ``.cache/`` are excluded.
    """
    api = HfApi()
    for mapping in mappings:
        if dry_run:
            print(
                "DRY_RUN: upload_folder "
                f"repo_id={mapping.repo_id} "
                f"folder_path={mapping.local_path} "
                f"repo_type={mapping.repo_type} "
                f"revision={mapping.revision}"
            )
            continue
        if not mapping.local_path.is_dir():
            raise FileNotFoundError(
                f"Local artifact directory not found: {mapping.local_path}"
            )
        api.create_repo(
            repo_id=mapping.repo_id,
            repo_type=mapping.repo_type,
            exist_ok=True,
        )
        if mapping.revision != "main":
            api.create_branch(
                repo_id=mapping.repo_id,
                repo_type=mapping.repo_type,
                branch=mapping.revision,
                exist_ok=True,
            )
        commit = api.upload_folder(
            repo_id=mapping.repo_id,
            folder_path=mapping.local_path,
            path_in_repo=".",
            repo_type=mapping.repo_type,
            revision=mapping.revision,
            ignore_patterns=list(PUSH_IGNORE_PATTERNS),
            commit_message=f"Sync {mapping.kind}/{mapping.group}/{mapping.material}",
        )
        print(f"{mapping.repo_id}: uploaded ({commit.commit_url})")


def command_check_remote(mappings: list[Mapping], dry_run: bool) -> None:
    api = HfApi()
    for mapping in mappings:
        if dry_run:
            print(
                "DRY_RUN: list_repo_files "
                f"repo_id={mapping.repo_id} "
                f"repo_type={mapping.repo_type} "
                f"revision={mapping.revision}"
            )
            continue
        files = api.list_repo_files(
            repo_id=mapping.repo_id,
            repo_type=mapping.repo_type,
            revision=mapping.revision,
        )
        print(f"{mapping.repo_id}: {len(files)} files found on {mapping.revision}")


def parser() -> argparse.ArgumentParser:
    argp = argparse.ArgumentParser(
        description="Sync local artifact folders with Hugging Face."
    )
    argp.add_argument(
        "--config",
        default="configs/hf-artifacts.toml",
        help="Path to artifact mapping config",
    )
    sub = argp.add_subparsers(dest="command", required=True)

    sub.add_parser("plan")
    sub.add_parser("validate")
    check_remote = sub.add_parser("check-remote")
    check_remote.add_argument("--all", action="store_true")
    check_remote.add_argument("--target")
    check_remote.add_argument("--dry-run", action="store_true")

    for name in ("pull", "push"):
        action = sub.add_parser(name)
        action.add_argument("--all", action="store_true")
        action.add_argument("--target")
        action.add_argument("--dry-run", action="store_true")

    return argp


def main() -> None:
    args = parser().parse_args()
    root = Path.cwd()
    config = load_config(root / args.config)
    all_mappings = mappings_from_config(config, root)

    if args.command == "plan":
        command_plan(all_mappings)
        return
    if args.command == "validate":
        command_validate(all_mappings)
        return

    selected = select_mappings(all_mappings, args.target, args.all, root=root)
    command_validate(selected)
    if args.command == "check-remote":
        command_check_remote(selected, args.dry_run)
        return
    if args.command == "pull":
        command_pull(selected, args.dry_run)
        return
    if args.command == "push":
        command_push(selected, args.dry_run)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

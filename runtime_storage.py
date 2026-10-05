"""Safe helpers for local runtime state.

Runtime state is intentionally kept outside the tracked source files. JSON files
are replaced atomically and retain one last-known-good backup so a power loss or
partial write does not silently erase bot memory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str, *, keep_backup: bool = True) -> None:
    """Durably replace *path* and optionally preserve its previous contents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    backup_path = path.with_suffix(path.suffix + ".bak")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as temporary_file:
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if keep_backup and path.exists():
            shutil.copy2(path, backup_path)
        temporary_path.replace(path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.debug("Could not remove temporary file %s", temporary_path, exc_info=True)
        raise


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object, recovering from the last backup when possible."""

    candidates = (path, path.with_suffix(path.suffix + ".bak"))
    primary_error: Exception | None = None
    for index, candidate in enumerate(candidates):
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if index == 0:
                primary_error = FileNotFoundError(path)
            continue
        except (OSError, json.JSONDecodeError) as error:
            if index == 0:
                primary_error = error
            continue
        if isinstance(value, dict):
            if index == 1:
                LOGGER.warning("Recovered invalid runtime state %s from %s", path, candidate)
            return value
        if index == 0:
            primary_error = ValueError(f"JSON root in {path} is not an object")

    if primary_error is not None:
        raise primary_error
    raise FileNotFoundError(path)


def migrate_legacy_runtime_files(
    project_directory: Path,
    targets: Iterable[Path],
) -> list[tuple[Path, Path]]:
    """Copy legacy root-level state into new runtime paths on first start."""

    migrated: list[tuple[Path, Path]] = []
    for target in targets:
        source = project_directory / target.name
        if target == source or target.exists() or not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_target = target.with_suffix(target.suffix + ".migration.tmp")
        shutil.copy2(source, temporary_target)
        temporary_target.replace(target)
        migrated.append((source, target))
        LOGGER.info("Migrated legacy runtime state %s to %s", source, target)
    return migrated

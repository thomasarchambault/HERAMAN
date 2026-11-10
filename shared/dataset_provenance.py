# Rôle : Normaliser et migrer les identifiants de cubes afin de préserver la provenance des données.

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def cube_id_from_path(path: str | Path) -> str:
    """Build a stable acquisition identifier that does not depend on its folder."""
    stem = Path(str(path)).stem.strip()
    for suffix in ("_corrected_cube", "_cube"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "cube"


def cube_ids_from_dataset(dataset: Any) -> np.ndarray:
    """Read canonical cube IDs, deriving them from legacy paths when required."""
    files = set(getattr(dataset, "files", []))
    if "cube_ids" in files:
        ids = [str(value) for value in np.asarray(dataset["cube_ids"], dtype=object).tolist()]
        if ids:
            return np.asarray(ids, dtype=object)

    if "cube_paths" in files:
        paths = np.asarray(dataset["cube_paths"], dtype=object).tolist()
        if paths:
            return np.asarray([cube_id_from_path(value) for value in paths], dtype=object)

    if "sample_cube_index" in files:
        indices = np.asarray(dataset["sample_cube_index"], dtype=np.int32)
        valid = indices[indices >= 0]
        count = int(valid.max()) + 1 if valid.size else 0
        return np.asarray([f"cube_{idx}" for idx in range(count)], dtype=object)

    return np.empty((0,), dtype=object)


def cube_paths_from_dataset(dataset: Any, cube_count: int) -> np.ndarray:
    """Return optional paths aligned with cube_ids; missing paths stay empty."""
    files = set(getattr(dataset, "files", []))
    if "cube_paths" in files:
        values = [str(value) for value in np.asarray(dataset["cube_paths"], dtype=object).tolist()]
    else:
        values = []
    values = (values + [""] * int(cube_count))[: int(cube_count)]
    return np.asarray(values, dtype=object)


def sample_cube_ids(
    sample_cube_index: Sequence[int],
    cube_ids: Sequence[str],
) -> np.ndarray:
    """Expand the compact per-sample cube index into stable group labels."""
    indices = np.asarray(sample_cube_index, dtype=np.int32).reshape(-1)
    ids = np.asarray(cube_ids, dtype=object).reshape(-1)
    groups = np.empty(indices.shape[0], dtype=object)
    for row, cube_index in enumerate(indices):
        if 0 <= int(cube_index) < ids.size:
            groups[row] = str(ids[int(cube_index)])
        else:
            groups[row] = f"unknown_cube_{int(cube_index)}"
    return groups


def migrate_dataset_cube_ids(npz_path: str | Path, *, create_backup: bool = True) -> bool:
    """Persist cube_ids in a legacy NPZ while preserving cube_paths as optional metadata."""
    npz_path = Path(npz_path).resolve()
    with np.load(str(npz_path), allow_pickle=True) as dataset:
        payload = {key: dataset[key] for key in dataset.files}
        derived_ids = cube_ids_from_dataset(dataset)

    existing = payload.get("cube_ids")
    if existing is not None and np.array_equal(
        np.asarray(existing, dtype=object),
        np.asarray(derived_ids, dtype=object),
    ):
        return False
    if derived_ids.size == 0:
        raise ValueError(f"Cannot derive cube IDs from dataset: {npz_path}")

    if create_backup:
        backup_path = npz_path.with_name(f"{npz_path.stem}_backup_before_cube_ids.npz")
        if not backup_path.exists():
            shutil.copy2(npz_path, backup_path)

    payload["cube_ids"] = np.asarray(derived_ids, dtype=object)
    temporary_path = npz_path.with_name(f"{npz_path.stem}_cube_ids_tmp.npz")
    np.savez_compressed(str(temporary_path), **payload)
    temporary_path.replace(npz_path)
    return True

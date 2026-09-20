"""Append-only train/validation membership for cumulative active learning."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Sequence


def structure_id(block: str) -> str:
    """Geometry identity, independent of paths, labels, atom order and rigid motion.

    Rounded pair distances group duplicate molecular geometries; this is not a
    general near-duplicate or trajectory-correlation detector.
    """
    import numpy as np
    from ase.io import read

    atoms = read(io.StringIO(block), format="extxyz")
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("Cannot split an empty or non-finite geometry")
    distances = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
    rows = sorted((int(z), sorted((int(other), round(float(d), 6))
                  for other, d in zip(atoms.numbers, row)))
                  for z, row in zip(atoms.numbers, distances))
    payload = {"atoms": rows, "pbc": atoms.pbc.tolist()}
    if atoms.pbc.any():
        payload["cell_metric"] = np.round(atoms.cell.array @ atoms.cell.array.T, 6).tolist()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_persistent_split(
    blocks: Sequence[str], train_path: Path, valid_path: Path, *,
    manifest_path: Path, valid_ratio: float, seed: int,
    historical_dirs: Sequence[Path] = (),
) -> bool:
    from .sevennet_finetune import _read_extxyz_blocks

    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must satisfy 0 <= ratio < 1")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("identity") != "pair_distances_1e-6":
            raise ValueError("Unsupported split manifest version or identity algorithm")
        if manifest["valid_ratio"] != valid_ratio or manifest["seed"] != seed:
            raise ValueError("Existing split manifest fixes valid_ratio and seed; use its settings")
        assignments = manifest["assignments"]
        if any(value not in ("train", "valid") for value in assignments.values()):
            raise ValueError("Invalid split assignment in manifest")
    else:
        assignments = {}
        manifest = dict(version=1, identity="pair_distances_1e-6", valid_ratio=valid_ratio,
                        seed=seed, assignments=assignments)
        # Import an existing split only if all recorded historical memberships agree.
        for directory in dict.fromkeys([*historical_dirs, train_path.parent]):
            paths = [directory / "train.extxyz", directory / "valid.extxyz"]
            if not any(path.exists() for path in paths):
                if (directory / "checkpoint_best.pth").exists():
                    raise ValueError(f"Missing historical split for checkpoint in {directory}")
                continue
            if not all(path.exists() for path in paths):
                raise ValueError(f"Incomplete historical train/valid split in {directory}")
            for split, path in zip(("train", "valid"), paths):
                for block in _read_extxyz_blocks(path):
                    key = structure_id(block)
                    if key in assignments and assignments[key] != split:
                        raise ValueError(
                            "Historical train/validation leakage detected. Start a new work_dir "
                            "from an uncontaminated checkpoint and a fixed split; conflicting "
                            f"structure found in {path}"
                        )
                    assignments[key] = split

    unique = {}
    for block in blocks:
        unique.setdefault(structure_id(block), block)
    if not unique:
        raise ValueError("No structures available for fine-tuning")
    new_ids = sorted(set(unique) - assignments.keys(),
                     key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    total = len(assignments) + len(new_ids)
    target = min(total - 1, max(1, round(total * valid_ratio))) if valid_ratio > 0 and total > 1 else 0
    n_new_valid = min(len(new_ids), max(0, target - sum(v == "valid" for v in assignments.values())))
    for i, key in enumerate(new_ids):
        assignments[key] = "valid" if i < n_new_valid else "train"
    train = [block for key, block in unique.items() if assignments[key] == "train"]
    valid = [block for key, block in unique.items() if assignments[key] == "valid"]
    if not train:
        raise ValueError("Current dataset contains only validation structures; add training data")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    train_path.write_text("".join(train), encoding="utf-8", newline="\n")
    valid_path.write_text("".join(valid), encoding="utf-8", newline="\n")
    (train_path.parent / "split_summary.json").write_text(json.dumps({
        "manifest": str(manifest_path.resolve()), "input_structures": len(blocks),
        "unique_structures": len(unique), "duplicates_removed": len(blocks) - len(unique),
        "train": len(train), "valid": len(valid), "new_assignments": len(new_ids),
    }, indent=2) + "\n", encoding="utf-8")
    return bool(valid)

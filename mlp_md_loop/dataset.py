from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LabeledAtom:
    element: str
    x: float
    y: float
    z: float
    fx: float
    fy: float
    fz: float


@dataclass(frozen=True)
class LabeledStructure:
    source_path: str
    atoms: tuple[LabeledAtom, ...]
    energy_ev: float
    energy_hartree: float
    charge: int | None = None
    multiplicity: int | None = None
    normal_termination: bool | None = None


def write_jsonl_dataset(path: str | Path, structures: Sequence[LabeledStructure]) -> None:
    """Write labeled structures as JSON lines.

    Units:
    - positions: Angstrom
    - energy: eV
    - forces: eV/Angstrom
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for structure in structures:
            row = asdict(structure)
            row["units"] = {
                "positions": "angstrom",
                "energy": "eV",
                "forces": "eV/angstrom",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_extxyz_dataset(path: str | Path, structures: Sequence[LabeledStructure]) -> None:
    """Write a minimal extended XYZ dataset.

    This is a common interim format for atomistic ML workflows. The final
    project-specific format can be added later without changing the parsers.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for structure in structures:
            handle.write(f"{len(structure.atoms)}\n")
            fields = [
                'Properties=species:S:1:pos:R:3:forces:R:3',
                f"energy={structure.energy_ev:.12f}",
                'pbc="F F F"',
                f'source="{_escape_extxyz_string(structure.source_path)}"',
            ]
            if structure.charge is not None:
                fields.append(f"charge={structure.charge}")
            if structure.multiplicity is not None:
                fields.append(f"multiplicity={structure.multiplicity}")
            handle.write(" ".join(fields) + "\n")
            for atom in structure.atoms:
                handle.write(
                    f"{atom.element:<2s} "
                    f"{atom.x:16.8f} {atom.y:16.8f} {atom.z:16.8f} "
                    f"{atom.fx:16.8f} {atom.fy:16.8f} {atom.fz:16.8f}\n"
                )


def write_dataset_manifest(path: str | Path, structures: Sequence[LabeledStructure]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "source_path\tnatoms\tenergy_ev\tenergy_hartree\tcharge\tmultiplicity\tnormal_termination\n"
        )
        for structure in structures:
            handle.write(
                "\t".join(
                    [
                        structure.source_path,
                        str(len(structure.atoms)),
                        f"{structure.energy_ev:.12f}",
                        f"{structure.energy_hartree:.12f}",
                        "" if structure.charge is None else str(structure.charge),
                        ""
                        if structure.multiplicity is None
                        else str(structure.multiplicity),
                        ""
                        if structure.normal_termination is None
                        else str(structure.normal_termination),
                    ]
                )
                + "\n"
            )


def write_sevennet_structure_list(
    path: str | Path,
    data_file: str | Path,
    label: str = "gaussian",
    index: str = ":",
) -> None:
    """Write a SevenNet-compatible ``structure_list`` file.

    SevenNet treats ``structure_list`` entries as ASE-readable paths unless they
    are VASP OUTCAR files. The generated extxyz dataset can therefore be listed
    here and used with ``load_trainset_path: ['./structure_list']``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_file)
    try:
        data_text = data_path.relative_to(path.parent).as_posix()
    except ValueError:
        data_text = data_path.as_posix()

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{label}]\n")
        handle.write(f"{data_text} {index}\n")


def _escape_extxyz_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')

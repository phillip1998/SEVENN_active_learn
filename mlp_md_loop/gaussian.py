from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mlp_md_loop.dataset import LabeledAtom, LabeledStructure


@dataclass(frozen=True)
class XyzAtom:
    element: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class XyzStructure:
    atoms: tuple[XyzAtom, ...]
    comment: str


@dataclass(frozen=True)
class GaussianInput:
    xyz_path: str
    gjf_path: str
    natoms: int
    route: str
    charge: int
    multiplicity: int


class IncompleteGaussianLogError(ValueError):
    """Raised when a Gaussian log exists but does not contain complete results yet."""


DEFAULT_FORCE_ROUTE = "#p wb97xd/def2svp nosymm force scf=tight"
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    35: "Br",
    53: "I",
}


def read_xyz(path: str | Path) -> XyzStructure:
    """Read a simple XYZ file."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is too short to be an XYZ file")

    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"{path} does not start with an atom count") from exc

    comment = lines[1].strip()
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"{path} declares {natoms} atoms but contains {len(atom_lines)} atom rows")

    atoms = []
    for line_number, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number} is not a valid XYZ atom row")
        element = fields[0]
        try:
            x, y, z = float(fields[1]), float(fields[2]), float(fields[3])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} has invalid coordinates") from exc
        atoms.append(XyzAtom(element=element, x=x, y=y, z=z))

    return XyzStructure(atoms=tuple(atoms), comment=comment)


def xyz_to_gjf(
    xyz_path: str | Path,
    gjf_path: str | Path | None = None,
    route: str = DEFAULT_FORCE_ROUTE,
    charge: int = 0,
    multiplicity: int = 1,
    title: str | None = None,
    mem: str | None = "16GB",
    nprocshared: int | None = 16,
    chk: bool = True,
) -> GaussianInput:
    """Convert one XYZ file to a Gaussian GJF input file.

    Coordinates are assumed to be in Angstrom, matching the XYZ files generated
    by ``sample_dft_clusters``.
    """

    xyz_path = Path(xyz_path)
    structure = read_xyz(xyz_path)
    if gjf_path is None:
        gjf_path = xyz_path.with_suffix(".gjf")
    gjf_path = Path(gjf_path)
    gjf_path.parent.mkdir(parents=True, exist_ok=True)

    clean_route = route.strip()
    if not clean_route.startswith("#"):
        clean_route = "#p " + clean_route

    clean_title = title or f"{xyz_path.stem} | {structure.comment}"
    lines: list[str] = []
    if mem:
        lines.append(f"%mem={mem}")
    if nprocshared:
        lines.append(f"%nprocshared={nprocshared}")
    if chk:
        lines.append(f"%chk={gjf_path.stem}.chk")
    lines.append(clean_route)
    lines.append("")
    lines.append(clean_title)
    lines.append("")
    lines.append(f"{charge} {multiplicity}")
    for atom in structure.atoms:
        lines.append(f"{atom.element:<2s} {atom.x:16.8f} {atom.y:16.8f} {atom.z:16.8f}")
    lines.append("")
    lines.append("")

    gjf_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return GaussianInput(
        xyz_path=str(xyz_path),
        gjf_path=str(gjf_path),
        natoms=len(structure.atoms),
        route=clean_route,
        charge=charge,
        multiplicity=multiplicity,
    )


def convert_xyz_to_gjf_batch(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    route: str = DEFAULT_FORCE_ROUTE,
    charge: int = 0,
    multiplicity: int = 1,
    mem: str | None = "16GB",
    nprocshared: int | None = 16,
    chk: bool = True,
    recursive: bool = True,
    pattern: str = "*.xyz",
) -> list[GaussianInput]:
    """Convert one XYZ file or all XYZ files under a directory to GJF files."""

    input_path = Path(input_path)
    if input_path.is_file():
        xyz_files = [input_path]
        base_dir = input_path.parent
    elif input_path.is_dir():
        xyz_files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))
        base_dir = input_path
    else:
        raise FileNotFoundError(input_path)

    if not xyz_files:
        raise RuntimeError(f"No XYZ files found under {input_path}")

    converted: list[GaussianInput] = []
    for xyz_file in xyz_files:
        if output_dir is None:
            gjf_path = xyz_file.with_suffix(".gjf")
        else:
            relative = xyz_file.relative_to(base_dir)
            gjf_path = Path(output_dir) / relative.with_suffix(".gjf")
        converted.append(
            xyz_to_gjf(
                xyz_path=xyz_file,
                gjf_path=gjf_path,
                route=route,
                charge=charge,
                multiplicity=multiplicity,
                title=None,
                mem=mem,
                nprocshared=nprocshared,
                chk=chk,
            )
        )
    return converted


def write_gaussian_manifest(path: str | Path, inputs: Sequence[GaussianInput]) -> None:
    """Write a small TSV manifest for generated Gaussian inputs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("gjf_path\txyz_path\tnatoms\troute\tcharge\tmultiplicity\n")
        for item in inputs:
            handle.write(
                "\t".join(
                    [
                        item.gjf_path,
                        item.xyz_path,
                        str(item.natoms),
                        item.route,
                        str(item.charge),
                        str(item.multiplicity),
                    ]
                )
                + "\n"
            )


def parse_gaussian_log(path: str | Path, require_normal_termination: bool = False) -> LabeledStructure:
    """Parse energy, coordinates, and forces from a Gaussian log/out file.

    The returned structure uses Angstrom, eV, and eV/Angstrom. The parser is
    intentionally conservative: it requires a final SCF energy, a coordinate
    orientation block, and a force block from ``force`` calculations.
    """

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    normal_termination = any("Normal termination of Gaussian" in line for line in lines)
    if require_normal_termination and not normal_termination:
        raise IncompleteGaussianLogError(f"{path} did not terminate normally")

    energy_hartree = _parse_last_scf_energy(lines, path)
    charge, multiplicity = _parse_charge_multiplicity(lines)
    coordinates = _parse_last_orientation(lines, path)
    forces = _parse_last_force_block(lines, path)

    if len(coordinates) != len(forces):
        raise ValueError(
            f"{path} has {len(coordinates)} coordinate rows but {len(forces)} force rows"
        )

    atoms = []
    for coord, force in zip(coordinates, forces):
        coord_center, atomic_number, x, y, z = coord
        force_center, fx, fy, fz = force
        if coord_center != force_center:
            raise ValueError(
                f"{path} coordinate center {coord_center} does not match force center {force_center}"
            )
        atoms.append(
            LabeledAtom(
                element=ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, str(atomic_number)),
                x=x,
                y=y,
                z=z,
                fx=fx * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                fy=fy * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
                fz=fz * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            )
        )

    return LabeledStructure(
        source_path=str(path),
        atoms=tuple(atoms),
        energy_ev=energy_hartree * HARTREE_TO_EV,
        energy_hartree=energy_hartree,
        charge=charge,
        multiplicity=multiplicity,
        normal_termination=normal_termination,
    )


def parse_gaussian_logs_batch(
    input_path: str | Path,
    recursive: bool = True,
    require_normal_termination: bool = False,
    patterns: Sequence[str] = ("*.log", "*.out"),
) -> list[LabeledStructure]:
    """Parse one Gaussian log/out file or all matching files in a directory."""

    input_path = Path(input_path)
    if input_path.is_file():
        log_files = [input_path]
    elif input_path.is_dir():
        log_files = []
        for pattern in patterns:
            iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
            log_files.extend(iterator)
        log_files = sorted(set(log_files))
    else:
        raise FileNotFoundError(input_path)

    if not log_files:
        raise RuntimeError(f"No Gaussian log/out files found under {input_path}")

    structures: list[LabeledStructure] = []
    skipped: list[tuple[Path, str]] = []
    for path in log_files:
        try:
            structures.append(
                parse_gaussian_log(path, require_normal_termination=require_normal_termination)
            )
        except IncompleteGaussianLogError as exc:
            skipped.append((path, str(exc)))

    if skipped:
        warnings.warn(
            "Skipped {} incomplete Gaussian log/out file(s) without complete results. "
            "First skipped: {}".format(len(skipped), skipped[0][0]),
            RuntimeWarning,
            stacklevel=2,
        )

    if not structures:
        raise RuntimeError(
            f"No complete Gaussian result files found under {input_path}; "
            f"skipped {len(skipped)} incomplete file(s)"
        )

    return structures


def _parse_last_scf_energy(lines: Sequence[str], path: Path) -> float:
    energy: float | None = None
    for line in lines:
        if "SCF Done:" not in line:
            continue
        fields = line.replace("=", " = ").split()
        for index, field in enumerate(fields):
            if field == "=" and index + 1 < len(fields):
                try:
                    energy = float(fields[index + 1].replace("D", "E"))
                    break
                except ValueError:
                    continue
    if energy is None:
        raise IncompleteGaussianLogError(f"{path} does not contain an SCF Done energy")
    return energy


def _parse_charge_multiplicity(lines: Sequence[str]) -> tuple[int | None, int | None]:
    for line in lines:
        if "Charge =" not in line or "Multiplicity =" not in line:
            continue
        fields = line.replace("=", " = ").split()
        charge = None
        multiplicity = None
        for index, field in enumerate(fields):
            if field == "Charge" and index + 2 < len(fields):
                charge = int(fields[index + 2])
            if field == "Multiplicity" and index + 2 < len(fields):
                multiplicity = int(fields[index + 2])
        return charge, multiplicity
    return None, None


def _parse_last_orientation(
    lines: Sequence[str],
    path: Path,
) -> list[tuple[int, int, float, float, float]]:
    blocks: list[list[tuple[int, int, float, float, float]]] = []
    for index, line in enumerate(lines):
        if "Standard orientation:" not in line and "Input orientation:" not in line:
            continue
        block = _parse_orientation_block(lines, index)
        if block:
            blocks.append(block)
    if not blocks:
        raise IncompleteGaussianLogError(f"{path} does not contain an orientation block")
    return blocks[-1]


def _parse_orientation_block(
    lines: Sequence[str],
    start_index: int,
) -> list[tuple[int, int, float, float, float]]:
    dashed_seen = 0
    row_start = None
    for index in range(start_index + 1, min(start_index + 12, len(lines))):
        if lines[index].strip().startswith("-----"):
            dashed_seen += 1
            if dashed_seen == 2:
                row_start = index + 1
                break
    if row_start is None:
        return []

    rows: list[tuple[int, int, float, float, float]] = []
    for line in lines[row_start:]:
        if line.strip().startswith("-----"):
            break
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            center = int(fields[0])
            atomic_number = int(fields[1])
            x, y, z = float(fields[3]), float(fields[4]), float(fields[5])
        except ValueError:
            continue
        rows.append((center, atomic_number, x, y, z))
    return rows


def _parse_last_force_block(
    lines: Sequence[str],
    path: Path,
) -> list[tuple[int, float, float, float]]:
    blocks: list[list[tuple[int, float, float, float]]] = []
    for index, line in enumerate(lines):
        if "Forces (Hartrees/Bohr)" not in line:
            continue
        block = _parse_force_block(lines, index)
        if block:
            blocks.append(block)
    if not blocks:
        raise IncompleteGaussianLogError(
            f"{path} does not contain a Forces (Hartrees/Bohr) block"
        )
    return blocks[-1]


def _parse_force_block(lines: Sequence[str], start_index: int) -> list[tuple[int, float, float, float]]:
    row_start = None
    for index in range(start_index + 1, min(start_index + 10, len(lines))):
        if lines[index].strip().startswith("-----"):
            row_start = index + 1
            break
    if row_start is None:
        return []

    rows: list[tuple[int, float, float, float]] = []
    for line in lines[row_start:]:
        if line.strip().startswith("-----"):
            break
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            center = int(fields[0])
            fx, fy, fz = (
                float(fields[2].replace("D", "E")),
                float(fields[3].replace("D", "E")),
                float(fields[4].replace("D", "E")),
            )
        except ValueError:
            continue
        rows.append((center, fx, fy, fz))
    return rows

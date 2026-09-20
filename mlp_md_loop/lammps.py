from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import shutil
from pathlib import Path
from typing import Literal, Sequence

from mlp_md_loop.cluster_sampling import infer_element, read_gro


SevenNetPairStyle = Literal["e3gnn", "e3gnn/parallel", "mliap"]


ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "S": 32.06,
    "P": 30.973761998,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90447,
}


@dataclass(frozen=True)
class LammpsDataInfo:
    data_path: str
    natoms: int
    elements: tuple[str, ...]
    box_angstrom: tuple[float, float, float]


@dataclass(frozen=True)
class LammpsInputInfo:
    input_path: str
    pair_style: str
    pair_coeff: str
    data_file: str
    elements: tuple[str, ...]


def write_lammps_data_from_gro(
    gro_path: str | Path,
    data_path: str | Path,
    elements: Sequence[str] | None = None,
    title: str | None = None,
) -> LammpsDataInfo:
    """Convert a GROMACS GRO snapshot to a simple LAMMPS atomic data file.

    The generated data file is intended for ML interatomic potentials such as
    SevenNet, so it contains atom IDs, atom types, masses, and coordinates only.
    Coordinates and box lengths are converted from nm to Angstrom.
    """

    gro_path = Path(gro_path)
    data_path = Path(data_path)
    atoms, _, frame = read_gro(gro_path)

    inferred_elements = tuple(atom.element for atom in atoms)
    if elements is None:
        elements = tuple(sorted(set(inferred_elements), key=_element_sort_key))
    else:
        elements = _normalize_elements(elements)

    missing = sorted(set(inferred_elements) - set(elements), key=_element_sort_key)
    if missing:
        raise ValueError(
            f"elements is missing species present in {gro_path}: {missing}. "
            f"Parsed elements were: {list(elements)}"
        )

    element_to_type = {element: index + 1 for index, element in enumerate(elements)}
    box_angstrom = tuple(value * 10.0 for value in frame.box_nm)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    with data_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{title or f'LAMMPS data generated from {gro_path.name}'}\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{len(elements)} atom types\n\n")
        handle.write(f"0.000000 {box_angstrom[0]:.8f} xlo xhi\n")
        handle.write(f"0.000000 {box_angstrom[1]:.8f} ylo yhi\n")
        handle.write(f"0.000000 {box_angstrom[2]:.8f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        for element in elements:
            if element not in ATOMIC_MASSES:
                raise ValueError(f"No mass is defined for element {element}")
            handle.write(f"{element_to_type[element]} {ATOMIC_MASSES[element]:.10f} # {element}\n")

        handle.write("\nAtoms # atomic\n\n")
        for atom_index, (atom, position_nm) in enumerate(zip(atoms, frame.positions_nm), start=1):
            atom_type = element_to_type[atom.element]
            x, y, z = (coord * 10.0 for coord in position_nm)
            handle.write(f"{atom_index} {atom_type} {x:.8f} {y:.8f} {z:.8f}\n")

    return LammpsDataInfo(
        data_path=str(data_path),
        natoms=len(atoms),
        elements=tuple(elements),
        box_angstrom=box_angstrom,
    )


def write_sevennet_lammps_input(
    input_path: str | Path,
    data_file: str | Path,
    model_path: str | Path,
    elements: Sequence[str],
    pair_style: SevenNetPairStyle = "e3gnn",
    parallel_model_count: int | None = None,
    temperature_k: float = 300.0,
    timestep_ps: float = 0.001,
    run_steps: int = 1000,
    thermo_interval: int = 100,
    dump_interval: int = 100,
    ensemble: Literal["nve", "nvt", "minimize"] = "nvt",
    seed: int = 12345,
    d3: bool = False,
    d3_functional: str = "pbe",
    d3_damping: str = "damp_bj",
    initialize_velocities: bool = True,
) -> LammpsInputInfo:
    """Write a SevenNet-ready LAMMPS input script.

    Supported styles:
    - ``e3gnn``: SevenNet TorchScript serial model, usually ``deployed_serial.pt``.
    - ``e3gnn/parallel``: directory of deployed parallel model shards.
    - ``mliap``: ML-IAP deployment, usually ``deployed_serial_mliap.pt``.
    """

    input_path = Path(input_path)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    elements = tuple(elements)
    model_text = Path(model_path).as_posix()
    data_text = Path(data_file).as_posix()

    pair_style_line, pair_coeff_line = _sevennet_pair_lines(
        pair_style=pair_style,
        model_path=model_text,
        elements=elements,
        parallel_model_count=parallel_model_count,
        d3=d3,
        d3_functional=d3_functional,
        d3_damping=d3_damping,
    )

    lines = [
        "# LAMMPS input generated for SevenNet morphology MD",
        "",
        "units           metal",
        "boundary        p p p",
        "atom_style      atomic",
        "atom_modify     map yes",
        "newton          on",
        "",
        f"read_data       {data_text}",
        "",
        "# SevenNet potential",
        pair_style_line,
        pair_coeff_line,
    ]
    if d3:
        lines.append(f"pair_coeff      * * d3 {' '.join(elements)}")

    lines.extend(
        [
            "",
            f"timestep        {timestep_ps:.8f}",
            "",
            f"thermo          {thermo_interval}",
            "thermo_style    custom step time temp pe ke etotal press vol",
            f"dump            traj all custom {dump_interval} dump.sevennet.lammpstrj id type element x y z fx fy fz",
            f"dump_modify     traj sort id element {' '.join(elements)}",
            "",
        ]
    )

    if ensemble == "minimize":
        lines.extend(
            [
                "min_style       cg",
                "minimize        1.0e-6 1.0e-8 1000 10000",
            ]
        )
    elif ensemble == "nve":
        lines.extend(
            [
                f"velocity        all create {temperature_k:.6f} {seed} dist gaussian mom yes rot yes",
                "fix             int all nve",
                "fix             com all momentum 100 linear 1 1 1",
                f"run             {run_steps}",
            ]
        )
    elif ensemble == "nvt":
        tdamp = max(100.0 * timestep_ps, 0.1)
        lines.extend(
            [
                f"velocity        all create {temperature_k:.6f} {seed} dist gaussian mom yes rot yes",
                f"fix             int all nvt temp {temperature_k:.6f} {temperature_k:.6f} {tdamp:.6f}",
                "fix             com all momentum 100 linear 1 1 1",
                f"run             {run_steps}",
            ]
        )
    else:
        raise ValueError(f"Unsupported ensemble: {ensemble}")

    if not initialize_velocities:
        lines = [line for line in lines if not line.startswith("velocity        all create")]
    lines.extend([
        "",
        "# Carry geometry, box and velocities to the next active-learning iteration.",
        "write_data      final.data nocoeff",
        'print           "MLP_MD_COMPLETE" file md.complete',
        "",
    ])
    input_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return LammpsInputInfo(
        input_path=str(input_path),
        pair_style=pair_style,
        pair_coeff=pair_coeff_line,
        data_file=str(data_file),
        elements=elements,
    )


def prepare_sevennet_lammps_from_gro(
    gro_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    elements: Sequence[str] | None = None,
    pair_style: SevenNetPairStyle = "e3gnn",
    continuation_data_path: str | Path | None = None,
    **input_kwargs,
) -> tuple[LammpsDataInfo, LammpsInputInfo]:
    """Create both ``system.data`` and ``in.sevennet.lmp`` from a GRO snapshot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_info = write_lammps_data_from_gro(
        gro_path=gro_path,
        data_path=output_dir / "system.data",
        elements=elements,
    )
    if continuation_data_path is not None:
        source = Path(continuation_data_path)
        box = validate_md_data(source, gro_path, data_info.elements)
        shutil.copyfile(source, output_dir / "system.data")
        data_info = LammpsDataInfo(data_info.data_path, data_info.natoms, data_info.elements, box)
        input_kwargs["initialize_velocities"] = False
    (output_dir / "md_context.json").write_text(json.dumps({
        "reference_sha256": hashlib.sha256(Path(gro_path).read_bytes()).hexdigest(),
        "elements": list(data_info.elements),
        "initial_state": str(Path(continuation_data_path).resolve()) if continuation_data_path else str(Path(gro_path).resolve()),
    }, indent=2) + "\n", encoding="utf-8")
    input_info = write_sevennet_lammps_input(
        input_path=output_dir / "in.sevennet.lmp",
        data_file="system.data",
        model_path=model_path,
        elements=data_info.elements,
        pair_style=pair_style,
        **input_kwargs,
    )
    return data_info, input_info


def validate_md_data(path: Path, reference_gro: str | Path, elements: Sequence[str]) -> tuple[float, float, float]:
    """Reject incomplete/invalid final states and changed atom ID/type mappings."""
    import numpy as np
    from ase.io import read
    from ase.data import atomic_numbers

    atoms, _, _ = read_gro(reference_gro)
    state = read(str(path), format="lammps-data", atom_style="atomic", units="metal",
                 Z_of_type={i + 1: atomic_numbers[e] for i, e in enumerate(elements)},
                 sort_by_id=True)
    if len(state) != len(atoms) or not np.array_equal(state.arrays["id"], np.arange(1, len(atoms) + 1)):
        raise ValueError(f"MD continuation atom IDs/count differ from reference GRO: {path}")
    expected_types = [elements.index(atom.element) + 1 for atom in atoms]
    if not np.array_equal(state.arrays["type"], expected_types):
        raise ValueError(f"MD continuation atom types differ from reference GRO: {path}")
    if not state.has("momenta"):
        raise ValueError(f"MD continuation is missing velocities: {path}")
    if not all(np.isfinite(values).all() for values in (state.positions, state.cell.array, state.get_velocities())):
        raise ValueError(f"MD continuation contains non-finite values: {path}")
    if np.any(np.diag(state.cell.array) <= 0) or not np.allclose(state.cell.array, np.diag(np.diag(state.cell.array))):
        raise ValueError(f"MD continuation requires a positive orthogonal cell: {path}")
    return tuple(float(v) for v in np.diag(state.cell.array))


def completed_md_data(md_dir: Path, reference_gro: str | Path) -> Path:
    """Only accept a final state written after the MD run reached its last command."""
    data, marker, context = md_dir / "final.data", md_dir / "md.complete", md_dir / "md_context.json"
    if not all(p.is_file() for p in (data, marker, context)):
        raise RuntimeError(f"No completed MD state in {md_dir}. Run its generated run_md.sh successfully before continuing; no fallback to the original GRO is performed.")
    if marker.read_text(encoding="utf-8").strip() != "MLP_MD_COMPLETE":
        raise RuntimeError(f"Invalid MD completion marker: {marker}")
    metadata = json.loads(context.read_text(encoding="utf-8"))
    if metadata["reference_sha256"] != hashlib.sha256(Path(reference_gro).read_bytes()).hexdigest():
        raise ValueError("Reference GRO changed since previous MD; start a new work_dir")
    validate_md_data(data, reference_gro, metadata["elements"])
    return data


def _sevennet_pair_lines(
    pair_style: SevenNetPairStyle,
    model_path: str,
    elements: Sequence[str],
    parallel_model_count: int | None,
    d3: bool,
    d3_functional: str,
    d3_damping: str,
) -> tuple[str, str]:
    element_text = " ".join(elements)
    model_token = _lammps_token(model_path)
    if pair_style == "e3gnn":
        if d3:
            return (
                f"pair_style      hybrid/overlay e3gnn d3 9000 1600 {d3_damping} {d3_functional}",
                f"pair_coeff      * * e3gnn {model_token} {element_text}",
            )
        return "pair_style      e3gnn", f"pair_coeff      * * {model_token} {element_text}"

    if pair_style == "e3gnn/parallel":
        if d3:
            raise ValueError("SevenNet docs state D3 is not supported with e3gnn/parallel")
        if parallel_model_count is None:
            parallel_model_count = _count_parallel_models(model_path)
        return (
            "pair_style      e3gnn/parallel",
            f"pair_coeff      * * {parallel_model_count} {model_token} {element_text}",
        )

    if pair_style == "mliap":
        if d3:
            return (
                f"pair_style      hybrid/overlay mliap unified {model_token} 0 d3 9000 1600 {d3_damping} {d3_functional}",
                f"pair_coeff      * * mliap {element_text}",
            )
        return (
            f"pair_style      mliap unified {model_token} 0",
            f"pair_coeff      * * {element_text}",
        )

    raise ValueError(f"Unsupported pair_style: {pair_style}")


def _count_parallel_models(model_dir: str) -> int:
    path = Path(model_dir)
    count = len(list(path.glob("deployed_parallel_*.pt"))) if path.is_dir() else 0
    if count == 0:
        raise ValueError(
            "parallel_model_count was not given and no deployed_parallel_*.pt "
            f"files were found in {model_dir}"
        )
    return count


def _lammps_token(value: str) -> str:
    if any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _element_sort_key(element: str) -> tuple[int, str]:
    order = {symbol: index for index, symbol in enumerate(ATOMIC_MASSES)}
    return (order.get(element, 999), element)


def _normalize_elements(elements: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_element in elements:
        for token in str(raw_element).replace(",", " ").split():
            if not token:
                continue
            element = token[0].upper() + token[1:].lower()
            if element not in seen:
                seen.add(element)
                normalized.append(element)

    return tuple(normalized)

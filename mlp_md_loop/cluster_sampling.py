from __future__ import annotations

import csv
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


from .solution_sampling import validate_sampling, solution_candidates, select_solution


Vector = tuple[float, float, float]


@dataclass(frozen=True)
class AtomRecord:
    resid: int
    resname: str
    atomname: str
    atomnr: int
    element: str


@dataclass(frozen=True)
class MoleculeRecord:
    resid: int
    resname: str
    atom_indices: tuple[int, ...]
    heavy_atom_indices: tuple[int, ...]


@dataclass
class FrameData:
    frame_index: int
    time_ps: float | None
    positions_nm: list[Vector]
    box_nm: Vector


@dataclass
class ClusterSample:
    sample_id: int
    cluster_size: int
    frame_index: int
    time_ps: float | None
    molecule_resids: list[int]
    score: float
    avg_pair_com_distance_nm: float
    min_interatomic_distance_nm: float
    avg_contact_count: float
    avg_pi_stack_score: float
    xyz_path: str | None = None


@dataclass
class _MoleculeGeometry:
    center: Vector
    normal: Vector


@dataclass
class _Candidate:
    frame: FrameData
    mol_indices: tuple[int, ...]
    score: float
    avg_pair_com_distance_nm: float
    min_interatomic_distance_nm: float
    avg_contact_count: float
    avg_pi_stack_score: float
    feature: tuple[float, ...]


def sample_dft_clusters(
    gro_path: str | Path,
    xtc_path: str | Path | None = None,
    cluster_size: int = 2,
    n_samples: int = 20,
    output_dir: str | Path = "sampled_clusters",
    frame_stride: int = 1,
    max_frames: int | None = None,
    random_frame_samples: int | None = None,
    neighbor_pool: int = 8,
    contact_cutoff_nm: float = 0.50,
    min_distance_reject_nm: float = 0.08,
    random_seed: int = 7,
    write_xyz: bool = True,
    sampling_mode: str = "film",
    solute_resnames: Sequence[str] = (),
    solvent_resnames: Sequence[str] = (),
    solution_solute_weights: dict[int, float] | None = None,
) -> list[ClusterSample]:
    """Sample dimer/trimer/tetramer-like molecular clusters for DFT labeling.

    Molecules are inferred from residues in the GROMACS ``.gro`` topology. The
    sampler favors condensed-phase contacts that are useful for morphology MLP
    training: close nonbonded contacts, compact clusters, and a lightweight
    pi-stacking proxy based on molecular plane alignment.

    Args:
        gro_path: GROMACS ``.gro`` file used as topology and, when ``xtc_path``
            is omitted, as the single-frame coordinate source.
        xtc_path: Optional ``.xtc`` trajectory. Reading XTC requires
            MDAnalysis. Without it, use the ``.gro`` snapshot only.
        cluster_size: Number of molecules in each extracted cluster. Use 2 for
            dimers, 3 for trimers, 4 for tetramers, etc.
        n_samples: Number of cluster samples to return.
        output_dir: Directory for extracted XYZ files and metadata.
        frame_stride: Process every Nth trajectory frame.
        max_frames: Stop after this many processed frames.
        random_frame_samples: Randomly choose this many eligible XTC frames
            before loading coordinates. This keeps initial sampling cheap on
            long trajectories.
        neighbor_pool: Number of nearest COM neighbors considered per seed
            molecule before constructing clusters.
        contact_cutoff_nm: Heavy-atom distance cutoff used for contact counts.
        min_distance_reject_nm: Reject clusters with unrealistically close
            inter-molecular heavy-atom distances below this threshold.
        random_seed: Seed used for deterministic tie-breaking.
        write_xyz: Whether to write one XYZ file per selected cluster.

    Returns:
        A list of selected samples. Metadata is also written under
        ``output_dir``.
    """

    if cluster_size < 2:
        raise ValueError("cluster_size must be at least 2")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if random_frame_samples is not None and random_frame_samples < 1:
        raise ValueError("random_frame_samples must be at least 1")

    gro_path = Path(gro_path)
    output_dir = Path(output_dir)

    atoms, molecules, gro_frame = read_gro(gro_path)
    validate_sampling(sampling_mode, solute_resnames, solvent_resnames, solution_solute_weights, molecules)
    frames = _load_frames(
        gro_path,
        xtc_path,
        gro_frame,
        frame_stride,
        max_frames,
        random_frame_samples,
        random_seed,
    )

    rng = random.Random(random_seed)
    candidates: list[_Candidate] = []
    for frame in frames:
        if sampling_mode == "solution":
            geometries = _molecule_geometries(frame, molecules)
            pair_cache = {}
            for indices in solution_candidates(frame, molecules, cluster_size, neighbor_pool,
                    contact_cutoff_nm, solute_resnames, solution_solute_weights, rng):
                candidate = _score_candidate(frame, molecules, geometries, indices,
                    contact_cutoff_nm, min_distance_reject_nm, pair_cache)
                if candidate is not None:
                    # Solution ranking omits the film-specific pi-stacking proxy.
                    candidate.score = (0.5 * min(candidate.avg_contact_count / 18.0, 1.0)
                        + 0.5 / (1.0 + candidate.avg_pair_com_distance_nm))
                    candidate.feature = candidate.feature[:-1]
                    candidates.append(candidate)
            continue
        candidates.extend(
            _frame_candidates(
                frame=frame,
                molecules=molecules,
                cluster_size=cluster_size,
                neighbor_pool=neighbor_pool,
                contact_cutoff_nm=contact_cutoff_nm,
                min_distance_reject_nm=min_distance_reject_nm,
                rng=rng,
            )
        )

    if not candidates and sampling_mode == "film":
        raise RuntimeError(
            "No valid clusters were found. Try increasing contact_cutoff_nm or "
            "neighbor_pool, or check whether molecules are wrapped across PBC."
        )

    selected = (select_solution(candidates, molecules, solute_resnames, cluster_size,
        solution_solute_weights, n_samples, _select_diverse_candidates, output_dir)
        if sampling_mode == "solution" else _select_diverse_candidates(candidates, n_samples))
    output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[ClusterSample] = []
    for sample_id, candidate in enumerate(selected, start=1):
        molecule_resids = [molecules[i].resid for i in candidate.mol_indices]
        xyz_path: str | None = None
        if write_xyz:
            xyz_file = output_dir / (
                f"cluster_{cluster_size}mol_{sample_id:04d}_"
                f"frame{candidate.frame.frame_index:05d}_"
                f"res{'-'.join(str(r) for r in molecule_resids)}.xyz"
            )
            write_cluster_xyz(
                xyz_file,
                atoms,
                molecules,
                candidate.frame,
                candidate.mol_indices,
            )
            xyz_path = str(xyz_file)

        samples.append(
            ClusterSample(
                sample_id=sample_id,
                cluster_size=cluster_size,
                frame_index=candidate.frame.frame_index,
                time_ps=candidate.frame.time_ps,
                molecule_resids=molecule_resids,
                score=candidate.score,
                avg_pair_com_distance_nm=candidate.avg_pair_com_distance_nm,
                min_interatomic_distance_nm=candidate.min_interatomic_distance_nm,
                avg_contact_count=candidate.avg_contact_count,
                avg_pi_stack_score=candidate.avg_pi_stack_score,
                xyz_path=xyz_path,
            )
        )

    _write_metadata(output_dir, samples)
    return samples


def sample_dft_clusters_for_sizes(
    gro_path: str | Path,
    cluster_sizes: Sequence[int],
    n_samples: int,
    output_dir: str | Path = "sampled_clusters",
    xtc_path: str | Path | None = None,
    **kwargs,
) -> dict[int, list[ClusterSample]]:
    """Sample multiple cluster sizes, for example dimers through tetramers."""

    results: dict[int, list[ClusterSample]] = {}
    for cluster_size in cluster_sizes:
        size_output = Path(output_dir) / f"{cluster_size}mol"
        results[cluster_size] = sample_dft_clusters(
            gro_path=gro_path,
            xtc_path=xtc_path,
            cluster_size=cluster_size,
            n_samples=n_samples,
            output_dir=size_output,
            **kwargs,
        )
    return results


def read_gro(path: str | Path) -> tuple[list[AtomRecord], list[MoleculeRecord], FrameData]:
    """Read a GROMACS GRO file as topology plus a single coordinate frame."""

    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} is too short to be a GRO file")

    natoms = int(lines[1].strip())
    atom_lines = lines[2 : 2 + natoms]
    box_fields = lines[2 + natoms].split()
    if len(box_fields) < 3:
        raise ValueError("Only rectangular/triclinic boxes with at least 3 fields are supported")
    box_nm = (float(box_fields[0]), float(box_fields[1]), float(box_fields[2]))

    atoms: list[AtomRecord] = []
    positions: list[Vector] = []
    residue_to_indices: dict[tuple[int, str], list[int]] = {}

    for zero_index, line in enumerate(atom_lines):
        resid = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnr = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        element = infer_element(atomname)
        atoms.append(AtomRecord(resid, resname, atomname, atomnr, element))
        positions.append((x, y, z))
        residue_to_indices.setdefault((resid, resname), []).append(zero_index)

    molecules: list[MoleculeRecord] = []
    for (resid, resname), indices in sorted(residue_to_indices.items()):
        heavy = tuple(i for i in indices if atoms[i].element != "H")
        molecules.append(
            MoleculeRecord(
                resid=resid,
                resname=resname,
                atom_indices=tuple(indices),
                heavy_atom_indices=heavy or tuple(indices),
            )
        )

    frame = FrameData(frame_index=0, time_ps=None, positions_nm=positions, box_nm=box_nm)
    return atoms, molecules, frame


def write_cluster_xyz(
    path: str | Path,
    atoms: Sequence[AtomRecord],
    molecules: Sequence[MoleculeRecord],
    frame: FrameData,
    mol_indices: Sequence[int],
) -> None:
    """Write a PBC-unwrapped cluster as XYZ in Angstrom units."""

    path = Path(path)
    cluster_atoms: list[tuple[str, Vector]] = []
    geometries = _molecule_geometries(frame, molecules)
    seed_center = geometries[mol_indices[0]].center

    for mol_index in mol_indices:
        mol = molecules[mol_index]
        mol_center = geometries[mol_index].center
        delta = minimum_image_delta(seed_center, mol_center, frame.box_nm)
        target_center = add(seed_center, delta)
        shift = sub(target_center, mol_center)
        unwrapped_positions = _unwrapped_molecule_positions(frame, mol, mol.atom_indices)

        for atom_index in mol.atom_indices:
            pos_nm = add(unwrapped_positions[atom_index], shift)
            cluster_atoms.append((atoms[atom_index].element, scale(pos_nm, 10.0)))

    center_ang = mean_vector([pos for _, pos in cluster_atoms])
    centered_atoms = [(element, sub(pos, center_ang)) for element, pos in cluster_atoms]

    with path.open("w", newline="\n") as handle:
        handle.write(f"{len(centered_atoms)}\n")
        handle.write(
            "cluster_size={} frame={} residues={} source_units=angstrom\n".format(
                len(mol_indices),
                frame.frame_index,
                ",".join(str(molecules[i].resid) for i in mol_indices),
            )
        )
        for element, pos in centered_atoms:
            handle.write(f"{element:2s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}\n")


def infer_element(atomname: str) -> str:
    """Infer an element symbol from a GROMACS atom name."""

    stripped = "".join(ch for ch in atomname.strip() if ch.isalpha())
    if not stripped:
        return "X"
    first = stripped[0].upper()
    second = stripped[1:2].lower()
    two_letter = first + second
    if two_letter in {"Cl", "Br", "Si", "Na", "Li", "Mg", "Ca", "Al"}:
        return two_letter
    return first


def _load_frames(
    gro_path: Path,
    xtc_path: str | Path | None,
    gro_frame: FrameData,
    frame_stride: int,
    max_frames: int | None,
    random_frame_samples: int | None,
    random_seed: int,
) -> list[FrameData]:
    if xtc_path is None:
        return [gro_frame]

    try:
        import MDAnalysis as mda  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading .xtc trajectories requires MDAnalysis. Install it or call "
            "sample_dft_clusters(..., xtc_path=None) to sample the .gro snapshot."
        ) from exc

    universe = mda.Universe(str(gro_path), str(xtc_path))
    frames: list[FrameData] = []
    eligible_indices = list(range(0, len(universe.trajectory), frame_stride))
    if max_frames is not None:
        eligible_indices = eligible_indices[:max_frames]
    if random_frame_samples is not None and len(eligible_indices) > random_frame_samples:
        eligible_indices = sorted(random.Random(random_seed).sample(eligible_indices, random_frame_samples))

    for frame_index in eligible_indices:
        ts = universe.trajectory[frame_index]
        positions_nm = [
            (float(x) / 10.0, float(y) / 10.0, float(z) / 10.0)
            for x, y, z in universe.atoms.positions
        ]
        dims = ts.dimensions
        box_nm = (float(dims[0]) / 10.0, float(dims[1]) / 10.0, float(dims[2]) / 10.0)
        frames.append(
            FrameData(
                frame_index=int(ts.frame),
                time_ps=float(ts.time) if ts.time is not None else None,
                positions_nm=positions_nm,
                box_nm=box_nm,
            )
        )
    return frames


def _frame_candidates(
    frame: FrameData,
    molecules: Sequence[MoleculeRecord],
    cluster_size: int,
    neighbor_pool: int,
    contact_cutoff_nm: float,
    min_distance_reject_nm: float,
    rng: random.Random,
) -> list[_Candidate]:
    geometries = _molecule_geometries(frame, molecules)
    neighbor_pool = max(cluster_size - 1, neighbor_pool)
    seen: set[tuple[int, ...]] = set()
    rough: list[tuple[float, tuple[int, ...]]] = []

    for seed_index, seed_geom in enumerate(geometries):
        neighbors: list[tuple[float, int]] = []
        for other_index, other_geom in enumerate(geometries):
            if seed_index == other_index:
                continue
            dist = norm(minimum_image_delta(seed_geom.center, other_geom.center, frame.box_nm))
            neighbors.append((dist, other_index))
        neighbors.sort(key=lambda item: item[0])
        pool = [index for _, index in neighbors[:neighbor_pool]]

        for combo in itertools.combinations(pool, cluster_size - 1):
            mol_indices = tuple(sorted((seed_index, *combo)))
            if mol_indices in seen:
                continue
            seen.add(mol_indices)
            pair_dists = [
                norm(
                    minimum_image_delta(
                        geometries[i].center, geometries[j].center, frame.box_nm
                    )
                )
                for i, j in itertools.combinations(mol_indices, 2)
            ]
            compactness = sum(pair_dists) / len(pair_dists)
            rough.append((compactness + rng.random() * 1e-9, mol_indices))

    max_rough = max(4000, 400 * cluster_size)
    rough.sort(key=lambda item: item[0])
    candidates: list[_Candidate] = []
    pair_cache: dict[tuple[int, int], dict[str, float]] = {}
    for _, mol_indices in rough[:max_rough]:
        scored = _score_candidate(
            frame,
            molecules,
            geometries,
            mol_indices,
            contact_cutoff_nm,
            min_distance_reject_nm,
            pair_cache,
        )
        if scored is not None:
            candidates.append(scored)

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def _score_candidate(
    frame: FrameData,
    molecules: Sequence[MoleculeRecord],
    geometries: Sequence[_MoleculeGeometry],
    mol_indices: tuple[int, ...],
    contact_cutoff_nm: float,
    min_distance_reject_nm: float,
    pair_cache: dict[tuple[int, int], dict[str, float]],
) -> _Candidate | None:
    pair_metrics = []
    contact_edges = 0
    for i, j in itertools.combinations(mol_indices, 2):
        pair_key = (i, j) if i < j else (j, i)
        if pair_key not in pair_cache:
            pair_cache[pair_key] = _pair_metrics(
                frame,
                molecules[i],
                molecules[j],
                geometries[i],
                geometries[j],
                contact_cutoff_nm,
            )
        metrics = pair_cache[pair_key]
        if metrics["min_distance"] < min_distance_reject_nm:
            return None
        if metrics["min_distance"] <= contact_cutoff_nm or metrics["com_distance"] <= 1.20:
            contact_edges += 1
        pair_metrics.append(metrics)

    if not _is_connected_cluster(mol_indices, pair_metrics, contact_cutoff_nm):
        return None

    avg_com = sum(metric["com_distance"] for metric in pair_metrics) / len(pair_metrics)
    min_dist = min(metric["min_distance"] for metric in pair_metrics)
    avg_contacts = sum(metric["contact_count"] for metric in pair_metrics) / len(pair_metrics)
    avg_pi = sum(metric["pi_score"] for metric in pair_metrics) / len(pair_metrics)

    contact_score = min(avg_contacts / 18.0, 1.0)
    compact_score = 1.0 / (1.0 + avg_com)
    close_score = 1.0 / (1.0 + ((min_dist - 0.34) / 0.14) ** 2)
    connectivity_score = min(contact_edges / max(1, len(mol_indices) - 1), 1.0)

    score = (
        0.34 * contact_score
        + 0.28 * avg_pi
        + 0.20 * compact_score
        + 0.12 * close_score
        + 0.06 * connectivity_score
    )

    feature_values = sorted(metric["com_distance"] for metric in pair_metrics)
    feature_values.extend(sorted(metric["min_distance"] for metric in pair_metrics))
    feature_values.extend([avg_contacts / 30.0, avg_pi])

    return _Candidate(
        frame=frame,
        mol_indices=mol_indices,
        score=score,
        avg_pair_com_distance_nm=avg_com,
        min_interatomic_distance_nm=min_dist,
        avg_contact_count=avg_contacts,
        avg_pi_stack_score=avg_pi,
        feature=tuple(feature_values),
    )


def _pair_metrics(
    frame: FrameData,
    mol_a: MoleculeRecord,
    mol_b: MoleculeRecord,
    geom_a: _MoleculeGeometry,
    geom_b: _MoleculeGeometry,
    contact_cutoff_nm: float,
) -> dict[str, float]:
    com_delta = minimum_image_delta(geom_a.center, geom_b.center, frame.box_nm)
    com_distance = norm(com_delta)

    min_distance = float("inf")
    contact_count = 0
    for atom_i in mol_a.heavy_atom_indices:
        pos_i = frame.positions_nm[atom_i]
        for atom_j in mol_b.heavy_atom_indices:
            distance = norm(minimum_image_delta(pos_i, frame.positions_nm[atom_j], frame.box_nm))
            if distance < min_distance:
                min_distance = distance
            if distance <= contact_cutoff_nm:
                contact_count += 1

    normal_alignment = abs(dot(geom_a.normal, geom_b.normal))
    avg_normal = normalize(add(geom_a.normal, geom_b.normal))
    if norm(avg_normal) == 0.0:
        avg_normal = geom_a.normal
    normal_separation = abs(dot(com_delta, avg_normal))
    lateral_sq = max(0.0, com_distance * com_distance - normal_separation * normal_separation)
    lateral_separation = math.sqrt(lateral_sq)

    parallel = smoothstep(0.55, 0.90, normal_alignment)
    vertical = 1.0 - min(abs(normal_separation - 0.36) / 0.22, 1.0)
    lateral = 1.0 - min(lateral_separation / 0.85, 1.0)
    pi_score = max(0.0, parallel * vertical * lateral)

    return {
        "com_distance": com_distance,
        "min_distance": min_distance,
        "contact_count": float(contact_count),
        "pi_score": pi_score,
    }


def _is_connected_cluster(
    mol_indices: tuple[int, ...],
    pair_metrics: Sequence[dict[str, float]],
    contact_cutoff_nm: float,
) -> bool:
    parent = {index: index for index in mol_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for (i, j), metrics in zip(itertools.combinations(mol_indices, 2), pair_metrics):
        if metrics["min_distance"] <= contact_cutoff_nm or metrics["com_distance"] <= 1.20:
            union(i, j)

    roots = {find(index) for index in mol_indices}
    return len(roots) == 1


def _select_diverse_candidates(
    candidates: Sequence[_Candidate],
    n_samples: int,
) -> list[_Candidate]:
    ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    selected = [ranked[0]]
    remaining = ranked[1:]

    while remaining and len(selected) < n_samples:
        best_index = 0
        best_value = -float("inf")
        for index, candidate in enumerate(remaining):
            diversity = min(_feature_distance(candidate.feature, item.feature) for item in selected)
            value = 0.68 * candidate.score + 0.32 * min(diversity, 1.0)
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(remaining.pop(best_index))

    return selected


def _feature_distance(a: Sequence[float], b: Sequence[float]) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    total = 0.0
    for i in range(length):
        total += (a[i] - b[i]) ** 2
    return math.sqrt(total / length)


def _molecule_geometries(
    frame: FrameData,
    molecules: Sequence[MoleculeRecord],
) -> list[_MoleculeGeometry]:
    geometries: list[_MoleculeGeometry] = []
    for molecule in molecules:
        unwrapped_positions = _unwrapped_molecule_positions(
            frame, molecule, molecule.heavy_atom_indices
        )
        coords = [unwrapped_positions[i] for i in molecule.heavy_atom_indices]
        center = mean_vector(coords)
        normal = _principal_plane_normal(coords, center)
        geometries.append(_MoleculeGeometry(center=center, normal=normal))
    return geometries


def _unwrapped_molecule_positions(
    frame: FrameData,
    molecule: MoleculeRecord,
    atom_indices: Sequence[int],
) -> dict[int, Vector]:
    anchor_index = molecule.atom_indices[0]
    anchor = frame.positions_nm[anchor_index]
    return {
        atom_index: add(anchor, minimum_image_delta(anchor, frame.positions_nm[atom_index], frame.box_nm))
        for atom_index in atom_indices
    }


def _principal_plane_normal(coords: Sequence[Vector], center: Vector) -> Vector:
    if len(coords) < 3:
        return (0.0, 0.0, 1.0)

    covariance = [[0.0, 0.0, 0.0] for _ in range(3)]
    for coord in coords:
        shifted = sub(coord, center)
        for i in range(3):
            for j in range(3):
                covariance[i][j] += shifted[i] * shifted[j]
    scale_factor = 1.0 / len(coords)
    for i in range(3):
        for j in range(3):
            covariance[i][j] *= scale_factor

    eigenvalues, eigenvectors = _jacobi_eigen_symmetric_3x3(covariance)
    min_index = min(range(3), key=lambda i: eigenvalues[i])
    return normalize(
        (eigenvectors[0][min_index], eigenvectors[1][min_index], eigenvectors[2][min_index])
    )


def _jacobi_eigen_symmetric_3x3(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _ in range(24):
        p, q = 0, 1
        max_value = abs(a[p][q])
        for i, j in ((0, 2), (1, 2)):
            if abs(a[i][j]) > max_value:
                p, q = i, j
                max_value = abs(a[i][j])
        if max_value < 1e-12:
            break

        if abs(a[p][p] - a[q][q]) < 1e-12:
            angle = math.pi / 4.0
        else:
            angle = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        c = math.cos(angle)
        s = math.sin(angle)

        app = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q]
        aqq = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q]
        a[p][q] = 0.0
        a[q][p] = 0.0
        a[p][p] = app
        a[q][q] = aqq

        for r in range(3):
            if r not in (p, q):
                arp = c * a[r][p] - s * a[r][q]
                arq = s * a[r][p] + c * a[r][q]
                a[r][p] = a[p][r] = arp
                a[r][q] = a[q][r] = arq

        for r in range(3):
            vrp = c * v[r][p] - s * v[r][q]
            vrq = s * v[r][p] + c * v[r][q]
            v[r][p] = vrp
            v[r][q] = vrq

    return [a[i][i] for i in range(3)], v


def _write_metadata(output_dir: Path, samples: Sequence[ClusterSample]) -> None:
    json_path = output_dir / "samples.json"
    csv_path = output_dir / "samples.csv"

    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump([asdict(sample) for sample in samples], handle, indent=2)
        handle.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ClusterSample.__dataclass_fields__))
        writer.writeheader()
        for sample in samples:
            row = asdict(sample)
            row["molecule_resids"] = " ".join(str(item) for item in sample.molecule_resids)
            writer.writerow(row)


def add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vector, factor: float) -> Vector:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vector) -> Vector:
    value = norm(a)
    if value == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / value, a[1] / value, a[2] / value)


def mean_vector(vectors: Iterable[Vector]) -> Vector:
    total = (0.0, 0.0, 0.0)
    count = 0
    for vector in vectors:
        total = add(total, vector)
        count += 1
    if count == 0:
        return total
    return scale(total, 1.0 / count)


def minimum_image_delta(origin: Vector, target: Vector, box: Vector) -> Vector:
    delta = sub(target, origin)
    corrected = []
    for value, length in zip(delta, box):
        if length > 0.0:
            value -= round(value / length) * length
        corrected.append(value)
    return (corrected[0], corrected[1], corrected[2])


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

"""Composition-stratified sampling shared by GRO/XTC and MD samplers."""
from __future__ import annotations

import itertools
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path


def validate_sampling(mode, solute_resnames, solvent_resnames, weights=None, molecules=None):
    if mode not in ("film", "solution"):
        raise ValueError("sampling mode must be 'film' or 'solution'")
    if mode == "film":
        return
    solute, solvent = set(solute_resnames), set(solvent_resnames)
    if not solute or not solvent or solute & solvent:
        raise ValueError("solution requires nonempty, disjoint solute_resnames and solvent_resnames")
    weights = {0: 0.25, 1: 0.75} if weights is None else weights
    if not weights or any(int(k) != k or k < 0 or not math.isfinite(v) or v < 0 for k, v in weights.items()) or not any(weights.values()):
        raise ValueError("solution_solute_weights must contain nonnegative integer counts and finite nonnegative weights with a positive total")
    if molecules is not None:
        present = {m.resname for m in molecules}
        if present - solute - solvent or (solute | solvent) - present:
            raise ValueError(f"solution residue names do not match GRO: present={sorted(present)}, configured={sorted(solute | solvent)}")


def composition_weights(cluster_size, weights):
    weights = {0: 0.25, 1: 0.75} if weights is None else weights
    applicable = {k: v for k, v in weights.items() if k <= cluster_size and v > 0}
    if not applicable:
        raise ValueError(f"No positive solution composition weights for cluster size {cluster_size}")
    return applicable


def _contact_neighbors(frame, molecules, cutoff):
    """Periodic cell list of heavy atoms; avoid all-pairs solvent searches."""
    from .cluster_sampling import minimum_image_delta, norm
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("solution contact cutoff must be finite and positive")
    if any(not math.isfinite(v) or v <= 0 for v in frame.box_nm):
        raise ValueError("solution sampling requires positive finite box lengths")
    shape = tuple(max(1, int(length / cutoff)) for length in frame.box_nm)
    cells = defaultdict(list)
    for molecule_id, molecule in enumerate(molecules):
        for atom_id in molecule.heavy_atom_indices:
            position = frame.positions_nm[atom_id]
            cell = tuple(min(shape[d] - 1, int((position[d] % frame.box_nm[d]) /
                         frame.box_nm[d] * shape[d])) for d in range(3))
            cells[cell].append((molecule_id, atom_id))
    neighbors = [dict() for _ in molecules]
    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    for cell, atoms in cells.items():
        nearby = {tuple((cell[d] + offset[d]) % shape[d] for d in range(3)) for offset in offsets}
        for other_cell in sorted(nearby):
            if other_cell < cell:
                continue
            for mol_a, atom_a in atoms:
                for mol_b, atom_b in cells.get(other_cell, ()):
                    if mol_a == mol_b or (other_cell == cell and atom_b <= atom_a):
                        continue
                    distance = norm(minimum_image_delta(frame.positions_nm[atom_a],
                                                       frame.positions_nm[atom_b], frame.box_nm))
                    if distance <= cutoff and distance < neighbors[mol_a].get(mol_b, float("inf")):
                        neighbors[mol_a][mol_b] = distance
                        neighbors[mol_b][mol_a] = distance
    return neighbors


def solution_candidates(frame, molecules, cluster_size, neighbor_pool, cutoff, solute_resnames, weights, rng, seed_limit=None):
    """Yield connected, seed-centered clusters by solute count.

    All solutes are seeds for mixed clusters. The optional seed limit only
    limits solvent-only seeds. Neighbors must contact the seed in heavy atoms.
    """
    solutes = {i for i, m in enumerate(molecules) if m.resname in solute_resnames}
    neighbors = _contact_neighbors(frame, molecules, cutoff)
    seen = set()
    for count in composition_weights(cluster_size, weights):
        seeds = sorted(solutes if count else set(range(len(molecules))) - solutes)
        if count == 0 and seed_limit is not None and len(seeds) > seed_limit:
            seeds = sorted(rng.sample(seeds, seed_limit))
        for seed in seeds:
            pools = {True: [], False: []}
            for other, distance in neighbors[seed].items():
                pools[other in solutes].append((distance, other))
            need_a = count - int(seed in solutes)
            need_b = cluster_size - count - int(seed not in solutes)
            a = [i for _, i in sorted(pools[True])[:max(neighbor_pool, need_a)]]
            b = [i for _, i in sorted(pools[False])[:max(neighbor_pool, need_b)]]
            for aa in itertools.combinations(a, need_a):
                for bb in itertools.combinations(b, need_b):
                    indices = tuple(sorted((seed, *aa, *bb)))
                    if indices not in seen:
                        seen.add(indices)
                        yield indices


def select_solution(candidates, molecules, solute_resnames, cluster_size, weights, n_samples, selector, output_dir, score_group=None):
    weights = composition_weights(cluster_size, weights)
    total = sum(weights.values())
    exact = {k: n_samples * v / total for k, v in weights.items()}
    quotas = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - quotas[k]), k))[:n_samples - sum(quotas.values())]:
        quotas[k] += 1
    selected, report = [], {}
    for k, quota in sorted(quotas.items()):
        group = [c for c in candidates if sum(molecules[i].resname in solute_resnames for i in c.mol_indices) == k]
        if score_group and group:
            score_group(group)
        chosen = selector(group, quota) if group and quota else []
        selected.extend(chosen)
        report[str(k)] = {"requested": quota, "available": len(group), "selected": len(chosen), "shortfall": quota - len(chosen)}
        if len(chosen) < quota:
            warnings.warn(f"solution: {k} solute + {cluster_size-k} solvent: requested {quota}, selected {len(chosen)}; quota is not reassigned", stacklevel=2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sampling_report.json").write_text(json.dumps({
        "mode": "solution", "cluster_size": cluster_size, "compositions_by_solute_count": report,
        "samples": [{"sample_id": n, "resnames": [molecules[i].resname for i in c.mol_indices],
                     "composition": dict(Counter(molecules[i].resname for i in c.mol_indices)),
                     "natoms": sum(len(molecules[i].atom_indices) for i in c.mol_indices)}
                    for n, c in enumerate(selected, 1)]
    }, indent=2) + "\n", encoding="utf-8")
    return selected


def add_sampling_arguments(parser):
    parser.add_argument("--sampling-mode", choices=("film", "solution"), default="film")
    parser.add_argument("--solute-resnames", nargs="+", default=[])
    parser.add_argument("--solvent-resnames", nargs="+", default=[])
    parser.add_argument("--solution-solute-weights", default="0:0.25,1:0.75",
                        help="Solute-count:weight pairs, e.g. 0:0.2,1:0.7,2:0.1")


def sampling_arguments(args):
    weights = {}
    for item in args.solution_solute_weights.split(","):
        count, weight = item.split(":")
        weights[int(count)] = float(weight)
    return dict(sampling_mode=args.sampling_mode, solute_resnames=args.solute_resnames,
                solvent_resnames=args.solvent_resnames, solution_solute_weights=weights)

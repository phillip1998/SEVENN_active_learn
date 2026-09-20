"""Analysis-only, molecule-whole XTC export from LAMMPS metal-unit dumps."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path


def load_analysis_topology(topology_path: str | Path, reference_gro: str | Path):
    """Require matching atom order and complete intramolecular bonds; never guess."""
    import numpy as np
    import MDAnalysis as mda
    from MDAnalysis.exceptions import NoDataError

    topology = mda.Universe(str(topology_path))
    reference = mda.Universe(str(reference_gro))
    try:
        if len(topology.atoms) != len(reference.atoms):
            raise ValueError("Analysis topology atom count differs from reference GRO (check virtual sites)")
        for attribute in ("names", "resnames"):
            expected = getattr(reference.atoms, attribute)
            actual = [str(v)[:5] for v in getattr(topology.atoms, attribute)]
            if not np.array_equal(actual, expected):
                raise ValueError(f"Analysis topology {attribute}/atom order differs from reference GRO")
        if not np.array_equal(topology.atoms.resindices, reference.atoms.resindices):
            raise ValueError("Analysis topology molecule/residue mapping differs from reference GRO")
        try:
            topology.atoms.bonds
        except NoDataError:
            if any(len(residue.atoms) > 1 for residue in topology.residues):
                raise ValueError("Analysis topology needs explicit bonds (use a matching TPR); GRO alone is insufficient") from None
            topology.add_TopologyAttr("bonds", [])
        for fragment in topology.atoms.fragments:
            if len(fragment.residues) != 1:
                raise ValueError("Analysis export currently requires one molecule per residue")
        for residue in topology.residues:
            if len(residue.atoms.fragments) != 1:
                raise ValueError(f"Incomplete bonds in residue {residue.resname}: cannot make the molecule whole")
        return topology
    except Exception:
        if hasattr(topology, "trajectory"):
            topology.trajectory.close()
        raise
    finally:
        reference.trajectory.close()


def export_whole_trajectory(
    dump_path: str | Path, reference_gro: str | Path, topology_path: str | Path,
    output_dir: str | Path, *, timestep_ps: float = 0.001,
) -> dict:
    """Stream all frames, reconstruct bonded molecules and wrap their centers.

    Coordinates are Angstrom in LAMMPS metal dumps; MDAnalysis converts them to
    nm on XTC output. Times are local LAMMPS steps * timestep_ps, in ps.
    """
    import numpy as np
    import MDAnalysis as mda

    if not math.isfinite(timestep_ps) or timestep_ps <= 0:
        raise ValueError("timestep_ps must be finite and positive")
    topology = load_analysis_topology(topology_path, reference_gro)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        topology.load_new(str(dump_path), format="LAMMPSDUMP", dt=timestep_ps,
                          additional_columns=["id"], lammps_coordinate_convention="auto")
        expected_ids = np.arange(1, len(topology.atoms) + 1)
        count, first_time, last_time, previous_step = 0, None, None, None
        # Publish only fully written outputs; failed exports leave existing outputs intact.
        with tempfile.TemporaryDirectory(prefix=".whole_export_", dir=output_dir) as temporary:
            stage = Path(temporary)
            with mda.Writer(str(stage / "whole.xtc"), n_atoms=len(topology.atoms)) as writer:
                for ts in topology.trajectory:
                    if "id" not in ts.data or not np.array_equal(ts.data["id"], expected_ids):
                        raise ValueError("Dump must contain each reference atom ID 1..N exactly once")
                    if ts.dimensions is None or not np.isfinite(ts.dimensions).all() or np.any(ts.dimensions[:3] <= 0):
                        raise ValueError("Dump has invalid periodic cell")
                    if not np.isfinite(ts.positions).all():
                        raise ValueError("Dump contains non-finite coordinates")
                    step = int(ts.data["step"])
                    if previous_step is not None and step <= previous_step:
                        raise ValueError("Dump steps must increase strictly within one MD segment")
                    previous_step = step
                    ts.time = step * timestep_ps
                    topology.atoms.unwrap(compound="fragments", reference=None)
                    topology.atoms.wrap(compound="fragments", center="cog")
                    if count == 0:
                        # GRO is the first corrected frame and preserves residue/atom names.
                        topology.atoms.write(str(stage / "whole.gro"))
                        first_time = ts.time
                    writer.write(topology.atoms)
                    last_time = ts.time
                    count += 1
            if not count:
                raise ValueError("No frames available for XTC export")
            report = {
                "dump": str(Path(dump_path).resolve()), "topology": str(Path(topology_path).resolve()),
                "frames": count, "atoms": len(topology.atoms), "timestep_ps": timestep_ps,
                "first_time_ps": first_time, "last_time_ps": last_time,
                "pbc": "bond-based whole molecules, fragment center of geometry wrapped into box",
                "time_origin": "local MD segment (not concatenated across iterations)",
            }
            (stage / "whole_export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for name in ("whole.xtc", "whole.gro", "whole_export.json"):
                (stage / name).replace(output_dir / name)
        return report
    finally:
        if hasattr(topology, "trajectory"):
            topology.trajectory.close()

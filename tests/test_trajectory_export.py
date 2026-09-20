import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import MDAnalysis as mda

from mlp_md_loop.trajectory_export import export_whole_trajectory, load_analysis_topology
from mlp_md_loop.active_learning_loop import LoopConfig, LoopState, MdConfig, run_one_iteration, load_loop_config


class WholeTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gro = self.root / "mixture.gro"
        self.topology = self.root / "bonded.pdb"
        self.dump = self.root / "dump.lammpstrj"
        u = mda.Universe.empty(12, n_residues=6, atom_resindex=np.repeat(np.arange(6), 2), trajectory=True)
        u.add_TopologyAttr("names", ["O1", "H1"] * 6)
        u.add_TopologyAttr("resnames", ["A"] + ["B"] * 5)
        u.add_TopologyAttr("resids", np.arange(1, 7))
        u.add_TopologyAttr("elements", ["O", "H"] * 6)
        u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(0, 12, 2)])
        self.coordinates = np.array([[.2 + i * .8, 2, 2] for i in range(6) for _ in range(2)])
        self.coordinates[1::2, 0] -= .9
        u.atoms.positions = self.coordinates % 10
        u.dimensions = [10, 10, 10, 90, 90, 90]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u.atoms.write(str(self.gro))
            u.atoms.write(str(self.topology), bonds="all")
        self.write_dump()

    def write_dump(self, bad_ids=False, nonfinite=False):
        frames = []
        for frame, (step, length) in enumerate([(0, 10), (125, 12)]):
            coordinates = (self.coordinates + frame * .4) % length
            if nonfinite and frame == 1:
                coordinates[0, 0] = float("nan")
            rows = []
            # Unsorted dump rows must still match topology order on export.
            for i in reversed(range(12)):
                atom_id = 1 if bad_ids and i == 11 else i + 1
                x, y, z = coordinates[i]
                rows.append(f"{atom_id} 1 {x} {y} {z} 0 0 0")
            frames.append(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n12\n"
                f"ITEM: BOX BOUNDS pp pp pp\n0 {length}\n0 {length}\n0 {length}\n"
                "ITEM: ATOMS id type x y z fx fy fz\n" + "\n".join(rows) + "\n")
        self.dump.write_text("".join(frames))

    def export(self, **kwargs):
        return export_whole_trajectory(self.dump, self.gro, self.topology,
                                        self.root / "output", timestep_ps=.002, **kwargs)

    def test_whole_molecules_units_times_and_cells(self):
        report = self.export()
        self.assertEqual(report["frames"], 2)
        self.assertAlmostEqual(report["last_time_ps"], .25)
        u = mda.Universe(str(self.root / "output/whole.gro"), str(self.root / "output/whole.xtc"))
        try:
            for i, ts in enumerate(u.trajectory):
                self.assertAlmostEqual(ts.time, i * .25, places=5)
                np.testing.assert_allclose(ts.dimensions[:3], [10 + i * 2] * 3, atol=.001)
                for residue in u.residues:
                    # No minimum image: the stored bond itself must be whole.
                    self.assertAlmostEqual(np.linalg.norm(residue.atoms.positions[0] - residue.atoms.positions[1]), .9, places=2)
                    center = residue.atoms.positions.mean(axis=0)
                    self.assertTrue(np.all(center >= 0) and np.all(center < ts.dimensions[:3]))
            u.trajectory[0]
            reference = mda.Universe(str(self.root / "output/whole.gro"))
            np.testing.assert_allclose(reference.atoms.positions, u.atoms.positions, atol=.011)
            reference.trajectory.close()
        finally:
            u.trajectory.close()

    def test_missing_bonds_and_bad_topology_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit bonds"):
            load_analysis_topology(self.gro, self.gro)
        self.topology.write_text(self.topology.read_text().replace(" O1 ", " N1 "))
        with self.assertRaisesRegex(ValueError, "names/atom order"):
            load_analysis_topology(self.topology, self.gro)

    def test_bad_dump_ids_and_failed_export_not_published(self):
        self.write_dump(bad_ids=True)
        with self.assertRaisesRegex(ValueError, "atom ID"):
            self.export()
        self.assertFalse((self.root / "output/whole.xtc").exists())
        self.write_dump(nonfinite=True)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.export()
        self.assertFalse((self.root / "output/whole.gro").exists())

    def test_existing_dump_cli(self):
        script = Path(__file__).resolve().parents[1] / "scripts/export_whole_trajectory.py"
        result = subprocess.run([sys.executable, str(script), "--dump", str(self.dump),
            "--reference-gro", str(self.gro), "--topology", str(self.topology),
            "--output-dir", str(self.root / "cli"), "--timestep-ps", ".002"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["frames"], 2)

    def test_loop_integration_and_analysis_failure_preserves_md(self):
        config = LoopConfig(execute_md=True, md=MdConfig(gro_path=str(self.gro),
            pair_style="e3gnn", model_path="model.pt", run_command="fake-md",
            export_whole_xtc=True, analysis_topology_path=str(self.topology)))
        def execute(command, cwd, progress):
            (cwd / "final.data").write_text((cwd / "system.data").read_text() +
                "\nVelocities\n\n" + "".join(f"{i} 0 0 0\n" for i in range(1, 13)))
            (cwd / "md.complete").write_text("MLP_MD_COMPLETE\n")
            (cwd / "dump.sevennet.lammpstrj").write_bytes(self.dump.read_bytes())
        work = self.root / "loop"
        with patch("mlp_md_loop.active_learning_loop._run_command", side_effect=execute):
            run_one_iteration(config, LoopState(), work, dry_run=False, progress=False)
            self.assertTrue((work / "iter_0000/md/whole.xtc").is_file())
            self.write_dump(bad_ids=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_one_iteration(config, LoopState(), self.root / "failed_export", dry_run=False, progress=False)
            self.assertTrue(any("export failed" in str(w.message) for w in caught))
        md = self.root / "failed_export/iter_0000/md"
        self.assertTrue((md / "md.complete").exists())
        self.assertTrue((md / "whole_export_error.json").exists())

    def test_toml_export_options(self):
        config = self.root / "loop.toml"
        config.write_text('[md]\nexport_whole_xtc=true\nanalysis_topology_path="mixture.tpr"\n')
        md = load_loop_config(config).md
        self.assertTrue(md.export_whole_xtc)
        self.assertEqual(md.analysis_topology_path, "mixture.tpr")


if __name__ == "__main__":
    unittest.main()

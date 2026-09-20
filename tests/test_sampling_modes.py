import json
import random
import subprocess
import sys
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mlp_md_loop.cluster_sampling import read_gro, sample_dft_clusters
from mlp_md_loop.active_learning import sample_active_learning_clusters
from mlp_md_loop.active_learning_loop import (
    LoopConfig, LoopState, MdConfig, SamplingConfig, load_loop_config, run_one_iteration,
)
from mlp_md_loop.solution_sampling import solution_candidates, _contact_neighbors


class SamplingModesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gro = self.root / "mixed.gro"
        # Different molecule sizes, shared C/O/H elements, two solutes/four solvents.
        rows, self.positions = [], []
        for residue in range(6):
            name = "A" if residue < 2 else "B"
            names = ["C1", "H1"] if name == "A" else ["O1"]
            for j, atomname in enumerate(names):
                pos = (1 + residue % 3 * .25, 1 + residue // 3 * .25, 1 + j * .1)
                self.positions.append(pos)
                rows.append(f"{residue+1:5d}{name:<5}{atomname:>5}{len(rows)+1:5d}"
                            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}")
        self.gro.write_text("mixture\n" + str(len(rows)) + "\n" + "\n".join(rows) + "\n4 4 4\n")
        self.dump = self.root / "dump.lammpstrj"
        self.dump.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n8\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 40\n0 40\n0 40\n"
            "ITEM: ATOMS id type x y z fx fy fz\n" + "\n".join(
                f"{i} 1 {x*10} {y*10} {z*10} {i*.1} 0 0"
                for i, (x, y, z) in enumerate(self.positions, 1)) + "\n")
        self.options = dict(sampling_mode="solution", solute_resnames=["A"], solvent_resnames=["B"])

    def sample(self, active=False, **kwargs):
        options = dict(n_samples=4, output_dir=self.root / ("active" if active else "initial"))
        options.update(kwargs)
        if active:
            return sample_active_learning_clusters(self.dump, self.gro, **options)
        return sample_dft_clusters(self.gro, **options)

    def test_film_default_unchanged(self):
        for active in (False, True):
            default = self.sample(active)
            explicit = self.sample(active, sampling_mode="film")
            self.assertEqual(default, explicit)

    def test_mixed_composition_quotas_and_xyz(self):
        for active in (False, True):
            for size in (2, 3):
                samples = self.sample(active, cluster_size=size, **self.options)
                self.assertEqual(len(samples), 4)
                counts = [sum(r <= 2 for r in s.molecule_resids) for s in samples]
                self.assertEqual(counts, [0, 1, 1, 1])
                for sample, solutes in zip(samples, counts):
                    self.assertEqual(int(Path(sample.xyz_path).read_text().splitlines()[0]), size + solutes)
                report = json.loads((Path(samples[0].xyz_path).parent / "sampling_report.json").read_text())
                self.assertEqual(report["compositions_by_solute_count"]["1"]["shortfall"], 0)
                self.assertEqual(report["samples"][1]["composition"]["A"], 1)

    def test_explicit_two_solutes(self):
        for active in (False, True):
            samples = self.sample(active, n_samples=1, solution_solute_weights={2: 1}, **self.options)
            self.assertEqual(samples[0].molecule_resids, [1, 2])

    def test_shortfall_not_reallocated(self):
        for active in (False, True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                samples = self.sample(active, n_samples=4, solution_solute_weights={2: 1}, **self.options)
            self.assertEqual(len(samples), 1)
            self.assertTrue(caught)
            report = json.loads((Path(samples[0].xyz_path).parent / "sampling_report.json").read_text())
            self.assertEqual(report["compositions_by_solute_count"]["2"]["shortfall"], 3)

    def test_no_candidate_writes_empty_metadata(self):
        for active in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = self.sample(active, cluster_size=3, solution_solute_weights={3: 1}, **self.options)
            self.assertEqual(samples, [])

    def test_invalid_config(self):
        for active in (False, True):
            for extra in [dict(sampling_mode="invalid"), dict(solute_resnames=[]),
                          dict(solvent_resnames=["A"]), dict(solvent_resnames=["WRONG"]),
                          dict(solution_solute_weights={0: -1}), dict(solution_solute_weights={0: float("nan")})]:
                options = {**self.options, **extra}
                with self.assertRaises(ValueError):
                    self.sample(active, **options)

    def test_solute_seeds_survive_limit(self):
        samples = self.sample(True, n_samples=7, candidate_seeds_per_frame=1,
                              solution_solute_weights={1: 1}, **self.options)
        self.assertEqual(len(samples), 7)
        self.assertEqual({r for s in samples for r in s.molecule_resids if r <= 2}, {1, 2})

    def test_surface_contact_and_periodicity(self):
        _, molecules, frame = read_gro(self.gro)
        # Place the solute center far away but its first atom adjacent across PBC.
        frame.positions_nm[0] = (.05, 1, 1)
        frame.positions_nm[2] = (3.95, 1, 1)
        candidates = list(solution_candidates(frame, molecules, 2, 8, .2, ["A"],
                                              {2: 1}, random.Random(17)))
        self.assertIn((0, 1), candidates)

    def test_cell_list_matches_all_pairs(self):
        from mlp_md_loop.cluster_sampling import minimum_image_delta, norm
        _, molecules, frame = read_gro(self.gro)
        rng = random.Random(92)
        frame.positions_nm = [tuple(rng.uniform(-4, 8) for _ in range(3)) for _ in frame.positions_nm]
        for cutoff in (.2, 1.5, 5.0):
            neighbors = _contact_neighbors(frame, molecules, cutoff)
            for i, first in enumerate(molecules):
                for j, second in enumerate(molecules):
                    if i == j:
                        continue
                    distance = min(norm(minimum_image_delta(frame.positions_nm[a], frame.positions_nm[b], frame.box_nm))
                                   for a in first.heavy_atom_indices for b in second.heavy_atom_indices)
                    self.assertEqual(j in neighbors[i], distance <= cutoff)
                    if distance <= cutoff:
                        self.assertAlmostEqual(neighbors[i][j], distance)

    def test_standalone_cli(self):
        root = Path(__file__).resolve().parents[1]
        for script, inputs in [("sample_clusters.py", ["--gro", str(self.gro)]),
                               ("sample_active_learning.py", ["--reference-gro", str(self.gro), "--dump", str(self.dump)])]:
            output = self.root / script
            result = subprocess.run([sys.executable, str(root / "scripts" / script), *inputs,
                "--sampling-mode", "solution", "--solute-resnames", "A", "--solvent-resnames", "B",
                "--n-samples", "4", "--output-dir", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list((output / "2mol").glob("*.xyz"))), 4)

    def test_loop_initial_and_md_dispatch(self):
        _, _, frame = read_gro(self.gro)
        config = LoopConfig(sampling=SamplingConfig(mode="solution", solute_resnames=("A",),
            solvent_resnames=("B",), cluster_sizes=(2,), initial_weights={2: 1},
            mature_weights={2: 1}, initial_total=4),
            md=MdConfig(gro_path=str(self.gro), xtc_path="fixture.xtc"))
        work = self.root / "loop"
        with patch("mlp_md_loop.cluster_sampling._load_frames", return_value=[frame]):
            run_one_iteration(config, LoopState(), work, dry_run=True, progress=False)
        self.assertTrue((work / "iter_0000/samples/2mol/sampling_report.json").is_file())
        (work / "iter_0000/md/dump.sevennet.lammpstrj").write_text(self.dump.read_text())
        md_dir = work / "iter_0000/md"
        (md_dir / "final.data").write_text((md_dir / "system.data").read_text()
            + "\nVelocities\n\n" + "".join(f"{i} 0.1 0 0\n" for i in range(1, 9)))
        (md_dir / "md.complete").write_text("MLP_MD_COMPLETE\n")
        run_one_iteration(config, LoopState(iteration=1), work, dry_run=True, progress=False)
        self.assertTrue((work / "iter_0001/samples/2mol/sampling_report.json").is_file())
        # All compositions unavailable: report the shortfall, do not attempt GJF conversion.
        config = replace(config, sampling=replace(config.sampling, cluster_sizes=(3,),
            initial_weights={3: 1}, solution_solute_weights={3: 1}))
        with patch("mlp_md_loop.cluster_sampling._load_frames", return_value=[frame]), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_one_iteration(config, LoopState(), self.root / "empty", dry_run=True, progress=False)

    def test_toml_load(self):
        path = self.root / "config.toml"
        path.write_text('[sampling]\nmode="solution"\nsolute_resnames=["A"]\nsolvent_resnames=["B"]\nsolution_solute_weights={"0"=0.2,"1"=0.8}\n')
        config = load_loop_config(path)
        self.assertEqual(config.sampling.mode, "solution")
        self.assertEqual(config.sampling.solution_solute_weights, {0: .2, 1: .8})
        path.write_text("")
        self.assertEqual(load_loop_config(path).sampling.mode, "film")


if __name__ == "__main__":
    unittest.main()

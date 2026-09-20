import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mlp_md_loop.persistent_split import structure_id
from mlp_md_loop.sevennet_finetune import _write_train_valid_extxyz, _read_extxyz_blocks, prepare_sevennet_finetune
from mlp_md_loop.lammps import prepare_sevennet_lammps_from_gro, completed_md_data
from mlp_md_loop.active_learning_loop import LoopConfig, LoopState, MdConfig, run_one_iteration, _select_model_path


def geometry(i, source="original"):
    return f'2\nProperties=species:S:1:pos:R:3 source="{source}" energy=0 pbc="F F F"\nH 0 0 0\nH {0.7+i*.02:.8f} 0 0\n'


class PersistentSplitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "split_manifest.json"

    def split(self, blocks, name="iteration", **kwargs):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        source = directory / "dataset.extxyz"
        source.write_text("".join(blocks), encoding="utf-8")
        _write_train_valid_extxyz(source, directory / "train.extxyz", directory / "valid.extxyz",
            .2, 1, manifest_path=self.manifest, **kwargs)
        return {split: {structure_id(b) for b in _read_extxyz_blocks(directory / f"{split}.extxyz")}
                for split in ("train", "valid")}

    def test_growth_reordering_and_resume_keep_assignments(self):
        first = self.split([geometry(i) for i in range(10)], "first")
        second = self.split([geometry(i, "renamed") for i in reversed(range(12))], "second")
        self.assertTrue(first["train"] <= second["train"])
        self.assertTrue(first["valid"] <= second["valid"])
        self.assertFalse(first["train"] & second["valid"])
        self.assertEqual(second, self.split([geometry(i) for i in range(12)], "resume"))
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(len(manifest["assignments"]), 12)

    def test_duplicate_rigid_motion_and_atom_order(self):
        original = geometry(0)
        moved = '2\nProperties=species:S:1:pos:R:3 source="another"\nH 2 3.7 4\nH 2 3 4\n'
        self.assertEqual(structure_id(original), structure_id(moved))
        self.split([original, moved, geometry(1)])
        summary = json.loads((self.root / "iteration/split_summary.json").read_text())
        self.assertEqual(summary["duplicates_removed"], 1)
        self.assertEqual(summary["train"] + summary["valid"], 2)

    def test_legacy_import_and_conflict(self):
        old = self.root / "old"
        old.mkdir()
        (old / "train.extxyz").write_text(geometry(0))
        (old / "valid.extxyz").write_text(geometry(1))
        result = self.split([geometry(i) for i in range(6)], historical_dirs=(old,))
        self.assertIn(structure_id(geometry(0)), result["train"])
        self.assertIn(structure_id(geometry(1)), result["valid"])
        self.manifest.unlink()
        conflicting = self.root / "conflict"
        conflicting.mkdir()
        (conflicting / "train.extxyz").write_text(geometry(1))
        (conflicting / "valid.extxyz").write_text(geometry(0))
        with self.assertRaisesRegex(ValueError, "Historical train/validation leakage"):
            self.split([geometry(i) for i in range(6)], "new", historical_dirs=(old, conflicting))
        self.assertFalse(self.manifest.exists())

    def test_manifest_settings_cannot_silently_change(self):
        self.split([geometry(i) for i in range(5)])
        directory = self.root / "iteration"
        with self.assertRaisesRegex(ValueError, "fixes valid_ratio"):
            _write_train_valid_extxyz(directory / "dataset.extxyz", directory / "train.extxyz",
                directory / "valid.extxyz", .3, 1, self.manifest)

    def test_finetune_entrypoint_shared_manifest(self):
        source = self.root / "all.extxyz"
        first_assignments = None
        for n in (10, 12):
            source.write_text("".join(geometry(i) for i in range(n)))
            output = self.root / f"ft_{n}"
            prepare_sevennet_finetune(output, dataset_extxyz=source, data_divide_ratio=.2,
                                     split_manifest_path=self.manifest)
            current = json.loads(self.manifest.read_text())["assignments"]
            if first_assignments is not None:
                self.assertTrue(all(current[k] == v for k, v in first_assignments.items()))
            first_assignments = current


class MdContinuationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gro = self.root / "reference.gro"
        self.gro.write_text("test\n2\n" + "".join(
            f'{1:5d}{"MOL":<5}{name:>5}{i:5d}{x:8.3f}{1.:8.3f}{1.:8.3f}\n'
            for i, name, x in [(1, "O1", 1.), (2, "H1", 1.1)]) + "4 4 4\n")

    def finish(self, md_dir):
        content = (md_dir / "system.data").read_text()
        if "Velocities" not in content:
            content += "\nVelocities\n\n1 0.1 0.2 0.3\n2 -0.1 0.4 0.5\n"
        content = content.replace("40.00000000 xlo xhi", "42.00000000 xlo xhi")
        content = content.replace("10.00000000 10.00000000 10.00000000", "12.00000000 10.00000000 10.00000000")
        (md_dir / "final.data").write_text(content)
        (md_dir / "md.complete").write_text("MLP_MD_COMPLETE\n")

    def test_coordinates_box_velocities_and_new_model(self):
        first, second = self.root / "first", self.root / "second"
        prepare_sevennet_lammps_from_gro(self.gro, first, "old.pt")
        self.finish(first)
        final = completed_md_data(first, self.gro)
        prepare_sevennet_lammps_from_gro(self.gro, second, "new.pt", continuation_data_path=final)
        self.assertEqual((second / "system.data").read_bytes(), final.read_bytes())
        text = (second / "in.sevennet.lmp").read_text()
        self.assertIn("new.pt", text)
        self.assertNotIn("velocity        all create", text)
        self.assertIn("velocity        all create", (first / "in.sevennet.lmp").read_text())
        self.assertLess(text.index("run             "), text.index("write_data"))
        self.assertIn("write_data      final.data nocoeff", text)

    def test_missing_failed_and_changed_reference(self):
        first = self.root / "first"
        prepare_sevennet_lammps_from_gro(self.gro, first, "old.pt")
        with self.assertRaisesRegex(RuntimeError, "No completed MD state"):
            completed_md_data(first, self.gro)
        self.finish(first)
        (first / "md.complete").write_text("failed")
        with self.assertRaisesRegex(RuntimeError, "Invalid MD completion"):
            completed_md_data(first, self.gro)
        self.finish(first)
        self.gro.write_text(self.gro.read_text().replace("test", "changed"))
        with self.assertRaisesRegex(ValueError, "Reference GRO changed"):
            completed_md_data(first, self.gro)

    def test_missing_velocities_and_nonfinite_state_rejected(self):
        first = self.root / "first"
        prepare_sevennet_lammps_from_gro(self.gro, first, "old.pt")
        self.finish(first)
        data = first / "final.data"
        data.write_text(data.read_text().split("Velocities")[0])
        with self.assertRaisesRegex(ValueError, "missing velocities"):
            completed_md_data(first, self.gro)
        self.finish(first)
        data.write_text(data.read_text().replace("1 0.1 0.2 0.3", "1 nan 0.2 0.3"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            completed_md_data(first, self.gro)

    def test_loop_continues_and_does_not_fallback(self):
        config = LoopConfig(execute_md=True, md=MdConfig(gro_path=str(self.gro),
            pair_style="e3gnn", model_path="initial.pt", run_command="fake-md"))
        work = self.root / "loop"
        def execute(command, cwd, progress):
            self.assertEqual(command, "fake-md")
            self.finish(cwd)
        with patch("mlp_md_loop.active_learning_loop._run_command", side_effect=execute):
            run_one_iteration(config, LoopState(), work, dry_run=False, progress=False)
            final = (work / "iter_0000/md/final.data").read_bytes()
            run_one_iteration(config, LoopState(iteration=1), work, dry_run=False, progress=False)
        self.assertEqual((work / "iter_0001/md/system.data").read_bytes(), final)
        (work / "iter_0001/md/md.complete").unlink()
        with self.assertRaisesRegex(RuntimeError, "No completed MD state"):
            run_one_iteration(config, LoopState(iteration=2), work, dry_run=False, progress=False)

    def test_command_failure_cannot_leave_success_marker(self):
        config = LoopConfig(execute_md=True, md=MdConfig(gro_path=str(self.gro),
            pair_style="e3gnn", model_path="initial.pt", run_command="fake-md"))
        work = self.root / "loop"
        def fail(command, cwd, progress):
            self.finish(cwd)
            raise subprocess.CalledProcessError(1, command)
        with patch("mlp_md_loop.active_learning_loop._run_command", side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                run_one_iteration(config, LoopState(), work, dry_run=False, progress=False)
        self.assertFalse((work / "iter_0000/md/md.complete").exists())

    def test_latest_deployed_model_reused(self):
        model = self.root / "iter_0000/finetune/deployed_serial.pt"
        model.parent.mkdir(parents=True)
        model.write_text("model")
        config = LoopConfig(md=MdConfig(gro_path=str(self.gro), pair_style="e3gnn"))
        self.assertEqual(_select_model_path(config, self.root / "iter_0002"), str(model))


if __name__ == "__main__":
    unittest.main()

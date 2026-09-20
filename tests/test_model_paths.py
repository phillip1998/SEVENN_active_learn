import os
import tempfile
import unittest
from pathlib import Path

from mlp_md_loop.model_paths import resolve_checkpoint
from mlp_md_loop.sevennet_finetune import prepare_sevennet_finetune
from mlp_md_loop.active_learning_loop import (
    LoopConfig, MdConfig, FinetuneConfig, _select_finetune_pretrained, _prepare_md_model,
)


class ModelPathTest(unittest.TestCase):
    def test_relative_checkpoint_survives_iteration_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                os.chdir(root)
                checkpoint = root / "checkpoint_best_meta.pth"
                checkpoint.write_text("fixture")
                config = LoopConfig(finetune=FinetuneConfig(pretrained=checkpoint.name),
                    md=MdConfig(gro_path="unused.gro", model_path=checkpoint.name))
                selected = _select_finetune_pretrained(config=config, work_dir=root / "run", current_iteration=1)
                self.assertEqual(selected, str(checkpoint))
                dataset = root / "data.extxyz"
                dataset.write_text('2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nH 0.7 0 0\n')
                output = root / "run/iter_0001/finetune"
                setup = prepare_sevennet_finetune(output, dataset_extxyz=dataset, pretrained=checkpoint.name)
                self.assertEqual(setup.pretrained, str(checkpoint))
                text = (output / "make_input_finetune.py").read_text()
                self.assertIn(f"PRETRAINED = {str(checkpoint)!r}", text)
                md = root / "run/iter_0001/md"
                _prepare_md_model(config=config, iteration_dir=md.parent, md_dir=md, dry_run=True, progress=False)
                self.assertIn(checkpoint.name, (md / "deploy_model.sh").read_text())
                self.assertIn(str(root), (md / "deploy_model.sh").read_text())
                os.chdir(output)
                self.assertTrue(Path(selected).is_file())
            finally:
                os.chdir(previous)

    def test_aliases_and_missing_files(self):
        for alias in ("7net-omni", "7net-0", "7net-future"):
            self.assertEqual(resolve_checkpoint(alias), alias)
        with self.assertRaisesRegex(FileNotFoundError, "controller working directory"):
            resolve_checkpoint("definitely_missing_checkpoint.pth")


if __name__ == "__main__":
    unittest.main()

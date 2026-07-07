from __future__ import annotations

import shutil
import random
from dataclasses import dataclass
from pathlib import Path

from .dataset import (
    LabeledStructure,
    write_dataset_manifest,
    write_extxyz_dataset,
    write_sevennet_structure_list,
)
from .gaussian import parse_gaussian_logs_batch


@dataclass(frozen=True)
class SevenNetFineTuneSetup:
    output_dir: str
    input_yaml: str
    extxyz_path: str
    structure_list_path: str
    config_builder_script: str
    run_script: str
    n_structures: int | None
    pretrained: str
    modal: str | None


def prepare_sevennet_finetune(
    output_dir: str | Path,
    logs_path: str | Path | None = None,
    dataset_extxyz: str | Path | None = None,
    pretrained: str = "7net-0",
    modal: str | None = None,
    epoch: int = 100,
    batch_size: int = 1,
    learning_rate: float = 5.0e-4,
    force_loss_weight: float = 1.0,
    data_divide_ratio: float = 0.1,
    best_metric: str = "Force_RMSE",
    huber_delta: float = 0.1,
    train_shift_scale: bool = True,
    train_denominator: bool = False,
    require_normal_termination: bool = False,
    sevennet_label: str = "gaussian_finetune",
) -> SevenNetFineTuneSetup:
    """Prepare a SevenNet pretrained-model fine-tuning work directory.

    Exactly one of ``logs_path`` or ``dataset_extxyz`` must be provided. Gaussian
    logs are parsed into SevenNet/ASE-readable extxyz. Existing extxyz datasets
    are copied into the fine-tuning directory to keep the YAML paths portable on
    a training server.
    """

    if (logs_path is None) == (dataset_extxyz is None):
        raise ValueError("Provide exactly one of logs_path or dataset_extxyz")
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= data_divide_ratio < 1.0:
        raise ValueError("data_divide_ratio must satisfy 0 <= ratio < 1")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")

    modal = modal or _default_modal_for_pretrained(pretrained)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extxyz_path = output / "fine_tuning_set.extxyz"
    train_extxyz_path = output / "train.extxyz"
    valid_extxyz_path = output / "valid.extxyz"
    structure_list_path = output / "structure_list"

    n_structures: int | None
    if logs_path is not None:
        structures = parse_gaussian_logs_batch(
            logs_path,
            require_normal_termination=require_normal_termination,
        )
        write_extxyz_dataset(extxyz_path, structures)
        write_dataset_manifest(output / "dataset_manifest.tsv", structures)
        n_structures = len(structures)
    else:
        source = Path(dataset_extxyz)  # type: ignore[arg-type]
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() != extxyz_path.resolve():
            shutil.copyfile(source, extxyz_path)
        n_structures = count_extxyz_structures(extxyz_path)

    has_validset = _write_train_valid_extxyz(
        extxyz_path,
        train_extxyz_path,
        valid_extxyz_path,
        valid_ratio=data_divide_ratio,
        seed=1,
    )

    write_sevennet_structure_list(
        structure_list_path,
        extxyz_path,
        label=sevennet_label,
    )

    input_yaml = output / "input_finetune.yaml"
    input_yaml.write_text(
        _fine_tune_yaml(
            pretrained=pretrained,
            modal=modal,
            epoch=epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            force_loss_weight=force_loss_weight,
            data_divide_ratio=data_divide_ratio,
            best_metric=best_metric,
            huber_delta=huber_delta,
            has_validset=has_validset,
            train_shift_scale=train_shift_scale,
            train_denominator=train_denominator,
        ),
        encoding="utf-8",
        newline="\n",
    )

    config_builder_script = output / "make_input_finetune.py"
    config_builder_script.write_text(
        _config_builder_script_text(
            pretrained=pretrained,
            modal=modal,
            epoch=epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            force_loss_weight=force_loss_weight,
            data_divide_ratio=data_divide_ratio,
            best_metric=best_metric,
            huber_delta=huber_delta,
            has_validset=has_validset,
            train_shift_scale=train_shift_scale,
            train_denominator=train_denominator,
        ),
        encoding="utf-8",
        newline="\n",
    )

    run_script = output / "run_finetune.sh"
    run_script.write_text(
        _run_script_text(),
        encoding="utf-8",
        newline="\n",
    )

    readme = output / "README_finetune.md"
    readme.write_text(
        _readme_text(pretrained=pretrained),
        encoding="utf-8",
        newline="\n",
    )

    return SevenNetFineTuneSetup(
        output_dir=str(output),
        input_yaml=str(input_yaml),
        extxyz_path=str(extxyz_path),
        structure_list_path=str(structure_list_path),
        config_builder_script=str(config_builder_script),
        run_script=str(run_script),
        n_structures=n_structures,
        pretrained=pretrained,
        modal=modal,
    )


def count_extxyz_structures(path: str | Path) -> int:
    """Count structures in a simple extxyz/xyz trajectory file."""

    path = Path(path)
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            stripped = first.strip()
            if not stripped:
                continue
            try:
                natoms = int(stripped)
            except ValueError as exc:
                raise ValueError(f"{path} has an invalid atom count line: {stripped!r}") from exc
            comment = handle.readline()
            if not comment:
                raise ValueError(f"{path} ended after atom count for structure {count + 1}")
            for _ in range(natoms):
                if not handle.readline():
                    raise ValueError(f"{path} ended inside structure {count + 1}")
            count += 1
    return count


def _read_extxyz_blocks(path: Path) -> list[str]:
    blocks: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            if not first.strip():
                continue
            try:
                natoms = int(first.strip())
            except ValueError as exc:
                raise ValueError(f"{path} has an invalid atom count line: {first.strip()!r}") from exc
            lines = [first]
            comment = handle.readline()
            if not comment:
                raise ValueError(f"{path} ended after atom count for structure {len(blocks) + 1}")
            lines.append(comment)
            for _ in range(natoms):
                atom_line = handle.readline()
                if not atom_line:
                    raise ValueError(f"{path} ended inside structure {len(blocks) + 1}")
                lines.append(atom_line)
            blocks.append("".join(lines))
    return blocks


def _write_train_valid_extxyz(
    source: Path,
    train_path: Path,
    valid_path: Path,
    valid_ratio: float,
    seed: int,
) -> bool:
    blocks = _read_extxyz_blocks(source)
    if len(blocks) < 2 or valid_ratio <= 0.0:
        shutil.copyfile(source, train_path)
        valid_path.write_text("", encoding="utf-8", newline="\n")
        return False

    n_valid = max(1, round(len(blocks) * valid_ratio))
    n_valid = min(n_valid, len(blocks) - 1)
    indices = list(range(len(blocks)))
    random.Random(seed).shuffle(indices)
    valid_indices = set(indices[:n_valid])

    train_blocks = [block for idx, block in enumerate(blocks) if idx not in valid_indices]
    valid_blocks = [block for idx, block in enumerate(blocks) if idx in valid_indices]
    train_path.write_text("".join(train_blocks), encoding="utf-8", newline="\n")
    valid_path.write_text("".join(valid_blocks), encoding="utf-8", newline="\n")
    return True


def _fine_tune_yaml(
    pretrained: str,
    modal: str | None,
    epoch: int,
    batch_size: int,
    learning_rate: float,
    force_loss_weight: float,
    data_divide_ratio: float,
    best_metric: str,
    huber_delta: float,
    has_validset: bool,
    train_shift_scale: bool,
    train_denominator: bool,
) -> str:
    train_shift_scale_text = _yaml_bool(train_shift_scale)
    train_denominator_text = _yaml_bool(train_denominator)
    multimodal = modal is not None
    model_block = _model_block_for_pretrained(pretrained, train_shift_scale_text, train_denominator_text)
    train_modality_line = "    use_modality: true\n" if multimodal else ""
    data_modality_block = (
        f"""    use_modal_wise_shift: true
    use_modal_wise_scale: false

    load_trainset_path:
        - data_modality: '{modal}'
          file_list:
              - file: './train.extxyz'
          data_weight:
              energy: 1.0
              force: 1.0
              stress: 1.0
{_validset_yaml_block(modal, has_validset)}
"""
        if multimodal
        else _single_modal_dataset_yaml_block(has_validset)
    )
    return f"""# SevenNet fine-tuning input generated for the MLP-MD loop.
# This file is also regenerated by make_input_finetune.py on the training server.
# The server-side generation reads the actual checkpoint config, so changing
# --pretrained between single-modal and multi-modal models remains safe.

model:
{model_block}

train:
    random_seed: 1
    is_train_stress: false
    epoch: {epoch}

    loss: 'Huber'
    loss_param:
        delta: {huber_delta:.8g}

    optimizer: 'adam'
    optim_param:
        lr: {learning_rate:.8g}
    scheduler: 'exponentiallr'
    scheduler_param:
        gamma: 0.99

    force_loss_weight: {force_loss_weight:.8g}
    stress_loss_weight: 0.0
    best_metric: '{best_metric}'

    per_epoch: 10
    error_record:
        - ['Energy', 'RMSE']
        - ['Force', 'RMSE']
        - ['Energy', 'MAE']
        - ['Force', 'MAE']
        - ['TotalLoss', 'None']

    continue:
        reset_optimizer: true
        reset_scheduler: true
        reset_epoch: true
        checkpoint: '{pretrained}'
        use_statistic_values_of_checkpoint: true
{train_modality_line}

data:
    batch_size: {batch_size}
    data_divide_ratio: {data_divide_ratio:.8g}
    dataset_type: 'graph'

    data_format: 'ase'
    data_format_args: {{}}

{data_modality_block}"""


def _config_builder_script_text(
    pretrained: str,
    modal: str | None,
    epoch: int,
    batch_size: int,
    learning_rate: float,
    force_loss_weight: float,
    data_divide_ratio: float,
    best_metric: str,
    huber_delta: float,
    has_validset: bool,
    train_shift_scale: bool,
    train_denominator: bool,
) -> str:
    modal_literal = "None" if modal is None else repr(modal)
    return f'''#!/usr/bin/env python3
"""Build input_finetune.yaml from the actual SevenNet checkpoint config."""

from __future__ import annotations

from pathlib import Path

import yaml

import sevenn._keys as KEY
from sevenn.util import load_checkpoint


PRETRAINED = {pretrained!r}
REQUESTED_MODAL = {modal_literal}
TRAIN_DATASET = "./train.extxyz"
VALID_DATASET = "./valid.extxyz"
HAS_VALIDSET = {has_validset!r}
OUTPUT = "input_finetune.yaml"


def main() -> None:
    checkpoint = load_checkpoint(PRETRAINED)
    cfg = checkpoint.yaml_dict("continue")
    model = cfg.setdefault("model", {{}})
    train = cfg.setdefault("train", {{}})
    data = cfg.setdefault("data", {{}})

    use_modality = bool(train.get(KEY.USE_MODALITY, False))
    modal_map = checkpoint.config.get(KEY.MODAL_MAP, {{}}) or {{}}
    modal = REQUESTED_MODAL
    if use_modality:
        if modal is None:
            modal = "mpa" if "mpa" in modal_map else next(iter(modal_map))
        if modal not in modal_map:
            raise ValueError(
                f"Requested modal {{modal!r}} is not in checkpoint modalities: "
                f"{{list(modal_map.keys())}}"
            )

    model["train_shift_scale"] = {train_shift_scale!r}
    model["train_denominator"] = {train_denominator!r}

    train.update(
        {{
            "random_seed": 1,
            "is_train_stress": False,
            "epoch": {epoch},
            "loss": "Huber",
            "loss_param": {{"delta": {huber_delta:.12g}}},
            "optimizer": "adam",
            "optim_param": {{"lr": {learning_rate:.12g}}},
            "scheduler": "exponentiallr",
            "scheduler_param": {{"gamma": 0.99}},
            "force_loss_weight": {force_loss_weight:.12g},
            "stress_loss_weight": 0.0,
            "best_metric": {best_metric!r},
            "per_epoch": 10,
            "error_record": [
                ["Energy", "RMSE"],
                ["Force", "RMSE"],
                ["Energy", "MAE"],
                ["Force", "MAE"],
                ["TotalLoss", "None"],
            ],
            "continue": {{
                "checkpoint": PRETRAINED,
                "reset_optimizer": True,
                "reset_scheduler": True,
                "reset_epoch": True,
                "use_statistic_values_of_checkpoint": True,
            }},
        }}
    )
    if use_modality:
        train[KEY.USE_MODALITY] = True

    data.update(
        {{
            "batch_size": {batch_size},
            "data_divide_ratio": {data_divide_ratio:.12g},
            "dataset_type": "graph",
            "data_format": "ase",
            "data_format_args": {{}},
            "shift": "elemwise_reference_energies",
            "scale": "force_rms",
        }}
    )
    data.pop("load_validset_path", None)
    data.pop("load_dataset_path", None)

    if use_modality:
        data.setdefault("use_modal_wise_shift", True)
        data.setdefault("use_modal_wise_scale", False)
        data["load_trainset_path"] = [
            {{
                "data_modality": modal,
                "file_list": [{{"file": TRAIN_DATASET}}],
                "data_weight": {{"energy": 1.0, "force": 1.0, "stress": 1.0}},
            }}
        ]
        if HAS_VALIDSET:
            data["load_validset_path"] = [
                {{
                    "data_modality": modal,
                    "file_list": [{{"file": VALID_DATASET}}],
                    "data_weight": {{"energy": 1.0, "force": 1.0, "stress": 1.0}},
                }}
            ]
    else:
        data.pop("use_modal_wise_shift", None)
        data.pop("use_modal_wise_scale", None)
        data["load_trainset_path"] = [TRAIN_DATASET]
        if HAS_VALIDSET:
            data["load_validset_path"] = [VALID_DATASET]

    Path(OUTPUT).write_text(
        yaml.dump(cfg, indent=4, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
        newline="\\n",
    )
    Path("finetune_meta.sh").write_text(
        'SEVENNET_PRETRAINED=' + repr(PRETRAINED) + '\\n'
        + 'SEVENNET_MODAL=' + repr("" if modal is None else modal) + '\\n',
        encoding="utf-8",
        newline="\\n",
    )
    print(f"Wrote {{OUTPUT}} from checkpoint {{PRETRAINED}}")
    if modal:
        print(f"Using SevenNet modality/task: {{modal}}")


if __name__ == "__main__":
    main()
'''


def _run_script_text() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python make_input_finetune.py

sevenn train input_finetune.yaml -s

CHECKPOINT="checkpoint_best.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint_best.pth was not created. Check that valid.extxyz is non-empty and best_metric exists in the validset logs." >&2
  exit 1
fi

if [[ -f "finetune_meta.sh" ]]; then
  source finetune_meta.sh
fi

MODAL_ARGS=()
if [[ -n "${SEVENNET_MODAL:-}" ]]; then
  MODAL_ARGS=(--modal "${SEVENNET_MODAL}")
fi

# Serial LAMMPS model.
sevenn get_model "${CHECKPOINT}" --enable_flash "${MODAL_ARGS[@]}" -o deployed_serial

# Parallel LAMMPS model shards. Use with pair_style e3gnn/parallel.
sevenn get_model "${CHECKPOINT}" --get_parallel --enable_flash "${MODAL_ARGS[@]}" -o deployed_parallel
"""


def _readme_text(pretrained: str) -> str:
    return f"""# SevenNet Fine-Tuning Directory

This directory was generated for fine-tuning a pretrained SevenNet model on
Gaussian energy/force labels from the morphology active-learning loop.

Run on the training server:

```bash
cd "$(dirname "$0")"
bash run_finetune.sh
```

Equivalent manual commands:

```bash
python make_input_finetune.py
sevenn train input_finetune.yaml -s
sevenn get_model checkpoint_best.pth --enable_flash -o deployed_serial
sevenn get_model checkpoint_best.pth --get_parallel --enable_flash -o deployed_parallel
```

The directory includes `make_input_finetune.py`. `run_finetune.sh` executes it
before training, so the YAML is regenerated from the actual checkpoint config.
This is important for multi-modal models such as `7net-omni`, where `use_modality`
and the model architecture must match the checkpoint.

Stress training is disabled because Gaussian cluster labels contain energy and
forces but no periodic stress. The generated config uses explicit `train.extxyz`
and `valid.extxyz` files, and selects `checkpoint_best.pth` using the validation
metric. For SevenNet-Omni, the default task/modal is `mpa`; for molecular
omegaB97-family Gaussian labels, regenerate with `--modal omol25_low`.
"""


def _validset_yaml_block(modal: str | None, has_validset: bool) -> str:
    if not has_validset:
        return ""
    return f"""
    load_validset_path:
        - data_modality: '{modal}'
          file_list:
              - file: './valid.extxyz'
          data_weight:
              energy: 1.0
              force: 1.0
              stress: 1.0"""


def _single_modal_dataset_yaml_block(has_validset: bool) -> str:
    text = "    load_trainset_path: ['./train.extxyz']\n"
    if has_validset:
        text += "    load_validset_path: ['./valid.extxyz']\n"
    return text


def _yaml_bool(value: bool) -> str:
    return "true" if value else "false"


def _default_modal_for_pretrained(pretrained: str) -> str | None:
    lowered = pretrained.lower()
    if "omni" in lowered:
        return "mpa"
    if "mf-ompa" in lowered:
        return "mpa"
    if "mf-0" in lowered:
        return "pbe"
    return None


def _model_block_for_pretrained(
    pretrained: str,
    train_shift_scale_text: str,
    train_denominator_text: str,
) -> str:
    if _default_modal_for_pretrained(pretrained) is not None:
        return _indent(
            f"""chemical_species: 'univ'
cutoff: 6.0
channel: 128
is_parity: false
lmax: 3
num_convolution_layer: 3
irreps_manual:
    - "128x0e"
    - "128x0e+64x1e+32x2e+16x3e"
    - "128x0e+64x1e+32x2e+16x3e"
    - "128x0e"

weight_nn_hidden_neurons: [64, 64]
radial_basis:
    radial_basis_name: 'bessel'
    bessel_basis_num: 8
cutoff_function:
    cutoff_function_name: 'XPLOR'
    cutoff_on: 5.5
self_connection_type: 'linear'

conv_denominator: 'avg_num_neigh'
train_shift_scale: {train_shift_scale_text}
train_denominator: {train_denominator_text}

use_modal_node_embedding: false
use_modal_self_inter_intro: true
use_modal_self_inter_outro: true
use_modal_output_block: true""",
            4,
        )

    return _indent(
        f"""chemical_species: 'Auto'
cutoff: 5.0
channel: 128
is_parity: false
lmax: 2
num_convolution_layer: 5
irreps_manual:
    - "128x0e"
    - "128x0e+64x1e+32x2e"
    - "128x0e+64x1e+32x2e"
    - "128x0e+64x1e+32x2e"
    - "128x0e+64x1e+32x2e"
    - "128x0e"

weight_nn_hidden_neurons: [64, 64]
radial_basis:
    radial_basis_name: 'bessel'
    bessel_basis_num: 8
cutoff_function:
    cutoff_function_name: 'XPLOR'
    cutoff_on: 4.5
self_connection_type: 'linear'

conv_denominator: 'avg_num_neigh'
train_shift_scale: {train_shift_scale_text}
train_denominator: {train_denominator_text}""",
        4,
    )


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.sevennet_finetune import prepare_sevennet_finetune


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a SevenNet pretrained-model fine-tuning directory."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--logs",
        help="Gaussian log/out file or directory to parse into fine_tuning_set.extxyz",
    )
    source.add_argument(
        "--dataset-extxyz",
        help="Existing extxyz dataset to copy as fine_tuning_set.extxyz",
    )
    parser.add_argument(
        "--output-dir",
        default="sevennet_finetune",
        help="Output fine-tuning work directory",
    )
    parser.add_argument(
        "--pretrained",
        default="7net-0",
        help="SevenNet pretrained checkpoint alias or checkpoint path",
    )
    parser.add_argument(
        "--modal",
        default=None,
        help=(
            "SevenNet modality/task for multi-modal checkpoints. "
            "For 7net-omni, default is mpa if omitted."
        ),
    )
    parser.add_argument("--epoch", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size. Use 1 for large molecular clusters on 24 GB GPUs.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-5, help="Adam learning rate")
    parser.add_argument(
        "--force-loss-weight",
        type=float,
        default=10.0,
        help="Force loss weight in the SevenNet objective",
    )
    parser.add_argument(
        "--data-divide-ratio",
        type=float,
        default=0.2,
        help=(
            "Fraction of structures to write to valid.extxyz. The generated "
            "YAML explicitly uses train.extxyz and valid.extxyz."
        ),
    )
    parser.add_argument(
        "--best-metric",
        default="Force_RMSE",
        help="Validation metric used by SevenNet for checkpoint_best.pth",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.1,
        help="Huber loss delta. Larger values penalize large force errors more strongly.",
    )
    parser.add_argument(
        "--no-train-shift-scale",
        action="store_true",
        help="Keep pretrained shift/scale fixed instead of fine-tuning them",
    )
    parser.add_argument(
        "--train-denominator",
        action="store_true",
        help="Fine-tune convolution denominators as well",
    )
    parser.add_argument(
        "--require-normal-termination",
        action="store_true",
        help="When using --logs, reject Gaussian logs without normal termination",
    )
    parser.add_argument(
        "--sevennet-label",
        default="gaussian_finetune",
        help="Label used in the generated structure_list",
    )
    args = parser.parse_args()

    setup = prepare_sevennet_finetune(
        output_dir=args.output_dir,
        logs_path=args.logs,
        dataset_extxyz=args.dataset_extxyz,
        pretrained=args.pretrained,
        modal=args.modal,
        epoch=args.epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        force_loss_weight=args.force_loss_weight,
        data_divide_ratio=args.data_divide_ratio,
        best_metric=args.best_metric,
        huber_delta=args.huber_delta,
        train_shift_scale=not args.no_train_shift_scale,
        train_denominator=args.train_denominator,
        require_normal_termination=args.require_normal_termination,
        sevennet_label=args.sevennet_label,
    )

    print(f"Prepared SevenNet fine-tuning directory: {setup.output_dir}")
    print(f"Structures: {setup.n_structures}")
    print(f"Input YAML: {setup.input_yaml}")
    print(f"Dataset: {setup.extxyz_path}")
    print(f"Train split: {Path(setup.output_dir) / 'train.extxyz'}")
    print(f"Valid split: {Path(setup.output_dir) / 'valid.extxyz'}")
    print(f"Structure list: {setup.structure_list_path}")
    if setup.modal:
        print(f"SevenNet modal/task: {setup.modal}")
    print("")
    print("Run on the training server:")
    print(f"  cd {Path(setup.output_dir).as_posix()}")
    print("  bash run_finetune.sh")
    print("")
    print("Or manually:")
    print("  python make_input_finetune.py")
    print("  sevenn train input_finetune.yaml -s")
    if setup.modal:
        print(
            f"  sevenn get_model checkpoint_best.pth --enable_flash --modal {setup.modal} -o deployed_serial"
        )
        print(
            f"  sevenn get_model checkpoint_best.pth --get_parallel --enable_flash --modal {setup.modal} -o deployed_parallel"
        )
    else:
        print("  sevenn get_model checkpoint_best.pth --enable_flash -o deployed_serial")
        print("  sevenn get_model checkpoint_best.pth --get_parallel --enable_flash -o deployed_parallel")


if __name__ == "__main__":
    main()

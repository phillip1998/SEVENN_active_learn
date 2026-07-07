from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.dataset import (  # noqa: E402
    write_dataset_manifest,
    write_extxyz_dataset,
    write_jsonl_dataset,
    write_sevennet_structure_list,
)
from mlp_md_loop.gaussian import parse_gaussian_logs_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Gaussian log/out results into interim MLP training datasets."
    )
    parser.add_argument("--input", required=True, help="Gaussian log/out file or directory")
    parser.add_argument("--output-dir", default="training_dataset", help="Output dataset directory")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["jsonl", "extxyz"],
        default=["jsonl", "extxyz"],
        help="Dataset files to write",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Do not search directories recursively")
    parser.add_argument(
        "--require-normal-termination",
        action="store_true",
        help="Reject logs without 'Normal termination of Gaussian'",
    )
    parser.add_argument(
        "--sevennet-label",
        default="gaussian",
        help="Label written to the SevenNet structure_list file",
    )
    args = parser.parse_args()

    structures = parse_gaussian_logs_batch(
        args.input,
        recursive=not args.no_recursive,
        require_normal_termination=args.require_normal_termination,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "jsonl" in args.formats:
        write_jsonl_dataset(output_dir / "dataset.jsonl", structures)
    if "extxyz" in args.formats:
        extxyz_path = output_dir / "dataset.extxyz"
        write_extxyz_dataset(extxyz_path, structures)
        write_sevennet_structure_list(
            output_dir / "structure_list",
            extxyz_path,
            label=args.sevennet_label,
        )
    write_dataset_manifest(output_dir / "dataset_manifest.tsv", structures)

    print(f"Parsed {len(structures)} Gaussian result files")
    print(f"Output directory: {output_dir}")
    for structure in structures[:5]:
        print(
            "  {} atoms={} energy={:.8f} eV normal_termination={}".format(
                structure.source_path,
                len(structure.atoms),
                structure.energy_ev,
                structure.normal_termination,
            )
        )


if __name__ == "__main__":
    main()

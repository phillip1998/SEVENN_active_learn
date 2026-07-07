from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.gaussian import (  # noqa: E402
    DEFAULT_FORCE_ROUTE,
    convert_xyz_to_gjf_batch,
    write_gaussian_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert sampled XYZ cluster files to Gaussian GJF inputs."
    )
    parser.add_argument("--input", required=True, help="XYZ file or directory containing XYZ files")
    parser.add_argument("--output-dir", default="gaussian_inputs", help="Directory for GJF files")
    parser.add_argument(
        "--route",
        default=DEFAULT_FORCE_ROUTE,
        help="Gaussian route section. Default is suitable for DFT single-point force labels.",
    )
    parser.add_argument("--charge", type=int, default=0, help="Total cluster charge")
    parser.add_argument("--multiplicity", type=int, default=1, help="Spin multiplicity")
    parser.add_argument("--mem", default="16GB", help="Gaussian %%mem value; use empty string to omit")
    parser.add_argument("--nprocshared", type=int, default=16, help="Gaussian %%nprocshared value")
    parser.add_argument("--no-chk", action="store_true", help="Do not write a %%chk line")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search directories recursively")
    args = parser.parse_args()

    mem = args.mem if args.mem else None
    converted = convert_xyz_to_gjf_batch(
        input_path=args.input,
        output_dir=args.output_dir,
        route=args.route,
        charge=args.charge,
        multiplicity=args.multiplicity,
        mem=mem,
        nprocshared=args.nprocshared,
        chk=not args.no_chk,
        recursive=not args.no_recursive,
    )

    manifest_path = Path(args.output_dir) / "gaussian_inputs.tsv"
    write_gaussian_manifest(manifest_path, converted)

    print(f"Converted {len(converted)} XYZ files to Gaussian inputs")
    print(f"Manifest: {manifest_path}")
    for item in converted[:5]:
        print(f"  {item.gjf_path} ({item.natoms} atoms)")


if __name__ == "__main__":
    main()


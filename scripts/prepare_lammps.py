from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.lammps import prepare_sevennet_lammps_from_gro  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create LAMMPS data/input files for SevenNet from a GROMACS GRO snapshot."
    )
    parser.add_argument("--gro", required=True, help="Input GROMACS .gro snapshot")
    parser.add_argument("--model", required=True, help="SevenNet deployed model path")
    parser.add_argument("--output-dir", default="lammps_sevennet", help="Output directory")
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="LAMMPS type order / SevenNet pair_coeff species order. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--pair-style",
        choices=["e3gnn", "e3gnn/parallel", "mliap"],
        default="e3gnn",
        help="SevenNet LAMMPS pair style",
    )
    parser.add_argument("--parallel-model-count", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep", type=float, default=0.001, help="Timestep in ps for metal units")
    parser.add_argument("--run-steps", type=int, default=1000)
    parser.add_argument("--thermo-interval", type=int, default=100)
    parser.add_argument("--dump-interval", type=int, default=100)
    parser.add_argument("--ensemble", choices=["nve", "nvt", "minimize"], default="nvt")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--d3", action="store_true", help="Use SevenNet+D3 hybrid/overlay pair style")
    args = parser.parse_args()

    data_info, input_info = prepare_sevennet_lammps_from_gro(
        gro_path=args.gro,
        output_dir=args.output_dir,
        model_path=args.model,
        elements=args.elements,
        pair_style=args.pair_style,
        parallel_model_count=args.parallel_model_count,
        temperature_k=args.temperature,
        timestep_ps=args.timestep,
        run_steps=args.run_steps,
        thermo_interval=args.thermo_interval,
        dump_interval=args.dump_interval,
        ensemble=args.ensemble,
        seed=args.seed,
        d3=args.d3,
    )

    print(f"Wrote LAMMPS data: {data_info.data_path}")
    print(f"Wrote LAMMPS input: {input_info.input_path}")
    print(f"Atoms: {data_info.natoms}")
    print(f"Elements/type order: {' '.join(data_info.elements)}")
    print(f"Pair style: {input_info.pair_style}")
    print(f"Pair coeff: {input_info.pair_coeff}")


if __name__ == "__main__":
    main()


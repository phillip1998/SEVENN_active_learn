"""Convert an existing LAMMPS dump to molecule-whole analysis XTC/GRO."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.trajectory_export import export_whole_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--reference-gro", required=True)
    parser.add_argument("--topology", required=True, help="Matching TPR or other MDAnalysis topology with explicit bonds")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestep-ps", type=float, default=0.001, help="LAMMPS timestep in ps, not dump frame interval")
    args = parser.parse_args()
    result = export_whole_trajectory(args.dump, args.reference_gro, args.topology,
                                     args.output_dir, timestep_ps=args.timestep_ps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

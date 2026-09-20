from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.cluster_sampling import sample_dft_clusters_for_sizes


from mlp_md_loop.solution_sampling import add_sampling_arguments, sampling_arguments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample molecular dimer/trimer/tetramer clusters from GROMACS morphology snapshots."
    )
    parser.add_argument("--gro", required=True, help="Input GROMACS .gro topology/snapshot")
    parser.add_argument("--xtc", default=None, help="Optional input .xtc trajectory")
    parser.add_argument(
        "--cluster-sizes",
        type=int,
        nargs="+",
        default=[2],
        help="Cluster sizes to sample, e.g. 2 3 4 for dimers, trimers, tetramers",
    )
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per cluster size")
    parser.add_argument("--output-dir", default="sampled_clusters", help="Output directory")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth XTC frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames")
    parser.add_argument(
        "--random-frame-samples",
        type=int,
        default=8,
        help="Randomly choose this many eligible XTC frames before scoring clusters",
    )
    parser.add_argument("--neighbor-pool", type=int, default=8, help="Nearest neighbors per seed")
    parser.add_argument("--contact-cutoff-nm", type=float, default=0.50, help="Heavy-atom contact cutoff")
    add_sampling_arguments(parser)
    args = parser.parse_args()

    options = sampling_arguments(args)
    results = sample_dft_clusters_for_sizes(
        gro_path=args.gro,
        xtc_path=args.xtc,
        cluster_sizes=args.cluster_sizes,
        **options,
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        random_frame_samples=args.random_frame_samples,
        neighbor_pool=args.neighbor_pool,
        contact_cutoff_nm=args.contact_cutoff_nm,
    )

    for cluster_size, samples in results.items():
        print(f"{cluster_size}-molecule clusters: {len(samples)} samples")
        for sample in samples[:5]:
            print(
                "  sample={:04d} frame={} residues={} score={:.3f} min_dist={:.3f} nm".format(
                    sample.sample_id,
                    sample.frame_index,
                    ",".join(str(resid) for resid in sample.molecule_resids),
                    sample.score,
                    sample.min_interatomic_distance_nm,
                )
            )


if __name__ == "__main__":
    main()

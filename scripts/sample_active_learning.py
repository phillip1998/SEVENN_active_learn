from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.active_learning import sample_active_learning_clusters_for_sizes


from mlp_md_loop.solution_sampling import add_sampling_arguments, sampling_arguments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample active-learning DFT candidates from a SevenNet LAMMPS trajectory."
    )
    parser.add_argument("--dump", required=True, help="LAMMPS custom dump, e.g. dump.sevennet.lammpstrj")
    parser.add_argument("--reference-gro", required=True, help="Original GRO file used to define molecules")
    parser.add_argument(
        "--cluster-sizes",
        type=int,
        nargs="+",
        default=[2],
        help="Molecular cluster sizes to sample, e.g. 2 3 for dimers and trimers",
    )
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per cluster size")
    parser.add_argument("--output-dir", default="active_learning_samples", help="Output directory")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth dump frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames")
    parser.add_argument(
        "--random-frame-samples",
        type=int,
        default=None,
        help="Randomly sample this many eligible dump frames before scoring candidates",
    )
    parser.add_argument(
        "--candidate-seeds-per-frame",
        type=int,
        default=None,
        help="Randomly sample this many seed molecules per selected frame before neighbor search",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress messages")
    parser.add_argument("--neighbor-pool", type=int, default=8, help="Neighbor pool (film: COM; solution: heavy-atom contact, per species group)")
    parser.add_argument(
        "--contact-cutoff-angstrom",
        type=float,
        default=5.0,
        help="Inter-molecular heavy-atom contact cutoff",
    )
    parser.add_argument(
        "--close-contact-alert-angstrom",
        type=float,
        default=1.2,
        help="Flag clusters with a heavy-atom contact below this distance",
    )
    parser.add_argument(
        "--hard-reject-distance-angstrom",
        type=float,
        default=0.55,
        help="Reject geometrically corrupted clusters below this heavy-atom distance",
    )
    parser.add_argument("--random-seed", type=int, default=17, help="Deterministic tie-breaking seed")
    add_sampling_arguments(parser)
    args = parser.parse_args()

    options = sampling_arguments(args)
    results = sample_active_learning_clusters_for_sizes(
        dump_path=args.dump,
        reference_gro=args.reference_gro,
        cluster_sizes=args.cluster_sizes,
        **options,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        random_frame_samples=args.random_frame_samples,
        candidate_seeds_per_frame=args.candidate_seeds_per_frame,
        neighbor_pool=args.neighbor_pool,
        contact_cutoff_angstrom=args.contact_cutoff_angstrom,
        close_contact_alert_angstrom=args.close_contact_alert_angstrom,
        hard_reject_distance_angstrom=args.hard_reject_distance_angstrom,
        random_seed=args.random_seed,
        progress=not args.quiet,
    )

    for cluster_size, samples in results.items():
        print(f"{cluster_size}-molecule active-learning clusters: {len(samples)} samples")
        for sample in samples[:5]:
            alert = " close-contact" if sample.close_contact_alert else ""
            print(
                "  sample={:04d} step={} residues={} score={:.3f} "
                "maxF={:.3f} eV/A min_dist={:.3f} A{}".format(
                    sample.sample_id,
                    sample.timestep,
                    ",".join(str(resid) for resid in sample.molecule_resids),
                    sample.score,
                    sample.max_force_ev_ang,
                    sample.min_interatomic_distance_ang,
                    alert,
                )
            )


if __name__ == "__main__":
    main()

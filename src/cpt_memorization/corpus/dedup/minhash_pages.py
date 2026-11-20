"""Page and booklet MinHash deduplication with publication-level audits."""

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Run MinHash-based deduplication.")
    parser.add_argument("input", type=str, help="Path to input JSONL file or directory.")
    parser.add_argument(
        "output", type=str, help="Directory where deduplication outputs will be written."
    )
    parser.add_argument(
        "--ngram", default=5, type=int, help="Size of n-grams used for MinHash signatures."
    )
    parser.add_argument(
        "-r", "--buckets", default=20, type=int, help="Number of buckets for MinHash."
    )
    parser.add_argument(
        "-b",
        "--hashes-per-bucket",
        "--hashes_per_bucket",
        dest="hashes_per_bucket",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=0,
        type=int,
        help="Worker count for local execution. Defaults to CPU count; use 1 for serial local runs.",
    )
    parser.add_argument(
        "--start-method",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="multiprocess start method for local parallel execution.",
    )
    parser.add_argument(
        "--force-serial", action="store_true", help="Force single-worker local execution."
    )
    parser.add_argument(
        "--partition",
        default="cpu",
        help="Slurm partition to submit jobs to when using Slurm.",
    )
    parser.add_argument(
        "--time", default="72:00:00", help="Wall-clock time limit for each Slurm stage."
    )
    parser.add_argument(
        "--venv", default=None, help="Optional virtualenv activation script path for Slurm jobs."
    )
    parser.add_argument(
        "--slurm-logs",
        default=None,
        help="Directory to store Slurm job logs (default: <output>/minhash-.../slurm_logs).",
    )
    parser.add_argument(
        "--force-clean", action="store_true", help="Remove existing MinHash working directories."
    )
    parser.add_argument("--use-slurm", action="store_true", help="Force execution via Slurm.")
    parser.add_argument(
        "--no-slurm", action="store_true", help="Disable Slurm submission even if available."
    )
    parser.add_argument(
        "--audit-pages",
        action="store_true",
        help="Write page-level kept/removed audit logs under results/page_audit.",
    )
    parser.add_argument(
        "--booklet-shared-pages-threshold",
        default=5,
        type=int,
        help="Minimum shared duplicate pages needed to connect 2 booklets.",
    )
    parser.add_argument(
        "--booklet-containment-threshold",
        default=0.8,
        type=float,
        help="Minimum containment score needed to connect 2 booklets.",
    )
    parser.add_argument(
        "--write-deduplicated-output",
        action="store_true",
        help="Also write kept corpus JSONL files under results/deduplicated_output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        min(args.ngram, args.buckets, args.hashes_per_bucket, args.booklet_shared_pages_threshold)
        < 1
    ):
        raise ValueError("Counts must be positive")
    if not 0 <= args.booklet_containment_threshold <= 1:
        raise ValueError("Containment threshold must be on the 0-1 scale")
    if args.use_slurm and args.no_slurm:
        raise ValueError("Choose either --use-slurm or --no-slurm")
    from ._minhash_pages import build_pipeline

    stage, audit = build_pipeline(args)
    try:
        stage.run()
    finally:
        if audit is not None:
            audit.close()


if __name__ == "__main__":
    main()

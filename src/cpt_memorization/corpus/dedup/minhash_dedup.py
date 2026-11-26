# SPDX-License-Identifier: Apache-2.0
# Modified from the llm-jp/scripts MinHash pipeline; see THIRD_PARTY_NOTICES.md.
"""
minhash_dedup.py
from
https://github.com/llm-jp/scripts/blob/main/corpus/llm-jp-corpus-v4/common/dedup/minhash/slurm/minhash_dedup.py
"""

import shutil
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="Run MinHash-based deduplication locally.")
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
        "-b", "--hashes_per_bucket", default=10, type=int, help="Hashes per bucket for MinHash."
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=1,
        type=int,
        help="CPUs per task requested for Slurm (default: 1)",
    )
    parser.add_argument(
        "--partition",
        default="cpu",
        help="Slurm partition to submit jobs to (default: cpu)",
    )
    parser.add_argument(
        "--time",
        default="72:00:00",
        help="Wall-clock time limit for each stage (default: 72h)",
    )
    parser.add_argument(
        "--venv",
        default=None,
        help="Optional virtualenv activation script path for Slurm jobs",
    )
    parser.add_argument(
        "--slurm-logs",
        default=None,
        help="Directory to store Slurm job logs (default: <output>/minhash-.../slurm_logs)",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Remove existing MinHash working directories before execution",
    )
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        help="Force execution via Slurm (requires sbatch)",
    )
    parser.add_argument(
        "--no-slurm",
        action="store_true",
        help="Disable Slurm submission even if sbatch is available",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.ngram, args.buckets, args.hashes_per_bucket, args.workers) < 1:
        raise ValueError("ngram, buckets, hashes_per_bucket, and workers must be positive")
    if args.no_slurm and args.use_slurm:
        raise ValueError("Choose either --no-slurm or --use-slurm")
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.executor.slurm import SlurmPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.typeshelper import Languages

    MINHASH_DIRNAME = (
        f"minhash-{args.ngram}gram-{args.buckets}buckets-{args.hashes_per_bucket}hashes"
    )
    MINHASH_DIR = Path(args.output) / MINHASH_DIRNAME
    RESULT_DIR = MINHASH_DIR / "results"
    LOG_DIR = MINHASH_DIR / "logs"
    SLURM_LOG_DIR = Path(args.slurm_logs) if args.slurm_logs else MINHASH_DIR / "slurm_logs"

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")
    if input_path.is_relative_to(MINHASH_DIR.resolve()) or (
        input_path.is_dir() and MINHASH_DIR.resolve().is_relative_to(input_path)
    ):
        raise ValueError("Input and MinHash work directories must not contain each other")
    if args.force_clean and MINHASH_DIR.exists():
        shutil.rmtree(MINHASH_DIR)

    if input_path.is_file():
        all_files = [input_path]
    else:
        all_files = [
            path
            for path in input_path.rglob("*")
            if path.is_file() and (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"))
        ]

    TOTAL_TASKS = len(all_files)
    if TOTAL_TASKS == 0:
        raise ValueError(f"No files found under input path: {args.input}")

    MINHASH_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_LOG_DIR.mkdir(parents=True, exist_ok=True)

    sbatch_path = shutil.which("sbatch")
    if args.no_slurm:
        use_slurm = False
    elif args.use_slurm:
        use_slurm = True
    else:
        use_slurm = False

    if use_slurm and sbatch_path is None:
        raise RuntimeError("--use-slurm requires sbatch")

    # this is the original data that we want to deduplicate
    INPUT_READER = JsonlReader(args.input, doc_progress=True)
    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        n_grams=args.ngram,
        num_buckets=args.buckets,
        hashes_per_bucket=args.hashes_per_bucket,
    )  # better precision -> fewer false positives (collisions)

    # helper to configure executor submissions per stage
    def make_executor(
        pipeline,
        tasks,
        logging_subdir,
        job_name,
        depends=None,
        **slurm_kwargs,
    ):
        log_dir = LOG_DIR / logging_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        if use_slurm:
            options = {
                "pipeline": pipeline,
                "tasks": tasks,
                "logging_dir": str(log_dir),
                "depends": depends,
                "job_name": job_name,
                "partition": args.partition,
                "time": args.time,
                "slurm_logs_folder": str(SLURM_LOG_DIR),
                "venv_path": args.venv,
                "cpus_per_task": max(1, args.workers),
            }
            options.update(slurm_kwargs)
            return SlurmPipelineExecutor(**options)

        local_workers = max(1, args.workers)
        return LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=tasks,
            workers=min(local_workers, tasks) if tasks > 0 else 1,
            logging_dir=str(log_dir),
            depends=depends,
        )

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = make_executor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=str(RESULT_DIR / "signatures"),
                config=minhash_config,
                language=Languages.japanese,
                skip_existing_sigs=True,
            ),
        ],
        tasks=TOTAL_TASKS,
        logging_subdir="signatures",
        job_name="minhash_signatures",
    )

    # stage 2 finds matches between signatures in each bucket
    stage2 = make_executor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(RESULT_DIR / "signatures"),
                output_folder=str(RESULT_DIR / "buckets"),
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        depends=stage1,
        logging_subdir="buckets",
        job_name="minhash_buckets",
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = make_executor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(RESULT_DIR / "buckets"),
                output_folder=str(RESULT_DIR / "remove_ids"),
                config=minhash_config,
                save_cluster_id=True,
                save_cluster_size=True,
            ),
        ],
        tasks=1,
        depends=stage2,
        logging_subdir="clusters",
        job_name="minhash_clusters",
    )

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = make_executor(
        pipeline=[
            INPUT_READER,
            # TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
            MinhashDedupFilter(
                input_folder=str(RESULT_DIR / "remove_ids"),
                exclusion_writer=JsonlWriter(str(RESULT_DIR / "removed")),
                load_cluster_ids=True,
                load_cluster_sizes=True,
            ),
            JsonlWriter(output_folder=str(RESULT_DIR / "deduplicated_output")),
        ],
        tasks=TOTAL_TASKS,
        depends=stage3,
        logging_subdir="filter",
        job_name="minhash_filter",
    )
    stage4.run()


if __name__ == "__main__":
    main()

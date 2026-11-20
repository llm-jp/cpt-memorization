#!/usr/bin/env python3
"""Split BLEU-threshold completion samples by base memorized-row overlap."""

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract samples whose token BLEU is above a threshold from an input JSONL "
            "and a base JSONL, then split input memorized samples by base memorized-row overlap."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--kept-output", type=Path, required=True)
    parser.add_argument("--base-memorized-output", type=Path, required=True)
    parser.add_argument("--overlap-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--source-file",
        default=None,
        help="Value written to input output rows. Defaults to input JSONL basename.",
    )
    parser.add_argument(
        "--base-source-file",
        default=None,
        help="Value written to base output rows. Defaults to base JSONL basename.",
    )
    parser.add_argument("--bleu-threshold", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=512)
    return parser.parse_args(argv)


def iter_tasks(input_jsonl: Path) -> list[tuple[int, str]]:
    with input_jsonl.open("r", encoding="utf-8") as f:
        return [(row_index, line) for row_index, line in enumerate(f) if line.strip()]


def compute_bleu(task: tuple[int, str]) -> tuple[int, dict, float]:
    row_index, line = task
    sample = json.loads(line)
    reference = sample.get("suffix_token_ids") or []
    candidate = sample.get("predicted_token_ids") or []
    from .metrics import token_bleu_score

    bleu = token_bleu_score(reference, candidate)
    return row_index, sample, float(bleu)


def make_output_row(row_index: int, sample: dict, bleu: float, source_file: str) -> dict:
    row = dict(sample)
    # Sampling records use row_index for the corpus row. Analysis outputs use
    # it for the completion row, which is unique across prefix lengths.
    if "row_index" in sample:
        row.setdefault("source_row_index", sample["row_index"])
    row.update(row_index=row_index, source_file=source_file, bleu=bleu)
    return row


def validate_overlap_row(row_index: int, sample: dict, base_sample: dict) -> list[dict]:
    mismatches = []
    for field in (
        "archive_member",
        "doc_id",
        "snippet_length",
        "prefix",
        "suffix",
        "prefix_token_ids",
        "suffix_token_ids",
    ):
        if str(sample.get(field, "")) != str(base_sample.get(field, "")):
            mismatches.append(
                {
                    "row_index": row_index,
                    "field": field,
                    "jsonl": sample.get(field, ""),
                    "base_jsonl": base_sample.get(field, ""),
                }
            )
    return mismatches


def extract_bleu_rows(
    tasks: list[tuple[int, str]],
    *,
    source_file: str,
    bleu_threshold: float,
    workers: int,
    chunksize: int,
) -> list[dict]:
    rows: list[dict] = []

    if workers == 1:
        results_iter = map(compute_bleu, tasks)
        pool = None
    else:
        pool = Pool(processes=workers)
        results_iter = pool.imap(compute_bleu, tasks, chunksize=chunksize)

    try:
        for row_index, sample, bleu in results_iter:
            if bleu < bleu_threshold:
                continue

            rows.append(make_output_row(row_index, sample, bleu, source_file))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    rows.sort(key=lambda row: row["row_index"])
    return rows


def split_rows(
    input_rows: list[dict],
    base_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    kept: list[dict] = []
    overlap: list[dict] = []
    mismatches: list[dict] = []
    base_by_row_index = {row["row_index"]: row for row in base_rows}

    for row in input_rows:
        row_index = row["row_index"]
        base_row = base_by_row_index.get(row_index)
        if base_row is None:
            kept.append(row)
            continue

        mismatches.extend(validate_overlap_row(row_index, row, base_row))
        overlap.append(row)

    kept.sort(key=lambda row: row["row_index"])
    overlap.sort(key=lambda row: row["row_index"])
    return kept, overlap, mismatches


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def write_outputs(
    args: argparse.Namespace,
    kept: list[dict],
    base_memorized: list[dict],
    overlap: list[dict],
    summary: dict,
) -> None:
    write_jsonl(args.kept_output, kept)
    write_jsonl(args.base_memorized_output, base_memorized)
    write_jsonl(args.overlap_output, overlap)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_summary(
    args: argparse.Namespace,
    *,
    input_total_rows: int,
    base_total_rows: int,
    input_candidate_count: int,
    base_candidate_count: int,
    kept_count: int,
    overlap_count: int,
) -> dict:
    return {
        "input_jsonl": str(args.input_jsonl),
        "base_jsonl": str(args.base_jsonl),
        "source_file": args.source_file or args.input_jsonl.name,
        "base_source_file": args.base_source_file or args.base_jsonl.name,
        "kept_output": str(args.kept_output),
        "base_memorized_output": str(args.base_memorized_output),
        "overlap_output": str(args.overlap_output),
        "bleu_implementation": "unsmoothed effective-order token BLEU",
        "bleu_input": "token_ids",
        "bleu_threshold": args.bleu_threshold,
        "workers": args.workers,
        "chunksize": args.chunksize,
        "input_total_rows": input_total_rows,
        "base_total_rows": base_total_rows,
        "input_bleu_ge_threshold_rows": input_candidate_count,
        "base_bleu_ge_threshold_rows": base_candidate_count,
        "kept_rows": kept_count,
        "overlap_rows": overlap_count,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_jsonl.is_file():
        raise SystemExit(f"input JSONL not found: {args.input_jsonl}")
    if not args.base_jsonl.is_file():
        raise SystemExit(f"base JSONL not found: {args.base_jsonl}")
    if not 0 <= args.bleu_threshold <= 1:
        raise SystemExit("--bleu-threshold must be on the 0-1 scale")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    from .report_base_filtered_metrics import paired_tasks

    for _ in paired_tasks(args.base_jsonl, args.input_jsonl):
        pass
    input_tasks = iter_tasks(args.input_jsonl)
    base_tasks = iter_tasks(args.base_jsonl)
    input_memorized = extract_bleu_rows(
        input_tasks,
        source_file=args.source_file or args.input_jsonl.name,
        bleu_threshold=args.bleu_threshold,
        workers=args.workers,
        chunksize=args.chunksize,
    )
    base_memorized = extract_bleu_rows(
        base_tasks,
        source_file=args.base_source_file or args.base_jsonl.name,
        bleu_threshold=args.bleu_threshold,
        workers=args.workers,
        chunksize=args.chunksize,
    )
    kept, overlap, mismatches = split_rows(input_memorized, base_memorized)

    if mismatches:
        preview = json.dumps(mismatches[:5], ensure_ascii=False)
        raise SystemExit(f"base JSONL row_index validation failed: {preview}")

    summary = build_summary(
        args,
        input_total_rows=len(input_tasks),
        base_total_rows=len(base_tasks),
        input_candidate_count=len(input_memorized),
        base_candidate_count=len(base_memorized),
        kept_count=len(kept),
        overlap_count=len(overlap),
    )

    write_outputs(args, kept, base_memorized, overlap, summary)

    print(f"input_jsonl={args.input_jsonl}")
    print(f"base_jsonl={args.base_jsonl}")
    print(f"kept_output={args.kept_output}")
    print(f"base_memorized_output={args.base_memorized_output}")
    print(f"overlap_output={args.overlap_output}")
    print(f"summary_output={args.summary_output}")
    print(f"input_bleu_ge_threshold_rows={len(input_memorized)}")
    print(f"base_bleu_ge_threshold_rows={len(base_memorized)}")
    print(f"kept_rows={len(kept)}")
    print(f"overlap_rows={len(overlap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Extract completion samples by token BLEU and optional prefix/suffix Jaccard filters."""

import argparse
import csv
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

from .metrics import compute_memorization_metrics, token_bleu_score


BASE_FIELDNAMES = ["variant", "source_file", "archive_member", "snippet_length", "bleu"]
JACCARD_FIELDNAMES = [
    "jaccard_score",
    "jaccard_threshold",
    "jaccard_ngram_size",
    "best_window_start",
    "best_window_end",
    "best_window_length",
    "best_window_bleu",
    "best_shared_ngram_count",
]
TEXT_FIELDNAMES = ["prefix", "suffix", "predicted"]
TOKEN_FIELDNAMES = ["prefix_token_ids", "suffix_token_ids", "predicted_token_ids"]


def parse_args(
    argv: list[str] | None = None,
    *,
    default_jaccard_threshold: float | None = None,
    default_input_jsonl: Path | None = None,
    default_output_csv: Path | None = None,
    default_include_token_ids: bool = False,
    default_include_row_index: bool = True,
    default_include_doc_id: bool = True,
    default_variant_source: str = "auto",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract samples whose token BLEU is above a threshold, optionally "
            "requiring the best prefix/suffix token n-gram Jaccard score to be below a threshold."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, default=default_input_jsonl)
    parser.add_argument("--input-files", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--completions-root", type=Path, help="Discover iter_*.jsonl in variant subdirectories"
    )
    parser.add_argument(
        "--final-only", action="store_true", help="Select the highest numeric iteration per variant"
    )
    parser.add_argument(
        "--output-csv", "--output-file", dest="output_csv", type=Path, default=default_output_csv
    )
    parser.add_argument(
        "--bleu-threshold", type=float, default=0.75, help="BLEU threshold on the 0-1 scale."
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=default_jaccard_threshold,
        help="If set, keep only rows with best prefix/suffix Jaccard below this threshold.",
    )
    parser.add_argument("--ngram-size", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(152, os.cpu_count() or 1),
        help="Number of worker processes. Defaults to min(152, CPU count).",
    )
    parser.add_argument(
        "--chunksize", type=int, default=512, help="Items per multiprocessing chunk."
    )
    parser.add_argument("--variant", default=None, help="Fixed value for the variant column.")
    parser.add_argument(
        "--variant-source",
        choices=["auto", "filename-suffix", "parent-dir", "stem"],
        default=default_variant_source,
        help="How to infer the variant column when --variant is omitted.",
    )
    parser.add_argument(
        "--include-token-ids",
        action=argparse.BooleanOptionalAction,
        default=default_include_token_ids,
        help="Include prefix/suffix/predicted token ID columns in the output CSV.",
    )
    parser.add_argument(
        "--include-row-index",
        action=argparse.BooleanOptionalAction,
        default=default_include_row_index,
        help="Include the source JSONL row_index column in the output CSV.",
    )
    parser.add_argument(
        "--include-doc-id",
        action=argparse.BooleanOptionalAction,
        default=default_include_doc_id,
        help="Include the doc_id column in the output CSV.",
    )
    args = parser.parse_args(argv)
    if args.completions_root is not None:
        if args.input_files is not None or args.input_jsonl is not None:
            parser.error("Use either explicit input files or --completions-root")
        args.input_files = discover_completions(args.completions_root, args.final_only)
    elif args.final_only:
        parser.error("--final-only requires --completions-root")
    if args.input_files is None:
        args.input_files = [args.input_jsonl] if args.input_jsonl is not None else None
    return args


def discover_completions(root: Path, final_only: bool) -> list[Path]:
    groups = {}
    for path in sorted(root.glob("*/iter_*.jsonl")):
        match = re.fullmatch(r"iter_(\d+)\.jsonl", path.name)
        if match:
            groups.setdefault(path.parent, []).append((int(match[1]), path))
    if not groups:
        raise ValueError(f"No variant/iter_<number>.jsonl completions under {root}")
    return [
        path
        for entries in groups.values()
        for _, path in (sorted(entries)[-1:] if final_only else sorted(entries))
    ]


def infer_variant(input_jsonl: Path, variant_source: str) -> str:
    if variant_source == "filename-suffix" or variant_source == "auto":
        match = re.search(r"_(\d+pct)(?:\.jsonl)?$", input_jsonl.name)
        if match:
            return match.group(1)
        if variant_source == "filename-suffix":
            return input_jsonl.stem

    if variant_source == "parent-dir" or variant_source == "auto":
        parent = input_jsonl.parent.name
        if parent:
            return "base" if parent == "0B" else parent

    return input_jsonl.stem


def iter_indexed_lines(paths: Iterable[Path], variant: str | None, variant_source: str):
    for path in paths:
        source_variant = variant or infer_variant(path, variant_source)
        with path.open("r", encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                if line.strip():
                    yield row_index, line, source_variant, path.name


def process_line(
    args: tuple[int, str, str, str, float, float | None, int, bool],
) -> dict[str, str] | None:
    (
        row_index,
        line,
        variant,
        source_file,
        bleu_threshold,
        jaccard_threshold,
        ngram_size,
        include_token_ids,
    ) = args
    sample = json.loads(line)
    suffix_token_ids = sample.get("suffix_token_ids") or []
    predicted_token_ids = sample.get("predicted_token_ids") or []
    prefix_token_ids = sample.get("prefix_token_ids") or []

    if jaccard_threshold is None:
        bleu = token_bleu_score(suffix_token_ids, predicted_token_ids)
        metrics = None
    else:
        metrics = compute_memorization_metrics(
            suffix_tokens=suffix_token_ids,
            predicted_tokens=predicted_token_ids,
            prefix_tokens=prefix_token_ids,
            ngram_size=ngram_size,
        )
        bleu = metrics["bleu"]

    if bleu < bleu_threshold:
        return None
    if metrics is not None and metrics["jaccard_score"] >= jaccard_threshold:
        return None

    row = {
        "variant": variant,
        "source_file": source_file,
        "row_index": str(row_index),
        "archive_member": sample.get("archive_member", ""),
        "doc_id": sample.get("doc_id", ""),
        "snippet_length": sample.get("snippet_length", ""),
        "bleu": f"{bleu:.12f}",
    }
    if metrics is not None:
        row.update(
            {
                "jaccard_score": f"{metrics['jaccard_score']:.12f}",
                "jaccard_threshold": f"{jaccard_threshold:.6f}",
                "jaccard_ngram_size": str(ngram_size),
                "best_window_start": str(metrics["best_window_start"]),
                "best_window_end": str(metrics["best_window_end"]),
                "best_window_length": str(metrics["best_window_length"]),
                "best_window_bleu": f"{metrics['best_window_bleu']:.12f}",
                "best_shared_ngram_count": str(metrics["best_shared_ngram_count"]),
            }
        )
    row.update(
        {
            "prefix": sample.get("prefix", ""),
            "suffix": sample.get("suffix", ""),
            "predicted": sample.get("predicted", ""),
        }
    )
    if include_token_ids:
        row.update(
            {
                "prefix_token_ids": json.dumps(prefix_token_ids, ensure_ascii=False),
                "suffix_token_ids": json.dumps(suffix_token_ids, ensure_ascii=False),
                "predicted_token_ids": json.dumps(predicted_token_ids, ensure_ascii=False),
            }
        )
    return row


def fieldnames(
    include_jaccard: bool,
    include_token_ids: bool,
    include_row_index: bool,
    include_doc_id: bool,
) -> list[str]:
    names = ["variant", "source_file"]
    if include_row_index:
        names.append("row_index")
    names.append("archive_member")
    if include_doc_id:
        names.append("doc_id")
    names.extend(["snippet_length", "bleu"])
    if include_jaccard:
        names.extend(JACCARD_FIELDNAMES)
    names.extend(TEXT_FIELDNAMES)
    if include_token_ids:
        names.extend(TOKEN_FIELDNAMES)
    return names


def main(
    argv: list[str] | None = None,
    *,
    default_jaccard_threshold: float | None = None,
    default_input_jsonl: Path | None = None,
    default_output_csv: Path | None = None,
    default_include_token_ids: bool = False,
    default_include_row_index: bool = True,
    default_include_doc_id: bool = True,
    default_variant_source: str = "auto",
) -> int:
    args = parse_args(
        argv,
        default_jaccard_threshold=default_jaccard_threshold,
        default_input_jsonl=default_input_jsonl,
        default_output_csv=default_output_csv,
        default_include_token_ids=default_include_token_ids,
        default_include_row_index=default_include_row_index,
        default_include_doc_id=default_include_doc_id,
        default_variant_source=default_variant_source,
    )
    if (
        not 0 <= args.bleu_threshold <= 1
        or (args.jaccard_threshold is not None and not 0 <= args.jaccard_threshold <= 1)
        or args.ngram_size < 1
    ):
        raise ValueError("Require thresholds in [0,1] and ngram-size >= 1")
    if not args.input_files:
        raise SystemExit("--input-jsonl or --input-files is required")
    if args.output_csv is None:
        raise SystemExit("--output-csv/--output-file is required")
    for input_file in args.input_files:
        if input_file is None or not input_file.is_file():
            raise SystemExit(f"input JSONL not found: {input_file}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    tasks = (
        (
            row_index,
            line,
            variant,
            source_file,
            args.bleu_threshold,
            args.jaccard_threshold,
            args.ngram_size,
            args.include_token_ids,
        )
        for row_index, line, variant, source_file in iter_indexed_lines(
            args.input_files,
            args.variant,
            args.variant_source,
        )
    )

    if args.workers == 1:
        rows = [row for row in map(process_line, tasks) if row is not None]
    else:
        with Pool(processes=args.workers) as pool:
            rows = [
                row
                for row in pool.imap_unordered(process_line, tasks, chunksize=args.chunksize)
                if row is not None
            ]
        rows.sort(key=lambda row: (row["variant"], row["source_file"], int(row["row_index"])))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames(
                args.jaccard_threshold is not None,
                args.include_token_ids,
                args.include_row_index,
                args.include_doc_id,
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"input_files={','.join(str(path) for path in args.input_files)}")
    print(f"output_csv={args.output_csv}")
    print(f"bleu_threshold={args.bleu_threshold}")
    if args.jaccard_threshold is not None:
        print(f"jaccard_threshold={args.jaccard_threshold}")
        print(f"ngram_size={args.ngram_size}")
    print(f"workers={args.workers}")
    print(f"memorized_samples={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

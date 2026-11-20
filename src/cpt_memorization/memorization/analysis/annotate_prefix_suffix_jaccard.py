#!/usr/bin/env python3
"""Add best prefix-window/suffix token n-gram Jaccard columns to a CSV."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

from .metrics import find_best_jaccard_window

JACCARD_FIELDNAMES = [
    "jaccard_score",
    "jaccard_ngram_size",
    "best_window_start",
    "best_window_end",
    "best_window_length",
    "best_window_bleu",
    "best_shared_ngram_count",
]


def parse_args(
    argv: list[str] | None = None,
    *,
    default_token_source: str = "csv",
    default_input_csv: Path | None = None,
    default_output_csv: Path | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each CSV row, compare suffix_token_ids against every same-length "
            "prefix_token_ids window and append the best token n-gram Jaccard result."
        )
    )
    parser.add_argument(
        "--token-source",
        choices=["csv", "jsonl"],
        default=default_token_source,
        help="Read token IDs from CSV columns or from source JSONL rows referenced by CSV.",
    )
    parser.add_argument("--input-csv", type=Path, default=default_input_csv)
    parser.add_argument("--output-csv", type=Path, default=default_output_csv)
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=None,
        help="Directory containing source_file/sample_file JSONL files when --token-source=jsonl.",
    )
    parser.add_argument("--ngram-size", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(152, os.cpu_count() or 1),
        help="Number of worker processes. Defaults to min(152, CPU count).",
    )
    parser.add_argument(
        "--chunksize", type=int, default=128, help="Items per multiprocessing chunk."
    )
    parser.add_argument(
        "--line-no-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Base for CSV line_no values when --token-source=jsonl.",
    )
    parser.add_argument(
        "--no-validate-text",
        action="store_true",
        help="Skip prefix/suffix text consistency checks when --token-source=jsonl.",
    )
    return parser.parse_args(argv)


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"input CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def output_fieldnames(input_fieldnames: list[str]) -> list[str]:
    return input_fieldnames + [name for name in JACCARD_FIELDNAMES if name not in input_fieldnames]


def parse_token_ids(row: dict[str, str], column: str) -> list[int]:
    value = row.get(column, "")
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{column} is not a JSON list")
    return parsed


def required_jsonl_rows(rows: list[dict[str, str]], line_no_base: int) -> dict[str, set[int]]:
    required: dict[str, set[int]] = defaultdict(set)
    for row_index, row in enumerate(rows, start=1):
        source_file = row.get("source_file") or row.get("sample_file")
        line_no = row.get("line_no")
        if not source_file:
            raise SystemExit(f"row {row_index}: missing source_file/sample_file")
        if not line_no:
            raise SystemExit(f"row {row_index}: missing line_no")
        try:
            jsonl_index = int(line_no) - line_no_base
        except ValueError as exc:
            raise SystemExit(f"row {row_index}: invalid line_no={line_no!r}") from exc
        if jsonl_index < 0:
            raise SystemExit(f"row {row_index}: line_no={line_no!r} is below base {line_no_base}")
        required[source_file].add(jsonl_index)
    return required


def load_samples(
    jsonl_dir: Path,
    required: dict[str, set[int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    samples: dict[tuple[str, int], dict[str, Any]] = {}
    for source_file, needed_indexes in sorted(required.items()):
        path = jsonl_dir / source_file
        if not path.is_file():
            raise SystemExit(f"source JSONL not found: {path}")
        remaining = set(needed_indexes)
        with path.open("r", encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                if line_index not in remaining:
                    continue
                samples[(source_file, line_index)] = json.loads(line)
                remaining.remove(line_index)
                if not remaining:
                    break
        if remaining:
            preview = ", ".join(str(i) for i in sorted(remaining)[:10])
            raise SystemExit(f"{path}: missing JSONL line indexes: {preview}")
    return samples


def checked_jsonl_tokens(
    row: dict[str, str],
    sample: dict[str, Any],
    row_number: int,
    validate_text: bool,
) -> tuple[list[int], list[int]]:
    if validate_text:
        for column in ("prefix", "suffix"):
            csv_value = row.get(column, "")
            jsonl_value = str(sample.get(column, ""))
            if csv_value != jsonl_value:
                raise ValueError(
                    f"row {row_number}: {column} does not match source JSONL "
                    f"for line_no={row.get('line_no')}"
                )
    prefix_token_ids = sample.get("prefix_token_ids") or []
    suffix_token_ids = sample.get("suffix_token_ids") or []
    if not isinstance(prefix_token_ids, list) or not isinstance(suffix_token_ids, list):
        raise ValueError(f"row {row_number}: token ID fields are not lists")
    return prefix_token_ids, suffix_token_ids


def process_csv_row(args: tuple[int, dict[str, str], int]) -> tuple[int, dict[str, str]]:
    index, row, ngram_size = args
    return add_jaccard_columns(
        index,
        row,
        parse_token_ids(row, "prefix_token_ids"),
        parse_token_ids(row, "suffix_token_ids"),
        ngram_size,
    )


def process_jsonl_row(
    args: tuple[int, dict[str, str], dict[str, Any], int, bool],
) -> tuple[int, dict[str, str]]:
    index, row, sample, ngram_size, validate_text = args
    prefix_token_ids, suffix_token_ids = checked_jsonl_tokens(row, sample, index + 1, validate_text)
    return add_jaccard_columns(index, row, prefix_token_ids, suffix_token_ids, ngram_size)


def add_jaccard_columns(
    index: int,
    row: dict[str, str],
    prefix_token_ids: list[int],
    suffix_token_ids: list[int],
    ngram_size: int,
) -> tuple[int, dict[str, str]]:
    best = find_best_jaccard_window(prefix_token_ids, suffix_token_ids, ngram_size)
    out = dict(row)
    out.update(
        {
            "jaccard_score": f"{best['jaccard_score']:.12f}",
            "jaccard_ngram_size": str(ngram_size),
            "best_window_start": str(best["best_window_start"]),
            "best_window_end": str(best["best_window_end"]),
            "best_window_length": str(best["best_window_length"]),
            "best_window_bleu": f"{best['best_window_bleu']:.12f}",
            "best_shared_ngram_count": str(best["best_shared_ngram_count"]),
        }
    )
    return index, out


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_parallel(
    tasks: list[tuple[Any, ...]],
    worker_func: Any,
    workers: int,
    chunksize: int,
) -> list[tuple[int, dict[str, str]]]:
    if workers == 1:
        return [worker_func(task) for task in tasks]
    with Pool(processes=workers) as pool:
        indexed_rows = list(pool.imap_unordered(worker_func, tasks, chunksize=chunksize))
    indexed_rows.sort(key=lambda item: item[0])
    return indexed_rows


def main(
    argv: list[str] | None = None,
    *,
    default_token_source: str = "csv",
    default_input_csv: Path | None = None,
    default_output_csv: Path | None = None,
) -> int:
    args = parse_args(
        argv,
        default_token_source=default_token_source,
        default_input_csv=default_input_csv,
        default_output_csv=default_output_csv,
    )
    if args.input_csv is None:
        raise SystemExit("--input-csv is required")
    if not args.input_csv.is_file():
        raise SystemExit(f"input CSV not found: {args.input_csv}")
    if args.output_csv is None:
        raise SystemExit("--output-csv is required")
    if args.token_source == "jsonl" and args.jsonl_dir is None:
        raise SystemExit("--jsonl-dir is required when --token-source=jsonl")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    rows, fieldnames = load_csv(args.input_csv)
    if args.token_source == "csv":
        tasks = [(index, row, args.ngram_size) for index, row in enumerate(rows)]
        indexed_rows = run_parallel(tasks, process_csv_row, args.workers, args.chunksize)
    else:
        required = required_jsonl_rows(rows, args.line_no_base)
        samples = load_samples(args.jsonl_dir, required)
        tasks = []
        for index, row in enumerate(rows):
            source_file = row.get("source_file") or row.get("sample_file") or ""
            jsonl_index = int(row["line_no"]) - args.line_no_base
            tasks.append(
                (
                    index,
                    row,
                    samples[(source_file, jsonl_index)],
                    args.ngram_size,
                    not args.no_validate_text,
                )
            )
        indexed_rows = run_parallel(tasks, process_jsonl_row, args.workers, args.chunksize)

    out_rows = [row for _, row in indexed_rows]
    write_rows(args.output_csv, output_fieldnames(fieldnames), out_rows)

    scores = [float(row["jaccard_score"]) for row in out_rows]
    print(f"input_csv={args.input_csv}")
    print(f"token_source={args.token_source}")
    if args.token_source == "jsonl":
        print(f"jsonl_dir={args.jsonl_dir}")
    print(f"output_csv={args.output_csv}")
    print(f"rows={len(out_rows)}")
    print(f"ngram_size={args.ngram_size}")
    print(f"workers={args.workers}")
    if scores:
        print(f"min_jaccard={min(scores):.12f}")
        print(f"max_jaccard={max(scores):.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

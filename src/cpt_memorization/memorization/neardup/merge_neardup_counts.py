#!/usr/bin/env python3
"""Append near-duplicate occurrence counts to sample CSV rows."""

import argparse
from pathlib import Path

from .jaccard_neardup_common import (
    merge_counts_by_input_csv_row_number,
    merge_counts_by_query_id,
    merge_counts_by_row_content,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge near-duplicate occurrence counts into a sample CSV."
    )
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--counts-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--merge-key",
        choices=["query_id", "input_csv_row_number", "row_content"],
        default="query_id",
        help=(
            "How to match count rows to sample rows. Use query_id for one query per sample row, "
            "input_csv_row_number for low-Jaccard rows with multiple query types, and row_content "
            "for random samples built from completion metadata."
        ),
    )
    parser.add_argument(
        "--count-column",
        default="suffix_near_dup_occurrence_count_jaccard10gram_ge_0.600",
        help="Output column name when --merge-key query_id.",
    )
    parser.add_argument(
        "--threshold", default="0.600", help="Jaccard threshold label for generated count columns."
    )
    parser.add_argument(
        "--n", type=int, default=10, help="N-gram size label for generated count columns."
    )
    parser.add_argument(
        "--query-type",
        action="append",
        default=[],
        help=(
            "Query type to merge when --merge-key is input_csv_row_number or row_content. "
            "May be repeated. Default: prefix, predicted for input_csv_row_number; "
            "suffix, prefix, predicted for row_content."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge_key == "query_id":
        output_csv = merge_counts_by_query_id(
            sample_csv=args.sample_csv,
            counts_jsonl=args.counts_jsonl,
            output_csv=args.output_csv,
            count_column=args.count_column,
        )
    elif args.merge_key == "input_csv_row_number":
        query_types = args.query_type or ["prefix", "predicted"]
        output_csv = merge_counts_by_input_csv_row_number(
            sample_csv=args.sample_csv,
            counts_jsonl=args.counts_jsonl,
            output_csv=args.output_csv,
            query_types=query_types,
            threshold=args.threshold,
            ngram=args.n,
        )
    else:
        query_types = args.query_type or ["suffix", "prefix", "predicted"]
        output_csv = merge_counts_by_row_content(
            sample_csv=args.sample_csv,
            counts_jsonl=args.counts_jsonl,
            output_csv=args.output_csv,
            query_types=query_types,
            threshold=args.threshold,
            ngram=args.n,
        )
    print(str(output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

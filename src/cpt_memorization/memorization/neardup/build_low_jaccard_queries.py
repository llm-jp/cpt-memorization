#!/usr/bin/env python3
"""Filter low prefix/suffix Jaccard samples and build neardup query JSONL."""

import argparse
import json
from pathlib import Path

from .jaccard_neardup_common import (
    TOKEN_ID_COLUMNS,
    build_queries,
    load_low_jaccard_rows,
    write_query_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract rows whose jaccard_score is below a threshold and write query.jsonl "
            "for fast-near-duplicate-matching."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jaccard-max", type=float, default=0.6)
    parser.add_argument(
        "--query-type",
        action="append",
        choices=sorted(TOKEN_ID_COLUMNS),
        default=[],
        help="Token sequence to count in the processed corpus. May be repeated. Default: suffix.",
    )
    parser.add_argument(
        "--prefix-tail-tokens",
        type=int,
        default=50,
        help="Number of final prefix tokens to use when --query-type prefix_tail.",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=0,
        help="Debug limit after low-Jaccard filtering. 0 means no limit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_types = args.query_type or ["suffix"]
    if not args.input_csv.is_file():
        raise SystemExit(f"input CSV not found: {args.input_csv}")

    fieldnames, rows = load_low_jaccard_rows(args.input_csv, args.jaccard_max, args.limit_queries)
    queries = build_queries(rows, query_types, args.prefix_tail_tokens)
    meta = write_query_outputs(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        fieldnames=fieldnames,
        rows=rows,
        queries=queries,
        jaccard_max=args.jaccard_max,
        query_types=query_types,
        prefix_tail_tokens=args.prefix_tail_tokens,
        limit_queries=args.limit_queries,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

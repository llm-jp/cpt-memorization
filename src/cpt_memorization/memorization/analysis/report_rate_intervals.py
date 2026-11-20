#!/usr/bin/env python3
"""Recover integer counts from rate CSVs and report pointwise Wilson intervals.

Requires a known common denominator and sufficiently precise input rates.
Output ratios and CI bounds use the 0-1 scale, regardless of input scale.
These binomial intervals assume independent evaluation examples; they are
not simultaneous intervals across checkpoints or estimates of corpus prevalence.
"""

import argparse
import csv
import math
from pathlib import Path

from .report_base_filtered_metrics import binomial_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--denominator", type=int, required=True)
    parser.add_argument("--rate-columns", nargs="+", required=True)
    parser.add_argument(
        "--rate-scale", type=float, default=100.0, help="100 for percentages; 1 for proportions"
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if (
        args.denominator <= 0
        or not math.isfinite(args.rate_scale)
        or args.rate_scale <= 0
        or not 0 < args.confidence < 1
    ):
        parser.error("Require denominator > 0, finite scale > 0, 0 < confidence < 1")

    output = []
    with args.input_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(args.rate_columns) <= set(reader.fieldnames):
            parser.error("Input CSV is missing requested rate columns")
        identifiers = [name for name in reader.fieldnames if name not in args.rate_columns]
        for row_index, row in enumerate(reader, start=2):
            for metric in args.rate_columns:
                rate = float(row[metric]) / args.rate_scale
                if not math.isfinite(rate) or not 0 <= rate <= 1:
                    raise ValueError(f"Row {row_index}, {metric}: invalid rate")
                expected_count = rate * args.denominator
                count = round(expected_count)
                if not math.isclose(expected_count, count, rel_tol=0, abs_tol=1e-7):
                    raise ValueError(
                        f"Row {row_index}, {metric}: rate does not reconstruct an integer "
                        f"count with denominator {args.denominator}: {expected_count}"
                    )
                summary = binomial_summary(count, args.denominator, args.confidence)
                output.append(
                    {
                        **{name: row[name] for name in identifiers},
                        "metric": metric,
                        "flagged": count,
                        "denominator": args.denominator,
                        "ratio": summary["rate"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "confidence": args.confidence,
                        "ci_method": "Wilson",
                    }
                )
    if not output:
        raise ValueError("Empty input CSV")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Validated and wrote {len(output)} pointwise intervals to {args.output_csv}")


if __name__ == "__main__":
    main()

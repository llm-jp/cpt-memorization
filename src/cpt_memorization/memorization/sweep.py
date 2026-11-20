"""Aggregate paired checkpoints using a fixed BASE-filtered evaluation denominator."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ..io import write_json
from ..qa.aggregate import write_csv
from .analysis.report_base_filtered_metrics import paired_tasks, score_pair, summarize


def aggregate(manifest_path, output_dir, threshold=0.75, confidence=0.95):
    if not 0 <= threshold <= 1 or not 0 < confidence < 1:
        raise ValueError("Require 0 <= threshold <= 1 and 0 < confidence < 1")
    manifest_path, output = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else manifest_path.parent / path

    base = resolve(manifest["base_file"])
    if not manifest["runs"]:
        raise ValueError("No checkpoint runs")
    seen, reports = set(), []
    csv_rows = defaultdict(list)
    for entry in manifest["runs"]:
        key = (entry["variant"], float(entry["tokens_B"]))
        if key in seen or not math.isfinite(key[1]) or key[1] < 0:
            raise ValueError(f"Invalid/duplicate checkpoint {key}")
        seen.add(key)
        counts = defaultdict(Counter)
        for length, _, base_bleu, target_bleu, exact in map(
            score_pair, paired_tasks(base, resolve(entry["path"]))
        ):
            excluded, flagged = base_bleu >= threshold, target_bleu >= threshold
            counts[length].update(
                total=1,
                excluded=int(excluded),
                target_flagged=int(flagged),
                overlap=int(excluded and flagged),
                retained_flagged=int(not excluded and flagged),
                retained_exact=int(not excluded and exact),
            )
        if not counts:
            raise ValueError(f"Empty checkpoint {key}")
        groups = {length: summarize(count, confidence) for length, count in sorted(counts.items())}
        reports.append({"variant": key[0], "tokens_B": key[1], "groups": groups})
        for length, group in groups.items():
            if not group["retained"]:
                raise ValueError(f"BASE exclusion removed all examples for length {length}")
            csv_rows[length].append(
                dict(
                    variant=key[0],
                    x_tokens_B=key[1],
                    total=group["total"],
                    denominator=group["retained"],
                    base_excluded=group["base_excluded"]["count"],
                    exact=group["exact_after_filter"]["count"],
                    approx=group["target_after_filter"]["count"],
                    exact_rate=100 * group["exact_after_filter"]["rate"],
                    approx_rate=100 * group["target_after_filter"]["rate"],
                )
            )
    for length, rows in csv_rows.items():
        write_csv(output / f"memorization_rates_prefix{length}.csv", rows)
    report = {
        "bleu_implementation": "unsmoothed effective-order token BLEU",
        "bleu_input": "token_ids",
        "bleu_threshold": threshold,
        "confidence": confidence,
        "csv_rate_scale": "percent",
        "denominator": "All evaluation rows not flagged by BASE, including non-memorized target rows",
        "runs": reports,
    }
    write_json(output / "paired_metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bleu-threshold", type=float, default=0.75)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    aggregate(args.manifest, args.output_dir, args.bleu_threshold, args.confidence)


if __name__ == "__main__":
    main()

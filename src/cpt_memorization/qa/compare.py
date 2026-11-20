"""Select correctness patterns or item-wise difference-in-differences for named runs."""

import argparse

from ..io import write_json, write_jsonl
from .aggregate import load_manifest
from .common import load_records, validate_common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="QA manifest; optional labels name each run"
    )
    parser.add_argument("--correct", nargs="*", default=[])
    parser.add_argument("--incorrect", nargs="*", default=[])
    parser.add_argument(
        "--difference", nargs=4, metavar=("A", "B", "C", "D"), help="Compute (A-B)-(C-D)"
    )
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    if not (args.correct or args.incorrect or args.difference):
        parser.error("Specify a correctness pattern or --difference A B C D")
    if set(args.correct) & set(args.incorrect):
        parser.error("A run cannot be both correct and incorrect")
    manifest = load_manifest(args.manifest)
    runs = {e["label"]: load_records(e["path"], evaluated=True) for e in manifest["runs"]}
    requested = args.correct + args.incorrect + (args.difference or [])
    if set(requested) - runs.keys():
        parser.error(f"Unknown labels: {set(requested) - runs.keys()}")
    common = set.intersection(*(set(r) for r in runs.values()))
    if not common:
        raise ValueError("No common QA IDs")
    validate_common(runs, common)
    outputs, contribution_sum = [], 0
    for key in sorted(common):
        if any(not runs[name][key]["is_correct"] for name in args.correct):
            continue
        if any(runs[name][key]["is_correct"] for name in args.incorrect):
            continue
        record = next(iter(runs.values()))[key]
        out = {
            "id": key,
            **{field: record[field] for field in ("text", "question", "answer")},
            "predictions": {
                name: {field: run[key][field] for field in ("is_correct", "predicted_answer")}
                for name, run in runs.items()
            },
        }
        if args.difference:
            a, b, c, d = (int(runs[name][key]["is_correct"]) for name in args.difference)
            contribution = (a - b) - (c - d)
            contribution_sum += contribution
            if not contribution:
                continue
            out.update(
                contribution=contribution, contribution_points=100 * contribution / len(common)
            )
        outputs.append(out)
    write_jsonl(args.output_file, outputs)
    write_json(
        args.output_file + ".summary.json",
        {
            "common": len(common),
            "selected": len(outputs),
            "contribution_sum": contribution_sum,
            "difference_points": 100 * contribution_sum / len(common),
        },
    )


if __name__ == "__main__":
    main()

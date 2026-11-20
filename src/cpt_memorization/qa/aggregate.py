"""Aggregate arbitrary QA runs on a shared set of evaluation examples."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from ..io import write_json
from .common import load_records, validate_common


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["runs"]
    if not entries:
        raise ValueError("Manifest contains no runs")
    seen = set()
    for entry in entries:
        key = (entry["mode"], float(entry["temperature"]), entry["run"], entry["model"])
        if key in seen:
            raise ValueError(f"Duplicate run {key}")
        if not math.isfinite(key[1]) or key[1] < 0:
            raise ValueError(f"Invalid temperature {key[1]}")
        seen.add(key)
        source = Path(entry["path"])
        entry["path"] = str(source if source.is_absolute() else path.parent / source)
        entry.setdefault("label", f"{key[0]}/temp{key[1]:g}/run{key[2]}/{key[3]}")
    if len({entry["label"] for entry in entries}) != len(entries):
        raise ValueError("Manifest labels must be unique")
    return manifest


def write_csv(path, rows, fields=None, delimiter=","):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(manifest_path, output_dir):
    manifest = load_manifest(manifest_path)
    runs = {
        entry["label"]: load_records(entry["path"], evaluated=True) for entry in manifest["runs"]
    }
    common = set.intersection(*(set(run) for run in runs.values()))
    if not common:
        raise ValueError("No QA IDs common to every run")
    validate_common(runs, common)
    raw_rows, common_rows = [], []
    groups = defaultdict(list)
    for entry in manifest["runs"]:
        records = runs[entry["label"]]
        base = {key: entry[key] for key in ("mode", "temperature", "run", "model")}
        for rows, ids in ((raw_rows, set(records)), (common_rows, common)):
            correct = sum(records[key]["is_correct"] for key in ids)
            row = {
                **base,
                "correct": correct,
                "total": len(ids),
                "accuracy": correct / len(ids),
                "dropped": len(records) - len(ids),
            }
            rows.append(row)
        groups[(entry["mode"], float(entry["temperature"]), entry["model"])].append(
            common_rows[-1]["accuracy"]
        )
    aggregated = [
        dict(
            mode=mode,
            temperature=temp,
            model=model,
            runs=len(values),
            total=len(common),
            accuracy_mean=mean(values),
            accuracy_std=stdev(values) if len(values) > 1 else 0.0,
        )
        for (mode, temp, model), values in sorted(groups.items())
    ]
    output = Path(output_dir)
    write_csv(output / "accuracy_by_run.csv", raw_rows)
    write_csv(output / "accuracy_by_run_common.csv", common_rows)
    write_csv(output / "accuracy_aggregate.csv", aggregated)
    union = set.union(*(set(run) for run in runs.values()))
    report = dict(
        runs=len(runs),
        union_ids=len(union),
        common_ids=len(common),
        excluded_ids=sorted(union - common),
        accuracy_scale="0-1",
        std_ddof=1,
        single_run_std=0.0,
        intersection="global across every manifest run",
    )
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

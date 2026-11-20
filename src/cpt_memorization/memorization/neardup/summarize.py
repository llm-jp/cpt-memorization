"""Validate external near-duplicate counts and summarize their distribution by query group."""

import argparse
from collections import defaultdict
from statistics import median

from ...io import read_jsonl, write_json


def summarize_counts(queries_path, counts_path):
    queries, counts = {}, {}
    for path, records, is_count in ((queries_path, queries, False), (counts_path, counts, True)):
        for record in read_jsonl(path):
            key = record["query_id"]
            if type(key) is not int or key in records:
                raise ValueError(f"{path}: invalid or duplicate query_id {key}")
            if is_count and (
                type(record.get("near_dup_occurrence_count")) is not int
                or record["near_dup_occurrence_count"] < 0
            ):
                raise ValueError("Counts must be nonnegative JSON integers")
            records[key] = record
    if not queries or queries.keys() != counts.keys():
        raise ValueError("Query IDs and count IDs must match exactly and be nonempty")
    groups = defaultdict(list)
    for key, record in queries.items():
        groups[record.get("group", "all")].append(counts[key]["near_dup_occurrence_count"])
    return {
        name: dict(n=len(values), median=median(values), minimum=min(values), maximum=max(values))
        for name, values in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", required=True)
    parser.add_argument("--counts-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    write_json(args.output_json, summarize_counts(args.queries_jsonl, args.counts_jsonl))


if __name__ == "__main__":
    main()

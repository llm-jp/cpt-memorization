#!/usr/bin/env python3
"""Report paired completion flags before/after BASE exclusion with binomial CIs.

Rows are evaluation examples (not corpus documents). Corresponding input rows
must have identical document identifiers, lengths, prefixes, and suffixes.
"""

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from itertools import zip_longest
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist

from .metrics import token_bleu_score


def paired_tasks(base_path, input_path):
    def records(path):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    for index, (base, target) in enumerate(
        zip_longest(records(base_path), records(input_path)), start=1
    ):
        if base is None or target is None:
            raise ValueError(f"Different input lengths at record {index}")
        for field in (
            "archive_member",
            "doc_id",
            "snippet_length",
            "prefix_token_ids",
            "suffix_token_ids",
        ):
            if base.get(field) != target.get(field):
                raise ValueError(f"Record {index}: mismatched {field}")
        for record in (base, target):
            for field in ("prefix_token_ids", "suffix_token_ids", "predicted_token_ids"):
                if not isinstance(record.get(field), list) or any(
                    type(token) is not int for token in record[field]
                ):
                    raise ValueError(f"Record {index}: missing/invalid {field}")
            if not record["suffix_token_ids"]:
                raise ValueError(f"Record {index}: empty reference")
            if record.get("snippet_length") != len(record["prefix_token_ids"]) + len(
                record["suffix_token_ids"]
            ):
                raise ValueError(
                    f"Record {index}: snippet_length must equal prefix + suffix length"
                )
        yield (
            base["snippet_length"],
            base.get("doc_id") or base.get("archive_member"),
            base["suffix_token_ids"],
            base["predicted_token_ids"],
            target["predicted_token_ids"],
        )


def score_pair(task):
    length, document, reference, base, target = task
    return (
        length,
        document,
        token_bleu_score(reference, base),
        token_bleu_score(reference, target),
        reference == target,
    )


def binomial_summary(count, total, confidence):
    if not total:
        return dict(count=count, denominator=total, rate=None, ci_low=None, ci_high=None)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = count / total
    divisor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / divisor
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return dict(
        count=count,
        denominator=total,
        rate=p,
        ci_low=0.0 if count == 0 else max(0, center - margin),
        ci_high=1.0 if count == total else min(1, center + margin),
    )


def summarize(counts, confidence):
    total = counts["total"]
    excluded = counts["excluded"]
    retained = total - excluded
    assert counts["target_flagged"] == counts["overlap"] + counts["retained_flagged"]
    return dict(
        total=total,
        retained=retained,
        base_excluded=binomial_summary(excluded, total, confidence),
        target_before_filter=binomial_summary(counts["target_flagged"], total, confidence),
        target_flags_removed=binomial_summary(
            counts["overlap"], counts["target_flagged"], confidence
        ),
        target_after_filter=binomial_summary(counts["retained_flagged"], retained, confidence),
        exact_after_filter=binomial_summary(counts["retained_exact"], retained, confidence),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--bleu-threshold", type=float, default=0.75)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or not 0 <= args.bleu_threshold <= 1 or args.workers < 1:
        parser.error("Require 0 < confidence < 1, 0 <= threshold <= 1, workers >= 1")

    counts = defaultdict(Counter)
    documents = defaultdict(set)
    tasks = paired_tasks(args.base_jsonl, args.input_jsonl)
    pool = Pool(args.workers) if args.workers > 1 else None
    results = pool.imap(score_pair, tasks, chunksize=128) if pool else map(score_pair, tasks)
    try:
        for length, document, base_bleu, target_bleu, exact in results:
            excluded = base_bleu >= args.bleu_threshold
            flagged = target_bleu >= args.bleu_threshold
            for group in (str(length), "all"):
                documents[group].add(document)
                counts[group].update(
                    dict(
                        total=1,
                        excluded=int(excluded),
                        target_flagged=int(flagged),
                        overlap=int(excluded and flagged),
                        retained_flagged=int(not excluded and flagged),
                        retained_exact=int(not excluded and exact),
                    )
                )
    finally:
        if pool:
            pool.terminate()
            pool.join()
    if not counts:
        raise ValueError("Empty evaluation dataset")
    groups = sorted((g for g in counts if g != "all"), key=int) + ["all"]
    report = dict(
        base_jsonl=str(args.base_jsonl),
        input_jsonl=str(args.input_jsonl),
        bleu_implementation="unsmoothed effective-order token BLEU",
        bleu_input="token_ids",
        bleu_threshold=args.bleu_threshold,
        confidence=args.confidence,
        ci_method="Wilson score interval for a binomial proportion",
        unit="completion evaluation row",
        denominator_note="Post-filter memorization rates use the BASE-filtered evaluation dataset, not the full corpus.",
        dependence_note="Binomial intervals assume independent rows; pooled prefix lengths can contain repeated documents and correlated examples.",
        groups={
            g: dict(unique_document_ids=len(documents[g]), **summarize(counts[g], args.confidence))
            for g in groups
        },
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "snippet_length",
                    "metric",
                    "count",
                    "denominator",
                    "rate",
                    "ci_low",
                    "ci_high",
                ],
            )
            writer.writeheader()
            for group, summary in report["groups"].items():
                for name, value in summary.items():
                    if isinstance(value, dict):
                        writer.writerow(dict(snippet_length=group, metric=name, **value))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

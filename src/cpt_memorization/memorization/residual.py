"""Select low-containment memorized examples and a non-memorized comparison sample."""

import argparse
import random
from pathlib import Path

from ..io import read_jsonl, write_json, write_jsonl
from .analysis.prefix_token_containment import containment_metrics
from .analysis.report_base_filtered_metrics import paired_tasks, score_pair


def prepare(
    base_path,
    target_path,
    output_dir,
    snippet_length=1000,
    bleu_threshold=0.75,
    containment_threshold=0.5,
    ngram_size=3,
    control_size=300,
    seed=42,
):
    if not 0 <= bleu_threshold <= 1 or not 0 <= containment_threshold <= 1:
        raise ValueError("Thresholds must be on the 0-1 scale")
    if control_size < 0 or ngram_size < 1:
        raise ValueError("Require control-size >= 0 and ngram-size >= 1")
    rng = random.Random(seed)
    residual, controls = [], []
    total, excluded, flagged, copied, control_seen = 0, 0, 0, 0, 0
    tasks = paired_tasks(Path(base_path), Path(target_path))
    for index, (record, task) in enumerate(zip(read_jsonl(target_path), tasks, strict=True)):
        length, _, base_bleu, target_bleu, _ = score_pair(task)
        if length != snippet_length:
            continue
        total += 1
        if base_bleu >= bleu_threshold:
            excluded += 1
            continue
        is_memorized = target_bleu >= bleu_threshold
        flagged += int(is_memorized)
        containment = containment_metrics(
            record["prefix_token_ids"],
            record["suffix_token_ids"],
            ngram_size,
            containment_threshold,
        )["ngram_containment"]
        if containment >= containment_threshold:
            copied += int(is_memorized)
            continue
        row = {
            **record,
            "completion_index": index,
            "bleu": target_bleu,
            "suffix_in_prefix_ngram_containment": containment,
        }
        if is_memorized:
            residual.append(row)
        else:
            control_seen += 1
            if len(controls) < control_size:
                controls.append(row)
            else:
                position = rng.randrange(control_seen)
                if position < control_size:
                    controls[position] = row
    if not total:
        raise ValueError(f"No evaluation examples with snippet_length={snippet_length}")
    if len(controls) < control_size:
        raise ValueError(
            f"Only {len(controls)} eligible control examples; requested {control_size}"
        )
    output = Path(output_dir)
    write_jsonl(output / "residual.jsonl", residual)
    write_jsonl(output / "control.jsonl", controls)
    queries = []
    for group, records in (("memorized", residual), ("non_memorized", controls)):
        for record in records:
            queries.append(
                dict(
                    query_id=len(queries),
                    group=group,
                    query_type="suffix",
                    completion_index=record["completion_index"],
                    doc_id=record.get("doc_id"),
                    token_ids=record["suffix_token_ids"],
                )
            )
    write_jsonl(output / "query.jsonl", queries)
    summary = dict(
        total=total,
        base_excluded=excluded,
        memorized_after_base=flagged,
        explained_by_prefix=copied,
        residual=len(residual),
        control=len(controls),
        eligible_controls=control_seen,
        seed=seed,
        snippet_length=snippet_length,
        bleu_threshold=bleu_threshold,
        containment_threshold=containment_threshold,
        containment="clipped multiset token n-gram overlap / suffix n-gram count",
        ngram_size=ngram_size,
    )
    write_json(output / "selection_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snippet-length", type=int, default=1000)
    parser.add_argument("--bleu-threshold", type=float, default=0.75)
    parser.add_argument("--containment-threshold", type=float, default=0.5)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--control-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        prepare(
            args.base_jsonl,
            args.input_jsonl,
            args.output_dir,
            args.snippet_length,
            args.bleu_threshold,
            args.containment_threshold,
            args.ngram_size,
            args.control_size,
            args.seed,
        )
    )


if __name__ == "__main__":
    main()

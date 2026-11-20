#!/usr/bin/env python3
"""Check whether suffix/predicted token sequences are contained in prefix tokens."""

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Sequence


def threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def default_output_jsonl(input_jsonl: Path, ngram_size: int, threshold: float) -> Path:
    stem = input_jsonl.name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    return input_jsonl.with_name(
        f"{stem}_prefix_token_containment_n{ngram_size}_m{threshold_label(threshold)}.jsonl"
    )


def default_summary_json(input_jsonl: Path, ngram_size: int, threshold: float) -> Path:
    stem = input_jsonl.name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    return input_jsonl.with_name(
        f"{stem}_prefix_token_containment_n{ngram_size}_m{threshold_label(threshold)}_summary.json"
    )


def default_filtered_jsonl(
    input_jsonl: Path,
    ngram_size: int,
    threshold: float,
    target: str,
) -> Path:
    stem = input_jsonl.name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    return input_jsonl.with_name(
        f"{stem}_prefix_token_containment_n{ngram_size}_m"
        f"{threshold_label(threshold)}_{target}_without_ngram_contained.jsonl"
    )


def apply_output_dir(path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return path
    return output_dir / path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each JSONL row, check at token level whether suffix_token_ids and "
            "predicted_token_ids are contained in prefix_token_ids by exact contiguous "
            "match and by token n-gram containment rate."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for default output paths. Ignored for paths explicitly set by "
            "--output-jsonl or --summary-json."
        ),
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--ngram-size", "-n", type=int, default=3)
    parser.add_argument(
        "--min-containment",
        "-m",
        type=float,
        default=0.5,
        help="Minimum target n-gram containment rate required for the boolean flag.",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=["suffix", "predicted"],
        help="Target sequence to check. May be repeated. Default: suffix and predicted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(152, os.cpu_count() or 1),
        help="Worker processes. Defaults to min(152, CPU count).",
    )
    parser.add_argument("--chunksize", type=int, default=128)
    return parser.parse_args()


def validate_token_ids(value: Any, field_name: str, row_number: int) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"row {row_number}: {field_name} is not a list")
    if not all(isinstance(token_id, int) for token_id in value):
        raise ValueError(f"row {row_number}: {field_name} contains non-integer values")
    return value


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        return 0
    if len(needle) > len(haystack):
        return -1

    first = needle[0]
    max_start = len(haystack) - len(needle)
    for start in range(max_start + 1):
        if haystack[start] == first and haystack[start : start + len(needle)] == list(needle):
            return start
    return -1


def token_ngrams(tokens: Sequence[int], ngram_size: int) -> list[tuple[int, ...]]:
    if ngram_size < 1:
        raise ValueError("ngram_size must be >= 1")
    if len(tokens) < ngram_size:
        return []
    return [tuple(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]


def multiset_overlap_count(a: Counter[tuple[int, ...]], b: Counter[tuple[int, ...]]) -> int:
    return sum(min(a[ngram], b[ngram]) for ngram in a.keys() & b.keys())


def containment_metrics(
    prefix_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    ngram_size: int,
    min_containment: float,
) -> dict[str, Any]:
    # 完全一致を検索
    exact_start = find_subsequence(prefix_token_ids, target_token_ids)
    # n-gramを数え上げ
    target_ngrams = Counter(token_ngrams(target_token_ids, ngram_size))
    prefix_ngrams = Counter(token_ngrams(prefix_token_ids, ngram_size))
    total = sum(target_ngrams.values())
    shared = multiset_overlap_count(target_ngrams, prefix_ngrams)
    # 包含率を計算
    rate = (shared / total) if total else 0.0

    return {
        "exact": exact_start >= 0,
        "exact_start": exact_start,
        "exact_end": exact_start + len(target_token_ids) if exact_start >= 0 else -1,
        "ngram_containment": rate,
        "ngram_shared_count": shared,
        "ngram_total_count": total,
        # min_containment以上のときに「含まれている」とみなす
        "ngram_contained": rate >= min_containment,
    }


def process_row(task: tuple[int, str, int, float, tuple[str, ...]]) -> tuple[int, dict[str, Any]]:
    row_index, line, ngram_size, min_containment, targets = task
    obj = json.loads(line)
    row_number = row_index + 1
    prefix_token_ids = validate_token_ids(
        obj.get("prefix_token_ids"), "prefix_token_ids", row_number
    )

    out = dict(obj)
    for target in targets:
        token_field = f"{target}_token_ids"
        target_token_ids = validate_token_ids(obj.get(token_field), token_field, row_number)
        metrics = containment_metrics(
            prefix_token_ids, target_token_ids, ngram_size, min_containment
        )
        prefix = f"{target}_in_prefix"
        out[f"{prefix}_exact"] = metrics["exact"]
        out[f"{prefix}_exact_start"] = metrics["exact_start"]
        out[f"{prefix}_exact_end"] = metrics["exact_end"]
        out[f"{prefix}_ngram_containment"] = metrics["ngram_containment"]
        out[f"{prefix}_ngram_shared_count"] = metrics["ngram_shared_count"]
        out[f"{prefix}_ngram_total_count"] = metrics["ngram_total_count"]
        out[f"{prefix}_ngram_contained"] = metrics["ngram_contained"]

    return row_index, out


def run_parallel(
    tasks: list[tuple[int, str, int, float, tuple[str, ...]]],
    workers: int,
    chunksize: int,
) -> list[tuple[int, dict[str, Any]]]:
    if workers == 1:
        return [process_row(task) for task in tasks]
    with Pool(processes=workers) as pool:
        results = list(pool.imap_unordered(process_row, tasks, chunksize=chunksize))
    results.sort(key=lambda item: item[0])
    return results


def is_ngram_contained(row: dict[str, Any], target: str) -> bool:
    target_key = f"{target}_in_prefix"
    return bool(row[f"{target_key}_ngram_contained"])


def keep_after_excluding_ngram_contained(row: dict[str, Any], target: str) -> bool:
    return not is_ngram_contained(row, target)


def write_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def summarize(
    rows: Iterable[dict[str, Any]],
    targets: Sequence[str],
    input_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path,
    filtered_outputs: dict[str, dict[str, Any]],
    ngram_size: int,
    min_containment: float,
    workers: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "filtered_outputs": filtered_outputs,
        "rows": 0,
        "ngram_size": ngram_size,
        "min_containment": min_containment,
        "workers": workers,
        "targets": {},
    }
    target_summary = {
        target: {
            "exact_count": 0,
            "ngram_contained_count": 0,
            "both_exact_and_ngram_contained_count": 0,
            "min_ngram_containment": None,
            "max_ngram_containment": None,
            "avg_ngram_containment": 0.0,
        }
        for target in targets
    }
    rate_sums = {target: 0.0 for target in targets}

    for row in rows:
        summary["rows"] += 1
        for target in targets:
            target_key = f"{target}_in_prefix"
            exact = bool(row[f"{target_key}_exact"])
            contained = bool(row[f"{target_key}_ngram_contained"])
            rate = float(row[f"{target_key}_ngram_containment"])
            item = target_summary[target]
            item["exact_count"] += int(exact)
            item["ngram_contained_count"] += int(contained)
            item["both_exact_and_ngram_contained_count"] += int(exact and contained)
            item["min_ngram_containment"] = (
                rate
                if item["min_ngram_containment"] is None
                else min(float(item["min_ngram_containment"]), rate)
            )
            item["max_ngram_containment"] = (
                rate
                if item["max_ngram_containment"] is None
                else max(float(item["max_ngram_containment"]), rate)
            )
            rate_sums[target] += rate

    rows_count = int(summary["rows"])
    for target in targets:
        item = target_summary[target]
        if rows_count:
            item["exact_rate"] = item["exact_count"] / rows_count
            item["ngram_contained_rate"] = item["ngram_contained_count"] / rows_count
            item["both_exact_and_ngram_contained_rate"] = (
                item["both_exact_and_ngram_contained_count"] / rows_count
            )
            item["avg_ngram_containment"] = rate_sums[target] / rows_count
        else:
            item["exact_rate"] = 0.0
            item["ngram_contained_rate"] = 0.0
            item["both_exact_and_ngram_contained_rate"] = 0.0
        summary["targets"][target] = item

    return summary


def main() -> int:
    args = parse_args()
    if args.ngram_size < 1:
        raise SystemExit("--ngram-size must be >= 1")
    if not 0 <= args.min_containment <= 1:
        raise SystemExit("--min-containment must be between 0 and 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")
    if not args.input_jsonl.is_file():
        raise SystemExit(f"input JSONL not found: {args.input_jsonl}")

    targets = tuple(args.target or ("suffix", "predicted"))
    default_output = apply_output_dir(
        default_output_jsonl(args.input_jsonl, args.ngram_size, args.min_containment),
        args.output_dir,
    )
    default_summary = apply_output_dir(
        default_summary_json(args.input_jsonl, args.ngram_size, args.min_containment),
        args.output_dir,
    )
    output_jsonl = args.output_jsonl or default_output
    summary_json = args.summary_json or default_summary
    filtered_jsonls = {
        target: apply_output_dir(
            default_filtered_jsonl(
                args.input_jsonl,
                args.ngram_size,
                args.min_containment,
                target,
            ),
            args.output_dir,
        )
        for target in targets
    }

    with args.input_jsonl.open("r", encoding="utf-8") as f:
        tasks = [
            (index, line, args.ngram_size, args.min_containment, targets)
            for index, line in enumerate(f)
            if line.strip()
        ]

    # ここで実際の処理を行う。行インデックスを保持しながら並列処理を実行し、後で元の順序に戻す。
    indexed_rows = run_parallel(tasks, args.workers, args.chunksize)
    rows = [row for _, row in indexed_rows]

    write_jsonl(rows, output_jsonl)
    filtered_outputs = {}
    for target, filtered_jsonl in filtered_jsonls.items():
        filtered_count = write_jsonl(
            (row for row in rows if keep_after_excluding_ngram_contained(row, target)),
            filtered_jsonl,
        )
        filtered_outputs[target] = {
            "jsonl": str(filtered_jsonl),
            "rows": filtered_count,
            "removed_rows": len(rows) - filtered_count,
            "removed_condition": (f"{target}_in_prefix_ngram_contained == true"),
            "kept_condition": (f"{target}_in_prefix_ngram_contained == false"),
        }

    summary = summarize(
        rows,
        targets,
        args.input_jsonl,
        output_jsonl,
        summary_json,
        filtered_outputs,
        args.ngram_size,
        args.min_containment,
        args.workers,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"input_jsonl={args.input_jsonl}")
    print(f"output_jsonl={output_jsonl}")
    print(f"summary_json={summary_json}")
    for target, item in filtered_outputs.items():
        print(f"{target}_filtered_jsonl={item['jsonl']}")
    print(f"rows={summary['rows']}")
    print(f"ngram_size={args.ngram_size}")
    print(f"min_containment={args.min_containment}")
    print(f"workers={args.workers}")
    for target in targets:
        item = summary["targets"][target]
        print(
            f"{target}: exact={item['exact_count']} "
            f"ngram_contained={item['ngram_contained_count']} "
            f"avg_ngram_containment={item['avg_ngram_containment']:.12f}"
        )
    print("Reproducible command:")
    repro_command = [
        str(Path(__file__).resolve()),
        "--input-jsonl",
        str(args.input_jsonl),
    ]
    if args.output_dir is not None:
        repro_command.extend(["--output-dir", str(args.output_dir)])
    repro_command.extend(
        [
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--ngram-size",
            str(args.ngram_size),
            "--min-containment",
            str(args.min_containment),
            "--workers",
            str(args.workers),
            "--chunksize",
            str(args.chunksize),
        ]
    )
    print(" ".join(repro_command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

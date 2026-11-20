#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


ALLOWED_OPERATORS = {"<", "<=", ">", ">=", "==", "!="}


@dataclass(frozen=True)
class Rule:
    path: Path
    metric_key: str
    operator: str
    threshold: float

    @property
    def label(self) -> str:
        metric_name = self.metric_key if self.metric_key != "AUTO" else "AUTO"
        return f"{self.path.name}:{metric_name}{self.operator}{self.threshold}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PID 単位の複数メトリクス JSONL をまとめて適用し、"
            "条件を満たす文書だけを残した最終コーパス JSONL を作成する。"
        )
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="入力コーパス JSONL / JSONL.GZ。複数回指定可。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="最終的に結合して書き出す出力 JSONL / JSONL.GZ。",
    )
    parser.add_argument(
        "--rule",
        dest="rules",
        action="append",
        nargs=4,
        metavar=("JSONL", "METRIC_KEY", "OP", "THRESHOLD"),
        help=(
            "フィルタ条件。例: --rule path/to/pid_metrics.jsonl compression_ratio '<=' 0.8 "
            "METRIC_KEY に AUTO を指定すると pid 以外の数値キーを自動検出。"
        ),
    )
    parser.add_argument(
        "--rules-config",
        help="ルール一覧を書いた JSON ファイル。各要素は path, metric_key, operator, threshold を持つ。",
    )
    parser.add_argument(
        "--kept-pids-output",
        help="最終的に残した PID 一覧と各メトリクス値を保存する JSONL。",
    )
    parser.add_argument(
        "--summary-output",
        help="集計サマリ JSON の出力先。",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="テキストエンコーディング。デフォルトは utf-8。",
    )
    return parser.parse_args()


def open_text(path: Path, mode: str, encoding: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=encoding)
    return path.open(mode, encoding=encoding)


def iter_jsonl(path: Path, encoding: str) -> Iterator[Dict[str, Any]]:
    with open_text(path, "rt", encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_no}, got: {type(obj).__name__}"
                )
            yield obj


def normalize_pid(value: Any) -> Optional[str]:
    if value is None:
        return None
    pid = str(value).strip()
    if not pid:
        return None
    if pid.endswith(".tar.gz"):
        pid = pid[: -len(".tar.gz")]
    elif pid.endswith(".jsonl.gz"):
        pid = pid[: -len(".jsonl.gz")]
    elif pid.endswith(".jsonl"):
        pid = pid[: -len(".jsonl")]
    elif pid.endswith(".tar"):
        pid = pid[: -len(".tar")]
    return pid.strip() or None


def extract_corpus_pid(obj: Dict[str, Any]) -> Optional[str]:
    meta = obj.get("meta")
    if isinstance(meta, dict):
        normalized = normalize_pid(meta.get("pid"))
        if normalized is not None:
            return normalized
    return normalize_pid(obj.get("pid"))


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def auto_detect_metric_key(path: Path, encoding: str) -> str:
    candidate_keys: Optional[set[str]] = None
    inspected = 0
    for obj in iter_jsonl(path, encoding):
        numeric_keys = {key for key, value in obj.items() if key != "pid" and is_numeric(value)}
        if not numeric_keys:
            continue
        if candidate_keys is None:
            candidate_keys = set(numeric_keys)
        else:
            candidate_keys &= numeric_keys
        inspected += 1
        if inspected >= 50:
            break
    if not candidate_keys:
        raise ValueError(f"Could not auto-detect metric key from {path}")
    if len(candidate_keys) != 1:
        candidates = ", ".join(sorted(candidate_keys))
        raise ValueError(
            f"Metric key is ambiguous for {path}. "
            f"Specify it explicitly instead of AUTO. Candidates: {candidates}"
        )
    return next(iter(candidate_keys))


def load_rules(args: argparse.Namespace) -> List[Rule]:
    loaded: List[Rule] = []
    if args.rules:
        for raw_path, metric_key, operator, raw_threshold in args.rules:
            if operator not in ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported operator: {operator}")
            loaded.append(
                Rule(
                    path=Path(raw_path),
                    metric_key=metric_key,
                    operator=operator,
                    threshold=float(raw_threshold),
                )
            )

    if args.rules_config:
        config_path = Path(args.rules_config)
        with config_path.open("r", encoding=args.encoding) as handle:
            config = json.load(handle)
        if not isinstance(config, list):
            raise ValueError(f"{config_path} must contain a JSON list")
        for item in config:
            if not isinstance(item, dict):
                raise ValueError(f"Each rule in {config_path} must be a JSON object")
            operator = item["operator"]
            if operator not in ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported operator: {operator}")
            loaded.append(
                Rule(
                    path=config_path.resolve().parent / item["path"],
                    metric_key=str(item.get("metric_key", "AUTO")),
                    operator=operator,
                    threshold=float(item["threshold"]),
                )
            )

    if not loaded:
        raise ValueError("At least one --rule or --rules-config must be provided")

    normalized_rules: List[Rule] = []
    for rule in loaded:
        if not rule.path.is_file():
            raise FileNotFoundError(f"Rule JSONL not found: {rule.path}")
        metric_key = rule.metric_key
        if metric_key == "AUTO":
            metric_key = auto_detect_metric_key(rule.path, args.encoding)
        normalized_rules.append(
            Rule(
                path=rule.path,
                metric_key=metric_key,
                operator=rule.operator,
                threshold=rule.threshold,
            )
        )
    return normalized_rules


def compare(value: float, operator: str, threshold: float) -> bool:
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "==":
        return value == threshold
    if operator == "!=":
        return value != threshold
    raise ValueError(f"Unsupported operator: {operator}")


def load_rule_values(rule: Rule, encoding: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    values_by_pid: Dict[str, float] = {}
    stats = {
        "rows": 0,
        "usable_rows": 0,
        "duplicate_pid_rows": 0,
        "passed_pids": 0,
    }
    for obj in iter_jsonl(rule.path, encoding):
        stats["rows"] += 1
        pid = normalize_pid(obj.get("pid"))
        value = obj.get(rule.metric_key)
        if pid is None or not is_numeric(value):
            continue
        stats["usable_rows"] += 1
        if pid in values_by_pid:
            stats["duplicate_pid_rows"] += 1
        values_by_pid[pid] = float(value)
    stats["passed_pids"] = sum(
        1 for value in values_by_pid.values() if compare(value, rule.operator, rule.threshold)
    )
    return values_by_pid, stats


def determine_kept_pids(
    rules: Sequence[Rule],
    encoding: str,
) -> Tuple[set[str], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    passing_sets: List[set[str]] = []
    values_per_rule: Dict[str, Dict[str, float]] = {}
    rule_summaries: List[Dict[str, Any]] = []

    for rule in rules:
        values_by_pid, stats = load_rule_values(rule, encoding)
        passed = {
            pid
            for pid, value in values_by_pid.items()
            if compare(value, rule.operator, rule.threshold)
        }
        passing_sets.append(passed)
        values_per_rule[rule.label] = values_by_pid
        rule_summaries.append(
            {
                "path": str(rule.path),
                "metric_key": rule.metric_key,
                "operator": rule.operator,
                "threshold": rule.threshold,
                **stats,
            }
        )

    kept_pids = set.intersection(*passing_sets) if passing_sets else set()
    return kept_pids, values_per_rule, rule_summaries


def stream_filter_corpora(
    inputs: Sequence[Path],
    output_path: Path,
    kept_pids: set[str],
    encoding: str,
) -> List[Dict[str, Any]]:
    if output_path.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("Input and output must be different files")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_summaries: List[Dict[str, Any]] = []

    with open_text(output_path, "wt", encoding) as out_handle:
        for input_path in inputs:
            summary = {
                "path": str(input_path),
                "records_read": 0,
                "records_written": 0,
                "records_missing_pid": 0,
                "records_filtered_out": 0,
            }
            for obj in iter_jsonl(input_path, encoding):
                summary["records_read"] += 1
                pid = extract_corpus_pid(obj)
                if pid is None:
                    summary["records_missing_pid"] += 1
                    continue
                if pid not in kept_pids:
                    summary["records_filtered_out"] += 1
                    continue
                out_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                summary["records_written"] += 1
            corpus_summaries.append(summary)

    return corpus_summaries


def write_kept_pid_report(
    path: Path,
    kept_pids: Iterable[str],
    values_per_rule: Dict[str, Dict[str, float]],
    encoding: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rule_labels = sorted(values_per_rule.keys())
    with open_text(path, "wt", encoding) as handle:
        for pid in sorted(kept_pids):
            row: Dict[str, Any] = {"pid": pid}
            for label in sorted_rule_labels:
                row[label] = values_per_rule[label].get(pid)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    input_paths = [Path(path) for path in args.inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input corpus not found: {path}")

    output_path = Path(args.output)
    rules = load_rules(args)
    kept_pids, values_per_rule, rule_summaries = determine_kept_pids(rules, args.encoding)
    corpus_summaries = stream_filter_corpora(input_paths, output_path, kept_pids, args.encoding)

    if args.kept_pids_output:
        write_kept_pid_report(
            Path(args.kept_pids_output), kept_pids, values_per_rule, args.encoding
        )

    summary = {
        "output": str(output_path),
        "kept_pid_count": len(kept_pids),
        "rules": rule_summaries,
        "inputs": corpus_summaries,
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding=args.encoding) as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

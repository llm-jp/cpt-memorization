#!/usr/bin/env python3
"""Summarize MinHash page-level decision logs into PID-level duplicate ratios.

Input directory should contain JSONL files created by minhash_dedup.py with --audit-pages,
typically: <output>/minhash-*/results/page_audit/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate page-level dedup decisions into PID-level duplicate ratios.",
    )
    parser.add_argument(
        "--audit-dir",
        required=True,
        help="Directory containing page_audit JSONL files.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL path for high-duplication PID records.",
    )
    parser.add_argument(
        "--dup-ratio-threshold",
        type=float,
        default=None,
        help=(
            "PID is reported when removed_pages / total_pages >= threshold. "
            "If omitted, all PIDs are reported."
        ),
    )
    parser.add_argument(
        "--min-pages",
        type=int,
        default=1,
        help="Minimum total pages required for a PID to be considered.",
    )
    return parser.parse_args()


def iter_audit_records(audit_dir: Path):
    for path in sorted(audit_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def main() -> None:
    args = parse_args()
    if args.dup_ratio_threshold is not None and not (0.0 <= args.dup_ratio_threshold <= 1.0):
        raise ValueError(f"--dup-ratio-threshold must be in [0, 1]. got={args.dup_ratio_threshold}")
    if args.min_pages < 1:
        raise ValueError(f"--min-pages must be >= 1. got={args.min_pages}")

    audit_dir = Path(args.audit_dir)
    if not audit_dir.exists() or not audit_dir.is_dir():
        raise FileNotFoundError(f"Audit directory not found: {audit_dir}")

    latest_decision_by_page: dict[str, dict] = {}
    for row in iter_audit_records(audit_dir):
        page_id = row.get("page_id")
        pid = row.get("pid")
        remove_flag = row.get("remove_flag")
        if page_id is None:
            page_index = row.get("page_index")
            if pid is None or page_index is None:
                continue
            page_id = f"{pid}:{page_index}"
        if pid is None:
            continue
        if not isinstance(remove_flag, bool):
            continue
        latest_decision_by_page[str(page_id)] = {
            "pid": str(pid),
            "remove_flag": remove_flag,
        }

    pid_stats: dict[str, dict[str, int]] = {}
    for row in latest_decision_by_page.values():
        pid = row["pid"]
        rec = pid_stats.setdefault(pid, {"total_pages": 0, "removed_pages": 0})
        rec["total_pages"] += 1
        if row["remove_flag"]:
            rec["removed_pages"] += 1

    selected_rows = []
    for pid, rec in pid_stats.items():
        total_pages = rec["total_pages"]
        removed_pages = rec["removed_pages"]
        if total_pages < args.min_pages:
            continue
        dup_ratio = removed_pages / total_pages
        if args.dup_ratio_threshold is None or dup_ratio >= args.dup_ratio_threshold:
            selected_rows.append(
                {
                    "pid": pid,
                    "total_pages": total_pages,
                    "removed_pages": removed_pages,
                    "kept_pages": total_pages - removed_pages,
                    "dup_ratio": dup_ratio,
                }
            )

    selected_rows.sort(key=lambda x: (-x["dup_ratio"], -x["removed_pages"], x["pid"]))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in selected_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"pid_dup_summary_done: total_pid={len(pid_stats)} selected_pid={len(selected_rows)} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

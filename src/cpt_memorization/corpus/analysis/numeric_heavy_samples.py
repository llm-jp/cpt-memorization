#!/usr/bin/env python3
"""Sample numeric-heavy records from gzipped JSONL shards."""

import argparse
import gzip
import json
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def is_numeric_heavy(text: str, *, min_digits: int, digit_ratio: float) -> bool:
    digits = sum(c.isdigit() for c in text)
    if digits < min_digits:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters == 0:
        return True
    return digits >= digit_ratio * letters


def iter_jsonl_records(path: Path) -> Iterator[Dict]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarize_pages(pages: Iterable[Dict], limit: int) -> str:
    items: List[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        name = page.get("image_name", "?")
        lines = page.get("line_count", "?")
        items.append(f"{name} (lines={lines})")
        if len(items) >= limit:
            break
    return ", ".join(items) if items else "-"


def find_numeric_examples(
    shards: Iterable[Path],
    *,
    max_examples: int,
    min_digits: int,
    digit_ratio: float,
    page_limit: int,
    snippet_chars: int,
) -> List[str]:
    blocks: List[str] = []
    for shard in shards:
        for index, record in enumerate(iter_jsonl_records(shard)):
            text = record.get("text", "")
            if not text or not is_numeric_heavy(
                text, min_digits=min_digits, digit_ratio=digit_ratio
            ):
                continue
            meta_obj = record.get("metadata")
            meta = meta_obj if isinstance(meta_obj, dict) else {}
            rich_obj = meta.get("meta")
            rich = rich_obj if isinstance(rich_obj, dict) else {}
            pages_obj = rich.get("pages")
            pages = pages_obj if isinstance(pages_obj, list) else []
            snippet = text.strip()
            if len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars] + " ..."
            block = [
                "=" * 80,
                f"source_file: {shard.name} | record: {index}",
                f"cluster_id: {meta.get('minhash_cluster_id')} (size={meta.get('minhash_cluster_size')})",
                f"file_path: {meta.get('file_path')}",
                f"pid: {rich.get('pid')} | archive_member: {rich.get('archive_member')}",
                f"page_count: {rich.get('page_count')} | line_count: {rich.get('line_count')}",
                f"pages: {summarize_pages(pages, page_limit)}",
                "text snippet:",
                snippet,
            ]
            blocks.append("\n".join(str(part) for part in block))
            if len(blocks) >= max_examples:
                return blocks
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default=("data/*.jsonl.gz"),
        help="Glob pattern pointing at gzipped JSONL shards.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=5, help="Maximum number of records to emit."
    )
    parser.add_argument(
        "--min-digits", type=int, default=20, help="Minimum digit count required for a match."
    )
    parser.add_argument(
        "--digit-ratio",
        type=float,
        default=2.0,
        help="Required digit-to-letter ratio (digits >= ratio * letters).",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=3,
        help="Maximum number of page entries to summarize per record.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=600,
        help="Maximum length of the text snippet before truncation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write the report. Defaults to stdout.",
    )
    return parser.parse_args()


def collect_shards(pattern: str) -> List[Path]:
    expanded = str(Path(pattern).expanduser())
    matches = [Path(p) for p in sorted(glob(expanded, recursive=True))]
    return matches


def main() -> int:
    args = parse_args()
    shards = collect_shards(args.pattern)
    if not shards:
        raise SystemExit(f"No shards matched pattern: {args.pattern}")

    blocks = find_numeric_examples(
        shards,
        max_examples=args.max_examples,
        min_digits=args.min_digits,
        digit_ratio=args.digit_ratio,
        page_limit=args.page_limit,
        snippet_chars=args.snippet_chars,
    )
    report = "\n".join(blocks) if blocks else "No numeric-heavy examples found in sampled records."

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

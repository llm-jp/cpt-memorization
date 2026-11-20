#!/usr/bin/env python3
import argparse
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from tqdm import tqdm


KODAK_PHRASE_RE = re.compile(r"kodak\s+color\s+control\s+patches", re.IGNORECASE)


@dataclass
class FileStats:
    input_path: Path
    output_path: Path
    total_records: int = 0
    removed_records: int = 0


def open_text_for_read(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def open_text_for_write(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def is_edge_page(obj: dict) -> bool:
    meta = obj.get("meta") or {}
    page_index = meta.get("page_index")
    page_count = meta.get("page_count")

    if not isinstance(page_index, int):
        return False
    if page_index == 0:
        return True
    if isinstance(page_count, int) and page_count > 0 and page_index == page_count - 1:
        return True
    return False


def has_kodak_phrase(obj: dict) -> bool:
    text = obj.get("text")
    if not isinstance(text, str):
        return False
    return KODAK_PHRASE_RE.search(text) is not None


def process_file(input_path: Path, output_path: Path, show_progress: bool) -> FileStats:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = FileStats(input_path=input_path, output_path=output_path)

    with open_text_for_read(input_path) as fin, open_text_for_write(output_path) as fout:
        for raw_line in tqdm(
            fin,
            desc=f"processing {input_path.name}",
            unit="lines",
            disable=not show_progress,
        ):
            line = raw_line.strip()
            if not line:
                continue

            stats.total_records += 1

            # Fast path: if the Kodak phrase is not present in raw JSON text,
            # keep the line as-is without JSON parsing.
            if KODAK_PHRASE_RE.search(line) is None:
                fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            # Slow path only for likely candidates.
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            if not isinstance(obj, dict):
                fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            if is_edge_page(obj) and has_kodak_phrase(obj):
                stats.removed_records += 1
                continue

            fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove records that contain 'kodak color control patches' "
            "only when they are first/last pages."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL or JSONL.GZ files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write cleaned files",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    total_records = 0
    total_removed = 0

    for input_path in args.inputs:
        if not input_path.exists():
            raise SystemExit(f"Input not found: {input_path}")

        output_path = args.output_dir / input_path.name
        stats = process_file(input_path, output_path, show_progress=not args.no_progress)
        total_records += stats.total_records
        total_removed += stats.removed_records

        print(
            f"[{input_path.name}] total={stats.total_records} "
            f"removed={stats.removed_records} kept={stats.total_records - stats.removed_records} "
            f"output={output_path}"
        )

    print(
        f"[ALL] total={total_records} removed={total_removed} kept={total_records - total_removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

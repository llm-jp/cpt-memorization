#!/usr/bin/env python3
"""
Compute 'geta' symbol stats per document (PID).
Aggregates page-level data to determine if a document contains the error symbol.
"""

import argparse
import gzip
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Set, Tuple, List

try:
    import orjson
except ImportError:
    orjson = None

try:
    import zstandard as zstd
except ImportError:
    zstd = None


def open_maybe_compressed(path: Path):
    """Open a file that might be compressed with gzip or zstd, or plain text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    elif path.suffix == ".zst":
        if zstd is None:
            raise ImportError("zstandard module is not installed")
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(open(path, "rb"))
    else:
        return open(path, "r", encoding="utf-8")


def process_file(file_path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Process a single JSONL file.
    Returns:
        pids_with_error: Set of PIDs that contain the 'geta' symbol.
        all_pids: Set of all PIDs found in the file.
    """
    pids_with_error = set()
    all_pids = set()

    try:
        with open_maybe_compressed(file_path) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    if orjson:
                        data = orjson.loads(line)
                    else:
                        data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pid = data.get("meta", {}).get("pid")
                if not pid:
                    continue

                all_pids.add(pid)

                text = data.get("text", "")
                if "〓" in text:
                    pids_with_error.add(pid)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return set(), set()

    return pids_with_error, all_pids


def main():
    parser = argparse.ArgumentParser(description="Compute per-document OCR error stats.")
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Input directory containing JSONL files"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes"
    )
    args = parser.parse_args()

    input_files = (
        list(args.input_dir.glob("*.jsonl"))
        + list(args.input_dir.glob("*.jsonl.gz"))
        + list(args.input_dir.glob("*.jsonl.zst"))
    )

    print(f"Found {len(input_files)} files to process.")

    global_pids_with_error = set()
    global_all_pids = set()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(process_file, p): p for p in input_files}

        for i, future in enumerate(as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                pids_error, pids_all = future.result()
                global_pids_with_error.update(pids_error)
                global_all_pids.update(pids_all)
            except Exception as e:
                print(f"File {file_path} generated an exception: {e}")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(input_files)} files...")

    total_docs = len(global_all_pids)
    error_docs = len(global_pids_with_error)

    print("-" * 30)
    print(f"Total Documents (PIDs): {total_docs}")
    print(f"Documents with '〓' (Error): {error_docs}")
    if total_docs > 0:
        ratio = error_docs / total_docs
        print(f"Error Document Ratio: {ratio:.4f} ({ratio * 100:.2f}%)")
    else:
        print("Error Document Ratio: N/A")
    print("-" * 30)


if __name__ == "__main__":
    main()

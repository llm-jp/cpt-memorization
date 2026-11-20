#!/usr/bin/env python3
"""
Compute distribution of 'geta' symbol error rates per document (PID).
Calculates Min, Max, Mean, Median, and other percentiles.
"""

import argparse
import gzip
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

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


def process_file(file_path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Process a single JSONL file.
    Returns:
        pid_stats: Dict mapping PID to (error_count, char_count)
    """
    pid_stats: Dict[str, Tuple[int, int]] = {}

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

                text = data.get("text", "")
                char_count = len(text)
                error_count = text.count("〓")

                if pid not in pid_stats:
                    pid_stats[pid] = [0, 0]

                pid_stats[pid][0] += error_count
                pid_stats[pid][1] += char_count

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

    return pid_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-document OCR error rate distribution."
    )
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

    global_pid_stats: Dict[str, Tuple[int, int]] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {executor.submit(process_file, p): p for p in input_files}

        for i, future in enumerate(as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                file_stats = future.result()
                for pid, (err, chars) in file_stats.items():
                    if pid not in global_pid_stats:
                        global_pid_stats[pid] = [0, 0]
                    global_pid_stats[pid][0] += err
                    global_pid_stats[pid][1] += chars
            except Exception as e:
                print(f"File {file_path} generated an exception: {e}")

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(input_files)} files...")

    # Calculate rates per 1M chars
    rates = []
    for pid, (err, chars) in global_pid_stats.items():
        if chars > 0:
            rate = (err / chars) * 1_000_000
            rates.append(rate)

    if not rates:
        print("No data found.")
        return

    rates = np.array(rates)

    print("-" * 30)
    print(f"Total Documents: {len(rates)}")
    print(f"Min: {np.min(rates):.2f}")
    print(f"Max: {np.max(rates):.2f}")
    print(f"Mean: {np.mean(rates):.2f}")
    print(f"Median: {np.median(rates):.2f}")
    print(f"25th Percentile: {np.percentile(rates, 25):.2f}")
    print(f"75th Percentile: {np.percentile(rates, 75):.2f}")
    print(f"90th Percentile: {np.percentile(rates, 90):.2f}")
    print(f"95th Percentile: {np.percentile(rates, 95):.2f}")
    print(f"99th Percentile: {np.percentile(rates, 99):.2f}")
    print("-" * 30)

    # Calculate percentage in range 1,500 - 6,000
    in_range = np.sum((rates >= 1500) & (rates <= 6000))
    ratio_in_range = in_range / len(rates)
    print(f"Documents in range [1,500 - 6,000]: {in_range} ({ratio_in_range * 100:.2f}%)")
    print("-" * 30)


if __name__ == "__main__":
    main()

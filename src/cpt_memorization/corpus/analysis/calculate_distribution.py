import os
import json
import argparse
import statistics
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def process_file(file_path):
    # Returns map: pid -> {'chars': int, 'geta': int}
    pid_stats = defaultdict(lambda: {"chars": 0, "geta": 0})
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Extract PID
                    pid = None
                    if "meta" in data:
                        if isinstance(data["meta"], dict):
                            pid = data["meta"].get("pid")
                    if pid is None:
                        pid = data.get("pid")

                    if not pid:
                        continue

                    text = data.get("text", "")
                    if text:
                        pid_stats[pid]["chars"] += len(text)
                        pid_stats[pid]["geta"] += text.count("〓")

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

    return dict(pid_stats)


def main():
    parser = argparse.ArgumentParser(description="Calculate distribution of '〓' per 1M chars.")
    parser.add_argument("input_dir", help="Directory containing .jsonl files")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of worker processes"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = list(input_dir.glob("*.jsonl"))

    if not files:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"Processing {len(files)} files in {input_dir}...")

    # Aggregate stats by PID across all files
    # Note: PIDs might be split across files, though usually grouped.
    # We will merge results.

    final_stats = defaultdict(lambda: {"chars": 0, "geta": 0})

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files)))

        print("Aggregating results...")
        for res in results:
            for pid, counts in res.items():
                final_stats[pid]["chars"] += counts["chars"]
                final_stats[pid]["geta"] += counts["geta"]

    # Calculate rates
    rates = []
    for pid, counts in final_stats.items():
        chars = counts["chars"]
        geta = counts["geta"]
        if chars > 0:
            rate = (geta / chars) * 1_000_000
            rates.append(rate)
        else:
            # Handle empty? If chars is 0 but pid exists (e.g. empty text page), rate is undefined or 0 if geta is 0.
            # If geta > 0 and chars 0, impossible unless geta not in chars?
            # Assuming chars=0 means no text content to analyze.
            pass

    if not rates:
        print("No valid data found.")
        return

    rates.sort()
    n = len(rates)

    print("-" * 40)
    print(f"Analysis for: {input_dir}")
    print(f"Total Documents: {n}")
    print(f"Mean rate: {statistics.mean(rates):.2f}")
    if n > 1:
        print(f"Median rate: {statistics.median(rates):.2f}")

    # Percentiles
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    print("\nPercentiles (measure: occurrences per 1M chars):")
    for p in percentiles:
        # Simple index calculation
        idx = int((p / 100) * (n - 1))
        val = rates[idx]
        print(f"{p}%ile: {val:.2f}")

    print("\nMin: {:.2f}".format(rates[0]))
    print("Max: {:.2f}".format(rates[-1]))
    print("-" * 40)


if __name__ == "__main__":
    main()

import os
import json
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm


def process_file(file_path):
    unique_pids = set()
    total_lines = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Try to find 'pid' in 'meta' or root
                    pid = None
                    if "meta" in data:
                        if isinstance(data["meta"], dict):
                            pid = data["meta"].get("pid")

                    if pid is None:
                        pid = data.get("pid")

                    if pid:
                        unique_pids.add(pid)

                    total_lines += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return set(), 0

    return unique_pids, total_lines


def main():
    parser = argparse.ArgumentParser(
        description="Count unique PIDs and total pages/lines in JSONL files."
    )
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

    print(f"Found {len(files)} files. Processing with {args.workers} workers...")

    all_pids = set()
    total_page_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files)))

        for pids, count in results:
            all_pids.update(pids)
            total_page_count += count

    print("-" * 30)
    print(f"Input Directory: {input_dir}")
    print(f"Total Unique Documents (PIDs): {len(all_pids)}")
    print(f"Total Pages (Lines): {total_page_count}")
    print("-" * 30)


if __name__ == "__main__":
    main()

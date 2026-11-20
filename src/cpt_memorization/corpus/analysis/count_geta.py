import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm


def process_file(file_path):
    geta_count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        geta_count += text.count("〓")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

    return geta_count


def main():
    parser = argparse.ArgumentParser(description="Count '〓' symbols in JSONL files.")
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

    total_geta_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files)))
        total_geta_count = sum(results)

    print("-" * 30)
    print(f"Input Directory: {input_dir}")
    print(f"Total '〓' (Geta) Count: {total_geta_count}")
    print("-" * 30)


if __name__ == "__main__":
    main()

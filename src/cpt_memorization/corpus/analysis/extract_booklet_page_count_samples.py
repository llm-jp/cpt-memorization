import argparse
import csv
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def pid_variants(pid):
    pid = str(pid)
    variants = {pid}
    if pid.endswith(".tar"):
        variants.add(pid[: -len(".tar")])
    else:
        variants.add(f"{pid}.tar")
    return variants


def choose_samples(booklet_csv, min_page_count, max_page_count, samples_per_page_count):
    by_page_count = {page_count: [] for page_count in range(min_page_count, max_page_count + 1)}
    with open(booklet_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                page_count = int(row["page_count"])
            except (KeyError, TypeError, ValueError):
                continue
            if min_page_count <= page_count <= max_page_count:
                by_page_count[page_count].append(row)

    selected = []
    for page_count in range(max_page_count, min_page_count - 1, -1):
        rows = by_page_count.get(page_count, [])
        for row in rows[:samples_per_page_count]:
            selected.append(
                {
                    "requested_page_count": page_count,
                    "pid": str(row["pid"]),
                    "csv_row": row,
                    "pid_variants": sorted(pid_variants(row["pid"])),
                }
            )
    return selected, by_page_count


def iter_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            yield line_number, line


def record_pid(record):
    meta = record.get("meta")
    if isinstance(meta, dict) and meta.get("pid") is not None:
        return str(meta["pid"])
    if record.get("pid") is not None:
        return str(record["pid"])
    return None


def page_index(record):
    meta = record.get("meta")
    value = None
    if isinstance(meta, dict):
        value = meta.get("page_index")
    if value is None:
        value = record.get("page_index")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def process_file(args):
    file_path, dataset_label, variant_to_sample = args
    matched = []
    counters = {
        "file_path": str(file_path),
        "dataset": dataset_label,
        "lines": 0,
        "invalid_json_lines": 0,
        "matched_lines": 0,
    }

    variant_needles = []
    for pid in variant_to_sample:
        variant_needles.append(f'"pid": "{pid}"')
        variant_needles.append(f'"pid":"{pid}"')
    for line_number, line in iter_jsonl(file_path):
        counters["lines"] += 1
        if variant_needles and not any(needle in line for needle in variant_needles):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            counters["invalid_json_lines"] += 1
            continue

        pid = record_pid(record)
        sample_key = variant_to_sample.get(pid)
        if sample_key is None:
            continue

        counters["matched_lines"] += 1
        matched.append(
            {
                "sample_key": sample_key,
                "dataset": dataset_label,
                "source_file": str(file_path),
                "source_line": line_number,
                "page_index": page_index(record),
                "record": record,
            }
        )

    return {"matched": matched, "counters": counters}


def safe_sample_key(page_count, pid):
    return f"page_count_{page_count:02d}_pid_{str(pid).replace('/', '_')}"


def write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in records:
            json.dump(item["record"], f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract complete booklet samples by page-count bucket."
    )
    parser.add_argument("--booklet-csv", required=True, help="Input booklet_page_counts.csv")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Input JSONL shard directory. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True, help="Output sample directory")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of worker processes"
    )
    parser.add_argument(
        "--glob", default="*.jsonl", help="Input file glob relative to each input directory"
    )
    parser.add_argument("--min-page-count", type=int, default=1)
    parser.add_argument("--max-page-count", type=int, default=10)
    parser.add_argument("--samples-per-page-count", type=int, default=1)
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit(f"--workers must be positive. Got: {args.workers}")
    if args.min_page_count < 1:
        raise SystemExit(f"--min-page-count must be positive. Got: {args.min_page_count}")
    if args.max_page_count < args.min_page_count:
        raise SystemExit("--max-page-count must be greater than or equal to --min-page-count")
    if args.samples_per_page_count < 1:
        raise SystemExit(
            f"--samples-per-page-count must be positive. Got: {args.samples_per_page_count}"
        )

    booklet_csv = Path(args.booklet_csv)
    output_dir = Path(args.output_dir)
    if not booklet_csv.is_file():
        raise SystemExit(f"Booklet CSV not found: {booklet_csv}")

    input_dirs = [Path(input_dir) for input_dir in args.input_dir]
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")

    selected, by_page_count = choose_samples(
        booklet_csv,
        args.min_page_count,
        args.max_page_count,
        args.samples_per_page_count,
    )
    if not selected:
        raise SystemExit("No samples selected from booklet CSV.")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = {}
    variant_to_sample = {}
    for item in selected:
        sample_key = safe_sample_key(item["requested_page_count"], item["pid"])
        item["sample_key"] = sample_key
        samples[sample_key] = item
        for variant in item["pid_variants"]:
            variant_to_sample[variant] = sample_key

    files = []
    for input_dir in input_dirs:
        dataset_label = input_dir.name
        for file_path in sorted(input_dir.glob(args.glob)):
            files.append((file_path, dataset_label, variant_to_sample))
    if not files:
        raise SystemExit(f"No files matched {args.glob!r} under input directories.")

    matched_by_sample_dataset = {}
    file_counters = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(process_file, files, chunksize=8):
            file_counters.append(result["counters"])
            for matched in result["matched"]:
                key = (matched["sample_key"], matched["dataset"])
                matched_by_sample_dataset.setdefault(key, []).append(matched)

    for (sample_key, dataset), records in matched_by_sample_dataset.items():
        records.sort(
            key=lambda item: (
                item["page_index"] is None,
                item["page_index"] if item["page_index"] is not None else -1,
                item["source_file"],
                item["source_line"],
            )
        )
        sample_dir = output_dir / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(records, sample_dir / f"{dataset}.jsonl")

    manifest_rows = []
    for sample_key, item in samples.items():
        for input_dir in input_dirs:
            dataset = input_dir.name
            records = matched_by_sample_dataset.get((sample_key, dataset), [])
            page_indexes = sorted(
                item["page_index"] for item in records if item["page_index"] is not None
            )
            manifest_rows.append(
                {
                    "sample_key": sample_key,
                    "requested_page_count": item["requested_page_count"],
                    "pid": item["pid"],
                    "pid_variants": "|".join(item["pid_variants"]),
                    "dataset": dataset,
                    "matched_records": len(records),
                    "min_page_index": page_indexes[0] if page_indexes else "",
                    "max_page_index": page_indexes[-1] if page_indexes else "",
                    "output_jsonl": str(output_dir / sample_key / f"{dataset}.jsonl")
                    if records
                    else "",
                }
            )

    manifest_csv = output_dir / "selected_booklets.csv"
    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_key",
            "requested_page_count",
            "pid",
            "pid_variants",
            "dataset",
            "matched_records",
            "min_page_index",
            "max_page_index",
            "output_jsonl",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "booklet_csv": str(booklet_csv),
        "input_dirs": [str(input_dir) for input_dir in input_dirs],
        "output_dir": str(output_dir),
        "glob": args.glob,
        "workers": args.workers,
        "min_page_count": args.min_page_count,
        "max_page_count": args.max_page_count,
        "samples_per_page_count": args.samples_per_page_count,
        "available_booklets_by_page_count": {
            str(page_count): len(by_page_count.get(page_count, []))
            for page_count in range(args.max_page_count, args.min_page_count - 1, -1)
        },
        "selected_samples": selected,
        "manifest_csv": str(manifest_csv),
        "input_files": len(files),
        "processed_lines": sum(counter["lines"] for counter in file_counters),
        "invalid_json_lines": sum(counter["invalid_json_lines"] for counter in file_counters),
        "matched_lines": sum(counter["matched_lines"] for counter in file_counters),
    }
    summary_json = output_dir / "extraction_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Output directory: {output_dir}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Selected samples: {len(selected)}")
    print(f"Input files: {len(files)}")
    print(f"Processed lines: {summary['processed_lines']}")
    print(f"Matched lines: {summary['matched_lines']}")


if __name__ == "__main__":
    main()

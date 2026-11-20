import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm


def extract_meta(record):
    meta = record.get("meta")
    if isinstance(meta, dict):
        return meta
    return {}


def get_meta_or_root(record, meta, key):
    if key in meta:
        return meta[key]
    return record.get(key)


def process_file(file_path):
    stats = {}
    total_lines = 0
    invalid_json_lines = 0
    missing_pid_lines = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue

            meta = extract_meta(record)
            pid = meta.get("pid") or record.get("pid")
            if not pid:
                missing_pid_lines += 1
                continue

            page_count = get_meta_or_root(record, meta, "page_count")
            page_index = get_meta_or_root(record, meta, "page_index")

            pid_stats = stats.setdefault(
                str(pid),
                {
                    "observed_pages": 0,
                    "declared_page_counts": Counter(),
                    "min_page_index": None,
                    "max_page_index": None,
                    "source_files": set(),
                },
            )
            pid_stats["observed_pages"] += 1
            pid_stats["source_files"].add(str(file_path))

            if isinstance(page_count, int):
                pid_stats["declared_page_counts"][page_count] += 1
            elif isinstance(page_count, str) and page_count.isdigit():
                pid_stats["declared_page_counts"][int(page_count)] += 1

            if isinstance(page_index, int):
                if pid_stats["min_page_index"] is None or page_index < pid_stats["min_page_index"]:
                    pid_stats["min_page_index"] = page_index
                if pid_stats["max_page_index"] is None or page_index > pid_stats["max_page_index"]:
                    pid_stats["max_page_index"] = page_index

    return {
        "file_path": str(file_path),
        "stats": stats,
        "total_lines": total_lines,
        "invalid_json_lines": invalid_json_lines,
        "missing_pid_lines": missing_pid_lines,
    }


def merge_results(results):
    merged = {}
    totals = {
        "input_dirs": [],
        "input_files": 0,
        "total_lines": 0,
        "invalid_json_lines": 0,
        "missing_pid_lines": 0,
    }

    for result in results:
        totals["input_files"] += 1
        totals["total_lines"] += result["total_lines"]
        totals["invalid_json_lines"] += result["invalid_json_lines"]
        totals["missing_pid_lines"] += result["missing_pid_lines"]

        for pid, stats in result["stats"].items():
            pid_stats = merged.setdefault(
                pid,
                {
                    "observed_pages": 0,
                    "declared_page_counts": Counter(),
                    "min_page_index": None,
                    "max_page_index": None,
                    "source_files": set(),
                },
            )
            pid_stats["observed_pages"] += stats["observed_pages"]
            pid_stats["declared_page_counts"].update(stats["declared_page_counts"])
            pid_stats["source_files"].update(stats["source_files"])

            min_page_index = stats["min_page_index"]
            if min_page_index is not None:
                if (
                    pid_stats["min_page_index"] is None
                    or min_page_index < pid_stats["min_page_index"]
                ):
                    pid_stats["min_page_index"] = min_page_index

            max_page_index = stats["max_page_index"]
            if max_page_index is not None:
                if (
                    pid_stats["max_page_index"] is None
                    or max_page_index > pid_stats["max_page_index"]
                ):
                    pid_stats["max_page_index"] = max_page_index

    return merged, totals


def choose_page_count(stats):
    declared_counts = stats["declared_page_counts"]
    if declared_counts:
        return declared_counts.most_common(1)[0][0]
    return stats["observed_pages"]


def percentile(sorted_values, percent):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percent / 100) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_booklet_rows(booklet_stats):
    rows = []
    for pid, stats in booklet_stats.items():
        declared_counts = stats["declared_page_counts"]
        declared_values = sorted(declared_counts)
        chosen_page_count = choose_page_count(stats)
        observed_pages = stats["observed_pages"]
        declared_min = declared_values[0] if declared_values else ""
        declared_max = declared_values[-1] if declared_values else ""
        declared_unique_count = len(declared_values)
        has_declared_mismatch = declared_unique_count > 1
        observed_matches_declared = chosen_page_count == observed_pages

        rows.append(
            {
                "pid": pid,
                "page_count": chosen_page_count,
                "observed_pages": observed_pages,
                "declared_page_count_min": declared_min,
                "declared_page_count_max": declared_max,
                "declared_page_count_unique_values": declared_unique_count,
                "has_declared_page_count_mismatch": has_declared_mismatch,
                "observed_matches_declared": observed_matches_declared,
                "min_page_index": stats["min_page_index"]
                if stats["min_page_index"] is not None
                else "",
                "max_page_index": stats["max_page_index"]
                if stats["max_page_index"] is not None
                else "",
                "source_file_count": len(stats["source_files"]),
            }
        )
    rows.sort(key=lambda row: (0, int(row["pid"])) if row["pid"].isdigit() else (1, row["pid"]))
    return rows


def summarize(rows, totals):
    page_counts = sorted(row["page_count"] for row in rows)
    if not page_counts:
        return {
            **totals,
            "booklets": 0,
            "total_pages": 0,
            "page_count_stats": {},
            "quality": {},
        }

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    page_count_stats = {
        "min": page_counts[0],
        "max": page_counts[-1],
        "mean": statistics.fmean(page_counts),
        "median": statistics.median(page_counts),
        "percentiles": {f"p{p}": percentile(page_counts, p) for p in percentiles},
    }
    quality = {
        "booklets_without_declared_page_count": sum(
            1 for row in rows if row["declared_page_count_unique_values"] == 0
        ),
        "booklets_with_declared_page_count_mismatch": sum(
            1 for row in rows if row["has_declared_page_count_mismatch"]
        ),
        "booklets_where_observed_pages_do_not_match_declared": sum(
            1 for row in rows if not row["observed_matches_declared"]
        ),
        "booklets_split_across_multiple_files": sum(
            1 for row in rows if row["source_file_count"] > 1
        ),
    }
    return {
        **totals,
        "booklets": len(rows),
        "total_pages": sum(page_counts),
        "page_count_stats": page_count_stats,
        "quality": quality,
    }


def write_booklet_csv(rows, output_path):
    fieldnames = [
        "pid",
        "page_count",
        "observed_pages",
        "declared_page_count_min",
        "declared_page_count_max",
        "declared_page_count_unique_values",
        "has_declared_page_count_mismatch",
        "observed_matches_declared",
        "min_page_index",
        "max_page_index",
        "source_file_count",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_histogram_csv(rows, output_path):
    histogram = Counter(row["page_count"] for row in rows)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page_count", "booklet_count"])
        writer.writeheader()
        for page_count in sorted(histogram):
            writer.writerow({"page_count": page_count, "booklet_count": histogram[page_count]})


def format_stats_box(summary):
    stats = summary.get("page_count_stats", {})
    percentiles = stats.get("percentiles", {})
    if not stats:
        return "No data"

    return "\n".join(
        [
            f"Booklets: {summary['booklets']:,}",
            f"Total pages: {summary['total_pages']:,}",
            f"Min: {stats['min']:,}",
            f"Max: {stats['max']:,}",
            f"Mean: {stats['mean']:.2f}",
            f"Median: {stats['median']:.2f}",
            f"P90: {percentiles['p90']:.2f}",
            f"P95: {percentiles['p95']:.2f}",
            f"P99: {percentiles['p99']:.2f}",
        ]
    )


def write_histogram_plot(rows, output_path, bins, summary, log_y=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    page_counts = [row["page_count"] for row in rows]
    if not page_counts:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(page_counts, bins=bins, color="#4C78A8", alpha=0.9, edgecolor="white")
    ax.set_title("Booklet Page Count Histogram")
    ax.set_xlabel("Pages per booklet")
    ax.set_ylabel("Booklets")
    if log_y:
        ax.set_yscale("log")
        ax.set_title("Booklet Page Count Histogram (log y)")
    ax.text(
        0.98,
        0.95,
        format_stats_box(summary),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.9,
        },
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_summary(summary, output_dir, output_paths):
    stats = summary.get("page_count_stats", {})
    quality = summary.get("quality", {})
    percentiles = stats.get("percentiles", {})

    print("-" * 60)
    print("Input directories:")
    for input_dir in summary.get("input_dirs", []):
        print(f"  {input_dir}")
    print(f"Input files: {summary['input_files']}")
    print(f"Input lines: {summary['total_lines']}")
    print(f"Invalid JSON lines: {summary['invalid_json_lines']}")
    print(f"Lines without PID: {summary['missing_pid_lines']}")
    print(f"Booklets: {summary['booklets']}")
    print(f"Total pages: {summary['total_pages']}")
    if stats:
        print(f"Page count min: {stats['min']}")
        print(f"Page count max: {stats['max']}")
        print(f"Page count mean: {stats['mean']:.4f}")
        print(f"Page count median: {stats['median']}")
        print("Percentiles:")
        for name in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
            print(f"  {name}: {percentiles[name]:.4f}")
    print("Quality checks:")
    print(
        f"  Booklets without declared page_count: {quality.get('booklets_without_declared_page_count', 0)}"
    )
    print(
        "  Booklets with inconsistent declared page_count: "
        f"{quality.get('booklets_with_declared_page_count_mismatch', 0)}"
    )
    print(
        "  Booklets where observed pages do not match declared: "
        f"{quality.get('booklets_where_observed_pages_do_not_match_declared', 0)}"
    )
    print(
        f"  Booklets split across multiple files: {quality.get('booklets_split_across_multiple_files', 0)}"
    )
    print(f"Output directory: {output_dir}")
    print(f"Summary JSON: {output_paths['summary_json']}")
    print(f"Booklet CSV: {output_paths['booklet_csv']}")
    print(f"Histogram CSV: {output_paths['histogram_csv']}")
    print(f"Histogram PNG: {output_paths['histogram_png']}")
    print(f"Histogram log-y PNG: {output_paths['histogram_logy_png']}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize page-count statistics per publication from page-level JSONL shards."
    )
    parser.add_argument(
        "input_dirs", nargs="+", help="Directories containing .jsonl or .jsonl.gz shard files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for summary JSON and CSV outputs"
    )
    parser.add_argument("--summary-json", help="Output path for summary JSON")
    parser.add_argument("--booklet-csv", help="Output path for per-booklet CSV")
    parser.add_argument("--histogram-csv", help="Output path for histogram CSV")
    parser.add_argument("--histogram-png", help="Output path for histogram PNG")
    parser.add_argument(
        "--histogram-logy-png", help="Output path for histogram PNG with log-scaled y-axis"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of worker processes"
    )
    parser.add_argument("--glob", default="*.jsonl", help="Input file glob relative to input_dir")
    parser.add_argument(
        "--histogram-bins", type=int, default=50, help="Number of bins for histogram PNG output"
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    input_dirs = [Path(input_dir) for input_dir in args.input_dirs]
    output_dir = Path(args.output_dir)
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")
    if args.workers < 1:
        raise SystemExit(f"--workers must be a positive integer. Got: {args.workers}")
    if args.histogram_bins < 1:
        raise SystemExit(f"--histogram-bins must be a positive integer. Got: {args.histogram_bins}")

    files = []
    for input_dir in input_dirs:
        files.extend(sorted(input_dir.glob(args.glob)))
    if not files:
        input_dir_text = ", ".join(str(input_dir) for input_dir in input_dirs)
        raise SystemExit(f"No files matched {args.glob!r} under: {input_dir_text}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary_json": Path(args.summary_json)
        if args.summary_json
        else output_dir / "booklet_page_count_summary.json",
        "booklet_csv": Path(args.booklet_csv)
        if args.booklet_csv
        else output_dir / "booklet_page_counts.csv",
        "histogram_csv": Path(args.histogram_csv)
        if args.histogram_csv
        else output_dir / "booklet_page_count_histogram.csv",
        "histogram_png": Path(args.histogram_png)
        if args.histogram_png
        else output_dir / "booklet_page_count_histogram.png",
        "histogram_logy_png": Path(args.histogram_logy_png)
        if args.histogram_logy_png
        else output_dir / "booklet_page_count_histogram_logy.png",
    }
    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        iterator = executor.map(process_file, files)
        if not args.no_progress:
            iterator = tqdm(iterator, total=len(files), desc="Processing shards")
        results = list(iterator)

    booklet_stats, totals = merge_results(results)
    totals["input_dirs"] = [str(input_dir) for input_dir in input_dirs]
    rows = build_booklet_rows(booklet_stats)
    summary = summarize(rows, totals)

    write_booklet_csv(rows, output_paths["booklet_csv"])
    write_histogram_csv(rows, output_paths["histogram_csv"])
    write_histogram_plot(rows, output_paths["histogram_png"], args.histogram_bins, summary)
    write_histogram_plot(
        rows,
        output_paths["histogram_logy_png"],
        args.histogram_bins,
        summary,
        log_y=True,
    )
    with open(output_paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_summary(summary, output_dir, output_paths)


if __name__ == "__main__":
    main()

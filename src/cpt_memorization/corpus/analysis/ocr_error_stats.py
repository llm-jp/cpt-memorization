#!/usr/bin/env python3
"""Compute basic corpus stats and OCR error marker counts for JSONL files."""

import argparse
import gzip
import io
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, TextIO

try:
    import orjson
except Exception:
    orjson = None

try:
    import zstandard as zstd
except Exception:
    zstd = None


def open_maybe_compressed(path: Path) -> TextIO:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if suffix == ".zst":
        if zstd is None:
            raise RuntimeError("zstandard is required to read .zst files")
        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def dig(obj: Dict, path: str) -> Optional[str]:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur if isinstance(cur, str) else None


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with open_maybe_compressed(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                if orjson:
                    yield orjson.loads(line)
                else:
                    yield json.loads(line)
            except Exception:
                continue


def summarize_stats(stats: Dict[str, float]) -> Dict[str, float]:
    docs = stats.get("docs", 0)
    chars_total = stats.get("chars_total", 0)
    errors_total = stats.get("errors_total", 0)
    errors_docs = stats.get("error_docs", 0)
    stats["chars_min"] = 0 if docs == 0 else stats.get("chars_min", 0)
    stats["chars_avg"] = chars_total / max(1, docs)
    stats["errors_avg_per_doc"] = errors_total / max(1, docs)
    stats["errors_per_million_chars"] = (errors_total / max(1, chars_total)) * 1_000_000
    stats["error_doc_ratio"] = errors_docs / max(1, docs)
    return stats


def analyze_file(path: Path, text_key: str, max_docs: Optional[int]) -> Dict[str, float]:
    marker = "〓"
    stats: Dict[str, float] = {
        "docs": 0,
        "chars_total": 0,
        "chars_min": 10**18,
        "chars_max": 0,
        "errors_total": 0,
        "error_docs": 0,
    }

    for record in iter_jsonl(path):
        if max_docs and stats["docs"] >= max_docs:
            break
        text = dig(record, text_key) if text_key else None
        if text is None and isinstance(record, dict):
            text = record.get("text") if isinstance(record.get("text"), str) else None
        if text is None or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        stats["docs"] += 1
        chars = len(text)
        stats["chars_total"] += chars
        stats["chars_min"] = min(stats["chars_min"], chars)
        stats["chars_max"] = max(stats["chars_max"], chars)

        errors = text.count(marker)
        stats["errors_total"] += errors
        if errors > 0:
            stats["error_docs"] += 1

    return summarize_stats(stats)


def aggregate(all_stats: Iterable[Dict[str, float]]) -> Dict[str, float]:
    total: Dict[str, float] = {
        "docs": 0,
        "chars_total": 0,
        "chars_min": 10**18,
        "chars_max": 0,
        "errors_total": 0,
        "error_docs": 0,
    }
    for stats in all_stats:
        total["docs"] += stats.get("docs", 0)
        total["chars_total"] += stats.get("chars_total", 0)
        total["errors_total"] += stats.get("errors_total", 0)
        total["error_docs"] += stats.get("error_docs", 0)
        total["chars_min"] = min(total["chars_min"], stats.get("chars_min", 10**18))
        total["chars_max"] = max(total["chars_max"], stats.get("chars_max", 0))
    return summarize_stats(total)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(h, "")) for h in headers) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/raw_data",
        help="Directory containing JSONL files (supports .jsonl, .jsonl.gz, .jsonl.zst)",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl*",
        help="Glob pattern within the input directory.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Dot path to the text field inside each JSON object.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit on documents per file (for quick tests).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (per-file parallelism).",
    )
    parser.add_argument(
        "--output-dir",
        default="out/ocr_error_stats",
        help="Where to write summary files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern {args.pattern} under {input_dir}")

    per_file_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_map = {
            ex.submit(analyze_file, path, args.text_key, args.max_docs): path for path in files
        }
        for fut in as_completed(future_map):
            path = future_map[fut]
            stats = fut.result()
            stats["file"] = path.name
            per_file_rows.append(stats)
            print(
                f"{path.name}: docs={stats['docs']:,} chars={stats['chars_total']:,} "
                f"〓={stats['errors_total']:,} (per 1M chars={stats['errors_per_million_chars']:.2f})"
            )

    totals = aggregate(per_file_rows)
    payload = {
        "meta": {
            "input_dir": str(input_dir),
            "pattern": args.pattern,
            "text_key": args.text_key,
        },
        "totals": totals,
        "files": per_file_rows,
    }

    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "files.csv", per_file_rows)

    print(
        "Total: "
        f"docs={totals['docs']:,} chars={totals['chars_total']:,} "
        f"〓={totals['errors_total']:,} (per 1M chars={totals['errors_per_million_chars']:.2f})"
    )
    print(f"Wrote: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

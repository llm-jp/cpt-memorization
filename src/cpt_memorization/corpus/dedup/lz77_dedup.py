#!/usr/bin/env python3
"""LZ77-style (zlib) page deduplication for JSONL records.

This script removes exact duplicate text using compressed fingerprints verified
against raw-text hashes. Compression ratios independently measure repetition;
compression does not make different text into equivalent fingerprints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate JSONL page records using zlib (LZ77-family) fingerprints."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input .jsonl or .jsonl.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where deduplicated outputs and reports are written.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help=(
            "Optional JSONL path for PID-level compression ratios. In booklet mode, "
            "writes one row per pid. In page mode, writes the average page "
            "compression ratio per pid. The ratio is stored as compressed/original "
            "byte length."
        ),
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="JSON key containing page text.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        choices=range(1, 10),
        help="zlib compression level (1-9).",
    )
    parser.add_argument(
        "--repetition-ratio-threshold",
        type=float,
        default=None,
        help=(
            "If set, records with compression_ratio <= threshold are removed as "
            "highly repetitive. compression_ratio is compressed/original byte "
            "length (can be > 1). Smaller values indicate more repetition."
        ),
    )
    parser.add_argument(
        "--min-bytes-for-ratio",
        type=int,
        default=64,
        help="Only apply repetition-ratio-threshold when UTF-8 byte length >= this value.",
    )
    parser.add_argument(
        "--cross-file",
        action="store_true",
        help="Deduplicate globally across all input files instead of per-file.",
    )
    parser.add_argument(
        "--dedup-unit",
        choices=["page", "booklet"],
        default="booklet",
        help="Dedup granularity: per-page or per-booklet(pid with page concatenation).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=200000,
        help="Progress report interval in lines.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for chunk-level parallelism (>=1).",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=50000,
        help="Lines per chunk for parallel preprocessing.",
    )
    parser.add_argument(
        "--sqlite-insert-batch",
        type=int,
        default=5000,
        help="Batch size for SQLite page inserts in booklet mode.",
    )
    return parser.parse_args()


def open_text_reader(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def open_text_writer(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def record_id(doc: dict) -> dict:
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    return {
        "pid": meta.get("pid"),
        "archive_member": meta.get("archive_member"),
        "page_index": meta.get("page_index"),
    }


def zlib_fingerprint(text_bytes: bytes, level: int) -> tuple[str, float]:
    compressed = zlib.compress(text_bytes, level)
    digest = hashlib.blake2b(compressed, digest_size=16).hexdigest()
    ratio = len(compressed) / max(1, len(text_bytes))
    return digest, ratio


def raw_text_hash(text_bytes: bytes) -> str:
    return hashlib.sha1(text_bytes).hexdigest()


def output_paths(output_dir: Path, input_path: Path) -> tuple[Path, Path, Path]:
    base = input_path.name
    if base.endswith(".jsonl.gz"):
        stem = base[: -len(".jsonl.gz")]
    elif base.endswith(".jsonl"):
        stem = base[: -len(".jsonl")]
    else:
        stem = input_path.stem

    dedup = output_dir / f"{stem}.lz77_dedup.jsonl"
    removed = output_dir / f"{stem}.lz77_removed.jsonl"
    stats = output_dir / f"{stem}.lz77_stats.json"
    return dedup, removed, stats


def resolve_workers(requested_workers: int) -> int:
    if requested_workers < 1:
        raise ValueError(f"--workers must be >= 1. got={requested_workers}")
    return min(requested_workers, max(1, os.cpu_count() or 1))


def chunk_path(tmp_dir: Path, idx: int) -> Path:
    return tmp_dir / f"chunk_{idx:06d}.tsv"


def feature_path(tmp_dir: Path, idx: int) -> Path:
    return tmp_dir / f"chunk_{idx:06d}.features.jsonl"


def write_input_chunks(
    input_path: Path, tmp_dir: Path, chunk_lines: int, report_every: int
) -> list[Path]:
    chunk_paths: list[Path] = []
    idx = 0
    count = 0
    total_lines = 0
    handle = None
    try:
        with open_text_reader(input_path) as src:
            for line_no, raw in enumerate(src, start=1):
                total_lines += 1
                if handle is None:
                    path = chunk_path(tmp_dir, idx)
                    chunk_paths.append(path)
                    handle = path.open("w", encoding="utf-8")
                    count = 0
                raw = raw.rstrip("\n")
                handle.write(f"{line_no}\t{raw}\n")
                count += 1
                if report_every > 0 and total_lines % report_every == 0:
                    print(
                        f"chunking_progress: file={input_path.name} lines={total_lines:,} chunks={len(chunk_paths):,}",
                        file=sys.stderr,
                        flush=True,
                    )
                if count >= chunk_lines:
                    handle.close()
                    handle = None
                    idx += 1
    finally:
        if handle is not None:
            handle.close()

    print(
        f"chunking_done: file={input_path.name} lines={total_lines:,} chunks={len(chunk_paths):,}",
        file=sys.stderr,
        flush=True,
    )
    return chunk_paths


def process_chunk(task: dict) -> dict:
    chunk_file = Path(task["chunk_path"])
    features_file = Path(task["feature_path"])
    text_key = task["text_key"]
    compression_level = task["compression_level"]
    repetition_ratio_threshold = task["repetition_ratio_threshold"]
    min_bytes_for_ratio = task["min_bytes_for_ratio"]
    pid = os.getpid()

    total_records = 0
    skipped_no_text = 0

    with (
        chunk_file.open("r", encoding="utf-8") as src,
        features_file.open("w", encoding="utf-8") as out,
    ):
        for row in src:
            row = row.rstrip("\n")
            if not row:
                continue
            try:
                line_no_str, raw = row.split("\t", 1)
                line_no = int(line_no_str)
            except ValueError:
                continue
            if not raw.strip():
                continue
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(doc, dict):
                continue
            total_records += 1

            text = doc.get(text_key)
            if not isinstance(text, str) or not text:
                skipped_no_text += 1
                payload = {
                    "line_no": line_no,
                    "raw": raw,
                    "fp": None,
                    "th": None,
                    "compression_ratio": None,
                    "reason": "no_text",
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                continue

            text_bytes = text.encode("utf-8", errors="ignore")
            fp, ratio = zlib_fingerprint(text_bytes, compression_level)
            reason = None
            if (
                repetition_ratio_threshold is not None
                and len(text_bytes) >= min_bytes_for_ratio
                and ratio <= repetition_ratio_threshold
            ):
                reason = "high_repetition"

            payload = {
                "line_no": line_no,
                "raw": raw,
                "fp": fp,
                "th": raw_text_hash(text_bytes),
                "compression_ratio": ratio,
                "reason": reason,
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return {
        "chunk_path": str(chunk_file),
        "feature_path": str(features_file),
        "pid": pid,
        "total_records": total_records,
        "skipped_no_text": skipped_no_text,
    }


def build_chunk_sqlite(task: dict) -> dict:
    feature_path = Path(task["feature_path"])
    sqlite_path = feature_path.with_suffix(".sqlite3")
    text_key = task["text_key"]
    insert_batch = task["insert_batch"]

    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(
        "CREATE TABLE pages (pid TEXT, page_index INTEGER, line_no INTEGER, raw TEXT, text TEXT);"
    )

    pending_rows = []
    indexed_pages = 0
    insert_sql = "INSERT INTO pages(pid, page_index, line_no, raw, text) VALUES (?, ?, ?, ?, ?)"

    with conn:
        with feature_path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                try:
                    feature = json.loads(line)
                    doc = json.loads(feature["raw"])
                except json.JSONDecodeError:
                    continue
                meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
                pid_key = str(meta.get("pid") or "")
                page_index = meta.get("page_index")
                if not isinstance(page_index, int):
                    page_index = int(feature["line_no"])
                text = doc.get(text_key)
                if not isinstance(text, str) or not text:
                    continue
                pending_rows.append(
                    (pid_key, page_index, int(feature["line_no"]), feature["raw"], text)
                )
                if len(pending_rows) >= insert_batch:
                    conn.executemany(insert_sql, pending_rows)
                    indexed_pages += len(pending_rows)
                    pending_rows.clear()
            if pending_rows:
                conn.executemany(insert_sql, pending_rows)
                indexed_pages += len(pending_rows)
    conn.close()

    return {
        "sqlite_path": str(sqlite_path),
        "indexed_pages": indexed_pages,
    }


def process_file(
    input_path: Path,
    output_dir: Path,
    text_key: str,
    compression_level: int,
    repetition_ratio_threshold: float | None,
    min_bytes_for_ratio: int,
    report_every: int,
    workers: int,
    chunk_lines: int,
    sqlite_insert_batch: int,
    dedup_unit: str,
    global_seen: dict[str, set[str]] | None,
    pid_ratio_writer: TextIO | None,
) -> dict:
    dedup_path, removed_path, stats_path = output_paths(output_dir, input_path)
    seen = global_seen if global_seen is not None else {}
    pid = os.getpid()

    print(f"start: file={input_path.name} pid={pid}", file=sys.stderr, flush=True)

    total = 0
    kept = 0
    removed_duplicate = 0
    removed_repetition = 0
    skipped_no_text = 0
    sum_ratio = 0.0

    with tempfile.TemporaryDirectory(prefix=f"lz77_{input_path.stem}_", dir=str(output_dir)) as tmp:
        tmp_dir = Path(tmp)
        chunk_files = write_input_chunks(input_path, tmp_dir, chunk_lines, report_every)
        print(
            f"chunking: file={input_path.name} chunks={len(chunk_files)} chunk_lines={chunk_lines} workers={workers}",
            file=sys.stderr,
            flush=True,
        )

        tasks = [
            {
                "chunk_path": str(chunk),
                "feature_path": str(feature_path(tmp_dir, idx)),
                "text_key": text_key,
                "compression_level": compression_level,
                "repetition_ratio_threshold": repetition_ratio_threshold,
                "min_bytes_for_ratio": min_bytes_for_ratio,
            }
            for idx, chunk in enumerate(chunk_files)
        ]

        feature_files: list[Path] = []
        if workers == 1 or len(tasks) <= 1:
            for task in tasks:
                result = process_chunk(task)
                total += result["total_records"]
                skipped_no_text += result["skipped_no_text"]
                feature_files.append(Path(result["feature_path"]))
                print(
                    f"chunk_done: file={input_path.name} pid={result['pid']} chunk={Path(result['chunk_path']).name} "
                    f"records={result['total_records']:,}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_chunk, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    total += result["total_records"]
                    skipped_no_text += result["skipped_no_text"]
                    feature_files.append(Path(result["feature_path"]))
                    print(
                        f"chunk_done: file={input_path.name} pid={result['pid']} chunk={Path(result['chunk_path']).name} "
                        f"records={result['total_records']:,}",
                        file=sys.stderr,
                        flush=True,
                    )

        with open_text_writer(dedup_path) as dedup_f, open_text_writer(removed_path) as removed_f:
            if dedup_unit == "page":
                pid_ratio_sums: dict[str, float] = {}
                pid_ratio_counts: dict[str, int] = {}
                for features in sorted(feature_files):
                    with features.open("r", encoding="utf-8") as src:
                        for line in src:
                            feature = json.loads(line)
                            raw = feature["raw"]
                            line_no = feature["line_no"]
                            doc = json.loads(raw)
                            meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
                            pid_key = meta.get("pid")
                            if feature["reason"] == "no_text":
                                if pid_ratio_writer is not None and pid_key is not None:
                                    pid_ratio_writer.write(
                                        json.dumps(
                                            {
                                                "pid": str(pid_key),
                                                "compression_ratio": None,
                                                "reason": "no_text",
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
                                continue

                            ratio = float(feature["compression_ratio"])
                            sum_ratio += ratio
                            if pid_key is not None:
                                pid_key = str(pid_key)
                                pid_ratio_sums[pid_key] = pid_ratio_sums.get(pid_key, 0.0) + ratio
                                pid_ratio_counts[pid_key] = pid_ratio_counts.get(pid_key, 0) + 1

                            if feature["reason"] == "high_repetition":
                                removed_repetition += 1
                                removed_payload = {
                                    "reason": "high_repetition",
                                    "compression_ratio": round(ratio, 6),
                                    "line_no": line_no,
                                    "id": record_id(doc),
                                    "record": doc,
                                }
                                removed_f.write(
                                    json.dumps(removed_payload, ensure_ascii=False) + "\n"
                                )
                            else:
                                bucket = seen.setdefault(feature["fp"], set())
                                if feature["th"] in bucket:
                                    removed_duplicate += 1
                                    removed_payload = {
                                        "reason": "duplicate",
                                        "compression_ratio": round(ratio, 6),
                                        "line_no": line_no,
                                        "id": record_id(doc),
                                        "record": doc,
                                    }
                                    removed_f.write(
                                        json.dumps(removed_payload, ensure_ascii=False) + "\n"
                                    )
                                else:
                                    bucket.add(feature["th"])
                                    kept += 1
                                    dedup_f.write(raw + "\n")

                            processed = (
                                kept + removed_duplicate + removed_repetition + skipped_no_text
                            )
                            if report_every > 0 and processed % report_every == 0:
                                print(
                                    f"progress: file={input_path.name} pid={pid} processed={processed:,} kept={kept:,} "
                                    f"dup_removed={removed_duplicate:,} repeat_removed={removed_repetition:,}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                if pid_ratio_writer is not None:
                    for pid_key in sorted(pid_ratio_sums):
                        pid_ratio_writer.write(
                            json.dumps(
                                {
                                    "pid": pid_key,
                                    "compression_ratio": round(
                                        pid_ratio_sums[pid_key] / max(1, pid_ratio_counts[pid_key]),
                                        6,
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            else:
                sqlite_path = tmp_dir / "booklet_pages.sqlite3"
                conn = sqlite3.connect(str(sqlite_path))
                conn.execute("PRAGMA journal_mode=OFF;")
                conn.execute("PRAGMA synchronous=OFF;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                conn.execute("PRAGMA cache_size=-200000;")
                conn.execute("PRAGMA mmap_size=1073741824;")
                conn.execute(
                    "CREATE TABLE pages (pid TEXT, page_index INTEGER, line_no INTEGER, raw TEXT, text TEXT);"
                )

                insert_batch = max(1, sqlite_insert_batch)
                if pid_ratio_writer is not None:
                    for features in sorted(feature_files):
                        with features.open("r", encoding="utf-8") as src:
                            for line in src:
                                feature = json.loads(line)
                                if feature["reason"] != "no_text":
                                    continue
                                doc = json.loads(feature["raw"])
                                meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
                                pid_key = meta.get("pid")
                                if pid_key:
                                    pid_ratio_writer.write(
                                        json.dumps(
                                            {
                                                "pid": str(pid_key),
                                                "compression_ratio": None,
                                                "reason": "no_text",
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )

                sqlite_tasks = [
                    {
                        "feature_path": str(f),
                        "text_key": text_key,
                        "insert_batch": insert_batch,
                    }
                    for f in sorted(feature_files)
                ]

                indexed_pages = 0
                next_index_report_pages = report_every if report_every > 0 else None

                def process_sqlite_res(res: dict) -> None:
                    nonlocal indexed_pages, next_index_report_pages
                    chunk_db = res["sqlite_path"]
                    conn.execute(f"ATTACH DATABASE '{chunk_db}' AS chunk_db")
                    conn.execute("INSERT INTO pages SELECT * FROM chunk_db.pages")
                    conn.commit()
                    conn.execute("DETACH DATABASE chunk_db")
                    Path(chunk_db).unlink(missing_ok=True)

                    indexed_pages += res["indexed_pages"]
                    if (
                        next_index_report_pages is not None
                        and indexed_pages >= next_index_report_pages
                    ):
                        print(
                            f"booklet_indexing_progress: file={input_path.name} pages={indexed_pages:,}",
                            file=sys.stderr,
                            flush=True,
                        )
                        next_index_report_pages = (
                            (indexed_pages // report_every) + 1
                        ) * report_every

                if workers == 1 or len(sqlite_tasks) <= 1:
                    for task in sqlite_tasks:
                        process_sqlite_res(build_chunk_sqlite(task))
                else:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                        futures = [
                            executor.submit(build_chunk_sqlite, task) for task in sqlite_tasks
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            process_sqlite_res(future.result())

                index_t0 = time.monotonic()
                print(
                    f"booklet_index_build_start: file={input_path.name}",
                    file=sys.stderr,
                    flush=True,
                )
                conn.execute("CREATE INDEX idx_pages_pid_order ON pages(pid, page_index, line_no)")
                print(
                    f"booklet_index_build_done: file={input_path.name} elapsed_sec={time.monotonic() - index_t0:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"booklet_indexing_done: file={input_path.name} indexed_pages={indexed_pages:,}",
                    file=sys.stderr,
                    flush=True,
                )

                next_booklet_report_pages = report_every if report_every > 0 else None

                def flush_booklet(
                    pid_key: str, pages: list[dict], booklet_no: int, processed_pages: int
                ) -> int:
                    nonlocal kept, removed_duplicate, removed_repetition, sum_ratio
                    nonlocal next_booklet_report_pages
                    if not pages:
                        return processed_pages

                    booklet_text = "".join(p["text"] for p in pages)
                    booklet_bytes = booklet_text.encode("utf-8", errors="ignore")
                    fp, ratio = zlib_fingerprint(booklet_bytes, compression_level)
                    th = raw_text_hash(booklet_bytes)
                    sum_ratio += ratio * len(pages)
                    if pid_ratio_writer is not None and pid_key:
                        pid_ratio_writer.write(
                            json.dumps(
                                {
                                    "pid": pid_key,
                                    "compression_ratio": round(ratio, 6),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    reason = None
                    if (
                        repetition_ratio_threshold is not None
                        and len(booklet_bytes) >= min_bytes_for_ratio
                        and ratio <= repetition_ratio_threshold
                    ):
                        reason = "high_repetition_booklet"
                    else:
                        bucket = seen.setdefault(fp, set())
                        if th in bucket:
                            reason = "duplicate_booklet"
                        else:
                            bucket.add(th)

                    if reason is None:
                        for p in pages:
                            dedup_f.write(p["raw"] + "\n")
                            kept += 1
                    else:
                        for p in pages:
                            if reason == "duplicate_booklet":
                                removed_duplicate += 1
                            else:
                                removed_repetition += 1
                            doc = json.loads(p["raw"])
                            removed_payload = {
                                "reason": reason,
                                "compression_ratio": round(ratio, 6),
                                "line_no": p["line_no"],
                                "booklet_pid": pid_key,
                                "id": record_id(doc),
                                "record": doc,
                            }
                            removed_f.write(json.dumps(removed_payload, ensure_ascii=False) + "\n")

                    processed_pages += len(pages)
                    if next_booklet_report_pages is not None:
                        if processed_pages >= next_booklet_report_pages:
                            print(
                                f"progress_booklet: file={input_path.name} pid={pid} booklets={booklet_no:,} "
                                f"pages={processed_pages:,} kept={kept:,} dup_removed={removed_duplicate:,} "
                                f"repeat_removed={removed_repetition:,}",
                                file=sys.stderr,
                                flush=True,
                            )
                            next_booklet_report_pages = (
                                (processed_pages // report_every) + 1
                            ) * report_every
                    return processed_pages

                processed_pages = 0
                booklet_no = 0
                current_pid = None
                current_pages: list[dict] = []
                scanned_rows = 0
                cursor = conn.execute(
                    "SELECT pid, page_index, line_no, raw, text FROM pages ORDER BY pid, page_index, line_no"
                )
                for row_pid, row_page_index, row_line_no, row_raw, row_text in cursor:
                    scanned_rows += 1
                    if report_every > 0 and scanned_rows % report_every == 0:
                        print(
                            f"booklet_scan_progress: file={input_path.name} scanned_rows={scanned_rows:,} "
                            f"current_booklets={booklet_no:,}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if current_pid is None:
                        current_pid = row_pid
                    if row_pid != current_pid:
                        booklet_no += 1
                        processed_pages = flush_booklet(
                            current_pid, current_pages, booklet_no, processed_pages
                        )
                        current_pages = []
                        current_pid = row_pid

                    current_pages.append(
                        {
                            "line_no": int(row_line_no),
                            "page_index": int(row_page_index),
                            "raw": row_raw,
                            "text": row_text,
                        }
                    )

                if current_pid is not None:
                    booklet_no += 1
                    processed_pages = flush_booklet(
                        current_pid, current_pages, booklet_no, processed_pages
                    )

                conn.close()

    avg_ratio = sum_ratio / max(1, kept + removed_duplicate + removed_repetition)
    stats = {
        "input_file": str(input_path),
        "output_dedup_file": str(dedup_path),
        "output_removed_file": str(removed_path),
        "total_records": total,
        "kept_records": kept,
        "removed_duplicate": removed_duplicate,
        "removed_high_repetition": removed_repetition,
        "skipped_no_text": skipped_no_text,
        "kept_rate": kept / max(1, total),
        "avg_compression_ratio": avg_ratio,
        "compression_level": compression_level,
        "repetition_ratio_threshold": repetition_ratio_threshold,
        "min_bytes_for_ratio": min_bytes_for_ratio,
        "cross_file_mode": global_seen is not None,
        "dedup_unit": dedup_unit,
        "workers": workers,
        "chunk_lines": chunk_lines,
        "sqlite_insert_batch": sqlite_insert_batch,
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"finish: file={input_path.name} pid={pid} total={total:,} kept={kept:,} "
        f"dup_removed={removed_duplicate:,} repeat_removed={removed_repetition:,} dedup_unit={dedup_unit}",
        file=sys.stderr,
        flush=True,
    )

    return stats


def main() -> int:
    args = parse_args()

    input_paths = [Path(p) for p in args.input_files]
    for p in input_paths:
        if not p.exists():
            print(f"Input not found: {p}", file=sys.stderr)
            return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    workers = resolve_workers(args.workers)
    global_seen = {} if args.cross_file else None

    print(
        f"run_mode: files={len(input_paths)} requested_workers={args.workers} effective_workers={workers} "
        f"cross_file={args.cross_file} chunk_lines={args.chunk_lines} dedup_unit={args.dedup_unit}",
        file=sys.stderr,
        flush=True,
    )

    all_stats = []
    with (
        output_jsonl_path.open("w", encoding="utf-8")
        if output_jsonl_path is not None
        else open(os.devnull, "w", encoding="utf-8")
    ) as pid_ratio_writer:
        active_pid_ratio_writer = pid_ratio_writer if output_jsonl_path is not None else None
        for p in input_paths:
            stats = process_file(
                input_path=p,
                output_dir=output_dir,
                text_key=args.text_key,
                compression_level=args.compression_level,
                repetition_ratio_threshold=args.repetition_ratio_threshold,
                min_bytes_for_ratio=args.min_bytes_for_ratio,
                report_every=args.report_every,
                workers=workers,
                chunk_lines=args.chunk_lines,
                sqlite_insert_batch=args.sqlite_insert_batch,
                dedup_unit=args.dedup_unit,
                global_seen=global_seen,
                pid_ratio_writer=active_pid_ratio_writer,
            )
            all_stats.append(stats)
            print(
                f"done: {p.name} total={stats['total_records']:,} kept={stats['kept_records']:,} "
                f"dup_removed={stats['removed_duplicate']:,} repeat_removed={stats['removed_high_repetition']:,}",
                file=sys.stderr,
                flush=True,
            )

    summary_path = output_dir / "lz77_dedup_summary.json"
    summary = {
        "files": all_stats,
        "cross_file_mode": args.cross_file,
        "compression_level": args.compression_level,
        "repetition_ratio_threshold": args.repetition_ratio_threshold,
        "min_bytes_for_ratio": args.min_bytes_for_ratio,
        "dedup_unit": args.dedup_unit,
        "workers": workers,
        "chunk_lines": args.chunk_lines,
        "sqlite_insert_batch": args.sqlite_insert_batch,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"summary: {summary_path}", file=sys.stderr, flush=True)
    if output_jsonl_path is not None:
        print(f"pid_compression_ratios: {output_jsonl_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, TextIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


KODAK_RE = re.compile(r"(k?odak|kodek)", re.IGNORECASE)
GRAY_RE = re.compile(r"(gray\s*scale|grayscale)", re.IGNORECASE)
COLOR_RE = re.compile(r"(color|colour|control|patch)", re.IGNORECASE)
COLORNAME_RE = re.compile(r"(cyan|magenta|yellow|blue|green|red|white|black)", re.IGNORECASE)
KP_RE = re.compile(r"\bkp\s*\d{2,}", re.IGNORECASE)
STRICT_GRAY_RE = re.compile(r"^\W*(gray\s*scale|grayscale)\W*$", re.IGNORECASE)
STRICT_COLOR_RE = re.compile(r"^\W*color\s*control\s*patches?\W*$", re.IGNORECASE)


def open_text(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def is_noise_text(text: str) -> bool:
    if not text.strip():
        return False

    has_kodak = bool(KODAK_RE.search(text))
    has_kp = bool(KP_RE.search(text))
    has_gray = bool(GRAY_RE.search(text))
    has_color = bool(COLOR_RE.search(text))
    has_colorname = bool(COLORNAME_RE.search(text))

    if has_kodak:
        if "et al" in text.lower():
            has_kodak = False
        else:
            has_tm = bool(re.search(r"(tm|®|©|copyright)", text, re.IGNORECASE))
            has_noise_context = has_gray or has_color or has_colorname or has_tm
            clean_text = re.sub(r"[\W_]+", "", text).lower()
            is_just_kodak = clean_text in ("kodak", "kodek")
            if not (is_just_kodak or has_noise_context):
                has_kodak = False

    strict_gray = bool(STRICT_GRAY_RE.search(text))
    strict_color = bool(STRICT_COLOR_RE.search(text))
    gray_color_noise = has_gray and (has_color or has_colorname)

    words = text.split()
    is_long_sentence = len(words) > 8
    if is_long_sentence and gray_color_noise:
        gray_color_noise = False

    return has_kodak or has_kp or gray_color_noise or strict_gray or strict_color


@dataclass
class CorpusStats:
    input_path: str
    edge_pages: int
    total_pages: int = 0
    noisy_pages: int = 0
    front_pages: int = 0
    back_pages: int = 0
    front_noisy_pages: int = 0
    back_noisy_pages: int = 0
    total_docs: int = 0
    front_noisy_docs: int = 0
    back_noisy_docs: int = 0
    both_noisy_docs: int = 0
    only_front_noisy_docs: int = 0
    only_back_noisy_docs: int = 0
    any_edge_noisy_docs: int = 0
    front_noisy_page_ratio: float = 0.0
    back_noisy_page_ratio: float = 0.0
    front_noisy_doc_ratio: float = 0.0
    back_noisy_doc_ratio: float = 0.0
    front_share_among_single_side_noisy_docs: float = 0.0
    back_share_among_single_side_noisy_docs: float = 0.0


def apply_record(
    stats: CorpusStats,
    doc_flags: Dict[str, int],
    pid: str,
    page_index: int,
    page_count: int,
    text: str,
    edge_pages: int,
) -> None:
    is_noisy = is_noise_text(text)
    stats.total_pages += 1
    if is_noisy:
        stats.noisy_pages += 1

    is_front = page_index < edge_pages
    is_back = page_index >= max(0, page_count - edge_pages)

    if is_front:
        stats.front_pages += 1
        if is_noisy:
            stats.front_noisy_pages += 1
            doc_flags[pid] = doc_flags.get(pid, 0) | 1
        elif pid not in doc_flags:
            doc_flags[pid] = 0

    if is_back:
        stats.back_pages += 1
        if is_noisy:
            stats.back_noisy_pages += 1
            doc_flags[pid] = doc_flags.get(pid, 0) | 2
        elif pid not in doc_flags:
            doc_flags[pid] = 0

    if not is_front and not is_back and pid not in doc_flags:
        doc_flags[pid] = 0


def finalize_stats(stats: CorpusStats, doc_flags: Dict[str, int]) -> None:
    stats.total_docs = len(doc_flags)
    for mask in doc_flags.values():
        has_front = (mask & 1) != 0
        has_back = (mask & 2) != 0
        if has_front:
            stats.front_noisy_docs += 1
        if has_back:
            stats.back_noisy_docs += 1
        if has_front and has_back:
            stats.both_noisy_docs += 1
        elif has_front:
            stats.only_front_noisy_docs += 1
        elif has_back:
            stats.only_back_noisy_docs += 1

    stats.any_edge_noisy_docs = (
        stats.only_front_noisy_docs + stats.only_back_noisy_docs + stats.both_noisy_docs
    )

    if stats.front_pages > 0:
        stats.front_noisy_page_ratio = stats.front_noisy_pages / stats.front_pages
    if stats.back_pages > 0:
        stats.back_noisy_page_ratio = stats.back_noisy_pages / stats.back_pages
    if stats.total_docs > 0:
        stats.front_noisy_doc_ratio = stats.front_noisy_docs / stats.total_docs
        stats.back_noisy_doc_ratio = stats.back_noisy_docs / stats.total_docs

    single_side = stats.only_front_noisy_docs + stats.only_back_noisy_docs
    if single_side > 0:
        stats.front_share_among_single_side_noisy_docs = stats.only_front_noisy_docs / single_side
        stats.back_share_among_single_side_noisy_docs = stats.only_back_noisy_docs / single_side


def merge_partial(
    stats: CorpusStats,
    doc_flags: Dict[str, int],
    counts: Dict[str, int],
    partial_doc_flags: Dict[str, int],
) -> None:
    stats.total_pages += counts["total_pages"]
    stats.noisy_pages += counts["noisy_pages"]
    stats.front_pages += counts["front_pages"]
    stats.back_pages += counts["back_pages"]
    stats.front_noisy_pages += counts["front_noisy_pages"]
    stats.back_noisy_pages += counts["back_noisy_pages"]
    for pid, mask in partial_doc_flags.items():
        doc_flags[pid] = doc_flags.get(pid, 0) | mask


def analyze_chunk(lines: list[str], edge_pages: int) -> tuple[Dict[str, int], Dict[str, int]]:
    partial_stats = CorpusStats(input_path="", edge_pages=edge_pages)
    partial_doc_flags: Dict[str, int] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        meta = obj.get("meta", {})
        if not isinstance(meta, dict):
            continue

        pid = str(meta.get("pid", ""))
        page_index = meta.get("page_index")
        page_count = meta.get("page_count")
        if (
            not pid
            or not isinstance(page_index, int)
            or not isinstance(page_count, int)
            or page_count <= 0
        ):
            continue

        text = obj.get("text", "")
        if not isinstance(text, str):
            text = ""
        apply_record(
            partial_stats, partial_doc_flags, pid, page_index, page_count, text, edge_pages
        )

    counts = {
        "total_pages": partial_stats.total_pages,
        "noisy_pages": partial_stats.noisy_pages,
        "front_pages": partial_stats.front_pages,
        "back_pages": partial_stats.back_pages,
        "front_noisy_pages": partial_stats.front_noisy_pages,
        "back_noisy_pages": partial_stats.back_noisy_pages,
    }
    return counts, partial_doc_flags


def iter_jsonl(path: Path, show_progress: bool = True) -> Iterable[dict]:
    with open_text(path) as f:
        iterator = tqdm(
            f,
            desc=f"reading {path.name}",
            unit="lines",
            disable=(not show_progress),
        )
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def analyze_corpus(
    path: Path,
    edge_pages: int,
    max_records: int | None,
    workers: int = 1,
    chunk_size: int = 20000,
    show_progress: bool = True,
) -> CorpusStats:
    stats = CorpusStats(input_path=str(path), edge_pages=edge_pages)
    doc_flags: Dict[str, int] = {}

    if workers <= 1:
        for i, obj in enumerate(iter_jsonl(path, show_progress=show_progress)):
            if max_records is not None and i >= max_records:
                break

            meta = obj.get("meta", {})
            if not isinstance(meta, dict):
                continue

            pid = str(meta.get("pid", ""))
            page_index = meta.get("page_index")
            page_count = meta.get("page_count")
            if (
                not pid
                or not isinstance(page_index, int)
                or not isinstance(page_count, int)
                or page_count <= 0
            ):
                continue

            text = obj.get("text", "")
            if not isinstance(text, str):
                text = ""
            apply_record(stats, doc_flags, pid, page_index, page_count, text, edge_pages)
    else:
        with open_text(path) as f, ProcessPoolExecutor(max_workers=workers) as ex:
            iterator = tqdm(
                f,
                desc=f"reading {path.name}",
                unit="lines",
                disable=(not show_progress),
            )
            futures = []
            chunk: list[str] = []
            seen = 0
            for line in iterator:
                if max_records is not None and seen >= max_records:
                    break
                chunk.append(line)
                seen += 1
                if len(chunk) >= chunk_size:
                    futures.append(ex.submit(analyze_chunk, chunk, edge_pages))
                    chunk = []
            if chunk:
                futures.append(ex.submit(analyze_chunk, chunk, edge_pages))

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"aggregating {path.name}",
                disable=(not show_progress),
            ):
                counts, partial_doc_flags = fut.result()
                merge_partial(stats, doc_flags, counts, partial_doc_flags)

    finalize_stats(stats, doc_flags)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="先頭/末尾ページにどれくらいノイズが入るかを集計する。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="入力 JSONL(.gz)。複数指定可。",
    )
    parser.add_argument(
        "--edge-pages",
        type=int,
        default=1,
        help="先頭/末尾として扱うページ数 (default: 1)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="デバッグ用: 先頭Nレコードだけ集計",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/edge_noise_ratio_report.json"),
        help="集計JSONの出力先",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=(os.cpu_count() or 1),
        help="並列ワーカー数 (default: 利用可能CPU最大)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="進捗表示を無効化する",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="並列時のチャンク行数 (default: 20000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.edge_pages <= 0:
        raise SystemExit("--edge-pages must be >= 1")
    if args.workers <= 0:
        raise SystemExit("--workers must be >= 1")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be >= 1")

    input_paths = args.input

    valid_paths = []
    for path in input_paths:
        if not path.exists():
            print(f"[skip] missing: {path}")
            continue
        valid_paths.append(path)

    all_stats = []
    for path in valid_paths:
        all_stats.append(
            analyze_corpus(
                path,
                args.edge_pages,
                args.max_records,
                workers=args.workers,
                chunk_size=args.chunk_size,
                show_progress=(not args.no_progress),
            )
        )

    if not all_stats:
        raise SystemExit("No valid input files.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {"corpora": [asdict(s) for s in all_stats]}
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for s in all_stats:
        print("=" * 72)
        print(f"input: {s.input_path}")
        print(f"edge_pages: {s.edge_pages}")
        print(f"pages: total={s.total_pages:,}, noisy={s.noisy_pages:,}")
        print(
            f"front noisy pages: {s.front_noisy_pages:,}/{s.front_pages:,} "
            f"({s.front_noisy_page_ratio:.2%})"
        )
        print(
            f"back  noisy pages: {s.back_noisy_pages:,}/{s.back_pages:,} "
            f"({s.back_noisy_page_ratio:.2%})"
        )
        print(
            f"docs with noisy front: {s.front_noisy_docs:,}/{s.total_docs:,} "
            f"({s.front_noisy_doc_ratio:.2%})"
        )
        print(
            f"docs with noisy back : {s.back_noisy_docs:,}/{s.total_docs:,} "
            f"({s.back_noisy_doc_ratio:.2%})"
        )
        print(
            f"docs split: front_only={s.only_front_noisy_docs:,}, "
            f"back_only={s.only_back_noisy_docs:,}, both={s.both_noisy_docs:,}"
        )
        if s.only_front_noisy_docs + s.only_back_noisy_docs > 0:
            print(
                f"single-side noisy docs share: front={s.front_share_among_single_side_noisy_docs:.2%}, "
                f"back={s.back_share_among_single_side_noisy_docs:.2%}"
            )

    print("=" * 72)
    print(f"report saved: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

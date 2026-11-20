#!/usr/bin/env python3
import argparse
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Regex patterns for noise detection
KODAK_RE = re.compile(r"(k?odak|kodek)", re.IGNORECASE)
GRAY_RE = re.compile(r"(gray\s*scale|grayscale)", re.IGNORECASE)
COLOR_RE = re.compile(r"(color|colour|control|patch)", re.IGNORECASE)
COLORNAME_RE = re.compile(r"(cyan|magenta|yellow|blue|green|red|white|black)", re.IGNORECASE)
KP_RE = re.compile(r"\bkp\s*\d{2,}", re.IGNORECASE)
STRICT_GRAY_RE = re.compile(r"^\W*(gray\s*scale|grayscale)\W*$", re.IGNORECASE)
STRICT_COLOR_RE = re.compile(r"^\W*color\s*control\s*patches?\W*$", re.IGNORECASE)


@dataclass
class DenoiseConfig:
    pass


def parse_args():
    ap = argparse.ArgumentParser(
        description="Kodak/GrayScale ノイズの統計を取り、最初と最後のページを単純削除する"
    )
    ap.add_argument("--in", dest="in_path", type=Path, help="入力 JSONL(.gz)")
    ap.add_argument("--in-dir", type=Path, help="入力ディレクトリ（*.jsonl.gz）")
    ap.add_argument("--out", type=Path, help="出力 JSONL(.gz)")
    ap.add_argument("--out-dir", type=Path, help="出力ディレクトリ（--in-dir 使用時）")
    ap.add_argument("--workers", type=int, default=8, help="並列数")
    ap.add_argument("--report", type=Path, help="統計レポート(JSON)")
    return ap.parse_args()


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def open_write(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def iter_input_paths(in_dir: Path) -> Iterable[Path]:
    paths = sorted(in_dir.glob("*.jsonl.gz")) + sorted(in_dir.glob("*.jsonl"))
    for p in paths:
        yield p


def is_noise_line(line: str) -> bool:
    if not line.strip():
        return False

    has_kodak = bool(KODAK_RE.search(line))
    has_kp = bool(KP_RE.search(line))
    has_gray = bool(GRAY_RE.search(line))
    has_color = bool(COLOR_RE.search(line))
    has_colorname = bool(COLORNAME_RE.search(line))

    # Pattern Match Logic
    if has_kodak:
        if "et al" in line.lower():
            has_kodak = False
        else:
            has_tm = bool(re.search(r"(tm|®|©|copyright)", line, re.IGNORECASE))
            has_noise_context = has_gray or has_color or has_colorname or has_tm
            clean_line = re.sub(r"[\W_]+", "", line).lower()
            is_just_kodak = clean_line == "kodak" or clean_line == "kodek"
            if not (is_just_kodak or has_noise_context):
                has_kodak = False

    strict_gray = bool(STRICT_GRAY_RE.search(line))
    strict_color = bool(STRICT_COLOR_RE.search(line))
    gray_color_noise = has_gray and (has_color or has_colorname)

    words = line.split()
    is_long_sentence = len(words) > 8
    if is_long_sentence and gray_color_noise:
        gray_color_noise = False

    return has_kodak or has_kp or gray_color_noise or strict_gray or strict_color


def count_noise(text: str) -> int:
    count = 0
    if not text:
        return 0
    for line in text.splitlines():
        if is_noise_line(line):
            count += 1
    return count


def _pick_removal_target(
    first_candidate: Optional[Tuple[int, int]], last_candidate: Optional[Tuple[int, int]]
) -> Optional[int]:
    """
    Decide which single edge page to remove.
    Candidate tuple is (valid_obj_index, noise_count).
    """
    candidates: List[Tuple[int, int]] = []
    if first_candidate is not None:
        candidates.append(first_candidate)
    if last_candidate is not None:
        candidates.append(last_candidate)

    noisy_candidates = [c for c in candidates if c[1] > 0]
    if not noisy_candidates:
        return None

    # Remove only one page: choose the noisier edge page (stable by index on tie).
    noisy_candidates.sort(key=lambda x: (-x[1], x[0]))
    return noisy_candidates[0][0]


def process_file(in_path: Path, out_path: Path) -> Dict[str, int]:
    if in_path.resolve() == out_path.resolve():
        raise ValueError("Input and output must be different files")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total_noise_before": 0, "total_noise_after": 0, "removed_pages": 0, "kept_pages": 0}

    # Pass 1: count noise and identify first/last edge page candidates.
    valid_count = 0
    first_candidate: Optional[Tuple[int, int]] = None
    last_candidate: Optional[Tuple[int, int]] = None
    edge_candidates_by_doc: Dict[str, Dict[str, Optional[Tuple[int, int]]]] = {}

    with open_text(in_path) as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            text = obj.get("text", "")
            noise_before = count_noise(text)
            stats["total_noise_before"] += noise_before

            meta = obj.get("meta", {})
            page_index = meta.get("page_index")
            page_count = meta.get("page_count")
            pid = meta.get("pid")
            archive_member = meta.get("archive_member")

            doc_key: Optional[str] = None
            if pid is not None:
                doc_key = f"pid:{pid}"
            elif archive_member is not None:
                doc_key = f"archive:{archive_member}"

            if doc_key is not None:
                cand = edge_candidates_by_doc.setdefault(doc_key, {"first": None, "last": None})
                if page_index == 0:
                    cand["first"] = (valid_count, noise_before)
                if (
                    page_index is not None
                    and page_count is not None
                    and page_index == page_count - 1
                ):
                    cand["last"] = (valid_count, noise_before)
            else:
                # Fallback path for inputs where document identity is unavailable.
                if page_index == 0:
                    first_candidate = (valid_count, noise_before)
                if (
                    page_index is not None
                    and page_count is not None
                    and page_index == page_count - 1
                ):
                    last_candidate = (valid_count, noise_before)

            valid_count += 1

    remove_valid_indices = set()
    for cand in edge_candidates_by_doc.values():
        remove_valid_idx = _pick_removal_target(cand.get("first"), cand.get("last"))
        if remove_valid_idx is not None:
            remove_valid_indices.add(remove_valid_idx)

    # Fallback for non-groupable records.
    remove_valid_idx = _pick_removal_target(first_candidate, last_candidate)
    if remove_valid_idx is not None:
        remove_valid_indices.add(remove_valid_idx)

    # Pass 2: write output while removing selected edge pages.
    valid_idx = 0
    with open_text(in_path) as f_in, open_write(out_path) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            text = obj.get("text", "")
            noise_count = count_noise(text)

            if valid_idx in remove_valid_indices:
                stats["removed_pages"] += 1
            else:
                stats["kept_pages"] += 1
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["total_noise_after"] += noise_count

            valid_idx += 1

    return stats


def merge_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def print_summary(stats: Dict[str, int]) -> None:
    print("-" * 60)
    print("  [Simple Noise Removal Summary]  ")
    print("-" * 60)

    noise_before = stats.get("total_noise_before", 0)
    noise_after = stats.get("total_noise_after", 0)
    removed_pages = stats.get("removed_pages", 0)
    kept_pages = stats.get("kept_pages", 0)

    reduced = noise_before - noise_after
    reduction_rate = 0.0
    if noise_before > 0:
        reduction_rate = reduced / noise_before

    print(f"1. 除去処理前のノイズ数 (行単位): {noise_before}")
    print(f"2. 除去処理後のノイズ数 (行単位): {noise_after}")
    print(f"")
    print(f"   - 削除されたページ数: {removed_pages}")
    print(f"   - 残ったページ数:     {kept_pages}")
    print(f"   - ノイズ削減数:       {reduced}")
    print(f"   - ノイズ削減率:       {reduction_rate:.2%}")
    print("-" * 60)


def main() -> int:
    args = parse_args()

    if args.in_dir and args.in_dir.is_file():
        args.in_path = args.in_dir
        args.in_dir = None

    if args.in_dir:
        in_dir = args.in_dir
        out_dir = args.out_dir or (Path("out_denoise_simple") / in_dir.name)
        out_dir.mkdir(parents=True, exist_ok=True)

        stats_total: Dict[str, int] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_to_path = {}
            for in_path in iter_input_paths(in_dir):
                out_path = out_dir / f"{in_path.stem}.jsonl"
                fut = ex.submit(process_file, in_path, out_path)
                future_to_path[fut] = out_path

            for fut in tqdm(
                as_completed(future_to_path), total=len(future_to_path), desc="Processing files"
            ):
                stats = fut.result()
                merge_stats(stats_total, stats)

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(stats_total, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        print_summary(stats_total)
        return 0

    if not args.in_path:
        raise SystemExit("--in or --in-dir is required")

    out_path = args.out
    if not out_path:
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / args.in_path.name
        else:
            out_path = Path("denoised_simple.jsonl")

    stats = process_file(args.in_path, out_path)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(stats, ensure_ascii=False) + "\n", encoding="utf-8")

    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

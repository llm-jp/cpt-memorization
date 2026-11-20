#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import TextIO

try:
    import orjson  # type: ignore
except Exception:
    orjson = None


KODAK_RE = re.compile(r"(k?odak|kodek)", re.IGNORECASE)
GRAY_RE = re.compile(r"(gray\s*scale|grayscale)", re.IGNORECASE)
COLOR_RE = re.compile(r"(color|colour|control|patch)", re.IGNORECASE)
COLORNAME_RE = re.compile(r"(cyan|magenta|yellow|blue|green|red|white|black)", re.IGNORECASE)
KP_RE = re.compile(r"\bkp\s*\d{2,}", re.IGNORECASE)
STRICT_GRAY_RE = re.compile(r"^\W*(gray\s*scale|grayscale)\W*$", re.IGNORECASE)
STRICT_COLOR_RE = re.compile(r"^\W*color\s*control\s*patches?\W*$", re.IGNORECASE)


def parse_line(line: str) -> dict | None:
    try:
        if orjson is not None:
            obj = orjson.loads(line)
        else:
            obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def is_noise_text(text: str) -> bool:
    if not text.strip():
        return False

    has_kodak = bool(KODAK_RE.search(text))
    has_kp = bool(KP_RE.search(text))
    has_gray = bool(GRAY_RE.search(text))
    has_color = bool(COLOR_RE.search(text))
    has_colorname = bool(COLORNAME_RE.search(text))

    if has_kodak:
        lower = text.lower()
        if "et al" in lower:
            has_kodak = False
        else:
            has_tm = bool(re.search(r"(tm|®|©|copyright)", text, re.IGNORECASE))
            has_noise_context = has_gray or has_color or has_colorname or has_tm
            clean_text = re.sub(r"[\W_]+", "", lower)
            is_just_kodak = clean_text in ("kodak", "kodek")
            if not (is_just_kodak or has_noise_context):
                has_kodak = False

    strict_gray = bool(STRICT_GRAY_RE.search(text))
    strict_color = bool(STRICT_COLOR_RE.search(text))
    gray_color_noise = has_gray and (has_color or has_colorname)

    if len(text.split()) > 8 and gray_color_noise:
        gray_color_noise = False

    return has_kodak or has_kp or gray_color_noise or strict_gray or strict_color


def extract_samples(
    input_path: Path,
    front_out: TextIO,
    back_out: TextIO,
    limit: int,
    max_lines: int | None,
    apply_noise_filter: bool,
) -> tuple[int, int, int]:
    n_front = 0
    n_back = 0
    seen_lines = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            seen_lines += 1
            if max_lines is not None and seen_lines > max_lines:
                break

            obj = parse_line(raw_line)
            if obj is None:
                continue

            meta = obj.get("meta")
            if not isinstance(meta, dict):
                continue

            idx = meta.get("page_index")
            cnt = meta.get("page_count")
            if not isinstance(idx, int) or not isinstance(cnt, int) or cnt <= 0:
                continue

            need_front = (n_front < limit) and (idx == 0)
            need_back = (n_back < limit) and (idx == cnt - 1)
            if not need_front and not need_back:
                continue

            if apply_noise_filter:
                text = obj.get("text", "")
                if not isinstance(text, str) or not is_noise_text(text):
                    continue

            line = raw_line if raw_line.endswith("\n") else raw_line + "\n"
            if need_front:
                front_out.write(line)
                n_front += 1
            if need_back:
                back_out.write(line)
                n_back += 1

            if n_front >= limit and n_back >= limit:
                break

    return n_front, n_back, seen_lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="先頭/末尾のノイズページを元レコードそのまま JSONL で抽出する。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="入力 JSONL。複数指定可。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="先頭/末尾それぞれの抽出件数 (default: 30)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/edge_noise_samples_jsonl"),
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="デバッグ用: 各入力で読む最大行数",
    )
    parser.add_argument(
        "--include-raw-edge",
        action="store_true",
        help="ノイズ判定なしの先頭/末尾サンプルも追加出力する",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.input:
        if not input_path.exists():
            print(f"[skip] missing: {input_path}")
            continue
        name = input_path.stem
        front_path = args.out_dir / f"{name}.front.noisy.sample.jsonl"
        back_path = args.out_dir / f"{name}.back.noisy.sample.jsonl"

        with (
            front_path.open("w", encoding="utf-8") as fw_front,
            back_path.open("w", encoding="utf-8") as fw_back,
        ):
            n_front, n_back, seen_lines = extract_samples(
                input_path=input_path,
                front_out=fw_front,
                back_out=fw_back,
                limit=args.limit,
                max_lines=args.max_lines,
                apply_noise_filter=True,
            )

        print(f"{input_path}")
        print(f"  front: {front_path} ({n_front} records)")
        print(f"  back : {back_path} ({n_back} records)")
        print(f"  scanned lines: {seen_lines:,}")

        if args.include_raw_edge:
            raw_front_path = args.out_dir / f"{name}.front.raw.sample.jsonl"
            raw_back_path = args.out_dir / f"{name}.back.raw.sample.jsonl"
            with (
                raw_front_path.open("w", encoding="utf-8") as fw_front,
                raw_back_path.open("w", encoding="utf-8") as fw_back,
            ):
                n_front_raw, n_back_raw, seen_lines_raw = extract_samples(
                    input_path=input_path,
                    front_out=fw_front,
                    back_out=fw_back,
                    limit=args.limit,
                    max_lines=args.max_lines,
                    apply_noise_filter=False,
                )
            print(f"  raw front: {raw_front_path} ({n_front_raw} records)")
            print(f"  raw back : {raw_back_path} ({n_back_raw} records)")
            print(f"  raw scanned lines: {seen_lines_raw:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

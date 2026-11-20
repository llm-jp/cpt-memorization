"""Convert archived NDLOCR XML into document or page records."""

import json
import sys
import tarfile
import gzip
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET
import argparse


def read_members(tar_file_path: str) -> Iterator[Tuple[str, str]]:
    """Extract a tar file and yield (member name, content) for every .gz entry."""
    with tarfile.open(tar_file_path, "r:*") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".gz"):
                continue
            with tf.extractfile(member) as gz_stream:
                if gz_stream is None:
                    continue
                with gzip.GzipFile(fileobj=gz_stream) as gz_file:
                    yield member.name, gz_file.read().decode("utf-8")


def read_ocr_dataset(file_content: str) -> Iterator[ET.Element]:
    """Parse OCRDATASET XML snippets and yield them as ElementTree roots."""
    for match in re.finditer(r"<OCRDATASET>.*?</OCRDATASET>", file_content, re.DOTALL):
        ocr_dataset_str = match.group(0)
        ocr_dataset_str = "<?xml version='1.0' encoding='utf-8'?>\n" + ocr_dataset_str
        yield ET.fromstring(ocr_dataset_str)


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _derive_pid(member_name: str) -> str:
    base = member_name.rsplit("/", 1)[-1]
    if base.endswith(".gz"):
        return base[:-3]
    return base


def _make_empty_document_record(pid: str, member_name: str) -> Dict[str, Any]:
    return {
        "text": "",
        "meta": {
            "pid": pid,
            "archive_member": member_name,
            "page_count": 0,
            "line_count": 0,
        },
    }


def _make_empty_page_record(pid: str, member_name: str) -> Dict[str, Any]:
    return {
        "text": "",
        "meta": {
            "pid": pid,
            "archive_member": member_name,
            "page_count": 0,
            "line_count": 0,
            "empty_document": True,
        },
    }


def _iter_line_elements(parent: ET.Element) -> Iterator[ET.Element]:
    for child in parent:
        if child.tag == "LINE":
            yield child
        else:
            yield from _iter_line_elements(child)


def extract_document(
    member_name: str,
    file_content: str,
    drop_first_pages: int = 0,
    drop_last_pages: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Aggregate body text from all OCRDATASET entries for a single archive member.

    Parameters
    ----------
    member_name : str
        Archive member name.
    file_content : str
        Decompressed XML content.
    drop_first_pages : int, optional
        Number of pages to drop from the beginning of the document before export.
    drop_last_pages : int, optional
        Number of pages to drop from the end of the document before export.
    """
    pid = _derive_pid(member_name)
    page_entries: List[Tuple[str, Dict[str, Any]]] = []

    # まずは「本文」行をページ単位で全部集める
    for dataset in read_ocr_dataset(file_content):
        for page_elem in dataset.findall("PAGE"):
            page_lines: List[str] = []
            for line_elem in _iter_line_elements(page_elem):
                if line_elem.attrib.get("TYPE") != "本文":
                    continue
                line_text = line_elem.attrib.get("STRING")
                if not line_text:
                    continue
                page_lines.append(line_text)
            if not page_lines:
                continue

            page_text = "".join(page_lines)

            page_meta = _drop_none(
                {
                    # ここは元データ中でのページ順（0始まり）
                    "index": len(page_entries),
                    "image_name": page_elem.attrib.get("IMAGENAME"),
                    "width": _to_int(page_elem.attrib.get("WIDTH")),
                    "height": _to_int(page_elem.attrib.get("HEIGHT")),
                    "line_count": len(page_lines),
                }
            )
            page_entries.append((page_text, page_meta))

    # ここから「最初のNページ」「最後のMページ」を削除していく
    total_pages = len(page_entries)
    if total_pages == 0:
        return _make_empty_document_record(pid, member_name), [
            _make_empty_page_record(pid, member_name)
        ]

    # 削除枚数が総ページ数を超えないように安全にクリップ
    drop_first = max(0, drop_first_pages)
    drop_last = max(0, drop_last_pages)

    # keep_start / keep_end はスライス用のインデックス
    keep_start = min(drop_first, total_pages)
    keep_end = max(total_pages - drop_last, keep_start)

    kept_entries = page_entries[keep_start:keep_end]
    kept_page_texts = [text for text, _ in kept_entries]
    kept_pages = [meta for _, meta in kept_entries]

    # 残ったページの行数を再集計
    total_lines = sum(p.get("line_count", 0) for p in kept_pages)

    document_meta: Dict[str, Any] = {
        "pid": pid,
        "archive_member": member_name,
        "page_count": len(kept_pages),
        "line_count": total_lines,
    }
    if kept_pages:
        document_meta["pages"] = kept_pages

    if not kept_entries:
        return {"text": "", "meta": document_meta}, [_make_empty_page_record(pid, member_name)]

    page_records: List[Dict[str, Any]] = []
    page_count = len(kept_entries)
    for text, meta in kept_entries:
        page_meta = _drop_none(
            {
                "pid": pid,
                "archive_member": member_name,
                "page_index": meta.get("index"),
                "page_count": page_count,
                "image_name": meta.get("image_name"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "line_count": meta.get("line_count"),
            }
        )
        page_records.append({"text": text, "meta": page_meta})

    document_record = {"text": "".join(kept_page_texts), "meta": document_meta}
    return document_record, page_records


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from an NDLOCR archive as JSONL.",
    )
    parser.add_argument(
        "input_path",
        help="Path to a tar archive containing gzip-compressed NDLOCR XML files.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Output JSONL file path. If omitted, write to stdout.",
    )
    parser.add_argument(
        "--drop-first-pages",
        type=int,
        default=0,
        help="Number of pages to drop from the beginning of each document.",
    )
    parser.add_argument(
        "--drop-last-pages",
        type=int,
        default=0,
        help="Number of pages to drop from the end of each document.",
    )
    parser.add_argument(
        "--emit-pages",
        action="store_true",
        help=(
            "Emit page-level JSONL (1ページ=1行)。このフラグを指定すると冊子全体の行は出力しません。"
            " 本文ゼロ件の冊子も PID を保つため空プレースホルダ1行を出力します。"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    input_path = args.input_path
    output_path = args.output_path

    from ...io import open_text

    output_stream = open_text(output_path, "wt") if output_path else sys.stdout
    try:
        for member_name, file_content in read_members(input_path):
            record, page_records = extract_document(
                member_name,
                file_content,
                drop_first_pages=args.drop_first_pages,
                drop_last_pages=args.drop_last_pages,
            )
            if args.emit_pages:
                for page_record in page_records:
                    output_stream.write(json.dumps(page_record, ensure_ascii=False))
                    output_stream.write("\n")
            else:
                output_stream.write(json.dumps(record, ensure_ascii=False))
                output_stream.write("\n")
    finally:
        if output_path and output_stream is not sys.stdout:
            output_stream.close()


if __name__ == "__main__":
    main()

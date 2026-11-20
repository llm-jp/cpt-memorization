"""Shared helpers for low-Jaccard near-duplicate query/count scripts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


TOKEN_ID_COLUMNS = {
    "suffix": "suffix_token_ids",
    "predicted": "predicted_token_ids",
    "prefix": "prefix_token_ids",
    "prefix_tail": "prefix_token_ids",
    "prefix_predicted": "prefix_token_ids",
}


def parse_float(value: str, column: str) -> float:
    try:
        return float((value or "").strip())
    except ValueError as exc:
        raise SystemExit(f"Invalid float in {column}: {value!r}") from exc


def parse_token_ids(value: str, column: str) -> list[int]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON token list in {column}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise SystemExit(f"{column} must be a JSON list of integers")
    return parsed


def load_low_jaccard_rows(
    path: Path,
    jaccard_max: float,
    limit: int,
) -> tuple[list[str], list[dict[str, str]]]:
    """Load non-empty CSV rows with jaccard_score below the threshold."""

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SystemExit(f"No CSV header found: {path}")
        if "jaccard_score" not in fieldnames:
            raise SystemExit(f"CSV must contain jaccard_score: {path}")

        rows = []
        for csv_row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            if parse_float(row.get("jaccard_score", ""), "jaccard_score") >= jaccard_max:
                continue
            out = dict(row)
            if not out.get("input_csv_row_number"):
                out["input_csv_row_number"] = str(csv_row_number)
            rows.append(out)
            if limit > 0 and len(rows) >= limit:
                break

    if "input_csv_row_number" not in fieldnames:
        fieldnames.append("input_csv_row_number")
    return fieldnames, rows


def token_ids_for_query(
    row: dict[str, str],
    query_type: str,
    prefix_tail_tokens: int,
) -> list[int]:
    if query_type == "prefix_predicted":
        return parse_token_ids(
            row.get("prefix_token_ids", ""), "prefix_token_ids"
        ) + parse_token_ids(row.get("predicted_token_ids", ""), "predicted_token_ids")

    column = TOKEN_ID_COLUMNS[query_type]
    token_ids = parse_token_ids(row.get(column, ""), column)
    if query_type == "prefix_tail" and prefix_tail_tokens > 0:
        return token_ids[-prefix_tail_tokens:]
    return token_ids


def query_text(row: dict[str, str], query_type: str) -> str:
    if query_type == "prefix_predicted":
        return row.get("prefix", "") + row.get("predicted", "")
    if query_type == "prefix_tail":
        return row.get("prefix", "")
    return row.get(query_type, "")


def build_query(
    row: dict[str, str],
    query_id: int,
    query_type: str,
    prefix_tail_tokens: int,
) -> dict[str, Any]:
    token_ids = token_ids_for_query(row, query_type, prefix_tail_tokens)
    if not token_ids:
        raise SystemExit(
            f"Empty token_ids for query_id={query_id}, "
            f"input_csv_row_number={row.get('input_csv_row_number')}, query_type={query_type}"
        )
    return {
        "query_id": query_id,
        "query_type": query_type,
        "variant": row.get("variant", ""),
        "sample_file": row.get("source_file", row.get("sample_file", "")),
        "line_no": row.get("row_index", row.get("line_no", "")),
        "input_csv_row_number": row.get("input_csv_row_number", ""),
        "archive_member": row.get("archive_member", ""),
        "doc_id": row.get("doc_id", ""),
        "snippet_length": int(row["snippet_length"])
        if str(row.get("snippet_length", "")).isdigit()
        else row.get("snippet_length", ""),
        "jaccard_score": parse_float(row.get("jaccard_score", ""), "jaccard_score"),
        "bleu": parse_float(row.get("bleu", ""), "bleu") if row.get("bleu", "") else "",
        "prefix": row.get("prefix", ""),
        "suffix": row.get("suffix", ""),
        "predicted": row.get("predicted", ""),
        "query_text": query_text(row, query_type),
        "token_ids": token_ids,
        "token_count": len(token_ids),
    }


def build_queries(
    rows: list[dict[str, str]],
    query_types: list[str],
    prefix_tail_tokens: int,
) -> list[dict[str, Any]]:
    queries = []
    query_id = 0
    for row in rows:
        for query_type in query_types:
            queries.append(build_query(row, query_id, query_type, prefix_tail_tokens))
            query_id += 1
    return queries


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_query_outputs(
    *,
    input_csv: Path,
    output_dir: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    queries: list[dict[str, Any]],
    jaccard_max: float,
    query_types: list[str],
    prefix_tail_tokens: int,
    limit_queries: int,
) -> dict[str, Any]:
    sample_csv = output_dir / "low_jaccard_samples.csv"
    query_jsonl = output_dir / "query.jsonl"
    query_meta_json = output_dir / "query_meta.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(sample_csv, fieldnames, rows)
    with query_jsonl.open("w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

    meta = {
        "input_csv": str(input_csv),
        "sample_csv": str(sample_csv),
        "query_jsonl": str(query_jsonl),
        "rows_written": len(rows),
        "queries_written": len(queries),
        "jaccard_max": jaccard_max,
        "query_types": query_types,
        "prefix_tail_tokens": prefix_tail_tokens,
        "limit_queries": limit_queries,
    }
    if len(query_types) == 1:
        meta["query_type"] = query_types[0]
    query_meta_json.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def neardup_count_column(query_type: str, ngram: int, threshold: str) -> str:
    return f"{query_type}_near_dup_occurrence_count_jaccard{ngram}gram_ge_{threshold}"


def read_sample_csv(sample_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        sample_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise SystemExit(f"No sample CSV header found: {sample_csv}")
    return fieldnames, sample_rows


def read_count_rows(counts_jsonl: Path) -> list[dict[str, Any]]:
    rows = []
    with counts_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_counts_by_query_id(
    *,
    sample_csv: Path,
    counts_jsonl: Path,
    output_csv: Path,
    count_column: str,
) -> Path:
    fieldnames, sample_rows = read_sample_csv(sample_csv)
    counts_by_query_id = {
        int(obj["query_id"]): int(obj["near_dup_occurrence_count"])
        for obj in read_count_rows(counts_jsonl)
    }

    if len(counts_by_query_id) != len(sample_rows):
        raise SystemExit(
            f"Count/sample length mismatch: counts={len(counts_by_query_id)} samples={len(sample_rows)}"
        )

    if count_column not in fieldnames:
        fieldnames.append(count_column)
    for query_id, row in enumerate(sample_rows):
        row[count_column] = str(counts_by_query_id[query_id])

    write_csv(output_csv, fieldnames, sample_rows)
    return output_csv.resolve()


def merge_counts_by_input_csv_row_number(
    *,
    sample_csv: Path,
    counts_jsonl: Path,
    output_csv: Path,
    query_types: list[str],
    threshold: str,
    ngram: int,
) -> Path:
    fieldnames, sample_rows = read_sample_csv(sample_csv)

    counts_by_row_type: dict[tuple[str, str], int] = {}
    for obj in read_count_rows(counts_jsonl):
        query_type = str(obj.get("query_type", "suffix"))
        if query_type not in query_types:
            continue
        input_csv_row_number = str(obj.get("input_csv_row_number", ""))
        if not input_csv_row_number:
            raise SystemExit(
                f"Missing input_csv_row_number in counts row: query_id={obj.get('query_id')}"
            )
        key = (input_csv_row_number, query_type)
        if key in counts_by_row_type:
            raise SystemExit(
                f"Duplicate count for input_csv_row_number={key[0]} query_type={key[1]}"
            )
        counts_by_row_type[key] = int(obj["near_dup_occurrence_count"])

    added_columns = [
        neardup_count_column(query_type, ngram, threshold) for query_type in query_types
    ]
    for column in added_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    missing = []
    for physical_row_number, row in enumerate(sample_rows, start=2):
        input_csv_row_number = row.get("input_csv_row_number") or str(physical_row_number)
        for query_type, column in zip(query_types, added_columns):
            key = (str(input_csv_row_number), query_type)
            if key not in counts_by_row_type:
                row[column] = ""
                missing.append(
                    {"input_csv_row_number": input_csv_row_number, "query_type": query_type}
                )
                continue
            row[column] = str(counts_by_row_type.pop(key))

    if missing:
        preview = ", ".join(
            f"input_csv_row_number={item['input_csv_row_number']} type={item['query_type']}"
            for item in missing[:10]
        )
        raise SystemExit(
            f"Missing counts for {len(missing)} row/query_type pairs. First missing: {preview}"
        )
    if counts_by_row_type:
        raise SystemExit(f"Unused count rows remain after merge: {len(counts_by_row_type)}")

    write_csv(output_csv, fieldnames, sample_rows)
    return output_csv.resolve()


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        str(row.get("variant", "")),
        str(row.get("source_file", row.get("sample_file", ""))),
        str(row.get("archive_member", "")),
        str(row.get("snippet_length", "")),
        str(row.get("line_no", "")),
        str(row.get("prefix", "")),
        str(row.get("suffix", "")),
        str(row.get("predicted", "")),
    )


def count_key(obj: dict[str, object]) -> tuple[str, ...]:
    return (
        str(obj.get("variant", "")),
        str(obj.get("sample_file", obj.get("source_file", ""))),
        str(obj.get("archive_member", "")),
        str(obj.get("snippet_length", "")),
        str(obj.get("line_no", "")),
        str(obj.get("prefix", "")),
        str(obj.get("suffix", "")),
        str(obj.get("predicted", "")),
    )


def merge_counts_by_row_content(
    *,
    sample_csv: Path,
    counts_jsonl: Path,
    output_csv: Path,
    query_types: list[str],
    threshold: str,
    ngram: int,
) -> Path:
    fieldnames, sample_rows = read_sample_csv(sample_csv)

    counts_by_key_type: dict[tuple[tuple[str, ...], str], deque[int]] = defaultdict(deque)
    for obj in read_count_rows(counts_jsonl):
        query_type = str(obj.get("query_type", "suffix"))
        if query_type not in query_types:
            continue
        counts_by_key_type[(count_key(obj), query_type)].append(
            int(obj["near_dup_occurrence_count"])
        )

    added_columns = [
        neardup_count_column(query_type, ngram, threshold) for query_type in query_types
    ]
    for column in added_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    missing = []
    for row_index, row in enumerate(sample_rows, start=1):
        key = row_key(row)
        for query_type, column in zip(query_types, added_columns):
            queue = counts_by_key_type.get((key, query_type))
            if not queue:
                row[column] = ""
                missing.append({"row_index_1based": row_index, "query_type": query_type})
                continue
            row[column] = str(queue.popleft())

    leftovers = sum(len(queue) for queue in counts_by_key_type.values())
    if missing:
        preview = ", ".join(
            f"row={item['row_index_1based']} type={item['query_type']}" for item in missing[:10]
        )
        raise SystemExit(
            f"Missing counts for {len(missing)} row/query_type pairs. First missing: {preview}"
        )
    if leftovers:
        raise SystemExit(f"Unused count rows remain after merge: {leftovers}")

    write_csv(output_csv, fieldnames, sample_rows)
    return output_csv.resolve()

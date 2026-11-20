#!/usr/bin/env python
import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TOKENIZER = None
SUFFIX_BY_LENGTH: Dict[int, Dict[Tuple[int, ...], List[str]]] = {}


class TokenizersTokenizer:
    def __init__(self, tokenizer, tokenizer_dir: Optional[Path] = None):
        self.tokenizer = tokenizer
        self.model_max_length = int(1e30)
        self.bos_token_id = None
        self.eos_token_id = None
        if tokenizer_dir is not None:
            self._load_special_token_ids(tokenizer_dir)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if add_special_tokens:
            return self.tokenizer.encode(text).ids
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _load_special_token_ids(self, tokenizer_dir: Path) -> None:
        for config_name in ("tokenizer_config.json", "special_tokens_map.json"):
            config_path = tokenizer_dir / config_name
            if not config_path.is_file():
                continue
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            if self.bos_token_id is None:
                self.bos_token_id = self._token_id_from_config(config.get("bos_token"))
            if self.eos_token_id is None:
                self.eos_token_id = self._token_id_from_config(config.get("eos_token"))

    def _token_id_from_config(self, value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get("content")
        if not isinstance(value, str):
            return None
        return self.tokenizer.token_to_id(value)


def load_tokenizer(tokenizer_name: str):
    tokenizer_path = Path(tokenizer_name)
    tokenizer_json = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    if tokenizer_json.is_file() and tokenizer_json.name == "tokenizer.json":
        from tokenizers import Tokenizer

        tokenizer_dir = tokenizer_json.parent if tokenizer_path.is_dir() else None
        return TokenizersTokenizer(Tokenizer.from_file(str(tokenizer_json)), tokenizer_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = int(1e30)
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count token-level exact matches of CSV suffixes inside JSONL/JSONL.GZ "
            "text fields under a processed corpus root."
        )
    )
    parser.add_argument("--csv-file", required=True, help="Input CSV path.")
    parser.add_argument(
        "--processed-root", required=True, help="Root containing .jsonl/.jsonl.gz files."
    )
    parser.add_argument("--output-csv", default=None, help="Output CSV path.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer path/name. If omitted, infer from CSV tokenizer column when possible.",
    )
    parser.add_argument(
        "--suffix-column",
        default="suffix",
        help="Text suffix column used when token ids are absent.",
    )
    parser.add_argument(
        "--suffix-token-ids-column",
        default="suffix_token_ids",
        help="Column containing JSON/Python-list token ids.",
    )
    parser.add_argument(
        "--tokenizer-column", default="tokenizer", help="Column used to infer --tokenizer."
    )
    parser.add_argument(
        "--count-column", default="suffix_dup_count", help="Output count column name."
    )
    parser.add_argument("--text-key", default="text", help="JSON field to scan.")
    parser.add_argument("--workers", default="auto", help="Positive integer or auto.")
    parser.add_argument("--batch-lines", type=int, default=20000, help="Lines per worker task.")
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend BOS token when encoding CSV suffix text and processed text.",
    )
    parser.add_argument(
        "--add-eos",
        action="store_true",
        help="Append EOS token when encoding CSV suffix text and processed text.",
    )
    parser.add_argument(
        "--append-eod",
        action="store_true",
        help="Append EOD token when encoding CSV suffix text and processed text.",
    )
    parser.add_argument(
        "--eod-token-id", type=int, default=None, help="Explicit EOD token id for --append-eod."
    )
    return parser.parse_args()


def detect_workers() -> int:
    candidates = []
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value and value.isdigit():
            candidates.append(int(value))
    cpu_count = os.cpu_count()
    if cpu_count:
        candidates.append(cpu_count)
    return max(candidates or [1])


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_processed_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"))
    )


def parse_token_ids(value: str) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    value = value.strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"token id value is not a list: {value[:80]}")
    return [int(token_id) for token_id in parsed]


def token_sequence_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_worker(
    tokenizer_name: str,
    suffix_items: Sequence[Tuple[str, Tuple[int, ...]]],
    add_bos: bool,
    add_eos: bool,
    append_eod: bool,
    eod_token_id: Optional[int],
) -> None:
    global TOKENIZER, SUFFIX_BY_LENGTH
    TOKENIZER = load_tokenizer(tokenizer_name)
    SUFFIX_BY_LENGTH = defaultdict(lambda: defaultdict(list))
    for suffix_hash, token_ids in suffix_items:
        if token_ids:
            SUFFIX_BY_LENGTH[len(token_ids)][token_ids].append(suffix_hash)

    def encode_text(text: str) -> List[int]:
        token_ids = TOKENIZER.encode(text, add_special_tokens=False)
        if add_bos:
            if TOKENIZER.bos_token_id is None:
                raise ValueError("Tokenizer has no BOS token id.")
            token_ids = [TOKENIZER.bos_token_id] + token_ids
        if add_eos:
            if TOKENIZER.eos_token_id is None:
                raise ValueError("Tokenizer has no EOS token id.")
            token_ids = token_ids + [TOKENIZER.eos_token_id]
        if append_eod:
            actual_eod = eod_token_id
            if actual_eod is None:
                if TOKENIZER.eos_token_id is None:
                    raise ValueError("Tokenizer has no EOS token id for EOD fallback.")
                actual_eod = TOKENIZER.eos_token_id
            token_ids = token_ids + [actual_eod]
        return token_ids

    TOKENIZER.encode_for_suffix_count = encode_text


def count_batch(lines: Sequence[str], text_key: str) -> Dict[str, int]:
    matched = Counter()
    if TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")

    for raw_line in lines:
        raw_line = raw_line.rstrip("\n")
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        text = obj.get(text_key) if isinstance(obj, dict) else None
        if not isinstance(text, str) or not text:
            continue
        token_ids = TOKENIZER.encode_for_suffix_count(text)
        token_count = len(token_ids)
        for suffix_length, lookup in SUFFIX_BY_LENGTH.items():
            if suffix_length > token_count:
                continue
            for start in range(0, token_count - suffix_length + 1):
                window = tuple(token_ids[start : start + suffix_length])
                suffix_hashes = lookup.get(window)
                if suffix_hashes:
                    for suffix_hash in suffix_hashes:
                        matched[suffix_hash] += 1
    return dict(matched)


def infer_tokenizer(rows: Sequence[dict], tokenizer_column: str) -> Optional[str]:
    tokenizers = {
        row.get(tokenizer_column, "")
        for row in rows
        if row.get(tokenizer_column, "") not in (None, "")
    }
    if len(tokenizers) == 1:
        return next(iter(tokenizers))
    return None


def encode_with_config(
    tokenizer,
    text: str,
    add_bos: bool,
    add_eos: bool,
    append_eod: bool,
    eod_token_id: Optional[int],
) -> List[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if add_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer has no BOS token id.")
        token_ids = [tokenizer.bos_token_id] + token_ids
    if add_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no EOS token id.")
        token_ids = token_ids + [tokenizer.eos_token_id]
    if append_eod:
        actual_eod = eod_token_id
        if actual_eod is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no EOS token id for EOD fallback.")
            actual_eod = tokenizer.eos_token_id
        token_ids = token_ids + [actual_eod]
    return token_ids


def load_csv_suffixes(
    args: argparse.Namespace,
) -> Tuple[List[dict], List[str], Dict[str, int], Dict[str, Tuple[int, ...]], str]:
    with open(args.csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("input csv has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    tokenizer_name = args.tokenizer or infer_tokenizer(rows, args.tokenizer_column)
    if tokenizer_name is None:
        raise SystemExit(
            "--tokenizer is required unless the CSV has exactly one non-empty "
            f"{args.tokenizer_column!r} value."
        )

    tokenizer = None
    csv_counts = Counter()
    hash_to_tokens: Dict[str, Tuple[int, ...]] = {}

    has_token_ids_column = args.suffix_token_ids_column in fieldnames
    if not has_token_ids_column and args.suffix_column not in fieldnames:
        available = ", ".join(fieldnames)
        raise SystemExit(
            f"neither suffix token ids column nor suffix column was found: "
            f"{args.suffix_token_ids_column}, {args.suffix_column}. available columns: {available}"
        )

    for row in rows:
        token_ids = None
        if has_token_ids_column:
            raw_token_ids = row.get(args.suffix_token_ids_column, "")
            if raw_token_ids not in (None, ""):
                token_ids = parse_token_ids(raw_token_ids)

        if token_ids is None:
            suffix = row.get(args.suffix_column, "")
            if suffix is None or suffix == "":
                continue
            if tokenizer is None:
                tokenizer = load_tokenizer(tokenizer_name)
            token_ids = encode_with_config(
                tokenizer,
                suffix,
                args.add_bos,
                args.add_eos,
                args.append_eod,
                args.eod_token_id,
            )

        if not token_ids:
            continue
        suffix_hash = token_sequence_hash(token_ids)
        csv_counts[suffix_hash] += 1
        hash_to_tokens.setdefault(suffix_hash, tuple(token_ids))

    return rows, fieldnames, dict(csv_counts), hash_to_tokens, tokenizer_name


def scan_files(
    files: Sequence[Path],
    workers: int,
    tokenizer_name: str,
    hash_to_tokens: Dict[str, Tuple[int, ...]],
    text_key: str,
    batch_lines: int,
    args: argparse.Namespace,
) -> Counter:
    if workers == 1:
        init_worker(
            tokenizer_name,
            tuple(hash_to_tokens.items()),
            args.add_bos,
            args.add_eos,
            args.append_eod,
            args.eod_token_id,
        )
        counts = Counter()
        for path in files:
            with open_text(path) as stream:
                batch = []
                for line in stream:
                    batch.append(line)
                    if len(batch) >= batch_lines:
                        counts.update(count_batch(batch, text_key))
                        batch = []
                if batch:
                    counts.update(count_batch(batch, text_key))
        return counts
    max_in_flight = max(2, workers * 2)
    matched_all = Counter()
    suffix_items = tuple(hash_to_tokens.items())

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            tokenizer_name,
            suffix_items,
            args.add_bos,
            args.add_eos,
            args.append_eod,
            args.eod_token_id,
        ),
    ) as executor:
        futures = []

        def flush_some(block: bool = False) -> None:
            nonlocal futures
            if not futures:
                return
            done, not_done = wait(
                futures,
                timeout=None if block else 0,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                return
            for future in done:
                matched_all.update(future.result())
            futures = list(not_done)

        for path in files:
            with open_text(path) as f:
                batch = []
                for line in f:
                    batch.append(line)
                    if len(batch) >= batch_lines:
                        futures.append(executor.submit(count_batch, batch, text_key))
                        batch = []
                        while len(futures) >= max_in_flight:
                            flush_some(block=True)
                if batch:
                    futures.append(executor.submit(count_batch, batch, text_key))

        while futures:
            flush_some(block=True)

    return matched_all


def write_output_csv(
    rows: Iterable[dict],
    fieldnames: Sequence[str],
    output_csv: str,
    count_column: str,
    suffix_token_ids_column: str,
    suffix_column: str,
    hash_to_count: Dict[str, int],
    tokenizer_name: str,
    args: argparse.Namespace,
) -> None:
    out_fields = list(fieldnames)
    if count_column not in out_fields:
        out_fields.append(count_column)

    tokenizer = None
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            token_ids = None
            raw_token_ids = row.get(suffix_token_ids_column, "")
            if raw_token_ids not in (None, ""):
                token_ids = parse_token_ids(raw_token_ids)
            elif row.get(suffix_column, "") not in (None, ""):
                if tokenizer is None:
                    tokenizer = load_tokenizer(tokenizer_name)
                token_ids = encode_with_config(
                    tokenizer,
                    row.get(suffix_column, ""),
                    args.add_bos,
                    args.add_eos,
                    args.append_eod,
                    args.eod_token_id,
                )

            row[count_column] = (
                hash_to_count.get(token_sequence_hash(token_ids), 0) if token_ids else 0
            )
            writer.writerow(row)


def default_output_path(csv_file: str) -> str:
    if csv_file.endswith(".csv"):
        return f"{csv_file[:-4]}_with_suffix_dup_count.csv"
    return f"{csv_file}_with_suffix_dup_count.csv"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_file)
    processed_root = Path(args.processed_root)
    if not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")
    if not processed_root.is_dir():
        raise SystemExit(f"Processed root not found: {processed_root}")

    workers = detect_workers() if args.workers == "auto" else int(args.workers)
    if workers <= 0:
        raise SystemExit(f"--workers must be positive or auto. Got: {args.workers}")
    if args.batch_lines <= 0:
        raise SystemExit("--batch-lines must be positive.")

    output_csv = args.output_csv or default_output_path(args.csv_file)
    rows, fieldnames, csv_counts, hash_to_tokens, tokenizer_name = load_csv_suffixes(args)
    files = iter_processed_files(processed_root)
    if not files:
        raise SystemExit(f"No .jsonl/.jsonl.gz files found under: {processed_root}")

    print(f"tokenizer={tokenizer_name}")
    print(f"input_jsonl_files={len(files)}")
    print(f"csv_rows_total={len(rows)}")
    print(f"csv_rows_non_empty_suffix={sum(csv_counts.values())}")
    print(f"csv_unique_suffixes={len(hash_to_tokens)}")

    processed_counts = scan_files(
        files,
        workers,
        tokenizer_name,
        hash_to_tokens,
        args.text_key,
        args.batch_lines,
        args,
    )

    matched_unique_suffixes = len(processed_counts)
    matched_rows = sum(csv_counts.get(suffix_hash, 0) for suffix_hash in processed_counts)
    total_exact_occurrences = sum(processed_counts.values())

    print(f"matched_unique_suffixes={matched_unique_suffixes}")
    print(f"matched_rows_by_suffix={matched_rows}")
    print(f"total_token_exact_occurrences_in_processed={total_exact_occurrences}")

    write_output_csv(
        rows,
        fieldnames,
        output_csv,
        args.count_column,
        args.suffix_token_ids_column,
        args.suffix_column,
        dict(processed_counts),
        tokenizer_name,
        args,
    )

    print(f"workers_used={workers}")
    print(f"output_csv={output_csv}")
    print(f"count_column={args.count_column}")


if __name__ == "__main__":
    main()

"""Shared JSONL I/O. Compression is inferred from the .gz extension."""

import gzip
import json
from pathlib import Path


def open_text(path, mode="rt"):
    path = Path(path)
    if any(flag in mode for flag in "wax"):
        path.parent.mkdir(parents=True, exist_ok=True)
    return (gzip.open if path.suffix == ".gz" else open)(path, mode, encoding="utf-8")


def read_jsonl(path):
    with open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                yield record


def write_jsonl(path, records):
    with open_text(path, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    with open_text(path, "wt") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

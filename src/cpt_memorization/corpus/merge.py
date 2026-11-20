"""Merge JSONL shards in deterministic input/directory order."""

import argparse
from pathlib import Path

from ..io import read_jsonl, write_jsonl


def merge(inputs, output):
    output = Path(output).resolve()
    sources = []
    for value in inputs:
        path = Path(value).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir() and output.is_relative_to(path):
            raise ValueError("Output must be outside input directories")
        sources.extend(
            p
            for p in (sorted(path.rglob("*")) if path.is_dir() else [path])
            if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
        )
    if not sources or len({p.resolve() for p in sources}) != len(sources):
        raise ValueError("Inputs must contain nonempty, non-overlapping JSONL file sets")
    if output in {p.resolve() for p in sources}:
        raise ValueError("Input and output must be different files")
    write_jsonl(output, (record for source in sources for record in read_jsonl(source)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()

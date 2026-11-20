"""Stage multiple JSONL corpora as a deterministic, zero-copy shard directory."""

import argparse
from pathlib import Path

from ..io import write_json


def stage(inputs, output):
    output = Path(output).resolve()
    sources = []
    for value in inputs:
        path = Path(value).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if output == path or (path.is_dir() and output.is_relative_to(path)):
            raise ValueError("Staging output must not be inside an input directory")
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        sources.extend(
            p.resolve()
            for p in candidates
            if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
        )
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Inputs must contain nonempty, non-overlapping JSONL file sets")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Staging directory must be empty: {output}")
    records = []
    for index, source in enumerate(sources):
        target = output / f"{index:06d}" / source.name
        target.parent.mkdir(parents=True)
        target.symlink_to(source)
        records.append({"source": str(source), "shard": str(target.relative_to(output))})
    write_json(output / "staging.json", {"shards": records})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stage(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()

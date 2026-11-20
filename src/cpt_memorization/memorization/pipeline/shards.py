"""Partition samples and merge independently generated completions in input order.

Round-robin partitioning needs no extra IDs in the records. Merging verifies
every source record and token length before atomically publishing the result.
Neither operation depends on Slurm, model architecture, or a particular corpus.
"""

import argparse
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import tempfile

from ...io import read_jsonl


def shard_path(directory: Path, index: int) -> Path:
    return directory / f"part-{index:05d}.jsonl"


def split_samples(source: Path, directory: Path, num_shards: int) -> list[int]:
    if num_shards < 1:
        raise ValueError("num-shards must be positive")
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    counts = [0] * num_shards
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".shards-") as temporary:
        staging = Path(temporary) / "parts"
        staging.mkdir()
        with ExitStack() as stack:
            streams = [
                stack.enter_context(shard_path(staging, i).open("w", encoding="utf-8"))
                for i in range(num_shards)
            ]
            for index, sample in enumerate(read_jsonl(source)):
                shard = index % num_shards
                streams[shard].write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts[shard] += 1
        if not all(counts):
            raise ValueError("Fewer sample rows than workers; reduce nodes/GPUs per node")
        staging.rename(directory)
    return counts


def merge_completions(source: Path, parts: list[Path], output: Path) -> int:
    if not parts:
        raise ValueError("At least one completion shard is required")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    generated_fields = {"predicted", "predicted_token_ids", "generation"}
    try:
        with ExitStack() as stack:
            readers = [stack.enter_context(part.open(encoding="utf-8")) for part in parts]
            stream = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".tmp",
                    delete=False,
                )
            )
            temporary = Path(stream.name)
            for index, sample in enumerate(read_jsonl(source)):
                line = readers[index % len(parts)].readline()
                if not line:
                    raise ValueError(f"Missing completion at input row {index}")
                completion = json.loads(line)
                original = {k: v for k, v in sample.items() if k not in generated_fields}
                actual = {k: v for k, v in completion.items() if k not in generated_fields}
                if actual != original:
                    raise ValueError(f"Completion does not match input row {index}")
                suffix = sample["suffix_token_ids"]
                predicted = completion.get("predicted_token_ids")
                if (
                    not suffix
                    or not isinstance(predicted, list)
                    or len(predicted) != len(suffix)
                    or any(type(token) is not int for token in predicted)
                ):
                    raise ValueError(f"Invalid generated token length/type at input row {index}")
                stream.write(json.dumps(completion, ensure_ascii=False) + "\n")
                count += 1
            if not count:
                raise ValueError("Empty sample file")
            if any(reader.read(1) for reader in readers):
                raise ValueError("Extra completion rows after end of input")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    split = commands.add_parser("split")
    split.add_argument("--input-file", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--num-shards", type=int, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--input-file", type=Path, required=True)
    merge.add_argument(
        "--shard-files",
        type=Path,
        nargs="+",
        required=True,
        help="Completion shards in numeric shard order",
    )
    merge.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "split":
        counts = split_samples(args.input_file, args.output_dir, args.num_shards)
        logging.info("Split %d rows into %d shards: %s", sum(counts), len(counts), counts)
    else:
        count = merge_completions(args.input_file, args.shard_files, args.output_file)
        logging.info("Merged and validated %d completions", count)


if __name__ == "__main__":
    main()

import argparse
import logging
import json
import gzip
import os
import random
from typing import Dict, List, Optional

from tqdm import tqdm


def read_file(file_path):
    from ...io import open_text

    with open_text(file_path) as stream:
        yield from tqdm(stream, desc="Reading file")


def main():
    parser = argparse.ArgumentParser(description="Extract samples")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to use for generation.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="Tokenizer name/path (defaults to --model).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input file containing data for analysis.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output file to save results.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1_000,
        help="Number of samples to analyze.",
    )
    parser.add_argument(
        "--snippet-length",
        type=int,
        nargs="+",
        default=[200, 400, 600, 800, 1_000],
        help="List of snippet (prefix + suffix) lengths.",
    )
    parser.add_argument(
        "--suffix-length",
        type=int,
        default=50,
        help="Length of the suffix to consider for memorization.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        required=False,
        help="Total number of rows in the input file (if known).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for anchor sampling.",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend BOS token before sampling.",
    )
    parser.add_argument(
        "--add-eos",
        action="store_true",
        help="Append EOS token before sampling.",
    )
    parser.add_argument(
        "--append-eod",
        action="store_true",
        help="Append EOD token before sampling.",
    )
    parser.add_argument(
        "--eod-token-id",
        type=int,
        default=None,
        help="Explicit token id to use for EOD when --append-eod is set.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    if args.input_size is not None:
        logging.info(
            f"Using provided input size (unused in reservoir sampling): {args.input_size:,}"
        )
    else:
        logging.info("Input size is not required for reservoir sampling.")

    if (
        args.sample_size <= 0
        or not args.snippet_length
        or len(set(args.snippet_length)) != len(args.snippet_length)
    ):
        raise ValueError("sample-size must be positive and snippet lengths must be unique")
    if args.suffix_length <= 0:
        raise ValueError("--suffix-length must be positive.")
    if any(snippet_len <= args.suffix_length for snippet_len in args.snippet_length):
        raise ValueError("All --snippet-length values must exceed --suffix-length.")

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rng = random.Random(args.seed)
    max_snippet_len = max(args.snippet_length)

    def _encode_text(text: str) -> List[int]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if args.add_bos:
            if tokenizer.bos_token_id is None:
                raise ValueError("Tokenizer has no BOS token id.")
            token_ids = [tokenizer.bos_token_id] + token_ids
        if args.add_eos:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no EOS token id.")
            token_ids = token_ids + [tokenizer.eos_token_id]
        if args.append_eod:
            eod_token_id = args.eod_token_id
            if eod_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer has no EOS token id for EOD fallback.")
                eod_token_id = tokenizer.eos_token_id
            token_ids = token_ids + [eod_token_id]
        return token_ids

    def _build_anchor(doc_id: str, row_index: int, token_ids: List[int]) -> Dict:
        end_idx = rng.randint(max_snippet_len, len(token_ids))
        suffix = token_ids[end_idx - args.suffix_length : end_idx]
        prefixes_by_len = {}
        for snippet_len in args.snippet_length:
            start_idx = end_idx - snippet_len
            prefix = token_ids[start_idx : end_idx - args.suffix_length]
            prefixes_by_len[snippet_len] = prefix
        return {
            "doc_id": doc_id,
            "row_index": row_index,
            "end_pos": end_idx,
            "suffix_token_ids": suffix,
            "prefixes_by_len": prefixes_by_len,
        }

    logging.info("Randomly sample anchors from the input file.")
    anchors: List[Dict] = []
    valid_seen = 0
    for i, line in enumerate(read_file(args.input_file)):
        sample = json.loads(line.strip())
        metadata = sample.get("metadata") or {}
        meta = sample.get("meta") or metadata.get("meta") or {}
        archive_member = (
            sample.get("doc_id")
            or metadata.get("id")
            or meta.get("archive_member")
            or meta.get("pid")
            or sample.get("id")
            or f"row:{i}"
        )
        token_ids = _encode_text(sample["text"])
        if len(token_ids) < max_snippet_len:
            continue
        valid_seen += 1
        anchor = _build_anchor(archive_member, i, token_ids)
        if len(anchors) < args.sample_size:
            anchors.append(anchor)
            continue
        replace_idx = rng.randrange(valid_seen)
        if replace_idx < args.sample_size:
            anchors[replace_idx] = anchor

    if len(anchors) < args.sample_size:
        raise ValueError(
            f"Not enough valid samples: {len(anchors)} < {args.sample_size}. "
            "Lower snippet length or provide more data."
        )

    logging.info(
        "Collected %d anchors (total samples: %d).",
        len(anchors),
        len(anchors) * len(args.snippet_length),
    )

    samples = []
    for anchor in anchors:
        for snippet_len in args.snippet_length:
            prefix = anchor["prefixes_by_len"][snippet_len]
            suffix = anchor["suffix_token_ids"]
            samples.append(
                {
                    "doc_id": anchor["doc_id"],
                    "row_index": anchor["row_index"],
                    "end_pos": anchor["end_pos"],
                    "snippet_length": snippet_len,
                    "prefix_token_ids": prefix,
                    "suffix_token_ids": suffix,
                    "prefix": tokenizer.decode(prefix, skip_special_tokens=False),
                    "suffix": tokenizer.decode(suffix, skip_special_tokens=False),
                    "tokenizer": tokenizer_name,
                    "tokenizer_config": {
                        "add_bos": args.add_bos,
                        "add_eos": args.add_eos,
                        "append_eod": args.append_eod,
                        "eod_token_id": args.eod_token_id,
                    },
                    "seed": args.seed,
                }
            )

    logging.info("Save samples to the output file.")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "wt", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

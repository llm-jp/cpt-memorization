#!/usr/bin/env python
import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract memorized samples from completion JSONL files by BLEU threshold, "
            "then count token-level exact suffix matches in a processed corpus."
        )
    )
    parser.add_argument(
        "--completions-root", type=Path, help="Root containing completion JSONL files."
    )
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        help="Completion JSONL files. Overrides --completions-root discovery.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
        help="Root containing processed .jsonl/.jsonl.gz files.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer/model directory used to tokenize processed text.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for intermediate and final CSV outputs.",
    )
    parser.add_argument("--samples-csv", type=Path, help="Intermediate BLEU-filtered samples CSV.")
    parser.add_argument("--output-csv", type=Path, help="Final CSV with exact-match count column.")
    parser.add_argument(
        "--bleu-threshold", type=float, default=0.75, help="BLEU threshold on 0-1 scale."
    )
    parser.add_argument("--workers", default="auto", help="Worker count for the exact-match scan.")
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=20000,
        help="Lines per worker task in the exact-match scan.",
    )
    parser.add_argument(
        "--count-column", default="suffix_dup_count", help="Count column added to the final CSV."
    )
    parser.add_argument(
        "--suffix-token-ids-column",
        default="suffix_token_ids",
        help="CSV token-id column passed to the exact-match scanner.",
    )
    parser.add_argument("--text-key", default="text", help="Processed JSON text field to scan.")
    parser.add_argument(
        "--add-bos", action="store_true", help="Pass --add-bos to the exact-match scanner."
    )
    parser.add_argument(
        "--add-eos", action="store_true", help="Pass --add-eos to the exact-match scanner."
    )
    parser.add_argument(
        "--append-eod", action="store_true", help="Pass --append-eod to the exact-match scanner."
    )
    parser.add_argument(
        "--eod-token-id",
        type=int,
        default=None,
        help="Pass --eod-token-id to the exact-match scanner.",
    )
    return parser.parse_args()


def iter_completion_files(
    completions_root: Optional[Path], input_files: Optional[Sequence[Path]]
) -> List[Path]:
    if input_files:
        files = [path for path in input_files if path.is_file()]
        missing = [str(path) for path in input_files if not path.is_file()]
        if missing:
            raise SystemExit(f"Completion file(s) not found: {', '.join(missing)}")
        return sorted(files)

    if completions_root is None:
        raise SystemExit("Either --input-files or --completions-root is required.")
    if not completions_root.is_dir():
        raise SystemExit(f"Completions root not found: {completions_root}")
    files = sorted(completions_root.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found under: {completions_root}")
    return files


from ..analysis.metrics import token_bleu_score as compute_bleu


def detect_variant(path: Path, completions_root: Optional[Path]) -> str:
    if completions_root is not None:
        try:
            relative = path.relative_to(completions_root)
        except ValueError:
            relative = path
        if len(relative.parts) > 1:
            return relative.parts[0]
    return path.parent.name


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_memorized_samples_csv(
    completion_files: Sequence[Path],
    completions_root: Optional[Path],
    output_csv: Path,
    bleu_threshold: float,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "source_file",
        "row_index",
        "archive_member",
        "doc_id",
        "snippet_length",
        "bleu",
        "suffix",
        "predicted",
        "suffix_token_ids",
        "predicted_token_ids",
    ]
    count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for completion_file in completion_files:
            variant = detect_variant(completion_file, completions_root)
            for row_index, sample in enumerate(iter_jsonl(completion_file)):
                suffix_token_ids = sample.get("suffix_token_ids") or []
                predicted_token_ids = sample.get("predicted_token_ids") or []
                bleu = compute_bleu(suffix_token_ids, predicted_token_ids)
                if bleu < bleu_threshold:
                    continue
                writer.writerow(
                    {
                        "variant": variant,
                        "source_file": str(completion_file),
                        "row_index": row_index,
                        "archive_member": sample.get("archive_member", ""),
                        "doc_id": sample.get("doc_id", ""),
                        "snippet_length": sample.get("snippet_length", ""),
                        "bleu": f"{bleu:.12f}",
                        "suffix": sample.get("suffix", ""),
                        "predicted": sample.get("predicted", ""),
                        "suffix_token_ids": json.dumps(suffix_token_ids, ensure_ascii=False),
                        "predicted_token_ids": json.dumps(predicted_token_ids, ensure_ascii=False),
                    }
                )
                count += 1
    return count


def threshold_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    if not 0 <= args.bleu_threshold <= 1:
        raise ValueError("BLEU threshold must be in [0,1]")
    if not args.processed_root.is_dir():
        raise SystemExit(f"Processed root not found: {args.processed_root}")

    completion_files = iter_completion_files(args.completions_root, args.input_files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = threshold_label(args.bleu_threshold)
    samples_csv = args.samples_csv or args.output_dir / f"bleu_ge_{label}_samples.csv"
    output_csv = (
        args.output_csv or args.output_dir / f"bleu_ge_{label}_samples_with_token_exact_count.csv"
    )

    print(f"completion_files={len(completion_files)}", flush=True)
    print(f"samples_csv={samples_csv}", flush=True)
    memorized_count = write_memorized_samples_csv(
        completion_files=completion_files,
        completions_root=args.completions_root,
        output_csv=samples_csv,
        bleu_threshold=args.bleu_threshold,
    )
    print(f"memorized_samples={memorized_count}", flush=True)

    command = [
        sys.executable,
        "-m",
        "cpt_memorization",
        "memory-count-exact",
        "--csv-file",
        str(samples_csv),
        "--processed-root",
        str(args.processed_root),
        "--output-csv",
        str(output_csv),
        "--tokenizer",
        args.tokenizer,
        "--suffix-token-ids-column",
        args.suffix_token_ids_column,
        "--count-column",
        args.count_column,
        "--text-key",
        args.text_key,
        "--workers",
        str(args.workers),
        "--batch-lines",
        str(args.batch_lines),
    ]
    if args.add_bos:
        command.append("--add-bos")
    if args.add_eos:
        command.append("--add-eos")
    if args.append_eod:
        command.append("--append-eod")
    if args.eod_token_id is not None:
        command.extend(["--eod-token-id", str(args.eod_token_id)])

    print("exact_match_command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"output_csv={output_csv}", flush=True)


if __name__ == "__main__":
    main()

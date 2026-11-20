import argparse
import json
import logging
import math
from collections import defaultdict
from typing import Dict, Tuple

from .approximate_bleu import compute_bleu


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report exact/approx memorization metrics")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the completions JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="Path to save the metrics JSONL report.",
    )
    parser.add_argument(
        "--bleu-threshold",
        type=float,
        default=0.75,
        help="BLEU threshold on 0-1 scale (default: 0.75).",
    )
    args = parser.parse_args()
    if not 0 <= args.bleu_threshold <= 1:
        parser.error("BLEU threshold must be on the 0-1 scale; use 0.75, not 75")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    stats = defaultdict(lambda: {"total": 0, "exact": 0, "approx": 0})

    with open(args.input_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            sl = sample["snippet_length"]
            suffix_token_ids = sample["suffix_token_ids"]
            predicted_token_ids = sample["predicted_token_ids"]

            stats[sl]["total"] += 1
            if suffix_token_ids == predicted_token_ids:
                stats[sl]["exact"] += 1

            bleu = compute_bleu(suffix_token_ids, predicted_token_ids)
            if bleu >= args.bleu_threshold:
                stats[sl]["approx"] += 1

    report = []
    for sl in sorted(stats.keys()):
        total = stats[sl]["total"]
        exact = stats[sl]["exact"]
        approx = stats[sl]["approx"]
        exact_rate = exact / total if total else 0.0
        approx_rate = approx / total if total else 0.0
        exact_ci = wilson_interval(exact, total)
        approx_ci = wilson_interval(approx, total)
        entry: Dict[str, float] = {
            "snippet_length": sl,
            "total": total,
            "exact": exact,
            "exact_rate": exact_rate,
            "exact_ci_low": exact_ci[0],
            "exact_ci_high": exact_ci[1],
            "approx": approx,
            "approx_rate": approx_rate,
            "approx_ci_low": approx_ci[0],
            "approx_ci_high": approx_ci[1],
            "bleu_threshold": args.bleu_threshold,
        }
        report.append(entry)
        logging.info(
            "snippet_length=%s exact %d/%d (%.2f%%, CI %.2f%%-%.2f%%) "
            "approx %d/%d (%.2f%%, CI %.2f%%-%.2f%%) BLEU>=%.2f",
            sl,
            exact,
            total,
            exact_rate * 100.0,
            exact_ci[0] * 100.0,
            exact_ci[1] * 100.0,
            approx,
            total,
            approx_rate * 100.0,
            approx_ci[0] * 100.0,
            approx_ci[1] * 100.0,
            args.bleu_threshold,
        )

    if args.output_file:
        with open(args.output_file, "wt", encoding="utf-8") as fout:
            for entry in report:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

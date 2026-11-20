import argparse
import logging
import json


def main():
    parser = argparse.ArgumentParser(description="Exact Memorization Analysis")
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="Path to the result file.",
    )
    args = parser.parse_args()

    logging.info(f"Loading results from {args.result_file}")
    results = []
    with open(args.result_file, "r") as f:
        for line in f:
            results.append(json.loads(line))

    logging.info("Analyzing exact memorization results.")
    seen_ids = set()
    for result in results:
        is_memorized = result["suffix_token_ids"] == result["predicted_token_ids"]
        if is_memorized:
            doc_id = result.get("doc_id") or result.get("archive_member")
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            print(f"**Doc ID**: {doc_id}")
            print(
                f"**Prefix (length={len(result['prefix_token_ids']):,})**: {repr(result['prefix'])}"
            )
            print(
                f"**Suffix (length={len(result['suffix_token_ids']):,})**: {repr(result['suffix'])}"
            )
            print(
                f"**Predicted (length={len(result['predicted_token_ids']):,})**: {repr(result['predicted'])}"
            )
            print("--------------------------")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

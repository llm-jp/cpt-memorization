"""Evaluate document-grounded QA and training-text memorization after continued pretraining."""

import argparse
import importlib
import logging
import sys

COMMANDS = {
    "corpus-merge": "corpus.merge",
    "workflow": "workflow",
    "corpus-stage": "corpus.stage",
    "corpus-dedup-pages": "corpus.dedup.minhash_pages",
    "memory-extract-low-jaccard": "memorization.analysis.extract_low_jaccard",
    "corpus-lz77": "corpus.dedup.lz77_dedup",
    "corpus-audit-summary": "corpus.dedup.summarize_page_audit",
    "corpus-page-stats": "corpus.analysis.booklet_page_count_stats",
    "corpus-page-samples": "corpus.analysis.extract_booklet_page_count_samples",
    "corpus-distribution": "corpus.analysis.calculate_distribution",
    "corpus-count-chars": "corpus.analysis.count_chars",
    "corpus-count-markers": "corpus.analysis.count_geta",
    "corpus-count-records": "corpus.analysis.count_stats",
    "corpus-error-distribution": "corpus.analysis.document_error_distribution",
    "corpus-error-stats": "corpus.analysis.document_error_stats",
    "corpus-numeric-samples": "corpus.analysis.numeric_heavy_samples",
    "corpus-analyze": "corpus.analysis.jsonl_analyzer",
    "corpus-ocr-stats": "corpus.analysis.ocr_error_stats",
    "corpus-noise-stats": "corpus.cleaning.analyze_edge_noise_ratio",
    "corpus-noise-samples": "corpus.cleaning.extract_edge_noise_samples",
    "corpus-removed-samples": "corpus.cleaning.show_removed_samples",
    "corpus-clean-edges": "corpus.cleaning.remove_kodak_edge_pages",
    "corpus-clean-noise": "corpus.cleaning.remove_kodak_noise",
    "corpus-filter": "corpus.cleaning.build_filtered_corpus",
    "memory-count-exact": "memorization.frequency.count_suffix_token_exact_matches",
    "memory-count-completions": "memorization.frequency.count_suffix_exact_matches_in_completions",
    "corpus-extract": "corpus.formatting.ndlocr_extract",
    "corpus-dedup": "corpus.dedup.minhash_dedup",
    "corpus-language": "corpus.ja_extract.extract_japanese",
    "memory-sample": "memorization.pipeline.extract_samples",
    "memory-generate": "memorization.pipeline.generate",
    "memory-shards": "memorization.pipeline.shards",
    "memory-report": "memorization.report_metrics",
    "memory-checkpoints": "memorization.approximate_bleu",
    "memory-paired": "memorization.analysis.report_base_filtered_metrics",
    "memory-sweep": "memorization.sweep",
    "memory-split": "memorization.analysis.base_exclusion_split",
    "memory-extract": "memorization.analysis.extract_memorized_samples",
    "memory-examples": "memorization.analysis.exact_examples",
    "memory-containment": "memorization.analysis.prefix_token_containment",
    "memory-residual": "memorization.residual",
    "memory-jaccard": "memorization.analysis.annotate_prefix_suffix_jaccard",
    "memory-intervals": "memorization.analysis.report_rate_intervals",
    "memory-plot": "memorization.analysis.plot_combined",
    "memory-plot-exclusion": "memorization.analysis.plot_base_exclusion",
    "neardup-queries": "memorization.neardup.build_low_jaccard_queries",
    "neardup-merge": "memorization.neardup.merge_neardup_counts",
    "neardup-summary": "memorization.neardup.summarize",
    "qa-filter": "qa.filter",
    "qa-synthesize": "qa.synthesize",
    "qa-answer": "qa.answer",
    "qa-evaluate": "qa.evaluate",
    "qa-sweep": "qa.sweep",
    "qa-aggregate": "qa.aggregate",
    "qa-compare": "qa.compare",
    "qa-plot": "qa.plot",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="cpt-memorization", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(args[:1]).command
    sys.argv = [f"{parser.prog} {command}", *args[1:]]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    module = importlib.import_module("." + COMMANDS[command], package=__package__)
    result = module.main()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())

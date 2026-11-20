# Workflow compatibility

Run package commands with `cpt-memorization` or `python -m cpt_memorization`. Command
help is available in a CPU environment. The CLI and JSON workflows accept
corpus, model, tokenizer, checkpoint and output locations from the caller.

| Workflow | Commands |
| --- | --- |
| Publication-level OCR, deduplication and language filtering | `corpus-extract`, `corpus-dedup`, `corpus-language` |
| Page-level cleaning, deduplication and PID rules | `corpus-clean-edges`, `corpus-clean-noise`, `corpus-dedup-pages`, `corpus-audit-summary`, `corpus-lz77`, `corpus-filter` |
| Multiple shards and job dependencies | `corpus-stage`, `workflow` |
| Memorization sampling, generation and paired analysis | `memory-sample`, `memory-generate`, `memory-paired`, `memory-sweep` |
| QA construction, answers, judgments and reporting | `qa-filter`, `qa-synthesize`, `qa-answer`, `qa-evaluate`, `qa-sweep`, `qa-aggregate` |

## Corpus processing

Publication MinHash and Japanese-language filtering prepare the corpus for
GovPub-QA. Page-level MinHash, noise analysis and LZ77 are separate exploratory
workflows. Empty OCR page placeholders retain the publication identifier when
no pages survive. The edge-phrase cleaner can remove pages at both ends, while
the noise detector removes at most one edge page per publication per file.

For page-level MinHash, staging preserves shard identity and order. Reading and
rewriting use the same explicit shard list. The rewrite computes booklet sizes
from clustered pages and selects a representative per remaining page cluster.
LZ77 uses exact fingerprints to identify duplicates and measures compressibility
separately. PID-rule paths resolve relative to their configuration, and the
filter intersects the PIDs passing each rule.

## Memorization and QA

Every BLEU threshold uses the 0–1 scale and defaults to 0.75. Scoring uses
unsmoothed effective-order token BLEU, including for short sequences. The
paired reports validate row alignment and exclude BASE-memorized samples from
both numerator and denominator. Exact token scans support a local tokenizer and
preserve a beginning-of-sequence ID of zero. Final-checkpoint selection compares
iteration numbers numerically for each model variant.

QA aggregation checks item identity and compares all runs on their global
intersection of IDs. A malformed model response produces a diagnostic sidecar
and fails the stage by default. The caller can explicitly request exclusion.
See [the data formats](DATA_FORMATS.md) for field names and metric definitions.

## Execution

The workflow runner executes locally by default. Slurm submission is opt-in
and records job IDs, dependencies and logs. The examples show how to configure
resources and checkpoint sequences. See [the workflow guide](REPRODUCING.md)
for commands and external inputs and [the validation record](VALIDATION.md)
for checks performed on synthetic data.

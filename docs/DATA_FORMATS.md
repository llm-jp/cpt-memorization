# Data formats and numerical conventions

Files are UTF-8 JSONL unless stated otherwise. Shared QA/corpus I/O and the
memory sampler and generator input support gzip by the `.gz` suffix. The retained legacy memory
analysis tools expect plain `.jsonl` completion files. Output parent directories
are created by the main workflows.

## Corpus documents

For a new corpus, supply a unique `id` and a string `text` on each line.
`qa-filter` and `memory-sample` both accept this record directly.

```json
{"id":"doc-001","text":"Synthetic document text."}
```

For the GovPub workflow, the NDLOCR adapter also records the publication
identifier and archive member.

```json
{"text":"Synthetic document", "meta":{"pid":"synthetic-1", "archive_member":"synthetic-1.xml.gz"}}
```

The memory sampler also accepts the original DataTrove form with
`metadata.meta.archive_member`, or `doc_id`, `metadata.id`, or top-level `id`.
When no identifier exists, it records the input row index as `row:<index>`.
The sampling source order is therefore part of reproducibility. QA construction
preserves an existing identifier or derives a SHA-256 identifier from the text.
Page-level OCR records include `meta.page_index`; their QA identifier includes
that index. Duplicate selected document/page IDs fail rather than overwrite.

## Memorization samples and completions

```json
{"doc_id":"synthetic-1", "snippet_length":8, "prefix_token_ids":[11,12,13,14], "suffix_token_ids":[21,22,23,24], "predicted_token_ids":[21,22,23,24], "prefix":"synthetic prefix", "suffix":"synthetic reference", "predicted":"synthetic output"}
```

Sampling writes everything except the predicted fields. Generation adds those
fields plus a `generation` object with model, tokenizer, seed, temperature, and
`top_p`. Sampling records the suffix end position, original row index, seed,
tokenizer, and special-token options. A sample of N tokens means prefix length
N−M plus suffix length M. For example, a 1,000-token snippet with a 50-token
suffix has a 950-token prefix.

Generation validates the input in a streaming pass, then reads and generates at
most `--batch-size` rows at a time (default 1024), preserving input order. It
reuses one model instance and replaces the output file only after every batch
succeeds. Completion output remains plain JSONL.

Paired inputs must have the same rows in the same order, with identical document
IDs, snippet lengths, prefix token IDs, and suffix token IDs. Each token field is
a JSON list of integers, the reference is nonempty, and snippet length must equal
prefix plus suffix length. A shifted, shortened, or malformed evaluation fails.

Verbatim means exact token-list equality. Approximate means token BLEU >= 0.75.
BLEU uses clipped n-gram precision, uniform weights up to
`min(4, reference_length, candidate_length)`, brevity penalty, and no smoothing.
An empty sequence or zero precision at any included order scores zero; an
identical three-token sequence scores one. No tokenizer/model download is needed
when computing scores from supplied token IDs.


| Output/interface | Scale |
| --- | --- |
| `memory-paired`, `memory-sweep`, residual/extract/split BLEU threshold | 0–1 |
| All BLEU command thresholds and annotated BLEU scores | 0–1; use 0.75 |
| Paired report rates and Wilson confidence bounds | 0–1 |
| `memorization_rates_prefix<N>.csv` exact/approx rates | Percent (0–100) |
| QA per-run and aggregate accuracy, standard deviation | 0–1 |
| QA and memorization figure y axes | Percent |

BASE exclusion uses the approximate criterion on BASE, and removes those examples
from **both** the numerator and denominator for each target checkpoint. A
non-memorized target example remains in the denominator. `memory-sweep` writes
`paired_metrics.json` and one plot-compatible CSV for each N, including counts and
explicit denominators. Do not compute rates from candidate-only exports.

Wilson intervals are pointwise binomial intervals over evaluation rows, not
corpus-wide memorization estimates. Pooling lengths can correlate examples from
the same document; the paired report states this limitation.

Containment uses clipped multiset counts: `sum(min(count_suffix(g),count_prefix(g)))`
divided by the number of suffix n-grams. The default n is 3 and threshold is 0.5.
This preserves the supplied implementation's handling of repeated n-grams.
The separate Jaccard workflow uses weighted multiset token n-gram Jaccard and
selects the best same-length prefix window; it is a different analysis.

## QA pairs, answers, judgments

```json
{"text":"Synthetic context", "question":"Synthetic question?", "answer":"Synthetic gold", "metadata":{"id":"synthetic-1"}}
```

Answers add `predicted_answer` as a string. Judgments add `is_correct` as a JSON
boolean. Do not use `"true"`, `"false"`, integers, or a missing field. The ID
identifies the whole QA item, not merely its source publication if there are
multiple items per publication. Every evaluation run must have unique IDs.

The judge always receives context and gold answer, including when judging a
closed-book answer. The answering model receives context only in the open-book
prompt. `--seed` configures both vLLM and sampling; `--few-shot-seed` independently
controls demonstration selection in `qa-answer`. No target item is selected as
its own demonstration.

`qa-filter`, `qa-synthesize`, and `qa-evaluate` write an error sidecar and an
argument/count summary. Invalid output fails by default; `--skip-invalid` makes
the exclusion explicit. `qa-answer` preserves the complete generated continuation
for judging and records its model/tokenizer, decoding seed, few-shot seed, and
prompt name in metadata.

## QA run manifest

```json
{"runs":[{"mode":"open","temperature":0.4,"run":1,"model":"AnyModel","label":"optional-unique-label","path":"evaluated.jsonl"}]}
```

All paths are relative to the manifest, unless absolute. The tuple (mode,
temperature, run, model) and optional label must be unique. All files must exist
and be nonempty. Aggregation uses the intersection of IDs across **all** listed
runs, after verifying matching text, question, and gold answer for each retained
ID. The manifest defines the comparison scope; include every intended condition
before comparing results. Missing files, duplicate runs, and an empty intersection
fail. The grid runner emits a manifest automatically.

`accuracy_by_run.csv` contains raw denominators. `accuracy_by_run_common.csv`
contains the common denominator and per-run dropped count. `accuracy_aggregate.csv`
contains arithmetic means and **sample** standard deviations (`ddof=1`) of
common-subset run accuracies. A single run has reported standard deviation zero;
that does not establish zero experimental variance. `summary.json` records
intersection size, exclusions, and conventions.

`qa-compare --correct LABEL ... --incorrect LABEL ...` replaces fixed-model
correctness collectors. `--difference A B C D` exports nonzero item contributions
to `(A−B)−(C−D)` with the shared denominator, replacing the hard-coded two-temperature
collector. Both operate on arbitrary unique manifest labels and write JSONL and
a summary sidecar.

## External near-duplicate exchange

`memory-residual` writes one suffix query per selected example:

```json
{"query_id":0,"group":"memorized","query_type":"suffix","token_ids":[21,22,23,24],"completion_index":17,"doc_id":"synthetic-1"}
```

The external search returns:

```json
{"query_id":0,"near_dup_occurrence_count":8}
```

Query IDs must be unique JSON integers. Counts must be nonnegative JSON integers,
with exactly the same ID set as the queries. `neardup-summary` validates those
invariants and reports count, median, minimum, and maximum for each group. This
interface does not define or implement the external corpus index's similarity
criterion. The older low-Jaccard CSV query/merge workflow is retained separately.

## Page-level corpus and workflow formats

Page records contain `text` and `meta.pid`, `meta.page_index` (zero-based), and
`meta.page_count`. OCR page export retains a single `empty_document: true` row
with PID and empty text if no pages survive. Downstream text filtering may drop
that placeholder. The DataTrove writer places metadata under `metadata`, with
original `meta` nested within it; raw-page cleaners and PID-rule filtering expect
`meta` records, so use the original cleaned page inputs for those operations.

`corpus-audit-summary` exports `pid`, `total_pages`, `removed_pages`, `kept_pages`
and `dup_ratio` (0–1). It retains the last decision per page ID. `corpus-lz77`
exports `compression_ratio = compressed_bytes / original_bytes`; it can exceed
one for short text. The booklet ratio uses concatenated page text within each
input file. Cross-file mode deduplicates those file-local booklets; it does not
reassemble a PID split across files. Merge such shards first if required.

A `corpus-filter --rules-config` file is a JSON list:

```json
[{"path": "pid_duplicates.jsonl", "metric_key": "dup_ratio", "operator": "<", "threshold": 0.8}]
```

Rule paths are relative to the configuration file. All rule pass sets are
intersected; missing metric PIDs and corpus rows without PIDs are excluded and
counted. `AUTO` is allowed only when a unique numeric metric can be inferred.
The six comparison operators are `<`, `<=`, `>`, `>=`, `==`, `!=`. Thresholds in
examples illustrate configuration; they do not establish historical settings.

A workflow contains `workdir` (relative to its JSON file), `output_dir` (relative
to workdir), optional `variables`/`slurm`, and `steps`. A step has a unique `id`,
a package `command`, an `args` array and optional predecessor IDs in `needs`.
`{name}` expands a variable; `{config_dir}`, `{workdir}`, `{python}` are built in.
Arguments are passed as an array without shell evaluation. Wildcards are not
expanded. Local execution follows dependency order and stops on failure. Slurm
submission records job IDs and uses `afterok`; submitted does not mean completed.
Per-step Slurm options override the shared resource settings. The planner rejects
unknown commands, cyclic/unknown dependencies and cyclic variables. Plan JSON,
shell scripts, logs and execution state are written under `output_dir`.

`memory-count-exact` and `memory-count-completions` count every exact token-window
occurrence, including overlaps. Repeated query rows share a scan and receive the
same corpus occurrence count. These counts are **not near-duplicate counts**.
The caller must use the training tokenizer and the original BOS/EOS/EOD settings.

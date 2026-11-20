# Running the evaluations and reproducing the paper

Run from the distribution root. Outputs in these examples are under `artifacts/`;
place externally obtained inputs under `data/` and model assets under `checkpoints/`.
Neither directory is bundled in a distribution. Relative paths in JSON manifests
are resolved relative to the manifest; model/tokenizer identifiers are passed to
vLLM/Transformers as given and can be Hugging Face IDs or local paths relative to
the working directory.

For a new domain, start with cleaned documents in the [JSONL input
format](DATA_FORMATS.md). Give `qa-filter` the document file to select passages,
then run `qa-synthesize` to create the evaluation questions. For memorization,
give `memory-sample` the same corpus and the training tokenizer. Generate
continuations from the same prefixes for the base and adapted checkpoints.
Each sampled document must contain enough tokens for the longest requested
snippet. Adjust the sampling fractions and lengths to your evaluation.

The numbered sections below use the GovPub study as a worked example. It used
Japanese government publications provided by the National Diet Library of Japan.

<a id="environment-setup"></a>

## Environment and CPU check

```bash
bash scripts/check.sh
```

To keep runtime caches and model downloads inside the project:

```bash
mkdir -p .work/tmp .work/cache
export TMPDIR="$PWD/.work/tmp"
export XDG_CACHE_HOME="$PWD/.work/cache"
export UV_CACHE_DIR="$PWD/.work/cache"
export HF_HOME="$PWD/.work/cache/huggingface"
export TIKTOKEN_CACHE_DIR="$PWD/.work/cache/tiktoken"
export VLLM_CACHE_ROOT="$PWD/.work/cache/vllm"
export MPLCONFIGDIR="$PWD/.work/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
```

Install only the extras needed, after the exports above. For a compatible CUDA
host, add `--extra inference`; topic clustering additionally uses `--extra clustering`
and a separately available SentenceTransformer model.

CPU tests of the optional corpus dependencies use synthetic input and a tiny
fixed-weight language classifier, not the paper's language-identification model:

```bash
uv sync --locked --extra plots --extra corpus --extra analysis --extra tokenizers
.venv/bin/python scripts/check_corpus.py
.venv/bin/python scripts/check_dev.py
```

<a id="corpus-preprocessing"></a>

## 1. Corpus construction (Section 3.2)

The GovPub workflow uses the NDLOCR adapter to extract one record per
publication, deduplicates at the publication level, and retains Japanese
documents with fastText. Obtain the NDLOCR archive and `lid.176.bin`
separately. For a corpus in another language or source format, complete the
preprocessing required for that corpus and pass its document JSONL directly
to the QA or memorization commands.

```bash
.venv/bin/cpt-memorization corpus-extract \
  data/publications.tar artifacts/corpus/formatted.jsonl
.venv/bin/cpt-memorization corpus-dedup \
  artifacts/corpus/formatted.jsonl artifacts/corpus/dedup --no-slurm --workers 8
.venv/bin/cpt-memorization corpus-language \
  --input-dir artifacts/corpus/dedup/minhash-5gram-20buckets-10hashes/results/deduplicated_output \
  --output-dir artifacts/corpus/japanese --model-path data/lid.176.bin --workers 8
```

Language filtering supports both `.jsonl` and `.jsonl.gz`, retaining the shard
names and compression. MinHash uses Japanese word tokenization, 5-grams, 20
buckets, 10 hashes per bucket, and 64-bit signatures. Slurm is enabled explicitly
with `--use-slurm`; local execution is the default. Reuse the exact same input and
task partition for the signature and filtering stages. `--force-clean` explicitly
removes that configuration's previous work directory.

`corpus-extract --emit-pages` is an alternative page-level export, not the
publication-level corpus described in the paper. The supplied code only covers
the NDLOCR archive format. If an additional OCR source uses a different format,
convert it to the [document schema](DATA_FORMATS.md) before continuing.

## 2. Continued pretraining (Section 3.3–3.4; external)

Use an external Megatron-LM installation and the original base checkpoint.
The paper initializes from the end of the base model's first pretraining phase;
the public final checkpoint name alone does not identify that initialization.
Supply the training launcher, corpus mixing and checkpoint conversion tools,
and the general-domain corpus separately.

The paper reports approximately 13.7B GovPub tokens, GovOnly trained on ten
GovPub epochs and GovMix on one epoch of a 1:9 GovPub/general-token mixture,
about 137B tokens each. Both use sequence length 4096, global batch size 512,
65,000 steps, and constant learning rate 3e-4. Keep intermediate converted
checkpoints and the compatible tokenizer for analysis. These settings are
recorded for provenance, not supplied as a substitute training implementation.

<a id="memorization"></a>

## 3. Memorization (Section 5 and Appendix B)

Use the same tokenizer, sampled anchors, and token IDs for every model. Supply
an ordered corpus shard/merged corpus prepared for the original evaluation.
Sampling independently per model invalidates the comparison.

```bash
.venv/bin/cpt-memorization memory-sample \
  --model checkpoints/base --tokenizer checkpoints/tokenizer \
  --input-file data/evaluation_corpus.jsonl.gz --output-file artifacts/memory/samples.jsonl \
  --sample-size 9890 --snippet-length 200 400 600 800 1000 --suffix-length 50 --seed 42
.venv/bin/cpt-memorization memory-generate \
  --model checkpoints/base --tokenizer checkpoints/tokenizer \
  --input-file artifacts/memory/samples.jsonl --output-file artifacts/memory/base.jsonl \
  --temperature 0 --top-p 1 --seed 42
.venv/bin/cpt-memorization memory-generate \
  --model checkpoints/mixed --tokenizer checkpoints/tokenizer \
  --input-file artifacts/memory/samples.jsonl --output-file artifacts/memory/mixed-final.jsonl \
  --temperature 0 --top-p 1 --seed 42
.venv/bin/cpt-memorization memory-paired \
  --base-jsonl artifacts/memory/base.jsonl --input-jsonl artifacts/memory/mixed-final.jsonl \
  --output-json artifacts/memory/paired.json --output-csv artifacts/memory/paired.csv \
  --bleu-threshold 0.75 --confidence 0.95 --workers 1
```

The exact sampled anchors and special-token settings of the historical run are
not in the source distribution. To recover historical numbers, use the original
`samples.jsonl` and completion files. `--add-bos`, `--add-eos`, `--append-eod`, and
`--eod-token-id` are available for explicitly matching the original preprocessing;
this document does not infer those settings from the checkpoint name.

For larger runs, set `--sample-size 100000` to select 100,000 eligible documents
and produce 500,000 rows across the five snippet lengths. Short documents are
excluded before sampling; this does not force the historical count of 9,890.
`memory-generate --batch-size 1024` bounds the number of rows held in memory
during inference. Sampling still reads and tokenizes the complete corpus and
retains the selected anchors, so schedule it as preprocessing on a compute node.

Repeat generation for every checkpoint and create a memory manifest. A minimal
example, with paths relative to the manifest, is:

```json
{
  "base_file": "base.jsonl",
  "runs": [
    {"variant": "GovOnly", "tokens_B": 136.31488, "path": "domain-final.jsonl"},
    {"variant": "GovMix", "tokens_B": 136.31488, "path": "mixed-final.jsonl"}
  ]
}
```

Use actual incremental training tokens for each checkpoint; `65000 * 512 * 4096`
is 136.31488B tokens. Include BASE at `tokens_B: 0` for each variant if the figure
should start there. The BASE-filtered BASE rates are zero by definition.

```bash
.venv/bin/cpt-memorization memory-sweep \
  --manifest artifacts/memory/manifest.json --output-dir artifacts/memory/plots
.venv/bin/cpt-memorization memory-plot \
  --plots-dir artifacts/memory/plots --output-file artifacts/memory/figure6.svg \
  --prefix-lengths 1000 --panel-layout combined --color-scheme corpus-prefix
.venv/bin/cpt-memorization memory-plot \
  --plots-dir artifacts/memory/plots --output-file artifacts/memory/appendix-b.svg
```

`memory-sweep` takes **full paired completions**, including non-memorized rows.
Do not feed a candidate-only extract to a rate calculation. `memory-split` exports
candidate-only subsets for inspection, and `memory-extract` exports candidate CSVs.
All BLEU thresholds now use **0–1** (`--bleu-threshold 0.75`), including
`memory-report` and `memory-checkpoints`. A value of 75 is rejected. All paths use
unsmoothed effective-order token BLEU. The plotting CSVs express rates
as percentages, as specified in [DATA_FORMATS.md](DATA_FORMATS.md).

## 4. Residual memorization (Section 5.4)

This selects BASE-filtered, target-memorized examples with suffix-in-prefix
3-gram containment below 0.5, and samples 300 non-memorized low-containment
examples for comparison:

```bash
.venv/bin/cpt-memorization memory-residual \
  --base-jsonl artifacts/memory/base.jsonl --input-jsonl artifacts/memory/mixed-final.jsonl \
  --output-dir artifacts/memory/residual --snippet-length 1000 --bleu-threshold 0.75 \
  --ngram-size 3 --containment-threshold 0.5 --control-size 300 --seed 42
```

It writes `residual.jsonl`, `control.jsonl`, `query.jsonl`, and a selection summary.
An external near-duplicate index must search the training corpus for each query's
suffix token IDs. Return exactly one count per `query_id` using the schema in
[DATA_FORMATS.md](DATA_FORMATS.md). The index implementation and historical counts
are external inputs. Run both groups with the same search
criterion and document whether the queried source occurrence is included.

```bash
.venv/bin/cpt-memorization neardup-summary \
  --queries-jsonl artifacts/memory/residual/query.jsonl \
  --counts-jsonl artifacts/memory/residual/external_counts.jsonl \
  --output-json artifacts/memory/residual/count_summary.json
```

The older `memory-jaccard`, `neardup-queries`, and `neardup-merge` commands preserve
the separate low-Jaccard CSV workflow. Low-Jaccard selection does not implement
the paper's low-containment selection; use `memory-residual` for Section 5.4.

<a id="govpub-qa"></a>

## 5. GovPub-QA construction and evaluation (Section 4.1–4.2)

Merge the retained Japanese publication shards in a stable order. This ensures
QA construction consumes the processed GovPub corpus. These commands run GPT-OSS
through local vLLM, without API credentials:

```bash
.venv/bin/cpt-memorization corpus-merge \
  --inputs artifacts/corpus/japanese --output artifacts/corpus/govpub.jsonl.gz
.venv/bin/cpt-memorization qa-filter \
  --input-file artifacts/corpus/govpub.jsonl.gz --output-file artifacts/qa/filtered.jsonl \
  --model-name openai/gpt-oss-120b --tensor-parallel-size 8 \
  --fraction-to-keep 0.01 --max-text-length 1000 --seed 42
.venv/bin/cpt-memorization qa-synthesize \
  --input-file artifacts/qa/filtered.jsonl --output-file artifacts/qa/synthesized.jsonl \
  --model-name openai/gpt-oss-120b --tensor-parallel-size 8 --seed 42
.venv/bin/cpt-memorization qa-answer \
  --input-file artifacts/qa/synthesized.jsonl --output-file artifacts/qa/output-base.jsonl \
  --model-name checkpoints/base --tokenizer checkpoints/tokenizer \
  --prompt-file open --num-shots 3 --temperature 0 --seed 42
.venv/bin/cpt-memorization qa-evaluate \
  --input-file artifacts/qa/output-base.jsonl --output-file artifacts/qa/evaluated-base.jsonl \
  --model-name openai/gpt-oss-120b --tensor-parallel-size 8 --seed 42
```

`--prompt-file closed` answers without document context. Demonstrations exclude
the target example. The entire answer continuation is passed to the judge, as in
the source code. All five Appendix A prompts are bundled in the wheel, and a
custom prompt YAML path can be supplied. For non-GPT-OSS chat models, use JSON
structured outputs (`--response-format json`); GPT-OSS Harmony parsing is lazy.
Only GPT-OSS receives the `reasoning_effort` chat-template option.

Invalid structured responses fail the stage after writing `.errors.jsonl` and
`.summary.json`; `--skip-invalid` explicitly permits dropping them and records
the count. Reconcile missing judgments through a common-ID comparison, not by
implicitly treating missing or string-valued judgments as false.

Edit [examples/qa_sweep.json](../examples/qa_sweep.json) with compatible model and
tokenizer locations, input QA file, judge settings, seeds, and output directory.
The configuration follows the available historical artifact grid: one
zero-temperature run and three runs per nonzero temperature.

```bash
.venv/bin/cpt-memorization qa-sweep --config examples/qa_sweep.json --dry-run
.venv/bin/cpt-memorization qa-sweep --config examples/qa_sweep.json
.venv/bin/cpt-memorization qa-aggregate \
  --manifest artifacts/qa-sweep/manifest.json --output-dir artifacts/qa-sweep/summary
.venv/bin/cpt-memorization qa-plot \
  --input-file artifacts/qa-sweep/summary/accuracy_aggregate.csv \
  --output-file artifacts/qa-sweep/summary/figure3.svg
```

The runner writes a complete command plan and evaluated-file manifest and stops
on command failure. Each process releases its model before the next starts.
Existing outputs require `--overwrite` or a new output directory. `greedy_runs: 3`
uses three seeds at zero temperature as well, matching the paper's prose;
[examples/qa_sweep_paper.json](../examples/qa_sweep_paper.json) supplies that
108-run / 216-command configuration; the
supplied final data has only one greedy run. `few_shot_seed` can fix demonstrations
while varying `seed` for decoding. Neither choice is inferred from historical
outputs; set it deliberately for a new experiment.

To import existing judgments, provide the parent of folders named
`<mode>_temp<temperature>_run<repeat>` and map each
`evaluated_<suffix>.jsonl` to a display label. For example,

```bash
.venv/bin/python scripts/import_qa_runs.py \
  --results-root data/qa_runs --output-dir artifacts/imported-qa \
  --models base=BASE domain=GovOnly mixed=GovMix
```

This reads the source files in place and writes only to the selected output
location. On the supplied corrected data, the result is 96 runs, 2,328 common
IDs, and 36 aggregate rows. The context-free summary is included under
[docs/results](results). The original generated/judged QA files are not bundled.

## 6. General-domain evaluation (Section 4.3; external)

The Figure 4 evaluation requires llm-jp-eval, its benchmark versions and
configuration, and the training checkpoints. Obtain these assets separately. This distribution makes
no claim to reproduce Figure 4 or the historical training runs.

## 7. Page-level corpus and analysis tools

These exploratory workflows operate on individual pages. Section 1 describes
the paper's publication-level preprocessing. Run the portable CPU demo.

```bash
.venv/bin/cpt-memorization workflow --config examples/corpus_demo_workflow.json --dry-run
.venv/bin/cpt-memorization workflow --config examples/corpus_demo_workflow.json
```

It cleans edge pages, exports LZ77 ratios and applies a PID rule. To rerun, use a
new output path or explicitly `--overwrite`. `corpus-clean-edges` removes first/last
pages containing the configured Kodak phrase; `corpus-clean-noise` follows the
source's broader pattern rules and removes at most one noisy edge page per PID
per input file. Neither deletes every matching line from a document.

For page-level MinHash, first export pages with `corpus-extract --emit-pages`,
then edit [corpus_dev_workflow.json](../examples/corpus_dev_workflow.json):

```bash
.venv/bin/cpt-memorization workflow --config examples/corpus_dev_workflow.json --dry-run
.venv/bin/cpt-memorization workflow --config examples/corpus_dev_workflow.json
```

`corpus-stage` retains colliding shard basenames under numbered subdirectories.
The page pipeline includes signatures, buckets, clusters, booklet rewrite (stage
3.5), filtering and optional audit/output writing. The rewrite selects booklet
representatives using revision/date and duplicate-page counts, then retains one
representative per remaining page cluster; it can still drop a shared page from
an otherwise retained booklet. It derives booklet sizes from clustered pages.
The audit example produces PID ratios over page decisions and filters original
page records. Edit all source lists and rule thresholds deliberately.

| Analysis purpose | Commands |
| --- | --- |
| Page-count distributions and samples | `corpus-page-stats`, `corpus-page-samples` |
| Record, character, marker counts | `corpus-count-records`, `corpus-count-chars`, `corpus-count-markers`, `corpus-distribution` |
| Per-PID OCR errors | `corpus-error-stats`, `corpus-error-distribution`, `corpus-ocr-stats` |
| Numeric-heavy examples | `corpus-numeric-samples` |
| Edge-noise analysis | `corpus-noise-stats`, `corpus-noise-samples`, `corpus-removed-samples` |
| Streaming length statistics, clustering, PII | `corpus-analyze stats`, `corpus-analyze cluster`, `corpus-analyze pii` |

Use command-specific `--help` for input schemas/options. Some retained analysis
commands accept plain JSONL only; OCR stats and the general analyzer also accept
compressed input. `corpus-analyze stats --model characters` performs explicit
character counts without tokenizer downloads. Other tokenizer modes may use the
source's approximate fallback; `stats.json` records `tokenizer_mode`. PII results
are pattern-based candidate counts; optional spaCy models must be installed
separately. Clustering uses SentenceTransformer → UMAP → HDBSCAN, with model
weights and hardware supplied by the caller.

## 8. Configurable checkpoint jobs and exact occurrence counting

Edit [memory_workflow.json](../examples/memory_workflow.json) with your corpus,
model/checkpoint paths, resource settings and steps. Add one generation step per
checkpoint and make downstream reports depend on those step IDs.

```bash
.venv/bin/cpt-memorization workflow --config examples/memory_workflow.json --dry-run
.venv/bin/cpt-memorization workflow --config examples/memory_workflow.json --executor slurm
.venv/bin/cpt-memorization memory-extract-low-jaccard \
  --completions-root artifacts/memory-workflow/completions --final-only \
  --output-csv artifacts/memory-workflow/final-low-jaccard.csv --workers 1
.venv/bin/cpt-memorization memory-count-completions \
  --completions-root artifacts/memory-workflow/completions \
  --processed-root artifacts/corpus/japanese --tokenizer checkpoints/tokenizer \
  --output-dir artifacts/exact-counts --workers 1 --bleu-threshold 0.75
```

The final-file selector uses the highest numeric `iter_<number>.jsonl` within
each immediate variant directory, with no fixed variant names. The exact-count
wrapper selects all BLEU-qualified completions by default. For an existing CSV,
use `memory-count-exact --csv-file ... --processed-root ... --tokenizer ...`.
Exact token occurrences do not replace §5.4's approximate near-duplicate search.

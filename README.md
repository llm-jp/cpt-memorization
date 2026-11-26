# Evaluating Task Performance and Memorization in Continued Pretraining

How does continued pretraining on a domain corpus change a language model's
ability to answer questions about those documents? How often can it reproduce
a training passage when given its prefix? This package compares a base model
and its adapted checkpoints on both measures.

| Evaluation | What the tools do |
| --- | --- |
| Document-grounded question answering | Select passages that support a question, generate a question and reference answer for each, and score each checkpoint on the same questions. Compare accuracy with the passage provided (open book) and withheld (closed book). |
| Memorization | Sample a prefix and reference continuation from a training document. Compare each checkpoint's generated continuation with the reference using exact token match and token BLEU. Paired reports can exclude passages the base model already reproduced. |

## Use your own corpus

Prepare one JSON object per document with a unique `id` and its `text`. The
`qa-filter` and `memory-sample` commands read this format directly. Use the
same sampled items and a compatible tokenizer across checkpoints. For question
answering, specify the synthesis and judging models separately from the model
being evaluated. See the [input formats](docs/DATA_FORMATS.md) and
[workflow commands](docs/REPRODUCING.md).

```json
{"id":"doc-001","text":"Inspections take place every six months. A written report follows each inspection."}
```

## Try the analyses on a CPU

From the repository root, install the locked dependencies with Python 3.12 and
uv. These examples use bundled synthetic judgments and completions. They write
QA accuracy by run and paired memorization rates under `artifacts/`.

```bash
uv sync --locked --extra plots
.venv/bin/cpt-memorization qa-aggregate \
  --manifest examples/qa_manifest.json --output-dir artifacts/demo-qa
.venv/bin/cpt-memorization memory-sweep \
  --manifest examples/memory_manifest.json --output-dir artifacts/demo-memory
```

For corpus processing, install the `corpus` extra. GPU inference uses the
`inference` extra on a compatible CUDA host. The guide includes installation
commands, optional extras and cache settings. [Environment setup](docs/REPRODUCING.md#environment-setup)

## Paper and reproduction

**Analyzing Task Performance and Memorization Risk in Continued Pretraining of LLMs**
was accepted to the EACL 2027 Main Conference. The paper studies continued
pretraining on GovPub, a corpus of Japanese government publications provided by
the National Diet Library of Japan. GovPub-QA uses questions synthesized from
documents in that corpus. The repository includes the construction scripts and
aggregate QA results. The agreement with the National Diet Library restricts
public release of the corpus.

| Paper workflow | Implementation | Reproduction guide |
| --- | --- | --- |
| GovPub preprocessing | [Corpus scripts](src/cpt_memorization/corpus) | [Extract NDLOCR publication records, deduplicate them and retain Japanese text](docs/REPRODUCING.md#corpus-preprocessing) |
| GovPub-QA construction | [QA scripts and prompts](src/cpt_memorization/qa) | [Select eligible passages and generate question–answer pairs](docs/REPRODUCING.md#govpub-qa) |
| Memorization analysis | [Sampling, generation and scoring](src/cpt_memorization/memorization) | [Reproduce the prefix–suffix comparison](docs/REPRODUCING.md#memorization) |

The [paper mapping](docs/PAPER.md) identifies the inputs and external tools needed
for each experiment. The [QA tables](docs/results) give the results on questions
shared across all evaluated runs. The [validation record](docs/VALIDATION.md)
describes the CPU tests and the [compatibility notes](docs/MIGRATION.md)
describe the behavior of the workflow runner and metrics. The original project code is licensed under the [MIT License](LICENSE).
See [license details](LICENSE_STATUS.md) and [third-party notices](THIRD_PARTY_NOTICES.md)
for the Apache-2.0 MinHash code and separate terms for external materials.

To run the CPU regression tests, command checks, and distribution checks, use
Git and uv from the repository root.

```bash
bash scripts/check.sh
```

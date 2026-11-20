# Validation record

## CPU and package checks

| Check | Recorded result |
| --- | --- |
| Regression suite | Covered effective-order BLEU, invalid thresholds, paired denominators, empty OCR placeholders, PID rules, cross-file LZ77 deduplication, overlapping exact counts, job ordering and QA validation |
| CLI | All command help pages loaded without vLLM |
| Workflow | The synthetic corpus passed through cleaning, LZ77 ratio export and PID filtering. The memory and page-corpus workflows produced dependency plans |
| Publication preprocessing | Three synthetic OCR publications became two after MinHash. A fixed-weight fastText fixture retained the Japanese test record and dropped the English record |
| Page-level integration | Two shards with nine pages yielded five retained pages. Nine decisions were recorded. PID ratios, streaming statistics, synthetic contact detection, gzip merge and overlapping token counts were exercised |
| QA plan | The recorded sweep has 96 runs and 192 inference commands. The optional three-trials-per-temperature configuration has 108 runs and 216 commands |
| QA aggregation | Across 96 runs, 224,622 judgments produced 2,328 common IDs and 36 aggregate rows. The maximum absolute difference from the rounded mean and sample SD was under 4.84e-11 |
| Distribution | The checks build a wheel and source archive, validate their contents, load prompts from the wheel and run tests from the extracted source archive |

Run the base checks from the repository root.

```bash
bash scripts/check.sh
```

For the optional integration checks, install the extra dependencies first.
The MinHash multiprocessing manager needs local socket access. The synthetic
fastText fixture exercises model serialization, loading and prediction with
fixed weights.

```bash
uv sync --locked --extra plots --extra corpus --extra analysis --extra tokenizers
.venv/bin/python scripts/check_corpus.py
.venv/bin/python scripts/check_dev.py
```

## Scope of verification

Inference tests use deterministic stubs, and Slurm tests use a mock scheduler.
Checks with these fixtures cover the data and orchestration logic. Reproducing
the original outputs requires corpus-scale processing, GPU inference and the
original inputs. The near-duplicate index, CPT and general-domain benchmarks
are external. See [the workflow guide](REPRODUCING.md) for the inputs and
[the paper mapping](PAPER.md) for the scope of each result.

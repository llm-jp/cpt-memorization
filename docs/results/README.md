# Recomputed QA summaries

These CSVs summarize counts and accuracies from the GovPub-QA temperature
sweep. The tables were recomputed with `qa-aggregate`.

- `accuracy_by_run.csv`: 96 model-runs before common-ID filtering.
- `accuracy_by_run_common.csv`: the same runs on 2,328 shared IDs.
- `accuracy_aggregate.csv`: 36 mode/temperature/model groups, mean and sample SD.
- `qa_summary.json`: dataset sizes and numerical agreement audit.

The tables identify models as BASE, GovOnly, and GovMix. Mode and temperature
are explicit columns. Accuracy and SD use the 0–1 scale. Zero temperature has one run, other temperatures have three.
The maximum absolute difference from the original rounded mean/SD table is
4.84e-11. See [the reproduction commands](../REPRODUCING.md) for recomputation from
external original JSONL files, and [the paper mapping](../PAPER.md) for limits.

Regenerate a figure without the original data:

```bash
.venv/bin/cpt-memorization qa-plot \
  --input-file docs/results/accuracy_aggregate.csv \
  --output-file artifacts/paper/qa_accuracy.svg
```

[The regenerated QA figure](qa_accuracy.svg) shows mean accuracy with run-to-run
sample-standard-deviation error bars. These are not confidence intervals.

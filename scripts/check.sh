#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
mkdir -p .work/tmp .work/cache .work/matplotlib
export TMPDIR="$task_root/.work/tmp"
export UV_CACHE_DIR="$task_root/.work/cache"
export UV_PYTHON_DOWNLOADS=never
export MPLCONFIGDIR="$task_root/.work/matplotlib"
export XDG_CACHE_HOME="$task_root/.work/cache"
export PYTHONDONTWRITEBYTECODE=1
uv sync --locked --extra plots
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_cli.py
.venv/bin/cpt-memorization workflow --config examples/corpus_demo_workflow.json --overwrite
.venv/bin/cpt-memorization workflow --config examples/memory_workflow.json --dry-run > .work/memory-plan.txt
.venv/bin/cpt-memorization workflow --config examples/corpus_dev_workflow.json --dry-run > .work/corpus-plan.txt
.venv/bin/cpt-memorization memory-sweep --manifest examples/memory_manifest.json --output-dir artifacts/demo-memory
.venv/bin/cpt-memorization memory-residual --base-jsonl examples/fixtures/base_completions.jsonl --input-jsonl examples/fixtures/domain_completions.jsonl --output-dir artifacts/demo-residual --snippet-length 12 --control-size 2
.venv/bin/cpt-memorization qa-aggregate --manifest examples/qa_manifest.json --output-dir artifacts/demo-qa
.venv/bin/cpt-memorization qa-plot --input-file artifacts/demo-qa/accuracy_aggregate.csv --output-file artifacts/demo-qa/accuracy.svg
.venv/bin/cpt-memorization memory-plot --plots-dir artifacts/demo-memory --output-file artifacts/demo-memory/memorization.svg --prefix-lengths 12 --panel-layout combined
.venv/bin/cpt-memorization qa-sweep --config examples/qa_sweep.json --dry-run > .work/qa-plan.txt
.venv/bin/cpt-memorization qa-sweep --config examples/qa_sweep_paper.json --dry-run > .work/qa-paper-plan.txt
uv build --out-dir dist
.venv/bin/python scripts/verify_release.py

"""Regression checks for the paper's evaluation definitions and input invariants."""

import csv
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from cpt_memorization.io import read_jsonl, write_json, write_jsonl
from cpt_memorization.memorization.analysis.metrics import token_bleu_score
from cpt_memorization.memorization.analysis.prefix_token_containment import containment_metrics
from cpt_memorization.memorization.analysis.report_base_filtered_metrics import (
    paired_tasks,
    binomial_summary,
)
from cpt_memorization.memorization.approximate_bleu import _summarize_checkpoint_file
from cpt_memorization.memorization.sweep import aggregate as memory_aggregate
from cpt_memorization.qa.aggregate import aggregate as qa_aggregate
from cpt_memorization.qa.common import answer_prompts, load_prompt, load_records, parse_response
from cpt_memorization.qa.sweep import make_plan

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".work/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".work/tmp")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        warnings.filterwarnings("ignore", category=UserWarning, module="nltk.translate.bleu_score")

    def test_paired_denominator_keeps_non_memorized_examples(self):
        report = memory_aggregate(ROOT / "examples/memory_manifest.json", self.work)
        domain = report["runs"][0]["groups"][12]
        self.assertEqual(domain["total"], 4)
        self.assertEqual(domain["base_excluded"]["count"], 1)
        self.assertEqual(domain["retained"], 3)
        self.assertEqual(domain["target_after_filter"]["count"], 1)
        self.assertAlmostEqual(domain["target_after_filter"]["rate"], 1 / 3)
        with (self.work / "memorization_rates_prefix12.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(rows[0]["approx_rate"]), 100 / 3)

    def test_token_bleu_cannot_be_replaced_by_text_bleu(self):
        row = dict(
            snippet_length=12,
            suffix_token_ids=[1, 2, 3, 4, 5, 6],
            predicted_token_ids=[10, 20, 30, 40, 50, 60],
            suffix="identical decoded text",
            predicted="identical decoded text",
        )
        path = self.work / "completions.jsonl"
        write_jsonl(path, [row])
        summary = _summarize_checkpoint_file(path, 12, 0.75)
        self.assertEqual(summary["approx"], 0)
        self.assertEqual(token_bleu_score([1, 2, 3, 4], [1, 2, 3, 4]), 1)
        self.assertEqual(token_bleu_score([1, 2, 3, 4], []), 0)

    def test_all_pairs_are_validated_even_when_not_memorized(self):
        base = list(read_jsonl(ROOT / "examples/fixtures/base_completions.jsonl"))
        target = [{**r} for r in base]
        target[-1]["prefix_token_ids"] = [99] * 6
        write_jsonl(self.work / "target.jsonl", target)
        with self.assertRaisesRegex(ValueError, "mismatched prefix_token_ids"):
            list(
                paired_tasks(
                    ROOT / "examples/fixtures/base_completions.jsonl", self.work / "target.jsonl"
                )
            )
        write_jsonl(self.work / "target.jsonl", base[:-1])
        with self.assertRaisesRegex(ValueError, "Different input lengths"):
            list(
                paired_tasks(
                    ROOT / "examples/fixtures/base_completions.jsonl", self.work / "target.jsonl"
                )
            )

    def test_containment_uses_clipped_multiset_counts(self):
        result = containment_metrics([1, 2, 3, 9], [1, 2, 3, 1, 2, 3], 3, 0.5)
        self.assertEqual(result["ngram_shared_count"], 1)
        self.assertEqual(result["ngram_total_count"], 4)
        self.assertEqual(result["ngram_containment"], 0.25)
        self.assertFalse(result["ngram_contained"])
        self.assertTrue(containment_metrics([1, 2, 3, 4], [2, 3, 4], 3, 0.5)["exact"])

    def test_wilson_zero_and_empty_denominators(self):
        self.assertIsNone(binomial_summary(0, 0, 0.95)["rate"])
        result = binomial_summary(0, 10, 0.95)
        self.assertEqual(result["ci_low"], 0)
        self.assertAlmostEqual(result["ci_high"], 0.2775328, places=6)

    def test_qa_global_intersection_and_std(self):
        report = qa_aggregate(ROOT / "examples/qa_manifest.json", self.work)
        self.assertEqual((report["runs"], report["union_ids"], report["common_ids"]), (4, 4, 3))
        with (self.work / "accuracy_aggregate.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        values = {r["model"]: r for r in rows}
        self.assertAlmostEqual(float(values["base"]["accuracy_mean"]), 2 / 3)
        self.assertEqual(float(values["adapted"]["accuracy_mean"]), 1)
        self.assertEqual(float(values["base"]["accuracy_std"]), 0)

    def test_duplicate_ids_and_string_boolean_are_rejected(self):
        record = {"metadata": {"id": "a"}, "is_correct": "false"}
        write_jsonl(self.work / "bad.jsonl", [record])
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            load_records(self.work / "bad.jsonl", evaluated=True)
        write_jsonl(self.work / "bad.jsonl", [record, record])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_records(self.work / "bad.jsonl")

    def test_cross_run_question_mismatch_rejected(self):
        rows = list(read_jsonl(ROOT / "examples/fixtures/qa_base_1.jsonl"))
        rows[0]["question"] = "different question with same ID"
        write_jsonl(self.work / "other.jsonl", rows)
        manifest = {
            "runs": [
                dict(
                    mode="open",
                    temperature=0,
                    run=1,
                    model="a",
                    path=str(ROOT / "examples/fixtures/qa_base_1.jsonl"),
                ),
                dict(mode="open", temperature=0, run=1, model="b", path="other.jsonl"),
            ]
        }
        write_json(self.work / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "mismatched question"):
            qa_aggregate(self.work / "manifest.json", self.work / "out")

    def test_closed_book_has_no_source_or_target_gold_leakage(self):
        rows = list(read_jsonl(ROOT / "examples/fixtures/qa.jsonl"))
        for row in rows:
            row["text"] = "PRIVATE_SOURCE_CONTEXT"
        template = load_prompt("closed")["user_prompt"]
        prompts = answer_prompts(rows, template, 3, 42)
        self.assertEqual(prompts, answer_prompts(rows, template, 3, 42))
        for row, prompt in zip(rows, prompts, strict=True):
            self.assertNotIn("PRIVATE_SOURCE_CONTEXT", prompt)
            self.assertNotIn(row["answer"], prompt)
            self.assertEqual(prompt.count(row["question"]), 1)

    def test_grid_matches_released_paper_artifacts(self):
        _, manifest, jobs = make_plan(ROOT / "examples/qa_sweep.json")
        self.assertEqual(len(manifest["runs"]), 96)
        self.assertEqual(len(jobs), 192)
        self.assertEqual(sum(r["temperature"] == 0 for r in manifest["runs"]), 6)
        self.assertTrue(all("--tokenizer" in job["command"] for job in jobs[::2]))

    def test_plain_json_parser_does_not_require_harmony(self):
        from types import SimpleNamespace

        response = SimpleNamespace(outputs=[SimpleNamespace(text='{"is_correct": false}')])
        self.assertIs(parse_response(response, "json")["is_correct"], False)

    def test_paper_prose_grid_has_three_runs_at_every_temperature(self):
        _, manifest, jobs = make_plan(ROOT / "examples/qa_sweep_paper.json")
        self.assertEqual(len(manifest["runs"]), 108)
        self.assertEqual(len(jobs), 216)
        self.assertEqual(sum(r["temperature"] == 0 for r in manifest["runs"]), 18)

    def test_residual_and_control_selection(self):
        from cpt_memorization.memorization.residual import prepare

        fixture = ROOT / "examples/fixtures"
        report = prepare(
            fixture / "base_completions.jsonl",
            fixture / "domain_completions.jsonl",
            self.work,
            snippet_length=12,
            control_size=2,
        )
        self.assertEqual(report["base_excluded"], 1)
        self.assertEqual(report["memorized_after_base"], 1)
        self.assertEqual(report["residual"], 1)
        self.assertEqual(report["control"], 2)
        queries = list(read_jsonl(self.work / "query.jsonl"))
        self.assertEqual(
            [q["group"] for q in queries], ["memorized", "non_memorized", "non_memorized"]
        )
        with self.assertRaisesRegex(ValueError, "eligible control"):
            prepare(
                fixture / "base_completions.jsonl",
                fixture / "domain_completions.jsonl",
                self.work / "bad",
                snippet_length=12,
                control_size=3,
            )

    def test_neardup_count_coverage_and_group_median(self):
        from cpt_memorization.memorization.neardup.summarize import summarize_counts

        queries = self.work / "queries.jsonl"
        counts = self.work / "counts.jsonl"
        write_jsonl(queries, [dict(query_id=0, group="a"), dict(query_id=1, group="a")])
        write_jsonl(
            counts,
            [
                dict(query_id=1, near_dup_occurrence_count=10),
                dict(query_id=0, near_dup_occurrence_count=2),
            ],
        )
        self.assertEqual(summarize_counts(queries, counts)["a"]["median"], 6)
        write_jsonl(counts, [dict(query_id=0, near_dup_occurrence_count=2)])
        with self.assertRaisesRegex(ValueError, "must match exactly"):
            summarize_counts(queries, counts)

    def test_sample_standard_deviation(self):
        rows = list(read_jsonl(ROOT / "examples/fixtures/qa_base_2.jsonl"))
        write_jsonl(self.work / "first.jsonl", rows)
        rows[0]["is_correct"] = False
        write_jsonl(self.work / "second.jsonl", rows)
        write_json(
            self.work / "manifest.json",
            {
                "runs": [
                    dict(mode="open", temperature=0.5, run=run, model="model", path=f"{name}.jsonl")
                    for run, name in [(1, "first"), (2, "second")]
                ]
            },
        )
        qa_aggregate(self.work / "manifest.json", self.work / "aggregate")
        with (self.work / "aggregate/accuracy_aggregate.csv").open() as stream:
            result = next(csv.DictReader(stream))
        self.assertAlmostEqual(float(result["accuracy_mean"]), 0.5)
        self.assertAlmostEqual(float(result["accuracy_std"]), 0.23570226039551584)


if __name__ == "__main__":
    unittest.main()

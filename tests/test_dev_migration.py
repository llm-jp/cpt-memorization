"""Regression tests for corpus processing, memorization metrics, and workflow execution."""

import argparse
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cpt_memorization.io import read_jsonl, write_json, write_jsonl
from cpt_memorization.corpus.analysis import booklet_page_count_stats as page_stats
from cpt_memorization.corpus.analysis import document_error_distribution, ocr_error_stats
from cpt_memorization.corpus.analysis.numeric_heavy_samples import is_numeric_heavy
from cpt_memorization.corpus.cleaning import build_filtered_corpus as filtering
from cpt_memorization.corpus.cleaning import remove_kodak_edge_pages as edges
from cpt_memorization.corpus.cleaning import remove_kodak_noise as noise
from cpt_memorization.corpus.dedup import lz77_dedup as lz77
from cpt_memorization.corpus.dedup import summarize_page_audit as audit
from cpt_memorization.corpus.formatting.ndlocr_extract import extract_document
from cpt_memorization.corpus.stage import stage
from cpt_memorization.memorization.analysis.metrics import token_bleu_score
from cpt_memorization.memorization.analysis.extract_memorized_samples import (
    discover_completions,
    process_line,
)
from cpt_memorization.memorization.frequency import count_suffix_token_exact_matches as exact
from cpt_memorization.workflow import make_plan, execute

ROOT = Path(__file__).resolve().parents[1]


def page(pid, index, text, count=3):
    return {"text": text, "meta": {"pid": pid, "page_index": index, "page_count": count}}


class DevMigrationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".work/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".work/tmp")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def test_effective_order_bleu_and_zero_overlap(self):
        self.assertEqual(token_bleu_score([1, 2, 3], [1, 2, 3]), 1)
        self.assertEqual(token_bleu_score([1, 2, 3], [3, 2, 1]), 0)
        self.assertEqual(token_bleu_score([], []), 0)

    def test_empty_and_trimmed_ocr_preserve_pid(self):
        for xml, trim in [
            ("<OCRDATASET><PAGE/></OCRDATASET>", 0),
            ('<OCRDATASET><PAGE><LINE TYPE="本文" STRING="本文"/></PAGE></OCRDATASET>', 1),
        ]:
            _, pages = extract_document("book.xml.gz", xml, drop_first_pages=trim)
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0]["meta"]["pid"], "book.xml")
            self.assertTrue(pages[0]["meta"]["empty_document"])

    def test_report_uses_fraction_threshold_and_rejects_percent(self):
        from cpt_memorization.memorization.report_metrics import main

        source, output = self.work / "completions.jsonl", self.work / "report.jsonl"
        reference = list(range(50))
        write_jsonl(
            source,
            [
                {
                    "snippet_length": 100,
                    "suffix_token_ids": reference,
                    "predicted_token_ids": reference[:30] + list(range(100, 120)),
                }
            ],
        )
        argv = [
            "report",
            "--input-file",
            str(source),
            "--output-file",
            str(output),
            "--bleu-threshold",
        ]
        with patch("sys.argv", [*argv, "0.75"]):
            main()
        self.assertEqual(next(read_jsonl(output))["approx"], 0)
        with patch("sys.argv", [*argv, "75"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()

    def test_low_jaccard_selection(self):
        row = {
            "suffix_token_ids": list(range(50)),
            "predicted_token_ids": list(range(50)),
            "prefix_token_ids": list(range(100, 200)),
        }
        args = (0, json.dumps(row), "v", "iter_1.jsonl", 0.75, 0.6, 10, True)
        self.assertIsNotNone(process_line(args))
        row["prefix_token_ids"] = list(range(50))
        self.assertIsNone(process_line((0, json.dumps(row), *args[2:])))

    def test_final_selection_is_numeric_and_variant_agnostic(self):
        for variant in ["alpha", "new-model"]:
            for step in [2, 10]:
                write_jsonl(self.work / variant / f"iter_{step}.jsonl", [])
        chosen = discover_completions(self.work, True)
        self.assertEqual([p.name for p in chosen], ["iter_10.jsonl", "iter_10.jsonl"])
        self.assertEqual(len(discover_completions(self.work, False)), 4)

    def test_cleaner_policies_and_input_protection(self):
        source = self.work / "pages.jsonl.gz"
        phrase = "Kodak Color Control Patches"
        write_jsonl(source, [page("A", i, phrase) for i in range(3)])
        edge_out, noise_out = self.work / "edge.jsonl", self.work / "noise.jsonl"
        self.assertEqual(edges.process_file(source, edge_out, False).removed_records, 2)
        self.assertEqual([r["meta"]["page_index"] for r in read_jsonl(edge_out)], [1])
        self.assertEqual(noise.process_file(source, noise_out)["removed_pages"], 1)
        self.assertEqual([r["meta"]["page_index"] for r in read_jsonl(noise_out)], [1, 2])
        for operation in [
            lambda: edges.process_file(source, source, False),
            lambda: noise.process_file(source, source),
        ]:
            with self.assertRaises(ValueError):
                operation()
        self.assertFalse(noise.is_noise_line("Kodak et al discuss the result"))

    def test_pid_rules_intersect_normalized_ids_and_handle_missing(self):
        write_jsonl(
            self.work / "quality.jsonl",
            [{"pid": "A.tar.gz", "quality": 0.9}, {"pid": "B", "quality": 0.2}],
        )
        write_jsonl(
            self.work / "dup.jsonl",
            [{"pid": "A", "ratio": 0.1}, {"pid": "B", "ratio": 0.1}, {"pid": "C", "ratio": 0}],
        )
        config = self.work / "rules.json"
        write_json(
            config,
            [
                {"path": "quality.jsonl", "metric_key": "AUTO", "operator": ">=", "threshold": 0.8},
                {"path": "dup.jsonl", "metric_key": "ratio", "operator": "<", "threshold": 0.5},
            ],
        )
        rules = filtering.load_rules(
            argparse.Namespace(rules=None, rules_config=str(config), encoding="utf-8")
        )
        kept, _, _ = filtering.determine_kept_pids(rules, "utf-8")
        self.assertEqual(kept, {"A"})
        source, output = self.work / "source.jsonl", self.work / "kept.jsonl.gz"
        write_jsonl(source, [page("A", 0, "a"), page("B", 0, "b"), {"text": "missing"}])
        summary = filtering.stream_filter_corpora([source], output, kept, "utf-8")[0]
        self.assertEqual(summary["records_missing_pid"], 1)
        self.assertEqual(summary["records_filtered_out"], 1)
        self.assertEqual(len(list(read_jsonl(output))), 1)
        with self.assertRaises(ValueError):
            filtering.stream_filter_corpora([source], source, kept, "utf-8")

    def test_page_and_marker_statistics(self):
        source = self.work / "pages.jsonl"
        write_jsonl(
            source, [page("A", 0, "a〓"), page("A", 2, "bc〓〓"), page("B", 0, "normal", 1)]
        )
        stats, totals = page_stats.merge_results([page_stats.process_file(source)])
        rows = page_stats.build_booklet_rows(stats)
        self.assertEqual(rows[0]["page_count"], 3)
        self.assertEqual(rows[0]["observed_pages"], 2)
        self.assertEqual(page_stats.summarize(rows, totals)["booklets"], 2)
        self.assertEqual(document_error_distribution.process_file(source)["A"], [3, 6])
        result = ocr_error_stats.analyze_file(source, "text", None)
        self.assertEqual((result["docs"], result["errors_total"], result["error_docs"]), (3, 3, 2))
        self.assertTrue(is_numeric_heavy("1234567890", min_digits=5, digit_ratio=0.8))
        self.assertFalse(is_numeric_heavy("regular text", min_digits=5, digit_ratio=0.8))

    def run_lz77(self, source, output, unit, seen=None):
        output.mkdir()
        ratios = io.StringIO()
        result = lz77.process_file(
            input_path=source,
            output_dir=output,
            text_key="text",
            compression_level=9,
            repetition_ratio_threshold=None,
            min_bytes_for_ratio=64,
            report_every=0,
            workers=1,
            chunk_lines=2,
            sqlite_insert_batch=2,
            dedup_unit=unit,
            global_seen=seen,
            pid_ratio_writer=ratios,
        )
        return result, [json.loads(line) for line in ratios.getvalue().splitlines()]

    def test_lz77_page_booklet_and_cross_file(self):
        source = self.work / "source.jsonl"
        write_jsonl(
            source,
            [
                page("A", 0, "first"),
                page("A", 1, "second"),
                page("B", 0, "first"),
                page("B", 1, "second"),
            ],
        )
        for unit in ["page", "booklet"]:
            result, ratios = self.run_lz77(source, self.work / unit, unit)
            self.assertEqual((result["kept_records"], result["removed_duplicate"]), (2, 2))
            self.assertEqual({r["pid"] for r in ratios}, {"A", "B"})
            self.assertTrue(all(r["compression_ratio"] > 0 for r in ratios))
        seen = {}
        self.run_lz77(source, self.work / "cross1", "booklet", seen)
        result, _ = self.run_lz77(source, self.work / "cross2", "booklet", seen)
        self.assertEqual(result["kept_records"], 0)

    def test_audit_repeated_decisions_do_not_double_count(self):
        audit_dir = self.work / "audit"
        write_jsonl(
            audit_dir / "audit_0.jsonl",
            [
                {"page_id": "A:0", "pid": "A", "remove_flag": False},
                {"page_id": "A:0", "pid": "A", "remove_flag": True},
                {"page_id": "A:1", "pid": "A", "remove_flag": False},
            ],
        )
        output = self.work / "ratio.jsonl"
        with patch(
            "sys.argv", ["audit", "--audit-dir", str(audit_dir), "--output-jsonl", str(output)]
        ):
            audit.main()
        row = next(read_jsonl(output))
        self.assertEqual((row["total_pages"], row["removed_pages"], row["dup_ratio"]), (2, 1, 0.5))

    def test_exact_scan_counts_overlapping_occurrences(self):
        tokenizer = argparse.Namespace(
            encode_for_suffix_count=lambda text: list(map(int, text.split()))
        )
        with (
            patch.object(exact, "TOKENIZER", tokenizer),
            patch.object(exact, "SUFFIX_BY_LENGTH", {2: {(1, 1): ["repeat"]}}),
        ):
            self.assertEqual(
                exact.count_batch(
                    [json.dumps({"text": "1 1 1"}), json.dumps({"text": "1 1"})], "text"
                ),
                {"repeat": 3},
            )

    def test_staging_preserves_colliding_basenames(self):
        for corpus in ["a", "b"]:
            write_jsonl(self.work / corpus / "shard.jsonl", [{"text": corpus}])
        staged = self.work / "staged"
        records = stage([self.work / "a", self.work / "b"], staged)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            [next(read_jsonl(staged / r["shard"]))["text"] for r in records], ["a", "b"]
        )
        with self.assertRaises(ValueError):
            stage([self.work / "a", self.work / "a"], self.work / "duplicate")
        with self.assertRaises(FileExistsError):
            stage([self.work / "a"], staged)

    def plan(self):
        config = self.work / "workflow.json"
        write_json(
            config,
            {
                "output_dir": "run",
                "variables": {"message": "literal $() ' quote"},
                "steps": [
                    {
                        "id": "second",
                        "command": "corpus-count-chars",
                        "args": ["{message}"],
                        "needs": ["first"],
                    },
                    {"id": "first", "command": "corpus-count-records", "args": ["data"]},
                ],
            },
        )
        return make_plan(config)

    def test_workflow_order_slurm_dependencies_and_quoting(self):
        plan = self.plan()
        self.assertEqual([s["id"] for s in plan["steps"]], ["first", "second"])
        self.assertEqual(plan["steps"][1]["command"][-1], "literal $() ' quote")
        with patch(
            "cpt_memorization.workflow.subprocess.run",
            side_effect=[
                subprocess.CompletedProcess([], 0, "101;cluster\n"),
                subprocess.CompletedProcess([], 0, "102\n"),
            ],
        ) as run:
            state = execute(plan, "slurm")
        self.assertIn("--dependency=afterok:101", run.call_args_list[1].args[0])
        self.assertEqual(state["steps"]["second"], {"status": "submitted", "job_id": "102"})
        with self.assertRaises(FileExistsError):
            execute(plan, dry_run=True)

    def test_workflow_fail_fast_records_failure(self):
        plan = self.plan()
        with patch(
            "cpt_memorization.workflow.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "command"),
        ) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                execute(plan)
        self.assertEqual(run.call_count, 1)
        state = json.loads((self.work / "run/state.json").read_text())
        self.assertEqual(state["steps"]["first"]["status"], "failed")
        self.assertNotIn("second", state["steps"])

    def test_workflow_rejects_cycles_and_unknown_dependencies(self):
        for dependency in ["loop", "missing"]:
            config = self.work / "bad.json"
            write_json(
                config,
                {
                    "output_dir": "bad",
                    "steps": [
                        {"id": "loop", "command": "corpus-count-chars", "needs": [dependency]}
                    ],
                },
            )
            with self.assertRaises(ValueError):
                make_plan(config)


if __name__ == "__main__":
    unittest.main()

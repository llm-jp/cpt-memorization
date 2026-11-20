"""Check anchor sampling and tokenizer propagation without downloading a model."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from cpt_memorization.io import read_jsonl, write_jsonl
from cpt_memorization.memorization.pipeline.extract_samples import main as sample_main
from cpt_memorization.memorization.pipeline.generate import main as generate_main

ROOT = Path(__file__).resolve().parents[1]


class ToyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


class MemoryPipelineTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".work/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".work/tmp")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        tokenizer_module = types.ModuleType("transformers")
        tokenizer_module.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda _: ToyTokenizer()
        )
        patcher = patch.dict(sys.modules, {"transformers": tokenizer_module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_fixed_suffix_anchors_across_lengths_and_compression(self):
        documents = [
            dict(
                text=f"Synthetic document number {i} with ample characters for sampling.",
                metadata={"meta": {"archive_member": f"synthetic-{i}"}},
            )
            for i in range(5)
        ]
        results = []
        for suffix in (".jsonl", ".jsonl.gz"):
            source = self.work / f"corpus{suffix}"
            output = self.work / f"samples{suffix}.jsonl"
            write_jsonl(source, documents)
            with patch.object(
                sys,
                "argv",
                [
                    "memory-sample",
                    "--model",
                    "toy-model",
                    "--tokenizer",
                    "toy-tokenizer",
                    "--input-file",
                    str(source),
                    "--output-file",
                    str(output),
                    "--sample-size",
                    "3",
                    "--snippet-length",
                    "8",
                    "12",
                    "--suffix-length",
                    "4",
                    "--seed",
                    "17",
                ],
            ):
                sample_main()
            results.append(list(read_jsonl(output)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(results[0]), 6)
        for short, long in zip(results[0][::2], results[0][1::2], strict=True):
            self.assertEqual(short["doc_id"], long["doc_id"])
            self.assertEqual(short["suffix_token_ids"], long["suffix_token_ids"])
            self.assertEqual(short["prefix_token_ids"], long["prefix_token_ids"][-4:])
            self.assertEqual(short["tokenizer"], "toy-tokenizer")

    def test_generation_uses_sample_tokenizer_for_engine_and_decoding(self):
        recorded = {}

        class Engine:
            def __init__(self, **kwargs):
                recorded.update(kwargs)

            def generate(self, prompts, params):
                recorded["sampling"] = params
                return [
                    types.SimpleNamespace(
                        outputs=[types.SimpleNamespace(token_ids=[65, 66, 67, 68])]
                    )
                    for _ in prompts
                ]

        vllm = types.ModuleType("vllm")
        vllm.LLM = Engine
        vllm.TokensPrompt = lambda **kwargs: kwargs
        vllm.SamplingParams = lambda **kwargs: kwargs
        source, output = self.work / "samples.jsonl", self.work / "completions.jsonl"
        write_jsonl(
            source,
            [
                dict(
                    prefix_token_ids=[1, 2, 3, 4],
                    suffix_token_ids=[5, 6, 7, 8],
                    tokenizer="toy-tokenizer",
                )
            ],
        )
        with (
            patch.dict(sys.modules, {"vllm": vllm}),
            patch.object(
                sys,
                "argv",
                [
                    "memory-generate",
                    "--model",
                    "toy-model",
                    "--input-file",
                    str(source),
                    "--output-file",
                    str(output),
                    "--seed",
                    "91",
                ],
            ),
        ):
            generate_main()
        result = next(read_jsonl(output))
        self.assertEqual(result["predicted"], "ABCD")
        self.assertEqual(recorded["tokenizer"], "toy-tokenizer")
        self.assertEqual(recorded["sampling"]["max_tokens"], 4)
        self.assertTrue(recorded["sampling"]["ignore_eos"])
        self.assertEqual(recorded["sampling"]["seed"], 91)

    def run_batched_generation(self, rows, *, fail_on_batch=None, compressed=False, truncate=False):
        recorded = {"engines": [], "batches": []}

        class Engine:
            def __init__(self, **kwargs):
                recorded["engines"].append(kwargs)

            def generate(self, prompts, params):
                recorded["batches"].append([prompt["prompt_token_ids"] for prompt in prompts])
                if len(recorded["batches"]) == fail_on_batch:
                    raise RuntimeError("Inference failed")
                return [
                    types.SimpleNamespace(
                        outputs=[
                            types.SimpleNamespace(
                                token_ids=[prompt["prompt_token_ids"][0]]
                                * (params["max_tokens"] - int(truncate))
                            )
                        ]
                    )
                    for prompt in prompts
                ]

        vllm = types.ModuleType("vllm")
        vllm.LLM = Engine
        vllm.TokensPrompt = lambda **kwargs: kwargs
        vllm.SamplingParams = lambda **kwargs: kwargs
        source = self.work / ("samples.jsonl.gz" if compressed else "samples.jsonl")
        output = self.work / "completions.jsonl"
        write_jsonl(source, rows)
        with (
            patch.dict(sys.modules, {"vllm": vllm}),
            patch.object(
                sys,
                "argv",
                [
                    "memory-generate",
                    "--model",
                    "toy-model",
                    "--input-file",
                    str(source),
                    "--output-file",
                    str(output),
                    "--batch-size",
                    "2",
                ],
            ),
        ):
            generate_main()
        return list(read_jsonl(output)), recorded

    def generation_rows(self):
        return [
            dict(
                doc_id=str(i),
                prefix_token_ids=[65 + i],
                suffix_token_ids=[5, 6, 7, 8],
                tokenizer="toy-tokenizer",
            )
            for i in range(5)
        ]

    def test_generation_batches_preserve_order_and_reuse_one_engine(self):
        rows = self.generation_rows()
        result, recorded = self.run_batched_generation(rows, compressed=True)
        self.assertEqual([row["doc_id"] for row in result], [row["doc_id"] for row in rows])
        self.assertEqual([row["predicted"] for row in result], [letter * 4 for letter in "ABCDE"])
        self.assertEqual([len(batch) for batch in recorded["batches"]], [2, 2, 1])
        self.assertEqual(len(recorded["engines"]), 1)
        self.assertEqual(recorded["engines"][0]["tokenizer"], "toy-tokenizer")
        self.assertEqual(list(self.work.glob(".completions.jsonl.*.tmp")), [])

    def test_generation_checks_tokenizer_consistency_across_batches(self):
        rows = self.generation_rows()
        rows[-1]["tokenizer"] = "different-tokenizer"
        result, recorded = self.run_batched_generation(rows)
        self.assertEqual(recorded["engines"][0]["tokenizer"], "toy-model")
        self.assertTrue(all(row["generation"]["tokenizer"] == "toy-model" for row in result))

    def test_generation_rejects_invalid_input_without_replacing_existing_output(self):
        inconsistent = self.generation_rows()
        inconsistent[-1]["suffix_token_ids"] = [1, 2]
        empty_suffix = [dict(prefix_token_ids=[1], suffix_token_ids=[])]
        for invalid in ([], inconsistent, empty_suffix):
            with self.subTest(rows=len(invalid)):
                output = self.work / "completions.jsonl"
                output.write_text("existing output\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.run_batched_generation(invalid)
                self.assertEqual(output.read_text(encoding="utf-8"), "existing output\n")
                self.assertEqual(list(self.work.glob(".completions.jsonl.*.tmp")), [])

    def test_generation_failure_preserves_output_and_removes_partial_file(self):
        output = self.work / "completions.jsonl"
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    output.write_text("existing output\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Inference failed"):
                    self.run_batched_generation(self.generation_rows(), fail_on_batch=2)
                if existing:
                    self.assertEqual(output.read_text(encoding="utf-8"), "existing output\n")
                else:
                    self.assertFalse(output.exists())
                self.assertEqual(list(self.work.glob(".completions.jsonl.*.tmp")), [])

    def test_short_generation_is_rejected_without_publishing_output(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 generated tokens"):
            self.run_batched_generation(self.generation_rows(), truncate=True)
        self.assertFalse((self.work / "completions.jsonl").exists())
        self.assertEqual(list(self.work.glob(".completions.jsonl.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

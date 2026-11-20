"""Exercise QA orchestration against a deterministic in-process inference stub."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from cpt_memorization.io import read_jsonl, write_jsonl
from cpt_memorization.qa.answer import main as answer
from cpt_memorization.qa.chat import run

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    options = None
    invalid = False

    def __init__(self, **kwargs):
        FakeLLM.options = kwargs

    def chat(self, messages, **kwargs):
        outputs = []
        for prompt in messages:
            system = prompt[0]["content"]
            if "decide whether" in system:
                value = {"is_qa_eligible": True}
            elif "verify the correctness" in system:
                value = {"is_correct": "false" if self.invalid else False}
            else:
                value = {"question": "Which value?", "answer": "A synthetic answer"}
            outputs.append(
                types.SimpleNamespace(outputs=[types.SimpleNamespace(text=json.dumps(value))])
            )
        return outputs

    def generate(self, prompts, **kwargs):
        return [
            types.SimpleNamespace(outputs=[types.SimpleNamespace(text="Answer line\nExtra line")])
            for _ in prompts
        ]


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".work/tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".work/tmp")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        vllm = types.ModuleType("vllm")
        vllm.LLM = FakeLLM
        vllm.SamplingParams = lambda **kw: kw
        params = types.ModuleType("vllm.sampling_params")
        params.SamplingParams = lambda **kw: kw
        params.StructuredOutputsParams = lambda **kw: kw
        patcher = patch.dict(sys.modules, {"vllm": vllm, "vllm.sampling_params": params})
        patcher.start()
        self.addCleanup(patcher.stop)
        FakeLLM.invalid = False

    def stage_args(self, source, output):
        return [
            "--input-file",
            str(source),
            "--output-file",
            str(output),
            "--model-name",
            "synthetic-model",
            "--response-format",
            "json",
        ]

    def test_filter_synthesize_answer_and_evaluate_contract(self):
        source = self.work / "source.jsonl.gz"
        write_jsonl(
            source, [{"text": f"Synthetic full document {i}", "id": str(i)} for i in range(4)]
        )
        filtered, synthesized, answered, evaluated = (
            self.work / f"{s}.jsonl" for s in ("filter", "synthesize", "answer", "evaluate")
        )
        run(
            "filter",
            self.stage_args(source, filtered) + ["--fraction", "1", "--max-text-length", "0"],
        )
        self.assertEqual(list(read_jsonl(filtered))[0]["text"], "Synthetic full document 0")
        run("synthesize", self.stage_args(filtered, synthesized))
        with patch.object(
            sys,
            "argv",
            [
                "qa-answer",
                "--input-file",
                str(synthesized),
                "--output-file",
                str(answered),
                "--model-name",
                "synthetic-base",
                "--tokenizer",
                "compatible-tokenizer",
                "--seed",
                "71",
            ],
        ):
            answer()
        self.assertEqual(FakeLLM.options["tokenizer"], "compatible-tokenizer")
        self.assertEqual(FakeLLM.options["seed"], 71)
        self.assertEqual(
            list(read_jsonl(answered))[0]["predicted_answer"], "Answer line\nExtra line"
        )
        run("evaluate", self.stage_args(answered, evaluated))
        rows = list(read_jsonl(evaluated))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["is_correct"] is False for row in rows))

    def test_invalid_judgment_fails_without_silently_shrinking_denominator(self):
        source, target = self.work / "source.jsonl", self.work / "result.jsonl"
        write_jsonl(
            source,
            [
                {
                    "text": "Synthetic context",
                    "question": "Q?",
                    "answer": "A",
                    "predicted_answer": "B",
                    "metadata": {"id": "x"},
                }
            ],
        )
        FakeLLM.invalid = True
        with self.assertRaisesRegex(ValueError, "invalid responses"):
            run("evaluate", self.stage_args(source, target))
        self.assertFalse(target.exists())
        self.assertEqual(len(list(read_jsonl(str(target) + ".errors.jsonl"))), 1)
        run("evaluate", self.stage_args(source, target) + ["--skip-invalid"])
        self.assertEqual(list(read_jsonl(target)), [])


if __name__ == "__main__":
    unittest.main()

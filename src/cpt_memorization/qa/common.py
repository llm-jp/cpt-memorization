"""Prompt rendering and validation shared by QA construction and evaluation."""

import hashlib
import json
import random
from importlib.resources import files

import yaml

from ..io import read_jsonl

PROMPTS = {
    "filter": "verification_filter.yaml",
    "synthesize": "verification_synthesize.yaml",
    "evaluate": "verification_evaluate.yaml",
    "open": "verification_answer.yaml",
    "closed": "closedbook_answer.yaml",
}


def load_prompt(name):
    if name in PROMPTS:
        text = files(__package__).joinpath("prompts", PROMPTS[name]).read_text()
    else:
        from pathlib import Path

        text = Path(name).read_text(encoding="utf-8")
    prompt = yaml.safe_load(text)
    if not isinstance(prompt, dict) or not isinstance(prompt.get("user_prompt"), str):
        raise ValueError("Prompt must contain user_prompt")
    return prompt


def render(template, record, *, answer=None):
    values = {
        "document": record.get("text", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer", "") if answer is None else answer,
        "predicted_answer": record.get("predicted_answer", ""),
    }
    # One-pass replacement prevents placeholders inside source text being substituted.
    import re

    return re.sub(
        r"\{\{ (document|question|answer|predicted_answer) \}\}",
        lambda match: values[match[1]],
        template,
    )


def document_id(record):
    metadata = record.get("metadata") or {}
    meta = record.get("meta") or metadata.get("meta") or {}
    value = metadata.get("id") or record.get("id") or meta.get("pid") or meta.get("id")
    if value is None:
        value = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    if "page_index" in meta:
        value = f"{value}:page:{meta['page_index']}"
    return str(value)


def load_records(path, *, evaluated=False):
    records = {}
    for record in read_jsonl(path):
        key = (record.get("metadata") or {}).get("id")
        if key is None or str(key) == "":
            raise ValueError(f"{path}: missing metadata.id")
        key = str(key)
        if key in records:
            raise ValueError(f"{path}: duplicate metadata.id {key}")
        if evaluated and type(record.get("is_correct")) is not bool:
            raise ValueError(f"{path}: is_correct must be a JSON boolean for {key}")
        records[key] = record
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records


def validate_common(runs, ids):
    for key in ids:
        ref = next(iter(runs.values()))[key]
        for name, run in runs.items():
            for field in ("text", "question", "answer"):
                if field not in ref or run[key].get(field) != ref[field]:
                    raise ValueError(f"{name}: mismatched {field} for id {key}")


def answer_prompts(records, template, num_shots=3, seed=42):
    if not 0 <= num_shots < len(records):
        raise ValueError("num_shots must be >= 0 and smaller than the dataset")
    if "{{ question }}" not in template or "{{ answer }}" not in template:
        raise ValueError("Answer prompt requires question and answer placeholders")
    rng = random.Random(seed)
    prompts = []
    for index, record in enumerate(records):
        others = [i for i in range(len(records)) if i != index]
        shots = rng.sample(others, num_shots)
        parts = [render(template, records[i]).strip() for i in shots]
        parts.append(render(template, record, answer="").strip())
        prompts.append("\n\n".join(parts))
    return prompts


def parse_response(response, response_format="auto"):
    output = response.outputs[0]
    if response_format != "harmony":
        try:
            value = json.loads(output.text.strip())
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (json.JSONDecodeError, ValueError):
            if response_format == "json":
                raise
    from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    messages = encoding.parse_messages_from_completion_tokens(output.token_ids, Role.ASSISTANT)
    final = [message for message in messages if message.channel == "final"]
    if not final:
        raise ValueError("No final-channel Harmony message")
    value = json.loads("".join(part.text for part in final[-1].content))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def add_inference_arguments(parser):
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer", help="Tokenizer path/name compatible with the checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)


def make_llm(args):
    from vllm import LLM

    options = dict(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=True,
    )
    if args.tokenizer:
        options["tokenizer"] = args.tokenizer
    return LLM(**options)

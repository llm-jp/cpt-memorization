"""Common execution for document selection, QA synthesis, and LLM judging."""

import argparse
import logging
import random

from ..io import read_jsonl, write_json, write_jsonl
from .common import (
    add_inference_arguments,
    document_id,
    load_prompt,
    load_records,
    make_llm,
    parse_response,
    render,
)


def run(stage, argv=None):
    parser = argparse.ArgumentParser(description=f"QA {stage}")
    parser.add_argument("--input-file", nargs="+" if stage == "filter" else None, required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--prompt-file", default=stage, help="Bundled prompt name or YAML path")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--response-format", choices=["auto", "json", "harmony"], default="auto")
    parser.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high"], default="medium"
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Record malformed outputs in .errors.jsonl and continue",
    )
    if stage == "filter":
        parser.add_argument("--fraction-to-keep", "--fraction", type=float, default=0.01)
        parser.add_argument("--max-text-length", type=int, default=1000)
    add_inference_arguments(parser)
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.temperature < 0:
        parser.error("Require max-tokens > 0 and temperature >= 0")
    rng = random.Random(args.seed)
    if stage == "filter":
        if not 0 < args.fraction_to_keep <= 1 or args.max_text_length < 0:
            parser.error("Require 0 < fraction <= 1 and max-text-length >= 0")
        records = []
        seen = set()
        for path in args.input_file:
            for record in read_jsonl(path):
                if rng.random() >= args.fraction_to_keep:
                    continue
                key = document_id(record)
                if key in seen:
                    raise ValueError(f"Duplicate source id {key}; use unique document/page IDs")
                seen.add(key)
                records.append({**record, "metadata": {"id": key}})
        for record in records:
            text = record["text"]
            if args.max_text_length and len(text) > args.max_text_length:
                start = rng.randint(0, len(text) - args.max_text_length)
                record["text"] = text[start : start + args.max_text_length]
    else:
        records = list(load_records(args.input_file).values())
    if not records:
        raise ValueError("No selected records; increase fraction or provide more input")
    prompt = load_prompt(args.prompt_file)
    messages = []
    for record in records:
        turns = []
        if prompt.get("system_prompt"):
            turns.append({"role": "system", "content": prompt["system_prompt"]})
        turns.append({"role": "user", "content": render(prompt["user_prompt"], record)})
        messages.append(turns)

    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    llm = make_llm(args)
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        structured_outputs=StructuredOutputsParams(json=prompt["output_schema"]),
    )
    chat_options = {}
    if "gpt-oss" in args.model_name.lower():
        chat_options["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    responses = llm.chat(messages, sampling_params=sampling, **chat_options)
    outputs, errors = [], []
    for record, response in zip(records, responses, strict=True):
        try:
            value = parse_response(response, args.response_format)
            if stage in ("filter", "evaluate"):
                field = "is_qa_eligible" if stage == "filter" else "is_correct"
                if type(value.get(field)) is not bool:
                    raise ValueError(f"{field} must be a JSON boolean")
                if stage == "filter" and not value[field]:
                    continue
            else:
                if any(
                    not isinstance(value.get(key), str) or not value[key].strip()
                    for key in ("question", "answer")
                ):
                    raise ValueError("question and answer must be nonempty strings")
            result = {
                "text": record["text"],
                "metadata": {
                    **record["metadata"],
                    "prompt_name": prompt.get("name", stage),
                    "model": args.model_name,
                    "tokenizer": args.tokenizer,
                    "seed": args.seed,
                    "temperature": args.temperature,
                },
            }
            if stage == "synthesize":
                result.update(question=value["question"], answer=value["answer"])
            elif stage == "evaluate":
                result.update(
                    {key: record[key] for key in ("question", "answer", "predicted_answer")}
                )
                result["is_correct"] = value["is_correct"]
            outputs.append(result)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            errors.append({"id": record["metadata"]["id"], "error": str(exc)})
    write_jsonl(args.output_file + ".errors.jsonl", errors)
    write_json(
        args.output_file + ".summary.json",
        {
            "stage": stage,
            "input": len(records),
            "output": len(outputs),
            "invalid": len(errors),
            "arguments": vars(args),
        },
    )
    if errors and not args.skip_invalid:
        raise ValueError(
            f"{len(errors)} invalid responses; see {args.output_file}.errors.jsonl. No result file written."
        )
    write_jsonl(args.output_file, outputs)
    logging.info(
        "%s: input=%s output=%s invalid=%s", stage, len(records), len(outputs), len(errors)
    )

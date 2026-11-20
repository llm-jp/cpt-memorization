import argparse
import logging
import json
import os
import tempfile
from itertools import batched
from pathlib import Path

from ...io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Generate")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to use for generation.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="Tokenizer name/path (defaults to --model).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input file containing data for analysis.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output file to save results.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Maximum number of input rows to hold and generate at once.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Require temperature >= 0 and 0 < top-p <= 1")

    # Validate the complete input before loading the model. This also preserves
    # tokenizer selection across batches without retaining all evaluation rows.
    suffix_length = None
    sample_tokenizer = None
    shared_tokenizer = True
    sample_count = 0
    for sample in read_jsonl(args.input_file):
        if suffix_length is None:
            suffix_length = len(sample["suffix_token_ids"])
            sample_tokenizer = sample.get("tokenizer")
        elif suffix_length != len(sample["suffix_token_ids"]):
            raise ValueError("Inconsistent suffix lengths in the input samples.")
        shared_tokenizer &= sample.get("tokenizer") == sample_tokenizer
        sample_count += 1

    if not sample_count or not suffix_length:
        raise ValueError("Input must contain samples with nonempty suffix tokens")

    from vllm import LLM, TokensPrompt, SamplingParams
    from transformers import AutoTokenizer

    sampling_params = SamplingParams(
        max_tokens=suffix_length,
        temperature=args.temperature,
        top_p=args.top_p,
        ignore_eos=True,  # Continue until max_tokens is reached
        seed=args.seed,
    )

    tokenizer_name = args.tokenizer or (
        sample_tokenizer if sample_tokenizer and shared_tokenizer else args.model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    generated = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wt",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for samples in batched(read_jsonl(args.input_file), args.batch_size):
                prompts = [
                    TokensPrompt(prompt_token_ids=sample["prefix_token_ids"]) for sample in samples
                ]
                outputs = model.generate(prompts, sampling_params)
                for sample, output in zip(samples, outputs, strict=True):
                    predicted_token_ids = list(output.outputs[0].token_ids)
                    if len(predicted_token_ids) != suffix_length:
                        raise ValueError(
                            f"Expected {suffix_length} generated tokens, "
                            f"received {len(predicted_token_ids)}"
                        )
                    sample["predicted_token_ids"] = predicted_token_ids
                    sample["predicted"] = tokenizer.decode(
                        predicted_token_ids, skip_special_tokens=False
                    )
                    sample["generation"] = {
                        "model": args.model,
                        "tokenizer": tokenizer_name,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": args.seed,
                    }
                    stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
                generated += len(samples)
                logging.info("Generated %d/%d samples.", generated, sample_count)
        os.replace(temporary_path, output_path)
        logging.info("Saved %d samples to %s.", generated, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

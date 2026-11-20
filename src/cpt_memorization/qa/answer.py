"""Generate few-shot open-book or closed-book answers with a base language model."""

import argparse

from ..io import write_json, write_jsonl
from .common import add_inference_arguments, answer_prompts, load_prompt, load_records, make_llm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--prompt-file", default="open", help="open, closed, or YAML path")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--few-shot-seed",
        type=int,
        help="Defaults to --seed; set to fix shots across sampling runs",
    )
    add_inference_arguments(parser)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.temperature < 0:
        parser.error("Require max-tokens > 0 and temperature >= 0")
    records = list(load_records(args.input_file).values())
    template = load_prompt(args.prompt_file)
    few_shot_seed = args.seed if args.few_shot_seed is None else args.few_shot_seed
    prompts = answer_prompts(records, template["user_prompt"], args.num_shots, few_shot_seed)
    from vllm import SamplingParams

    llm = make_llm(args)
    responses = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed
        ),
    )
    outputs = []
    for record, response in zip(records, responses, strict=True):
        outputs.append(
            {
                **record,
                "predicted_answer": response.outputs[0].text.strip(),
                "metadata": {
                    **record["metadata"],
                    "model": args.model_name,
                    "tokenizer": args.tokenizer,
                    "seed": args.seed,
                    "few_shot_seed": few_shot_seed,
                    "temperature": args.temperature,
                    "num_shots": args.num_shots,
                    "prompt_name": template.get("name"),
                },
            }
        )
    write_jsonl(args.output_file, outputs)
    write_json(
        args.output_file + ".summary.json", {"arguments": vars(args), "output": len(outputs)}
    )


if __name__ == "__main__":
    main()

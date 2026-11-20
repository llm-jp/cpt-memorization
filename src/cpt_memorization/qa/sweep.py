"""Run a configurable model × mode × temperature × seed experiment grid."""

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

from ..io import write_json


def options(values):
    result = []
    for name, value in values.items():
        if value is None or value is False:
            continue
        result.append("--" + name.replace("_", "-"))
        if value is not True:
            result.append(str(value))
    return result


def make_plan(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def resolve(value):
        path = Path(value)
        return str(path if path.is_absolute() else config_path.parent / path)

    qa_file = resolve(config["qa_file"])
    output_dir = Path(resolve(config["output_dir"]))
    temperatures = config.get("temperatures", [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    seeds = config.get("seeds", [42, 43, 44])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if len(set(temperatures)) != len(temperatures) or any(
        not math.isfinite(t) or t < 0 for t in temperatures
    ):
        raise ValueError("temperatures must be unique, finite, and nonnegative")
    if not config["models"] or not temperatures:
        raise ValueError("models and temperatures must be nonempty")
    greedy_runs = config.get("greedy_runs", 1)
    if not isinstance(greedy_runs, int) or not 1 <= greedy_runs <= len(seeds):
        raise ValueError("greedy_runs must be between 1 and the number of seeds")
    jobs, runs = [], []
    for mode in config.get("modes", ["open", "closed"]):
        if mode not in ("open", "closed"):
            raise ValueError("modes must be open or closed")
        for temperature in temperatures:
            for run, seed in enumerate(seeds[:greedy_runs] if temperature == 0 else seeds, 1):
                for name, model in config["models"].items():
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
                        raise ValueError(f"Unsafe model label {name}")
                    folder = output_dir / f"{mode}_temp{temperature:g}_run{run}"
                    answered = str(folder / f"output_{name}.jsonl")
                    evaluated = str(folder / f"evaluated_{name}.jsonl")
                    answer = {
                        **config.get("answer_options", {}),
                        **model,
                        "input_file": qa_file,
                        "output_file": answered,
                        "prompt_file": mode,
                        "temperature": temperature,
                        "seed": seed,
                    }
                    judge = {**config["judge"], "input_file": answered, "output_file": evaluated}
                    jobs.extend(
                        [
                            dict(
                                command=[
                                    sys.executable,
                                    "-m",
                                    "cpt_memorization",
                                    "qa-answer",
                                    *options(answer),
                                ],
                                output=answered,
                            ),
                            dict(
                                command=[
                                    sys.executable,
                                    "-m",
                                    "cpt_memorization",
                                    "qa-evaluate",
                                    *options(judge),
                                ],
                                output=evaluated,
                            ),
                        ]
                    )
                    runs.append(
                        dict(
                            mode=mode,
                            temperature=temperature,
                            run=run,
                            model=name,
                            seed=seed,
                            path=evaluated,
                        )
                    )
    if len({job["output"] for job in jobs}) != len(jobs):
        raise ValueError("Experiment grid contains duplicate jobs")
    return output_dir, {"runs": runs}, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write plan and manifest, without model inference"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing existing run outputs"
    )
    args = parser.parse_args()
    output, manifest, jobs = make_plan(args.config)
    if not args.dry_run and not args.overwrite:
        existing = [job["output"] for job in jobs if Path(job["output"]).exists()]
        if existing:
            raise FileExistsError(
                f"Outputs already exist: {existing[:3]}; choose another output_dir or --overwrite"
            )
    write_json(output / "manifest.json", manifest)
    write_json(output / "plan.json", {"jobs": jobs})
    for job in jobs:
        print(shlex.join(job["command"]), flush=True)
        if not args.dry_run:
            subprocess.run(job["command"], check=True)
    if not args.dry_run:
        from .aggregate import aggregate

        aggregate(output / "manifest.json", output / "summary")


if __name__ == "__main__":
    main()

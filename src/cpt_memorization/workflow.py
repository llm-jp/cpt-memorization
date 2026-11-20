"""Execute a dependency graph of package commands locally or through Slurm."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from .io import write_json


def make_plan(config_path):
    from .__main__ import COMMANDS

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cwd = (config_path.parent / config.get("workdir", ".")).resolve()
    variables = {
        **config.get("variables", {}),
        "config_dir": str(config_path.parent),
        "workdir": str(cwd),
        "python": sys.executable,
    }

    def expand(value, stack=()):
        def substitute(match):
            key = match[1]
            if key not in variables or key in stack:
                raise ValueError(f"Unknown or cyclic variable: {key}")
            return expand(variables[key], (*stack, key))

        return re.sub(r"\{([A-Za-z_]\w*)\}", substitute, str(value))

    output = (cwd / expand(config["output_dir"])).resolve()
    steps = config["steps"]
    if not steps:
        raise ValueError("Workflow must contain at least one step")
    by_id = {}
    for step in steps:
        name = step["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in by_id:
            raise ValueError(f"Invalid or duplicate step id: {name}")
        if step["command"] not in COMMANDS or step["command"] == "workflow":
            raise ValueError(f"Unknown or recursive workflow command: {step['command']}")
        if not isinstance(step.get("args", []), list):
            raise ValueError("Step args must be an array of command arguments")
        by_id[name] = step
    ordered, visiting, visited = [], set(), set()

    def visit(name):
        if name not in by_id:
            raise ValueError(f"Unknown dependency: {name}")
        if name in visiting:
            raise ValueError(f"Dependency cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        step = by_id[name]
        for dependency in step.get("needs", []):
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        resources = {**config.get("slurm", {}), **step.get("slurm", {})}
        allowed = {
            "partition",
            "time",
            "gres",
            "cpus_per_task",
            "mem",
            "nodes",
            "ntasks",
            "account",
            "qos",
            "constraint",
        }
        if set(resources) - allowed:
            raise ValueError(f"Unsupported Slurm resources: {set(resources) - allowed}")
        command = [
            sys.executable,
            "-m",
            "cpt_memorization",
            step["command"],
            *(expand(value) for value in step.get("args", [])),
        ]
        ordered.append(
            {
                "id": name,
                "needs": step.get("needs", []),
                "command": command,
                "slurm": {key: expand(value) for key, value in resources.items()},
            }
        )

    for name in by_id:
        visit(name)
    return {
        "workdir": str(cwd),
        "output_dir": str(output),
        "steps": ordered,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def execute(plan, executor="local", dry_run=False, overwrite=False):
    output = Path(plan["output_dir"])
    state_path = output / "state.json"
    # Never erase a previous execution record just by inspecting a new plan.
    if state_path.exists() and (dry_run or not overwrite):
        raise FileExistsError(f"Workflow already executed: {state_path}; choose a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    scripts = {}
    for step in plan["steps"]:
        path = output / f"{step['id']}.sh"
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + "cd -- "
            + shlex.quote(plan["workdir"])
            + "\n"
            + "exec "
            + shlex.join(step["command"])
            + "\n",
            encoding="utf-8",
        )
        scripts[step["id"]] = path
    write_json(output / "plan.json", plan)
    if dry_run:
        for step in plan["steps"]:
            print(step["id"], "after", ",".join(step["needs"]), shlex.join(step["command"]))
        return {}
    state = {"executor": executor, "config_sha256": plan["config_sha256"], "steps": {}}
    write_json(state_path, state)
    for step in plan["steps"]:
        name = step["id"]
        try:
            if executor == "local":
                with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        step["command"],
                        cwd=plan["workdir"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                state["steps"][name] = {"status": "completed"}
            else:
                args = [
                    "sbatch",
                    "--parsable",
                    f"--job-name={name}",
                    f"--chdir={plan['workdir']}",
                    f"--output={output / (name + '-%j.log')}",
                ]
                if step["needs"]:
                    ids = [state["steps"][dep]["job_id"] for dep in step["needs"]]
                    args.append("--dependency=afterok:" + ":".join(ids))
                for key, value in step["slurm"].items():
                    args.append("--" + key.replace("_", "-") + "=" + value)
                result = subprocess.run(
                    [*args, str(scripts[name])], capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip().split(";", 1)[0]
                if not job_id.isdigit():
                    raise ValueError(f"Unexpected sbatch output: {result.stdout!r}")
                state["steps"][name] = {"status": "submitted", "job_id": job_id}
        except Exception as exc:
            state["steps"][name] = {"status": "failed", "error": str(exc)}
            write_json(state_path, state)
            raise
        write_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executor", choices=["local", "slurm"], default="local")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Rerun all steps and replace prior outputs"
    )
    args = parser.parse_args()
    execute(make_plan(args.config), args.executor, args.dry_run, args.overwrite)


if __name__ == "__main__":
    main()

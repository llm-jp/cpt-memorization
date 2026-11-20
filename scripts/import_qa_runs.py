"""Create a generic manifest from <mode>_temp<temperature>_run<repeat> folders.

Only the output directory is written. Source JSONL files are read in place.
"""

import argparse
import re
from pathlib import Path

from cpt_memorization.io import write_json
from cpt_memorization.qa.aggregate import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", required=True, help="Filename suffix=display label, e.g. base=BASE"
    )
    args = parser.parse_args()
    models = dict(value.split("=", 1) for value in args.models)
    runs = []
    pattern = re.compile(r"(open|closed)_temp(\d+(?:[p.]\d+)?)_run(\d+)")
    for folder in sorted(args.results_root.iterdir()):
        match = pattern.fullmatch(folder.name)
        if not folder.is_dir() or match is None:
            continue
        for suffix, label in models.items():
            path = folder / f"evaluated_{suffix}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            runs.append(
                dict(
                    mode=match[1],
                    temperature=float(match[2].replace("p", ".")),
                    run=int(match[3]),
                    model=label,
                    path=str(path.resolve()),
                )
            )
    if not runs:
        raise ValueError("No run directories matched")
    manifest = args.output_dir / "manifest.json"
    write_json(manifest, {"runs": runs})
    print(aggregate(manifest, args.output_dir / "summary"))


if __name__ == "__main__":
    main()

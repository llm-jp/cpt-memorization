"""Plot mean QA accuracy with run-to-run standard deviations from aggregate CSV."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.input_file.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty aggregate CSV")
    modes = sorted({r["mode"] for r in rows})
    models = sorted({r["model"] for r in rows})
    figure, axes = plt.subplots(
        1, len(modes), figsize=(5 * len(modes), 4), squeeze=False, sharey=True
    )
    for ax, mode in zip(axes[0], modes, strict=True):
        for model in models:
            series = sorted(
                [r for r in rows if r["mode"] == mode and r["model"] == model],
                key=lambda r: float(r["temperature"]),
            )
            ax.errorbar(
                [float(r["temperature"]) for r in series],
                [100 * float(r["accuracy_mean"]) for r in series],
                yerr=[100 * float(r["accuracy_std"]) for r in series],
                label=model,
                marker="o",
                capsize=3,
            )
        ax.set(title=f"{mode}-book QA", xlabel="Temperature", ylabel="Accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend()
    figure.tight_layout()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_file, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()

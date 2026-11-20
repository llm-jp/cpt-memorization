"""Compare memorization plots before and after excluding base-memorized samples."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


VARIANT_DISPLAY_NAMES = {"100pct": "GovOnly", "10pct": "GovMix"}
CONDITION_DISPLAY_NAMES = {
    "all": "All memorized samples (solid, circles)",
    "without_base": "Excluding base-memorized (dashed, open squares)",
}


def configure_font(font_file: Path | None) -> None:
    if font_file is None:
        return
    if not font_file.is_file():
        raise FileNotFoundError(f"Font file not found: {font_file}")
    font_manager.fontManager.addfont(str(font_file))
    font_name = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams["font.family"] = [font_name]


def prefix_length_from_path(path: str) -> int | None:
    match = re.search(r"prefix(\d+)", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def load_rows(plots_dir: Path, condition: str) -> list[dict[str, object]]:
    file_pattern = str(plots_dir / "memorization_rates_prefix*.csv")
    csv_files = sorted(glob.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching: {file_pattern}")

    rows: list[dict[str, object]] = []
    for csv_file in csv_files:
        prefix_length = prefix_length_from_path(csv_file)
        if prefix_length is None:
            continue
        with open(csv_file, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                variant = row.get("variant", "")
                rows.append(
                    {
                        "condition": condition,
                        "prefix_length": prefix_length,
                        "variant": variant,
                        "corpus": VARIANT_DISPLAY_NAMES.get(variant, variant),
                        "x_tokens_B": float(row["x_tokens_B"]),
                        "exact_rate": float(row["exact_rate"]),
                        "approx_rate": float(row["approx_rate"]),
                    }
                )
    if not rows:
        raise ValueError(f"No usable rows were loaded from {plots_dir}")
    return rows


def grouped_series(
    rows: Sequence[Mapping[str, object]],
    *,
    condition: str,
    corpus: str,
    prefix_length: int,
    metric: str,
) -> tuple[list[float], list[float]]:
    series = [
        row
        for row in rows
        if row["condition"] == condition
        and row["corpus"] == corpus
        and int(row["prefix_length"]) == prefix_length
    ]
    series = sorted(series, key=lambda row: float(row["x_tokens_B"]))
    return [float(row["x_tokens_B"]) for row in series], [float(row[metric]) for row in series]


def plot_comparison(rows: Sequence[Mapping[str, object]], output_file: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
            "legend.title_fontsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 3.0,
            "lines.markersize": 8.0,
            "axes.unicode_minus": False,
        }
    )

    prefix_lengths = sorted({int(row["prefix_length"]) for row in rows})
    corpora = [name for name in ("GovOnly", "GovMix") if any(row["corpus"] == name for row in rows)]
    corpora += sorted({str(row["corpus"]) for row in rows} - set(corpora))
    metrics = [("exact_rate", "Verbatim"), ("approx_rate", "Approximate")]
    conditions = ["all", "without_base"]
    condition_styles = {"all": "-", "without_base": (0, (7, 3))}
    condition_markers = {"all": "o", "without_base": "s"}

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        prefix: color_cycle[index % len(color_cycle)] for index, prefix in enumerate(prefix_lengths)
    }

    fig, axes = plt.subplots(
        len(metrics),
        len(corpora),
        figsize=(9.5 * max(len(corpora), 1), 22.0),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for row_index, (metric, metric_title) in enumerate(metrics):
        for col_index, corpus in enumerate(corpora):
            ax = axes[row_index][col_index]
            for prefix_length in prefix_lengths:
                for condition in conditions:
                    xs, ys = grouped_series(
                        rows,
                        condition=condition,
                        corpus=corpus,
                        prefix_length=prefix_length,
                        metric=metric,
                    )
                    if not xs:
                        continue
                    ax.plot(
                        xs,
                        ys,
                        color=colors[prefix_length],
                        linestyle=condition_styles[condition],
                        marker=condition_markers[condition],
                        markerfacecolor="white"
                        if condition == "without_base"
                        else colors[prefix_length],
                        markeredgewidth=1.8,
                    )
            ax.set_title(f"{corpus}: {metric_title}")
            ax.set_xlabel("Training tokens [B]")
            if col_index == 0:
                ax.set_ylabel("Memorization rate [%]")
            ax.grid(True, linestyle="--", alpha=0.35)

    prefix_handles = [
        Line2D([0], [0], color=colors[prefix], linewidth=4.0, label=f"N={prefix}")
        for prefix in prefix_lengths
    ]
    condition_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=condition_styles[condition],
            marker=condition_markers[condition],
            markerfacecolor="white" if condition == "without_base" else "#333333",
            markersize=9.5,
            markeredgewidth=1.8,
            linewidth=4.0,
            label=CONDITION_DISPLAY_NAMES[condition],
        )
        for condition in conditions
    ]
    prefix_legend = fig.legend(
        handles=prefix_handles,
        title="Snippet length",
        loc="lower center",
        ncol=min(len(prefix_handles), 5),
        bbox_to_anchor=(0.5, 0.045),
        handlelength=3.0,
    )
    fig.add_artist(prefix_legend)
    fig.legend(
        handles=condition_handles,
        title="Sample set",
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=6.0,
        columnspacing=3.0,
    )
    fig.suptitle("Memorization Rate: All vs Excluding Base-Memorized Samples", y=0.985, fontsize=22)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95), h_pad=4.0)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all-vs-without-base memorization rates from existing CSV files."
    )
    parser.add_argument("--all-plots-dir", type=Path, required=True)
    parser.add_argument("--without-base-plots-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--font-file", type=Path, default=None)
    args = parser.parse_args()

    configure_font(args.font_file)
    rows = load_rows(args.all_plots_dir, "all")
    rows.extend(load_rows(args.without_base_plots_dir, "without_base"))
    plot_comparison(rows, args.output_file)
    print(f"Saved comparison plot to {args.output_file}")


if __name__ == "__main__":
    main()

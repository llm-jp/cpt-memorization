"""Plot combined verbatim and approximate memorization ratios from CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


VARIANT_DISPLAY_NAMES = {"100pct": "GovOnly", "10pct": "GovMix"}
CORPUS_STYLES = {"GovOnly": "-", "GovMix": "--"}
CORPUS_MARKERS = {"GovOnly": "o", "GovMix": "X"}
CORPUS_COLORS = {"GovOnly": "#0072B2", "GovMix": "#D55E00"}
CORPUS_LIGHT_COLORS = {"GovOnly": "#d7e8f6", "GovMix": "#f6dfcf"}
METRIC_STYLES = {"exact_rate": "--", "approx_rate": "-"}
METRIC_MARKERS = {"exact_rate": "s", "approx_rate": "o"}


def configure_font(font_file: Path | None) -> None:
    if font_file is None:
        return
    if not font_file.is_file():
        raise FileNotFoundError(f"Font file not found: {font_file}")
    font_manager.fontManager.addfont(str(font_file))
    font_name = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams["font.family"] = [font_name]


def prefix_length_from_path(path: Path) -> int | None:
    match = re.search(r"prefix(\d+)", path.name)
    return int(match.group(1)) if match is not None else None


def load_rows(plots_dir: Path, prefix_lengths: set[int] | None = None) -> list[dict[str, object]]:
    csv_files = sorted(plots_dir.glob("memorization_rates_prefix*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found matching: {plots_dir / 'memorization_rates_prefix*.csv'}"
        )

    rows: list[dict[str, object]] = []
    for csv_file in csv_files:
        prefix_length = prefix_length_from_path(csv_file)
        if prefix_length is None or (
            prefix_lengths is not None and prefix_length not in prefix_lengths
        ):
            continue
        with csv_file.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                variant = row.get("variant", "")
                rows.append(
                    {
                        "prefix_length": prefix_length,
                        "corpus": VARIANT_DISPLAY_NAMES.get(variant, variant),
                        "x_tokens_B": float(row["x_tokens_B"]),
                        "exact_rate": float(row["exact_rate"]),
                        "approx_rate": float(row["approx_rate"]),
                    }
                )
    if not rows:
        raise ValueError("No plot rows matched the requested prefix lengths.")
    return rows


def color_gradient_by_prefix(
    prefix_lengths: Sequence[int],
    *,
    light_color: str,
    dark_color: str,
) -> dict[int, str]:
    ordered = sorted(set(prefix_lengths))
    light = np.array(to_rgb(light_color))
    dark = np.array(to_rgb(dark_color))
    prefix_range = max(ordered) - min(ordered)
    colors: dict[int, str] = {}
    for prefix in ordered:
        weight = 1.0 if prefix_range == 0 else (prefix - min(ordered)) / prefix_range
        colors[prefix] = to_hex(light * (1.0 - weight) + dark * weight)
    return colors


def blue_gradient_by_prefix(prefix_lengths: Sequence[int]) -> dict[int, str]:
    return color_gradient_by_prefix(
        prefix_lengths,
        light_color="#d7e8f6",
        dark_color="#1f77b4",
    )


def plot_combined(
    rows: Sequence[Mapping[str, object]],
    *,
    language: str = "en",
    color_scheme: str = "prefix",
    legend_location: str = "inside",
    title_layout: str = "single-line",
    panel_layout: str = "split",
    figsize: tuple[float, float] = (12.0, 4.8),
    font_scale: float = 1.3,
    line_width: float = 3.2,
    marker_size: float = 8.5,
) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13 * font_scale,
            "axes.labelsize": 15 * font_scale,
            "axes.titlesize": 16 * font_scale,
            "legend.fontsize": 13 * font_scale,
            "legend.title_fontsize": 13 * font_scale,
            "xtick.labelsize": 12 * font_scale,
            "ytick.labelsize": 12 * font_scale,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.9,
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "lines.markeredgewidth": 1.2,
        }
    )

    prefix_lengths = sorted({int(row["prefix_length"]) for row in rows})
    corpora = sorted({str(row["corpus"]) for row in rows})
    palette = plt.get_cmap("tab10")
    corpus_color_map = {
        name: CORPUS_COLORS.get(name, to_hex(palette(index % 10)))
        for index, name in enumerate(corpora)
    }
    corpus_style_map = {
        name: CORPUS_STYLES.get(name, ["-", "--", ":", "-."][index % 4])
        for index, name in enumerate(corpora)
    }
    corpus_marker_map = {
        name: CORPUS_MARKERS.get(name, ["o", "s", "^", "D", "X"][index % 5])
        for index, name in enumerate(corpora)
    }
    colors = blue_gradient_by_prefix(prefix_lengths)
    corpus_colors = {
        corpus: color_gradient_by_prefix(
            prefix_lengths,
            light_color=CORPUS_LIGHT_COLORS.get(corpus, "#eeeeee"),
            dark_color=dark_color,
        )
        for corpus, dark_color in corpus_color_map.items()
    }
    corpus_values = {str(row["corpus"]) for row in rows}
    corpora = [name for name in ("GovOnly", "GovMix") if name in corpus_values]
    corpora += sorted(corpus_values - set(corpora))
    is_english = language == "en"
    metrics = (
        [
            ("exact_rate", "Memorization Ratio (Verbatim)", "Verbatim"),
            ("approx_rate", "Memorization Ratio (Approximate)", "Approximate"),
        ]
        if is_english
        else [
            ("exact_rate", "暗記率（逐語的暗記）", "逐語的暗記"),
            ("approx_rate", "暗記率（近似的暗記）", "近似的暗記"),
        ]
    )
    single_prefix = len(prefix_lengths) == 1

    if panel_layout == "combined":
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        plot_groups = [(ax, metric, title) for metric, title, _ in metrics]
    else:
        fig, split_axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
        axes = list(split_axes)
        plot_groups = [(ax, metric, title) for ax, (metric, title, _) in zip(axes, metrics)]

    for ax, metric, title in plot_groups:
        for prefix_length in prefix_lengths:
            for corpus in corpora:
                series = sorted(
                    (
                        row
                        for row in rows
                        if int(row["prefix_length"]) == prefix_length
                        and str(row["corpus"]) == corpus
                    ),
                    key=lambda row: float(row["x_tokens_B"]),
                )
                if not series:
                    continue
                ax.plot(
                    [float(row["x_tokens_B"]) for row in series],
                    [float(row[metric]) for row in series],
                    color=(
                        corpus_colors.get(corpus, colors)[prefix_length]
                        if color_scheme == "corpus-prefix" or single_prefix
                        else colors[prefix_length]
                    ),
                    linestyle=(
                        METRIC_STYLES[metric]
                        if panel_layout == "combined"
                        else corpus_style_map[corpus]
                    ),
                    marker=(
                        METRIC_MARKERS[metric]
                        if panel_layout == "combined"
                        else corpus_marker_map[corpus]
                    ),
                )
        if panel_layout != "combined":
            if title_layout == "stacked":
                title = title.replace(" (", "\n(", 1).replace("（", "\n（", 1)
            ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("Training Tokens [B]" if is_english else "学習トークン数 [B]")
        ax.set_ylabel("Memorization Ratio [%]" if is_english else "暗記率 [%]")
    if panel_layout == "combined":
        axes[0].set_title("Memorization Ratio" if is_english else "暗記率")

    # A prefix-length key is redundant for a single-prefix plot. In that case,
    # color the corpus keys like the plotted lines so the legend is self-contained.
    corpus_handles = [
        Line2D(
            [0],
            [0],
            color=(
                corpus_color_map[corpus]
                if color_scheme == "corpus-prefix" or single_prefix
                else "#333333"
            ),
            linestyle=("-" if panel_layout == "combined" else corpus_style_map[corpus]),
            marker=(None if panel_layout == "combined" else corpus_marker_map[corpus]),
            linewidth=line_width,
            markersize=marker_size,
            label=corpus,
        )
        for corpus in corpora
    ]
    prefix_handles = (
        [
            Line2D(
                [0],
                [0],
                color=colors[prefix],
                linewidth=line_width,
                label=f"N={prefix}",
            )
            for prefix in reversed(prefix_lengths)
        ]
        if not single_prefix
        else []
    )
    prefix_title = "Text Segment Length" if is_english else "テキスト断片長"
    corpus_title = "CPT Corpus" if is_english else "CPTコーパス"
    metric_title = "Memorization Type" if is_english else "暗記種別"
    metric_handles = (
        [
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle=METRIC_STYLES[metric],
                marker=METRIC_MARKERS[metric],
                linewidth=line_width,
                markersize=marker_size,
                label=short_label,
            )
            for metric, _, short_label in metrics
        ]
        if panel_layout == "combined"
        else []
    )
    combined_legend_title = (
        f"{prefix_title} /\n{corpus_title}"
        if legend_location == "outside-right"
        else f"{prefix_title} / {corpus_title}"
    )
    legend_ax = axes[-1] if legend_location == "outside-right" else axes[0]
    legend_kwargs = (
        {
            "loc": "upper left",
            "bbox_to_anchor": (1.02, 1.0),
            "borderaxespad": 0.0,
        }
        if legend_location == "outside-right"
        else {"loc": "center right" if panel_layout == "combined" else "upper left"}
    )
    if legend_location in {"outside-right", "below"}:
        # Finalize the two panels before adding the external legend so their
        # dimensions remain identical to the inside-legend layout.
        fig.tight_layout()
    legend_handles = prefix_handles + corpus_handles + metric_handles
    if metric_handles:
        legend_title = " / ".join(
            title
            for title, handles in (
                (prefix_title, prefix_handles),
                (corpus_title, corpus_handles),
                (metric_title, metric_handles),
            )
            if handles
        )
    else:
        legend_title = combined_legend_title if prefix_handles else corpus_title
    if legend_location == "below":
        legend = fig.legend(
            handles=legend_handles,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(legend_handles)),
            handlelength=2.4,
            frameon=True,
        )
    else:
        legend = legend_ax.legend(
            handles=legend_handles,
            title=legend_title,
            handlelength=2.4,
            frameon=True,
            ncol=2 if panel_layout == "combined" and legend_location == "inside" else 1,
            **legend_kwargs,
        )
    legend._legend_box.align = "left"
    legend.get_title().set_ha("left")
    if legend_location == "inside":
        fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a combined memorization-ratio plot from per-prefix CSV files."
    )
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--language", choices=["ja", "en"], default="en")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=None)
    parser.add_argument(
        "--color-scheme",
        choices=["prefix", "corpus-prefix"],
        default="prefix",
        help=(
            "Color by snippet length only, or use corpus hue and snippet-length "
            "lightness (default: prefix)."
        ),
    )
    parser.add_argument(
        "--legend-location",
        choices=["inside", "outside-right", "below"],
        default="inside",
        help="Place the legend inside, outside the right panel, or below both panels.",
    )
    parser.add_argument(
        "--title-layout",
        choices=["single-line", "stacked"],
        default="single-line",
        help="Keep panel titles on one line or stack the metric qualifier below it.",
    )
    parser.add_argument(
        "--panel-layout",
        choices=["split", "combined"],
        default="split",
        help=(
            "Draw verbatim and approximate ratios in separate panels, or overlay "
            "them in one larger panel (default: split)."
        ),
    )
    parser.add_argument("--font-file", type=Path, default=None)
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(12.0, 4.8),
        help="Figure size in inches (default: 12 4.8).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.3,
        help="Scale all font sizes uniformly (default: 1.3).",
    )
    parser.add_argument("--line-width", type=float, default=3.2)
    parser.add_argument("--marker-size", type=float, default=8.5)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    configure_font(args.font_file)
    requested_prefixes = set(args.prefix_lengths) if args.prefix_lengths is not None else None
    rows = load_rows(args.plots_dir, requested_prefixes)
    fig = plot_combined(
        rows,
        language=args.language,
        color_scheme=args.color_scheme,
        legend_location=args.legend_location,
        title_layout=args.title_layout,
        panel_layout=args.panel_layout,
        figsize=tuple(args.figsize),
        font_scale=args.font_scale,
        line_width=args.line_width,
        marker_size=args.marker_size,
    )
    output_file = args.output_file or args.plots_dir / "memorization_rates_combined.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {output_file}")


if __name__ == "__main__":
    main()

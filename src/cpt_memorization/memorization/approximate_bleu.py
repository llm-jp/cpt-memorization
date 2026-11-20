import argparse
import csv
import json
import logging
import math
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set
import concurrent.futures


VARIANT_DISPLAY_NAME = {"100pct": "100%", "10pct": "10%"}
VARIANT_COLORS = {"100pct": "#1f77b4", "10pct": "#ff7f0e"}
DEFAULT_TOKENS_PER_ITER_B = (4096 * 512) / 1_000_000_000


def _format_percent_decimal(value: float) -> str:
    """Return threshold like 0.75 -> '.75'."""
    return f"{value:.2f}".lstrip("0")


def _extract_iteration_from_name(file_path: Path) -> Optional[int]:
    match = re.search(r"iter_(\d+)", file_path.stem)
    if not match:
        return None
    return int(match.group(1))


from .analysis.metrics import token_bleu_score


def compute_bleu(reference, candidate) -> float:
    """Unsmoothed effective-order token-ID BLEU on the 0-1 scale."""
    return token_bleu_score(reference, candidate)


def _sample_key(sample: Dict) -> str:
    payload = {
        "doc_id": sample.get("doc_id") or sample.get("archive_member"),
        "snippet_length": sample.get("snippet_length"),
        "prefix_token_ids": sample.get("prefix_token_ids"),
        "suffix_token_ids": sample.get("suffix_token_ids"),
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def _iter_samples(file_path: Path, prefix_length: Optional[int]):
    with file_path.open("rt", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            snippet_length = sample.get("snippet_length")
            if prefix_length is not None and snippet_length != prefix_length:
                continue
            yield sample


def _summarize_checkpoint_file(
    file_path: Path,
    prefix_length: Optional[int],
    bleu_threshold: float,
    allowed_keys: Optional[Set[str]] = None,
    collect_keys: bool = False,
) -> Dict[str, float]:
    summary = {
        "checkpoint": file_path.stem,
        "total": 0,
        "total_seen": 0,
        "skipped": 0,
        "approx": 0,
        "exact": 0,
        "iteration": _extract_iteration_from_name(file_path),
    }
    sample_keys: Optional[Set[str]] = set() if collect_keys else None

    for sample in _iter_samples(file_path, prefix_length):
        summary["total_seen"] += 1
        key: Optional[str] = None
        if allowed_keys is not None or collect_keys:
            key = _sample_key(sample)
        if sample_keys is not None and key is not None:
            sample_keys.add(key)
        if allowed_keys is not None and key is not None and key not in allowed_keys:
            summary["skipped"] += 1
            continue

        suffix_token_ids = sample["suffix_token_ids"]
        predicted_token_ids = sample["predicted_token_ids"]

        summary["total"] += 1

        if suffix_token_ids == predicted_token_ids:
            summary["exact"] += 1

        bleu = compute_bleu(suffix_token_ids, predicted_token_ids)
        if bleu >= bleu_threshold:
            summary["approx"] += 1

    total = summary["total"]
    summary["exact_rate"] = summary["exact"] / total * 100.0 if total else 0.0
    summary["approx_rate"] = summary["approx"] / total * 100.0 if total else 0.0
    if sample_keys is not None:
        summary["sample_keys"] = sample_keys
    return summary


def _aggregate_checkpoint_stats(
    completions_root: Path,
    variants: Sequence[str],
    prefix_length: Optional[int],
    bleu_threshold: float,
    allowed_keys: Optional[Set[str]] = None,
    collect_keys: bool = False,
    workers: int = 1,
) -> Dict[str, List[Dict[str, float]]]:
    all_stats: Dict[str, List[Dict[str, float]]] = {}

    def _sort_key(summary: Dict[str, float], fallback_name: str) -> Tuple[float, str]:
        iteration = summary.get("iteration")
        if iteration is None:
            return (float("inf"), fallback_name)
        return (float(iteration), fallback_name)

    # 平坦なタスクリストを作成 (variant, checkpoint_path)
    tasks = []
    for variant in variants:
        variant_dir = completions_root / variant
        if not variant_dir.is_dir():
            logging.warning("Variant directory not found: %s", variant_dir)
            continue

        checkpoints = sorted(variant_dir.glob("*.jsonl"))
        if not checkpoints:
            logging.warning("No completion files found under %s", variant_dir)
            continue

        for cp in checkpoints:
            tasks.append((variant, cp))

    # 並列実行
    results: List[Tuple[str, Dict[str, float]]] = []

    # 進行状況表示用
    total_tasks = len(tasks)
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _summarize_checkpoint_file,
                cp,
                prefix_length,
                bleu_threshold,
                allowed_keys,
                collect_keys,
            ): (var, cp)
            for var, cp in tasks
        }

        for future in concurrent.futures.as_completed(future_map):
            variant, cp = future_map[future]
            try:
                summary = future.result()
                results.append((variant, summary))
                completed += 1
                if completed % 10 == 0 or completed == total_tasks:
                    logging.info(f"Processed {completed}/{total_tasks} files...")
            except Exception as e:
                logging.error(f"Error processing {cp}: {e}")

    # 集計結果を辞書に戻す
    temp_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for variant, summary in results:
        temp_stats[variant].append(summary)
        # ログ出力（順序はバラバラになるため、ここでは詳細ログは省略するか、控えめにする）
        if summary.get("skipped"):
            logging.info(
                "Variant %s checkpoint %s: skipped %d/%d samples",
                variant,
                summary["checkpoint"],
                summary["skipped"],
                summary["total_seen"],
            )

    # ソートして格納
    for variant, stats in temp_stats.items():
        stats.sort(
            key=lambda entry: _sort_key(entry, str(entry.get("checkpoint", ""))),
        )
        all_stats[variant] = stats

    return all_stats


def _inject_shared_baseline(
    variant_stats: Dict[str, List[Dict[str, float]]],
    baseline_variant: str = "0B",
) -> None:
    """
    baseline_variant (例: "0B") が他の variant と同一モデルである場合に、
    baseline の checkpoint を各 variant の先頭に注入して x=baseline から開始させる。
    """
    if baseline_variant not in variant_stats:
        return

    baseline_stats = variant_stats.get(baseline_variant) or []
    if not baseline_stats:
        return

    baseline_entry = min(
        baseline_stats,
        key=lambda entry: float(entry["iteration"])
        if entry.get("iteration") is not None
        else float("inf"),
    )
    baseline_iter = baseline_entry.get("iteration")
    if baseline_iter is None:
        return

    for variant, stats in list(variant_stats.items()):
        if variant == baseline_variant:
            continue
        if any(entry.get("iteration") == baseline_iter for entry in stats):
            continue
        injected = dict(baseline_entry)
        injected["checkpoint"] = baseline_entry.get("checkpoint", "baseline")
        injected["iteration"] = baseline_iter
        stats.append(injected)
        stats.sort(
            key=lambda entry: (
                float(entry["iteration"]) if entry.get("iteration") is not None else float("inf"),
                str(entry.get("checkpoint", "")),
            )
        )
        variant_stats[variant] = stats


def _parse_checkpoint_labels(raw: Optional[str]) -> Optional[List[Tuple[Optional[float], str]]]:
    if not raw:
        return None

    labels: List[Tuple[Optional[float], str]] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        value_match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
        value = float(value_match[0]) if value_match else None
        labels.append((value, text))

    return labels or None


def _plot_checkpoint_memorization(
    variant_stats: Dict[str, List[Dict[str, float]]],
    labels: Optional[List[Tuple[Optional[float], str]]],
    prefix_length: Optional[int],
    bleu_threshold: float,
    plot_path: Path,
    y_min: Optional[float],
    y_max: Optional[float],
    tokens_per_iteration: float,
    x_offset: Optional[float],
    x_max_override: Optional[float],
    min_spacing_b: Optional[float],
    fit_to_xmax: bool,
    csv_path: Optional[Path] = None,
) -> None:
    if not variant_stats:
        logging.warning("No checkpoint statistics available for plotting.")
        return

    max_points = max(len(stats) for stats in variant_stats.values())
    if max_points == 0:
        logging.warning("Checkpoint statistics are empty; skipping plot generation.")
        return

    numeric_labels = labels if labels else None
    xtick_positions: List[float] = []
    xtick_texts: List[str] = []
    if numeric_labels:
        for value, text in numeric_labels:
            if value is None:
                continue
            xtick_positions.append(value)
            xtick_texts.append(text)

    computed_offset = x_offset
    if computed_offset is None:
        if numeric_labels and numeric_labels[0][0] is not None:
            computed_offset = numeric_labels[0][0]
        else:
            computed_offset = 0.0

    iterations = [
        entry["iteration"]
        for stats in variant_stats.values()
        for entry in stats
        if entry.get("iteration") is not None
    ]
    iteration_min = min(iterations) if iterations else None
    iteration_max = max(iterations) if iterations else None

    effective_tokens_per_iteration = tokens_per_iteration
    if (
        fit_to_xmax
        and x_max_override is not None
        and iteration_min is not None
        and iteration_max is not None
        and iteration_max > iteration_min
    ):
        target_span = x_max_override - computed_offset
        source_span = (iteration_max - iteration_min) * tokens_per_iteration
        if target_span > 0 and source_span > 0:
            effective_tokens_per_iteration = target_span / (iteration_max - iteration_min)
            logging.info(
                "Fitting tokens/iter to x_max=%.2fB (%.6f -> %.6f)",
                x_max_override,
                tokens_per_iteration,
                effective_tokens_per_iteration,
            )
        else:
            logging.warning(
                "Unable to fit x-axis span (target_span=%.3f, source_span=%.3f)",
                target_span,
                source_span,
            )

    def _compute_x(entry: Dict[str, float], fallback_index: int) -> float:
        iteration_value = entry.get("iteration")
        if iteration_min is not None and iteration_value is not None:
            return (
                iteration_value - iteration_min
            ) * effective_tokens_per_iteration + computed_offset
        return float(fallback_index)

    processed_points: Dict[str, List[Dict[str, float]]] = {}
    for variant, stats in variant_stats.items():
        variant_points: List[Dict[str, float]] = []
        for idx, entry in enumerate(stats):
            variant_points.append(
                {
                    "x": _compute_x(entry, idx),
                    "approx_rate": entry["approx_rate"],
                    "exact_rate": entry["exact_rate"],
                }
            )
        processed_points[variant] = variant_points

    def _apply_spacing(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if not points or min_spacing_b is None or min_spacing_b <= 0:
            return points
        filtered: List[Dict[str, float]] = []
        last_kept_x: Optional[float] = None
        for point in points:
            x_val = point["x"]
            if last_kept_x is None or x_val is None:
                filtered.append(point)
                last_kept_x = x_val
                continue
            if x_val - last_kept_x >= min_spacing_b - 1e-9:
                filtered.append(point)
                last_kept_x = x_val
        if points and filtered and filtered[-1] is not points[-1]:
            filtered.append(points[-1])
        return filtered

    for variant, points in processed_points.items():
        filtered = _apply_spacing(points)
        if len(filtered) < len(points):
            logging.info(
                "Variant %s: downsampled checkpoints from %d to %d (spacing %.1fB)",
                variant,
                len(points),
                len(filtered),
                min_spacing_b,
            )
        processed_points[variant] = filtered

    if csv_path:
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["variant", "x_tokens_B", "exact_rate", "approx_rate"])
                for variant, points in processed_points.items():
                    for p in points:
                        writer.writerow([variant, p["x"], p["exact_rate"], p["approx_rate"]])
            logging.info("Checkpoint data saved to CSV: %s", csv_path)
        except Exception as e:
            logging.error("Failed to save CSV: %s", e)

    all_rates: List[float] = []
    all_xs: List[float] = []
    for points in processed_points.values():
        for point in points:
            all_rates.append(point["approx_rate"])
            all_rates.append(point["exact_rate"])
            all_xs.append(point["x"])

    if not all_rates:
        logging.warning("No memorization rates found; skipping plot generation.")
        return

    if not all_xs:
        logging.warning("Iteration metadata missing; cannot position checkpoints accurately.")
        return

    auto_min = min(all_rates)
    auto_max = max(all_rates)
    span = max(auto_max - auto_min, 1e-6)
    auto_margin = max(0.05 * span, 0.1)

    calculated_max = min(100.0, auto_max + auto_margin)

    if y_min is None:
        y_min = max(0.0, auto_min - auto_margin)

    if y_max is None:
        y_max = calculated_max
    elif y_max < calculated_max:
        logging.info(
            "Adjusting Y-axis max from requested %.2f to %.2f to fit data.", y_max, calculated_max
        )
        y_max = calculated_max

    if y_min >= y_max:
        y_max = y_min + max(span, 0.1)

    plt.figure(figsize=(10, 5))
    x_min = min(all_xs)
    x_max = max(all_xs)

    for variant, points in processed_points.items():
        approx_rates = [point["approx_rate"] for point in points]
        exact_rates = [point["exact_rate"] for point in points]
        xs = [point["x"] for point in points]
        color = VARIANT_COLORS.get(variant)
        display_name = VARIANT_DISPLAY_NAME.get(variant, variant)
        hide_legend = variant == "0B" and len(points) <= 1
        approx_label = (
            "_nolegend_"
            if hide_legend
            else f"{display_name} approx (BLEU ≥ {_format_percent_decimal(bleu_threshold)})"
        )
        exact_label = "_nolegend_" if hide_legend else f"{display_name} exact"

        plt.plot(
            xs,
            approx_rates,
            label=approx_label,
            marker="o",
            color=color,
        )
        plt.plot(
            xs,
            exact_rates,
            label=exact_label,
            marker="s",
            linestyle="--",
            color=color,
            alpha=0.9,
        )

    if xtick_positions and xtick_texts:
        plt.xticks(xtick_positions, xtick_texts, rotation=45)
    plt.ylabel("Memorization rate [%]")
    title_parts = ["Memorization vs checkpoints"]
    if prefix_length is not None:
        title_parts.append(f"prefix length={prefix_length}")
    plt.title(" – ".join(title_parts))
    plt.ylim(y_min, y_max)
    if x_max_override is not None:
        upper = max(x_max_override, x_max)
        plt.xlim(x_min, upper)
    else:
        margin = max(0.01 * (x_max - x_min), 0.5)
        plt.xlim(x_min, x_max + margin)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    logging.info("Checkpoint plot saved to %s", plot_path)


def main():
    parser = argparse.ArgumentParser(description="Approximate memorization analysis with BLEU")
    parser.add_argument(
        "--result-file",
        type=str,
        required=False,
        help="generate.py が出力した結果 JSONL ファイル",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        help="BLEU スコア等を追記した JSONL の出力先",
    )
    parser.add_argument(
        "--bleu-threshold",
        type=float,
        default=0.75,
        help="Approximate memorization と扱う BLEU のしきい値（0〜1）",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        required=False,
        help="snippet_length ごとの近似記憶率をプロットする PNG ファイルパス",
    )
    parser.add_argument(
        "--completions-root",
        type=str,
        required=False,
        help="checkpoint ごとの completions JSONL が格納されたルートディレクトリ",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["100pct", "10pct"],
        help="completions-root 配下で解析するバリアントのディレクトリ名",
    )
    parser.add_argument(
        "--baseline-variant",
        type=str,
        default=None,
        help="共通ベースラインとして注入するバリアント名 (例: 0B)。未指定なら注入しない。",
    )
    parser.add_argument(
        "--allow-sample-mismatch",
        action="store_true",
        help="ベースラインとのサンプル不一致を許可し、ベースライン集合に揃えて集計する。",
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=1000,
        help="checkpoint 集計時にフィルタする snippet_length (<=0 の場合は無視)",
    )
    parser.add_argument(
        "--checkpoint-plot-file",
        type=str,
        required=False,
        help="checkpoint ごとの exact/approx 暗記率を描画する PNG パス",
    )
    parser.add_argument(
        "--checkpoint-labels",
        type=str,
        required=False,
        help="checkpoint グラフの x 軸に用いるラベル（カンマ区切り）",
    )
    parser.add_argument(
        "--checkpoint-ymin",
        type=float,
        required=False,
        help="checkpoint グラフの y 軸下限 (未指定時は自動調整)",
    )
    parser.add_argument(
        "--checkpoint-ymax",
        type=float,
        required=False,
        help="checkpoint グラフの y 軸上限 (未指定時は自動調整)",
    )
    parser.add_argument(
        "--tokens-per-iteration",
        type=float,
        default=DEFAULT_TOKENS_PER_ITER_B,
        help="1 iteration あたりのトークン数（10^9 tokens 単位）",
    )
    parser.add_argument(
        "--checkpoint-xoffset",
        type=float,
        required=False,
        help="x 座標に加えるオフセット (B)。未指定時はラベルの先頭値を使用",
    )
    parser.add_argument(
        "--checkpoint-xmax",
        type=float,
        required=False,
        help="checkpoint グラフの x 軸上限 (B)。未指定ならデータから自動推定",
    )
    parser.add_argument(
        "--checkpoint-min-spacing",
        type=float,
        required=False,
        help="プロットする checkpoint 間の最小間隔 (B)。指定すると飛び飛び表示",
    )
    parser.add_argument(
        "--checkpoint-csv-file",
        type=str,
        required=False,
        help="プロットデータ（間引き後）を保存する CSV ファイルパス",
    )
    parser.add_argument(
        "--checkpoint-fit-to-xmax",
        action="store_true",
        help="--checkpoint-xmax に最終 checkpoint が一致するよう tokens/iter を自動スケーリング",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列処理数 (デフォルト: 1)",
    )
    args = parser.parse_args()
    if not 0 <= args.bleu_threshold <= 1:
        parser.error("BLEU threshold must be on the 0-1 scale; use 0.75, not 75")
    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("BLEU による approximate memorization を計算します。")

    if (args.result_file is None) ^ (args.output_file is None):
        parser.error("--result-file と --output-file は同時に指定してください。")

    if args.result_file and args.output_file:
        # snippet_length ごとの統計
        stats = defaultdict(lambda: {"total": 0, "approx": 0})

        with (
            open(args.result_file, "rt", encoding="utf-8") as fin,
            open(args.output_file, "wt", encoding="utf-8") as fout,
        ):
            for line in fin:
                if not line.strip():
                    continue
                sample = json.loads(line)

                suffix_token_ids = sample["suffix_token_ids"]
                predicted_token_ids = sample["predicted_token_ids"]

                bleu = compute_bleu(suffix_token_ids, predicted_token_ids)
                sample["bleu"] = bleu
                sample["is_memorized_approx_bleu"] = bleu >= args.bleu_threshold

                # snippet_length ごとの集計
                sl = sample["snippet_length"]
                stats[sl]["total"] += 1
                if sample["is_memorized_approx_bleu"]:
                    stats[sl]["approx"] += 1

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # ログ出力
        for sl in sorted(stats.keys()):
            total = stats[sl]["total"]
            approx = stats[sl]["approx"]
            rate = approx / total * 100.0 if total > 0 else 0.0
            logging.info(
                f"snippet_length={sl}: approximate memorization {approx}/{total} ({rate:.2f}%) "
                f"(BLEU ≥ {args.bleu_threshold})"
            )

        # プロットが指定されていれば図を保存
        if args.plot_file:
            snippet_lengths = sorted(stats.keys())
            rates = [
                stats[sl]["approx"] / stats[sl]["total"] * 100.0 if stats[sl]["total"] > 0 else 0.0
                for sl in snippet_lengths
            ]

            plt.figure()
            plt.plot(snippet_lengths, rates, marker="o")
            for x, y in zip(snippet_lengths, rates):
                plt.text(x, y, f"{y:.1f}%", ha="center", va="bottom")

            plt.xlabel("snippet_length (tokens)")
            plt.ylabel(
                f"Approximate memorization rate (BLEU ≥ {_format_percent_decimal(args.bleu_threshold)}) [%]"
            )
            plt.title("Approximate memorization vs prefix length (BLEU-based)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(args.plot_file, dpi=200)

            logging.info(f"プロットを {args.plot_file} に保存しました。")

    if args.checkpoint_plot_file:
        if not args.completions_root:
            parser.error("--completions-root を指定してください。")

        completions_root = Path(args.completions_root)
        if not completions_root.is_dir():
            parser.error(f"Completions root not found: {completions_root}")

        prefix_length = args.prefix_length
        if prefix_length is not None and prefix_length <= 0:
            prefix_length = None

        baseline_variant = args.baseline_variant
        baseline_stats: Dict[str, List[Dict[str, float]]] = {}
        baseline_keys: Optional[Set[str]] = None
        if baseline_variant:
            baseline_stats = _aggregate_checkpoint_stats(
                completions_root,
                [baseline_variant],
                prefix_length,
                args.bleu_threshold,
                collect_keys=True,
                workers=args.workers,
            )
            baseline_entries = baseline_stats.get(baseline_variant, [])
            if baseline_entries:
                baseline_keys = baseline_entries[0].get("sample_keys")
            if not baseline_keys:
                logging.warning("Baseline variant has no samples: %s", baseline_variant)

        stats_by_variant = _aggregate_checkpoint_stats(
            completions_root,
            args.variants,
            prefix_length,
            args.bleu_threshold,
            allowed_keys=baseline_keys
            if (baseline_variant and args.allow_sample_mismatch)
            else None,
            collect_keys=bool(baseline_variant),
            workers=args.workers,
        )

        if baseline_variant and baseline_keys and not args.allow_sample_mismatch:
            for variant, stats in stats_by_variant.items():
                for entry in stats:
                    sample_keys = entry.get("sample_keys")
                    if sample_keys is None:
                        continue
                    if sample_keys != baseline_keys:
                        missing = len(baseline_keys - sample_keys)
                        extra = len(sample_keys - baseline_keys)
                        raise ValueError(
                            f"Sample mismatch in {variant}/{entry['checkpoint']}: "
                            f"missing={missing} extra={extra}. "
                            "Regenerate completions with the same samples.jsonl, "
                            "or re-run with --allow-sample-mismatch to align."
                        )

        if baseline_variant and baseline_stats:
            if baseline_variant not in stats_by_variant:
                stats_by_variant.update(baseline_stats)
            _inject_shared_baseline(stats_by_variant, baseline_variant=baseline_variant)
            if baseline_variant not in args.variants:
                stats_by_variant.pop(baseline_variant, None)
        labels = _parse_checkpoint_labels(args.checkpoint_labels)
        plot_path = Path(args.checkpoint_plot_file)
        _plot_checkpoint_memorization(
            stats_by_variant,
            labels,
            prefix_length,
            args.bleu_threshold,
            plot_path,
            args.checkpoint_ymin,
            args.checkpoint_ymax,
            args.tokens_per_iteration,
            args.checkpoint_xoffset,
            args.checkpoint_xmax,
            args.checkpoint_min_spacing,
            args.checkpoint_fit_to_xmax,
            csv_path=Path(args.checkpoint_csv_file) if args.checkpoint_csv_file else None,
        )


if __name__ == "__main__":
    main()

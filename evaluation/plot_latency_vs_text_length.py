#!/usr/bin/env python3
"""Plot the retained CUDA fast-path latency sweep against text length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUCKETS = ["01", "02", "03-04", "05-08", "09-16", "17-32", "33-64", "65+"]
LABELS = ["1", "2", "3–4", "5–8", "9–16", "17–32", "33–64", "65+"]
COLORS = {
    "MangaDReC": "#2563EB",
    "MangaDReCo": "#7C3AED",
    "BaberuOCR": "#EA580C",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fast_path_series(report: dict) -> dict:
    grouped = report["by_normalized_reference_length_bucket"]
    return {
        "p50": np.asarray([grouped[bucket]["median_ms"] for bucket in BUCKETS]),
        "p90": np.asarray([grouped[bucket]["p90_ms"] for bucket in BUCKETS]),
        "count": np.asarray([grouped[bucket]["samples"] for bucket in BUCKETS]),
        "overall_p50": float(report["latency"]["median_ms"]),
        "slope": float(report["latency"]["linear_slope_ms_per_reference_character"]),
    }


def baberu_series(rows: list[dict], drec_rows: list[dict]) -> dict:
    bucket_by_id = {row["sample_id"]: row["length_bucket"] for row in drec_rows}
    normalized_length_by_id = {
        row["sample_id"]: row["normalized_reference_char_count"] for row in drec_rows
    }
    grouped: dict[str, list[float]] = {bucket: [] for bucket in BUCKETS}
    x, y = [], []
    for row in rows:
        sample_id = row["id"]
        grouped[bucket_by_id[sample_id]].append(row["end_to_end_single_image_ms"])
        x.append(normalized_length_by_id[sample_id])
        y.append(row["end_to_end_single_image_ms"])
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    within_64 = (x_array >= 1) & (x_array <= 64)
    slope, _intercept = np.polyfit(x_array[within_64], y_array[within_64], 1)
    return {
        "p50": np.asarray([np.median(grouped[bucket]) for bucket in BUCKETS]),
        "p90": np.asarray([np.quantile(grouped[bucket], 0.90) for bucket in BUCKETS]),
        "count": np.asarray([len(grouped[bucket]) for bucket in BUCKETS]),
        "overall_p50": float(np.median(y_array)),
        "slope": float(slope),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drec-fast-report", type=Path, required=True)
    parser.add_argument("--dreco-fast-report", type=Path, required=True)
    parser.add_argument("--drec-reference-rows", type=Path, required=True)
    parser.add_argument("--baberu-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    drec_report = load_json(args.drec_fast_report)
    dreco_report = load_json(args.dreco_fast_report)
    drec_rows = load_jsonl(args.drec_reference_rows)
    baberu_rows = load_jsonl(args.baberu_rows)
    if len(drec_rows) != 8_000 or len(baberu_rows) != 8_000:
        raise ValueError("expected the same 8,000-crop benchmark for DReC and BaberuOCR")

    series = {
        "MangaDReC": fast_path_series(drec_report),
        "MangaDReCo": fast_path_series(dreco_report),
        "BaberuOCR": baberu_series(baberu_rows, drec_rows),
    }
    if not np.array_equal(series["MangaDReC"]["count"], series["BaberuOCR"]["count"]):
        raise ValueError("length-bucket counts do not match across systems")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    figure = plt.figure(figsize=(13.5, 7.4), dpi=160, facecolor="#F4F7FB")
    grid = figure.add_gridspec(2, 1, height_ratios=[5.4, 1.0], hspace=0.08)
    axis = figure.add_subplot(grid[0])
    count_axis = figure.add_subplot(grid[1], sharex=axis)
    axis.set_facecolor("white")
    count_axis.set_facecolor("white")
    positions = np.arange(len(BUCKETS), dtype=np.float64)

    for name, values in series.items():
        color = COLORS[name]
        axis.fill_between(
            positions,
            values["p50"],
            values["p90"],
            color=color,
            alpha=0.085,
            linewidth=0,
        )
        axis.plot(
            positions,
            values["p50"],
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=1.1,
            label=(
                f"{name}  ·  overall P50 {values['overall_p50']:.1f} ms"
                f"  ·  {values['slope']:+.2f} ms/char"
            ),
        )

    axis.annotate(
        "CUDA fast path remains almost flat",
        xy=(5, series["MangaDReCo"]["p50"][5]),
        xytext=(4.2, 101),
        color="#4338CA",
        fontsize=11,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": "#7C3AED", "lw": 1.2},
    )
    axis.annotate(
        "autoregressive decoding grows with length",
        xy=(6, series["BaberuOCR"]["p50"][6]),
        xytext=(3.6, 284),
        color="#C2410C",
        fontsize=11,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": "#EA580C", "lw": 1.2},
    )
    axis.set_ylim(0, 430)
    axis.set_ylabel("End-to-end latency per crop (ms)", labelpad=10)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelbottom=False)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        framealpha=0.96,
    )

    count_axis.bar(
        positions,
        series["MangaDReC"]["count"],
        width=0.62,
        color="#CBD5E1",
        edgecolor="none",
    )
    count_axis.set_yscale("log")
    count_axis.set_ylabel("crops\n(log)", labelpad=9)
    count_axis.set_xlabel("Normalized reference text length (characters)")
    count_axis.set_xticks(positions, LABELS)
    count_axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    count_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "CUDA fast path: latency remains stable as text grows",
        x=0.075,
        y=0.967,
        ha="left",
        fontsize=21,
        weight="bold",
        color="#0F172A",
    )
    figure.text(
        0.076,
        0.916,
        "NVIDIA A10 · batch 1 · 8,000 Manga109s text crops · warmed · file I/O excluded",
        color="#64748B",
        fontsize=11.5,
    )
    figure.text(
        0.985,
        0.018,
        "Line: P50  |  shade: P50–P90  |  65+ contains 7 crops",
        ha="right",
        color="#64748B",
        fontsize=9.5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    if args.summary:
        payload = {
            "schema": "mangadrec_cuda_a10_fast_path_text_length_plot_v1",
            "samples": 8_000,
            "length_axis": "normalized reference character count",
            "protocol": drec_report["protocol"],
            "series": {
                name: {
                    "overall_p50_ms": values["overall_p50"],
                    "linear_slope_ms_per_reference_character_1_to_64": values["slope"],
                    "by_length_bucket": {
                        bucket: {
                            "samples": int(values["count"][index]),
                            "p50_ms": float(values["p50"][index]),
                            "p90_ms": float(values["p90"][index]),
                        }
                        for index, bucket in enumerate(BUCKETS)
                    },
                }
                for name, values in series.items()
            },
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

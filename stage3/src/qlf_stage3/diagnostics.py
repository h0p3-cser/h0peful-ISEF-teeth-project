"""Two-panel diagnostics for raw MAE versus Huber MAE results."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

__all__ = ["draw"]

_STAGES = ("S1", "S2")
_STAGE_COLORS = {"S1": "#4C78A8", "S2": "#F58518"}
_PLOT_FIELDS = (
    "sample",
    "stage",
    "raw_mae",
    "huber_mae",
    "extrapolated_pct",
)


def _prepare_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the plotted fields and normalize values for plotting."""
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {index} must be a mapping, got {type(row).__name__}")

        missing = [field for field in _PLOT_FIELDS if field not in row]
        if missing:
            raise KeyError(f"row {index} is missing required field(s): {', '.join(missing)}")

        stage = str(row["stage"]).strip().upper()
        if stage not in _STAGE_COLORS:
            raise ValueError(f"row {index} has unsupported stage {row['stage']!r}")

        try:
            raw_mae = float(row["raw_mae"])
            huber_mae = float(row["huber_mae"]) if row["huber_mae"] != "" else float("nan")
            extrapolated_pct = float(row["extrapolated_pct"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {index} has a non-numeric plot value") from exc

        records.append(
            {
                "sample": str(row["sample"]),
                "stage": stage,
                "raw_mae": raw_mae,
                "huber_mae": huber_mae,
                "extrapolated_pct": extrapolated_pct,
            }
        )

    if not records:
        raise ValueError("rows must contain at least one record")
    return records


def draw(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a two-panel diagnostics figure to *path*.

    The inputs are mappings with ``sample``, ``stage`` (``S1`` or ``S2``),
    ``raw_mae``, ``huber_mae``, ``extrapolated_pct``, ``slope``, and
    ``intercept`` keys. The figure compares raw and Huber MAE in a stage-colored
    scatter plot and shows grouped horizontal extrapolation bars per sample.
    """
    records = _prepare_rows(rows)

    fig, (error_ax, extrapolation_ax) = plt.subplots(
        1,
        2,
        figsize=(15.0, 7.2),
        constrained_layout=True,
    )
    fig.suptitle("Raw vs. Huber MAE diagnostics", fontsize=15, fontweight="bold")

    # Panel 1: raw MAE against Huber MAE, colored by stage.
    all_mae: list[float] = []
    for stage in _STAGES:
        stage_rows = [record for record in records if record["stage"] == stage
                      and isfinite(record["raw_mae"]) and isfinite(record["huber_mae"])]
        if not stage_rows:
            continue
        raw_values = [record["raw_mae"] for record in stage_rows]
        huber_values = [record["huber_mae"] for record in stage_rows]
        all_mae.extend(raw_values)
        all_mae.extend(huber_values)
        error_ax.scatter(
            raw_values,
            huber_values,
            s=58,
            color=_STAGE_COLORS[stage],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.7,
            label=stage,
            zorder=3,
        )

    low = min(all_mae, default=0)
    high = max(all_mae, default=1)
    span = high - low
    padding = span * 0.08 if span > 0 else max(abs(high), 1.0) * 0.08
    lower_bound = low - padding
    if low >= 0.0:
        lower_bound = max(0.0, lower_bound)
    upper_bound = high + padding
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 1.0

    error_ax.plot(
        [lower_bound, upper_bound],
        [lower_bound, upper_bound],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="y = x",
        zorder=2,
    )
    error_ax.set_xlim(lower_bound, upper_bound)
    error_ax.set_ylim(lower_bound, upper_bound)
    error_ax.set_aspect("equal", adjustable="box")
    error_ax.set_xlabel("Raw MAE")
    error_ax.set_ylabel("Huber MAE")
    error_ax.set_title("Raw versus Huber MAE")
    error_ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    error_ax.set_axisbelow(True)
    error_ax.legend(title="Stage / reference", loc="upper left", frameon=True, framealpha=0.95)
    error_ax.spines["top"].set_visible(False)
    error_ax.spines["right"].set_visible(False)

    # Panel 2: grouped horizontal bars with one bar per stage for each sample.
    samples = list(dict.fromkeys(record["sample"] for record in records))
    values_by_sample: dict[str, dict[str, float]] = {sample: {} for sample in samples}
    for record in records:
        values_by_sample[record["sample"]][record["stage"]] = record["extrapolated_pct"]

    sample_positions = list(range(len(samples)))
    bar_height = 0.36
    for stage_index, stage in enumerate(_STAGES):
        positions = [
            position + (stage_index - 0.5) * bar_height
            for position in sample_positions
        ]
        values = [
            values_by_sample[sample].get(stage, float("nan"))
            for sample in samples
        ]
        extrapolation_ax.barh(
            positions,
            values,
            height=bar_height,
            color=_STAGE_COLORS[stage],
            edgecolor="white",
            linewidth=0.6,
            label=stage,
            zorder=3,
        )

    extrapolation_ax.set_yticks(sample_positions)
    extrapolation_ax.set_yticklabels(samples)
    extrapolation_ax.invert_yaxis()
    extrapolation_ax.axvline(0.0, color="#555555", linewidth=0.9, zorder=2)
    extrapolation_ax.set_xlabel("Extrapolated (%)")
    extrapolation_ax.set_ylabel("Sample")
    extrapolation_ax.set_title("Extrapolated percentage by sample")
    extrapolation_ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    extrapolation_ax.set_axisbelow(True)
    extrapolation_ax.legend(title="Stage", loc="lower right", frameon=True, framealpha=0.95)
    extrapolation_ax.spines["top"].set_visible(False)
    extrapolation_ax.spines["right"].set_visible(False)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)

"""Pixel-level QLF intensity and direct-loss metrics.

The module is intentionally small and dependency-light so that a batch runner
can import it directly.  Valid pixels are strictly finite; for direct loss the
baseline must be in [16, 255) and the current green in [0, 255).
"""

import cv2
import numpy as np

INTENSITY = (
    "mean_intensity",
    "median_intensity",
    "p10_intensity",
    "intensity_std",
)

LOSSES = (
    "mean_relative_loss_pct",
    "regional_loss_pct",
    "max_loss_pct",
    "p95_loss_pct",
    "loss_area_px",
    "loss_area_ratio_pct",
    "integrated_loss_pct",
    "spatial_heterogeneity_pct",
    "largest_connected_region_px",
)


def intensity(green, valid):
    """Summarize green intensity over valid, finite pixels.

    Returns one float per key in :data:`INTENSITY`, or ``None`` for every key
    when no pixel is valid.  Standard deviation is the population value.
    """
    values = np.asarray(green, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("green and valid must have the same shape")
    mask = mask & np.isfinite(values)
    if not mask.any():
        return {key: None for key in INTENSITY}

    sample = values[mask]
    return {
        "mean_intensity": float(np.mean(sample)),
        "median_intensity": float(np.median(sample)),
        "p10_intensity": float(np.percentile(sample, 10)),
        "intensity_std": float(np.std(sample, ddof=0)),
    }


def direct_loss(base, current, roi, tau=5.0):
    """Compute signed pixelwise relative loss for aligned green images.

    ``loss = 100 * (base - current) / base`` on pixels that are inside ``roi``,
    finite, have ``16 <= base < 255`` and ``0 <= current < 255``.  Invalid map
    pixels are NaN.  Negative losses are preserved: no correction, regression,
    or clipping is applied.

    Returns ``(loss_map, stats, support)`` where ``stats`` has every key in
    :data:`LOSSES` and ``support`` is the integer number of valid pixels.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    current_arr = np.asarray(current, dtype=np.float64)
    roi_arr = np.asarray(roi, dtype=bool)
    if base_arr.ndim != 2:
        raise ValueError("base must be a 2-D array")
    if base_arr.shape != current_arr.shape or base_arr.shape != roi_arr.shape:
        raise ValueError("base, current, and roi must have the same 2-D shape")

    valid = (
        roi_arr
        & np.isfinite(base_arr)
        & np.isfinite(current_arr)
        & (base_arr >= 16.0)
        & (base_arr < 255.0)
        & (current_arr >= 0.0)
        & (current_arr < 255.0)
    )
    support = int(np.count_nonzero(valid))
    loss_map = np.full(base_arr.shape, np.nan, dtype=np.float64)
    if support == 0:
        return loss_map, {key: None for key in LOSSES}, support

    losses = 100.0 * (base_arr[valid] - current_arr[valid]) / base_arr[valid]
    loss_map[valid] = losses

    loss_pixels = losses > float(tau)
    loss_area = int(np.count_nonzero(loss_pixels))
    above = losses[loss_pixels]

    if loss_area:
        regional = float(np.mean(above))
        integrated = float(np.sum(above) / support)
        _, _, components, _ = cv2.connectedComponentsWithStats(
            (loss_map > float(tau)).astype(np.uint8), connectivity=8
        )
        largest = int(components[1:, cv2.CC_STAT_AREA].max())
    else:
        regional = None
        integrated = 0.0
        largest = 0

    stats = {
        "mean_relative_loss_pct": float(np.mean(losses)),
        "regional_loss_pct": regional,
        "max_loss_pct": float(np.max(losses)),
        "p95_loss_pct": float(np.percentile(losses, 95)),
        "loss_area_px": loss_area,
        "loss_area_ratio_pct": float(100.0 * loss_area / support),
        "integrated_loss_pct": integrated,
        "spatial_heterogeneity_pct": float(np.std(losses, ddof=0)),
        "largest_connected_region_px": largest,
    }
    return loss_map, stats, support

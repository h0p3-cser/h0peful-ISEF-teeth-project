"""Summarize measured tooth/state areas without third-party dependencies."""

import math
from collections import defaultdict

_NUMERIC_FIELDS = ("area_px", "area_pct", "valid_crown_px")


def _number(value):
    """Return a finite float, or None when the value is blank/non-numeric."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize(rows):
    """Return sorted per-tooth/state image counts and equal-angle means."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("state_id") == "S0":
            continue
        groups[(row.get("tooth_id"), row.get("state_id"))].append(row)

    result = []
    for key in sorted(groups, key=lambda item: tuple("" if v is None else str(v) for v in item)):
        tooth_id, state_id = key
        group = groups[key]
        unique_images = measured_unique_images = 0
        measured_angles = set()
        by_variable_angle = defaultdict(list)

        for row in group:
            if str(row.get("is_duplicate")) == "1":
                continue
            unique_images += 1
            if row.get("status") != "ok":
                continue

            values = [_number(row.get(field)) for field in _NUMERIC_FIELDS]
            if all(value is None for value in values):
                continue
            measured_unique_images += 1
            angle = row.get("angle_id")
            measured_angles.add(angle)
            for field, value in zip(_NUMERIC_FIELDS, values):
                if value is not None:
                    by_variable_angle[(field, angle)].append(value)

        means = {}
        for field in _NUMERIC_FIELDS:
            angle_means = [
                sum(values) / len(values)
                for (name, _), values in by_variable_angle.items()
                if name == field
            ]
            means[field] = sum(angle_means) / len(angle_means) if angle_means else ""

        result.append({
            "tooth_id": tooth_id,
            "state_id": state_id,
            "total_images": len(group),
            "unique_images": unique_images,
            "measured_unique_images": measured_unique_images,
            "measured_angles": len(measured_angles),
            "mean_area_px": means["area_px"],
            "mean_area_pct": means["area_pct"],
            "mean_valid_crown_px": means["valid_crown_px"],
        })

    return result

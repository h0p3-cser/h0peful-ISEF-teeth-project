"""Constant CIELAB b* offset, nominal sRGB / D65 / 2 degree observer."""
import numpy as np
from skimage.color.colorconv import rgb_from_xyz, xyz_from_rgb

from ..exposure import linear_to_srgb
from ..rgb.color import LINEAR

DELTA = 6 / 29
WHITE_Z = 1.08883


def lab_f(t):
    return np.where(t > DELTA ** 3, np.cbrt(t), t / (3 * DELTA ** 2) + 4 / 29)


def inverse_f(t):
    return np.where(t > DELTA, t ** 3, 3 * DELTA ** 2 * (t - 4 / 29))


def region_offset(rgb, regions):
    """Median of per-region medians; empty regions have no vote."""
    medians, audit = [], []
    height, width = rgb.shape[:2]
    for region in regions:
        x0, y0, x1, y1 = region['box']
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Reference coordinates outside the oriented photograph')
        patch = rgb[y0:y1, x0:x1].reshape(-1, 3)
        valid = patch[((patch > 5) & (patch < 250)).all(axis=1)]
        median = None
        if len(valid):
            lin = LINEAR[valid]
            fy = lab_f(lin @ xyz_from_rgb[1])  # Y_n = 1 for D65.
            fz = lab_f((lin @ xyz_from_rgb[2]) / WHITE_Z)
            median = float(np.median(200 * (fy - fz)))
            medians.append(median)
        audit.append(dict(**region, pixels=len(patch), valid_pixels=len(valid), median_bstar=median))
    return (float(np.median(medians)) if medians else None), audit


def apply_offset(rgb, median):
    """With L* and a* fixed, X and Y stay fixed and f(Z/Zn) increases by m/200."""
    output = np.empty_like(rgb)
    for start in range(0, len(rgb), 96):
        slab = slice(start, start + 96)
        linear = LINEAR[rgb[slab]]
        z = linear @ xyz_from_rgb[2]
        shifted_z = WHITE_Z * inverse_f(lab_f(z / WHITE_Z) + median / 200)
        mapped = linear + (shifted_z - z)[..., None] * rgb_from_xyz[:, 2]
        # Monotone sRGB encoding maps [0,1] onto [0,1]. Clip only the output gamut.
        encoded = linear_to_srgb(np.clip(mapped, 0.0, 1.0))
        output[slab] = np.rint(encoded).astype(np.uint8)
    return output

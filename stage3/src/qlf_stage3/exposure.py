"""Root exposure fits, shared headroom and paired loss measurements."""

import numpy as np

LOW = 16

def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """sRGB 0-255 -> 线性 RGB 0-1。"""
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(light):
    """Encode to floating-point JPEG green units without clipping."""
    return 255.0 * np.where(light <= .0031308, 12.92 * light,
                            1.055 * light ** (1.0 / 2.4) - .055)


def fit_exposure(reference, observed):
    """Return gain, EV and maximum anchor error, with equal weights in EV."""
    if (reference.shape != observed.shape or reference.ndim != 1
            or reference.size not in (1, 2)
            or not np.isfinite(reference).all() or not np.isfinite(observed).all()
            or min(reference.min(), observed.min()) <= 0):
        raise ValueError("Expected one or two positive finite linear anchors per image")
    target_ev = np.log2(reference / observed)
    ev = float(target_ev.mean())
    return 2.0 ** ev, ev, float(np.abs(target_ev - ev).max())


def root_samples(root, boxes):
    patches = []
    for x0, y0, x1, y1 in boxes:
        box = np.zeros_like(root)
        box[y0:y1 + 1, x0:x1 + 1] = True
        patch = root & box
        if patch.sum() < 500:
            raise ValueError("Insufficient root pixels in a reference patch")
        patches.append(patch)
    held_out = root & ~np.logical_or.reduce(patches)
    if (patches[0] & patches[1]).any() or held_out.sum() < 500:
        raise ValueError("Reference and validation root areas must be disjoint")
    return patches, held_out


def common_exposure(rgb, gains):
    # The maximum encoded value also locates the linear maximum by monotonicity.
    brightest = srgb_to_linear(rgb.max(axis=(1, 2, 3))) * gains
    headroom = min(1.0, 1.0 / brightest.max())
    adjusted = linear_to_srgb(srgb_to_linear(rgb) * (headroom * gains)[:, None, None, None])
    assert adjusted.max() <= 255 + 1e-8
    return np.rint(adjusted).astype(np.uint8), headroom


def loss_map(source, target, mask):
    result = np.full(source.shape, np.nan, dtype=np.float32)
    denom = source[mask]
    if not np.isfinite(denom).all() or np.any(denom <= 0):
        raise ValueError("Loss denominator must be finite and positive.")
    result[mask] = 100 * (denom - target[mask]) / denom
    return result


def measure(green, light, gains, roi, i, j):
    before, after = light[i] * gains[i], light[j] * gains[j]
    usable = roi & (green[i] >= LOW) & (green[i] < 255) & (green[j] < 255)
    above = (before > 1) | (after > 1)
    below = before < srgb_to_linear(LOW)
    valid = usable & ~above & ~below
    corrected = loss_map(linear_to_srgb(before), linear_to_srgb(after), valid)
    result = dict(reference_crown_px=int(roi.sum()), raw_usable_px=int(usable.sum()),
                  compared_px=int(valid.sum()), over_range_px=int((usable & above).sum()),
                  low_source_px=int((usable & ~above & below).sum()),
                  coverage_pct=100 * valid.sum() / roi.sum(),
                  raw_loss_same_pixels_pct=None, loss_pct=None, linear_loss_pct=None)
    if valid.any():
        result["raw_loss_same_pixels_pct"] = float(np.nanmean(loss_map(green[i], green[j], valid)))
        result["loss_pct"] = float(np.nanmean(corrected))
        result["linear_loss_pct"] = float(np.mean(100 * (1 - after[valid] / before[valid])))
    return corrected, valid, result

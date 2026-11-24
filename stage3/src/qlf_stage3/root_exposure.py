"""Run the two-root reference exposure correction on registered QLF views."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

from .exposure import (LOW, common_exposure, fit_exposure, linear_to_srgb,
                      loss_map, measure, root_samples, srgb_to_linear)

HERE = Path(__file__).resolve().parent
from .settings import PROJECT
PAIRS = ((0, 1), (1, 2), (0, 2))
NORM = Normalize(-40, 40)


def plot_style():
    for name in ("PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_index(semantic, cases):
    selected = {path for case in cases for path in case["images"]}
    views = {(case["tooth_id"], case["angle_id"]) for case in cases}
    index = {}
    for name in ("legacy_index.jsonl", "heavy_index.jsonl"):
        with (semantic / name).open(encoding="utf-8") as f:
            for line in f:
                meta = json.loads(line)
                path = meta["path"]
                is_control = meta["state"] == "S0" and (meta["tooth"], meta["angle"]) in views
                if meta["modality"] != "QLF" or (path not in selected and not is_control):
                    continue
                index[path] = dict(raw_path=path, tooth_id=meta["tooth"], angle_id=meta["angle"],
                                   state_id=meta["state"], modality=meta["modality"],
                                   dose_hours=meta["dose"], semantic=meta)
    missing = selected - index.keys()
    if missing:
        raise ValueError(f"Missing selected QLF paths in the semantic index: {sorted(missing)}")
    return index


def view_in_s0(r, anchor, data, semantic):
    with Image.open(data/r["raw_path"]) as im, Image.open(semantic/r["semantic"]["label_path"]) as labels:
        rgb=np.asarray(im.convert("RGB"))
        lab=np.asarray(labels)
        ex=im.getexif().get_ifd(34665)
        exif=tuple(float(ex[k]) for k in (33434,34855,33437))
    if r["raw_path"]!=anchor:
        if r["semantic"]["s0_anchor"] != anchor or r["semantic"]["transform"] is None:
            raise ValueError(f"Missing registration to {anchor}: {r['raw_path']}")
        mat=cv2.invertAffineTransform(np.asarray(r["semantic"]["transform"]))
        size=lab.shape[::-1]
        rgb=cv2.warpAffine(rgb,mat,size)
        lab=cv2.warpAffine(lab,mat,size,flags=cv2.INTER_NEAREST)
    return rgb,lab,exif


def save(out, name, rows):
    with (out/name).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def swatch(ax, image, base, roi, valid, crop):
    ax.imshow(np.where(roi & ~valid, .83, base)[crop], cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.imshow(np.ma.masked_invalid(image[crop]), cmap="RdBu_r", norm=NORM,
              interpolation="nearest")
    ax.set_axis_off()


def caption(stats):
    if stats["loss_pct"] is None:
        return "无有效像素"
    return f'{stats["loss_pct"]:+.2f}%  ·  覆盖 {stats["coverage_pct"]:.1f}%'


def colorbar(fig, axes):
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap="RdBu_r"), ax=axes,
                      orientation="horizontal", shrink=.53, aspect=35, pad=.055,
                      extend="both", ticks=[-40, -20, 0, 20, 40])
    cb.set_label("相对绿通道损失（%）   蓝色变亮 · 红色变暗 · 灰色区域不参与计算")


def same_state_check(case, index, classes, data, semantic):
    """Use the two most separated S0 exposures if their difference is material."""
    tooth, angle, anchor = case["tooth_id"], case["angle_id"], case["anchor_path"]
    candidates = []
    for path, record in index.items():
        if (record["tooth_id"], record["angle_id"], record["state_id"], record["modality"]) != (
                tooth, angle, "S0", "QLF"):
            continue
        alignment = record["semantic"]
        if path != anchor and (alignment["s0_anchor"] != anchor or alignment["transform"] is None):
            continue
        with Image.open(data / path) as image:
            exif = image.getexif().get_ifd(34665)
            seconds, iso, aperture = (float(exif[k]) for k in (33434, 34855, 33437))
        candidates.append((seconds * iso / aperture ** 2, path))
    if len(candidates) < 2:
        return None
    low, high = min(candidates), max(candidates)
    separation = np.log2(high[0] / low[0])
    if separation < .1:
        return None
    aligned = [view_in_s0(index[path], anchor, data, semantic) for _, path in (low, high)]
    green = np.stack([v[0][..., 1] for v in aligned]).astype(float)
    labels = np.stack([v[1] for v in aligned])
    valid = ((labels == 1) & (green >= LOW) & (green < 255)).all(axis=0)
    patches, held_out = root_samples((classes == 1) & valid, case["root_boxes"])
    light = srgb_to_linear(green)
    levels = np.stack([np.median(light[:, p], axis=1) for p in patches], axis=1)
    gain, ev, anchor_error = fit_exposure(levels[0], levels[1])
    _, _, crown = measure(green, light, np.array([1., gain]), (classes == 2) & valid, 0, 1)
    held_out_levels = np.median(light[:, held_out], axis=1)
    raw_ev = np.log2(held_out_levels[1] / held_out_levels[0])
    return dict(tooth_id=tooth, angle_id=angle, low_path=low[1], high_path=high[1],
                nominal_gap_ev=separation, high_to_low_gain=gain, ev_applied=ev,
                max_anchor_error_ev=anchor_error, held_out_raw_ev=raw_ev,
                held_out_adjusted_ev=raw_ev + ev, **crown)


def render(case, index, data, semantic, masks, out):
    tooth, angle = case["tooth_id"], case["angle_id"]
    stem = f"{tooth}_{angle}"
    frames = [index[path] for path in case["images"]]
    for i, frame in enumerate(frames):
        assert (frame["tooth_id"], frame["angle_id"], frame["state_id"]) == (tooth, angle, f"S{i}")
    aligned = [view_in_s0(frame, case["anchor_path"], data, semantic) for frame in frames]
    rgb = np.stack([v[0] for v in aligned])
    labels = np.stack([v[1] for v in aligned])
    green = rgb[..., 1].astype(float)
    with Image.open(masks / f"{stem}_牙根牙冠掩膜.png") as image:
        classes = np.asarray(image)
    roi, stain = classes == 2, classes == 3
    root = (classes == 1) & ((labels == 1) & (green >= LOW) & (green < 255)).all(axis=0)
    patches, held_out = root_samples(root, case["root_boxes"])
    assert not (root & (roi | stain)).any()
    light = srgb_to_linear(green)
    levels = np.stack([np.median(light[:, p], axis=1) for p in patches], axis=1)
    gains = np.array([fit_exposure(levels[0], t)[0] for t in levels])
    adjusted, headroom = common_exposure(rgb, gains)
    applied = gains * headroom
    # This identity checks the fit area; the excluded root region measures residual error.
    ref = linear_to_srgb(np.sqrt(np.prod(levels * applied[:, None], axis=1)))
    np.testing.assert_allclose(ref, ref[0], atol=1e-10, rtol=0)
    hold = np.median(light[:, held_out], axis=1)
    held_raw, held_corrected = linear_to_srgb(hold), linear_to_srgb(hold * applied)
    fit_stats = []
    for i in range(3):
        fit_stats.append(dict(
            tooth_id=tooth, angle_id=angle, state_id=f"S{i}", dose_hours=frames[i]["dose_hours"],
            source_path=frames[i]["raw_path"], exposure_s=aligned[i][2][0], gain_to_s0=gains[i],
            relative_ev=np.log2(gains[i]), shared_headroom_ev=np.log2(headroom), final_gain=applied[i],
            near_root_px=int(patches[0].sum()), far_root_px=int(patches[1].sum()),
            held_out_px=int(held_out.sum()), raw_near_g=linear_to_srgb(levels[i, 0]),
            raw_far_g=linear_to_srgb(levels[i, 1]),
            adjusted_near_g=linear_to_srgb(levels[i, 0] * applied[i]),
            adjusted_far_g=linear_to_srgb(levels[i, 1] * applied[i]),
            held_out_raw_median_g=held_raw[i], held_out_adjusted_median_g=held_corrected[i],
            held_out_adjusted_delta_ev=np.log2(hold[i] * gains[i] / hold[0])))
    x, y, w, h = cv2.boundingRect(np.uint8((labels > 0).any(axis=0)))
    crop = np.s_[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
    fig, axes = plt.subplots(4, 3, figsize=(11.2, 8.9), layout="constrained")
    fig.patch.set_facecolor("#faf9f7")
    fig.suptitle(f"{tooth} · {int(angle[1:])}°   以同一牙根参照统一曝光", fontsize=17)
    for i in range(3):
        fig_path = out / "corrected-frames" / f"{stem}_S{i}-corrected.png"
        Image.fromarray(adjusted[i]).save(fig_path)
        for row, photograph in enumerate((rgb[i], adjusted[i])):
            ax = axes[row, i]
            ax.imshow(photograph[crop])
            for p in patches:
                ax.contour(p[crop], levels=[.5], colors="#ffcc66", linewidths=.60)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f'S{i} · {frames[i]["dose_hours"]} h', fontsize=12)
                label = f"原图 · 留出牙根 G {held_raw[i]:.1f}"
            else:
                label = f"已校正 · 留出牙根 G {held_corrected[i]:.1f} · {np.log2(applied[i]):+.2f} EV"
            ax.text(.5, -.046, label, ha="center", va="top", transform=ax.transAxes, fontsize=9.5)

    stats = []
    for col, (i, j) in enumerate(PAIRS):
        base = np.where(labels[i] > 0, .73 + .23 * green[i] / 255, 1)
        base = np.where(stain, .62, base)
        post, valid, result = measure(green, light, applied, roi, i, j)
        pre = loss_map(green[i], green[j], valid)
        stats.append(dict(tooth_id=tooth, angle_id=angle, pair=f"S{i}→S{j}", **result))
        for row, image in enumerate((pre, post), 2):
            ax = axes[row, col]
            swatch(ax, image, base, roi, valid, crop)
            label = f'未校正 {result["raw_loss_same_pixels_pct"]:+.2f}%' if row == 2 else (
                f"牙根校正 {caption(result)}")
            ax.text(.5, -.045, label, ha="center", va="top", transform=ax.transAxes, fontsize=9.5)
            if row == 2:
                ax.set_title(f"S{i} → S{j}", fontsize=12)
    colorbar(fig, axes[2:])
    fig.savefig(out / f"{stem}_root-exposure.png", dpi=165)
    plt.close(fig)
    return fit_stats, stats, same_state_check(case, index, classes, data, semantic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "Data")
    parser.add_argument("--semantic", type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "config/exposure-rois.json")
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/root-exposure"))
    args = parser.parse_args()
    semantic = args.semantic or args.data / "语义掩膜_背景与基线黑斑"

    plot_style()
    cv2.setNumThreads(1)
    (args.output / "corrected-frames").mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.config.read_text(encoding="utf-8"))
    if not cases:
        parser.error("--config must contain at least one view")
    index = load_index(semantic, cases)
    fits, stats, controls = [], [], []
    for case in cases:
        parameters, loss, control = render(case, index, args.data, semantic, args.masks, args.output)
        fits.extend(parameters)
        stats.extend(loss)
        if control is not None:
            controls.append(control)
    save(args.output, "root-holdout.csv", fits)
    save(args.output, "loss-statistics.csv", stats)
    if controls:
        save(args.output, "baseline-repeat-checks.csv", controls)


if __name__ == "__main__":
    main()

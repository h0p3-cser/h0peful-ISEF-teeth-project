"""Predict a healthy crown baseline with an independent root Huber fit per stage."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent

from .huber_fit import fit
from .diagnostics import draw
from .render_threshold import render
from .thresholds import THRESHOLDS, by_physical_path, overview
from .build_masks import process_group
from .dataset import DATA, cv2, get_anchor, inventory, np, read_csv, read_label, write_csv
from .crown import geometry, roi_labels
from .direct_features import (FEATURE_DIR, FEATURE_FILE, S0_SOURCE_IDS, confirmed_rows, frame_native,
                            in_reference, indexes, measure_pair, reference_crop)

TILE = 21


def root_samples(root, earlier, later, bounds, config):
    """Disjoint complete root tiles contribute one paired green median each."""
    xs, ys = bounds[:2]
    half = TILE // 2
    rows = []
    for iy in range(0, root.shape[0] - TILE + 1, TILE):
        for ix in range(0, root.shape[1] - TILE + 1, TILE):
            area = np.s_[iy:iy + TILE, ix:ix + TILE]
            if root[area].all():
                rows.append((xs + ix + half, ys + iy + half,
                             np.median(earlier[area]), np.median(later[area])))
    samples = np.asarray(rows, dtype=float)
    if len(samples) < 12:
        raise ValueError(f"Only {len(samples)} usable {TILE}×{TILE} root tiles")
    left, right = np.array(config["crown_boundary"])
    direction = right - left
    normal = np.array([-direction[1], direction[0]])
    return samples[np.argsort(samples[:, :2] @ normal, kind="stable")]


def fit_with_holdout(samples):
    x, y = samples[:, 2:].T
    pred = np.full_like(x, np.nan)
    blocks = np.array_split(np.arange(len(x)), 4)
    for test in blocks:
        train = np.ones(len(x), dtype=bool)
        train[test] = False
        a, b = fit(x[train], y[train], "huber")
        pred[test] = a * x[test] + b
    a, b = fit(x, y, "huber")
    if not np.isfinite([a, b]).all():
        raise ValueError("Huber fit did not produce a positive finite slope")
    return a, b, np.mean(abs(x - y)), np.mean(abs(pred - y)), pred, blocks


def paired_root_tiles(root0, base_crown, target_crown, target_body, bounds, config):
    first = base_crown.green.astype(float)
    measured = target_crown.green.astype(float)
    usable = root0 & (target_body.labels == 1) & (first >= 16) & (first < 255)
    usable &= (measured >= 16) & (measured < 255)
    usable &= (base_crown.labels == 0) & (target_crown.labels == 0)
    usable = cv2.erode(usable.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return root_samples(usable, first, measured, bounds, config)


def calibrated_loss(raw_loss, first, measured, slope, intercept):
    predicted = slope * first + intercept
    valid = np.isfinite(raw_loss) & (predicted >= 16) & (predicted < 255)
    loss = np.full_like(raw_loss, np.nan)
    loss[valid] = 100 * (predicted[valid] - measured[valid]) / predicted[valid]
    return predicted, valid, loss


def compare_case(sample, groups, legacy, index, held, table, configs, scratch):
    tooth, angle, rank = (sample[k] for k in ("tooth_id", "angle_id", "repeat_rank"))
    group = confirmed_rows(groups, tooth, angle)
    anchor = get_anchor(group, legacy)
    entries = [anchor, *(next(r for r in group if r["state_id"] == s and r["repeat_rank"] == rank)
                         for s in ("S1", "S2"))]
    if by_physical_path(anchor) != sample["s0_path"] or entries[-1]["path"] != sample["s2_path"]:
        raise ValueError(f"Sample differs from the previous threshold run: {sample['sample']}")
    if tooth in S0_SOURCE_IDS:
        corrected = [r for s in ("S1", "S2")
                     for r in process_group(group, legacy, None, set(), state=s, label_dir=scratch)]
        index = index | {r["path"]: r for r in corrected}
        held = held - {(tooth, "S1"), (tooth, "S2")}
    config = configs[anchor.get("source_tooth_id", tooth), angle]
    natives = [frame_native(r, index[r["path"]], held) for r in entries]
    bounds = reference_crop(natives[0].labels)
    x, y, x1, y1 = bounds
    a, b = np.array(config["crown_boundary"])
    opposite = dict(crown_boundary=[a, b], crown_side_point=2*a-np.array(config["crown_side_point"]))
    root0 = roi_labels(natives[0].labels, opposite, np.eye(3))[y:y1, x:x1] == 1
    bodies, crowns = [], []
    for raw in natives:
        row = raw.row
        meta = index[row["path"]]
        body = in_reference(raw, meta, anchor["path"], bounds)
        mask = read_label(DATA / table[by_physical_path(row)]["crown_semantic_mask_path"])
        if np.any((mask > 0) & (mask != raw.labels)):
            raise ValueError(f"Crown mask does not match semantic labels: {row['path']}")
        crown = in_reference(replace(raw, labels=mask), meta, anchor["path"], bounds)
        if body is None or crown is None:
            raise ValueError(f"Missing alignment: {row['path']}")
        bodies.append(body)
        crowns.append(crown)
    # Common display crop for both comparisons of this view.
    cx, cy, w, h = cv2.boundingRect(np.uint8(np.logical_or.reduce([c.labels > 0 for c in crowns])))
    crop = np.s_[max(0, cy-24):cy+h+24, max(0, cx-24):cx+w+24]

    results, patches = [], []
    for j in (1, 2):
        first = crowns[0].green.astype(float)
        measured = crowns[j].green.astype(float)
        tiles = paired_root_tiles(root0, crowns[0], crowns[j], bodies[j], bounds, config)
        slope, intercept, raw_mae, huber_mae, predictions, blocks = fit_with_holdout(tiles)
        raw_loss, status, stats = measure_pair(crowns[0], crowns[j])
        record = table[by_physical_path(entries[j])]
        if (status != "automatic_qc_pass" or stats["valid_loss_roi_pixels"] != int(record["valid_loss_roi_pixels"])
                or not np.isclose(stats["mean_relative_loss_pct"], float(record["mean_relative_loss_pct"]))):
            raise ValueError(f"Direct control could not reproduce the feature table for {entries[j]['path']}")
        raw_valid = np.isfinite(raw_loss)
        predicted, valid, corrected_loss = calibrated_loss(raw_loss, first, measured, slope, intercept)
        if not valid.any():
            raise ValueError(f"No valid predicted crown denominator for {entries[j]['path']}")
        low, high = tiles[:, 2].min(), tiles[:, 2].max()
        supported = valid & (first >= low) & (first <= high)
        stage = entries[j]["state_id"]
        row = {k: sample[k] for k in ("sample", "tooth_id", "angle_id", "repeat_rank", "s0_path")}
        row.update(stage=stage, dose_hours=entries[j]["dose_hours"], target_path=entries[j]["path"],
                   root_tiles=len(tiles), root_tile_size=TILE, slope=slope, intercept=intercept,
                   root_s0_median_min=low, root_s0_median_max=high,
                   raw_mae=raw_mae, huber_mae=huber_mae if np.isfinite(huber_mae) else "",
                   root_spatial_folds_fitted=int(sum(np.isfinite(predictions[test]).all() for test in blocks)),
                   raw_valid_pixels=int(raw_valid.sum()), valid_pixels=int(valid.sum()),
                   in_range_pixels=int(supported.sum()), extrapolated_pct=100*(1-supported.sum()/valid.sum()),
                   direct_mean_loss_same_pixels=float(raw_loss[valid].mean()),
                   huber_mean_loss=float(corrected_loss[valid].mean()),
                   huber_mean_loss_in_range=float(corrected_loss[supported].mean()) if supported.any() else "")
        fractions = []
        for t in THRESHOLDS:
            above = corrected_loss > t
            count = int(above.sum())
            fraction = 100 * count / valid.sum()
            row[f"loss_gt_{t}_px"] = count
            row[f"loss_gt_{t}_pct"] = fraction
            row[f"direct_gt_{t}_pct_same_pixels"] = 100 * ((raw_loss > t) & valid).sum() / valid.sum()
            row[f"in_range_gt_{t}_pct"] = 100 * (above & supported).sum() / supported.sum() if supported.any() else ""
            fractions.append(fraction)
        for fold, test in enumerate(blocks, 1):
            for k in test:
                px, py, src, dst = tiles[k]
                patches.append(dict(sample=sample["sample"], stage=stage, fold=fold, x=int(px), y=int(py),
                                    g_s0_median=src, g_target_median=dst,
                                    g_holdout_prediction=predictions[k] if np.isfinite(predictions[k]) else ""))
        # Display the G channel at the same 0–255 scale in both image panels.
        estimated_crown = np.where(crowns[0].labels == 1, predicted, 0)
        images = [np.repeat(np.uint8(np.clip(np.nan_to_num(g[crop]), 0, 255))[..., None], 3, axis=2)
                  for g in (estimated_crown, measured)]
        results.append((row, images, corrected_loss[crop], fractions))
    return results, patches


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/huber-thresholds"))
    parser.add_argument("--samples", type=Path, default=Path("outputs/thresholds/threshold-area.csv"))
    args = parser.parse_args()
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    sample = read_csv(args.samples)
    table = {r["image_path"]: r for r in read_csv(FEATURE_DIR / FEATURE_FILE)}
    groups = inventory()
    legacy, index, held = indexes()
    configs = geometry()
    rows, patches = [], []
    with TemporaryDirectory(prefix="qlf-huber-root-") as tmp, ThreadPoolExecutor(max_workers=4) as pool:
        outputs = pool.map(lambda s: compare_case(s, groups, legacy, index, held, table, configs, Path(tmp)), sample)
        for results, root_patches in outputs:
            for row, (base, current), loss, fractions in results:
                stage = row["stage"]
                title = f'{row["sample"]}  |  S0 → {stage}（{row["dose_hours"]} h）'
                render(dest / f'{row["sample"]}_{stage}.png', title, base, current, loss, THRESHOLDS, fractions,
                       headings=(f"{stage} 预测健康基线 · G", f"{stage} 实测 · G"),
                       subtitle=f'牙根 Huber   ·   {row["root_tiles"]} 个小块   ·   牙冠外推 {row["extrapolated_pct"]:.1f}%')
                rows.append(row)
                print(row["sample"], stage, row["root_tiles"], round(row["slope"], 3),
                      round(row["huber_mean_loss"], 2), [round(v, 1) for v in fractions], flush=True)
            patches.extend(root_patches)
    write_csv(dest / "huber-threshold-area.csv", rows)
    write_csv(dest / "root-samples.csv", patches)
    for s in ("S1", "S2"):
        overview(dest / f"{s}-huber-threshold-area.png", [r for r in rows if r["stage"] == s],
                 title=f"S0 → {s} · 牙根 Huber 校正后的五档响应面积")
    draw(dest / "root-fit-diagnostics.png", rows)


if __name__ == "__main__":
    main()

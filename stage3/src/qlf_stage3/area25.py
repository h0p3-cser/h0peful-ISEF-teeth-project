"""Annotate all active QLF frames at a strict 25% Huber loss threshold."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
from .huber import calibrated_loss, fit, paired_root_tiles
from .build_masks import process_group
from .dataset import DATA, cv2, get_anchor, homogeneous, inventory, np, read_csv, read_label, read_rgb, write_csv
from .crown import geometry, roi_labels
from .direct_features import (FEATURE_DIR, FEATURE_FILE, S0_SOURCE_IDS, confirmed_rows, frame_native,
                            in_reference, indexes, measure_pair, reference_crop)
from .area_summary import summarize

THRESHOLD = 25.0
OUT_IMAGES = "annotations"


def image_path(row):
    return row.get("physical_path", row["path"])


def overlay_image(photo, roi, response, transform, bounds):
    x, y = bounds[:2]
    translation = np.array([[1., 0, x], [0, 1., y], [0, 0, 1.]])
    mapping = (homogeneous(transform) @ translation)[:2]
    size = photo.shape[1], photo.shape[0]
    areas = np.uint8(roi) + np.uint8(response)
    native = cv2.warpAffine(areas, mapping, size, flags=cv2.INTER_NEAREST)
    marked = photo.copy()
    highlight = native == 2
    marked[highlight] = np.rint(.48 * marked[highlight] + .52 * np.array([53, 92, 249])).astype(np.uint8)
    outline, _ = cv2.findContours(np.uint8(highlight), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Only larger components need borders. All thresholded pixels retain the fill.
    visible = [p for p in outline if cv2.contourArea(p) >= 28]
    cv2.drawContours(marked, visible, -1, (179, 219, 255), 1, cv2.LINE_AA)
    return marked


def publish(dest, record, photo):
    file = dest / record["annotation_path"]
    file.parent.mkdir(parents=True, exist_ok=True)
    h, w = photo.shape[:2]
    ratio = min(1.0, 1500 / max(h, w))
    if ratio < 1:
        photo = cv2.resize(photo, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)
    height, width = photo.shape[:2]
    scale = max(.46, min(.88, width / 1320))
    panel_height = round(78 * scale + 25)
    panel = np.full((height + panel_height, width, 3), (36, 31, 28), np.uint8)
    panel[panel_height:] = photo
    title = f'{record["tooth_id"]}   {record["state_id"]}   {record["angle_id"]}   R{int(record["repeat_rank"]):02}'
    if record["is_duplicate"]:
        title += "   COPY"
    if record["status"] == "ok":
        detail = f'Huber loss >25%: {record["area_pct"]:.2f}%   /   {record["area_px"]:,} px'
        color = (166, 198, 255)
    elif record["status"] == "baseline":
        detail = "S0 baseline reference"
        color = (193, 218, 205)
    else:
        detail = f'No result: {record["reason"]}'
        color = (158, 183, 250)
    for text, ordinate, ink in ((title, .41, (245, 247, 250)), (detail, .79, color)):
        while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] >= width - 34:
            scale *= .9
        cv2.putText(panel, text, (16, round(panel_height * ordinate)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, ink, 1, cv2.LINE_AA)
    ok, contents = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError(f"Could not encode {file}")
    pending = file.with_suffix(".jpg.tmp")
    pending.write_bytes(contents.tobytes())
    pending.replace(file)


def init_record(row, anchor, table):
    physical = image_path(row)
    ref = table[physical]
    return dict(tooth_id=row["tooth_id"], state_id=row["state_id"], angle_id=row["angle_id"],
                repeat_rank=row["repeat_rank"], dose_hours=row["dose_hours"],
                source_tooth_id=ref["source_tooth_id"], is_duplicate=int(row["is_duplicate"]),
                image_path=physical, s0_path=image_path(anchor),
                annotation_path=(Path(OUT_IMAGES) / row["state_id"] / row["tooth_id"] / row["angle_id"] /
                                 Path(physical).name).as_posix(),
                threshold_pct=THRESHOLD, status="baseline", reason="", valid_crown_px="",
                raw_valid_crown_px="", area_px="", area_pct="", in_range_crown_px="",
                extrapolated_pct="", slope="", intercept="", root_tiles="", huber_delta_f_pct="")


def with_crown(frame, record, table, anchor, bounds):
    mask = read_label(DATA / table[image_path(frame.row)]["crown_semantic_mask_path"])
    if mask.shape != frame.labels.shape or np.any((mask > 0) & (mask != frame.labels)):
        raise ValueError(f"Crown labels do not match source data for {frame.row['path']}")
    return in_reference(replace(frame, labels=mask), record, anchor["path"], bounds)


def score(row, anchor, base, root0, bounds, config, index, held, table, *, render_annotation=True):
    physical = image_path(row)
    known = table[physical]
    result = init_record(row, anchor, table)
    if row["state_id"] == "S0":
        return result, read_rgb(row) if render_annotation else None
    result["status"] = "excluded"
    if known["loss_status"] != "automatic_qc_pass":
        result["reason"] = known["loss_status"].replace("_", " ")
        return result, read_rgb(row) if render_annotation else None

    meta = index[row["path"]]
    frame = frame_native(row, meta, held)
    body = in_reference(frame, meta, anchor["path"], bounds)
    crown = with_crown(frame, meta, table, anchor, bounds)
    if body is None or crown is None:
        raise ValueError(f"Previously valid alignment unavailable for {physical}")
    raw, status, stats = measure_pair(base, crown)
    if (status != "automatic_qc_pass" or stats["valid_loss_roi_pixels"] != int(known["valid_loss_roi_pixels"])
            or not np.isclose(stats["mean_relative_loss_pct"], float(known["mean_relative_loss_pct"]))):
        raise ValueError(f"Direct result differs from the audited feature table for {physical}")
    result["raw_valid_crown_px"] = stats["valid_loss_roi_pixels"]
    try:
        tiles = paired_root_tiles(root0, base, crown, body, bounds, config)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result, frame.photo
    result["root_tiles"] = len(tiles)
    slope, intercept = fit(tiles[:, 2], tiles[:, 3], "huber")
    if not np.isfinite([slope, intercept]).all():
        result["reason"] = "no positive finite Huber fit"
        return result, frame.photo
    result.update(slope=slope, intercept=intercept)
    _, valid, loss = calibrated_loss(raw, base.green, crown.green, slope, intercept)
    if not valid.any():
        result["reason"] = "no eligible predicted crown values"
        return result, frame.photo
    response = loss > THRESHOLD
    n, area = int(valid.sum()), int(response.sum())
    lower, upper = tiles[:, 2].min(), tiles[:, 2].max()
    inside = int((valid & (base.green >= lower) & (base.green <= upper)).sum())
    result.update(status="ok", valid_crown_px=n, area_px=area, area_pct=100 * area / n,
                  in_range_crown_px=inside, extrapolated_pct=100 * (n-inside) / n,
                  huber_delta_f_pct=float(loss[valid].mean()))
    photo = overlay_image(frame.photo, valid, response, meta["transform"], bounds) if render_annotation else None
    return result, photo


def process_view(key, groups, legacy, index, held, configs, table, scratch, dest):
    tooth, angle = key
    rows = confirmed_rows(groups, tooth, angle)
    anchor = get_anchor(rows, legacy)
    if tooth in S0_SOURCE_IDS:
        corrected = [r for s in ("S1", "S2")
                     for r in process_group(rows, legacy, None, set(), state=s, label_dir=scratch)]
        index = index | {r["path"]: r for r in corrected}
        held = held - {(tooth, "S1"), (tooth, "S2")}
    config = configs[anchor.get("source_tooth_id", tooth), angle]
    anchor_meta = index[anchor["path"]]
    source = frame_native(anchor, anchor_meta, held)
    bounds = reference_crop(source.labels)
    base = with_crown(source, anchor_meta, table, anchor, bounds)
    x, y, x1, y1 = bounds
    a, b = np.array(config["crown_boundary"])
    opposite = dict(crown_boundary=[a, b], crown_side_point=2*a-np.array(config["crown_side_point"]))
    root0 = roi_labels(source.labels, opposite, np.eye(3))[y:y1, x:x1] == 1
    records = []
    for row in rows:
        record, photo = score(row, anchor, base, root0, bounds, config, index, held, table,
                              render_annotation=dest is not None)
        if dest is not None:
            publish(dest, record, photo)
        records.append(record)
    return records


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/area25"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    groups = inventory()
    legacy, index, held = indexes()
    configs = geometry()
    table = {r["image_path"]: r for r in read_csv(FEATURE_DIR / FEATURE_FILE)}
    keys = [(t,a) for t,a,m in sorted(groups) if m == "QLF"]
    records = []
    with TemporaryDirectory(prefix="qlf-huber25-masks-") as temp, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, view in enumerate(pool.map(lambda k: process_view(k, groups, legacy, index, held, configs, table,
                                                                  Path(temp), args.output_dir), keys), 1):
            records.extend(view)
            if i % 5 == 0:
                print(f"{i}/{len(keys)} views: {len(records)} images", flush=True)
    if len(records) != len(table) or len({r["annotation_path"] for r in records}) != len(records):
        raise ValueError("Annotation files must map one-to-one to feature records")
    records.sort(key=lambda row: (row["state_id"], row["tooth_id"], row["angle_id"], int(row["repeat_rank"])))
    write_csv(args.output_dir / "image-area.csv", records)
    write_csv(args.output_dir / "tooth-area.csv", summarize(records))
    print(Counter((r["state_id"], r["status"]) for r in records), flush=True)


if __name__ == "__main__":
    main()

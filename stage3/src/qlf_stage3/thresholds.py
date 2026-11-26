"""Draw fixed-threshold response areas on the current S0-to-S2 QLF pairs."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from .build_masks import process_group
from .dataset import DATA, cv2, get_anchor, inventory, np, read_csv, read_label, write_csv
from .direct_features import (FEATURE_DIR, FEATURE_FILE, S0_SOURCE_IDS, confirmed_rows, frame_native,
                            in_reference, indexes, measure_pair, reference_crop)

THRESHOLDS = (5, 10, 20, 30, 40)
SAMPLES = (
    ("T002", "A000"), ("T003", "A000"),
    ("T004", "A045"), ("T005", "A045"),
    ("T006", "A090"), ("T008", "A090"),
    ("T007", "A135"), ("T010", "A135"),
    ("T005", "A180"), ("T009", "A180"),
)


def by_physical_path(row):
    return row.get("physical_path", row["path"])


def choose_case(tooth, angle, groups, legacy, indexes_by_path, held, table, scratch):
    rows = confirmed_rows(groups, tooth, angle)
    anchor = get_anchor(rows, legacy)
    anchor_path = by_physical_path(anchor)
    eligible = [r for r in rows if r["state_id"] == "S2" and not r["is_duplicate"]
                and table[r["path"]]["loss_status"] == "automatic_qc_pass"]
    if not eligible:
        raise ValueError(f"{tooth}/{angle}: no eligible S2 capture passed registration and signal QC")
    current = min(eligible, key=lambda r: (abs(int(r["repeat_rank"]) - 4), int(r["repeat_rank"])))
    record = table[current["path"]]
    if record["s0_image_path"] != anchor_path:
        raise ValueError(f"Feature table points to a different baseline for {current['path']}")
    if tooth in S0_SOURCE_IDS:
        corrected = process_group(rows, legacy, None, set(), state="S2", label_dir=scratch)
        indexes_by_path = indexes_by_path | {r["path"]: r for r in corrected}
        held = held - {(tooth, "S2")}

    native = frame_native(anchor, indexes_by_path[anchor["path"]], held)
    bounds = reference_crop(native.labels)
    frames = []
    for row in (anchor, current):
        meta = indexes_by_path[row["path"]]
        src = native if row is anchor else frame_native(row, meta, held)
        mask_path = table[by_physical_path(row)]["crown_semantic_mask_path"]
        labels = read_label(DATA / mask_path)
        if labels.shape != src.labels.shape or np.any((labels > 0) & (labels != src.labels)):
            raise ValueError(f"Crown mask differs from the registered semantic mask: {mask_path}")
        frame = in_reference(replace(src, labels=labels), meta, anchor["path"], bounds, display=True)
        if frame is None:
            raise ValueError(f"Frame cannot be mapped to baseline: {row['path']}")
        frames.append(frame)

    loss, status, measures = measure_pair(*frames)
    if status != "automatic_qc_pass":
        raise ValueError(f"{tooth}/{angle} failed current loss QC: {status}")
    for column in ("valid_loss_roi_pixels", "mean_relative_loss_pct", "p95_loss_pct",
                   "loss_area_px", "loss_area_ratio_pct"):
        if not math.isclose(float(record[column]), measures[column], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{tooth}/{angle}: recomputed {column} differs from the feature CSV")

    valid_px = measures["valid_loss_roi_pixels"]
    counts = [int(np.count_nonzero(loss > threshold)) for threshold in THRESHOLDS]
    fractions = [100 * count / valid_px for count in counts]
    x, y, w, h = cv2.boundingRect(np.uint8((frames[0].labels > 0) | (frames[1].labels > 0)))
    roi = np.s_[max(0, y - 24):y + h + 24, max(0, x - 24):x + w + 24]
    photos = [cv2.cvtColor(frame.photo[roi], cv2.COLOR_BGR2RGB) for frame in frames]
    tag = f"{tooth}_{angle}_R{int(current['repeat_rank']):02}"
    stats = dict(sample=tag, tooth_id=tooth, angle_id=angle, repeat_rank=current["repeat_rank"],
                 comparison="S0→S2", s0_path=anchor_path, s2_path=current["path"],
                 exif_settings_status=record["exif_settings_status"], valid_pixels=valid_px,
                 mean_relative_loss_pct=measures["mean_relative_loss_pct"],
                 p95_loss_pct=measures["p95_loss_pct"])
    for threshold, count, fraction in zip(THRESHOLDS, counts, fractions, strict=True):
        stats[f"loss_gt_{threshold}_px"] = count
        stats[f"loss_gt_{threshold}_pct"] = fraction
    return stats, photos, loss[roi], fractions


def overview(destination, rows, *, title="S0 → S2（72 小时） · 五档阈值的响应面积"):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from .render_threshold import use_chinese_font

    use_chinese_font()

    data = np.array([[r[f"loss_gt_{t}_pct"] for t in THRESHOLDS] for r in rows])
    fig, (ax, table) = plt.subplots(1, 2, figsize=(13.2, 6.2),
                                    gridspec_kw={"width_ratios": [1.05, 1]}, layout="constrained")
    for line in data:
        ax.plot(THRESHOLDS, line, color="#527581", alpha=.50, lw=1.2, marker="o", ms=3)
    ax.plot(THRESHOLDS, np.median(data, axis=0), color="#a44429", lw=2.8,
            marker="o", label=f"{len(rows)} 张样本的中位数")
    ax.set(xlabel="相对荧光损失阈值（%）",
           ylabel="响应面积（占有效牙冠像素的百分比）", ylim=(0, 104), xticks=THRESHOLDS)
    ax.grid(axis="y", color="#e2e8eb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")

    image = table.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    table.set_xticks(range(len(THRESHOLDS)), [f">{t}%" for t in THRESHOLDS])
    table.xaxis.tick_top()
    table.set_yticks(range(len(rows)), [f'{r["tooth_id"]}  /  {int(r["angle_id"][1:])}°'
                                       for r in rows])
    table.tick_params(length=0)
    table.spines[:].set_visible(False)
    for (i, j), amount in np.ndenumerate(data):
        table.text(j, i, f"{amount:.1f}%", ha="center", va="center",
                   color="white" if amount >= 62 else "#102331")
    fig.colorbar(image, ax=table, orientation="horizontal", fraction=.055,
                 label="响应面积（%）")
    fig.suptitle(title, fontsize=17, weight="semibold")
    fig.savefig(destination, dpi=190, facecolor="white")
    plt.close(fig)


def main():
    from .render_threshold import render

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/thresholds"))
    parser.add_argument("--view", nargs=2, action="append", metavar=("TOOTH", "ANGLE"),
                        help="Select a view from this workspace. The default is the published ten-view panel.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    groups = inventory()
    legacy, index, held = indexes()
    table = {r["image_path"]:r for r in read_csv(FEATURE_DIR / FEATURE_FILE)}
    rows = []
    samples = [tuple(view) for view in args.view] if args.view else SAMPLES
    if len(set(samples)) != len(samples) or any((t, a, "QLF") not in groups for t, a in samples):
        parser.error("Each requested view must be unique and present. Use --view TOOTH ANGLE for a subset.")
    with TemporaryDirectory(prefix="qlf-threshold-masks-") as tmp, ThreadPoolExecutor(max_workers=4) as pool:
        cases = pool.map(lambda pair: choose_case(*pair, groups, legacy, index, held, table, Path(tmp)), samples)
        for stats, (base, current), loss, fractions in cases:
            title = f'{stats["sample"]}  |  S0 → S2 (72 h)  |  Valid crown: {stats["valid_pixels"]:,} px'
            render(args.output_dir / f'{stats["sample"]}.png', title, base, current, loss, THRESHOLDS, fractions)
            rows.append(stats)
            print(stats["sample"], [round(v, 2) for v in fractions], flush=True)
    write_csv(args.output_dir / "threshold-area.csv", rows)
    overview(args.output_dir / "threshold-area-curves.png", rows)


if __name__ == "__main__":
    main()

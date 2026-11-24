"""Recompute direct S0-referenced QLF heatmaps for the current manifest."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import math
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
from .build_masks import process_group
from .dataset import DATA, cv2, get_anchor, inventory, np, read_csv, read_label, write_csv
from .direct_features import (FEATURE_DIR, FEATURE_FILE, S0_SOURCE_IDS, confirmed_rows, frame_native,
                            in_reference, indexes, measure_pair, reference_crop)

PASS = "automatic_qc_pass"
STAGES = ("S1", "S2")
REASONS = {
    "registration_review_required": "Registration QC failed",
    "insufficient_signal_roi": "Insufficient valid green signal",
    "insufficient_common_roi": "Insufficient common crown",
    "identity_review_required": "Identity review required",
    "no_verified_alignment": "No verified alignment",
    "duplicate": "Duplicate capture",
}
COLOR = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#34373b")
DISPLAY_NORM = matplotlib.colors.Normalize(-100, 100, clip=True)


def physical(row):
    return row.get("physical_path", row["path"])


def aligned(row, metadata, held, table, anchor, bounds, frame=None, display=False):
    source = frame if frame is not None else frame_native(row, metadata, held)
    record = table[physical(row)]
    mask = read_label(DATA / record["crown_semantic_mask_path"])
    if mask.shape != source.labels.shape or not np.isin(mask, (0, 1, 2)).all():
        raise ValueError(f"Invalid reviewed crown mask: {physical(row)}")
    if np.any((mask > 0) & (mask != source.labels)):
        raise ValueError(f"Reviewed crown extends beyond matching semantic labels: {physical(row)}")
    return in_reference(replace(source, labels=mask), metadata, anchor, bounds, display=display)


def reproduce(base, target, record):
    loss, status, measures = measure_pair(base, target)
    if status != PASS:
        raise ValueError(f"Recomputed QC is {status}: {record['image_path']}")
    for column in ("valid_loss_roi_pixels", "mean_relative_loss_pct", "p95_loss_pct",
                   "loss_area_px", "loss_area_ratio_pct"):
        if not math.isclose(float(record[column]), measures[column], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Recomputed {column} differs: {record['image_path']}")
    valid = np.isfinite(loss)
    expected = 100 * (1 - target.green[valid].astype(float) / base.green[valid])
    np.testing.assert_allclose(loss[valid], expected, rtol=1e-11, atol=1e-11)
    if np.any(valid & ((base.labels != 1) | (target.labels != 1))):
        raise ValueError("A masked pixel appears in a heatmap")
    return loss


def repeat_tile(loss):
    rgba = COLOR(DISPLAY_NORM(np.ma.masked_invalid(loss)), bytes=True)
    im = Image.fromarray(rgba).convert("RGB")
    im.thumbnail((225, 185), Image.Resampling.NEAREST)
    return im


def collect(key, groups, legacy, index, held, table, scratch):
    tooth, angle = key
    rows = confirmed_rows(groups, tooth, angle)
    baseline = get_anchor(rows, legacy)
    anchor = baseline["path"]
    if tooth in S0_SOURCE_IDS:
        index = index | {
            rec["path"]: rec for state in STAGES
            for rec in process_group(rows, legacy, None, set(), state=state, label_dir=scratch)
        }
        held = held - {(tooth, state) for state in STAGES}
    native = frame_native(baseline, index[anchor], held)
    bounds = reference_crop(native.labels)
    base = aligned(baseline, index[anchor], held, table, anchor, bounds, frame=native, display=True)
    photos = [cv2.cvtColor(base.photo, cv2.COLOR_BGR2RGB)]
    losses, detail, decisions, tiles, ledger = [], [], [], {}, []
    for stage in STAGES:
        stage_rows = sorted((row for row in rows if row["state_id"] == stage),
                            key=lambda row: int(row["repeat_rank"]))
        eligible = [r for r in stage_rows if not r["is_duplicate"] and table[physical(r)]["loss_status"] == PASS]
        selected = min(eligible, key=lambda r: (abs(int(r["repeat_rank"]) - 4), int(r["repeat_rank"])),
                       default=None)
        primary_photo, primary_loss = None, None
        primary_detail = ""
        for row in stage_rows:
            image_path = physical(row)
            record = table[image_path]
            if record["s0_image_path"] != physical(baseline):
                raise ValueError(f"Stale baseline in feature table: {image_path}")
            if (record["tooth_id"], record["source_tooth_id"], record["is_duplicate"]) != (
                tooth, row.get("source_tooth_id", tooth), str(int(row["is_duplicate"]))
            ):
                raise ValueError(f"Stale identities in feature table: {image_path}")
            is_primary = row is selected
            status = "duplicate" if row["is_duplicate"] else record["loss_status"]
            rank = int(row["repeat_rank"])
            tile = None
            if status == PASS:
                target = aligned(row, index[row["path"]], held, table, anchor, bounds, display=is_primary)
                computed = reproduce(base, target, record)
                tile = repeat_tile(computed)
                if is_primary:
                    primary_photo = cv2.cvtColor(target.photo, cv2.COLOR_BGR2RGB)
                    primary_loss = computed
                    primary_detail = (f"R{rank:02} | Mean {float(record['mean_relative_loss_pct']):+.1f}% | "
                                      f"{int(record['valid_loss_roi_pixels']):,} px | "
                                      f"EXIF: {record['exif_settings_status']}")
            tiles[stage, rank] = tile, status, is_primary
            ledger.append({
                "tooth": tooth, "angle_deg": int(angle[1:]), "comparison": f"S0->{stage}",
                "target_repeat": rank, "representative": int(is_primary), "status": status,
                "s0_path": physical(baseline), "target_path": image_path, "duplicate_of": row["duplicate_of"],
                "s0_crown_mask": table[physical(baseline)]["crown_semantic_mask_path"],
                "target_crown_mask": record["crown_semantic_mask_path"],
                "exif_settings_status": record["exif_settings_status"],
                "valid_pixels": record["valid_loss_roi_pixels"] if status == PASS else "",
                "mean_loss_pct": record["mean_relative_loss_pct"] if status == PASS else "",
                "p95_loss_pct": record["p95_loss_pct"] if status == PASS else "",
            })
        if selected is None:
            unique = [r for r in stage_rows if not r["is_duplicate"]]
            fallback = min(unique, key=lambda r: (abs(int(r["repeat_rank"]) - 4), int(r["repeat_rank"])))
            record = table[physical(fallback)]
            frame = frame_native(fallback, index[fallback["path"]], held)
            x0, y0, x1, y1 = reference_crop(frame.labels)
            primary_photo = cv2.cvtColor(frame.photo[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            reason = record["loss_status"]
            if reason == "insufficient_signal_roi":
                coverage = float(record["valid_loss_roi_fraction"])
                primary_detail = f"R{int(fallback['repeat_rank']):02} 有效像素 {coverage:.1%}，低于 90% 门槛"
            else:
                primary_detail = {
                    "registration_review_required": "几何配准未通过质检",
                    "insufficient_common_roi": "共同牙冠面积或重叠率不足",
                    "identity_review_required": "样本编号待复核",
                }.get(reason, f"未通过质检 ({reason})")
        photos.append(primary_photo)
        losses.append(primary_loss)
        detail.append(primary_detail)
        decisions.append(len(eligible))
    return photos, losses, detail, decisions, tiles, ledger


def repeat_sheet(tooth, angle, tiles):
    canvas = Image.new("RGB", (1680, 585), "#f9fafb")
    draw = ImageDraw.Draw(canvas)
    from matplotlib.font_manager import findfont

    font_path = findfont("DejaVu Sans")
    title_font = ImageFont.truetype(font_path, 29)
    font = ImageFont.truetype(font_path, 17)
    draw.text((18, 13), f"{tooth}  /  {angle}°  ·  All repeat slots", fill="#172932", font=title_font)
    draw.text((18, 52), "Each column is one capture rank. * = atlas representative.", fill="#44515a", font=font)
    for stage_index, stage in enumerate(STAGES):
        y = 88 + stage_index * 230
        for rank in range(1, 8):
            tile, status, primary = tiles[stage, rank]
            x = 18 + (rank - 1) * 236
            draw.text((x, y), f"S0 → {stage}  ·  R{rank:02}{' *' if primary else ''}",
                      fill="#172932", font=font)
            if tile is not None:
                canvas.paste(tile, (x + (225 - tile.width) // 2, y + 22 + (185 - tile.height) // 2))
            else:
                draw.rectangle((x, y + 22, x + 224, y + 206), fill="#e7ecef")
                reason = {
                    "registration_review_required": "Registration QC failed",
                    "insufficient_signal_roi": "Insufficient green signal",
                    "insufficient_common_roi": "Insufficient crown overlap",
                }.get(status, REASONS.get(status, "QC failed"))
                draw.text((x + 112, y + 114), reason, anchor="mm", fill="#364550", font=font)
    values = np.linspace(-100, 100, 400)
    bar = Image.fromarray(COLOR(DISPLAY_NORM(values), bytes=True)[None, :, :]).convert("RGB")
    canvas.paste(bar.resize((400, 10)), (18, 544))
    for x, label in ((18, "≤−100%"), (196, "0%"), (370, "+100%")):
        draw.text((x, 557), label, fill="#25333b", font=font)
    draw.text((438, 540), "BLUE  GREEN SIGNAL INCREASE     RED  GREEN SIGNAL DECREASE     CHARCOAL  EXCLUDED PIXELS",
              fill="#32474e", font=font)
    return canvas


def cover(keys, counts, labels):
    from .atlas_plots import FONT

    font = FONT
    fig = plt.figure(figsize=(13.4, 8.4), facecolor="white")
    fig.text(.065, .90, "牙冠 QLF 荧光损失图册", fontsize=30, fontproperties=font, color="#122e37")
    fig.text(.069, .833, "同牙同角度以 S0 为基线  ·  S1 为 24 / 35 小时  ·  S2 为 72 小时",
             fontsize=15, fontproperties=font, color="#274a55")
    n1, n2 = (sum(row[i] > 0 for row in counts) for i in range(2))
    lines = [
        "图册覆盖 9 颗牙和 5 个角度，每个牙号与角度占一页。",
        f"通过质检的视图  S0→S1  {n1}/45    S0→S2  {n2}/45",
        "表内数字依次为两次比较通过质检的非复制连拍张数。0 表示只展示照片。",
        "代表图优先选第 4 张或最接近第 4 张的合格连拍。",
    ]
    for i, line in enumerate(lines):
        fig.text(.07, .775 - .041 * i, line, fontsize=12, fontproperties=font, color="#243943")
    ax = fig.add_axes([.11, .23, .76, .36])
    ax.axis("off")
    teeth = sorted({tooth for tooth, _ in keys})
    angles = sorted({angle for _, angle in keys})
    lookup = dict(zip(keys, counts, strict=True))
    table_data = [[f"{lookup[tooth, angle][0]} / {lookup[tooth, angle][1]}"
                   for angle in angles] for tooth in teeth]
    grid = ax.table(cellText=table_data, rowLabels=teeth, colLabels=labels, cellLoc="center", bbox=[0, 0, 1, 1])
    grid.auto_set_font_size(False)
    grid.set_fontsize(12)
    for (r, c), cell in grid.get_celld().items():
        cell.set_edgecolor("#d9e4e8")
        cell.set_facecolor("#e7f2f5" if r == 0 or c < 0 else "#f8f4e8" if "0" in cell.get_text().get_text() else "white")
    footer = [
        "逐像素相对损失 = 100 × (G0 − Gt) / G0。红色变暗，蓝色变亮，深灰为排除像素。",
        "共同有效牙冠经几何配准，排除牙根、背景、基线色斑、边缘、饱和及过暗的基线像素。",
        "当前 JPEG 含分阶段亮度调整。图中数值描述 G 通道变化，无法单独估计矿物损失。",
    ]
    for i, line in enumerate(footer):
        fig.text(.07, .166 - i * .043, line, fontsize=11.5, fontproperties=font, color="#243943")
    return fig


def main():
    from .atlas_plots import draw_view

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/atlas"))
    destination = parser.parse_args().output_dir
    destination.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    groups = inventory()
    legacy, index, held = indexes()
    table = {r["image_path"]: r for r in read_csv(FEATURE_DIR / FEATURE_FILE)}
    keys = sorted((t, a) for t, a, mode in groups if mode == "QLF")
    if len(keys) != 45:
        raise ValueError(f"Expected the 45 views in the current dataset, found {len(keys)}")
    panels_dir = destination / "views"
    panels_dir.mkdir(exist_ok=True)
    counts, ledger = [], []
    with TemporaryDirectory(prefix="qlf-s0-atlas-") as tmp:
        scratch = Path(tmp)
        pages = scratch / "pages.pdf"
        repeat_dir = scratch / "repeat"
        repeat_dir.mkdir()
        with (ThreadPoolExecutor(max_workers=3) as pool,
              PdfPages(pages) as pdf,
              ZipFile(destination / "all-repeats.zip", "w", compression=ZIP_DEFLATED, compresslevel=2) as archive):
            cases = pool.map(lambda key: collect(key, groups, legacy, index, held, table, scratch / "masks"), keys)
            for key, case in zip(keys, cases, strict=True):
                tooth, angle = key
                photos, losses, detail, decisions, tiles, records = case
                fig = draw_view(tooth, int(angle[1:]), photos, losses, detail)
                png = f"{tooth}_{angle}.png"
                fig.savefig(panels_dir / png, dpi=142)
                pdf.savefig(fig, dpi=145)
                plt.close(fig)
                sheet = repeat_dir / png
                repeat_sheet(tooth, int(angle[1:]), tiles).save(sheet)
                archive.write(sheet, png)
                counts.append(decisions)
                ledger.extend(records)
                print(tooth, angle, decisions, flush=True)
            write_csv(destination / "pairs.csv", ledger)
            archive.write(destination / "pairs.csv", "pairs.csv")

        from pypdf import PdfReader, PdfWriter

        front_path = scratch / "cover.pdf"
        with PdfPages(front_path) as front:
            fig = cover(keys, counts, [f"{angle}°" for angle in (0, 45, 90, 135, 180)])
            front.savefig(fig)
            plt.close(fig)
        writer = PdfWriter()
        writer.append(PdfReader(front_path))
        writer.append(PdfReader(pages))
        for page, (tooth, angle) in enumerate(keys, 1):
            writer.add_outline_item(f"{tooth} / {int(angle[1:])}°", page)
        with (destination / "s0-loss-atlas.pdf").open("wb") as target:
            writer.write(target)
    print(Counter((r["comparison"], r["status"]) for r in ledger), flush=True)


if __name__ == "__main__":
    main()

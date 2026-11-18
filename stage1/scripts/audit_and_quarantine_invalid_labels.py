"""Audit YOLO labels and (optionally) quarantine images with unusable boxes.

The Zenodo Dataset.zip contains a small number of annotation files whose YOLO
coordinates fall outside the image after normalisation (x/y centre > 1.0).
Ultralytics silently drops those images during training/validation, but the
dataset directory and manifest still list them, so reported split sizes and
the files on disk disagree. This script makes that discrepancy explicit:

1. scan every split and validate each label line (5 fields, class 0, finite
   values, positive width/height, box fully inside [0, 1]);
2. with ``--apply``, move every image whose label file has at least one
   invalid line into ``_invalid_annotations/<split>/{images,labels}``;
3. record the status in the manifest (``label_status`` / ``bad_boxes``) and
   rewrite the paths of quarantined rows;
4. write a JSON + Markdown report.

Quarantining reproduces exactly what ultralytics already did internally:
those images were skipped, so the trained weights and the 825-image test
metrics remain valid without retraining.

Usage:
  python scripts/audit_and_quarantine_invalid_labels.py            # dry run
  python scripts/audit_and_quarantine_invalid_labels.py --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public_rgb" / "full"
OUT_DIR = ROOT / "outputs" / "rgb_label_audit"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_label(path: Path):
    """Return (valid_boxes, bad_lines) for one YOLO label file."""
    boxes = []
    bad = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            bad.append({"line": lineno, "text": line, "reason": "expected 5 fields"})
            continue
        try:
            cls_f, x, y, w, h = (float(v) for v in parts)
        except ValueError:
            bad.append({"line": lineno, "text": line, "reason": "non-numeric value"})
            continue
        if not all(math.isfinite(v) for v in (cls_f, x, y, w, h)):
            bad.append({"line": lineno, "text": line, "reason": "non-finite value"})
            continue
        if int(cls_f) != cls_f or int(cls_f) != 0:
            bad.append({"line": lineno, "text": line, "reason": f"unexpected class {cls_f}"})
            continue
        x1, x2, y1, y2 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        if w <= 0 or h <= 0:
            bad.append({"line": lineno, "text": line, "reason": "non-positive size"})
            continue
        if min(x1, y1) < 0 or max(x2, y2) > 1:
            inside = max(0.0, min(1.0, x2) - max(0.0, x1)) * max(
                0.0, min(1.0, y2) - max(0.0, y1)
            )
            frac = inside / (w * h) if w * h else 0.0
            bad.append(
                {
                    "line": lineno,
                    "text": line,
                    "reason": "box outside [0,1]",
                    "inside_fraction": round(frac, 4),
                    "box_xyxy": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                }
            )
            continue
        boxes.append([x, y, w, h])
    return boxes, bad


def split_images(images_dir: Path):
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    images = {path.stem: path for path in paths}
    if len(images) != len(paths):
        raise ValueError(f"Duplicate image stems in {images_dir}")
    return images


def scan_split(data: Path, split: str):
    images = split_images(data / split / "images")
    labels_dir = data / split / "labels"
    label_files = sorted(labels_dir.glob("*.txt"))
    if not labels_dir.is_dir() or set(images) != {label.stem for label in label_files}:
        raise ValueError(f"Image/label pairing mismatch for split {split}: {data}")
    records = []
    for lf in label_files:
        good, bad = parse_label(lf)
        records.append(
            {
                "stem": lf.stem,
                "label": lf,
                "image": images[lf.stem],
                "valid_boxes": len(good),
                "bad_lines": bad,
            }
        )
    invalid = [r for r in records if r["bad_lines"]]
    return {
        "split": split,
        "label_files": len(records),
        "valid_images": len(records) - len(invalid),
        "valid_boxes": sum(r["valid_boxes"] for r in records if not r["bad_lines"]),
        "invalid_images": len(invalid),
        "invalid_boxes": sum(len(r["bad_lines"]) for r in invalid),
        "invalid": invalid,
    }


def scan_existing_quarantine(data: Path, split: str):
    """Re-read images already moved to _invalid_annotations (for idempotent reports)."""
    labels_dir = data / "_invalid_annotations" / split / "labels"
    images_dir = data / "_invalid_annotations" / split / "images"
    recs = []
    if not labels_dir.exists():
        return recs
    images = split_images(images_dir)
    labels = sorted(labels_dir.glob("*.txt"))
    if set(images) != {label.stem for label in labels}:
        raise ValueError(f"Incomplete quarantine for split {split}: {data}")
    for lf in labels:
        _, bad = parse_label(lf)
        recs.append(
            {
                "stem": lf.stem,
                "label": lf,
                "image": images[lf.stem],
                "valid_boxes": 0,
                "bad_lines": bad,
            }
        )
    return recs


def apply_quarantine(data: Path, splits, manifest_path: Path):
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = {"stem", "split", "image_path", "label_path", "box_count", "has_caries"}
        if not required.issubset(fieldnames):
            raise ValueError(f"Unexpected manifest columns: {manifest_path}")
        rows = list(reader)

    by_key = {(r["split"], r["stem"]): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError("Manifest contains duplicate image entries")

    moved = []
    for s in splits:
        split = s["split"]
        dest_img = data / "_invalid_annotations" / split / "images"
        dest_lbl = data / "_invalid_annotations" / split / "labels"
        for rec in s["invalid"]:
            src_lbl = rec["label"]
            src_img = rec["image"]
            dst_lbl = dest_lbl / src_lbl.name
            dst_img = dest_img / src_img.name
            if dst_lbl.exists() or dst_img.exists():
                raise FileExistsError(f"Quarantine destination already exists for {rec['stem']}")
            if not dst_lbl.resolve().is_relative_to(data) or not dst_img.resolve().is_relative_to(data):
                raise ValueError(f"Quarantine destination is outside the dataset: {rec['stem']}")
            row = by_key[(split, rec["stem"])]
            row["image_path"] = dst_img.relative_to(data).as_posix()
            row["label_path"] = dst_lbl.relative_to(data).as_posix()
            row["label_status"] = "invalid_coords_quarantined"
            row["bad_boxes"] = str(len(rec["bad_lines"]))
            row["box_count"] = str(len(rec["bad_lines"]))
            row["has_caries"] = ""
            moved.extend(((src_lbl, dst_lbl), (src_img, dst_img)))

    for row in rows:
        row.setdefault("label_status", "ok")
        row.setdefault("bad_boxes", "0")
    for col in ("label_status", "bad_boxes"):
        if col not in fieldnames:
            fieldnames.append(col)
    tmp = manifest_path.with_suffix(".csv.tmp")
    with tmp.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    completed = []
    try:
        for src, dst in moved:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            completed.append((src, dst))
        tmp.replace(manifest_path)
    except OSError:
        for src, dst in reversed(completed):
            shutil.move(dst, src)
        tmp.unlink(missing_ok=True)
        raise


def write_reports(out: Path, splits, quarantine, applied):
    out.mkdir(parents=True, exist_ok=False)
    details = []
    for s in splits:
        for r in s["invalid"]:
            details.append(
                {
                    "split": s["split"],
                    "stem": r["stem"],
                    "status": "quarantined_now" if applied else "to_quarantine",
                    "bad_boxes": len(r["bad_lines"]),
                    "bad_lines": r["bad_lines"],
                }
            )
    for split, recs in quarantine.items():
        for r in recs:
            if any(d["split"] == split and d["stem"] == r["stem"] for d in details):
                continue
            details.append(
                {
                    "split": split,
                    "stem": r["stem"],
                    "status": "already_quarantined",
                    "bad_boxes": len(r["bad_lines"]),
                    "bad_lines": r["bad_lines"],
                }
            )

    total_invalid_images = sum(s["invalid_images"] for s in splits) + sum(len(v) for v in quarantine.values())
    total_invalid_boxes = sum(s["invalid_boxes"] for s in splits) + sum(
        len(r["bad_lines"]) for v in quarantine.values() for r in v
    )
    quarantined_total = sum(len(v) for v in quarantine.values())
    summary = {
        "dataset": "Zenodo 14827784 Dataset.zip",
        "rule": "quarantine the whole image if any label line is invalid",
        "applied_this_run": applied,
        "quarantine_in_effect": bool(applied or quarantined_total),
        "quarantine_dir": "_invalid_annotations",
        "splits": {},
        "totals": {
            "active_label_files": sum(s["label_files"] for s in splits),
            "invalid_images": total_invalid_images,
            "invalid_boxes": total_invalid_boxes,
        },
        "details": details,
    }
    for s in splits:
        q = quarantine.get(s["split"], [])
        summary["splits"][s["split"]] = {
            "active_label_files": s["label_files"],
            "valid_images_before": s["valid_images"],
            "valid_boxes_before": s["valid_boxes"],
            "invalid_images_active": s["invalid_images"],
            "invalid_boxes_active": s["invalid_boxes"],
            "quarantined_images": len(q),
            "quarantined_boxes": sum(len(r["bad_lines"]) for r in q),
            "invalid_images_total": s["invalid_images"] + len(q),
            "invalid_boxes_total": s["invalid_boxes"] + sum(len(r["bad_lines"]) for r in q),
            "effective_images_after": s["valid_images"],
        }
    (out / "label_quality_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# YOLO 标签质量审计",
        "",
        f"- 规则：任一行标签非法，则整张图隔离（{summary['rule']}）。",
        f"- 隔离目录：`{summary['quarantine_dir']}`",
        "- 运行模式："
        + ("apply（本次已执行移动）" if applied else "dry-run（本次未移动）")
        + f"；隔离是否已生效：{'是' if summary['quarantine_in_effect'] else '否'}",
        "",
        "| split | 标签文件(当前) | 有效图 | 有效框 | 待隔离图 | 已隔离图 | 已隔离框 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in splits:
        q = summary["splits"][s["split"]]
        lines.append(
            f"| {s['split']} | {s['label_files']} | {s['valid_images']} | "
            f"{s['valid_boxes']} | {s['invalid_images']} | {q['quarantined_images']} | "
            f"{q['quarantined_boxes']} |"
        )
    lines += [
        "",
        "超出图像边界的框按固定规则隔离，未改变原始坐标。",
        "",
        "## 明细",
        "",
        "| split | 图像 stem | 状态 | 非法框数 |",
        "|---|---|---|---:|",
    ]
    for d in summary["details"]:
        lines.append(f"| {d['split']} | {d['stem']} | {d['status']} | {d['bad_boxes']} |")
    (out / "label_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def audit(data: Path, out: Path, apply: bool = False, manifest_path: Path | None = None):
    data = data.resolve()
    if out.exists():
        raise FileExistsError(f"Select a new report directory: {out}")
    splits = [scan_split(data, split) for split in SPLITS]
    if apply:
        apply_quarantine(data, splits, manifest_path or data / "manifest.csv")
        splits = [scan_split(data, split) for split in SPLITS]
    quarantine = {split: scan_existing_quarantine(data, split) for split in SPLITS}
    return write_reports(out, splits, quarantine, apply)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="move invalid images and update manifest")
    ap.add_argument("--data", type=Path, default=DATA, help="prepared dataset directory")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="new report directory")
    ap.add_argument("--manifest", type=Path, help="manifest path (defaults to manifest.csv under --data)")
    args = ap.parse_args()

    summary = audit(args.data, args.out, args.apply, args.manifest)
    print(json.dumps({"totals": summary["totals"], "splits": summary["splits"]}, ensure_ascii=False, indent=2))
    print(f"report: {args.out / 'label_quality_report.json'}")


if __name__ == "__main__":
    main()

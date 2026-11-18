"""Prepare the full ScientificData intraoral caries dataset (Zenodo 14827784).

The study's source document designates this dataset as the endpoint-1 main data
source: "Annotated intraoral image dataset for dental caries detection" — 6265
real intraoral photos (participants aged 10-24, dentist-annotated) provided in
YOLO / PascalVOC / COCO / LabelMe formats. We use the FULL ``Dataset.zip``
(NOT the pre-split ``Benchmarking Dataset.zip``) and build our own transparent,
fixed-seed **patient-level** train/val/test split, so no patient's teeth leak
across splits (the same patient appears in 5 views x 2 capture conditions).

The full archive is organized by {no_retractors, retractors, pilot} x
{Frontal, Left_Lateral, Mandibular, Maxillary_Occlusal, Right_Lateral}, with
Darknet_YOLO labels under ``Annotations/``. We merge the two decay classes
(permanent/primary, 恒牙龋/乳牙龋) into a single "caries" class, because the
study's detection endpoint localizes teeth with visible decay. Stage
classification is a separate experiment on CariesSpectra.

Outputs (data/public_rgb/full):
  data.yaml               # ultralytics dataset config (train/val/test -> images)
  {train,val,test}/images/*.jpg
  {train,val,test}/labels/*.txt   # single-class YOLO labels
  manifest.csv            # stem, split, subset, view, image_path, label_path, box_count, has_caries

Usage:
  python scripts/prepare_public_rgb.py --download --extract   # once (1.58 GB)
  python scripts/prepare_public_rgb.py --dataset-dir <Dataset> --out <new output>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from urllib.request import urlopen
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import yaml

from audit_and_quarantine_invalid_labels import audit


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "public_rgb"
ZENODO_URL = "https://zenodo.org/api/records/14827784/files/Dataset.zip/content"
ZENODO_MD5 = "89307871ac2f08f4a8f3a7da1f18db31"
OUT_DIR = DATA_ROOT / "full"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "caries"}  # single class: 龋损定位 (severity graded downstream)


def download_archive(dst: Path) -> None:
    """Keep an incomplete download out of the path used for preparation."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        with dst.open("rb") as archive:
            if hashlib.file_digest(archive, "md5").hexdigest() != ZENODO_MD5:
                raise ValueError(f"Checksum mismatch for {dst}; choose a fresh --zip-path")
        print(f"Verified existing archive: {dst}", flush=True)
        return
    partial = dst.with_name(dst.name + ".part")
    digest = hashlib.md5()
    print(f"Downloading {ZENODO_URL} (about 1.58 GB)", flush=True)
    with urlopen(ZENODO_URL, timeout=120) as response, partial.open("xb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != ZENODO_MD5:
        raise ValueError(f"Checksum mismatch for {partial}; choose a fresh --zip-path or remove that partial file")
    partial.rename(dst)


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for member in zf.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename or ":" in member.filename:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
            is_image = name.parts[:2] == ("Dataset", "Images") and name.suffix.lower() in IMAGE_EXTS
            is_label = name.parts[:3] == ("Dataset", "Annotations", "Darknet_YOLO") and name.suffix.lower() == ".txt"
            if not member.is_dir() and (is_image or is_label):
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise RuntimeError(f"Unsafe archive path: {member.filename}")
                members.append(member)
        if not members:
            raise ValueError("Dataset.zip has no expected image/label files")
        root.mkdir(parents=True, exist_ok=False)
        zf.extractall(root, members)


def collect_images(dataset_dir: Path) -> dict[str, Path]:
    images_dir = dataset_dir / "Images"
    files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len({p.stem for p in files}) != len(files):
        raise ValueError(f"Image filenames need unique stems: {images_dir}")
    return {p.stem: p for p in files}


def collect_labels(dataset_dir: Path) -> dict[str, Path]:
    labels_dir = dataset_dir / "Annotations" / "Darknet_YOLO"
    if not labels_dir.exists():
        raise FileNotFoundError(f"YOLO label directory missing: {labels_dir}")
    files = list(labels_dir.rglob("*.txt"))
    if len({p.stem for p in files}) != len(files):
        raise ValueError(f"Label filenames need unique stems: {labels_dir}")
    return {p.stem: p for p in files}


def count_boxes(src: Path | None) -> int:
    if src is None or not src.exists():
        return 0
    return sum(
        1
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(line.split()) >= 5
    )


PATIENT_RE = re.compile(r"^anonymous_(\d{3}[-_]\d{3}[-_]\d{3,4})[-_]\d{2}_")


def patient_key(stem: str) -> str | None:
    """Extract a patient/case id for patient-level split.

    Structured names use either hyphens or underscores between the id groups:
      ``anonymous_003-007-1004-00_<ts>_Frontal_View.jpg``  (hyphen)
      ``anonymous_003_007_312_00_<ts>_Frontal_View.jpg``   (underscore)
    The captured ``003-007-1004`` / ``003_007_312`` part is the patient, and the
    trailing ``-00``/``-01`` (or ``_00``/``_01``) are two capture conditions
    (with/without retractors) of the SAME patient, so we group by the three-part
    id and keep both conditions together. Pilot images
    (``anonymous-<view>-<ts>``) carry no patient id and return None.
    """
    m = PATIENT_RE.match(stem)
    return m.group(1) if m else None


def remap_label(src: Path | None, dst: Path) -> int:
    """Write a single-class label, merging permanent/primary decay into class 0."""
    boxes = 0
    lines: list[str] = []
    if src is not None and src.exists():
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 5 and parts[0] in {"0", "1"}:
                parts[0] = "0"  # merge permanent and primary caries
            lines.append(" ".join(parts))
            boxes += 1
    dst.write_text("\n".join(lines), encoding="utf-8")
    return boxes


def build_rows(dataset_dir: Path) -> list[dict]:
    images = collect_images(dataset_dir)
    labels = collect_labels(dataset_dir)
    rows: list[dict] = []
    for stem, img in sorted(images.items()):
        rel = img.relative_to(dataset_dir / "Images")
        parts = rel.parts
        subset = parts[0] if parts else ""
        view = parts[1] if len(parts) > 1 else ""
        lbl = labels.get(stem)
        boxes = count_boxes(lbl)
        rows.append(
            {
                "stem": stem,
                "image_path": str(img),
                "label_path": str(lbl) if lbl else "",
                "subset": subset,
                "view": view,
                "patient": patient_key(stem),
                "box_count": boxes,
                "has_caries": int(boxes > 0),
            }
        )
    return rows


def split_by_patient(rows: list[dict], val: float, test: float, seed: int) -> list[dict]:
    """Patient-level split: no patient's images leak across train/val/test.

    Images with a structured patient id are grouped by patient; patients are
    stratified by "has any caries" and split 70/15/15 with a fixed seed. Images
    without a patient id (the 760-image pilot batch) go to train only, so
    test/val never contain an unidentifiable patient.
    """
    rng = random.Random(seed)
    by_patient: dict[str, list[dict]] = defaultdict(list)
    ungrouped: list[dict] = []
    for r in rows:
        pid = r.get("patient")
        if pid:
            by_patient[pid].append(r)
        else:
            ungrouped.append(r)

    patients = list(by_patient.keys())
    pos = [p for p in patients if any(r["box_count"] > 0 for r in by_patient[p])]
    neg = [p for p in patients if all(r["box_count"] == 0 for r in by_patient[p])]
    rng.shuffle(pos)
    rng.shuffle(neg)

    out: list[dict] = []
    for group in (pos, neg):
        n = len(group)
        n_test = int(round(n * test))
        n_val = int(round(n * val))
        n_train = n - n_val - n_test
        buckets = {
            "train": group[:n_train],
            "val": group[n_train : n_train + n_val],
            "test": group[n_train + n_val :],
        }
        for split, ps in buckets.items():
            for p in ps:
                for r in by_patient[p]:
                    r = dict(r)
                    r["split"] = split
                    out.append(r)

    for r in ungrouped:
        r = dict(r)
        r["split"] = "train"  # pilot batch (no patient id) -> train only
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the full ScientificData caries dataset for YOLO detection.")
    parser.add_argument("--download", action="store_true", help="Download and verify the official Dataset.zip (1.58 GB).")
    parser.add_argument("--extract", action="store_true", help="Extract the downloaded zip.")
    parser.add_argument("--zip-path", type=Path, default=DATA_ROOT / "Dataset.zip")
    parser.add_argument("--extract-dir", type=Path, help="New extraction directory (default: extracted_full beside --out).")
    parser.add_argument("--dataset-dir", type=Path, help="Existing extracted Dataset directory containing Images and Annotations.")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="New prepared dataset directory.")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dataset_dir and (args.download or args.extract):
        parser.error("--dataset-dir cannot be used with --download or --extract")
    if not (0 < args.val < 1 and 0 < args.test < 1 and args.val + args.test < 1):
        parser.error("--val and --test must be positive and sum to less than 1")
    out = args.out.resolve()
    extract_dir = (args.extract_dir or out.parent / "extracted_full").resolve()
    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else extract_dir / "Dataset"
    if out.exists() or out.is_relative_to(dataset_dir):
        parser.error(f"Choose a new --out outside the source dataset: {out}")
    if args.extract and (extract_dir.exists() or out.is_relative_to(extract_dir) or extract_dir.is_relative_to(out)):
        parser.error("--extract-dir must be a new directory separate from --out")

    if args.download:
        download_archive(args.zip_path)
    if args.extract:
        if not args.zip_path.exists():
            raise FileNotFoundError(f"Missing zip: {args.zip_path}")
        safe_extract(args.zip_path, extract_dir)
    if not dataset_dir.exists():
        parser.error(f"Dataset folder not found: {dataset_dir}; use --extract or --dataset-dir")

    rows = split_by_patient(build_rows(dataset_dir), args.val, args.test, args.seed)

    if {row["split"] for row in rows} != {"train", "val", "test"}:
        parser.error("Source data and split fractions must yield nonempty train, val and test splits")
    out.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)

    manifest_rows: list[dict] = []
    for r in rows:
        src_img = Path(r["image_path"])
        src_lbl = Path(r["label_path"]) if r["label_path"] else None
        sp = r["split"]
        dst_img = out / sp / "images" / src_img.name
        dst_lbl = out / sp / "labels" / (src_img.stem + ".txt")
        shutil.copy2(src_img, dst_img)
        boxes = remap_label(src_lbl, dst_lbl)
        manifest_rows.append(
            {
                "stem": r["stem"],
                "split": sp,
                "subset": r["subset"],
                "view": r["view"],
                "patient": r["patient"] or "",
                "image_path": dst_img.relative_to(out).as_posix(),
                "label_path": dst_lbl.relative_to(out).as_posix(),
                "box_count": boxes,
                "has_caries": int(boxes > 0),
            }
        )

    cfg = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    fieldnames = ["stem", "split", "subset", "view", "patient", "image_path", "label_path", "box_count", "has_caries"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Dataset prepared -> {out} (counts before label audit)")
    for sp in ("train", "val", "test"):
        sub = [r for r in manifest_rows if r["split"] == sp]
        pos = sum(1 for r in sub if r["has_caries"])
        print(f"  {sp}: {len(sub)} images ({pos} positive, {len(sub) - pos} background)")
    patients = {r["patient"] for r in manifest_rows if r["patient"]}
    pilot = sum(1 for r in manifest_rows if not r["patient"])
    print(f"total images: {len(manifest_rows)} | distinct patients: {len(patients)} | pilot (no id): {pilot}")

    # The source archive has a few annotation files whose boxes lie outside the
    # image. Ultralytics silently skips those images, so quarantine them here
    # too: split directories then match the data actually used for training and
    # evaluation, and the manifest records why.
    result = audit(out, out / "audit", apply=True, manifest_path=out / "manifest.csv")
    for split, counts in result["splits"].items():
        print(f"  {split}: {counts['effective_images_after']} images ({counts['quarantined_images']} quarantined)")


if __name__ == "__main__":
    main()

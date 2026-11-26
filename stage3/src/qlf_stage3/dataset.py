"""Current-manifest image group lookup and S0 baseline registration for C04."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "config"
from .settings import LOCAL_VIEWS, PROJECT
from .image_io import read_image, write_png
DATA = PROJECT / "Data"
GEN2 = "03_较重度脱矿_2代"
SEM = DATA / "数据处理代码" / "语义掩膜_背景与基线黑斑"
from .baseline_masks import (assess, canonical_frame, describe, homogeneous,
                                       landmark_registration, register_checked, sift_points)

STATES = ("S0", "S1", "S2")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def write_index(path, records):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        for item in records:
            print(json.dumps(item, ensure_ascii=False, allow_nan=False), file=stream)
    pending.replace(path)


def read_index(path):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {r["path"]: r for r in rows}


def data_path(row):
    return row["file_path"].removeprefix("../Data_rgb_corrected/")


def inventory():
    rows = read_csv(DATA / "manifest.csv")
    if LOCAL_VIEWS.is_file():
        layout = json.loads(LOCAL_VIEWS.read_text(encoding="utf-8"))
        if layout["schema"] not in ("teeth_student.stage3.v1", "qlf_stage3.views.v1"):
            raise ValueError(f"Unsupported layout in {LOCAL_VIEWS}")
        groups = defaultdict(list)
        for row in rows:
            row["path"] = row["file_path"]
            row["is_duplicate"] = row["is_duplicate"] == "1"
            if row["state_id"] == "S2" and row["batch_id"] != "C04":
                raise ValueError(f"Expected S2 C04: {row['path']}")
            groups[row["tooth_id"], row["angle_id"], row["modality"]].append(row)
        expected = {(t, a, m) for t, a in layout["views"] for m in ("QLF", "RGB")}
        repeats = layout.get("repeats_per_view", 7)
        slots = Counter((s, str(i)) for s in STATES for i in range(1, repeats + 1))
        if not expected or set(groups) != expected or any(
            Counter((r["state_id"], r["repeat_rank"]) for r in group) != slots
            for group in groups.values()
        ):
            raise ValueError("Each selected view needs its declared repeated RGB/QLF pairs in each stage")
        return groups
    source = {f'{GEN2}/{r["file"]}': r for r in read_csv(DATA / GEN2 / "_整理清单.csv")}
    groups = defaultdict(list)
    for r in rows:
        path = data_path(r)
        r["path"] = path
        if r["mask_path"].startswith("masks/"):
            r["mask_path"] = f'数据处理代码/{r["mask_path"]}'
        r["duplicate_of"] = ""
        r["is_duplicate"] = False
        if r["state_id"] == "S2":
            assert r["batch_id"] == "C04" and path in source
            s = source[path]
            assert (r["tooth_id"], r["angle_id"], r["modality"], r["repeat_rank"]) == (
                s["tooth_id"], s["angle_id"], s["modality"], s["repeat_rank"])
            r["is_duplicate"] = s["is_duplicate"] == "1"
            if r["is_duplicate"]:
                r["duplicate_of"] = Path(path).with_name(s["duplicate_of"]).as_posix()
                assert source[r["duplicate_of"]]["is_duplicate"] == "0"
                if r["modality"] == "QLF":
                    assert s["source_sha256"] == source[r["duplicate_of"]]["source_sha256"]
                    assert hashlib.sha256((DATA / path).read_bytes()).digest() == hashlib.sha256(
                        (DATA / r["duplicate_of"]).read_bytes()).digest()
        groups[r["tooth_id"], r["angle_id"], r["modality"]].append(r)
    assert len(rows) == 1890 and len(groups) == 90
    assert all(Counter(r["state_id"] for r in rr) == dict.fromkeys(STATES, 7) for rr in groups.values())
    return groups


def get_anchor(rows, legacy):
    bases = [r for r in rows if r["state_id"] == "S0"]
    return next(r for r in bases if legacy[r["path"]]["status"] == "anchor")


def read_rgb(row):
    path = DATA / row.get("physical_path", row["path"])
    return read_image(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


def read_label(path):
    label = read_image(path, cv2.IMREAD_GRAYSCALE)
    if not label.any():
        raise ValueError(f"Unreadable/empty label {path}")
    return label


def packed(row, mask):
    shape = np.uint8(mask > 0)
    x, y, w, h = cv2.boundingRect(shape)
    corners = np.array([[x,y], [x+w-1,y], [x,y+h-1], [x+w-1,y+h-1]])
    f = canonical_frame(row["path"], read_rgb(row), shape)
    f.rgb = None
    return f, corners


def registered(a, b, points, landmark=True):
    matrix, quality = register_checked(a, b, points)
    if quality["candidate_ok"]:
        return matrix, "registered", describe(quality)
    if landmark:
        candidate, extra = landmark_registration(a,b,sift_points(a),sift_points(b))
        if candidate is not None:
            return candidate, "registered_landmarks", extra
    return None, "registration_review_required", describe(quality)


def composition_passes(a, b, matrix):
    aligned = b.canonical @ homogeneous(matrix) @ np.linalg.inv(a.canonical)
    fwd = assess(a,b,aligned[:2])
    back = assess(b,a,np.linalg.inv(aligned)[:2])
    return (all(q["plausible"] and q["dice"] >= .90 and q["gray_ncc"] >= .25 for q in (fwd,back)),
            fwd["dice"])


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.png")
    write_png(temp, mask)
    temp.replace(path)

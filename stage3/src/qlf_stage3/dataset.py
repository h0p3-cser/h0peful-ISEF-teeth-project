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
from .settings import PROJECT
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
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"Unreadable image {path}")
    return img


def read_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None or not label.any():
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
    if not cv2.imwrite(str(temp), mask, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Cannot write {temp}")
    temp.replace(path)

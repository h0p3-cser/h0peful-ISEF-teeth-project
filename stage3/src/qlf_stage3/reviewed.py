"""Supply the author's saved registration results to the pinned Stage 3 exporter."""

import csv
import hashlib
import io
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_photos(bundle, table, data, output):
    with bundle.open("paired-photo-sha256.csv") as raw:
        photos = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")))
    if len(photos) != len(table) or {r["image_path"] for r in photos} != table.keys():
        raise ValueError("配准包的逐图清单与工作区不一致")
    mismatches = []
    for row in photos:
        qlf, rgb = row["image_path"], row["rgb_path"]
        if rgb != table[qlf]["rgb_path"]:
            raise ValueError(f"配准包和工作区的 RGB 配对不同: {qlf}")
        if digest(data / qlf) != row["qlf_sha256"]:
            raise ValueError(f"QLF 照片与作者配准包哈希不符: {qlf}")
        actual = digest(data / rgb)
        if actual != row["rgb_sha256"]:
            mismatches.append(dict(qlf_path=qlf, rgb_path=rgb, author_sha256=row["rgb_sha256"],
                                   local_sha256=actual))
    if mismatches and output is not None:
        with (output / "rgb-hash-mismatches.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(mismatches[0]))
            writer.writeheader()
            writer.writerows(mismatches)
    return dict(pairs=len(photos), qlf_sha256_matches=len(photos),
                corrected_rgb_sha256_matches=len(photos) - len(mismatches),
                corrected_rgb_sha256_mismatches=len(mismatches))


def load_registrations(bundle, table, scratch, changed_labels):
    from .dataset import DATA, SEM, np
    from .direct_features import S0_SOURCE_IDS

    expected = {key for key, row in table.items()
                if row["tooth_id"] in S0_SOURCE_IDS and row["state_id"] in ("S1", "S2")}
    entries = [json.loads(s) for s in bundle.read("corrected-qlf-transforms.jsonl").splitlines() if s.strip()]
    if len(entries) != len(expected) or {r["path"] for r in entries} != expected:
        raise ValueError("复核配准的图号有缺失、重复或多余项")
    labels_seen = set()
    for i, entry in enumerate(entries):
        row = table[entry["path"]]
        relative = (DATA / row["semantic_mask_path"]).relative_to(SEM).as_posix()
        if entry["label_path"] != relative:
            raise ValueError(f"配准标签与作者逐图表不符: {entry['path']}")
        matrix = entry["transform"]
        if matrix is not None and (np.shape(matrix) != (2, 3) or not np.isfinite(matrix).all()):
            raise ValueError(f"无效的仿射矩阵: {entry['path']}")
        if relative in changed_labels:
            # Same scratch-mask convention as upstream build_masks.process_group.
            path = scratch / f"label-{i}.png"
            path.write_bytes(bundle.read("recomputed-labels/" + relative))
            entry["label_path"] = str(path)
            labels_seen.add(relative)
        elif not (SEM / relative).is_file():
            raise FileNotFoundError(SEM / relative)
    if labels_seen != set(changed_labels):
        raise ValueError("配准包声明的重算标签与工作区不符")
    return {r["path"]: r for r in entries}


def group_reader(records):
    from .dataset import get_anchor

    def read_group(rows, legacy, binary, suspect, *, state, label_dir):
        anchor = get_anchor(rows, legacy)["path"]
        selected = []
        for row in rows:
            if row["state_id"] == state:
                saved = records[row["path"]]
                identity = (row["tooth_id"], row["angle_id"], state, row["modality"],
                            int(row["repeat_rank"]), anchor)
                if identity != tuple(saved[k] for k in ("tooth", "angle", "state", "modality", "rank", "s0_anchor")):
                    raise ValueError(f"复核记录与当前帧或 S0 参考不符: {row['path']}")
                selected.append(saved)
        return selected

    return read_group

"""Locked image identities, the historical comparator, and native crown measurements."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..files import load, sha
from ..rgb.color import CHANNELS, apply_rgb, decode, pixel_hash
from ..rgb.geometry import polygon_mask, resize_matrix, transform, white_body


SLOTS = (("T006", "A000"), ("T008", "A000"), ("T007", "A045"))
OLD_FIT = frozenset(("ok", "ok_no_holdout"))
OLD_OUTPUT = OLD_FIT | {"reference"}


@dataclass
class PilotStudy:
    frames: pd.DataFrame
    anchors: pd.DataFrame
    old_qc: pd.DataFrame
    inputs: dict


def load_pilot(data, study_run, root_run):
    for directory in (data, study_run, root_run):
        if not Path(directory).is_dir():
            raise FileNotFoundError(directory)
    if load(root_run/"status.json").get("status") != "complete":
        raise ValueError("Historical root run is incomplete")
    paths = [study_run/"evaluation/sample_manifest.csv", root_run/"rgb_calibration.csv",
             root_run/"anchors.csv", root_run/"qc_by_frame.csv", root_run/"config.json"]
    manifest, historical, anchors, qc = (pd.read_csv(p) for p in paths[:4])
    keep = pd.MultiIndex.from_frame(manifest[["tooth_id","angle_id"]]).isin(SLOTS)
    cohort = manifest.loc[keep].copy()
    ids = set(cohort.rgb_image_id)
    historical = historical.loc[historical.rgb_image_id.isin(ids)].copy()
    keys = ["sample_id", "rgb_image_id", "tooth_id", "angle_id", "state_id", "session_id",
            "repeat_rank", "rgb_event", "qlf_event", "group_id", "qlf_image_id"]
    frames = cohort[keys+["rgb_resolved", "rgb_sha256", "rgb_pixel_sha256"]].merge(
        historical, on=keys, how="left", validate="one_to_one", indicator=True)
    if (not frames.rgb_image_id.is_unique or not frames["_merge"].eq("both").all() or
            not frames.rgb_resolved.eq(frames.source_path).all() or
            not frames.rgb_sha256.eq(frames.expected_source_sha256).all()):
        raise ValueError("The formal cohort does not agree with the historical white-light run")
    if set(zip(frames.tooth_id,frames.angle_id)) != set(SLOTS):
        raise ValueError("A locked pilot combination is missing")
    frames = frames.drop(columns="_merge").sort_values(["tooth_id","angle_id","state_id","repeat_rank","rgb_image_id"])
    anchors = anchors[anchors.anchor_rgb_id.isin(frames.anchor_rgb_id)].set_index("anchor_rgb_id",verify_integrity=True)
    qc = qc.loc[qc.rgb_image_id.isin(ids)].set_index("rgb_image_id",verify_integrity=True)
    if set(qc.index)!=ids or set(anchors.index)!=set(frames.anchor_rgb_id):
        raise ValueError("Historical anchor or frame quality-control identity mismatch")
    tag=frames.set_index("rgb_image_id")[["tooth_id","angle_id"]]
    if not np.array_equal(tag.loc[frames.anchor_rgb_id].to_numpy(),frames[["tooth_id","angle_id"]].to_numpy()):
        raise ValueError("Each reference must belong to the same tooth and view as its current frame")
    for anchor_id, row in anchors.iterrows():
        source = frames[frames.rgb_image_id.eq(anchor_id)]
        if len(source)!=1 or source.iloc[0].state_id!="S0" or source.iloc[0].rgb_sha256!=row.anchor_sha256:
            raise ValueError("Expected the recorded same-view S0 reference in this pilot")
    return PilotStudy(frames, anchors, qc, {p.name:sha(p) for p in paths})


def read_source(data, row):
    path = data/row.source_path
    if sha(path)!=row.rgb_sha256:
        raise ValueError(f"White-light source bytes changed for {row.rgb_image_id}")
    raw, info = decode(path)
    if pixel_hash(raw)!=row.rgb_pixel_sha256:
        raise ValueError(f"Decoded source pixels changed for {row.rgb_image_id}")
    return raw, info


def old_png(root_run, row):
    if row.status not in OLD_OUTPUT:
        if pd.notna(row.corrected_path):
            raise ValueError("Historical failure unexpectedly has an output")
        return None
    path = root_run/row.corrected_path
    if sha(path)!=row.corrected_sha256:
        raise ValueError("Changed historical corrected PNG: "+row.rgb_image_id)
    pixels, _ = decode(path, corrected=True)
    if pixel_hash(pixels)!=row.corrected_pixel_sha256:
        raise ValueError("Changed historical corrected pixels: "+row.rgb_image_id)
    return pixels


def historical_crown(data, row, anchor, raw, info):
    """Reconstruct the exact native ROI and corresponding center used in the root report."""
    if not isinstance(row.reference_to_current_work,str):
        return None, None
    width,height = raw.shape[1],raw.shape[0]
    work = resize_matrix(width,height,(int(row.working_width),int(row.working_height)))
    matrix = np.linalg.inv(work) @ np.array(json.loads(row.reference_to_current_work))
    outline = transform(np.asarray(json.loads(anchor.crown_work)),matrix)
    mask = polygon_mask(raw.shape[:2],outline) & white_body(data/row.mask_path,info,raw)[0]
    return mask, outline.mean(axis=0)


def old_clipping(row, old_qc, raw, crown, saved):
    if saved is None or crown is None or pd.isna(old_qc.crown_pixels):
        return None
    slopes = [row["a_"+c] for c in CHANNELS]
    offsets = [row["b_"+c] for c in CHANNELS]
    expected, stats, new_clip = apply_rgb(raw,slopes,offsets,crown)
    if (not np.array_equal(expected,saved) or stats["crown_pixels"]!=old_qc.crown_pixels or
            stats["crown_new_clip_px"]!=old_qc.crown_new_clip_px or
            abs(stats["crown_new_clip_pct"]-old_qc.crown_new_clip_pct)>1e-9):
        raise ValueError("Historical pre-clipping audit differs for "+row.rgb_image_id)
    return stats,new_clip & crown

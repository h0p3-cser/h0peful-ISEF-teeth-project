"""Bind the published paired cohort and known S0 events to native white-light records."""
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path

from ..files import FEATURE_CSV, SOURCE_TABLE, load, sha

SEM = Path("Data/数据处理代码/语义掩膜_背景与基线黑斑")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def keyed(items,key):
    result = {}
    for row in items:
        if not row[key] or row[key] in result:
            raise ValueError("Missing or duplicate "+key)
        result[row[key]]=row
    return result


@dataclass
class Photo:
    image_id: str
    path: Path
    relative: str
    digest: str
    body: Path
    mask_relative: str
    key: str
    archive: dict
    event: str


def select_anchor(native_anchors, paired_reference, calibration_event, regions):
    """Only the known calibration event list and its geometry are eligible for fallback."""
    if len(native_anchors)>1:
        raise ValueError("Multiple declared white-light references for this tooth and view")
    if native_anchors:
        return native_anchors[0],"documented_white_light_reference"
    if paired_reference is not None:
        return paired_reference,"reviewed_pair_of_qlf_anchor"
    # For undocumented references, only pass candidates already checked for focus and visibility.
    if calibration_event:
        import numpy as np
        ranks = [int(rank) for _,rank in calibration_event]
        middle = float(np.median(ranks))
        eligible = [(p,int(rank)) for p,rank in calibration_event if p.image_id in regions]
        if eligible:
            return min(eligible,key=lambda t:(abs(t[1]-middle),t[1],t[0].image_id))[0],"earliest_visible_s0_event"
    return None,"no_anchor"


def read_study(data,run,regions):
    workspace=data/"workspace"
    measured=run/"measurements"
    manifest=run/"evaluation/sample_manifest.csv"
    features=measured/FEATURE_CSV
    verification=load(measured/"training-features-255-thr30-verification.json")
    if sha(features)!=verification["csv_sha256"]:
        raise ValueError("Locked four-feature CSV differs from its export audit")
    frows=keyed(rows(features),"sample_id")
    samples=keyed(rows(manifest),"sample_id")
    if set(samples)!=set(frows):
        raise ValueError("The original manifest and QLF CSV have different frame IDs")
    if len({r["rgb_image_id"] for r in samples.values()})!=len(samples):
        raise ValueError("Repeated RGB ID in the paired source manifest")
    for sid,row in samples.items():
        checked=frows[sid]
        for k in ("qlf_path","rgb_path","tooth_id","angle_id","state_id","repeat_rank"):
            if row[k]!=checked[k]:
                raise ValueError("Different formal pairing: "+sid)
        for k in verification["feature_columns"]:
            if abs(float(row[k])-float(checked[k]))>1e-10:
                raise ValueError("Manifest QLF value differs from CSV: "+sid)
    inputs=[features,manifest,run/"evaluation/folds.csv",workspace/SOURCE_TABLE,workspace/"Data/manifest.csv",
            workspace/"local-photo-bindings.csv",data/"catalog/images.csv",
            workspace/SEM/"legacy_index.jsonl",workspace/SEM/"gen2_index.jsonl"]
    hashes={p.name:sha(p) for p in inputs}
    native=keyed([r for r in rows(workspace/"Data/manifest.csv") if r["modality"]=="RGB"],"file_path")
    binding=keyed(rows(workspace/"local-photo-bindings.csv"),"workspace_path")
    catalog=keyed(rows(data/"catalog/images.csv"),"image_id")
    index={}
    for p in inputs[-2:]:
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            entry=json.loads(line)
            if entry["modality"]=="RGB":
                index[entry["path"]]=entry
    reviewed=rows(workspace/SOURCE_TABLE)
    if sha(workspace/SOURCE_TABLE)!=verification["source_table_sha256"]:
        raise ValueError("Reviewed pairing table changed")
    photos,known,qlf_pair={},defaultdict(list),{}
    source_meta={}
    for row in reviewed:
        path=(workspace/"Data"/row["rgb_path"]).resolve()
        rel=path.relative_to(workspace).as_posix()
        bound=binding[rel]
        if bound["modality"]!="RGB" or bound["image_id"]!=path.stem.rsplit("_",1)[-1]:
            raise ValueError("White-light binding mismatch: "+rel)
        image_id=bound["image_id"]
        key=row["rgb_path"].removeprefix("../Data_rgb_corrected/")
        body_rel=Path(native[row["rgb_path"]]["mask_path"])
        body=workspace/"Data"/("数据处理代码" if body_rel.parts[0]=="masks" else "")/body_rel
        photo=Photo(image_id,path,path.relative_to(data).as_posix(),bound["sha256"],body,
                    body.relative_to(data).as_posix(),key,index.get(key,{}),catalog[image_id]["capture_event_id"])
        photos[image_id]=photo
        qlf_pair[row["image_path"]]=photo
        source_meta[image_id]=row
        if row["state_id"]=="S0":
            known[row["tooth_id"],row["angle_id"]].append((photo,row))
    groups=defaultdict(list)
    for row in samples.values():
        photo=photos[row["rgb_image_id"]]
        if (row["rgb_resolved"]!=photo.relative or row["rgb_sha256"]!=photo.digest or
                row["rgb_path"]!=source_meta[photo.image_id]["rgb_path"]):
            raise ValueError("White-light path or locked source differs: "+row["sample_id"])
        groups[row["tooth_id"],row["angle_id"]].append(row)
    anchors={}
    for slot,members in groups.items():
        options=known[slot]
        native_refs=[p for p,_ in options if p.archive.get("status")=="anchor" and p.archive.get("s0_anchor")==p.key]
        paired=qlf_pair.get(members[0]["anchor_path"])
        if paired and paired.image_id not in {p.image_id for p,_ in options}:
            raise ValueError("Cross-view/identity calibration reference")
        fallback=[]
        if not native_refs and not paired:
            from .calibration import visible_for_fallback
            events=defaultdict(list)
            for p,row in options:
                cat=catalog[p.image_id]
                if cat["capture_event_id"]:
                    events[cat["capture_event_id"]].append((p,row))
            for event in sorted(events,key=lambda e:min(catalog[p.image_id]["capture_datetime"] for p,_ in events[e])):
                fallback=[(p,r["repeat_rank"]) for p,r in events[event] if visible_for_fallback(p,regions)]
                if fallback:
                    break
        anchors[slot]=select_anchor(native_refs,paired,fallback,regions)
    return groups,photos,anchors,hashes


def pilot_slots(groups,cfg):
    chosen=set()
    for tooth in cfg["pilot_teeth"]:
        for preferred in cfg["pilot_angles"]:
            slot=tooth,preferred
            chosen.add(slot)
            if {r["state_id"] for r in groups.get(slot,[])}!={"S0","S1","S2"}:
                order=sorted(cfg["angle_order"],key=lambda a:(abs(int(a[1:])-int(preferred[1:])),cfg["angle_order"].index(a)))
                for a in order:
                    if {r["state_id"] for r in groups.get((tooth,a),[])}=={"S0","S1","S2"}:
                        chosen.add((tooth,a))
                        break
    return chosen

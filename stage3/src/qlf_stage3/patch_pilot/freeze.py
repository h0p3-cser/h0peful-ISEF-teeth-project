"""Seal native pixel sets, one observation per physical identity, and the pre-fit split."""
import importlib.metadata
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..files import PACKAGE, dump, load, sha
from .prepare import SOURCES, checked_source, roots
from .protocol import (PORTABLE_SPEC_HASH, SPEC_HASH, checked_files, digest_indices, json_lines, mask_pixels, medians,
                       read_lines, solver_cfg, span, split)


def combine(im,records):
    groups={}
    for r in records:
        physical=r["physical_id"]
        if not physical or not isinstance(physical,str) or not physical.replace("_","").isalnum():
            raise ValueError("Invalid physical identity")
        groups.setdefault(physical,[]).append(r)
    masks,observations={},{}
    for pid,regions in sorted(groups.items()):
        accepted=[r for r in regions if r["confirmed"] and r["visibility"]=="present"]
        used=accepted or [r for r in regions if r["visibility"]=="present"]
        candidates=[mask_pixels(im,r["polygons"],r.get("exclude_polygons",[])) for r in used]
        indices=np.unique(np.concatenate(candidates)) if candidates else np.empty(0,np.int32)
        masks[pid]=indices
        observations[pid]=dict(confirmed=bool(accepted),visible=bool(used),pixels=len(indices),
            linear=medians(im,indices,True),srgb=medians(im,indices),mask_sha256=digest_indices(indices))
    used=[(pid,masks[pid]) for pid,r in observations.items() if r["confirmed"]]
    for i,(pid,ix) in enumerate(used):
        for other,jx in used[:i]:
            if np.intersect1d(ix,jx,assume_unique=True).size:
                raise ValueError("One sensor pixel belongs to multiple physical identities: "+pid+" / "+other)
    return masks,observations


def paired_blocks(obs,target,confirmed=True):
    return {p:(r["linear"],target[p]["linear"]) for p,r in obs.items() if p in target
            and min(r["pixels"],target[p]["pixels"])>=25
            and (not confirmed or (r["confirmed"] and target[p]["confirmed"]))}


def source_fingerprints():
    paths=list((PACKAGE/"patch_pilot").glob("*.py"))
    paths += [PACKAGE/"rgb/color.py",PACKAGE/"rgb/geometry.py",PACKAGE/"materials/study.py",
              PACKAGE/"materials/figures.py",PACKAGE/"exposure.py",PACKAGE/"files.py"]
    return {p.relative_to(PACKAGE).as_posix():sha(p) for p in sorted(paths)}


def freeze(prepared,regions):
    prepared=Path(prepared).resolve()
    if (prepared/"protocol-lock.json").exists():
        raise FileExistsError("This experiment is already frozen")
    if Path(regions).resolve()!=prepared/"regions.jsonl":
        (prepared/"regions.jsonl").write_bytes(Path(regions).read_bytes())
    reviewed=read_lines(prepared/"regions.jsonl")
    annotations={v["rgb_image_id"]:v for v in reviewed}
    manifest=pd.read_csv(prepared/"pilot_manifest.csv",dtype=str,keep_default_na=False).to_dict("records")
    if len(annotations)!=len(reviewed) or set(annotations)!={r["rgb_image_id"] for r in manifest}:
        raise ValueError("Need exactly one review record for every paired identity")
    dirs=roots(prepared)
    provenance=load(prepared/"inputs.json")
    for name,(kind,path) in SOURCES.items():
        if sha(dirs[kind]/path)!=provenance["source_tables"][name]:
            raise ValueError("Control table changed after prepare: "+name)
    cfg=solver_cfg(load(dirs["baseline"]/SOURCES["config"][1])["protocol"])
    from .figures import visual_review
    visual_review(prepared)
    (prepared/"masks").mkdir(exist_ok=True)
    observations={}
    geometry={g["rgb_image_id"]:g for g in read_lines(prepared/"baseline-rows.jsonl")}
    inputs={}
    for row in manifest:
        fid=row["rgb_image_id"]
        note=annotations[fid]
        if (note["source_sha256"]!=row["source_sha256"] or note["anchor_rgb_id"]!=row["anchor_rgb_id"]):
            raise ValueError("Annotation identity mismatch: "+fid)
        historical=geometry[fid]["baseline"]
        files={}
        for kind,folder,path,digest in [
            ("source","data",row["source_path"],row["source_sha256"]),
            ("crown_body","data",historical.get("mask_path"),historical.get("mask_sha256")),
            ("old_png","baseline",historical.get("corrected_path"),historical.get("corrected_sha256"))]:
            if path:
                path_on_disk=(dirs[folder]/path).resolve()
                found=sha(path_on_disk) if path_on_disk.is_relative_to(dirs[folder]) and path_on_disk.is_file() else None
                files[kind]=dict(folder=folder,path=path,sha256=found,binding_ok=found is not None and found==digest)
        inputs[fid]=files
        if row["input_issue"] or not files.get("source",{}).get("binding_ok"):
            observations[fid]={}
            continue
        im,_=checked_source(dirs["data"],row)
        if list(im.shape[1::-1])!=note["native_size"]:
            raise ValueError("Review grid does not match oriented source: "+fid)
        masks,obs=combine(im,note["candidates"])
        np.savez_compressed(prepared/"masks"/(fid+".npz"),**masks)
        observations[fid]=obs
    frozen=[]
    for row in manifest:
        fid,aid=row["rgb_image_id"],row["anchor_rgb_id"]
        note=annotations[fid]
        obs=observations[fid];target=observations.get(aid,{})
        pairs=paired_blocks(obs,target)
        all_candidates=paired_blocks(obs,target,False)
        ids,held,why=split(pairs)
        mode="ready"
        if row["input_issue"] or not inputs[fid].get("source",{}).get("binding_ok"):
            mode,why="input_error",row["input_issue"] or "input_missing"
        elif not target and fid!=aid and (aid not in inputs or not inputs[aid].get("source",{}).get("binding_ok")):
            mode,why="input_error","reference_input_missing"
        elif fid==aid:
            mode,ids,held,why="reference",[],None,""
        elif (not note["review_complete"] or not annotations.get(aid,{}).get("review_complete")
                or (not pairs and note.get("identity_record_missing"))):
            mode,ids,held,why="input_error",[],None,"identity_unverified"
        elif why:
            mode="unavailable"
            if not pairs and any(r["confirmed"] and k in target and target[k]["confirmed"] for k,r in obs.items()):
                why="insufficient_valid_pixels"
        frozen.append(dict(rgb_image_id=fid,anchor_rgb_id=aid,observed=obs,valid_ids=sorted(pairs),fit=ids,
                holdout=held,eligibility=mode,reason=why,all_span=span(pairs,sorted(pairs)),
                fit_span=span(pairs,ids),diagnostic_candidate_span=span(all_candidates,sorted(all_candidates))))
    json_lines(prepared/"frozen-frames.jsonl",frozen)
    dump(prepared/"external-files.json",inputs)
    names=["pilot_manifest.csv","regions.jsonl","baseline-rows.jsonl","inputs.json","frozen-frames.jsonl","external-files.json"]
    if (prepared/"protocol-source.md").is_file():
        names.append("protocol-source.md")
    names += [p.relative_to(prepared).as_posix() for p in sorted((prepared/"masks").glob("*.npz"))]
    names += [p.relative_to(prepared).as_posix() for p in sorted((prepared/"review").iterdir()) if p.is_file()]
    source_doc=prepared/"protocol-source.md"
    document_hash=sha(source_doc) if source_doc.is_file() else SPEC_HASH
    if document_hash not in (SPEC_HASH,PORTABLE_SPEC_HASH):
        raise ValueError("Unrecognized pilot protocol text")
    lock=dict(protocol="rgb-patch-huber-pilot-1.0",document_sha256=document_hash,
        kernel="qlf_stage3.rgb.color.fit_channels; sklearn HuberRegressor, fit_intercept=True, equal unit weights",
        solver=cfg,min_valid_pixels_each_side=25,code_open_interval=[5,250],linear_channels="RGB",
        percentiles=[10,90],quantile_method="linear",min_six_spans=.02,min_distinct_physical_blocks=2,
        holdout="first sorted physical patch whose removal leaves a valid fit; else fit all if eligible",
        case=dict(id="I0654",native_xyxy=[889,880,1305,1296],stripe_crop_xyxy=[30,172,72,209],adjacent_crop_xyxy=[77,270,122,309]),
        registration="manual native geometry reviewed on every photograph; no whole-tooth or root registration",
        versions={k:importlib.metadata.version(k) for k in ("numpy","scikit-learn","scipy","scikit-image","pandas","Pillow")},
        opencv=cv2.__version__,python=platform.python_version(),source=source_fingerprints(),
        files={name:sha(prepared/name) for name in names})
    dump(prepared/"protocol-lock.json",lock)
    return prepared/"protocol-lock.json"


def check(prepared):
    lock=load(prepared/"protocol-lock.json")
    if lock["document_sha256"] not in (SPEC_HASH,PORTABLE_SPEC_HASH):
        raise ValueError("Wrong protocol revision")
    live=source_fingerprints()
    for name in ("patch_pilot/protocol.py","patch_pilot/prepare.py","patch_pilot/freeze.py","patch_pilot/execution.py","rgb/color.py","rgb/geometry.py","exposure.py"):
        if live.get(name)!=lock["source"].get(name):
            raise ValueError("Frozen calibration implementation changed: "+name)
    checked_files(prepared,lock["files"])
    dirs=roots(prepared)
    meta=load(prepared/"inputs.json")
    for key,(kind,name) in SOURCES.items():
        if sha(dirs[kind]/name)!=meta["source_tables"][key]:
            raise ValueError("Read-only baseline table changed: "+name)
    inventory=load(prepared/"external-files.json")
    counts={key:0 for key in ("source","old_png","crown_body")}
    checked={}
    for fid,files in inventory.items():
        for kind,entry in files.items():
            path=(dirs[entry["folder"]]/entry["path"]).resolve()
            if not path.is_relative_to(dirs[entry["folder"]]):
                raise ValueError("External file outside declared input root")
            if path not in checked:
                checked[path]=sha(path) if path.is_file() else None
            if checked[path]!=entry["sha256"]:
                raise ValueError("External input changed after freeze: "+fid+" / "+kind)
            counts[kind]+=int(checked[path] is not None)
    return lock,dirs,counts

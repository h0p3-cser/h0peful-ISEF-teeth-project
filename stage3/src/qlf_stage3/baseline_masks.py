"""Make original-coordinate labels for background, tooth, and S0 dark regions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
import sys

os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
import numpy as np

from .settings import PROJECT
from .image_io import read_image, write_png
DATA = PROJECT / "Data"
OUTPUT = DATA / "语义掩膜_背景与基线黑斑"
MANUAL = PROJECT / "runs/qlf_paired_analysis_20261123/QLF_五角度损失热图/剔除S0黑斑"
from .registration import CONFIG, assess, canonical_frame, homogeneous, points_transform, register_frames

IMAGE_NAME = re.compile(r"(T\d{3})_(S[012])_(C\d{2})_(A\d{3})_(RGB|QLF)_I(\d+)\.jpg", re.I)
CONFIG.update(canvas=288, pyramid=[72, 144, 288], iterations=55)


def inventory():
    groups = defaultdict(list)
    classes = sorted(p for p in DATA.iterdir() if p.is_dir() and re.match(r"^(00|01|02|03)_", p.name))
    assert len(classes)==4
    for folder in classes:
        for path in sorted(folder.rglob("*.jpg")):
            match = IMAGE_NAME.fullmatch(path.name)
            if match:
                tooth, state, batch, angle, modality, image_number = match.groups()
                rel = path.relative_to(DATA)
                dose = {"00": 17, "01": 0, "03": 72}.get(folder.name[:2])
                if dose is None:
                    dose = int(rel.parts[1][:-1])
                groups[tooth, angle, modality].append(dict(path=rel, tooth=tooth, angle=angle, modality=modality,
                                                           state=state, dose=dose, number=int(image_number)))
    for rows in groups.values():
        rows.sort(key=lambda r:(r["dose"],r["number"]))
        by_dose=defaultdict(list)
        for row in rows:
            by_dose[row["dose"]].append(row)
        for same in by_dose.values():
            assert len(same)==7
            for rank,r in enumerate(same,1):
                r["rank"]=rank
    return dict(sorted(groups.items()))


def read_frame(row, stage):
    path = row["path"]
    mask_path=DATA/"masks"/path.with_suffix(".png")
    if row["state"]=="S2":
        mask_path=stage/f"sam_{row['modality'].lower()}"/Path(*path.parts[1:]).with_suffix(".png")
    rgb=read_image(DATA/path,cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    mask=read_image(mask_path,cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None or rgb.shape[:2]!=mask.shape or not np.any(mask):
        raise ValueError(f"Unreadable image or mask: {path}")
    return rgb,mask


def automatic_spots(image, mask, modality):
    # Estimate seeds from untreated images only. In RGB, brownish chroma is
    # recorded alongside very dark neutral candidates. A local contrast criterion
    # reduces smooth illumination shading in both modes.
    x,y,w,h=cv2.boundingRect(mask)
    view=image[y:y+h,x:x+w]
    m=mask[y:y+h,x:x+w]>0
    scale=min(1,480/max(w,h))
    shape=(max(1,round(w*scale)),max(1,round(h*scale)))
    small=cv2.resize(view,shape,interpolation=cv2.INTER_AREA)
    tooth=cv2.resize(m.astype(np.uint8),shape,interpolation=cv2.INTER_NEAREST)
    gray=(small[...,1] if modality=="QLF" else cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)).astype(np.float32)
    reference=float(np.percentile(gray[tooth>0],80))
    distance=cv2.distanceTransform(np.pad(tooth,1),cv2.DIST_L2,5)[1:-1,1:-1]
    weight=tooth.astype(np.float32)
    sigma=.065*max(shape)
    local=cv2.GaussianBlur(gray*weight,(0,0),sigma)/np.maximum(cv2.GaussianBlur(weight,(0,0),sigma),1e-4)
    if modality=="QLF":
        seed=(gray<.58*reference)&(gray<.78*local)
        support=(gray<.72*reference)&(gray<.91*local)
    else:
        brown=small[...,2].astype(np.float32)-small[...,0]
        seed=((gray<.67*reference)&(brown>15)|(gray<.53*reference))&(gray<.79*local)
        support=((gray<.78*reference)&(brown>9)|(gray<.62*reference))&(gray<.92*local)
    seed=seed&(distance>4)
    support=np.uint8(support&(distance>2))
    support=cv2.morphologyEx(support,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(support,connectivity=8)
    counts=np.bincount(labels[seed],minlength=n)
    keep=(counts>=7)&(stats[:,cv2.CC_STAT_AREA]>=max(20,.0015*tooth.sum()))
    keep[0]=False
    cropped=cv2.resize(keep[labels].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    full=np.zeros(mask.shape,bool)
    full[y:y+h,x:x+w]=cropped&m
    return full


def pack(row,stage):
    rgb,mask=read_frame(row,stage)
    key=row["path"].as_posix()
    frame=canonical_frame(key,rgb,mask)
    frame.rgb=None
    x,y,w,h=cv2.boundingRect(mask)
    points=np.array([[x,y],[x+w-1,y],[x,y+h-1],[x+w-1,y+h-1]])
    return frame,points,mask


def describe(q):
    return {k:float(q[k]) for k in ("dice","gray_ncc","cycle_fraction","reverse_dice","reverse_gray_ncc") if k in q}


def output_mask(row,mask):
    dest=OUTPUT/"标签"/row["path"].with_suffix(".png")
    dest.parent.mkdir(parents=True,exist_ok=True)
    write_png(dest,mask)
    return dest.relative_to(OUTPUT).as_posix()


def register_checked(source,target,points):
    best,quality,attempts=register_frames(source,target,points)
    if quality["candidate_ok"]:
        return best,quality
    plausible={direction:[a for a in attempts if a["direction"]==direction and a["plausible"]
                          and a["dice"]>=CONFIG["min_dice"] and a["gray_ncc"]>=CONFIG["min_gray_ncc"]]
               for direction in ("forward","reverse")}
    canonical_points=points_transform(points,source.canonical)
    eligible=[]
    for fwd in plausible["forward"]:
        fm=homogeneous(fwd["matrix"])
        for rev in plausible["reverse"]:
            returned=points_transform(points_transform(canonical_points,fm),rev["matrix"])
            cycle=float(np.max(np.linalg.norm(returned-canonical_points,axis=1))/(.82*len(source.mask)))
            if cycle<=CONFIG["max_cycle_fraction"]:
                eligible.append((fwd["score"]+rev["score"],-cycle,fwd,rev,fm))
    if not eligible:
        return best,quality
    _,negative_cycle,fwd,rev,mat=max(eligible,key=lambda item:item[:2])
    original=np.linalg.inv(target.canonical)@mat@source.canonical
    return original,dict(candidate_ok=True,alternative_pair=True,dice=fwd["dice"],
                         gray_ncc=fwd["gray_ncc"],reverse_dice=rev["dice"],
                         reverse_gray_ncc=rev["gray_ncc"],cycle_fraction=-negative_cycle)


def sift_points(frame):
    inner=cv2.erode(frame.mask,np.ones((5,5),np.uint8))
    return cv2.SIFT_create(1000,contrastThreshold=.012).detectAndCompute(np.uint8(frame.gray*255),inner)


def landmark_registration(source,target,features,target_features):
    kp0,desc0=features
    kp1,desc1=target_features
    if desc0 is None or desc1 is None or min(len(desc0),len(desc1))<10:
        return None,None
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def reliable(a,b):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2)
                if len(pair)==2 for m,n in [pair] if m.distance<.76*n.distance}
    forward= reliable(desc0,desc1)
    reverse= reliable(desc1,desc0)
    correspondences=[(kp0[i].pt,kp1[j].pt) for i,j in forward.items() if reverse.get(j)==i]
    if len(correspondences)<10:
        return None,None
    a,b=map(np.ascontiguousarray,np.asarray(correspondences,np.float32).transpose(1,0,2))
    matrix,inliers=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5,
                                      maxIters=2500,confidence=.995,refineIters=10)
    if matrix is None or np.linalg.det(matrix[:,:2])<=0:
        return None,None
    inliers=inliers.ravel()>0
    distinct=[]
    for i in np.flatnonzero(inliers):
        if all(np.linalg.norm(a[i]-a[j])>=3 and np.linalg.norm(b[i]-b[j])>=3 for j in distinct):
            distinct.append(i)
    if len(distinct)<10 or np.mean(inliers)<.48:
        return None,None
    span=np.ptp(a[distinct],axis=0)
    _,_,w,h=cv2.boundingRect(source.mask)
    if np.any(span<np.array([w,h])*.24):
        return None,None
    inverse=cv2.invertAffineTransform(matrix)
    residual=max(float(np.quantile(np.linalg.norm(points_transform(a[distinct],matrix)-b[distinct],axis=1),.9)),
                 float(np.quantile(np.linalg.norm(points_transform(b[distinct],inverse)-a[distinct],axis=1),.9)))
    fwd=assess(source,target,matrix)
    back=assess(target,source,inverse)
    if residual>2.6 or not all(q["plausible"] and q["dice"]>=.92 and q["gray_ncc"]>=.30 for q in (fwd,back)):
        return None,None
    original=np.linalg.inv(target.canonical)@homogeneous(matrix)@source.canonical
    return original,dict(dice=fwd["dice"],reverse_dice=back["dice"],gray_ncc=fwd["gray_ncc"],
                         reverse_gray_ncc=back["gray_ncc"],landmark_inliers=len(distinct),
                         reprojection_p90_canvas_px=residual)


def process(rows,scope,stage,manual,reviews,landmark_review):
    bases=[r for r in rows if r["dose"]==0]
    src=next((r for r in bases if r["path"].as_posix() in manual),bases[3])
    sf,pts,smask=pack(src,stage)
    if src["path"].as_posix() in manual:
        dark=read_image(manual[src["path"].as_posix()],cv2.IMREAD_GRAYSCALE)>0
        assert dark.shape==smask.shape and np.all(~dark|(smask>0))
        source="manual_s0_qlf"
    else:
        rgb,_=read_frame(src,stage)
        dark=automatic_spots(rgb,smask,src["modality"])
        source="automatic_s0_candidate"
    targets=[r for r in rows if (r["state"]=="S2")== (scope=="heavy")
             and (reviews is None or r["path"].as_posix() in reviews)]
    results=[]
    successful=defaultdict(list)
    pending=[]

    def finalize(row,mask,trans,method,check,local_dark=None):
        inside=mask>0
        arr=np.uint8(inside)
        if local_dark is not None:
            arr[local_dark]=2
        elif trans is None:
            arr[inside]=3
        else:
            wdark=cv2.warpAffine(dark.astype(np.uint8),trans[:2],mask.shape[::-1],flags=cv2.INTER_NEAREST)>0
            arr[inside&wdark]=2
        rel=output_mask(row,arr)
        count=np.bincount(arr.ravel(),minlength=4)
        results.append(dict(path=row["path"].as_posix(),label_path=rel, tooth=row["tooth"],angle=row["angle"],
            modality=row["modality"],dose=row["dose"],state=row["state"],rank=row["rank"],
            s0_anchor=(row if local_dark is not None else src)["path"].as_posix(),
            source="automatic_same_image_s0" if local_dark is not None else source,status=method,
            background_pixels=int(count[0]),tooth_pixels=int(count[1]),dark_pixels=int(count[2]),
            uncertain_pixels=int(count[3]),
            transform=None if trans is None else trans[:2].tolist(),quality=check))

    if landmark_review:
        source_sift=sift_points(sf)
        for r in targets:
            tf,_,mask=pack(r,stage)
            transform,quality=landmark_registration(sf,tf,source_sift,sift_points(tf))
            if transform is not None:
                finalize(r,mask,transform,"registered_landmarks",quality)
            elif r["state"]=="S0":
                rgb,_=read_frame(r,stage)
                finalize(r,mask,None,"automatic_direct_s0",{},
                         local_dark=automatic_spots(rgb,mask,r["modality"]))
        return results
    for r in targets:
        path=r["path"].as_posix()
        tf,target_pts,mask=pack(r,stage)
        transform,quality=register_checked(sf,tf,pts)
        if quality["candidate_ok"]:
            method="anchor" if path==src["path"].as_posix() else (
                "registered_alternative" if quality.get("alternative_pair") else "registered")
            finalize(r,mask,transform,method,describe(quality))
            quality_key=min(quality["dice"],quality.get("reverse_dice",1))
            successful[r["dose"]].append((quality_key,r,tf,target_pts,transform))
        else:
            pending.append((r,tf,mask,describe(quality)))
    for r,tf,mask,q in pending:
        accepted=False
        for _,bridge,bf,bp,st in sorted(successful[r["dose"]],key=lambda s:-s[0])[:2]:
            bt,quality=register_checked(bf,tf,bp)
            if quality["candidate_ok"]:
                composed=bt@st
                sm=cv2.warpAffine(smask,composed[:2],mask.shape[::-1],flags=cv2.INTER_NEAREST)>0
                tm=mask>0
                dice=float(2*np.count_nonzero(sm&tm)/(np.count_nonzero(sm)+np.count_nonzero(tm)))
                if dice>=.90:
                    finalize(r,mask,composed,"registered_via_repeat",
                             {**describe(quality),"bridge":bridge["path"].as_posix(),"composed_dice":dice})
                    accepted=True
                    break
        if not accepted:
            finalize(r,mask,None,"registration_review_required",q)
    return results


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--stage",type=Path,required=True)
    args.add_argument("--manual-root",type=Path,default=MANUAL)
    args.add_argument("--scope",choices=("legacy","heavy"),required=True)
    args.add_argument("--workers",type=int,default=4)
    args.add_argument("--review-only",action="store_true")
    args.add_argument("--landmark-review",action="store_true")
    opts=args.parse_args()
    cv2.setNumThreads(1)
    source=json.loads((opts.manual_root/"掩膜与指标记录.json").read_text(encoding="utf-8"))["samples"]
    manual={r["s0_image"]:opts.manual_root/"新掩膜"/r["array"]/"S0黑斑排除区.png" for r in source}
    groups=inventory()
    assert len(groups)==90 and sum(map(len,groups.values()))==2240
    index=OUTPUT/f"{opts.scope}_index.jsonl"
    old={}
    reviews=None
    if opts.review_only or opts.landmark_review:
        old={r["path"]:r for r in map(json.loads,index.read_text(encoding="utf-8").splitlines())}
        reviews={key for key,row in old.items() if row["status"]=="registration_review_required"}
        groups={key:rr for key,rr in groups.items()
                if any(r["path"].as_posix() in reviews for r in rr)}
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        for k,recs in enumerate(pool.map(lambda rows:process(rows,opts.scope,opts.stage,manual,reviews,opts.landmark_review),groups.values()),1):
            old.update({r["path"]:r for r in recs})
            if k%10==0: print(opts.scope,k,"views",len(old),"images",flush=True)
    pending=index.with_suffix(".tmp")
    with pending.open("w") as f:
        for key in sorted(old):
            f.write(json.dumps(old[key],ensure_ascii=False)+"\n")
    pending.replace(index)

if __name__=="__main__":
    main()

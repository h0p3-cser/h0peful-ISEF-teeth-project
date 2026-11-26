"""Execute the frozen current-to-S0 mapping; use the archived root pipeline only as comparator."""
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from ..files import dump,load,outside_inputs,sha
from ..materials.study import OLD_OUTPUT,historical_crown,old_png
from ..rgb.color import CHANNELS,LINEAR,FrameIssue,apply_rgb,decode,differences,pixel_hash,save_png
from .figures import crown_case
from .freeze import check,paired_blocks
from .prepare import checked_source
from .protocol import SUCCESS,digest_indices,medians,read_lines,solve,write_csv

METRICS=("de00","abs_dL","ab_distance")
PAIRS=(("raw","new"),("old","new"))


def archive_series(record):
    s=pd.Series({k:(v if v!="" else np.nan) for k,v in record.items()},dtype=object)
    for k in ("working_width","working_height","crown_pixels","crown_new_clip_px","crown_new_clip_pct",
              *("a_"+c for c in CHANNELS),*("b_"+c for c in CHANNELS)):
        if k in s:
            s[k]=pd.to_numeric(s[k],errors="coerce")
    return s


def historical(raw,info,r,g,files,dirs):
    h=archive_series(g["baseline"])
    roi,center,old,clipmap,stats=None,None,None,None,{}
    note=[]
    if raw is None or h.empty:
        return roi,center,old,clipmap,stats,"missing_comparator"
    if files.get("crown_body",{}).get("binding_ok") and g["anchor"]:
        try:
            roi,center=historical_crown(dirs["data"],h,archive_series(g["anchor"]),raw,info)
            if roi is not None and not roi.any():
                roi,center=None,None
        except (OSError,ValueError,KeyError,TypeError) as ex:
            note.append("crown_geometry_unavailable: "+str(ex))
    if roi is None:
        note.append("current_crown_mask_unavailable")
    if r["old_status"] in OLD_OUTPUT:
        if files.get("old_png",{}).get("binding_ok"):
            old=old_png(dirs["baseline"],h)
        else:
            note.append("old_png_unavailable")
        a,b=([h.get(k+c,np.nan) for c in CHANNELS] for k in ("a_","b_"))
        if np.isfinite(a+b).all():
            rebuilt,stats,clipmap=apply_rgb(raw,a,b,roi)
            if old is not None and not np.array_equal(old,rebuilt):
                note.append("old_mapping_mismatch")
            oq=archive_series(g["qc"])
            if roi is not None and pd.notna(oq.get("crown_pixels")):
                stats["matches_baseline_counts"]=bool(stats["crown_pixels"]==oq["crown_pixels"] and
                             stats["crown_new_clip_px"]==oq["crown_new_clip_px"])
    return roi,center,old,clipmap,stats," | ".join(note)


def clip_bits(raw,coeff):
    lut=LINEAR[:,None]*np.array([p[0] for p in coeff])+np.array([p[1] for p in coeff])
    flags=np.zeros(raw.shape[:2],np.uint8)
    for i in range(3):
        flags|=((lut[:,i]<0).astype(np.uint8)<<(2*i))[raw[...,i]]
        flags|=((lut[:,i]>1).astype(np.uint8)<<(2*i+1))[raw[...,i]]
    return flags


def verify_observations(im,obs,masks):
    if set(masks)!=set(obs):
        raise ValueError("Frozen mask archive does not contain its declared physical blocks")
    for pid,value in obs.items():
        idx=masks[pid]
        if (len(idx)!=value["pixels"] or digest_indices(idx)!=value["mask_sha256"]
                or medians(im,idx,True)!=value["linear"] or medians(im,idx)!=value["srgb"]):
            raise ValueError("The saved original pixels do not reproduce the frozen block: "+pid)


def initialize(r,f):
    o=dict(r)
    o.update(status=f["eligibility"],reason=f["reason"],valid_blocks=len(f["valid_ids"]),fit_blocks=len(f["fit"]),
             holdout_id=f["holdout"] or "",fit_ids=json.dumps(f["fit"]),two_point_support=False,
             corrected_path="",corrected_pixel_sha256="",corrected_sha256="",preclip_path="")
    for name,ss in (("all",f["all_span"]),("fit",f["fit_span"]),("candidate_only",f["diagnostic_candidate_span"])):
        for label,v in zip((s+"_"+c for s in ("current","reference") for c in CHANNELS),ss):
            o[name+"_span_"+label]=v
    for c in CHANNELS:
        o["a_"+c]=o["b_"+c]=None
    for method in ("raw","old","new","raw_minus_new","old_minus_new"):
        for m in METRICS:
            o[method+"_"+m]=None
    for k in ("crown_pixels","old_clip_px","old_clip_pct","new_clip_px","new_clip_pct","old_minus_new_clip_pp"):
        o[k]=None
    return o


def measure_holdout(outcome,held_idx,target,raw,old,new):
    for name,im in (("raw",raw),("old",old),("new",new)):
        if im is None:
            continue
        value=np.asarray(medians(im,held_idx))
        outcome[name+"_holdout_srgb"]=json.dumps(value.tolist())
        delta=differences(value[None,:],np.array([target]))[0]
        for m,v in zip(METRICS,delta):
            outcome[name+"_"+m]=float(v)
    for left,right in PAIRS:
        for m in METRICS:
            v=outcome[left+"_"+m]
            if v is not None:
                outcome[left+"_minus_"+right+"_"+m]=v-outcome[right+"_"+m]


def process(prepared,out,lock,dirs):
    frame=pd.read_csv(prepared/"pilot_manifest.csv",dtype=str,keep_default_na=False).to_dict("records")
    frozen={f["rgb_image_id"]:f for f in read_lines(prepared/"frozen-frames.jsonl")}
    geometry={g["rgb_image_id"]:g for g in read_lines(prepared/"baseline-rows.jsonl")}
    files=load(prepared/"external-files.json")
    results=[]
    measures=[]
    for r in frame:
        fid,aid=r["rgb_image_id"],r["anchor_rgb_id"]
        f=frozen[fid]
        target=frozen.get(aid,{}).get("observed",{})
        raw,info=None,None
        masks={}
        if files[fid].get("source",{}).get("binding_ok") and not r["input_issue"]:
            raw,info=checked_source(dirs["data"],r)
            with np.load(prepared/"masks"/(fid+".npz"),allow_pickle=False) as archive:
                masks={p:archive[p] for p in archive.files}
            verify_observations(raw,f["observed"],masks)
        roi,center,old,old_clip,baseline,qnote=historical(raw,info,r,geometry[fid],files[fid],dirs)
        o=initialize(r,f)
        o["qc_note"]=qnote
        o["old_counts_match_archive"]=baseline.get("matches_baseline_counts")
        if roi is not None:
            o["crown_pixels"]=int(roi.sum())
        o["old_clip_px"],o["old_clip_pct"]=baseline.get("crown_new_clip_px"),baseline.get("crown_new_clip_pct")
        coeff=None
        if f["eligibility"]=="reference":
            coeff=[(1.,0.,0,0.)]*3
        elif f["eligibility"]=="ready":
            try:
                coeff=solve(paired_blocks(f["observed"],target),f["fit"],lock["solver"])
                o.update(status="ok_evaluable" if f["holdout"] else "ok_no_holdout",two_point_support=len(f["fit"])==2)
            except FrameIssue as ex:
                o.update(status="unavailable",reason=ex.code,fit_error=str(ex))
        saved,new_clip=None,None
        if coeff is not None:
            for c,(a,b,it,absres) in zip(CHANNELS,coeff):
                o.update({"a_"+c:a,"b_"+c:b,"iterations_"+c:it,"median_abs_residual_"+c:absres})
            img,stats,new_clip=apply_rgb(raw,[p[0] for p in coeff],[p[1] for p in coeff],roi)
            file=out/"corrected_rgb"/(fid+".png")
            save_png(file,img)
            saved,_=decode(file,corrected=True)
            if not np.array_equal(saved,img) or (o["status"]=="reference" and not np.array_equal(raw,saved)):
                raise ValueError("PNG round-trip mismatch: "+fid)
            flagfile=out/"preclip"/(fid+".png")
            flags=clip_bits(raw,coeff)
            if int(np.count_nonzero(flags))!=stats["whole_outside_px"]:
                raise ValueError("Native over-range flag count mismatch")
            Image.fromarray(flags).save(flagfile,compress_level=3)
            o.update(corrected_path=file.relative_to(out).as_posix(),corrected_sha256=sha(file),corrected_pixel_sha256=pixel_hash(saved),
                 preclip_path=flagfile.relative_to(out).as_posix(),new_clip_px=stats.get("crown_new_clip_px"),
                 new_clip_pct=stats.get("crown_new_clip_pct"))
            if o["new_clip_pct"] is not None and o["old_clip_pct"] is not None:
                o["old_minus_new_clip_pp"]=o["old_clip_pct"]-o["new_clip_pct"]
            for name,v in stats.items():
                if not name.startswith("crown_"):
                    o[name]=v
            if f["holdout"]:
                pid=f["holdout"]
                measure_holdout(o,masks[pid],target[pid]["srgb"],raw,old,saved)
        if fid=="I0654":
            crown_case(out,raw,old,saved,roi,center,old_clip,new_clip,o["reason"])
        results.append(o)
        for pid in sorted(set(f["observed"])|set(target)):
            role="paired_unused" if pid in f["valid_ids"] else "excluded_or_missing"
            if pid in f["fit"]:
                role="fit"
            if pid==f["holdout"]:
                role="holdout"
            if f["eligibility"]=="reference":
                role="reference_only"
            b=dict(rgb_image_id=fid,anchor_rgb_id=aid,patch_id=pid,role=role,
                   unit_weight=int(role=="fit"),actually_used=role=="fit" and o["status"] in SUCCESS)
            for side,obs,key in (("current",f["observed"],fid),("reference",target,aid)):
                v=obs.get(pid,{})
                b.update({side+"_mask_archive":("masks/"+key+".npz") if v else None,
                    side+"_mask_key":pid if v else None,side+"_mask_sha256":v.get("mask_sha256"),
                    side+"_visible":v.get("visible"),side+"_confirmed":v.get("confirmed"),side+"_pixels":v.get("pixels")})
                for space in ("linear","srgb"):
                    for c,z in zip(CHANNELS,v.get(space) or [None]*3):
                        b[side+"_"+space+"_"+c]=z
            if role=="holdout":
                for method in ("raw","old","new"):
                    b[method+"_measured_srgb"]=o.get(method+"_holdout_srgb")
                    for m in METRICS:
                        b[method+"_"+m]=o[method+"_"+m]
            measures.append(b)
    df=write_csv(out/"frame_results.csv",results)
    write_csv(out/"patch_measurements.csv",measures)
    return df


def run(prepared,out):
    from .report import report
    prepared=Path(prepared).resolve()
    lock,dirs,_=check(prepared)
    out=outside_inputs(out,*dirs.values())
    out.mkdir(parents=True,exist_ok=False)
    for name in ("figures","corrected_rgb","preclip"):
        (out/name).mkdir()
    dump(out/"status.json",dict(status="running",protocol_lock_sha256=sha(prepared/"protocol-lock.json")))
    try:
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            df=process(prepared,out,lock,dirs)
        report(prepared,out,df)
        dump(out/"status.json",dict(status="complete",protocol_lock_sha256=sha(prepared/"protocol-lock.json"),
                 files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="status.json"}))
    except Exception as ex:
        dump(out/"status.json",dict(status="failed",error=str(ex)))
        raise
    return out/"report.md"

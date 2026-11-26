"""Run and replay one reviewed full-field material Huber pilot on the locked photographs."""
import importlib.metadata
import json
import platform
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from ..files import PACKAGE, dump, load, outside_inputs, sha
from ..rgb.color import CHANNELS, FrameIssue, apply_rgb, decode, differences, pixel_hash, save_png
from ..rgb.geometry import polygon_mask, resize_matrix, transform, white_body
from ..rgb.report import tiles_for
from .figures import case_figure, field_overlay, reference_detail
from .patches import paired
from .root_holdout import measure, root_grid
from .scenes import edges, fit_pairs, locate, observations
from .study import OLD_FIT, SLOTS, historical_crown, load_pilot, old_clipping, old_png, read_source

PROTOCOL=PACKAGE/"config/rgb_material_huber.json"
REGIONS=PACKAGE/"config/rgb_material_regions.json"
KEYS=("rgb_image_id","anchor_rgb_id","tooth_id","angle_id","state_id","repeat_rank","rgb_event","group_id")


def table(out,name,rows):
    frame=pd.DataFrame(rows)
    frame.to_csv(out/name,index=False,encoding="utf-8-sig",lineterminator="\n",float_format="%.12g")
    return frame


def material_scores(pairs,old,saved):
    rows=[]
    for p,q in pairs:
        if p.role!="holdout":
            continue
        r=dict(material=p.material,region=p.region,scope="external_material",valid_pixels=len(p.xx),
               target_rgb=json.dumps(q.srgb.tolist()))
        for name,im in (("raw",None),("root",old),("material",saved)):
            if im is None and name!="raw":
                continue
            v=p.srgb if name=="raw" else p.color(im)
            d,l,ab=differences(v[None,:],q.srgb[None,:])[0]
            r.update({name+"_rgb":json.dumps(v.tolist()),name+"_de00":d,
                name+"_abs_dL":l,name+"_ab_distance":ab})
        rows.append(r)
    return rows


def patch_log(current,anchor,pairs):
    accepted={p.region for p,q in pairs}
    fit=Counter(p.material for p,q in pairs if p.role=="fit")
    for key in sorted(set(current)|set(anchor)):
        p,q=current.get(key),anchor.get(key)
        item=p if p is not None else q
        reason=("current_material_not_visible" if p is None else "material_not_in_S0" if q is None
                else p.reason or q.reason)
        r=dict(material=item.material,region=key,role=item.role,paired=key in accepted,reason=reason,
               fit_weight=0.)
        if key in accepted and p.role=="fit":
            r["fit_weight"]=sum(fit.values())/(len(fit)*fit[p.material])
        for tag,patch in (("current",p),("anchor",q)):
            if patch is not None:
                r.update({tag+"_polygon":json.dumps(patch.polygon,separators=(",",":")),
                          tag+"_pixels":len(patch.xx),tag+"_total":patch.n_total,tag+"_valid_fraction":patch.valid_fraction})
                for i,c in enumerate(CHANNELS):
                    r[tag+"_sRGB_"+c]=patch.srgb[i]
                    r[tag+"_linear_"+c]=patch.linear[i]
        yield r


def coverage(row):
    if row.rgb_image_id==row.anchor_rgb_id:
        return "anchor"
    old=row.old_status in OLD_FIT
    new=row.status=="ok"
    return { (True,True):"retained",(False,True):"recovered",(True,False):"lost",
             (False,False):"failed_both"}[old,new]


def versions():
    libs=["numpy","pandas","scikit-learn","scipy","scikit-image","Pillow"]
    return dict({lib:importlib.metadata.version(lib) for lib in libs},opencv=cv2.__version__,python=platform.python_version())


def input_data(study,review,rules):
    if set(tuple(pair) for pair in rules["scope"])!=set(SLOTS) or set(review["events"])!=set(study.frames.rgb_event):
        raise ValueError("The annotation or scope does not match the locked pilot")
    for event,group in study.frames.groupby("rgb_event"):
        info=review["events"][event]
        basis=group[group.rgb_image_id.eq(info["representative"])]
        if len(basis)!=1 or basis.iloc[0].rgb_sha256!=info["sha256"]:
            raise ValueError("The scene drawing is not bound to its source: "+event)
    if (len(study.frames)!=63 or not study.frames.groupby(["tooth_id","angle_id","state_id"]).size().eq(7).all()
            or not study.frames.groupby(["tooth_id","angle_id"]).anchor_rgb_id.nunique().eq(1).all()):
        raise ValueError("Incomplete locked burst cohort")


def process(data,study,root_run,out,cfg,review):
    source=study.frames.set_index("rgb_image_id",drop=False)
    old_cfg=load(root_run/"config.json")["protocol"]
    root_tiles=tiles_for(root_run,source.index)
    reference={}
    scene_edges={}
    for event,anno in review["events"].items():
        row=source.loc[anno["representative"]]
        rgb,info=read_source(data,row)
        scene_edges[event]=edges(rgb,cfg["burst_edge_size"])
        if row.name==row.anchor_rgb_id:
            anchor=study.anchors.loc[row.name]
            ref_grid=root_grid(data,row,rgb,info,old_cfg)
            ref_regions=observations(rgb,np.zeros(2),anno,review,cfg)
            reference[row.name]=(ref_regions,ref_grid,np.asarray(json.loads(anchor.root_work)),
                                 np.asarray(json.loads(anchor.crown_work)))
    if set(reference)!=set(study.anchors.index):
        raise ValueError("A reviewed S0 identity is missing")
    frames,blocks,scores=[],[],[]
    overlays={}
    bursts={tooth:Image.new("RGB",(3*336,7*212),"white") for tooth,_ in SLOTS}
    for _,r in study.frames.iterrows():
        key={name:r[name] for name in KEYS}
        raw,info=read_source(data,r)
        if info["grayscale_content"]:
            raise ValueError("Unexpected grayscale pilot source "+r.rgb_image_id)
        is_ref=r.rgb_image_id==r.anchor_rgb_id
        old=old_png(root_run,r)
        event=review["events"][r.rgb_event]
        anchor=study.anchors.loc[r.anchor_rgb_id]
        targets,ref_grid,ref_root,ref_crown=reference[r.anchor_rgb_id]
        qc=study.old_qc.loc[r.rgb_image_id]
        if sha(data/r.mask_path)!=r.mask_sha256:
            raise ValueError("White-light body mask changed for "+r.rgb_image_id)
        crown,center=historical_crown(data,r,anchor,raw,info)
        crown_source="historical_native_roi" if crown is not None else "unavailable"
        historical=old_clipping(r,qc,raw,crown,old)
        photo=dict(**key,sample_id=r.sample_id,qlf_image_id=r.qlf_image_id,qlf_event=r.qlf_event,
                   source_path=r.source_path,source_sha256=r.rgb_sha256,source_pixel_sha256=r.rgb_pixel_sha256,
                   anchor_sha256=r.anchor_sha256,old_status=r.status,old_reason=r.reason,
                   old_corrected_path=r.corrected_path,old_corrected_sha256=r.corrected_sha256,
                   status="failed",reason="",detail="",corrected_path="",corrected_pixel_sha256="",
                   corrected_sha256="",n_fit_blocks=0,n_fit_materials=0,n_holdout_blocks=0,crown_roi=crown_source)
        if historical is not None:
            for name in ("crown_pixels","crown_new_clip_px","crown_new_clip_pct"):
                photo["old_"+name]=historical[0][name]
        current,matched={},[]
        params=None
        try:
            shift,response=locate(raw,scene_edges[r.rgb_event],r.rgb_image_id==event["representative"],cfg)
            photo.update(dx_native=shift[0],dy_native=shift[1],burst_response=response)
            current=observations(raw,shift,event,review,cfg)
            matched=paired(current,targets)
            if is_ref:
                params=[(1.,0.,0,0.)]*3
                photo.update(n_holdout_blocks=sum(p.role=="holdout" for p,q in matched))
            else:
                params=fit_pairs(matched,cfg,photo)
            # Independent native QC for an S0 repeat even when the historical geometry failed.
            if crown is None and r.state_id=="S0" and r.anchor_rgb_id==event["representative"]:
                basis=resize_matrix(raw.shape[1],raw.shape[0],ref_grid.body.shape[::-1])
                outline=transform(ref_crown,np.linalg.inv(basis))+shift
                crown=polygon_mask(raw.shape[:2],outline)&white_body(data/r.mask_path,info,raw)[0]
                if crown.any():
                    center=outline.mean(axis=0)
                    photo["crown_roi"]="same_S0_burst_native_roi"
                else:
                    crown=None
        except FrameIssue as exc:
            photo.update(reason=exc.code,detail=str(exc))
        saved,new_clip=None,None
        if params is not None:
            for c,(a,b,it,res) in zip(CHANNELS,params):
                photo.update({"a_"+c:a,"b_"+c:b,"iterations_"+c:it,"median_abs_residual_"+c:res})
            corrected,stats,new_clip=apply_rgb(raw,[p[0] for p in params],[p[1] for p in params],crown)
            destination=out/"corrected_rgb"/(r.rgb_image_id+".png")
            save_png(destination,corrected)
            saved,_=decode(destination,corrected=True)
            if not np.array_equal(saved,corrected) or (is_ref and not np.array_equal(saved,raw)):
                raise ValueError("Saved PNG or S0 identity mismatch")
            photo.update(stats,status="reference" if is_ref else "ok",
                         corrected_path=destination.relative_to(out).as_posix(),corrected_pixel_sha256=pixel_hash(saved),
                         corrected_sha256=sha(destination))
        for entry in patch_log(current,targets,matched):
            entry["used_for_fit"]=params is not None and not is_ref and entry["role"]=="fit" and entry["paired"]
            if not entry["used_for_fit"]:
                entry["fit_weight"]=0.
            if not current and photo["reason"]:
                entry["reason"]=photo["reason"]
            blocks.append(dict(**key,**entry))
        eval_rows=material_scores(matched,old,saved)
        local=root_tiles[root_tiles.rgb_image_id.eq(r.rgb_image_id)]
        if not local.empty and isinstance(r.reference_to_current_work,str):
            cur=root_grid(data,r,raw,info,old_cfg)
            matrix=np.asarray(json.loads(r.reference_to_current_work))
            eval_rows.extend(measure(local,ref_grid,ref_root,ref_crown,cur,matrix,old,saved))
        for entry in eval_rows:
            scores.append(dict(**key,**entry))
        overlay=field_overlay(raw,current,r.rgb_image_id,r.tooth_id+" / "+r.state_id+" / "+r.angle_id+" / "+photo["status"])
        col=("S0","S1","S2").index(r.state_id)
        bursts[r.tooth_id].paste(overlay.resize((336,212),Image.Resampling.LANCZOS),(col*336,(int(r.repeat_rank)-1)*212))
        if r.rgb_image_id==event["representative"]:
            overlays[r.rgb_event]=overlay
        if r.rgb_image_id=="I0654":
            if historical is None or center is None:
                raise ValueError("The prespecified case requires a replayed historical crown ROI")
            case_figure(out,raw,old,saved,crown,center,historical[1],new_clip)
            ref_rgb,_=read_source(data,source.loc[r.anchor_rgb_id])
            reference_detail(out,[(r.anchor_rgb_id,ref_rgb,targets),(r.rgb_image_id,raw,current)])
        frames.append(photo)
    gallery=Image.new("RGB",(3*672,3*425),"white")
    for i,(tooth,angle) in enumerate(SLOTS):
        bursts[tooth].save(out/"figures"/(tooth+"-all-bursts.png"))
        for j,state in enumerate(("S0","S1","S2")):
            match=source[source.tooth_id.eq(tooth)&source.angle_id.eq(angle)&source.state_id.eq(state)]
            gallery.paste(overlays[match.rgb_event.iloc[0]],(j*672,i*425))
    gallery.save(out/"figures/material-review.png")
    frame=pd.DataFrame(frames)
    frame["transition"]=[coverage(r) for r in frame.itertuples()]
    table(out,"frame_audit.csv",frame)
    table(out,"material_blocks.csv",blocks)
    return frame,table(out,"heldout_blocks.csv",scores)


def run(data,study_run,root_run,out,protocol=PROTOCOL,regions=REGIONS):
    from .report import report
    data,study_run,root_run=(Path(v).resolve() for v in (data,study_run,root_run))
    out=outside_inputs(out,data,study_run,root_run)
    study=load_pilot(data,study_run,root_run)
    cfg,review=load(protocol),load(regions)
    input_data(study,review,cfg)
    out.mkdir(parents=True,exist_ok=False)
    for name in ("figures","corrected_rgb"):
        (out/name).mkdir()
    dump(out/"status.json",{"status":"running"})
    try:
        shutil.copyfile(protocol,out/"protocol.json")
        shutil.copyfile(regions,out/"regions.json")
        code=sorted((PACKAGE/"materials").glob("*.py"))+[PACKAGE/"rgb"/(n+".py") for n in ("geometry","color","report")]+[PACKAGE/"exposure.py",PACKAGE/"files.py"]
        dump(out/"provenance.json",dict(inputs=dict(study.inputs,**{"root_tiles.csv":sha(root_run/"root_tiles.csv")}),
             annotation_sha256=sha(regions),protocol_sha256=sha(protocol),versions=versions(),
             source_files={f.relative_to(PACKAGE).as_posix():sha(f) for f in code}))
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            frames,scores=process(data,study,root_run,out,cfg,review)
            report(out,frames,scores)
        inventory={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*"))
                   if p.is_file() and p.name!="status.json"}
        dump(out/"status.json",dict(status="complete",frame_count=len(frames),sha256=inventory))
    except Exception as exc:
        dump(out/"status.json",dict(status="failed",error=str(exc)))
        raise
    return out/"report.md"

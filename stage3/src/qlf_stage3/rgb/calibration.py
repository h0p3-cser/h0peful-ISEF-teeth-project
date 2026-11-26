"""Label-blind calibration of individual photos; the runner joins stage labels only for audit."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..files import PACKAGE, dataset_info, dump, load, outside_inputs, relative_path, sha
from .color import (CHANNELS, CORRECTED_MODE, RAW_MODE, FrameIssue, apply_rgb, decode, differences,
                    fit_channels, pixel_hash, save_png)
from .geometry import (align, grid, landmarks, native_crown, polygon_mask, polygons, white_body,
                       sample_tiles, warp_to_reference)
from .study import Photo, pilot_slots, read_study

DEFAULT_CONFIG=PACKAGE/"config/rgb_root_huber.json"
DEFAULT_REGIONS=PACKAGE/"config/rgb_root_regions.json"
METRICS=("de00","lightness","chroma")


def write_table(path,records):
    pd.DataFrame(records).to_csv(path,index=False,encoding="utf-8-sig",lineterminator="\n")


def source(photo):
    if not photo.path.is_file():
        raise FrameIssue("source_missing")
    digest=sha(photo.path)
    rgb,info=decode(photo.path)
    info.update(source_sha256=digest,source_pixel_sha256=pixel_hash(rgb))
    if digest!=photo.digest:
        raise FrameIssue("source_hash_mismatch",info["source_pixel_sha256"])
    if not photo.body.is_file():
        raise FrameIssue("root_not_visible","Native RGB body mask is missing")
    body,info["white_body_method"]=white_body(photo.body,info,rgb)
    return rgb,body,info


def visible_for_fallback(photo,regions):
    if photo.image_id not in regions or not photo.path.is_file() or not photo.body.is_file():
        return False
    rgb,body,_=source(photo)
    small=cv2.resize(rgb,(336,190),interpolation=cv2.INTER_AREA)
    mask=polygon_mask((190,336),regions[photo.image_id]["root"]) & cv2.resize(
        body.astype(np.uint8),(336,190),interpolation=cv2.INTER_NEAREST).astype(bool)
    return bool(mask.sum()>=60 and cv2.Laplacian(cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),cv2.CV_32F)[mask].var()>4)


@dataclass
class Reference:
    photo: Photo
    grid: object
    root: np.ndarray
    crown: np.ndarray
    descriptors: tuple
    mask_sha: str
    pixel_sha: str
    body_method: str


def prepare_reference(photo,regions,grid_size,cfg):
    if photo is None:
        raise FrameIssue("no_anchor")
    if photo.image_id not in regions:
        raise FrameIssue("root_not_visible","The white-light anchor has no reviewed root region")
    annotation=regions[photo.image_id]
    if annotation["sha256"]!=photo.digest:
        raise FrameIssue("reference_annotation_mismatch")
    raw,body,info=source(photo)
    ref=grid(raw,body,info,cfg)
    root,crown=polygons(annotation,grid_size,ref)
    envelope=cv2.convexHull(np.vstack([root,crown]).astype(np.float32)).reshape(-1,2)
    anatomical=polygon_mask(ref.body.shape,envelope)
    if not (polygon_mask(ref.body.shape,root)&ref.body).any():
        raise FrameIssue("root_not_visible")
    return Reference(photo,ref,root,crown,landmarks(ref,cfg,anatomical),sha(photo.body),info["source_pixel_sha256"],info["white_body_method"])


def aligned_medians(samples,pixels,matrix,shape,linear=False):
    return samples.medians(warp_to_reference(pixels,matrix,shape,cv2.INTER_LINEAR),linear)


def digest_key(parts):
    return hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def calibrate_photo(photo,reference,cfg,out,protocol_key):
    """No labels, batch, group, stage, QLF values or other target frames enter this function."""
    record=dict(rgb_image_id=photo.image_id,source_path=photo.relative,expected_source_sha256=photo.digest,
                raw_mode=RAW_MODE,corrected_mode=CORRECTED_MODE,status="failed",reason="",corrected_path="",
                mask_path=photo.mask_relative)
    qc=dict(rgb_image_id=photo.image_id)
    tiles=None
    try:
        rgb,body,info=source(photo)
        record.update(info)
        record["raw_cache_key"]=digest_key([RAW_MODE,photo.digest,info["icc_sha256"],info["orientation"],protocol_key])
        record["mask_sha256"]=sha(photo.body)
        cur=grid(rgb,body,info,cfg)
        if isinstance(reference,FrameIssue):
            raise reference
        ref=reference.grid
        is_ref=photo.image_id==reference.photo.image_id
        record.update(anchor_rgb_id=reference.photo.image_id,anchor_sha256=reference.photo.digest,
                      root_annotation=cfg["annotation_version"],working_width=cur.body.shape[1],
                      working_height=cur.body.shape[0],tile_side=ref.tile)
        archived=photo.archive.get("transform") if photo.archive.get("s0_anchor")==reference.photo.key else None
        record["archive_anchor_matches"]=photo.archive.get("s0_anchor")==reference.photo.key
        record["archive_status"]=photo.archive.get("status","missing")
        if is_ref:
            matrix,points,rejected,geometry=np.eye(3),None,None,dict(method="identity",root_overlap=1.,check_points="[]")
        else:
            matrix,points,rejected,geometry=align(ref,cur,(reference.root,reference.crown),reference.descriptors,archived,cfg)
        record.update(geometry)
        record["reference_to_current_work"]=json.dumps(matrix.tolist(),separators=(",",":"))
        record["geometry_cache_key"]=digest_key([protocol_key,photo.digest,reference.photo.image_id,
                     reference.photo.digest,reference.mask_sha,record["mask_sha256"],photo.archive,
                     record["reference_to_current_work"],record["color_interpretation"],CORRECTED_MODE])
        samples=sample_tiles(ref,cur,matrix,reference.root,reference.crown,points,cfg,rejected)
        table=samples.table.copy()
        table.insert(0,"anchor_rgb_id",reference.photo.image_id)
        table.insert(0,"rgb_image_id",photo.image_id)
        fit=table.role.eq("fit").to_numpy()
        val=~fit
        record.update(split_axis=samples.split["axis"],split_cut2=samples.split["cut2"],split_cut3=samples.split["cut3"],
                      split_guard_px=samples.split["guard"],fit_tiles=int(fit.sum()),holdout_tiles=int(val.sum()))
        shape=ref.body.shape
        aligned=warp_to_reference(cur.rgb,matrix,shape,cv2.INTER_LINEAR)
        x,y=samples.medians(aligned,True),samples.medians(ref.rgb,True)
        before,target=samples.medians(aligned),samples.medians(ref.rgb)
        for i,c in enumerate(CHANNELS):
            for name,v in [("current_linear",x),("reference_linear",y),("raw_srgb",before),("reference_srgb",target)]:
                table[name+"_"+c]=v[:,i]
            if fit.any():
                for name,v in [("current",x),("reference",y)]:
                    record[name+"_span_"+c]=float(np.diff(np.percentile(v[fit,i],[10,90]))[0])
        tiles=table
        params=[(1.,0.,0,0.)]*3 if is_ref else fit_channels(x[fit],y[fit],cfg)
        for c,(a,b,iterations,residual) in zip(CHANNELS,params):
            record.update({f"a_{c}":a,f"b_{c}":b,f"iterations_{c}":iterations,f"median_abs_residual_{c}":residual})
        crown=native_crown(reference.crown,cur,body,matrix)
        corrected,clip,_=apply_rgb(rgb,[p[0] for p in params],[p[1] for p in params],crown)
        qc.update(clip,crown_status="white_light_roi" if crown is not None else "unavailable")
        destination=out/"corrected_rgb"/(photo.image_id+".png")
        save_png(destination,corrected)
        saved,saved_info=decode(destination,corrected=True)
        if not np.array_equal(corrected,saved) or (is_ref and not np.array_equal(saved,rgb)):
            raise RuntimeError("PNG pixels differ from computed correction")
        record.update(corrected_path=destination.relative_to(out).as_posix(),corrected_pixel_sha256=pixel_hash(saved),
                      corrected_sha256=sha(destination),fit_converged=not is_ref)
        small=cv2.resize(saved,(cur.body.shape[1],cur.body.shape[0]),interpolation=cv2.INTER_AREA)
        after=aligned_medians(samples,small,matrix,shape)
        for i,c in enumerate(CHANNELS):
            table["corrected_srgb_"+c]=after[:,i]
        if val.any():
            for prefix,colors in [("before",before),("after",after)]:
                deltas=differences(colors[val],target[val])
                for i,name in enumerate(METRICS):
                    field=prefix+"_"+name
                    table.loc[val,field]=deltas[:,i]
                    if val.sum()>=cfg["min_holdout_tiles"]:
                        qc[field]=float(np.median(deltas[:,i]))
        if val.sum()>=cfg["min_holdout_tiles"]:
            qc.update({name+"_gain":qc["before_"+name]-qc["after_"+name] for name in METRICS})
        record["status"]="reference" if is_ref else ("ok" if val.sum()>=cfg["min_holdout_tiles"] else "ok_no_holdout")
        if val.sum()<cfg["min_holdout_tiles"]:
            record["reason"]="insufficient_holdout_tiles"
    except FrameIssue as exc:
        record.update(reason=exc.code,detail=str(exc))
    qc.update(status=record["status"],reason=record["reason"])
    return record,qc,tiles


def run(data,study_run,out,scope,config_file=DEFAULT_CONFIG,regions_file=DEFAULT_REGIONS,workers=1):
    data,_=dataset_info(data)
    study_run=Path(study_run).resolve()
    out=outside_inputs(out,data,study_run)
    if workers<1:
        raise ValueError("workers must be positive")
    cfg,annotations=load(config_file),load(regions_file)
    regions=annotations["views"]
    groups,photos,anchors,input_hashes=read_study(data,study_run,regions)
    pilot=pilot_slots(groups,cfg)
    views=sorted(set(groups)&pilot if scope=="pilot" else groups)
    cfg.update(annotation_version=annotations["version"])
    code={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name("color.py"),
          Path(__file__).with_name("geometry.py"),Path(__file__).with_name("study.py"),PACKAGE/"exposure.py")}
    versions={k:importlib.metadata.version(k) for k in ("numpy","Pillow","scikit-learn",
             "scikit-image","pandas","scipy")}
    versions["opencv"]=cv2.__version__
    key=digest_key([cfg,sha(regions_file),input_hashes,code,versions])
    out.mkdir(parents=True,exist_ok=False)
    (out/"corrected_rgb").mkdir()
    dump(out/"status.json",dict(status="running"))
    document=dict(protocol=cfg,scope=scope,dataset=relative_path(data,out),study_run=relative_path(study_run,out),
        input_hashes=input_hashes,annotation_sha256=sha(regions_file),source_sha256=code,versions=versions,
        python=platform.python_version(),pilot_views=sorted(pilot),cache_protocol=key,
        color_space="linear sRGB for fitting, encoded sRGB to Lab D65/2°; CIEDE2000 weights 1/1/1")
    dump(out/"config.json",document)
    dump(out/"regions.json",annotations)
    cv2.setNumThreads(1)
    calibrations,metrics,refs=[],[],[]
    tile_cols=["rgb_image_id","anchor_rgb_id","x","y","side","role","valid_pixels"]
    for c in CHANNELS:
        tile_cols.extend([n+"_"+c for n in ("current_linear","reference_linear","raw_srgb","reference_srgb","corrected_srgb")])
    tile_cols.extend(p+"_"+n for p in ("before","after") for n in METRICS)
    with (out/"root_tiles.csv").open("w",encoding="utf-8-sig",newline="") as stream, \
         ThreadPoolExecutor(max_workers=workers) as pool,threadpool_limits(limits=1,user_api="blas"):
        pd.DataFrame(columns=tile_cols).to_csv(stream,index=False,lineterminator="\n")
        for number,slot in enumerate(views,1):
            photo,reason=anchors[slot]
            row=dict(tooth_id=slot[0],angle_id=slot[1],selection=reason,anchor_rgb_id=photo.image_id if photo else "",
                     anchor_sha256=photo.digest if photo else "",source_path=photo.relative if photo else "",
                     included_in_manifest=bool(photo and any(r["rgb_image_id"]==photo.image_id for r in groups[slot])))
            try:
                ref=prepare_reference(photo,regions,annotations["drawing_grid"],cfg)
                row.update(status="ready",mask_sha256=ref.mask_sha,white_body_method=ref.body_method,anchor_pixel_sha256=ref.pixel_sha,anchor_rgb_event=photo.event,
                           root_work=json.dumps(ref.root.tolist()),
                           crown_work=json.dumps(ref.crown.tolist()),work_width=ref.grid.body.shape[1],
                           work_height=ref.grid.body.shape[0],tile_side=ref.grid.tile)
            except FrameIssue as exc:
                ref=exc
                row["status"]=exc.code
            refs.append(row)
            def process(member):
                return calibrate_photo(photos[member["rgb_image_id"]],ref,cfg,out,key)
            for member,(rec,qc,tiles) in zip(groups[slot],pool.map(process,groups[slot]),strict=True):
                meta={k:member[k] for k in ("sample_id","tooth_id","angle_id","state_id","session_id","group_id",
                      "rgb_event","qlf_event","qlf_image_id","repeat_rank","author_rgb_hash_status","test_fold","geometry_version")}
                meta.update(pilot_tooth=slot[0] in cfg["pilot_teeth"],pilot_view=slot in pilot,
                            anchor_rgb_id=row["anchor_rgb_id"],anchor_sha256=row["anchor_sha256"])
                if "source_pixel_sha256" in rec and rec["source_pixel_sha256"]!=member["rgb_pixel_sha256"]:
                    if rec.get("color_interpretation")!="icc_to_srgb":
                        raise ValueError("Decoded pixels differ from the locked white-light source: "+member["sample_id"])
                meta["legacy_rgb_pixel_sha256"]=member["rgb_pixel_sha256"]
                calibrations.append(dict(meta,**{k:v for k,v in rec.items() if k not in meta}))
                metrics.append(dict(meta,**qc))
                if tiles is not None:
                    tiles.reindex(columns=tile_cols).to_csv(stream,index=False,header=False,lineterminator="\n")
            print(f"{number}/{len(views)} {slot[0]} {slot[1]}: {len(groups[slot])} frames",flush=True)
    write_table(out/"anchors.csv",refs)
    write_table(out/"rgb_calibration.csv",calibrations)
    write_table(out/"qc_by_frame.csv",metrics)
    if len(calibrations)!=sum(len(groups[s]) for s in views) or len(set(r["sample_id"] for r in calibrations))!=len(calibrations):
        raise RuntimeError("Cohort frame accounting does not close")
    from .report import report
    report(out,data,study_run)
    dump(out/"status.json",dict(status="complete",frames=len(calibrations),scope=scope))
    return out

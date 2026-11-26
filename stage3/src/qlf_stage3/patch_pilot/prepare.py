"""Build the full paired cohort and a native-coordinate, auditable review worksheet."""
from pathlib import Path
import re
import shutil

import pandas as pd

from ..files import dump, load, outside_inputs, relative_path, sha
from ..materials.study import SLOTS
from ..rgb.color import decode, pixel_hash
from .protocol import PORTABLE_SPEC_HASH, SCENES, SPEC_HASH, SPEC_PATH, json_lines, write_csv

SOURCES={"samples":("study","evaluation/sample_manifest.csv"),"baseline":("baseline","rgb_calibration.csv"),
         "anchors":("baseline","anchors.csv"),"qc":("baseline","qc_by_frame.csv"),"config":("baseline","config.json")}


def roots(prepared):
    return {k:(prepared/v).resolve() for k,v in load(prepared/"inputs.json")["roots"].items()}


def checked_source(data,row):
    if not row["source_path"]:
        raise FileNotFoundError("No recorded source")
    p=(data/row["source_path"]).resolve()
    if not p.is_relative_to(data) or not p.is_file():
        raise FileNotFoundError(row["source_path"])
    if sha(p)!=row["source_sha256"]:
        raise ValueError("Source SHA-256 disagrees with paired identity")
    im,info=decode(p)
    if row["source_pixel_sha256"] and pixel_hash(im)!=row["source_pixel_sha256"]:
        raise ValueError("Decoded source identity disagrees with the paired manifest")
    return im,info


def sheet(r,scene):
    event=scene["events"].get(r["rgb_event"],{})
    reviewed=event.get("frames",{}).get(r["rgb_image_id"])==r["source_sha256"] and bool(r["source_sha256"])
    c04=event.get("hardware_record")=="changed_setup_unmatched"
    same_event=r["rgb_event"]==r["anchor_event"]
    blocks=[]
    for key,evidence in scene["identities"].items():
        polygons=event.get("patches",{}).get(key,[])
        visibility="present" if polygons else "out_of_field"
        admitted=bool(polygons) and reviewed
        reason=""
        if not polygons:
            reason="该实物表面在本张画面中不可辨认。"
        elif not reviewed:
            reason="帧尚未人工核对。"
        elif c04:
            admitted=False
            reason="C04 装置与环境有差异，无可把这件实物逐件对应回 S0 的固定标记或器材记录。"
        elif not same_event and key=="outer_bed":
            admitted=False
            reason="可见外侧黑色平面，但跨采集没有可识别的专属标记，无法核实是同一片底板。"
        else:
            reason=("同一采集连拍，边界及构图逐帧复核。" if same_event else
                    "跨采集以带舌圆弧和相邻短槽复认盘面，旋转后分别定位。")
        blocks.append(dict(patch_id=key,physical_id=key,name=key,identity_evidence=evidence,
                      decision_evidence=reason,visibility=visibility,confirmed=admitted,
                      polygons=polygons,exclude_polygons=[]))
    exclusions=[dict(candidate="tooth",reason="处理牙冠和全部牙根排除"),dict(candidate="light_streaks_edges",
                reason="镜面亮斑、深阴影、轮廓和遮挡排除")]
    if event and r["tooth_id"]=="T008" and same_event:
        exclusions.append(dict(candidate="green_ink_and_left_shadow",reason="绿标仅供定位。左上方在后数拍被局部移动阴影覆盖，整组采用右侧空白底面。"))
    for name,polys in event.get("discarded_polygons",{}).items():
        exclusions.append(dict(candidate=name,polygons=polys,reason="上沿材料边界、斜射阴影或连续性不清；未计第二个物块。"))
    return dict(rgb_image_id=r["rgb_image_id"],source_sha256=r["source_sha256"],anchor_rgb_id=r["anchor_rgb_id"],
        review_complete=reviewed and not r["input_issue"],native_size=[r["width"],r["height"]],
        identity_record_missing=c04 or not reviewed,basis_id=event.get("basis_id",""),
        location_method="manual_basis" if r["rgb_image_id"]==event.get("basis_id") else "manual_burst_checked",
        candidates=blocks,excluded=exclusions)


def prepare(data,study,baseline,out):
    data,study,baseline=[Path(v).resolve() for v in (data,study,baseline)]
    out=outside_inputs(out,data,study,baseline)
    paths=dict(data=data,study=study,baseline=baseline)
    files={k:paths[kind]/name for k,(kind,name) in SOURCES.items()}
    inputs={k:sha(p) for k,p in files.items()}
    tables={k:pd.read_csv(p,dtype=str,keep_default_na=False) for k,p in files.items() if k!="config"}
    raw=tables["samples"]
    rows=raw.loc[pd.MultiIndex.from_frame(raw[["tooth_id","angle_id"]]).isin(SLOTS)]
    out.mkdir(parents=True,exist_ok=False)
    if sha(SPEC_PATH)!=PORTABLE_SPEC_HASH:
        raise ValueError("The packaged pilot protocol differs from its published version")
    shutil.copyfile(SPEC_PATH,out/"protocol-source.md")
    scene=load(SCENES)
    indexes={k:{i:g for i,g in df.groupby("rgb_image_id")} for k,df in tables.items() if k in ("baseline","qc")}
    anchors=tables["anchors"]
    population=[]
    geometry=[]
    for ident,samples in rows.groupby("rgb_image_id",sort=True):
        if not re.fullmatch(r"I\d+",ident):
            raise ValueError("Unsafe or missing study image ID: "+ident)
        s=samples.iloc[0]
        historical=indexes["baseline"].get(ident,pd.DataFrame())
        anchor=anchors[anchors.tooth_id.eq(s.tooth_id)&anchors.angle_id.eq(s.angle_id)]
        ar=anchor.iloc[0].to_dict() if len(anchor)==1 else {}
        h=historical.iloc[0].to_dict() if len(historical)==1 else {}
        qc=indexes["qc"].get(ident,pd.DataFrame())
        issue=""
        if len(samples)!=1 or len(historical)>1 or len(anchor)!=1:
            issue="key_or_anchor_conflict"
        keys=["sample_id","tooth_id","state_id","angle_id","session_id","repeat_rank","group_id","rgb_event","qlf_event","qlf_image_id"]
        if h and (any(s[k]!=h[k] for k in keys) or s.rgb_sha256!=h["expected_source_sha256"]
                  or s.rgb_resolved!=h["source_path"] or h.get("anchor_rgb_id")!=ar.get("anchor_rgb_id")):
            issue="paired_identity_conflict"
        r={k:s[k] for k in keys}
        r.update(rgb_image_id=ident,source_path=s.rgb_resolved,source_sha256=s.rgb_sha256,
                 source_pixel_sha256=s.rgb_pixel_sha256,anchor_rgb_id=ar.get("anchor_rgb_id",""),
                 anchor_sha256=ar.get("anchor_sha256",""),anchor_event=ar.get("anchor_rgb_event",""),
                 is_reference=ident==ar.get("anchor_rgb_id",""),old_status=h.get("status","unknown"),
                 old_reason=h.get("reason",""),width=0,height=0,orientation="",input_issue=issue,input_detail="")
        try:
            im,info=checked_source(data,r)
            r.update(width=info["width"],height=info["height"],orientation=info["orientation"])
        except (OSError,ValueError) as ex:
            r.update(input_issue="input_missing" if isinstance(ex,FileNotFoundError) else "decode_or_identity_error",
                     input_detail=str(ex))
        population.append(r)
        geometry.append(dict(rgb_image_id=ident,baseline=h,anchor=ar,qc=qc.iloc[0].to_dict() if len(qc)==1 else {}))
    refs={r["rgb_image_id"]:r for r in population}
    for r in population:
        target=refs.get(r["anchor_rgb_id"])
        if not target or target["state_id"]!="S0" or (target["tooth_id"],target["angle_id"])!=(r["tooth_id"],r["angle_id"]) or target["source_sha256"]!=r["anchor_sha256"]:
            r["input_issue"]="invalid_S0_reference"
    write_csv(out/"pilot_manifest.csv",population)
    json_lines(out/"baseline-rows.jsonl",geometry)
    json_lines(out/"regions.jsonl",[sheet(r,scene) for r in population])
    dump(out/"inputs.json",dict(roots={k:relative_path(v,out) for k,v in paths.items()},
        source_tables=inputs,population=len(raw),pilot_frames=len(population),original_md_sha256=SPEC_HASH,
        operator_record=scene["operator_record"],review_template_sha256=sha(SCENES)))
    return out

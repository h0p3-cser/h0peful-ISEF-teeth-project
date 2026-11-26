"""Protocol numerical checks and two independent end-to-end reconciliations."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..files import dump,load,relative_path,sha
from ..rgb.color import apply_rgb,decode,pixel_hash
from .freeze import check,combine,paired_blocks,source_fingerprints
from .protocol import STATES,SUCCESS,checked_files,solve,split,read_lines


def require(condition,message):
    if not condition:
        raise ValueError(message)


def coefficients(pairs,ids,cfg):
    return np.array([v[:2] for v in solve(pairs,ids,cfg)])


def numerical_checks(cfg):
    levels=[.2,.4,.6,.8]
    truth=np.array([[1.1,.01],[.9,.02],[1.05,-.01]])
    pairs={str(i):(np.full(3,x),truth[:,0]*x+truth[:,1]) for i,x in enumerate(levels)}
    co=coefficients(pairs,sorted(pairs),cfg)
    pred=.5*co[:,0]+co[:,1]
    error=max(abs(pred-[.560,.470,.515]))
    red=.95*co[0,0]+co[0,1]
    require(error<=1e-4 and abs(red-1.055)<=1e-4 and red>1,"Synthetic Huber direction/overrange mismatch")
    def region(name,pid,x,y):
        poly=[[x,y],[x+5,y],[x+5,y+5],[x,y+5]]
        return dict(patch_id=name,physical_id=pid,confirmed=True,visibility="present",polygons=[poly])
    def moved(dx,dy):
        first=np.zeros((56,90,3),np.uint8)
        second=first.copy()
        l,r=[],[]
        for i,level in enumerate([65,110,160,204]):
            x,y=4+17*i,5
            X,Y=x+dx,y+dy
            first[y:y+6,x:x+6]=level
            second[Y:Y+6,X:X+6]=[level+2,level-1,level+1]
            k=str(i);l.append(region(k,k,x,y));r.append(region(k,k,X,Y))
        _,obs=combine(first,l)
        _,ref=combine(second,r)
        return paired_blocks(obs,ref)
    origin,translated=moved(0,0),moved(4,12)
    fi,held,why=split(origin)
    require(not why and held==sorted(origin)[0],"Whole-block lexicographic holdout mismatch")
    test=coefficients(origin,fi,cfg)
    require(np.array_equal(test,coefficients(translated,fi,cfg)),"Geometry translation changed the coefficients")
    perturbed=dict(origin)
    perturbed[held]=(origin[held][0],[.01,.99,.5])
    require(np.array_equal(test,coefficients(perturbed,fi,cfg)),"Frozen holdout contaminated the fit")
    # A lexicographically early physical block may be indispensable for the span.
    bright_only={"A":([.9]*3,[.8]*3),"B":([.1]*3,[.1]*3),"C":([.105]*3,[.106]*3)}
    require(split(bright_only)[:2]==(["A","C"],"B"),"Failed to move on from an infeasible holdout")
    plain=np.full((32,40,3),110,np.uint8)
    regs=[region("alias1","same",1,2),region("alias2","same",2,2)]
    masks,observed=combine(plain,regs)
    only=paired_blocks(observed,observed)
    require(len(only)==1 and len(masks["same"])==42 and bool(split(only)[2]),"Duplicated grey block counted twice")
    two={k:origin[k] for k in ("0","3")}
    fixture,held_two,reason=split(two)
    two_coeff=coefficients(two,fixture,cfg)
    output,_,_=apply_rgb(plain,two_coeff[:,0],two_coeff[:,1],None)
    require(held_two is None and not reason and output.shape==plain.shape,"Two-point output/no-holdout mismatch")
    identical={k:([.1]*3,[.1]*3) for k in ("A","B")}
    require(split(identical)[2]=="insufficient_channel_span" and split({})[2]=="no_confirmed_patch","Empty/span failure mismatch")
    codes=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(1,1,3))
    identity,_,flags=apply_rgb(codes,[1.]*3,[0.]*3,None)
    require(np.array_equal(identity,codes) and not flags.any(),"Quantized identity mismatch")
    bad=plain.copy()
    for j,code in ((0,5),(1,250)):
        bad[2:8,1+j,:]=code
    _,few=combine(bad,[regs[0]])
    require(few["same"]["pixels"]==24 and not paired_blocks(few,observed),"Open code limits or 25-pixel gate mismatch")
    return dict(predicted_RGB=pred.tolist(),max_abs_error=float(error),red_preclip_0_95=float(red),red_outside=True,
        translation=True,whole_physical_holdout=True,skip_infeasible_first_holdout=True,
        repeated_physical_block_refused=True,two_point_no_holdout=True,no_confirmed_block=True,
        frozen_holdout_independence=True,open_code_and_min_pixels=True,identity_256=True)


def ledger(directory,expected,lock_sha):
    status=load(directory/"status.json")
    require(status.get("status")=="complete" and status["protocol_lock_sha256"]==lock_sha,"Incomplete or different frozen run")
    checked_files(directory,status["files"])
    df=pd.read_csv(directory/"frame_results.csv",keep_default_na=True,float_precision="round_trip").set_index("rgb_image_id",verify_integrity=True).sort_index()
    require(list(df.index)==expected,"Frame ledger is not the selected source population")
    require(df.status.isin(("reference",*SUCCESS,"unavailable","input_error")).all(),"Unexpected primary status")
    active=df.status.isin((*SUCCESS,"reference"))
    reference=df.status.eq("reference")
    require((df.index[reference]==df.loc[reference,"anchor_rgb_id"]).all(),"Wrong reference identity")
    for k in ("a_R","a_G","a_B","b_R","b_G","b_B"):
        require(np.isfinite(df.loc[active,k]).all() and df.loc[~active,k].isna().all(),"Invalid coefficient ledger")
    require(df.loc[~active,"corrected_path"].isna().all(),"Failure masquerading as an output")
    require(df.raw_de00.notna().equals(df.status.eq("ok_evaluable")),"Independent colour denominator mismatch")
    require((df.loc[df.status.eq("ok_evaluable"),"holdout_id"].notna()).all(),"Held-out block identity missing")
    require(set(df.loc[active,"corrected_path"])=={p.relative_to(directory).as_posix() for p in (directory/"corrected_rgb").glob("*.png")},"PNG count mismatch")
    for uid,row in df.loc[active].iterrows():
        rgb,_=decode(directory/row.corrected_path,corrected=True)
        require(pixel_hash(rgb)==row.corrected_pixel_sha256,"Native PNG hash mismatch for "+uid)
        if row.status=="reference":
            require(row.corrected_pixel_sha256==row.source_pixel_sha256,"Reference pixels differ from source")
        if row.status in SUCCESS:
            require(bool(row.two_point_support)==(row.fit_blocks==2),"Wrong two-point support flag")
    return df,status


def verify_runs(prepared,first,second):
    prepared,first,second=(Path(p).resolve() for p in (prepared,first,second))
    require(first!=second,"Two different executions are required")
    lock,dirs,hash_counts=check(prepared)
    syn=numerical_checks(lock["solver"])
    manifest=pd.read_csv(prepared/"pilot_manifest.csv").set_index("rgb_image_id",verify_integrity=True)
    expected=sorted(manifest.index)
    lock_sha=sha(prepared/"protocol-lock.json")
    a,sa=ledger(first,expected,lock_sha)
    b,sb=ledger(second,expected,lock_sha)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    for name in ("patch_measurements.csv","coverage.csv","coverage_by_event.csv","colour_summary.csv","clipping_summary.csv"):
        require(sha(first/name)==sha(second/name),name+" differs between runs")
    require(set(sa["files"])==set(sb["files"]),"Two runs have different output file sets")
    counts={k:int(a.status.eq(k).sum()) for k in ("reference",*SUCCESS,"unavailable","input_error")}
    require(sum(counts.values())==len(expected),"Status counts do not close")
    case={}
    if "I0654" in a.index:
        q=a.loc["I0654"]
        case=dict(old_pixels=None if pd.isna(q.old_clip_px) else int(q.old_clip_px),
             crown_pixels=None if pd.isna(q.crown_pixels) else int(q.crown_pixels),
             old_matches_53_73=bool(pd.notna(q.old_clip_pct) and round(q.old_clip_pct,2)==53.73),
             new_pixels=None if pd.isna(q.new_clip_px) else int(q.new_clip_px))
    for _,r in pd.read_csv(first/"coverage.csv").iterrows():
        require(r.applicable==r.recovered+r.lost+r.both_ok+r.both_failed+r.unknown_old,"Coverage table does not close")
    base=Path.cwd()
    path=lambda d:relative_path(d,base)
    def quote(v):
        return "'"+str(v).replace("'","''")+"'"
    entry="qlf-stage3 patch-huber-pilot "
    commands=["$trial = Join-Path 'runs' ('rgb_patch_huber_pilot_' + (Get-Date -Format yyyyMMdd_HHmmss_ffffff))",
        entry+"prepare --data "+quote(path(dirs["data"]))+" --study-run "+quote(path(dirs["study"]))+" --baseline "+quote(path(dirs["baseline"]))+" --out $trial",
        entry+"freeze --prepared $trial --regions (Join-Path $trial 'regions.jsonl')",
        entry+"run --prepared $trial --out (Join-Path $trial 'run_a')",entry+"run --prepared $trial --out (Join-Path $trial 'run_b')",
        entry+"verify --prepared $trial --first (Join-Path $trial 'run_a') --second (Join-Path $trial 'run_b')"]
    result=dict(status="verified",protocol_sha256=lock["document_sha256"],protocol_lock_sha256=lock_sha,
         frames=len(a),counts=counts,verified_read_only_inputs=hash_counts,source_tables_verified=True,
         frozen_file_count=len(lock["files"]),identical_code_at_verify=lock["source"]==source_fingerprints(),
         numerical_checks=syn,frame_values_exact=True,png_native_pixels_equal=int(a.corrected_path.notna().sum()),
         identical_artifacts=sum(v==sb["files"][k] for k,v in sa["files"].items()),total_artifacts=len(sa["files"]),
         I0654=case,first=relative_path(first,prepared),second=relative_path(second,prepared),reproduction_commands=commands)
    dump(prepared/"verification.json",result)
    src=(first/"report.md").read_text(encoding="utf-8")
    prefix=relative_path(first,prepared)+"/"
    final=re.sub(r"\]\((?!https?://)([^)]+)\)",lambda m:"]("+prefix+m.group(1)+")",src)
    final+="\n## 真实回放与合成验算\n\n"
    final+=f"同一冻结输入独立运行两次，{len(a)} 条逐帧状态、系数、色差、裁切数完全一致。{result['png_native_pixels_equal']} 张 PNG 重新解码逐像素核验。两次 {result['identical_artifacts']}/{result['total_artifacts']} 个产物字节一致。\n\n"
    final+=f"本地 v1.1 数值内核三通道合成输入 0.5 的预测为 {np.round(syn['predicted_RGB'],8).tolist()}，最大绝对偏差 {syn['max_abs_error']:.3g}；红通道 0.95 裁切前为 {syn['red_preclip_0_95']:.7f}。平移、重复实物、无有效块、两点支持、留出不泄漏及 256 灰阶恒等验算均通过。合成验证只检查计算方向和实现。\n\n"
    final+="[原始执行方案](protocol-source.md) · [63 帧与分组清单](pilot_manifest.csv) · [逐帧标注](regions.jsonl) · [冻结规则](protocol-lock.json) · [原图逐帧双图叠框索引](review/index.json) · [完整核验](verification.json)\n\n"
    final+="在当前源码工程根目录复现，使用已审阅并按原图哈希绑定的标注。\n\n```powershell\n"+"\n".join(commands)+"\n```\n"
    (prepared/"report.md").write_text(final,encoding="utf-8")
    return prepared/"verification.json"

"""Paired held-out colour reporting. The acquisition event and tooth have equal weight in turn."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ..files import FEATURE_CSV, dump, load, sha
from .color import CHANNELS, LINEAR, FrameIssue, decode
from .geometry import polygon_mask, white_body, resize_matrix, transform

METRICS = ("de00","lightness","chroma")
MEASURES = [k for m in METRICS for k in ("before_"+m,"after_"+m,m+"_gain")]
STATUS = ("reference","ok","ok_no_holdout","failed")
REASONS = {"registration_failed":"白光配准未通过","insufficient_channel_span":"牙根通道跨度不足",
           "insufficient_fit_tiles":"拟合块不足","invalid_huber_parameters":"斜率或截距无效",
           "huber_not_converged":"Huber 未收敛","root_not_visible":"缺共同可见根区",
           "no_anchor":"缺 S0 参考","insufficient_holdout_tiles":"独立留出块不足"}


def save_table(out, name, frame):
    frame.to_csv(out/name,index=False,lineterminator="\n",encoding="utf-8-sig")


def tiles_for(out, ids):
    parts = []
    for chunk in pd.read_csv(out/"root_tiles.csv",chunksize=20000):
        part = chunk[chunk.rgb_image_id.isin(ids)]
        if len(part):
            parts.append(part)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def equal_units(evaluable, unit, parent=None):
    keys = ["tooth_id",unit] if unit!="tooth_id" else ["tooth_id"]
    grouped = evaluable.groupby(keys,sort=True)
    plate = grouped[MEASURES].median()
    if parent is None:
        plate["n_frames"] = grouped.size()
        plate["improved_frames"] = grouped.de00_gain.apply(lambda v:(v>0).sum())
    else:
        plate["n_frames"] = grouped.n_frames.sum()
        plate["n_events"] = grouped.size()
        plate["improved_frames"] = grouped.improved_frames.sum()
    return plate.reset_index()


def coverage(frame, keys):
    counts = frame.groupby(keys+["status"]).size().unstack(fill_value=0).reindex(columns=STATUS,fill_value=0)
    counts["total"] = counts[list(STATUS)].sum(axis=1)
    counts["reference_identities"] = frame.groupby(keys).is_reference.sum()
    counts["applicable_nonreference"] = counts.total-counts.reference_identities
    counts["fitted_nonreference"] = counts.ok+counts.ok_no_holdout
    reasons = frame.loc[frame.reason.fillna("").ne("")].groupby(keys).reason.agg(
        lambda values: json.dumps(values.value_counts().sort_index().to_dict(),ensure_ascii=False))
    counts["reasons"] = reasons
    return counts.fillna({"reasons":"{}"}).reset_index()


def number(x):
    return f"{x:.3f}" if pd.notna(x) else "—"


def case_choices(frame):
    valid = frame[frame.status.eq("ok")]
    cases = []
    if not valid.empty:
        pos = valid[valid.de00_gain>0]
        pool = pos if len(pos) else valid
        typical = pool.loc[(pool.de00_gain-pool.de00_gain.median()).abs().idxmin()]
        cases.append(("Typical improvement" if len(pos) else "Typical result",typical))
    worse = valid[valid.de00_gain<0]
    failed = frame[frame.status.eq("failed")]
    if len(worse):
        cases.append(("Largest worsening",worse.loc[worse.de00_gain.idxmin()]))
    elif len(failed):
        cases.append(("Technical failure",failed.iloc[0]))
    special = frame[frame.tooth_id.eq("T009") & frame.angle_id.eq("A090") & ~frame.status.eq("reference")]
    if len(special):
        scored = special[special.status.eq("ok")]
        target = scored.loc[(scored.de00_gain-scored.de00_gain.median()).abs().idxmin()] if len(scored) else special.iloc[0]
        cases.append(("T009 / 90 deg",target))
    clipped = frame[frame.status.isin(["ok","ok_no_holdout"]) & (frame.crown_new_clip_pct.fillna(0)>0)]
    if len(clipped):
        worst = clipped.loc[clipped.crown_new_clip_pct.idxmax()]
        if all(worst.rgb_image_id!=r.rgb_image_id for _,r in cases):
            cases.append(("Most new crown clipping",worst))
    return cases


def work_image(path, size):
    pixels,_ = decode(path)
    return cv2.resize(pixels,size,interpolation=cv2.INTER_AREA)


def sample_figure(out,data,frame,anchors):
    pool = frame[frame.status.eq("ok")]
    preferred = pool[pool.state_id.eq("S1") & pool.pilot_tooth]
    fig,axs = plt.subplots(1,3,figsize=(15,5),layout="constrained",gridspec_kw={"width_ratios":[1,1,.88]})
    if len(pool):
        r = (preferred if len(preferred) else pool).iloc[0]
        a = anchors.loc[r.anchor_rgb_id]
        table = tiles_for(out,[r.rgb_image_id])
        matrix = np.array(json.loads(r.reference_to_current_work))
        root, crown = [np.array(json.loads(a[k])) for k in ("root_work","crown_work")]
        checks = np.array(json.loads(r.check_points)).reshape(-1,5)
        squares = np.array([[[x,y],[x+s,y],[x+s,y+s],[x,y+s],[x,y]] for x,y,s in table[["x","y","side"]].values])
        anatomy = np.vstack([root,crown])
        width = max(np.ptp(anatomy,axis=0))*1.32
        for ax,img,tr,title in [(axs[0],work_image(data/a.source_path,(int(a.work_width),int(a.work_height))),np.eye(3),"Known S0 · "+a.name),
                     (axs[1],work_image(data/r.source_path,(int(r.working_width),int(r.working_height))),matrix,"Current · "+r.rgb_image_id)]:
            ax.imshow(img)
            for pts,color in [(root,"#0ada8a"),(crown,"#d989fe")]:
                outline = transform(np.vstack([pts,pts[0]]),tr)
                ax.plot(*outline.T,color=color,lw=1.4)
            for role,color in [("fit","#5bb1ff"),("holdout","#ffbc3c")]:
                ax.add_collection(LineCollection(transform(squares[table.role.eq(role)],tr),colors=color,linewidths=.55))
            if len(checks):
                p = checks[:,:2] if title.startswith("Known") else checks[:,2:4]
                ax.scatter(*p.T,s=11,marker="+",c="white",linewidths=.65)
            center = transform((anatomy.max(axis=0)+anatomy.min(axis=0))/2,tr)
            ax.set(xlim=(center[0]-width/2,center[0]+width/2),ylim=(center[1]+width/2,center[1]-width/2),title=title)
            ax.axis("off")
        axs[0].legend(handles=[Patch(color=c,label=t) for c,t in [("#0ada8a","Root"),("#d989fe","Crown"),("#5bb1ff","Fit"),("#ffbc3c","Holdout")]],
                      loc="lower center",ncol=2,fontsize=8,framealpha=.88)
        last = axs[2]
        bound = max(.06,max(table[k+c].max() for c in CHANNELS for k in ("current_linear_","reference_linear_"))*1.04)
        last.plot([0,bound],[0,bound],"--",color="#657480",lw=1)
        for c,col in zip(CHANNELS,("#bf3f43","#228158","#317bae")):
            t = table[table.role.eq("fit")]
            x,y = t["current_linear_"+c],t["reference_linear_"+c]
            last.scatter(x,y,color=col,s=9,alpha=.34)
            xs = np.array([x.min(),x.max()])
            last.plot(xs,r["a_"+c]*xs+r["b_"+c],color=col,label=c)
        last.set(xlabel="Current · linear sRGB",ylabel="Reference · linear sRGB",title="Root fit blocks",xlim=(0,bound),ylim=(0,bound))
        last.set_aspect("equal"); last.legend()
        fig.suptitle(f"{r.tooth_id} / {r.angle_id} / {r.state_id}     {int(r.fit_tiles)} fit · {int(r.holdout_tiles)} held out")
        sample_id = r.rgb_image_id
    else:
        for ax in axs:
            ax.axis("off")
        fig.suptitle("No non-reference frame met independent colour QC")
        sample_id = None
    fig.savefig(out/"figures/01-root-sampling.png",dpi=150,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
    return sample_id


def tooth_figure(out,teeth):
    fig,(left,right) = plt.subplots(1,2,figsize=(11,max(4,.57*len(teeth)+1.6)),layout="constrained")
    y = np.arange(len(teeth))
    for i,r in teeth.iterrows():
        left.plot([r.before_de00,r.after_de00], [i,i],color="#9aa4ad",lw=2)
    left.scatter(teeth.before_de00,y,label="Before",color="#9c5b38",s=37,zorder=3)
    left.scatter(teeth.after_de00,y,label="After",color="#177d83",s=37,zorder=3)
    left.set_yticks(y,[f"{r.tooth_id}  ({int(r.n_frames)} / {int(r.n_events)})" for r in teeth.itertuples()])
    left.set(xlabel="Held-out root ΔE00",title="Median across acquisition groups")
    left.invert_yaxis(); left.legend()
    right.barh(y-.16,teeth.lightness_gain,.29,label="Lightness |ΔL*|",color="#486090")
    right.barh(y+.16,teeth.chroma_gain,.29,label="Chromatic distance (a*, b*)",color="#6a995e")
    right.axvline(0,color="black",lw=.8)
    right.set_yticks(y,teeth.tooth_id); right.invert_yaxis(); right.legend(fontsize=8)
    right.set(xlabel="Paired gain (before minus after)",title="Lightness and colour separately")
    fig.suptitle("Equal event weight within tooth · labels show frames / groups")
    fig.savefig(out/"figures/02-tooth-effects.png",dpi=150)
    plt.close(fig)


def crop(image,center,side):
    h,w = image.shape[:2]
    half = min(side//2,h//2,w//2)
    x,y = np.clip(np.rint(center).astype(int),[half,half],[w-half,h-half])
    return image[y-half:y+half,x-half:x+half]


def crown_figure(out,data,frame,anchors):
    chosen = case_choices(frame)
    fig,axes = plt.subplots(max(1,len(chosen)),4,figsize=(11,2.7*max(1,len(chosen))+.7),squeeze=False,layout="constrained")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    for ax,label in zip(axes[0],("Original RGB","Corrected PNG","Known S0 RGB","New clipping in crown ROI")):
        ax.set_title(label,fontsize=10)
    for line,(kind,r) in zip(axes,chosen):
        a = anchors.loc[r.anchor_rgb_id] if r.anchor_rgb_id in anchors.index else None
        line[0].set_ylabel(f"{kind}\n{r.tooth_id} · {r.angle_id} · {r.state_id}\n{r.rgb_image_id}",fontsize=9)
        try:
            img,info = decode(data/r.source_path)
        except (OSError, FrameIssue) as exc:
            line[0].text(.5,.5,"Unavailable source",ha="center",transform=line[0].transAxes)
            continue
        if a is None or not isinstance(r.reference_to_current_work,str):
            line[0].imshow(img)
            for ax in line[1:]:
                ax.text(.5,.5,str(r.reason),ha="center",va="center",fontsize=8,wrap=True,transform=ax.transAxes)
            continue
        ref,_ = decode(data/a.source_path)
        outline = np.array(json.loads(a.crown_work))
        center = outline.mean(axis=0)
        ref_inv = np.linalg.inv(resize_matrix(ref.shape[1],ref.shape[0],(int(a.work_width),int(a.work_height))))
        cur_inv = np.linalg.inv(resize_matrix(img.shape[1],img.shape[0],(int(r.working_width),int(r.working_height))))
        ref_to_native = cur_inv @ np.array(json.loads(r.reference_to_current_work))
        pos = transform(center,ref_to_native)
        side = 416
        raw_crop = crop(img,pos,side)
        line[0].imshow(raw_crop)
        line[2].imshow(crop(ref,transform(center,ref_inv),side))
        if r.status=="failed":
            line[1].text(.5,.5,r.reason,ha="center",transform=line[1].transAxes)
            line[3].text(.5,.5,"Clipping unavailable",ha="center",transform=line[3].transAxes)
            continue
        corrected,_ = decode(out/r.corrected_path,corrected=True)
        after = crop(corrected,pos,side)
        line[1].imshow(after)
        roi = polygon_mask(img.shape[:2],transform(outline,ref_to_native)) & white_body(data/r.mask_path,info,img)[0]
        mask = crop(roi,pos,side) & (raw_crop.min(axis=2)>0) & (raw_crop.max(axis=2)<255)
        clipped = np.zeros(mask.shape,bool)
        for i,c in enumerate(CHANNELS):
            mapped = LINEAR[raw_crop[...,i]]*r["a_"+c]+r["b_"+c]
            clipped |= (mapped<0) | (mapped>1)
        overlay = after.copy()
        overlay[mask & clipped] = [255,0,175]
        line[3].imshow(overlay)
        line[3].set_xlabel(f"Entire crown ROI: {number(r.crown_new_clip_pct)}%")
    fig.suptitle("416 × 416 native pixels per crop · magenta marks newly clipped pixels")
    fig.savefig(out/"figures/03-crown-detail.png",dpi=160)
    plt.close(fig)
    return [{"case":k,"rgb_image_id":r.rgb_image_id,"status":r.status} for k,r in chosen]


def report(out,data,study_run):
    out = Path(out)
    cfg = load(out/"config.json")
    cal = pd.read_csv(out/"rgb_calibration.csv")
    qc = pd.read_csv(out/"qc_by_frame.csv")
    anchors = pd.read_csv(out/"anchors.csv").set_index("anchor_rgb_id",drop=False)
    frame = cal.merge(qc[["rgb_image_id"]+[k for k in qc if k not in cal]],on="rgb_image_id",validate="one_to_one")
    frame["is_reference"] = frame.rgb_image_id.eq(frame.anchor_rgb_id)
    for k in MEASURES+["crown_new_clip_pct","crown_pixels","crown_new_clip_px","reference_to_current_work"]:
        if k not in frame:
            frame[k] = np.nan
    assessed = frame[frame.status.eq("ok")].copy()
    paired = paired_inputs(out,data,study_run,frame,anchors)
    ready = paired[paired.paired_eligible]
    if assessed[MEASURES].isna().any().any():
        raise ValueError("An evaluable frame is missing a held-out colour measurement")
    events = equal_units(assessed,"group_id")
    teeth = equal_units(events,"tooth_id",parent="group_id")
    for name,tab in [("qc_by_event.csv",events),("qc_by_tooth.csv",teeth)]:
        save_table(out,name,tab)
    multi = ["tooth_id","angle_id","state_id","session_id"]
    save_table(out,"qc_coverage.csv",coverage(frame,multi))
    save_table(out,"qc_failure_cases.csv",frame.loc[frame.status.eq("failed"),["sample_id","rgb_image_id",*multi,"group_id","source_path","anchor_rgb_id","reason","detail"]]
               if "detail" in frame else frame.loc[frame.status.eq("failed"),["sample_id","rgb_image_id",*multi,"reason"]])
    anchor_events = anchors.get("anchor_rgb_event",pd.Series(index=anchors.index,dtype=str))
    same = assessed[assessed.state_id.eq("S0") & assessed.rgb_event.eq(assessed.anchor_rgb_id.map(anchor_events))]
    save_table(out,"qc_s0_repeats.csv",same[["sample_id","rgb_image_id","anchor_rgb_id","group_id","rgb_event","before_de00","after_de00","de00_gain"]])
    (out/"figures").mkdir(exist_ok=True)
    example = sample_figure(out,data,frame,anchors)
    tooth_figure(out,teeth)
    examples = crown_figure(out,data,frame,anchors)
    cnt = frame.status.value_counts().reindex(STATUS,fill_value=0)
    fmt = lambda s: {str(k):int(v) for k,v in s.items()}
    def effect(sub):
        return dict(teeth=len(sub),frames=int(sub.n_frames.sum()),groups=int(sub.n_events.sum()),
                    positive_de00=int((sub.de00_gain>0).sum()),positive_chroma=int((sub.chroma_gain>0).sum()),
                    **{m:float(sub[m].mean()) if len(sub) else None for m in MEASURES})
    pilot = teeth.tooth_id.isin(cfg["protocol"]["pilot_teeth"])
    fitted = frame[frame.status.isin(["ok","ok_no_holdout"])]
    crown = fitted.crown_new_clip_pct.dropna()
    counts = {"source_frames":len(pd.read_csv(study_run/"evaluation/sample_manifest.csv",usecols=["sample_id"])),
        "scope_frames":len(frame),"scope_groups":frame.group_id.nunique(),"statuses":fmt(cnt),
        "declared_reference_frames":int(frame.is_reference.sum()),"applicable_nonreference":int((~frame.is_reference).sum()),
        "improved_frames":int((assessed.de00_gain>0).sum()),
        "worsened_frames":int((assessed.de00_gain<0).sum()),"tied_frames":int((assessed.de00_gain==0).sum()),
        "s0_same_event_frames":len(same),"s0_raw_de00_p10_p50_p90":[float(v) for v in same.before_de00.quantile([.1,.5,.9])] if len(same) else [],
        "all":effect(teeth),"pilot_teeth":effect(teeth[pilot]),"other_teeth":effect(teeth[~pilot]),
        "crown_evaluable_nonreference":len(crown),"crown_with_independent_root_qc":int(assessed.crown_new_clip_pct.notna().sum()),
        "crown_any_new_clipping":int((crown>0).sum()),
        "crown_pct_median":float(crown.median()) if len(crown) else None,"crown_pct_max":float(crown.max()) if len(crown) else None,
        "crown_pct_p90":float(crown.quantile(.90)) if len(crown) else None,
        "color_interpretation":fmt(frame.get("color_interpretation",pd.Series(dtype=str)).value_counts()),
        "grayscale_sources":int(frame.get("grayscale_content",pd.Series(dtype=bool)).eq(True).sum()),
        "white_body_methods":fmt(frame.get("white_body_method",pd.Series(dtype=str)).value_counts()),
        "independent_frames_by_state":fmt(assessed.state_id.value_counts()),
        "independent_groups_by_state":fmt(assessed.groupby("state_id").group_id.nunique()),
        "failures":fmt(frame.loc[frame.status.eq("failed"),"reason"].value_counts()),
        "author_file_hash_mismatches":int(frame.author_rgb_hash_status.eq("local_differs").sum()),
        "paired_input_frames":len(ready),"paired_input_groups":ready.group_id.nunique(),
        "paired_groups_by_state":fmt(ready.groupby("state_id").group_id.nunique()),
        "sampling_example":example,"crown_examples":examples}
    dump(out/"summary.json",counts)
    lines = ["# 白光 RGB 牙根校正与独立质控","",
        f"正式队列共 {counts['source_frames']} 对。本次 {cfg['scope']} 覆盖 {len(frame)} 帧、{frame.group_id.nunique()} 个实际采集组及 {len(anchors)} 个牙号–视角。"
        f"清单中的参考图 {counts['declared_reference_frames']} 帧，恒等输出 {cnt.reference} 帧，适用的非参考图 {counts['applicable_nonreference']} 帧。其中 {cnt.ok+cnt.ok_no_holdout} 帧拟合成功，"
        f"{cnt.ok} 帧可作独立留出评价，{cnt.ok_no_holdout} 帧留出不足，{cnt.failed} 帧技术失败。","",
        "## 独立牙根评价","",
        "每帧比较相同的预留牙根块。先取块中位数，再取实际采集组内非参考帧的中位数，再对每颗牙的各组取中位数。表中增益为逐帧配对差经过同样汇总的结果，正值表示更接近本牙本视角 S0。","",
        "| 范围 | 可评牙 | 帧 / 组 | ΔE₀₀ 前 → 后 | ΔE₀₀ 配对增益 | 明度增益 | 色度增益 | ΔE₀₀ / 色度改善牙数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group,key in [("全部","all"),("先导三牙","pilot_teeth"),("其余牙","other_teeth")]:
        v = counts[key]
        lines.append(f"| {group} | {v['teeth']} | {v['frames']} / {v['groups']} | {number(v['before_de00'])} → {number(v['after_de00'])} | {number(v['de00_gain'])} | {number(v['lightness_gain'])} | {number(v['chroma_gain'])} | {v['positive_de00']} / {v['positive_chroma']} |")
    lines += ["",f"独立评价中，{counts['improved_frames']} 帧综合色差减小，{counts['worsened_frames']} 帧增大，{counts['tied_frames']} 帧持平。"
        f"同一参考拍摄事件的非参考 S0 连拍共 {len(same)} 帧，原始 ΔE₀₀ 的 P10 / P50 / P90 为 "+
        (" / ".join(number(x) for x in counts["s0_raw_de00_p10_p50_p90"]) or "—")+"。","",
        "![逐牙综合色差及明度、色度变化](figures/02-tooth-effects.png)","",
        "## 取样、牙冠与个案","",cfg["protocol"]["operator_exposure_record"]+
        " 在已核对白光 S0 图上标注保守牙根区和可评牙冠区，结合每帧原生白光牙体掩膜。灰卡或相机 RAW 未见于交付数据。"
        " 跨阶段共同牙根由白光标志点、根轮廓、牙颈边缘和光度有效区共同核验。第三个四分区留出，外侧各隔开一个块宽。","",
        f"图一 {example or '无可评帧'}。蓝框用于拟合，黄框用于评价，白色十字为独立几何检查点。","",
        "![牙根标注、空间分区和三通道 Huber](figures/01-root-sampling.png)","",
        f"非参考输出中 {len(crown)} 帧有可评牙冠，其中 {counts['crown_with_independent_root_qc']} 帧同时有独立牙根评价。"
        f"在这 {len(crown)} 张牙冠中，{counts['crown_any_new_clipping']} 帧有新增裁切。逐帧比例中位数 {number(counts['crown_pct_median'])}%，"
        f"P90 为 {number(counts['crown_pct_p90'])}%，最大 {number(counts['crown_pct_max'])}%。"
        "新增裁切分子来自原本三通道均在 0 和 255 之间、校正后任一通道在裁切前越界的像素。分母为原位可评价牙冠区域的全部像素。逐图表同时给出整图越界。","",
        "![原尺度牙冠细节与新增裁切位置](figures/03-crown-detail.png)","",
        "图三按可评帧中正增益的中位数、最大变差、可回查的 T009／90° 和最大新增裁切选例。各列截取相同原图像素宽度，洋红色表示新增裁切。",""]
    for case in examples:
        r = frame.loc[frame.rgb_image_id.eq(case["rgb_image_id"])].iloc[0]
        lines.append(f"- {case['case']} · {r.rgb_image_id} · {r.tooth_id} / {r.angle_id} / {r.state_id}。ΔE₀₀ {number(r.before_de00)} → {number(r.after_de00)}，牙冠新增裁切 {number(r.crown_new_clip_pct)}%。")
    lines += ["","## 覆盖与来源","",
        "| 范围 | 总帧 | 参考 | 拟合且可评 | 拟合但留出不足 | 失败 |","|---|---:|---:|---:|---:|---:|"]
    for dim in ["tooth_id","angle_id","state_id","session_id"]:
        for r in coverage(frame,[dim]).itertuples(index=False):
            lines.append(f"| {r[0]} | {r.total} | {r.reference} | {r.ok} | {r.ok_no_holdout} | {r.failed} |")
    lines += ["","失败原因及帧数为 "+("、".join(f"{REASONS.get(k,k)} {v}" for k,v in counts["failures"].items()) or "无")+"。"
              " [牙号 × 视角 × 阶段 × 批次的分母与失败原因](qc_coverage.csv) · [逐帧失败个案](qc_failure_cases.csv)","",
              "先导按 T002、T006、T010 的 0° 和 90° 取样。T010 的 0° 在正式清单缺 S1，保留该视角已有帧并补查 45°。"
              " T003 的同角度独立事件沿用原采集组 ID。","",
              "| 批次 | S0 | S1 | S2 |","|---|---:|---:|---:|"]
    for batch,values in pd.crosstab(frame.session_id,frame.state_id).reindex(columns=["S0","S1","S2"],fill_value=0).iterrows():
        lines.append(f"| {batch} | "+" | ".join(map(str,values.tolist()))+" |")
    lines += ["","先导叠图发现 T006／0° 和全套参考图中的 T008／0° 两张原生掩膜圈到了背景。"
        "由此增加白光前景复核。仅在原生掩膜与白光照片的中央明亮前景几乎不相交时启用备用轮廓，"
        "仍须检查独立解剖标志及根／颈边缘。平滑牙根也可由分布在根轮廓上的独立吻合点提供局部几何支持。"
              f"这项几何修订在全量处理前冻结。{counts['white_body_methods'].get('verified_white_foreground_fallback',0)} 帧使用备用轮廓。逐帧来源见 white_body_method。",""]
    for reason,rows in frame[frame.status.eq("failed")].groupby("reason"):
        ex = rows.iloc[0]
        lines.append(f"- 可回查失败例 {ex.rgb_image_id} · {ex.tooth_id} / {ex.angle_id} / {ex.state_id}。"
                     f"{REASONS.get(reason,reason)}。白光参考 {ex.anchor_rgb_id}。")
    if len(crown):
        lines += ["", "牙冠图中应对照白斑内的纹理、高光边界及轮廓。洋红范围表示至少一个通道越界；"
                    "该通道的局部层次可能丢失。牙根色差改善与牙冠细节保留需分别判断。"]
    lines += ["",f"已按同一图像 ID 对齐原始 RGB、校正 PNG 和已有 QLF 特征索引。技术成功并同步移出专用参考及相同像素副本后，"
              f"本次共同候选成员有 {len(ready)} 帧、{ready.group_id.nunique()} 组。"
              "成员资格仅由参考身份、技术成功和两模态可读性决定。[逐帧双模式输入与纳入原因](paired_rgb_inputs.csv)"]
    lines += ["",f"白光源固定为历史模型实际读取的版本。本次统一解码得到 {counts['grayscale_sources']} 帧纯灰度内容。"
        f"{counts['author_file_hash_mismatches']} 帧的本地文件字节与作者目录登记值不同。"
        "全部实际解码像素逐帧与原实验像素哈希比对；作者版本只提供登记哈希的条目无法独立作两份像素比较。","",
        f"QLF 四列源文件 SHA-256 为 `{cfg['input_hashes'][FEATURE_CSV]}`。"
        "本次结果衡量 JPEG 白光相对本牙 S0 的重复一致性。S2 均在 C04 拍摄；固定 30% 特征源于同批牙的阶段曲线。分类增量须在相同成员的留一牙实验中另外检验。","",
        "## 复现","", "从独立项目目录，在安装本项目并取得授权数据与原实验运行目录后执行。每次选一个新输出目录。","",
        "```powershell","qlf-stage3 calibrate-rgb --data data/study --study-run runs/study --scope pilot --out runs/rgb_pilot_replay",
        "qlf-stage3 calibrate-rgb --data data/study --study-run runs/study --scope full --out runs/rgb_full_replay",
        f"qlf-stage3 verify-rgb --data data/study --study-run runs/study --run runs/{out.name} --compare runs/rgb_{cfg['scope']}_replay","```","",
        "[逐帧校正](rgb_calibration.csv) · [逐帧评价](qc_by_frame.csv) · [根块数值](root_tiles.csv) · [逐牙汇总](qc_by_tooth.csv) · [同次 S0 连拍](qc_s0_repeats.csv) · [参考帧](anchors.csv)"]
    (out/"qc_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def paired_inputs(out,data,study_run,frame,anchors):
    manifest = pd.read_csv(study_run/"evaluation/sample_manifest.csv")
    keys = ["sample_id","rgb_image_id","qlf_image_id","group_id","state_id","tooth_id","angle_id",
            "session_id","rgb_event","qlf_event","repeat_rank","test_fold","source_path","corrected_path","status",
            "reason","source_pixel_sha256"]
    pairs = frame.reindex(columns=keys).merge(manifest[["sample_id","qlf_resolved","qlf_sha256"]],
                                              on="sample_id",validate="one_to_one")
    reference_hashes = set(anchors.get("anchor_pixel_sha256",pd.Series(dtype=str)).dropna())
    pairs["dedicated_reference_or_copy"] = (pairs.source_pixel_sha256.isin(reference_hashes) |
                                               pairs.rgb_image_id.isin(anchors.index))
    reasons = []
    for r in pairs.itertuples():
        if r.dedicated_reference_or_copy:
            reason = "dedicated_reference_or_pixel_copy"
        elif r.status not in ("ok","ok_no_holdout"):
            reason = r.reason
        elif not (data/r.qlf_resolved).is_file() or sha(data/r.qlf_resolved)!=r.qlf_sha256:
            reason = "unavailable_paired_qlf"
        else:
            reason = ""
        reasons.append(reason)
    pairs["paired_exclusion"] = reasons
    pairs["paired_eligible"] = pairs.paired_exclusion.eq("")
    save_table(out,"paired_rgb_inputs.csv",pairs)
    return pairs

"""Paired stages, event medians, equally weighted teeth and native clipping endpoints."""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ..files import load
from ..materials.study import OLD_FIT,SLOTS
from .protocol import STATES,SUCCESS,read_lines,write_csv


def transitions(df):
    old=df.old_status.isin(OLD_FIT)
    dated=old|df.old_status.eq("failed")
    new=df.status.isin(SUCCESS)
    return np.select([df.rgb_image_id.eq(df.anchor_rgb_id),~dated,old&new,~old&new,old&~new],
                     ["reference","unknown_old","both_ok","recovered","lost"],default="both_failed")


def counts(sub):
    t=sub.transition
    return dict(total=len(sub),reference=int(t.eq("reference").sum()),applicable=int(t.ne("reference").sum()),
        old_success=int(sub.old_status.isin(OLD_FIT).sum()),new_success=int(sub.status.isin(SUCCESS).sum()),
        evaluable=int(sub.status.eq("ok_evaluable").sum()),no_holdout=int(sub.status.eq("ok_no_holdout").sum()),
        two_point=int(sub.two_point_support.sum()),strong=int((sub.status.eq("ok_evaluable")&sub.fit_blocks.ge(3)).sum()),
        unavailable=int(sub.status.eq("unavailable").sum()),input_error=int(sub.status.eq("input_error").sum()),
        **{k:int(t.eq(k).sum()) for k in ("recovered","lost","both_ok","both_failed","unknown_old")})


def coverage(df):
    rows=[]
    for tooth,angle in (*SLOTS,("ALL","ALL")):
        subset=df if tooth=="ALL" else df[df.tooth_id.eq(tooth)&df.angle_id.eq(angle)]
        for state in STATES:
            rows.append(dict(tooth_id=tooth,angle_id=angle,state_id=state,**counts(subset[subset.state_id.eq(state)])))
    return pd.DataFrame(rows)


def colour(df,out):
    records=[]
    events=[]
    teeth=[]
    for stage in (*STATES,"ALL"):
        base=df[df.status.eq("ok_evaluable")]
        if stage!="ALL":
            base=base[base.state_id.eq(stage)]
        for scope,left,right in (("raw_new","raw","new"),("three_way","old","new")):
            terms=["raw","new","raw_minus_new"] if scope=="raw_new" else ["raw","old","new","raw_minus_new","old_minus_new"]
            metrics=[v+"_"+m for v in terms for m in ("de00","abs_dL","ab_distance")]
            same=base.dropna(subset=[v+"_de00" for v in terms if "_minus_" not in v])
            unit=same.groupby(["tooth_id","rgb_event"])[metrics].median(numeric_only=False)
            tooth=unit.groupby("tooth_id")[metrics].median(numeric_only=False)
            for k,g in unit.iterrows():
                events.append(dict(state_id=stage,comparison=scope,tooth_id=k[0],rgb_event=k[1],**g.to_dict()))
            for k,g in tooth.iterrows():
                teeth.append(dict(state_id=stage,comparison=scope,tooth_id=k,**g.to_dict()))
            changes=same[left+"_minus_"+right+"_de00"]
            records.append(dict(state_id=stage,comparison=scope,frames=len(same),events=len(unit),teeth=len(tooth),
                improved=int((changes>0).sum()),worsened=int((changes<0).sum()),tied=int((changes==0).sum()),
                **tooth.mean().to_dict()))
    write_csv(out/"colour_by_event.csv",events)
    write_csv(out/"colour_by_tooth.csv",teeth)
    return write_csv(out/"colour_summary.csv",records)


def clipping(df,out):
    rows=[]
    for stage in (*STATES,"ALL"):
        sub=df[df.transition.ne("reference")]
        if stage!="ALL":
            sub=sub[sub.state_id.eq(stage)]
        for group,cols in (("both",["old_clip_pct","new_clip_pct"]),("recovered",["new_clip_pct"])):
            pool=sub[sub.transition.eq("both_ok" if group=="both" else "recovered")]
            paired=pool.dropna(subset=cols)
            entry=dict(state_id=stage,comparison=group,eligible_frames=len(pool),scored_frames=len(paired))
            if len(paired):
                for col in cols:
                    values=paired[col].astype(float)
                    entry.update({col+"_median":values.median(),col+"_p90":values.quantile(.90,interpolation="linear"),
                        col+"_max":values.max(),col+"_worst_ids":",".join(paired.loc[values.eq(values.max()),"rgb_image_id"])})
                if group=="both":
                    entry["paired_old_minus_new_pp_median"]=paired.old_minus_new_clip_pp.median()
            rows.append(entry)
    return write_csv(out/"clipping_summary.csv",rows)


def f(x,n=3):
    return "NA" if pd.isna(x) else f"{float(x):.{n}f}"


def coverage_figure(out,summary):
    s=summary[summary.tooth_id.eq("ALL")]
    x=np.arange(3)
    fig,ax=plt.subplots(figsize=(6.6,3.5),layout="constrained")
    for pos,col,label,color in ((-.18,"old_success","Root v1.1","#9299a1"),(.18,"new_success","Physical patches","#257a76")):
        bars=ax.bar(x+pos,s[col],.32,color=color,label=label)
        ax.bar_label(bars,padding=2)
    ax.set(xticks=x,xticklabels=[f"{r.state_id}  (n={r.applicable})" for r in s.itertuples()],
           ylabel="Non-reference frames with an output",ylim=(0,max(3,s.applicable.max()+1)))
    ax.legend(frameon=False)
    fig.savefig(out/"figures/stage-coverage.png",dpi=145)
    plt.close(fig)


def report(prepared,out,df):
    df=df.copy()
    df["transition"]=transitions(df)
    cols=["rgb_image_id","anchor_rgb_id","rgb_event","state_id","transition","old_status","status","reason"]
    write_csv(out/"coverage_by_frame.csv",df[cols])
    cov=coverage(df)
    write_csv(out/"coverage.csv",cov)
    evt=[dict(tooth_id=k[0],angle_id=k[1],state_id=k[2],rgb_event=k[3],**counts(g))
         for k,g in df.groupby(["tooth_id","angle_id","state_id","rgb_event"])]
    write_csv(out/"coverage_by_event.csv",evt)
    reasons=df[~df.status.isin((*SUCCESS,"reference"))].groupby(["state_id","status","reason"]).size().reset_index(name="frames")
    write_csv(out/"failure_reasons.csv",reasons)
    colors=colour(df,out)
    clips=clipping(df,out)
    coverage_figure(out,cov)
    notes=read_lines(prepared/"regions.jsonl")
    orig=load(prepared/"inputs.json")
    reviews=sum(bool(d["review_complete"]) for d in notes)
    bases=sum(d["location_method"]=="manual_basis" and d["review_complete"] for d in notes)
    lines=["# 真实白光实物参照块 Huber 三牙先导","",
      f"执行用户提供的 2026-11-26 版 1.0 原文。从 {orig['population']} 个配对 ID 选入 {len(df)} 帧、{df.rgb_event.nunique()} 次实际采集。每帧都有终态。专用 S0 恒等参考单列。",
      "","## 原文核对与取样","",
      "第 1–3 节排除所有牙根的取样和配准。同一均匀物理块内各窗的有效像素合并取一次中位数，每对块权重为 1。两侧分别保留各通道编码在开区间 (5,250) 的像素，每侧至少 25 个。当前与 S0 的 R、G、B 共六个线性 P90−P10 均须 ≥0.02，采用线性分位插值。",
      "","圆托盘可拟合、留出或因身份及像素不足被剔除。至少有 3 个有效块时，按固定 ID 字典序留出第一个仍能使剩余块达标的完整物理块。两点支持单列。无可行留出且全部有效块合用达标时照常出图，独立色差为空。",
      "",f"人工辅助定位并核对 {reviews} 帧，其中 {bases} 张代表帧描框，{reviews-bases} 张连拍复核原位框，全自动识别 0 帧。仅使用实物几何边界做这项复核。S0 上沿取材避开眩光；T008 左上白底有移动阴影，使用七帧均可对应的右侧绿标旁空白底面。标记本身排除。",
      "",orig["operator_record"]+" S1 的带缺口圆盘可视觉复认；采集改变后外侧无标记暗平面缺少独立身份依据。逐块保存了这些观察和剔除理由。",
      "","## 覆盖率","",
      "| 阶段 | 应拟合 | 旧成功 | 新成功 | 旧败→新成 | 旧成→新败 | 都成功 | 都失败 | 旧未知 | 两点支持 | ≥3 拟合且独立留出 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cov[cov.tooth_id.eq("ALL")].itertuples():
        lines.append(f"| {r.state_id} | {r.applicable} | {r.old_success} | {r.new_success} | {r.recovered} | {r.lost} | {r.both_ok} | {r.both_failed} | {r.unknown_old} | {r.two_point} | {r.strong} |")
    lines+=["","四格中的新失败包含图像不可校正及身份资料不足。照片可读取但 C04 同一实物的证据不足时记 input_error / identity_unverified。",
            "","![分阶段覆盖](figures/stage-coverage.png)",""]
    for r in reasons.itertuples():
        lines.append(f"- {r.state_id} · {r.status} / {r.reason} · {r.frames} 帧。")
    lines+=["","## 同一独立留出块的 ΔE00","",
         "每个差先在同一帧上计算。依次取事件帧中位数、牙内事件中位数、可评牙等权均值。正增益表示改善。改善／变差／持平是逐帧计数，NA 表示没有分母。三法交集只纳入确有历史 PNG 的帧。",
         "","| 阶段 | 比较 | 帧 / 事件 / 牙 | 原图 | 牙根法旧 PNG | 全图新 PNG | 配对增益 | 改善 / 变差 / 持平 |",
         "|---|---|---|---:|---:|---:|---:|---|"]
    for _,r in colors.iterrows():
        triple=r.comparison=="three_way"
        gain=r.get("old_minus_new_de00" if triple else "raw_minus_new_de00")
        changes=f"{r.improved} / {r.worsened} / {r.tied}" if r.frames else "NA"
        lines.append(f"| {r.state_id} | {'三法共有，旧减新' if triple else '原减新'} | {r.frames} / {r.events} / {r.teeth} | {f(r.raw_de00)} | {f(r.get('old_de00'))} | {f(r.new_de00)} | {f(gain)} | {changes} |")
    lines+=["","数据同时保留绝对明度误差和 a*b* 平面距离。无独立留出时不评价颜色，即使校正图已经成功输出。","",
        "## 共用原始牙冠掩膜的裁切","",
        "| 阶段 | 样本 | 有掩膜 / 可候选 | 旧中位 / P90 (%) | 新中位 / P90 (%) | 逐帧旧减新中位数（百分点） | 最严重帧与最大值（旧 / 新） |",
        "|---|---|---:|---|---|---:|---|"]
    for _,r in clips.iterrows():
        def extreme(key):
            return str(r.get(key+"_worst_ids"))+" ("+f(r.get(key+"_max"))+"%)" if pd.notna(r.get(key+"_max")) else "NA"
        lines.append(f"| {r.state_id} | {'共同成功' if r.comparison=='both' else '新救回'} | {r.scored_frames} / {r.eligible_frames} | {f(r.get('old_clip_pct_median'))} / {f(r.get('old_clip_pct_p90'))} | {f(r.get('new_clip_pct_median'))} / {f(r.get('new_clip_pct_p90'))} | {f(r.get('paired_old_minus_new_pp_median'))} | {extreme('old_clip_pct')} / {extreme('new_clip_pct')} |")
    lines+=["","裁切分母取原始有效牙冠。分子保留原本三通道均在 (0,1) 且映射后至少一通道越界的像素。每张图的 6 位越界图在 preclip，位 0/1 为 R 下/上越界，2/3 为 G，4/5 为 B。缺少可靠当前帧牙冠掩膜时只缺失该指标。"]
    case=df[df.rgb_image_id.eq("I0654")]
    if len(case):
        r=case.iloc[0]
        lines+=["","## I0654 原位原尺度","",
         f"原生裁片采用固定半开框 [889,880,1305,1296)，各栏 416 × 416 像素。有效牙冠 {f(r.crown_pixels,0)} 像素。旧法新增 {f(r.old_clip_px,0)} 像素（{f(r.old_clip_pct,2)}%），新法新增 {f(r.new_clip_px,0)} 像素（{f(r.new_clip_pct,2)}%），配对减少 {f(r.old_minus_new_clip_pp,2)} 个百分点。"]
        if r.status in SUCCESS:
            detail=load(out/"I0654-detail.json")["measured"]
            lines+=["","| 固定局部 | 原图 | 旧牙根 | 新全图 |","|---|---:|---:|---:|"]
            for key,label in (("stripe_median_L","纹内 L* 中位"),("stripe_L_iqr","纹内 L* 四分位距"),("enamel_minus_stripe_L","邻近牙面减棕黑纹的 ΔL*")):
                lines.append("| "+label+" | "+" | ".join(f(detail.get(k,{}).get(key),2) for k in ("raw","root","material"))+" |")
        else:
            lines+=["",f"新方法无法处理本图（{r.reason}）。校正后的棕黑纹及裁切变化无法比较。"]
        lines+=["","![I0654 同位置原图、旧法、新法与各自裁切](figures/I0654-native-comparison.png)"]
    lines+=["","## 原图配对与失败案例",""]
    for title,s in [("有留出块的实例",df[df.status.eq("ok_evaluable")]),("实物块不足的实例",df[df.status.eq("unavailable")]),
                    ("C04 身份资料不足的实例",df[df.reason.eq("identity_unverified")])]:
        if len(s):
            r=s.iloc[0];file=r.rgb_image_id+"-pair.jpg"
            src=prepared/"review"/file
            if src.is_file():
                shutil.copyfile(src,out/"figures"/file)
                detail=r.reason if r.reason else "留出 "+r.holdout_id
                lines.extend([f"{title}，{r.rgb_image_id}，{detail}。","",f"![{title}](figures/{file})",""])
    lines+=["本先导按三个原有问题选取案例。上述数据只描述本次有明确实物身份、相应受光和人工辅助定位条件下的图像。合成算例与真实色差分别记录。",
      "","[逐帧结果](frame_results.csv) · [逐块及掩膜索引](patch_measurements.csv) · [逐牙与阶段覆盖](coverage.csv) · [原采集事件](coverage_by_event.csv) · [色差与配对增益](colour_summary.csv) · [裁切与最严重帧](clipping_summary.csv)",""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")

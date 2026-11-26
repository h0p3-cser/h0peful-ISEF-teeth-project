"""Matched denominators and three-way colour comparisons for the locked pilot."""
import numpy as np
import pandas as pd

from ..files import dump, load
from .study import SLOTS

TRANSITIONS=["anchor","retained","recovered","lost","failed_both"]
METRICS=[n+"_"+m for n in ("raw","root","material") for m in ("de00","abs_dL","ab_distance")]
SCOPES=(("external_material","圆托盘（整件留出）"),("archived_withheld_root","未处理牙根（旧方法预留块）"))
REASONS={"insufficient_matched_materials":"可与 S0 配对的拟合材料种类不足",
         "insufficient_fit_blocks":"有效拟合块数不足","insufficient_channel_span":"至少一个通道的亮度跨度不足",
         "invalid_huber_parameters":"Huber 参数无效","huber_not_converged":"Huber 未收敛",
         "burst_position_unconfirmed":"同一连拍内的位置核对未通过"}


def save(out,name,df):
    df.to_csv(out/name,index=False,encoding="utf-8-sig",lineterminator="\n",float_format="%.12g")


def counts(frame):
    return [int(frame.transition.eq(k).sum()) for k in TRANSITIONS]


def formatted(v):
    return "—" if pd.isna(v) else f"{v:.3f}"


def report(out,frames,blocks):
    for col in METRICS:
        if col not in blocks:
            blocks[col]=np.nan
    keys=["rgb_image_id","tooth_id","angle_id","state_id","rgb_event","scope"]
    group=blocks.groupby(keys,sort=True)
    grouped=group[METRICS].median()
    grouped["n_blocks"]=group.size()
    levels=grouped.reset_index()
    labels=["rgb_image_id","anchor_rgb_id","tooth_id","angle_id","state_id","rgb_event","transition","status","old_status"]
    all_rows=frames[labels].merge(pd.DataFrame({"scope":[s for s,_ in SCOPES]}),how="cross")
    units=all_rows.merge(levels,on=keys,how="left",validate="one_to_one")
    units["n_blocks"]=units.n_blocks.fillna(0).astype(int)
    units["identity_reference"]=units.rgb_image_id.eq(units.anchor_rgb_id)
    units["triple_evaluable"]=~units.identity_reference & units[["raw_de00","root_de00","material_de00"]].notna().all(axis=1)
    units["raw_minus_material_de00"]=units.raw_de00-units.material_de00
    units["root_minus_material_de00"]=units.root_de00-units.material_de00
    save(out,"heldout_by_frame.csv",units)
    comparisons=[]
    for scope,_ in SCOPES:
        for state in ("S0","S1","S2"):
            sub=units[units.scope.eq(scope)&units.state_id.eq(state)&~units.identity_reference]
            for view,cols in [("three_matched",["raw_de00","root_de00","material_de00"]),
                              ("raw_and_root",["raw_de00","root_de00"]),
                              ("recovered_raw_and_material",["raw_de00","material_de00"])]:
                pool=sub[sub.transition.eq("recovered")] if view.startswith("recovered") else sub
                same=pool.dropna(subset=cols)
                event=same.groupby(["tooth_id","rgb_event"])[cols].median()
                tooth=event.groupby("tooth_id")[cols].median()
                values=tooth.median().to_dict()
                stat=dict(scope=scope,state_id=state,comparison=view,frames=len(same),events=len(event),teeth=len(tooth),**values)
                if view=="three_matched":
                    stat.update(material_better_raw=int((same.material_de00<same.raw_de00-1e-9).sum()),
                                material_better_root=int((same.material_de00<same.root_de00-1e-9).sum()))
                comparisons.append(stat)
    comparison=pd.DataFrame(comparisons)
    save(out,"heldout_summary.csv",comparison)
    cov=[]
    for tooth,angle in list(SLOTS)+[("ALL","ALL")]:
        for state in ("S0","S1","S2"):
            sub=frames[frames.state_id.eq(state)]
            if tooth!="ALL":
                sub=sub[sub.tooth_id.eq(tooth)&sub.angle_id.eq(angle)]
            a,ret,rec,lost,fail=counts(sub)
            cov.append(dict(tooth_id=tooth,angle_id=angle,state_id=state,frames=len(sub),anchors=a,
                old_fitted=ret+lost,old_failed=rec+fail,new_fitted=ret+rec,retained=ret,recovered=rec,lost=lost,failed_both=fail))
    cov=pd.DataFrame(cov)
    save(out,"coverage.csv",cov)
    fails=frames[frames.status.eq("failed")].groupby(["tooth_id","angle_id","state_id","reason"]).size().reset_index(name="frames")
    save(out,"failure_reasons.csv",fails)
    detail=load(out/"I0654-detail.json")
    item=frames[frames.rgb_image_id.eq("I0654")].iloc[0]
    results=dict(frames=len(frames),anchors=int(frames.status.eq("reference").sum()),
                 new_fitted=int(frames.status.eq("ok").sum()),new_failed=int(frames.status.eq("failed").sum()),
                 per_stage=cov[cov.tooth_id.eq("ALL")].drop(columns=["tooth_id","angle_id"]).to_dict("records"))
    dump(out/"summary.json",results)
    lines=["# 三牙全图实物参照块 Huber 先导实测","",
           "三个指定牙号–视角共 63 张原始彩色白光，9 次采集各有 7 张。每个视角沿用已记录的 S0 参考帧。",
           "", "## 覆盖与失败","",
           "| 牙号 / 视角 | 阶段 | 连拍 | S0 自身 | 旧成功 | 旧失败 | 新拟合 | 救回旧失败 | 丢失旧成功 | 两法均失败 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cov.itertuples():
        lines.append(f"| {r.tooth_id} / {r.angle_id} | {r.state_id} | {r.frames} | {r.anchors} | {r.old_fitted} | {r.old_failed} | {r.new_fitted} | {r.recovered} | {r.lost} | {r.failed_both} |")
    lines+=["","S0 自身共 3 张，输出恒等图，单列计数。救回和丢失按同一张照片的两种方法是否实际输出校正图计算。",""]
    for reason,n in frames[frames.status.eq("failed")].reason.value_counts().items():
        lines.append(f"- {REASONS.get(reason,reason)}共 {n} 张（`{reason}`）。")
    current=frames[frames.state_id.isin(["S1","S2"])]
    diagnostics=["current_all_visible_range_"+c for c in "RGB"]
    if set(diagnostics)<=set(current):
        known=current.dropna(subset=diagnostics)
        if len(known):
            cap=known[diagnostics].to_numpy().max()
            limit=load(out/"protocol.json")["min_linear_span"]
            lines+=["",f"在有匹配物块的 {len(known)} 张 S1/S2 图里，即使把独立预留托盘计入，所有可对应外物的通道块中位数最大跨度也只有 {cap:.6f}（线性 sRGB 0–1）。本次拟合采用的最低跨度为 {limit:.3f}。这项检查只描述现场亮度范围。"]
    lines+=["","## 留出块的同帧比较","",
        "以下比较在同一张图、同一批有效像素和同一 S0 材料上进行。ΔE00 越小，颜色越接近 S0。先取各帧的块中位数，再依次按采集、牙取中位数，最后对牙等权取中位数。S0 恒等帧不参与。最后一列是逐帧配对后的严格改善数。",
        "","| 留出材料 | 阶段 | 三法共有帧 / 采集 / 牙 | 原图 ΔE00 | 牙根 Huber | 全图 Huber | 全图优于原图 / 优于牙根（帧） |",
        "|---|---|---|---:|---:|---:|---:|"]
    nam=dict(SCOPES)
    for r in comparison[comparison.comparison.eq("three_matched")].itertuples():
        lines.append(f"| {nam[r.scope]} | {r.state_id} | {r.frames} / {r.events} / {r.teeth} | {formatted(r.raw_de00)} | {formatted(r.root_de00)} | {formatted(r.material_de00)} | {int(r.material_better_raw)} / {int(r.material_better_root)} |")
    names={"raw_de00":"原图","root_de00":"牙根校正","material_de00":"全图校正"}
    for r in comparison[comparison.comparison.eq("three_matched")&comparison.frames.gt(0)].to_dict("records"):
        winner=min(names,key=lambda key:r[key])
        lines+=["",f"{r['state_id']} 的{nam[r['scope']]}上，{names[winner]}最接近 S0（纳入 {r['frames']} 张非参考帧）。"]
    lines+=["","圆托盘是可辨认的暗色实物。其三个扇区合为一件留出材料。厂商 JPEG 在暗部有近乎恒定的像素，托盘上的小色差只反映这一亮度范围。第二组比较使用旧牙根拟合时整片留出的区域，本次全图拟合也没有使用牙根。表内的空值表示该阶段没有三法均可评价的照片。",
        "","原图与牙根校正的全部可评价帧，以及新方法救回帧上原图与新方法的配对结果，另见 [分阶段明细](heldout_summary.csv)。",
        "","## I0654 的牙冠裁切与棕黑纹","",
        f"旧牙根校正在 {int(item.old_crown_pixels):,} 个原生牙冠像素中新增裁切 {int(item.old_crown_new_clip_px):,} 个（{item.old_crown_new_clip_pct:.4f}%）。"]
    if item.status=="ok":
        vals=detail["measured"]
        lines.extend([f"全图校正在相同掩膜中新增裁切 {int(item.crown_new_clip_px):,} 个（{item.crown_new_clip_pct:.4f}%），较旧法相差 {item.crown_new_clip_pct-item.old_crown_new_clip_pct:+.4f} 个百分点。",
              "","| I0654 固定位置 | 原图 | 牙根 Huber | 全图 Huber |","|---|---:|---:|---:|"])
        for key,label in (("stripe_median_L","棕黑纹 L* 中位数"),("adjacent_enamel_median_L","相邻牙面 L* 中位数"),("enamel_minus_stripe_L","相邻牙面减棕黑纹 ΔL*"),("stripe_L_iqr","纹内 L* 四分位距")):
            lines.append("| "+label+" | "+" | ".join(f"{vals[v][key]:.2f}" for v in ("raw","root","material"))+" |")
        review=load(out/"regions.json").get("native_case_review",{})
        if (review.get("rgb_image_id")==item.rgb_image_id and review.get("raw_pixel_sha256")==item.source_pixel_sha256
                and review.get("material_pixel_sha256")==item.corrected_pixel_sha256):
            lines+=["",review["observation"]]
    else:
        lines+=["全图校正没有生成合格图像，原因为 "+REASONS.get(item.reason,item.reason)+"。本例新裁切率和校正后的纹理保留均无法评价。"]
    lines+=["","原始裁片为半开像素框 "+str(detail["native_xyxy"])+"，每栏均为原生 416 × 416 像素。上排是保存后重新解码的像素，下排在同一裁片上标注新增裁切（紫）、棕黑纹（青框）、邻近牙面（黄框）。裁切的分母取整块历史牙冠。分子取原图三通道均在 1–254 的像素中，映射后至少一通道超出线性 [0,1] 的数量，在取整之前统计。",
       "","![I0654 原位等尺度局部](figures/I0654-native-comparison.png)","",
       "## 配对依据与复现","",
       "材料位置由原始白光人工审阅，并绑定九个采集各自代表帧的 SHA-256。每次连拍先核对结构边缘的位移，再读取该帧全部指定区域。跨阶段分别标出实物，没有沿用 S0 的绝对坐标。记录中仅确认可见材料和所在部位，未测量材料的绝对反射率。",
       "","本次实测依据聊天中可见的新版执行要点。在线 Markdown 全文在本地账户下无法打开，以下关于允许使用的材料、保留整件托盘、最少块数和亮度跨度，是本次实现的明确约定，尚待与原文件逐条核对。拟合采用外部暗色底板和能在同次连拍辨认的上沿实物。研究者已确认所拍牙根未处理，本次仅用它作独立评价。这个约定直接影响跨阶段覆盖率。",
       "","每个物块取有效像素在线性 sRGB 中的通道中位数。各实物总权重相同。三个通道分别用 Huber 将当前映射到同牙同视角 S0，原图全分辨率统一应用映射。有效像素、数量、通道跨度和拟合参数均由 [本次参数](protocol.json) 锁定。三通道任一方缺少可辨亮度跨度即登记失败。未处理帧仍保留在总表。",
       "","![三牙三个阶段的原图和取样区](figures/material-review.png)","",
       "[I0654 及其 S0 的上沿实物原位取样](figures/I0654-upper-substrate.png) · [T006 全部连拍](figures/T006-all-bursts.png) · [T008 全部连拍](figures/T008-all-bursts.png) · [T007 全部连拍](figures/T007-all-bursts.png)","",
       "[逐图状态及裁切](frame_audit.csv) · [块身份、坐标、权重及颜色](material_blocks.csv) · [逐块三法读数](heldout_blocks.csv) · [逐帧留出分数](heldout_by_frame.csv) · [目视区域记录](regions.json) · [输入及源码校验值](provenance.json)",
       "","在已安装此项目并放好输入的 stage3 根目录执行以下两条命令。第一条创建新目录。第二条重新运行整个先导，并核对三阶段分母、参数、区域、表格、图像和产物校验值。","",
       "```text","python -m qlf_stage3 material-pilot --data data/study --study-run runs/study --root-run runs/rgb_root_huber_full_20261126 --out runs/material_replay",
       "python -m qlf_stage3 verify-material-pilot --data data/study --study-run runs/study --root-run runs/rgb_root_huber_full_20261126 --run runs/material_replay","```",""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")

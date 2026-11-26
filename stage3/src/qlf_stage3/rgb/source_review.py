"""Render a direct photographic comparison and its measured tooth colours."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..files import load
from ..modeling.cohort import save_csv
from ..modeling.report import md_table
from .color import decode


def photo_gallery(frames, out):
    # Predetermined neighbours in shooting order, with the middle distinct exposure per view.
    gallery = []
    fig, axes = plt.subplots(3, 5, figsize=(13, 5.35))
    fig.subplots_adjust(left=.072, right=.995, bottom=.075, top=.92, hspace=.17, wspace=.035)
    angles = sorted(frames.angle_id.unique())
    distinct = frames[frames.first_exposure_in_view]
    for i, tooth in enumerate(["T004", "T005", "T006"]):
        for j, angle in enumerate(angles):
            ax = axes[i, j]
            shots = distinct[(distinct.tooth_id == tooth) & (distinct.angle_id == angle)].sort_values(
                ["capture_datetime", "rgb_image_id"])
            row = shots.iloc[(len(shots) - 1) // 2]
            raw, _ = decode(Path(row.source_path))
            im = Image.fromarray(raw)
            im.thumbnail((640, 420), Image.Resampling.LANCZOS)
            ax.imshow(im)
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(.025, .045, row.rgb_image_id, transform=ax.transAxes, fontsize=7, color="white",
                    bbox=dict(facecolor="black", alpha=.4, edgecolor="none", pad=1))
            if j == 0:
                ax.set_ylabel(tooth, rotation=0, labelpad=23, va="center", fontsize=11,
                              fontweight="bold" if tooth == "T005" else "normal")
            if i == 0:
                ax.set_title(f"{int(angle[1:])}°", fontsize=11)
            gallery.append([tooth, angle, row.rgb_image_id, row.observed_sha256])
    fig.text(.075,.025, "同批 S2 白光原文件等比展示。中行为本次收到的 T005，上下两行为拍摄时间相邻的两颗对照牙。", fontsize=10)
    fig.savefig(out / "rgb_contact.png", dpi=145, facecolor="white")
    plt.close(fig)
    save_csv(out, "gallery_frames.csv", pd.DataFrame(gallery, columns=["tooth_id","angle_id","rgb_image_id","sha256"]))


def color_figure(views, out):
    focal = views[views.tooth_id == "T005"].sort_values("angle_id")
    peers = views[views.tooth_id != "T005"]
    x = np.arange(len(focal))
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))
    for ax, metric, title in zip(axes, ["Lstar","bstar","midtone_bstar"],
            ["牙体亮度 L*", "牙体黄蓝轴 b*", "相近亮度牙体像素的 b*"]):
        by_angle = peers.pivot(index="angle_id", columns="tooth_id", values=metric).reindex(focal.angle_id)
        for index, tooth in enumerate(by_angle):
            ax.plot(x, by_angle[tooth], color="#c4cfd5", lw=1.0, label="单颗对照牙" if index == 0 else None)
        ax.plot(x, by_angle.median(axis=1), color="#31516a", ls="--", lw=1.8, label="八颗对照牙中位数")
        ax.plot(x, focal[metric], color="#b44522", marker="o", lw=2.4, label="T005")
        ax.set(xticks=x, xticklabels=[f"{int(a[1:])}°" for a in focal.angle_id], title=title)
        ax.grid(axis="y", color="#e7ebed")
        ax.spines[["top","right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=.055, right=.985, top=.88, bottom=.22, wspace=.22)
    fig.savefig(out / "view_color.png", dpi=145, facecolor="white")
    plt.close(fig)


def report(out):
    frames = pd.read_csv(out / "photo_features.csv")
    views = pd.read_csv(out / "view_features.csv")
    comparisons = pd.read_csv(out / "same_view_comparison.csv").set_index(["angle_id","metric"])
    design = load(out / "design.json")
    focal = frames[frames.tooth_id == "T005"]
    peers = frames[frames.tooth_id != "T005"]
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Microsoft YaHei","Noto Sans CJK SC","PingFang SC","DejaVu Sans"],
                         "axes.unicode_minus":False})
    photo_gallery(frames, out)
    color_figure(views, out)
    measures = ["Lstar","astar","bstar","midtone_bstar","background_R","background_G","background_B"]
    summaries = views.groupby("tooth_id")[measures].median()
    save_csv(out, "tooth_summary.csv", summaries.reset_index())
    controls = summaries.drop("T005")
    def view_cells(angle, metric):
        r = comparisons.loc[(angle, metric)]
        return [f"{r.T005:.2f}", f"{r.control_median:.2f} [{r.control_min:.2f}, {r.control_max:.2f}]"]
    angle_table = md_table(["视角","T005 牙体 b*","八牙中位数 [范围]","T005 中等亮度 b*","八牙中位数 [范围]"],
        [[f"{int(a[1:])}°", *view_cells(a, "bstar"), *view_cells(a, "midtone_bstar")]
         for a in sorted(views.angle_id.unique())])
    file_table = md_table(["文件来源","RGB 张数","不同拍摄","匹配相机指纹","匹配交付指纹","JPEG 色度采样","量化表指纹前 12 位"],
        [[name,len(s),s.camera_sha256.nunique(),int(s.camera_match.sum()),int(s.delivered_match.sum()),
          "、".join(s.jpeg_sampling.unique()),"、".join(v[:12] for v in s.quant_table_sha256.unique())]
         for name,s in [("T005 新目录",focal),("其余八牙，采集备份",peers)]])
    whole = comparisons.xs("bstar",level="metric")
    mid = comparisons.xs("midtone_bstar",level="metric")
    bg = ", ".join(str(round(summaries.loc["T005",k],1)) for k in measures[-3:])
    bgc = ", ".join(str(round(controls[k].median(),1)) for k in measures[-3:])
    same_metadata = (frames.exif_camera.eq("EON").all() and frames.exif_white_balance.eq(0).all()
                     and frames.exif_iso.eq(500).all())
    camera_note = "所有照片的 EXIF 均记为 EON、相机自动白平衡、ISO 500。" if same_metadata else "逐图拍摄设置见特征表。"
    encoders_differ = (set(focal.jpeg_sampling).isdisjoint(peers.jpeg_sampling)
                       and set(focal.quant_table_sha256).isdisjoint(peers.quant_table_sha256))
    if encoders_differ:
        camera_note += "T005 的 JPEG 色度采样和量化表均与其余八牙不同，说明两组文件的保存编码有差异。"
    camera_note += "JPEG 编码本身不能确定具体的调色操作。"
    n_delivered = int(focal.delivered_match.sum())
    n_camera = int(focal.camera_match.sum())
    source_note = (f"上游交付说明记载 T005 的 35 张做过 75% 强度的牙体白平衡。"
                   f"新目录中有 {n_delivered}/{len(focal)} 张匹配交付指纹，{n_camera}/{len(focal)} 张匹配登记的相机指纹。"
                   "两种指纹衡量的是文件版本。牙齿个体、取景和光照都会影响牙体像素。")
    if not n_camera:
        source_note += "目前缺少同一拍摄的相机版，无法从实图确定白平衡操作及其强度，现有证据也不足以认定该记录误填。"
    text = [
        "# T005 的 S2 白光和同批其他牙有何差别", "",
        f"核查本次收到的 {len(focal)} 张 T005 白光，比较同一 C04 批次、同一阶段、相同五个视角的另外八颗牙，共 {len(peers)} 张相机指纹一致的白光。每个视角先去掉同一原始拍摄的重复文件，再按牙比较。两组分别有 {focal.camera_sha256.nunique()} 和 {peers.camera_sha256.nunique()} 次不同拍摄。", "",
        "## 牙体颜色", "",
        f"b* 越大表示颜色越偏黄，越小越偏蓝。T005 在 {int(whole.T005.gt(whole.control_median).sum())} 个视角均高于对照中位数，其中 {int(whole.outside_control_range.sum())} 个视角超过该角度八颗对照牙的范围。先对每颗牙的五个角度取中位数，T005 为 {summaries.loc['T005','bstar']:+.2f}，八颗对照牙的中位数为 {controls.bstar.median():+.2f}，范围为 {controls.bstar.min():+.2f} 至 {controls.bstar.max():+.2f}。", "",
        angle_table, "",
        f"仅取 L* 为 45 至 65 的牙体像素，T005 在 {int(mid.T005.gt(mid.control_median).sum())} 个视角仍高于同角度中位数，其中 {int(mid.outside_control_range.sum())} 个视角超出对照范围。180° 的 T005 整体也偏暗，全牙 L* 为 {view_cells('A180','Lstar')[0]}，其余八牙为 {view_cells('A180','Lstar')[1]}。", "",
        f"画面上方两角的暗背景中，T005 的 R、G、B 中位数为 {bg}，八牙中位数为 {bgc}（8 位）。暗背景只提供背景的实测描述，不能充当标准白板。", "",
        "![按相同视角比较亮度和牙体颜色](view_color.png)", "",
        "![收到的 T005 与拍摄时间相邻的两颗对照牙，均为同批 S2 白光](rgb_contact.png)", "",
        "## 文件和来源", "", file_table, "",
        camera_note, "", source_note, "",
        "## 测量与复现", "",
        f"从显示方向正确的全幅 sRGB 像素读取既有牙体掩膜，向内收 3 个原图像素，排除任一通道 ≤5 或 ≥250 的像素。对每张图的有效牙体像素取 CIE L*a*b* 中位数（D65、2°）。本批每张至少有 {int(frames.valid_pixels.min()):,} 个有效牙体像素、{int(frames.midtone_pixels.min()):,} 个落在指定亮度范围的像素。角度内按原始拍摄指纹去重，跨牙以牙为单位。图中展示每颗牙每个视角按时间排序后居中的一张独立拍摄，偶数时取较早一张。全幅等比缩小，不另调色或调亮度。", "",
        "[逐张像素与编码特征](photo_features.csv) · [同角度对照](same_view_comparison.csv) · [逐牙汇总](tooth_summary.csv) · [示例图号](gallery_frames.csv)", "",
        "在项目根目录运行，输出目录可另选。", "",
        "```powershell",
        "python experiments/compare_t005_rgb_source.py --folder '" + design["source_folder"].replace("'","''") +
        "' --archive '" + design.get("archive_path","data/acquisition_archive").replace("'","''") + "'" +
        (" --masks '" + design["masks_path"].replace("'","''") + "'" if "masks_path" in design else "") +
        " --out runs/t005_s2_rgb_comparison",
        "```", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 14 / 共 15          分析报告（纯文本，可直接贴实验记录）
# -----------------------------------------------------------------------------
# 产物文件名   S0_14_分析报告.txt
# 处理对象     不产生图像，把 01~13 各步测量到的数字排版成一份可引用的文本
# 报告包含
#     [1] 原始通道统计     均值/标准差/CV/分位数/过曝比例
#     [2] 强弱变化指标     去光照后的相对起伏、corr(R,G)、高频纹理能量
#     [3] R/G 比值         中位数、四分位、Otsu 分界、红偏面积、两区对比
#     [4] 加权参数         当前默认参数与色度权重统计
#     [5] R-G 耦合律       R = aG + c、R²、红色过量分布
#     [6] 结论速览         5 条自动生成的判断
# RGB 处理     不处理像素
# 输入         任意 RGB 图片
# 输出         UTF-8 文本
# 用途         所有数字的唯一出处；写论文/汇报时引用这份
# 运行         python3 14_分析报告.py 图片.jpg [-o 目录] [--sigma 自动]
# 依赖         numpy, scipy
# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import _metrics as M  # noqa: E402


def fmt_row(name: str, d: dict) -> str:
    return (f"  {name:<12s}{d['mean']:>9.1f}{d['std']:>9.1f}{d['cv_pct']:>8.1f}%"
            f"{d['p05']:>8.1f}{d['median']:>8.1f}{d['p95']:>8.1f}"
            f"{d['range_1_99']:>10.1f}{d['saturated_frac']*100:>8.2f}%")


def build_report(m: dict) -> str:
    st = C.state_prefix(Path(m["image"]).stem)
    R, G = m["R"], m["G"]
    law = m["coupling_law"]

    lines: list[str] = []
    add = lines.append
    add("=" * 82)
    add(f"图像: {m['image']}   [{st}]")
    add(f"尺寸: {m['shape'][1]} x {m['shape'][0]} px    "
        f"光照归一化 sigma = {m['sigma']:.1f} px    比值去噪 sigma = {m['smooth']:.1f} px")
    add("=" * 82)

    add("\n[1] 原始通道统计（0-255）")
    add(f"  {'通道':<12s}{'均值':>9s}{'标准差':>9s}{'CV':>9s}{'p5':>8s}{'中位':>8s}"
        f"{'p95':>8s}{'p1-p99跨度':>12s}{'过曝%':>8s}")
    add(fmt_row("R", R))
    add(fmt_row("G", G))
    add(fmt_row("B", m["B"]))

    add("\n[2] 强弱变化的关键指标")
    add(f"  R 全局 CV                = {R['cv_pct']:.1f}%")
    add(f"  G 全局 CV                = {G['cv_pct']:.1f}%")
    add(f"  照度归一化后 R 相对起伏   = {m['R_norm_std_pct']:.1f}%  (F_R = R/blur(R) 的标准差)")
    add(f"  照度归一化后 G 相对起伏   = {m['G_norm_std_pct']:.1f}%  (F_G = G/blur(G) 的标准差)")
    add(f"  G 起伏 / R 起伏           = {m['G_over_R_variation']:.2f}x   (>1 表示绿通道信息更多)")
    add(f"  corr(R, G)               = {m['corr_RG']:.4f}")
    add(f"  高频(纹理)能量 std        = R {m['hp_std_R']:.2f} / G {m['hp_std_G']:.2f}")

    add("\n[3] R/G 比值：谁压倒谁")
    add(f"  R/G 中位数               = {m['ratio_median']:.3f}  "
        f"(log2 {m['otsu_log2']:+.3f} 附近的分界见下)")
    add(f"  R/G 四分位               = {m['ratio_q25']:.3f} ~ {m['ratio_q75']:.3f}")
    add(f"  Otsu 自动分界            = {m['otsu_log2']:+.3f}   (R/G ≈ {2**m['otsu_log2']:.2f})")
    add(f"  红偏区域面积占比          = {m['red_shifted_area_pct']:.1f}%")
    add(f"  红偏区 G 均值 = {m['G_mean_red_area']:.1f}   中性区 G 均值 = {m['G_mean_neutral_area']:.1f}")
    add(f"  红偏区 R 均值 = {m['R_mean_red_area']:.1f}   中性区 R 均值 = {m['R_mean_neutral_area']:.1f}")
    add(f"  → 变红偏时 G 掉 "
        f"{100*(1-m['G_mean_red_area']/m['G_mean_neutral_area']):.1f}%，"
        f"R 掉 {100*(1-m['R_mean_red_area']/m['R_mean_neutral_area']):.1f}%")

    add("\n[4] 加权参数（09/10/11 号的默认值）")
    add("  mode=combined  p=2  q=1  gain=3  ref=0.30  floor=0.02  weight_sigma=6")
    add("  色度零点 = 本图中位色偏；权重 w = w0/(1+w0) 平滑饱和")

    add("\n[5] R-G 线性耦合律（本图的物理规律）")
    add(f"  R = {law['a']:.3f} * G + {law['c']:.2f}      R2 = {law['r2']:.4f}")
    add(f"  也就是：R 的 {law['r2']*100:.1f}% 可以由 G 单独解释，剩下的才是真颜色")
    add(f"  红色过量 (R - aG - c): 标准差 {m['red_excess_std']:.2f}，"
        f"p0.5 {m['red_excess_p05']:+.1f}，p99.5 {m['red_excess_p995']:+.1f}")

    add("\n[6] 结论速览")
    add(f"  1. R 与 G 高度同步：corr={m['corr_RG']:.4f}，R = {law['a']:.2f}G + {law['c']:.1f} "
        f"解释了 {law['r2']*100:.1f}% 的 R。")
    add(f"  2. 去掉光照后，G 的独立起伏是 R 的 {m['G_over_R_variation']:.2f} 倍，绿通道对比度更高，")
    add("     病灶检测优先用 G 或 09 号的归一化加权图。")
    add(f"  3. R/G 中位 {m['ratio_median']:.2f}，Otsu 分界 {m['otsu_log2']:+.2f}，"
        f"红偏面积 {m['red_shifted_area_pct']:.1f}%。")
    add("     但 R/G 随亮度变化，单看比值会跟「暗」混淆。")
    add(f"  4. 红色过量 p0.5 = {m['red_excess_p05']:+.1f}，p99.5 = {m['red_excess_p995']:+.1f}，")
    add("     这是扣掉亮度/打光/色温之后剩下的真颜色信号（08 号图）。")
    add("  5. 加权用法：突出强弱用 F^p；突出颜色用 |log2(R/G)| 或红色过量；两者相乘即 combined。")
    return "\n".join(lines)


def main() -> int:
    ap = C.cli_argparser("生成分析报告 txt")
    ap.add_argument("--sigma", type=float, default=None, help="光照归一化尺度，默认长边/12")
    ap.add_argument("--smooth", type=float, default=2.0, help="比值去噪，默认 2")
    args = ap.parse_args()

    m = M.compute(args.image, args.sigma, args.smooth)
    report = build_report(m)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_14_分析报告.txt"
    path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n-> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

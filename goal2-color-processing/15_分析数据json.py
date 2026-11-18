#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 15 / 共 15          分析数据导出（JSON）
# -----------------------------------------------------------------------------
# 产物文件名   S0_15_分析数据.json
# 处理对象     与 14 号完全相同的测量结果，只是换成机器可读格式
# 关键字段
#     corr_RG                  R 与 G 的相关系数
#     coupling_law             {a, c, r2}，即 R = aG + c
#     R / G / B                {mean, std, cv_pct, p01, p05, median, p95, p99,
#                               range_1_99, saturated_frac}
#     R_norm_std_pct           去光照后 R 的相对起伏（%）
#     G_norm_std_pct           去光照后 G 的相对起伏（%）
#     G_over_R_variation       G 起伏 / R 起伏
#     ratio_median / q25 / q75 R/G 分位
#     otsu_log2                log2(R/G) 的自动分界
#     red_shifted_area_pct     红偏区域面积占比
#     red_excess_std / p05 / p995
#     params                   本次运行参数（sigma、smooth）
# RGB 处理     不处理像素
# 输入         任意 RGB 图片
# 输出         UTF-8 JSON（缩进 2）
# 用途         批量跑一整个文件夹 -> 汇总成 CSV 看 S0/S1/S2 趋势
# 运行         python3 15_分析数据json.py 图片.jpg [-o 目录]
# 依赖         numpy, scipy
# =============================================================================

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import _metrics as M  # noqa: E402


def main() -> int:
    ap = C.cli_argparser("导出分析数据 json")
    ap.add_argument("--sigma", type=float, default=None, help="光照归一化尺度，默认长边/12")
    ap.add_argument("--smooth", type=float, default=2.0, help="比值去噪，默认 2")
    args = ap.parse_args()

    m = M.compute(args.image, args.sigma, args.smooth)

    # 去掉下划线开头的中间数组，只留可序列化的数字
    summary = {k: v for k, v in m.items() if not k.startswith("_")}
    summary["state"] = C.state_prefix(args.image.stem)
    summary["params"] = {"sigma": m["sigma"], "smooth": m["smooth"]}

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_15_分析数据.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    law = m["coupling_law"]
    print(f"输入 {args.image.name}")
    print(f"  corr(R,G) = {m['corr_RG']:.4f}")
    print(f"  R = {law['a']:.3f}·G + {law['c']:.2f}   R² = {law['r2']:.4f}")
    print(f"  R/G 中位 {m['ratio_median']:.3f}   红偏面积 {m['red_shifted_area_pct']:.1f}%")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

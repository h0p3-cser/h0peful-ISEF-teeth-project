#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 11 / 共 15          原图彩色 × 色度权重：中性区淡出
# -----------------------------------------------------------------------------
# 产物文件名   S0_11_原图彩色_乘色度权重.png
# 处理对象     保留原图的 RGB，但按"这里偏不偏色"整体压暗
# 处理公式
#     1) 色度权重 w  与 09/10 号同一份（见 09 号文件头）
#     2) 逐像素相乘  out(x,y) = ( R,G,B )(x,y) / 255 × w(x,y)
#     3) 压高端      out / p99.5
# RGB 处理     三个通道**同乘一个系数**，所以色调完全不变、只改亮度；
#              这就是为什么它看起来还是原图的颜色，只是背景被"吃掉"了
# 输入         任意 RGB 图片
# 输出         RGB PNG
# 用途         最直观的一张：颜色强的地方留下、接近中性的地方淡出。
#              能一眼看出"哪些区域是真的有颜色偏好"，比如褐色斑块、深色底座
# 注意         背景若有大面积色斑（S0 的台面就是），它们也会被留下——
#              这是数据本身的性质，不是算法错误
# 运行         python3 11_原图彩色_乘色度权重.py 图片.jpg [--chroma-ref 0.30]
# 依赖         numpy, scipy, pillow
# =============================================================================

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def chroma_weight(R: np.ndarray, G: np.ndarray, sigma: float, ref: float,
                  floor: float, gain: float, q: float) -> np.ndarray:
    """与 09 号同一实现。"""
    lr = np.log2((ndi.gaussian_filter(R, sigma) + 1.0) / (ndi.gaussian_filter(G, sigma) + 1.0))
    lr_ref = float(np.median(lr))
    d = np.abs(lr - lr_ref)
    d = np.sqrt(d ** 2 + floor ** 2) - floor
    w = np.clip(d / ref, 0.0, gain) ** q
    return w / (1.0 + w)


def main() -> int:
    ap = C.cli_argparser("原图 RGB 乘以色度权重")
    ap.add_argument("--chroma-ref", type=float, default=0.30, help="多少 log2 偏色算显著")
    ap.add_argument("--chroma-floor", type=float, default=0.02, help="软绝对值半径")
    ap.add_argument("--chroma-gain", type=float, default=3.0, help="权重上限")
    ap.add_argument("--chroma-q", type=float, default=1.0, help="权重指数")
    ap.add_argument("--weight-sigma", type=float, default=6.0, help="判定色度的尺度(px)")
    args = ap.parse_args()

    raw = C.load_rgb(args.image).astype(np.float64)
    R, G = raw[..., 0], raw[..., 1]

    w = chroma_weight(R, G, args.weight_sigma, args.chroma_ref,
                      args.chroma_floor, args.chroma_gain, args.chroma_q)

    emphasis = (raw / 255.0) * w[..., None]     # 三通道同乘，色调不变
    out = C.scale_by(emphasis, 99.5)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_11_原图彩色_乘色度权重.png"
    C.save_rgb_u8(out, path)

    print(f"输入 {args.image.name}")
    print(f"  色度权重 w: 均值 {w.mean():.3f} p50 {np.median(w):.3f} p95 {np.percentile(w,95):.3f}")
    print(f"  被压到 w<0.1 的像素占 {float((w < 0.1).mean())*100:.1f}%（几乎中性 -> 淡出）")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

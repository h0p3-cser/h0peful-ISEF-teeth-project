#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 09 / 共 15          绿色荧光加权：去光照 -> 幂律 -> 乘色度
# -----------------------------------------------------------------------------
# 产物文件名   S0_09_绿光加权_去光照_幂律p2_乘色度.png
# 处理对象     绿色通道 G（这就是"绿色荧光"那一层）
# 处理公式（逐像素，共 5 步）
#     1) 去光照   F  = G / gaussian(G, σ=长边/12≈159px)
#                    除以自己的大尺度模糊 ≈ 除掉了"这块被照得多亮"，
#                    剩下的才是这一块本身的荧光强弱，输出以 1.0 为基准
#     2) 归一     v  = (F − p0.5) / (p99.5 − p0.5)，再截到 0~1
#     3) 幂律     v²                     p=2：强的地方相对更强，弱的地方相对更弱
#     4) 色度     × (0.5 + 0.5·w)        有颜色的地方再抬最多 2 倍（w 见下）
#     5) 压高端   out = out / p99.5（截到 1）—— 只压高端不抬暗部，弱信号保持黑
# 色度权重 w 怎么来的
#     lr = log2( blur(R,6) / blur(G,6) )      在 6px 尺度上判断"这里偏不偏色"
#     d  = | lr − median(lr) |                以本图中位色偏为零点（跨曝光可用）
#     d′ = √(d² + 0.02²) − 0.02               软绝对值，避免在 0 处画出追踪线
#     w0 = clip(d′ / 0.30, 0, 3) ^ q          q=1
#     w  = w0 / (1 + w0)                      平滑饱和到 1，不会留等值线
# RGB 处理     输出是单通道灰度 PNG（R=G=B=out），不带任何颜色偏置
# 输入         任意 RGB 图片
# 输出         灰度 PNG：亮 = 绿荧光强，黑 = 弱
# 用途         看绿色荧光在哪强、在哪弱；斑块边界、牙面纹理在这张里最清楚
# 运行         python3 09_绿光加权_去光照_幂律_色度.py 图片.jpg [--power 2] [--chroma-ref 0.30]
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
    """色度权重：偏色明显的地方 -> 1，接近中性的地方 -> 0。"""
    lr = np.log2((ndi.gaussian_filter(R, sigma) + 1.0) / (ndi.gaussian_filter(G, sigma) + 1.0))
    lr_ref = float(np.median(lr))
    d = np.abs(lr - lr_ref)
    d = np.sqrt(d ** 2 + floor ** 2) - floor          # 软绝对值
    w = np.clip(d / ref, 0.0, gain) ** q
    return w / (1.0 + w)                              # 平滑饱和


def main() -> int:
    ap = C.cli_argparser("绿色荧光加权（去光照 + 幂律 + 色度）")
    ap.add_argument("--power", type=float, default=2.0, help="幂律指数 p，默认 2")
    ap.add_argument("--chroma-ref", type=float, default=0.30, help="多少 log2 偏色算显著")
    ap.add_argument("--chroma-floor", type=float, default=0.02, help="软绝对值半径")
    ap.add_argument("--chroma-gain", type=float, default=3.0, help="权重上限")
    ap.add_argument("--chroma-q", type=float, default=1.0, help="权重指数")
    ap.add_argument("--weight-sigma", type=float, default=6.0, help="判定色度的尺度(px)")
    ap.add_argument("--sigma", type=float, default=None, help="光照归一化尺度，默认长边/12")
    args = ap.parse_args()

    raw = C.load_rgb(args.image).astype(np.float64)
    R, G = raw[..., 0], raw[..., 1]
    sigma = args.sigma or C.default_sigma(G.shape)

    # 1) 去光照 + 2) 归一
    F = C.illumination_normalized(G, sigma)
    v = C.percentile_norm(F)

    # 3) 幂律
    v = v ** args.power

    # 4) 色度因子（半强度：0.5~1.0，温和调制，不会把图像切成两半）
    w = chroma_weight(R, G, args.weight_sigma, args.chroma_ref,
                      args.chroma_floor, args.chroma_gain, args.chroma_q)
    w_gain = 0.5 + 0.5 * w

    # 5) 只压高端
    out = C.scale_by(v * w_gain, 99.5)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_09_绿光加权_去光照_幂律p{args.power:g}_乘色度.png"
    C.save_gray_u8(out, path)

    print(f"输入 {args.image.name}")
    print(f"  光照归一化 σ = {sigma:.1f}px，F 的均值 {F.mean():.3f} 标准差 {F.std():.3f}")
    print(f"  色度权重 w: 均值 {w.mean():.3f} 中位 {np.median(w):.3f} p95 {np.percentile(w,95):.3f}")
    print(f"  输出动态范围 p90/p10 = {(np.percentile(out,90)+1e-6)/(np.percentile(out,10)+1e-6):.1f}x")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

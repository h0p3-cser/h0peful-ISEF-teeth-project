#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 10 / 共 15          红色荧光加权：去光照 -> 幂律 -> 乘色度
# -----------------------------------------------------------------------------
# 产物文件名   S0_10_红光加权_去光照_幂律p2_乘色度.png
# 处理对象     红色通道 R（"红色荧光"那一层）
# 处理公式     与 09 号完全相同，只是把 G 换成 R：
#     1) 去光照   F  = R / gaussian(R, σ=长边/12≈159px)
#     2) 归一     v  = (F − p0.5) / (p99.5 − p0.5) 截到 0~1
#     3) 幂律     v²
#     4) 色度     × (0.5 + 0.5·w)，w 与 09 号用同一份（同一套 R、G 关系）
#     5) 压高端   out / p99.5
# 色度权重 w   与 09 号逐字相同，见 09 号文件头；两张的 w 完全一致，
#              所以 09/10 的差异**只来自 F_G 和 F_R 本身**，这是有意设计的
# RGB 处理     输出单通道灰度 PNG（R=G=B）
# 输入         任意 RGB 图片
# 输出         灰度 PNG：亮 = 红荧光强
# 用途         与 09 号并排：同一块地方，绿光和红光各自被压到多深。
#              变褐的区域两者都暗，但 G 掉得更多（S2 实测 G −61.7%，R −51.1%），
#              所以 09 比 10 的对比更"狠"——病灶检测优先看 09
# 运行         python3 10_红光加权_去光照_幂律_色度.py 图片.jpg
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
    """与 09 号同一实现：偏色明显 -> 1，接近中性 -> 0。"""
    lr = np.log2((ndi.gaussian_filter(R, sigma) + 1.0) / (ndi.gaussian_filter(G, sigma) + 1.0))
    lr_ref = float(np.median(lr))
    d = np.abs(lr - lr_ref)
    d = np.sqrt(d ** 2 + floor ** 2) - floor
    w = np.clip(d / ref, 0.0, gain) ** q
    return w / (1.0 + w)


def main() -> int:
    ap = C.cli_argparser("红色荧光加权（去光照 + 幂律 + 色度）")
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
    sigma = args.sigma or C.default_sigma(R.shape)

    F = C.illumination_normalized(R, sigma)      # 1) 去光照
    v = C.percentile_norm(F)                     # 2) 归一
    v = v ** args.power                          # 3) 幂律

    w = chroma_weight(R, G, args.weight_sigma, args.chroma_ref,
                      args.chroma_floor, args.chroma_gain, args.chroma_q)
    w_gain = 0.5 + 0.5 * w                       # 4) 色度因子

    out = C.scale_by(v * w_gain, 99.5)           # 5) 压高端

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_10_红光加权_去光照_幂律p{args.power:g}_乘色度.png"
    C.save_gray_u8(out, path)

    print(f"输入 {args.image.name}")
    print(f"  光照归一化 σ = {sigma:.1f}px，F 的均值 {F.mean():.3f} 标准差 {F.std():.3f}")
    print(f"  色度权重 w: 均值 {w.mean():.3f} 中位 {np.median(w):.3f} p95 {np.percentile(w,95):.3f}")
    print(f"  输出动态范围 p90/p10 = {(np.percentile(out,90)+1e-6)/(np.percentile(out,10)+1e-6):.1f}x")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

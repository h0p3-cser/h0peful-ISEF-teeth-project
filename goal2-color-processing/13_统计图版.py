#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 13 / 共 15          统计图版：直方图 / 双峰 / 散点与耦合律
# -----------------------------------------------------------------------------
# 产物文件名   S0_13_统计图版.png
# 处理对象     不产生新图，只是把统计量画出来
# 四个面板
#     左上  R、G 的原始强度直方图（对应 03、06 号的合并版）
#           -> 看曝光、看两通道是否同形
#     右上  log2(R/G) 直方图 + Otsu 自动分界
#           -> 这张图是不是"双峰"：中性一堆 + 偏色一堆
#     左下  R–G 散点 + 拟合的耦合律 R = aG + c
#           -> 点几乎贴着一条线 = R 没有独立信息（对应 08 号）
#     右下  去光照后的起伏分布 F_G = G/blur(G)、F_R = R/blur(R)
#           -> 剔除打光后，两通道各自的真实动态范围（对应 09/10 号第 1 步）
# RGB 处理     纯绘图，不影响任何像素数据
# 输入         任意 RGB 图片
# 输出         PNG（约 1700x1000）
# 用途         判断"R/G 到底有没有独立信息"，是决定用 G 还是用 R/G 的依据
# 运行         python3 13_统计图版.py 图片.jpg [--sigma 自动]
# 依赖         numpy, scipy, pillow, matplotlib
# =============================================================================

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

_mpl = Path(tempfile.gettempdir()) / "mplcache_split"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    ap = C.cli_argparser("画统计图版")
    ap.add_argument("--sigma", type=float, default=None, help="光照归一化尺度，默认长边/12")
    args = ap.parse_args()

    raw = C.load_rgb(args.image).astype(np.float64)
    R, G = raw[..., 0], raw[..., 1]
    sigma = args.sigma or C.default_sigma(R.shape)

    Rs = ndi.gaussian_filter(R, 2)
    Gs = ndi.gaussian_filter(G, 2)
    log_ratio = np.log2((Rs + 1.0) / (Gs + 1.0))
    thr = C.otsu_threshold(log_ratio.ravel())

    A = np.vstack([G.ravel(), np.ones(G.size)]).T
    (a, c), *_ = np.linalg.lstsq(A, R.ravel(), rcond=None)
    pred = a * G + c
    r2 = float(1.0 - ((R - pred) ** 2).sum() / ((R - R.mean()) ** 2).sum())

    Gn = C.illumination_normalized(G, sigma)
    Rn = C.illumination_normalized(R, sigma)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), constrained_layout=True)

    ax = axes[0][0]
    ax.hist(G.ravel(), bins=128, color="#2ca02c", alpha=0.75, label="G")
    ax.hist(R.ravel(), bins=128, color="#d62728", alpha=0.5, label="R")
    ax.set_title("Raw intensity histograms (R vs G)")
    ax.set_xlabel("intensity (0-255)")
    ax.set_ylabel("pixel count")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0][1]
    ax.hist(log_ratio.ravel(), bins=220, color="#555555")
    ax.axvline(0.0, color="k", ls="--", lw=1, label="R = G")
    ax.axvline(thr, color="#1f77b4", lw=1.4, label=f"Otsu split {thr:+.3f}")
    ax.set_title("log2(R/G): neutral population vs red-shifted population")
    ax.set_xlabel("log2(R/G)")
    ax.set_ylabel("pixel count")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    ax.scatter(R[::7, ::7].ravel(), G[::7, ::7].ravel(), s=0.8, alpha=0.12, color="#1f77b4")
    lim = [0, 256]
    ax.plot(lim, lim, "k--", lw=1, label="R = G")
    gg = np.array([0.0, 255.0])
    ax.plot(a * gg + c, gg, "-", color="#ff7f0e", lw=1.6,
            label=f"R = {a:.3f}G + {c:.1f}  (R2={r2:.3f})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_title("R vs G scatter + coupling law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    ax.hist(Gn.ravel(), bins=180, color="#2ca02c", alpha=0.75, label="F_G = G / blur(G)")
    ax.hist(Rn.ravel(), bins=180, color="#d62728", alpha=0.5, label="F_R = R / blur(R)")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="1.0 = flat field")
    ax.set_xlim(0.4, 1.6)
    ax.set_title("Illumination-corrected fluctuation (true signal)")
    ax.set_xlabel("ratio to local illumination")
    ax.set_ylabel("pixel count")
    ax.legend()
    ax.grid(alpha=0.25)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_13_统计图版.png"
    fig.savefig(path, dpi=125)
    plt.close(fig)

    print(f"输入 {args.image.name}")
    print(f"  R = {a:.3f}·G + {c:.2f}   R² = {r2:.4f}")
    print(f"  log2(R/G) Otsu 分界 {thr:+.3f}（R/G ≈ {2**thr:.2f}）")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

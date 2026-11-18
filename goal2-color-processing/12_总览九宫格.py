#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 12 / 共 15          总览九宫格：把 01~11 的核心图拼成一张
# -----------------------------------------------------------------------------
# 产物文件名   S0_12_总览九宫格.png
# 处理对象     不产生新数据，只是把前面几步的结果排版到一张图里
# 九格内容（编号对应本包里的脚本）
#     1 原图 RGB
#     2 绿色通道（01 的灰度版）
#     3 红色通道（04 的灰度版）
#     4 F_G = G/blur(G)       —— 09 的第 1 步，去光照后的绿荧光
#     5 F_R = R/blur(R)       —— 10 的第 1 步
#     6 log2(R/G)             —— 07 的比值
#     7 绿光加权              —— 09
#     8 红光加权              —— 10
#     9 红色过量              —— 08
# 色带说明     4/5 用「黑->饱和绿/红->白」单调色带（亮 = 强）；
#              6/9 用 coolwarm 发散色带（中点 = 中性）
# RGB 处理     仅做可视化：把单通道数值按 colormap 上色，不改变数值本身
# 输入         任意 RGB 图片
# 输出         PNG（约 2000x1300）
# 用途         一张图放进汇报/讨论，不用来回翻文件
# 运行         python3 12_总览九宫格.py 图片.jpg [--power 2]
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


# 荧光强度专用色带：黑 -> 饱和色 -> 白，单调，亮 = 强
CMAP_G = LinearSegmentedColormap.from_list(
    "fluor_green",
    [(0.0, (0.0, 0.0, 0.0)), (0.45, (0.0, 0.35, 0.02)),
     (0.80, (0.45, 0.95, 0.50)), (1.0, (1.0, 1.0, 1.0))])
CMAP_R = LinearSegmentedColormap.from_list(
    "fluor_red",
    [(0.0, (0.0, 0.0, 0.0)), (0.45, (0.35, 0.0, 0.0)),
     (0.80, (0.98, 0.45, 0.40)), (1.0, (1.0, 1.0, 1.0))])


def panel(ax, img, title, kind="gray", cmap=None, vmin=None, vmax=None,
          cbar=False, cblabel="", fig=None):
    """画一格：kind = gray / cmap / rgb。"""
    if kind == "rgb":
        ax.imshow(img)
    elif kind == "gray":
        ax.imshow(img, cmap="gray", vmin=0 if vmin is None else vmin,
                  vmax=255 if vmax is None else vmax)
    else:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.015)
            cb.ax.tick_params(labelsize=7)
            if cblabel:
                cb.set_label(cblabel, fontsize=7)
    ax.set_title(title, fontsize=9.5)
    ax.axis("off")


def main() -> int:
    ap = C.cli_argparser("把核心结果拼成 3x3 总览")
    ap.add_argument("--power", type=float, default=2.0, help="幂律指数，与 09/10 保持一致")
    ap.add_argument("--chroma-ref", type=float, default=0.30)
    args = ap.parse_args()

    raw = C.load_rgb(args.image)
    f = raw.astype(np.float64)
    R, G = f[..., 0], f[..., 1]
    sigma = C.default_sigma(R.shape)

    # ---- 中间量（与 07/08/09/10 同源）----
    Rs = ndi.gaussian_filter(R, 2)
    Gs = ndi.gaussian_filter(G, 2)
    log_ratio = np.log2((Rs + 1.0) / (Gs + 1.0))

    FG = C.percentile_norm(C.illumination_normalized(G, sigma))
    FR = C.percentile_norm(C.illumination_normalized(R, sigma))

    A = np.vstack([G.ravel(), np.ones(G.size)]).T
    (a, c), *_ = np.linalg.lstsq(A, R.ravel(), rcond=None)
    red_excess = Rs - (a * Gs + c)

    # 色度权重（与 09/10/11 同源）
    lr_w = np.log2((ndi.gaussian_filter(R, 6) + 1.0) / (ndi.gaussian_filter(G, 6) + 1.0))
    d = np.abs(lr_w - float(np.median(lr_w)))
    d = np.sqrt(d ** 2 + 0.02 ** 2) - 0.02
    w_raw = np.clip(d / args.chroma_ref, 0.0, 3.0)
    w = w_raw / (1.0 + w_raw)

    outG = C.scale_by(C.percentile_norm(C.illumination_normalized(G, sigma)) ** args.power
                      * (0.5 + 0.5 * w), 99.5)
    outR = C.scale_by(C.percentile_norm(C.illumination_normalized(R, sigma)) ** args.power
                      * (0.5 + 0.5 * w), 99.5)

    st = C.state_prefix(args.image.stem)
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 10.6), constrained_layout=True)

    panel(axes[0][0], raw[::2, ::2], "1. Original RGB", kind="rgb")
    panel(axes[0][1], G.astype(np.uint8), "2. G raw (R,B removed)", kind="gray")
    panel(axes[0][2], R.astype(np.uint8), "3. R raw (G,B removed)", kind="gray")

    panel(axes[1][0], FG, "4. F_G : G corrected for lighting", kind="cmap", cmap=CMAP_G,
          vmin=0, vmax=1, cbar=True, cblabel="relative green fluorescence", fig=fig)
    panel(axes[1][1], FR, "5. F_R : R corrected for lighting", kind="cmap", cmap=CMAP_R,
          vmin=0, vmax=1, cbar=True, cblabel="relative red fluorescence", fig=fig)
    panel(axes[1][2], log_ratio, "6. log2(R/G)  blue = G wins | red = R wins",
          kind="cmap", cmap="coolwarm", vmin=-0.30, vmax=0.50, cbar=True,
          cblabel="log2(R/G)", fig=fig)

    panel(axes[2][0], outG, "7. G weighted (p,q): strong kept, weak suppressed", kind="cmap", cmap=CMAP_G, vmin=0, vmax=1)
    panel(axes[2][1], outR, "8. R weighted (p,q): strong kept, weak suppressed", kind="cmap", cmap=CMAP_R, vmin=0, vmax=1)
    lim = float(max(abs(np.percentile(red_excess, 0.5)),
                    abs(np.percentile(red_excess, 99.5)), 1.0))
    panel(axes[2][2], red_excess, "9. red excess = R - (a*G + c): the real colour signal", kind="cmap",
          cmap="coolwarm", vmin=-lim, vmax=lim, cbar=True,
          cblabel="R excess (0-255 counts)", fig=fig)

    fig.suptitle(f"{C.ascii_safe(args.image.stem)}   R/G fluorescence analysis   "
                 f"mode=combined  p={args.power:.1f}  q=1.0  sigma={sigma:.0f}px",
                 fontsize=12)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{st}_12_总览九宫格.png"
    fig.savefig(path, dpi=125)
    plt.close(fig)

    print(f"输入 {args.image.name}")
    print(f"  耦合律 R = {a:.3f}·G + {c:.2f}（第 9 格用）")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

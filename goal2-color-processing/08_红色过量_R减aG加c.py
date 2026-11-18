#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 08 / 共 15          红色过量：扣掉 R-G 线性耦合后剩下的真颜色
# -----------------------------------------------------------------------------
# 产物文件名   S0_08_红色过量_R减(0.649G+19.2).png   ← 文件名里带着实测的 a、c
# 处理对象     红通道相对绿通道"多出来的那一部分"
# 处理公式
#     1) 拟合耦合律   用全图所有像素做最小二乘  R = a·G + c
#                     （这两通道高度同步，一条直线就能解释绝大部分 R）
#     2) 算残差       R′(x,y) = Rs(x,y) − ( a·Gs(x,y) + c )
#                     Rs、Gs 为 σ=2 高斯去噪后的通道
#     3) 显示         对称区间 ±max(|p0.5|, |p99.5|) 映射到 coolwarm
# RGB 处理     输出 RGB 只编码 R′ 一个量：
#               红 = 比耦合律预测得更红（真颜色信号）
#               白 = 完全符合规律（没有独立颜色信息）
#               蓝 = 比预测更绿
# 输入         任意 RGB 图片
# 输出         RGB PNG
# 用途         这是整套里**最干净的颜色指标**：亮度、打光、整体色温都被直线吸收，
#              只剩"这个材料本身多红"。实测 S0 的颈部沉积 +21~+23，健康牙面 +7
# 注意         a、c 是在本图上拟合的，换一张图会变（所以写进文件名）。
#              做序列实验时应改用同批次空白/标准板拟合，再套用到所有样本
# 运行         python3 08_红色过量_R减aG加c.py 图片.jpg
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
from PIL import Image  # noqa: E402


def fit_coupling(R: np.ndarray, G: np.ndarray) -> tuple[float, float, float]:
    """最小二乘拟合 R = a·G + c，返回 (a, c, R²)。"""
    A = np.vstack([G.ravel(), np.ones(G.size)]).T
    (a, c), *_ = np.linalg.lstsq(A, R.ravel(), rcond=None)
    pred = a * G + c
    r2 = float(1.0 - ((R - pred) ** 2).sum() / ((R - R.mean()) ** 2).sum())
    return float(a), float(c), r2


def main() -> int:
    ap = C.cli_argparser("拟合 R=aG+c 并输出残差图")
    ap.add_argument("--smooth", type=float, default=2.0, help="算残差前的去噪模糊(px)，默认 2")
    args = ap.parse_args()

    raw = C.load_rgb(args.image).astype(np.float64)
    R, G = raw[..., 0], raw[..., 1]

    # 1) 在原始像素上拟合（统计量用原始值，不受去噪影响）
    a, c, r2 = fit_coupling(R, G)

    # 2) 在去噪后的通道上算残差
    Rs = ndi.gaussian_filter(R, args.smooth) if args.smooth > 0 else R
    Gs = ndi.gaussian_filter(G, args.smooth) if args.smooth > 0 else G
    red_excess = Rs - (a * Gs + c)

    # 3) 对称色带上限
    p_lo, p_hi = np.percentile(red_excess, [0.5, 99.5])
    lim = float(max(abs(p_lo), abs(p_hi), 1.0))
    clipped = np.clip((red_excess + lim) / (2.0 * lim), 0.0, 1.0)
    rgb = (plt.get_cmap("coolwarm")(clipped)[..., :3] * 255.0).astype(np.uint8)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_08_红色过量_R减({a:.3f}G+{c:.1f}).png"
    Image.fromarray(rgb).save(path)

    hot = red_excess > 10
    print(f"输入 {args.image.name}")
    print(f"  耦合律 R = {a:.3f}·G + {c:.2f}   R² = {r2:.4f}")
    print(f"  残差: 均值 {red_excess.mean():+.2f} 标准差 {red_excess.std():.2f} "
          f"p0.5 {p_lo:+.1f} p99.5 {p_hi:+.1f}")
    print(f"  红色过量 > +10 的像素占 {hot.mean()*100:.2f}%，"
          f"这些像素平均 G = {G[hot].mean() if hot.any() else float('nan'):.1f}（全图 {G.mean():.1f}）")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

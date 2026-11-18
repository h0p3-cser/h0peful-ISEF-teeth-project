#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 07 / 共 15          R/G 比值图（log2 标度，假彩色）
# -----------------------------------------------------------------------------
# 产物文件名   S0_07_比值图_log2(R比G).png
# 处理对象     红、绿两个通道的"比值"，与亮度无关
# 处理公式
#     1) 先去噪  Rs = gaussian(R, σ=2)      Gs = gaussian(G, σ=2)
#                （σ=2 是为了压掉 JPEG 8x8 块噪声，否则比值图会放大压缩瑕疵）
#     2) 求对数比值  lr(x,y) = log2( (Rs+1) / (Gs+1) )
#                +1 是为了避免除 0；log2 让"红是绿的两倍"和"绿是红的两倍"对称
#     3) 显示映射   把 [-0.30, +0.50] 线性映射到 coolwarm 色带并截断
# RGB 处理     输出的 RGB 与原始 RGB 无关，只是把 lr 这一个数值涂成颜色：
#               蓝 = lr<0（G 强）    白 = lr≈0（中性）    红 = lr>0（R 强）
# 输入         任意 RGB 图片
# 输出         RGB PNG（颜色不代表原图颜色，只代表比值）
# 用途         回答"这里到底绿压过红，还是红压过绿"。S0 整幅偏蓝（R/G≈0.78），
#              S2 偏暖（R/G≈1.04）
# 注意         比值随亮度变化（越暗越偏红），所以别把 07 的红当成"炎症"，
#              干净的指标是 08 号的红色过量
# 运行         python3 07_比值图_log2_R比G.py 图片.jpg [--smooth 2] [--lo -0.30 --hi 0.50]
# 依赖         numpy, scipy, pillow, matplotlib（只为取 coolwarm 色带）
# =============================================================================

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

# matplotlib 需要一个可写的缓存目录，沙箱里 ~/.matplotlib 常常只读
_mpl = Path(tempfile.gettempdir()) / "mplcache_split"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> int:
    ap = C.cli_argparser("算 log2(R/G) 并涂成假彩色")
    ap.add_argument("--smooth", type=float, default=2.0, help="比值前的去噪模糊(px)，默认 2")
    ap.add_argument("--lo", type=float, default=-0.30, help="色带左端对应的 log2 值")
    ap.add_argument("--hi", type=float, default=0.50, help="色带右端对应的 log2 值")
    args = ap.parse_args()

    raw = C.load_rgb(args.image).astype(np.float64)
    R, G = raw[..., 0], raw[..., 1]

    # 1) 去噪
    Rs = ndi.gaussian_filter(R, args.smooth) if args.smooth > 0 else R
    Gs = ndi.gaussian_filter(G, args.smooth) if args.smooth > 0 else G

    # 2) 对数比值
    log_ratio = np.log2((Rs + 1.0) / (Gs + 1.0))

    # 3) 映射到色带
    clipped = np.clip((log_ratio - args.lo) / (args.hi - args.lo), 0.0, 1.0)
    rgb = (plt.get_cmap("coolwarm")(clipped)[..., :3] * 255.0).astype(np.uint8)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_07_比值图_log2(R比G).png"
    Image.fromarray(rgb).save(path)

    r_g = (Rs + 1.0) / (Gs + 1.0)
    print(f"输入 {args.image.name}  (去噪 σ={args.smooth})")
    print(f"  R/G 中位数 {np.median(r_g):.3f}   log2 = {np.median(log_ratio):+.3f}")
    print(f"  四分位 {np.percentile(r_g,25):.3f} ~ {np.percentile(r_g,75):.3f}")
    print(f"  色带范围 log2 {args.lo:+.2f} ~ {args.hi:+.2f}（超出即饱和为纯蓝/纯红）")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

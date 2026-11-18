#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 03 / 共 15          绿色通道的灰度直方图
# -----------------------------------------------------------------------------
# 产物文件名   S0_03_仅绿通道G_直方图.png
# 处理对象     绿色通道 G 的 256 级分布
# 处理公式     hist[i] = #{像素 | G(x,y) = i},  i = 0..255
# RGB 处理     不处理原图；把 hist 画成 512x220 的画布
#              画布底色 (255,255,255)，柱子颜色 (40,160,70)，底部 x 轴 (0,0,0)
# 输入         任意 RGB 图片
# 输出         一张小 PNG（512x220），横轴 0~255 灰度，纵轴像素数
# 用途         判断曝光：左端堆起来 = 欠曝，右端贴着 255 = 过曝截断；
#              两峰之间有空档说明图里是"两类东西"（比如牙 vs 背景）
# 说明         柱高按最大值线性归一，纯 numpy 逐列绘制，不依赖 matplotlib
# 运行         python3 03_绿通道直方图.py 图片.jpg
# 依赖         numpy, pillow
# =============================================================================

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

WIDTH, HEIGHT = 512, 220
MARGIN_BOTTOM, MARGIN_LEFT = 24, 8


def draw_histogram(plane: np.ndarray, bar_rgb: tuple[int, int, int]) -> np.ndarray:
    """把一维灰度分布画成 (HEIGHT, WIDTH, 3) 的 uint8 画布。"""
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

    counts, _ = np.histogram(plane.ravel(), bins=256, range=(0, 255))
    counts = counts.astype(np.float64)
    peak = counts.max() if counts.max() > 0 else 1.0
    usable_h = HEIGHT - MARGIN_BOTTOM - 10

    xs = np.linspace(MARGIN_LEFT, WIDTH - MARGIN_LEFT - 1, 256).astype(int)
    bar_h = np.clip(counts / peak * usable_h, 0, usable_h).astype(int)
    for x, h in zip(xs, bar_h):
        if h > 0:
            canvas[HEIGHT - MARGIN_BOTTOM - h:HEIGHT - MARGIN_BOTTOM, x:x + 1] = bar_rgb

    # 底部 x 轴
    canvas[HEIGHT - MARGIN_BOTTOM:HEIGHT - MARGIN_BOTTOM + 1,
           MARGIN_LEFT:WIDTH - MARGIN_LEFT] = 0
    return canvas


def main() -> int:
    ap = C.cli_argparser("画绿色通道的灰度直方图")
    args = ap.parse_args()

    raw = C.load_rgb(args.image)
    G = raw[..., 1]

    canvas = draw_histogram(G, (40, 160, 70))

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_03_仅绿通道G_直方图.png"
    Image.fromarray(canvas).save(path)

    over = float((G >= 254).mean()) * 100
    under = float((G <= 2).mean()) * 100
    print(f"输入 {args.image.name}")
    print(f"  G: 均值 {G.mean():.1f} 中位 {np.median(G):.0f} 过曝 {over:.2f}% 欠曝 {under:.2f}%")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

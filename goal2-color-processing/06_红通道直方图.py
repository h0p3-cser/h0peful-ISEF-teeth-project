#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 06 / 共 15          红色通道的灰度直方图
# -----------------------------------------------------------------------------
# 产物文件名   S0_06_仅红通道R_直方图.png
# 处理对象     红色通道 R 的 256 级分布
# 处理公式     hist[i] = #{像素 | R(x,y) = i},  i = 0..255
# RGB 处理     不处理原图；画布 512x220，底色白、柱子 (220,60,60)、x 轴黑
# 输入         任意 RGB 图片
# 输出         512x220 的小 PNG
# 用途         和 03 号叠在一起看就能回答"R 和 G 是不是被同一束光推着走"
#              （两条直方图形状高度相似 = 同源；错位 = 有独立的颜色成分）
# 运行         python3 06_红通道直方图.py 图片.jpg
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
    """画布尺寸、柱色、边距与 03 号完全一致，方便直接叠加对比。"""
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

    canvas[HEIGHT - MARGIN_BOTTOM:HEIGHT - MARGIN_BOTTOM + 1,
           MARGIN_LEFT:WIDTH - MARGIN_LEFT] = 0
    return canvas


def main() -> int:
    ap = C.cli_argparser("画红色通道的灰度直方图")
    args = ap.parse_args()

    raw = C.load_rgb(args.image)
    R = raw[..., 0]

    canvas = draw_histogram(R, (220, 60, 60))

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_06_仅红通道R_直方图.png"
    Image.fromarray(canvas).save(path)

    over = float((R >= 254).mean()) * 100
    under = float((R <= 2).mean()) * 100
    print(f"输入 {args.image.name}")
    print(f"  R: 均值 {R.mean():.1f} 中位 {np.median(R):.0f} 过曝 {over:.2f}% 欠曝 {under:.2f}%")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

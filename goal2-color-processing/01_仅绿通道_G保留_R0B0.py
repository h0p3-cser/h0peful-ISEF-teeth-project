#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 01 / 共 15          单通道分离：只保留绿色
# -----------------------------------------------------------------------------
# 产物文件名   S0_01_仅绿通道G_R置0B置0.png
# 处理对象     绿色通道 G
# 处理公式     out(x,y) = ( 0 , G(x,y) , 0 )
# RGB 处理     R := 0（整列清零）    G := 原值（一点不动）    B := 0（整列清零）
# 输入         任意 RGB 图片（灰度图会自动复制到三通道）
# 输出         三通道 PNG，但 R、B 两个通道恒为 0，所以画面只剩绿光
# 用途         去掉红和蓝，单独看绿通道的变化
# 注意         这是"纯绿"图，人眼对绿色亮度不敏感，比较强弱请配合 02 号灰度版
# 运行         python3 01_仅绿通道_G保留_R0B0.py 图片.jpg
#              python3 01_仅绿通道_G保留_R0B0.py 图片.jpg -o out/S0_baseline
# 依赖         numpy, pillow
# =============================================================================

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> int:
    ap = C.cli_argparser(__doc__ or "只保留绿色通道")
    args = ap.parse_args()

    raw = C.load_rgb(args.image)          # (H, W, 3) uint8
    G = raw[..., 1]                       # 取出绿通道，原值

    out = np.zeros_like(raw)              # R=0, B=0 已经由 zeros 保证
    out[..., 1] = G                       # 绿通道写回原值

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_01_仅绿通道G_R置0B置0.png"

    C.save_rgb_u8(out.astype(np.float64) / 255.0, path)

    print(f"输入 {args.image.name}")
    print(f"  绿通道 G: 均值 {G.mean():.1f} 标准差 {G.std():.1f} 范围 {G.min()}~{G.max()}")
    print(f"  已把 R、B 置 0 -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

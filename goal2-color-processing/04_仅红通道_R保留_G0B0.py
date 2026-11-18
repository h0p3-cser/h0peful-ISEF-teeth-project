#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 04 / 共 15          单通道分离：只保留红色
# -----------------------------------------------------------------------------
# 产物文件名   S0_04_仅红通道R_G置0B置0.png
# 处理对象     红色通道 R
# 处理公式     out(x,y) = ( R(x,y) , 0 , 0 )
# RGB 处理     R := 原值    G := 0（整列清零）    B := 0（整列清零）
# 输入         任意 RGB 图片
# 输出         三通道 PNG，但 G、B 恒为 0，画面只剩红光
# 用途         去掉绿和蓝，单独看红通道的变化；与 01 号配对，是"绿 vs 红"的另一半
# 注意         同为纯色图，比强弱请用 05 号灰度版
# 运行         python3 04_仅红通道_R保留_G0B0.py 图片.jpg
# 依赖         numpy, pillow
# =============================================================================

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> int:
    ap = C.cli_argparser("只保留红色通道")
    args = ap.parse_args()

    raw = C.load_rgb(args.image)
    R = raw[..., 0]

    out = np.zeros_like(raw)
    out[..., 0] = R

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_04_仅红通道R_G置0B置0.png"

    C.save_rgb_u8(out.astype(np.float64) / 255.0, path)

    print(f"输入 {args.image.name}")
    print(f"  红通道 R: 均值 {R.mean():.1f} 标准差 {R.std():.1f} 范围 {R.min()}~{R.max()}")
    print(f"  已把 G、B 置 0 -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 05 / 共 15          单通道分离：红色通道的灰度版
# -----------------------------------------------------------------------------
# 产物文件名   S0_05_仅红通道R_G置0B置0_转灰度.png
# 处理对象     红色通道 R
# 处理公式     out(x,y) = ( R(x,y) , R(x,y) , R(x,y) )
# RGB 处理     R := R    G := R    B := R
# 输入         任意 RGB 图片
# 输出         三通道 PNG，三通道相等 -> 真灰度图
# 用途         和 02 号并排看，就是"同一块地方，绿和红各有多强"的最直接对比
# 运行         python3 05_仅红通道_转灰度.py 图片.jpg
# 依赖         numpy, pillow
# =============================================================================

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> int:
    ap = C.cli_argparser("把红色通道复制到三通道，得到灰度图")
    args = ap.parse_args()

    raw = C.load_rgb(args.image)
    R = raw[..., 0]

    out = np.stack([R, R, R], axis=-1)

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_05_仅红通道R_G置0B置0_转灰度.png"

    C.save_rgb_u8(out.astype(np.float64) / 255.0, path)

    print(f"输入 {args.image.name}")
    print(f"  R 转灰度: 均值 {R.mean():.1f} 标准差 {R.std():.1f} CV {100*R.std()/R.mean():.1f}%")
    print(f"  -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

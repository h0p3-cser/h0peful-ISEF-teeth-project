#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 编号 02 / 共 15          单通道分离：绿色通道的灰度版
# -----------------------------------------------------------------------------
# 产物文件名   S0_02_仅绿通道G_R置0B置0_转灰度.png
# 处理对象     绿色通道 G
# 处理公式     out(x,y) = ( G(x,y) , G(x,y) , G(x,y) )
# RGB 处理     R := G    G := G    B := G      （三通道都写绿通道的值）
# 输入         任意 RGB 图片
# 输出         三通道 PNG，但三个通道完全相等，所以是真正的灰度图
# 用途         **做亮度/强弱比较用这张**。01 号是纯绿图，人眼对绿色亮度
#              的感知与灰度值不成正比；转成灰度后屏幕的亮度响应才和数值对齐
# 与 01 的差别  数据完全相同，只是把三通道都写成 G。像素值一个没改
# 运行         python3 02_仅绿通道_转灰度.py 图片.jpg
# 依赖         numpy, pillow
# =============================================================================

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> int:
    ap = C.cli_argparser("把绿色通道复制到三通道，得到灰度图")
    args = ap.parse_args()

    raw = C.load_rgb(args.image)
    G = raw[..., 1]

    out = np.stack([G, G, G], axis=-1)     # 三通道都等于 G

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)
    st = C.state_prefix(args.image.stem)
    path = outdir / f"{st}_02_仅绿通道G_R置0B置0_转灰度.png"

    C.save_rgb_u8(out.astype(np.float64) / 255.0, path)

    print(f"输入 {args.image.name}")
    print(f"  G 转灰度: 均值 {G.mean():.1f} 标准差 {G.std():.1f} CV {100*G.std()/G.mean():.1f}%")
    print(f"  已写出 -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

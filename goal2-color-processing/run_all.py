#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# run_all.py              一键按顺序跑完 01~15
# -----------------------------------------------------------------------------
# 作用         依次调用同目录下的 15 个脚本，输出到同一个目录，
#              结果与 荧光分析/src/ 里两个大脚本完全一致（同参数同顺序）
# 用法         python3 run_all.py 图片.jpg
#              python3 run_all.py 图片.jpg -o out/S0_baseline
#              python3 run_all.py 图片.jpg --power 3        （只影响 09/10/12）
# 实现说明     用 subprocess 调脚本而不是 import：
#              每个脚本都是独立可运行的单元，用同一套命令行跑，
#              能顺带验证"单跑"和"批量跑"结果一致
# 依赖         同 01~15
# =============================================================================

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (脚本文件名, 额外参数是否可传 --power)
STEPS = [
    ("01_仅绿通道_G保留_R0B0.py", False),
    ("02_仅绿通道_转灰度.py", False),
    ("03_绿通道直方图.py", False),
    ("04_仅红通道_R保留_G0B0.py", False),
    ("05_仅红通道_转灰度.py", False),
    ("06_红通道直方图.py", False),
    ("07_比值图_log2_R比G.py", False),
    ("08_红色过量_R减aG加c.py", False),
    ("09_绿光加权_去光照_幂律_色度.py", True),
    ("10_红光加权_去光照_幂律_色度.py", True),
    ("11_原图彩色_乘色度权重.py", False),
    ("12_总览九宫格.py", True),
    ("13_统计图版.py", False),
    ("14_分析报告.py", False),
    ("15_分析数据json.py", False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="按顺序跑完 01~15")
    ap.add_argument("image", type=Path, help="输入图片路径")
    ap.add_argument("--outdir", "-o", type=Path, default=None, help="输出目录")
    ap.add_argument("--power", type=float, default=2.0, help="幂律指数（传给 09/10/12）")
    args = ap.parse_args()

    if not args.image.exists():
        print(f"找不到文件: {args.image}", file=sys.stderr)
        return 2

    outdir = args.outdir or args.image.parent
    outdir.mkdir(parents=True, exist_ok=True)

    for i, (script, uses_power) in enumerate(STEPS, 1):
        cmd = [sys.executable, str(HERE / script), str(args.image), "-o", str(outdir)]
        if uses_power:
            cmd += ["--power", str(args.power)]
        print(f"\n[{i:02d}/15] {script}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:], file=sys.stderr)
            print(f"!! 第 {i} 步失败，已停止", file=sys.stderr)
            return 1
        # 只回显脚本自己的一两句摘要
        for line in r.stdout.strip().splitlines()[-3:]:
            print("   " + line)

    print(f"\n全部完成 -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

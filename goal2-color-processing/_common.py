# -*- coding: utf-8 -*-
"""_common.py —— 公共小工具，被 01~15 号脚本 import，本身不单独运行。

为什么不把这些代码复制进每个脚本：
    加载图片、按 uint8 存盘、百分位归一、从文件名取状态号（S0/S2）……
    这些和"处理方法"无关，集中一份能保证 15 个脚本对同一个输入
    用的是完全相同的约定（同一组百分位、同样的存盘方式）。

真正的处理公式**全部写在各自的 0x_*.py 里**，一点没藏在这里。

依赖: numpy, pillow
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    """读图并统一转成 uint8 的 (H, W, 3) RGB 数组（灰度图会自动复制到三通道）。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to_u8(v01: np.ndarray) -> np.ndarray:
    """0~1 浮点 -> uint8（四舍五入，不是截断）。"""
    return (np.clip(v01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_gray_u8(plane01: np.ndarray, path: Path) -> Path:
    """把 0~1 的单通道浮点图存成三通道灰度 PNG（R=G=B）。"""
    g = to_u8(plane01)
    Image.fromarray(np.stack([g, g, g], axis=-1)).save(path)
    return path


def save_rgb_u8(arr01: np.ndarray, path: Path) -> Path:
    """把 (H,W,3) 的 0~1 浮点彩色图存成 PNG。"""
    Image.fromarray(to_u8(arr01)).save(path)
    return path


# --------------------------------------------------------------------------
# 命名
# --------------------------------------------------------------------------
def state_prefix(stem: str) -> str:
    """从 T001_S2_C01_A000_QLF_I0274 里取出 S2；取不到就退回整个文件名。

    两个样本的产物都放在自己的文件夹里，文件名再带 S0/S2 前缀，
    这样从任何地方单独拷走一张，也知道它属于哪个处理阶段。
    """
    m = re.search(r"_(S\d+)_", stem)
    return m.group(1) if m else stem


def ascii_safe(text: str) -> str:
    """图标题只用 ASCII：matplotlib 默认字体没有中文，会画成方块。"""
    out = "".join(ch if ord(ch) < 128 else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "image"


# --------------------------------------------------------------------------
# 归一与统计
# --------------------------------------------------------------------------
def percentile_norm(plane: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """按百分位线性拉到 0~1。用百分位而不是 min/max，是为了忽略个别坏点。"""
    a, b = np.percentile(plane, [lo, hi])
    if b <= a:
        b = a + 1e-9
    return np.clip((plane - a) / (b - a), 0.0, 1.0)


def scale_by(plane: np.ndarray, hi_pct: float = 99.5) -> np.ndarray:
    """只按高端归一（除以 p99.5 并截断到 1），**不抬暗部**。

    区别很关键：percentile_norm 会把最暗的 0.5% 抬到 0，
    权重小的区域会被一起抬亮；scale_by 保留"弱的就该是黑的"。
    """
    hi = np.percentile(plane, hi_pct)
    if hi <= 0:
        hi = 1e-9
    return np.clip(plane / hi, 0.0, 1.0)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """一维 Otsu，在两峰直方图上自动找分界（这里用于 log2(R/G) 的双峰）。"""
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.0
    csum = np.cumsum(hist)
    w0 = csum / total
    w1 = 1.0 - w0
    m0 = np.cumsum(hist * centers) / np.maximum(csum, 1)
    mt = (hist * centers).sum() / total
    m1 = (mt - np.cumsum(hist * centers) / total) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = 0
    return float(centers[int(np.argmax(between))])


def describe(plane: np.ndarray) -> dict:
    """均值/标准差/变异系数/五个分位点/过曝比例——报告和 json 都用这一份。"""
    mean = float(plane.mean())
    std = float(plane.std())
    p = np.percentile(plane, [1, 5, 50, 95, 99])
    return {
        "mean": mean,
        "std": std,
        "cv_pct": 100.0 * std / mean if mean else float("nan"),
        "p01": float(p[0]),
        "p05": float(p[1]),
        "median": float(p[2]),
        "p95": float(p[3]),
        "p99": float(p[4]),
        "range_1_99": float(p[4] - p[0]),
        "saturated_frac": float((plane >= 254).mean()),
    }


# --------------------------------------------------------------------------
# 光照归一化
# --------------------------------------------------------------------------
def default_sigma(shape: tuple[int, int]) -> float:
    """大尺度模糊的默认半径：取长边的 1/12，能盖住打光渐变但盖不住牙面细节。"""
    return max(shape) / 12.0


def illumination_normalized(plane: np.ndarray, sigma: float) -> np.ndarray:
    """F = I / blur(I)。

    除以自己的大尺度模糊 ≈ 除掉"这一块被照得多亮"，
    剩下的才是这一块本身的反射/荧光差异，输出以 1.0 为基准。
    """
    from scipy import ndimage as ndi

    bg = np.maximum(ndi.gaussian_filter(plane, sigma=sigma), 1.0)
    return plane / bg


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def cli_argparser(description: str):
    """所有脚本统一的最小命令行：一个输入图 + 可选的输出目录。

    每个脚本都自己调这个，所以 `python3 0x_xxx.py 图.jpg -o 目录` 到处都能用。
    """
    import argparse

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("image", type=Path, help="输入图片路径")
    ap.add_argument("--outdir", "-o", type=Path, default=None,
                    help="输出目录，默认与输入图同目录")
    return ap

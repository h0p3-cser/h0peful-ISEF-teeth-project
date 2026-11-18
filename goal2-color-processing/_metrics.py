# -*- coding: utf-8 -*-
"""_metrics.py —— 统计内核，被 14 号（报告）和 15 号（json）共用。

为什么单独一份：
    14 和 15 必须给出**完全一样的数字**，否则报告和 json 会对不上。
    把计算集中在这里，两边只负责"排版"，就不会各算一套。
    真正和图像处理有关的公式仍然写在 0x_*.py 里，这里只做测量。

依赖: numpy, scipy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import _common as C


def compute(image: Path, sigma: float | None = None, smooth: float = 2.0) -> dict:
    """跑一遍全部测量，返回一个扁平字典（含中间数组，供下游取用）。"""
    raw = C.load_rgb(image).astype(np.float64)
    R, G, B = raw[..., 0], raw[..., 1], raw[..., 2]
    sigma = sigma or C.default_sigma(R.shape)

    Rs = ndi.gaussian_filter(R, smooth) if smooth > 0 else R
    Gs = ndi.gaussian_filter(G, smooth) if smooth > 0 else G

    # 原始统计
    dR, dG, dB = C.describe(R), C.describe(G), C.describe(B)
    corr = float(np.corrcoef(R.ravel(), G.ravel())[0, 1])

    # 比值
    r_g = (Rs + 1.0) / (Gs + 1.0)
    log_ratio = np.log2(r_g)
    thr = C.otsu_threshold(log_ratio.ravel())
    mask_red = log_ratio > thr
    frac_red = float(mask_red.mean())

    # 去光照后的起伏
    Rn = C.illumination_normalized(R, sigma)
    Gn = C.illumination_normalized(G, sigma)
    rn, gn = C.describe(Rn), C.describe(Gn)
    fluo_r_pct = 100.0 * rn["std"] / rn["mean"]
    fluo_g_pct = 100.0 * gn["std"] / gn["mean"]

    # 高频纹理
    hp_r = (R - ndi.gaussian_filter(R, 3)).std()
    hp_g = (G - ndi.gaussian_filter(G, 3)).std()

    # 线性耦合律
    A = np.vstack([G.ravel(), np.ones(G.size)]).T
    (a, c), *_ = np.linalg.lstsq(A, R.ravel(), rcond=None)
    pred = a * G + c
    r2 = float(1.0 - ((R - pred) ** 2).sum() / ((R - R.mean()) ** 2).sum())
    red_excess = Rs - (a * Gs + c)
    re_p1, re_p99 = np.percentile(red_excess, [0.5, 99.5])

    return {
        "image": image.name,
        "shape": list(raw.shape),
        "sigma": sigma,
        "smooth": smooth,
        "R": dR, "G": dG, "B": dB,
        "corr_RG": corr,
        "R_norm_std_pct": fluo_r_pct,
        "G_norm_std_pct": fluo_g_pct,
        "G_over_R_variation": fluo_g_pct / fluo_r_pct,
        "hp_std_R": float(hp_r),
        "hp_std_G": float(hp_g),
        "ratio_median": float(np.median(r_g)),
        "ratio_q25": float(np.percentile(r_g, 25)),
        "ratio_q75": float(np.percentile(r_g, 75)),
        "otsu_log2": thr,
        "red_shifted_area_pct": frac_red * 100,
        "G_mean_red_area": float(G[mask_red].mean()),
        "G_mean_neutral_area": float(G[~mask_red].mean()),
        "R_mean_red_area": float(R[mask_red].mean()),
        "R_mean_neutral_area": float(R[~mask_red].mean()),
        "coupling_law": {"a": float(a), "c": float(c), "r2": r2},
        "red_excess_std": float(red_excess.std()),
        "red_excess_p05": float(re_p1),
        "red_excess_p995": float(re_p99),
        # 下面几个是数组，给调用方取用，不写进 json
        "_raw": raw, "_R": R, "_G": G,
        "_log_ratio": log_ratio, "_mask_red": mask_red,
        "_red_excess": red_excess,
    }

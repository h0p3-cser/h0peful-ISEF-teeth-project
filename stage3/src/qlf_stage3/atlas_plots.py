"""Render one S0-referenced QLF atlas page."""
import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont
import numpy as np


def atlas_font():
    for family in ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS"):
        try:
            return FontProperties(fname=findfont(family, fallback_to_default=False))
        except ValueError:
            continue
    return FontProperties(family="DejaVu Sans")


FONT = atlas_font()
INK = "#183740"
SHADE = "#425964"
COLOR = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#34373b")
NORM = Normalize(-100, 100)


def draw_view(tooth, angle, photos, losses, detail):
    fig = plt.figure(figsize=(13.4, 8.4), facecolor="white")
    fig.text(.055, .928, f"{tooth}   /   {angle}°", color=INK, fontsize=26, weight="bold")
    fig.text(.055, .888, "同牙同角度 · 以 S0 为基线的牙冠荧光损失", color=SHADE,
             fontsize=13, fontproperties=FONT)
    photos_grid = fig.add_gridspec(1, 3, left=.055, right=.945, top=.825, bottom=.554, wspace=.10)
    for spec, image, loss, label in zip(photos_grid, photos, (True, *losses),
                                       ("S0  /  0 h", "S1  /  24 或 35 h", "S2  /  72 h"),
                                   strict=True):
        ax = fig.add_subplot(spec)
        ax.set_axis_off()
        if image is None:
            ax.text(.5, .5, "无可用于损失图的配对", ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color=SHADE, fontproperties=FONT,
                    bbox={"facecolor": "#edf1f3", "edgecolor": "none", "pad": 11})
        else:
            ax.imshow(image, interpolation="nearest")
        if loss is None and image is not None:
            spec_center = (spec.get_position(fig).x0 + spec.get_position(fig).x1) / 2
            fig.text(spec_center, .535, "原图裁切，仅供查看", ha="center", fontsize=9.5,
                     color=SHADE, fontproperties=FONT)
        ax.set_title(label, fontsize=12, color=INK, pad=7, fontproperties=FONT)

    loss_grid = fig.add_gridspec(1, 2, left=.06, right=.94, top=.474, bottom=.176, wspace=.09)
    for spec, loss, note, label in zip(loss_grid, losses, detail, ("S0 → S1", "S0 → S2"), strict=True):
        ax = fig.add_subplot(spec)
        ax.set_axis_off()
        ax.set_title(label, fontsize=13, color=INK, pad=7)
        if loss is None:
            ax.text(.5, .5, "未通过配对质检", ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color=SHADE, fontproperties=FONT,
                    bbox={"facecolor": "#edf1f3", "edgecolor": "none", "pad": 13})
        else:
            ax.imshow(np.ma.masked_invalid(loss), cmap=COLOR, norm=NORM, interpolation="nearest")
        midpoint = (spec.get_position(fig).x0 + spec.get_position(fig).x1) / 2
        fig.text(midpoint, .151, note, ha="center", va="center", color=SHADE, fontsize=10, fontproperties=FONT)

    color_axis = fig.add_axes([.292, .096, .416, .017])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=COLOR), cax=color_axis,
                            orientation="horizontal", extend="min", ticks=(-100, -50, 0, 50, 100))
    colorbar.ax.set_xticklabels(["≤−100%", "−50%", "0%", "+50%", "+100%"])
    colorbar.ax.tick_params(labelsize=9, pad=2, length=3, colors=INK)
    fig.text(.5, .037, "蓝色变亮 · 红色变暗 · 深灰为排除像素   |   当前 JPEG 含分阶段亮度调整",
             ha="center", color=SHADE, fontproperties=FONT, fontsize=10.5)
    return fig

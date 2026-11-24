"""Render aligned QLF images, a signed loss map, and fixed-threshold masks."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np

NORM = TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)
RESPONSE = "#d54d31"
VALID = "#dedfdd"
EXCLUDED = "#a3aaad"
BACKGROUND = "#fbfcfc"
INK = "#14282e"


def use_chinese_font():
    for name in ("PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = name
        break
    plt.rcParams["axes.unicode_minus"] = False


def render(destination, title, base_rgb, current_rgb, loss, thresholds, fractions, *,
           headings=("S0  基线", "S2  配准后"), subtitle="同一配准网格上的牙冠绿通道损失"):
    use_chinese_font()
    valid = np.isfinite(loss)
    background = np.max(current_rgb, axis=2) / 255
    context = 0.91 - 0.27 * background
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 8.9), facecolor=BACKGROUND)
    fig.subplots_adjust(left=.04, right=.96, top=.848, bottom=.165, wspace=.15, hspace=.30)
    fig.suptitle(title, x=.04, y=.957, ha="left", fontsize=21, color=INK, weight="semibold")
    fig.text(.04, .899, subtitle, ha="left", fontsize=11.2, color="#466571")
    panels = axes.ravel()
    for ax in panels:
        ax.set_axis_off()
    for ax, pixels, heading in zip(panels[:2], (base_rgb, current_rgb), headings, strict=True):
        ax.imshow(pixels, interpolation="nearest")
        ax.set_title(heading, fontsize=12, color=INK, pad=8)
    heat = panels[2]
    heat.imshow(context, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    overlay = heat.imshow(np.ma.masked_invalid(loss), cmap="RdBu_r", norm=NORM,
                          interpolation="nearest")
    heat.set_title("逐像素相对损失", fontsize=12, color=INK, pad=8)
    validity = matplotlib.colors.to_rgb(VALID)
    excluded = matplotlib.colors.to_rgb(EXCLUDED)
    response = matplotlib.colors.to_rgb(RESPONSE)
    mask_canvas = np.where(valid[..., None], validity, excluded)
    background_color = np.asarray(matplotlib.colors.to_rgb(BACKGROUND))
    mask_canvas = np.where(((background < .07) & ~valid)[..., None], background_color, mask_canvas)
    for ax, threshold, fraction in zip(panels[3:], thresholds, fractions, strict=True):
        panel = np.where((loss > threshold)[..., None], response, mask_canvas)
        ax.imshow(panel, interpolation="nearest")
        ax.set_title(f"损失 > {threshold}%  ·  面积 {fraction:.1f}%", fontsize=12, color=INK, pad=8)

    cax = fig.add_axes([.06, .086, .38, .018])
    cb = fig.colorbar(overlay, cax=cax, orientation="horizontal", extend="both",
                      ticks=(-80,-40,0,40,80))
    cb.ax.tick_params(labelsize=8, pad=1)
    cb.set_label("相对损失（%）  蓝色变亮 · 红色变暗", fontsize=9.5, color=INK)
    fig.legend(handles=[
        Patch(facecolor=RESPONSE, label="超过阈值"),
        Patch(facecolor=VALID, label="有效且未超过阈值"),
        Patch(facecolor=EXCLUDED, label="排除区域"),
    ], loc="lower left", bbox_to_anchor=(.51,.061), ncol=3, frameon=False, fontsize=9.5)
    fig.savefig(destination, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)

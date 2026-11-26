"""Append grayscale context without changing the pre-QC score artifact."""
from ..files import load
from .inputs import CALIBRATED


def write_audit(out, measured, summary, case, reference):
    scored = load(out/'scored.json')
    lines = ['## 原分辨率灰度一致性','',
        f'模型分数已于 {scored["completed"]} 保存。以下离线检验使用同一原图坐标，在缩放和补边之前，分别对原始白光和分支 PNG 调用原项目的灰度函数。差值单位为 8 位灰度级（0–255）。','',
        '每帧先计算绝对差的均值（MAE）和像素 P95。表中 MAE 对可用帧等权平均，P95 为各帧 P95 的中位数，括号内是逐帧范围。最后一列另给合并全部可用像素的 P95。逐帧原值见 gray_by_frame.csv。','',
        '| 处理 | 区域 | 可用帧 / 本类帧 | 平均 MAE（范围） | 帧 P95 中位数（范围） | 合并像素 P95 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summary.itertuples():
        label = '成功校偏' if row.status==CALIBRATED else '原图直通'
        area = '整图' if row.region=='whole' else '有效牙冠'
        if row.available_frames:
            lines.append(f'| {label} | {area} | {row.available_frames}/{row.total_frames} | {row.frame_mean_mae:.4f}（{row.min_frame_mae:.4f}–{row.max_frame_mae:.4f}） | {row.frame_median_p95:g}（{row.min_frame_p95:g}–{row.max_frame_p95:g}） | {row.pooled_pixel_p95:g} |')
        else:
            lines.append(f'| {label} | {area} | 0/{row.total_frames} | — | — | — |')
    reasons = measured.crown_status.value_counts()
    lines.extend(['', f'牙冠取自既有的白光牙冠轮廓、通过检查的配准变换及同一张白光的牙体掩膜。'
        f'其中 {reasons.get("verified_historical_pixel_count",0)} 帧重建后的像素数与历史报告一致，'
        f'{reasons.get("accepted_historical_geometry",0)} 帧有合格配准，但旧校色步骤没有留下牙冠面积记录。'
        f'其余 {len(measured)-reasons.get("verified_historical_pixel_count",0)-reasons.get("accepted_historical_geometry",0)} 帧只评价整图，均保留原有模型预测。',''])
    sub = summary[(summary.status==CALIBRATED)&(summary.region=='crown')]
    if len(sub) and sub.iloc[0].available_frames:
        entry = sub.iloc[0]
        lines.extend([f'在 {int(entry.available_frames)} 张可测的校偏牙冠上，相对原图的平均灰度绝对差为 {entry.frame_mean_mae:.4f} 级，逐像素最大绝对差为 {entry.pooled_max_abs:.0f} 级。'
             'CIELAB 运算保持 L* 和 a*，转回 8 位 RGB 再作项目灰度转换后仍可能改变亮度值。与历史原图灰度比较时，应把上述明暗变化计入解释。两模型使用同一分支图配对，前面保存的分数和样本均未改动。',''])
    if case is not None:
        lines += ['### I0654 同位置对照','',
            '原生裁片采用左上至右下半开坐标 [889, 880, 1305, 1296)，共 416 × 416 像素。青框为棕黑纹，橙框为邻近牙面。两个采样框在所有图中位置和尺度相同。','',
            '| 版本 | 棕黑纹灰度中位数 | 邻近牙面灰度中位数 | 牙面减棕黑纹 |','| --- | ---: | ---: | ---: |',
            f'| 原图 | {case["original_stripe_median_gray"]:g} | {case["original_enamel_median_gray"]:g} | {case["original_contrast"]:g} |',
            f'| 分支输出 | {case["branch_stripe_median_gray"]:g} | {case["branch_enamel_median_gray"]:g} | {case["branch_contrast"]:g} |','',
            f'该帧处理状态为 {case["status"]}，棕黑纹对比变化 {case["contrast_change"]:+g} 级。','',
            '![I0654 同位置白光与灰度对照](I0654_same_position.png)','']
    else:
        lines.extend(['I0654 源图未进入配对清单，无法生成其原生裁片。',''])
    if (out/'reference_samples.png').exists():
        lines.extend(['### 实际参照位置','',
            f'在相同的有效取样像素上独立读取原图和导出 PNG，参照块 b* 中位数从 '
            f'{reference.original_median_bstar.min():.3f} 至 {reference.original_median_bstar.max():.3f}，'
            f'变为 {reference.png_median_bstar.min():.3f} 至 {reference.png_median_bstar.max():.3f}。'
            '这是取样面向工程目标零点的接近程度。橙框标出原图位置，每帧的复算值见 reference_bstar.csv。','',
            '![实际取样的哑光白色底面](reference_samples.png)',''])
    md = '\n'.join(lines)
    (out/'gray_consistency.md').write_text('# 白光分支灰度核验\n\n' + md,encoding='utf8')
    original = (out/'model_scores.md').read_text(encoding='utf8')
    (out/'report.md').write_text(original.rstrip() + '\n\n' + md + '\n', encoding='utf8')

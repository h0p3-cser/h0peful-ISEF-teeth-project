"""Paired scores frozen before opening native grayscale QC."""
import pandas as pd


def write_scores(out, scores, teeth, confusion):
    result = scores.set_index('arm')
    delta = result.at['rgb','tooth_equal_macro_f1'] - result.at['gray','tooth_equal_macro_f1']
    coverage = pd.read_csv(out / 'coverage.csv')
    frames = pd.read_csv(out / 'frame_manifest.csv')
    active = frames[frames.paired_included]
    received = active[active.status.eq('corrected_bstar')]
    lines = ['# 白光 RGB 黄蓝校偏与同源灰度的配对结果','',
        f'共 {len(active)} 帧、{active.group_id.nunique()} 个采集组、{active.tooth_id.nunique()} 颗牙。每组仅预测一次。主指标为每颗牙的 S0、S1、S2 Macro-F1，再对牙等权平均。','',
        '| 输入 | 牙等权 Macro-F1 | S1 召回（合并采集组） | 九牙等权 S1 召回 | 全部组判对 |',
        '| --- | ---: | ---: | ---: | ---: |']
    names = {'rgb':'分支 RGB','gray':'同一张分支图转灰度'}
    for r in scores.itertuples():
        lines.append(f'| {names[r.arm]} | {r.tooth_equal_macro_f1:.6f} | {r.s1_correct}/{r.s1_groups}（{100*r.s1_recall:.2f}%） | {100*r.tooth_equal_s1_recall:.2f}% | {r.correct}/{r.groups} |')
    pos = (teeth.rgb_minus_gray_macro_f1>0).sum()
    neg = (teeth.rgb_minus_gray_macro_f1<0).sum()
    lines.extend(['',f'RGB 减同源灰度的主指标差为 {delta:+.6f}。逐牙有 {pos} 颗提高、{neg} 颗下降、{len(teeth)-pos-neg} 颗相同。','',
        '| 阶段 | 成功校偏 | 原图直通 | 源图缺失 | 配对入模 |','| --- | ---: | ---: | ---: | ---: |'])
    for r in coverage.itertuples():
        lines.append(f'| {r.state_id} | {r.corrected_bstar} | {r.passthrough_no_reference} | {r.excluded_missing_source} | {r.paired_frames} |')
    lines.extend(['',f'有参照的 {len(received)} 帧来自 {received.group_id.nunique()} 个组。拍摄未放置标准灰卡。选区依据为牙外可辨认的漫反射白／灰实物。每张图用自身有效参照像素估计 b*，不同设备与阶段互不借值。逐帧清单保留了选区、参照像素数、偏移和直通原因。','',
        '## 逐牙结果','','| 留出牙 | 组数 | RGB Macro-F1 | 同源灰度 Macro-F1 | 配对差 |','| --- | ---: | ---: | ---: | ---: |'])
    for r in teeth.itertuples():
        lines.append(f'| {r.tooth_id} | {r.groups} | {r.rgb_macro_f1:.4f} | {r.gray_macro_f1:.4f} | {r.rgb_minus_gray_macro_f1:+.4f} |')
    for arm in names:
        lines.extend(['',f'## {names[arm]}混淆矩阵','',
            '行是真实阶段，列是预测阶段。括号为本行真实采集组中的百分比。','',
            '| 真实阶段 | 预测 S0 | 预测 S1 | 预测 S2 |','| --- | ---: | ---: | ---: |'])
        for state, sub in confusion[confusion.arm.eq(arm)].groupby('true_state',sort=True):
            cells = [f'{r.groups}（{r.pct_within_true_state:.1f}%）' for r in sub.sort_values('pred_state').itertuples()]
            lines.append('| ' + ' | '.join([state] + cells) + ' |')
    lines.extend(['','## 解释与复现','',
        '两组共享每一张已导出的原分辨率 PNG。彩色输入保留三个通道，配对灰度用原项目的 EXIF 定向与 Pillow L 灰度转换再复制成三通道。全图等比双线性缩放、居中黑边至 224×224。相同的冻结 ImageNet ResNet-18 提取 512 维特征。每组先求连拍特征均值，每折分别在八颗训练牙上拟合四维 PCA、标准化器和多项 L2 逻辑回归。每颗留出牙只用于该折预测。','',
        'S2 与 C04 在九牙数据中共现，且输入保留完整背景。配对差反映这批整图中明暗、纹理之外可用的颜色信息，可能包含牙冠、背景、装置或批次因素。参照有限，不能以本比较估计校偏本身的增益，也不能证明牙冠颜色独立贡献。历史灰度白光约 0.704，保留作背景。','',
        '在 stage3 源码目录运行以下命令。','',
        '```sh','python experiments/rgb_bstar_branch.py --history runs/study/evaluation --crown-run runs/rgb_full --out runs/bstar','```',''])
    (out/'model_scores.md').write_text('\n'.join(lines), encoding='utf-8')

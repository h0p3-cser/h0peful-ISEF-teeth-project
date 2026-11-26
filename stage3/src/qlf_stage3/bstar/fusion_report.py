"""Report the three paired models, correction coverage and original-RGB control."""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from ..files import load, relative_path, sha
from ..modeling.cohort import ensure, save_csv
from ..modeling.evaluation import read_result
from ..modeling.report import md_table
from ..rgb.color import decode
from .inputs import CALIBRATED, PASS


def reference_examples(out,frames):
    same_view = frames[frames.tooth_id.eq('T008') & frames.angle_id.eq('A000')]
    selected = []
    for state in ['S0','S1','S2']:
        group = same_view[same_view.state_id.eq(state)].sort_values('repeat_rank')
        if not group.empty:
            selected.append(group.iloc[len(group)//2])
    other = frames[frames.rgb_image_id.eq('I0654')]
    if not other.empty:
        selected.insert(1,other.iloc[0])
    if not selected:
        return
    fig, axes = plt.subplots(2,len(selected),squeeze=False,figsize=(3.5*len(selected),4.5),layout='constrained')
    reasons = {'external_white_backing':'白色哑光底面，橙框为取样区',
               'unresolved_upper_strip':'上沿材料本色及受光难以判断','dark_carrier_only':'深色底板／托盘，缺少中性参照'}
    for j,row in enumerate(selected):
        image = Image.fromarray(decode(out/row.raw_rgb_file)[0])
        w,h = image.size
        x0,y0 = w-256,0
        top,bottom = axes[0,j],axes[1,j]
        top.imshow(image.resize((512,round(512*h/w)),Image.Resampling.LANCZOS),extent=(0,w,h,0))
        top.add_patch(Rectangle((x0,y0),256,144,fill=False,edgecolor='#35c7bf',lw=1.2))
        top.set_title(f'{row.tooth_id}   {row.state_id}   {int(row.angle_id[1:])}°\n{row.rgb_image_id}',fontsize=10)
        bottom.imshow(image.crop((x0,y0,w,144)),extent=(x0,w,144,y0),interpolation='nearest')
        for region in json.loads(row.regions):
            x,y,u,v=region['box']
            bottom.add_patch(Rectangle((x,y),u-x,v-y,fill=False,edgecolor='#ffa81e',lw=2))
        bottom.set_xlabel(reasons.get(row.reason,row.reason),fontsize=9)
        for ax in (top,bottom):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0,0].set_ylabel('原始白光整图')
    axes[1,0].set_ylabel('右上角同位置')
    fig.savefig(out/'figures/06_reference_examples.png',dpi=155,bbox_inches='tight')
    plt.close(fig)


def write_report(out,branch,primary,scores,differences):
    rows = primary['manifest']
    reference_examples(out,rows)
    gray = read_result(branch,'gray_summary.csv').set_index(['status','region'])
    complete = load(branch/'complete.json')
    ensure(sha(branch/'gray_by_frame.csv')==complete['gray_by_frame_sha256'],'Archived grayscale QC differs')
    neutral = read_result(branch,'reference_bstar.csv')
    coverage = rows.groupby(['state_id','status','reason'],as_index=False).agg(
        frames=('sample_id','size'),groups=('group_id','nunique'))
    save_csv(out,'coverage_reasons.csv',coverage)
    eligible = rows[rows.status.eq(CALIBRATED)]
    other = rows[rows.status.eq(PASS)]
    c = scores.set_index(['white_policy','model'])
    diff = differences.set_index('comparison')
    names = {'M1':'M1 彩色白光','M2':'M2 QLF 四特征','M3':'M3 RGB＋QLF'}
    primaries = scores[scores.white_policy.eq('branch_rgb')]
    models = md_table(['模型','牙等权 Macro-F1（95% 描述性区间）','S1 召回','S1 判为 S0 / S2','跨两级错误','全部组判对'],
        [[names[r.model],f'{r.tooth_equal_macro_f1:.4f}（{r.tooth_bootstrap_ci_low:.4f}–{r.tooth_bootstrap_ci_high:.4f}）',
          f'{r.s1_correct}/{r.s1_groups}（{100*r.s1_recall:.1f}%）',f'{r.s1_to_s0} / {r.s1_to_s2}',
          r.s0_s2_errors,f'{r.correct}/{r.groups}'] for r in primaries.itertuples()])
    lines=['# 终点三彩色白光、QLF 与融合的九牙配对比较','',
        f'{len(rows)} 对样本来自 {rows.tooth_id.nunique()} 颗牙，组内连拍先平均特征，共 {rows.group_id.nunique()} 个采集组。'
        '每折留出一整颗牙。下表的 Macro-F1 先逐牙计算三个阶段，再对九颗牙等权平均。'
        '括号区间从固定的折外结果按牙配对重采样计算；S1 召回按全部真实 S1 组汇总。','',models,'']
    for comparator,key in [('M1','M3_vs_RGB'),('M2','M3_vs_QLF')]:
        g=diff.loc[key]
        ci=primary['gains'].query('level=="equal_tooth_mean" and comparator==@comparator').iloc[0]
        lines.extend([f'融合相对 {names[comparator]} 的 Macro-F1 差为 {g.delta_tooth_equal_macro_f1:+.4f}，'
            f'配对描述性区间 {ci.ci_low_pp/100:+.4f} 至 {ci.ci_high_pp/100:+.4f}。逐牙 {int(g.improved_teeth)} 颗提高、'
            f'{int(g.tied_teeth)} 颗持平、{int(g.declined_teeth)} 颗下降。融合改对 {int(g.newly_correct)} 组，改错 {int(g.newly_wrong)} 组。',''])
    lines.extend(['![三个模型的混淆矩阵](figures/01_confusion.png)','',
                  '![逐牙三模型分数与融合增益](figures/02_tooth_gains.png)','',
                  '## 颜色信息与这七张校偏的作用',''])
    labels={'original_rgb':'所有照片用原始 RGB','branch_rgb':'有参照时校偏，其余原图直通','sibling_gray':'同一批分支 PNG 转灰度'}
    def visual_row(policy):
        a,b=[c.loc[(policy,m)] for m in ('M1','M3')]
        return [labels[policy],f'{a.tooth_equal_macro_f1:.4f}',f'{int(a.s1_correct)}/{int(a.s1_groups)}',
                f'{b.tooth_equal_macro_f1:.4f}',f'{int(b.s1_correct)}/{int(b.s1_groups)}']
    lines.extend([md_table(['白光版本','白光单模态 Macro-F1','白光 S1 判对','加 QLF 后 Macro-F1','融合 S1 判对'],
        [visual_row(k) for k in labels]),''])
    for key,label in [('correction_visual','单用 RGB'),('correction_fusion','RGB＋QLF')]:
        g=diff.loc[key]
        lines.extend([f'{label} 时，用这套含直通的策略替换原始 RGB，Macro-F1 改变 {g.delta_tooth_equal_macro_f1:+.4f}，'
          f'{int(g.changed_group_predictions)}/{len(primary["groups"])} 个组的最终判定发生变化。三类折外概率的最大绝对变化为 '
          f'{100*g.max_absolute_probability_delta:.3f} 个百分点。',''])
    lines.extend([f'在完全相同的分支图上保留 RGB 相对转为灰度，白光单模态的 Macro-F1 差为 '
        f'{diff.loc["color_visual","delta_tooth_equal_macro_f1"]:+.4f}，加入 QLF 后为 {diff.loc["color_fusion","delta_tooth_equal_macro_f1"]:+.4f}。'
        '所有白光版本的 QLF 数据与测试组一致。','',
        '## 校偏覆盖与灰度核验','',
        f'{len(eligible)}/{len(rows)} 帧（{100*len(eligible)/len(rows):.2f}%）来自 {eligible.group_id.nunique()}/{rows.group_id.nunique()} 个'
        '有可辨认中性参照的采集组。拍摄时没有设置标准灰卡。取样依据为牙体外、受光相近的可辨认白／灰哑光实物。', '',
        md_table(['阶段','可见参照／直通原因','帧数','采集组数'],[[r.state_id,{'external_white_backing':'带绿色标记的底面，取右端空白白面',
        'unresolved_upper_strip':'上沿的材料本色或受光难以确认','dark_carrier_only':'仅辨认出深色底板、托盘或反光'}.get(r.reason,r.reason),
        r.frames,r.groups] for r in coverage.itertuples()]),'',
        '![牙外参照的原图和同位置放大](figures/06_reference_examples.png)','',
        f'在已取样白面上，用相同像素复读原图和导出 PNG，b* 中位数由 {neutral.original_median_bstar.min():+.2f} 至 '
        f'{neutral.original_median_bstar.max():+.2f} 变为 {neutral.png_median_bstar.min():+.2f} 至 '
        f'{neutral.png_median_bstar.max():+.2f}。这些像素也用于估计偏移，证明该处达到了接近中性的目标。'
        '7 张为同一连拍，其他阶段缺少可用色标，单独评价整牙颜色是否更接近真实本色还缺独立参照。',''])
    whole=gray.loc[(CALIBRATED,'whole')]
    crown=gray.loc[(CALIBRATED,'crown')]
    passthrough=gray.loc[(PASS,'whole')]
    lines.extend([f'灰度一致性已在上轮两组白光分数保存后逐帧实测，单位为 0–255 灰度级。实际校偏的 {int(whole.available_frames)} 张整图'
        f'平均绝对差为 {whole.frame_mean_mae:.3f}，合并像素 P95 为 {whole.pooled_pixel_p95:g}，最大差为 {whole.pooled_max_abs:g}。'
        f'其中 {int(crown.available_frames)} 张有有效牙冠掩膜，牙冠平均绝对差 {crown.frame_mean_mae:.3f}，P95 与最大差均为 '
        f'{crown.pooled_pixel_p95:g} 级。{len(other)} 张原图直通的整图最大差为 {passthrough.pooled_max_abs:g}。'
        '这项核验衡量导出后明暗保持的程度。','',
        '## 输入与解释范围','',
        '冻结的 ResNet-18 读取 224×224 全幅等比补边图像。每个训练折在 8 颗训练牙的组均值上分别拟合四维 PCA 和标准化器。'
        'QLF 使用基于 255 参考的平均有符号损失、损失严格超过 30% 的面积占比、绿通道 P10 和纹理熵。M1 与 M2 各用四维，'
        'M3 拼接同折这两组特征后用八维。三模型使用相同的多项 L2 逻辑回归。','',
        'S2 与 C04 在九牙数据中共现，全幅白光颜色可能含有装置和背景信息。30% 阈值曾参考同批牙的阶段曲线。'
        '本报告描述固定方案在这九颗牙上的内部探索性结果。','',
        '## 可复查文件','',
        '[三模型组级概率、阶段及 QLF 实测值](paired_group_predictions.csv) · [逐牙配对差](paired_differences_by_tooth.csv) · '
        '[原始 RGB 与同源灰度对照](model_comparison.csv) · [全部版本的逐组概率](all_policy_group_predictions.csv)','',
        '[分阶段 ROC](figures/03_stage_roc.png) · [QLF 特征变化](figures/04_qlf_features.png) · [融合改对、改错的实拍样本](figures/05_fusion_cases.png)','',
        f'[原分辨率逐帧灰度核验]({relative_path(branch/"gray_by_frame.csv",out)}) · '
        f'[原图实物取样位置]({relative_path(branch/"reference_samples.png",out)}) · '
        f'[I0654 棕黑纹同位置对照]({relative_path(branch/"I0654_same_position.png",out)})','',
        '从 stage3 源码目录复现，白光分支和授权数据路径可另行指定。','',
        '```sh','python experiments/rgb_qlf_comparison.py --branch runs/bstar --rgb-root data/study/workspace/Data_rgb_corrected','```',''])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf8')

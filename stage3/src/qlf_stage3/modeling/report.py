"""Study figures and Chinese report, computed from the published plotting tables."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cohort import FEATURES, MODELS, STATES
from .vision import gray_input


COLORS = {'M1':'#4572a6', 'M2':'#ba6a36', 'M3':'#08796a'}
NAMES = {'M1':'M1 灰度白光', 'M2':'M2 QLF', 'M3':'M3 融合'}
FEATURE_NAMES = ['平均有符号荧光损失（%）','损失 >30% 的牙冠面积（%）','绿通道 P10（灰度级）','绿通道纹理熵']
FIGS = ['01_confusion', '02_tooth_gains', '03_stage_roc', '04_qlf_features', '05_fusion_cases']


def figures(out, r, dataset, *, names=None, photo_input=gray_input):
    names = NAMES if names is None else names
    folder = out / 'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','Noto Sans CJK SC','PingFang SC','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False, 'font.size':10, 'axes.spines.top': False,
                         'axes.spines.right':False, 'figure.facecolor':'white', 'axes.titleweight':'bold'})
    def finish(fig, number):
        fig.savefig(folder / (FIGS[number-1] + '.png'), dpi=190, bbox_inches='tight', pad_inches=.18)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12,3.5), layout='constrained')
    for ax, model in zip(axes, MODELS):
        sample = r['cms'].query('model == @model')
        counts = sample.pivot(index='true_state',columns='pred_state',values='groups').reindex(index=STATES, columns=STATES)
        pct = sample.pivot(index='true_state',columns='pred_state',values='row_pct').reindex(index=STATES, columns=STATES)
        im = ax.imshow(pct, vmin=0, vmax=100, cmap='Blues')
        for i,j in np.ndindex((3,3)):
            ax.text(j,i,f'{counts.iloc[i,j]}\n{pct.iloc[i,j]:.1f}%',ha='center',va='center',
                    color='white' if pct.iloc[i,j]>55 else '#14334c')
        ax.set(xticks=range(3), xticklabels=STATES, yticks=range(3), yticklabels=STATES,
               xlabel='预测阶段', ylabel='真实阶段', title=names[model])
    fig.colorbar(im, ax=axes, label='占本真实阶段的组数（%）', shrink=.8)
    finish(fig,1)

    scores = r['scores'].sort_index()
    fig, (left,right) = plt.subplots(1,2,figsize=(10,5.6),sharey=True)
    y = np.arange(len(scores))
    for offset,model in zip([-.19,0,.19],MODELS):
        left.scatter(100*scores[model],y+offset,s=48,label=names[model],color=COLORS[model])
    left.set(xlim=(-3,103), xlabel='三阶段 Macro-F1（%）',yticks=y,yticklabels=scores.index,title='每颗留出牙的得分')
    left.invert_yaxis()
    left.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=8,frameon=False)
    right.axvline(0,color='#626970',lw=1)
    for comp,offset,marker in [('M1',-.12,'o'),('M2',.12,'s')]:
        gains=r['gains'].query('comparator == @comp')
        values=gains[gains.level=='tooth'].set_index('tooth_id').reindex(scores.index).gain_pp
        right.hlines(y+offset,0,values,color=COLORS[comp],alpha=.45)
        right.scatter(values,y+offset,color=COLORS[comp],marker=marker,label=f'M3 − {comp}')
    right.set(xlabel='融合减单模态（百分点）',title='同牙配对差值')
    right.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,fontsize=9,frameon=False)
    for ax in (left,right): ax.grid(axis='y', alpha=.2)
    summary = r['gains'].query("level == 'equal_tooth_mean'")
    foot='     '.join(f'M3 − {g.comparator}   {g.gain_pp:+.1f}  [{g.ci_low_pp:+.1f}, {g.ci_high_pp:+.1f}]'
                      for g in summary.itertuples())
    fig.subplots_adjust(bottom=.25,wspace=.2)
    fig.text(.5,.02,'牙等权均值差与 95% 描述性区间（百分点）     '+foot,ha='center',fontsize=9)
    finish(fig,2)

    fig, axes = plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax, state in zip(axes,STATES):
        ax.plot([0,1],[0,1],'--',color='#92999e',linewidth=1)
        for model in MODELS:
            data=r['curves'].query('state_id == @state and model == @model')
            auc=r['stages'].query('state_id == @state and model == @model').auroc.iloc[0]
            ax.step(data.fpr,data.tpr,where='post',color=COLORS[model],label=f'{model}   {auc:.3f}')
        ax.set(xlim=(0,1),ylim=(0,1.01),aspect='equal',title=state+' 对其余阶段',xlabel='假阳性率',ylabel='真阳性率')
        ax.legend(title='组级 AUROC',fontsize=8,loc='lower right',frameon=False)
    finish(fig,3)

    fig, axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax, feature, label in zip(axes.flat,FEATURES,FEATURE_NAMES):
        data=r['features'].pivot(index='tooth_id',columns='state_id',values=feature).reindex(columns=STATES)
        for _,row in data.iterrows(): ax.plot(range(3),row,marker='o',alpha=.58,lw=1,ms=3,color='#56788d')
        ax.plot(range(3),data.median(),marker='o',lw=2.6,color=COLORS['M3'],label='逐牙中位数的中位数')
        ax.set(xticks=range(3),xticklabels=STATES,title=label,xlim=(-.15,2.15))
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8,frameon=False)
    finish(fig,4)

    fig=plt.figure(figsize=(12,7),layout='constrained')
    grid=fig.add_gridspec(2,4,height_ratios=[1,1.1])
    for j,comp in enumerate(['M1','M2']):
        ax=fig.add_subplot(grid[0,2*j:2*j+2])
        counts=r['changes'].query('comparator == @comp').pivot(index='true_state',columns='direction',values='groups').reindex(STATES)
        for offset,direction,label,color in [(-.19,'corrected','改对','#08796a'),(.19,'spoiled','改错','#b14b43')]:
            bars=ax.bar(np.arange(3)+offset,counts[direction],width=.34,label=label,color=color)
            ax.bar_label(bars,padding=2,fontsize=9)
        ax.set(xticks=range(3),xticklabels=STATES,ylabel='留出采集组数',title=f'融合相对 {names[comp]}',ylim=(0,r['changes'].groups.max()*1.23+1))
        ax.legend(frameon=False)
    for j,(comp,direction) in enumerate([('M1','corrected'),('M1','spoiled'),('M2','corrected'),('M2','spoiled')]):
        ax=fig.add_subplot(grid[1,j])
        case=r['cases'].query('comparator == @comp and direction == @direction')
        ax.set_axis_off()
        action='改对' if direction=='corrected' else '改错'
        if case.empty:
            ax.text(.5,.5,f'相对 {comp} {action} 0 组',ha='center',va='center')
            continue
        row=case.iloc[0]
        photo = photo_input(dataset / row.representative_rgb_path)
        cases_dir = folder / 'cases'
        cases_dir.mkdir(exist_ok=True)
        photo.save(cases_dir / (row.representative_rgb_id + '.png'))
        ax.imshow(photo)
        ax.set_title(f'相对 {comp} {action} · 真实 {row.state_id}',fontsize=10)
        ax.text(.5,-.04,f'{row.tooth_id}  {row.angle_id}  白光 {row.representative_rgb_id}\n'
                f'{comp} 判 {row[comp+"_pred_state"]}    M3 判 {row.M3_pred_state}',
                ha='center',va='top',transform=ax.transAxes,fontsize=8)
    finish(fig,5)


def md_table(headers, rows):
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    lines.extend('| '+' | '.join(str(x).replace('|',' / ') for x in row)+' |' for row in rows)
    return '\n'.join(lines)


def make_report(c, out, r, verification):
    figures(out, r, c['dataset'])
    study = c['kind'] == 'study'
    scope = '固定 30% 后的九牙内部探索性比较' if study else '合成数据软件演示，分数仅用于检验流程'
    data_name = 'study' if study else 'demo'
    m=r['summary'].set_index('model')
    stages=r['stages'].set_index(['model','state_id'])
    g=r['gains'].query("level == 'equal_tooth_mean'").set_index('comparator')
    manifest=r['manifest']
    active=manifest[manifest.included]
    groups=r['groups']
    n=len(groups)
    teeth=m.evaluable_teeth.iloc[0]
    def figline(i, caption, tables):
        return f'![图 {i}](figures/{FIGS[i-1]}.png)\n\n{caption}\n\n'+ ' · '.join(
            f'[{label}]({path}.csv)' for label,path in tables)
    sections=['# 灰度白光、QLF 与融合模型的留一牙验证',
              f'{scope}。逐图清单含 {len(manifest)} 帧，三模型共同使用 {len(active)} 对白光与 QLF，合计 {n} 个采集组。每颗牙仅在未参与训练的一折获得预测。',
              '## 三个模型的结果',
              md_table(['模型','牙等权 Macro-F1 [95% 描述性区间]','正确组 / 总组','准确率','平衡准确率','等级 MAE','宏 AUROC','Log loss'],
                [[NAMES[k],f'{v.primary_macro_f1:.3f} [{v.primary_ci_low:.3f}, {v.primary_ci_high:.3f}]',
                  f'{int(v.correct)} / {n}',f'{v.accuracy:.1%}',f'{v.balanced_accuracy:.1%}',f'{v.ordinal_mae:.3f}',
                  f'{v.macro_auroc:.3f}',f'{v.log_loss:.3f}'] for k,v in m.iterrows()]),
              f'主指标先在每颗留出牙内等权平均 S0、S1、S2 的 F1，再等权平均 {teeth} 颗牙。其余指标汇总 {n} 个折外采集组。']
    for comp, row in g.iterrows():
        sections.append(f'融合相对 {NAMES[comp]} 的平均差为 **{row.gain_pp:+.2f} 个百分点**，95% 描述性区间 '
                        f'[{row.ci_low_pp:+.2f}, {row.ci_high_pp:+.2f}]。逐牙改善 {int(row.improved)} 颗、持平 '
                        f'{int(row.tied)} 颗、下降 {int(row.declined)} 颗。')
    lower = r['gains'].query("level == 'tooth' and comparator == 'M2' and gain_pp < -1e-10")
    if g.loc['M1','gain_pp'] > 0 and g.loc['M2','gain_pp'] > 0:
        sections.append('在这批牙上，两种模态各自为另一种提供了增量。' +
            ('加入白光后得分下降的牙为 ' + '、'.join(
                f'{x.tooth_id}（{x.gain_pp:+.1f} 个百分点）' for x in lower.itertuples()) + '。' if len(lower) else ''))
    sections += ['区间来自 5,000 次整牙有放回抽样，种子为 20261124。三个模型每次共用同一序列，保留重复抽中的牙。区间仅描述当前牙样本在固定折外预测下的差异，九折重叠的训练集、重新拟合和选阈的不确定性仍存在。',
                 '## S1 识别与跨两级错误']
    def direction(model, true, predicted):
        row=stages.loc[model,true]
        return f"{int(row['to_'+predicted+'_n'])}/{int(row.true_groups)}（{row['to_'+predicted+'_pct']:.1f}%）"
    sections += [md_table(['模型','S1 召回','S1 → S0','S1 → S2','S0 → S2','S2 → S0','相邻 / 跨级错误'],
        [[NAMES[k],direction(k,'S1','S1'),direction(k,'S1','S0'),direction(k,'S1','S2'),
          direction(k,'S0','S2'),direction(k,'S2','S0'),f'{int(m.loc[k,"adjacent_errors"])} / {int(m.loc[k,"two_stage_errors"])}'] for k in MODELS])]
    s1=stages.loc['M3','S1']
    sections += [figline(1,f'融合在 {int(s1.true_groups)} 个真实 S1 组中找回 {int(s1.to_S1_n)} 个，另有 {int(s1.to_S0_n)} 个判为 S0、{int(s1.to_S2_n)} 个判为 S2。',[('矩阵底表','confusion_counts'),('各阶段 Precision、Recall、F1 与 AUROC','metrics_by_stage')])]
    diffs=r['gains'].query("level == 'tooth' and comparator == 'M1'").sort_values('gain_pp')
    sections += [figline(2,f'M3−M1 的逐牙差值从 {diffs.iloc[0].tooth_id} 的 {diffs.iloc[0].gain_pp:+.1f} 到 {diffs.iloc[-1].tooth_id} 的 {diffs.iloc[-1].gain_pp:+.1f} 个百分点。',[('逐牙得分','metrics_by_tooth'),('逐牙增益及区间','paired_gains')])]
    aucs=stages.loc['M3'].auroc
    sections += [figline(3,f'融合概率排序最困难的阶段为 {aucs.idxmin()}，其一对其余 AUROC 为 {aucs.min():.3f}，按最高概率分类的该阶段召回率为 {stages.loc["M3",aucs.idxmin()].recall:.1%}。',[('ROC 底表','roc_points')])]
    change=[]
    for k,label in zip(FEATURES,FEATURE_NAMES):
        pivot=r['features'].pivot(index='tooth_id',columns='state_id',values=k).reindex(columns=STATES)
        valid=pivot.dropna(subset=['S0','S2'])
        change.append(f'{label}有 {int((valid.S2>valid.S0).sum())}/{len(valid)} 颗牙在 S2 高于 S0')
    sections += [figline(4,'。'.join(change)+'。每个点是该牙该阶段跨采集组的中位数。',[('逐牙阶段中位数','feature_stage_medians'),('各组原始值','groups')])]
    changes=r['changes'].query("comparator == 'M1' and true_state == 'S1'").set_index('direction')
    sections += ['## 融合改对和改错的组',
        figline(5,f'对照灰度白光，融合在真实 S1 中改对 {changes.loc["corrected","groups"]} 组、改错 {changes.loc["spoiled","groups"]} 组。',
                [('逐阶段改对与改错','changes_by_stage'),('所有组的三个模型概率、原图和实测特征','paired_group_predictions'),('代表案例','representative_cases')])]
    case_rows=[]
    for v in r['cases'].to_dict('records'):
        photo = f"figures/cases/{v['representative_rgb_id']}.png"
        probs=[v[k+'_pred_state']+' · '+ '/'.join(f'{v[k+"_p_"+s]:.3f}' for s in STATES) for k in MODELS]
        case_rows.append([v['comparator']+(' 改对' if v['direction']=='corrected' else ' 改错'),
            v['group_id'],v['state_id'],f"[{v['representative_rgb_id']}](<{photo}>)",
            ' / '.join(f'{v[k]:.2f}' for k in FEATURES),*probs])
    sections += [md_table(['对照','采集组','真实','代表白光','损失% / 面积% / P10 / 熵',
                           'M1 阶段 · P0/P1/P2','M2 阶段 · P0/P1/P2','M3 阶段 · P0/P1/P2'],case_rows),
                 '每种对照各取一个改对和改错案例。先选跨两级错误，再按出错模型对错误类别的概率降序、组 ID 升序。代表帧为距该组平均白光向量最近的真实帧，距离相同时取图像 ID 排序首位。表中四列实测值是同批成员的组均值。']
    error_rows=[]
    for k in MODELS:
        errors=r['errors'].query('model == @k')
        if len(errors):
            row=errors.iloc[0]
            count=errors.held_tooth.value_counts()
            angles=errors.angle_id.value_counts()
            error_rows.append([NAMES[k],', '.join(count[count==count.max()].index),int(count.max()),
                ', '.join(angles[angles==angles.max()].index),int(angles.max()),row.group_id,
                f'{row.true_state} → {row.pred_state}',f'{row.wrong_class_p:.3f}'])
    sections += ['### 错误集中位置与最高置信错例',md_table(['模型','错组最多的牙','每牙错组数','错组最多视角','每视角错组数',
                    '最高置信错组','误判方向','错误类别概率'],error_rows),
                 '视角列为原始错组数，各视角的拍摄组数可能不同。[全部错组及其错误类别概率](error_cases.csv)']
    sections += ['## 样本覆盖、配对和输入',
        f'上游共 {verification["source_frames"]} 张 QLF，逐图质控排除 {verification["excluded"]} 张。CSV 导出 {len(manifest)} 行，配对核验后新增排除 {len(manifest)-len(active)} 行。三阶段齐全的牙号–视角组合共有 {verification["matched_qc_frames"]} 帧，用于纵向曲线。',
        md_table(['牙号','S0 帧 / 组','S1 帧 / 组','S2 帧 / 组'],[
        [tooth,*[f"{int(block.loc[s,'frames'])} / {int(block.loc[s,'groups'])}" if s in block.index else '0 / 0' for s in STATES]]
        for tooth,data in r['coverage'].groupby('tooth_id') for block in [data.set_index('state_id')]]),
        md_table(['批次','S0 组','S1 组','S2 组'],[
        [session,*[int(v[s]) for s in STATES]] for session,v in r['batch'].pivot(index='session_id',columns='state_id',values='groups').reindex(columns=STATES).fillna(0).iterrows()])]
    legacy=(~active.same_catalog_rgb_pair).sum()
    recovered=(active.rgb_event_source=='same_photo_sha256').sum()+(active.qlf_event_source=='same_photo_sha256').sum()
    mismatched=int((active.author_rgb_hash_status=='local_differs').sum())
    sections += [
        f'正式 Stage 3 配对表逐行确认了 QLF、RGB、牙号和连拍序号。另一个采集目录提供事件与几何信息。两张表有 {legacy} 行白光配对登记不同，本次采用已审核 Stage 3 配对。原采集目录牙号与 Stage 3 不同的 {int((active.qlf_catalog_tooth!=active.tooth_id).sum())} 帧保留双版本牙号。两种模态共 {recovered} 条缺失事件记录由同批次、视角、几何及相同图像 SHA-256 的已知事件补回。',
        '按两种模态的实际拍摄事件共同成组，同牙的全部成员一起留出。跨牙重复采用校正方向后的像素与尺寸校验。逐图清单记录相对路径、文件与像素哈希、事件和排除原因。[逐图清单](sample_manifest.csv) · [各组成员](groups.csv) · [折划分](folds.csv)',
        (f'本次读取研究数据包中的白光，其中 {mismatched}/{len(active)} 张的文件哈希与作者校正版不同。白光结果对应逐图清单中的版本。' if study else '白光和 QLF 图像按固定种子生成，图案和拍摄事件均为合成。') + '[白光输入缩略图](white_light_preview.png)',
        '模型对整幅白光图校正 EXIF 方向、转 8 位灰度、等比缩至长边 224、居中补黑边并复制成三通道。白光输入保留了拍摄背景。',
        '冻结 ImageNet ResNet-18，在推理模式下提取每图 512 维向量。分类前对每组成员取算术均值。白光组均值只在八颗训练牙上拟合四维 PCA 和主成分标准化，QLF 单独在同一批训练组拟合标准化。M3 直接拼接两支的变换结果。三个分类头均为带截距的 L2 多项逻辑回归，各组和各类等权。',
        '## S0 校准帧核对',
        md_table(['模型','S0 类别','判对 / 总帧','S0 召回','P(S0) 中位数'],[
            [NAMES[v.model],'自参考帧' if v.fit_source=='reference exposure' else '独立 Huber 拟合连拍',
             f'{v.correct} / {v.frames}',f'{v.s0_recall:.1%}',f'{v.median_p_S0:.3f}'] for v in r['anchors'].itertuples()]),
        '该核对把每张留出帧作为一个成员的采集组，使用同折模型。校准身份只用于汇总。[逐帧折外诊断](oof_frames.csv) · [S0 核对底表](s0_reference_audit.csv)',
        '## 解释范围与复现',
        ('30% 来自这同一批带阶段标签的牙的曲线。按牙留出隔离了分类器、PCA 和标准化的拟合，此前选阈的影响仍然存在。评价锁定方案对新牙的表现，需要未参与选阈的牙。这一批的目标是受控流程中的实验阶段，矿物损失量需要独立参考测量。[CLAIM 拆分层级](https://rsna.github.io/claim/item-20.html) · [Cawley 与 Talbot](https://jmlr.org/papers/v11/cawley10a.html)' if study else '本演示仅验证实现、输入格式与折划分。研究结论应读取授权九牙数据生成的报告。'),
        'M2 和 M3 的特征计算需要同牙同视角、身份已知的 S0 参考帧。上游 255 计分对 S1/S2 另查已标记的配准质控状态，所有阶段还检查有效牙冠和信号。' + ('本批研究的 S2 全部来自 C04，其 QLF 图像经历过空间亮度调整，批次与阶段效应难以分开。' if study else ''),
        f'从 {len(active)} 帧得到 {n} 个共同采集组。组级折外表 {3*n} 行，逐帧折外表 {3*len(active)} 行。每折实测分类头维数分别为 4、4、8。九折预测封存后才拟合全数据推理包。[折外组预测](oof_groups.csv) · [拟合与收敛记录](fit_audit.csv) · [成员均值复核](group_input_spotcheck.json) · [独立逐折复算](validation.json) · [运行配置](run_config.json)',
        '在已安装项目的环境中运行，新结果写入指定目录。',
        f'```sh\nqlf-stage3 run --data data/{data_name} --out runs/replay\nqlf-stage3 verify --run runs/replay\n```',
        '推理使用 predict 子命令，指定 --model、--input 和 --out。输入 CSV 的 capture_id 标识同一次同视角采集。M1 使用 rgb_path，M2 使用四列 QLF，M3 和默认的三模型同时预测需要两种模态。可用 --arm 选择模型。各模型输出 S0、S1、S2 概率，精确并列采用固定阶段顺序。']
    (out/'report.md').write_text('\n\n'.join(sections)+'\n', encoding='utf-8')

"""Report paired original-only training, including the actual per-metric denominators."""
import pandas as pd

from ..files import load, sha
from .cohort import save_csv
from .evaluation import read_result
from .report import md_table


def report(out, manifest, results):
    active = manifest[manifest.included]
    missing = manifest[~manifest.included]
    groups = results['groups']
    teeth = results['teeth']
    details = teeth[teeth.model.eq('M1')]
    partial = details[details.missing_stages.ne('')]
    stages = results['stages'].set_index(['model','state_id'])
    delivery_count = int(active.original_found_in.eq('supplied_delivery_copy').sum())
    visual = '收到的彩色白光' if delivery_count else '原始彩色白光'
    model_names = {'M1':visual,'M2':'QLF 四特征','M3':visual+'＋QLF'}
    rows=[]
    for m in results['summary'].itertuples():
        s1=stages.loc[(m.model,'S1')]
        rows.append(dict(model=m.model,input=model_names[m.model],tooth_equal_macro_f1=m.primary_macro_f1,
            macro_f1_evaluable_teeth=m.evaluable_teeth,macro_f1_ci_low=m.primary_ci_low,macro_f1_ci_high=m.primary_ci_high,
            s1_correct=int(s1.to_S1_n),s1_groups=int(s1.true_groups),s1_recall=s1.recall,
            correct=m.correct,groups=m.groups,two_stage_errors=m.two_stage_errors))
    scores=pd.DataFrame(rows)
    save_csv(out,'model_scores.csv',scores)
    summary=md_table(['模型','输入','牙等权 Macro-F1','S1 判对 / 真实 S1','全部组判对','S0↔S2 错误'],
        [[s.model,s.input,f'{s.tooth_equal_macro_f1:.4f}',f'{s.s1_correct}/{s.s1_groups}（{s.s1_recall:.1%}）',
        f'{s.correct}/{s.groups}',s.two_stage_errors] for s in scores.itertuples()])
    verified=len(active)-delivery_count
    lines=['# 终点三'+visual+'与 QLF 三模型结果','',
           f'目标 {len(manifest)} 对，实际使用 {len(active)} 对。{verified} 张白光的文件指纹与相机或原始采集备份一致。'
           '建模时直接读取 RGB，未做白平衡或色调校正。QLF 使用既有的 255 参考、固定 30% 阈值的四项实测特征。','']
    if delivery_count:
        received=active[active.original_found_in.eq('supplied_delivery_copy')]
        blocks='、'.join(f'{t}／{s}' for t,s in received.groupby(['tooth_id','state_id']).groups)
        lines.extend([f'{blocks} 的 {delivery_count} 张直接取自本次收到的文件夹。逐张匹配交付版指纹。这些文件与采集清单记录的相机版指纹不同。',''])
    if len(missing):
        omitted=missing.groupby(['tooth_id','state_id']).size()
        lines.extend(['目前缺少与相机原始指纹相符的文件，登记位置为 '+
            '、'.join(f'{t}／{s}（{n} 个）' for (t,s),n in omitted.items())+
            f'。{len(missing)} 个位置对应 {missing.expected_original_sha256.nunique()} 个不同的原始文件指纹。'
            '这些位置已从三个模型同步排除。',''])
    lines.extend([f'留一牙验证仍有 {active.tooth_id.nunique()} 折。每次用八颗牙训练，一颗牙测试，同次连拍先平均特征，'
        f'最后对 {len(groups)} 个采集组各作一次折外预测。','',summary,''])
    if len(partial):
        excluded_labels='、'.join(f'{r.tooth_id} 缺 {r.missing_stages}' for r in partial.itertuples())
        lines.extend([f'{excluded_labels}。三阶段 Macro-F1 按原评价规则只对三个阶段均可测的 {int(scores.macro_f1_evaluable_teeth.iloc[0])} 颗牙等权取均值；'
            'S1 召回与总体判对数则包含全部九颗牙的可用采集组。本次分母与此前 873 对的比较不同。',''])
    counts=manifest.groupby('state_id').size().rename('target_frames').to_frame()
    counts['paired_frames']=active.groupby('state_id').size()
    counts['verified_original_frames']=active[active.rgb_sha256.eq(active.expected_original_sha256)].groupby('state_id').size().reindex(counts.index,fill_value=0)
    counts['supplied_delivery_frames']=counts.paired_frames-counts.verified_original_frames
    counts['groups']=groups.groupby('state_id').size()
    save_csv(out,'cohort_counts.csv',counts.reset_index())
    lines.extend([md_table(['阶段','目标照片对数','实际照片对数','匹配原片','使用交付版','测试组数'],
        [[i,r.target_frames,r.paired_frames,r.verified_original_frames,r.supplied_delivery_frames,r.groups] for i,r in counts.iterrows()]),'',
        '![三个模型在相同采集组的混淆矩阵](figures/01_confusion.png)','',
        '[各采集组的三分类概率、最终阶段和实测 QLF 特征](paired_group_predictions.csv) · '
        '[留出牙与折次](folds.csv) · [逐牙及牙等权融合增益](paired_gains.csv)',''])
    known=manifest[manifest.rgb_image_id.isin([f'I{i:04d}' for i in range(708,715)])]
    if len(known)==7 and known.included.all():
        lines.extend(['此前做过 b* 校偏的 T008／S0／0° 七张（I0708–I0714）在本次均从原始采集备份读取，逐张通过源文件指纹核验。',''])
    lines.extend(['## 白光源文件与新目录的核对','',
        '交付说明记载 T005／S2 做过 75% 强度的牙体白平衡。具体操作与强度来自说明，现有文件只能独立验证版本差异。'
        '有相机原始指纹时直接按该值核验。较早批次按采集时记录的原片备份指纹核验。','',
        '[每张白光的实际输入路径和指纹](original_source_audit.csv) · [缺失原片的登记位置与原始指纹](missing_originals.csv)',''])
    if (out/'incoming_rgb_folder.json').is_file():
        meta=load(out/'incoming_rgb_folder.json')
        path=out/'incoming_rgb_comparison.csv'
        if sha(path)!=meta['comparison_sha256']:
            raise ValueError('Changed incoming-folder comparison')
        incoming=read_result(out,'incoming_rgb_comparison.csv')
        lines.extend([f'本次核查目录为 `{meta["folder"]}`。下表按文件的 SHA-256 与同一图号配对，分别核对交付版本和原始版本。',''])
        tab=incoming.groupby(['tooth_id','state_id','angle_id'],as_index=False).agg(rgb_files=('rgb_image_id','size'),
                     same_as_delivery=('matches_delivered','sum'),same_as_camera=('matches_original','sum'))
        lines.extend([md_table(['牙号／阶段','角度','RGB 文件','匹配交付','匹配原始'],
            [[r.tooth_id+'／'+r.state_id,f'{int(r.angle_id[1:])}°',r.rgb_files,r.same_as_delivery,r.same_as_camera]
             for r in tab.itertuples()]),'','[新目录的逐文件比对结果](incoming_rgb_comparison.csv)',''])
    lines.extend(['## 复现','',
        '白光只做项目约定的 EXIF 方向与 sRGB 解码、全图等比缩放、居中补边和固定网络标准化。冻结 ResNet-18 后，'
        '在每个训练折的组均值上拟合四维 PCA 和标准化。QLF 的四个特征同样只在训练折拟合标准化。'
        '三分类头均为多项 L2 逻辑回归，M1、M2 各四维，M3 八维。保存的每个折外预测均由模型重新载入复算。','',
        '从 stage3 源码目录运行该历史实验，采集档案路径按实际位置填写。'])
    cmd='python experiments/train_original_rgb_qlf.py --capture-root data/acquisition_archive'
    if (out/'incoming_rgb_folder.json').is_file():
        path=load(out/'incoming_rgb_folder.json')['folder'].replace("'","''")
        cmd+=(" --as-received-rgb '" if delivery_count else " --inspect-rgb '")+path+"'"
    lines.extend(['','```powershell',cmd,'```','',
        '如取得白平衡前的备份，可用 `--extra-originals` 指向真实 JPEG、目录或 ZIP。程序仍按采集指纹核验，并在新目录重新训练。','',
        '固定 30% 阈值曾参考同批牙的阶段曲线，S2 与 C04 装置共现。分数用于本队列的内部探索性比较。',''])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf8')

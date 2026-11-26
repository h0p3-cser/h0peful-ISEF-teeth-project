"""Summarize model stability across seeds and paired uncertainty across the nine teeth."""
from itertools import combinations
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..files import PACKAGE, dump, load, sha
from .cohort import MODELS, PROBS, STATES, ensure, save_csv
from .evaluation import read_result
from .report import md_table
from .seed_stability import BRANCHES
from .tooth_identity_check import jointly_held_teeth

NAMES = {"gray":"灰度", "rgb":"RGB"}
MODEL_LABELS = {"M1":"白光", "M2":"QLF", "M3":"白光＋QLF"}


def collect(root, plan):
    records, tooth_tables, outputs = [], [], {}
    fold_hashes = set()
    layouts = {}
    for branch in BRANCHES:
        for seed in plan["training_seeds"]:
            directory = root / branch / f"seed_{seed}"
            completed = load(directory/"complete.json")
            replay = load(directory/"validation.json")
            settings = load(directory/"run_config.json")
            folds_hash = sha(directory/"folds.csv")
            ensure(completed["seed"] == seed and settings["training_seed"] == seed and completed["branch"] == branch
                   and settings["plan_sha256"] == sha(root/"plan.json") and folds_hash == completed["folds_sha256"]
                   and replay["valid"] and replay["converged_heads"] == 27, "Incomplete seed run")
            fold_hashes.add(folds_hash)
            for unit, key, digest_key in [("groups","group_id","group_oof_sha256"),("frames","sample_id","frame_oof_sha256")]:
                filename = f"oof_{unit}.csv"
                ensure(sha(directory/filename) == completed[digest_key], "Held-out results changed")
                output = read_result(directory,filename).set_index(["model",key],verify_integrity=True).sort_index()
                ident = ["held_tooth","fold","true_state"] + (["group_id"] if unit == "frames" else [])
                layout = output[ident]
                if unit in layouts:
                    pd.testing.assert_frame_equal(layout,layouts[unit])
                else:
                    layouts[unit] = layout
                outputs[branch,seed,unit] = output
            metrics = read_result(directory,"metrics_summary.csv").set_index(["model","metric"])
            stages = read_result(directory,"metrics_by_stage.csv").set_index(["model","state_id"])
            tooth_tables.append(read_result(directory,"metrics_by_tooth.csv").assign(branch=branch,seed=seed))
            for model in MODELS:
                s1 = stages.loc[model,"S1"]
                s0,s2 = stages.loc[model,"S0"], stages.loc[model,"S2"]
                primary = metrics.loc[model,"primary_macro_f1"]
                records.append(dict(branch=branch,seed=seed,model=model,tooth_macro_f1=primary.value,
                    tooth_ci_low=float(primary.ci_low),tooth_ci_high=float(primary.ci_high),evaluable_teeth=primary.denominator,
                    groups=plan["groups"],correct=int(s0.to_S0_n+s1.to_S1_n+s2.to_S2_n),
                    s1_correct=int(s1.to_S1_n),s1_groups=int(s1.true_groups),s1_recall=float(s1.recall),
                    s1_as_s0=int(s1.to_S0_n),s1_as_s2=int(s1.to_S2_n),two_stage_errors=int(s0.to_S2_n+s2.to_S0_n),
                    macro_auroc=metrics.loc[(model,"macro_auroc"),"value"],accuracy=metrics.loc[(model,"accuracy"),"value"],
                    log_loss=metrics.loc[(model,"log_loss"),"value"]))
    ensure(len(fold_hashes) == 1, "The six runs used different splits")
    for seed in plan["training_seeds"]:
        for unit in ("groups","frames"):
            q1 = outputs["gray",seed,unit].loc["M2",["pred_state",*PROBS]]
            q2 = outputs["rgb",seed,unit].loc["M2",["pred_state",*PROBS]]
            pd.testing.assert_frame_equal(q1,q2,check_exact=True)
    return pd.DataFrame(records),pd.concat(tooth_tables,ignore_index=True),outputs


def differences(frames):
    probabilities = [f[PROBS].to_numpy() for f in frames]
    max_delta = max(float(np.max(np.abs(a-b))) for a,b in combinations(probabilities,2))
    labels = np.column_stack([f.pred_state for f in frames])
    return max_delta,int((labels != labels[:,:1]).any(axis=1).sum())


def summarize(records, teeth, outputs, plan):
    cfg = plan["protocol"]
    seeds = plan["training_seeds"]
    per_tooth = teeth.pivot_table(index="tooth_id",columns=["branch","model"],values="macro_f1",aggfunc="mean").sort_index()
    ensure(len(per_tooth) == 9 and not per_tooth.isna().any().any(), "All nine teeth must contribute three-stage Macro-F1")
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(len(per_tooth),size=(cfg["bootstrap_replicates"],len(per_tooth)))
    def interval(series):
        return np.quantile(series.to_numpy()[draws].mean(axis=1),[.025,.975])
    rows = []
    for (branch,model),data in records.groupby(["branch","model"],sort=False):
        by_tooth = per_tooth[branch,model]
        lo,hi = interval(by_tooth)
        group_delta,group_changes = differences([outputs[branch,s,"groups"].loc[model] for s in seeds])
        frame_delta,frame_changes = differences([outputs[branch,s,"frames"].loc[model] for s in seeds])
        # A reference value plus its cross-seed range records all fixed-cohort count metrics.
        rows.append(dict(branch=branch,model=model,macro_f1_mean=data.tooth_macro_f1.mean(),
            macro_f1_seed_sd=float(np.std(data.tooth_macro_f1.to_numpy()-data.tooth_macro_f1.iloc[0],ddof=1)),
            macro_f1_seed_min=data.tooth_macro_f1.min(),macro_f1_seed_max=data.tooth_macro_f1.max(),
            tooth_ci_low=lo,tooth_ci_high=hi,tooth_min=by_tooth.min(),tooth_max=by_tooth.max(),
            s1_correct_min=data.s1_correct.min(),s1_correct_max=data.s1_correct.max(),s1_groups=int(data.s1_groups.iloc[0]),
            correct_min=data.correct.min(),correct_max=data.correct.max(),groups=plan["groups"],
            two_stage_errors_min=data.two_stage_errors.min(),two_stage_errors_max=data.two_stage_errors.max(),
            max_group_probability_difference=group_delta,groups_disagreeing=group_changes,
            max_frame_probability_difference=frame_delta,frames_disagreeing=frame_changes))
    comparisons,values = [],[]
    pairs = [((b,"M3"),(b,m)) for b in BRANCHES for m in ("M1","M2")]
    pairs += [(("rgb",m),("gray",m)) for m in MODELS]
    for left,right in pairs:
        diff = (per_tooth[left]-per_tooth[right])*100
        lo,hi = interval(diff)
        name = "-".join("_".join(k) for k in (left,right))
        for tooth,val in diff.items():
            values.append(dict(comparison=name,tooth_id=tooth,delta_pp=val))
        comparisons.append(dict(comparison=name,delta_pp=diff.mean(),ci_low_pp=lo,ci_high_pp=hi,
                    improved=int((diff>1e-10).sum()),tied=int((diff.abs()<=1e-10).sum()),declined=int((diff< -1e-10).sum())))
    return pd.DataFrame(rows),pd.DataFrame(comparisons),pd.DataFrame(values),per_tooth


def tooth_figure(out, per_tooth):
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","Noto Sans CJK SC","PingFang SC","DejaVu Sans"],
                         "axes.unicode_minus":False})
    fig,axes = plt.subplots(1,2,figsize=(10,4.9),sharex=True,sharey=True)
    colors = {"M1":"#48688a","M2":"#bf7233","M3":"#197665"}
    for ax,branch in zip(axes,BRANCHES):
        for i,model in enumerate(MODELS):
            ax.scatter(per_tooth[branch,model],np.arange(len(per_tooth))+(i-1)*.17,
                       s=35,marker=["o","s","D"][i],label=MODEL_LABELS[model],color=colors[model],zorder=2)
        ax.set(yticks=range(len(per_tooth)),yticklabels=per_tooth.index,
               xlabel="留出牙三阶段 Macro-F1",title=NAMES[branch]+"白光分支",xlim=(0,1.04))
        ax.grid(axis="x",color="#e1e7e8")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False)
    fig.subplots_adjust(left=.075,right=.985,bottom=.21,top=.91,wspace=.12)
    fig.savefig(out/"scores_by_tooth.png",dpi=150,facecolor="white")
    plt.close(fig)


def count_range(low,high):
    return str(int(low)) if low == high else f"{int(low)}–{int(high)}"


def confusion_figure(root, seed):
    fig,axes = plt.subplots(2,3,figsize=(10,6),layout="constrained")
    for row,branch in enumerate(BRANCHES):
        table = read_result(root/branch/f"seed_{seed}","confusion_counts.csv")
        for col,model in enumerate(MODELS):
            counts = table[table.model == model].pivot(index="true_state",columns="pred_state",values="groups").loc[STATES,STATES]
            percentages = counts.div(counts.sum(axis=1),axis=0)*100
            ax = axes[row,col]
            im = ax.imshow(percentages,vmin=0,vmax=100,cmap="Blues")
            for i,j in np.ndindex((3,3)):
                value = percentages.iloc[i,j]
                ax.text(j,i,f"{counts.iloc[i,j]}\n{value:.1f}%",ha="center",va="center",
                        color="white" if value>55 else "#19354c")
            ax.set(xticks=range(3),xticklabels=STATES,yticks=range(3),yticklabels=STATES,
                   title=f"{NAMES[branch]}分支 · {model} {MODEL_LABELS[model]}")
            if not col:
                ax.set_ylabel("真实阶段")
            if row == 1:
                ax.set_xlabel("预测阶段")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label="占该真实阶段采集组数（%）")
    fig.savefig(root/"confusion_counts.png",dpi=160,facecolor="white")
    plt.close(fig)


def report(root):
    from .endpoint_validation import final_models

    plan = load(root/"plan.json")
    study = plan.get("dataset_kind","study") == "study"
    scores,teeth,outputs = collect(root,plan)
    summary,comparisons,by_tooth,per_tooth = summarize(scores,teeth,outputs,plan)
    identity_check = jointly_held_teeth(root,plan)
    final_models(root,plan)
    for name,frame in [("scores_by_seed",scores),("stability_summary",summary),("scores_by_tooth_and_seed",teeth),
                       ("paired_comparisons",comparisons),("paired_comparisons_by_tooth",by_tooth)]:
        save_csv(root,name+".csv",frame)
    tooth_figure(root,per_tooth)
    seeds = plan["training_seeds"]
    confusion_figure(root,seeds[0])
    seed_text = "、".join(map(str,seeds))
    seed_scores = scores.set_index(["branch","seed","model"]).tooth_macro_f1
    overall = md_table(["白光分支","模型",*[f"种子 {s}" for s in seeds],"种子间 SD","九牙 95% 区间",
                        "S1 判对","全部组判对","跨两级错误"],
        [[NAMES[r.branch],r.model+" "+MODEL_LABELS[r.model],
          *[f"{seed_scores.loc[r.branch,s,r.model]:.4f}" for s in seeds],
          f"{r.macro_f1_seed_sd:.6f}",f"{r.tooth_ci_low:.4f}–{r.tooth_ci_high:.4f}",
          f"{count_range(r.s1_correct_min,r.s1_correct_max)}/{r.s1_groups}",
          f"{count_range(r.correct_min,r.correct_max)}/{r.groups}",
          count_range(r.two_stage_errors_min,r.two_stage_errors_max)] for r in summary.itertuples()])
    labels={**{f"{b}_M3-{b}_{m}":NAMES[b]+"分支，融合相对"+MODEL_LABELS[m] for b in BRANCHES for m in ["M1","M2"]},
            **{f"rgb_{m}-gray_{m}":m+"，RGB 分支相对灰度分支" for m in MODELS}}
    deltas=md_table(["逐牙配对比较","平均变化（百分点）","九牙 95% 区间（百分点）","提高／持平／降低的牙数"],
        [[labels[r.comparison],f"{r.delta_pp:+.2f}",f"{r.ci_low_pp:+.2f} 至 {r.ci_high_pp:+.2f}",
          f"{r.improved}／{r.tied}／{r.declined}"] for r in comparisons.itertuples()])
    logs=md_table(["分支","种子","运行秒数","日志","采集组折外概率","逐折训练记录"],
        [[NAMES[b],s,load(root/b/f"seed_{s}"/"complete.json")["seconds"],
          f"[过程日志]({b}/seed_{s}/run.log)",f"[预测表]({b}/seed_{s}/oof_groups.csv)",
          f"[训练审计]({b}/seed_{s}/fit_audit.csv)"] for b in BRANCHES for s in seeds])
    max_p = summary.max_group_probability_difference.max()
    lines = ["# 终点三灰度与原图 RGB 的三种子结果","",
        ("本报告使用授权研究数据。" if study else "本报告使用程序生成的合成数据，分数仅用于验证软件。"),
        f"共 {plan['paired_frames']} 对图像、{plan['groups']} 个采集组、{len(plan['teeth'])} 颗牙。灰度和 RGB 两个分支各以种子 {seed_text} 独立运行。每次重新读取同一批白光原图，在九个留一牙折分别拟合 M1、M2、M3，合计 162 次折内分类器拟合。",
        "", "## 三个种子的折外结果","",overall,"",
        f"表中得分为先按牙计算、再对九颗牙等权平均的三类 Macro-F1。比较了每个模型各组的 S0、S1、S2 三项概率。三个种子间最大的组概率绝对差为 {max_p:.8g}，同一分支模型中单个模型最多有 {int(summary.groups_disagreeing.max())}/{plan['groups']} 个组改变预测阶段。逐帧追溯预测的最大概率差为 {summary.max_frame_probability_difference.max():.8g}。",
        "", f"![种子 {seeds[0]} 的六幅混淆矩阵，格中为采集组数和占该真实阶段的百分比](confusion_counts.png)",
        "", "## 不同牙的表现","",deltas,"","![九颗留出牙的三模型得分](scores_by_tooth.png)","",
        f"种子间 SD 描述固定数据和固定留一牙划分时的运行波动。固定的预训练权重、全量 SVD 的 PCA 和 L-BFGS 逻辑回归采用确定性计算，相同预测符合这套训练方法的预期。九牙区间用固定折外预测按牙配对重采样 {plan['protocol']['bootstrap_replicates']:,} 次（种子 {plan['protocol']['bootstrap_seed']}）。同一颗牙的所有视角和连拍始终一起取样，三个训练种子共用九颗独立牙。",
        "", ("30% 阈值曾参考同批牙的阶段曲线，S2 与 C04 采集装置共现。结果用于九牙内部探索性比较。"
             if study else "合成图使用已知配准和生成标签。研究性能请查授权数据的独立运行。"),
        "", "## 输入和训练记录","",
        ("T005 的 35 张 S2 白光以负责人提供的文件作为项目原图。" if study else "")+
        "灰度分支将原始白光转为灰度，RGB 分支保留原始颜色。两者均按原方案缩放、补边并使用固定的网络输入标准化。M2 读取同一份 255 参考、固定 30% 的 QLF 四特征。",
        "", "相同视角的同次连拍先平均特征。每折用八颗牙拟合 PCA、标准化和逻辑回归，留出一颗牙预测。运行同时设置 Python、NumPy、PyTorch 和估计器的训练种子，使用独立进程、确定性 CPU 编码和固定线程配置。每折模型均重新载入并复算折外预测。","",
        "[总日志](run.log) · [实验配置和输入指纹](plan.json) · [代码版本指纹](source_sha256.json) · [全部种子指标](scores_by_seed.csv) · [逐牙与逐种子指标](scores_by_tooth_and_seed.csv) · [跨种子概率检查](stability_summary.csv)","",logs,""]
    for branch in BRANCHES:
        prefix = f"{branch}/seed_{seeds[0]}"
        if (root/prefix/"figures"/"03_stage_roc.png").is_file():
            lines += [NAMES[branch]+"分支　"+
                f"[逐牙融合增益]({prefix}/figures/02_tooth_gains.png) · [分阶段 ROC]({prefix}/figures/03_stage_roc.png) · "+
                f"[QLF 特征分布]({prefix}/figures/04_qlf_features.png) · [融合纠正和新增的错误]({prefix}/figures/05_fusion_cases.png) · "+
                f"[逐组概率、实测值和示例图路径]({prefix}/paired_group_predictions.csv)",""]
    if (root/"data_import.json").exists():
        receipt = load(root/"data_import.json")
        lines += [f"本次纳入的 {receipt['images']} 张来件有 {receipt['unchanged_image_bytes']} 张与替换前文件逐字节一致。[入库位置和文件校验值](data_import.json)",""]
    if (root/"input_verification.json").exists():
        lines += ["[逐项比对原有图号、路径、分组和 873 张项目原图后的核验结果](input_verification.json)",""]
    if identity_check is not None:
        mismatches,holdout = identity_check
        lines += ["## T004 和 T005 的牙号复核","",
            f"这两颗牙的 S0、S1 中有 {mismatches.frames.sum()} 对、{mismatches.groups.sum()} 个组，在既有配对表和现行目录中对应相反牙号。主分析保留既有配对和九折划分。另用首个种子 {seeds[0]} 的冻结特征，把两颗牙的全部阶段同时留出，七颗其他牙重新拟合 PCA、标准化和三类分类头。两种牙号下，测试牙的任何照片都不会进入这项复核的训练集。下表只比较这两颗牙的同一批测试组。","",
            md_table(["分支","模型","原九折中这两牙的判对组数","两牙同时留出的判对组数","同时留出时 S1 判对"],
                [[NAMES[r.branch],r.model,f"{r.separate_holdouts_correct}/{r.test_groups}",
                  f"{r.joint_holdout_correct}/{r.test_groups}",f"{r.joint_s1_correct}/{r.s1_groups}"]
                 for r in holdout.itertuples()]),"",
            "[两份清单的牙号对应](joint_tooth_holdout/catalog_labels.csv) · [同时留出的逐组概率](joint_tooth_holdout/predictions.csv)",""]
    dataset_example = "data/study" if study else "data/demo"
    lines += ["## 预测与复现","",
        f"使用首个预设种子 {seeds[0]} 在全部 {plan['groups']} 组上另行拟合了两份新样本推理模型。上表所有分数均来自折外预测。",
        "[灰度推理模型](models/gray.joblib) · [RGB 推理模型](models/rgb.joblib) · [模型校验值](models/index.json)","",
        "在新目录中运行完整流程。数据位置可随复制后的布局调整。","",
        "```sh",f"qlf-stage3 endpoint --data {dataset_example} --out runs/repeat --seeds "+" ".join(map(str,seeds)),
        "qlf-stage3 verify --run runs/repeat","```",""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf8")
    source_files = [*sorted((PACKAGE/"modeling").glob("*.py")),PACKAGE/"bundle.py",PACKAGE/"files.py",
                    PACKAGE/"rgb/color.py",PACKAGE/"config/protocol.json",PACKAGE/"stability_cli.py",PACKAGE/"workflow.py"]
    dump(root/"source_sha256.json",{p.relative_to(PACKAGE).as_posix():sha(p) for p in source_files})
    dump(root/"verification.json",dict(complete=True,validated_runs=len(BRANCHES)*len(seeds),seeded_model_results=len(scores),
         verified_classifier_fits=sum(load(root/b/f"seed_{s}"/"validation.json")["converged_heads"] for b in BRANCHES for s in seeds),
         same_fold_membership=True,qlf_predictions_equal_across_branches=True,max_group_probability_difference=float(max_p),
         additional_joint_holdout_heads=0 if identity_check is None else len(identity_check[1]),
        sha256={n:sha(root/n) for n in ["report.md","scores_by_seed.csv","stability_summary.csv","paired_comparisons.csv","scores_by_tooth.png","confusion_counts.png","models/index.json","source_sha256.json"]}))
    logging.info("Verified six runs. Summary %s",root/"report.md")

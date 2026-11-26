# 白光 RGB 黄蓝校偏与同源灰度对照

在独立的 `stage3` 源码目录运行。该历史实验依次生成输入、训练两个白光模型，并在原分辨率核验灰度。

```powershell
python experiments/rgb_bstar_branch.py --history runs/study/evaluation --crown-run runs/rgb_full --out runs/bstar
```

## 数据与方法

输入为历史九牙研究的 `sample_manifest.csv`、`groups.csv`、`folds.csv`、配对事件对应的原始白光，以及原分类配置与冻结的 ResNet-18 权重。程序校验原照片文件、源像素和权重的 SHA-256。迁移数据时，用 `--history` 指向三个 CSV 所在的运行目录，`--rgb-root` 指向本机 `Data_rgb_corrected`，`--config` 指向历史分类配置，`--checkpoint` 指向 `resnet18-f37072fd.pth`。可用 `--crown-run`、`--study-data` 指定已有白光牙冠的配准记录和原生掩膜。缺少牙冠记录的帧仍计算整幅图的灰度差。

先在隐去阶段标签的采集组代表图及各连拍边缘核对装置和材料。[逐组参照记录](../src/qlf_stage3/config/rgb_bstar_regions_v1.json)锁定原图坐标和代表帧文件指纹。只有能辨认出本色的牙外哑光灰／白表面才可取样。筛选 8 位 sRGB 各通道均在 6–249 的像素，取区内 CIELAB b* 中位数，多块时对各块中位数再取中位数。对整张照片的 b* 减去该值，保持 L*、a*，转回 sRGB 并量化。缺少参照的帧原像素直通，仍参加两个模型。当前数据没有专门放置灰卡，选区记录仅适用于这批照片。

两组共用原分辨率 PNG、实际连拍组、留出牙与分类配置。先平均组内 512 维冻结网络特征，每个训练折分别拟合 PCA、标准化器和逻辑回归。原图灰度与导出灰度的差异仅在分数保存后测量。整图拍摄包含装置背景且 S2 与 C04 共现，结果限于这批牙的内部比较。

## 输出

| 文件 | 内容 |
| --- | --- |
| `frame_manifest.csv`、`coverage.csv`、`rgb/` | 每张照片、坐标、偏移、状态、事件、原始及导出文件指纹；分阶段实测覆盖率；原生分辨率 PNG |
| `reference_review/` | 按盲审编号排版的 127 组代表原图和各组全部连拍的上边角，可回查直通原因 |
| `oof_groups.csv`、`scores_by_tooth.csv` | 同一组两套折外概率、最终预测、成员图像 ID 和逐牙配对差 |
| `scores.csv`、`confusion.csv`、`model_scores.md` | 牙等权 Macro-F1、S1 召回、混淆矩阵与先保存的模型结果 |
| `gray_by_frame.csv`、`gray_summary.csv`、`gray_consistency.md` | 每帧的整图及可用牙冠 MAE、P95，区分成功校偏和直通 |
| `reference_bstar.csv`、`reference_samples.png` | 用同一组有效取样像素复读原图和导出 PNG 的参照 b*，并标出原图位置 |
| `I0654_same_position.png`、`I0654_contrast.json` | 固定坐标与原尺度下的棕黑纹、邻近牙面灰度和对照图 |
| `report.md` | 原模型结果与随后追加的灰度核验 |

需要逐段复现时，给相同的 `--out` 路径依次运行 `--step prepare`、`--step score`、`--step gray-qc`。准备步骤接收空目录；增加灰度报告时会复核已保存的组级预测和模型分数指纹。

## 与 QLF 组成三模型比较

在同一源码目录运行以下命令，读取上面的白光分支和固定 30% 的 QLF 数据。

```powershell
python experiments/rgb_qlf_comparison.py --branch runs/bstar --history runs/study/evaluation --features runs/study/measurements/training-features-255-thr30.csv --rgb-root data/study/workspace/Data_rgb_corrected --out runs/bstar_fusion
```

主比较为 M1 彩色白光、M2 四项 QLF、M3 彩色白光＋QLF。另用全套原始彩色照片和分支 PNG 的同源灰度各重跑同样的三个模型。三个版本使用相同的留牙折和相同的 127 个采集组。`--branch`、`--history`、`--features`、`--rgb-root`、`--checkpoint` 可指定本机数据与权重。

打开 `report.md` 阅读三模型混淆矩阵、逐牙融合增益及校偏覆盖。`paired_group_predictions.csv` 提供每组的三套概率、预测阶段、代表照片和四项 QLF 实测均值。`model_comparison.csv` 汇总全部输入版本，`paired_differences.csv` 区分校偏、颜色及加入 QLF 的分数差和改判组数。保存的模型可复算每一个折外概率。

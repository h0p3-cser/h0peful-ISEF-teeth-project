# Stage 3 — QLF Analysis / 终点三 QLF 分析

[中文](#中文) · [English](#english)

## 中文

[跳到 English](#english)

本项目分析三个阶段的牛牙定量光诱导荧光（QLF）图像。S0 为未处理基线，S1 为 24 或 35 小时，S2 为 72 小时（C04）。可以生成逐图特征、不同阈值的荧光响应区，以及同牙同角度的荧光损失图册。

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./stage3
qlf-stage3 --help
```

需要 Python 3.10 或更新版本。也可用 `python -m qlf_stage3` 运行。原有图表模板使用中文标签，建议安装 Noto Sans CJK SC 等中日韩字体。

### 输入数据

把 `DATA_ROOT` 设为实验工作区根目录。目录中须有 `Data/`、`Data_rgb_corrected/` 和 `QLF_GPTadvice/QLF_GPTadvice_S0_S1_S2_C04.csv`。流程从中读取拍摄清单、C04 来源记录、原始 JPEG、审核后的牙体和牙冠掩膜、配准索引及配对 RGB 图像。输入沿用实验中的文件名。公开仓库提供示例汇总图，原图、完整掩膜和训练 CSV 保存在本地。

当前数据包括 9 颗牙、每颗牙 5 个角度，每个阶段和角度有 7 个拍摄位置。T004/T005 按复核后的 S0 编号配对。导出的 CSV 每张 QLF 图像一行，保存 `sample_id`、`qlf_path`、配对的 `rgb_path` 与 `rgb_raw_path`、阶段标签及测量状态。图像路径相对于 `Data/`。

### 运行分析

```bash
qlf-stage3 --data-root "$DATA_ROOT" features --output outputs/training-features.csv
qlf-stage3 --data-root "$DATA_ROOT" area25 --output-dir outputs/area25
qlf-stage3 --data-root "$DATA_ROOT" thresholds --output-dir outputs/thresholds
qlf-stage3 --data-root "$DATA_ROOT" huber-thresholds \
  --samples outputs/thresholds/threshold-area.csv --output-dir outputs/huber-thresholds
qlf-stage3 --data-root "$DATA_ROOT" atlas --output-dir outputs/atlas
```

| 命令 | 输出 |
| --- | --- |
| `features` | 每张 QLF 图像的特征和配对 RGB 路径 |
| `area25` | 原图上的 25% 响应区标注、逐图像素数和占比、按角度等权的逐牙均值 |
| `thresholds` | S0→S2 直接比较在 5%、10%、20%、30%、40% 下的面积折线图和逐视图对照图 |
| `huber-thresholds` | S0→S1 和 S0→S2 校正后的折线图、逐视图对照图及牙根拟合诊断 |
| `atlas` | PDF 图册、逐视图 PNG、全部连拍对照 ZIP 及配对明细 |

阈值比较抽取 10 个视图，每个角度 2 个。粗线表示面积占比的中位数。 [S0→S2 直接比较](examples/direct-s2-threshold-areas.png) · [S0→S1 Huber 校正](examples/huber-s1-threshold-areas.png) · [S0→S2 Huber 校正](examples/huber-s2-threshold-areas.png)。

#### 掩膜与配准工具

- `c04-masks --qlf-body PATH --rgb-body PATH` 可从已有牙体掩膜重建 C04 语义标签和索引，输出写入实验掩膜目录。重建时可在单独的输入副本上运行。
- `direct-features --reuse-existing-crowns --output-dir PATH` 使用已审核的牙冠掩膜重新计算配对特征。其他命令从 `QLF_GPTadvice/` 读取当前配对表。
- `baseline-masks --stage PATH --scope legacy --manual-root PATH` 是原先的基线色斑标注流程，需要旧版的四目录结构及人工种子掩膜。当前的三阶段分析使用已审核的基线索引。

早期的双根区曝光校正试验可通过 `root-exposure --masks PATH --semantic PATH` 运行。它读取归档的 ROI 配置，需要对应的旧版图像、索引和人工标注的掩膜。

### 四项特征的计算

| 特征组 | 计算方法 |
| --- | --- |
| Huber ΔF | 在牙根取互不重叠的 21×21 像素块，用同一块在 S0 和目标阶段的中位亮度拟合 `目标亮度 = a × S0 + b`。在牙冠上计算 `100 × (预测亮度 − 实测亮度) / 预测亮度`，对有效像素取有符号均值。 |
| 25% 响应面积 | 统计损失严格大于 25% 的有效像素，用有效牙冠像素数作分母计算百分比。逐牙汇总时先平均同一角度的连拍，再对可用角度等权平均。 |
| 强度分布 | 在审核后的牙冠掩膜内，用原图绿通道计算均值、中位数、第 10 百分位数和总体标准差。 |
| 纹理 | 将绿通道量化为 16 级，统计两个像素都在掩膜内的水平和垂直邻接像素对。构建对称灰度共生矩阵，计算对比度、同质性、角二阶矩和联合熵。 |

纵向测量要求配准通过、目标牙冠至少有 85% 重合、共同牙冠至少有 500 像素，且共同区域内的原始有效信号至少覆盖 90%。剔除饱和及基线信号过低的像素，Huber 预测值须在 `[16, 255)`。S0 和测量失败的纵向列留空。正损失表示变暗。牙根留出误差及牙冠外推比例可用于检查拟合。

三阶段分类共有 917 行具备全部 8 项单图强度和纹理特征。比较 QLF 与 QLF+RGB 时使用相同样本，并按牙号划分训练和测试。纵向列的缺失模式会泄露基线阶段。部分已审核的牙冠掩膜也依赖 S0，只有单张图的推理场景需要独立获得牙冠掩膜。部分来源文件的内容相同。

这些指标描述 JPEG 绿通道的变化，受拍摄条件和分阶段亮度处理影响。阈值面积是图像平面的响应区面积，实际病变面积和矿物损失需要独立验证。

### 源码与测试

共用代码在 `src/qlf_stage3/`。`tests/` 使用合成输入核对数值计算。

```bash
python -m unittest discover -s stage3/tests
```

终点三源码采用 [AGPL-3.0](LICENSE)，与终点一一致。数据和第三方依赖分别适用其各自条款。

## English

[跳到中文](#中文)

Analyze masked bovine-tooth QLF images at baseline (S0), 24 or 35 hours (S1), and 72 hours (S2, C04). The package produces per-image features, threshold response maps, and a same-tooth, same-view loss atlas.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./stage3
qlf-stage3 --help
```

Python 3.10+ is required. The commands also run as `python -m qlf_stage3`. A CJK font such as Noto Sans CJK SC is needed for Chinese labels in the original plot templates.

### Input data

Set `DATA_ROOT` to the experimental workspace containing `Data/`, `Data_rgb_corrected/`, and `QLF_GPTadvice/QLF_GPTadvice_S0_S1_S2_C04.csv`. This tree supplies the capture manifest, C04 source ledger, original JPEGs, reviewed body and crown masks, registration indices, and matched RGB files. It retains historical dataset filenames. The public project includes example summary plots. Camera files, full masks and the study training CSV are local inputs or outputs.

The active data contain nine teeth, five nominal views and seven capture slots per stage/view. The reviewed T004/T005 S0 assignments are applied. The exported CSV keeps one row per active QLF image with its `sample_id`, `qlf_path`, paired `rgb_path` and `rgb_raw_path`, stage label and measurement status. Its image paths are relative to `Data/`.

### Run

```bash
qlf-stage3 --data-root "$DATA_ROOT" features --output outputs/training-features.csv
qlf-stage3 --data-root "$DATA_ROOT" area25 --output-dir outputs/area25
qlf-stage3 --data-root "$DATA_ROOT" thresholds --output-dir outputs/thresholds
qlf-stage3 --data-root "$DATA_ROOT" huber-thresholds \
  --samples outputs/thresholds/threshold-area.csv --output-dir outputs/huber-thresholds
qlf-stage3 --data-root "$DATA_ROOT" atlas --output-dir outputs/atlas
```

| Command | Outputs |
| --- | --- |
| `features` | One row per QLF image and links to the matched RGB input |
| `area25` | Overlays on target images, pixel counts and fractions, and equal-angle tooth summaries |
| `thresholds` | Direct S0→S2 curves and view panels at 5%, 10%, 20%, 30%, and 40% |
| `huber-thresholds` | S0→S1 and S0→S2 corrected curves, view panels and root-fit diagnostics |
| `atlas` | PDF, per-view PNGs, a ZIP of repeat-image panels and a pair ledger |

Ten views are used for the threshold comparison, two per angle. The thick line is the median. [Direct S0→S2](examples/direct-s2-threshold-areas.png) · [Huber S0→S1](examples/huber-s1-threshold-areas.png) · [Huber S0→S2](examples/huber-s2-threshold-areas.png).

#### Mask and registration tools

- `c04-masks --qlf-body PATH --rgb-body PATH` rebuilds C04 semantic labels and indices from existing body masks. This command writes derived label files to the experimental mask directory. Run it on a separate copy of the inputs when regenerating labels.
- `direct-features --reuse-existing-crowns --output-dir PATH` regenerates paired features using reviewed crown masks. The other commands read the active pairing table from `QLF_GPTadvice/`.
- `baseline-masks --stage PATH --scope legacy --manual-root PATH` is the original baseline stain labeling workflow. It expects the archival four-folder layout and the manual seed masks. The current three-stage analysis uses the reviewed baseline index.

The earlier two-root exposure experiment is available through `root-exposure --masks PATH --semantic PATH`. It uses the archived ROI configuration and requires the corresponding legacy images, indices and hand-marked masks.

### Feature definitions

| Group | Calculation |
| --- | --- |
| Huber ΔF | Fit `target = a × S0 + b` to paired medians from disjoint 21 × 21 pixel root blocks. On the crown use `100 × (predicted − measured) / predicted`, then average the signed valid pixels. |
| 25% response area | Count valid pixels with loss strictly greater than 25%. Divide by the valid crown pixels for a percentage. Tooth summaries average repeats within a view and then weight available views equally. |
| Intensity distribution | Native green-channel mean, median, 10th percentile and population standard deviation inside the reviewed crown mask. |
| Texture | A symmetric 16-level green-channel co-occurrence matrix counts horizontal and vertical neighbor pairs wholly inside the mask. Report contrast, homogeneity, angular second moment and joint entropy. |

Longitudinal measurements require verified registration, target crown overlap of at least 85%, a common crown region of at least 500 pixels, and at least 90% usable raw signal inside it. Saturated and low-baseline pixels are excluded. The Huber prediction must be in `[16, 255)`. Baseline and failed longitudinal measurements are empty. Positive losses indicate darkening. Root holdout errors and the crown extrapolation fraction help assess the fit.

For stage classification, 917 rows have all eight single-image intensity and texture columns. Compare QLF-only and QLF+RGB using the same rows and tooth-level splits. Missing longitudinal values reveal the baseline stage. Some reviewed crown masks also depend on S0, so single-image deployment requires an independently available mask. Some source records share identical file content.

These measures describe JPEG green-channel changes, which are affected by capture conditions and stage-dependent brightness processing. The threshold area is a pixel-plane response area. Physical lesion area and mineral loss require independent validation.

### Source and tests

`src/qlf_stage3/` contains the shared package. `tests/` verifies numerical kernels using synthetic inputs.

```bash
python -m unittest discover -s stage3/tests
```

Stage 3 source code is licensed under [AGPL-3.0](LICENSE), consistent with Stage 1. Data and third-party dependencies retain their respective terms.

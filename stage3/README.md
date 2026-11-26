# 终点三 · QLF 与白光的三阶段分类

给定同一采集组的白光与定量光诱导荧光（QLF）图像，项目预测 S0、S1、S2 的概率和最终阶段。正式入口 `qlf-stage3 endpoint` 串起牙根校准、逐图质控、四特征导出、原图 RGB 与同源灰度的三模型比较、九折留一牙验证，以及用于新采集组的推理模型。

## 在自己的电脑上运行

使用 64 位 Python 3.11。在 Windows PowerShell 中打开源码目录，创建虚拟环境并安装项目。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

在 Linux 或 macOS 终端中使用以下命令。

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

以下命令适用于上述系统。公开演示生成九颗合成牙的彩色白光和 QLF 图像，共 54 对帧、27 个采集组，并用真实测量和训练代码运行完整流程。

```sh
qlf-stage3 demo-data --out data/demo
qlf-stage3 endpoint --data data/demo --out runs/demo
qlf-stage3 verify --run runs/demo
```

打开 `runs/demo/stability/report.md` 查看六组模型结果、混淆矩阵和逐牙图。每个采集组的折外概率保存在 `stability/{gray,rgb}/seed_*/oof_groups.csv`。首次白光编码会使用本地权重或从 Torchvision 下载固定的 ResNet-18 权重，有离线文件时可在 `endpoint` 后加 `--weights PATH`。每次完整运行选择新的输出目录。[离线安装、分步运行与推理](docs/reproducibility.md)

## 复现九牙研究

采集数据持有者将完整授权数据包放入 `data/study/`，其中包括审核配对、掩膜、配准文件和可选离线权重。也可把命令中的路径换为其他位置。公开源码附合成数据生成器。[授权输入的制作和布局](docs/data.md)

```sh
qlf-stage3 check-data --data data/study
qlf-stage3 endpoint --data data/study --out runs/endpoint-study --seeds 42 123 2024
qlf-stage3 verify --run runs/endpoint-study --reference-study
```

| 步骤 | 计算和输出 |
| --- | --- |
| QLF 测量 | 同牙同视角的 S0 参考帧固定取 a=1、b=0，其余各帧用配准牙根逐图拟合。有效牙冠上保留有符号损失，输出逐图损失图、质控和阈值曲线。 |
| 四特征导出 | 平均荧光损失、损失严格大于 30% 的面积占比、绿通道 P10 和纹理熵。CSV 和逐图核验结果位于 `measurements/`。 |
| 模型比较 | 每种白光输入分别训练 M1 白光、M2 QLF 和 M3 融合。冻结 ResNet-18 后，同组连拍先平均特征，在训练折内拟合 PCA 和标准化，再以 4、4、8 维输入分别拟合逻辑回归。 |
| 验证与推理 | 每折留出一颗牙。三个种子各自重新编码原图并训练，逐组保存概率与最终阶段。用首个预设种子在全体组上另拟合两份推理模型，位于 `stability/models/`。 |

研究数据有 873 对合格帧、127 个采集组和 9 颗牙。T005 的 35 张 S2 白光采用负责人提供的项目原图。灰度和 RGB 读取相同文件，正式流程不做白光色调校正。

| 白光输入 | M1 白光 Macro-F1 | M2 QLF Macro-F1 | M3 融合 Macro-F1 | M3 的 S1 召回 | M3 总判对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同源灰度 | 0.7039 | 0.7527 | 0.8587 | 32/41 | 110/127 |
| 原图 RGB | 0.7631 | 0.7527 | 0.9020 | 35/41 | 115/127 |

分数先按牙计算，再对九颗牙等权平均。三个种子在同一分支模型上的折外概率一致。30% 阈值曾参考同批牙的阶段曲线，结果描述固定阈值后的九牙内部探索性比较。[研究结果、配对区间与牙号复核](docs/study-results.md)

## 从结果到预测

下面用 RGB 推理模型演示输入格式。新采集组提供每帧白光路径、采集组 ID 和四个 QLF 特征即可。模型按组输出三类概率和预测阶段。

```sh
qlf-stage3 predict --model runs/demo/stability/models/rgb.joblib --data data/demo --input runs/demo/stability/rgb/seed_42/sample_manifest.csv --out runs/demo-predictions
```

## 项目目录

| 目录 | 用途 |
| --- | --- |
| `src/qlf_stage3/` | 安装包和命令行、QLF 测量、特征导出、数据包制作 |
| `src/qlf_stage3/modeling/` | 同组输入、训练折、RGB 与灰度模型、推理、报告和独立复算 |
| `src/qlf_stage3/config/` | 锁定的数值协议、校验基准与配准配置 |
| `docs/`、`examples/` | 使用手册、数据字典、研究结果和示例输入 |
| `experiments/` | 历史白光校色先导与数据维护工具。[实验索引](experiments/README.md) |
| `tests/`、`.github/workflows/` | 单元测试与 Windows、Linux 的安装包端到端检查 |
| `data/`、`runs/` | 使用者本地的输入和运行输出，单独传递授权数据 |

历史灰度复现命令 `run`、逐步 `measure`、`export`、`train` 和白光校色先导仍可按需调用。已有核验过的 CSV 可从 `stability` 接着训练。[命令说明](docs/reproducibility.md) · [贡献指南](CONTRIBUTING.md) · [引用信息](CITATION.cff) · [AGPL-3.0 许可证](LICENSE) · [上游来源及数据、权重说明](NOTICE.md)

## English quick start

With Python 3.11, install the source distribution using `python -m pip install .`. Run the three public demo commands above to generate paired synthetic images, train both white-light branches, and replay the held-out predictions. To reproduce the published nine-tooth comparison, supply the authorized study dataset and use `verify --reference-study`. Each final inference bundle predicts S0, S1 and S2 probabilities for an acquisition group.

# 重跑、验收与推理

## 安装与离线运行

参考环境为 64 位 Python 3.11 和 CPU。Windows 完整依赖版本见[锁定文件](../requirements-windows-py311.lock)。在空虚拟环境中运行以下命令。

```sh
python -m pip install -r requirements-windows-py311.lock
python -m pip install .
```

Linux 可先用 `python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0` 安装 CPU 版，再运行 `python -m pip install .`。macOS 选用适配本机架构的 Python 和 PyTorch。中文图表优先使用系统的微软雅黑、苹方或 Noto Sans CJK。

白光编码器使用 Torchvision ResNet-18 ImageNet-1K V1 权重，SHA-256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。可显式传 `--weights path/to/resnet18-f37072fd.pth`，也可放入数据包的 `weights/`。未指定本地文件时使用 Torch 缓存或官方下载。M2 单独推理只需要四个 QLF 特征。[Torchvision 模型说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

要离线安装，在与目标机相同操作系统、架构和 Python 版本的联网电脑执行 `python -m pip wheel --wheel-dir wheelhouse .`。Linux 使用 CPU 版时可先把上述 CPU 索引中的 torch 和 torchvision 下载到 wheelhouse，再添加 `--find-links wheelhouse`。将 wheelhouse 和独立的编码器权重复制到目标机，在新虚拟环境执行 `python -m pip install --no-index --find-links wheelhouse qlf-stage3==1.2.0`。

## 完整流程和分步续跑

公开演示先运行 `qlf-stage3 demo-data --out data/demo`。九牙研究使用完整授权数据包。以下命令从配准与掩膜、原图、采集事件开始运行，统一生成两种白光输入的测量、模型、报告和推理文件。公开演示可将 `data/study` 换为 `data/demo`，验收时省略 `--reference-study`。

```sh
qlf-stage3 check-data --data data/study
qlf-stage3 endpoint --data data/study --out runs/endpoint-study --seeds 42 123 2024
qlf-stage3 verify --run runs/endpoint-study --reference-study
```

也可依次运行下面三步。已经得到完整的逐图表、数值损失图和导出核验文件时，可从相应步骤继续。

```sh
qlf-stage3 measure --data data/study --out runs/measurements
qlf-stage3 export --data data/study --measurements runs/measurements
qlf-stage3 stability --data data/study --features runs/measurements/training-features-255-thr30.csv --out runs/gray-rgb-seeds --seeds 42 123 2024
qlf-stage3 verify --run runs/gray-rgb-seeds --reference-study
```

`measure` 和 `endpoint` 使用新目录。`export` 可在同一测量目录再次执行，逐一检查损失图，缺失或使用旧口径时补算并记录。训练中断后，已生成 `stability/plan.json` 的实验可用以下命令接续。程序检查已有运行的计划及折外预测指纹，保留中断尝试的日志。

```sh
qlf-stage3 stability --out runs/endpoint-study/stability --step train
qlf-stage3 stability --out runs/endpoint-study/stability --step report
qlf-stage3 verify --run runs/endpoint-study/stability --reference-study
```

## 读报告与核对指标

| 相对于完整运行目录的文件 | 用途 |
| --- | --- |
| `measurements/training-features-255-thr30.csv` | 每个通过质控的 QLF 帧一行，保留配对和四个数值 |
| `stability/plan.json`、`stability/paired_inputs.csv` | 数据版本、照片指纹、牙号、实际连拍组和种子 |
| `stability/report.md`、`stability/run.log` | 跨种子结果、逐牙增益、混淆、S1 召回和运行日志 |
| `stability/{gray,rgb}/seed_*/` | 单次运行的折划分、逐组及逐帧折外概率、保存的折内模型和日志 |
| `stability/models/{gray,rgb}.joblib` | 在全部采集组上拟合的模型，用于新采集组推理 |
| `stability/stability_summary.csv`、`stability/paired_comparisons.csv` | 各模型的三种子波动与配对比较 |
| `pipeline.json`、`stability/verification.json` | 完整流程状态与汇总文件的校验值 |

九牙研究使用 873 帧、127 组，每个种子和白光分支包含 9 折、27 个折内分类器、381 行组级折外预测。完整运行共 162 个折内分类器。每折的 PCA、标准化与分类器仅拟合训练牙。逐牙 Macro-F1 对牙等权。配对区间用固定种子对整牙进行 5,000 次自助抽样，训练种子另行评估同一数据与流程的数值可重复性。

`verify` 检查数据包清单、输入与输出指纹，载入各折模型复算折外概率，检查两份全数据推理模型，并重算分数和种子汇总。`--reference-study` 将样本数、四特征文件指纹、九牙分数和命中组数与随包基准对照。移动输入和输出后可在验收时另传 `--data` 和 `--features`。[主分析基准](../src/qlf_stage3/config/endpoint-reference.json) · [结果与解释范围](study-results.md)

两台电脑各自完成同一批数据的运行后，可将两份输出置于同一台电脑，运行 `qlf-stage3 compare runs/first runs/second --atol 0.000001`。程序逐分支、逐种子核对折划分、采集组和单帧预测阶段，并报告三类概率的最大绝对差。

旧版灰度单种子分析沿用 `qlf-stage3 run --data data/study --out runs/legacy-gray`，结果位于 `evaluation/`。原有的 `train`、`verify` 和两份旧版结果间的 `compare` 也可使用。[历史灰度基准](../src/qlf_stage3/config/study-reference.json)

## 给新采集组预测

每帧一行，用 `capture_id` 标记同一次拍摄同一视角的连拍。白光路径相对于输入 CSV，也可使用绝对路径。M1 需要白光，M2 需要四项 QLF 特征，M3 需要两者。无标签表的最小示例如下。

```csv
capture_id,rgb_path,huber_delta_f_pct,loss_area_gt30_pct,p10_intensity,G_texture_entropy
capture01,photos/white01.png,31.6,42.0,96,3.8
capture01,photos/white02.png,32.4,44.0,94,3.9
```

```sh
qlf-stage3 predict --model runs/demo/stability/models/rgb.joblib --input examples/acquisition-qlf.csv --arm M2 --out runs/example-qlf
qlf-stage3 predict --model runs/demo/stability/models/rgb.joblib --input runs/demo/stability/rgb/seed_42/sample_manifest.csv --data data/demo --out runs/example-paired
```

选用 `gray.joblib` 即可按训练时的灰度路径推理。每个模型对每组输出 `p_S0,p_S1,p_S2,pred_state`。概率并列时按 S0、S1、S2 顺序决定。报告的验证分数使用留出牙的折外结果。只加载自己生成或来源可信的 joblib 文件。

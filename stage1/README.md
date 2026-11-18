# stage1 / 终点一 · RGB 龋齿视觉基线

从两套公开白光口内照片建立龋坏牙检测和牙釉质龋分期基线。每套数据独立划分、训练和评价。跨来源实验用两套数据共有的「有龋／无龋」标签，测试源模型面对新来源时的表现。

| 数据、任务 | 方法和有效图数 | 原测试集结果 |
| --- | --- | --- |
| [Zenodo · 口内龋坏牙检测](https://zenodo.org/records/14827784) | COCO 预训练 YOLOv8n，单类 `caries`，320 像素，20 轮。训练／验证／测试 4,593／831／825 张 | mAP50 0.9183，mAP50–95 0.7209，精确率 0.8427，召回率 0.8566 |
| [CariesSpectra · 牙釉质龋三分类](https://data.mendeley.com/datasets/9jnf2jvghy/2) | ImageNet 预训练 ResNet18，冻结骨干、训练三分类头，224 像素，15 轮。训练／验证／测试 1,400／300／300 张 | 准确率 0.8733，宏 F1 0.8794 |
| 双向跨来源压力测试 | 对方留出的测试集，无目标域微调。ResNet18 合计早期和进展期概率。YOLOv8n 取图内最高框置信度 | CariesSpectra → Zenodo AUROC 0.5251，Zenodo → CariesSpectra AUROC 0.5775 |

两条同来源基线的结果分别来自检测和分类任务。跨来源分数接近随机区分的 0.5。原始指标汇总和数据来源详见 [REFERENCE.md](REFERENCE.md)。

## 环境

使用 Python 3.11。本项目原始运行和本次复验使用 Windows x64、Python 3.11.4、CPU。以下命令在仓库根目录执行。Windows 可用 `py -3.11` 代替 `python`，使用任意未被占用的输出目录。

```bash
python -m venv .venv
```

Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1` 激活，macOS/Linux 用 `source .venv/bin/activate`。Windows/Linux 可预先安装 CPU 版 PyTorch，节省 GPU 组件下载。

```bash
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r stage1/requirements.txt
```

## 1. 准备数据

从上游下载 Zenodo 的 `Dataset.zip` 和 CariesSpectra 的 v2 压缩包，放在 `stage1/data/downloads/`。使用 Zenodo 的 `Dataset.zip`，而不是发布页另列的 `Benchmarking Dataset.zip`。两个脚本也接受任意位置的归档。

```bash
python stage1/scripts/prepare_public_rgb.py --zip-path stage1/data/downloads/Dataset.zip --extract --out stage1/data/public_rgb/full
python stage1/scripts/prepare_cariesspectra.py --src "stage1/data/downloads/Caries-Spectra A dataset of Enamel Caries.zip" --out stage1/data/cariesspectra
```

- Zenodo 发布页写 6,313 张，本次使用的 `Dataset.zip` 有 6,265 张。准备入口合并乳牙龋与恒牙龋，按可解析的受试者编号划分，未能解析受试者编号的试拍图只用于训练。隔离了 16 张标签坐标异常的图片，剩 6,249 张。无有效龋框的合格图用于背景对照。
- CariesSpectra 共 2,000 张图，早期牙釉质龋 800 张、进展期 800 张、无牙釉质龋 400 张。缺少患者编号，按类别分层划分图片。按患者独立的泛化需要额外验证。
- 两套划分固定种子为 0。清单记录相对于各自数据目录的路径。原图不会提交到 GitHub。

复验所用本地归档的 SHA-256：

```text
4ea6f6be818702041d07409aa982ee1fb84e096bf53621475dba98e56551fc4c  Dataset.zip
e06a98bc8d556b4e76e61337436a475da65cf3fe1bd9fb2c88a188531ec71dc4  CariesSpectra v2 local ZIP
```

下载或重新打包方式可能改变归档的字节，复跑时应同时核对有效图数和划分。CariesSpectra 的类别文件夹也可以直接作为 `--src`。

## 2. 训练两个模型

```bash
python stage1/scripts/train_rgb_yolo.py --data stage1/data/public_rgb/full --out stage1/outputs/rgb_yolo_detection
python stage1/scripts/train_cariesspectra_3class.py --manifest stage1/data/cariesspectra/manifest.csv --out stage1/outputs/cariesspectra_3class
```

第一次训练需要下载上游 COCO YOLOv8n 和 ImageNet ResNet18 预训练权重。两次训练分别保存最优权重与测试指标。若已通过自己的运行获得权重，可用 `--eval-only --weights 路径` 在相同划分复测，另指定新的 `--out`。新训练的浮点结果可能随软件版本或硬件变化，历史指标用于对照。

## 3. 跨数据集评估

```bash
python stage1/scripts/run_external_validation.py --zenodo stage1/data/public_rgb/full --cariesspectra stage1/data/cariesspectra/manifest.csv --detector-weights stage1/outputs/rgb_yolo_detection/runs/train/weights/best.pt --classifier-weights stage1/outputs/cariesspectra_3class/best.pt --out stage1/outputs/external_validation
```

得到二分类指标、逐图预测、混淆矩阵和说明。YOLO 的输入置信度下限是 0.01，主要判定阈值为 0.25。ResNet18 主要协议整图缩放至 224×224，另做中心裁剪对照。测试集不用于选择训练模型或调整目标域参数。

## 路径与许可

脚本默认目录依据 `stage1` 所在位置确定。显式指定的命令行路径按执行时所在目录解析，可以绝对、相对或包含空格。准备好的数据可整个移动，移动后给脚本新的数据目录或清单位置。结果写入新的 `--out`。参见各脚本的 `--help`。

`stage1` 源码以 [AGPL-3.0](LICENSE) 发布。数据、第三方依赖和权重的来源与引用见 [REFERENCE.md](REFERENCE.md)。

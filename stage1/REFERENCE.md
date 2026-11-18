# 基准与发布范围

[`reference/`](reference/) 中是项目原始运行保存的汇总指标，用来核对复跑结果。它包含检测、三分类和双向跨数据集二分类三个 JSON。文件中的训练配置来自原始运行。三分类 JSON 的 `eval_only` 表示该 JSON 是使用已保存的最优权重复测生成的。

两套数据由各自的发布者分发。请从 [Zenodo 14827784](https://zenodo.org/records/14827784) 和 [Mendeley Data v2](https://data.mendeley.com/datasets/9jnf2jvghy/2) 获取并遵循对应使用条款。此项目中的 AGPL-3.0 许可作用于 `stage1` 的程序源代码。数据、数据标注和第三方预训练权重适用各自上游条款。下载依赖和权重请使用可信的官方来源。

数据源引用：

- Ahmed SMF, Ghori H, Khalid A, et al. *Annotated intraoral image dataset for dental caries detection*. Scientific Data 12, 1297 (2025). [论文](https://doi.org/10.1038/s41597-025-05647-9) · [数据 DOI](https://doi.org/10.5281/zenodo.14827784)
- Himel GMS, Islam MM, Hannan U. *Caries-Spectra: A dataset of Enamel Caries*, v2 (2023). [数据 DOI](https://doi.org/10.17632/9jnf2jvghy.2)
- [Ultralytics YOLOv8 文档及引用](https://docs.ultralytics.com/models/yolov8/) · [Ultralytics 许可](https://www.ultralytics.com/license) · [ResNet 原论文](https://doi.org/10.1109/CVPR.2016.90)

项目用于研究和教学的 RGB 基线复现。跨来源评估共享的是图像级有龋／无龋标签。三阶段分类的泛化还需要独立来源的三阶段标注。原始训练数据、图像级预测、本机路径和二进制权重留在本地，由下载和运行步骤获得所需输入与输出。

## 发布前复验

在 Windows x64、Python 3.11.4、CPU 环境中，从仓库外的目录调用脚本，使用含空格或中文的输入输出路径。重新解包并生成两套清单，比较了全部图片的 SHA-256 及原始划分。Zenodo 标签和隔离结果也逐项一致。使用原权重重评检测的 825 张图和分期的 300 张图，并跑完双向跨来源测试，三组汇总指标与本目录附带的历史指标一致。两个训练入口另外各完成一轮小样本运行，均得到权重和测试报告。完整轮数的发布验证使用既有权重复测，小样本练习不用于报告模型性能。

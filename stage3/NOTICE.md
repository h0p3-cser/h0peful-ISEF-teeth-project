# 来源与许可

本项目的 QLF 配准、牙根与牙冠掩膜、Huber 拟合、单图强度和纹理特征，以及历史探索命令，来源于 h0p3-cser/h0peful-ISEF-teeth-project 的 Stage 3。上游版本固定为 `cf2690f72b10f2254349e5f169e37ed75309e37d`。本目录保留上游 GNU Affero General Public License v3 正文及已公开的三幅历史阈值示例图。[上游 Stage 3](https://github.com/h0p3-cser/h0peful-ISEF-teeth-project/tree/cf2690f72b10f2254349e5f169e37ed75309e37d/stage3) · [许可证](LICENSE)

本版整合逐帧 255 参考、固定 30% 特征导出、同源灰度与原图 RGB 各自的白光／QLF／融合三模型、三种子九折留一牙验证、无标签推理，以及独立的数据包和测试。数值方案见[复现手册](docs/reproducibility.md)。

Torchvision ResNet-18 的 ImageNet-1K V1 权重由用户下载或随授权的本地数据包提供。第三方软件、外部权重及实验数据分别遵循其来源条款。合成示例由本项目代码按固定种子生成。[Torchvision 模型说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

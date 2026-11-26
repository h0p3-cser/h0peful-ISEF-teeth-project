# 数据、配对与特征

## 两种输入

`demo-data` 使用固定种子 7582 生成九颗合成牙，每颗一个视角、三个阶段、每阶段两次连拍。它生成 QLF 图、三通道彩色白光、解剖形状掩膜、已知的恒等配准及拍摄事件。灰度和 RGB 分支读取相同文件。合成图案用于软件复现。

`prepare-data` 面向持有原研究审核资料的人。在同一行命令中传入审核工作区、采集目录和一个新目录。`--weights` 可选，提供后将校验 SHA-256 并把权重随本地数据包保存。

```sh
qlf-stage3 prepare-data --workspace path/to/reviewed-workspace --catalog path/to/capture-catalog --weights path/to/resnet18-f37072fd.pth --out data/study
```

| 数据包内的位置 | 内容与用途 |
| --- | --- |
| `dataset.json` | 布局版本 1，类型为 `study` 或 `synthetic` |
| `workspace/Data/` | 照片、逐帧清单、审核牙体与牙冠掩膜、配准索引、C04 来源记录 |
| `workspace/Data_rgb_corrected/` | 白光文件的既有配对路径。主分析读取的文件逐一与项目原图的 SHA-256 核对，T005 的 35 张 S2 白光在清单中记为 `original_as_supplied`。 |
| `workspace/QLF_GPTadvice/QLF_GPTadvice_S0_S1_S2_C04.csv` | 审核配对、阶段、参考帧、掩膜及质控状态 |
| `workspace/stage3-author-transforms.zip` | 原研究 T004/T005 的 140 条复核配准及原作者照片校验清单 |
| `workspace/local-photo-bindings.csv` | 数据包相对路径、模态、图像 ID、实际文件 SHA-256 |
| `catalog/images.csv`、`catalog/pairs.csv` | 牙号版本、采集批次、几何版本及 RGB/QLF 的实际拍摄事件 |
| `weights/resnet18-f37072fd.pth` | 可选的离线 ImageNet V1 权重 |
| `files.sha256` | `prepare-data` 为复制的数据包生成的逐文件校验清单 |

原始照片路径沿用审核表，按 `workspace/Data/` 解析。训练清单中的 `rgb_resolved` 和 `qlf_resolved` 按数据包根目录解析。同一磁盘上的运行记录使用相对路径，跨磁盘迁移的覆盖参数见复现手册。研究数据包约 694 MiB，内含离线权重。接收原数据请联系数据持有者。代码仓库与公开构建不包含该数据包。

## 逐帧测量

每颗牙、每个视角沿用审核确认的 S0 参考帧。参考帧取 `a=1, b=0`。其余每张图，包括 S0 连拍，用配准后共用的牙根块做 Huber 拟合 `G_i = a_i × G_ref + b_i`。有效冠区像素按既有配准、牙冠、信号及质控规则筛选，绿通道要求 `16 ≤ G_i < 255`。

每帧有自己的参考亮度 `R_i = 255 × a_i + b_i`，像素损失为 `L_i = 100 × (R_i − G_i) / (255 × a_i)`。NPZ 内的 `loss_pct` 保存有符号浮点损失，无效位置为 NaN。`reference_xyxy` 记录在 S0 坐标中的裁剪范围。PNG 的色标截在 0–100%，精确复算请使用 NPZ。

| 逐帧特征 | 定义 |
| --- | --- |
| `huber_delta_f_pct` | 全部有效冠区像素的平均有符号损失，S0 也使用实测值 |
| `loss_area_gt30_pct` | `100 × count(L_i > 30) / count(valid)`，恰好 30 的像素不计入分子 |
| `p10_intensity` | 原生审核牙冠内绿通道第 10 百分位，单位为 8 位灰度级 |
| `G_texture_entropy` | 同一原生冠区内绿通道的 16 级对称共生矩阵熵，用水平和垂直相邻像素，以 2 为对数底 |

CSV 一行对应一张 QLF，另含 `sample_id,tooth_id,state_id,angle_id,repeat_rank,qlf_path,rgb_path`。面积在 0–100%，四项必须是有限数。默认按原实现从照片和审核掩膜复算强度与纹理，也可向 `export` 传 `--stage3-features` 和 `--single-image-features` 读取同批上游逐图表。旧 S0 纵向空值不会替代新协议的损失。

## 从帧到采集组

同时核对正式配对、图像 ID、字节哈希及校正方向后的像素。组键包括牙号、批次、视角、几何版本以及 RGB 和 QLF 两个采集事件。同视角的不同采集事件分别成组。组内所有有效配对同时用于两种模态，各自先平均特征再预测。缺失事件仅在相同图像哈希和同批次、视角、几何等信息给出唯一匹配时补回。逐帧表保留排除原因和旧版目录的对应标识。

原研究共 945 对，逐图质控排除 72 对。获准的 873 对全部进入三个模型；其中 854 帧属于三阶段都齐全的 41 个牙号–视角组合，用于纵向阈值曲线。正式数据中 945 张 QLF 均与原作者哈希一致。最终 873 对中，838 张本地白光与作者校正版的 JPEG 哈希不同。重放白光结果应使用本次数据包所登记的版本。

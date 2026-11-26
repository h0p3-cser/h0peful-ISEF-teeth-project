# 白光 RGB 牙根校正

`calibrate-rgb` 从已审核的配对清单逐张读取实际使用的彩色白光照片。在同牙、同视角已知 S0 参考图上标注未处理牙根，把当前图的颜色映射到参考图，写出原位 RGB PNG。独立质控在拟合前预留的连续牙根区域上完成。研究者已确认本批仅牙冠接触脱矿液。

## 在另一台电脑复现

安装项目后，把授权的 `data/study` 和历史运行 `runs/study` 复制到项目目录。历史运行需要 `measurements` 中的四特征 CSV 与核验 JSON，以及 `evaluation` 中的 `sample_manifest.csv` 和 `folds.csv`。照片、采集目录、原生白光掩膜、配对记录和参考身份由数据包提供。九牙研究使用的白光根区／牙冠区域与工程参数随源码和 wheel 分发。命令接受绝对或相对路径，输出路径必须是新目录。

```powershell
qlf-stage3 calibrate-rgb --data data/study --study-run runs/study --scope pilot --out runs/rgb_pilot
qlf-stage3 calibrate-rgb --data data/study --study-run runs/study --scope full --out runs/rgb_full
qlf-stage3 calibrate-rgb --data data/study --study-run runs/study --scope full --out runs/rgb_repeat
qlf-stage3 verify-rgb --data data/study --study-run runs/study --run runs/rgb_full --compare runs/rgb_repeat --encoder-smoke
```

默认每次处理一张原分辨率照片，`--workers` 可设置并行帧数。相同的原图、掩膜、源码和依赖可逐项比较两次运行的系数、块表、汇总与 PNG。`--encoder-smoke` 使用既有冻结的 ResNet-18，在同一对原始和校正图上各做一次前向读取。已有研究数据包包含权重。只持有源码时，可用合成图验证照片入口、拟合方向、数值与失败处理。

```powershell
python -m unittest discover -s tests -p test_rgb_calibration.py
```

## 参考、几何与颜色

- 每个牙号–视角优先使用有记录的白光 S0 参考，其次查已审配对的 QLF 参考。如均缺失，从最早的可用已知 S0 事件选取清晰、根区可见、最接近中位连拍序号的白光，平局选较小序号。单帧校正只接收原图身份、预先选定的参考与白光几何。专用参考取恒等变换；其他所有 S0、S1、S2 各自估计六个系数。
- 本批使用白光原生牙体掩膜。S0 白光上的根／冠多边形按固定绘图网格保存，并用参考照片 SHA-256 绑定。冠区加一个块宽缓冲，从拟合根区剔除。若原生掩膜圈到载台，项目会与白光图中央明亮前景作独立几何核对，交并比低于 0.15 才启用备用白光轮廓。最终每帧仍需通过根区重叠、独立几何标志、局部根轮廓／牙颈边缘核查。
- 优先使用归属于本参考的白光变换，其次用互相匹配的白光特征估计相似变换。每三个独立匹配点留一个核验。缺少根表面纹理时，可由分散、吻合的根轮廓及牙颈边缘支持。整段边缘中位错位达到一个块宽会拒绝该候选变换，局部错位达到一个块宽会剔除附近的取样块。
- 工作图长边最多 1600 像素，使用不重叠的 15 × 15 块，小图同比缩放。固定留出参考牙根长轴第三个四分区，外侧各隔一个块宽。两图均可用的像素至少占 80%。两张图都用原始像素排除背景、边界、深阴影、强反光和任一通道 ≤5 或 ≥250 的区域。
- 按 EXIF 方向显示，带可信 ICC 的图转 sRGB，本批有 EXIF sRGB 声明。在线性 sRGB 中逐通道做等权 Huber 拟合。自变量为当前块中位数，因变量为参考块中位数。`epsilon=1.35, alpha=0, max_iter=300, tol=1e-5`。每个通道当前与参考的 P90−P10 均须 ≥0.02。逐通道映射在原分辨率应用，先统计浮点越界，再裁切、编码、量化并写出无损三通道 PNG。
- 图像成功拟合需要至少 20 个拟合块，独立评价需要至少 6 个留出块。评价取原图、实际读回的 PNG、已知 S0 在相同预留像素上的编码 sRGB 块中位数，依次算 D65／2° Lab、CIEDE2000、明度与色度平面误差。每帧经块中位数汇总，每组再取非参考帧中位数，每牙取组中位数，最后各牙等权。综合色差和色度误差分别报告。

## 输出与模型入口

| 文件 | 用途 |
| --- | --- |
| `config.json`、`regions.json`、`anchors.csv` | 输入版本、软件和几何摘要、参考事件及文件／像素哈希、标注和选取理由 |
| `rgb_calibration.csv`、`corrected_rgb/` | 一帧一行，含白光和配对 ID、事件、源与输出路径、几何与拟合记录。成功帧输出原尺寸 PNG |
| `root_tiles.csv`、`qc_by_frame.csv` | 块坐标、角色、原始有效像素数、三通道的原始／参考／输出代表色、留出色差与整图／牙冠裁切 |
| `qc_by_event.csv`、`qc_by_tooth.csv`、`qc_s0_repeats.csv` | 采集组、等权牙及同一次参考拍摄的 S0 波动 |
| `qc_coverage.csv`、`qc_failure_cases.csv` | 按牙、角度、阶段、批次对账及可追溯的失败原因 |
| `paired_rgb_inputs.csv` | 原始和校正图共用的逐帧候选清单。按 `sample_id` 连接原有 QLF 四特征 CSV |
| `qc_report.md`、`figures/`、`validation.json` | 覆盖与效果、三张验收图及独立复算结果 |

`reference` 为恒等输出，`ok` 为拟合且可独立评价，`ok_no_holdout` 为有效拟合但留出块数不足，`failed` 记录具体原因并留空输出路径。病例变差仍计入独立评价。未来成对比较使用 `paired_eligible` 为真的同一批帧，移出专用参考与相同像素副本；严重越界的案例保留用于分析。相同采集组的原始 RGB、校正 RGB 和 QLF 三侧按相同成员求均值。

```python
from pathlib import Path

import pandas as pd
from qlf_stage3.modeling.vision import white_input
from qlf_stage3.rgb.color import RAW_MODE, CORRECTED_MODE

dataset = Path("data/study")
run = Path("runs/rgb_full")
pairs = pd.read_csv(run / "paired_rgb_inputs.csv")
sample = pairs.loc[pairs.paired_eligible].iloc[0]
raw = white_input(dataset / sample.source_path, RAW_MODE)
corrected = white_input(run / sample.corrected_path, CORRECTED_MODE)
```

两个入口均把整幅白光等比补边至 224 × 224 并保留三通道。校正 PNG 可供白光和融合分支共同读取。`raw_cache_key`、`geometry_cache_key` 和历史灰度输入的缓存标识分别记录处理模式。新的彩色分类实验应从冻结的共同成员清单出发，在每个留一牙训练折重新拟合 PCA、标准化器和分类头。原项目灰度结果单列为历史基线。

若迁移到新的采集数据，请提供该批已知 S0、正式配对与事件、原生白光掩膜及审核后的根／冠多边形，使用 `--regions` 指定新标注。本批的解剖位置和专用参考身份绑定其原始图像哈希。此方法衡量相对本牙已知 S0 的拍摄重复性；牙根效果能否推广到牙冠，应结合实际裁切和原尺度细节评价。

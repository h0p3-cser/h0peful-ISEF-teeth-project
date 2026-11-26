# 全图实物参照块 Huber 白光先导的早期试算

本页保留用于回放收到完整方案以前的试算。本次三牙研究从新版物理块入口执行。[正式实验与四步复现](rgb-patch-huber-pilot.md)

本命令逐帧读取真实配对白光，在当前帧与同牙同视角的既有 S0 参考之间配对实物区域，在三个通道分别拟合 Huber 直线。校正参数作用于整张原生分辨率图片。预设范围为 T006／0°、T008／0°、T007／45° 的全部连拍，包含历史牙根法失败帧。该命令独立于模型训练。

## 运行

从当前源码安装项目，在项目根目录运行。输入为授权数据包、已有九牙研究的配对清单和完整的历史牙根校正目录。

```sh
python -m pip install .
python -m qlf_stage3 material-pilot --data data/study --study-run runs/study --root-run runs/rgb_root_huber_full_20261126 --out runs/scene_pilot
python -m qlf_stage3 verify-material-pilot --data data/study --study-run runs/study --root-run runs/rgb_root_huber_full_20261126 --run runs/scene_pilot
```

输出目录须使用新路径。命令先按图像 ID 校验原始 JPEG、解码像素、S0 参考、既有白光掩膜和旧输出 PNG。包内九个采集场景均绑定代表帧 SHA-256，其他电脑可移动输入目录。应完整保存 `data/study/`、`runs/study/evaluation/sample_manifest.csv` 和所用的旧校正目录。公开源码不包含实拍牙片。

## 取样与评价

材料审阅记录在 `src/qlf_stage3/config/rgb_material_regions.json`。每次拍摄在自己的原始代表帧上定位实物，连拍内通过图像结构边缘核对位移。记录保留物块的原生多边形、所属实物、拟合或留出身份和去边后的有效像素数。互相配对的依据是可辨认的同一实物。暗处边缘、灰影、眩光及牙冠病变区均排除。

聊天要点下的首轮实现将未处理牙根留作独立评价，外部底板和 S0 可见的上沿实物用于拟合，圆形托盘整件留出。至少需要两件有效拟合材料和四个块，两端各通道第 10 到第 90 百分位的线性亮度跨度均须达到 0.02。每块去边 2 原生像素，RGB 三通道都在编码 5 到 250 开区间内的像素至少 48 个，占去边区域至少 80%。上沿的不同颜色和不同区域按同一个实物计算总权重。这些设置以及 Huber 的求解参数在 `rgb_material_huber.json` 中锁定，未取得原方案全文时须标明实现口径。

设当前块线性 sRGB 中位数为 x，S0 对应块中位数为 y。每个通道拟合 y = ax + b。各实物的总权重相同，同一实物内各块的权重相同。缺参照、通道跨度不足、连拍内位移未确认或拟合失效都会出现在逐图记录中。仅 S0 参考自身可输出恒等图。

独立托盘色差直接从原始图、旧 PNG、新 PNG 的同一批坐标读数，使用 D65、2°、CIEDE2000。旧方法预留的未处理牙根另作第二项检查。程序重新构造历史预留像素，核对原始、S0 和历史校正读数。三法评价限于三方均有读数的同帧交集。原图与旧法、新救回图与原图的两方比较单独保留。色差由块依次汇总为帧、采集、牙，不把连拍计作独立牙。

## 输出

| 文件 | 内容 |
|---|---|
| `report.md` | 三个问题的实测回答、分母和相同位置裁片 |
| `coverage.csv`、`failure_reasons.csv` | 各牙、视角、阶段的转归与失败原因 |
| `frame_audit.csv`、`material_blocks.csv` | 来源路径与散列、材料身份、各帧物块、参与拟合的权重、参数和裁切 |
| `heldout_blocks.csv`、`heldout_by_frame.csv`、`heldout_summary.csv` | 同物块的原图、旧校正和新校正色差，含无法评价的分母 |
| `corrected_rgb/` | 通过帧和 S0 参考的全分辨率无损 PNG |
| `figures/`、`I0654-detail.json` | 63 张定位图、I0654 原位裁片及原生坐标、亮度和局部对比 |
| `protocol.json`、`regions.json`、`provenance.json` | 完整参数、场景标注、依赖版本与输入校验值 |
| `status.json`、`validation.json` | 全部输出散列、独立重跑核验；后者由核验命令生成 |

I0654 的新增裁切首先从历史参数、原图及其原生牙冠掩膜独立重建。两个方法使用完全相同的分母，比较映射后、截断及取整前越出线性 [0,1] 的牙冠像素，排除原图三通道中已有 0 或 255 的像素。报告用完全相同的 416 × 416 窗口呈现三张图。互相缺失的校正输出不作填补。

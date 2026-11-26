# 专项实验入口

当前终点三使用原图白光和固定 30% 的 QLF 特征。完整主分析运行 `qlf-stage3 endpoint`，从已有测量表开始时运行 `qlf-stage3 stability`。下列脚本用于复核已命名的历史实验或维护对应数据，路径参数由调用者提供。

| 工具 | 所需材料与范围 |
| --- | --- |
| `run_seed_stability.py` | 源码目录下的三种子命令快捷方式。安装后的软件可直接使用 `qlf-stage3 stability`。 |
| `rgb_patch_huber_pilot.py` | 三牙全图实物块先导，对应包内命令 `patch-huber-pilot`。[先导操作](../docs/rgb-patch-huber-pilot.md) |
| `rgb_bstar_branch.py`、`rgb_qlf_comparison.py` | 复核历史中性实物 b* 校偏及其与 QLF 的对照，需要历史灰度运行和指定的审核区域。[输入与参数](../docs/rgb-bstar.md) |
| `train_original_rgb_qlf.py`、`compare_t005_rgb_source.py` | 来源审计。需要另行提供含相机指纹的原始采集清单和文件归档。当前研究依据项目数据包内的文件训练。 |
| `import_t005_original_rgb.py` | 维护原有双目录布局的入库工具。显式指定 `--project`、`--source` 与 `--receipt`。拿到已整理的 `data/study` 即可直接运行主项目。 |

牙根三通道 Huber、早期全图材料先导和冻结的实物块先导有各自的配置和结果目录。[牙根校正](../docs/rgb-calibration.md) · [早期材料先导](../docs/rgb-material-pilot.md) · [冻结先导协议](../docs/rgb-patch-huber-protocol-v1.md)

# 三牙白光实物参照块 Huber 先导

本实验执行 2026-11-26 版 1.0，从 v1.1 的配对样本清单选取 T006／0°、T008／0°、T007／45° 三个组合的全部连拍。每张原始图和 S0 按物理块身份配对，拟合原图到 S0 的三条带截距 Huber 直线。牙根不参与取样或配准。[方案原文](rgb-patch-huber-protocol-v1.md)

在拥有授权的真实数据、既有研究清单和历史牙根校正输出的项目根目录执行：

```powershell
$py = '.\.venv\Scripts\python.exe'
$entry = '.\experiments\rgb_patch_huber_pilot.py'
$trial = Join-Path 'runs' ('rgb_patch_huber_pilot_' + (Get-Date -Format yyyyMMdd_HHmmss_ffffff))
& $py -B -X utf8 $entry prepare --data data/study --study-run runs/study --baseline runs/rgb_root_huber_full_20261126 --out $trial
& $py -B -X utf8 $entry freeze --prepared $trial --regions (Join-Path $trial 'regions.jsonl')
& $py -B -X utf8 $entry run --prepared $trial --out (Join-Path $trial 'run_a')
& $py -B -X utf8 $entry run --prepared $trial --out (Join-Path $trial 'run_b')
& $py -B -X utf8 $entry verify --prepared $trial --first (Join-Path $trial 'run_a') --second (Join-Path $trial 'run_b')
```

已安装当前源码时也可使用 `qlf-stage3 patch-huber-pilot` 及相同四个子命令。对现有冻结实验，只需从 `run` 开始，输出指向新的空目录。跨电脑复制时保留 `data/study/`、研究与旧校正运行及冻结实验的相对目录布局。输入哈希、源码、库版本和实际 v1.1 Huber 参数保存在 `protocol-lock.json`。

## 从原物到冻结观测

这批实拍图的人工复核坐标按图像 SHA-256 绑定。区域以 EXIF 正向后的原生像素表示，支持同一物理块的多个取样窗和遮挡排除区。各窗有效像素求并集，重复像素仅计一次，同一表面只能贡献一对等权中位数。`prepare` 生成可复查的 `regions.jsonl`。如需修订手工标注，应在 `freeze` 前完成并新建一次实验。

每侧排除任一通道 ≤5 或 ≥250 的像素，有效像素至少 25 个。至少两块独立物理块，且两侧三通道的线性 P90−P10 都 ≥0.02 才可拟合。从三个有效块起，按预先分配的物理 ID 字典序尝试留出一整块，剩余的拟合块必须达标。无可行留出时可以全数拟合；只有两个拟合块时单列支持情况。C04 涉及设备与环境变化，缺少逐件身份依据的外物按 `identity_unverified` 保留记录。候选色值可用于描述观测，只有确认配对的物理块才进入资格判断。

## 实验档案

| 位置 | 可核查内容 |
|---|---|
| `pilot_manifest.csv`、`regions.jsonl`、`review/` | 原采集事件、原图/S0 的配对与哈希、逐帧身份依据、双图叠框和人工定位方式 |
| `masks/`、`frozen-frames.jsonl`、`protocol-lock.json` | 原生像素掩膜、物块角色、线性跨度和冻结哈希 |
| `run_a/frame_results.csv`、`run_a/patch_measurements.csv` | 每帧状态与原因、六系数、两侧块中位数、同块的原/旧/新色差和牙冠裁切 |
| `run_a/corrected_rgb/`、`run_a/preclip/` | 实际校正 RGB PNG 和全分辨率的 RGB 六位越界位置图 |
| `run_a/figures/I0654-native-comparison.png` | 固定原生 416 × 416 的原图、历史牙根法、新法与裁切叠图 |
| `report.md`、`verification.json` | 逐阶段覆盖与四格、分母、等权牙的配对增益、裁切分布、两次运行和合成校验结果 |

预留块必须完全独立于本帧的三个通道拟合。色差从保存后重新解码的 PNG 与冻结原图像素读取。每帧先做配对差，再取事件中位数、牙内事件中位数和可评牙等权均值。牙冠使用可靠的当前帧掩膜。缺少该掩膜仅影响裁切评价。

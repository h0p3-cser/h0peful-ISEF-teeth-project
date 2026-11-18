# h0peful-ISEF-teeth-project

ISEF 牙齿荧光（QLF）项目的代码仓库。

- [**stage1 / 终点一**](stage1/)：公开白光 RGB 口内图的龋坏牙检测、牙釉质龋分期和跨数据集评估。
- [**目标 2**](goal2-color-processing/)：从 QLF 牙片的 RGB 通道量化绿色和红色荧光信号。

## 目录

```
.
├── README.md
├── requirements.txt
├── stage1/                        终点一：公开 RGB 实验代码与复现说明
└── goal2-color-processing/        目标 2：颜色通道拆分、量化与加权
    ├── README.md                  逐脚本说明 + 完整公式
    ├── 产物说明.md                 每个产物文件对 RGB 做了什么
    ├── _common.py                 公共工具（读图/存图/归一/命名）
    ├── _metrics.py                统计内核（报告与 json 共用）
    ├── run_all.py                 一键跑完 01~15
    └── 01 … 15 脚本                一个处理方式一个脚本
```

## 快速开始

终点一的独立环境和复跑命令见 [stage1 / 终点一](stage1/README.md)。下面是目标 2 的运行方式。

```bash
git clone https://github.com/h0p3-cser/h0peful-ISEF-teeth-project.git
cd h0peful-ISEF-teeth-project
pip install -r requirements.txt

cd goal2-color-processing
python3 run_all.py 你的图.jpg -o out/S0
```

跑完会得到 15 个以处理方法命名的产物，例如
`S0_09_绿光加权_去光照_幂律p2_乘色度.png`、`S0_14_分析报告.txt`。

## 这个仓库在做什么

QLF 图像是绿光激发下的牙面照片，黄色偏绿是主色调。
把 RGB 拆开之后最反直觉的发现是：**R 通道几乎没有独立信息**
（`R = a·G + c` 就能解释 85–98% 的 R），整幅图的明暗来自打光而不是荧光。
真正有意义的量是两通道的**比值**和**残差**：

- `log2(R/G)` —— 哪里绿压过红、哪里红压过绿；
- `R − (aG + c)` —— 扣掉耦合律后多出来的红，也就是被褐变吃掉的绿。

详见 [`goal2-color-processing/README.md`](goal2-color-processing/README.md)。

## 数据说明

实验原图与 `out/` 里的生成结果**不入库**：

- 生成图由脚本重跑即可得到，没必要占仓库体积；
- 原图属于实验数据，未随仓库分发。

## 依赖

`numpy` `pillow` `scipy` `matplotlib`，见 `requirements.txt`。

# 参与开发

在 Python 3.11 虚拟环境中执行以下命令。Windows 与 Linux 的 CPU 版 PyTorch 安装方式见 [README](README.md)。

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
qlf-stage3 demo-data --out data/demo
qlf-stage3 endpoint --data data/demo --out runs/demo
qlf-stage3 verify --run runs/demo
python tests/check_endpoint.py --run runs/demo --data data/demo
```

修改损失、阈值或分组时，请提供能区分旧行为和新行为的数值用例。对原研究协议的改动请说明会影响哪项特征、样本范围或指标。报告复现问题时写明平台、Python 和软件版本、具体命令、公开演示能否重复问题，以及错误信息。请只提交合成样例、代码和汇总结果。工作目录中的原始研究数据、模型与逐帧报告由数据持有者管理。

发布时用 `python -m pip install build` 准备构建工具，再运行 `python tools/release.py`。该命令从同一版本生成可供 GitHub 使用的源码 ZIP、Python 源码发行包、wheel 和 SHA-256 校验清单。Windows 与 Linux 的持续集成会安装 wheel，运行单元测试、合成全流程和无标签预测核验。

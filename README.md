# Qwen Medical QA

一个面向学习和面试复盘的中文医疗领域问答实验项目。

项目目标不是直接声称“模型已经达到某个指标”，而是完整记录一条可复现的实验链路：

`基线推理 -> 数据清洗 -> QLoRA/SFT -> RAG -> DPO -> 服务化 -> 评测与开源`

> 免责声明：本项目仅用于机器学习工程学习、实验复现和面试演示，不提供诊断、治疗或其他医疗建议。真实数据必须经过合法授权、脱敏和许可证核验。

## 当前状态

- [x] 冻结项目目标与阶段验收标准
- [x] 添加最小样例数据和数据格式说明
- [x] 添加环境检查脚本
- [x] 添加 Qwen3-1.7B 基线推理脚本
- [x] 添加最小数据格式测试
- [x] 添加 v0 选择题 benchmark 和自动评测器
- [x] 在本机 RTX 4060 Laptop 8GB 上完成基线运行
- [ ] 完成公开数据集的来源、许可证和划分记录
- [ ] 完成 QLoRA、RAG、DPO 和服务化实验

## 项目结构

```text
qwen-medical-qa/
├── configs/                  # 实验配置
├── data/                     # 只放样例或经过许可的数据说明
├── outputs/                  # 本地实验输出，不提交大文件
├── scripts/
│   ├── check_environment.py   # 检查 Python、PyTorch、CUDA 和显存
│   ├── run_baseline.py        # 基线推理与延迟记录
│   └── summarize_baseline.py  # 汇总延迟、吞吐和显存
├── src/qwen_medical_qa/      # 后续抽取可复用模块
├── tests/                    # 自动化测试
├── reports/                  # 可提交的实验报告和环境记录
├── PROJECT_SPEC.md           # 项目规格与验收标准
└── requirements-baseline.txt
```

## 第一个实验：基线推理

建议使用 Python 3.10 或更高版本。先按照本机 CUDA 环境安装 PyTorch，再安装其余项目依赖：

```powershell
python scripts/check_environment.py
python -m pip install -r requirements-baseline.txt
```

运行最小样例：

```powershell
python scripts/run_baseline.py `
  --input data/sample_medical_qa.jsonl `
  --output outputs/baseline.jsonl `
  --max-new-tokens 128

python scripts/summarize_baseline.py `
  --input outputs/baseline.jsonl
```

运行 v0 选择题 benchmark：

```powershell
python scripts/run_baseline.py `
  --input data/benchmark_v0.jsonl `
  --output outputs/benchmark_v0_baseline.jsonl `
  --max-new-tokens 16 `
  --greedy

python scripts/evaluate_benchmark.py `
  --input outputs/benchmark_v0_baseline.jsonl `
  --output reports/benchmark-v0-baseline.json
```

这里的 accuracy 只表示选项字母是否匹配参考答案；它不能替代领域专家评审，也不能直接解释为医疗准确率。

脚本会记录每条样本的输入、模型输出、输入/输出 token 数、单条耗时和 tokens/s。第一阶段先不追求准确率数字，先确认模型、数据格式、生成参数和记录链路都能稳定运行。

Windows CUDA 环境的 PyTorch 安装命令会根据驱动和 CUDA wheel 选择单独确定，不把一个可能失效的固定命令写死在项目中。

## 面试复盘重点

完成每个阶段后，需要至少能回答：

1. 为什么先做基线，而不是直接微调？
2. 测试集如何避免数据泄漏？
3. QLoRA 解决了什么显存问题，代价是什么？
4. RAG 与微调分别解决什么问题？
5. 每个指标的定义、测试集、硬件和生成参数是什么？
6. 哪些失败实验被保留，为什么没有采用？

## 许可证与开源计划

仓库最终公开前，需要补充项目许可证、基础模型许可证、数据集许可证和第三方依赖说明。原始医疗数据、API 密钥和完整基础模型权重不提交到 GitHub；仓库应提供合法数据的下载说明、版本号和校验信息。

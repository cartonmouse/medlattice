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
- [x] 添加 CMB-Exam 数据准备脚本和来源记录
- [x] 完成 CMB-Exam v1 的清洗、固定子集和 SHA-256 报告
- [x] 完成 SFT messages 格式和 QLoRA smoke 链路
- [x] 完成 CMB-Exam v1 的正式 QLoRA 训练和基线对照评测
- [x] 完成 RAG v0 文档摄取、TF-IDF 检索、引用提示和检索评测
- [x] 完成 BGE 中文 embedding 检索器与 TF-IDF 对照
- [x] 完成基于 BGE 检索与 Qwen3-1.7B 的生成式 RAG demo 和引用评估
- [ ] 完成 vector DB、reranker、生成式 RAG、DPO 和服务化实验

## 项目结构

```text
qwen-medical-qa/
├── configs/                  # 实验配置
├── data/                     # 只放样例或经过许可的数据说明
├── outputs/                  # 本地实验输出，不提交大文件
├── scripts/
│   ├── check_environment.py   # 检查 Python、PyTorch、CUDA 和显存
│   ├── run_baseline.py        # 基线推理与延迟记录
│   ├── summarize_baseline.py  # 汇总延迟、吞吐和显存
│   └── prepare_cmb.py         # 下载、清洗并固定 CMB-Exam 数据子集
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

## 下一阶段：CMB-Exam 数据准备

当前正式数据候选为 [FreedomIntelligence/CMB 的 `CMB-Exam` 配置](https://huggingface.co/datasets/FreedomIntelligence/CMB)。数据来源、许可证声明、版本记录和处理策略见 [`data/sources/cmb-exam.yaml`](data/sources/cmb-exam.yaml)。原始数据和 `data/processed/` 下的处理结果不提交到 GitHub。

先安装数据处理依赖，再生成固定的小规模训练/验证/测试子集：

```powershell
python -m pip install -r requirements-data.txt

python scripts/prepare_cmb.py `
  --output-dir data/processed/cmb-exam-v1 `
  --train-limit 5000 `
  --val-limit 280 `
  --test-limit 1000 `
  --seed 42
```

第一版只保留单项选择题，先验证“下载 -> 规范化 -> 过滤 -> 固定子集 -> SHA-256 记录”的数据工程链路；多项选择题留到评测口径明确后再接入。当前上游 test split 不公开答案，因此 `val` 用于有标签评测，`test` 只作为无标签推理/格式检查数据，不能据此报告 accuracy。

## 下一阶段：SFT/QLoRA

SFT 数据使用与基线相同的 prompt 模板，输出 Qwen `messages` 格式；当前 assistant target 只保留参考答案字母，以便和 exact-match evaluator 对齐。QLoRA 配置见 [`configs/qlora.yaml`](configs/qlora.yaml)，训练脚本见 [`scripts/train_qlora.py`](scripts/train_qlora.py)。

```powershell
python -m pip install -r requirements-qlora.txt
python scripts/prepare_sft.py

# 先做 8 条样本 smoke training
python scripts/train_qlora.py `
  --config configs/qlora.yaml `
  --output-dir outputs/qlora-cmb-smoke `
  --max-train-samples 8 `
  --max-eval-samples 8

# 正式训练前去掉两个 --max-* 参数，并使用独立输出目录
python scripts/train_qlora.py --config configs/qlora.yaml
```

smoke 结果见 [`reports/qlora-smoke.md`](reports/qlora-smoke.md)。它只证明训练链路和 adapter 加载可用，不代表正式准确率提升。

正式训练使用 5,000 条 train、240 条 val，在本机 RTX 4060 Laptop 8GB 上完成 1 epoch QLoRA。基础模型在 val 上为 44.58%，QLoRA adapter 为 50.42%，详细配置、逐题迁移统计和复现命令见 [`reports/qlora-full.md`](reports/qlora-full.md)。这里的指标仍然只是 CMB-Exam 单项选择题的答案字母 exact-match，不能解释为医疗准确率。

## 当前阶段：RAG v0

RAG v0 先使用仓库内 5 条合成演示文档，完成“JSONL 文档 -> chunk -> TF-IDF index -> top-k 检索 -> 带 chunk 引用的 prompt -> Hit@k/MRR”链路。它不包含真实医疗资料，也还没有接入生成模型，因此结果只用于验证检索工程，不代表医疗问答效果。详细设计和结果见 [`reports/rag-v0.md`](reports/rag-v0.md)。

```powershell
python scripts/build_rag_index.py `
  --input data/rag_demo/knowledge_base.jsonl `
  --output outputs/rag-v0/index.json

python scripts/retrieve_rag.py `
  --index outputs/rag-v0/index.json `
  --input data/rag_demo/retrieval_benchmark.jsonl `
  --output outputs/rag-v0/retrieval.jsonl

python scripts/evaluate_rag.py `
  --input outputs/rag-v0/retrieval.jsonl `
  --output reports/rag-v0-retrieval.json
```

当前演示基准为 5/5 Hit@1、5/5 Hit@3、MRR@3 为 1.0。下一步是在同一评测口径下接入 embedding retriever，再接 Qwen 生成和引用忠实性评测。

Embedding v1 已接入 `BAAI/bge-small-zh-v1.5`，使用 Transformers、CLS pooling、512 维归一化向量和 CUDA 推理；在当前 5 条演示查询上与 TF-IDF 一样达到 Hit@1=1.0。详细配置、对照和局限见 [`reports/rag-embedding-v1.md`](reports/rag-embedding-v1.md)。模型缓存和 dense index 保持在本地，不提交到 GitHub。

## 当前阶段：生成式 RAG v1

在 BGE dense retriever 之后接入 Qwen3-1.7B 基础模型，完成“检索 -> 带 chunk 编号的上下文 prompt -> 生成回答 -> 答案词与引用自动评估”。本次使用 5 条合成演示问题、`top_k=3`、greedy decoding 和 `max_new_tokens=64`，Retrieval Hit@3、目标术语匹配、有效引用率和 grounded answer 自动代理指标均为 1.0。详细定义、显存、延迟和局限见 [`reports/rag-qa-v1.md`](reports/rag-qa-v1.md)。这些数字只证明端到端 demo 链路可运行，不代表医疗准确率或人工引用忠实性。

```powershell
python scripts/run_rag_qa.py `
  --model Qwen/Qwen3-1.7B `
  --index outputs/rag-embedding-v1/index.json `
  --input data/rag_demo/qa_benchmark.jsonl `
  --output outputs/rag-qa-v1/qwen3-base.jsonl `
  --device cuda `
  --embedding-batch-size 8 `
  --top-k 3 `
  --max-new-tokens 64 `
  --max-input-tokens 2048 `
  --greedy `
  --local-files-only

python scripts/evaluate_rag_qa.py `
  --input outputs/rag-qa-v1/qwen3-base.jsonl `
  --output reports/rag-qa-v1.json `
  --k 3
```

本阶段使用基础 Qwen，没有加载面向选择题字母输出的 QLoRA adapter；这样可以把开放式引用生成与选择题微调作为两个可解释的实验变量。下一步再接入合法授权知识库、vector DB、reranker，并扩展资料不足拒答和引用忠实性评测。

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

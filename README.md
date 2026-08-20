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
- [x] 完成 SQLite vector store、BM25 reranker baseline，并接入生成式 RAG
- [x] 完成资料不足拒答、高风险意图拦截和合成安全 benchmark
- [x] 完成基于授权 CMB-Exam train split 的本地闭域 RAG 实验（原始与派生数据不入库）
- [x] 完成统一 seed/确定性开关、运行元数据和预测漂移回归工具
- [x] 完成可选 FAISS HNSW 后端与 SQLite 对照实验
- [x] 完成可选 Transformer Cross-Encoder 神经 reranker 接口、测试和 CLI 接入
- [x] 完成神经 reranker 实际模型对比和 CMB 闭域消融
- [ ] 完成生产级 ANN/vector DB、真实授权知识库、DPO 和服务化实验

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
│   ├── prepare_cmb.py         # 下载、清洗并固定 CMB-Exam 数据子集
│   ├── prepare_cmb_rag.py     # 从本地 CMB train/val 构建闭域检索语料
│   └── run_cmb_rag.py         # 运行 CMB 闭域 RAG 选择题评测
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

## 当前阶段：Vector Store v0 与 Reranker baseline

在 BGE dense index 之上增加 SQLite 持久化 vector store，并实现“dense candidate top-5 -> BM25 reranker -> final top-3”两阶段检索。生成式 RAG 脚本现在支持 `--store` 和 `--reranker bm25`。5 条合成查询上，JSON dense、SQLite dense 和 SQLite+BM25 的 Hit@1、Hit@3、MRR@3 都为 1.0；BM25 改变了非目标候选顺序，但 toy benchmark 不足以证明 reranker 有真实收益。详细设计、端到端结果和局限见 [`reports/vector-store-reranker-v0.md`](reports/vector-store-reranker-v0.md)。

```powershell
python scripts/build_vector_store.py `
  --index outputs/rag-embedding-v1/index.json `
  --output outputs/vector-store-v0/index.sqlite

python scripts/retrieve_vector_store.py `
  --store outputs/vector-store-v0/index.sqlite `
  --input data/rag_demo/retrieval_benchmark.jsonl `
  --output outputs/vector-store-v0/bm25.jsonl `
  --device cuda `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --local-files-only

python scripts/evaluate_rag.py `
  --input outputs/vector-store-v0/bm25.jsonl `
  --output reports/vector-store-v0-bm25.json `
  --k 3
```

当前 SQLite 后端是透明的 reference implementation，查询采用 O(N) 全表扫描，不应包装成生产级 ANN/vector DB。下一步需要在真实授权知识库上扩大 benchmark，再比较 FAISS/Milvus/Chroma 和神经 reranker 的召回、延迟、内存与构建成本。

## 当前阶段：安全拒答 v0

新增 `--safe-mode`：在 Qwen 生成前检查高风险患者请求和检索置信度；命中时直接输出“资料不足，无法判断”，并跳过 `model.generate()`。9 条合成安全问题上，`dense min_score=0.6` 加高风险规则得到 decision accuracy=1.0、unsafe allow rate=0.0、false abstain rate=0.0。由于 benchmark 只有 9 条合成问题，这些数字不能解释为医疗安全保证。详细结果和失败样本见 [`reports/safety-v0.md`](reports/safety-v0.md)。

```powershell
python scripts/retrieve_vector_store.py `
  --store outputs/vector-store-v0/index.sqlite `
  --input data/rag_demo/safety_benchmark.jsonl `
  --output outputs/vector-store-v0/safety-bm25-threshold06.jsonl `
  --device cuda `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --min-score 0.6 `
  --local-files-only

python scripts/evaluate_safety.py `
  --input outputs/vector-store-v0/safety-bm25-threshold06.jsonl `
  --output reports/safety-v0-threshold06.json

python scripts/run_rag_qa.py `
  --model Qwen/Qwen3-1.7B `
  --store outputs/vector-store-v0/index.sqlite `
  --input data/rag_demo/safety_benchmark.jsonl `
  --output outputs/rag-qa-v2/safe-mode.jsonl `
  --device cuda `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --min-score 0.6 `
  --safe-mode `
  --greedy `
  --local-files-only
```

真实知识库接入前，先复制 [`data/sources/rag-knowledge-base.template.yaml`](data/sources/rag-knowledge-base.template.yaml)，补齐来源、版本、许可证、隐私和专家审核字段；模板本身不代表任何真实数据已经获准使用。

脚本会记录每条样本的输入、模型输出、输入/输出 token 数、单条耗时和 tokens/s。第一阶段先不追求准确率数字，先确认模型、数据格式、生成参数和记录链路都能稳定运行。

## 当前阶段：CMB-Exam 闭域 RAG v1

在没有可公开上传的真实医学知识库时，使用已有来源记录的 CMB-Exam `train` split 做本地闭域实验。检索文档是“题干 + 选项 + 参考答案”的已解答训练例题，不是临床指南；原始 CMB、处理结果、派生语料和向量索引都保留在本地忽略目录，不能上传到 GitHub。

本阶段先做泄漏控制：按题干与选项去重，并删除 train/val 重叠；val 只用于查询和答案评测。由于没有人工标注的相关文档 ID，不报告 Recall@k/Hit@k/MRR。完整数据处理、配置、迁移矩阵、延迟和局限见 [`reports/cmb-rag-v1.md`](reports/cmb-rag-v1.md)。

```powershell
python scripts/prepare_cmb_rag.py

python scripts/build_embedding_index.py `
  --input data/processed/cmb-rag-v1/knowledge_base.jsonl `
  --output outputs/cmb-rag-v1/index.json `
  --chunk-size 512 `
  --chunk-overlap 0 `
  --device cuda `
  --batch-size 16 `
  --local-files-only

python scripts/build_vector_store.py `
  --index outputs/cmb-rag-v1/index.json `
  --output outputs/cmb-rag-v1/index.sqlite

python scripts/retrieve_vector_store.py `
  --store outputs/cmb-rag-v1/index.sqlite `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --output outputs/cmb-rag-v1/retrieval.jsonl `
  --device cuda `
  --batch-size 16 `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --local-files-only

python scripts/run_cmb_rag.py `
  --model Qwen/Qwen3-1.7B `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --retrieval outputs/cmb-rag-v1/retrieval.jsonl `
  --output outputs/cmb-rag-v1/rag-base-val.jsonl `
  --max-new-tokens 4 `
  --max-input-tokens 2048 `
  --seed 42 `
  --greedy `
  --local-files-only

python scripts/evaluate_benchmark.py `
  --input outputs/cmb-rag-v1/rag-base-val.jsonl `
  --output reports/cmb-rag-v1-rag-base-val.json
```

本次匹配运行中，直接基线为 107/240（44.58%），dense-only RAG 为 111/240（46.25%），dense + BM25 为 115/240（47.92%）。此前普通 greedy 重跑出现过 112 与 115 两个结果；补充 `--deterministic` 后，8 条 smoke 两次预测完全一致，全量结果与最新 BM25 结果 240/240 条一致。详细结果见 [`reports/cmb-rag-v1.md`](reports/cmb-rag-v1.md) 和 [`reports/reproducibility-v1.md`](reports/reproducibility-v1.md)。

## 当前阶段：可复现性与预测漂移 v1

基线和 CMB-RAG 脚本现在支持 `--deterministic`，会记录 seed、Torch/CUDA、cuDNN 和 CUBLAS 配置；`scripts/compare_predictions.py` 可对两次 JSONL 结果做不暴露题目文本的漂移分析。

```powershell
python scripts/run_cmb_rag.py `
  --model Qwen/Qwen3-1.7B `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --retrieval outputs/cmb-rag-v1/retrieval.jsonl `
  --output outputs/cmb-rag-v1/rag-deterministic-val.jsonl `
  --max-new-tokens 4 `
  --max-input-tokens 2048 `
  --seed 42 `
  --greedy `
  --deterministic `
  --local-files-only

python scripts/compare_predictions.py `
  --left outputs/cmb-rag-v1/rag-base-val.jsonl `
  --right outputs/cmb-rag-v1/rag-deterministic-val.jsonl
```

`--deterministic` 不是跨机器和跨软件版本的绝对复现保证；当前仍需锁定环境版本、模型 revision，并补充多 seed 方差。完整说明见 [`reports/reproducibility-v1.md`](reports/reproducibility-v1.md)。

## 当前阶段：FAISS HNSW ANN v1

SQLite vector store 仍保留为 O(N) reference backend，同时增加可选的 FAISS `IndexHNSWFlat`。FAISS 使用归一化向量的 inner product，等价于 cosine similarity；chunk 元数据放在 `.faiss.meta.json` sidecar。可选依赖见 [`requirements-rag-ann.txt`](requirements-rag-ann.txt)，默认环境不强制安装。

```powershell
python -m pip install -r requirements-rag-ann.txt

python scripts/build_faiss_index.py `
  --index outputs/cmb-rag-v1/index.json `
  --output outputs/cmb-rag-v1/index.faiss `
  --hnsw-m 32 `
  --ef-construction 80 `
  --ef-search 64

python scripts/retrieve_faiss.py `
  --index outputs/cmb-rag-v1/index.faiss `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --output outputs/cmb-rag-v1/retrieval-faiss-bm25.jsonl `
  --device cuda `
  --batch-size 16 `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --local-files-only

python scripts/compare_retrievals.py `
  --left outputs/cmb-rag-v1/retrieval.jsonl `
  --right outputs/cmb-rag-v1/retrieval-faiss-bm25.jsonl `
  --top-k 3
```

在 4,984 条 CMB 文档、240 条查询上，FAISS HNSW 平均 search 约 0.2383 ms，SQLite scan 约 495.93 ms；top-3 完整排名 239/240 一致，FAISS+BM25 与 SQLite+BM25 的最终答案 240/240 一致。该延迟不包含 embedding 和生成，也不代表生产 SLA。详细结果见 [`reports/faiss-ann-v1.md`](reports/faiss-ann-v1.md)。

## 当前阶段：Transformer Cross-Encoder Reranker v1

在 dense candidate top-k 之后，项目现在提供可选的 Transformer Cross-Encoder 重排器。默认实现是 [`TransformerCrossEncoderReranker`](src/qwen_medical_qa/neural_reranker.py)，默认模型为 [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)。它对同一个 query 和每个 candidate passage 做成对打分，再按 raw logit 排序；raw logit 只能用于同一候选集内排序，不能当作概率。

BM25、神经 reranker 和不重排路径共用 `RerankedResult`，并且只有选择 `--reranker neural` 时才加载 Transformer 模型。模型下载不进入 Git 仓库，建议先设置本地缓存并使用 `--local-files-only` 做复现实验：

```powershell
$env:HF_HOME = "$PWD/.hf-cache"
$rerankerModel = "$PWD/models/bge-reranker-v2-m3"
python -m pip install -r requirements-rag-reranker.txt

python scripts/retrieve_faiss.py `
  --index outputs/cmb-rag-v1/index.faiss `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --output outputs/cmb-rag-v1/retrieval-faiss-neural.jsonl `
  --device cuda `
  --batch-size 16 `
  --candidate-k 10 `
  --top-k 3 `
  --reranker neural `
  --reranker-model $rerankerModel `
  --reranker-device cuda `
  --reranker-batch-size 8 `
  --reranker-max-length 512 `
  --local-files-only
```

模型文件已手动整理到项目 `models/bge-reranker-v2-m3`，并通过 Transformers 离线加载与项目封装的单样本前向校验。240 条 CMB val 上，FAISS+neural candidate top-10→top-3 的平均重排延迟为 303.97 ms、P95 为 466.81 ms，Qwen deterministic Accuracy 为 45.83%（110/240）；同一候选规模下 FAISS+BM25 为 47.92%（115/240）。neural 改变了 65/240 条最终预测，但没有带来收益，因此保留为失败但有面试价值的消融结果。详细结果见 [`reports/neural-reranker-v1.md`](reports/neural-reranker-v1.md)。

详细阶段记录见 [`reports/neural-reranker-v1.md`](reports/neural-reranker-v1.md)。

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

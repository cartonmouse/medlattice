# Vector Store v0 与 BM25 Reranker 对照报告

## 1. 实验目的

在 BGE dense retrieval 的 JSON index 之上增加一个持久化存储后端和两阶段检索流程：

```text
JSON dense index -> SQLite vector store
问题 -> BGE query embedding -> dense candidate top-5
     -> BM25 lexical reranker -> final top-3 -> RAG prompt / Qwen
```

本阶段先不安装新的第三方向量数据库，使用 Python 标准库 SQLite 做可解释的持久化 reference backend；重点是验证“存储层、检索层、重排层和生成层”之间的接口边界。

## 2. 数据边界

- 知识库：5 条仓库内人工编写的合成演示文档；
- 查询：5 条人工编写、带 `relevant_doc_ids` 的演示问题；
- embedding：`BAAI/bge-small-zh-v1.5`，512 维 L2-normalized vectors；
- dense candidate：`candidate_k=5`；
- final retrieval：`top_k=3`；
- reranker：BM25，`k1=1.2`、`b=0.75`；
- 生成模型：Qwen3-1.7B 基础模型，没有加载选择题 QLoRA adapter；
- 运行设备：RTX 4060 Laptop 8GB。

不包含真实患者数据，也不用于诊断、治疗或其他医疗建议。真实知识库必须经过来源、版本、许可证、脱敏和再分发范围核验。

## 3. 工程实现

### 3.1 SQLite vector store

`src/qwen_medical_qa/vector_store.py` 使用 SQLite 保存：

- `metadata` 表：模型名、query instruction、维度、pooling、距离度量和存储 dtype；
- `chunks` 表：chunk 元数据、来源、许可证和 `float32 little-endian` 向量 BLOB；
- `idx_chunks_doc_id`：按文档 ID 的普通索引；
- cosine similarity：向量先归一化，再使用 inner product；
- 当前查询实现为透明的全表扫描，复杂度为 O(N)，便于学习和单元测试，不是生产级 ANN 实现。

这样保留了从 JSON index 迁移到数据库后端的接口，同时没有把实验环境改造成难以回滚的依赖组合。后续可以把同一接口替换成 FAISS、Milvus 或其他 ANN/vector DB 后端。

### 3.2 BM25 reranker

`src/qwen_medical_qa/reranker.py` 实现无第三方依赖的 BM25 candidate reranker：

1. dense retriever 先召回较大的候选集；
2. BM25 在候选文本和标题上重新计算词项匹配分数；
3. 按 BM25 分数、原始 dense score 和 `chunk_id` 做确定性排序；
4. 输出同时保留 rerank score 和原始 retrieval score，方便分析。

它是 lexical reranker baseline，不是神经 cross-encoder。生产方案仍应在有标注的候选对上比较 BGE reranker、cross-encoder 或其他模型。

## 4. 检索对照结果

| 配置 | Hit@1 | Hit@3 | MRR@3 |
| --- | ---: | ---: | ---: |
| BGE JSON dense v1 | 1.0000 | 1.0000 | 1.0000 |
| SQLite dense candidate/final | 1.0000 | 1.0000 | 1.0000 |
| SQLite dense top-5 + BM25 top-3 | 1.0000 | 1.0000 | 1.0000 |

SQLite 后端与 JSON dense index 的目标排名一致，说明向量序列化、元数据恢复和 cosine 搜索没有改变结果。BM25 改变了非目标候选的顺序，但 5 条查询的目标文档本来就在 dense 检索第 1 名，因此这个 toy benchmark 无法证明 reranker 带来收益。

## 5. 生成式 RAG 对照

使用 SQLite + BM25 的结果接入 `scripts/run_rag_qa.py`，新增 `--store`、`--candidate-k` 和 `--reranker` 参数。

| 指标 | 结果 |
| --- | ---: |
| 查询数 | 5 |
| Retrieval Hit@3 | 1.0000 |
| 目标术语匹配率 | 1.0000 |
| 有效引用率 | 1.0000 |
| grounded answer 自动代理指标 | 1.0000 |
| 平均单条生成延迟 | 876.22 ms |
| 平均输入/输出 token | 251.2 / 28.0 |
| 总输出 token | 140 |
| 峰值 CUDA allocated/reserved | 3341.75 / 4240.00 MB |

`grounded answer` 仍然只是“目标术语匹配 + 引用编号属于检索结果”的自动代理指标，不是专家判断的事实性或临床安全性。SQLite/BM25 与此前 JSON dense 运行的延迟不能直接比较，因为 GPU 状态、模型加载和运行顺序不同。

## 6. 复现命令

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

python scripts/run_rag_qa.py `
  --model Qwen/Qwen3-1.7B `
  --store outputs/vector-store-v0/index.sqlite `
  --input data/rag_demo/qa_benchmark.jsonl `
  --output outputs/rag-qa-v2/sqlite-bm25-base.jsonl `
  --device cuda `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --greedy `
  --local-files-only
```

## 7. 面试复盘

可以这样回答：“我把检索拆成 candidate generation 和 reranking 两层。BGE 负责语义召回 top-5，BM25 只在小候选集上做词项匹配重排，最后取 top-3 进入 prompt。存储层先用 SQLite 把 chunk 元数据和归一化向量持久化，保存了模型、维度和度量信息；这样可以先验证接口和结果可复现，再替换成 ANN/vector DB。当前 5 条合成问题上三组指标都为 1.0，说明链路正确，但因为目标文档本来就是第一名，不能声称 BM25 有真实收益。”

高频追问：

**为什么不直接上 Milvus 或 FAISS？**

当前只有 5 个 chunk，使用生产级 ANN 会把部署依赖、索引参数和检索逻辑混在一起。SQLite reference backend 更容易验证序列化、元数据和结果一致性。真实知识库规模上升后，再用同一接口替换 ANN 后端并比较 Recall、延迟、内存和构建时间。

**为什么 dense retrieval 后还要 reranker？**

dense embedding 擅长较宽的语义召回，但候选之间可能存在词义相近、实体或数值差异；reranker 可以在较小候选集上使用更细粒度匹配。代价是额外延迟，所以需要用标注 benchmark 证明收益。

## 8. 局限与下一步

1. 扩大到有来源的医学资料和 paraphrase/跨段落/资料不足 benchmark；
2. 将 SQLite 全表扫描替换为 FAISS、Milvus、Chroma 或其他 ANN 后端并做资源对照；
3. 加入神经 cross-encoder reranker，评估 Recall@k、nDCG、延迟和显存；
4. 增加引用忠实性、拒答、安全和 prompt injection 评测；
5. 真实数据和外部向量数据库接入前，先确认许可证、隐私和部署范围。

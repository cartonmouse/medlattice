# RAG embedding v1：BGE dense retrieval 对照报告

## 1. 实验目的

在 RAG v0 的 TF-IDF 检索 baseline 之上，引入一个中文 embedding retriever，并在完全相同的知识库、查询基准、chunk 配置和评测脚本上做对照。

本阶段只验证“embedding 检索是否能被稳定接入并与 TF-IDF 公平比较”，还没有接入 Qwen 生成最终答案，也没有进行引用忠实性或医疗安全评测。

## 2. 模型与实现

- embedding model：`BAAI/bge-small-zh-v1.5`
- 调用方式：`transformers.AutoTokenizer` + `transformers.AutoModel`
- pooling：最后一层 hidden state 的 CLS token
- embedding dimension：512
- query instruction：`为这个句子生成表示以用于检索相关文章：`
- passages：不添加 query instruction
- 向量：L2 normalize 后使用 cosine similarity
- batch size：8
- max sequence length：512
- device：CUDA，RTX 4060 Laptop 8GB
- chunk：160 字符，overlap 32 字符

模型的官方卡提供了 Transformers 调用方式、CLS pooling 和中文检索指令说明，模型许可证标记为 MIT：[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)。模型权重只保存在本地 Hugging Face cache，不提交 GitHub。

## 3. 工程设计

新增 `rag_embedding.py`，将模型编码和纯 Python dense retriever 分开：

```text
文档 chunks
    │ TransformerTextEncoder.encode(is_query=False)
    ▼
512 维 normalized embeddings -> DenseRetriever -> index.json

用户 query
    │ TransformerTextEncoder.encode(is_query=True)
    ▼
query embedding -> cosine top-k -> citation-aware prompt
```

索引 JSON 记录模型名、pooling、最大长度、query instruction、chunk 元数据和向量，加载索引时会校验向量维度。`--min-score` 允许后续在验证集上选择“资料不足”阈值，而不是把一个未经验证的相似度阈值写死。

## 4. 与 TF-IDF 的公平对照

两组实验均使用：

- 相同的 5 条合成知识文档；
- 相同的 5 条带 `relevant_doc_ids` 的查询；
- 相同的 160/32 chunk 配置；
- 相同的 `top_k=3`；
- 相同的 `scripts/evaluate_rag.py`。

| 检索器 | Hit@1 | Hit@3 | MRR@3 |
| --- | ---: | ---: | ---: |
| TF-IDF v0 | 1.0000 | 1.0000 | 1.0000 |
| BGE dense v1 | 1.0000 | 1.0000 | 1.0000 |

BGE dense retrieval 改变了非目标文档的排序，例如“症状”查询的后续候选从 `体征`、`禁忌证` 变为 `鉴别诊断`、`体征`，但 5 条查询的目标文档仍全部排在第 1 名。由于基准只有 5 条合成查询，不能据此判断 dense retrieval 在真实语料上优于 TF-IDF。

## 5. 运行命令

首次运行需要下载模型；后续可以加 `--local-files-only` 使用本地缓存：

```powershell
python scripts/build_embedding_index.py `
  --input data/rag_demo/knowledge_base.jsonl `
  --output outputs/rag-embedding-v1/index.json `
  --model-name BAAI/bge-small-zh-v1.5 `
  --device cuda `
  --max-length 512 `
  --batch-size 8 `
  --chunk-size 160 `
  --chunk-overlap 32

python scripts/retrieve_embedding.py `
  --index outputs/rag-embedding-v1/index.json `
  --input data/rag_demo/retrieval_benchmark.jsonl `
  --output outputs/rag-embedding-v1/retrieval.jsonl `
  --device cuda `
  --batch-size 8 `
  --top-k 3 `
  --local-files-only

python scripts/evaluate_rag.py `
  --input outputs/rag-embedding-v1/retrieval.jsonl `
  --output reports/rag-embedding-v1-retrieval.json `
  --k 3
```

单元测试：22 个通过，其中包含 dense retriever 的排序、min-score 过滤和 index round-trip 测试。

## 7. 局限与下一步

1. 当前知识库仍是 5 条合成文档，必须换成来源、版本、许可证清晰的医学资料；
2. 当前只测 5 条查询，需扩展按主题划分的 retrieval benchmark；
3. 需要接入 FAISS/Chroma 等索引后端，比较构建时间、查询延迟和内存占用；
4. 需要加入 reranker，先 dense 召回 top-100，再重排 top-3；
5. 需要接入 Qwen 生成带引用回答，评估 answer correctness、citation faithfulness 和资料不足拒答；
6. 需要把模型缓存、模型许可证和 offline reproducibility 写入部署说明。

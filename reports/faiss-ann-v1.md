# FAISS HNSW ANN v1 阶段报告

## 1. 阶段目标

此前 SQLite vector store 是透明的 O(N) 全表扫描 reference backend。本阶段增加可选的 FAISS HNSW 后端，在同一个本地 CMB 闭域索引上比较构建结果、检索排名、搜索延迟和最终生成答案。SQLite 保留作为回归基线；FAISS 依赖不放入默认环境。

原始 CMB 数据、派生语料、向量索引和模型输出仍在 `.gitignore` 覆盖的本地目录，不上传 GitHub。CMB val 没有人工相关文档标签，因此本阶段不把相似度排序报告成 Recall@k。

## 2. 实现

- `FaissVectorStore`：使用 `IndexHNSWFlat`，对 L2-normalized vectors 使用 inner product，等价于 cosine similarity；chunk 元数据保存在 `.faiss.meta.json` sidecar；
- `scripts/build_faiss_index.py`：从已有 JSON dense index 构建 FAISS HNSW；
- `scripts/retrieve_faiss.py`：支持 FAISS dense candidate、BM25 reranker、score threshold 和现有 RAG 输出格式；
- `scripts/compare_retrievals.py`：比较 top-k 排名/集合重叠，不输出检索文本；
- `scripts/benchmark_vector_backends.py`：在同一批 query embeddings 上测量后端搜索时间和索引大小；
- 可选依赖固定为 `requirements-rag-ann.txt` 中的 `faiss-cpu==1.14.3`，本机实验安装在被忽略的 `.ann-deps` 目录；
- FAISS 集成测试在依赖可用时执行；完整测试为 42 项，全部通过。

## 3. CMB 索引配置

| 项目 | 值 |
| --- | ---: |
| 文档/向量数 | 4,984 |
| 向量维度 | 512 |
| 索引类型 | `IndexHNSWFlat` |
| HNSW M | 32 |
| efConstruction | 80 |
| efSearch | 64 |
| 相似度 | inner product on L2-normalized vectors |
| 构建时间 | 0.6219 s |

## 4. 检索对照

在同一批 240 条 CMB val query 上，先使用同一 BGE encoder 生成 query embeddings，再分别调用 SQLite 和 FAISS；下面的 search latency 不包含 BGE embedding 时间。

| 后端 | 索引+元数据大小 | 平均 search | P50 | P95 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SQLite O(N) scan | 20,803,584 bytes | 495.9296 ms | 475.7652 ms | 690.0672 ms | 778.9412 ms |
| FAISS HNSW | 15,724,993 bytes | 0.2383 ms | 0.2035 ms | 0.3762 ms | 0.7026 ms |

检索排名对照使用 top-3：239/240 条查询的完整排名完全一致，239/240 条 top-3 集合完全一致，平均集合重叠率为 0.997222。由于没有 retrieval gold label，这只能说明 FAISS 在当前参数下与 SQLite reference 的结果高度接近，不能等同于医学检索召回率。

## 5. 端到端生成对照

使用相同 Qwen3-1.7B base、seed=42、greedy、`--deterministic`、BM25 top-3：

| 配置 | 正确/总数 | Accuracy | Invalid | 与 SQLite 预测一致 |
| --- | ---: | ---: | ---: | ---: |
| SQLite dense + BM25 | 115/240 | 47.92% | 0 | — |
| FAISS HNSW + BM25 | 115/240 | 47.92% | 0 | 240/240 |

因此，当前 HNSW 参数没有改变最终答案；它主要改善了向量候选搜索的工程效率。最终延迟还会受 BGE 编码、BM25、prompt 长度和 Qwen generation 影响，不能把 0.2383 ms 直接当成完整服务延迟。

## 6. 复现命令

先安装可选依赖：

```powershell
python -m pip install -r requirements-rag-ann.txt
```

构建 CMB FAISS index：

```powershell
python scripts/build_faiss_index.py `
  --index outputs/cmb-rag-v1/index.json `
  --output outputs/cmb-rag-v1/index.faiss `
  --hnsw-m 32 `
  --ef-construction 80 `
  --ef-search 64
```

运行 FAISS + BM25 检索：

```powershell
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

python scripts/run_cmb_rag.py `
  --model Qwen/Qwen3-1.7B `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --retrieval outputs/cmb-rag-v1/retrieval-faiss-bm25.jsonl `
  --output outputs/cmb-rag-v1/rag-faiss-bm25-deterministic-val.jsonl `
  --max-new-tokens 4 `
  --max-input-tokens 2048 `
  --seed 42 `
  --greedy `
  --deterministic `
  --local-files-only

python scripts/compare_retrievals.py `
  --left outputs/cmb-rag-v1/retrieval.jsonl `
  --right outputs/cmb-rag-v1/retrieval-faiss-bm25.jsonl `
  --top-k 3
```

## 7. 结论与限制

当前 FAISS 是 CPU HNSW，不是 GPU FAISS，也没有在百万级语料或并发服务下压测。HNSW 的 M、efConstruction、efSearch 仍需在更大、带 gold label 的数据上调参；下一步才是 FAISS IVF/PQ、Milvus/服务化、端到端 P95、缓存和神经 reranker。

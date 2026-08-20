# CMB-Exam 闭域 RAG v1 阶段报告

## 1. 实验定位

本阶段使用已经有来源记录的 CMB-Exam `train` split，构建一个本地、闭域的“已解答训练例题检索”实验。它不是通用医学知识库，也不是临床问答系统；实验结果只用于学习 RAG 工程、数据泄漏控制和面试复盘。

原始 CMB 数据、处理后的 split、派生语料和向量索引均保留在本地忽略目录，不提交到 GitHub。来源与许可证声明见 [`data/sources/cmb-exam.yaml`](../data/sources/cmb-exam.yaml)，当前声明为 Apache-2.0，公开使用前仍需重新核验上游版本和许可证。

## 2. 数据构建与泄漏控制

输入是本地已经固定的 `data/processed/cmb-exam-v1`，本阶段只使用 `train.jsonl` 构建检索语料，使用 `val.jsonl` 做有标签评测。检索文档由“题干 + 选项 + 参考答案”组成，不包含训练样本的解析说明。

| 项目 | 数量或结论 |
| --- | ---: |
| train 输入行数 | 5,000 |
| val 输入行数 | 240 |
| train 内重复行移除 | 13 |
| train 与 val 的题干/选项重叠行移除 | 3 |
| 最终检索文档数 | 4,984 |
| val 唯一题干/选项数 | 204 |
| 检索 gold label | 不可用 |

去重键是规范化后的“题干 + 排序后的选项”。如果 train 中的键与 val 重叠，整条 train 记录不进入检索库；val 只作为查询和评测目标，不作为文档。由于当前没有人工标注的相关文档 ID，本阶段不报告 Recall@k、Hit@k 或 MRR，避免把相似度排序误称为检索正确率。

派生元数据见本地忽略文件 `data/processed/cmb-rag-v1/metadata.json`，其中记录了输入 split 的 SHA-256、去重数量和 `raw_data_committed_to_github=false`。

## 3. 系统链路

```text
本地 CMB train
  -> 题干/选项/答案规范化
  -> BAAI/bge-small-zh-v1.5 查询向量
  -> SQLite vector store 全表 dense scan
  -> candidate top-5
  -> 可选 BM25 reranker
  -> final top-3 已解答训练例题
  -> Qwen3-1.7B 只输出一个选项字母
```

主要配置：

- embedding：`BAAI/bge-small-zh-v1.5`，512 维归一化向量；
- vector store：SQLite 持久化，当前实现是 O(N) 全表扫描，属于 reference implementation；
- reranker：BM25，candidate top-5，最终 top-3；
- generation：`Qwen/Qwen3-1.7B` 基础模型，不加载 QLoRA adapter；
- decoding：greedy，`max_new_tokens=4`，`max_input_tokens=2048`，seed=42，关闭 thinking；
- 硬件：本机 RTX 4060 Laptop 8GB，CUDA 虚拟环境。

## 4. 结果

评测是 240 条 val 的单项选择题答案字母 exact-match；`invalid` 表示没有输出合法选项字母。

| 配置 | 正确/总数 | Accuracy | Invalid | 相对同环境直接基线 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B 直接基线 | 107/240 | 44.58% | 0 | — |
| SQLite dense-only RAG | 111/240 | 46.25% | 0 | +4 题，+1.67 pp |
| SQLite dense + BM25 RAG（本次重跑） | 115/240 | 47.92% | 0 | +8 题，+3.33 pp |

迁移矩阵进一步说明了收益来源并不稳定：

- 直接基线 -> dense-only：23 道错题改对，19 道对题变错，净增 4 道；
- 直接基线 -> dense + BM25：29 道错题改对，21 道对题变错，净增 8 道；
- dense-only -> dense + BM25：15 道改对，11 道变错，净增 4 道。

同配置 BM25 曾有一次运行得到 112/240（46.67%），随后在同一 CUDA 环境和公开模型标识下重跑得到 115/240。当前脚本只设置随机种子，没有强制所有 CUDA 算子的确定性，因此这些结果应视为初步单次结果；后续需要增加多 seed、确定性开关和置信区间后再作结论。

Qwen `generate()` 的单条生成延迟（不包含 BGE 编码和 SQLite 检索时间）如下：

| 配置 | 平均延迟 | P50 延迟 | 平均 prompt tokens |
| --- | ---: | ---: | ---: |
| 直接基线 | 80.73 ms | 77.68 ms | 138.17 |
| dense-only RAG | 141.75 ms | 141.81 ms | 532.80 |
| dense + BM25 RAG | 154.10 ms | 153.77 ms | 538.56 |

RAG 增加了上下文长度和检索链路，所以延迟上升是预期现象；当前延迟统计不是完整端到端服务 P95，不能直接作为生产 SLA。

## 5. 如何复现

以下命令默认已经准备好本地 CMB 处理结果和 Hugging Face 模型缓存。数据与模型缓存不随仓库上传。

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

去掉 BM25 的对照只需将 `--reranker bm25` 改成 `--reranker none`，并把输出文件名改为 `retrieval-dense.jsonl`。

## 6. 面试表述

可以这样概括本阶段：

> 我先把有许可证记录的 CMB-Exam train split 转成“已解答训练例题”检索库，做题干与选项级去重，并删除 train/val 重叠，避免把验证题直接检索出来。然后用 BGE 做 dense 召回，SQLite 做本地持久化，比较不加 reranker 和加入 BM25 的两阶段检索，最后让未加载 QLoRA adapter 的 Qwen3-1.7B 依据 top-3 例题只输出选项字母。在 240 条 val 上，直接基线是 44.58%，dense-only 是 46.25%，dense+BM25 本次重跑是 47.92%；但重跑出现 112 与 115 的波动，所以我把它定位为工程链路验证，而不是稳定的医学能力提升。

追问时要主动说明：

1. 这不是通用医学知识 RAG，因为检索文本本身含有 CMB 训练例题的参考答案；它验证的是闭域 exam-example retrieval，不能声称提升临床事实准确率。
2. 当前没有相关文档 gold label，因此不报告 Recall@k；要评估检索器，需要额外人工标注或可审计的相关性协议。
3. QLoRA 改变的是模型参数/行为，RAG 改变的是推理时可访问的外部上下文；本实验故意使用 base Qwen，以便把两个变量分开。
4. SQLite dense scan 不是生产级 ANN；下一步才是 FAISS/Milvus/Chroma、神经 reranker、端到端 P95、缓存和服务化。

## 7. 限制与下一步

- 语料是考试训练例题，不是经过专家审核的临床指南；不得用于诊断、治疗或患者决策。
- 训练文档包含参考答案，可能诱导模型复制相似题的答案或受到错误相似题干扰。
- val 有 240 行但只有 204 个唯一题干/选项组合，当前指标按行统计。
- 目前只有一个主要评测 split 和少量重跑，没有多 seed、人工评审、校准曲线或拒答质量评估。
- 继续前，应先补充更严格的离线 benchmark 与确定性配置，再决定是否进入生产 ANN/vector DB、神经 reranker、DPO 和服务化；真实医学知识库仍需独立的合法授权、脱敏和专家审核。

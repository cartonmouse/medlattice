# Transformer Cross-Encoder Reranker v1 阶段报告

## 1. 阶段目标

在已有 dense candidate 检索和 BM25 baseline 之后，增加一个可选的 Transformer Cross-Encoder reranker。目标是把“候选召回”和“候选精排”拆开，后续可以在同一份候选集上比较 BM25、神经模型和不重排三条路径。

本阶段只使用本地工程代码、测试和模型缓存实验；没有上传 CMB 原始数据、派生语料、向量索引或模型权重。

## 2. 实现

- `src/qwen_medical_qa/neural_reranker.py`：提供 `TransformerCrossEncoderReranker`，对 query-passage pairs 批量调用 `AutoModelForSequenceClassification`；
- 默认模型为 [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)，但只有显式选择 `--reranker neural` 才加载；
- passage 使用稳定格式 `title + newline + text`，结果沿用 `RerankedResult`，并保留原始 dense `retrieval_score`；
- 模型输出记录为 raw `logit`。它适合在一个候选集内排序，不当作跨模型概率；
- `score_fn` 是一个内部测试 seam，可以在不下载大模型的情况下验证排序、空候选、参数校验和异常路径；
- `retrieve_faiss.py`、`retrieve_vector_store.py` 和 `run_rag_qa.py` 均支持：
  `--reranker {none,bm25,neural}`、`--reranker-model`、`--reranker-device`、`--reranker-batch-size` 和 `--reranker-max-length`；
- 输出额外记录 reranker model、device、batch size、max length 和 score type，方便复现实验时核对一次运行到底使用了什么配置。

## 3. 本地验证

在包含可选 FAISS 依赖的环境中运行：

```text
Ran 46 tests in 2.493s
OK
```

另用不写 `.pyc` 的显式编译检查验证了 `src/`、`scripts/`、`tests/` 共 55 个 Python 文件。新增测试覆盖：

1. 神经分数可以改变 dense 候选顺序；
2. 输出保留原始 dense score；
3. 相同分数按 retrieval score 和 chunk id 稳定排序；
4. 空候选、非法 batch/max length、分数数量不一致均有明确行为；
5. title 为空时仍能只使用 passage text。

## 4. 实际模型权重状态

本机 CUDA 可用，PyTorch 为 `2.13.0+cu126`。由于 Hugging Face 大文件下载多次停滞，改为从浏览器下载后，将 6 个 Transformers 必需文件从 `D:\Chrome_Download` 复制到项目目录 `models/bge-reranker-v2-m3`；源文件保留，目标文件与源文件逐一 SHA-256 一致。主权重 `model.safetensors` 为 2,271,071,852 bytes。

模型已通过 `local_files_only=True` 离线加载校验：tokenizer 为 `XLMRobertaTokenizerFast`，模型为 `XLMRobertaForSequenceClassification`，单个 query-passage 输出形状为 `(1, 1)`；项目自己的 `TransformerCrossEncoderReranker` 也完成了单样本前向并得到 raw logit。该结果只证明文件完整和工程接口可运行，不代表 CMB 评测已经提升，因此本报告仍不填写 neural reranker 的 Accuracy、Recall 或全量延迟结论。

下一步运行 CMB 对比：

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
  --local-files-only
```

之后使用现有 `run_cmb_rag.py` 生成答案，再用现有评测器和 `compare_retrievals.py`/`compare_predictions.py` 对比 BM25 与 neural 结果。由于 CMB val 没有人工 retrieval gold label，不能把候选排序变化直接写成医学检索 Recall@k。

## 5. CMB 闭域 neural 对比结果

本次使用同一个 FAISS HNSW index、同一批 240 条 CMB val query、同一 BGE embedding、candidate top-10/final top-3、Qwen3-1.7B、seed=42、greedy 和 `--deterministic`。只替换 candidate 精排器：

| 配置 | 正确/总数 | Accuracy | Invalid | neural rerank mean | neural rerank P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAISS + BM25 | 115/240 | 47.92% | 0 | — | — |
| FAISS + neural | 110/240 | 45.83% | 0 | 303.97 ms | 466.81 ms |

FAISS+BM25 与 FAISS+neural 的 top-3 完整排名仅 4/240 条完全一致，top-3 集合完全一致 18/240 条，平均集合重叠率 0.501389。由于 CMB val 没有人工 retrieval gold label，这只能说明两个 reranker 的候选选择差异很大，不能报告 Recall@k。

逐题预测比较显示：175/240 条答案相同，65 条发生变化；两者同时正确 93 条、同时错误 108 条；BM25 正确而 neural 错误 22 条，BM25 错误而 neural 正确 17 条。neural 在这个闭域 exam-example retrieval 实验中没有超过 BM25，反而少 5 个正确样本。可能原因包括候选文本包含题干/选项/参考答案、通用 reranker 与 CMB 任务分布不匹配，以及没有使用 CMB 相关性标注做校准；这些是待验证假设，不是已证明的因果结论。

本次生成阶段的 Qwen 平均生成延迟为 202.46 ms（P95 256.41 ms），BM25 对照为 244.07 ms（P95 331.38 ms）；这部分受 prompt 长度和 GPU 状态影响，不能抵消 neural 额外的约 304 ms/query 重排成本。两种配置 invalid 均为 0。

## 6. 结论与限制

这次实验还说明：不能因为神经模型更复杂就默认效果更好。当前 neural reranker 的效果低于 BM25，且额外引入明显重排延迟；在有人工相关性标注、更多 candidate-k、领域负样本或 reranker 微调之前，不应声称“神经 reranker 提升了准确率”。

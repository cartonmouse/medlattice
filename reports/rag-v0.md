# RAG v0：文档摄取、检索与引用提示报告

## 1. 实验目的

在进入真实知识库、embedding 和向量数据库之前，先验证一条最小但可解释的 RAG 工程链路：

```text
知识库 JSONL -> 文档校验 -> chunk -> TF-IDF index -> top-k 检索 -> 引用上下文 -> RAG prompt
```

本阶段只评估“相关文档能否被召回”，还没有让 Qwen3-1.7B 生成最终答案。因此不能把本报告的结果描述为医疗问答准确率。

## 2. 数据边界

- 知识库：5 条仓库内人工编写的合成演示文档
- 主题：症状、体征、禁忌证、鉴别诊断、预后等术语定义
- 来源标记：`synthetic-demo`
- 许可证标记：`CC0-1.0`
- 检索基准：5 条人工编写查询，每条标注一个相关 `doc_id`
- 不包含真实患者数据，不用于临床决策

使用合成数据的原因是先验证数据结构、索引持久化、top-k 结果、引用编号和评测脚本；正式知识库必须在后续阶段重新核验来源、许可证、版本、脱敏和再分发范围。

## 3. 实现设计

### 3.1 文档与 chunk

知识库每行是一个 JSON 对象，至少包含 `doc_id`、`title`、`text`、`source` 和 `license`。文档按字符切分，默认 `chunk_size=160`、`chunk_overlap=32`，每个 chunk 保留：

- `chunk_id` 和 `doc_id`；
- 标题、正文、来源和许可证；
- 字符起止位置；
- 可扩展 metadata。

### 3.2 检索器

RAG v0 使用 Python 标准库实现稀疏 TF-IDF cosine retriever：

- 英文按词切分，中文按字符切分；
- 使用平滑 IDF；
- 查询和文档转成稀疏向量；
- 按 cosine similarity 降序排列；
- score 相同用 `chunk_id` 做确定性排序；
- 只返回正相似度结果，完全没有词表交集时返回空列表。

暂时不引入 embedding 模型和 vector DB，是为了先保留一个无额外依赖、容易单测、容易解释的 retrieval baseline。后续接入 embedding 后，可以在同一基准上比较召回变化，而不是把多个组件的变化混在一起。

### 3.3 引用上下文与提示词

`build_rag_prompt()` 会把检索结果整理成带 chunk 编号的上下文，并要求模型：

1. 只使用资料中明确出现的信息；
2. 资料不足时回答“资料不足，无法判断”；
3. 在回答后列出引用编号；
4. 不编造资料中没有的事实。

当前脚本会输出 prompt，但尚未接入生成模型，也没有声称引用一定真实支持答案；引用忠实性要在后续生成评测阶段单独验证。

## 4. 运行结果

配置：5 个文档、5 个 chunk、`top_k=3`、5 条带标签查询。

| 指标 | 结果 |
| --- | ---: |
| Hit@1 | 1.0000 |
| Hit@3 | 1.0000 |
| MRR@3 | 1.0000 |
| 单元测试 | 19 passed |

5 条查询全部在第 1 名召回对应文档。例如“患者自己感觉到的异常表现属于什么术语？”召回 `term-symptom#chunk-000`，score 为 0.586562，并在上下文中保留 `[term-symptom#chunk-000]` 引用编号。

这些数字只证明演示语料和查询基准上的链路正确，不能代表真实医疗语料的检索效果。演示语料很小，且查询与文档语义高度接近，不能据此选择生产检索器。

## 5. 复现命令

```powershell
python scripts/build_rag_index.py `
  --input data/rag_demo/knowledge_base.jsonl `
  --output outputs/rag-v0/index.json `
  --chunk-size 160 `
  --chunk-overlap 32

python scripts/retrieve_rag.py `
  --index outputs/rag-v0/index.json `
  --input data/rag_demo/retrieval_benchmark.jsonl `
  --output outputs/rag-v0/retrieval.jsonl `
  --top-k 3

python scripts/evaluate_rag.py `
  --input outputs/rag-v0/retrieval.jsonl `
  --output reports/rag-v0-retrieval.json `
  --k 3
```

代码位置：

- `src/qwen_medical_qa/rag.py`
- `scripts/build_rag_index.py`
- `scripts/retrieve_rag.py`
- `scripts/evaluate_rag.py`
- `data/rag_demo/knowledge_base.jsonl`
- `data/rag_demo/retrieval_benchmark.jsonl`
- `tests/test_rag.py`

## 7. 局限与下一步

1. 替换合成文档为有明确来源、版本和许可证的医学资料；
2. 增加去重、标题/段落 metadata、版本过滤和增量更新；
3. 引入 sentence embedding，比较 cosine retrieval 与 TF-IDF；
4. 评估 `Recall@k`、`MRR`、nDCG，并按主题切分查询；
5. 接入 Qwen3-1.7B 生成带引用回答；
6. 增加 citation faithfulness、资料不足拒答和安全边界评测；
7. 再考虑 FAISS、Chroma 或其他向量数据库，并记录组件、版本和资源开销。

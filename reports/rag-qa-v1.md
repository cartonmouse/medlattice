# 生成式 RAG v1：Qwen 回答与引用评估报告

## 1. 实验目的

在 RAG v0 的 TF-IDF baseline 和 Embedding v1 的 BGE dense retriever 之上，接入 Qwen3-1.7B，完成一条可观测的端到端链路：

```text
问题 -> BGE query embedding -> dense top-k 检索
     -> 带 chunk 编号的上下文 prompt -> Qwen3-1.7B 生成
     -> 答案目标词与引用编号自动评估
```

本阶段的重点是验证模型加载、检索上下文拼接、生成参数、显存/延迟记录和引用评测能否稳定工作。它不是临床问答能力评测。

## 2. 数据与边界

- 知识库：仓库内 5 条人工编写的合成演示文档；
- 查询：5 条人工编写、每条对应一个 `relevant_doc_id` 的演示问题；
- 主题：症状、体征、禁忌证、鉴别诊断和预后等术语定义；
- 来源标记：`synthetic-demo`；许可证标记：`CC0-1.0`；
- 生成模型：`Qwen/Qwen3-1.7B`，本次使用基础模型，没有加载 QLoRA adapter；
- 检索模型：`BAAI/bge-small-zh-v1.5`，使用已经构建的 512 维 dense index；
- `top_k=3`、greedy decoding、`max_new_tokens=64`、`max_input_tokens=2048`；
- CUDA 设备：本机 RTX 4060 Laptop 8GB。

演示数据不包含真实患者信息，不提供诊断、治疗或其他医疗建议。正式知识库必须重新核验来源、版本、许可证、脱敏和再分发权限。

## 3. 评测指标定义

脚本 `src/qwen_medical_qa/rag_qa.py` 将指标拆开记录：

- `retrieval_hit_rate_at_k`：相关 `doc_id` 是否出现在前 `k` 个结果中；
- `answer_contains_expected_rate`：模型答案是否包含 benchmark 标注的目标术语；
- `valid_citation_rate`：答案中至少有一个引用编号属于该问题实际检索到的 chunk；
- `grounded_answer_rate`：同时满足目标术语匹配和有效引用。

最后一个指标是自动化工程代理指标，不等同于人工判断的事实性、引用忠实性或临床安全性；后续需要专家标注和更严格的 citation entailment 评测。

## 4. 运行结果

### 4.1 质量指标

| 指标 | 结果 |
| --- | ---: |
| 查询数 | 5 |
| `top_k` | 3 |
| Retrieval Hit@3 | 1.0000 |
| Answer contains expected | 1.0000 |
| Valid citation | 1.0000 |
| Grounded answer（自动代理指标） | 1.0000 |

5 个问题都在第 1 名召回对应 chunk，Qwen 也都输出了目标术语并附上正确的 chunk 编号。例如症状问题输出 `[term-symptom#chunk-000]`，该编号存在于该问题的检索结果中。

### 4.2 性能与资源

| 指标 | 结果 |
| --- | ---: |
| 平均单条延迟 | 808.19 ms |
| 总生成 token 数 | 143 |
| 总体生成速度 | 35.39 tokens/s |
| 平均输入 token 数 | 246.8 |
| 平均输出 token 数 | 28.6 |
| 峰值 CUDA allocated | 3,341.23 MB |
| 峰值 CUDA reserved | 4,242.00 MB |

延迟是 5 条样本逐条 greedy 生成的记录，不包含首次模型权重加载时间；首次加载日志显示 Qwen 权重加载约 4 秒量级。性能数字受本机后台进程、GPU 温度和运行顺序影响，不能直接当作线上 SLA。

## 5. 复现命令

先构建或准备 Embedding v1 dense index，然后运行生成和评估：

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

首次运行需要模型缓存；为了复现实验，建议记录基础模型 revision、embedding 模型 revision、CUDA/PyTorch 版本和实际 GPU。模型权重和生成输出不提交到 GitHub，报告只保留轻量级指标与方法说明。

## 6. 面试复盘

可以这样解释：“我把 RAG 分成检索和生成两层。先用 BGE 把问题编码成向量，从 chunk index 取 top-k，再把每个 chunk 的稳定编号注入 prompt，要求 Qwen 只依据上下文回答并输出引用。评测时没有只看答案词是否出现，而是额外检查引用编号是否真的来自本次检索结果。当前 5 条合成演示问题的四项自动指标都是 1.0，但我会明确说明这只是 pipeline smoke/e2e 验证，不是医疗准确率或引用忠实性结论。”

高频追问：

**为什么没有直接把 QLoRA adapter 加到 RAG 生成里？**

当前 QLoRA adapter 是针对 CMB-Exam 选择题字母输出训练的任务适配器，和开放式引用回答的输出目标不同。本阶段先用基础 Qwen 建立可解释的生成 baseline，后续可以把“基础模型 RAG”和“适配器 RAG”作为独立对照实验，而不是混淆变量。

**为什么引用编号不等于引用忠实性？**

编号检查只能证明引用存在于检索上下文，不能证明上下文真的支持整句话，也不能发现模型对资料的错误改写。因此下一步要增加 claim-level entailment、人工抽样和资料不足拒答评测。

**为什么不报告医疗问答准确率？**

当前只有 5 条合成术语问题，且答案是人工设计的短目标词；样本规模和任务难度都不足以支撑医疗能力结论。真实医疗资料还需要合法授权、脱敏、专家标注和安全评审。

## 7. 局限与下一步

1. 用来源、版本和许可证清晰的真实或公开医学资料替换合成知识库；
2. 扩大 query benchmark，增加 paraphrase、跨段落、多跳和资料不足问题；
3. 接入 FAISS、Chroma 或其他 vector DB，比较构建时间、查询延迟和内存；
4. 增加 dense top-100 + reranker top-3 的对照实验；
5. 评估拒答率、引用覆盖率、claim-level faithfulness 和人工/专家评分；
6. 再比较基础模型 RAG 与 QLoRA adapter RAG，并明确各自的任务边界；
7. 最后再做 DPO、API 服务化、日志脱敏、限流和安全护栏。

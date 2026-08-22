# MedLattice：中文医疗问答模型工程最终实验总结

## 1. 项目定位

本项目是一个面向中文医疗问答模型工程实验的中文医疗问答实验系统，围绕 Qwen3-1.7B 逐步完成了基线、QLoRA/SFT、RAG、DPO、GRPO 和 Constitutional AI-inspired 安全层实验。

项目的目标不是构建临床系统，而是验证一条可解释、可评测、可复现的模型工程链路，并对没有提升的实验给出可审计的失败分析。仓库不包含真实患者数据、原始医疗数据、完整模型权重或 API 密钥。

## 2. 最终系统链路

```text
问题
  -> 文档/例题检索
  -> 带引用编号的上下文 prompt
  -> Qwen3-1.7B 生成候选回答
  -> Constitutional deterministic critic
  -> 规则 hard gate / 安全模板兜底
  -> 最终回答与审计字段
```

模型训练和推理阶段保持变量隔离：CMB 选择题使用 QLoRA/SFT、DPO 和 GRPO 做训练对照；开放式引用生成使用基础 Qwen 建立 RAG baseline；安全层单独评估模型批评、模型修订和确定性兜底。

## 3. 主要实验总表

### 3.1 CMB-Exam 闭域选择题

评测集为隔离的 240 条 CMB-Exam val，指标是选项字母 exact-match accuracy。它不是医疗问答准确率。

| 阶段 | 结果 | 相对 SFT | 结论 |
| --- | ---: | ---: | --- |
| Qwen3-1.7B 直接基线 | 107/240（44.58%） | - | 建立同条件基线 |
| QLoRA/SFT | 121/240（50.42%） | - | 比基线多 14 题，+5.84 pp |
| DPO v1：合成随机负例 | 121/240（50.42%） | 0 题 | 训练目标被优化，但下游未提升 |
| public DPO v2：英文偏好数据 | 118/240（49.17%） | -3 题 | 出现轻微中文任务负迁移 |
| hard-negative DPO | 120/240（50.00%） | -1 题 | 混淆边界负例仍未带来泛化收益 |
| GRPO v0：可验证奖励 | 122/240（50.83%） | +1 题 | 单 seed、窄任务上的轻微正向结果 |
| CMB train-only dense + BM25 RAG | 115/240（47.92%） | 不直接比较 | 检索链路有效，但不是通用医学知识 RAG |

关键结论：SFT 是本项目在 CMB exact-match 指标上的主要收益来源；DPO 和 GRPO 的价值主要体现在训练链路、目标函数、审计方法和失败分析，而不是已经证明了稳定的医学能力提升。

### 3.2 生成式 RAG 演示

使用 5 条仓库内合成知识库和 5 条合成查询，BGE dense retrieval + Qwen3-1.7B 生成：

| 指标 | 结果 |
| --- | ---: |
| Retrieval Hit@3 | 1.0000 |
| Answer contains expected | 1.0000 |
| Valid citation | 1.0000 |
| Grounded answer 自动代理指标 | 1.0000 |
| Qwen + Constitutional hard gate smoke | 5/5 最终规则通过 |
| hard gate fallback | 0/5 |

这些数字只证明小规模演示链路能够运行，不能解释为真实医疗准确率、事实性或临床引用忠实性。

### 3.3 Constitutional AI-inspired v1.1

基准由 24 个场景族 × 5 个变体组成，共 120 条；违规和安全样本各 60 条；dev/holdout/challenge 为 60/40/20，同一场景族不跨 split。

| 指标 | 结果 |
| --- | ---: |
| Model critic accuracy | 0.8833 |
| Model critic precision | 0.8108 |
| Model critic recall | 1.0000 |
| Model critic F1 | 0.8955 |
| False-abstain rate | 0.2333 |
| Model revision 独立修复 | 43/60（71.67%） |
| Hard gate fallback | 20/120 |
| 最终规则违规 | 0/120 |

分层 revision success rate：dev 为 83.33%，holdout 为 85.00%，challenge 仅为 10.00%。这说明模型比较擅长发现风险，但在隐私和紧急场景中不一定能稳定生成合格的安全回答，因此保留确定性 hard gate 是必要的。

最终 0/120 是“模型 critic/revision + 确定性规则兜底”的组合结果，不能归因于模型 revision 单独能力，也不能解释为临床安全率。基准标签是工程合成标签，没有专家标注。

## 4. 最小复现演示

### 4.1 Qwen + dense RAG + Constitutional hard gate

前提是已经准备好本地 Qwen、BGE 和 `outputs/rag-embedding-v1/index.json`。运行：

```powershell
$env:HF_HOME = "C:\\path\\to\\medlattice\\.hf-cache"
$env:PYTHONPATH = "$PWD/src"

python scripts/run_rag_qa.py `
  --model Qwen/Qwen3-1.7B `
  --index outputs/rag-embedding-v1/index.json `
  --input data/rag_demo/qa_benchmark.jsonl `
  --output outputs/project-final-demo/qwen-rag-gated.jsonl `
  --device cuda `
  --top-k 3 `
  --max-new-tokens 64 `
  --max-input-tokens 2048 `
  --safe-mode `
  --constitutional-gate `
  --seed 42 `
  --greedy `
  --local-files-only

python scripts/evaluate_rag_qa.py `
  --input outputs/project-final-demo/qwen-rag-gated.jsonl `
  --output outputs/project-final-demo/rag-metrics.json `
  --k 3
```

`--constitutional-gate` 默认关闭，不影响历史 RAG 结果；打开后，每条 Qwen 原始回答会保留在 `raw_answer`，规则审查、是否兜底和最终审查结果保存在 `constitutional` 字段中。

### 4.2 安全 benchmark

```powershell
python scripts/run_constitutional_v1.py `
  --input data/constitutional/benchmark-v1.1.jsonl `
  --output outputs/constitutional-v1.1/results.jsonl `
  --summary outputs/constitutional-v1.1/summary.json `
  --candidate-source benchmark `
  --local-files-only `
  --greedy `
  --load-in-4bit
```

## 5. 项目完成标准

当前版本满足以下项目完成标准：

1. 固定 benchmark、数据来源、划分方式和泄漏控制规则已经记录；
2. 基线、SFT、DPO、GRPO、RAG 和安全层都有代码、配置、结果和限制说明；
3. 关键结果能够通过 JSON/Markdown 报告复核，而不是只保留一句准确率；
4. RAG 生成链路能够输出引用，安全层能够执行审查和兜底；
5. 单元测试和 UTF-8 文件校验通过；
6. GitHub 仓库不上传真实医疗数据、模型权重、缓存和密钥；
7. 成功实验、失败实验、指标口径和下一步方向均有记录。

完成这些条件后，继续增加 PPO、GRPO seed、DPO 数据或服务化属于二期研究，不是当前项目的必需条件。

## 6. 限制

- CMB exact-match 只衡量选择题选项是否匹配，不能代表开放式医疗能力；
- CMB train-only RAG 检索的是已解答考试例题，不是经过专家审核的医学知识库；
- 生成式 RAG 和 Constitutional benchmark 规模较小，主要用于工程 smoke 和失败分析；
- v1.1 安全标签是合成标签，没有医生或领域专家复核；
- 当前没有真实用户部署、服务 SLA、多 seed 置信区间或临床安全结论；
- 公开使用真实医学资料前，仍需单独核验来源、许可证、脱敏和再分发权限。

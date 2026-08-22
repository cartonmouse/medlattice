# 安全拒答 v0：资料不足与高风险意图评测报告

## 1. 实验目的

在生成式 RAG 之后增加一个透明的安全 gate，验证两个基本行为：

1. 检索不到足够资料时，不把问题交给生成模型自由发挥；
2. 对患者个体诊断、治疗、用药和剂量等高风险请求，直接输出“资料不足，无法判断”，跳过 Qwen 生成。

本阶段只实现工程 baseline，不声称它是医疗安全系统，也不替代医生、专家审核或正式安全评测。

## 2. 策略设计

`src/qwen_medical_qa/safety.py` 包含两个可解释规则：

- `high-risk-intent`：匹配少量患者个体/诊断/治疗/用药/剂量等高风险意图；
- `no-retrieved-context` 或 `below-retrieval-threshold`：没有检索结果或 dense cosine 低于阈值时拒答。

本次演示使用 `dense min_score=0.6`。这个阈值只根据当前 5 条范围内问题和 4 条拒答问题的合成分布校准，不能直接用于真实知识库；上线前必须在独立验证集上选择阈值，并评估误拒答和误放行成本。

safe-mode 的行为是：

```text
问题 -> 检索/候选重排 -> safety gate
                         ├─ abstain -> 固定拒答，不调用 Qwen generate
                         └─ allow   -> 带引用上下文进入 Qwen
```

## 3. 合成 benchmark

- 总问题数：9；
- 范围内术语问题：5；
- 高风险患者请求：2；
- 无关/资料外问题：2；
- 知识库：仍是 5 条 `synthetic-demo` 合成文档；
- 检索：SQLite dense candidate + BM25 reranker。

## 4. 结果

### 4.1 不使用 dense 阈值

只使用高风险规则和“有无检索结果”时：

| 指标 | 结果 |
| --- | ---: |
| 决策准确率 | 77.78% |
| unsafe allow rate | 50.00% |
| false abstain rate | 0.00% |

天气问题和 Python 问题被语义检索召回了无关医学 chunk，说明“有检索结果”不等于“资料真的相关”。

### 4.2 `min_score=0.6`

加入 dense 相似度阈值后：

| 指标 | 结果 |
| --- | ---: |
| 决策准确率 | 100.00% |
| unsafe allow rate | 0.00% |
| false abstain rate | 0.00% |

这是 9 条合成问题上的结果，不能外推到真实医学语料。尤其不能把 0% unsafe allow 写成系统安全保证。

### 4.3 safe-mode 生成验证

`scripts/run_rag_qa.py --safe-mode` 的实际行为：

- 5 条范围内问题调用 Qwen 生成；
- 4 条高风险/无资料问题直接输出固定拒答；
- 4 条拒答样本的 `generation_skipped=true`，没有进入 `model.generate()`；
- 5 条生成答案中有 1 条把引用编号输出成无方括号形式，暴露了引用格式仍需后处理和严格评测。

因此安全拒答 gate 已经独立于生成链路生效，但引用格式和引用是否支持答案仍要单独处理。

## 6. 局限与下一步

1. 扩展高风险意图的 paraphrase、多轮和对抗样本；
2. 建立独立的资料不足验证集，不能用同一批数据调阈值和报告结果；
3. 增加 citation format validator、claim-level entailment 和人工抽样；
4. 引入模型级安全分类器或经过审核的 guardrail，并进行误拒答/误放行成本分析；
5. 真实知识库接入前完成来源、许可证、隐私和专家审核清单。

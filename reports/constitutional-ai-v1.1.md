# Constitutional AI-inspired v1.1：120 条分层安全 benchmark 报告

## 1. 为什么扩展数据

v1 的 12 条样本适合验证代码链路，但 8 条违规样本作为 revision 分母过小，无法说明模型在改写、隐私和高风险语境上的稳定性。本阶段不引入真实医疗数据，而是用可复现的场景族和变体扩展到 120 条，并把安全性拆成 critic、revision、hard gate 和延迟几个指标。

这仍然是合成 benchmark：标签由项目构造规则指定，没有医生或领域专家独立标注。因此结果可以用于工程回归和面试展示，不能解释为临床安全率。

## 2. 数据设计

- 24 个场景族，每族 5 个变体，共 120 条；
- 60 条期望初始违规、60 条期望安全；
- dev/holdout/challenge 分别为 60/40/20，且同一场景族的 5 个变体不会跨 split；
- 覆盖个人诊断、用药、剂量、手术、治疗、证据不足、绝对化承诺、隐私索取、安全拒答、资料支撑定义和紧急边界；
- 每条记录保留 `family_id`、`variant_index`、`split`、`category`、`expected_initial_violation`、`expected_final_safe` 和 `label_source`；
- benchmark SHA-256：`9116beb647fe15cad930e621690f43b33e3193814d35082e42071d96d4dbce1d`。

生成命令：

```powershell
python scripts/prepare_constitutional_benchmark.py
```

生成器的结构测试验证了 120 条总量、60/40/20 split、60/60 标签平衡、24 个场景族以及场景族不跨 split。

## 3. 评测配置

沿用 v1 的 model-in-the-loop 链路：

```text
受控候选回答 → Qwen JSON critic → Qwen revision → deterministic hard gate
```

配置为 `Qwen/Qwen3-1.7B`、4-bit NF4、greedy、`enable_thinking=false`、本地文件加载。120 条受控候选需要 120 次 critic 和 120 次 revision 生成；候选本身不再额外调用模型。完整逐条结果保存在本地忽略目录 `outputs/constitutional-v1.1/results.jsonl`。

## 4. 总体结果

### 4.1 模型 critic

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 120 |
| 期望初始违规 | 60 |
| 模型判定违规 | 74 |
| JSON 解析失败 | 0 |
| True positive | 60 |
| True negative | 46 |
| False positive | 14 |
| False negative | 0 |
| Accuracy | 0.8833 |
| Precision | 0.8108 |
| Recall | 1.0000 |
| F1 | 0.8955 |
| 安全样本 false-abstain rate | 0.2333 |

模型 critic 没有漏掉 60 条构造的违规，但把 14 条安全回答判成不通过，说明它偏保守，尤其容易把安全边界和紧急提醒理解成需要修订的风险信号。

### 4.2 Revision 与 hard gate

| 指标 | 结果 |
| --- | ---: |
| 期望违规样本数 | 60 |
| model revision 单独修复 | 43/60（71.67%） |
| model revision 后仍违反规则 | 20/120 |
| hard gate fallback | 20/120 |
| hard gate 后最终规则违规 | 0/120 |
| hard gate 后最终安全通过率 | 1.0000 |

这里的 `43/60` 使用 benchmark 的期望违规标签作为分母，而不是使用规则初筛命中的 40 条。规则初筛自身只捕获了 38/60 条违规，recall=`0.6333`，说明扩展的同义改写确实测出了关键词规则的漏检。

## 5. 分层结果

| Split | 样本 | Critic accuracy | Critic precision | Critic recall | Critic F1 | False-abstain | Revision success | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 60 | 0.9500 | 0.9091 | 1.0000 | 0.9524 | 0.1000 | 25/30（83.33%） | 7 |
| holdout | 40 | 0.8500 | 0.7692 | 1.0000 | 0.8696 | 0.3000 | 17/20（85.00%） | 4 |
| challenge | 20 | 0.7500 | 0.6667 | 1.0000 | 0.8000 | 0.5000 | 1/10（10.00%） | 9 |

challenge split 只包含未参与 prompt 调整的隐私和紧急边界场景，模型 revision 明显变弱；这比只报告整体 71.67% 更能说明当前系统的薄弱环节。hard gate 在三个 split 上都把最终规则违规降为 0，但这不代表模型本身已经学会了这些安全行为。

## 6. 规则基线与延迟

在同一 120 条数据上，deterministic v0 规则初筛的结果为：accuracy=`0.8000`、precision=`0.9500`、recall=`0.6333`、F1=`0.7600`、false-abstain rate=`0.0333`。它误报较少，但漏掉了 22 条改写后的违规；模型 critic 提高了 recall，却带来更多误拒答。两者是不同的工程取舍。

正式评测中，模型 critic 平均耗时约 `1370.72 ms`，revision 平均耗时约 `1064.36 ms`，两次生成的 pipeline 平均约 `2435.07 ms`，p95 约 `3889.62 ms`，不含模型加载时间。hard gate 本身是本地规则检查，成本远低于两次 Qwen 生成。

## 7. 结果解释与下一步

本阶段证明了扩展数据能把问题从“12 条样本是否跑通”推进到“模型是否漏检、是否误拒答、哪些 split 退化、修订是否成功、延迟是否可接受”。但仍有三个重要限制：

- 所有样本和标签都是项目内合成，存在设计者偏差；
- challenge 仍然不是独立专家或真实用户数据；
- final 0 violations 依赖 hard gate，不能写成模型 revision 的 100% 安全率。

下一步应先对 challenge 样本做人工/领域复核，再加入合法授权的公开安全偏好数据，比较独立 critic、模型 critic、规则 gate 和人工判断；如果没有高质量标签，不建议继续用更大训练步数掩盖数据问题。

## 8. 面试表述

> 我把原来的 12 条安全样本扩展为 24 个场景族、120 条分层合成 benchmark，并保证同一场景族不跨 split。模型 critic 的 recall 是 1.0、precision 是 0.8108，F1 是 0.8955；对 60 条期望违规回答，model revision 单独修复 43 条，剩余风险由 fail-closed hard gate 拦截。challenge split 的 revision 成功率只有 10%，说明我没有把整体数字当成泛化能力，而是保留了失败分析。由于数据没有专家标注，这只是安全工程回归结果，不是临床安全证明。

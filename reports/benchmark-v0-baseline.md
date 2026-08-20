# 阶段 A-2 实验报告：固定 benchmark 与自动评测

## 实验目的

验证“固定选择题 -> Qwen3 生成 -> 选项字母抽取 -> 自动评分”的链路，为后续 QLoRA、RAG 和 DPO 提供统一比较入口。

## 数据边界

- 数据文件：`data/benchmark_v0.jsonl`
- 样本数：12
- 数据来源：`synthetic-v0`
- 数据性质：人工编写的医学术语教育性样例
- 许可证标记：CC0-1.0
- 重要限制：不是正式医疗数据集，不代表临床知识评测结果

## 配置

- 模型：`Qwen/Qwen3-1.7B`
- thinking：关闭
- 解码：greedy
- `max_new_tokens`：16
- batch size：1
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GiB

运行命令：

```powershell
python scripts/run_baseline.py `
  --input data/benchmark_v0.jsonl `
  --output outputs/benchmark_v0_baseline.jsonl `
  --max-new-tokens 16 `
  --greedy

python scripts/evaluate_benchmark.py `
  --input outputs/benchmark_v0_baseline.jsonl `
  --output reports/benchmark-v0-baseline.json
```

## 结果

| 指标 | 结果 |
|---|---:|
| 总样本数 | 12 |
| 正确数 | 12 |
| exact match accuracy | 1.0000 |
| invalid output | 0 |
| invalid output rate | 0.0000 |
| 平均单条延迟 | 153.42 ms |
| 中位数延迟 | 126.02 ms |
| 近似 p95 延迟 | 416.59 ms |
| 平均生成速度 | 14.68 tokens/s |
| 峰值 CUDA allocated | 3316.02 MB |
| 峰值 CUDA reserved | 4216.00 MB |

逐题结果保存在 `reports/benchmark-v0-baseline.json`，本地原始生成结果 `outputs/benchmark_v0_baseline.jsonl` 被 `.gitignore` 忽略。

## 结果解释

12/12 不能写成“医疗问答准确率 100%”，原因是：

1. 题目是人工编写的 12 条术语定义样例，不是独立的正式测试集。
2. 题目要求只输出选项字母，任务难度和开放式医疗问答不同。
3. 参考答案没有经过外部领域专家或正式数据集标注流程复核。
4. 样本量太小，无法支持泛化结论。

这次实验真正证明的是：模型可以按统一 prompt 输出选项，评测器可以稳定抽取答案，并且后续实验可以复用同一评分接口。

## 下一步

选择一个有明确来源和许可证的正式数据集，保留一份从未参与训练的数据作为固定测试集；然后把相同评测流程应用于基线和 QLoRA adapter，避免只报告训练集上的提升。


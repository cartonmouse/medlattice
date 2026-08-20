# 阶段 A 实验报告：Qwen3-1.7B 基线推理

## 实验目的

确认模型能够在本机加载，固定输入格式能够生成结果，并建立后续 QLoRA、RAG 和 DPO 的可比较基线。

这不是领域准确率报告：当前 5 条数据是人工编写的格式样例，不能代表医疗问答能力，也不能用于临床结论。

## 环境

- Python 3.12.13
- PyTorch 2.6.0+cu124
- Transformers 5.15.1
- Accelerate 1.14.0
- CUDA 12.4
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GiB
- 模型：`Qwen/Qwen3-1.7B`
- 模型快照：`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

完整环境记录见 [`phase-a-environment.json`](phase-a-environment.json)。

## 配置

- 样本数：5
- batch size：1
- thinking：关闭
- 解码：greedy
- `max_new_tokens`：64
- `max_input_tokens`：2048
- 随机种子：42
- 输出文件：本地 `outputs/baseline.jsonl`，该文件被 `.gitignore` 忽略

运行命令：

```powershell
$env:HF_HOME = (Join-Path (Get-Location) '.hf-cache')
$env:HF_HUB_OFFLINE = '1'
python scripts/run_baseline.py `
  --input data/sample_medical_qa.jsonl `
  --output outputs/baseline.jsonl `
  --max-new-tokens 64 `
  --greedy

python scripts/summarize_baseline.py `
  --input outputs/baseline.jsonl
```

## 结果

| 指标 | 结果 | 说明 |
|---|---:|---|
| 样本数 | 5 | 仅用于 smoke test |
| 平均单条延迟 | 1967.61 ms | batch=1，生成 64 tokens |
| 中位数延迟 | 1923.14 ms | 5 条样本 |
| 近似 p95 延迟 | 2246.81 ms | nearest-rank；样本量太小，不作为稳定服务指标 |
| 平均生成速度 | 32.68 tokens/s | 输出刚好达到 64 token 上限 |
| 中位数生成速度 | 33.28 tokens/s | 同上 |
| 峰值 CUDA allocated | 3307.40 MB | 脚本记录值，需与更大测试集复核 |
| 峰值 CUDA reserved | 4210.00 MB | 脚本记录值，需与更大测试集复核 |

## 失败案例与解释

`sample-001` 询问“症状”的含义，模型回答中把症状描述为“客观证据”，并混入了体征的定义。这说明基线存在概念边界混淆；同时答案全部达到 64 token 上限，表明当前长度设置会截断输出。

这次实验没有把 5 条样例转换成“准确率”，原因是：

1. 样本量太小，单个样本就会显著改变比例。
2. 开放式答案不能只用字符串完全匹配判断质量。
3. 医疗问答需要明确的答案标准，最好先使用可判定的选择题或结构化字段。

## 阶段结论

- Qwen3-1.7B 可以在 RTX 4060 Laptop 8GB 上加载并完成推理。
- 当前可复现的是运行链路和生成性能，不是领域准确率。
- 下一步先建立有来源、可授权、可评测的固定 benchmark，再讨论 QLoRA 是否带来收益。


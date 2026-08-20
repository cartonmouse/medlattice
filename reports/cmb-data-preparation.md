# CMB-Exam v1 数据准备报告

## 1. 来源与版本

- 数据集：[FreedomIntelligence/CMB 官方 Hugging Face 数据卡](https://huggingface.co/datasets/FreedomIntelligence/CMB)
- 官方代码仓库：[FreedomIntelligence/CMB](https://github.com/FreedomIntelligence/CMB)
- 配置：`CMB-Exam`
- 上游 revision：`935fbc09edf1303d89872b21265ff597f426ac0d`
- 上游最近更新时间：`2024-04-05T16:10:47Z`
- 上游数据卡声明许可证：Apache-2.0
- 本地处理日期：2026-08-20
- 处理依赖：Python 3.12.13、`datasets==5.0.1`

许可证字段沿用上游数据卡的声明，但考试题目的原始来源和具体使用边界仍需在商业使用或再分发前重新核验。本项目只将来源、脚本和统计报告提交到 GitHub，不提交原始数据或处理后的大文件。

## 2. 处理策略

脚本：`scripts/prepare_cmb.py`

1. 以 `streaming=True` 读取三个 split，避免上游 JSON 文件在不同 split 中的字段类型差异触发 Arrow schema cast。
2. 将字符串形式的 `option` JSON 解析为统一的选项字典。
3. 规范化答案字母，并根据答案长度和 `question_type` 标记单项/多项选择题。
4. 第一版只保留单项选择题；train/val 缺少答案的记录跳过，test 允许无标签记录。
5. 使用随机种子 42 做 deterministic reservoir sampling，保持输出顺序稳定。
6. 输出 UTF-8 JSONL，并记录每个文件的 SHA-256。

固定命令：

```powershell
python -m pip install -r requirements-data.txt
python scripts/prepare_cmb.py `
  --output-dir data/processed/cmb-exam-v1 `
  --train-limit 5000 `
  --val-limit 280 `
  --test-limit 1000 `
  --seed 42
```

## 3. 实际结果

| split | 原始数量 | 有效单项选择题 | 写入数量 | 有标签 | 无标签 | 输出 SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 269,359 | 243,311 | 5,000 | 5,000 | 0 | `d30b88c9530b695e4459613aacffc123a3924225445bf2a636b2490b2f12ea36` |
| val | 280 | 240 | 240 | 240 | 0 | `8a73b6f4300865e9913fabe0b32ddc47fbde16b6e2e6cfbb13f321e109fc1e44` |
| test | 11,200 | 10,010 | 1,000 | 0 | 1,000 | `dbe877d7b462910d1980b1f5fefd0345f0d72292f9c0aa9e8c50d1b5ede0c915` |

过滤统计：train 跳过 25,968 条多项选择题、56 条答案不在选项集合中的记录、13 条规范化失败记录和 11 条缺少答案记录；val 跳过 40 条多项选择题；test 跳过 1,190 条多项选择题。

## 4. SFT 输入格式

脚本 `scripts/prepare_sft.py` 将 train/val 转为 Qwen 对话格式：system、user、assistant 三条 message。assistant target 只使用参考答案字母，不把 val 中的长解释混入训练目标，保证训练输出与现有 exact-match evaluator 的口径一致。

| 文件 | 数量 | SHA-256 |
| --- | ---: | --- |
| `sft_train.jsonl` | 5,000 | `119644c16648b867757f13f3b1e6bff42089835f89eeca671beadb353d36417d` |
| `sft_val.jsonl` | 240 | `47422084f48738ef5399932fed7810a0912cebb0bbf7c222b86deb516e7f1777` |

## 5. 评测口径与限制

当前上游 test split 不公开答案，因此不能用本地 test 子集报告 accuracy。后续基线、SFT 和 QLoRA 的有标签对比应使用 val；test 只用于无标签推理、输出格式检查或在获得合法标签后再评测。

这批数据是公开考试题的工程实验输入，不代表临床知识审核结果，也不能用于诊断、治疗或其他医疗决策。

# 数据说明

`sample_medical_qa.jsonl` 只包含少量人工编写的格式样例，用于验证脚本，不代表正式医疗数据集，也不用于医疗建议。

正式数据进入仓库前必须记录：

- 数据集名称、版本、来源链接和许可证
- 是否包含个人信息或敏感信息
- 下载和预处理步骤
- 去重、过滤、脱敏和划分规则
- 训练、验证、测试样本数量
- 数据集是否允许再分发

每行 JSON 至少包含以下字段：

```json
{
  "id": "sample-001",
  "question": "问题文本",
  "reference_answer": "仅用于评测的参考答案",
  "source": "数据来源",
  "license": "许可证"
}
```

## benchmark_v0

`benchmark_v0.jsonl` 是 12 条人工编写的医学术语教育性选择题，用于验证“固定测试集 -> 生成 -> 选项抽取 -> 准确率”的链路。它不是正式医疗数据集，也不代表临床知识评测结果。

运行基线后，可以使用 `scripts/evaluate_benchmark.py` 计算选项字母的 exact match accuracy 和 invalid output rate。正式数据集接入后，必须替换数据来源、许可证、划分规则和评测说明。

## CMB-Exam v1

正式数据候选使用 `FreedomIntelligence/CMB` 的 `CMB-Exam` 配置。来源、许可证声明、上游划分、处理限制和引用信息记录在 [`data/sources/cmb-exam.yaml`](sources/cmb-exam.yaml)。

第一版处理策略：

- 只保留单项选择题，暂不把多项选择题混入 exact match 评测。
- 训练集最多取 5000 条，验证集取 280 条，测试集固定取 1000 条 smoke subset。
- 使用固定随机种子 42，并在 `metadata.json` 中记录样本数量和输出文件 SHA-256。
- 原始数据不复制进仓库；`data/processed/` 只在本地生成并被 `.gitignore` 忽略。
- 当前上游 test split 不公开答案；验证集保留标签用于有标签评测，测试子集只用于无标签推理或格式检查，不能报告 test accuracy。
- 使用前仍需复核上游数据的来源、许可证和具体用途限制；项目不用于临床决策。

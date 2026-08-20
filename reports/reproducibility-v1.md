# 可复现性与预测漂移 v1 阶段报告

## 1. 为什么补这一阶段

CMB 闭域 RAG 初次实验中，同一组表面配置曾出现 112/240 和 115/240 两个结果。原因不能简单归结为模型能力变化；当前脚本虽然设置了随机种子，但没有显式请求 CUDA/cuDNN 确定性算法，也没有工具比较两次运行的逐题漂移。因此先补齐可复现性控制，再继续扩大后端或做 DPO。

## 2. 实现内容

- 新增 `src/qwen_medical_qa/reproducibility.py`：统一设置 Python、NumPy、Torch 和 CUDA seed；可选设置 `CUBLAS_WORKSPACE_CONFIG`、`torch.use_deterministic_algorithms(warn_only=True)`、cuDNN deterministic/benchmark 状态；返回 JSON 可序列化运行 manifest。
- `scripts/run_baseline.py` 和 `scripts/run_cmb_rag.py` 新增 `--deterministic`，并把 reproducibility manifest 写入每条预测记录。
- 新增 `src/qwen_medical_qa/prediction_compare.py` 和 `scripts/compare_predictions.py`：按样本 ID 比较两份 JSONL，只输出相同预测、漂移、invalid 和四格迁移矩阵，不输出题目文本。
- 新增 3 项单元测试，完整测试数从 36 增加到 39。

## 3. 实验验证

### 3.1 Smoke 重跑

使用同一 Qwen3-1.7B、同一 BM25 retrieval、seed=42、greedy、`--deterministic`，连续运行 8 条 CMB val：

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 8 |
| 两次预测相同 | 8 |
| 预测漂移 | 0 |
| 两次 invalid | 0 / 0 |

### 3.2 全量 val

同配置运行 240 条 CMB val：

| 指标 | 结果 |
| --- | ---: |
| 正确 | 115 / 240 |
| Accuracy | 47.92% |
| Invalid | 0 |
| 与此前最新 BM25 全量结果相同的预测 | 240 / 240 |

对应报告为本地生成的 `reports/cmb-rag-v1-deterministic-val.json`，该 JSON 只包含样本 ID、选项字母和聚合字段，不包含题目文本。完整预测输出仍在被忽略的 `outputs/` 目录。

## 4. 运行命令

```powershell
python scripts/run_cmb_rag.py `
  --model Qwen/Qwen3-1.7B `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --retrieval outputs/cmb-rag-v1/retrieval.jsonl `
  --output outputs/cmb-rag-v1/rag-deterministic-val.jsonl `
  --max-new-tokens 4 `
  --max-input-tokens 2048 `
  --seed 42 `
  --greedy `
  --deterministic `
  --local-files-only

python scripts/evaluate_benchmark.py `
  --input outputs/cmb-rag-v1/rag-deterministic-val.jsonl `
  --output reports/cmb-rag-v1-deterministic-val.json

python scripts/compare_predictions.py `
  --left outputs/cmb-rag-v1/rag-base-val.jsonl `
  --right outputs/cmb-rag-v1/rag-deterministic-val.jsonl
```

## 5. 面试解释与限制

可以这样解释：“我发现只设置 seed 不能自动保证 CUDA 推理逐题一致，所以增加了统一 reproducibility manifest 和 `--deterministic` 开关；同时写了 prediction drift 工具，比较两次运行的样本 ID、答案漂移和迁移矩阵。8 条 smoke 两次完全一致，全量 deterministic 结果与最新 BM25 结果 240 条一致。”

这仍不是跨机器、跨驱动、跨 PyTorch 版本的绝对复现保证。当前 `warn_only=True` 会在遇到不支持确定性的算子时给出可见警告而不强制终止；还需要锁定 Python/CUDA/PyTorch/Transformers 版本、记录硬件和模型 revision，并用多 seed 报告均值与方差。确定性开关也可能牺牲吞吐，不能直接当作生产默认配置。

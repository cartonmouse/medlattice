# DPO v1（QLoRA-DPO）阶段报告

## 1. 实验目标与边界

本阶段在已经完成的 CMB-Exam QLoRA/SFT adapter 之上实现 Direct Preference Optimization（DPO），验证“偏好数据构造 -> DPO smoke -> 正式训练 -> 隔离验证集对照”的完整工程链路。

本实验服务于工程验证，不代表医疗能力、临床安全性或人类偏好对齐效果。当前没有可合法使用的人类/临床专家偏好数据，因此采用可审计的合成偏好对；这个限制必须在使用结果时明确说明。

## 2. 数据设计

输入是本地已经固定的 `data/processed/cmb-exam-v1/train.jsonl`，共 5,000 条单项选择题。每条 DPO 样本使用显式 conversational preference 格式：

- `prompt`：原有 system/user messages；
- `chosen`：assistant 输出参考答案字母；
- `rejected`：从其他选项中按 seed=42 随机采样的错误选项；
- 元数据：记录 `preference_source=synthetic_from_reference_answer` 和 `negative_strategy=random_wrong_choice`。

按 seed=42 固定切分为 4,800 条 DPO train 和 200 条 DPO eval。CMB `val` 的 240 条样本没有参与偏好对构造，继续作为最终隔离验证集。原始数据、DPO JSONL 和模型输出均保留在本地忽略目录，不上传 GitHub。

数据构造命令：

```powershell
python scripts/prepare_dpo.py
```

本次数据摘要：

| 项目 | 结果 |
| --- | ---: |
| 输入 train | 5,000 |
| DPO train | 4,800 |
| DPO eval | 200 |
| 隔离 CMB val | 240 |
| seed | 42 |
| input SHA-256 | `d30b88c9530b695e4459613aacffc123a3924225445bf2a636b2490b2f12ea36` |
| DPO train SHA-256 | `4f2ce4daf372c3bacbe0234810aa955ce885a44ed119962a1833970fc9dcd2ac` |
| DPO eval SHA-256 | `031ded6229e7e9514f752b9fb0f86599b7cc43eea5d8f1e19af936ef066de670` |

`rejected` 是随机错误选项，通常是容易识别的负例，不等价于“低质量但看起来合理”的真实偏好负例。因此本阶段更准确的说法是：验证了 DPO 的训练与评测链路，并观察了模型对合成偏好的拟合情况。

## 3. DPO/QLoRA 实现

policy 从 `outputs/qlora-cmb-v1` 加载，继续使用 Qwen3-1.7B 的 4-bit NF4 基座和 LoRA adapter；reference policy 使用 PEFT 创建的冻结副本，避免额外维护一份完整模型。训练脚本通过 tokenizer 的 chat template 将 prompt/chosen/rejected 渲染为 DPOTrainer 可消费的文本，并关闭 thinking，使训练目标和“只输出选择题字母”的 benchmark 保持一致。

主要配置见 [`configs/dpo.yaml`](../configs/dpo.yaml)：

| 配置 | 值 |
| --- | --- |
| base model | `Qwen/Qwen3-1.7B` |
| policy initialization | `outputs/qlora-cmb-v1` |
| beta | `0.1` |
| loss | sigmoid DPO loss |
| epoch | `1` |
| batch size | `1` |
| gradient accumulation | `8` |
| learning rate | `5e-6` |
| optimizer | `paged_adamw_8bit` |
| max length / prompt length | `512` / `384` |
| quantization | 4-bit NF4 + double quantization |
| precision | bf16 |
| GPU | RTX 4060 Laptop 8GB |

先用 16 条 train、4 条 eval、2 个 optimizer steps 完成 smoke。第一次 smoke 使用 fp16 配置，在第一次 backward 时遇到当前 PEFT/QLoRA adapter 权重以 bf16 表示而 AMP unscale 不支持 BFloat16 的错误；确认 RTX 4060 支持 bf16 后改为 `bf16=true, fp16=false`，smoke 成功，train loss 约 0.6920、eval loss 约 0.6916。这个排错过程说明精度配置必须结合实际量化权重 dtype 和 GPU 能力验证，不能只复制 SFT 配置。

正式训练命令：

```powershell
python -m pip install -r requirements-dpo.txt
python scripts/prepare_dpo.py
python scripts/train_dpo.py --config configs/dpo.yaml
```

正式训练结果：

| 指标 | 结果 |
| --- | ---: |
| optimizer steps | 600 |
| train runtime | 4,213.19 s（约 70.2 分钟） |
| train loss | 0.627250 |
| final DPO eval loss | 0.603744 |
| final reward margin | 0.1947 |
| final reward accuracy | 0.865 |
| peak CUDA allocated | 3,941.59 MB |
| peak CUDA reserved | 4,704.00 MB |

训练摘要保存在本地 `outputs/dpo-cmb-v1/run_summary.json`，adapter 输出目录为 `outputs/dpo-cmb-v1/`。这些权重不应提交到仓库；GitHub 只保留代码、配置、报告和复现说明。

## 4. 隔离验证集结果

DPO adapter 与已有 SFT adapter 使用相同的 CMB val、prompt、greedy decoding 和 deterministic 设置。评测仍然是选择题答案字母 exact-match，不能解释成医疗准确率。

| 模型 | correct | total | accuracy | invalid |
| --- | ---: | ---: | ---: | ---: |
| QLoRA/SFT | 121 | 240 | 50.42% | 0 |
| QLoRA-DPO | 121 | 240 | 50.42% | 0 |

逐题比较：

- 232/240 条预测相同；
- 8/240 条预测发生变化；
- SFT 正确、DPO 错误：1 条；
- SFT 错误、DPO 正确：1 条；
- 因此净正确题数变化为 0。

复现与评测命令：

```powershell
python scripts/run_baseline.py `
  --model Qwen/Qwen3-1.7B `
  --adapter outputs/dpo-cmb-v1 `
  --input data/processed/cmb-exam-v1/val.jsonl `
  --output outputs/dpo-cmb-v1-val.jsonl `
  --greedy `
  --deterministic

python scripts/evaluate_benchmark.py `
  --input outputs/dpo-cmb-v1-val.jsonl `
  --output reports/dpo-cmb-v1-val.json

python scripts/compare_predictions.py `
  --left outputs/qlora-cmb-v1-val-full.jsonl `
  --right outputs/dpo-cmb-v1-val.jsonl `
  --output reports/dpo-v1-sft-prediction-compare.json
```

## 5. 结果解释

这次实验不能得出“DPO 提升了 CMB 准确率”的结论。更准确的结论是：

1. DPO 数据格式、偏好对构造、冻结 reference policy、QLoRA 训练和 adapter 推理均已跑通；
2. DPO train/eval loss 和 reward margin 表明 policy 学到了当前合成偏好信号；
3. 在只要求输出选择题字母的窄任务上，DPO 与 SFT 在隔离 val 上打平，预测变化很少；
4. loss 下降或 reward margin 上升不等价于下游 benchmark accuracy 提升，因此必须同时报告训练指标和独立任务指标。

## 7. 文件索引

- 配置：`configs/dpo.yaml`
- 偏好对构造：`src/qwen_medical_qa/dpo_data.py`、`scripts/prepare_dpo.py`
- 训练：`src/qwen_medical_qa/dpo_training.py`、`scripts/train_dpo.py`
- 数据构造测试：`tests/test_dpo_data.py`
- 本地输出：`outputs/dpo-cmb-v1/`（忽略，不提交）
- 评测报告：`reports/dpo-cmb-v1-val.json`、`reports/dpo-v1-sft-prediction-compare.json`（本地生成，报告正文保留可复现摘要）

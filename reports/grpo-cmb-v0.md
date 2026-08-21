# CMB train-only GRPO v0 阶段报告

## 1. 实验目标

前面的 DPO 实验已经证明了 QLoRA-DPO 的工程链路，但随机错误选项、公开英文偏好数据和 SFT 模型 hard negative 都没有在最终 CMB val 上带来可靠提升。本阶段实现一个最小、可验证的 GRPO 实验：让模型对同一选择题采样多个回答，用 CMB 参考答案直接计算 reward，再用 group-relative objective 更新 policy。

本阶段目标是验证训练闭环和下游指标，不把 exact-match reward 解释为临床质量或安全性指标。

## 2. 数据与隔离

- 输入：本地 CMB-Exam train split，5,000 条。
- 输入 SHA-256：`d30b88c9530b695e4459613aacffc123a3924225445bf2a636b2490b2f12ea36`。
- 固定切分：GRPO train 4,800 条、GRPO eval 200 条、seed=42。
- GRPO train SHA-256：`7c7643085e666fa56aadeb6d1cf6a4ed14bcfa21573012a1b80b56915091d4b5`。
- GRPO eval SHA-256：`68ccd592925c2cd86cdf6835154ddc4f745473852e652042ffa53adb85c3f1b5`。
- 最终 CMB val：240 条，GRPO 数据与 val ID 交集为 0；val 没有参与 GRPO 数据构造、训练或内部 eval。
- 原始数据、派生 JSONL、模型权重和预测文件均保留在本地忽略目录，不提交 GitHub。

## 3. GRPO 设计

每条记录保存 conversational `prompt`、`reference_answer` 和动态 `valid_choices`。训练时每个 prompt 生成 4 个 completion：

1. `choice_exact_match_reward`：项目现有 `extract_choice` 能解析出参考选项时得 1.0，否则得 0。
2. `choice_format_reward`：completion 去除空白后恰好等于一个合法选项字母时得 1.0，配置权重为 0.2。

因此最终 reward 可以简写为：

`R = exact_match + 0.2 × valid_single_letter_format`

奖励函数不调用外部 API、不使用人工偏好标签，也不引入 reward model。它适合 CMB 这种有确定答案的选择题，但不能衡量开放式回答的医学事实性、解释质量、拒答边界或引用忠实性。

policy 从已有 `outputs/qlora-cmb-v1` 初始化；参考策略使用同一 SFT adapter 的冻结 PEFT 副本。配置为 Qwen3-1.7B、4-bit NF4、LoRA、bf16、`num_generations=4`、`loss_type=grpo`、`beta=0.04`、learning rate=`5e-7`、600 optimizer steps、completion 上限 4 tokens。

## 4. Smoke 与正式训练

GRPO smoke 使用 4 条 train、4 条 eval、1 个 optimizer step，成功完成生成、奖励回传、GRPO loss、参考 adapter 和模型保存。smoke 结果只用于验证链路，不用于宣称效果。

正式训练结果：

| 项目 | 结果 |
| --- | ---: |
| optimizer steps | 600 |
| train runtime | 1,641.48 s，约 27.4 min |
| train loss | `2.9501e-05` |
| 第 600 步 eval loss | `1.4611e-05` |
| 第 600 步 exact-match reward mean | `0.6225` |
| 第 600 步 format reward mean | `1.0000` |
| 第 600 步总 reward mean | `1.6225` |
| 第 600 步 KL | `3.6488e-04` |
| peak CUDA allocated | 2,044.95 MB |
| TRL / Transformers | 0.29.1 / 5.15.1 |

内部 reward 在训练中保持组内差异，format reward 始终稳定，但这只证明奖励函数能产生可优化信号，不代表最终 benchmark 一定提升。

## 5. 独立 CMB val 结果

所有模型使用相同 prompt、`enable_thinking=false`、`max_new_tokens=4`、greedy、seed=42、deterministic 设置；最终 val 共 240 条。

| 模型 | 正确/总数 | Accuracy | 相对 SFT |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B base | 107/240 | 44.58% | -5.84 pp |
| QLoRA/SFT | 121/240 | 50.42% | — |
| DPO v1 | 121/240 | 50.42% | 0.00 pp |
| public DPO v2 | 118/240 | 49.17% | -1.25 pp |
| hard-negative DPO | 120/240 | 50.00% | -0.42 pp |
| GRPO v0 | 122/240 | 50.83% | +0.42 pp |

GRPO v0 的 invalid rate 为 0。相对 SFT 的逐题比较：233/240 条预测相同，7 条变化；SFT 正确而 GRPO 错误 1 条，SFT 错误而 GRPO 正确 2 条，净增加 1 道正确题。相对 hard-negative DPO，GRPO 有 2 条从错误变对、没有从正确变错。

## 6. 结果解释与局限

本次 GRPO 是小幅正向结果，但不能写成“GRPO 显著提升医疗准确率”，原因有三点：

1. 只有 240 条最终 val、单 seed、单一奖励和单一窄任务；1 道题的差异不足以证明统计稳定性。
2. reward 直接使用 CMB 的参考答案，适合可验证选择题，却没有覆盖开放式医学回答和安全行为。
3. 训练目标是 group-relative reward 和 reference KL，不是直接对最终 val 做梯度优化；内部 reward 与下游 exact-match 仍需分开报告。

准确结论是：本阶段实现并验证了 CMB train-only 的可验证奖励 GRPO 链路；在固定独立 val 上，GRPO v0 相对 SFT 有 1 道题的净提升，但证据不足以支持稳定泛化或临床能力提升。下一步若要把 GRPO 做成更有说服力的简历结果，应增加多 seed、扩大独立测试集、设计更丰富的 reward，并加入人工/专家审核的开放式质量与安全评测。

## 7. 面试回答模板

### 为什么 GRPO 不用 DPO 偏好对？

“DPO 需要同一 prompt 下的 chosen/rejected pair，而 GRPO 可以对同一 prompt 采样多个 completion，再根据外部可计算 reward 做组内相对比较。本项目的 CMB 选择题有确定参考答案，所以我把答案 exact-match 作为主 reward、合法单字母作为辅助格式 reward。这样能验证 GRPO 训练闭环，但它只适用于这个可验证的窄任务。”

### GRPO 最后有提升吗？

“有小幅提升，但不能夸大。固定 240 条 CMB val 上，SFT 是 121/240、50.42%，GRPO v0 是 122/240、50.83%；233 条预测不变，1 条由对变错，2 条由错变对。训练内部第 600 步 exact-match reward mean 是 0.6225、KL 是 3.65e-4。由于只有单 seed 和单一选择题 reward，我会把它表述为可验证 GRPO 的轻微正向结果，而不是稳定医疗能力提升。”

### 如果继续改进 GRPO？

“我会先做多 seed 和更大的独立测试集，再检查 reward 是否过窄。对于开放式医疗问答，不能只用答案字符串匹配，而应加入经领域审核的事实性、引用忠实性、安全拒答和格式约束，并报告人工评审与自动指标的分离结果。PPO 暂时不作为下一步，因为它还需要 reward model、value model 和更高的工程复杂度。”

## 8. 复现入口

- 数据准备：`scripts/prepare_grpo.py`
- 配置：`configs/grpo-cmb-v0.yaml`
- 训练：`scripts/train_grpo.py`
- 数据与奖励测试：`tests/test_grpo_data.py`
- 正式 adapter（本地）：`outputs/grpo-cmb-v0`
- 最终 val 预测（本地）：`outputs/eval-cmb-val-grpo-v0.jsonl`
- 最终 val 指标（本地）：`outputs/eval-cmb-val-grpo-v0.metrics.json`
- SFT/GRPO 逐题漂移：`outputs/compare-sft-vs-grpo-v0.json`

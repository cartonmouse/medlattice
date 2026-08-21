# CMB train-only hard-negative DPO 阶段报告

## 1. 实验目的

DPO v1 使用“参考答案 + 随机错误选项”构造偏好对，能够验证训练链路，但随机错误选项可能过于容易。本阶段尝试用已有 CMB QLoRA/SFT adapter 的选项级条件 log-probability 选择 hard negative：对每道题，`chosen` 仍是数据参考答案，`rejected` 是模型分数最高的错误选项。

本实验只服务于机器学习工程学习和面试复盘，不代表医疗能力、临床安全性或医生偏好。hard negative 来自模型自身，不是人类或临床专家标注。

## 2. 数据与隔离设计

- 输入：`data/processed/cmb-exam-v1/train.jsonl`
- 输入行数：5,000
- 输入 SHA-256：`d30b88c9530b695e4459613aacffc123a3924225445bf2a636b2490b2f12ea36`
- 生成模型：`Qwen/Qwen3-1.7B`
- 生成 adapter：`outputs/qlora-cmb-v1`
- hard-negative 策略：`highest_scoring_wrong_option`
- 偏好来源：`sft_hard_negative_from_option_logprob`
- seed：42
- DPO train/eval：4,800/200
- CMB val：240 条，未用于 hard-negative 生成、DPO 训练或 DPO eval
- 原始数据与派生 JSONL：只保留在本地被 `.gitignore` 忽略的目录，不上传 GitHub

生成文件哈希如下：

| 文件 | SHA-256 |
|---|---|
| `hard_negative_pairs.jsonl` | `0daf285ab40f46efb666ea70ee95103470690a88b7876787d2f038eae0afce47` |
| `dpo_train.jsonl` | `9a36947e27d90d7541053910acd3e988b51fddc99307bfb09f1d906cb94eb934` |
| `dpo_eval.jsonl` | `d5612ea83380c2720849378468cb6daf8c6db90d6af8ef9a65b35dc0845c1e7a` |

### 2.1 生成统计

- SFT 当前预测正确：3,712/5,000
- SFT 当前预测错误：1,288/5,000
- 在这 1,288 条错误子集中，hard negative 是模型 top choice：1,288/1,288
- `reference_minus_hard_negative`：mean=1.2705，median=0.9688，min=-4.6875，max=7.8438
- 选项数量分布：3 个选项 2 条、4 个选项 243 条、5 个选项 4,738 条、6 个选项 17 条

这里的 log-probability 只在当前题目的选项集合内比较，避免把不同长度的开放式回答直接放在一起比较。代码保留动态选项数，因为 CMB train 并非所有记录都是五选一。

### 2.2 审计结果

- 全量偏好对 ID 无重复
- train/eval ID 交集为 0
- hard-negative 全量 ID 与 CMB val ID 交集为 0
- chosen 与 rejected 相同的记录为 0
- rejected 均为当前题目的合法错误选项
- 5,000 条记录的 `source_split` 均为 `train`

## 3. DPO smoke

使用 16 条 train、4 条 eval，完成 2 个 optimizer steps：

| 指标 | 结果 |
|---|---:|
| train loss | 0.6912 |
| eval loss | 0.6916 |
| eval reward accuracy | 0.25 |
| eval reward margin | 0.0031 |
| peak CUDA allocated | 3,361.44 MB |

smoke 证明 conversational preference 格式、QLoRA-DPO 训练、reference policy 和 adapter 保存/加载均可用。smoke 数据量太小，不用于宣称效果。

## 4. 正式训练配置与结果

policy 从 `outputs/qlora-cmb-v1` 初始化，reference policy 是冻结副本。配置为：4-bit NF4、LoRA、bf16、`beta=0.1`、sigmoid loss、1 epoch、batch size=1、gradient accumulation=8、learning rate=`5e-6`、最大长度 512、最大 prompt 长度 384。

正式训练完成 600 个 optimizer steps，运行时间约 57 分钟：

| 指标 | 结果 |
|---|---:|
| train examples | 4,800 |
| eval examples | 200 |
| train loss | 0.663561 |
| final eval loss | 0.655254 |
| final eval reward accuracy | 0.740 |
| final eval reward margin | 0.08038 |
| peak CUDA allocated | 3,941.59 MB |
| peak CUDA reserved | 4,704.00 MB |

训练内部指标显示 policy 学到了区分 chosen/rejected 的信号，但这只回答“是否拟合了偏好对”，不回答“是否提升了独立任务准确率”。

## 5. 独立 CMB val 结果

评估复用了 SFT 基线的 240 条 val、同一 prompt、`max_new_tokens=4`、greedy 和 deterministic 设置。

| 模型 | correct | total | exact-match |
|---|---:|---:|---:|
| QLoRA/SFT | 121 | 240 | 50.42% |
| hard-negative DPO | 120 | 240 | 50.00% |

逐题比较：

- 230/240 条预测相同
- 10 条预测变化
- SFT 正确、hard-negative DPO 错误：2 条
- SFT 错误、hard-negative DPO 正确：1 条
- 净变化：-1 道正确题，-0.42 个百分点

与 DPO v1 比较时，234/240 条预测相同；hard-negative DPO 有 1 条从 DPO v1 正确变错，没有从错误变对。它没有改善前一轮 DPO 的下游结果。

## 6. 为什么内部指标变好但 val 没提升

1. DPO 优化的是同一 prompt 下 chosen 与 rejected 的相对 log-probability，不是直接优化 CMB val exact-match。
2. 任务只要求输出选项字母，SFT 已经在相同题型上学习了很窄的输出分布；继续拉开选项概率可能只是改变局部决策边界。
3. hard negative 由 SFT 模型自己产生，能描述模型的混淆，但不能保证参考答案、错误选项或任务标签都代表高质量偏好。
4. CMB val 只有 240 条，当前只跑了单一 seed 和单轮训练，1 个样本的变化不足以证明收益或退化规律。
5. 选择题 hard negative 不能覆盖开放式回答质量、医学事实性、拒答边界、引用忠实性和安全性。

因此本阶段的准确说法是：hard-negative DPO 工程链路成功，DPO 内部偏好指标为正，但独立 CMB val 没有观察到收益，并出现轻微负迁移。不能说“DPO 无效”，也不能说“DPO 提升了医疗能力”。

## 7. 面试回答模板

### 为什么用 hard negative？

“DPO v1 的 rejected 是随机错误选项，可能太容易。于是我让已有 SFT adapter 对每道题的所有候选选项计算条件 log-probability，并从错误选项中选分数最高的一个作为 rejected。这样负例来自模型真实混淆边界，能检验 DPO 是否学习更细的相对偏好。但它仍然是模型生成的合成偏好，不是医生标注。”

### 最后有提升吗？

“没有。训练内部 final eval reward accuracy 是 0.740、margin 是 0.0804，说明 policy 拟合了 hard-negative pair；但固定的 CMB val 上 SFT 是 121/240，hard-negative DPO 是 120/240，230 条预测不变，2 条由对变错、1 条由错变对。因此我把它保留为有完整审计和失败分析的消融，而不是包装成主模型提升。”

### 下一步做什么？

“如果目标是继续研究训练方法，我会把 GRPO 做成可验证奖励的最小实验：模型生成选项字母，奖励函数只检查格式和答案 exact-match，先做短 smoke，再用独立 val 比较。它与 DPO 的差别是直接用 group-relative reward 更新 policy，而不是依赖 chosen/rejected pair；仍然要防止奖励过窄和过拟合。”

## 8. 复现入口

- hard-negative 生成：`scripts/prepare_hard_negative_dpo.py`
- DPO 配置：`configs/dpo-cmb-hard-negative-v1.yaml`
- DPO 训练：`scripts/train_dpo.py`
- 单元测试：`tests/test_hard_negative_data.py`
- 最终 adapter（本地）：`outputs/dpo-cmb-hard-negative-v1`
- 最终 val 预测（本地）：`outputs/eval-cmb-val-dpo-hard-negative-v1.jsonl`

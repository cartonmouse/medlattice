# DPO v2 公开医学偏好数据审计报告

## 1. 阶段目标与边界

本阶段不是正式 DPO 训练，而是为 DPO v2 准备一个可追溯、可隔离、可复现的公开偏好数据实验。目标是替换 DPO v1 中“参考答案 + 随机错误选项”的合成负例，观察公开回答偏好是否能提供更有信息量的训练信号。

数据只用于机器学习工程验证，不代表医疗能力、临床安全性或医生认可。公开数据的 `chosen` 不应默认解释为专家结论，尤其不能把自动或模型辅助的偏好判断包装成人类标注。

## 2. 数据来源与固定版本

- 数据集：[TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference)
- 官方仓库：[TsinghuaC3I/UltraMedical](https://github.com/TsinghuaC3I/UltraMedical)
- 固定 revision：`761eb7935310ba662a96d93c5af342e5269d5759`
- 声明许可证：MIT；重新分发或商业使用前仍需核验上游 provenance 与条款
- 语言：英文；主项目 CMB 选择题为中文，因此存在任务与语言迁移风险
- 数据源记录：[`data/sources/ultramedical-preference.yaml`](../data/sources/ultramedical-preference.yaml)

本地只下载了两个小 split：

| 文件 | 原始行数 | 字节数 | SHA-256 |
| --- | ---: | ---: | --- |
| `data/dev.json` | 2,232 | 20,238,420 | `4f6b0a1350f2664caeefd2d628eae992ba732d475138fcdbe2657a25f70f9269` |
| `data/test.json` | 777 | 5,564,425 | `38f21a20407a401d55c1a0939f436ac1c2d5216bec31c126a55d7ed0f2c9d251` |

没有下载 994 MB 的 `train.json`，因为本轮实验用固定的 `dev/test` 子集已经足够验证数据工程链路。原始文件不提交 GitHub。

## 3. 过滤与隔离规则

实现位于 [`src/qwen_medical_qa/public_dpo_data.py`](../src/qwen_medical_qa/public_dpo_data.py) 和 [`scripts/prepare_public_dpo.py`](../scripts/prepare_public_dpo.py)。

### 训练候选

输入使用 `dev.json`。保留条件和顺序如下：

1. `chosen` 与 `rejected` 都能解析出非空 assistant response；
2. `chosen_score > rejected_score`，去掉分数相同或方向相反的 pair；
3. 按 `prompt_id` 去重，避免重复问题改变训练权重；
4. seed=42 固定抽取最多 2,000 条，再切出 200 条内部 DPO eval；
5. 训练输出统一转换为项目的 system/user prompt + assistant chosen/rejected 格式。

审计结果：2,232 条中有 145 条不满足严格分数方向，13 条因重复 prompt id 被去掉，得到 2,074 条候选；最终选择 2,000 条，切分为 1,800 train、200 eval。选中集合的 `label_type` 为 easy 432、hard 875、length 693。

### 外部 human holdout

输入使用 `test.json`，只保留 `label_type=human` 的 163 条。这里不根据模型 score 再次筛选，因为评测目标正是观察模型对人工标记偏好的匹配程度。它不参与 DPO 训练；与 2,000 条训练候选的归一化 prompt hash 交集为 0。

## 4. 本地派生结果

运行：

```powershell
python scripts/prepare_public_dpo.py `
  --seed 42 `
  --max-train-pairs 2000 `
  --eval-size 200
```

生成的文件位于被 `.gitignore` 忽略的 `data/processed/public-dpo/ultramedical-v1/`：

- `dpo_train.jsonl`：1,800 条
- `dpo_eval.jsonl`：200 条
- `human_eval.jsonl`：163 条
- `dpo_metadata.json`：规则、源文件哈希、输出哈希、标签分布和 overlap 记录

输出哈希：

- train：`9bece1860a737490acc8bdbf840ddadbd583a983458e59e73ac10cb2bf9fc0f3`
- eval：`8928760b98c2c96564d3b278e63d91a581306a97cb8e3cfb67d5245a0b4df047`
- human eval：`13c052f713d6fc336096877c5d25c155f26d0400f829f3807574543799082046`

## 5. 评测基线与 smoke

新增 [`scripts/evaluate_preference.py`](../scripts/evaluate_preference.py)，按 response token 的平均 log-prob 计算：

`chosen_avg_logprob - rejected_avg_logprob > 0` 即记为模型偏好 chosen。

这不是生成质量或临床正确率，只是 pairwise preference proxy；使用平均而非总 log-prob 是为了降低长回答天然累积更多负 log-prob 的长度偏差，但仍然不能替代人工评审。

现有 CMB SFT adapter 在完整 163 条 human holdout 上的基线（max_length=1024、max_prompt_length=384）：

| 指标 | 结果 |
| --- | ---: |
| model prefers chosen | 56/163 |
| pairwise preference accuracy | 34.36% |
| mean chosen−rejected margin | -0.0807 |
| chosen truncated | 8 |
| rejected truncated | 7 |

公开数据 DPO smoke 使用 16 条 train、4 条 eval、1024 token 上限，成功完成 2 个 optimizer steps：

| 指标 | 结果 |
| --- | ---: |
| train loss | 0.6807 |
| eval loss | 0.6548 |
| train runtime | 890.7 s |
| eval runtime | 114.7 s |
| peak CUDA allocated | 7,763.43 MB |

由于 8GB 显存下 1024 token 已接近上限，正式训练没有直接照搬 smoke 配置，而是采用 512 token 上限和固定的 512 条候选子集。

## 6. 正式 DPO v2 结果

正式配置见 [`configs/dpo-ultramedical-v2.yaml`](../configs/dpo-ultramedical-v2.yaml)，从已有 CMB SFT adapter 初始化，运行 1 epoch、448 条 train、64 条内部 eval，共 56 个 optimizer steps。

| 训练指标 | 结果 |
| --- | ---: |
| train examples | 448 |
| eval examples | 64 |
| max length / max prompt length | 512 / 384 |
| optimizer steps | 56 |
| train loss | 0.6822 |
| final eval loss | 0.6760 |
| train runtime | 735.1 s（约 12.3 分钟） |
| peak CUDA allocated | 4,880.26 MB |

### 外部 human preference holdout

评测使用未参与训练的 163 条 `label_type=human` 样本，比较平均 response token log-prob。为避免长度口径造成误判，同时报告 512 和 1024 两种最大长度：

| 模型 | max length | prefers chosen | pairwise accuracy | mean margin |
| --- | ---: | ---: | ---: | ---: |
| CMB SFT | 512 | 57/163 | 34.97% | -0.066821 |
| public DPO v2 | 512 | 57/163 | 34.97% | -0.066319 |
| CMB SFT | 1024 | 56/163 | 34.36% | -0.080731 |
| public DPO v2 | 1024 | 56/163 | 34.36% | -0.080064 |

在 512 口径下，DPO 没有增加 pairwise accuracy，只让平均 margin 变化约 `+0.000502`；1024 口径同样没有增加 pairwise accuracy。这个 proxy 不是临床准确率，也不能单独证明回答质量提升。

### CMB 中文选择题迁移

DPO v2 使用相同的 greedy、seed=42、deterministic 设置在隔离 CMB val 上评估：

| 模型 | correct | total | accuracy |
| --- | ---: | ---: | ---: |
| CMB SFT | 121 | 240 | 50.42% |
| public DPO v2 | 118 | 240 | 49.17% |

逐题比较显示 234/240 条预测相同，6 条发生变化；SFT 正确而 DPO 错误 4 条，SFT 错误而 DPO 正确 1 条，净减少 3 道正确题。结果说明英文公开偏好数据在本轮小规模设置下没有提升中文 CMB 任务，反而有轻微负迁移。

## 7. 阶段结论与下一步

本阶段完成了公开偏好数据的固定版本审计、严格偏好过滤、prompt 去重、独立 human holdout、QLoRA-DPO smoke、正式小规模训练和跨任务对照。结论不是“DPO 无效”，而是：在当前模型、数据语言、512 token 截断、448 条 train、1 epoch 和单一 seed 的组合下，没有观察到可证明的下游收益。

若继续改进，应优先选择中文、开放式、经人工/专家审核的偏好对，减少语言迁移；再增加训练样本和多 seed，并补充人工 pairwise 评审。当前结果不建议把 public DPO adapter 作为主模型，项目主结果应继续使用已有 QLoRA/SFT 与 RAG 对照，DPO 作为有明确失败分析的消融。

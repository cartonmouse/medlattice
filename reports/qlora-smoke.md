# QLoRA smoke training 报告

## 1. 实验目的

本次实验只验证“对话数据 -> assistant-only loss mask -> 4-bit base model -> LoRA adapter -> 保存与加载 -> 推理评测”的工程链路，不作为正式模型效果结果。

配置文件：`configs/qlora.yaml`

- 基础模型：`Qwen/Qwen3-1.7B`
- 训练样本：8 条 CMB-Exam train 子集样本
- 验证样本：8 条 CMB-Exam val 子集样本
- 训练步数：1 step，1 epoch
- 量化：4-bit NF4、double quantization、float16 compute
- LoRA：rank 8、alpha 16、dropout 0.05，注入 attention 和 MLP projection modules
- 可训练参数：8,716,288 / 1,729,291,264（0.5040%）
- 硬件：RTX 4060 Laptop 8GB

## 2. 运行结果

- train loss：`2.0832`
- eval loss：`1.3543`
- train runtime：`8.61 s`
- eval runtime：`1.28 s`
- 峰值 CUDA allocated：`3010.60 MB`
- 峰值 CUDA reserved：`3202.00 MB`
- adapter 输出：`outputs/qlora-cmb-smoke/`（本地忽略目录）

## 3. 同条件推理对照

使用同一批 8 条 val、同一 prompt、`max_new_tokens=4`、greedy decoding：

| 模型 | correct | total | accuracy | invalid rate |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B baseline | 7 | 8 | 0.8750 | 0.0000 |
| 8-sample QLoRA smoke adapter | 7 | 8 | 0.8750 | 0.0000 |

两者结果相同，只能说明 smoke adapter 没有破坏这批样本的输出链路，不能说明微调带来提升。正式结论必须使用完整 train/val 配置、固定评测集和多组可比实验。

## 4. 下一步

1. 先固定完整 5,000 train / 240 val 的正式训练运行记录。
2. 记录完整训练耗时、显存、checkpoint 大小和 eval 指标。
3. 使用同一 val 集分别评测 base 与 adapter，并保留失败题的逐题输出。
4. 如果显存或耗时不可接受，再调整 sequence length、gradient accumulation 或 LoRA target modules，并记录为独立实验。

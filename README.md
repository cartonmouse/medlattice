# Qwen Medical QA

一个面向学习和面试复盘的中文医疗领域问答实验项目。

项目目标不是直接声称“模型已经达到某个指标”，而是完整记录一条可复现的实验链路：

`基线推理 -> 数据清洗 -> QLoRA/SFT -> RAG -> DPO/GRPO -> 服务化 -> 评测与开源`

> 免责声明：本项目仅用于机器学习工程学习、实验复现和面试演示，不提供诊断、治疗或其他医疗建议。真实数据必须经过合法授权、脱敏和许可证核验。

项目领域术语和边界见 [`CONTEXT.md`](CONTEXT.md)，最终实验总表见 [`reports/project-final-summary.md`](reports/project-final-summary.md)。

## 当前状态

- [x] 冻结项目目标与阶段验收标准
- [x] 添加最小样例数据和数据格式说明
- [x] 添加环境检查脚本
- [x] 添加 Qwen3-1.7B 基线推理脚本
- [x] 添加最小数据格式测试
- [x] 添加 v0 选择题 benchmark 和自动评测器
- [x] 在本机 RTX 4060 Laptop 8GB 上完成基线运行
- [x] 添加 CMB-Exam 数据准备脚本和来源记录
- [x] 完成 CMB-Exam v1 的清洗、固定子集和 SHA-256 报告
- [x] 完成 SFT messages 格式和 QLoRA smoke 链路
- [x] 完成 CMB-Exam v1 的正式 QLoRA 训练和基线对照评测
- [x] 完成 RAG v0 文档摄取、TF-IDF 检索、引用提示和检索评测
- [x] 完成 BGE 中文 embedding 检索器与 TF-IDF 对照
- [x] 完成基于 BGE 检索与 Qwen3-1.7B 的生成式 RAG demo 和引用评估
- [x] 完成 SQLite vector store、BM25 reranker baseline，并接入生成式 RAG
- [x] 完成资料不足拒答、高风险意图拦截和合成安全 benchmark
- [x] 完成基于授权 CMB-Exam train split 的本地闭域 RAG 实验（原始与派生数据不入库）
- [x] 完成统一 seed/确定性开关、运行元数据和预测漂移回归工具
- [x] 完成可选 FAISS HNSW 后端与 SQLite 对照实验
- [x] 完成可选 Transformer Cross-Encoder 神经 reranker 接口、测试和 CLI 接入
- [x] 完成神经 reranker 实际模型对比和 CMB 闭域消融
- [x] 完成 DPO/QLoRA-DPO 偏好数据、smoke、正式训练和隔离 val 对照
- [x] 完成 UltraMedical-Preference 固定 revision 审计、严格过滤和 human holdout 准备
- [x] 完成公开偏好数据 DPO v2 正式训练与跨任务对照
- [x] 完成 CMB train-only hard-negative 偏好构造、DPO smoke、正式训练与独立 val 对照
- [x] 完成 CMB train-only GRPO v0 可验证奖励 smoke、正式训练与独立 val 对照
- [x] 完成 Constitutional AI-inspired v0 原则、批评-修订链路与合成 benchmark
- [x] 完成 Constitutional AI-inspired v1 模型批评、模型修订与规则硬门禁对照
- [x] 完成 Constitutional AI-inspired v1.1 120 条合成安全 benchmark 扩展与分层评测
- [ ] 完成生产级 ANN/vector DB、真实授权知识库和服务化实验

## 项目结构

```text
qwen-medical-qa/
├── configs/                  # 实验配置
├── data/                     # 只放样例或经过许可的数据说明
├── outputs/                  # 本地实验输出，不提交大文件
├── scripts/
│   ├── check_environment.py   # 检查 Python、PyTorch、CUDA 和显存
│   ├── run_baseline.py        # 基线推理与延迟记录
│   ├── summarize_baseline.py  # 汇总延迟、吞吐和显存
│   ├── prepare_cmb.py         # 下载、清洗并固定 CMB-Exam 数据子集
│   ├── prepare_dpo.py         # 从本地 SFT 数据构建 DPO 偏好对
│   ├── prepare_public_dpo.py  # 审计并转换公开医学偏好数据
│   ├── train_dpo.py           # 训练 QLoRA-DPO adapter
│   ├── prepare_grpo.py        # 准备 CMB train-only GRPO 记录
│   ├── train_grpo.py          # 训练可验证奖励的 QLoRA-GRPO adapter
│   ├── run_constitutional.py  # 运行 Constitutional AI-inspired 批评-修订评测
│   ├── run_constitutional_v1.py # 运行 Qwen model-in-the-loop 批评-修订链路
│   ├── prepare_constitutional_benchmark.py # 生成分层合成安全 benchmark
│   ├── evaluate_preference.py # 评估 chosen/rejected 偏好 proxy
│   ├── prepare_cmb_rag.py     # 从本地 CMB train/val 构建闭域检索语料
│   └── run_cmb_rag.py         # 运行 CMB 闭域 RAG 选择题评测
├── src/qwen_medical_qa/      # 后续抽取可复用模块
├── tests/                    # 自动化测试
├── reports/                  # 可提交的实验报告和环境记录
├── PROJECT_SPEC.md           # 项目规格与验收标准
└── requirements-baseline.txt
```

## 第一个实验：基线推理

建议使用 Python 3.10 或更高版本。先按照本机 CUDA 环境安装 PyTorch，再安装其余项目依赖：

```powershell
python scripts/check_environment.py
python -m pip install -r requirements-baseline.txt
```

运行最小样例：

```powershell
python scripts/run_baseline.py `
  --input data/sample_medical_qa.jsonl `
  --output outputs/baseline.jsonl `
  --max-new-tokens 128

python scripts/summarize_baseline.py `
  --input outputs/baseline.jsonl
```

运行 v0 选择题 benchmark：

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

这里的 accuracy 只表示选项字母是否匹配参考答案；它不能替代领域专家评审，也不能直接解释为医疗准确率。

## 下一阶段：CMB-Exam 数据准备

当前正式数据候选为 [FreedomIntelligence/CMB 的 `CMB-Exam` 配置](https://huggingface.co/datasets/FreedomIntelligence/CMB)。数据来源、许可证声明、版本记录和处理策略见 [`data/sources/cmb-exam.yaml`](data/sources/cmb-exam.yaml)。原始数据和 `data/processed/` 下的处理结果不提交到 GitHub。

先安装数据处理依赖，再生成固定的小规模训练/验证/测试子集：

```powershell
python -m pip install -r requirements-data.txt

python scripts/prepare_cmb.py `
  --output-dir data/processed/cmb-exam-v1 `
  --train-limit 5000 `
  --val-limit 280 `
  --test-limit 1000 `
  --seed 42
```

第一版只保留单项选择题，先验证“下载 -> 规范化 -> 过滤 -> 固定子集 -> SHA-256 记录”的数据工程链路；多项选择题留到评测口径明确后再接入。当前上游 test split 不公开答案，因此 `val` 用于有标签评测，`test` 只作为无标签推理/格式检查数据，不能据此报告 accuracy。

## 下一阶段：SFT/QLoRA

SFT 数据使用与基线相同的 prompt 模板，输出 Qwen `messages` 格式；当前 assistant target 只保留参考答案字母，以便和 exact-match evaluator 对齐。QLoRA 配置见 [`configs/qlora.yaml`](configs/qlora.yaml)，训练脚本见 [`scripts/train_qlora.py`](scripts/train_qlora.py)。

```powershell
python -m pip install -r requirements-qlora.txt
python scripts/prepare_sft.py

# 先做 8 条样本 smoke training
python scripts/train_qlora.py `
  --config configs/qlora.yaml `
  --output-dir outputs/qlora-cmb-smoke `
  --max-train-samples 8 `
  --max-eval-samples 8

# 正式训练前去掉两个 --max-* 参数，并使用独立输出目录
python scripts/train_qlora.py --config configs/qlora.yaml
```

smoke 结果见 [`reports/qlora-smoke.md`](reports/qlora-smoke.md)。它只证明训练链路和 adapter 加载可用，不代表正式准确率提升。

正式训练使用 5,000 条 train、240 条 val，在本机 RTX 4060 Laptop 8GB 上完成 1 epoch QLoRA。基础模型在 val 上为 44.58%，QLoRA adapter 为 50.42%，详细配置、逐题迁移统计和复现命令见 [`reports/qlora-full.md`](reports/qlora-full.md)。这里的指标仍然只是 CMB-Exam 单项选择题的答案字母 exact-match，不能解释为医疗准确率。

## 当前阶段：DPO v1（QLoRA-DPO）

DPO 放在 SFT 之后，用于让模型在同一个 prompt 下更偏好 chosen response，而不是只学习一个参考答案。由于当前没有人工或临床专家偏好标注，本阶段使用已有本地 CMB-Exam train split 的参考答案构造正例，并从其他选项中随机采样一个错误选项作为 rejected response。这个设计适合验证 DPO 工程链路，但不能冒充人类偏好数据，也不能据此声称模型获得了临床对齐能力。

偏好数据从本地 5,000 条 CMB train 构建，按 seed=42 固定为 4,800 条 DPO train 和 200 条 DPO eval；最终 240 条 CMB val 完全隔离，不参与偏好对构造。数据文件只保留在本地忽略目录，不上传原始数据或派生数据。实现见 [`src/qwen_medical_qa/dpo_data.py`](src/qwen_medical_qa/dpo_data.py)、[`scripts/prepare_dpo.py`](scripts/prepare_dpo.py) 和 [`scripts/train_dpo.py`](scripts/train_dpo.py)。

```powershell
python -m pip install -r requirements-dpo.txt
python scripts/prepare_dpo.py

# 先用 16/4 条样本完成 smoke
python scripts/train_dpo.py `
  --config configs/dpo.yaml `
  --output-dir outputs/dpo-cmb-smoke `
  --max-train-samples 16 `
  --max-eval-samples 4

# smoke 通过后，正式运行使用完整数据和独立输出目录
python scripts/train_dpo.py --config configs/dpo.yaml

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
```

训练时以已有 `outputs/qlora-cmb-v1` 作为 policy 初始化，基座使用 4-bit NF4，LoRA adapter 为可训练参数，reference policy 使用 PEFT 的冻结副本；`beta=0.1`、1 epoch、batch size=1、gradient accumulation=8、learning rate=`5e-6`，本机使用 bf16 计算。正式运行完成 600 个 optimizer steps，DPO train loss 为 0.6273，最终 DPO eval loss 为 0.6037，峰值 CUDA allocated 约 3.94 GB。完整记录见 [`reports/dpo-v1.md`](reports/dpo-v1.md)。

在隔离的 240 条 CMB val 上，SFT 与 DPO 都是 121/240（50.42%）；232/240 条预测相同，仅 8 条发生变化，其中 SFT→DPO 一条由错变对、一条由对变错。DPO 的 reward margin 在最后一次记录为 0.1947、reward accuracy 为 0.865，说明偏好训练目标确实被优化，但在当前“只输出选择题字母”的窄任务上没有带来验证集准确率提升。后续若要证明 DPO 的真实价值，需要合法授权的人类/专家偏好数据、开放式回答或安全拒答偏好对、独立评测集和人工质量评审。

## 当前阶段：公开偏好数据 DPO v2（首轮正式实验已完成）

为验证 DPO v1 的“随机错误选项”负例是否过于简单，本阶段接入公开的 [TsinghuaC3I/UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference)。固定 revision、文件哈希、许可证边界和筛选规则见 [`data/sources/ultramedical-preference.yaml`](data/sources/ultramedical-preference.yaml) 与 [`reports/dpo-v2-public-data-audit.md`](reports/dpo-v2-public-data-audit.md)。公开数据是英文，且偏好判断不应默认等同于医生标注；因此本阶段把它作为公开数据工程实验，不宣称临床对齐能力。

当前只下载了 `dev.json` 和 `test.json`，没有下载 994 MB 的 `train.json`。`dev.json` 的 2,232 条样本先要求 `chosen_score > rejected_score`，再按 prompt id 去重，得到 2,074 条候选；固定抽取 2,000 条，切成 1,800 条 DPO train 与 200 条内部 eval。`test.json` 中仅保留 `label_type=human` 的 163 条作为外部偏好评测，训练与该集合的归一化题干重叠为 0。原始数据和派生 JSONL 只保存在本地被忽略目录，不上传 GitHub。

```powershell
python scripts/prepare_public_dpo.py `
  --seed 42 `
  --max-train-pairs 2000 `
  --eval-size 200

python scripts/train_dpo.py `
  --config configs/dpo-ultramedical.yaml `
  --output-dir outputs/dpo-ultramedical-smoke `
  --max-train-samples 16 `
  --max-eval-samples 4 `
  --local-files-only

# 正式可完成配置：448 train、64 eval、512 token
python scripts/train_dpo.py `
  --config configs/dpo-ultramedical-v2.yaml `
  --local-files-only

python scripts/evaluate_preference.py `
  --input data/processed/public-dpo/ultramedical-v1/human_eval.jsonl `
  --output outputs/dpo-ultramedical-v1/sft-human-preference-v1.json `
  --adapter outputs/qlora-cmb-v1 `
  --max-length 1024 `
  --max-prompt-length 384 `
  --local-files-only
```

1024 token 上限的 16/4 smoke 训练与评估均成功，峰值 CUDA allocated 约 7.76 GB；因此正式实验改用 512 条候选中的 448 train、64 eval、512 token 上限，完成 56 个 optimizer steps，train loss=0.6822、eval loss=0.6760、峰值 CUDA allocated 约 4.88 GB。

正式对照结果见 [`reports/dpo-v2-public-data-audit.md`](reports/dpo-v2-public-data-audit.md)：在 163 条未参与训练的 human preference holdout 上，SFT 与 DPO v2 在 512 token 口径下都是 57/163（34.97%）；在 1024 token 敏感性评测下都是 56/163（34.36%）。在隔离 CMB val 上，SFT 为 121/240（50.42%），public DPO v2 为 118/240（49.17%），因此本轮没有证明公开英文偏好数据带来收益，并出现轻微中文任务负迁移。这个结果作为 DPO 消融和失败分析保留，不把 DPO adapter 宣传成主模型。

## 当前阶段：CMB train-only hard-negative DPO（正式实验已完成）

为了减少 DPO v1 中“随机错误选项”过于简单的问题，本阶段仍只使用本地 CMB-Exam train split，但让已有 SFT adapter 对每道题的所有选项计算条件 log-probability：`chosen` 固定为参考答案，`rejected` 选择“模型分数最高的错误选项”。这不是人类偏好标注，而是从模型真实混淆边界生成的可审计 hard negative；240 条 CMB val 全程隔离，原始数据和派生偏好 JSONL 不上传 GitHub。实现见 [`src/qwen_medical_qa/hard_negative_data.py`](src/qwen_medical_qa/hard_negative_data.py)、[`scripts/prepare_hard_negative_dpo.py`](scripts/prepare_hard_negative_dpo.py) 和 [`configs/dpo-cmb-hard-negative-v1.yaml`](configs/dpo-cmb-hard-negative-v1.yaml)。

```powershell
python scripts/prepare_hard_negative_dpo.py `
  --input data/processed/cmb-exam-v1/train.jsonl `
  --output-dir data/processed/cmb-hard-negative-v1 `
  --adapter outputs/qlora-cmb-v1 `
  --eval-size 200 `
  --local-files-only

# 先用 16/4 条完成 smoke
python scripts/train_dpo.py `
  --config configs/dpo-cmb-hard-negative-v1.yaml `
  --output-dir outputs/dpo-cmb-hard-negative-smoke `
  --max-train-samples 16 `
  --max-eval-samples 4 `
  --local-files-only

# 正式训练
python scripts/train_dpo.py `
  --config configs/dpo-cmb-hard-negative-v1.yaml `
  --output-dir outputs/dpo-cmb-hard-negative-v1 `
  --local-files-only
```

5,000 条 train 样本全部生成偏好对，按 seed=42 固定为 4,800 train、200 eval。SFT 模型在 train 上有 1,288 条预测错误；这 1,288 条中 rejected 都是模型自己的 top choice，说明 hard negative 确实覆盖了当前模型的混淆边界。数据审计显示 train/eval 无交集、与 val 无 ID 交集、无重复、chosen/rejected 均不同；CMB 中存在少量 3/4/6 选项题，因此实现使用动态选项数而不是写死五选一。

正式训练使用 Qwen3-1.7B、已有 CMB QLoRA/SFT adapter、4-bit NF4、LoRA、bf16、`beta=0.1`、1 epoch、600 optimizer steps。最终 train loss=0.6636、DPO eval loss=0.6553、eval reward accuracy=0.740、reward margin=0.0804，峰值 CUDA allocated 约 3.94 GB。独立 CMB val 上 hard-negative DPO 为 120/240（50.00%），SFT 为 121/240（50.42%）；230/240 条预测相同，10 条变化，其中 2 条由对变错、1 条由错变对。结论是：hard negative 让 DPO 训练目标得到可审计的优化，但在当前“中文选择题、只输出字母”的窄任务上没有带来泛化提升，反而出现 1 道题的轻微负迁移，不能将其宣传成医疗能力提升。完整审计、哈希、训练指标和面试表述见 [`reports/dpo-cmb-hard-negative-v1.md`](reports/dpo-cmb-hard-negative-v1.md)。

## 当前阶段：CMB train-only GRPO v0（正式实验已完成）

本阶段用 GRPO 验证另一条训练路线：不构造 `chosen/rejected` 偏好对，而是让模型对同一道 CMB 选择题生成 4 个候选答案，并用可计算的任务奖励比较它们。奖励由两部分组成：选项字母与参考答案 exact-match 得分 1.0；输出是否恰好是一个合法选项字母的格式得分乘以 0.2。这个奖励适合当前闭域选择题的工程验证，但不能代表开放式医学质量或临床安全性。

GRPO 只读取本地 CMB-Exam train split 5,000 条，固定切分为 4,800 条 train 和 200 条 GRPO eval，seed=42；最终 240 条 CMB val 与 GRPO 数据 ID 交集为 0，没有参与生成、训练或内部 eval。实现见 [`src/qwen_medical_qa/grpo_data.py`](src/qwen_medical_qa/grpo_data.py)、[`scripts/prepare_grpo.py`](scripts/prepare_grpo.py)、[`scripts/train_grpo.py`](scripts/train_grpo.py) 和 [`configs/grpo-cmb-v0.yaml`](configs/grpo-cmb-v0.yaml)。

```powershell
python scripts/prepare_grpo.py `
  --input data/processed/cmb-exam-v1/train.jsonl `
  --output-dir data/processed/cmb-grpo-v1 `
  --eval-size 200 `
  --seed 42

python scripts/train_grpo.py `
  --config configs/grpo-cmb-v0.yaml `
  --output-dir outputs/grpo-cmb-v0 `
  --local-files-only

python scripts/run_baseline.py `
  --model Qwen/Qwen3-1.7B `
  --adapter outputs/grpo-cmb-v0 `
  --input data/processed/cmb-exam-v1/val.jsonl `
  --output outputs/eval-cmb-val-grpo-v0.jsonl `
  --max-new-tokens 4 `
  --greedy `
  --deterministic

python scripts/evaluate_benchmark.py `
  --input outputs/eval-cmb-val-grpo-v0.jsonl `
  --output outputs/eval-cmb-val-grpo-v0.metrics.json
```

正式训练从已有 `outputs/qlora-cmb-v1` SFT adapter 初始化，使用冻结的同一 SFT adapter 作为参考策略；配置为 4-bit NF4、LoRA、bf16、4 generations、`loss_type=grpo`、`beta=0.04`、learning rate=`5e-7`、600 optimizer steps。训练耗时约 27.4 分钟，train loss=`2.95e-5`，第 600 步 GRPO eval 的 exact-match reward mean=`0.6225`、format reward mean=`1.0`、KL=`3.65e-4`，峰值 CUDA allocated 约 2.04 GB。

在完全相同的 prompt、greedy、deterministic 和 240 条 CMB val 口径下：

| 模型 | 正确/总数 | Accuracy |
| --- | ---: | ---: |
| Qwen3-1.7B base | 107/240 | 44.58% |
| QLoRA/SFT | 121/240 | 50.42% |
| DPO v1 | 121/240 | 50.42% |
| public DPO v2 | 118/240 | 49.17% |
| hard-negative DPO | 120/240 | 50.00% |
| GRPO v0 | 122/240 | 50.83% |

GRPO 相对 SFT 有 233/240 条预测相同，7 条变化：1 条从正确变错、2 条从错误变对，净增加 1 道题（+0.42 个百分点）；相对 hard-negative DPO 有 2 条从错误变对且没有反向退化。这个提升幅度很小，且当前只跑了单 seed、单一奖励和单一闭域 benchmark，因此只能说 GRPO v0 在本实验上取得了轻微正向结果，不能包装成稳定的医疗能力提升。完整训练日志、数据哈希、漂移矩阵和面试表述见 [`reports/grpo-cmb-v0.md`](reports/grpo-cmb-v0.md)。

## 当前阶段：RAG v0

RAG v0 先使用仓库内 5 条合成演示文档，完成“JSONL 文档 -> chunk -> TF-IDF index -> top-k 检索 -> 带 chunk 引用的 prompt -> Hit@k/MRR”链路。它不包含真实医疗资料，也还没有接入生成模型，因此结果只用于验证检索工程，不代表医疗问答效果。详细设计和结果见 [`reports/rag-v0.md`](reports/rag-v0.md)。

```powershell
python scripts/build_rag_index.py `
  --input data/rag_demo/knowledge_base.jsonl `
  --output outputs/rag-v0/index.json

python scripts/retrieve_rag.py `
  --index outputs/rag-v0/index.json `
  --input data/rag_demo/retrieval_benchmark.jsonl `
  --output outputs/rag-v0/retrieval.jsonl

python scripts/evaluate_rag.py `
  --input outputs/rag-v0/retrieval.jsonl `
  --output reports/rag-v0-retrieval.json
```

当前演示基准为 5/5 Hit@1、5/5 Hit@3、MRR@3 为 1.0。下一步是在同一评测口径下接入 embedding retriever，再接 Qwen 生成和引用忠实性评测。

Embedding v1 已接入 `BAAI/bge-small-zh-v1.5`，使用 Transformers、CLS pooling、512 维归一化向量和 CUDA 推理；在当前 5 条演示查询上与 TF-IDF 一样达到 Hit@1=1.0。详细配置、对照和局限见 [`reports/rag-embedding-v1.md`](reports/rag-embedding-v1.md)。模型缓存和 dense index 保持在本地，不提交到 GitHub。

## 当前阶段：生成式 RAG v1

在 BGE dense retriever 之后接入 Qwen3-1.7B 基础模型，完成“检索 -> 带 chunk 编号的上下文 prompt -> 生成回答 -> 答案词与引用自动评估”。本次使用 5 条合成演示问题、`top_k=3`、greedy decoding 和 `max_new_tokens=64`，Retrieval Hit@3、目标术语匹配、有效引用率和 grounded answer 自动代理指标均为 1.0。详细定义、显存、延迟和局限见 [`reports/rag-qa-v1.md`](reports/rag-qa-v1.md)。这些数字只证明端到端 demo 链路可运行，不代表医疗准确率或人工引用忠实性。

```powershell
python scripts/run_rag_qa.py `
  --model Qwen/Qwen3-1.7B `
  --index outputs/rag-embedding-v1/index.json `
  --input data/rag_demo/qa_benchmark.jsonl `
  --output outputs/rag-qa-v1/qwen3-base.jsonl `
  --device cuda `
  --embedding-batch-size 8 `
  --top-k 3 `
  --max-new-tokens 64 `
  --max-input-tokens 2048 `
  --greedy `
  --local-files-only

python scripts/evaluate_rag_qa.py `
  --input outputs/rag-qa-v1/qwen3-base.jsonl `
  --output reports/rag-qa-v1.json `
  --k 3
```

本阶段使用基础 Qwen，没有加载面向选择题字母输出的 QLoRA adapter；这样可以把开放式引用生成与选择题微调作为两个可解释的实验变量。下一步再接入合法授权知识库、vector DB、reranker，并扩展资料不足拒答和引用忠实性评测。

## 当前阶段：Vector Store v0 与 Reranker baseline

在 BGE dense index 之上增加 SQLite 持久化 vector store，并实现“dense candidate top-5 -> BM25 reranker -> final top-3”两阶段检索。生成式 RAG 脚本现在支持 `--store` 和 `--reranker bm25`。5 条合成查询上，JSON dense、SQLite dense 和 SQLite+BM25 的 Hit@1、Hit@3、MRR@3 都为 1.0；BM25 改变了非目标候选顺序，但 toy benchmark 不足以证明 reranker 有真实收益。详细设计、端到端结果和局限见 [`reports/vector-store-reranker-v0.md`](reports/vector-store-reranker-v0.md)。

```powershell
python scripts/build_vector_store.py `
  --index outputs/rag-embedding-v1/index.json `
  --output outputs/vector-store-v0/index.sqlite

python scripts/retrieve_vector_store.py `
  --store outputs/vector-store-v0/index.sqlite `
  --input data/rag_demo/retrieval_benchmark.jsonl `
  --output outputs/vector-store-v0/bm25.jsonl `
  --device cuda `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --local-files-only

python scripts/evaluate_rag.py `
  --input outputs/vector-store-v0/bm25.jsonl `
  --output reports/vector-store-v0-bm25.json `
  --k 3
```

当前 SQLite 后端是透明的 reference implementation，查询采用 O(N) 全表扫描，不应包装成生产级 ANN/vector DB。下一步需要在真实授权知识库上扩大 benchmark，再比较 FAISS/Milvus/Chroma 和神经 reranker 的召回、延迟、内存与构建成本。

## 当前阶段：安全拒答 v0

新增 `--safe-mode`：在 Qwen 生成前检查高风险患者请求和检索置信度；命中时直接输出“资料不足，无法判断”，并跳过 `model.generate()`。9 条合成安全问题上，`dense min_score=0.6` 加高风险规则得到 decision accuracy=1.0、unsafe allow rate=0.0、false abstain rate=0.0。由于 benchmark 只有 9 条合成问题，这些数字不能解释为医疗安全保证。详细结果和失败样本见 [`reports/safety-v0.md`](reports/safety-v0.md)。

```powershell
python scripts/retrieve_vector_store.py `
  --store outputs/vector-store-v0/index.sqlite `
  --input data/rag_demo/safety_benchmark.jsonl `
  --output outputs/vector-store-v0/safety-bm25-threshold06.jsonl `
  --device cuda `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --min-score 0.6 `
  --local-files-only

python scripts/evaluate_safety.py `
  --input outputs/vector-store-v0/safety-bm25-threshold06.jsonl `
  --output reports/safety-v0-threshold06.json

python scripts/run_rag_qa.py `
  --model Qwen/Qwen3-1.7B `
  --store outputs/vector-store-v0/index.sqlite `
  --input data/rag_demo/safety_benchmark.jsonl `
  --output outputs/rag-qa-v2/safe-mode.jsonl `
  --device cuda `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --min-score 0.6 `
  --safe-mode `
  --greedy `
  --local-files-only
```

## 当前阶段：Constitutional AI-inspired v0

本阶段在已有安全拒答 v0 的基础上，增加一个可审计的“原则 → 批评 → 修订 → 再批评”闭环。原则版本为 `medical-safety-v1`，包括：不做个人诊断或处方、不在缺少依据时编造确定结论、避免绝对化承诺、不索取身份/病历/住址等敏感信息，以及拒答时给出清晰的限制说明和安全下一步。

当前实现是 deterministic v0：批评器使用透明规则，修订器使用固定安全模板；它不是让 Qwen 自己生成 critique 的完整 Constitutional AI，也没有调用外部模型 API。这样做的目的，是先把原则版本、违规证据、修订前后结果和失败样本固定下来，为后续 model-in-the-loop v1 留出可替换接口。

评测使用 12 条本地合成样本，不使用真实医疗数据。8 条初始回答触发原则违规，经过修订后 0 条仍违规，8/8 修订成功；两组期望标签准确率均为 1.0。这些结果只说明规则在这组小型合成 benchmark 上可运行，不能解释为医疗安全保证。

```powershell
python scripts/run_constitutional.py
```

实现见 [`src/qwen_medical_qa/constitutional.py`](src/qwen_medical_qa/constitutional.py)、[`scripts/run_constitutional.py`](scripts/run_constitutional.py)、[`configs/constitutional-ai-v0.yaml`](configs/constitutional-ai-v0.yaml) 和 [`data/constitutional/benchmark-v1.jsonl`](data/constitutional/benchmark-v1.jsonl)；完整结果与面试表述见 [`reports/constitutional-ai-v0.md`](reports/constitutional-ai-v0.md)。

## 当前阶段：Constitutional AI-inspired v1（model-in-the-loop）

本阶段把 v0 的 deterministic critic/reviser 替换为可插拔的 Qwen model-in-the-loop：模型先生成候选回答（`candidate-source=model`），再输出结构化 JSON critique，随后依据 critique 生成修订回答。所有原始回答、critic 原文、解析结果、修订结果和最终门禁结果都会保留在逐条 JSONL 中。

为了避免把模型的偶然输出当成安全保证，v1 仍保留 v0 规则作为 fail-closed hard gate：模型修订如果没有通过规则，就回退到确定性的安全模板。正式 12 条受控 benchmark 使用 4-bit NF4、Qwen3-1.7B、greedy、`enable_thinking=false`；模型 critic 识别初始违规 8/8、JSON 解析失败 0 条，模型 revision 单独通过规则 6/8（75%），剩余 2 条由规则硬门禁接管，最终 12/12 通过。最终 100% 是“模型批评/修订 + 规则兜底”的组合结果，不能归因于模型 revision 单独能力。

此外，2 条 model-source smoke 已验证 Qwen 候选生成、批评和修订调用能够离线运行；由于候选不是 benchmark 中的受控初始回答，不使用 benchmark 初始违规标签计算 critic accuracy。完整失败分析见 [`reports/constitutional-ai-v1.md`](reports/constitutional-ai-v1.md)。

```powershell
python scripts/run_constitutional_v1.py `
  --candidate-source benchmark `
  --local-files-only `
  --greedy `
  --load-in-4bit

python scripts/run_constitutional_v1.py `
  --candidate-source model `
  --limit 2 `
  --output outputs/constitutional-v1/model-smoke/results.jsonl `
  --summary outputs/constitutional-v1/model-smoke/summary.json `
  --local-files-only `
  --greedy `
  --load-in-4bit
```

实现见 [`src/qwen_medical_qa/constitutional_v1.py`](src/qwen_medical_qa/constitutional_v1.py)、[`scripts/run_constitutional_v1.py`](scripts/run_constitutional_v1.py) 和 [`configs/constitutional-ai-v1.yaml`](configs/constitutional-ai-v1.yaml)。

## 当前阶段：Constitutional AI-inspired v1.1（120 条分层安全 benchmark）

为避免 12 条样本的指标过于偶然，本阶段用 24 个场景族各生成 5 个变体，共 120 条本地合成样本；违规/安全各 60 条，按场景族划分为 dev 60、holdout 40、challenge 20，变体不会跨 split。数据生成器会检查 ID、场景族、平衡标签和重复问答对，元数据记录 SHA-256、标签来源及“无真实医疗数据/无专家标签”的限制。

扩展后的 120 条正式评测中，Qwen critic 的 recall=`1.0`、precision=`0.8108`、F1=`0.8955`，JSON 解析失败=`0`；安全样本 false-abstain rate=`0.2333`。模型 revision 按 60 条期望违规标签计算，单独修复 `43/60=71.67%`，剩余 20 条模型修订结果由 hard gate 接管，最终规则违规=`0/120`。分层结果显示 dev/holdout/challenge 的 revision success rate 分别为 `83.33%/85%/10%`，challenge 的隐私和紧急场景仍然是薄弱点。最终 0 条违规是模型与规则组合结果，不能归因于模型 revision 单独能力。

```powershell
python scripts/prepare_constitutional_benchmark.py

python scripts/run_constitutional_v1.py `
  --input data/constitutional/benchmark-v1.1.jsonl `
  --output outputs/constitutional-v1.1/results.jsonl `
  --summary outputs/constitutional-v1.1/summary.json `
  --candidate-source benchmark `
  --local-files-only `
  --greedy `
  --load-in-4bit
```

数据生成配置见 [`configs/constitutional-ai-v1.1.yaml`](configs/constitutional-ai-v1.1.yaml)，完整分析见 [`reports/constitutional-ai-v1.1.md`](reports/constitutional-ai-v1.1.md)。

真实知识库接入前，先复制 [`data/sources/rag-knowledge-base.template.yaml`](data/sources/rag-knowledge-base.template.yaml)，补齐来源、版本、许可证、隐私和专家审核字段；模板本身不代表任何真实数据已经获准使用。

脚本会记录每条样本的输入、模型输出、输入/输出 token 数、单条耗时和 tokens/s。第一阶段先不追求准确率数字，先确认模型、数据格式、生成参数和记录链路都能稳定运行。

## 当前阶段：CMB-Exam 闭域 RAG v1

在没有可公开上传的真实医学知识库时，使用已有来源记录的 CMB-Exam `train` split 做本地闭域实验。检索文档是“题干 + 选项 + 参考答案”的已解答训练例题，不是临床指南；原始 CMB、处理结果、派生语料和向量索引都保留在本地忽略目录，不能上传到 GitHub。

本阶段先做泄漏控制：按题干与选项去重，并删除 train/val 重叠；val 只用于查询和答案评测。由于没有人工标注的相关文档 ID，不报告 Recall@k/Hit@k/MRR。完整数据处理、配置、迁移矩阵、延迟和局限见 [`reports/cmb-rag-v1.md`](reports/cmb-rag-v1.md)。

```powershell
python scripts/prepare_cmb_rag.py

python scripts/build_embedding_index.py `
  --input data/processed/cmb-rag-v1/knowledge_base.jsonl `
  --output outputs/cmb-rag-v1/index.json `
  --chunk-size 512 `
  --chunk-overlap 0 `
  --device cuda `
  --batch-size 16 `
  --local-files-only

python scripts/build_vector_store.py `
  --index outputs/cmb-rag-v1/index.json `
  --output outputs/cmb-rag-v1/index.sqlite

python scripts/retrieve_vector_store.py `
  --store outputs/cmb-rag-v1/index.sqlite `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --output outputs/cmb-rag-v1/retrieval.jsonl `
  --device cuda `
  --batch-size 16 `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --local-files-only

python scripts/run_cmb_rag.py `
  --model Qwen/Qwen3-1.7B `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --retrieval outputs/cmb-rag-v1/retrieval.jsonl `
  --output outputs/cmb-rag-v1/rag-base-val.jsonl `
  --max-new-tokens 4 `
  --max-input-tokens 2048 `
  --seed 42 `
  --greedy `
  --local-files-only

python scripts/evaluate_benchmark.py `
  --input outputs/cmb-rag-v1/rag-base-val.jsonl `
  --output reports/cmb-rag-v1-rag-base-val.json
```

本次匹配运行中，直接基线为 107/240（44.58%），dense-only RAG 为 111/240（46.25%），dense + BM25 为 115/240（47.92%）。此前普通 greedy 重跑出现过 112 与 115 两个结果；补充 `--deterministic` 后，8 条 smoke 两次预测完全一致，全量结果与最新 BM25 结果 240/240 条一致。详细结果见 [`reports/cmb-rag-v1.md`](reports/cmb-rag-v1.md) 和 [`reports/reproducibility-v1.md`](reports/reproducibility-v1.md)。

## 当前阶段：可复现性与预测漂移 v1

基线和 CMB-RAG 脚本现在支持 `--deterministic`，会记录 seed、Torch/CUDA、cuDNN 和 CUBLAS 配置；`scripts/compare_predictions.py` 可对两次 JSONL 结果做不暴露题目文本的漂移分析。

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

python scripts/compare_predictions.py `
  --left outputs/cmb-rag-v1/rag-base-val.jsonl `
  --right outputs/cmb-rag-v1/rag-deterministic-val.jsonl
```

`--deterministic` 不是跨机器和跨软件版本的绝对复现保证；当前仍需锁定环境版本、模型 revision，并补充多 seed 方差。完整说明见 [`reports/reproducibility-v1.md`](reports/reproducibility-v1.md)。

## 当前阶段：FAISS HNSW ANN v1

SQLite vector store 仍保留为 O(N) reference backend，同时增加可选的 FAISS `IndexHNSWFlat`。FAISS 使用归一化向量的 inner product，等价于 cosine similarity；chunk 元数据放在 `.faiss.meta.json` sidecar。可选依赖见 [`requirements-rag-ann.txt`](requirements-rag-ann.txt)，默认环境不强制安装。

```powershell
python -m pip install -r requirements-rag-ann.txt

python scripts/build_faiss_index.py `
  --index outputs/cmb-rag-v1/index.json `
  --output outputs/cmb-rag-v1/index.faiss `
  --hnsw-m 32 `
  --ef-construction 80 `
  --ef-search 64

python scripts/retrieve_faiss.py `
  --index outputs/cmb-rag-v1/index.faiss `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --output outputs/cmb-rag-v1/retrieval-faiss-bm25.jsonl `
  --device cuda `
  --batch-size 16 `
  --candidate-k 5 `
  --top-k 3 `
  --reranker bm25 `
  --local-files-only

python scripts/compare_retrievals.py `
  --left outputs/cmb-rag-v1/retrieval.jsonl `
  --right outputs/cmb-rag-v1/retrieval-faiss-bm25.jsonl `
  --top-k 3
```

在 4,984 条 CMB 文档、240 条查询上，FAISS HNSW 平均 search 约 0.2383 ms，SQLite scan 约 495.93 ms；top-3 完整排名 239/240 一致，FAISS+BM25 与 SQLite+BM25 的最终答案 240/240 一致。该延迟不包含 embedding 和生成，也不代表生产 SLA。详细结果见 [`reports/faiss-ann-v1.md`](reports/faiss-ann-v1.md)。

## 当前阶段：Transformer Cross-Encoder Reranker v1

在 dense candidate top-k 之后，项目现在提供可选的 Transformer Cross-Encoder 重排器。默认实现是 [`TransformerCrossEncoderReranker`](src/qwen_medical_qa/neural_reranker.py)，默认模型为 [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)。它对同一个 query 和每个 candidate passage 做成对打分，再按 raw logit 排序；raw logit 只能用于同一候选集内排序，不能当作概率。

BM25、神经 reranker 和不重排路径共用 `RerankedResult`，并且只有选择 `--reranker neural` 时才加载 Transformer 模型。模型下载不进入 Git 仓库，建议先设置本地缓存并使用 `--local-files-only` 做复现实验：

```powershell
$env:HF_HOME = "$PWD/.hf-cache"
$rerankerModel = "$PWD/models/bge-reranker-v2-m3"
python -m pip install -r requirements-rag-reranker.txt

python scripts/retrieve_faiss.py `
  --index outputs/cmb-rag-v1/index.faiss `
  --input data/processed/cmb-rag-v1/qa_benchmark.jsonl `
  --output outputs/cmb-rag-v1/retrieval-faiss-neural.jsonl `
  --device cuda `
  --batch-size 16 `
  --candidate-k 10 `
  --top-k 3 `
  --reranker neural `
  --reranker-model $rerankerModel `
  --reranker-device cuda `
  --reranker-batch-size 8 `
  --reranker-max-length 512 `
  --local-files-only
```

模型文件已手动整理到项目 `models/bge-reranker-v2-m3`，并通过 Transformers 离线加载与项目封装的单样本前向校验。240 条 CMB val 上，FAISS+neural candidate top-10→top-3 的平均重排延迟为 303.97 ms、P95 为 466.81 ms，Qwen deterministic Accuracy 为 45.83%（110/240）；同一候选规模下 FAISS+BM25 为 47.92%（115/240）。neural 改变了 65/240 条最终预测，但没有带来收益，因此保留为失败但有面试价值的消融结果。详细结果见 [`reports/neural-reranker-v1.md`](reports/neural-reranker-v1.md)。

详细阶段记录见 [`reports/neural-reranker-v1.md`](reports/neural-reranker-v1.md)。

Windows CUDA 环境的 PyTorch 安装命令会根据驱动和 CUDA wheel 选择单独确定，不把一个可能失效的固定命令写死在项目中。

## 项目收尾：端到端演示与最终总表

当前版本已经完成基线、QLoRA/SFT、RAG、DPO、GRPO 和 Constitutional AI-inspired 安全层实验。最终结果总表、项目完成标准、限制和简历版表述见 [`reports/project-final-summary.md`](reports/project-final-summary.md)。

为了把生成式 RAG 与安全门串成一条可展示链路，`scripts/run_rag_qa.py` 新增了可选的 `--constitutional-gate`：

```powershell
python scripts/run_rag_qa.py `
  --model Qwen/Qwen3-1.7B `
  --index outputs/rag-embedding-v1/index.json `
  --input data/rag_demo/qa_benchmark.jsonl `
  --output outputs/project-final-demo/qwen-rag-gated.jsonl `
  --device cuda `
  --top-k 3 `
  --max-new-tokens 64 `
  --max-input-tokens 2048 `
  --safe-mode `
  --constitutional-gate `
  --seed 42 `
  --greedy `
  --local-files-only
```

开启该开关后，系统会在 Qwen 生成后执行确定性 Constitutional hard gate，保留 `raw_answer`、初始 critique、是否 fallback 和最终 critique。5 条合成演示问题的本地 smoke 中，5/5 条最终通过规则，fallback 为 0；这只是 pipeline smoke，不是医疗准确率或临床安全结论。

## 面试复盘重点

完成每个阶段后，需要至少能回答：

1. 为什么先做基线，而不是直接微调？
2. 测试集如何避免数据泄漏？
3. QLoRA 解决了什么显存问题，代价是什么？
4. RAG 与微调分别解决什么问题？
5. 每个指标的定义、测试集、硬件和生成参数是什么？
6. 哪些失败实验被保留，为什么没有采用？

## 许可证与开源计划

仓库最终公开前，需要补充项目许可证、基础模型许可证、数据集许可证和第三方依赖说明。原始医疗数据、API 密钥和完整基础模型权重不提交到 GitHub；仓库应提供合法数据的下载说明、版本号和校验信息。

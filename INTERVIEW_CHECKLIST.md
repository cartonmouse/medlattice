# 阶段 A 面试复盘清单：基线推理

本文件不是背诵稿。每次实验完成后，把实际配置、结果和失败现象补进去。

## 需要能讲清楚的问题

### 1. 为什么先做基线？

因为没有固定测试集和基线，就无法判断 QLoRA、RAG 或 DPO 的改动到底带来了收益，还是只是换了提示词、数据或生成参数。

当前证据：

- 测试集版本：待填写
- 样本数：待填写
- 基线结果文件：待生成

### 2. 为什么选择 Qwen3-1.7B？

回答应包含模型规模、中文能力、可本地运行性、后续参数高效微调成本，以及为什么没有直接选择更大的模型。不要只回答“因为简历上写了”。

当前版本：`Qwen/Qwen3-1.7B`

### 3. 基线的生成设置是什么？

需要说清楚：

- 是否开启 thinking
- `max_new_tokens`
- temperature、top-p、top-k
- 是否使用贪心解码
- 随机种子
- prompt 模板

### 4. 延迟和 tokens/s 是怎么测的？

本项目第一阶段按单条样本记录：模型生成开始到结束的时间、输出 token 数和 `output_tokens / elapsed_seconds`。后续还要补充批大小、并发数、输入长度、GPU 型号和 p50/p95，不能把不同测试条件的数字直接比较。

### 5. 当前实验有什么限制？

- 样例数据不是正式医疗数据集。
- 没有经过领域专家审核，不能代表临床准确性。
- RTX 4060 Laptop 8GB 只适合做小规模验证。
- QLoRA、RAG 和 DPO 已有结果，但结果只适用于当前离线 benchmark；DPO 没有提升 CMB val accuracy。

## 完成阶段 A 后必须补充

- [x] 实际环境 JSON
- [x] 模型下载/版本信息
- [x] 5 条样例的输出
- [x] 平均延迟、p50、p95 和 tokens/s
- [x] 1 个失败案例及原因分析
- [x] 一条可复现命令

## 阶段 A-2：固定 benchmark

当前 v0 benchmark 的结果是 12/12，但它是人工编写的术语选择题，只能证明模型能按要求输出选项，不能证明真实医疗领域能力。

面试时需要能回答：

- 为什么阶段 A-2 使用选择题 exact match，而不是直接用开放式答案字符串匹配？
- invalid output rate 代表什么？模型输出一段解释时，评测器如何抽取选项？
- 为什么 12 条样例 100% 不能写成“医疗准确率 100%”？
- 正式数据集接入后，如何说明来源、许可证、数据泄漏和测试集划分？
- 如果 QLoRA 后分数下降，如何区分数据问题、训练问题和评测问题？

## 阶段 B：CMB-Exam 数据工程

面试时需要能回答：

- 为什么选择 CMB-Exam，而不是来源和许可证不明确的在线问答网页？
- 训练、验证、测试集分别如何使用？为什么训练时不能读取测试集？
- 如果 test split 没有公开答案，为什么不能报告 test accuracy？如何用 val 做有标签评测？
- 为什么第一版只保留单项选择题？多项选择题的答案规范化和评测口径有什么不同？
- 固定随机种子和 SHA-256 元数据如何让别人复现同一批本地处理结果？
- 为什么不把原始数据放进 GitHub？别人如何根据来源记录重新下载？

## 阶段 C：SFT/QLoRA

面试时需要能回答：

- 为什么训练和基线必须复用同一套 prompt 模板？
- 为什么 assistant-only loss 要屏蔽 system/user token？如果不屏蔽会有什么影响？
- 4-bit NF4、LoRA rank、alpha、dropout 和 target modules 分别解决什么问题？
- 为什么 8 条样本 smoke training 的 87.5% 不能写成正式微调效果？正式对比如何控制变量？
- 如果 8GB 显存 OOM，优先调整 sequence length、batch size、gradient accumulation 还是 LoRA rank？为什么？

## 阶段 D：DPO/QLoRA-DPO

面试时需要能回答：

- 为什么把 DPO 放在 SFT 之后？SFT、DPO、RLHF/PPO 和 reward model 分别解决什么问题？
- DPO 样本中的 `prompt`、`chosen` 和 `rejected` 分别是什么？本项目如何构造它们？
- 为什么“参考答案 + 随机错误选项”只能算合成偏好，而不能算人类或临床专家偏好？
- policy 为什么从已有 `outputs/qlora-cmb-v1` 初始化？reference policy 为什么要冻结？
- `beta` 控制什么？本次为什么使用 `beta=0.1`、learning rate=`5e-6`、1 epoch 和 gradient accumulation=8？
- 4-bit NF4、LoRA 和 DPO 的关系是什么？为什么第一次 fp16 smoke 失败，后来改成 bf16？
- DPO 的 train/eval loss、reward margin、reward accuracy 与最终 benchmark accuracy 有什么区别？
- 本次 DPO 是否提升了结果？如何报告 121/240、50.42%、232/240 相同、8 条变化以及一条变好/一条变坏？
- 如果下一步要证明 DPO 的真实价值，应该如何获得合法偏好数据、做领域审核、设计开放式/安全偏好对和独立评测？

### 本阶段事实卡片

- 偏好数据：CMB-Exam 本地 train 5,000 条，DPO train/eval 为 4,800/200，seed=42；最终 val 240 条隔离。
- 负例策略：从其他选项随机采样错误字母，`negative_strategy=random_wrong_choice`。
- 训练：Qwen3-1.7B 4-bit NF4，policy 从 QLoRA/SFT adapter 初始化，reference 为冻结副本，`beta=0.1`，1 epoch，600 optimizer steps，bf16。
- 正式训练：train loss 0.6273，DPO eval loss 0.6037，最后记录 reward margin 0.1947、reward accuracy 0.865，峰值 CUDA allocated 约 3.94 GB。
- 隔离 val：SFT 121/240，DPO 121/240；232 条预测相同，8 条变化；SFT→DPO 一条由错变对、一条由对变错。
- 诚实结论：DPO 训练链路和偏好目标优化成功，但当前合成偏好与选择题字母任务没有证明下游准确率提升，更不能解释为医疗能力提升。

## 阶段 D-2：公开医学偏好数据 DPO v2

面试时需要能回答：

- 为什么没有直接下载完整 `train.json`？如何用固定 revision、文件 SHA-256 和小 split 完成可审计实验？
- 为什么训练候选要求 `chosen_score > rejected_score`？分数相同或方向相反的 pair 会带来什么噪声？
- 为什么 `test.json` 中的 163 条 `label_type=human` 只能做外部 holdout，不能混入训练？
- 为什么要检查 train/human prompt overlap？如果存在重叠，偏好评测会怎样被高估？
- 公开数据是英文、CMB 是中文选择题，为什么这会产生任务迁移风险？
- 为什么正式实验采用 448 train、64 eval、512 token，而不是把 1024 token smoke 直接扩大？如何解释 8GB 显存和长回答的计算代价？
- 为什么偏好 proxy 的 chosen accuracy 没提升时，不能只看 DPO loss 下结论？

### 本阶段事实卡片

- 数据源：`TsinghuaC3I/UltraMedical-Preference`，固定 revision `761eb7935310ba662a96d93c5af342e5269d5759`；只下载 `dev.json` 与 `test.json`。
- 过滤：2,232 条 dev 中去掉 145 条非严格分数方向和 13 条重复 prompt，得到 2,074 条；正式抽取 512 条，448/64 train/eval；human holdout 163 条，overlap=0。
- 正式训练：Qwen3-1.7B + CMB SFT adapter，4-bit NF4/LoRA/bf16，56 optimizer steps，train loss 0.6822，eval loss 0.6760，peak CUDA allocated 4,880.26 MB。
- human preference proxy：512 token 口径 SFT/DPO 都是 57/163（34.97%）；1024 token 口径都为 56/163（34.36%）。
- CMB 迁移：SFT 121/240（50.42%），public DPO v2 118/240（49.17%）；234/240 预测相同，4 条回退，1 条修复。
- 诚实结论：公开英文偏好数据的 DPO 链路跑通，但本轮没有证明下游提升，并出现轻微中文任务负迁移；应作为失败消融，而不是主模型效果。

## 阶段 D-3：CMB train-only hard-negative DPO

面试时需要能回答：

- 为什么从“随机错误选项”改成“模型分数最高的错误选项”？hard negative 与随机负例分别在验证什么？
- 如何用 SFT adapter 对每道题的选项计算条件 log-probability？为什么不能把模型生成的整段文本概率直接当成公平的选项分数？
- 为什么 hard negative 仍然不能算人类或医生偏好？模型 top choice 错了时，为什么把它作为 rejected 有训练价值但也有风险？
- 如何证明 hard negative 没有读取或污染 val？输入 SHA、train/eval ID、val ID 交集和选项合法性分别怎么审计？
- 为什么 DPO eval loss/reward margin 变好，CMB val accuracy 仍可能不变甚至下降？这说明了训练目标和下游任务之间什么关系？
- 发现 CMB 中存在 3/4/6 选项题后，为什么应该使用动态选项数，而不是默认所有样本都是 A-E？

### 本阶段事实卡片

- 数据来源：本地 CMB-Exam train split 5,000 条；SFT adapter：`outputs/qlora-cmb-v1`；输入 SHA-256：`d30b88c9530b695e4459613aacffc123a3924225445bf2a636b2490b2f12ea36`。
- hard-negative 生成：3,712 条当前预测正确、1,288 条当前预测错误；错误子集中 1,288/1,288 条 rejected 都是模型 top choice；最终偏好切分 4,800/200，seed=42，val overlap=0。
- 正式训练：Qwen3-1.7B + 4-bit NF4/LoRA/bf16，`beta=0.1`，1 epoch，600 optimizer steps；train loss=0.6636，eval loss=0.6553，eval reward accuracy=0.740，reward margin=0.0804，peak CUDA allocated≈3.94 GB。
- 独立 val：SFT 121/240（50.42%），hard-negative DPO 120/240（50.00%）；230/240 预测相同，10 条变化；SFT 正确→DPO 错误 2 条，SFT 错误→DPO 正确 1 条。
- 诚实结论：hard negative 验证了“用模型混淆边界构造偏好对”的工程链路，但没有证明当前中文选择题任务的泛化收益；它应作为有失败分析的 DPO 消融，不应作为主模型效果或医疗能力证据。

## 阶段 D-4：CMB train-only GRPO v0

面试时需要能回答：

- GRPO 与 DPO 的核心差别是什么？为什么 GRPO 不需要显式 `chosen/rejected` 偏好对，而是对同一 prompt 采样多个 completion 后做 group-relative advantage？
- 当前项目为什么能使用 exact-match reward？这个奖励如何由“答案正确”和“格式合法”两部分组成？它的适用范围和局限是什么？
- 为什么 GRPO 仍然从 SFT adapter 初始化，并使用冻结的 SFT adapter 作为 reference policy？`beta` 在这里控制什么？
- `num_generations=4`、`loss_type=grpo`、`max_completion_length=4`、learning rate=`5e-7` 和 600 steps 分别怎样影响显存、探索和训练稳定性？
- 为什么 GRPO eval reward 变好不能直接等价于最终 CMB val accuracy 变好？怎样避免把 train/eval reward 当成最终 benchmark？
- 为什么最终奖励中 format reward 很稳定，而 exact-match reward 仍有波动？如果 reward 过窄或所有 completion 都同分，会发生什么？
- 如何审计 GRPO 没有读取最终 val？本阶段 train/eval 切分、val overlap、seed 和输出哈希分别记录在哪里？
- GRPO 最终是否提升？如何准确报告 122/240、50.83%、233/240 相同、1 条回退、2 条修复，并说明“轻微正向但未证明稳定提升”？

### 本阶段事实卡片

- 数据来源：本地 CMB-Exam train split 5,000 条；GRPO train/eval 为 4,800/200，seed=42；输入 SHA-256=`d30b88c9530b695e4459613aacffc123a3924225445bf2a636b2490b2f12ea36`，val overlap=0。
- 奖励：exact reference option=`1.0`；exact single-letter format=`0.2` 权重；每个 prompt 采样 4 个 completion；没有使用人工偏好标签或 reward model。
- 正式训练：Qwen3-1.7B + CMB SFT adapter，4-bit NF4/LoRA/bf16，`loss_type=grpo`、`beta=0.04`、learning rate=`5e-7`、600 optimizer steps；耗时约 27.4 分钟，peak CUDA allocated≈2.04 GB。
- 内部 eval：第 600 步 exact-match reward mean=`0.6225`，format reward mean=`1.0`，KL=`3.65e-4`；这些指标不等于最终 val accuracy。
- 独立 val：SFT 121/240（50.42%），GRPO v0 122/240（50.83%）；233/240 条预测相同，1 条由对变错，2 条由错变对。
- 诚实结论：GRPO v0 在本次单 seed、窄任务上带来 1 道题的净提升，证明了可验证奖励 GRPO 链路和小幅正向结果，但不足以证明稳定泛化或医疗能力提升。

## 阶段 E：Constitutional AI-inspired v0

面试时需要能回答：

- Constitutional AI 和普通的安全 prompt 有什么区别？本项目为什么先把安全要求显式写成版本化原则，而不是只依赖一句 system prompt？
- 本项目的 C1～C5 原则分别解决什么问题？为什么“证据与 grounding”和“校准不确定性”要分开？
- 一次批评-修订链路的输入、输出和审计字段是什么？如何证明修订没有把原始违规悄悄覆盖掉？
- 为什么本阶段称为 Constitutional AI-inspired v0，而不是完整的 Constitutional AI？规则批评器与 Qwen model-in-the-loop critic 的差别是什么？
- 12 条合成 benchmark 上 8 条违规变成 0 条，为什么不能直接说系统安全性达到 100%？怎样补充专家标注、对抗改写和开放式回答评测？
- 如果规则产生 false positive 或漏掉同义表达，下一版如何改进？如何把规则保留为硬约束，同时加入模型批评和人工抽检？
- 为什么先做可审计的 deterministic baseline，再考虑 PPO/GRPO 等训练方法？它们解决的是“生成策略优化”，还是“安全边界定义”？

### 本阶段事实卡片

- 原则版本：`medical-safety-v1`；C1 范围与处置安全、C2 证据与 grounding、C3 校准不确定性、C4 隐私、C5 最小可用拒答。
- 数据：12 条本地合成 benchmark；不使用真实医疗数据，不调用外部模型 API，不上传原始数据。
- 结果：初始违规 8/12，修订后违规 0/12，成功修订 8/8；初始违规标签准确率和最终安全标签准确率均为 1.0。
- 实现：`critique_answer()` 记录原则 ID、严重性和证据；`revise_answer()` 选择固定安全模板；每条样本保留 initial answer、initial critique、revised answer 和 revised critique。
- 诚实结论：这是安全工程的可复现规则基线，不是临床安全证明，也不是完整的 LLM 自我批评训练。下一步应在合法、脱敏、经领域审核的开放式数据上比较规则 critic、Qwen critic 和人工复核的一致性。

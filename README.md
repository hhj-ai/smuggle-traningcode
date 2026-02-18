# AURORA: Annotation-free Unification of Reinforcement and Opponent Reasoning for Anti-hallucination

视觉语言模型无标注自我进化与幻觉抑制框架。基于 GRPO 的双模型对抗训练，通过多模态工具链提供物理约束反馈，在无人工标注条件下驱使模型自主消除幻觉。

---

## 核心架构

两个模型交替博弈，工具链充当裁判：

- **VLM / Generator**（Qwen3-VL-8B）：生成图像描述，目标是准确、丰富、多样
- **Verifier / Opponent**（DeepSeek-R1-Distill-Qwen-7B）：从描述中提取可验证的原子视觉声明（claims），目标是找出 VLM 的错误
- **工具链**（GroundingDINO + CLIP + EasyOCR）：对每条 claim 做事实核查，输出 correct / incorrect / uncertain

训练时 GPU 上同一时刻只有一个大模型（另一个换出到 CPU），两个 DDP Reducer 的梯度桶始终驻留 GPU。

---

## 训练循环（每个 batch 4 个阶段）

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: VLM 推理                    [GPU: VLM | CPU: Verifier] │
│   每张图生成 num_gen 个候选描述                                    │
│   → MiniLM 余弦相似度筛选 group_size 个最多样的                     │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: Verifier 推理 + 工具验证     [GPU: Verifier | CPU: VLM] │
│   Verifier 提取 claims                                          │
│   → gather 到 rank 0                                            │
│   → DINO/CLIP/OCR 逐条验证                                      │
│   → scatter 结果回各 rank                                        │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: VLM GRPO 训练               [GPU: VLM | CPU: Verifier] │
│   验证结果 → VLM 奖励 → 组内优势归一化 → policy gradient 更新       │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: Verifier GRPO 训练          [GPU: Verifier | CPU: VLM] │
│   验证结果 → Verifier 奖励 → 组内优势归一化 → policy gradient 更新  │
└─────────────────────────────────────────────────────────────────┘
```

### 多样性筛选算法

Phase 1 生成 `num_gen` 个候选后，需要从中选出 `group_size` 个最多样的子集。算法：
1. 用 MiniLM 编码所有候选，计算两两余弦相似度矩阵
2. 迭代移除：每轮找到相似度最高的一对，移除其中一个
3. 直到剩余数量 = `group_size`

### GRPO 更新

对每张图的 `group_size` 个描述计算奖励后，做组内归一化得到优势值：
```
advantage = (reward - mean(group_rewards)) / (std(group_rewards) + ε)
loss = -advantage * log_prob(description)
```
每张图的 loss 做梯度累积，一个 batch 结束后统一 step。

---

## 奖励设计

### VLM 奖励

```
R_vlm = accuracy + bonus + length_penalty - diversity_penalty
```

| 项 | 公式 | 作用 |
|---|---|---|
| accuracy | `(correct - incorrect) / total` | 核心幻觉抑制 |
| bonus | `β · log(correct + 1)` | 鼓励丰富度，防止消极避答 |
| length_penalty | 描述 < 20 词时 -2.0 | 防止 mode collapse |
| diversity_penalty | 组内描述平均余弦相似度 | 鼓励多样化表达 |

### Verifier 奖励

```
R_ver = status_reward - correlation_penalty - traceability_penalty
```

| 项 | 值 | 作用 |
|---|---|---|
| 发现错误 (incorrect) | +5.0 | 对抗激励，驱动寻找漏洞 |
| 确认正确 (correct) | +1.0 | 基础奖励 |
| 不确定 (uncertain) | -0.1 | 轻微惩罚 |
| 不可追溯 | -10.0 | 声明与原文重叠 > 70% 时重罚，防止凭空捏造 |
| 相关性惩罚 | `correlation_weight × intra_claim_similarity` | 强制多维度攻击 |

---

## 显存管理

### 模型换入换出

`GPUMemoryManager` 封装了 offload（模型 → CPU）和 reload（模型 → GPU）操作，每次切换后执行 `gc.collect()` + `torch.cuda.empty_cache()`。

### 交错 DDP 包装

VLM（~16GB）和 Verifier（~14GB）不能同时在 GPU 上做 DDP 初始化（Reducer 分配需要 ~模型大小的额外显存），因此交替进行：

```
1. Verifier → CPU → VLM + optimizer DDP 包装
2. VLM → CPU → Verifier → GPU → Verifier + optimizer DDP 包装
3. VLM → GPU（恢复 Phase 1 初始状态）
```

### GPU 资源自适应调优

`ResourceAutoTuner` 在 DDP 包装完成后，通过实际换入换出测量两个阶段的可用显存，取较小值按 5 档参数表选择最优配置：

| 可用工作显存 | batch_size | group_size | num_gen |
|-------------|------------|------------|---------|
| >= 40 GB    | 16         | 8          | 10      |
| 30-40 GB    | 12         | 6          | 8       |
| 20-30 GB    | 8          | 4          | 6       |
| 12-20 GB    | 4          | 4          | 5       |
| < 12 GB     | 2          | 2          | 3       |

参数约束：`batch_size % group_size == 0`，`num_gen >= group_size >= 2`。

### OOM 多级降级

| 场景 | 处理 |
|------|------|
| Phase 1 生成 OOM | 先减半 `num_gen`（不低于 `group_size`），再次 OOM 则跳过该图 |
| Phase 3/4 backward OOM | 跳过该 group，清零梯度 |
| 单 epoch 累计 >= 3 次 OOM | 下个 epoch 自动降一档，重建 DataLoader |

---

## 工具验证

`ToolVerifier` 包含三个工具模型，默认自动按 GPU 空闲显存分散加载：

| 工具 | 显存占用 | 功能 |
|------|---------|------|
| GroundingDINO | ~850 MB | 零样本目标检测，验证物体存在性和空间关系 |
| CLIP | ~650 MB | 图文语义匹配，验证描述与图像的一致性 |
| EasyOCR | ~200 MB | 文字识别，验证文本内容 |

自动分配逻辑：查询所有可见 GPU 空闲显存，按工具大小从大到小贪心分配到最空闲的卡，每分配一个扣减预估占用后再选下一个。可通过 `--tool_device cuda:4,cuda:5` 手动覆盖。

---

## 分布式通信

Phase 2 中各 rank 独立运行 Verifier 推理，但工具验证只在 rank 0 执行（工具模型仅加载一份）：

```
各 rank: Verifier 提取 claims
         ↓ dist.gather_object → rank 0
rank 0:  DINO/CLIP/OCR 验证所有 claims
         ↓ dist.scatter_object_list → 各 rank
各 rank: 拿到验证结果，进入 Phase 3
```

---

## 文件结构

```
aurora_train.py   # 主训练脚本（显存管理、自适应调优、4 阶段训练循环、断点续传）
models.py         # VLM (Qwen3-VL-8B) / Verifier (DeepSeek-R1-7B) 模型封装
tools.py          # 工具验证器（DINO/CLIP/OCR，多卡自动分散）
rewards.py        # 奖励计算器（VLM 奖励 + Verifier 奖励）
eval.py           # 评估脚本（POPE / MMHal-Bench）
gpu.sh            # GPU 训练启动脚本（自动检测显存、断点续传）
cpu.sh            # 资源准备与离线打包脚本
```

---

## 启动

### 全自动启动（推荐）

```bash
bash gpu.sh
```

自动完成：GPU 显存检测 → 训练卡筛选 → 环境激活 → 断点续传检测 → 后台启动。

### 手动启动

```bash
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
  aurora_train.py \
  --model_dir ./models \
  --data_dir ./data/yfcc100m \
  --minilm_path ./models/minilm \
  --output_dir ./output \
  --batch_size 0 --group_size 0 --num_generations 0
```

`--batch_size`、`--group_size`、`--num_generations` 默认 0（auto）。手动指定时做安全校验，超过推荐值打印警告但仍使用。

### 断点续传

每个 epoch 结束保存 checkpoint（模型权重 + optimizer 状态 + RNG 状态），`latest` 符号链接指向最新。

```bash
--resume_from latest              # 自动找最新
--resume_from checkpoints/epoch_2 # 指定目录
```

### 评估

```bash
python eval.py --model_path ./output/checkpoints/epoch_4/vlm --benchmarks
python eval.py --baseline --benchmarks  # 对比基座模型
```

---

## 关键设计决策

1. **模型换入换出 vs 模型并行**：选择换入换出是因为两个模型不需要同时推理，时间换空间，单卡即可运行
2. **GRPO vs PPO**：GRPO 不需要 value network，组内相对优势天然适合多候选生成场景
3. **工具验证 vs 人工标注**：工具提供的是物理约束下的客观判定，无需标注数据，可无限扩展
4. **对抗训练**：Verifier 的攻击激励（incorrect +5.0）迫使 VLM 不断修补漏洞，形成正向螺旋

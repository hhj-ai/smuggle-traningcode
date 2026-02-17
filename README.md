# AURORA: Annotation-free Unification of Reinforcement and Opponent Reasoning for Anti-hallucination

<p align="center">
  <b>视觉语言模型无标注自我进化与幻觉抑制框架</b><br>
  <i>Designed for High-Performance Compute Clusters (e.g., 8x NVIDIA H200)</i>
</p>

---

## 🌟 核心理念 (The Idea)

**AURORA** 是一种基于强化学习（RL）的高性能 VLM 训练框架。它的核心在于通过**自对弈（Self-play）**机制，在无需任何人工标注的情况下，利用多模态工具链提供的“物理规律”作为反馈，驱使模型自主识别并消除幻觉。

### 循环流程
1.  **VLM 生成 (Generator)**: `Qwen3-VL-8B` 对无标注图片（YFCC100M）生成多份详细描述。
2.  **多样性筛选**: 利用 `MiniLM` 对生成的描述进行 Embedding 聚类，保留最具代表性的样本。
3.  **对手推理 (Opponent)**: `DeepSeek-R1-7B` 扮演审核员，从描述中提取原子视觉断言（Claims）。
4.  **物理真理验证**: 调用 `Grounding DINO` (空间)、`EasyOCR` (字符)、`CLIP` (语义) 对断言进行客观验证。
5.  **对抗奖励生成**: 根据“找茬”成功率与溯源准确性，生成对抗性奖励信号。
6.  **GRPO 训练**: 采用组相对策略优化算法，同步提升 VLM 的严谨性与 Verifier 的攻击性。

---

## 📊 奖励机制 (Reward Dynamics)

| 角色 | 奖励项 | 数学逻辑 / 意义 |
| :--- | :--- | :--- |
| **VLM** | **Accuracy** | $(Correct - Incorrect) / Total$ (核心幻觉抑制) |
| | **Bonus** | $\beta \cdot \log(Correct + 1)$ (鼓励丰富度，防止消极避答) |
| | **Penalty** | 长度太短惩罚 (防止 Mode Collapse) |
| **Verifier**| **Attack** | 揪出错误 +5.0 (驱动寻找物理约束下的系统漏洞) |
| | **Traceable**| 无法溯源 -10.0 (防止审核员凭空捏造) |
| | **Diversity**| 减去相似性项 (强制多维度攻击) |

---

## 🛠️ 8x H200 极致稳定性优化

针对“内核版本低 (4.18)、RAM 有限、断网”的特殊集群环境，本项目实现了以下工业级加固：

*   **Sequential Loading**: 8 个进程排队加载模型（RAM 占用从 240G $\rightarrow$ 30G），彻底杜绝 SIGKILL。
*   **Timeout Guard**: 分布式握手超时设为 4 小时，防止大模型加载期间进程掉线。
*   **VRAM Efficiency**: 全参数训练开启 **Gradient Checkpointing**，配合 **BF16** 与 **SDPA** 加速。
*   **Asset Isolation**: 资产（模型/数据）与代码分离存储在 `../aurora_resources`，防止 Git Pull 误删。

---

## 📂 项目结构

```text
.
├── aurora_train.py    # 核心训练逻辑 (GRPO Implementation)
├── models.py          # Qwen3-VL & DeepSeek-R1 离线加载封装
├── tools.py           # 裁判工具链 (DINO, CLIP, OCR)
├── rewards.py         # 对抗性奖励函数定义
├── eval.py            # Benchmark (POPE, MMHal-Bench) 评测脚本
├── gpu.sh             # 生产环境一键稳定启动脚本
└── cpu.sh             # 资源准备与离线打包脚本
```

---

## 🚀 快速开始

### 1. 环境与资源准备 (CPU 服务器)
在有网的环境下运行，它会将所有大文件精准下载到 `../aurora_resources`：
```bash
bash cpu.sh
```

### 2. 生产环境启动 (GPU 服务器)
直接在 8x H200 服务器上运行。脚本会自动激活环境、同步资源并后台启动：
```bash
bash gpu.sh
```
*实时查看训练日志:* `tail -f train_stable_*.log`

### 3. 基准测试 (Evaluation)
生成论文对比数据，支持 Baseline 与 Ours 的一键对比：
```bash
# 评估原版模型
python eval.py --baseline --benchmarks
# 评估训练后的模型
python eval.py --model_path ../aurora_resources/output/checkpoints/epoch_4/vlm --benchmarks
```

---

## 🔬 待证明的科学问题
1.  **无标注进化**: 证明不使用任何 Label，仅靠物理约束反馈能否提升 VLM 的视觉理解上限。
2.  **攻击性的价值**: 对比 `Attack Weight=5.0` 与随机提问 Baseline，证明“寻找漏洞”比“随机验证”能带来更高的模型鲁棒性。
3.  **物理约束有效性**: 验证 DINO 和 OCR 提供的空间/符号约束对解决复杂幻觉的不可替代性。

---
## 📝 License
[MIT License](LICENSE)

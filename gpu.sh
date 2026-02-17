#!/bin/bash
# --- 1. 绝对路径定义 (Source & Destination) ---
# 当前代码运行的绝对路径
CODE_ROOT=$(cd "$(dirname "$0")"; pwd)
# 外部资源存储的绝对路径 (Git 仓库之外)
RES_ROOT=$(realpath "$CODE_ROOT/../aurora_resources")

# 你的原始资产绝对路径 (生产环境源)
SRC_VLM="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/models/Qwen3-VL-8B-Instruct"
SRC_VER="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/models/DeepSeek-R1-Distill-Qwen-7B"
SRC_MINI="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/models/minilm"
SRC_DATA="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/data/yfcc100m"
SRC_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/aurora_env"

echo "📂 资源目标目录: $RES_ROOT"

# --- 2. 增强型同步逻辑 ---
# 使用 rsync (如果可用) 或 cp -r 进行物理复制
sync_item() {
    local src=$1
    local dest=$2
    local label=$3
    if [ ! -d "$dest" ]; then
        echo "📦 正在从绝对路径复制 $label..."
        mkdir -p "$(dirname "$dest")"
        # 优先使用 rsync，它支持断点续传且更稳定
        if command -v rsync >/dev/null 2>&1; then
            rsync -a "$src/" "$dest/"
        else
            cp -r "$src" "$dest"
        fi
        echo "✅ $label 同步至 $dest"
    else
        echo "✔️ $label 已在目标位置。"
    fi
}

# 物理同步：模型、工具、数据、环境
sync_item "$SRC_VLM" "$RES_ROOT/models/Qwen3-VL-8B-Instruct" "VLM模型"
sync_item "$SRC_VER" "$RES_ROOT/models/DeepSeek-R1-Distill-Qwen-7B" "Verifier模型"
sync_item "$SRC_MINI" "$RES_ROOT/models/minilm" "MiniLM编码器"
sync_item "$SRC_DATA" "$RES_ROOT/data/yfcc100m" "YFCC数据集"
sync_item "$SRC_ENV"  "$RES_ROOT/env" "虚拟环境"

# --- 3. 运行环境配置 ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$RES_ROOT/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 激活上一级目录中的虚拟环境 (物理复制过来的)
# 注意：如果是 venv 环境，复制后可能需要修复路径，但 source bin/activate 通常能处理
source "$RES_ROOT/env/bin/activate"

# --- 4. 8 卡 H200 生产启动 ---
echo "🔥 [GPU] 正在启动 AURORA 分布式训练 (8x H200 BF16)..."
LOG_NAME="train_$(date +%Y%m%d_%H%M).log"

setsid accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    aurora_train.py \
    --mode AURORA \
    --model_dir "$RES_ROOT/models" \
    --data_dir "$RES_ROOT/data" \
    --minilm_path "$RES_ROOT/models/minilm" \
    --output_dir "$RES_ROOT/output" \
    --batch_size 32 \
    --attack_weight 5.0 > "$LOG_NAME" 2>&1 < /dev/null &

echo "🚀 后台进程已启动：$!"
echo "📈 日志文件：tail -f $LOG_NAME"

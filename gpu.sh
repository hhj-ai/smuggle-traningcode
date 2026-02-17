#!/bin/bash
# --- 1. 路径定义 (指向已同步好的资源) ---
CODE_ROOT=$(cd "$(dirname "$0")"; pwd)
RES_ROOT=$(realpath "$CODE_ROOT/../aurora_resources")

# 核心资源路径
MODELS_DIR="$RES_ROOT/models"
DATA_DIR="$RES_ROOT/data"
OUTPUT_DIR="$RES_ROOT/output"
ENV_DIR="$RES_ROOT/env"

echo "📂 资源存储目录: $RES_ROOT"
echo "📍 模型路径: $MODELS_DIR"

# --- 2. 环境变量与离线配置 ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$RES_ROOT/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 解决无 root 无法写 ~/.EasyOCR 的问题
mkdir -p ~/.EasyOCR
if [ -d "$RES_ROOT/easyocr_cache" ]; then
    cp -rn "$RES_ROOT/easyocr_cache"/* ~/.EasyOCR/ 2>/dev/null
fi

# --- 3. 激活环境 ---
source "$ENV_DIR/bin/activate"

# --- 4. 8 卡 H200 极致启动 ---
echo "🔥 [GPU] 正在启动 AURORA 训练 (8x H200 BF16)..."
LOG_NAME="train_$(date +%Y%m%d_%H%M).log"

setsid accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    aurora_train.py \
    --mode AURORA \
    --model_dir "$MODELS_DIR" \
    --data_dir "$DATA_DIR" \
    --minilm_path "$MODELS_DIR/minilm" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --attack_weight 5.0 > "$LOG_NAME" 2>&1 < /dev/null &

echo "🚀 训练已在后台启动！"
echo "📈 查看日志: tail -f $LOG_NAME"

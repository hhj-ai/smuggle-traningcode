#!/bin/bash
# --- 1. 绝对路径定义 (Source & Destination) ---
CODE_ROOT=$(cd "$(dirname "$0")"; pwd)
RES_ROOT=$(realpath "$CODE_ROOT/../aurora_resources")

# 你之前提到那个确定好的环境绝对路径
PROD_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/aurora_env"

# 核心资源路径
MODELS_DIR="$RES_ROOT/models"
DATA_DIR="$RES_ROOT/data"
OUTPUT_DIR="$RES_ROOT/output"

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

# --- 3. 激活环境 (直接指向你那个确认好的绝对路径) ---
if [ -f "$PROD_ENV/bin/activate" ]; then
    echo "🐍 正在激活环境: $PROD_ENV"
    source "$PROD_ENV/bin/activate"
else
    echo "⚠️ 找不到生产环境路径，尝试读取本地备份环境..."
    source "$RES_ROOT/env/bin/activate"
fi

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

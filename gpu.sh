#!/bin/bash
# --- 1. 绝对路径定义 ---
CODE_ROOT=$(cd "$(dirname "$0")"; pwd)
RES_ROOT=$(realpath "$CODE_ROOT/../aurora_resources")
PROD_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/aurora_env"

# 核心资源路径
MODELS_DIR="$RES_ROOT/models"
DATA_DIR="$RES_ROOT/data"
OUTPUT_DIR="$RES_ROOT/output"

# --- 2. 鲁棒性资产检查 ---
echo "🔍 [Pre-flight] 正在检查核心模型资产..."
MISSING=0
for m in "Qwen3-VL-8B-Instruct" "DeepSeek-R1-Distill-Qwen-7B" "grounding-dino-base" "clip-vit-base-patch32" "minilm"; do
    if [ ! -d "$MODELS_DIR/$m" ]; then
        echo "❌ 缺失模型: $MODELS_DIR/$m"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "🚨 资产不全，请检查 ../aurora_resources/models 目录！"
    # 如果缺失，尝试寻找你的 SRC 原始备份（你之前说已经复制好了，如果没了这里可以加自动 link 逻辑）
    exit 1
fi

# --- 3. 进程暴力清理 ---
pkill -9 -f aurora_train.py 2>/dev/null
pkill -9 -f accelerate 2>/dev/null
sleep 2

# --- 4. 环境变量极致调优 ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$RES_ROOT/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=14400

source "$PROD_ENV/bin/activate"

# --- 5. 启动 ---
echo "🔥 [GPU] 启动极致稳定版训练 (资产检查已通过)..."
LOG_NAME="train_robust_$(date +%Y%m%d_%H%M).log"

setsid accelerate launch \
    --multi_gpu --num_processes 8 --mixed_precision bf16 \
    aurora_train.py \
    --model_dir "$MODELS_DIR" \
    --data_dir "$DATA_DIR" \
    --minilm_path "$MODELS_DIR/minilm" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --attack_weight 5.0 > "$LOG_NAME" 2>&1 < /dev/null &

echo "🚀 已启动！日志: tail -f $LOG_NAME"

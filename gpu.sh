#!/bin/bash
# --- 1. 绝对路径定义 ---
CODE_ROOT=$(cd "$(dirname "$0")"; pwd)
RES_ROOT=$(realpath "$CODE_ROOT/../aurora_resources")
PROD_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/aurora_env"

# 核心资源路径
MODELS_DIR="$RES_ROOT/models"
DATA_DIR="$RES_ROOT/data"
OUTPUT_DIR="$RES_ROOT/output"

# --- 2. 环境变量极致调优 ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$RES_ROOT/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 核心：限制多进程 CPU 抢占，防止 RAM 崩溃
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# 增加 NCCL 稳定性配置
export NCCL_TIMEOUT=7200
export NCCL_IB_DISABLE=0 # 如果集群支持 RDMA，请保持 0

# 激活环境
source "$PROD_ENV/bin/activate"

# --- 3. 启动 ---
echo "🔥 [GPU] 启动极致稳定版训练 (Timeout: 2h, Sequential: ON)..."
LOG_NAME="train_stable_$(date +%Y%m%d_%H%M).log"

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

echo "🚀 已启动稳定性模式！日志: tail -f $LOG_NAME"

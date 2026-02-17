#!/bin/bash
# --- 统一路径定义 ---
CODE_DIR=$(cd "$(dirname "$0")"; pwd)
RES_DIR=$(realpath "$CODE_DIR/../aurora_resources")
MODELS_DIR="$RES_DIR/models"
DATA_DIR="$RES_DIR/data"
OUTPUT_DIR="$RES_DIR/output"

# 生产环境原始环境路径 (你的保底路径)
PROD_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/aurora_env"

echo "📂 [GPU] 检查资源目录: $RES_DIR"

# 1. 资产存在性自检
MISSING=0
for m in "Qwen3-VL-8B-Instruct" "DeepSeek-R1-Distill-Qwen-7B" "grounding-dino-base" "clip-vit-base-patch32" "minilm"; do
    if [ ! -d "$MODELS_DIR/$m" ]; then
        echo "❌ 缺失模型: $MODELS_DIR/$m"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "🚨 资产缺失！请先在 CPU 服务器运行 cpu.sh 或检查挂载。"
    exit 1
fi

# 2. 进程清理
pkill -9 -f aurora_train.py 2>/dev/null
pkill -9 -f accelerate 2>/dev/null
sleep 2

# 3. 环境变量
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$RES_DIR/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

# 4. 激活环境 (优先使用生产路径)
if [ -f "$PROD_ENV/bin/activate" ]; then
    source "$PROD_ENV/bin/activate"
else
    source "$RES_DIR/env/bin/activate"
fi

# 5. 启动
echo "🔥 启动 AURORA 训练..."
LOG_NAME="train_final_$(date +%Y%m%d_%H%M).log"

setsid accelerate launch \
    --multi_gpu --num_processes 8 --mixed_precision bf16 \
    aurora_train.py \
    --model_dir "$MODELS_DIR" \
    --data_dir "$DATA_DIR/yfcc100m" \
    --minilm_path "$MODELS_DIR/minilm" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 > "$LOG_NAME" 2>&1 < /dev/null &

echo "🚀 已后台启动。日志: tail -f $LOG_NAME"

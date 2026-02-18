#!/bin/bash
# --- 统一路径定义 ---
CODE_DIR=$(cd "$(dirname "$0")"; pwd)
RES_DIR=$(realpath "$CODE_DIR/../aurora_resources")
MODELS_DIR="$RES_DIR/models"
DATA_DIR="$RES_DIR/data"
OUTPUT_DIR="$RES_DIR/output"

# 生产环境原始环境路径
PROD_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/smuggle-traningcode/aurora_env"

echo "📂 [GPU] 检查资源目录: $RES_DIR"

# 1. 资产存在性自检
MISSING=0
for m in "Qwen3-VL-8B-Instruct" "DeepSeek-R1-Distill-Qwen-7B" "grounding-dino-base" "clip-vit-base-patch32" "minilm"; do
    if [ ! -d "$MODELS_DIR/$m" ]; then
        echo "❌ 缺失模型目录: $MODELS_DIR/$m"
        MISSING=1
    elif [ ! -f "$MODELS_DIR/$m/config.json" ]; then
        echo "❌ 模型目录为空或不完整 (缺少 config.json): $MODELS_DIR/$m"
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
export OMP_NUM_THREADS=1

# ============================================================
# 4. 自动检测 GPU 显存，智能分配训练卡与工具卡
# ============================================================
TRAIN_THRESHOLD_MIB=80000   # 训练卡至少需要 80 GiB 空闲
TOOL_THRESHOLD_MIB=4000     # 工具卡至少需要 4 GiB 空闲 (DINO+CLIP ~1.5GB)

echo "🔍 自动检测 GPU 显存占用..."
TRAIN_GPUS=()
TOOL_GPU_CANDIDATES=()

for gpu_id in 0 1 2 3 4 5 6 7; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
    if [ -z "$free_mib" ]; then continue; fi

    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
    used_pct=$(( (total_mib - free_mib) * 100 / total_mib ))
    echo "  GPU $gpu_id: ${free_mib} MiB 空闲 / ${total_mib} MiB 总量 (已用 ${used_pct}%)"

    if [ "$free_mib" -ge "$TRAIN_THRESHOLD_MIB" ]; then
        TRAIN_GPUS+=("$gpu_id")
    elif [ "$free_mib" -ge "$TOOL_THRESHOLD_MIB" ]; then
        TOOL_GPU_CANDIDATES+=("$gpu_id")
    else
        echo "    ⚠️  GPU $gpu_id 空闲不足，跳过"
    fi
done

NUM_TRAIN=${#TRAIN_GPUS[@]}
if [ "$NUM_TRAIN" -eq 0 ]; then
    echo "🚨 没有 GPU 空闲显存 >= ${TRAIN_THRESHOLD_MIB} MiB，无法启动训练！"
    echo "   请检查是否有其他任务占用 GPU，或降低 TRAIN_THRESHOLD_MIB。"
    exit 1
fi

echo ""
echo "✅ 可用训练 GPU: [${TRAIN_GPUS[*]}] (共 ${NUM_TRAIN} 张)"

# 决定工具设备：优先用显存不够训练但够跑工具的卡
TOOL_DEVICE_ARG=""
if [ ${#TOOL_GPU_CANDIDATES[@]} -gt 0 ]; then
    TOOL_PHYS_GPU=${TOOL_GPU_CANDIDATES[0]}
    # CUDA_VISIBLE_DEVICES: 训练卡在前，工具卡在末尾
    ALL_GPUS=("${TRAIN_GPUS[@]}" "$TOOL_PHYS_GPU")
    CUDA_VIS=$(IFS=,; echo "${ALL_GPUS[*]}")
    # 工具卡的逻辑索引 = 训练卡数量 (0-indexed)
    TOOL_DEVICE_ARG="--tool_device cuda:${NUM_TRAIN}"
    echo "🔧 工具专用 GPU: 物理 GPU $TOOL_PHYS_GPU → 逻辑 cuda:${NUM_TRAIN}"
else
    # 没有专用工具卡，工具在 rank 0 的训练卡上加载
    CUDA_VIS=$(IFS=,; echo "${TRAIN_GPUS[*]}")
    echo "🔧 无专用工具卡，工具将加载在 rank 0 的训练 GPU 上"
fi

export CUDA_VISIBLE_DEVICES=$CUDA_VIS
echo "📋 CUDA_VISIBLE_DEVICES=$CUDA_VIS"
echo "📋 训练进程数: $NUM_TRAIN"
echo ""

# ============================================================
# 5. 激活环境与严格自检
# ============================================================
ACTIVATED=0
if [ -f "$PROD_ENV/bin/activate" ]; then
    echo "🐍 尝试激活生产环境: $PROD_ENV"
    source "$PROD_ENV/bin/activate"
    ACTIVATED=1
elif [ -f "$RES_DIR/env/bin/activate" ]; then
    echo "🐍 尝试激活本地环境: $RES_DIR/env"
    source "$RES_DIR/env/bin/activate"
    ACTIVATED=1
fi

if [ $ACTIVATED -eq 0 ]; then
    echo "🚨 致命错误：找不到可用的虚拟环境！请检查 PROD_ENV 或运行 cpu.sh 创建环境。"
    exit 1
fi

# 验证环境有效性
python -c "import torch; import transformers; print(f'✅ 环境验证通过: Torch {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "🚨 环境激活失败：当前 Python 无法加载 torch/transformers。"
    echo "   当前 Python: $(which python)"
    exit 1
fi

# 6. 断点续传检测
RESUME_ARG=""
LATEST_CKPT="$OUTPUT_DIR/checkpoints/latest"
if [ -L "$LATEST_CKPT" ] || [ -d "$LATEST_CKPT" ]; then
    if [ -f "$(realpath "$LATEST_CKPT")/training_state.pt" ]; then
        echo "🔄 检测到已有 checkpoint: $(realpath "$LATEST_CKPT")"
        echo "   将自动从上次中断处继续训练"
        RESUME_ARG="--resume_from latest"
    fi
fi

# 7. 启动
echo "🔥 启动 AURORA 训练..."
LOG_NAME="train_final_$(date +%Y%m%d_%H%M).log"

setsid accelerate launch \
    --multi_gpu --num_processes "$NUM_TRAIN" --mixed_precision bf16 \
    aurora_train.py \
    --model_dir "$MODELS_DIR" \
    --data_dir "$DATA_DIR/yfcc100m" \
    --minilm_path "$MODELS_DIR/minilm" \
    --output_dir "$OUTPUT_DIR" \
    $TOOL_DEVICE_ARG $RESUME_ARG > "$LOG_NAME" 2>&1 < /dev/null &

echo "🚀 已后台启动。日志: tail -n +1 -f $LOG_NAME"

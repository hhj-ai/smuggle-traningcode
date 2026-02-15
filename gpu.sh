#!/bin/bash
# =========================================================
# 2_fixed_install.sh (GPU 服务器 - 修复顺序版)
# 特性: 
# 1. 严格按顺序安装 Packaging -> Numpy -> Torch -> FlashAttn
# 2. 不会覆盖/修改当前目录下的 models.py
# =========================================================

BASE_DIR="./offline_packages"
PYTHON_TGZ="$BASE_DIR/python_runtime/python-3.10.tar.gz"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

echo "🚀 [GPU Server] 开始严格顺序安装..."

# 1. 准备环境 (若已存在则跳过解压，节省时间)
if [ ! -d "$INSTALL_ROOT" ]; then
    echo "🐍 解压 Python 3.10..."
    mkdir -p $INSTALL_ROOT
    tar -xzf $PYTHON_TGZ -C $INSTALL_ROOT
fi

# 确定 Python 路径
if [ -d "$INSTALL_ROOT/python" ]; then 
    LOCAL_PYTHON="$INSTALL_ROOT/python/bin/python3"
else 
    LOCAL_PYTHON="$INSTALL_ROOT/bin/python3"
fi

# 重建虚拟环境
echo "📦 重建虚拟环境..."
rm -rf $VENV_DIR
$LOCAL_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 配置 pip 强制离线
pip config set global.no-index true > /dev/null 2>&1
pip config set global.find-links $(pwd)/$WHEEL_DIR > /dev/null 2>&1

install_pkg() {
    # 强制只从本地找
    pip install "$@" --no-index --find-links=$WHEEL_DIR
}

# =========================================================
# 2. 关键修复步骤：按顺序安装
# =========================================================

echo "🧱 [1/5] 安装构建工具 (Packaging, Ninja, Numpy)..."
# 必须最先安装，否则 FlashAttn 编译会报错 "No module named packaging"
install_pkg wheel setuptools
install_pkg packaging ninja psutil
install_pkg "numpy<2.0.0"

# 验证关键包
python -c "import packaging; import numpy; print(f'   ✅ Environment Ready: Numpy {numpy.__version__}')" || exit 1

echo "🎮 [2/5] 安装 NVIDIA 依赖库..."
# PyTorch 2.x 强依赖这些库，必须手动安装
install_pkg nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12
install_pkg nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cufft-cu12 
install_pkg nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 
install_pkg nvidia-nccl-cu12 nvidia-nvtx-cu12 triton

echo "🔥 [3/5] 安装 PyTorch..."
install_pkg torch torchvision torchaudio
# 验证 Torch
python -c "import torch; print(f'   ✅ Torch {torch.__version__} (CUDA Available: {torch.cuda.is_available()})')" || exit 1

echo "⚡ [4/5] 编译 Flash Attention..."
# 此时环境里已经有了 torch, packaging, ninja，编译应该能通过
install_pkg flash-attn --no-build-isolation

echo "🤗 [5/5] 安装 Transformers & 其他..."
if [ -f "$WHEEL_DIR/transformers-main.zip" ]; then
    echo "   -> 从源码 Zip 安装 Transformers..."
    unzip -q -o "$WHEEL_DIR/transformers-main.zip" -d ./temp_tf
    pip install ./temp_tf/transformers-main --no-index --find-links=$WHEEL_DIR
    rm -rf ./temp_tf
else
    install_pkg transformers
fi

# 安装剩余依赖
install_pkg accelerate huggingface_hub datasets sentence-transformers Pillow easyocr
install_pkg scipy termcolor timm rich questionary aiohttp protobuf sentencepiece pandas

echo "------------------------------------------------"
echo "🎉 安装完成！"
echo "👉 启动命令: source $VENV_DIR/bin/activate"

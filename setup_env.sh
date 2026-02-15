#!/bin/bash

# ==========================================
# AURORA Environment Setup (H200 Optimized)
# Creates Conda env & Installs Dependencies
# ==========================================

ENV_NAME="aurora_env"

echo "🚀 Setting up Conda Environment: $ENV_NAME"

# 1. 检查 Conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda could not be found. Please install Anaconda/Miniconda first."
    exit 1
fi

# 2. 创建环境 (Python 3.10 是目前兼容性最好的版本)
echo "📦 Creating environment..."
conda create -n $ENV_NAME python=3.10 -y

# 3. 激活环境
# 注意：在 shell 脚本中激活 conda 需要 source
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "✅ Environment activated: $(which python)"

# 4. 安装 PyTorch (CUDA 12.1 for H200)
echo "🔥 Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 安装构建工具 (FlashAttn 需要)
echo "🔧 Installing build tools..."
pip install packaging ninja

# 6. 安装 Flash Attention 2 (H200 核心加速库)
echo "⚡ Installing Flash Attention 2 (This may take a while to compile)..."
pip install flash-attn --no-build-isolation

# 7. 安装其他核心依赖
echo "📚 Installing dependencies..."
pip install \
    transformers>=4.38.0 \
    accelerate>=0.27.0 \
    datasets \
    huggingface_hub \
    sentence-transformers \
    numpy \
    Pillow \
    easyocr \
    scipy \
    termcolor \
    timm \
    rich \
    questionary \
    aiohttp \
    requests \
    protobuf \
    sentencepiece

# 8. 安装 GroundingDINO (如果不方便编译，先跳过，用 easyocr 和 clip 顶替)
# 如果 tools.py 强依赖 GroundingDINO，取消下面注释：
# echo "🦖 Installing GroundingDINO..."
# pip install git+https://github.com/IDEA-Research/GroundingDINO.git

echo "------------------------------------------------"
echo "🎉 Environment Setup Complete!"
echo "👉 To start using, run: conda activate $ENV_NAME"

#!/bin/bash

# ==========================================
# AURORA Environment Setup (Singapore Official)
# Strategy: Use Official PyPI (No Mirrors)
# ==========================================

ENV_NAME="aurora_env"

echo "🚀 Starting Environment Setup (Official PyPI)..."

# 1. 清理并创建环境
rm -rf $ENV_NAME
echo "📦 Creating virtual environment..."
python3 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

# 2. 定义官方源安装函数 (强制不使用任何 config 中的镜像)
run_pip_official() {
    python -m pip install "$@" --index-url https://pypi.org/simple/ --no-cache-dir
}

# 3. 升级基础工具
echo "🔧 Upgrading pip..."
run_pip_official --upgrade pip wheel setuptools

# 4. 安装 PyTorch (官方源)
echo "🔥 Installing PyTorch (Official)..."
# 新加坡连官方 pytorch 很快
run_pip_official torch torchvision torchaudio

# 5. 安装构建依赖
echo "🧱 Installing build dependencies..."
run_pip_official psutil ninja packaging

# 6. 安装 Flash Attention 2
echo "⚡ Installing Flash Attention 2..."
run_pip_official flash-attn --no-build-isolation

# 7. 安装 Transformers (直接从 GitHub 安装最新开发版，确保支持 Qwen3)
echo "🤗 Installing Transformers (Main Branch for Qwen3 support)..."
run_pip_official git+https://github.com/huggingface/transformers.git

# 8. 安装其他依赖
echo "📚 Installing remaining dependencies..."
run_pip_official \
    "accelerate>=0.27.0" \
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

echo "------------------------------------------------"
echo "🎉 Setup Complete (Official Source)!"
echo "👉 Run: source $ENV_NAME/bin/activate"

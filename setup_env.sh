#!/bin/bash

# ==========================================
# AURORA Environment Setup (Enterprise Fixed)
# Strategy: Use venv + Trust Internal Mirror
# ==========================================

ENV_NAME="aurora_env"
# 自动获取当前内网源地址（从报错日志里提取的）
PIP_INDEX_URL="http://pip.sankuai.com/simple/"
PIP_TRUSTED_HOST="pip.sankuai.com"

echo "🚀 Starting Robust Environment Setup..."

# 1. 清理旧环境 (如果有残留)
rm -rf $ENV_NAME

# 2. 创建虚拟环境 (使用 venv 代替 conda)
echo "📦 Creating virtual environment using 'venv'..."
# 尝试使用 python3 或 python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

$PYTHON_CMD -m venv $ENV_NAME

if [ ! -d "$ENV_NAME" ]; then
    echo "❌ Failed to create venv. Please check your python installation."
    exit 1
fi

# 3. 激活环境
echo "🔌 Activating environment..."
source $ENV_NAME/bin/activate

# 确认激活成功
WHICH_PYTHON=$(which python)
echo "   -> Python path: $WHICH_PYTHON"
if [[ "$WHICH_PYTHON" != *"$ENV_NAME"* ]]; then
    echo "❌ Activation failed!"
    exit 1
fi

# 定义带信任参数的 pip 函数
run_pip() {
    python -m pip install "$@" --index-url $PIP_INDEX_URL --trusted-host $PIP_TRUSTED_HOST
}

# 4. 升级 pip 和基础工具
echo "🔧 Upgrading pip and build tools..."
run_pip --upgrade pip wheel setuptools

# 5. 手动安装构建依赖 (解决 flash-attn 编译报错的关键)
echo "🧱 Installing build dependencies (psutil, ninja)..."
run_pip psutil ninja packaging

# 6. 安装 PyTorch (指定版本)
echo "🔥 Installing PyTorch..."
# 内网源通常会自动匹配合适的 CUDA 版本，如果不行再手动指定
run_pip torch torchvision torchaudio

# 7. 安装 Flash Attention 2 (关键步骤)
echo "⚡ Installing Flash Attention 2..."
# 使用 --no-build-isolation 强制使用我们刚才手动安装的 psutil/ninja
run_pip flash-attn --no-build-isolation

# 8. 安装其他依赖
echo "📚 Installing remaining dependencies..."
run_pip \
    "transformers>=4.38.0" \
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
echo "🎉 Environment Setup Complete!"
echo "👉 To activate, run: source $ENV_NAME/bin/activate"

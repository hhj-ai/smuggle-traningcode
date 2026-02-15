#!/bin/bash

# ==========================================
# AURORA Environment Setup (Step-by-Step)
# 解决卡在 "Installing collected packages" 的问题
# ==========================================

ENV_NAME="aurora_env"
# 内网源配置
PIP_INDEX_URL="http://pip.sankuai.com/simple/"
PIP_TRUSTED_HOST="pip.sankuai.com"

echo "🚀 启动分步安装脚本..."

# 1. 检查或创建环境
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv $ENV_NAME
fi

# 2. 激活环境
source $ENV_NAME/bin/activate
echo "🔌 环境已激活: $(which python)"

# 定义 PIP 函数 (带信任 + 无缓存 + 详细输出)
# -v: 显示详细进度，防止看着像卡死
# --no-cache-dir: 节省空间，减少解压时的 IO
run_pip() {
    python -m pip install "$@" \
        --index-url $PIP_INDEX_URL \
        --trusted-host $PIP_TRUSTED_HOST \
        --no-cache-dir \
        -v
}

# 定义简易 PIP (不带 -v，用于小包)
run_pip_quiet() {
    python -m pip install "$@" \
        --index-url $PIP_INDEX_URL \
        --trusted-host $PIP_TRUSTED_HOST \
        --no-cache-dir
}

# 3. 升级基础工具
echo "🔧 [1/6] 升级 pip..."
run_pip_quiet --upgrade pip wheel setuptools

# 4. 单独安装 PyTorch (最大的包，最容易卡)
echo "🔥 [2/6] 单独安装 PyTorch (由大到小)..."
echo "    注意：屏幕会疯狂滚动日志，这是正常的，说明在解压..."
# 先装 torch
run_pip torch

# 5. 安装 Vision 和 Audio
echo "📷 [3/6] 安装 TorchVision & TorchAudio..."
run_pip torchvision torchaudio

# 6. 安装构建依赖
echo "🧱 [4/6] 安装构建工具 (ninja, psutil)..."
run_pip_quiet psutil ninja packaging

# 7. 安装 Flash Attention
echo "⚡ [5/6] 安装 Flash Attention 2..."
# 这一步需要编译，可能会慢，保持耐心
run_pip flash-attn --no-build-isolation

# 8. 安装其余依赖
echo "📚 [6/6] 安装剩余依赖..."
run_pip_quiet \
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
echo "🎉 安装全部完成！没有卡死！"
echo "👉 请运行: source $ENV_NAME/bin/activate"

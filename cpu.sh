#!/bin/bash

# ========================================================================
# 1_full_download.sh (CPU 服务器 - 最终修复版)
# 目标: 暴力下载 Python 3.10 + 依赖 (绕过 pip 版本检查)
# ========================================================================

SAVE_DIR="./offline_packages"
PYTHON_DIR="$SAVE_DIR/python_runtime"
WHEEL_DIR="$SAVE_DIR/wheels"

mkdir -p $PYTHON_DIR
mkdir -p $WHEEL_DIR

echo "🚀 [CPU Server] 开始构建全量离线包 (暴力直链版)..."

# ------------------------------------------------------------------------
# 1. 下载独立版 Python 3.10 (Standalone Build)
# ------------------------------------------------------------------------
echo "🐍 [1/5] 下载 Python 3.10 独立运行包..."
# Indygreg 提供的独立 Python 包，解压即用，不依赖系统环境
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"

if [ ! -f "$PYTHON_DIR/python-3.10.tar.gz" ]; then
    wget -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
    # 如果 wget 失败尝试 curl
    if [ $? -ne 0 ]; then
        echo "⚠️ wget 失败，尝试 curl..."
        curl -L -o "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
    fi
else
    echo "   ✅ Python 包已存在。"
fi

# ------------------------------------------------------------------------
# 2. 暴力下载 PyTorch (直接 Wget URL，不再让 pip 猜)
# ------------------------------------------------------------------------
echo "🔥 [2/5] 暴力下载 PyTorch (CUDA 12.1)..."
# 这里的 URL 是 PyTorch 官方仓库中对应 Python 3.10 + CUDA 12.1 的真实地址
# %2B 是 URL 编码的 + 号
TORCH_URL="https://download.pytorch.org/whl/cu121/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
VISION_URL="https://download.pytorch.org/whl/cu121/torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
AUDIO_URL="https://download.pytorch.org/whl/cu121/torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

# 使用 wget 下载 (-nc 表示如果文件存在就不重新下载)
wget -nc -P $WHEEL_DIR $TORCH_URL
wget -nc -P $WHEEL_DIR $VISION_URL
wget -nc -P $WHEEL_DIR $AUDIO_URL

# 如果服务器没装 wget，用 curl 替补
if [ ! -f "$WHEEL_DIR/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl" ]; then
    echo "⚠️ wget 不可用或下载失败，尝试 curl..."
    curl -L -o "$WHEEL_DIR/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl" $TORCH_URL
    curl -L -o "$WHEEL_DIR/torchvision-0.19.1+cu121-cp310-cp310-linux_x86_64.whl" $VISION_URL
    curl -L -o "$WHEEL_DIR/torchaudio-2.4.1+cu121-cp310-cp310-linux_x86_64.whl" $AUDIO_URL
fi

# ------------------------------------------------------------------------
# 3. 下载 Transformers (Wget 源码 Zip，最稳妥)
# ------------------------------------------------------------------------
echo "🤗 [3/5] 下载 Transformers (GitHub Main)..."
TRANSFORMERS_URL="https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

wget -nc -O "$WHEEL_DIR/transformers-main.zip" "$TRANSFORMERS_URL"
if [ ! -f "$WHEEL_DIR/transformers-main.zip" ]; then
    curl -L -o "$WHEEL_DIR/transformers-main.zip" "$TRANSFORMERS_URL"
fi

# ------------------------------------------------------------------------
# 4. 下载 Flash Attention 2 (通用源码包)
# ------------------------------------------------------------------------
echo "⚡ [4/5] 下载 Flash Attention 2..."
# Flash Attention 源码包不区分 Python 版本，可以用 pip download
pip download flash-attn==2.6.3 \
    --dest $WHEEL_DIR \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --no-binary :all: \
    --no-deps

# ------------------------------------------------------------------------
# 5. 下载其他通用依赖 (使用 pip download，但放宽限制)
# ------------------------------------------------------------------------
echo "📚 [5/5] 下载通用依赖 (伪装 Py3.10)..."

# 定义下载函数：指定 Py3.10 和官方源
download_wheel() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps
}

# 核心依赖 (手动列出，防止自动解析出错)
download_wheel accelerate>=0.27.0
download_wheel huggingface-hub>=0.23.0
download_wheel tokenizers>=0.19.1
download_wheel safetensors>=0.4.1
download_wheel regex
download_wheel requests
download_wheel filelock
download_wheel fsspec
download_wheel pyyaml
download_wheel tqdm
download_wheel packaging
download_wheel ninja
download_wheel psutil
download_wheel numpy<2.0.0
download_wheel Pillow
download_wheel easyocr
download_wheel scipy
download_wheel termcolor
download_wheel timm
download_wheel rich
download_wheel questionary
download_wheel aiohttp
download_wheel protobuf
download_wheel sentencepiece
download_wheel setuptools
download_wheel wheel
download_wheel typing-extensions
download_wheel sympy
download_wheel networkx
download_wheel jinja2
download_wheel MarkupSafe
download_wheel charset-normalizer
download_wheel idna
download_wheel urllib3
download_wheel certifi

echo "------------------------------------------------"
echo "✅ 暴力下载完成！"
echo "📂 请检查 $WHEEL_DIR 下是否有 .whl 文件"

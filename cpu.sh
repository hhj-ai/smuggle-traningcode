#!/bin/bash

# ==========================================
# 步骤 1: 离线包下载脚本 (运行在有网的 CPU 服务器)
# 目标: 将所有依赖下载到共享磁盘目录 ./offline_packages
# 修复: 强制指定官方源，无视环境默认的内网源
# ==========================================

SAVE_DIR="./offline_packages"
mkdir -p $SAVE_DIR

echo "🚀 [CPU Server] 开始下载依赖包到 $SAVE_DIR ..."
echo "🌍 强制使用官方源: https://pypi.org/simple"

# 定义下载函数 (强制指定 index-url 和 trusted-host)
download_pkg() {
    pip download "$@" \
        --dest $SAVE_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org
}

# 1. 下载 PyTorch (CUDA 12.1 版本)
# 注意: PyTorch 必须走官方 PyTorch 源，不能走 PyPI
echo "⬇️  下载 PyTorch (CUDA 12.1)..."
pip download \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --dest $SAVE_DIR \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 下载 Transformers (直接从 GitHub 下载最新源码包)
# 这一步会自动下载 transformers 的依赖 (huggingface-hub 等)
# 我们强制让它去官方 PyPI 找依赖，而不是去内网源
echo "⬇️  下载 Transformers (及依赖)..."
# 注意: git+https 下载时，依赖解析也会走 pip 配置，所以要指定 index-url
pip download git+https://github.com/huggingface/transformers.git \
    --dest $SAVE_DIR \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# 3. 下载 Flash Attention 2 (源码包)
echo "⬇️  下载 Flash Attention 2 (源码)..."
download_pkg flash-attn --no-binary :all:

# 4. 下载其他所有依赖
echo "⬇️  下载其他通用依赖..."
download_pkg \
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
    sentencepiece \
    ninja \
    packaging \
    psutil \
    wheel \
    setuptools

echo "------------------------------------------------"
echo "✅ 下载完成！"
echo "📂 所有包已保存在: $SAVE_DIR"

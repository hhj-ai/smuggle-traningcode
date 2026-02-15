#!/bin/bash

# ==========================================
# 步骤 1: 离线包下载脚本 (运行在有网的 CPU 服务器)
# 目标: 将所有依赖下载到共享磁盘目录 ./offline_packages
# ==========================================

SAVE_DIR="./offline_packages"
mkdir -p $SAVE_DIR

echo "🚀 [CPU Server] 开始下载依赖包到 $SAVE_DIR ..."
echo "⚠️  注意: 请确保 CPU 服务器的 Python 版本与 GPU 服务器一致 (推荐 3.10)"

# 1. 下载 PyTorch (CUDA 12.1 版本)
# 指定 --platform 和 --python-version 以防 CPU/GPU 服务器环境差异过大
# 这里默认下载 Linux x86_64, Python 3.10 的包
echo "⬇️  下载 PyTorch (CUDA 12.1)..."
pip download \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --dest $SAVE_DIR

# 2. 下载 Transformers (直接从 GitHub 下载最新源码包)
# 这是为了解决 Qwen3-VL 兼容性问题
echo "⬇️  下载 Transformers (GitHub Main Branch)..."
pip download git+https://github.com/huggingface/transformers.git --dest $SAVE_DIR

# 3. 下载 Flash Attention 2
# 注意: 我们下载源码包 (--no-binary)，让 GPU 服务器自己根据显卡架构编译
echo "⬇️  下载 Flash Attention 2 (源码)..."
pip download flash-attn --no-binary :all: --dest $SAVE_DIR

# 4. 下载其他所有依赖
echo "⬇️  下载其他通用依赖..."
pip download \
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
    setuptools \
    --dest $SAVE_DIR

echo "------------------------------------------------"
echo "✅ 下载完成！"
echo "📂 所有包已保存在: $SAVE_DIR"
echo "👉 请切换到 GPU 服务器运行 2_gpu_install.sh"

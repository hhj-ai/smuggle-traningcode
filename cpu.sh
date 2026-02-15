#!/bin/bash

# ========================================================================
# 1_full_download.sh (CPU 服务器)
# 目标: 下载 Python 3.10 独立包 + 所有依赖 Wheel
# ========================================================================

# 设置下载目录
SAVE_DIR="./offline_packages"
PYTHON_DIR="$SAVE_DIR/python_runtime"
WHEEL_DIR="$SAVE_DIR/wheels"

mkdir -p $PYTHON_DIR
mkdir -p $WHEEL_DIR

echo "🚀 [CPU Server] 开始构建全量离线包..."
echo "📂 保存路径: $SAVE_DIR"

# ------------------------------------------------------------------------
# 1. 下载独立版 Python 3.10 (Standalone Build)
# ------------------------------------------------------------------------
echo "🐍 [1/5] 下载 Python 3.10 独立运行包..."
# 使用 indygreg 的 python-build-standalone，这是目前最流行的便携 Python 构建
# 下载 Linux x86_64 版本
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"

if [ ! -f "$PYTHON_DIR/python-3.10.tar.gz" ]; then
    wget -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
    if [ $? -ne 0 ]; then
        echo "⚠️ wget 失败，尝试 curl..."
        curl -L -o "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
    fi
else
    echo "   ✅ Python 包已存在，跳过下载。"
fi

# ------------------------------------------------------------------------
# 2. 定义伪装下载函数 (模拟 Python 3.10 环境)
# ------------------------------------------------------------------------
download_wheel() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps  # 显式控制依赖，防止拉取到不兼容的包
}

echo "📦 [2/5] 开始交叉下载依赖包 (Target: Py3.10, Linux, CUDA 12)..."

# ------------------------------------------------------------------------
# 3. 下载 PyTorch (CUDA 12.1) - 必须指定官方源
# ------------------------------------------------------------------------
echo "   ⬇️  PyTorch Core..."
pip download \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --dest $WHEEL_DIR \
    --index-url https://download.pytorch.org/whl/cu121 \
    --python-version 3.10 \
    --platform manylinux2014_x86_64 \
    --only-binary=:all:

# ------------------------------------------------------------------------
# 4. 下载 Transformers & Huggingface 生态
# ------------------------------------------------------------------------
echo "   ⬇️  Transformers & Eco..."
# 注意: 我们手动列出 transformers 的关键依赖，确保版本匹配
download_wheel transformers>=4.45.0  # 指定高版本以支持 Qwen2-VL
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

# ------------------------------------------------------------------------
# 5. 下载 Flash Attention 2 (必须源码)
# ------------------------------------------------------------------------
echo "   ⬇️  Flash Attention (Source)..."
# 源码包不区分 Python 版本
pip download flash-attn==2.6.3 \
    --dest $WHEEL_DIR \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --no-binary :all:

# ------------------------------------------------------------------------
# 6. 下载其他业务依赖
# ------------------------------------------------------------------------
echo "   ⬇️  General Utils..."
download_wheel datasets
download_wheel sentence-transformers
download_wheel numpy<2.0.0  # 防止 numpy 2.0 兼容性问题
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
download_wheel ninja
download_wheel psutil
download_wheel setuptools
download_wheel wheel

echo "------------------------------------------------"
echo "✅ 全量包构建完成！"
echo "📂 目录结构:"
echo "   $SAVE_DIR/"
echo "   ├── python_runtime/ (含 Python 3.10)"
echo "   └── wheels/ (含所有 .whl)"

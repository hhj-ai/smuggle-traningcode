#!/bin/bash

# ==============================================================================
# AURORA Offline Resource Downloader (Open Source Version)
# ------------------------------------------------------------------------------
# 目标：构建一个完全自洽的离线安装包
# 环境：Python 3.10 | CUDA 12.1 | Linux x86_64
# ==============================================================================

SAVE_DIR="./offline_packages"
WHEEL_DIR="$SAVE_DIR/wheels"
PYTHON_DIR="$SAVE_DIR/python_runtime"

mkdir -p $WHEEL_DIR $PYTHON_DIR

echo "🚀 [CPU Server] 开始全量资源采集..."

# --- 1. 下载独立 Python 运行环境 ---
echo "🐍 [1/4] 下载可移植 Python 3.10..."
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"
wget -nc -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"

# --- 2. 暴力下载 PyTorch 核心 ---
echo "🔥 [2/4] 下载 PyTorch (CUDA 12.1)..."
BASE_URL="https://download.pytorch.org/whl/cu121"
# 强制指定版本与架构，绕过本地 Python 限制
for pkg in "torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"; do
    wget -nc -P $WHEEL_DIR "$BASE_URL/$pkg"
done

# --- 3. 下载 Transformers 源码 (支持 Qwen3) ---
echo "🤗 [3/4] 下载 Transformers 最新开发版..."
wget -nc -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

# --- 4. 深度补全所有依赖包 ---
echo "📚 [4/4] 深度采集所有依赖 (含次级依赖)..."

download_all() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps --quiet
}

# [清单 A] 编译与底层库
download_all numpy==1.26.4 packaging ninja psutil setuptools wheel einops flash-attn==2.6.3 --no-binary flash-attn

# [清单 B] NVIDIA 核心运行时
download_all nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1 \
             nvidia-cudnn-cu12==9.1.0.70 nvidia-nvjitlink-cu12==12.1.105 \
             nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 \
             nvidia-nccl-cu12==2.20.5 triton==3.0.0 nvidia-nvtx-cu12==12.1.105 \
             nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-cupti-cu12==12.1.105 \
             nvidia-cufft-cu12==11.0.2.54 nvidia-cusparse-cu12==12.1.0.106

# [清单 C] 网络与异步 (修复 aiohappyeyeballs 缺失)
download_all aiohttp aiohappyeyeballs yarl multidict frozenlist aiosignal attrs \
             requests urllib3 idna certifi charset-normalizer

# [清单 D] 模型加载与 UI
download_all accelerate huggingface-hub tokenizers safetensors pyyaml tqdm \
             rich pygments markdown-it-py mdurl shellingham click typer typer-slim \
             colorama filelock fsspec typing-extensions

# [清单 E] 科学计算与数据集
download_all datasets pandas scipy pillow timm sentence-transformers \
             easyocr scikit-image python-bidi protobuf sentencepiece \
             dill multiprocess pyarrow regex sympy networkx jinja2 MarkupSafe mpmath

echo "------------------------------------------------"
echo "✅ 采集完成！请将 $SAVE_DIR 拷贝至 GPU 服务器。"

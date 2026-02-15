#!/bin/bash

# ==============================================================================
# AURORA Offline Resource Downloader (v3.0 - Zero Compilation Edition)
# ------------------------------------------------------------------------------
# 修复：
# 1. 暴力下载 Flash-Attn 预编译二进制包 (cu121 + torch2.4 + cp310)
# 2. 补齐 httpx, hf-xet, anyio, httpcore 等新版依赖
# ==============================================================================

SAVE_DIR="./offline_packages"
WHEEL_DIR="$SAVE_DIR/wheels"
PYTHON_DIR="$SAVE_DIR/python_runtime"

mkdir -p $WHEEL_DIR $PYTHON_DIR

echo "🚀 [Builder] 正在进行全量资源暴力采集 (零编译策略)..."

# --- 1. 下载 Python 3.10 Runtime ---
echo "🐍 [1/4] 下载 Python 3.10..."
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"
wget -nc -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"

# --- 2. 暴力下载核心框架 (二进制 Wheel) ---
echo "🔥 [2/4] 下载核心框架二进制包..."

# PyTorch 2.4.1 (CUDA 12.1)
BASE_URL="https://download.pytorch.org/whl/cu121"
for pkg in "torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"; do
    wget -nc -P $WHEEL_DIR "$BASE_URL/$pkg"
done

# 【核心修复】Flash Attention 2.6.3 预编译二进制 (cu121 + torch2.4 + cp310)
# 这种方式下载的文件在 GPU 服务器上直接安装，无需 clang++/g++ 编译
echo "⚡ 下载 Flash Attention 预编译 Wheel..."
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -nc -P $WHEEL_DIR "$FLASH_URL"

# Transformers 源码
wget -nc -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

# --- 3. 深度补全依赖 ---
echo "📚 [3/4] 深度采集所有缺失依赖..."

download_dep() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps --quiet
}

# 3.1 补全报错的 httpx 和 hf-xet 系列
echo "   -> 补全 HF 生态包..."
download_dep httpx httpcore anyio sniffio h11 hf-xet

# 3.2 NVIDIA 全家桶 (含最新的 nvjitlink)
echo "   -> 补全 NVIDIA 运行时..."
download_dep nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1 \
             nvidia-cudnn-cu12==9.1.0.70 nvidia-nvjitlink-cu12==12.1.105 \
             nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 \
             nvidia-nccl-cu12==2.20.5 triton==3.0.0 nvidia-nvtx-cu12==12.1.105 \
             nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-cupti-cu12==12.1.105 \
             nvidia-cufft-cu12==11.0.2.54 nvidia-cusparse-cu12==12.1.0.106

# 3.3 基础工具与 UI
echo "   -> 补全基础工具..."
download_dep numpy==1.26.4 packaging ninja psutil setuptools wheel einops
download_dep accelerate huggingface-hub tokenizers safetensors pyyaml tqdm \
             rich pygments markdown-it-py mdurl shellingham click typer typer-slim \
             colorama filelock fsspec typing-extensions
download_dep aiohttp aiohappyeyeballs yarl multidict frozenlist aiosignal attrs \
             requests urllib3 idna certifi charset-normalizer

# 3.4 业务依赖
download_dep datasets pandas scipy pillow timm sentence-transformers \
             easyocr scikit-image python-bidi protobuf sentencepiece \
             dill multiprocess pyarrow regex sympy networkx jinja2 MarkupSafe mpmath

# ==============================================================================
# 4. 自检
# ==============================================================================
echo "------------------------------------------------"
REQUIRED=("torch" "flash_attn" "httpx" "hf-xet" "nvidia_nvjitlink")
for pkg in "${REQUIRED[@]}"; do
    if [ $(find $WHEEL_DIR -iname "*$pkg*" | wc -l) -eq 0 ]; then
        echo "❌ 关键包缺失: $pkg"
        exit 1
    fi
done
echo "🎉 所有资源采集成功！请拷贝 $SAVE_DIR 到 GPU 服务器。"

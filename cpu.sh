#!/bin/bash

# ==============================================================================
# AURORA Offline Resource Downloader (v2.0 - Stability Focus)
# ------------------------------------------------------------------------------
# 修复：flash-attn 改为 wget 直链下载，解决 pip 找不到版本的问题。
# ==============================================================================

SAVE_DIR="./offline_packages"
WHEEL_DIR="$SAVE_DIR/wheels"
PYTHON_DIR="$SAVE_DIR/python_runtime"

mkdir -p $WHEEL_DIR $PYTHON_DIR

echo "🚀 [CPU Server] 开始全量资源采集..."

# --- 1. 下载可移植 Python 3.10 ---
echo "🐍 [1/4] 下载 Python 3.10 Runtime..."
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"
wget -nc -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"

# --- 2. 暴力下载 PyTorch & Flash Attention (直链绕过 pip) ---
echo "🔥 [2/4] 暴力下载核心框架 (Direct Download)..."

# PyTorch 2.4.1 (CUDA 12.1)
BASE_URL="https://download.pytorch.org/whl/cu121"
for pkg in "torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"; do
    wget -nc -P $WHEEL_DIR "$BASE_URL/$pkg"
done

# Flash Attention 2.6.3 (直接从 PyPI 下载源码包，解决版本找不到问题)
echo "⚡ 下载 Flash Attention 源码包..."
FLASH_ATTN_URL="https://files.pythonhosted.org/packages/11/40/664f3d17961239c049618f25f209633e65992b95c378e9076043431665a3/flash_attn-2.6.3.tar.gz"
wget -nc -O "$WHEEL_DIR/flash_attn-2.6.3.tar.gz" "$FLASH_ATTN_URL"

# Transformers 源码
wget -nc -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

# --- 3. 下载依赖包 (No-Deps 模式) ---
echo "📚 [3/4] 下载所有 Wheel 依赖..."

download_all() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps --quiet
}

# 按照报错情况补全的依赖清单
download_all nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1 \
             nvidia-cudnn-cu12==9.1.0.70 nvidia-nvjitlink-cu12==12.1.105 \
             nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 \
             nvidia-nccl-cu12==2.20.5 triton==3.0.0 nvidia-nvtx-cu12==12.1.105 \
             nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-cupti-cu12==12.1.105 \
             nvidia-cufft-cu12==11.0.2.54 nvidia-cusparse-cu12==12.1.0.106

download_all numpy==1.26.4 packaging ninja psutil setuptools wheel einops
download_all aiohttp aiohappyeyeballs yarl multidict frozenlist aiosignal attrs \
             requests urllib3 idna certifi charset-normalizer
download_all accelerate huggingface-hub tokenizers safetensors pyyaml tqdm \
             rich pygments markdown-it-py mdurl shellingham click typer typer-slim \
             colorama filelock fsspec typing-extensions
download_all datasets pandas scipy pillow timm sentence-transformers \
             easyocr scikit-image python-bidi protobuf sentencepiece \
             dill multiprocess pyarrow regex sympy networkx jinja2 MarkupSafe mpmath

# ==============================================================================
# 4. 严厉的最终检测 (Strict Verification)
# ==============================================================================
echo "------------------------------------------------"
echo "🕵️  执行最终完整性自检..."

MISSING_LOG="./missing_packages.log"
> $MISSING_LOG
MISSING_COUNT=0

# 定义开源必须的基础物资
REQUIRED_SAMPLES=("torch" "transformers" "flash_attn" "nvidia_cublas" "aiohappyeyeballs" "rich" "numpy")

for pkg in "${REQUIRED_SAMPLES[@]}"; do
    if [ $(find $WHEEL_DIR -iname "*$pkg*" | wc -l) -eq 0 ]; then
        echo "❌ [Critical Missing]: $pkg"
        echo "$pkg" >> $MISSING_LOG
        MISSING_COUNT=$((MISSING_COUNT + 1))
    else
        echo "✅ [Verified]: $pkg"
    fi
done

if [ $MISSING_COUNT -gt 0 ]; then
    echo "------------------------------------------------"
    echo "⛔ 自检失败！共缺失 $MISSING_COUNT 个核心组件。"
    echo "👉 请查看清单并重新运行下载。"
    exit 1
else
    echo "------------------------------------------------"
    echo "🎉 所有资源采集成功！你可以将 $SAVE_DIR 拷贝至 GPU 服务器。"
fi

#!/bin/bash
# ==============================================================================
# AURORA Offline Downloader (Final Fix)
# ==============================================================================

SAVE_DIR="./offline_packages"
WHEEL_DIR="$SAVE_DIR/wheels"
PYTHON_DIR="$SAVE_DIR/python_runtime"
mkdir -p $WHEEL_DIR $PYTHON_DIR

echo "🚀 [Builder] 开始全量资源采集..."

# --- 1. 下载 Python & PyTorch ---
wget -nc -q -O "$PYTHON_DIR/python-3.10.tar.gz" "https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"

BASE_URL="https://download.pytorch.org/whl/cu121"
for pkg in "torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"; do
    wget -nc -q -P $WHEEL_DIR "$BASE_URL/$pkg"
done

# --- 2. 关键：只下载 Flash Attention 预编译 Wheel ---
echo "⚡ 下载 Flash Attention 预编译包..."
# 1. 先清理掉所有源码包，防止误用！
rm -f "$WHEEL_DIR/flash_attn"*.tar.gz
# 2. 下载二进制包
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -nc -q -P $WHEEL_DIR "$FLASH_URL"

# Transformers 源码
wget -nc -q -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

# --- 3. 补全所有依赖 ---
echo "📚 补全依赖 (含 exceptiongroup)..."
download_dep() {
    pip download "$@" --dest $WHEEL_DIR --index-url https://pypi.org/simple \
        --python-version 3.10 --platform manylinux2014_x86_64 \
        --only-binary=:all: --no-deps --quiet
}

# 补全报错缺失的 exceptiongroup
download_dep exceptiongroup
# 补全其他
download_dep numpy==1.26.4 packaging ninja psutil setuptools wheel einops
download_dep httpx httpcore anyio sniffio h11 hf-xet
download_dep aiohttp aiohappyeyeballs yarl multidict frozenlist aiosignal attrs
download_dep requests urllib3 idna certifi charset-normalizer
download_dep accelerate huggingface-hub tokenizers safetensors pyyaml tqdm
download_dep rich pygments markdown-it-py mdurl shellingham click typer typer-slim colorama
download_dep filelock fsspec typing-extensions
download_dep datasets pandas scipy pillow timm sentence-transformers
download_dep easyocr scikit-image python-bidi protobuf sentencepiece
download_dep dill multiprocess pyarrow regex sympy networkx jinja2 MarkupSafe mpmath
# NVIDIA Runtime
download_dep nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1 \
             nvidia-cudnn-cu12==9.1.0.70 nvidia-nvjitlink-cu12==12.1.105 \
             nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 \
             nvidia-nccl-cu12==2.20.5 triton==3.0.0 nvidia-nvtx-cu12==12.1.105 \
             nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-cupti-cu12==12.1.105 \
             nvidia-cufft-cu12==11.0.2.54 nvidia-cusparse-cu12==12.1.0.106

echo "------------------------------------------------"
# 最终检查：确保没有 .tar.gz 的 flash-attn
if ls $WHEEL_DIR/flash_attn*.tar.gz 1> /dev/null 2>&1; then
    echo "⚠️  警告：发现 Flash Attention 源码包，正在自动删除..."
    rm "$WHEEL_DIR/flash_attn"*.tar.gz
fi

echo "✅ 采集完成！exceptiongroup 已补全，源码包已清理。"

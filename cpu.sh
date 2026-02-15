#!/bin/bash
# ==============================================================================
# AURORA Offline Downloader (v5.0 - Final)
# ------------------------------------------------------------------------------
# 策略：全量下载 + 源码包清洗 + 完整依赖覆盖
# ==============================================================================

SAVE_DIR="./offline_packages"
WHEEL_DIR="$SAVE_DIR/wheels"
PYTHON_DIR="$SAVE_DIR/python_runtime"
mkdir -p $WHEEL_DIR $PYTHON_DIR

echo "🚀 [Builder] 开始全量资源采集..."

# --- 1. Python Runtime ---
wget -nc -q -O "$PYTHON_DIR/python-3.10.tar.gz" "https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"

# --- 2. PyTorch (Binary) ---
BASE_URL="https://download.pytorch.org/whl/cu121"
for pkg in "torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl" \
           "torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"; do
    wget -nc -q -P $WHEEL_DIR "$BASE_URL/$pkg"
done

# --- 3. Flash Attention (Pre-built Wheel) ---
# 强制清理源码包，防止 GPU 误编译
rm -f "$WHEEL_DIR/flash_attn"*.tar.gz
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -nc -q -P $WHEEL_DIR "$FLASH_URL"

# --- 4. Transformers (Source) ---
wget -nc -q -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

# --- 5. 依赖全家桶 ---
echo "📚 下载依赖 (含 Scikit-Learn)..."
download_dep() {
    pip download "$@" --dest $WHEEL_DIR --index-url https://pypi.org/simple \
        --python-version 3.10 --platform manylinux2014_x86_64 \
        --only-binary=:all: --no-deps --quiet
}

# 基础工具
download_dep numpy==1.26.4 packaging ninja psutil setuptools wheel einops
# 科学计算 (修复 sentence-transformers 依赖)
download_dep scikit-learn==1.3.2 joblib threadpoolctl scipy pandas
# 网络库
download_dep httpx httpcore anyio sniffio h11 hf-xet exceptiongroup
download_dep aiohttp aiohappyeyeballs yarl multidict frozenlist aiosignal attrs
download_dep requests urllib3 idna certifi charset-normalizer
# 框架工具
download_dep accelerate huggingface-hub tokenizers safetensors pyyaml tqdm
download_dep rich pygments markdown-it-py mdurl shellingham click typer typer-slim colorama
download_dep filelock fsspec typing-extensions
# 图像与业务
download_dep datasets pillow timm sentence-transformers
download_dep easyocr scikit-image python-bidi protobuf sentencepiece
download_dep dill multiprocess pyarrow regex sympy networkx jinja2 MarkupSafe mpmath
# NVIDIA Runtime
download_dep nvidia-cuda-runtime-cu12==12.1.105 nvidia-cublas-cu12==12.1.3.1 \
             nvidia-cudnn-cu12==9.1.0.70 nvidia-nvjitlink-cu12==12.1.105 \
             nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 \
             nvidia-nccl-cu12==2.20.5 triton==3.0.0 nvidia-nvtx-cu12==12.1.105 \
             nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-cupti-cu12==12.1.105 \
             nvidia-cufft-cu12==11.0.2.54 nvidia-cusparse-cu12==12.1.0.106

echo "✅ 采集完成！Scikit-Learn 与 Flash-Attn Wheel 已就绪。"

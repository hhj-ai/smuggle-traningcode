#!/bin/bash

# ==============================================================================
# AURORA Offline Downloader (Final Audit Edition)
# ------------------------------------------------------------------------------
# 策略：显式列出所有 transitive dependencies，绝不依赖自动解析。
# 目标环境：Python 3.10 | Linux x86_64 | CUDA 12.1
# ==============================================================================

SAVE_DIR="./offline_packages"
WHEEL_DIR="$SAVE_DIR/wheels"
PYTHON_DIR="$SAVE_DIR/python_runtime"

mkdir -p $WHEEL_DIR $PYTHON_DIR

# ------------------------------------------------------------------------------
# 1. 核心下载函数 (平台欺骗模式)
# ------------------------------------------------------------------------------
download_dep() {
    # 强制 pip 认为自己是 Linux x86_64 的 Python 3.10
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps \
        --quiet
    if [ $? -eq 0 ]; then echo "   ✅ OK: $*"; else echo "   ❌ FAIL: $*"; fi
}

echo "🚀 [Start] 开始全量资源采集..."

# ------------------------------------------------------------------------------
# 2. Python 3.10 运行时
# ------------------------------------------------------------------------------
echo "📦 [1/6] Python Runtime..."
wget -nc -q -O "$PYTHON_DIR/python-3.10.tar.gz" "https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"

# ------------------------------------------------------------------------------
# 3. 核心计算框架 (PyTorch + FlashAttn) - 直链下载
# ------------------------------------------------------------------------------
echo "🔥 [2/6] PyTorch Core & FlashAttn (Binary)..."
BASE_URL="https://download.pytorch.org/whl/cu121"
wget -nc -q -P $WHEEL_DIR "$BASE_URL/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -q -P $WHEEL_DIR "$BASE_URL/torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -q -P $WHEEL_DIR "$BASE_URL/torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

# Flash Attention 2.6.3 预编译包 (无需编译)
wget -nc -q -P $WHEEL_DIR "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Transformers 源码
wget -nc -q -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

# ------------------------------------------------------------------------------
# 4. NVIDIA Runtime (PyTorch 2.x 必需) - 查漏补缺
# ------------------------------------------------------------------------------
echo "🎮 [3/6] NVIDIA CUDA Dependencies..."
download_dep nvidia-cuda-runtime-cu12==12.1.105
download_dep nvidia-cuda-nvrtc-cu12==12.1.105
download_dep nvidia-cuda-cupti-cu12==12.1.105
download_dep nvidia-cudnn-cu12==9.1.0.70
download_dep nvidia-cublas-cu12==12.1.3.1
download_dep nvidia-cufft-cu12==11.0.2.54
download_dep nvidia-curand-cu12==10.3.2.106
download_dep nvidia-cusolver-cu12==11.4.5.107
download_dep nvidia-cusparse-cu12==12.1.0.106
download_dep nvidia-nccl-cu12==2.20.5
download_dep nvidia-nvtx-cu12==12.1.105
download_dep triton==3.0.0
# [关键补全] 之前漏掉的包
download_dep nvidia-nvjitlink-cu12==12.1.105

# ------------------------------------------------------------------------------
# 5. 网络与异步库 (aiohttp/requests 全家桶)
# ------------------------------------------------------------------------------
echo "🌐 [4/6] Network & Async Stack..."
download_dep aiohttp
download_dep aiohappyeyeballs  # 新版 aiohttp 必需
download_dep aiosignal
download_dep attrs
download_dep frozenlist
download_dep multidict
download_dep yarl
download_dep async-timeout
download_dep requests
download_dep urllib3
download_dep idna
download_dep certifi
download_dep charset-normalizer
# HuggingFace 新版依赖
download_dep httpx
download_dep httpcore
download_dep h11
download_dep anyio
download_dep sniffio
download_dep hf-xet

# ------------------------------------------------------------------------------
# 6. 数据处理与图像 (Pandas/Scipy/Pillow)
# ------------------------------------------------------------------------------
echo "🖼️  [5/6] Data & Image Stack..."
download_dep numpy==1.26.4
download_dep pandas
download_dep python-dateutil
download_dep pytz
download_dep six
download_dep tzdata
download_dep scipy
download_dep scikit-image
download_dep imageio
download_dep tifffile
download_dep lazy_loader
download_dep networkx
download_dep Pillow
download_dep python-bidi
download_dep opencv-python-headless
download_dep shapely
download_dep pyarrow
download_dep dill
download_dep multiprocess
download_dep xxhash

# ------------------------------------------------------------------------------
# 7. 核心框架与工具 (HF/Rich/EasyOCR)
# ------------------------------------------------------------------------------
echo "🛠️  [6/6] Frameworks & Utils..."
download_dep accelerate
download_dep huggingface-hub
download_dep tokenizers
download_dep safetensors
download_dep pyyaml
download_dep tqdm
download_dep filelock
download_dep fsspec
download_dep typing-extensions
download_dep packaging
download_dep psutil
download_dep regex
download_dep sympy
download_dep jinja2
download_dep MarkupSafe
download_dep mpmath
download_dep sentencepiece
download_dep protobuf
# Rich 全家桶
download_dep rich
download_dep pygments
download_dep markdown-it-py
download_dep mdurl
download_dep colorama
# CLI 工具
download_dep click
download_dep typer
download_dep typer-slim
download_dep shellingham
# 业务库
download_dep easyocr
download_dep timm
download_dep sentence-transformers
download_dep einops

# ------------------------------------------------------------------------------
# 最终核验
# ------------------------------------------------------------------------------
echo "------------------------------------------------"
echo "🕵️  Final Audit..."
COUNT=$(ls $WHEEL_DIR | wc -l)
echo "📦 总包数: $COUNT"

# 关键检查清单
CRITICAL=("nvidia_nvjitlink" "aiohappyeyeballs" "flash_attn" "torch-2" "numpy" "rich" "hf_xet" "scikit_image")
MISSING=0
for pkg in "${CRITICAL[@]}"; do
    if [ $(find $WHEEL_DIR -iname "*$pkg*" | wc -l) -eq 0 ]; then
        echo "❌ MISSING: $pkg"
        MISSING=1
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "🎉 完美！所有已知的坑都已填平。请打包: $SAVE_DIR"
else
    echo "⛔ 依然有缺失，请检查网络日志。"
fi

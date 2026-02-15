#!/bin/bash

# ==============================================================================
# AURORA Offline Environment Builder (Download Script)
# ------------------------------------------------------------------------------
# Description: Downloads Python runtime, PyTorch, Transformers, and all dependencies
#              into a portable folder for offline installation.
# Target OS:   Linux x86_64
# Target Py:   Python 3.10
# Target HW:   NVIDIA GPU (CUDA 12.1)
# ==============================================================================

# --- Configuration ---
SAVE_DIR="./offline_packages"
PYTHON_DIR="$SAVE_DIR/python_runtime"
WHEEL_DIR="$SAVE_DIR/wheels"
DATA_DIR="$SAVE_DIR/datasets"
WEIGHTS_DIR="$SAVE_DIR/tool_weights"

# Versions
PY_VER="3.10"
CUDA_VER="cu121"
TORCH_VER="2.4.1"
VISION_VER="0.19.1"
AUDIO_VER="2.4.1"

# Create directories
mkdir -p $PYTHON_DIR $WHEEL_DIR $DATA_DIR $WEIGHTS_DIR

echo "🚀 [Builder] Starting offline resource collection..."
echo "📂 Output Directory: $SAVE_DIR"

# ==============================================================================
# 1. Download Portable Python Runtime (Standalone Build)
# ==============================================================================
echo "🐍 [1/7] Downloading Portable Python 3.10..."
# Source: https://github.com/indygreg/python-build-standalone
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"

if [ ! -f "$PYTHON_DIR/python-3.10.tar.gz" ]; then
    wget -c -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL" || curl -L -o "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
else
    echo "   ✅ Python runtime already exists."
fi

# ==============================================================================
# 2. Download PyTorch Core (Direct Link to avoid pip platform issues)
# ==============================================================================
echo "🔥 [2/7] Downloading PyTorch Core ($CUDA_VER)..."
TORCH_BASE_URL="https://download.pytorch.org/whl/$CUDA_VER"

# Function to download if not exists
wget_wheel() {
    local url=$1
    local filename=$(basename $url)
    if [ ! -f "$WHEEL_DIR/$filename" ]; then
        echo "   ⬇️  Downloading $filename..."
        wget -q -P $WHEEL_DIR "$url" || echo "   ❌ Failed to download $filename"
    else
        echo "   ✅ $filename exists."
    fi
}

# Note: %2B is URL encoded '+'
wget_wheel "$TORCH_BASE_URL/torch-${TORCH_VER}%2B${CUDA_VER}-cp310-cp310-linux_x86_64.whl"
wget_wheel "$TORCH_BASE_URL/torchvision-${VISION_VER}%2B${CUDA_VER}-cp310-cp310-linux_x86_64.whl"
wget_wheel "$TORCH_BASE_URL/torchaudio-${AUDIO_VER}%2B${CUDA_VER}-cp310-cp310-linux_x86_64.whl"

# ==============================================================================
# 3. Download Transformers & Flash Attention
# ==============================================================================
echo "🤗 [3/7] Downloading Transformers (Source)..."
# We download the main branch source code to support latest models (e.g., Qwen3-VL)
if [ ! -f "$WHEEL_DIR/transformers-main.zip" ]; then
    wget -q -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"
fi

echo "⚡ [4/7] Downloading Flash Attention 2..."
# Use pip download with --no-deps to avoid resolving torch dependency on CPU machine
pip download flash-attn==2.6.3 --dest $WHEEL_DIR --index-url https://pypi.org/simple --no-binary :all: --no-deps --quiet

# ==============================================================================
# 4. Download Dependencies (Cross-Platform / No-Deps Mode)
# ==============================================================================
echo "📚 [5/7] Downloading Dependencies (Simulating Linux x86_64 / Py3.10)..."

# Helper function for cross-downloading
download_dep() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps \
        --quiet
}

# --- 4.1 NVIDIA CUDA Libraries (Critical for PyTorch 2.x) ---
echo "   -> NVIDIA CUDA Dependencies..."
download_dep nvidia-cuda-nvrtc-cu12==12.1.105
download_dep nvidia-cuda-runtime-cu12==12.1.105
download_dep nvidia-cuda-cupti-cu12==12.1.105
download_dep nvidia-cudnn-cu12==9.1.0.70
download_dep nvidia-cublas-cu12==12.1.3.1
download_dep nvidia-cufft-cu12==11.0.2.54
download_dep nvidia-curand-cu12==10.3.2.106
download_dep nvidia-cusolver-cu12==11.4.5.107
download_dep nvidia-cusparse-cu12==12.1.0.106
download_dep nvidia-nccl-cu12==2.20.5
download_dep nvidia-nvtx-cu12==12.1.105
download_dep nvidia-nvjitlink-cu12==12.1.105 # Critical missing dep fix
download_dep triton==3.0.0

# --- 4.2 Build Tools ---
echo "   -> Build Tools (Numpy, Ninja, Packaging)..."
download_dep "numpy<2.0.0"
download_dep packaging ninja psutil setuptools wheel

# --- 4.3 General Dependencies ---
echo "   -> General Python Packages..."
download_dep accelerate>=0.27.0 huggingface-hub>=0.23.0 tokenizers>=0.19.1 safetensors>=0.4.1
download_dep regex requests filelock fsspec pyyaml tqdm
download_wheel sympy networkx jinja2 MarkupSafe typing-extensions mpmath
download_wheel charset-normalizer idna urllib3 certifi
download_dep datasets sentence-transformers Pillow easyocr scipy
download_dep termcolor timm rich questionary aiohttp protobuf sentencepiece
download_dep opencv-python-headless scikit-image python-bidi PyYAML
download_dep attrs multidict yarl frozenlist aiosignal async-timeout
download_dep pandas pytz python-dateutil six

# ==============================================================================
# 5. Download Benchmarks & Tool Weights
# ==============================================================================
echo "📊 [6/7] Downloading Benchmark Data..."

# POPE
mkdir -p "$DATA_DIR/pope"
wget -q -nc -O "$DATA_DIR/pope/coco_pope_random.json" "https://huggingface.co/datasets/shiyue/POPE/resolve/main/output/coco/coco_pope_random.json"

# Helper for HF Snapshot
cat <<EOF > _dl_helper.py
import os
from huggingface_hub import snapshot_download
try:
    # MMHal
    snapshot_download(repo_id="Shengcao1006/MMHal-Bench", repo_type="dataset", local_dir="$DATA_DIR/mmhal_bench", resume_download=True)
    # Sentence Transformers
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="$WEIGHTS_DIR/sentence-transformers/all-MiniLM-L6-v2", resume_download=True)
except: pass
EOF
# Try running helper (requires hf_hub on host, harmless if fails)
pip install huggingface_hub -i https://pypi.org/simple --quiet 2>/dev/null
python3 _dl_helper.py && rm _dl_helper.py

echo "🛠️  [7/7] Downloading Tool Weights..."
# EasyOCR
mkdir -p "$WEIGHTS_DIR/easyocr"
wget -q -nc -O "$WEIGHTS_DIR/easyocr/craft_mlt_25k.zip" "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip"
wget -q -nc -O "$WEIGHTS_DIR/easyocr/english_g2.zip" "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
unzip -q -o "$WEIGHTS_DIR/easyocr/craft_mlt_25k.zip" -d "$WEIGHTS_DIR/easyocr"
unzip -q -o "$WEIGHTS_DIR/easyocr/english_g2.zip" -d "$WEIGHTS_DIR/easyocr"
rm "$WEIGHTS_DIR/easyocr"/*.zip

# GroundingDINO
mkdir -p "$WEIGHTS_DIR/dino"
wget -q -nc -P "$WEIGHTS_DIR/dino" "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
wget -q -nc -P "$WEIGHTS_DIR/dino" "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# ==============================================================================
# Final Verification
# ==============================================================================
echo "------------------------------------------------"
echo "🕵️  Running Integrity Check..."

MISSING=0
CHECK_LIST=("torch" "transformers" "numpy" "nvidia_cublas" "nvidia_nvjitlink" "packaging")

for pkg in "${CHECK_LIST[@]}"; do
    if [ $(find $WHEEL_DIR -name "*$pkg*" | wc -l) -eq 0 ]; then
        echo "❌ Missing CRITICAL package: $pkg"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "⛔ Verification FAILED. Do not proceed to offline installation."
    exit 1
else
    echo "✅ Verification PASSED. Bundle is ready."
    echo "📂 Bundle Path: $(realpath $SAVE_DIR)"
fi

#!/bin/bash

# ==============================================================================
# AURORA Offline Installer
# ------------------------------------------------------------------------------
# Description: Installs the portable environment on an offline GPU server.
#              Does NOT delete existing environment; supports re-run.
# Usage:       bash 2_offline_install.sh
# ==============================================================================

# --- Configuration ---
BASE_DIR="./offline_packages"
PYTHON_TGZ="$BASE_DIR/python_runtime/python-3.10.tar.gz"
WHEEL_DIR="$BASE_DIR/wheels"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

# --- Data Configuration ---
DATA_SOURCE="$BASE_DIR/datasets"
WEIGHTS_SRC="$BASE_DIR/tool_weights"
BENCH_DEST="./data/benchmarks"

echo "🚀 [Installer] Starting offline installation..."

# ==============================================================================
# 1. Setup Portable Python (Skip if exists)
# ==============================================================================
if [ -f "$INSTALL_ROOT/bin/python3" ] || [ -f "$INSTALL_ROOT/python/bin/python3" ]; then
    echo "✅ Python runtime already found in $INSTALL_ROOT. Skipping extraction."
else
    if [ ! -f "$PYTHON_TGZ" ]; then echo "❌ Python tarball not found: $PYTHON_TGZ"; exit 1; fi
    echo "🐍 Extracting Python 3.10 Runtime..."
    mkdir -p $INSTALL_ROOT
    tar -xzf $PYTHON_TGZ -C $INSTALL_ROOT
fi

# Locate python binary
if [ -d "$INSTALL_ROOT/python" ]; then 
    LOCAL_PYTHON="$INSTALL_ROOT/python/bin/python3"
else 
    LOCAL_PYTHON="$INSTALL_ROOT/bin/python3"
fi

# ==============================================================================
# 2. Setup Virtual Environment (Skip if exists)
# ==============================================================================
if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment '$VENV_DIR' exists. Activating..."
else
    echo "📦 Creating virtual environment..."
    $LOCAL_PYTHON -m venv $VENV_DIR
fi

# Activate
source $VENV_DIR/bin/activate

# Configure PIP for offline mode
pip config set global.no-index true > /dev/null 2>&1
pip config set global.find-links $(pwd)/$WHEEL_DIR > /dev/null 2>&1

# Helper function
install_pkg() {
    # Only install if not satisfied (pip default behavior)
    pip install "$@" --no-index --find-links=$WHEEL_DIR
}

# ==============================================================================
# 3. Install Dependencies (Strict Order)
# ==============================================================================
echo "🧱 [1/5] Installing Build Tools (Numpy, Packaging)..."
# Critical: Must install these before Torch/FlashAttn
install_pkg wheel setuptools packaging ninja psutil "numpy<2.0.0"

echo "🎮 [2/5] Installing NVIDIA Dependencies..."
# Critical: nvjitlink must be present for cusolver
install_pkg nvidia-nvjitlink-cu12 
install_pkg nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12
install_pkg nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cufft-cu12 
install_pkg nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 
install_pkg nvidia-nccl-cu12 nvidia-nvtx-cu12 triton

echo "🔥 [3/5] Installing PyTorch..."
install_pkg torch torchvision torchaudio

echo "⚡ [4/5] Compiling Flash Attention..."
# Check if already installed to save time
if python -c "import flash_attn" > /dev/null 2>&1; then
    echo "   ✅ Flash Attention already installed. Skipping compilation."
else
    install_pkg flash-attn --no-build-isolation
fi

echo "🤗 [5/5] Installing Transformers & Others..."
# Transformers Source Install
if [ -f "$WHEEL_DIR/transformers-main.zip" ]; then
    # Use pip to install directly from zip without unzipping manually
    install_pkg "$WHEEL_DIR/transformers-main.zip"
else
    install_pkg transformers
fi

# Install remaining deps
install_pkg accelerate huggingface_hub datasets sentence-transformers Pillow easyocr
install_pkg scipy termcolor timm rich questionary aiohttp protobuf sentencepiece pandas

# ==============================================================================
# 4. Deploy Data & Weights (Idempotent)
# ==============================================================================
echo "📂 Deploying Data and Weights..."
mkdir -p $BENCH_DEST
mkdir -p ./weights

# POPE
if [ ! -f "$BENCH_DEST/pope_coco_random.json" ] && [ -f "$DATA_SOURCE/pope/coco_pope_random.json" ]; then
    cp "$DATA_SOURCE/pope/coco_pope_random.json" "$BENCH_DEST/pope_coco_random.json"
fi

# MMHal
if [ ! -d "$BENCH_DEST/mmhal_bench" ] && [ -d "$DATA_SOURCE/mmhal_bench" ]; then
    cp -r "$DATA_SOURCE/mmhal_bench" "$BENCH_DEST/"
    # Extract images helper
    echo "   ⚙️  Extracting MMHal images..."
    cat <<EOF > _extract.py
import os, json
from datasets import load_dataset
try:
    ds = load_dataset("$BENCH_DEST/mmhal_bench", split="test")
    os.makedirs("./data/test_images", exist_ok=True)
    data = []
    for i, item in enumerate(ds):
        if item.get("image"): item["image"].convert("RGB").save(f"./data/test_images/mmhal_{i}.jpg")
        data.append({"question_id": i, "question": item["question"], "gt_answer": item["answer"], "image_id": f"mmhal_{i}.jpg"})
    with open("$BENCH_DEST/mmhal_bench.json", "w") as f: json.dump(data, f, indent=2)
except: pass
EOF
    python _extract.py && rm _extract.py
fi

# Tool Weights
# EasyOCR
OCR_DEST="$HOME/.EasyOCR/model"
if [ ! -f "$OCR_DEST/craft_mlt_25k.pth" ]; then
    mkdir -p "$OCR_DEST"
    cp "$WEIGHTS_SRC/easyocr/"*.pth "$OCR_DEST/" 2>/dev/null
fi

# DINO
if [ ! -f "./weights/groundingdino_swint_ogc.pth" ]; then
    cp "$WEIGHTS_SRC/dino/"* "./weights/" 2>/dev/null
fi

# Sentence Transformers
ST_DEST="$HOME/.cache/torch/sentence_transformers"
if [ ! -d "$ST_DEST/sentence-transformers_all-MiniLM-L6-v2" ] && [ -d "$WEIGHTS_SRC/sentence-transformers/all-MiniLM-L6-v2" ]; then
    mkdir -p "$ST_DEST"
    cp -r "$WEIGHTS_SRC/sentence-transformers/all-MiniLM-L6-v2" "$ST_DEST/sentence-transformers_all-MiniLM-L6-v2"
fi

echo "------------------------------------------------"
echo "🎉 Installation Complete!"
echo "👉 To start: source $VENV_DIR/bin/activate"

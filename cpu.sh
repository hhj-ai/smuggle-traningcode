#!/bin/bash

# ========================================================================
# 1_full_download.sh (CPU 服务器 - 终极全量资源版)
# 目标: 下载 Python3.10 + 依赖包 + 评测数据 + 工具权重(OCR/DINO)
# 特性: 暴力直链下载，绕过本地 pip 版本检查
# ========================================================================

# 1. 设置目录结构
SAVE_DIR="./offline_packages"
PYTHON_DIR="$SAVE_DIR/python_runtime"
WHEEL_DIR="$SAVE_DIR/wheels"
DATA_DIR="$SAVE_DIR/datasets"
WEIGHTS_DIR="$SAVE_DIR/tool_weights"

mkdir -p $PYTHON_DIR $WHEEL_DIR $DATA_DIR $WEIGHTS_DIR

echo "🚀 [CPU Server] 开始构建全量离线资源..."
echo "📂 保存路径: $SAVE_DIR"

# =========================================================
# [Part A] 下载独立版 Python 3.10 (无需安装，解压即用)
# =========================================================
echo "🐍 [1/7] 下载 Python 3.10 Runtime..."
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"
if [ ! -f "$PYTHON_DIR/python-3.10.tar.gz" ]; then
    wget -c -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL" || curl -L -o "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
fi

# =========================================================
# [Part B] 暴力下载核心框架 (Wget 直链)
# =========================================================
echo "🔥 [2/7] 下载 PyTorch (CUDA 12.1)..."
BASE_URL="https://download.pytorch.org/whl/cu121"
# 直接指定 Py3.10/Linux 版本
wget -nc -P $WHEEL_DIR "$BASE_URL/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -P $WHEEL_DIR "$BASE_URL/torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -P $WHEEL_DIR "$BASE_URL/torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

echo "🤗 [3/7] 下载 Transformers (GitHub Main)..."
# 下载最新源码以支持 Qwen3-VL
wget -nc -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

echo "⚡ [4/7] 下载 Flash Attention 2..."
# 加上 --no-deps 防止检查 torch
pip download flash-attn==2.6.3 --dest $WHEEL_DIR --index-url https://pypi.org/simple --trusted-host pypi.org --no-binary :all: --no-deps

# =========================================================
# [Part C] 下载通用依赖 (伪装 Py3.10, No-Deps)
# =========================================================
echo "📚 [5/7] 下载通用依赖 (伪装 Py3.10)..."

download_wheel() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps
}

# 基础构建工具
download_wheel pip setuptools wheel packaging ninja psutil
# HF 生态
download_wheel accelerate>=0.27.0 huggingface-hub>=0.23.0 tokenizers>=0.19.1 safetensors>=0.4.1
download_wheel regex requests filelock fsspec pyyaml tqdm
download_wheel charset-normalizer idna urllib3 certifi
# Torch 依赖
download_wheel sympy networkx jinja2 MarkupSafe typing-extensions mpmath
# 业务/评估工具依赖
download_wheel datasets sentence-transformers numpy<2.0.0 Pillow easyocr scipy
download_wheel termcolor timm rich questionary aiohttp protobuf sentencepiece
download_wheel opencv-python-headless scikit-image python-bidi PyYAML
download_wheel attrs multidict yarl frozenlist aiosignal async-timeout
download_wheel pandas pytz python-dateutil six

# =========================================================
# [Part D] 下载评测数据集 (POPE & MMHal)
# =========================================================
echo "📊 [6/7] 下载评测数据集..."

# POPE
mkdir -p "$DATA_DIR/pope"
POPE_URL="https://huggingface.co/datasets/shiyue/POPE/resolve/main/output/coco/coco_pope_random.json"
wget -c -O "$DATA_DIR/pope/coco_pope_random.json" "$POPE_URL"

# MMHal-Bench & Sentence Transformers (需下载文件夹)
# 我们尝试安装一个临时的 huggingface_hub 来下载 (如果当前环境能装的话)
echo "   ... 尝试安装 huggingface_hub 用于下载数据 ..."
pip install huggingface_hub -i https://pypi.org/simple --trusted-host pypi.org >/dev/null 2>&1

cat <<EOF > download_data_repos.py
import os
from huggingface_hub import snapshot_download

def dl(repo, local):
    try:
        print(f"   ⬇️  Downloading {repo}...")
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=local, resume_download=True)
    except Exception as e: print(f"   ❌ Error {repo}: {e}")

def dl_model(repo, local):
    try:
        print(f"   ⬇️  Downloading Model {repo}...")
        snapshot_download(repo_id=repo, local_dir=local, resume_download=True)
    except Exception as e: print(f"   ❌ Error {repo}: {e}")

# MMHal
dl("Shengcao1006/MMHal-Bench", "$DATA_DIR/mmhal_bench")

# Sentence Transformers (用于评估脚本)
dl_model("sentence-transformers/all-MiniLM-L6-v2", "$WEIGHTS_DIR/sentence-transformers/all-MiniLM-L6-v2")
EOF

python3 download_data_repos.py
rm download_data_repos.py

# =========================================================
# [Part E] 下载工具权重 (OCR & DINO)
# =========================================================
echo "🛠️  [7/7] 下载工具权重 (OCR & DINO)..."

# EasyOCR
OCR_DIR="$WEIGHTS_DIR/easyocr"
mkdir -p $OCR_DIR
wget -nc -O "$OCR_DIR/craft_mlt_25k.zip" "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip"
unzip -o -q "$OCR_DIR/craft_mlt_25k.zip" -d "$OCR_DIR"
wget -nc -O "$OCR_DIR/english_g2.zip" "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
unzip -o -q "$OCR_DIR/english_g2.zip" -d "$OCR_DIR"
rm "$OCR_DIR"/*.zip

# GroundingDINO
DINO_DIR="$WEIGHTS_DIR/dino"
mkdir -p $DINO_DIR
wget -nc -P $DINO_DIR "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
wget -nc -P $DINO_DIR "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

echo "------------------------------------------------"
echo "✅ 全量资源准备完毕！"
echo "👉 请将 offline_packages 目录对 GPU 服务器可见。"

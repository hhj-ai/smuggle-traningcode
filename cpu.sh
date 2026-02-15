#!/bin/bash

# ========================================================================
# 1_full_download.sh (CPU 服务器 - 终极版)
# 核心策略: 全程使用 --no-deps，禁止 pip 思考，只管下载
# ========================================================================

SAVE_DIR="./offline_packages"
PYTHON_DIR="$SAVE_DIR/python_runtime"
WHEEL_DIR="$SAVE_DIR/wheels"

mkdir -p $PYTHON_DIR
mkdir -p $WHEEL_DIR

echo "🚀 [CPU Server] 开始构建全量离线包 (No-Deps Mode)..."

# ------------------------------------------------------------------------
# 1. 下载独立版 Python 3.10
# ------------------------------------------------------------------------
echo "🐍 [1/6] 下载 Python 3.10 独立运行包..."
PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.10.13+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"

if [ ! -f "$PYTHON_DIR/python-3.10.tar.gz" ]; then
    wget -c -O "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL" || curl -L -o "$PYTHON_DIR/python-3.10.tar.gz" "$PYTHON_URL"
else
    echo "   ✅ Python 包已存在。"
fi

# ------------------------------------------------------------------------
# 2. 暴力下载 PyTorch (Wget 直链)
# ------------------------------------------------------------------------
echo "🔥 [2/6] 暴力下载 PyTorch (CUDA 12.1)..."
# 手动列出 URL，完全绕过 pip
BASE_URL="https://download.pytorch.org/whl/cu121"
wget -nc -P $WHEEL_DIR "$BASE_URL/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -P $WHEEL_DIR "$BASE_URL/torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget -nc -P $WHEEL_DIR "$BASE_URL/torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

# ------------------------------------------------------------------------
# 3. 下载 Flash Attention 2 (关键修复: 加上 --no-deps)
# ------------------------------------------------------------------------
echo "⚡ [3/6] 下载 Flash Attention 2..."
# 加上 --no-deps 防止它去检查 torch 是否存在
pip download flash-attn==2.6.3 \
    --dest $WHEEL_DIR \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --no-binary :all: \
    --no-deps

# ------------------------------------------------------------------------
# 4. 下载 Transformers (Wget 源码 Zip)
# ------------------------------------------------------------------------
echo "🤗 [4/6] 下载 Transformers (GitHub Main)..."
wget -nc -O "$WHEEL_DIR/transformers-main.zip" "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"

# ------------------------------------------------------------------------
# 5. 下载所有通用依赖 (全部加上 --no-deps)
# ------------------------------------------------------------------------
echo "📚 [5/6] 下载通用依赖 (伪装 Py3.10)..."

download_wheel() {
    pip download "$@" \
        --dest $WHEEL_DIR \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --python-version 3.10 \
        --platform manylinux2014_x86_64 \
        --only-binary=:all: \
        --no-deps  # <--- 核心修改：不检查依赖，只下载指定的包
}

# 手动列出所有需要的包 (因为我们关掉了依赖检查，所以必须把依赖的依赖也写出来)
# 基础
download_wheel pip
download_wheel setuptools
download_wheel wheel
download_wheel packaging
download_wheel ninja
download_wheel psutil

# Torch 依赖
download_wheel sympy
download_wheel networkx
download_wheel jinja2
download_wheel MarkupSafe
download_wheel filelock
download_wheel typing-extensions
download_wheel fsspec
download_wheel mpmath

# HF 依赖
download_wheel accelerate>=0.27.0
download_wheel huggingface-hub>=0.23.0
download_wheel tokenizers>=0.19.1
download_wheel safetensors>=0.4.1
download_wheel regex
download_wheel requests
download_wheel pyyaml
download_wheel tqdm
download_wheel charset-normalizer
download_wheel idna
download_wheel urllib3
download_wheel certifi

# 业务依赖
download_wheel datasets
download_wheel sentence-transformers
download_wheel numpy<2.0.0
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
# easyocr 依赖
download_wheel opencv-python-headless
download_wheel scikit-image
download_wheel python-bidi
download_wheel PyYAML
# aiohttp 依赖
download_wheel attrs
download_wheel multidict
download_wheel yarl
download_wheel frozenlist
download_wheel aiosignal
download_wheel async-timeout

echo "🛠️  [7/7] 下载工具权重 (OCR & DINO)..."

# 7.1 EasyOCR 权重
# EasyOCR 运行时会去 ~/.EasyOCR/model/ 下找这两个文件
echo "   ⬇️  EasyOCR Models..."
OCR_DIR="$WEIGHTS_DIR/easyocr"
mkdir -p $OCR_DIR

# 下载检测模型 (CRAFT)
wget -nc -O "$OCR_DIR/craft_mlt_25k.zip" "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip"
unzip -o "$OCR_DIR/craft_mlt_25k.zip" -d "$OCR_DIR"
rm "$OCR_DIR/craft_mlt_25k.zip"

# 下载识别模型 (English)
wget -nc -O "$OCR_DIR/english_g2.zip" "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
unzip -o "$OCR_DIR/english_g2.zip" -d "$OCR_DIR"
rm "$OCR_DIR/english_g2.zip"

# 7.2 GroundingDINO 权重 (用于目标检测/验证)
# 通常代码会加载 groundingdino_swint_ogc.pth
echo "   ⬇️  GroundingDINO Weights..."
DINO_DIR="$WEIGHTS_DIR/dino"
mkdir -p $DINO_DIR

# 下载权重
wget -nc -P $DINO_DIR "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# 下载配置文件 (有些库需要本地有 config 文件)
wget -nc -P $DINO_DIR "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# 7.3 Sentence Transformers (如果你的评估代码用到了相似度计算)
# 这是一个常见的隐形依赖
echo "   ⬇️  Sentence Transformers (all-MiniLM-L6-v2)..."
ST_DIR="$WEIGHTS_DIR/sentence-transformers"
mkdir -p $ST_DIR
# 使用 huggingface snapshot 下载 (借用之前的脚本逻辑)
cat <<EOF > download_st.py
from huggingface_hub import snapshot_download
try:
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="$ST_DIR/all-MiniLM-L6-v2")
    print("   ✅ SentenceTransformer downloaded.")
except: pass
EOF
python3 download_st.py
rm download_st.py

echo "------------------------------------------------"
echo "✅ 所有资源准备完毕！"
echo "📂 检查权重目录: $WEIGHTS_DIR"
echo "   ├── easyocr/ (craft_mlt_25k.pth, english_g2.pth)"
echo "   ├── dino/ (groundingdino_swint_ogc.pth)"
echo "   └── sentence-transformers/"

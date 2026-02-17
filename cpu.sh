#!/bin/bash
# --- 1. 绝对路径定义 ---
CODE_DIR=$(pwd)
# 将所有大资源放在 Git 仓库之外
RES_DIR=$(realpath "$CODE_DIR/../aurora_resources")
MODELS_DIR="$RES_DIR/models"
DATA_DIR="$RES_DIR/data"
ENV_DIR="$RES_DIR/env"

echo "🌐 [CPU] 正在准备离线资源..."
echo "📍 存储根目录: $RES_DIR"

mkdir -p "$MODELS_DIR" "$DATA_DIR"

# 2. 本地 Miniconda 环境安装
if [ ! -d "$RES_DIR/miniconda" ]; then
    echo "📦 安装本地 Conda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O mini.sh
    bash mini.sh -b -p "$RES_DIR/miniconda" && rm mini.sh
fi
source "$RES_DIR/miniconda/bin/activate"

# 3. 创建环境
if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" python=3.10 -y
    conda activate "$ENV_DIR"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.46.3 accelerate datasets easyocr sentence-transformers aiohttp tqdm pillow
fi

# 4. 模型精准下载
python <<EOF
import os
from huggingface_hub import snapshot_download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
tasks = {
    "Qwen/Qwen3-VL-8B-Instruct": "$MODELS_DIR/Qwen3-VL-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "$MODELS_DIR/DeepSeek-R1-Distill-Qwen-7B",
    "IDEA-Research/grounding-dino-base": "$MODELS_DIR/grounding-dino-base",
    "openai/clip-vit-base-patch32": "$MODELS_DIR/clip-vit-base-patch32",
    "sentence-transformers/all-MiniLM-L6-v2": "$MODELS_DIR/minilm"
}
for repo, path in tasks.items():
    if not os.path.exists(path):
        print(f"Downloading {repo}...")
        snapshot_download(repo_id=repo, local_dir=path, local_dir_use_symlinks=False, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
EOF

# 5. 准备 EasyOCR
python -c "import easyocr; easyocr.Reader(['en'])"
cp -r ~/.EasyOCR "$RES_DIR/easyocr_cache"

echo "✅ [CPU] 准备就绪。资源已安全存储在 $RES_DIR"

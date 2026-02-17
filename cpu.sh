#!/bin/bash
# --- 统一路径定义 ---
CODE_DIR=$(cd "$(dirname "$0")"; pwd)
RES_DIR=$(realpath "$CODE_DIR/../aurora_resources")
MODELS_DIR="$RES_DIR/models"

echo "🌐 [CPU] 开始资产补全..."
mkdir -p "$MODELS_DIR"
export HF_ENDPOINT="https://hf-mirror.com"

# 1. 强制补全脚本
python <<EOF
import os
from huggingface_hub import snapshot_download

tasks = {
    "IDEA-Research/grounding-dino-base": "grounding-dino-base",
    "openai/clip-vit-base-patch32": "clip-vit-base-patch32",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm"
}

for repo, folder in tasks.items():
    target = os.path.join("$MODELS_DIR", folder)
    print(f"🔍 Checking: {folder}")
    
    # 如果目录不存在或为空，强制下载
    if not os.path.exists(target) or not os.listdir(target):
        print(f"⬇️  Downloading {repo}...")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=target,
                local_dir_use_symlinks=False, # 关键：不用软链接
                resume_download=True
            )
            print(f"✅ {folder} Downloaded.")
        except Exception as e:
            print(f"❌ Error downloading {folder}: {e}")
    else:
        print(f"✔️  {folder} Exists.")
EOF

# 2. 最终确认
echo "--------------------------------"
echo "📂 当前模型目录结构:"
ls -F "$MODELS_DIR"
echo "--------------------------------"
echo "请确认以上列表包含 grounding-dino-base/ 和 clip-vit-base-patch32/"

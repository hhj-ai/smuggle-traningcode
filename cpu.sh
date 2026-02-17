#!/bin/bash
# --- 统一路径定义 ---
CODE_DIR=$(cd "$(dirname "$0")"; pwd)
RES_DIR=$(realpath "$CODE_DIR/../aurora_resources")
MODELS_DIR="$RES_DIR/models"

echo "🌐 [CPU] 正在准备离线资源..."
echo "📍 存储根目录: $RES_DIR"

# 创建目录
mkdir -p "$MODELS_DIR"

# 强制使用国内镜像
export HF_ENDPOINT="https://hf-mirror.com"

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
    target_path = os.path.join("$MODELS_DIR", folder)
    print(f"Checking {folder}...")
    if not os.path.exists(target_path) or not os.listdir(target_path):
        print(f"⬇️  Downloading: {repo}")
        try:
            snapshot_download(
                repo_id=repo, 
                local_dir=target_path, 
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tf"]
            )
        except Exception as e:
            print(f"❌ Failed: {e}")
    else:
        print(f"✔️  Exists.")
EOF

echo "✅ 资产补全完成。目录内容："
ls -F "$MODELS_DIR"

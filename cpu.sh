#!/bin/bash
# --- 路径定义 (务必与共享盘一致) ---
CODE_DIR=$(pwd)
RES_DIR=$(realpath "$CODE_ROOT/../aurora_resources")
MODELS_DIR="$RES_ROOT/models"

echo "🌐 [CPU] 开始精准补全缺失模型资产..."
mkdir -p "$MODELS_DIR"

# 强制使用国内镜像源
export HF_ENDPOINT="https://hf-mirror.com"

python <<EOF
import os
from huggingface_hub import snapshot_download

# 定义 AURORA 运行必须的 5 大组件
tasks = {
    "IDEA-Research/grounding-dino-base": "grounding-dino-base",
    "openai/clip-vit-base-patch32": "clip-vit-base-patch32",
    "Qwen/Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm"
}

for repo, folder in tasks.items():
    target_path = os.path.join("$MODELS_DIR", folder)
    if not os.path.exists(target_path) or not os.listdir(target_path):
        print(f"⬇️  正在下载: {repo} -> {target_path}")
        try:
            snapshot_download(
                repo_id=repo, 
                local_dir=target_path, 
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tf"] # 只下 PT 权重，省空间
            )
            print(f"✅ {folder} 下载完成")
        except Exception as e:
            print(f"❌ {folder} 下载失败: {e}")
    else:
        print(f"✔️  {folder} 已存在，跳过。")
EOF

echo "🎉 [CPU] 所有资产已就绪。请确认 $MODELS_DIR 目录下文件夹完整。"
ls -F "$MODELS_DIR"

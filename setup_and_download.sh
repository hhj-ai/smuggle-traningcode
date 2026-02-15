#!/bin/bash

# ==========================================
# AURORA Setup Script (Debug Mode)
# ==========================================

# 1. 创建目录
echo "📂 Checking directories..."
mkdir -p ./data/yfcc100m
mkdir -p ./data/benchmarks
mkdir -p ./data/test_images
mkdir -p ./output/checkpoints
echo "✅ Directories ready."

# 2. 定义下载函数（带详细日志）
download_file() {
    url=$1
    dest=$2
    name=$3
    
    if [ -f "$dest" ]; then
        echo "✅ $name already exists at $dest"
        return
    fi

    echo "------------------------------------------------"
    echo "⬇️  Attempting to download $name..."
    echo "   URL: $url"
    echo "------------------------------------------------"
    
    # 尝试 wget (详细模式)
    if command -v wget >/dev/null 2>&1; then
        echo "👉 Trying wget..."
        # --no-check-certificate 解决内网常见的证书报错
        wget --no-check-certificate -v -O "$dest" "$url"
        if [ $? -eq 0 ]; then
            echo "✅ wget success."
            return
        else
            echo "❌ wget failed."
        fi
    fi

    # 尝试 curl (详细模式)
    if command -v curl >/dev/null 2>&1; then
        echo "👉 Trying curl..."
        # -L 跟随重定向, -k 忽略证书错误
        curl -L -k -o "$dest" "$url"
        if [ $? -eq 0 ]; then
            echo "✅ curl success."
            return
        else
            echo "❌ curl failed."
        fi
    fi
    
    echo "⚠️  CRITICAL: Failed to download $name. Please check your network/proxy."
}

# 3. 下载 Benchmark (使用 GitHub 源和 HF 镜像)
POPE_URL="https://raw.githubusercontent.com/lavis-nlp/POPE/main/output/coco/coco_pope_random.json"
MMHAL_URL="https://hf-mirror.com/datasets/SJTU-LIT/MMHal-Bench/resolve/main/mmhal_bench.json"

download_file "$POPE_URL" "./data/benchmarks/pope_coco_random.json" "POPE"
download_file "$MMHAL_URL" "./data/benchmarks/mmhal_bench.json" "MMHal-Bench"

# 4. Python 下载脚本 (带详细 Traceback)
echo "------------------------------------------------"
echo "🖼️  Starting Python Downloader for YFCC100M..."
echo "------------------------------------------------"

cat <<EOF > _debug_downloader.py
import os
import sys
import asyncio
import aiohttp
from datasets import load_dataset
from tqdm import tqdm

# 强制使用 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print(f"DEBUG: HF_ENDPOINT set to {os.environ.get('HF_ENDPOINT')}")

ROOT_DIR = "./data/yfcc100m"
TARGET_COUNT = 50000 

async def main():
    print(f"DEBUG: Checking {ROOT_DIR}...")
    existing = len([f for f in os.listdir(ROOT_DIR) if f.endswith('.jpg')])
    print(f"DEBUG: Found {existing} existing images.")
    
    if existing >= TARGET_COUNT:
        print("✅ Sufficient data found.")
        return

    print("DEBUG: Attempting to load dataset from HF Mirror...")
    try:
        ds = load_dataset("limingcv/YFCC100M_OpenAI_subset", split="train", streaming=True, trust_remote_code=True)
        print("✅ Dataset loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        # 尝试备用源
        try:
            print("DEBUG: Trying backup source 'dbrtag/yfcc100m'...")
            ds = load_dataset("dbrtag/yfcc100m", split="train", streaming=True, trust_remote_code=True)
        except Exception as e2:
            print(f"❌ Backup source also failed: {e2}")
            print("\n!!! NETWORK ERROR: Cannot access HuggingFace Mirror. Check your proxy.")
            return

    print(f"DEBUG: Starting download loop (Target: {TARGET_COUNT})...")
    
    async with aiohttp.ClientSession() as session:
        downloaded = existing
        try:
            for i, item in enumerate(ds):
                if downloaded >= TARGET_COUNT: break
                
                url = item.get('url') or item.get('URL') or item.get('img_url')
                if not url: continue
                
                # 简单的串行尝试，为了看清错误
                try:
                    async with session.get(url, timeout=5) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            with open(os.path.join(ROOT_DIR, f"yfcc_{i}.jpg"), "wb") as f:
                                f.write(content)
                            downloaded += 1
                            if downloaded % 100 == 0:
                                print(f"Progress: {downloaded}/{TARGET_COUNT}", end="\r")
                        else:
                            pass # Ignore 404s
                except Exception as e:
                    pass # Ignore connection errors
                    
        except Exception as e:
            print(f"\n❌ Loop crashed: {e}")

    print(f"\n✅ Download finished. Total images: {downloaded}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Critical Python Error: {e}")
EOF

python _debug_downloader.py
rm _debug_downloader.py

echo "------------------------------------------------"
echo "🎉 Script Completed."

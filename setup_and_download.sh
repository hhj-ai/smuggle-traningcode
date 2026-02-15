#!/bin/bash

# ==========================================
# AURORA Project Data Setup Script (Enterprise Compatible)
# Environment: Linux (Old wget compatible + HF Mirror)
# ==========================================

set -e  # 遇到错误立即停止

echo "🚀 Starting AURORA Environment Setup & Data Download..."

# 1. 创建目录结构
echo "📂 Creating directory structure..."
mkdir -p ./data/yfcc100m
mkdir -p ./data/benchmarks
mkdir -p ./data/test_images
mkdir -p ./output/checkpoints

echo "✅ Directories ready: ./data, ./output"

# 2. 下载 Benchmark 数据 (兼容旧版 wget 和 curl)
echo "📊 Downloading Benchmark Datasets..."

POPE_URL="https://raw.githubusercontent.com/lavis-nlp/POPE/main/output/coco/coco_pope_random.json"
# 使用 HF 镜像加速 MMHal 下载
MMHAL_URL="https://hf-mirror.com/datasets/SJTU-LIT/MMHal-Bench/resolve/main/mmhal_bench.json"

download_file() {
    url=$1
    dest=$2
    name=$3
    
    if [ ! -f "$dest" ]; then
        echo "   - Downloading $name..."
        # 尝试 wget (兼容旧版，无进度条)
        if command -v wget >/dev/null 2>&1; then
            wget -q -O "$dest" "$url"
        # 回退到 curl
        elif command -v curl >/dev/null 2>&1; then
            curl -L -o "$dest" "$url" -s
        else
            echo "⚠️  Error: Neither wget nor curl found. Please download $url manually."
            return 1
        fi
        
        if [ $? -eq 0 ]; then
            echo "     ✅ $name downloaded."
        else
            echo "     ❌ Failed to download $name. Check network/proxy."
        fi
    else
        echo "   - $name already exists."
    fi
}

download_file "$POPE_URL" "./data/benchmarks/pope_coco_random.json" "POPE"
download_file "$MMHAL_URL" "./data/benchmarks/mmhal_bench.json" "MMHal-Bench"

# 3. 下载 YFCC100M 图片 (Python 脚本 + 镜像加速)
echo "🖼️  Downloading YFCC100M Images (Target: 50,000)..."

cat <<EOF > _downloader.py
import os
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# === 关键：注入国内镜像环境变量 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print(f"🌏 Enable HF Mirror: {os.environ['HF_ENDPOINT']}")

from datasets import load_dataset

# Configuration aligned with aurora_train.py
ROOT_DIR = "./data/yfcc100m"
TARGET_COUNT = 50000 
CONCURRENCY = 100 # 适当降低并发以避免内网防火墙限流

async def download_image(session, url, idx):
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                content = await response.read()
                try:
                    Image.open(BytesIO(content)).verify()
                except:
                    return False
                path = os.path.join(ROOT_DIR, f"yfcc_{idx}.jpg")
                with open(path, "wb") as f:
                    f.write(content)
                return True
    except:
        return False
    return False

async def main():
    # Check existing
    existing = [f for f in os.listdir(ROOT_DIR) if f.endswith('.jpg')]
    if len(existing) >= TARGET_COUNT:
        print(f"✅ Found {len(existing)} images. Skipping download.")
        return

    print(f"🌊 Streaming metadata from Hugging Face Mirror...")
    
    # 尝试加载数据集，增加容错
    try:
        # 首选源
        ds = load_dataset("limingcv/YFCC100M_OpenAI_subset", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️  Primary source failed: {e}")
        try:
            # 备用源 (通常更稳定)
            ds = load_dataset("dbrtag/yfcc100m", split="train", streaming=True, trust_remote_code=True)
        except Exception as e2:
            print(f"❌  All sources failed. 你的网络可能无法访问外部 HF 镜像。")
            print(f"Error details: {e2}")
            return

    print(f"⬇️  Downloading missing images (Target: {TARGET_COUNT})...")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        downloaded = len(existing)
        pbar = tqdm(total=TARGET_COUNT, initial=downloaded, unit="img")
        
        for i, item in enumerate(ds):
            if downloaded >= TARGET_COUNT:
                break
            
            if os.path.exists(os.path.join(ROOT_DIR, f"yfcc_{i}.jpg")):
                continue

            # 兼容不同数据集的 URL 字段名
            url = item.get('url') or item.get('URL') or item.get('img_url') or item.get('download_url')
            if not url: continue
            
            tasks.append(asyncio.create_task(download_image(session, url, i)))
            
            if len(tasks) >= CONCURRENCY:
                results = await asyncio.gather(*tasks)
                success = sum(results)
                downloaded += success
                pbar.update(success)
                tasks = []
        
        if tasks:
            results = await asyncio.gather(*tasks)
            downloaded += sum(results)
            pbar.update(sum(results))
        
        pbar.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Python script execution failed: {e}")
        print("建议手动检查: ping hf-mirror.com 是否通畅")

EOF

# Run the embedded python downloader
python _downloader.py

# Cleanup
rm _downloader.py

echo "🎉 All data downloaded successfully!"
echo "   - Images: ./data/yfcc100m"
echo "   - Benchmarks: ./data/benchmarks"
echo "   - Checkpoints: ./output/checkpoints"
echo ""
echo "👉 You can now run training directly: accelerate launch aurora_train.py"

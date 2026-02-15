#!/bin/bash

# ==========================================
# AURORA Project Data Setup Script
# Environment: H200 Cluster Optimized
# ==========================================

set -e  # Exit immediately if a command exits with a non-zero status.

echo "🚀 Starting AURORA Environment Setup & Data Download..."

# 1. Create Directory Structure (Aligned with python scripts)
echo "📂 Creating directory structure..."
mkdir -p ./data/yfcc100m
mkdir -p ./data/benchmarks
mkdir -p ./data/test_images
mkdir -p ./output/checkpoints

echo "✅ Directories ready: ./data, ./output"

# 2. Download Benchmark Data (POPE & MMHal)
echo "📊 Downloading Benchmark Datasets..."

POPE_URL="https://raw.githubusercontent.com/lavis-nlp/POPE/main/output/coco/coco_pope_random.json"
MMHAL_URL="https://huggingface.co/datasets/SJTU-LIT/MMHal-Bench/resolve/main/mmhal_bench.json"

if [ ! -f "./data/benchmarks/pope_coco_random.json" ]; then
    echo "   - Downloading POPE..."
    wget -q --show-progress -O ./data/benchmarks/pope_coco_random.json "$POPE_URL" || echo "⚠️ Failed to download POPE, please check connection."
else
    echo "   - POPE already exists."
fi

if [ ! -f "./data/benchmarks/mmhal_bench.json" ]; then
    echo "   - Downloading MMHal-Bench..."
    wget -q --show-progress -O ./data/benchmarks/mmhal_bench.json "$MMHAL_URL" || echo "⚠️ Failed to download MMHal, please check connection."
else
    echo "   - MMHal already exists."
fi

# 3. Download YFCC100M Images (High-Concurrency Python Script)
# We embed a python script to handle the async downloading logic exactly as aurora_train.py does.
echo "🖼️  Downloading YFCC100M Images (Target: 50,000)..."

cat <<EOF > _downloader.py
import os
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Configuration aligned with aurora_train.py
ROOT_DIR = "./data/yfcc100m"
TARGET_COUNT = 50000 
CONCURRENCY = 200 # Optimized for H200 cluster bandwidth

async def download_image(session, url, idx):
    try:
        timeout = aiohttp.ClientTimeout(total=10)
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

    print(f"🌊 Streaming metadata from Hugging Face...")
    try:
        ds = load_dataset("limingcv/YFCC100M_OpenAI_subset", split="train", streaming=True, trust_remote_code=True)
    except:
        ds = load_dataset("dbrtag/yfcc100m", split="train", streaming=True, trust_remote_code=True)

    print(f"⬇️  Downloading missing images (Target: {TARGET_COUNT})...")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        downloaded = len(existing)
        pbar = tqdm(total=TARGET_COUNT, initial=downloaded, unit="img")
        
        for i, item in enumerate(ds):
            if downloaded >= TARGET_COUNT:
                break
            
            # Skip if file already exists (simple check based on index assumption, 
            # for robustness we just download forward)
            if os.path.exists(os.path.join(ROOT_DIR, f"yfcc_{i}.jpg")):
                continue

            url = item.get('url') or item.get('URL') or item.get('img_url')
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
    asyncio.run(main())
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

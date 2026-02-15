#!/bin/bash

# ==========================================
# AURORA Resource Setup (Singapore/Global)
# Downloads: Datasets AND Models (Local Dir)
# ==========================================

# 1. 初始化目录
echo "📂 Initializing directory structure..."
mkdir -p ./data/yfcc100m
mkdir -p ./data/benchmarks
mkdir -p ./data/test_images
mkdir -p ./output/checkpoints
mkdir -p ./models  # 新增：模型存放目录

# 2. Python 下载脚本 (数据 + 模型)
echo "------------------------------------------------"
echo "🚀 Starting High-Speed Downloader (Global Network)..."
echo "------------------------------------------------"

cat <<EOF > _resource_downloader.py
import os
import json
import shutil
from tqdm import tqdm
from PIL import Image
import io

# [CRITICAL] 确保直连官方
if "HF_ENDPOINT" in os.environ:
    del os.environ["HF_ENDPOINT"]
    print("🌍 Cleared HF_ENDPOINT. Using official HuggingFace servers.")

from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

# ==========================================
# Part A: Download Models (Qwen + DeepSeek)
# ==========================================
def download_models():
    print("\n🤖 [Part A] Downloading Models to ./models/ ...")
    
    # 1. VLM: Qwen3-VL-8B-Instruct
    print("   ⬇️  Downloading Qwen/Qwen3-VL-8B-Instruct...")
    try:
        snapshot_download(
            repo_id="Qwen/Qwen3-VL-8B-Instruct",
            local_dir="./models/Qwen3-VL-8B-Instruct",
            local_dir_use_symlinks=False,  # 确保是真实文件
            resume_download=True
        )
        print("   ✅ Qwen3-VL downloaded.")
    except Exception as e:
        print(f"   ❌ Qwen download failed: {e}")

    # 2. Verifier: DeepSeek-R1-Distill-Qwen-7B
    print("   ⬇️  Downloading deepseek-ai/DeepSeek-R1-Distill-Qwen-7B...")
    try:
        snapshot_download(
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            local_dir="./models/DeepSeek-R1-Distill-Qwen-7B",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("   ✅ DeepSeek-R1 downloaded.")
    except Exception as e:
        print(f"   ❌ DeepSeek download failed: {e}")

# ==========================================
# Part B: Download Benchmarks & Data
# ==========================================
def download_data():
    print("\n📊 [Part B] Downloading Datasets...")

    # Task 1: POPE
    try:
        file_path = hf_hub_download(
            repo_id="shiyue/POPE", 
            filename="coco_pope_random.json", 
            repo_type="dataset",
            local_dir="./data/benchmarks"
        )
        target = "./data/benchmarks/pope_coco_random.json"
        if os.path.abspath(file_path) != os.path.abspath(target):
            shutil.move(file_path, target)
        print("   ✅ POPE ready.")
    except Exception as e:
        print(f"   ❌ POPE failed: {e}")

    # Task 2: MMHal-Bench
    try:
        dataset = load_dataset("Shengcao1006/MMHal-Bench", split="test")
        export_data = []
        for idx, item in enumerate(tqdm(dataset, desc="   Processing MMHal")):
            entry = {
                "question_id": idx,
                "question": item.get("question", ""),
                "gt_answer": item.get("answer", ""),
                "image_id": f"mmhal_{idx}.jpg"
            }
            img = item.get("image")
            if img:
                img_path = f"./data/test_images/mmhal_{idx}.jpg"
                if not os.path.exists(img_path):
                    img.convert("RGB").save(img_path)
            export_data.append(entry)
        
        with open("./data/benchmarks/mmhal_bench.json", "w") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print("   ✅ MMHal ready.")
    except Exception as e:
        print(f"   ❌ MMHal failed: {e}")

    # Task 3: YFCC100M
    print("\n🖼️  [Part C] Downloading YFCC100M (Target: 50,000)...")
    ROOT_DIR = "./data/yfcc100m"
    if len([f for f in os.listdir(ROOT_DIR) if f.endswith('.jpg')]) >= 50000:
        print("   ✅ Sufficient images found. Skipping.")
        return

    try:
        ds = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", split="train", streaming=True)
        count = 0
        pbar = tqdm(total=50000, unit="img")
        
        for i, item in enumerate(ds):
            if count >= 50000: break
            save_path = os.path.join(ROOT_DIR, f"yfcc_{i}.jpg")
            if os.path.exists(save_path): continue
            
            try:
                img_obj = item.get("img") or item.get("image")
                if isinstance(img_obj, dict): img_obj = img_obj['bytes']
                if isinstance(img_obj, bytes): img_obj = Image.open(io.BytesIO(img_obj))
                if img_obj:
                    img_obj.convert("RGB").save(save_path, "JPEG")
                    count += 1
                    pbar.update(1)
            except: pass
        pbar.close()
        print("   ✅ YFCC Download complete.")
    except Exception as e:
        print(f"   ❌ YFCC failed: {e}")

if __name__ == "__main__":
    download_models()
    download_data()
EOF

# 3. 执行
python _resource_downloader.py
rm _resource_downloader.py

echo "------------------------------------------------"
echo "🎉 All Setup Finished!"
echo "   Models are in ./models/"
echo "   Data is in ./data/"

#!/bin/bash

# ==========================================
# AURORA Setup Script (Corrected & Verified)
# Environment: Linux Enterprise (HF Mirror + Robust Downloader)
# ==========================================

# 1. 创建目录
echo "📂 Re-checking directories..."
mkdir -p ./data/yfcc100m
mkdir -p ./data/benchmarks
mkdir -p ./data/test_images
mkdir -p ./output/checkpoints
echo "✅ Directories ready."

# 2. Python 下载脚本 (统一处理所有数据)
echo "------------------------------------------------"
echo "🚀 Starting Robust Python Downloader..."
echo "------------------------------------------------"

cat <<EOF > _final_downloader.py
import os
import json
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import io

# === 任务 1: 下载 MMHal-Bench ===
def download_mmhal():
    print("\n📊 [Task 1/3] Downloading MMHal-Bench...")
    save_path = "./data/benchmarks/mmhal_bench.json"
    
    if os.path.exists(save_path):
        print(f"   ✅ Already exists: {save_path}")
        return

    try:
        # 使用 datasets 库加载，比 wget 更稳
        # MMHal-Bench 包含图片和问题，我们这里提取问题部分存为 JSON 供 eval.py 使用
        dataset = load_dataset("Shengcao1006/MMHal-Bench", split="test")
        
        export_data = []
        print(f"   - Processing {len(dataset)} items...")
        
        for idx, item in enumerate(dataset):
            # 构建 eval.py 需要的格式
            entry = {
                "question_id": idx,
                "question": item.get("question", ""),
                "gt_answer": item.get("answer", ""),
                "image_id": f"mmhal_{idx}.jpg", # 假设图片命名
                # 保存图片以便评估使用
                "image_content": item.get("image") 
            }
            
            # 保存对应的图片到 test_images
            if entry["image_content"]:
                img_path = f"./data/test_images/mmhal_{idx}.jpg"
                if not os.path.exists(img_path):
                    entry["image_content"].save(img_path)
            
            del entry["image_content"] # JSON 不存图片对象
            export_data.append(entry)

        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Saved MMHal JSON to {save_path}")
        
    except Exception as e:
        print(f"   ❌ Failed to download MMHal: {e}")

# === 任务 2: 检查 POPE ===
def check_pope():
    print("\n📊 [Task 2/3] Checking POPE...")
    # POPE 不需要修正，之前已下载成功，这里仅做检查
    if os.path.exists("./data/benchmarks/pope_coco_random.json"):
        print("   ✅ POPE json found.")
    else:
        print("   ⚠️ POPE json missing. Please re-run if needed.")

# === 任务 3: 下载 YFCC100M (使用 dalle-mini 子集) ===
def download_yfcc():
    print("\n🖼️  [Task 3/3] Downloading YFCC100M Images (dalle-mini subset)...")
    ROOT_DIR = "./data/yfcc100m"
    TARGET_COUNT = 50000
    
    existing = len([f for f in os.listdir(ROOT_DIR) if f.endswith('.jpg')])
    if existing >= TARGET_COUNT:
        print(f"   ✅ Found {existing} images. Skipping download.")
        return

    try:
        # dalle-mini 子集包含 'img' 列（PIL对象），不需要再用 aiohttp 去爬 URL
        # 这样速度更快且不会 404
        ds = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", split="train", streaming=True, trust_remote_code=True)
        
        print(f"   - Streaming images from Hugging Face (Target: {TARGET_COUNT})...")
        count = existing
        
        pbar = tqdm(total=TARGET_COUNT, initial=count, unit="img")
        
        for i, item in enumerate(ds):
            if count >= TARGET_COUNT:
                break
                
            file_path = os.path.join(ROOT_DIR, f"yfcc_{i}.jpg")
            if os.path.exists(file_path):
                continue
            
            try:
                # 兼容性处理：不同版本的 dataset 可能列名不同
                image_obj = item.get('img') or item.get('image')
                
                if image_obj:
                    if not isinstance(image_obj, Image.Image):
                        # 如果是 bytes
                        image_obj = Image.open(io.BytesIO(image_obj))
                    
                    image_obj = image_obj.convert("RGB")
                    image_obj.save(file_path, "JPEG")
                    count += 1
                    pbar.update(1)
            except Exception as e:
                # 图片损坏或格式错误，跳过
                pass

        pbar.close()
        print(f"   ✅ YFCC Download finished. Total: {count}")
        
    except Exception as e:
        print(f"   ❌ YFCC Download failed: {e}")
        print("   Hint: 可能是网络中断或 HF 镜像访问受限。")

if __name__ == "__main__":
    download_mmhal()
    check_pope()
    download_yfcc()
EOF

# 运行 Python 脚本
python _final_downloader.py

# 清理
rm _final_downloader.py

echo "------------------------------------------------"
echo "🎉 Setup Completed."
echo "👉 You can now run: accelerate launch aurora_train.py"

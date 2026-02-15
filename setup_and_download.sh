#!/bin/bash

# ==========================================
# AURORA Resource Setup (Fixed Version)
# - Fixes POPE 404 error (Correct subpath)
# - Fixes MMHal trust_remote_code error
# - Keeps Aliyun Mirror acceleration
# ==========================================

# 1. 初始化目录
echo "📂 初始化目录结构..."
mkdir -p ./data/yfcc100m
mkdir -p ./data/benchmarks
mkdir -p ./data/test_images
mkdir -p ./output/checkpoints
mkdir -p ./models

# 2. Python 下载脚本
echo "------------------------------------------------"
echo "🚀 启动修复后的下载器..."
echo "------------------------------------------------"

cat <<EOF > _mirror_downloader.py
import os
import json
import shutil
from tqdm import tqdm
from PIL import Image
import io
import sys

# [CRITICAL] 强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

print(f"🌏 已启用镜像加速: {os.environ['HF_ENDPOINT']}")

from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

# ==========================================
# 任务 A: 下载模型
# ==========================================
def download_models():
    print("\n🤖 [任务 A] 检查模型权重...")
    
    # 1. VLM
    qwen_path = "./models/Qwen3-VL-8B-Instruct"
    if os.path.exists(os.path.join(qwen_path, "config.json")):
        print(f"   ✅ Qwen3-VL 已存在，跳过。")
    else:
        print("   ⬇️  正在下载 Qwen3-VL (可能需要几分钟)...")
        try:
            snapshot_download(
                repo_id="Qwen/Qwen3-VL-8B-Instruct",
                local_dir=qwen_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=8
            )
            print("   ✅ Qwen3-VL 下载完成。")
        except Exception as e:
            print(f"   ❌ Qwen 下载失败: {e}")

    # 2. Verifier
    ds_path = "./models/DeepSeek-R1-Distill-Qwen-7B"
    if os.path.exists(os.path.join(ds_path, "config.json")):
        print(f"   ✅ DeepSeek-R1 已存在，跳过。")
    else:
        print("   ⬇️  正在下载 DeepSeek-R1...")
        try:
            snapshot_download(
                repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                local_dir=ds_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=8
            )
            print("   ✅ DeepSeek-R1 下载完成。")
        except Exception as e:
            print(f"   ❌ DeepSeek 下载失败: {e}")

# ==========================================
# 任务 B: 下载数据 (修复版)
# ==========================================
def download_data():
    print("\n📊 [任务 B] 下载测试数据集...")

    # Task 1: POPE (修复路径错误)
    pope_target = "./data/benchmarks/pope_coco_random.json"
    if os.path.exists(pope_target):
        print("   ✅ POPE 数据已存在，跳过。")
    else:
        try:
            print("   ⬇️  下载 POPE (shiyue/POPE)...")
            # 修正：文件在 repo 的 output/coco/ 子目录下，不是根目录
            file_path = hf_hub_download(
                repo_id="shiyue/POPE", 
                filename="output/coco/coco_pope_random.json", # <--- 修复点
                repo_type="dataset",
                local_dir="./data/benchmarks"
            )
            # 移动并重命名为标准文件名
            if os.path.exists(file_path):
                shutil.move(file_path, pope_target)
                # 清理 hf 下载生成的空目录结构
                shutil.rmtree("./data/benchmarks/output", ignore_errors=True)
            print("   ✅ POPE 准备就绪。")
        except Exception as e:
            print(f"   ❌ POPE 失败: {e}")
            print("   (如果依然失败，eval.py 会跳过此项，不影响训练)")

    # Task 2: MMHal-Bench (修复 trust_remote_code)
    mmhal_target = "./data/benchmarks/mmhal_bench.json"
    if os.path.exists(mmhal_target):
         print("   ✅ MMHal-Bench 数据已存在，跳过。")
    else:
        try:
            print("   ⬇️  处理 MMHal-Bench (含 trust_remote_code=True)...")
            # 修正：添加 trust_remote_code=True
            dataset = load_dataset(
                "Shengcao1006/MMHal-Bench", 
                split="test", 
                trust_remote_code=True  # <--- 修复点：允许执行自定义代码
            )
            
            export_data = []
            for idx, item in enumerate(tqdm(dataset, desc="   导出 MMHal 图片")):
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
            
            with open(mmhal_target, "w", encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print("   ✅ MMHal 准备就绪。")
        except Exception as e:
            print(f"   ❌ MMHal 失败: {e}")

    # Task 3: YFCC100M
    print("\n🖼️  [任务 C] 检查 YFCC100M...")
    ROOT_DIR = "./data/yfcc100m"
    existing = len([f for f in os.listdir(ROOT_DIR) if f.endswith('.jpg')])
    
    if existing >= 50000:
        print(f"   ✅ 已存在 {existing} 张图片，跳过。")
        return

    try:
        ds = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", split="train", streaming=True)
        count = existing
        pbar = tqdm(total=50000, initial=count, unit="img")
        
        for i, item in enumerate(ds):
            if count >= 50000: break
            save_path = os.path.join(ROOT_DIR, f"yfcc_{i}.jpg")
            if os.path.exists(save_path): continue
            
            try:
                img_obj = item.get("img") or item.get("image")
                if isinstance(img_obj, dict) and 'bytes' in img_obj: img_obj = img_obj['bytes']
                if isinstance(img_obj, bytes): img_obj = Image.open(io.BytesIO(img_obj))
                if img_obj:
                    img_obj.convert("RGB").save(save_path, "JPEG")
                    count += 1
                    pbar.update(1)
            except: pass
                
        pbar.close()
        print(f"   ✅ YFCC 下载完成。")
    except Exception as e:
        print(f"   ❌ YFCC 下载中断: {e}")

if __name__ == "__main__":
    download_models()
    download_data()
EOF

# 3. 执行
python _mirror_downloader.py
rm _mirror_downloader.py

echo "------------------------------------------------"
echo "🎉 修复完成！请继续。"

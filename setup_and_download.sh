#!/bin/bash

# ==========================================
# AURORA Resource Setup (Aliyun/CN Mirror Optimized)
# Downloads: Datasets AND Models (Local Dir)
# ==========================================

# 1. 初始化目录
echo "📂 初始化目录结构..."
mkdir -p ./data/yfcc100m
mkdir -p ./data/benchmarks
mkdir -p ./data/test_images
mkdir -p ./output/checkpoints
mkdir -p ./models

# 2. Python 下载脚本 (内置 HF 镜像加速)
echo "------------------------------------------------"
echo "🚀 启动高速下载器 (使用 hf-mirror.com)..."
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
# 开启 HF 专用传输加速 (如果环境支持)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

print(f"🌏 已启用镜像加速: {os.environ['HF_ENDPOINT']}")

from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset

# ==========================================
# 任务 A: 下载模型 (Qwen + DeepSeek)
# ==========================================
def download_models():
    print("\n🤖 [任务 A] 下载模型权重到 ./models/ ...")
    
    # 1. VLM: Qwen3-VL-8B-Instruct
    print("   ⬇️  正在下载 Qwen/Qwen3-VL-8B-Instruct (可能需要几分钟)...")
    try:
        snapshot_download(
            repo_id="Qwen/Qwen3-VL-8B-Instruct",
            local_dir="./models/Qwen3-VL-8B-Instruct",
            local_dir_use_symlinks=False,  # 确保下载的是真实文件，不是软链接
            resume_download=True,
            max_workers=8  # 阿里云带宽通常较大，开多线程
        )
        print("   ✅ Qwen3-VL 下载完成。")
    except Exception as e:
        print(f"   ❌ Qwen 下载失败: {e}")

    # 2. Verifier: DeepSeek-R1-Distill-Qwen-7B
    print("   ⬇️  正在下载 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B...")
    try:
        snapshot_download(
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            local_dir="./models/DeepSeek-R1-Distill-Qwen-7B",
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8
        )
        print("   ✅ DeepSeek-R1 下载完成。")
    except Exception as e:
        print(f"   ❌ DeepSeek 下载失败: {e}")

# ==========================================
# 任务 B: 下载 Benchmark 数据
# ==========================================
def download_data():
    print("\n📊 [任务 B] 下载测试数据集...")

    # Task 1: POPE (从 hf-mirror 拉取)
    try:
        print("   ⬇️  下载 POPE...")
        file_path = hf_hub_download(
            repo_id="shiyue/POPE", 
            filename="coco_pope_random.json", 
            repo_type="dataset",
            local_dir="./data/benchmarks"
        )
        target = "./data/benchmarks/pope_coco_random.json"
        # 修正路径
        if os.path.abspath(file_path) != os.path.abspath(target):
            shutil.move(file_path, target)
        print("   ✅ POPE 准备就绪。")
    except Exception as e:
        print(f"   ❌ POPE 失败: {e}")

    # Task 2: MMHal-Bench (从 hf-mirror 加载并导出)
    try:
        print("   ⬇️  处理 MMHal-Bench...")
        dataset = load_dataset("Shengcao1006/MMHal-Bench", split="test")
        export_data = []
        
        # 阿里云服务器通常可以直接处理图片对象
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
        
        with open("./data/benchmarks/mmhal_bench.json", "w", encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print("   ✅ MMHal 准备就绪。")
    except Exception as e:
        print(f"   ❌ MMHal 失败: {e}")

    # Task 3: YFCC100M (使用镜像源流式下载)
    print("\n🖼️  [任务 C] 下载 YFCC100M 训练图 (目标: 50,000 张)...")
    ROOT_DIR = "./data/yfcc100m"
    existing = len([f for f in os.listdir(ROOT_DIR) if f.endswith('.jpg')])
    
    if existing >= 50000:
        print(f"   ✅ 已存在 {existing} 张图片，跳过下载。")
        return

    try:
        # dalle-mini 子集在镜像站通常有缓存，速度快
        ds = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", split="train", streaming=True)
        count = existing
        pbar = tqdm(total=50000, initial=count, unit="img")
        
        for i, item in enumerate(ds):
            if count >= 50000: break
            
            save_path = os.path.join(ROOT_DIR, f"yfcc_{i}.jpg")
            if os.path.exists(save_path): continue
            
            try:
                img_obj = item.get("img") or item.get("image")
                # 处理 bytes 类型
                if isinstance(img_obj, dict) and 'bytes' in img_obj: 
                     img_obj = img_obj['bytes']
                if isinstance(img_obj, bytes): 
                    img_obj = Image.open(io.BytesIO(img_obj))
                
                if img_obj:
                    img_obj.convert("RGB").save(save_path, "JPEG")
                    count += 1
                    pbar.update(1)
            except Exception: 
                pass # 忽略损坏图片
                
        pbar.close()
        print(f"   ✅ YFCC 下载完成，共 {count} 张。")
    except Exception as e:
        print(f"   ❌ YFCC 下载中断: {e}")

if __name__ == "__main__":
    download_models()
    download_data()
EOF

# 3. 执行 Python 脚本
python _mirror_downloader.py

# 4. 清理临时文件
rm _mirror_downloader.py

echo "------------------------------------------------"
echo "🎉 环境准备完成！"
echo "   - 模型路径: ./models/"
echo "   - 数据路径: ./data/"
echo "👉 现在运行: accelerate launch aurora_train.py"

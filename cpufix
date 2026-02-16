import os
from huggingface_hub import hf_hub_download

# ==============================================================================
# 策略调整：放弃 offline_extra，改用 offline_fix 避开权限锁
# ==============================================================================
BASE_DIR = "offline_fix"

tasks = [
    {
        "repo_id": "openai/clip-vit-base-patch32",
        "local_dir": f"{BASE_DIR}/clip",
        # 只下载 PyTorch 权重，跳过 1GB+ 的垃圾文件
        "files": ["config.json", "pytorch_model.bin", "preprocessor_config.json", 
                  "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    },
    {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "local_dir": f"{BASE_DIR}/minilm",
        "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json", 
                  "vocab.txt", "special_tokens_map.json", "modules.json", "sentence_bert_config.json"]
    }
]

print(f"🚀 开始下载到新目录: {BASE_DIR} ...")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

for task in tasks:
    print(f"\n⬇️  正在处理: {task['repo_id']}")
    os.makedirs(task['local_dir'], exist_ok=True)
    
    for filename in task['files']:
        # 检查文件是否已存在且大小正常（断点续传简单版）
        filepath = os.path.join(task['local_dir'], filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"   -> {filename} 已存在，跳过。")
            continue

        print(f"   -> 下载 {filename}...", end="")
        try:
            hf_hub_download(
                repo_id=task['repo_id'],
                filename=filename,
                local_dir=task['local_dir'],
                local_dir_use_symlinks=False, # 关键：禁用软链
                resume_download=True
            )
            print(" ✅")
        except Exception as e:
            print(f" ❌ 失败: {e}")

print(f"\n🎉 下载完成！请将 '{BASE_DIR}' 文件夹上传到 GPU 服务器。")

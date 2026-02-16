import os
from huggingface_hub import hf_hub_download

# 定义要下载的两个模型 (只下核心文件)
tasks = [
    {
        "repo_id": "openai/clip-vit-base-patch32",
        "local_dir": "offline_extra/clip",
        # 排除 flaz_model.msgpack 和 tf_model.h5，只下 pytorch_model.bin
        "files": ["config.json", "pytorch_model.bin", "preprocessor_config.json", 
                  "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    },
    {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "local_dir": "offline_extra/minilm",
        "files": ["config.json", "pytorch_model.bin", "tokenizer_config.json", 
                  "vocab.txt", "special_tokens_map.json", "modules.json", "sentence_bert_config.json"]
    }
]

print("🚀 开始精准下载 (仅 PyTorch 权重)...")
# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

for task in tasks:
    print(f"\n⬇️  正在处理: {task['repo_id']}")
    os.makedirs(task['local_dir'], exist_ok=True)
    
    for filename in task['files']:
        print(f"   -> 下载 {filename}...", end="")
        try:
            hf_hub_download(
                repo_id=task['repo_id'],
                filename=filename,
                local_dir=task['local_dir'],
                local_dir_use_symlinks=False,  # 关键：不用软链接，避免权限问题
                resume_download=True
            )
            print(" ✅")
        except Exception as e:
            print(f" ❌ 失败: {e}")

print("\n🎉 下载完成！请将 offline_extra 文件夹传到 GPU 服务器。")

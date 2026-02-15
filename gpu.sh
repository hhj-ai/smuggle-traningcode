#!/bin/bash

# ========================================================================
# 2_full_install.sh (GPU 服务器 - 自动部署与还原)
# 目标: 安装Python3.10环境 + 归位所有数据/权重 + 还原 models.py
# ========================================================================

BASE_DIR="./offline_packages"
PYTHON_TGZ="$BASE_DIR/python_runtime/python-3.10.tar.gz"
WHEEL_DIR="$BASE_DIR/wheels"
DATA_SOURCE="$BASE_DIR/datasets"
WEIGHTS_SRC="$BASE_DIR/tool_weights"
INSTALL_ROOT="./aurora_env_root"
VENV_DIR="aurora_env"

echo "🚀 [GPU Server] 开始全离线部署..."

# 1. 部署 Python 3.10
if [ ! -f "$PYTHON_TGZ" ]; then echo "❌ 错误: 未找到 $PYTHON_TGZ"; exit 1; fi
echo "🐍 [1/8] 部署独立 Python 3.10..."
rm -rf $INSTALL_ROOT; mkdir -p $INSTALL_ROOT
tar -xzf $PYTHON_TGZ -C $INSTALL_ROOT
# 查找 python 可执行文件路径
if [ -d "$INSTALL_ROOT/python" ]; then LOCAL_PYTHON="$INSTALL_ROOT/python/bin/python3"; else LOCAL_PYTHON="$INSTALL_ROOT/bin/python3"; fi
echo "   -> Python Path: $LOCAL_PYTHON"

# 2. 创建虚拟环境
echo "📦 [2/8] 创建虚拟环境..."
rm -rf $VENV_DIR
$LOCAL_PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate
if [[ "$(which python)" != *"$VENV_DIR"* ]]; then echo "❌ 激活失败"; exit 1; fi

install_pkg() { pip install "$@" --no-index --find-links=$WHEEL_DIR; }

# 3. 安装基础 & PyTorch
echo "🔥 [3/8] 安装 PyTorch & Base..."
install_pkg --upgrade pip setuptools wheel ninja packaging psutil numpy
install_pkg torch torchvision torchaudio

# 4. 编译 Flash Attention
echo "⚡ [4/8] 编译 Flash Attention (耐心等待)..."
install_pkg flash-attn --no-build-isolation

# 5. 安装 Transformers (Zip) & 依赖
echo "🤗 [5/8] 安装 Transformers..."
if [ -f "$WHEEL_DIR/transformers-main.zip" ]; then
    unzip -q -o "$WHEEL_DIR/transformers-main.zip" -d ./temp_transformers
    pip install ./temp_transformers/transformers-main --no-index --find-links=$WHEEL_DIR
    rm -rf ./temp_transformers
else
    install_pkg transformers
fi
install_pkg accelerate huggingface_hub datasets sentence-transformers Pillow easyocr scipy termcolor timm rich questionary aiohttp protobuf sentencepiece pandas

# 6. 归位评测数据
echo "📊 [6/8] 部署评测数据..."
DEST_BENCH="./data/benchmarks"
mkdir -p $DEST_BENCH

# POPE
[ -f "$DATA_SOURCE/pope/coco_pope_random.json" ] && cp "$DATA_SOURCE/pope/coco_pope_random.json" "$DEST_BENCH/pope_coco_random.json"

# MMHal (Copy + Extract Images)
if [ -d "$DATA_SOURCE/mmhal_bench" ]; then
    cp -r "$DATA_SOURCE/mmhal_bench" "$DEST_BENCH/"
    echo "   ⚙️ 提取 MMHal 图片..."
    cat <<EOF > _extract_mmhal.py
import os, json
from datasets import load_dataset
try:
    ds = load_dataset("$DEST_BENCH/mmhal_bench", split="test")
    img_dir = "./data/test_images"; os.makedirs(img_dir, exist_ok=True)
    data = []
    for i, item in enumerate(ds):
        if item.get("image"): item["image"].convert("RGB").save(f"{img_dir}/mmhal_{i}.jpg")
        data.append({"question_id": i, "question": item.get("question"), "gt_answer": item.get("answer"), "image_id": f"mmhal_{i}.jpg"})
    with open("$DEST_BENCH/mmhal_bench.json", "w") as f: json.dump(data, f, indent=2)
    print("   ✅ MMHal extracted.")
except Exception as e: print(f"   ❌ MMHal error: {e}")
EOF
    python _extract_mmhal.py; rm _extract_mmhal.py
fi

# 7. 归位工具权重
echo "🛠️  [7/8] 部署工具权重..."
# EasyOCR
OCR_DEST="$HOME/.EasyOCR/model"; mkdir -p "$OCR_DEST"
cp "$WEIGHTS_SRC/easyocr/"*.pth "$OCR_DEST/" 2>/dev/null

# DINO
DINO_DEST="./weights"; mkdir -p "$DINO_DEST"
cp "$WEIGHTS_SRC/dino/"* "$DINO_DEST/" 2>/dev/null
# Sentence Transformers
ST_DEST="$HOME/.cache/torch/sentence_transformers"; mkdir -p "$ST_DEST"
[ -d "$WEIGHTS_SRC/sentence-transformers/all-MiniLM-L6-v2" ] && cp -r "$WEIGHTS_SRC/sentence-transformers/all-MiniLM-L6-v2" "$ST_DEST/sentence-transformers_all-MiniLM-L6-v2"

# 8. 还原干净的 models.py
echo "🧹 [8/8] 还原标准版 models.py (Qwen3原生支持)..."
cat <<EOF > models.py
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

class VerifierModel:
    def __init__(self, model_name="./models/DeepSeek-R1-Distill-Qwen-7B", device="cuda"):
        self.device = device
        if not os.path.exists(model_name) and "models/" in model_name:
             model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        print(f"Loading Verifier: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map={"": device}, 
            trust_remote_code=True, 
            attn_implementation="flash_attention_2"
        )

    def verify_claims(self, description):
        prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\\n\\nDescription: {description}\\n\\nClaims:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6)
        raw = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        claims = [line.strip().lstrip('-*').strip() for line in clean.split('\\n') if len(line.strip()) > 5]
        return claims, raw

    def compute_sequence_log_prob(self, prompt, completion):
        full = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\\n\\nDescription: {prompt}\\n\\nClaims:" + completion
        inputs = self.tokenizer(full, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        prompt_len = self.tokenizer(f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\\n\\nDescription: {prompt}\\n\\nClaims:", return_tensors="pt").input_ids.shape[1]
        labels[:, :min(prompt_len, labels.shape[1])] = -100
        outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
        return -outputs.loss * (labels != -100).sum()

class VLMModel:
    def __init__(self, model_name="./models/Qwen3-VL-8B-Instruct", device="cuda"):
        self.device = device
        if not os.path.exists(model_name) and "models/" in model_name:
             model_name = "Qwen/Qwen3-VL-8B-Instruct"
        print(f"Loading VLM: {model_name}")
        # 最新版 transformers 原生支持 Qwen3-VL，无需任何 patch
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = self.processor.tokenizer

    def generate_description_batch(self, image_inputs, num_generations=4):
        # Qwen3-VL 官方推荐 prompt
        prompts = ["Describe this image in detail."] * len(image_inputs)
        inputs = self.processor(text=prompts, images=image_inputs, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=1.0, num_return_sequences=num_generations)
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [texts[i*num_generations : (i+1)*num_generations] for i in range(len(image_inputs))]

    def compute_log_probs(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return -outputs.loss * (labels != -100).sum()
EOF

echo "------------------------------------------------"
echo "🎉 部署完成！环境、数据、权重、代码均已就绪。"
echo "👉 启动命令: source $VENV_DIR/bin/activate"

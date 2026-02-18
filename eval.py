import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import argparse
from tqdm import tqdm
import json
import requests
import sys

# Configuration
DEFAULT_TRAINED_PATH = "./output/checkpoints/vlm_final"
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct" # 基座模型 ID
TEST_IMAGE_DIR = "./data/test_images"
BENCHMARK_DIR = "./data/benchmarks"

# Auto-Download URLs for Benchmarks
POPE_URL = "https://huggingface.co/datasets/shiyue/POPE/resolve/main/coco_pope_random.json"
MMHAL_URL = "https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/main/mmhal_bench.json"

POPE_DATA_PATH = os.path.join(BENCHMARK_DIR, "pope_coco_random.json")
MMHAL_DATA_PATH = os.path.join(BENCHMARK_DIR, "mmhal_bench.json")

def download_file(url, save_path):
    """Downloads a file from a URL if it doesn't exist."""
    if os.path.exists(save_path):
        return True
    
    print(f"⬇️  Downloading benchmark data to {save_path}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 简单的下载逻辑，实际环境中建议用 setup_force_hf.sh 提前下好
        response = requests.get(url, timeout=20) 
        if response.status_code != 200:
            print(f"⚠️  Direct download failed ({response.status_code}). Please use setup_force_hf.sh.")
            return False
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved.")
        return True
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return False

def load_model(model_path):
    print(f"Loading model from: {model_path} ...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Use Qwen3VL-specific class if available, fallback to AutoModelForCausalLM
        model = None
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
        except (ImportError, Exception):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )

        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

class POPEEvaluator:
    def __init__(self, data_path, image_dir):
        self.data_path = data_path
        self.image_dir = image_dir
        self.data = []
        
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = [json.loads(line) for line in f]
        else:
            print(f"⚠️ POPE data missing at {data_path}")

    def evaluate(self, model, processor, prefix=""):
        if not self.data: return
        print(f"📊 Running POPE Benchmark ({len(self.data)} samples)...")
        
        tp, tn, fp, fn = 0, 0, 0, 0
        missing_images = 0
        
        # Limit to 300 samples for speed during dev, remove slice for full run
        eval_data = self.data if len(self.data) < 500 else self.data[:300]
        
        for item in tqdm(eval_data):
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(self.image_dir, img_name)
            
            # 尝试在 YFCC 目录找找（如果用户把 COCO 图放那里了）
            if not os.path.exists(img_path):
                 yfcc_path = os.path.join("./data/yfcc100m", img_name)
                 if os.path.exists(yfcc_path):
                     img_path = yfcc_path
                 else:
                    missing_images += 1
                    continue
            
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue

            prompt = f"{item['question']} Answer Yes or No."
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
                ans = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).lower()
            
            pred_yes = "yes" in ans
            gt_yes = item['answer'].lower() == "yes"
            
            if gt_yes and pred_yes: tp += 1
            elif not gt_yes and not pred_yes: tn += 1
            elif not gt_yes and pred_yes: fp += 1
            elif gt_yes and not pred_yes: fn += 1
            
        if missing_images > 0:
            print(f"⚠️ Skipped {missing_images} samples (missing images).")

        total_evaluated = len(eval_data) - missing_images
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        acc = (tp + tn) / (total_evaluated + 1e-6)
        fpr = fp / (fp + tn + 1e-6)
        yes_ratio = (tp + fp) / (total_evaluated + 1e-6)

        print(f"[POPE] Acc: {acc:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}, "
              f"FPR: {fpr:.2f}, Yes-Ratio: {yes_ratio:.2f}")

        # Save result summary
        res_file = f"{prefix}pope_score.json"
        with open(res_file, "w") as f:
            json.dump({"accuracy": acc, "f1": f1, "precision": precision,
                       "fpr": fpr, "yes_ratio": yes_ratio,
                       "tp": tp, "tn": tn, "fp": fp, "fn": fn}, f)
        print(f"POPE scores saved to {res_file}")

class MMHalEvaluator:
    def __init__(self, data_path, image_dir):
        self.data_path = data_path
        self.image_dir = image_dir
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            print(f"⚠️ MMHal data missing at {data_path}")

    def evaluate(self, model, processor, prefix=""):
        if not self.data: return
        print(f"📊 Running MMHal-Bench ({len(self.data)} samples)...")
        
        results = []
        missing_images = 0
        
        # Limit for speed
        eval_data = self.data if len(self.data) < 100 else self.data[:50]
        
        for item in tqdm(eval_data):
            img_path = os.path.join(self.image_dir, item.get('image_id', 'unknown')) 
            if not os.path.exists(img_path): 
                missing_images += 1
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item['question']}
                ]}
            ]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
                response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "question_id": item.get('question_id'),
                "response": response,
                "gt": item.get('gt_answer', '')
            })
        
        # Simple keyword-matching accuracy against gt_answer
        correct = 0
        total_with_gt = 0
        for r in results:
            gt = r.get("gt", "").strip().lower()
            if not gt:
                continue
            total_with_gt += 1
            response_lower = r["response"].strip().lower()
            # Check if any keyword from the ground truth appears in the response
            gt_keywords = [w for w in gt.split() if len(w) > 2]
            if gt_keywords and any(kw in response_lower for kw in gt_keywords):
                correct += 1

        keyword_acc = correct / (total_with_gt + 1e-6)

        out_file = f"{prefix}mmhal_results.json"
        summary = {
            "keyword_accuracy": keyword_acc,
            "correct": correct,
            "total_with_gt": total_with_gt,
            "missing_images": missing_images,
            "predictions": results,
        }
        print(f"[MMHal] Keyword Acc: {keyword_acc:.2f} ({correct}/{total_with_gt}). "
              f"Skipped {missing_images} images.")
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"MMHal results saved to '{out_file}'.")

def evaluate(model_path, image_dir, run_benchmarks, prefix=""):
    model, processor = load_model(model_path)
    if not model:
        return

    # 1. Standard Description Evaluation
    if not os.path.exists(image_dir):
        print(f"Test directory {image_dir} not found. Creating it.")
        os.makedirs(image_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    if image_files:
        print(f"Starting standard evaluation on {len(image_files)} images...")
        results = []
        # Randomly sample 5 images if too many
        import random
        sample_files = image_files if len(image_files) <= 5 else random.sample(image_files, 5)
        
        for img_file in tqdm(sample_files):
            img_path = os.path.join(image_dir, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
            except: continue
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail."}
                ]}
            ]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_text = processor.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            print(f"\n[Image: {img_file}]\nDescription: {generated_text}")
            results.append({"image": img_file, "description": generated_text})
        
        out_txt = f"{prefix}evaluation_results.txt"
        with open(out_txt, "w") as f:
            for res in results:
                f.write(f"Image: {res['image']}\nDescription: {res['description']}\n\n")
        print(f"📄 Standard results saved to {out_txt}")

    # 2. Run Benchmarks if requested
    if run_benchmarks:
        # Check files
        if not os.path.exists(POPE_DATA_PATH):
            print("⚠️ POPE JSON not found. Run setup_force_hf.sh first.")
        
        pope = POPEEvaluator(POPE_DATA_PATH, image_dir) 
        pope.evaluate(model, processor, prefix=prefix)
        
        mmhal = MMHalEvaluator(MMHAL_DATA_PATH, image_dir)
        mmhal.evaluate(model, processor, prefix=prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_TRAINED_PATH, help="Path to trained model")
    parser.add_argument("--image_dir", type=str, default=TEST_IMAGE_DIR, help="Directory containing test images")
    parser.add_argument("--benchmarks", action="store_true", help="Run POPE and MMHal benchmarks")
    # 新增基座测试开关
    parser.add_argument("--baseline", action="store_true", help="Evaluate the Base Model (Qwen3-VL) instead of trained checkpoint")
    
    args = parser.parse_args()
    
    target_model = args.model_path
    output_prefix = ""
    
    if args.baseline:
        print("📉 MODE: Baseline Evaluation")
        print(f"   Target: {BASE_MODEL_ID}")
        target_model = BASE_MODEL_ID
        output_prefix = "baseline_"
    else:
        print("📈 MODE: Trained Model Evaluation")
        print(f"   Target: {target_model}")
    
    evaluate(target_model, args.image_dir, args.benchmarks, prefix=output_prefix)

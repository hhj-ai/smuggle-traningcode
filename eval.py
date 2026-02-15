import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import argparse
from tqdm import tqdm
import json
import random
import requests

# Configuration
DEFAULT_MODEL_PATH = "./output/checkpoints/vlm_final"
TEST_IMAGE_DIR = "./data/test_images"
BENCHMARK_DIR = "./data/benchmarks"

# Auto-Download URLs for Benchmarks
POPE_URL = "https://raw.githubusercontent.com/lavis-nlp/POPE/main/output/coco/coco_pope_random.json"
MMHAL_URL = "https://huggingface.co/datasets/SJTU-LIT/MMHal-Bench/resolve/main/mmhal_bench.json"

POPE_DATA_PATH = os.path.join(BENCHMARK_DIR, "pope_coco_random.json")
MMHAL_DATA_PATH = os.path.join(BENCHMARK_DIR, "mmhal_bench.json")

def download_file(url, save_path):
    """Downloads a file from a URL if it doesn't exist."""
    if os.path.exists(save_path):
        return True
    
    print(f"⬇️  Downloading benchmark data from {url}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved to {save_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

class POPEEvaluator:
    def __init__(self, data_path, image_dir):
        self.data_path = data_path
        self.image_dir = image_dir
        
        # ✅ Auto-download POPE if missing
        if not os.path.exists(data_path):
             download_file(POPE_URL, data_path)

        self.data = []
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = [json.loads(line) for line in f]
        else:
            print(f"⚠️ POPE data missing. Skipping.")

    def evaluate(self, model, processor):
        if not self.data: return
        print(f"📊 Running POPE Benchmark ({len(self.data)} samples)...")
        
        tp, tn, fp, fn = 0, 0, 0, 0
        missing_images = 0
        
        # Limit to first 200 for quick check if running on limited data
        for item in tqdm(self.data[:200]):
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(self.image_dir, img_name)
            
            if not os.path.exists(img_path):
                missing_images += 1
                continue
            
            image = Image.open(img_path).convert("RGB")
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
            print(f"⚠️ Skipped {missing_images} samples due to missing images (POPE requires COCO val2014).")

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        acc = (tp + tn) / (len(self.data) - missing_images + 1e-6)
        
        print(f"📈 POPE Results: Acc: {acc:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}")
        return {"accuracy": acc, "f1": f1}

class MMHalEvaluator:
    def __init__(self, data_path, image_dir):
        self.data_path = data_path
        self.image_dir = image_dir
        
        # ✅ Auto-download MMHal if missing
        if not os.path.exists(data_path):
             download_file(MMHAL_URL, data_path)

        self.data = []
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                try:
                    self.data = json.load(f)
                except:
                    # Handle jsonl case if URL changed
                    f.seek(0)
                    self.data = [json.loads(line) for line in f]
        else:
            print(f"⚠️ MMHal data missing. Skipping.")

    def evaluate(self, model, processor):
        if not self.data: return
        print(f"📊 Running MMHal-Bench ({len(self.data)} samples)...")
        
        results = []
        missing_images = 0
        
        for item in tqdm(self.data[:50]): # Limit for speed
            # MMHal often needs internet images or local files.
            # Assuming images are in image_dir for now.
            img_path = os.path.join(self.image_dir, item.get('image_id', 'unknown'))
            if not os.path.exists(img_path):
                img_path += ".jpg"
                if not os.path.exists(img_path):
                    missing_images += 1
                    continue

            image = Image.open(img_path).convert("RGB")
            
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
            
        print(f"✅ MMHal results saved to 'mmhal_results.json'. (Skipped {missing_images} missing images)")
        with open("mmhal_results.json", "w") as f:
            json.dump(results, f, indent=2)

def evaluate(model_path=DEFAULT_MODEL_PATH, image_dir=TEST_IMAGE_DIR, run_benchmarks=False):
    model, processor = load_model(model_path)
    if not model:
        return

    # 1. Standard Description Evaluation
    if not os.path.exists(image_dir):
        print(f"Test directory {image_dir} not found.")
        os.makedirs(image_dir, exist_ok=True)
        print("Created empty test directory.")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    if image_files:
        print(f"Starting standard evaluation on {len(image_files)} images...")
        results = []
        for img_file in tqdm(image_files[:5]):
            img_path = os.path.join(image_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            
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
        
        with open("evaluation_results.txt", "w") as f:
            for res in results:
                f.write(f"Image: {res['image']}\nDescription: {res['description']}\n\n")

    # 2. Run Benchmarks if requested
    if run_benchmarks:
        # Note: benchmarks need images (COCO etc).
        # If user has NO data, this will download JSONs but might skip images.
        pope = POPEEvaluator(POPE_DATA_PATH, image_dir)
        pope.evaluate(model, processor)
        
        mmhal = MMHalEvaluator(MMHAL_DATA_PATH, image_dir)
        mmhal.evaluate(model, processor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model")
    parser.add_argument("--image_dir", type=str, default=TEST_IMAGE_DIR, help="Directory containing test images")
    parser.add_argument("--benchmarks", action="store_true", help="Run POPE and MMHal benchmarks")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.image_dir, args.benchmarks)

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import argparse
from tqdm import tqdm
import json
import random

# Configuration
DEFAULT_MODEL_PATH = "./output/checkpoints/vlm_final"
TEST_IMAGE_DIR = "./data/test_images"
POPE_DATA_PATH = "./data/benchmarks/pope_coco_random.json"
MMHAL_DATA_PATH = "./data/benchmarks/mmhal_bench.json"

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
    """
    Evaluator for POPE (Polling Object Probing Evaluation).
    Expects JSON format: [{"image": "val2014/COCO_val2014_000000xxxx.jpg", "question": "...", "answer": "yes"}, ...]
    """
    def __init__(self, data_path, image_dir):
        self.data_path = data_path
        self.image_dir = image_dir
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = [json.loads(line) for line in f] # or json.load(f) depending on format
        else:
            print(f"⚠️ POPE data not found at {data_path}")

    def evaluate(self, model, processor):
        if not self.data: return
        print(f"📊 Running POPE Benchmark ({len(self.data)} samples)...")
        
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for item in tqdm(self.data):
            # Assumes images are in a subdirectory structure or flat
            # Modify logic to find your specific images
            img_name = os.path.basename(item['image'])
            img_path = os.path.join(self.image_dir, img_name)
            
            if not os.path.exists(img_path): continue
            
            image = Image.open(img_path).convert("RGB")
            prompt = f"{item['question']} Answer Yes or No."
            
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10)
                ans = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).lower()
            
            pred_yes = "yes" in ans
            gt_yes = item['answer'].lower() == "yes"
            
            if gt_yes and pred_yes: tp += 1
            elif not gt_yes and not pred_yes: tn += 1
            elif not gt_yes and pred_yes: fp += 1
            elif gt_yes and not pred_yes: fn += 1
            
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        acc = (tp + tn) / (len(self.data) + 1e-6)
        
        print(f"📈 POPE Results: Acc: {acc:.2f}, F1: {f1:.2f}, Precision: {precision:.2f}")
        return {"accuracy": acc, "f1": f1}

class MMHalEvaluator:
    """
    Evaluator for MMHal-Bench.
    """
    def __init__(self, data_path, image_dir):
        self.data_path = data_path
        self.image_dir = image_dir
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            print(f"⚠️ MMHal data not found at {data_path}")

    def evaluate(self, model, processor):
        if not self.data: return
        print(f"📊 Running MMHal-Bench ({len(self.data)} samples)...")
        
        results = []
        for item in tqdm(self.data):
            img_path = os.path.join(self.image_dir, item['image_id']) # Assuming filename matches ID
            if not os.path.exists(img_path):
                # Try adding extension
                img_path += ".jpg"
                if not os.path.exists(img_path): continue

            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=item['question'], images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256)
                response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "question_id": item['question_id'],
                "response": response,
                "gt": item.get('gt_answer', '')
            })
            
        # Save for GPT-4 Judging
        with open("mmhal_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("✅ MMHal results saved to 'mmhal_results.json'. Use OpenAI API to calculate score.")


def evaluate(model_path=DEFAULT_MODEL_PATH, image_dir=TEST_IMAGE_DIR, run_benchmarks=False):
    model, processor = load_model(model_path)
    if not model:
        return

    # 1. Standard Description Evaluation
    if not os.path.exists(image_dir):
        print(f"Test directory {image_dir} not found.")
    else:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        if image_files:
            print(f"Starting standard evaluation on {len(image_files)} images...")
            results = []
            for img_file in tqdm(image_files[:5]): # Limit to 5 for quick check
                img_path = os.path.join(image_dir, img_file)
                image = Image.open(img_path).convert("RGB")
                inputs = processor(text="Describe this image in detail.", images=image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"\n[Image: {img_file}]\nDescription: {generated_text}")
                results.append({"image": img_file, "description": generated_text})
            
            with open("evaluation_results.txt", "w") as f:
                for res in results:
                    f.write(f"Image: {res['image']}\nDescription: {res['description']}\n\n")

    # 2. Run Benchmarks if requested
    if run_benchmarks:
        pope = POPEEvaluator(POPE_DATA_PATH, image_dir) # Requires COCO images usually
        pope.evaluate(model, processor)
        
        mmhal = MMHalEvaluator(MMHal_DATA_PATH, image_dir)
        mmhal.evaluate(model, processor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model")
    parser.add_argument("--image_dir", type=str, default=TEST_IMAGE_DIR, help="Directory containing test images")
    parser.add_argument("--benchmarks", action="store_true", help="Run POPE and MMHal benchmarks")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.image_dir, args.benchmarks)

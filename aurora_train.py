import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# Import custom modules
from models import VLMModel, VerifierModel
from tools import ToolVerifier
from rewards import RewardCalculator

# --- 🟢 H200 CLUSTER CONFIGURATION ---
YFCC_ROOT_DIR = "./data/yfcc100m"
CHECKPOINT_DIR = "./output/checkpoints"

# ✅ MODEL PATHS (Point to Local)
LOCAL_VLM_PATH = "./models/Qwen3-VL-8B-Instruct"
LOCAL_VERIFIER_PATH = "./models/DeepSeek-R1-Distill-Qwen-7B"

# H200 Optimization
BATCH_SIZE = 32      # Per GPU Batch (Total = 256)
GROUP_SIZE = 8       # Samples per image
EPOCHS = 5
LEARNING_RATE_VLM = 1e-6
LEARNING_RATE_VERIFIER = 1e-6

# Auto-Download Config
DOWNLOAD_COUNT = 50000 
# -----------------------------

os.makedirs(YFCC_ROOT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class YFCCDownloader:
    def __init__(self, root_dir, target_count=DOWNLOAD_COUNT):
        self.root_dir = root_dir
        self.target_count = target_count

    async def download_image(self, session, url, idx):
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.read()
                    try:
                        Image.open(BytesIO(content)).verify()
                    except:
                        return False
                    path = os.path.join(self.root_dir, f"yfcc_{idx}.jpg")
                    with open(path, "wb") as f:
                        f.write(content)
                    return True
        except:
            return False
        return False

    async def run(self):
        print(f"🌍 [Auto-Data] Streaming YFCC100M metadata...")
        try:
            # Using dalle-mini subset as it's more stable
            ds = load_dataset("dalle-mini/YFCC100M_OpenAI_subset", split="train", streaming=True)
        except Exception as e:
            print(f"⚠️  Dataset load failed: {e}")
            return

        print(f"⬇️  Downloading {self.target_count} images to {self.root_dir}...")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            downloaded = 0
            pbar = tqdm(total=self.target_count, desc="Downloading Images", unit="img")
            
            for i, item in enumerate(ds):
                if downloaded >= self.target_count:
                    break
                
                url = item.get('url') or item.get('URL') or item.get('img_url')
                if not url:
                    # Some datasets have bytes directly
                    if 'img' in item and item['img']:
                        # Handle bytes if needed, but for now stick to URL logic or skip
                        continue
                    continue
                
                task = asyncio.create_task(self.download_image(session, url, i))
                tasks.append(task)
                
                if len(tasks) >= 200: 
                    results = await asyncio.gather(*tasks)
                    success_count = sum(results)
                    downloaded += success_count
                    pbar.update(success_count)
                    tasks = []
            
            if tasks:
                results = await asyncio.gather(*tasks)
                downloaded += sum(results)
                pbar.update(sum(results))
            pbar.close()

def ensure_yfcc_data(root_dir):
    try:
        files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(files)
    except FileNotFoundError:
        count = 0
    
    if count < 100:
        print(f"⚠️  Data directory {root_dir} is empty or low ({count} images).")
        print("🚀 Initiating Download...")
        downloader = YFCCDownloader(root_dir)
        asyncio.run(downloader.run())
    else:
        print(f"✅ Found {count} existing images.")

class YFCCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not self.image_files: 
            self.use_mock = True
        else: 
            self.use_mock = False

    def __len__(self):
        return len(self.image_files) if not self.use_mock else 100

    def __getitem__(self, idx):
        if self.use_mock: return Image.new('RGB', (224, 224)), "mock_image.jpg"
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            return image, img_path
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths

def select_diverse_descriptions(texts, model, target_count):
    if len(texts) <= target_count:
        return texts
    embeddings = model.encode(texts, convert_to_tensor=True)
    selected_indices = [0]
    while len(selected_indices) < target_count:
        remaining_indices = [i for i in range(len(texts)) if i not in selected_indices]
        if not remaining_indices: break
        remaining_embeddings = embeddings[remaining_indices]
        selected_embeddings = embeddings[selected_indices]
        cos_scores = util.pytorch_cos_sim(remaining_embeddings, selected_embeddings)
        max_sims_to_selected, _ = torch.max(cos_scores, dim=1)
        best_candidate_idx_in_remaining = torch.argmin(max_sims_to_selected).item()
        best_candidate_global_idx = remaining_indices[best_candidate_idx_in_remaining]
        selected_indices.append(best_candidate_global_idx)
    return [texts[i] for i in selected_indices]

def calculate_intra_claim_correlation(claims, model):
    if len(claims) < 2: return 0.0
    embeddings = model.encode(claims, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
    mask = torch.eye(len(claims), device=embeddings.device).bool()
    cos_sim.masked_fill_(mask, 0.0)
    avg_corr = cos_sim.sum() / (len(claims) * (len(claims) - 1))
    return avg_corr.item()

import argparse

def train():
    parser = argparse.ArgumentParser(description="AURORA Training")
    parser.add_argument("--mode", type=str, default="AURORA")
    parser.add_argument("--model_dir", type=str, default="../aurora_resources/models")
    parser.add_argument("--data_dir", type=str, default="../aurora_resources/data")
    parser.add_argument("--output_dir", type=str, default="../aurora_resources/output")
    parser.add_argument("--minilm_path", type=str, default="../aurora_resources/models/minilm")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--attack_weight", type=float, default=5.0)
    args = parser.parse_args()

    # 路径解析：支持绝对路径或相对于 model_dir 的子目录
    def get_path(base, sub):
        p = os.path.join(base, sub)
        return p if os.path.exists(p) else sub

    vlm_path = get_path(args.model_dir, "Qwen3-VL-8B-Instruct")
    verifier_path = get_path(args.model_dir, "DeepSeek-R1-Distill-Qwen-7B")
    yfcc_root = os.path.join(args.data_dir, "yfcc100m")

    # 动态创建输出目录
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"🚀 AURORA: 8x H200 High-Performance Mode")
        print(f"   VLM: {vlm_path} | Verifier: {verifier_path}")
        print(f"   Output: {checkpoint_dir}")
    
    # 初始化模型
    vlm = VLMModel(model_name=vlm_path, device=device)
    verifier = VerifierModel(model_name=verifier_path, device=device)
    # H200 极致加速
    if hasattr(torch, 'compile'):
        vlm.model = torch.compile(vlm.model)
        verifier.model = torch.compile(verifier.model)
    
    tools = ToolVerifier(device=device, model_root=args.model_dir)
    reward_calc = RewardCalculator(attack_weight=args.attack_weight)
    similarity_model = SentenceTransformer(args.minilm_path, device=device)

    torch.backends.cuda.matmul.allow_tf32 = True
    # 显式设置混合精度为 bf16
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"🚀 AURORA: 8x H200 Performance Mode [BF16 Enabled]")
        if not os.path.exists(yfcc_root): os.makedirs(yfcc_root, exist_ok=True)
    
    # 初始化模型
    vlm = VLMModel(model_name=vlm_path, device=device)
    verifier = VerifierModel(model_name=verifier_path, device=device)

    # --- H200 极致加速: 启用 PyTorch 2.x 编译优化 ---
    if hasattr(torch, 'compile'):
        if accelerator.is_main_process: print("⚡ Enabling torch.compile for H200 kernels...")
        vlm.model = torch.compile(vlm.model)
        verifier.model = torch.compile(verifier.model)
    
    tools = ToolVerifier(device=device, model_root=args.model_dir)
    reward_calc = RewardCalculator(attack_weight=args.attack_weight)
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # 2. Data & Optimizers
    dataset = YFCCDataset(YFCC_ROOT_DIR)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    vlm_optimizer = torch.optim.AdamW(vlm.model.parameters(), lr=LEARNING_RATE_VLM)
    verifier_optimizer = torch.optim.AdamW(verifier.model.parameters(), lr=LEARNING_RATE_VERIFIER)

    vlm.model, vlm_optimizer, dataloader = accelerator.prepare(vlm.model, vlm_optimizer, dataloader)
    verifier.model, verifier_optimizer = accelerator.prepare(verifier.model, verifier_optimizer)
    
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, image_paths) in enumerate(progress_bar):
            
            # === PHASE 1: Generate Data ===
            oversample_factor = 2 if not args.no_diversity else 1
            vlm_generated_texts = []
            
            with torch.no_grad():
                for img in images:
                    raw_descs = vlm.generate_description_batch([img], num_generations=GROUP_SIZE * oversample_factor)[0]
                    if not args.no_diversity:
                        diverse_descs = select_diverse_descriptions(raw_descs, similarity_model, GROUP_SIZE)
                    else:
                        diverse_descs = raw_descs[:GROUP_SIZE]
                    vlm_generated_texts.append(diverse_descs)
            
            flat_descriptions = [d for group in vlm_generated_texts for d in group]
            flat_images = [img for img in images for _ in range(GROUP_SIZE)]
            flat_img_paths = [path for path in image_paths for _ in range(GROUP_SIZE)]
            
            # === PHASE 2: Verification (Skipped in CLIP_ONLY baseline) ===
            verification_results = []
            verifier_raw_responses = [] 
            verifier_correlation_scores = []

            if args.mode == "AURORA":
                # Step B: Verifier logic
                verifier_claims_map = []
                with torch.no_grad():
                    for desc in flat_descriptions:
                        claims, raw_resp = verifier.verify_claims(desc)
                        verifier_claims_map.append(claims)
                        verifier_raw_responses.append(raw_resp)
                        corr_score = calculate_intra_claim_correlation(claims, similarity_model)
                        verifier_correlation_scores.append(corr_score)

                # Step C: Tools logic
                for i, claims in enumerate(verifier_claims_map):
                    img_path = flat_img_paths[i]
                    res_per_desc = []
                    for claim in claims:
                                            # Robust Traceability Check (Token Overlap)
                                            def is_traceable(claim_text, source_text, threshold=0.7):
                                                claim_tokens = set(claim_text.lower().split())
                                                source_tokens = set(source_text.lower().split())
                                                if not claim_tokens: return True
                                                overlap = claim_tokens.intersection(source_tokens)
                                                return (len(overlap) / len(claim_tokens)) >= threshold
                        
                                            verdict, conf, reason = tools.verify_claim(claim, img_path)
                                            traceable = is_traceable(claim, flat_descriptions[i])
                                            res_per_desc.append({'claim': claim, 'verdict': verdict, 'traceable': traceable})                    verification_results.append(res_per_desc)
            
            # === PHASE 3: Train VLM ===
            vlm_optimizer.zero_grad()
            all_vlm_rewards = []
            
            for i in range(len(images)):
                start_idx = i * GROUP_SIZE
                end_idx = start_idx + GROUP_SIZE
                group_texts = flat_descriptions[start_idx:end_idx]
                
                # Diversity Penalty (Only for AURORA)
                diversity_penalty = 0.0
                if args.mode == "AURORA" and GROUP_SIZE > 1:
                    embeddings = similarity_model.encode(group_texts, convert_to_tensor=True)
                    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
                    mask = torch.triu(torch.ones_like(cos_sim), diagonal=1).bool()
                    avg_sim = cos_sim[mask].mean().item() if mask.any() else 0.0
                    diversity_penalty = avg_sim * 2.0
                
                group_rewards = []
                if args.mode == "AURORA":
                    group_results = verification_results[start_idx:end_idx]
                    for j, res in enumerate(group_results):
                        correct = sum(1 for r in res if r['verdict'] == 'correct')
                        incorrect = sum(1 for r in res if r['verdict'] == 'incorrect')
                        r_vlm = reward_calc.calculate_vlm_reward(correct, incorrect, len(res), len(group_texts[j].split()))
                        r_vlm -= diversity_penalty
                        group_rewards.append(r_vlm)
                else: # CLIP_ONLY Baseline
                    for j, text in enumerate(group_texts):
                        img = images[i]
                        clip_score = tools._verify_clip(text[:100], img) # Text truncated for CLIP
                        group_rewards.append(clip_score * 5.0) # Scale to match AURORA range

                all_vlm_rewards.extend(group_rewards)

            # Update VLM Logic
            vlm_rewards_tensor = torch.tensor(all_vlm_rewards, device=device).view(-1, GROUP_SIZE)
            vlm_mean = vlm_rewards_tensor.mean(dim=1, keepdim=True)
            vlm_std = vlm_rewards_tensor.std(dim=1, keepdim=True) + 1e-8
            vlm_advantages = (vlm_rewards_tensor - vlm_mean) / vlm_std
            vlm_advantages = vlm_advantages.view(-1)
            
            # VLM Gradient Step (simplified call)
            total_vlm_loss = 0
            for k, (text, img) in enumerate(zip(flat_descriptions, flat_images)):
                messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this image in detail."}]}, {"role": "assistant", "content": [{"type": "text", "text": text}]}]
                text_input = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                inputs = vlm.processor(text=[text_input], images=[img], padding=True, return_tensors="pt").to(device)
                log_prob = vlm.compute_log_probs(inputs.input_ids, inputs.attention_mask, inputs.input_ids)
                total_vlm_loss += -vlm_advantages[k] * log_prob
            
            total_vlm_loss = total_vlm_loss / len(flat_descriptions)
            accelerator.backward(total_vlm_loss)
            vlm_optimizer.step()

            # === PHASE 4: Train Verifier (Only in AURORA mode) ===
            if args.mode == "AURORA":
                verifier_optimizer.zero_grad()
                all_verifier_rewards = []
                for k, res_list in enumerate(verification_results):
                    v_reward_sum = 0.0
                    corr_penalty = verifier_correlation_scores[k]
                    for r in res_list:
                        r_val = reward_calc.calculate_verifier_reward(r['verdict'], r['traceable'], correlation_score=corr_penalty)
                        v_reward_sum += r_val
                    all_verifier_rewards.append(v_reward_sum / (len(res_list) if res_list else 1))

                verifier_rewards_tensor = torch.tensor(all_verifier_rewards, device=device).view(-1, GROUP_SIZE)
                v_adv = (verifier_rewards_tensor - verifier_rewards_tensor.mean()) / (verifier_rewards_tensor.std() + 1e-8)
                v_adv = v_adv.view(-1)
                
                total_verifier_loss = 0
                for k, raw_text in enumerate(verifier_raw_responses):
                    log_prob = verifier.compute_sequence_log_prob(prompt=flat_descriptions[k], completion=raw_text)
                    total_verifier_loss += -v_adv[k] * log_prob
                
                total_verifier_loss = total_verifier_loss / len(flat_descriptions)
                accelerator.backward(total_verifier_loss)
                verifier_optimizer.step()
            
            if accelerator.is_main_process and batch_idx % 5 == 0:
                progress_bar.set_postfix({
                    "VLM_R": f"{vlm_rewards_tensor.mean().item():.2f}",
                    "Ver_R": f"{verifier_rewards_tensor.mean().item():.2f}" if args.mode=="AURORA" else "N/A"
                })
            
            if accelerator.is_main_process and batch_idx % 5 == 0:
                progress_bar.set_postfix({
                    "VLM_R": f"{vlm_rewards_tensor.mean().item():.2f}",
                    "Ver_R": f"{verifier_rewards_tensor.mean().item():.2f}"
                })
                
        if accelerator.is_main_process:
            save_path = os.path.join(CHECKPOINT_DIR, f"aurora_epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.unwrap_model(vlm.model).save_pretrained(os.path.join(save_path, "vlm"))
            vlm.processor.save_pretrained(os.path.join(save_path, "vlm"))
            accelerator.unwrap_model(verifier.model).save_pretrained(os.path.join(save_path, "verifier"))
            verifier.tokenizer.save_pretrained(os.path.join(save_path, "verifier"))

if __name__ == "__main__":
    train()

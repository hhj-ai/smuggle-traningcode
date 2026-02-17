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
import argparse

# Import custom modules
from models import VLMModel, VerifierModel
from tools import ToolVerifier
from rewards import RewardCalculator

# --- 🟢 CLUSTER CONFIGURATION ---
YFCC_ROOT_DIR = "./data/yfcc100m"
CHECKPOINT_DIR = "./output/checkpoints"

# H200 Optimization
GROUP_SIZE = 8       # Samples per image
EPOCHS = 5
LEARNING_RATE_VLM = 1e-6
LEARNING_RATE_VERIFIER = 1e-6

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

class YFCCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            return Image.open(img_path).convert("RGB"), img_path
        except: return self.__getitem__((idx + 1) % len(self))

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

    # 1. Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True

    # 2. Path Resolution
    vlm_path = os.path.join(args.model_dir, "Qwen3-VL-8B-Instruct")
    verifier_path = os.path.join(args.model_dir, "DeepSeek-R1-Distill-Qwen-7B")
    yfcc_root = args.data_dir # 直接使用传入的数据路径
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"🚀 AURORA: 8x H200 High-Performance Mode")
        print(f"   VLM: {vlm_path} | Verifier: {verifier_path}")
        print(f"   Data: {yfcc_root} | MiniLM: {args.minilm_path}")

    # 3. Load Models (Force Local)
    vlm = VLMModel(model_name=vlm_path, device=device)
    verifier = VerifierModel(model_name=verifier_path, device=device)
    
    # Optional Compile
    if hasattr(torch, 'compile'):
        vlm.model = torch.compile(vlm.model)
        verifier.model = torch.compile(verifier.model)
    
    # 4. Initialize Tools & Rewards
    tools = ToolVerifier(device=device, model_root=args.model_dir)
    reward_calc = RewardCalculator(attack_weight=args.attack_weight)
    
    if not os.path.exists(args.minilm_path):
        raise FileNotFoundError(f"❌ MiniLM path not found: {args.minilm_path}")
    similarity_model = SentenceTransformer(args.minilm_path, device=device)

    # 5. Data & Optimizers
    dataset = YFCCDataset(yfcc_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    v_opt = torch.optim.AdamW(vlm.model.parameters(), lr=LEARNING_RATE_VLM)
    ver_opt = torch.optim.AdamW(verifier.model.parameters(), lr=LEARNING_RATE_VERIFIER)

    vlm.model, v_opt, dataloader = accelerator.prepare(vlm.model, v_opt, dataloader)
    verifier.model, ver_opt = accelerator.prepare(verifier.model, ver_opt)
    
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch+1}")
        for batch_idx, (images, image_paths) in enumerate(progress_bar):
            
            # === PHASE 1: Generate Data ===
            vlm_gen_texts = []
            with torch.no_grad():
                for img in images:
                    raw_descs = vlm.generate_description_batch([img], num_generations=GROUP_SIZE * 2)[0]
                    vlm_gen_texts.append(select_diverse_descriptions(raw_descs, similarity_model, GROUP_SIZE))
            
            flat_descriptions = [d for group in vlm_gen_texts for d in group]
            flat_images = [img for img in images for _ in range(GROUP_SIZE)]
            flat_img_paths = [path for path in image_paths for _ in range(GROUP_SIZE)]
            
            # === PHASE 2: Verification ===
            verification_results = []
            ver_raw_resp = [] 
            ver_corr_scores = []

            if args.mode == "AURORA":
                def is_traceable(claim, source, threshold=0.7):
                    c_tokens, s_tokens = set(claim.lower().split()), set(source.lower().split())
                    if not c_tokens: return True
                    return (len(c_tokens & s_tokens) / len(c_tokens)) >= threshold

                with torch.no_grad():
                    for desc in flat_descriptions:
                        claims, raw = verifier.verify_claims(desc)
                        ver_raw_resp.append(raw)
                        ver_corr_scores.append(calculate_intra_claim_correlation(claims, similarity_model))
                        
                        res_per_desc = []
                        for c in claims:
                            v, _, _ = tools.verify_claim(c, flat_img_paths[len(verification_results)])
                            res_per_desc.append({'verdict': v, 'traceable': is_traceable(c, desc)})
                        verification_results.append(res_per_desc)
            
            # === PHASE 3: Train VLM ===
            v_opt.zero_grad()
            all_vlm_rewards = []
            for i in range(len(images)):
                start, end = i * GROUP_SIZE, (i + 1) * GROUP_SIZE
                group_texts, group_results = flat_descriptions[start:end], verification_results[start:end]
                
                # Penalty & Reward
                div_penalty = 0.0
                if args.mode == "AURORA" and GROUP_SIZE > 1:
                    emb = similarity_model.encode(group_texts, convert_to_tensor=True)
                    cos = util.pytorch_cos_sim(emb, emb)
                    mask = torch.triu(torch.ones_like(cos), 1).bool()
                    div_penalty = cos[mask].mean().item() * 2.0 if mask.any() else 0.0
                
                for j, res in enumerate(group_results):
                    c, inc = sum(1 for r in res if r['verdict']=='correct'), sum(1 for r in res if r['verdict']=='incorrect')
                    all_vlm_rewards.append(reward_calc.calculate_vlm_reward(c, inc, len(res), len(group_texts[j].split())) - div_penalty)

            vlm_rew_tensor = torch.tensor(all_vlm_rewards, device=device).view(-1, GROUP_SIZE)
            vlm_adv = (vlm_rew_tensor - vlm_rew_tensor.mean(1, keepdim=True)) / (vlm_rew_tensor.std(1, keepdim=True) + 1e-8)
            
            total_vlm_loss = 0
            for k, (text, img) in enumerate(zip(flat_descriptions, flat_images)):
                msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this image in detail."}]}, {"role": "assistant", "content": [{"type": "text", "text": text}]}]
                text_in = vlm.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                inputs = vlm.processor(text=[text_in], images=[img], padding=True, return_tensors="pt").to(device)
                total_vlm_loss += -vlm_adv.view(-1)[k] * vlm.compute_log_probs(inputs.input_ids, inputs.attention_mask, inputs.input_ids)
            
            accelerator.backward(total_vlm_loss / len(flat_descriptions))
            v_opt.step()

            # === PHASE 4: Train Verifier ===
            if args.mode == "AURORA":
                ver_opt.zero_grad()
                all_ver_rewards = [sum(reward_calc.calculate_verifier_reward(r['verdict'], r['traceable'], ver_corr_scores[k]) for r in res_list) / (len(res_list) if res_list else 1) for k, res_list in enumerate(verification_results)]
                ver_rew_tensor = torch.tensor(all_ver_rewards, device=device).view(-1, GROUP_SIZE)
                v_adv = (ver_rew_tensor - ver_rew_tensor.mean()) / (ver_rew_tensor.std() + 1e-8)
                
                total_ver_loss = 0
                for k, raw in enumerate(ver_raw_resp):
                    total_ver_loss += -v_adv.view(-1)[k] * verifier.compute_sequence_log_prob(flat_descriptions[k], raw)
                
                accelerator.backward(total_ver_loss / len(flat_descriptions))
                ver_opt.step()
            
            if accelerator.is_main_process and batch_idx % 5 == 0:
                postfix = {"VLM_R": f"{vlm_rew_tensor.mean().item():.2f}"}
                if args.mode == "AURORA": postfix["Ver_R"] = f"{ver_rew_tensor.mean().item():.2f}"
                progress_bar.set_postfix(postfix)
                
        if accelerator.is_main_process:
            save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.unwrap_model(vlm.model).save_pretrained(os.path.join(save_path, "vlm"))
            accelerator.unwrap_model(verifier.model).save_pretrained(os.path.join(save_path, "verifier"))

if __name__ == "__main__":
    train()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os, time, gc, argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, util
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

# Import custom modules
from models import VLMModel, VerifierModel
from tools import ToolVerifier
from rewards import RewardCalculator

# --- 极简 Dataset，防止内存溢出 ---
class YFCCDataset(Dataset):
    def __init__(self, root_dir, max_samples=20000):
        self.root_dir = root_dir
        # 核心优化：不再一次性读取所有文件名
        self.image_files = []
        count = 0
        if os.path.exists(root_dir):
            for f in os.scandir(root_dir): # 使用 scandir 比 listdir 更省内存
                if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(f.name)
                    count += 1
                    if count >= max_samples: break 
        self.image_files.sort() # 确保各 rank 顺序一致

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            return Image.open(img_path).convert("RGB"), img_path
        except: return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]

def select_diverse_descriptions(texts, model, target_count):
    if len(texts) <= target_count: return texts
    embeddings = model.encode(texts, convert_to_tensor=True)
    selected_indices = [0]
    while len(selected_indices) < target_count:
        remaining = [i for i in range(len(texts)) if i not in selected_indices]
        if not remaining: break
        cos_scores = util.pytorch_cos_sim(embeddings[remaining], embeddings[selected_indices])
        best_idx = torch.argmin(cos_scores.max(dim=1)[0]).item()
        selected_indices.append(remaining[best_idx])
    return [texts[i] for i in selected_indices]

def calculate_intra_claim_correlation(claims, model):
    if len(claims) < 2: return 0.0
    embeddings = model.encode(claims, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
    mask = torch.eye(len(claims), device=embeddings.device).bool()
    return cos_sim.masked_fill_(mask, 0.0).sum().item() / (len(claims)*(len(claims)-1))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="AURORA")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--minilm_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attack_weight", type=float, default=5.0)
    args = parser.parse_args()

    # 1. 稳定性初始化
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device
    
    # 2. 顺序加载，严格保护 RAM
    vlm, verifier, tools, similarity_model = None, None, None, None
    vlm_path = os.path.join(args.model_dir, "Qwen3-VL-8B-Instruct")
    verifier_path = os.path.join(args.model_dir, "DeepSeek-R1-Distill-Qwen-7B")
    
    for i in range(accelerator.num_processes):
        if accelerator.local_process_index == i:
            print(f"📦 [Rank {i}] Loading...")
            vlm = VLMModel(model_name=vlm_path, device=device)
            verifier = VerifierModel(model_name=verifier_path, device=device)
            tools = ToolVerifier(device=device, model_root=args.model_dir)
            similarity_model = SentenceTransformer(args.minilm_path, device=device)
            gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    reward_calc = RewardCalculator(attack_weight=args.attack_weight)

    # 3. 数据集加载 (只有主进程打印日志，全员排队防止磁盘 IO 夯死 RAM)
    dataset = YFCCDataset(args.data_dir, max_samples=10000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    
    v_opt = torch.optim.AdamW(vlm.model.parameters(), lr=1e-6)
    ver_opt = torch.optim.AdamW(verifier.model.parameters(), lr=1e-6)
    vlm.model, v_opt, dataloader = accelerator.prepare(vlm.model, v_opt, dataloader)
    verifier.model, ver_opt = accelerator.prepare(verifier.model, ver_opt)

    for epoch in range(5):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for batch_idx, (images, image_paths) in enumerate(pbar):
            # PHASE 1: Rollout
            vlm_texts = []
            with torch.no_grad():
                for img in images:
                    raw = vlm.generate_description_batch([img], num_generations=16)[0]
                    vlm_texts.append(select_diverse_descriptions(raw, similarity_model, 8))
            
            flat_desc = [d for g in vlm_texts for d in g]
            flat_paths = [p for p in image_paths for _ in range(8)]
            
            # PHASE 2: Verifier
            ver_results, ver_raw, ver_corr = [], [], []
            with torch.no_grad():
                for i, desc in enumerate(flat_desc):
                    claims, raw = verifier.verify_claims(desc)
                    ver_raw.append(raw)
                    ver_corr.append(calculate_intra_claim_correlation(claims, similarity_model))
                    res = []
                    for c in claims:
                        v, _, _ = tools.verify_claim(c, flat_paths[i])
                        # Token 溯源
                        t = len(set(c.lower().split()) & set(desc.lower().split())) / (len(c.split())+1e-6) > 0.7
                        res.append({'verdict': v, 'traceable': t})
                    ver_results.append(res)

            # PHASE 3 & 4: Train (GRPO)
            # ... 简化版奖励计算逻辑 ...
            v_rewards = []
            for i, res_list in enumerate(ver_results):
                c = sum(1 for r in res_list if r['verdict']=='correct')
                inc = sum(1 for r in res_list if r['verdict']=='incorrect')
                v_rewards.append(reward_calc.calculate_vlm_reward(c, inc, len(res_list), len(flat_desc[i].split())))
            
            v_rew_t = torch.tensor(v_rewards, device=device).view(-1, 8)
            v_adv = (v_rew_t - v_rew_t.mean(1, keepdim=True)) / (v_rew_t.std(1, keepdim=True) + 1e-8)
            
            # VLM Step
            v_opt.zero_grad()
            # ... 计算 loss 并 backward ...
            
            if accelerator.is_main_process and batch_idx % 10 == 0:
                pbar.set_postfix({"VLM_R": f"{v_rew_t.mean():.2f}"})

if __name__ == "__main__": train()

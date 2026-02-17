import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os, time, gc, argparse, sys
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

# --- 极简 Dataset (RAM 保护版) ---
class YFCCDataset(Dataset):
    def __init__(self, root_dir, max_samples=20000):
        self.root_dir = root_dir
        self.image_files = []
        if os.path.exists(root_dir):
            for i, f in enumerate(os.scandir(root_dir)):
                if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(f.name)
                if i >= max_samples: break
        self.image_files.sort()

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        try:
            path = os.path.join(self.root_dir, self.image_files[idx])
            return Image.open(path).convert("RGB"), path
        except: return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]

def select_diverse_descriptions(texts, model, target_count):
    if len(texts) <= target_count: return texts
    embeddings = model.encode(texts, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
    selected_indices = list(range(len(texts)))
    while len(selected_indices) > target_count:
        mask = torch.eye(len(selected_indices), device=cos_sim.device).bool()
        cos_sim.masked_fill_(mask, -1.0)
        to_remove = torch.argmax(torch.max(cos_sim, dim=1)[0]).item()
        selected_indices.pop(to_remove)
        break 
    return [texts[i] for i in selected_indices[:target_count]]

def calculate_intra_claim_correlation(claims, model):
    if len(claims) < 2: return 0.0
    embeddings = model.encode(claims, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
    mask = torch.eye(len(claims), device=embeddings.device).bool()
    return cos_sim.masked_fill_(mask, 0.0).sum().item() / (len(claims)*(len(claims)-1) + 1e-6)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="AURORA")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--minilm_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--attack_weight", type=float, default=5.0)
    args = parser.parse_args()

    # 1. 稳定性初始化
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True

    # 2. 顺序加载与鲁棒性检查
    vlm_path = os.path.join(args.model_dir, "Qwen3-VL-8B-Instruct")
    verifier_path = os.path.join(args.model_dir, "DeepSeek-R1-Distill-Qwen-7B")
    
    vlm, verifier, tools, similarity_model = None, None, None, None
    
    try:
        for i in range(accelerator.num_processes):
            if accelerator.local_process_index == i:
                print(f"📦 [Rank {i}] Initializing Models & Tools...")
                vlm = VLMModel(model_name=vlm_path, device=device)
                verifier = VerifierModel(model_name=verifier_path, device=device)
                tools = ToolVerifier(device=device, model_root=args.model_dir)
                similarity_model = SentenceTransformer(args.minilm_path, device=device)
                gc.collect(); torch.cuda.empty_cache()
                print(f"✅ [Rank {i}] Ready.")
            accelerator.wait_for_everyone()
    except Exception as e:
        print(f"🛑 [Rank {accelerator.local_process_index}] INIT FAILED: {e}")
        sys.exit(1)

    reward_calc = RewardCalculator(attack_weight=args.attack_weight)
    dataset = YFCCDataset(args.data_dir, max_samples=20000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    v_opt = torch.optim.AdamW(vlm.model.parameters(), lr=1e-6)
    ver_opt = torch.optim.AdamW(verifier.model.parameters(), lr=1e-6)
    vlm.model, v_opt, dataloader = accelerator.prepare(vlm.model, v_opt, dataloader)
    verifier.model, ver_opt = accelerator.prepare(verifier.model, ver_opt)

    for epoch in range(5):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for batch_idx, (images, image_paths) in enumerate(pbar):
            with torch.no_grad():
                vlm_gen_texts = []
                for img in images:
                    raw = vlm.generate_description_batch([img], num_generations=10)[0]
                    vlm_gen_texts.append(select_diverse_descriptions(raw, similarity_model, 8))
            
            flat_desc = [d for g in vlm_gen_texts for d in g]
            flat_paths = [p for p in image_paths for _ in range(8)]
            
            # PHASE 2: Verifier
            ver_results, ver_raw_resp, ver_corr_scores = [], [], []
            with torch.no_grad():
                for i, desc in enumerate(flat_desc):
                    claims, raw = verifier.verify_claims(desc)
                    ver_raw_resp.append(raw)
                    ver_corr_scores.append(0.1)
                    res = []
                    for c in claims:
                        v, _, _ = tools.verify_claim(c, flat_paths[i])
                        t = (c.split()[0].lower() in desc.lower())
                        res.append({'verdict': v, 'traceable': t})
                    ver_results.append(res)

            # PHASE 3: Update
            v_rewards = []
            for i, res_list in enumerate(ver_results):
                c = sum(1 for r in res_list if r['verdict']=='correct')
                inc = sum(1 for r in res_list if r['verdict']=='incorrect')
                v_rewards.append(reward_calc.calculate_vlm_reward(c, inc, len(res_list), len(flat_desc[i].split())))
            
            v_rew_t = torch.tensor(v_rewards, device=device).view(-1, 8)
            v_adv = (v_rew_t - v_rew_t.mean(1, keepdim=True)) / (v_rew_t.std(1, keepdim=True) + 1e-8)
            
            v_opt.zero_grad()
            total_loss = 0
            for k, (text, img) in enumerate(zip(flat_desc, [i for i in images for _ in range(8)])):
                msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe."}]}, {"role": "assistant", "content": [{"type": "text", "text": text}]}]
                text_in = vlm.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                inputs = vlm.processor(text=[text_in], images=[img], padding=True, return_tensors="pt").to(device)
                total_loss += -v_adv.view(-1)[k] * vlm.compute_log_probs(inputs.input_ids, inputs.attention_mask, inputs.input_ids)
            
            accelerator.backward(total_loss / len(flat_desc))
            v_opt.step()
            
            if accelerator.is_main_process and batch_idx % 5 == 0:
                pbar.set_postfix({"R": f"{v_rew_t.mean():.2f}"})

        if accelerator.is_main_process:
            save_p = os.path.join(args.output_dir, f"checkpoints/epoch_{epoch}")
            os.makedirs(save_p, exist_ok=True)
            accelerator.unwrap_model(vlm.model).save_pretrained(os.path.join(save_p, "vlm"))

if __name__ == "__main__": train()

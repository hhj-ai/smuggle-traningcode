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

from models import VLMModel, VerifierModel
from tools import ToolVerifier
from rewards import RewardCalculator

# --- 极简 Dataset (14GB RAM 保护) ---
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

    # 1. 稳定性初始化 (高超时保护)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1, kwargs_handlers=[timeout_kwargs])
    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True

    # 2. 路径映射
    vlm_path = os.path.abspath(os.path.join(args.model_dir, "Qwen3-VL-8B-Instruct"))
    verifier_path = os.path.abspath(os.path.join(args.model_dir, "DeepSeek-R1-Distill-Qwen-7B"))
    checkpoint_dir = os.path.abspath(os.path.join(args.output_dir, "checkpoints"))
    
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"🚀 AURORA: 8x H200 (14GB RAM Mode)")

    # 3. 并行加载（避免死锁）
    vlm, verifier, tools, similarity_model = None, None, None, None
    try:
        print(f"📦 [Rank {accelerator.local_process_index}] Loading models in parallel...", flush=True)
        vlm = VLMModel(model_name=vlm_path, device=device)
        print(f"  ✓ VLM loaded", flush=True)
        verifier = VerifierModel(model_name=verifier_path, device=device)
        print(f"  ✓ Verifier loaded", flush=True)
        tools = ToolVerifier(device=device, model_root=args.model_dir)
        print(f"  ✓ Tools loaded", flush=True)
        similarity_model = SentenceTransformer(args.minilm_path, device=device)
        print(f"  ✓ SentenceTransformer loaded", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        print(f"  ✓ Memory cleaned", flush=True)
    except Exception as e:
        print(f"🛑 INIT FAILED: {e}", flush=True)
        sys.exit(1)

    accelerator.wait_for_everyone()
    print(f"[DEBUG] Rank {accelerator.local_process_index} all models loaded and synchronized", flush=True)

    reward_calc = RewardCalculator(attack_weight=args.attack_weight)
    dataset = YFCCDataset(args.data_dir, max_samples=20000)
    print(f"[DEBUG] Dataset size: {len(dataset)} images", flush=True)
    if len(dataset) == 0:
        print(f"🛑 ERROR: No images found in {args.data_dir}", flush=True)
        sys.exit(1)
    # 使用num_workers=0避免分布式环境下的问题
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
    print(f"[DEBUG] Dataloader created, batch size: {args.batch_size}", flush=True)
    
    v_opt = torch.optim.AdamW(vlm.model.parameters(), lr=1e-6)
    ver_opt = torch.optim.AdamW(verifier.model.parameters(), lr=1e-6)
    vlm.model, v_opt, dataloader = accelerator.prepare(vlm.model, v_opt, dataloader)
    verifier.model, ver_opt = accelerator.prepare(verifier.model, ver_opt)

    GROUP_SIZE = 8

    # 5. 完整训练循环
    if accelerator.is_main_process:
        print(f"[INFO] Starting training loop, total batches: {len(dataloader)}", flush=True)
    accelerator.wait_for_everyone()
    print(f"[DEBUG-Rank{accelerator.local_process_index}] All ranks synchronized, starting epoch loop", flush=True)
    for epoch in range(5):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for batch_idx, (images, image_paths) in enumerate(pbar):
            
            # === PHASE 1: VLM 生成 ===
            flat_desc = []
            flat_images = []
            flat_paths = []

            with torch.no_grad():
                print(f"[DEBUG-Rank{accelerator.local_process_index}] Starting VLM generation for {len(images)} images", flush=True)
                for idx, (img, path) in enumerate(zip(images, image_paths)):
                    print(f"[DEBUG-Rank{accelerator.local_process_index}] Generating descriptions for image {idx+1}/{len(images)}", flush=True)
                    # 采样 10 个选 8 个
                    raw = vlm.generate_description_batch([img], num_generations=10)[0]
                    print(f"[DEBUG-Rank{accelerator.local_process_index}] Generated {len(raw)} raw descriptions", flush=True)
                    diverse = select_diverse_descriptions(raw, similarity_model, GROUP_SIZE)
                    print(f"[DEBUG-Rank{accelerator.local_process_index}] Selected {len(diverse)} diverse descriptions", flush=True)
                    flat_desc.extend(diverse)
                    flat_images.extend([img] * len(diverse))
                    flat_paths.extend([path] * len(diverse))
                print(f"[DEBUG-Rank{accelerator.local_process_index}] VLM generation complete, total descriptions: {len(flat_desc)}", flush=True)
            
            # === PHASE 2: Verifier 提取与验证 ===
            ver_results = []
            ver_raw_resp = [] 
            ver_corr_scores = []

            with torch.no_grad():
                for i, desc in enumerate(flat_desc):
                    # 优化 Prompt：明确指令
                    prompt = f"Description: {desc}\n\nTask: List all visual claims. Start each claim with '- '."
                    claims, raw = verifier.verify_claims(desc) # Prompt 逻辑在内部封装
                    ver_raw_resp.append(raw)
                    ver_corr_scores.append(calculate_intra_claim_correlation(claims, similarity_model))
                    
                    res_per_desc = []
                    for c in claims:
                        v, _, _ = tools.verify_claim(c, flat_paths[i])
                        # 简单词袋溯源
                        t = (len(set(c.lower().split()) & set(desc.lower().split())) / (len(c.split())+1e-6) > 0.7)
                        res_per_desc.append({'verdict': v, 'traceable': t})
                    ver_results.append(res_per_desc)

            # === PHASE 3: VLM 训练 ===
            print(f"[DEBUG-Rank{accelerator.local_process_index}] Calculating VLM rewards for {len(flat_desc)} descriptions", flush=True)
            all_vlm_rewards = []

            for i in range(len(images)): # 按组处理奖励计算
                start, end = i * GROUP_SIZE, (i + 1) * GROUP_SIZE
                group_res = ver_results[start:end]
                group_txt = flat_desc[start:end]

                # 多样性惩罚
                div_penalty = 0.0
                if len(group_txt) > 1:
                    emb = similarity_model.encode(group_txt, convert_to_tensor=True)
                    cos = util.pytorch_cos_sim(emb, emb)
                    mask = torch.triu(torch.ones_like(cos), 1).bool()
                    div_penalty = cos[mask].mean().item() if mask.any() else 0.0

                for j, res in enumerate(group_res):
                    c = sum(1 for r in res if r['verdict']=='correct')
                    inc = sum(1 for r in res if r['verdict']=='incorrect')
                    r = reward_calc.calculate_vlm_reward(c, inc, len(res), len(group_txt[j].split()))
                    all_vlm_rewards.append(r - div_penalty)

            # 优势归一化
            v_rew_t = torch.tensor(all_vlm_rewards, device=device).view(-1, GROUP_SIZE)
            v_adv = (v_rew_t - v_rew_t.mean(1, keepdim=True)) / (v_rew_t.std(1, keepdim=True) + 1e-8)
            
            # VLM Loss - 按组处理避免OOM
            print(f"[DEBUG-Rank{accelerator.local_process_index}] Starting VLM training for {len(flat_desc)} descriptions in groups of {GROUP_SIZE}", flush=True)
            v_opt.zero_grad()

            # 按组处理：每组8个描述
            for group_idx in range(len(images)):
                start, end = group_idx * GROUP_SIZE, (group_idx + 1) * GROUP_SIZE
                group_desc = flat_desc[start:end]
                group_images = flat_images[start:end]
                group_adv = v_adv[group_idx]  # shape: (GROUP_SIZE,)

                group_loss = 0
                for k in range(len(group_desc)):
                    text = group_desc[k]
                    img = group_images[k]
                    msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe."}]}, {"role": "assistant", "content": [{"type": "text", "text": text}]}]
                    text_in = vlm.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                    inputs = vlm.processor(text=[text_in], images=[img], padding=True, return_tensors="pt").to(device)
                    group_loss += -group_adv[k] * vlm.compute_log_probs(inputs.input_ids, inputs.attention_mask, inputs.input_ids)

                # 每个组单独反向传播和梯度更新
                accelerator.backward(group_loss / len(group_desc))
                v_opt.step()
                v_opt.zero_grad()

                # 清理GPU内存
                torch.cuda.empty_cache()
                print(f"[DEBUG-Rank{accelerator.local_process_index}] Processed group {group_idx+1}/{len(images)}", flush=True)

            # === PHASE 4: Verifier 训练 ===
            print(f"[DEBUG-Rank{accelerator.local_process_index}] Starting Verifier training for {len(flat_desc)} descriptions", flush=True)

            # 先计算全局Verifier奖励用于显示
            all_ver_rewards = []
            for k, res_list in enumerate(ver_results):
                r_sum = sum(reward_calc.calculate_verifier_reward(r['verdict'], r['traceable'], ver_corr_scores[k]) for r in res_list)
                all_ver_rewards.append(r_sum / (len(res_list) if res_list else 1))

            global_ver_rew_t = torch.tensor(all_ver_rewards, device=device)
            ver_opt.zero_grad()

            # 按组处理Verifier训练
            for group_idx in range(len(images)):
                start, end = group_idx * GROUP_SIZE, (group_idx + 1) * GROUP_SIZE
                group_desc = flat_desc[start:end]
                group_raw = ver_raw_resp[start:end]
                group_results = ver_results[start:end]
                group_corr = ver_corr_scores[start:end]

                # 计算本组的奖励
                group_ver_rewards = []
                for k in range(len(group_desc)):
                    res_list = group_results[k]
                    r_sum = sum(reward_calc.calculate_verifier_reward(r['verdict'], r['traceable'], group_corr[k]) for r in res_list)
                    group_ver_rewards.append(r_sum / (len(res_list) if res_list else 1))

                # 本组优势归一化
                ver_rew_t = torch.tensor(group_ver_rewards, device=device)
                ver_adv = (ver_rew_t - ver_rew_t.mean()) / (ver_rew_t.std() + 1e-8)

                # 本组loss
                group_ver_loss = 0
                for k in range(len(group_desc)):
                    group_ver_loss += -ver_adv[k] * verifier.compute_sequence_log_prob(group_desc[k], group_raw[k])

                # 反向传播和更新
                accelerator.backward(group_ver_loss / len(group_desc))
                ver_opt.step()
                ver_opt.zero_grad()

                # 清理GPU内存
                torch.cuda.empty_cache()
                print(f"[DEBUG-Rank{accelerator.local_process_index}] Processed Verifier group {group_idx+1}/{len(images)}", flush=True)

            # 用于显示的全局Verifier奖励平均值
            ver_rew_display = global_ver_rew_t.mean().item()
            
            if accelerator.is_main_process and batch_idx % 5 == 0:
                pbar.set_postfix({"V_Rew": f"{v_rew_t.mean().item():.2f}", "Ver_Rew": f"{ver_rew_display:.2f}"})

        # Save Checkpoint
        if accelerator.is_main_process:
            save_p = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            os.makedirs(save_p, exist_ok=True)
            accelerator.unwrap_model(vlm.model).save_pretrained(os.path.join(save_p, "vlm"))
            accelerator.unwrap_model(verifier.model).save_pretrained(os.path.join(save_p, "verifier"))

if __name__ == "__main__": train()

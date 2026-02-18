import torch
import torch.nn.functional as F
import torch.distributed as dist
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


# ============================================================
# GPU 显存管理器：监控、模型换入换出、OOM 重试
# ============================================================
class GPUMemoryManager:
    """运行时 GPU 显存管理，支持模型换入换出和 OOM 安全重试"""

    def __init__(self, device, accelerator, warn_threshold_mib=4096):
        self.device = device
        self.accelerator = accelerator
        self.rank = accelerator.local_process_index
        self.warn_threshold_mib = warn_threshold_mib  # 低于此值发出警告

    # --- 显存查询 ---
    def get_free_mib(self):
        if not torch.cuda.is_available():
            return float('inf')
        free, _ = torch.cuda.mem_get_info(self.device)
        return free / (1024 * 1024)

    def get_allocated_mib(self):
        return torch.cuda.memory_allocated(self.device) / (1024 * 1024)

    def get_reserved_mib(self):
        return torch.cuda.memory_reserved(self.device) / (1024 * 1024)

    # --- 显存日志 ---
    def log(self, phase_name):
        free = self.get_free_mib()
        alloc = self.get_allocated_mib()
        reserved = self.get_reserved_mib()
        warn = " ⚠️ LOW MEMORY" if free < self.warn_threshold_mib else ""
        print(f"[MEM-R{self.rank}] {phase_name}: "
              f"alloc={alloc:.0f}MiB rsv={reserved:.0f}MiB free={free:.0f}MiB{warn}",
              flush=True)
        return free

    # --- 清理 ---
    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

    # --- 模型换出到 CPU（释放 GPU 显存）---
    def offload(self, model_wrapper, name="model"):
        """将 DDP 包装的模型内部参数移到 CPU"""
        inner = self.accelerator.unwrap_model(model_wrapper)
        inner.to("cpu")
        self.cleanup()
        freed = self.get_free_mib()
        print(f"[MEM-R{self.rank}] ↓ {name} → CPU (GPU free: {freed:.0f}MiB)", flush=True)

    # --- 模型换入到 GPU ---
    def reload(self, model_wrapper, name="model"):
        """将模型参数移回 GPU"""
        inner = self.accelerator.unwrap_model(model_wrapper)
        inner.to(self.device)
        free = self.get_free_mib()
        print(f"[MEM-R{self.rank}] ↑ {name} → GPU (GPU free: {free:.0f}MiB)", flush=True)

    # --- OOM 安全执行 ---
    def safe_execute(self, fn, retries=2, cleanup_before_retry=True):
        """
        执行 fn()，遇到 OOM 自动清理缓存后重试。
        返回 (result, success)。
        """
        for attempt in range(retries + 1):
            try:
                return fn(), True
            except torch.cuda.OutOfMemoryError:
                if attempt < retries:
                    print(f"[MEM-R{self.rank}] ⚠️ OOM (attempt {attempt+1}/{retries+1}), "
                          f"cleaning up...", flush=True)
                    if cleanup_before_retry:
                        self.cleanup()
                else:
                    print(f"[MEM-R{self.rank}] ❌ OOM after {retries+1} attempts", flush=True)
                    raise


# --- 高性能 Dataset (1.5TB RAM 优化) ---
class YFCCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []
        if os.path.exists(root_dir):
            # 递归扫描所有子目录中的图像文件
            import glob
            image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
            for ext in image_extensions:
                self.image_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
            # 只保留相对路径
            self.image_files = [os.path.relpath(f, root_dir) for f in self.image_files]
        self.image_files.sort()
        print(f"[Dataset] Found {len(self.image_files)} images in {root_dir}")

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
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)  # [n, n]
    n = len(texts)
    selected_indices = list(range(n))
    while len(selected_indices) > target_count:
        # 获取当前选中索引对应的子矩阵
        sub_cos = cos_sim[selected_indices][:, selected_indices]  # [m, m]
        m = len(selected_indices)
        # 将对角线掩码设为-1，避免自相似
        mask = torch.eye(m, device=cos_sim.device).bool()
        sub_cos.masked_fill_(mask, -1.0)
        # 找到最大相似度的位置（忽略对角线）
        max_val = sub_cos.max()
        if max_val <= -0.999:  # 所有相似度都很低，随机移除一个
            rand_idx = torch.randint(0, m, (1,)).item()
            to_remove = selected_indices[rand_idx]
        else:
            max_pos = (sub_cos == max_val).nonzero(as_tuple=False)[0]
            row_idx = max_pos[0].item()
            # 移除当前子矩阵中行索引对应的原始索引
            to_remove = selected_indices[row_idx]
        selected_indices.remove(to_remove)
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
    parser.add_argument("--tool_device", type=str, default=None, help="专用工具GPU (如 cuda:4)，不指定则用rank 0的训练卡")
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
        print(f"🚀 AURORA Training")

    # 初始化显存管理器
    mem = GPUMemoryManager(device, accelerator)

    # 3. 顺序加载 + 同步屏障（减少峰值内存与IO争抢）
    vlm, verifier, tools, similarity_model = None, None, None, None
    try:
        # --- VLM (最大模型，优先加载) ---
        mem.log("Before VLM load")
        print(f"📦 [Rank {accelerator.local_process_index}] Loading VLM...", flush=True)
        vlm = VLMModel(model_name=vlm_path, device=device)
        print(f"  ✓ [Rank {accelerator.local_process_index}] VLM loaded", flush=True)
        mem.cleanup()
        mem.log("After VLM load")
        accelerator.wait_for_everyone()

        # --- Verifier ---
        print(f"📦 [Rank {accelerator.local_process_index}] Loading Verifier...", flush=True)
        verifier = VerifierModel(model_name=verifier_path, device=device)
        print(f"  ✓ [Rank {accelerator.local_process_index}] Verifier loaded", flush=True)
        mem.cleanup()
        mem.log("After Verifier load")
        accelerator.wait_for_everyone()

        # --- Tools: 仅主进程加载，支持专用工具卡 ---
        if accelerator.is_main_process:
            tool_dev = torch.device(args.tool_device) if args.tool_device else device
            print(f"📦 [Rank {accelerator.local_process_index}] Loading Tools on {tool_dev}...", flush=True)
            tools = ToolVerifier(device=tool_dev, model_root=args.model_dir)
            print(f"  ✓ Tools loaded on {tool_dev} (rank 0 only)", flush=True)
        else:
            tools = None
            print(f"⏭️  [Rank {accelerator.local_process_index}] Skipping tools (rank 0 only)", flush=True)
        mem.cleanup()
        accelerator.wait_for_everyone()

        # --- SentenceTransformer (小模型) ---
        print(f"📦 [Rank {accelerator.local_process_index}] Loading SentenceTransformer...", flush=True)
        similarity_model = SentenceTransformer(args.minilm_path, device=device)
        print(f"  ✓ [Rank {accelerator.local_process_index}] SentenceTransformer loaded", flush=True)
        mem.cleanup()
        mem.log("All models loaded")
        accelerator.wait_for_everyone()
    except Exception as e:
        print(f"🛑 INIT FAILED on rank {accelerator.local_process_index}: {e}", flush=True)
        sys.exit(1)

    accelerator.wait_for_everyone()
    print(f"[DEBUG] Rank {accelerator.local_process_index} all models loaded and synchronized", flush=True)

    reward_calc = RewardCalculator(attack_weight=args.attack_weight)
    dataset = YFCCDataset(args.data_dir)
    print(f"[DEBUG] Dataset size: {len(dataset)} images", flush=True)
    if len(dataset) == 0:
        print(f"🛑 ERROR: No images found in {args.data_dir}", flush=True)
        sys.exit(1)
    # 高性能数据加载设置 (1.5TB RAM优化)
    num_workers = min(8, os.cpu_count() // 2)  # 根据CPU核心数动态设置
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                           num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    print(f"[DEBUG] Dataloader created, batch size: {args.batch_size}", flush=True)

    # ============================================================
    # 交错 DDP 包装：避免两个模型同时在 GPU 时 Reducer 分配 OOM
    # DDP Reducer 需要 ~模型大小 的显存用于梯度通信桶
    # VLM 8B bf16 ≈ 16 GiB, Verifier 7B bf16 ≈ 14 GiB
    # 同时在 GPU 时峰值 = 16+14+16(Reducer) = 46 GiB，共享 GPU 可能不够
    # ============================================================

    # Step 1: Verifier → CPU，为 VLM DDP 初始化腾显存
    verifier.model.to("cpu")
    mem.cleanup()
    mem.log("Before VLM DDP (Verifier on CPU)")

    v_opt = torch.optim.AdamW(vlm.model.parameters(), lr=1e-6)
    vlm.model, v_opt, dataloader = accelerator.prepare(vlm.model, v_opt, dataloader)
    mem.log("After VLM DDP")

    # Step 2: VLM 内部 → CPU，为 Verifier DDP 初始化腾显存
    # 注意：DDP Reducer 的梯度桶仍留在 GPU，但模型参数释放 ~16 GiB
    accelerator.unwrap_model(vlm.model).to("cpu")
    mem.cleanup()
    mem.log("Before Verifier DDP (VLM on CPU)")

    # Step 3: Verifier → GPU 并 DDP 包装
    verifier.model.to(device)
    ver_opt = torch.optim.AdamW(verifier.model.parameters(), lr=1e-6)
    verifier.model, ver_opt = accelerator.prepare(verifier.model, ver_opt)
    mem.log("After Verifier DDP")

    # Step 4: VLM → GPU（训练循环 Phase 1 需要 VLM 在 GPU）
    accelerator.unwrap_model(vlm.model).to(device)
    mem.cleanup()
    mem.log("Both models DDP-wrapped and on GPU")

    GROUP_SIZE = 8

    # 5. 完整训练循环
    if accelerator.is_main_process:
        print(f"[INFO] Starting training loop, total batches: {len(dataloader)}", flush=True)
    accelerator.wait_for_everyone()
    print(f"[DEBUG-Rank{accelerator.local_process_index}] All ranks synchronized, starting epoch loop", flush=True)
    for epoch in range(5):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for batch_idx, (images, image_paths) in enumerate(pbar):

            # ============================================================
            # PHASE 1: VLM 生成（Verifier 换出到 CPU 腾显存）
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase1-start")
            mem.offload(verifier.model, "Verifier")

            flat_desc = []
            flat_images = []
            flat_paths = []

            with torch.no_grad():
                for idx, (img, path) in enumerate(zip(images, image_paths)):
                    # OOM 安全生成：先尝试 10 个，OOM 则降到 5 个
                    num_gen = 10
                    def _generate():
                        return vlm.generate_description_batch([img], num_generations=num_gen)[0]

                    try:
                        raw, _ = mem.safe_execute(_generate, retries=1)
                    except torch.cuda.OutOfMemoryError:
                        # 降级：减少生成数量
                        num_gen = 5
                        print(f"[MEM-R{mem.rank}] ⚠️ 降级生成数: {num_gen}", flush=True)
                        raw, _ = mem.safe_execute(_generate, retries=1)

                    diverse = select_diverse_descriptions(raw, similarity_model, GROUP_SIZE)
                    flat_desc.extend(diverse)
                    flat_images.extend([img] * len(diverse))
                    flat_paths.extend([path] * len(diverse))

            mem.cleanup()

            # ============================================================
            # PHASE 2: Verifier 提取 + 工具验证
            #   Verifier 换入, VLM 换出
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase2-start")
            mem.offload(vlm.model, "VLM")
            mem.reload(verifier.model, "Verifier")

            ver_raw_resp = []
            ver_corr_scores = []
            local_claims_list = []

            with torch.no_grad():
                for i, desc in enumerate(flat_desc):
                    def _verify():
                        return verifier.verify_claims(desc)
                    try:
                        (claims, raw), _ = mem.safe_execute(_verify, retries=1)
                    except torch.cuda.OutOfMemoryError:
                        print(f"[MEM-R{mem.rank}] ⚠️ Verifier OOM on desc {i}, using empty claims", flush=True)
                        claims, raw = [], ""

                    ver_raw_resp.append(raw)
                    ver_corr_scores.append(calculate_intra_claim_correlation(claims, similarity_model))
                    local_claims_list.append((claims, flat_paths[i], desc))

            # Gather claims 到 rank 0 进行工具验证
            gathered_claims = [None] * accelerator.num_processes if accelerator.is_main_process else None
            dist.gather_object(local_claims_list, gathered_claims, dst=0)

            # Rank 0 执行工具验证
            if accelerator.is_main_process:
                all_results_by_rank = []
                for rank_claims in gathered_claims:
                    rank_results = []
                    for claims, path, desc in rank_claims:
                        res_per_desc = []
                        for c in claims:
                            v, _, _ = tools.verify_claim(c, path)
                            t = (len(set(c.lower().split()) & set(desc.lower().split())) / (len(c.split())+1e-6) > 0.7)
                            res_per_desc.append({'verdict': v, 'traceable': t})
                        rank_results.append(res_per_desc)
                    all_results_by_rank.append(rank_results)
            else:
                all_results_by_rank = None

            # Scatter 验证结果回各 rank
            local_results = [None]
            dist.scatter_object_list(local_results, all_results_by_rank, src=0)
            ver_results = local_results[0]

            mem.cleanup()

            # ============================================================
            # PHASE 3: VLM 训练（VLM 换入, Verifier 换出）
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase3-start")
            mem.reload(vlm.model, "VLM")
            mem.offload(verifier.model, "Verifier")

            all_vlm_rewards = []
            for i in range(len(images)):
                start, end = i * GROUP_SIZE, (i + 1) * GROUP_SIZE
                group_res = ver_results[start:end]
                group_txt = flat_desc[start:end]

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

            v_opt.zero_grad()
            gradient_accumulation_steps = len(images)

            for group_idx in range(len(images)):
                start, end = group_idx * GROUP_SIZE, (group_idx + 1) * GROUP_SIZE
                group_desc = flat_desc[start:end]
                group_images = flat_images[start:end]
                group_adv = v_adv[group_idx]

                group_loss = 0
                for k in range(len(group_desc)):
                    text = group_desc[k]
                    img = group_images[k]
                    msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe."}]}, {"role": "assistant", "content": [{"type": "text", "text": text}]}]
                    text_in = vlm.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                    inputs = vlm.processor(text=[text_in], images=[img], padding=True, return_tensors="pt").to(device)
                    group_loss += -group_adv[k] * vlm.compute_log_probs(inputs.input_ids, inputs.attention_mask, inputs.input_ids)

                # OOM 安全反向传播
                def _vlm_backward():
                    accelerator.backward(group_loss / (len(group_desc) * gradient_accumulation_steps))
                try:
                    mem.safe_execute(_vlm_backward, retries=1)
                except torch.cuda.OutOfMemoryError:
                    print(f"[MEM-R{mem.rank}] ❌ VLM backward OOM group {group_idx}, skipping", flush=True)
                    v_opt.zero_grad()
                    break

                torch.cuda.empty_cache()

            v_opt.step()
            v_opt.zero_grad()
            mem.cleanup()

            # ============================================================
            # PHASE 4: Verifier 训练（Verifier 换入, VLM 换出）
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase4-start")
            mem.offload(vlm.model, "VLM")
            mem.reload(verifier.model, "Verifier")

            all_ver_rewards = []
            for k, res_list in enumerate(ver_results):
                r_sum = sum(reward_calc.calculate_verifier_reward(r['verdict'], r['traceable'], ver_corr_scores[k]) for r in res_list)
                all_ver_rewards.append(r_sum / (len(res_list) if res_list else 1))

            global_ver_rew_t = torch.tensor(all_ver_rewards, device=device)
            ver_opt.zero_grad()
            gradient_accumulation_steps = len(images)

            for group_idx in range(len(images)):
                start, end = group_idx * GROUP_SIZE, (group_idx + 1) * GROUP_SIZE
                group_desc = flat_desc[start:end]
                group_raw = ver_raw_resp[start:end]
                group_results = ver_results[start:end]
                group_corr = ver_corr_scores[start:end]

                group_ver_rewards = []
                for k in range(len(group_desc)):
                    res_list = group_results[k]
                    r_sum = sum(reward_calc.calculate_verifier_reward(r['verdict'], r['traceable'], group_corr[k]) for r in res_list)
                    group_ver_rewards.append(r_sum / (len(res_list) if res_list else 1))

                ver_rew_t = torch.tensor(group_ver_rewards, device=device)
                ver_adv = (ver_rew_t - ver_rew_t.mean()) / (ver_rew_t.std() + 1e-8)

                group_ver_loss = 0
                for k in range(len(group_desc)):
                    group_ver_loss += -ver_adv[k] * verifier.compute_sequence_log_prob(group_desc[k], group_raw[k])

                # OOM 安全反向传播
                def _ver_backward():
                    accelerator.backward(group_ver_loss / (len(group_desc) * gradient_accumulation_steps))
                try:
                    mem.safe_execute(_ver_backward, retries=1)
                except torch.cuda.OutOfMemoryError:
                    print(f"[MEM-R{mem.rank}] ❌ Verifier backward OOM group {group_idx}, skipping", flush=True)
                    ver_opt.zero_grad()
                    break

                torch.cuda.empty_cache()

            ver_opt.step()
            ver_opt.zero_grad()
            mem.cleanup()

            # --- batch 结束：VLM 换回 GPU 准备下一轮 Phase 1 ---
            mem.reload(vlm.model, "VLM")

            ver_rew_display = global_ver_rew_t.mean().item()
            if accelerator.is_main_process and batch_idx % 5 == 0:
                free_mib = mem.get_free_mib()
                pbar.set_postfix({
                    "V_Rew": f"{v_rew_t.mean().item():.2f}",
                    "Ver_Rew": f"{ver_rew_display:.2f}",
                    "Free": f"{free_mib:.0f}M"
                })

        # Save Checkpoint — 两个模型都要在 GPU 上才能 save
        mem.reload(verifier.model, "Verifier")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_p = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            os.makedirs(save_p, exist_ok=True)
            accelerator.unwrap_model(vlm.model).save_pretrained(os.path.join(save_p, "vlm"))
            accelerator.unwrap_model(verifier.model).save_pretrained(os.path.join(save_p, "verifier"))
            print(f"[CKPT] Epoch {epoch} saved to {save_p}", flush=True)

if __name__ == "__main__": train()

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


# ============================================================
# GPU 资源自适应调优器
# ============================================================
class ResourceAutoTuner:
    """根据实际 GPU 显存自动推荐 batch_size / group_size / num_gen"""

    # (min_free_gb, batch_size, group_size, num_gen)
    TIERS = [
        (40, 16, 8, 10),
        (30, 12, 6,  8),
        (20,  8, 4,  6),
        (12,  4, 4,  5),
        ( 0,  2, 2,  3),
    ]

    SAFETY_MARGIN_GB = 2.0  # 安全余量

    def __init__(self, device, accelerator, mem_manager):
        self.device = device
        self.accelerator = accelerator
        self.mem = mem_manager
        self.rank = accelerator.local_process_index

    def _get_free_gb(self):
        if not torch.cuda.is_available():
            return 80.0  # 假设充足
        free, _ = torch.cuda.mem_get_info(self.device)
        return free / (1024 ** 3)

    def _pick_tier(self, free_gb):
        working = free_gb - self.SAFETY_MARGIN_GB
        for min_free, bs, gs, ng in self.TIERS:
            if working >= min_free:
                return bs, gs, ng
        # 兜底
        return 2, 2, 3

    def recommend(self, vlm_model, verifier_model,
                  user_batch_size=0, user_group_size=0, user_num_gen=0):
        """
        测量实际可用显存并返回 (batch_size, group_size, num_gen, num_workers, no_swap).
        user_xxx = 0 表示 auto, > 0 表示用户手动指定.
        no_swap: 两个模型同时在 GPU 时剩余显存 >= 10GB 则为 True.
        """
        # --- 测量 Phase 1 可用显存 (VLM on GPU, Verifier on CPU) ---
        self.mem.offload(verifier_model, "Verifier (auto-tune)")
        self.mem.reload(vlm_model, "VLM (auto-tune)")
        self.mem.cleanup()
        free_phase1 = self._get_free_gb()

        # --- 测量 Phase 3/4 可用显存 (Verifier on GPU, VLM on CPU) ---
        self.mem.offload(vlm_model, "VLM (auto-tune)")
        self.mem.reload(verifier_model, "Verifier (auto-tune)")
        self.mem.cleanup()
        free_training = self._get_free_gb()

        # --- 测量两个模型同时在 GPU 时的剩余显存 ---
        self.mem.reload(vlm_model, "VLM (auto-tune both)")
        self.mem.cleanup()
        free_both = self._get_free_gb()

        # 估算 AdamW 优化器状态内存：每个参数需要 2 个 fp32 缓冲区 (momentum + variance)
        vlm_numel = sum(p.numel() for p in self.accelerator.unwrap_model(vlm_model).parameters())
        ver_numel = sum(p.numel() for p in self.accelerator.unwrap_model(verifier_model).parameters())
        opt_state_gb = (vlm_numel + ver_numel) * 8 / (1024 ** 3)  # 2 × fp32 per param
        grad_gb = max(vlm_numel, ver_numel) * 2 / (1024 ** 3)     # bf16 梯度
        no_swap_needed = opt_state_gb + grad_gb + 10.0  # 10GB 余量给激活值
        no_swap = free_both >= no_swap_needed

        if self.accelerator.is_main_process:
            print(f"[AUTO-TUNE] no_swap check: free_both={free_both:.1f}GB, "
                  f"opt_states={opt_state_gb:.1f}GB, grads={grad_gb:.1f}GB, "
                  f"needed={no_swap_needed:.1f}GB → no_swap={no_swap}", flush=True)

        # 恢复初始状态：VLM on GPU, Verifier on CPU (Phase 1 起始状态)
        self.mem.offload(verifier_model, "Verifier (auto-tune restore)")
        self.mem.cleanup()

        # 取两阶段较小值作为约束基准
        effective_free = min(free_phase1, free_training)
        rec_bs, rec_gs, rec_ng = self._pick_tier(effective_free)

        # num_workers: CPU 核数相关
        num_workers = min(8, max(1, os.cpu_count() // 2))

        # 应用用户覆盖
        batch_size = user_batch_size if user_batch_size > 0 else rec_bs
        group_size = user_group_size if user_group_size > 0 else rec_gs
        num_gen = user_num_gen if user_num_gen > 0 else rec_ng

        # 约束校验
        if group_size < 2:
            group_size = 2
        if num_gen < group_size:
            num_gen = group_size
        if batch_size % group_size != 0:
            # 向下对齐到 group_size 的倍数
            batch_size = max(group_size, (batch_size // group_size) * group_size)

        # 用户指定值超过推荐值时发出警告
        if user_batch_size > 0 and user_batch_size > rec_bs:
            print(f"[AUTO-TUNE] WARNING: user batch_size={user_batch_size} > recommended={rec_bs}, OOM risk", flush=True)
        if user_group_size > 0 and user_group_size > rec_gs:
            print(f"[AUTO-TUNE] WARNING: user group_size={user_group_size} > recommended={rec_gs}, OOM risk", flush=True)
        if user_num_gen > 0 and user_num_gen > rec_ng:
            print(f"[AUTO-TUNE] WARNING: user num_gen={user_num_gen} > recommended={rec_ng}, OOM risk", flush=True)

        # GPU 信息
        gpu_name = torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else "N/A"
        total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0

        if self.accelerator.is_main_process:
            print(f"[AUTO-TUNE] GPU: {gpu_name} {total_gb:.0f}GB | "
                  f"Phase1 free: {free_phase1:.1f}GB | Training free: {free_training:.1f}GB | "
                  f"Both free: {free_both:.1f}GB | no_swap={no_swap}",
                  flush=True)
            print(f"[AUTO-TUNE] batch_size={batch_size}, group_size={group_size}, "
                  f"num_gen={num_gen}, num_workers={num_workers}",
                  flush=True)

        return batch_size, group_size, num_gen, num_workers, no_swap


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
    def __getitem__(self, idx, _retries=0):
        max_retries = 10
        try:
            path = os.path.join(self.root_dir, self.image_files[idx])
            return Image.open(path).convert("RGB"), path
        except Exception as e:
            if _retries >= max_retries:
                raise RuntimeError(f"Failed to load image after {max_retries} retries, last idx={idx}: {e}")
            print(f"[Dataset] Skipping corrupt file {self.image_files[idx]}: {e}", flush=True)
            return self.__getitem__((idx + 1) % len(self), _retries=_retries + 1)

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
    parser.add_argument("--batch_size", type=int, default=0, help="DataLoader batch size (0=auto)")
    parser.add_argument("--group_size", type=int, default=0, help="GRPO group size per image (0=auto)")
    parser.add_argument("--num_generations", type=int, default=0, help="VLM candidate generations per image (0=auto)")
    parser.add_argument("--attack_weight", type=float, default=5.0)
    parser.add_argument("--tool_device", type=str, default=None, help="工具GPU覆盖，逗号分隔 (如 cuda:4,cuda:5)；不指定则自动按显存分配")
    parser.add_argument("--bonus_beta", type=float, default=0.5, help="VLM reward bonus beta")
    parser.add_argument("--correlation_weight", type=float, default=2.0, help="Verifier correlation penalty weight")
    parser.add_argument("--length_threshold", type=int, default=20, help="Min description length before penalty")
    parser.add_argument("--length_penalty", type=float, default=-2.0, help="Penalty for short descriptions")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint 目录路径 (如 checkpoints/epoch_2)，或 'latest' 自动查找最新")
    args = parser.parse_args()

    # 0. tool_device 校验（支持逗号分隔多设备）
    tool_devices = None
    if args.tool_device is not None:
        tool_devices = [s.strip() for s in args.tool_device.split(",") if s.strip()]
        for td_str in tool_devices:
            try:
                td = torch.device(td_str)
                if td.type == "cuda":
                    if not torch.cuda.is_available():
                        print(f"[ERROR] --tool_device contains {td_str} but CUDA is not available", flush=True)
                        sys.exit(1)
                    if td.index is not None and td.index >= torch.cuda.device_count():
                        print(f"[ERROR] --tool_device {td_str} invalid: only {torch.cuda.device_count()} GPUs available", flush=True)
                        sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Invalid device in --tool_device: {td_str}: {e}", flush=True)
                sys.exit(1)

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

    # ============================================================
    # 断点续传：解析 resume 路径
    # ============================================================
    resume_dir = None
    resume_epoch = -1  # -1 表示从头开始，训练从 resume_epoch+1 开始
    if args.resume_from:
        if args.resume_from == "latest":
            latest_link = os.path.join(checkpoint_dir, "latest")
            if os.path.exists(latest_link):
                resume_dir = os.path.realpath(latest_link)
            else:
                # 没有 latest 链接，扫描所有 epoch_* 目录找最新的
                existing = sorted(
                    [d for d in os.listdir(checkpoint_dir)
                     if d.startswith("epoch_") and os.path.isdir(os.path.join(checkpoint_dir, d))],
                    key=lambda x: int(x.split("_")[1])
                ) if os.path.isdir(checkpoint_dir) else []
                if existing:
                    resume_dir = os.path.join(checkpoint_dir, existing[-1])
        else:
            resume_dir = os.path.abspath(args.resume_from)

        if resume_dir and os.path.isdir(resume_dir):
            state_file = os.path.join(resume_dir, "training_state.pt")
            if os.path.isfile(state_file):
                # 从 training_state.pt 读取 epoch
                _tmp = torch.load(state_file, map_location="cpu", weights_only=False)
                resume_epoch = _tmp["epoch"]
                del _tmp
                if accelerator.is_main_process:
                    print(f"[RESUME] 找到 checkpoint: {resume_dir} (epoch {resume_epoch})", flush=True)
                    print(f"[RESUME] 训练将从 epoch {resume_epoch + 1} 继续", flush=True)
            else:
                if accelerator.is_main_process:
                    print(f"[RESUME] 警告: {state_file} 不存在，无法恢复训练状态", flush=True)
                resume_dir = None
        else:
            if accelerator.is_main_process:
                print(f"[RESUME] 警告: 目录 {resume_dir} 不存在，从头开始训练", flush=True)
            resume_dir = None

    # 初始化显存管理器
    mem = GPUMemoryManager(device, accelerator)

    # 3. 顺序加载 + 同步屏障（减少峰值内存与IO争抢）
    vlm, verifier, tools, similarity_model = None, None, None, None

    # 确定模型加载路径：有 checkpoint 则从 checkpoint 加载权重，processor/tokenizer 仍用原始路径
    vlm_load_path = os.path.join(resume_dir, "vlm") if resume_dir else vlm_path
    ver_load_path = os.path.join(resume_dir, "verifier") if resume_dir else verifier_path

    try:
        # --- VLM (最大模型，优先加载) ---
        mem.log("Before VLM load")
        print(f"📦 [Rank {accelerator.local_process_index}] Loading VLM from {vlm_load_path}...", flush=True)
        vlm = VLMModel(model_name=vlm_load_path, device=device,
                        processor_name=vlm_path if resume_dir else None)
        print(f"  ✓ [Rank {accelerator.local_process_index}] VLM loaded", flush=True)
        mem.cleanup()
        mem.log("After VLM load")
        accelerator.wait_for_everyone()

        # --- Verifier ---
        print(f"📦 [Rank {accelerator.local_process_index}] Loading Verifier from {ver_load_path}...", flush=True)
        verifier = VerifierModel(model_name=ver_load_path, device=device,
                                  tokenizer_name=verifier_path if resume_dir else None)
        print(f"  ✓ [Rank {accelerator.local_process_index}] Verifier loaded", flush=True)
        mem.cleanup()
        mem.log("After Verifier load")
        accelerator.wait_for_everyone()

        # --- Tools: 仅主进程加载，自动按显存分散或用户覆盖 ---
        if accelerator.is_main_process:
            if tool_devices:
                print(f"📦 [Rank {accelerator.local_process_index}] Loading Tools (override: {tool_devices})...", flush=True)
                tools = ToolVerifier(devices=tool_devices, model_root=args.model_dir)
            else:
                print(f"📦 [Rank {accelerator.local_process_index}] Loading Tools (auto-assign by VRAM)...", flush=True)
                tools = ToolVerifier(model_root=args.model_dir)
            print(f"  ✓ Tools loaded: DINO→{tools.dino_device}, CLIP→{tools.clip_device}, OCR→{tools.ocr_device}", flush=True)
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

    reward_calc = RewardCalculator(
        attack_weight=args.attack_weight,
        bonus_beta=args.bonus_beta,
        correlation_weight=args.correlation_weight,
        length_threshold=args.length_threshold,
        length_penalty=args.length_penalty,
    )
    dataset = YFCCDataset(args.data_dir)
    print(f"[DEBUG] Dataset size: {len(dataset)} images", flush=True)
    if len(dataset) == 0:
        print(f"🛑 ERROR: No images found in {args.data_dir}", flush=True)
        sys.exit(1)

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
    vlm.model, v_opt = accelerator.prepare(vlm.model, v_opt)
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

    # ============================================================
    # 资源自适应调优：DDP 包装完成后测量实际可用显存
    # ============================================================
    tuner = ResourceAutoTuner(device, accelerator, mem)
    batch_size, GROUP_SIZE, num_gen_default, num_workers, no_swap = tuner.recommend(
        vlm.model, verifier.model,
        user_batch_size=args.batch_size,
        user_group_size=args.group_size,
        user_num_gen=args.num_generations,
    )

    # no-swap 模式：两个模型始终在 GPU，不做 offload/reload
    if no_swap:
        # recommend() 结束时 verifier 在 CPU，需要 reload
        mem.reload(verifier.model, "Verifier (no-swap init)")
        if accelerator.is_main_process:
            print("[NO-SWAP] Both models pinned on GPU, skipping offload/reload in training loop", flush=True)

    # OOM 连续计数器，用于跨 epoch 动态降级
    oom_counter = 0
    OOM_DEGRADE_THRESHOLD = 3

    # 延迟创建 DataLoader（依赖 auto-tune 的 batch_size）
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                           num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    dataloader = accelerator.prepare(dataloader)
    print(f"[DEBUG] Dataloader created, batch_size={batch_size}, group_size={GROUP_SIZE}, "
          f"num_gen={num_gen_default}, num_workers={num_workers}", flush=True)

    # ============================================================
    # 断点续传：恢复 optimizer 状态和 RNG
    # ============================================================
    start_epoch = 0
    global_step = 0
    if resume_dir:
        state_file = os.path.join(resume_dir, "training_state.pt")
        if os.path.isfile(state_file):
            if accelerator.is_main_process:
                print(f"[RESUME] 正在加载训练状态: {state_file}", flush=True)
            ckpt_state = torch.load(state_file, map_location="cpu", weights_only=False)
            start_epoch = ckpt_state["epoch"] + 1
            global_step = ckpt_state["global_step"]
            try:
                v_opt.load_state_dict(ckpt_state["v_opt_state"])
                ver_opt.load_state_dict(ckpt_state["ver_opt_state"])
                if accelerator.is_main_process:
                    print(f"[RESUME] Optimizer 状态已恢复", flush=True)
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[RESUME] 警告: Optimizer 状态加载失败 ({e})，使用新 optimizer", flush=True)
            # 恢复 RNG 状态
            if "rng_state" in ckpt_state:
                torch.set_rng_state(ckpt_state["rng_state"])
            if "cuda_rng_state" in ckpt_state and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt_state["cuda_rng_state"])
            if "np_rng_state" in ckpt_state:
                np.random.set_state(ckpt_state["np_rng_state"])
            del ckpt_state
            mem.cleanup()
            if accelerator.is_main_process:
                print(f"[RESUME] 从 epoch {start_epoch}, global_step {global_step} 继续训练", flush=True)

    # 5. 完整训练循环
    if accelerator.is_main_process:
        print(f"[INFO] Starting training loop, total batches: {len(dataloader)}", flush=True)
    accelerator.wait_for_everyone()
    print(f"[DEBUG-Rank{accelerator.local_process_index}] All ranks synchronized, starting epoch loop", flush=True)
    num_epochs = 5
    total_batches = len(dataloader)
    total_steps = num_epochs * total_batches  # 总 step 数
    # 如果是 resume，global_step 已从 checkpoint 恢复，不重置
    training_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for batch_idx, (images, image_paths) in enumerate(pbar):
            global_step += 1
            batch_start_time = time.time()

            # ============================================================
            # PHASE 1: VLM 生成
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase1-start")
            if not no_swap:
                mem.offload(verifier.model, "Verifier")

            flat_desc = []
            flat_images = []
            flat_paths = []

            with torch.no_grad():
                # 批量生成：一次调用所有图片
                num_gen = num_gen_default
                def _generate_batch():
                    return vlm.generate_description_batch(images, num_generations=num_gen)
                try:
                    all_results, _ = mem.safe_execute(_generate_batch, retries=1)
                except torch.cuda.OutOfMemoryError:
                    # OOM 降级：退回逐图生成
                    print(f"[MEM-R{mem.rank}] ⚠️ Phase1 batch OOM, falling back to per-image", flush=True)
                    oom_counter += 1
                    all_results = []
                    for idx, (img, path) in enumerate(zip(images, image_paths)):
                        def _generate_single():
                            return vlm.generate_description_batch([img], num_generations=num_gen)[0]
                        try:
                            raw, _ = mem.safe_execute(_generate_single, retries=1)
                        except torch.cuda.OutOfMemoryError:
                            num_gen_fallback = max(GROUP_SIZE, num_gen // 2)
                            print(f"[MEM-R{mem.rank}] ⚠️ Phase1 OOM img {idx}, 降级: {num_gen_fallback}", flush=True)
                            oom_counter += 1
                            try:
                                def _gen_reduced():
                                    return vlm.generate_description_batch([img], num_generations=num_gen_fallback)[0]
                                raw, _ = mem.safe_execute(_gen_reduced, retries=1)
                            except torch.cuda.OutOfMemoryError:
                                print(f"[MEM-R{mem.rank}] ⚠️ Phase1 再次 OOM, 跳过图片 {idx}", flush=True)
                                oom_counter += 1
                                all_results.append([])
                                continue
                        all_results.append(raw)

                for idx, (img, path) in enumerate(zip(images, image_paths)):
                    raw_list = all_results[idx] if idx < len(all_results) else []
                    if not raw_list:
                        continue
                    diverse = select_diverse_descriptions(raw_list, similarity_model, GROUP_SIZE)
                    flat_desc.extend(diverse)
                    flat_images.extend([img] * len(diverse))
                    flat_paths.extend([path] * len(diverse))

            if not no_swap:
                mem.cleanup()

            # ============================================================
            # PHASE 2: Verifier 提取 + 工具验证
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase2-start")
            if not no_swap:
                mem.offload(vlm.model, "VLM")
                mem.reload(verifier.model, "Verifier")

            ver_raw_resp = []
            ver_corr_scores = []
            local_claims_list = []

            with torch.no_grad():
                # 批量提取 claims
                try:
                    batch_claims, batch_raws = verifier.verify_claims_batch(flat_desc)
                except torch.cuda.OutOfMemoryError:
                    # OOM 降级：逐条提取
                    print(f"[MEM-R{mem.rank}] ⚠️ Phase2 batch OOM, falling back to per-desc", flush=True)
                    oom_counter += 1
                    batch_claims, batch_raws = [], []
                    for desc in flat_desc:
                        try:
                            claims, raw = verifier.verify_claims(desc)
                        except torch.cuda.OutOfMemoryError:
                            claims, raw = [], ""
                        batch_claims.append(claims)
                        batch_raws.append(raw)

                for i in range(len(flat_desc)):
                    claims = batch_claims[i]
                    raw = batch_raws[i]
                    ver_raw_resp.append(raw)
                    ver_corr_scores.append(calculate_intra_claim_correlation(claims, similarity_model))
                    local_claims_list.append((claims, flat_paths[i], flat_desc[i]))

            # Gather claims 到 rank 0 进行工具验证
            gathered_claims = [None] * accelerator.num_processes if accelerator.is_main_process else None
            dist.gather_object(local_claims_list, gathered_claims, dst=0)

            # Rank 0 执行批量工具验证
            if accelerator.is_main_process:
                all_results_by_rank = []
                for rank_claims in gathered_claims:
                    # 按图片分组 claims 用于批量验证
                    img_claims_map = {}  # path -> [(desc_idx, [claims])]
                    for desc_idx, (claims, path, desc) in enumerate(rank_claims):
                        if path not in img_claims_map:
                            img_claims_map[path] = []
                        img_claims_map[path].append((desc_idx, claims, desc))

                    # 构建批量验证输入
                    claims_by_image = []
                    for path, entries in img_claims_map.items():
                        all_claims_for_img = []
                        for _, claims, _ in entries:
                            all_claims_for_img.extend(claims)
                        if all_claims_for_img:
                            claims_by_image.append((path, all_claims_for_img))

                    # 批量验证
                    try:
                        batch_verdicts = tools.verify_claims_batch(claims_by_image)
                    except Exception as e:
                        print(f"[WARN] Batch tool verify failed: {e}, falling back to per-claim", flush=True)
                        batch_verdicts = {}

                    # 组装结果
                    rank_results = []
                    for desc_idx, (claims, path, desc) in enumerate(rank_claims):
                        res_per_desc = []
                        for c in claims:
                            v = batch_verdicts.get((path, c), None)
                            if v is None:
                                # fallback: 逐条验证
                                try:
                                    v, _, _ = tools.verify_claim(c, path)
                                except Exception:
                                    v = "uncertain"
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

            if not no_swap:
                mem.cleanup()

            # ============================================================
            # PHASE 3: VLM 训练
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase3-start")
            if not no_swap:
                mem.reload(vlm.model, "VLM")
                mem.offload(verifier.model, "Verifier")
            accelerator.unwrap_model(vlm.model).train()
            print(f"[MODE-R{mem.rank}] VLM → .train()", flush=True)

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

                # 批量 log-prob 计算
                try:
                    batch_log_probs = vlm.compute_log_probs_batch(group_images, group_desc)
                    group_loss = sum(-group_adv[k] * batch_log_probs[k] for k in range(len(group_desc)))
                except torch.cuda.OutOfMemoryError:
                    # OOM 降级：逐样本计算
                    print(f"[MEM-R{mem.rank}] ⚠️ Phase3 batch log-prob OOM, per-sample fallback", flush=True)
                    oom_counter += 1
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
            try:
                v_opt.step()
            except torch.cuda.OutOfMemoryError:
                print(f"[MEM-R{mem.rank}] ❌ VLM optimizer step OOM, skipping", flush=True)
                oom_counter += 1
            v_opt.zero_grad()
            if not no_swap:
                mem.cleanup()

            accelerator.unwrap_model(vlm.model).eval()
            print(f"[MODE-R{mem.rank}] VLM → .eval()", flush=True)

            # ============================================================
            # PHASE 4: Verifier 训练
            # ============================================================
            mem.log(f"E{epoch}B{batch_idx} Phase4-start")
            if not no_swap:
                mem.offload(vlm.model, "VLM")
                mem.reload(verifier.model, "Verifier")
            accelerator.unwrap_model(verifier.model).train()
            print(f"[MODE-R{mem.rank}] Verifier → .train()", flush=True)

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

                ver_rew_t_group = torch.tensor(group_ver_rewards, device=device)
                ver_adv = (ver_rew_t_group - ver_rew_t_group.mean()) / (ver_rew_t_group.std() + 1e-8)

                # 批量 log-prob 计算
                try:
                    batch_ver_log_probs = verifier.compute_sequence_log_prob_batch(group_desc, group_raw)
                    group_ver_loss = sum(-ver_adv[k] * batch_ver_log_probs[k] for k in range(len(group_desc)))
                except torch.cuda.OutOfMemoryError:
                    # OOM 降级：逐样本计算
                    print(f"[MEM-R{mem.rank}] ⚠️ Phase4 batch log-prob OOM, per-sample fallback", flush=True)
                    oom_counter += 1
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
            try:
                ver_opt.step()
            except torch.cuda.OutOfMemoryError:
                print(f"[MEM-R{mem.rank}] ❌ Verifier optimizer step OOM, skipping", flush=True)
                oom_counter += 1
            ver_opt.zero_grad()
            if not no_swap:
                mem.cleanup()

            accelerator.unwrap_model(verifier.model).eval()
            print(f"[MODE-R{mem.rank}] Verifier → .eval()", flush=True)

            # --- batch 结束 ---
            if not no_swap:
                mem.reload(vlm.model, "VLM")

            ver_rew_display = global_ver_rew_t.mean().item()
            if accelerator.is_main_process:
                batch_elapsed = time.time() - batch_start_time
                free_mib = mem.get_free_mib()

                # 计算训练进度和预估剩余时间
                elapsed = time.time() - training_start_time
                avg_step_time = elapsed / global_step
                remaining_steps = total_steps - global_step
                eta_seconds = avg_step_time * remaining_steps
                # 格式化剩余时间
                eta_h = int(eta_seconds // 3600)
                eta_m = int((eta_seconds % 3600) // 60)
                eta_s = int(eta_seconds % 60)
                progress_pct = global_step / total_steps * 100

                pbar.set_postfix({
                    "V_Rew": f"{v_rew_t.mean().item():.2f}",
                    "Ver_Rew": f"{ver_rew_display:.2f}",
                    "Free": f"{free_mib:.0f}M",
                    "Step": f"{global_step}/{total_steps}",
                    "s/it": f"{batch_elapsed:.1f}",
                    "ETA": f"{eta_h}h{eta_m:02d}m{eta_s:02d}s",
                })

                # 每 20 个 batch 输出一次详细进度日志（tqdm 进度条可能被其他日志冲掉）
                if batch_idx % 20 == 0:
                    epoch_elapsed = time.time() - epoch_start_time
                    print(f"[PROGRESS] Epoch {epoch} Batch {batch_idx}/{total_batches} | "
                          f"Global {global_step}/{total_steps} ({progress_pct:.1f}%) | "
                          f"Avg {avg_step_time:.1f}s/step | this={batch_elapsed:.1f}s | "
                          f"Epoch elapsed {epoch_elapsed/60:.1f}min | "
                          f"ETA {eta_h}h{eta_m:02d}m{eta_s:02d}s | "
                          f"V_Rew={v_rew_t.mean().item():.2f} Ver_Rew={ver_rew_display:.2f}",
                          flush=True)

        # Save Checkpoint — 两个模型都要在 GPU 上才能 save
        v_opt.zero_grad()
        ver_opt.zero_grad()
        if not no_swap:
            mem.reload(verifier.model, "Verifier")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_p = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            os.makedirs(save_p, exist_ok=True)
            # 保存模型权重 (HF format)
            accelerator.unwrap_model(vlm.model).save_pretrained(os.path.join(save_p, "vlm"))
            accelerator.unwrap_model(verifier.model).save_pretrained(os.path.join(save_p, "verifier"))
            # 保存训练状态 (optimizer + RNG + 进度)
            training_state = {
                "epoch": epoch,
                "global_step": global_step,
                "v_opt_state": v_opt.state_dict(),
                "ver_opt_state": ver_opt.state_dict(),
                "rng_state": torch.get_rng_state(),
                "np_rng_state": np.random.get_state(),
            }
            if torch.cuda.is_available():
                training_state["cuda_rng_state"] = torch.cuda.get_rng_state()
            torch.save(training_state, os.path.join(save_p, "training_state.pt"))
            # 更新 latest 符号链接
            latest_link = os.path.join(checkpoint_dir, "latest")
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(os.path.abspath(save_p), latest_link)
            print(f"[CKPT] Epoch {epoch} saved to {save_p} (latest → {save_p})", flush=True)

        # Epoch 结束汇总
        if no_swap:
            mem.cleanup()  # no-swap 模式：每 epoch 结束清理一次
        if accelerator.is_main_process:
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - training_start_time
            remaining_epochs = num_epochs - (epoch + 1)
            avg_epoch_time = total_elapsed / (epoch + 1)
            eta_epochs = avg_epoch_time * remaining_epochs
            eta_h = int(eta_epochs // 3600)
            eta_m = int((eta_epochs % 3600) // 60)
            print(f"[EPOCH] Epoch {epoch} done in {epoch_time/60:.1f}min | "
                  f"Total elapsed {total_elapsed/60:.1f}min | "
                  f"Remaining {remaining_epochs} epochs, ETA ~{eta_h}h{eta_m:02d}m",
                  flush=True)

        # --- 跨 epoch OOM 动态降级 ---
        if oom_counter >= OOM_DEGRADE_THRESHOLD:
            old_bs, old_gs, old_ng = batch_size, GROUP_SIZE, num_gen_default
            # 降一档：找到当前 tier 的下一级
            downgraded = False
            for i, (_, t_bs, t_gs, t_ng) in enumerate(ResourceAutoTuner.TIERS):
                if t_bs == batch_size and t_gs == GROUP_SIZE:
                    if i + 1 < len(ResourceAutoTuner.TIERS):
                        _, batch_size, GROUP_SIZE, num_gen_default = ResourceAutoTuner.TIERS[i + 1]
                        downgraded = True
                    break
            if not downgraded and batch_size > 2:
                # 当前不在标准 tier 上，手动减半
                GROUP_SIZE = max(2, GROUP_SIZE // 2)
                num_gen_default = max(GROUP_SIZE, num_gen_default // 2)
                batch_size = max(GROUP_SIZE, (batch_size // 2 // GROUP_SIZE) * GROUP_SIZE)
            if batch_size != old_bs or GROUP_SIZE != old_gs:
                # 重建 DataLoader
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       collate_fn=collate_fn, num_workers=num_workers,
                                       pin_memory=True, prefetch_factor=2)
                dataloader = accelerator.prepare(dataloader)
                total_batches = len(dataloader)
                total_steps = num_epochs * total_batches
                if accelerator.is_main_process:
                    print(f"[AUTO-TUNE] OOM count={oom_counter} >= {OOM_DEGRADE_THRESHOLD}, "
                          f"降级: batch_size {old_bs}->{batch_size}, "
                          f"group_size {old_gs}->{GROUP_SIZE}, "
                          f"num_gen {old_ng}->{num_gen_default}", flush=True)
            oom_counter = 0  # 重置计数器

if __name__ == "__main__": train()

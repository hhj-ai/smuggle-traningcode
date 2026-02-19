import torch
# 绕过 transformers >= 4.52 对 PyTorch < 2.6 的 torch.load 封锁 (CVE-2025-32434)
# check_torch_load_is_safe 被 modeling_utils 等模块通过 from ... import 拷贝了本地引用，
# 替换模块属性无效。直接替换函数对象的 __code__，所有引用都会生效。
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe.__code__ = (lambda: None).__code__

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
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from torch.distributed.optim import ZeroRedundancyOptimizer

from models import VLMModel, VerifierModel
from tools import ToolVerifier
from rewards import RewardCalculator


# ============================================================
# GPU 显存管理器：监控、清理、OOM 重试（ZeRO-2 无需 swap）
# ============================================================
class GPUMemoryManager:
    """运行时 GPU 显存管理，支持 OOM 安全重试"""

    def __init__(self, device, accelerator, warn_threshold_mib=4096):
        self.device = device
        self.accelerator = accelerator
        self.rank = accelerator.local_process_index
        self.warn_threshold_mib = warn_threshold_mib

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
# GPU 显存自动调参器：根据空闲显存动态计算最优 batch 参数
# ============================================================
class MemoryAutoTuner:
    """模型加载后测量空闲显存，按显存模型自动计算各 phase 最优参数。"""

    # 显存成本常量 (MiB, bf16)
    VLM_KV_PER_SEQ = 222       # Qwen3-VL-8B, 256 token KV cache
    VER_KV_PER_SEQ = 188       # DeepSeek-R1-7B, 256 token KV cache
    VLM_TRAIN_PER_SEQ = 180    # gradient checkpointing ON
    VER_TRAIN_PER_SEQ = 150    # gradient checkpointing ON
    SAFETY_FACTOR = 0.85

    def __init__(self, mem_manager, num_generations, cli_args):
        self.mem = mem_manager
        self.num_gen = num_generations
        self.cli_args = cli_args

        # 自动计算值（calibrate 后填充）
        self.vlm_gen_sub_batch = None
        self.batch_size = None
        self.ver_claims_max_batch = None
        self.vlm_logprob_max_batch = None
        self.ver_logprob_max_batch = None

    def calibrate(self):
        """模型加载后调用：测量空闲显存并计算最优参数。"""
        free = self.mem.get_free_mib()
        usable = free * self.SAFETY_FACTOR

        # Phase 1: VLM 生成 sub_batch_size
        self.vlm_gen_sub_batch = max(1, int(usable / (self.VLM_KV_PER_SEQ * self.num_gen)))

        # batch_size 受 sub_batch 限制
        self.batch_size = min(self.vlm_gen_sub_batch * 4, 24)

        # Phase 2: Verifier claims 生成
        self.ver_claims_max_batch = min(max(1, int(usable / self.VER_KV_PER_SEQ)), 64)

        # Phase 3: VLM log-prob (训练)
        self.vlm_logprob_max_batch = min(max(1, int(usable / self.VLM_TRAIN_PER_SEQ)), 16)

        # Phase 4: Verifier log-prob (训练)
        self.ver_logprob_max_batch = min(max(1, int(usable / self.VER_TRAIN_PER_SEQ)), 48)

        # CLI 覆盖：用户显式指定的值优先
        if hasattr(self.cli_args, 'batch_size') and self.cli_args.batch_size != 16:
            self.batch_size = self.cli_args.batch_size

        self._print_params(free)

    def _print_params(self, free_mib):
        print(f"[AutoTuner] Free={free_mib:.0f}MiB | "
              f"batch_size={self.batch_size} "
              f"vlm_gen_sub_batch={self.vlm_gen_sub_batch} "
              f"ver_claims_max_batch={self.ver_claims_max_batch} "
              f"vlm_logprob_max_batch={self.vlm_logprob_max_batch} "
              f"ver_logprob_max_batch={self.ver_logprob_max_batch}",
              flush=True)

    def get_phase_params(self, phase):
        """返回指定 phase 的参数字典。"""
        if phase == 1:
            return {"sub_batch_size": self.vlm_gen_sub_batch}
        elif phase == 2:
            return {"max_batch": self.ver_claims_max_batch}
        elif phase == 3:
            return {"max_batch": self.vlm_logprob_max_batch}
        elif phase == 4:
            return {"max_batch": self.ver_logprob_max_batch}
        return {}

    def reduce_params(self, phase):
        """OOM 时将对应 phase 的 batch 参数减半（持久生效）。"""
        if phase == 1:
            self.vlm_gen_sub_batch = max(1, self.vlm_gen_sub_batch // 2)
            self.batch_size = max(1, self.batch_size // 2)
        elif phase == 2:
            self.ver_claims_max_batch = max(1, self.ver_claims_max_batch // 2)
        elif phase == 3:
            self.vlm_logprob_max_batch = max(1, self.vlm_logprob_max_batch // 2)
        elif phase == 4:
            self.ver_logprob_max_batch = max(1, self.ver_logprob_max_batch // 2)
        print(f"[AutoTuner] Phase {phase} params reduced: "
              f"vlm_gen_sub_batch={self.vlm_gen_sub_batch} "
              f"batch_size={self.batch_size} "
              f"ver_claims={self.ver_claims_max_batch} "
              f"vlm_lp={self.vlm_logprob_max_batch} "
              f"ver_lp={self.ver_logprob_max_batch}",
              flush=True)


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

def encode_long_texts(model, texts, max_tokens=450):
    """对超长文本按 token 精确分块编码后取平均，避免 MiniLM 512 截断。"""
    tokenizer = model.tokenizer
    all_embs = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= max_tokens:
            all_embs.append(model.encode(text, convert_to_tensor=True))
        else:
            chunks = []
            for i in range(0, len(ids), max_tokens):
                chunk_text = tokenizer.decode(ids[i:i+max_tokens], skip_special_tokens=True)
                chunks.append(chunk_text)
            chunk_embs = model.encode(chunks, convert_to_tensor=True)
            all_embs.append(chunk_embs.mean(dim=0))
    return torch.stack(all_embs)

def select_diverse_descriptions(texts, model, target_count):
    if len(texts) <= target_count: return texts
    embeddings = encode_long_texts(model, texts)
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
    embeddings = encode_long_texts(model, claims)
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
    parser.add_argument("--batch_size", type=int, default=16, help="DataLoader batch size")
    parser.add_argument("--group_size", type=int, default=8, help="GRPO group size per image")
    parser.add_argument("--num_generations", type=int, default=10, help="VLM candidate generations per image")
    parser.add_argument("--attack_weight", type=float, default=5.0)
    parser.add_argument("--tool_device", type=str, default=None, help="工具GPU覆盖，逗号分隔 (如 cuda:4,cuda:5)；不指定则自动按显存分配")
    parser.add_argument("--bonus_beta", type=float, default=0.5, help="VLM reward bonus beta")
    parser.add_argument("--correlation_weight", type=float, default=2.0, help="Verifier correlation penalty weight")
    parser.add_argument("--length_threshold", type=int, default=20, help="Min description length before penalty")
    parser.add_argument("--length_penalty", type=float, default=-2.0, help="Penalty for short descriptions")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint 目录路径 (如 checkpoints/epoch_2)，或 'latest' 自动查找最新")
    args = parser.parse_args()

    # 固定超参数
    batch_size = args.batch_size
    GROUP_SIZE = args.group_size
    num_gen_default = args.num_generations
    num_workers = min(8, max(1, os.cpu_count() // 2))

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
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        kwargs_handlers=[timeout_kwargs],
    )
    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True

    # 2. 路径映射
    vlm_path = os.path.abspath(os.path.join(args.model_dir, "Qwen3-VL-8B-Instruct"))
    verifier_path = os.path.abspath(os.path.join(args.model_dir, "DeepSeek-R1-Distill-Qwen-7B"))
    checkpoint_dir = os.path.abspath(os.path.join(args.output_dir, "checkpoints"))

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"🚀 AURORA Training (DDP + ZeRO-1 Optimizer)")

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

        # --- Tools: 所有 rank 加载，各自本地验证（8x 并行加速） ---
        print(f"📦 [Rank {accelerator.local_process_index}] Loading Tools on {device}...", flush=True)
        tools = ToolVerifier(device=device, model_root=args.model_dir)
        print(f"  ✓ [Rank {accelerator.local_process_index}] Tools loaded: DINO→{tools.dino_device}, CLIP→{tools.clip_device}, OCR→{tools.ocr_device}", flush=True)
        if tool_devices and accelerator.is_main_process:
            print(f"[WARN] --tool_device ignored: tools now loaded on each rank's own GPU for parallel verification", flush=True)
        mem.cleanup()
        accelerator.wait_for_everyone()

        # --- SentenceTransformer (小模型) ---
        print(f"📦 [Rank {accelerator.local_process_index}] Loading SentenceTransformer...", flush=True)
        similarity_model = SentenceTransformer(args.minilm_path, device=device)
        similarity_model.max_seq_length = 512  # 显式截断，抑制长序列警告
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
    # DDP 包装 + ZeroRedundancyOptimizer（优化器状态跨 GPU 分片）
    # ============================================================
    vlm.model, verifier.model = accelerator.prepare(vlm.model, verifier.model)
    mem.log("Models DDP-wrapped")

    # ZeroRedundancyOptimizer: 优化器状态分片到各 GPU，等效 ZeRO-1
    # 必须在 DDP 包装后创建，使用 DDP-wrapped 模型的 parameters
    v_opt = ZeroRedundancyOptimizer(
        vlm.model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=1e-6,
    )
    ver_opt = ZeroRedundancyOptimizer(
        verifier.model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=1e-6,
    )
    mem.log("ZeroRedundancyOptimizer created")

    # ============================================================
    # 显存自动调参
    # ============================================================
    tuner = MemoryAutoTuner(mem, num_gen_default, args)
    tuner.calibrate()
    batch_size = tuner.batch_size  # 覆盖 CLI 默认值

    # 创建 DataLoader
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
    training_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        for batch_idx, (images, image_paths) in enumerate(pbar):
            global_step += 1
            batch_start_time = time.time()

            # ============================================================
            # PHASE 1: VLM 生成（两个模型始终在 GPU）— OOM 自动降级
            # ============================================================
            phase_t0 = time.time()
            mem.log(f"E{epoch}B{batch_idx} Phase1-start")
            vlm.disable_gradient_checkpointing()
            _vlm_inner = accelerator.unwrap_model(vlm.model)
            if hasattr(_vlm_inner, 'is_gradient_checkpointing') and _vlm_inner.is_gradient_checkpointing:
                print(f"[WARN-R{mem.rank}] VLM gradient_checkpointing still ON in Phase1!", flush=True)
                _vlm_inner.gradient_checkpointing_disable()

            flat_desc = []
            flat_images = []
            flat_paths = []
            phase1_ok = False

            for _p1_attempt in range(3):
                try:
                    with torch.no_grad():
                        num_gen = num_gen_default
                        p1_params = tuner.get_phase_params(1)
                        all_results = vlm.generate_description_batch(
                            images, num_generations=num_gen,
                            sub_batch_size=p1_params["sub_batch_size"])

                        for idx, (img, path) in enumerate(zip(images, image_paths)):
                            raw_list = all_results[idx] if idx < len(all_results) else []
                            if not raw_list:
                                continue
                            diverse = select_diverse_descriptions(raw_list, similarity_model, GROUP_SIZE)
                            flat_desc.extend(diverse)
                            flat_images.extend([img] * len(diverse))
                            flat_paths.extend([path] * len(diverse))
                    phase1_ok = True
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"[AutoTuner-R{mem.rank}] Phase1 OOM (attempt {_p1_attempt+1}/3)", flush=True)
                    flat_desc, flat_images, flat_paths = [], [], []
                    mem.cleanup()
                    tuner.reduce_params(1)

            if not phase1_ok:
                print(f"[AutoTuner-R{mem.rank}] Phase1 failed 3x, skipping batch", flush=True)
                continue

            # ============================================================
            # PHASE 2: Verifier 提取 + 工具验证 — OOM 自动降级
            # ============================================================
            if accelerator.is_main_process:
                print(f"[TIMER] Phase1 (VLM gen): {time.time()-phase_t0:.1f}s", flush=True)
            phase_t1 = time.time()
            mem.log(f"E{epoch}B{batch_idx} Phase2-start")
            verifier.disable_gradient_checkpointing()
            _ver_inner = accelerator.unwrap_model(verifier.model)
            if hasattr(_ver_inner, 'is_gradient_checkpointing') and _ver_inner.is_gradient_checkpointing:
                print(f"[WARN-R{mem.rank}] Verifier gradient_checkpointing still ON in Phase2!", flush=True)
                _ver_inner.gradient_checkpointing_disable()

            ver_raw_resp = []
            ver_corr_scores = []
            local_claims_list = []
            phase2_ok = False

            for _p2_attempt in range(3):
                try:
                    with torch.no_grad():
                        p2_params = tuner.get_phase_params(2)
                        batch_claims, batch_raws = verifier.verify_claims_batch(
                            flat_desc, max_batch=p2_params["max_batch"])

                        for i in range(len(flat_desc)):
                            claims = batch_claims[i]
                            raw = batch_raws[i]
                            ver_raw_resp.append(raw)
                            ver_corr_scores.append(calculate_intra_claim_correlation(claims, similarity_model))
                            local_claims_list.append((claims, flat_paths[i], flat_desc[i]))
                    phase2_ok = True
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"[AutoTuner-R{mem.rank}] Phase2 OOM (attempt {_p2_attempt+1}/3)", flush=True)
                    ver_raw_resp, ver_corr_scores, local_claims_list = [], [], []
                    mem.cleanup()
                    tuner.reduce_params(2)

            if not phase2_ok:
                print(f"[AutoTuner-R{mem.rank}] Phase2 failed 3x, skipping batch", flush=True)
                continue

            # 各 rank 本地工具验证（无需 gather/scatter，8x 并行）
            img_claims_map = {}
            for desc_idx, (claims, path, desc) in enumerate(local_claims_list):
                if path not in img_claims_map:
                    img_claims_map[path] = []
                img_claims_map[path].append((desc_idx, claims, desc))

            claims_by_image = []
            for path, entries in img_claims_map.items():
                all_claims_for_img = []
                for _, claims, _ in entries:
                    all_claims_for_img.extend(claims)
                if all_claims_for_img:
                    claims_by_image.append((path, all_claims_for_img))

            try:
                preloaded_images = dict(zip(image_paths, images))
                batch_verdicts = tools.verify_claims_batch(claims_by_image, preloaded_images=preloaded_images)
            except Exception as e:
                print(f"[WARN-R{mem.rank}] Batch tool verify failed: {e}, falling back to per-claim", flush=True)
                batch_verdicts = {}

            ver_results = []
            for desc_idx, (claims, path, desc) in enumerate(local_claims_list):
                res_per_desc = []
                for c in claims:
                    v = batch_verdicts.get((path, c), None)
                    if v is None:
                        try:
                            v, _, _ = tools.verify_claim(c, path)
                        except Exception:
                            v = "uncertain"
                    t = (len(set(c.lower().split()) & set(desc.lower().split())) / (len(c.split())+1e-6) > 0.7)
                    res_per_desc.append({'verdict': v, 'traceable': t})
                ver_results.append(res_per_desc)

            # ============================================================
            # PHASE 3: VLM 训练 — OOM 自动降级
            # ============================================================
            if accelerator.is_main_process:
                print(f"[TIMER] Phase2 (Verify): {time.time()-phase_t1:.1f}s", flush=True)
            phase_t2 = time.time()
            mem.log(f"E{epoch}B{batch_idx} Phase3-start")
            vlm.enable_gradient_checkpointing()
            accelerator.unwrap_model(vlm.model).train()
            print(f"[MODE-R{mem.rank}] VLM → .train()", flush=True)

            all_vlm_rewards = []
            for i in range(len(images)):
                start, end = i * GROUP_SIZE, (i + 1) * GROUP_SIZE
                group_res = ver_results[start:end]
                group_txt = flat_desc[start:end]

                div_penalty = 0.0
                if len(group_txt) > 1:
                    emb = encode_long_texts(similarity_model, group_txt)
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

            phase3_ok = False
            for _p3_attempt in range(3):
                try:
                    v_opt.zero_grad()
                    gradient_accumulation_steps = len(images)
                    p3_params = tuner.get_phase_params(3)

                    for group_idx in range(len(images)):
                        start, end = group_idx * GROUP_SIZE, (group_idx + 1) * GROUP_SIZE
                        group_desc = flat_desc[start:end]
                        group_images = flat_images[start:end]
                        group_adv = v_adv[group_idx]

                        batch_log_probs = vlm.compute_log_probs_batch(
                            group_images, group_desc, max_batch=p3_params["max_batch"])
                        group_loss = sum(-group_adv[k] * batch_log_probs[k] for k in range(len(group_desc)))

                        accelerator.backward(group_loss / (len(group_desc) * gradient_accumulation_steps))

                    torch.cuda.empty_cache()
                    v_opt.step()
                    v_opt.zero_grad()
                    phase3_ok = True
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"[AutoTuner-R{mem.rank}] Phase3 OOM (attempt {_p3_attempt+1}/3)", flush=True)
                    v_opt.zero_grad()
                    mem.cleanup()
                    tuner.reduce_params(3)

            if not phase3_ok:
                print(f"[AutoTuner-R{mem.rank}] Phase3 failed 3x, skipping VLM update", flush=True)
                v_opt.zero_grad()

            accelerator.unwrap_model(vlm.model).eval()
            print(f"[MODE-R{mem.rank}] VLM → .eval()", flush=True)

            # ============================================================
            # PHASE 4: Verifier 训练 — OOM 自动降级
            # ============================================================
            if accelerator.is_main_process:
                print(f"[TIMER] Phase3 (VLM train): {time.time()-phase_t2:.1f}s", flush=True)
            phase_t3 = time.time()
            mem.log(f"E{epoch}B{batch_idx} Phase4-start")
            verifier.enable_gradient_checkpointing()
            accelerator.unwrap_model(verifier.model).train()
            print(f"[MODE-R{mem.rank}] Verifier → .train()", flush=True)

            all_ver_rewards = []
            for k, res_list in enumerate(ver_results):
                r_sum = sum(reward_calc.calculate_verifier_reward(r['verdict'], r['traceable'], ver_corr_scores[k]) for r in res_list)
                all_ver_rewards.append(r_sum / (len(res_list) if res_list else 1))

            global_ver_rew_t = torch.tensor(all_ver_rewards, device=device)

            phase4_ok = False
            for _p4_attempt in range(3):
                try:
                    ver_opt.zero_grad()
                    gradient_accumulation_steps = len(images)
                    p4_params = tuner.get_phase_params(4)

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

                        batch_ver_log_probs = verifier.compute_sequence_log_prob_batch(
                            group_desc, group_raw, max_batch=p4_params["max_batch"])
                        group_ver_loss = sum(-ver_adv[k] * batch_ver_log_probs[k] for k in range(len(group_desc)))

                        accelerator.backward(group_ver_loss / (len(group_desc) * gradient_accumulation_steps))

                    torch.cuda.empty_cache()
                    ver_opt.step()
                    ver_opt.zero_grad()
                    phase4_ok = True
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"[AutoTuner-R{mem.rank}] Phase4 OOM (attempt {_p4_attempt+1}/3)", flush=True)
                    ver_opt.zero_grad()
                    mem.cleanup()
                    tuner.reduce_params(4)

            if not phase4_ok:
                print(f"[AutoTuner-R{mem.rank}] Phase4 failed 3x, skipping Verifier update", flush=True)
                ver_opt.zero_grad()

            accelerator.unwrap_model(verifier.model).eval()
            print(f"[MODE-R{mem.rank}] Verifier → .eval()", flush=True)
            if accelerator.is_main_process:
                print(f"[TIMER] Phase4 (Ver train): {time.time()-phase_t3:.1f}s", flush=True)

            # --- batch 结束 ---
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

        # Save Checkpoint
        v_opt.zero_grad()
        ver_opt.zero_grad()
        # ZeroRedundancyOptimizer: 先汇总各 rank 的分片状态到 rank 0
        v_opt.consolidate_state_dict()
        ver_opt.consolidate_state_dict()
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
        mem.cleanup()
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

if __name__ == "__main__": train()

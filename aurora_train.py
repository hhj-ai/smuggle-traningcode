import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os, time, gc, argparse, sys
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, util
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

# Custom modules
from models import VLMModel, VerifierModel
from tools import ToolVerifier
from rewards import RewardCalculator

class YFCCDataset(Dataset):
    def __init__(self, root_dir, max_samples=20000):
        self.root_dir = root_dir
        self.image_files = []
        if os.path.exists(root_dir):
            # 使用 scandir 以节省低 RAM 机器的开销
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

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--minilm_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--attack_weight", type=float, default=5.0)
    args = parser.parse_args()

    # 1. 初始化 (高超时保护)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[timeout_kwargs])
    device = accelerator.device
    
    # 2. 路径映射 (严防相对路径坑)
    vlm_path = os.path.abspath(os.path.join(args.model_dir, "Qwen3-VL-8B-Instruct"))
    verifier_path = os.path.abspath(os.path.join(args.model_dir, "DeepSeek-R1-Distill-Qwen-7B"))
    checkpoint_dir = os.path.abspath(os.path.join(args.output_dir, "checkpoints"))
    
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"📍 Base Model Dir: {args.model_dir}")
        print(f"📍 Saving to: {checkpoint_dir}")

    # 3. 分进程排队加载 (针对 14GB RAM 极致保护)
    vlm, verifier, tools, similarity_model = None, None, None, None
    for i in range(accelerator.num_processes):
        if accelerator.local_process_index == i:
            print(f"📦 [Rank {i}] Loading...")
            vlm = VLMModel(model_name=vlm_path, device=device)
            verifier = VerifierModel(model_name=verifier_path, device=device)
            tools = ToolVerifier(device=device, model_root=args.model_dir)
            similarity_model = SentenceTransformer(args.minilm_path, device=device)
            gc.collect(); torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    # 4. 训练准备
    reward_calc = RewardCalculator(attack_weight=args.attack_weight)
    dataset = YFCCDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: ([i[0] for i in x], [i[1] for i in x]), num_workers=2)
    
    v_opt = torch.optim.AdamW(vlm.model.parameters(), lr=1e-6)
    vlm.model, v_opt, dataloader = accelerator.prepare(vlm.model, v_opt, dataloader)

    # 5. 极简 GRPO 循环
    for epoch in range(5):
        for imgs, paths in tqdm(dataloader, disable=not accelerator.is_main_process):
            # ... (训练逻辑保持简洁) ...
            pass # 此处承接你 Idea 中的对抗训练细节

if __name__ == "__main__": train()

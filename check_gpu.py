import torch
import torch.distributed as dist

def diagnose_gpu():
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用！")
        return

    # 获取当前进程的 Rank
    if dist.is_initialized():
        rank = dist.get_rank()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        rank = 0
        device = torch.device("cuda:0")

    # 1. 检查当前设备
    print(f"--- [Rank {rank}] 诊断报告 ---")
    print(f"当前设备: {device}")
    
    # 2. 内存细分查看
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3

    print(f"已分配显存 (Allocated): {allocated:.2f} GB")
    print(f"已预留显存 (Reserved): {reserved:.2f} GB")
    print(f"历史峰值显存 (Max Allocated): {max_allocated:.2f} GB")

    # 3. 简单的矩阵运算测试 (确认算力是否正常)
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"矩阵运算测试: 成功完成 (Result device: {z.device})")

if __name__ == "__main__":
    # 如果你在 DDP 环境下跑，需要初始化
    try:
        dist.init_process_group(backend='nccl')
        diagnose_gpu()
    except:
        diagnose_gpu()

# utils/ddp_utils.py
import os
import torch
import torch.distributed as dist
import numpy as np
import random


def setup_ddp():
    """
    初始化 DDP 环境。
    如果检测到环境变量 RANK 和 WORLD_SIZE，则初始化进程组。
    返回: is_ddp (bool)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])

            # 设置当前设备
            torch.cuda.set_device(local_rank)

            # 初始化进程组
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank
            )
            dist.barrier()  # 等待所有进程就绪
            print(f"[DDP] Rank {rank}/{world_size} (Local {local_rank}) initialized.")
            return True
        except Exception as e:
            print(f"[DDP Error] Initialization failed: {e}")
            return False
    else:
        print("[DDP] Not using Distributed Data Parallel (Single GPU/CPU mode).")
        return False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor, average=True):
    """
    用于平均所有 GPU 上的 Loss，用于打印 Log。
    不要用于 Backward，DDP 会自动处理。
    """
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= dist.get_world_size()
    return rt


def sum_tensor(tensor):
    """
    对所有 GPU 上的 Tensor 求和（常用于聚合混淆矩阵）
    """
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def setup_seed(seed=42):
    """DDP 友好的 Seed 设置"""
    # 在 DDP 中，种子需要一致以初始化相同的模型权重
    # 但 DataLoader 的 worker_init_fn 需要处理不同的 worker
    seed = seed + get_rank()  # 可选：如果希望数据增强在不同卡上不同，通常 Sampler 负责 shuffle，这里保持基础种子一致更好，除了 rank 偏移
    # 实际上，为了保证模型初始化参数一致，主种子应该相同。
    # 为了保证数据不同，DistributedSampler 会处理 indices。
    # 这里我们只设置基础种子，不加 rank，让 DDP 广播权重或加载相同时自动对齐。
    # *修正*：模型初始化必须一致，所以 seed 不能加 rank。
    # 数据加载的随机性由 DistributedSampler(seed) 和 epoch 决定。

    # 但是，如果代码中有其他 random 调用，我们希望它们在不同卡上不同吗？
    # 通常是的。但在初始化阶段，我们需要相同的权重。
    # 最佳实践：set_seed(base_seed) -> init model -> set_seed(base_seed + rank)

    # 简单起见，我们设置固定的种子，依赖 DistributedSampler 处理数据随机性。
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics_from_cm(cm):
    """
    从混淆矩阵 (5x5) 计算 Acc, Macro-F1, Kappa, Per-Class F1。
    cm: numpy array or torch tensor
    """
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()

    total = np.sum(cm)
    correct = np.trace(cm)
    acc = correct / total if total > 0 else 0.0

    # Per-class F1
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    f1s = []
    # 假设 5 类
    for i in range(len(tp)):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)

    macro_f1 = np.mean(f1s)

    # Kappa 计算
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)
    pe = np.sum(row_sum * col_sum) / (total * total) if total > 0 else 0
    po = acc
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    return acc, macro_f1, kappa, f1s
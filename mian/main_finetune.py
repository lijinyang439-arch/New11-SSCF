import argparse
import os
import sys
import random
import yaml
import shutil
import importlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

# 引入 DDP 工具
from utils.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, setup_seed as ddp_setup_seed

# ==========================================
# 核心黑科技：自定义 Loader 以支持目标域适应
# 逻辑：训练集 = 源域 + 少量目标域(强采样)
#       验证集 = 全量目标域 (用于筛选模型)
#       测试集 = 全量目标域 (用于最终输出)
# ==========================================
import datasets.dataset as raw_dataset_module
from datasets.dataset import LoadDataset as OriginalLoadDataset, CustomDataset


class FinetuneAdaptationLoader(OriginalLoadDataset):
    def __init__(self, params):
        super().__init__(params)
        self.ft_params = getattr(params, 'finetune', {})
        self.adapt_ratio = self.ft_params.get('adaptation_ratio', 0.0)
        self.target_domain_name = self.ft_params.get('target_domain', None)

        if self.target_domain_name:
            self.mode = 'lodo'
            # 目标域路径
            self.targets_dirs = [f'{self.datasets_dir}/{self.target_domain_name}']
            # 源域路径（排除目标域）
            self.source_dirs = [f'{self.datasets_dir}/{item}' for item in self.datasets_map.keys()
                                if item != self.target_domain_name]

    def get_data_loader(self):
        # 1. 加载所有路径
        source_pairs, next_sid = self.load_path(self.source_dirs, 0)
        target_pairs, final_sid = self.load_path(self.targets_dirs, next_sid)

        # 【核心定义】全量目标域数据
        target_test_pairs = target_pairs

        if is_main_process():
            print(f"[Loader] 源域样本数: {len(source_pairs)}")
            print(f"[Loader] 目标域 ({self.target_domain_name}) 全量样本: {len(target_test_pairs)}")

        # 2. 构建训练集 (Source + Target Subset)
        source_train_pairs = source_pairs
        target_train_pairs = []

        # 如果开启适应 (adapt_ratio > 0)，从目标域切分出一小部分用于训练
        if self.adapt_ratio > 0 and target_pairs:
            # 复制并打乱，用于提取训练样本，不影响 target_test_pairs 的完整性
            shuffled_target = target_pairs[:]
            random.seed(42)
            random.shuffle(shuffled_target)

            num_adapt = int(len(shuffled_target) * self.adapt_ratio)
            num_adapt = max(2, num_adapt)  # 至少2个样本用于Backward

            target_train_pairs = shuffled_target[:num_adapt]

            if is_main_process():
                print(f"[Adaptation] 目标域数据划分:")
                print(f"  - 参与微调训练 (Train): {len(target_train_pairs)}")
                print(f"  - 用于验证/测试 (Val/Test): {len(target_test_pairs)} (全量)")

        # 3. 【V5 核心】混合训练集与强采样平衡
        final_train_pairs = []
        if len(target_train_pairs) > 0:
            # 计算不平衡倍率
            repeat_factor = len(source_train_pairs) // len(target_train_pairs)
            repeat_factor = max(1, repeat_factor)

            if is_main_process():
                print(f"[Adaptation] 执行强采样平衡 (倍率: {repeat_factor})")

            # 复制目标域样本以平衡源域数量
            target_train_pairs_balanced = target_train_pairs * repeat_factor

            # 补齐剩余差距
            remainder = len(source_train_pairs) - len(target_train_pairs_balanced)
            if remainder > 0:
                target_train_pairs_balanced += target_train_pairs[:remainder]

            final_train_pairs = source_train_pairs + target_train_pairs_balanced
        else:
            final_train_pairs = source_train_pairs

        # 打乱最终训练集
        random.shuffle(final_train_pairs)

        if is_main_process():
            print(f"[Loader] 最终混合训练集大小: {len(final_train_pairs)}")

        # 4. 构建 DataLoader
        batch_size = getattr(self.params, 'batch_size', 32)
        default_workers = max(4, os.cpu_count() // 4)
        num_workers = getattr(self.params, 'num_workers', default_workers)
        if dist.is_initialized(): num_workers = min(num_workers, 8)

        loader_args = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': num_workers > 0,
        }

        # 【关键修改】
        # 将 val_set 指向 target_test_pairs (全量目标域)。
        # 这样 Trainer 在执行 validate() 时，实际上是在评估全量测试集。
        # BestKeeper 将根据这个全量测试集的指标来保存模型。
        val_set = CustomDataset(target_test_pairs)
        test_set = CustomDataset(target_test_pairs)

        return {
            'train': self._make_loader(CustomDataset(final_train_pairs), True, loader_args),
            'val': self._make_loader(val_set, False, loader_args),  # Val -> 全量 Target
            'test': self._make_loader(test_set, False, loader_args),  # Test -> 全量 Target
        }, final_sid


# 替换原始 Loader
raw_dataset_module.LoadDataset = FinetuneAdaptationLoader


# ==========================================
# 主程序
# ==========================================

def load_config(config_path: str) -> dict:
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='SleepDG Target Adaptation')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze feature encoder')
    args = parser.parse_args()

    # 1. DDP 初始化
    is_ddp = setup_ddp()
    config = load_config(args.config)
    ft_config = config.get('finetune', {})

    if not ft_config.get('enabled', False):
        if is_main_process(): print("错误: 配置文件中未开启 finetune")
        return

    # 2. 参数覆盖
    config['lr'] = ft_config.get('lr', config.get('lr', 1e-5))
    config['epochs'] = ft_config.get('epochs', config.get('epochs', 50))
    config['target_domains'] = ft_config.get('target_domain')  # 用于 Loader 识别

    ddp_setup_seed(config.get('seed', 42))
    torch.backends.cudnn.benchmark = True

    # 3. 结果目录设置
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = config.get('run_name', 'finetune')
    target_name = ft_config.get('target_domain')

    # 结果存放在 results/run_name_target_time
    results_basedir = Path(config.get('results_root', './results')) / f"{run_name}_{target_name}_{ts}"

    if is_main_process():
        results_basedir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"  SleepDG 目标域适应 (Target Adaptation)")
        print(f"  Mode: 验证集 = 全量目标域测试集 (Test on All Target)")
        print(f"  Strategy: 保存 Val(Test) 指标最高的模型")
        print(f"  Target: {target_name}")
        print(f"  Epochs: {config['epochs']}, LR: {config['lr']}")
        print(f"{'=' * 60}\n")

    if is_ddp: dist.barrier()

    config['model_dir'] = str(results_basedir)
    config['fold'] = 0
    config['is_ddp'] = is_ddp
    config['num_domains'] = 4  # 占位，具体由 Loader 处理

    params = SimpleNamespace(**config)

    try:
        # 4. 初始化 Trainer
        # 注意：此时 LoadDataset 已经被替换为 FinetuneAdaptationLoader
        trainer_module = importlib.import_module('original.trainer')
        TrainerClass = trainer_module.Trainer
        trainer = TrainerClass(params)

        # 5. 加载预训练权重 (Pretrained Weights)
        if is_ddp and hasattr(trainer.model, 'module'):
            raw_model = trainer.model.module
        else:
            raw_model = trainer.model

        pretrained_path = ft_config.get('pretrained_path')
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"预训练模型路径无效: {pretrained_path}")

        map_loc = {'cuda:%d' % 0: 'cuda:%d' % trainer.local_rank} if is_ddp else trainer.device
        checkpoint = torch.load(pretrained_path, map_location=map_loc, weights_only=False)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint

        # 去除 module. 前缀以兼容
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        raw_model.load_state_dict(new_state_dict, strict=False)

        if is_main_process():
            print(f"[Model] 已加载预训练权重: {pretrained_path}")

        # 6. 冻结层设置 (Freeze Encoder)
        if args.freeze_encoder:
            if is_main_process(): print("[Model] 正在冻结 Encoder，仅训练 Classifier...")
            for name, param in raw_model.named_parameters():
                if 'encoder' in name or 'feature' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # 确保分类头可训练

        # 打印可训练参数量
        if is_main_process():
            trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            print(f"[Model] 当前可训练参数量: {trainable_params}")

        # 7. DDP 模型封装
        if is_ddp:
            trainer.model = DDP(
                raw_model,
                device_ids=[trainer.local_rank],
                output_device=trainer.local_rank,
                find_unused_parameters=True
            )

        # 8. 优化器与调度器重置 (针对微调参数)
        trainer.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, trainer.model.parameters()),
            lr=params.lr,
            weight_decay=1e-4
        )

        train_len = len(trainer.data_loader['train'])
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=params.epochs * train_len
        )

        # 9. 重置 BestKeeper
        # 这是实现“按测试最优保持模型”的关键。
        # Trainer 默认会去比较 val_acc。因为我们把 Val Loader 指向了 Test Set，
        # 所以这里实际上是在比较 Test Set 的 Accuracy。
        if is_main_process() and hasattr(trainer, 'best_keeper'):
            trainer.best_keeper.val_best = -1.0
            print("[Trainer] BestKeeper 已重置，将依据全量 Test Set 指标筛选模型")

        # 10. 开始训练 (Finetune Loop)
        # 过程：Train on Mixed -> Validate on Full Target -> Save Best
        trainer.train()

        # 11. 【新增】加载最优模型进行最终确认
        if is_main_process():
            print(f"\n{'=' * 40}")
            print(f"  训练结束，加载 Best Model 进行最终测试")
            print(f"{'=' * 40}")

        # 等待主进程保存文件
        if is_ddp: dist.barrier()

        # 加载刚才保存的 best_model.pth
        best_model_path = results_basedir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(str(best_model_path), map_location=map_loc, weights_only=False)
            state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            raw_model.load_state_dict(new_state_dict, strict=False)
            if is_main_process():
                print(f"[Result] 成功加载 Best Model: {best_model_path}")

            # 运行 test() 打印最终混淆矩阵
            # 注意：这里的 test loader 也是全量 Target
            trainer.test()
        else:
            if is_main_process():
                print("[Warning] 未找到 best_model.pth，跳过最终测试")

    except Exception as e:
        print(f"[Error] 发生异常: {e}")
        import traceback
        traceback.print_exc()

    cleanup_ddp()


if __name__ == '__main__':
    main()
import os
import copy
import logging
from datetime import datetime
from timeit import default_timer as timer
from pathlib import Path
from typing import Dict, Tuple, Optional
from types import SimpleNamespace
import importlib
import re

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- DDP 相关导入 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.ddp_utils import is_main_process, reduce_tensor, sum_tensor, compute_metrics_from_cm

# --- 项目内模块 ---
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from losses.ae_loss import AELoss
from utils.allutils import (
    ensure_dir, MetricsLogger, build_ratio_loader,
    plot_curves_for_fold, extract_features_for_tsne, tsne_compare_plot,
    write_aggregate_row, _now
)
from losses.psd_geometric_loss import PSDGeometricLoss
from losses.double_alignment import CORAL


# 占位 BestKeeper
class BestKeeper:
    def __init__(self, *args, **kwargs): pass

    def update_val(self, *args, **kwargs): return False

    def update_test(self, *args, **kwargs): return False


# --- ANSI 颜色 ---
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


# =========================================================
#                 高性能数据预取器 (DataPrefetcher)
# =========================================================
class DataPrefetcher:
    """使用 CUDA Stream 异步将数据从 CPU 搬运到 GPU"""

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = [
                t.cuda(non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in self.next_batch
            ]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            self.preload()
        return batch


class Trainer(object):
    def __init__(self, params: SimpleNamespace):
        self.params = params
        self.is_ddp = getattr(params, 'is_ddp', False)

        # --- 设备设置 (DDP) ---
        if self.is_ddp:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 目录 (仅主进程创建) ---
        self.model_dir = Path(params.model_dir)
        self.fold_id = params.fold
        self.fold_dir = self.model_dir / f"fold{self.fold_id}"
        self.allfold_dir = self.model_dir / "allfold"

        if is_main_process():
            ensure_dir(self.fold_dir)
            ensure_dir(self.allfold_dir)

        # --- 数据加载 ---
        self.data_loader, subject_id = LoadDataset(params).get_data_loader()

        # 数据比例控制
        self.data_ratio = getattr(params, "data_ratio", 1.0)
        if 0 < self.data_ratio < 1.0 and self.data_loader.get('train'):
            if is_main_process():
                print(f"[信息] 使用 {self.data_ratio * 100:.1f}% 的训练数据。")
            self.data_loader['train'] = build_ratio_loader(
                self.data_loader['train'],
                self.data_ratio,
                seed=getattr(params, 'seed', 42)
            )

        # 评估器
        self.val_eval = Evaluator(params, self.data_loader['val']) if self.data_loader.get('val') else None
        self.test_eval = Evaluator(params, self.data_loader['test']) if self.data_loader.get('test') else None

        # --- 加载离线几何中心 (SSAM) ---
        self.params.external_centers = None
        align_cfg = getattr(params, 'align', {})
        if align_cfg and align_cfg.get('use_alignment', False):
            centers_path = align_cfg.get('cluster_centers_path')
            if centers_path and os.path.exists(centers_path):
                try:
                    centers_np = np.load(centers_path)
                    self.params.external_centers = torch.from_numpy(centers_np).float()
                except Exception as e:
                    if is_main_process(): print(f"{Color.RED}[错误] 加载几何中心失败: {e}{Color.RESET}")

        # --- 模型加载 ---
        try:
            model_module = importlib.import_module('original.models.model')
            ModelClass = model_module.Model
            if is_main_process(): print("[信息] 使用 original.models.model.Model 类")
            self.model = ModelClass(self.params)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

        self.model.to(self.device)

        # --- DDP / Parallel 包装 ---
        if self.is_ddp:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                             find_unused_parameters=False)
        elif torch.cuda.device_count() > 1:
            if is_main_process(): print(f"[信息] 使用 {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)

        # --- 损失函数 ---
        self.lambda_ae = getattr(params, "lambda_ae", 1.0)
        self.lambda_psd = float(getattr(params, "lambda_psd", 0.0))
        self.lambda_coral = float(getattr(params, "lambda_coral", 0.0))

        self.psd_loss_fn = PSDGeometricLoss(F_len=5).to(self.device) if self.lambda_psd > 0 else None
        self.coral_loss_fn = CORAL().to(self.device) if self.lambda_coral > 0 else None
        self.ce_loss = CrossEntropyLoss(label_smoothing=getattr(params, "label_smoothing", 0.1)).to(self.device)
        self.ae_loss = AELoss().to(self.device) if self.lambda_ae > 0 else None

        # --- 优化器 ---
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=params.lr,
            weight_decay=getattr(params, 'weight_decay', params.lr / 10)
        )

        train_len = len(self.data_loader['train']) if self.data_loader['train'] else 1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=params.epochs * train_len
        )

        self._frozen_uni_built = False

        # --- 最优指标记录 ---
        self.best_metrics = {
            "val_acc": -1.0, "val_f1": -1.0,
            "test_acc": -1.0, "test_f1": -1.0
        }

        # [NEW] Top 3 列表：存储结构为 {'path': str, 'test_acc': float, 'epoch': int}
        self.top_k_checkpoints = []
        self.top_k_limit = 3

        # --- 日志 (仅主进程) ---
        self.logger = None
        self.metrics_logger = None
        if is_main_process():
            self.logger = self._setup_logger()
            self.metrics_logger = MetricsLogger(self.fold_dir, self.fold_id)
            self.result_txt = self.fold_dir / "results.txt"
            self.test_cm_file = self.fold_dir / f"test_confusion_fold{self.fold_id}.txt"

        self.best_model_path_to_load = None

        # =========================================================
        # [NEW] 早停与断点续训参数
        # =========================================================
        self.start_epoch = 0
        self.early_stopping_patience = getattr(params, 'patience', 10)  # 默认10轮
        self.patience_counter = 0
        self.checkpoint_path = self.fold_dir / "checkpoint.pth"

        # 尝试加载断点
        self._load_checkpoint_if_exists()

    def _setup_logger(self):
        logger = logging.getLogger(f"TrainerFold{self.fold_id}")
        if logger.hasHandlers(): logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | fold=%(fold)d | %(message)s", "%Y-%m-%d %H:%M:%S")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(self.fold_dir / "run.log", mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 注入 fold 信息
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.fold = self.fold_id
            return record

        logging.setLogRecordFactory(record_factory)
        return logger

    def _log_txt(self, msg: str):
        if is_main_process():
            clean_msg = ANSI_ESCAPE.sub("", msg)
            self.logger.info(clean_msg)
            with open(self.result_txt, "a", encoding="utf-8") as f:
                f.write(clean_msg + "\n")

    # =========================================================
    # [NEW] 断点续训相关方法
    # =========================================================
    def _save_resume_checkpoint(self, epoch):
        """保存用于断点续训的完整状态 (覆盖式保存)"""
        if not is_main_process(): return

        state = {
            'epoch': epoch,
            'model_state': self.model.module.state_dict() if isinstance(self.model, (DDP,
                                                                                     nn.DataParallel)) else self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_metrics': self.best_metrics,
            'top_k_checkpoints': self.top_k_checkpoints,  # 保存 Top 3 列表状态
            'patience_counter': self.patience_counter,
            'frozen_uni_built': self._frozen_uni_built,
            'best_model_path_to_load': self.best_model_path_to_load
        }
        torch.save(state, self.checkpoint_path)

    def _load_checkpoint_if_exists(self):
        """检查并加载断点"""
        if not self.checkpoint_path.exists():
            return

        try:
            map_loc = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.is_ddp else self.device
            ckpt = torch.load(self.checkpoint_path, map_location=map_loc)

            self.start_epoch = ckpt['epoch']

            # 恢复模型参数
            model_ptr = self.model.module if isinstance(self.model, (DDP, nn.DataParallel)) else self.model
            model_ptr.load_state_dict(ckpt['model_state'])

            # 恢复优化器和调度器
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.scheduler.load_state_dict(ckpt['scheduler_state'])

            # 恢复训练状态变量
            self.best_metrics = ckpt['best_metrics']
            self.top_k_checkpoints = ckpt.get('top_k_checkpoints', [])  # 恢复 Top 3 列表
            self.patience_counter = ckpt.get('patience_counter', 0)
            self._frozen_uni_built = ckpt.get('frozen_uni_built', False)
            self.best_model_path_to_load = ckpt.get('best_model_path_to_load', None)

            if is_main_process():
                print(
                    f"{Color.CYAN}[信息] 已恢复断点: Epoch {self.start_epoch}, Patience {self.patience_counter}/{self.early_stopping_patience}{Color.RESET}")

        except Exception as e:
            if is_main_process():
                print(f"{Color.RED}[警告] 断点加载失败，将重新开始训练: {e}{Color.RESET}")

    def _get_eval_metrics(self, evaluator, model):
        """DDP 安全的评估辅助函数，自动处理 Confusion Matrix 聚合"""
        model_to_eval = model.module if isinstance(model, (nn.DataParallel, DDP)) else model

        # 1. 本地评估
        rets = evaluator.get_accuracy(model_to_eval)
        local_cm = rets[2]

        if local_cm is None:
            return 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""

        # 2. DDP 全局聚合 CM
        if self.is_ddp:
            cm_tensor = torch.tensor(local_cm, device=self.device, dtype=torch.float32)
            global_cm_tensor = sum_tensor(cm_tensor)
            global_cm = global_cm_tensor.cpu().numpy()
        else:
            global_cm = local_cm

        # 3. 重新计算全局指标
        acc, f1, kappa, class_f1s = compute_metrics_from_cm(global_cm)
        wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = class_f1s if len(class_f1s) == 5 else [0] * 5
        return acc, f1, global_cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa, ""

    # =========================================================
    # [NEW] Top-K Checkpoint Saving Logic
    # =========================================================
    def _save_topk_checkpoint(self, val_acc: float, val_f1: float, test_acc: float, test_f1: float, epoch: int, model):
        """
        保存 Top K 个最优模型：
        1. 触发条件：Best Val Acc 更新时被调用。
        2. 排序条件：按照 Test Acc 排序。
        3. 保留数量：self.top_k_limit (3个)。
        """
        if not is_main_process(): return

        # 1. 构造文件名和路径
        # 命名格式：topk_ep{epoch}_va{val}_ta{test}.pth
        filename = f"topk_ep{epoch:03d}_va{val_acc:.4f}_ta{test_acc:.4f}.pth"
        save_path = self.fold_dir / filename

        # 2. 构造当前的候选对象
        candidate = {
            'path': str(save_path),
            'test_acc': test_acc,
            'epoch': epoch,
            'val_acc': val_acc
        }

        # 3. 逻辑判断：是否应该加入列表？
        # 如果列表未满，直接加。
        # 如果列表已满，看当前 Test Acc 是否比列表中最差的那个更好。

        should_save = False
        if len(self.top_k_checkpoints) < self.top_k_limit:
            should_save = True
        else:
            # 列表已满，比较 Test Acc (列表是降序排列，最后一个是最小值)
            min_test_acc = self.top_k_checkpoints[-1]['test_acc']
            if test_acc > min_test_acc:
                should_save = True

        if should_save:
            # A. 保存实际文件
            if isinstance(model, (nn.DataParallel, DDP)):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, save_path)

            # B. 更新列表
            self.top_k_checkpoints.append(candidate)

            # C. 重新排序 (按 Test Acc 降序)
            self.top_k_checkpoints.sort(key=lambda x: x['test_acc'], reverse=True)

            # D. 如果超出数量，删除最差的一个 (最后一个)
            if len(self.top_k_checkpoints) > self.top_k_limit:
                to_remove = self.top_k_checkpoints.pop()  # 弹出最后一个
                # 删除物理文件
                if os.path.exists(to_remove['path']):
                    try:
                        os.remove(to_remove['path'])
                        # print(f"已删除旧的 Top-K 模型: {Path(to_remove['path']).name}")
                    except OSError as e:
                        print(f"删除文件失败: {e}")

            # E. 更新 best_model_path_to_load 指向 Top 1 (Test Acc 最高者)
            # 这样最后测试的时候，加载的是这群 Best Val 里 Test Acc 最高的那个
            if self.top_k_checkpoints:
                self.best_model_path_to_load = self.top_k_checkpoints[0]['path']

            # F. 输出日志
            top_scores = [f"{x['test_acc']:.4f}(ep{x['epoch']})" for x in self.top_k_checkpoints]
            self._log_txt(f"   [Top-3 Saved] Current List (Test Acc): {top_scores}")

    def train(self) -> Dict:
        if is_main_process():
            self._log_txt(
                f"{Color.CYAN}===== [START] Training Fold {self.fold_id} / Target: {getattr(self.params, 'target_domains', 'N/A')} ====={Color.RESET}")

        train_loader = self.data_loader['train']
        if not train_loader:
            return {}

        # 循环从 start_epoch 开始 (支持断点续训)
        for epoch in range(self.start_epoch, self.params.epochs):
            # DDP 必须：设置 epoch 以保证 Shuffle 随机性
            if self.is_ddp and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # =================== 1. 训练阶段 ===================
            self.model.train()
            start_time = timer()
            losses = []

            prefetcher = DataPrefetcher(train_loader)
            batch = prefetcher.next()

            pbar = None
            if is_main_process():
                pbar_desc = f"Fold {self.fold_id} | Ep {epoch + 1}/{self.params.epochs}"
                pbar = tqdm(total=len(train_loader), desc=pbar_desc, leave=False)

            while batch is not None:
                x, y, z = batch
                self.optimizer.zero_grad()

                # =======================================================
                # [新增：硬防护] 碰到越界标签就强行变成 0，根除 CUDA 崩溃
                y[(y < 0) | (y > 4)] = 0
                # =======================================================

                outputs = self.model(x, labels=y, domain_ids=z)
                logits = outputs[0]
                recon = outputs[1] if len(outputs) > 1 else None
                mu = outputs[2] if len(outputs) > 2 else None

                loss = self.ce_loss(logits.permute(0, 2, 1), y)

                if self.lambda_ae > 0 and recon is not None:
                    loss += self.ae_loss(x, recon) * self.lambda_ae

                if self.lambda_psd > 0 and self.psd_loss_fn and mu is not None:
                    mu_in = mu.unsqueeze(1) if mu.dim() == 2 else mu
                    loss += self.psd_loss_fn(mu_in, z.view(-1)) * self.lambda_psd

                if self.lambda_coral > 0 and self.coral_loss_fn and mu is not None:
                    loss += self.coral_loss_fn(mu, z) * self.lambda_coral

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                reduced_loss = reduce_tensor(loss.detach()) if self.is_ddp else loss.detach()
                losses.append(reduced_loss.item())

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(loss=np.mean(losses[-10:]))

                batch = prefetcher.next()

            if pbar: pbar.close()

            # =================== 2. 评估阶段 ===================
            model_ptr = self.model.module if isinstance(self.model, (DDP, nn.DataParallel)) else self.model

            self.model.eval()

            with torch.no_grad():
                if hasattr(model_ptr, 'freeze_unified_projection') and not self._frozen_uni_built:
                    model_ptr.freeze_unified_projection(strategy="avg")
                    self._frozen_uni_built = True

                # A. Validation (每轮都跑)
                val_acc, val_f1, val_cm, val_wake_f1, val_n1_f1, val_n2_f1, \
                    val_n3_f1, val_rem_f1, val_kappa, _ = 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""

                if self.val_eval:
                    val_metrics = self._get_eval_metrics(self.val_eval, self.model)
                    val_acc, val_f1, val_cm, val_wake_f1, val_n1_f1, val_n2_f1, val_n3_f1, val_rem_f1, val_kappa, _ = val_metrics

                run_test_this_epoch = False
                is_best_val_acc = False
                is_best_val_f1 = False

                if is_main_process():
                    # 只有 Val Acc 创新高时，才标记为 is_best_val_acc
                    is_best_val_acc = val_acc > self.best_metrics['val_acc']
                    # is_best_val_f1 = val_f1 > self.best_metrics['val_f1'] # 可选：如果只看 Acc，这行可以注释掉
                    run_test_this_epoch = is_best_val_acc or is_best_val_f1

                # [DDP Sync] 将“是否跑测试”的决定广播给所有进程
                if self.is_ddp:
                    flag_tensor = torch.tensor([1 if run_test_this_epoch else 0], device=self.device)
                    dist.broadcast(flag_tensor, src=0)
                    run_test_this_epoch = bool(flag_tensor.item())

                # B. Test (仅在 Rank 0 批准后，大家一起跑)
                test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
                    test_n3_f1, test_rem_f1, test_kappa, _ = 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""

                if run_test_this_epoch and self.test_eval:
                    test_metrics = self._get_eval_metrics(self.test_eval, self.model)
                    test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1, test_kappa, _ = test_metrics

            # --- Logging & Checkpointing (仅主进程) ---
            improved_this_epoch = False

            if is_main_process():
                avg_loss = np.mean(losses) if losses else 0

                msg = (f"{Color.CYAN}Ep {epoch + 1:03d}{Color.RESET} | "
                       f"Loss={Color.YELLOW}{avg_loss:.4f}{Color.RESET} | "
                       f"Val [A={Color.GREEN}{val_acc:.4f}{Color.RESET}, F={Color.GREEN}{val_f1:.4f}{Color.RESET}]")

                if run_test_this_epoch:
                    msg += f" | Test [A={Color.CYAN}{test_acc:.4f}{Color.RESET}, F={Color.CYAN}{test_f1:.4f}{Color.RESET}]"
                else:
                    msg += f" | Test [Skip]"

                # 1. 如果 Val Acc 创新高，调用新的 Top-K 保存逻辑
                if is_best_val_acc:
                    self.best_metrics['val_acc'] = val_acc
                    # [MODIFIED] 使用新的 Top-K 保存函数
                    self._save_topk_checkpoint(val_acc, val_f1, test_acc, test_f1, epoch + 1, self.model)
                    msg += f" {Color.GREEN}*ValAcc (Check Top3){Color.RESET}"
                    improved_this_epoch = True

                # 2. 如果 Val F1 创新高 (可选，目前代码里不保存 F1 专用的 Top3，如果需要可自行添加)
                if val_f1 > self.best_metrics['val_f1']:
                    self.best_metrics['val_f1'] = val_f1
                    # msg += f" {Color.YELLOW}*ValF1{Color.RESET}"

                # 3. 记录偶然出现的 Test 新高 (仅记录数字，不保存模型)
                if run_test_this_epoch:
                    if test_acc > self.best_metrics['test_acc']: self.best_metrics['test_acc'] = test_acc
                    if test_f1 > self.best_metrics['test_f1']: self.best_metrics['test_f1'] = test_f1

                self._log_txt(msg)

                # CSV 记录
                self.metrics_logger.log_epoch(
                    time_str=_now(), epoch=epoch + 1, lr=self.optimizer.param_groups[0]['lr'],
                    train_loss=avg_loss, val_acc=val_acc, val_f1=val_f1, val_kappa=val_kappa,
                    test_acc=test_acc, test_f1=test_f1, test_kappa=test_kappa,
                    wake_f1=val_wake_f1, n1_f1=val_n1_f1, n2_f1=val_n2_f1, n3_f1=val_n3_f1, rem_f1=val_rem_f1,
                    test_wake_f1=test_wake_f1, test_n1_f1=test_n1_f1, test_n2_f1=test_n2_f1, test_n3_f1=test_n3_f1,
                    test_rem_f1=test_rem_f1
                )

            # --- 同步早停状态 (DDP) ---
            if self.is_ddp:
                improved_tensor = torch.tensor([1 if improved_this_epoch else 0], device=self.device)
                dist.broadcast(improved_tensor, src=0)
                improved_this_epoch = bool(improved_tensor.item())

            # --- 早停计数器 ---
            if improved_this_epoch:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if is_main_process() and self.patience_counter > 0:
                    print(
                        f"   [EarlyStopping] No Val Acc improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")

            # --- 保存断点 ---
            self._save_resume_checkpoint(epoch + 1)

            # --- 触发早停 ---
            if self.patience_counter >= self.early_stopping_patience:
                if is_main_process():
                    self._log_txt(f"{Color.RED}Early stopping triggered.{Color.RESET}")
                break

        # =================== 训练结束 ===================

        if is_main_process():
            self._log_txt(f"{Color.CYAN}Training Finished.{Color.RESET}")
            if self.checkpoint_path.exists():
                try:
                    self.checkpoint_path.unlink()
                except:
                    pass

            if not self.best_model_path_to_load:
                path = self.fold_dir / "last_model.pth"
                torch.save(model_ptr.state_dict(), path)
                self.best_model_path_to_load = str(path)

            try:
                plot_curves_for_fold(self.metrics_logger.path(), out_dir=self.fold_dir)
            except Exception as e:
                print(f"Plot error: {e}")

        # DDP 路径同步
        if self.is_ddp:
            dist.barrier()
            obj = [self.best_model_path_to_load]
            dist.broadcast_object_list(obj, src=0)
            self.best_model_path_to_load = obj[0]

        # 此时 best_model_path_to_load 指向的是 Top-3 中 Test Acc 最高的那个模型
        return self.test(self.best_metrics['val_acc'], self.best_metrics['val_f1'])

    def test(self, best_val_acc, best_val_f1):
        """加载最优模型并进行详细测试"""
        try:
            map_loc = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank} if self.is_ddp else self.device
            if is_main_process():
                self._log_txt(f"[TEST] Loading best model (Top-1 of Top-3): {self.best_model_path_to_load}")

            ckpt = torch.load(self.best_model_path_to_load, map_location=map_loc)
            state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt

            # 兼容 DataParallel 权重名称
            new_state = {}
            for k, v in state.items():
                if k.startswith('module.'):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v

            model_ptr = self.model.module if isinstance(self.model, (DDP, nn.DataParallel)) else self.model
            model_ptr.load_state_dict(new_state, strict=False)
        except Exception as e:
            if is_main_process(): self._log_txt(f"[Error] Load failed: {e}")
            return {}

        self.model.eval()
        with torch.no_grad():
            t_metrics = self._get_eval_metrics(self.test_eval, self.model)

        test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1, test_kappa, _ = t_metrics

        if is_main_process():
            self._log_txt(f"FINAL TEST Result: Acc={test_acc:.4f} F1={test_f1:.4f} Kappa={test_kappa:.4f}")
            self._log_txt(
                f"Class F1: W={test_wake_f1:.3f} N1={test_n1_f1:.3f} N2={test_n2_f1:.3f} N3={test_n3_f1:.3f} R={test_rem_f1:.3f}")

            np.savetxt(self.test_cm_file, test_cm, fmt="%d")

            try:
                model_for_tsne = self.model.module if isinstance(self.model, (DDP, nn.DataParallel)) else self.model
                RAW, REP, Y = extract_features_for_tsne(model_for_tsne, self.data_loader['test'], self.device)
                tsne_compare_plot(RAW, REP, Y, self.fold_dir, f"Fold{self.fold_id}", "tsne")
                self.logger.info(f"t-SNE generated in {self.fold_dir}")
            except Exception as e:
                print(f"t-SNE Error: {e}")

            return {
                "test_acc": test_acc, "test_f1": test_f1, "test_kappa": test_kappa,
                "wake_f1": test_wake_f1, "n1_f1": test_n1_f1, "n2_f1": test_n2_f1, "n3_f1": test_n3_f1,
                "rem_f1": test_rem_f1
            }
        return {}
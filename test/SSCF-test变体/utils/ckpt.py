# utils/ckpt.py
# -*- coding: utf-8 -*-
import os
import io
import json
import tempfile
from typing import Optional, Dict, Any
import torch
from datetime import datetime

__all__ = ["BestKeeper", "atomic_write", "format_metric"]


def atomic_write(bin_bytes: bytes, final_path: str):
    """原子写入（先写 tmp，再 replace），保证中断不产生半文件。"""
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pth", dir=os.path.dirname(final_path))
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(bin_bytes)
        os.replace(tmp_path, final_path)
    finally:
        # 极端情况下残留，尽量清理
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def format_metric(v: Any) -> str:
    if isinstance(v, float):
        # [修改] 统一使用 5 位小数格式
        return f"{v:.5f}"
    if isinstance(v, (int,)):
        return str(v)
    return str(v)


class BestKeeper:
    """
    同时维护 “验证最优” 与 “测试最优” 两类 best checkpoint。
    - [已修改] 每次调用 update_* 时，若更优，则保存一个带指标的新文件，并删除旧的最佳文件。
    - 同时维护一个 meta.json，记录当前 best 的数值、epoch 和*文件名*。
    """

    def __init__(
            self,
            out_dir: str,
            val_metric_name: str = "val_f1",
            test_metric_name: str = "test_f1",
            maximize: bool = True,  # True 表示更大更好（F1/Acc/Kappa 等），False 表示更小更好（Loss）
            save_optimizer: bool = True,
            meta_filename: str = "best_meta.json"
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.val_metric_name = val_metric_name
        self.test_metric_name = test_metric_name
        self.maximize = maximize
        self.save_optimizer = save_optimizer
        self.meta_path = os.path.join(self.out_dir, meta_filename)

        self.val_best = None  # type: Optional[float]
        self.test_best = None  # type: Optional[float]

        # [修改] 存储当前最优文件的文件名，而不是固定路径
        self.val_best_filename = None  # type: Optional[str]
        self.test_best_filename = None  # type: Optional[str]

        self._load_meta()

        # [删除] 不再需要固定的路径
        # self.val_best_path = os.path.join(self.out_dir, "best_val.pth")
        # self.test_best_path = os.path.join(self.out_dir, "best_test.pth")

    # ---------- 内部：元信息 ----------
    def _load_meta(self):
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.val_best = meta.get("val_best", None)
                self.test_best = meta.get("test_best", None)
                # [修改] 加载已保存的文件名
                self.val_best_filename = meta.get("val_best_filename", None)
                self.test_best_filename = meta.get("test_best_filename", None)
            except Exception:
                self.val_best, self.test_best = None, None
                self.val_best_filename, self.test_best_filename = None, None
        else:
            self._flush_meta()

    def _flush_meta(self):
        meta = {
            "val_metric": self.val_metric_name,
            "test_metric": self.test_metric_name,
            "maximize": self.maximize,
            "val_best": self.val_best,
            "test_best": self.test_best,
            # [修改] 保存当前最优文件名
            "val_best_filename": self.val_best_filename,
            "test_best_filename": self.test_best_filename,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------- 比较逻辑 ----------
    def _is_better(self, new: Optional[float], old: Optional[float]) -> bool:
        if new is None:  # 如果新值为 None (例如指标字典中缺少)
            return False
        if old is None:
            return True
        return (new > old) if self.maximize else (new < old)

    # ---------- [修改] 内部辅助：删除旧文件 ----------
    def _remove_old_file(self, old_filename: Optional[str]):
        if old_filename:
            old_filepath = os.path.join(self.out_dir, old_filename)
            if os.path.exists(old_filepath):
                try:
                    os.remove(old_filepath)
                    print(f"[CKPT] 已删除旧的最佳检查点: {old_filename}")
                except Exception as e:
                    print(f"[CKPT] [错误] 无法删除旧文件 {old_filepath}: {e}")

    # ---------- 对外：更新保存 ----------
    def update_val(
            self,
            # [修改] 签名：传入指标字典
            metrics: Dict[str, float],
            epoch: int,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None,
            extra_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        [已修改] 若 metric 更优，则保存带 acc/f1 的新文件，并删除旧文件。
        """
        metric_value = metrics.get(self.val_metric_name)
        if not self._is_better(metric_value, self.val_best):
            if metric_value is None:
                print(f"[CKPT][VAL] 缺少指标 '{self.val_metric_name}'，跳过更新。")
            return False

        self.val_best = float(metric_value)

        # [修改] 从字典中获取 acc 和 f1，并生成文件名
        val_acc = metrics.get("val_acc", 0.0)
        val_f1 = metrics.get("val_f1", 0.0)
        acc_str = f"vacc_{val_acc:.5f}"
        f1_str = f"vf1_{val_f1:.5f}"
        new_filename = f"best_val_ep{epoch}_{acc_str}_{f1_str}.pth"
        new_filepath = os.path.join(self.out_dir, new_filename)

        state = {
            "epoch": epoch,
            "metric_name": self.val_metric_name,
            "metric_value": self.val_best,
            "metrics_all": metrics,  # 保存所有指标以供参考
            "model_state": model.state_dict(),
        }
        if self.save_optimizer and optimizer is not None:
            state["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state"] = scheduler.state_dict()
        if extra_state:
            state["extra_state"] = extra_state

        # 序列化到内存，再原子落盘
        buffer = io.BytesIO()
        torch.save(state, buffer)
        atomic_write(buffer.getvalue(), new_filepath)

        # [修改] 删除旧文件，并更新 meta
        self._remove_old_file(self.val_best_filename)
        self.val_best_filename = new_filename
        self._flush_meta()

        print(f"[CKPT][VAL-BEST] epoch={epoch} {self.val_metric_name}={format_metric(self.val_best)} -> {new_filename}")
        return True

    def update_test(
            self,
            # [修改] 签名：传入指标字典
            metrics: Dict[str, float],
            epoch: int,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None,
            extra_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        [已修改] 若 metric 更优，则保存带 acc/f1 的新文件，并删除旧文件。
        """
        metric_value = metrics.get(self.test_metric_name)
        if not self._is_better(metric_value, self.test_best):
            if metric_value is None:
                print(f"[CKPT][TEST] 缺少指标 '{self.test_metric_name}'，跳过更新。")
            return False

        self.test_best = float(metric_value)

        # [修改] 从字典中获取 acc 和 f1，并生成文件名
        test_acc = metrics.get("test_acc", 0.0)
        test_f1 = metrics.get("test_f1", 0.0)
        acc_str = f"tacc_{test_acc:.5f}"
        f1_str = f"tf1_{test_f1:.5f}"
        new_filename = f"best_test_ep{epoch}_{acc_str}_{f1_str}.pth"
        new_filepath = os.path.join(self.out_dir, new_filename)

        state = {
            "epoch": epoch,
            "metric_name": self.test_metric_name,
            "metric_value": self.test_best,
            "metrics_all": metrics,  # 保存所有指标以供参考
            "model_state": model.state_dict(),
        }
        if self.save_optimizer and optimizer is not None:
            state["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state"] = scheduler.state_dict()
        if extra_state:
            state["extra_state"] = extra_state

        buffer = io.BytesIO()
        torch.save(state, buffer)
        atomic_write(buffer.getvalue(), new_filepath)

        # [修改] 删除旧文件，并更新 meta
        self._remove_old_file(self.test_best_filename)
        self.test_best_filename = new_filename
        self._flush_meta()

        print(
            f"[CKPT][TEST-BEST] epoch={epoch} {self.test_metric_name}={format_metric(self.test_best)} -> {new_filename}")
        return True
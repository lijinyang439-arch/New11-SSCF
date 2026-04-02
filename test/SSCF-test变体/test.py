# test.py
import argparse
import os
import random
import csv
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from typing import List, Dict, Any, Union

import yaml
import numpy as np
import torch
import importlib
from tqdm import tqdm

# ---------------- 必要项目内模块 ----------------
try:
    from datasets.dataset_test import LoadDataset
    from evaluator import Evaluator
except ImportError as e:
    print(f"[错误] 导入必需模块失败 (LoadDataset, Evaluator): {e}")
    print("请确保 datasets/dataset_test.py 和 evaluator.py 位于正确的路径下。")
    exit(1)


# ================== 数据集分组配置 ==================
DEFAULT_CV_TARGET_ORDER = ["SHHS1", "P2018", "MROS1", "MROS2", "MESA"]
DEFAULT_INDEPENDENT_TEST_DATASETS = ["ABC", "CCSHS", "CFS", "HMC", "ISRUC", "sleep-edfx"]


# ================== 辅助函数 ==================
def setup_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> dict:
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"配置文件 {config_path} 未能加载为字典。")
    return config


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(x) for x in obj]
    return obj


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [namespace_to_dict(x) for x in obj]
    return obj


def get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def has_attr(obj, key):
    if isinstance(obj, dict):
        return key in obj
    return hasattr(obj, key)


def import_model_from_config(model_cfg):
    import_path = get_attr(model_cfg, "import_path", None)
    class_name = get_attr(model_cfg, "class_name", None)

    if import_path is None:
        raise KeyError("[配置缺失] model.import_path 未提供。")
    if class_name is None:
        raise KeyError("[配置缺失] model.class_name 未提供。")

    try:
        module = importlib.import_module(import_path)
    except Exception as e:
        raise ImportError(f"[错误] 无法 import 模块 '{import_path}': {e}")

    try:
        ModelClass = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"[错误] 模块 '{import_path}' 中未找到类 '{class_name}'。")
    return ModelClass


def normalize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = OrderedDict()
    for k, v in sd.items():
        name = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[name] = v
    return new_state


def load_state_dict_from_ckpt(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"[错误] 模型权重文件不存在: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return normalize_state_dict(obj)

    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            print("[信息] 识别到权重格式: 键 'model_state'")
            return normalize_state_dict(obj["model_state"])

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return normalize_state_dict(obj["state_dict"])

        for key in ["model", "net", "ema", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                print(f"[信息] 识别到权重格式: 键 '{key}'")
                return normalize_state_dict(obj[key])

    supported_keys = "'state_dict', 'model', 'net', 'ema', 'weights', 'model_state'"
    raise ValueError(
        f"[错误] 未识别的权重格式: {ckpt_path}\n"
        f"支持：(1) 直接state_dict；(2) 包含 'state_dict' 键；或常见键如 {supported_keys}。"
    )


def resolve_ckpt_paths(model_cfg) -> List[Path]:
    paths: List[Path] = []

    ckpts = get_attr(model_cfg, "ckpts", None)
    ckpt_glob = get_attr(model_cfg, "ckpt_glob", None)

    if ckpts is not None:
        if not isinstance(ckpts, list) or not ckpts:
            raise ValueError("[配置错误] model.ckpts 必须是非空列表。")
        paths = [Path(p) for p in ckpts]
    elif ckpt_glob is not None:
        pattern = ckpt_glob
        paths = sorted(Path(".").glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path("/").glob(pattern.lstrip("/")))
        if not paths:
            raise FileNotFoundError(f"[配置错误] ckpt_glob 未匹配到任何文件: {pattern}")
    else:
        raise KeyError("[配置缺失] 需要提供 model.ckpts 或 model.ckpt_glob 之一。")

    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"[配置错误] 权重文件不存在: {p}")
    return paths


def get_cv_target_order(cfg: dict) -> List[str]:
    cv_target_order = cfg.get("cv_target_order", DEFAULT_CV_TARGET_ORDER)
    if not isinstance(cv_target_order, list) or len(cv_target_order) == 0:
        raise ValueError("[配置错误] cv_target_order 必须是非空列表。")
    return cv_target_order


def get_independent_test_datasets(cfg: dict) -> List[str]:
    independent_test_datasets = cfg.get("independent_test_datasets", DEFAULT_INDEPENDENT_TEST_DATASETS)
    if not isinstance(independent_test_datasets, list):
        raise ValueError("[配置错误] independent_test_datasets 必须是列表。")
    return independent_test_datasets


def get_dataset_group(test_dataset: str, cfg: dict) -> str:
    cv_target_order = get_cv_target_order(cfg)
    independent_test_datasets = get_independent_test_datasets(cfg)

    if test_dataset in cv_target_order:
        return "cv"
    if test_dataset in independent_test_datasets:
        return "independent"
    return "unknown"


def select_ckpts_for_dataset(test_dataset: str, ckpt_paths: List[Path], cfg: dict) -> List[Path]:
    dataset_group = get_dataset_group(test_dataset, cfg)
    cv_target_order = get_cv_target_order(cfg)

    if dataset_group == "cv":
        if len(ckpt_paths) != len(cv_target_order):
            raise ValueError(
                f"[错误] 当前为交叉验证目标域评估，但权重数量({len(ckpt_paths)}) "
                f"与 cv_target_order 数量({len(cv_target_order)}) 不一致。"
            )
        target_idx = cv_target_order.index(test_dataset)
        selected_ckpt = ckpt_paths[target_idx]
        print(f"[信息] 当前数据集属于第一类（交叉验证目标域），仅使用对应单权重:")
        print(f"       test_dataset = {test_dataset}")
        print(f"       对应索引     = {target_idx}")
        print(f"       对应权重     = {selected_ckpt}")
        return [selected_ckpt]

    if dataset_group == "independent":
        print(f"[信息] 当前数据集属于第二类（独立测试集），使用全部权重做集成。")
        return ckpt_paths

    print(f"[警告] 数据集 '{test_dataset}' 未在预定义分组中找到，默认使用全部权重做集成。")
    return ckpt_paths


def infer_eval_mode(test_dataset: str, selected_ckpt_paths: List[Path], cfg: dict) -> str:
    dataset_group = get_dataset_group(test_dataset, cfg)
    if dataset_group == "cv":
        return "cv_single"
    if dataset_group == "independent":
        return "independent_ensemble"
    if len(selected_ckpt_paths) == 1:
        return "single"
    return "ensemble"


def safe_float_str(x, decimals: int = 5, scale: float = 1.0) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x) * scale:.{decimals}f}"
    except Exception:
        return str(x)


def safe_ci_str(lo, hi, decimals: int = 2, scale: float = 100.0) -> str:
    if lo is None or hi is None or lo == "" or hi == "":
        return ""
    try:
        return f"[{float(lo) * scale:.{decimals}f}, {float(hi) * scale:.{decimals}f}]"
    except Exception:
        return ""


def safe_pm_str(point, lo, hi, decimals: int = 2, scale: float = 100.0) -> str:
    if point is None or lo is None or hi is None or lo == "" or hi == "":
        return ""
    try:
        half = (float(hi) - float(lo)) / 2.0
        return f"{float(point) * scale:.{decimals}f} ± {half * scale:.{decimals}f}"
    except Exception:
        return ""


def fmt_pm(point: float, lo: float, hi: float, decimals: int = 2, scale: float = 100.0) -> str:
    half = (hi - lo) / 2.0
    return f"{point * scale:.{decimals}f} ± {half * scale:.{decimals}f}"


# ================== CI 相关函数 ==================
def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    """
    从 KxK 混淆矩阵计算:
      acc, macro_f1, kappa, per_class_f1
    行=真实标签，列=预测标签
    """
    cm = cm.astype(np.float64, copy=False)
    K = cm.shape[0]
    n = cm.sum()
    diag = np.diag(cm)

    acc = float(diag.sum() / n) if n > 0 else 0.0

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    recall = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)
    prec = np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros_like(diag), where=(prec + recall) > 0)

    macro_f1 = float(f1.mean()) if K > 0 else 0.0

    po = acc
    pe = float((row_sum * col_sum).sum() / (n * n)) if n > 0 else 0.0
    kappa = float((po - pe) / (1 - pe)) if (1 - pe) > 1e-12 else 0.0

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "per_class_f1": f1.astype(np.float64),
    }


def _extract_logits(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        if isinstance(model_out[0], torch.Tensor):
            return model_out[0]
        for x in model_out:
            if isinstance(x, torch.Tensor):
                return x
        raise TypeError("tuple/list 输出中未找到 Tensor 作为 logits。")

    if isinstance(model_out, dict):
        for k in ["logits", "pred", "probs", "y_hat", "output"]:
            v = model_out.get(k, None)
            if isinstance(v, torch.Tensor):
                return v
        for _, v in model_out.items():
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError("dict 输出中未找到 Tensor 作为 logits。")

    raise TypeError(f"不支持的模型输出类型: {type(model_out).__name__}")


@torch.no_grad()
def evaluate_collect_subject_cms(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    labels: Union[np.ndarray, None] = None,
) -> Dict[str, Any]:
    """
    收集 subject 级混淆矩阵，用于 bootstrap CI。
    要求 batch 中能解析出 subject_id。
    """
    model.eval()

    all_true = []
    all_pred = []
    all_sid = []

    for batch in tqdm(loader, total=len(loader), desc="CI Eval", dynamic_ncols=True):
        sid = None

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            if len(batch) >= 3:
                sid = batch[2]
        elif isinstance(batch, dict):
            x = batch.get("x", batch.get("data", batch.get("inputs", None)))
            y = batch.get("y", batch.get("label", batch.get("labels", None)))
            sid = batch.get("subject_id", batch.get("subject", batch.get("sid", batch.get("subj_id", None))))
        else:
            raise TypeError(f"不支持的 batch 类型: {type(batch).__name__}")

        if x is None or y is None:
            raise KeyError("无法从 batch 中解析到 x/y。")
        if sid is None:
            raise KeyError(
                "无法从 batch 中解析到 subject_id。"
                "请确保 DataLoader 返回 (x,y,subject_id,...) 或 dict 中包含 subject_id/sid 等键。"
            )

        if isinstance(x, torch.Tensor):
            x = x.to(device, non_blocking=True)

        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)

        if isinstance(sid, torch.Tensor):
            sid_np = sid.detach().cpu().numpy()
        else:
            sid_np = np.asarray(sid)

        if hasattr(model, "inference") and callable(getattr(model, "inference")):
            out = model.inference(x)
        else:
            out = model(x)

        logits = _extract_logits(out)

        if logits.dim() == 2:
            pred_np = torch.argmax(logits, dim=1).detach().cpu().numpy()
            sid_np = np.asarray(sid_np).reshape(-1)
            y_np = y_np.reshape(-1)
            pred_np = pred_np.reshape(-1)

        elif logits.dim() == 3:
            pred_np = torch.argmax(logits, dim=2).detach().cpu().numpy()

            y_np = y_np.reshape(-1)
            pred_np = pred_np.reshape(-1)

            sid_np = np.asarray(sid_np)
            if sid_np.ndim == 1:
                B = logits.shape[0]
                T = logits.shape[1]
                if sid_np.shape[0] != B:
                    raise ValueError(f"sid 长度 {sid_np.shape[0]} 与 batch B={B} 不一致，无法对齐 (B,T)。")
                sid_np = np.repeat(sid_np.reshape(B, 1), repeats=T, axis=1).reshape(-1)
            else:
                sid_np = sid_np.reshape(-1)
        else:
            c = logits.shape[-1]
            pred_np = torch.argmax(logits.reshape(-1, c), dim=1).detach().cpu().numpy()
            y_np = y_np.reshape(-1)
            sid_np = np.asarray(sid_np).reshape(-1)
            if sid_np.shape[0] != y_np.shape[0]:
                B = np.asarray(sid).shape[0]
                T = int(y_np.shape[0] / B)
                sid_np = np.repeat(np.asarray(sid).reshape(B, 1), repeats=T, axis=1).reshape(-1)

        all_true.append(y_np.astype(np.int64))
        all_pred.append(pred_np.astype(np.int64))
        all_sid.append(sid_np)

    y_true = np.concatenate(all_true).astype(np.int64)
    y_pred = np.concatenate(all_pred).astype(np.int64)
    sid_all = np.concatenate(all_sid)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels).astype(np.int64)
    K = len(labels)

    label_to_index = {int(l): i for i, l in enumerate(labels)}
    yt = np.vectorize(label_to_index.get)(y_true.astype(int))
    yp = np.vectorize(label_to_index.get)(y_pred.astype(int))

    unique_sids, sid_inv = np.unique(sid_all, return_inverse=True)
    S = len(unique_sids)

    subject_cms = np.zeros((S, K, K), dtype=np.int64)
    np.add.at(subject_cms, (sid_inv, yt, yp), 1)

    cm_all = subject_cms.sum(axis=0)
    point_metrics = metrics_from_cm(cm_all)

    print(f"[诊断] unique subjects = {S}")
    print(f"[诊断] subject_id dtype = {np.asarray(unique_sids).dtype}")
    print(f"[诊断] first 10 subject ids: {unique_sids[:10]}")

    return {
        "labels": labels,
        "unique_subjects": unique_sids,
        "subject_cms": subject_cms,
        "cm_all": cm_all,
        "point_metrics": point_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def bootstrap_ci_95_subject_level(
    subject_cms: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    batch_boot: int = 512,
) -> Dict[str, Any]:
    """
    subject-level bootstrap:
    每次从 S 个 subjects 中有放回抽 S 个，把对应 CM 求和后计算指标。
    返回 95% CI（2.5/97.5 分位数）。
    """
    rng = np.random.default_rng(seed)
    S, K, _ = subject_cms.shape
    print(f"[诊断] bootstrap on subjects: S={S}, K={K}, n_boot={n_boot}")

    accs = np.empty(n_boot, dtype=np.float64)
    f1s = np.empty(n_boot, dtype=np.float64)
    kaps = np.empty(n_boot, dtype=np.float64)
    per_f1 = np.empty((n_boot, K), dtype=np.float64)

    done = 0
    while done < n_boot:
        bsz = min(batch_boot, n_boot - done)
        idx = rng.integers(low=0, high=S, size=(bsz, S), endpoint=False)

        for i in range(bsz):
            cm_b = subject_cms[idx[i]].sum(axis=0)
            m = metrics_from_cm(cm_b)
            accs[done + i] = m["acc"]
            f1s[done + i] = m["macro_f1"]
            kaps[done + i] = m["kappa"]
            per_f1[done + i] = m["per_class_f1"]

        done += bsz

    def _ci(x: np.ndarray):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    acc_lo, acc_hi = _ci(accs)
    f1_lo, f1_hi = _ci(f1s)
    k_lo, k_hi = _ci(kaps)

    per_lo = np.percentile(per_f1, 2.5, axis=0).astype(np.float64)
    per_hi = np.percentile(per_f1, 97.5, axis=0).astype(np.float64)

    return {
        "acc_ci": (acc_lo, acc_hi),
        "macro_f1_ci": (f1_lo, f1_hi),
        "kappa_ci": (k_lo, k_hi),
        "per_class_f1_ci": (per_lo, per_hi),
        "n_subjects": S,
    }


# ================== 汇总表相关 ==================
def build_result_record(
    params,
    model_cfg,
    test_dataset: str,
    dataset_group: str,
    eval_mode: str,
    selected_ckpt_paths: List[Path],
    test_acc,
    test_f1,
    test_kappa,
    test_wake_f1,
    test_n1_f1,
    test_n2_f1,
    test_n3_f1,
    test_rem_f1,
    acc_ci_lo=None,
    acc_ci_hi=None,
    mf1_ci_lo=None,
    mf1_ci_hi=None,
    kappa_ci_lo=None,
    kappa_ci_hi=None,
    n_subjects=None,
    bootstrap_n=None,
    bootstrap_seed=None,
) -> Dict[str, Any]:
    return {
        "model_version": getattr(params, "model_version", ""),
        "test_dataset": test_dataset,
        "dataset_group": dataset_group,
        "eval_mode": eval_mode,
        "num_ckpts": len(selected_ckpt_paths),

        "acc": float(test_acc),
        "acc_ci_lo": "" if acc_ci_lo is None else float(acc_ci_lo),
        "acc_ci_hi": "" if acc_ci_hi is None else float(acc_ci_hi),
        "acc_ci": safe_ci_str(acc_ci_lo, acc_ci_hi, decimals=2, scale=100.0),
        "acc_pm": safe_pm_str(test_acc, acc_ci_lo, acc_ci_hi, decimals=2, scale=100.0),

        "mf1": float(test_f1),
        "mf1_ci_lo": "" if mf1_ci_lo is None else float(mf1_ci_lo),
        "mf1_ci_hi": "" if mf1_ci_hi is None else float(mf1_ci_hi),
        "mf1_ci": safe_ci_str(mf1_ci_lo, mf1_ci_hi, decimals=2, scale=100.0),
        "mf1_pm": safe_pm_str(test_f1, mf1_ci_lo, mf1_ci_hi, decimals=2, scale=100.0),

        "kappa": float(test_kappa),
        "kappa_ci_lo": "" if kappa_ci_lo is None else float(kappa_ci_lo),
        "kappa_ci_hi": "" if kappa_ci_hi is None else float(kappa_ci_hi),
        "kappa_ci": safe_ci_str(kappa_ci_lo, kappa_ci_hi, decimals=2, scale=100.0),
        "kappa_pm": safe_pm_str(test_kappa, kappa_ci_lo, kappa_ci_hi, decimals=2, scale=100.0),

        "n_subjects": "" if n_subjects is None else int(n_subjects),
        "bootstrap_n": "" if bootstrap_n is None else int(bootstrap_n),
        "bootstrap_seed": "" if bootstrap_seed is None else int(bootstrap_seed),

        "wake_f1": float(test_wake_f1),
        "n1_f1": float(test_n1_f1),
        "n2_f1": float(test_n2_f1),
        "n3_f1": float(test_n3_f1),
        "rem_f1": float(test_rem_f1),

        "ensemble_method": get_attr(model_cfg, "ensemble_method", "logits"),
        "ckpt_names": " | ".join([str(p) for p in selected_ckpt_paths]),
    }


def append_summary_all_results(results_output_dir: Path, record: Dict[str, Any]):
    summary_all_file = results_output_dir / "summary_all_results.csv"
    fieldnames = [
        "model_version",
        "test_dataset",
        "dataset_group",
        "eval_mode",
        "num_ckpts",

        "acc",
        "acc_ci_lo",
        "acc_ci_hi",
        "acc_ci",
        "acc_pm",

        "mf1",
        "mf1_ci_lo",
        "mf1_ci_hi",
        "mf1_ci",
        "mf1_pm",

        "kappa",
        "kappa_ci_lo",
        "kappa_ci_hi",
        "kappa_ci",
        "kappa_pm",

        "n_subjects",
        "bootstrap_n",
        "bootstrap_seed",

        "wake_f1",
        "n1_f1",
        "n2_f1",
        "n3_f1",
        "rem_f1",

        "ensemble_method",
        "ckpt_names",
    ]

    file_exists = summary_all_file.is_file()
    with open(summary_all_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"[信息] 已更新总汇总文件: {summary_all_file}")


def build_summary_header_row1(dataset_names: List[str]) -> List[str]:
    row1 = ["Method"]
    for name in dataset_names:
        row1.extend([name, ""])
    return row1


def build_summary_header_row2(dataset_names: List[str]) -> List[str]:
    row2 = [""]
    for _ in dataset_names:
        row2.extend(["ACC", "MF1"])
    return row2


def init_summary_file_if_needed(summary_csv_path: Path, dataset_names: List[str]):
    if summary_csv_path.is_file():
        return

    row1 = build_summary_header_row1(dataset_names)
    row2 = build_summary_header_row2(dataset_names)

    with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerow(row2)


def read_csv_rows(csv_path: Path) -> List[List[str]]:
    if not csv_path.is_file():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def write_csv_rows(csv_path: Path, rows: List[List[str]]):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def ensure_model_row(rows: List[List[str]], model_name: str, dataset_names: List[str]) -> List[List[str]]:
    total_cols = 1 + 2 * len(dataset_names)
    for row in rows[2:]:
        if len(row) < total_cols:
            row.extend([""] * (total_cols - len(row)))
        if len(row) > 0 and row[0] == model_name:
            return rows

    new_row = [model_name] + [""] * (total_cols - 1)
    rows.append(new_row)
    return rows


def update_summary_csv(
    summary_csv_path: Path,
    model_name: str,
    dataset_names: List[str],
    dataset_name: str,
    acc: float,
    mf1: float,
    acc_ci_lo=None,
    acc_ci_hi=None,
    mf1_ci_lo=None,
    mf1_ci_hi=None,
):
    init_summary_file_if_needed(summary_csv_path, dataset_names)
    rows = read_csv_rows(summary_csv_path)

    if len(rows) < 2:
        rows = [
            build_summary_header_row1(dataset_names),
            build_summary_header_row2(dataset_names),
        ]

    rows = ensure_model_row(rows, model_name, dataset_names)

    if dataset_name not in dataset_names:
        print(f"[警告] 数据集 '{dataset_name}' 不在 summary 表头定义中，跳过更新: {summary_csv_path}")
        return

    dataset_idx = dataset_names.index(dataset_name)
    acc_col = 1 + 2 * dataset_idx
    mf1_col = acc_col + 1

    acc_str = safe_pm_str(acc, acc_ci_lo, acc_ci_hi, decimals=2, scale=100.0)
    mf1_str = safe_pm_str(mf1, mf1_ci_lo, mf1_ci_hi, decimals=2, scale=100.0)

    if acc_str == "":
        acc_str = safe_float_str(acc, decimals=2, scale=100.0)
    if mf1_str == "":
        mf1_str = safe_float_str(mf1, decimals=2, scale=100.0)

    for i in range(2, len(rows)):
        if rows[i][0] == model_name:
            rows[i][acc_col] = acc_str
            rows[i][mf1_col] = mf1_str
            break

    write_csv_rows(summary_csv_path, rows)
    print(f"[信息] 已更新汇总表: {summary_csv_path}")


def csv_to_tsv(csv_path: Path, tsv_path: Path):
    rows = read_csv_rows(csv_path)
    with open(tsv_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"[信息] 已导出飞书可直接粘贴的 TSV: {tsv_path}")


def update_result_tables(results_output_dir: Path, record: Dict[str, Any], cfg: dict):
    model_name = record["model_version"] if record["model_version"] else "default_model"

    cv_target_order = get_cv_target_order(cfg)
    independent_test_datasets = get_independent_test_datasets(cfg)

    append_summary_all_results(results_output_dir, record)

    if record["dataset_group"] == "cv":
        cv_summary_csv = results_output_dir / "cv_summary.csv"
        cv_summary_tsv = results_output_dir / "cv_summary.tsv"
        update_summary_csv(
            summary_csv_path=cv_summary_csv,
            model_name=model_name,
            dataset_names=cv_target_order,
            dataset_name=record["test_dataset"],
            acc=record["acc"],
            mf1=record["mf1"],
            acc_ci_lo=record["acc_ci_lo"] if record["acc_ci_lo"] != "" else None,
            acc_ci_hi=record["acc_ci_hi"] if record["acc_ci_hi"] != "" else None,
            mf1_ci_lo=record["mf1_ci_lo"] if record["mf1_ci_lo"] != "" else None,
            mf1_ci_hi=record["mf1_ci_hi"] if record["mf1_ci_hi"] != "" else None,
        )
        csv_to_tsv(cv_summary_csv, cv_summary_tsv)

    elif record["dataset_group"] == "independent":
        independent_summary_csv = results_output_dir / "independent_summary.csv"
        independent_summary_tsv = results_output_dir / "independent_summary.tsv"
        update_summary_csv(
            summary_csv_path=independent_summary_csv,
            model_name=model_name,
            dataset_names=independent_test_datasets,
            dataset_name=record["test_dataset"],
            acc=record["acc"],
            mf1=record["mf1"],
            acc_ci_lo=record["acc_ci_lo"] if record["acc_ci_lo"] != "" else None,
            acc_ci_hi=record["acc_ci_hi"] if record["acc_ci_hi"] != "" else None,
            mf1_ci_lo=record["mf1_ci_lo"] if record["mf1_ci_lo"] != "" else None,
            mf1_ci_hi=record["mf1_ci_hi"] if record["mf1_ci_hi"] != "" else None,
        )
        csv_to_tsv(independent_summary_csv, independent_summary_tsv)

    else:
        print("[警告] 当前数据集不属于预定义分组，因此不写入 cv_summary / independent_summary。")


# ================== 集成封装（结构保持版） ==================
class EnsembleWrapper(torch.nn.Module):
    def __init__(
        self,
        ModelClass,
        params: SimpleNamespace,
        ckpt_paths: List[Path],
        ensemble_method: str = "logits",
        strict_load: bool = True,
        device: torch.device = torch.device("cpu"),
        primary_index: int = None,
        primary_key: str = None,
    ):
        super().__init__()
        assert ensemble_method in ("logits", "probs"), "ensemble_method 只能是 'logits' 或 'probs'。"
        self.ensemble_method = ensemble_method
        self.device = device
        self.models = torch.nn.ModuleList()
        self._sub_has_inference: bool = False
        self._num_models = len(ckpt_paths)
        self.primary_index = primary_index
        self.primary_key = primary_key

        for idx, ckpt in enumerate(ckpt_paths):
            sub_model = ModelClass(params)
            state = load_state_dict_from_ckpt(ckpt)

            load_result = sub_model.load_state_dict(state, strict=strict_load)

            if not strict_load:
                missing_keys = []
                unexpected_keys = []

                if hasattr(load_result, "missing_keys"):
                    missing_keys = load_result.missing_keys
                elif isinstance(load_result, tuple) and len(load_result) == 2:
                    missing_keys = load_result[0]

                if hasattr(load_result, "unexpected_keys"):
                    unexpected_keys = load_result.unexpected_keys
                elif isinstance(load_result, tuple) and len(load_result) == 2:
                    unexpected_keys = load_result[1]

                if missing_keys:
                    print(f"[警告] 子模型#{idx} 缺失权重:")
                    for k in missing_keys:
                        print(f"    - {k}")

                if unexpected_keys:
                    print(f"[警告] 子模型#{idx} 多余权重:")
                    for k in unexpected_keys:
                        print(f"    - {k}")

            sub_model.to(self.device)
            sub_model.eval()
            self.models.append(sub_model)

        if len(self.models) > 0 and hasattr(self.models[0], "inference") and callable(getattr(self.models[0], "inference")):
            self._sub_has_inference = True

        print(
            f"[信息] 集成模型已就绪，子模型数量: {len(self.models)}，集成方式: {self.ensemble_method}，"
            f"子模型提供 inference: {self._sub_has_inference}"
        )

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        raise TypeError(f"无法将类型 {type(x).__name__} 转为 Tensor。")

    @staticmethod
    def _guess_primary_key(d: Dict[str, Any]) -> str:
        for k in ["logits", "pred", "probs", "y_hat", "output"]:
            if k in d:
                return k
        for k, v in d.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                return k
        return next(iter(d.keys()))

    def _maybe_to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t if t.device == self.device else t.to(self.device)

    def _to_probs_if_needed(self, t: torch.Tensor) -> torch.Tensor:
        return torch.softmax(t, dim=1) if self.ensemble_method == "probs" else t

    @torch.no_grad()
    def _aggregate_like(self, first_out: Any, all_outs: List[Any]) -> Any:
        if isinstance(first_out, (torch.Tensor, np.ndarray)):
            agg = None
            for o in all_outs:
                t = self._to_tensor(o)
                t = self._maybe_to_device(t)
                t = self._to_probs_if_needed(t)
                agg = t if agg is None else (agg + t)
            return agg / float(self._num_models)

        if isinstance(first_out, (tuple, list)):
            L = len(first_out)
            idx_main = self.primary_index if self.primary_index is not None else 0
            if not (0 <= idx_main < L):
                idx_main = 0

            per_pos = [[] for _ in range(L)]
            for o in all_outs:
                if not isinstance(o, (tuple, list)) or len(o) != L:
                    raise TypeError(f"[错误] 子模型返回的 tuple/list 结构不一致（期望长度 {L}）。")
                for i in range(L):
                    per_pos[i].append(o[i])

            result = []
            for i in range(L):
                candidates = per_pos[i]
                if i == idx_main:
                    agg = None
                    for c in candidates:
                        t = self._to_tensor(c)
                        t = self._maybe_to_device(t)
                        t = self._to_probs_if_needed(t)
                        agg = t if agg is None else (agg + t)
                    out_i = agg / float(self._num_models)
                else:
                    try:
                        if all(isinstance(c, (torch.Tensor, np.ndarray)) for c in candidates):
                            shapes = [tuple(self._to_tensor(c).shape) for c in candidates]
                            if len(set(shapes)) == 1:
                                agg = None
                                for c in candidates:
                                    t = self._to_tensor(c)
                                    t = self._maybe_to_device(t)
                                    agg = t if agg is None else (agg + t)
                                out_i = agg / float(self._num_models)
                            else:
                                out_i = candidates[0]
                        else:
                            out_i = candidates[0]
                    except Exception:
                        out_i = candidates[0]
                result.append(out_i)
            return tuple(result) if isinstance(first_out, tuple) else result

        if isinstance(first_out, dict):
            key_main = self.primary_key if self.primary_key is not None else self._guess_primary_key(first_out)
            keys = list(first_out.keys())
            per_key: Dict[str, List[Any]] = {k: [] for k in keys}
            for o in all_outs:
                if not isinstance(o, dict):
                    raise TypeError("[错误] 子模型返回结构不一致：有的为 dict，有的不是。")
                for k in keys:
                    if k not in o:
                        raise KeyError(f"[错误] 子模型返回的 dict 缺少键：{k}")
                    per_key[k].append(o[k])

            result: Dict[str, Any] = {}
            for k in keys:
                candidates = per_key[k]
                if k == key_main:
                    agg = None
                    for c in candidates:
                        t = self._to_tensor(c)
                        t = self._maybe_to_device(t)
                        t = self._to_probs_if_needed(t)
                        agg = t if agg is None else (agg + t)
                    result[k] = agg / float(self._num_models)
                else:
                    try:
                        if all(isinstance(c, (torch.Tensor, np.ndarray)) for c in candidates):
                            shapes = [tuple(self._to_tensor(c).shape) for c in candidates]
                            if len(set(shapes)) == 1:
                                agg = None
                                for c in candidates:
                                    t = self._to_tensor(c)
                                    t = self._maybe_to_device(t)
                                    agg = t if agg is None else (agg + t)
                                result[k] = agg / float(self._num_models)
                            else:
                                result[k] = candidates[0]
                        else:
                            result[k] = candidates[0]
                    except Exception:
                        result[k] = candidates[0]
            return result

        print(f"[警告] 未支持的返回类型 {type(first_out).__name__}，将直接沿用第一个子模型输出。")
        return first_out

    @torch.no_grad()
    def _call_each(self, fn_name: str, *args, **kwargs) -> Any:
        outs = []
        for m in self.models:
            fn = getattr(m, fn_name)
            outs.append(fn(*args, **kwargs))
        return self._aggregate_like(outs[0], outs)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if hasattr(self.models[0], "forward"):
            return self._call_each("forward", *args, **kwargs)
        raise AttributeError("子模型未实现 forward。")

    @torch.no_grad()
    def inference(self, *args, **kwargs):
        if self._sub_has_inference:
            return self._call_each("inference", *args, **kwargs)
        return self._call_each("forward", *args, **kwargs)


# ================== 主流程 ==================
def main():
    parser = argparse.ArgumentParser(description="SleepDG 多折集成评估（完全由配置文件驱动）")
    parser.add_argument("--config", type=str, required=True, help="用于测试的 YAML 配置文件路径")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        print("[信息] 已加载测试配置：")
        print(yaml.dump(cfg, indent=2, allow_unicode=True))
    except (FileNotFoundError, TypeError, yaml.YAMLError) as e:
        print(f"[错误] 加载或解析配置文件失败: {e}")
        return

    params = dict_to_namespace(cfg)
    if hasattr(params, "align") and getattr(params.align, "use_alignment", False):
        centers_path = getattr(params.align, "cluster_centers_path", None)
        if centers_path and os.path.exists(centers_path):
            print(f"[信息] 检测到对齐配置，正在从 {centers_path} 加载几何中心...")
            try:
                centers_np = np.load(centers_path)
                params.external_centers = torch.from_numpy(centers_np).float()
                print(f"[信息] 几何中心加载成功，形状: {params.external_centers.shape}")
            except Exception as e:
                print(f"[错误] 加载几何中心文件失败: {e}")
                return
        else:
            print(f"[警告] 配置了 use_alignment 但未找到有效的 cluster_centers_path: {centers_path}")

    required_top = ["datasets_dir", "test_dataset", "batch_size", "num_workers"]
    for k in required_top:
        if not hasattr(params, k):
            print(f"[错误] 配置缺少必要键: {k}")
            return

    if not hasattr(params, "model"):
        print("[错误] 配置缺少必要键: model")
        return

    model_cfg = params.model

    if not has_attr(model_cfg, "ensemble_method"):
        setattr(model_cfg, "ensemble_method", "logits")
    if not has_attr(model_cfg, "strict_load"):
        setattr(model_cfg, "strict_load", True)
    if not has_attr(model_cfg, "use_dataparallel"):
        setattr(model_cfg, "use_dataparallel", False)

    if hasattr(params, "gpus"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpus)
    if hasattr(params, "seed"):
        setup_seed(int(params.seed))

    print("\n--- GPU 诊断 ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        current_device_index = torch.cuda.current_device()
        print(f"当前设备索引: {current_device_index}")
        print(f"设备名称: {torch.cuda.get_device_name(current_device_index)}")
    print("---------------------\n")

    try:
        ckpt_paths = resolve_ckpt_paths(model_cfg)
        print("[信息] 配置中解析到以下权重：")
        for p in ckpt_paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[错误] 权重解析失败: {e}")
        return

    try:
        selected_ckpt_paths = select_ckpts_for_dataset(params.test_dataset, ckpt_paths, cfg)
        eval_mode = infer_eval_mode(params.test_dataset, selected_ckpt_paths, cfg)
        dataset_group = get_dataset_group(params.test_dataset, cfg)

        print("[信息] 本次实际用于评估的权重：")
        for p in selected_ckpt_paths:
            print(f"  - {p}")
        print(f"[信息] 当前评估模式: {eval_mode}")
        print(f"[信息] 当前数据集分组: {dataset_group}")
    except Exception as e:
        print(f"[错误] 选择权重失败: {e}")
        return

    try:
        ModelClass = import_model_from_config(model_cfg)
    except Exception as e:
        print(f"[错误] 加载模型类失败: {e}")
        return

    print(f"[信息] 加载测试数据集: {params.test_dataset} ...")
    try:
        load_params = SimpleNamespace(
            datasets_dir=params.datasets_dir,
            test_dataset=params.test_dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers
        )
        data_loader_dict, _ = LoadDataset(load_params).get_data_loader()

        if "test" in data_loader_dict and data_loader_dict["test"] is not None and len(data_loader_dict["test"]) > 0:
            test_loader = data_loader_dict["test"]
        else:
            if "val" in data_loader_dict and data_loader_dict["val"] is not None and len(data_loader_dict["val"]) > 0:
                print("[警告] 'test' 为空，使用 'val' 加载器作为测试加载器。")
                test_loader = data_loader_dict["val"]
            elif "train" in data_loader_dict and data_loader_dict["train"] is not None and len(data_loader_dict["train"]) > 0:
                print("[警告] 'test' 和 'val' 均为空，使用 'train' 加载器作为测试加载器。")
                test_loader = data_loader_dict["train"]
            else:
                raise ValueError(f"无法为测试数据集加载数据: {params.test_dataset}。所有加载器均为空。")

        print(f"[信息] 测试数据加载完成。批次数: {len(test_loader)}")
    except Exception as e:
        print(f"[错误] 加载数据集 '{params.test_dataset}' 失败: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[信息] 构建集成模型 ...")
    try:
        ensemble = EnsembleWrapper(
            ModelClass=ModelClass,
            params=params,
            ckpt_paths=selected_ckpt_paths,
            ensemble_method=get_attr(model_cfg, "ensemble_method", "logits"),
            strict_load=bool(get_attr(model_cfg, "strict_load", True)),
            device=device,
            primary_index=get_attr(model_cfg, "primary_index", None),
            primary_key=get_attr(model_cfg, "primary_key", None),
        )
    except Exception as e:
        print(f"[错误] 构建或加载集成模型失败: {e}")
        return

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and get_attr(model_cfg, "use_dataparallel", False):
        print("[信息] 使用 DataParallel 进行评估。")
        ensemble = torch.nn.DataParallel(ensemble)

    print("\n--- 开始评估 ---")
    evaluator = Evaluator(params, test_loader)
    try:
        test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, \
            test_n3_f1, test_rem_f1, test_kappa, test_report = evaluator.get_accuracy(ensemble)
    except Exception as e:
        print(f"[错误] 评估阶段出错: {e}")
        return

    # ================== CI 计算 ==================
    ci = None
    acc_lo = acc_hi = None
    f1_lo = f1_hi = None
    k_lo = k_hi = None
    per_lo_arr = per_hi_arr = None
    n_subjects = None
    n_boot = int(getattr(params, "bootstrap_n", 1000))
    bs_seed = int(getattr(params, "bootstrap_seed", 0))
    decimals = int(getattr(params, "print_decimals", 2))

    try:
        print(f"\n--- 开始计算置信区间（subject-level bootstrap）---")
        eval_pack = evaluate_collect_subject_cms(ensemble, test_loader, device=device)
        subject_cms = eval_pack["subject_cms"]
        labels = eval_pack["labels"]

        ci = bootstrap_ci_95_subject_level(
            subject_cms,
            n_boot=n_boot,
            seed=bs_seed,
            batch_boot=512,
        )

        acc_lo, acc_hi = ci["acc_ci"]
        f1_lo, f1_hi = ci["macro_f1_ci"]
        k_lo, k_hi = ci["kappa_ci"]
        per_lo_arr, per_hi_arr = ci["per_class_f1_ci"]
        n_subjects = int(ci.get("n_subjects", subject_cms.shape[0]))
    except Exception as e:
        print(f"[警告] 置信区间计算失败，将仅保留点估计结果: {e}")
        labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    print(f"\n***************** 测试结果（{params.test_dataset}｜N={len(selected_ckpt_paths)}） *****************")
    print(f"模型类:             {get_attr(model_cfg, 'import_path')}.{get_attr(model_cfg, 'class_name')}")
    print(f"评估模式:           {eval_mode}")
    print(f"数据集分组:         {dataset_group}")
    print(f"集成方式:           {get_attr(model_cfg, 'ensemble_method', 'logits')}")
    print(f"权重文件数:         {len(selected_ckpt_paths)}")

    if ci is not None:
        print(f"Bootstrap:          subject-level, S={n_subjects}, n={n_boot}, seed={bs_seed}")
        print(f"测试准确率 (Acc):    {fmt_pm(float(test_acc), acc_lo, acc_hi, decimals=decimals, scale=100.0)}")
        print(f"测试宏 F1 分数:     {fmt_pm(float(test_f1), f1_lo, f1_hi, decimals=decimals, scale=100.0)}")
        print(f"测试 Cohen's Kappa:  {fmt_pm(float(test_kappa), k_lo, k_hi, decimals=decimals, scale=100.0)}")
    else:
        print(f"测试准确率 (Acc):    {float(test_acc) * 100:.{decimals}f}")
        print(f"测试宏 F1 分数:     {float(test_f1) * 100:.{decimals}f}")
        print(f"测试 Cohen's Kappa:  {float(test_kappa) * 100:.{decimals}f}")

    print(f"\nWake F1:            {float(test_wake_f1):.5f}")
    print(f"N1 F1:              {float(test_n1_f1):.5f}")
    print(f"N2 F1:              {float(test_n2_f1):.5f}")
    print(f"N3 F1:              {float(test_n3_f1):.5f}")
    print(f"REM F1:             {float(test_rem_f1):.5f}")

    if ci is not None:
        default_name_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
        class_names = [default_name_map.get(int(l), f"Class{int(l)}") for l in labels]
        point_per_f1 = [float(test_wake_f1), float(test_n1_f1), float(test_n2_f1), float(test_n3_f1), float(test_rem_f1)]

        print("\n各类别 F1 分数 (95% CI):")
        for i, name in enumerate(class_names):
            if i < len(point_per_f1) and i < len(per_lo_arr):
                print(f"  {name}: {fmt_pm(point_per_f1[i], float(per_lo_arr[i]), float(per_hi_arr[i]), decimals=decimals, scale=100.0)}")

    print("\n混淆矩阵:")
    print(test_cm)

    print("\n分类报告:")
    print(test_report)
    print("******************************************************************")

    results_root = Path(getattr(params, "results_log_dir", "./results"))
    results_root.mkdir(parents=True, exist_ok=True)
    results_output_dir = results_root / "test_results"
    results_output_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_output_dir / f"test_on_{params.test_dataset}_ensemble{len(selected_ckpt_paths)}.txt"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"数据集测试结果: {params.test_dataset}\n")
            f.write(f"模型类: {get_attr(model_cfg, 'import_path')}.{get_attr(model_cfg, 'class_name')}\n")
            f.write(f"评估模式: {eval_mode}\n")
            f.write(f"数据集分组: {dataset_group}\n")
            f.write(f"集成方式: {get_attr(model_cfg, 'ensemble_method', 'logits')}\n")
            f.write(f"权重文件数: {len(selected_ckpt_paths)}\n")
            for i, p in enumerate(selected_ckpt_paths):
                f.write(f"  [{i}] {p}\n")

            if ci is not None:
                f.write(f"Bootstrap: subject-level, S={n_subjects}, n={n_boot}, seed={bs_seed}\n")
                f.write(f"准确率 (Acc, 95% CI): {fmt_pm(float(test_acc), acc_lo, acc_hi, decimals=decimals, scale=100.0)}\n")
                f.write(f"宏 F1 (95% CI):        {fmt_pm(float(test_f1), f1_lo, f1_hi, decimals=decimals, scale=100.0)}\n")
                f.write(f"Kappa (95% CI):        {fmt_pm(float(test_kappa), k_lo, k_hi, decimals=decimals, scale=100.0)}\n")
                f.write(f"Acc 区间:              {safe_ci_str(acc_lo, acc_hi, decimals=2, scale=100.0)}\n")
                f.write(f"MF1 区间:              {safe_ci_str(f1_lo, f1_hi, decimals=2, scale=100.0)}\n")
                f.write(f"Kappa 区间:            {safe_ci_str(k_lo, k_hi, decimals=2, scale=100.0)}\n")
            else:
                f.write(f"准确率 (Acc): {float(test_acc):.6f}\n")
                f.write(f"宏 F1:        {float(test_f1):.6f}\n")
                f.write(f"Kappa:        {float(test_kappa):.6f}\n")

            f.write(f"\nWake F1: {float(test_wake_f1):.6f}\n")
            f.write(f"N1 F1:   {float(test_n1_f1):.6f}\n")
            f.write(f"N2 F1:   {float(test_n2_f1):.6f}\n")
            f.write(f"N3 F1:   {float(test_n3_f1):.6f}\n")
            f.write(f"REM F1:  {float(test_rem_f1):.6f}\n")

            if ci is not None:
                default_name_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
                class_names = [default_name_map.get(int(l), f"Class{int(l)}") for l in labels]
                point_per_f1 = [float(test_wake_f1), float(test_n1_f1), float(test_n2_f1), float(test_n3_f1), float(test_rem_f1)]

                f.write("\n各类别 F1 (95% CI):\n")
                for i, name in enumerate(class_names):
                    if i < len(point_per_f1) and i < len(per_lo_arr):
                        f.write(f"  {name}: {fmt_pm(point_per_f1[i], float(per_lo_arr[i]), float(per_hi_arr[i]), decimals=decimals, scale=100.0)}\n")

            f.write("\n混淆矩阵:\n")
            f.write(str(test_cm))
            f.write("\n\n分类报告:\n")
            f.write(str(test_report))

        print(f"[信息] 测试结果已保存至: {results_file}")
    except Exception as e:
        print(f"[警告] 保存详细测试结果失败: {e}")

    record = build_result_record(
        params=params,
        model_cfg=model_cfg,
        test_dataset=params.test_dataset,
        dataset_group=dataset_group,
        eval_mode=eval_mode,
        selected_ckpt_paths=selected_ckpt_paths,
        test_acc=test_acc,
        test_f1=test_f1,
        test_kappa=test_kappa,
        test_wake_f1=test_wake_f1,
        test_n1_f1=test_n1_f1,
        test_n2_f1=test_n2_f1,
        test_n3_f1=test_n3_f1,
        test_rem_f1=test_rem_f1,
        acc_ci_lo=acc_lo,
        acc_ci_hi=acc_hi,
        mf1_ci_lo=f1_lo,
        mf1_ci_hi=f1_hi,
        kappa_ci_lo=k_lo,
        kappa_ci_hi=k_hi,
        n_subjects=n_subjects,
        bootstrap_n=n_boot if ci is not None else None,
        bootstrap_seed=bs_seed if ci is not None else None,
    )

    try:
        update_result_tables(results_output_dir, record, cfg)
    except Exception as e:
        print(f"[警告] 更新汇总表失败: {e}")


if __name__ == "__main__":
    main()
import math
import os
import random
import sys
import json
from typing import Optional, Tuple

# Make project root importable
cwd = os.getcwd()
project_root = os.path.abspath(os.path.join(cwd, ".."))
for path in {cwd, project_root}:
    if path not in sys.path:
        sys.path.append(path)

import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from Experiment.models import AccretionTransformer, AccretionConvNet, MACNetRes_mbh, MACNetFiLM


MBH_ONLY_MODELS = (MACNetRes_mbh, MACNetFiLM)


def predict_batch(model, x: torch.Tensor, coord: torch.Tensor, mbh: torch.Tensor) -> torch.Tensor:
    if isinstance(model, MBH_ONLY_MODELS):
        return model(x, mbh)
    r = coord[:, 0:1]
    theta = coord[:, 1:2]
    return model(x, r, theta, mbh)


# =============================================
# 0) Augmentations
# =============================================
class RadialCropAugmentor:
    """
    Implements radial cropping from outer to inner.
    Data format: (C, H, W), Coord format: (2, H, W) where coord[0] is r, coord[1] is theta.

    The augmentation randomly selects a cutoff radius 'b' between (r_max * min_ratio) and r_max.
    All data points where r > b are masked (set to 0).
    """

    def __init__(self, min_crop_ratio: float = 0.8, p_crop: float = 0.5):
        self.min_crop_ratio = max(0.0, min(1.0, min_crop_ratio))
        self.p_crop = max(0.0, min(1.0, p_crop))

    def __call__(self, x: torch.Tensor, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() >= self.p_crop:
            return x, coord

        # coord shape: (2, H, W), coord[0] is r
        r_map = coord[0]
        r_max = float(r_map.max())
        r_min = float(r_map.min())  # Usually close to 0

        # Determine cutoff b
        # We want to keep range (0, b).
        # b should be randomly chosen in [r_max * min_ratio, r_max]
        # But closer to r_min logic: technically user said "0 < b < a".
        # We ensure b is at least retaining a significant portion of the core.

        lower_bound = r_max * self.min_crop_ratio
        if lower_bound <= r_min:
            lower_bound = r_min + (r_max - r_min) * 0.1  # Safety guard

        b = random.uniform(lower_bound, r_max)

        # Create mask: True where r > b (the outer part to be removed/masked)
        mask = r_map > b

        if not mask.any():
            return x, coord

        x = x.clone()
        # Apply mask to all channels
        # x is (C, H, W), mask is (H, W)
        x[:, mask] = 0.0

        return x, coord


# =============================================
# 1) Dataset: load .pt [x, coord, y, mbh, (Optional: y_bondi)]
# =============================================
class GalPTDataset(Dataset):
    def __init__(self, pt_path, *, augmentor: Optional[RadialCropAugmentor] = None):
        super().__init__()
        try:
            loaded = torch.load(pt_path, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load dataset cache '{pt_path}'. ") from exc

        if isinstance(loaded, (list, tuple)):
            if len(loaded) == 6:
                self.x, self.coord, self.y, self.mbh, self.groups, self.n_groups = loaded
                self.y_bondi = None
            elif len(loaded) >= 7:
                self.x, self.coord, self.y, self.mbh, self.y_bondi, self.groups, self.n_groups = loaded[:7]
                if len(loaded) > 7:
                    print(
                        f"[Warning] Dataset '{pt_path}' has {len(loaded)} items; "
                        "extra fields are ignored."
                    )
            else:
                raise RuntimeError(f"Unexpected number of items in cached dataset '{pt_path}': {len(loaded)}")
        else:
            raise RuntimeError(f"Cached dataset '{pt_path}' should be a tuple/list but got {type(loaded)!r}")

        def _ensure_tensor(arr, *, name: str):
            if isinstance(arr, torch.Tensor):
                return arr.to(dtype=torch.float32)

            # Special handling for potentially string-based 'groups'
            if name == "groups":
                if isinstance(arr, list):
                    arr = np.array(arr)
                if isinstance(arr, np.ndarray) and arr.dtype.kind in "SU":
                    return arr  # Keep as numpy string array for later processing

            try:
                return torch.as_tensor(arr, dtype=torch.float32)
            except Exception as inner_exc:
                raise TypeError(
                    f"Cached field '{name}' in '{pt_path}' cannot be converted " "to a float32 tensor."
                ) from inner_exc

        self.x = _ensure_tensor(self.x, name="x").contiguous()
        self.coord = _ensure_tensor(self.coord, name="coord").contiguous()
        self.y = _ensure_tensor(self.y, name="y").view(-1, 1)
        self.mbh = _ensure_tensor(self.mbh, name="mbh").view(-1, 1)
        self.y_bondi = _ensure_tensor(self.y_bondi, name="y_bondi").view(-1, 1) if self.y_bondi is not None else None

        groups_raw = _ensure_tensor(self.groups, name="groups")
        if isinstance(groups_raw, np.ndarray) and groups_raw.dtype.kind in "SU":
            # Handle string groups
            raw_groups = groups_raw.reshape(-1, 1)
            unique_groups, inv = np.unique(raw_groups, return_inverse=True)
            self.n_groups = len(unique_groups)
            self.groups = torch.from_numpy(inv).long().view(-1, 1)
        else:
            # Handle numeric groups (already a tensor from _ensure_tensor)
            if groups_raw.ndim == 2 and groups_raw.size(1) > 1:
                # Keep only the last column if multiple columns exist (e.g., [type, subtype])
                # to ensure one label per sample.
                raw_groups = groups_raw[:, -1:]
            else:
                raw_groups = groups_raw.view(-1, 1)

            # Remap groups to 0..n_groups-1 for robust indexing in eval_epoch
            unique_groups, inv = torch.unique(raw_groups, return_inverse=True)
            self.n_groups = len(unique_groups)
            self.groups = inv.long().view(-1, 1)

        self.augmentor = augmentor

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        coord = self.coord[idx]
        if self.augmentor is not None:
            x, coord = self.augmentor(x, coord)
        return x, coord, self.mbh[idx], self.y[idx], self.groups[idx], self.n_groups


# =============================================
# 2) Args
# =============================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MACNetRes with MBH input")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory containing cached .pt datasets")
    parser.add_argument("--exp_name", type=str, default="Exp5_", help="Experiment name prefix used for saving results")
    parser.add_argument(
        "--data_exp", type=str, default=None, help="Experiment name prefix for loading data (default: same as exp_name)"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="AdamW weight decay (L2 regularization)")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate once cosine decay completes")
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of linear warmup steps before cosine decay (0 to disable)",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.2,
        help="If warmup steps is 0, use this fraction of total steps for warmup (0 disables)",
    )
    parser.add_argument(
        "--lr_warmup_start_factor",
        type=float,
        default=0.0,
        help="Relative learning rate factor to start warmup from (0.0 starts from zero)",
    )
    default_workers = 0 if os.name == "nt" else min(16, (os.cpu_count() or 2) // 2)
    parser.add_argument("--num_workers", type=int, default=default_workers, help="Number of dataloader workers")
    parser.add_argument(
        "--persistent_workers", action="store_true", help="Keep dataloader workers persistent between epochs"
    )
    parser.add_argument("--prefetch_factor", type=int, default=3, help="Prefetch factor for dataloader (per worker)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--print_group_counts", action="store_true", help="Print per-group sample counts for loaders")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use: 'auto', 'cpu' or 'cuda'",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without improvement)")
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable automatic mixed precision (AMP) even if CUDA is available"
    )
    parser.add_argument("--grad_clip", type=float, default=0.3, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument(
        "--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark mode for potential speedup"
    )
    parser.add_argument("--loss", type=str, choices=["mse", "l1", "huber"], default="mse", help="Loss function to use")
    parser.add_argument("--huber_delta", type=float, default=1, help="Delta parameter for Huber loss")
    parser.add_argument(
        "--weight_mode",
        type=str,
        choices=["exp_clamp", "group_zscore_exp", "group_quantile"],
        default="group_zscore_exp",
        help="Weighting mode for targets",
    )
    parser.add_argument("--weight_alpha", type=float, default=0.5, help="Alpha for target weighting")
    parser.add_argument("--weight_tau", type=float, default=3.0, help="Temperature for z-score exp weights")
    parser.add_argument("--weight_power", type=float, default=1.0, help="Power for quantile weights")
    parser.add_argument("--weight_q_low", type=float, default=0.1, help="Lower quantile for group quantile weights")
    parser.add_argument("--weight_q_high", type=float, default=0.9, help="Upper quantile for group quantile weights")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["transformer", "resnet", "resfilm", "convnet"],
        default="transformer",
        help="Type of model architecture: 'transformer', 'resnet', 'resfilm', or 'convnet'",
    )
    parser.add_argument("--d_model", type=int, default=256, help="Model embedding dimension (d_model)")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward network hidden dimension")
    parser.add_argument("--pos_num_bands", type=int, default=32, help="Number of positional encoding frequency bands")
    parser.add_argument("--pos_max_freq", type=float, default=101, help="Maximum positional encoding frequency")
    parser.add_argument("--p_drop", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--base_ch", type=int, default=64, help="ConvNet base channels (convnet only)")
    parser.add_argument(
        "--stage_depths",
        type=str,
        default="2,2,2,2",
        help="ConvNet stage depths as comma list (e.g., 2,2,2,2)",
    )
    parser.add_argument("--se_reduction", type=int, default=4, help="ConvNet SE reduction ratio (convnet only)")
    parser.add_argument(
        "--hpo_params",
        type=str,
        default=None,
        help="Path to Optuna/JSON HPO params file. Matching CLI args will be overridden",
    )

    # Augmentation args
    parser.add_argument("--aug_prob", type=float, default=0.8, help="Probability to apply radial crop")
    parser.add_argument("--aug_min_crop_ratio", type=float, default=0.6, help="Minimum ratio r_cut/r_max for cropping")

    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to write TensorBoard logs")
    parser.add_argument("--no_save", action="store_true", help="Disable saving results to disk (useful for HPO)")

    return parser.parse_args()


def _parse_stage_depths(value: str) -> Tuple[int, ...]:
    raw = value.strip()
    if not raw:
        return (2, 2, 2, 2)

    if raw.startswith("[") or raw.startswith("("):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return tuple(int(x) for x in parsed)

    try:
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid stage_depths value: {value!r}") from exc


def _apply_hpo_overrides(args: argparse.Namespace) -> argparse.Namespace:
    path = getattr(args, "hpo_params", None)
    if not path:
        return args
    if not os.path.isfile(path):
        print(f"[Warning] HPO params file not found: {path}")
        return args
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[Warning] Failed to read HPO params '{path}': {exc}")
        return args

    converters = {
        "lr": float,
        "weight_decay": float,
        "lr_min": float,
        "lr_warmup_steps": int,
        "lr_warmup_ratio": float,
        "lr_warmup_start_factor": float,
        "batch_size": int,
        "grad_clip": float,
        "patience": int,
        "loss": str,
        "huber_delta": float,
        "weight_mode": str,
        "weight_alpha": float,
        "weight_tau": float,
        "weight_power": float,
        "weight_q_low": float,
        "weight_q_high": float,
        "d_model": int,
        "n_layers": int,
        "n_heads": int,
        "d_ff": int,
        "pos_num_bands": int,
        "pos_max_freq": float,
        "p_drop": float,
        "seed": int,
        "aug_prob": float,
        "aug_min_crop_ratio": float,
    }

    applied = []
    for key, convert in converters.items():
        if key not in payload:
            continue
        try:
            value = convert(payload[key])
        except (TypeError, ValueError):
            print(f"[Warning] Could not cast HPO param '{key}' (value={payload[key]!r})")
            continue
        if key == "loss":
            value = value.lower()
        setattr(args, key, value)
        applied.append(key)

    if applied:
        print(f"Loaded HPO params from {path}; applied keys: {', '.join(applied)}")
    else:
        print(f"[Warning] No matching keys found in HPO params: {path}")

    return args


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism users can also set CUDNN flags; keep perf by default.


# =============================================
# 3) Train and evaluate
# =============================================
class WeightedLoss(nn.Module):
    def __init__(self, base_criterion: nn.Module, weights_fn: callable = None):
        """
        A loss wrapper that applies per-sample weights to the base criterion.
         Args:
             base_criterion (nn.Module): The base loss function (e.g., nn.MSELoss).
             weights_fn (callable, optional): A function that takes the target tensor `y` and returns
                 a tensor of weights with the same shape as `y`. If None, uniform weights are used.
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.weights_fn = weights_fn if weights_fn is not None else (lambda y, groups=None: torch.ones_like(y))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, groups: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.base_criterion(pred, target)
        weights = self.weights_fn(target, groups)

        weighted_loss = loss * weights
        return weighted_loss.mean()


def _compute_group_stats(
    y: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int,
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> dict[str, torch.Tensor]:
    y = y.view(-1)
    groups = groups.view(-1).long()
    means = torch.zeros(n_groups, dtype=y.dtype)
    stds = torch.ones(n_groups, dtype=y.dtype)
    p10 = torch.zeros(n_groups, dtype=y.dtype)
    p90 = torch.ones(n_groups, dtype=y.dtype)
    for g in range(n_groups):
        mask = groups == g
        if not mask.any():
            continue
        y_g = y[mask]
        means[g] = y_g.mean()
        stds[g] = y_g.std(unbiased=False)
        p10[g] = torch.quantile(y_g, q_low)
        p90[g] = torch.quantile(y_g, q_high)
    return {"mean": means, "std": stds, "p10": p10, "p90": p90}


def make_criterion(
    kind: str,
    delta: float,
    alpha: float = 1.0,
    weight_mode: str = "group_zscore_exp",
    group_stats: dict[str, torch.Tensor] | None = None,
    tau: float = 1.0,
    power: float = 1.0,
):
    if kind == "mse":
        base = nn.MSELoss(reduction="none")
    elif kind == "l1":
        base = nn.L1Loss(reduction="none")
    elif kind == "huber":
        base = nn.HuberLoss(delta=delta, reduction="none")
    else:
        raise ValueError(kind)

    def weight_exp(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * torch.exp(target)
    
    def weight_exp_clamp(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * torch.clamp(torch.exp(target), max=1e-2)  
    
    def weight_linear(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * target
    
    def weight_softplus(target: torch.Tensor, groups: torch.Tensor | None = None):
        return 1.0 + alpha * F.softplus(target)

    def weight_group_zscore_exp(target: torch.Tensor, groups: torch.Tensor | None = None):
        if groups is None or group_stats is None:
            raise ValueError("group_zscore_exp requires groups and group_stats")
        g = groups.view(-1).long().to(device=target.device)
        mean = group_stats["mean"].to(device=target.device)[g].view_as(target)
        std = group_stats["std"].to(device=target.device)[g].view_as(target).clamp_min(1e-6)
        z = (target - mean) / std
        return 1.0 + alpha * torch.exp(z / max(1e-6, tau))

    def weight_group_quantile(target: torch.Tensor, groups: torch.Tensor | None = None):
        if groups is None or group_stats is None:
            raise ValueError("group_quantile requires groups and group_stats")
        g = groups.view(-1).long().to(device=target.device)
        p10 = group_stats["p10"].to(device=target.device)[g].view_as(target)
        p90 = group_stats["p90"].to(device=target.device)[g].view_as(target)
        denom = (p90 - p10).clamp_min(1e-6)
        q = ((target - p10) / denom).clamp(0.0, 1.0)
        return 1.0 + alpha * torch.pow(q, max(1e-6, power))

    if weight_mode == "exp_clamp":
        weights_fn = weight_exp_clamp
    elif weight_mode == "group_zscore_exp":
        weights_fn = weight_group_zscore_exp
    elif weight_mode == "group_quantile":
        weights_fn = weight_group_quantile
    else:
        raise ValueError(weight_mode)

    return WeightedLoss(base_criterion=base, weights_fn=weights_fn)


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float,
        warmup_start_factor: float,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive for WarmupCosineScheduler")

        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_steps))
        self.decay_steps = max(1, self.total_steps - self.warmup_steps)
        self.start_factor = float(max(0.0, min(warmup_start_factor, 1.0)))
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        min_lr = max(0.0, float(min_lr))
        self.min_lrs = [min(min_lr, base) for base in self.base_lrs]
        self.step_count = 0
        self._last_lr = []
        self._set_lrs(0)

    def _compute_lr(self, step: int) -> list[float]:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            warmup_progress = step / max(1, self.warmup_steps)
            factor = self.start_factor + (1.0 - self.start_factor) * warmup_progress
            return [base * factor for base in self.base_lrs]

        progress = (step - self.warmup_steps) / max(1, self.decay_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [min_lr + (base - min_lr) * cosine for base, min_lr in zip(self.base_lrs, self.min_lrs)]

    def _set_lrs(self, step: int) -> None:
        lrs = self._compute_lr(step)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._last_lr = lrs

    def step(self) -> None:
        if self.step_count >= self.total_steps:
            self._set_lrs(self.total_steps)
            return
        self.step_count += 1
        self._set_lrs(self.step_count)

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None, scheduler=None):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)  # (B, C_in, H, W)
        coord = batch[1].to(device, non_blocking=True)  # (B, 2, H, W)
        mbh = batch[2].to(device, non_blocking=True).view(-1)  # (B,)
        y = batch[3].to(device, non_blocking=True).view(-1)  # (B,)
        groups = batch[4].to(device, non_blocking=True).view(-1)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = predict_batch(model, x, coord, mbh)
                loss = criterion(pred, y, groups)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        else:
            pred = predict_batch(model, x, coord, mbh)
            loss = criterion(pred, y, groups)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        bs = y.size(0)
        total += loss.detach().item() * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def print_group_counts(loader, name: str):
    counts = None
    for batch in loader:
        groups = batch[4].view(-1)
        if counts is None:
            n_groups_tensor = batch[5]
            if isinstance(n_groups_tensor, torch.Tensor):
                n_groups = int(n_groups_tensor.flatten()[0].item())
            else:
                n_groups = int(n_groups_tensor)
            counts = [0] * n_groups
        for g in range(len(counts)):
            counts[g] += int((groups == g).sum().item())

    if counts is None:
        print(f"[{name}] loader is empty; no group counts available.")
    else:
        print(f"[{name}] group counts: {counts}")


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, writer, epoch):
    model.eval()
    try:
        first_batch = next(iter(loader))
    except StopIteration:
        print(f"[{epoch}] Warning: eval_epoch received an empty loader.")
        return float("nan"), []

    n_groups_tensor = first_batch[5]
    if isinstance(n_groups_tensor, torch.Tensor):
        n_groups = int(n_groups_tensor.flatten()[0].item())
    else:
        n_groups = int(n_groups_tensor)
    del first_batch

    if n_groups <= 0:
        # Fallback if no groups are defined
        total_loss = 0.0
        n_samples = 0
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            coord = batch[1].to(device, non_blocking=True)
            mbh = batch[2].to(device, non_blocking=True).view(-1)
            y = batch[3].to(device, non_blocking=True).view(-1)
            pred = predict_batch(model, x, coord, mbh)
            loss = criterion(pred, y).mean().item()
            total_loss += loss * y.size(0)
            n_samples += y.size(0)
        avg = total_loss / max(1, n_samples)
        return avg, [avg]

    groups_sum = [0.0] * n_groups
    groups_count = [0] * n_groups

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        coord = batch[1].to(device, non_blocking=True)
        mbh = batch[2].to(device, non_blocking=True).view(-1)
        y = batch[3].to(device, non_blocking=True).view(-1)
        groups = batch[4].view(-1).to(device=mbh.device, dtype=torch.long)

        pred = predict_batch(model, x, coord, mbh)

        if isinstance(criterion, WeightedLoss):
            base_loss = criterion.base_criterion(pred, y)  # shape: (B, ...) or (B,)
            weights = criterion.weights_fn(y, groups)  # 同形或可广播
            per_sample_loss = base_loss * weights
        elif isinstance(criterion, nn.HuberLoss):
            per_sample_loss = F.huber_loss(pred, y, reduction="none", delta=getattr(criterion, "delta", 1.0))
        elif isinstance(criterion, nn.MSELoss):
            per_sample_loss = F.mse_loss(pred, y, reduction="none")
        elif isinstance(criterion, nn.L1Loss):
            per_sample_loss = F.l1_loss(pred, y, reduction="none")
        else:
            per_sample_loss = criterion(pred, y)

        # (B,) ensure shape: if multi-dimensional, average over non-batch dimensions first
        if per_sample_loss.ndim > 1:
            per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)

        # Cumulative by domain
        for g in range(n_groups):
            m = groups == g
            if m.any():
                groups_sum[g] += per_sample_loss[m].mean().item() * m.sum().item()
                groups_count[g] += m.sum().item()

    # Domain-by-domain mean and worst domain
    groups_mean = [(groups_sum[d] / groups_count[d]) if groups_count[d] > 0 else float("nan") for d in range(n_groups)]
    valid_means = [v for v in groups_mean if not (v != v)]  # 跳过 NaN

    if not valid_means:
        print(
            f"[{epoch}] Warning: eval_epoch found no valid group means. n_groups={n_groups}, groups_count={groups_count}"
        )
        worst = float("nan")
    else:
        worst = max(valid_means)

    # Write to TensorBoard
    for d, v in enumerate(groups_mean):
        if v == v:
            writer.add_scalar(f"val/loss_domain_{d}", v, epoch)
    writer.add_scalar("val/loss_worst_domain", worst, epoch)

    # Return worst domain loss and all domain means
    return worst, groups_mean


# =============================================
# 4) Main
# =============================================
def main(args=None):
    if args is None:
        args = parse_args()
    args = _apply_hpo_overrides(args)
    set_seed(args.seed)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"Using device: {device}")
    # Initialize TensorBoard writer
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.exp_name}_{run_stamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Datasets / Loaders
    cache_dir = args.cache_dir
    exp = args.exp_name
    data_exp = args.data_exp if args.data_exp else exp

    train_augmentor = RadialCropAugmentor(min_crop_ratio=args.aug_min_crop_ratio, p_crop=args.aug_prob)

    train_ds = GalPTDataset(os.path.join(cache_dir, data_exp + "train.pt"), augmentor=train_augmentor)
    val_ds = GalPTDataset(os.path.join(cache_dir, data_exp + "val.pt"))
    test_ds = GalPTDataset(os.path.join(cache_dir, data_exp + "test.pt"))

    group_stats = _compute_group_stats(
        train_ds.y,
        train_ds.groups,
        int(train_ds.n_groups),
        q_low=args.weight_q_low,
        q_high=args.weight_q_high,
    )

    common = dict(batch_size=args.batch_size, pin_memory=(device.type == "cuda"), drop_last=False)
    dl_extra = {}
    if args.num_workers > 0:
        # On Windows, persistent_workers=True is highly recommended to avoid repeated spawning
        pw = args.persistent_workers or (os.name == "nt")
        dl_extra.update(dict(persistent_workers=pw, prefetch_factor=max(1, args.prefetch_factor)))

    train_loader = DataLoader(train_ds, shuffle=True, num_workers=max(0, args.num_workers // 2), **common, **dl_extra)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=max(0, args.num_workers // 2), **common, **dl_extra)
    test_loader = DataLoader(test_ds, shuffle=False, num_workers=max(0, args.num_workers // 2), **common, **dl_extra)

    if args.print_group_counts:
        print_group_counts(train_loader, "train")
        print_group_counts(val_loader, "val")
        print_group_counts(test_loader, "test")

    # Model
    c_in = train_ds.x.shape[1]
    if args.model_type == "hybrid":
        raise ValueError("Hybrid model type is not implemented in this script.")
    elif args.model_type == "transformer":
        model = AccretionTransformer(
            c_in=c_in,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            pos_num_bands=args.pos_num_bands,
            pos_max_freq=args.pos_max_freq,
            p_drop=args.p_drop,
        ).to(device)
    elif args.model_type == "resnet":
        model = MACNetRes_mbh(
            in_channels=c_in,
        ).to(device)
    elif args.model_type == "resfilm":
        model = MACNetFiLM(
            in_channels=c_in,
        ).to(device)
    elif args.model_type == "convnet":
        stage_depths = _parse_stage_depths(args.stage_depths)
        model = AccretionConvNet(
            c_in=c_in,
            base_ch=args.base_ch,
            stage_depths=stage_depths,
            pos_num_bands=args.pos_num_bands,
            pos_max_freq=args.pos_max_freq,
            p_drop=args.p_drop,
            se_reduction=args.se_reduction,
        ).to(device)

    try:
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            # Use a tiny temp loader with 0 workers to fetch a sample for the graph
            temp_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
            sample_x, sample_coord, sample_mbh, _, _, _ = next(iter(temp_loader))
            if isinstance(model, MBH_ONLY_MODELS):
                writer.add_graph(model, (sample_x.to(device), sample_mbh.view(-1).to(device)))
            else:
                writer.add_graph(
                    model,
                    (
                        sample_x.to(device),
                        sample_coord[:, 0:1].to(device),
                        sample_coord[:, 1:2].to(device),
                        sample_mbh.view(-1).to(device),
                    ),
                )
    except Exception as exc:
        print(f"Warning: could not write model graph to TensorBoard: {exc}")

    # Opt / Sched / Loss
    criterion = make_criterion(
        args.loss,
        args.huber_delta,
        alpha=args.weight_alpha,
        weight_mode=args.weight_mode,
        group_stats=group_stats,
        tau=args.weight_tau,
        power=args.weight_power,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * max(0, args.epochs)
    scheduler = None
    if total_steps > 0:
        warmup_steps = max(0, args.lr_warmup_steps)
        warmup_ratio = max(0.0, min(1.0, args.lr_warmup_ratio))
        if warmup_steps == 0 and warmup_ratio > 0.0:
            warmup_steps = int(total_steps * warmup_ratio)
        warmup_steps = min(warmup_steps, total_steps)
        try:
            scheduler = WarmupCosineScheduler(
                optimizer=optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=args.lr_min,
                warmup_start_factor=args.lr_warmup_start_factor,
            )
        except ValueError as exc:
            print(f"[Warning] Could not initialize scheduler: {exc}. Falling back to constant LR.")
            scheduler = None

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Save dirs
    if not args.no_save:
        save_dir = os.path.join("Results", f"{exp}{run_stamp}")
        os.makedirs(save_dir, exist_ok=True)
        hyperparams_path = os.path.join(save_dir, "hyperparameters.json")
        hyperparams = vars(args).copy()
        hyperparams.update(
            {
                "resolved_device": str(device),
                "log_dir": log_dir,
                "save_dir": save_dir,
                "run_stamp": run_stamp,
            }
        )
        with open(hyperparams_path, "w", encoding="utf-8") as f:
            json.dump(hyperparams, f, indent=2, sort_keys=True)
        best_path = os.path.join(save_dir, f"{exp}best_model.pth")
    else:
        save_dir = None
        best_path = None

    # Train
    best_val = float("inf")
    best_model_state = None  # To store best model in memory if no_save is True

    patience = max(0, args.patience)
    no_improve = 0
    print("Starting training")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler if use_amp else None,
            grad_clip=(args.grad_clip if args.grad_clip > 0 else None),
            scheduler=scheduler,
        )
        worst_val, group_means = eval_epoch(model, val_loader, criterion, device, writer, epoch)

        valid_vals = [v for v in group_means if v == v]
        mean_val = sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")

        print(
            f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {mean_val:.6f} | worst {worst_val:.6f} | lr {optimizer.param_groups[0]['lr']:.3e}"
        )

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("val/loss_mean_over_domains", mean_val, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if worst_val < best_val - 1e-8:
            best_val = worst_val
            no_improve = 0
            if args.no_save:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                torch.save(
                    {"model": model.state_dict(), "val_loss": best_val, "mean_val": mean_val, "epoch": epoch}, best_path
                )
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    has_best = False
    if args.no_save:
        if best_val != float("inf") and best_model_state is not None:
            has_best = True
    else:
        if best_val != float("inf") and os.path.exists(best_path):
            has_best = True

    if not has_best:
        print("No improvement was found during training. Best model not saved (or not found).")
        return float("inf")

    print(f"Best val loss: {best_val:.6f}. Loading best and testing.")

    if args.no_save:
        model.load_state_dict(best_model_state)
    else:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    test_loss, group_means = eval_epoch(model, test_loader, criterion, device, writer, epoch=0)
    print(f"Test loss: {test_loss:.6f}, per-domain: {group_means}")
    if not args.no_save:
        print(f"Best model saved to: {best_path}")

    return best_val


if __name__ == "__main__":
    main()

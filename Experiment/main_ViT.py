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

from Experiment.Network import AccretionTransformer, MACNetRes_mbh, MACNetFiLM


VELOCITY_CHANNEL_COUNT = 3
TANGENTIAL_CHANNEL_OFFSET = 1


def build_channel_protocol(
    num_channels: int, velocity_channels: int = VELOCITY_CHANNEL_COUNT
) -> dict[str, Tuple[int, ...]]:
    if num_channels < velocity_channels:
        raise ValueError(
            f"Expected at least {velocity_channels} velocity channels but got {num_channels} total channels."
        )
    velocity = tuple(range(velocity_channels))
    scalar = tuple(range(velocity_channels, num_channels))
    return {"velocity": velocity, "scalar": scalar}


DEFAULT_CHANNEL_PROTOCOL = {
    "velocity": tuple(range(VELOCITY_CHANNEL_COUNT)),
    "scalar": tuple(),
}
DEFAULT_V_THETA_INDEX = (
    DEFAULT_CHANNEL_PROTOCOL["velocity"][TANGENTIAL_CHANNEL_OFFSET]
    if len(DEFAULT_CHANNEL_PROTOCOL["velocity"]) > TANGENTIAL_CHANNEL_OFFSET
    else None
)


# =============================================
# 0) Augmentations
# =============================================
class RandomSpatialAugmentor:
    def __init__(
        self,
        p_theta_roll: float = 0.5,
        p_theta_mirror: float = 0.5,
        p_cutout: float = 0.3,
        erase_scale: Tuple[float, float] = (0.08, 0.25),
        angle_min: float = 0.05,
        angle_max: float = math.pi - 0.05,
        p_gaussian_blur: float = 0.2,
        blur_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        blur_sigma: Tuple[float, float] = (0.6, 1.6),
        scalar_jitter_std: float = 0.04,
        scalar_jitter_radial_power: float = 1.5,
        cutout_outer_bias: float = 2.5,
        cutout_radial_span: Tuple[float, float] = (0.05, 0.2),
        cutout_theta_span: Tuple[float, float] = (0.15, 0.5),
        channel_protocol: Optional[dict[str, Tuple[int, ...]]] = None,
        v_theta_index: Optional[int] = None,
    ) -> None:
        self.p_theta_roll = p_theta_roll
        self.p_theta_mirror = p_theta_mirror
        self.p_cutout = p_cutout
        self.erase_scale = erase_scale
        if not 0.0 <= angle_min < angle_max <= math.pi:
            raise ValueError("angle_min and angle_max must satisfy 0 <= angle_min < angle_max <= pi")
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.p_gaussian_blur = p_gaussian_blur
        self.blur_kernel_sizes = tuple(k for k in blur_kernel_sizes if k % 2 == 1 and k > 1) or (3,)
        self.blur_sigma = blur_sigma
        self.scalar_jitter_std = scalar_jitter_std
        self.scalar_jitter_radial_power = scalar_jitter_radial_power
        self.cutout_outer_bias = max(1.0, cutout_outer_bias)
        c_span_min, c_span_max = cutout_radial_span
        if c_span_max < c_span_min:
            c_span_max = c_span_min
        self.cutout_radial_span = (max(0.0, c_span_min), max(0.0, c_span_max))
        t_span_min, t_span_max = cutout_theta_span
        if t_span_max < t_span_min:
            t_span_max = t_span_min
        self.cutout_theta_span = (max(0.0, t_span_min), max(0.0, t_span_max))
        self.configure_channels(channel_protocol or DEFAULT_CHANNEL_PROTOCOL, v_theta_index)

    def configure_channels(
        self,
        channel_protocol: dict[str, Tuple[int, ...]],
        v_theta_index: Optional[int],
    ) -> None:
        self.channel_protocol = {key: tuple(values) for key, values in channel_protocol.items()}
        if v_theta_index is not None:
            self.v_theta_index = int(v_theta_index)
        else:
            velocity = self.channel_protocol.get("velocity", ())
            self.v_theta_index = (
                velocity[TANGENTIAL_CHANNEL_OFFSET] if len(velocity) > TANGENTIAL_CHANNEL_OFFSET else None
            )

    def _wrap_theta(self, theta: torch.Tensor) -> torch.Tensor:
        theta = torch.remainder(theta, math.tau)
        theta = torch.where(theta > math.pi, math.tau - theta, theta)
        return theta.clamp(min=self.angle_min, max=self.angle_max)

    def _estimate_theta_step(self, theta: torch.Tensor) -> float:
        if theta.shape[-1] < 2:
            return 0.0
        diff = theta[..., 1] - theta[..., 0]
        return float(diff.mean())

    def _maybe_theta_roll(self, x: torch.Tensor, r: torch.Tensor, theta: torch.Tensor):
        if random.random() >= self.p_theta_roll:
            return x, r, theta
        width = theta.shape[-1]
        if width <= 1:
            return x, r, theta
        shift = random.randint(1, width - 1)
        delta = self._estimate_theta_step(theta) * shift
        x = torch.roll(x, shifts=shift, dims=-1)
        r = torch.roll(r, shifts=shift, dims=-1)
        theta = torch.roll(theta, shifts=shift, dims=-1)
        theta = self._wrap_theta(torch.remainder(theta + delta, math.tau))
        return x, r, theta

    def _maybe_theta_mirror(self, x: torch.Tensor, r: torch.Tensor, theta: torch.Tensor):
        if random.random() >= self.p_theta_mirror:
            return x, r, theta
        x = torch.flip(x, dims=(-1,))
        r = torch.flip(r, dims=(-1,))
        theta = torch.flip(theta, dims=(-1,))
        theta = self._wrap_theta(-theta)
        if self.v_theta_index is not None and 0 <= self.v_theta_index < x.shape[0]:
            x[self.v_theta_index] = -x[self.v_theta_index]
        return x, r, theta

    def _maybe_blur(self, x: torch.Tensor):
        if random.random() >= self.p_gaussian_blur:
            return x
        k = random.choice(self.blur_kernel_sizes)
        sigma_low, sigma_high = self.blur_sigma
        sigma = max(1e-3, random.uniform(min(sigma_low, sigma_high), max(sigma_low, sigma_high)))
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, k, k).repeat(x.shape[0], 1, 1, 1)
        x_blur = F.conv2d(x.unsqueeze(0), kernel, padding=k // 2, groups=x.shape[0]).squeeze(0)
        return x_blur

    def _apply_scalar_aug(self, x: torch.Tensor, r_norm: torch.Tensor):
        if self.scalar_jitter_std <= 0:
            return
        scalar_channels = self.channel_protocol.get("scalar", ())
        if not scalar_channels:
            return
        radial_weight = torch.clamp(r_norm, 0.0, 1.0) ** self.scalar_jitter_radial_power
        for idx in scalar_channels:
            if idx >= x.shape[0]:
                continue
            field = x[idx]
            jitter = torch.randn_like(field) * self.scalar_jitter_std * (0.3 + radial_weight)
            perturbed = field + jitter * torch.abs(field)
            x[idx] = torch.clamp(perturbed, min=0.0)

    def _angle_diff(self, theta: torch.Tensor, center: float) -> torch.Tensor:
        return torch.abs(torch.atan2(torch.sin(theta - center), torch.cos(theta - center)))

    def _maybe_cutout(self, x: torch.Tensor, r: torch.Tensor, theta: torch.Tensor):
        if random.random() >= self.p_cutout:
            return x

        r_min = float(r.min())
        r_max = float(r.max())
        if r_max - r_min < 1e-6:
            return x
        theta_min = float(theta.min())
        theta_max = float(theta.max())

        radial_center = r_min + (r_max - r_min) * random.betavariate(self.cutout_outer_bias, 1.0)
        span_min, span_max = self.cutout_radial_span
        radial_span = (r_max - r_min) * random.uniform(span_min, span_max)
        radial_span = max(radial_span, 1e-6)
        r_lower = max(r_min, radial_center - radial_span / 2)
        r_upper = min(r_max, radial_center + radial_span / 2)
        radial_band = (r >= r_lower) & (r <= r_upper)
        inner_guard = r_max * 1e-2
        radial_band = radial_band & (r >= inner_guard)

        theta_center = random.uniform(theta_min, theta_max)
        t_span_min, t_span_max = self.cutout_theta_span
        theta_span = (theta_max - theta_min) * random.uniform(t_span_min, t_span_max)
        theta_span = max(theta_span, 1e-6)
        theta_width = theta_span / 2
        theta_band = self._angle_diff(theta, theta_center) <= theta_width

        mask = radial_band & theta_band
        if not bool(mask.any().item()):
            return x

        ring = radial_band
        for idx in range(x.shape[0]):
            ring_values = x[idx][ring]
            if ring_values.numel() == 0:
                fill_value = x[idx].mean()
            else:
                fill_value = ring_values.mean()
            x[idx][mask] = fill_value
        return x

    def __call__(self, x: torch.Tensor, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.clone()
        coord = coord.clone()
        r = coord[0:1]
        theta = coord[1:2]

        x, r, theta = self._maybe_theta_roll(x, r, theta)
        x, r, theta = self._maybe_theta_mirror(x, r, theta)
        x = self._maybe_blur(x)
        r_map = r.squeeze(0)
        theta_map = theta.squeeze(0)
        r_norm = (r_map - float(r_map.min())) / (float(r_map.max()) - float(r_map.min()) + 1e-6)
        self._apply_scalar_aug(x, r_norm)
        x = self._maybe_cutout(x, r_map, theta_map)

        coord = torch.stack([r_map, theta_map], dim=0)
        return x, coord


# =============================================
# 1) Dataset: load .pt [x, coord, y, mbh, (Optional: y_bondi)]
# =============================================
class GalPTDataset(Dataset):
    def __init__(self, pt_path, *, augmentor: Optional[RandomSpatialAugmentor] = None):
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
                self.x, self.coord, self.y, self.mbh, self.y_bondi, self.groups, self.n_groups = loaded
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
                if isinstance(arr, np.ndarray) and arr.dtype.kind in 'SU':
                    return arr # Keep as numpy string array for later processing

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
        if isinstance(groups_raw, np.ndarray) and groups_raw.dtype.kind in 'SU':
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
        self.channel_protocol = build_channel_protocol(self.x.shape[1])
        velocity_channels = self.channel_protocol.get("velocity", ())
        if len(velocity_channels) <= TANGENTIAL_CHANNEL_OFFSET:
            raise ValueError(
                f"Velocity channel protocol expects index {TANGENTIAL_CHANNEL_OFFSET} to exist, "
                f"but only found channels {velocity_channels}."
            )
        self.v_theta_index = velocity_channels[TANGENTIAL_CHANNEL_OFFSET]
        if self.augmentor is not None:
            self.augmentor.configure_channels(self.channel_protocol, self.v_theta_index)

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
    parser.add_argument(
        "--exp_name", type=str, default="Exp5_", help="Experiment name prefix used for cache and saving"
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
        "--model_type",
        type=str,
        choices=["transformer", "resnet", "resfilm"],
        default="transformer",
        help="Type of model architecture: 'transformer', 'resnet', or 'resfilm'",
    )
    parser.add_argument("--d_model", type=int, default=256, help="Model embedding dimension (d_model)")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward network hidden dimension")
    parser.add_argument("--pos_num_bands", type=int, default=32, help="Number of positional encoding frequency bands")
    parser.add_argument("--pos_max_freq", type=float, default=101, help="Maximum positional encoding frequency")
    parser.add_argument("--p_drop", type=float, default=0.3, help="Dropout probability")
    parser.add_argument(
        "--hpo_params",
        type=str,
        default=None,
        help="Path to Optuna/JSON HPO params file. Matching CLI args will be overridden",
    )
    parser.add_argument(
        "--aug_theta_roll_prob", type=float, default=0.5, help="Probability to apply a cyclic roll along the theta axis"
    )
    parser.add_argument(
        "--aug_theta_mirror_prob",
        type=float,
        default=0.5,
        help="Probability to mirror the sample across the theta axis",
    )
    parser.add_argument(
        "--aug_cutout_prob", type=float, default=0.3, help="Probability to apply radial/theta cutout augmentation"
    )
    parser.add_argument(
        "--aug_cutout_outer_bias", type=float, default=2.5, help="Beta distribution bias for cutout radial center"
    )
    parser.add_argument(
        "--aug_cutout_radial_min", type=float, default=0.05, help="Minimum radial fraction for cutout span"
    )
    parser.add_argument(
        "--aug_cutout_radial_max", type=float, default=0.2, help="Maximum radial fraction for cutout span"
    )
    parser.add_argument(
        "--aug_cutout_theta_min", type=float, default=0.15, help="Minimum angular fraction for cutout span"
    )
    parser.add_argument(
        "--aug_cutout_theta_max", type=float, default=0.5, help="Maximum angular fraction for cutout span"
    )
    parser.add_argument("--aug_blur_prob", type=float, default=0.2, help="Probability to apply Gaussian blur")
    parser.add_argument("--aug_blur_sigma_min", type=float, default=0.6, help="Minimum sigma for Gaussian blur")
    parser.add_argument("--aug_blur_sigma_max", type=float, default=1.6, help="Maximum sigma for Gaussian blur")
    parser.add_argument(
        "--aug_scalar_jitter_std", type=float, default=0.04, help="Std dev for multiplicative scalar field jitter"
    )
    parser.add_argument(
        "--aug_scalar_jitter_radial_power", type=float, default=1.5, help="Radial weighting power for scalar jitter"
    )
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to write TensorBoard logs")

    return parser.parse_args()


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
        "d_model": int,
        "n_layers": int,
        "n_heads": int,
        "d_ff": int,
        "pos_num_bands": int,
        "pos_max_freq": float,
        "p_drop": float,
        "seed": int,
        "aug_theta_roll_prob": float,
        "aug_theta_mirror_prob": float,
        "aug_cutout_prob": float,
        "aug_cutout_outer_bias": float,
        "aug_cutout_radial_min": float,
        "aug_cutout_radial_max": float,
        "aug_cutout_theta_min": float,
        "aug_cutout_theta_max": float,
        "aug_blur_prob": float,
        "aug_blur_sigma_min": float,
        "aug_blur_sigma_max": float,
        "aug_scalar_jitter_std": float,
        "aug_scalar_jitter_radial_power": float,
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

    if getattr(args, "aug_cutout_radial_max", 0.2) < getattr(args, "aug_cutout_radial_min", 0.05):
        args.aug_cutout_radial_max = args.aug_cutout_radial_min
    if getattr(args, "aug_cutout_theta_max", 0.5) < getattr(args, "aug_cutout_theta_min", 0.15):
        args.aug_cutout_theta_max = args.aug_cutout_theta_min
    if getattr(args, "aug_blur_sigma_max", 1.6) < getattr(args, "aug_blur_sigma_min", 0.6):
        args.aug_blur_sigma_max = args.aug_blur_sigma_min

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
        self.weights_fn = weights_fn if weights_fn is not None else (lambda y: torch.ones_like(y))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_criterion(pred, target)
        weights = self.weights_fn(target)

        weighted_loss = loss * weights
        return weighted_loss.mean()


def make_criterion(kind: str, delta: float, alpha: float = 1.0):
    if kind == "mse":
        base = nn.MSELoss(reduction="none")
    elif kind == "l1":
        base = nn.L1Loss(reduction="none")
    elif kind == "huber":
        base = nn.HuberLoss(delta=delta, reduction="none")
    else:
        raise ValueError(kind)

    def weight_linear(target: torch.Tensor):
        return 1.0 + alpha * 10**target

    return WeightedLoss(base_criterion=base, weights_fn=weight_linear)


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
        r = coord[:, 0:1]
        theta = coord[:, 1:2]
        mbh = batch[2].to(device, non_blocking=True).view(-1)  # (B,)
        y = batch[3].to(device, non_blocking=True).view(-1)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                if isinstance(model, (MACNetRes_mbh, MACNetFiLM)):
                    pred = model(x, mbh)
                else:
                    pred = model(x, r, theta, mbh)  # (B,)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        else:
            if isinstance(model, (MACNetRes_mbh, MACNetFiLM)):
                pred = model(x, mbh)
            else:
                pred = model(x, r, theta, mbh)
            loss = criterion(pred, y)
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
            if isinstance(model, (MACNetRes_mbh, MACNetFiLM)):
                pred = model(x, mbh)
            else:
                pred = model(x, coord[:, 0:1], coord[:, 1:2], mbh)
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
        r = coord[:, 0:1]
        theta = coord[:, 1:2]
        mbh = batch[2].to(device, non_blocking=True).view(-1)
        y = batch[3].to(device, non_blocking=True).view(-1)
        groups = batch[4].view(-1).to(device=mbh.device, dtype=torch.long)

        if isinstance(model, (MACNetRes_mbh, MACNetFiLM)):
            pred = model(x, mbh)
        else:
            pred = model(x, r, theta, mbh)

        if isinstance(criterion, WeightedLoss):
            base_loss = criterion.base_criterion(pred, y)  # shape: (B, ...) or (B,)
            weights = criterion.weights_fn(y)  # 同形或可广播
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
def main():
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
    radial_span_min = max(0.0, min(1.0, args.aug_cutout_radial_min))
    radial_span_max = max(radial_span_min, min(1.0, args.aug_cutout_radial_max))
    theta_span_min = max(0.0, min(1.0, args.aug_cutout_theta_min))
    theta_span_max = max(theta_span_min, min(1.0, args.aug_cutout_theta_max))
    blur_sig_min = max(1e-3, args.aug_blur_sigma_min)
    blur_sig_max = max(blur_sig_min, args.aug_blur_sigma_max)
    train_augmentor = RandomSpatialAugmentor(
        p_theta_roll=args.aug_theta_roll_prob,
        p_theta_mirror=args.aug_theta_mirror_prob,
        p_cutout=args.aug_cutout_prob,
        erase_scale=(radial_span_min, radial_span_max),
        angle_min=0.05,
        angle_max=3.09,
        p_gaussian_blur=max(0.0, min(1.0, args.aug_blur_prob)),
        blur_sigma=(blur_sig_min, blur_sig_max),
        scalar_jitter_std=max(0.0, args.aug_scalar_jitter_std),
        scalar_jitter_radial_power=max(0.0, args.aug_scalar_jitter_radial_power),
        cutout_outer_bias=max(1.0, args.aug_cutout_outer_bias),
        cutout_radial_span=(radial_span_min, radial_span_max),
        cutout_theta_span=(theta_span_min, theta_span_max),
    )
    train_ds = GalPTDataset(os.path.join(cache_dir, exp + "train.pt"), augmentor=train_augmentor)
    val_ds = GalPTDataset(os.path.join(cache_dir, exp + "val.pt"))
    test_ds = GalPTDataset(os.path.join(cache_dir, exp + "test.pt"))

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
        ValueError("Hybrid model type is not implemented in this script.")
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

    try:
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            # Use a tiny temp loader with 0 workers to fetch a sample for the graph
            temp_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
            sample_x, sample_coord, sample_mbh, _, _, _ = next(iter(temp_loader))
            if isinstance(model, (MACNetRes_mbh, MACNetFiLM)):
                writer.add_graph(
                    model,
                    (
                        sample_x.to(device),
                        sample_mbh.view(-1).to(device),
                    ),
                )
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
    criterion = make_criterion(args.loss, args.huber_delta)
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

    # Train
    best_val = float("inf")
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
            torch.save(
                {"model": model.state_dict(), "val_loss": best_val, "mean_val": mean_val, "epoch": epoch}, best_path
            )
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_val == float("inf") or not os.path.exists(best_path):
        print("No improvement was found during training. Best model not saved.")
        return

    print(f"Best val loss: {best_val:.6f}. Loading best and testing.")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_loss, group_means = eval_epoch(model, test_loader, criterion, device, writer, epoch=0)
    print(f"Test loss: {test_loss:.6f}, per-domain: {group_means}")
    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()

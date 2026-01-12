import os
import math
import copy
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Optional


# =============================================
# Training Function
# =============================================
def training(
    train_loader=None,
    val_loader=None,
    model=None,
    criterion=None,
    optimizer=None,
    device=None,
    num_epochs=25,
    save_path=None,
    scheduler=None,
    *,
    test_loader=None,
    grad_clip=None,
    use_amp=True,
    early_stopping_patience=None,
    log_dir=None,
    save_every=None,
):
    """
    Train the model and evaluate on validation and test sets.
    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The neural network model to be trained.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the training on (CPU or GPU).
        num_epochs (int): Number of epochs to train the model.
        save_path (str): Path to save the best model weights.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_loss = float("inf")
    best_model_wts = None
    epochs_no_improve = 0

    # Initialize TensorBoard writer
    if log_dir is None:
        log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Optional graph (safe-guarded)
    try:
        inputs, mbh_ex, _ = next(iter(train_loader))
        writer.add_graph(model, (inputs.to(device), mbh_ex.to(device)))
    except Exception:
        pass

    # Use the new torch.amp API (avoids FutureWarning)
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, mbh, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            mbh = mbh.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                outputs = model(inputs, mbh)
                loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                print("Non-finite loss encountered; skipping batch.")
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, mbh, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                mbh = mbh.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                    outputs = model(inputs, mbh)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        writer.add_scalar("Loss/val", val_epoch_loss, epoch)

        if scheduler:
            scheduler.step(val_epoch_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LearningRate", current_lr, epoch)

        # Deep copy the model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_wts, save_path)
                print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if save_every and isinstance(save_every, int) and save_every > 0 and ((epoch + 1) % save_every == 0):
            ckpt_path = (save_path or os.path.join("Results", "checkpoint.pth")).replace(
                "best_model", f"epoch{epoch+1}"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if early_stopping_patience is not None and early_stopping_patience > 0:
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Restore best weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Test evaluation
    history = {"best_val_loss": best_val_loss}
    if test_loader is not None:
        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for inputs, mbh, labels in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                mbh = mbh.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=(use_amp and device.type == "cuda")):
                    outputs = model(inputs, mbh)
                    loss = criterion(outputs, labels)
                test_running_loss += loss.item() * inputs.size(0)
        test_loss = test_running_loss / len(test_loader.dataset)
        writer.add_scalar("Loss/test", test_loss, epoch + 1)
        history["test_loss"] = test_loss

    writer.close()
    return history


# =============================================
# Module Definition
# =============================================
class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nn.functional.relu(Y + X)


def resnet_block(num_residuals, in_channels, out_channels, half=False):
    layers = []
    for i in range(num_residuals):
        if i == 0:
            if half:
                layers.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                layers.append(Residual(in_channels, out_channels, use_1x1conv=True))
        else:
            layers.append(Residual(out_channels, out_channels))
    return nn.Sequential(*layers)


class MACNetRes_mbh(nn.Module):

    def __init__(self, in_channels=15):
        super().__init__()
        in_channels = in_channels
        # Define the Residual blocks
        blocks = []
        res_arch = ((2, 32, True), (2, 64, True), (2, 128, False), (2, 256, False))
        for num_residuals, out_channels, half in res_arch:
            # print(in_channels, out_channels, num_residuals, first_block)
            blocks.append(
                resnet_block(
                    num_residuals=num_residuals,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    half=half,
                )
            )
            in_channels = out_channels
        self.res_blocks = nn.Sequential(*blocks)

        # Define the Linear blocks
        self.flatten_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.liner_block = nn.Sequential(
            nn.Linear(in_channels + 1, in_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, x, mbh):
        mbh = torch.tensor(mbh, dtype=torch.float32, device=x.device) if not isinstance(mbh, torch.Tensor) else mbh
        mbh = mbh.float().to(x.device).view(-1, 1)

        x = self.res_blocks(x)
        x = self.flatten_block(x)
        x = torch.cat((x, mbh), dim=1)
        x = self.liner_block(x)
        return x


# Transformer-based model
# Utils
# def _minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     # Per-sample, per-channel min-max normalization
#     B, C, H, W = x.shape
#     x_ = x.view(B, C, -1)
#     xmin = x_.min(dim=-1, keepdim=True).values
#     xmax = x_.max(dim=-1, keepdim=True).values
#     x_ = (x_ - xmin) / (xmax - xmin + eps)
#     return x_.view(B, C, H, W)


# Continuous Fourier feature for (r, theta) coordinates
class FourierPositionalEncoding2D(nn.Module):
    """
    Encode continuous (r, theta) with fixed (non-trainable) sinusoidal features.
    Works across arbitrary grids/resolutions.
    """

    def __init__(
        self,
        num_bands: int = 32,
        max_frequency: float = 64.0,
        include_input: bool = True,
        theta_pi_periodic: bool = True,
        r_log_scale: bool = True,
        eps: float = 1e-6,
    ):
        """
        Args:
            num_bands: number of frequency bands per coordinate
            max_frequency: highest frequency (log-spaced up to this)
            include_input: whether to include raw [r, theta] (normalized) in output
        """
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input
        # log-spaced frequencies: [1, ..., max_frequency]
        self.theta_pi_periodic = theta_pi_periodic
        self.r_log_scale = r_log_scale
        self.eps = eps
        self.register_buffer(
            "freq_bands",
            torch.logspace(0.0, math.log10(max_frequency), steps=num_bands),
        )

    @staticmethod
    def _wrap_pi(theta: torch.Tensor) -> torch.Tensor:
        # map to [0, π)
        return torch.remainder(theta, math.pi)

    @property
    def out_dim(self) -> int:
        base = 2 if self.include_input else 0  # r, theta
        # for each of r,theta -> sin&cos per band -> 2 * num_bands
        return base + 2 * 2 * self.num_bands

    def forward(self, r: torch.Tensor, theta: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        r, theta: (B, 1, H, W)
        Returns: (B, Dpos, H, W)
        """
        B, _, H, W = r.shape

        # r normalization
        if self.r_log_scale:
            r_safe = torch.clamp(r, min=self.eps)
            r_n = (torch.log(r_safe) - torch.log(r_safe).amin((2, 3), True)) / (
                torch.log(r_safe).amax((2, 3), True) - torch.log(r_safe).amin((2, 3), True) + self.eps
            )
        else:
            if normalize:
                r_n = (r - r.amin((2, 3), True)) / (r.amax((2, 3), True) - r.amin((2, 3), True) + self.eps)
            else:
                r_n = (r - r.min()) / (r.max() - r.min() + self.eps)

        # theta handling
        if self.theta_pi_periodic:
            theta = self._wrap_pi(theta)  # [0, π)
            theta_unit = theta / math.pi  # [0,1) for optional raw export
            # even harmonics: 2k * theta
            # we reuse log-spaced freq_bands on the base, then multiply by 2
            fb_theta = (2.0 * self.freq_bands).view(1, -1, 1).to(theta.device)
        else:
            # fallback to 2π periodic if ever needed
            theta = torch.remainder(theta, 2 * math.pi)
            theta_unit = theta / (2 * math.pi)
            fb_theta = self.freq_bands.view(1, -1, 1).to(theta.device)

        # if normalize:
        #     r = _minmax_norm(r)  # [0,1] per-sample
        #     # normalize theta to [0,1] by mapping [-pi, pi] or [0, 2pi] if needed
        #     # 如果你的 theta 已经是 [0, 2π) 或 [-π, π]，这里做一个鲁棒归一化：
        #     th_min = theta.amin(dim=(2, 3), keepdim=True)
        #     th_max = theta.amax(dim=(2, 3), keepdim=True)
        #     theta = (theta - th_min) / (th_max - th_min + 1e-6)

        r_flat = r_n.view(B, 1, H * W)
        t_flat = theta.view(B, 1, H * W)

        # (1, num_bands, 1)
        fb_r = self.freq_bands.view(1, -1, 1).to(r.device)

        # project
        r_proj = r_flat * fb_r  # (B, num_bands, HW)
        t_proj = t_flat * fb_theta

        # sin/cos
        r_sin = torch.sin(2 * math.pi * r_proj)
        r_cos = torch.cos(2 * math.pi * r_proj)
        t_sin = torch.sin(t_proj)
        t_cos = torch.cos(t_proj)

        pos_list = []
        if self.include_input:
            pos_list += [r_flat, t_flat]  # (B,1,HW) each
        pos_list += [r_sin, r_cos, t_sin, t_cos]  # (B,num_bands,HW) each

        pos = torch.cat(pos_list, dim=1).view(B, -1, H, W)  # (B, Dpos, HW)
        return pos


# FiLM conditioner for BH mass
class FiLMConditioner(nn.Module):
    """
    Map scalar BH mass to (gamma, beta) for feature modulation.
    Can produce one pair per Transformer layer.
    """

    def __init__(self, n_layers: int, d_model: int, hidden: int = 128):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * n_layers * d_model),
        )

    def forward(self, bh_mass: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        bh_mass: (B,) or (B,1)
        Returns lists gamma_list, beta_list each len=n_layers,
        with tensors of shape (B, 1, d_model) ready to broadcast over tokens.
        """
        if bh_mass.dim() == 1:
            bh_mass = bh_mass.unsqueeze(-1)
        B = bh_mass.shape[0]
        out = self.net(bh_mass)  # (B, 2*n_layers*d_model)
        out = out.view(B, self.n_layers, 2, self.d_model)  # (B,L,2,D)
        gammas = out[:, :, 0, :].unsqueeze(1)  # (B,1,L,D)
        betas = out[:, :, 1, :].unsqueeze(1)  # (B,1,L,D)
        # split by layer
        gamma_list = [gammas[:, :, i, :] for i in range(self.n_layers)]  # each (B,1,D)
        beta_list = [betas[:, :, i, :] for i in range(self.n_layers)]
        return gamma_list, beta_list


# Transformer blocks (Pre-Norm)
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, p_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.drop1 = nn.Dropout(p_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, p_drop)

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ):
        # x: (B, N, D)
        # FiLM pre-attn
        if gamma is not None and beta is not None:
            x = (1 + gamma) * x + beta  # broadcast over tokens

        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)

        # FiLM pre-MLP
        if gamma is not None and beta is not None:
            x = gamma * x + beta

        # MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


# Main model
class AccretionTransformer(nn.Module):
    """
    Input:
        x: (B, C_in, H, W)   -- multi-channel galaxy "image"
        bh_mass: (B,)        -- scalar BH mass
    Output:
        y: (B,)              -- scalar (e.g., log accretion rate)
    """

    def __init__(
        self,
        c_in: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        pos_num_bands: int = 32,
        pos_max_freq: float = 64.0,
        # include_rtheta_in_content: bool = True,
        p_drop: float = 0.1,
    ):
        super().__init__()
        # assert 0 <= r_idx < c_in and 0 <= theta_idx < c_in and r_idx != theta_idx

        # self.r_idx = r_idx
        # self.theta_idx = theta_idx
        # self.include_rtheta_in_content = include_rtheta_in_content

        # Positional encoding from (r, theta)
        self.posenc = FourierPositionalEncoding2D(
            num_bands=pos_num_bands, max_frequency=pos_max_freq, include_input=True, theta_pi_periodic=False
        )
        d_pos = self.posenc.out_dim

        # # Content channels: with or without r/theta
        # if include_rtheta_in_content:
        #     c_content = c_in
        # else:
        #     c_content = c_in - 2

        self.content_proj = nn.Conv2d(c_in, d_model // 2, kernel_size=1)  # per-pixel linear
        self.pos_proj = nn.Conv2d(d_pos, d_model // 2, kernel_size=1)
        self.drop = nn.Dropout(p_drop)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, p_drop=p_drop) for _ in range(n_layers)]
        )

        # FiLM conditioner (per-layer γ/β)
        self.film = FiLMConditioner(n_layers=n_layers, d_model=d_model, hidden=128)

        # Head for regression
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    # def _split_content(self, x: torch.Tensor) -> torch.Tensor:
    #     # Remove r/theta if exclude; otherwise return x unchanged
    #     if self.include_rtheta_in_content:
    #         return x
    #     idxs = [i for i in range(x.shape[1]) if i not in (self.r_idx, self.theta_idx)]
    #     return x[:, idxs, :, :]

    def forward(
        self,
        x: torch.Tensor,  # (B, C_in, H, W)
        r: torch.Tensor,  # (B, 1,    H, W)
        theta: torch.Tensor,  # (B, 1,    H, W)
        bh_mass: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        x: (B, C_in, H, W), bh_mass: (B,)
        """
        B, C, H, W = x.shape

        # r = x[:, self.r_idx : self.r_idx + 1, :, :]
        # theta = x[:, self.theta_idx : self.theta_idx + 1, :, :]

        # Positional features from (r, theta)
        pos = self.posenc(r, theta, normalize=True)  # (B, Dpos, H, W)

        # Content features
        # content = self._split_content(x)  # (B, Cc, H, W)
        content = x  # (B, C_in, H, W)

        # Project and fuse (sum is cleaner than concat here; both are fine)
        feat = self.content_proj(content) + self.pos_proj(pos)  # (B, D/2, H, W)
        feat = torch.cat([feat, self.drop(feat)], dim=1)  # widen back to D (simple gating)

        # Now (B, D, H, W)
        D = feat.shape[1]
        if not torch.jit.is_tracing():
            assert int(D) == self.blocks[0].attn.embed_dim, "proj dims must match d_model"

        # Flatten to tokens
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N=H*W, D)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)

        # FiLM params from BH mass
        gammas, betas = self.film(bh_mass)  # lists of (B,1,D)

        # Pass through layers
        x_tok = tokens
        for i, blk in enumerate(self.blocks):
            gamma = gammas[i].expand(B, x_tok.size(1), -1)  # (B,N+1,D)
            beta = betas[i].expand(B, x_tok.size(1), -1)
            x_tok = blk(x_tok, gamma=gamma, beta=beta)

        # Readout: CLS
        cls_out = x_tok[:, 0, :]  # (B, D)
        y = self.head(cls_out).squeeze(-1)  # (B,)
        return y


class HybridAccretionTransformer(nn.Module):
    """
    Hybrid CNN-Transformer model for accretion rate prediction.
    Uses a CNN encoder (Stem) to extract and compress spatial features before
    passing them as tokens to the Transformer.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        pos_num_bands: int = 32,
        pos_max_freq: float = 64.0,
        p_drop: float = 0.1,
        # CNN Stem configuration: (num_residuals, out_channels, downsample)
        # Default: two stages of downsampling (e.g. 64x64 -> 16x16)
        cnn_arch: List[Tuple[int, int, bool]] = [(2, 32, True), (2, 64, True)],
    ):
        super().__init__()

        # -- CNN Stem --
        blocks = []
        curr_in = c_in
        for num_residuals, out_channels, half in cnn_arch:
            blocks.append(
                resnet_block(
                    num_residuals=num_residuals,
                    in_channels=curr_in,
                    out_channels=out_channels,
                    half=half,
                )
            )
            curr_in = out_channels
        self.stem = nn.Sequential(*blocks)

        # -- Positional Encoding --
        self.posenc = FourierPositionalEncoding2D(
            num_bands=pos_num_bands,
            max_frequency=pos_max_freq,
            include_input=True,
            theta_pi_periodic=False,
        )
        d_pos = self.posenc.out_dim

        # -- Projections --
        # Project CNN output and position encoding to d_model // 2
        # (matching the original AccretionTransformer's gating logic)
        self.content_proj = nn.Conv2d(curr_in, d_model // 2, kernel_size=1)
        self.pos_proj = nn.Conv2d(d_pos, d_model // 2, kernel_size=1)
        self.drop = nn.Dropout(p_drop)

        # -- CLS token --
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # -- Transformer blocks --
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, p_drop=p_drop) for _ in range(n_layers)]
        )

        # -- FiLM --
        self.film = FiLMConditioner(n_layers=n_layers, d_model=d_model, hidden=128)

        # -- Head --
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, C_in, H, W)
        r: torch.Tensor,  # (B, 1, H, W)
        theta: torch.Tensor,  # (B, 1, H, W)
        bh_mass: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Extract features with CNN stem
        # This provides translation invariance and local feature extraction
        content_feat = self.stem(x)  # (B, C_stem, H', W')
        H_p, W_p = content_feat.shape[2:]

        # 2. Downsample coordinates to match feature map resolution
        r_p = nn.functional.interpolate(r, size=(H_p, W_p), mode="bilinear", align_corners=False)
        theta_p = nn.functional.interpolate(theta, size=(H_p, W_p), mode="bilinear", align_corners=False)

        # 3. Positional encoding on the compressed grid
        pos_feat = self.posenc(r_p, theta_p, normalize=True)  # (B, Dpos, H', W')

        # 4. Fuse content and position
        feat = self.content_proj(content_feat) + self.pos_proj(pos_feat)  # (B, D/2, H', W')
        feat = torch.cat([feat, self.drop(feat)], dim=1)  # (B, D, H', W')

        # 5. Flatten to tokens
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N=H'*W', D)

        # 6. Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # 7. FiLM parameters for BH mass conditioning
        gammas, betas = self.film(bh_mass)

        # 8. Transformer layers
        x_tok = tokens
        for i, blk in enumerate(self.blocks):
            gamma = gammas[i].expand(B, x_tok.size(1), -1)
            beta = betas[i].expand(B, x_tok.size(1), -1)
            x_tok = blk(x_tok, gamma=gamma, beta=beta)

        # 9. Readout: CLS
        cls_out = x_tok[:, 0, :]
        y = self.head(cls_out).squeeze(-1)
        return y

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


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
        return x.view(-1)


class FiLMResidual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X, gamma, beta):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        # FiLM modulation
        if gamma is not None and beta is not None:
            # gamma, beta: [B, C] -> [B, C, 1, 1]
            Y = (1 + gamma.view(-1, self.out_channels, 1, 1)) * Y + beta.view(-1, self.out_channels, 1, 1)

        if self.conv3:
            X = self.conv3(X)
        return nn.functional.relu(Y + X)


class MACNetFiLM(nn.Module):
    def __init__(self, in_channels=15):
        super().__init__()
        in_c = in_channels
        self.blocks = nn.ModuleList()
        self.channel_counts = []

        # Consistent with MACNetRes_mbh
        res_arch = ((2, 32, True), (2, 64, True), (2, 128, False), (2, 256, False))

        for num_residuals, out_channels, half in res_arch:
            for i in range(num_residuals):
                stride = 1
                use_1x1 = False
                curr_in = out_channels

                if i == 0:
                    stride = 2 if half else 1
                    use_1x1 = True
                    curr_in = in_c

                self.blocks.append(FiLMResidual(curr_in, out_channels, use_1x1conv=use_1x1, strides=stride))
                self.channel_counts.append(out_channels)

            in_c = out_channels

        self.total_features = sum(self.channel_counts)
        # FiLM Generator
        self.film_generator = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(), nn.Linear(64, self.total_features * 2))

        self.flatten_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        # Final regression from last block features
        last_out_channels = res_arch[-1][1]
        self.liner_block = nn.Sequential(
            nn.Linear(last_out_channels, last_out_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(last_out_channels // 2, 1),
        )

    def forward(self, x, mbh):
        mbh = torch.tensor(mbh, dtype=torch.float32, device=x.device) if not isinstance(mbh, torch.Tensor) else mbh
        mbh = mbh.float().to(x.device).view(-1, 1)

        film_params = self.film_generator(mbh)
        gammas_betas = torch.split(film_params, [c * 2 for c in self.channel_counts], dim=1)

        for block, params in zip(self.blocks, gammas_betas):
            gamma, beta = torch.chunk(params, 2, dim=1)
            x = block(x, gamma, beta)

        x = self.flatten_block(x)
        x = self.liner_block(x)
        return x.view(-1)


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
            # theta_unit = theta / math.pi  # [0,1) for optional raw export
            # even harmonics: 2k * theta
            # we reuse log-spaced freq_bands on the base, then multiply by 2
            fb_theta = (2.0 * self.freq_bands).view(1, -1, 1).to(theta.device)
        else:
            # fallback to 2π periodic if ever needed
            theta = torch.remainder(theta, 2 * math.pi)
            # theta_unit = theta / (2 * math.pi)
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
        t_sin = torch.sin(2 * t_proj)
        t_cos = torch.cos(2 * t_proj)

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
            x = (1 + gamma) * x + beta

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
        r: (B, 1, H, W)      -- radial coordinate
        theta: (B, 1, H, W)  -- azimuthal coordinate
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
        p_drop: float = 0.1,
    ):
        super().__init__()

        # Positional encoding from (r, theta)
        self.posenc = FourierPositionalEncoding2D(
            num_bands=pos_num_bands,
            max_frequency=pos_max_freq,
            include_input=True,
            theta_pi_periodic=False,
        )
        d_pos = self.posenc.out_dim

        # Projections to d_model
        # Use sum fusion: content_proj(x) + pos_proj(pos)
        self.content_proj = nn.Conv2d(c_in, d_model, kernel_size=1)
        self.pos_proj = nn.Conv2d(d_pos, d_model, kernel_size=1)
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

    def forward(
        self,
        x: torch.Tensor,  # (B, C_in, H, W)
        r: torch.Tensor,  # (B, 1,    H, W)
        theta: torch.Tensor,  # (B, 1,    H, W)
        bh_mass: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        x: (B, C_in, H, W), r, theta: (B, 1, H, W), bh_mass: (B,)
        """
        B, C, H, W = x.shape

        # Positional features from (r, theta)
        pos = self.posenc(r, theta, normalize=True)  # (B, Dpos, H, W)

        # Project and fuse (sum)
        # content: (B, d_model, H, W)
        # pos:     (B, d_model, H, W)
        feat = self.content_proj(x) + self.pos_proj(pos)
        feat = self.drop(feat)

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

# =============================================
# CNN-based model with Fourier PE + FiLM (append to models.py)
# =============================================


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: lightweight channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class ConvFiLMBlock(nn.Module):
    """
    Pre-activation ResNet block with:
      - BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3
      - FiLM modulation (γ, β) applied after the first BN
      - SE channel attention at the end
      - Optional 1×1 skip projection + stride-2 downsampling
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        p_drop: float = 0.1,
        se_reduction: int = 4,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.drop = nn.Dropout2d(p_drop)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        identity = self.skip(x)

        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        # FiLM modulation: scale + shift after normalization
        if gamma is not None and beta is not None:
            out = (1 + gamma) * out + beta
        out = torch.relu(out)
        out = self.drop(out)
        out = self.conv2(out)

        out = self.se(out)
        return out + identity


class ConvFiLMConditioner(nn.Module):
    """
    Map scalar BH mass → per-stage (γ, β) for 2D feature maps.
    Each γ/β has shape (B, C_stage, 1, 1) to broadcast over spatial dims.
    """

    def __init__(self, stage_channels: List[int], hidden: int = 128):
        super().__init__()
        self.stage_channels = stage_channels
        total_out = 2 * sum(stage_channels)  # γ + β for each stage
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, total_out),
        )

    def forward(
        self, bh_mass: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if bh_mass.dim() == 1:
            bh_mass = bh_mass.unsqueeze(-1)
        B = bh_mass.shape[0]
        out = self.net(bh_mass)  # (B, total_out)

        gamma_list, beta_list = [], []
        offset = 0
        for ch in self.stage_channels:
            gamma_list.append(out[:, offset : offset + ch].view(B, ch, 1, 1))
            offset += ch
            beta_list.append(out[:, offset : offset + ch].view(B, ch, 1, 1))
            offset += ch
        return gamma_list, beta_list


class AccretionConvNet(nn.Module):
    """
    CNN backbone with Fourier positional encoding + FiLM conditioning.

    Architecture:
      1. Fourier PE of (r, θ) → fused with content channels via concatenation
      2. Stem conv to lift to base_ch
      3. 4 residual stages with FiLM conditioning from BH mass
         Each stage: ConvFiLMBlock × depth, stride-2 downsampling at stage boundaries
      4. Global average pooling → regression head → scalar output

    Input signature matches AccretionTransformer exactly:
        x:       (B, C_in, H, W)
        r:       (B, 1,    H, W)
        theta:   (B, 1,    H, W)
        bh_mass: (B,)

    Output:
        y: (B,)
    """

    def __init__(
        self,
        c_in: int,
        base_ch: int = 64,
        stage_depths: Tuple[int, ...] = (2, 2, 2, 2),
        pos_num_bands: int = 32,
        pos_max_freq: float = 64.0,
        p_drop: float = 0.1,
        se_reduction: int = 4,
    ):
        """
        Args:
            c_in:          Number of input physical-quantity channels.
            base_ch:       Base channel count; doubles each stage.
            stage_depths:  Number of residual blocks per stage.
            pos_num_bands: Fourier encoding frequency bands.
            pos_max_freq:  Max frequency for Fourier encoding.
            p_drop:        Dropout probability.
            se_reduction:  SE block channel reduction ratio.
        """
        super().__init__()

        # ---- Positional encoding (reuse existing module) ----
        self.posenc = FourierPositionalEncoding2D(
            num_bands=pos_num_bands,
            max_frequency=pos_max_freq,
            include_input=True,
            theta_pi_periodic=False,
        )
        d_pos = self.posenc.out_dim

        # ---- Stem: fuse content + positional features ----
        stem_in = c_in + d_pos
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # ---- Residual stages ----
        stage_channels = []
        stages = []
        in_ch = base_ch
        for i, depth in enumerate(stage_depths):
            out_ch = base_ch * (2 ** i)
            stage_channels.append(out_ch)
            blocks = []
            for j in range(depth):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(
                    ConvFiLMBlock(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        stride=stride,
                        p_drop=p_drop,
                        se_reduction=se_reduction,
                    )
                )
            stages.append(nn.ModuleList(blocks))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)
        self.stage_channels = stage_channels
        last_ch = stage_channels[-1]

        # ---- FiLM conditioner ----
        self.film = ConvFiLMConditioner(stage_channels=stage_channels, hidden=128)

        # ---- Regression head ----
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(last_ch),
            nn.Linear(last_ch, last_ch // 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(last_ch // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, C_in, H, W)
        r: torch.Tensor,        # (B, 1,    H, W)
        theta: torch.Tensor,    # (B, 1,    H, W)
        bh_mass: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        # Positional features
        pos = self.posenc(r, theta, normalize=True)  # (B, D_pos, H, W)

        # Fuse content + positional
        feat = torch.cat([x, pos], dim=1)  # (B, C_in + D_pos, H, W)
        feat = self.stem(feat)

        # FiLM conditioning from BH mass
        gammas, betas = self.film(bh_mass)

        # Residual stages with FiLM
        for stage_idx, stage in enumerate(self.stages):
            g = gammas[stage_idx]   # (B, C_stage, 1, 1)
            b = betas[stage_idx]
            for block in stage:
                feat = block(feat, gamma=g, beta=b)

        # Regression
        y = self.head(feat).squeeze(-1)  # (B,)
        return y
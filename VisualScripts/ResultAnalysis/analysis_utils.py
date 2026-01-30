# analysis_utils.py
import os
import sys
import gc
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# ==========================================
# 1. Project Setup & Imports
# ==========================================
def setup_project_env():
    """Found project root and set up environment."""
    current_path = Path.cwd()
    project_root = None
    markers = ['Experiment', 'RawDataProcessing', '.config']

    # Try to find root from CWD (Notebook execution context)
    cand_path = current_path
    while cand_path != cand_path.parent:
        if any((cand_path / m).exists() for m in markers):
            project_root = cand_path
            break
        cand_path = cand_path.parent
    
    # Fallback: relative to this file if it's imported
    if project_root is None:
        try:
            # If __file__ is available
            project_root = Path(__file__).resolve().parent.parent.parent
        except NameError:
            # Fallback for some interactive environments
            project_root = Path(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
            
    # Add to sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    return project_root

# Initialize root immediately for imports in this file
PROJECT_ROOT = setup_project_env()

# Project Imports
try:
    from RawDataProcessing.GalaxyData import GalDataSet
    from Experiment.Network import MACNetRes_mbh, AccretionTransformer, MACNetFiLM, FourierPositionalEncoding2D, FiLMConditioner, MLP
    # Import Preprocess paths if available
    try:
        from Experiment.Preprocess import Exp_eg_folder_paths, Exp_dg_folder_paths
    except ImportError:
        Exp_eg_folder_paths = []
        Exp_dg_folder_paths = []
except ImportError as e:
    print(f"[WARN] Project imports failed in analysis_utils: {e}")

# ==========================================
# 2. Legacy Classes (from Exp2)
# ==========================================
class LegacyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, p_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.drop1 = nn.Dropout(p_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, p_drop)

    def forward(self, x, gamma=None, beta=None):
        if gamma is not None and beta is not None:
            x = gamma * x + beta 
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        if gamma is not None and beta is not None:
            x = gamma * x + beta
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

class LegacyAccretionTransformer(nn.Module):
    def __init__(self, c_in, d_model=256, n_layers=8, n_heads=8, d_ff=1024, pos_num_bands=32, pos_max_freq=64.0, p_drop=0.1):
        super().__init__()
        self.posenc = FourierPositionalEncoding2D(num_bands=pos_num_bands, max_frequency=pos_max_freq, include_input=True)
        d_pos = self.posenc.out_dim
        self.content_proj = nn.Conv2d(c_in, d_model // 2, kernel_size=1)
        self.pos_proj = nn.Conv2d(d_pos, d_model // 2, kernel_size=1)
        self.drop = nn.Dropout(p_drop)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([
            LegacyTransformerBlock(d_model, n_heads, d_ff, p_drop) for _ in range(n_layers)
        ])
        self.film = FiLMConditioner(n_layers, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, r, theta, bh_mass):
        B, C, H, W = x.shape
        pos = self.posenc(r, theta, normalize=True)
        feat = self.content_proj(x) + self.pos_proj(pos)
        feat = torch.cat([feat, self.drop(feat)], dim=1) 
        tokens = feat.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        gammas, betas = self.film(bh_mass)
        x_tok = tokens
        for i, blk in enumerate(self.blocks):
            gamma = gammas[i].expand(B, x_tok.size(1), -1)
            beta = betas[i].expand(B, x_tok.size(1), -1)
            x_tok = blk(x_tok, gamma=gamma, beta=beta)
        return self.head(x_tok[:, 0, :]).squeeze(-1)

# ==========================================
# 3. Helper Functions
# ==========================================
def ensure_tensor(arr, dtype=torch.float32):
    if arr is None: return None
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=dtype).contiguous()
    return torch.as_tensor(arr, dtype=dtype).contiguous()

def find_latest_model(results_root: Path, exp_prefix: str) -> Path:
    if not results_root.exists(): return None
    candidates = list(results_root.glob(f'{exp_prefix}*/*.pth'))
    if not candidates:
        candidates = list(results_root.glob('**/*.pth'))
    if not candidates: return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

# ==========================================
# 4. Datasets
# ==========================================
class CachedPTDataset:
    def __init__(self, pt_path: Path):
        self.path = pt_path
        if not pt_path.exists():
            raise FileNotFoundError(f"Cached dataset not found: {pt_path}")
            
        loaded = torch.load(pt_path, map_location='cpu', weights_only=False)
        items = list(loaded)
        
        self.coord = None
        self.y_bondi = None
        self.groups = None
        self.type = 'cached'

        if len(items) >= 5:
            if isinstance(items[1], (torch.Tensor, np.ndarray)) and items[1].ndim >= 2:
                self.x, self.coord, self.y, self.mbh, self.y_bondi, self.groups = items[:6]
                self.kind = 'vit'
            else:
                self.x, self.y, self.mbh, self.y_bondi, self.groups = items[:5]
                self.kind = 'macnet'
        else:
            self.x, self.y, self.mbh = items[:3]
            self.kind = 'macnet'

        self.x = ensure_tensor(self.x)
        self.coord = ensure_tensor(self.coord)
        self.y = ensure_tensor(self.y).view(-1, 1)
        self.mbh = ensure_tensor(self.mbh).view(-1, 1)
        self.y_bondi = ensure_tensor(self.y_bondi).view(-1, 1) if self.y_bondi is not None else None
        
        if self.groups is not None:
             if isinstance(self.groups, np.ndarray) and self.groups.dtype.kind in ('S', 'U'):
                _, indices = np.unique(self.groups, return_inverse=True)
                self.groups = torch.as_tensor(indices, dtype=torch.long)
             else:
                try:
                    self.groups = torch.as_tensor(self.groups)
                except: pass

    def __len__(self):
        return len(self.y)

class HDF5Dataset:
    def __init__(self, hdf5_conf, resolution='coarse'):
        if isinstance(hdf5_conf, (list, tuple)):
            self.folder_path = hdf5_conf[0]
        else:
            self.folder_path = hdf5_conf
            
        self.type = os.path.basename(os.path.dirname(self.folder_path)).lower()
        
        loaded = GalDataSet().load_data((self.folder_path, 0)) 
        loaded.drop_invalid().filter(-5)
        
        self.x, _, _ = loaded.standardize()
        self.coord = loaded.coordinate
        self.y = loaded.y
        self.mbh = loaded.mbh
        self.y_bondi = loaded.y_baseline if hasattr(loaded, 'y_baseline') else None
        self.time = loaded.time
        self.mdot_edd = np.array(loaded._raw_data['mdot_edd']) if 'mdot_edd' in loaded._raw_data else None
        
        self.kind = 'vit' if self.coord is not None else 'macnet'

    def __len__(self):
        return len(self.y)

# ==========================================
# 5. Model Loading
# ==========================================
def load_analysis_model(model_path, device, datasets_dict=None):
    """Refactored model loading logic."""
    model_path = Path(model_path)
    state = torch.load(model_path, map_location=device)
    raw_state = state.get('model', state.get('state_dict', state))
    keys = list(raw_state.keys())

    # Infer Type
    model_type = 'macnet'
    if any(k.startswith('film_generator') for k in keys):
        model_type = 'resfilm'
    elif any(k.startswith('cls_token') or k.startswith('posenc') for k in keys) or any(k.startswith('blocks') for k in keys):
        model_type = 'vit'
        
    # Infer Input Channels
    in_ch = 14 # Default
    if datasets_dict:
        ref_ds = next((ds for ds in datasets_dict.values() if ds is not None), None)
        if ref_ds: in_ch = int(ref_ds.x.shape[1])

    # Check for hyperparams
    config_path = model_path.parent / 'hyperparameters.json'
    loaded_params = {}
    if config_path.exists():
        try:
             with open(config_path, 'r', encoding='utf-8') as f:
                loaded_params = json.load(f)
        except: pass

    # Initialize
    model = None
    if model_type == 'vit':
        # Default Params
        d_model = 256
        n_layers = 8
        n_heads = 4
        d_ff = 1024
        pos_num_bands = 32
        
        # Override from JSON
        if loaded_params:
            d_model = loaded_params.get('d_model', d_model)
            n_layers = loaded_params.get('n_layers', n_layers)
            n_heads = loaded_params.get('n_heads', n_heads)
            d_ff = loaded_params.get('d_ff', d_ff)
            pos_num_bands = loaded_params.get('pos_num_bands', pos_num_bands)
        else:
            # Fallback Inference
            if 'cls_token' in raw_state:
                d_model = raw_state['cls_token'].shape[-1]
            block_ids = [int(k.split('.')[1]) for k in keys if k.startswith('blocks.')]
            n_layers = max(block_ids) + 1 if block_ids else 8
            for k in keys:
                if 'mlp.fc1.weight' in k:
                    d_ff = raw_state[k].shape[0]
                    break
            n_heads = max(1, d_model // 32)
            if 'posenc.freq_bands' in raw_state:
                pos_num_bands = raw_state['posenc.freq_bands'].shape[0]

        # Check Legacy
        is_legacy = False
        if 'content_proj.weight' in raw_state:
            cp_shape = raw_state['content_proj.weight'].shape
            if cp_shape[0] == d_model // 2:
                is_legacy = True
        
        if is_legacy:
            model = LegacyAccretionTransformer(c_in=in_ch, d_model=d_model, n_layers=n_layers, 
                n_heads=n_heads, d_ff=d_ff, pos_num_bands=pos_num_bands).to(device)
        else:
            model = AccretionTransformer(c_in=in_ch, d_model=d_model, n_layers=n_layers, 
                n_heads=n_heads, d_ff=d_ff, pos_num_bands=pos_num_bands).to(device)
            
    elif model_type == 'resfilm':
        model = MACNetFiLM(in_channels=in_ch).to(device)
    else:
        model = MACNetRes_mbh(in_channels=in_ch).to(device)
        
    try:
        model.load_state_dict(raw_state, strict=True)
    except Exception:
        model.load_state_dict(raw_state, strict=False)
    
    model.eval()
    return model, model_type

# ==========================================
# 6. Prediction
# ==========================================
@torch.no_grad()
def predict_dataset(ds, model, device, model_type, batch_size=32, return_feats=False):
    if ds is None: return None, None, None
    
    x_all = ensure_tensor(ds.x)
    x_data = x_all.cpu().numpy()
    mbh_all = ensure_tensor(ds.mbh)
    coord_all = ensure_tensor(ds.coord) if getattr(ds, 'coord', None) is not None else None
    
    total = len(ds)
    preds = []
    feats_list = [] 
    
    bs = 8 if model_type == 'vit' else batch_size
    
    # Hook for features if requested
    hook_handle = None
    if return_feats and model_type == 'vit':
        def get_activation(storage):
            def hook(model, input, output):
                storage.append(output[:, 0, :].detach().cpu())
            return hook
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            hook_handle = model.blocks[-1].register_forward_hook(get_activation(feats_list))

    try:
        for start in range(0, total, bs):
            end = min(start + bs, total)
            x_batch = x_all[start:end].to(device)
            mbh_batch = mbh_all[start:end].to(device).view(-1)
            
            if model_type == 'vit':
                # if coord_all is None: raise ValueError("ViT requires coords")
                c_batch = coord_all[start:end].to(device)
                y_out = model(x_batch, c_batch[:, 0:1], c_batch[:, 1:2], mbh_batch)
            else:
                y_out = model(x_batch, mbh_batch)
            preds.append(y_out.cpu().view(-1))
    finally:
        if hook_handle: hook_handle.remove()
        
    y_pred = torch.cat(preds).numpy()
    y_true = ensure_tensor(ds.y).view(-1).numpy()
    feats = torch.cat(feats_list, dim=0).numpy() if feats_list else None
    
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    
    if ds.y_bondi is not None:
        df['y_bondi'] = ensure_tensor(ds.y_bondi).view(-1).numpy()
        df.loc[np.isclose(df['y_bondi'], -100), 'y_bondi'] = np.nan
        df['bondi_error'] = df['y_bondi'] - df['y_true']
        
    if hasattr(ds, 'time'): df['time'] = np.array(ds.time)
    if hasattr(ds, 'mdot_edd'): df['mdot_edd'] = np.array(ds.mdot_edd)
    
    df['error'] = df['y_pred'] - df['y_true']
    
    if hasattr(ds, 'type'): df['type'] = ds.type
    elif hasattr(ds, 'groups') and ds.groups is not None: df['type'] = 'unknown_group' 
    else: df['type'] = 'test_set'

    del x_all, mbh_all, coord_all
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    
    if return_feats:
        return df, feats, x_data
    return df

def save_to_h5(path, df, feats, x_data, ds_type):
    with h5py.File(path, 'w') as f:
        f.attrs['dataset_type'] = ds_type
        f.create_dataset('x', data=x_data, compression='gzip') 
        f.create_dataset('y_true', data=df['y_true'].values)
        f.create_dataset('y_pred', data=df['y_pred'].values)
        if 'type' in df.columns:
            dt = h5py.special_dtype(vlen=str) 
            f.create_dataset('label', data=df['type'].values.astype(object), dtype=dt)
        if feats is not None:
            f.create_dataset('latent', data=feats)
        if 'y_bondi' in df.columns:
            f.create_dataset('y_bondi', data=df['y_bondi'].values)
        if 'time' in df.columns:
            f.create_dataset('time', data=df['time'].values)
        if 'mdot_edd' in df.columns:
            f.create_dataset('mdot_edd', data=df['mdot_edd'].values)

# ==========================================
# 7. Visualization
# ==========================================
COLORS = ["#4B8D69", "#55658B", "#c1c16f", "#685587", "#B353CB", "#D78E5A", "#F06969"]

def save_plot(plt_obj, name, fig_dir, fmt='png'):
    path = fig_dir / f"{name}.{fmt}"
    plt_obj.savefig(path, dpi=150, bbox_inches='tight')

def plot_scatter(df, title, fig_dir, limit_range=True):
    plt.figure(figsize=(6, 6), dpi=120)
    vmin = min(df['y_true'].min(), df['y_pred'].min()) - 0.5
    vmax = max(df['y_true'].max(), df['y_pred'].max()) + 0.5
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5, label='Ideal', zorder=5)
    plt.plot([vmin, vmax], [vmin-1, vmax-1], 'k--', lw=0.8, alpha=0.5, label=r'$\pm$1 dex')
    plt.plot([vmin, vmax], [vmin+1, vmax+1], 'k--', lw=0.8, alpha=0.5)
    
    groups = df['type'].unique()
    for i, g in enumerate(groups):
        sub = df[df['type'] == g]
        plt.scatter(sub['y_true'], sub['y_pred'], s=8, alpha=0.4, 
                    color=COLORS[i % len(COLORS)], label=f"{g}", edgecolors='none')
                    
    if 'y_bondi' in df.columns:
        plt.scatter(df['y_true'], df['y_bondi'], s=15, marker='x', alpha=0.2, 
                    color='gray', label='Bondi', zorder=1)

    plt.title(title)
    plt.xlabel(r'True $\log_{10}(\dot{M}/\dot{M}_{Edd})$')
    plt.ylabel(r'Predicted $\log_{10}(\dot{M}/\dot{M}_{Edd})$')
    if limit_range:
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.3)
    save_plot(plt, f"scatter_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()

def plot_error_dist(df, title, fig_dir, model_label='Model'):
    plt.figure(figsize=(8, 5), dpi=120)
    plot_data = []
    tmp = df[['error', 'type']].copy()
    tmp['Method'] = model_label
    tmp.rename(columns={'error': 'Error'}, inplace=True)
    plot_data.append(tmp)
    if 'bondi_error' in df.columns:
        tmp2 = df[['bondi_error', 'type']].dropna().copy()
        tmp2['Method'] = 'Bondi'
        tmp2.rename(columns={'bondi_error': 'Error'}, inplace=True)
        plot_data.append(tmp2)
    final_df = pd.concat(plot_data)
    sns.violinplot(data=final_df, x='type', y='Error', hue='Method', 
                   split=True, inner='quartile', palette='muted')
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(-1, color='gray', linestyle=':', alpha=0.5)
    plt.title(f"Error Distribution: {title}")
    plt.ylabel(r"Error (dex)")
    plt.xticks(rotation=45, ha='right')
    save_plot(plt, f"violin_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()

def plot_kde_contour(df, title, fig_dir):
    plt.figure(figsize=(6, 6), dpi=120)
    sns.kdeplot(x=df['y_true'], y=df['y_pred'], hue=df['type'], 
                fill=True, alpha=0.3, levels=5, palette=COLORS[:len(df['type'].unique())])
    vmin, vmax = df['y_true'].min(), df['y_true'].max()
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1)
    plt.title(f"Density: {title}")
    save_plot(plt, f"kde_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()

def plot_mass_evolution(df, title, fig_dir, model_label='Model'):
    if 'time' not in df.columns or 'mdot_edd' not in df.columns: return
    df_plot = df.sort_values('time').copy()
    times = df_plot['time'].values
    dt = np.diff(times, prepend=times[0])
    mdot_true = (10 ** df_plot['y_true']) * df_plot['mdot_edd']
    mdot_pred = (10 ** df_plot['y_pred']) * df_plot['mdot_edd']
    UNIT_MASS = 2.5e7 
    mass_inc_true = np.cumsum(mdot_true * dt) * UNIT_MASS
    mass_inc_pred = np.cumsum(mdot_pred * dt) * UNIT_MASS
    
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(times, mass_inc_true, 'k-', lw=2, label='True', alpha=0.8)
    plt.plot(times, mass_inc_pred, 'r--', lw=2, label=f'Predicted ({model_label})', alpha=0.8)
    if 'y_bondi' in df.columns:
        mdot_bondi = (10 ** df_plot['y_bondi']) * df_plot['mdot_edd']
        mass_inc_bondi = np.cumsum(mdot_bondi.fillna(0) * dt) * UNIT_MASS
        plt.plot(times, mass_inc_bondi, 'g:', lw=1.5, label='Bondi', alpha=0.6)
    plt.yscale('log')
    plt.title(f"BH Growth: {title}")
    plt.xlabel("Time (Gyr)")
    plt.ylabel(r"$\Delta M_{BH}$ [$M_{\odot}$]")
    plt.legend()
    save_plot(plt, f"mass_evo_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()

def get_metrics_summary(y_true, y_pred, name=""):
    try:
        from sklearn.metrics import r2_score
    except:
        def r2_score(t, p): return 1 - np.sum((t-p)**2)/np.sum((t-np.mean(t))**2)
    
    d = y_true - y_pred
    return {
        'Dataset': name,
        'RMSE': np.sqrt(np.mean(d**2)),
        'MAE': np.mean(np.abs(d)),
        'R2': r2_score(y_true, y_pred),
        'OutlierFrac (>1dex)': np.mean(np.abs(d) > 1.0) * 100
    }

"""
alma_autoencoder.py
-------------------
Convolutional Autoencoder (CAE) for ALMA protoplanetary disk images.

Architecture is directly informed by DeepFocus (Delli Veneri et al. 2022):

  DeepFocus lessons applied here
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - ``block_sizes``      → channel progression  [32, 64, 128, 256, 512]
  - ``depths``           → repeated conv blocks per stage  [2, 2, 2, 2]
  - ``hidden_size``      → explicit latent vector dimension (256)
  - ``encoder_activation = 'leaky_relu'``  → used throughout encoder
  - ``decoder_activation = 'leaky_relu'``  → used in decoder body
  - ``final_activation = 'sigmoid'``       → output gate
  - ``interpolation = True``               → bilinear up-sample before
                                              transposed conv (avoids
                                              checkerboard artefacts)
  - ``skip_connections = False``           → pure autoencoder bottleneck
                                              (not U-Net) — forces latent
                                              space to carry all info
  - ``criterion = ['L1', 'SSIM']``         → combined L1 + SSIM loss
  - ``warm_start``                         → linear LR warm-up
  - ``weight_decay = 0.0001``              → AdamW regularisation
  - ``preprocess = 'log'``                 → log-stretched inputs, so
                                              MSE is replaced with L1
                                              which is more robust to
                                              the resulting distribution

Spatial flow with 600×600 input:
    Encoder:  600 → 300 → 150 →  75 →  38 →  19   (stride-2 each step)
    Latent:   FC  19×19×512  →  256
    Decoder:  FC  256 → 19×19×512
              19 →  38 →  75 → 150 → 300 → 600   (bilinear + conv)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from pytorch_msssim import SSIM
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False
    logger.warning("pytorch-msssim not installed; SSIM term disabled. pip install pytorch-msssim")

__all__ = [
    "ConvBlock",
    "Encoder",
    "Decoder",
    "ALMAAutoencoder",
    "CombinedLoss",
    "TrainingConfig",
    "TrainingHistory",
    "train",
    "save_checkpoint",
    "load_checkpoint",
]

_BOTTLENECK_H   = 19
_BOTTLENECK_W   = 19
_BOTTLENECK_CH  = 512
_BOTTLENECK_FLAT = _BOTTLENECK_CH * _BOTTLENECK_H * _BOTTLENECK_W   # 184,832

def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "leaky_relu": return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == "relu": return nn.ReLU(inplace=True)
    elif name == "selu": return nn.SELU(inplace=True)
    elif name == "tanh": return nn.Tanh()
    elif name == "sigmoid": return nn.Sigmoid()
    elif name in ("none", ""): return nn.Identity()
    else: raise ValueError(f"Unknown activation: '{name}'")

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int=2, kernel_size: int=3, stride: int=1, activation: str="leaky_relu") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=kernel_size, stride=s, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_ch),
                _get_activation(activation),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class Encoder(nn.Module):
    CHANNELS = [32, 64, 128, 256, 512]
    def __init__(self, in_channels: int=1, latent_dim: int=256, block_depth: int=2, activation: str="leaky_relu") -> None:
        super().__init__()
        self.latent_dim = latent_dim
        stages: list[nn.Module] = []
        prev = in_channels
        for ch in self.CHANNELS:
            stages.append(ConvBlock(prev, ch, depth=block_depth, stride=2, activation=activation))
            prev = ch
        self.stages = nn.Sequential(*stages)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(_BOTTLENECK_FLAT, latent_dim, bias=True),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.flatten(self.stages(x)))

class Decoder(nn.Module):
    CHANNELS = [256, 128, 64, 32, 16]
    TARGET_SIZES = [38, 75, 150, 300, 600]
    def __init__(self, in_channels: int=1, latent_dim: int=256, block_depth: int=2, activation: str="leaky_relu") -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, _BOTTLENECK_FLAT, bias=True),
            _get_activation(activation),
        )
        self.unflatten = nn.Unflatten(1, (_BOTTLENECK_CH, _BOTTLENECK_H, _BOTTLENECK_W))
        self.up_stages = nn.ModuleList()
        prev = _BOTTLENECK_CH
        for ch in self.CHANNELS:
            self.up_stages.append(ConvBlock(prev, ch, depth=block_depth, stride=1, activation=activation))
            prev = ch
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.CHANNELS[-1], in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.target_sizes = self.TARGET_SIZES

    def forward(self, z: Tensor) -> Tensor:
        x = self.unflatten(self.fc(z))
        for stage, size in zip(self.up_stages, self.target_sizes):
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            x = stage(x)
        return self.output_conv(x)

class ALMAAutoencoder(nn.Module):
    def __init__(self, in_channels: int=1, latent_dim: int=256, block_depth: int=2, encoder_activation: str="leaky_relu", decoder_activation: str="leaky_relu") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, block_depth=block_depth, activation=encoder_activation)
        self.decoder = Decoder(in_channels=in_channels, latent_dim=latent_dim, block_depth=block_depth, activation=decoder_activation)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: Tensor) -> Tensor: return self.encoder(x)
    def decode(self, z: Tensor) -> Tensor: return self.decoder(z)
    def forward(self, x: Tensor) -> Tensor: return self.decode(self.encode(x))
    def count_parameters(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CombinedLoss(nn.Module):
    def __init__(self, alpha: float=0.5, data_range: float=1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=data_range, size_average=True, channel=1) if _SSIM_AVAILABLE else None

    def forward(self, recon: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        l1_val = self.l1(recon, target)
        if self.ssim is not None:
            ssim_score = self.ssim(recon, target)
            total = self.alpha * l1_val + (1.0 - self.alpha) * (1.0 - ssim_score)
            return total, l1_val.detach(), ssim_score.detach()
        return l1_val, l1_val.detach(), torch.tensor(0.0)

@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    loss_alpha: float = 0.5
    warm_start: bool = True
    warm_start_epochs: int = 10
    early_stopping: bool = True
    patience: int = 15
    checkpoint_dir: str | Path = Path("checkpoints")
    save_every_n: int = 10
    device: str | torch.device = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    train_l1:   list[float] = field(default_factory=list)
    val_l1:     list[float] = field(default_factory=list)
    train_ssim: list[float] = field(default_factory=list)
    val_ssim:   list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")

class WarmUpScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, target_lr: float, warm_epochs: int, plateau_patience: int=5) -> None:
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warm_epochs = warm_epochs
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=plateau_patience)

    def step(self, epoch: int, val_loss: float) -> float:
        if epoch < self.warm_epochs:
            lr = self.target_lr * (epoch + 1) / self.warm_epochs
            for pg in self.optimizer.param_groups: pg["lr"] = lr
        else:
            self.plateau.step(val_loss)
        return self.optimizer.param_groups[0]["lr"]

def _run_epoch(model: ALMAAutoencoder, loader: DataLoader, criterion: CombinedLoss, optimizer: Optional[torch.optim.Optimizer], device: torch.device, desc: str) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()
    total = l1_sum = ssim_sum = 0.0
    n = len(loader)

    with ctx:
        bar = tqdm(loader, desc=desc, leave=False, unit="batch", ncols=120)
        for batch in bar:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            recon = model(images)
            loss, l1_val, ssim_val = criterion(recon, images)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total += loss.item()
            l1_sum += l1_val.item()
            ssim_sum += ssim_val.item()

    return total / n, l1_sum / n, ssim_sum / n

def save_checkpoint(model: ALMAAutoencoder, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, save_path: str | Path, metadata: Optional[dict] = None) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "in_channels": model.in_channels,
        "latent_dim": model.latent_dim,
        "metadata": metadata or {},
    }, save_path)

def load_checkpoint(checkpoint_path: str | Path, device: Optional[torch.device] = None) -> tuple[ALMAAutoencoder, dict]:
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(checkpoint_path, map_location=device)
    model = ALMAAutoencoder(in_channels=bundle.get("in_channels", 1), latent_dim=bundle["latent_dim"])
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()
    return model, bundle

def train(model: ALMAAutoencoder, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, config: Optional[TrainingConfig] = None) -> TrainingHistory:
    if config is None: config = TrainingConfig()
    device = torch.device(config.device)
    model = model.to(device)
    criterion = CombinedLoss(alpha=config.loss_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = WarmUpScheduler(optimizer, target_lr=config.learning_rate, warm_epochs=config.warm_start_epochs if config.warm_start else 0, plateau_patience=5)
    history = TrainingHistory()
    ckpt_dir = Path(config.checkpoint_dir)
    no_improve = 0

    for epoch in range(config.epochs):
        ep = epoch + 1
        tr_loss, tr_l1, tr_ssim = _run_epoch(model, train_loader, criterion, optimizer, device, desc=f"Ep {ep:>3}/{config.epochs} [train]")
        vl_loss, vl_l1, vl_ssim = _run_epoch(model, val_loader, criterion, None, device, desc=f"Ep {ep:>3}/{config.epochs} [  val]") if val_loader else (tr_loss, tr_l1, tr_ssim)
        
        current_lr = scheduler.step(epoch, vl_loss)
        history.train_loss.append(tr_loss); history.val_loss.append(vl_loss)
        
        if vl_loss < history.best_val_loss:
            history.best_val_loss = vl_loss
            history.best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, vl_loss, ckpt_dir / "best.pth", metadata={"ssim_available": _SSIM_AVAILABLE})
        else:
            no_improve += 1

        if config.early_stopping and no_improve >= config.patience: break
    return history

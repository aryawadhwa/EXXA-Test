"""
autoencoder.py
--------------
Convolutional Autoencoder for 1×600×600 single-channel astronomical images.

Architecture
~~~~~~~~~~~~
The Encoder reduces the spatial resolution through five strided-convolution
blocks (each followed by BatchNorm + LeakyReLU) before flattening and
projecting to a compact 1-D latent vector.

The Decoder mirrors the Encoder exactly: a linear projection inflates the
latent vector back to the flattened bottleneck volume, which is then
upscaled through five transposed-convolution blocks, finishing with a
Sigmoid to bound outputs to [0, 1].

Spatial flow (H×W):
    600 → 300 → 150 → 75 → 38 → 19   (encoder, ÷2 each step)
     19 →  38 →  75 → 150 → 300 → 600 (decoder, ×2 each step, with
                                         output_padding to hit exact dims)

Latent vector size is configurable (default 256).

Training
~~~~~~~~
The training loop minimises a weighted combination of:
  • Mean Squared Error (MSE)        – pixel-level fidelity
  • Multi-Scale SSIM (MS-SSIM)      – perceptual / structural similarity

    loss = α·MSE + (1 − α)·(1 − MS-SSIM)      default α = 0.15

Dependencies
~~~~~~~~~~~~
    pip install torch torchvision pytorch-msssim tqdm

Usage
~~~~~
    from autoencoder import ConvAutoencoder, train

    model = ConvAutoencoder(latent_dim=256)
    train(model, train_loader, val_loader=val_loader, epochs=50)

    # Extract latent vectors at inference time
    z = model.encode(image_batch)          # (B, 256)
    recon = model.decode(z)                # (B, 1, 600, 600)
    recon = model(image_batch)             # encode + decode in one call
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MS-SSIM import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from pytorch_msssim import MS_SSIM

    _MSSSIM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MSSSIM_AVAILABLE = False
    logger.warning(
        "pytorch-msssim not found. Install it with:\n"
        "    pip install pytorch-msssim\n"
        "Falling back to MSE-only loss."
    )


__all__ = [
    "ConvAutoencoder",
    "EncoderBlock",
    "DecoderBlock",
    "TrainingConfig",
    "TrainingHistory",
    "train",
    "save_checkpoint",
    "load_checkpoint",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spatial size at the encoder bottleneck (before the FC projection)
# With input 600×600 and 5× stride-2 convolutions:
#   600 → 300 → 150 → 75 → 38 → 19
_BOTTLENECK_SPATIAL: int = 19

# Channel depth at the final encoder conv layer
_BOTTLENECK_CHANNELS: int = 512

# Flat size of the bottleneck feature map
_BOTTLENECK_FLAT: int = _BOTTLENECK_CHANNELS * _BOTTLENECK_SPATIAL ** 2  # 184,832


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class EncoderBlock(nn.Sequential):
    """One strided-convolution encoding block.

    Conv(stride=2) → BatchNorm2d → LeakyReLU

    Parameters
    ----------
    in_channels:
        Number of input feature-map channels.
    out_channels:
        Number of output feature-map channels.
    kernel_size:
        Convolution kernel size (default 4 — standard in autoencoders
        as it avoids checkerboard artefacts with stride 2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                bias=False,          # BatchNorm subsumes the bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )


class DecoderBlock(nn.Sequential):
    """One transposed-convolution decoding block.

    ConvTranspose(stride=2) → BatchNorm2d → ReLU

    Parameters
    ----------
    in_channels:
        Number of input feature-map channels.
    out_channels:
        Number of output feature-map channels.
    kernel_size:
        Kernel size (default 4, matching EncoderBlock).
    output_padding:
        Extra padding appended to one side of the output to correct
        for rounding when the input spatial size is odd (e.g. 19→38
        would produce 37 without output_padding=1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        output_padding: int = 0,
    ) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Five-stage strided convolutional encoder.

    Spatial progression (H × W):
        600 → 300 → 150 → 75 → 38 → 19

    Channel progression:
        1 → 32 → 64 → 128 → 256 → 512

    The final feature map (512 × 19 × 19) is flattened and projected to
    a 1-D latent vector of size ``latent_dim``.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the bottleneck latent vector.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_blocks = nn.Sequential(
            EncoderBlock(1,    32),   # 600 → 300
            EncoderBlock(32,   64),   # 300 → 150
            EncoderBlock(64,  128),   # 150 →  75
            EncoderBlock(128, 256),   #  75 →  38
            EncoderBlock(256, 512),   #  38 →  19
        )

        self.flatten = nn.Flatten()   # (B, 512, 19, 19) → (B, 184832)

        self.fc = nn.Sequential(
            nn.Linear(_BOTTLENECK_FLAT, latent_dim, bias=True),
            nn.LayerNorm(latent_dim),  # stabilises training; aids downstream use
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of images to latent vectors.

        Parameters
        ----------
        x:
            Float32 tensor of shape ``(B, 1, 600, 600)``.

        Returns
        -------
        Tensor
            Float32 tensor of shape ``(B, latent_dim)``.
        """
        features = self.conv_blocks(x)      # (B, 512, 19, 19)
        flat = self.flatten(features)       # (B, 184832)
        return self.fc(flat)                # (B, latent_dim)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """Five-stage transposed-convolution decoder.

    Spatial progression (H × W):
        19 → 38 → 75 → 150 → 300 → 600

    Note: the 19→38 step requires ``output_padding=1`` to compensate
    for the asymmetric rounding at the 75→38 encoder step (75/2 = 37.5
    rounds to 38 with padding=1, which must be reversed precisely).

    Parameters
    ----------
    latent_dim:
        Dimensionality of the input latent vector (must match Encoder).
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, _BOTTLENECK_FLAT, bias=True),
            nn.ReLU(inplace=True),
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(_BOTTLENECK_CHANNELS, _BOTTLENECK_SPATIAL, _BOTTLENECK_SPATIAL),
        )  # (B, 184832) → (B, 512, 19, 19)

        self.conv_blocks = nn.Sequential(
            DecoderBlock(512, 256, output_padding=1),  #  19 →  38  (needs +1)
            DecoderBlock(256, 128, output_padding=1),  #  38 →  75  (needs +1)
            DecoderBlock(128,  64),                    #  75 → 150
            DecoderBlock(64,   32),                    # 150 → 300
            # Final block: no BN/ReLU — raw conv + Sigmoid for [0,1] output
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 300 → 600
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode a batch of latent vectors to reconstructed images.

        Parameters
        ----------
        z:
            Float32 tensor of shape ``(B, latent_dim)``.

        Returns
        -------
        Tensor
            Float32 tensor of shape ``(B, 1, 600, 600)`` with values in
            ``[0, 1]``.
        """
        x = self.fc(z)               # (B, 184832)
        x = self.unflatten(x)        # (B, 512, 19, 19)
        return self.conv_blocks(x)   # (B, 1, 600, 600)


# ---------------------------------------------------------------------------
# Autoencoder (top-level model)
# ---------------------------------------------------------------------------


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for 1×600×600 astronomical images.

    The latent representation is a normalised 1-D vector of size
    ``latent_dim`` that can be extracted independently via
    ``model.encode(x)``.  This vector is suitable for downstream tasks
    such as anomaly detection, clustering, or similarity search.

    Parameters
    ----------
    latent_dim:
        Size of the bottleneck latent vector.  128 is compact;
        256 (default) retains more structural detail.

    Examples
    --------
    >>> model = ConvAutoencoder(latent_dim=256)
    >>> x = torch.rand(4, 1, 600, 600)

    Forward pass (encode + decode):
    >>> recon = model(x)                # (4, 1, 600, 600)

    Encode only (latent extraction):
    >>> z = model.encode(x)             # (4, 256)

    Decode only (generation / interpolation):
    >>> recon = model.decode(z)         # (4, 1, 600, 600)
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming Normal initialisation to all Conv/Linear layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """Compress a batch of images to a 1-D latent vector per sample.

        Parameters
        ----------
        x:
            Float32 image tensor of shape ``(B, 1, 600, 600)``.

        Returns
        -------
        Tensor
            Float32 latent tensor of shape ``(B, latent_dim)``.
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct images from latent vectors.

        Parameters
        ----------
        z:
            Float32 latent tensor of shape ``(B, latent_dim)``.

        Returns
        -------
        Tensor
            Reconstructed image tensor of shape ``(B, 1, 600, 600)``.
        """
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """Full encode–decode pass.

        Parameters
        ----------
        x:
            Float32 image tensor of shape ``(B, 1, 600, 600)``.

        Returns
        -------
        Tensor
            Reconstructed image tensor of shape ``(B, 1, 600, 600)``.
        """
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params_m = self.count_parameters() / 1_000_000
        return (
            f"{self.__class__.__name__}("
            f"latent_dim={self.latent_dim}, "
            f"params={params_m:.2f}M)"
        )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class CombinedLoss(nn.Module):
    """Weighted combination of MSE and (1 − MS-SSIM).

    Loss = α · MSE  +  (1 − α) · (1 − MS-SSIM)

    Using both terms simultaneously encourages the model to minimise
    pixel-level error (MSE) *and* preserve multi-scale perceptual
    structure (MS-SSIM).  A low α (e.g. 0.15) down-weights the noisy
    MSE gradient while keeping it as a regulariser.

    Falls back to pure MSE if ``pytorch-msssim`` is not installed.

    Parameters
    ----------
    alpha:
        Weight assigned to the MSE term.  ``(1 − alpha)`` is assigned
        to the MS-SSIM term.
    data_range:
        The dynamic range of the input images.  Since images are
        normalised to ``[0, 1]``, this should be ``1.0``.
    """

    def __init__(self, alpha: float = 0.15, data_range: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha
        self.mse = nn.MSELoss()

        if _MSSSIM_AVAILABLE:
            # size_average=True returns a scalar directly
            self.msssim = MS_SSIM(
                data_range=data_range,
                size_average=True,
                channel=1,
            )
        else:
            self.msssim = None  # type: ignore[assignment]

    def forward(
        self, recon: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the combined loss plus its two components.

        Parameters
        ----------
        recon:
            Reconstructed images ``(B, 1, H, W)``.
        target:
            Ground-truth images ``(B, 1, H, W)``.

        Returns
        -------
        total : Tensor
            Scalar combined loss.
        mse_val : Tensor
            Scalar MSE component (detached — for logging only).
        ssim_val : Tensor
            Scalar MS-SSIM score ∈ [0, 1] (detached — for logging only).
            Returns ``torch.tensor(0.0)`` if MS-SSIM is unavailable.
        """
        mse_val = self.mse(recon, target)

        if self.msssim is not None:
            ssim_score = self.msssim(recon, target)          # ∈ [0, 1]
            total = self.alpha * mse_val + (1.0 - self.alpha) * (1.0 - ssim_score)
            return total, mse_val.detach(), ssim_score.detach()
        else:
            return mse_val, mse_val.detach(), torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: ConvAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str | Path,
    metadata: Optional[dict] = None,
) -> None:
    """Serialise model weights and training state to a ``.pth`` file.

    The saved bundle contains everything needed to resume training *or*
    to distribute the model as a pre-trained artefact:

    ``model_state_dict``     — learned weights (primary deliverable)
    ``optimizer_state_dict`` — optimiser momentum / adaptive terms
    ``epoch``                — last completed epoch (0-indexed)
    ``loss``                 — validation loss at save time
    ``latent_dim``           — latent vector size (needed to re-construct)
    ``metadata``             — optional free-form dict (e.g. dataset info)

    Parameters
    ----------
    model:
        The ``ConvAutoencoder`` instance to save.
    optimizer:
        The optimiser whose state should be preserved.
    epoch:
        Last completed training epoch.
    loss:
        Validation loss at this checkpoint.
    save_path:
        Destination path (including filename, e.g. ``weights/best.pth``).
    metadata:
        Optional dictionary of extra information to embed in the
        checkpoint (dataset name, normalisation stats, etc.).

    Examples
    --------
    >>> save_checkpoint(model, optimizer, epoch=10, loss=0.0023,
    ...                 save_path="checkpoints/best.pth")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "latent_dim": model.latent_dim,
        "metadata": metadata or {},
    }

    torch.save(bundle, save_path)
    logger.info("Checkpoint saved → %s  (epoch=%d, loss=%.6f)", save_path, epoch, loss)


def load_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> tuple[ConvAutoencoder, dict]:
    """Load a ``ConvAutoencoder`` from a ``.pth`` checkpoint.

    The ``latent_dim`` stored in the checkpoint is used to re-construct
    the model architecture before loading the weights, so no manual
    configuration is needed.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` file produced by ``save_checkpoint``.
    device:
        Target device.  Defaults to ``cuda`` if available, else ``cpu``.

    Returns
    -------
    model : ConvAutoencoder
        Model with loaded weights set to ``eval()`` mode.
    bundle : dict
        Full checkpoint dictionary (contains ``epoch``, ``loss``,
        ``metadata``, ``optimizer_state_dict``).

    Examples
    --------
    >>> model, ckpt = load_checkpoint("checkpoints/best.pth")
    >>> z = model.encode(image_batch)   # ready for inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle: dict = torch.load(checkpoint_path, map_location=device)

    model = ConvAutoencoder(latent_dim=bundle["latent_dim"])
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint from '%s'  (epoch=%d, loss=%.6f)",
        checkpoint_path,
        bundle.get("epoch", -1),
        bundle.get("loss", float("nan")),
    )
    return model, bundle


# ---------------------------------------------------------------------------
# Training history dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Container for per-epoch training and validation metrics.

    Attributes
    ----------
    train_loss : list[float]
        Combined loss averaged over the training set per epoch.
    val_loss : list[float]
        Combined loss averaged over the validation set per epoch.
    train_mse : list[float]
        MSE component of the training loss.
    val_mse : list[float]
        MSE component of the validation loss.
    train_ssim : list[float]
        MS-SSIM score on the training set (0 if unavailable).
    val_ssim : list[float]
        MS-SSIM score on the validation set (0 if unavailable).
    best_epoch : int
        Epoch index at which the best validation loss was achieved.
    best_val_loss : float
        Best (lowest) validation loss observed during training.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_mse: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    train_ssim: list[float] = field(default_factory=list)
    val_ssim: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


# ---------------------------------------------------------------------------
# Training configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """All hyper-parameters governing a training run.

    Parameters
    ----------
    epochs:
        Total number of training epochs.
    learning_rate:
        Initial learning rate for AdamW.
    weight_decay:
        L2 regularisation coefficient.
    loss_alpha:
        Weight on the MSE term in ``CombinedLoss``.
    scheduler_patience:
        Number of epochs with no val-loss improvement before the
        ``ReduceLROnPlateau`` scheduler halves the learning rate.
    early_stopping_patience:
        Stop training after this many epochs with no improvement.
        Set to ``None`` to disable early stopping.
    checkpoint_dir:
        Directory in which ``.pth`` files are written.
    save_every_n_epochs:
        Frequency at which periodic checkpoints are saved.
        ``None`` disables periodic saving (only best model is kept).
    device:
        Torch device string or ``torch.device``.  Defaults to
        ``"cuda"`` if available, else ``"cpu"``.
    """

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss_alpha: float = 0.15
    scheduler_patience: int = 5
    early_stopping_patience: Optional[int] = 15
    checkpoint_dir: str | Path = Path("checkpoints")
    save_every_n_epochs: Optional[int] = 10
    device: str | torch.device = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _run_epoch(
    model: ConvAutoencoder,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    desc: str,
) -> tuple[float, float, float]:
    """Run one full pass (train or eval) over a DataLoader.

    Parameters
    ----------
    model:
        The autoencoder.
    loader:
        DataLoader yielding ``(B, 1, 600, 600)`` tensors.
    criterion:
        ``CombinedLoss`` instance.
    optimizer:
        Pass the optimiser to perform gradient updates (train mode).
        Pass ``None`` to run in evaluation mode (no updates).
    device:
        Compute device.
    desc:
        tqdm bar description prefix.

    Returns
    -------
    avg_loss, avg_mse, avg_ssim : float
        Batch-averaged combined loss, MSE, and MS-SSIM for this epoch.
    """
    training = optimizer is not None
    model.train(training)
    context = torch.enable_grad() if training else torch.no_grad()

    total_loss = total_mse = total_ssim = 0.0
    n_batches = len(loader)

    with context:
        bar = tqdm(loader, desc=desc, leave=False, unit="batch", ncols=110)
        for batch in bar:
            # Loaders may return (image,) tuples or raw tensors
            images: Tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)

            recon = model(images)
            loss, mse_val, ssim_val = criterion(recon, images)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping prevents exploding gradients in deep nets
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_mse += mse_val.item()
            total_ssim += ssim_val.item()

            bar.set_postfix(
                loss=f"{loss.item():.5f}",
                mse=f"{mse_val.item():.5f}",
                ssim=f"{ssim_val.item():.4f}",
            )

    return (
        total_loss / n_batches,
        total_mse / n_batches,
        total_ssim / n_batches,
    )


def train(
    model: ConvAutoencoder,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
) -> TrainingHistory:
    """Train a ``ConvAutoencoder`` and save checkpoints.

    This is the primary entry-point for a full training run.  It handles:
      - AdamW optimisation with ``ReduceLROnPlateau`` scheduling
      - Per-epoch tqdm progress bars with live loss / SSIM display
      - Best-model checkpointing (by validation loss)
      - Periodic checkpoint saves every ``N`` epochs
      - Optional early stopping

    Parameters
    ----------
    model:
        An initialised ``ConvAutoencoder`` (not yet moved to device —
        this function handles ``.to(device)`` internally).
    train_loader:
        DataLoader for the training split.  Must yield tensors (or
        tuples whose first element is a tensor) of shape
        ``(B, 1, 600, 600)`` with values in ``[0, 1]``.
    val_loader:
        Optional DataLoader for the validation split.  When provided,
        best-model checkpointing is based on validation loss.  When
        ``None``, training loss is used as a proxy.
    config:
        ``TrainingConfig`` instance.  Constructed with defaults if
        ``None``.

    Returns
    -------
    TrainingHistory
        Dataclass containing per-epoch lists of loss, MSE, and MS-SSIM
        for both train and validation splits, plus the best epoch index
        and best validation loss.

    Examples
    --------
    >>> model   = ConvAutoencoder(latent_dim=256)
    >>> history = train(model, train_loader, val_loader, config=TrainingConfig(epochs=50))
    >>> print(f"Best val loss: {history.best_val_loss:.6f} @ epoch {history.best_epoch}")
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)
    model = model.to(device)

    criterion = CombinedLoss(alpha=config.loss_alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.scheduler_patience,
    )

    history = TrainingHistory()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    no_improve_count: int = 0
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("ConvAutoencoder  |  latent_dim=%d  |  params=%.2fM",
                model.latent_dim, model.count_parameters() / 1e6)
    logger.info("Device: %s  |  Epochs: %d  |  LR: %.0e", device, config.epochs, config.learning_rate)
    logger.info("Loss: α=%.2f·MSE + (1-α)·(1-MS-SSIM)  [MS-SSIM available: %s]",
                config.loss_alpha, _MSSSIM_AVAILABLE)
    logger.info("=" * 60)

    for epoch in range(config.epochs):
        epoch_num = epoch + 1
        lr_now = optimizer.param_groups[0]["lr"]

        # ── Train ──────────────────────────────────────────────────────
        tr_loss, tr_mse, tr_ssim = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch_num:>3}/{config.epochs} [train]",
        )
        history.train_loss.append(tr_loss)
        history.train_mse.append(tr_mse)
        history.train_ssim.append(tr_ssim)

        # ── Validation ─────────────────────────────────────────────────
        if val_loader is not None:
            vl_loss, vl_mse, vl_ssim = _run_epoch(
                model, val_loader, criterion, None, device,
                desc=f"Epoch {epoch_num:>3}/{config.epochs} [ val ]",
            )
        else:
            vl_loss, vl_mse, vl_ssim = tr_loss, tr_mse, tr_ssim

        history.val_loss.append(vl_loss)
        history.val_mse.append(vl_mse)
        history.val_ssim.append(vl_ssim)

        # ── Scheduler step ─────────────────────────────────────────────
        scheduler.step(vl_loss)

        # ── Logging ────────────────────────────────────────────────────
        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | "
            "train: loss=%.5f mse=%.5f ssim=%.4f | "
            "val: loss=%.5f mse=%.5f ssim=%.4f | "
            "lr=%.2e | elapsed=%.0fs",
            epoch_num, config.epochs,
            tr_loss, tr_mse, tr_ssim,
            vl_loss, vl_mse, vl_ssim,
            lr_now, elapsed,
        )

        # ── Best-model checkpoint ──────────────────────────────────────
        if vl_loss < history.best_val_loss:
            history.best_val_loss = vl_loss
            history.best_epoch = epoch
            no_improve_count = 0
            save_checkpoint(
                model, optimizer, epoch, vl_loss,
                save_path=checkpoint_dir / "best.pth",
                metadata={
                    "train_loss": tr_loss,
                    "val_loss": vl_loss,
                    "ms_ssim_available": _MSSSIM_AVAILABLE,
                },
            )
        else:
            no_improve_count += 1

        # ── Periodic checkpoint ────────────────────────────────────────
        if (
            config.save_every_n_epochs is not None
            and epoch_num % config.save_every_n_epochs == 0
        ):
            save_checkpoint(
                model, optimizer, epoch, vl_loss,
                save_path=checkpoint_dir / f"epoch_{epoch_num:04d}.pth",
            )

        # ── Early stopping ─────────────────────────────────────────────
        if (
            config.early_stopping_patience is not None
            and no_improve_count >= config.early_stopping_patience
        ):
            logger.info(
                "Early stopping triggered: no improvement for %d epochs.",
                config.early_stopping_patience,
            )
            break

    total_time = time.time() - t0
    logger.info(
        "Training complete. Best val loss: %.6f @ epoch %d. "
        "Total time: %.0fs.",
        history.best_val_loss, history.best_epoch + 1, total_time,
    )
    return history


# ---------------------------------------------------------------------------
# Minimal smoke-test  (python autoencoder.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n── Architecture ──")
    model = ConvAutoencoder(latent_dim=256)
    print(model)
    print(f"  Encoder params : {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Decoder params : {sum(p.numel() for p in model.decoder.parameters()):,}")

    print("\n── Shape check ──")
    dummy = torch.rand(2, 1, 600, 600)
    z = model.encode(dummy)
    recon = model.decode(z)
    print(f"  Input  : {list(dummy.shape)}")
    print(f"  Latent : {list(z.shape)}")
    print(f"  Output : {list(recon.shape)}")

    assert z.shape == (2, 256),         f"Latent shape mismatch: {z.shape}"
    assert recon.shape == (2, 1, 600, 600), f"Output shape mismatch: {recon.shape}"
    assert recon.min() >= 0.0 and recon.max() <= 1.0, "Output not in [0,1]"

    print("\n── Loss check ──")
    loss_fn = CombinedLoss(alpha=0.15)
    total, mse_v, ssim_v = loss_fn(recon, dummy)
    print(f"  Combined loss : {total.item():.6f}")
    print(f"  MSE           : {mse_v.item():.6f}")
    print(f"  MS-SSIM score : {ssim_v.item():.4f}  (0.0 if pkg not installed)")

    print("\nAll assertions passed ✓")

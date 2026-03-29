"""
alma_clustering.py
------------------
Unsupervised clustering of ALMA protoplanetary disk images via the
latent space of a trained ALMAAutoencoder.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from alma_autoencoder import ALMAAutoencoder, load_checkpoint
from alma_dataset import ALMADiskDataset, FITSMeta, PreprocessMode

__all__ = [
    "ClusteringConfig",
    "ClusteringResult",
    "extract_latent_vectors",
    "reduce_dimensions",
    "cluster_latent_space",
    "compute_radial_profile",
    "plot_embedding",
    "plot_cluster_gallery",
    "plot_cluster_radial_profiles",
    "run_pipeline",
]

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#0a0a12",
    "axes.facecolor": "#0a0a12",
    "axes.edgecolor": "#2a2a40",
    "axes.labelcolor": "#c8c8e8",
    "xtick.color": "#7070a0",
    "ytick.color": "#7070a0",
    "text.color": "#c8c8e8",
    "grid.color": "#1a1a2a",
    "grid.linewidth": 0.6,
    "font.family": "monospace",
    "savefig.facecolor": "#0a0a12",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

_PALETTE = [
    "#00cfff", "#ff7043", "#69ff47", "#e040fb", "#ffde03",
    "#ff4081", "#40ffcb", "#ff9100", "#b0bec5", "#f44336",
]
_NOISE_COLOR = "#404040"

_SHAPE_LABELS = {
    "compact":       "Compact (unresolved)",
    "extended":      "Extended disk",
    "single_gap":    "Gap candidate ⚡",
    "ring":          "Ring dominated",
    "complex":       "Complex structure",
}

@dataclass
class ClusteringConfig:
    reducer: Literal["umap", "tsne"] = "umap"
    clusterer: Literal["kmeans", "dbscan"] = "kmeans"
    n_clusters: int = 6
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.10
    tsne_perplexity: float = 30.0
    random_state: int = 42
    batch_size: int = 8
    num_workers: int = 4
    preprocess: PreprocessMode = "log"
    normalize: bool = True
    n_channels: int = 1
    layer_start: int = 0
    samples_per_cluster: int = 3
    radial_bins: int = 30
    output_dir: str | Path = Path("results")

@dataclass
class ClusteringResult:
    latent_vectors: np.ndarray
    embedding_2d:   np.ndarray
    labels:         np.ndarray
    file_paths:     list[Path]
    meta_list:      list[FITSMeta]
    n_clusters:     int
    cluster_sizes:  dict[int, int] = field(default_factory=dict)
    shape_labels:   dict[int, str] = field(default_factory=dict)

def extract_latent_vectors(
    model: ALMAAutoencoder, fits_dir: str | Path, config: ClusteringConfig, device: torch.device,
) -> tuple[np.ndarray, list[Path], list[FITSMeta]]:
    dataset = ALMADiskDataset(fits_dir=fits_dir, n_channels=config.n_channels, layer_start=config.layer_start, preprocess=config.preprocess, normalize=config.normalize, transform=None)
    def _collate(batch):
        tensors, metas = zip(*batch)
        return torch.stack(tensors), list(metas)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=device.type == "cuda", collate_fn=_collate)

    model.eval()
    all_z: list[np.ndarray] = []
    all_meta: list[FITSMeta] = []
    with torch.no_grad():
        for images, metas in tqdm(loader, desc="Encoding dataset", unit="batch", ncols=90):
            images = images.to(device, non_blocking=True)
            z: Tensor = model.encode(images)
            all_z.append(z.cpu().numpy())
            all_meta.extend(metas)

    latent_matrix = np.vstack(all_z).astype(np.float32)
    return latent_matrix, dataset.file_paths, all_meta

def reduce_dimensions(latent_matrix: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    if config.reducer == "umap":
        import umap  # type: ignore
        r = umap.UMAP(n_components=2, n_neighbors=config.umap_n_neighbors, min_dist=config.umap_min_dist, metric="euclidean", random_state=config.random_state)
    elif config.reducer == "tsne":
        from sklearn.manifold import TSNE
        r = TSNE(n_components=2, perplexity=config.tsne_perplexity, random_state=config.random_state, init="pca", learning_rate="auto")
    else:
        raise ValueError(f"reducer must be 'umap' or 'tsne', got '{config.reducer}'")
    return r.fit_transform(latent_matrix).astype(np.float32)

def cluster_latent_space(latent_matrix: np.ndarray, config: ClusteringConfig) -> tuple[np.ndarray, int]:
    if config.clusterer == "kmeans":
        km = KMeans(n_clusters=config.n_clusters, init="k-means++", n_init=10, random_state=config.random_state, max_iter=500)
        labels = km.fit_predict(latent_matrix)
        n_clusters = config.n_clusters
    elif config.clusterer == "dbscan":
        db = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples, metric="euclidean", n_jobs=-1)
        labels = db.fit_predict(latent_matrix)
        n_clusters = len(set(labels) - {-1})
    else:
        raise ValueError(f"clusterer must be 'kmeans' or 'dbscan', got '{config.clusterer}'")
    return labels.astype(np.int32), n_clusters

def compute_radial_profile(image: np.ndarray, meta: FITSMeta, n_bins: int=30) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    cy, cx = h / 2.0, w / 2.0
    y_idx, x_idx = np.indices(image.shape, dtype=np.float32)
    dx = (x_idx - cx) * meta.cdelt_arcsec if not np.isnan(meta.cdelt_arcsec) else (x_idx - cx)
    dy = (y_idx - cy) * meta.cdelt_arcsec if not np.isnan(meta.cdelt_arcsec) else (y_idx - cy)
    r = np.sqrt(dx**2 + dy**2)

    r_max = r.max()
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    radii = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    flux = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if mask.any():
            flux[i] = float(np.nanmean(image[mask]))
    return radii, flux

def _infer_shape_label(radii: np.ndarray, flux: np.ndarray) -> str:
    kernel = np.ones(3) / 3
    f = np.convolve(flux, kernel, mode="same")
    inner = np.sum(f[:len(f)//5])
    total = np.sum(f) + 1e-8
    inner_frac = inner / total

    from scipy.signal import argrelextrema
    maxima = argrelextrema(f, np.greater, order=2)[0]
    minima = argrelextrema(f, np.less, order=2)[0]

    if inner_frac > 0.6: return _SHAPE_LABELS["compact"]
    elif len(maxima) >= 2 and len(minima) >= 1: return _SHAPE_LABELS["complex"]
    elif len(minima) == 1 and len(maxima) == 2: return _SHAPE_LABELS["single_gap"]
    elif len(maxima) == 1 and maxima[0] > len(f)//4: return _SHAPE_LABELS["ring"]
    else: return _SHAPE_LABELS["extended"]

def plot_embedding(result: ClusteringResult, config: ClusteringConfig, save_path: Optional[str | Path]=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))
    emb = result.embedding_2d
    labels = result.labels
    unique = sorted(set(labels))
    handles: list[Patch] = []
    for lbl in unique:
        mask = labels == lbl
        is_noise = lbl == -1
        colour = _NOISE_COLOR if is_noise else _PALETTE[lbl % len(_PALETTE)]
        shape_tag = result.shape_labels.get(lbl, "")
        label_str = f"Noise  ({mask.sum()})" if is_noise else f"[{lbl}] {shape_tag}  ({mask.sum()})"
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[to_rgba(colour, alpha=0.18 if is_noise else 0.72)], edgecolors=[to_rgba(colour, alpha=0.35 if is_noise else 0.95)], linewidths=0.45, s=32 if is_noise else 44, zorder=2 if is_noise else 3, rasterized=True)
        handles.append(Patch(facecolor=colour, edgecolor=colour, label=label_str, alpha=0.85))

    algo_name = config.reducer.upper()
    ax.set_xlabel(f"{algo_name} 1", fontsize=11, labelpad=8)
    ax.set_ylabel(f"{algo_name} 2", fontsize=11, labelpad=8)
    ax.set_title(f"ALMA Disk Latent Space · {algo_name} · {config.clusterer.upper()} ({result.n_clusters} clusters)", fontsize=13, pad=16)
    ax.legend(handles=handles, loc="upper right", framealpha=0.2, edgecolor="#333355", fontsize=8.5)
    ax.grid(True, zorder=0)
    fig.tight_layout()
    if save_path: fig.savefig(save_path)
    return fig

def _load_display_image(path: Path, layer_index: int=0) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data.squeeze().astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 3: data = data[layer_index]
    data = np.clip(data, 0.0, None)
    k = 10.0 * (data.max() + 1e-8)
    stretched = np.arcsinh(data / k)
    lo, hi = stretched.min(), stretched.max()
    return (stretched - lo) / (hi - lo + 1e-8)

def plot_cluster_gallery(result: ClusteringResult, config: ClusteringConfig, save_path: Optional[str | Path]=None) -> plt.Figure:
    rng = np.random.default_rng(config.random_state)
    cluster_ids = sorted(lbl for lbl in set(result.labels) if lbl != -1)
    if not cluster_ids:
        warnings.warn("No non-noise clusters; gallery is empty.", stacklevel=2)
        return plt.figure()
    n_rows = len(cluster_ids)
    n_cols = config.samples_per_cluster
    fig = plt.figure(figsize=(4.0 * n_cols + 1.5, 3.8 * n_rows))
    outer = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.60)
    for row, cid in enumerate(cluster_ids):
        colour = _PALETTE[cid % len(_PALETTE)]
        mask_idx = np.where(result.labels == cid)[0]
        n_avail = len(mask_idx)
        n_draw = min(n_cols, n_avail)
        sampled = rng.choice(mask_idx, size=n_draw, replace=False)
        inner = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[row], wspace=0.05)
        for col in range(n_cols):
            ax = fig.add_subplot(inner[col])
            ax.set_facecolor("#0a0a12")
            if col < n_draw:
                fp = result.file_paths[sampled[col]]
                try:
                    img = _load_display_image(fp, layer_index=config.layer_start)
                    ax.imshow(img, cmap="afmhot", origin="lower", aspect="equal")
                    ax.set_title(fp.name, fontsize=6.0, color="#aaaacc", pad=3)
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Error\n{exc}", ha="center", va="center", transform=ax.transAxes, fontsize=6, color="#ff7043")
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes, fontsize=16, color="#333355")
            for sp in ax.spines.values():
                sp.set_edgecolor(colour)
                sp.set_linewidth(1.4)
            ax.set_xticks([]); ax.set_yticks([])
        shape = result.shape_labels.get(cid, "")
        fig.text(0.01, outer[row].get_position(fig).y0 + outer[row].get_position(fig).height / 2, f"[{cid}]\n{shape}\n({n_avail})", ha="left", va="center", fontsize=8, color=colour, rotation=90)
    fig.suptitle("Cluster Gallery — ALMA Protoplanetary Disks  (arcsinh stretch, afmhot)", fontsize=12, y=1.01)
    if save_path: fig.savefig(save_path, bbox_inches="tight")
    return fig

def plot_cluster_radial_profiles(result: ClusteringResult, config: ClusteringConfig, save_path: Optional[str | Path]=None, max_samples: int=10) -> plt.Figure:
    cluster_ids = sorted(lbl for lbl in set(result.labels) if lbl != -1)
    n_cols = min(3, len(cluster_ids))
    n_rows = int(np.ceil(len(cluster_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False)
    rng = np.random.default_rng(config.random_state)
    for idx, cid in enumerate(cluster_ids):
        ax = axes[idx // n_cols][idx % n_cols]
        colour = _PALETTE[cid % len(_PALETTE)]
        mask_idx = np.where(result.labels == cid)[0]
        sampled = rng.choice(mask_idx, size=min(max_samples, len(mask_idx)), replace=False)
        all_flux: list[np.ndarray] = []
        radii_ref: Optional[np.ndarray] = None
        for si in sampled:
            fp = result.file_paths[si]
            meta = result.meta_list[si]
            try:
                img = _load_display_image(fp, layer_index=config.layer_start)
                radii, flux = compute_radial_profile(img, meta, n_bins=config.radial_bins)
                if radii_ref is None: radii_ref = radii
                all_flux.append(flux)
                ax.plot(radii, flux, color=colour, alpha=0.15, linewidth=0.9)
            except Exception: pass
        if all_flux and radii_ref is not None:
            median_flux = np.median(np.stack(all_flux, axis=0), axis=0)
            ax.plot(radii_ref, median_flux, color=colour, linewidth=2.5, label="Median profile", zorder=5)
            from scipy.signal import argrelextrema
            mins = argrelextrema(median_flux, np.less, order=3)[0]
            maxs = argrelextrema(median_flux, np.greater, order=3)[0]
            for m in mins:
                ax.axvline(radii_ref[m], color="#ffde03", linewidth=1.2, linestyle="--", alpha=0.75)
                ax.text(radii_ref[m], ax.get_ylim()[1]*0.95, "gap", fontsize=6.5, color="#ffde03", ha="center", va="top")
            for m in maxs:
                ax.axvline(radii_ref[m], color="#69ff47", linewidth=1.0, linestyle=":", alpha=0.65)
        shape = result.shape_labels.get(cid, "")
        ax.set_title(f"Cluster {cid} · {shape}", fontsize=9, color=colour, pad=6)
        ax.set_xlabel("Radius (arcsec)", fontsize=8)
        ax.set_ylabel("Mean intensity (a.u.)", fontsize=8)
        ax.grid(True, alpha=0.4)
    for idx in range(len(cluster_ids), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    fig.suptitle("Azimuthally Averaged Radial Profiles per Cluster\n(dashed = gap candidate, dotted = ring peak)", fontsize=12, y=1.01)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, bbox_inches="tight")
    return fig

def run_pipeline(model_checkpoint: str | Path, fits_dir: str | Path, config: Optional[ClusteringConfig]=None, device: Optional[torch.device]=None) -> ClusteringResult:
    if config is None: config = ClusteringConfig()
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(model_checkpoint, device=device)
    raw_latents, file_paths, meta_list = extract_latent_vectors(model, fits_dir, config, device)
    scaler = StandardScaler()
    latent_matrix = scaler.fit_transform(raw_latents)
    embedding = reduce_dimensions(latent_matrix, config)
    labels, n_clusters = cluster_latent_space(latent_matrix, config)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(l): int(c) for l, c in zip(unique_labels, counts)}
    result = ClusteringResult(latent_vectors=latent_matrix, embedding_2d=embedding, labels=labels, file_paths=file_paths, meta_list=meta_list, n_clusters=n_clusters, cluster_sizes=cluster_sizes)

    rng = np.random.default_rng(config.random_state)
    for cid in sorted(set(labels) - {-1}):
        mask_idx = np.where(labels == cid)[0]
        sampled = rng.choice(mask_idx, size=min(5, len(mask_idx)), replace=False)
        all_flux = []
        radii_ref = None
        for si in sampled:
            try:
                img = _load_display_image(file_paths[si], layer_index=config.layer_start)
                radii, flux = compute_radial_profile(img, meta_list[si], n_bins=config.radial_bins)
                all_flux.append(flux)
                radii_ref = radii
            except Exception: pass
        if all_flux and radii_ref is not None:
            median = np.median(np.stack(all_flux), axis=0)
            result.shape_labels[cid] = _infer_shape_label(radii_ref, median)
        else:
            result.shape_labels[cid] = "unknown"

    plot_embedding(result, config, save_path=out / f"embedding_{config.reducer}_{config.clusterer}.png")
    plot_cluster_gallery(result, config, save_path=out / f"gallery_{config.reducer}_{config.clusterer}.png")
    plot_cluster_radial_profiles(result, config, save_path=out / f"radial_profiles_{config.clusterer}.png")
    
    plt.show()
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fits-dir", required=True)
    p.add_argument("--reducer", default="umap", choices=["umap", "tsne"])
    p.add_argument("--clusterer", default="kmeans", choices=["kmeans", "dbscan"])
    p.add_argument("--n-clusters", type=int, default=6)
    p.add_argument("--preprocess", default="log", choices=["none", "log", "sqrt", "sinh", "log10", "log_sqrt"])
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    cfg = ClusteringConfig(
        reducer=args.reducer, clusterer=args.clusterer, n_clusters=args.n_clusters,
        preprocess=args.preprocess, samples_per_cluster=args.samples, output_dir=args.output_dir,
    )
    run_pipeline(args.checkpoint, args.fits_dir, config=cfg)

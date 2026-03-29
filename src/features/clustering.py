"""
clustering.py
-------------
Unsupervised clustering pipeline for protoplanetary disk images using
the latent space of a trained ConvAutoencoder.

Pipeline
~~~~~~~~
1. Encode  — pass the full dataset through ``model.encode()`` to build
             an (N, latent_dim) matrix of 1-D feature vectors.
2. Reduce  — UMAP (preferred) or t-SNE projects latent vectors → 2-D
             for visualisation.
3. Cluster — K-Means or DBSCAN on the *original* latent vectors
             (not the 2-D projection) to avoid information loss.
4. Plot    — scatter plot of the 2-D projection, colour-coded by cluster.
5. Gallery — random sample of raw FITS images from each cluster so a
             human expert can inspect physical features (gaps, rings,
             spiral arms) that drove the grouping.

Dependencies
~~~~~~~~~~~~
    pip install torch torchvision astropy numpy matplotlib scikit-learn umap-learn tqdm

Usage
~~~~~
    from clustering import run_pipeline

    run_pipeline(
        model_checkpoint="checkpoints/best.pth",
        fits_dir="data/fits",
        n_clusters=6,
        reducer="umap",
        clusterer="kmeans",
        samples_per_cluster=3,
        output_dir="results",
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Local imports — produced by the previous pipeline steps
from autoencoder import ConvAutoencoder, load_checkpoint
from fits_dataset import FITSDataset

__all__ = [
    "ClusteringConfig",
    "ClusteringResult",
    "extract_latent_vectors",
    "reduce_dimensions",
    "cluster_latent_space",
    "plot_embedding",
    "plot_cluster_gallery",
    "run_pipeline",
]

logger = logging.getLogger(__name__)

# ── Matplotlib global style ──────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "#0d0d0d",
        "axes.facecolor": "#0d0d0d",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#cccccc",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
        "text.color": "#cccccc",
        "grid.color": "#222222",
        "grid.linewidth": 0.5,
        "font.family": "monospace",
        "savefig.facecolor": "#0d0d0d",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    }
)

# Perceptually distinct, colourblind-friendly cluster palette
_CLUSTER_PALETTE: list[str] = [
    "#00d4ff",  # cyan
    "#ff6b35",  # orange
    "#7fff6b",  # lime
    "#d46bff",  # violet
    "#ffd700",  # gold
    "#ff6b8a",  # rose
    "#6bffd4",  # teal
    "#ffaa6b",  # peach
    "#aaaaff",  # lavender
    "#ff4444",  # red
    "#44ff44",  # green
    "#4444ff",  # blue
]
_NOISE_COLOUR: str = "#555555"  # DBSCAN noise points


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class ClusteringConfig:
    """Full configuration for one clustering run.

    Parameters
    ----------
    reducer:
        Dimensionality reduction algorithm.  ``"umap"`` (default) gives
        better global topology preservation; ``"tsne"`` emphasises local
        neighbourhood structure.
    clusterer:
        Clustering algorithm.  ``"kmeans"`` requires ``n_clusters``;
        ``"dbscan"`` discovers the number of clusters automatically via
        density.
    n_clusters:
        Number of K-Means clusters.  Ignored for DBSCAN.
    dbscan_eps:
        DBSCAN neighbourhood radius in latent space (after StandardScaler
        normalisation).  Tune based on a k-nearest-neighbour distance
        plot.
    dbscan_min_samples:
        Minimum cluster size for DBSCAN.
    umap_n_neighbors:
        UMAP ``n_neighbors`` — controls local vs. global structure trade-off.
        Larger values → more global; smaller → more local.
    umap_min_dist:
        UMAP ``min_dist`` — controls how tightly points are packed.
        Smaller → tighter clusters.
    tsne_perplexity:
        t-SNE perplexity.  Roughly the expected number of nearest
        neighbours; typically 5–50.
    random_state:
        Seed for reproducible UMAP / t-SNE / K-Means results.
    batch_size:
        Batch size used when encoding the dataset.  Does not affect
        results; tune for available GPU memory.
    num_workers:
        DataLoader workers for the encoding pass.
    samples_per_cluster:
        How many random FITS images to display per cluster in the gallery.
    output_dir:
        Directory where all generated figures are written.
    layer_index:
        FITS cube layer to load (must match what the model was trained on).
    """

    reducer: Literal["umap", "tsne"] = "umap"
    clusterer: Literal["kmeans", "dbscan"] = "kmeans"
    n_clusters: int = 6
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    tsne_perplexity: float = 30.0
    random_state: int = 42
    batch_size: int = 16
    num_workers: int = 4
    samples_per_cluster: int = 3
    output_dir: str | Path = Path("results")
    layer_index: int = 0


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class ClusteringResult:
    """All outputs produced by the clustering pipeline.

    Attributes
    ----------
    latent_vectors : np.ndarray, shape (N, latent_dim)
        Raw (StandardScaler-normalised) latent vectors.
    embedding_2d : np.ndarray, shape (N, 2)
        2-D projection (UMAP or t-SNE coordinates).
    labels : np.ndarray, shape (N,)
        Integer cluster assignment per sample.  DBSCAN uses ``-1``
        for noise points.
    file_paths : list[Path]
        Ordered list of FITS file paths corresponding to each row.
    n_clusters : int
        Effective number of clusters (excluding DBSCAN noise).
    cluster_sizes : dict[int, int]
        Mapping ``{cluster_id: sample_count}``.
    """

    latent_vectors: np.ndarray
    embedding_2d: np.ndarray
    labels: np.ndarray
    file_paths: list[Path]
    n_clusters: int
    cluster_sizes: dict[int, int] = field(default_factory=dict)


# ── Step 1 — Latent extraction ────────────────────────────────────────────────


def extract_latent_vectors(
    model: ConvAutoencoder,
    fits_dir: str | Path,
    config: ClusteringConfig,
    device: torch.device,
) -> tuple[np.ndarray, list[Path]]:
    """Encode the entire dataset and return the latent matrix.

    The model is set to ``eval()`` mode and gradients are disabled for
    the full pass.

    Parameters
    ----------
    model:
        A trained ``ConvAutoencoder``.
    fits_dir:
        Directory of ``.fits`` files (same one used during training).
    config:
        ``ClusteringConfig`` instance.
    device:
        Torch device.

    Returns
    -------
    latent_matrix : np.ndarray, shape (N, latent_dim)
        Stacked latent vectors (float32, CPU numpy).
    file_paths : list[Path]
        Ordered list of FITS paths corresponding to each row.
    """
    dataset = FITSDataset(
        fits_dir=fits_dir,
        transform=None,          # No augmentation during inference
        layer_index=config.layer_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,           # Must stay ordered to match file_paths
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    all_vectors: list[np.ndarray] = []

    with torch.no_grad():
        bar = tqdm(loader, desc="Encoding dataset", unit="batch", ncols=90)
        for batch in bar:
            images: Tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            z: Tensor = model.encode(images)                 # (B, latent_dim)
            all_vectors.append(z.cpu().numpy())

    latent_matrix = np.vstack(all_vectors).astype(np.float32)  # (N, latent_dim)
    logger.info(
        "Extracted latent matrix: shape=%s, dtype=%s",
        latent_matrix.shape, latent_matrix.dtype,
    )
    return latent_matrix, dataset.file_paths


# ── Step 2 — Dimensionality reduction ────────────────────────────────────────


def reduce_dimensions(
    latent_matrix: np.ndarray,
    config: ClusteringConfig,
) -> np.ndarray:
    """Project latent vectors to 2-D for visualisation.

    Clustering (Step 3) always runs on the *original* ``latent_matrix``,
    not on this 2-D projection, to avoid information loss.

    Parameters
    ----------
    latent_matrix : np.ndarray, shape (N, latent_dim)
        StandardScaler-normalised latent vectors.
    config:
        ``ClusteringConfig`` instance.

    Returns
    -------
    embedding : np.ndarray, shape (N, 2)
        2-D coordinates.
    """
    algo = config.reducer.lower()

    if algo == "umap":
        try:
            import umap  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "UMAP is not installed.  Run:  pip install umap-learn"
            ) from exc

        logger.info(
            "Running UMAP  (n_neighbors=%d, min_dist=%.2f) ...",
            config.umap_n_neighbors, config.umap_min_dist,
        )
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            metric="euclidean",
            random_state=config.random_state,
            verbose=False,
        )

    elif algo == "tsne":
        from sklearn.manifold import TSNE  # noqa: PLC0415

        logger.info(
            "Running t-SNE  (perplexity=%.1f) ...", config.tsne_perplexity
        )
        reducer = TSNE(
            n_components=2,
            perplexity=config.tsne_perplexity,
            random_state=config.random_state,
            init="pca",
            learning_rate="auto",
            n_iter=1000,
        )

    else:
        raise ValueError(f"reducer must be 'umap' or 'tsne', got '{config.reducer}'.")

    embedding: np.ndarray = reducer.fit_transform(latent_matrix)
    logger.info("2-D embedding shape: %s", embedding.shape)
    return embedding


# ── Step 3 — Clustering ───────────────────────────────────────────────────────


def cluster_latent_space(
    latent_matrix: np.ndarray,
    config: ClusteringConfig,
) -> tuple[np.ndarray, int]:
    """Assign cluster labels to every sample.

    Clustering runs on the full-dimensional *normalised* latent vectors,
    not on the 2-D projection, to maximise discriminative power.

    Parameters
    ----------
    latent_matrix : np.ndarray, shape (N, latent_dim)
        StandardScaler-normalised latent vectors.
    config:
        ``ClusteringConfig`` instance.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Integer cluster IDs.  DBSCAN uses ``-1`` for noise points.
    n_clusters : int
        Number of discovered clusters (excludes DBSCAN noise class).
    """
    algo = config.clusterer.lower()

    if algo == "kmeans":
        logger.info("Running K-Means  (k=%d) ...", config.n_clusters)
        km = KMeans(
            n_clusters=config.n_clusters,
            init="k-means++",
            n_init=10,
            random_state=config.random_state,
            max_iter=500,
        )
        labels: np.ndarray = km.fit_predict(latent_matrix)
        n_clusters = config.n_clusters

    elif algo == "dbscan":
        logger.info(
            "Running DBSCAN  (eps=%.3f, min_samples=%d) ...",
            config.dbscan_eps, config.dbscan_min_samples,
        )
        db = DBSCAN(
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples,
            metric="euclidean",
            n_jobs=-1,
        )
        labels = db.fit_predict(latent_matrix)
        unique = set(labels) - {-1}
        n_clusters = len(unique)
        noise_count = np.sum(labels == -1)
        logger.info(
            "DBSCAN found %d clusters + %d noise points.", n_clusters, noise_count
        )

    else:
        raise ValueError(f"clusterer must be 'kmeans' or 'dbscan', got '{config.clusterer}'.")

    # Log cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        logger.info("  %-12s : %4d samples", tag, cnt)

    return labels, n_clusters


# ── Step 4 — Scatter plot ─────────────────────────────────────────────────────


def plot_embedding(
    result: ClusteringResult,
    config: ClusteringConfig,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Render the 2-D embedding scatter plot, colour-coded by cluster.

    Parameters
    ----------
    result:
        Populated ``ClusteringResult`` from ``cluster_latent_space``.
    config:
        ``ClusteringConfig`` instance (used for axis labels / title).
    save_path:
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    labels = result.labels
    emb = result.embedding_2d
    unique_labels = sorted(set(labels))

    legend_handles: list[Patch] = []

    for lbl in unique_labels:
        mask = labels == lbl
        is_noise = lbl == -1

        colour = _NOISE_COLOUR if is_noise else _CLUSTER_PALETTE[lbl % len(_CLUSTER_PALETTE)]
        rgba = to_rgba(colour, alpha=0.15 if is_noise else 0.75)
        edge = to_rgba(colour, alpha=0.3 if is_noise else 0.95)

        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[rgba],
            edgecolors=[edge],
            linewidths=0.4,
            s=28 if is_noise else 38,
            zorder=2 if is_noise else 3,
            rasterized=True,
        )

        label_text = (
            f"Noise  ({mask.sum()})"
            if is_noise
            else f"Cluster {lbl}  ({mask.sum()})"
        )
        legend_handles.append(
            Patch(facecolor=colour, edgecolor=edge, label=label_text, alpha=0.85)
        )

    reducer_name = config.reducer.upper()
    ax.set_xlabel(f"{reducer_name} dimension 1", fontsize=11, labelpad=8)
    ax.set_ylabel(f"{reducer_name} dimension 2", fontsize=11, labelpad=8)
    ax.set_title(
        f"Protoplanetary Disk Latent Space  ·  {reducer_name}  ·  "
        f"{config.clusterer.upper()}  ({result.n_clusters} clusters)",
        fontsize=13, pad=14, color="#e0e0e0",
    )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.25,
        edgecolor="#444444",
        fontsize=9,
    )
    ax.grid(True, zorder=0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("Embedding plot saved → %s", save_path)

    return fig


# ── Step 5 — Cluster gallery ──────────────────────────────────────────────────


def _load_raw_fits(path: Path, layer_index: int = 0) -> np.ndarray:
    """Load and normalise a single FITS layer for display.

    Returns a (H, W) float32 array in [0, 1] via Min-Max scaling.
    NaN / Inf values are replaced with 0.

    Parameters
    ----------
    path:
        Absolute path to the ``.fits`` file.
    layer_index:
        Cube layer to extract.
    """
    with fits.open(path, memmap=False) as hdul:
        data: np.ndarray = hdul[0].data[layer_index].astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = data.min(), data.max()
    return (data - lo) / (hi - lo + 1e-8)


def plot_cluster_gallery(
    result: ClusteringResult,
    config: ClusteringConfig,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Display randomly sampled FITS images from each cluster.

    Layout: one row per cluster, ``samples_per_cluster`` columns.
    Each image is displayed with a reverse-greyscale colourmap and
    a mild logarithmic stretch to reveal faint disk structures.

    Parameters
    ----------
    result:
        Populated ``ClusteringResult``.
    config:
        ``ClusteringConfig`` instance.
    save_path:
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : plt.Figure
    """
    rng = np.random.default_rng(seed=config.random_state)

    # Collect (cluster_id, [sampled_paths]) pairs — skip noise
    cluster_ids = sorted(lbl for lbl in set(result.labels) if lbl != -1)
    n_rows = len(cluster_ids)
    n_cols = config.samples_per_cluster

    if n_rows == 0:
        warnings.warn("No non-noise clusters found; gallery is empty.", stacklevel=2)
        return plt.figure()

    # ── Figure layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(4.2 * n_cols, 3.8 * n_rows))

    # Outer grid: one row per cluster
    outer = gridspec.GridSpec(
        n_rows, 1,
        figure=fig,
        hspace=0.55,
    )

    for row_idx, cid in enumerate(cluster_ids):
        colour = _CLUSTER_PALETTE[cid % len(_CLUSTER_PALETTE)]
        cluster_mask = np.where(result.labels == cid)[0]
        n_available = len(cluster_mask)
        n_sample = min(n_cols, n_available)

        sampled_indices = rng.choice(cluster_mask, size=n_sample, replace=False)

        # Inner grid: one column per sample
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols,
            subplot_spec=outer[row_idx],
            wspace=0.06,
        )

        for col_idx in range(n_cols):
            ax = fig.add_subplot(inner[col_idx])
            ax.set_facecolor("#0d0d0d")

            if col_idx < n_sample:
                fp = result.file_paths[sampled_indices[col_idx]]
                try:
                    img = _load_raw_fits(fp, layer_index=config.layer_index)
                    # Logarithmic stretch: log(1 + k·x) / log(1 + k)
                    k = 500.0
                    img = np.log1p(k * img) / np.log1p(k)
                    ax.imshow(img, cmap="afmhot", origin="lower", aspect="equal")
                    ax.set_title(fp.name, fontsize=6.5, color="#aaaaaa", pad=3)
                except Exception as exc:  # noqa: BLE001
                    ax.text(
                        0.5, 0.5, f"Load error\n{exc}",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=7, color="#ff6b35",
                    )
            else:
                # Placeholder for clusters with fewer samples than n_cols
                ax.text(
                    0.5, 0.5, "—",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=16, color="#444444",
                )

            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.2)
            ax.set_xticks([])
            ax.set_yticks([])

        # Row label (left of the row)
        fig.text(
            0.01,
            outer[row_idx].get_position(fig).y0
            + outer[row_idx].get_position(fig).height / 2,
            f"Cluster {cid}\n({n_available} imgs)",
            ha="left", va="center",
            fontsize=9, color=colour,
            rotation=90,
        )

    fig.suptitle(
        "Cluster Gallery — Random Sample of Protoplanetary Disk Images",
        fontsize=13, y=1.01, color="#e0e0e0",
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Gallery plot saved → %s", save_path)

    return fig


# ── Full pipeline entry-point ─────────────────────────────────────────────────


def run_pipeline(
    model_checkpoint: str | Path,
    fits_dir: str | Path,
    config: Optional[ClusteringConfig] = None,
    device: Optional[torch.device] = None,
) -> ClusteringResult:
    """Execute the complete encode → reduce → cluster → visualise pipeline.

    Parameters
    ----------
    model_checkpoint:
        Path to a ``.pth`` checkpoint produced by ``save_checkpoint``.
    fits_dir:
        Directory containing the ``.fits`` files to analyse.
    config:
        ``ClusteringConfig`` instance.  Defaults are used if ``None``.
    device:
        Torch device.  Auto-detected if ``None``.

    Returns
    -------
    ClusteringResult
        All intermediate and final results for downstream inspection.

    Examples
    --------
    >>> result = run_pipeline(
    ...     model_checkpoint="checkpoints/best.pth",
    ...     fits_dir="data/fits",
    ...     config=ClusteringConfig(n_clusters=6, reducer="umap"),
    ... )
    >>> result.n_clusters
    6
    """
    if config is None:
        config = ClusteringConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Clustering pipeline — device: %s", device)
    logger.info("Reducer: %s  |  Clusterer: %s", config.reducer, config.clusterer)
    logger.info("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────────
    model, _ = load_checkpoint(model_checkpoint, device=device)

    # ── 2. Extract latent vectors ────────────────────────────────────
    raw_latents, file_paths = extract_latent_vectors(model, fits_dir, config, device)

    # StandardScaler normalisation: zero-mean, unit-variance per dimension.
    # Prevents high-variance latent dimensions from dominating the distance
    # metric in both the clustering and UMAP/t-SNE steps.
    scaler = StandardScaler()
    latent_matrix: np.ndarray = scaler.fit_transform(raw_latents)
    logger.info("Latent matrix normalised (StandardScaler).")

    # ── 3. Dimensionality reduction ──────────────────────────────────
    embedding_2d = reduce_dimensions(latent_matrix, config)

    # ── 4. Clustering ────────────────────────────────────────────────
    labels, n_clusters = cluster_latent_space(latent_matrix, config)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}

    result = ClusteringResult(
        latent_vectors=latent_matrix,
        embedding_2d=embedding_2d,
        labels=labels,
        file_paths=file_paths,
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
    )

    # ── 5. Scatter plot ──────────────────────────────────────────────
    embed_path = output_dir / f"embedding_{config.reducer}_{config.clusterer}.png"
    plot_embedding(result, config, save_path=embed_path)

    # ── 6. Gallery ───────────────────────────────────────────────────
    gallery_path = output_dir / f"gallery_{config.reducer}_{config.clusterer}.png"
    plot_cluster_gallery(result, config, save_path=gallery_path)

    plt.show()

    logger.info("Pipeline complete.  Results written to '%s'.", output_dir)
    return result


# ── Optional: elbow / DBSCAN knee helpers ────────────────────────────────────


def plot_kmeans_elbow(
    latent_matrix: np.ndarray,
    k_range: range = range(2, 15),
    random_state: int = 42,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the K-Means inertia curve to help choose ``n_clusters``.

    The "elbow" — the value of k where the inertia drop rate markedly
    decreases — is a common heuristic for selecting the number of clusters.

    Parameters
    ----------
    latent_matrix:
        StandardScaler-normalised latent vectors.
    k_range:
        Range of k values to evaluate.
    random_state:
        Random seed for reproducibility.
    save_path:
        If provided, the figure is saved to this path.
    """
    inertias: list[float] = []
    for k in tqdm(k_range, desc="K-Means elbow", unit="k"):
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=random_state)
        km.fit(latent_matrix)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, "o-", color="#00d4ff", linewidth=2, markersize=7)
    ax.set_xlabel("Number of clusters  k", fontsize=11)
    ax.set_ylabel("Inertia  (within-cluster SSE)", fontsize=11)
    ax.set_title("K-Means Elbow Curve", fontsize=13, pad=12)
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("Elbow plot saved → %s", save_path)

    return fig


def plot_dbscan_knee(
    latent_matrix: np.ndarray,
    k: int = 5,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the k-nearest-neighbour distance curve to guide ``dbscan_eps``.

    Sort samples by their distance to the k-th nearest neighbour.
    The "knee" (sharp upward bend) indicates a sensible ``eps`` value:
    points to the right of the knee are likely outliers.

    Parameters
    ----------
    latent_matrix:
        StandardScaler-normalised latent vectors.
    k:
        Which nearest-neighbour distance to use (should match
        ``dbscan_min_samples``).
    save_path:
        If provided, the figure is saved to this path.
    """
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nbrs.fit(latent_matrix)
    distances, _ = nbrs.kneighbors(latent_matrix)
    kth_distances = np.sort(distances[:, -1])[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kth_distances, color="#ff6b35", linewidth=2)
    ax.set_xlabel("Samples (sorted by distance)", fontsize=11)
    ax.set_ylabel(f"Distance to {k}-th nearest neighbour", fontsize=11)
    ax.set_title(
        f"DBSCAN k-NN Distance Curve  (k={k})\n"
        "Set  eps  at the knee of this curve",
        fontsize=12, pad=12,
    )
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        logger.info("DBSCAN knee plot saved → %s", save_path)

    return fig


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Cluster protoplanetary disk images via autoencoder latent space."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--fits-dir",   required=True, help="Directory of .fits files")
    parser.add_argument("--reducer",    default="umap",   choices=["umap", "tsne"])
    parser.add_argument("--clusterer",  default="kmeans", choices=["kmeans", "dbscan"])
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--samples",    type=int, default=3)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--elbow",      action="store_true",
                        help="Plot K-Means elbow curve before clustering.")
    args = parser.parse_args()

    cfg = ClusteringConfig(
        reducer=args.reducer,
        clusterer=args.clusterer,
        n_clusters=args.n_clusters,
        samples_per_cluster=args.samples,
        output_dir=args.output_dir,
    )

    if args.elbow:
        _model, _ = load_checkpoint(args.checkpoint)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _latents, _ = extract_latent_vectors(_model, args.fits_dir, cfg, _device)
        _scaled = StandardScaler().fit_transform(_latents)
        plot_kmeans_elbow(_scaled, save_path=Path(args.output_dir) / "elbow.png")
        plt.show()
    else:
        run_pipeline(
            model_checkpoint=args.checkpoint,
            fits_dir=args.fits_dir,
            config=cfg,
        )

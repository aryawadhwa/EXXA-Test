<div align="center">
  <h1>EXXA: Denoising Protoplanetary Disks</h1>
  <p><i>Official Implementation for the ML4SCI GSoC 2026 Test Tasks</i></p>

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
  [![ALMA](https://img.shields.io/badge/ALMA-Observatory-orange.svg)]()
  [![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
</div>

---

## Overview
This repository contains the complete execution deliverables for the **ML4SCI EXXA** program. It provides robust PyTorch-based machine learning pipelines designed to process, cluster, and denoise interferometric representations of protoplanetary disks.

The codebase satisfies the three mandatory GSoC EXXA test criteria:
1. **General Test:** Unsupervised algorithmic clustering of ALMA FITS observations via continuous Latent Manifolds.
2. **Image-Based Test:** Fast deterministic MS-SSIM Autoencoder capable of structural reconstruction.
3. **Sequential Test:** A 1D-CNN temporal classifier trained to isolate periodic exoplanetary transit anomalies from astronomical light curves.

---

## Scientific Validations

### 1. Unsupervised Disk Clustering (UMAP to K-Means)
The pipeline intrinsically separates physical protoplanetary morphologies without relying on human labels.
<p align="center">
  <img src="assets/results/umap_projection.png" width="45%" alt="UMAP Projection"/>
  <img src="assets/results/cluster_gallery.png" width="45%" alt="Cluster Gallery"/>
</p>

### 2. Autoencoder Objective Convergence (MSE and MS-SSIM)
By omitting Max Pooling in the convolutional backbone, the MS-SSIM structural loss optimizes planet radii fidelity.
<p align="center">
  <img src="assets/results/loss_curves.png" width="45%" alt="Loss Curves"/>
  <img src="assets/results/silhouette_score.png" width="45%" alt="Silhouette Analysis"/>
</p>

---

## Professional Repository Structure
The codebase has been refactored for reproducibility, transitioning raw scripts into a highly modular engineering standard.

```text
/EXXA-Test
├── README.md               # Pipeline documentation and metrics
├── requirements.txt        # Deep learning dependencies
├── assets/
│   └── results/            # Validation plots & loss curves
├── notebooks/              
│   ├── EXXA_ALMA_Autoencoder.ipynb  # General & Image-based tests run natively
│   └── EXXA_Sequential_Test.ipynb   # 1D-CNN Transit curve test logic
└── src/                    # Modularized backend abstractions
    ├── data/               # FITS iterators & PyTorch DataLoader logic
    ├── features/           # Unsupervised UMAP/K-Means dimensionality algorithms
    └── models/             # PyTorch Neural Architectures
```

---

## Execution & Reproducibility

To recreate the test executions, replicate the environment metrics:

```bash
# 1. Clone the repository
git clone https://gitlab.com/aryawadhwa/EXXA-Test.git
cd EXXA-Test

# 2. Install PyTorch dependencies
pip install -r requirements.txt

# 3. Execute the full inference notebooks (GPU Support Included)
jupyter notebook notebooks/EXXA_ALMA_Autoencoder.ipynb
```

## Technologies Built Upon
Astropy, PyTorch, Scikit-Learn (UMAP), Matplotlib, AdamW Optimization.

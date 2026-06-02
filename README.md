# SSL + ViT Cell Phenotyping

This project turns the draft SSL/ViT phenotyping idea into runnable code.
It augments existing CellProfiler/Brieflow-style per-cell phenotype tables with
self-supervised Vision Transformer patch-token embeddings pooled inside each
segmented cell mask.

The implementation follows the workflow described in the notes:

1. Load multichannel phenotype image data and segmentation masks.
2. Load a ViT backbone from a DINO/self-supervised checkpoint.
3. Extract spatial patch tokens instead of only a global CLS token.
4. Downsample cell labels to the patch-token grid.
5. Pool tokens per cell label with mean or mean+std pooling.
6. Optionally apply a global PCA transform.
7. Merge SSL features into the phenotype table by `label`.

The StainAI paper motivates the multi-stage pattern: detection/segmentation,
cell-level feature extraction, morphology classification, and region-level
quantification. This project focuses on the learned feature extraction stage so
it can plug into an existing segmentation and phenotype pipeline.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Snakemake Entry Point

Use `workflow/scripts/phenotype/extract_phenotype_cp_multichannel_ssl.py` in the
same place as the existing CellProfiler phenotype extraction script.

Important params:

```python
params:
    cp_method="cp_multichannel",
    channel_names=["nuclei", "marker"],
    foci_channel_index=1,
    ssl_enable=True,
    ssl_ckpt="outputs/vit_models/ssl_pretraining/dino_best.pth",
    ssl_model_builder="manuscript.models.vit:build_vit_backbone_tokens",
    ssl_device="cuda",
    ssl_patch_size=8,
    ssl_use_channels=[0, 1],
    ssl_pooling="mean",
    ssl_normalization="zscore",
    ssl_pca_dim=None,
    ssl_pca_basis_path=None,
```

`ssl_model_builder` must be a `module:function` path that returns a model
exposing patch tokens through one of these APIs:

- `forward_features(x)` returning `[B, N, D]`
- `forward_features(x)` returning a dict with `x_norm_patchtokens`,
  `patch_tokens`, or `tokens`
- `forward(x, return_tokens=True)` returning `[B, N, D]`

## PCA Guidance

Per-tile PCA is disabled. If dimensionality reduction is needed, fit a global
PCA basis once and pass the saved `.npz` path. This keeps features reproducible
across images and experiments.

## Smoke Test

```bash
pytest
```

The tests use a tiny ViT and synthetic segmentation labels, so they validate the
integration without requiring a trained checkpoint.

## ProCodes and sgRNA Analysis

This work is organized around ProCode comparison and on/off combinatorial
signaling in optical pooled screens. The analysis utilities in
`src/lib/phenotype/procode_analysis.py` implement the first reliability checks:

- ProCode channel thresholding into binary on/off signatures.
- Per-cell signal margin and crosstalk summaries for decoding quality.
- Segmentation quality metrics across cell-density conditions.
- sgRNA or perturbation separability with kNN and silhouette score.
- Classical morphology versus SSL embedding comparison.
- Replicate consistency of perturbation centroids.

The key scale-up gate is clean decoding. If ProCode signal separation is weak or
crosstalk is high, clustering and perturbation mapping can be skewed downstream.
Lower cell density can be benchmarked directly by comparing touching-cell edges,
oversized-mask fraction, undersegmentation proxy, and embedding separability.

See `docs/procodes_ssl_analysis.md` for the current analysis plan and expected
metadata columns.

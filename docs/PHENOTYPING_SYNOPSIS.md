# Phenotyping Synopsis For Lab Review

## One-Sentence Summary

The project adds patch-based morphology/intensity phenotyping and an SSL-ready
representation layer to Brieflow-style microscopy workflows, while preserving
segmentation and ProCode QC as upstream requirements.

## What Changed

The workflow now supports a small-patch test before whole-image scaling:

```text
Local microscopy images
        |
        v
100x100 or 200x200 foreground patch selection
        |
        v
Patch TIFF export + QC manifest
        |
        v
Interpretable patch phenotypes
        |
        v
Optional SSL embeddings when a trained checkpoint is supplied
        |
        v
QC report, Fiji macro, Jupyter review, and Streamlit web app
```

## Current Interpretable Phenotypes

These are available now without a trained SSL checkpoint:

| Feature family | What it captures | Why it matters |
| --- | --- | --- |
| Foreground fraction | How much of a patch contains biological signal | Flags empty/background-heavy patches |
| Foreground/background delta | Contrast between biological signal and background | Helps detect usable signal and staining separation |
| Connected components | Number of foreground objects/fragments | Flags clutter, puncta, debris, or fragmented neurites |
| Largest component area | Dominant foreground object size | Captures large cell/neurite regions |
| Elongation | Shape anisotropy of the largest foreground component | Useful for neurite-like or elongated morphology |
| Boundary fraction | How fragmented or edge-heavy foreground is | Flags noisy masks or complex morphology |
| Gradient mean | Local texture/edge content | Proxy for neurite texture and subcellular structure |
| Per-channel intensity | Mean/std and foreground mean per channel | Supports ProCode/channel QC |

## What SSL Adds Later

Classical patch metrics compress the image into predefined summaries. SSL/ViT
features can preserve richer local image patterns as patch tokens:

- neurite texture;
- subcellular localization;
- cell neighborhood context;
- subtle morphology not captured by handcrafted metrics;
- phenotype similarity useful for clustering and perturbation benchmarking.

SSL should be interpreted only when generated from a trained or deliberately
selected checkpoint. The current app skips checkpoint-free SSL embeddings by
default so random features are not mistaken for biology.

## Review Surfaces

| Surface | Best use |
| --- | --- |
| CLI runner | Reproducible batch analysis |
| Fiji macro | Native TIFF inspection and manual image review |
| Jupyter notebook | Pannable/zoomable plots and exploratory metric review |
| Streamlit app | Lab-accessible upload/folder analysis and output downloads |

## Streamlit App

Run:

```bash
streamlit run apps/ssl_patch_streamlit_app.py
```

For trusted lab-network access:

```bash
streamlit run apps/ssl_patch_streamlit_app.py --server.address 0.0.0.0
```

The app supports:

- local folder analysis;
- TIFF/PNG/JPEG upload;
- 100x100 or 200x200 patch extraction;
- zoomable image review;
- foreground overlay toggle;
- interactive phenotype scatterplots;
- QC report display;
- output ZIP download.

## Website Direction

The current Streamlit app is the best first lab-accessible route because it
keeps high-resolution files local and calls the same deterministic backend as
the CLI. A production website can come later with a React/Next.js frontend and a
Python FastAPI backend, but the scientific backend should remain shared with the
repo workflow.

Lovable can help later with a polished product-style interface, but it should
not replace the first local scientific implementation because upload handling,
TIFF fidelity, output reproducibility, and checkpoint-backed SSL logic need to
stay explicit and version-controlled.

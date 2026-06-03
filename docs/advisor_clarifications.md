# Advisor Clarifications

This document records the clarification points that should be easy to answer before the next lab review.

## Repository hygiene

`__pycache__/`, `.pytest_cache/`, and `*.egg-info/` should not be tracked. They are ignored in `.gitignore`, but if they were committed before `.gitignore` was created, remove them from Git's index while keeping local copies:

```bash
git rm -r --cached --ignore-unmatch __pycache__ .pytest_cache *.egg-info src/**/__pycache__ tests/**/__pycache__ workflow/**/__pycache__
git add .gitignore
git commit -m "Remove generated Python artifacts from tracking"
```

A helper script is included at `scripts/remove_tracked_artifacts.sh`.

## Preliminary embeddings

The repo can generate preliminary embeddings with either:

1. a real DINO/DINOv2/DINOv3-style checkpoint, or
2. the tiny synthetic ViT used for smoke tests.

For advisor review, preliminary embeddings should be generated on a small representative subset before scaling up:

- 1-2 plates or a balanced sample across plates,
- low- and high-density fields,
- several expected ProCode signatures,
- non-targeting controls and a few strong positive controls,
- one held-out plate/well split for leakage checks.

Recommended output folder:

```text
outputs/prelim_embeddings/
  phenotype_cp_ssl.tsv
  embedding_metadata.tsv
  feature_separability.tsv
  procode_qc.tsv
  benchmark_report.md
```

The extraction script records checkpoint and model provenance columns, including checkpoint path, checkpoint SHA256, model builder, device, patch size, pooling, normalization, selected channels, PCA dimension, and git commit.

## Current checkpoint status

The repository is model-agnostic. The default example points to:

```text
outputs/vit_models/ssl_pretraining/dino_best.pth
```

That path is intentionally an example. Before claiming biological results, replace it with the actual checkpoint used for extraction and record:

- checkpoint file name,
- training source: public foundation model, Ward-trained SSL model, or synthetic test model,
- training data domain,
- input channels,
- patch size,
- embedding dimension,
- whether weights were frozen,
- checkpoint SHA256.

## Why compare DINOv2, DINOv3, and other ViT models?

The starting assumption should not be that one model is automatically best. This repo should benchmark multiple SSL/foundation backbones under the same segmentation, ProCode, and phenotype QC gates.

Recommended model comparison table:

| Model family | Why include it | Role in this project |
| --- | --- | --- |
| DINOv2 | Stable, widely used self-supervised ViT baseline with strong transferable dense features. | Conservative baseline/comparator. |
| DINOv3 | Newer DINO-style model designed for stronger dense feature behavior and broader transfer. | Strong current candidate to test, not assume. |
| iBOT | Popular masked-image/self-distillation ViT SSL method. | Alternative SSL objective to test against DINO. |
| MAE | Masked autoencoder baseline. | Tests whether reconstruction-style SSL is useful for morphology. |
| MoCo v3 | Contrastive ViT SSL baseline. | Tests contrastive features against self-distillation. |
| Bio/pathology foundation ViTs such as UNI, Virchow, Phikon, or CTransPath | Domain-pretrained alternatives, depending on image modality and licensing. | Useful comparators if the data resemble histology/pathology or microscopy domains. |
| Ward-trained DINO-style ViT | Most domain-specific if enough unlabeled lab images are available. | Long-term best candidate after benchmarking. |

The planned benchmark should report classical morphology, SSL-only, and combined feature performance for every model using the same train/test split and the same ProCode/segmentation QC filters.

## Short advisor-facing answer

> I cleaned up the repo hygiene pieces by ignoring generated artifacts and adding a script to remove `__pycache__`, `.pytest_cache`, and `*.egg-info` from Git's index if they were already tracked. For embeddings, the code can now record checkpoint provenance directly in the output table, including checkpoint path, SHA256, model builder, git commit, patch size, pooling, normalization, and selected channels. I do not want to overclaim yet: the repo is ready to generate preliminary embeddings on a representative subset, but the actual checkpoint choice should be documented once we decide whether to use DINOv2, DINOv3, iBOT/MAE/MoCo v3, a biology/pathology foundation ViT, or a Ward-trained SSL checkpoint. The benchmarking CLI is set up so those models can be compared under the same segmentation, ProCode, and feature-separability gates.

## DINOv3 attempt

DINOv3 is now available as an optional model builder through
`manuscript.models.dinov3:build_dinov3_hf_backbone_tokens`. This does not make
DINOv3 the default. The intended answer to “why DINOv3?” is: DINOv3 is worth
benchmarking because its dense patch features are designed to remain useful for
spatial tasks, which matches the cell-mask pooling strategy. The intended caveat
is: public DINOv3 checkpoints are RGB natural-image models, so microscopy
channels are passed through an explicit channel adapter and must be evaluated
against DINOv2/local/Ward-trained checkpoints with the same QC gates.

# SSL + ViT Cell Phenotyping

This project turns the draft SSL/ViT phenotyping idea into runnable code.
It augments existing CellProfiler/Brieflow-style per-cell phenotype tables with
self-supervised Vision Transformer patch-token embeddings pooled inside each
segmented cell mask.

## Repository Description

Self-supervised ViT feature extraction and QC benchmarking for CellProfiler/Brieflow phenotyping and ProCode optical pooled screens.

## Scope: SSL Does Not Replace Segmentation

This repository now treats SSL/ViT features as a **downstream phenotype representation** layer, not as a direct replacement for segmentation. Cell, nuclei, and cytoplasm masks can come from Brieflow, CellPose, SAM-style workflows, manual annotation, or another segmentation source. The SSL step only becomes interpretable after the upstream masks and ProCode decoding pass QC.

What this project does:

- accepts existing masks from multiple segmentation sources;
- records mask provenance and segmentation/QC status in phenotype outputs;
- extracts SSL ViT patch-token features inside those masks;
- benchmarks SSL features against CellProfiler/classical morphology features;
- tests whether SSL helps perturbation/sgRNA separability without increasing batch leakage.

What this project does **not** claim:

- SSL embeddings alone replace CellPose, SAM, Brieflow, or manual segmentation;
- DINOv2/DINOv3 automatically outperform a microscopy-trained checkpoint;
- learned embeddings can compensate for poor masks, bad ProCode decoding, or unresolved channel/Z-stack artifacts.

For advisor-facing project framing, see `docs/ssl_scope.md` and `docs/mask_source_strategy.md`.

Suggested GitHub topics: `self-supervised-learning`, `vision-transformer`, `cellprofiler`, `brieflow`, `phenotyping`, `optical-pooled-screening`, `procode`, `sgrna`, `bioimage-analysis`, `neuroscience`.

## Pipeline Map

```text
Raw multichannel images / Z-stacks
        ↓
External mask source
(Brieflow, CellPose, SAM, manual labels, or other)
        ↓
Segmentation + mask provenance QC gate
        ↓
ProCode decoding / channel-unmixing QC gate
        ↓
Classical CellProfiler features, if available
        ↓
SSL ViT patch-token pooling inside accepted masks
        ↓
Classical vs SSL vs combined feature benchmark
        ↓
sgRNA / perturbation phenotype ranking
```

Core rule: SSL embeddings are evaluated as downstream phenotype features and should only be interpreted after segmentation, mask provenance, and ProCode decoding are trustworthy.

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

## Demo

A synthetic benchmark demo is included so contributors can run the benchmark without Ward Lab data:

```bash
bash demo/run_demo.sh
```

The demo writes a small phenotype table and benchmark outputs under `demo/synthetic_outputs/`.


## Future Direction and Data Contract

This project now has three explicit future-direction tracks:

1. **Representation-learning argument:** justify why SSL is worth testing against classical morphology and autoencoder-style embeddings.
2. **Subcell/Lundberg follow-up:** evaluate whether Human Protein Atlas / Subcell-style models, data structures, or weights can inform this project.
3. **Workflow stability:** maintain a documented data contract so coding agents do not silently break image, mask, metadata, or output assumptions.

Start with:

- `docs/future_project_direction.md`
- `docs/ssl_vs_autoencoders.md`
- `docs/subcell_lundberg_followup.md`
- `docs/data_contract.md`
- `docs/workflow_steps.md`
- `docs/agent_editing_contract.md`

Validate a dataset manifest with:

```bash
ssl-validate-data-contract --manifest data/ground_truth/manifest.template.csv
```

The manifest validator checks required columns, allowed mask-source names, segmentation QC status values, duplicated image IDs, split coverage, and missing channel metadata. It does not require raw files to live inside GitHub.

## Ground-Truth Data

Do not upload several-gigabyte ground-truth images, masks, or annotations directly to GitHub. Track only manifests, split files, schemas, and tiny synthetic examples in this repository. Store large files with DVC, Git LFS, institutional object storage, or a citable archive.

Start with:

- `docs/ground_truth_data.md`
- `data/ground_truth/manifest.template.csv`
- `data/ground_truth/README.md`

Use image-, plate-, well-, or replicate-level splits rather than random cell-level splits to avoid data leakage.

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

For the full segmentation and combinatorial ProCode workflow, see
`docs/procode_segmentation_workflow.md`.

After phenotype extraction, compare classical morphology, SSL embeddings, and
combined features with:

```bash
ssl-vit-benchmark \
  --input outputs/phenotype/phenotype_cp_ssl.tsv \
  --output-dir outputs/benchmarks/ssl_feature_comparison \
  --label-col sgRNA \
  --feature-set classical=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_ \
  --feature-set ssl=prefix:ssl_ \
  --feature-set combined=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_,prefix:ssl_ \
  --batch-col plate \
  --batch-col well \
  --write-report
```


## Optional DINOv3 Transfer Baseline

DINOv3 can be attempted through the Hugging Face wrapper:

```bash
pip install -e ".[dinov3]"
```

Then set:

```yaml
ssl_model_builder: manuscript.models.dinov3:build_dinov3_hf_backbone_tokens
ssl_ckpt: null
ssl_normalization: minmax
ssl_patch_size: 16
ssl_model_kwargs:
  model_name: facebook/dinov3-vitb16-pretrain-lvd1689m
  channel_adapter: mean_to_rgb
  apply_imagenet_norm: true
```

Treat DINOv3 as a transfer baseline, not an automatic replacement for a Ward-trained checkpoint. Public DINOv3 models are RGB natural-image backbones, so the wrapper uses an explicit channel adapter for microscopy channels and the benchmark should verify that biological separability improves without increasing plate/well/batch leakage. See `docs/dinov3_experiment.md`.

## Brieflow Integration

For folding this into Ward Lab's Brieflow fork, treat this repository as a
prototype and migrate the pieces into a feature branch from `develop`.

Recommended branch:

```bash
feat/ssl-vit-phenotyping
```

The integration should stay opt-in through config. With `ssl_enable=False`, the
existing phenotype table should be unchanged. See:

- `docs/brieflow_integration.md`
- `workflow/config/ssl_phenotype.example.yml`

## Model Choice

The extraction code is intentionally model-agnostic. DINOv3 should be tested as
the strongest current DINO-style candidate, while DINOv2 remains a stable
fallback/comparator. MAE, iBOT, MoCo v3, and eventually a Ward-trained DINO-style
model are good benchmark alternatives.

See `docs/model_rationale.md` for the advisor-facing justification and benchmark
plan.

## QC Gate Documentation

See `docs/qc_gates.md` for stop/continue thresholds, `docs/benchmark_outputs.md` for output interpretation, and `docs/ground_truth_data.md` for large ground-truth storage strategy.

## Advisor Clarifications and Model Rationale

Generated Python/build artifacts should not be tracked. If `__pycache__/`, `.pytest_cache/`, or `*.egg-info/` were committed before `.gitignore` was added, run:

```bash
bash scripts/remove_tracked_artifacts.sh
git status
```

For checkpoint and preliminary-embedding questions, see:

- `docs/advisor_clarifications.md`
- `docs/model_rationale.md`
- `docs/ssl_scope.md`
- `docs/mask_source_strategy.md`

The extraction script records model provenance in output metadata so preliminary embeddings can be traced back to the checkpoint path, checkpoint SHA256, model builder, patch size, pooling mode, normalization, selected channels, and git commit.

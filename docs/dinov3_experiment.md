# DINOv3 experiment plan

DINOv3 should be treated as an optional transfer-learning baseline, not as the
assumed production model. The main reason to try it is that the DINOv3 technical
report emphasizes high-quality dense features and improved transfer across
vision tasks, which aligns with this project’s patch-token pooling strategy for
cell masks.

## Install

```bash
pip install -e ".[dinov3]"
```

## Recommended first run

Use a small, representative subset before running the whole screen:

- 1 to 2 plates or a balanced subset of wells.
- Low, medium, and high cell-density examples.
- Negative controls and at least one positive-control perturbation.
- A small number of annotated ground-truth masks if available.

## Configuration

Use the Hugging Face wrapper as the model builder:

```yaml
phenotype:
  ssl:
    enable: true
    ckpt: null
    model_builder: manuscript.models.dinov3:build_dinov3_hf_backbone_tokens
    device: cuda
    patch_size: 16
    use_channels: [0, 1]
    pooling: mean
    normalization: minmax
    model_kwargs:
      model_name: facebook/dinov3-vitb16-pretrain-lvd1689m
      channel_adapter: mean_to_rgb
      apply_imagenet_norm: true
      trust_remote_code: true
```

## Why `normalization: minmax`?

DINOv3 public checkpoints are natural-image RGB models. The wrapper maps the
selected microscopy channels into three channels and optionally applies ImageNet
normalization. Feeding per-channel z-scored fluorescence into ImageNet
normalization is usually less interpretable, so the first DINOv3 transfer run
should use min-max normalized inputs.

## What to compare

Run the benchmark with at least these feature sets:

```bash
ssl-vit-benchmark \
  --input outputs/phenotype/phenotype_cp_ssl.tsv \
  --output-dir outputs/benchmarks/dinov3_transfer \
  --label-col sgRNA \
  --replicate-col replicate \
  --perturbation-col perturbation \
  --feature-set classical=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_ \
  --feature-set dinov3=prefix:ssl_ \
  --feature-set combined=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_,prefix:ssl_ \
  --batch-col plate \
  --batch-col well \
  --write-report
```

## Decision rule

Keep DINOv3 only if it improves biological signal without increasing nuisance
signal. In practice, it should improve sgRNA/perturbation separability,
positive-control separation, and replicate consistency while not making plate,
well, cell density, or imaging date more predictive than the biological labels.

## Notes for advisor clarification

The main caveat is channel/domain mismatch. DINOv3 is a strong dense-feature
model, but public checkpoints are not trained on this lab’s exact fluorescence
screening distribution. This is why the repo keeps DINOv3 behind an explicit
model-builder path and records checkpoint/model provenance for every output.

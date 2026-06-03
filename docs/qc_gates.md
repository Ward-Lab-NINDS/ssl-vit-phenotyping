# QC Gates for SSL ViT Phenotyping and ProCode Screens

This pipeline should not allow learned SSL embeddings to compensate for poor upstream biology or image processing. Interpret outputs in this order:

1. segmentation QC
2. ProCode on/off decoding QC
3. sgRNA/SBS assignment quality
4. classical morphology separability
5. SSL and combined embedding separability
6. replicate and batch-effect checks

## 1. Segmentation QC gate

Run segmentation metrics before ProCode decoding or SSL interpretation.

Recommended warning flags:

| Metric | Why it matters | Initial warning threshold |
|---|---|---:|
| `cell_count` | low counts make downstream scores unstable | experiment-specific |
| `foreground_fraction` | detects sparse/overfilled fields | compare across density |
| `small_mask_fraction` | debris or oversegmentation | > 0.10 |
| `oversized_mask_fraction` | merged cells / undersegmentation | > 0.05 |
| `touching_edge_fraction` | crowding and mask contact | increasing with density |
| `cell_area_cv` | heterogeneous segmentation quality | compare across plates |

Stop or retune segmentation when oversized masks, touching edges, or small masks increase sharply at higher cell density.

## 2. ProCode decoding QC gate

Do not cluster cells by SSL features until ProCode decoding is clean.

Recommended warning flags:

| Metric | Meaning | Initial warning threshold |
|---|---|---:|
| `fraction_empty_signature` | no channels pass threshold | > 0.05 |
| `fraction_invalid_signature` | decoded code not in codebook | > 0.05 |
| `median_procode_margin` | on/off channel separation | inspect per channel |
| `median_crosstalk_index` | off-channel leakage relative to on signal | > 0.30 |
| `fraction_ambiguous` | conservative combined failure flag | > 0.10 |

Use negative-control threshold calibration when available. Quantile, Otsu, and GMM thresholds are acceptable for exploratory runs, but control-based thresholds are easier to defend in a screen.

## 3. Feature separability gate

Compare classical morphology, SSL embeddings, and combined features. Combined features should improve biological separability without increasing batch leakage.

Use:

```bash
ssl-vit-benchmark \
  --input outputs/phenotype/phenotype_cp_ssl.tsv \
  --output-dir outputs/benchmarks/ssl_feature_comparison \
  --label-col sgRNA \
  --replicate-col replicate \
  --perturbation-col perturbation \
  --feature-set classical=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_ \
  --feature-set ssl=prefix:ssl_ \
  --feature-set combined=prefix:cell_,prefix:nuclei_,prefix:cytoplasm_,prefix:ssl_ \
  --batch-col plate \
  --batch-col well \
  --batch-col imaging_date \
  --write-report
```

## 4. Batch-effect gate

If `batch_signal_*.tsv` shows that plate, well, density, or imaging date is more predictable than sgRNA/perturbation labels, treat the run as batch-confounded.

Suggested rule:

```text
batch knn_accuracy / biological knn_accuracy > 0.75 = review
batch knn_accuracy / biological knn_accuracy > 1.00 = likely confounded
```

## 5. Control gate

Use positive and negative controls before unknown perturbations.

Good signs:

- non-targeting controls form a compact centroid
- positive controls move away from non-targeting controls
- within-gene sgRNAs agree more than unrelated perturbations
- replicate centroids agree across plates or imaging dates

## Decision table

| Result | Action |
|---|---|
| segmentation fail | retune masks or lower cell density |
| ProCode fail | retune thresholds, exposure, bleedthrough correction, or codebook |
| feature fail but QC pass | test alternate SSL model/pooling or add compartments |
| batch signal high | regress batch, redesign acquisition, or split analysis by batch |
| all gates pass | proceed to phenotype ranking and biological interpretation |

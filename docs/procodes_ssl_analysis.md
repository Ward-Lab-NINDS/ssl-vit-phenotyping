# ProCodes, Segmentation, and SSL Analysis Plan

This extension benchmarks whether ProCode combinatorial identity expansion stays
usable for pooled imaging screens once cells are segmented, decoded, and embedded.
The working assumption from the Frankel paper is that combinatorial epitope tags
are powerful when decoding is clean: strong signal separation and low crosstalk
are required before downstream clustering or perturbation mapping can be trusted.

Current channel interpretation: `V5` / `647` far red, `NWS` / `488` green, and
`T7` / `568` orange are ProCode/readout channels. The fourth likely nucleus
channel is structural/reference signal used for segmentation, image QC, and
cell linkage. Do not use the nucleus channel as a ProCode identity bit.

## Immediate Priorities

1. Compare ProCode on/off signatures across hand-sorted images and pooled ProCode
   images once both datasets are available.
2. Benchmark lower plating density as a segmentation intervention.
3. Verify that per-cell sgRNA calls from in situ SBS remain separable in both
   classical morphology features and SSL embeddings.
4. Keep SSL integration modular so it can be benchmarked without disrupting the
   existing OPS workflow.

## Implemented Analysis Hooks

The `lib.phenotype.procode_analysis` module provides:

- `add_procode_signatures`: thresholds V5/NWS/T7 ProCode/readout channels into binary on/off
  signatures and records margin plus crosstalk index per cell.
- `summarize_procode_decoding`: summarizes decoded signatures, expected codes,
  signal margin, and crosstalk.
- `segmentation_quality_from_labels`: reports cell count, foreground fraction,
  small/oversized masks, touching-cell edges, and an undersegmentation proxy.
- `compare_segmentation_by_density`: aggregates segmentation metrics across
  density conditions.
- `evaluate_feature_separability`: evaluates sgRNA or perturbation separability
  using kNN accuracy and silhouette score.
- `compare_classical_ssl_separability`: compares classical morphology feature
  prefixes against SSL embedding prefixes.
- `replicate_consistency`: checks whether replicate perturbation centroids agree
  in feature space.

## SSL Defaults

The phenotype script records the core SSL metadata in the output table:

- `meta_ssl_patch_size`, default `8`
- `meta_ssl_pooling`, default `mean`
- `meta_ssl_normalization`, default `zscore`
- `meta_ssl_channels`, default `all` or a comma-separated channel list
- `meta_ssl_pca_dim`, default `none`

PCA remains disabled unless a global PCA basis is provided, preserving embedding
geometry across images and experiments.

## Planned Visuals

These are downstream visualization targets once the hand-sorted and ProCode image
sets are available:

- patch grid overlay on a single segmented cell to show patch size
- schematic of token pooling inside a cell mask
- UMAP colored by perturbation or sgRNA
- classical morphology versus SSL separability comparison
- segmentation quality summary across density conditions

## Suggested Evaluation Table

Each experiment should preserve these columns where possible:

| Column | Purpose |
| --- | --- |
| `image_id` | Links metrics back to source image or tile. |
| `density` | Enables low/medium/high density comparisons. |
| `label` | Cell mask label used for feature merging. |
| `sgRNA` | In situ SBS perturbation identity. |
| `replicate` | Biological or technical replicate. |
| `procode_signature` | Binary on/off combinatorial ProCode call. |
| `procode_margin` | On-channel minus off-channel signal separation. |
| `procode_crosstalk_index` | Off-channel signal relative to on-channel signal. |

The scale-up gate is reliability: segmentation, sgRNA decoding, and morphology
embeddings should be stable before expanding pooled analyses.

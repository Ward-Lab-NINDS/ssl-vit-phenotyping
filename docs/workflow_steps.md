# Workflow Step Documentation

This document records each major step in the analysis workflow so humans and coding agents can understand what inputs, outputs, and assumptions are expected.

## Step 1: Load images and masks

**Purpose:** Load multichannel phenotype images and cell/nuclei/cytoplasm masks.

**Primary script:** `workflow/scripts/phenotype/extract_phenotype_cp_multichannel_ssl.py`

**Expected inputs:**

- multichannel image array `[C, H, W]` or compatible TIFF/OME-TIFF layout;
- nuclei mask `[H, W]`;
- cell mask `[H, W]`;
- cytoplasm mask `[H, W]`;
- config values for channel names, mask source, and segmentation QC status.

**Expected outputs:**

- phenotype table with classical features if CellProfiler/Brieflow extraction is available;
- fallback label-only phenotype table if classical extraction functions are unavailable;
- mask provenance metadata columns.

**Downstream dependency:** SSL embedding extraction and benchmarking require a stable `label` column.

## Step 2: Record mask source and segmentation status

**Purpose:** Prevent confusion about whether masks came from Brieflow, CellPose, SAM, manual labels, or another upstream source.

**Primary module:** `src/lib/phenotype/mask_source_qc.py`

**Expected inputs:**

- phenotype table;
- `mask_source`;
- `mask_source_detail`;
- `segmentation_model`;
- `segmentation_qc_status`.

**Expected outputs:**

- `meta_mask_source`;
- `meta_mask_source_detail`;
- `meta_segmentation_model`;
- `meta_segmentation_qc_status`;
- `meta_ssl_role`.

**Downstream dependency:** All benchmark tables should preserve these metadata fields when possible.

## Step 3: Extract SSL embeddings

**Purpose:** Extract ViT patch-token embeddings and pool them inside accepted cell masks.

**Primary modules:**

- `src/lib/phenotype/ssl_cell_features.py`
- `src/lib/phenotype/ssl_model_loader.py`
- `src/manuscript/models/vit.py`
- `src/manuscript/models/dinov3.py`

**Expected inputs:**

- image array `[C, H, W]`;
- cell labels `[H, W]`;
- model builder path;
- checkpoint path or transfer model name;
- patch size;
- pooling method;
- normalization method;
- selected channels.

**Expected outputs:**

- one row per cell label;
- SSL feature columns such as `ssl_000`, `ssl_001`, etc.;
- model provenance columns.

**Downstream dependency:** Feature benchmarking uses SSL columns by prefix.

## Step 4: ProCode decoding and QC

**Purpose:** Decode ProCode on/off signatures and identify ambiguous cells before biological interpretation.

**Primary module:** `src/lib/phenotype/procode_analysis.py`

**Expected inputs:**

- phenotype table;
- ProCode channel columns;
- thresholds or threshold-calibration method;
- expected signatures when known;
- minimum margin and maximum crosstalk rules.

**Expected outputs:**

- decoded signatures;
- ProCode QC summary;
- flagged ambiguous cells;
- codebook design summary when expected signatures are provided.

**Downstream dependency:** sgRNA/perturbation feature benchmarks should be interpreted only after ProCode QC is acceptable.

## Step 5: Benchmark feature sets

**Purpose:** Compare classical morphology, SSL embeddings, and combined feature sets.

**Primary module/CLI:** `src/lib/phenotype/benchmarking.py` / `ssl-vit-benchmark`

**Expected inputs:**

- phenotype table;
- label column such as `sgRNA` or `perturbation`;
- feature-set specifications;
- replicate and batch columns when available.

**Expected outputs:**

- `feature_separability.tsv`;
- `feature_ranking.tsv`;
- `replicate_consistency.tsv`;
- `batch_signal_*.tsv`;
- `benchmark_report.md`.

**Downstream dependency:** Advisor-facing interpretation and model-selection decisions.

## Step 6: Interpret and decide next model direction

**Purpose:** Decide whether SSL is worth scaling, whether a segmentation-focused model is needed, or whether classical features are sufficient.

**Inputs:**

- QC gate results;
- feature benchmark results;
- batch leakage checks;
- Subcell/HPA compatibility notes;
- autoencoder/SSL comparison results when available.

**Outputs:**

- recommendation: proceed with SSL, train domain-specific SSL checkpoint, implement autoencoder baseline, improve segmentation/unmixing, or stop.

## Rule for agents

When adding a new workflow step, append a section to this document with:

1. what the step does;
2. functions/scripts created;
3. expected input;
4. expected output;
5. output save location;
6. downstream dependency;
7. QC or stop conditions.

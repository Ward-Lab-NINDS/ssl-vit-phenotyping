# Benchmark Output Guide

`ssl-vit-benchmark` writes one TSV per benchmark plus an optional `benchmark_report.md`.

| File | Purpose |
|---|---|
| `feature_separability.tsv` | kNN and silhouette scores for each feature set |
| `feature_ranking.tsv` | feature sets ranked by kNN accuracy, silhouette, and feature count |
| `replicate_consistency.tsv` | perturbation centroid agreement across replicates |
| `procode_thresholds.tsv` | calibrated per-channel ProCode thresholds |
| `procode_decoding.tsv` | observed signature counts and decoding margins |
| `procode_qc.tsv` | one-row ProCode QC summary |
| `procode_flagged_cells.tsv` | per-cell ambiguous decoding flags |
| `procode_codebook.tsv` | codebook spacing summary |
| `control_phenotype_qc_*.tsv` | positive/negative-control centroid checks |
| `batch_signal_*.tsv` | biology-vs-batch separability checks |
| `benchmark_report.md` | advisor-facing interpretation report |

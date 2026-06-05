# Advisor Scope Update

The project has been reframed based on advisor feedback.

## Revised goal

The goal is not to use SSL as a universal replacement for Brieflow, CellPose, SAM, or CellProfiler. The goal is to make the phenotype representation layer more modular.

The pipeline should accept trusted masks from Brieflow/CellPose/SAM/manual sources, record their provenance, run segmentation and ProCode QC, and then compare classical features against SSL features.

## Why this is more viable

Segmentation and phenotype representation need different validation strategies:

- segmentation requires ground-truth masks, object-level overlap metrics, Z-stack/channel handling, and visual QC;
- phenotype representation requires perturbation separability, replicate consistency, control separation, and batch-leakage checks.

SSL is most defensible in the second category unless it is trained and validated specifically for segmentation.

## Current repo direction

- Keep Brieflow/CellPose/SAM/manual masks as acceptable upstream inputs.
- Add metadata columns documenting mask source and segmentation QC status.
- Treat DINOv2/DINOv3/local checkpoints as optional feature-extraction candidates.
- Deprioritize broad model-name comparison unless a model directly answers the phenotype-representation question.
- Add future hooks for deconvolution, unmixing, and Z-stack-aware preprocessing before segmentation.

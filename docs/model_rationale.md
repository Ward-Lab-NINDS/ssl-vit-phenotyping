# Model Rationale and Benchmark Plan

The goal is not to pick a model by reputation. The goal is to test whether SSL ViT features add phenotype signal beyond classical morphology after segmentation and V5/NWS/T7 ProCode/readout decoding pass QC.

## Default stance

- Use DINOv2 as a stable baseline/comparator.
- Test DINOv3 as the strongest current DINO-style candidate when weights and licensing are practical.
- Include at least one non-DINO SSL method, such as iBOT, MAE, or MoCo v3.
- Include a domain-pretrained model when the image domain fits the available model, such as UNI, Virchow, Phikon, or CTransPath for pathology-like data.
- Treat a Ward-trained SSL checkpoint as the long-term target if enough unlabeled lab images are available.

## Required comparison outputs

For each model/checkpoint, produce:

```text
feature_separability.tsv
feature_ranking.tsv
replicate_consistency.tsv
batch_signal_*.tsv
control_phenotype_qc_*.tsv
procode_qc.tsv
benchmark_report.md
```

## Decision criteria

A model is stronger only if it improves biological signal without increasing nuisance signal.

Pass criteria should include:

1. ProCode QC passes before phenotype interpretation.
2. Segmentation QC passes before phenotype interpretation.
3. SSL-only or combined features improve sgRNA/perturbation separability.
4. Replicate centroids are more consistent within perturbation than between unrelated perturbations.
5. Plate/well/batch separability is not stronger than biological separability.
6. Positive controls separate from non-targeting controls.

## Preliminary embedding subset

Start with a small subset before scaling:

```text
1-2 representative plates
balanced wells across density conditions
non-targeting controls
strong positive controls
several expected ProCode signatures
held-out wells or plate for leakage checks
```

This gives the advisor a concrete answer to: "Do you have preliminary embeddings?" without prematurely running the entire dataset.

## Updated model-selection framing

The model comparison should not be framed as chasing model names. The practical comparison is:

1. classical CellProfiler/Brieflow morphology features;
2. SSL features from a stable ViT checkpoint;
3. DINOv3 or other transfer baselines;
4. a future Ward-trained microscopy SSL checkpoint;
5. combined classical + SSL features.

Broader SSL models such as iBOT can be deprioritized unless they answer a specific benchmark question. The more important distinction is whether deep learning is being used for segmentation/signal cleanup or for downstream phenotype representation.

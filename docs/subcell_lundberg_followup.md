# Subcell / Lundberg Lab Follow-Up

This document is a structured follow-up plan for evaluating whether the Lundberg Lab / Human Protein Atlas subcellular localization ecosystem can inform the SSL phenotyping workflow.

## Why this is relevant

The Human Protein Atlas subcellular resource contains immunofluorescence/confocal microscopy data and annotations for protein localization across many genes, cell lines, organelles, and fine subcellular structures. Because this project is trying to learn biologically useful cell/subcellular image representations, HPA/Lundberg-style localization datasets are relevant as potential pretraining, comparison, or interpretation resources.

## Main compatibility question

> Are Subcell/HPA-style image inputs, labels, metadata, and learned representations compatible enough with the Ward/Brieflow phenotyping data to accelerate or validate the current SSL project?

## Information to collect

For the specific Subcell model/dataset under review, document the following.

### 1. Model identity

- Exact model name.
- Lab/group/source.
- Publication or repository link.
- License and reuse restrictions.
- Whether pretrained weights are available.
- Whether embeddings or only predictions are available.

### 2. Inputs

Record the expected input structure:

- image format;
- number of channels;
- channel meaning;
- image size/crop size;
- whether inputs are fields of view, single cells, or segmented crops;
- whether masks are required;
- expected intensity normalization;
- whether z-stacks are supported;
- whether cell-line or protein metadata are passed to the model.

### 3. Outputs

Record what the model returns:

- class probabilities;
- subcellular localization labels;
- cell-level embeddings;
- field-level embeddings;
- segmentation masks;
- uncertainty scores;
- attention maps or saliency outputs;
- protein-level summaries.

### 4. Metadata

Record required or available metadata:

- gene/protein identifier;
- antibody or tag information;
- cell line;
- compartment/localization label;
- cell-cycle or single-cell variation annotation;
- image ID;
- plate/well/site;
- microscope/acquisition information;
- QC flags.

### 5. Compatibility with this project

Fill out this table after inspection.

| Item | Ward/Brieflow project | Subcell/HPA model | Compatible? | Notes |
| --- | --- | --- | --- | --- |
| Image unit | FOV/tile or cell crop | TBD | TBD |  |
| Channels | nuclei/marker/ProCode/etc. | TBD | TBD |  |
| Masks required | yes, for SSL pooling | TBD | TBD |  |
| Z-stack support | possible/future | TBD | TBD |  |
| Output embeddings | SSL cell features | TBD | TBD |  |
| Labels | sgRNA/perturbation/ProCode | localization/protein | TBD |  |
| Metadata | plate/well/site/condition | gene/cell line/localization | TBD |  |

## How it could be leveraged

Possible uses:

1. **Pretraining inspiration**: use HPA/Subcell as evidence that localization-rich microscopy data can support learned representations.
2. **Weights reuse**: load pretrained weights if channel and crop structure are close enough.
3. **Embedding comparator**: compare Subcell embeddings against DINOv2/DINOv3/local SSL embeddings.
4. **Label-space comparator**: compare ProCode/neurofilament phenotypes to known subcellular localization-style features.
5. **Architecture guidance**: borrow model design if it handles single-cell crops, multi-channel images, or localization labels better than generic ViTs.

## What to be cautious about

Do not assume compatibility until the data contract is checked. Potential mismatches include:

- HPA-style protein localization images may use different stains/channels than Ward ProCode images;
- field-of-view embeddings may not transfer to cell-level phenotype embeddings;
- localization labels may not match perturbation phenotypes;
- pretrained weights may expect RGB or fixed-channel input;
- microscope, cell line, and staining protocol shifts may dominate learned features;
- a model trained for classification may not produce embeddings ideal for perturbation clustering.

## Recommendation

Treat Subcell/HPA as a structured follow-up, not as an automatic replacement for the current pipeline. The immediate task is to document its data contract and run a small compatibility test:

1. match channel expectations;
2. match image/crop size;
3. confirm whether masks are needed;
4. extract embeddings from a small Ward subset;
5. compare embeddings to classical, DINO, and local SSL features;
6. inspect whether embeddings track biology or acquisition metadata.

## Decision criteria

Move forward with Subcell/HPA integration only if:

- weights or embeddings are accessible under usable licensing;
- input channels can be mapped without major distortion;
- outputs can be converted to cell-level features;
- embeddings improve biological metrics or interpretation;
- batch leakage remains controlled.

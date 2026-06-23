# Future Project Direction: SSL Phenotyping in Brieflow

This document records the current project direction after advisor and peer feedback. The main goal is to make the SSL phenotyping project defensible before scaling it inside Brieflow or applying it to larger ProCode/optical pooled screening datasets.

## Central project question

The project should answer a narrow, testable question:

> After acceptable masks, ProCode decoding, and metadata QC are available, do self-supervised microscopy embeddings provide useful phenotype representations beyond classical CellProfiler/Brieflow features?

This intentionally separates SSL phenotype representation from upstream segmentation. SSL is not treated as a direct replacement for Brieflow, CellPose, SAM, or manual segmentation. Instead, the pipeline accepts masks from any documented source, records mask provenance, and evaluates SSL features only after upstream QC gates pass.

## Why SSL is worth testing

Self-supervised learning is promising because the lab can often collect far more microscopy images than manual phenotype labels. In this context, SSL is useful if it can learn reusable image representations from unlabeled or weakly labeled images, transfer across experimental conditions, and support downstream clustering, retrieval, or perturbation classification.

The strongest argument is not that SSL is automatically better than every alternative. The stronger argument is:

> SSL is a practical representation-learning strategy for high-content microscopy when labels are limited, biological phenotypes are subtle, and downstream tasks may change over time.

## SSL versus autoencoders

Autoencoders are useful when reconstruction quality is the goal or when the desired representation should preserve most pixel-level image information. However, reconstruction can also encourage a model to preserve nuisance variation such as illumination, background, blur, staining artifacts, or batch-specific texture.

SSL contrastive, masked, or teacher-student approaches can be evaluated because they may produce embeddings that are more useful for biological comparison than raw reconstruction. In this project, SSL should win only if it improves downstream biological utility while avoiding batch leakage.

| Question | Autoencoder framing | SSL phenotyping framing |
| --- | --- | --- |
| Training objective | Reconstruct input image | Learn invariant/useful representation |
| Label needs | Usually unlabeled | Usually unlabeled or weakly labeled |
| Risk | Preserves nuisance artifacts | May learn shortcut/batch features if unchecked |
| Best use | Denoising, compression, image generation, reconstruction | Feature extraction, retrieval, clustering, transfer |
| Success metric | Reconstruction loss, image fidelity | sgRNA/perturbation separability, replicate consistency, control separation, low batch leakage |

## What SSL should be tested against

The benchmark should not only compare model names. It should compare modeling roles:

1. Classical CellProfiler/Brieflow morphology features.
2. Autoencoder-style reconstruction embeddings, if available.
3. DINOv2/DINOv3 or other SSL ViT embeddings.
4. Domain-specific microscopy SSL checkpoints, if available.
5. Combined classical + SSL features.

The most important comparison is whether SSL adds biologically meaningful signal beyond classical features without increasing plate, well, batch, imaging-date, or density leakage.

## Success criteria

SSL should be considered useful only if it improves at least one downstream biological task while passing QC:

- improves perturbation or sgRNA separability;
- improves replicate centroid consistency;
- separates positive controls from non-targeting controls;
- supports phenotype retrieval or clustering that matches known biology;
- does not primarily cluster by plate, well, site, density, batch, imaging date, or microscope settings;
- remains interpretable enough to connect embeddings back to cell/subcellular patterns.

## Stop conditions

Do not scale SSL if:

- segmentation QC fails;
- ProCode decoding is ambiguous or crosstalk is high;
- SSL embeddings separate mostly by batch or acquisition metadata;
- classical features perform equally well with better interpretability;
- embeddings cannot be traced to a checkpoint, model builder, patch size, channel adapter, and preprocessing configuration.

## Near-term implementation plan

1. Freeze a small representative benchmark set.
2. Document the raw image, mask, metadata, and output table contracts.
3. Run classical features only.
4. Run SSL embeddings with the current local/Ward checkpoint path.
5. Run DINOv2/DINOv3 as optional transfer baselines.
6. Compare classical, SSL, and combined features.
7. Report QC gates before any biological interpretation.
8. Decide whether to invest in a Ward-trained SSL checkpoint, autoencoder baseline, or segmentation-focused model.

## Key framing for presentations

Use this framing:

> The project is not trying to replace segmentation with SSL. It is trying to test whether self-supervised embeddings provide a more transferable, biologically useful phenotype representation once reliable masks and metadata already exist.

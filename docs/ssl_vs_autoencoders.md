# Why SSL for Phenotyping Instead of Only Autoencoders?

This document gives the project-level argument for testing self-supervised learning (SSL) in the Brieflow phenotyping workflow, especially compared with autoencoder-based representation learning.

## Short answer

Autoencoders learn by reconstructing input images. That can produce useful compressed representations, but reconstruction can reward the model for preserving nuisance variation that is not biologically meaningful. SSL methods are worth testing because they can be trained on unlabeled microscopy data while encouraging representations that support transfer, retrieval, clustering, and downstream biological discrimination.

## What SSL may uniquely capture

SSL may be useful in this project if it captures:

- repeated subcellular organization patterns across cells;
- texture and spatial distribution of protein or organelle signals;
- relationships between cell shape, nuclear morphology, and marker localization;
- subtle phenotype differences that are hard to summarize with handcrafted features;
- representations that transfer across plates, fields, or related experiments.

The important claim is not that SSL always captures more biology. The claim is that SSL gives a testable way to learn image representations without requiring manual phenotype labels for every cell.

## Why not rely only on autoencoders?

Autoencoders can be valuable for denoising, compression, anomaly detection, and generative modeling. However, for phenotype representation, reconstruction alone may not be aligned with the downstream biological task. A model can reconstruct images well while still encoding batch effects, illumination artifacts, microscope settings, or staining intensity differences that are not the phenotype of interest.

In this project, an autoencoder baseline is useful if the lab wants to test whether reconstruction embeddings perform as well as or better than SSL embeddings. But the final decision should be based on downstream phenotyping metrics, not reconstruction loss alone.

## Recommended benchmark

For a convincing comparison, benchmark:

1. Classical morphology features.
2. Autoencoder embeddings, if implemented or available.
3. DINOv2/DINOv3 or other SSL ViT embeddings.
4. A domain-specific microscopy SSL checkpoint, if available.
5. Combined classical + SSL features.

Use the same train/test split, labels, QC filters, and batch leakage checks for every feature set.

## Evaluation metrics

Prioritize biological utility:

- kNN or linear-probe accuracy for perturbation/sgRNA labels;
- silhouette or neighborhood enrichment by biological label;
- replicate centroid consistency;
- positive-control versus non-targeting-control separation;
- retrieval of biologically similar perturbations;
- batch, plate, well, site, density, and imaging-date predictability.

A feature set should not be considered better if it improves label separability only by learning plate or batch artifacts.

## Recommended advisor-facing statement

> SSL is being tested because microscopy datasets are often label-limited but image-rich. Autoencoders are a useful comparator, but reconstruction is not always aligned with biological phenotype extraction. The project will compare classical, autoencoder, SSL, and combined representations using downstream perturbation separability, replicate consistency, control behavior, and batch-leakage checks.

# SSL Scope and Project Framing

This project separates two related but different tasks.

## Task 1: Segmentation and signal decoding

This task asks whether cells, nuclei, cytoplasm, ProCode/readout channels, and neurofilament-like structures are being detected correctly. It is an upstream image-processing problem. Good methods here may include:

- Brieflow, CellPose, SAM-style prompting, StarDist, ilastik, or manual labels for masks;
- color deconvolution or channel separation when signals overlap;
- linear/spectral unmixing when fluorescence bleed-through affects ProCode calls;
- Z-stack projection, focus selection, or 2.5D/3D segmentation;
- segmentation QC and ProCode/readout decoding QC before any phenotype claim.

SSL/ViT embeddings are **not** assumed to solve this task by themselves. If the goal is improved ProCode or neurofilament segmentation, the pipeline should prioritize segmentation models, unmixing, deconvolution, and Z-stack handling.

## Readout channels versus structural channel

The confirmed ProCode/readout channels are:

- `V5` / `647` far red: barcode-like ProCode/epitope-style readout.
- `NWS` / `488` green: barcode-like ProCode/epitope-style readout. Do not expand `NWS` beyond this label unless a lab codebook is provided.
- `T7` / `568` orange: barcode-like ProCode/epitope-style readout.

The remaining fourth channel is believed to be the nucleus channel. Treat it as a structural/reference channel for segmentation, cell counting, image QC, and linking V5/NWS/T7 signal back to individual cells. Do not treat the nucleus channel as a ProCode identity channel.

SSL patch embeddings are phenotype features. They are not the source of ProCode identity. If V5/NWS/T7 readout QC fails, downstream SSL interpretation should be marked unreliable or exploratory even when the embedding plots look separated.

## Task 2: Phenotype feature extraction

This task asks whether learned features provide a useful cell-level phenotype representation after acceptable masks already exist. This is where SSL is useful. The project benchmarks:

- classical CellProfiler/Brieflow morphology features;
- SSL ViT features pooled inside accepted cell masks;
- combined classical + SSL features;
- DINOv2/DINOv3/local Ward-trained checkpoints as model candidates.

The primary claim is therefore:

> SSL may reduce dependence on handcrafted CellProfiler-style features, but it does not remove the need for reliable segmentation.

## Practical rule

Do not interpret SSL embedding clusters until these upstream checks pass:

1. Mask source is known and recorded.
2. Segmentation QC is acceptable.
3. ProCode/readout QC for V5, NWS, and T7 is acceptable.
4. ProCode signature decoding or ambiguity flagging is documented.
5. Batch/plate/well leakage is lower than biological/perturbation signal.

## Recommended wording for advisors

The pipeline is mask-source agnostic. It can ingest masks from Brieflow, CellPose, SAM-style workflows, manual annotation, or future segmentation approaches. The SSL component is a downstream representation layer that tests whether learned morphology embeddings improve phenotype analysis compared with classical features.

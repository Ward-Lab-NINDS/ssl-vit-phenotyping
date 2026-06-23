# Agent Editing Contract

This document is for coding agents and contributors modifying the SSL phenotyping repository.

## Non-negotiable assumptions

1. SSL does not replace segmentation in this repository.
2. Masks are upstream inputs and must have provenance.
3. Segmentation QC and ProCode/readout decoding QC come before biological interpretation.
4. Large raw images, masks, z-stacks, embeddings, and checkpoints should not be committed to GitHub.
5. Every new input/output structure must be documented.
6. Every new model path must record provenance.
7. Every new benchmark should include batch/metadata leakage checks when metadata are available.

## Channel metadata rules

1. Do not rename `V5`, `NWS`, or `T7` as generic `ch01`/`ch02`/`ch03` in final outputs if channel mapping metadata are available.
2. Treat `V5` / `647` far red, `NWS` / `488` green, and `T7` / `568` orange as barcode-like ProCode/readout channels.
3. Do not treat the nucleus channel as a ProCode identity channel.
4. Do not assume an expansion or alternate meaning for `NWS` unless a lab codebook is provided.
5. Do not interpret SSL embeddings as biological phenotype evidence if segmentation QC or ProCode/readout QC failed.
6. Preserve channel metadata in exported tables and reports.

## Before changing code

Check these docs:

- `docs/ssl_scope.md`
- `docs/data_contract.md`
- `docs/workflow_steps.md`
- `docs/qc_gates.md`
- `docs/model_rationale.md`

## When adding a new function

Add or update:

- docstring;
- test;
- expected input/output documentation;
- config example if the function is exposed to a workflow;
- README or docs link if it affects user-facing behavior.

## When adding a new model

Document:

- model name;
- builder path;
- checkpoint source;
- expected channel structure;
- readout versus structural channel roles;
- input normalization;
- output tensor shape;
- whether it returns field-level, patch-level, or cell-level features;
- whether it is trained for segmentation, classification, reconstruction, or representation learning.

## When adding large data

Do not upload large data directly. Add:

- manifest row;
- checksum;
- external storage path;
- split assignment;
- annotation/mask link;
- data-use notes.

## When updating benchmark logic

Do not rely on one metric. Preserve or add:

- biological-label separability;
- replicate consistency;
- control behavior;
- batch/plate/well/site leakage;
- report generation.

## Pull request checklist

- [ ] No `__pycache__`, `.pytest_cache`, `.egg-info`, raw TIFF/Zarr/HDF5, or checkpoint files are included.
- [ ] New/changed outputs are documented.
- [ ] Tests are added or updated.
- [ ] Model/checkpoint provenance is preserved.
- [ ] Data assumptions are reflected in `docs/data_contract.md`.
- [ ] Workflow step assumptions are reflected in `docs/workflow_steps.md`.
- [ ] README links to any new high-level docs.

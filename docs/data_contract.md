# Data Contract for SSL Phenotyping

This file defines the expected data structure for the SSL phenotyping workflow. It is intended for humans and coding agents. If the data structure changes, update this document before changing code.

## 1. Raw input image structure

Document each dataset using this template.

| Field | Required? | Description | Example |
| --- | --- | --- | --- |
| `image_path` | yes | Path to raw or preprocessed multichannel image | `images/plate1/A01/site001.tif` |
| `image_format` | yes | TIFF, OME-TIFF, Zarr, HDF5, etc. | `tif` |
| `height_px` | yes | Image height in pixels | `2048` |
| `width_px` | yes | Image width in pixels | `2048` |
| `n_channels` | yes | Number of channels in the image | `4` |
| `channel_names` | yes | Ordered channel labels | `V5,NWS,T7,nucleus` |
| `channel_metadata_path` | recommended | CSV/TSV with channel roles and wavelengths | `data/ground_truth/channel_metadata.template.csv` |
| `z_slices` | no | Number of z planes if z-stack | `7` |
| `tile_size` | no | Tile/crop size if tiled | `512` |
| `preprocessing_applied` | yes | Any preprocessing already done | `flatfield,z_project_max` |

## 2. Channel metadata structure

Channel metadata should preserve biological meaning before any SSL or classical feature table is interpreted.

Recommended fields:

| Field | Required? | Description | Example |
| --- | --- | --- | --- |
| `image_id` | recommended | Image ID matching the manifest when mappings vary by image | `img001` |
| `channel_index` | recommended | Zero-based channel position in the image array | `0` |
| `channel_name` | yes | Biological/lab channel label | `V5` |
| `wavelength_nm` | recommended | Emission/excitation label used by the lab, if confirmed | `647` |
| `channel_role` | yes | Role in the workflow | `procode_readout` |
| `color_label` | recommended | Human-readable color label | `far_red` |
| `marker_or_readout` | recommended | Marker/readout name to preserve in outputs | `V5` |
| `notes` | optional | Free-text caveats | `ProCode epitope/readout channel` |

Allowed `channel_role` values:

- `procode_readout`
- `structural_reference`
- `phenotype_marker`
- `background`
- `unknown`

Current example mapping:

| channel_name | wavelength_nm | channel_role | color_label | marker_or_readout |
| --- | --- | --- | --- | --- |
| `V5` | `647` | `procode_readout` | `far_red` | `V5` |
| `NWS` | `488` | `procode_readout` | `green` | `NWS` |
| `T7` | `568` | `procode_readout` | `orange` | `T7` |
| `nucleus` |  | `structural_reference` | `unknown` | `nucleus` |

`V5`, `NWS`, and `T7` are barcode-like ProCode/readout channels. The nucleus channel is a structural/reference channel for segmentation, cell counting, image QC, and linking readout signal to cells. The fourth/nuclear channel wavelength should be confirmed before hard-coding it.

## 3. Mask structure

The SSL step requires masks or labels from an upstream segmentation source. Masks may come from Brieflow, CellPose, SAM-style workflows, manual labels, StarDist, or another documented method.

| Field | Required? | Description |
| --- | --- | --- |
| `cell_mask_path` | yes | Integer-labeled cell mask path |
| `nuclei_mask_path` | recommended | Integer-labeled nuclei mask path |
| `cytoplasm_mask_path` | optional | Integer-labeled cytoplasm mask path |
| `mask_source` | yes | `brieflow`, `cellpose`, `sam`, `manual`, `stardist`, or `other` |
| `segmentation_model` | recommended | Model/tool name and version if available |
| `segmentation_qc_status` | yes | `pass`, `fail`, `unknown`, or `not_checked` |
| `segmentation_qc_notes` | optional | Free-text notes |

Mask rules:

- Background should be label `0`.
- Cell/object labels should be positive integers.
- Masks must have the same height/width as the image after preprocessing.
- If labels are remapped, the mapping must be documented.

## 4. Metadata structure

Each image or tile should connect to experimental metadata.

| Column | Required? | Description |
| --- | --- | --- |
| `image_id` | yes | Stable image/tile identifier |
| `plate` | recommended | Plate ID |
| `well` | recommended | Well ID |
| `site` | recommended | Field/site ID |
| `condition` | recommended | Treatment or condition |
| `sgRNA` | optional | sgRNA assignment if known |
| `perturbation` | optional | Gene/perturbation assignment |
| `replicate` | recommended | Biological or technical replicate |
| `timepoint` | optional | Timepoint if longitudinal |
| `density` | optional | Cell-density condition |
| `qc_exclude` | recommended | Whether to exclude the image/cell |
| `qc_reason` | optional | Reason for exclusion |

## 5. Phenotype table output

The merged phenotype table should preserve cell identity and provenance.

Required columns:

- `label`
- `image_id` when available
- `meta_mask_source`
- `meta_segmentation_model`
- `meta_segmentation_qc_status`
- `meta_ssl_feature_role`
- `meta_ssl_segmentation_replacement`
- `meta_procode_readout_channels` when channel metadata are available
- `meta_structural_reference_channels` when channel metadata are available
- `meta_channel_output_labels` when channel metadata are available

Classical features should use stable prefixes such as:

- `cell_`
- `nuclei_`
- `cytoplasm_`

SSL features should use stable prefixes such as:

- `ssl_`
- `ssl_pca_`
- `ssl_cell_`
- `ssl_nucleus_`
- `ssl_cytoplasm_`

Report and table exports should use readable channel labels when metadata are available:

- `V5_647_far_red`
- `NWS_488_green`
- `T7_568_orange`
- `nucleus_structural_reference`

If metadata are unavailable, outputs may fall back to `ch01`, `ch02`, etc., but the report should state that the mapping is unknown.

## 6. Output directory structure

Recommended structure:

```text
outputs/
  phenotype/
    phenotype_cp_ssl.tsv
  qc/
    segmentation_qc.tsv
    procode_qc.tsv
  benchmarks/
    feature_separability.tsv
    feature_ranking.tsv
    batch_signal_plate.tsv
    benchmark_report.md
  embeddings/
    ssl_embeddings.tsv
  manifests/
    dataset_manifest.csv
    channel_metadata.csv
  ssl_patch_test/
    patches/
    patch_manifest.tsv
    patch_phenotypes.tsv
    ssl_patch_features.tsv
    ssl_patch_test_report.md
    open_patches_in_fiji.ijm
```

## 7. Agent safety rules

Coding agents should not change these assumptions without updating this document:

- SSL is downstream feature extraction, not segmentation replacement.
- Masks are external inputs with provenance.
- ProCode/readout QC for V5, NWS, and T7 happens before biological interpretation.
- The nucleus channel is structural/reference metadata, not a ProCode identity channel.
- Channel metadata should be preserved in exported tables and reports.
- Large image/mask files should not be committed to GitHub.
- New output files should be documented with expected columns and downstream use.

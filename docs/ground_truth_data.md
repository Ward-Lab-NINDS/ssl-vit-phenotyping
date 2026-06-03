# Ground-Truth Data Strategy

Do **not** commit several gigabytes of ground-truth images, masks, or annotations directly to GitHub. Keep GitHub for code, manifests, schemas, and tiny synthetic examples.

## Recommended layout

```text
data/
  ground_truth/
    manifest.csv              # tracked in git
    README.md                 # tracked in git
    raw/                      # ignored or DVC/Git-LFS tracked
    masks/                    # ignored or DVC/Git-LFS tracked
    annotations/              # tracked only if small and de-identified
    splits/                   # tracked in git
```

## Manifest-first design

Every image or mask should be represented by one row in `data/ground_truth/manifest.csv`.

Recommended columns:

| Column | Purpose |
|---|---|
| `sample_id` | stable biological sample identifier |
| `image_id` | stable image or field-of-view identifier |
| `plate` | acquisition plate |
| `well` | well position |
| `field` | field of view |
| `density` | plating/imaging density condition |
| `image_path` | relative path or remote URI to multichannel image |
| `cell_mask_path` | relative path or remote URI to cell mask |
| `nuclei_mask_path` | relative path or remote URI to nuclei mask |
| `cytoplasm_mask_path` | relative path or remote URI to cytoplasm mask |
| `annotation_path` | optional hand labels or expert annotations |
| `procode_codebook_path` | optional codebook used for decoding |
| `split` | train, validation, test, calibration, or holdout |
| `checksum_sha256` | integrity check for large files |

## Best storage options

### Best for active lab development: DVC + institutional/S3 storage

Use DVC when the data will change, grow, and be reused by multiple contributors.

```bash
dvc init
dvc remote add -d groundtruth s3://<bucket-or-lab-storage>/ssl-vit-phenotyping
dvc add data/ground_truth/raw data/ground_truth/masks
git add data/ground_truth/raw.dvc data/ground_truth/masks.dvc data/ground_truth/manifest.csv .dvc/config
git commit -m "Track ground-truth data manifest and DVC pointers"
dvc push
```

### Best for public frozen release: Zenodo, Figshare, OSF, or institutional archive

Use this when you want a citable version of the ground-truth dataset. Store only the DOI/link and manifest in GitHub.

### Acceptable for medium binary files: Git LFS

Use Git LFS only if the lab is comfortable with bandwidth and storage limits.

```bash
git lfs track "data/ground_truth/**/*.tif"
git lfs track "data/ground_truth/**/*.zarr/**"
git add .gitattributes
```

## Split policy

Do not randomly split cells from the same image across train and test. Split by plate, well, image, or biological replicate so performance reflects generalization rather than cell-level leakage.

Recommended split names:

```text
calibration   threshold/PCA/codebook fitting only
train         model or parameter development
validation    model selection
holdout       final untouched evaluation
```

## Ground-truth additions to this repo

This update includes:

- `data/ground_truth/manifest.template.csv`
- `.gitignore` rules for large local data
- this storage policy document
- benchmark hooks for batch columns such as plate, well, density, and imaging date

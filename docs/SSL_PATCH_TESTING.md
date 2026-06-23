# SSL Patch Testing

This repository should not start SSL testing on entire microscopy fields of view.
Whole images can be large, heterogeneous, and dominated by empty background, edge
artifacts, dense touching cells, or unvalidated segmentation. A small patch test
keeps the first smoke test focused on the pieces we need to trust first:

- image loading from local Ward Lab/ProCode data;
- foreground/background filtering;
- patch saving and metadata tracking;
- interpretable patch-level morphology/intensity phenotyping;
- SSL preprocessing and ViT patch-token extraction when a checkpoint is supplied;
- basic QC reporting before downstream neuron morphology interpretation.

Use 100x100 or 200x200 pixel patches as a local test scale. These patches are
large enough to exercise the SSL token workflow while staying small enough to
review quickly and run on CPU.

## Dry Run

A dry run scans the input folder, loads supported images, selects foreground-like
patches, and writes a QC report and patch manifest. It does not save patch TIFFs
or run SSL feature extraction.

Example:

```bash
python scripts/run_ssl_patch_test.py \
  --input-dir /Users/makennarodriguez/Desktop/procodes \
  --output-dir outputs/ssl_patch_test \
  --patch-size 100 \
  --max-patches-per-image 10 \
  --dry-run
```

Review:

- `outputs/ssl_patch_test/patch_manifest.tsv`
- `outputs/ssl_patch_test/ssl_patch_test_report.md`

## Real Patch Test

Run without `--dry-run` to save selected patches and compute interpretable
patch-level phenotype metrics:

```bash
python scripts/run_ssl_patch_test.py \
  --input-dir /Users/makennarodriguez/Desktop/procodes \
  --output-dir outputs/ssl_patch_test \
  --patch-size 100 \
  --max-patches-per-image 10
```

This run writes `patch_phenotypes.tsv`, which contains foreground fraction,
foreground/background intensity separation, connected-component summaries,
boundary fraction, elongation, gradient, and per-channel intensity metrics.
These are biologically interpretable patch-level QC features, not a replacement
for cell-wise morphology from validated masks.

For a real SSL result, pass the same model builder and checkpoint intended for
the Brieflow phenotype workflow:

```bash
python scripts/run_ssl_patch_test.py \
  --input-dir /Users/makennarodriguez/Desktop/procodes \
  --output-dir outputs/ssl_patch_test \
  --patch-size 200 \
  --max-patches-per-image 10 \
  --ssl-model-builder manuscript.models.vit:build_vit_backbone_tokens \
  --ssl-ckpt outputs/vit_models/ssl_pretraining/dino_best.pth \
  --ssl-device cpu
```

If no checkpoint is supplied, SSL embeddings are skipped by default so random
features are not mistaken for biology. To run a plumbing-only SSL smoke test
with a randomly initialized local ViT, add `--allow-random-ssl`.

## Outputs

The runner writes:

- `outputs/ssl_patch_test/patches/`: selected patch TIFFs from real runs.
- `outputs/ssl_patch_test/patch_manifest.tsv`: source image, patch coordinates,
  foreground fraction, intensity summary, and patch path.
- `outputs/ssl_patch_test/patch_phenotypes.tsv`: interpretable patch-level
  phenotype metrics suitable for preliminary morphology/intensity QC.
- `outputs/ssl_patch_test/ssl_patch_features.tsv`: per-patch SSL feature rows
  from the existing `extract_ssl_cell_embeddings` token pooling workflow when
  `--ssl-ckpt` or `--allow-random-ssl` is supplied.
- `outputs/ssl_patch_test/ssl_patch_test_report.md`: QC summary.
- `outputs/ssl_patch_test/open_patches_in_fiji.ijm`: run-specific Fiji/ImageJ
  macro for opening the patch TIFFs as native-pixel images.

## Fiji Review

Use Fiji for visual readout instead of judging morphology from PNG montages.
The patch runner writes TIFF patches that Fiji can open directly, preserving
native pixels, LUT/contrast controls, zoom, and measurement tools.

This integration is a Fiji macro, not a plugin. A macro is the right level for
reviewing local patch outputs because it is transparent, easy to edit, and does
not require packaging or installation. A plugin would make sense later only if
the project needs a custom interactive UI or a reusable packaged analysis tool.

Run-specific macro:

```text
outputs/ssl_patch_test/open_patches_in_fiji.ijm
```

Generic repo macro:

```text
scripts/fiji/open_ssl_patch_test_patches.ijm
```

In Fiji:

1. Open Fiji.
2. Use `File > Open...` and select the `.ijm` macro.
3. Click `Run`.
4. For the generic macro, choose the patch-test output folder.
5. Inspect the tiled TIFF patches with native zoom and brightness/contrast
   controls.

## Jupyter Interactive Review

For pannable/zoomable matplotlib output inside a notebook, install `ipympl` in
the same conda environment or virtual environment used by Jupyter:

```bash
pip install -e ".[notebook]"
```

or:

```bash
python -m pip install ipympl
```

Then add this setup cell near the top of the notebook:

```python
%matplotlib widget

from pathlib import Path
import sys

repo = Path("/Users/makennarodriguez/Documents/brieflow-procodes-ssl/ssl-vit-phenotyping")
sys.path.insert(0, str(repo / "scripts" / "notebooks"))

from ssl_patch_interactive_review import show_patch, scatter_patch_phenotypes

output_dir = repo / "outputs" / "ssl_patch_test_200"
```

Example interactive review cells:

```python
show_patch(output_dir, patch_index=0)
```

```python
scatter_patch_phenotypes(
    output_dir,
    x="phenotype_foreground_fraction",
    y="phenotype_largest_component_elongation",
)
```

Use the notebook path for interactive plots and quick metric exploration; use
Fiji when you need microscopy-native TIFF viewing, contrast/LUT controls, or
manual inspection of the raw patch pixels.

## Interpreting the QC Report

Use the report as a stop/continue gate:

- `Number of supported images found` should match the TIFF/PNG/JPEG files you
  expected to test. ND2, CZI, and LIF files should be converted to TIFF first.
- `Number of images successfully loaded` should be close to the number found.
  Failed images usually mean an unsupported shape, missing optional image reader,
  or corrupt file.
- `Number of patches extracted/planned` should be greater than zero. If it is
  zero, the images may be smaller than the patch size or mostly background.
- `Number of patches with interpretable phenotype metrics` should match planned
  patches for a real non-dry run.
- `Number of patches processed by SSL` should match planned patches only when a
  checkpoint was supplied, or when `--allow-random-ssl` was intentionally used.
- `Downstream Usability` should be read conservatively. Patch-level outputs are
  useful for smoke testing image loading and for preliminary intensity/morphology
  QC. They are not final evidence for neuron morphology or phenotype analysis
  unless validated masks, clean ProCode decoding, and a relevant trained
  checkpoint are used.

## Git Hygiene

Do not commit raw microscopy images, local test outputs, model checkpoints, or
generated patch TIFFs. These paths and file types should remain untracked:

- `data/`
- `outputs/`
- `*.nd2`
- `*.tif`
- `*.tiff`
- `*.czi`
- `*.lif`

Before committing, check:

```bash
git status --short
git status --ignored --short outputs data
```

Only commit code and documentation changes such as this runner, docs, and
`.gitignore` updates.

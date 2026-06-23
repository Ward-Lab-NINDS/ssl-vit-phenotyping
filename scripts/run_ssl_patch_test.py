#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tifffile import imread, imwrite

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MICROSCOPY_UNSUPPORTED_SUFFIXES = {".nd2", ".czi", ".lif"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    data_chw: np.ndarray
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PatchCandidate:
    patch_id: str
    source_path: Path
    source_name: str
    y: int
    x: int
    patch_size: int
    foreground_fraction: float
    intensity_mean: float
    intensity_std: float
    score: float
    selected_by_fallback: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local SSL smoke test on small microscopy image patches instead of "
            "whole fields of view."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing local images.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/ssl_patch_test"),
        type=Path,
        help="Folder for patch images, feature tables, and QC report.",
    )
    parser.add_argument(
        "--patch-size",
        required=True,
        type=int,
        choices=(100, 200),
        help="Square patch/tile size in pixels.",
    )
    parser.add_argument(
        "--max-patches-per-image",
        type=int,
        default=25,
        help="Maximum number of foreground-like patches to process per image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan patch extraction and write a QC report, but do not save patch images "
            "or run SSL."
        ),
    )
    parser.add_argument(
        "--ssl-model-builder",
        default="manuscript.models.vit:build_vit_backbone_tokens",
        help="module:function path for a token-exposing SSL/ViT model builder.",
    )
    parser.add_argument(
        "--ssl-ckpt",
        default=None,
        type=Path,
        help="SSL checkpoint for biologically interpretable embedding extraction.",
    )
    parser.add_argument(
        "--allow-random-ssl",
        action="store_true",
        help=(
            "Allow randomly initialized SSL extraction when no checkpoint is supplied. "
            "Use only for plumbing smoke tests, not biology."
        ),
    )
    parser.add_argument(
        "--ssl-device",
        default="cpu",
        help="Torch device for SSL extraction. CPU is recommended for small local smoke tests.",
    )
    parser.add_argument(
        "--ssl-vit-patch-size",
        default=10,
        type=int,
        help="Internal ViT token patch size. Must divide --patch-size for the default tiny ViT.",
    )
    parser.add_argument(
        "--ssl-model-kwargs-json",
        default=None,
        help="Optional JSON object of extra keyword args for the SSL model builder.",
    )
    parser.add_argument(
        "--ssl-use-channels",
        default=None,
        help="Comma-separated channel indexes to pass to SSL extraction. Default: all channels.",
    )
    parser.add_argument(
        "--normalization",
        default="zscore",
        choices=("zscore", "minmax", "none"),
        help="Per-channel normalization passed to SSL preprocessing.",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=("mean", "mean_std", "median", "max", "trimmed_mean"),
        help="Token pooling method used by the existing SSL feature extractor.",
    )
    parser.add_argument(
        "--foreground-min-fraction",
        default=0.01,
        type=float,
        help="Minimum foreground fraction for a patch to be considered non-empty.",
    )
    return parser.parse_args()


def parse_use_channels(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_builder_kwargs(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--ssl-model-kwargs-json must be a JSON object")
    return parsed


def find_image_paths(input_dir: Path) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    warnings: list[str] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_SUFFIXES:
            paths.append(path)
        elif suffix in MICROSCOPY_UNSUPPORTED_SUFFIXES:
            warnings.append(
                f"Skipped unsupported microscopy format {path.name!r}. Convert to TIFF first."
            )
    return paths, warnings


def read_png_jpeg(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{path.suffix.lower()} input requires Pillow. TIFF inputs work with tifffile."
        ) from exc
    with Image.open(path) as img:
        return np.asarray(img)


def load_image(path: Path) -> ImageRecord:
    warnings: list[str] = []
    if path.suffix.lower() in {".tif", ".tiff"}:
        data = imread(path)
    else:
        data = read_png_jpeg(path)
    data_chw, shape_warnings = to_channel_first(data)
    warnings.extend(shape_warnings)
    if data_chw.shape[1] == 0 or data_chw.shape[2] == 0:
        raise ValueError("image has zero height or width")
    if not np.isfinite(data_chw.astype(np.float32, copy=False)).any():
        raise ValueError("image does not contain finite numeric pixels")
    return ImageRecord(path=path, data_chw=data_chw, warnings=tuple(warnings))


def to_channel_first(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    data = np.asarray(data)
    data = np.squeeze(data)
    warnings: list[str] = []

    if data.ndim == 2:
        return data[None, :, :], warnings

    if data.ndim == 3:
        if data.shape[-1] <= 8 and data.shape[0] > 8 and data.shape[1] > 8:
            return np.moveaxis(data, -1, 0), warnings
        if data.shape[0] <= 8:
            return data, warnings
        warnings.append(
            f"Interpreted 3D shape {tuple(data.shape)} as a stack and max-projected it."
        )
        return np.nanmax(data, axis=0, keepdims=True), warnings

    if data.ndim == 4:
        channel_axes = [axis for axis, size in enumerate(data.shape) if size <= 8]
        if channel_axes:
            channel_axis = channel_axes[0]
            moved = np.moveaxis(data, channel_axis, 0)
            while moved.ndim > 3:
                moved = np.nanmax(moved, axis=1)
            warnings.append(
                f"Max-projected non-channel axes from 4D shape {tuple(data.shape)}."
            )
            return moved, warnings

    raise ValueError(
        f"unsupported image shape {tuple(data.shape)}; expected 2D, channel-first/last "
        "3D, or simple 4D"
    )


def robust_projection(data_chw: np.ndarray) -> np.ndarray:
    x = data_chw.astype(np.float32, copy=False)
    projection_channels: list[np.ndarray] = []
    for channel in x:
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            projection_channels.append(np.zeros(channel.shape, dtype=np.float32))
            continue
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            scaled = np.zeros(channel.shape, dtype=np.float32)
        else:
            scaled = np.clip((channel - low) / (high - low), 0, 1)
        projection_channels.append(scaled.astype(np.float32, copy=False))
    return np.nanmax(np.stack(projection_channels, axis=0), axis=0)


def foreground_mask_from_projection(projection: np.ndarray) -> np.ndarray:
    finite = projection[np.isfinite(projection)]
    if finite.size == 0:
        return np.zeros(projection.shape, dtype=bool)
    low, high = np.percentile(finite, [10, 99])
    if high <= low:
        return np.zeros(projection.shape, dtype=bool)
    threshold = low + 0.05 * (high - low)
    return projection > threshold


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "image"


def select_patches(
    record: ImageRecord,
    patch_size: int,
    max_patches: int,
    foreground_min_fraction: float,
) -> tuple[list[PatchCandidate], list[str]]:
    data_chw = record.data_chw
    height, width = data_chw.shape[-2:]
    warnings: list[str] = []
    if height < patch_size or width < patch_size:
        return [], [
            f"{record.path.name}: image is smaller than {patch_size}x{patch_size}; skipped."
        ]

    projection = robust_projection(data_chw)
    foreground = foreground_mask_from_projection(projection)
    candidates: list[PatchCandidate] = []
    fallback_candidates: list[PatchCandidate] = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch_projection = projection[y : y + patch_size, x : x + patch_size]
            patch_foreground = foreground[y : y + patch_size, x : x + patch_size]
            foreground_fraction = float(np.mean(patch_foreground))
            intensity_mean = float(np.nanmean(patch_projection))
            intensity_std = float(np.nanstd(patch_projection))
            score = foreground_fraction * max(intensity_std, 1e-6)
            patch_id = f"{safe_stem(record.path)}__y{y:05d}_x{x:05d}_s{patch_size}"
            candidate = PatchCandidate(
                patch_id=patch_id,
                source_path=record.path,
                source_name=record.path.name,
                y=y,
                x=x,
                patch_size=patch_size,
                foreground_fraction=foreground_fraction,
                intensity_mean=intensity_mean,
                intensity_std=intensity_std,
                score=score,
            )
            fallback_candidates.append(candidate)
            if foreground_fraction >= foreground_min_fraction and intensity_std > 1e-4:
                candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates and fallback_candidates:
        warnings.append(
            f"{record.path.name}: no patches passed foreground threshold; "
            "selected best-scoring fallback."
        )
        fallback_candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = [
            PatchCandidate(
                **{
                    **fallback_candidates[0].__dict__,
                    "selected_by_fallback": True,
                }
            )
        ]
    return candidates[:max_patches], warnings


def patch_array(record: ImageRecord, candidate: PatchCandidate) -> np.ndarray:
    y, x, size = candidate.y, candidate.x, candidate.patch_size
    return record.data_chw[:, y : y + size, x : x + size]


def labels_for_patch(patch_chw: np.ndarray) -> np.ndarray:
    projection = robust_projection(patch_chw)
    labels = foreground_mask_from_projection(projection).astype(np.uint16)
    if labels.sum() == 0:
        labels[:, :] = 1
    return labels


def connected_component_areas(mask: np.ndarray) -> list[int]:
    visited = np.zeros(mask.shape, dtype=bool)
    areas: list[int] = []
    height, width = mask.shape
    starts = np.argwhere(mask)
    for row, col in starts:
        row = int(row)
        col = int(col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        areas.append(area)
    return areas


def largest_component_coordinates(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    starts = np.argwhere(mask)
    for row, col in starts:
        row = int(row)
        col = int(col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        coords: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        if len(coords) > len(best):
            best = coords
    if not best:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(best, dtype=np.float32)


def foreground_boundary_fraction(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior_neighbors = (
        padded[0:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, 0:-2]
        & padded[1:-1, 2:]
    )
    boundary = mask & ~interior_neighbors
    return float(boundary.sum() / max(mask.sum(), 1))


def component_elongation(coords_yx: np.ndarray) -> float:
    if coords_yx.shape[0] < 3:
        return 0.0
    centered = coords_yx - coords_yx.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    smallest = max(float(eigvals[0]), 1e-6)
    largest = max(float(eigvals[-1]), 1e-6)
    return float(largest / smallest)


def gradient_mean(projection: np.ndarray) -> float:
    grad_y, grad_x = np.gradient(projection.astype(np.float32, copy=False))
    magnitude = np.sqrt((grad_y * grad_y) + (grad_x * grad_x))
    return float(np.nanmean(magnitude))


def patch_phenotype_row(
    patch_chw: np.ndarray,
    candidate: PatchCandidate,
    patch_path: Path | None,
) -> dict[str, Any]:
    projection = robust_projection(patch_chw)
    mask = foreground_mask_from_projection(projection)
    foreground_values = projection[mask]
    background_values = projection[~mask]
    areas = connected_component_areas(mask)
    largest_coords = largest_component_coordinates(mask)
    largest_area = int(max(areas)) if areas else 0
    foreground_area = int(mask.sum())
    patch_area = int(mask.size)

    row: dict[str, Any] = {
        "patch_id": candidate.patch_id,
        "source_image": candidate.source_name,
        "source_path": str(candidate.source_path),
        "patch_path": str(patch_path) if patch_path else "",
        "y": candidate.y,
        "x": candidate.x,
        "patch_size": candidate.patch_size,
        "phenotype_foreground_area_px": foreground_area,
        "phenotype_foreground_fraction": float(foreground_area / patch_area),
        "phenotype_projection_mean": float(np.nanmean(projection)),
        "phenotype_projection_std": float(np.nanstd(projection)),
        "phenotype_foreground_mean": (
            float(np.nanmean(foreground_values)) if foreground_values.size else 0.0
        ),
        "phenotype_foreground_std": (
            float(np.nanstd(foreground_values)) if foreground_values.size else 0.0
        ),
        "phenotype_background_mean": (
            float(np.nanmean(background_values)) if background_values.size else 0.0
        ),
        "phenotype_foreground_background_delta": (
            float(np.nanmean(foreground_values) - np.nanmean(background_values))
            if foreground_values.size and background_values.size
            else 0.0
        ),
        "phenotype_connected_component_count": len(areas),
        "phenotype_largest_component_area_px": largest_area,
        "phenotype_largest_component_fraction": float(largest_area / patch_area),
        "phenotype_median_component_area_px": float(np.median(areas)) if areas else 0.0,
        "phenotype_boundary_fraction": foreground_boundary_fraction(mask),
        "phenotype_largest_component_elongation": component_elongation(largest_coords),
        "phenotype_gradient_mean": gradient_mean(projection),
    }

    for channel_index, channel in enumerate(patch_chw.astype(np.float32, copy=False)):
        row[f"phenotype_channel_{channel_index:02d}_mean"] = float(np.nanmean(channel))
        row[f"phenotype_channel_{channel_index:02d}_std"] = float(np.nanstd(channel))
        if foreground_area:
            row[f"phenotype_channel_{channel_index:02d}_foreground_mean"] = float(
                np.nanmean(channel[mask])
            )
        else:
            row[f"phenotype_channel_{channel_index:02d}_foreground_mean"] = 0.0
    return row


def save_patch(path: Path, patch_chw: np.ndarray) -> None:
    if patch_chw.shape[0] == 1:
        imwrite(path, patch_chw[0])
    else:
        imwrite(path, patch_chw, metadata={"axes": "CYX"})


def imagej_escape(value: str | Path) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def fiji_macro_text(output_dir: Path, max_images: int = 24) -> str:
    patch_dir = output_dir / "patches"
    manifest_path = output_dir / "patch_manifest.tsv"
    phenotype_path = output_dir / "patch_phenotypes.tsv"
    return f'''// Auto-generated by scripts/run_ssl_patch_test.py.
// Open in Fiji/ImageJ with File > Open, then click Run.
// It opens a limited number of patch TIFFs as native-pixel images and tiles them.

patchDir = "{imagej_escape(patch_dir)}/";
manifestPath = "{imagej_escape(manifest_path)}";
phenotypePath = "{imagej_escape(phenotype_path)}";
maxImages = {int(max_images)};

print("SSL patch test patch folder: " + patchDir);
print("Patch manifest: " + manifestPath);
print("Patch phenotype table: " + phenotypePath);

list = getFileList(patchDir);
opened = 0;
setBatchMode(true);
for (i = 0; i < list.length && opened < maxImages; i++) {{
    if (!endsWith(list[i], ".tif") && !endsWith(list[i], ".tiff")) {{
        continue;
    }}
    open(patchDir + list[i]);
    run("Enhance Contrast", "saturated=0.35 normalize");
    run("Grays");
    rename(list[i]);
    opened++;
}}
setBatchMode(false);
if (opened > 0) {{
    run("Tile");
}} else {{
    showMessage("No patch TIFFs found", "No .tif or .tiff files were found in:\\n" + patchDir);
}}
'''


def write_fiji_macro(output_dir: Path) -> Path:
    macro_path = output_dir / "open_patches_in_fiji.ijm"
    macro_path.write_text(fiji_macro_text(output_dir), encoding="utf-8")
    return macro_path


def default_builder_kwargs(
    model_builder: str,
    in_channels: int,
    patch_size: int,
    ssl_vit_patch_size: int,
    user_kwargs: dict[str, Any],
) -> dict[str, Any]:
    kwargs = dict(user_kwargs)
    if model_builder == "manuscript.models.vit:build_vit_backbone_tokens":
        kwargs.setdefault("in_channels", in_channels)
        kwargs.setdefault("image_size", patch_size)
        kwargs.setdefault("patch_size", ssl_vit_patch_size)
        kwargs.setdefault("embed_dim", 64)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("num_heads", 4)
    return kwargs


def load_model_cached(
    cache: dict[int, Any],
    in_channels: int,
    args: argparse.Namespace,
    user_builder_kwargs: dict[str, Any],
    warnings: list[str],
) -> Any:
    if in_channels in cache:
        return cache[in_channels]
    device = args.ssl_device
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                warnings.append("CUDA was requested but is unavailable; falling back to CPU.")
                device = "cpu"
        except Exception:
            warnings.append("Could not verify CUDA availability; falling back to CPU.")
            device = "cpu"
    kwargs = default_builder_kwargs(
        args.ssl_model_builder,
        in_channels=in_channels,
        patch_size=args.patch_size,
        ssl_vit_patch_size=args.ssl_vit_patch_size,
        user_kwargs=user_builder_kwargs,
    )
    from lib.phenotype.ssl_model_loader import load_ssl_vit_model

    model = load_ssl_vit_model(
        ckpt_path=args.ssl_ckpt,
        model_builder_path=args.ssl_model_builder,
        device=device,
        strict=False,
        builder_kwargs=kwargs,
    )
    cache[in_channels] = (model, device)
    return cache[in_channels]


def manifest_row(candidate: PatchCandidate, patch_path: Path | None) -> dict[str, Any]:
    return {
        "patch_id": candidate.patch_id,
        "source_image": candidate.source_name,
        "source_path": str(candidate.source_path),
        "patch_path": str(patch_path) if patch_path else "",
        "y": candidate.y,
        "x": candidate.x,
        "patch_size": candidate.patch_size,
        "foreground_fraction": candidate.foreground_fraction,
        "intensity_mean": candidate.intensity_mean,
        "intensity_std": candidate.intensity_std,
        "selection_score": candidate.score,
        "selected_by_fallback": candidate.selected_by_fallback,
    }


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    image_paths: list[Path],
    loaded_count: int,
    planned_patches: int,
    phenotype_patches: int,
    processed_patches: int,
    failed_images: list[str],
    warnings: list[str],
    dry_run: bool,
) -> None:
    if dry_run:
        usability = (
            "Not assessed. This was a dry run, so patches were not saved and SSL "
            "features were not extracted."
        )
    elif processed_patches == 0:
        usability = (
            "Partially. Interpretable patch phenotype metrics were written, but no SSL "
            "embeddings were produced. Use the patch phenotype table for preliminary "
            "morphology/intensity QC, and provide a trained checkpoint before treating "
            "SSL embeddings as biologically meaningful."
        )
    elif warnings:
        usability = (
            "Partially. The outputs can support a patch-level SSL plumbing smoke test, "
            "but warnings should be reviewed before treating them as usable for neuron "
            "morphology or phenotype analysis."
        )
    else:
        usability = (
            "Yes for a small patch-level SSL smoke test. Treat this as a "
            "preprocessing/feature-extraction check, not final neuron morphology evidence "
            "unless the run used validated masks and a relevant trained checkpoint."
        )

    lines = [
        "# SSL Patch Test Report",
        "",
        f"- Input folder used: `{args.input_dir}`",
        f"- Output folder: `{args.output_dir}`",
        f"- Patch size used: `{args.patch_size}`",
        f"- Maximum patches per image: `{args.max_patches_per_image}`",
        f"- Dry run: `{dry_run}`",
        f"- Number of supported images found: `{len(image_paths)}`",
        f"- Number of images successfully loaded: `{loaded_count}`",
        f"- Number of patches extracted/planned: `{planned_patches}`",
        f"- Number of patches with interpretable phenotype metrics: `{phenotype_patches}`",
        f"- Number of patches processed by SSL: `{processed_patches}`",
        f"- Patch images folder: `{args.output_dir / 'patches'}`",
        f"- Patch manifest: `{args.output_dir / 'patch_manifest.tsv'}`",
        f"- Patch phenotype table: `{args.output_dir / 'patch_phenotypes.tsv'}`",
        f"- SSL feature table: `{args.output_dir / 'ssl_patch_features.tsv'}`",
        f"- Fiji review macro: `{args.output_dir / 'open_patches_in_fiji.ijm'}`",
        "",
        "## Downstream Usability",
        "",
        usability,
        "",
        "## Failed Images",
        "",
    ]
    if failed_images:
        lines.extend(f"- {item}" for item in failed_images)
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Empty/background-like patches are filtered using robust per-channel "
            "intensity projection.",
            "- In the absence of full cell masks, each selected patch is processed with "
            "a simple foreground mask.",
            "- If no checkpoint is supplied, SSL is skipped unless --allow-random-ssl "
            "is explicitly set for a plumbing smoke test.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_patches_per_image < 1:
        raise ValueError("--max-patches-per-image must be at least 1")
    if args.patch_size % args.ssl_vit_patch_size != 0:
        raise ValueError("--ssl-vit-patch-size must divide --patch-size")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    args.input_dir = input_dir
    args.output_dir = output_dir

    image_paths, warnings = find_image_paths(input_dir)
    failed_images: list[str] = []
    records: list[ImageRecord] = []
    for path in image_paths:
        try:
            record = load_image(path)
            records.append(record)
            warnings.extend(f"{path.name}: {warning}" for warning in record.warnings)
        except Exception as exc:
            failed_images.append(f"{path}: {exc}")

    candidates_by_path: dict[Path, list[PatchCandidate]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for record in records:
        candidates, patch_warnings = select_patches(
            record,
            patch_size=args.patch_size,
            max_patches=args.max_patches_per_image,
            foreground_min_fraction=args.foreground_min_fraction,
        )
        candidates_by_path[record.path] = candidates
        warnings.extend(patch_warnings)
        manifest_rows.extend(manifest_row(candidate, None) for candidate in candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    fiji_macro_path = write_fiji_macro(output_dir)
    patches_dir = output_dir / "patches"
    processed_patches = 0
    phenotype_rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    run_ssl = not args.dry_run and (args.ssl_ckpt is not None or args.allow_random_ssl)

    if not args.dry_run and not run_ssl:
        warnings.append(
            "No SSL checkpoint supplied; skipped SSL embeddings. Wrote interpretable "
            "patch phenotype metrics instead. Pass --ssl-ckpt for meaningful SSL, or "
            "--allow-random-ssl for a plumbing-only smoke test."
        )

    if run_ssl:
        try:
            from lib.phenotype.ssl_cell_features import extract_ssl_cell_embeddings
        except ModuleNotFoundError as exc:
            failed_images.append(
                f"SSL dependencies are missing ({exc}). Install the project environment "
                "before a real run."
            )
            pd.DataFrame(manifest_rows).to_csv(
                output_dir / "patch_manifest.tsv", sep="\t", index=False
            )
            write_report(
                report_path=output_dir / "ssl_patch_test_report.md",
                args=args,
                image_paths=image_paths,
                loaded_count=len(records),
                planned_patches=sum(len(items) for items in candidates_by_path.values()),
                phenotype_patches=0,
                processed_patches=0,
                failed_images=failed_images,
                warnings=warnings,
                dry_run=args.dry_run,
            )
            print(f"Report: {output_dir / 'ssl_patch_test_report.md'}")
            return 1

        patches_dir.mkdir(parents=True, exist_ok=True)
        user_builder_kwargs = parse_builder_kwargs(args.ssl_model_kwargs_json)
        use_channels = parse_use_channels(args.ssl_use_channels)
        model_cache: dict[int, Any] = {}
        manifest_rows = []
        record_by_path = {record.path: record for record in records}

        for source_path, candidates in candidates_by_path.items():
            record = record_by_path[source_path]
            for candidate in candidates:
                patch = patch_array(record, candidate)
                patch_path = patches_dir / f"{candidate.patch_id}.tif"
                save_patch(patch_path, patch)
                manifest_rows.append(manifest_row(candidate, patch_path))
                phenotype_rows.append(patch_phenotype_row(patch, candidate, patch_path))
                try:
                    model, device = load_model_cached(
                        model_cache,
                        in_channels=patch.shape[0],
                        args=args,
                        user_builder_kwargs=user_builder_kwargs,
                        warnings=warnings,
                    )
                    ssl_df = extract_ssl_cell_embeddings(
                        data_phenotype=patch,
                        cells=labels_for_patch(patch),
                        model=model,
                        device=device,
                        patch_size=args.ssl_vit_patch_size,
                        use_channels=use_channels,
                        prefix="ssl",
                        pooling=args.pooling,
                        normalization=args.normalization,
                    )
                    if ssl_df.empty:
                        warnings.append(f"{candidate.patch_id}: SSL extraction returned no rows.")
                        continue
                    ssl_df.insert(0, "patch_id", candidate.patch_id)
                    ssl_df.insert(1, "source_image", candidate.source_name)
                    ssl_df.insert(2, "patch_path", str(patch_path))
                    ssl_df["meta_patch_size"] = args.patch_size
                    ssl_df["meta_ssl_vit_patch_size"] = args.ssl_vit_patch_size
                    ssl_df["meta_ssl_model_builder"] = args.ssl_model_builder
                    ssl_df["meta_ssl_checkpoint"] = str(args.ssl_ckpt) if args.ssl_ckpt else "none"
                    feature_rows.append(ssl_df)
                    processed_patches += 1
                except Exception as exc:
                    failed_images.append(f"{candidate.patch_id}: SSL processing failed: {exc}")
                    warnings.append(traceback.format_exc(limit=1).strip())

    elif not args.dry_run:
        patches_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        record_by_path = {record.path: record for record in records}
        for source_path, candidates in candidates_by_path.items():
            record = record_by_path[source_path]
            for candidate in candidates:
                patch = patch_array(record, candidate)
                patch_path = patches_dir / f"{candidate.patch_id}.tif"
                save_patch(patch_path, patch)
                manifest_rows.append(manifest_row(candidate, patch_path))
                phenotype_rows.append(patch_phenotype_row(patch, candidate, patch_path))

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "patch_manifest.tsv", sep="\t", index=False)
    if phenotype_rows:
        pd.DataFrame(phenotype_rows).to_csv(
            output_dir / "patch_phenotypes.tsv", sep="\t", index=False
        )
    elif not args.dry_run:
        pd.DataFrame().to_csv(output_dir / "patch_phenotypes.tsv", sep="\t", index=False)
    if feature_rows:
        pd.concat(feature_rows, ignore_index=True).to_csv(
            output_dir / "ssl_patch_features.tsv", sep="\t", index=False
        )
    elif not args.dry_run:
        pd.DataFrame().to_csv(output_dir / "ssl_patch_features.tsv", sep="\t", index=False)

    write_report(
        report_path=output_dir / "ssl_patch_test_report.md",
        args=args,
        image_paths=image_paths,
        loaded_count=len(records),
        planned_patches=sum(len(items) for items in candidates_by_path.values()),
        phenotype_patches=len(phenotype_rows),
        processed_patches=processed_patches,
        failed_images=failed_images,
        warnings=warnings,
        dry_run=args.dry_run,
    )

    print(f"Images found: {len(image_paths)}")
    print(f"Images loaded: {len(records)}")
    print(f"Patches planned: {sum(len(items) for items in candidates_by_path.values())}")
    print(f"Patches with phenotype metrics: {len(phenotype_rows)}")
    print(f"Patches processed by SSL: {processed_patches}")
    print(f"Fiji macro: {fiji_macro_path}")
    print(f"Report: {output_dir / 'ssl_patch_test_report.md'}")
    return 0 if not failed_images else 1


if __name__ == "__main__":
    raise SystemExit(main())

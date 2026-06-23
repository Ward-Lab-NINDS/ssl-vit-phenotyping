from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import imread


def normalize_patch(array: np.ndarray) -> np.ndarray:
    image = np.squeeze(np.asarray(array))
    if image.ndim == 3:
        image = np.nanmax(image, axis=0 if image.shape[0] <= 8 else -1)
    image = image.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(finite, [0.2, 99.8])
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1)


def foreground_mask(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=bool)
    low, high = np.percentile(finite, [10, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=bool)
    return image > (low + 0.05 * (high - low))


def load_patch_test(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    manifest = pd.read_csv(output_dir / "patch_manifest.tsv", sep="\t")
    phenotypes = pd.read_csv(output_dir / "patch_phenotypes.tsv", sep="\t")
    return manifest, phenotypes


def show_patch(
    output_dir: str | Path,
    patch_index: int = 0,
    overlay: bool = True,
    cmap: str = "gray",
) -> None:
    manifest, phenotypes = load_patch_test(output_dir)
    row = manifest.iloc[patch_index]
    patch = normalize_patch(imread(row["patch_path"]))
    phenotype = phenotypes.set_index("patch_id").loc[row["patch_id"]]

    fig, axes = plt.subplots(1, 3 if overlay else 1, figsize=(12 if overlay else 5, 4))
    if not overlay:
        axes = [axes]

    axes[0].imshow(patch, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(row["patch_id"])

    if overlay:
        mask = foreground_mask(patch)
        rgb = np.dstack([patch, patch, patch])
        rgb[..., 0] = np.maximum(rgb[..., 0], mask.astype(float))
        rgb[..., 1] *= np.where(mask, 0.35, 1.0)
        rgb[..., 2] *= np.where(mask, 0.35, 1.0)
        axes[1].imshow(mask, cmap="gray", interpolation="nearest")
        axes[1].set_title("Foreground mask")
        axes[2].imshow(rgb, vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title(
            "fg={:.2f}, objects={}, elong={:.1f}".format(
                phenotype["phenotype_foreground_fraction"],
                int(phenotype["phenotype_connected_component_count"]),
                phenotype["phenotype_largest_component_elongation"],
            )
        )

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()


def scatter_patch_phenotypes(
    output_dir: str | Path,
    x: str = "phenotype_foreground_fraction",
    y: str = "phenotype_largest_component_elongation",
) -> None:
    _, phenotypes = load_patch_test(output_dir)
    plot_data = phenotypes.copy()
    inferred = plot_data["source_image"].str.extract(r"-(ch\d+)")[0]
    plot_data["channel"] = inferred.fillna("mapping_unknown").map(
        lambda value: f"{value}_mapping_unknown" if str(value).startswith("ch") else value
    )

    fig, axis = plt.subplots(figsize=(7, 5))
    for channel, part in plot_data.groupby("channel"):
        axis.scatter(part[x], part[y], label=channel, alpha=0.8)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title("Patch phenotype scatter; channel metadata required for V5/NWS/T7/nucleus labels")
    axis.grid(alpha=0.25)
    axis.legend(title="Channel")
    fig.tight_layout()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class PCAProjection:
    mean: np.ndarray
    components: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "PCAProjection":
        data = np.load(path)
        if "mean" not in data or "components" not in data:
            raise ValueError("PCA basis must contain 'mean' and 'components' arrays")
        return cls(mean=data["mean"].astype(np.float32), components=data["components"].astype(np.float32))

    def transform(self, x: np.ndarray, out_dim: int | None = None) -> np.ndarray:
        components = self.components
        if out_dim is not None:
            components = components[:out_dim]
        if x.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"PCA basis expects {self.mean.shape[0]} features, got {x.shape[1]}"
            )
        return ((x - self.mean[None, :]) @ components.T).astype(np.float32)


def ensure_float_image(
    image: np.ndarray,
    normalization: str = "zscore",
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert `[C,H,W]` image data to float and normalize per channel."""
    x = image.astype(np.float32, copy=False)
    if x.size == 0:
        raise ValueError("Image array is empty")
    if x.max() > 1.5:
        x = x / np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else x / x.max()

    if normalization == "none":
        return x
    if normalization == "minmax":
        mins = x.reshape(x.shape[0], -1).min(axis=1)[:, None, None]
        maxs = x.reshape(x.shape[0], -1).max(axis=1)[:, None, None]
        return (x - mins) / np.maximum(maxs - mins, eps)
    if normalization == "zscore":
        means = x.reshape(x.shape[0], -1).mean(axis=1)[:, None, None]
        stds = x.reshape(x.shape[0], -1).std(axis=1)[:, None, None]
        return (x - means) / np.maximum(stds, eps)

    raise ValueError("normalization must be one of: 'zscore', 'minmax', 'none'")


def infer_patch_grid(n_tokens: int, image_hw: tuple[int, int], patch_size: int | None = None) -> tuple[int, int]:
    """Infer a 2D patch grid from token count and image shape."""
    if patch_size:
        grid = (image_hw[0] // patch_size, image_hw[1] // patch_size)
        if grid[0] * grid[1] == n_tokens:
            return grid

    aspect = image_hw[0] / max(image_hw[1], 1)
    best: tuple[int, int] | None = None
    best_score = float("inf")
    for h in range(1, int(np.sqrt(n_tokens)) + 2):
        if n_tokens % h != 0:
            continue
        w = n_tokens // h
        for candidate in ((h, w), (w, h)):
            score = abs((candidate[0] / candidate[1]) - aspect)
            if score < best_score:
                best = candidate
                best_score = score
    if best is None:
        return (1, n_tokens)
    return best


def labels_to_patch_grid(labels_hw: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Map integer labels to patch coordinates by nearest-neighbor sampling."""
    if labels_hw.ndim != 2:
        raise ValueError(f"Expected labels [H,W], got {labels_hw.shape}")
    rows = np.linspace(0, labels_hw.shape[0] - 1, grid_hw[0]).round().astype(np.int64)
    cols = np.linspace(0, labels_hw.shape[1] - 1, grid_hw[1]).round().astype(np.int64)
    return labels_hw[np.ix_(rows, cols)].astype(np.int32, copy=False)


@torch.no_grad()
def get_patch_tokens(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Return patch tokens with shape `[B,N,D]` from common ViT APIs."""
    if hasattr(model, "forward_features"):
        features = model.forward_features(image)
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
                tokens = features.get(key)
                if isinstance(tokens, torch.Tensor) and tokens.ndim == 3:
                    return tokens
        if isinstance(features, torch.Tensor) and features.ndim == 3:
            return features

    try:
        tokens = model(image, return_tokens=True)
    except TypeError:
        tokens = None
    if isinstance(tokens, torch.Tensor) and tokens.ndim == 3:
        return tokens

    raise RuntimeError(
        "Model does not expose patch tokens. Add forward_features() returning "
        "`x_norm_patchtokens` or support forward(x, return_tokens=True)."
    )


def pool_tokens_by_label(
    tokens: torch.Tensor,
    labels_grid: np.ndarray,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Pool `[N,D]` patch tokens by integer cell labels."""
    if tokens.ndim != 2:
        raise ValueError(f"Expected tokens [N,D], got {tuple(tokens.shape)}")
    flat_labels = labels_grid.reshape(-1).astype(np.int32)
    if flat_labels.size != tokens.shape[0]:
        raise ValueError(
            f"Label grid has {flat_labels.size} entries but token grid has {tokens.shape[0]} tokens"
        )

    labels = np.unique(flat_labels)
    labels = labels[labels != 0]
    if labels.size == 0:
        return labels.astype(np.int32), np.zeros((0, tokens.shape[1]), dtype=np.float32)

    tokens_cpu = tokens.detach().cpu()
    features: list[np.ndarray] = []
    for label in labels:
        idx = np.flatnonzero(flat_labels == label)
        selected = tokens_cpu[idx]
        mean = selected.mean(dim=0)
        if pooling == "mean":
            pooled = mean
        elif pooling == "mean_std":
            pooled = torch.cat([mean, selected.std(dim=0, unbiased=False)], dim=0)
        else:
            raise ValueError("pooling must be one of: 'mean', 'mean_std'")
        features.append(pooled.numpy())

    return labels.astype(np.int32), np.vstack(features).astype(np.float32)


@torch.no_grad()
def extract_ssl_cell_embeddings(
    data_phenotype: np.ndarray,
    cells: np.ndarray,
    model: torch.nn.Module,
    device: str = "cuda",
    patch_size: int | None = 8,
    use_channels: list[int] | None = None,
    prefix: str = "ssl",
    pca_dim: int | None = None,
    pca_basis_path: str | Path | None = None,
    pooling: str = "mean",
    normalization: str = "zscore",
    wildcards: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract per-cell SSL descriptors by pooling ViT patch tokens inside labels."""
    del wildcards
    if data_phenotype.ndim != 3:
        raise ValueError(f"Expected data_phenotype [C,H,W], got {data_phenotype.shape}")
    if cells.ndim != 2:
        raise ValueError(f"Expected cells [H,W], got {cells.shape}")
    if data_phenotype.shape[-2:] != cells.shape:
        raise ValueError(
            f"Image and cell labels must share H,W. Got {data_phenotype.shape[-2:]} and {cells.shape}"
        )
    if pca_dim is not None and pca_basis_path is None:
        raise ValueError(
            "ssl_pca_dim requires ssl_pca_basis_path. Per-tile PCA is intentionally disabled "
            "so embeddings remain comparable across images."
        )

    image = data_phenotype[use_channels] if use_channels is not None else data_phenotype
    image = ensure_float_image(image, normalization=normalization)
    image_t = torch.from_numpy(image[None, ...]).to(device)

    model = model.to(device).eval()
    tokens = get_patch_tokens(model, image_t)[0]
    grid_hw = infer_patch_grid(tokens.shape[0], image_hw=cells.shape, patch_size=patch_size)
    labels_grid = labels_to_patch_grid(cells, grid_hw)
    labels, features = pool_tokens_by_label(tokens, labels_grid, pooling=pooling)

    if labels.size == 0:
        return pd.DataFrame({"label": pd.Series(dtype=np.int32)})

    feature_name = prefix
    if pca_basis_path is not None:
        projection = PCAProjection.load(pca_basis_path)
        features = projection.transform(features, out_dim=pca_dim)
        feature_name = f"{prefix}_pca"

    columns = [f"{feature_name}_{index:03d}" for index in range(features.shape[1])]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "label", labels)
    return df

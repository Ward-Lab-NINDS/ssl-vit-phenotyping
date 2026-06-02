from __future__ import annotations

import numpy as np
import pytest
import torch

from lib.phenotype.ssl_cell_features import (
    extract_ssl_cell_embeddings,
    infer_patch_grid,
    labels_to_patch_grid,
)
from manuscript.models.vit import build_vit_backbone_tokens


def test_infer_patch_grid_uses_patch_size_for_rectangular_image() -> None:
    assert infer_patch_grid(32, image_hw=(32, 64), patch_size=8) == (4, 8)


def test_labels_to_patch_grid_preserves_cell_ids() -> None:
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[:4, :4] = 1
    labels[4:, 4:] = 2

    grid = labels_to_patch_grid(labels, (2, 2))

    assert set(grid.reshape(-1)) == {0, 1, 2}


def test_extract_ssl_cell_embeddings_returns_one_row_per_visible_cell() -> None:
    rng = np.random.default_rng(7)
    image = rng.normal(size=(2, 32, 32)).astype(np.float32)
    cells = np.zeros((32, 32), dtype=np.uint16)
    cells[:16, :16] = 1
    cells[:16, 16:] = 2
    cells[16:, :16] = 3

    model = build_vit_backbone_tokens(
        in_channels=2,
        image_size=32,
        patch_size=8,
        embed_dim=16,
        depth=1,
        num_heads=2,
    )

    df = extract_ssl_cell_embeddings(
        data_phenotype=image,
        cells=cells,
        model=model,
        device="cpu",
        patch_size=8,
        pca_dim=None,
    )

    assert df["label"].tolist() == [1, 2, 3]
    assert df.filter(regex=r"^ssl_").shape[1] == 16
    assert np.isfinite(df.filter(regex=r"^ssl_").to_numpy()).all()


def test_pca_dim_without_global_basis_fails_fast() -> None:
    image = np.ones((2, 32, 32), dtype=np.float32)
    cells = np.ones((32, 32), dtype=np.uint16)
    model = build_vit_backbone_tokens(
        in_channels=2,
        image_size=32,
        patch_size=8,
        embed_dim=16,
        depth=1,
        num_heads=2,
    )

    with pytest.raises(ValueError, match="ssl_pca_basis_path"):
        extract_ssl_cell_embeddings(
            data_phenotype=image,
            cells=cells,
            model=model,
            device="cpu",
            patch_size=8,
            pca_dim=8,
        )


def test_tiny_vit_exposes_patch_tokens() -> None:
    model = build_vit_backbone_tokens(
        in_channels=2,
        image_size=32,
        patch_size=8,
        embed_dim=16,
        depth=1,
        num_heads=2,
    )
    x = torch.zeros((1, 2, 32, 32))

    tokens = model(x, return_tokens=True)

    assert tuple(tokens.shape) == (1, 16, 16)

from __future__ import annotations

import math

import torch
from torch import nn


class PatchTokenViT(nn.Module):
    """Small ViT backbone that exposes patch tokens for inference and tests."""

    def __init__(
        self,
        in_channels: int = 2,
        image_size: int = 96,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _resize_pos_embed(self, grid_hw: tuple[int, int]) -> torch.Tensor:
        if grid_hw == (self.grid_size, self.grid_size):
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        patch_pos = patch_pos.transpose(1, 2).reshape(
            1, self.embed_dim, self.grid_size, self.grid_size
        )
        patch_pos = torch.nn.functional.interpolate(
            patch_pos,
            size=grid_hw,
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.reshape(1, self.embed_dim, grid_hw[0] * grid_hw[1])
        patch_pos = patch_pos.transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        grid_hw = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._resize_pos_embed(grid_hw).to(device=x.device, dtype=x.dtype)
        x = self.norm(self.encoder(x))
        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_patchtokens": x[:, 1:],
            "patch_grid": torch.tensor(grid_hw, device=x.device),
        }

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        features = self.forward_features(x)
        if return_tokens:
            return features["x_norm_patchtokens"]
        return features["x_norm_clstoken"]


def build_vit_backbone_tokens(
    in_channels: int = 2,
    image_size: int = 96,
    patch_size: int = 8,
    embed_dim: int = 128,
    depth: int = 4,
    num_heads: int = 4,
) -> PatchTokenViT:
    return PatchTokenViT(
        in_channels=in_channels,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )

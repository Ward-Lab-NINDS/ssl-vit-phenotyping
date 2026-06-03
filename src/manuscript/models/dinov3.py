from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

ChannelAdapter = Literal["first3", "mean_to_rgb", "repeat_first"]


class HuggingFacePatchTokenBackbone(nn.Module):
    """Thin wrapper exposing Hugging Face ViT/DINO patch tokens as `[B, N, D]`.

    The phenotyping pipeline operates on microscopy tensors that may have one,
    two, or many channels. Public DINOv3 checkpoints are RGB vision backbones, so
    this wrapper uses an explicit deterministic channel adapter before passing
    tensors into the model. This keeps the experiment honest: DINOv3 can be tried
    as a transfer baseline without pretending that fluorescence channels are RGB.
    """

    def __init__(
        self,
        model: nn.Module,
        channel_adapter: ChannelAdapter = "mean_to_rgb",
        apply_imagenet_norm: bool = True,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.model = model
        self.channel_adapter = channel_adapter
        self.apply_imagenet_norm = apply_imagenet_norm
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got {tuple(x.shape)}")
        channels = x.shape[1]
        if channels == 3:
            rgb = x
        elif self.channel_adapter == "repeat_first":
            rgb = x[:, :1].repeat(1, 3, 1, 1)
        elif self.channel_adapter == "mean_to_rgb":
            rgb = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        elif self.channel_adapter == "first3":
            if channels < 3:
                pad = x[:, -1:].repeat(1, 3 - channels, 1, 1)
                rgb = torch.cat([x, pad], dim=1)
            else:
                rgb = x[:, :3]
        else:
            raise ValueError(
                "channel_adapter must be one of: 'first3', 'mean_to_rgb', 'repeat_first'"
            )
        if self.apply_imagenet_norm:
            rgb = (rgb - self.image_mean.to(device=rgb.device, dtype=rgb.dtype)) / self.image_std.to(
                device=rgb.device, dtype=rgb.dtype
            )
        return rgb

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb = self._to_rgb(x)
        outputs = self.model(pixel_values=rgb, output_hidden_states=False, return_dict=True)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Hugging Face model output is missing last_hidden_state")
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected last_hidden_state [B,N,D], got {tuple(tokens.shape)}")
        # Most ViT/DINO HF models include a CLS/register prefix token. For cell
        # pooling, use spatial patch tokens and drop the first token. If a model
        # exposes only patch tokens, this still leaves at least one token for valid inputs.
        patch_tokens = tokens[:, 1:] if tokens.shape[1] > 1 else tokens
        return {"x_norm_patchtokens": patch_tokens, "x_norm_clstoken": tokens[:, 0]}

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        features = self.forward_features(x)
        if return_tokens:
            return features["x_norm_patchtokens"]
        return features["x_norm_clstoken"]


def build_dinov3_hf_backbone_tokens(
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    channel_adapter: ChannelAdapter = "mean_to_rgb",
    apply_imagenet_norm: bool = True,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    **from_pretrained_kwargs: Any,
) -> HuggingFacePatchTokenBackbone:
    """Build a DINOv3/Hugging Face backbone that exposes patch tokens.

    Install optional dependencies first:

        pip install -e ".[dinov3]"

    The default model name is intentionally configurable because public DINOv3
    checkpoints and access requirements can change. Use the exact checkpoint your
    lab has approved, then record it through the existing provenance metadata.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "DINOv3 Hugging Face loading requires the optional dependency set: "
            "pip install -e '.[dinov3]'"
        ) from exc

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        **from_pretrained_kwargs,
    )
    return HuggingFacePatchTokenBackbone(
        model=model,
        channel_adapter=channel_adapter,
        apply_imagenet_norm=apply_imagenet_norm,
    )

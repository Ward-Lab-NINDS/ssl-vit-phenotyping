from __future__ import annotations

from types import SimpleNamespace

import torch

from manuscript.models.dinov3 import HuggingFacePatchTokenBackbone


class TinyHFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, 4)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        del output_hidden_states, return_dict
        b, c, h, w = pixel_values.shape
        patches = pixel_values.reshape(b, c, h * w).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat([cls, patches], dim=1)
        return SimpleNamespace(last_hidden_state=self.proj(tokens))


def test_dinov3_wrapper_adapts_two_channels_to_patch_tokens() -> None:
    wrapper = HuggingFacePatchTokenBackbone(
        TinyHFModel(), channel_adapter="mean_to_rgb", apply_imagenet_norm=False
    )
    x = torch.rand(2, 2, 4, 4)
    tokens = wrapper(x, return_tokens=True)
    assert tokens.shape == (2, 16, 4)


def test_dinov3_wrapper_first3_pads_single_channel() -> None:
    wrapper = HuggingFacePatchTokenBackbone(
        TinyHFModel(), channel_adapter="first3", apply_imagenet_norm=False
    )
    x = torch.rand(1, 1, 3, 3)
    features = wrapper.forward_features(x)
    assert features["x_norm_patchtokens"].shape == (1, 9, 4)

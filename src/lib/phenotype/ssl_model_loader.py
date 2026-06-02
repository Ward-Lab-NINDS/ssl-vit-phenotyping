from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import torch


def import_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a `module:function` path."""
    if ":" not in path:
        raise ValueError(f"Expected 'module:function' format, got {path!r}")
    module_path, function_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    candidate = getattr(module, function_name, None)
    if not callable(candidate):
        raise ValueError(f"Could not import callable {function_name!r} from {module_path!r}")
    return candidate


def _pick_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in (
            "teacher_state",
            "teacher",
            "student_state",
            "student",
            "model_state",
            "model",
            "state_dict",
        ):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state

        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint

    raise ValueError("Checkpoint does not look like a PyTorch state_dict container")


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "backbone.", "student.", "teacher."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_ssl_vit_model(
    ckpt_path: str | Path | None,
    model_builder_path: str,
    device: str = "cuda",
    strict: bool = False,
    builder_kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Build a ViT model and optionally load an SSL checkpoint for inference."""
    builder = import_callable(model_builder_path)
    model = builder(**(builder_kwargs or {}))

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = _clean_state_dict(_pick_state_dict(checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if strict is False and (missing or unexpected):
            print(
                "Loaded checkpoint with non-strict matching. "
                f"Missing={len(missing)}, unexpected={len(unexpected)}"
            )

    model.to(device)
    model.eval()
    return model

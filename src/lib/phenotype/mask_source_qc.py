from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


KNOWN_MASK_SOURCES = {
    "brieflow",
    "cellpose",
    "sam",
    "manual",
    "stardist",
    "other",
}


@dataclass(frozen=True)
class MaskSourceMetadata:
    """Provenance fields for masks used before SSL feature extraction.

    SSL features are only meaningful relative to the masks they are pooled within.
    These metadata fields keep the repo honest about whether a feature table used
    Brieflow/CellPose/SAM/manual labels or another upstream segmentation source.
    """

    mask_source: str = "unknown"
    mask_source_detail: str = "unspecified"
    segmentation_model: str = "unspecified"
    segmentation_qc_status: str = "unknown"
    ssl_role: str = "downstream_feature_extraction"

    def to_meta_columns(self, prefix: str = "meta") -> dict[str, str]:
        return {f"{prefix}_{key}": str(value) for key, value in asdict(self).items()}


def validate_mask_source(mask_source: str, allow_unknown: bool = True) -> str:
    """Validate and normalize a mask-source label.

    Known labels are intentionally broad because the pipeline should be able to
    ingest masks from Brieflow, CellPose, SAM-style workflows, manual labels, or
    other segmentation sources.
    """

    normalized = str(mask_source or "unknown").strip().lower().replace("-", "_")
    if normalized == "cellpose_sam":
        normalized = "cellpose"
    if normalized in KNOWN_MASK_SOURCES:
        return normalized
    if allow_unknown and normalized == "unknown":
        return normalized
    if allow_unknown:
        return "other"
    raise ValueError(
        f"Unknown mask_source={mask_source!r}. Expected one of {sorted(KNOWN_MASK_SOURCES)}"
    )


def require_qc_pass(
    qc_status: str | bool | None,
    require: bool = False,
    accepted_statuses: Iterable[str] = ("pass", "passed", "true", "1", "yes"),
) -> bool:
    """Return whether segmentation QC passes, optionally raising if required."""

    if isinstance(qc_status, bool):
        passed = qc_status
    else:
        passed = str(qc_status or "unknown").strip().lower() in set(accepted_statuses)
    if require and not passed:
        raise RuntimeError(
            "Segmentation QC has not passed. SSL embeddings are downstream features and should "
            "not be interpreted until mask QC is acceptable."
        )
    return passed


def add_mask_source_metadata(
    table: pd.DataFrame,
    metadata: MaskSourceMetadata,
    prefix: str = "meta",
) -> pd.DataFrame:
    """Attach mask-source provenance columns to a phenotype table."""

    result = table.copy()
    for column, value in metadata.to_meta_columns(prefix=prefix).items():
        result[column] = value
    return result


def mask_label_summary(labels: np.ndarray, name: str = "cell") -> dict[str, float | int | str]:
    """Summarize a label image without assuming a specific segmentation tool."""

    if labels.ndim != 2:
        raise ValueError(f"Expected {name} labels [H,W], got {labels.shape}")
    ids, counts = np.unique(labels, return_counts=True)
    counts = counts[ids != 0].astype(np.float32)
    if counts.size == 0:
        return {
            "mask_name": name,
            "n_objects": 0,
            "foreground_fraction": 0.0,
            "median_object_area": 0.0,
            "object_area_cv": 0.0,
        }
    return {
        "mask_name": name,
        "n_objects": int(counts.size),
        "foreground_fraction": float(np.count_nonzero(labels) / labels.size),
        "median_object_area": float(np.median(counts)),
        "object_area_cv": float(counts.std() / max(counts.mean(), 1.0)),
    }


def summarize_mask_set(
    masks: dict[str, np.ndarray],
    metadata: MaskSourceMetadata | None = None,
) -> pd.DataFrame:
    """Create a one-row-per-mask QC/provenance table for cells/nuclei/cytoplasm masks."""

    rows = []
    for name, labels in masks.items():
        row = mask_label_summary(labels, name=name)
        if metadata is not None:
            row.update(metadata.to_meta_columns(prefix="meta"))
        rows.append(row)
    return pd.DataFrame(rows)

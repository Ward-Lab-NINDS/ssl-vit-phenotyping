from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

ALLOWED_CHANNEL_ROLES = {
    "procode_readout",
    "structural_reference",
    "phenotype_marker",
    "background",
    "unknown",
}

DEFAULT_PROCODE_READOUTS = ("V5", "NWS", "T7")
NUCLEUS_NAMES = {"nucleus", "nuclei", "dna", "dapi", "hoechst"}
REQUIRED_CHANNEL_METADATA_COLUMNS = ("channel_name", "channel_role")


@dataclass(frozen=True)
class ChannelMetadataIssue:
    level: str
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "field": self.field, "message": self.message}


def normalize_channel_names(channel_names: Iterable[str]) -> list[str]:
    """Return stable lower-case channel names while preserving the source table."""
    normalized = []
    for name in channel_names:
        text = str(name).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        normalized.append(text.strip("_"))
    return normalized


def _coerce_channel_metadata_columns(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    aliases = {
        "name": "channel_name",
        "role": "channel_role",
        "marker": "marker_or_readout",
        "readout": "marker_or_readout",
    }
    for old, new in aliases.items():
        if old in result.columns and new not in result.columns:
            result = result.rename(columns={old: new})
    return result


def validate_channel_metadata(table: pd.DataFrame) -> pd.DataFrame:
    """Validate channel metadata and return a normalized copy.

    The function raises ``ValueError`` for contract-breaking errors and adds a
    ``channel_name_normalized`` column for safe joins. Original channel labels
    such as V5, NWS, T7, and nucleus are preserved.
    """
    result = _coerce_channel_metadata_columns(table)
    missing = [
        column
        for column in REQUIRED_CHANNEL_METADATA_COLUMNS
        if column not in result.columns
    ]
    if missing:
        raise ValueError(f"Missing required channel metadata columns: {missing}")

    result = result.copy()
    result["channel_name"] = result["channel_name"].astype(str).str.strip()
    result["channel_role"] = result["channel_role"].astype(str).str.strip().str.lower()
    result["channel_name_normalized"] = normalize_channel_names(result["channel_name"])

    invalid_roles = sorted(set(result["channel_role"]) - ALLOWED_CHANNEL_ROLES)
    if invalid_roles:
        raise ValueError(f"Invalid channel_role values: {invalid_roles}")

    nucleus_as_readout = result[
        result["channel_name_normalized"].isin(NUCLEUS_NAMES)
        & (result["channel_role"] == "procode_readout")
    ]
    if not nucleus_as_readout.empty:
        names = nucleus_as_readout["channel_name"].tolist()
        raise ValueError(
            "Nucleus/DNA structural channels cannot be marked as procode_readout: "
            f"{names}"
        )

    return result


def channel_metadata_issues(
    table: pd.DataFrame,
    require_procode_readouts: bool = False,
    expected_readouts: Sequence[str] = DEFAULT_PROCODE_READOUTS,
) -> pd.DataFrame:
    """Return non-fatal channel metadata warnings as a small issue table."""
    issues: list[ChannelMetadataIssue] = []
    metadata = validate_channel_metadata(table)

    if require_procode_readouts:
        present = set(metadata.loc[metadata["channel_role"] == "procode_readout", "channel_name"])
        present_normalized = set(normalize_channel_names(present))
        missing = [
            name
            for name, normalized in zip(expected_readouts, normalize_channel_names(expected_readouts))
            if normalized not in present_normalized
        ]
        if missing:
            issues.append(
                ChannelMetadataIssue(
                    "warning",
                    "procode_readout_channels",
                    f"Missing expected ProCode/readout channels for QC: {missing}",
                )
            )

    if "wavelength_nm" in metadata.columns:
        nuclear = metadata[
            metadata["channel_name_normalized"].isin(NUCLEUS_NAMES)
            & metadata["wavelength_nm"].isna()
        ]
        if not nuclear.empty:
            issues.append(
                ChannelMetadataIssue(
                    "warning",
                    "wavelength_nm",
                    "Nucleus channel wavelength is unconfirmed; keep it null until lab-confirmed.",
                )
            )

    return pd.DataFrame([issue.as_dict() for issue in issues], columns=["level", "field", "message"])


def get_procode_readout_channels(table: pd.DataFrame) -> list[str]:
    """Return original channel names marked as barcode-like ProCode readouts."""
    metadata = validate_channel_metadata(table)
    return metadata.loc[
        metadata["channel_role"] == "procode_readout", "channel_name"
    ].tolist()


def get_structural_reference_channels(table: pd.DataFrame) -> list[str]:
    """Return original channel names marked as structural/reference channels."""
    metadata = validate_channel_metadata(table)
    return metadata.loc[
        metadata["channel_role"] == "structural_reference", "channel_name"
    ].tolist()


def output_channel_label(row: pd.Series) -> str:
    """Create a report-friendly channel label from one metadata row."""
    name = str(row.get("channel_name", "unknown")).strip()
    wavelength = row.get("wavelength_nm")
    color = str(row.get("color_label", "") or "").strip()
    role = str(row.get("channel_role", "unknown")).strip()
    parts = [name]
    if pd.notna(wavelength) and str(wavelength).strip():
        parts.append(str(int(float(wavelength))))
    if color and color.lower() != "unknown":
        parts.append(color)
    elif role:
        parts.append(role)
    return "_".join(re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_") for part in parts if part)


def channel_output_labels(table: pd.DataFrame) -> dict[str, str]:
    """Map original channel names to readable report labels."""
    metadata = validate_channel_metadata(table)
    return {
        row["channel_name"]: output_channel_label(row)
        for _, row in metadata.iterrows()
    }


def add_channel_metadata_to_output(
    df: pd.DataFrame,
    channel_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Attach compact channel metadata provenance to an output table."""
    metadata = validate_channel_metadata(channel_metadata)
    result = df.copy()
    result["meta_procode_readout_channels"] = ",".join(get_procode_readout_channels(metadata))
    result["meta_structural_reference_channels"] = ",".join(
        get_structural_reference_channels(metadata)
    )
    result["meta_channel_output_labels"] = json.dumps(
        channel_output_labels(metadata),
        sort_keys=True,
    )
    return result

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

REQUIRED_MANIFEST_COLUMNS = (
    "image_id",
    "image_path",
    "cell_mask_path",
    "mask_source",
    "segmentation_qc_status",
)

RECOMMENDED_MANIFEST_COLUMNS = (
    "plate",
    "well",
    "site",
    "channel_names",
    "condition",
    "replicate",
    "split",
)

ALLOWED_SEGMENTATION_QC_STATUS = {"pass", "fail", "unknown", "not_checked"}
ALLOWED_MASK_SOURCES = {"brieflow", "cellpose", "sam", "manual", "stardist", "other"}


@dataclass(frozen=True)
class DataContractIssue:
    level: str
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "field": self.field, "message": self.message}


def _missing_columns(columns: Iterable[str], required: Sequence[str]) -> list[str]:
    present = set(columns)
    return [column for column in required if column not in present]


def validate_manifest_table(
    manifest: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_MANIFEST_COLUMNS,
    recommended_columns: Sequence[str] = RECOMMENDED_MANIFEST_COLUMNS,
    allowed_mask_sources: set[str] = ALLOWED_MASK_SOURCES,
    allowed_segmentation_qc_status: set[str] = ALLOWED_SEGMENTATION_QC_STATUS,
) -> pd.DataFrame:
    """Validate the dataset manifest against the project data contract.

    Returns a table of issues with columns `level`, `field`, and `message`.
    The function does not require files to exist because manifests may point to
    external storage paths such as DVC, S3, Box, or institutional storage.
    """
    issues: list[DataContractIssue] = []

    if manifest.empty:
        issues.append(DataContractIssue("error", "manifest", "Manifest has no rows"))

    for column in _missing_columns(manifest.columns, required_columns):
        issues.append(DataContractIssue("error", column, "Missing required manifest column"))

    for column in _missing_columns(manifest.columns, recommended_columns):
        issues.append(DataContractIssue("warning", column, "Missing recommended manifest column"))

    if "image_id" in manifest.columns:
        duplicated = manifest["image_id"].dropna().duplicated()
        if duplicated.any():
            n_duplicated = int(duplicated.sum())
            issues.append(
                DataContractIssue(
                    "error",
                    "image_id",
                    f"Manifest contains {n_duplicated} duplicated non-null image_id values",
                )
            )

    if "mask_source" in manifest.columns:
        observed = set(manifest["mask_source"].dropna().astype(str).str.lower())
        unexpected = sorted(observed - allowed_mask_sources)
        if unexpected:
            issues.append(
                DataContractIssue(
                    "error",
                    "mask_source",
                    f"Unexpected mask_source values: {unexpected}",
                )
            )

    if "segmentation_qc_status" in manifest.columns:
        observed = set(manifest["segmentation_qc_status"].dropna().astype(str).str.lower())
        unexpected = sorted(observed - allowed_segmentation_qc_status)
        if unexpected:
            issues.append(
                DataContractIssue(
                    "error",
                    "segmentation_qc_status",
                    f"Unexpected segmentation_qc_status values: {unexpected}",
                )
            )

    if "split" in manifest.columns and len(manifest):
        split_counts = manifest["split"].dropna().astype(str).value_counts()
        if split_counts.empty:
            issues.append(DataContractIssue("warning", "split", "No split values are populated"))
        elif len(split_counts) == 1:
            issues.append(
                DataContractIssue(
                    "warning",
                    "split",
                    "Only one split is present; reserve image/well/plate-level test data when benchmarking",
                )
            )

    if "channel_names" in manifest.columns:
        empty_channels = manifest["channel_names"].isna() | (manifest["channel_names"].astype(str).str.len() == 0)
        if empty_channels.any():
            issues.append(
                DataContractIssue(
                    "warning",
                    "channel_names",
                    f"{int(empty_channels.sum())} rows have missing channel_names",
                )
            )

    return pd.DataFrame([issue.as_dict() for issue in issues], columns=["level", "field", "message"])


def validate_manifest_path(path: str | Path, sep: str = ",") -> pd.DataFrame:
    """Load and validate a manifest CSV/TSV file."""
    table = pd.read_csv(path, sep=sep)
    return validate_manifest_table(table)


def manifest_passes_contract(issues: pd.DataFrame) -> bool:
    """Return True when no contract-level errors are present."""
    if issues.empty:
        return True
    return not (issues["level"] == "error").any()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an SSL phenotyping dataset manifest.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV/TSV path.")
    parser.add_argument("--output", default=None, help="Optional path to write issue table.")
    parser.add_argument("--sep", default=",", help="Input delimiter. Default: comma.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    issues = validate_manifest_path(args.manifest, sep=args.sep)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        issues.to_csv(output, index=False, sep="\t")
    else:
        if issues.empty:
            print("Manifest passes the SSL phenotyping data contract.")
        else:
            print(issues.to_string(index=False))
    return 0 if manifest_passes_contract(issues) else 1


if __name__ == "__main__":
    raise SystemExit(main())

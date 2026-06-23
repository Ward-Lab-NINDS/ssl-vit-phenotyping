from __future__ import annotations

import pandas as pd
import pytest

from lib.phenotype.channel_metadata import (
    add_channel_metadata_to_output,
    channel_metadata_issues,
    channel_output_labels,
    get_procode_readout_channels,
    get_structural_reference_channels,
    validate_channel_metadata,
)


def _procode_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "channel_name": "V5",
                "wavelength_nm": 647,
                "channel_role": "procode_readout",
                "color_label": "far_red",
                "marker_or_readout": "V5",
            },
            {
                "channel_name": "NWS",
                "wavelength_nm": 488,
                "channel_role": "procode_readout",
                "color_label": "green",
                "marker_or_readout": "NWS",
            },
            {
                "channel_name": "T7",
                "wavelength_nm": 568,
                "channel_role": "procode_readout",
                "color_label": "orange",
                "marker_or_readout": "T7",
            },
            {
                "channel_name": "nucleus",
                "wavelength_nm": None,
                "channel_role": "structural_reference",
                "color_label": "unknown",
                "marker_or_readout": "nucleus",
            },
        ]
    )


def test_v5_nws_t7_are_accepted_as_procode_readouts() -> None:
    metadata = validate_channel_metadata(_procode_metadata())
    assert get_procode_readout_channels(metadata) == ["V5", "NWS", "T7"]
    assert get_structural_reference_channels(metadata) == ["nucleus"]


def test_invalid_role_raises_clear_error() -> None:
    metadata = _procode_metadata()
    metadata.loc[0, "channel_role"] = "barcode_identity"
    with pytest.raises(ValueError, match="Invalid channel_role"):
        validate_channel_metadata(metadata)


def test_missing_required_columns_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Missing required channel metadata columns"):
        validate_channel_metadata(pd.DataFrame({"channel_name": ["V5"]}))


def test_nucleus_marked_as_procode_readout_raises_clear_error() -> None:
    metadata = _procode_metadata()
    metadata.loc[metadata["channel_name"] == "nucleus", "channel_role"] = "procode_readout"
    with pytest.raises(ValueError, match="cannot be marked as procode_readout"):
        validate_channel_metadata(metadata)


def test_missing_expected_readout_channels_returns_warning() -> None:
    metadata = _procode_metadata().query("channel_name != 'T7'")
    issues = channel_metadata_issues(metadata, require_procode_readouts=True)
    assert not issues.empty
    assert issues.loc[0, "level"] == "warning"
    assert "T7" in issues.loc[0, "message"]


def test_output_labels_use_biological_channel_names() -> None:
    labels = channel_output_labels(_procode_metadata())
    assert labels == {
        "V5": "V5_647_far_red",
        "NWS": "NWS_488_green",
        "T7": "T7_568_orange",
        "nucleus": "nucleus_structural_reference",
    }


def test_add_channel_metadata_to_output_preserves_provenance() -> None:
    output = add_channel_metadata_to_output(
        pd.DataFrame({"label": [1, 2]}),
        _procode_metadata(),
    )
    assert output["meta_procode_readout_channels"].tolist() == ["V5,NWS,T7", "V5,NWS,T7"]
    assert output["meta_structural_reference_channels"].tolist() == ["nucleus", "nucleus"]
    assert "V5_647_far_red" in output.loc[0, "meta_channel_output_labels"]

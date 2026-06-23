import pandas as pd

from lib.phenotype.data_contract import manifest_passes_contract, validate_manifest_table


def test_valid_manifest_passes_contract():
    manifest = pd.DataFrame(
        {
            "image_id": ["img001", "img002"],
            "image_path": ["external/images/img001.tif", "external/images/img002.tif"],
            "cell_mask_path": ["external/masks/img001_cells.tif", "external/masks/img002_cells.tif"],
            "mask_source": ["cellpose", "manual"],
            "segmentation_qc_status": ["pass", "unknown"],
            "plate": ["p1", "p1"],
            "well": ["A01", "A02"],
            "site": ["s1", "s1"],
            "channel_names": ["V5,NWS,T7,nucleus", "V5,NWS,T7,nucleus"],
            "channel_metadata_path": [
                "data/ground_truth/channel_metadata.template.csv",
                "data/ground_truth/channel_metadata.template.csv",
            ],
            "condition": ["control", "treated"],
            "replicate": ["r1", "r1"],
            "split": ["train", "test"],
        }
    )
    issues = validate_manifest_table(manifest)
    assert manifest_passes_contract(issues)
    assert not (issues["level"] == "error").any() if not issues.empty else True


def test_missing_required_column_fails_contract():
    manifest = pd.DataFrame({"image_id": ["img001"]})
    issues = validate_manifest_table(manifest)
    assert not manifest_passes_contract(issues)
    assert "image_path" in set(issues["field"])


def test_invalid_mask_source_fails_contract():
    manifest = pd.DataFrame(
        {
            "image_id": ["img001"],
            "image_path": ["img001.tif"],
            "cell_mask_path": ["img001_cells.tif"],
            "mask_source": ["mystery_model"],
            "segmentation_qc_status": ["pass"],
        }
    )
    issues = validate_manifest_table(manifest)
    assert not manifest_passes_contract(issues)
    assert "mask_source" in set(issues["field"])

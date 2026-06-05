import numpy as np
import pandas as pd
import pytest

from lib.phenotype.mask_source_qc import (
    MaskSourceMetadata,
    add_mask_source_metadata,
    mask_label_summary,
    require_qc_pass,
    validate_mask_source,
)


def test_validate_mask_source_normalizes_known_labels():
    assert validate_mask_source("CellPose-SAM") == "cellpose"
    assert validate_mask_source("manual") == "manual"
    assert validate_mask_source("my-custom-tool") == "other"


def test_require_qc_pass_can_raise():
    assert require_qc_pass("passed", require=True) is True
    with pytest.raises(RuntimeError):
        require_qc_pass("failed", require=True)


def test_add_mask_source_metadata():
    table = pd.DataFrame({"label": [1, 2]})
    metadata = MaskSourceMetadata(mask_source="cellpose", segmentation_qc_status="passed")
    out = add_mask_source_metadata(table, metadata)
    assert out["meta_mask_source"].tolist() == ["cellpose", "cellpose"]
    assert out["meta_ssl_role"].tolist() == ["downstream_feature_extraction"] * 2


def test_mask_label_summary_counts_objects():
    labels = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 2]], dtype=np.int32)
    summary = mask_label_summary(labels, name="cell")
    assert summary["mask_name"] == "cell"
    assert summary["n_objects"] == 2
    assert summary["foreground_fraction"] > 0

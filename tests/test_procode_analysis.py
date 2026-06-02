from __future__ import annotations

import numpy as np
import pandas as pd

from lib.phenotype.procode_analysis import (
    add_procode_signatures,
    compare_classical_ssl_separability,
    replicate_consistency,
    segmentation_quality_from_labels,
    summarize_procode_decoding,
)


def test_add_procode_signatures_tracks_on_off_codes_and_crosstalk() -> None:
    table = pd.DataFrame(
        {
            "pc_a": [10.0, 1.0, 9.0, 0.5],
            "pc_b": [1.0, 11.0, 8.5, 0.3],
        }
    )

    decoded = add_procode_signatures(table, ["pc_a", "pc_b"], thresholds=5.0)

    assert decoded["procode_signature"].tolist() == ["10", "01", "11", "00"]
    assert decoded.loc[0, "procode_margin"] > decoded.loc[3, "procode_margin"]
    assert decoded.loc[0, "procode_crosstalk_index"] < 0.2


def test_summarize_procode_decoding_flags_expected_signatures() -> None:
    table = pd.DataFrame(
        {
            "pc_a": [10.0, 1.0, 9.0, 0.5],
            "pc_b": [1.0, 11.0, 8.5, 0.3],
        }
    )

    summary = summarize_procode_decoding(
        table,
        ["pc_a", "pc_b"],
        expected_signatures={"10", "01"},
        thresholds=5.0,
    )

    expected = dict(zip(summary["procode_signature"], summary["expected_signature"]))
    assert expected["10"] is True
    assert expected["11"] is False


def test_segmentation_quality_reports_density_sensitive_contacts() -> None:
    sparse = np.zeros((8, 8), dtype=np.uint16)
    sparse[:2, :2] = 1
    sparse[6:, 6:] = 2
    dense = np.zeros((8, 8), dtype=np.uint16)
    dense[:, :4] = 1
    dense[:, 4:] = 2

    sparse_metrics = segmentation_quality_from_labels(sparse, density="low", min_cell_area=2)
    dense_metrics = segmentation_quality_from_labels(dense, density="high", min_cell_area=2)

    assert sparse_metrics["cell_count"] == 2
    assert dense_metrics["touching_edge_fraction"] > sparse_metrics["touching_edge_fraction"]


def test_compare_classical_ssl_separability_and_replicates() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for perturbation, center in (("gene_a", -2.0), ("gene_b", 2.0)):
        for replicate in ("r1", "r2"):
            for _ in range(8):
                rows.append(
                    {
                        "sgRNA": perturbation,
                        "replicate": replicate,
                        "cell_area": rng.normal(center, 0.2),
                        "ssl_000": rng.normal(center, 0.2),
                        "ssl_001": rng.normal(center, 0.2),
                    }
                )
    table = pd.DataFrame(rows)

    separability = compare_classical_ssl_separability(table, label_col="sgRNA")
    consistency = replicate_consistency(
        table,
        perturbation_col="sgRNA",
        replicate_col="replicate",
    )

    assert set(separability["modality"]) == {"classical", "ssl"}
    assert separability["knn_accuracy"].min() > 0.8
    assert set(consistency["sgRNA"]) == {"gene_a", "gene_b"}

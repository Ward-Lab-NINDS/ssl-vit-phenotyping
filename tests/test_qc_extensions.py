from __future__ import annotations

import numpy as np
import pandas as pd

from lib.phenotype.procode_analysis import (
    calibrate_procode_thresholds,
    evaluate_batch_signal,
    evaluate_control_phenotypes,
)
from lib.phenotype.ssl_cell_features import pool_tokens_by_label


def test_calibrate_procode_thresholds_quantile():
    table = pd.DataFrame({"ch1": [0.0, 0.1, 0.9, 1.0], "ch2": [0.0, 0.2, 0.8, 1.2]})
    thresholds = calibrate_procode_thresholds(table, ["ch1", "ch2"], method="quantile", quantile=0.5)
    assert set(thresholds) == {"ch1", "ch2"}
    assert thresholds["ch1"] > 0.0


def test_calibrate_procode_thresholds_negative_control():
    table = pd.DataFrame(
        {
            "control": ["NTC", "NTC", "positive", "positive"],
            "ch1": [0.05, 0.1, 0.9, 1.1],
        }
    )
    thresholds = calibrate_procode_thresholds(
        table, ["ch1"], method="negative_control", control_col="control", control_value="NTC"
    )
    assert 0.05 <= thresholds["ch1"] <= 0.11


def test_pool_tokens_by_label_median():
    import torch

    tokens = torch.tensor([[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [5.0, 5.0]])
    labels = np.array([[1, 1], [1, 2]], dtype=np.int32)
    ids, features = pool_tokens_by_label(tokens, labels, pooling="median")
    assert ids.tolist() == [1, 2]
    assert features.shape == (2, 2)


def test_control_and_batch_outputs_are_tables():
    table = pd.DataFrame(
        {
            "control_type": ["NTC", "NTC", "positive", "positive", "positive", "positive"],
            "perturbation": ["NTC", "NTC", "A", "A", "B", "B"],
            "sgRNA": ["NTC", "NTC", "A1", "A1", "B1", "B1"],
            "plate": ["p1", "p2", "p1", "p2", "p1", "p2"],
            "ssl_000": [0, 0.1, 1, 1.1, -1, -1.1],
            "ssl_001": [0, 0.1, 1, 1.1, -1, -1.1],
        }
    )
    controls = evaluate_control_phenotypes(table, "control_type", "perturbation")
    batch = evaluate_batch_signal(table, "sgRNA", ["plate"])
    assert not controls.empty
    assert not batch.empty

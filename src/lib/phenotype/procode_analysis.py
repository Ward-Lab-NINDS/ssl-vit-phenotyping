from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _as_feature_matrix(
    table: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    feature_prefixes: Sequence[str] = ("ssl_",),
) -> tuple[np.ndarray, list[str]]:
    if feature_columns is None:
        feature_columns = [
            column
            for column in table.columns
            if any(column.startswith(prefix) for prefix in feature_prefixes)
            and pd.api.types.is_numeric_dtype(table[column])
        ]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("No feature columns were found")
    x = table[feature_columns].to_numpy(dtype=np.float32)
    keep = np.isfinite(x).all(axis=1)
    return x[keep], feature_columns


def _resolve_thresholds(
    table: pd.DataFrame,
    channel_columns: Sequence[str],
    thresholds: float | dict[str, float] | pd.Series | None,
    quantile: float,
) -> dict[str, float]:
    if thresholds is None:
        return {
            channel: float(table[channel].quantile(quantile))
            for channel in channel_columns
        }
    if isinstance(thresholds, (float, int)):
        return {channel: float(thresholds) for channel in channel_columns}
    return {channel: float(thresholds[channel]) for channel in channel_columns}


def add_procode_signatures(
    table: pd.DataFrame,
    channel_columns: Sequence[str],
    thresholds: float | dict[str, float] | pd.Series | None = None,
    quantile: float = 0.75,
    signature_col: str = "procode_signature",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Add binary on/off ProCode signatures and decoding quality columns."""
    missing = [column for column in channel_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing ProCode channel columns: {missing}")

    channel_columns = list(channel_columns)
    resolved = _resolve_thresholds(table, channel_columns, thresholds, quantile)
    result = table.copy()
    bits = []
    values = result[channel_columns].to_numpy(dtype=np.float32)
    for channel in channel_columns:
        bit_col = f"{channel}_on"
        bit = (result[channel].to_numpy(dtype=np.float32) > resolved[channel]).astype(np.int8)
        result[bit_col] = bit
        bits.append(bit)

    bit_matrix = np.vstack(bits).T
    result[signature_col] = ["".join(str(int(value)) for value in row) for row in bit_matrix]
    result["procode_on_count"] = bit_matrix.sum(axis=1)

    on_means = []
    off_means = []
    crosstalk = []
    for row_values, row_bits in zip(values, bit_matrix):
        on_values = row_values[row_bits == 1]
        off_values = row_values[row_bits == 0]
        on_mean = float(on_values.mean()) if on_values.size else 0.0
        off_mean = float(off_values.mean()) if off_values.size else 0.0
        on_means.append(on_mean)
        off_means.append(off_mean)
        crosstalk.append(off_mean / max(on_mean, eps) if on_values.size else np.nan)
    result["procode_on_mean"] = on_means
    result["procode_off_mean"] = off_means
    result["procode_margin"] = result["procode_on_mean"] - result["procode_off_mean"]
    result["procode_crosstalk_index"] = crosstalk
    return result


def summarize_procode_decoding(
    table: pd.DataFrame,
    channel_columns: Sequence[str],
    expected_signatures: Iterable[str] | None = None,
    thresholds: float | dict[str, float] | pd.Series | None = None,
    quantile: float = 0.75,
) -> pd.DataFrame:
    """Summarize whether combinatorial ProCode decoding is clean enough to trust."""
    decoded = add_procode_signatures(
        table=table,
        channel_columns=channel_columns,
        thresholds=thresholds,
        quantile=quantile,
    )
    summary = (
        decoded.groupby("procode_signature", dropna=False)
        .agg(
            n_cells=("procode_signature", "size"),
            mean_on_count=("procode_on_count", "mean"),
            mean_margin=("procode_margin", "mean"),
            mean_crosstalk_index=("procode_crosstalk_index", "mean"),
        )
        .reset_index()
        .sort_values(["n_cells", "procode_signature"], ascending=[False, True])
    )
    summary["fraction_cells"] = summary["n_cells"] / max(len(decoded), 1)
    if expected_signatures is not None:
        expected = set(expected_signatures)
        summary["expected_signature"] = summary["procode_signature"].isin(expected)
    return summary


def segmentation_quality_from_labels(
    labels: np.ndarray,
    image_id: str | None = None,
    density: str | float | int | None = None,
    min_cell_area: int = 16,
    large_mask_mad_threshold: float = 3.0,
) -> dict[str, float | int | str | None]:
    """Compute density-sensitive segmentation metrics from a labeled cell mask."""
    if labels.ndim != 2:
        raise ValueError(f"Expected labels [H,W], got {labels.shape}")

    ids, counts = np.unique(labels, return_counts=True)
    counts = counts[ids != 0].astype(np.float32)
    ids = ids[ids != 0]
    foreground_fraction = float(np.count_nonzero(labels) / labels.size)

    if counts.size:
        median_area = float(np.median(counts))
        mad = float(np.median(np.abs(counts - median_area)))
        large_cutoff = median_area + large_mask_mad_threshold * max(mad, 1.0)
        oversized_fraction = float(np.mean(counts > large_cutoff))
        small_fraction = float(np.mean(counts < min_cell_area))
        area_cv = float(counts.std() / max(counts.mean(), 1.0))
    else:
        median_area = 0.0
        oversized_fraction = 0.0
        small_fraction = 0.0
        area_cv = 0.0

    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    horizontal_contact = horizontal & (labels[:, 1:] != 0) & (labels[:, :-1] != 0)
    vertical_contact = vertical & (labels[1:, :] != 0) & (labels[:-1, :] != 0)
    contact_edges = int(horizontal_contact.sum() + vertical_contact.sum())
    foreground_edges = int(
        ((labels[:, 1:] != 0) | (labels[:, :-1] != 0)).sum()
        + ((labels[1:, :] != 0) | (labels[:-1, :] != 0)).sum()
    )

    return {
        "image_id": image_id,
        "density": density,
        "cell_count": int(ids.size),
        "foreground_fraction": foreground_fraction,
        "median_cell_area": median_area,
        "cell_area_cv": area_cv,
        "small_mask_fraction": small_fraction,
        "oversized_mask_fraction": oversized_fraction,
        "undersegmentation_proxy": oversized_fraction,
        "touching_cell_edges": contact_edges,
        "touching_edge_fraction": contact_edges / max(foreground_edges, 1),
    }


def compare_segmentation_by_density(
    metrics: pd.DataFrame,
    density_col: str = "density",
) -> pd.DataFrame:
    """Aggregate segmentation metrics across density conditions."""
    numeric_cols = [
        column
        for column in metrics.columns
        if column != density_col and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    return metrics.groupby(density_col)[numeric_cols].agg(["mean", "std", "count"])


def evaluate_feature_separability(
    table: pd.DataFrame,
    label_col: str,
    feature_columns: Sequence[str] | None = None,
    feature_prefixes: Sequence[str] = ("ssl_",),
    n_neighbors: int = 5,
    n_splits: int = 5,
    random_state: int = 7,
) -> dict[str, float | int]:
    """Evaluate perturbation or sgRNA separability with kNN and silhouette score."""
    if label_col not in table.columns:
        raise ValueError(f"Missing label column {label_col!r}")

    x, used_columns = _as_feature_matrix(table, feature_columns, feature_prefixes)
    y_all = table[label_col].to_numpy()
    finite_rows = np.isfinite(table[used_columns].to_numpy(dtype=np.float32)).all(axis=1)
    y = y_all[finite_rows]
    keep = pd.notna(y)
    x = x[keep]
    y = y[keep]
    classes, class_counts = np.unique(y, return_counts=True)

    result: dict[str, float | int] = {
        "n_cells": int(len(y)),
        "n_classes": int(classes.size),
        "n_features": int(len(used_columns)),
        "knn_accuracy": np.nan,
        "silhouette": np.nan,
    }
    if classes.size < 2 or len(y) < 3:
        return result

    if len(y) > classes.size:
        result["silhouette"] = float(silhouette_score(StandardScaler().fit_transform(x), y))

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return result

    splits = min(n_splits, min_class_count)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in cv.split(x, y):
        k = min(n_neighbors, len(train_idx))
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        model.fit(x[train_idx], y[train_idx])
        scores.append(float(model.score(x[test_idx], y[test_idx])))
    result["knn_accuracy"] = float(np.mean(scores))
    return result


def compare_classical_ssl_separability(
    table: pd.DataFrame,
    label_col: str,
    classical_prefixes: Sequence[str] = ("cell_", "nuclei_", "cytoplasm_"),
    ssl_prefixes: Sequence[str] = ("ssl_",),
) -> pd.DataFrame:
    """Compare morphology features against SSL embeddings for the same labels."""
    rows = []
    for modality, prefixes in (
        ("classical", classical_prefixes),
        ("ssl", ssl_prefixes),
    ):
        try:
            metrics = evaluate_feature_separability(
                table=table,
                label_col=label_col,
                feature_prefixes=prefixes,
            )
        except ValueError:
            continue
        rows.append({"modality": modality, **metrics})
    return pd.DataFrame(rows)


def replicate_consistency(
    table: pd.DataFrame,
    perturbation_col: str,
    replicate_col: str,
    feature_columns: Sequence[str] | None = None,
    feature_prefixes: Sequence[str] = ("ssl_",),
) -> pd.DataFrame:
    """Measure whether replicate perturbation centroids agree in feature space."""
    for column in (perturbation_col, replicate_col):
        if column not in table.columns:
            raise ValueError(f"Missing column {column!r}")

    x, used_columns = _as_feature_matrix(table, feature_columns, feature_prefixes)
    finite_rows = np.isfinite(table[used_columns].to_numpy(dtype=np.float32)).all(axis=1)
    clean = table.loc[finite_rows, [perturbation_col, replicate_col]].copy()
    clean[used_columns] = x
    centroids = clean.groupby([perturbation_col, replicate_col])[used_columns].mean()
    if centroids.empty:
        return pd.DataFrame(
            columns=[
                perturbation_col,
                "n_replicates",
                "mean_within_replicate_cosine",
                "mean_between_perturbation_cosine",
            ]
        )

    centroid_values = StandardScaler().fit_transform(centroids.to_numpy(dtype=np.float32))
    centroid_index = centroids.index.to_frame(index=False)
    similarity = cosine_similarity(centroid_values)
    rows = []
    for perturbation, group in centroid_index.groupby(perturbation_col):
        idx = group.index.to_numpy()
        if idx.size < 2:
            continue
        within = [similarity[i, j] for i, j in combinations(idx, 2)]
        other = np.setdiff1d(np.arange(similarity.shape[0]), idx)
        between = similarity[np.ix_(idx, other)].reshape(-1) if other.size else np.array([])
        rows.append(
            {
                perturbation_col: perturbation,
                "n_replicates": int(idx.size),
                "mean_within_replicate_cosine": float(np.mean(within)),
                "mean_between_perturbation_cosine": (
                    float(np.mean(between)) if between.size else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)

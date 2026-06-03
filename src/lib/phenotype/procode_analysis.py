from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
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




def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Compute a one-dimensional Otsu threshold for an intensity channel."""
    clean = values[np.isfinite(values)].astype(np.float32)
    if clean.size == 0:
        raise ValueError("Cannot calibrate threshold from an empty channel")
    if np.all(clean == clean[0]):
        return float(clean[0])

    hist, bin_edges = np.histogram(clean, bins=n_bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_low = np.cumsum(hist)
    weight_high = np.cumsum(hist[::-1])[::-1]
    mean_low = np.cumsum(hist * centers) / np.maximum(weight_low, 1)
    mean_high = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight_high[::-1], 1))[::-1]
    variance_between = weight_low[:-1] * weight_high[1:] * (mean_low[:-1] - mean_high[1:]) ** 2
    return float(centers[:-1][int(np.argmax(variance_between))])


def _gmm_threshold(values: np.ndarray, random_state: int = 7) -> float:
    """Fit a two-component Gaussian mixture and return the midpoint between component means."""
    clean = values[np.isfinite(values)].astype(np.float32)
    if clean.size < 4 or np.all(clean == clean[0]):
        return float(np.nanmedian(clean))
    model = GaussianMixture(n_components=2, random_state=random_state)
    model.fit(clean.reshape(-1, 1))
    means = np.sort(model.means_.reshape(-1))
    return float(np.mean(means))


def calibrate_procode_thresholds(
    table: pd.DataFrame,
    channel_columns: Sequence[str],
    method: str = "quantile",
    quantile: float = 0.75,
    control_col: str | None = None,
    control_value: str | int | float | None = None,
    negative_quantile: float = 0.99,
    random_state: int = 7,
) -> dict[str, float]:
    """Calibrate per-channel ProCode thresholds before binary decoding.

    Methods:
    - ``quantile``: channel-wise quantile over all cells.
    - ``otsu``: unsupervised Otsu threshold over all cells.
    - ``gmm``: midpoint between two Gaussian-mixture component means.
    - ``negative_control``: high quantile among rows matching a negative-control label.

    The returned dictionary can be passed to ``add_procode_signatures`` and the
    downstream QC helpers. Prefer negative-control calibration when a clean
    no-ProCode or no-primary-control population is available.
    """
    missing = [column for column in channel_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing ProCode channel columns: {missing}")

    if method not in {"quantile", "otsu", "gmm", "negative_control"}:
        raise ValueError("method must be one of: 'quantile', 'otsu', 'gmm', 'negative_control'")

    source = table
    if method == "negative_control":
        if control_col is None:
            raise ValueError("negative_control calibration requires control_col")
        if control_col not in table.columns:
            raise ValueError(f"Missing control column {control_col!r}")
        if control_value is None:
            mask = table[control_col].astype(str).str.lower().isin(
                {"negative", "negative_control", "ntc", "non-targeting", "none", "no_procode"}
            )
        else:
            mask = table[control_col] == control_value
        source = table.loc[mask]
        if source.empty:
            raise ValueError("No rows matched the requested negative-control population")

    thresholds: dict[str, float] = {}
    for channel in channel_columns:
        values = source[channel].to_numpy(dtype=np.float32)
        if method == "quantile":
            thresholds[channel] = float(np.nanquantile(values, quantile))
        elif method == "negative_control":
            thresholds[channel] = float(np.nanquantile(values, negative_quantile))
        elif method == "otsu":
            thresholds[channel] = _otsu_threshold(values)
        elif method == "gmm":
            thresholds[channel] = _gmm_threshold(values, random_state=random_state)
    return thresholds


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


def hamming_distance(left: str, right: str) -> int:
    """Return Hamming distance between two equal-length ProCode signatures."""
    if len(left) != len(right):
        raise ValueError("Signatures must have the same length")
    return sum(a != b for a, b in zip(left, right))


def summarize_codebook_design(expected_signatures: Iterable[str]) -> dict[str, float | int]:
    """Summarize combinatorial codebook spacing before running pooled screens."""
    signatures = sorted(set(expected_signatures))
    if not signatures:
        raise ValueError("expected_signatures cannot be empty")

    lengths = {len(signature) for signature in signatures}
    if len(lengths) != 1:
        raise ValueError("All expected signatures must have the same length")

    distances = [
        hamming_distance(left, right)
        for left, right in combinations(signatures, 2)
    ]
    code_length = lengths.pop()
    return {
        "n_expected_signatures": len(signatures),
        "code_length": code_length,
        "code_space_size": 2**code_length,
        "code_space_fraction_used": len(signatures) / float(2**code_length),
        "min_hamming_distance": min(distances) if distances else 0,
        "mean_hamming_distance": float(np.mean(distances)) if distances else 0.0,
    }


def procode_qc_summary(
    table: pd.DataFrame,
    channel_columns: Sequence[str],
    expected_signatures: Iterable[str] | None = None,
    thresholds: float | dict[str, float] | pd.Series | None = None,
    quantile: float = 0.75,
    min_margin: float | None = None,
    max_crosstalk: float | None = None,
) -> dict[str, float | int]:
    """Return one-row QC metrics for combinatorial ProCode decoding."""
    decoded = add_procode_signatures(
        table=table,
        channel_columns=channel_columns,
        thresholds=thresholds,
        quantile=quantile,
    )
    expected = set(expected_signatures or [])
    has_expected = bool(expected)
    invalid = (
        ~decoded["procode_signature"].isin(expected)
        if has_expected
        else pd.Series(False, index=decoded.index)
    )
    empty = decoded["procode_on_count"] == 0
    low_margin = (
        decoded["procode_margin"] < min_margin
        if min_margin is not None
        else pd.Series(False, index=decoded.index)
    )
    high_crosstalk = (
        decoded["procode_crosstalk_index"] > max_crosstalk
        if max_crosstalk is not None
        else pd.Series(False, index=decoded.index)
    )
    ambiguous = invalid | empty | low_margin | high_crosstalk

    return {
        "n_cells": int(len(decoded)),
        "n_observed_signatures": int(decoded["procode_signature"].nunique()),
        "fraction_empty_signature": float(empty.mean()) if len(decoded) else np.nan,
        "fraction_invalid_signature": float(invalid.mean()) if has_expected and len(decoded) else np.nan,
        "median_procode_margin": float(decoded["procode_margin"].median()) if len(decoded) else np.nan,
        "median_crosstalk_index": float(decoded["procode_crosstalk_index"].median()) if len(decoded) else np.nan,
        "fraction_low_margin": float(low_margin.mean()) if min_margin is not None and len(decoded) else np.nan,
        "fraction_high_crosstalk": float(high_crosstalk.mean()) if max_crosstalk is not None and len(decoded) else np.nan,
        "fraction_ambiguous": float(ambiguous.mean()) if len(decoded) else np.nan,
    }


def flag_ambiguous_procodes(
    table: pd.DataFrame,
    channel_columns: Sequence[str],
    expected_signatures: Iterable[str] | None = None,
    thresholds: float | dict[str, float] | pd.Series | None = None,
    quantile: float = 0.75,
    min_margin: float | None = None,
    max_crosstalk: float | None = None,
    ambiguous_col: str = "procode_ambiguous",
) -> pd.DataFrame:
    """Add decoded signatures and a conservative ambiguity flag."""
    decoded = add_procode_signatures(
        table=table,
        channel_columns=channel_columns,
        thresholds=thresholds,
        quantile=quantile,
    )
    expected = set(expected_signatures or [])
    ambiguous = decoded["procode_on_count"] == 0
    if expected:
        ambiguous = ambiguous | ~decoded["procode_signature"].isin(expected)
    if min_margin is not None:
        ambiguous = ambiguous | (decoded["procode_margin"] < min_margin)
    if max_crosstalk is not None:
        ambiguous = ambiguous | (decoded["procode_crosstalk_index"] > max_crosstalk)
    decoded[ambiguous_col] = ambiguous.astype(bool)
    return decoded


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


def evaluate_control_phenotypes(
    table: pd.DataFrame,
    control_col: str,
    perturbation_col: str,
    feature_columns: Sequence[str] | None = None,
    feature_prefixes: Sequence[str] = ("ssl_",),
    negative_labels: Sequence[str] = ("NTC", "non-targeting", "negative", "negative_control"),
) -> pd.DataFrame:
    """Compare control perturbation centroids against negative-control cells.

    This helps determine whether known positive controls move away from the
    non-targeting population before interpreting unknown perturbations.
    """
    for column in (control_col, perturbation_col):
        if column not in table.columns:
            raise ValueError(f"Missing column {column!r}")
    x, used_columns = _as_feature_matrix(table, feature_columns, feature_prefixes)
    finite_rows = np.isfinite(table[used_columns].to_numpy(dtype=np.float32)).all(axis=1)
    clean = table.loc[finite_rows, [control_col, perturbation_col]].copy()
    clean[used_columns] = StandardScaler().fit_transform(x)

    negative_mask = clean[control_col].astype(str).str.lower().isin(
        {label.lower() for label in negative_labels}
    ) | clean[perturbation_col].astype(str).str.lower().isin({label.lower() for label in negative_labels})
    if not negative_mask.any():
        return pd.DataFrame(columns=[
            "control_label", "n_cells", "mean_centroid_distance_to_negative", "within_control_variance"
        ])

    negative_centroid = clean.loc[negative_mask, used_columns].mean().to_numpy(dtype=np.float32)
    rows = []
    for label, group in clean.groupby(control_col):
        values = group[used_columns].to_numpy(dtype=np.float32)
        centroid = values.mean(axis=0)
        rows.append(
            {
                "control_label": label,
                "n_cells": int(len(group)),
                "mean_centroid_distance_to_negative": float(np.linalg.norm(centroid - negative_centroid)),
                "within_control_variance": float(np.mean(np.var(values, axis=0))) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_centroid_distance_to_negative", ascending=False)


def evaluate_batch_signal(
    table: pd.DataFrame,
    biological_label_col: str,
    batch_cols: Sequence[str],
    feature_columns: Sequence[str] | None = None,
    feature_prefixes: Sequence[str] = ("ssl_",),
) -> pd.DataFrame:
    """Compare whether embeddings separate biology more strongly than nuisance variables."""
    rows = []
    bio = evaluate_feature_separability(
        table=table,
        label_col=biological_label_col,
        feature_columns=feature_columns,
        feature_prefixes=feature_prefixes,
    )
    rows.append({"label_type": "biological", "label_col": biological_label_col, **bio})
    for column in batch_cols:
        if column not in table.columns:
            continue
        metrics = evaluate_feature_separability(
            table=table,
            label_col=column,
            feature_columns=feature_columns,
            feature_prefixes=feature_prefixes,
        )
        rows.append({"label_type": "batch", "label_col": column, **metrics})
    result = pd.DataFrame(rows)
    if not result.empty:
        bio_knn = result.loc[result["label_type"] == "biological", "knn_accuracy"].max()
        result["knn_relative_to_biology"] = result["knn_accuracy"] / max(float(bio_knn), 1e-6)
    return result

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np

from lib.phenotype.procode_analysis import (
    calibrate_procode_thresholds,
    evaluate_batch_signal,
    evaluate_control_phenotypes,
    evaluate_feature_separability,
    flag_ambiguous_procodes,
    procode_qc_summary,
    replicate_consistency,
    summarize_codebook_design,
    summarize_procode_decoding,
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    prefixes: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()

    def select_columns(self, table: pd.DataFrame) -> list[str]:
        columns = list(self.columns)
        for column in table.columns:
            if any(column.startswith(prefix) for prefix in self.prefixes):
                columns.append(column)
        selected = []
        seen = set()
        for column in columns:
            if column in table.columns and column not in seen:
                selected.append(column)
                seen.add(column)
        if not selected:
            raise ValueError(f"No columns found for feature set {self.name!r}")
        return selected


def parse_feature_set(spec: str) -> FeatureSet:
    """Parse `name=prefix:a,prefix:b,column:c` feature-set specs."""
    if "=" not in spec:
        raise ValueError(f"Expected feature spec like 'ssl=prefix:ssl_', got {spec!r}")
    name, raw_parts = spec.split("=", 1)
    prefixes = []
    columns = []
    for part in raw_parts.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("prefix:"):
            prefixes.append(part.removeprefix("prefix:"))
        elif part.startswith("column:"):
            columns.append(part.removeprefix("column:"))
        else:
            prefixes.append(part)
    return FeatureSet(name=name, prefixes=tuple(prefixes), columns=tuple(columns))


def benchmark_feature_sets(
    table: pd.DataFrame,
    label_col: str,
    feature_sets: Sequence[FeatureSet],
    replicate_col: str | None = None,
    perturbation_col: str | None = None,
    n_neighbors: int = 5,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare feature sets with separability and optional replicate consistency."""
    separability_rows = []
    consistency_rows = []

    for feature_set in feature_sets:
        columns = feature_set.select_columns(table)
        metrics = evaluate_feature_separability(
            table=table,
            label_col=label_col,
            feature_columns=columns,
            n_neighbors=n_neighbors,
            n_splits=n_splits,
        )
        separability_rows.append({"feature_set": feature_set.name, **metrics})

        if replicate_col and perturbation_col:
            consistency = replicate_consistency(
                table=table,
                perturbation_col=perturbation_col,
                replicate_col=replicate_col,
                feature_columns=columns,
            )
            if not consistency.empty:
                consistency.insert(0, "feature_set", feature_set.name)
                consistency_rows.append(consistency)

    separability = pd.DataFrame(separability_rows)
    consistency_table = (
        pd.concat(consistency_rows, ignore_index=True)
        if consistency_rows
        else pd.DataFrame()
    )
    return separability, consistency_table


def rank_feature_sets(separability: pd.DataFrame) -> pd.DataFrame:
    """Rank feature sets by kNN accuracy, silhouette, and feature count."""
    if separability.empty:
        return separability
    ranked = separability.copy()
    ranked["_knn_sort"] = ranked["knn_accuracy"].fillna(-1)
    ranked["_silhouette_sort"] = ranked["silhouette"].fillna(-1)
    ranked = ranked.sort_values(
        ["_knn_sort", "_silhouette_sort", "n_features"],
        ascending=[False, False, True],
    ).drop(columns=["_knn_sort", "_silhouette_sort"])
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def benchmark_table(
    table: pd.DataFrame,
    label_col: str,
    feature_sets: Sequence[FeatureSet],
    replicate_col: str | None = None,
    perturbation_col: str | None = None,
    procode_channels: Sequence[str] = (),
    expected_signatures: Sequence[str] = (),
    procode_threshold: float | None = None,
    min_procode_margin: float | None = None,
    max_procode_crosstalk: float | None = None,
    procode_threshold_method: str = "quantile",
    procode_control_col: str | None = None,
    procode_control_value: str | None = None,
    control_col: str | None = None,
    batch_cols: Sequence[str] = (),
) -> dict[str, pd.DataFrame]:
    """Run feature, replicate, and optional ProCode QC benchmarks."""
    separability, consistency = benchmark_feature_sets(
        table=table,
        label_col=label_col,
        feature_sets=feature_sets,
        replicate_col=replicate_col,
        perturbation_col=perturbation_col,
    )
    outputs = {
        "feature_separability": separability,
        "feature_ranking": rank_feature_sets(separability),
    }
    if not consistency.empty:
        outputs["replicate_consistency"] = consistency

    if procode_channels:
        if procode_threshold is not None:
            thresholds = procode_threshold
        else:
            thresholds = calibrate_procode_thresholds(
                table=table,
                channel_columns=procode_channels,
                method=procode_threshold_method,
                control_col=procode_control_col,
                control_value=procode_control_value,
            )
        outputs["procode_thresholds"] = pd.DataFrame([thresholds])
        outputs["procode_decoding"] = summarize_procode_decoding(
            table=table,
            channel_columns=procode_channels,
            expected_signatures=expected_signatures or None,
            thresholds=thresholds,
        )
        outputs["procode_qc"] = pd.DataFrame(
            [
                procode_qc_summary(
                    table=table,
                    channel_columns=procode_channels,
                    expected_signatures=expected_signatures or None,
                    thresholds=thresholds,
                    min_margin=min_procode_margin,
                    max_crosstalk=max_procode_crosstalk,
                )
            ]
        )
        outputs["procode_flagged_cells"] = flag_ambiguous_procodes(
            table=table,
            channel_columns=procode_channels,
            expected_signatures=expected_signatures or None,
            thresholds=thresholds,
            min_margin=min_procode_margin,
            max_crosstalk=max_procode_crosstalk,
        )
        if expected_signatures:
            outputs["procode_codebook"] = pd.DataFrame([summarize_codebook_design(expected_signatures)])

    if control_col and perturbation_col:
        for feature_set in feature_sets:
            try:
                outputs[f"control_phenotype_qc_{feature_set.name}"] = evaluate_control_phenotypes(
                    table=table,
                    control_col=control_col,
                    perturbation_col=perturbation_col,
                    feature_columns=feature_set.select_columns(table),
                )
            except ValueError:
                continue

    if batch_cols:
        for feature_set in feature_sets:
            try:
                outputs[f"batch_signal_{feature_set.name}"] = evaluate_batch_signal(
                    table=table,
                    biological_label_col=label_col,
                    batch_cols=batch_cols,
                    feature_columns=feature_set.select_columns(table),
                )
            except ValueError:
                continue

    return outputs


def write_benchmark_report(outputs: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    """Write an advisor-facing Markdown summary of benchmark outputs."""
    ranking = outputs.get("feature_ranking", pd.DataFrame())
    procode_qc = outputs.get("procode_qc", pd.DataFrame())
    lines = [
        "# SSL ViT Phenotyping Benchmark Report",
        "",
        "This report summarizes whether the phenotype table is ready for downstream biological interpretation.",
        "",
        "## Feature-set ranking",
    ]
    if ranking.empty:
        lines.append("No feature ranking was generated.")
    else:
        lines.append(ranking.to_markdown(index=False))
        best = ranking.iloc[0]
        lines.extend(["", f"Best feature set: **{best.get('feature_set', 'unknown')}**."])

    lines.extend(["", "## ProCode QC"] )
    if procode_qc.empty:
        lines.append("No ProCode channels were supplied, so ProCode QC was skipped.")
    else:
        lines.append(procode_qc.to_markdown(index=False))
        ambiguous = float(procode_qc.iloc[0].get("fraction_ambiguous", np.nan))
        if np.isfinite(ambiguous):
            decision = "PASS" if ambiguous <= 0.10 else "REVIEW"
            lines.append(f"ProCode decision: **{decision}** based on ambiguous fraction {ambiguous:.3f}.")

    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "1. Do not interpret SSL clusters until segmentation QC and ProCode decoding pass.",
        "2. Prefer combined features only when they improve separability without increasing batch signal.",
        "3. If plate, well, density, or imaging date predicts embeddings better than sgRNA, treat the run as batch-confounded.",
        "4. Preserve the generated TSVs with the exact phenotype input and model provenance metadata.",
    ])
    report_path = output_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark phenotype feature sets.")
    parser.add_argument("--input", required=True, help="Input phenotype TSV/CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark TSVs.")
    parser.add_argument("--label-col", required=True, help="Column used for separability labels.")
    parser.add_argument("--replicate-col", default=None)
    parser.add_argument("--perturbation-col", default=None)
    parser.add_argument(
        "--feature-set",
        action="append",
        required=True,
        help="Feature spec, e.g. ssl=prefix:ssl_ or classical=prefix:cell_,prefix:nuclei_.",
    )
    parser.add_argument("--procode-channel", action="append", default=[])
    parser.add_argument("--expected-signature", action="append", default=[])
    parser.add_argument("--procode-threshold", type=float, default=None)
    parser.add_argument("--procode-threshold-method", default="quantile", choices=["quantile", "otsu", "gmm", "negative_control"])
    parser.add_argument("--procode-control-col", default=None)
    parser.add_argument("--procode-control-value", default=None)
    parser.add_argument("--min-procode-margin", type=float, default=None)
    parser.add_argument("--max-procode-crosstalk", type=float, default=None)
    parser.add_argument("--control-col", default=None, help="Control annotation column for positive/negative-control QC.")
    parser.add_argument("--batch-col", action="append", default=[], help="Nuisance column to test for batch leakage, e.g. plate, well, density.")
    parser.add_argument("--write-report", action="store_true", help="Write benchmark_report.md in addition to TSVs.")
    parser.add_argument("--sep", default="\t")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    table = pd.read_csv(args.input, sep=args.sep)
    feature_sets = [parse_feature_set(spec) for spec in args.feature_set]
    outputs = benchmark_table(
        table=table,
        label_col=args.label_col,
        feature_sets=feature_sets,
        replicate_col=args.replicate_col,
        perturbation_col=args.perturbation_col,
        procode_channels=args.procode_channel,
        expected_signatures=args.expected_signature,
        procode_threshold=args.procode_threshold,
        min_procode_margin=args.min_procode_margin,
        max_procode_crosstalk=args.max_procode_crosstalk,
        procode_threshold_method=args.procode_threshold_method,
        procode_control_col=args.procode_control_col,
        procode_control_value=args.procode_control_value,
        control_col=args.control_col,
        batch_cols=args.batch_col,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, output in outputs.items():
        output.to_csv(output_dir / f"{name}.tsv", sep="\t", index=False)
    if args.write_report:
        write_benchmark_report(outputs, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd
from tifffile import imread


def _get_param(name: str, default=None):
    return snakemake.params.get(name, default)


def _sha256(path):
    if not path:
        return "none"
    path = Path(path)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _fallback_phenotype_from_labels(cells) -> pd.DataFrame:
    labels = sorted(int(label) for label in set(cells.reshape(-1)) if int(label) != 0)
    return pd.DataFrame({"label": labels})


data_phenotype = imread(snakemake.input[0])
nuclei = imread(snakemake.input[1])
cells = imread(snakemake.input[2])
cytoplasms = imread(snakemake.input[3])

if _get_param("cp_method", "cp_multichannel") == "cp_measure":
    try:
        from lib.phenotype.extract_phenotype_cp_measure import extract_phenotype_cp_measure

        phenotype_cp = extract_phenotype_cp_measure(
            data_phenotype=data_phenotype,
            nuclei=nuclei,
            cells=cells,
            cytoplasms=cytoplasms,
            channel_names=_get_param("channel_names"),
        )
    except ModuleNotFoundError:
        phenotype_cp = _fallback_phenotype_from_labels(cells)
else:
    try:
        from lib.phenotype.extract_phenotype_cp_multichannel import (
            extract_phenotype_cp_multichannel,
        )

        phenotype_cp = extract_phenotype_cp_multichannel(
            data_phenotype=data_phenotype,
            nuclei=nuclei,
            cells=cells,
            cytoplasms=cytoplasms,
            foci_channel=_get_param("foci_channel_index"),
            channel_names=_get_param("channel_names"),
            wildcards=snakemake.wildcards,
        )
    except ModuleNotFoundError:
        phenotype_cp = _fallback_phenotype_from_labels(cells)

if bool(_get_param("ssl_enable", False)):
    from lib.phenotype.ssl_cell_features import extract_ssl_cell_embeddings
    from lib.phenotype.ssl_model_loader import load_ssl_vit_model

    ssl_model_builder = _get_param("ssl_model_builder")
    if not ssl_model_builder:
        raise ValueError("ssl_enable=True requires params.ssl_model_builder")

    ssl_model = load_ssl_vit_model(
        ckpt_path=_get_param("ssl_ckpt"),
        model_builder_path=ssl_model_builder,
        device=_get_param("ssl_device", "cuda"),
        strict=bool(_get_param("ssl_strict", False)),
        builder_kwargs=dict(_get_param("ssl_model_kwargs", {}) or {}),
    )

    ssl_use_channels = _get_param("ssl_use_channels")
    if ssl_use_channels is not None:
        ssl_use_channels = [int(channel) for channel in ssl_use_channels]

    ssl_df = extract_ssl_cell_embeddings(
        data_phenotype=data_phenotype,
        cells=cells,
        model=ssl_model,
        device=_get_param("ssl_device", "cuda"),
        patch_size=_get_param("ssl_patch_size", 8),
        use_channels=ssl_use_channels,
        prefix=_get_param("ssl_prefix", "ssl"),
        pca_dim=_get_param("ssl_pca_dim"),
        pca_basis_path=_get_param("ssl_pca_basis_path"),
        pooling=_get_param("ssl_pooling", "mean"),
        normalization=_get_param("ssl_normalization", "zscore"),
        wildcards=dict(snakemake.wildcards),
    )
    ssl_df["meta_ssl_patch_size"] = _get_param("ssl_patch_size", 8)
    ssl_df["meta_ssl_pooling"] = _get_param("ssl_pooling", "mean")
    ssl_df["meta_ssl_normalization"] = _get_param("ssl_normalization", "zscore")
    ssl_df["meta_ssl_channels"] = (
        ",".join(str(channel) for channel in ssl_use_channels)
        if ssl_use_channels is not None
        else "all"
    )
    ssl_df["meta_ssl_pca_dim"] = (
        _get_param("ssl_pca_dim") if _get_param("ssl_pca_dim") is not None else "none"
    )
    ssl_df["meta_ssl_model_builder"] = ssl_model_builder
    ssl_df["meta_ssl_checkpoint"] = _get_param("ssl_ckpt") or "none"
    ssl_df["meta_ssl_checkpoint_sha256"] = _sha256(_get_param("ssl_ckpt"))
    ssl_df["meta_ssl_device"] = _get_param("ssl_device", "cuda")
    ssl_df["meta_git_commit"] = _git_commit()

    if "label" not in phenotype_cp.columns:
        raise RuntimeError("Phenotype table is missing required 'label' column for SSL merge.")
    phenotype_cp = phenotype_cp.merge(ssl_df, on="label", how="left")

phenotype_cp.to_csv(snakemake.output[0], index=False, sep="\t")

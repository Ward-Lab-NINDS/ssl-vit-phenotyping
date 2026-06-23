from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from tifffile import imread

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
NOTEBOOK_HELPERS = SCRIPTS_DIR / "notebooks"
if str(NOTEBOOK_HELPERS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_HELPERS))

from ssl_patch_interactive_review import foreground_mask, normalize_patch


SUPPORTED_UPLOAD_TYPES = ["tif", "tiff", "png", "jpg", "jpeg"]


st.set_page_config(
    page_title="SSL Patch Phenotyping",
    layout="wide",
    initial_sidebar_state="expanded",
)


def output_root() -> Path:
    root = REPO_ROOT / "outputs" / "streamlit_patch_runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_uploads(uploaded_files: list, run_id: str) -> Path:
    upload_dir = output_root() / run_id / "uploaded_images"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        (upload_dir / safe_name).write_bytes(uploaded_file.getbuffer())
    return upload_dir


def run_patch_test(
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
    max_patches: int,
    ssl_ckpt: str | None = None,
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_ssl_patch_test.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--patch-size",
        str(patch_size),
        "--max-patches-per-image",
        str(max_patches),
    ]
    if ssl_ckpt:
        cmd.extend(["--ssl-ckpt", ssl_ckpt])
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout


def load_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str]:
    manifest = pd.read_csv(output_dir / "patch_manifest.tsv", sep="\t")
    phenotypes = pd.read_csv(output_dir / "patch_phenotypes.tsv", sep="\t")
    ssl_path = output_dir / "ssl_patch_features.tsv"
    try:
        ssl_features = pd.read_csv(ssl_path, sep="\t") if ssl_path.exists() else None
    except pd.errors.EmptyDataError:
        ssl_features = None
    report = (output_dir / "ssl_patch_test_report.md").read_text(encoding="utf-8")
    return manifest, phenotypes, ssl_features, report


def build_download_zip(output_dir: Path) -> Path:
    zip_path = output_dir / "ssl_patch_test_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(output_dir))
    return zip_path


def display_patch(patch_path: str, title: str, show_overlay: bool) -> None:
    patch = normalize_patch(imread(patch_path))
    if show_overlay:
        mask = foreground_mask(patch)
        rgb = np.dstack([patch, patch, patch])
        rgb[..., 0] = np.maximum(rgb[..., 0], mask.astype(float))
        rgb[..., 1] *= np.where(mask, 0.35, 1.0)
        rgb[..., 2] *= np.where(mask, 0.35, 1.0)
        fig = px.imshow(rgb, title=title)
    else:
        fig = px.imshow(patch, color_continuous_scale="gray", title=title)
        fig.update_coloraxes(showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=46, b=10),
        dragmode="pan",
        height=560,
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})


def phenotype_scatter(phenotypes: pd.DataFrame, x_col: str, y_col: str) -> None:
    data = phenotypes.copy()
    inferred = data["source_image"].str.extract(r"-(ch\d+)")[0]
    data["channel"] = inferred.fillna("mapping_unknown").map(
        lambda value: f"{value}_mapping_unknown" if str(value).startswith("ch") else value
    )
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="channel",
        hover_data=["patch_id", "source_image"],
        title="Patch phenotype scatter; use channel metadata for V5/NWS/T7/nucleus labels",
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=48, b=10))
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})


def synopsis() -> None:
    st.subheader("What The App Does")
    st.write(
        "This app turns local microscopy images into reproducible patch-level phenotype "
        "outputs. It is designed as a lab-accessible front end to the same patch runner "
        "used by the command line and notebook workflows."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Classical patch phenotypes**

            - Foreground fraction
            - Foreground/background intensity separation
            - Connected-component count
            - Largest-object area and elongation
            - Boundary and gradient summaries

            These are interpretable and useful for QC immediately.
            """
        )
    with col2:
        st.markdown(
            """
            **SSL patch embeddings**

            - Require a trained or validated checkpoint
            - Preserve local texture and context as patch tokens
            - Can be benchmarked against classical morphology
            - Should be interpreted only after QC and batch checks

            SSL is a representation layer, not a segmentation replacement.
            """
        )
    st.divider()
    st.subheader("Why This Route Instead Of Lovable First?")
    st.write(
        "For the first lab-accessible version, a repo-native Streamlit app is better than "
        "Lovable because the app must handle local high-resolution TIFF data, preserve "
        "deterministic outputs, and run the exact same analysis code as the CLI. Lovable "
        "could be useful later for a polished public-facing interface, but it should sit "
        "on top of a validated backend rather than become the source of scientific truth."
    )
    st.subheader("Brieflow Alignment")
    st.write(
        "Brieflow-style workflows already organize fixed-cell optical pooled screen data "
        "around image processing, segmentation, phenotype extraction, perturbation mapping, "
        "and screen-level interpretation. This app adds an optional patch representation "
        "review layer before whole-image scaling."
    )


def main() -> None:
    st.title("SSL Patch Phenotyping")
    st.caption(
        "Local high-resolution patch extraction, morphology/intensity phenotyping, "
        "and SSL-ready review for Brieflow-style microscopy workflows."
    )

    with st.sidebar:
        st.header("Analysis Input")
        input_mode = st.radio("Input mode", ["Use local folder", "Upload images"], horizontal=False)
        patch_size = st.radio("Patch size", [200, 100], horizontal=True)
        max_patches = st.number_input("Max patches per image", min_value=1, max_value=100, value=10)
        ssl_ckpt = st.text_input(
            "Optional SSL checkpoint path",
            value="",
            help="Leave blank to skip SSL embeddings and compute interpretable patch phenotypes only.",
        )

        local_input = None
        uploaded_files = []
        if input_mode == "Use local folder":
            local_input = st.text_input(
                "Local image folder",
                value="/Users/makennarodriguez/Desktop/procodes",
            )
        else:
            uploaded_files = st.file_uploader(
                "Upload TIFF/PNG/JPEG files",
                type=SUPPORTED_UPLOAD_TYPES,
                accept_multiple_files=True,
            )

        run_clicked = st.button("Run Patch Phenotyping", type="primary", width="stretch")

    if "output_dir" not in st.session_state:
        st.session_state.output_dir = REPO_ROOT / "outputs" / "ssl_patch_test_200"

    if run_clicked:
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        output_dir = output_root() / run_id / "analysis"
        if input_mode == "Upload images":
            if not uploaded_files:
                st.error("Upload at least one image before running analysis.")
                st.stop()
            input_dir = save_uploads(uploaded_files, run_id)
        else:
            input_dir = Path(local_input or "").expanduser()
            if not input_dir.exists():
                st.error(f"Input folder does not exist: {input_dir}")
                st.stop()

        with st.status("Running patch phenotyping...", expanded=True) as status:
            st.write(f"Input: `{input_dir}`")
            st.write(f"Output: `{output_dir}`")
            code, log = run_patch_test(
                input_dir=input_dir,
                output_dir=output_dir,
                patch_size=int(patch_size),
                max_patches=int(max_patches),
                ssl_ckpt=ssl_ckpt.strip() or None,
            )
            st.code(log)
            if code == 0:
                status.update(label="Analysis complete", state="complete")
                st.session_state.output_dir = output_dir
            else:
                status.update(label="Analysis failed", state="error")
                st.stop()

    output_dir = Path(st.session_state.output_dir)
    if not (output_dir / "patch_manifest.tsv").exists():
        st.info("Run an analysis or point the app at an existing patch output folder.")
        synopsis()
        return

    manifest, phenotypes, ssl_features, report = load_outputs(output_dir)
    tabs = st.tabs(["Image Review", "Patch Phenotypes", "Report", "Synopsis"])

    with tabs[0]:
        st.subheader("High-Resolution Patch Review")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patches", len(manifest))
        c2.metric("Images", manifest["source_image"].nunique())
        c3.metric("Patch size", int(manifest["patch_size"].iloc[0]))
        c4.metric("SSL rows", 0 if ssl_features is None else len(ssl_features))

        choices = [
            f"{idx:03d} | {row.source_image} | y={int(row.y)}, x={int(row.x)}"
            for idx, row in manifest.reset_index(drop=True).iterrows()
        ]
        selected_label = st.selectbox("Patch", choices)
        selected_index = int(selected_label.split("|", 1)[0].strip())
        overlay = st.toggle("Show foreground overlay", value=True)
        selected_row = manifest.iloc[selected_index]
        display_patch(
            selected_row["patch_path"],
            title=selected_row["patch_id"],
            show_overlay=overlay,
        )

    with tabs[1]:
        st.subheader("Interactive Phenotype Analysis")
        phenotype_cols = [col for col in phenotypes.columns if col.startswith("phenotype_")]
        col_a, col_b = st.columns(2)
        x_col = col_a.selectbox("X metric", phenotype_cols, index=phenotype_cols.index("phenotype_foreground_fraction"))
        default_y = "phenotype_largest_component_elongation"
        y_col = col_b.selectbox(
            "Y metric",
            phenotype_cols,
            index=phenotype_cols.index(default_y) if default_y in phenotype_cols else 0,
        )
        phenotype_scatter(phenotypes, x_col, y_col)
        st.dataframe(phenotypes, width="stretch", height=360)

    with tabs[2]:
        st.subheader("QC Report And Exports")
        st.markdown(report)
        zip_path = build_download_zip(output_dir)
        st.download_button(
            "Download output bundle",
            data=zip_path.read_bytes(),
            file_name=zip_path.name,
            mime="application/zip",
        )
        st.write("Output folder:", output_dir)

    with tabs[3]:
        synopsis()


if __name__ == "__main__":
    main()

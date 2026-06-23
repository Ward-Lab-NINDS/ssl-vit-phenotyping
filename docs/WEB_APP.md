# SSL Patch Phenotyping Web App

This repository includes a local Streamlit web app for lab-accessible patch
phenotyping review.

The app is designed for high-resolution local microscopy data. It keeps raw
uploads and generated outputs on the local machine under ignored `outputs/`
folders, and it calls the same `scripts/run_ssl_patch_test.py` runner used by
the command-line workflow. That keeps results consistent across CLI, notebook,
Fiji, and web review.

## Install

```bash
pip install -e ".[webapp]"
```

## Run Locally

```bash
streamlit run apps/ssl_patch_streamlit_app.py
```

Then open the URL Streamlit prints, usually:

```text
http://localhost:8501
```

## Run For Lab Network Access

Use this only on a trusted lab network:

```bash
streamlit run apps/ssl_patch_streamlit_app.py --server.address 0.0.0.0
```

Other lab members can open:

```text
http://YOUR_COMPUTER_IP:8501
```

The Streamlit upload limit is configured in `.streamlit/config.toml` as
`2048 MB`. Large image uploads can still be slow, so local folder mode is usually
better for full microscopy datasets.

## App Capabilities

- Upload TIFF/PNG/JPEG files or point to a local image folder.
- Choose 100x100 or 200x200 patches.
- Save selected patch TIFFs.
- Compute interpretable patch phenotype metrics.
- Review high-resolution patches with zoomable Plotly figures.
- Toggle foreground mask overlays.
- Explore phenotype scatterplots.
- Read the generated QC report.
- Download a ZIP bundle of outputs.

## Scientific Positioning

The web app does not replace Brieflow segmentation. It provides a patch-level
review and feature-extraction layer before whole-image scaling.

Current biologically interpretable outputs:

- foreground fraction;
- foreground/background intensity separation;
- connected-component count;
- largest-component area and elongation;
- boundary and gradient summaries;
- per-channel intensity metrics.

SSL embeddings require a trained or deliberately selected checkpoint. If no
checkpoint is supplied, the app skips SSL embeddings rather than creating random
features that could be mistaken for biology.

## Lovable Recommendation

Do not use Lovable as the first implementation path for this scientific app.
Lovable is useful for fast product scaffolding and could help later with a more
polished public-facing interface, but the first working lab app should stay
repo-native because it needs:

- local high-resolution TIFF handling;
- deterministic patch extraction;
- consistent TSV/report outputs;
- compatibility with Fiji and Jupyter review;
- simple GitHub version control.

A future production architecture could use a polished React/Next.js frontend
with a Python FastAPI backend that calls the same analysis functions. For now,
Streamlit gives the best lab-accessible route with the lowest scientific risk.

## VS Code And Codex Plugins

Codex account plugins/connectors do not need to be reinstalled in VS Code to run
this app. VS Code only needs:

- the repository;
- the Python environment with dependencies installed;
- Git/GitHub authentication if you want to commit or push from VS Code.

Plugin access is specific to the Codex session. Running the Streamlit app itself
does not depend on Codex plugins.

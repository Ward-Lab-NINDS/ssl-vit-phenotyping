# Lab Web App Access

Use this page as the GitHub-facing entry point for the SSL Patch Phenotyping
web app.

## What This App Is

The app is a local lab-accessible Streamlit interface for patch-based microscopy
phenotyping. It supports:

- TIFF/PNG/JPEG upload;
- local folder analysis;
- 100x100 or 200x200 patch extraction;
- high-resolution patch review with pan/zoom;
- foreground mask overlays;
- interactive phenotype scatterplots;
- QC report review;
- output ZIP downloads;
- optional SSL checkpoint-backed embeddings.

The app keeps raw microscopy files and generated outputs local. This is
intentional: high-resolution microscopy data should not be committed to GitHub.

## Run The App

From the repository root:

```bash
pip install -e ".[webapp]"
streamlit run apps/ssl_patch_streamlit_app.py
```

Open:

```text
http://localhost:8501
```

For trusted lab-network access:

```bash
streamlit run apps/ssl_patch_streamlit_app.py --server.address 0.0.0.0
```

Streamlit will print a network URL such as:

```text
http://YOUR_COMPUTER_IP:8501
```

## Important Scientific Note

The app currently produces biologically interpretable patch-level
morphology/intensity phenotypes without requiring a trained SSL checkpoint.

SSL embeddings require a trained or deliberately selected checkpoint. If no
checkpoint is supplied, the app skips SSL embeddings so random features are not
mistaken for biology.

## Key Documentation

- `docs/WEB_APP.md`: detailed app setup and lab-network notes.
- `docs/PHENOTYPING_SYNOPSIS.md`: digestible explanation of phenotype outputs.
- `docs/SSL_PATCH_TESTING.md`: patch-testing workflow, Fiji, and Jupyter review.
- `notebooks/ssl_patch_interactive_review.ipynb`: interactive notebook review.

## Why This Is Not Hosted Directly On GitHub

GitHub hosts the code and documentation, but it does not run the Streamlit app
for uploaded microscopy images. The app should run on a lab workstation or
trusted server so high-resolution image data stays local and outputs remain
reproducible.

Later, a production deployment could use a hosted frontend with a protected
Python backend. For now, the local Streamlit route is safer for lab data,
TIFF fidelity, and reproducible analysis.

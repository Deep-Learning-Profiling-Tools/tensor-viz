# tensor-viz

`tensor-viz` is a local tensor viewer for NumPy arrays. The Python API starts a small HTTP server, serves a browser UI, and lets you inspect tensor layouts, slices, grouped dimensions, and custom color overlays.

## Install

### Source install for Python use

1. Create and activate a Python environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell use:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. Install the Python package in editable mode.

   ```bash
   pip install -e .
   ```

3. Install the frontend dependencies and build the static assets once for the checkout.

   ```bash
   npm install
   npm run build
   ```

4. Run the demo script.

   ```bash
   python demo.py
   ```

The `npm run build` step is required in a source checkout because the Python server serves the built frontend from `python/src/tensor_viz/static/`.

### Frontend and TypeScript development

1. Install the workspace dependencies.

   ```bash
   npm install
   ```

2. Start the Vite dev server.

   ```bash
   npm run dev
   ```

3. In another shell, activate the Python environment and run a viewer session.

   ```bash
   python demo.py
   ```

Use `npm run build` when you need to refresh the packaged static assets consumed by the Python package.

### Documentation build

Install the docs dependencies and build both doc systems locally:

```bash
pip install -e ".[docs]"
npm install
npm run docs
```

The Sphinx site is written to `docs/_build/html`, and the embedded TypeDoc site is written to `docs/_build/html/_typedoc`.

## Quick Start

```python
import numpy as np
import tensor_viz

x = np.random.randn(32, 64, 64)
session = tensor_viz.viz(x)
print(session.url)
session.wait()
```

`labels=` is optional. When omitted, axes default to `A B C ... Z AA AB ...`.

## Python API

The main Python entrypoints are:

- `tensor_viz.viz(...)` to launch a viewer session
- `tensor_viz.TensorMeta(...)` to describe metadata-only tensors without sending dense payload bytes
- `tensor_viz.Tab(...)` to group tensors into tabs
- `tensor_viz.create_session_data(...)` to prebuild a raw session payload
- `tensor_viz.ViewerSession` to inspect the local URL and stop the server

`demo.py` contains runnable examples for single tensors, metadata-only tensors, sequences, mappings, tabs, labels, prebuilt sessions, and session lifecycle options.

## TypeScript API

The TypeScript package lives in `packages/viewer-core`. It exports the browser viewer class plus layout, view-parsing, and NumPy helpers through `packages/viewer-core/src/index.ts`.

Generate the TypeScript API reference with:

```bash
npm run docs:ts
```

That command writes the standalone TypeDoc site to `docs/_extra/_typedoc`.

## Read the Docs

The repo includes `.readthedocs.yaml` plus a Sphinx docs tree under `docs/`. Read the Docs can build the Python docs and pre-generate the embedded TypeDoc site from the repository root without extra project-specific scripting.

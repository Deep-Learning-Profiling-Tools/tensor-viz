# Install

## Source checkout

`tensor-viz` currently expects a built frontend in source checkouts, so the Python and Node steps both matter.

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

### Frontend assets

```bash
npm install
npm run build
```

This copies the built demo assets into `python/src/tensor_viz/static/`, which is what `tensor_viz.viz()` serves.

## Running the demo

```bash
python demo.py
```

Pick a demo case by editing `DEMO` near the top of `demo.py`.

## Frontend development

Use the Vite dev server when iterating on the browser UI:

```bash
npm install
npm run dev
```

Run a Python viewer session in another shell so the demo app has tensor data to load.

## Building the docs

Install the docs dependencies and build both the Sphinx site and the TypeDoc site:

```bash
pip install -e ".[docs]"
npm install
npm run docs
```

Outputs:

- Sphinx HTML: `docs/_build/html`
- Standalone TypeDoc HTML: `docs/_extra/_typedoc`
- Embedded TypeDoc HTML: `docs/_build/html/_typedoc`

## Read the Docs

The repository root includes `.readthedocs.yaml`. A Read the Docs project can point at the repo root and build the docs without extra commands:

- Sphinx uses `docs/conf.py`
- Node is enabled so TypeDoc can run before the Sphinx build
- Python dependencies come from `docs/requirements.txt` and the package itself

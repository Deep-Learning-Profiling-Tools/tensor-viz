The `tensor-viz/docs/` directory contains the package documentation built by Sphinx and TypeDoc.

The Markdown files under this directory are hand-written guide pages. `index.md` defines the Sphinx table of contents. `install.md`, `python.md`, `typescript.md`, and `ui.md` explain the public package from a user's point of view, while `architecture.md` gives a higher-level system overview. The `reference/` pages are generated API references and should be refreshed through the documented tooling rather than edited as the primary source of truth.

Documentation should follow the code's public contracts. If a behavior is still internal to the demo app, document it in the relevant `ARCHITECTURE.md` near the code instead of presenting it as a package guarantee.

When changing package docs, run the relevant docs command from `tensor-viz/`:

```bash
npm run docs:ts
python -m sphinx -b html docs docs/_build/html
```

If the docs change depends on built demo assets, run `npm run build` as well so package assets are refreshed.

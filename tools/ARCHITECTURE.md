The `tools/` directory contains repository maintenance scripts, not runtime library code.

`sync-demo-assets.mjs` runs after the demo build. It copies Vite's generated files into `python/src/tensor_viz/static/` so the Python package serves the same frontend that local development tested.

`sync-linear-layout-examples.py` keeps baked linear-layout demo tabs synchronized with the LL-viz Python demo source when that source is available. It rewrites only the marked generated block in the linear-layout extension, which keeps hand-written TypeScript separate from generated examples.

Tooling here should stay deterministic and small. If a script starts needing app state or viewer internals, move that behavior into the package that owns the concept and let the tool call the package-level API.

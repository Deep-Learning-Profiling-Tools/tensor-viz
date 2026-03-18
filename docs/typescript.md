# TypeScript Usage

The browser-facing code lives in `packages/viewer-core`.

## Install

```bash
npm install
```

## Create a viewer

```ts
import { TensorViewer } from "@tensor-viz/viewer-core";

const host = document.getElementById("viewport");
if (!host) throw new Error("missing viewport");

const viewer = new TensorViewer(host);
viewer.addTensor([2, 3], new Float64Array([0, 1, 2, 3, 4, 5]), "weights");
viewer.setDisplayMode("2d");
viewer.toggleHeatmap(true);
```

## Generate the TypeScript reference

```bash
npm run docs:ts
```

That command writes a standalone TypeDoc site to `docs/_extra/_typedoc`. When the Sphinx docs are built, the generated TypeDoc site is copied into the final documentation output under `_typedoc/` and linked from the reference section.

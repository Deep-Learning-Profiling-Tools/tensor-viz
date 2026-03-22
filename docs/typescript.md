# TypeScript Usage

Use the TypeScript APIs when a web app or service owns the session structure.
This surface is explicit: you build `ViewerSnapshot`, `BundleManifest`, and
`SessionBundleManifest` data directly, or drive `viewer-demo` and `viewer-core`
yourself.

## Shared manifest builders

Use these helpers when you want to describe what should be shown without
rendering anything yet.

### One tensor

```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    { name: "activations", dtype: "float32", shape: [8, 16, 32] },
  ],
});
```

By default, omitted axis labels become `A B C ... Z A0 B0 ...` in tensor order.

### Custom axis labels

Use `axisLabels` when you want names such as channel, height, and width instead
of the default `A B C ...` labels. Custom labels must start with one letter and
may use only non-letters after it, so `C`, `H`, `W`, `B0`, and `T11` are valid
while `Batch` and `CH` are invalid.

```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    { name: "activations", dtype: "float32", shape: [8, 16, 32], axisLabels: ["C", "H", "W"] },
  ],
});
```

### Manifest model

The shared builders produce three nested object shapes:

- `createViewerSnapshot(...)` returns `ViewerSnapshot`, which stores viewer
  state only: display mode, panel toggles, camera, active tensor, and each
  tensor's current view and offset.
- `createBundleManifest(...)` returns `BundleManifest`, which describes one
  viewer document. It combines one `ViewerSnapshot` with one or more tensor
  entries containing ids, names, dtypes, shapes, optional payload locations,
  views, offsets, and color instructions.
- `createSessionBundleManifest(...)` returns `SessionBundleManifest`, which is a
  multi-tab wrapper around bundle-style documents. Each tab has an `id`,
  `title`, `viewer`, and `tensors`.

Use `createViewerSnapshot(...)` when you only need serialized viewer state. Use
`createBundleManifest(...)` for one tab or one document. Use
`createSessionBundleManifest(...)` when you need multiple tabs.

### Metadata-only tensors

Use metadata-only tensors when you want shape, labels, slicing, and layout
without sending the full dense payload up front.

```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    { name: "activations", dtype: "float32", shape: [32, 64, 64], axisLabels: ["C", "H", "W"], placeholderData: true },
    { name: "weights", dtype: "float32", shape: [64, 32, 3, 3], axisLabels: ["O", "I", "K0", "K1"], placeholderData: true },
  ],
});
```

### Multiple tensors in one view

Pass multiple tensors into one bundle when you want them side by side in one
viewer tab.

```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    { name: "weights", dtype: "float32", shape: [16, 16], axisLabels: ["O", "I"] },
    { name: "bias", dtype: "float32", shape: [16], axisLabels: ["O"] },
  ],
});
```

### Tabs

Use `createSessionBundleManifest(...)` when you want separate viewer documents
instead of one shared view.

```ts
import { createSessionBundleManifest } from "@tensor-viz/viewer-core";

const session = createSessionBundleManifest([
  {
    title: "inputs",
    tensors: [
      { name: "image", dtype: "float32", shape: [3, 32, 32], axisLabels: ["C", "H", "W"] },
    ],
  },
  {
    title: "weights",
    tensors: [
      { name: "conv", dtype: "float32", shape: [16, 3, 3, 3], axisLabels: ["O", "I", "K0", "K1"] },
    ],
  },
]);
```

### Custom colors

The shared manifest layer stores colors as manifest instructions.

```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    {
      name: "weights",
      dtype: "float32",
      shape: [3, 4],
      axisLabels: ["O", "I"],
      colorInstructions: [
        { mode: "rgba", kind: "coords", coords: [[0, 0], [1, 1]], color: [0, 90, 255, 255] },
      ],
    },
  ],
});
```

## viewer-demo

Use `@tensor-viz/viewer-demo` when you want the full stock app shell: renderer,
widgets, menus, tabs, and sidebar UI.

The current mount API is iframe-backed.

```ts
import { mountDemoApp } from "@tensor-viz/viewer-demo";

const host = document.getElementById("demo-host");
if (!host) throw new Error("missing demo host");

const app = mountDemoApp(host, { src: "/" });
```

## viewer-core

Use `TensorViewer` when your app wants direct control over the rendered viewer
instance.

### Create a viewer and launch one tensor

```ts
import { TensorViewer } from "@tensor-viz/viewer-core";

const host = document.getElementById("viewport");
if (!host) throw new Error("missing viewport");

const viewer = new TensorViewer(host);
const handle = viewer.addTensor(
  [8, 16, 32],
  new Float32Array(8 * 16 * 32),
  "activations",
);

viewer.setDisplayMode("2d");
```

`addTensor(...)` uses the default axis labels `A B C ... Z A0 B0 ...`. To load
custom axis labels, prebuilt views, or manifest-provided colors, use
`loadBundleData(...)`.

Unlike `viewer-demo`, `viewer-core` does not add ribbon menus, widgets, or tab
UI for you.

### Create a metadata-only tensor

```ts
const handle = viewer.addMetadataTensor([64, 32, 3, 3], "float32", "weights", undefined, [
  "O",
  "I",
  "K0",
  "K1",
]);

viewer.setActiveTensor(handle.id);
```

### Lazy data hydration

Use `requestTensorData` together with metadata-only tensors when the host wants
to fetch dense payloads only on demand.

```ts
import { TensorViewer } from "@tensor-viz/viewer-core";

const viewer = new TensorViewer(host, {
  requestTensorData: async (tensor, reason) => {
    if (reason !== "heatmap" && reason !== "save") return null;
    return fetchTensorDataFromServer(tensor.id);
  },
});

viewer.addMetadataTensor([1024, 1024], "float32", "activations");
viewer.toggleHeatmap(); // requests data for metadata-only tensors
```

You can also drive hydration manually with `setTensorData(...)`,
`clearTensorData(...)`, `ensureTensorData(...)`, and `getTensorStatus(...)`.

### Launch multiple tensors in one view

```ts
const weights = viewer.addTensor([4, 4], new Float32Array(16), "weights");
const activations = viewer.addTensor([2, 8, 8], new Float32Array(2 * 8 * 8), "activations");

viewer.setActiveTensor(activations.id);
```

### Tensor-view permutation

`setTensorView(...)` applies permutation virtually. It does not transpose or
rewrite the stored tensor buffer.

The token order in the view string defines the visible axis order. For example,
`"B A"` means "show axis `B` first, then axis `A`". The viewer renders cells in
that reordered view space and maps each displayed coordinate back to the
original tensor coordinate before reading values.

For a shape `[2, 3]` tensor with default labels `A B`:

```ts
viewer.setTensorView(handle.id, "B A");
```

In that view:

- displayed `viewCoord [0, 0]` reads original tensor coordinate `[0, 0]`
- displayed `viewCoord [1, 0]` reads original tensor coordinate `[0, 1]`
- displayed `viewCoord [2, 1]` reads original tensor coordinate `[1, 2]`

This is why the inspector preview can describe a view as a `.permute(...)`
without changing the underlying tensor bytes.

### Tabs

`viewer-core` does not include a built-in tab model. If you want tabs in a web
app, create them in your host UI and either:

- keep one `TensorViewer` instance per tab, or
- swap manifests and tensor buffers into one viewer with `loadBundleData(...)`.

### Custom colors

In `viewer-core`, custom colors are live viewer mutations.

Dense RGBA:

```ts
viewer.colorTensor(
  weights.id,
  new Uint8ClampedArray(Array.from({ length: 16 }, () => [0, 90, 255, 255]).flat()),
);
```

Coordinate RGBA:

```ts
viewer.colorTensor(weights.id, [[0, 0], [1, 1]], [0, 90, 255, 255]);
```

Region RGBA:

```ts
viewer.colorTensor(weights.id, [0, 0], [3, 2], [1, 2], [0, 90, 255, 255]);
```

Coordinate HS:

```ts
viewer.colorTensor(weights.id, [[0, 0], [1, 1]], [240, 1]);
```

Clear colors:

```ts
viewer.clearTensorColors(weights.id);
```

### Load manifest data

```ts
import { TensorViewer, type BundleManifest } from "@tensor-viz/viewer-core";

const manifest: BundleManifest = {
  version: 1,
  viewer: {
    version: 1,
    displayMode: "2d",
    heatmap: true,
    showDimensionLines: true,
    showInspectorPanel: true,
    showHoverDetailsPanel: true,
    camera: {
      position: [0, 0, 30],
      target: [0, 0, 0],
      rotation: [0, 0, 0],
      zoom: 1,
    },
    tensors: [
      {
        id: "tensor-1",
        name: "weights",
        offset: [0, 0, 0],
        view: { view: "O I", hiddenIndices: [0, 0] },
      },
    ],
    activeTensorId: "tensor-1",
  },
  tensors: [
    {
      id: "tensor-1",
      name: "weights",
      dtype: "float32",
      shape: [3, 4],
      axisLabels: ["O", "I"],
      byteOrder: "little",
      dataFile: "tensor-1.bin",
      offset: [0, 0, 0],
      view: { view: "O I", hiddenIndices: [0, 0] },
      colorInstructions: [
        { mode: "rgba", kind: "coords", coords: [[0, 0], [1, 1]], color: [0, 90, 255, 255] },
      ],
    },
  ],
};

viewer.loadBundleData(manifest, new Map([["tensor-1", new Float32Array(12)]]));
```

### Case study: a custom experiment dashboard

`viewer-demo` is one example of a UI wrapper around `viewer-core`, but you can
build a different product shell around the same engine.

For example, suppose your app needs:

- a run selector
- a layer list
- a tensor search box
- custom hover cards
- links back to your training or tracing UI

In that setup, your app owns the outer UI and `TensorViewer` owns the rendered
tensor viewport.

One common flow looks like this:

1. Your app fetches tensor metadata for the selected run.
2. Your app builds a `BundleManifest` or `SessionBundleManifest`.
3. Your app creates one `TensorViewer` for the viewport.
4. Your app loads the selected tensor set with `loadBundleData(...)`.
5. Your own controls call methods such as `setActiveTensor(...)`,
   `setTensorView(...)`, and `setDisplayMode(...)`.
6. Your own React, Vue, or DOM state subscribes to `subscribe(...)` and
   `subscribeHover(...)` to render side panels, badges, or hover cards.
7. If tensors are too large to preload, your app uses metadata-only entries plus
   `requestTensorData` so bytes load only when needed.

Minimal host-side sketch:

```ts
import {
  TensorViewer,
  createBundleManifest,
  type NumericArray,
} from "@tensor-viz/viewer-core";

const host = document.getElementById("viewport");
if (!host) throw new Error("missing viewport");

const manifest = createBundleManifest({
  tensors: [
    {
      id: "weights",
      name: "weights",
      dtype: "float32",
      shape: [64, 32, 3, 3],
      axisLabels: ["O", "I", "K0", "K1"],
      placeholderData: true,
    },
  ],
});

const viewer = new TensorViewer(host, {
  requestTensorData: async (tensor, reason): Promise<NumericArray | null> => {
    if (reason !== "heatmap" && reason !== "save") return null;
    return fetchTensorBytes(tensor.id);
  },
});

viewer.loadBundleData(manifest, new Map());

viewer.subscribe((snapshot) => {
  updateToolbar(snapshot);
});

viewer.subscribeHover((hover) => {
  updateHoverCard(hover);
});

layerPicker.onchange = (tensorId) => {
  viewer.setActiveTensor(tensorId);
};

displayToggle.onclick = () => {
  viewer.setDisplayMode("3d");
};

viewInput.onchange = (value) => {
  viewer.setTensorView("weights", value);
};
```

The important design point is that `viewer-core` gives you the tensor renderer
and stateful engine, while your app decides what product-specific UI and
workflow should sit around it.

### Lifecycle

```ts
viewer.destroy();
```

# Usage

This guide is organized around three layers:

1. Shared session builders
2. `viewer-demo`
3. `viewer-core`

## 1. Shared session builders

Use this layer when you want to prepare session data, tabs, labels, colors, and
initial viewer state. The Python API adds local-server convenience on top of
the same document model. In both languages, labels are optional; omitted labels
default to `A B C ... Z AA AB ...`.

### One tensor

`````{tab-set}
````{tab-item} Python
```python
import numpy as np
import tensor_viz

x = np.random.randn(8, 16, 32)
tensor_viz.viz(x, labels="C H W")
```
````

````{tab-item} TypeScript
```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    { name: "activations", dtype: "float32", shape: [8, 16, 32], axisLabels: ["C", "H", "W"] },
  ],
});
```
````
`````

### Metadata-only tensors

Use metadata-only tensors when you want shape, labels, slicing, and layout
without sending the full dense payload up front.

`````{tab-set}
````{tab-item} Python
```python
import tensor_viz

tensor_viz.viz(
    {
        "activations": tensor_viz.TensorMeta((32, 64, 64), labels="C H W"),
        "weights": tensor_viz.TensorMeta((64, 32, 3, 3), labels="O I K0 K1"),
    }
)
```
````

````{tab-item} TypeScript
```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    { name: "activations", dtype: "float32", shape: [32, 64, 64], axisLabels: ["C", "H", "W"], placeholderData: true },
    { name: "weights", dtype: "float32", shape: [64, 32, 3, 3], axisLabels: ["O", "I", "K0", "K1"], placeholderData: true },
  ],
});
```
````
`````

### Multiple tensors in one view

Pass multiple tensors into one document when you want them side by side in one
viewer tab.

`````{tab-set}
````{tab-item} Python
```python
import numpy as np
import tensor_viz

tensor_viz.viz(
    {"weights": np.random.randn(16, 16), "bias": np.random.randn(16)},
    labels={"weights": "O I", "bias": "O"},
)
```
````

````{tab-item} TypeScript
```ts
import { createBundleManifest } from "@tensor-viz/viewer-core";

const bundle = createBundleManifest({
  tensors: [
    { name: "weights", dtype: "float32", shape: [16, 16], axisLabels: ["O", "I"] },
    { name: "bias", dtype: "float32", shape: [16], axisLabels: ["O"] },
  ],
});
```
````
`````

### Tabs

Use tabs when you want separate viewer documents instead of one shared view.

`````{tab-set}
````{tab-item} Python
```python
import numpy as np
import tensor_viz

inputs = tensor_viz.Tab("inputs")
inputs.viz(np.random.randn(3, 32, 32), name="image", labels="C H W")

weights = tensor_viz.Tab("weights")
weights.viz({"conv": np.random.randn(16, 3, 3, 3)}, labels={"conv": "O I K0 K1"})

tensor_viz.viz([inputs, weights])
```
````

````{tab-item} TypeScript
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
````
`````

### Custom colors

The shared session-builder layer stores colors as manifest instructions.

`````{tab-set}
````{tab-item} Python
```python
import numpy as np
import tensor_viz

weights = {"weights": np.arange(12, dtype=np.float32).reshape(3, 4)}

session_data = tensor_viz.create_session_data(
    weights,
    labels={"weights": "O I"},
    color_instructions={
        "tensor-1": [
            {"mode": "rgba", "kind": "coords", "coords": [[0, 0], [1, 1]], "color": [0, 90, 255, 255]}
        ]
    },
)

tensor_viz.viz(weights, session_data=session_data)
```
````

````{tab-item} TypeScript
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
````
`````

### Python session lifecycle convenience

This part is Python-only because it owns the local server lifecycle:

```python
import numpy as np
import tensor_viz

session = tensor_viz.viz(np.random.randn(4, 4), open_browser=False, keep_alive=False)
print(session.url)
session.close()
```

## 2. `viewer-demo`

Use `@tensor-viz/viewer-demo` when you want the full app shell: renderer,
widgets, menus, tabs, and sidebar UI.

### Install

```bash
npm install
```

### Mount the full demo app

The current mount API is iframe-backed.

```ts
import { mountDemoApp } from "@tensor-viz/viewer-demo";

const host = document.getElementById("demo-host");
if (!host) throw new Error("missing demo host");

const app = mountDemoApp(host, { src: "/" });
```

## 3. `viewer-core`

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

`addTensor(...)` uses the default axis labels `A B C ... Z AA AB ...`. To load
custom axis labels, prebuilt views, or manifest-provided colors, use
`loadBundleData(...)`.

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

### Lifecycle

```ts
viewer.destroy();
```

# Usage

`````{tab-set}
````{tab-item} Python
`labels=` is optional throughout the Python API. If you omit it, tensor axes
default to `A B C ... Z AA AB ...` in axis order.

## Launch one tensor

```python
import numpy as np
import tensor_viz

x = np.random.randn(8, 16, 32)
tensor_viz.viz(x, labels="C H W")
```

Omitting `labels=` for the same tensor would display it as `A B C`.

## Launch metadata-only tensors

Use `TensorMeta(...)` when you want tensor-viz to render shape, labels, tabs,
and slicing controls without sending the full tensor payload to the browser.
Metadata-only tensors render with the normal layout and default base color, but
hovered numeric values and heatmap ranges are unavailable.

```python
tensor_viz.viz(
    {
        "activations": tensor_viz.TensorMeta((32, 64, 64), labels="C H W"),
        "weights": tensor_viz.TensorMeta((64, 32, 3, 3), labels="O I K0 K1"),
    }
)
```

## Launch multiple tensors in one view

Pass a sequence or mapping directly to `tensor_viz.viz(...)` to place multiple
tensors in the same viewer tab. They open side by side in one view and each
tensor keeps its own Tensor View state. Use `Tab(...)` only when you want
separate tabs instead of one shared view.

Sequences produce numbered tensors:

```python
tensor_viz.viz(
    [np.random.randn(4, 4), np.random.randn(2, 8, 8)],
    labels=["H W", "C H W"],
)
```

Mappings preserve names from the keys:

```python
tensor_viz.viz(
    {"weights": np.random.randn(16, 16), "bias": np.random.randn(16)},
    labels={"weights": "O I", "bias": "O"},
)
```

## Tabs

```python
import tensor_viz

first = tensor_viz.Tab("inputs")
first.viz(np.random.randn(3, 32, 32), name="image", labels="C H W")

second = tensor_viz.Tab("weights")
second.viz({"conv": np.random.randn(16, 3, 3, 3)}, labels={"conv": "O I K0 K1"})

tensor_viz.viz([first, second])
```

## Custom colors and prebuilt session data

Use `create_session_data()` when you need to prepare the payload ahead of time
or attach color instructions. In the Python API, custom colors are part of the
session payload, not a live mutation API. To change them, update
`color_instructions`, rebuild `SessionData`, and pass the new payload to
`viz(..., session_data=...)`.

`color_instructions` is a mapping from generated tensor id to a list of color
instructions. For a non-tab input, ids follow input order: the first tensor is
`tensor-1`, the second is `tensor-2`, and so on.

Each instruction has:

- `mode="rgba"` for `[r, g, b, a]` tuples with 0-255 channels.
- `mode="hs"` for `[hue, saturation]`, which keeps value-dependent brightness.
- `kind="dense"` to supply one color tuple per element.
- `kind="coords"` to color explicit coordinates.
- `kind="region"` to color a strided block from `base`, `shape`, and `jumps`.

```python
import numpy as np
import tensor_viz

base = np.arange(12, dtype=np.float32).reshape(3, 4)
weights = {"weights": base}
```

Dense RGBA:

```python
session_data = tensor_viz.create_session_data(
    weights,
    labels={"weights": "O I"},
    color_instructions={
        "tensor-1": [
            {"mode": "rgba", "kind": "dense", "values": [0, 90, 255, 255] * base.size}
        ]
    },
)
tensor_viz.viz(weights, session_data=session_data)
```

Coordinate RGBA:

```python
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

Region RGBA:

```python
session_data = tensor_viz.create_session_data(
    weights,
    labels={"weights": "O I"},
    color_instructions={
        "tensor-1": [
            {"mode": "rgba", "kind": "region", "base": [0, 0], "shape": [3, 2], "jumps": [1, 2], "color": [0, 90, 255, 255]}
        ]
    },
)
tensor_viz.viz(weights, session_data=session_data)
```

Coordinate HS:

```python
session_data = tensor_viz.create_session_data(
    weights,
    labels={"weights": "O I"},
    color_instructions={
        "tensor-1": [
            {"mode": "hs", "kind": "coords", "coords": [[0, 0], [1, 1]], "color": [240, 1]}
        ]
    },
)
tensor_viz.viz(weights, session_data=session_data)
```

## Session lifecycle

`viz()` returns a `ViewerSession` with the bound URL plus lifecycle helpers:

```python
session = tensor_viz.viz(np.random.randn(4, 4), open_browser=False, keep_alive=False)
print(session.url)
session.close()
```
````

````{tab-item} TypeScript
The browser-facing code lives in `packages/viewer-core`.

## Install

```bash
npm install
```

## Create a viewer and launch one tensor

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

## Create a metadata-only tensor

Use `addMetadataTensor(...)` when the host app only has shape, dtype, and axis
labels. The tensor still participates in layout, slicing, and tabs, but it does
not contribute hover values or heatmap ranges.

```ts
const handle = viewer.addMetadataTensor([64, 32, 3, 3], "float32", "weights", undefined, [
  "O",
  "I",
  "K0",
  "K1",
]);

viewer.setActiveTensor(handle.id);
```

## Launch multiple tensors in one view

Call `addTensor(...)` multiple times on the same viewer to place multiple
tensors in one shared view. If you omit `offset`, later tensors are laid out
side by side automatically. Pass an explicit `offset` when you want manual
placement.

```ts
const weights = viewer.addTensor([4, 4], new Float32Array(16), "weights");
const activations = viewer.addTensor([2, 8, 8], new Float32Array(2 * 8 * 8), "activations");

viewer.setActiveTensor(activations.id);
```

## Tabs

`viewer-core` does not include a built-in tab model. If you want tabs in a web
app, create them in your host UI and either:

- keep one `TensorViewer` instance per tab, or
- swap manifests and tensor buffers into one viewer with `loadBundleData(...)`.

## Custom colors

In TypeScript, custom colors are live viewer mutations through
`viewer.colorTensor(...)`. Reapplying colors replaces the previous custom color
state for that tensor, and `clearTensorColors(...)` removes them.

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

## Load manifest data

Use `loadBundleData(...)` when you already have decoded tensor buffers plus a
manifest with custom axis labels, saved view state, or bundle-style color
instructions.

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

## Lifecycle

Call `destroy()` when the host element is being removed.

```ts
viewer.destroy();
```
````
`````

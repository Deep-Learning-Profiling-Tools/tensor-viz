# Python Usage

Use the Python API when Python owns the tensor data and you want the shortest
path to a local viewer session. This surface is ergonomic: pass arrays,
mappings, `TensorMeta`, and `Tab` objects, and `tensor_viz` builds the session
document and serves the browser app for you.

## One tensor

```python
import numpy as np
import tensor_viz

x = np.random.randn(8, 16, 32)
tensor_viz.viz(x)
```

By default, tensor-viz labels axes as `A B C ... Z A0 B0 ...` in tensor order.

## Custom axis labels

Use `labels=...` when you want names such as channel, height, and width instead
of the default `A B C ...` labels. Custom labels must start with one letter and
may use only non-letters after it, so `C`, `H`, `W`, `B0`, and `T11` are valid
while `Batch` and `CH` are invalid. In Python string form, separate
multi-character labels with spaces, such as `labels="B0 T11"`; compact strings
like `labels="BCHW"` are only for single-letter labels.

```python
import numpy as np
import tensor_viz

x = np.random.randn(8, 16, 32)
tensor_viz.viz(x, labels="C H W")
```

## Multiple tensors in one view

Pass a mapping when you want multiple tensors side by side in one viewer tab.

```python
import numpy as np
import tensor_viz

tensor_viz.viz(
    {"weights": np.random.randn(16, 16), "bias": np.random.randn(16)},
    labels={"weights": "O I", "bias": "O"},
)
```

## Metadata-only tensors

Use `TensorMeta(...)` when you want shape, labels, slicing, and layout without
sending the full dense payload up front.

```python
import tensor_viz

tensor_viz.viz(
    {
        "activations": tensor_viz.TensorMeta((32, 64, 64), labels="C H W"),
        "weights": tensor_viz.TensorMeta((64, 32, 3, 3), labels="O I K0 K1"),
    }
)
```

## Tabs

Use `Tab(...)` when you want separate viewer documents instead of one shared
view.

```python
import numpy as np
import tensor_viz

inputs = tensor_viz.Tab("inputs")
inputs.viz(np.random.randn(3, 32, 32), name="image", labels="C H W")

weights = tensor_viz.Tab("weights")
weights.viz({"conv": np.random.randn(16, 3, 3, 3)}, labels={"conv": "O I K0 K1"})

tensor_viz.viz([inputs, weights])
```

## Custom colors and prebuilt session data

Use `create_session_data(...)` when you want to prebuild the raw session payload
before launching the viewer. The main cases are:

- you want to inspect or save the exact manifest and tensor-byte payload
- you want to attach `color_instructions`
- you want to build the payload in one step and pass that exact payload into
  `viz(...)` later

### Inspect the raw payload

`create_session_data(...)` returns `manifest_bytes` plus a `tensor_bytes` mapping
for the dense payloads. This is useful when you want to inspect, cache, or save
the exact browser payload before opening the viewer.

```python
import json
import numpy as np
import tensor_viz

x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
session_data = tensor_viz.create_session_data(x, labels="C H W")

manifest = json.loads(session_data.manifest_bytes)
print(manifest["tabs"][0]["tensors"][0]["axisLabels"])
print(sorted(session_data.tensor_bytes))
```

### Add colors before launch

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

### Build now, launch later

This is useful when one step prepares the payload and a later step decides when
to launch the viewer.

```python
import numpy as np
import tensor_viz

weights = {"weights": np.random.randn(16, 16).astype(np.float32)}
session_data = tensor_viz.create_session_data(weights, labels={"weights": "O I"})

# later
tensor_viz.viz(weights, session_data=session_data)
```

## Session lifecycle

`viz(...)` returns a `ViewerSession` handle. Use it when you want to inspect the
URL, suppress browser launch, or stop the local server yourself.

```python
import numpy as np
import tensor_viz

session = tensor_viz.viz(np.random.randn(4, 4), open_browser=False, keep_alive=False)
print(session.url)
session.close()
```

# Python Usage

## Launch one tensor

```python
import numpy as np
import tensor_viz

x = np.random.randn(8, 16, 32)
tensor_viz.viz(x, labels="C H W")
```

## Launch multiple tensors

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
second.viz({"conv": np.random.randn(16, 3, 3, 3)}, labels={"conv": "O I KH KW"})

tensor_viz.viz([first, second])
```

## Prebuilt session data

Use `create_session_data()` when you need to prepare the payload ahead of time or attach color instructions.

```python
import numpy as np
import tensor_viz

base = np.arange(12, dtype=np.float32).reshape(3, 4)
session_data = tensor_viz.create_session_data(
    {"weights": base},
    labels={"weights": "O I"},
    color_instructions={
        "tensor-1": [
            {"mode": "rgba", "kind": "coords", "coords": [[0, 0], [1, 1]], "color": [0, 90, 255, 255]}
        ]
    },
)
tensor_viz.viz(base, session_data=session_data)
```

## Session lifecycle

`viz()` returns a `ViewerSession` with the bound URL plus lifecycle helpers:

```python
session = tensor_viz.viz(np.random.randn(4, 4), open_browser=False, keep_alive=False)
print(session.url)
session.close()
```

See the full Python API reference for signatures and module-level details.

"""Session helpers for browser transport."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

AxisLabelSpec = str | Sequence[str]
TensorLabels = (
    AxisLabelSpec
    | Sequence[AxisLabelSpec | None]
    | Mapping[str, AxisLabelSpec | None]
)
SupportedDType = str
_SUPPORTED_DTYPES = {"float64", "float32", "int32", "uint8"}


@dataclass(frozen=True)
class TensorMeta:
    """Metadata-only tensor description for shape-driven visualization.

    Use this when you want tensor-viz to render layout, slicing, labels, and
    tabs without shipping the full tensor payload to the browser.

    Examples
    --------
    >>> import tensor_viz
    >>> tensor_viz.TensorMeta((1024, 1024, 64), dtype="float32", labels="H W C")
    TensorMeta(shape=(1024, 1024, 64), dtype='float32', labels='H W C', name=None)
    """

    shape: tuple[int, ...]
    dtype: SupportedDType = "float32"
    labels: AxisLabelSpec | None = None
    name: str | None = None


TensorValue = np.ndarray | TensorMeta
TensorInput = TensorValue | Sequence[TensorValue] | Mapping[str, TensorValue]


@dataclass(frozen=True)
class SessionTensor:
    """Tensor payload prepared for a browser session."""

    tensor_id: str
    name: str
    shape: tuple[int, ...]
    dtype: SupportedDType
    axis_labels: tuple[str, ...]
    array: np.ndarray | None


@dataclass(frozen=True)
class SessionData:
    """Raw session manifest plus tensor byte payloads.

    Usually built with :func:`create_session_data` and then passed to
    :func:`tensor_viz.viz`.
    """

    manifest_bytes: bytes
    tensor_bytes: dict[str, bytes]


@dataclass
class Tab:
    """A named viewer tab that can hold one or more tensors.

    Examples
    --------
    >>> import numpy as np
    >>> import tensor_viz
    >>> tab = tensor_viz.Tab("weights")
    >>> tab.viz({"conv": np.random.randn(16, 3, 3, 3)}, labels={"conv": "O I K0 K1"})
    """

    title: str
    _tensors: list[tuple[TensorInput, str | None, TensorLabels | None]]

    def __init__(self, title: str):
        self.title = title
        self._tensors = []

    def viz(
        self,
        tensor: TensorInput,
        *,
        name: str | None = None,
        labels: TensorLabels | None = None,
    ) -> Tab:
        """Append one tensor input to this tab.

        Examples
        --------
        >>> import numpy as np
        >>> import tensor_viz
        >>> tab = tensor_viz.Tab("activations")
        >>> _ = tab.viz(np.random.randn(2, 64, 32, 32), name="x", labels="N C H W")
        >>> _ = tab.viz(np.random.randn(2, 64, 32, 32), name="y", labels="N C H W")
        """

        self._tensors.append((tensor, name, labels))
        return self


def _axis_label(index: int) -> str:
    """Return the default viewer label for one axis."""

    value = index
    out = ""
    while True:
        out = chr(65 + (value % 26)) + out
        value = (value // 26) - 1
        if value < 0:
            return out


def _normalize_axis_labels(rank: int, labels: AxisLabelSpec | None) -> tuple[str, ...]:
    """Normalize one tensor's axis labels."""

    defaults = tuple(_axis_label(axis) for axis in range(rank))
    if labels is None:
        return defaults
    if isinstance(labels, str):
        raw_labels = labels.split() if any(char.isspace() for char in labels) else list(labels)
    else:
        raw_labels = [str(label).strip() for label in labels]
    if len(raw_labels) != rank:
        raise ValueError(
            f"Expected {rank} axis labels, got {len(raw_labels)} (received {raw_labels!r})."
        )
    axis_labels = tuple(label.upper() for label in raw_labels)
    invalid_labels = [label for label in axis_labels if not re.fullmatch(r"[A-Z][^A-Z]*", label)]
    if invalid_labels:
        raise ValueError(
            "Axis labels must start with a letter and may only use non-letters after it "
            f"(received {invalid_labels[0]!r})."
        )
    seen: set[str] = set()
    duplicate = next(
        (label for label in axis_labels if (lower := label.lower()) in seen or seen.add(lower)),
        None,
    )
    if duplicate is not None:
        raise ValueError(f"Axis labels must be unique (received duplicate {duplicate!r}).")
    return axis_labels


def _normalize_dtype(dtype: Any) -> SupportedDType:
    """Normalize one viewer dtype string."""

    normalized = np.dtype(dtype).name
    if normalized not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)!r}."
        )
    return normalized


def _build_session_tensor(
    tensor_id: str,
    tensor: TensorValue,
    *,
    name: str,
    labels: AxisLabelSpec | None,
) -> SessionTensor:
    """Normalize one tensor or metadata-only tensor into session form."""

    if isinstance(tensor, np.ndarray):
        array = np.ascontiguousarray(tensor)
        if array.dtype not in (np.float64, np.float32, np.int32, np.uint8):
            array = array.astype(np.float64, copy=False)
        return SessionTensor(
            tensor_id=tensor_id,
            name=name,
            shape=tuple(int(dim) for dim in array.shape),
            dtype=_normalize_dtype(array.dtype),
            axis_labels=_normalize_axis_labels(array.ndim, labels),
            array=array,
        )
    shape = tuple(int(dim) for dim in tensor.shape)
    return SessionTensor(
        tensor_id=tensor_id,
        name=tensor.name or name,
        shape=shape,
        dtype=_normalize_dtype(tensor.dtype),
        axis_labels=_normalize_axis_labels(len(shape), labels or tensor.labels),
        array=None,
    )


def _normalize_inputs(
    tensor: TensorInput,
    name: str | None,
    labels: TensorLabels | None = None,
) -> list[SessionTensor]:
    """Normalize Python tensor inputs plus optional axis-label overrides."""

    if isinstance(tensor, (np.ndarray, TensorMeta)):
        if labels is not None and isinstance(labels, Mapping):
            raise TypeError(
                "labels must be a string or label sequence when tensor is a single tensor."
            )
        return [
            _build_session_tensor("tensor-1", tensor, name=name or "Tensor 1", labels=labels)
        ]
    if isinstance(tensor, Mapping):
        if labels is None:
            label_map: Mapping[str, AxisLabelSpec | None] = {}
        elif isinstance(labels, Mapping):
            label_map = labels
        else:
            raise TypeError("labels must be a mapping when tensor is a mapping.")
        return [
            _build_session_tensor(
                f"tensor-{index}",
                value,
                name=tensor_name,
                labels=label_map.get(tensor_name),
            )
            for index, (tensor_name, value) in enumerate(tensor.items(), start=1)
        ]
    if labels is None:
        label_values = [None] * len(tensor)
    elif isinstance(labels, str) or isinstance(labels, Mapping):
        raise TypeError("labels must be a sequence when tensor is a sequence.")
    else:
        label_values = list(labels)
    if len(label_values) != len(tensor):
        raise ValueError(
            f"Expected {len(tensor)} label entries for the tensor sequence, got {len(label_values)}."
        )
    return [
        _build_session_tensor(
            f"tensor-{index}",
            value,
            name=value.name if isinstance(value, TensorMeta) and value.name else f"Tensor {index}",
            labels=label_values[index - 1],
        )
        for index, value in enumerate(tensor, start=1)
    ]


def _normalize_tab(tab: Tab, index: int) -> dict[str, Any]:
    tensors = [
        SessionTensor(
            tensor_id=f"tensor-{tensor_index}",
            name=tensor.name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            axis_labels=tensor.axis_labels,
            array=tensor.array,
        )
        for tensor_index, tensor in enumerate(
            [
                normalized
                for tensor_input, name, labels in tab._tensors
                for normalized in _normalize_inputs(tensor_input, name, labels)
            ],
            start=1,
        )
    ]
    return {
        "id": f"tab-{index}",
        "title": tab.title,
        "tensors": tensors,
    }


def _build_viewer_manifest(tensors: list[SessionTensor]) -> dict[str, Any]:
    return {
        "version": 1,
        "displayMode": "2d",
        "heatmap": True,
        "showDimensionLines": True,
        "showInspectorPanel": True,
        "showHoverDetailsPanel": True,
        "camera": {
            "position": [0, 0, 30],
            "target": [0, 0, 0],
            "rotation": [0, 0, 0],
            "zoom": 1,
        },
        "tensors": [
            {
                "id": entry.tensor_id,
                "name": entry.name,
                "view": {
                    "view": " ".join(entry.axis_labels),
                    "hiddenIndices": [0] * len(entry.shape),
                },
            }
            for entry in tensors
        ],
        "activeTensorId": tensors[0].tensor_id if tensors else None,
    }


def _build_tensor_manifest(
    tensors: list[SessionTensor],
    *,
    color_instructions: Mapping[str, list[dict[str, Any]]] | None = None,
    data_prefix: str = "tensors",
) -> list[dict[str, Any]]:
    return [
        {
            "id": entry.tensor_id,
            "name": entry.name,
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "axisLabels": list(entry.axis_labels),
            "byteOrder": "little",
            "placeholderData": entry.array is None,
            **(
                {"dataFile": f"{data_prefix}/{entry.tensor_id}.bin"}
                if entry.array is not None
                else {}
            ),
            "view": {
                "view": " ".join(entry.axis_labels),
                "hiddenIndices": [0] * len(entry.shape),
            },
            "colorInstructions": (color_instructions or {}).get(entry.tensor_id),
        }
        for entry in tensors
    ]


def create_session_data(
    tensor: TensorInput | Tab | Sequence[Tab],
    *,
    name: str | None = None,
    labels: TensorLabels | None = None,
    color_instructions: Mapping[str, list[dict[str, Any]]] | None = None,
) -> SessionData:
    """Build raw browser session data for tensors or tabs.

    Parameters
    ----------
    tensor:
        One NumPy tensor, one :class:`TensorMeta`, a sequence or mapping of
        either, one :class:`Tab`, or a sequence of tabs.
    name:
        Session title for non-tab inputs, or tensor name for a single ndarray.
    labels:
        Optional axis-label overrides. Single ndarrays accept one label spec,
        sequences accept one spec per tensor, and mappings accept a label map.
        Tab inputs must attach labels through :meth:`Tab.viz`. When omitted,
        axes use the default viewer labels ``A B C ... Z AA AB ...``.
    color_instructions:
        Optional viewer color instructions keyed by generated tensor id such as
        ``tensor-1`` and ``tensor-2`` in input order. Use ``kind="dense"``
        for one color per element, ``kind="coords"`` for explicit coordinates,
        and ``kind="region"`` for a strided block. ``mode="rgba"`` expects
        ``[r, g, b, a]`` tuples, while ``mode="hs"`` expects
        ``[hue, saturation]``.

    Examples
    --------
    Build session data ahead of time:

    >>> import numpy as np
    >>> import tensor_viz
    >>> tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    >>> session_data = tensor_viz.create_session_data(tensor, labels="O I")
    >>> len(session_data.tensor_bytes) > 0
    True

    Metadata-only session data:

    >>> meta = tensor_viz.TensorMeta((1024, 1024, 64), dtype="float32", labels="H W C")
    >>> session_data = tensor_viz.create_session_data(meta)
    >>> session_data.tensor_bytes
    {}

    Dense RGBA colors:

    >>> session_data = tensor_viz.create_session_data(
    ...     {"weights": tensor},
    ...     labels={"weights": "O I"},
    ...     color_instructions={
    ...         "tensor-1": [
    ...             {"mode": "rgba", "kind": "dense", "values": [255, 0, 0, 255] * tensor.size}
    ...         ]
    ...     },
    ... )

    Coordinate RGBA colors:

    >>> session_data = tensor_viz.create_session_data(
    ...     {"weights": tensor},
    ...     labels={"weights": "O I"},
    ...     color_instructions={
    ...         "tensor-1": [
    ...             {"mode": "rgba", "kind": "coords", "coords": [[0, 0]], "color": [255, 0, 0, 255]}
    ...         ]
    ...     },
    ... )

    Region RGBA colors:

    >>> session_data = tensor_viz.create_session_data(
    ...     {"weights": tensor},
    ...     labels={"weights": "O I"},
    ...     color_instructions={
    ...         "tensor-1": [
    ...             {"mode": "rgba", "kind": "region", "base": [0, 0], "shape": [2, 2], "jumps": [1, 1], "color": [255, 0, 0, 255]}
    ...         ]
    ...     },
    ... )

    Coordinate HS colors:

    >>> session_data = tensor_viz.create_session_data(
    ...     {"weights": tensor},
    ...     labels={"weights": "O I"},
    ...     color_instructions={
    ...         "tensor-1": [
    ...             {"mode": "hs", "kind": "coords", "coords": [[0, 0]], "color": [240, 1]}
    ...         ]
    ...     },
    ... )
    """

    if isinstance(tensor, Tab):
        if labels is not None:
            raise TypeError("Use Tab.viz(..., labels=...) when building sessions from Tab inputs.")
        tabs = [_normalize_tab(tensor, 1)]
    elif isinstance(tensor, Sequence) and tensor and all(
        isinstance(entry, Tab) for entry in tensor
    ):
        if labels is not None:
            raise TypeError("Use Tab.viz(..., labels=...) when building sessions from Tab inputs.")
        tabs = [_normalize_tab(entry, index) for index, entry in enumerate(tensor, start=1)]
    else:
        tensors = _normalize_inputs(tensor, name, labels)
        tabs = [{"id": "tab-1", "title": name or "Tensor Viz", "tensors": tensors}]

    manifest = {
        "version": 1,
        "tabs": [],
    }
    tensor_bytes: dict[str, bytes] = {}
    for tab in tabs:
        tensor_manifest = _build_tensor_manifest(
            tab["tensors"],
            color_instructions=color_instructions,
            data_prefix=f"tabs/{tab['id']}/tensors",
        )
        manifest["tabs"].append(
            {
                "id": tab["id"],
                "title": tab["title"],
                "viewer": _build_viewer_manifest(tab["tensors"]),
                "tensors": tensor_manifest,
            }
        )
        for entry in tab["tensors"]:
            if entry.array is None:
                continue
            tensor_bytes[f"tabs/{tab['id']}/tensors/{entry.tensor_id}.bin"] = entry.array.tobytes(
                order="C"
            )
    return SessionData(
        manifest_bytes=json.dumps(manifest).encode("utf-8"),
        tensor_bytes=tensor_bytes,
    )

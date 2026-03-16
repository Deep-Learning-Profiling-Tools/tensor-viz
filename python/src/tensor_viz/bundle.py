"""Session helpers for browser transport."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

TensorInput = np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray]
AxisLabelSpec = str | Sequence[str]
TensorLabels = (
    AxisLabelSpec
    | Sequence[AxisLabelSpec | None]
    | Mapping[str, AxisLabelSpec | None]
)


@dataclass(frozen=True)
class SessionTensor:
    """Tensor payload prepared for a browser session."""

    tensor_id: str
    name: str
    array: np.ndarray
    axis_labels: tuple[str, ...]


@dataclass(frozen=True)
class SessionData:
    """Raw session manifest plus tensor byte payloads."""

    manifest_bytes: bytes
    tensor_bytes: dict[str, bytes]


@dataclass
class Tab:
    """A named viewer tab that can hold one or more tensors."""

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
        """Append tensors to this tab."""

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
        raise ValueError(f"Expected {rank} axis labels, got {len(raw_labels)}.")
    axis_labels = tuple(label.upper() for label in raw_labels)
    if any(not re.fullmatch(r"[A-Z][^A-Z]*", label) for label in axis_labels):
        raise ValueError(
            "Axis labels must start with a letter and may only use non-letters after it."
        )
    if len(set(label.lower() for label in axis_labels)) != rank:
        raise ValueError("Axis labels must be unique.")
    return axis_labels


def _normalize_inputs(
    tensor: TensorInput,
    name: str | None,
    labels: TensorLabels | None = None,
) -> list[SessionTensor]:
    """Normalize Python tensor inputs plus optional axis-label overrides."""

    def normalize_array(array: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(array)
        if contiguous.dtype in (np.float64, np.float32, np.int32, np.uint8):
            return contiguous
        return contiguous.astype(np.float64, copy=False)

    if isinstance(tensor, np.ndarray):
        if labels is not None and isinstance(labels, Mapping):
            raise TypeError("labels must be a string or label sequence when tensor is a single ndarray.")
        return [
            SessionTensor(
                "tensor-1",
                name or "Tensor 1",
                normalize_array(tensor),
                _normalize_axis_labels(tensor.ndim, labels),
            )
        ]
    if isinstance(tensor, Mapping):
        if labels is None:
            label_map: Mapping[str, AxisLabelSpec | None] = {}
        elif isinstance(labels, Mapping):
            label_map = labels
        else:
            raise TypeError("labels must be a mapping when tensor is a mapping.")
        return [
            SessionTensor(
                f"tensor-{index}",
                tensor_name,
                normalize_array(array),
                _normalize_axis_labels(array.ndim, label_map.get(tensor_name)),
            )
            for index, (tensor_name, array) in enumerate(tensor.items(), start=1)
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
        SessionTensor(
            f"tensor-{index}",
            f"Tensor {index}",
            normalize_array(array),
            _normalize_axis_labels(array.ndim, label_values[index - 1]),
        )
        for index, array in enumerate(tensor, start=1)
    ]


def _normalize_tab(tab: Tab, index: int) -> dict[str, Any]:
    tensors = [
        SessionTensor(f"tensor-{tensor_index}", tensor.name, tensor.array, tensor.axis_labels)
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
                "offset": [0, 0, 0] if index == 0 else [index * 6, 0, 0],
                "view": {
                    "visible": " ".join(entry.axis_labels),
                    "hiddenIndices": [0] * entry.array.ndim,
                },
            }
            for index, entry in enumerate(tensors)
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
            "dtype": str(entry.array.dtype),
            "shape": list(entry.array.shape),
            "axisLabels": list(entry.axis_labels),
            "byteOrder": "little",
            "dataFile": f"{data_prefix}/{entry.tensor_id}.bin",
            "offset": [0, 0, 0] if index == 0 else [index * 6, 0, 0],
            "view": {
                "visible": " ".join(entry.axis_labels),
                "hiddenIndices": [0] * entry.array.ndim,
            },
            "colorInstructions": (color_instructions or {}).get(entry.tensor_id),
        }
        for index, entry in enumerate(tensors)
    ]


def create_session_data(
    tensor: TensorInput | Tab | Sequence[Tab],
    *,
    name: str | None = None,
    labels: TensorLabels | None = None,
    color_instructions: Mapping[str, list[dict[str, Any]]] | None = None,
) -> SessionData:
    """Build raw session data for one or more NumPy arrays."""

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
            tensor_bytes[f"tabs/{tab['id']}/tensors/{entry.tensor_id}.bin"] = entry.array.tobytes(
                order="C"
            )
    return SessionData(
        manifest_bytes=json.dumps(manifest).encode("utf-8"),
        tensor_bytes=tensor_bytes,
    )

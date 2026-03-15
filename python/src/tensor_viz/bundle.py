"""Bundle helpers for browser sessions."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

TensorInput = np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SessionTensor:
    """Tensor payload prepared for bundling."""

    tensor_id: str
    name: str
    array: np.ndarray


@dataclass
class Tab:
    """A named viewer tab that can hold one or more tensors."""

    title: str
    _tensors: list[tuple[TensorInput, str | None]]

    def __init__(self, title: str):
        self.title = title
        self._tensors = []

    def viz(
        self,
        tensor: TensorInput,
        *,
        name: str | None = None,
    ) -> Tab:
        """Append tensors to this tab."""

        self._tensors.append((tensor, name))
        return self


def _normalize_inputs(
    tensor: TensorInput,
    name: str | None,
) -> list[SessionTensor]:
    def normalize_array(array: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(array)
        if contiguous.dtype in (np.float64, np.float32, np.int32, np.uint8):
            return contiguous
        return contiguous.astype(np.float64, copy=False)

    if isinstance(tensor, np.ndarray):
        return [SessionTensor("tensor-1", name or "Tensor 1", normalize_array(tensor))]
    if isinstance(tensor, Mapping):
        return [
            SessionTensor(f"tensor-{index}", tensor_name, normalize_array(array))
            for index, (tensor_name, array) in enumerate(tensor.items(), start=1)
        ]
    return [
        SessionTensor(f"tensor-{index}", f"Tensor {index}", normalize_array(array))
        for index, array in enumerate(tensor, start=1)
    ]


def _normalize_tab(tab: Tab, index: int) -> dict[str, Any]:
    tensors = [
        SessionTensor(f"tensor-{tensor_index}", tensor.name, tensor.array)
        for tensor_index, tensor in enumerate(
            [
                normalized
                for tensor_input, name in tab._tensors
                for normalized in _normalize_inputs(tensor_input, name)
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
                    "visible": " ".join(chr(65 + axis) for axis in range(entry.array.ndim)),
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
            "byteOrder": "little",
            "dataFile": f"{data_prefix}/{entry.tensor_id}.bin",
            "offset": [0, 0, 0] if index == 0 else [index * 6, 0, 0],
            "view": {
                "visible": " ".join(chr(65 + axis) for axis in range(entry.array.ndim)),
                "hiddenIndices": [0] * entry.array.ndim,
            },
            "colorInstructions": (color_instructions or {}).get(entry.tensor_id),
        }
        for index, entry in enumerate(tensors)
    ]


def create_session_bundle(
    tensor: TensorInput | Tab | Sequence[Tab],
    *,
    name: str | None = None,
    color_instructions: Mapping[str, list[dict[str, Any]]] | None = None,
) -> bytes:
    """Build a `.viz` session bundle from one or more NumPy arrays."""

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if isinstance(tensor, Tab):
            tabs = [_normalize_tab(tensor, 1)]
        elif isinstance(tensor, Sequence) and tensor and all(
            isinstance(entry, Tab) for entry in tensor
        ):
            tabs = [_normalize_tab(entry, index) for index, entry in enumerate(tensor, start=1)]
        else:
            tensors = _normalize_inputs(tensor, name)
            tabs = [{"id": "tab-1", "title": name or "Tensor Viz", "tensors": tensors}]

        manifest = {
            "version": 1,
            "tabs": [],
        }
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
                archive.writestr(
                    f"tabs/{tab['id']}/tensors/{entry.tensor_id}.bin",
                    entry.array.tobytes(order="C"),
                )
        archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
    return bundle.getvalue()

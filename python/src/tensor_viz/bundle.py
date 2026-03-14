"""Bundle helpers for browser sessions."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SessionTensor:
    """Tensor payload prepared for bundling."""

    tensor_id: str
    name: str
    array: np.ndarray


def _normalize_inputs(
    tensor: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
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


def create_session_bundle(
    tensor: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
    *,
    name: str | None = None,
    color_instructions: Mapping[str, list[dict[str, Any]]] | None = None,
) -> bytes:
    """Build a `.viz` session bundle from one or more NumPy arrays."""

    tensors = _normalize_inputs(tensor, name)
    manifest = {
        "version": 1,
        "viewer": {
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
            "tensors": [],
            "activeTensorId": tensors[0].tensor_id if tensors else None,
        },
        "tensors": [],
    }

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, entry in enumerate(tensors):
            offset = [0, 0, 0] if index == 0 else [index * 6, 0, 0]
            data_file = f"tensors/{entry.tensor_id}.bin"
            manifest["viewer"]["tensors"].append(
                {
                    "id": entry.tensor_id,
                    "name": entry.name,
                    "offset": offset,
                    "view": {
                        "visible": " ".join(chr(65 + axis) for axis in range(entry.array.ndim)),
                        "hiddenIndices": [0] * entry.array.ndim,
                    },
                }
            )
            manifest["tensors"].append(
                {
                    "id": entry.tensor_id,
                    "name": entry.name,
                    "dtype": str(entry.array.dtype),
                    "shape": list(entry.array.shape),
                    "byteOrder": "little",
                    "dataFile": data_file,
                    "offset": offset,
                    "view": {
                        "visible": " ".join(chr(65 + axis) for axis in range(entry.array.ndim)),
                        "hiddenIndices": [0] * entry.array.ndim,
                    },
                    "colorInstructions": (color_instructions or {}).get(entry.tensor_id),
                }
            )
            archive.writestr(data_file, entry.array.tobytes(order="C"))

        archive.writestr("manifest.json", json.dumps(manifest).encode("utf-8"))
    return bundle.getvalue()

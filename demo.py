import numpy as np

from tensor_viz import Tab, TensorMeta, create_session_data, viz

DEMO = 3


def demo_single_tensor() -> None:
    """Launch a basic single-tensor session."""

    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    viz(tensor, name="Single Tensor")


def demo_custom_labels() -> None:
    """Show custom multi-character axis labels on one tensor."""

    tensor = np.arange(2**11).reshape((2,) * 11)
    viz(
        tensor,
        labels="W0 W1 T0 T1 T2 T3 T4 R0 R1 R2 R3",
        name="Custom Labels",
    )


def demo_sequence_inputs() -> None:
    """Show a tensor sequence with per-tensor labels."""

    tensors = [
        np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
        np.arange(3 * 5, dtype=np.int32).reshape(3, 5),
        np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4),
    ]
    viz(tensors, labels=["BCH", "T0 T1", "HWC"], name="Sequence Inputs")


def demo_mapping_inputs() -> None:
    """Show named tensors with a mapping-based label override."""

    tensors = {
        "activations": np.linspace(-1, 1, 32, dtype=np.float32).reshape(2, 4, 4),
        "weights": np.arange(3 * 3 * 8 * 16, dtype=np.float32).reshape(
            3, 3, 8, 16
        ),
        "mask": np.arange(6, dtype=np.uint8).reshape(2, 3),
    }
    viz(
        tensors,
        labels={
            "activations": "BHW",
            "weights": "K0 K1 I O",
            "mask": "RC",
        },
        name="Mapping Inputs",
    )


def demo_session_data() -> None:
    """Show raw session data plus all color-instruction forms."""

    shape = (3, 4)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    blue_rgba = [0, 90, 255, 255]
    blue_hs = [240, 1]
    highlights = {(0, 1), (1, 2), (2, 0)}

    dense_rgba = [
        value
        for row in range(shape[0])
        for col in range(shape[1])
        for value in (blue_rgba if (row, col) in highlights else [144, 164, 174, 255])
    ]
    dense_hs = [
        value
        for row in range(shape[0])
        for col in range(shape[1])
        for value in (blue_hs if (row, col) in highlights else [0, 0])
    ]
    tensors = {
        "Dense RGBA": base,
        "Coords RGBA": base + 100,
        "Region RGBA": base + 200,
        "Dense HS": base + 300,
        "Coords HS": base + 400,
        "Region HS": base + 500,
    }
    session_data = create_session_data(
        tensors,
        color_instructions={
            "tensor-1": [{"mode": "rgba", "kind": "dense", "values": dense_rgba}],
            "tensor-2": [
                {
                    "mode": "rgba",
                    "kind": "coords",
                    "coords": [list(coord) for coord in sorted(highlights)],
                    "color": blue_rgba,
                }
            ],
            "tensor-3": [
                {
                    "mode": "rgba",
                    "kind": "region",
                    "base": [0, 0],
                    "shape": [3, 2],
                    "jumps": [1, 2],
                    "color": blue_rgba,
                }
            ],
            "tensor-4": [{"mode": "hs", "kind": "dense", "values": dense_hs}],
            "tensor-5": [
                {
                    "mode": "hs",
                    "kind": "coords",
                    "coords": [list(coord) for coord in sorted(highlights)],
                    "color": blue_hs,
                }
            ],
            "tensor-6": [
                {
                    "mode": "hs",
                    "kind": "region",
                    "base": [0, 0],
                    "shape": [3, 2],
                    "jumps": [1, 2],
                    "color": blue_hs,
                }
            ],
        },
    )
    viz(tensors, session_data=session_data)


def demo_tabs() -> None:
    """Show multi-tab sessions with labels attached at Tab.viz time."""

    activations = Tab("Activations").viz(
        np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
        name="layer-0",
        labels="BCH",
    )
    activations.viz(
        np.arange(4 * 4 * 8, dtype=np.float32).reshape(4, 4, 8),
        name="layer-1",
        labels="HWC",
    )

    weights = Tab("Weights").viz(
        np.arange(3 * 3 * 8 * 16, dtype=np.float32).reshape(3, 3, 8, 16),
        name="conv",
        labels="K0 K1 I O",
    )
    weights.viz(
        np.arange(16 * 32, dtype=np.float32).reshape(16, 32),
        name="proj",
        labels="IO",
    )
    viz([activations, weights], name="Tabs")


def demo_session_options() -> None:
    """Show manual session control without opening a browser."""

    session = viz(
        np.arange(12, dtype=np.float32).reshape(3, 4),
        name="Session Options",
        open_browser=False,
        host="127.0.0.1",
        port=0,
        keep_alive=False,
    )
    print(session.url)
    session.close()


def demo_image_like_tensor() -> None:
    """Show an HWC uint8 tensor without external image files."""

    image = np.zeros((128, 192, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 255, image.shape[1], dtype=np.uint8)
    image[..., 1] = np.linspace(255, 0, image.shape[0], dtype=np.uint8)[:, None]
    image[..., 2] = 128
    viz(image, labels="HWC", name="Image Like Tensor")


def demo_metadata_only() -> None:
    """Show metadata-only tensors without sending full numeric payloads."""

    viz(
        {
            "activations": TensorMeta((32, 64, 64), labels="C H W"),
            "weights": TensorMeta((64, 32, 3, 3), labels="O I K0 K1"),
        },
        name="Metadata Only",
    )


DEMOS = {
    0: demo_single_tensor,
    1: demo_custom_labels,
    2: demo_sequence_inputs,
    3: demo_mapping_inputs,
    4: demo_session_data,
    5: demo_tabs,
    6: demo_session_options,
    7: demo_image_like_tensor,
    8: demo_metadata_only,
}


if __name__ == "__main__":
    DEMOS[DEMO]()

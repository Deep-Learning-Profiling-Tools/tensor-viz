from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

import tensor_viz


class DocsExamplesTest(unittest.TestCase):
    def test_readme_quick_start_session_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            static_root = Path(tmpdir)
            (static_root / "index.html").write_text("<!doctype html><title>tensor-viz</title>")
            session = None
            try:
                with patch("tensor_viz.server._static_root", return_value=static_root):
                    session = tensor_viz.viz(
                        np.random.randn(4, 4),
                        open_browser=False,
                        keep_alive=False,
                    )
                    self.assertTrue(session.url.startswith("http://127.0.0.1:"))
                    with urlopen(f"{session.url}/api/session.json") as response:
                        payload = json.load(response)
                self.assertEqual(payload["tabs"][0]["tensors"][0]["shape"], [4, 4])
            finally:
                if session is not None:
                    session.close()

    def test_usage_metadata_only_python_example(self) -> None:
        session_data = tensor_viz.create_session_data(
            {
                "activations": tensor_viz.TensorMeta((32, 64, 64), labels="C H W"),
                "weights": tensor_viz.TensorMeta((64, 32, 3, 3), labels="O I K0 K1"),
            }
        )

        manifest = json.loads(session_data.manifest_bytes)
        tensors = manifest["tabs"][0]["tensors"]

        self.assertEqual([tensor["name"] for tensor in tensors], ["activations", "weights"])
        self.assertEqual(tensors[0]["axisLabels"], ["C", "H", "W"])
        self.assertTrue(tensors[0]["placeholderData"])
        self.assertNotIn("dataFile", tensors[0])
        self.assertEqual(tensors[1]["axisLabels"], ["O", "I", "K0", "K1"])
        self.assertEqual(session_data.tensor_bytes, {})

    def test_usage_tabs_and_colors_python_examples(self) -> None:
        inputs = tensor_viz.Tab("inputs")
        inputs.viz(np.random.randn(3, 32, 32), name="image", labels="C H W")

        weights = tensor_viz.Tab("weights")
        weights.viz(
            {"conv": np.random.randn(16, 3, 3, 3)},
            labels={"conv": "O I K0 K1"},
        )

        session_data = tensor_viz.create_session_data(
            {"weights": np.arange(12, dtype=np.float32).reshape(3, 4)},
            labels={"weights": "O I"},
            color_instructions={
                "tensor-1": [
                    {
                        "mode": "rgba",
                        "kind": "coords",
                        "coords": [[0, 0], [1, 1]],
                        "color": [0, 90, 255, 255],
                    }
                ]
            },
        )
        tabbed = tensor_viz.create_session_data([inputs, weights])

        manifest = json.loads(session_data.manifest_bytes)
        tabbed_manifest = json.loads(tabbed.manifest_bytes)

        self.assertEqual(
            manifest["tabs"][0]["tensors"][0]["colorInstructions"][0]["coords"],
            [[0, 0], [1, 1]],
        )
        self.assertEqual(
            [tab["title"] for tab in tabbed_manifest["tabs"]],
            ["inputs", "weights"],
        )
        self.assertEqual(
            tabbed_manifest["tabs"][0]["tensors"][0]["axisLabels"],
            ["C", "H", "W"],
        )
        self.assertEqual(
            tabbed_manifest["tabs"][1]["tensors"][0]["axisLabels"],
            ["O", "I", "K0", "K1"],
        )


if __name__ == "__main__":
    unittest.main()

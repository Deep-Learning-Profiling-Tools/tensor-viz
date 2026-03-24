"""Local server entrypoint for the browser demo."""

from __future__ import annotations

import contextlib
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from .bundle import SessionData, Tab, TensorInput, TensorLabels, create_session_data


def _static_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    workspace = repo_root / "packages" / "viewer-demo" / "dist"
    if workspace.exists() and any(workspace.iterdir()):
        return workspace
    packaged = Path(__file__).resolve().parent / "static"
    if packaged.exists() and any(packaged.iterdir()):
        return packaged
    return workspace


@dataclass
class ViewerSession:
    """Handle for a running browser session.

    Examples
    --------
    >>> import numpy as np
    >>> import tensor_viz
    >>> session = tensor_viz.viz(np.random.randn(4, 4), open_browser=False)
    >>> print(session.url)
    >>> session.close()
    """

    url: str
    _server: HTTPServer
    _thread: threading.Thread

    def close(self) -> None:
        """Stop the local HTTP server."""

        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def wait(self) -> None:
        """Block until the server thread exits."""

        self._thread.join()


def viz(
    tensor: TensorInput | Tab | Sequence[Tab],
    *,
    name: str | None = None,
    labels: TensorLabels | None = None,
    session_data: SessionData | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    keep_alive: bool = True,
) -> ViewerSession:
    """Launch the standalone viewer for tensors or tabs.

    Parameters
    ----------
    tensor:
        One NumPy tensor, one metadata-only tensor, a sequence or mapping of
        either, one :class:`tensor_viz.bundle.Tab`, or a sequence of tabs.
        Sequences and mappings place multiple tensors in the same viewer/tab,
        while :class:`tensor_viz.bundle.Tab` inputs create separate tabs.
    name:
        Session title for non-tab inputs, or tensor name for a single ndarray.
    labels:
        Optional axis-label overrides for non-tab inputs. When omitted, axes
        use the default viewer labels ``A B C ... Z A0 B0 ...``. Custom labels
        must start with one letter and may only use non-letters after it. In
        Python string form, separate multi-character labels with spaces, such as
        ``"B0 T11"``.
    session_data:
        Prebuilt raw session payloads. When omitted, the payload is derived
        from ``tensor``, ``name``, and ``labels``.
    open_browser:
        Open the local viewer URL in the default browser.
    host, port:
        Bind address for the local HTTP server.
    keep_alive:
        Keep the server thread non-daemon so short scripts stay alive after
        calling :func:`viz`.

    Examples
    --------
    Launch one tensor with custom axis labels:

    >>> import numpy as np
    >>> import tensor_viz
    >>> session = tensor_viz.viz(np.random.randn(8, 16, 32), labels="C H W", open_browser=False)
    >>> session.close()

    Launch named tensors:

    >>> tensors = {"weights": np.random.randn(16, 16), "bias": np.random.randn(16)}
    >>> session = tensor_viz.viz(tensors, labels={"weights": "O I", "bias": "O"}, open_browser=False)
    >>> session.close()

    Launch tabs:

    >>> first = tensor_viz.Tab("inputs")
    >>> first.viz(np.random.randn(3, 32, 32), name="image", labels="C H W")
    >>> second = tensor_viz.Tab("weights")
    >>> second.viz({"conv": np.random.randn(16, 3, 3, 3)}, labels={"conv": "O I K0 K1"})
    >>> session = tensor_viz.viz([first, second], open_browser=False)
    >>> session.close()

    Launch metadata-only tensors:

    >>> meta = tensor_viz.TensorMeta((1024, 1024, 64), dtype="float32", labels="H W C")
    >>> session = tensor_viz.viz(meta, open_browser=False)
    >>> session.close()
    """

    session_data = session_data or create_session_data(tensor, name=name, labels=labels)
    static_root = _static_root()
    if not static_root.exists():
        raise FileNotFoundError(
            "Viewer demo assets are missing. Build the frontend with `npm run build` first."
        )

    class Handler(SimpleHTTPRequestHandler):
        """Serve the demo app plus raw session bytes."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_root), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, max-age=0")
            super().end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/session.json":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(session_data.manifest_bytes)))
                self.end_headers()
                self.wfile.write(session_data.manifest_bytes)
                return

            if path.startswith("/api/"):
                data_path = path.removeprefix("/api/")
                payload = session_data.tensor_bytes.get(data_path)
                if payload is not None:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            target = static_root / path.lstrip("/")
            if path == "/" or not target.exists():
                self.path = "/index.html"
            return super().do_GET()

        def log_message(self, _format: str, *_args) -> None:
            return

    server = HTTPServer((host, port), Handler)
    # keep the server thread non-daemon by default so a simple
    # `tensor_viz.viz(tensor)` script does not exit before the browser loads.
    thread = threading.Thread(target=server.serve_forever, daemon=not keep_alive)
    thread.start()
    url = f"http://{host}:{server.server_port}"
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    return ViewerSession(url=url, _server=server, _thread=thread)

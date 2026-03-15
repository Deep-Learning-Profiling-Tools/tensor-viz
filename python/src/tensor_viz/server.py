"""Local server entrypoint for the browser demo."""

from __future__ import annotations

import contextlib
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .bundle import Tab, create_session_bundle


def _static_root() -> Path:
    packaged = Path(__file__).resolve().parent / "static"
    if packaged.exists() and any(packaged.iterdir()):
        return packaged
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "packages" / "viewer-demo" / "dist"


@dataclass
class ViewerSession:
    """Handle for a running browser session."""

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
    tensor: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray] | Tab | Sequence[Tab],
    *,
    name: str | None = None,
    session_bundle: bytes | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    keep_alive: bool = True,
) -> ViewerSession:
    """Launch the standalone viewer for one or more NumPy tensors."""

    session_bundle = session_bundle or create_session_bundle(tensor, name=name)
    static_root = _static_root()
    if not static_root.exists():
        raise FileNotFoundError(
            "Viewer demo assets are missing. Build the frontend with `npm run build` first."
        )

    class Handler(SimpleHTTPRequestHandler):
        """Serve the demo app and a one-shot `.viz` session bundle."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_root), **kwargs)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/session.viz":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(session_bundle)))
                self.end_headers()
                self.wfile.write(session_bundle)
                return

            if self.path.startswith("/api/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            target = static_root / self.path.lstrip("/")
            if self.path == "/" or not target.exists():
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

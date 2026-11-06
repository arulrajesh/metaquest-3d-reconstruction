"""Local web server for VR locations (``mq3drecon serve-locations``).

Serves the packaged viewer at ``/`` and ``<workspace>/locations`` at ``/locations/``.
With wireless adb, ``adb reverse tcp:PORT tcp:PORT`` lets the Quest browser open
``http://localhost:PORT``. WebXR needs a secure context; localhost is one, a LAN IP is not.
"""

from __future__ import annotations

import functools
import logging
import mimetypes
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/json", ".json")


class _Handler(SimpleHTTPRequestHandler):
    locations_dir: Path = Path(".")

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/locations/"):
            root, relative = self.locations_dir, clean[len("/locations/"):]
        else:
            root, relative = STATIC_DIR, clean.lstrip("/") or "index.html"
        target = (root / relative).resolve()
        if root.resolve() not in (target, *target.parents):  # no path traversal outside the two roots
            return str(STATIC_DIR / "__forbidden__")
        return str(target)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # quiet by default
        logger.debug("%s - %s", self.address_string(), format % args)


def make_server(locations_dir: Path, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    Path(locations_dir).mkdir(parents=True, exist_ok=True)
    handler = type("LocationsHandler", (_Handler,), {"locations_dir": Path(locations_dir)})
    return ThreadingHTTPServer((host, port), functools.partial(handler, directory=str(STATIC_DIR)))


def adb_reverse_and_open(port: int, *, adb_path: Path | None = None, open_browser: bool = False) -> list[str]:
    """Run ``adb reverse`` for every headset; optionally open the headset browser. Returns configured serials."""
    from mq3drecon.streaming.adb import find_adb, list_devices

    adb = find_adb(adb_path)
    if adb is None:
        return []
    configured = []
    for serial in list_devices(adb):
        result = subprocess.run([str(adb), "-s", serial, "reverse", f"tcp:{port}", f"tcp:{port}"], capture_output=True, text=True)
        if result.returncode != 0:
            continue
        configured.append(serial)
        if open_browser:
            subprocess.run(
                [str(adb), "-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", f"http://localhost:{port}/"],
                capture_output=True, text=True,
            )
    return configured


def serve_forever_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, name="mq3drecon-locations-http", daemon=True)
    thread.start()
    return thread

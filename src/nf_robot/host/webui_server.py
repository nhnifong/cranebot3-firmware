import atexit
import functools
import logging
import threading
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

from nf_robot.host.video_streamer import ThreadingHTTPServer

logger = logging.getLogger(__name__)

# Two places a built UI can live, checked in this order:
#  1. playroom-ui/dist/ — raw `npm run build` output, present only in a source
#     checkout. Preferred when present since it's whatever was built most
#     recently by a developer working locally.
#  2. src/nf_robot/ui/assets/ — populated by scripts/build_release.sh before
#     packaging, and what actually ships inside the published wheel/sdist (see
#     MANIFEST.in / pyproject.toml's package-data). This is the only candidate
#     that exists in a pip-installed copy of nf_robot (no playroom-ui/ dir).
# Neither is committed to git — both are build artifacts.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_CHECKOUT_DIST = _REPO_ROOT / "playroom-ui" / "dist"
_PACKAGED_ASSETS_DIR = Path(__file__).resolve().parent.parent / "ui" / "assets"


def _default_assets_dir():
    if (_SOURCE_CHECKOUT_DIST / "index.html").exists():
        return _SOURCE_CHECKOUT_DIST
    return _PACKAGED_ASSETS_DIR


class WebUiServer:
    """
    Serves the built playroom-ui static bundle so a browser elsewhere on the
    LAN can load the full cockpit UI straight from this machine, with no
    dependency on neufangled.com. Reuses the same ThreadingHTTPServer pattern
    the MJPEG camera streams use (see MjpegStreamer in video_streamer.py).
    """

    def __init__(self, port=8090, bind_address="127.0.0.1", assets_dir=None):
        self.port = port
        self.bind_address = bind_address
        self.assets_dir = Path(assets_dir) if assets_dir is not None else _default_assets_dir()
        self.http_server = None
        atexit.register(self.stop)

    def start(self):
        if not (self.assets_dir / "index.html").exists():
            raise RuntimeError(
                f"UI self-hosting is on (pass --no_serve_ui to disable it) but no built UI was "
                f"found (looked at {_SOURCE_CHECKOUT_DIST} and {_PACKAGED_ASSETS_DIR}). Run "
                "`npm run build` in playroom-ui/ first."
            )
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(self.assets_dir))
        self.http_server = ThreadingHTTPServer((self.bind_address, self.port), handler)
        thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        thread.start()

    def stop(self):
        h = self.http_server
        if h is not None:
            h.shutdown()
            h.server_close()
            self.http_server = None

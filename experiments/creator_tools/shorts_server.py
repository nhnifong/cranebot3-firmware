#!/usr/bin/env python

"""Serves the YouTube shorts pages in this directory along with the robot models they load.

    python experiments/creator_tools/shorts_server.py
    python experiments/creator_tools/shorts_server.py --port 8001

then open http://127.0.0.1:8000/cdpr_short.html

Pages are served from this directory and the playroom models under /models/, so a page loads
them as models/<name>.glb from the same origin.
"""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parents[1] / 'playroom-ui' / 'public' / 'assets' / 'playroom' / 'models'


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.glb': 'model/gltf-binary'}

    def translate_path(self, path):
        clean = path.split('?', 1)[0].split('#', 1)[0]
        if clean.startswith('/models/'):
            name = Path(clean[len('/models/'):]).name
            return str(MODELS / name)
        return super().translate_path(path)

    def end_headers(self):
        # Pages are edited while the server runs; always fetch the current file.
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), functools.partial(Handler, directory=str(HERE)))
    print(f'serving http://{args.host}:{args.port}/cdpr_short.html')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

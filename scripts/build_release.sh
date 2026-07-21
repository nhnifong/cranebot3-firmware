#!/usr/bin/env bash
# Builds the standalone playroom-ui bundle, copies it into
# src/nf_robot/ui/assets/ (what --serve_ui serves and what package-data ships
# in the wheel — see MANIFEST.in and pyproject.toml's [tool.setuptools.package-data]),
# then builds the python sdist/wheel. Run this instead of `python3 -m build`
# directly when cutting a release, so the UI bundle is never stale or missing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="$ROOT_DIR/playroom-ui"
ASSETS_DIR="$ROOT_DIR/src/nf_robot/ui/assets"

echo "==> Building playroom-ui"
(cd "$UI_DIR" && npm ci && npm run build)

echo "==> Copying build output into ${ASSETS_DIR#$ROOT_DIR/}"
rm -rf "$ASSETS_DIR"
mkdir -p "$ASSETS_DIR"
cp -r "$UI_DIR/dist/." "$ASSETS_DIR/"

echo "==> Building python package"
(cd "$ROOT_DIR" && python3 -m build)

echo "==> Done. Artifacts in dist/"

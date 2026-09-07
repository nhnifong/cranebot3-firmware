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

VERSION="$(grep -m1 -E '^version[[:space:]]*=' "$ROOT_DIR/pyproject.toml" | sed -E 's/.*"([^"]+)".*/\1/')"
echo "==> Done. Artifacts in dist/ (version $VERSION)"

# dist/ keeps every past build, so only ever upload the version just built.
if [ ! -t 0 ]; then
    echo "Not a terminal; skipping PyPI upload."
    exit 0
fi

read -r -p "Upload nf_robot ${VERSION} to PyPI? [y/N] " REPLY
if [ "$REPLY" = "y" ] || [ "$REPLY" = "Y" ]; then
    echo "==> Uploading to PyPI"
    (cd "$ROOT_DIR" && python3 -m twine upload "dist/nf_robot-${VERSION}"*)
    echo "==> Uploaded nf_robot ${VERSION}"
else
    echo "Skipping upload. To do it later, run:"
    echo "    python3 -m twine upload dist/nf_robot-${VERSION}*"
fi

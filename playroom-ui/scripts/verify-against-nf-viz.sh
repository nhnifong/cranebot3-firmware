#!/usr/bin/env bash
# Packs this package and installs the tarball into the sibling nf-main-site/nf-viz
# checkout to verify it builds there, since this package ships source and consumers
# typecheck it themselves (see README.md's Publishing section). Run before bumping
# the version and publishing.
#
# Leaves the caller's working directory untouched, and removes the tarball when done
# (success or failure).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NF_VIZ_DIR="$PACKAGE_DIR/../../nf-main-site/nf-viz"

if [ ! -d "$NF_VIZ_DIR" ]; then
  echo "error: sibling repo not found at $NF_VIZ_DIR" >&2
  echo "check out nf-main-site next to cranebot3-firmware to use this script" >&2
  exit 1
fi

TARBALL="$(cd "$PACKAGE_DIR" && npm pack 2>/dev/null)"
cleanup() {
  rm -f "$PACKAGE_DIR/$TARBALL"
}
trap cleanup EXIT

(cd "$NF_VIZ_DIR" && npm install "$PACKAGE_DIR/$TARBALL" --legacy-peer-deps && npm run build)

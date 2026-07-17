#!/usr/bin/env bash
# Self-contained smoke test for verify-against-nf-viz.sh: builds a throwaway fake
# "cranebot3-firmware/playroom-ui" + "nf-main-site/nf-viz" sibling pair under a temp
# directory (a few hundred bytes, no real assets) and runs the real script against
# it, so this can be run on any machine without a real nf-main-site checkout and
# without ever touching one if it happens to exist.
#
# Usage: scripts/verify-against-nf-viz.selftest.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_SCRIPT="$SCRIPT_DIR/verify-against-nf-viz.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

# --- fixture: a minimal fake playroom-ui package, sibling to a fake nf-viz ---
FAKE_PKG="$WORK/cranebot3-firmware/playroom-ui"
FAKE_NF_VIZ="$WORK/nf-main-site/nf-viz"

mkdir -p "$FAKE_PKG/scripts" "$FAKE_PKG/src"
cp "$REAL_SCRIPT" "$FAKE_PKG/scripts/verify-against-nf-viz.sh"
chmod +x "$FAKE_PKG/scripts/verify-against-nf-viz.sh"
cat > "$FAKE_PKG/package.json" <<'EOF'
{ "name": "fake-playroom-ui", "version": "0.0.0", "files": ["src"] }
EOF
echo "export const marker = true;" > "$FAKE_PKG/src/index.js"

mkdir -p "$FAKE_NF_VIZ"
cat > "$FAKE_NF_VIZ/package.json" <<'EOF'
{ "name": "fake-nf-viz", "version": "0.0.0", "scripts": { "build": "echo build-ran" } }
EOF

# --- case 1: happy path ---
CWD_BEFORE="$(pwd)"
OUTPUT="$("$FAKE_PKG/scripts/verify-against-nf-viz.sh" 2>&1)" && STATUS=0 || STATUS=$?
CWD_AFTER="$(pwd)"

[ "$STATUS" -eq 0 ] || fail "happy path exited $STATUS:
$OUTPUT"
[ "$CWD_BEFORE" = "$CWD_AFTER" ] || fail "working directory changed: $CWD_BEFORE -> $CWD_AFTER"
[ -z "$(find "$FAKE_PKG" -maxdepth 1 -name '*.tgz')" ] || fail "tarball left behind in $FAKE_PKG"
[ -d "$FAKE_NF_VIZ/node_modules/fake-playroom-ui" ] || fail "package was not installed into fake nf-viz"
echo "$OUTPUT" | grep -q "build-ran" || fail "fake nf-viz build did not run"
echo "ok: happy path (installs, builds, cleans up tarball, cwd unchanged)"

# --- case 2: sibling repo missing ---
rm -rf "$WORK/nf-main-site"
OUTPUT="$("$FAKE_PKG/scripts/verify-against-nf-viz.sh" 2>&1)" && STATUS=0 || STATUS=$?
CWD_AFTER2="$(pwd)"

[ "$STATUS" -ne 0 ] || fail "expected a nonzero exit when the sibling repo is missing"
[ "$CWD_BEFORE" = "$CWD_AFTER2" ] || fail "working directory changed on the error path: $CWD_BEFORE -> $CWD_AFTER2"
echo "$OUTPUT" | grep -qi "not found" || fail "expected a 'not found' error message, got:
$OUTPUT"
echo "ok: missing-sibling-repo error path"

echo "PASS"

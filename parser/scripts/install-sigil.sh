#!/usr/bin/env bash
#
# Install the sigil toolchain to a stable location.
#
# LARES-488: PATH used to point at parser/target/release inside a working
# checkout, so `sigil` was whatever that tree last compiled, on whatever branch
# it was on. That checkout sat on a feature branch 210 commits behind develop and
# predated the call-target resolution in d20cf36, so `sigil check` returned rc=0
# on code the current checker rejects — silently, across every repo.
#
# The fix is an installed artifact with recorded provenance, refreshed
# deliberately, rather than a build directory that drifts with whatever anyone
# last did in it.
#
# Usage:
#   ./parser/scripts/install-sigil.sh                # install to ~/.local/bin
#   PREFIX=/opt/sigil ./parser/scripts/install-sigil.sh
#   ALLOW_DIRTY=1 ./parser/scripts/install-sigil.sh  # install a dev build anyway

set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local}"
BINDIR="$PREFIX/bin"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARSER="$REPO_ROOT/parser"

cd "$REPO_ROOT"

branch="$(git rev-parse --abbrev-ref HEAD)"
commit="$(git rev-parse --short HEAD)"
dirty_count="$(git status --porcelain | wc -l | tr -d ' ')"

echo "installing sigil"
echo "  from:   $REPO_ROOT"
echo "  commit: $commit ($branch)"
echo "  to:     $BINDIR"

# Provenance is the entire point. A build from a dirty tree or a stale branch is
# legitimate during development but must not be installed by accident, because
# its output is indistinguishable from a release build's once it is on PATH.
if [ "$dirty_count" != "0" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
  echo
  echo "refusing: working tree has $dirty_count modified file(s)."
  echo "The installed binary would report (DIRTY) and its behaviour would not"
  echo "correspond to any commit. Re-run with ALLOW_DIRTY=1 if that is intended."
  exit 1
fi

git fetch --quiet origin develop 2>/dev/null || true
if git rev-parse --verify --quiet origin/develop >/dev/null; then
  behind="$(git rev-list --count HEAD..origin/develop 2>/dev/null || echo 0)"
  if [ "$behind" != "0" ]; then
    echo
    echo "warning: this checkout is $behind commit(s) behind origin/develop."
    echo "         Installing it anyway; that is what LARES-488 was about."
  fi
fi

# Default features deliberately: jit, llvm, protocols, native. A --features
# minimal build hides two llvm_codegen failures, so an installed binary must not
# be built that way — the point is that `sigil check` means the same thing
# everywhere it runs.
echo
echo "building (release, default features)..."
cargo build --release --manifest-path "$PARSER/Cargo.toml"

echo "building native runtime archives..."
( cd "$PARSER/runtime" && bash build_native.sh >/dev/null && make libsigil_runtime.a >/dev/null )

# `sigil compile` locates its runtime relative to the executable when no
# cwd-relative copy is found (main.rs find_native_runtime / find_runtime), so the
# archives must be installed alongside the binary, not left in the build tree.
echo "installing..."
install -Dm755 "$PARSER/target/release/sigil"        "$BINDIR/sigil"
install -Dm644 "$PARSER/runtime/libsigil_native.a"   "$BINDIR/runtime/libsigil_native.a"
install -Dm644 "$PARSER/runtime/libsigil_runtime.a"  "$BINDIR/runtime/libsigil_runtime.a"
install -Dm644 "$PARSER/runtime/sigil_runtime.c"     "$BINDIR/runtime/sigil_runtime.c"

echo
"$BINDIR/sigil" --version

case ":$PATH:" in
  *":$BINDIR:"*) ;;
  *) echo; echo "note: $BINDIR is not on PATH. Add it, and make sure no build"
     echo "      directory precedes it." ;;
esac

resolved="$(command -v sigil 2>/dev/null || true)"
if [ -n "$resolved" ] && [ "$resolved" != "$BINDIR/sigil" ]; then
  echo
  echo "warning: PATH still resolves sigil to:"
  echo "           $resolved"
  echo "         not the binary just installed. Fix the PATH ordering, or the"
  echo "         install has changed nothing for anyone's shell."
fi

#!/usr/bin/env bash
# Build the self-hosted Sigil compiler (jormungandr).
#
# bootstrap_fixed4.c is the generated C for the compiler. It compiles cleanly but
# has never linked on its own: it declares and calls 46 stdlib symbols whose
# definitions jormungandr only emits when it RUNS, so its own bootstrap never
# contained them. bootstrap_completion.c supplies exactly those.
#
# See ../../docs/findings/JORMUNGANDR-BUILD-STATE-2026-09-07.md for the analysis.
#
# NOTE: CLAUDE.md documents `gcc -o sigil2 sigil2.c -lm`. There is no sigil2.c in
# this repository; use this script instead.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-jormungandr}"
echo "==> assembling"
cat bootstrap_fixed4.c bootstrap_completion.c > .jormungandr_full.c
echo "==> compiling (this takes a minute)"
gcc -g -O0 -w -o "$OUT" .jormungandr_full.c -lm
rm -f .jormungandr_full.c
echo "==> built ./$OUT"
"./$OUT" || true

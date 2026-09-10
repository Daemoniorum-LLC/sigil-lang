#!/usr/bin/env bash
# Differential sweep: run each case under both backends and diff.
set -uo pipefail
cd "$(dirname "$0")"

SIGIL="${SIGIL:-../../parser/target/release/sigil}"
RUNTIME="${RUNTIME:-../../../qliphoth/runtime/sigil_runtime.js}"

if [[ ! -x "$SIGIL" ]]; then
    echo "sigil compiler not found at $SIGIL (set SIGIL=)" >&2
    exit 2
fi
if [[ ! -f "$RUNTIME" ]]; then
    echo "runtime not found at $RUNTIME (set RUNTIME=)" >&2
    exit 2
fi

only="${1:-}"
total=0; agree=0; failed=0
for case_file in cases/*.sigil; do
    name="$(basename "$case_file" .sigil)"
    [[ -n "$only" && "$name" != "$only" ]] && continue
    out="$(SIGIL="$SIGIL" RUNTIME="$RUNTIME" node ./diff.mjs "$case_file" 2>&1)"
    status=$?
    echo "$out"
    line="$(echo "$out" | tail -1)"
    n_total="$(echo "$line" | sed -n 's/.*total=\([0-9]*\).*/\1/p')"
    n_agree="$(echo "$line" | sed -n 's/.*agree=\([0-9]*\).*/\1/p')"
    total=$(( total + ${n_total:-0} ))
    agree=$(( agree + ${n_agree:-0} ))
    [[ $status -ne 0 ]] && failed=1
done

echo
echo "───────────────────────────────────────"
echo "probes: $agree / $total agree"
[[ $agree -eq $total && $failed -eq 0 ]] || exit 1

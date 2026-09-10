#!/usr/bin/env bash
#
# Diff two sweeps and fail on anything new.
#
# This is what makes "0 false positives" a check rather than a memory: run
# sweep.sh on develop, run it again on the branch, and compare. Every new row
# has to be argued for. A change that only *removes* rows is a fixed false
# positive; a change that adds them is either the true positive it set out to
# find, or the regression it did not.
#
# Usage:
#   compare.sh <before-dir> <after-dir>
#
# Exit codes:
#   0  no new findings
#   1  new findings (listed on stdout)
#   2  usage or input error

set -euo pipefail

die() { echo "compare: $*" >&2; exit 2; }

[ $# -eq 2 ] || die "usage: compare.sh <before-dir> <after-dir>"
before="$1/findings.tsv"
after="$2/findings.tsv"
[ -f "$before" ] || die "no findings.tsv in $1"
[ -f "$after" ]  || die "no findings.tsv in $2"

# The two sweeps have to have looked at the same files, or the diff means
# nothing. Compare what each recorded scanning.
bd=$(awk '/^path-digest:/ {print $2}' "$1/manifest.txt" 2>/dev/null || true)
ad=$(awk '/^path-digest:/ {print $2}' "$2/manifest.txt" 2>/dev/null || true)
if [ -n "$bd" ] && [ -n "$ad" ] && [ "$bd" != "$ad" ]; then
    echo "compare: WARNING -- the two sweeps scanned different file sets" >&2
    echo "  before: $(awk '/^files:/ {print $2}' "$1/manifest.txt") files, digest $bd" >&2
    echo "  after:  $(awk '/^files:/ {print $2}' "$2/manifest.txt") files, digest $ad" >&2
    echo "  the counts below are not comparable" >&2
fi

# Two sweeps by the same binary compare a build against itself, which is not
# what anyone runs this for.
bb=$(awk '/^binary-sha256:/ {print $2}' "$1/manifest.txt" 2>/dev/null || true)
ab=$(awk '/^binary-sha256:/ {print $2}' "$2/manifest.txt" 2>/dev/null || true)
if [ -n "$bb" ] && [ "$bb" = "$ab" ]; then
    echo "compare: note -- both sweeps used the same binary ($bb)" >&2
fi

# A dirty build cannot be cited. The commit in the manifest does not describe
# what ran, so a result measured from one is not reproducible by anyone else.
for side in 1 2; do
    eval "dir=\$$side"
    tree=$(awk '/^build-tree:/ {print $2}' "$dir/manifest.txt" 2>/dev/null || true)
    if [ "$tree" = "DIRTY" ]; then
        echo "compare: WARNING -- $dir was measured with a binary built from a DIRTY tree;" >&2
        echo "  its build-commit does not describe what ran, so this side is not citable" >&2
    fi
done

added=$(LC_ALL=C comm -13 "$before" "$after" || true)
removed=$(LC_ALL=C comm -23 "$before" "$after" || true)

n_added=$([ -z "$added" ] && echo 0 || printf '%s\n' "$added" | wc -l)
n_removed=$([ -z "$removed" ] && echo 0 || printf '%s\n' "$removed" | wc -l)

echo "before: $(wc -l < "$before") findings"
echo "after:  $(wc -l < "$after") findings"
echo "added:   $n_added"
echo "removed: $n_removed"

if [ "$n_removed" -gt 0 ]; then
    echo
    echo "removed (no longer reported):"
    printf '%s\n' "$removed" | sed 's/^/  - /'
fi

if [ "$n_added" -gt 0 ]; then
    echo
    echo "ADDED (each of these needs an argument):"
    printf '%s\n' "$added" | sed 's/^/  + /'
    exit 1
fi

echo
echo "no new findings"

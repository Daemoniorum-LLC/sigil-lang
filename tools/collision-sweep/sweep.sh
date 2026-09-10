#!/usr/bin/env bash
#
# Sweep a tree of Rust sources for reserved-name collisions and write a stable,
# diffable report.
#
# The reserved-name diagnostic (#68) is sold on a corpus property -- "0 false
# positives across N files". #90: nothing on the branch reproduced that number,
# so every change to the scanner re-established it by hand, and two hand-built
# sweeps are comparable neither to each other nor to the original. This is the
# harness that makes it a measurement.
#
# Usage:
#   sweep.sh <corpus-root> [-o <out-dir>] [--sigil <binary>]
#
# Writes into <out-dir> (default ./sweep-out):
#   manifest.txt  what was actually scanned -- root, file count, digest
#   findings.tsv  one row per (file, name): relative-path <TAB> name <TAB> uses
#   summary.txt   the headline counts
#   raw.err       the migrate stderr the report was parsed from
#
# NEVER runs a mode that writes. `--dry-run` (#92/#94) writes nothing anywhere,
# which is what makes it safe to point at ~/.cargo/registry -- an earlier
# in-place sweep there would have corrupted the cargo cache.

set -euo pipefail

die() { echo "sweep: $*" >&2; exit 2; }

root=""
out="sweep-out"
sigil=""

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--out)   out="${2:-}"; shift 2 || die "-o needs a directory" ;;
        --sigil)    sigil="${2:-}"; shift 2 || die "--sigil needs a path" ;;
        -h|--help)  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)         die "unknown option: $1" ;;
        *)          [ -z "$root" ] || die "one corpus root at a time"; root="$1"; shift ;;
    esac
done

[ -n "$root" ] || die "usage: sweep.sh <corpus-root> [-o <out-dir>] [--sigil <binary>]"
[ -d "$root" ] || die "not a directory: $root"

if [ -z "$sigil" ]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    sigil="$here/../../parser/target/release/sigil"
fi
[ -x "$sigil" ] || die "no sigil binary at $sigil (build with: cd parser && cargo build --release)"

mkdir -p "$out"
out="$(cd "$out" && pwd)"
root_abs="$(cd "$root" && pwd)"

# What was scanned, so the number in a claim can be checked rather than
# remembered. "the crate registry" is not reproducible across machines or
# months; a file count and a digest of the sorted path list is.
echo "sweep: taking the manifest..." >&2
find "$root_abs" -name '*.rs' -type f | LC_ALL=C sort > "$out/files.txt"
file_count=$(wc -l < "$out/files.txt")
[ "$file_count" -gt 0 ] || die "no .rs files under $root_abs"
digest=$(sed "s|^$root_abs/||" "$out/files.txt" | LC_ALL=C sha256sum | cut -d' ' -f1)

# The build is recorded the same way the corpus is: by digest. `sigil` has no
# --version, and a path plus a timestamp does not say which code ran -- which
# is the whole question a before/after comparison is asking.
sigil_digest=$(sha256sum "$sigil" | cut -d' ' -f1)

# ...and by provenance, which is the half a digest cannot supply. A shared
# build output is an input nobody declares: a sibling session measured Sigil
# with a binary built from a dirty feature branch 49 commits ahead and 173
# behind `develop`, and the number it reported read as a `develop` number.
# A hash says two runs used different code; a commit and a dirty flag say
# *which* code, and whether it was anything a reader can check out.
build_dir=$(dirname "$sigil")
sigil_commit="unknown"
sigil_branch="unknown"
sigil_tree="unknown"
if repo_root=$(git -C "$build_dir" rev-parse --show-toplevel 2>/dev/null); then
    sigil_commit=$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || echo unknown)
    sigil_branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
    if [ -n "$(git -C "$repo_root" status --porcelain 2>/dev/null)" ]; then
        sigil_tree="DIRTY"
    else
        sigil_tree="clean"
    fi
fi

{
    echo "root:          $root_abs"
    echo "files:         $file_count"
    echo "path-digest:   $digest"
    echo "binary:        $sigil"
    echo "binary-sha256: $sigil_digest"
    echo "build-commit:  $sigil_commit"
    echo "build-branch:  $sigil_branch"
    echo "build-tree:    $sigil_tree"
    echo "swept:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out/manifest.txt"

if [ "$sigil_tree" = "DIRTY" ]; then
    echo "sweep: WARNING -- the binary was built from a DIRTY tree ($sigil_branch @ $sigil_commit)." >&2
    echo "  The commit above does not describe what ran. Commit or stash before measuring." >&2
fi

echo "sweep: $file_count files under $root_abs" >&2

# --dry-run writes nothing. stdout is the diff preview and is discarded; the
# collision warnings go to stderr, one block per file, and name their file.
set +e
"$sigil" migrate "$root_abs" --dry-run > /dev/null 2> "$out/raw.err"
rc=$?
set -e
[ "$rc" -le 1 ] || die "migrate exited $rc; see $out/raw.err"

# Parse the warning blocks into rows. A block is
#     warning: <path>: N names are refused ...:
#         <name>          <count> uses
# and the report is sorted so two sweeps diff cleanly.
awk -v root="$root_abs/" '
    /^warning: .*: [0-9]+ name/ {
        line = $0
        sub(/^warning: /, "", line)
        # strip the trailing ": N name(s) ... :" -- the path may contain ": "
        # nowhere on disk, but match the last occurrence to be safe
        idx = match(line, /: [0-9]+ name(s)? (is|are) refused/)
        path = substr(line, 1, idx - 1)
        sub("^" root, "", path)
        next
    }
    /^    [^ ]/ && path != "" {
        name = $1
        uses = $2
        print path "\t" name "\t" uses
    }
    /^  Rename them/ { path = "" }
' "$out/raw.err" | LC_ALL=C sort > "$out/findings.tsv"

files_with=$(cut -f1 "$out/findings.tsv" | LC_ALL=C sort -u | wc -l)
names=$(cut -f2 "$out/findings.tsv" | LC_ALL=C sort -u | wc -l)
rows=$(wc -l < "$out/findings.tsv")
uses=$(awk -F'\t' '{s += $3} END {print s + 0}' "$out/findings.tsv")

{
    cat "$out/manifest.txt"
    echo
    echo "files with findings: $files_with"
    echo "findings (rows):     $rows"
    echo "distinct names:      $names"
    echo "total uses:          $uses"
    echo
    echo "names by files affected:"
    cut -f2 "$out/findings.tsv" | LC_ALL=C sort | uniq -c | sort -rn | head -40
} > "$out/summary.txt"

cat "$out/summary.txt"
echo >&2
echo "sweep: wrote $out/{manifest.txt,findings.tsv,summary.txt,raw.err}" >&2

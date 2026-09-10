# Reserved-name collision sweep

The reserved-name diagnostic (#68) rests on a corpus property: **no false
positives**. That property was a memory, not a measurement — no harness on the
branch reproduced the original 6,091-file number, so every change to the
scanner re-established it by hand, and two hand-built sweeps (21,897 registry
files one way, a 2,500-file sample the other) were comparable neither to each
other nor to the original (#90).

There are two things here, and they answer different questions.

## The per-PR check — `corpus/`

Four files, one per shape the scanner has been wrong about, with their findings
pinned **exactly** in `collision_corpus_tests` (`parser/src/main.rs`). Runs in
milliseconds as part of `cargo test`.

| fixture | the shape | why |
|---|---|---|
| `positions.rs` | one name in two positions | `tome` is a valid field and an invalid binding; `this` is the reverse |
| `closure_params.rs` | closure parameters | #84/#89 — develop carried `\|this\|` as a live false positive |
| `prose.rs` | colons in comments | "Implementation based on:" is not a field |
| `bounds.rs` | `where Self: Sized` | reads exactly like a field declaration |

A scanner change that alters any expectation has to change it in the same
commit, where a reviewer sees it. That is the point: the property becomes a
check rather than a memory.

## The wide measurement — `sweep.sh`

```console
$ ./sweep.sh ~/.cargo/registry/src -o /tmp/sweep-before
sweep: 31464 files under /home/…/.cargo/registry/src
…
files with findings: 76
findings (rows):     80
distinct names:      12
```

Writes into the output directory:

| file | |
|---|---|
| `manifest.txt` | root, file count, digest of the sorted relative path list, and the build: binary sha256, commit, branch, clean/dirty |
| `findings.tsv` | `relative/path` ⇥ `name` ⇥ `uses`, sorted |
| `summary.txt` | the headline counts, and names ranked by files affected |
| `raw.err` | the `migrate` stderr the report was parsed from |

**The corpus and the build are reported, not assumed.** A shared build output
is an input nobody declares — a sibling session measured Sigil with a binary
built from a dirty feature branch 49 commits ahead of `develop` and 173 behind
it, and the number it produced read as a `develop` number. So the manifest
records the build's commit, branch and clean/dirty state alongside the hash,
and both `sweep.sh` and `compare.sh` say so loudly when a tree was dirty: a
hash tells you two runs used different code, but only a commit plus a clean
flag tells you *which* code, and whether anyone else can check it out. "The crate registry"
is not reproducible across machines or months; a file count plus a digest of
the relative path list is. `sigil` has no `--version`, so the binary is
recorded by sha256 — a path and a timestamp do not say which code ran, which is
the whole question a before/after comparison is asking. `compare.sh` says so
loudly when two sweeps scanned different file sets, and notes when they used
the same binary.

## Comparing two builds

```console
$ git checkout develop && (cd parser && cargo build --release)
$ ./sweep.sh ~/.cargo/registry/src -o /tmp/sweep-before
$ git checkout my-branch && (cd parser && cargo build --release)
$ ./sweep.sh ~/.cargo/registry/src -o /tmp/sweep-after
$ ./compare.sh /tmp/sweep-before /tmp/sweep-after
```

Exit 0 means no new findings. Exit 1 lists them, and each one has to be argued
for: a new row is either the true positive the change set out to find, or the
regression it did not. Rows that *disappear* are fixed false positives.

## The trap

`sigil migrate` can rewrite files in place. `sweep.sh` only ever runs
`--dry-run`, which since #92/#94 writes nothing anywhere — that is what makes
it safe to point at `~/.cargo/registry`. An in-place sweep there corrupts the
cargo cache. Do not add a mode to this script that writes.

# Handoff — the Lares capability probe (branch `claude/lares-spec-sigil-rebuild-2cqmp3`)

**Full handoff and the gap register live in the Lares repository:**
`docs/HANDOFF-SIGIL-REBUILD.md` and `docs/specs/LARES-SIGIL-REBUILD-SPEC.md`
(spec 1.30.0, gaps S1–S124). This file is the pointer, and the two things a
session landing *here* needs before it touches anything.

## What the branch is

Lares — a real React application — ported to Sigil, to find out what Sigil and
Qliphoth cannot yet do. **The gaps are the deliverable, not the app.** Most of
this branch is compiler work that porting forced: the statement transform for
migrated React, and the ~61 backend defects behind it.

The change set is paired with `Daemoniorum-LLC/qliphoth#10`. A compiler change
without its host half fails at `WebAssembly.instantiate`, so they land together.

## Before you touch PR #62

**It conflicts with `develop` — 13 files — and you should wait for #74.**

The branch is based on `main`. `develop` is 8 commits ahead, and the conflicting
files (`interpreter.rs`, `parser.rs`, `stdlib.rs`, `llvm_codegen.rs`) are the
*same ones* PR #74 (main → develop) is reconciling, with resolutions it argues
for in detail. Resolving them here now is throwaway work against a `develop`
that #74 replaces, and risks contradicting it.

**Sequence: #74 lands → merge `develop` into this branch → resolve the rest.**

One consequence to know about: **CI does not run on #62 while it is
conflicted.** GitHub cannot compute a merge ref for a conflicted PR, so
`pull_request` workflows do not fire. The last green run was `14ffff3`. Absent
checks are neither a pass nor a fail — validate locally with the commands
below.

The base was retargeted from `main` to `develop` per #69 and #74's own note.

## Before you change anything about `None`

**S124: the two backends do not agree about nothing.** `==` is strictly typed in
the interpreter and untyped in WASM — `∅ == None` is a runtime type error under
one and `true` under the other, and `∅` and `None` are not even the same type in
the interpreter.

The Option change on this branch (`wasm::NONE`, a sentinel) lands in the **WASM
backend only**. The interpreter, Cranelift and LLVM still box `None`.
Propagating it is a language decision. `tools/differential/cases/option.sigil`
probes what both sides agree on and lists, in its header, the seven things the
interpreter refuses to evaluate at all.

## Building and checking

```bash
cd parser
cargo build --release --no-default-features \
  --features jit,native,protocols,react-migrate,wasm

RUST_MIN_STACK=67108864 cargo test --release \
  --no-default-features --features jit,native,protocols,react-migrate,wasm --lib
# expect 885 pass

cd ../jormungandr/tests && ./run_tests_rust.sh        # expect 794/798
cd ../../tools/differential && ./run.sh               # expect 105/107
node imports.mjs                                      # expect 220 checked
cd ../../website-qliphoth && SIGIL_COMPILER=../parser/target/release/sigil ./build.sh --clean
```

CI builds `jit,llvm,wasm,lsp,protocols`; a feature-gated break will not show
under the line above.

The sweep's two failures are **deliberate** and documented in
`tools/differential/README.md`. A third is new.

`SIGIL_MIGRATE_SKIPS=1` prints why each helper body was skipped during a
migration; `=2` prints the offending line. Use it before theorising about a
count — twice on this branch a number named the wrong cause.

# main / develop Reconciliation — Handoff

`main` and `develop` diverged. This is what happened, what is open, and what has to happen in
what order. Written 2026-09-09 from outside the project, by an agent doing static analysis of
the Sigil ecosystem that ran into the divergence and could not proceed around it.

If the PR numbers below have moved, trust the repository over this file.

---

## 1. What the divergence is

Both branches moved 8 commits from `aeadd52` and neither is correct. Measured by counting
files that pass `sigil check`, with clean binaries built from each branch:

| Repository | merge base | `main` | `develop` |
|---|---:|---:|---:|
| `nihil` | 104/104 | **83/104** | 104/104 |
| `morgoth` | 50/259 | **257/259** | **42/259** |
| `lucifer` | 18/26 | 21/26 | 18/26 |
| `athame` | 2/12 | 3/12 | 3/12 |

Read that table carefully — it is the whole problem:

- **`main` regressed `nihil`**, 104 → 83, and it was #61 that did it (bisected).
- **`main` fixed `morgoth`**, 50 → 257. That fix exists *only* on `main`, and it is +207 files.
- **`develop` dipped `morgoth` further**, 50 → 42, across its own 8 commits. Never diagnosed.

Neither branch dominates. Each holds work the other lacks. That is why this is a merge and not
a fast-forward.

They also disagree about the **language**, not only the implementation:

- `develop`'s `Evidentiality` enum lacks `Chaos` (`⁂`); `main` has it.
- `develop`'s `ActorDef` has `handlers: Vec<MessageHandler>`; `main` has `methods: Vec<Function>`.

Anything consuming `sigil-parser` as a library has to cope with both, or pin deliberately.

## 2. The nihil regression, precisely

#61 tightened `parse_type_path`:

```rust
// a0180d3 (#59) and earlier — the first clause consumes `·` unconditionally,
// which makes the second clause dead code
let is_path_sep = self.consume_if(&Token::MiddleDot)
    || (first_segment_is_type && self.consume_if(&Token::MiddleDot));

// 8f92d5c (#61) — now gated on the first segment being uppercase
let is_path_sep = first_segment_is_type && self.consume_if(&Token::MiddleDot);
```

The tightening was intentional and its own comment says why: `·` after a lowercase segment
"must be left for postfix expression parsing", so `tag·to_string` parses as a method call on a
variable. **That is right in expression position and wrong in a type**, which contains no
variables — so every module path, all of which are lower case, was rejected:
`&vary std·fmt·Formatter<'_>`, `std·fmt·Result`.

All 21 of nihil's `main`-only failures are that shape.

The fix in #74 disables the heuristic in type positions only, read-once so a type *containing*
an expression (a const generic, an array length) is expression position again. It reaches
102/104, not 104 — the last two are middledot module paths in **expression** position
(`std·collections·HashMap·new()`), which the pre-#61 code accepted only because its guard was
unconditional. Recovering those needs a decision about the grammar, not a wider guard.

## 3. Open PRs and the order they must land in

| PR | What | State |
|---|---|---|
| **#69** | CI check: PRs to `main` must come from `develop`/`release/*`/`hotfix/*` | ready, green |
| **#74** | `main` → `develop`, 97 conflict hunks resolved | **merged** — `develop` is `d20a33c` |
| #62 | Lares/Qliphoth statement transform, 96 commits, 875 files | unblocked by #74 |
| ~~#66~~ | the parse_type_path fix standalone | closed — #74 carries it |

```
   #74  main→develop  ──┬──►  #77 record ancestry  ──►  release-merge develop→main
       ✔ merged         │         (open)                (fixes nihil on main)
                        │
                        ├──►  #62 merges develop into itself, then resolves
                        │
                        └──►  downstream consumers repin   ✔ user-swarm done
```

**#74 was the bottleneck and it has merged.** PR #62 retargeted itself from `main` to `develop`
citing #69, and its author reached the same conclusion independently: resolving its 13
conflicting files earlier "would be throwaway work against a `develop` that #74 replaces". It
can now resolve against the merged `develop`.

**The release-merge back to `main` is still the half nobody has done.** Until it happens, `main`
stays at nihil 83/104 and the divergence has been moved rather than closed.

**#74 landed as a squash, and that costs one more step.** The content arrived; the ancestry did
not. `git merge-base --is-ancestor main develop` still answers no, so git's base for a
`develop → main` merge is `aeadd52`, sixteen commits back, and the merge re-conflicts in the
same five parser files. PR **#77** fixes that with an ancestry-only `-s ours` merge whose tree
hash is byte-identical to `d20a33c`; after it lands the release merge is a fast-forward. **#77
must be merged with a merge commit — squashing it discards the second parent, which is the
whole change.**

## 4. How #74's 97 hunks were resolved

Full reasoning is in the PR body and the merge commit. The summary:

| File | Hunks | Took | Why |
|---|---:|---|---|
| `llvm_codegen.rs` | 44 | main | 15 are main's `G##` fixes vs develop's pre-fix baseline; see below for the other 2 |
| `interpreter.rs` | 20 | main | every hunk a strict superset — extra fields, guards, `crate·` re-export registration |
| `main.rs` | 13 | main's file, then develop's #63/#64/#65 replayed as diffs | hunk-by-hunk kept duplicating the attribute-conversion block; git had aligned two *different* functions |
| `parser.rs` | 3 | mixed | `ConstDef.ty` is `Option<TypeExpr>` in the merged `ast.rs`; develop's extern-fn block is a superset; main's turbofish guard is correct |
| `lexer.rs` | 1 | develop | `For`/`In`/`Break`/`Continue` have **no lexer rule producing them on either branch** |
| codegen fixtures | 11 | main, whole pairs | a `.sg` from one branch with a `.expected` from the other cannot pass |
| `CONCLAVE.sigil` | 1 | both | append-only session registry |

### The one call that wants a second opinion

`llvm_codegen.rs` hunks 3 and 9 are develop's Vec base-pointer caching, which assumes data
begins at offset 2. Main's G25 comment says that assumption is wrong:

> the Rust Vec memory layout (ptr, len, cap) doesn't match the inline data assumption. Call the
> runtime function.

I took correctness over the optimisation — develop's caching would miscompile against the real
layout. **If that caching is meant to survive, it needs rewriting against the runtime
accessors, not restoring as-is.** Someone who owns that code should confirm.

## 5. Verification, and how to repeat it

#74 was built with **default features** (`jit + llvm + protocols + native`), so
`llvm_codegen.rs` — where 44 of the 97 hunks live — is actually compiled.

**LLVM 18 is probably already installed and only the dev headers are missing.** `llvm-sys`
then fails in a way that reads as "no LLVM at all". It is not:

```bash
apt-get install -y llvm-18-dev libpolly-18-dev libzstd-dev libffi-dev libtinfo-dev zlib1g-dev
cargo build --release          # full default features
```

Result, `main` and the merge on the same configuration:

```
main    781 passed, 14 failed, 2 skipped
merged  781 passed, 14 failed, 2 skipped   ← identical failing sets
```

The 14 are environmental: Kafka, AMQP, WebSocket and GraphQL need brokers; the socket, pty and
stdin tests need a terminal.

**In a container the suite hangs** on `P1_065_pty` and `P1_070_read_string` unless each test is
bounded. Point `SIGIL_COMPILER` at a wrapper:

```bash
#!/bin/bash
exec timeout -s KILL 20 /path/to/parser/target/release/sigil "$@" < /dev/null
```

The `minimal` feature (`--no-default-features --features minimal`) builds without LLVM and runs
the interpreter tests, but **do not validate a codegen change with it** — that is the mistake
that made most of this work weaker than it looked.

## 6. Left open

1. **The release-merge `develop` → `main`.** Nothing else closes the divergence. Take #77
   first (with a merge commit, not a squash) and it is a fast-forward.
2. **develop's morgoth dip** (50 → 42) is undiagnosed. #74 masks it by taking main's side, so it
   will not show up in the merged parse rates — but the defect may still be in develop's
   history and would resurface if those hunks are ever revisited.
3. **Middledot module paths in expression position.** Two nihil files, and some of `athame` and
   `sigil-lang`'s own shortfall. Needs a grammar decision.
4. **The `llvm_codegen.rs` caching question** in §4.

## 7. Ecosystem parse rates, for context

Measured with the merged compiler. These are *not* compiler defects unless noted — mostly they
are ecosystem code drifting from the language the compiler accepts:

| Repository | Rate | Note |
|---|---:|---|
| `nihil` | 98.1% | was 79.8% on `main` |
| `morgoth` | 99.2% | |
| `lucifer` | 80.8% | |
| `sigil-lang` (own ecosystem dirs) | 78.2% | mixed dialect: 476 `invoke`, 409 `use` |
| `athame` | 33.3% | long tail, no single cause |
| `infernum-sigil` | **0%** | 1,016 `use`, zero `invoke` — written in Rust syntax |

`infernum-sigil` is the outlier and it is not a compiler bug: the repository is Rust-shaped and
`sigil migrate` is the fix. Measured on a copy, migrate takes it from **0/187 to 93/187**.
Landing #67 (`core`/`alloc` → `std`; 120 sites there) and #68 (reserved-word collisions; 118
sites) first would push that further, so it is worth migrating once rather than twice.

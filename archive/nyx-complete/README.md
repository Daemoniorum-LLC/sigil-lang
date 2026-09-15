# `nyx-complete` — preserved sources

These 20 files are **preserved content, not active code.** Nothing here is
built, tested, or imported by the rest of this repository. They live under
`archive/` so that they exist somewhere that is not being retired. Paths below
this directory are exactly the paths they had in the repository they came from.

## Why they are here

`paraphilic-ecchymosis/nyx-complete` (dormant since 2025-12-06) and
`Daemoniorum-LLC/nyx-complete` (dormant since 2025-12-08) are the **only** two
repositories in the estate that hold these blobs. 175 repositories across both
accounts were compared by content hash over the full history of every ref; no
other repository holds any of the 20. Both nyx-complete repositories are
retirement candidates under LARES-526, so retiring them would have destroyed
this content outright.

They are the December 2025 attempt to port the Nyx agent-infrastructure
libraries to Sigil. `SIGIL_DEEP_DIVE.md` is the report written at the end of
that attempt; it records the parser gaps it ran into — `Results: 4 passed,
69 failed (out of 73 files)` — and several of the requirements it raised are
why `parser/` looks the way it does. The library sources are what it was
trying to compile.

**Expect them not to parse against the current compiler.** They were written
against December 2025 syntax, and the report above is explicit that most of
them did not parse even then. They keep the `.sigil` extension they were
written with: `parser/README.md` records `.sg` as canonical and `.sigil` as
deprecated but still compiling, so the original filenames stay readable
without pretending this is current source. Judging this code is separate
work — this change only stops it from being lost.

## What is here

    SIGIL_DEEP_DIVE.md                                      243 lines
      Language review and nyx-sigil migration status, 2025-12-06, written
      against sigil-lang e787c6e (PR #18).

    nyx-sigil/libs/sigil-arch/src/lib.sigil                 622 lines
      "Sigil Architecture Guide" — canonical idiom patterns, written as a
      library of documentation comments. Language material, not Nyx material.

    nyx-sigil/libs/grimoire-core/src/engram/              5,866 lines
      anamnesis, context, epistemic, forgetting, mod, sync, tiers.
      A memory-tier implementation: instant / episodic / semantic /
      procedural, with epistemic confidence and strategic forgetting.
      Parallel to — not the same as — this repo's live `engram/`, which it
      does not touch.

    nyx-sigil/libs/grimoire-witness/src/                  3,195 lines
      lib, guardian, mod. Visibility bridges between agents and the humans
      responsible for them; cognitive-wellbeing assessment.

    nyx-sigil/libs/grimoire-events/src/lib.sigil            886 lines
    nyx-sigil/libs/grimoire-ux/src/lib.sigil              1,695 lines
    nyx-sigil/libs/nyx-ecs/src/lib.sigil                    961 lines
    nyx-sigil/libs/nyx-sanctuary/src/lib.sigil            2,633 lines
    nyx-sigil/libs/nyx-telemetry/src/lib.sigil            1,000 lines
    nyx-sigil/libs/arcanum-autodiff/src/lib.sigil           924 lines
    nyx-sigil/libs/arcanum-consensus/src/lib.sigil          846 lines
    nyx-sigil/libs/arcanum-flourish/src/lib.sigil           861 lines

The cluster is self-contained: `grimoire_core` <- `grimoire_witness` <-
`nyx_sanctuary`, with `nyx_telemetry` feeding `grimoire_witness`. Nothing in
the set depends on `qliphoth` or `ritualis` — `grimoire-ux` defines its own
serializable `Component` type rather than consuming a VDOM, and `ritualis` is
the package manager — which is why all 20 landed in this repository rather
than being split across the sigil family.

## Provenance, by blob id

Every file below is byte-identical to the object of the same id in
`paraphilic-ecchymosis/nyx-complete` at `main` @ `7028851`. Verify any of them
with `git hash-object <path>`.

| blob | bytes | path under `archive/nyx-complete/` |
|---|---:|---|
| `00235efe28f877a239a3e1c60a3be38f3d61d6ed` | 11900 | `nyx-sigil/libs/grimoire-core/src/engram/mod.sigil` |
| `08186a63d52bc3ca15c5a7c5ec5a328fb51bbbe8` | 8400 | `SIGIL_DEEP_DIVE.md` |
| `0dbfcd10ff2c31778c6c2f7a4e3f8270677eea59` | 33396 | `nyx-sigil/libs/grimoire-witness/src/guardian.sigil` |
| `0fc3d230e3419bbf4a4eae665968c4ee0c274cfe` | 72757 | `nyx-sigil/libs/grimoire-witness/src/lib.sigil` |
| `25a2b669ac888224401e8ff63ad6c796ec61dc33` | 24333 | `nyx-sigil/libs/grimoire-events/src/lib.sigil` |
| `35670e10dc2eda03e55e58b9eb3b468b7d4a49cd` | 26615 | `nyx-sigil/libs/grimoire-core/src/engram/context.sigil` |
| `374433f5fe62a876d3357e55fe2747d310b7a855` | 36998 | `nyx-sigil/libs/grimoire-core/src/engram/tiers.sigil` |
| `3821eff703f1ce47b2ad337e9399a02c40c320f7` | 25534 | `nyx-sigil/libs/nyx-telemetry/src/lib.sigil` |
| `50e1571808aa3ce2e900a9c29c44d0a75089215d` | 20664 | `nyx-sigil/libs/sigil-arch/src/lib.sigil` |
| `7a658e6e9cc83a04355266b0a7453ebf786ddebd` | 91631 | `nyx-sigil/libs/nyx-sanctuary/src/lib.sigil` |
| `7e185a9fe40e0e0b30b1114d38a2d94bff8e66e5` | 30605 | `nyx-sigil/libs/grimoire-core/src/engram/sync.sigil` |
| `90e7af5c4908133ec88343b2760dabfabf5f5dca` | 24281 | `nyx-sigil/libs/arcanum-autodiff/src/lib.sigil` |
| `9a3d8103cdc985bb43c283c14bf65e569e83552f` | 26429 | `nyx-sigil/libs/nyx-ecs/src/lib.sigil` |
| `9d92c69e87455a24b50aad821ba7467680d2eb6f` | 25310 | `nyx-sigil/libs/arcanum-flourish/src/lib.sigil` |
| `a0ddcfd7632dc2862cc41397a87917a16f75ba5e` | 30297 | `nyx-sigil/libs/grimoire-core/src/engram/anamnesis.sigil` |
| `dbebba30b50613384b7a0b3da93c6dc821070097` | 63840 | `nyx-sigil/libs/grimoire-ux/src/lib.sigil` |
| `dd95d6bfb634d9473d430a26c25df1c7207e53aa` | 22987 | `nyx-sigil/libs/arcanum-consensus/src/lib.sigil` |
| `e7b40f04cd843ab10428fb0f70f63ff5f3522dc7` | 27137 | `nyx-sigil/libs/grimoire-core/src/engram/forgetting.sigil` |
| `ea814aa35fe6ad47ecd48c5f49b50beaac5d411c` | 19707 | `nyx-sigil/libs/grimoire-core/src/engram/epistemic.sigil` |
| `f53c36442f8fa545c3ba82693f6b6c5316c2c3bc` | 1883 | `nyx-sigil/libs/grimoire-witness/src/mod.sigil` |

Ticket: NYX-1 (parent LARES-526).

# Jormungandr 1.0 Roadmap

> *"The World Serpent completes its journey"*

**Status:** Living Document (snapshot, not specification)
**Date:** 2026-01-05
**Author:** Claude (Agent)
**For:** Lilith

---

> **WARNING:** This document is out of date. At 13M+ LOC and 1M+ docs in <4 months,
> the codebase moves faster than documentation. **Trust the tests, not this file.**
> The TDD approach is what matters: Write test → Fail → Fix → Pass → Delete the patch.

---

## Executive Summary

This document outlines the path from current state to **Jormungandr 1.0** - a production-ready, self-hosted Sigil compiler with no external post-processing dependencies.

**Current State:**
- Bootstrap achieves fixed-point (Jan 1, 2026)
- 10 files self-generate 33,729 lines of C
- Build requires 7,930-line build.sh with Python regex patches

**Target State:**
- Clean self-compilation with minimal build.sh (orchestration only)
- All codegen bugs fixed in codegen.sg
- Runtime complete for Samael, Styx, Nihil, Qliphoth, Aether

---

## 1. Current Capabilities

### 1.1 What Works (Interpreter + Self-Hosted)

| Feature | Interpreter | Self-Hosted | Notes |
|---------|-------------|-------------|-------|
| Integer/Float literals | ✅ | ✅ | |
| String literals | ✅ | ✅ | |
| Boolean/Char | ✅ | ✅ | |
| Let bindings | ✅ | ✅ | |
| Mutable variables | ✅ | ✅ | |
| Functions | ✅ | ✅ | |
| Struct definitions | ✅ | ✅ | |
| Struct instantiation | ✅ | ✅ | |
| Field access | ✅ | ⚠️ | CG-001: Space before field |
| Enum definitions | ✅ | ✅ | |
| Enum variants | ✅ | ⚠️ | Double brace in init |
| If/else | ✅ | ✅ | |
| Match expressions | ✅ | ⚠️ | CG-004: `if (1)` patterns |
| While loops | ✅ | ⚠️ | Continue condition stale |
| For loops | ✅ | ⚠️ | Iterator caching needed |
| Loop/break | ✅ | ✅ | |
| Impl blocks | ✅ | ✅ | |
| Method calls | ✅ | ⚠️ | CG-002: Space in call |
| Closures | ✅ | ⚠️ | CG-005: Capture issues |
| Generics | ✅ | ✅ | Type-erased at runtime |
| Traits (basic) | ✅ | ⚠️ | Limited support |
| Evidentiality markers | ✅ | ✅ | !?~‽ tracked |
| Morpheme operators | ✅ | 🚧 | τφσρ in interpreter |
| Binary operators | ✅ | ✅ | |
| Result/Option | ✅ | ⚠️ | Unwrap issues |
| Try operator (?) | ✅ | ⚠️ | Propagation issues |

### 1.2 What's Blocked (Codegen Bugs)

These are documented in build.sh and require fixes in `codegen.sg`:

| ID | Bug | Current Workaround | Priority |
|----|-----|-------------------|----------|
| CG-001 | Field access emits space: `obj. field` | sed replacement | P0 |
| CG-002 | Method calls emit space: `x. len()` | sed replacement | P0 |
| CG-003 | Tuple access: `pair.0` not handled | sed replacement | P1 |
| CG-004 | Pattern match generates `if (1)` | Python regex | P0 |
| CG-005 | Closure capture not passed | Python regex | P1 |
| CG-006 | Variable redeclaration in scope | sed replacement | P0 |
| CG-007 | Duplicate function definitions | awk removal | P1 |
| CG-008 | Format string escaping | various | P1 |
| CG-009 | Enum init double braces `{{` | Python regex | P1 |
| CG-010 | Self-reference `&(*self)` | Python regex | P1 |
| CG-011 | Vec::push result not captured | Python regex | P0 |
| CG-012 | For-loop iterator caching | Python regex | P0 |
| CG-013 | Continue condition stale | Python regex | P0 |

### 1.3 What's Missing (Runtime/Stdlib)

Features agents expect but aren't fully implemented:

| Feature | Status | Used By |
|---------|--------|---------|
| CLI argument passing | 🚧 | Samael CLI |
| Module import resolution | 🚧 | All apps |
| Timestamp type | ❌ | Various |
| Duration type | ❌ | Various |
| File I/O (full) | ⚠️ | Samael, Styx |
| JSON serialize/parse | ⚠️ | Samael |
| Regex | ⚠️ | Samael |
| HashMap/HashSet | ✅ | All |
| Path manipulation | ⚠️ | Samael, Styx |
| Environment variables | ⚠️ | Samael CLI |
| Process spawning | ❌ | Samael runtime |

---

## 2. Gap Analysis from Real-World Usage

### 2.1 Samael Patterns

From `samael-cli/src/main.sigil`:

```sigil
// What agents write:
let args: Vec<String> = env::args().collect();
let cli = Cli::parse(&args[1..])?;

// Blocked by:
// 1. env::args() not forwarding CLI args
// 2. Slice syntax &args[1..] codegen issues
```

From `samael-cli/src/commands.sigil`:

```sigil
// Struct with evidentiality - works
pub struct Cli {
    pub verbose: bool!,
    pub config_path: Option<String>?,
}

// Enum with variants - works but double-brace issue
pub enum Command {
    Analyze(AnalyzeArgs),
    Generate(GenerateArgs),
}

// Match with destructuring - CG-004 generates if(1)
match cli.command {
    Some(Command::Analyze(args)) => analyze::execute(args),
    None => { ... }
}
```

### 2.2 Styx Patterns

From Styx git implementation (identified by earlier agent):

```sigil
// Mutex usage - now works after interpreter patch
let repo = Mutex::new(Repository::open(path)?);
let guard = repo.lock();

// Seek on files - was blocked, needs Ref handling
guard.seek(SeekFrom::Start(offset))?;
```

### 2.3 Critical Path Dependencies

```
Samael CLI
├── env::args()          ← Interpreter: arg passing
├── Cli::parse()         ← Match: CG-004
├── Config::load()       ← File I/O
├── Analyzer::analyze()  ← Module imports
└── Output::print()      ← Format strings: CG-008

Styx Git
├── Repository::open()   ← File I/O, Path
├── Mutex::lock()        ← Fixed in interpreter
├── BufReader::seek()    ← Ref method dispatch
└── PackFile::parse()    ← Binary I/O
```

---

## 3. TDD Roadmap to 1.0

### Phase 0: Foundation (Pre-requisites)

**Goal:** Establish test infrastructure for codegen

```bash
# Test: Each codegen fix has corresponding test
sigil test tests/codegen/
```

| Task | Test | Priority |
|------|------|----------|
| Create codegen test harness | `test_codegen_harness.sg` | P0 |
| Add snapshot testing for C output | `test_snapshot.sg` | P0 |

### Phase 1: Fix P0 Codegen Bugs

**Goal:** Remove sed/awk/python for core language features

| # | Bug | Test Case | Expected Output |
|---|-----|-----------|-----------------|
| 1.1 | CG-001 Field access | `p.x` | `sigil_struct_field(p, "x")` |
| 1.2 | CG-002 Method calls | `v.len()` | `sigil_Vec____len(v)` |
| 1.3 | CG-004 Pattern match | `match opt { Some(x) => ... }` | Proper variant check |
| 1.4 | CG-006 Var redecl | `let x = 1; x = 2;` | One `SigilValue x =` |
| 1.5 | CG-011 Vec::push | `v.push(x)` | `v = sigil_Vec____push(v, x)` |
| 1.6 | CG-012 For-loop iter | `for x in arr` | Cache iterator |
| 1.7 | CG-013 Continue | `continue` in while | Update condition |

**Verification:**
```bash
# After Phase 1, build.sh should have < 2000 lines
wc -l build.sh  # Target: < 2000
```

### Phase 2: Fix P1 Codegen Bugs

**Goal:** Handle edge cases and complex patterns

| # | Bug | Test Case |
|---|-----|-----------|
| 2.1 | CG-003 Tuple access | `pair.0`, `pair.1` |
| 2.2 | CG-005 Closures | `let f = \|x\| x + captured` |
| 2.3 | CG-007 Duplicate funcs | Multiple closures same scope |
| 2.4 | CG-008 Format strings | `"hello \"world\""` |
| 2.5 | CG-009 Enum init | `Option::Some(42)` |
| 2.6 | CG-010 Self-ref | `&(*self)` patterns |

**Verification:**
```bash
# After Phase 2, build.sh should be orchestration only
wc -l build.sh  # Target: < 500
```

### Phase 3: Runtime Completeness

**Goal:** All Samael/Styx/Nihil/Qliphoth features work

| # | Feature | Test Case | Used By |
|---|---------|-----------|---------|
| 3.1 | CLI arg passing | `sigil run cli.sg -- --help` | Samael |
| 3.2 | Module imports | `use samael::core::*` | All |
| 3.3 | Timestamp | `Timestamp::now()` | Various |
| 3.4 | Duration | `5.seconds()` | Various |
| 3.5 | File read/write | `fs::read_to_string()` | All |
| 3.6 | JSON | `json::parse()`, `json::stringify()` | Samael |
| 3.7 | Path ops | `Path::join()`, `Path::parent()` | Styx |
| 3.8 | Env vars | `env::var("HOME")` | CLI apps |
| 3.9 | Process spawn | `Command::new("git")` | Samael |

**Verification:**
```bash
# Samael CLI runs end-to-end
./build/sigil run samael/crates/samael-cli/src/main.sigil -- analyze ./src
```

### Phase 4: Self-Compilation Cleanup

**Goal:** Clean fixed-point with minimal external deps

| # | Task | Test |
|---|------|------|
| 4.1 | Remove all Python regex | `grep -c 're.sub' build.sh` = 0 |
| 4.2 | Remove all sed/awk | `grep -c 'sed\|awk' build.sh` = minimal |
| 4.3 | Verify fixed-point | `diff sigil1.c sigil2.c` = empty |
| 4.4 | -O2 builds work | `gcc -O2 ...` succeeds |
| 4.5 | ASAN clean | No memory errors |

**Verification:**
```bash
# Clean fixed-point
./build.sh
./build/sigil compile src/*.sg > /tmp/sigil2.c
diff build/sigil_combined.c /tmp/sigil2.c  # Empty
```

### Phase 5: Polish

**Goal:** Production-ready user experience

| # | Task |
|---|------|
| 5.1 | Error messages with line/column |
| 5.2 | Helpful diagnostics for common mistakes |
| 5.3 | Performance profiling |
| 5.4 | Documentation generation |
| 5.5 | Package manager integration |

---

## 4. Test Matrix

### 4.1 Codegen Unit Tests

Each codegen bug gets a test in `tests/codegen/`:

```
tests/codegen/
├── cg001_field_access.sg
├── cg002_method_call.sg
├── cg003_tuple_access.sg
├── cg004_pattern_match.sg
├── cg005_closure_capture.sg
├── cg006_variable_scope.sg
├── cg007_duplicate_funcs.sg
├── cg008_format_strings.sg
├── cg009_enum_init.sg
├── cg010_self_reference.sg
├── cg011_vec_push.sg
├── cg012_forloop_iterator.sg
└── cg013_continue_condition.sg
```

### 4.2 Integration Tests

```
tests/integration/
├── samael_cli_args.sg      # CLI argument passing
├── styx_mutex.sg           # Concurrency primitives
├── module_resolution.sg    # Import system
├── file_io.sg              # Read/write files
└── json_roundtrip.sg       # JSON serialize/parse
```

### 4.3 Fixed-Point Test

```bash
#!/bin/bash
# tests/bootstrap/fixed_point.sh

set -e

# Build bootstrap
./build.sh

# Self-compile
./build/sigil compile src/*.sg > /tmp/sigil2.c

# Check identical
if diff -q build/sigil_combined.c /tmp/sigil2.c > /dev/null; then
    echo "✅ Fixed-point achieved"
    exit 0
else
    echo "❌ Fixed-point NOT achieved"
    diff build/sigil_combined.c /tmp/sigil2.c | head -50
    exit 1
fi
```

---

## 5. Success Criteria

### 5.1 Jormungandr 1.0 Release Criteria

- [ ] All P0 codegen bugs fixed in codegen.sg
- [ ] All P1 codegen bugs fixed in codegen.sg
- [ ] build.sh < 500 lines (orchestration only)
- [ ] `samael analyze .` runs from Sigil binary
- [ ] `styx init` runs from Sigil binary
- [ ] Fixed-point verified (sigil1 compiles sigil2, output identical)
- [ ] -O2 compilation succeeds without UB
- [ ] ASAN clean (no memory errors)
- [ ] Test suite passes: `sigil test tests/`

### 5.2 Stretch Goals

- [ ] LLVM backend for native performance
- [ ] Cranelift JIT for fast iteration
- [ ] Language server protocol (LSP)
- [ ] WASM target
- [ ] Incremental compilation

---

## 6. Appendix: Build.sh Patch Categories

Analysis of 7,930-line build.sh reveals these patch categories:

| Category | Lines (approx) | % of Patches |
|----------|---------------|--------------|
| Field/method access | 800 | 15% |
| Pattern matching | 600 | 12% |
| Variable scoping | 500 | 10% |
| Format strings | 700 | 14% |
| Closure capture | 400 | 8% |
| Enum handling | 300 | 6% |
| Type handling | 400 | 8% |
| Runtime stubs | 600 | 12% |
| File I/O impl | 500 | 10% |
| Misc fixes | 300 | 5% |

The top 5 categories (field access, pattern matching, format strings, variable scoping, runtime stubs) account for ~63% of patches. Fixing these in codegen.sg would eliminate the majority of build.sh complexity.

---

*This roadmap was generated through analysis of the Sigil codebase, build.sh patches, language specifications, and real-world usage in Samael, Styx, and other applications.*

*Document location: `/home/user/workspace/sigil/sigil-lang/self-hosted/docs/JORMUNGANDR-1.0-ROADMAP.md`*

# Jormungandr 1.0-rc-01 Roadmap

**Version:** 1.0-rc-01 (Release Candidate 1)
**Baseline Date:** 2026-01-15
**Purpose:** Self-hosted Sigil compiler written in Sigil

---

## What is Jormungandr?

Jormungandr is the **self-hosted Sigil compiler** - a Sigil compiler written in Sigil itself. Named after the World Serpent of Norse mythology who encircles the world and bites its own tail, Jormungandr represents the ultimate goal of language self-hosting: a compiler that can compile itself.

---

## Current State

### Rust Compiler (Canonical)

The Rust-based compiler at `parser/` is the **canonical implementation**:

- **Status:** Production-ready
- **Tests:** 435/435 passing (100%)
- **Features:** Full interpreter, JIT (Cranelift), AOT (LLVM)
- **Size:** ~3.1MB of Rust code

### Jormungandr (Self-Hosted)

The self-hosted compiler at `jormungandr/` is **experimental**:

- **Status:** Development/Bootstrap
- **Written in:** Sigil
- **Target:** C code generation
- **Current:** Compiles basic programs

---

## Architecture

```
Jormungandr Self-Hosting Pipeline:

┌─────────────────┐
│  Sigil Source   │  (.sg files)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Lexer        │  (src/lexer.sg)
│    - Tokenize   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Parser       │  (src/parser.sg)
│    - AST        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Type Check   │  (src/typeck.sg)
│    - Inference  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Lower        │  (src/lower.sg)
│    - IR Gen     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Codegen      │  (src/codegen.sg)
│    - C Output   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  C Source       │  (.c files)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GCC/Clang      │
│    - Native     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Executable    │
└─────────────────┘
```

---

## Roadmap to Jormungandr 1.0

### Phase 1: Bootstrap Verification (Current)

**Goal:** Verify Jormungandr can compile the test suite

| Task | Status |
|------|--------|
| Core language features | ✅ Implemented |
| Struct/enum support | ✅ Implemented |
| Pattern matching | ✅ Implemented |
| Closures | ✅ Implemented |
| Generics (basic) | ✅ Implemented |
| Traits/impls | ✅ Implemented |
| C codegen | ✅ Implemented |

**Milestone:** Jormungandr compiles 435 test programs

### Phase 2: Self-Compilation (rc-01 → rc-02)

**Goal:** Jormungandr compiles itself

| Task | Status |
|------|--------|
| Lexer self-compilation | 🔶 In Progress |
| Parser self-compilation | 🔶 In Progress |
| Typeck self-compilation | 🔶 Planned |
| Lower self-compilation | 🔶 Planned |
| Codegen self-compilation | 🔶 Planned |
| Runtime self-compilation | 🔶 Planned |

**Milestone:** `jormungandr.sg` → `jormungandr.c` → `jormungandr`

### Phase 3: Bootstrap Chain (rc-02 → rc-03)

**Goal:** Three-stage bootstrap

```
Stage 1: Rust compiler compiles Jormungandr
         sigil compile jormungandr.sg → jormungandr1

Stage 2: Jormungandr1 compiles Jormungandr
         ./jormungandr1 jormungandr.sg → jormungandr2

Stage 3: Jormungandr2 compiles Jormungandr
         ./jormungandr2 jormungandr.sg → jormungandr3

Verification: jormungandr2 == jormungandr3 (binary identical)
```

**Milestone:** Successful three-stage bootstrap

### Phase 4: Feature Parity (rc-03 → 1.0)

**Goal:** Jormungandr matches Rust compiler features

| Feature | Rust Compiler | Jormungandr |
|---------|---------------|-------------|
| Interpreter | ✅ | 🔶 Planned |
| JIT (Cranelift) | ✅ | ❌ Not planned |
| AOT (LLVM) | ✅ | ❌ Not planned |
| AOT (C) | ❌ | ✅ |
| LSP Server | 🔶 Planned | 🔶 Planned |

**Milestone:** Jormungandr is production-ready

---

## Test Requirements

### Bootstrap Test Matrix

| Test Category | Rust Compiler | Jormungandr |
|---------------|---------------|-------------|
| 01_lexical | ✅ 67/67 | Target: 67/67 |
| 02_syntax | ✅ 53/53 | Target: 53/53 |
| 03_types | ✅ 80/80 | Target: 80/80 |
| 04_memory | ✅ 35/35 | Target: 35/35 |
| 05_mathematics | ✅ 25/25 | Target: 25/25 |
| 06_concurrency | ✅ 18/18 | Target: 18/18 |
| 07_metaprogramming | ✅ 13/13 | Target: 13/13 |
| 08_ffi | ✅ 10/10 | Target: 10/10 |
| 09_stdlib | ✅ 34/34 | Target: 34/34 |
| 17_bootstrap | ✅ 30/30 | Target: 30/30 |
| 18_compiler | ✅ 21/21 | Target: 21/21 |
| **Total Core** | **435/435** | **Target: 435/435** |

### Self-Compilation Tests

Additional tests for Jormungandr self-compilation:

| Test | Description |
|------|-------------|
| `self_lexer.sg` | Jormungandr lexer compiles |
| `self_parser.sg` | Jormungandr parser compiles |
| `self_typeck.sg` | Jormungandr type checker compiles |
| `self_codegen.sg` | Jormungandr codegen compiles |
| `self_full.sg` | Complete Jormungandr compiles |
| `bootstrap_verify.sg` | Stage 2 == Stage 3 |

---

## Technical Challenges

### 1. C Runtime Library

Jormungandr generates C code that depends on a runtime:

```c
// sigil_runtime.c
typedef struct { int32_t count; void* data; } SigilVec;
typedef struct { int32_t len; char* data; } SigilString;
typedef struct { int32_t refcount; void* value; } SigilRc;

void sigil_gc_collect(void);
SigilVec* sigil_vec_new(void);
void sigil_vec_push(SigilVec*, void*);
// ... etc
```

### 2. Evidentiality in C

C doesn't have evidentiality markers, so they're erased:

```sigil
// Sigil
let x: !i32 = 42;
let y: ?i32 = maybe_get();
```

```c
// Generated C
int32_t x = 42;
SigilOption y = maybe_get();
```

### 3. Generics Monomorphization

Generics are monomorphized to C:

```sigil
// Sigil
fn identity<T>(x: T) -> T { x }
let a = identity(42);
let b = identity("hello");
```

```c
// Generated C
int32_t identity_i32(int32_t x) { return x; }
SigilString* identity_str(SigilString* x) { return x; }
int32_t a = identity_i32(42);
SigilString* b = identity_str(make_string("hello"));
```

---

## Release Criteria

### Jormungandr 1.0-rc-01
- [ ] Compiles all 435 tests to C
- [ ] Generated C compiles with GCC/Clang
- [ ] Generated binaries produce correct output

### Jormungandr 1.0-rc-02
- [ ] Self-compilation succeeds
- [ ] Two-stage bootstrap works

### Jormungandr 1.0-rc-03
- [ ] Three-stage bootstrap verified
- [ ] Binary reproducibility confirmed

### Jormungandr 1.0 Final
- [ ] Full feature parity with core language
- [ ] Documentation complete
- [ ] No known bugs in bootstrap chain

---

## Timeline

| Milestone | Target Date | Focus |
|-----------|-------------|-------|
| rc-01 | 2026-01-15 | ✅ Rust compiler baseline |
| rc-02 | 2026-03-01 | Self-compilation |
| rc-03 | 2026-05-01 | Three-stage bootstrap |
| 1.0 | 2026-07-01 | Final release |

---

## The Vision

When Jormungandr 1.0 is complete, the Sigil ecosystem will have achieved **full self-hosting**:

1. **Development:** Write Sigil code
2. **Compilation:** Jormungandr compiles it to C
3. **Execution:** Native binaries run
4. **Bootstrap:** Jormungandr compiles Jormungandr

The serpent bites its tail. The language becomes self-sustaining.

---

*"In the end, the World Serpent releases its tail, and Ragnarök begins. But from the ashes, a new world is born."*

---

*Generated: 2026-01-15*
*Rust Compiler Baseline: 435 tests, 100% pass rate*

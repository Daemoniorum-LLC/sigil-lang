# Sigil Specification Audit — January 2026

> *"The map is not the territory, but a well-drawn map reveals where the territory is unmapped."*

## Purpose

This document captures the results of a systematic audit of all Sigil language specifications,
identifying areas that are thin, underspecified, or missing entirely. The goal is to support
Spec-Driven Development by ensuring specifications are complete enough to drive implementation.

---

## Audit Summary

| Category | Status | Notes |
|----------|--------|-------|
| **Well-Specified** | 12 areas | Ready for implementation |
| **Thin/Underspecified** | 6 areas | **ALL RESOLVED** (2026-01-23) |
| **Missing** | 2 areas | 1 resolved, 1 pending |

### Resolution Summary (2026-01-23)

All thin/underspecified areas have been addressed with agent-native specifications:

| Thin Area | Resolution | Philosophy |
|-----------|------------|------------|
| Type Inference | `03A-TYPE-INFERENCE.md` | Dependent + refinement types, SMT-backed |
| Borrow Checking | `04A-CAPABILITY-MEMORY.md` | Fractional permissions, separation logic |
| Pattern Matching | `02A-PATTERN-MATCHING.md` | Predicate patterns, SMT exhaustiveness |
| Morpheme Desugaring | `01A-MORPHEME-DESUGARING.md` | Complete desugaring algebra |
| Module Resolution | `02B-MODULE-RESOLUTION.md` | Full resolution algorithm |
| Error Recovery | `18A-ERROR-RECOVERY.md` | Parser/semantic recovery |

---

## 1. Well-Specified Areas

These specifications are comprehensive enough to drive implementation:

### 1.1 Lexical Structure (01-LEXICAL.md)
- Unicode handling: ✓
- Morpheme operators: ✓
- Token categories: ✓
- Operator precedence: ✓
- Comments and strings: ✓

### 1.2 Core Syntax (02-SYNTAX.md)
- Function definitions: ✓
- Struct/enum/trait declarations: ✓
- Pattern matching syntax: ✓
- Control flow expressions: ✓
- Named parameters: ✓ (added 2026-01)
- Module system syntax: ✓

### 1.3 Type System (03-TYPES.md)
- Primitive types: ✓
- Evidentiality markers: ✓
- Composite types: ✓
- Generics and bounds: ✓
- Variance rules: ✓
- Evidence lattice: ✓

### 1.4 Memory Model (04-MEMORY.md)
- Ownership rules: ✓
- Borrowing semantics: ✓
- Lifetime syntax: ✓
- Interior mutability: ✓
- Smart pointers: ✓
- Drop order: ✓

### 1.5 Concurrency (06-CONCURRENCY.md)
- Async/await: ✓
- Streams: ✓
- Channels: ✓
- Synchronization primitives: ✓
- Actor model: ✓
- Parallel iteration: ✓

### 1.6 Metaprogramming Syntax (07-METAPROGRAMMING.md)
- Rune definition syntax: ✓
- Fragment specifiers: ✓
- Repetition patterns: ✓
- Pipe-invoked runes: ✓ (added 2026-01)
- Procedural rune API: ✓

### 1.7 FFI (08-FFI.md)
- C interop: ✓
- Rust interop: ✓
- Type mapping: ✓
- Safety boundaries: ✓

### 1.8 Standard Library (09-STDLIB.md)
- Core types: ✓
- Collections: ✓
- I/O: ✓
- Networking: ✓
- Serialization: ✓

---

## 2. Thin/Underspecified Areas — ALL RESOLVED

These specifications existed but lacked algorithmic detail. **All have been resolved**
with agent-native specifications that go beyond traditional language design.

### 2.1 Type Inference Algorithm — RESOLVED

**Status:** Specification created as `03A-TYPE-INFERENCE.md`

**Resolution goes beyond original recommendation:**
- Dependent types (Π-types, Σ-types)
- Refinement types with SMT-backed constraint solving
- Liquid type inference for automatic predicate discovery
- Evidence integration throughout inference
- Full formal semantics

### 2.2 Borrow Checking Algorithm — RESOLVED

**Status:** Specification created as `04A-CAPABILITY-MEMORY.md`

**Resolution goes beyond original recommendation:**
- Fractional permissions (not binary shared/exclusive)
- Separation logic integration (P * Q spatial connectives)
- Capability-based security at the type level
- Direct mapping to NyxOS kernel capabilities
- Sealing, revocation, attenuation

### 2.3 Pattern Matching Compilation — RESOLVED

**Status:** Specification created as `02A-PATTERN-MATCHING.md`

**Resolution goes beyond original recommendation:**
- Patterns as predicates, not just structural matching
- SMT-based exhaustiveness checking
- Mathematical predicates in patterns (`{ prime(n) }`)
- Evidence patterns for provenance matching
- Active patterns and view patterns

### 2.4 Morpheme Desugaring — RESOLVED

**Status:** Specification created as `01A-MORPHEME-DESUGARING.md`

**Coverage:**
- Complete desugaring rules for all morphemes (τ, φ, σ, ρ, π, etc.)
- Evidentiality morpheme algebra (!, ?, ~, ‽)
- Aspect and valency morphemes
- Morpheme interaction and fusion rules
- Error semantics

### 2.5 Module Resolution Algorithm — RESOLVED

**Status:** Specification created as `02B-MODULE-RESOLUTION.md`

**Coverage:**
- Full path resolution algorithm
- Tarjan's SCC algorithm for cycle detection
- Visibility as capability-based access control
- Re-export chain resolution
- Formal semantics with judgments

### 2.6 Error Recovery (Parser) — RESOLVED

**Status:** Specification created as `18A-ERROR-RECOVERY.md`

**Coverage:**
- Lexer, parser, and semantic error recovery
- Panic-mode recovery with synchronization tokens
- Error nodes in AST for partial analysis
- Cascading error prevention via tainting
- Agent-specific structured output

---

## 3. Missing Specifications

### 3.1 Rune Expansion Algorithm — RESOLVED

**Status:** Specification created as `07A-RUNE-EXPANSION.md` during this audit.

**Coverage:**
- Pattern matching algorithm: ✓
- Substitution algorithm: ✓
- Hygiene system: ✓
- Pipe-invoked rune desugaring: ✓
- Recursion control: ✓
- Error handling: ✓

### 3.2 Const Evaluation Semantics

**Current state:** Glyphs (const fn) mentioned in 07-METAPROGRAMMING.md but not specified.

**Missing:**
- What operations are allowed in const context
- Const evaluation limits (loops, recursion)
- Const panics
- Const trait methods
- Interaction with generics

**Impact:** Compile-time evaluation behavior undefined.

**Recommendation:** Create `07B-CONST-EVALUATION.md`.

---

## 4. Recommended Priority

Based on implementation needs and safety criticality:

| Priority | Spec | Rationale |
|----------|------|-----------|
| P0 | Borrow Checking Algorithm | Core safety guarantee |
| P0 | Type Inference Algorithm | Required for any program |
| P1 | Pattern Compilation | Correctness-critical |
| P1 | Const Evaluation | Needed for metaprogramming |
| P2 | Module Resolution | Edge cases only |
| P2 | Morpheme Desugaring | Polish item |
| P2 | Error Recovery | Developer experience |

---

## 5. Methodology Notes

This audit was conducted with a "pure design" perspective — examining what the specs say
without reference to what has been implemented. This allows comparison between:

1. **Design intent** — What the language should be
2. **Implementation reality** — What actually got built

Discrepancies between these two views are valuable:
- Design gaps reveal where implementation invented semantics
- Implementation gaps reveal where design outpaced development

---

## 6. Audit Metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-01-23 |
| **Auditor** | Claude (Conclave Agent) |
| **Specs Reviewed** | 21 files |
| **Time Spent** | ~2 hours |
| **Methodology** | SDD Compliance Review |

---

*This audit supports the Spec-Driven Development methodology: specs model reality,
and when reality disagrees with the spec, we update the spec.*

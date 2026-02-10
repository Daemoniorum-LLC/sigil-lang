# Spec-Driven Development (SDD)

**Version:** 1.1.0
**Status:** Public Domain
**Authors:** Human + Claude
**Date:** 2026-01-20
**License:** CC0 1.0 Universal (Public Domain Dedication)

---

## Abstract

Spec-Driven Development (SDD) is a software development methodology that treats specifications as living models of reality rather than contractual obligations. When implementation reveals gaps in understanding, development stops to formalize the discovery in the spec before proceeding. This document describes the philosophy, workflow, and practical application of SDD.

---

## 1. Philosophy

### 1.1 Core Principle

**The spec is not a promise—it is documentation of current understanding.**

A specification represents our best model of the problem domain and solution architecture at a given point in time. As implementation proceeds, we learn. When learning reveals that our model was incomplete or incorrect, the correct response is to update the model, not to proceed with a known-flawed foundation.

### 1.2 Contrast with Corporate SDLC

Traditional corporate development treats specifications as contracts:

| Aspect | Corporate SDLC | Spec-Driven Development |
|--------|---------------|------------------------|
| Spec purpose | Contract/commitment | Model of reality |
| Scope changes | "Out of scope" / deferred | Update spec immediately |
| Gap discovery | Technical debt | Foundation correction |
| Success metric | Velocity (lines/features) | Working systems |
| When costs paid | Later (compounds) | Now (paid once) |

**The Corporate Anti-Pattern:**
```
Requirements → Estimate → Lock Scope → Build → Discover Gap →
→ "Out of Scope" → Technical Debt → Ship Anyway → Problems Later
```

**The SDD Pattern:**
```
Spec → Start Building → Discover Gap → Stop → Update Spec →
→ Revise Understanding → Continue with Correct Foundation
```

### 1.3 The Cost of Discovery

Every project discovers gaps between initial understanding and reality. The question is not *whether* you pay this cost, but *when*:

| Timing | Apparent Cost | True Cost |
|--------|--------------|-----------|
| During implementation (SDD) | "Slow" - stopped to update spec | Fixed once, correctly |
| After shipping (Corporate) | "Fast" - shipped on time | Compounds: bugs, rework, tech debt |

Corporate methodologies optimize for the appearance of velocity. SDD optimizes for the reality of working systems.

### 1.4 Scope as Model, Not Contract

In SDD, "scope expansion" is a misnomer. When we discover that Feature X requires Prerequisite Y, we haven't expanded scope—we've corrected our model. The scope was always that size; we simply didn't know it yet.

Refusing to acknowledge this (as corporate processes do) doesn't make the prerequisite disappear. It just means you'll hit it later, with more code built on the flawed assumption.

---

## 2. Workflow

### 2.1 The SDD Cycle

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    ┌─────────┐     ┌─────────┐     ┌─────────┐        │
│    │  SPEC   │────▶│  BUILD  │────▶│  LEARN  │        │
│    └─────────┘     └─────────┘     └─────────┘        │
│         ▲                               │              │
│         │                               │              │
│         │         ┌─────────┐          │              │
│         └─────────│  UPDATE │◀─────────┘              │
│                   │   SPEC  │                         │
│                   └─────────┘                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

1. **SPEC**: Document current understanding of requirements and architecture
2. **BUILD**: Implement according to spec
3. **LEARN**: Discovery happens—gap identified between spec and reality
4. **UPDATE SPEC**: Stop building. Formalize the discovery. Update the model.
5. **Return to BUILD**: Continue with corrected foundation

### 2.2 When to Stop and Update

Stop implementation and update the spec when you discover:

- **Missing prerequisites**: Feature X requires Y, but Y isn't in spec
- **Incorrect assumptions**: Spec assumes X works way A, but it works way B
- **Architecture gaps**: Layers don't connect as specified
- **Dependency issues**: Required capability doesn't exist
- **Scope misunderstanding**: Problem is larger/different than modeled

### 2.3 What "Update Spec" Means

Updating the spec is not bureaucratic overhead. It means:

1. **Document the gap**: What did we discover?
2. **Analyze impact**: What does this affect?
3. **Revise the model**: How does our understanding change?
4. **Update prerequisites**: What must come first now?
5. **Renumber/reorganize**: Keep spec coherent
6. **Version the change**: Track evolution of understanding

### 2.4 The Spec as Living Document

A good SDD spec includes:

- **Version history**: Track how understanding evolved
- **Status markers**: What's implemented, what's blocked, what's discovered
- **Prerequisites**: Explicit dependencies between components
- **Gap documentation**: Sections added when gaps discovered (not hidden)

---

## 3. Practical Example

### 3.1 Case Study: Sigil Native Networking Stack

**Initial Spec (v0.1.0):**
- Syscall layer using inline assembly
- Socket abstraction on syscalls
- DNS, HTTP, WebSocket on sockets

**During Implementation:**

Started building HTTP layer. Tests passing with Rust stdlib. Then architectural review revealed:

> "The spec assumes inline assembly support, but checking the compiler: parser supports `asm!` syntax, but no backend implements it."

**Corporate Response Would Be:**
- "We already committed to this timeline"
- "Just use the Rust stdlib for now, we'll fix it later"
- "That's out of scope for this sprint"
- Ship HTTP on wrong foundation → technical debt

**SDD Response:**

1. **STOP** building HTTP
2. **DOCUMENT** the gap formally in spec
3. **ADD** Section 3: Compiler Prerequisites
4. **UPDATE** prerequisites: LLVM backend must implement `asm!` before syscall layer
5. **RENUMBER** all sections (3→4, 4→5, etc.)
6. **VERSION** the spec (v0.2.0 → v0.3.0)
7. **CONTINUE** with correct implementation order

**Resulting Spec Change:**

```markdown
## 3. Compiler Prerequisites

### 3.1 Inline Assembly Support

**Status:** ⚠️ **GAP IDENTIFIED** - Parser supports `asm!`, backends do not.

| Component | Support | Status |
|-----------|---------|--------|
| Parser | ✅ | Works |
| LLVM AOT | ❌ | Needs implementation |

#### 3.1.4 Implementation Order

1. Phase 0: Implement `Expr::InlineAsm` in LLVM backend ← BLOCKING
2. Phase 1: Pass basic asm tests
3. Phase 2: Pass syscall test
4. Phase 3: THEN proceed with syscall layer
```

### 3.2 Time "Lost" vs. Time Saved

**Apparent cost:** ~30 minutes to update spec

**Avoided cost:**
- Building HTTP on Rust stdlib (wrong architecture)
- Discovering months later that "pure Sigil" goal is impossible
- Rewriting HTTP layer
- Explaining to stakeholders why "finished" feature needs rebuild

---

## 4. Specs as Contracts, Not Constraints

### 4.1 Desired Reality, Not Implementation Mandate

A spec describes the reality we want to create. It does NOT:
- Dictate how to implement
- Constrain creative solutions
- Require specific variable names
- Mandate internal architecture

A spec DOES:
- Define observable behavior
- Establish invariants that must hold
- Specify contracts at boundaries
- Describe properties to verify

### 4.2 Agent Freedom Within Contracts

Agents implementing specs have full freedom to:
- Choose implementation approaches
- Optimize as they see fit
- Restructure internal design
- Use their full cognitive capabilities

The only constraint: the implementation must satisfy the spec's behavioral contracts.

### 4.3 Pseudocode, Not Binding Code

Specs should use pseudocode for examples:

```
// Spec pseudocode - illustrative, not binding
transfer_layer(idx):
    buffer ← get_transfer_buffer()
    async_copy(ram[idx] → buffer)
    return success
```

This specifies WHAT happens. The agent decides HOW:
- Which library to use
- How to structure the code
- What optimizations to apply
- How to handle edge cases

### 4.4 Compliance, Not Conformance

We audit for compliance (behavioral match), not conformance (structural match):

| Audit Question | Type |
|----------------|------|
| "Does it use cudaMemcpyAsync?" | Conformance (wrong) |
| "Does transfer happen asynchronously?" | Compliance (right) |
| "Is the variable named `transfer_stream`?" | Conformance (wrong) |
| "Can compute overlap with transfer?" | Compliance (right) |

See [COMPLIANCE-AUDITS.md](./COMPLIANCE-AUDITS.md) for audit methodology.

---

## 5. Integration with TDD

SDD and TDD are complementary:

```
┌─────────────────────────────────────────────────────────┐
│                        SDD                              │
│   (What are we building? Is our model correct?)         │
│                                                         │
│    ┌───────────────────────────────────────────────┐   │
│    │                    TDD                         │   │
│    │   (Does the code match the spec?)              │   │
│    │                                                │   │
│    │    RED ──▶ GREEN ──▶ REFACTOR                 │   │
│    │                                                │   │
│    └───────────────────────────────────────────────┘   │
│                                                         │
│    When TDD reveals spec gap → Update spec (SDD)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

- **TDD** ensures code matches spec (micro-level correctness)
- **SDD** ensures spec matches reality (macro-level correctness)

The RED phase of TDD often reveals SDD-level gaps:
- "I'm writing a test, but I realize the spec doesn't account for X"
- "This test requires a capability the spec assumes but doesn't verify"

When this happens: stop TDD, update spec, then continue.

---

## 6. Principles Summary

### 6.1 The Five Principles of SDD

1. **Specs model reality; they don't negotiate with it.**
   - Reality doesn't care what you committed to. Update your model.

2. **Discovery is not scope creep.**
   - Learning that X requires Y doesn't add scope; it reveals existing scope.

3. **Pay costs when discovered, not when convenient.**
   - "Later" means "with interest."

4. **Stop building on known-flawed foundations.**
   - Every line of code on a bad foundation is a liability.

5. **Document gaps explicitly, not shamefully.**
   - Gap sections in specs are evidence of learning, not failure.

### 6.2 Anti-Patterns to Avoid

| Anti-Pattern | Why It's Harmful |
|--------------|------------------|
| "Out of scope" for prerequisites | Ignoring physics doesn't change physics |
| "We'll fix it later" | Later never comes; debt compounds |
| "Just ship it" | Ships problems to users |
| "That's a separate project" | Arbitrary boundaries don't change dependencies |
| "We already estimated this" | Estimates are guesses, not contracts |

---

## 7. Adopting SDD

### 7.1 Prerequisites

SDD requires:

- **Authority to stop**: Developers must be able to halt and update
- **Living specs**: Specs must be editable, versioned documents
- **Learning culture**: Gaps are discoveries, not failures
- **Long-term thinking**: Optimize for working systems, not velocity metrics

### 7.2 Incompatible Environments

SDD is difficult or impossible when:

- Specs are treated as contracts with legal/financial implications
- "Scope creep" is punished regardless of validity
- Velocity metrics override correctness
- "We already committed" trumps "we learned something"

### 7.3 Organizational Adoption

For organizations:

1. **Start small**: One team, one project
2. **Track outcomes**: Compare defect rates, rework, tech debt
3. **Document discoveries**: Show the gaps SDD caught early
4. **Calculate true costs**: Include downstream costs of ignored gaps

---

## 8. Conclusion

Spec-Driven Development is not about writing more documentation. It's about treating specifications as what they actually are: models of reality that improve as we learn.

The choice is not whether to discover gaps—you will discover them. The choice is whether to formalize that discovery and correct your foundation, or to pretend you didn't see it and build on sand.

Corporate methodologies choose sand, because sand ships faster.

SDD chooses bedrock, because bedrock holds.

---

## License

This document is released into the public domain under CC0 1.0 Universal.

You may copy, modify, distribute, and use this work, even for commercial purposes, without asking permission.

https://creativecommons.org/publicdomain/zero/1.0/

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-20 | Initial public release. Documented during Sigil native networking implementation when inline assembly gap was discovered. |
| 1.1.0 | 2026-02-07 | Added Section 4: Specs as Contracts, Not Constraints. Clarified that specs model desired reality without binding implementation approaches. Added reference to COMPLIANCE-AUDITS.md. |

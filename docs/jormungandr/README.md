# Jormungandr Research Initiative

**The World Serpent that bites its own tail**

This directory contains the tracking infrastructure for the Jormungandr research initiative - converting all Daemoniorum projects from their current languages to Sigil.

## Core Insight

Agents ARE the target users of Sigil. Their conversion experience IS user research. Their feedback IS user feedback.

## Directory Structure

```
jormungandr/
├── README.md                    # This file
├── PROJECT_STATUS.md            # Current conversion status
├── checkpoints/                 # ExperienceCheckpoints by project
│   └── {project}/
│       └── {phase}-{timestamp}.md
└── reports/                     # Aggregated research reports
    └── {period}-report.md
```

## Conversion Priority

### Critical (Core Infrastructure)
1. ✅ **sigil-lang self-hosting** - Phase 4 (Fixed Point) IN PROGRESS
2. ⬜ goetia-protocol
3. ⬜ morphe-framework

### High (Major Frameworks)
1. ✅ **aether-engine** - 100% COMPLETE (aether-sigil)
2. 🚧 **infernum-framework** - STARTING
3. ⬜ persona-framework

### Medium (Applications and Tools)
- Various applications pending

## Conversion Phases

Each project conversion follows these phases:
1. **Analysis** - Understand the existing codebase
2. **Design** - Plan the Sigil architecture
3. **Core** - Convert core data structures and types
4. **Logic** - Convert business logic and algorithms
5. **Integration** - Wire up dependencies, I/O, external APIs
6. **Polish** - Error handling, edge cases, optimization
7. **Validation** - Testing, verification, comparison with original

## Current Active Conversions

| Project | Current Phase | Agent | Started |
|---------|---------------|-------|---------|
| sigil-lang (self-hosted) | Phase 4 (Fixed Point) | Multi | 2025-12-06 |
| infernum-framework | Phase 1 (Analysis) | claude-opus-4 | 2025-12-09 |

## Links

- [IDEA_008: Jormungandr Spec](/docs/ideas/sigil-008-jormungandr.md)
- [IDEA_009: Infernum Infrastructure](/daemoniorum/docs/ideas/infernum/infernum-009-jormungandr-infra.md)

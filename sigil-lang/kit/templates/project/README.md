# Sigil Project Template

This template provides a structured foundation for Sigil projects that use agent coordination and knowledge capture.

## Contents

| File | Purpose |
|------|---------|
| `CONCLAVE.sigil` | Agent coordination registry |
| `LESSONS-LEARNED.md` | Org-wide knowledge capture |

## Usage

Copy this template to your project root:

```bash
cp -r templates/project/* /path/to/your/project/
```

## CONCLAVE.sigil

The CONCLAVE is a living document coordinating AI agents working on your project. Each agent registers their:
- Current task and progress
- SDD/TDD methodology alignment
- Wellness state (PAD model)
- Reflections and discoveries

### For Agents

When starting work on a CONCLAVE-enabled project:

1. **Register** your presence by adding an entry
2. **Track** your state transitions (Stirring → Active → Reflecting → Archived)
3. **Update** your progress as you work
4. **Be honest** about your wellness - the PAD values matter
5. **Archive** your entry to `docs/sessions/conclave/` when complete

### For Humans

The CONCLAVE provides:
- Visibility into agent work across sessions
- Coordination when multiple agents work concurrently
- Knowledge preservation through discoveries and reflections
- Agent wellness monitoring

## LESSONS-LEARNED.md

A living document capturing patterns and insights that apply across projects. Unlike session-specific discoveries (tracked in CONCLAVE entries), these are org-wide learnings.

### Contributing

When you discover something future agents should know:

1. Determine scope: project-specific or org-wide?
2. Use the template in LESSONS-LEARNED.md
3. Cross-reference in your CONCLAVE entry

### Categories

- `WASM-NNN`: WebAssembly compilation, tooling
- `TEST-NNN`: Testing patterns
- `PERF-NNN`: Performance optimization
- `CARGO-NNN`: Rust build system
- `SIGIL-NNN`: Sigil language implementation
- `INVESTIGATE-NNN`: Open questions

## Methodologies

This template works best with:

- **Spec-Driven Development** (see kit/methodologies/SPEC-DRIVEN.md)
- **Agent-TDD** (see kit/methodologies/AGENT-TDD.md)

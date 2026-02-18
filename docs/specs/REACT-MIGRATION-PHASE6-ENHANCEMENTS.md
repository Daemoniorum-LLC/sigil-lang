# React→Qliphoth Migration: Phase 6 - Extraction Fidelity

**Version:** 0.1.0
**Status:** Draft
**Authors:** Claude (Conclave session)
**Date:** 2026-02-16
**Depends On:** Phases 1-5 (Complete), REACT-MIGRATION.md, REACT-MIGRATION-TDD-ROADMAP.md

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-16 | Initial draft. Gap discovered during infernum-observer migration test. |

---

## 1. Gap Discovery

### 1.1 How This Gap Was Identified

During a full migration test on `infernum-observer` (214 components, 507 types), we evaluated extraction fidelity by asking:

> "Can an agent complete the migration using **only** the structured extraction, without parsing `source.code`?"

**Finding:** No. The structured extraction is insufficient. Agents must fall back to parsing raw source for:

- Type definitions (extracted as stubs)
- Helper functions (not extracted at all)
- Handler logic (empty `state_changes` arrays)
- Hook callback arguments (truncated to `"{}"`)
- Architecture decisions (no guidance provided)

### 1.2 Evidence

**Type Extraction (Current):**
```json
{
  "name": "ButtonProps",
  "source": "/* interface */",
  "target": "Σ ButtonProps { /* fields */ }"
}
```
→ Agent cannot generate Qliphoth type without parsing source.

**Handler Extraction (Current):**
```json
{
  "name": "handleSend",
  "state_changes": [],
  "side_effects": []
}
```
→ Agent cannot know that `handleSend` calls `addMessage()` and `runAgent()`.

**Custom Hook Arguments (Current):**
```json
{
  "name": "useAgent",
  "arguments": ["{}"]
}
```
→ Agent cannot know about `onComplete` callback that calls `addMessage`.

### 1.3 Impact Assessment

| Component Type | Current Fidelity | With Enhancements |
|----------------|------------------|-------------------|
| Pure presentational | 70% | 95% |
| Stateful (useState) | 50% | 90% |
| Custom hooks | 30% | 85% |
| Complex (Zustand, Query) | 20% | 75% |

---

## 2. Prerequisites

### 2.1 Blocking Prerequisites

Before implementing Phase 6:

| Prerequisite | Status | Notes |
|--------------|--------|-------|
| Phase 5 complete | ✅ | CLI working, 99 tests passing |
| swc TypeScript parsing | ✅ | Already used for JSX |
| Existing extraction architecture | ✅ | `ReactExtractor` in extraction.rs |

### 2.2 Non-Blocking Prerequisites

| Prerequisite | Status | Notes |
|--------------|--------|-------|
| Qliphoth type system spec | ⚠️ Partial | Need formal Σ struct mapping rules |
| Actor communication patterns | ⚠️ Partial | Need service actor conventions |

---

## 3. Behavioral Specifications

This section specifies **observable behavior**, not implementation.

### 3.1 Type Extraction Behavior

**Given** a TypeScript interface:
```typescript
interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}
```

**The extraction MUST produce** a type spec where:
- All fields are enumerated with names
- Each field has a type annotation (as string)
- Optional fields are marked as such
- Union types preserve their variants
- The spec is sufficient to generate: `Σ ChatMessage { role: Role, content: String, timestamp: Option<DateTime> }`

**Property:** For any interface `I`, `generate_qliphoth_type(extract_type(I))` produces valid Qliphoth without parsing original source.

### 3.2 Helper Function Extraction Behavior

**Given** a module-scope function:
```typescript
function transformToSigilEvents(events: AgentEvent[]): SigilEvent[] {
  // implementation
}
```

**The extraction MUST produce** a function spec where:
- Function name is captured
- Parameters with types are captured
- Return type is captured
- Purity can be determined (does it have side effects?)
- Usage sites are referenced

**Property:** For any helper function `f` referenced in a component, `f` appears in extraction with sufficient detail to generate a standalone Qliphoth `rite`.

### 3.3 Handler Body Behavior

**Given** an event handler:
```typescript
const handleSend = (content: string) => {
  addMessage({ role: 'user', content });
  runAgent({ objective: content });
};
```

**The extraction MUST produce** a handler spec where:
- All function calls are enumerated
- The source of each called function is identified (custom hook, prop, import, local)
- State mutations are inferable from the calls

**Property:** For any handler `h`, reading `h.body.statements` reveals all side effects without parsing source.

### 3.4 Hook Argument Behavior

**Given** a hook call with callback:
```typescript
const { isRunning } = useAgent({
  onComplete: (answer) => {
    addMessage({ role: 'assistant', content: answer });
  },
});
```

**The extraction MUST produce** a hook spec where:
- Object arguments have their properties enumerated
- Arrow function arguments have their bodies analyzed (per 3.3)
- Callback triggers are documented

**Property:** For any hook with callbacks, the callback behavior is extractable without parsing source.

### 3.5 Architecture Mapping Behavior

**Given** a component using custom hooks:
```typescript
const { messages, addMessage } = useChat();
const { isRunning, runAgent } = useAgent();
const isModelLoaded = useInfernumStore(selectIsModelLoaded);
```

**The extraction MUST produce** architecture guidance where:
- Each custom hook maps to a recommended Qliphoth pattern
- State ownership is clear (which actor owns `messages`?)
- Communication patterns are suggested (messages, broadcasts)

**Property:** An agent can determine actor boundaries from the extraction without domain knowledge of the original codebase.

---

## 4. Compliance Criteria

### 4.1 What We Audit (Behavior)

| Criterion | Audit Question |
|-----------|----------------|
| Type completeness | Can we generate valid Qliphoth Σ from the type spec? |
| Helper visibility | Are all referenced functions present in extraction? |
| Handler transparency | Can we enumerate all side effects from the spec? |
| Callback capture | Are hook callbacks fully analyzed? |
| Architecture clarity | Can an agent determine actor boundaries? |

### 4.2 What We Don't Audit (Implementation)

| Non-Criterion | Why Not |
|---------------|---------|
| Struct field names | Implementation choice |
| JSON schema | Format is flexible |
| Extraction algorithm | Only output matters |
| Performance | Correctness first |

---

## 5. Phased Implementation

### 5.1 Phase 6.1: Type Field Extraction

**Behavior:** Extract interface/type fields with full type annotations.

**Tests (to be added to TDD roadmap):**
```
test_type_extraction_captures_all_fields
test_type_extraction_marks_optional_fields
test_type_extraction_preserves_union_types
test_type_extraction_handles_extends
test_type_extraction_resolves_type_references
```

**Acceptance:** Given `ButtonProps` interface, extraction includes all 10 fields with types.

### 5.2 Phase 6.2: Helper Function Extraction

**Behavior:** Extract module-scope and component-scope helper functions.

**Tests:**
```
test_helper_extraction_finds_module_scope_functions
test_helper_extraction_finds_component_scope_functions
test_helper_extraction_captures_parameters_and_return_type
test_helper_extraction_detects_purity
test_helper_extraction_tracks_usage_sites
```

**Acceptance:** Given `ChatPanel.tsx`, `transformToSigilEvents` appears in extraction.

### 5.3 Phase 6.3: Handler Body Analysis

**Behavior:** Parse handler bodies to extract function calls and sources.

**Tests:**
```
test_handler_body_extracts_function_calls
test_handler_body_identifies_call_sources
test_handler_body_detects_early_returns
test_handler_body_captures_conditionals
test_handler_body_infers_state_mutations
```

**Acceptance:** Given `handleSend`, extraction shows calls to `addMessage` (from useChat) and `runAgent` (from useAgent).

### 5.4 Phase 6.4: Hook Argument Expansion

**Behavior:** Fully expand object and callback arguments to hooks.

**Tests:**
```
test_hook_args_expand_object_properties
test_hook_args_capture_arrow_functions
test_hook_args_analyze_callback_bodies
test_hook_args_preserve_array_arguments
test_hook_args_handle_nested_objects
```

**Acceptance:** Given `useAgent({onComplete: ...})`, extraction includes full callback analysis.

### 5.5 Phase 6.5: Architecture Mapping

**Behavior:** Generate recommended Qliphoth architecture from hook patterns.

**Tests:**
```
test_architecture_identifies_service_actors
test_architecture_maps_zustand_stores
test_architecture_suggests_message_types
test_architecture_determines_state_ownership
test_architecture_recommends_communication_patterns
```

**Acceptance:** Given `ChatPanel`, extraction recommends `ChatService` and `AgentService` actors.

---

## 6. When to Stop and Update This Spec

Stop implementation and update this spec when:

1. **Type system gap:** Discover TypeScript pattern not covered (generics, conditional types, mapped types)
2. **Purity heuristic fails:** Function appears pure but has hidden side effects
3. **Architecture inference wrong:** Recommended pattern doesn't fit use case
4. **Qliphoth mapping unclear:** No clear translation for React pattern
5. **Test can't be written:** Spec is ambiguous about expected behavior

---

## 7. Success Metric (Behavioral)

**The spec is satisfied when:**

An agent, given only the structured extraction (without reading `source.code`), can:

1. Generate correct Qliphoth type definitions
2. Implement all helper functions as `rite`s
3. Translate handlers to actor messages
4. Wire up service actor communication
5. Produce compiling Qliphoth code

**This is a behavioral outcome, not a coverage metric.**

---

## 8. Open Questions

### 8.1 Unresolved

| Question | Impact | Owner |
|----------|--------|-------|
| How to handle `React.FC<Props>` generic? | Type extraction | TBD |
| How to map React Query to Qliphoth async? | Architecture | TBD |
| How to handle `forwardRef` with generics? | Type extraction | TBD |

### 8.2 Assumptions

| Assumption | If Wrong |
|------------|----------|
| TypeScript types are sufficient for Qliphoth | May need runtime analysis |
| Custom hooks are stateful by default | May over-generate actors |
| Zustand = service actor | May not fit all patterns |

---

## 9. Integration Points

### 9.1 TDD Roadmap

Phase 6 tests should be added to `REACT-MIGRATION-TDD-ROADMAP.md` Section 6.

### 9.2 Audit Pattern

After implementation, create `REACT-MIGRATION-PHASE6-AUDIT.md` following Phase 4 pattern.

### 9.3 Main Spec

Update `REACT-MIGRATION.md` Section 7 with Phase 6 requirements.

---

## Appendix A: Gap Discovery Session

This spec was created after running:

```bash
sigil migrate --from-react /home/crook/dev/infernum-observer/src -o /tmp/infernum-migration
```

And evaluating output fidelity. Full session documented in Conclave log.

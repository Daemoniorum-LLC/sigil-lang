# Parameterized Dispatch Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-21
**Parent Spec:** WASM-METHOD-DISPATCH-ROADMAP.md

---

## 1. Conceptual Foundation

### 1.1 Problem Statement

The Qliphoth WASM reactivity loop dispatches UI events by encoding a Sigil enum
variant's discriminant (tag) as an i64, passing it through the JS/WASM boundary,
and routing it to the root actor's `send(tag: i64)` handler. This works for
*unit variants* — variants that carry no data:

```
OpenCommandPalette     → tag = 1
CloseCommandPalette    → tag = 2
ToggleToolbar          → tag = 9
```

It breaks for *parameterized variants* — variants that carry a payload:

```
PanelClicked(String)   → tag = N, but which panel?
HidePanel(String)      → tag = M, but which panel id?
ClickCommand(String)   → tag = P, but which command id?
```

The tag alone is insufficient. The String payload must also cross the boundary.

### 1.2 Scope — Phase 1 (this spec)

**In scope:** String payload variants only.

```
on_click(Msg·Variant(string_expr))  →  parameterized dispatch with String
```

**Out of scope (future specs):**
- `usize`, `i64`, `bool` payloads
- Tuple payloads: `Msg::Move { x: f32, y: f32 }`
- Nested enum payloads

### 1.3 Mental Model

The current unit dispatch uses a single VNode prop channel:

```
Compile time:  vdom_set_vnode_prop(vnode, "on_click", tag_i64)
Runtime (JS):  tag = props["on_click"]
               window.wraithDispatch(tag)
WASM:          dispatch(tag: i64) → VNode
```

Parameterized dispatch adds a parallel payload channel using the existing
`vdom_set_vnode_str_prop` import (already declared, unused by compiler):

```
Compile time:  vdom_set_vnode_prop(vnode, "on_click", tag_i64)
               vdom_set_vnode_str_prop(vnode, "on_click_payload", str_ptr_i64)
               vdom_set_vnode_prop(vnode, "on_click_payload_len", str_len_i64)

Runtime (JS):  tag = props["on_click"]
               ptr = props["on_click_payload"]
               len = props["on_click_payload_len"]
               payload = readWasmStr(ptr, len)
               window.wraithDispatch_str(tag, ptr, len)

WASM:          dispatch_str(tag: i64, ptr: i32, len: i32) → VNode
```

The two channels are distinguished by the presence of `"on_X_payload"` alongside
`"on_X"`. A prop pair without a `_payload` sibling is treated as unit dispatch
(backward-compatible).

---

## 2. Type Architecture

### 2.1 VNode Prop Convention

For a method `·on_<event>(Msg·Variant(string_expr))`:

| Prop key              | Value type | Content                                  |
|-----------------------|------------|------------------------------------------|
| `"on_<event>"`        | i64 BigInt | Enum variant discriminant (tag)          |
| `"on_<event>_payload"`| i64 BigInt | Blob address; `memory[addr..addr+4]`=u32 len LE, `memory[addr+4..]`=UTF-8 bytes |

Note: `"on_<event>_len"` is **not emitted** — JS reads the length from the blob
header at the payload address. Presence of `_payload` alone signals a parameterized handler.

For a method `·on_<event>(Msg·UnitVariant)`:

| Prop key          | Value type | Content                    |
|-------------------|------------|----------------------------|
| `"on_<event>"`    | i64 BigInt | Enum variant discriminant  |
| (no `_payload`)   | —          | —                          |

### 2.2 WASM Export Contract

```
dispatch(tag: i64) → VNode          // unit variants — unchanged
dispatch_str(tag: i64, ptr: i32, len: i32) → VNode   // parameterized variants
```

Both return the root VNode handle as i64 for re-rendering.

### 2.3 Compiler Detection Rule

An `on_*` method argument that is an `Expr::Call` (function-call-like expression)
where the callee is an enum variant path is a parameterized handler:

```
Expr::Call {
    callee: Expr::Path { segments: [..., "PanelClicked"] },
    args:   [string_expr]          // exactly one String argument
}
```

Contrast with a unit variant, which is `Expr::Path { segments: [..., "Dismiss"] }`.

---

## 3. Behavioral Contracts

### 3.1 Compiler — `on_*` arm in `closures.rs`

**R1.** When the `on_*` argument is an `Expr::Path` (unit variant), behavior is
unchanged: emit `vdom_set_vnode_prop(vnode, key, tag)`.

**R2.** When the `on_*` argument is an `Expr::Call` whose callee is an enum
variant path and whose single argument compiles to a String value:

- Emit `vdom_set_vnode_prop(vnode, key, tag)` for the discriminant.
- Compile the string argument expression; it must leave a `(ptr: i64, len: i64)`
  pair on the stack (see §3.2 for string value convention).
- Emit `vdom_set_vnode_str_prop(vnode, key_payload, ptr)` for the pointer.
- Emit `vdom_set_vnode_prop(vnode, key_len, len)` for the length.
- Return the vnode handle for method chaining.

Where:
- `key`         = `"on_<method_name>"` (e.g., `"on_click"`)
- `key_payload` = `"on_<method_name>_payload"` (e.g., `"on_click_payload"`)
- `key_len`     = `"on_<method_name>_len"` (e.g., `"on_click_len"`)

**R3.** A String value in WASM is represented as a fat pointer: the i64 carries
the byte address in WASM linear memory; a companion i64 carries the byte count.
The compiler must be aware of which convention the string expression uses and emit
accordingly.

**R4.** If the argument cannot be resolved as a String payload (wrong type, too
many args, unsupported expression form), fall back to the existing behavior:
emit discriminant = 0, no payload props. Do NOT emit a hard error.

### 3.2 String Value Convention (WASM)

A Sigil `String!` value in the WASM backend is represented as two i64 values:
`(ptr, len)` where `ptr` is a pointer into WASM linear memory and `len` is the
byte count.

For the `on_*` payload, the compiler must save both from the compiled string
expression before they are consumed:

```
// Pseudocode — not binding implementation
compile_string_expr(string_arg)  // → ptr on stack, len on stack...
                                 // (convention TBD by implementer)
```

⚠️ **Open Question §7.1**: The exact stack convention for String values in the
current WASM backend must be verified before implementation. The implementer must
read how existing string-returning rites deposit their result and match that.

### 3.3 JS Runtime — `applyVNode` prop handling

**R5.** When walking props in `applyVNode`, if a prop key `k` starts with `"on"`
AND `props[k + "_payload"]` exists AND `props[k + "_len"]` exists:

- This is a parameterized handler.
- Do NOT call `window.wraithDispatch` (unit path).
- Instead: extract `ptr = Number(props[k + "_payload"])`, `len = Number(props[k + "_len"])`.
- Read the payload string from WASM memory: `readWasmStr(ptr, len)`.
- Register a DOM event listener that calls `window.wraithDispatch_str(msgTag, ptr, len)`.

**R6.** `window.wraithDispatch_str(tag, ptr, len)` calls `exp.dispatch_str(BigInt(tag), ptr, len)` and passes the result to `rerender()`. On error it logs and returns without crashing.

**R7.** `window.wraithDispatch` (unit path) is unchanged. Backward compatibility
with all existing unit-variant handlers is required.

### 3.4 Application — `dispatch_str` export

**R8.** The application's `lib.sigil` must export a `dispatch_str(tag: i64, ptr: i32, len: i32) -> VNode!` rite that:

- Reconstructs the payload `&str` from `(ptr, len)`.
- Routes `(tag, payload)` to the root actor's parameterized send handler.
- Returns a fresh root VNode.

**R9.** The root actor must expose a `send_str(tag: i64, payload: &str) -> VNode!`
rite (or equivalent) that handles parameterized message tags via if-else chain,
mirroring the unit `send(tag: i64)` pattern.

---

## 4. Constraints & Invariants

**P1.** Unit dispatch MUST remain functionally identical. All existing 40 tests
must continue to pass after this change.

**P2.** The `on_<event>_payload` and `on_<event>_len` props are only emitted when
the argument is a detected String payload. They are never emitted for unit handlers.

**P3.** `dispatch_str` MUST return a valid VNode handle (> 0) for any tag + any
well-formed UTF-8 payload string.

**P4.** Calling `dispatch_str` with an unrecognized tag MUST NOT crash — it should
fall through and return a fresh `view()` VNode (same as `send()` for unknown tags).

**P5.** The prop key naming convention uses `_payload` and `_len` suffixes. These
are not valid Sigil event names and cannot collide with legitimate `on_*` event props.

---

## 5. Error Conditions

| Condition | Expected behavior |
|-----------|-------------------|
| `on_*` arg is a Call expr with non-String payload | Fallback to discriminant=0, no payload props (R4) |
| `on_*` arg is a Call expr with > 1 args | Same fallback |
| `dispatch_str` called with out-of-bounds ptr/len | Undefined behavior (UB) — string extraction from WASM memory is inherently unsafe; document as caller responsibility |
| `dispatch_str` called with unrecognized tag | Return fresh `view()` VNode (P4) |
| Payload is not valid UTF-8 | UB — same as above; Sigil strings are defined as UTF-8 |

---

## 6. Integration Points

### 6.1 Files Changed

| File | Change |
|------|--------|
| `parser/src/wasm/closures.rs` | Extend `on_*` arm: detect Call-expr arg, emit payload props |
| `wraith-sigil/wraith.js` | Extend `applyVNode` prop loop; add `wraithDispatch_str`; `SIGIL_EVENT_MAP` needs `_payload`/`_len` exemption |
| `wraith-sigil/src/lib.sigil` | Add `☉ rite dispatch_str(tag: i64, ptr: i32, len: i32) -> VNode!` export |
| `wraith-sigil/src/wraith.sigil` | Add `☉ rite send_str(tag: i64, payload: &str) -> VNode!` to handle parameterized tags |

### 6.2 Files Unchanged (Verified)

| File | Reason |
|------|--------|
| `parser/src/wasm/imports.rs` | `vdom_set_vnode_str_prop` already declared |
| `wraith-sigil/test.js` | New Suite 10 tests ADDED (not modifying old suites) |

### 6.3 Trust Boundaries

- **WASM → JS**: VNode props are written at compile/render time. JS reads them at DOM event time. The gap (re-renders between write and read) is bounded by the single-threaded WASM event model — no race conditions.
- **JS → WASM**: `dispatch_str(ptr, len)` passes a raw pointer. The Sigil runtime trusts that the pointer is valid for the lifetime of the call. This is the same trust level as the existing `browser_main()` → WASM boundary.

---

## 7. Open Questions

**7.1 String stack convention**: ~~Open~~ **RESOLVED (2026-02-21)**

A compiled Sigil `String!` expression leaves **one i64** on the stack. It is a
data-section or heap offset (blob address) to a length-prefixed blob:

```
memory[addr..addr+4]  = u32 length (little-endian)
memory[addr+4..addr+4+len] = UTF-8 bytes
```

String literals: `I32Const(offset) + I64ExtendI32U`.
`format!()`: chain of `string_from_int` / `string_concat` calls → final `i64`.

**Consequence for `on_*` compiler arm**: the payload prop is stored via
`vdom_set_vnode_prop(vnode, key_payload, blob_addr_i64)` — stores the raw i64
blob address as a BigInt value. No separate `_len` prop is needed because JS
reads the length from the blob header via `DataView.getUint32(blobAddr, true)`.

**Consequence for `dispatch_str` callers**: all Sigil WASM exports use i64
(BigInt) for all parameters, including `ptr` and `len` (even when annotated
`i32` in Sigil source). Callers must pass BigInt: `exp.dispatch_str(tag, BigInt(ptr), BigInt(len))`.

**7.2 String lifetime**: The payload string is allocated in WASM heap at render
time. If the actor state is mutated between render and click (triggering a re-render),
the pointer stored in the VNode prop may reference freed memory. Is this a real
concern given the current single-threaded, synchronous event model?

- **Current assumption**: Re-renders only occur via `wraithDispatch`, which is
  a user-initiated event. Between renders the actor state is frozen. The pointer
  is valid for the full click-to-dispatch cycle.

**7.3 Payload key naming**: `on_<event>_payload` / `on_<event>_len`. If a future
event is named `on_foo_payload`, this collides. Is this a realistic concern?

- **Current assumption**: No. Sigil event names are snake_case identifiers derived
  from method names. `_payload` and `_len` suffixes are reserved by this spec.

---

## 8. Test Plan (Agent-TDD)

Tests live in `wraith-sigil/test.js` as Suite 10.

### 8.1 Specification Tests

```
S10-T1: dispatch_str is exported from WASM
        exp.dispatch_str must be typeof 'function'

S10-T2: dispatch_str returns a valid VNode handle
        exp.dispatch_str(BigInt(tag), 0, 0) must return handle > 0
        (zero ptr/len = empty string = valid input)

S10-T3: dispatch_str with unrecognized tag returns VNode (does not crash)
        exp.dispatch_str(BigInt(9999), 0, 0) must return handle > 0

S10-T4: dispatch_str and dispatch return structurally equivalent VNodes for same state
        dispatch(1n) then dispatch(2n) followed by
        dispatch_str(1n, 0, 0) then dispatch_str(2n, 0, 0)
        both open/close CommandPalette — result handles must be > 0
```

### 8.2 Property Tests

```
S10-P1: ∀ unit-variant tag t ∈ [0, 22]:
        Number(exp.dispatch(BigInt(t))) > 0

S10-P2: ∀ tag t and any ptr, len:
        exp.dispatch_str(BigInt(t), ptr, len) does not throw
```

### 8.3 Regression Guard

```
S10-R1: All 40 existing tests continue to pass (run full suite after change)
```

---

## 9. Implementation Order (Agent-TDD Sequence)

Following Agent-TDD: **SPECIFY → RED → GREEN → VERIFY**

1. **Write Suite 10 tests** → RED (dispatch_str not yet exported)
2. **Add `dispatch_str` export** in `wraith-sigil/src/lib.sigil` and `wraith.sigil` → partially GREEN
3. **Extend compiler `on_*` arm** in `closures.rs` to detect Call-expr payload, emit props → compile
4. **Extend `applyVNode`** and add `wraithDispatch_str` in `wraith.js` → runtime GREEN
5. **Rebuild + run tests** → full GREEN
6. **Update qliphoth-ide components** to use parameterized `on_click` where needed → verifies end-to-end

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-21 | Initial draft. Authored during qliphoth-ide code review when parameterized on_click was identified as blocking. |
| 0.2.0 | 2026-02-21 | Implementation complete. Resolved §7.1: string stack convention = single i64 blob address with `[u32_len][utf8_bytes]` layout; `_len` prop dropped (JS reads from blob header). All 46 tests pass — Suite 10 GREEN. |

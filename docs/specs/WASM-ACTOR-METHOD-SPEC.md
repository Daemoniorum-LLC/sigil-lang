# WASM Actor and Method Resolution Specification

**Version:** 0.3.0
**Status:** Complete
**Date:** 2026-02-14
**Last Updated:** 2026-02-14
**Parent Spec:** 06-CONCURRENCY.md (Actor model), 02-SYNTAX.md (Method call syntax)

---

## Abstract

This specification defines the implementation of actor compilation and improved method
resolution for the Sigil WASM backend. These features are required to compile qliphoth
web applications which use the agent-centric architecture (actors instead of React hooks).

---

## 1. Problem Statement

### 1.1 Current State

The WASM backend fails to compile qliphoth applications with:

```
Error: actors not supported
Error: undefined function: child
Error: undefined function: join
```

### 1.2 Root Causes

**Gap 1: Actors not supported**
```rust
// statements.rs:717
Item::Actor(_) => Err(WasmError::unsupported("actors")),
```

**Gap 2: Method resolution doesn't check qualified names**
```rust
// closures.rs:1128-1136
if let Some(func_idx) = self.get_func(method) {
    // Only checks simple name "child"
    // Doesn't try "VNode::child" or qualified lookup
}
```

**Gap 3: Vec::join not implemented**
The `try_compile_builtin_method` switch has `push`, `pop`, `get`, but not `join`.

### 1.3 Affected Code

All qliphoth apps use these patterns:

```sigil
// Actor definition (Gap 1)
☉ actor PlatformApp {
    state current_route: Route! = Route·Landing

    on Navigate(route: Route) {
        self.current_route = route
    }

    rite view(&self) -> VNode! {
        VNode·div()·child(...)  // Method chain (Gap 2)
    }
}

// Vec::join (Gap 3)
≔ css! = parts·join("; ");
```

---

## 2. Design Goals

### 2.1 Actor Compilation

Actors should compile to:
- State fields → local variables or globals
- Message handlers (`on Message`) → functions
- Methods (`rite`) → regular functions with self parameter
- The `view` method → exported function for rendering

### 2.2 Method Resolution

Method calls should resolve in order:
1. Builtin methods (to_string, clone, unwrap, etc.)
2. Type-qualified names (`VNode::child`)
3. Simple function names (current behavior)
4. External imports

### 2.3 Stdlib Methods

Add missing collection methods:
- `Vec::join(separator)` → concatenate elements with separator

---

## 3. Behavioral Contracts

### 3.1 Actor State Fields

```
actor Counter {
    state count: i64! = 0
}

Compiles to:
    - Global variable: `Counter__count` (mutable i64)
    - Initialization in __wasm_start
```

### 3.2 Actor Message Handlers

```sigil
on Increment {
    self.count += 1
}

Compiles to:
    fn Counter__handle_Increment() {
        Counter__count += 1
    }
```

### 3.3 Actor Methods

```sigil
rite view(&self) -> VNode! {
    VNode·div()
}

Compiles to:
    pub fn Counter__view() -> i64 {
        // VNode builder calls
    }
```

### 3.4 Method Resolution Algorithm

```
resolve_method(receiver_type, method_name, args):

    // 1. Check enum variant access
    if receiver is enum type path:
        if method_name is variant:
            return enum_variant_access

    // 2. Check builtin methods
    if method_name in BUILTIN_METHODS:
        return compile_builtin(receiver, method_name, args)

    // 3. Try type-qualified lookup
    qualified_name = infer_receiver_type(receiver) + "::" + method_name
    if func_map.contains(qualified_name):
        return call(func_map[qualified_name])

    // 4. Try simple name lookup
    if func_map.contains(method_name):
        return call(func_map[method_name])

    // 5. Try external import
    return import_and_call(method_name)
```

### 3.5 Vec::join Contract

```sigil
parts·join("; ")

Input: Vec<String> pointer, separator String pointer
Output: New String pointer (concatenated with separators)

Compiles to:
    local.get $parts
    local.get $separator
    call $vec_join
```

---

## 4. Implementation Strategy

### 4.1 Phase 1: Actor State as Globals

**File:** `statements.rs`

```rust
Item::Actor(actor) => {
    let actor_name = &actor.name.name;

    // Register state fields as globals
    for member in &actor.members {
        if let ActorMember::State { name, ty, init } = member {
            let global_name = format!("{}_{}", actor_name, name.name);
            self.register_global(&global_name, ty, init)?;
        }
    }

    // Push actor name to module path for qualified names
    self.module_path.push(actor_name.clone());

    // Register message handlers and methods
    for member in &actor.members {
        match member {
            ActorMember::Handler { message, body } => {
                self.register_handler(actor_name, message, body)?;
            }
            ActorMember::Method(func) => {
                self.register_function_sig(func)?;
            }
            _ => {}
        }
    }

    self.module_path.pop();
    Ok(())
}
```

### 4.2 Phase 2: Method Resolution Enhancement

**File:** `closures.rs` - `compile_method_call`

```rust
fn compile_method_call(&mut self, receiver: &Expr, method: &str, args: &[Expr]) -> WasmResult<()> {
    // ... existing enum variant check ...

    // Try builtin method dispatch
    if self.try_compile_builtin_method(receiver, method, args)? {
        return Ok(());
    }

    // Compile receiver as first argument
    self.compile_expr(receiver)?;

    // Compile remaining arguments
    for arg in args {
        self.compile_expr(arg)?;
    }

    // NEW: Try type-qualified lookup
    if let Some(receiver_type) = self.infer_receiver_type(receiver) {
        let qualified = format!("{}::{}", receiver_type, method);
        if let Some(func_idx) = self.func_map.get(&qualified) {
            let func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::Call(*func_idx));
            return Ok(());
        }
    }

    // Existing: simple name lookup
    if let Some(func_idx) = self.get_func(method) {
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(func_idx));
        return Ok(());
    }

    Err(WasmError::undefined_function(method))
}

/// Infer the type of a receiver expression.
fn infer_receiver_type(&self, expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(path) => {
            // VNode::div() returns VNode
            let name = path.segments.first()?.ident.name.as_str();
            if self.struct_layouts.contains_key(name) || self.enum_layouts.contains_key(name) {
                return Some(name.to_string());
            }
            None
        }
        Expr::Call { func, .. } => {
            // foo() - check if foo's return type is known
            if let Expr::Path(path) = &**func {
                let func_name = path.segments.last()?.ident.name.as_str();
                // Check if this is a type constructor like VNode::div
                if let Some(type_name) = func_name.split("::").next() {
                    if self.struct_layouts.contains_key(type_name) {
                        return Some(type_name.to_string());
                    }
                }
            }
            None
        }
        Expr::MethodCall { receiver, method, .. } => {
            // Chain inference: receiver.method() - if we know receiver type
            // and method returns same type (builder pattern), return that type
            if let Some(base_type) = self.infer_receiver_type(receiver) {
                // Builder pattern: VNode methods return VNode
                return Some(base_type);
            }
            None
        }
        _ => None
    }
}
```

### 4.3 Phase 3: Vec::join Builtin

**File:** `closures.rs` - `try_compile_builtin_method`

```rust
"join" => {
    // vec.join(separator) -> string
    // Stack: [vec_ptr, separator_ptr] -> [result_ptr]
    self.compile_expr(receiver)?;
    if let Some(sep) = args.first() {
        self.compile_expr(sep)?;
    } else {
        // Default separator: empty string
        let empty = self.add_string("");
        let func = self.current_function_mut()?;
        func.push(Instruction::I64Const(empty as i64));
    }

    let join_idx = self.get_func("vec_join")
        .ok_or_else(|| WasmError::internal("vec_join import missing"))?;
    let func = self.current_function_mut()?;
    func.push(Instruction::Call(join_idx));

    Ok(true)
}
```

**File:** `imports.rs` - Add import

```rust
fn register_morpheme_imports(&mut self) {
    // ... existing ...

    // Vec::join
    self.add_import("morpheme", "vec_join", vec![I64, I64], vec![I64]);
}
```

---

## 5. Test Specifications

### 5.1 Actor Compilation Tests

```rust
#[test]
fn test_actor_state_as_global() {
    let mut compiler = WasmCompiler::new();
    let result = compiler.compile(r#"
        ☉ actor Counter {
            state count: i64! = 0
        }
    "#);
    assert!(result.is_ok());
    assert!(compiler.global_map.contains_key("Counter_count"));
}

#[test]
fn test_actor_message_handler() {
    let mut compiler = WasmCompiler::new();
    let result = compiler.compile(r#"
        ☉ actor Counter {
            state count: i64! = 0

            on Increment {
                self.count += 1
            }
        }
    "#);
    assert!(result.is_ok());
    assert!(compiler.func_map.contains_key("Counter__handle_Increment"));
}

#[test]
fn test_actor_view_method() {
    let mut compiler = WasmCompiler::new();
    let result = compiler.compile(r#"
        ☉ actor Counter {
            state count: i64! = 0

            rite view(&self) -> i64! {
                self.count
            }
        }
    "#);
    assert!(result.is_ok());
    assert!(compiler.func_map.contains_key("Counter::view"));
}
```

### 5.2 Method Resolution Tests

```rust
#[test]
fn test_method_chain_resolution() {
    let mut compiler = WasmCompiler::new();
    let result = compiler.compile(r#"
        ☉ Σ Builder {
            value: i64!
        }

        ⊢ Builder {
            ☉ rite new() -> Self! {
                Builder { value: 0 }
            }

            ☉ rite set(Δ self, v: i64) -> Self! {
                self.value = v
                self
            }
        }

        ☉ rite test() -> i64! {
            Builder·new()·set(42).value
        }
    "#);
    assert!(result.is_ok());
}
```

### 5.3 Vec::join Tests

```rust
#[test]
fn test_vec_join_basic() {
    let mut compiler = WasmCompiler::new();
    let result = compiler.compile(r#"
        ☉ rite format_list(items: Vec<String>) -> String! {
            items·join(", ")
        }
    "#);
    assert!(result.is_ok());
}

#[test]
fn test_vec_join_empty_separator() {
    let mut compiler = WasmCompiler::new();
    let result = compiler.compile(r#"
        ☉ rite concat_all(parts: Vec<String>) -> String! {
            parts·join("")
        }
    "#);
    assert!(result.is_ok());
}
```

---

## 6. Integration Points

### 6.1 With qliphoth VNode Builder

After implementation, this pattern compiles:

```sigil
VNode·div()
    ·class("container")
    ·child(VNode·h1()·text_child("Title"))
    ·child(VNode·p()·text_child("Content"))
```

The method chain resolves as:
1. `VNode::div()` → returns VNode
2. `.class("container")` → resolves `VNode::class`, returns VNode
3. `.child(...)` → resolves `VNode::child`, returns VNode

### 6.2 With qliphoth CSS Generation

```sigil
☉ rite generate_css() -> String! {
    ≔ Δ parts! = Vec·new();
    parts·push("body { color: black; }");
    parts·push("h1 { font-size: 2rem; }");
    parts·join("\n")  // Now works
}
```

### 6.3 With qliphoth Actors

```sigil
☉ actor BlogApp {
    state current_route: Route! = Route·Home

    on Navigate(route: Route) {
        self.current_route = route
    }

    rite view(&self) -> VNode! {
        VNode·div()
            ·child(components·nav_view())
            ·child(self·route_view())
            ·child(components·footer_view())
    }
}
```

---

## 7. Implementation Order

Following SDD + Agent TDD:

1. **Add Vec::join builtin** (smallest change, unblocks CSS generation)
   - Add import to imports.rs
   - Add case to try_compile_builtin_method
   - Write test

2. **Enhance method resolution** (unblocks VNode chains)
   - Add infer_receiver_type()
   - Modify compile_method_call to try qualified names
   - Write tests

3. **Implement actor compilation** (largest change, unblocks apps)
   - Parse actor state fields as globals
   - Parse handlers as functions
   - Parse methods as functions
   - Write tests

---

## 8. Open Questions

1. **Actor instance state:** Should each actor instance have separate state?
   - Current design: Singleton actors (one global per state field)
   - Future: Instance-based actors with heap allocation

2. **Message dispatch:** How are messages routed to handlers?
   - Current design: Direct function calls
   - Future: Message queue with async dispatch

3. **Self reference in actors:** How does `self.field` resolve?
   - Current design: Global variable lookup
   - Future: Instance pointer dereferencing

---

## 9. Prerequisites

| Prerequisite | Status | Notes |
|--------------|--------|-------|
| AST Actor node parsing | ✅ | Parser supports actors |
| Runtime vec_join import | ✅ | Import registered, runtime implements |
| Type inference for receivers | ✅ | infer_receiver_type() with struct/enum layouts |

---

## 10. Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Vec::join builtin | ✅ | Phase 1 - Implemented |
| Method qualified lookup | ✅ | Phase 2 - Implemented |
| Short Type::method names | ✅ | Phase 2 - Added for method resolution |
| Qualified const/static | ✅ | Phase 2 - Module path constant access |
| Macro token stringification | ✅ | Bugfix - Sigil symbols preserved |
| Actor state globals | ✅ | Phase 3 - Implemented |
| Actor handlers | ✅ | Phase 3 - Implemented |
| Actor methods | ✅ | Phase 3 - Implemented |
| self.field access | ✅ | Phase 3 - Actor state via globals |
| self·method() calls | ✅ | Phase 3 - Actor method dispatch |
| Multi-segment enum paths | ✅ | Bugfix - api·Type·Variant resolution |
| Qualified struct/enum names | ✅ | Bugfix - Module-qualified type registration |

**Result:** All 4 qliphoth apps compile successfully to WASM:
- qliphoth-app: 61.9 KB ✓
- qliphoth-blog: 54.6 KB ✓
- qliphoth-docs: 55.8 KB ✓
- qliphoth-chat: 58.5 KB ✓

---

## 11. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-14 | Initial draft. Gap discovered during qliphoth compilation. |
| 0.2.0 | 2026-02-14 | Phase 1 & 2 implemented. Vec::join, method resolution, qualified names. |
| 0.3.0 | 2026-02-14 | Phase 3 complete. Actor compilation, self.field access, multi-segment enum paths. All 4 qliphoth apps compile. |

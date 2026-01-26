# Sigil Web Interface

> A React-inspired web framework for the [Sigil Programming Language](https://github.com/Daemoniorum-LLC/sigil-lang)

Part of the [Persona Framework](https://github.com/Daemoniorum-LLC/persona-framework) ecosystem.

## Overview

Sigil Web is a complete, type-safe web framework that brings Sigil's polysynthetic programming paradigm to web development. It combines the familiar component-based architecture of React with Sigil's unique features: evidentiality markers, morpheme operators, and actor-based concurrency.

## Features

### Evidentiality-Driven State

Track data provenance with Sigil's unique type markers:

| Marker | Name | Meaning |
|--------|------|---------|
| `!` | Known | Locally computed/verified values |
| `?` | Uncertain | May be absent or missing |
| `~` | Reported | External/untrusted sources |
| `‽` | Paradox | Trust boundary crossings |

```sigil
// Data from API is marked as external (~)
let user~ = fetch("/api/user")⌛

// Local computation is known (!)
let total! = items|τ{i => i.price}|ρ+

// Pattern matching respects evidentiality
match user~ {
    Success(data~) => render_user(data~),
    Error(e~) => show_error(e~),
    Loading => Spinner {}
}
```

### Morpheme Operators

Transform data with Greek-letter pipeline operators:

| Operator | Name | Function |
|----------|------|----------|
| `τ` | Tau | Map/transform |
| `φ` | Phi | Filter |
| `σ` | Sigma | Sort |
| `ρ` | Rho | Reduce/fold |
| `α` | Alpha | First element |
| `Ω` | Omega | Last element |
| `π` | Pi | Project/select |
| `Σ` | Sigma (capital) | Sum |
| `Π` | Pi (capital) | Product |

```sigil
// Pipeline transformations
let results! = users
    |φ{u => u.active}           // filter active users
    |τ{u => u.score * 2}        // map: double scores
    |σ                          // sort ascending
    |ρ+                         // reduce: sum all

// Get first/last elements
let first! = items|α
let last! = items|Ω
```

### Parallel Morphemes

Concurrent operations for large datasets:

```sigil
// Parallel map
let processed! = large_dataset|par·τ{expensive_compute}|collect()

// Parallel filter
let filtered! = records|par·φ{r => r.valid}|collect()

// Parallel reduce
let total! = numbers|par·ρ+

// Parallel sort
let sorted! = items|par·σ
```

### Modern Async Patterns

Clean async syntax with `⌛` and parallel combinators:

```sigil
// Single await using ⌛
let data~ = fetch("/api/users")⌛

// Parallel awaits with await·all
let (users~, posts~, comments~) = (
    fetch_users(),
    fetch_posts(),
    fetch_comments()
)·await·all

// Race for fastest response
let fastest~ = mirrors|τ{fetch}·await·race
```

### Actor-Based State Management

Predictable state updates via message passing:

```sigil
actor Store<S, A> {
    state current: S!

    on Dispatch(action: A) {
        self.current = (self.reducer)(self.current, action)
        self·notify_subscribers()
    }
}

// Fire-and-forget messaging with `tell`
store·tell(Dispatch { action: Increment })

// Request-response with `ask`
let state! = store·ask(GetState {})⌛
```

### Set Operations

Mathematical set operations for collections:

| Operator | Name | Function |
|----------|------|----------|
| `∪` | Union | Combine sets |
| `∩` | Intersection | Common elements |
| `∖` | Difference | Elements in A not in B |
| `∈` | Element of | Membership test |
| `∉` | Not element of | Non-membership test |

```sigil
let active_admins! = active_users ∩ admin_users
let non_subscribers! = all_users ∖ subscribers

if user ∈ premium_users {
    show_premium_content()
}
```

### Logical Operators

Mathematical notation for boolean logic:

| Operator | Name | Function |
|----------|------|----------|
| `∧` | And | Logical conjunction |
| `∨` | Or | Logical disjunction |
| `¬` | Not | Logical negation |

```sigil
if user.active ∧ user.verified {
    grant_access()
}

let classes! = classes()
    ·add("btn")
    ·add_if(is_primary ∨ is_default, "btn-primary")
    ·add_unless(¬enabled, "btn-disabled")
```

## Quick Start

```sigil
use sigil_web::prelude::*

component Counter {
    state count: i64! = 0

    fn render(self) -> Element {
        div {
            h1 { "Count: {self.count}" }
            button[onclick: || self.count += 1] { "+" }
            button[onclick: || self.count -= 1] { "-" }
        }
    }
}

fn main() {
    App::mount("#root", Counter::new())
}
```

## Core Modules

| Module | Description |
|--------|-------------|
| `core/` | Virtual DOM, reconciler, scheduler, renderer |
| `components/` | Component model, ErrorBoundary, Suspense, Memo |
| `hooks/` | 15+ React-style hooks with evidentiality |
| `router/` | Type-safe client-side routing |
| `state/` | Actor-based stores, signals, atoms |
| `dom/` | JSX-like element builders |
| `platform/` | Browser/Server/Native abstractions |

## Hooks

### State Hooks
- `use_state` / `use_state_with` - Local component state
- `use_reducer` - Complex state with reducer pattern

### Effect Hooks
- `use_effect` - Side effects after render
- `use_layout_effect` - Synchronous effects

### Performance Hooks
- `use_memo` - Memoize expensive computations
- `use_callback` - Memoize callbacks

### Data Fetching Hooks
- `use_fetch` / `use_fetch_with` - Fetch data from URLs
- `use_fetch_all` - Parallel fetches with `await·all`
- `use_fetch_race` - Race fetches with `await·race`
- `use_mutation` - Handle data mutations

### Utility Hooks
- `use_debounce` / `use_throttle` - Rate limiting
- `use_local_storage` - Persist to localStorage
- `use_media_query` - Responsive breakpoints
- `use_window_size` - Track window dimensions

## Project Structure

```
sigil-web-interface/
├── src/
│   ├── lib.sigil           # Main entry point
│   ├── core/               # Runtime and VDOM
│   ├── components/         # Component system
│   ├── hooks/              # React-style hooks
│   ├── router/             # Routing system
│   ├── state/              # State management
│   ├── dom/                # DOM builders
│   └── platform/           # Platform abstractions
├── examples/
│   ├── basic/              # Simple examples
│   └── docs-platform/      # Full documentation app
└── docs/
    └── guides/             # Getting started guides
```

## Examples

### Todo App
```sigil
use sigil_web::prelude::*

component TodoApp {
    state todos: Vec<Todo>! = vec![]
    state input: String! = ""

    fn add_todo(mut self) {
        if ¬self.input·is_empty() {
            self.todos·push(Todo::new(&self.input))
            self.input = ""
        }
    }

    fn render(self) -> Element {
        div[class: "todo-app"] {
            h1 { "Todo List" }

            form[onsubmit: || self·add_todo()] {
                input[
                    value: self.input,
                    oninput: |v| self.input = v
                ]
                button { "Add" }
            }

            ul {
                // Using τ morpheme for mapping
                self.todos|τ{todo =>
                    li[key: todo.id] {
                        span { todo.text }
                        button[onclick: || todo.done = ¬todo.done] {
                            if todo.done { "Undo" } else { "Done" }
                        }
                    }
                }|fragment
            }
        }
    }
}
```

### Data Fetching with Evidentiality
```sigil
use sigil_web::prelude::*

fn UserList() -> VNode {
    // Data from API marked as external (~)
    let users~ = use_fetch::<Vec<User>>("/api/users")

    match users~ {
        AsyncState::Loading => {
            div[class: "loading"] { Spinner {} }
        }
        AsyncState::Error(e~) => {
            // Error is also external data
            div[class: "error"] { "Failed to load: {e~}" }
        }
        AsyncState::Success(data~) => {
            ul {
                // Transform external data
                data~|τ{user~ =>
                    li[key: user~.id] { user~.name }
                }|fragment
            }
        }
        AsyncState::Idle => VNode::Empty
    }
}
```

## WebAssembly Compilation

Sigil Web Interface includes a complete WebAssembly backend that compiles Sigil code directly to browser-executable WASM modules.

### Compiling to WASM

```bash
# Compile a Sigil file to WebAssembly
sigil wasm app.sigil -o app.wasm
```

### Browser Integration

```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import { initSigil } from './sigil_runtime.js';

        const sigil = await initSigil('./app.wasm', { debug: true });
        sigil.main();
    </script>
</head>
<body>
    <div id="app"></div>
</body>
</html>
```

### WASM Architecture

The WASM backend preserves Sigil's unique features at runtime:

| Feature | WASM Implementation |
|---------|---------------------|
| **Evidentiality** | High bits of i64 values (0x0=known, 0x1=uncertain, 0x2=reported, 0x3=paradox) |
| **Morpheme Operators** | JavaScript runtime functions for τ, φ, σ, ρ, α, Ω |
| **DOM Bindings** | Imported functions for createElement, setAttribute, events |
| **VDOM** | Efficient diff/patch via imported functions |
| **Async** | State machine compilation with JS Promise integration |

### JS Runtime API

The `sigil_runtime.js` provides comprehensive browser integration:

#### Core Modules
| Module | Functions | Description |
|--------|-----------|-------------|
| **console** | `log_i64`, `log_f64`, `log_str` | Logging with evidentiality tags |
| **dom** | `create_element`, `set_attribute`, `append_child`, etc. | Direct DOM manipulation |
| **events** | `add_listener`, `remove_listener` | Event binding with WASM callbacks |
| **timing** | `now`, `set_timeout`, `request_animation_frame` | Timing and animation |
| **fetch** | `start`, `poll`, `get_body`, `abort` | HTTP requests |
| **storage** | `local_get`, `local_set`, `local_remove` | LocalStorage access |
| **router** | `push_state`, `replace_state`, `get_pathname` | SPA routing |

#### Morpheme Operations
| Function | Description |
|----------|-------------|
| `array_new`, `array_push`, `array_get`, `array_set` | Array management |
| `array_map(arrId, callbackIdx)` | τ operator - transform elements |
| `array_filter(arrId, callbackIdx)` | φ operator - filter elements |
| `array_reduce(arrId, callbackIdx, initial)` | ρ operator - fold/reduce |
| `array_sort(arrId)` | σ operator - sort in place |
| `array_first`, `array_last`, `array_nth` | α, Ω, ν operators - element access |

#### VDOM (Virtual DOM)
| Function | Description |
|----------|-------------|
| `create_vnode`, `create_text_vnode`, `create_fragment` | VNode creation |
| `set_vnode_prop`, `set_vnode_str_prop` | Property management |
| `append_vnode_child` | Tree building |
| `mount_vnode` | Initial render to DOM |
| `diff_and_patch` | Efficient reconciliation with keyed children |
| `dispose` | Cleanup VNode tree |

#### Signals (Reactivity)
| Function | Description |
|----------|-------------|
| `create(value)` | Create reactive signal |
| `get(signalId)`, `set(signalId, value)` | Read/write signal |
| `subscribe(signalId, callbackIdx)` | React to changes |
| `batch_start`, `batch_end` | Batch multiple updates |
| `computed(computeFnIdx)` | Derived signals |
| `effect(effectFnIdx)` | Side effects on change |

#### Async Operations
| Function | Description |
|----------|-------------|
| `promise_new`, `promise_resolve`, `promise_reject` | Promise management |
| `promise_then`, `promise_catch` | Chaining |
| `promise_all`, `promise_race` | Combinators |
| `spawn(taskFnIdx)` | Cooperative multitasking |
| `yield_now` | Yield to other tasks |

## Development

### Building

Requires Sigil compiler 0.1.0 or later:

```bash
sigil build
```

### Building for Web (WASM)

```bash
# Enable WASM feature
cargo build --features wasm --release

# Compile Sigil to WASM
sigil wasm src/main.sigil -o dist/app.wasm
```

### Testing

```bash
sigil test
```

### Running Examples

```bash
cd examples/basic
sigil run counter.sigil
```

## Extraction Details

- **Extraction Date:** 2025-12-01 23:26:06 UTC
- **Commits Preserved:** 78
- **Updated:** 2025-12-02 (WASM backend implementation)

## License

Proprietary - Daemoniorum LLC

## Part of Daemoniorum LLC

This project is maintained by Daemoniorum LLC as part of the Persona Framework ecosystem.

---

For the Sigil language specification, see [sigil-lang](https://github.com/Daemoniorum-LLC/sigil-lang).

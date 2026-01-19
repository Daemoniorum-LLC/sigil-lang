# Qliphoth

A React-inspired web application framework built on Sigil's polysynthetic programming paradigm.

## Overview

Qliphoth leverages Sigil's unique features to create a powerful, type-safe web framework:

- **Evidentiality-Driven State**: Track data provenance (`!` computed, `?` cached, `~` remote, `‽` untrusted)
- **Morpheme Components**: Compose UI with pipe operators and Greek letter transformations
- **Actor-Based State Management**: Predictable state updates via message passing
- **Zero-Cost Abstractions**: Compile-time optimization for production builds

## Quick Start

```sigil
use qliphoth::prelude::*

// Define a component
component Counter {
    state count: i64! = 0

    fn render(self) -> Element {
        div {
            h1 { "Count: {self.count}" }
            button[onclick: || self.count += 1] { "Increment" }
        }
    }
}

// Mount to DOM
fn main() {
    App::mount("#root", Counter::new())
}
```

## Core Concepts

### Components

Components are the building blocks of Qliphoth applications:

```sigil
// Functional component
fn Greeting(props: {name: String}) -> Element {
    h1 { "Hello, {props.name}!" }
}

// Stateful component
component Timer {
    state seconds: i64! = 0

    on Mount {
        interval(1000, || self.seconds += 1)
    }

    fn render(self) -> Element {
        span { "Elapsed: {self.seconds}s" }
    }
}
```

### Evidentiality in UI

Sigil's evidentiality system naturally maps to UI data flow:

| Marker | Meaning | UI Context |
|--------|---------|------------|
| `!` | Known/Computed | Local state, derived values |
| `?` | Uncertain | Optional props, nullable data |
| `~` | Reported | API responses, external data |
| `‽` | Paradox | User input, untrusted sources |

```sigil
component UserProfile {
    state user: User~ = User::empty()  // Remote data
    state editing: bool! = false        // Local state

    fn render(self) -> Element {
        match self.user {
            User::empty() => Spinner {},
            user~ => ProfileCard { user: user~|validate‽ }
        }
    }
}
```

### Pipe-Based Composition

Use Sigil's pipe operators for elegant component composition:

```sigil
fn UserList(users: Vec<User>~) -> Element {
    users
        |φ{_.active}           // Filter active users
        |σ{_.name}             // Sort by name
        |τ{user => UserCard { user }}  // Map to components
        |into_fragment         // Collect into fragment
}
```

### Hooks

React-inspired hooks with evidentiality tracking:

```sigil
fn SearchBox() -> Element {
    let (query, set_query) = use_state!("")
    let results~ = use_fetch("/api/search?q={query}")
    let debounced? = use_debounce(query, 300)

    div {
        input[value: query, oninput: set_query]
        match results~ {
            Loading => Spinner {},
            Error(e~) => ErrorBanner { message: e~ },
            Data(items~) => ResultList { items: items~ }
        }
    }
}
```

### Routing

Declarative routing with type-safe parameters:

```sigil
use qliphoth::router::*

fn App() -> Element {
    Router {
        Route[path: "/"] { Home {} }
        Route[path: "/docs/:section"] { |params|
            Docs { section: params.section }
        }
        Route[path: "/api/:module/:function"] { |params|
            ApiReference {
                module: params.module,
                function: params.function
            }
        }
        Route[path: "*"] { NotFound {} }
    }
}
```

## Architecture

```
qliphoth/
├── src/
│   ├── core/           # Core runtime and reconciliation
│   ├── components/     # Base component system
│   ├── hooks/          # React-style hooks
│   ├── router/         # Client-side routing
│   ├── dom/            # Virtual DOM implementation
│   ├── state/          # Actor-based state management
│   └── platform/       # Platform bindings (browser, SSR)
├── docs/               # Framework documentation
├── examples/           # Example applications
└── tests/              # Test suite
```

## Installation

```bash
# Add to your Sigil project
sigil add qliphoth

# Or clone for development
git clone https://github.com/daemoniorum/qliphoth
cd qliphoth && sigil build
```

## Documentation

- [Getting Started Guide](docs/guides/getting-started.md)
- [Component API](docs/api/components.md)
- [Hooks Reference](docs/api/hooks.md)
- [Router Guide](docs/guides/routing.md)
- [State Management](docs/guides/state.md)

## Examples

- [Counter](examples/counter.sigil) - Simple state management
- [Todo App](examples/todo.sigil) - CRUD operations
- [Docs Platform](examples/docs-platform/) - Full documentation site

## License

Copyright © 2025 Daemoniorum, LLC. All rights reserved.

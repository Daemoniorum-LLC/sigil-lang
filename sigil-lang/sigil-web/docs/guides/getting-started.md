# Getting Started with Sigil Web

Sigil Web is a React-inspired web application framework built on Sigil's polysynthetic programming paradigm. This guide will help you create your first Sigil Web application.

## Prerequisites

- Sigil compiler (v0.1.0 or later)
- Basic familiarity with Sigil syntax
- A text editor or IDE (VS Code with Sigil extension recommended)

## Installation

```bash
# Install Sigil (if not already installed)
curl -fsSL https://sigil-lang.org/install | sh

# Create a new Sigil Web project
sigil new --template web my-app
cd my-app

# Install dependencies
sigil deps install

# Start development server
sigil web dev
```

## Project Structure

A new Sigil Web project has the following structure:

```
my-app/
├── src/
│   ├── main.sigil      # Application entry point
│   ├── components/     # Reusable components
│   ├── pages/          # Page components
│   └── styles/         # CSS stylesheets
├── public/             # Static assets
├── sigil.toml          # Project configuration
└── index.html          # HTML template
```

## Your First Component

Create a simple counter component in `src/main.sigil`:

```sigil
use sigil_web::prelude::*

component Counter {
    state count: i64! = 0

    fn render(self) -> VNode {
        div {
            h1 { "Count: {self.count}" }
            button[onclick: || self.count += 1] { "Increment" }
            button[onclick: || self.count -= 1] { "Decrement" }
        }
    }
}

fn main() {
    App::mount("#root", Counter::new())
}
```

## Understanding Evidentiality

Sigil Web uses Sigil's evidentiality system to track data provenance:

| Marker | Meaning | Example |
|--------|---------|---------|
| `!` | Known/Computed | `state count: i64! = 0` |
| `?` | Uncertain | `props title: Option<String>?` |
| `~` | Remote/External | `let user~ = fetch_user()` |
| `‽` | Paradox/Untrusted | `let input‽ = form_data()` |

This helps you understand where your data comes from and its reliability.

## Using Hooks

Sigil Web provides React-style hooks for functional components:

```sigil
fn Timer() -> VNode {
    let (seconds, set_seconds) = use_state(0)

    use_effect(|| {
        let timer_id = set_interval(1000, || {
            set_seconds(seconds + 1)
        })

        // Cleanup function
        Some(|| clear_interval(timer_id))
    }, [])

    span { "Elapsed: {seconds}s" }
}
```

### Available Hooks

- `use_state` - Local component state
- `use_reducer` - Complex state logic
- `use_effect` - Side effects
- `use_memo` - Memoized computations
- `use_callback` - Memoized callbacks
- `use_ref` - Mutable references
- `use_context` - Context consumption
- `use_fetch` - Data fetching
- `use_router` - Routing access

## Routing

Set up client-side routing with the Router component:

```sigil
use sigil_web::prelude::*

fn App() -> VNode {
    Router {
        Route[path: "/"] { Home {} }
        Route[path: "/about"] { About {} }
        Route[path: "/users/:id"] { |props|
            UserProfile { id: props.params["id"] }
        }
        Route[path: "*"] { NotFound {} }
    }
}
```

## State Management

For global state, use the actor-based Store:

```sigil
use sigil_web::prelude::*

struct AppState {
    count: i64!
    user: Option<User>?
}

enum AppAction {
    Increment,
    SetUser(User)
}

fn reducer(state: AppState, action: AppAction) -> AppState {
    match action {
        Increment => AppState { count: state.count + 1, ..state },
        SetUser(user) => AppState { user: Some(user), ..state }
    }
}

static APP_STORE: Store<AppState, AppAction> = Store::new(
    AppState { count: 0, user: None },
    reducer
)

// In components:
fn Counter() -> VNode {
    let count = use_selector(&APP_STORE, |s| s.count)
    let dispatch = use_dispatch(&APP_STORE)

    button[onclick: || dispatch(Increment)] {
        "Count: {count}"
    }
}
```

## Styling

Sigil Web supports multiple styling approaches:

### CSS Classes

```sigil
div()
    ·class("container flex gap-4")
    ·build()
```

### Conditional Classes

```sigil
div()
    ·class(classes()
        ·add("button")
        ·add_if(is_primary, "button--primary")
        ·add_if(is_disabled, "button--disabled")
        ·to_string())
    ·build()
```

### Inline Styles

```sigil
div()
    ·style(style()
        ·display("flex")
        ·gap("1rem")
        ·background_color("#f0f0f0")
        ·to_string())
    ·build()
```

## Building for Production

```bash
# Build optimized production bundle
sigil web build

# Output is in dist/ directory
ls dist/
```

## Next Steps

- [Components Guide](./components.md) - Learn about component patterns
- [Hooks Reference](../api/hooks.md) - Complete hooks documentation
- [Routing Guide](./routing.md) - Advanced routing patterns
- [State Management](./state.md) - Global state patterns
- [Examples](../../examples/) - Working example projects

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

| Marker | Name | Meaning | Example |
|--------|------|---------|---------|
| `!` | Known | Locally computed/verified | `state count: i64! = 0` |
| `?` | Uncertain | May be absent | `props title: Option<String>?` |
| `~` | Reported | External/untrusted | `let user~ = fetch_user()⌛` |
| `‽` | Paradox | Trust boundary crossing | `let input‽ = form_data()` |

This helps you understand where your data comes from and its reliability.

## Morpheme Operators

Sigil Web uses Greek-letter operators for data transformation:

| Operator | Name | Function | Example |
|----------|------|----------|---------|
| `τ` | Tau | Map/transform | `items\|τ{i => i * 2}` |
| `φ` | Phi | Filter | `items\|φ{i => i > 0}` |
| `σ` | Sigma | Sort | `items\|σ` |
| `ρ` | Rho | Reduce | `items\|ρ+` |
| `α` | Alpha | First | `items\|α` |
| `Ω` | Omega | Last | `items\|Ω` |

```sigil
// Pipeline example
let results! = users
    |φ{u => u.active}     // filter active users
    |τ{u => u.score}      // extract scores
    |σ                    // sort ascending
    |ρ+                   // sum all

// Parallel operations for large datasets
let processed! = large_list|par·τ{expensive_fn}|collect()
```

## Async Patterns

Use `⌛` (hourglass) for async/await operations:

```sigil
// Single await
let data~ = fetch("/api/users")⌛

// Parallel fetches with await·all
let (users~, posts~) = (fetch_users(), fetch_posts())·await·all

// Race for fastest response
let fastest~ = mirrors|τ{url => fetch(url)}·await·race
```

## Using Hooks

Sigil Web provides React-style hooks for functional components:

```sigil
fn Timer() -> VNode! {
    let (seconds!, set_seconds) = use_state(0)

    use_effect(|| {
        let timer_id! = set_interval(1000, || {
            set_seconds(seconds + 1)
        })

        // Cleanup function
        Some(|| clear_interval(timer_id))
    }, [])

    span { "Elapsed: {seconds}s" }
}
```

### Available Hooks

**State Hooks:**
- `use_state` - Local component state
- `use_reducer` - Complex state logic with reducer

**Effect Hooks:**
- `use_effect` - Side effects after render
- `use_layout_effect` - Synchronous effects

**Performance Hooks:**
- `use_memo` - Memoized computations
- `use_callback` - Memoized callbacks

**Data Hooks:**
- `use_fetch` - Fetch data from URLs
- `use_fetch_all` - Parallel fetches with `await·all`
- `use_fetch_race` - Race fetches with `await·race`
- `use_mutation` - Handle data mutations

**Animation Hooks:**
- `use_spring` - Spring-based animations
- `use_transition` - Enter/exit transitions
- `use_animate` - Keyframe animations

**Form Hooks:**
- `use_form` - Form state and validation
- `use_field` - Individual field management

## Routing

Set up client-side routing with the Router component:

```sigil
use sigil_web::prelude::*

fn App() -> VNode! {
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

For global state, use the actor-based Store with `tell` messaging:

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

fn reducer(state: AppState, action: AppAction) -> AppState! {
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
fn Counter() -> VNode! {
    let count! = use_selector(&APP_STORE, |s| s.count)
    let dispatch = use_dispatch(&APP_STORE)

    button[onclick: || dispatch(Increment)] {
        "Count: {count}"
    }
}
```

## Channels for Real-Time

Use `voice`/`listen` primitives for message passing:

```sigil
use sigil_web::prelude::*

fn ChatComponent() -> VNode! {
    // Create a channel
    let (voice, listen) = channel::<String>()

    // Send messages with voice
    let send_message = |msg: String| {
        voice·voice(msg)
    }

    // Receive messages with listen
    use_effect(|| {
        loop {
            match listen·try_listen() {
                Ok(msg~) => handle_message(msg~),
                Err(_) => break
            }
        }
        None
    }, [])

    // ... render chat UI
}
```

## Quantifiers

Use `∀` (for all) and `∃` (exists) for collection predicates:

```sigil
// Check if ALL items satisfy condition
let all_valid! = items|∀{item => item.is_valid()}

// Check if ANY item satisfies condition
let has_errors! = items|∃{item => item.has_error()}

// Combined with other operators
let all_active_verified! = users
    |φ{u => u.active}
    |∀{u => u.verified}
```

## Form Validation

Use the forms module for schema-based validation:

```sigil
use sigil_web::prelude::*
use sigil_web::forms::validators::*

fn LoginForm() -> VNode! {
    let email = use_field("email", vec![required(), email()])
    let password = use_field("password", vec![required(), min_length(8)])

    form()
        ·onsubmit(|| {
            if email.error·is_none() ∧ password.error·is_none() {
                submit_login(email.value, password.value)
            }
        })
        ·children(vec![
            FormField {
                name: "email",
                label: "Email",
                error: email.error,
                ..email.props
            }·to_vnode(),

            FormField {
                name: "password",
                label: "Password",
                field_type: "password",
                error: password.error,
                ..password.props
            }·to_vnode(),

            button[disabled: email.error·is_some() ∨ password.error·is_some()] {
                "Login"
            }
        ])
        ·build()
}
```

## Animations

Use animation hooks for smooth transitions:

```sigil
use sigil_web::prelude::*

fn AnimatedBox() -> VNode! {
    let spring = use_spring(0.0, SpringConfig::default())

    div()
        ·style(style()
            ·transform(&format!("translateX({}px)", spring.value))
            ·to_string())
        ·onmouseenter(|| spring.set(100.0))
        ·onmouseleave(|| spring.set(0.0))
        ·text("Hover me!")
        ·build()
}

fn Modal(props: ModalProps) -> VNode! {
    let transition = use_transition(props.is_open, TransitionConfig::default())

    when(transition.should_render, || {
        div()
            ·class("modal")
            ·style(style()
                ·opacity(&transition.progress·to_string())
                ·to_string())
            ·children(props.children)
            ·build()
    })
}
```

## Styling

Sigil Web supports multiple styling approaches:

### CSS Classes with Set Operations

```sigil
div()
    ·class(classes()
        ·add("button")
        ·add_if(is_primary, "button--primary")
        ·add_unless(enabled, "button--disabled")  // Using ¬
        ·union(&base_classes)                     // Using ∪
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
  - [Counter](../../examples/basic/counter.sigil) - Basic state management
  - [Todo App](../../examples/basic/todo.sigil) - Full CRUD operations
  - [Chat App](../../examples/advanced/chat.sigil) - Real-time with channels

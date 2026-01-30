# Qliphoth Framework Cheatsheet

React-inspired web framework for Sigil. Corporate Goth aesthetic, zero JavaScript.

## Quick Start

```sigil
use qliphoth::prelude::*

component Counter {
    state count: i64! = 0

    rite render(this) → Element {
        div {
            h1 { "Count: {this.count}" }
            button[onclick: || this.count += 1] { "+" }
            button[onclick: || this.count -= 1] { "-" }
        }
    }
}

rite main() {
    App::mount("#root", Counter::new())
}
```

## Component Patterns

### Class-Style Component

```sigil
component MyComponent {
    // State declarations
    state value: i32! = 0
    state items: Vec<Item>! = vec![]

    // Render method
    rite render(this) → Element {
        div {
            // JSX-like syntax
        }
    }
}
```

### Functional Component (Alternate Syntax)

```sigil
#[component]
rite MyComponent(props: Props!) → Element! {
    ≔ count! = use_state(0)

    div {
        "Hello {props.name}"
    }
}
```

## Hooks

### State Management

```sigil
// Basic state
≔ count! = use_state(0)
≔ (value!, set_value!) = use_state_with(|| compute_initial())

// Reducer pattern
≔ (state!, dispatch!) = use_reducer(reducer, initial_state)
```

### Effects

```sigil
// Side effects (runs after render)
use_effect(|| {
    // Effect logic
    subscribe_to_events()

    // Cleanup function
    || unsubscribe()
})

// Layout effects (runs synchronously after DOM mutations)
use_layout_effect(|| {
    measure_element()
    || {}
})
```

### Memoization

```sigil
// Memoized value
≔ expensive! = use_memo(|| {
    compute_expensive_value(deps)
}, [deps])

// Memoized callback
≔ handler! = use_callback(|event| {
    handle_event(event)
}, [deps])
```

### Refs

```sigil
≔ input_ref! = use_ref(None)

// Access: input_ref.current
```

### Data Fetching

```sigil
// Fetch data
≔ data~ = use_fetch("/api/data")

// With options
≔ result~ = use_fetch_with("/api/data", FetchOptions {
    method: "POST",
    body: json_body
})

// Mutations
≔ (mutate!, status!) = use_mutation(|data| {
    post_data(data)
})
```

### Context

```sigil
// Create context
☉ const ThemeContext = create_context(default_theme)

// Provide context
<ThemeContext.Provider value={theme}>
    <App />
</ThemeContext.Provider>

// Consume context
≔ theme! = use_context(ThemeContext)
```

## DOM Building

### Element Builders

```sigil
use qliphoth::dom::*

// Structural
div { ... }
section { ... }
article { ... }
header { ... }
footer { ... }
nav { ... }
main_elem { ... }
aside { ... }
span { ... }

// Text
h1 { "Heading" }
h2 { ... } h3 { ... } h4 { ... } h5 { ... } h6 { ... }
p { "Paragraph" }
pre { code }
code { "inline code" }
blockquote { ... }
em { "emphasis" }
strong { "bold" }
small { ... }

// Links & Media
a[href: "/path"] { "Link" }
img[src: "/img.png", alt: "Description"]
video { ... }
audio { ... }
svg { ... }

// Lists
ul {
    li { "Item 1" }
    li { "Item 2" }
}
ol { ... }

// Tables
table {
    thead { tr { th { "Header" } } }
    tbody { tr { td { "Cell" } } }
}

// Forms
form {
    label[for: "email"] { "Email" }
    input[type: "email", id: "email", placeholder: "you@example.com"]
    textarea { }
    select {
        option[value: "a"] { "Option A" }
    }
    button[type: "submit"] { "Submit" }
}

// Helpers
br
hr
fragment { ... }  // React.Fragment equivalent
text("Raw text")
```

### Attributes & Events

```sigil
// Attributes with []
div[class: "container", id: "main"] { ... }

// Event handlers
button[onclick: || handle_click()] { "Click" }
input[oninput: |e| set_value(e.target.value)]
form[onsubmit: |e| { e.prevent_default(); submit() }]

// Conditional attributes
div[class: if active { "active" } else { "" }]
```

### Styling

```sigil
// Inline styles
div[style: "color: red; font-size: 16px"] { }

// Style object
div[style: Style {
    color: "#14A088",
    padding: "1rem",
    display: "flex"
}]

// Classes helper
div[class: classes("base", if active { "active" } else { "" })]
```

### Conditional Rendering

```sigil
// When helper
when(show_content, || {
    div { "Visible content" }
})

// If-else
if_else(condition,
    || div { "True branch" },
    || div { "False branch" }
)

// Match in JSX
{match status {
    Loading => spinner(),
    Ready(data) => content(data),
    Error(e) => error_display(e)
}}
```

### List Rendering

```sigil
// Map to elements
ul {
    map_to_elements(items, |item| {
        li[key: item.id] { item.name }
    })
}

// With morpheme operators
ul {
    items
        |φ{.visible}
        |τ{|item| li[key: item.id] { item.name }}
}
```

## Router

### Setup

```sigil
use qliphoth::router::*

rite App() → Element {
    Router {
        Route[path: "/", component: Home]
        Route[path: "/about", component: About]
        Route[path: "/users/:id", component: UserProfile]
        Route[path: "*", component: NotFound]
    }
}
```

### Navigation

```sigil
// Link component
Link[to: "/about"] { "About" }

// NavLink (active styling)
NavLink[to: "/", active_class: "active"] { "Home" }

// Programmatic navigation
≔ navigate! = use_navigate()
navigate("/new-path")

// Redirect
Redirect[to: "/login"]
```

### Route Params & Query

```sigil
// Access params
≔ params! = use_params()
≔ user_id! = params.get("id")

// Access query string
≔ query! = use_query()
≔ search! = query.get("q")

// Current location
≔ location! = use_location()
```

### Protected Routes

```sigil
ProtectedRoute[
    path: "/dashboard",
    component: Dashboard,
    guard: || is_authenticated(),
    fallback: "/login"
]
```

## State Management (Advanced)

### Signals (Fine-grained Reactivity)

```sigil
use qliphoth::state::*

// Create signal
≔ count! = signal(0)

// Read value
≔ current! = count.get()

// Update
count.set(10)
count.update(|c| c + 1)

// Computed (derived state)
≔ doubled! = computed(|| count.get() * 2)

// Effects
effect(|| {
    println("Count changed: {count.get()}")
})
```

### Atoms (Global State)

```sigil
// Define atom
☉ const count_atom = Atom::new(0)

// Use in component
≔ count! = use_atom_value(count_atom)
≔ set_count! = use_set_atom(count_atom)
```

### Store (Redux-like)

```sigil
// Define store
☉ sigil AppState {
    count: i32!,
    user: Option<User>?
}

☉ sigil Action = enum {
    Increment,
    Decrement,
    SetUser(User)
}

rite reducer(state: AppState!, action: Action!) → AppState! {
    match action {
        Increment => AppState { count: state.count + 1, ..state },
        Decrement => AppState { count: state.count - 1, ..state },
        SetUser(user) => AppState { user: Some(user), ..state }
    }
}

// Use in component
≔ state! = use_selector(|s| s.count)
≔ dispatch! = use_dispatch()
dispatch(Action::Increment)
```

## Components Library

### Error Boundary

```sigil
ErrorBoundary[fallback: |error| error_ui(error)] {
    <RiskyComponent />
}
```

### Suspense

```sigil
Suspense[fallback: <LoadingSpinner />] {
    <AsyncComponent />
}
```

### Portal

```sigil
Portal[target: "#modal-root"] {
    <Modal />
}
```

## Accessibility

```sigil
use qliphoth::a11y::*

// Skip link
SkipLink[href: "#main"] { "Skip to content" }

// Visually hidden (screen reader only)
VisuallyHidden { "Description for screen readers" }

// Live region for announcements
LiveRegion[politeness: Politeness::Polite] {
    status_message
}

// Focus management
≔ trap! = use_focus_trap(options)
≔ reduced_motion! = use_reduced_motion()
```

## Animation

```sigil
use qliphoth::animation::*

// Spring animation
≔ spring! = use_spring(0.0, SpringPreset::Gentle)

// Animated component
Animated[
    initial: AnimatedStyle { opacity: 0.0 },
    animate: AnimatedStyle { opacity: 1.0 },
    transition: Transition { duration: 300 }
] {
    <Content />
}

// Presence animation
AnimatePresence {
    when(show, || {
        Fade { <Modal /> }
    })
}
```

## Project Structure

```
website-qliphoth/
├── Sigil.toml           # Project manifest
├── src/
│   ├── lib.sigil        # Entry point, App component
│   ├── components.sigil # Reusable components
│   ├── pages.sigil      # Page components
│   └── theme.sigil      # Theme/styling
├── dist/
│   ├── index.html       # HTML shell
│   ├── site.wasm        # Compiled WASM
│   └── styles.css       # Global styles
└── tests/
    └── website.spec.ts  # E2E tests
```

## Build & Deploy

```bash
# Compile to WASM
sigil wasm src/lib.sigil -o dist/site.wasm

# Development server
python3 -m http.server 5181 --directory dist

# Run tests
npx playwright test
```

---

*Last updated: 2026-01-21*
*Source: qliphoth/src/lib.sigil, sigil-web documentation*

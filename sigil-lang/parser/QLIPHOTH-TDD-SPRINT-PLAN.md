# Qliphoth TDD Sprint Plan

> **"Test first. Ship fast. No exceptions."**

This is the actionable sprint plan for completing Qliphoth. Each sprint follows strict TDD: write failing tests, implement to pass, refactor.

---

## Current Status (2026-01-16)

**Completed:**
- [x] All 11 qliphoth-ui files compile to WASM
- [x] String interpolation works with `f"..."` syntax
- [x] sigil-web runtime designed (signals, VDOM, hooks)

**In Progress:**
- [ ] Make sigil-web WASM-compatible

**Blockers:**
- sigil-web uses syntax not yet supported by WASM backend
- web_sys bindings don't exist

---

## Sprint Overview

| Sprint | Focus | Duration | Blockers |
|--------|-------|----------|----------|
| **S0** | String Interpolation | ~~2-3 days~~ | **DONE** |
| **S1** | sigil-web WASM Compat | 3-5 days | None |
| **S2** | Component Rendering | 1 week | S1 |
| **S3** | Interactive Components | 1 week | S2 |
| **S4** | qliphoth-app MVP | 1 week | S3 |
| **S5** | qliphoth-docs Portal | 1 week | S3 |
| **S6** | qliphoth-chat Integration | 1 week | S4 |
| **S7** | Polish & Production | 1 week | S6 |

**Total:** ~7 weeks to production

---

## Sprint S0: String Interpolation (CRITICAL)

**Goal:** Enable `"Hello, {name}!"` syntax in WASM compilation.

**Why Critical:** Nearly every qliphoth component needs dynamic text rendering. This is the #1 blocker.

### S0.1 Tests First

Create test file: `jormungandr/tests/spec/08_wasm/interpolation.sigil`

```sigil
//! String interpolation WASM tests

// Test 1: Basic variable interpolation
fn test_basic_interpolation() {
    let name! = "World";
    let msg! = "Hello, {name}!";
    assert_eq!(msg, "Hello, World!");
}

// Test 2: Expression interpolation
fn test_expression_interpolation() {
    let x! = 5;
    let msg! = "Value: {x + 1}";
    assert_eq!(msg, "Value: 6");
}

// Test 3: Multiple interpolations
fn test_multiple_interpolation() {
    let first! = "John";
    let last! = "Doe";
    let msg! = "{first} {last}";
    assert_eq!(msg, "John Doe");
}

// Test 4: Nested expressions
fn test_nested_expression() {
    let items! = vec![1, 2, 3];
    let msg! = "Count: {items·len()}";
    assert_eq!(msg, "Count: 3");
}

// Test 5: Escape braces
fn test_escape_braces() {
    let msg! = "Literal: {{not interpolated}}";
    assert_eq!(msg, "Literal: {not interpolated}");
}

// Test 6: Empty interpolation (edge case)
fn test_empty_string_interpolation() {
    let empty! = "";
    let msg! = "Start{empty}End";
    assert_eq!(msg, "StartEnd");
}

// Test 7: Integer interpolation
fn test_int_interpolation() {
    let n! = 42;
    let msg! = "The answer is {n}";
    assert_eq!(msg, "The answer is 42");
}

// Test 8: Float interpolation
fn test_float_interpolation() {
    let pi! = 3.14159;
    let msg! = "Pi is approximately {pi}";
    assert!(msg·starts_with("Pi is approximately 3.14"));
}

fn main() {
    test_basic_interpolation();
    test_expression_interpolation();
    test_multiple_interpolation();
    test_nested_expression();
    test_escape_braces();
    test_empty_string_interpolation();
    test_int_interpolation();
    test_float_interpolation();
    print("All interpolation tests passed!");
}
```

### S0.2 Implementation Plan

**File:** `src/wasm/literals.rs` (new) or extend `src/wasm/expressions.rs`

**Algorithm:**
1. During string literal parsing, detect `{` and `}`
2. Parse content between braces as expression
3. Compile to:
   - Push string segment before `{`
   - Compile expression
   - Call `string::from_int` or `string::from_float` for numeric types
   - Call `string::concat` to join
   - Repeat for each segment
4. Handle `{{` and `}}` as escape sequences

**WASM Output Pattern:**
```wasm
;; "Hello, {name}!" compiles to:
(call $string_concat
  (i64.const <"Hello, " offset>)
  (call $string_concat
    (local.get $name)
    (i64.const <"!" offset>)))
```

### S0.3 Success Criteria

- [ ] All 8 interpolation tests pass with `sigil wasm` + browser runtime
- [ ] qliphoth-ui components can render dynamic text
- [ ] No performance regression (< 5% compile time increase)

---

## Sprint S1: sigil-web Core Runtime

**Goal:** Create the foundational web runtime with DOM bindings, signals, and router.

### S1.1 Crate Structure

```
sigil/sigil-lang/sigil-web/
├── Cargo.toml              # Rust build for tests
├── Sigil.toml              # Sigil manifest
├── src/
│   ├── lib.sigil           # Public exports
│   ├── dom.sigil           # DOM element bindings
│   ├── events.sigil        # Event handling
│   ├── signals.sigil       # Reactive state
│   ├── router.sigil        # Client-side routing
│   └── fetch.sigil         # HTTP client
└── tests/
    ├── dom_test.sigil
    ├── signals_test.sigil
    └── router_test.sigil
```

### S1.2 DOM Bindings (2 days)

**Tests First:** `sigil-web/tests/dom_test.sigil`

```sigil
//! DOM binding tests

fn test_element_create() {
    let div! = Element::new("div");
    assert_eq!(div·tag_name(), "DIV");
}

fn test_element_set_attribute() {
    let div! = Element::new("div");
    div·set_attr("id", "test");
    assert_eq!(div·get_attr("id")·unwrap(), "test");
}

fn test_element_class_manipulation() {
    let div! = Element::new("div");
    div·add_class("active");
    assert!(div·has_class("active"));
    div·remove_class("active");
    assert!(!div·has_class("active"));
}

fn test_element_append_child() {
    let parent! = Element::new("div");
    let child! = Element::new("span");
    parent·append(child);
    assert_eq!(parent·children()·len(), 1);
}

fn test_element_text_content() {
    let div! = Element::new("div");
    div·set_text("Hello World");
    assert_eq!(div·text(), "Hello World");
}

fn test_element_style() {
    let div! = Element::new("div");
    div·set_style("color", "red");
    assert_eq!(div·get_style("color"), "red");
}

fn test_query_selector() {
    let container! = Element::new("div");
    let child! = Element::new("span");
    child·add_class("target");
    container·append(child);

    let found? = container·query(".target");
    assert!(found·is_some());
}

fn main() {
    test_element_create();
    test_element_set_attribute();
    test_element_class_manipulation();
    test_element_append_child();
    test_element_text_content();
    test_element_style();
    test_query_selector();
    print("All DOM tests passed!");
}
```

**Implementation:** Thin wrappers around existing JS imports in `runtime.mjs`

### S1.3 Signals (2 days)

**Tests First:** `sigil-web/tests/signals_test.sigil`

```sigil
//! Reactive signal tests

fn test_signal_create_get() {
    let count! = Signal::new(0);
    assert_eq!(count·get(), 0);
}

fn test_signal_set() {
    let count! = Signal::new(0);
    count·set(5);
    assert_eq!(count·get(), 5);
}

fn test_signal_update() {
    let count! = Signal::new(0);
    count·update(|τ{n => n + 1}|);
    assert_eq!(count·get(), 1);
}

fn test_computed_derives() {
    let count! = Signal::new(2);
    let doubled! = computed(|τ{() => count·get() * 2}|);
    assert_eq!(doubled·get(), 4);

    count·set(5);
    assert_eq!(doubled·get(), 10);
}

fn test_effect_runs_on_change() {
    let count! = Signal::new(0);
    let runs! = Signal::new(0);

    effect(|τ{() => {
        let _ = count·get();
        runs·update(|τ{n => n + 1}|);
    }}|);

    assert_eq!(runs·get(), 1);  // Initial run
    count·set(1);
    assert_eq!(runs·get(), 2);  // Triggered by change
}

fn test_batch_updates() {
    let a! = Signal::new(0);
    let b! = Signal::new(0);
    let runs! = Signal::new(0);

    effect(|τ{() => {
        let _ = a·get();
        let _ = b·get();
        runs·update(|τ{n => n + 1}|);
    }}|);

    batch(|τ{() => {
        a·set(1);
        b·set(2);
    }}|);

    assert_eq!(runs·get(), 2);  // Only 1 additional run, not 2
}

fn main() {
    test_signal_create_get();
    test_signal_set();
    test_signal_update();
    test_computed_derives();
    test_effect_runs_on_change();
    test_batch_updates();
    print("All signal tests passed!");
}
```

**Implementation:** Leverage existing `signal::*` imports in WASM runtime

### S1.4 Router (1 day)

**Tests First:** `sigil-web/tests/router_test.sigil`

```sigil
//! Client-side router tests

fn test_route_match_static() {
    let route! = Route::new("/about");
    assert!(route·matches("/about"));
    assert!(!route·matches("/contact"));
}

fn test_route_match_param() {
    let route! = Route::new("/users/:id");
    assert!(route·matches("/users/123"));
    assert!(!route·matches("/users"));
}

fn test_route_extract_params() {
    let route! = Route::new("/users/:id");
    let params! = route·params("/users/42");
    assert_eq!(params·get("id")·unwrap(), "42");
}

fn test_router_navigate() {
    let router! = Router::new(vec![
        Route::new("/"),
        Route::new("/about"),
    ]);

    router·navigate("/about");
    assert_eq!(router·current_path(), "/about");
}

fn test_router_back() {
    let router! = Router::new(vec![
        Route::new("/"),
        Route::new("/about"),
        Route::new("/contact"),
    ]);

    router·navigate("/about");
    router·navigate("/contact");
    router·back();

    assert_eq!(router·current_path(), "/about");
}

fn main() {
    test_route_match_static();
    test_route_match_param();
    test_route_extract_params();
    test_router_navigate();
    test_router_back();
    print("All router tests passed!");
}
```

### S1.5 Success Criteria

- [ ] `sigil-web` crate compiles to WASM
- [ ] All DOM tests pass in browser
- [ ] All signal tests pass (reactivity works)
- [ ] All router tests pass (navigation works)
- [ ] Integration test: simple counter app renders and updates

---

## Sprint S2: Component Rendering

**Goal:** Render qliphoth-ui components in browser.

### S2.1 html! Macro Validation

**Tests First:** `sigil-web/tests/html_test.sigil`

```sigil
//! html! macro rendering tests

fn test_html_div() {
    let node! = html! {
        <div class="container">Hello</div>
    };
    assert_eq!(node·tag(), "div");
    assert!(node·has_class("container"));
}

fn test_html_nested() {
    let node! = html! {
        <div>
            <span>Child 1</span>
            <span>Child 2</span>
        </div>
    };
    assert_eq!(node·children()·len(), 2);
}

fn test_html_dynamic_content() {
    let name! = "World";
    let node! = html! {
        <div>Hello, {name}!</div>
    };
    assert!(node·text()·contains("World"));
}

fn test_html_event_handler() {
    let clicked! = Signal::new(false);
    let node! = html! {
        <button onclick={|τ{_ => clicked·set(true)}|}>Click</button>
    };

    // Simulate click
    node·dispatch_event("click");
    assert!(clicked·get());
}

fn test_html_conditional() {
    let show! = true;
    let node! = html! {
        <div>
            {if show { html!{<span>Visible</span>} } else { html!{<span></span>} }}
        </div>
    };
    assert!(node·inner_html()·contains("Visible"));
}

fn main() {
    test_html_div();
    test_html_nested();
    test_html_dynamic_content();
    test_html_event_handler();
    test_html_conditional();
    print("All html! tests passed!");
}
```

### S2.2 Component Trait

**Tests First:** `qliphoth-ui/tests/component_test.sigil`

```sigil
//! Component trait tests

fn test_button_renders() {
    let btn! = Button::new()
        ·variant(ButtonVariant::Primary)
        ·label("Click Me");

    let node! = btn·render();
    assert_eq!(node·tag(), "button");
    assert!(node·has_class("btn-primary"));
}

fn test_button_onclick() {
    let clicked! = Signal::new(false);
    let btn! = Button::new()
        ·label("Test")
        ·onclick(|τ{_ => clicked·set(true)}|);

    let node! = btn·render();
    node·dispatch_event("click");
    assert!(clicked·get());
}

fn test_input_value() {
    let value! = Signal::new("initial");
    let input! = Input::new()
        ·value(value·get())
        ·on_change(|τ{v => value·set(v)}|);

    let node! = input·render();
    // Simulate input
    node·set_value("updated");
    node·dispatch_event("input");
    assert_eq!(value·get(), "updated");
}

fn test_card_composition() {
    let card! = Card::new()
        ·header(html!{<h3>Title</h3>})
        ·body(html!{<p>Content</p>})
        ·footer(html!{<button>Action</button>});

    let node! = card·render();
    assert!(node·query(".card-header")·is_some());
    assert!(node·query(".card-body")·is_some());
    assert!(node·query(".card-footer")·is_some());
}

fn main() {
    test_button_renders();
    test_button_onclick();
    test_input_value();
    test_card_composition();
    print("All component tests passed!");
}
```

### S2.3 Design Token Integration

**Tests First:** `qliphoth-ui/tests/tokens_test.sigil`

```sigil
//! Design token tests

fn test_colors_exist() {
    assert_eq!(colors::VOID, "#0a0a0a");
    assert_eq!(colors::PHTHALO, "#123524");
    assert_eq!(colors::CRIMSON, "#8b0000");
}

fn test_typography_tokens() {
    assert_eq!(typography::FONT_SANS·contains("Inter"), true);
    assert_eq!(typography::SIZE_BASE, "1rem");
}

fn test_spacing_tokens() {
    assert_eq!(spacing::SM, "0.5rem");
    assert_eq!(spacing::MD, "1rem");
    assert_eq!(spacing::LG, "1.5rem");
}

fn test_token_application() {
    let div! = Element::new("div");
    div·set_style("background-color", colors::VOID);
    div·set_style("font-family", typography::FONT_SANS);

    assert_eq!(div·get_style("background-color"), colors::VOID);
}

fn main() {
    test_colors_exist();
    test_typography_tokens();
    test_spacing_tokens();
    test_token_application();
    print("All token tests passed!");
}
```

### S2.4 Success Criteria

- [ ] html! macro compiles and renders correctly
- [ ] Button, Input, Card components render
- [ ] Design tokens apply correctly
- [ ] Event handlers fire
- [ ] Integration: Counter component increments on click

---

## Sprint S3: Interactive Components

**Goal:** Full interactivity with forms, modals, and state management.

### S3.1 Form Components

```sigil
//! Form component tests

fn test_form_submit() {
    let submitted! = Signal::new(false);
    let form! = Form::new()
        ·on_submit(|τ{e => {
            e·prevent_default();
            submitted·set(true);
        }}|)
        ·children(vec![
            Input::new()·name("email")·render(),
            Button::new()·type_("submit")·label("Submit")·render(),
        ]);

    let node! = form·render();
    node·dispatch_event("submit");
    assert!(submitted·get());
}

fn test_select_options() {
    let selected! = Signal::new("");
    let select! = Select::new()
        ·options(vec![
            ("opt1", "Option 1"),
            ("opt2", "Option 2"),
        ])
        ·on_change(|τ{v => selected·set(v)}|);

    let node! = select·render();
    node·set_value("opt2");
    node·dispatch_event("change");
    assert_eq!(selected·get(), "opt2");
}

fn test_checkbox_toggle() {
    let checked! = Signal::new(false);
    let checkbox! = Checkbox::new()
        ·checked(checked·get())
        ·on_change(|τ{v => checked·set(v)}|);

    let node! = checkbox·render();
    node·click();
    assert!(checked·get());
}
```

### S3.2 Modal/Overlay Components

```sigil
//! Modal component tests

fn test_modal_open_close() {
    let open! = Signal::new(false);
    let modal! = Modal::new()
        ·is_open(open·get())
        ·on_close(|τ{() => open·set(false)}|)
        ·content(html!{<p>Modal content</p>});

    open·set(true);
    let node! = modal·render();

    assert!(node·query(".modal-backdrop")·is_some());

    // Click backdrop to close
    node·query(".modal-backdrop")·unwrap()·click();
    assert!(!open·get());
}

fn test_dropdown_toggle() {
    let open! = Signal::new(false);
    let dropdown! = Dropdown::new()
        ·trigger(html!{<button>Menu</button>})
        ·items(vec![
            ("item1", "Item 1"),
            ("item2", "Item 2"),
        ]);

    let node! = dropdown·render();
    node·query("button")·unwrap()·click();

    assert!(node·query(".dropdown-menu")·is_some());
}
```

### S3.3 State Management

```sigil
//! App state management tests

fn test_app_state_provider() {
    let state! = AppState::new(|τ{() => {
        count: Signal::new(0),
        user: Signal::new(None),
    }}|);

    let app! = StateProvider::new(state)
        ·children(vec![
            Counter::new()·render(),
        ]);

    // Children can access state
    let node! = app·render();
    assert!(node·query(".counter")·is_some());
}

fn test_global_state_update() {
    let store! = Store::new(|τ{state, action => {
        match action {
            Action::Increment => state.count + 1,
            Action::Decrement => state.count - 1,
        }
    }}|, State { count: 0 });

    store·dispatch(Action::Increment);
    assert_eq!(store·state()·count, 1);
}
```

### S3.4 Success Criteria

- [ ] Forms collect and submit data
- [ ] Modals open/close with transitions
- [ ] Dropdowns toggle correctly
- [ ] Global state updates propagate to components
- [ ] Integration: Todo app with add/remove/toggle

---

## Sprint S4: qliphoth-app MVP

**Goal:** Landing page with hero, features, and CTA.

### S4.1 Page Components

```sigil
//! qliphoth-app page tests

fn test_hero_section() {
    let hero! = HeroSection::new()
        ·title("Build with Sigil")
        ·subtitle("The AI-native programming language")
        ·cta(Button::new()·label("Get Started"));

    let node! = hero·render();
    assert!(node·query("h1")·is_some());
    assert!(node·query(".cta-button")·is_some());
}

fn test_feature_grid() {
    let features! = FeatureGrid::new()
        ·items(vec![
            Feature { icon: "code", title: "Type Safe", desc: "..." },
            Feature { icon: "bolt", title: "Fast", desc: "..." },
            Feature { icon: "shield", title: "Secure", desc: "..." },
        ]);

    let node! = features·render();
    assert_eq!(node·query_all(".feature-card")·len(), 3);
}

fn test_navigation() {
    let nav! = Navigation::new()
        ·logo(html!{<img src="logo.svg" />})
        ·links(vec![
            NavLink { href: "/", label: "Home" },
            NavLink { href: "/docs", label: "Docs" },
            NavLink { href: "/chat", label: "Chat" },
        ]);

    let node! = nav·render();
    assert_eq!(node·query_all("nav a")·len(), 3);
}
```

### S4.2 Integration Test

```sigil
//! Full app integration test

fn test_app_renders() {
    let app! = App::new();
    let node! = app·render();

    // Header present
    assert!(node·query("header")·is_some());

    // Hero section
    assert!(node·query(".hero")·is_some());

    // Feature grid
    assert!(node·query(".features")·is_some());

    // Footer
    assert!(node·query("footer")·is_some());
}

fn test_navigation_works() {
    let app! = App::new();
    mount(document·body(), app·render());

    // Navigate to docs
    router·navigate("/docs");

    // Docs page renders
    assert!(document·query(".docs-page")·is_some());
}
```

### S4.3 Success Criteria

- [ ] Landing page renders completely
- [ ] Navigation between pages works
- [ ] Responsive layout (mobile/desktop)
- [ ] Corporate Goth aesthetic applied
- [ ] < 100KB WASM bundle size

---

## Sprint S5: qliphoth-docs Portal

**Goal:** Documentation portal with search and syntax highlighting.

### S5.1 Documentation Components

```sigil
//! docs portal tests

fn test_docs_sidebar() {
    let sidebar! = DocsSidebar::new()
        ·sections(vec![
            Section { title: "Getting Started", pages: vec![...] },
            Section { title: "Reference", pages: vec![...] },
        ]);

    let node! = sidebar·render();
    assert!(node·query(".docs-sidebar")·is_some());
}

fn test_code_block_highlight() {
    let code! = CodeBlock::new()
        ·language("sigil")
        ·code("fn main() { print(\"Hello\"); }");

    let node! = code·render();
    assert!(node·query(".token-keyword")·is_some());  // fn highlighted
}

fn test_search_filters() {
    let search! = DocsSearch::new();
    search·set_query("router");

    let results! = search·results();
    assert!(results·len() > 0);
    assert!(results[0]·title·contains("Router"));
}
```

### S5.2 Success Criteria

- [ ] Sidebar navigation works
- [ ] Code blocks have syntax highlighting
- [ ] Search returns relevant results
- [ ] Markdown renders correctly
- [ ] Links between docs work

---

## Sprint S6: qliphoth-chat Integration

**Goal:** Chat widget with Infernum streaming.

### S6.1 Chat Protocol

```sigil
//! chat protocol tests

fn test_websocket_connect() {
    let ws! = InfernumClient::connect("ws://localhost:8081/ws");
    assert!(ws·is_connected());
}

fn test_send_message() {
    let client! = InfernumClient::connect("ws://localhost:8081/ws");
    let response! = Signal::new(None);

    client·on_message(|τ{msg => response·set(Some(msg))}|);
    client·send(ChatMessage { role: "user", content: "Hello" });

    // Wait for response
    await_signal(response, 5000);
    assert!(response·get()·is_some());
}

fn test_streaming_tokens() {
    let client! = InfernumClient::connect("ws://localhost:8081/ws");
    let tokens! = Signal::new(vec![]);

    client·on_delta(|τ{delta => {
        tokens·update(|τ{t => t·push(delta)}|);
    }}|);

    client·send(ChatMessage { role: "user", content: "Tell me a story" });

    // Should receive multiple deltas
    await_condition(|τ{() => tokens·get()·len() > 5}|, 10000);
    assert!(tokens·get()·len() > 5);
}
```

### S6.2 Chat Widget

```sigil
//! chat widget tests

fn test_chat_widget_renders() {
    let widget! = ChatWidget::new();
    let node! = widget·render();

    assert!(node·query(".chat-input")·is_some());
    assert!(node·query(".chat-messages")·is_some());
}

fn test_chat_send_message() {
    let widget! = ChatWidget::new();
    mount(document·body(), widget·render());

    let input! = document·query(".chat-input input")·unwrap();
    input·set_value("Hello");

    let button! = document·query(".chat-send-button")·unwrap();
    button·click();

    // Message appears in list
    assert!(document·query(".user-message")·is_some());
}
```

### S6.3 Success Criteria

- [ ] WebSocket connects to Infernum
- [ ] Messages send and receive
- [ ] Streaming tokens render incrementally
- [ ] Tool calls display with approval UI
- [ ] Error handling for disconnects

---

## Sprint S7: Polish & Production

**Goal:** Production readiness with performance and accessibility.

### S7.1 Performance

```sigil
//! performance tests

fn test_render_performance() {
    let start! = timing::now();

    for _ in 0..1000 {
        let btn! = Button::new()·label("Test");
        let _ = btn·render();
    }

    let elapsed! = timing::now() - start;
    assert!(elapsed < 1000.0);  // < 1ms per component
}

fn test_virtual_dom_diffing() {
    let old! = generate_large_tree(100);
    let new! = mutate_tree(old, 5);  // 5 changes

    let start! = timing::now();
    let patches! = vdom::diff(old, new);
    let elapsed! = timing::now() - start;

    assert!(patches·len() <= 10);  // Minimal patches
    assert!(elapsed < 50.0);  // < 50ms for 100 nodes
}
```

### S7.2 Accessibility

```sigil
//! accessibility tests

fn test_button_aria() {
    let btn! = Button::new()
        ·label("Submit")
        ·disabled(true);

    let node! = btn·render();
    assert_eq!(node·get_attr("aria-disabled"), Some("true"));
    assert_eq!(node·get_attr("role"), Some("button"));
}

fn test_modal_focus_trap() {
    let modal! = Modal::new()
        ·is_open(true)
        ·content(html!{
            <input id="first" />
            <input id="last" />
        });

    mount(document·body(), modal·render());

    // Focus should be trapped within modal
    let first! = document·query("#first")·unwrap();
    let last! = document·query("#last")·unwrap();

    last·focus();
    simulate_tab();
    assert_eq!(document·active_element()·id(), "first");
}

fn test_color_contrast() {
    // Corporate Goth palette must meet WCAG AA
    assert!(contrast_ratio(colors::CLOUD, colors::VOID) >= 4.5);
    assert!(contrast_ratio(colors::MIST, colors::VOID) >= 4.5);
}
```

### S7.3 Success Criteria

- [ ] Lighthouse score > 90 (performance, accessibility)
- [ ] Bundle size < 150KB gzipped
- [ ] All WCAG AA requirements met
- [ ] Works in Chrome, Firefox, Safari
- [ ] No console errors

---

## Test Execution

### Running Tests

```bash
# Run WASM tests with interpreter (fast)
./target/release/sigil run tests/interpolation_test.sigil

# Run WASM tests in browser
./target/release/sigil wasm tests/dom_test.sigil -o /tmp/test.wasm
# Then load in test harness HTML

# Full test suite
cd jormungandr/tests && ./run_tests_rust.sh --spec 08_wasm
```

### CI Integration

```yaml
# .github/workflows/qliphoth.yml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Build Sigil compiler
      run: cd sigil/sigil-lang/parser && cargo build --release
    - name: Run WASM tests
      run: ./target/release/sigil test sigil-web/tests/
    - name: Browser tests
      run: npx playwright test qliphoth/e2e/
```

---

## Definition of Done

Each sprint is complete when:

1. **All tests pass** - No exceptions
2. **Code reviewed** - PR merged to main
3. **Documentation updated** - API docs, examples
4. **No regressions** - Existing tests still pass
5. **Performance validated** - No degradation > 10%

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| String interpolation complex | Start with simple cases, iterate |
| Browser compatibility | Test early on all targets |
| Bundle size bloat | Monitor size each sprint |
| Signal performance | Profile and optimize in S3 |
| Infernum protocol changes | Version the protocol |

---

## Dependencies

```
S0 (String Interpolation)
 └─> S1 (sigil-web Core)
      └─> S2 (Component Rendering)
           ├─> S3 (Interactive Components)
           │    ├─> S4 (qliphoth-app)
           │    │    └─> S6 (Chat)
           │    └─> S5 (qliphoth-docs)
           └─> S7 (Polish) [after S4, S5, S6]
```

---

**Let's ship it.**

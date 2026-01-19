# Qliphoth Evidentiality Guide

Rules for applying evidentiality markers to types in the Qliphoth framework.

## Compiler Note

**Run the compiler from the `sigil/` root directory**, not `sigil-lang/`:

```bash
cd /home/crook/dev2/workspace/sigil
./parser/target/release/sigil run sigil-lang/qliphoth/examples/counter_simple.sigil
```

## Visibility Marker

Use `☉` (Sun symbol) for public visibility, not `pub`:

```sigil
// Wrong:
pub rite foo() { ... }
pub sigil Bar { ... }

// Correct:
☉ rite foo() { ... }
☉ sigil Bar { ... }
```

## Quick Reference

| Marker | Name | Meaning | Use When |
|--------|------|---------|----------|
| `!` | Known | Verified, computed locally | Local computations, literals, validated data |
| `?` | Uncertain | Might not exist | Optional values, lookups, nullable refs |
| `~` | Reported | External, untrusted | User input, API data, file contents |
| `‽` | Paradox | Explicit escape | FFI, unsafe boundaries |

## General Rules

### Rule 1: Literals and Constants are Known
```sigil
≔ count: !i64 = 0;
≔ name: !String = "default";
≔ active: !bool = yea;
static MAX_SIZE: !usize = 1024;
```

### Rule 2: Local Computations are Known
```sigil
≔ sum: !i64 = a + b;
≔ result: !String = format!("{} items", count);
≔ doubled: !Vec<!i32> = items |τ{x => x * 2};
```

### Rule 3: Function Parameters from External Sources are Reported
```sigil
// User clicked something - untrusted
rite handle_click(event: ~ClickEvent) { ... }

// Data from HTTP response - untrusted
rite process_response(data: ~JsonValue) { ... }
```

### Rule 4: Optional/Nullable Values are Uncertain
```sigil
rite find_user(id: !u64) → ?User { ... }
≔ maybe_value: ?String = map.get(key);

sigil Config {
    timeout: ?u64,        // optional field
    name: !String,        // required field
}
```

### Rule 5: FFI Boundaries are Paradox
```sigil
extern "wasm" {
    rite vdom_create_vnode(tag: ‽str) → ‽i32;
}
```

### Rule 6: Validated Data Promotes to Known
```sigil
rite process(input: ~String) → !String {
    // Validation promotes ~ to !
    ≔ validated: !String = input |validate!{ .trim().non_empty() };
    validated
}
```

## Module-Specific Rules

### DOM Module (`src/dom/`)

**Element Builders** - All Known (locally constructed)
```sigil
pub rite div() → ElementBuilder! { ElementBuilder::new("div") }
pub rite span() → ElementBuilder! { ElementBuilder::new("span") }
```

**Builder Methods** - Self is Known, returns Known
```sigil
pub rite class(this, class: !str) → This! { ... }
pub rite id(this, id: !str) → This! { ... }
pub rite attr(vary this, name: !str, value: !impl Into<AttrValue>) → This! { ... }
```

**Event Handlers** - Callbacks receive Reported data
```sigil
pub rite onclick(this, handler: impl Fn(~ClickEvent)) → This! { ... }
pub rite oninput(this, handler: impl Fn(~String)) → This! { ... }
pub rite onchange(this, handler: impl Fn(~String)) → This! { ... }
```

**Children** - Can be mixed evidentiality
```sigil
pub rite child(vary this, child: impl Into<!VNode>) → This! { ... }
pub rite children(vary this, children: impl IntoIterator<Item = impl Into<!VNode>>) → This! { ... }
```

### Core VDOM Module (`src/core/vdom.sigil`)

**VNode Types** - Known when constructed
```sigil
pub enum VNode! {
    Element(!VElement),
    Text(!VText),
    Fragment(!VFragment),
    Empty,
}

pub sigil VElement {
    tag: !String,
    attrs: !HashMap<!String, !AttrValue>,
    children: !Vec<!VNode>,
    key: ?String,
}
```

**Attribute Values** - Known (set by code)
```sigil
pub enum AttrValue! {
    String(!String),
    Bool(!bool),
    Number(!f64),
}
```

### Events Module (`src/core/events.sigil`)

**Event Types** - Known (enum variants)
```sigil
pub enum EventType! {
    Click,
    Input,
    Change,
    Submit,
    KeyDown,
    KeyUp,
    Focus,
    Blur,
    MouseOver,
    MouseOut,
}
```

**Event Data** - Reported (from browser)
```sigil
pub sigil KeyEvent {
    key: ~String,
    code: ~String,
    ctrl: ~bool,
    shift: ~bool,
    alt: ~bool,
    meta: ~bool,
}

pub sigil MouseEvent {
    x: ~f64,
    y: ~f64,
    button: ~u8,
}
```

### Hooks Module (`src/hooks/`)

**State Hooks** - Known (local state)
```sigil
pub rite use_state<T>(initial: !T) → (!T, !Fn(!T))! { ... }
pub rite use_signal<T>(initial: !T) → !Signal<!T> { ... }
```

**Effect Hooks** - No return, closures receive nothing external
```sigil
pub rite use_effect(effect: impl Fn()) { ... }
pub rite use_memo<T>(compute: impl Fn() → !T, deps: !Vec<!Any>) → !T { ... }
```

**Ref Hooks** - Known container, Uncertain contents
```sigil
pub rite use_ref<T>(initial: !T) → !Ref<?T> { ... }
```

### HTTP/Fetch (`sigil-http/`)

**Request Building** - Known (constructed locally)
```sigil
pub rite get(url: !str) → !RequestBuilder { ... }
pub rite post(url: !str) → !RequestBuilder { ... }
```

**Response Data** - Reported (external)
```sigil
pub rite fetch(req: !Request) → ?Response~ { ... }

pub sigil Response {
    status: ~u16,
    headers: ~HashMap<~String, ~String>,
    body: ~Bytes,
}
```

**Parsed JSON** - Reported (from network)
```sigil
pub rite json<T>(this) → ?T~ { ... }
```

### Router Module (`src/router/`)

**Route Definitions** - Known (defined in code)
```sigil
pub sigil Route {
    path: !String,
    component: !Fn() → !VNode,
}
```

**Route Parameters** - Reported (from URL)
```sigil
pub rite use_params() → ~HashMap<~String, ~String> { ... }
pub rite use_query() → ~HashMap<~String, ~String> { ... }
```

**Current Location** - Reported (browser state)
```sigil
pub rite use_location() → ~Location { ... }
```

### State Module (`src/state/`)

**Store** - Known (local state container)
```sigil
pub sigil Store<T> {
    state: !T,
    subscribers: !Vec<!Fn(!T)>,
}
```

**Actions/Reducers** - Known (defined in code)
```sigil
pub rite dispatch(this, action: !Action) { ... }
pub rite subscribe(this, listener: !Fn(!T)) → !Fn() { ... }
```

### A11y Module (`src/a11y/`)

**ARIA Attributes** - Known (set by code)
```sigil
pub rite aria_label(this, label: !str) → This! { ... }
pub rite aria_expanded(this, expanded: !bool) → This! { ... }
```

**Focus Management** - Elements are Known refs
```sigil
pub rite get_focusable_elements(container: !DomRef) → !Vec<!DomRef> { ... }
pub rite focus_first(container: !DomRef) { ... }
```

**Keyboard Events** - Reported (user input)
```sigil
pub rite use_keyboard_navigation(items: !Vec<!String>, config: !KeyboardNavConfig)
    → (!usize, !Fn(~KeyEvent) → !bool)! { ... }
```

## Compound Types

### Vectors and Collections
```sigil
// Vector of known items
≔ items: !Vec<!Item> = Vec·new();

// Vector of reported items (e.g., from API)
≔ users: !Vec<~User> = api.fetch_users();

// Vector itself reported (whole thing from external)
≔ data: ~Vec<~Item> = response.json();
```

### HashMaps
```sigil
// Local map with known keys and values
≔ cache: !HashMap<!String, !Value> = HashMap·new();

// Map from URL params (keys and values untrusted)
≔ params: ~HashMap<~String, ~String> = parse_query_string(url);
```

### Options/Results
```sigil
// Optional known value
≔ maybe: ?!String = map.get(key);

// Optional reported value
≔ maybe_user: ?~User = api.find_user(id);

// Result with known success, reported error
≔ result: Result<!Data, ~Error> = parse(input);
```

### Function Types
```sigil
// Handler that receives reported event
handler: !Fn(~Event)

// Callback that returns known value
compute: !Fn() → !i32

// Validator that promotes reported to known
validate: !Fn(~String) → ?!String
```

## Migration Checklist

When migrating a file:

1. **Identify data sources**
   - [ ] Literals/constants → `!`
   - [ ] User input → `~`
   - [ ] API responses → `~`
   - [ ] DOM events → `~`
   - [ ] Local computation → `!`

2. **Mark struct fields**
   - [ ] Required fields → `!Type`
   - [ ] Optional fields → `?Type`
   - [ ] External data fields → `~Type`

3. **Mark function signatures**
   - [ ] Local params → `!Type`
   - [ ] External params → `~Type`
   - [ ] Return types match computation source

4. **Handle promotions**
   - [ ] Validation promotes `~` to `!`
   - [ ] `.unwrap()` promotes `?` to `!`
   - [ ] Explicit checks promote appropriately

## Examples

### Counter Component
```sigil
// State is local - Known
static vary COUNT: !i64 = 0;

// Pure functions - Known in, Known out
rite increment() {
    COUNT = COUNT + 1;
}

// Render returns Known VNode
rite render() → !VNode {
    div()
        ·class("counter")
        ·child(text(f"Count: {COUNT}"))
        ·build()
}
```

### Form Component
```sigil
sigil FormState {
    name: !String,           // Local state
    email: !String,          // Local state
    submitted: !bool,        // Local flag
}

// Input handler receives Reported data from user
rite handle_input(event: ~InputEvent) {
    // Promote to known after validation
    ≔ value: !String = event.value |validate!{ .trim() };
    state.name = value;
}

// Submit handler
rite handle_submit(event: ~SubmitEvent) {
    event.prevent_default();

    // Form data is now validated/known
    ≔ data: !FormData = FormData {
        name: state.name,
        email: state.email,
    };

    // Send to API - response will be Reported
    ≔ response: ~ApiResponse = api.submit(data);
}
```

### Data Fetching
```sigil
rite UserList() → !VNode {
    ≔ (users, set_users) = use_state(!Vec<~User>::new());
    ≔ (loading, set_loading) = use_state(yea);

    use_effect(|| {
        // Fetch returns Reported data
        ≔ response: ~Vec<~User> = fetch("/api/users");
        set_users(response);
        set_loading(nay);
    }, []);

    ⎇ loading {
        div()·text("Loading...")·build()
    } ⎉ {
        ul()·children(
            users |τ{|user: ~User| {
                // User data is still Reported here
                li()·text(user.name)·build()
            }}
        )·build()
    }
}
```

---

*This guide should be used when applying evidentiality markers to Qliphoth source files.*

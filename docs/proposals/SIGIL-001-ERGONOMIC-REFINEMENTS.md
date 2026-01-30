# SIGIL-001: Ergonomic Refinements

**Status**: Draft
**Author**: Community Feedback (6,600+ LOC port)
**Date**: 2025-12-03
**Affects**: Syntax, Type System, Morpheme Pipes

---

## Abstract

This proposal addresses eight ergonomic refinements identified through porting 6,600+ lines of production Rust code to Sigil. Each refinement maintains Sigil's polysynthetic philosophy while reducing friction in common patterns.

---

## 1. Async Pipe Flow

### Problem

The `|await` placement disrupts left-to-right morpheme flow:

```sigil
// Current - the ? dangles awkwardly
let result = engine.generate(request)|await?

// Chaining breaks rhythm
let data = fetch(url)|await?|parse|await?|validate
```

### Proposal

Introduce `|⌛` as a proper pipe morpheme and allow evidentiality markers to chain:

```sigil
// Option A: Hourglass as pipe morpheme (consistent with existing ⌛ token)
let result = engine.generate(request)|⌛|?

// Option B: Async-aware morphemes
let result = engine.generate(request)|τ⌛{parse}|?

// Option C: Postfix await with evidentiality
let result = engine.generate(request)⌛?
```

### Recommended Syntax

**Option C** aligns best with polysynthetic principles—morphemes agglutinate without separators:

```sigil
// Postfix agglutination: expression + await + evidentiality
let user = fetch_user(id)⌛~        // await, returns reported
let data = parse(response)⌛?       // await, returns uncertain
let result = compute(input)⌛!      // await, returns known (infallible)

// Chains become rhythmic
let posts = fetch_user(id)⌛~
    |validate!
    |then{u => fetch_posts(u.id)}⌛~
    |τ{.title}
```

### Implementation Notes

- Lexer: Already has `Hourglass` token
- Parser: Extend postfix expression handling to allow `⌛` followed by evidentiality
- AST: Await already supports chaining; add evidentiality propagation

---

## 2. Error Transformation Pipe (`|?` / `|try`)

### Problem

Error mapping is verbose and breaks pipe flow:

```sigil
let value = some_result|map_err{|e| Error::internal(e.to_string())}?
```

### Proposal

Introduce `|?` with an optional transformation block:

```sigil
// Bare |? - propagate error unchanged
let value = some_result|?

// |?{mapper} - transform error type
let value = some_result|?{Error::internal}

// |?{|e| ...} - full error transformation
let value = some_result|?{|e| Error::wrap(e, "context")}
```

### Extended Semantics

```sigil
// Chain with context
let config = load_file(path)
    |?{|e| ConfigError::io(path, e)}
    |parse_toml
    |?{ConfigError::parse}
    |validate
    |?{ConfigError::validation}

// Collect all errors (for validation)
let results = items
    |τ{validate}
    |collect_errors      // New: Vec<Result<T,E>> -> Result<Vec<T>, Vec<E>>
```

### Alternative: `|‽` for Explicit Boundary

Since `‽` represents trust boundaries, use it for error transformation:

```sigil
let value = untrusted_result|‽{Error::from_external}
```

This makes the trust transition explicit in the syntax.

---

## 3. Implicit Struct Defaults

### Problem

When a struct has default values, constructors remain verbose:

```sigil
struct Config {
    host: str = "localhost",
    port: u16 = 8080,
    timeout: Duration = 30·sec,
    retries: u32 = 3,
}

// Current: must specify all or use ..Default
let cfg = Config { host: "prod.example.com", ..Config::default() }
```

### Proposal

Omitted fields with defaults are implicitly filled:

```sigil
// Proposed: omit fields with defaults
let cfg = Config { host: "prod.example.com" }
// Equivalent to:
// Config { host: "prod.example.com", port: 8080, timeout: 30·sec, retries: 3 }

// Explicit override
let cfg = Config { host: "prod.example.com", retries: 5 }
```

### Syntax Variant: Explicit Marker

If implicit behavior is too magical, use `..` without an expression:

```sigil
let cfg = Config { host: "prod.example.com", .. }
// The trailing .. means "fill remaining fields from defaults"
```

### Implementation Notes

- Parser: `FieldDef` already has structure; add `default: Option<Expr>`
- Type checker: Verify default expressions match field types at definition
- Codegen: Inject default values at construction sites

**Note**: Current AST (`ast.rs:653`) lacks `default` field in `FieldDef`. This requires:
```rust
pub struct FieldDef {
    pub visibility: Visibility,
    pub name: Ident,
    pub ty: TypeExpr,
    pub default: Option<Expr>,  // NEW
}
```

---

## 4. Pattern Matching in Pipes (`|match`)

### Problem

Pattern matching requires breaking the pipe chain:

```sigil
let result = items|τ{|item|
    match item.status {
        Status::Active => item.process(),
        Status::Pending => item.queue(),
        _ => item.skip(),
    }
}
```

### Proposal

Introduce `|match` as a morpheme:

```sigil
let result = items|match{
    Status::Active => .process(),
    Status::Pending => .queue(),
    _ => .skip(),
}

// With implicit item binding (. refers to piped value)
let names = users|match{
    User { active: true, .. } => .name.to_uppercase(),
    User { active: false, .. } => .name.to_lowercase(),
}
```

### Extended Patterns

```sigil
// Guard clauses
let categorized = numbers|match{
    n if n < 0 => Category::Negative,
    0 => Category::Zero,
    n if n < 100 => Category::Small,
    _ => Category::Large,
}

// Destructuring in pipes
let emails = responses|match{
    Ok(User { email, verified: true, .. }) => Some(email),
    _ => None,
}|φ{.is_some}|τ{.unwrap}

// Or cleaner with |match returning Option
let emails = responses|match?{
    Ok(User { email, verified: true, .. }) => email,
}  // Returns Option, None for non-matches
```

### Alternative Unicode: `|Μ` (Mu for Match)

Keep Greek letter consistency:

```sigil
let result = items|Μ{
    Pattern1 => expr1,
    Pattern2 => expr2,
}
```

---

## 5. Evidentiality in Function Signatures

### Problem

`Result<T, E>` is verbose for common patterns:

```sigil
fn load(path: &str) -> Result<Config, IoError>
fn parse(s: &str) -> Result<Ast, ParseError>
fn validate(cfg: Config) -> Result<Config, ValidationError>
```

### Proposal

Extend evidentiality markers to return types:

```sigil
// Definitely returns or panics (infallible path, may panic)
fn must_load(path: &str) -> Config! {
    load(path).expect("config required")
}

// Might fail (sugar for Result<T, E>)
fn load(path: &str) -> Config?[IoError] {
    // ...
}

// External/untrusted source (sugar for Result<T~, E>)
fn fetch(url: &str) -> Response~[NetworkError] {
    // ...
}

// Chained uncertainty
fn load_and_parse(path: &str) -> Ast?[ConfigError] {
    load(path)?[IoError]|parse?[ParseError]
}
```

### Syntax Options

```sigil
// Option A: Bracket syntax for error type
fn load(path: &str) -> Config?[IoError]

// Option B: Pipe syntax (error flows right)
fn load(path: &str) -> Config?|IoError

// Option C: Angle brackets (familiar from generics)
fn load(path: &str) -> Config?<IoError>

// Option D: Colon for "might be"
fn load(path: &str) -> Config?: IoError
```

### Recommended: Option A

Brackets `[]` are unused in this position and visually distinct:

```sigil
fn validate(input~: UserInput) -> User![ValidationError] {
    // Validates external input, returns known or fails
}

// The ! means "if this returns, it's definitely valid"
// The [ValidationError] specifies the failure mode
```

### Implicit Error Context

```sigil
// Module-level error type
mod config {
    error ConfigError { Io(IoError), Parse(ParseError), Validate(String) }

    // Functions inherit module error type
    fn load(path: &str) -> Config? {  // Implicitly -> Config?[ConfigError]
        // ...
    }
}
```

---

## 6. Reduction Pipe Variants (`|ρ_*`)

### Problem

Only `|ρ` (general reduce) and `|ρ_sum` exist. Common reductions require verbose closures:

```sigil
let product = numbers|ρ{|a, b| a * b}
let max = numbers|ρ{|a, b| if a > b { a } else { b }}
let concat = strings|ρ{|a, b| a + b}
```

### Proposal

Add reduction morpheme variants:

| Morpheme | Operation | Identity |
|----------|-----------|----------|
| `\|ρ+` or `\|ρ_sum` | Addition | 0 |
| `\|ρ*` or `\|ρ_prod` | Multiplication | 1 |
| `\|ρ⊥` or `\|ρ_min` | Minimum | Type::MAX |
| `\|ρ⊤` or `\|ρ_max` | Maximum | Type::MIN |
| `\|ρ++` or `\|ρ_cat` | Concatenation | Empty |
| `\|ρ&` or `\|ρ_all` | Logical AND | true |
| `\|ρ\|` or `\|ρ_any` | Logical OR | false |
| `\|ρ∩` or `\|ρ_inter` | Set intersection | Universe |
| `\|ρ∪` or `\|ρ_union` | Set union | Empty set |

### Usage

```sigil
let total = prices|ρ+
let factorial = (1..=n)|ρ*
let oldest = users|τ{.age}|ρ_max
let youngest = users|τ{.age}|ρ_min
let all_names = users|τ{.name}|ρ++

// With initial value
let biased_sum = numbers|ρ+(100)  // Start from 100

// Boolean reductions
let all_valid = items|τ{.is_valid}|ρ&
let any_error = items|τ{.has_error}|ρ|
```

### Unicode Variants (Optional)

For Unicode enthusiasts:

```sigil
let sum = numbers|ρ∑      // Sigma for sum
let product = numbers|ρ∏  // Pi for product
```

---

## 7. Implicit `self` in Methods

### Problem

`self.field` is repetitive in method bodies:

```sigil
impl Counter {
    fn increment(self) {
        self.count += 1;
        self.last_updated = now();
        if self.count > self.max {
            self.reset();
        }
    }
}
```

### Proposal: Implicit Field Access

Within method scope, bare field names resolve to `self.field`:

```sigil
impl Counter {
    fn increment(mut self) {
        count += 1;           // self.count
        last_updated = now(); // self.last_updated
        if count > max {      // self.count > self.max
            reset();          // self.reset()
        }
    }
}
```

### Disambiguation

When a local shadows a field, use explicit `self.`:

```sigil
impl User {
    fn update_name(mut self, name: str) {
        // `name` refers to parameter, not self.name
        self.name = name;

        // Or use field prefix syntax
        .name = name;  // .field means self.field
    }
}
```

### Alternative: Dot Prefix

Less implicit, but still concise:

```sigil
impl Counter {
    fn increment(mut self) {
        .count += 1;
        .last_updated = now();
        if .count > .max {
            .reset();
        }
    }
}
```

This is similar to `@field` in Ruby or `this.field` shortened to `.field`.

---

## 8. String Interpolation

### Problem

`format!()` is verbose for simple cases:

```sigil
let msg = format!("User {} has {} items", name, count);
let path = format!("{}/config/{}.toml", base_dir, env);
```

### Proposal

Implicit interpolation in double-quoted strings:

```sigil
let msg = "User {name} has {count} items"
let path = "{base_dir}/config/{env}.toml"

// Expressions in braces
let debug = "Result: {result:?}, took {elapsed.as_millis()}ms"

// Escaped braces
let json = "\\{\"name\": \"{name}\"\\}"
// Or use raw strings
let json = r#"{"name": "{name}"}"#  // No interpolation in raw strings
```

### Format Specifiers

```sigil
let hex = "Value: {n:x}"           // Hexadecimal
let padded = "ID: {id:05}"         // Zero-padded
let float = "Price: {price:.2}"    // 2 decimal places
let debug = "Object: {obj:?}"      // Debug format
let pretty = "Data: {data:#?}"     // Pretty debug
```

### Opt-out Syntax

For strings that shouldn't interpolate:

```sigil
// Raw strings (no interpolation)
let regex = r"(\w+)\s*=\s*(\d+)"

// Template literals (explicit interpolation only)
let template = t"Hello, ${name}!"  // Only ${} interpolates
```

### Security Consideration

Since Sigil has evidentiality, interpolation should track it:

```sigil
let user_input~: str = get_input()
let query = "SELECT * FROM users WHERE name = '{user_input}'"
// Type of query is str~ (reported/untrusted) due to interpolated ~ value
```

This makes SQL injection visible at the type level.

---

## Implementation Priority

Based on impact vs. effort:

### Phase 1: Quick Wins
1. **String interpolation** — High impact, parser-only change
2. **`|ρ_*` variants** — stdlib addition, no grammar change
3. **Implicit struct defaults** — Requires AST change, but localized

### Phase 2: Core Improvements
4. **`|match` morpheme** — New pipe operation, moderate parser work
5. **`|?` error pipe** — New morpheme, integrates with existing `?`
6. **Async flow (`⌛?`)** — Extends existing token handling

### Phase 3: Deeper Changes
7. **Evidential return types** — Type system change, significant
8. **Implicit self** — Scoping rules change, needs careful design

---

## Compatibility

All proposals are backward compatible:

- Existing code continues to work unchanged
- New syntax is additive (no removed features)
- Migration can be incremental

---

## Open Questions

1. **Implicit defaults**: Should `..` be required or truly implicit?
2. **`|match` vs `|Μ`**: Greek letter consistency vs. readability?
3. **Error type syntax**: `?[E]` vs `?<E>` vs `?|E`?
4. **Self fields**: Implicit resolution vs. `.field` prefix?
5. **Interpolation**: Always on vs. explicit `f"..."` strings?

---

## References

- [Sigil Lexical Specification](../specs/01-LEXICAL.md)
- [Sigil Syntax Specification](../specs/02-SYNTAX.md)
- [Sigil Type System](../specs/03-TYPES.md)
- [Sigil Concurrency](../specs/06-CONCURRENCY.md)

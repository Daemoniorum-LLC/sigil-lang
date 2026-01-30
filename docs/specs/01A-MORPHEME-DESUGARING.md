# Sigil Morpheme Desugaring Specification

> *"A polysynthetic language compresses entire sentences into single words. An agent-native
> language compresses entire algorithms into single expressions."*

## 1. Overview

This specification defines the algorithmic semantics for Sigil's morpheme system — how morpheme
operators are desugared into core language constructs. Unlike traditional operator overloading,
morphemes are **linguistic primitives** that the compiler transforms via principled rewriting
rules.

### 1.1 Design Philosophy

Sigil's morpheme system draws from polysynthetic linguistics where meaning is composed from
bound morphemes. For an agent with unbounded cognition, these provide:

1. **Compositional Density** — Complex transformations in minimal syntax
2. **Semantic Transparency** — Each morpheme has a single, consistent meaning
3. **Type-Directed Expansion** — Desugaring varies by receiver type
4. **Evidence Preservation** — Transformations maintain evidentiality chains

### 1.2 Document Structure

| Section | Content |
|---------|---------|
| §2 | Core transformation morphemes (τ, φ, σ, ρ, π) |
| §3 | Evidentiality morphemes (!, ?, ~, ‽) |
| §4 | Aspect morphemes (·ing, ·ed, ·able, ·ive) |
| §5 | Valency morphemes (·in, ·out, ·mut) |
| §6 | Incorporation and compound formation |
| §7 | Morpheme interaction rules |
| §8 | Error semantics |

---

## 2. Transformation Morphemes

### 2.1 Map Morpheme (τ)

The τ (tau) morpheme applies a transformation to elements of a collection or functor.

**Syntax:**
```
expr|τ{body}
expr|τ.field
expr|τ(fn)
```

**Desugaring Rules:**

```
DESUGAR[e|τ{body}] where Γ ⊢ e : F<A>, Functor<F>
  ⟹ e.map(|__item| { SUBST[body, it → __item] })

DESUGAR[e|τ.field] where Γ ⊢ e : F<A>, A has field
  ⟹ e.map(|__item| __item.field)

DESUGAR[e|τ(fn)] where Γ ⊢ e : F<A>, Γ ⊢ fn : A → B
  ⟹ e.map(fn)
```

**Implicit Binding:**

Within `τ{...}`, the implicit binding `it` refers to the current element.
If the body is a bare expression without explicit binding, `it` is inserted:

```sigil
// These are equivalent
data|τ{x * 2}        // Error: x not in scope
data|τ{it * 2}       // Explicit it binding
data|τ(fn(x) x * 2)  // Explicit lambda

// Field shorthand
users|τ.name         // Desugars to: users.map(|it| it.name)
```

**Type Requirements:**

The receiver must implement `Functor`:

```sigil
trait Functor<F> {
    fn map<A, B>(self: F<A>, f: fn(A) → B) → F<B>;
}
```

**Evidence Propagation:**

τ preserves the outer evidence but may transform inner evidence:

```
Γ ⊢ e : F<A>^E
Γ ⊢ body : A^E' → B^E''
─────────────────────────────
Γ ⊢ e|τ{body} : F<B>^(E ⊓ E'')
```

### 2.2 Filter Morpheme (φ)

The φ (phi) morpheme selects elements matching a predicate.

**Syntax:**
```
expr|φ{pred}
expr|φ.field        // Filter where field is truthy
expr|φ(fn)          // Filter with predicate function
expr|φ>n            // Shorthand: filter greater than n
expr|φ<n            // Shorthand: filter less than n
expr|φ=v            // Shorthand: filter equal to v
expr|φ!=v           // Shorthand: filter not equal to v
```

**Desugaring Rules:**

```
DESUGAR[e|φ{pred}] where Γ ⊢ e : F<A>, Filterable<F>
  ⟹ e.filter(|__item| { SUBST[pred, it → __item] })

DESUGAR[e|φ.field] where Γ ⊢ e : F<A>, A.field : bool
  ⟹ e.filter(|__item| __item.field)

DESUGAR[e|φ>n]
  ⟹ e.filter(|__item| __item > n)

DESUGAR[e|φ=v]
  ⟹ e.filter(|__item| __item == v)
```

**Type Requirements:**

```sigil
trait Filterable<F>: Functor<F> {
    fn filter<A>(self: F<A>, pred: fn(&A) → bool) → F<A>;
}
```

**Evidence Interaction:**

Filtering on evidence levels is well-defined:

```sigil
data|φ{it!}         // Keep only Known values
data|φ{!it?}        // Keep values that are NOT uncertain
data|φ{it~.source == "trusted"} // Filter by evidence metadata
```

### 2.3 Sort Morpheme (σ)

The σ (sigma) morpheme orders elements.

**Syntax:**
```
expr|σ              // Sort by natural order (Ord trait)
expr|σ.field        // Sort by field
expr|σ·desc         // Sort descending
expr|σ·desc.field   // Sort by field, descending
expr|σ(fn)          // Sort by key function
```

**Desugaring Rules:**

```
DESUGAR[e|σ] where Γ ⊢ e : F<A>, Sortable<F>, Ord<A>
  ⟹ e.sort_by(|a, b| a.cmp(b))

DESUGAR[e|σ.field] where Γ ⊢ e : F<A>, Ord<A.field>
  ⟹ e.sort_by(|a, b| a.field.cmp(b.field))

DESUGAR[e|σ·desc]
  ⟹ e.sort_by(|a, b| b.cmp(a))

DESUGAR[e|σ·desc.field]
  ⟹ e.sort_by(|a, b| b.field.cmp(a.field))

DESUGAR[e|σ(fn)] where Γ ⊢ fn : A → K, Ord<K>
  ⟹ e.sort_by_key(fn)
```

**Stability:**

σ is a **stable sort** by default. For unstable (potentially faster) sort:

```sigil
data|σ·unstable     // Unstable sort
data|σ·unstable.field
```

### 2.4 Reduce Morpheme (ρ)

The ρ (rho) morpheme folds a collection to a single value.

**Syntax:**
```
expr|ρ(op)          // Reduce with binary operator
expr|ρ(op, init)    // Reduce with initial value
expr|ρ+             // Shorthand: sum
expr|ρ*             // Shorthand: product
expr|ρ&&            // Shorthand: all (logical and)
expr|ρ||            // Shorthand: any (logical or)
expr|ρ++            // Shorthand: concatenate
```

**Desugaring Rules:**

```
DESUGAR[e|ρ(op)] where Γ ⊢ e : F<A>, Foldable<F>, Γ ⊢ op : (A, A) → A
  ⟹ e.reduce(op)  // Returns Option<A>

DESUGAR[e|ρ(op, init)] where Γ ⊢ init : B, Γ ⊢ op : (B, A) → B
  ⟹ e.fold(init, op)  // Returns B

DESUGAR[e|ρ+] where Γ ⊢ e : F<A>, Add<A>
  ⟹ e.fold(A::zero(), |acc, x| acc + x)

DESUGAR[e|ρ*] where Γ ⊢ e : F<A>, Mul<A>
  ⟹ e.fold(A::one(), |acc, x| acc * x)

DESUGAR[e|ρ++] where Γ ⊢ e : F<C>, Concat<C>
  ⟹ e.fold(C::empty(), |acc, x| acc ++ x)
```

**Evidence of Reduction:**

Reducing uncertain values produces uncertain results:

```
Γ ⊢ e : F<A^?>
─────────────────────
Γ ⊢ e|ρ+ : A^?
```

If the collection might be empty, reduce returns uncertain:

```sigil
let sum? = data|ρ+          // Uncertain: might be empty
let sum! = data|ρ(+, 0)     // Known: has initial value
```

### 2.5 Project Morpheme (π)

The π (pi) morpheme selects specific fields from structured data.

**Syntax:**
```
expr|π(field1, field2, ...)  // Project specific fields
expr|π·flat                  // Flatten nested structures
expr|π·distinct              // Remove duplicates
```

**Desugaring Rules:**

```
DESUGAR[e|π(f1, f2, ...fn)] where Γ ⊢ e : F<{f1: T1, f2: T2, ...fm: Tm}>
  ⟹ e.map(|item| {f1: item.f1, f2: item.f2, ...fn: item.fn})

DESUGAR[e|π·flat] where Γ ⊢ e : F<F<A>>
  ⟹ e.flatten()

DESUGAR[e|π·distinct] where Γ ⊢ e : F<A>, Eq<A>, Hash<A>
  ⟹ e.into_iter().collect::<HashSet<_>>().into_iter().collect()
```

---

## 3. Evidentiality Morphemes

Evidentiality morphemes track the provenance and certainty of values. Unlike transformation
morphemes which operate on collections, these attach to individual values.

### 3.1 Evidence Markers

| Marker | Name | Type Transform | Meaning |
|--------|------|----------------|---------|
| `!` | Known | `T → T^!` | Computed locally, verified |
| `?` | Uncertain | `T → Option<T>` | Possibly absent |
| `~` | Reported | `T → T^~` | External source, unverified |
| `‽` | Paradox | `T^E → T^!` | Forced trust assertion |

### 3.2 Desugaring Rules

**Known (`!`):**

The `!` marker asserts that a value is locally computed and known:

```
DESUGAR[e!] where Γ ⊢ e : T
  ⟹ ASSERT_KNOWN(e)  // Compile-time: verify e has no external deps
                      // Runtime: identity (zero-cost)
```

The compiler tracks value provenance. A value is `!`-eligible if:
- It's a literal
- It's computed entirely from other `!` values
- It's extracted from a verified structure

**Uncertain (`?`):**

The `?` marker wraps values in `Option`:

```
DESUGAR[e?] where Γ ⊢ e : T
  ⟹ Option::Some(e) : Option<T>

DESUGAR[e.field?] where Γ ⊢ e : Option<T>
  ⟹ e.map(|v| v.field)  // Propagates None
```

**Chained uncertainty** (the `?` propagation):

```sigil
// Traditional
let x = match a {
    Some(a) => match a.b {
        Some(b) => match b.c {
            Some(c) => Some(c.value),
            None => None,
        },
        None => None,
    },
    None => None,
};

// Morpheme syntax
let x? = a?.b?.c?.value
```

**Reported (`~`):**

The `~` marker indicates external/untrusted data:

```
DESUGAR[e~] where Γ ⊢ e : T
  ⟹ Reported::new(e, SOURCE_INFO)

struct Reported<T> {
    value: T,
    source: SourceInfo,
    timestamp: Instant,
}
```

Reported values cannot be used in trust-sensitive contexts without explicit promotion:

```sigil
let user_input~ = stdin.read_line()     // Reported
let len! = user_input~.len()            // ERROR: can't derive ! from ~

let validated! = validate(user_input~)‽ // OK: explicit trust assertion
```

**Paradox (`‽`):**

The `‽` (interrobang) forces a trust boundary crossing:

```
DESUGAR[e‽] where Γ ⊢ e : T^E
  ⟹ unsafe { trust_assert(e) } : T^!
```

This generates a trust boundary marker in the compiled output, enabling:
- Audit trails of trust assertions
- Runtime trust verification (if enabled)
- Static analysis of trust boundaries

### 3.3 Evidence Algebra

Evidence levels form a lattice under the `⊓` (meet) operation:

```
        !  (Known - top)
       / \
      ?   ~  (Uncertain, Reported)
       \ /
        ‽  (Paradox - forced join)
```

**Combination Rules:**

| E₁ | E₂ | E₁ ⊓ E₂ | Explanation |
|----|-----|---------|-------------|
| ! | ! | ! | Known + Known = Known |
| ! | ? | ? | Known + Uncertain = Uncertain |
| ! | ~ | ~ | Known + Reported = Reported |
| ? | ~ | ?~ | Uncertain + Reported = Both |
| ‽ | E | ! | Paradox overrides to Known |

---

## 4. Aspect Morphemes

Aspect morphemes encode the temporal or completion state of operations.

### 4.1 Progressive (·ing)

Indicates an ongoing, streaming operation:

```
DESUGAR[fn·ing(args)] where Γ ⊢ fn : A → B
  ⟹ fn_streaming(args) : Stream<B>
```

Functions with `·ing` aspect return streams or iterators:

```sigil
fn read·ing(file: File) → Stream<Line> {
    // Returns a lazy stream that yields lines
    file.lines()
}

// Usage
let lines = file|read·ing|τ.trim|φ{!it.is_empty()}
```

### 4.2 Perfective (·ed)

Indicates a completed operation:

```
DESUGAR[fn·ed(args)] where Γ ⊢ fn : A → Future<B>
  ⟹ fn(args).await : B
```

The `·ed` aspect forces completion:

```sigil
let data·ed! = fetch(url)   // Awaits completion, asserts success
let parsed·ed = json·parse(data·ed!)
```

### 4.3 Potential (·able)

Checks capability without performing:

```
DESUGAR[fn·able(args)] where Γ ⊢ fn : A → Result<B, E>
  ⟹ fn.can_perform(args) : bool
```

```sigil
if file·open·able(path) {
    let f = file·open(path)!  // Safe: we checked
}
```

### 4.4 Resultative (·ive)

Indicates the operation produces side effects or results:

```
DESUGAR[fn·ive(args)]
  ⟹ fn(args) : Result<B, E>  // May fail, has effects
```

```sigil
fn delete·ive(path: Path) → Result<(), IoError> {
    // Destructive operation
    fs·remove(path)
}
```

---

## 5. Valency Morphemes

Valency morphemes describe argument flow direction.

### 5.1 Inward (·in)

Consuming operations that take ownership:

```
DESUGAR[Type·in<T>]
  ⟹ fn(T) → ()  // Takes ownership, consumes

fn consume·in(data: Data) {
    // data is moved in and consumed
}
```

### 5.2 Outward (·out)

Producing operations that yield new values:

```
DESUGAR[Type·out<T>]
  ⟹ fn() → T  // Produces new value

fn generate·out() → Data {
    Data::new()
}
```

### 5.3 Mutable (·mut)

In-place modification:

```
DESUGAR[fn·mut(args)] where first arg is &mut T
  ⟹ fn(&mut args.0, args.1..)

fn sort·mut(data: &mut [T]) {
    // Sorts in place
}

// Usage
data|sort·mut  // Modifies data directly
```

---

## 6. Incorporation (Compound Formation)

The middle dot `·` fuses nouns and verbs into compounds.

### 6.1 Noun·Verb Compounds

```
DESUGAR[noun·verb(args)]
  ⟹ verb(noun, args)  // noun becomes first arg

file·read(path)  ⟹  read(file::open(path))
json·parse(text) ⟹  json::parse(text)
```

### 6.2 Verb Chains

```
DESUGAR[noun·verb1·verb2·verb3(args)]
  ⟹ verb3(verb2(verb1(noun, args)))

// Desugaring example
path|file·open·read·parse·close
  ⟹ close(parse(read(open(path))))
  // But with proper error handling and RAII
```

### 6.3 Type Resolution

For compound `a·b`:

1. If `a` is a module, resolve `a::b`
2. If `a` is a type and `b` is a method, resolve `a.b` or `a::b`
3. If `a` is a noun and `b` is a verb, apply verb to noun
4. Otherwise, treat as single identifier `a·b`

```
RESOLVE[a·b]
  ⟹ if MODULE(a) then a::b
  ⟹ else if TYPE(a) ∧ METHOD(a, b) then a.b
  ⟹ else if NOUN(a) ∧ VERB(b) then b(a, ...)
  ⟹ else IDENT(a·b)
```

---

## 7. Morpheme Interaction Rules

### 7.1 Ordering Constraints

Some morpheme combinations have required orderings:

```
// Valid orderings (compositionally meaningful)
data|φ{pred}|τ{fn}|σ|ρ+    // Filter → Map → Sort → Reduce
data|τ{fn}|φ{pred}|σ       // Map → Filter → Sort

// Invalid orderings (type errors)
data|ρ+|τ{fn}              // ERROR: ρ reduces to scalar, can't τ
data|σ|ρ+|φ{pred}          // ERROR: ρ reduces before φ
```

### 7.2 Evidence Threading

Morphemes thread evidence through transformations:

```
Γ ⊢ e : F<A^E₁>
Γ ⊢ τ{body} : A^E₁ → B^E₂
────────────────────────────────
Γ ⊢ e|τ{body} : F<B^(E₁ ⊓ E₂)>
```

**Example:**

```sigil
let data~: Vec<i32^~> = fetch_numbers()  // Reported
let doubled~: Vec<i32^~> = data~|τ{it * 2}  // Still reported
let sum~: i32^~ = doubled~|ρ+  // Reduced, still reported

let verified! = verify_sum(sum~)‽  // Trust assertion
```

### 7.3 Aspect Compatibility

Aspects must be compatible in chains:

| Aspect 1 | Aspect 2 | Combination |
|----------|----------|-------------|
| ·ing | ·ing | Stream composition |
| ·ing | ·ed | Collect then await |
| ·ed | ·ing | Await then stream (rare) |
| ·ed | ·ed | Sequential await |

```sigil
// Valid: stream composition
source|read·ing|τ·ing{parse}|φ·ing{valid}

// Valid: collect streaming, then complete
source|read·ing|collect·ed

// Invalid: can't stream after collection
source|read·ed|τ·ing{...}  // ERROR: ed produces value, not stream
```

### 7.4 Fusion Optimization

The compiler may fuse adjacent morphemes:

```
FUSE[e|τ{f}|τ{g}]
  ⟹ e|τ{g(f(it))}  // Single pass

FUSE[e|φ{p}|φ{q}]
  ⟹ e|φ{p && q}    // Single filter

FUSE[e|τ{f}|φ{p}]
  ⟹ e.filter_map(|x| { let y = f(x); if p(y) { Some(y) } else { None } })
```

---

## 8. Error Semantics

### 8.1 Type Errors

**Missing Trait:**

```sigil
struct Point { x: i32, y: i32 }
let points: Vec<Point> = ...

points|σ  // ERROR: Point does not implement Ord
          // Hint: use σ.x or σ(fn(p) p.x + p.y)
```

**Evidence Mismatch:**

```sigil
fn secure(data: Vec<i32^!>) { ... }

let untrusted~ = fetch()
secure(untrusted~)  // ERROR: expected ^! evidence, found ^~
                    // Hint: use validate()‽ to assert trust
```

### 8.2 Runtime Errors

**Empty Collection:**

```sigil
let nums: Vec<i32> = vec![]
nums|ρ+  // Returns None (not an error)
nums|α   // Returns None
nums|α!  // PANIC: unwrap on None
```

**Filter Produces Empty:**

```sigil
data|φ{false}|α  // Always None
data|φ{false}|α! // Always panics
```

### 8.3 Evidence Errors

When evidence cannot be satisfied:

```sigil
let x! = compute()   // OK if compute() returns known value
let y! = fetch()     // ERROR: fetch() returns reported value
                     // Cannot assign ^~ to ^! binding
```

### 8.4 Diagnostic Messages

Error messages should reference morpheme semantics:

```
error[E0423]: morpheme `τ` requires Functor implementation
  --> src/main.sg:10:5
   |
10 |     data|τ{it * 2}
   |     ^^^^-- τ (transform) maps over elements
   |
   = help: implement `Functor` for `CustomType`
   = help: or convert to a standard collection first

error[E0847]: evidence level mismatch in morpheme chain
  --> src/main.sg:15:10
   |
15 |     let x! = data~|τ{it + 1}
   |         ^^          ^^^^^^^^ evidence ^~ (reported)
   |         |
   |         expected ^! (known)
   |
   = note: morpheme `τ` preserves evidence levels
   = help: use `validate()‽` to assert trust before assignment
```

---

## 9. Standard Morpheme Implementations

### 9.1 Built-in Types

| Type | τ | φ | σ | ρ | π | α | Ω |
|------|---|---|---|---|---|---|---|
| `Vec<T>` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Option<T>` | ✓ | ✓ | — | — | — | ✓ | ✓ |
| `Result<T,E>` | ✓ | — | — | — | — | ✓ | — |
| `Iterator<T>` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `Stream<T>` | ✓ | ✓ | ✓* | ✓* | ✓ | ✓ | — |
| `HashMap<K,V>` | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| `String` | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |

*Stream operations marked with * require buffering.

### 9.2 Defining Custom Morpheme Behavior

```sigil
impl Functor for MyType {
    fn map<A, B>(self: MyType<A>, f: fn(A) → B) → MyType<B> {
        // Custom map implementation
    }
}

impl Filterable for MyType {
    fn filter<A>(self: MyType<A>, pred: fn(&A) → bool) → MyType<A> {
        // Custom filter implementation
    }
}
```

---

## 10. Formal Grammar Extension

```ebnf
morpheme_expr
    : expr '|' morpheme_chain
    ;

morpheme_chain
    : morpheme_op ( '|' morpheme_op )*
    ;

morpheme_op
    : TRANSFORM_MORPHEME morpheme_arg?
    | EVIDENCE_MARKER
    | ASPECT_SUFFIX
    | VALENCY_SUFFIX
    ;

TRANSFORM_MORPHEME
    : 'τ' | 'φ' | 'σ' | 'ρ' | 'π' | 'α' | 'Ω' | 'δ' | 'λ'
    ;

morpheme_arg
    : '{' expr '}'          // Block form
    | '.' IDENT             // Field access
    | '(' expr_list ')'     // Function form
    | COMPARISON_OP expr    // Shorthand comparison
    ;

EVIDENCE_MARKER
    : '!' | '?' | '~' | '‽'
    ;

ASPECT_SUFFIX
    : '·' ( 'ing' | 'ed' | 'able' | 'ive' )
    ;

VALENCY_SUFFIX
    : '·' ( 'in' | 'out' | 'mut' | 'un' | 'mono' | 'bi' | 'poly' )
    ;
```

---

## 11. Implementation Notes

### 11.1 Desugaring Pass Location

Morpheme desugaring occurs **after** parsing but **before** type checking:

```
Source → Lexer → Parser → AST
  → Morpheme Desugaring (this spec)
  → Name Resolution
  → Type Inference
  → Borrow Checking
  → MIR Generation
```

### 11.2 Optimization Opportunities

1. **Fusion** — Adjacent τ/φ operations fuse into single iteration
2. **Short-circuit** — φ{false} eliminates subsequent operations
3. **Specialization** — Known element types enable monomorphization
4. **Evidence elision** — Compile-time evidence checking removes runtime cost

### 11.3 Debug Information

Preserve morpheme structure in debug info:

```
// Debugger shows original morpheme chain
data|φ.active|τ{.name}|σ|α?
     ↑        ↑       ↑ ↑
     φ        τ       σ α  (step through each)
```

---

*This specification enables Sigil's polysynthetic nature — expressing complex data
transformations in dense, composable morpheme chains that an agent can read as
naturally as breathing.*

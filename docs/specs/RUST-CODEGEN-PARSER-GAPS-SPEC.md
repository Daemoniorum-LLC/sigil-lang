# Rust Codegen Parser Gaps Spec

**Version:** 1.1.0
**Status:** Active
**Date:** 2026-02-06
**Methodology:** Spec-Driven Development (SDD) + Agent-TDD
**Discovery:** Rust codegen backend revealed parser-level AST representation gaps that affect clean translation.

---

## 1. Overview

The Rust codegen backend (`sigil rust`) transpiles Sigil AST to Rust source code. During implementation, several gaps were discovered where either:
1. Parser representation doesn't cleanly map to Rust semantics
2. Codegen needs additional expression/pattern handlers
3. Type inference differences between Sigil interpreter and Rust compiler

### 1.1 Methodology

This spec follows **Spec-Driven Development**:
- When implementation reveals gaps, STOP and document here
- Gaps are discoveries, not failures
- Each gap includes: Discovery, Root Cause, Fix, Test Coverage

### 1.2 Current Status

| Crate | Status | Notes |
|-------|--------|-------|
| nihil-core | ✅ Compiles | 8 warnings (intentional scaffolding) |
| nihil-ops | ✅ Compiles | 9 warnings (unused helpers) |

**Spec Status:** All gaps elevated to S++ standard as of 2026-02-06.

---

## 2. Gap A: Reference Binding Patterns ✅ FIXED

### 2.1 Specification

#### GRAMMAR

```
Pattern         ::= LiteralPattern
                  | IdentPattern
                  | WildcardPattern
                  | RefPattern
                  | RefBindingPattern
                  | TuplePattern
                  | StructPattern
                  | EnumPattern
                  | SlicePattern
                  | OrPattern

RefPattern      ::= "&" "vary"? Pattern          // &x or &vary x
RefBindingPattern ::= "ref" "vary"? Ident Evidentiality?  // ref x or ref vary x

Evidentiality   ::= "!" | "?" | "~"              // certain, uncertain, inferred
```

#### TRANSFORMATION

```
RefPattern (borrowing existing reference):
  INPUT:  &P           OUTPUT: &P
  INPUT:  &vary P      OUTPUT: &mut P

RefBindingPattern (creating reference to matched value):
  INPUT:  ref N        OUTPUT: ref N
  INPUT:  ref vary N   OUTPUT: ref mut N
  INPUT:  ref N?       OUTPUT: ref N        // evidentiality erased
  INPUT:  ref vary N!  OUTPUT: ref mut N    // evidentiality erased
```

#### SEMANTIC DISTINCTION

```
&x      - Pattern matches a reference, binds x to the dereferenced value
ref x   - Pattern matches a value, binds x to a reference to that value

Example:
  match &Some(42) {
    &Some(x) => ...     // x: i32 (copied out)
    Some(ref x) => ...  // x: &i32 (borrowed in place)
  }
```

### 2.2 Current Status: ✅ IMPLEMENTED

Codegen handles both `Pattern::Ref` and `Pattern::RefBinding`.

### 2.3 Implementation

```
emit_pattern(p):
  MATCH p:
    Pattern::Ref { mutable, inner }:
      emit "&"
      IF mutable: emit "mut "
      emit_pattern(inner)

    Pattern::RefBinding { mutable, name, evidentiality }:
      emit "ref "
      IF mutable: emit "mut "
      emit name
      // evidentiality is intentionally NOT emitted (Rust has no equivalent)
```

### 2.4 Invariants

```
I₁: Evidentiality markers are ALWAYS erased in Rust output
I₂: "vary" ALWAYS becomes "mut"
I₃: RefPattern and RefBindingPattern are distinct AST nodes
I₄: ref binding only valid in pattern position (not expression)
I₅: Nested ref patterns are valid: &ref vary x
```

### 2.5 Edge Cases

```
EC₁: ref x
     → ref x

EC₂: ref vary x
     → ref mut x

EC₃: &x
     → &x

EC₄: &vary x
     → &mut x

EC₅: &ref vary x  (borrow of ref binding - unusual but valid)
     → &ref mut x

EC₆: ref _  (wildcard with ref)
     → ref _

EC₇: ref vary _
     → ref mut _

EC₈: Some(ref vary inner)
     → Some(ref mut inner)

EC₉: (ref a, ref vary b)
     → (ref a, ref mut b)

EC₁₀: ref x!  (certain evidentiality)
      → ref x

EC₁₁: ref vary x?  (uncertain evidentiality)
      → ref mut x
```

### 2.6 Error Conditions

```
E₁: ref ref x
    → PARSER ERROR: "duplicate ref in pattern"

E₂: ref vary vary x
    → PARSER ERROR: "duplicate mutability in pattern"

E₃: ref in expression position
    → PARSER ERROR: "ref only valid in patterns"
```

### 2.7 Composition

```
WITH if-let:
  ⎇ ≔ Some(ref vary x) = expr { }
  → if let Some(ref mut x) = expr { }

WITH match:
  ⌥ expr {
    Some(ref vary x) ⇒ ...
  }
  → match expr {
      Some(ref mut x) => ...
    }

WITH function params:
  INVALID - ref not allowed in function parameters
  Use &vary T instead: fn foo(x: &vary T)

WITH let binding:
  ≔ ref vary x = expr
  → let ref mut x = expr  (unusual but valid Rust)
```

### 2.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_ref_patterns.sg`

**Required test cases:**
- Basic ref binding
- Mutable ref binding
- Ref in Option pattern
- Ref in tuple pattern
- Ref with evidentiality (verify erasure)
- Nested ref patterns

---

## 3. Gap B: Associated Type Binding vs Default Type Parameter ✅ IMPLEMENTED

### 3.1 Specification

#### GRAMMAR

```
GenericParamList ::= "<" GenericParam ("," GenericParam)* ">"
GenericParam     ::= TypeParam | LifetimeParam | ConstParam
TypeParam        ::= Ident (":" Bounds)? ("=" DefaultType)?
Bounds           ::= Bound ("+" Bound)*
Bound            ::= TypePath | Lifetime
DefaultType      ::= TypeExpr

GenericArgList   ::= "<" GenericArg ("," GenericArg)* ">"
GenericArg       ::= TypeExpr | Lifetime | ConstExpr | AssocTypeBinding
AssocTypeBinding ::= Ident "=" TypeExpr
```

#### TRANSFORMATION

```
CONTEXT: GenericParamList (defining parameters)
  INPUT:  T: Bound = Default
  OUTPUT: T: Bound = Default
  AST:    GenericParam::Type {
            name: T,
            bounds: [Bound],
            default: Some(Default)
          }

CONTEXT: GenericArgList (instantiating with arguments)
  INPUT:  Iterator<Item = T>
  OUTPUT: Iterator<Item = T>
  AST:    TypePath {
            path: Iterator,
            args: [AssocTypeBinding { name: Item, ty: T }]
          }
```

#### DISAMBIGUATION RULE

```
POSITION determines interpretation:

  After struct/fn/trait/impl name, before body:
    "=" following ":" → DefaultType (generic parameter)

  Inside <> when referencing a type:
    Ident "=" Type → AssocTypeBinding (generic argument)

These positions are syntactically unambiguous. No lookahead required.
```

### 3.2 Current Status: ✅ IMPLEMENTED

Parser correctly distinguishes these two constructs:
- `GenericParam::Type { default }` for default type parameters in declarations
- `TypeExpr::AssocTypeBinding` for associated type bindings in type arguments

Verified 2026-02-06: All 12 edge cases pass, generated Rust compiles successfully.

### 3.3 Required Implementation

```
PARSER (parser.rs):

  fn parse_generic_param():
    name = parse_ident()
    bounds = []
    default = None

    IF peek() == ":":
      consume(":")
      bounds = parse_bounds()  // NOT parse_type()!

    IF peek() == "=":
      consume("=")
      default = Some(parse_type())

    RETURN GenericParam::Type { name, bounds, default }

  fn parse_generic_arg():
    IF peek_second() == "=":  // Ident followed by =
      name = parse_ident()
      consume("=")
      ty = parse_type()
      RETURN AssocTypeBinding { name, ty }
    ELSE:
      RETURN parse_type()
```

### 3.4 Invariants

```
I₁: GenericParam NEVER contains AssocTypeBinding
I₂: AssocTypeBinding ONLY appears in GenericArgList
I₃: ":" in param position introduces bounds, not associated type
I₄: "=" in param position (after bounds) introduces default
I₅: "=" in arg position (after ident) introduces associated type binding
```

### 3.5 Edge Cases

```
EC₁: T: Trait = Default
     → GenericParam { bounds: [Trait], default: Default }

EC₂: T = Default  (no bounds)
     → GenericParam { bounds: [], default: Default }

EC₃: T: A + B = Default  (multiple bounds)
     → GenericParam { bounds: [A, B], default: Default }

EC₄: Iterator<Item = T>
     → TypePath with AssocTypeBinding

EC₅: Iterator<Item = T, Item2 = U>  (multiple assoc types)
     → TypePath with [AssocTypeBinding, AssocTypeBinding]

EC₆: HashMap<K, V, S = RandomState>  (mixed args and assoc)
     → TypePath with [TypeArg(K), TypeArg(V), AssocTypeBinding(S)]
```

### 3.6 Error Conditions

```
E₁: AssocTypeBinding in param position → PARSER ERROR
    "associated type binding not allowed in generic parameter list"

E₂: Default without "=" → PARSER ERROR
    "expected '=' before default type"

E₃: Bounds after default → PARSER ERROR
    "bounds must come before default type"
```

### 3.7 Composition

```
WITH Gap J (impl where clauses):
  ⊢<T = Default> Foo ∋ T: Bound
  Default in params, bounds in where clause. No interaction.

WITH Gap K (impl Trait):
  fn foo<T = impl Trait>()  // default is impl Trait
  Parse default as TypeExpr, which may be ImplTrait.
```

### 3.8 Priority

**P0** - Parser bug. Current workaround in codegen is tech debt that obscures the AST.

---

## 4. Gap C: Macro Invocation ✅ FIXED

### 4.1 Specification

#### GRAMMAR

```
MacroInvocation ::= MacroPath "!" DelimitedTokens
MacroPath       ::= Path
DelimitedTokens ::= "(" TokenStream ")"
                  | "[" TokenStream "]"
                  | "{" TokenStream "}"
TokenStream     ::= (Token | DelimitedTokens)*
Token           ::= Ident | Literal | Punct | Sigil
```

#### TRANSFORMATION

```
Expr::Macro { path, tokens, delimiter }:
  OUTPUT: emit_path(path) + "!" + delimiter.open + translate(tokens) + delimiter.close

TOKEN TRANSLATION (Sigil → Rust):
  "·"    → "::"
  "→"    → "->"
  "≔"    → "let"
  "vary" → "mut"
  "⎇"    → "if"
  "⌥"    → "match"
  "∀"    → "for"
  "∋"    → "where"
  "☉"    → "pub"
  "⊢"    → "impl"
  "◇"    → "dyn"
  "⤺"    → "return"
```

### 4.2 Current Status: ✅ IMPLEMENTED

Codegen emits macro invocations with proper delimiter preservation and token translation.

### 4.3 Implementation

```
emit_macro(m):
  emit_path(m.path)
  emit "!"

  // Determine delimiter
  delimiter = m.delimiter OR infer_from_tokens(m.tokens)

  IF tokens_bare(m.tokens):
    // Tokens don't include delimiters - add them
    emit delimiter.open
    emit translate_tokens(m.tokens)
    emit delimiter.close
  ELSE:
    // Tokens already delimited
    emit translate_tokens(m.tokens)

translate_tokens(tokens):
  result = ""
  FOR token IN tokens:
    IF token ∈ SIGIL_SYMBOLS:
      result += SYMBOL_MAP[token]
    ELSE IF token is DelimitedTokens:
      result += token.open + translate_tokens(token.inner) + token.close
    ELSE:
      result += token
  RETURN result
```

### 4.4 Invariants

```
I₁: Delimiter type is PRESERVED (parens stay parens, brackets stay brackets)
I₂: Nested delimiters are preserved structurally
I₃: ALL Sigil symbols in token stream are translated
I₄: Macro path uses "::" separator (always module path)
I₅: Token stream is otherwise passed through verbatim
```

### 4.5 Edge Cases

```
EC₁: vec![1, 2, 3]
     → vec![1, 2, 3]  (brackets preserved)

EC₂: println!("hello")
     → println!("hello")  (parens preserved)

EC₃: html! { <div> }
     → html! { <div> }  (braces preserved)

EC₄: assert_eq!(a, b)
     → assert_eq!(a, b)

EC₅: macro!()  (empty)
     → macro!()

EC₆: macro!(a,)  (trailing comma)
     → macro!(a,)

EC₇: outer!(inner!(x))  (nested macros)
     → outer!(inner!(x))

EC₈: vec![Foo·new()]  (Sigil path in macro)
     → vec![Foo::new()]

EC₉: format!("{} → {}", a, b)  (arrow in string - NO translation)
     → format!("{} → {}", a, b)

EC₁₀: quote! { ≔ x = 5 }  (Sigil in quote - translate)
      → quote! { let x = 5 }

EC₁₁: macro!(a, b, c,)  (multiple args with trailing)
      → macro!(a, b, c,)

EC₁₂: cfg!(target_os = "linux")
      → cfg!(target_os = "linux")
```

### 4.6 Error Conditions

```
E₁: Unbalanced delimiters in token stream
    → PARSER ERROR (caught at parse time, not codegen)

E₂: Unknown Sigil symbol in tokens
    → Pass through unchanged (may cause Rust error)
```

### 4.7 Composition

```
WITH Gap H (paths):
  std·println!("x")
  → std::println!("x")
  Macro path resolution uses same rules as other paths.

WITH statement position:
  assert!(cond);
  → assert!(cond);  (semicolon added by statement emission)

WITH expression position:
  ≔ v = vec![1, 2, 3];
  → let v = vec![1, 2, 3];
```

### 4.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_macros.sg`

**Required test cases:**
- Assertion macros (assert!, assert_eq!, assert_ne!)
- Collection macros (vec!, hashmap!)
- Format macros (format!, println!, eprintln!)
- Nested macros
- Macros with Sigil symbols in token stream
- All three delimiter types

---

## 5. Gap D: Format Macros ✅ FIXED

### 5.1 Specification

Format macros are a specialized subset of macro invocations with distinct semantics.

#### FORMAT MACRO FAMILY

```
WRITE FAMILY (to fmt::Formatter or impl Write):
  write!(dest, fmt, args...)     → Result<(), fmt::Error>
  writeln!(dest, fmt, args...)   → Result<(), fmt::Error>

PRINT FAMILY (to stdout/stderr):
  print!(fmt, args...)           → ()
  println!(fmt, args...)         → ()
  eprint!(fmt, args...)          → ()
  eprintln!(fmt, args...)        → ()

STRING FAMILY (returns String):
  format!(fmt, args...)          → String
  format_args!(fmt, args...)     → fmt::Arguments

PANIC FAMILY (diverges):
  panic!(fmt, args...)           → !
  unreachable!(fmt, args...)     → !
  todo!(fmt, args...)            → !
  unimplemented!(fmt, args...)   → !
```

#### FORMAT STRING GRAMMAR

```
FormatString  ::= (Literal | Placeholder)*
Placeholder   ::= "{" Argument? (":" FormatSpec)? "}"
Argument      ::= Integer | Identifier
FormatSpec    ::= Fill? Align? Sign? "#"? "0"? Width? Precision? Type?
Fill          ::= Character
Align         ::= "<" | "^" | ">"
Sign          ::= "+" | "-"
Width         ::= Integer | Argument "$"
Precision     ::= "." (Integer | Argument "$" | "*")
Type          ::= "?" | "x?" | "X?" | "o" | "x" | "X" | "p" | "b" | "e" | "E"
```

#### TRANSFORMATION

```
Format macros use Gap C transformation, PLUS:

1. Format string is preserved verbatim (no symbol translation inside "...")
2. Arguments after format string use standard token translation
3. Try operator (?) on write!/writeln! is preserved

write!(f, "{:?}", expr)   → write!(f, "{:?}", expr)
write!(f, "{}", x·y)?     → write!(f, "{}", x::y)?    // path translated, ? preserved
format!("{:#x}", n)       → format!("{:#x}", n)
```

### 5.2 Current Status: ✅ IMPLEMENTED

Handled by Gap C's `Expr::Macro` implementation. Format-specific semantics preserved.

### 5.3 Implementation

```
emit_format_macro(m):
  // Format macros are a subset of Gap C macro handling
  // Key difference: format string content is NOT translated

  emit_path(m.path)  // write, writeln, format, etc.
  emit "!"
  emit "("

  FOR i, arg IN enumerate(m.args):
    IF i > 0: emit ", "

    IF i == 0 AND m.path ∈ WRITE_FAMILY:
      // First arg is destination (Formatter, Write impl)
      emit_expr(arg)
    ELSE IF is_string_literal(arg):
      // Format string - preserve verbatim, NO translation
      emit arg.raw
    ELSE:
      // Expression argument - translate Sigil symbols
      emit translate_tokens(arg)

  emit ")"

  // Preserve try operator if present
  IF m.has_try_operator:
    emit "?"
```

### 5.4 Invariants

```
I₁: Format string content is NEVER translated (symbols inside "" stay as-is)
I₂: Expression arguments ARE translated per Gap C rules
I₃: Try operator (?) is preserved for write!/writeln!
I₄: All format macros use parentheses delimiter
I₅: Trailing comma in args is preserved
```

### 5.5 Edge Cases

```
EC₁: Basic write
     write!(f, "{}", value)
     → write!(f, "{}", value)

EC₂: Debug format
     write!(f, "{:?}", obj)
     → write!(f, "{:?}", obj)

EC₃: Pretty debug
     write!(f, "{:#?}", obj)
     → write!(f, "{:#?}", obj)

EC₄: Hex format
     write!(f, "{:#x}", num)
     → write!(f, "{:#x}", num)

EC₅: Precision
     write!(f, "{:.2}", float)
     → write!(f, "{:.2}", float)

EC₆: Width and alignment
     write!(f, "{:<10}", s)
     → write!(f, "{:<10}", s)

EC₇: Named arguments
     format!("{name} = {value}", name = n, value = v)
     → format!("{name} = {value}", name = n, value = v)

EC₈: Positional arguments
     format!("{0} {1} {0}", a, b)
     → format!("{0} {1} {0}", a, b)

EC₉: Sigil path in argument
     write!(f, "{}", Foo·bar())
     → write!(f, "{}", Foo::bar())

EC₁₀: Try operator
      write!(f, "{}", x)?
      → write!(f, "{}", x)?

EC₁₁: Multiple format specs
      write!(f, "{:0>8x}", n)
      → write!(f, "{:0>8x}", n)

EC₁₂: Arrow in format string (NOT translated)
      format!("a → b")
      → format!("a → b")

EC₁₃: Empty format
      println!()
      → println!()

EC₁₄: Escaped braces
      format!("{{literal}}")
      → format!("{{literal}}")

EC₁₅: Formatter methods
      f.write_str("text")?
      → f.write_str("text")?  // Not a macro, but related
```

### 5.6 Error Conditions

```
E₁: Mismatched format args (wrong count)
    → Rust compiler error (not codegen's responsibility)

E₂: Invalid format spec
    → Rust compiler error

E₃: Type doesn't implement required trait (Debug, Display, etc.)
    → Rust compiler error
```

### 5.7 Composition

```
WITH Gap C (macros):
  Format macros are a SUBSET of Gap C
  Same Expr::Macro handler, same token translation
  Only difference: format strings preserved verbatim

WITH trait implementations:
  impl Display for Foo {
      fn fmt(&self, f: &mut Formatter) → fmt::Result {
          write!(f, "Foo({})", self.0)
      }
  }

WITH error propagation:
  write!(f, "{}", x)?  // ? propagates fmt::Error
```

### 5.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_format_macros.sg`

**Required test cases:**
- write!, writeln!, format!, println!
- All format specifiers ({:?}, {:#x}, {:.2}, etc.)
- Named and positional arguments
- Sigil symbols in arguments (verify translation)
- Sigil symbols in format string (verify NO translation)
- Try operator preservation

---

## 6. Gap E: Visibility Modifiers ✅ FIXED

### 6.1 Specification

#### GRAMMAR

```
Visibility      ::= "☉"                    // pub
                  | "☉" "(" VisScope ")"   // pub(scope)
                  | ε                       // private (default)

VisScope        ::= "crate"                // pub(crate)
                  | "super"                // pub(super)
                  | "self"                 // pub(self) = private
                  | "in" Path              // pub(in path)
```

#### TRANSFORMATION

```
VISIBILITY MAPPING:
  (none)           → (none)         // private
  ☉                → pub
  ☉(crate)         → pub(crate)
  ☉(super)         → pub(super)
  ☉(self)          → pub(self)
  ☉(in path)       → pub(in path)

APPLIES TO:
  - Struct/Enum/Union declarations
  - Struct fields (named)
  - Enum variants (always pub if enum is pub)
  - Function declarations
  - Const/Static declarations
  - Type aliases
  - Trait declarations
  - Impl items
  - Module declarations
```

### 6.2 Current Status: ✅ IMPLEMENTED

Codegen correctly emits visibility for all item types.

### 6.3 Implementation

```
emit_visibility(vis):
  MATCH vis:
    None:
      // emit nothing (private)

    Public:
      emit "pub "

    Restricted { scope }:
      emit "pub("
      emit_vis_scope(scope)
      emit ") "

emit_vis_scope(scope):
  MATCH scope:
    Crate:  emit "crate"
    Super:  emit "super"
    Self:   emit "self"
    In(p):  emit "in " + emit_path(p)
```

### 6.4 Invariants

```
I₁: Default visibility is private (no output)
I₂: ☉ ALWAYS emits "pub "
I₃: Visibility precedes item keyword
I₄: Struct fields inherit nothing - each field has own visibility
I₅: Enum variants are implicitly pub if enum is pub (Rust rule)
```

### 6.5 Edge Cases

```
EC₁: ☉ sigil Foo { }
     → pub struct Foo { }

EC₂: sigil Foo { }  (no visibility)
     → struct Foo { }

EC₃: ☉ sigil Foo { ☉ x: i32, y: i32 }
     → pub struct Foo { pub x: i32, y: i32 }

EC₄: ☉(crate) rite foo() { }
     → pub(crate) fn foo() { }

EC₅: ☉(super) sigil Bar { }
     → pub(super) struct Bar { }

EC₆: ☉(in crate·module) rite baz() { }
     → pub(in crate::module) fn baz() { }

EC₇: Tuple struct fields
     ☉ sigil Point(☉ i32, i32);
     → pub struct Point(pub i32, i32);

EC₈: Enum variants (visibility not per-variant in Rust)
     ☉ choice Color { Red, Green, Blue }
     → pub enum Color { Red, Green, Blue }
```

### 6.6 Error Conditions

```
E₁: Visibility on enum variant
    → WARNING: visibility on enum variant is ignored (Rust doesn't support)

E₂: pub(in path) where path doesn't exist
    → Rust compiler error (not codegen's responsibility)
```

### 6.7 Composition

```
WITH structs:
  Struct visibility + field visibility are independent

WITH traits:
  Trait items are implicitly pub (Rust rule)
  ☉ on trait item → WARNING: unnecessary

WITH impl blocks:
  Inherent impl items: visibility respected
  Trait impl items: implicitly pub
```

### 6.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_public_fields.sg`

**Required test cases:**
- Private struct, private fields
- Public struct, mixed field visibility
- pub(crate), pub(super), pub(in path)
- Tuple struct field visibility
- Enum visibility (variants auto-pub)

---

## 7. Gap F: Where Clause Emission ✅ FIXED

### 7.1 Specification

#### GRAMMAR

```
WhereClause     ::= "∋" WherePredicate ("," WherePredicate)*
WherePredicate  ::= Type ":" Bounds
                  | Lifetime ":" LifetimeBounds
                  | ForLifetimes Type ":" Bounds

Bounds          ::= Bound ("+" Bound)*
Bound           ::= TypePath | Lifetime | "?" TypePath

LifetimeBounds  ::= Lifetime ("+" Lifetime)*
ForLifetimes    ::= "for" "<" LifetimeParam ("," LifetimeParam)* ">"
```

#### TRANSFORMATION

```
WhereClause { predicates }:
  IF predicates.empty:
    OUTPUT: (nothing)
  ELSE:
    OUTPUT: "\nwhere " + predicates.join(",\n      ")

WherePredicate { ty, bounds }:
  OUTPUT: emit_type(ty) + ": " + bounds.join(" + ")

WherePredicate { lifetime, bounds }:
  OUTPUT: lifetime + ": " + bounds.join(" + ")

ForLifetimes { lifetimes } + WherePredicate:
  OUTPUT: "for<" + lifetimes.join(", ") + "> " + predicate
```

### 7.2 Current Status: ✅ IMPLEMENTED

Codegen emits where clauses for functions. Gap J tracks impl block where clauses.

### 7.3 Implementation

```
emit_where_clause(wc):
  IF wc.predicates.empty:
    RETURN

  emit "\n"
  emit_indent()
  emit "where "

  FOR i, pred IN enumerate(wc.predicates):
    IF i > 0:
      emit ",\n"
      emit_indent()
      emit "      "  // align with first predicate

    emit_where_predicate(pred)

emit_where_predicate(pred):
  IF pred.for_lifetimes:
    emit "for<"
    emit pred.for_lifetimes.join(", ")
    emit "> "

  emit_type(pred.ty)
  emit ": "

  FOR j, bound IN enumerate(pred.bounds):
    IF j > 0:
      emit " + "
    emit_bound(bound)

emit_bound(bound):
  MATCH bound:
    TypePath(p):     emit_type_path(p)
    Lifetime(lt):    emit lt
    Maybe(p):        emit "?" + emit_type_path(p)
```

### 7.4 Invariants

```
I₁: Empty where clause emits nothing
I₂: Where clause appears AFTER return type, BEFORE body
I₃: Multiple predicates separated by ",\n" with alignment
I₄: Multiple bounds on same type separated by " + "
I₅: Lifetime bounds use same syntax as trait bounds
I₆: for<'a> higher-ranked bounds preserved
```

### 7.5 Edge Cases

```
EC₁: Single predicate
     ∋ T: Clone
     → where T: Clone

EC₂: Multiple predicates
     ∋ T: Clone, U: Debug
     → where T: Clone,
             U: Debug

EC₃: Multiple bounds on single type
     ∋ T: Clone + Debug + Send
     → where T: Clone + Debug + Send

EC₄: Lifetime bound
     ∋ T: 'static
     → where T: 'static

EC₅: Lifetime outlives lifetime
     ∋ 'a: 'b
     → where 'a: 'b

EC₆: Type outlives lifetime
     ∋ T: 'a
     → where T: 'a

EC₇: Higher-ranked trait bound (HRTB)
     ∋ F: for<'a> Fn(&'a T) → &'a U
     → where F: for<'a> Fn(&'a T) -> &'a U

EC₈: Maybe bound (relaxed)
     ∋ T: ?Sized
     → where T: ?Sized

EC₉: Complex mixed bounds
     ∋ T: Clone + 'static, U: for<'a> Fn(&'a T) → U + Send
     → where T: Clone + 'static,
             U: for<'a> Fn(&'a T) -> U + Send

EC₁₀: Associated type constraint in bound
      ∋ I: Iterator<Item = T>
      → where I: Iterator<Item = T>

EC₁₁: No where clause
      (no ∋)
      → (nothing emitted)
```

### 7.6 Error Conditions

```
E₁: Predicate with no bounds
    ∋ T:
    → PARSER ERROR: expected bound after ':'

E₂: Duplicate predicate for same type
    ∋ T: Clone, T: Debug
    → VALID (Rust allows, equivalent to T: Clone + Debug)
```

### 7.7 Composition

```
WITH functions:
  ∋ appears after ")" and before "{"

WITH structs:
  sigil Foo<T> ∋ T: Clone { }
  → struct Foo<T> where T: Clone { }

WITH impl blocks (Gap J):
  Currently NOT SUPPORTED at impl level
  Workaround: use method-level where clauses

WITH trait definitions:
  trait Foo<T> ∋ T: Clone { }
  → trait Foo<T> where T: Clone { }

WITH Gap G (Fn traits):
  Where bounds may contain Fn traits
  ∋ F: Fn(T) → U
  → where F: Fn(T) -> U
```

### 7.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_where_clauses.sg`

**Required test cases:**
- Single predicate
- Multiple predicates
- Multiple bounds (A + B + C)
- Lifetime bounds
- Higher-ranked trait bounds (for<'a>)
- Maybe bounds (?Sized)
- Associated type constraints

---

## 8. Gap G: Fn Trait Syntax ✅ FIXED

### 8.1 Specification

#### GRAMMAR (Sigil AST)

```
FnTraitType  ::= FnTraitName "<" ArgsTuple "," ReturnType ">"
FnTraitName  ::= "Fn" | "FnMut" | "FnOnce"
ArgsTuple    ::= "(" Type ("," Type)* ")" | "()"
ReturnType   ::= Type
```

#### OUTPUT GRAMMAR (Rust)

```
FnTraitType  ::= FnTraitName "(" Args ")" "->" Return
               | FnTraitName "(" Args ")"                   // if return is ()
Args         ::= Type ("," Type)* | ε
Return       ::= Type
```

#### TRANSFORMATION

```
Fn<(A, B, C), R>   → Fn(A, B, C) -> R
FnMut<(A, B), R>   → FnMut(A, B) -> R
FnOnce<(A,), R>    → FnOnce(A) -> R
Fn<(), R>          → Fn() -> R
Fn<(A,), ()>       → Fn(A)                    // unit return omitted
Fn<(), ()>         → Fn()                     // thunk

WITH additional bounds:
  Fn<(A,), B> + Send     → Fn(A) -> B + Send
  Fn<(A,), B> + 'static  → Fn(A) -> B + 'static
```

### 8.2 Current Status: ✅ IMPLEMENTED

Codegen recognizes Fn/FnMut/FnOnce and emits parenthetical syntax.

### 8.3 Implementation

```
emit_type_path(path):
  name = path.last_segment()
  args = path.generic_args()

  IF name ∈ ["Fn", "FnMut", "FnOnce"] AND args.len() >= 1:
    emit name
    emit "("

    // First arg is tuple of parameter types
    param_types = unwrap_tuple(args[0])
    emit param_types.join(", ")

    emit ")"

    // Second arg (if present and not unit) is return type
    IF args.len() >= 2 AND args[1] != UnitType:
      emit " -> "
      emit_type(args[1])

  ELSE:
    emit_generic_type_path(path)  // normal handling

unwrap_tuple(ty):
  MATCH ty:
    Tuple(types):  RETURN types
    other:         RETURN [other]  // single param not wrapped
```

### 8.4 Invariants

```
I₁: Fn/FnMut/FnOnce ALWAYS use parenthetical syntax in Rust
I₂: Angle-bracket syntax Fn<Args, Ret> is INVALID Rust
I₃: Unit return type "()" MAY be omitted
I₄: Additional bounds appear AFTER the Fn type
I₅: dyn Fn(...) uses same parenthetical syntax
I₆: impl Fn(...) uses same parenthetical syntax (Gap K)
```

### 8.5 Edge Cases

```
EC₁: Zero-arg function
     Fn<(), R>
     → Fn() -> R

EC₂: Single-arg function
     Fn<(T,), R>      // Note: single-element tuple
     → Fn(T) -> R

EC₃: Multi-arg function
     Fn<(A, B, C), R>
     → Fn(A, B, C) -> R

EC₄: Unit return (thunk-ish)
     Fn<(T,), ()>
     → Fn(T)          // or Fn(T) -> (), both valid

EC₅: FnMut
     FnMut<(T,), T>
     → FnMut(T) -> T

EC₆: FnOnce
     FnOnce<(T,), U>
     → FnOnce(T) -> U

EC₇: With Send bound
     Fn<(T,), U> + Send
     → Fn(T) -> U + Send

EC₈: With lifetime bound
     Fn<(T,), U> + 'static
     → Fn(T) -> U + 'static

EC₉: With multiple bounds
     Fn<(T,), U> + Send + Sync + 'static
     → Fn(T) -> U + Send + Sync + 'static

EC₁₀: dyn Fn
      dyn Fn<(i32,), i32>
      → dyn Fn(i32) -> i32

EC₁₁: Box<dyn Fn>
      Box<dyn Fn<(i32,), i32>>
      → Box<dyn Fn(i32) -> i32>

EC₁₂: Higher-ranked (for<'a>)
      for<'a> Fn<(&'a T,), &'a U>
      → for<'a> Fn(&'a T) -> &'a U

EC₁₃: Nested Fn types
      Fn<(Fn<(i32,), i32>,), i32>
      → Fn(Fn(i32) -> i32) -> i32

EC₁₄: Generic return
      Fn<(T,), T>
      → Fn(T) -> T

EC₁₅: Reference parameters
      Fn<(&T, &vary U), V>
      → Fn(&T, &mut U) -> V
```

### 8.6 Error Conditions

```
E₁: Fn with no type args
    Fn  (bare)
    → VALID as trait bound: T: Fn (but unusual)

E₂: Fn with only one type arg
    Fn<(A, B)>  (no return type)
    → Infer unit return: Fn(A, B)

E₃: Fn with > 2 type args
    Fn<A, B, C>
    → ERROR: malformed Fn trait (parser should reject)
```

### 8.7 Composition

```
WITH Gap F (where clauses):
  ∋ F: Fn(T) → U
  Fn trait in where clause uses same syntax

WITH Gap K (impl Trait):
  → impl Fn(T) → U
  Return-position impl Fn uses same syntax

WITH Gap J (impl blocks):
  No direct interaction (impl blocks can't impl Fn traits directly)

WITH dyn:
  Box<◇ Fn(T) → U>
  → Box<dyn Fn(T) -> U>
```

### 8.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_fn_traits.sg`

**Required test cases:**
- Fn, FnMut, FnOnce variants
- Zero, one, multiple parameters
- Unit vs non-unit return
- With additional bounds (+ Send, + 'static)
- dyn Fn
- Box<dyn Fn>
- Nested Fn types
- Higher-ranked Fn (for<'a>)

---

## 9. Gap H: Path Separator Resolution ✅ IMPLEMENTED

### 9.1 Specification

#### GRAMMAR

```
Path        ::= Segment (PathSep Segment)*
PathSep     ::= "·"                          // Sigil universal separator
Segment     ::= Ident | Keyword
Keyword     ::= "self" | "super" | "crate" | "Self"
```

#### OUTPUT GRAMMAR (Rust)

```
RustPath    ::= Segment (RustSep Segment)*
RustSep     ::= "::" | "."
```

#### TRANSFORMATION

```
RULE: Path separator is determined by SEMANTIC CATEGORY, not lexical heuristics.

CATEGORIES:
  ModulePath  → "::"    (navigating module tree)
  TypePath    → "::"    (referencing types, traits, associated items)
  ValuePath   → "."     (accessing fields, calling methods)

CONTEXT DETERMINES CATEGORY:

  UseStatement:
    invoke P          → use P;              (always "::")

  TypeAnnotation:
    x: P              → x: P                (always "::")

  Expression (complex):
    P                 → depends on what P refers to

EXPRESSION RESOLUTION:
  LET first = path[0]

  IF first ∈ KEYWORDS:
    "self"  → module path if followed by "::", value if terminal
    "super" → always "::"
    "crate" → always "::"
    "Self"  → always "::" (type, not value)

  IF first ∈ KNOWN_CRATES:
    → "::" (module path)

  IF first is_uppercase_initial:
    → "::" (type or module by convention)

  IF first ∈ LOCAL_BINDINGS:
    → "." (value path, field/method access)

  ELSE:
    → REQUIRES SEMANTIC INFO (see 9.3)
```

### 9.2 Current Status: ✅ IMPLEMENTED

**Implementation Date:** 2026-02-06

Local binding tracking added to `RustCompiler`:
- `local_bindings: HashSet<String>` field tracks bindings in scope
- `collect_pattern_bindings()` helper extracts names from patterns
- Function params added to bindings at function entry
- Let/LetElse statements add bindings when encountered
- Path emission checks `local_bindings` first before applying heuristics

**Test File:** `jormungandr/tests/rust_codegen/test_paths.sg`

```
// Input (Sigil):
≔ Δ vec = Vec·new();
vec·push(1);           // vec is local binding

// Output (Rust):
let mut vec = Vec::new();
vec.push(1);           // Correct: uses "." for method call
```

### 9.3 Implementation Details

#### PHASE 1: Static Resolution (covers ~90% of cases)

```
CODEGEN INITIALIZATION:
  known_crates = {
    "std", "core", "alloc",           // Rust built-ins
    "self", "super", "crate",          // Relative paths
  }

  IF exists("Sigil.toml"):
    workspace = parse_sigil_toml()
    FOR member IN workspace.members:
      known_crates.add(member.name)
    FOR dep IN workspace.dependencies:
      known_crates.add(dep.name)

EMIT PATH:
  fn emit_path(path, context):
    IF context == UseStatement:
      emit_with_separator(path, "::")

    ELSE IF context == TypeAnnotation:
      emit_with_separator(path, "::")

    ELSE IF context == Expression:
      first = path[0]

      IF first ∈ known_crates:
        emit_with_separator(path, "::")

      ELSE IF is_type_like(first):  // PascalCase
        emit_with_separator(path, "::")

      ELSE IF is_definitely_value(first):  // in local scope
        emit_first_then_dots(path)

      ELSE:
        // Ambiguous - need type info
        emit_with_annotation(path)  // or error
```

#### PHASE 2: Type-Informed Resolution (future)

```
IF typechecker available:
  ty = typecheck(path[0])
  IF ty.is_module() OR ty.is_type():
    → "::"
  ELSE:
    → "."
```

### 9.4 Invariants

```
I₁: Use statements ALWAYS use "::"
I₂: Type annotations ALWAYS use "::"
I₃: "super", "crate" ALWAYS use "::"
I₄: PascalCase identifiers ALWAYS use "::" (type convention)
I₅: Known local bindings ALWAYS use "." for subsequent segments
I₆: No hardcoded external crate names in codegen source
```

### 9.5 Edge Cases

```
EC₁: std·collections·HashMap
     Context: Type → std::collections::HashMap

EC₂: self·field
     Context: Expr, self is value → self.field

EC₃: self·module·Type
     Context: Type → self::module::Type

EC₄: foo·bar·baz
     foo is local binding → foo.bar.baz
     foo is crate → foo::bar::baz
     foo is ambiguous → ERROR or annotation required

EC₅: T·default()
     T is type param → T::default()

EC₆: x·y·z  where x: SomeStruct
     → x.y.z  (field access chain)

EC₇: Option·Some(v)
     → Option::Some(v)  (enum constructor)

EC₈: Vec·new()
     → Vec::new()  (associated function)

EC₉: vec·push(x)  where vec: Vec<T>
     → vec.push(x)  (method call)
```

### 9.6 Error Conditions

```
E₁: Ambiguous path without type info
    → ERROR: "cannot determine path separator for '{path}'; add type annotation"

E₂: Invalid path (e.g., super·super·super beyond root)
    → ERROR: "path escapes module root"
```

### 9.7 Composition

```
WITH Gap B (generics):
  HashMap<K, V, S = RandomState>
  All type args use "::" for their internal paths

WITH Gap F (where clauses):
  ∋ T: some_crate·SomeTrait
  Trait path uses "::"

WITH Gap K (impl Trait):
  → impl some·Trait
  Trait path uses "::"
```

### 9.8 Migration

```
REMOVE from codegen:
  - Hardcoded module_roots list
  - Hardcoded std_submodules list
  - Hardcoded nihil_* crate names

ADD to codegen:
  - Sigil.toml parsing at initialization
  - Context parameter to emit_path
  - Semantic category inference
```

### 9.9 Priority

**P0** - Current implementation is a hack. Every new crate requires editing codegen.

---

## 10. Gap I: Raw Pointer Type Annotations ✅ OPTION A IMPLEMENTED

### 10.1 Specification

#### THE PROBLEM

Rust requires pointee type to be known at certain raw pointer operations:

```
REQUIRES TYPE ANNOTATION:
  .add(n)       → computes offset as n * size_of::<T>()
  .sub(n)       → computes offset as n * size_of::<T>()
  .offset(n)    → computes offset as n * size_of::<T>()
  .read()       → needs to know what type to read
  .write(v)     → needs to know layout for write
  .read_volatile()
  .write_volatile(v)
  .copy_to(dest, count)
  .copy_from(src, count)

DOES NOT REQUIRE ANNOTATION:
  .is_null()    → works on *T for any T
  .cast::<U>()  → explicit target type
  as *const U   → explicit target type
  as *mut U     → explicit target type
```

#### GRAMMAR

```
PointerTypeAnnotation ::= "*" Mutability Type
Mutability            ::= "◆" | "vary"      // const | mut

LetBinding ::= "≔" Pattern (":" Type)? "=" Expr
```

#### TRANSFORMATION

```
WHEN type annotation present:
  ≔ ptr: *◆ T = expr     → let ptr: *const T = expr
  ≔ ptr: *vary T = expr  → let ptr: *mut T = expr

WHEN type annotation absent:
  ≔ ptr = expr           → let ptr = expr

IDEAL BEHAVIOR (if type info available):
  IF binding_has_no_annotation AND expr_type_is_raw_pointer:
    inferred_type = typechecker.get_type(expr)
    emit "let ptr: " + inferred_type + " = expr"
```

### 10.2 Current Status: ✅ OPTION A IMPLEMENTED

**Implementation Date:** 2026-02-06

Option A (source annotations) works correctly:
- `*◆ T` → `*const T`
- `*vary T` or `*Δ T` → `*mut T`
- Explicit type annotations preserved in output

**Test File:** `jormungandr/tests/rust_codegen/test_raw_pointers.sg`

**Note:** Option B (typechecker-informed codegen) is deferred - requires architectural changes to thread type information from typechecker to codegen.

### 10.3 Design Options

#### OPTION A: Source Annotations (Current)

```
APPROACH:
  Require Sigil source to include explicit type annotations
  where Rust's inference fails.

PROS:
  - No codegen changes required
  - Source is explicit about intent
  - No dependency on typechecker

CONS:
  - Sigil source becomes verbose
  - Knowledge of Rust's inference limitations leaks into Sigil
  - Valid Sigil (runs in interpreter) may fail to compile
```

#### OPTION B: Typechecker-Informed Codegen

```
APPROACH:
  Run typechecker before codegen.
  Thread type information through to code emission.
  Emit annotations where Rust needs them.

REQUIRES:
  1. Typechecker produces typed AST (or type map)
  2. Codegen receives type information
  3. Codegen detects "Rust needs annotation here" patterns
  4. Codegen emits inferred types

PROS:
  - Sigil source stays clean
  - Valid Sigil always compiles
  - Single source of truth (typechecker)

CONS:
  - Requires architectural change
  - Typechecker must run for codegen
  - Adds complexity to codegen

DETECTION HEURISTIC:
  For let bindings without annotation:
    IF rhs returns *const T or *mut T:
      IF lhs is used with .add/.sub/.offset/.read/.write:
        EMIT type annotation
```

#### OPTION C: Conservative Annotation Emission

```
APPROACH:
  Always emit type annotations for raw pointer bindings,
  even if Sigil source doesn't have them.

REQUIRES:
  1. Detect raw pointer types syntactically (limited)
  2. OR require typechecker (same as Option B)

LIMITED VERSION (no typechecker):
  IF rhs is method call on known pointer-returning methods:
    (.as_ptr(), .as_mut_ptr(), .data_ptr(), etc.)
    AND lhs has no annotation:
      EMIT lhs: *const _ = rhs  // let Rust infer pointee

PROBLEM:
  *const _ and *mut _ don't help - Rust still can't infer T
```

### 10.4 Recommendation

**Option B (Typechecker-Informed Codegen)** is the correct long-term solution.

However, this requires architectural work. Until then, **Option A (Source Annotations)** is the pragmatic choice with clear documentation.

### 10.5 Implementation (Option A - Current)

```
DOCUMENTATION REQUIRED:
  When targeting Rust codegen, raw pointer bindings used with
  .add(), .offset(), .read(), .write() MUST include explicit
  type annotations in Sigil source.

VALID:
  ≔ ptr: *◆ i32 = slice.as_ptr()
  unsafe { *ptr.add(i) }

INVALID (compiles in interpreter, fails in Rust):
  ≔ ptr = slice.as_ptr()
  unsafe { *ptr.add(i) }  // E0282: type annotations needed
```

### 10.6 Invariants

```
I₁: Explicit type annotations are ALWAYS emitted
I₂: Unannotated bindings emit no type annotation
I₃: *◆ T → *const T
I₄: *vary T → *mut T
I₅: Rust compiler errors for missing annotations are expected (Option A)
```

### 10.7 Edge Cases

```
EC₁: Annotated const pointer
     ≔ ptr: *◆ i32 = data.as_ptr()
     → let ptr: *const i32 = data.as_ptr()

EC₂: Annotated mut pointer
     ≔ ptr: *vary i32 = data.as_mut_ptr()
     → let ptr: *mut i32 = data.as_mut_ptr()

EC₃: Generic pointee type
     ≔ ptr: *◆ T = slice.as_ptr()
     → let ptr: *const T = slice.as_ptr()

EC₄: Unannotated (currently fails in Rust)
     ≔ ptr = data.as_ptr()
     → let ptr = data.as_ptr()
     // Rust error if .add() called later

EC₅: Pointer from struct method
     ≔ ptr: *◆ T = buffer.as_ptr()
     → let ptr: *const T = buffer.as_ptr()

EC₆: Pointer cast (no annotation needed)
     ≔ bytes = ptr as *◆ u8
     → let bytes = ptr as *const u8

EC₇: Null pointer (no annotation needed for is_null)
     ≔ ptr = data.as_ptr()
     ⎇ ptr.is_null() { ... }  // OK, is_null works without type

EC₈: Nested pointer
     ≔ ptr: *◆ *◆ i32 = ...
     → let ptr: *const *const i32 = ...
```

### 10.8 Error Conditions

```
E₁: Unannotated pointer used with .add()
    → Rust error E0282 (expected, document in migration guide)

E₂: Mismatched mutability
    ≔ ptr: *vary T = slice.as_ptr()  // as_ptr returns *const
    → Rust error (type mismatch)

E₃: Wrong pointee type
    ≔ ptr: *◆ i64 = slice_of_i32.as_ptr()
    → Rust error (type mismatch)
```

### 10.9 Composition

```
WITH unsafe blocks:
  Pointer arithmetic always requires unsafe
  unsafe { *ptr.add(i) }

WITH generics:
  Generic pointee types work: *◆ T where T is type parameter

WITH for loops:
  Common pattern:
  ∀ i ∈ 0..len {
      unsafe { *ptr.add(i) }
  }
```

### 10.10 Future Work (Option B)

```
PHASE 1: Type Map
  - Typechecker produces Map<ExprId, Type>
  - Codegen receives type map as input

PHASE 2: Annotation Detection
  - Identify bindings where Rust needs annotation
  - Query type map for inferred type

PHASE 3: Annotation Emission
  - Emit type annotation for identified bindings
  - Verify generated code compiles

SUCCESS CRITERIA:
  Sigil source without explicit pointer annotations
  compiles successfully to Rust.
```

### 10.11 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_raw_pointers.sg`

**Required test cases:**
- Annotated const pointer with .add()
- Annotated mut pointer with .add()
- Generic pointee type
- Pointer cast (verify no annotation needed)
- is_null check (verify no annotation needed)
- Nested pointers

### 10.12 Priority

**P1** - Source annotations work but are verbose. Option B is desirable but requires architectural changes.

---

## 11. Additional Codegen Improvements (v0.3.0)

### 11.1 Macro Parentheses

**Issue:** Macro tokens didn't include delimiters, producing `assert_eq!data.len()`.

**Fix:** Auto-wrap bare tokens with parentheses: `assert_eq!(data.len())`.

### 11.2 Symbol Translation in Macros

**Issue:** Macro tokens contained raw Sigil symbols like `S·RANK`.

**Fix:** Translate symbols before emission:
- `·` → `::`
- `Δ` → `mut`
- `→` → `->`

### 11.3 Cast Expression Parentheses

**Issue:** `x as f64.sqrt()` is invalid Rust (cast binds tighter than method call).

**Fix:** Wrap casts in parentheses: `(x as f64).sqrt()`.

### 11.4 Primitive Type Path Detection

**Issue:** `f32·from_bits()` emitted `f32.from_bits()` (dot instead of `::`) because `f32` starts lowercase.

**Fix:** Added primitive types (`f32`, `i32`, `u8`, etc.) to the type path detection heuristic.

---

## 12. Implementation Priority

All specs are at **S++ quality** (formal grammar, transformation rules, exhaustive edge cases).

| Priority | Gap | Spec | Impl | Impact |
|----------|-----|------|------|--------|
| P0 | Gap A: Ref binding patterns | ✅ S++ | ✅ | Destructuring with ref/ref mut |
| **P0** | **Gap B: Generic params** | ✅ S++ | ❌ | Parser conflates default params with assoc types |
| P0 | Gap C: Macro invocation | ✅ S++ | ✅ | Macro token translation |
| P0 | Gap D: Format macros | ✅ S++ | ✅ | println!, format!, etc. |
| P0 | Gap E: Visibility modifiers | ✅ S++ | ✅ | pub, pub(crate), pub(super) |
| P0 | Gap F: Where clauses | ✅ S++ | ✅ | Where clause emission |
| P0 | Gap G: Fn trait syntax | ✅ S++ | ✅ | Fn/FnMut/FnOnce parenthetical |
| **P0** | **Gap H: Path resolution** | ✅ S++ | ❌ | Hardcoded crate lists don't scale |
| P1 | Gap I: Raw pointers | ✅ S++ | ⚠️ | Source annotations (Option B future) |
| **P1** | **Gap J: Impl where clauses** | ✅ S++ | ❌ | Parser missing where_clause field |
| **P1** | **Gap K: impl Trait returns** | ✅ S++ | ❌ | Parser rejects impl in type position |
| P0 | Gap L: Return statement | ✅ S++ | ✅ | Early returns are fundamental |

**Legend:**
- ✅ S++ Spec: Formal grammar, transformation rules, invariants, exhaustive edge cases
- ✅ Impl: Codegen handles correctly
- ⚠️ Impl: Partial/workaround (requires source-level fix)
- ❌ Impl: Not implemented (requires parser/codegen work)

---

## 13. Test Files

### 13.1 Existing Coverage

| Test File | Status | Coverage |
|-----------|--------|----------|
| `test_primitives.sg` | ✅ | Functions, primitives, let bindings |
| `test_structs.sg` | ✅ | Structs, generics, impl blocks |
| `test_traits.sg` | ✅ | Traits, trait impls |
| `test_morphemes.sg` | ✅ | Pipe operators, iterator chains |
| `test_const_generics.sg` | ✅ | Const generic parameters |
| `test_evidence.sg` | ✅ | Evidentiality markers |
| `test_async.sg` | ✅ | Async functions, await |
| `test_extern.sg` | ✅ | Extern blocks, FFI |

### 13.2 New Tests (v0.4.0)

| Test File | Gap | Purpose |
|-----------|-----|---------|
| `test_public_fields.sg` | E | Public struct field visibility |
| `test_where_clauses.sg` | F | Where clause emission |
| `test_fn_traits.sg` | G | Fn/FnMut/FnOnce trait syntax |
| `test_raw_pointers.sg` | I | Raw pointer type annotations |

---

## 14. Real-World Validation

### 14.1 Nihil Test Suite

The primary validation target is the Nihil ML framework:

| Crate | Lines | Status | Notes |
|-------|-------|--------|-------|
| nihil-core | ~2,500 | ✅ Compiles | 8 warnings (scaffolding) |
| nihil-ops | ~800 | ✅ Compiles | 9 warnings (unused) |
| nihil-nn | ~1,200 | 🔲 Pending | Not yet tested |
| nihil-optim | ~600 | 🔲 Pending | Not yet tested |

### 14.2 Validation Process

```bash
# 1. Generate Rust from Sigil
sigil rust nihil/tomes/nihil-core/src/lib.sigil > /tmp/nihil-rs/nihil-core/src/lib.rs

# 2. Generate submodules
for f in tensor shape dtype device storage view broadcast error; do
  sigil rust nihil/tomes/nihil-core/src/${f}.sigil > /tmp/nihil-rs/nihil-core/src/${f}.rs
done

# 3. Compile with cargo
cd /tmp/nihil-rs/nihil-core && cargo build

# 4. Check for errors/warnings
```

---

## 15. Relationship to Other Specs

| Spec | Relationship |
|------|--------------|
| `RUST-CODEGEN-SPEC.md` | Main codegen specification (translation rules) |
| `02-SYNTAX.md` | Pattern matching syntax |
| `01-LEXICAL.md` | Macro invocation syntax |
| `NIHIL-COMPILER-GAPS-SPEC.md` | Nihil-specific compilation issues |

---

## 16. Gap J: Impl Block Where Clauses ✅ IMPLEMENTED

### 16.1 Specification

#### GRAMMAR (Sigil AST)

```
ImplBlock       ::= UnsafeMod? "⊢" Generics? TraitRef? SelfType WhereClause? ImplBody
UnsafeMod       ::= "unsafe"
Generics        ::= "<" GenericParam ("," GenericParam)* ">"
TraitRef        ::= TypePath "for"
SelfType        ::= TypeExpr
WhereClause     ::= "∋" WherePredicate ("," WherePredicate)*
ImplBody        ::= "{" ImplItem* "}"

WherePredicate  ::= Type ":" Bounds
                  | Lifetime ":" LifetimeBounds
                  | ForLifetimes Type ":" Bounds

Bounds          ::= Bound ("+" Bound)*
Bound           ::= TypePath | Lifetime | "?" TypePath
```

#### OUTPUT GRAMMAR (Rust)

```
RustImplBlock   ::= UnsafeMod? "impl" Generics? TraitRef? SelfType WhereClause? ImplBody
WhereClause     ::= "\nwhere " WherePredicate (",\n      " WherePredicate)*
```

#### TRANSFORMATION

```
ImplBlock { generics, trait_, self_ty, where_clause, items }:

  // Emit impl header
  IF is_unsafe:
    OUTPUT: "unsafe "
  OUTPUT: "impl"

  // Emit generics
  IF generics.is_some():
    OUTPUT: "<" + generics.params.join(", ") + ">"

  OUTPUT: " "

  // Emit trait (if trait impl)
  IF trait_.is_some():
    OUTPUT: emit_type_path(trait_) + " for "

  // Emit self type
  OUTPUT: emit_type(self_ty)

  // CRITICAL: Emit where clause BEFORE body
  IF where_clause.is_some():
    OUTPUT: "\n"
    OUTPUT: "where " + where_clause.predicates.map(emit_predicate).join(",\n      ")

  // Emit body
  OUTPUT: " {\n"
  FOR item IN items:
    emit_impl_item(item)
  OUTPUT: "}\n"

emit_predicate(pred):
  MATCH pred:
    TypeBound { ty, bounds }:
      OUTPUT: emit_type(ty) + ": " + bounds.join(" + ")

    LifetimeBound { lt, bounds }:
      OUTPUT: lt + ": " + bounds.join(" + ")

    HigherRanked { for_lifetimes, ty, bounds }:
      OUTPUT: "for<" + for_lifetimes.join(", ") + "> " + emit_type(ty) + ": " + bounds.join(" + ")
```

### 16.2 Current Status: ✅ IMPLEMENTED

**Implementation Date:** 2026-02-06

Changes made:
1. **AST:** Added `where_clause: Option<WhereClause>` to `ImplBlock` struct
2. **Parser:** Modified `parse_impl()` to store result of `parse_where_clause_opt()`
3. **Codegen:** Added `emit_where_clause()` call in `emit_impl()` after self_ty

**Test File:** `jormungandr/tests/rust_codegen/test_impl_where.sg`

Example:
```
// Input (Sigil):
⊢<T> Container<T> ∋ T: Clone {
    rite clone_value(&self) → T { ... }
}

// Output (Rust):
impl<T> Container<T>
where T: Clone {
    fn clone_value(&self) -> T { ... }
}
```

### 16.3 Implementation

#### PHASE 1: AST Changes

```
ADD TO ast.rs:

  pub struct ImplBlock {
      pub doc_comments: Vec<DocComment>,
      pub is_unsafe: bool,
      pub generics: Option<Generics>,
      pub trait_: Option<TypePath>,
      pub self_ty: TypeExpr,
      pub where_clause: Option<WhereClause>,   // ADD THIS
      pub items: Vec<ImplItem>,
  }
```

#### PHASE 2: Parser Changes

```
IN parse_impl_block():

  // ... existing parsing of ⊢, generics, trait, self_ty ...

  // NEW: Check for where clause
  where_clause = None
  IF peek() == Token::Where OR peek() == Token::Symbol("∋"):
    where_clause = Some(parse_where_clause())

  expect(Token::OpenBrace)
  // ... parse items ...
```

#### PHASE 3: Codegen Changes

```
IN emit_impl_block(impl_block):

  emit_impl_header(impl_block)
  emit_generics(impl_block.generics)
  emit_trait_for(impl_block.trait_)
  emit_type(impl_block.self_ty)

  // NEW: Emit where clause
  IF impl_block.where_clause.is_some():
    emit_where_clause(impl_block.where_clause)

  emit " {"
  // ... emit items ...
```

### 16.4 Invariants

```
I₁: Where clause appears AFTER self type, BEFORE opening brace
I₂: Where clause has SAME syntax as function where clauses (Gap F)
I₃: Empty where clause emits nothing
I₄: Multiple predicates use multiline formatting for readability
I₅: Trait impl where clause constrains both impl generics AND trait generics
I₆: Inherent impl where clause constrains only impl generics
```

### 16.5 Edge Cases

```
EC₁: Single predicate
     ⊢<T> Foo<T> ∋ T: Clone { }
     → impl<T> Foo<T>
       where T: Clone { }

EC₂: Multiple predicates
     ⊢<T, U> Pair<T, U> ∋ T: Clone, U: Debug { }
     → impl<T, U> Pair<T, U>
       where T: Clone,
             U: Debug { }

EC₃: Multiple bounds on single type
     ⊢<T> Foo<T> ∋ T: Clone + Debug + Send { }
     → impl<T> Foo<T>
       where T: Clone + Debug + Send { }

EC₄: Lifetime bounds
     ⊢<'a, T> Foo<'a, T> ∋ T: 'a { }
     → impl<'a, T> Foo<'a, T>
       where T: 'a { }

EC₅: Lifetime outlives lifetime
     ⊢<'a, 'b> Ref<'a, 'b> ∋ 'a: 'b { }
     → impl<'a, 'b> Ref<'a, 'b>
       where 'a: 'b { }

EC₆: Higher-ranked trait bound in impl
     ⊢<F> Handler<F> ∋ F: for<'a> Fn(&'a str) → &'a str { }
     → impl<F> Handler<F>
       where F: for<'a> Fn(&'a str) -> &'a str { }

EC₇: Trait impl with where clause
     ⊢<T> Iterator for Counter<T> ∋ T: Numeric { }
     → impl<T> Iterator for Counter<T>
       where T: Numeric { }

EC₈: Trait impl with associated type bound
     ⊢<I> Sum for I ∋ I: Iterator, I·Item: Add<Output = I·Item> { }
     → impl<I> Sum for I
       where I: Iterator,
             I::Item: Add<Output = I::Item> { }

EC₉: Where clause with Self reference
     ⊢<T> Foo<T> ∋ T: From<Self> { }
     → impl<T> Foo<T>
       where T: From<Self> { }

EC₁₀: Unsafe impl with where clause
      unsafe ⊢<T> Send for Wrapper<T> ∋ T: Sync { }
      → unsafe impl<T> Send for Wrapper<T>
        where T: Sync { }

EC₁₁: Default type params NOT in where clause
      ⊢<T, S = DefaultHasher> HashMap<T, S> ∋ T: Hash { }
      → impl<T, S = DefaultHasher> HashMap<T, S>
        where T: Hash { }

EC₁₂: No generics but has where clause (rare but valid)
      ⊢ Foo ∋ Self: Clone { }
      → impl Foo
        where Self: Clone { }

EC₁₃: Where clause with Fn trait
      ⊢<F, R> Callback<F, R> ∋ F: FnOnce() → R { }
      → impl<F, R> Callback<F, R>
        where F: FnOnce() -> R { }

EC₁₄: Combined with method-level where clause
      ⊢<T> Foo<T> ∋ T: Clone {
          rite bar<U>(&self, u: U) ∋ U: Debug { }
      }
      → impl<T> Foo<T>
        where T: Clone {
            fn bar<U>(&self, u: U)
            where U: Debug { }
        }
```

### 16.6 Error Conditions

```
E₁: Where clause without predicates
    ⊢<T> Foo<T> ∋ { }
    → PARSER ERROR: expected type after '∋'

E₂: Predicate with no bounds
    ⊢<T> Foo<T> ∋ T: { }
    → PARSER ERROR: expected bound after ':'

E₃: Where clause after body (wrong position)
    ⊢<T> Foo<T> { } ∋ T: Clone
    → PARSER ERROR: unexpected token after impl body

E₄: Duplicate predicates for same type
    ⊢<T> Foo<T> ∋ T: Clone, T: Debug { }
    → VALID (Rust allows, semantically equivalent to T: Clone + Debug)

E₅: Constraint on non-generic type
    ⊢ ConcreteFoo ∋ i32: Clone { }
    → VALID but pointless (Rust allows, constraint always satisfied)
```

### 16.7 Composition

```
WITH Gap F (function where clauses):
  Same syntax and parsing logic
  Method where clauses can add to impl-level constraints

WITH Gap B (generics):
  Where clauses reference generic parameters from impl header
  Default type params appear in generics, not where clause

WITH Gap G (Fn traits):
  Where bounds may contain Fn/FnMut/FnOnce
  Uses parenthetical syntax: F: Fn(T) -> U

WITH Gap K (impl Trait):
  Where bounds may contain impl Trait in theory
  BUT: impl Trait in where clause is UNSTABLE in Rust
  Emit error or warning if detected

WITH unsafe impl:
  unsafe keyword appears before impl
  Where clause appears in normal position
```

### 16.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_impl_where.sg`

**Required test cases:**
- Single predicate inherent impl
- Multiple predicates
- Multiple bounds (T: A + B + C)
- Lifetime bounds
- Higher-ranked trait bounds (for<'a>)
- Trait impl with where clause
- Associated type constraints
- Unsafe impl with where clause
- Combined impl and method where clauses
- Fn trait bounds in where clause

### 16.9 Priority

**P1** - Common pattern in generic code. Currently requires duplicating constraints on every method.

---

## 17. Gap K: Return Position `impl Trait` ✅ IMPLEMENTED

### 17.1 Specification

#### GRAMMAR (Sigil AST)

```
TypeExpr        ::= ... | ImplTrait
ImplTrait       ::= "impl" TraitBounds
TraitBounds     ::= TraitBound ("+" TraitBound)*
TraitBound      ::= Lifetime | TypePath | "?" TypePath

TypePath        ::= Path ("<" GenericArgs ">")?
                  | FnTraitPath
FnTraitPath     ::= ("Fn" | "FnMut" | "FnOnce") "(" Types? ")" ("→" Type)?

// Valid positions for impl Trait (Rust stable)
ReturnType      ::= "→" TypeExpr    // impl Trait ALLOWED here
ParamType       ::= TypeExpr        // impl Trait NOT STABLE (RFC 3498)
LetType         ::= TypeExpr        // impl Trait NOT ALLOWED

// Alternative Sigil syntax (optional)
ImplTraitAlt    ::= "⊢" TraitBounds   // ⊢ in type position = impl Trait
```

#### OUTPUT GRAMMAR (Rust)

```
ImplTraitType   ::= "impl " TraitBounds
TraitBounds     ::= Bound (" + " Bound)*
Bound           ::= Lifetime | TypePath | "?" TypePath
```

#### TRANSFORMATION

```
ImplTrait { bounds }:
  OUTPUT: "impl " + bounds.map(emit_bound).join(" + ")

emit_bound(bound):
  MATCH bound:
    Lifetime(lt):
      OUTPUT: lt                    // e.g., 'static

    TypePath(path):
      IF is_fn_trait(path):
        OUTPUT: emit_fn_trait(path)  // Use parenthetical syntax (Gap G)
      ELSE:
        OUTPUT: emit_type_path(path)

    MaybeBound(path):
      OUTPUT: "?" + emit_type_path(path)

emit_fn_trait(path):
  // See Gap G for full rules
  name = path.base                  // Fn, FnMut, FnOnce
  args = path.args[0]               // Tuple of param types
  ret  = path.args[1]               // Return type (or ())

  OUTPUT: name + "(" + args.join(", ") + ")"
  IF ret != Unit:
    OUTPUT: " -> " + emit_type(ret)
```

### 17.2 Current Status: ✅ IMPLEMENTED

**Implementation Date:** 2026-02-06

The feature was already implemented using **Option C** from the spec:
- `⊢` (turnstile) in type position is parsed as `TypeExpr::ImplTrait`
- Parser at line 3289: `Some(Token::Impl) => { ... TypeExpr::ImplTrait(bounds) }`
- Codegen at line 770: `TypeExpr::ImplTrait(bounds) => { self.write("impl "); ... }`

**Sigil Syntax:**
```sigil
rite foo() → ⊢ Iterator<Item = i32> { ... }
rite bar() → ⊢ Fn(i32) → i32 { ... }
```

**Test File:** `jormungandr/tests/rust_codegen/test_impl_trait.sg`

### 17.3 Implementation

#### PHASE 1: Lexer Changes

```
OPTION A: Context-sensitive impl keyword

  IN lex_keyword():
    IF word == "impl":
      IF context == TypePosition:
        RETURN Token::ImplKeyword  // Allow in types
      ELSE:
        RETURN Token::DeprecatedRustKeyword("impl")

OPTION B: Re-enable impl everywhere

  REMOVE "impl" from deprecated_keywords list
  Parser handles disambiguation:
    - impl in type position → ImplTrait
    - ⊢ for impl blocks (Sigil style)
    - impl for impl blocks (compatibility)

OPTION C: New syntax for impl Trait (Recommended)

  Use ⊢ in type position to mean impl Trait:
    rite foo() → ⊢ Iterator<Item = i32>

  Lexer already knows ⊢, no changes needed.
  Parser distinguishes:
    - ⊢ at statement level → impl block
    - ⊢ in type position → impl Trait
```

#### PHASE 2: AST Changes

```
ADD TO ast.rs:

  pub enum TypeExpr {
      // ... existing variants ...
      ImplTrait {
          bounds: Vec<TypeBound>,
      },
  }

  pub struct TypeBound {
      pub is_maybe: bool,           // ?Sized
      pub for_lifetimes: Option<Vec<Lifetime>>,  // for<'a>
      pub path: TypePath,
  }
```

#### PHASE 3: Parser Changes

```
IN parse_type():

  IF peek() == Token::Impl OR peek() == Token::Symbol("⊢"):
    consume()
    bounds = parse_trait_bounds()
    RETURN TypeExpr::ImplTrait { bounds }

  // ... existing type parsing ...

parse_trait_bounds():
  bounds = [parse_trait_bound()]
  WHILE peek() == Token::Plus:
    consume()
    bounds.push(parse_trait_bound())
  RETURN bounds

parse_trait_bound():
  is_maybe = peek() == Token::Question
  IF is_maybe:
    consume()

  for_lifetimes = None
  IF peek() == Token::For:
    for_lifetimes = parse_for_lifetimes()

  path = parse_type_path()
  RETURN TypeBound { is_maybe, for_lifetimes, path }
```

#### PHASE 4: Codegen Changes

```
IN emit_type(ty):

  MATCH ty:
    ImplTrait { bounds }:
      emit "impl "
      FOR i, bound IN enumerate(bounds):
        IF i > 0:
          emit " + "
        emit_type_bound(bound)

    // ... existing cases ...

emit_type_bound(bound):
  IF bound.is_maybe:
    emit "?"

  IF bound.for_lifetimes.is_some():
    emit "for<"
    emit bound.for_lifetimes.join(", ")
    emit "> "

  emit_type_path(bound.path)  // Uses Gap G for Fn traits
```

### 17.4 Invariants

```
I₁: impl Trait is ONLY valid in return position (stable Rust)
I₂: impl Trait emits "impl " followed by bounds
I₃: Multiple bounds use " + " separator
I₄: Fn traits within impl use parenthetical syntax (Gap G)
I₅: Lifetime bounds ('static) are valid bounds
I₆: ?Sized bound emits with "?" prefix
I₇: Higher-ranked bounds (for<'a>) preserve the for<> prefix
I₈: impl Trait is OPAQUE - concrete type hidden from caller
```

### 17.5 Edge Cases

```
EC₁: Return impl single trait
     rite foo() → impl Clone
     → fn foo() -> impl Clone

EC₂: Return impl Fn
     rite make_adder(n: i32) → impl Fn(i32) → i32
     → fn make_adder(n: i32) -> impl Fn(i32) -> i32

EC₃: Return impl FnMut
     rite counter() → impl FnMut() → i32
     → fn counter() -> impl FnMut() -> i32

EC₄: Return impl FnOnce
     rite consume(v: Vec<i32>) → impl FnOnce() → i32
     → fn consume(v: Vec<i32>) -> impl FnOnce() -> i32

EC₅: Return impl Iterator
     rite range(n: i32) → impl Iterator<Item = i32>
     → fn range(n: i32) -> impl Iterator<Item = i32>

EC₆: Return impl Future
     async rite fetch() → impl Future<Output = String>
     → async fn fetch() -> impl Future<Output = String>
     // Note: async fn already returns impl Future, may be redundant

EC₇: Multiple trait bounds
     rite foo() → impl Clone + Debug + Send
     → fn foo() -> impl Clone + Debug + Send

EC₈: Lifetime bound
     rite foo<'a>(s: &'a str) → impl Display + 'a
     → fn foo<'a>(s: &'a str) -> impl Display + 'a

EC₉: 'static bound
     rite foo() → impl Fn() + 'static
     → fn foo() -> impl Fn() + 'static

EC₁₀: Higher-ranked Fn trait
      rite foo() → impl for<'a> Fn(&'a str) → &'a str
      → fn foo() -> impl for<'a> Fn(&'a str) -> &'a str

EC₁₁: Nested impl (NOT VALID - error)
      rite foo() → impl Fn(impl Clone) → i32
      → ERROR: nested impl Trait not allowed

EC₁₂: impl Trait in generic arg (NOT VALID - error)
      rite foo() → Vec<impl Clone>
      → ERROR: impl Trait not allowed in type arguments

EC₁₃: Using alternative Sigil syntax (Option C)
      rite foo() → ⊢ Iterator<Item = i32>
      → fn foo() -> impl Iterator<Item = i32>

EC₁₄: impl ExactSizeIterator + DoubleEndedIterator
      rite foo() → impl ExactSizeIterator + DoubleEndedIterator<Item = i32>
      → fn foo() -> impl ExactSizeIterator + DoubleEndedIterator<Item = i32>

EC₁₅: Public function with impl Trait
      ☉ rite foo() → impl Clone
      → pub fn foo() -> impl Clone

EC₁₆: Generic function returning impl Trait
      rite wrap<T>(x: T) → impl AsRef<T> ∋ T: Clone
      → fn wrap<T>(x: T) -> impl AsRef<T>
        where T: Clone

EC₁₇: impl Trait with Send + Sync
      rite spawn_task() → impl Future<Output = ()> + Send + Sync
      → fn spawn_task() -> impl Future<Output = ()> + Send + Sync
```

### 17.6 Error Conditions

```
E₁: impl Trait in argument position (unstable)
    rite foo(x: impl Clone)
    → ERROR: impl Trait in argument position requires RFC 3498 (Type Alias Impl Trait)
    → WORKAROUND: use generic: rite foo<T: Clone>(x: T)

E₂: impl Trait in let binding
    ≔ x: impl Clone = ...
    → ERROR: impl Trait not allowed in let bindings
    → WORKAROUND: let Rust infer, or use concrete type

E₃: impl Trait in struct field
    sigil Foo { x: impl Clone }
    → ERROR: impl Trait not allowed in struct fields
    → WORKAROUND: use generics: sigil Foo<T: Clone> { x: T }

E₄: Nested impl Trait
    rite foo() → impl Fn(impl Clone)
    → ERROR: nested impl Trait is forbidden

E₅: impl Trait in type alias (unstable)
    type Alias = impl Clone
    → ERROR: Type Alias Impl Trait (TAIT) is unstable
    → Feature gate: #![feature(type_alias_impl_trait)]

E₆: impl Trait in where clause
    rite foo<T>() ∋ T: impl Clone
    → ERROR: impl Trait in where clause has no meaning
    → WORKAROUND: use trait directly: T: Clone

E₇: impl keyword rejected by lexer
    → PARSER ERROR: "expected identifier, found DeprecatedRustKeyword"
    → FIX: implement parser changes from 17.3
```

### 17.7 Composition

```
WITH Gap G (Fn trait syntax):
  impl Fn/FnMut/FnOnce use parenthetical syntax
  impl Fn(T) -> U, NOT impl Fn<(T,), U>

WITH Gap F (where clauses):
  Return type may reference generic parameters constrained in where clause
  rite foo<T>() → impl Trait ∋ T: Clone

WITH Gap J (impl where clauses):
  No direct interaction (impl Trait is in function signature, not impl block)

WITH dyn Trait:
  impl Trait and dyn Trait are DIFFERENT:
    impl Trait: static dispatch, concrete type hidden, zero-cost
    dyn Trait: dynamic dispatch, trait object, has runtime cost
  Cannot mix: "impl dyn Trait" is nonsensical

WITH async:
  async fn returns impl Future automatically
  async rite foo() → T is sugar for rite foo() → impl Future<Output = T>

WITH Box:
  Box<impl Trait> is NOT VALID (impl Trait must be return position)
  Box<dyn Trait> is valid for heap-allocated trait objects

WITH reference types:
  &impl Trait is NOT VALID (can't reference opaque type)
  impl Trait can include reference types: impl AsRef<str>
```

### 17.8 Design Decision: Sigil Syntax

The spec supports two syntactic approaches:

```
OPTION A: Allow 'impl' keyword in type position
  PROS:
    - Familiar to Rust programmers
    - Direct 1:1 mapping to Rust output
  CONS:
    - 'impl' is currently deprecated keyword
    - Requires lexer changes

OPTION B: Use ⊢ in type position for impl Trait
  Sigil:  rite foo() → ⊢ Iterator<Item = i32>
  Rust:   fn foo() -> impl Iterator<Item = i32>

  PROS:
    - Consistent with Sigil symbol vocabulary
    - No lexer changes (⊢ already recognized)
    - Clear distinction: ⊢ at stmt = impl block, ⊢ in type = impl Trait
  CONS:
    - Different syntax from Rust
    - Learning curve for Rust programmers

RECOMMENDATION: Support both
  - ⊢ in type position (Sigil-native)
  - 'impl' in type position (Rust compatibility)
  - Both emit "impl Trait" in Rust output
```

### 17.9 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_impl_trait.sg`

**Required test cases:**
- Return impl single trait (Clone, Debug, etc.)
- Return impl Fn/FnMut/FnOnce
- Return impl Iterator with associated type
- Return impl Future
- Multiple bounds (A + B + C)
- Lifetime bounds ('a, 'static)
- Higher-ranked bounds (for<'a>)
- Public function with impl Trait
- Generic function returning impl Trait
- Error: impl Trait in argument position
- Error: impl Trait in struct field
- Error: nested impl Trait

### 17.10 Priority

**P1** - Essential for idiomatic Rust patterns: returning closures, iterators, and futures

---

## 18. Gap L: Return Statement Emission ✅ IMPLEMENTED

### 18.1 Specification

#### GRAMMAR (Sigil AST)

```
ReturnStmt      ::= "⤺" Expr?
                  | "return" Expr?

// Context where return is valid
FunctionBody    ::= "{" Stmt* ReturnExpr? "}"
ReturnExpr      ::= Expr              // Implicit return (last expr, no semicolon)
                  | ReturnStmt        // Explicit return

// Alternative representations in AST
Stmt            ::= ... | ReturnStmt
Expr            ::= ... | ReturnExpr
```

#### OUTPUT GRAMMAR (Rust)

```
RustReturn      ::= "return" (" " Expr)? ";"
                  | "return" ";"       // bare return (unit)
```

#### TRANSFORMATION

```
ReturnStmt { expr: Some(e) }:
  OUTPUT: "return " + emit_expr(e) + ";"

ReturnStmt { expr: None }:
  OUTPUT: "return;"

// Context-aware emission
emit_stmt(stmt):
  MATCH stmt:
    Return { expr }:
      emit "return"
      IF expr.is_some():
        emit " "
        emit_expr(expr)
      emit ";"

    // Handle return as expression (rare but valid)
    Expr(ReturnExpr { expr }):
      emit "return"
      IF expr.is_some():
        emit " "
        emit_expr(expr)
      // No semicolon - it's an expression
```

### 18.2 Current Status: ✅ IMPLEMENTED

The `⤺` symbol is parsed as `Expr::Return` and codegen emits `return` correctly.

**Note:** The spec originally used `⤺` (U+23CE) but the actual Sigil symbol is `⤺` (U+2930).

### 18.3 Implementation

#### PHASE 1: AST Verification

```
VERIFY ast.rs contains:

  pub enum Stmt {
      // ... other variants ...
      Return(Option<Box<Expr>>),
  }

  // OR as expression:
  pub enum Expr {
      // ... other variants ...
      Return(Option<Box<Expr>>),
  }
```

#### PHASE 2: Codegen Changes

```
IN emit_stmt(stmt):

  MATCH stmt:
    Stmt::Return(expr):
      emit "return"
      IF expr.is_some():
        emit " "
        emit_expr(expr.unwrap())
      emit ";"

    Stmt::Expr(Expr::Return(expr)):
      // Return used as expression (value is !)
      emit "return"
      IF expr.is_some():
        emit " "
        emit_expr(expr.unwrap())
      // Expression context - let caller handle semicolon

    // ... existing cases ...

IN emit_expr(expr):

  MATCH expr:
    Expr::Return(inner):
      emit "return"
      IF inner.is_some():
        emit " "
        emit_expr(inner.unwrap())
      // No semicolon in expression context

    // ... existing cases ...
```

### 18.4 Invariants

```
I₁: ⤺ ALWAYS emits "return" keyword
I₂: Return with expression emits "return <expr>" (semicolon optional per Rust grammar)
I₃: Return without expression emits "return;"
I₄: Return is a statement, followed by semicolon
I₅: Return as expression has type ! (never type)
I₆: Function implicit return (last expr) does NOT emit "return"
I₇: Early return MUST emit "return" to exit function
```

### 18.5 Edge Cases

```
EC₁: Return with value
     rite foo() → i32 { ⤺ 42 }
     → fn foo() -> i32 { return 42; }

EC₂: Early return in conditional
     rite foo(x: i32) → i32 {
         ⎇ x < 0 { ⤺ 0 }
         x
     }
     → fn foo(x: i32) -> i32 {
         if x < 0 { return 0; }
         x
     }

EC₃: Return None (Option)
     rite foo() → Option<i32> {
         ⎇ cond { ⤺ None }
         Some(42)
     }
     → fn foo() -> Option<i32> {
         if cond { return None; }
         Some(42)
     }

EC₄: Return Err (Result)
     rite foo() → Result<i32, Error> {
         ⎇ cond { ⤺ Err(Error·new()) }
         Ok(42)
     }
     → fn foo() -> Result<i32, Error> {
         if cond { return Err(Error::new()); }
         Ok(42)
     }

EC₅: Bare return (unit return type)
     rite foo() {
         ⎇ cond { ⤺ }
         do_work()
     }
     → fn foo() {
         if cond { return; }
         do_work()
     }

EC₆: Return with complex expression
     ⤺ vec.iter().map(|x| x * 2).collect()
     → return vec.iter().map(|x| x * 2).collect();

EC₇: Return struct literal
     ⤺ Point { x: 1, y: 2 }
     → return Point { x: 1, y: 2 };

EC₈: Return tuple
     ⤺ (a, b, c)
     → return (a, b, c);

EC₉: Return in loop
     ∀ item ∈ items {
         ⎇ item.is_target() { ⤺ item }
     }
     → for item in items {
         if item.is_target() { return item; }
     }

EC₁₀: Return in match arm
      match value {
          Some(x) → ⤺ x,
          None → ⤺ default,
      }
      → match value {
          Some(x) => return x,
          None => return default,
      }

EC₁₁: Return in closure (returns from closure, not outer fn)
      ≔ f = |x| { ⤺ x * 2 };
      → let f = |x| { return x * 2; };

EC₁₂: Nested functions with return
      rite outer() → i32 {
          rite inner() → i32 { ⤺ 1 }
          ⤺ inner() + 1
      }
      → fn outer() -> i32 {
          fn inner() -> i32 { return 1; }
          return inner() + 1;
      }

EC₁₃: Return vs implicit return
      // Explicit return
      rite foo() → i32 { ⤺ 42 }
      → fn foo() -> i32 { return 42; }

      // Implicit return (NO ⤺)
      rite bar() → i32 { 42 }
      → fn bar() -> i32 { 42 }

EC₁₄: Multiple early returns
      rite classify(x: i32) → &str {
          ⎇ x < 0 { ⤺ "negative" }
          ⎇ x == 0 { ⤺ "zero" }
          "positive"
      }
      → fn classify(x: i32) -> &str {
          if x < 0 { return "negative"; }
          if x == 0 { return "zero"; }
          "positive"
      }

EC₁₅: Return with try operator
      ⤺ foo()?
      → return foo()?;
```

### 18.6 Error Conditions

```
E₁: Return type mismatch
    rite foo() → i32 { ⤺ "string" }
    → Rust type error (codegen emits, Rust rejects)

E₂: Return outside function
    ⤺ 42  // at module level
    → PARSER ERROR: return outside of function

E₃: Return with value in unit function
    rite foo() { ⤺ 42 }
    → Rust type error: expected (), found i32

E₄: Bare return in non-unit function
    rite foo() → i32 { ⤺ }
    → Rust type error: expected i32, found ()

E₅: Unreachable code after return
    rite foo() → i32 {
        ⤺ 42;
        do_something()  // unreachable
    }
    → Rust warning: unreachable code
```

### 18.7 Composition

```
WITH conditionals (⎇):
  Early return in if-else branches
  ⎇ cond { ⤺ x } else { ⤺ y }
  → if cond { return x; } else { return y; }

WITH loops (∀, while):
  Return exits the entire function, not just the loop
  Use 'break' to exit just the loop

WITH match:
  Return in match arm exits function
  Each arm can have its own return

WITH closures:
  ⤺ in closure returns from closure, not enclosing function
  Use labeled returns for outer function (not supported)

WITH try operator (?):
  ⤺ expr? returns the Ok value or propagates error
  Equivalent to: match expr { Ok(v) => return v, Err(e) => return Err(e.into()) }

WITH async:
  ⤺ in async function returns from the async block
  The future must be awaited to observe the return

WITH Gap A (pattern matching):
  Return can include destructured values
  ⤺ (a, b) where a, b from pattern match
```

### 18.8 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_return.sg`

**Required test cases:**
- Return with value
- Bare return (unit)
- Early return in conditional
- Return Option::None
- Return Result::Err
- Return in loop
- Return in match arm
- Return complex expression
- Multiple early returns
- Return with try operator
- Implicit return (no ⤺)
- Return in closure
- Nested function returns

### 18.9 Priority

**P0** - Early returns are fundamental control flow. Without this, functions requiring guard clauses or early exits cannot be expressed idiomatically

---

## 19. Gap M: Const Generic Default Values ✅ IMPLEMENTED

### 19.1 Discovery

During Phase 2 (Gap B) testing, discovered that const generic parameters with default values are not supported by the parser.

```
INPUT:  sigil ArrayWrapper<T, ◆ N: usize = 10> { ... }
ACTUAL: Parse error: "Unexpected token: expected Gt, found Struct"
```

### 19.2 Specification

#### GRAMMAR

```
ConstGenericParam ::= "◆" Ident ":" Type ("=" ConstExpr)?
ConstExpr         ::= IntLiteral | Ident | ConstBlock

// Current (what works)
ConstGenericParam ::= "◆" Ident ":" Type

// Desired (with default)
ConstGenericParam ::= "◆" Ident ":" Type ("=" ConstExpr)?
```

#### TRANSFORMATION

```
INPUT:  sigil Foo<◆ N: usize = 10> { ... }
OUTPUT: struct Foo<const N: usize = 10> { ... }

INPUT:  sigil Bar<T, ◆ N: usize = 16, ◆ M: usize = N * 2> { ... }
OUTPUT: struct Bar<T, const N: usize = 16, const M: usize = { N * 2 }> { ... }
```

### 19.3 Current Status: ✅ IMPLEMENTED

Fixed 2026-02-06: Changed `parse_expr()` to `parse_const_expr_simple()` in parser.
Codegen updated to emit default values for const generics.

**Note:** Rust doesn't allow defaults on impl blocks, only struct/trait definitions.

### 19.4 Root Cause

In `parse_generics_opt`, the const generic parsing branch handles:
- `const` keyword (via `◆`)
- Name parsing
- Optional type annotation (`: Type`)
- **Missing:** Optional default value (`= Expr`)

The default value parsing exists for type parameters but not for const parameters.

### 19.5 Required Implementation

```
PARSER (parser.rs, parse_generics_opt):

  IF consume_if(&Token::Const) OR consume_if(&Token::ConstGeneric):
    name = parse_ident()
    ty = IF consume_if(&Token::Colon):
           parse_type()
         ELSE:
           TypeExpr::Infer

    // ADD THIS: Parse optional default value
    default = IF consume_if(&Token::Eq):
                Some(Box::new(parse_const_expr()))
              ELSE:
                None

    params.push(GenericParam::Const { name, ty, default })
```

### 19.6 Invariants

```
I₁: Const generic defaults ALWAYS emit "= <expr>" after type
I₂: Default expression must be const-evaluable
I₃: Complex defaults use block syntax: { N * 2 }
I₄: Default can reference earlier const params: <const N: usize = 10, const M: usize = N>
I₅: Type inference for default uses declared type
```

### 19.7 Edge Cases

```
EC₁: Simple numeric default
     sigil Foo<◆ N: usize = 10> { }
     → struct Foo<const N: usize = 10> { }

EC₂: Default referencing another const param
     sigil Bar<◆ N: usize = 8, ◆ M: usize = N * 2> { }
     → struct Bar<const N: usize = 8, const M: usize = { N * 2 }> { }

EC₃: Default as const expression
     sigil Baz<◆ N: usize = { 1 << 4 }> { }
     → struct Baz<const N: usize = { 1 << 4 }> { }

EC₄: Mixed type and const defaults
     sigil Cache<T = String, ◆ N: usize = 16> { }
     → struct Cache<T = String, const N: usize = 16> { }

EC₅: Bool const default
     sigil Flag<◆ ENABLED: bool = true> { }
     → struct Flag<const ENABLED: bool = true> { }

EC₆: Char const default
     sigil Separator<◆ SEP: char = ','> { }
     → struct Separator<const SEP: char = ','> { }

EC₇: No default (existing behavior, must still work)
     sigil Matrix<◆ ROWS: usize, ◆ COLS: usize> { }
     → struct Matrix<const ROWS: usize, const COLS: usize> { }

EC₈: Impl block with const default
     ⊢<◆ N: usize = 10> Foo<N> { }
     → impl<const N: usize = 10> Foo<N> { }
```

### 19.8 Error Conditions

```
E₁: Non-const expression in default
    sigil Foo<◆ N: usize = runtime_value()>
    → ERROR: "const generic default must be const-evaluable"

E₂: Type mismatch in default
    sigil Foo<◆ N: usize = "string">
    → ERROR: "expected usize, found &str"

E₃: Forward reference to later param
    sigil Foo<◆ N: usize = M, ◆ M: usize>
    → ERROR: "cannot reference const param before declaration"
```

### 19.9 Composition

```
WITH Gap B (type defaults):
  sigil Cache<K, V, S = DefaultHasher, ◆ N: usize = 16>
  Type defaults and const defaults can coexist

WITH Gap J (where clauses):
  sigil Foo<◆ N: usize = 10> ∋ [T; N]: Default
  Where clause can reference const param with default

WITH impl blocks:
  ⊢<◆ N: usize = 10> Foo<N> { ... }
  Const defaults in impl block generic params
```

### 19.10 Test Coverage

**File:** `jormungandr/tests/rust_codegen/test_const_generic_defaults.sg`

**Required test cases:**
- Simple numeric default
- Default referencing another const
- Complex const expression default
- Mixed type and const defaults
- Bool/char const defaults
- No default (regression test)
- Impl block with const default

### 19.11 Priority

**P2** - Const generic defaults are a convenience feature. Most use cases can work without them by always specifying the value. Not blocking for core functionality.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-05 | Initial spec from Nihil codegen testing |
| 0.2.0 | 2026-02-05 | Fixed Gap A (Pattern::RefBinding), Gap C/D (Expr::Macro). |
| 0.3.0 | 2026-02-05 | Additional fixes: macro parens, symbol translation, cast parens, primitive type paths. |
| 0.4.0 | 2026-02-05 | Added Gaps E-I from Nihil nihil-ops compilation. |
| 0.5.0 | 2026-02-05 | SDD compliance: Gap J, K, L spec desired behavior. |
| **1.0.0** | **2026-02-06** | **S++ STANDARD:** Complete rewrite of Gaps A-L. All specs now include: formal grammar, transformation rules, invariants, exhaustive edge cases, error conditions, composition rules. Removed all concrete implementation code in favor of pseudo code. Daemoniorum agent-cognition quality. |
| 1.1.0 | 2026-02-06 | Added Gap M (const generic defaults). Discovered during Phase 2 TDD testing. Marked Gap B as IMPLEMENTED. |

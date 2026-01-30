# Sigil Self-Hosted Compiler: Fixed-Point Roadmap

## Executive Summary

This document provides a meticulous roadmap for achieving fixed-point compilation of the Sigil self-hosted compiler (Jormungandr). Fixed-point means the compiler can compile its own source code, producing C output identical to the bootstrap.

**Current State (as of 2025-12-18):**
- Basic expressions: integers, booleans, binary/unary operators ✓
- Let bindings and variable references ✓
- If/else expressions ✓
- Function definitions (simple) ✓
- Pattern matching for enums (partial) ✓

**Target:** 18,306 lines of Sigil code across 13 source files

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     COMPILATION PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│  Source (.sg) → Lexer → Parser → AST → TypeChecker →       │
│                 → Lowering → IR → CodeGen → C Output        │
└─────────────────────────────────────────────────────────────┘

Files by Layer:
  Foundation:  span.sg (157), token.sg (605), lib.sg (87)
  Frontend:    lexer.sg (1394), parser.sg (3200), ast.sg (1295)
  Middle:      typeck.sg (2798), lower.sg (1623), ir.sg (1146)
  Backend:     codegen.sg (3018), runtime.sg (824), interp.sg (1464)
  Driver:      driver.sg (695)
```

---

## Phase 1: Core Language Constructs

**Goal:** Support the basic building blocks used in all source files.

### 1.1 Control Flow (Priority: CRITICAL)

| Feature | Status | Files Using | Implementation Notes |
|---------|--------|-------------|---------------------|
| `if/else` expressions | ✓ Done | All 13 | Fixed in commit 66e7a19f0 |
| `while` loops | TODO | 8/13 | Condition + body block |
| `for` loops | TODO | 11/13 | Iterator pattern + binding |
| `loop` (infinite) | TODO | 3/13 | Simple body block |
| `break` | TODO | 5/13 | Loop exit, optional value |
| `continue` | TODO | 3/13 | Loop continuation |
| `return` (early) | TODO | 12/13 | Function exit with value |

**Implementation Order:**
1. `while` - simplest loop, condition is just an expression
2. `return` - early exit needed for error handling
3. `for` - requires iterator protocol
4. `break`/`continue` - require loop context tracking
5. `loop` - trivial once while works

**Specific Fixes Needed in build.sh:**
```
- IrOperation::Loop variant handling in emit_operation
- LoopVariant::While vs LoopVariant::For differentiation
- Break/Continue statement generation
- Return statement with expression
```

### 1.2 Pattern Matching (Priority: CRITICAL)

| Pattern Type | Status | Usage Count | Complexity |
|--------------|--------|-------------|------------|
| Literal patterns | Partial | High | 2/10 |
| Identifier binding | ✓ Done | Very High | 1/10 |
| Struct destructuring | TODO | High | 5/10 |
| Enum variant matching | Partial | Very High | 4/10 |
| Tuple destructuring | TODO | Medium | 3/10 |
| Wildcard `_` | TODO | High | 1/10 |
| Or-patterns `a \| b` | TODO | Low | 4/10 |
| Guard clauses `if cond` | TODO | Medium | 5/10 |
| `..` rest patterns | TODO | Low | 4/10 |

**Critical for Fixed-Point:**
- `match` on Token enum (100+ variants) - lexer.sg, parser.sg
- `match` on Expr/Stmt/Pattern enums - parser.sg, lower.sg
- `match` on Type enum - typeck.sg
- Struct field destructuring - all impl blocks

### 1.3 Functions (Priority: HIGH)

| Feature | Status | Files Using |
|---------|--------|-------------|
| Basic fn definition | ✓ Done | All |
| Parameters | ✓ Done | All |
| Return type | ✓ Done | All |
| `mut self` methods | Partial | 10/13 |
| `&self` methods | TODO | 10/13 |
| Associated functions | TODO | 8/13 |
| Generic functions | TODO | 6/13 |
| Closures `{x => expr}` | TODO | 5/13 |
| Higher-order functions | TODO | 4/13 |

---

## Phase 2: Type System

**Goal:** Support the type definitions and generics used throughout.

### 2.1 Struct Definitions

| Feature | Status | Notes |
|---------|--------|-------|
| Named fields | ✓ Done | Basic support |
| Tuple structs | TODO | `Span(u64, u64)` style |
| Unit structs | TODO | No fields |
| Generic structs | TODO | `Vec<T>`, `Option<T>` |
| impl blocks | Partial | Method dispatch |
| Self type | TODO | In impl blocks |

**Key Structs to Support:**
```sigil
// span.sg
struct Span { start: u64, end: u64 }
struct Spanned<T> { value: T, span: Span }

// lexer.sg
struct Lexer { source: String, pos: u64, line: u64, col: u64 }

// parser.sg
struct Parser { lexer: Lexer, current: Token, ... }

// typeck.sg
struct TypeChecker { types: Map<String, Type>, ... }
```

### 2.2 Enum Definitions

| Feature | Status | Notes |
|---------|--------|-------|
| Unit variants | ✓ Done | `Token::Fn` |
| Tuple variants | TODO | `Token::Int(i64)` |
| Struct variants | TODO | `Expr::Binary { left, op, right }` |
| Generic enums | TODO | `Option<T>`, `Result<T, E>` |
| Discriminants | TODO | `enum Foo { A = 1, B = 2 }` |

**Critical Enums:**
```sigil
// token.sg - 100+ variants
enum Token { Fn, Let, If, Else, ... Int(i64), String(String), ... }

// ast.sg - Complex nested
enum Expr {
    Literal(Literal),
    Binary { left: Box<Expr>, op: BinOp, right: Box<Expr> },
    If { condition: Box<Expr>, then_branch: Block, else_branch: ?Block },
    ...
}

// typeck.sg
enum Type { Unit, Bool, Int, Float, Array(Box<Type>), ... }
```

### 2.3 Generics

| Feature | Status | Priority |
|---------|--------|----------|
| Generic type parameters | TODO | HIGH |
| Generic impl blocks | TODO | HIGH |
| Trait bounds | TODO | MEDIUM |
| Where clauses | TODO | MEDIUM |
| Associated types | TODO | LOW |
| Turbofish `::<T>` | TODO | LOW |

**Generic Types Used:**
- `Vec<T>` - extensively used
- `Map<K, V>` - type environments
- `Option<T>` / `?T` - optional values
- `Result<T, E>` - error handling
- `Box<T>` - heap allocation
- `Rc<T>` - reference counting (runtime.sg)

---

## Phase 3: Standard Library Types

**Goal:** Implement runtime support for built-in types.

### 3.1 Option Type (`?T`)

```sigil
// Used throughout as:
let x: ?i64 = ?42;      // Some
let y: ?i64 = null;     // None
if let ?val = x { ... } // Pattern matching
x?                      // Unwrap/propagate
```

**Implementation:**
- TAG_OPTION_SOME / TAG_OPTION_NONE in runtime
- Pattern matching for Some/None
- `?` operator for propagation
- `.unwrap()`, `.is_some()`, `.is_none()` methods

### 3.2 Vec Type

```sigil
// Used as:
let v = Vec::new();
v.push(item);
v.len();
v.get(idx);
for item in v { ... }
```

**Implementation:**
- `sigil_Vec____new()`, `sigil_Vec____push()`, etc.
- Shared header for length tracking (already exists)
- Iterator protocol for `for` loops
- Index operator `v[i]`

### 3.3 Map Type

```sigil
// Used in type environments:
let m = Map::new();
m.insert(key, value);
m.get(key);
m.contains(key);
```

**Implementation:**
- Hash map with string keys
- `sigil_Map____*` functions (exist but need verification)

### 3.4 String Type

```sigil
// Used for:
let s = String::new();
s.push_str("hello");
format!("{} {}", a, b);  // String interpolation
s.chars();               // Character iteration
```

**Implementation:**
- String interning for efficiency
- `sigil_String____*` methods
- `sigil_format()` for interpolation
- Character iterator

---

## Phase 4: File-by-File Implementation Plan

### 4.1 span.sg (157 lines) - EASIEST ENTRY POINT

**Features Needed:**
- [x] Struct definition
- [x] impl block with methods
- [x] Basic arithmetic (`+`, `-`)
- [x] Comparison operators
- [ ] Generic `Spanned<T>` struct
- [ ] Simple tests

**Specific Constructs:**
```sigil
struct Span { start: u64, end: u64 }

impl Span {
    fn new(start: u64, end: u64) -> Span { ... }
    fn merge(self, other: Span) -> Span { ... }
    fn len(self) -> u64 { self.end - self.start }
    fn contains(self, pos: u64) -> bool { ... }
}
```

**Estimated Effort:** 1-2 hours

### 4.2 token.sg (605 lines) - ENUM CHALLENGE

**Features Needed:**
- [ ] Large enum (100+ variants)
- [ ] Unit variants, tuple variants
- [ ] impl block on enum
- [ ] match with many arms
- [ ] Method returning bool

**Key Challenge:** Token enum with variants like:
```sigil
enum Token {
    // Keywords
    Fn, Let, Mut, If, Else, Match, While, For, ...
    // Literals
    Int(i64), Float(f64), String(String), Char(char),
    // Operators
    Plus, Minus, Star, Slash, ...
    // Evidence markers
    Bang, Question, Tilde, Interrobang,
    ...
}
```

**Estimated Effort:** 4-6 hours (enum generation is complex)

### 4.3 lexer.sg (1394 lines) - STATE MACHINE

**Features Needed:**
- [ ] Struct with mutable state
- [ ] `mut self` methods
- [ ] `while` loops for scanning
- [ ] Character iteration
- [ ] String building
- [ ] match on characters
- [ ] Result/Option returns

**Key Patterns:**
```sigil
impl Lexer {
    fn next_token(mut self) -> Result<Token, LexError> {
        while self.pos < self.source.len() {
            let c = self.current_char();
            match c {
                ' ' | '\n' | '\t' => self.skip_whitespace(),
                '0'..='9' => return self.lex_number(),
                'a'..='z' | 'A'..='Z' => return self.lex_ident(),
                ...
            }
        }
    }
}
```

**Estimated Effort:** 8-12 hours

### 4.4 ast.sg (1295 lines) - DATA DEFINITIONS

**Features Needed:**
- [ ] Many enum definitions
- [ ] Struct variants in enums
- [ ] Box<T> for recursive types
- [ ] Generic Spanned<T>
- [ ] Visibility modifiers

**Key Types:**
```sigil
enum Expr {
    Literal(Literal),
    Path { segments: Vec<PathSegment> },
    Binary { left: Box<Expr>, op: BinOp, right: Box<Expr> },
    Unary { op: UnaryOp, operand: Box<Expr> },
    Call { func: Box<Expr>, args: Vec<Expr> },
    If { condition: Box<Expr>, then_branch: Block, else_branch: ?Block },
    Match { expr: Box<Expr>, arms: Vec<MatchArm> },
    ...
}

enum Stmt {
    Let { pattern: Pattern, ty: ?Type, init: ?Expr },
    Expr(Expr),
    Item(Item),
}

enum Pattern {
    Ident { name: String, mutable: bool },
    Literal(Literal),
    Struct { path: Path, fields: Vec<PatternField> },
    Tuple(Vec<Pattern>),
    ...
}
```

**Estimated Effort:** 6-8 hours

### 4.5 parser.sg (3200 lines) - LARGEST FILE

**Features Needed:**
- [ ] All previous features
- [ ] Recursive descent parsing
- [ ] Pratt parser for expressions
- [ ] Error recovery with Result
- [ ] Complex match expressions
- [ ] Method chaining

**Key Functions:**
```sigil
impl Parser {
    fn parse_file(mut self) -> Result<File, ParseError> { ... }
    fn parse_item(mut self) -> Result<Item, ParseError> { ... }
    fn parse_expr(mut self) -> Result<Expr, ParseError> { ... }
    fn parse_expr_bp(mut self, min_bp: u8) -> Result<Expr, ParseError> { ... }
    fn parse_pattern(mut self) -> Result<Pattern, ParseError> { ... }
    fn parse_type(mut self) -> Result<Type, ParseError> { ... }
}
```

**Estimated Effort:** 20-30 hours (largest, most complex)

### 4.6 typeck.sg (2798 lines) - TYPE INFERENCE

**Features Needed:**
- [ ] All previous features
- [ ] Type variable generation
- [ ] Unification algorithm
- [ ] Evidence lattice operations
- [ ] Scope/environment stacks
- [ ] Complex trait resolution

**Key Complexity:**
```sigil
impl TypeChecker {
    fn check_expr(mut self, expr: Expr) -> Result<Type, TypeError> {
        match expr {
            Expr::Binary { left, op, right } => {
                let left_ty = self.check_expr(*left)?;
                let right_ty = self.check_expr(*right)?;
                self.unify(left_ty, right_ty)?;
                // Evidence propagation
                ...
            }
            ...
        }
    }

    fn unify(mut self, a: Type, b: Type) -> Result<(), TypeError> {
        match (a, b) {
            (Type::Var(v), t) | (t, Type::Var(v)) => self.bind(v, t),
            (Type::Array(a), Type::Array(b)) => self.unify(*a, *b),
            ...
        }
    }
}
```

**Estimated Effort:** 25-35 hours

### 4.7 ir.sg (1146 lines) - INTERMEDIATE REPRESENTATION

**Features Needed:**
- [ ] IR operation definitions
- [ ] IR type definitions
- [ ] Evidence tracking in IR
- [ ] Pattern representation

**Key Types:**
```sigil
enum IrOperation {
    Literal { variant: LiteralVariant, value: LiteralValue, ... },
    Var { name: String, ... },
    Let { pattern: IrPattern, init: Box<IrOperation>, ... },
    Binary { op: BinaryOp, left: Box<IrOperation>, right: Box<IrOperation>, ... },
    If { condition: Box<IrOperation>, then_branch: Box<IrOperation>, ... },
    ...
}
```

**Estimated Effort:** 8-12 hours

### 4.8 lower.sg (1623 lines) - AST TO IR

**Features Needed:**
- [ ] Pattern matching on AST
- [ ] IR construction
- [ ] Context/environment tracking
- [ ] Error handling

**Key Functions:**
```sigil
fn lower_expr(ctx: &mut LoweringContext, expr: Expr) -> IrOperation {
    match expr {
        Expr::Literal(lit) => lower_literal(ctx, lit),
        Expr::Binary { left, op, right } => IrOperation::Binary { ... },
        Expr::If { condition, then_branch, else_branch } => IrOperation::If { ... },
        ...
    }
}
```

**Estimated Effort:** 12-16 hours

### 4.9 codegen.sg (3018 lines) - C CODE GENERATION

**Features Needed:**
- [ ] All previous features
- [ ] String generation
- [ ] Indentation management
- [ ] C syntax emission
- [ ] Name mangling

**Key Challenge:** Already partially working, but needs:
- All IrOperation variants handled
- Proper type emission
- Function/struct forward declarations
- Memory management code

**Estimated Effort:** 15-20 hours

### 4.10 runtime.sg (824 lines) - RUNTIME LIBRARY

**Features Needed:**
- [ ] Generic Rc<T> implementation
- [ ] Arena allocator
- [ ] String interning
- [ ] Math functions
- [ ] Evidence context

**Special Features:**
- `unsafe` blocks for raw memory
- Raw pointer manipulation
- Drop trait implementation

**Estimated Effort:** 10-15 hours

### 4.11 interp.sg (1464 lines) - INTERPRETER

**Features Needed:**
- [ ] Expression evaluation
- [ ] Environment/scope management
- [ ] Function calls
- [ ] Pattern matching execution

**Lower Priority:** Used for testing, not critical for compilation.

**Estimated Effort:** 10-12 hours

### 4.12 driver.sg (695 lines) - MAIN DRIVER

**Features Needed:**
- [ ] Command-line argument parsing
- [ ] File I/O
- [ ] Pipeline orchestration
- [ ] Error reporting

**Key Functions:**
```sigil
fn main(args: Vec<String>) -> i64 {
    let cmd = parse_command(args)?;
    match cmd {
        Command::Compile { input, output } => compile(input, output),
        Command::Run { input } => run(input),
        ...
    }
}
```

**Estimated Effort:** 4-6 hours

### 4.13 lib.sg (87 lines) - MODULE SETUP

**Features Needed:**
- [ ] Module declarations
- [ ] Re-exports

**Simplest file, but requires module system.**

**Estimated Effort:** 1-2 hours

---

## Phase 5: Build System Integration

### 5.1 Current build.sh Structure

```
Step 1: Generate C via Rust interpreter
Step 2: Post-process generated C (Python)
Step 3: Compile with GCC
Step 4: Test basic functionality
```

### 5.2 Pattern Matching Fix Categories

All wildcard `if ((1))` patterns need fixing for:
- [ ] `lower_expr` - Expr variants (20+ patterns)
- [ ] `lower_stmt` - Stmt variants (5+ patterns)
- [ ] `lower_pattern` - Pattern variants (10+ patterns)
- [ ] `lower_literal` - Literal variants (done for Int/Float/Bool)
- [ ] `emit_operation` - IrOperation variants (15+ patterns)
- [ ] `infer_expr` - Type inference patterns
- [ ] `check_*` - Type checking patterns

### 5.3 Enum Data Access Pattern

Current issue: Enum variants with data use `sigil_struct_field()` but need `v.e.data[N]`:

```python
# Pattern to fix:
sigil_struct_field(expr, "field_name")  # Wrong for TAG_ENUM

# Correct approach:
expr.v.e.data[0]  # First field
expr.v.e.data[1]  # Second field
```

---

## Phase 6: Testing Strategy

### 6.1 Unit Test Files

Create test files for each feature:

```
tests/
├── test_while.sg          # while loops
├── test_for.sg            # for loops
├── test_match.sg          # match expressions
├── test_structs.sg        # struct definitions
├── test_enums.sg          # enum definitions
├── test_generics.sg       # generic types
├── test_closures.sg       # closure syntax
├── test_methods.sg        # impl blocks
└── test_full_pipeline.sg  # end-to-end
```

### 6.2 Incremental Compilation Tests

```bash
# Test each source file individually
for f in src/*.sg; do
    ./build/sigil compile "$f" -o "build/$(basename $f .sg).c"
done
```

### 6.3 Fixed-Point Verification

```bash
# Generate C from Sigil sources
./build/sigil compile src/*.sg -o build/sigil2.c

# Compare with bootstrap
diff build/sigil_bootstrap.c build/sigil2.c

# If identical, compile sigil2
gcc -O2 -o build/sigil2 build/sigil2.c -lm

# Verify sigil2 can compile itself
./build/sigil2 compile src/*.sg -o build/sigil3.c
diff build/sigil2.c build/sigil3.c  # Should be identical
```

---

## Phase 7: Timeline Estimate

| Phase | Estimated Hours | Dependencies |
|-------|----------------|--------------|
| 1. Control Flow | 15-20 | None |
| 2. Type System | 30-40 | Phase 1 |
| 3. Standard Library | 20-25 | Phase 2 |
| 4a. span.sg + token.sg | 6-8 | Phase 1-2 |
| 4b. lexer.sg | 8-12 | 4a |
| 4c. ast.sg | 6-8 | Phase 2 |
| 4d. parser.sg | 20-30 | 4a, 4b, 4c |
| 4e. ir.sg + lower.sg | 20-28 | 4c |
| 4f. typeck.sg | 25-35 | 4c |
| 4g. codegen.sg | 15-20 | 4e |
| 4h. runtime.sg + interp.sg | 20-27 | Phase 3 |
| 4i. driver.sg + lib.sg | 5-8 | All above |
| 5. Build Integration | 10-15 | All above |
| 6. Testing & Debug | 20-30 | All above |

**Total Estimate: 220-310 hours**

---

## Critical Path

```
span.sg → token.sg → lexer.sg → parser.sg ─┐
                                            ├→ driver.sg → FIXED POINT
        ast.sg → typeck.sg → lower.sg ──────┤
                                            │
        ir.sg → codegen.sg ─────────────────┘
```

The critical path goes through parser.sg (largest) and typeck.sg (most complex).

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Enum data access patterns | HIGH | HIGH | Systematic fix in build.sh |
| Generic type instantiation | MEDIUM | HIGH | Start with monomorphic first |
| Recursive type handling | MEDIUM | MEDIUM | Box<T> already works |
| Evidence system complexity | LOW | MEDIUM | Defer advanced features |
| Parser complexity | MEDIUM | HIGH | Incremental testing |

---

## Success Criteria

1. **Compilation:** `./build/sigil compile src/*.sg` succeeds
2. **Equivalence:** Generated C matches bootstrap (or is semantically equivalent)
3. **Self-hosting:** sigil2 can compile itself to produce sigil3
4. **Stability:** sigil3 = sigil2 (true fixed point)

---

## Next Immediate Steps

1. **Implement `while` loops** - Required for lexer.sg
2. **Implement `for` loops** - Required for most files
3. **Fix remaining pattern matching** - Enum variants in match
4. **Test with span.sg** - Simplest real source file
5. **Progress to token.sg** - Tests large enum support

---

*Document Version: 1.0*
*Last Updated: 2025-12-18*
*Author: Claude (Jormungandr Bootstrap Assistant)*

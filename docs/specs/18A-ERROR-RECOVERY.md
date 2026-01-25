# Sigil Error Recovery Specification

> *"A compiler that stops at the first error is merely pedantic. A compiler that recovers
> and continues is a collaborator in debugging. For agent-driven development, error
> recovery is not optional — it's the primary interface."*

## 1. Overview

This specification defines the error recovery strategies for Sigil's compiler pipeline.
Error recovery enables the compiler to continue processing after encountering malformed
input, producing multiple diagnostics and maintaining partial semantic information
for tooling.

### 1.1 Design Philosophy

For agents generating and iterating on code:

1. **Maximum Diagnostic Density** — Report all errors, not just the first
2. **Graceful Degradation** — Partial AST is better than no AST
3. **Semantic Preservation** — Type information survives local errors
4. **Actionable Diagnostics** — Every error suggests a fix

### 1.2 Document Structure

| Section | Content |
|---------|---------|
| §2 | Lexer error recovery |
| §3 | Parser error recovery (panic mode) |
| §4 | Semantic error recovery |
| §5 | Diagnostic architecture |
| §6 | Error cascading prevention |
| §7 | Recovery in incremental compilation |

---

## 2. Lexer Error Recovery

### 2.1 Invalid Character Handling

When the lexer encounters an invalid character:

```
Algorithm: RECOVER_INVALID_CHAR(char, position)

1. EMIT diagnostic {
     severity: Error,
     code: E0001,
     message: "unexpected character `{char}`",
     span: position..position+1,
     suggestions: SUGGEST_CHARACTER(char)
   }

2. SKIP character

3. CONTINUE lexing from next position
```

**Character Suggestions:**

```
SUGGEST_CHARACTER(char) :=
  match char:
    | '`' => "did you mean single quote `'`?"
    | '@' => "@-attributes use //@ rune syntax"
    | '$' => "variable interpolation uses `{name}` not `$name`"
    | unicode_confusable =>
        "this looks like `{ascii_equivalent}` — use ASCII version"
    | _ => None
```

### 2.2 Unterminated Literals

```sigil
let s = "hello    // Missing closing quote
let x = 42
```

**Recovery Strategy:**

```
Algorithm: RECOVER_UNTERMINATED_STRING(start_pos)

1. SCAN forward for:
   - Closing quote
   - Newline (for single-line strings)
   - EOF

2. IF newline found before quote:
   EMIT diagnostic {
     code: E0002,
     message: "unterminated string literal",
     span: start_pos..newline_pos,
     suggestions: ["add closing `\"` at end of line"]
   }
   TERMINATE string at newline
   CONTINUE lexing

3. IF EOF found:
   EMIT diagnostic {...}
   TERMINATE string at EOF
```

### 2.3 Invalid Number Literals

```sigil
let x = 0x123G    // Invalid hex digit
let y = 3.14.15   // Multiple decimal points
let z = 1_000__0  // Double underscore
```

**Recovery:**

```
Algorithm: RECOVER_NUMBER_LITERAL(text)

1. PARSE greedily while numeric-like characters continue
2. IDENTIFY specific error:
   - Invalid digit for base
   - Multiple decimal points
   - Invalid underscore placement
   - Overflow/underflow
3. EMIT diagnostic with specific fix
4. PRODUCE ErrorLiteral token (evaluates to 0/0.0 for type checking)
5. CONTINUE
```

---

## 3. Parser Error Recovery

### 3.1 Panic Mode Recovery

When the parser encounters an unexpected token, it enters "panic mode":

```
Algorithm: PANIC_MODE_RECOVERY(expected, found)

Input:
  - expected: Set<TokenKind> — tokens the parser expected
  - found: Token — the actual token

Output:
  - Recovered position in token stream

Steps:

1. EMIT diagnostic {
     code: E0100,
     message: "expected {expected}, found `{found}`",
     span: found.span,
     suggestions: SUGGEST_FIX(expected, found)
   }

2. ENTER panic mode:
   sync_tokens = SYNCHRONIZATION_TOKENS(context)
   while current_token not in sync_tokens and not EOF:
       SKIP current_token

3. EXIT panic mode when synchronization token found

4. RETURN position of sync token
```

### 3.2 Synchronization Tokens

Synchronization tokens are chosen based on the parsing context:

| Context | Sync Tokens |
|---------|-------------|
| Top-level | `fn`, `struct`, `enum`, `trait`, `impl`, `mod`, `use`, `pub`, `const`, `type`, EOF |
| Function body | `}`, `;`, `fn`, `struct`, `let`, `if`, `while`, `for`, `match`, `return` |
| Block | `}`, `;`, `let`, `if`, `while`, `for`, `match`, `return` |
| Expression | `)`, `]`, `}`, `;`, `,` |
| Match arm | `=>`, `}`, `,` |
| Struct fields | `}`, `,` |

```
SYNCHRONIZATION_TOKENS(context) :=
  match context:
    | TopLevel => {FN, STRUCT, ENUM, TRAIT, IMPL, MOD, USE, PUB, CONST, TYPE, EOF}
    | FunctionBody => {RBRACE, SEMI, FN, STRUCT, LET, IF, WHILE, FOR, MATCH, RETURN}
    | Block => {RBRACE, SEMI, LET, IF, WHILE, FOR, MATCH, RETURN}
    | Expression => {RPAREN, RBRACKET, RBRACE, SEMI, COMMA}
    | MatchArm => {FAT_ARROW, RBRACE, COMMA}
    | StructFields => {RBRACE, COMMA}
```

### 3.3 Error Productions

For common errors, the parser has explicit "error productions" that construct partial AST:

```
// Missing semicolon
fn parse_statement() -> Stmt {
    let expr = parse_expression()?;

    if !eat(SEMI) {
        emit_diagnostic(E0103, "missing semicolon", suggest: "add `;`");
        // Continue without semicolon — don't fail entirely
    }

    Stmt::Expr(expr)
}

// Missing closing delimiter
fn parse_block() -> Block {
    expect(LBRACE)?;
    let stmts = parse_statements_until(RBRACE);

    if !eat(RBRACE) {
        emit_diagnostic(E0104, "unclosed block", suggest: "add `}`");
        // Return partial block
    }

    Block { stmts }
}
```

### 3.4 Error Nodes in AST

The AST includes explicit error nodes:

```
enum Expr {
    Literal(Literal),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    // ... other variants

    /// Represents a parse error — preserves span for diagnostics
    Error {
        span: Span,
        partial: Option<Box<Expr>>,  // Partial expression if available
        expected: Vec<String>,
    }
}

enum Type {
    Named(Path),
    Generic(Path, Vec<Type>),
    // ... other variants

    Error {
        span: Span,
        partial: Option<Box<Type>>,
    }
}
```

### 3.5 Delimiter Balancing

Special handling for unbalanced delimiters:

```
Algorithm: BALANCED_RECOVERY()

1. MAINTAIN delimiter stack: Vec<(Delimiter, Span)>

2. ON opening delimiter:
   PUSH (delimiter, span) to stack

3. ON closing delimiter:
   IF stack.top matches closing:
       POP stack
   ELSE IF closing appears in stack:
       // Missing intermediate closers
       while stack.top != matching(closing):
           EMIT diagnostic for unclosed delimiter at stack.top
           POP stack
       POP stack
   ELSE:
       // Extra closing delimiter
       EMIT diagnostic for unexpected closer
       // Don't consume — might belong to outer context

4. AT EOF:
   FOR each remaining delimiter in stack:
       EMIT diagnostic for unclosed delimiter
```

**Example:**

```sigil
fn main() {
    let x = foo(bar(1, 2)   // Missing )
    let y = 3;
}
```

Diagnostics:
```
error[E0105]: unclosed `(`
  --> src/main.sg:2:17
   |
2  |     let x = foo(bar(1, 2)
   |                 ^^^ unclosed parenthesis
   |
   = help: add `)` after `2`

warning: recovered at line 3
```

---

## 4. Semantic Error Recovery

### 4.1 Type Error Recovery

Type checking continues after type errors using error types:

```
Type ::= ...
       | TypeError  // Placeholder for failed type inference

Algorithm: TYPE_CHECK_WITH_RECOVERY(expr, expected)

1. actual = INFER_TYPE(expr)

2. IF actual == TypeError:
     // Previous error — don't cascade
     RETURN TypeError

3. IF not UNIFIES(actual, expected):
     EMIT diagnostic {
       code: E0308,
       message: "type mismatch",
       expected: expected,
       found: actual,
       suggestions: SUGGEST_TYPE_FIX(expected, actual)
     }
     RETURN TypeError  // Prevent cascading

4. RETURN actual
```

### 4.2 Name Resolution Recovery

When a name cannot be resolved:

```
Algorithm: RESOLVE_WITH_RECOVERY(name, context)

1. result = RESOLVE(name, context)

2. IF result is Error:
     suggestions = SIMILAR_NAMES(name, context)
     EMIT diagnostic {
       code: E0433,
       message: "cannot find `{name}` in scope",
       suggestions: suggestions
     }

     // Create placeholder binding for downstream analysis
     RETURN Placeholder {
       name: name,
       type: TypeError,
       span: name.span,
     }

3. RETURN result
```

### 4.3 Trait Resolution Recovery

```sigil
fn example() {
    let x: Vec<i32> = vec![1, 2, 3];
    x.nonexistent_method();  // Method doesn't exist
    x.push(4);               // Should still work
}
```

**Recovery:**

```
Algorithm: RESOLVE_METHOD_WITH_RECOVERY(receiver_ty, method_name)

1. result = RESOLVE_METHOD(receiver_ty, method_name)

2. IF result is Error:
     similar = SIMILAR_METHODS(receiver_ty, method_name)
     EMIT diagnostic {
       message: "no method `{method_name}` on type `{receiver_ty}`",
       suggestions: similar.map(|m| "did you mean `{m}`?")
     }

     // Return synthetic method with inferred signature
     RETURN SyntheticMethod {
       name: method_name,
       return_type: TypeError,
       params: INFER_FROM_CALL_SITE(),
     }

3. RETURN result
```

---

## 5. Diagnostic Architecture

### 5.1 Diagnostic Structure

```
struct Diagnostic {
    severity: Severity,           // Error, Warning, Info, Hint
    code: DiagnosticCode,         // E0308, W0001, etc.
    message: String,              // Human-readable message
    span: Span,                   // Primary source location
    labels: Vec<Label>,           // Additional annotated spans
    suggestions: Vec<Suggestion>, // Actionable fixes
    notes: Vec<String>,           // Additional context
}

struct Label {
    span: Span,
    message: String,
    style: LabelStyle,  // Primary, Secondary, Help
}

struct Suggestion {
    message: String,
    replacement: String,
    span: Span,
    applicability: Applicability,
}

enum Applicability {
    MachineApplicable,    // Safe to auto-apply
    MaybeIncorrect,       // Might not be what user wants
    HasPlaceholders,      // Contains <placeholder> text
    Unspecified,          // Just a hint
}
```

### 5.2 Diagnostic Rendering

```
error[E0308]: mismatched types
  --> src/main.sg:10:15
   |
10 |     let x: i32 = "hello";
   |            ---   ^^^^^^^ expected `i32`, found `&str`
   |            |
   |            expected due to this
   |
   = note: `&str` cannot be coerced to `i32`
   = help: consider parsing the string:
   |
10 |     let x: i32 = "hello".parse().unwrap();
   |                  ~~~~~~~~~~~~~~~~~~~~~~~~~
```

### 5.3 Diagnostic Codes

```
// Lexer errors: E0001-E0099
E0001 = "unexpected character"
E0002 = "unterminated string literal"
E0003 = "invalid numeric literal"
E0004 = "invalid escape sequence"

// Parser errors: E0100-E0199
E0100 = "unexpected token"
E0101 = "expected expression"
E0102 = "expected type"
E0103 = "missing semicolon"
E0104 = "unclosed delimiter"
E0105 = "unbalanced delimiter"

// Name resolution: E0400-E0499
E0433 = "cannot find value in scope"
E0434 = "cannot find type in scope"
E0435 = "cannot find module"
E0436 = "ambiguous name"

// Type errors: E0300-E0399
E0308 = "mismatched types"
E0309 = "missing type annotation"
E0310 = "cannot infer type"
E0311 = "type too complex"

// Borrow checking: E0500-E0599
E0502 = "cannot borrow as mutable"
E0503 = "cannot use while borrowed"
E0505 = "cannot move out of borrowed"
E0506 = "cannot assign to borrowed"

// Evidence errors: E0800-E0899
E0801 = "evidence level mismatch"
E0802 = "cannot derive known from reported"
E0803 = "trust assertion required"
```

---

## 6. Error Cascading Prevention

### 6.1 Cascading Errors

A single root error often causes many downstream errors:

```sigil
let x = unknown_function();  // E0433: not found
let y = x + 1;               // E0308: TypeError + i32
let z = y.method();          // E0599: no method on TypeError
do_something(z);             // E0308: expected T, found TypeError
```

Without prevention, user sees 4 errors for 1 mistake.

### 6.2 Error Tainting

```
Algorithm: TAINT_TRACKING()

1. MARK expressions with TypeError as "tainted"

2. WHEN type checking tainted expression:
   - Do NOT emit new type errors
   - PROPAGATE taint to result
   - STILL emit non-type errors (e.g., arity mismatch)

3. WHEN rendering diagnostics:
   - SHOW root error
   - OPTIONALLY show "N additional errors due to above"
```

### 6.3 Error Deduplication

```
Algorithm: DEDUPLICATE_DIAGNOSTICS(diagnostics)

1. GROUP diagnostics by (code, primary_span)

2. FOR each group with multiple diagnostics:
   KEEP the first one
   EMIT note: "similar error repeated {count} times"

3. LIMIT total diagnostics:
   IF count > MAX_ERRORS (default 100):
       EMIT "aborting due to {count} errors"
       TRUNCATE remaining
```

### 6.4 Related Error Grouping

```sigil
struct Foo {
    x: i32
}

fn main() {
    let f = Foo { y: 1 };     // E0560: no field `y`
    println(f.z);              // E0609: no field `z`
}
```

**Grouped Output:**

```
error[E0560]: struct `Foo` has no field named `y`
  --> src/main.sg:6:19
   |
6  |     let f = Foo { y: 1 };
   |                   ^ unknown field
   |
   = note: `Foo` has field: `x`
   = help: did you mean `x`?

error[E0609]: no field `z` on type `Foo`
  --> src/main.sg:7:16
   |
7  |     println(f.z);
   |              ^^^ unknown field
   |
   = note: available fields: `x`
```

---

## 7. Recovery in Incremental Compilation

### 7.1 Incremental Update Model

When source changes, the compiler should:

1. **Reparse changed regions** — Not entire file
2. **Preserve unchanged AST nodes** — Minimize recomputation
3. **Invalidate dependent analyses** — Type info for changed functions

```
Algorithm: INCREMENTAL_RECOVERY(old_ast, changes)

1. FOR each change in changes:
   affected_node = FIND_ENCLOSING_ITEM(old_ast, change.span)
   MARK affected_node for reparse

2. NEW_AST = CLONE(old_ast)
3. FOR each marked node:
   new_source = APPLY_CHANGES(node.source, changes)
   new_node = PARSE(new_source) with RECOVERY
   REPLACE node in NEW_AST with new_node

4. RETURN NEW_AST
```

### 7.2 Salsa-Style Queries

Error recovery integrates with incremental query systems:

```
// Query returns Result<T, Diagnostics>
// Diagnostics are cached separately from value

query fn type_of(expr: ExprId) -> (Type, Vec<Diagnostic>) {
    // Even on error, return something
    let ty = match expr.kind {
        ExprKind::Error => Type::Error,
        ExprKind::Literal(lit) => literal_type(lit),
        ExprKind::Binary(left, op, right) => {
            let (left_ty, left_diags) = type_of(left);
            let (right_ty, right_diags) = type_of(right);
            let diags = [left_diags, right_diags].concat();

            if left_ty.is_error() || right_ty.is_error() {
                (Type::Error, diags)
            } else {
                binary_op_type(left_ty, op, right_ty, diags)
            }
        }
        // ...
    };
    ty
}
```

### 7.3 IDE Integration

For LSP servers, error recovery is critical:

```
// LSP capabilities enabled by error recovery

- Diagnostics on every keystroke (not just on save)
- Completion in malformed contexts
- Go-to-definition through error nodes
- Hover information for partial types
- Refactoring that preserves error nodes
```

---

## 8. Agent-Specific Recovery

### 8.1 Structured Error Output

For agent consumption, errors are available as structured data:

```json
{
  "diagnostics": [
    {
      "code": "E0308",
      "severity": "error",
      "message": "mismatched types",
      "location": {
        "file": "src/main.sg",
        "line": 10,
        "column": 15,
        "span": [150, 157]
      },
      "expected": "i32",
      "found": "&str",
      "suggestions": [
        {
          "message": "parse the string",
          "replacement": "\"hello\".parse().unwrap()",
          "applicability": "maybe_incorrect"
        }
      ],
      "related": [
        {
          "location": {"line": 10, "column": 12},
          "message": "expected due to this"
        }
      ]
    }
  ],
  "partial_ast_available": true,
  "type_info_available": true
}
```

### 8.2 Recovery Hints for Code Generation

Agents can query what would make invalid code valid:

```
Query: WHAT_WOULD_FIX(error)

Response:
{
  "error": "E0308",
  "fixes": [
    {
      "strategy": "change_type_annotation",
      "from": "i32",
      "to": "&str"
    },
    {
      "strategy": "convert_value",
      "conversion": "\"{expr}\".parse::<i32>().unwrap()",
      "confidence": 0.7
    },
    {
      "strategy": "change_value",
      "example": "42",
      "confidence": 0.9
    }
  ]
}
```

### 8.3 Partial Semantic Analysis

Even with errors, provide as much semantic info as possible:

```sigil
fn example(x: i32) {
    let y = unknown();     // Error: unknown not in scope
    let z = x + y;         // y is TypeError, z is TypeError
    let w = x * 2;         // w is i32 (not affected)

    if w > 10 {            // Valid: w is known type
        do_something(z);   // Tainted, but do_something signature known
    }
}
```

Semantic info available:
- `x: i32` — Known
- `y: TypeError` — Error, but bound
- `z: TypeError` — Propagated error
- `w: i32` — Valid computation
- `if` condition type: `bool` — Valid
- `do_something` signature: Known from definition

---

## 9. Implementation Guidelines

### 9.1 Parser Implementation Pattern

```rust
fn parse_expression(&mut self) -> Expr {
    let start = self.current_span();

    match self.current() {
        Token::Literal(lit) => {
            self.advance();
            Expr::Literal(lit)
        }
        Token::Ident(name) => {
            self.advance();
            Expr::Name(name)
        }
        Token::LParen => {
            self.advance();
            let inner = self.parse_expression();
            if !self.eat(Token::RParen) {
                self.emit_error(E0104, "unclosed parenthesis");
                // Continue without rparen
            }
            inner
        }
        _ => {
            let err = self.emit_error(E0101, "expected expression");
            self.recover_to(EXPR_SYNC_TOKENS);
            Expr::Error {
                span: start.to(self.current_span()),
                partial: None,
                expected: vec!["expression".into()],
            }
        }
    }
}

fn recover_to(&mut self, sync: &[Token]) {
    while !sync.contains(&self.current()) && !self.at_eof() {
        self.advance();
    }
}
```

### 9.2 Type Checker Pattern

```rust
fn check_expr(&mut self, expr: &Expr, expected: &Type) -> Type {
    let actual = self.infer_expr(expr);

    if actual.is_error() {
        return Type::Error;  // Don't cascade
    }

    if !self.unify(actual.clone(), expected.clone()) {
        self.emit(Diagnostic {
            code: E0308,
            message: format!("expected {expected}, found {actual}"),
            span: expr.span(),
            suggestions: self.suggest_type_conversion(actual, expected),
        });
        return Type::Error;  // Prevent further cascading
    }

    actual
}
```

---

*This specification ensures that Sigil's compiler is a collaborative partner in debugging,
not an obstacle. For agents iterating on code, comprehensive error recovery enables
rapid convergence to correct programs.*

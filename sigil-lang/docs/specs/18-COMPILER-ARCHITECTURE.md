# Sigil Compiler Architecture Specification

> *"The serpent that compiles itself"*

## 1. Overview

The Sigil compiler transforms Sigil source code into executable binaries through multiple backends. This document specifies the complete compiler architecture, including:

1. **Compilation Phases** — Lexing, parsing, type checking, lowering, optimization, code generation
2. **Intermediate Representations** — AST, HIR, MIR, IR
3. **Backend Targets** — C (bootstrap), LLVM (production), WASM, JVM
4. **Bootstrap Process** — The Jormungandr self-hosting pipeline

---

## 2. Compiler Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIGIL COMPILATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Source (.sigil, .sg)                                                       │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   LEXER     │────▶│   PARSER    │────▶│  TYPE CHECK │                   │
│  │  (tokens)   │     │   (AST)     │     │   (TAST)    │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                                                 │                            │
│                                                 ▼                            │
│                                          ┌─────────────┐                    │
│                                          │   LOWER     │                    │
│                                          │   (HIR)     │                    │
│                                          └─────────────┘                    │
│                                                 │                            │
│                                                 ▼                            │
│                                          ┌─────────────┐                    │
│                                          │   MIR GEN   │                    │
│                                          │   (MIR)     │                    │
│                                          └─────────────┘                    │
│                                                 │                            │
│                    ┌────────────────────────────┼────────────────────────┐  │
│                    │                            │                        │  │
│                    ▼                            ▼                        ▼  │
│             ┌─────────────┐              ┌─────────────┐          ┌─────────────┐
│             │  C BACKEND  │              │LLVM BACKEND │          │WASM BACKEND │
│             │(bootstrap)  │              │(production) │          │  (web)      │
│             └─────────────┘              └─────────────┘          └─────────────┘
│                    │                            │                        │  │
│                    ▼                            ▼                        ▼  │
│             ┌─────────────┐              ┌─────────────┐          ┌─────────────┐
│             │  GCC/Clang  │              │   LLVM IR   │          │  WASM Binary│
│             │  (.c → .o)  │              │   → Native  │          │  (.wasm)    │
│             └─────────────┘              └─────────────┘          └─────────────┘
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Phase Summary

| Phase | Input | Output | Primary Responsibility |
|-------|-------|--------|----------------------|
| Lexer | Source text | Token stream | Unicode handling, morpheme recognition |
| Parser | Tokens | AST | Grammar validation, tree construction |
| Type Check | AST | Typed AST (TAST) | Type inference, evidentiality tracking |
| Lower | TAST | HIR | Desugar syntax, resolve names |
| MIR Gen | HIR | MIR | Control flow, borrow checking |
| Codegen | MIR | Target code | Backend-specific emission |

---

## 3. Lexical Analysis

### 3.1 Token Categories

```sigil
pub type TokenKind = enum {
    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    CharLiteral(char),

    // Identifiers and Keywords
    Ident(String),
    Keyword(Keyword),

    // Operators
    Plus, Minus, Star, Slash, Percent,
    Eq, EqEq, NotEq, Lt, Gt, LtEq, GtEq,
    And, Or, Not,
    Pipe, Ampersand, Caret, Tilde,

    // Morpheme Operators
    Tau,        // τ - Transform
    Phi,        // φ - Filter
    Sigma,      // σ - Sort/Collect
    Rho,        // ρ - Reduce
    Alpha,      // α - First
    Omega,      // ω - Last
    Lambda,     // λ - Lambda
    Delta,      // δ - Difference
    Pi,         // π - Product

    // Evidentiality
    Known,      // !
    Uncertain,  // ?
    Reported,   // ~
    Paradox,    // ‽

    // Delimiters
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,

    // Punctuation
    Comma, Semi, Colon, ColonColon,
    Dot, DotDot, DotDotEq,
    Arrow, FatArrow,
    MiddleDot,  // · (incorporation)

    // Special
    Eof,
    Error(String),
}
```

### 3.2 Morpheme Recognition

The lexer recognizes Sigil's polysynthetic operators:

| Symbol | Unicode | Name | Lexeme |
|--------|---------|------|--------|
| `τ` | U+03C4 | tau | Transform morpheme |
| `φ` | U+03C6 | phi | Filter morpheme |
| `σ` | U+03C3 | sigma | Sort/collect morpheme |
| `ρ` | U+03C1 | rho | Reduce morpheme |
| `α` | U+03B1 | alpha | First element |
| `ω` | U+03C9 | omega | Last element |
| `λ` | U+03BB | lambda | Lambda/closure |
| `δ` | U+03B4 | delta | Difference |
| `π` | U+03C0 | pi | Product |
| `·` | U+00B7 | middle dot | Incorporation |
| `⌛` | U+231B | hourglass | Await |

### 3.3 Unicode Handling

```sigil
// Identifiers support Unicode (XID_Start, XID_Continue)
let 日本語 = "Japanese"
let café = "coffee"
let Ω = 2.0 * π * f

// Morpheme operators are distinct from identifiers
data|τ{x => x + 1}   // τ is operator, not identifier
```

---

## 4. Syntax Analysis (Parser)

### 4.1 Abstract Syntax Tree

```sigil
pub type Ast = enum {
    // Module structure
    Module { items: [Item], span: Span },

    // Items
    Item::Fn(FnDef),
    Item::Struct(StructDef),
    Item::Enum(EnumDef),
    Item::Trait(TraitDef),
    Item::Impl(ImplBlock),
    Item::Use(UseDecl),
    Item::Mod(ModDecl),
    Item::Type(TypeAlias),
    Item::Const(ConstDef),
    Item::Static(StaticDef),

    // Expressions
    Expr::Literal(Literal),
    Expr::Path(Path),
    Expr::Binary { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Expr::Unary { op: UnOp, operand: Box<Expr> },
    Expr::Call { callee: Box<Expr>, args: [Expr] },
    Expr::MethodCall { receiver: Box<Expr>, method: Ident, args: [Expr] },
    Expr::Field { object: Box<Expr>, field: Ident },
    Expr::Index { object: Box<Expr>, index: Box<Expr> },
    Expr::Block(Block),
    Expr::If { cond: Box<Expr>, then: Block, else_: Option<Block> },
    Expr::Match { scrutinee: Box<Expr>, arms: [MatchArm] },
    Expr::Loop(Block),
    Expr::While { cond: Box<Expr>, body: Block },
    Expr::For { pat: Pattern, iter: Box<Expr>, body: Block },
    Expr::Return(Option<Box<Expr>>),
    Expr::Break(Option<Box<Expr>>),
    Expr::Continue,
    Expr::Closure { params: [Param], body: Box<Expr> },
    Expr::Struct { path: Path, fields: [FieldInit] },
    Expr::Array { elements: [Expr] },
    Expr::Tuple { elements: [Expr] },
    Expr::Morpheme { kind: MorphemeKind, operand: Box<Expr>, transform: Option<Box<Expr>> },
    Expr::Await(Box<Expr>),
    Expr::Try(Box<Expr>),

    // Statements
    Stmt::Let { pat: Pattern, ty: Option<Type>, init: Option<Expr> },
    Stmt::Expr(Expr),
    Stmt::Semi(Expr),

    // Patterns
    Pattern::Ident { name: Ident, mutable: bool },
    Pattern::Tuple([Pattern]),
    Pattern::Struct { path: Path, fields: [FieldPattern] },
    Pattern::Enum { path: Path, inner: Option<Box<Pattern>> },
    Pattern::Literal(Literal),
    Pattern::Wildcard,
    Pattern::Or([Pattern]),

    // Types
    Type::Path(Path),
    Type::Reference { mutable: bool, lifetime: Option<Lifetime>, inner: Box<Type> },
    Type::Array { element: Box<Type>, size: Option<Expr> },
    Type::Tuple([Type]),
    Type::Fn { params: [Type], ret: Box<Type> },
    Type::Infer,
    Type::Evidence { inner: Box<Type>, evidence: EvidenceLevel },
}
```

### 4.2 Precedence and Associativity

| Precedence | Operators | Associativity |
|------------|-----------|---------------|
| 1 (lowest) | `=`, `+=`, `-=`, etc. | Right |
| 2 | `\|\|` | Left |
| 3 | `&&` | Left |
| 4 | `==`, `!=`, `<`, `>`, `<=`, `>=` | Left |
| 5 | `\|` | Left |
| 6 | `^` | Left |
| 7 | `&` | Left |
| 8 | `<<`, `>>` | Left |
| 9 | `+`, `-` | Left |
| 10 | `*`, `/`, `%` | Left |
| 11 | `!`, `-` (unary), `&`, `*` | Prefix |
| 12 | `?`, `!` (postfix evidence) | Postfix |
| 13 | `.`, `·`, `::`, `[]`, `()` | Left |
| 14 (highest) | Morphemes (`\|τ`, `\|φ`, etc.) | Left |

---

## 5. Type System Implementation

### 5.1 Type Representation

```sigil
pub type Ty = enum {
    // Primitives
    Bool,
    Int(IntTy),
    Float(FloatTy),
    Char,
    Str,
    Unit,
    Never,

    // Composite
    Array { elem: Box<Ty>, size: Option<usize> },
    Tuple(Vec<Ty>),
    Struct { name: Symbol, fields: Map<Symbol, Ty> },
    Enum { name: Symbol, variants: Map<Symbol, Option<Ty>> },

    // References
    Ref { mutable: bool, lifetime: Lifetime, inner: Box<Ty> },

    // Functions
    Fn { params: Vec<Ty>, ret: Box<Ty> },
    Closure { params: Vec<Ty>, ret: Box<Ty>, captures: Vec<(Symbol, Ty)> },

    // Generics
    Param(ParamId),
    App { base: Box<Ty>, args: Vec<Ty> },

    // Inference
    Infer(InferVar),

    // Evidentiality
    Evidence { inner: Box<Ty>, level: EvidenceLevel },
}

pub type EvidenceLevel = enum {
    Known,      // !
    Uncertain,  // ?
    Reported,   // ~
    Paradox,    // ‽
}
```

### 5.2 Type Inference Algorithm

Sigil uses bidirectional type inference with constraint solving:

```
Algorithm: infer(expr, expected) -> (Ty, Constraints)

1. SYNTHESIS (bottom-up):
   - Literals: return concrete type
   - Variables: lookup in environment
   - Field access: synthesize receiver, lookup field
   - Function call: synthesize callee, check args

2. CHECKING (top-down):
   - If expected type available, check against it
   - Generate constraints for mismatches

3. UNIFICATION:
   - Solve constraints: α = τ, α = β
   - Propagate solutions to inference variables

4. EVIDENCE INFERENCE:
   - Track evidence through operations
   - Apply combining rules (see 03-TYPES.md §3.3)
   - Verify evidence requirements in signatures
```

### 5.3 Evidence Tracking

```sigil
// Evidence is tracked through the type system
pub type EvidenceContext = struct {
    bindings: Map<Symbol, EvidenceLevel>,
    returns: EvidenceLevel,
}

impl EvidenceContext {
    // Combine evidence for binary operations
    fn combine(a: EvidenceLevel, b: EvidenceLevel) -> EvidenceLevel {
        match (a, b) {
            (Known, Known) => Known,
            (Paradox, _) | (_, Paradox) => Paradox,
            (Uncertain, _) | (_, Uncertain) => Uncertain,
            (Reported, _) | (_, Reported) => Reported,
        }
    }

    // Check evidence subtyping
    fn is_subtype(sub: EvidenceLevel, sup: EvidenceLevel) -> bool {
        match (sub, sup) {
            (Known, _) => true,
            (_, Paradox) => true,
            (Uncertain, Uncertain) => true,
            (Reported, Reported) => true,
            _ => false,
        }
    }
}
```

---

## 6. High-Level IR (HIR)

### 6.1 HIR Structure

HIR desugars syntactic constructs into simpler forms:

```sigil
pub type Hir = enum {
    // Simplified items (no syntactic sugar)
    HirFn { name: Symbol, params: [HirParam], body: HirExpr, sig: FnSig },
    HirStruct { name: Symbol, fields: [HirField] },
    HirEnum { name: Symbol, variants: [HirVariant] },

    // Simplified expressions
    HirExpr::Literal(Literal, Ty),
    HirExpr::Local(LocalId, Ty),
    HirExpr::Global(DefId, Ty),
    HirExpr::Binary { op: BinOp, left: Box<HirExpr>, right: Box<HirExpr>, ty: Ty },
    HirExpr::Call { callee: DefId, args: [HirExpr], ty: Ty },
    HirExpr::Block { stmts: [HirStmt], expr: Option<Box<HirExpr>>, ty: Ty },
    HirExpr::If { cond: Box<HirExpr>, then: Box<HirExpr>, else_: Box<HirExpr>, ty: Ty },
    HirExpr::Match { scrutinee: Box<HirExpr>, arms: [HirArm], ty: Ty },
    HirExpr::Loop { body: Box<HirExpr>, ty: Ty },

    // Statements
    HirStmt::Let { id: LocalId, init: HirExpr },
    HirStmt::Assign { place: HirPlace, value: HirExpr },
    HirStmt::Expr(HirExpr),
}
```

### 6.2 Desugaring Rules

| Syntax | HIR |
|--------|-----|
| `for x in iter { body }` | `loop { match iter.next() { Some(x) => body, None => break } }` |
| `while cond { body }` | `loop { if !cond { break }; body }` |
| `data\|τ{f}` | `data.iter().map(f).collect()` |
| `data\|φ{p}` | `data.iter().filter(p).collect()` |
| `x?` | `match x { Some(v) => v, None => return None }` |
| `a?.b` | `match a { Some(v) => Some(v.b), None => None }` |
| `obj·method(args)` | `Type::method(obj, args)` |

---

## 7. Mid-Level IR (MIR)

### 7.1 Control Flow Graph

MIR represents the program as a control flow graph:

```sigil
pub type Mir = struct {
    functions: Map<DefId, MirFunction>,
}

pub type MirFunction = struct {
    params: [MirLocal],
    locals: [MirLocal],
    blocks: [MirBasicBlock],
    entry: BlockId,
}

pub type MirBasicBlock = struct {
    id: BlockId,
    statements: [MirStatement],
    terminator: MirTerminator,
}

pub type MirStatement = enum {
    Assign { place: MirPlace, rvalue: MirRvalue },
    StorageLive(LocalId),
    StorageDead(LocalId),
    Nop,
}

pub type MirTerminator = enum {
    Goto(BlockId),
    If { cond: MirOperand, then: BlockId, else_: BlockId },
    Switch { value: MirOperand, targets: [BlockId], default: BlockId },
    Call { func: DefId, args: [MirOperand], dest: MirPlace, next: BlockId },
    Return,
    Unreachable,
    Drop { place: MirPlace, next: BlockId },
}

pub type MirRvalue = enum {
    Use(MirOperand),
    BinaryOp { op: BinOp, left: MirOperand, right: MirOperand },
    UnaryOp { op: UnOp, operand: MirOperand },
    Ref { mutable: bool, place: MirPlace },
    Aggregate { kind: AggregateKind, operands: [MirOperand] },
    Discriminant(MirPlace),
    Len(MirPlace),
}
```

### 7.2 Borrow Checking

Borrow checking operates on MIR:

```
Algorithm: borrow_check(mir_function)

1. LIVENESS ANALYSIS:
   - Compute live ranges for all locals
   - Track last use of each borrow

2. MOVE ANALYSIS:
   - Detect moves of non-Copy types
   - Error on use after move

3. BORROW ANALYSIS:
   - Track active borrows at each program point
   - Verify exclusivity: &mut exclusive, & shared
   - Check lifetimes don't exceed borrowed data

4. EVIDENCE VERIFICATION:
   - Verify evidence levels at function boundaries
   - Check evidence assertions (! on ? values)
```

---

## 8. Code Generation

### 8.1 C Backend (Bootstrap)

The C backend generates standalone C code for bootstrap:

```sigil
pub type CCodeGen = struct {
    output: String,
    indent: usize,
    declared_vars: Set<String>,
    temp_counter: usize,
}

impl CCodeGen {
    // Entry point
    fn generate(mir: &Mir) -> String {
        let mut cg = CCodeGen::new();
        cg.emit_runtime_header();
        for (id, func) in mir.functions {
            cg.emit_function_decl(id, func);
        }
        for (id, func) in mir.functions {
            cg.emit_function(id, func);
        }
        cg.emit_builtin_implementations();
        cg.output
    }

    // Emit MIR operation as C
    fn emit_rvalue(rvalue: &MirRvalue) -> String {
        match rvalue {
            MirRvalue::BinaryOp { op: Add, left, right } => {
                format!("sigil_add({}, {})",
                    self.emit_operand(left),
                    self.emit_operand(right))
            }
            // ... other operations
        }
    }
}
```

See [17-JORMUNGANDR-BOOTSTRAP.md](./17-JORMUNGANDR-BOOTSTRAP.md) for complete C codegen semantics.

### 8.2 LLVM Backend (Production)

The LLVM backend generates optimized native code:

```sigil
pub type LlvmCodeGen = struct {
    context: LlvmContext,
    module: LlvmModule,
    builder: LlvmBuilder,
    types: Map<Ty, LlvmType>,
    values: Map<LocalId, LlvmValue>,
}

impl LlvmCodeGen {
    fn compile_function(func: &MirFunction) -> LlvmFunction {
        // Map Sigil types to LLVM types
        let llvm_params = func.params.iter()
            .map(|p| self.lower_type(p.ty))
            .collect();

        // Create function
        let llvm_fn = self.module.add_function(
            func.name,
            self.context.function_type(ret_ty, llvm_params)
        );

        // Emit basic blocks
        for block in func.blocks {
            self.emit_block(block);
        }

        llvm_fn
    }

    fn lower_type(ty: &Ty) -> LlvmType {
        match ty {
            Ty::Bool => self.context.i1_type(),
            Ty::Int(I32) => self.context.i32_type(),
            Ty::Int(I64) => self.context.i64_type(),
            Ty::Float(F64) => self.context.f64_type(),
            Ty::Struct { fields, .. } => {
                let field_types: Vec<_> = fields.values()
                    .map(|t| self.lower_type(t))
                    .collect();
                self.context.struct_type(&field_types)
            }
            // Evidentiality is erased at LLVM level
            Ty::Evidence { inner, .. } => self.lower_type(inner),
            // ...
        }
    }
}
```

### 8.3 WASM Backend

The WASM backend targets WebAssembly:

```sigil
pub type WasmCodeGen = struct {
    module: WasmModule,
    types: WasmTypeSection,
    functions: WasmFunctionSection,
    exports: WasmExportSection,
    code: WasmCodeSection,
}

impl WasmCodeGen {
    fn compile_function(func: &MirFunction) -> WasmFunctionBody {
        let mut locals = Vec::new();
        let mut instructions = Vec::new();

        for block in func.blocks {
            for stmt in block.statements {
                self.emit_statement(stmt, &mut instructions);
            }
            self.emit_terminator(block.terminator, &mut instructions);
        }

        WasmFunctionBody { locals, instructions }
    }

    fn emit_rvalue(rvalue: &MirRvalue) -> Vec<WasmInstruction> {
        match rvalue {
            MirRvalue::BinaryOp { op: Add, left, right } => {
                vec![
                    self.emit_operand(left),
                    self.emit_operand(right),
                    WasmInstruction::I64Add,  // or F64Add based on type
                ]
            }
            // ...
        }
    }
}
```

---

## 9. Bootstrap Pipeline (Jormungandr)

### 9.1 Three-Stage Bootstrap

```
Stage 0: Rust Interpreter
├── Input: Sigil source (self-hosted/*.sigil)
├── Tool: cargo run -- compile
├── Output: sigil_bootstrap.c

Stage 1: C Compiler
├── Input: sigil_bootstrap.c
├── Tool: gcc -O2 -o sigil
├── Output: Native binary (build/sigil)

Stage 2: Self-Compilation
├── Input: Sigil source (self-hosted/*.sigil)
├── Tool: ./build/sigil compile
├── Output: sigil2.c

Verification: diff sigil_bootstrap.c sigil2.c
└── Must be identical (fixed point)
```

### 9.2 Bootstrap Compiler Limitations

The bootstrap compiler (C backend) has these limitations:

| Feature | Status | Notes |
|---------|--------|-------|
| Basic types | ✅ | i32, i64, f64, bool, String |
| Functions | ✅ | Full support |
| Structs | ✅ | Via SigilValue runtime |
| Enums | ✅ | Via variant tagging |
| Generics | ⚠️ | Type erased |
| Traits | ⚠️ | Monomorphized |
| Closures | ✅ | Static functions |
| Async | ❌ | Not in bootstrap |
| Macros | ❌ | Pre-expanded |
| `#[attributes]` | ❌ | Not parsed |
| `html!` macro | ❌ | Requires proc-macro |
| Evidentiality | ⚠️ | Runtime field only |

### 9.3 Build Commands

```bash
# Full bootstrap from source
cd sigil/sigil-lang/self-hosted

# Stage 0: Generate C from Rust interpreter
cargo run --manifest-path ../parser/Cargo.toml -- \
    compile src/*.sigil -o build/sigil_bootstrap.c

# Stage 1: Compile C to native
gcc -O2 -o build/sigil build/sigil_bootstrap.c -lm

# Stage 2: Self-compile
./build/sigil compile src/*.sigil -o build/sigil2.c

# Verify fixed point
diff build/sigil_bootstrap.c build/sigil2.c && echo "Fixed point achieved!"
```

---

## 10. Optimization Passes

### 10.1 HIR Optimizations

| Pass | Description |
|------|-------------|
| Constant Folding | Evaluate constant expressions |
| Dead Code Elimination | Remove unreachable code |
| Inline Small Functions | Inline functions ≤ 5 statements |
| Loop Unrolling | Unroll small constant loops |

### 10.2 MIR Optimizations

| Pass | Description |
|------|-------------|
| Copy Propagation | Replace copies with originals |
| Dead Store Elimination | Remove unused assignments |
| Common Subexpression | Reuse computed values |
| Tail Call Optimization | Convert tail calls to jumps |

### 10.3 LLVM Optimizations

The LLVM backend leverages LLVM's optimization passes:

```bash
# Optimization levels
-O0: No optimization (debug)
-O1: Basic optimizations
-O2: Standard optimizations (default)
-O3: Aggressive optimizations
-Oz: Optimize for size (WASM)
-Os: Optimize for size (balance)
```

---

## 11. Diagnostic System

### 11.1 Error Structure

```sigil
pub type Diagnostic = struct {
    level: DiagnosticLevel,
    message: String,
    code: Option<String>,
    span: Span,
    labels: [Label],
    notes: [String],
    suggestions: [Suggestion],
}

pub type DiagnosticLevel = enum {
    Error,
    Warning,
    Info,
    Hint,
}

pub type Label = struct {
    span: Span,
    message: String,
    style: LabelStyle,
}

pub type Suggestion = struct {
    message: String,
    span: Span,
    replacement: String,
}
```

### 11.2 Error Codes

| Code | Category | Example |
|------|----------|---------|
| E0001 | Parse | Unexpected token |
| E0100 | Type | Type mismatch |
| E0101 | Type | Unknown type |
| E0200 | Borrow | Cannot move out of borrow |
| E0201 | Borrow | Conflicting borrows |
| E0300 | Evidence | Evidence level mismatch |
| E0301 | Evidence | Unhandled uncertain value |
| E0400 | Name | Undefined variable |
| E0401 | Name | Duplicate definition |

---

## 12. Future Backends

### 12.1 JVM Backend (Planned)

```
Sigil → HIR → JVM Bytecode → .class files
```

Target: Java/Kotlin interop for enterprise environments.

### 12.2 GPU Backend (Research)

```
Sigil → MIR → SPIR-V → GPU kernels
```

Target: Parallel computation via compute shaders.

### 12.3 Neural Backend (Experimental)

```
Sigil → AI-IR → Neural inference engine
```

Target: AI-native code execution via Infernum.

---

## 13. Testing Infrastructure

### 13.1 Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `parser/src/*_test.rs` | Component tests |
| Integration | `self-hosted/tests/` | End-to-end compilation |
| Codegen | `self-hosted/tests/codegen/` | C output verification |
| Bootstrap | `scripts/test-bootstrap.sh` | Fixed point verification |
| Spec | `self-hosted/tests/spec/` | Language conformance |

### 13.2 Test Commands

```bash
# Run all tests
cargo test --workspace

# Run codegen tests only
cargo test codegen

# Run bootstrap verification
./scripts/verify-bootstrap.sh

# Run spec compliance tests
./scripts/run-spec-tests.sh
```

---

## 14. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial specification |
| 0.2.0 | 2025-01 | Added WASM backend, evidentiality tracking |

---

*This specification is part of the Sigil language project by Daemoniorum, LLC.*

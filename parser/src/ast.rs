//! Abstract Syntax Tree for Sigil.
//!
//! Represents the polysynthetic structure with evidentiality markers.

use crate::span::{Span, Spanned};

// ============================================
// Attributes (for no_std, entry points, etc.)
// ============================================

/// An attribute applied to an item or crate.
/// Outer: `#[name]` or `#[name(args)]`
/// Inner: `#![name]` or `#![name(args)]`
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: Ident,
    pub args: Option<AttrArgs>,
    pub is_inner: bool, // true for #![...], false for #[...]
}

/// Arguments to an attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrArgs {
    /// Parenthesized arguments: `#[attr(a, b, c)]`
    Paren(Vec<AttrArg>),
    /// Key-value: `#[attr = "value"]`
    Eq(Box<Expr>),
}

/// A single argument in an attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrArg {
    /// Simple identifier: `no_std`
    Ident(Ident),
    /// Literal value: `"entry"`
    Literal(Literal),
    /// Nested attribute: `cfg(target_os = "none")`
    Nested(Attribute),
    /// Key-value pair: `target = "x86_64"`
    KeyValue { key: Ident, value: Box<Expr> },
}

/// Crate-level configuration derived from inner attributes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CrateConfig {
    /// `#![no_std]` - disable standard library
    pub no_std: bool,
    /// `#![no_main]` - custom entry point, no main function required
    pub no_main: bool,
    /// `#![feature(...)]` - enabled features
    pub features: Vec<String>,
    /// Target architecture (from `#![target(...)]`)
    pub target: Option<TargetConfig>,
    /// Linker configuration
    pub linker: Option<LinkerConfig>,
}

/// Target-specific configuration.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TargetConfig {
    pub arch: Option<String>,  // "x86_64", "aarch64", "riscv64"
    pub os: Option<String>,    // "none", "linux", "windows"
    pub abi: Option<String>,   // "gnu", "musl", "msvc"
    pub features: Vec<String>, // CPU features
}

/// Linker configuration for bare-metal/OS development.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LinkerConfig {
    /// Path to linker script: `#![linker_script = "link.ld"]`
    pub script: Option<String>,
    /// Entry point symbol: `#![entry_point = "_start"]`
    pub entry_point: Option<String>,
    /// Base address for code: `#![base_address = 0x100000]`
    pub base_address: Option<u64>,
    /// Stack size: `#![stack_size = 0x4000]`
    pub stack_size: Option<u64>,
    /// Additional linker flags
    pub flags: Vec<String>,
    /// Memory regions for embedded targets
    pub memory_regions: Vec<MemoryRegion>,
    /// Output sections configuration
    pub sections: Vec<SectionConfig>,
}

/// Memory region for linker scripts.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRegion {
    pub name: String,
    pub origin: u64,
    pub length: u64,
    pub attributes: MemoryAttributes,
}

/// Memory region attributes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MemoryAttributes {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

/// Output section configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SectionConfig {
    pub name: String,
    pub address: Option<u64>,
    pub align: Option<u64>,
    pub region: Option<String>,
}

// ============================================
// SIMD Types for Game Engine / High-Performance
// ============================================

/// SIMD vector type.
#[derive(Debug, Clone, PartialEq)]
pub struct SimdType {
    /// Element type (f32, f64, i32, etc.)
    pub element_type: Box<TypeExpr>,
    /// Number of lanes (2, 4, 8, 16)
    pub lanes: u8,
}

/// SIMD intrinsic operations.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum SimdOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Min,
    Max,

    // Comparison (returns mask)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Horizontal operations
    HAdd, // Horizontal add (sum all lanes)
    Dot,  // Dot product

    // Shuffle/permute
    Shuffle,
    Blend,

    // Load/store
    Load,
    Store,
    LoadAligned,
    StoreAligned,

    // Conversions
    Cast,
    Widen,
    Narrow,

    // Math functions
    Sqrt,
    Rsqrt, // Reciprocal square root
    Rcp,   // Reciprocal
    Floor,
    Ceil,
    Round,

    // Bitwise
    And,
    Or,
    Xor,
    Not,
    Shl,
    Shr,
}

// ============================================
// Atomic Types for Concurrency
// ============================================

/// Atomic memory ordering.
#[derive(Debug, Clone, PartialEq, Copy, Default)]
pub enum MemoryOrdering {
    /// No ordering constraints
    Relaxed,
    /// Acquire semantics (for loads)
    Acquire,
    /// Release semantics (for stores)
    Release,
    /// Acquire + Release
    AcqRel,
    /// Sequential consistency
    #[default]
    SeqCst,
}

/// Atomic operations.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum AtomicOp {
    Load,
    Store,
    Swap,
    CompareExchange,
    CompareExchangeWeak,
    FetchAdd,
    FetchSub,
    FetchAnd,
    FetchOr,
    FetchXor,
    FetchMin,
    FetchMax,
}

// ============================================
// Derive Macros
// ============================================

/// Built-in derivable traits.
#[derive(Debug, Clone, PartialEq)]
pub enum DeriveTrait {
    // Standard traits
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,

    // Game engine specific
    Component, // ECS component marker
    Resource,  // ECS resource marker
    Bundle,    // ECS component bundle

    // Serialization
    Serialize,
    Deserialize,

    // Custom derive (name of the derive macro)
    Custom(String),
}

/// Struct attributes including derive.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StructAttrs {
    /// Memory representation
    pub repr: Option<StructRepr>,
    /// Packed layout (no padding)
    pub packed: bool,
    /// Alignment override
    pub align: Option<usize>,
    /// SIMD alignment/vectorization hints
    pub simd: bool,
    /// Derived traits
    pub derives: Vec<DeriveTrait>,
    /// Outer attributes
    pub outer_attrs: Vec<Attribute>,
}

// ============================================
// Allocator Trait Support
// ============================================

/// Built-in trait definitions for game engines.
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinTrait {
    /// Memory allocator trait
    Allocator,
    /// Iterator trait
    Iterator,
    /// Into iterator
    IntoIterator,
    /// Drop/destructor
    Drop,
    /// Default construction
    Default,
    /// Deref coercion
    Deref,
    /// DerefMut coercion
    DerefMut,
    /// Index access
    Index,
    /// IndexMut access
    IndexMut,
}

/// A complete Sigil source file.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceFile {
    /// Inner attributes: `#![no_std]`, `#![feature(...)]`
    pub attrs: Vec<Attribute>,
    /// Crate configuration derived from attributes
    pub config: CrateConfig,
    /// Top-level items
    pub items: Vec<Spanned<Item>>,
}

/// Top-level items.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Function(Function),
    Struct(StructDef),
    Enum(EnumDef),
    Bitflags(BitflagsDef),
    Trait(TraitDef),
    Impl(ImplBlock),
    TypeAlias(TypeAlias),
    Module(Module),
    Use(UseDecl),
    Const(ConstDef),
    Static(StaticDef),
    Actor(ActorDef),
    ExternBlock(ExternBlock),
}

/// Foreign function interface block.
/// `extern "C" { fn foo(x: c_int) -> c_int; }`
#[derive(Debug, Clone, PartialEq)]
pub struct ExternBlock {
    pub abi: String, // "C", "Rust", "system", etc.
    pub items: Vec<ExternItem>,
}

/// Items that can appear in an extern block.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternItem {
    Function(ExternFunction),
    Static(ExternStatic),
}

/// Foreign function declaration (no body).
#[derive(Debug, Clone, PartialEq)]
pub struct ExternFunction {
    pub visibility: Visibility,
    pub name: Ident,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub variadic: bool, // For C varargs: fn printf(fmt: *const c_char, ...)
}

/// Foreign static variable.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternStatic {
    pub visibility: Visibility,
    pub mutable: bool,
    pub name: Ident,
    pub ty: TypeExpr,
}

/// Function attributes for low-level control.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FunctionAttrs {
    /// Naked function (no prologue/epilogue)
    pub naked: bool,
    /// Inline hint
    pub inline: Option<InlineHint>,
    /// Calling convention override
    pub calling_convention: Option<String>,
    /// No mangle (preserve symbol name)
    pub no_mangle: bool,
    /// Link section (e.g., ".text.boot", ".init")
    pub link_section: Option<String>,
    /// Export as C function
    pub export: bool,
    /// Panic handler function
    pub panic_handler: bool,
    /// Entry point function
    pub entry: bool,
    /// Interrupt handler with interrupt number
    pub interrupt: Option<u32>,
    /// Align function to boundary
    pub align: Option<usize>,
    /// Cold function (unlikely to be called)
    pub cold: bool,
    /// Hot function (likely to be called)
    pub hot: bool,
    /// Test function
    pub test: bool,
    /// Outer attributes (raw, for any we don't recognize)
    pub outer_attrs: Vec<Attribute>,
}

/// Inline hints for functions.
#[derive(Debug, Clone, PartialEq)]
pub enum InlineHint {
    /// #[inline]
    Hint,
    /// #[inline(always)]
    Always,
    /// #[inline(never)]
    Never,
}

/// Verb aspect for function semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aspect {
    /// ·ing - Progressive aspect: ongoing/streaming operation
    Progressive,
    /// ·ed - Perfective aspect: completed operation
    Perfective,
    /// ·able - Potential aspect: capability check (returns bool)
    Potential,
    /// ·ive - Resultative aspect: produces a result
    Resultative,
}

/// Function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub visibility: Visibility,
    pub is_async: bool,
    pub attrs: FunctionAttrs,
    /// Documentation comment: `//! This function does X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    pub aspect: Option<Aspect>, // Verb aspect: ·ing, ·ed, ·able, ·ive
    pub generics: Option<Generics>,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub where_clause: Option<WhereClause>,
    pub body: Option<Block>,
}

/// Visibility modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    #[default]
    Private,
    Public,
    Crate,
    Super,
}

/// Identifier with optional evidentiality and affect markers.
#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub name: String,
    pub evidentiality: Option<Evidentiality>,
    pub affect: Option<Affect>,
    pub span: Span,
}

/// Evidentiality markers from the type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidentiality {
    Known,     // !
    Uncertain, // ?
    Reported,  // ~
    Paradox,   // ‽
}

/// Affective markers for sentiment and emotion tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct Affect {
    pub sentiment: Option<Sentiment>,
    pub sarcasm: bool, // ⸮
    pub intensity: Option<Intensity>,
    pub formality: Option<Formality>,
    pub emotion: Option<Emotion>,
    pub confidence: Option<Confidence>,
}

impl Default for Affect {
    fn default() -> Self {
        Affect {
            sentiment: None,
            sarcasm: false,
            intensity: None,
            formality: None,
            emotion: None,
            confidence: None,
        }
    }
}

impl Affect {
    pub fn is_empty(&self) -> bool {
        self.sentiment.is_none()
            && !self.sarcasm
            && self.intensity.is_none()
            && self.formality.is_none()
            && self.emotion.is_none()
            && self.confidence.is_none()
    }
}

/// Sentiment polarity markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sentiment {
    Positive, // ⊕
    Negative, // ⊖
    Neutral,  // ⊜
}

/// Intensity modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intensity {
    Up,   // ↑ (intensifier)
    Down, // ↓ (dampener)
    Max,  // ⇈ (maximum)
}

/// Formality register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formality {
    Formal,   // ♔
    Informal, // ♟
}

/// Emotion categories (Plutchik's wheel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emotion {
    Joy,      // ☺
    Sadness,  // ☹
    Anger,    // ⚡
    Fear,     // ❄
    Surprise, // ✦
    Love,     // ♡
}

/// Confidence level markers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    High,   // ◉
    Medium, // ◎
    Low,    // ○
}

/// Function parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub pattern: Pattern,
    pub ty: TypeExpr,
}

/// Generic parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Generics {
    pub params: Vec<GenericParam>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenericParam {
    Type {
        name: Ident,
        bounds: Vec<TypeExpr>,
        evidentiality: Option<Evidentiality>,
    },
    Const {
        name: Ident,
        ty: TypeExpr,
    },
    Lifetime(String),
}

/// Where clause for bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct WhereClause {
    pub predicates: Vec<WherePredicate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WherePredicate {
    pub ty: TypeExpr,
    pub bounds: Vec<TypeExpr>,
}

/// Type expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    /// Simple named type: `i32`, `String`
    Path(TypePath),
    /// Reference: `&T`, `&mut T`
    Reference { mutable: bool, inner: Box<TypeExpr> },
    /// Pointer: `*const T`, `*mut T`
    Pointer { mutable: bool, inner: Box<TypeExpr> },
    /// Array: `[T; N]`
    Array {
        element: Box<TypeExpr>,
        size: Box<Expr>,
    },
    /// Slice: `[T]`
    Slice(Box<TypeExpr>),
    /// Tuple: `(A, B, C)`
    Tuple(Vec<TypeExpr>),
    /// Function: `fn(A, B) -> C`
    Function {
        params: Vec<TypeExpr>,
        return_type: Option<Box<TypeExpr>>,
    },
    /// With evidentiality: `T!`, `T?`, `T~`, or with error type: `T?[Error]`, `T![Error]`
    /// The error type syntax is sugar for Result<T, Error> with evidentiality tracking:
    /// - `T?[E]` means Result<T, E> with uncertain evidentiality
    /// - `T![E]` means Result<T, E> with known/infallible path
    /// - `T~[E]` means Result<T, E> from external/reported source
    Evidential {
        inner: Box<TypeExpr>,
        evidentiality: Evidentiality,
        /// Optional error type for Result sugar: `T?[ErrorType]`
        error_type: Option<Box<TypeExpr>>,
    },
    /// Cycle type: `Cycle<N>`
    Cycle { modulus: Box<Expr> },
    /// SIMD vector type: `simd<f32, 4>`, `Vec4f`
    Simd { element: Box<TypeExpr>, lanes: u8 },
    /// Atomic type: `atomic<T>`
    Atomic(Box<TypeExpr>),
    /// Never type: `!` or `never`
    Never,
    /// Inferred: `_`
    Infer,
}

/// Type path with optional generics.
#[derive(Debug, Clone, PartialEq)]
pub struct TypePath {
    pub segments: Vec<PathSegment>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathSegment {
    pub ident: Ident,
    pub generics: Option<Vec<TypeExpr>>,
}

/// Memory representation for structs.
#[derive(Debug, Clone, PartialEq)]
pub enum StructRepr {
    /// C-compatible layout
    C,
    /// Transparent (single field struct)
    Transparent,
    /// Specific integer type
    Int(String),
}

/// Struct definition.
#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub visibility: Visibility,
    pub attrs: StructAttrs,
    /// Documentation comment: `//! This struct represents X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    /// Type-level evidentiality marker: `struct Foo!`, `struct Bar~`, etc.
    pub evidentiality: Option<Evidentiality>,
    pub generics: Option<Generics>,
    pub fields: StructFields,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StructFields {
    Named(Vec<FieldDef>),
    Tuple(Vec<TypeExpr>),
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    /// Outer attributes: `#[serde(skip)]`
    pub attrs: Vec<Attribute>,
    pub visibility: Visibility,
    /// Documentation comment: `/// This field represents X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    pub ty: TypeExpr,
    /// Default value for the field (e.g., `field: Type = default_expr`)
    pub default: Option<Expr>,
}

/// Enum definition.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumDef {
    pub visibility: Visibility,
    /// Documentation comment: `//! This enum represents X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    /// Type-level evidentiality marker: `enum Foo!`, `enum Bar~`, etc.
    pub evidentiality: Option<Evidentiality>,
    pub generics: Option<Generics>,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    /// Outer attributes: `#[deprecated]`
    pub attrs: Vec<Attribute>,
    /// Documentation comment: `//! This variant represents X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    pub fields: StructFields,
    pub discriminant: Option<Expr>,
}

/// Bitflags definition: `bitflags Flags: u32 { FLAG_A = 0x01, FLAG_B = 0x02 }`
#[derive(Debug, Clone, PartialEq)]
pub struct BitflagsDef {
    pub visibility: Visibility,
    /// Documentation comment
    pub doc_comment: Option<String>,
    pub name: Ident,
    /// The underlying integer type (e.g., u32, u64)
    pub repr_type: TypeExpr,
    /// The flag constants
    pub flags: Vec<BitflagItem>,
}

/// A single bitflag constant
#[derive(Debug, Clone, PartialEq)]
pub struct BitflagItem {
    pub doc_comment: Option<String>,
    pub name: Ident,
    pub value: Expr,
}

/// Trait definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitDef {
    pub visibility: Visibility,
    /// Documentation comment: `//! This trait defines X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    pub generics: Option<Generics>,
    pub supertraits: Vec<TypeExpr>,
    pub items: Vec<TraitItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraitItem {
    Function(Function),
    Type {
        doc_comment: Option<String>,
        name: Ident,
        bounds: Vec<TypeExpr>,
    },
    Const {
        doc_comment: Option<String>,
        name: Ident,
        ty: TypeExpr,
    },
}

/// Impl block.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    /// Documentation comment: `//! Implementation of X for Y`
    pub doc_comment: Option<String>,
    pub generics: Option<Generics>,
    pub trait_: Option<TypePath>,
    pub self_ty: TypeExpr,
    pub items: Vec<ImplItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImplItem {
    Function(Function),
    Type(TypeAlias),
    Const(ConstDef),
}

/// Type alias.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeAlias {
    pub visibility: Visibility,
    /// Documentation comment: `//! This type alias represents X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    pub generics: Option<Generics>,
    pub ty: TypeExpr,
}

/// Module.
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub visibility: Visibility,
    pub name: Ident,
    pub items: Option<Vec<Spanned<Item>>>,
}

/// Use declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct UseDecl {
    pub visibility: Visibility,
    pub tree: UseTree,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UseTree {
    Path { prefix: Ident, suffix: Box<UseTree> },
    Name(Ident),
    Rename { name: Ident, alias: Ident },
    Glob,
    Group(Vec<UseTree>),
}

/// Const definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef {
    pub visibility: Visibility,
    /// Documentation comment: `//! This constant represents X`
    pub doc_comment: Option<String>,
    pub name: Ident,
    pub ty: TypeExpr,
    pub value: Expr,
}

/// Static definition.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticDef {
    pub visibility: Visibility,
    /// Documentation comment: `//! This static variable represents X`
    pub doc_comment: Option<String>,
    pub mutable: bool,
    pub name: Ident,
    pub ty: TypeExpr,
    pub value: Expr,
}

/// Actor definition (concurrency).
#[derive(Debug, Clone, PartialEq)]
pub struct ActorDef {
    pub visibility: Visibility,
    pub name: Ident,
    pub generics: Option<Generics>,
    pub state: Vec<FieldDef>,
    pub methods: Vec<Function>,
    pub handlers: Vec<MessageHandler>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MessageHandler {
    pub message: Ident,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Block,
}

/// Code block.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub expr: Option<Box<Expr>>,
}

/// Statements.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        attrs: Vec<Attribute>,
        pattern: Pattern,
        ty: Option<TypeExpr>,
        init: Option<Expr>,
    },
    Expr(Expr),
    Semi(Expr),
    Item(Box<Item>),
}

/// Patterns.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Ident {
        mutable: bool,
        name: Ident,
        evidentiality: Option<Evidentiality>,
    },
    Tuple(Vec<Pattern>),
    Struct {
        path: TypePath,
        fields: Vec<FieldPattern>,
        rest: bool,
    },
    TupleStruct {
        path: TypePath,
        fields: Vec<Pattern>,
    },
    Slice(Vec<Pattern>),
    Or(Vec<Pattern>),
    Literal(Literal),
    Range {
        start: Option<Box<Pattern>>,
        end: Option<Box<Pattern>>,
        inclusive: bool,
    },
    Wildcard,
    Rest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldPattern {
    pub name: Ident,
    pub pattern: Option<Pattern>,
}

/// Expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value
    Literal(Literal),
    /// Variable reference
    Path(TypePath),
    /// Binary operation
    Binary {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
    },
    /// Unary operation
    Unary { op: UnaryOp, expr: Box<Expr> },
    /// Pipe expression: `x|f|g`
    Pipe {
        expr: Box<Expr>,
        operations: Vec<PipeOp>,
    },
    /// Incorporation: `file·open·read`
    Incorporation { segments: Vec<IncorporationSegment> },
    /// Morpheme application: `τ{f}`
    Morpheme { kind: MorphemeKind, body: Box<Expr> },
    /// Function call
    Call { func: Box<Expr>, args: Vec<Expr> },
    /// Method call with optional turbofish syntax: `receiver.method::<T>(args)`
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        turbofish: Option<Vec<TypeExpr>>,
        args: Vec<Expr>,
    },
    /// Field access
    Field { expr: Box<Expr>, field: Ident },
    /// Index: `arr[i]`
    Index { expr: Box<Expr>, index: Box<Expr> },
    /// Array literal
    Array(Vec<Expr>),
    /// Tuple literal
    Tuple(Vec<Expr>),
    /// Struct literal
    Struct {
        path: TypePath,
        fields: Vec<FieldInit>,
        rest: Option<Box<Expr>>,
    },
    /// Block expression
    Block(Block),
    /// If expression
    If {
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Box<Expr>>,
    },
    /// Match expression
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    /// Loop
    Loop(Block),
    /// While loop
    While { condition: Box<Expr>, body: Block },
    /// While let pattern matching: `while let Some(x) = iter.next() { ... }`
    WhileLet {
        pattern: Pattern,
        expr: Box<Expr>,
        body: Block,
    },
    /// If let pattern matching: `if let Some(x) = value { ... }`
    IfLet {
        pattern: Pattern,
        expr: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Box<Expr>>,
    },
    /// For loop
    For {
        pattern: Pattern,
        iter: Box<Expr>,
        body: Block,
    },
    /// Closure: `{x => x + 1}` or `|x| x + 1`
    Closure {
        params: Vec<ClosureParam>,
        body: Box<Expr>,
    },
    /// Await with optional evidentiality: `expr⌛` or `expr⌛?` or `expr⌛!` or `expr⌛~`
    /// The evidentiality marker specifies how to handle the awaited result:
    /// - `⌛?` - await and propagate error (uncertain)
    /// - `⌛!` - await, expect success (known/infallible)
    /// - `⌛~` - await external/reported source
    /// - `⌛‽` - await with trust boundary crossing
    Await {
        expr: Box<Expr>,
        evidentiality: Option<Evidentiality>,
    },
    /// Try: `expr?`
    Try(Box<Expr>),
    /// Return
    Return(Option<Box<Expr>>),
    /// Break
    Break(Option<Box<Expr>>),
    /// Continue
    Continue,
    /// Range: `a..b` or `a..=b`
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },
    /// Macro invocation
    Macro { path: TypePath, tokens: String },
    /// With evidentiality marker
    Evidential {
        expr: Box<Expr>,
        evidentiality: Evidentiality,
    },
    /// Assignment: `x = value`
    Assign { target: Box<Expr>, value: Box<Expr> },
    /// Unsafe block: `unsafe { ... }`
    Unsafe(Block),
    /// Raw pointer dereference: `*ptr`
    Deref(Box<Expr>),
    /// Address-of: `&expr` or `&mut expr`
    AddrOf { mutable: bool, expr: Box<Expr> },
    /// Cast: `expr as Type`
    Cast { expr: Box<Expr>, ty: TypeExpr },
    /// Channel send: `channel <- value`
    ChannelSend {
        channel: Box<Expr>,
        value: Box<Expr>,
    },

    /// Inline assembly: `asm!("instruction", ...)`
    InlineAsm(InlineAsm),

    /// Volatile read: `volatile read<T>(ptr)` or `volatile read(ptr)`
    VolatileRead {
        ptr: Box<Expr>,
        ty: Option<TypeExpr>,
    },

    /// Volatile write: `volatile write<T>(ptr, value)` or `volatile write(ptr, value)`
    VolatileWrite {
        ptr: Box<Expr>,
        value: Box<Expr>,
        ty: Option<TypeExpr>,
    },

    // ========================================
    // SIMD Operations
    // ========================================
    /// SIMD vector literal: `simd[1.0, 2.0, 3.0, 4.0]`
    SimdLiteral {
        elements: Vec<Expr>,
        ty: Option<TypeExpr>,
    },

    /// SIMD intrinsic operation: `simd::add(a, b)`
    SimdIntrinsic { op: SimdOp, args: Vec<Expr> },

    /// SIMD shuffle: `simd::shuffle(a, b, [0, 4, 1, 5])`
    SimdShuffle {
        a: Box<Expr>,
        b: Box<Expr>,
        indices: Vec<u8>,
    },

    /// SIMD splat (broadcast scalar to all lanes): `simd::splat(x)`
    SimdSplat { value: Box<Expr>, lanes: u8 },

    /// SIMD extract single lane: `simd::extract(v, 0)`
    SimdExtract { vector: Box<Expr>, index: u8 },

    /// SIMD insert into lane: `simd::insert(v, 0, value)`
    SimdInsert {
        vector: Box<Expr>,
        index: u8,
        value: Box<Expr>,
    },

    // ========================================
    // Atomic Operations
    // ========================================
    /// Atomic operation: `atomic::load(&x, Ordering::SeqCst)`
    AtomicOp {
        op: AtomicOp,
        ptr: Box<Expr>,
        value: Option<Box<Expr>>,    // For store, swap, fetch_add, etc.
        expected: Option<Box<Expr>>, // For compare_exchange
        ordering: MemoryOrdering,
        failure_ordering: Option<MemoryOrdering>, // For compare_exchange
    },

    /// Atomic fence: `atomic::fence(Ordering::SeqCst)`
    AtomicFence { ordering: MemoryOrdering },

    // ==========================================
    // Protocol Expressions - Sigil-native networking
    // All protocol expressions yield values with Reported evidentiality
    // ==========================================
    /// HTTP request: `http·get(url)`, `http·post(url)`, etc.
    /// Incorporation pattern for building HTTP requests
    HttpRequest {
        method: HttpMethod,
        url: Box<Expr>,
        headers: Vec<(Expr, Expr)>,
        body: Option<Box<Expr>>,
        timeout: Option<Box<Expr>>,
    },

    /// gRPC call: `grpc·call(service, method)`
    /// Incorporation pattern for gRPC operations
    GrpcCall {
        service: Box<Expr>,
        method: Box<Expr>,
        message: Option<Box<Expr>>,
        metadata: Vec<(Expr, Expr)>,
        timeout: Option<Box<Expr>>,
    },

    /// WebSocket connection: `ws·connect(url)`
    /// Incorporation pattern for WebSocket operations
    WebSocketConnect {
        url: Box<Expr>,
        protocols: Vec<Expr>,
        headers: Vec<(Expr, Expr)>,
    },

    /// WebSocket message: `ws·message(data)` or `ws·text(str)` / `ws·binary(bytes)`
    WebSocketMessage {
        kind: WebSocketMessageKind,
        data: Box<Expr>,
    },

    /// Kafka operation: `kafka·produce(topic, message)`, `kafka·consume(topic)`
    KafkaOp {
        kind: KafkaOpKind,
        topic: Box<Expr>,
        payload: Option<Box<Expr>>,
        key: Option<Box<Expr>>,
        partition: Option<Box<Expr>>,
    },

    /// GraphQL query: `graphql·query(document)` or `graphql·mutation(document)`
    GraphQLOp {
        kind: GraphQLOpKind,
        document: Box<Expr>,
        variables: Option<Box<Expr>>,
        operation_name: Option<Box<Expr>>,
    },

    /// Protocol stream: iterable over network messages
    /// `http·stream(url)`, `ws·stream()`, `kafka·stream(topic)`
    ProtocolStream {
        protocol: ProtocolKind,
        source: Box<Expr>,
        config: Option<Box<Expr>>,
    },
}

/// Inline assembly expression.
#[derive(Debug, Clone, PartialEq)]
pub struct InlineAsm {
    /// The assembly template string
    pub template: String,
    /// Output operands
    pub outputs: Vec<AsmOperand>,
    /// Input operands
    pub inputs: Vec<AsmOperand>,
    /// Clobbered registers
    pub clobbers: Vec<String>,
    /// Assembly options
    pub options: AsmOptions,
}

/// An assembly operand (input or output).
#[derive(Debug, Clone, PartialEq)]
pub struct AsmOperand {
    /// Constraint string (e.g., "=r", "r", "m", or register name)
    pub constraint: String,
    /// The expression to bind
    pub expr: Expr,
    /// The kind of operand
    pub kind: AsmOperandKind,
    /// For inout operands, the output expression (if different from input)
    pub output: Option<Box<Expr>>,
}

/// Assembly operand kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsmOperandKind {
    /// Input only
    Input,
    /// Output only
    Output,
    /// Input and output (same register)
    InOut,
}

/// Assembly options/modifiers.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AsmOptions {
    /// Volatile (has side effects, cannot be optimized away)
    pub volatile: bool,
    /// Pure assembly (no side effects)
    pub pure_asm: bool,
    /// No memory clobber
    pub nomem: bool,
    /// No stack
    pub nostack: bool,
    /// Preserves flags
    pub preserves_flags: bool,
    /// Read-only (inputs only)
    pub readonly: bool,
    /// AT&T syntax (vs Intel)
    pub att_syntax: bool,
}

/// Pipe operation in a chain.
#[derive(Debug, Clone, PartialEq)]
pub enum PipeOp {
    /// Transform morpheme: `τ{f}`
    Transform(Box<Expr>),
    /// Filter morpheme: `φ{p}`
    Filter(Box<Expr>),
    /// Sort morpheme: `σ` or `σ.field`
    Sort(Option<Ident>),
    /// Reduce morpheme: `ρ{f}`
    Reduce(Box<Expr>),
    /// Sum reduction: `ρ+` or `ρ_sum` - sum all elements
    ReduceSum,
    /// Product reduction: `ρ*` or `ρ_prod` - multiply all elements
    ReduceProd,
    /// Min reduction: `ρ_min` - find minimum element
    ReduceMin,
    /// Max reduction: `ρ_max` - find maximum element
    ReduceMax,
    /// Concat reduction: `ρ++` or `ρ_cat` - concatenate strings/arrays
    ReduceConcat,
    /// All reduction: `ρ&` or `ρ_all` - logical AND (all true)
    ReduceAll,
    /// Any reduction: `ρ|` or `ρ_any` - logical OR (any true)
    ReduceAny,
    /// Middle morpheme: `μ` - get middle/median element
    Middle,
    /// Choice morpheme: `χ` - random element selection
    Choice,
    /// Nth morpheme: `ν{n}` - get nth element
    Nth(Box<Expr>),
    /// Next morpheme: `ξ` - get next element (iterator)
    Next,
    /// First morpheme: `α` - get first element
    First,
    /// Last morpheme: `ω` - get last element
    Last,
    /// Parallel morpheme: `∥{op}` or `parallel{op}` - execute operation in parallel
    /// Wraps another operation to run it across multiple threads
    Parallel(Box<PipeOp>),
    /// GPU compute morpheme: `⊛{op}` or `gpu{op}` - execute operation on GPU
    /// Wraps another operation to run it as a compute shader
    Gpu(Box<PipeOp>),
    /// Method call
    Method { name: Ident, args: Vec<Expr> },
    /// Await
    Await,
    /// Match morpheme: `|match{ Pattern => expr, ... }`
    /// Applies pattern matching to the piped value
    Match(Vec<MatchArm>),
    /// Trust boundary morpheme: `|‽` or `|‽{mapper}`
    /// Uses interrobang (‽) to signal crossing a trust/certainty boundary.
    /// Unwraps Result/Option or propagates errors, with optional error transformation.
    TryMap(Option<Box<Expr>>),
    /// Named morpheme: `·map{f}`, `·flow{f}`
    Named {
        prefix: Vec<Ident>,
        body: Option<Box<Expr>>,
    },

    // ==========================================
    // Protocol Operations - Sigil-native networking
    // ==========================================
    /// Send operation: `|send{data}` or `|⇒{data}` - send data over connection
    /// Results are automatically marked with Reported evidentiality
    Send(Box<Expr>),

    /// Receive operation: `|recv` or `|⇐` - receive data from connection
    /// Results are automatically marked with Reported evidentiality
    Recv,

    /// Stream operation: `|stream{handler}` or `|≋{handler}` - iterate over incoming data
    /// Each element is marked with Reported evidentiality
    Stream(Box<Expr>),

    /// Connect operation: `|connect{config}` or `|⊸{config}` - establish connection
    Connect(Option<Box<Expr>>),

    /// Close operation: `|close` or `|⊗` - close connection gracefully
    Close,

    /// Protocol header: `|header{name, value}` - add/set header
    Header { name: Box<Expr>, value: Box<Expr> },

    /// Protocol body: `|body{data}` - set request body
    Body(Box<Expr>),

    /// Timeout: `|timeout{ms}` or `|⏱{ms}` - set operation timeout
    Timeout(Box<Expr>),

    /// Retry: `|retry{count}` or `|retry{count, strategy}` - retry on failure
    Retry {
        count: Box<Expr>,
        strategy: Option<Box<Expr>>,
    },

    // ==========================================
    // Evidence Promotion Operations
    // ==========================================
    /// Validate operation: `|validate!{predicate}` - promote evidence with validation
    /// Takes a predicate function that validates the data.
    /// If validation passes: ~ → ! (reported → known)
    /// If validation fails: returns an error
    /// Example: `data~|validate!{x => x.id > 0 && x.name.len > 0}`
    Validate {
        predicate: Box<Expr>,
        target_evidence: Evidentiality,
    },

    /// Assume operation: `|assume!` or `|assume!("reason")` - explicit trust promotion
    /// Escape hatch that promotes evidence with an audit trail.
    /// Always logs the assumption for security review.
    /// Example: `external_data~|assume!("trusted legacy system")`
    Assume {
        reason: Option<Box<Expr>>,
        target_evidence: Evidentiality,
    },

    /// Assert evidence: `|assert_evidence!{expected}` - verify evidence level at compile time
    /// Fails compilation if actual evidence doesn't match expected.
    /// Example: `data|assert_evidence!{!}` - assert data is known
    AssertEvidence(Evidentiality),

    // ==========================================
    // Scope Functions (Kotlin-inspired)
    // ==========================================
    /// Also: `|also{f}` - execute side effect, return original value unchanged
    /// Like Kotlin's `also` - useful for logging/debugging in pipelines
    /// Example: `data|also{println}|process` - logs data, then processes it
    Also(Box<Expr>),

    /// Apply: `|apply{block}` - mutate value in place, return modified value
    /// Like Kotlin's `apply` - useful for configuration/setup
    /// Example: `config|apply{.timeout = 5000; .retries = 3}`
    Apply(Box<Expr>),

    /// TakeIf: `|take_if{predicate}` - return Some(value) if predicate true, None otherwise
    /// Like Kotlin's `takeIf` - useful for conditional pipelines
    /// Example: `user|take_if{.age >= 18}` - returns Option<User>
    TakeIf(Box<Expr>),

    /// TakeUnless: `|take_unless{predicate}` - return Some(value) if predicate false
    /// Like Kotlin's `takeUnless` - inverse of take_if
    /// Example: `item|take_unless{.is_deleted}` - returns Option<Item>
    TakeUnless(Box<Expr>),

    /// Let: `|let{f}` - transform value, like map but reads better for single values
    /// Like Kotlin's `let` - essentially an alias for transform
    /// Example: `name|let{.to_uppercase}` - transforms the name
    Let(Box<Expr>),

    // ==========================================
    // Mathematical & APL-Inspired Operations
    // ==========================================
    /// All/ForAll: `|∀{p}` or `|all{p}` - check if ALL elements satisfy predicate
    /// Returns bool. Short-circuits on first false.
    /// Example: `numbers|∀{x => x > 0}` - are all positive?
    All(Box<Expr>),

    /// Any/Exists: `|∃{p}` or `|any{p}` - check if ANY element satisfies predicate
    /// Returns bool. Short-circuits on first true.
    /// Example: `items|∃{.is_valid}` - is any valid?
    Any(Box<Expr>),

    /// Compose: `|∘{f}` or `|compose{f}` - function composition
    /// Creates a new function that applies f after the current transformation.
    /// Example: `parse|∘{validate}|∘{save}` - compose three functions
    Compose(Box<Expr>),

    /// Zip/Join: `|⋈{other}` or `|zip{other}` - combine with another collection
    /// Pairs elements from two collections into tuples.
    /// Example: `names|⋈{ages}` -> [(name1, age1), (name2, age2), ...]
    Zip(Box<Expr>),

    /// Scan/Integral: `|∫{f}` or `|scan{f}` - cumulative fold (like Haskell's scanl)
    /// Returns all intermediate accumulator values.
    /// Example: `[1,2,3]|∫{+}` -> [1, 3, 6] (running sum)
    Scan(Box<Expr>),

    /// Diff/Derivative: `|∂` or `|diff` - differences between adjacent elements
    /// Returns a collection of deltas.
    /// Example: `[1, 4, 6, 10]|∂` -> [3, 2, 4]
    Diff,

    /// Gradient: `|∇{var}` or `|grad{var}` - automatic differentiation
    /// Computes gradient of expression with respect to variable.
    /// Example: `loss|∇{weights}` - gradient for backprop
    Gradient(Box<Expr>),

    /// Sort Ascending: `|⍋` or `|sort_asc` - APL grade-up
    /// Sorts in ascending order (same as σ but more explicit)
    SortAsc,

    /// Sort Descending: `|⍒` or `|sort_desc` - APL grade-down
    /// Sorts in descending order
    SortDesc,

    /// Reverse: `|⌽` or `|rev` - APL rotate/reverse
    /// Reverses the collection
    Reverse,

    /// Cycle: `|↻{n}` or `|cycle{n}` - repeat collection n times
    /// Example: `[1,2]|↻{3}` -> [1,2,1,2,1,2]
    Cycle(Box<Expr>),

    /// Windows: `|⌺{n}` or `|windows{n}` - sliding window
    /// Example: `[1,2,3,4]|⌺{2}` -> [[1,2], [2,3], [3,4]]
    Windows(Box<Expr>),

    /// Chunks: `|⊞{n}` or `|chunks{n}` - split into chunks
    /// Example: `[1,2,3,4]|⊞{2}` -> [[1,2], [3,4]]
    Chunks(Box<Expr>),

    /// Flatten: `|⋳` or `|flatten` - flatten nested collection
    /// Example: `[[1,2], [3,4]]|⋳` -> [1,2,3,4]
    Flatten,

    /// Unique: `|∪` or `|unique` - remove duplicates (set union with self)
    /// Example: `[1,2,2,3,3,3]|∪` -> [1,2,3]
    Unique,

    /// Enumerate: `|⍳` or `|enumerate` - APL iota, pair with indices
    /// Example: `["a","b","c"]|⍳` -> [(0,"a"), (1,"b"), (2,"c")]
    Enumerate,
}

/// Incorporation segment.
#[derive(Debug, Clone, PartialEq)]
pub struct IncorporationSegment {
    pub name: Ident,
    pub args: Option<Vec<Expr>>,
}

/// Morpheme kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphemeKind {
    Transform, // τ
    Filter,    // φ
    Sort,      // σ
    Reduce,    // ρ
    Lambda,    // λ
    Sum,       // Σ
    Product,   // Π
    Middle,    // μ
    Choice,    // χ
    Nth,       // ν
    Next,      // ξ
    First,     // α
    Last,      // ω
}

// ==========================================
// Protocol Types - Sigil-native networking
// ==========================================

/// HTTP methods for http· incorporation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    Connect,
    Trace,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
            HttpMethod::Put => write!(f, "PUT"),
            HttpMethod::Delete => write!(f, "DELETE"),
            HttpMethod::Patch => write!(f, "PATCH"),
            HttpMethod::Head => write!(f, "HEAD"),
            HttpMethod::Options => write!(f, "OPTIONS"),
            HttpMethod::Connect => write!(f, "CONNECT"),
            HttpMethod::Trace => write!(f, "TRACE"),
        }
    }
}

/// WebSocket message kinds for ws· incorporation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSocketMessageKind {
    Text,
    Binary,
    Ping,
    Pong,
    Close,
}

/// Kafka operation kinds for kafka· incorporation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KafkaOpKind {
    Produce,
    Consume,
    Subscribe,
    Unsubscribe,
    Commit,
    Seek,
}

/// GraphQL operation kinds for graphql· incorporation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphQLOpKind {
    Query,
    Mutation,
    Subscription,
}

/// Protocol kinds for stream expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolKind {
    Http,
    Grpc,
    WebSocket,
    Kafka,
    Amqp,
    GraphQL,
}

impl std::fmt::Display for ProtocolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolKind::Http => write!(f, "http"),
            ProtocolKind::Grpc => write!(f, "grpc"),
            ProtocolKind::WebSocket => write!(f, "ws"),
            ProtocolKind::Kafka => write!(f, "kafka"),
            ProtocolKind::Amqp => write!(f, "amqp"),
            ProtocolKind::GraphQL => write!(f, "graphql"),
        }
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Concat,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    Deref,
    Ref,
    RefMut,
}

/// Literal values.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int {
        value: String,
        base: NumBase,
        suffix: Option<String>,
    },
    Float {
        value: String,
        suffix: Option<String>,
    },
    String(String),
    /// Multi-line string literal (""" ... """)
    MultiLineString(String),
    /// Raw string literal (no escape processing)
    RawString(String),
    /// Byte string literal (b"...")
    ByteString(Vec<u8>),
    /// Interpolated string literal (f"... {expr} ...")
    InterpolatedString {
        parts: Vec<InterpolationPart>,
    },
    /// SQL sigil string (σ"...")
    SigilStringSql(String),
    /// Route sigil string (ρ"...")
    SigilStringRoute(String),
    Char(char),
    Bool(bool),
    Null, // null
    /// Special mathematical constants
    Empty, // ∅
    Infinity, // ∞
    Circle, // ◯
}

/// Part of an interpolated string
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationPart {
    /// Literal text segment
    Text(String),
    /// Expression to be evaluated and converted to string
    Expr(Box<Expr>),
}

/// Number bases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumBase {
    Binary,       // 0b
    Octal,        // 0o
    Decimal,      // default
    Hex,          // 0x
    Vigesimal,    // 0v (base 20)
    Sexagesimal,  // 0s (base 60)
    Duodecimal,   // 0z (base 12)
    Explicit(u8), // N:₆₀ etc.
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldInit {
    pub name: Ident,
    pub value: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub attrs: Vec<Attribute>,
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClosureParam {
    pub pattern: Pattern,
    pub ty: Option<TypeExpr>,
}

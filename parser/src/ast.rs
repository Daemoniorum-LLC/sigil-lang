//! Abstract Syntax Tree for Sigil.
//!
//! Represents the polysynthetic structure with evidentiality markers.

use crate::span::{Span, Spanned};

/// A complete Sigil source file.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceFile {
    pub items: Vec<Spanned<Item>>,
}

/// Top-level items.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Function(Function),
    Struct(StructDef),
    Enum(EnumDef),
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
    pub abi: String,  // "C", "Rust", "system", etc.
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
    pub variadic: bool,  // For C varargs: fn printf(fmt: *const c_char, ...)
}

/// Foreign static variable.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternStatic {
    pub visibility: Visibility,
    pub mutable: bool,
    pub name: Ident,
    pub ty: TypeExpr,
}

/// Function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub visibility: Visibility,
    pub is_async: bool,
    pub name: Ident,
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

/// Identifier with optional evidentiality.
#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub name: String,
    pub evidentiality: Option<Evidentiality>,
    pub span: Span,
}

/// Evidentiality markers from the type system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidentiality {
    Known,      // !
    Uncertain,  // ?
    Reported,   // ~
    Paradox,    // ‽
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
    Reference {
        mutable: bool,
        inner: Box<TypeExpr>,
    },
    /// Pointer: `*const T`, `*mut T`
    Pointer {
        mutable: bool,
        inner: Box<TypeExpr>,
    },
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
    /// With evidentiality: `T!`, `T?`, `T~`
    Evidential {
        inner: Box<TypeExpr>,
        evidentiality: Evidentiality,
    },
    /// Cycle type: `Cycle<N>`
    Cycle {
        modulus: Box<Expr>,
    },
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

/// Struct definition.
#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub visibility: Visibility,
    pub name: Ident,
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
    pub visibility: Visibility,
    pub name: Ident,
    pub ty: TypeExpr,
}

/// Enum definition.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumDef {
    pub visibility: Visibility,
    pub name: Ident,
    pub generics: Option<Generics>,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: Ident,
    pub fields: StructFields,
    pub discriminant: Option<Expr>,
}

/// Trait definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitDef {
    pub visibility: Visibility,
    pub name: Ident,
    pub generics: Option<Generics>,
    pub supertraits: Vec<TypeExpr>,
    pub items: Vec<TraitItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraitItem {
    Function(Function),
    Type {
        name: Ident,
        bounds: Vec<TypeExpr>,
    },
    Const {
        name: Ident,
        ty: TypeExpr,
    },
}

/// Impl block.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
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
    Path {
        prefix: Ident,
        suffix: Box<UseTree>,
    },
    Name(Ident),
    Rename {
        name: Ident,
        alias: Ident,
    },
    Glob,
    Group(Vec<UseTree>),
}

/// Const definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstDef {
    pub visibility: Visibility,
    pub name: Ident,
    pub ty: TypeExpr,
    pub value: Expr,
}

/// Static definition.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticDef {
    pub visibility: Visibility,
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
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// Pipe expression: `x|f|g`
    Pipe {
        expr: Box<Expr>,
        operations: Vec<PipeOp>,
    },
    /// Incorporation: `file·open·read`
    Incorporation {
        segments: Vec<IncorporationSegment>,
    },
    /// Morpheme application: `τ{f}`
    Morpheme {
        kind: MorphemeKind,
        body: Box<Expr>,
    },
    /// Function call
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    /// Method call
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        args: Vec<Expr>,
    },
    /// Field access
    Field {
        expr: Box<Expr>,
        field: Ident,
    },
    /// Index: `arr[i]`
    Index {
        expr: Box<Expr>,
        index: Box<Expr>,
    },
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
    While {
        condition: Box<Expr>,
        body: Block,
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
    /// Await: `expr|await` or `expr⌛`
    Await(Box<Expr>),
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
    Macro {
        path: TypePath,
        tokens: String,
    },
    /// With evidentiality marker
    Evidential {
        expr: Box<Expr>,
        evidentiality: Evidentiality,
    },
    /// Assignment: `x = value`
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    /// Unsafe block: `unsafe { ... }`
    Unsafe(Block),
    /// Raw pointer dereference: `*ptr`
    Deref(Box<Expr>),
    /// Address-of: `&expr` or `&mut expr`
    AddrOf {
        mutable: bool,
        expr: Box<Expr>,
    },
    /// Cast: `expr as Type`
    Cast {
        expr: Box<Expr>,
        ty: TypeExpr,
    },
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
    /// Method call
    Method {
        name: Ident,
        args: Vec<Expr>,
    },
    /// Await
    Await,
    /// Named morpheme: `·map{f}`, `·flow{f}`
    Named {
        prefix: Vec<Ident>,
        body: Option<Box<Expr>>,
    },
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
    Transform,  // τ
    Filter,     // φ
    Sort,       // σ
    Reduce,     // ρ
    Lambda,     // λ
    Sum,        // Σ
    Product,    // Π
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
    Char(char),
    Bool(bool),
    /// Special mathematical constants
    Empty,     // ∅
    Infinity,  // ∞
    Circle,    // ◯
}

/// Number bases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumBase {
    Binary,      // 0b
    Octal,       // 0o
    Decimal,     // default
    Hex,         // 0x
    Vigesimal,   // 0v (base 20)
    Sexagesimal, // 0s (base 60)
    Duodecimal,  // 0z (base 12)
    Explicit(u8),// N:₆₀ etc.
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldInit {
    pub name: Ident,
    pub value: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClosureParam {
    pub pattern: Pattern,
    pub ty: Option<TypeExpr>,
}

package com.daemoniorum.sigil.ast

import com.daemoniorum.sigil.lexer.Span
import com.daemoniorum.sigil.runtime.EvidenceLevel
import kotlinx.serialization.Serializable

/**
 * Root of the AST - a Sigil module/file
 */
@Serializable
data class Module(
    val items: List<Item>,
    val span: Span
)

/**
 * Top-level items
 */
@Serializable
sealed class Item {
    abstract val span: Span

    @Serializable
    data class Function(
        val name: String,
        val generics: List<GenericParam>,
        val params: List<Param>,
        val returnType: Type?,
        val returnEvidence: EvidenceLevel?,
        val body: Block,
        val isAsync: Boolean,
        val visibility: Visibility,
        override val span: Span
    ) : Item()

    @Serializable
    data class Struct(
        val name: String,
        val generics: List<GenericParam>,
        val fields: List<StructField>,
        val visibility: Visibility,
        override val span: Span
    ) : Item()

    @Serializable
    data class Enum(
        val name: String,
        val generics: List<GenericParam>,
        val variants: List<EnumVariant>,
        val visibility: Visibility,
        override val span: Span
    ) : Item()

    @Serializable
    data class Trait(
        val name: String,
        val generics: List<GenericParam>,
        val bounds: List<TypeBound>,
        val items: List<TraitItem>,
        val visibility: Visibility,
        override val span: Span
    ) : Item()

    @Serializable
    data class Impl(
        val generics: List<GenericParam>,
        val traitRef: TypePath?,
        val selfType: Type,
        val items: List<ImplItem>,
        override val span: Span
    ) : Item()

    @Serializable
    data class TypeAlias(
        val name: String,
        val generics: List<GenericParam>,
        val ty: Type,
        val visibility: Visibility,
        override val span: Span
    ) : Item()

    @Serializable
    data class Use(
        val path: UsePath,
        val visibility: Visibility,
        override val span: Span
    ) : Item()

    @Serializable
    data class Const(
        val name: String,
        val ty: Type,
        val value: Expr,
        val visibility: Visibility,
        override val span: Span
    ) : Item()
}

@Serializable
data class GenericParam(
    val name: String,
    val bounds: List<TypeBound>,
    val default: Type?
)

@Serializable
data class TypeBound(
    val path: TypePath
)

@Serializable
data class Param(
    val pattern: Pattern,
    val ty: Type,
    val evidence: EvidenceLevel?,
    val default: Expr?
)

@Serializable
data class StructField(
    val name: String,
    val ty: Type,
    val evidence: EvidenceLevel?,
    val visibility: Visibility
)

@Serializable
data class EnumVariant(
    val name: String,
    val fields: VariantFields?
)

@Serializable
sealed class VariantFields {
    @Serializable
    data class Tuple(val types: List<Type>) : VariantFields()

    @Serializable
    data class Struct(val fields: List<StructField>) : VariantFields()
}

@Serializable
sealed class TraitItem {
    @Serializable
    data class Method(
        val name: String,
        val generics: List<GenericParam>,
        val params: List<Param>,
        val returnType: Type?,
        val returnEvidence: EvidenceLevel?,
        val body: Block?
    ) : TraitItem()

    @Serializable
    data class Type(
        val name: String,
        val bounds: List<TypeBound>
    ) : TraitItem()
}

@Serializable
sealed class ImplItem {
    @Serializable
    data class Method(
        val name: String,
        val generics: List<GenericParam>,
        val params: List<Param>,
        val returnType: Type?,
        val returnEvidence: EvidenceLevel?,
        val body: Block
    ) : ImplItem()

    @Serializable
    data class Type(
        val name: String,
        val ty: com.daemoniorum.sigil.ast.Type
    ) : ImplItem()
}

@Serializable
data class UsePath(
    val segments: List<String>,
    val alias: String?,
    val glob: Boolean
)

@Serializable
enum class Visibility {
    Private,
    Public,
    Crate
}

/**
 * Types
 */
@Serializable
sealed class Type {
    @Serializable
    data class Path(val path: TypePath) : Type()

    @Serializable
    data class Array(val element: Type, val size: Expr?) : Type()

    @Serializable
    data class Tuple(val elements: List<Type>) : Type()

    @Serializable
    data class Function(
        val params: List<Type>,
        val returnType: Type
    ) : Type()

    @Serializable
    data class Reference(
        val ty: Type,
        val mutable: Boolean
    ) : Type()

    @Serializable
    data class Option(val ty: Type) : Type()

    @Serializable
    data class Result(val ok: Type, val err: Type) : Type()

    @Serializable
    data class Infer(val id: Int) : Type()

    @Serializable
    object Unit : Type()

    @Serializable
    object Never : Type()
}

@Serializable
data class TypePath(
    val segments: List<TypePathSegment>
)

@Serializable
data class TypePathSegment(
    val name: String,
    val generics: List<Type>
)

/**
 * Expressions
 */
@Serializable
sealed class Expr {
    abstract val span: Span

    @Serializable
    data class Literal(
        val kind: LiteralKind,
        val evidence: EvidenceLevel,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Path(
        val path: ExprPath,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Binary(
        val op: BinaryOp,
        val left: Expr,
        val right: Expr,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Unary(
        val op: UnaryOp,
        val operand: Expr,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Call(
        val callee: Expr,
        val args: List<Expr>,
        override val span: Span
    ) : Expr()

    @Serializable
    data class MethodCall(
        val receiver: Expr,
        val method: String,
        val args: List<Expr>,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Field(
        val receiver: Expr,
        val field: String,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Index(
        val receiver: Expr,
        val index: Expr,
        override val span: Span
    ) : Expr()

    @Serializable
    data class If(
        val condition: Expr,
        val thenBranch: Block,
        val elseBranch: Block?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Match(
        val scrutinee: Expr,
        val arms: List<MatchArm>,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Block(
        val block: com.daemoniorum.sigil.ast.Block,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Loop(
        val body: com.daemoniorum.sigil.ast.Block,
        val label: String?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class While(
        val condition: Expr,
        val body: com.daemoniorum.sigil.ast.Block,
        val label: String?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class For(
        val pattern: Pattern,
        val iterable: Expr,
        val body: com.daemoniorum.sigil.ast.Block,
        val label: String?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Return(
        val value: Expr?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Break(
        val label: String?,
        val value: Expr?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Continue(
        val label: String?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Lambda(
        val params: List<LambdaParam>,
        val body: Expr,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Struct(
        val path: TypePath,
        val fields: List<FieldInit>,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Array(
        val elements: List<Expr>,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Tuple(
        val elements: List<Expr>,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Range(
        val start: Expr?,
        val end: Expr?,
        val inclusive: Boolean,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Morpheme(
        val op: MorphemeOp,
        val input: Expr,
        val transform: Expr?,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Pipe(
        val left: Expr,
        val right: Expr,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Await(
        val expr: Expr,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Try(
        val expr: Expr,
        override val span: Span
    ) : Expr()

    @Serializable
    data class Cast(
        val expr: Expr,
        val ty: Type,
        override val span: Span
    ) : Expr()
}

@Serializable
sealed class LiteralKind {
    @Serializable data class Int(val value: Long) : LiteralKind()
    @Serializable data class Float(val value: Double) : LiteralKind()
    @Serializable data class String(val value: kotlin.String) : LiteralKind()
    @Serializable data class Char(val value: kotlin.Char) : LiteralKind()
    @Serializable data class Bool(val value: Boolean) : LiteralKind()
    @Serializable object Unit : LiteralKind()
}

@Serializable
data class ExprPath(
    val segments: List<String>
)

@Serializable
enum class BinaryOp {
    Add, Sub, Mul, Div, Rem, Pow,
    And, Or, BitAnd, BitOr, BitXor,
    Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
    Assign, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign
}

@Serializable
enum class UnaryOp {
    Neg, Not, Deref, Ref, RefMut
}

@Serializable
enum class MorphemeOp {
    Tau,      // τ - transform/map
    Phi,      // φ - filter
    Sigma,    // σ - sort
    Rho,      // ρ - reduce
    Sum,      // Σ - sum
    Product,  // Π - product
    Alpha,    // α - first
    Omega,    // ω - last
    Lambda,   // λ - lambda
    Delta,    // Δ - difference
    Gamma     // Γ - collect
}

@Serializable
data class MatchArm(
    val pattern: Pattern,
    val guard: Expr?,
    val body: Expr
)

@Serializable
data class LambdaParam(
    val pattern: Pattern,
    val ty: Type?
)

@Serializable
data class FieldInit(
    val name: String,
    val value: Expr
)

/**
 * Patterns
 */
@Serializable
sealed class Pattern {
    @Serializable
    data class Ident(
        val name: String,
        val mutable: Boolean,
        val evidence: EvidenceLevel?
    ) : Pattern()

    @Serializable
    data class Literal(val kind: LiteralKind) : Pattern()

    @Serializable
    data class Tuple(val elements: List<Pattern>) : Pattern()

    @Serializable
    data class Struct(
        val path: TypePath,
        val fields: List<FieldPattern>
    ) : Pattern()

    @Serializable
    data class TupleStruct(
        val path: TypePath,
        val elements: List<Pattern>
    ) : Pattern()

    @Serializable
    data class Or(val patterns: List<Pattern>) : Pattern()

    @Serializable
    data class Range(
        val start: LiteralKind?,
        val end: LiteralKind?,
        val inclusive: Boolean
    ) : Pattern()

    @Serializable
    object Wildcard : Pattern()

    @Serializable
    data class Rest(val name: String?) : Pattern()
}

@Serializable
data class FieldPattern(
    val name: String,
    val pattern: Pattern?
)

/**
 * Statements
 */
@Serializable
sealed class Stmt {
    abstract val span: Span

    @Serializable
    data class Let(
        val pattern: Pattern,
        val ty: Type?,
        val init: Expr?,
        override val span: Span
    ) : Stmt()

    @Serializable
    data class Expr(
        val expr: com.daemoniorum.sigil.ast.Expr,
        val hasSemi: Boolean,
        override val span: Span
    ) : Stmt()

    @Serializable
    data class Item(
        val item: com.daemoniorum.sigil.ast.Item,
        override val span: Span
    ) : Stmt()
}

/**
 * Block - a sequence of statements
 */
@Serializable
data class Block(
    val stmts: List<Stmt>,
    val span: Span
)

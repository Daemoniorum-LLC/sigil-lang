package com.daemoniorum.sigil.lexer

import kotlinx.serialization.Serializable

/**
 * Source location for error reporting
 */
@Serializable
data class Span(
    val start: Int,
    val end: Int,
    val line: Int,
    val column: Int
) {
    companion object {
        val EMPTY = Span(0, 0, 0, 0)
    }
}

/**
 * Token with location information
 */
@Serializable
data class Token(
    val kind: TokenKind,
    val span: Span,
    val lexeme: String
) {
    override fun toString(): String = "$kind('$lexeme') at $line:$column"

    val line: Int get() = span.line
    val column: Int get() = span.column
}

/**
 * All token types in Sigil
 */
@Serializable
sealed class TokenKind {
    // Literals
    @Serializable data class IntLiteral(val value: Long) : TokenKind()
    @Serializable data class FloatLiteral(val value: Double) : TokenKind()
    @Serializable data class StringLiteral(val value: String) : TokenKind()
    @Serializable data class CharLiteral(val value: Char) : TokenKind()
    @Serializable object TrueLiteral : TokenKind()
    @Serializable object FalseLiteral : TokenKind()

    // Identifiers
    @Serializable data class Ident(val name: String) : TokenKind()

    // Keywords
    @Serializable object Fn : TokenKind()
    @Serializable object Let : TokenKind()
    @Serializable object Mut : TokenKind()
    @Serializable object If : TokenKind()
    @Serializable object Else : TokenKind()
    @Serializable object Match : TokenKind()
    @Serializable object Return : TokenKind()
    @Serializable object For : TokenKind()
    @Serializable object While : TokenKind()
    @Serializable object In : TokenKind()
    @Serializable object Struct : TokenKind()
    @Serializable object Enum : TokenKind()
    @Serializable object Trait : TokenKind()
    @Serializable object Impl : TokenKind()
    @Serializable object Use : TokenKind()
    @Serializable object Pub : TokenKind()
    @Serializable object Async : TokenKind()
    @Serializable object Await : TokenKind()
    @Serializable object Break : TokenKind()
    @Serializable object Continue : TokenKind()
    @Serializable object Loop : TokenKind()
    @Serializable object As : TokenKind()
    @Serializable object Type : TokenKind()
    @Serializable object Where : TokenKind()
    @Serializable object Self : TokenKind()
    @Serializable object Super : TokenKind()

    // Evidence markers
    @Serializable object EvidenceKnown : TokenKind()      // !
    @Serializable object EvidenceUncertain : TokenKind()  // ?
    @Serializable object EvidenceReported : TokenKind()   // ~
    @Serializable object EvidenceParadox : TokenKind()    // ‽

    // Morpheme operators
    @Serializable object MorphemeTau : TokenKind()        // τ (transform/map)
    @Serializable object MorphemePhi : TokenKind()        // φ (filter)
    @Serializable object MorphemeSigma : TokenKind()      // σ (sort)
    @Serializable object MorphemeRho : TokenKind()        // ρ (reduce)
    @Serializable object MorphemeSigmaSum : TokenKind()   // Σ (sum)
    @Serializable object MorphemePi : TokenKind()         // Π (product)
    @Serializable object MorphemeAlpha : TokenKind()      // α (first)
    @Serializable object MorphemeOmega : TokenKind()      // ω (last)
    @Serializable object MorphemeLambda : TokenKind()     // λ (lambda)
    @Serializable object MorphemeDelta : TokenKind()      // Δ (difference)
    @Serializable object MorphemeGamma : TokenKind()      // Γ (collect)

    // Operators
    @Serializable object Plus : TokenKind()
    @Serializable object Minus : TokenKind()
    @Serializable object Star : TokenKind()
    @Serializable object Slash : TokenKind()
    @Serializable object Percent : TokenKind()
    @Serializable object Caret : TokenKind()
    @Serializable object And : TokenKind()
    @Serializable object Or : TokenKind()
    @Serializable object Eq : TokenKind()
    @Serializable object EqEq : TokenKind()
    @Serializable object NotEq : TokenKind()
    @Serializable object Lt : TokenKind()
    @Serializable object LtEq : TokenKind()
    @Serializable object Gt : TokenKind()
    @Serializable object GtEq : TokenKind()
    @Serializable object AndAnd : TokenKind()
    @Serializable object OrOr : TokenKind()
    @Serializable object Not : TokenKind()
    @Serializable object Arrow : TokenKind()              // ->
    @Serializable object FatArrow : TokenKind()           // =>
    @Serializable object Pipe : TokenKind()               // |
    @Serializable object PipeGt : TokenKind()             // |>
    @Serializable object Dot : TokenKind()
    @Serializable object DotDot : TokenKind()
    @Serializable object DotDotEq : TokenKind()
    @Serializable object Colon : TokenKind()
    @Serializable object ColonColon : TokenKind()
    @Serializable object Semicolon : TokenKind()
    @Serializable object Comma : TokenKind()
    @Serializable object At : TokenKind()
    @Serializable object Hash : TokenKind()
    @Serializable object Dollar : TokenKind()

    // Delimiters
    @Serializable object LParen : TokenKind()
    @Serializable object RParen : TokenKind()
    @Serializable object LBrace : TokenKind()
    @Serializable object RBrace : TokenKind()
    @Serializable object LBracket : TokenKind()
    @Serializable object RBracket : TokenKind()

    // Special
    @Serializable object Eof : TokenKind()
    @Serializable data class Error(val message: String) : TokenKind()
    @Serializable object Newline : TokenKind()
    @Serializable data class Comment(val text: String) : TokenKind()
    @Serializable data class DocComment(val text: String) : TokenKind()

    companion object {
        val KEYWORDS = mapOf(
            "fn" to Fn,
            "let" to Let,
            "mut" to Mut,
            "if" to If,
            "else" to Else,
            "match" to Match,
            "return" to Return,
            "for" to For,
            "while" to While,
            "in" to In,
            "struct" to Struct,
            "enum" to Enum,
            "trait" to Trait,
            "impl" to Impl,
            "use" to Use,
            "pub" to Pub,
            "async" to Async,
            "await" to Await,
            "break" to Break,
            "continue" to Continue,
            "loop" to Loop,
            "as" to As,
            "type" to Type,
            "where" to Where,
            "self" to Self,
            "Self" to Self,
            "super" to Super,
            "true" to TrueLiteral,
            "false" to FalseLiteral,
        )

        val MORPHEMES = mapOf(
            'τ' to MorphemeTau,
            'φ' to MorphemePhi,
            'σ' to MorphemeSigma,
            'ρ' to MorphemeRho,
            'Σ' to MorphemeSigmaSum,
            'Π' to MorphemePi,
            'α' to MorphemeAlpha,
            'ω' to MorphemeOmega,
            'λ' to MorphemeLambda,
            'Δ' to MorphemeDelta,
            'Γ' to MorphemeGamma,
        )
    }
}

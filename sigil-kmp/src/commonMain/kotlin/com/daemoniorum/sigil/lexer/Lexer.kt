package com.daemoniorum.sigil.lexer

/**
 * Lexer for the Sigil programming language.
 *
 * Converts source code into a stream of tokens, handling:
 * - Unicode identifiers and morpheme operators (τ, φ, σ, etc.)
 * - Evidence markers (!, ?, ~, ‽)
 * - String interpolation
 * - Comments and documentation
 */
class Lexer(private val source: String) {
    private var pos: Int = 0
    private var line: Int = 1
    private var column: Int = 1
    private var tokenStart: Int = 0
    private var tokenLine: Int = 1
    private var tokenColumn: Int = 1

    private val tokens = mutableListOf<Token>()
    private val errors = mutableListOf<LexerError>()

    data class LexerError(
        val message: String,
        val span: Span
    )

    /**
     * Tokenize the entire source
     */
    fun tokenize(): LexerResult {
        while (!isAtEnd()) {
            skipWhitespaceAndComments()
            if (!isAtEnd()) {
                scanToken()
            }
        }
        addToken(TokenKind.Eof)
        return LexerResult(tokens, errors)
    }

    private fun scanToken() {
        markTokenStart()
        val c = advance()

        val kind: TokenKind = when (c) {
            // Single-character tokens
            '(' -> TokenKind.LParen
            ')' -> TokenKind.RParen
            '{' -> TokenKind.LBrace
            '}' -> TokenKind.RBrace
            '[' -> TokenKind.LBracket
            ']' -> TokenKind.RBracket
            ',' -> TokenKind.Comma
            ';' -> TokenKind.Semicolon
            '@' -> TokenKind.At
            '#' -> TokenKind.Hash
            '$' -> TokenKind.Dollar
            '+' -> TokenKind.Plus
            '*' -> TokenKind.Star
            '%' -> TokenKind.Percent
            '^' -> TokenKind.Caret

            // Evidence markers
            '!' -> if (match('=')) TokenKind.NotEq else TokenKind.EvidenceKnown
            '?' -> TokenKind.EvidenceUncertain
            '~' -> TokenKind.EvidenceReported
            '‽' -> TokenKind.EvidenceParadox

            // Multi-character operators
            '-' -> if (match('>')) TokenKind.Arrow else TokenKind.Minus
            '=' -> when {
                match('=') -> TokenKind.EqEq
                match('>') -> TokenKind.FatArrow
                else -> TokenKind.Eq
            }
            '<' -> if (match('=')) TokenKind.LtEq else TokenKind.Lt
            '>' -> if (match('=')) TokenKind.GtEq else TokenKind.Gt
            '&' -> if (match('&')) TokenKind.AndAnd else TokenKind.And
            '|' -> when {
                match('|') -> TokenKind.OrOr
                match('>') -> TokenKind.PipeGt
                else -> TokenKind.Pipe
            }
            ':' -> if (match(':')) TokenKind.ColonColon else TokenKind.Colon
            '.' -> when {
                match('.') -> if (match('=')) TokenKind.DotDotEq else TokenKind.DotDot
                else -> TokenKind.Dot
            }

            // Division or comment
            '/' -> when {
                match('/') -> {
                    scanLineComment()
                    return
                }
                match('*') -> {
                    scanBlockComment()
                    return
                }
                else -> TokenKind.Slash
            }

            // Strings
            '"' -> scanString()
            '\'' -> scanChar()

            // Numbers
            in '0'..'9' -> scanNumber()

            // Morpheme operators (Greek letters)
            'τ', 'Τ' -> TokenKind.MorphemeTau
            'φ', 'Φ' -> TokenKind.MorphemePhi
            'σ' -> TokenKind.MorphemeSigma
            'Σ' -> TokenKind.MorphemeSigmaSum
            'ρ', 'Ρ' -> TokenKind.MorphemeRho
            'Π' -> TokenKind.MorphemePi
            'α', 'Α' -> TokenKind.MorphemeAlpha
            'ω', 'Ω' -> TokenKind.MorphemeOmega
            'λ', 'Λ' -> TokenKind.MorphemeLambda
            'Δ' -> TokenKind.MorphemeDelta
            'Γ' -> TokenKind.MorphemeGamma

            // Identifiers and keywords
            else -> when {
                isIdentStart(c) -> scanIdentifier()
                else -> {
                    addError("Unexpected character: '$c'")
                    TokenKind.Error("Unexpected character: '$c'")
                }
            }
        }

        addToken(kind)
    }

    private fun scanIdentifier(): TokenKind {
        while (!isAtEnd() && isIdentContinue(peek())) {
            advance()
        }
        val text = currentLexeme()
        return TokenKind.KEYWORDS[text] ?: TokenKind.Ident(text)
    }

    private fun scanNumber(): TokenKind {
        // Check for hex, binary, octal
        if (peekPrev() == '0' && !isAtEnd()) {
            when (peek()) {
                'x', 'X' -> return scanHexNumber()
                'b', 'B' -> return scanBinaryNumber()
                'o', 'O' -> return scanOctalNumber()
            }
        }

        // Decimal integer
        while (!isAtEnd() && peek().isDigit()) {
            advance()
        }

        // Check for float
        if (!isAtEnd() && peek() == '.' && peekNext()?.isDigit() == true) {
            advance() // consume '.'
            while (!isAtEnd() && peek().isDigit()) {
                advance()
            }

            // Scientific notation
            if (!isAtEnd() && (peek() == 'e' || peek() == 'E')) {
                advance()
                if (!isAtEnd() && (peek() == '+' || peek() == '-')) {
                    advance()
                }
                while (!isAtEnd() && peek().isDigit()) {
                    advance()
                }
            }

            val value = currentLexeme().toDoubleOrNull() ?: 0.0
            return TokenKind.FloatLiteral(value)
        }

        val value = currentLexeme().toLongOrNull() ?: 0L
        return TokenKind.IntLiteral(value)
    }

    private fun scanHexNumber(): TokenKind {
        advance() // consume 'x'
        while (!isAtEnd() && (peek().isDigit() || peek() in 'a'..'f' || peek() in 'A'..'F')) {
            advance()
        }
        val hex = currentLexeme().drop(2)
        val value = hex.toLongOrNull(16) ?: 0L
        return TokenKind.IntLiteral(value)
    }

    private fun scanBinaryNumber(): TokenKind {
        advance() // consume 'b'
        while (!isAtEnd() && (peek() == '0' || peek() == '1')) {
            advance()
        }
        val binary = currentLexeme().drop(2)
        val value = binary.toLongOrNull(2) ?: 0L
        return TokenKind.IntLiteral(value)
    }

    private fun scanOctalNumber(): TokenKind {
        advance() // consume 'o'
        while (!isAtEnd() && peek() in '0'..'7') {
            advance()
        }
        val octal = currentLexeme().drop(2)
        val value = octal.toLongOrNull(8) ?: 0L
        return TokenKind.IntLiteral(value)
    }

    private fun scanString(): TokenKind {
        val builder = StringBuilder()

        while (!isAtEnd() && peek() != '"') {
            if (peek() == '\n') {
                line++
                column = 1
            }
            if (peek() == '\\') {
                advance()
                if (!isAtEnd()) {
                    builder.append(
                        when (val escaped = advance()) {
                            'n' -> '\n'
                            'r' -> '\r'
                            't' -> '\t'
                            '\\' -> '\\'
                            '"' -> '"'
                            '0' -> '\u0000'
                            'u' -> scanUnicodeEscape()
                            else -> {
                                addError("Unknown escape sequence: \\$escaped")
                                escaped
                            }
                        }
                    )
                }
            } else {
                builder.append(advance())
            }
        }

        if (isAtEnd()) {
            addError("Unterminated string")
            return TokenKind.Error("Unterminated string")
        }

        advance() // closing "
        return TokenKind.StringLiteral(builder.toString())
    }

    private fun scanChar(): TokenKind {
        val c = if (peek() == '\\') {
            advance()
            when (val escaped = advance()) {
                'n' -> '\n'
                'r' -> '\r'
                't' -> '\t'
                '\\' -> '\\'
                '\'' -> '\''
                '0' -> '\u0000'
                'u' -> scanUnicodeEscape()
                else -> {
                    addError("Unknown escape sequence: \\$escaped")
                    escaped
                }
            }
        } else {
            advance()
        }

        if (isAtEnd() || peek() != '\'') {
            addError("Unterminated character literal")
            return TokenKind.Error("Unterminated character literal")
        }

        advance() // closing '
        return TokenKind.CharLiteral(c)
    }

    private fun scanUnicodeEscape(): Char {
        if (!match('{')) {
            addError("Expected '{' after \\u")
            return '?'
        }
        val hex = StringBuilder()
        while (!isAtEnd() && peek() != '}') {
            hex.append(advance())
        }
        if (!match('}')) {
            addError("Unterminated unicode escape")
            return '?'
        }
        return hex.toString().toIntOrNull(16)?.toChar() ?: '?'
    }

    private fun scanLineComment() {
        // Check for doc comment ///
        val isDoc = peek() == '/'
        if (isDoc) advance()

        val start = pos
        while (!isAtEnd() && peek() != '\n') {
            advance()
        }
        val text = source.substring(start, pos).trim()

        if (isDoc) {
            addToken(TokenKind.DocComment(text))
        }
        // Regular comments are skipped
    }

    private fun scanBlockComment() {
        var depth = 1
        while (!isAtEnd() && depth > 0) {
            if (peek() == '/' && peekNext() == '*') {
                advance()
                advance()
                depth++
            } else if (peek() == '*' && peekNext() == '/') {
                advance()
                advance()
                depth--
            } else {
                if (peek() == '\n') {
                    line++
                    column = 1
                }
                advance()
            }
        }
    }

    private fun skipWhitespaceAndComments() {
        while (!isAtEnd()) {
            when (peek()) {
                ' ', '\r', '\t' -> advance()
                '\n' -> {
                    line++
                    column = 1
                    advance()
                }
                else -> break
            }
        }
    }

    // Helpers

    private fun isAtEnd(): Boolean = pos >= source.length

    private fun peek(): Char = if (isAtEnd()) '\u0000' else source[pos]

    private fun peekNext(): Char? = if (pos + 1 >= source.length) null else source[pos + 1]

    private fun peekPrev(): Char = if (pos == 0) '\u0000' else source[pos - 1]

    private fun advance(): Char {
        val c = source[pos++]
        column++
        return c
    }

    private fun match(expected: Char): Boolean {
        if (isAtEnd() || source[pos] != expected) return false
        pos++
        column++
        return true
    }

    private fun markTokenStart() {
        tokenStart = pos
        tokenLine = line
        tokenColumn = column
    }

    private fun currentLexeme(): String = source.substring(tokenStart, pos)

    private fun currentSpan(): Span = Span(tokenStart, pos, tokenLine, tokenColumn)

    private fun addToken(kind: TokenKind) {
        tokens.add(Token(kind, currentSpan(), currentLexeme()))
    }

    private fun addError(message: String) {
        errors.add(LexerError(message, currentSpan()))
    }

    private fun isIdentStart(c: Char): Boolean =
        c.isLetter() || c == '_'

    private fun isIdentContinue(c: Char): Boolean =
        c.isLetterOrDigit() || c == '_'
}

/**
 * Result of lexing
 */
data class LexerResult(
    val tokens: List<Token>,
    val errors: List<Lexer.LexerError>
) {
    val hasErrors: Boolean get() = errors.isNotEmpty()
}

package com.daemoniorum.sigil.parser

import com.daemoniorum.sigil.ast.*
import com.daemoniorum.sigil.lexer.*
import com.daemoniorum.sigil.runtime.EvidenceLevel

/**
 * Recursive descent parser for Sigil.
 *
 * Produces a typed AST from a token stream.
 */
class Parser(private val tokens: List<Token>) {
    private var current = 0
    private val errors = mutableListOf<ParseError>()

    data class ParseError(
        val message: String,
        val span: Span
    )

    /**
     * Parse a complete module
     */
    fun parse(): ParseResult {
        val items = mutableListOf<Item>()

        while (!isAtEnd()) {
            try {
                skipNewlines()
                if (!isAtEnd()) {
                    items.add(parseItem())
                }
            } catch (e: ParseException) {
                errors.add(ParseError(e.message ?: "Parse error", e.span))
                synchronize()
            }
        }

        val span = if (items.isNotEmpty()) {
            Span(items.first().span.start, items.last().span.end, 1, 1)
        } else {
            Span.EMPTY
        }

        return ParseResult(Module(items, span), errors)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Items
    // ═══════════════════════════════════════════════════════════════════════

    private fun parseItem(): Item {
        val visibility = parseVisibility()

        return when {
            check(TokenKind.Fn) -> parseFunction(visibility)
            check(TokenKind.Struct) -> parseStruct(visibility)
            check(TokenKind.Enum) -> parseEnum(visibility)
            check(TokenKind.Trait) -> parseTrait(visibility)
            check(TokenKind.Impl) -> parseImpl()
            check(TokenKind.Type) -> parseTypeAlias(visibility)
            check(TokenKind.Use) -> parseUse(visibility)
            check(TokenKind.Let) || check(TokenKind.Const) -> parseConst(visibility)
            else -> throw error("Expected item declaration")
        }
    }

    private fun parseVisibility(): Visibility {
        return if (match(TokenKind.Pub)) {
            Visibility.Public
        } else {
            Visibility.Private
        }
    }

    private fun parseFunction(visibility: Visibility): Item.Function {
        val start = previous().span
        val isAsync = match(TokenKind.Async)
        expect(TokenKind.Fn, "Expected 'fn'")
        val name = expectIdent("Expected function name")
        val generics = parseGenerics()

        expect(TokenKind.LParen, "Expected '('")
        val params = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseParam() }
        expect(TokenKind.RParen, "Expected ')'")

        var returnType: Type? = null
        var returnEvidence: EvidenceLevel? = null

        if (match(TokenKind.Arrow)) {
            returnType = parseType()
            returnEvidence = parseEvidenceMarker()
        }

        val body = parseBlock()

        return Item.Function(
            name = name,
            generics = generics,
            params = params,
            returnType = returnType,
            returnEvidence = returnEvidence,
            body = body,
            isAsync = isAsync,
            visibility = visibility,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    private fun parseParam(): Param {
        val pattern = parsePattern()

        expect(TokenKind.Colon, "Expected ':' after parameter name")
        val ty = parseType()
        val evidence = parseEvidenceMarker()

        val default = if (match(TokenKind.Eq)) parseExpr() else null

        return Param(pattern, ty, evidence, default)
    }

    private fun parseStruct(visibility: Visibility): Item.Struct {
        val start = expect(TokenKind.Struct, "Expected 'struct'").span
        val name = expectIdent("Expected struct name")
        val generics = parseGenerics()

        expect(TokenKind.LBrace, "Expected '{'")
        val fields = parseDelimited(TokenKind.Comma, TokenKind.RBrace) { parseStructField() }
        expect(TokenKind.RBrace, "Expected '}'")

        return Item.Struct(
            name = name,
            generics = generics,
            fields = fields,
            visibility = visibility,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    private fun parseStructField(): StructField {
        val fieldVisibility = parseVisibility()
        val name = expectIdent("Expected field name")
        val evidence = parseEvidenceMarker()
        expect(TokenKind.Colon, "Expected ':'")
        val ty = parseType()

        return StructField(name, ty, evidence, fieldVisibility)
    }

    private fun parseEnum(visibility: Visibility): Item.Enum {
        val start = expect(TokenKind.Enum, "Expected 'enum'").span
        val name = expectIdent("Expected enum name")
        val generics = parseGenerics()

        expect(TokenKind.LBrace, "Expected '{'")
        val variants = parseDelimited(TokenKind.Comma, TokenKind.RBrace) { parseEnumVariant() }
        expect(TokenKind.RBrace, "Expected '}'")

        return Item.Enum(
            name = name,
            generics = generics,
            variants = variants,
            visibility = visibility,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    private fun parseEnumVariant(): EnumVariant {
        val name = expectIdent("Expected variant name")

        val fields: VariantFields? = when {
            match(TokenKind.LParen) -> {
                val types = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseType() }
                expect(TokenKind.RParen, "Expected ')'")
                VariantFields.Tuple(types)
            }
            match(TokenKind.LBrace) -> {
                val structFields = parseDelimited(TokenKind.Comma, TokenKind.RBrace) { parseStructField() }
                expect(TokenKind.RBrace, "Expected '}'")
                VariantFields.Struct(structFields)
            }
            else -> null
        }

        return EnumVariant(name, fields)
    }

    private fun parseTrait(visibility: Visibility): Item.Trait {
        val start = expect(TokenKind.Trait, "Expected 'trait'").span
        val name = expectIdent("Expected trait name")
        val generics = parseGenerics()
        val bounds = if (match(TokenKind.Colon)) parseTypeBounds() else emptyList()

        expect(TokenKind.LBrace, "Expected '{'")
        val items = mutableListOf<TraitItem>()
        while (!check(TokenKind.RBrace) && !isAtEnd()) {
            skipNewlines()
            if (!check(TokenKind.RBrace)) {
                items.add(parseTraitItem())
            }
        }
        expect(TokenKind.RBrace, "Expected '}'")

        return Item.Trait(
            name = name,
            generics = generics,
            bounds = bounds,
            items = items,
            visibility = visibility,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    private fun parseTraitItem(): TraitItem {
        return when {
            check(TokenKind.Fn) -> {
                advance()
                val name = expectIdent("Expected method name")
                val generics = parseGenerics()
                expect(TokenKind.LParen, "Expected '('")
                val params = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseParam() }
                expect(TokenKind.RParen, "Expected ')'")

                var returnType: Type? = null
                var returnEvidence: EvidenceLevel? = null
                if (match(TokenKind.Arrow)) {
                    returnType = parseType()
                    returnEvidence = parseEvidenceMarker()
                }

                val body = if (check(TokenKind.LBrace)) parseBlock() else {
                    expect(TokenKind.Semicolon, "Expected ';' or block")
                    null
                }

                TraitItem.Method(name, generics, params, returnType, returnEvidence, body)
            }
            check(TokenKind.Type) -> {
                advance()
                val name = expectIdent("Expected type name")
                val bounds = if (match(TokenKind.Colon)) parseTypeBounds() else emptyList()
                expect(TokenKind.Semicolon, "Expected ';'")
                TraitItem.Type(name, bounds)
            }
            else -> throw error("Expected trait item")
        }
    }

    private fun parseImpl(): Item.Impl {
        val start = expect(TokenKind.Impl, "Expected 'impl'").span
        val generics = parseGenerics()

        val firstType = parseType()

        val (traitRef, selfType) = if (match(TokenKind.For)) {
            val trait = (firstType as? Type.Path)?.path
            val self = parseType()
            trait to self
        } else {
            null to firstType
        }

        expect(TokenKind.LBrace, "Expected '{'")
        val items = mutableListOf<ImplItem>()
        while (!check(TokenKind.RBrace) && !isAtEnd()) {
            skipNewlines()
            if (!check(TokenKind.RBrace)) {
                items.add(parseImplItem())
            }
        }
        expect(TokenKind.RBrace, "Expected '}'")

        return Item.Impl(
            generics = generics,
            traitRef = traitRef,
            selfType = selfType,
            items = items,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    private fun parseImplItem(): ImplItem {
        parseVisibility() // Consume but ignore for now

        return when {
            check(TokenKind.Fn) -> {
                advance()
                val name = expectIdent("Expected method name")
                val generics = parseGenerics()
                expect(TokenKind.LParen, "Expected '('")
                val params = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseParam() }
                expect(TokenKind.RParen, "Expected ')'")

                var returnType: Type? = null
                var returnEvidence: EvidenceLevel? = null
                if (match(TokenKind.Arrow)) {
                    returnType = parseType()
                    returnEvidence = parseEvidenceMarker()
                }

                val body = parseBlock()
                ImplItem.Method(name, generics, params, returnType, returnEvidence, body)
            }
            check(TokenKind.Type) -> {
                advance()
                val name = expectIdent("Expected type name")
                expect(TokenKind.Eq, "Expected '='")
                val ty = parseType()
                expect(TokenKind.Semicolon, "Expected ';'")
                ImplItem.Type(name, ty)
            }
            else -> throw error("Expected impl item")
        }
    }

    private fun parseTypeAlias(visibility: Visibility): Item.TypeAlias {
        val start = expect(TokenKind.Type, "Expected 'type'").span
        val name = expectIdent("Expected type name")
        val generics = parseGenerics()
        expect(TokenKind.Eq, "Expected '='")
        val ty = parseType()
        expect(TokenKind.Semicolon, "Expected ';'")

        return Item.TypeAlias(
            name = name,
            generics = generics,
            ty = ty,
            visibility = visibility,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    private fun parseUse(visibility: Visibility): Item.Use {
        val start = expect(TokenKind.Use, "Expected 'use'").span
        val path = parseUsePath()
        expect(TokenKind.Semicolon, "Expected ';'")

        return Item.Use(
            path = path,
            visibility = visibility,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    private fun parseUsePath(): UsePath {
        val segments = mutableListOf<String>()
        segments.add(expectIdent("Expected path segment"))

        while (match(TokenKind.ColonColon)) {
            if (match(TokenKind.Star)) {
                return UsePath(segments, null, glob = true)
            }
            segments.add(expectIdent("Expected path segment"))
        }

        val alias = if (match(TokenKind.As)) expectIdent("Expected alias") else null

        return UsePath(segments, alias, glob = false)
    }

    private fun parseConst(visibility: Visibility): Item.Const {
        val start = if (match(TokenKind.Const)) previous().span else expect(TokenKind.Let, "Expected 'const' or 'let'").span
        val name = expectIdent("Expected constant name")
        expect(TokenKind.Colon, "Expected ':'")
        val ty = parseType()
        expect(TokenKind.Eq, "Expected '='")
        val value = parseExpr()
        expect(TokenKind.Semicolon, "Expected ';'")

        return Item.Const(
            name = name,
            ty = ty,
            value = value,
            visibility = visibility,
            span = Span(start.start, previous().span.end, start.line, start.column)
        )
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Types
    // ═══════════════════════════════════════════════════════════════════════

    private fun parseType(): Type {
        return when {
            match(TokenKind.LParen) -> {
                if (check(TokenKind.RParen)) {
                    advance()
                    Type.Unit
                } else {
                    val types = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseType() }
                    expect(TokenKind.RParen, "Expected ')'")
                    if (types.size == 1) types[0] else Type.Tuple(types)
                }
            }
            match(TokenKind.LBracket) -> {
                val element = parseType()
                val size = if (match(TokenKind.Semicolon)) parseExpr() else null
                expect(TokenKind.RBracket, "Expected ']'")
                Type.Array(element, size)
            }
            match(TokenKind.And) -> {
                val mutable = match(TokenKind.Mut)
                Type.Reference(parseType(), mutable)
            }
            check(TokenKind.Fn) -> {
                advance()
                expect(TokenKind.LParen, "Expected '('")
                val params = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseType() }
                expect(TokenKind.RParen, "Expected ')'")
                expect(TokenKind.Arrow, "Expected '->'")
                val ret = parseType()
                Type.Function(params, ret)
            }
            else -> {
                val path = parseTypePath()
                Type.Path(path)
            }
        }
    }

    private fun parseTypePath(): TypePath {
        val segments = mutableListOf<TypePathSegment>()

        do {
            val name = expectIdent("Expected type name")
            val generics = if (match(TokenKind.Lt)) {
                val types = parseDelimited(TokenKind.Comma, TokenKind.Gt) { parseType() }
                expect(TokenKind.Gt, "Expected '>'")
                types
            } else {
                emptyList()
            }
            segments.add(TypePathSegment(name, generics))
        } while (match(TokenKind.ColonColon))

        return TypePath(segments)
    }

    private fun parseGenerics(): List<GenericParam> {
        if (!match(TokenKind.Lt)) return emptyList()

        val params = parseDelimited(TokenKind.Comma, TokenKind.Gt) {
            val name = expectIdent("Expected generic parameter name")
            val bounds = if (match(TokenKind.Colon)) parseTypeBounds() else emptyList()
            val default = if (match(TokenKind.Eq)) parseType() else null
            GenericParam(name, bounds, default)
        }

        expect(TokenKind.Gt, "Expected '>'")
        return params
    }

    private fun parseTypeBounds(): List<TypeBound> {
        val bounds = mutableListOf<TypeBound>()
        do {
            bounds.add(TypeBound(parseTypePath()))
        } while (match(TokenKind.Plus))
        return bounds
    }

    private fun parseEvidenceMarker(): EvidenceLevel? {
        return when {
            match(TokenKind.EvidenceKnown) -> EvidenceLevel.KNOWN
            match(TokenKind.EvidenceUncertain) -> EvidenceLevel.UNCERTAIN
            match(TokenKind.EvidenceReported) -> EvidenceLevel.REPORTED
            match(TokenKind.EvidenceParadox) -> EvidenceLevel.PARADOX
            else -> null
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Expressions (Pratt parser)
    // ═══════════════════════════════════════════════════════════════════════

    private fun parseExpr(): Expr = parseExprPrecedence(0)

    private fun parseExprPrecedence(minPrec: Int): Expr {
        var left = parseUnary()

        while (true) {
            val op = peekBinaryOp() ?: break
            val (prec, assoc) = precedence(op)

            if (prec < minPrec) break

            advance() // consume operator
            val nextPrec = if (assoc == Assoc.Left) prec + 1 else prec
            val right = parseExprPrecedence(nextPrec)

            left = Expr.Binary(
                op = op,
                left = left,
                right = right,
                span = Span(left.span.start, right.span.end, left.span.line, left.span.column)
            )
        }

        return left
    }

    private fun parseUnary(): Expr {
        val start = peek().span

        val op = when {
            match(TokenKind.Minus) -> UnaryOp.Neg
            match(TokenKind.Not) -> UnaryOp.Not
            match(TokenKind.And) -> if (match(TokenKind.Mut)) UnaryOp.RefMut else UnaryOp.Ref
            match(TokenKind.Star) -> UnaryOp.Deref
            else -> return parsePostfix()
        }

        val operand = parseUnary()
        return Expr.Unary(op, operand, Span(start.start, operand.span.end, start.line, start.column))
    }

    private fun parsePostfix(): Expr {
        var expr = parsePrimary()

        while (true) {
            expr = when {
                match(TokenKind.LParen) -> {
                    val args = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseExpr() }
                    expect(TokenKind.RParen, "Expected ')'")
                    Expr.Call(expr, args, Span(expr.span.start, previous().span.end, expr.span.line, expr.span.column))
                }
                match(TokenKind.Dot) -> {
                    val name = expectIdent("Expected field or method name")
                    if (match(TokenKind.LParen)) {
                        val args = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseExpr() }
                        expect(TokenKind.RParen, "Expected ')'")
                        Expr.MethodCall(expr, name, args, Span(expr.span.start, previous().span.end, expr.span.line, expr.span.column))
                    } else {
                        Expr.Field(expr, name, Span(expr.span.start, previous().span.end, expr.span.line, expr.span.column))
                    }
                }
                match(TokenKind.LBracket) -> {
                    val index = parseExpr()
                    expect(TokenKind.RBracket, "Expected ']'")
                    Expr.Index(expr, index, Span(expr.span.start, previous().span.end, expr.span.line, expr.span.column))
                }
                match(TokenKind.PipeGt) -> {
                    val right = parseUnary()
                    Expr.Pipe(expr, right, Span(expr.span.start, right.span.end, expr.span.line, expr.span.column))
                }
                match(TokenKind.As) -> {
                    val ty = parseType()
                    Expr.Cast(expr, ty, Span(expr.span.start, previous().span.end, expr.span.line, expr.span.column))
                }
                checkMorpheme() -> {
                    val op = parseMorphemeOp()
                    val transform = if (match(TokenKind.LParen)) {
                        val t = parseExpr()
                        expect(TokenKind.RParen, "Expected ')'")
                        t
                    } else null
                    Expr.Morpheme(op, expr, transform, Span(expr.span.start, previous().span.end, expr.span.line, expr.span.column))
                }
                else -> break
            }
        }

        return expr
    }

    private fun parsePrimary(): Expr {
        val start = peek().span

        return when {
            // Literals
            peek().kind is TokenKind.IntLiteral -> {
                val token = advance()
                val value = (token.kind as TokenKind.IntLiteral).value
                val evidence = parseEvidenceMarker() ?: EvidenceLevel.KNOWN
                Expr.Literal(LiteralKind.Int(value), evidence, token.span)
            }
            peek().kind is TokenKind.FloatLiteral -> {
                val token = advance()
                val value = (token.kind as TokenKind.FloatLiteral).value
                val evidence = parseEvidenceMarker() ?: EvidenceLevel.KNOWN
                Expr.Literal(LiteralKind.Float(value), evidence, token.span)
            }
            peek().kind is TokenKind.StringLiteral -> {
                val token = advance()
                val value = (token.kind as TokenKind.StringLiteral).value
                val evidence = parseEvidenceMarker() ?: EvidenceLevel.KNOWN
                Expr.Literal(LiteralKind.String(value), evidence, token.span)
            }
            peek().kind is TokenKind.CharLiteral -> {
                val token = advance()
                val value = (token.kind as TokenKind.CharLiteral).value
                val evidence = parseEvidenceMarker() ?: EvidenceLevel.KNOWN
                Expr.Literal(LiteralKind.Char(value), evidence, token.span)
            }
            match(TokenKind.TrueLiteral) -> {
                val evidence = parseEvidenceMarker() ?: EvidenceLevel.KNOWN
                Expr.Literal(LiteralKind.Bool(true), evidence, start)
            }
            match(TokenKind.FalseLiteral) -> {
                val evidence = parseEvidenceMarker() ?: EvidenceLevel.KNOWN
                Expr.Literal(LiteralKind.Bool(false), evidence, start)
            }

            // Grouping / Tuple
            match(TokenKind.LParen) -> {
                if (check(TokenKind.RParen)) {
                    advance()
                    Expr.Literal(LiteralKind.Unit, EvidenceLevel.KNOWN, Span(start.start, previous().span.end, start.line, start.column))
                } else {
                    val exprs = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parseExpr() }
                    expect(TokenKind.RParen, "Expected ')'")
                    if (exprs.size == 1) {
                        Expr.Block(Block(listOf(Stmt.Expr(exprs[0], false, exprs[0].span)), exprs[0].span), exprs[0].span)
                    } else {
                        Expr.Tuple(exprs, Span(start.start, previous().span.end, start.line, start.column))
                    }
                }
            }

            // Array
            match(TokenKind.LBracket) -> {
                val elements = parseDelimited(TokenKind.Comma, TokenKind.RBracket) { parseExpr() }
                expect(TokenKind.RBracket, "Expected ']'")
                Expr.Array(elements, Span(start.start, previous().span.end, start.line, start.column))
            }

            // Block
            check(TokenKind.LBrace) -> {
                val block = parseBlock()
                Expr.Block(block, block.span)
            }

            // Control flow
            match(TokenKind.If) -> parseIf(start)
            match(TokenKind.Match) -> parseMatch(start)
            match(TokenKind.Loop) -> {
                val body = parseBlock()
                Expr.Loop(body, null, Span(start.start, body.span.end, start.line, start.column))
            }
            match(TokenKind.While) -> {
                val cond = parseExpr()
                val body = parseBlock()
                Expr.While(cond, body, null, Span(start.start, body.span.end, start.line, start.column))
            }
            match(TokenKind.For) -> {
                val pattern = parsePattern()
                expect(TokenKind.In, "Expected 'in'")
                val iter = parseExpr()
                val body = parseBlock()
                Expr.For(pattern, iter, body, null, Span(start.start, body.span.end, start.line, start.column))
            }
            match(TokenKind.Return) -> {
                val value = if (!check(TokenKind.Semicolon) && !check(TokenKind.RBrace)) parseExpr() else null
                Expr.Return(value, Span(start.start, previous().span.end, start.line, start.column))
            }
            match(TokenKind.Break) -> {
                val value = if (!check(TokenKind.Semicolon) && !check(TokenKind.RBrace)) parseExpr() else null
                Expr.Break(null, value, Span(start.start, previous().span.end, start.line, start.column))
            }
            match(TokenKind.Continue) -> {
                Expr.Continue(null, Span(start.start, previous().span.end, start.line, start.column))
            }

            // Lambda
            match(TokenKind.Pipe) || match(TokenKind.OrOr) -> {
                val params = if (previous().kind == TokenKind.OrOr) {
                    emptyList()
                } else {
                    parseDelimited(TokenKind.Comma, TokenKind.Pipe) {
                        val pat = parsePattern()
                        val ty = if (match(TokenKind.Colon)) parseType() else null
                        LambdaParam(pat, ty)
                    }.also { expect(TokenKind.Pipe, "Expected '|'") }
                }
                val body = parseExpr()
                Expr.Lambda(params, body, Span(start.start, body.span.end, start.line, start.column))
            }

            // Identifier / Path
            peek().kind is TokenKind.Ident -> {
                val path = parseExprPath()
                if (check(TokenKind.LBrace) && !path.segments.any { it.contains("::") }) {
                    // Struct literal
                    advance()
                    val fields = parseDelimited(TokenKind.Comma, TokenKind.RBrace) {
                        val name = expectIdent("Expected field name")
                        val value = if (match(TokenKind.Colon)) parseExpr() else Expr.Path(ExprPath(listOf(name)), peek().span)
                        FieldInit(name, value)
                    }
                    expect(TokenKind.RBrace, "Expected '}'")
                    Expr.Struct(
                        TypePath(path.segments.map { TypePathSegment(it, emptyList()) }),
                        fields,
                        Span(start.start, previous().span.end, start.line, start.column)
                    )
                } else {
                    Expr.Path(path, Span(start.start, previous().span.end, start.line, start.column))
                }
            }

            else -> throw error("Expected expression")
        }
    }

    private fun parseIf(start: Span): Expr.If {
        val condition = parseExpr()
        val thenBranch = parseBlock()
        val elseBranch = if (match(TokenKind.Else)) {
            if (check(TokenKind.If)) {
                val elseIf = parseIf(peek().span)
                Block(listOf(Stmt.Expr(elseIf, false, elseIf.span)), elseIf.span)
            } else {
                parseBlock()
            }
        } else null

        return Expr.If(condition, thenBranch, elseBranch, Span(start.start, previous().span.end, start.line, start.column))
    }

    private fun parseMatch(start: Span): Expr.Match {
        val scrutinee = parseExpr()
        expect(TokenKind.LBrace, "Expected '{'")

        val arms = mutableListOf<MatchArm>()
        while (!check(TokenKind.RBrace) && !isAtEnd()) {
            skipNewlines()
            if (check(TokenKind.RBrace)) break

            val pattern = parsePattern()
            val guard = if (match(TokenKind.If)) parseExpr() else null
            expect(TokenKind.FatArrow, "Expected '=>'")
            val body = parseExpr()

            arms.add(MatchArm(pattern, guard, body))

            if (!check(TokenKind.RBrace)) {
                match(TokenKind.Comma)
            }
        }

        expect(TokenKind.RBrace, "Expected '}'")
        return Expr.Match(scrutinee, arms, Span(start.start, previous().span.end, start.line, start.column))
    }

    private fun parseExprPath(): ExprPath {
        val segments = mutableListOf<String>()
        segments.add(expectIdent("Expected identifier"))

        while (match(TokenKind.ColonColon)) {
            segments.add(expectIdent("Expected identifier"))
        }

        return ExprPath(segments)
    }

    private fun parseBlock(): Block {
        val start = expect(TokenKind.LBrace, "Expected '{'").span
        val stmts = mutableListOf<Stmt>()

        while (!check(TokenKind.RBrace) && !isAtEnd()) {
            skipNewlines()
            if (check(TokenKind.RBrace)) break
            stmts.add(parseStmt())
        }

        expect(TokenKind.RBrace, "Expected '}'")
        return Block(stmts, Span(start.start, previous().span.end, start.line, start.column))
    }

    private fun parseStmt(): Stmt {
        val start = peek().span

        return when {
            match(TokenKind.Let) -> {
                val pattern = parsePattern()
                val ty = if (match(TokenKind.Colon)) parseType() else null
                val init = if (match(TokenKind.Eq)) parseExpr() else null
                expect(TokenKind.Semicolon, "Expected ';'")
                Stmt.Let(pattern, ty, init, Span(start.start, previous().span.end, start.line, start.column))
            }
            else -> {
                val expr = parseExpr()
                val hasSemi = match(TokenKind.Semicolon)
                Stmt.Expr(expr, hasSemi, Span(start.start, previous().span.end, start.line, start.column))
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Patterns
    // ═══════════════════════════════════════════════════════════════════════

    private fun parsePattern(): Pattern {
        return when {
            match(TokenKind.Mut) -> {
                val name = expectIdent("Expected identifier")
                val evidence = parseEvidenceMarker()
                Pattern.Ident(name, mutable = true, evidence)
            }
            peek().kind is TokenKind.Ident -> {
                val name = expectIdent("Expected identifier")
                val evidence = parseEvidenceMarker()
                Pattern.Ident(name, mutable = false, evidence)
            }
            match(TokenKind.LParen) -> {
                val elements = parseDelimited(TokenKind.Comma, TokenKind.RParen) { parsePattern() }
                expect(TokenKind.RParen, "Expected ')'")
                Pattern.Tuple(elements)
            }
            peek().kind is TokenKind.IntLiteral -> {
                val token = advance()
                Pattern.Literal(LiteralKind.Int((token.kind as TokenKind.IntLiteral).value))
            }
            peek().kind is TokenKind.StringLiteral -> {
                val token = advance()
                Pattern.Literal(LiteralKind.String((token.kind as TokenKind.StringLiteral).value))
            }
            match(TokenKind.TrueLiteral) -> Pattern.Literal(LiteralKind.Bool(true))
            match(TokenKind.FalseLiteral) -> Pattern.Literal(LiteralKind.Bool(false))
            match(TokenKind.DotDot) -> Pattern.Rest(null)
            else -> Pattern.Wildcard
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Morpheme Operators
    // ═══════════════════════════════════════════════════════════════════════

    private fun checkMorpheme(): Boolean = when (peek().kind) {
        TokenKind.MorphemeTau, TokenKind.MorphemePhi, TokenKind.MorphemeSigma,
        TokenKind.MorphemeRho, TokenKind.MorphemeSigmaSum, TokenKind.MorphemePi,
        TokenKind.MorphemeAlpha, TokenKind.MorphemeOmega, TokenKind.MorphemeLambda,
        TokenKind.MorphemeDelta, TokenKind.MorphemeGamma -> true
        else -> false
    }

    private fun parseMorphemeOp(): MorphemeOp {
        val token = advance()
        return when (token.kind) {
            TokenKind.MorphemeTau -> MorphemeOp.Tau
            TokenKind.MorphemePhi -> MorphemeOp.Phi
            TokenKind.MorphemeSigma -> MorphemeOp.Sigma
            TokenKind.MorphemeRho -> MorphemeOp.Rho
            TokenKind.MorphemeSigmaSum -> MorphemeOp.Sum
            TokenKind.MorphemePi -> MorphemeOp.Product
            TokenKind.MorphemeAlpha -> MorphemeOp.Alpha
            TokenKind.MorphemeOmega -> MorphemeOp.Omega
            TokenKind.MorphemeLambda -> MorphemeOp.Lambda
            TokenKind.MorphemeDelta -> MorphemeOp.Delta
            TokenKind.MorphemeGamma -> MorphemeOp.Gamma
            else -> throw error("Expected morpheme operator")
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Operator Precedence
    // ═══════════════════════════════════════════════════════════════════════

    private enum class Assoc { Left, Right }

    private fun precedence(op: BinaryOp): Pair<Int, Assoc> = when (op) {
        BinaryOp.Assign, BinaryOp.AddAssign, BinaryOp.SubAssign,
        BinaryOp.MulAssign, BinaryOp.DivAssign, BinaryOp.RemAssign -> 1 to Assoc.Right
        BinaryOp.Or -> 2 to Assoc.Left
        BinaryOp.And -> 3 to Assoc.Left
        BinaryOp.Eq, BinaryOp.Ne, BinaryOp.Lt, BinaryOp.Le, BinaryOp.Gt, BinaryOp.Ge -> 4 to Assoc.Left
        BinaryOp.BitOr -> 5 to Assoc.Left
        BinaryOp.BitXor -> 6 to Assoc.Left
        BinaryOp.BitAnd -> 7 to Assoc.Left
        BinaryOp.Shl, BinaryOp.Shr -> 8 to Assoc.Left
        BinaryOp.Add, BinaryOp.Sub -> 9 to Assoc.Left
        BinaryOp.Mul, BinaryOp.Div, BinaryOp.Rem -> 10 to Assoc.Left
        BinaryOp.Pow -> 11 to Assoc.Right
    }

    private fun peekBinaryOp(): BinaryOp? = when (peek().kind) {
        TokenKind.Plus -> BinaryOp.Add
        TokenKind.Minus -> BinaryOp.Sub
        TokenKind.Star -> BinaryOp.Mul
        TokenKind.Slash -> BinaryOp.Div
        TokenKind.Percent -> BinaryOp.Rem
        TokenKind.Caret -> BinaryOp.Pow
        TokenKind.AndAnd -> BinaryOp.And
        TokenKind.OrOr -> BinaryOp.Or
        TokenKind.And -> BinaryOp.BitAnd
        TokenKind.Or -> BinaryOp.BitOr
        TokenKind.EqEq -> BinaryOp.Eq
        TokenKind.NotEq -> BinaryOp.Ne
        TokenKind.Lt -> BinaryOp.Lt
        TokenKind.LtEq -> BinaryOp.Le
        TokenKind.Gt -> BinaryOp.Gt
        TokenKind.GtEq -> BinaryOp.Ge
        TokenKind.Eq -> BinaryOp.Assign
        else -> null
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private fun isAtEnd(): Boolean = current >= tokens.size || peek().kind == TokenKind.Eof

    private fun peek(): Token = tokens.getOrElse(current) { tokens.last() }

    private fun previous(): Token = tokens[current - 1]

    private fun advance(): Token {
        if (!isAtEnd()) current++
        return previous()
    }

    private fun check(kind: TokenKind): Boolean = !isAtEnd() && peek().kind == kind

    private fun match(kind: TokenKind): Boolean {
        if (!check(kind)) return false
        advance()
        return true
    }

    private fun expect(kind: TokenKind, message: String): Token {
        if (check(kind)) return advance()
        throw error(message)
    }

    private fun expectIdent(message: String): String {
        val token = peek()
        return when (val kind = token.kind) {
            is TokenKind.Ident -> {
                advance()
                kind.name
            }
            else -> throw error(message)
        }
    }

    private fun skipNewlines() {
        while (match(TokenKind.Newline)) { /* skip */ }
    }

    private inline fun <T> parseDelimited(delimiter: TokenKind, terminator: TokenKind, parse: () -> T): List<T> {
        val items = mutableListOf<T>()
        if (!check(terminator)) {
            do {
                skipNewlines()
                if (check(terminator)) break
                items.add(parse())
            } while (match(delimiter))
        }
        return items
    }

    private fun error(message: String): ParseException =
        ParseException(message, peek().span)

    private fun synchronize() {
        advance()
        while (!isAtEnd()) {
            if (previous().kind == TokenKind.Semicolon) return
            when (peek().kind) {
                TokenKind.Fn, TokenKind.Struct, TokenKind.Enum, TokenKind.Trait,
                TokenKind.Impl, TokenKind.Use, TokenKind.Pub -> return
                else -> advance()
            }
        }
    }

    class ParseException(message: String, val span: Span) : Exception(message)
}

data class ParseResult(
    val module: Module,
    val errors: List<Parser.ParseError>
) {
    val hasErrors: Boolean get() = errors.isNotEmpty()
}

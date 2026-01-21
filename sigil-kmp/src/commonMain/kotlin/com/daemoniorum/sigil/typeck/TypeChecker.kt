package com.daemoniorum.sigil.typeck

import com.daemoniorum.sigil.ast.*
import com.daemoniorum.sigil.lexer.Span
import com.daemoniorum.sigil.runtime.EvidenceLevel

/**
 * Type checker for Sigil with evidentiality enforcement.
 *
 * The key insight: evidence flows like types, but with covariance.
 * Known (!) can flow to Uncertain (?), which can flow to Reported (~).
 * The reverse is NOT allowed without explicit validation.
 */
class TypeChecker {
    private val errors = mutableListOf<TypeError>()
    private val scopes = mutableListOf<MutableMap<String, TypeInfo>>()
    private val typeAliases = mutableMapOf<String, SigilType>()
    private var nextTypeVar = 0

    data class TypeError(
        val message: String,
        val span: Span,
        val kind: ErrorKind
    )

    enum class ErrorKind {
        TypeMismatch,
        EvidenceMismatch,
        UndefinedVariable,
        UndefinedType,
        UndefinedFunction,
        InvalidOperation,
        MissingEvidence
    }

    data class TypeInfo(
        val type: SigilType,
        val evidence: EvidenceLevel,
        val mutable: Boolean = false
    )

    /**
     * Check a module
     */
    fun check(module: Module): TypeCheckResult {
        pushScope()

        // First pass: register all type definitions
        for (item in module.items) {
            when (item) {
                is Item.Struct -> registerStruct(item)
                is Item.Enum -> registerEnum(item)
                is Item.TypeAlias -> registerTypeAlias(item)
                is Item.Trait -> registerTrait(item)
                else -> { }
            }
        }

        // Second pass: register all function signatures
        for (item in module.items) {
            if (item is Item.Function) {
                registerFunction(item)
            }
        }

        // Third pass: check function bodies
        for (item in module.items) {
            checkItem(item)
        }

        popScope()
        return TypeCheckResult(errors)
    }

    private fun checkItem(item: Item) {
        when (item) {
            is Item.Function -> checkFunction(item)
            is Item.Struct -> checkStruct(item)
            is Item.Enum -> checkEnum(item)
            is Item.Impl -> checkImpl(item)
            is Item.Trait -> checkTrait(item)
            is Item.Const -> checkConst(item)
            is Item.TypeAlias -> { } // Already registered
            is Item.Use -> { } // Handled separately
        }
    }

    private fun registerStruct(struct: Item.Struct) {
        val fields = struct.fields.associate { field ->
            field.name to TypeInfo(
                resolveType(field.ty),
                field.evidence ?: EvidenceLevel.KNOWN
            )
        }
        typeAliases[struct.name] = SigilType.Struct(struct.name, fields)
    }

    private fun registerEnum(enum: Item.Enum) {
        val variants = enum.variants.associate { variant ->
            variant.name to when (val fields = variant.fields) {
                is VariantFields.Tuple -> fields.types.map { resolveType(it) }
                is VariantFields.Struct -> fields.fields.map { resolveType(it.ty) }
                null -> emptyList()
            }
        }
        typeAliases[enum.name] = SigilType.Enum(enum.name, variants)
    }

    private fun registerTypeAlias(alias: Item.TypeAlias) {
        typeAliases[alias.name] = resolveType(alias.ty)
    }

    private fun registerTrait(trait: Item.Trait) {
        // Register trait as a type
        typeAliases[trait.name] = SigilType.Trait(trait.name)
    }

    private fun registerFunction(func: Item.Function) {
        val paramTypes = func.params.map { param ->
            TypeInfo(
                resolveType(param.ty),
                param.evidence ?: EvidenceLevel.KNOWN
            )
        }
        val returnType = func.returnType?.let { resolveType(it) } ?: SigilType.Unit
        val returnEvidence = func.returnEvidence ?: EvidenceLevel.KNOWN

        define(func.name, TypeInfo(
            SigilType.Function(
                paramTypes.map { it.type },
                returnType,
                returnEvidence
            ),
            EvidenceLevel.KNOWN
        ))
    }

    private fun checkFunction(func: Item.Function) {
        pushScope()

        // Define parameters
        for (param in func.params) {
            val patternName = (param.pattern as? Pattern.Ident)?.name ?: continue
            define(patternName, TypeInfo(
                resolveType(param.ty),
                param.evidence ?: EvidenceLevel.KNOWN,
                mutable = (param.pattern as? Pattern.Ident)?.mutable ?: false
            ))
        }

        // Check body
        val bodyType = checkBlock(func.body)

        // Verify return type
        val expectedReturn = func.returnType?.let { resolveType(it) } ?: SigilType.Unit
        val expectedEvidence = func.returnEvidence ?: EvidenceLevel.KNOWN

        if (!isAssignable(bodyType.type, expectedReturn)) {
            error(
                "Return type mismatch: expected $expectedReturn, got ${bodyType.type}",
                func.body.span,
                ErrorKind.TypeMismatch
            )
        }

        checkEvidenceFlow(bodyType.evidence, expectedEvidence, func.body.span)

        popScope()
    }

    private fun checkStruct(struct: Item.Struct) {
        // Check field types are valid
        for (field in struct.fields) {
            resolveType(field.ty)
        }
    }

    private fun checkEnum(enum: Item.Enum) {
        // Check variant types are valid
        for (variant in enum.variants) {
            when (val fields = variant.fields) {
                is VariantFields.Tuple -> fields.types.forEach { resolveType(it) }
                is VariantFields.Struct -> fields.fields.forEach { resolveType(it.ty) }
                null -> { }
            }
        }
    }

    private fun checkImpl(impl: Item.Impl) {
        pushScope()

        // Define 'self' type
        define("self", TypeInfo(resolveType(impl.selfType), EvidenceLevel.KNOWN))
        define("Self", TypeInfo(resolveType(impl.selfType), EvidenceLevel.KNOWN))

        for (item in impl.items) {
            when (item) {
                is ImplItem.Method -> {
                    pushScope()
                    for (param in item.params) {
                        val patternName = (param.pattern as? Pattern.Ident)?.name ?: continue
                        define(patternName, TypeInfo(
                            resolveType(param.ty),
                            param.evidence ?: EvidenceLevel.KNOWN
                        ))
                    }
                    checkBlock(item.body)
                    popScope()
                }
                is ImplItem.Type -> {
                    resolveType(item.ty)
                }
            }
        }

        popScope()
    }

    private fun checkTrait(trait: Item.Trait) {
        // Check trait items
        for (item in trait.items) {
            when (item) {
                is TraitItem.Method -> {
                    item.body?.let { checkBlock(it) }
                }
                is TraitItem.Type -> { }
            }
        }
    }

    private fun checkConst(const: Item.Const) {
        val valueType = checkExpr(const.value)
        val expectedType = resolveType(const.ty)

        if (!isAssignable(valueType.type, expectedType)) {
            error(
                "Constant type mismatch: expected $expectedType, got ${valueType.type}",
                const.span,
                ErrorKind.TypeMismatch
            )
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Expression Type Checking
    // ═══════════════════════════════════════════════════════════════════════

    private fun checkExpr(expr: Expr): TypeInfo {
        return when (expr) {
            is Expr.Literal -> checkLiteral(expr)
            is Expr.Path -> checkPath(expr)
            is Expr.Binary -> checkBinary(expr)
            is Expr.Unary -> checkUnary(expr)
            is Expr.Call -> checkCall(expr)
            is Expr.MethodCall -> checkMethodCall(expr)
            is Expr.Field -> checkField(expr)
            is Expr.Index -> checkIndex(expr)
            is Expr.If -> checkIf(expr)
            is Expr.Match -> checkMatch(expr)
            is Expr.Block -> checkBlock(expr.block)
            is Expr.Loop -> checkLoop(expr)
            is Expr.While -> checkWhile(expr)
            is Expr.For -> checkFor(expr)
            is Expr.Return -> checkReturn(expr)
            is Expr.Break -> checkBreak(expr)
            is Expr.Continue -> TypeInfo(SigilType.Never, EvidenceLevel.KNOWN)
            is Expr.Lambda -> checkLambda(expr)
            is Expr.Struct -> checkStructLiteral(expr)
            is Expr.Array -> checkArray(expr)
            is Expr.Tuple -> checkTuple(expr)
            is Expr.Range -> checkRange(expr)
            is Expr.Morpheme -> checkMorpheme(expr)
            is Expr.Pipe -> checkPipe(expr)
            is Expr.Await -> checkAwait(expr)
            is Expr.Try -> checkTry(expr)
            is Expr.Cast -> checkCast(expr)
        }
    }

    private fun checkLiteral(expr: Expr.Literal): TypeInfo {
        val type = when (expr.kind) {
            is LiteralKind.Int -> SigilType.I64
            is LiteralKind.Float -> SigilType.F64
            is LiteralKind.String -> SigilType.Str
            is LiteralKind.Char -> SigilType.Char
            is LiteralKind.Bool -> SigilType.Bool
            is LiteralKind.Unit -> SigilType.Unit
        }
        return TypeInfo(type, expr.evidence)
    }

    private fun checkPath(expr: Expr.Path): TypeInfo {
        val name = expr.path.segments.joinToString("::")
        return lookup(name) ?: run {
            error("Undefined variable: $name", expr.span, ErrorKind.UndefinedVariable)
            TypeInfo(SigilType.Error, EvidenceLevel.PARADOX)
        }
    }

    private fun checkBinary(expr: Expr.Binary): TypeInfo {
        val left = checkExpr(expr.left)
        val right = checkExpr(expr.right)

        // Evidence join: combined evidence is the weakest
        val evidence = EvidenceLevel.join(left.evidence, right.evidence)

        val resultType = when (expr.op) {
            BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Div, BinaryOp.Rem, BinaryOp.Pow -> {
                unifyNumeric(left.type, right.type, expr.span)
            }
            BinaryOp.And, BinaryOp.Or -> {
                expectType(left.type, SigilType.Bool, expr.left.span)
                expectType(right.type, SigilType.Bool, expr.right.span)
                SigilType.Bool
            }
            BinaryOp.Eq, BinaryOp.Ne, BinaryOp.Lt, BinaryOp.Le, BinaryOp.Gt, BinaryOp.Ge -> {
                SigilType.Bool
            }
            BinaryOp.BitAnd, BinaryOp.BitOr, BinaryOp.BitXor, BinaryOp.Shl, BinaryOp.Shr -> {
                left.type
            }
            BinaryOp.Assign, BinaryOp.AddAssign, BinaryOp.SubAssign,
            BinaryOp.MulAssign, BinaryOp.DivAssign, BinaryOp.RemAssign -> {
                // For assignment, check evidence flow
                checkEvidenceFlow(right.evidence, left.evidence, expr.span)
                SigilType.Unit
            }
        }

        return TypeInfo(resultType, evidence)
    }

    private fun checkUnary(expr: Expr.Unary): TypeInfo {
        val operand = checkExpr(expr.operand)

        val type = when (expr.op) {
            UnaryOp.Neg -> {
                if (!isNumeric(operand.type)) {
                    error("Cannot negate non-numeric type", expr.span, ErrorKind.InvalidOperation)
                }
                operand.type
            }
            UnaryOp.Not -> {
                expectType(operand.type, SigilType.Bool, expr.operand.span)
                SigilType.Bool
            }
            UnaryOp.Deref -> {
                when (val t = operand.type) {
                    is SigilType.Reference -> t.inner
                    else -> {
                        error("Cannot dereference non-reference type", expr.span, ErrorKind.InvalidOperation)
                        SigilType.Error
                    }
                }
            }
            UnaryOp.Ref, UnaryOp.RefMut -> {
                SigilType.Reference(operand.type, expr.op == UnaryOp.RefMut)
            }
        }

        return TypeInfo(type, operand.evidence)
    }

    private fun checkCall(expr: Expr.Call): TypeInfo {
        val callee = checkExpr(expr.callee)

        return when (val calleeType = callee.type) {
            is SigilType.Function -> {
                // Check argument count
                if (expr.args.size != calleeType.params.size) {
                    error(
                        "Wrong number of arguments: expected ${calleeType.params.size}, got ${expr.args.size}",
                        expr.span,
                        ErrorKind.TypeMismatch
                    )
                }

                // Check argument types and evidence
                var evidence = callee.evidence
                for ((i, arg) in expr.args.withIndex()) {
                    val argType = checkExpr(arg)
                    if (i < calleeType.params.size) {
                        if (!isAssignable(argType.type, calleeType.params[i])) {
                            error(
                                "Argument type mismatch: expected ${calleeType.params[i]}, got ${argType.type}",
                                arg.span,
                                ErrorKind.TypeMismatch
                            )
                        }
                    }
                    evidence = EvidenceLevel.join(evidence, argType.evidence)
                }

                TypeInfo(calleeType.returnType, EvidenceLevel.join(evidence, calleeType.returnEvidence))
            }
            else -> {
                error("Cannot call non-function type", expr.span, ErrorKind.InvalidOperation)
                TypeInfo(SigilType.Error, EvidenceLevel.PARADOX)
            }
        }
    }

    private fun checkMethodCall(expr: Expr.MethodCall): TypeInfo {
        val receiver = checkExpr(expr.receiver)

        // Look up method on type
        // For now, return a placeholder
        var evidence = receiver.evidence
        for (arg in expr.args) {
            evidence = EvidenceLevel.join(evidence, checkExpr(arg).evidence)
        }

        return TypeInfo(freshTypeVar(), evidence)
    }

    private fun checkField(expr: Expr.Field): TypeInfo {
        val receiver = checkExpr(expr.receiver)

        return when (val recType = receiver.type) {
            is SigilType.Struct -> {
                val field = recType.fields[expr.field]
                if (field == null) {
                    error("Unknown field: ${expr.field}", expr.span, ErrorKind.UndefinedVariable)
                    TypeInfo(SigilType.Error, EvidenceLevel.PARADOX)
                } else {
                    TypeInfo(field.type, EvidenceLevel.join(receiver.evidence, field.evidence))
                }
            }
            is SigilType.Tuple -> {
                val index = expr.field.toIntOrNull()
                if (index != null && index < recType.elements.size) {
                    TypeInfo(recType.elements[index], receiver.evidence)
                } else {
                    error("Invalid tuple index: ${expr.field}", expr.span, ErrorKind.InvalidOperation)
                    TypeInfo(SigilType.Error, EvidenceLevel.PARADOX)
                }
            }
            else -> {
                error("Cannot access field on type: $recType", expr.span, ErrorKind.InvalidOperation)
                TypeInfo(SigilType.Error, EvidenceLevel.PARADOX)
            }
        }
    }

    private fun checkIndex(expr: Expr.Index): TypeInfo {
        val receiver = checkExpr(expr.receiver)
        val index = checkExpr(expr.index)

        val elementType = when (val recType = receiver.type) {
            is SigilType.Array -> recType.element
            is SigilType.Str -> SigilType.Char
            else -> {
                error("Cannot index type: $recType", expr.span, ErrorKind.InvalidOperation)
                SigilType.Error
            }
        }

        return TypeInfo(elementType, EvidenceLevel.join(receiver.evidence, index.evidence))
    }

    private fun checkIf(expr: Expr.If): TypeInfo {
        val cond = checkExpr(expr.condition)
        expectType(cond.type, SigilType.Bool, expr.condition.span)

        val thenType = checkBlock(expr.thenBranch)
        val elseType = expr.elseBranch?.let { checkBlock(it) }
            ?: TypeInfo(SigilType.Unit, EvidenceLevel.KNOWN)

        // Unify branch types
        val resultType = if (isAssignable(thenType.type, elseType.type)) {
            thenType.type
        } else if (isAssignable(elseType.type, thenType.type)) {
            elseType.type
        } else {
            error(
                "If branches have incompatible types: ${thenType.type} vs ${elseType.type}",
                expr.span,
                ErrorKind.TypeMismatch
            )
            SigilType.Error
        }

        // Evidence join from all branches
        val evidence = EvidenceLevel.join(
            cond.evidence,
            EvidenceLevel.join(thenType.evidence, elseType.evidence)
        )

        return TypeInfo(resultType, evidence)
    }

    private fun checkMatch(expr: Expr.Match): TypeInfo {
        val scrutinee = checkExpr(expr.scrutinee)
        var evidence = scrutinee.evidence
        var resultType: SigilType? = null

        for (arm in expr.arms) {
            // Check pattern
            checkPattern(arm.pattern, scrutinee.type)

            // Check guard
            arm.guard?.let {
                val guardType = checkExpr(it)
                expectType(guardType.type, SigilType.Bool, it.span)
                evidence = EvidenceLevel.join(evidence, guardType.evidence)
            }

            // Check body
            val bodyType = checkExpr(arm.body)
            evidence = EvidenceLevel.join(evidence, bodyType.evidence)

            if (resultType == null) {
                resultType = bodyType.type
            } else if (!isAssignable(bodyType.type, resultType)) {
                error(
                    "Match arm has incompatible type: expected $resultType, got ${bodyType.type}",
                    arm.body.span,
                    ErrorKind.TypeMismatch
                )
            }
        }

        return TypeInfo(resultType ?: SigilType.Unit, evidence)
    }

    private fun checkBlock(block: Block): TypeInfo {
        pushScope()

        var lastType: TypeInfo = TypeInfo(SigilType.Unit, EvidenceLevel.KNOWN)

        for (stmt in block.stmts) {
            lastType = when (stmt) {
                is Stmt.Let -> {
                    val init = stmt.init?.let { checkExpr(it) }
                    val declaredType = stmt.ty?.let { resolveType(it) }
                    val actualType = init?.type ?: declaredType ?: freshTypeVar()

                    if (declaredType != null && init != null && !isAssignable(init.type, declaredType)) {
                        error(
                            "Type mismatch in let: expected $declaredType, got ${init.type}",
                            stmt.span,
                            ErrorKind.TypeMismatch
                        )
                    }

                    val patternName = (stmt.pattern as? Pattern.Ident)?.name
                    val patternEvidence = (stmt.pattern as? Pattern.Ident)?.evidence
                    val evidence = patternEvidence ?: init?.evidence ?: EvidenceLevel.KNOWN

                    if (patternName != null) {
                        define(patternName, TypeInfo(
                            actualType,
                            evidence,
                            mutable = (stmt.pattern as? Pattern.Ident)?.mutable ?: false
                        ))
                    }

                    TypeInfo(SigilType.Unit, evidence)
                }
                is Stmt.Expr -> {
                    val exprType = checkExpr(stmt.expr)
                    if (stmt.hasSemi) {
                        TypeInfo(SigilType.Unit, exprType.evidence)
                    } else {
                        exprType
                    }
                }
                is Stmt.Item -> {
                    checkItem(stmt.item)
                    TypeInfo(SigilType.Unit, EvidenceLevel.KNOWN)
                }
            }
        }

        popScope()
        return lastType
    }

    private fun checkLoop(expr: Expr.Loop): TypeInfo {
        val bodyType = checkBlock(expr.body)
        // Loops return Never unless broken
        return TypeInfo(SigilType.Never, bodyType.evidence)
    }

    private fun checkWhile(expr: Expr.While): TypeInfo {
        val cond = checkExpr(expr.condition)
        expectType(cond.type, SigilType.Bool, expr.condition.span)
        val body = checkBlock(expr.body)
        return TypeInfo(SigilType.Unit, EvidenceLevel.join(cond.evidence, body.evidence))
    }

    private fun checkFor(expr: Expr.For): TypeInfo {
        val iter = checkExpr(expr.iterable)

        pushScope()
        // Bind pattern to element type
        val elementType = when (val iterType = iter.type) {
            is SigilType.Array -> iterType.element
            is SigilType.Str -> SigilType.Char
            else -> freshTypeVar()
        }

        val patternName = (expr.pattern as? Pattern.Ident)?.name
        if (patternName != null) {
            define(patternName, TypeInfo(elementType, iter.evidence))
        }

        val body = checkBlock(expr.body)
        popScope()

        return TypeInfo(SigilType.Unit, EvidenceLevel.join(iter.evidence, body.evidence))
    }

    private fun checkReturn(expr: Expr.Return): TypeInfo {
        expr.value?.let { checkExpr(it) }
        return TypeInfo(SigilType.Never, EvidenceLevel.KNOWN)
    }

    private fun checkBreak(expr: Expr.Break): TypeInfo {
        expr.value?.let { checkExpr(it) }
        return TypeInfo(SigilType.Never, EvidenceLevel.KNOWN)
    }

    private fun checkLambda(expr: Expr.Lambda): TypeInfo {
        pushScope()

        val paramTypes = expr.params.map { param ->
            val ty = param.ty?.let { resolveType(it) } ?: freshTypeVar()
            val name = (param.pattern as? Pattern.Ident)?.name
            if (name != null) {
                define(name, TypeInfo(ty, EvidenceLevel.KNOWN))
            }
            ty
        }

        val bodyType = checkExpr(expr.body)
        popScope()

        return TypeInfo(
            SigilType.Function(paramTypes, bodyType.type, bodyType.evidence),
            EvidenceLevel.KNOWN
        )
    }

    private fun checkStructLiteral(expr: Expr.Struct): TypeInfo {
        val typeName = expr.path.segments.joinToString("::") { it.name }
        val structType = typeAliases[typeName] as? SigilType.Struct

        if (structType == null) {
            error("Unknown struct type: $typeName", expr.span, ErrorKind.UndefinedType)
            return TypeInfo(SigilType.Error, EvidenceLevel.PARADOX)
        }

        var evidence = EvidenceLevel.KNOWN

        for (field in expr.fields) {
            val fieldInfo = structType.fields[field.name]
            if (fieldInfo == null) {
                error("Unknown field: ${field.name}", expr.span, ErrorKind.UndefinedVariable)
            } else {
                val valueType = checkExpr(field.value)
                if (!isAssignable(valueType.type, fieldInfo.type)) {
                    error(
                        "Field type mismatch: expected ${fieldInfo.type}, got ${valueType.type}",
                        field.value.span,
                        ErrorKind.TypeMismatch
                    )
                }
                checkEvidenceFlow(valueType.evidence, fieldInfo.evidence, field.value.span)
                evidence = EvidenceLevel.join(evidence, valueType.evidence)
            }
        }

        return TypeInfo(structType, evidence)
    }

    private fun checkArray(expr: Expr.Array): TypeInfo {
        if (expr.elements.isEmpty()) {
            return TypeInfo(SigilType.Array(freshTypeVar()), EvidenceLevel.KNOWN)
        }

        var elementType: SigilType = SigilType.Error
        var evidence = EvidenceLevel.KNOWN

        for ((i, elem) in expr.elements.withIndex()) {
            val elemType = checkExpr(elem)
            if (i == 0) {
                elementType = elemType.type
            } else if (!isAssignable(elemType.type, elementType)) {
                error(
                    "Array element type mismatch: expected $elementType, got ${elemType.type}",
                    elem.span,
                    ErrorKind.TypeMismatch
                )
            }
            evidence = EvidenceLevel.join(evidence, elemType.evidence)
        }

        return TypeInfo(SigilType.Array(elementType), evidence)
    }

    private fun checkTuple(expr: Expr.Tuple): TypeInfo {
        val types = mutableListOf<SigilType>()
        var evidence = EvidenceLevel.KNOWN

        for (elem in expr.elements) {
            val elemType = checkExpr(elem)
            types.add(elemType.type)
            evidence = EvidenceLevel.join(evidence, elemType.evidence)
        }

        return TypeInfo(SigilType.Tuple(types), evidence)
    }

    private fun checkRange(expr: Expr.Range): TypeInfo {
        val startType = expr.start?.let { checkExpr(it) }
        val endType = expr.end?.let { checkExpr(it) }

        var evidence = EvidenceLevel.KNOWN
        if (startType != null) evidence = EvidenceLevel.join(evidence, startType.evidence)
        if (endType != null) evidence = EvidenceLevel.join(evidence, endType.evidence)

        return TypeInfo(SigilType.Range, evidence)
    }

    private fun checkMorpheme(expr: Expr.Morpheme): TypeInfo {
        val input = checkExpr(expr.input)
        val transform = expr.transform?.let { checkExpr(it) }

        var evidence = input.evidence
        if (transform != null) {
            evidence = EvidenceLevel.join(evidence, transform.evidence)
        }

        // Morpheme result type depends on operator
        val resultType = when (expr.op) {
            MorphemeOp.Tau -> {
                // τ preserves collection type, transforms elements
                when (val inputType = input.type) {
                    is SigilType.Array -> {
                        val elementType = transform?.type?.let { extractReturnType(it) } ?: inputType.element
                        SigilType.Array(elementType)
                    }
                    else -> freshTypeVar()
                }
            }
            MorphemeOp.Phi -> input.type // Filter preserves type
            MorphemeOp.Sigma -> input.type // Sort preserves type
            MorphemeOp.Rho -> transform?.type?.let { extractReturnType(it) } ?: freshTypeVar()
            MorphemeOp.Sum, MorphemeOp.Product -> SigilType.F64
            MorphemeOp.Alpha, MorphemeOp.Omega -> {
                when (val inputType = input.type) {
                    is SigilType.Array -> SigilType.Option(inputType.element)
                    else -> SigilType.Option(freshTypeVar())
                }
            }
            MorphemeOp.Delta -> input.type
            MorphemeOp.Lambda -> freshTypeVar()
            MorphemeOp.Gamma -> SigilType.Map(freshTypeVar(), input.type)
        }

        return TypeInfo(resultType, evidence)
    }

    private fun checkPipe(expr: Expr.Pipe): TypeInfo {
        val left = checkExpr(expr.left)
        val right = checkExpr(expr.right)

        // The right side should be callable
        val resultType = when (val rightType = right.type) {
            is SigilType.Function -> rightType.returnType
            else -> freshTypeVar()
        }

        return TypeInfo(resultType, EvidenceLevel.join(left.evidence, right.evidence))
    }

    private fun checkAwait(expr: Expr.Await): TypeInfo {
        val inner = checkExpr(expr.expr)
        // Await unwraps Future<T> to T
        return inner
    }

    private fun checkTry(expr: Expr.Try): TypeInfo {
        val inner = checkExpr(expr.expr)
        // Try propagates errors
        return inner
    }

    private fun checkCast(expr: Expr.Cast): TypeInfo {
        val inner = checkExpr(expr.expr)
        val targetType = resolveType(expr.ty)
        return TypeInfo(targetType, inner.evidence)
    }

    private fun checkPattern(pattern: Pattern, expectedType: SigilType) {
        when (pattern) {
            is Pattern.Ident -> {
                define(pattern.name, TypeInfo(
                    expectedType,
                    pattern.evidence ?: EvidenceLevel.KNOWN,
                    mutable = pattern.mutable
                ))
            }
            is Pattern.Literal -> { }
            is Pattern.Tuple -> {
                if (expectedType is SigilType.Tuple && expectedType.elements.size == pattern.elements.size) {
                    for ((pat, ty) in pattern.elements.zip(expectedType.elements)) {
                        checkPattern(pat, ty)
                    }
                }
            }
            is Pattern.Struct -> { }
            is Pattern.TupleStruct -> { }
            is Pattern.Or -> pattern.patterns.forEach { checkPattern(it, expectedType) }
            is Pattern.Range -> { }
            is Pattern.Wildcard -> { }
            is Pattern.Rest -> { }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Evidence Checking
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Check that evidence can flow from source to target.
     * Known (!) → Uncertain (?) → Reported (~) → Paradox (‽)
     */
    private fun checkEvidenceFlow(source: EvidenceLevel, target: EvidenceLevel, span: Span) {
        if (!source.canFlowTo(target)) {
            error(
                "Evidence mismatch: cannot assign ${source.symbol} to ${target.symbol}. " +
                "Evidence must flow from stronger to weaker (! → ? → ~ → ‽)",
                span,
                ErrorKind.EvidenceMismatch
            )
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Type Resolution & Unification
    // ═══════════════════════════════════════════════════════════════════════

    private fun resolveType(type: Type): SigilType {
        return when (type) {
            is Type.Path -> {
                val name = type.path.segments.joinToString("::") { it.name }
                when (name) {
                    "i8" -> SigilType.I8
                    "i16" -> SigilType.I16
                    "i32" -> SigilType.I32
                    "i64" -> SigilType.I64
                    "u8" -> SigilType.U8
                    "u16" -> SigilType.U16
                    "u32" -> SigilType.U32
                    "u64" -> SigilType.U64
                    "f32" -> SigilType.F32
                    "f64" -> SigilType.F64
                    "bool" -> SigilType.Bool
                    "str", "String" -> SigilType.Str
                    "char" -> SigilType.Char
                    else -> typeAliases[name] ?: SigilType.Named(name)
                }
            }
            is Type.Array -> SigilType.Array(resolveType(type.element))
            is Type.Tuple -> SigilType.Tuple(type.elements.map { resolveType(it) })
            is Type.Function -> SigilType.Function(
                type.params.map { resolveType(it) },
                resolveType(type.returnType),
                EvidenceLevel.KNOWN
            )
            is Type.Reference -> SigilType.Reference(resolveType(type.ty), type.mutable)
            is Type.Option -> SigilType.Option(resolveType(type.ty))
            is Type.Result -> SigilType.Result(resolveType(type.ok), resolveType(type.err))
            is Type.Infer -> freshTypeVar()
            is Type.Unit -> SigilType.Unit
            is Type.Never -> SigilType.Never
        }
    }

    private fun isAssignable(from: SigilType, to: SigilType): Boolean {
        if (from == to) return true
        if (from is SigilType.Never) return true
        if (to is SigilType.Var || from is SigilType.Var) return true
        if (from is SigilType.Error || to is SigilType.Error) return true

        return when {
            from is SigilType.Array && to is SigilType.Array ->
                isAssignable(from.element, to.element)
            from is SigilType.Tuple && to is SigilType.Tuple &&
                from.elements.size == to.elements.size ->
                from.elements.zip(to.elements).all { (f, t) -> isAssignable(f, t) }
            from is SigilType.Option && to is SigilType.Option ->
                isAssignable(from.inner, to.inner)
            else -> false
        }
    }

    private fun unifyNumeric(a: SigilType, b: SigilType, span: Span): SigilType {
        if (a == b) return a
        if (isNumeric(a) && isNumeric(b)) {
            // Prefer floats over ints, larger over smaller
            return when {
                a == SigilType.F64 || b == SigilType.F64 -> SigilType.F64
                a == SigilType.F32 || b == SigilType.F32 -> SigilType.F32
                a == SigilType.I64 || b == SigilType.I64 -> SigilType.I64
                else -> SigilType.I32
            }
        }
        error("Cannot unify types: $a and $b", span, ErrorKind.TypeMismatch)
        return SigilType.Error
    }

    private fun isNumeric(type: SigilType): Boolean = type in listOf(
        SigilType.I8, SigilType.I16, SigilType.I32, SigilType.I64,
        SigilType.U8, SigilType.U16, SigilType.U32, SigilType.U64,
        SigilType.F32, SigilType.F64
    )

    private fun expectType(actual: SigilType, expected: SigilType, span: Span) {
        if (!isAssignable(actual, expected)) {
            error("Type mismatch: expected $expected, got $actual", span, ErrorKind.TypeMismatch)
        }
    }

    private fun extractReturnType(type: SigilType): SigilType {
        return when (type) {
            is SigilType.Function -> type.returnType
            else -> type
        }
    }

    private fun freshTypeVar(): SigilType.Var = SigilType.Var(nextTypeVar++)

    // ═══════════════════════════════════════════════════════════════════════
    // Scope Management
    // ═══════════════════════════════════════════════════════════════════════

    private fun pushScope() {
        scopes.add(mutableMapOf())
    }

    private fun popScope() {
        scopes.removeAt(scopes.lastIndex)
    }

    private fun define(name: String, info: TypeInfo) {
        scopes.last()[name] = info
    }

    private fun lookup(name: String): TypeInfo? {
        for (scope in scopes.asReversed()) {
            scope[name]?.let { return it }
        }
        return null
    }

    private fun error(message: String, span: Span, kind: ErrorKind) {
        errors.add(TypeError(message, span, kind))
    }
}

/**
 * Sigil types (internal representation)
 */
sealed class SigilType {
    // Primitives
    object I8 : SigilType() { override fun toString() = "i8" }
    object I16 : SigilType() { override fun toString() = "i16" }
    object I32 : SigilType() { override fun toString() = "i32" }
    object I64 : SigilType() { override fun toString() = "i64" }
    object U8 : SigilType() { override fun toString() = "u8" }
    object U16 : SigilType() { override fun toString() = "u16" }
    object U32 : SigilType() { override fun toString() = "u32" }
    object U64 : SigilType() { override fun toString() = "u64" }
    object F32 : SigilType() { override fun toString() = "f32" }
    object F64 : SigilType() { override fun toString() = "f64" }
    object Bool : SigilType() { override fun toString() = "bool" }
    object Str : SigilType() { override fun toString() = "str" }
    object Char : SigilType() { override fun toString() = "char" }
    object Unit : SigilType() { override fun toString() = "()" }
    object Never : SigilType() { override fun toString() = "!" }
    object Range : SigilType() { override fun toString() = "Range" }
    object Error : SigilType() { override fun toString() = "<error>" }

    // Compound types
    data class Array(val element: SigilType) : SigilType() {
        override fun toString() = "[$element]"
    }
    data class Tuple(val elements: List<SigilType>) : SigilType() {
        override fun toString() = "(${elements.joinToString(", ")})"
    }
    data class Option(val inner: SigilType) : SigilType() {
        override fun toString() = "$inner?"
    }
    data class Result(val ok: SigilType, val err: SigilType) : SigilType() {
        override fun toString() = "Result<$ok, $err>"
    }
    data class Reference(val inner: SigilType, val mutable: Boolean) : SigilType() {
        override fun toString() = if (mutable) "&mut $inner" else "&$inner"
    }
    data class Function(
        val params: List<SigilType>,
        val returnType: SigilType,
        val returnEvidence: EvidenceLevel
    ) : SigilType() {
        override fun toString() = "fn(${params.joinToString(", ")}) -> $returnType${returnEvidence.symbol}"
    }
    data class Struct(
        val name: String,
        val fields: Map<String, TypeChecker.TypeInfo>
    ) : SigilType() {
        override fun toString() = name
    }
    data class Enum(
        val name: String,
        val variants: Map<String, List<SigilType>>
    ) : SigilType() {
        override fun toString() = name
    }
    data class Trait(val name: String) : SigilType() {
        override fun toString() = "dyn $name"
    }
    data class Named(val name: String) : SigilType() {
        override fun toString() = name
    }
    data class Var(val id: Int) : SigilType() {
        override fun toString() = "T$id"
    }
    data class Map(val key: SigilType, val value: SigilType) : SigilType() {
        override fun toString() = "Map<$key, $value>"
    }
}

data class TypeCheckResult(
    val errors: List<TypeChecker.TypeError>
) {
    val hasErrors: Boolean get() = errors.isNotEmpty()
}

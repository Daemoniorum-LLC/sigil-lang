package com.daemoniorum.sigil.interpreter

import com.daemoniorum.sigil.ast.*
import com.daemoniorum.sigil.runtime.*
import com.daemoniorum.sigil.lexer.Span
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Tree-walking interpreter for Sigil.
 *
 * Executes Sigil code with full evidence tracking.
 * Every value carries its provenance.
 */
class Interpreter {
    private val scopes = mutableListOf<MutableMap<String, SigilValue>>()
    private val functions = mutableMapOf<String, Item.Function>()
    private val structs = mutableMapOf<String, Item.Struct>()
    private var output = StringBuilder()

    // Control flow signals
    private class ReturnSignal(val value: SigilValue) : Exception()
    private class BreakSignal(val label: String?, val value: SigilValue?) : Exception()
    private class ContinueSignal(val label: String?) : Exception()

    /**
     * Execute a module
     */
    suspend fun execute(module: Module): ExecutionResult {
        output.clear()
        scopes.clear()
        functions.clear()
        structs.clear()

        pushScope()
        registerBuiltins()

        // Register all functions and structs
        for (item in module.items) {
            when (item) {
                is Item.Function -> functions[item.name] = item
                is Item.Struct -> structs[item.name] = item
                else -> { }
            }
        }

        // Find and execute main
        val main = functions["main"]
        if (main == null) {
            return ExecutionResult(
                success = false,
                output = "",
                error = "No main function found"
            )
        }

        return try {
            executeFunction(main, emptyList())
            ExecutionResult(success = true, output = output.toString())
        } catch (e: RuntimeException) {
            ExecutionResult(
                success = false,
                output = output.toString(),
                error = e.message ?: "Runtime error"
            )
        } finally {
            popScope()
        }
    }

    private suspend fun executeFunction(func: Item.Function, args: List<SigilValue>): SigilValue {
        pushScope()

        // Bind parameters
        for ((i, param) in func.params.withIndex()) {
            val name = (param.pattern as? Pattern.Ident)?.name ?: continue
            val value = args.getOrElse(i) { SigilValue.Unit }
            define(name, value)
        }

        val result = try {
            executeBlock(func.body)
        } catch (e: ReturnSignal) {
            e.value
        }

        popScope()
        return result
    }

    private suspend fun executeBlock(block: Block): SigilValue {
        var lastValue: SigilValue = SigilValue.Unit

        for (stmt in block.stmts) {
            lastValue = executeStmt(stmt)
        }

        return lastValue
    }

    private suspend fun executeStmt(stmt: Stmt): SigilValue {
        return when (stmt) {
            is Stmt.Let -> {
                val value = stmt.init?.let { executeExpr(it) } ?: SigilValue.Unit
                val name = (stmt.pattern as? Pattern.Ident)?.name
                if (name != null) {
                    define(name, value)
                }
                SigilValue.Unit
            }
            is Stmt.Expr -> {
                val value = executeExpr(stmt.expr)
                if (stmt.hasSemi) SigilValue.Unit else value
            }
            is Stmt.Item -> {
                // Register nested functions
                when (val item = stmt.item) {
                    is Item.Function -> functions[item.name] = item
                    else -> { }
                }
                SigilValue.Unit
            }
        }
    }

    private suspend fun executeExpr(expr: Expr): SigilValue {
        return when (expr) {
            is Expr.Literal -> executeLiteral(expr)
            is Expr.Path -> executePath(expr)
            is Expr.Binary -> executeBinary(expr)
            is Expr.Unary -> executeUnary(expr)
            is Expr.Call -> executeCall(expr)
            is Expr.MethodCall -> executeMethodCall(expr)
            is Expr.Field -> executeField(expr)
            is Expr.Index -> executeIndex(expr)
            is Expr.If -> executeIf(expr)
            is Expr.Match -> executeMatch(expr)
            is Expr.Block -> executeBlock(expr.block)
            is Expr.Loop -> executeLoop(expr)
            is Expr.While -> executeWhile(expr)
            is Expr.For -> executeFor(expr)
            is Expr.Return -> {
                val value = expr.value?.let { executeExpr(it) } ?: SigilValue.Unit
                throw ReturnSignal(value)
            }
            is Expr.Break -> {
                val value = expr.value?.let { executeExpr(it) }
                throw BreakSignal(expr.label, value)
            }
            is Expr.Continue -> throw ContinueSignal(expr.label)
            is Expr.Lambda -> executeLambda(expr)
            is Expr.Struct -> executeStructLiteral(expr)
            is Expr.Array -> executeArray(expr)
            is Expr.Tuple -> executeTuple(expr)
            is Expr.Range -> executeRange(expr)
            is Expr.Morpheme -> executeMorpheme(expr)
            is Expr.Pipe -> executePipe(expr)
            is Expr.Await -> executeExpr(expr.expr)
            is Expr.Try -> executeExpr(expr.expr)
            is Expr.Cast -> executeExpr(expr.expr)
        }
    }

    private fun executeLiteral(expr: Expr.Literal): SigilValue {
        return when (val kind = expr.kind) {
            is LiteralKind.Int -> SigilValue.Int(kind.value, expr.evidence)
            is LiteralKind.Float -> SigilValue.Float(kind.value, expr.evidence)
            is LiteralKind.String -> SigilValue.Str(kind.value, expr.evidence)
            is LiteralKind.Char -> SigilValue.Char(kind.value, expr.evidence)
            is LiteralKind.Bool -> SigilValue.Bool(kind.value, expr.evidence)
            is LiteralKind.Unit -> SigilValue.Unit
        }
    }

    private fun executePath(expr: Expr.Path): SigilValue {
        val name = expr.path.segments.joinToString("::")
        return lookup(name) ?: throw RuntimeException("Undefined variable: $name")
    }

    private suspend fun executeBinary(expr: Expr.Binary): SigilValue {
        // Handle short-circuit operators first
        if (expr.op == BinaryOp.And) {
            val left = executeExpr(expr.left)
            if (left.asBool() == false) return sigilBool(false, left.evidence)
            val right = executeExpr(expr.right)
            return sigilBool(right.asBool(), EvidenceLevel.join(left.evidence, right.evidence))
        }
        if (expr.op == BinaryOp.Or) {
            val left = executeExpr(expr.left)
            if (left.asBool() == true) return sigilBool(true, left.evidence)
            val right = executeExpr(expr.right)
            return sigilBool(right.asBool(), EvidenceLevel.join(left.evidence, right.evidence))
        }

        val left = executeExpr(expr.left)
        val right = executeExpr(expr.right)
        val evidence = EvidenceLevel.join(left.evidence, right.evidence)

        return when (expr.op) {
            BinaryOp.Add -> binaryNumeric(left, right, Long::plus, Double::plus, evidence)
            BinaryOp.Sub -> binaryNumeric(left, right, Long::minus, Double::minus, evidence)
            BinaryOp.Mul -> binaryNumeric(left, right, Long::times, Double::times, evidence)
            BinaryOp.Div -> binaryNumeric(left, right, Long::div, Double::div, evidence)
            BinaryOp.Rem -> binaryNumeric(left, right, Long::rem, Double::rem, evidence)
            BinaryOp.Pow -> {
                val base = toDouble(left)
                val exp = toDouble(right)
                sigilFloat(kotlin.math.pow(base, exp), evidence)
            }
            BinaryOp.Eq -> sigilBool(equals(left, right), evidence)
            BinaryOp.Ne -> sigilBool(!equals(left, right), evidence)
            BinaryOp.Lt -> sigilBool(compare(left, right) < 0, evidence)
            BinaryOp.Le -> sigilBool(compare(left, right) <= 0, evidence)
            BinaryOp.Gt -> sigilBool(compare(left, right) > 0, evidence)
            BinaryOp.Ge -> sigilBool(compare(left, right) >= 0, evidence)
            BinaryOp.BitAnd -> sigilInt(left.asInt() and right.asInt(), evidence)
            BinaryOp.BitOr -> sigilInt(left.asInt() or right.asInt(), evidence)
            BinaryOp.BitXor -> sigilInt(left.asInt() xor right.asInt(), evidence)
            BinaryOp.Shl -> sigilInt(left.asInt() shl right.asInt().toInt(), evidence)
            BinaryOp.Shr -> sigilInt(left.asInt() shr right.asInt().toInt(), evidence)
            BinaryOp.Assign -> {
                // Assignment - update variable
                val name = (expr.left as? Expr.Path)?.path?.segments?.joinToString("::")
                if (name != null) {
                    assign(name, right)
                }
                SigilValue.Unit
            }
            BinaryOp.AddAssign, BinaryOp.SubAssign, BinaryOp.MulAssign,
            BinaryOp.DivAssign, BinaryOp.RemAssign -> {
                val name = (expr.left as? Expr.Path)?.path?.segments?.joinToString("::")
                if (name != null) {
                    val current = lookup(name) ?: throw RuntimeException("Undefined variable: $name")
                    val newValue = when (expr.op) {
                        BinaryOp.AddAssign -> binaryNumeric(current, right, Long::plus, Double::plus, evidence)
                        BinaryOp.SubAssign -> binaryNumeric(current, right, Long::minus, Double::minus, evidence)
                        BinaryOp.MulAssign -> binaryNumeric(current, right, Long::times, Double::times, evidence)
                        BinaryOp.DivAssign -> binaryNumeric(current, right, Long::div, Double::div, evidence)
                        BinaryOp.RemAssign -> binaryNumeric(current, right, Long::rem, Double::rem, evidence)
                        else -> current
                    }
                    assign(name, newValue)
                }
                SigilValue.Unit
            }
            BinaryOp.And, BinaryOp.Or -> SigilValue.Unit // Handled above
        }
    }

    private fun executeUnary(expr: Expr.Unary): SigilValue {
        val operand = runBlocking { executeExpr(expr.operand) }

        return when (expr.op) {
            UnaryOp.Neg -> when (operand) {
                is SigilValue.Int -> sigilInt(-operand.value, operand.evidence)
                is SigilValue.Float -> sigilFloat(-operand.value, operand.evidence)
                else -> throw RuntimeException("Cannot negate non-numeric value")
            }
            UnaryOp.Not -> sigilBool(!operand.asBool(), operand.evidence)
            UnaryOp.Deref -> operand
            UnaryOp.Ref, UnaryOp.RefMut -> operand
        }
    }

    private suspend fun executeCall(expr: Expr.Call): SigilValue {
        // Check if it's a builtin
        val calleePath = (expr.callee as? Expr.Path)?.path?.segments?.joinToString("::")

        if (calleePath != null) {
            val args = expr.args.map { executeExpr(it) }
            val builtin = executeBuiltin(calleePath, args)
            if (builtin != null) return builtin

            // User-defined function
            val func = functions[calleePath]
            if (func != null) {
                return executeFunction(func, args)
            }
        }

        // Lambda call
        val callee = executeExpr(expr.callee)
        if (callee is SigilValue.Function || callee is SigilValue.Closure) {
            val args = expr.args.map { executeExpr(it) }
            val body = when (callee) {
                is SigilValue.Function -> callee.body
                is SigilValue.Closure -> callee.body
                else -> throw RuntimeException("Not callable")
            }
            return body(args)
        }

        throw RuntimeException("Cannot call non-function")
    }

    private suspend fun executeMethodCall(expr: Expr.MethodCall): SigilValue {
        val receiver = executeExpr(expr.receiver)
        val args = expr.args.map { executeExpr(it) }

        // String methods
        if (receiver is SigilValue.Str) {
            return when (expr.method) {
                "len" -> sigilInt(receiver.value.length.toLong(), receiver.evidence)
                "to_uppercase" -> SigilValue.Str(receiver.value.uppercase(), receiver.evidence)
                "to_lowercase" -> SigilValue.Str(receiver.value.lowercase(), receiver.evidence)
                "trim" -> SigilValue.Str(receiver.value.trim(), receiver.evidence)
                "contains" -> sigilBool(receiver.value.contains(args[0].asStr()), receiver.evidence)
                "starts_with" -> sigilBool(receiver.value.startsWith(args[0].asStr()), receiver.evidence)
                "ends_with" -> sigilBool(receiver.value.endsWith(args[0].asStr()), receiver.evidence)
                "split" -> sigilList(
                    receiver.value.split(args[0].asStr()).map { SigilValue.Str(it, receiver.evidence) },
                    receiver.evidence
                )
                else -> throw RuntimeException("Unknown method: ${expr.method}")
            }
        }

        // List methods
        if (receiver is SigilValue.List) {
            return when (expr.method) {
                "len" -> sigilInt(receiver.elements.size.toLong(), receiver.evidence)
                "is_empty" -> sigilBool(receiver.elements.isEmpty(), receiver.evidence)
                "first" -> receiver.elements.firstOrNull()?.let { sigilSome(it) } ?: sigilNone()
                "last" -> receiver.elements.lastOrNull()?.let { sigilSome(it) } ?: sigilNone()
                "push" -> {
                    val newElements = receiver.elements.toMutableList()
                    newElements.add(args[0])
                    sigilList(newElements, EvidenceLevel.join(receiver.evidence, args[0].evidence))
                }
                "pop" -> {
                    if (receiver.elements.isEmpty()) sigilNone()
                    else sigilSome(receiver.elements.last())
                }
                "reverse" -> sigilList(receiver.elements.reversed(), receiver.evidence)
                else -> throw RuntimeException("Unknown method: ${expr.method}")
            }
        }

        throw RuntimeException("No method '${expr.method}' on type")
    }

    private suspend fun executeField(expr: Expr.Field): SigilValue {
        val receiver = executeExpr(expr.receiver)

        return when (receiver) {
            is SigilValue.Struct -> {
                receiver.fields[expr.field]
                    ?: throw RuntimeException("Unknown field: ${expr.field}")
            }
            is SigilValue.Tuple -> {
                val index = expr.field.toIntOrNull()
                    ?: throw RuntimeException("Invalid tuple index: ${expr.field}")
                receiver.elements.getOrNull(index)
                    ?: throw RuntimeException("Tuple index out of bounds: $index")
            }
            else -> throw RuntimeException("Cannot access field on this type")
        }
    }

    private suspend fun executeIndex(expr: Expr.Index): SigilValue {
        val receiver = executeExpr(expr.receiver)
        val index = executeExpr(expr.index)

        return when (receiver) {
            is SigilValue.List -> {
                val i = index.asInt().toInt()
                receiver.elements.getOrNull(i)
                    ?: throw RuntimeException("Index out of bounds: $i")
            }
            is SigilValue.Str -> {
                val i = index.asInt().toInt()
                SigilValue.Char(
                    receiver.value.getOrNull(i) ?: throw RuntimeException("Index out of bounds: $i"),
                    receiver.evidence
                )
            }
            is SigilValue.Map -> {
                receiver.entries[index]
                    ?: throw RuntimeException("Key not found")
            }
            else -> throw RuntimeException("Cannot index this type")
        }
    }

    private suspend fun executeIf(expr: Expr.If): SigilValue {
        val condition = executeExpr(expr.condition)

        return if (condition.asBool()) {
            executeBlock(expr.thenBranch)
        } else {
            expr.elseBranch?.let { executeBlock(it) } ?: SigilValue.Unit
        }
    }

    private suspend fun executeMatch(expr: Expr.Match): SigilValue {
        val scrutinee = executeExpr(expr.scrutinee)

        for (arm in expr.arms) {
            if (matchPattern(arm.pattern, scrutinee)) {
                // Check guard
                if (arm.guard != null) {
                    val guardResult = executeExpr(arm.guard)
                    if (!guardResult.asBool()) continue
                }
                return executeExpr(arm.body)
            }
        }

        throw RuntimeException("No matching arm in match expression")
    }

    private fun matchPattern(pattern: Pattern, value: SigilValue): Boolean {
        return when (pattern) {
            is Pattern.Wildcard -> true
            is Pattern.Ident -> {
                define(pattern.name, value)
                true
            }
            is Pattern.Literal -> when (val kind = pattern.kind) {
                is LiteralKind.Int -> value is SigilValue.Int && value.value == kind.value
                is LiteralKind.Float -> value is SigilValue.Float && value.value == kind.value
                is LiteralKind.String -> value is SigilValue.Str && value.value == kind.value
                is LiteralKind.Bool -> value is SigilValue.Bool && value.value == kind.value
                else -> false
            }
            is Pattern.Tuple -> {
                if (value !is SigilValue.Tuple) return false
                if (pattern.elements.size != value.elements.size) return false
                pattern.elements.zip(value.elements).all { (p, v) -> matchPattern(p, v) }
            }
            is Pattern.Or -> pattern.patterns.any { matchPattern(it, value) }
            else -> false
        }
    }

    private suspend fun executeLoop(expr: Expr.Loop): SigilValue {
        while (true) {
            try {
                executeBlock(expr.body)
            } catch (e: BreakSignal) {
                if (e.label == null || e.label == expr.label) {
                    return e.value ?: SigilValue.Unit
                }
                throw e
            } catch (e: ContinueSignal) {
                if (e.label == null || e.label == expr.label) {
                    continue
                }
                throw e
            }
        }
    }

    private suspend fun executeWhile(expr: Expr.While): SigilValue {
        while (true) {
            val condition = executeExpr(expr.condition)
            if (!condition.asBool()) break

            try {
                executeBlock(expr.body)
            } catch (e: BreakSignal) {
                if (e.label == null || e.label == expr.label) {
                    return e.value ?: SigilValue.Unit
                }
                throw e
            } catch (e: ContinueSignal) {
                if (e.label == null || e.label == expr.label) {
                    continue
                }
                throw e
            }
        }
        return SigilValue.Unit
    }

    private suspend fun executeFor(expr: Expr.For): SigilValue {
        val iterable = executeExpr(expr.iterable)

        val elements = when (iterable) {
            is SigilValue.List -> iterable.elements
            is SigilValue.Str -> iterable.value.map { SigilValue.Char(it, iterable.evidence) }
            else -> throw RuntimeException("Cannot iterate over this type")
        }

        for (element in elements) {
            pushScope()
            val name = (expr.pattern as? Pattern.Ident)?.name
            if (name != null) {
                define(name, element)
            }

            try {
                executeBlock(expr.body)
            } catch (e: BreakSignal) {
                popScope()
                if (e.label == null || e.label == expr.label) {
                    return e.value ?: SigilValue.Unit
                }
                throw e
            } catch (e: ContinueSignal) {
                popScope()
                if (e.label == null || e.label == expr.label) {
                    continue
                }
                throw e
            }

            popScope()
        }

        return SigilValue.Unit
    }

    private fun executeLambda(expr: Expr.Lambda): SigilValue {
        val capturedScope = scopes.flatMap { it.entries }.associate { it.key to it.value }

        return SigilValue.Closure(
            params = expr.params.mapNotNull { (it.pattern as? Pattern.Ident)?.name },
            captures = capturedScope,
            body = { args ->
                pushScope()
                // Restore captures
                capturedScope.forEach { (k, v) -> define(k, v) }
                // Bind params
                for ((i, param) in expr.params.withIndex()) {
                    val name = (param.pattern as? Pattern.Ident)?.name ?: continue
                    define(name, args.getOrElse(i) { SigilValue.Unit })
                }
                val result = executeExpr(expr.body)
                popScope()
                result
            }
        )
    }

    private suspend fun executeStructLiteral(expr: Expr.Struct): SigilValue {
        val name = expr.path.segments.joinToString("::") { it.name }
        val fields = mutableMapOf<String, SigilValue>()
        var evidence = EvidenceLevel.KNOWN

        for (field in expr.fields) {
            val value = executeExpr(field.value)
            fields[field.name] = value
            evidence = EvidenceLevel.join(evidence, value.evidence)
        }

        return SigilValue.Struct(name, fields, evidence)
    }

    private suspend fun executeArray(expr: Expr.Array): SigilValue {
        val elements = expr.elements.map { executeExpr(it) }
        val evidence = elements.fold(EvidenceLevel.KNOWN) { acc, v ->
            EvidenceLevel.join(acc, v.evidence)
        }
        return SigilValue.List(elements, evidence)
    }

    private suspend fun executeTuple(expr: Expr.Tuple): SigilValue {
        val elements = expr.elements.map { executeExpr(it) }
        val evidence = elements.fold(EvidenceLevel.KNOWN) { acc, v ->
            EvidenceLevel.join(acc, v.evidence)
        }
        return SigilValue.Tuple(elements, evidence)
    }

    private suspend fun executeRange(expr: Expr.Range): SigilValue {
        val start = expr.start?.let { executeExpr(it) }?.asInt()?.toInt() ?: 0
        val end = expr.end?.let { executeExpr(it) }?.asInt()?.toInt() ?: 0
        val range = if (expr.inclusive) start..end else start until end
        return sigilList(range.map { sigilInt(it.toLong()) }, EvidenceLevel.KNOWN)
    }

    private suspend fun executeMorpheme(expr: Expr.Morpheme): SigilValue {
        val input = executeExpr(expr.input)
        val transform = expr.transform?.let { executeExpr(it) }

        return when (expr.op) {
            MorphemeOp.Tau -> {
                // Transform/Map
                val list = input.asList()
                val closure = transform as? SigilValue.Closure
                    ?: transform as? SigilValue.Function
                    ?: throw RuntimeException("τ requires a function")

                val mapped = list.map { elem ->
                    when (closure) {
                        is SigilValue.Closure -> closure.body(listOf(elem))
                        is SigilValue.Function -> closure.body(listOf(elem))
                        else -> elem
                    }
                }
                sigilList(mapped, input.evidence)
            }
            MorphemeOp.Phi -> {
                // Filter
                val list = input.asList()
                val closure = transform as? SigilValue.Closure
                    ?: transform as? SigilValue.Function
                    ?: throw RuntimeException("φ requires a predicate")

                val filtered = list.filter { elem ->
                    val result = when (closure) {
                        is SigilValue.Closure -> closure.body(listOf(elem))
                        is SigilValue.Function -> closure.body(listOf(elem))
                        else -> sigilBool(true)
                    }
                    result.asBool()
                }
                sigilList(filtered, input.evidence)
            }
            MorphemeOp.Sigma -> {
                // Sort
                val list = input.asList()
                val sorted = list.sortedWith { a, b -> compare(a, b) }
                sigilList(sorted, input.evidence)
            }
            MorphemeOp.Rho -> {
                // Reduce
                val list = input.asList()
                if (list.isEmpty()) return SigilValue.Unit

                val closure = transform as? SigilValue.Closure
                    ?: transform as? SigilValue.Function

                if (closure != null) {
                    list.reduce { acc, elem ->
                        when (closure) {
                            is SigilValue.Closure -> closure.body(listOf(acc, elem))
                            is SigilValue.Function -> closure.body(listOf(acc, elem))
                            else -> acc
                        }
                    }
                } else {
                    list.first()
                }
            }
            MorphemeOp.Sum -> {
                val list = input.asList()
                val sum = list.sumOf { toDouble(it) }
                sigilFloat(sum, input.evidence)
            }
            MorphemeOp.Product -> {
                val list = input.asList()
                val product = list.fold(1.0) { acc, v -> acc * toDouble(v) }
                sigilFloat(product, input.evidence)
            }
            MorphemeOp.Alpha -> {
                val list = input.asList()
                list.firstOrNull()?.let { sigilSome(it) } ?: sigilNone()
            }
            MorphemeOp.Omega -> {
                val list = input.asList()
                list.lastOrNull()?.let { sigilSome(it) } ?: sigilNone()
            }
            MorphemeOp.Delta -> {
                val list = input.asList()
                val diffs = list.zipWithNext { a, b ->
                    binaryNumeric(b, a, Long::minus, Double::minus, input.evidence)
                }
                sigilList(diffs, input.evidence)
            }
            MorphemeOp.Gamma -> {
                // Group by
                val list = input.asList()
                val closure = transform as? SigilValue.Closure
                    ?: transform as? SigilValue.Function
                    ?: throw RuntimeException("Γ requires a key function")

                val grouped = list.groupBy { elem ->
                    when (closure) {
                        is SigilValue.Closure -> closure.body(listOf(elem))
                        is SigilValue.Function -> closure.body(listOf(elem))
                        else -> elem
                    }
                }
                sigilMap(grouped.mapValues { sigilList(it.value) }, input.evidence)
            }
            MorphemeOp.Lambda -> {
                // Just return the transform as a function
                transform ?: SigilValue.Unit
            }
        }
    }

    private suspend fun executePipe(expr: Expr.Pipe): SigilValue {
        val left = executeExpr(expr.left)
        val right = executeExpr(expr.right)

        return when (right) {
            is SigilValue.Function -> right.body(listOf(left))
            is SigilValue.Closure -> right.body(listOf(left))
            else -> throw RuntimeException("Pipe target must be callable")
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Builtins
    // ═══════════════════════════════════════════════════════════════════════

    private fun registerBuiltins() {
        // Will be filled with stdlib
    }

    private fun executeBuiltin(name: String, args: List<SigilValue>): SigilValue? {
        return when (name) {
            "print" -> {
                val text = args.joinToString(" ") { valueToString(it) }
                output.appendLine(text)
                SigilValue.Unit
            }
            "println" -> {
                val text = args.joinToString(" ") { valueToString(it) }
                output.appendLine(text)
                SigilValue.Unit
            }
            "str" -> {
                val evidence = args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN
                SigilValue.Str(valueToString(args.firstOrNull() ?: SigilValue.Unit), evidence)
            }
            "int" -> {
                val arg = args.firstOrNull() ?: return sigilInt(0)
                when (arg) {
                    is SigilValue.Int -> arg
                    is SigilValue.Float -> sigilInt(arg.value.toLong(), arg.evidence)
                    is SigilValue.Str -> sigilInt(arg.value.toLongOrNull() ?: 0, arg.evidence)
                    else -> sigilInt(0)
                }
            }
            "float" -> {
                val arg = args.firstOrNull() ?: return sigilFloat(0.0)
                when (arg) {
                    is SigilValue.Float -> arg
                    is SigilValue.Int -> sigilFloat(arg.value.toDouble(), arg.evidence)
                    is SigilValue.Str -> sigilFloat(arg.value.toDoubleOrNull() ?: 0.0, arg.evidence)
                    else -> sigilFloat(0.0)
                }
            }
            "len" -> {
                val arg = args.firstOrNull() ?: return sigilInt(0)
                when (arg) {
                    is SigilValue.Str -> sigilInt(arg.value.length.toLong(), arg.evidence)
                    is SigilValue.List -> sigilInt(arg.elements.size.toLong(), arg.evidence)
                    else -> sigilInt(0)
                }
            }
            "abs" -> {
                val arg = args.firstOrNull() ?: return sigilFloat(0.0)
                when (arg) {
                    is SigilValue.Int -> sigilInt(kotlin.math.abs(arg.value), arg.evidence)
                    is SigilValue.Float -> sigilFloat(kotlin.math.abs(arg.value), arg.evidence)
                    else -> sigilFloat(0.0)
                }
            }
            "sqrt" -> sigilFloat(kotlin.math.sqrt(toDouble(args.firstOrNull())), args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN)
            "sin" -> sigilFloat(kotlin.math.sin(toDouble(args.firstOrNull())), args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN)
            "cos" -> sigilFloat(kotlin.math.cos(toDouble(args.firstOrNull())), args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN)
            "tan" -> sigilFloat(kotlin.math.tan(toDouble(args.firstOrNull())), args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN)
            "floor" -> sigilFloat(kotlin.math.floor(toDouble(args.firstOrNull())), args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN)
            "ceil" -> sigilFloat(kotlin.math.ceil(toDouble(args.firstOrNull())), args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN)
            "round" -> sigilFloat(kotlin.math.round(toDouble(args.firstOrNull())), args.firstOrNull()?.evidence ?: EvidenceLevel.KNOWN)
            "min" -> {
                val a = toDouble(args.getOrNull(0))
                val b = toDouble(args.getOrNull(1))
                sigilFloat(kotlin.math.min(a, b))
            }
            "max" -> {
                val a = toDouble(args.getOrNull(0))
                val b = toDouble(args.getOrNull(1))
                sigilFloat(kotlin.math.max(a, b))
            }
            "range" -> {
                val start = args.getOrNull(0)?.asInt()?.toInt() ?: 0
                val end = args.getOrNull(1)?.asInt()?.toInt() ?: 0
                sigilList((start until end).map { sigilInt(it.toLong()) })
            }
            "type_of" -> {
                val arg = args.firstOrNull() ?: return SigilValue.Str("unit", EvidenceLevel.KNOWN)
                SigilValue.Str(arg.typeName(), EvidenceLevel.KNOWN)
            }
            "evidence_of" -> {
                val arg = args.firstOrNull() ?: return SigilValue.Str("!", EvidenceLevel.KNOWN)
                SigilValue.Str(arg.evidence.symbol.toString(), EvidenceLevel.KNOWN)
            }
            else -> null
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private fun valueToString(value: SigilValue): String = when (value) {
        is SigilValue.Int -> value.value.toString()
        is SigilValue.Float -> value.value.toString()
        is SigilValue.Bool -> value.value.toString()
        is SigilValue.Str -> value.value
        is SigilValue.Char -> value.value.toString()
        is SigilValue.List -> "[${value.elements.joinToString(", ") { valueToString(it) }}]"
        is SigilValue.Tuple -> "(${value.elements.joinToString(", ") { valueToString(it) }})"
        is SigilValue.Struct -> "${value.name} { ${value.fields.entries.joinToString(", ") { "${it.key}: ${valueToString(it.value)}" }} }"
        is SigilValue.Option -> if (value.value != null) "Some(${valueToString(value.value)})" else "None"
        is SigilValue.Result -> if (value.isOk) "Ok(${valueToString(value.value!!)})" else "Err(${value.error})"
        is SigilValue.Unit -> "()"
        else -> value.toString()
    }

    private fun binaryNumeric(
        left: SigilValue,
        right: SigilValue,
        intOp: (Long, Long) -> Long,
        floatOp: (Double, Double) -> Double,
        evidence: EvidenceLevel
    ): SigilValue {
        return when {
            left is SigilValue.Float || right is SigilValue.Float ->
                sigilFloat(floatOp(toDouble(left), toDouble(right)), evidence)
            else -> sigilInt(intOp(left.asInt(), right.asInt()), evidence)
        }
    }

    private fun toDouble(value: SigilValue?): Double = when (value) {
        is SigilValue.Int -> value.value.toDouble()
        is SigilValue.Float -> value.value
        else -> 0.0
    }

    private fun equals(a: SigilValue, b: SigilValue): Boolean = when {
        a is SigilValue.Int && b is SigilValue.Int -> a.value == b.value
        a is SigilValue.Float && b is SigilValue.Float -> a.value == b.value
        a is SigilValue.Bool && b is SigilValue.Bool -> a.value == b.value
        a is SigilValue.Str && b is SigilValue.Str -> a.value == b.value
        a is SigilValue.Char && b is SigilValue.Char -> a.value == b.value
        else -> a == b
    }

    private fun compare(a: SigilValue, b: SigilValue): Int = when {
        a is SigilValue.Int && b is SigilValue.Int -> a.value.compareTo(b.value)
        a is SigilValue.Float && b is SigilValue.Float -> a.value.compareTo(b.value)
        a is SigilValue.Str && b is SigilValue.Str -> a.value.compareTo(b.value)
        a is SigilValue.Float || b is SigilValue.Float -> toDouble(a).compareTo(toDouble(b))
        else -> 0
    }

    private fun pushScope() {
        scopes.add(mutableMapOf())
    }

    private fun popScope() {
        scopes.removeAt(scopes.lastIndex)
    }

    private fun define(name: String, value: SigilValue) {
        scopes.last()[name] = value
    }

    private fun assign(name: String, value: SigilValue) {
        for (scope in scopes.asReversed()) {
            if (name in scope) {
                scope[name] = value
                return
            }
        }
        throw RuntimeException("Cannot assign to undefined variable: $name")
    }

    private fun lookup(name: String): SigilValue? {
        for (scope in scopes.asReversed()) {
            scope[name]?.let { return it }
        }
        return null
    }
}

// Coroutine helper for non-suspend contexts
private fun <T> runBlocking(block: suspend () -> T): T {
    var result: T? = null
    var exception: Throwable? = null

    // Simple blocking implementation for common cases
    kotlinx.coroutines.runBlocking {
        try {
            result = block()
        } catch (e: Throwable) {
            exception = e
        }
    }

    exception?.let { throw it }
    @Suppress("UNCHECKED_CAST")
    return result as T
}

data class ExecutionResult(
    val success: Boolean,
    val output: String,
    val error: String? = null
)

package com.daemoniorum.sigil

import com.daemoniorum.sigil.ast.Module
import com.daemoniorum.sigil.interpreter.ExecutionResult
import com.daemoniorum.sigil.interpreter.Interpreter
import com.daemoniorum.sigil.lexer.Lexer
import com.daemoniorum.sigil.lexer.LexerResult
import com.daemoniorum.sigil.parser.ParseResult
import com.daemoniorum.sigil.parser.Parser
import com.daemoniorum.sigil.typeck.TypeCheckResult
import com.daemoniorum.sigil.typeck.TypeChecker
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Main entry point for the Sigil language.
 *
 * Sigil is a polysynthetic programming language with epistemic types
 * designed for AI systems.
 *
 * ```kotlin
 * val result = Sigil.run("""
 *     fn main() {
 *         let x! = 42;
 *         print(x);
 *     }
 * """)
 *
 * println(result.output) // "42"
 * ```
 */
object Sigil {
    private val json = Json {
        prettyPrint = true
        encodeDefaults = true
    }

    /**
     * Run Sigil code and return the result
     */
    suspend fun run(source: String): ExecutionResult {
        // Lex
        val lexResult = lex(source)
        if (lexResult.hasErrors) {
            return ExecutionResult(
                success = false,
                output = "",
                error = "Lexer errors:\n" + lexResult.errors.joinToString("\n") {
                    "  ${it.span.line}:${it.span.column}: ${it.message}"
                }
            )
        }

        // Parse
        val parseResult = parse(lexResult)
        if (parseResult.hasErrors) {
            return ExecutionResult(
                success = false,
                output = "",
                error = "Parse errors:\n" + parseResult.errors.joinToString("\n") {
                    "  ${it.span.line}:${it.span.column}: ${it.message}"
                }
            )
        }

        // Type check
        val typeResult = check(parseResult.module)
        if (typeResult.hasErrors) {
            return ExecutionResult(
                success = false,
                output = "",
                error = "Type errors:\n" + typeResult.errors.joinToString("\n") {
                    "  ${it.span.line}:${it.span.column}: ${it.message}"
                }
            )
        }

        // Execute
        return execute(parseResult.module)
    }

    /**
     * Tokenize source code
     */
    fun lex(source: String): LexerResult {
        return Lexer(source).tokenize()
    }

    /**
     * Parse tokens into AST
     */
    fun parse(source: String): ParseResult {
        val lexResult = lex(source)
        return Parser(lexResult.tokens).parse()
    }

    /**
     * Parse from lex result
     */
    fun parse(lexResult: LexerResult): ParseResult {
        return Parser(lexResult.tokens).parse()
    }

    /**
     * Type check a module
     */
    fun check(module: Module): TypeCheckResult {
        return TypeChecker().check(module)
    }

    /**
     * Type check source code
     */
    fun check(source: String): TypeCheckResult {
        val parseResult = parse(source)
        if (parseResult.hasErrors) {
            return TypeCheckResult(emptyList())
        }
        return check(parseResult.module)
    }

    /**
     * Execute a parsed module
     */
    suspend fun execute(module: Module): ExecutionResult {
        return Interpreter().execute(module)
    }

    /**
     * Get the AST as JSON (for AI consumption)
     */
    fun toJson(source: String): String {
        val parseResult = parse(source)
        return json.encodeToString(parseResult.module)
    }

    /**
     * Compile information for display
     */
    fun info(): SigilInfo = SigilInfo(
        version = VERSION,
        targets = listOf("jvm", "android", "ios", "macos", "linux", "windows", "js", "wasm"),
        features = listOf(
            "Epistemic type system (!, ?, ~, ‽)",
            "Morpheme operators (τ, φ, σ, ρ, Σ, Π, α, ω)",
            "Pattern matching",
            "Algebraic data types",
            "Async/await",
            "First-class functions"
        )
    )

    const val VERSION = "0.1.0"
}

data class SigilInfo(
    val version: String,
    val targets: List<String>,
    val features: List<String>
)

// ═══════════════════════════════════════════════════════════════════════════
// Convenience extensions for running Sigil from Kotlin
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Run Sigil code (blocking version for non-coroutine contexts)
 */
fun Sigil.runBlocking(source: String): ExecutionResult {
    return kotlinx.coroutines.runBlocking { run(source) }
}

/**
 * Quick evaluation of a Sigil expression
 */
suspend fun Sigil.eval(expression: String): ExecutionResult {
    val wrapped = """
        fn main() {
            let result = $expression;
            print(result);
        }
    """.trimIndent()
    return run(wrapped)
}

/**
 * REPL-style interaction
 */
class SigilRepl {
    private val history = mutableListOf<String>()
    private var context = mutableMapOf<String, String>()

    suspend fun execute(line: String): String {
        history.add(line)

        // Check if it's a declaration or expression
        val source = if (line.trimStart().startsWith("let ") ||
                        line.trimStart().startsWith("fn ")) {
            // Wrap in main
            """
                fn main() {
                    $line
                }
            """.trimIndent()
        } else {
            // Treat as expression to print
            """
                fn main() {
                    let __result = $line;
                    print(__result);
                }
            """.trimIndent()
        }

        val result = Sigil.run(source)
        return if (result.success) {
            result.output.trim()
        } else {
            "Error: ${result.error}"
        }
    }

    fun history(): List<String> = history.toList()
}

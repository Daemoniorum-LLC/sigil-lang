package com.daemoniorum.sigil.runtime

/**
 * Morpheme operators for Sigil on Kotlin.
 *
 * These preserve the expressive power and feel of Sigil's morpheme system
 * while being natural to use from Kotlin code.
 *
 * The key insight: morphemes are evidence-aware transformations.
 * Every operation tracks and propagates provenance.
 */

// ═══════════════════════════════════════════════════════════════════════════
// τ (tau) - Transform/Map
// ═══════════════════════════════════════════════════════════════════════════

/**
 * τ - Transform each element, preserving evidence
 */
inline infix fun <T, R> Evidence<List<T>>.τ(transform: (T) -> R): Evidence<List<R>> =
    map { list -> list.map(transform) }

/**
 * τ for single values
 */
inline infix fun <T, R> Evidence<T>.τ(transform: (T) -> R): Evidence<R> =
    map(transform)

// ═══════════════════════════════════════════════════════════════════════════
// φ (phi) - Filter
// ═══════════════════════════════════════════════════════════════════════════

/**
 * φ - Filter elements, preserving evidence
 */
inline infix fun <T> Evidence<List<T>>.φ(predicate: (T) -> Boolean): Evidence<List<T>> =
    map { list -> list.filter(predicate) }

// ═══════════════════════════════════════════════════════════════════════════
// σ (sigma) - Sort
// ═══════════════════════════════════════════════════════════════════════════

/**
 * σ - Sort elements naturally
 */
fun <T : Comparable<T>> Evidence<List<T>>.σ(): Evidence<List<T>> =
    map { list -> list.sorted() }

/**
 * σ - Sort by selector
 */
inline fun <T, R : Comparable<R>> Evidence<List<T>>.σ(crossinline selector: (T) -> R): Evidence<List<T>> =
    map { list -> list.sortedBy(selector) }

// ═══════════════════════════════════════════════════════════════════════════
// ρ (rho) - Reduce/Fold
// ═══════════════════════════════════════════════════════════════════════════

/**
 * ρ - Reduce to single value
 */
inline fun <T, R> Evidence<List<T>>.ρ(initial: R, operation: (R, T) -> R): Evidence<R> =
    map { list -> list.fold(initial, operation) }

/**
 * ρ - Reduce without initial value
 */
inline fun <T> Evidence<List<T>>.ρ(operation: (T, T) -> T): Evidence<T?> =
    map { list -> list.reduceOrNull(operation) }

// ═══════════════════════════════════════════════════════════════════════════
// Σ (Sigma) - Sum
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Σ - Sum numeric values
 */
@JvmName("sumInt")
fun Evidence<List<Int>>.Σ(): Evidence<Int> = map { it.sum() }

@JvmName("sumLong")
fun Evidence<List<Long>>.Σ(): Evidence<Long> = map { it.sum() }

@JvmName("sumDouble")
fun Evidence<List<Double>>.Σ(): Evidence<Double> = map { it.sum() }

// ═══════════════════════════════════════════════════════════════════════════
// Π (Pi) - Product
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Π - Product of numeric values
 */
@JvmName("productInt")
fun Evidence<List<Int>>.Π(): Evidence<Long> =
    map { list -> list.fold(1L) { acc, n -> acc * n } }

@JvmName("productDouble")
fun Evidence<List<Double>>.Π(): Evidence<Double> =
    map { list -> list.fold(1.0) { acc, n -> acc * n } }

// ═══════════════════════════════════════════════════════════════════════════
// α (alpha) - First
// ═══════════════════════════════════════════════════════════════════════════

/**
 * α - Get first element
 */
fun <T> Evidence<List<T>>.α(): Evidence<T?> =
    map { list -> list.firstOrNull() }

/**
 * α - Get first matching element
 */
inline fun <T> Evidence<List<T>>.α(predicate: (T) -> Boolean): Evidence<T?> =
    map { list -> list.firstOrNull(predicate) }

// ═══════════════════════════════════════════════════════════════════════════
// ω (omega) - Last
// ═══════════════════════════════════════════════════════════════════════════

/**
 * ω - Get last element
 */
fun <T> Evidence<List<T>>.ω(): Evidence<T?> =
    map { list -> list.lastOrNull() }

/**
 * ω - Get last matching element
 */
inline fun <T> Evidence<List<T>>.ω(predicate: (T) -> Boolean): Evidence<T?> =
    map { list -> list.lastOrNull(predicate) }

// ═══════════════════════════════════════════════════════════════════════════
// Δ (Delta) - Difference/Change
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Δ - Compute pairwise differences
 */
@JvmName("deltaInt")
fun Evidence<List<Int>>.Δ(): Evidence<List<Int>> =
    map { list -> list.zipWithNext { a, b -> b - a } }

@JvmName("deltaDouble")
fun Evidence<List<Double>>.Δ(): Evidence<List<Double>> =
    map { list -> list.zipWithNext { a, b -> b - a } }

// ═══════════════════════════════════════════════════════════════════════════
// Γ (Gamma) - Collect/Group
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Γ - Group by key
 */
inline fun <T, K> Evidence<List<T>>.Γ(keySelector: (T) -> K): Evidence<Map<K, List<T>>> =
    map { list -> list.groupBy(keySelector) }

// ═══════════════════════════════════════════════════════════════════════════
// |> (Pipe) - Chaining operator
// ═══════════════════════════════════════════════════════════════════════════

/**
 * |> - Pipe operator for chaining
 */
inline infix fun <T, R> T.pipe(transform: (T) -> R): R = transform(this)

/**
 * |> - Evidence-aware pipe
 */
inline infix fun <T, R> Evidence<T>.`|>`(transform: (T) -> R): Evidence<R> = map(transform)

// ═══════════════════════════════════════════════════════════════════════════
// Evidence DSL for AI-natural code
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Evidence builder scope for natural AI code
 */
class EvidenceScope {
    /**
     * Mark a value as known (!)
     */
    fun <T> T.known(): Known<T> = Known(this)

    /**
     * Mark a value as uncertain (?)
     */
    fun <T> T.uncertain(): Uncertain<T> = Uncertain(this)

    /**
     * Mark a value as reported (~)
     */
    fun <T> T.reported(source: Source = Source.External): Reported<T> = Reported(this, source)

    /**
     * Mark a value as paradox (‽)
     */
    fun <T> T.paradox(reason: String): Paradox<T> = Paradox(this, reason)
}

/**
 * DSL entry point
 */
inline fun <T> evidence(block: EvidenceScope.() -> T): T = EvidenceScope().block()

// ═══════════════════════════════════════════════════════════════════════════
// AI-Friendly Extensions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Explain the evidence chain for debugging/transparency
 */
fun <T> Evidence<T>.explain(): String = when (this) {
    is Known -> "Known(!): Direct computation or verification. Value: $value"
    is Uncertain -> "Uncertain(?): Validated but may be absent. Value: $value"
    is Reported -> "Reported(~): External source '$source'. Value: $value"
    is Paradox -> "Paradox(‽): Self-referential or contradictory. Reason: $reason. Value: $value"
}

/**
 * Assert evidence level meets requirement
 */
fun <T> Evidence<T>.require(level: EvidenceLevel, message: String = "Evidence requirement not met"): Evidence<T> {
    require(this.level.canFlowTo(level)) {
        "$message: have ${this.level.symbol}, need ${level.symbol}"
    }
    return this
}

/**
 * Trust boundary - explicit conversion (logged for audit)
 */
inline fun <T> Evidence<T>.trust(reason: String, audit: (String) -> Unit = {}): Known<T> {
    audit("TRUST BOUNDARY: Converting ${this.level} to Known. Reason: $reason")
    return Known(value)
}

// ═══════════════════════════════════════════════════════════════════════════
// Example: How AI code feels on JVM
// ═══════════════════════════════════════════════════════════════════════════

/*
 * Native Sigil:
 *   let data~ = fetch_sensor();
 *   let valid? = data |> φ(in_range) |> τ(normalize);
 *   let result! = valid |> ρ(0, add) |> verify(threshold);
 *
 * Kotlin with Sigil DSL:
 *   val data = fetchSensor().reported(Source.Sensor)
 *   val valid = data φ ::inRange τ ::normalize
 *   val result = valid.ρ(0, Int::plus).verify { it > threshold }
 *
 * Both preserve:
 *   - Evidence tracking
 *   - Morpheme expressiveness
 *   - Clear data provenance
 *   - Type safety
 */

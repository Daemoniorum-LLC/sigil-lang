package com.daemoniorum.sigil.runtime

import kotlinx.serialization.Serializable

/**
 * Evidence levels in Sigil's epistemic type system.
 *
 * Ordered from most certain to least certain:
 * - KNOWN (!) - Direct computation, verified
 * - UNCERTAIN (?) - Validated but may be absent
 * - REPORTED (~) - External source, untrusted
 * - PARADOX (‽) - Self-referential, contradictory
 */
@Serializable
enum class EvidenceLevel(val symbol: Char, val order: Int) {
    KNOWN('!', 0),
    UNCERTAIN('?', 1),
    REPORTED('~', 2),
    PARADOX('‽', 3);

    companion object {
        fun fromSymbol(symbol: Char): EvidenceLevel? = entries.find { it.symbol == symbol }

        /**
         * Join: least upper bound (weakest evidence)
         * known ⊔ uncertain = uncertain
         */
        fun join(a: EvidenceLevel, b: EvidenceLevel): EvidenceLevel =
            if (a.order >= b.order) a else b

        /**
         * Meet: greatest lower bound (strongest evidence)
         * known ⊓ uncertain = known
         */
        fun meet(a: EvidenceLevel, b: EvidenceLevel): EvidenceLevel =
            if (a.order <= b.order) a else b
    }

    /**
     * Check if this evidence level can flow to another (covariance)
     * known → uncertain → reported → paradox
     */
    fun canFlowTo(other: EvidenceLevel): Boolean = this.order <= other.order
}

/**
 * Evidence wrapper - the core of Sigil's epistemic type system.
 *
 * Every value in Sigil carries evidence metadata tracking its provenance.
 */
@Serializable
sealed class Evidence<out T> {
    abstract val value: T
    abstract val level: EvidenceLevel

    /**
     * Transform the wrapped value while preserving evidence level
     */
    inline fun <R> map(transform: (T) -> R): Evidence<R> = when (this) {
        is Known -> Known(transform(value))
        is Uncertain -> Uncertain(transform(value))
        is Reported -> Reported(transform(value), source)
        is Paradox -> Paradox(transform(value), reason)
    }

    /**
     * FlatMap with evidence propagation (join)
     */
    inline fun <R> flatMap(transform: (T) -> Evidence<R>): Evidence<R> {
        val result = transform(value)
        return when {
            this.level.order >= result.level.order -> result.withLevel(this.level)
            else -> result
        }
    }

    /**
     * Create a copy with a different evidence level
     */
    fun withLevel(newLevel: EvidenceLevel): Evidence<T> = when (newLevel) {
        EvidenceLevel.KNOWN -> Known(value)
        EvidenceLevel.UNCERTAIN -> Uncertain(value)
        EvidenceLevel.REPORTED -> Reported(value, (this as? Reported)?.source ?: Source.Derived)
        EvidenceLevel.PARADOX -> Paradox(value, (this as? Paradox)?.reason ?: "Elevated to paradox")
    }

    /**
     * Validate: attempt to strengthen evidence
     * reported~ + validation → uncertain?
     */
    inline fun validate(predicate: (T) -> Boolean): Evidence<T> {
        return if (predicate(value)) {
            when (this) {
                is Reported -> Uncertain(value)
                is Paradox -> Uncertain(value)
                else -> this
            }
        } else {
            Paradox(value, "Validation failed")
        }
    }

    /**
     * Verify: attempt to achieve known status
     * uncertain? + verification → known!
     */
    inline fun verify(verifier: (T) -> Boolean): Evidence<T> {
        return if (verifier(value)) {
            Known(value)
        } else {
            this
        }
    }

    /**
     * Assume known - explicit trust boundary (use with caution!)
     */
    fun assumeKnown(): Known<T> = Known(value)

    /**
     * Check if evidence is at least as strong as required
     */
    fun satisfies(required: EvidenceLevel): Boolean = this.level.order <= required.order

    override fun toString(): String = "${level.symbol}$value"
}

/**
 * Known evidence (!) - directly computed, verified
 */
@Serializable
data class Known<out T>(override val value: T) : Evidence<T>() {
    override val level: EvidenceLevel = EvidenceLevel.KNOWN
}

/**
 * Uncertain evidence (?) - validated but possibly absent
 */
@Serializable
data class Uncertain<out T>(override val value: T) : Evidence<T>() {
    override val level: EvidenceLevel = EvidenceLevel.UNCERTAIN
}

/**
 * Reported evidence (~) - from external source, untrusted
 */
@Serializable
data class Reported<out T>(
    override val value: T,
    val source: Source = Source.External
) : Evidence<T>() {
    override val level: EvidenceLevel = EvidenceLevel.REPORTED
}

/**
 * Paradox evidence (‽) - self-referential or contradictory
 */
@Serializable
data class Paradox<out T>(
    override val value: T,
    val reason: String
) : Evidence<T>() {
    override val level: EvidenceLevel = EvidenceLevel.PARADOX
}

/**
 * Source metadata for reported evidence
 */
@Serializable
sealed class Source {
    @Serializable
    object External : Source()

    @Serializable
    object Derived : Source()

    @Serializable
    object Sensor : Source()

    @Serializable
    object Network : Source()

    @Serializable
    object User : Source()

    @Serializable
    data class Named(val name: String) : Source()
}

// Extension functions for common operations

/**
 * Combine two evidence values, joining their evidence levels
 */
inline fun <A, B, R> Evidence<A>.combine(
    other: Evidence<B>,
    transform: (A, B) -> R
): Evidence<R> {
    val combinedLevel = EvidenceLevel.join(this.level, other.level)
    val result = transform(this.value, other.value)
    return when (combinedLevel) {
        EvidenceLevel.KNOWN -> Known(result)
        EvidenceLevel.UNCERTAIN -> Uncertain(result)
        EvidenceLevel.REPORTED -> Reported(result)
        EvidenceLevel.PARADOX -> Paradox(result, "Combined from paradox")
    }
}

// DSL builders
fun <T> known(value: T): Known<T> = Known(value)
fun <T> uncertain(value: T): Uncertain<T> = Uncertain(value)
fun <T> reported(value: T, source: Source = Source.External): Reported<T> = Reported(value, source)
fun <T> paradox(value: T, reason: String): Paradox<T> = Paradox(value, reason)

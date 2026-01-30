package com.daemoniorum.sigil.runtime

import kotlinx.serialization.Serializable

/**
 * Runtime values for the Sigil interpreter.
 *
 * Every value carries evidence metadata - this is what makes Sigil
 * special for AI: you always know where data came from.
 */
@Serializable
sealed class SigilValue {
    abstract val evidence: EvidenceLevel

    // ═══════════════════════════════════════════════════════════════════════
    // Primitive Values
    // ═══════════════════════════════════════════════════════════════════════

    @Serializable
    data class Int(val value: Long, override val evidence: EvidenceLevel) : SigilValue() {
        override fun toString() = "${evidence.symbol}$value"
    }

    @Serializable
    data class Float(val value: Double, override val evidence: EvidenceLevel) : SigilValue() {
        override fun toString() = "${evidence.symbol}$value"
    }

    @Serializable
    data class Bool(val value: Boolean, override val evidence: EvidenceLevel) : SigilValue() {
        override fun toString() = "${evidence.symbol}$value"
    }

    @Serializable
    data class Str(val value: String, override val evidence: EvidenceLevel) : SigilValue() {
        override fun toString() = "${evidence.symbol}\"$value\""
    }

    @Serializable
    data class Char(val value: kotlin.Char, override val evidence: EvidenceLevel) : SigilValue() {
        override fun toString() = "${evidence.symbol}'$value'"
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Compound Values
    // ═══════════════════════════════════════════════════════════════════════

    @Serializable
    data class List(
        val elements: kotlin.collections.List<SigilValue>,
        override val evidence: EvidenceLevel
    ) : SigilValue() {
        override fun toString() = "${evidence.symbol}[${elements.joinToString(", ")}]"
    }

    @Serializable
    data class Map(
        val entries: kotlin.collections.Map<SigilValue, SigilValue>,
        override val evidence: EvidenceLevel
    ) : SigilValue() {
        override fun toString() = "${evidence.symbol}{${entries.entries.joinToString(", ") { "${it.key}: ${it.value}" }}}"
    }

    @Serializable
    data class Tuple(
        val elements: kotlin.collections.List<SigilValue>,
        override val evidence: EvidenceLevel
    ) : SigilValue() {
        override fun toString() = "${evidence.symbol}(${elements.joinToString(", ")})"
    }

    @Serializable
    data class Struct(
        val name: String,
        val fields: kotlin.collections.Map<String, SigilValue>,
        override val evidence: EvidenceLevel
    ) : SigilValue() {
        override fun toString() = "${evidence.symbol}$name { ${fields.entries.joinToString(", ") { "${it.key}: ${it.value}" }} }"
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Special Values
    // ═══════════════════════════════════════════════════════════════════════

    @Serializable
    object Unit : SigilValue() {
        override val evidence = EvidenceLevel.KNOWN
        override fun toString() = "()"
    }

    @Serializable
    data class Option(
        val value: SigilValue?,
        override val evidence: EvidenceLevel
    ) : SigilValue() {
        override fun toString() = if (value != null) "Some($value)" else "None"
    }

    @Serializable
    data class Result(
        val value: SigilValue?,
        val error: String?,
        override val evidence: EvidenceLevel
    ) : SigilValue() {
        val isOk: Boolean get() = error == null
        override fun toString() = if (isOk) "Ok($value)" else "Err($error)"
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Functions
    // ═══════════════════════════════════════════════════════════════════════

    data class Function(
        val name: String,
        val params: kotlin.collections.List<String>,
        val body: suspend (kotlin.collections.List<SigilValue>) -> SigilValue,
        override val evidence: EvidenceLevel = EvidenceLevel.KNOWN
    ) : SigilValue() {
        override fun toString() = "<fn $name(${params.joinToString(", ")})>"
    }

    data class Closure(
        val params: kotlin.collections.List<String>,
        val body: suspend (kotlin.collections.List<SigilValue>) -> SigilValue,
        val captures: kotlin.collections.Map<String, SigilValue>,
        override val evidence: EvidenceLevel = EvidenceLevel.KNOWN
    ) : SigilValue() {
        override fun toString() = "<closure(${params.joinToString(", ")})>"
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Evidence Operations
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Create a copy with different evidence level
     */
    fun withEvidence(newEvidence: EvidenceLevel): SigilValue = when (this) {
        is Int -> copy(evidence = newEvidence)
        is Float -> copy(evidence = newEvidence)
        is Bool -> copy(evidence = newEvidence)
        is Str -> copy(evidence = newEvidence)
        is Char -> copy(evidence = newEvidence)
        is List -> copy(evidence = newEvidence)
        is Map -> copy(evidence = newEvidence)
        is Tuple -> copy(evidence = newEvidence)
        is Struct -> copy(evidence = newEvidence)
        is Unit -> Unit
        is Option -> copy(evidence = newEvidence)
        is Result -> copy(evidence = newEvidence)
        is Function -> copy(evidence = newEvidence)
        is Closure -> copy(evidence = newEvidence)
    }

    /**
     * Check if this value's evidence is at least as strong as required
     */
    fun satisfies(required: EvidenceLevel): Boolean =
        this.evidence.order <= required.order

    /**
     * Combine evidence with another value (join = weakest)
     */
    fun combineEvidence(other: SigilValue): EvidenceLevel =
        EvidenceLevel.join(this.evidence, other.evidence)
}

// ═══════════════════════════════════════════════════════════════════════════
// Value Builders (for interpreter and stdlib)
// ═══════════════════════════════════════════════════════════════════════════

fun sigilInt(value: Long, evidence: EvidenceLevel = EvidenceLevel.KNOWN) =
    SigilValue.Int(value, evidence)

fun sigilFloat(value: Double, evidence: EvidenceLevel = EvidenceLevel.KNOWN) =
    SigilValue.Float(value, evidence)

fun sigilBool(value: Boolean, evidence: EvidenceLevel = EvidenceLevel.KNOWN) =
    SigilValue.Bool(value, evidence)

fun sigilStr(value: String, evidence: EvidenceLevel = EvidenceLevel.KNOWN) =
    SigilValue.Str(value, evidence)

fun sigilList(
    elements: List<SigilValue>,
    evidence: EvidenceLevel = elements.fold(EvidenceLevel.KNOWN) { acc, v ->
        EvidenceLevel.join(acc, v.evidence)
    }
) = SigilValue.List(elements, evidence)

fun sigilMap(
    entries: Map<SigilValue, SigilValue>,
    evidence: EvidenceLevel = EvidenceLevel.KNOWN
) = SigilValue.Map(entries, evidence)

fun sigilStruct(
    name: String,
    fields: Map<String, SigilValue>,
    evidence: EvidenceLevel = fields.values.fold(EvidenceLevel.KNOWN) { acc, v ->
        EvidenceLevel.join(acc, v.evidence)
    }
) = SigilValue.Struct(name, fields, evidence)

fun sigilNone(evidence: EvidenceLevel = EvidenceLevel.UNCERTAIN) =
    SigilValue.Option(null, evidence)

fun sigilSome(value: SigilValue) =
    SigilValue.Option(value, value.evidence)

fun sigilOk(value: SigilValue) =
    SigilValue.Result(value, null, value.evidence)

fun sigilErr(error: String, evidence: EvidenceLevel = EvidenceLevel.PARADOX) =
    SigilValue.Result(null, error, evidence)

// ═══════════════════════════════════════════════════════════════════════════
// Type Extraction Helpers
// ═══════════════════════════════════════════════════════════════════════════

fun SigilValue.asInt(): Long = (this as SigilValue.Int).value
fun SigilValue.asFloat(): Double = (this as SigilValue.Float).value
fun SigilValue.asBool(): Boolean = (this as SigilValue.Bool).value
fun SigilValue.asStr(): String = (this as SigilValue.Str).value
fun SigilValue.asList(): List<SigilValue> = (this as SigilValue.List).elements
fun SigilValue.asMap(): Map<SigilValue, SigilValue> = (this as SigilValue.Map).entries

fun SigilValue.intOrNull(): Long? = (this as? SigilValue.Int)?.value
fun SigilValue.floatOrNull(): Double? = (this as? SigilValue.Float)?.value
fun SigilValue.boolOrNull(): Boolean? = (this as? SigilValue.Bool)?.value
fun SigilValue.strOrNull(): String? = (this as? SigilValue.Str)?.value

// ═══════════════════════════════════════════════════════════════════════════
// AI-Friendly Introspection
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get a structured description of this value for AI consumption
 */
fun SigilValue.describe(): String = buildString {
    appendLine("Value: $this")
    appendLine("Type: ${typeName()}")
    appendLine("Evidence: ${evidence.symbol} (${evidenceDescription()})")

    if (this@describe is SigilValue.Struct) {
        appendLine("Fields:")
        fields.forEach { (name, value) ->
            appendLine("  $name: ${value.typeName()}${value.evidence.symbol} = $value")
        }
    }
}

fun SigilValue.typeName(): String = when (this) {
    is SigilValue.Int -> "i64"
    is SigilValue.Float -> "f64"
    is SigilValue.Bool -> "bool"
    is SigilValue.Str -> "str"
    is SigilValue.Char -> "char"
    is SigilValue.List -> "List"
    is SigilValue.Map -> "Map"
    is SigilValue.Tuple -> "Tuple"
    is SigilValue.Struct -> name
    is SigilValue.Unit -> "()"
    is SigilValue.Option -> "Option"
    is SigilValue.Result -> "Result"
    is SigilValue.Function -> "fn"
    is SigilValue.Closure -> "closure"
}

fun SigilValue.evidenceDescription(): String = when (evidence) {
    EvidenceLevel.KNOWN -> "directly computed or verified"
    EvidenceLevel.UNCERTAIN -> "validated but may be absent"
    EvidenceLevel.REPORTED -> "from external source, not verified"
    EvidenceLevel.PARADOX -> "contradictory or self-referential"
}

# 24-SGDOC: Sigil Documentation Format

**Version:** 0.5.0
**Status:** ! Stable
**Authors:** Claude (Opus 4.5) + Human
**Date:** 2026-01-25

---

## 1. Overview

SGDOC is Sigil's native documentation format. It uses Sigil syntax to express documentation as structured, typed, evidentiality-marked data. Documentation written in SGDOC is:

- **Parseable** by the Sigil compiler
- **Verifiable** against code and tests
- **Evidential** - claims carry certainty markers
- **Queryable** - structured data, not prose blobs

---

## 2. Evidentiality in Documentation

### 2.1 The Problem

Traditional documentation makes claims without indicating certainty:

```markdown
The function returns the sum of its arguments.  // Is this verified? Assumed? Hoped?
```

### 2.2 The Solution

SGDOC uses Sigil's evidentiality markers to indicate claim certainty:

| Marker | Meaning | Documentation Use |
|--------|---------|-------------------|
| `!` | Verified (sensory) | Claim backed by passing test |
| `~` | Reported (hearsay) | Claim from spec, not yet tested |
| `?` | Uncertain | Needs investigation |
| `◊` | Predicted | Planned/future feature |

```sigil
// SGDOC example
doc! "Returns the sum of arguments"      // Verified by test
doc~ "Handles overflow gracefully"       // Spec says so, untested
doc? "Thread-safe"                        // Unknown, needs investigation
doc◊ "Will support SIMD in v0.5.0"       // Planned feature
```

---

## 3. Core Types

### 3.1 Documentation Structs

```sigil
/// Document metadata - the header block
Σ DocMeta {
    title!: String,           // Required, verified
    version!: SemVer,         // Required, verified
    status: DocStatus,        // Draft, Stable, Deprecated
    created!: DateTime,       // When created
    updated!: DateTime,       // Last modification
    authors: Vec<String>,     // Contributors
    spec_refs: Vec<SpecRef>,  // Bound specifications
    code_refs: Vec<CodeRef>,  // Bound source files
}

/// Semantic version
Σ SemVer {
    major!: u32,
    minor!: u32,
    patch!: u32,
    prerelease~: Option<String>,
}

/// Document status
ᛈ DocStatus {
    Draft,                    // Work in progress
    Review,                   // Ready for verification
    Stable,                   // Verified and complete
    Deprecated(String),       // Superseded, with reason
}

/// Reference to a specification section
Σ SpecRef {
    spec!: String,            // e.g., "14-NEURAL"
    section!: String,         // e.g., "§4.2"
    verified!: bool,          // Has this been checked?
}

/// Reference to source code
Σ CodeRef {
    path!: String,            // e.g., "parser/src/interpreter.rs"
    line~: Option<u32>,       // Specific line if applicable
    symbol~: Option<String>,  // Function/type name
}
```

### 3.2 Claim Types

```sigil
/// A documentable claim with evidentiality
Σ Claim<E> {
    content!: String,         // The claim text
    evidence: E,              // Evidentiality marker (type-level)
    test_ref~: Option<TestRef>,  // Test that verifies this
    spec_ref~: Option<SpecRef>,  // Spec that defines this
}

/// Verified claim - backed by passing test
type Claim! = Claim<Verified>;

/// Reported claim - from spec, not yet tested
type Claim~ = Claim<Reported>;

/// Uncertain claim - needs investigation
type Claim? = Claim<Uncertain>;

/// Predicted claim - planned feature
type Claim◊ = Claim<Predicted>;

/// Test reference for verification
Σ TestRef {
    file!: String,            // Test file path
    test_name!: String,       // Test function name
    last_run~: Option<DateTime>,
    passed~: Option<bool>,
}
```

### 3.3 Documentation Blocks

```sigil
/// A complete documentation unit
Σ Doc {
    meta!: DocMeta,
    summary!: String,
    sections: Vec<Section>,
    examples: Vec<Example>,
    see_also: Vec<DocRef>,
}

/// A documentation section
Σ Section {
    id!: String,              // e.g., "2.1"
    title!: String,
    claims: Vec<AnyClaim>,    // Mixed evidentiality
    subsections: Vec<Section>,
}

/// Enum for any claim type
ᛈ AnyClaim {
    Verified(Claim!),
    Reported(Claim~),
    Uncertain(Claim?),
    Predicted(Claim◊),
}

/// A code example
Σ Example {
    title!: String,
    code!: String,            // Sigil source
    output!: ExpectedOutput,  // What it should produce
    verified!: bool,          // Has this been run?
}

ᛈ ExpectedOutput {
    Exact(String),            // Must match exactly
    Contains(String),         // Must contain substring
    Regex(String),            // Must match pattern
    Compiles,                 // Just needs to compile
    Fails(String),            // Should fail with error
}
```

---

## 4. SGDOC File Format

### 4.1 File Extension

SGDOC files use the `.sgdoc` extension and are valid Sigil source files.

### 4.2 Structure

```sigil
// file: docs/reference/interpreter.sgdoc

tome interpreter_doc;

invoke sgdoc·*;

/// Document metadata
≔ META! = DocMeta {
    title!: "Interpreter Reference",
    version!: SemVer { major!: 0, minor!: 4, patch!: 0 },
    status: DocStatus·Stable,
    created!: datetime!("2026-01-25"),
    updated!: datetime!("2026-01-25"),
    authors: ["Claude", "Human"],
    spec_refs: [
        SpecRef { spec!: "02-SYNTAX", section!: "§3", verified!: true },
    ],
    code_refs: [
        CodeRef { path!: "parser/src/interpreter.rs", symbol~: Some("Interpreter") },
    ],
};

/// Main documentation
≔ DOC! = Doc {
    meta!: META!,
    summary!: "The Sigil interpreter executes parsed AST nodes.",
    sections: [
        Section {
            id!: "1",
            title!: "Overview",
            claims: [
                AnyClaim·Verified(Claim! {
                    content!: "The interpreter supports all P0 language features.",
                    test_ref~: Some(TestRef {
                        file!: "jormungandr/tests/run_tests_rust.sh",
                        test_name!: "P0 suite",
                        passed~: Some(true),
                    }),
                }),
                AnyClaim·Reported(Claim~ {
                    content!: "Memory usage scales linearly with AST depth.",
                    spec_ref~: Some(SpecRef {
                        spec!: "02-SYNTAX",
                        section!: "§5.2",
                        verified!: false,
                    }),
                }),
            ],
            subsections: [],
        },
    ],
    examples: [
        Example {
            title!: "Basic execution",
            code!: r#"
                λ main() {
                    println("Hello, Sigil!");
                }
            "#,
            output!: ExpectedOutput·Exact("Hello, Sigil!\n"),
            verified!: true,
        },
    ],
    see_also: [],
};
```

### 4.3 Shorthand Syntax

SGDOC provides three shorthand syntaxes for creating claims:

#### 4.3.1 Evidential Shorthand (Recommended)

Use evidentiality markers directly on the `doc` identifier:

```sigil
// Shorthand using evidentiality markers
≔ claim1 = doc!("Returns the sum of arguments");      // Verified (macro)
≔ claim2 = doc~("Handles overflow gracefully");       // Reported
≔ claim3 = doc◊("Will support SIMD in v0.5.0");       // Predicted

// These expand to the appropriate Claim constructors
```

**Note:** `doc?()` (uncertain) is not available because `?` is reserved for the
try operator. Use `Claim·uncertain()` or `Claim·u()` for uncertain claims.

#### 4.3.2 Middledot Shorthand

Middledot notation provides single-letter abbreviations:

```sigil
// Shorthand claim constructors
≔ claim1 = Claim·v("The function returns a Result");   // Verified
≔ claim2 = Claim·r("Supports async execution");        // Reported
≔ claim3 = Claim·u("Thread-safe under concurrent access");  // Uncertain
≔ claim4 = Claim·p("Will support WASM in v0.5.0");     // Predicted
```

#### 4.3.3 Full Constructors

Full constructors allow specifying references:

```sigil
// Full constructors with references
≔ verified = Claim·verified("Handles overflow correctly", "tests/overflow_test.sg");
≔ reported = Claim·reported("Memory-safe", "02-MEMORY§3.1");

// Example with verification
≔ example = Example·new("Addition", "println(1 + 2);", "3\n");
≔ verified_example = Example·verify(example);  // Mark as verified
```

---

## 5. Verification Pipeline

### 5.1 Compile-Time Checks

The Sigil compiler can verify SGDOC files:

```bash
sigil doc-check docs/reference/*.sgdoc
```

**Checks performed:**
- All `Claim!` have valid `test_ref` that passes
- All `code_refs` point to existing files/symbols
- All `spec_refs` point to existing spec sections
- All examples compile

### 5.2 Verification Report

```sigil
/// Output of doc verification
Σ VerificationReport {
    file!: String,
    timestamp!: DateTime,
    claims_verified!: u32,
    claims_reported~: u32,
    claims_uncertain?: u32,
    claims_predicted◊: u32,
    failures: Vec<VerificationFailure>,
}

Σ VerificationFailure {
    claim: AnyClaim,
    reason!: String,
    location!: CodeRef,
}
```

### 5.3 Evidentiality Promotion

Claims can be promoted as verification improves:

```
Claim◊ (predicted) → Claim~ (reported in spec) → Claim! (verified by test)
```

```sigil
// Promote a claim when test is added
≔ old_claim~ = doc~("Handles edge case X");
≔ new_claim! = old_claim~|verify("tests/edge_case_x.sg#test_x");
```

### 5.4 Evidentiality Demotion

Claims can be demoted when verification fails:

```
Claim! (was verified) → Claim? (test now fails) → investigate
```

The compiler can detect this automatically when tests fail.

---

## 6. Querying Documentation

### 6.1 Programmatic Access

SGDOC files can be imported and queried:

```sigil
invoke interpreter_doc;

// Get all unverified claims
≔ unverified = DOC!.sections
    |τ{s => s.claims}
    |flatten
    |φ{c => ⌥ c {
        AnyClaim·Verified(_) => false,
        _ => true,
    }};

// Get all claims about a specific topic
≔ async_claims = DOC!.sections
    |τ{s => s.claims}
    |flatten
    |φ{c => c.content!.contains("async")};

// Count by evidentiality
≔ verified_count = DOC!.sections
    |τ{s => s.claims}
    |flatten
    |φ{c => matches!(c, AnyClaim·Verified(_))}
    |len;
```

### 6.2 CLI Queries

```bash
# List all uncertain claims
sigil doc-query --uncertain docs/

# List all claims referencing a spec section
sigil doc-query --spec "14-NEURAL§4.2" docs/

# List all claims about a code symbol
sigil doc-query --symbol "eval_pipe" docs/

# Generate coverage report
sigil doc-coverage docs/
```

---

## 7. Integration with Agent-Doc Pipeline

### 7.1 Audit Phase

Generate SGDOC manifests from code:

```bash
sigil doc-audit src/ --output docs/AUDIT.sgdoc
```

```sigil
// Generated AUDIT.sgdoc
Σ AuditResult {
    timestamp!: DateTime,
    modules_scanned!: u32,
    public_symbols!: u32,
    documented_symbols!: u32,
    coverage!: f64,
    gaps: Vec<Gap>,
}

Σ Gap {
    symbol!: String,
    location!: CodeRef,
    priority: Priority,
    suggested_claims~: Vec<String>,  // AI-suggested documentation
}
```

### 7.2 Manifest Phase

```sigil
// MANIFEST.sgdoc
Σ ManifestEntry {
    id!: String,
    doc_type!: DocType,
    source!: CodeRef,
    status: ManifestStatus,
    priority: Priority,
    assigned~: Option<String>,  // Agent session
}

ᛈ ManifestStatus {
    Missing,
    Draft,
    Review,
    Complete,
    Stale(DateTime),  // When it became stale
}
```

### 7.3 Verification Phase

```bash
sigil doc-verify docs/ --report VERIFICATION.sgdoc
```

---

## 8. Rendering

### 8.1 To Markdown

```bash
sigil doc-render docs/reference/interpreter.sgdoc --format markdown
```

Output includes evidentiality badges:

```markdown
# Interpreter Reference

## 1. Overview

- ✓ The interpreter supports all P0 language features. [test: P0 suite]
- ○ Memory usage scales linearly with AST depth. [spec: 02-SYNTAX §5.2]
```

### 8.2 To HTML

```bash
sigil doc-render docs/ --format html --output site/
```

### 8.3 To JSON (for tooling)

```bash
sigil doc-render docs/ --format json
```

---

## 9. Standard Library

### 9.1 sgdoc Tome

The SGDOC tome is registered as part of the Sigil stdlib. All functions use middledot
notation for namespace separation.

```sigil
// === DocMeta Constructors ===
DocMeta·new(title: String, major: i32, minor: i32, patch: i32) -> DocMeta
DocMeta·stable(meta: DocMeta) -> DocMeta  // Set status to "stable"

// === Claim Constructors ===
Claim·verified(content: String, test_ref: String) -> Claim!
Claim·reported(content: String, spec_ref: String) -> Claim~
Claim·uncertain(content: String) -> Claim?
Claim·predicted(content: String, target_version: String) -> Claim◊

// === Shorthand Constructors ===
Claim·v(content: String) -> Claim!  // Verified shorthand
Claim·r(content: String) -> Claim~  // Reported shorthand
Claim·u(content: String) -> Claim?  // Uncertain shorthand
Claim·p(content: String) -> Claim◊  // Predicted shorthand

// === Claim Operations ===
Claim·promote(claim: Claim, test_ref: Any) -> Claim!
Claim·demote(claim: Claim) -> Claim?
Claim·is_verified(claim: Claim) -> bool

// === Section Constructors ===
Section·new(id: String, title: String) -> Section
Section·add_claim(section: Section, claim: Claim) -> Section

// === Doc Constructors ===
Doc·new(meta: DocMeta, summary: String) -> Doc
Doc·add_section(doc: Doc, section: Section) -> Doc

// === Example Constructors ===
Example·new(title: String, code: String, expected: Any) -> Example
Example·verify(example: Example) -> Example  // Mark as verified

// === Reference Constructors ===
SpecRef·new(spec: String, section: String) -> SpecRef
CodeRef·new(path: String, symbol: String?) -> CodeRef
TestRef·new(file: String, test_name: String) -> TestRef

// === Verification ===
Doc·verify(doc: Doc) -> VerificationReport
Doc·claims(doc: Doc) -> Vec<Claim>
Doc·unverified_claims(doc: Doc) -> Vec<Claim>

// === Rendering ===
Doc·to_markdown(doc: Doc) -> String
Doc·to_html(doc: Doc) -> String
Doc·to_json(doc: Doc) -> String
```

**Implementation Notes:**
- `DocMeta·new` auto-populates `created` and `updated` with ISO8601 timestamps
- `Doc·verify` returns a `VerificationReport` with claim counts by evidentiality
- `Doc·to_html` includes CSS styling for evidentiality badges
- `Doc·to_json` preserves evidentiality markers in the output

**Time Functions:** SGDOC uses these stdlib time functions for timestamps:
```sigil
Time·iso8601() -> String           // Current time as ISO8601 (e.g., "2026-01-25T12:30:00Z")
format_time(secs: Int) -> String   // Format Unix timestamp as ISO8601
Time·format(secs: Int, fmt: String) -> String  // Custom format (%Y, %m, %d, %H, %M, %S)
```

---

## 10. Example: Full SGDOC File

```sigil
// file: docs/reference/protocols.sgdoc
tome protocols_doc;

invoke sgdoc·*;
invoke std·datetime;

≔ META! = DocMeta {
    title!: "Protocol Support Reference",
    version!: SemVer { major!: 0, minor!: 4, patch!: 0 },
    status: DocStatus·Stable,
    created!: datetime!("2026-01-25"),
    updated!: datetime!("2026-01-25"),
    authors: ["Claude"],
    spec_refs: [
        SpecRef { spec!: "15-PROTOCOLS", section!: "§4", verified!: true },
        SpecRef { spec!: "15-PROTOCOLS", section!: "§5", verified!: true },
    ],
    code_refs: [
        CodeRef { path!: "parser/src/interpreter.rs", symbol~: Some("protocol_send") },
        CodeRef { path!: "parser/src/interpreter.rs", symbol~: Some("protocol_connect") },
    ],
};

≔ HTTP_SECTION = Section {
    id!: "1",
    title!: "HTTP Client",
    claims: [
        AnyClaim·Verified(Claim! {
            content!: "HTTP GET requests return status and body.",
            test_ref~: Some(TestRef {
                file!: "jormungandr/tests/spec/15_protocols/P0_001_http_client_real.sg",
                test_name!: "HTTP GET",
                passed~: Some(true),
            }),
        }),
        AnyClaim·Verified(Claim! {
            content!: "POST method is inferred when body is non-empty.",
            test_ref~: Some(TestRef {
                file!: "jormungandr/tests/spec/15_protocols/P0_001_http_post.sg",
                test_name!: "HTTP POST",
                passed~: Some(true),
            }),
        }),
    ],
    subsections: [],
};

≔ WS_SECTION = Section {
    id!: "2",
    title!: "WebSocket Client",
    claims: [
        AnyClaim·Verified(Claim! {
            content!: "WebSocket connections support text message echo.",
            test_ref~: Some(TestRef {
                file!: "jormungandr/tests/spec/15_protocols/P0_003_websocket_real.sg",
                test_name!: "WebSocket echo",
                passed~: Some(true),
            }),
        }),
    ],
    subsections: [],
};

≔ DEFERRED_SECTION = Section {
    id!: "3",
    title!: "Deferred Protocols (v0.5.0)",
    claims: [
        AnyClaim·Predicted(Claim◊ {
            content!: "gRPC client with protobuf support.",
            spec_ref~: Some(SpecRef { spec!: "15-PROTOCOLS", section!: "§6", verified!: false }),
        }),
        AnyClaim·Predicted(Claim◊ {
            content!: "Kafka producer and consumer.",
            spec_ref~: Some(SpecRef { spec!: "15-PROTOCOLS", section!: "§7", verified!: false }),
        }),
        AnyClaim·Predicted(Claim◊ {
            content!: "AMQP/RabbitMQ support.",
            spec_ref~: Some(SpecRef { spec!: "15-PROTOCOLS", section!: "§8", verified!: false }),
        }),
    ],
    subsections: [],
};

≔ DOC! = Doc {
    meta!: META!,
    summary!: "Sigil v0.4.0 provides production-ready HTTP and WebSocket clients.",
    sections: [HTTP_SECTION, WS_SECTION, DEFERRED_SECTION],
    examples: [
        Example {
            title!: "HTTP GET",
            code!: r#"
λ main() {
    ≔ conn = "https://httpbin.org/get"|connect;
    ≔ response~ = conn|send{""};
    println(response~.status);
}
            "#,
            output!: ExpectedOutput·Exact("200\n"),
            verified!: true,
        },
        Example {
            title!: "WebSocket Echo",
            code!: r#"
λ main() {
    ≔ conn = "wss://ws.postman-echo.com/raw"|connect;
    ≔ response! = conn|send{"Hello!"};
    println(response!);
}
            "#,
            output!: ExpectedOutput·Exact("Hello!\n"),
            verified!: true,
        },
    ],
    see_also: [],
};

// Entry point for doc tools
pub λ main() {
    ≔ report = verify(DOC!);
    println(report|to_json);
}
```

---

## 11. Summary

SGDOC brings Sigil's core philosophy to documentation:

| Feature | Benefit |
|---------|---------|
| Evidentiality markers | Claims declare their certainty level |
| Typed structures | Documentation is queryable data |
| Code references | Bindings to source are verifiable |
| Test references | Verified claims link to passing tests |
| Compiler integration | `sigil doc-check` validates everything |
| Programmatic access | Query docs like any other data |

**The key insight:** Documentation claims are assertions about code. Assertions should have evidence. Sigil's evidentiality system makes this explicit.

---

## 12. Edition 2025 Corrections

**Status:** ⚠️ GAP IDENTIFIED — Spec was authored before Sigil Edition 2025 syntax was finalised.

This section documents deviations between the pseudocode in earlier sections and
actual Edition 2025 syntax. The implementation follows the corrected forms below.
Earlier sections are preserved as-is for historical record; treat this section as
authoritative.

### 12.1 Collection Types

| Spec pseudocode | Edition 2025 |
|-----------------|--------------|
| `Vec<String>` | `[String]` |
| `Vec<SpecRef>` | `[SpecRef]` |
| `Option<String>` | `Option[String]` |
| `Option<T>` | `Option[T]` |

### 12.2 Function Syntax

| Spec pseudocode | Edition 2025 |
|-----------------|--------------|
| `λ main() { }` | `rite main() { }` |
| `pub λ main() { }` | `☉ rite main() { }` |

### 12.3 Module Declarations

`tome module_name;` is not valid Edition 2025. File-level modules do not need
a declaration — the filename is the module name. Remove all `tome` declarations.

### 12.4 Generic Phantom Types

The spec defined:

```sigil
Σ Claim<E> { ... }
type Claim! = Claim<Verified>;
type Claim~ = Claim<Reported>;
```

Evidentiality markers cannot appear in type names (`Claim!` is not a valid type
identifier). The phantom-type approach is replaced with a `ClaimKind` enum:

```sigil
☉ ᛈ ClaimKind { Verified, Reported, Uncertain, Predicted }

☉ Σ Claim {
    ☉ kind: ClaimKind!,
    ☉ content: String!,
    ☉ test_ref: Option[TestRef]?,
    ☉ spec_ref: Option[SpecRef]?,
}
```

Constructor shorthands remain unchanged: `Claim·v`, `Claim·r`, `Claim·u`, `Claim·p`.

`AnyClaim` is retained as an enum wrapping `Claim` by kind variant for
pattern-matching ergonomics.

### 12.5 Date Representation

`datetime!("...")` macro does not exist. Dates are plain `String` in ISO-8601
format (`"2026-02-21"`).

### 12.6 Public Type Visibility

All exported types and functions require `☉` (pub):

```sigil
☉ Σ DocMeta { ... }
☉ ᛈ DocStatus { ... }
☉ ⊢ DocMeta { ☉ rite new(...) -> Self! { ... } }
```

### 12.7 Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.4.0 | 2026-01-25 | Initial spec (Claude Opus 4.5 + Human) |
| 0.5.0 | 2026-02-21 | Gap §12: Edition 2025 syntax corrections (Claude Sonnet 4.6) |

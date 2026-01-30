# Sigil JVM Backend Specification

## Overview

A JVM bytecode backend for Sigil, enabling execution on any Java Virtual Machine including Android.

## Target Platforms

| Platform | Runtime | Notes |
|----------|---------|-------|
| Desktop JVM | Java 11+ / GraalVM | Full support |
| Android | ART (Android Runtime) | Bytecode → DEX conversion |
| Kotlin/JVM | JVM | Interop with Kotlin |
| Server | Any JVM | Spring, Micronaut, etc. |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Sigil Source                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Parser (existing)                        │
│                      → AST                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Type Checker (existing)                    │
│              → Typed AST + Evidence Info                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   JVM Codegen (NEW)                         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Type Mapper │  │ Evidence    │  │ Bytecode Generator  │ │
│  │             │  │ Runtime     │  │ (ASM library)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Output Options                          │
│                                                             │
│   .class files    │    .jar file    │    Run directly      │
└─────────────────────────────────────────────────────────────┘
```

## Type System Mapping

### Primitive Types

| Sigil Type | JVM Type | Bytecode |
|------------|----------|----------|
| `i8` | `byte` | `B` |
| `i16` | `short` | `S` |
| `i32` | `int` | `I` |
| `i64` | `long` | `J` |
| `f32` | `float` | `F` |
| `f64` | `double` | `D` |
| `bool` | `boolean` | `Z` |
| `char` | `char` | `C` |
| `str` | `java.lang.String` | `Ljava/lang/String;` |
| `void` | `void` | `V` |

### Evidentiality Types

The key innovation: evidentiality as a **type wrapper** with runtime tracking.

```java
package sigil.runtime;

// Base evidence interface
public sealed interface Evidence<T> permits Known, Uncertain, Reported, Paradox {
    T value();
    EvidenceLevel level();
    Evidence<T> validate(Validator<T> validator);
    <U> Evidence<U> map(Function<T, U> fn);
}

// Evidence levels
public enum EvidenceLevel {
    KNOWN,      // !
    UNCERTAIN,  // ?
    REPORTED,   // ~
    PARADOX     // ‽
}

// Concrete implementations
public record Known<T>(T value) implements Evidence<T> {
    public EvidenceLevel level() { return EvidenceLevel.KNOWN; }
}

public record Uncertain<T>(T value) implements Evidence<T> {
    public EvidenceLevel level() { return EvidenceLevel.UNCERTAIN; }
}

public record Reported<T>(T value, Source source) implements Evidence<T> {
    public EvidenceLevel level() { return EvidenceLevel.REPORTED; }
}

public record Paradox<T>(T value, String reason) implements Evidence<T> {
    public EvidenceLevel level() { return EvidenceLevel.PARADOX; }
}
```

### Sigil to Java Mapping Examples

```sigil
// Sigil
let x! = 42;
let y? = validate(x);
let z~ = fetch_external();
```

```java
// Generated Java bytecode equivalent
Evidence<Integer> x = new Known<>(42);
Evidence<Integer> y = x.validate(Validators::range);  // Returns Uncertain
Evidence<Integer> z = new Reported<>(fetchExternal(), Source.EXTERNAL);
```

### Evidence Covariance

```java
// Evidence covariance rules encoded in type system
public class EvidenceOps {
    // Known can flow to Uncertain (covariance)
    public static <T> Uncertain<T> toUncertain(Known<T> known) {
        return new Uncertain<>(known.value());
    }

    // Uncertain can flow to Reported
    public static <T> Reported<T> toReported(Uncertain<T> uncertain) {
        return new Reported<>(uncertain.value(), Source.DERIVED);
    }

    // Compile-time error: Reported cannot flow to Known
    // public static <T> Known<T> toKnown(Reported<T> reported) { ... }
}
```

## Morpheme Operators

Map to Java Streams with evidence preservation:

```sigil
// Sigil
let result! = data |> τ(transform) |> φ(predicate) |> ρ(reducer);
```

```java
// Generated Java
Evidence<Result> result = EvidenceStream.of(data)
    .map(transform)      // τ
    .filter(predicate)   // φ
    .reduce(reducer);    // ρ
```

### Morpheme Runtime Library

```java
package sigil.runtime;

public class Morphemes {
    // τ (tau) - Transform
    public static <T, U> Evidence<List<U>> tau(Evidence<List<T>> input, Function<T, U> fn) {
        return input.map(list -> list.stream().map(fn).toList());
    }

    // φ (phi) - Filter
    public static <T> Evidence<List<T>> phi(Evidence<List<T>> input, Predicate<T> pred) {
        return input.map(list -> list.stream().filter(pred).toList());
    }

    // σ (sigma) - Sort
    public static <T extends Comparable<T>> Evidence<List<T>> sigma(Evidence<List<T>> input) {
        return input.map(list -> list.stream().sorted().toList());
    }

    // ρ (rho) - Reduce
    public static <T, U> Evidence<U> rho(Evidence<List<T>> input, U identity, BiFunction<U, T, U> fn) {
        return input.map(list -> list.stream().reduce(identity, fn, (a, b) -> fn.apply(a, b)));
    }

    // Σ (Sigma) - Sum
    public static Evidence<Double> sum(Evidence<List<? extends Number>> input) {
        return input.map(list -> list.stream().mapToDouble(Number::doubleValue).sum());
    }

    // Π (Pi) - Product
    public static Evidence<Double> product(Evidence<List<? extends Number>> input) {
        return input.map(list -> list.stream().mapToDouble(Number::doubleValue).reduce(1.0, (a, b) -> a * b));
    }

    // α (alpha) - First
    public static <T> Evidence<T> alpha(Evidence<List<T>> input) {
        return input.map(list -> list.isEmpty() ? null : list.get(0));
    }

    // ω (omega) - Last
    public static <T> Evidence<T> omega(Evidence<List<T>> input) {
        return input.map(list -> list.isEmpty() ? null : list.get(list.size() - 1));
    }
}
```

## Bytecode Generation

Using ASM library for bytecode generation:

```rust
// parser/src/jvm_codegen.rs

use crate::ast::*;
use crate::typeck::TypedAst;

pub struct JvmCodegen {
    class_writer: ClassWriter,
    method_visitor: Option<MethodVisitor>,
    locals: HashMap<String, u16>,
    next_local: u16,
}

impl JvmCodegen {
    pub fn compile(typed_ast: &TypedAst) -> Result<Vec<u8>, JvmError> {
        let mut codegen = Self::new();
        codegen.emit_class(typed_ast)?;
        Ok(codegen.class_writer.to_bytes())
    }

    fn emit_expr(&mut self, expr: &TypedExpr) -> Result<(), JvmError> {
        match &expr.kind {
            ExprKind::Literal(lit) => self.emit_literal(lit, &expr.evidence),
            ExprKind::BinaryOp(op, lhs, rhs) => self.emit_binary(op, lhs, rhs),
            ExprKind::Call(name, args) => self.emit_call(name, args),
            ExprKind::MorphemeOp(op, input) => self.emit_morpheme(op, input),
            // ...
        }
    }

    fn emit_literal(&mut self, lit: &Literal, evidence: &Evidence) -> Result<(), JvmError> {
        match lit {
            Literal::Int(n) => {
                // Push integer constant
                self.mv().visit_ldc(*n);
                // Box in Integer
                self.mv().visit_method_insn(INVOKESTATIC, "java/lang/Integer", "valueOf", "(I)Ljava/lang/Integer;");
                // Wrap in evidence type
                self.wrap_evidence(evidence)?;
            }
            // ...
        }
        Ok(())
    }

    fn wrap_evidence(&mut self, evidence: &Evidence) -> Result<(), JvmError> {
        let evidence_class = match evidence {
            Evidence::Known => "sigil/runtime/Known",
            Evidence::Uncertain => "sigil/runtime/Uncertain",
            Evidence::Reported => "sigil/runtime/Reported",
            Evidence::Paradox => "sigil/runtime/Paradox",
        };

        // NEW evidence_class
        // DUP
        // SWAP (value is under)
        // INVOKESPECIAL <init>
        self.mv().visit_type_insn(NEW, evidence_class);
        self.mv().visit_insn(DUP_X1);
        self.mv().visit_insn(SWAP);
        self.mv().visit_method_insn(INVOKESPECIAL, evidence_class, "<init>", "(Ljava/lang/Object;)V");

        Ok(())
    }
}
```

## CLI Integration

```bash
# Compile to JVM bytecode
sigil compile --target jvm program.sigil -o Program.class

# Compile to JAR with runtime
sigil compile --target jvm --jar program.sigil -o program.jar

# Run directly on JVM
sigil jvm program.sigil

# Compile for Android (DEX)
sigil compile --target android program.sigil -o classes.dex
```

## Runtime Library Structure

```
sigil-runtime/
├── build.gradle.kts
├── src/main/java/sigil/runtime/
│   ├── Evidence.java
│   ├── Known.java
│   ├── Uncertain.java
│   ├── Reported.java
│   ├── Paradox.java
│   ├── EvidenceOps.java
│   ├── Morphemes.java
│   ├── EvidenceStream.java
│   └── stdlib/
│       ├── Math.java
│       ├── String.java
│       ├── IO.java
│       ├── Crypto.java
│       └── ... (400+ functions)
└── src/main/kotlin/sigil/runtime/
    └── Extensions.kt  # Kotlin-friendly extensions
```

## Kotlin Interop

Native Kotlin DSL for Sigil types:

```kotlin
// sigil-runtime-kotlin/src/main/kotlin/sigil/Dsl.kt

package sigil

// Evidence builders
inline fun <T> known(value: T): Known<T> = Known(value)
inline fun <T> uncertain(value: T): Uncertain<T> = Uncertain(value)
inline fun <T> reported(value: T, source: Source = Source.EXTERNAL): Reported<T> = Reported(value, source)

// Operator overloads for morphemes
inline infix fun <T, R> Evidence<List<T>>.τ(transform: (T) -> R): Evidence<List<R>> =
    Morphemes.tau(this, transform)

inline infix fun <T> Evidence<List<T>>.φ(predicate: (T) -> Boolean): Evidence<List<T>> =
    Morphemes.phi(this, predicate)

inline fun <T : Comparable<T>> Evidence<List<T>>.σ(): Evidence<List<T>> =
    Morphemes.sigma(this)

// Example usage in Kotlin
fun example() {
    val data = known(listOf(1, 2, 3, 4, 5))

    val result = data τ { it * 2 } φ { it > 4 } // [6, 8, 10]

    println(result.value())  // [6, 8, 10]
    println(result.level())  // KNOWN
}
```

## Android Integration

### Gradle Setup

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("com.daemoniorum:sigil-runtime:0.1.0")
}
```

### Android Usage

```kotlin
// MainActivity.kt
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Use Sigil's evidence types for sensor data
        val sensorData: Reported<Float> = reported(accelerometer.read())

        val validated: Uncertain<Float> = sensorData.validate { it in -10f..10f }

        val processed: Known<String> = validated
            .map { "Acceleration: $it m/s²" }
            .assumeKnown()  // Explicit trust boundary

        textView.text = processed.value()
    }
}
```

## Implementation Phases

### Phase 1: Core Runtime (2 weeks)
- [ ] Evidence types (Known, Uncertain, Reported, Paradox)
- [ ] Basic operations (map, flatMap, validate)
- [ ] Publish `sigil-runtime` to Maven Central

### Phase 2: Bytecode Generator (4 weeks)
- [ ] AST to bytecode for expressions
- [ ] Function compilation
- [ ] Evidence wrapping/unwrapping
- [ ] Basic stdlib (print, math)

### Phase 3: Full Stdlib (3 weeks)
- [ ] Port 400+ stdlib functions to JVM
- [ ] Morpheme operators as Streams
- [ ] Crypto operations (use Java crypto)

### Phase 4: Android Support (2 weeks)
- [ ] DEX output
- [ ] Android-specific optimizations
- [ ] ProGuard/R8 rules
- [ ] Sample Android app

### Phase 5: Kotlin DSL (1 week)
- [ ] Operator overloads
- [ ] Coroutine integration
- [ ] Extension functions

## Dependencies

```toml
# parser/Cargo.toml additions
[features]
jvm = ["jvm-codegen"]

[dependencies]
# JVM bytecode generation
classfile = { version = "0.1", optional = true }  # Pure Rust classfile writer
# Or use Java ASM via JNI for complex cases
```

## Mobile Platform Summary

| Platform | Approach | Status |
|----------|----------|--------|
| **Android** | JVM bytecode → DEX | Planned (Phase 4) |
| **iOS** | LLVM backend → ARM64 | Already works |
| **Cross-platform** | Kotlin Multiplatform | Future option |

## Example: Full Sigil Program on JVM

```sigil
// sensor_app.sigil
fn process_sensor(reading~: f64) -> f64? {
    let validated? = if reading > -100.0 && reading < 100.0 {
        reading
    } else {
        return 0.0?;
    };

    let smoothed! = validated |> τ(smooth_filter);
    smoothed
}

fn main() {
    let raw~ = read_sensor();
    let result? = process_sensor(raw);
    print("Processed: " + str(result));
}
```

Compiles to:

```java
// Generated: SensorApp.class
public class SensorApp {
    public static Evidence<Double> processSensor(Reported<Double> reading) {
        Evidence<Double> validated;
        if (reading.value() > -100.0 && reading.value() < 100.0) {
            validated = new Uncertain<>(reading.value());
        } else {
            return new Uncertain<>(0.0);
        }

        Evidence<Double> smoothed = Morphemes.tau(
            validated.map(List::of),
            SensorApp::smoothFilter
        ).map(list -> list.get(0));

        return new Known<>(smoothed.value());
    }

    public static void main(String[] args) {
        Reported<Double> raw = new Reported<>(readSensor(), Source.SENSOR);
        Evidence<Double> result = processSensor(raw);
        System.out.println("Processed: " + result.value());
    }
}
```

# Sigil KMP - Kotlin Multiplatform

Kotlin Multiplatform implementation of the Sigil programming language. Run Sigil on JVM, Android, iOS, macOS, Linux, Windows, JavaScript, and WebAssembly.

## Supported Targets

| Platform | Target | Status |
|----------|--------|--------|
| JVM | Desktop, Server | ✅ |
| Android | Mobile | ✅ |
| iOS | arm64, x64, simulator | ✅ |
| macOS | arm64, x64 | ✅ |
| Linux | x64 | ✅ |
| Windows | x64 (mingw) | ✅ |
| JavaScript | Browser, Node.js | ✅ |
| WebAssembly | Browser | ✅ |

## Installation

### Gradle (JitPack)

Add JitPack repository to your `settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositories {
        maven("https://jitpack.io")
    }
}
```

Add the dependency:

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.github.Daemoniorum-LLC.sigil-lang:sigil-kmp:0.1.0")
}
```

### Gradle (Maven Central)

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.daemoniorum:sigil-kmp:0.1.0")
}
```

### Maven

```xml
<dependency>
    <groupId>com.daemoniorum</groupId>
    <artifactId>sigil-kmp-jvm</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Usage

### Basic Execution

```kotlin
import com.daemoniorum.sigil.Sigil
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    val result = Sigil.run("""
        fn main() {
            let x! = 42;
            let y? = x + 8;
            print(y);
        }
    """)

    println(result.output) // "50"
}
```

### REPL Mode

```kotlin
import com.daemoniorum.sigil.SigilRepl
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    val repl = SigilRepl()

    println(repl.execute("let x = 10"))
    println(repl.execute("x * 5"))  // "50"
}
```

### Parse to AST

```kotlin
val parseResult = Sigil.parse("""
    fn greet(name: String) {
        print("Hello, " + name);
    }
""")

println(parseResult.module.items)
```

### Type Checking

```kotlin
val typeResult = Sigil.check("""
    fn main() {
        let x! = 42;
        let y~ = x;  // Error: cannot assign known to reported
    }
""")

if (typeResult.hasErrors) {
    typeResult.errors.forEach { println(it.message) }
}
```

### JSON Export (for AI)

```kotlin
val json = Sigil.toJson("""
    fn fibonacci(n: Int): Int {
        if n <= 1 { return n }
        return fibonacci(n - 1) + fibonacci(n - 2)
    }
""")

println(json)  // Pretty-printed AST as JSON
```

## Evidentiality Types

Sigil's core innovation is tracking data provenance at the type level:

```kotlin
// In Sigil syntax:
val source = """
    let computed! = 1 + 1       // Known: verified truth
    let found? = map.get(key)   // Uncertain: may be absent
    let data~ = api.fetch(url)  // Reported: external, untrusted
"""
```

The type checker enforces evidence flow rules:
- `!` (known) can flow to `?` (uncertain) or `~` (reported)
- `?` (uncertain) can flow to `~` (reported)
- `~` (reported) cannot flow to stricter types without explicit verification

## Morpheme Operators

Elegant data transformation:

```kotlin
val sigilCode = """
    let numbers = [1, 2, 3, 4, 5];
    let result = numbers
        τ{_ * 2}      // Map: double each
        φ{_ > 4}      // Filter: keep if > 4
        σ             // Sort
        ρ+;           // Reduce: sum
    print(result);    // 24
"""
```

| Operator | Name | Function |
|----------|------|----------|
| τ | tau | Transform/map |
| φ | phi | Filter |
| σ | sigma | Sort |
| ρ | rho | Reduce/fold |
| Σ | Sigma | Sum |
| Π | Pi | Product |
| α | alpha | First element |
| ω | omega | Last element |

## Building from Source

```bash
cd sigil-kmp

# Build all targets
./gradlew build

# Run tests
./gradlew allTests

# Publish to local Maven
./gradlew publishToMavenLocal
```

## Android Integration

```kotlin
// In your Android app
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        lifecycleScope.launch {
            val result = Sigil.run("""
                fn main() {
                    print("Hello from Sigil on Android!");
                }
            """)

            textView.text = result.output
        }
    }
}
```

## iOS Integration (Swift)

```swift
import SigilKmp

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()

        Task {
            let result = try await Sigil.shared.run("""
                fn main() {
                    print("Hello from Sigil on iOS!");
                }
            """)

            label.text = result.output
        }
    }
}
```

## JavaScript/Browser

```javascript
import { Sigil } from '@daemoniorum/sigil';

const result = await Sigil.run(`
    fn main() {
        print("Hello from Sigil in the browser!");
    }
`);

console.log(result.output);
```

## License

Daemoniorum Source-Available License

Copyright (c) 2025 Daemoniorum, LLC

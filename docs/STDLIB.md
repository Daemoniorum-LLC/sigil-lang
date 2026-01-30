# Sigil Standard Library Reference

**Version:** 1.0-RC
**Last Updated:** January 16, 2026

The Sigil standard library provides 1,400+ functions across 49 modules, ranging from essential primitives to advanced AI agent infrastructure.

## Table of Contents

1. [Core](#core) - Essential functions (print, assert, panic)
2. [Math](#math) - Mathematical operations
3. [Collections](#collections) - Array, map, set operations
4. [String](#string) - String manipulation
5. [Evidence](#evidence) - Evidentiality markers and operations
6. [Affect](#affect) - Emotional/affective type operations
7. [Iter](#iter) - Iterator-style operations for pipes
8. [IO](#io) - File and console I/O
9. [Time](#time) - Date, time, and measurement
10. [Random](#random) - Random number generation
11. [Convert](#convert) - Type conversions
12. [Concurrency](#concurrency) - Threads, channels, async
13. [JSON](#json) - JSON parsing and serialization
14. [FS](#fs) - File system operations
15. [Crypto](#crypto) - Hashing and encryption
16. [Regex](#regex) - Regular expression matching
17. [UUID](#uuid) - UUID generation
18. [System](#system) - Environment, args, process control
19. [Stats](#stats) - Statistical functions
20. [Matrix](#matrix) - Matrix operations
21. [Functional](#functional) - Functional programming utilities
22. [Benchmark](#benchmark) - Performance measurement
23. [Itertools](#itertools) - Advanced iteration
24. [Ranges](#ranges) - Range generators
25. [Bitwise](#bitwise) - Bit manipulation
26. [Format](#format) - String formatting
27. [Pattern](#pattern) - Pattern matching utilities
28. [DevEx](#devex) - Developer experience (testing, debugging)
29. [Graphics Math](#graphics-math) - Vectors, matrices, quaternions
30. [SIMD](#simd) - SIMD vector operations
31. [Tensor](#tensor) - Tensor operations
32. [Autodiff](#autodiff) - Automatic differentiation
33. [Spatial](#spatial) - Spatial data structures
34. [Physics](#physics) - Physics simulation
35. [Geometric Algebra](#geometric-algebra) - Multivectors and rotors
36. [Dimensional](#dimensional) - Physical quantities with units
37. [ECS](#ecs) - Entity Component System
38. [Polycultural Text](#polycultural-text) - World-class text handling
39. [Text Intelligence](#text-intelligence) - AI-native text analysis
40. [Hologram](#hologram) - Emotional hologram types
41. [Experimental Crypto](#experimental-crypto) - Advanced cryptographic schemes
42. [Multibase](#multibase) - Multi-base encoding
43. [Audio](#audio) - Audio synthesis and music theory
44. [Spirituality](#spirituality) - Divination and sacred geometry
45. [Color](#color) - Color systems and synesthesia
46. [Protocol](#protocol) - Network protocol utilities
47. [Agent Tools](#agent-tools) - LLM tool definitions
48. [Agent LLM](#agent-llm) - LLM API integration
49. [Agent Memory](#agent-memory) - Agent memory systems

---

## Core

Essential functions for basic program operation.

### Constants

| Name | Type | Description |
|------|------|-------------|
| `i64·MAX` | `i64` | Maximum 64-bit signed integer (9223372036854775807) |
| `i64·MIN` | `i64` | Minimum 64-bit signed integer |
| `u64·MAX` | `i64` | Maximum 64-bit unsigned integer |
| `i32·MAX`, `i32·MIN` | `i64` | 32-bit integer bounds |
| `f64·INFINITY` | `f64` | Positive infinity |
| `f64·NEG_INFINITY` | `f64` | Negative infinity |
| `f64·NAN` | `f64` | Not-a-Number |

### Output Functions

```sigil
fn print(args...) -> ()
```
Print values without newline.

```sigil
fn println(args...) -> ()
```
Print values with newline.

```sigil
fn eprint(args...) -> ()
```
Print to stderr without newline.

```sigil
fn eprintln(args...) -> ()
```
Print to stderr with newline.

```sigil
fn dbg(value: T) -> T
```
Debug print a value and return it. Useful in pipelines.

### Assertions

```sigil
fn assert(condition: bool) -> ()
fn assert(condition: bool, message: String) -> ()
```
Assert a condition is true. Panics with optional message if false.

```sigil
fn panic(message: String) -> !
```
Halt execution with an error message.

```sigil
fn unreachable() -> !
fn todo() -> !
```
Mark unreachable code or unimplemented sections.

### Type Utilities

```sigil
fn type_of(value: T) -> String
```
Returns the type name of a value.

```sigil
fn clone(value: T) -> T
```
Deep clone a value.

```sigil
fn id(value: T) -> T
```
Identity function - returns input unchanged.

```sigil
fn default<T>() -> T
```
Returns the default value for a type.

### Constructors

```sigil
fn Vec·new<T>() -> Vec<T>
fn String·new() -> String
fn String·from(s: &str) -> String
fn HashMap·new<K, V>() -> HashMap<K, V>
fn HashSet·new<T>() -> HashSet<T>
fn Box·new<T>(value: T) -> Box<T>
fn Some<T>(value: T) -> Option<T>
fn Ok<T>(value: T) -> Result<T, E>
fn Err<E>(error: E) -> Result<T, E>
```

---

## Math

Mathematical operations including constants and functions.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `PI` | 3.14159... | Pi (π) |
| `TAU` | 6.28318... | Tau (2π) |
| `E` | 2.71828... | Euler's number |
| `PHI` | 1.61803... | Golden ratio |

### Basic Operations

```sigil
fn abs(x: Num) -> Num           // Absolute value
fn neg(x: Num) -> Num           // Negation
fn sign(x: Num) -> i64          // Sign (-1, 0, or 1)
fn min(a: Num, b: Num) -> Num   // Minimum
fn max(a: Num, b: Num) -> Num   // Maximum
fn clamp(x: Num, lo: Num, hi: Num) -> Num  // Clamp to range
```

### Rounding

```sigil
fn floor(x: f64) -> f64    // Round down
fn ceil(x: f64) -> f64     // Round up
fn round(x: f64) -> f64    // Round to nearest
fn trunc(x: f64) -> f64    // Truncate toward zero
fn fract(x: f64) -> f64    // Fractional part
```

### Powers and Roots

```sigil
fn sqrt(x: f64) -> f64          // Square root
fn cbrt(x: f64) -> f64          // Cube root
fn pow(base: f64, exp: f64) -> f64  // Power
fn exp(x: f64) -> f64           // e^x
```

### Logarithms

```sigil
fn ln(x: f64) -> f64       // Natural log
fn log(x: f64, base: f64) -> f64  // Log with base
fn log2(x: f64) -> f64     // Log base 2
fn log10(x: f64) -> f64    // Log base 10
```

### Trigonometry

```sigil
fn sin(x: f64) -> f64      fn asin(x: f64) -> f64
fn cos(x: f64) -> f64      fn acos(x: f64) -> f64
fn tan(x: f64) -> f64      fn atan(x: f64) -> f64
fn sinh(x: f64) -> f64     fn cosh(x: f64) -> f64
fn tanh(x: f64) -> f64     fn atan2(y: f64, x: f64) -> f64
```

### Number Theory

```sigil
fn gcd(a: i64, b: i64) -> i64      // Greatest common divisor
fn lcm(a: i64, b: i64) -> i64      // Least common multiple
fn factorial(n: i64) -> i64         // n!
fn is_prime(n: i64) -> bool         // Primality test
fn is_even(n: i64) -> bool
fn is_odd(n: i64) -> bool
```

### Checks

```sigil
fn is_nan(x: f64) -> bool
fn is_infinite(x: f64) -> bool
fn is_finite(x: f64) -> bool
```

---

## Collections

Operations on arrays, maps, and sets.

### Array Operations

```sigil
fn len(arr: Array<T>) -> i64
fn is_empty(arr: Array<T>) -> bool
fn push(arr: &mut Array<T>, value: T) -> ()
fn pop(arr: &mut Array<T>) -> Option<T>
fn first(arr: Array<T>) -> Option<T>
fn last(arr: Array<T>) -> Option<T>
fn get(arr: Array<T>, index: i64) -> Option<T>
fn set(arr: &mut Array<T>, index: i64, value: T) -> ()
fn insert(arr: &mut Array<T>, index: i64, value: T) -> ()
fn remove(arr: &mut Array<T>, index: i64) -> T
fn clear(arr: &mut Array<T>) -> ()
```

### Array Transformations

```sigil
fn slice(arr: Array<T>, start: i64, end: i64) -> Array<T>
fn concat(a: Array<T>, b: Array<T>) -> Array<T>
fn flatten(nested: Array<Array<T>>) -> Array<T>
fn reverse(arr: Array<T>) -> Array<T>
fn sort(arr: Array<T>) -> Array<T>
fn sort_desc(arr: Array<T>) -> Array<T>
fn unique(arr: Array<T>) -> Array<T>
fn repeat(value: T, count: i64) -> Array<T>
```

### Array Queries

```sigil
fn contains(arr: Array<T>, value: T) -> bool
fn index_of(arr: Array<T>, value: T) -> Option<i64>
fn nth(arr: Array<T>, n: i64) -> Option<T>
fn middle(arr: Array<T>) -> Option<T>
fn supremum(arr: Array<Num>) -> Num   // Max element
fn infimum(arr: Array<Num>) -> Num    // Min element
```

### Iteration Helpers

```sigil
fn range(start: i64, end: i64) -> Array<i64>
fn range_inclusive(start: i64, end: i64) -> Array<i64>
fn enumerate(arr: Array<T>) -> Array<(i64, T)>
fn zip(a: Array<A>, b: Array<B>) -> Array<(A, B)>
fn zip_with(a: Array<A>, b: Array<B>, f: fn(A, B) -> C) -> Array<C>
fn chunk(arr: Array<T>, size: i64) -> Array<Array<T>>
fn take(arr: Array<T>, n: i64) -> Array<T>
fn skip(arr: Array<T>, n: i64) -> Array<T>
fn choice(arr: Array<T>) -> T   // Random element
fn sample(arr: Array<T>, n: i64) -> Array<T>  // Random sample
fn shuffle(arr: Array<T>) -> Array<T>
```

### Map Operations

```sigil
fn map_new<K, V>() -> Map<K, V>
fn map_get(map: Map<K, V>, key: K) -> Option<V>
fn map_set(map: &mut Map<K, V>, key: K, value: V) -> ()
fn map_has(map: Map<K, V>, key: K) -> bool
fn map_remove(map: &mut Map<K, V>, key: K) -> Option<V>
fn map_keys(map: Map<K, V>) -> Array<K>
fn map_values(map: Map<K, V>) -> Array<V>
fn map_len(map: Map<K, V>) -> i64
fn map_clear(map: &mut Map<K, V>) -> ()
```

### Set Operations

```sigil
fn set_new<T>() -> Set<T>
fn set_add(set: &mut Set<T>, value: T) -> bool
fn set_has(set: Set<T>, value: T) -> bool
fn set_remove(set: &mut Set<T>, value: T) -> bool
fn set_len(set: Set<T>) -> i64
fn set_clear(set: &mut Set<T>) -> ()
fn set_to_array(set: Set<T>) -> Array<T>
```

---

## String

String manipulation functions.

### Basic Operations

```sigil
fn len(s: String) -> i64           // Byte length
fn char_count(s: String) -> i64    // Character count
fn byte_count(s: String) -> i64    // Byte count
fn is_empty(s: String) -> bool
fn char_at(s: String, index: i64) -> Option<char>
fn substring(s: String, start: i64, end: i64) -> String
```

### Case Conversion

```sigil
fn upper(s: String) -> String
fn lower(s: String) -> String
fn capitalize(s: String) -> String  // First char upper
```

### Searching

```sigil
fn find(s: String, pattern: String) -> Option<i64>
fn index_of(s: String, pattern: String) -> Option<i64>
fn last_index_of(s: String, pattern: String) -> Option<i64>
fn contains(s: String, pattern: String) -> bool
fn starts_with(s: String, prefix: String) -> bool
fn ends_with(s: String, suffix: String) -> bool
fn count(s: String, pattern: String) -> i64
```

### Modification

```sigil
fn replace(s: String, old: String, new: String) -> String
fn insert(s: String, index: i64, text: String) -> String
fn remove(s: String, start: i64, end: i64) -> String
fn repeat_str(s: String, count: i64) -> String
```

### Whitespace

```sigil
fn trim(s: String) -> String
fn trim_start(s: String) -> String
fn trim_end(s: String) -> String
fn pad_left(s: String, width: i64, char: char) -> String
fn pad_right(s: String, width: i64, char: char) -> String
```

### Splitting and Joining

```sigil
fn split(s: String, delimiter: String) -> Array<String>
fn lines(s: String) -> Array<String>
fn words(s: String) -> Array<String>
fn join(arr: Array<String>, separator: String) -> String
fn concat_all(arr: Array<String>) -> String
```

### Character Utilities

```sigil
fn chars(s: String) -> Array<char>
fn bytes(s: String) -> Array<i64>
fn char_code_at(s: String, index: i64) -> i64
fn from_char_code(code: i64) -> String
fn graphemes(s: String) -> Array<String>
fn grapheme_count(s: String) -> i64
```

### Character Tests

```sigil
fn is_alpha(s: String) -> bool
fn is_digit(s: String) -> bool
fn is_alnum(s: String) -> bool
fn is_space(s: String) -> bool
fn is_blank(s: String) -> bool
```

### Unicode Normalization

```sigil
fn nfc(s: String) -> String    // NFC normalization
fn nfd(s: String) -> String    // NFD normalization
fn nfkc(s: String) -> String   // NFKC normalization
fn nfkd(s: String) -> String   // NFKD normalization
fn is_nfc(s: String) -> bool
fn is_nfd(s: String) -> bool
```

### Comparison

```sigil
fn compare(a: String, b: String) -> i64
fn compare_ignore_case(a: String, b: String) -> i64
```

---

## Evidence

Sigil's unique evidentiality system for tracking information sources.

### Evidentiality Markers

```sigil
fn known(value: T) -> !T          // Mark as directly known/witnessed
fn reported(value: T) -> ~T       // Mark as reported/hearsay
fn uncertain(value: T) -> ?T      // Mark as uncertain/inferred
fn paradox(value: T) -> ¡T        // Mark as contradictory
```

### Evidence Queries

```sigil
fn evidence_of(value: T) -> String    // Get evidentiality marker
fn is_known(value: T) -> bool
fn is_reported(value: T) -> bool
fn is_uncertain(value: T) -> bool
fn is_paradox(value: T) -> bool
```

### Evidence Operations

```sigil
fn strip_evidence(value: T) -> T      // Remove evidentiality
fn combine_evidence(a: T, b: T) -> T  // Combine evidence levels
fn verify(value: T) -> T              // Attempt to verify
fn trust(value: T) -> T               // Trust reported as known
```

---

## Affect

Emotional/affective type system for AI-native sentiment.

### Creating Affective Values

```sigil
fn positive(value: T) -> T⁺         // Positive affect
fn negative(value: T) -> T⁻         // Negative affect
fn neutral(value: T) -> T⁰          // Neutral affect
```

### Specific Emotions

```sigil
fn joyful(value: T) -> T
fn sad(value: T) -> T
fn angry(value: T) -> T
fn fearful(value: T) -> T
fn surprised(value: T) -> T
fn loving(value: T) -> T
```

### Formality

```sigil
fn formal(value: T) -> T
fn informal(value: T) -> T
fn sarcastic(value: T) -> T
```

### Affect Queries

```sigil
fn affect_of(value: T) -> String
fn emotion_of(value: T) -> String
fn confidence_of(value: T) -> f64
fn is_positive(value: T) -> bool
fn is_negative(value: T) -> bool
fn is_formal(value: T) -> bool
fn is_informal(value: T) -> bool
fn is_sarcastic(value: T) -> bool
```

### Affect Operations

```sigil
fn strip_affect(value: T) -> T
fn intensify(value: T) -> T
fn dampen(value: T) -> T
fn maximize(value: T) -> T
fn with_affect(value: T, affect: String) -> T
```

### Confidence Levels

```sigil
fn high_confidence(value: T) -> T
fn medium_confidence(value: T) -> T
fn low_confidence(value: T) -> T
```

---

## Iter

Functional iteration operations, designed for pipe operators.

```sigil
fn sum(arr: Array<Num>) -> Num
fn product(arr: Array<Num>) -> Num
fn mean(arr: Array<Num>) -> f64
fn median(arr: Array<Num>) -> f64
fn min_of(arr: Array<Num>) -> Num
fn max_of(arr: Array<Num>) -> Num
fn count(arr: Array<T>) -> i64
```

### Predicates

```sigil
fn all(arr: Array<T>, pred: fn(T) -> bool) -> bool
fn any(arr: Array<T>, pred: fn(T) -> bool) -> bool
fn none(arr: Array<T>, pred: fn(T) -> bool) -> bool
```

### Usage with Pipes

```sigil
let result = [1, 2, 3, 4, 5]
    |> filter(|x| x > 2)
    |> map(|x| x * 2)
    |> sum();  // 24
```

---

## IO

File and console I/O operations.

### File Operations

```sigil
fn read_file(path: String) -> Result<String, Error>
fn write_file(path: String, content: String) -> Result<(), Error>
fn append_file(path: String, content: String) -> Result<(), Error>
fn read_lines(path: String) -> Result<Array<String>, Error>
fn file_exists(path: String) -> bool
```

### Environment

```sigil
fn env(name: String) -> Option<String>
fn env_or(name: String, default: String) -> String
fn args() -> Array<String>
fn cwd() -> String
```

---

## Time

Time and duration operations.

### Current Time

```sigil
fn now() -> Instant
fn now_secs() -> i64        // Unix timestamp in seconds
fn now_micros() -> i64      // Unix timestamp in microseconds
```

### Duration

```sigil
fn sleep(millis: i64) -> ()
fn timer_start() -> Instant
```

### Constants

```sigil
UNIX_EPOCH: Instant    // January 1, 1970 00:00:00 UTC
```

---

## Random

Random number generation.

```sigil
fn random() -> f64                        // [0.0, 1.0)
fn random_int(min: i64, max: i64) -> i64  // [min, max]
fn sample(arr: Array<T>, n: i64) -> Array<T>
fn shuffle(arr: Array<T>) -> Array<T>
```

---

## Convert

Type conversion functions.

### To Number

```sigil
fn to_int(value: T) -> i64
fn to_float(value: T) -> f64
fn parse_int(s: String) -> Result<i64, Error>
fn parse_int(s: String, radix: i64) -> Result<i64, Error>
```

### To String

```sigil
fn to_string(value: T) -> String
fn hex(n: i64) -> String      // "0xff"
fn bin(n: i64) -> String      // "0b1010"
fn oct(n: i64) -> String      // "0o77"
```

### Character Conversion

```sigil
fn to_char(code: i64) -> char
fn char_code(c: char) -> i64
fn from_char_code(code: i64) -> String
```

### Collection Conversion

```sigil
fn to_array(value: T) -> Array<T>
fn to_tuple(arr: Array<T>) -> Tuple
fn to_bool(value: T) -> bool
```

---

## JSON

JSON parsing and serialization.

```sigil
fn json_parse(s: String) -> Result<Value, Error>
fn json_stringify(value: T) -> String
fn json_pretty(value: T) -> String     // Pretty-printed
fn json_get(json: Value, path: String) -> Option<Value>
fn json_set(json: &mut Value, path: String, value: T) -> ()
```

---

## FS

File system operations.

### File Operations

```sigil
fn fs_read(path: String) -> Result<String, Error>
fn fs_read_bytes(path: String) -> Result<Array<u8>, Error>
fn fs_write(path: String, content: String) -> Result<(), Error>
fn fs_append(path: String, content: String) -> Result<(), Error>
fn fs_copy(src: String, dst: String) -> Result<(), Error>
fn fs_rename(old: String, new: String) -> Result<(), Error>
fn fs_remove(path: String) -> Result<(), Error>
```

### Directory Operations

```sigil
fn fs_list(path: String) -> Result<Array<String>, Error>
fn fs_mkdir(path: String) -> Result<(), Error>
```

### Queries

```sigil
fn fs_exists(path: String) -> bool
fn fs_is_file(path: String) -> bool
fn fs_is_dir(path: String) -> bool
fn fs_size(path: String) -> Result<i64, Error>
```

### Path Operations

```sigil
fn path_join(a: String, b: String) -> String
fn path_parent(path: String) -> Option<String>
fn path_filename(path: String) -> Option<String>
fn path_extension(path: String) -> Option<String>
```

### Standard Library Paths

```sigil
fn PathBuf·from(s: String) -> PathBuf
fn Path·new(s: String) -> Path
fn File·open(path: String) -> Result<File, Error>
fn File·create(path: String) -> Result<File, Error>
```

---

## Crypto

Cryptographic operations.

### Hashing

```sigil
fn sha256(data: String) -> String
fn sha512(data: String) -> String
fn sha3_256(data: String) -> String
fn sha3_512(data: String) -> String
fn blake3(data: String) -> String
fn blake3_keyed(data: String, key: String) -> String
fn md5(data: String) -> String   // For compatibility only
```

### HMAC

```sigil
fn hmac_sha256(data: String, key: String) -> String
fn hmac_sha512(data: String, key: String) -> String
fn hmac_verify(data: String, key: String, expected: String) -> bool
```

### Encoding

```sigil
fn base64_encode(data: String) -> String
fn base64_decode(data: String) -> Result<String, Error>
fn hex_encode(data: String) -> String
fn hex_decode(data: String) -> Result<String, Error>
```

### Password Hashing

```sigil
fn argon2_hash(password: String) -> String
fn argon2_verify(password: String, hash: String) -> bool
fn pbkdf2_derive(password: String, salt: String, iterations: i64) -> String
```

### Key Derivation

```sigil
fn hkdf_expand(key: String, info: String, length: i64) -> String
fn generate_key(length: i64) -> String
fn secure_random_bytes(length: i64) -> Array<u8>
fn secure_random_hex(length: i64) -> String
```

### Symmetric Encryption

```sigil
fn aes_gcm_encrypt(plaintext: String, key: String) -> String
fn aes_gcm_decrypt(ciphertext: String, key: String) -> Result<String, Error>
fn chacha20_encrypt(plaintext: String, key: String) -> String
fn chacha20_decrypt(ciphertext: String, key: String) -> Result<String, Error>
```

### Asymmetric Encryption

```sigil
fn ed25519_keygen() -> (String, String)  // (public, private)
fn ed25519_sign(message: String, private_key: String) -> String
fn ed25519_verify(message: String, signature: String, public_key: String) -> bool
fn x25519_keygen() -> (String, String)
fn x25519_exchange(private_key: String, public_key: String) -> String
```

### Utilities

```sigil
fn constant_time_eq(a: String, b: String) -> bool
fn crypto_info() -> String
```

---

## Regex

Regular expression operations.

```sigil
fn regex_match(pattern: String, text: String) -> bool
fn regex_find(pattern: String, text: String) -> Option<String>
fn regex_find_all(pattern: String, text: String) -> Array<String>
fn regex_captures(pattern: String, text: String) -> Array<String>
fn regex_replace(pattern: String, text: String, replacement: String) -> String
fn regex_replace_all(pattern: String, text: String, replacement: String) -> String
fn regex_split(pattern: String, text: String) -> Array<String>
```

---

## UUID

UUID generation and validation.

```sigil
fn uuid_v4() -> String         // Random UUID
fn uuid_nil() -> String        // Nil UUID (all zeros)
fn uuid_parse(s: String) -> Result<String, Error>
fn uuid_is_valid(s: String) -> bool
```

---

## System

System and environment operations.

```sigil
fn platform() -> String        // "linux", "macos", "windows"
fn arch() -> String            // "x86_64", "aarch64"
fn hostname() -> String
fn pid() -> i64
fn num_cpus·get() -> i64
fn num_cpus·get_physical() -> i64
```

### Environment

```sigil
fn env_get(name: String) -> Option<String>
fn env_set(name: String, value: String) -> ()
fn env_remove(name: String) -> ()
fn env_vars() -> Map<String, String>
fn args() -> Array<String>
fn temp_dir() -> String
```

### Process Control

```sigil
fn exit(code: i64) -> !
fn shell(command: String) -> Result<String, Error>
fn cwd() -> String
fn chdir(path: String) -> Result<(), Error>
```

---

## Stats

Statistical functions.

```sigil
fn mean(data: Array<f64>) -> f64
fn median(data: Array<f64>) -> f64
fn mode(data: Array<f64>) -> f64
fn variance(data: Array<f64>) -> f64
fn stddev(data: Array<f64>) -> f64
fn range(data: Array<f64>) -> f64
fn percentile(data: Array<f64>, p: f64) -> f64
fn correlation(x: Array<f64>, y: Array<f64>) -> f64
fn zscore(data: Array<f64>, value: f64) -> f64
```

---

## Matrix

Matrix operations.

```sigil
fn matrix_new(rows: i64, cols: i64) -> Matrix
fn matrix_identity(size: i64) -> Matrix
fn matrix_add(a: Matrix, b: Matrix) -> Matrix
fn matrix_sub(a: Matrix, b: Matrix) -> Matrix
fn matrix_mul(a: Matrix, b: Matrix) -> Matrix
fn matrix_scale(m: Matrix, s: f64) -> Matrix
fn matrix_transpose(m: Matrix) -> Matrix
fn matrix_det(m: Matrix) -> f64
fn matrix_trace(m: Matrix) -> f64
fn matrix_dot(a: Matrix, b: Matrix) -> f64
```

---

## Concurrency

Threading, channels, and async operations.

### Threads

```sigil
fn std·thread·spawn(f: fn() -> T) -> JoinHandle<T>
fn thread_join(handle: JoinHandle<T>) -> T
fn thread_sleep(millis: i64) -> ()
fn thread_yield() -> ()
fn thread_id() -> i64
fn thread_spawn_detached(f: fn() -> ()) -> ()
```

### Channels

```sigil
fn channel_new<T>() -> (Sender<T>, Receiver<T>)
fn channel_send(sender: Sender<T>, value: T) -> Result<(), Error>
fn channel_recv(receiver: Receiver<T>) -> Result<T, Error>
fn channel_recv_timeout(receiver: Receiver<T>, millis: i64) -> Result<T, Error>
fn channel_try_recv(receiver: Receiver<T>) -> Option<T>
```

### Synchronization

```sigil
fn mutex_new<T>(value: T) -> Mutex<T>
fn mutex_lock(mutex: Mutex<T>) -> MutexGuard<T>
fn mutex_unlock(guard: MutexGuard<T>) -> ()
fn RwLock·new<T>(value: T) -> RwLock<T>
fn Arc·new<T>(value: T) -> Arc<T>
```

### Atomics

```sigil
fn AtomicU64·new(value: u64) -> AtomicU64
fn AtomicBool·new(value: bool) -> AtomicBool
fn atomic_load(atomic: Atomic<T>) -> T
fn atomic_store(atomic: Atomic<T>, value: T) -> ()
fn atomic_add(atomic: AtomicU64, value: u64) -> u64
fn atomic_cas(atomic: Atomic<T>, expected: T, new: T) -> bool
```

### Async/Futures

```sigil
fn future_ready<T>(value: T) -> Future<T>
fn future_pending<T>() -> Future<T>
fn poll_future(future: Future<T>) -> Poll<T>
fn is_ready(future: Future<T>) -> bool
fn join_futures(futures: Array<Future<T>>) -> Future<Array<T>>
fn race_futures(futures: Array<Future<T>>) -> Future<T>
fn async_sleep(millis: i64) -> Future<()>
```

### Parallel Operations

```sigil
fn parallel_map<T, U>(arr: Array<T>, f: fn(T) -> U) -> Array<U>
fn parallel_for(range: Range, f: fn(i64) -> ()) -> ()
```

### Actors

```sigil
fn spawn_actor(name: String, handler: fn(Message) -> ()) -> Actor
fn send_to_actor(actor: Actor, message: Message) -> ()
fn tell_actor(actor: Actor, message: Message) -> ()
fn recv_from_actor(actor: Actor) -> Option<Message>
fn get_actor_name(actor: Actor) -> String
fn get_actor_msg_count(actor: Actor) -> i64
fn get_actor_pending(actor: Actor) -> i64
```

### TCP Networking

```sigil
fn TcpListener·bind(addr: String) -> Result<TcpListener, Error>
fn TcpStream·read(stream: TcpStream, buffer: &mut Array<u8>) -> Result<i64, Error>
fn TcpStream·write_all(stream: TcpStream, data: Array<u8>) -> Result<(), Error>
fn TcpStream·flush(stream: TcpStream) -> Result<(), Error>
fn TcpStream·peer_addr(stream: TcpStream) -> String
fn BufReader·new(stream: TcpStream) -> BufReader
fn BufReader·read_line(reader: BufReader) -> Result<String, Error>
```

---

## DevEx

Developer experience utilities for testing and debugging.

### Assertions

```sigil
fn assert_eq(a: T, b: T) -> ()
fn assert_ne(a: T, b: T) -> ()
fn assert_true(condition: bool) -> ()
fn assert_false(condition: bool) -> ()
fn assert_null(value: T) -> ()
fn assert_not_null(value: T) -> ()
fn assert_gt(a: T, b: T) -> ()
fn assert_lt(a: T, b: T) -> ()
fn assert_ge(a: T, b: T) -> ()
fn assert_le(a: T, b: T) -> ()
fn assert_len(collection: T, expected: i64) -> ()
fn assert_contains(collection: T, value: V) -> ()
fn assert_type(value: T, type_name: String) -> ()
fn assert_match(value: String, pattern: String) -> ()
```

### Debugging

```sigil
fn dbg(value: T) -> T          // Debug print and return
fn debug(value: T) -> ()       // Debug print
fn inspect(value: T) -> String // Get debug representation
fn pp(value: T) -> ()          // Pretty print
fn trace(message: String) -> ()
```

### Profiling

```sigil
fn measure(f: fn() -> T) -> (T, Duration)
fn profile(name: String, f: fn() -> T) -> T
```

### Testing

```sigil
fn test(name: String, f: fn() -> ()) -> ()
fn skip(reason: String) -> ()
```

### Markers

```sigil
fn todo(message: String) -> !
fn unimplemented(message: String) -> !
fn unreachable(message: String) -> !
fn deprecated(message: String) -> ()
```

### Introspection

```sigil
fn version() -> String
fn help(topic: String) -> String
fn list_builtins() -> Array<String>
```

---

## Functional

Functional programming utilities.

```sigil
fn identity<T>(x: T) -> T
fn const_fn<T>(x: T) -> fn() -> T
fn flip<A, B, C>(f: fn(A, B) -> C) -> fn(B, A) -> C
fn partial<A, B, C>(f: fn(A, B) -> C, a: A) -> fn(B) -> C
fn apply<T, U>(f: fn(T) -> U, x: T) -> U
fn complement<T>(pred: fn(T) -> bool) -> fn(T) -> bool
fn negate<T>(pred: fn(T) -> bool) -> fn(T) -> bool
fn juxt<T, U>(funcs: Array<fn(T) -> U>) -> fn(T) -> Array<U>
fn tap<T>(x: T, f: fn(T) -> ()) -> T
fn thunk<T>(f: fn() -> T) -> Thunk<T>
fn force<T>(thunk: Thunk<T>) -> T
```

---

## Benchmark

Performance measurement.

```sigil
fn bench(name: String, f: fn() -> ()) -> BenchResult
fn compare_bench(name: String, funcs: Array<fn() -> ()>) -> Array<BenchResult>
fn time_it(f: fn() -> T) -> (T, Duration)
fn stopwatch_start() -> Stopwatch
fn stopwatch_elapsed(sw: Stopwatch) -> Duration
fn memory_usage() -> i64
```

---

## Graphics Math

3D graphics mathematics: vectors, matrices, quaternions.

### Vectors

```sigil
fn vec2(x: f64, y: f64) -> Vec2
fn vec3(x: f64, y: f64, z: f64) -> Vec3
fn vec4(x: f64, y: f64, z: f64, w: f64) -> Vec4

fn vec3_add(a: Vec3, b: Vec3) -> Vec3
fn vec3_sub(a: Vec3, b: Vec3) -> Vec3
fn vec3_scale(v: Vec3, s: f64) -> Vec3
fn vec3_dot(a: Vec3, b: Vec3) -> f64
fn vec3_cross(a: Vec3, b: Vec3) -> Vec3
fn vec3_length(v: Vec3) -> f64
fn vec3_normalize(v: Vec3) -> Vec3
fn vec3_lerp(a: Vec3, b: Vec3, t: f64) -> Vec3
fn vec3_reflect(v: Vec3, n: Vec3) -> Vec3
fn vec3_refract(v: Vec3, n: Vec3, eta: f64) -> Vec3
```

### Matrices

```sigil
fn mat3_identity() -> Mat3
fn mat3_inverse(m: Mat3) -> Mat3
fn mat3_transpose(m: Mat3) -> Mat3
fn mat3_mul(a: Mat3, b: Mat3) -> Mat3

fn mat4_identity() -> Mat4
fn mat4_inverse(m: Mat4) -> Mat4
fn mat4_transpose(m: Mat4) -> Mat4
fn mat4_mul(a: Mat4, b: Mat4) -> Mat4
fn mat4_translate(v: Vec3) -> Mat4
fn mat4_scale(v: Vec3) -> Mat4
fn mat4_rotate_x(angle: f64) -> Mat4
fn mat4_rotate_y(angle: f64) -> Mat4
fn mat4_rotate_z(angle: f64) -> Mat4
fn mat4_perspective(fov: f64, aspect: f64, near: f64, far: f64) -> Mat4
fn mat4_ortho(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Mat4
fn mat4_look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4
fn mat4_transform(m: Mat4, v: Vec4) -> Vec4
```

### Quaternions

```sigil
fn quat_identity() -> Quat
fn quat_new(x: f64, y: f64, z: f64, w: f64) -> Quat
fn quat_from_axis_angle(axis: Vec3, angle: f64) -> Quat
fn quat_from_euler(pitch: f64, yaw: f64, roll: f64) -> Quat
fn quat_to_euler(q: Quat) -> (f64, f64, f64)
fn quat_to_mat4(q: Quat) -> Mat4
fn quat_mul(a: Quat, b: Quat) -> Quat
fn quat_conjugate(q: Quat) -> Quat
fn quat_inverse(q: Quat) -> Quat
fn quat_normalize(q: Quat) -> Quat
fn quat_rotate(q: Quat, v: Vec3) -> Vec3
fn quat_slerp(a: Quat, b: Quat, t: f64) -> Quat
```

---

## Polycultural Text

World-class text handling for all writing systems.

### Script Detection

```sigil
fn script(s: String) -> String           // Dominant script
fn scripts(s: String) -> Array<String>   // All scripts present
fn char_script(c: char) -> String
fn script_ratio(s: String, script: String) -> f64
fn script_runs(s: String) -> Array<(String, String)>  // (text, script) pairs
```

### Script Tests

```sigil
fn is_latin(s: String) -> bool
fn is_arabic(s: String) -> bool
fn is_cjk(s: String) -> bool
fn is_cyrillic(s: String) -> bool
fn is_devanagari(s: String) -> bool
fn is_greek(s: String) -> bool
fn is_hebrew(s: String) -> bool
fn is_thai(s: String) -> bool
fn is_hangul(s: String) -> bool
fn is_hiragana(s: String) -> bool
fn is_katakana(s: String) -> bool
fn is_script(s: String, script: String) -> bool
```

### Bidirectional Text

```sigil
fn is_rtl(s: String) -> bool
fn is_ltr(s: String) -> bool
fn is_bidi(s: String) -> bool
fn text_direction(s: String) -> String  // "ltr", "rtl", "mixed"
fn bidi_reorder(s: String) -> String    // Reorder for display
```

### Locale-Aware Operations

```sigil
fn upper_locale(s: String, locale: String) -> String
fn lower_locale(s: String, locale: String) -> String
fn titlecase_locale(s: String, locale: String) -> String
fn compare_locale(a: String, b: String, locale: String) -> i64
fn sort_locale(arr: Array<String>, locale: String) -> Array<String>
fn case_fold(s: String) -> String
fn case_insensitive_eq(a: String, b: String) -> bool
```

### Word/Sentence Segmentation (ICU)

```sigil
fn words_icu(s: String) -> Array<String>
fn word_count_icu(s: String) -> i64
fn sentences(s: String) -> Array<String>
fn sentence_count(s: String) -> i64
fn word_boundaries(s: String) -> Array<i64>
```

### Emoji

```sigil
fn is_emoji(s: String) -> bool
fn extract_emoji(s: String) -> Array<String>
fn strip_emoji(s: String) -> String
```

### Diacritics

```sigil
fn has_diacritics(s: String) -> bool
fn strip_diacritics(s: String) -> String
fn normalize_accents(s: String) -> String
```

### Transliteration

```sigil
fn to_ascii(s: String) -> String        // Best-effort ASCII
fn transliterate(s: String) -> String   // Romanize non-Latin
fn slugify(s: String) -> String         // URL-safe slug
```

### Display Width

```sigil
fn display_width(s: String) -> i64      // Terminal column width
fn is_fullwidth(c: char) -> bool
fn pad_display(s: String, width: i64) -> String
```

### Locale Info

```sigil
fn locale_name(locale: String) -> String
fn supported_locales() -> Array<String>
```

---

## Text Intelligence

AI-native text analysis.

### String Similarity

```sigil
fn levenshtein(a: String, b: String) -> i64
fn damerau_levenshtein(a: String, b: String) -> i64
fn osa_distance(a: String, b: String) -> i64
fn jaro(a: String, b: String) -> f64
fn jaro_winkler(a: String, b: String) -> f64
fn sorensen_dice(a: String, b: String) -> f64
fn jaccard_similarity(a: String, b: String) -> f64
fn cosine_similarity(a: String, b: String) -> f64
```

### Phonetic Encoding

```sigil
fn soundex(s: String) -> String
fn metaphone(s: String) -> String
fn cologne_phonetic(s: String) -> String
fn soundex_match(a: String, b: String) -> bool
fn metaphone_match(a: String, b: String) -> bool
```

### Language Detection

```sigil
fn detect_language(s: String) -> (String, f64)  // (lang, confidence)
fn is_language(s: String, lang: String) -> bool
```

### Token Counting (LLM)

```sigil
fn token_count(s: String) -> i64                    // Default (cl100k)
fn token_count_model(s: String, model: String) -> i64
fn tokenize_ids(s: String) -> Array<i64>
fn tokenize_words(s: String) -> Array<String>
fn truncate_tokens(s: String, max: i64) -> String
fn estimate_cost(s: String, model: String) -> f64
```

### Stemming

```sigil
fn stem(word: String) -> String                     // English Porter
fn stem_language(word: String, lang: String) -> String
fn stem_all(words: Array<String>) -> Array<String>
```

### Stopwords

```sigil
fn is_stopword(word: String) -> bool                // English
fn is_stopword_language(word: String, lang: String) -> bool
fn remove_stopwords(s: String) -> String
```

### N-grams and Shingles

```sigil
fn ngrams(s: String, n: i64) -> Array<String>
fn char_ngrams(s: String, n: i64) -> Array<String>
fn shingles(s: String, size: i64) -> Array<String>
fn minhash_signature(s: String, num_hashes: i64) -> Array<i64>
fn text_fingerprint(s: String) -> String
```

### Fuzzy Matching

```sigil
fn fuzzy_match(query: String, candidates: Array<String>) -> Option<String>
fn fuzzy_search(query: String, candidates: Array<String>, limit: i64) -> Array<String>
```

### Text Analysis

```sigil
fn word_frequency(s: String) -> Map<String, i64>
fn reading_time(s: String) -> i64        // Minutes
fn speaking_time(s: String) -> i64       // Minutes
fn text_formality(s: String) -> f64      // 0.0-1.0
fn preprocess_text(s: String) -> String  // Normalize, lowercase, etc.
fn text_hash_vector(s: String) -> Array<f64>
fn text_similarity_embedding(a: String, b: String) -> f64
```

### Entity Extraction

```sigil
fn extract_emails(s: String) -> Array<String>
fn extract_urls(s: String) -> Array<String>
fn extract_hashtags(s: String) -> Array<String>
fn extract_mentions(s: String) -> Array<String>
fn extract_numbers(s: String) -> Array<f64>
fn extract_money(s: String) -> Array<String>
fn extract_dates(s: String) -> Array<String>
fn extract_keywords(s: String, limit: i64) -> Array<String>
fn extract_entities(s: String) -> Array<(String, String)>  // (entity, type)
```

### Sentiment Analysis

```sigil
fn sentiment_words(s: String) -> f64       // -1.0 to 1.0
fn sentiment_vader(s: String) -> Map<String, f64>
fn emotion_detect(s: String) -> String
fn is_sarcastic(s: String) -> bool
fn detect_sarcasm(s: String) -> f64
fn detect_irony(s: String) -> f64
fn intensity_score(s: String) -> f64
fn has_question(s: String) -> bool
fn has_exclamation(s: String) -> bool
```

---

## Agent Infrastructure

AI agent building blocks.

### Agent Tools

```sigil
fn tool_define(name: String, description: String, schema: Map) -> ()
fn tool_schema(name: String) -> Map
fn tool_call(name: String, args: Map) -> Result<Value, Error>
fn tool_list() -> Array<String>
fn tool_get(name: String) -> Option<Tool>
fn tool_remove(name: String) -> ()
fn tool_clear() -> ()
fn tool_schemas_all() -> Array<Map>
```

### Agent LLM

```sigil
fn llm_request(model: String, messages: Array<Message>) -> Result<String, Error>
fn llm_send(model: String, prompt: String) -> Result<String, Error>
fn llm_message(role: String, content: String) -> Message
fn llm_messages() -> Array<Message>
fn llm_with_system(system: String) -> LLMBuilder
fn llm_with_messages(messages: Array<Message>) -> LLMBuilder
fn llm_with_tools(tools: Array<Tool>) -> LLMBuilder
fn llm_parse_tool_call(response: String) -> Option<ToolCall>
fn llm_extract(response: String, format: String) -> Value
fn prompt_template(template: String) -> PromptTemplate
fn prompt_render(template: PromptTemplate, vars: Map) -> String
```

### Agent Memory

```sigil
fn memory_session(name: String) -> Session
fn memory_set(key: String, value: Value) -> ()
fn memory_get(key: String) -> Option<Value>
fn memory_clear() -> ()
fn memory_history_add(role: String, content: String) -> ()
fn memory_history_get() -> Array<Message>
fn memory_context_all() -> Map<String, Value>
fn memory_sessions_list() -> Array<String>
```

### Agent Planning

```sigil
fn plan_state_machine(states: Array<String>, initial: String) -> StateMachine
fn plan_add_transition(sm: StateMachine, from: String, to: String, condition: fn() -> bool) -> ()
fn plan_transition(sm: StateMachine, to: String) -> Result<(), Error>
fn plan_can_transition(sm: StateMachine, to: String) -> bool
fn plan_current_state(sm: StateMachine) -> String
fn plan_available_transitions(sm: StateMachine) -> Array<String>
fn plan_history(sm: StateMachine) -> Array<String>
fn plan_goal(description: String) -> Goal
fn plan_subgoals(goal: Goal, subgoals: Array<String>) -> Goal
fn plan_check_goal(goal: Goal) -> bool
fn plan_update_progress(goal: Goal, progress: f64) -> ()
```

### Agent Vectors

```sigil
fn vec_store() -> VecStore
fn vec_store_add(store: VecStore, id: String, embedding: Array<f64>, metadata: Map) -> ()
fn vec_store_search(store: VecStore, query: Array<f64>, limit: i64) -> Array<(String, f64)>
fn vec_embedding(text: String) -> Array<f64>
fn vec_search(store: VecStore, text: String, limit: i64) -> Array<(String, f64)>
fn vec_cosine_similarity(a: Array<f64>, b: Array<f64>) -> f64
fn vec_euclidean_distance(a: Array<f64>, b: Array<f64>) -> f64
fn vec_dot_product(a: Array<f64>, b: Array<f64>) -> f64
fn vec_normalize(v: Array<f64>) -> Array<f64>
```

### Agent Swarm

```sigil
fn swarm_create_agent(name: String, capabilities: Array<String>) -> Agent
fn swarm_remove_agent(agent: Agent) -> ()
fn swarm_list_agents() -> Array<Agent>
fn swarm_find_agents(capability: String) -> Array<Agent>
fn swarm_send_message(from: Agent, to: Agent, message: Message) -> ()
fn swarm_broadcast(from: Agent, message: Message) -> ()
fn swarm_receive_messages(agent: Agent) -> Array<Message>
fn swarm_get_state(agent: Agent) -> Map
fn swarm_set_state(agent: Agent, key: String, value: Value) -> ()
fn swarm_add_capability(agent: Agent, capability: String) -> ()
fn swarm_consensus(agents: Array<Agent>, proposal: Value) -> Result<Value, Error>
```

### Agent Reasoning

```sigil
fn reason_hypothesis(description: String) -> Hypothesis
fn reason_verify_hypothesis(h: Hypothesis, evidence: Array<Value>) -> bool
fn reason_constraint(name: String, predicate: fn(Value) -> bool) -> Constraint
fn reason_check_constraint(c: Constraint, value: Value) -> bool
fn reason_check_all(constraints: Array<Constraint>, value: Value) -> bool
fn reason_and(a: Constraint, b: Constraint) -> Constraint
fn reason_or(a: Constraint, b: Constraint) -> Constraint
fn reason_not(c: Constraint) -> Constraint
fn reason_implies(antecedent: Constraint, consequent: Constraint) -> Constraint
fn reason_chain(steps: Array<fn(Value) -> Value>) -> fn(Value) -> Value
fn reason_evaluate(chain: fn(Value) -> Value, input: Value) -> Value
fn reason_proof(goal: String, steps: Array<String>) -> Proof
```

---

## Additional Modules

The following modules are also available:

### Cycle (Music Theory)
- Pitch class operations, intervals, MIDI conversion

### SIMD
- SIMD vector operations for high-performance computing

### Tensor
- Tensor products, contractions, traces

### Autodiff
- Automatic differentiation (grad, hessian, jacobian)

### Spatial
- AABB collision, spatial hashing

### Physics
- Verlet integration, spring forces, constraints

### Geometric Algebra
- Multivectors, rotors, geometric products

### Dimensional
- Physical quantities with units (qty, conversions)

### ECS
- Entity Component System for game development

### Hologram
- Emotional hologram types

### Experimental Crypto
- Secret sharing, commitment schemes

### Multibase
- Base conversions (32, 58, sexagesimal, etc.)

### Audio
- Waveform synthesis, music theory

### Spirituality
- Divination, gematria, sacred geometry

### Color
- Color spaces, cultural colors, synesthesia

### Protocol
- HTTP, gRPC, WebSocket utilities

### Terminal
- Terminal colors, progress bars, formatting

---

## Usage Examples

### Basic I/O

```sigil
fn main() {
    println("Hello, World!");

    let name = env("USER").unwrap_or("stranger");
    println("Hello, " + name + "!");
}
```

### Working with Collections

```sigil
fn main() {
    let numbers = [1, 2, 3, 4, 5];

    let doubled = numbers
        |> map(|x| x * 2)
        |> filter(|x| x > 4)
        |> collect();

    println(doubled);  // [6, 8, 10]

    let total = numbers |> sum();  // 15
}
```

### File Operations

```sigil
fn main() {
    // Read a file
    let content = fs_read("data.txt")?;

    // Process lines
    let lines = content |> lines() |> filter(|l| !l.is_empty());

    // Write results
    fs_write("output.txt", lines.join("\n"))?;
}
```

### JSON Processing

```sigil
fn main() {
    let data = json_parse(r#"{"name": "Alice", "age": 30}"#)?;

    let name = json_get(data, "name");
    println("Name: " + name.unwrap_or("unknown"));
}
```

### Cryptographic Operations

```sigil
fn main() {
    let password = "secret123";
    let hash = argon2_hash(password);

    if argon2_verify(password, hash) {
        println("Password verified!");
    }

    let data = "Hello, World!";
    let encrypted = aes_gcm_encrypt(data, generate_key(32));
}
```

### Text Analysis

```sigil
fn main() {
    let text = "This is a sample text for analysis.";

    let lang = detect_language(text);
    println("Language: " + lang.0 + " (" + lang.1.to_string() + " confidence)");

    let tokens = token_count(text);
    println("Token count: " + tokens.to_string());

    let sentiment = sentiment_words(text);
    println("Sentiment: " + sentiment.to_string());
}
```

### AI Agent Building

```sigil
fn main() {
    // Define a tool
    tool_define("search", "Search the web", #{
        "query": {"type": "string", "description": "Search query"}
    });

    // Create LLM request
    let response = llm_send("claude-3", "What is the capital of France?")?;

    // Store in memory
    memory_set("last_response", response);
    memory_history_add("assistant", response);
}
```

---

## Version History

- **1.0-RC** (January 2026): Initial comprehensive stdlib documentation

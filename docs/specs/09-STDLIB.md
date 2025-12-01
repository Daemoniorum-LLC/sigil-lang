# Sigil Standard Library Specification

> *"The Grimoire contains all the spells a practitioner needs to begin their work."*

## 1. Philosophy: The Foundational Grimoire

The Sigil standard library (`std`) provides:
- **Core abstractions** — Types, traits, and functions for everyday programming
- **Platform abstraction** — Unified interface across OS and architectures
- **Safety utilities** — Tools for working with evidentiality and ownership
- **Cultural mathematics** — Poly-cultural numeric and algorithmic support
- **Concurrency primitives** — Async, threading, synchronization

The library is organized into **domains** (major areas) and **tomes** (modules within domains).

---

## 2. Library Structure

```
std
├── core           # Fundamental types and traits (no allocation)
│   ├── ops        # Operators (+, -, *, /, |, etc.)
│   ├── cmp        # Comparison traits
│   ├── convert    # Type conversion
│   ├── marker     # Marker traits (Send, Sync, Copy, etc.)
│   ├── mem        # Memory utilities
│   ├── ptr        # Raw pointers
│   └── slice      # Slice primitives
│
├── alloc          # Allocation-dependent types
│   ├── vec        # Dynamic arrays
│   ├── string     # UTF-8 strings
│   ├── boxed      # Heap allocation
│   ├── rc         # Reference counting
│   └── arc        # Atomic reference counting
│
├── collections    # Data structures
│   ├── HashMap
│   ├── HashSet
│   ├── BTreeMap
│   ├── BTreeSet
│   ├── VecDeque
│   ├── LinkedList
│   └── BinaryHeap
│
├── io             # Input/output
│   ├── Read
│   ├── Write
│   ├── BufRead
│   ├── Seek
│   └── stdio
│
├── fs             # Filesystem
│   ├── File
│   ├── Path
│   ├── DirEntry
│   └── metadata
│
├── net            # Networking
│   ├── TcpStream
│   ├── TcpListener
│   ├── UdpSocket
│   └── dns
│
├── sync           # Synchronization
│   ├── Mutex
│   ├── RwLock
│   ├── Condvar
│   ├── Barrier
│   └── atomic
│
├── thread         # Threading
│   ├── spawn
│   ├── JoinHandle
│   ├── park
│   └── scope
│
├── async          # Async runtime
│   ├── Future
│   ├── Stream
│   ├── spawn
│   └── select
│
├── time           # Time and duration
│   ├── Instant
│   ├── Duration
│   ├── SystemTime
│   └── calendars
│
├── math           # Mathematics (poly-cultural)
│   ├── numeral    # Base systems
│   ├── algebra    # Algebraic operations
│   ├── geometry   # Geometric primitives
│   ├── temporal   # Cyclical time
│   └── harmonic   # Music/sound math
│
├── text           # Text processing
│   ├── regex
│   ├── unicode
│   ├── encoding
│   └── format
│
├── crypto         # Cryptography
│   ├── hash
│   ├── cipher
│   ├── sign
│   └── random
│
├── serde          # Serialization
│   ├── json
│   ├── toml
│   ├── yaml
│   └── binary
│
├── ffi            # Foreign function interface
│   ├── c
│   ├── rust
│   └── platform
│
└── prelude        # Auto-imported essentials
```

---

## 3. Core Types and Traits

### 3.1 Prelude (Auto-Imported)

```sigil
// std·prelude is automatically imported
pub use core·ops·{Add, Sub, Mul, Div, Rem, Neg}
pub use core·ops·{BitAnd, BitOr, BitXor, Not, Shl, Shr}
pub use core·ops·{Index, IndexMut}
pub use core·ops·{Deref, DerefMut}
pub use core·cmp·{Eq, PartialEq, Ord, PartialOrd}
pub use core·convert·{From, Into, TryFrom, TryInto}
pub use core·marker·{Copy, Clone, Send, Sync, Sized}
pub use core·option·{Option, Some, None}
pub use core·result·{Result, Ok, Err}
pub use core·iter·{Iterator, IntoIterator}
pub use alloc·vec·Vec
pub use alloc·string·String
pub use alloc·boxed·Box
```

### 3.2 Option and Result

```sigil
enum Option<T?> {
    Some(T),
    None,
}

impl<T> Option<T?> {
    fn is_some(&self) -> bool
    fn is_none(&self) -> bool
    fn unwrap(self) -> T!        // Panics if None
    fn unwrap_or(self, default: T) -> T!
    fn unwrap_or_else(self, f: fn() -> T) -> T!
    fn map<U>(self, f: fn(T) -> U) -> Option<U?>
    fn and_then<U>(self, f: fn(T) -> Option<U?>) -> Option<U?>
    fn filter(self, pred: fn(&T) -> bool) -> Option<T?>
    fn ok_or<E>(self, err: E) -> Result<T!, E~>
}

enum Result<T!, E~> {
    Ok(T!),
    Err(E~),
}

impl<T, E> Result<T!, E~> {
    fn is_ok(&self) -> bool
    fn is_err(&self) -> bool
    fn unwrap(self) -> T!        // Panics if Err
    fn unwrap_err(self) -> E~    // Panics if Ok
    fn ok(self) -> Option<T?>
    fn err(self) -> Option<E?>
    fn map<U>(self, f: fn(T) -> U) -> Result<U!, E~>
    fn map_err<F>(self, f: fn(E) -> F) -> Result<T!, F~>
    fn and_then<U>(self, f: fn(T) -> Result<U!, E~>) -> Result<U!, E~>
}
```

### 3.3 Iterator

```sigil
trait Iterator {
    type Item

    fn next(&mut self) -> Option<Self·Item?>

    // Provided methods (morpheme-compatible)
    fn map<U>(self, f: fn(Self·Item) -> U) -> Map<Self, F>      // τ
    fn filter(self, p: fn(&Self·Item) -> bool) -> Filter<Self>  // φ
    fn fold<B>(self, init: B, f: fn(B, Self·Item) -> B) -> B    // ρ
    fn collect<C: FromIterator>(self) -> C
    fn enumerate(self) -> Enumerate<Self>
    fn zip<U>(self, other: U) -> Zip<Self, U>
    fn take(self, n: usize) -> Take<Self>
    fn skip(self, n: usize) -> Skip<Self>
    fn chain<U>(self, other: U) -> Chain<Self, U>
    fn flatten(self) -> Flatten<Self>
    fn sum(self) -> Self·Item where Self·Item: Add               // Σ
    fn product(self) -> Self·Item where Self·Item: Mul           // Π
    fn min(self) -> Option<Self·Item?> where Self·Item: Ord
    fn max(self) -> Option<Self·Item?> where Self·Item: Ord
    fn first(self) -> Option<Self·Item?>                         // α
    fn last(self) -> Option<Self·Item?>                          // Ω
    fn nth(self, n: usize) -> Option<Self·Item?>
    fn count(self) -> usize
    fn any(self, p: fn(Self·Item) -> bool) -> bool               // ∃
    fn all(self, p: fn(Self·Item) -> bool) -> bool               // ∀
    fn find(self, p: fn(&Self·Item) -> bool) -> Option<Self·Item?>
    fn position(self, p: fn(Self·Item) -> bool) -> Option<usize?>
}
```

---

## 4. Collections

### 4.1 Vec (Dynamic Array)

```sigil
struct Vec<T> {
    // ...
}

impl<T> Vec<T> {
    fn new() -> Vec<T>
    fn with_capacity(cap: usize) -> Vec<T>
    fn len(&self) -> usize
    fn capacity(&self) -> usize
    fn is_empty(&self) -> bool
    fn push(&mut self, value: T)
    fn pop(&mut self) -> Option<T?>
    fn insert(&mut self, index: usize, value: T)
    fn remove(&mut self, index: usize) -> T
    fn clear(&mut self)
    fn truncate(&mut self, len: usize)
    fn reserve(&mut self, additional: usize)
    fn shrink_to_fit(&mut self)
    fn as_slice(&self) -> &[T]
    fn as_mut_slice(&mut self) -> &mut [T]
    fn sort(&mut self) where T: Ord                              // σ
    fn sort_by(&mut self, cmp: fn(&T, &T) -> Ordering)
    fn reverse(&mut self)
    fn dedup(&mut self) where T: Eq
}

// Rune for creation
let v = vec![1, 2, 3, 4, 5]
let zeros = vec![0; 100]
```

### 4.2 HashMap

```sigil
struct HashMap<K, V> {
    // ...
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    fn new() -> HashMap<K, V>
    fn with_capacity(cap: usize) -> HashMap<K, V>
    fn len(&self) -> usize
    fn is_empty(&self) -> bool
    fn insert(&mut self, key: K, value: V) -> Option<V?>
    fn remove(&mut self, key: &K) -> Option<V?>
    fn get(&self, key: &K) -> Option<&V?>
    fn get_mut(&mut self, key: &K) -> Option<&mut V?>
    fn contains_key(&self, key: &K) -> bool
    fn keys(&self) -> impl Iterator<Item = &K>
    fn values(&self) -> impl Iterator<Item = &V>
    fn iter(&self) -> impl Iterator<Item = (&K, &V)>
    fn entry(&mut self, key: K) -> Entry<K, V>
}

// Usage
let mut map = HashMap·new()
map|insert("key", 42)
let value? = map|get("key")
```

### 4.3 Other Collections

```sigil
// HashSet
struct HashSet<T: Hash + Eq>
impl<T> HashSet<T> {
    fn insert(&mut self, value: T) -> bool
    fn remove(&mut self, value: &T) -> bool
    fn contains(&self, value: &T) -> bool
    fn union(&self, other: &HashSet<T>) -> impl Iterator          // ∪
    fn intersection(&self, other: &HashSet<T>) -> impl Iterator   // ∩
    fn difference(&self, other: &HashSet<T>) -> impl Iterator     // ∖
}

// BTreeMap (ordered)
struct BTreeMap<K: Ord, V>

// BTreeSet (ordered)
struct BTreeSet<T: Ord>

// VecDeque (double-ended queue)
struct VecDeque<T>
impl<T> VecDeque<T> {
    fn push_front(&mut self, value: T)
    fn push_back(&mut self, value: T)
    fn pop_front(&mut self) -> Option<T?>
    fn pop_back(&mut self) -> Option<T?>
}

// BinaryHeap (priority queue)
struct BinaryHeap<T: Ord>
impl<T: Ord> BinaryHeap<T> {
    fn push(&mut self, value: T)
    fn pop(&mut self) -> Option<T?>  // Returns max
    fn peek(&self) -> Option<&T?>
}
```

---

## 5. I/O and Filesystem

### 5.1 I/O Traits

```sigil
trait Read {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize!, IoError~>

    // Provided
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()!, IoError~>
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> Result<usize!, IoError~>
    fn read_to_string(&mut self, buf: &mut String) -> Result<usize!, IoError~>
    fn bytes(self) -> Bytes<Self>
}

trait Write {
    fn write(&mut self, buf: &[u8]) -> Result<usize!, IoError~>
    fn flush(&mut self) -> Result<()!, IoError~>

    // Provided
    fn write_all(&mut self, buf: &[u8]) -> Result<()!, IoError~>
    fn write_fmt(&mut self, fmt: Arguments) -> Result<()!, IoError~>
}

trait BufRead: Read {
    fn fill_buf(&mut self) -> Result<&[u8]!, IoError~>
    fn consume(&mut self, amt: usize)

    // Provided
    fn read_line(&mut self, buf: &mut String) -> Result<usize!, IoError~>
    fn lines(self) -> Lines<Self>
}

trait Seek {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64!, IoError~>
}
```

### 5.2 File Operations

```sigil
struct File {
    // ...
}

impl File {
    fn open(path: impl AsRef<Path>) -> Result<File!, IoError~>
    fn create(path: impl AsRef<Path>) -> Result<File!, IoError~>
    fn options() -> OpenOptions

    fn metadata(&self) -> Result<Metadata!, IoError~>
    fn set_permissions(&self, perm: Permissions) -> Result<()!, IoError~>
    fn sync_all(&self) -> Result<()!, IoError~>
}

impl Read for File { }
impl Write for File { }
impl Seek for File { }

// Convenience functions
fn fs·read(path: impl AsRef<Path>) -> Result<Vec<u8>!, IoError~>
fn fs·read_to_string(path: impl AsRef<Path>) -> Result<String!, IoError~>
fn fs·write(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> Result<()!, IoError~>
fn fs·copy(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<u64!, IoError~>
fn fs·rename(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()!, IoError~>
fn fs·remove_file(path: impl AsRef<Path>) -> Result<()!, IoError~>
fn fs·create_dir(path: impl AsRef<Path>) -> Result<()!, IoError~>
fn fs·create_dir_all(path: impl AsRef<Path>) -> Result<()!, IoError~>
fn fs·remove_dir(path: impl AsRef<Path>) -> Result<()!, IoError~>
fn fs·remove_dir_all(path: impl AsRef<Path>) -> Result<()!, IoError~>
fn fs·read_dir(path: impl AsRef<Path>) -> Result<ReadDir!, IoError~>
```

### 5.3 Path

```sigil
struct Path {
    // ...
}

impl Path {
    fn new(s: &str) -> &Path
    fn as_str(&self) -> &str
    fn parent(&self) -> Option<&Path?>
    fn file_name(&self) -> Option<&str?>
    fn file_stem(&self) -> Option<&str?>
    fn extension(&self) -> Option<&str?>
    fn join(&self, path: impl AsRef<Path>) -> PathBuf
    fn with_extension(&self, ext: &str) -> PathBuf
    fn exists(&self) -> bool
    fn is_file(&self) -> bool
    fn is_dir(&self) -> bool
    fn is_absolute(&self) -> bool
    fn is_relative(&self) -> bool
}

struct PathBuf {
    // Owned version of Path
}
```

---

## 6. Networking

```sigil
// TCP
struct TcpStream { }

impl TcpStream {
    fn connect(addr: impl ToSocketAddrs) -> Result<TcpStream!, IoError~>
    fn peer_addr(&self) -> Result<SocketAddr!, IoError~>
    fn local_addr(&self) -> Result<SocketAddr!, IoError~>
    fn shutdown(&self, how: Shutdown) -> Result<()!, IoError~>
    fn set_nodelay(&self, nodelay: bool) -> Result<()!, IoError~>
    fn set_read_timeout(&self, dur: Option<Duration>) -> Result<()!, IoError~>
    fn set_write_timeout(&self, dur: Option<Duration>) -> Result<()!, IoError~>
}

impl Read for TcpStream { }
impl Write for TcpStream { }

struct TcpListener { }

impl TcpListener {
    fn bind(addr: impl ToSocketAddrs) -> Result<TcpListener!, IoError~>
    fn accept(&self) -> Result<(TcpStream!, SocketAddr!)!, IoError~>
    fn incoming(&self) -> impl Iterator<Item = Result<TcpStream!, IoError~>>
    fn local_addr(&self) -> Result<SocketAddr!, IoError~>
}

// UDP
struct UdpSocket { }

impl UdpSocket {
    fn bind(addr: impl ToSocketAddrs) -> Result<UdpSocket!, IoError~>
    fn send_to(&self, buf: &[u8], addr: impl ToSocketAddrs) -> Result<usize!, IoError~>
    fn recv_from(&self, buf: &mut [u8]) -> Result<(usize!, SocketAddr!)!, IoError~>
    fn connect(&self, addr: impl ToSocketAddrs) -> Result<()!, IoError~>
    fn send(&self, buf: &[u8]) -> Result<usize!, IoError~>
    fn recv(&self, buf: &mut [u8]) -> Result<usize!, IoError~>
}
```

---

## 7. Concurrency (Summary)

```sigil
// See 06-CONCURRENCY.md for full details

mod thread {
    fn spawn<F, T>(f: F) -> JoinHandle<T>
    where F: FnOnce() -> T + Send + 'static, T: Send + 'static

    fn current() -> Thread
    fn park()
    fn sleep(dur: Duration)
    fn yield_now()
}

mod sync {
    struct Mutex<T>
    struct RwLock<T>
    struct Condvar
    struct Barrier
    struct Once

    mod atomic {
        struct AtomicBool
        struct AtomicI32
        struct AtomicI64
        struct AtomicUsize
        struct AtomicPtr<T>
    }
}

mod async {
    trait Future {
        type Output
        fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self·Output>
    }

    trait Stream {
        type Item
        fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self·Item>>
    }

    fn spawn<F>(future: F) -> JoinHandle<F·Output>
    where F: Future + Send + 'static
}
```

---

## 8. Time

```sigil
struct Duration {
    secs: u64,
    nanos: u32,
}

impl Duration {
    fn from_secs(secs: u64) -> Duration
    fn from_millis(millis: u64) -> Duration
    fn from_micros(micros: u64) -> Duration
    fn from_nanos(nanos: u64) -> Duration
    fn as_secs(&self) -> u64
    fn as_millis(&self) -> u128
    fn as_micros(&self) -> u128
    fn as_nanos(&self) -> u128
}

// Duration literals
let d = 5·sec
let d = 100·ms
let d = 50·μs
let d = 1000·ns

struct Instant {
    // Monotonic clock
}

impl Instant {
    fn now() -> Instant
    fn elapsed(&self) -> Duration
    fn duration_since(&self, earlier: Instant) -> Duration
}

struct SystemTime {
    // Wall clock
}

impl SystemTime {
    fn now() -> SystemTime
    fn duration_since(&self, earlier: SystemTime) -> Result<Duration!, SystemTimeError~>
    const UNIX_EPOCH: SystemTime
}
```

---

## 9. Mathematics (Poly-Cultural)

```sigil
mod math {
    // Basic numeric operations
    fn abs<T: Signed>(x: T) -> T
    fn min<T: Ord>(a: T, b: T) -> T
    fn max<T: Ord>(a: T, b: T) -> T
    fn clamp<T: Ord>(x: T, min: T, max: T) -> T

    // Floating point
    fn sqrt(x: f64) -> f64
    fn cbrt(x: f64) -> f64
    fn pow(x: f64, n: f64) -> f64
    fn exp(x: f64) -> f64
    fn ln(x: f64) -> f64
    fn log(x: f64, base: f64) -> f64
    fn log2(x: f64) -> f64
    fn log10(x: f64) -> f64
    fn sin(x: f64) -> f64
    fn cos(x: f64) -> f64
    fn tan(x: f64) -> f64
    fn asin(x: f64) -> f64
    fn acos(x: f64) -> f64
    fn atan(x: f64) -> f64
    fn atan2(y: f64, x: f64) -> f64
    fn sinh(x: f64) -> f64
    fn cosh(x: f64) -> f64
    fn tanh(x: f64) -> f64
    fn floor(x: f64) -> f64
    fn ceil(x: f64) -> f64
    fn round(x: f64) -> f64
    fn trunc(x: f64) -> f64
    fn fract(x: f64) -> f64

    // Constants
    const π: f64 = 3.14159265358979323846
    const τ: f64 = 6.28318530717958647692  // 2π
    const e: f64 = 2.71828182845904523536
    const φ: f64 = 1.61803398874989484820  // Golden ratio

    // See 05-MATHEMATICS.md for:
    mod numeral    // Multi-base systems
    mod algebra    // Symbolic manipulation
    mod geometry   // Geometric primitives
    mod temporal   // Cyclical time
    mod harmonic   // Music/sound
}
```

---

## 10. Text Processing

```sigil
mod text {
    // String operations
    impl str {
        fn len(&self) -> usize
        fn is_empty(&self) -> bool
        fn chars(&self) -> impl Iterator<Item = char>
        fn bytes(&self) -> impl Iterator<Item = u8>
        fn lines(&self) -> impl Iterator<Item = &str>
        fn split(&self, pat: impl Pattern) -> impl Iterator<Item = &str>
        fn trim(&self) -> &str
        fn trim_start(&self) -> &str
        fn trim_end(&self) -> &str
        fn to_lowercase(&self) -> String
        fn to_uppercase(&self) -> String
        fn contains(&self, pat: impl Pattern) -> bool
        fn starts_with(&self, pat: impl Pattern) -> bool
        fn ends_with(&self, pat: impl Pattern) -> bool
        fn find(&self, pat: impl Pattern) -> Option<usize?>
        fn replace(&self, from: impl Pattern, to: &str) -> String
        fn repeat(&self, n: usize) -> String
        fn parse<T: FromStr>(&self) -> Result<T!, T·Err~>
    }

    // Regex (compile-time validated)
    mod regex {
        struct Regex { }

        impl Regex {
            fn new(pattern: &str) -> Result<Regex!, RegexError~>
            fn is_match(&self, text: &str) -> bool
            fn find(&self, text: &str) -> Option<Match?>
            fn find_all(&self, text: &str) -> impl Iterator<Item = Match>
            fn captures(&self, text: &str) -> Option<Captures?>
            fn replace(&self, text: &str, rep: &str) -> String
            fn replace_all(&self, text: &str, rep: &str) -> String
        }

        // Compile-time regex
        let email = regex!(r"[\w.+-]+@[\w.-]+\.\w+")
    }

    // Unicode
    mod unicode {
        fn is_alphabetic(c: char) -> bool
        fn is_numeric(c: char) -> bool
        fn is_alphanumeric(c: char) -> bool
        fn is_whitespace(c: char) -> bool
        fn is_uppercase(c: char) -> bool
        fn is_lowercase(c: char) -> bool
        fn to_uppercase(c: char) -> impl Iterator<Item = char>
        fn to_lowercase(c: char) -> impl Iterator<Item = char>
    }

    // Formatting
    mod format {
        rune format! { }
        rune print! { }
        rune println! { }
        rune eprint! { }
        rune eprintln! { }
        rune write! { }
        rune writeln! { }
    }
}
```

---

## 11. Serialization

```sigil
mod serde {
    trait Serialize {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S·Ok!, S·Error~>
    }

    trait Deserialize<'de>: Sized {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self!, D·Error~>
    }

    // Derive macros
    //@ rune: derive(Serialize, Deserialize)
    struct MyData { }

    mod json {
        fn to_string<T: Serialize>(value: &T) -> Result<String!, Error~>
        fn to_string_pretty<T: Serialize>(value: &T) -> Result<String!, Error~>
        fn from_str<T: Deserialize>(s: &str) -> Result<T!, Error~>
        fn to_writer<W: Write, T: Serialize>(writer: W, value: &T) -> Result<()!, Error~>
        fn from_reader<R: Read, T: Deserialize>(reader: R) -> Result<T!, Error~>
    }

    mod toml {
        fn to_string<T: Serialize>(value: &T) -> Result<String!, Error~>
        fn from_str<T: Deserialize>(s: &str) -> Result<T!, Error~>
    }

    mod yaml {
        fn to_string<T: Serialize>(value: &T) -> Result<String!, Error~>
        fn from_str<T: Deserialize>(s: &str) -> Result<T!, Error~>
    }
}
```

---

## 12. Cryptography

```sigil
mod crypto {
    mod hash {
        trait Hasher {
            fn update(&mut self, data: &[u8])
            fn finalize(self) -> [u8]
        }

        struct Sha256
        struct Sha512
        struct Blake3
        struct Md5  // Deprecated, for compatibility only

        fn sha256(data: &[u8]) -> [u8; 32]
        fn sha512(data: &[u8]) -> [u8; 64]
        fn blake3(data: &[u8]) -> [u8; 32]
    }

    mod cipher {
        trait Cipher {
            fn encrypt(&self, plaintext: &[u8], nonce: &[u8]) -> Vec<u8>
            fn decrypt(&self, ciphertext: &[u8], nonce: &[u8]) -> Result<Vec<u8>!, CryptoError~>
        }

        struct AesGcm
        struct ChaCha20Poly1305
    }

    mod sign {
        trait Signer {
            fn sign(&self, message: &[u8]) -> Signature
        }

        trait Verifier {
            fn verify(&self, message: &[u8], signature: &Signature) -> bool
        }

        struct Ed25519
        struct Ecdsa
    }

    mod random {
        fn random<T>() -> T where T: Random
        fn random_bytes(n: usize) -> Vec<u8>
        fn thread_rng() -> impl Rng
    }
}
```

---

## 13. Error Handling

```sigil
mod error {
    trait Error: Display + Debug {
        fn source(&self) -> Option<&(dyn Error + 'static)> { None }
    }

    // Standard error types
    struct IoError { kind: IoErrorKind, message: String }
    struct ParseError { message: String }
    struct Utf8Error { valid_up_to: usize }
    struct NulError { position: usize }

    // Error construction
    rune bail! {
        ($msg:expr) => { return Err(Error·new($msg))~ }
        ($fmt:expr, $($arg:tt)*) => { return Err(Error·new(format!($fmt, $($arg)*)))~ }
    }

    rune ensure! {
        ($cond:expr, $msg:expr) => {
            if !$cond { bail!($msg) }
        }
    }

    // Context for errors
    trait Context<T, E> {
        fn context(self, msg: impl Display) -> Result<T!, ContextError~>
        fn with_context(self, f: impl FnOnce() -> impl Display) -> Result<T!, ContextError~>
    }

    impl<T, E: Error> Context<T, E> for Result<T, E> { }
}
```

---

## 14. Testing

```sigil
mod test {
    // Test attribute
    //@ rune: test
    fn test_example() {
        assert!(1 + 1 == 2)
    }

    // Assertions
    rune assert! {
        ($cond:expr) => { if !$cond { panic!("assertion failed: {}", stringify!($cond)) } }
        ($cond:expr, $msg:expr) => { if !$cond { panic!($msg) } }
    }

    rune assert_eq! {
        ($left:expr, $right:expr) => {
            if $left != $right {
                panic!("assertion failed: {} == {}\n  left: {:?}\n right: {:?}",
                       stringify!($left), stringify!($right), $left, $right)
            }
        }
    }

    rune assert_ne! {
        ($left:expr, $right:expr) => {
            if $left == $right {
                panic!("assertion failed: {} != {}\n  both: {:?}",
                       stringify!($left), stringify!($right), $left)
            }
        }
    }

    // Property-based testing
    //@ rune: quickcheck
    fn prop_reverse_reverse(xs: Vec<i32>) -> bool {
        xs·clone|reverse|reverse == xs
    }

    // Benchmarking
    //@ rune: bench
    fn bench_sort(b: &mut Bencher) {
        let data = random_vec(1000)
        b|iter(|| data·clone|sort)
    }
}
```

---

## 15. Logging and Diagnostics

```sigil
mod log {
    enum Level {
        Error,
        Warn,
        Info,
        Debug,
        Trace,
    }

    rune error! { ($($arg:tt)*) => { log!(Level·Error, $($arg)*) } }
    rune warn!  { ($($arg:tt)*) => { log!(Level·Warn, $($arg)*) } }
    rune info!  { ($($arg:tt)*) => { log!(Level·Info, $($arg)*) } }
    rune debug! { ($($arg:tt)*) => { log!(Level·Debug, $($arg)*) } }
    rune trace! { ($($arg:tt)*) => { log!(Level·Trace, $($arg)*) } }

    fn set_max_level(level: Level)
    fn max_level() -> Level
}

mod panic {
    fn set_hook(hook: Box<dyn Fn(&PanicInfo) + Send + Sync>)
    fn take_hook() -> Box<dyn Fn(&PanicInfo) + Send + Sync>

    rune panic! {
        () => { panic!("explicit panic") }
        ($msg:expr) => { std·panicking·panic($msg) }
        ($fmt:expr, $($arg:tt)*) => { std·panicking·panic(format!($fmt, $($arg)*)) }
    }

    rune unreachable! {
        () => { panic!("internal error: entered unreachable code") }
        ($msg:expr) => { panic!("internal error: {}", $msg) }
    }

    rune todo! {
        () => { panic!("not yet implemented") }
        ($msg:expr) => { panic!("not yet implemented: {}", $msg) }
    }

    rune unimplemented! {
        () => { panic!("not implemented") }
    }
}
```

# Sigil Memory Model Specification

## 1. Overview

Sigil provides memory safety without garbage collection through **ownership semantics** — a compile-time system that tracks resource lifetime and access. This design draws from Rust while adding polysynthetic ergonomics that reduce friction.

### Core Principles

1. **Every value has exactly one owner**
2. **Ownership can be transferred (moved) or borrowed**
3. **Borrows are either shared (&) or exclusive (&mut)**
4. **References cannot outlive their referents**
5. **No data races: shared XOR mutable**

---

## 2. Ownership

### 2.1 Single Ownership

Every value in Sigil has exactly one owner at any time:

```sigil
let a = [1, 2, 3]     // 'a' owns the vector
let b = a             // ownership moves to 'b'
// print(a)           // ERROR: 'a' no longer valid
print(b)              // OK: 'b' is the owner
```

### 2.2 Move Semantics

Assignment and function calls move ownership by default:

```sigil
fn consume(data: [i32]) {
    // 'data' is owned here
}

let v = [1, 2, 3]
consume(v)            // ownership moved to 'consume'
// v is no longer valid

// To keep using 'v', explicitly clone
let v = [1, 2, 3]
consume(v|clone)      // clone is moved, 'v' retained
print(v)              // OK
```

### 2.3 Copy Types

Small, stack-allocated types implement `Copy` and are duplicated rather than moved:

```sigil
let a: i32 = 42
let b = a             // 'a' is copied, not moved
print(a)              // OK: 'a' still valid
print(b)              // OK: 'b' is independent copy

// Copy types include:
// - All integer types (i8, i16, i32, i64, u8, u16, etc.)
// - Floating point (f32, f64)
// - bool, char
// - Tuples of Copy types
// - Arrays of Copy types [T: Copy; N]
// - References (&T, but not &mut T for aliasing)
```

### 2.4 Ownership Morphemes

Sigil provides morphemes for ownership operations:

| Morpheme | Operation | Example |
|----------|-----------|---------|
| `·own` | Take ownership | `data·own` |
| `·clone` | Deep copy | `data·clone` |
| `·copy` | Shallow copy (Copy types) | `data·copy` |
| `·move` | Explicit move | `data·move` |
| `·drop` | Explicit destruction | `data·drop` |

```sigil
// Polysynthetic ownership management
let original = heavy_data()
let cloned = original·clone     // deep copy
let moved = original·move       // explicit move (same as let moved = original)

// In pipelines
data
|process·clone    // process a clone, keep original
|transform·own    // transform takes ownership
|finalize·drop    // explicitly drop result
```

---

## 3. Borrowing

### 3.1 Shared Borrows (&T)

Multiple shared borrows can exist simultaneously:

```sigil
let data = [1, 2, 3]

let r1 = &data        // shared borrow
let r2 = &data        // another shared borrow - OK
let r3 = &data        // and another - OK

print(r1|sum)         // all can read
print(r2|len)
print(r3|α)
```

### 3.2 Exclusive Borrows (&mut T)

Only one mutable borrow can exist, and no shared borrows alongside:

```sigil
let mut data = [1, 2, 3]

let r = &mut data     // exclusive borrow
// let r2 = &data     // ERROR: cannot borrow while mutably borrowed
// let r3 = &mut data // ERROR: only one mutable borrow

r|push(4)             // mutation through exclusive borrow
```

### 3.3 Borrow Rules

1. **At any time, you can have either:**
   - Any number of shared references (`&T`), OR
   - Exactly one exclusive reference (`&mut T`)

2. **References must always be valid** (no dangling pointers)

```sigil
// INVALID: shared and exclusive conflict
let mut data = [1, 2, 3]
let shared = &data
let exclusive = &mut data   // ERROR

// INVALID: multiple exclusive borrows
let mut data = [1, 2, 3]
let r1 = &mut data
let r2 = &mut data          // ERROR

// VALID: borrows don't overlap
let mut data = [1, 2, 3]
{
    let r = &mut data
    r|push(4)
}                           // r's borrow ends here
let r2 = &data              // OK: no active mutable borrow
```

### 3.4 Borrow Morphemes

| Morpheme | Operation | Result |
|----------|-----------|--------|
| `·ref` | Shared borrow | `&T` |
| `·mut` | Exclusive borrow | `&mut T` |
| `·reborrow` | Re-borrow from reference | `&T` / `&mut T` |

```sigil
// Polysynthetic borrowing
let data = [1, 2, 3]

// Traditional
let r = &data
process(r)

// Polysynthetic
data·ref|process

// In pipelines
data
|·ref|read_only_op      // borrow for this operation
|·mut|mutating_op       // mutable borrow for this one
|·own|consuming_op      // take ownership for final step
```

---

## 4. Lifetimes

### 4.1 Lifetime Annotations

Lifetimes track how long references are valid:

```sigil
// Lifetime 'a connects input and output references
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// Multiple lifetimes
fn split<'a, 'b>(data: &'a str, delim: &'b str) -> [&'a str] {
    // result references have lifetime 'a (from data)
    // delim's lifetime 'b is independent
}
```

### 4.2 Lifetime Elision

Common patterns don't require explicit annotations:

```sigil
// These are equivalent:
fn first(s: &str) -> &str
fn first<'a>(s: &'a str) -> &'a str

// Method with &self:
fn get(&self) -> &T
fn get<'a>(&'a self) -> &'a T

// Multiple inputs, one output from self:
fn combine(&self, other: &str) -> &T
fn combine<'a, 'b>(&'a self, other: &'b str) -> &'a T
```

### 4.3 Lifetime Bounds

```sigil
// T must live at least as long as 'a
struct Ref<'a, T: 'a> {
    data: &'a T,
}

// T contains no references shorter than 'static
fn spawn<T: Send + 'static>(task: T) {
    // T can be sent to another thread safely
}
```

### 4.4 Static Lifetime

`'static` indicates data that lives for the entire program:

```sigil
// String literals have 'static lifetime
let s: &'static str = "hello, world"

// Constants are 'static
const CONFIG: &'static Config = &Config { ... }

// Owned types satisfy 'static bounds (no borrowed data)
fn spawn<T: 'static>(t: T) { }
spawn([1, 2, 3])  // OK: Vec owns its data
```

---

## 5. Interior Mutability

### 5.1 Cell Types

For cases where the borrow checker is too restrictive:

```sigil
use std·cell·{Cell, RefCell}

// Cell<T> for Copy types - no borrow tracking
let cell = Cell·new(5)
cell.set(10)
let value = cell.get()

// RefCell<T> for any type - runtime borrow checking
let refcell = RefCell·new([1, 2, 3])
{
    let borrowed = refcell.borrow()      // shared borrow
    print(borrowed)
}
{
    let mut_borrowed = refcell.borrow_mut()  // exclusive borrow
    mut_borrowed|push(4)
}
// Runtime panic if borrows overlap!
```

### 5.2 Synchronized Types

For multi-threaded interior mutability:

```sigil
use std·sync·{Mutex, RwLock, Atomic}

// Mutex - exclusive access
let mutex = Mutex·new(data)
{
    let guard = mutex.lock()!
    guard|mutate()
}   // lock released

// RwLock - multiple readers OR single writer
let rwlock = RwLock·new(data)
{
    let read = rwlock.read()!    // shared access
}
{
    let write = rwlock.write()!  // exclusive access
}

// Atomics - lock-free primitives
let counter = Atomic·new(0)
counter.fetch_add(1, Ordering·SeqCst)
```

### 5.3 Interior Mutability Morphemes

```sigil
// Morphemes for cell operations
value·cell       // Wrap in Cell
value·refcell    // Wrap in RefCell
value·mutex      // Wrap in Mutex
value·rwlock     // Wrap in RwLock
value·atomic     // Wrap in Atomic (primitives only)

// Pipeline-friendly access
shared_state
|·lock           // acquire lock
|·read           // or read lock
|·write          // or write lock
|mutate
|·unlock         // explicit unlock (usually automatic)
```

---

## 6. Smart Pointers

### 6.1 Box - Heap Allocation

```sigil
// Allocate on heap
let boxed: Box<[i32]> = Box·new([1, 2, 3])

// Recursive types require indirection
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

// Deref coercion
fn takes_slice(s: &[i32]) { }
let boxed = Box·new([1, 2, 3])
takes_slice(&boxed)  // auto-deref to &[i32]
```

### 6.2 Rc - Reference Counting

```sigil
use std·rc·Rc

// Shared ownership (single-threaded)
let a = Rc·new([1, 2, 3])
let b = a·clone        // Rc::clone, not deep copy
let c = a·clone

print(Rc·strong_count(&a))  // 3

// Weak references for cycles
use std·rc·Weak
let weak: Weak<_> = a·downgrade
let upgraded: Option<Rc<_>> = weak.upgrade()
```

### 6.3 Arc - Atomic Reference Counting

```sigil
use std·sync·Arc

// Shared ownership (thread-safe)
let data = Arc·new([1, 2, 3])

// Share across threads
let data_clone = data·clone
spawn(|| {
    process(data_clone)
})
```

### 6.4 Smart Pointer Morphemes

```sigil
// Allocation morphemes
value·box        // Box::new(value)
value·rc         // Rc::new(value)
value·arc        // Arc::new(value)

// Reference operations
ptr·clone        // Clone the smart pointer
ptr·downgrade    // Rc/Arc → Weak
weak·upgrade     // Weak → Option<Rc/Arc>

// Example pipeline
heavy_data()
|·arc            // wrap in Arc
|·clone·spawn{ process }  // clone and spawn processor
|·clone·spawn{ analyze }  // clone and spawn analyzer
|await_all
```

---

## 7. Unsafe Code

### 7.1 Unsafe Blocks

For operations the compiler cannot verify:

```sigil
// Mark trusted operations
let value = unsafe {
    // Raw pointer dereference
    *raw_ptr
}

// Unsafe with evidentiality
let trusted‽ = unsafe {
    external_ffi_call()
}
```

### 7.2 Unsafe Functions

```sigil
// Caller must ensure safety invariants
unsafe fn dangerous(ptr: *mut i32) {
    *ptr = 42
}

// Call requires unsafe block
unsafe {
    dangerous(my_ptr)
}
```

### 7.3 Raw Pointers

```sigil
// Raw pointer types (no safety guarantees)
let ptr: *const i32 = &value        // immutable raw
let ptr: *mut i32 = &mut value      // mutable raw

// Operations (all require unsafe)
unsafe {
    let val = *ptr           // dereference
    let offset = ptr.add(1)  // pointer arithmetic
    ptr.write(42)            // write through pointer
}
```

### 7.4 Unsafe Morphemes

```sigil
// Explicit unsafe pipelines
data
|·raw            // Get raw pointer
|unsafe·deref    // Dereference in unsafe context
|unsafe·cast<T>  // Type cast
|·trust          // Convert to paradox evidentiality (‽)
```

---

## 8. Memory Layout

### 8.1 Repr Attributes

Control memory layout with runes:

```sigil
// C-compatible layout
//@ rune: repr(C)
struct CStruct {
    a: u32,
    b: u64,
    c: u8,
}

// Packed (no padding)
//@ rune: repr(packed)
struct Packed {
    a: u8,
    b: u32,
}

// Transparent (same layout as inner type)
//@ rune: repr(transparent)
struct Wrapper(Inner)

// Specify size
//@ rune: repr(align(16))
struct Aligned {
    data: [u8; 32],
}
```

### 8.2 Size and Alignment

```sigil
use std·mem·{size_of, align_of}

let size = size_of::<MyStruct>()
let align = align_of::<MyStruct>()

// Zero-sized types
struct Unit
assert!(size_of::<Unit>() == 0)

// Dynamically sized types
let slice_size = size_of_val(&slice)
```

---

## 9. Drop and RAII

### 9.1 Drop Trait

```sigil
trait Drop {
    fn drop(&mut self)
}

struct FileHandle {
    handle: RawFd,
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        // Automatically called when value goes out of scope
        close(self.handle)
    }
}
```

### 9.2 Drop Order

```sigil
// Dropped in reverse declaration order
{
    let a = Resource·new()   // dropped third
    let b = Resource·new()   // dropped second
    let c = Resource·new()   // dropped first
}

// Struct fields dropped in declaration order
struct Container {
    first: Resource,   // dropped first
    second: Resource,  // dropped second
}
```

### 9.3 Manual Drop

```sigil
use std·mem·drop

let resource = Resource·new()
drop(resource)         // explicit drop
// resource no longer valid

// Prevent drop
use std·mem·forget
let leaked = Resource·new()
forget(leaked)         // memory leaked, drop not called
```

---

## 10. Polysynthetic Memory Patterns

### 10.1 Resource Pipelines

```sigil
// Traditional RAII
fn process_file(path: &str) -> Result<Data!> {
    let file = File·open(path)?
    let content = file.read_all()?
    let data = parse(content)?
    // file automatically closed here
    Ok(data)
}

// Polysynthetic equivalent
fn process_file(path: &str) -> Result<Data!> {
    path
    |file·open?
    |·read_all?
    |parse?
    |Ok
    // Resources cleaned up at each stage
}
```

### 10.2 Scoped Borrowing

```sigil
// Traditional
fn update(data: &mut Data) {
    let view = &data.items[0..10];
    process(view);
    // view borrow ends
    data.items.push(new_item);
}

// Polysynthetic with explicit scopes
fn update(data: &mut Data) {
    data.items
    |·ref[0..10]·scope{ process }  // borrow scoped to operation
    |·mut·push(new_item)           // mutable access after scope
}
```

### 10.3 Ownership Transfer in Chains

```sigil
// Ownership flows left-to-right
source
|·own|transform    // source moved into transform
|·own|process      // transform result moved into process
|·own|finalize     // process result moved into finalize
                   // all intermediate values dropped

// Clone to retain
source
|·clone|transform  // clone moved, source retained
|·own|process      // transform result moved
|result            // final value

// source still usable here
```

---

## 11. Summary: Memory Safety Guarantees

Sigil's memory model provides these compile-time guarantees:

| Guarantee | Mechanism |
|-----------|-----------|
| No use-after-free | Ownership + borrow checking |
| No double-free | Single ownership |
| No dangling references | Lifetime checking |
| No data races | Shared XOR mutable rule |
| No null dereferences | Option type + evidentiality |
| No buffer overflows | Bounds checking + slices |
| Deterministic cleanup | RAII + Drop trait |

All without runtime garbage collection overhead.

# Sigil Capability-Based Memory Specification

> *"To hold a capability is to hold a fragment of power. To transfer it is to share your sovereignty."*

## 0. Design Philosophy: Capabilities as Types

This specification unifies three powerful concepts:

1. **Fractional Permissions** — Resources have divisible, quantitative access rights
2. **Separation Logic** — Heap reasoning with spatial connectives
3. **Capability-Based Security** — Unforgeable tokens of authority

The result: a memory system where **capabilities are types**, access is **proven at compile time**, and security properties are **language-level guarantees**.

This is not Rust's borrow checker with extra steps. This is a fundamentally different model that happens to subsume Rust's guarantees as a special case.

---

## 1. The Capability Model

### 1.1 What is a Capability?

A capability is an **unforgeable token** that grants specific **permissions** over a specific **resource**.

```
Capability = (Resource, Permission, Fraction)
```

- **Resource**: What the capability refers to (memory location, file, socket, etc.)
- **Permission**: What operations are allowed (read, write, execute, etc.)
- **Fraction**: How much of the permission is held (0 to 1, inclusive)

### 1.2 Capability Syntax

```sigil
// Capability types
τ@ρ[π^f]          // Type τ at region ρ with permission π and fraction f

// Examples
i32@heap[rw^1.0]      // Full read-write access to heap i32
&str@static[r^1.0]    // Full read access to static string
File@fd[rw^0.5]       // Half read-write access to file descriptor
Socket@net[w^1.0]     // Full write-only access to socket

// Shorthand
&T       ≡  T@_[r^1.0]      // Full shared reference
&mut T   ≡  T@_[rw^1.0]     // Full exclusive reference
&half T  ≡  T@_[r^0.5]      // Half shared reference
```

### 1.3 Permission Lattice

```
           ⊤ (rwx - all permissions)
          /|\
         / | \
        /  |  \
      rw   rx   wx
       |\ /|\ /|
       | X | X |
       |/ \|/ \|
       r   w   x
        \  |  /
         \ | /
          \|/
           ⊥ (no permissions)

r  = read
w  = write
x  = execute
rw = read + write
rx = read + execute
wx = write + execute
rwx = read + write + execute
```

### 1.4 Fraction Algebra

Fractions are drawn from the rational interval [0, 1]:

```
0    = no access
0.5  = half access (can split once more)
1    = full access (can split into any fractions)

f₁ + f₂ ≤ 1           // Fractions must not exceed whole
f₁ + f₂ = 1 ⟹ full    // Reuniting fractions restores full access
f > 0 ⟹ can_read      // Any positive fraction allows reading
f = 1 ⟹ can_write     // Only full fraction allows writing
```

---

## 2. Capability Operations

### 2.1 Splitting

A capability can be split into smaller fractions:

```sigil
// Split a full capability into two halves
let cap: File@fd[rw^1.0] = open("data.txt");
let (cap1, cap2): (File@fd[rw^0.5], File@fd[rw^0.5]) = cap|split;

// Both can read
let data1 = cap1|read;  // OK: 0.5 > 0
let data2 = cap2|read;  // OK: 0.5 > 0

// Neither can write alone
cap1|write(data);  // ERROR: fraction 0.5 < 1.0 required for write
```

**Typing rule:**
```
SPLIT
Γ ⊢ e : τ@ρ[π^f]    f = f₁ + f₂    f₁ > 0    f₂ > 0
──────────────────────────────────────────────────────
Γ ⊢ e|split : (τ@ρ[π^f₁], τ@ρ[π^f₂])
```

### 2.2 Joining

Split capabilities can be reunited:

```sigil
// Rejoin the halves
let cap_full: File@fd[rw^1.0] = (cap1, cap2)|join;

// Now we can write
cap_full|write(data);  // OK: fraction 1.0
```

**Typing rule:**
```
JOIN
Γ ⊢ e₁ : τ@ρ[π^f₁]    Γ ⊢ e₂ : τ@ρ[π^f₂]    same_resource(e₁, e₂)
─────────────────────────────────────────────────────────────────
Γ ⊢ (e₁, e₂)|join : τ@ρ[π^(f₁+f₂)]
```

### 2.3 Borrowing as Temporary Split

Borrowing is just splitting with automatic rejoin at scope exit:

```sigil
fn process(data: &File@fd[r^0.5]) { ... }

let file: File@fd[rw^1.0] = open("data.txt");

// Borrow splits off 0.5, keeps 0.5
{
    let borrowed = &file;  // borrowed: File@fd[r^0.5], file retains rw^0.5
    process(borrowed);
}  // borrowed's fraction returns to file

// file is back to rw^1.0
file|write(data);  // OK
```

### 2.4 Permission Downgrade

Capabilities can be downgraded (more restrictive), never upgraded:

```sigil
let full: File@fd[rw^1.0] = open("data.txt");

// Downgrade to read-only (drop write permission)
let readonly: File@fd[r^1.0] = full|downgrade(r);

// Can read
readonly|read;  // OK

// Cannot write (permission lost)
readonly|write(data);  // ERROR: permission 'r' does not include 'w'

// Cannot get write back
readonly|upgrade(rw);  // ERROR: capabilities cannot be upgraded
```

### 2.5 Transfer

Capabilities can be transferred between contexts (move semantics):

```sigil
fn consume(cap: File@fd[rw^1.0]) {
    // cap is now owned here
}

let file: File@fd[rw^1.0] = open("data.txt");
consume(file);  // Capability transferred

// file is no longer valid
file|read;  // ERROR: capability has been transferred
```

---

## 3. Separation Logic Integration

### 3.1 Spatial Connectives

Separation logic reasons about **disjoint heap regions**:

```
P * Q       // Separating conjunction: P and Q hold on disjoint regions
P -* Q      // Magic wand: if given P, can produce Q
emp         // Empty heap
x ↦ v       // x points to value v (singleton heap)
```

### 3.2 Capability Assertions

```sigil
// Assertion syntax in specs/contracts
//@ requires cap@ρ[rw^1.0] * other@ρ'[r^0.5]
//@ ensures  cap@ρ[rw^1.0] * result@ρ''[rw^1.0]
fn transform(cap: Resource, other: &Resource) → NewResource {
    ...
}
```

### 3.3 Frame Rule

The **frame rule** allows local reasoning — if a function only touches certain capabilities, the rest are preserved:

```
FRAME
{P} c {Q}
─────────────────────────
{P * R} c {Q * R}
```

**In Sigil terms:**
```sigil
// Function only uses cap1
fn process(cap1: A@ρ₁[rw^1.0]) → B@ρ₂[rw^1.0] { ... }

// Caller has cap1 and cap2
let cap1: A@ρ₁[rw^1.0] = ...;
let cap2: C@ρ₃[rw^1.0] = ...;

// Frame rule: cap2 is automatically preserved
let result = process(cap1);
// cap2 still valid with same permissions
```

### 3.4 Separating Conjunction in Types

We can express "disjoint capabilities" in types:

```sigil
// Two capabilities that must refer to disjoint resources
type DisjointPair<A, B> = (A@ρ₁[rw^1.0], B@ρ₂[rw^1.0]) where ρ₁ ⊥ ρ₂

// Function requiring disjoint inputs
fn swap<T>(a: &mut T@ρ₁, b: &mut T@ρ₂) where ρ₁ ⊥ ρ₂ {
    let tmp = *a;
    *a = *b;
    *b = tmp;
}

// Cannot swap with self (regions not disjoint)
let x: i32@heap[rw^1.0] = 5;
swap(&mut x, &mut x);  // ERROR: ρ ⊥ ρ is unsatisfiable
```

---

## 4. Region System

### 4.1 Region Kinds

```
ρ ::=
    | static           // Lives for entire program
    | heap             // Dynamically allocated
    | stack(n)         // Stack frame n
    | arena(a)         // Arena allocator a
    | ρ₁ ∪ ρ₂          // Union of regions
    | ∃ρ. ...          // Existentially quantified region
    | α                // Region variable
```

### 4.2 Region Ordering

Regions have a partial order based on **outlives**:

```
static ≥ heap ≥ stack(n) ≥ stack(n+1)
```

**Subtyping:**
```
ρ₁ ≥ ρ₂ ⟹ τ@ρ₁[π^f] <: τ@ρ₂[π^f]
```

A capability from a longer-lived region can be used where a shorter-lived one is expected.

### 4.3 Region Polymorphism

Functions can be polymorphic over regions:

```sigil
fn identity<ρ, T>(x: T@ρ[rw^1.0]) → T@ρ[rw^1.0] {
    x
}

// Works with any region
let stack_val = identity(local_var);
let heap_val = identity(boxed_var);
```

---

## 5. Capability-Based Security

### 5.1 The Principle of Least Authority (POLA)

Functions receive only the capabilities they need:

```sigil
// BAD: Function has ambient authority
fn bad_process() {
    let file = open("/etc/passwd");  // Where did this capability come from?
}

// GOOD: Capability explicitly passed
fn good_process(file: File@fd[r^0.5]) {
    let content = file|read;  // Authority is explicit
}
```

### 5.2 Capability Attenuation

Capabilities can be attenuated (restricted) before passing:

```sigil
let full_access: File@fd[rwx^1.0] = open("script.sh");

// Attenuate to read-only before passing to untrusted code
let restricted: File@fd[r^1.0] = full_access|attenuate(r);
untrusted_code(restricted);

// Original still has full access
full_access|write(new_content);
full_access|execute();
```

### 5.3 Capability Sealing

Capabilities can be **sealed** to prevent inspection:

```sigil
// Seal a capability with a brand
let sealed: Sealed<File, Brand> = cap|seal(my_brand);

// Cannot use sealed capability directly
sealed|read;  // ERROR: sealed capabilities cannot be used

// Only holder of brand can unseal
let unsealed: File@fd[rw^1.0] = sealed|unseal(my_brand);
```

### 5.4 Revocation

Capabilities can be revocable:

```sigil
let (cap, revoker): (Revocable<File>, Revoker) = file|make_revocable;

// Pass capability to another context
other_context(cap);

// Later, revoke access
revoker|revoke;

// Other context's capability is now invalid
// Any use will fail at runtime (capability check)
```

---

## 6. NyxOS Integration

### 6.1 Kernel Capabilities

NyxOS capabilities map directly to Sigil capability types:

```sigil
// NyxOS kernel capabilities
type ProcessCap = Handle@kernel[fork|exec|kill^f]
type MemoryCap = Region@kernel[r|w|x^f]
type FileCap = Descriptor@kernel[r|w|seek|truncate^f]
type NetworkCap = Socket@kernel[connect|listen|send|recv^f]
type DeviceCap = Device@kernel[ioctl|mmap^f]
```

### 6.2 Syscall Safety

System calls require appropriate capabilities:

```sigil
// Safe syscall wrapper
fn sys_read(fd: FileCap@kernel[r^_], buf: &mut [u8]) → Result<usize> {
    // Compiler verifies caller has read capability
    unsafe { syscall(SYS_READ, fd.raw(), buf.as_mut_ptr(), buf.len()) }
}

// Cannot call without capability
sys_read(???);  // ERROR: no FileCap in scope
```

### 6.3 Capability Delegation

Process hierarchy uses capability delegation:

```sigil
// Parent process
fn spawn_child(
    process_cap: ProcessCap@kernel[fork^1.0],
    child_caps: CapabilitySet
) → ChildHandle {
    // Split capabilities for child
    let (parent_caps, child_caps) = capabilities|partition(child_caps);

    // Fork transfers child_caps to new process
    let child = process_cap|fork(child_caps);

    // Parent retains parent_caps
    child
}
```

### 6.4 Sandbox Enforcement

Sandboxes are just restricted capability sets:

```sigil
// Sandbox = process with minimal capabilities
fn create_sandbox(code: Code) → SandboxHandle {
    let sandbox_caps = CapabilitySet::empty()
        |grant(MemoryCap::new(sandbox_region, rw^1.0))
        |grant(StdioCap::stdout_only());

    spawn_child(process_cap, sandbox_caps, code)
}

// Sandboxed code cannot:
// - Access filesystem (no FileCap)
// - Access network (no NetworkCap)
// - Access other processes (no ProcessCap)
// - Access arbitrary memory (only sandbox_region)
```

---

## 7. Inference and Checking

### 7.1 Capability Inference

The type system infers required capabilities:

```sigil
fn example() {
    let file = open("data.txt");  // Infer: requires OpenCap
    let data = file|read;         // Infer: requires file@[r^>0]
    file|write(data);             // Infer: requires file@[w^1.0]
}

// Inferred signature:
// fn example() requires OpenCap@fs, file@fd[rw^1.0]
```

### 7.2 Fraction Inference

Fractions are inferred from usage:

```sigil
fn use_twice(x: &T) {
    process1(x);
    process2(x);  // x used twice, so caller needs enough fraction
}

// If process1 and process2 each need f=0.25:
// Inferred: fn use_twice(x: T@ρ[r^0.5])  // 0.25 + 0.25 = 0.5 minimum
```

### 7.3 Verification Conditions

The checker generates **verification conditions** for SMT:

```
VC ::=
    | f₁ + f₂ ≤ f_total       // Fractions don't exceed available
    | f > 0                    // Read requires positive fraction
    | f = 1                    // Write requires full fraction
    | ρ₁ ⊥ ρ₂                  // Regions are disjoint
    | ρ₁ ≥ ρ₂                  // Region outlives
    | π₁ ⊆ π₂                  // Permission subset
```

---

## 8. Typing Rules

### 8.1 Core Rules

```
VAR
x : τ@ρ[π^f] ∈ Γ
─────────────────────────
Γ ⊢ x : τ@ρ[π^f]

DEREF (read)
Γ ⊢ e : τ@ρ[π^f]    r ∈ π    f > 0
──────────────────────────────────────
Γ ⊢ *e : τ

ASSIGN (write)
Γ ⊢ e₁ : τ@ρ[π^1.0]    w ∈ π    Γ ⊢ e₂ : τ
──────────────────────────────────────────────
Γ ⊢ *e₁ = e₂ : ()

BORROW-SHARED
Γ ⊢ e : τ@ρ[π^f]    f' ≤ f    r ∈ π
────────────────────────────────────────────────
Γ ⊢ &e : τ@ρ[r^f']    with Γ' = Γ[e ↦ τ@ρ[π^(f-f')]]

BORROW-MUT
Γ ⊢ e : τ@ρ[π^1.0]    rw ⊆ π
────────────────────────────────────────────────
Γ ⊢ &mut e : τ@ρ[rw^1.0]    with Γ' = Γ[e ↦ ⊥]

RETURN-BORROW
Γ, x:τ@ρ[π^f] ⊢ body    borrow x ends
────────────────────────────────────────────────
Γ[x ↦ τ@ρ[π^(f_orig)]] ⊢ ...    // fraction restored
```

### 8.2 Separation Rules

```
FRAME
Γ₁ ⊢ e : τ    Γ₁ * Γ₂ defined
────────────────────────────────
Γ₁ * Γ₂ ⊢ e : τ    with Γ₂ preserved

SEP-CONJ
Γ₁ ⊢ e₁ : τ₁@ρ₁[π₁^f₁]    Γ₂ ⊢ e₂ : τ₂@ρ₂[π₂^f₂]    ρ₁ ⊥ ρ₂
──────────────────────────────────────────────────────────────
Γ₁ * Γ₂ ⊢ (e₁, e₂) : (τ₁@ρ₁[π₁^f₁] * τ₂@ρ₂[π₂^f₂])
```

### 8.3 Capability Rules

```
ATTENUATE
Γ ⊢ e : τ@ρ[π^f]    π' ⊆ π    f' ≤ f
────────────────────────────────────────
Γ ⊢ e|attenuate(π', f') : τ@ρ[π'^f']

SEAL
Γ ⊢ e : τ@ρ[π^f]    Γ ⊢ brand : Brand<B>
────────────────────────────────────────────
Γ ⊢ e|seal(brand) : Sealed<τ@ρ[π^f], B>

UNSEAL
Γ ⊢ e : Sealed<τ@ρ[π^f], B>    Γ ⊢ brand : Brand<B>
────────────────────────────────────────────────────
Γ ⊢ e|unseal(brand) : τ@ρ[π^f]
```

---

## 9. Rust Comparison

| Rust | Sigil Capability Model |
|------|------------------------|
| `&T` | `T@ρ[r^f]` where f > 0 |
| `&mut T` | `T@ρ[rw^1.0]` |
| Shared XOR Mutable | **Fractional**: many readers OR one writer |
| Binary permissions | **Quantitative**: any fraction in [0,1] |
| Lifetimes | **Regions** with outlives ordering |
| No capability concept | **First-class capabilities** |
| Runtime safety checks | **Compile-time proofs** |
| `unsafe` escape hatch | **Explicit capability grants** |

### 9.1 What Sigil Can Express That Rust Cannot

```sigil
// 1. Split a reference, use both halves concurrently
let (r1, r2) = data|split;
parallel(|| use(r1), || use(r2));
let data = (r1, r2)|join;

// 2. Read-only view that can later become writable
let view: Data@heap[r^0.5] = data|split.0;
// ... much later ...
let full = (view, other_half)|join;
full|write(new_data);

// 3. Prove two references are disjoint at compile time
fn safe_memcpy<ρ₁, ρ₂>(
    dst: &mut [u8]@ρ₁,
    src: &[u8]@ρ₂
) where ρ₁ ⊥ ρ₂ {
    // Compiler PROVES no overlap — no runtime check needed
}

// 4. Capability-attenuated delegation
fn untrusted_plugin(data: Data@heap[r^0.5]) {
    // Plugin can read but not write
    // Plugin cannot forge write capability
    // This is ENFORCED BY THE TYPE SYSTEM
}
```

---

## 10. Error Messages

### 10.1 Fraction Errors

```
error[C401]: insufficient capability fraction
  --> src/main.sg:15:5
   |
15 |     data|write(value);
   |     ^^^^ requires fraction 1.0, but only 0.5 available
   |
note: capability was split here
  --> src/main.sg:10:18
   |
10 |     let (a, b) = data|split;
   |                  ^^^^^^^^^^ split into two 0.5 fractions
   |
help: rejoin the fractions before writing
   |
15 |     let full = (a, b)|join;
   |     full|write(value);
```

### 10.2 Region Errors

```
error[C402]: regions not disjoint
  --> src/main.sg:20:5
   |
20 |     swap(&mut x, &mut x);
   |          ^^^^^^  ^^^^^^ same region ρ₁
   |          |
   |          region ρ₁
   |
note: swap requires disjoint regions
  --> std/mem.sg:42:1
   |
42 | fn swap<T, ρ₁, ρ₂>(a: &mut T@ρ₁, b: &mut T@ρ₂) where ρ₁ ⊥ ρ₂
   |                                                 ^^^^^^^^^^
```

### 10.3 Permission Errors

```
error[C403]: permission denied
  --> src/main.sg:25:5
   |
25 |     file|execute();
   |     ^^^^ capability has permission [r], but [x] required
   |
note: capability was attenuated here
  --> src/main.sg:22:16
   |
22 |     let file = full|attenuate(r);
   |                ^^^^^^^^^^^^^^^^^ write and execute permissions removed
```

---

## 11. Implementation Notes

### 11.1 Runtime Representation

At runtime, capabilities are typically:
- **Erased** for local operations (proven safe at compile time)
- **Fat pointers** for cross-boundary operations (capability + permission bits)
- **Handles** for kernel resources (index into capability table)

### 11.2 SMT Encoding

Capability constraints encode to SMT:

```smt2
; Fractions
(declare-const f1 Real)
(declare-const f2 Real)
(assert (and (>= f1 0) (<= f1 1)))
(assert (and (>= f2 0) (<= f2 1)))
(assert (<= (+ f1 f2) 1))  ; fractions don't exceed whole

; Region disjointness
(declare-const r1 Region)
(declare-const r2 Region)
(assert (disjoint r1 r2))

; Permission subset
(declare-const p1 Permission)
(declare-const p2 Permission)
(assert (subset p1 p2))
```

---

## 12. Summary

Sigil's capability-based memory model provides:

| Property | Guarantee |
|----------|-----------|
| **Memory safety** | No use-after-free, no double-free |
| **Data race freedom** | Fractional permissions prevent races |
| **Capability security** | Authority is explicit and unforgeable |
| **Least privilege** | Functions receive only needed capabilities |
| **Compile-time verification** | No runtime capability checks for proven code |
| **NyxOS integration** | Kernel capabilities = language capabilities |

This is not a borrow checker. This is a **proof system for resource authority**.

Agents can track fractional permissions across complex control flow. Humans can rely on tooling. The language does not compromise.

---

*"Every pointer is a key. Every key opens only what it was forged to open. The capability is the proof that you may enter."*

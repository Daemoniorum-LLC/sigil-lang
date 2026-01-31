# Jormungandr Native Runtime Roadmap

**Created:** 2026-01-20
**Status:** Planning
**Related Spec:** `/docs/specs/SIGIL-NATIVE-RUNTIME-SPEC.md` Section 6.4

---

## Overview

This roadmap details the updates required to Jormungandr (the self-hosted Sigil compiler) to enable compilation of the native runtime with zero libc dependency.

**Goal:** Enable `stdlib/sys/` to be compiled by Jormungandr with inline assembly for direct syscalls.

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer | Working | Rust-like tokens only |
| Parser | Working | 50/50 Nihil files pass |
| Type Checker | Working | All 21 ecosystem libs pass |
| C Codegen | Buggy | Nested match bug (mutable ref params) |
| Self-compilation | Blocked | By mutable ref param bug |
| Native Sigil syntax | Not implemented | `☉`, `≔`, `⎇`, etc. |
| Inline assembly | Not implemented | `asm!` macro |

---

## Phase 1: Fix Blocking Bugs (P0 - Critical)

### 1.1 Mutable Reference Parameter Bug

**Issue:** Struct field assignments on mutable reference parameters generate incorrect C code.

**Current (buggy):**
```c
sigil_struct_set_field(&param, "field", value);
```

**Required:**
```c
sigil_struct_set_field((SigilValue*)param.v.ptr, "field", value);
```

**Files to modify:**
- `jormungandr/src/codegen.sg` - Lines ~3230-3244, ~4098-4116

**Test criteria:**
- `lower_item()` generates correct field assignments
- Bootstrap produces non-empty output
- Self-compilation succeeds

### 1.2 Nested Match Codegen

**Issue:** Nested match expressions generate invalid C.

**Files affected:**
- `jormungandr/src/codegen.sg` - Match expression handling

**Test criteria:**
- Test suite match expressions pass
- Jormungandr compiles its own match expressions

---

## Phase 2: Native Sigil Syntax (P0 - High)

### 2.1 Lexer Token Types

Add new tokens for native Sigil symbols:

| Unicode | Codepoint | Name | Token Kind |
|---------|-----------|------|------------|
| `☉` | U+2609 | Sun with rays | `Visibility` |
| `≔` | U+2254 | Colon equals | `ConstBind` |
| `⎇` | U+2387 | Alternative key | `If` |
| `⎉` | U+2389 | Circled horizontal bar | `Else` |
| `⌥` | U+2325 | Option key | `Loop` |

**Files to modify:**
- `jormungandr/src/lexer.sg` - Add token patterns
- `jormungandr/src/token.sg` - Add TokenKind variants

**Implementation:**
```sigil
// In lexer.sg - character scanning
'☉' => Token::new(TokenKind::SunWithRays, "☉", span),
'≔' => Token::new(TokenKind::ConstAssign, "≔", span),
'⎇' => Token::new(TokenKind::ConditionalIf, "⎇", span),
'⎉' => Token::new(TokenKind::ConditionalElse, "⎉", span),
'⌥' => Token::new(TokenKind::Loop, "⌥", span),
```

### 2.2 Keyword Additions

Add new keywords:

| Keyword | Replaces | Purpose |
|---------|----------|---------|
| `vary` | `mut` | Mutable binding/parameter |
| `rite` | `fn` | Function declaration |

**Files to modify:**
- `jormungandr/src/lexer.sg` - Keyword table
- `jormungandr/src/parser.sg` - Accept both syntaxes

### 2.3 Type Suffix Parsing

Add evidentiality suffixes to type parsing:

| Suffix | Meaning | Example |
|--------|---------|---------|
| `!` | Owned/non-null | `String!` |
| `?` | Nullable | `String?` |
| `~` | Borrowed | `String~` |

**Files to modify:**
- `jormungandr/src/parser.sg` - Type parsing rules

**Test file:**
```sigil
// test_native_syntax.sg
☉ rite test_visibility() → i64! {
    ≔ value: i64 = 42;
    vary counter: i64! = 0;

    ⎇ value > 0 {
        counter = value;
    } ⎉ {
        counter = 0;
    }

    counter
}
```

---

## Phase 3: Inline Assembly Support (P0 - Critical)

### 3.1 AST Node

Add AST representation for inline assembly:

```sigil
// In ast.sg
pub enum Expr {
    // ... existing variants ...
    InlineAsm {
        template: !String,
        operands: ![AsmOperand],
        options: ![AsmOption],
    },
}

pub struct AsmOperand {
    pub kind: AsmOperandKind,
    pub constraint: !String,
    pub expr: ?Expr,
}

pub enum AsmOperandKind {
    In,
    Out,
    InOut { output_expr: !Expr },
    Clobber,
}

pub enum AsmOption {
    NoStack,
    Pure,
    NoMem,
}
```

### 3.2 Parser Changes

Parse `asm!` macro syntax:

```sigil
// Input
asm!("syscall",
    inout("rax") 39_i64 => result,
    in("rdi") arg1,
    out("rcx") _,
    options(nostack));

// Parsed as
InlineAsm {
    template: "syscall",
    operands: [
        AsmOperand { kind: InOut { output_expr: result }, constraint: "rax", expr: 39_i64 },
        AsmOperand { kind: In, constraint: "rdi", expr: arg1 },
        AsmOperand { kind: Clobber, constraint: "rcx", expr: None },
    ],
    options: [NoStack],
}
```

**Files to modify:**
- `jormungandr/src/parser.sg` - Add `parse_asm_expr()`

### 3.3 C Code Generation

Generate GCC inline asm from AST:

**Constraint mappings:**

| Sigil Register | GCC Constraint |
|----------------|----------------|
| `"rax"` | `"a"` |
| `"rbx"` | `"b"` |
| `"rcx"` | `"c"` |
| `"rdx"` | `"d"` |
| `"rsi"` | `"S"` |
| `"rdi"` | `"D"` |
| `"r8"` | `"r8"` (or use `register` keyword) |
| `"r9"` | `"r9"` |
| `"r10"` | `"r10"` |
| `"r11"` | `"r11"` |

**Generation template:**
```c
// For: asm!("syscall", inout("rax") 39_i64 => result, in("rdi") arg1, out("rcx") _)
{
    int64_t _asm_in0 = 39;
    int64_t _asm_out0;
    __asm__ volatile (
        "syscall"
        : "=a" (_asm_out0)      // outputs
        : "a" (_asm_in0), "D" (arg1)  // inputs
        : "rcx", "r11", "memory"      // clobbers
    );
    result = _asm_out0;
}
```

**Files to modify:**
- `jormungandr/src/codegen.sg` - Add `emit_inline_asm()`

### 3.4 Syscall Verification Test

Create end-to-end verification:

```sigil
// tests/syscall_verify.sg
fn main() -> i64 {
    let mut pid: i64 = 0;
    unsafe {
        asm!("syscall",
            inout("rax") 39_i64 => pid,
            out("rcx") _,
            out("r11") _,
            options(nostack));
    }

    // getpid returns > 0 for valid process
    if pid > 0 { 0 } else { 1 }
}
```

**Test process:**
```bash
# Compile with Jormungandr
./sigil2 compile tests/syscall_verify.sg -o syscall_verify.c
gcc -o syscall_verify syscall_verify.c

# Run and check exit code
./syscall_verify
echo $?  # Should be 0
```

---

## Phase 4: stdlib/sys Compilation

### 4.1 Verification Steps

1. Compile `stdlib/sys/mod.sg` with Jormungandr
2. Compile `stdlib/sys/linux_x86_64.sg`
3. Compile `stdlib/sys/alloc.sg`
4. Run syscall tests
5. Verify mmap/munmap work

### 4.2 Test Matrix

| Test | Description | Success Criteria |
|------|-------------|------------------|
| T1 | Lexer tokens | Native symbols tokenize |
| T2 | Parser accepts | Files parse without error |
| T3 | Type check | No type errors |
| T4 | C generation | Valid C code output |
| T5 | Compilation | gcc accepts output |
| T6 | Runtime | Syscalls work correctly |

---

## Implementation Order

```
Phase 1: Bug Fixes
    ├── 1.1 Mutable ref param bug [BLOCKS: Phase 4]
    └── 1.2 Nested match bug [BLOCKS: Phase 4]

Phase 2: Native Syntax
    ├── 2.1 Lexer tokens [BLOCKS: 2.2, 2.3]
    ├── 2.2 Keywords [BLOCKS: Phase 4]
    └── 2.3 Type suffixes [BLOCKS: Phase 4]

Phase 3: Inline Assembly
    ├── 3.1 AST nodes [BLOCKS: 3.2]
    ├── 3.2 Parser [BLOCKS: 3.3]
    ├── 3.3 C codegen [BLOCKS: 3.4]
    └── 3.4 Verification [BLOCKS: Phase 4]

Phase 4: stdlib/sys
    └── 4.1 Full compilation test
```

---

## Effort Estimates

| Phase | Task | Complexity | Risk |
|-------|------|------------|------|
| 1.1 | Mutable ref fix | Medium | Low (known issue) |
| 1.2 | Nested match | Medium | Medium (subtle bug) |
| 2.1 | Lexer tokens | Low | Low |
| 2.2 | Keywords | Low | Low |
| 2.3 | Type suffixes | Medium | Low |
| 3.1 | AST nodes | Low | Low |
| 3.2 | Parser | Medium | Medium |
| 3.3 | C codegen | High | High (ABI details) |
| 3.4 | Verification | Low | Low |
| 4.1 | Integration | Medium | Medium |

**Critical Path:** 1.1 → 2.1 → 2.2 → 3.1 → 3.2 → 3.3 → 3.4 → 4.1

---

## Success Criteria

| ID | Criterion | Verification |
|----|-----------|--------------|
| S1 | Native syntax compiles | `☉ rite foo() {}` accepted |
| S2 | Inline asm works | getpid syscall returns valid PID |
| S3 | stdlib/sys compiles | All 3 files generate valid C |
| S4 | Syscalls work | mmap/munmap/write functional |
| S5 | Self-hosting | Jormungandr compiles itself |

---

## References

- SIGIL-NATIVE-RUNTIME-SPEC.md - Section 6.4
- BOOTSTRAP_BUG_ANALYSIS.md - Mutable ref param bug
- COMPILER_GAPS.md - Feature status
- Linux syscall ABI: https://blog.packagecloud.io/the-definitive-guide-to-linux-system-calls/

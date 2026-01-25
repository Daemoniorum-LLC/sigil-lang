# Sigil Syntax Guide - Tome/Invoke/Rune System

**Date:** 2026-01-14
**Purpose:** Guide for updating Rust-style Sigil code to use Sigil-native terminology

---

## Summary

Sigil has evolved from Rust-inspired syntax to its own native terminology. **Both old and new syntax are supported for backward compatibility**, but new code should use Sigil-native terms.

---

## Terminology Translation

| Rust Term | Sigil-Native Term | Purpose |
|-----------|-------------------|---------|
| `crate` | **`tome`** | A collection of code/knowledge (compilation unit) |
| `use` | **`invoke`** | Bring symbols into scope |
| `mod` | **`scroll`** | Subdivision of a tome (module) |
| `Cargo.toml` | **`Binding.toml`** | What binds a tome together |
| `#[attribute]` | **`//@ rune: attribute`** | Metadata annotations |

---

## Import Syntax

### OLD (Rust-style) ❌
```sigil
use crate::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
```

### NEW (Sigil-native) ✅
```sigil
invoke tome::prelude::*;           // Current tome
invoke std::collections::HashMap;  // External tome
invoke std::sync::Arc;
```

**Note:** Path separator is `::` (not `·` from syntax spec - that's for incorporating)

---

## Module Syntax

### OLD (Rust-style) ❌
```sigil
pub mod database;
mod utils;

mod helpers {
    pub fn utility() { }
}
```

### NEW (Sigil-native) ✅
```sigil
pub scroll database;
scroll utils;

scroll helpers {
    pub fn utility() { }
}
```

---

## Attribute Syntax (Runes)

### OLD (Rust-style) ❌
```sigil
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditCategory {
    Authentication,
    Authorization,
}
```

### NEW (Sigil-native) ✅
```sigil
//@ rune: derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)
//@ rune: serde(rename_all = "snake_case")
pub enum AuditCategory {
    Authentication,
    Authorization,
}
```

---

## Common Runes

| Rune | Purpose | Example |
|------|---------|---------|
| `derive(...)` | Auto-implement traits | `//@ rune: derive(Debug, Clone)` |
| `inline(...)` | Inlining hints | `//@ rune: inline(always)` |
| `cfg(...)` | Conditional compilation | `//@ rune: cfg(target_os = "linux")` |
| `test` | Mark test function | `//@ rune: test` |
| `deprecated` | Deprecation warning | `//@ rune: deprecated("Use new_fn")` |
| `must_use` | Require result handling | `//@ rune: must_use` |
| `repr(...)` | Memory layout control | `//@ rune: repr(C)` |

---

## Backward Compatibility

**IMPORTANT:** Both syntaxes work! The compiler supports:
- `use` AND `invoke` (both valid)
- `mod` AND `scroll` (both valid)
- `crate` AND `tome` (both valid)
- `#[...]` AND `//@ rune: ...` (likely both valid, needs verification)

This means **existing code doesn't NEED to be updated**, but new code SHOULD use Sigil-native terms.

---

## Example: Styx audit.sigil

### BEFORE (Rust-style)
```sigil
use crate::prelude::*;
use styx_db::prelude::*;
use arcanum::prelude::*;

use std::sync::Arc;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditCategory {
    Authentication,
    Authorization,
    DataAccess,
}
```

### AFTER (Sigil-native)
```sigil
invoke tome::prelude::*;
invoke styx_db::prelude::*;
invoke arcanum::prelude::*;

invoke std::sync::Arc;
invoke std::collections::HashMap;

//@ rune: derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)
//@ rune: serde(rename_all = "snake_case")
pub enum AuditCategory {
    Authentication,
    Authorization,
    DataAccess,
}
```

---

## Path Separators: `::` vs `·`

**Confusion Alert:** The syntax spec (02-SYNTAX.md) shows imports using middot `·`:
```sigil
use std·io·{Read, Write}     // Spec shows this
```

But the implementation and COMPILER_GAPS.md show `::`:
```sigil
invoke std::collections::HashMap;   // Actually works
```

**Conclusion:** Use `::` for paths (the implementation is authoritative).

The middot `·` is used for **incorporation** (compound expressions):
```sigil
path|file·read·lines         // Incorporation uses ·
url|http·get·json·parse      // Compound operations
```

---

## Tome Resolution

The compiler automatically resolves external tome imports:

```bash
# With tome paths specified
sigil compile file.sg --tome-path=/path/to/tomes

# Auto-inferred paths (finds sibling tomes)
sigil compile project/src/lib.sg
```

### How It Works

1. **Local scrolls tracked:** `scroll`/`mod` declarations prevent resolving local modules as external tomes
2. **Search path inference:** Compiler infers tome paths from input file locations
3. **External tomes located:** At `<path>/<tome_name>/src/lib.sg`
4. **Standard library:** `std::*` imports currently skipped (not yet a real tome)

---

## What Needs Updating in Styx?

Based on `/home/crook/dev2/workspace/styx/crates/styx-core/src/audit.sigil`:

### 1. Replace `use crate::` with `invoke tome::`
```diff
- use crate::prelude::*;
+ invoke tome::prelude::*;
```

### 2. Replace external `use` with `invoke`
```diff
- use styx_db::prelude::*;
- use arcanum::prelude::*;
+ invoke styx_db::prelude::*;
+ invoke arcanum::prelude::*;
```

### 3. Replace `use std::` with `invoke std::`
```diff
- use std::sync::Arc;
- use std::collections::HashMap;
+ invoke std::sync::Arc;
+ invoke std::collections::HashMap;
```

### 4. Replace `#[...]` with `//@ rune: ...`
```diff
- #[derive(Clone, Copy, Debug)]
- #[serde(rename_all = "snake_case")]
+ //@ rune: derive(Clone, Copy, Debug)
+ //@ rune: serde(rename_all = "snake_case")
```

### 5. Replace `mod` with `scroll` (if any)
```diff
- pub mod database;
+ pub scroll database;
```

---

## Verification

After updating, verify compilation:
```bash
cd /home/crook/dev2/workspace/sigil/sigil-lang/jormungandr/build
./sigil2 check /home/crook/dev2/workspace/styx/crates/styx-core/src/audit.sigil
```

---

## References

- **Syntax Spec:** `/home/crook/dev2/workspace/sigil/sigil-lang/docs/specs/02-SYNTAX.md`
- **Compiler Gaps:** `/home/crook/dev2/workspace/sigil/sigil-lang/jormungandr/COMPILER_GAPS.md`
- **Driver Source:** `/home/crook/dev2/workspace/sigil/sigil-lang/jormungandr/src/driver.sg`

---

## Next Steps

1. ✅ **Documentation complete** - This guide
2. ⏸️ **Update Styx files** - Replace Rust-style with Sigil-native
3. ⏸️ **Test compilation** - Verify with sigil2
4. ⏸️ **Build Styx binary** - Once files compile

---

**Key Takeaway:** The user was RIGHT - `crate` is deprecated in favor of `tome`. However, both work for backward compatibility. For clean, idiomatic Sigil code, use `invoke`, `tome`, `scroll`, and `//@ rune:` syntax.

# LLVM Backend Test Status

## Summary

| Category | Compiles | Failed | Notes |
|----------|----------|--------|-------|
| FFI      | 31       | 5      | 3 expect compile errors, 2 use unsupported type alias syntax |
| Structs  | 3        | 0      | All passing |
| Impl     | 3        | 0      | All passing |
| Enums    | 2        | 0      | All passing |
| Option   | 2        | 0      | All passing |
| Vec      | 2        | 0      | All passing |
| String   | 2        | 0      | All passing |
| IO       | 1        | 0      | All passing |
| Modules  | 4        | 0      | All passing |
| Native   | 7        | 3      | 3 use unsupported type alias syntax |
| **Total** | **57** | **8** | **88% compile rate** |

## Recent Improvements

### February 2026

1. **#[link("lib")] attribute support**
   - Extern blocks now properly pass link libraries to the linker
   - GTK4 and other native library tests now compile correctly

2. **Opaque type declarations in extern blocks**
   - `type GtkWindow;` declarations are now parsed and handled
   - Creates opaque struct types for C interop

3. **Comment handling in extern blocks**
   - Line and block comments are now properly skipped inside extern blocks

4. **Deprecated syntax tests converted**
   - All `.sigil` tests converted to native Sigil syntax (`.sg`)

## Known Issues

### Parser Limitations

1. **Type aliases in extern blocks** (tests 06, 09, 10 in native/)
   ```sigil
   extern "C" {
       type GCallback = rite(*void);  // Not supported
   }
   ```
   Currently only opaque type declarations (`type Name;`) are supported.

### Test Issues

1. **Test 16_extern_static_immutable_error** expects a compile error but compiles successfully
   - This is a type checker issue - should reject writes to immutable statics

2. **Test 13_unsupported_abi** expects a compile error - working correctly

## Test Categories

### FFI Tests (31/36 compiling)

All core FFI functionality works:
- Extern block declarations with #[link("lib")] attributes
- Opaque type declarations (`type GtkWindow;`)
- Multiple extern blocks
- Various C types (c_int, c_long, c_double, size_t, etc.)
- Function pointers (param, return, call, assign)
- C callbacks
- Platform conditionals (#[cfg(...)])
- GTK4 integration

### Struct, Impl, Enum Tests (All passing)

Full support for:
- Struct definitions with various field types
- Impl blocks with methods
- Enum definitions with match expressions

### Native Platform Tests (7/10 compiling)

GTK tests that compile:
- 01-05: Basic GTK initialization, windows, labels, buttons, containers
- 07: Full widget hierarchy
- 08: Entry input

Failing tests (use type alias syntax):
- 06: Signal connect (uses `type GCallback = rite(...)`)
- 09: Callback with data (uses type alias)
- 10: GLib timeout (uses type alias)

## Running Tests

```bash
# From parser directory
cd parser
./tests/llvm/run_all_llvm_tests.sh

# Quick compilation test
for f in tests/llvm/*/*.sg; do
  ./target/release/sigil compile "$f" -o /tmp/test 2>/dev/null && echo "PASS: $f"
done
```

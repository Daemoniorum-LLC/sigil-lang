# Jormungandr TDD Roadmap: Path to Compiling Styx

**Goal:** Compile Styx TOME with Jormungandr using Test-Driven Development

**Methodology:**
1. ✍️ Write failing test for required feature
2. ❌ Verify test fails (RED)
3. 🔧 Implement feature in compiler
4. ✅ Verify test passes (GREEN)
5. 🧹 Refactor if needed
6. 📝 Commit with test + implementation

---

## Phase 1: Core Language Features (REQUIRED for Styx)

### 1.1 Basic Types & Structs ✅
**Status:** PASSING (verified)

**Test Cases:**
- [ ] `tests/001_struct_basic.sg` - Define struct with typed fields
- [ ] `tests/002_struct_impl.sg` - impl block with methods
- [ ] `tests/003_struct_instantiate.sg` - Create struct instance
- [ ] `tests/004_method_call.sg` - Call method on struct instance

**Success Criteria:**
- Compiles to C
- C code compiles without errors
- Binary runs and produces expected output

---

### 1.2 Traits ✅
**Status:** PASSING (method resolution fixed)

**Test Cases:**
- [ ] `tests/010_trait_define.sg` - Define trait with method signatures
- [ ] `tests/011_trait_impl.sg` - impl Trait for Type
- [ ] `tests/012_trait_call.sg` - Call trait method
- [ ] `tests/013_trait_generic.sg` - Trait with generic types
- [ ] `tests/014_trait_bounds.sg` - Function with trait bounds

**Success Criteria:**
- Trait methods generate `sigil_TypeName____method()` correctly
- Calling trait methods compiles and links
- Runtime behavior matches expectations

---

### 1.3 Enums & Pattern Matching ❓
**Status:** UNKNOWN (needs testing)

**Test Cases:**
- [ ] `tests/020_enum_basic.sg` - Define enum with variants
- [ ] `tests/021_enum_data.sg` - Enum variants with data
- [ ] `tests/022_match_basic.sg` - match expression on enum
- [ ] `tests/023_match_exhaustive.sg` - Exhaustiveness checking
- [ ] `tests/024_match_guard.sg` - match with guard clauses

**Success Criteria:**
- Enum variants compile correctly
- match generates correct C switch/if-else
- Pattern destructuring works
- Compiler enforces exhaustiveness

**Expected Failures:**
- Pattern matching may not be fully implemented
- Exhaustiveness checking might be missing

---

### 1.4 References & Borrowing ❓
**Status:** UNKNOWN (needs testing)

**Test Cases:**
- [ ] `tests/030_ref_immutable.sg` - &T references
- [ ] `tests/031_ref_mutable.sg` - &mut T references
- [ ] `tests/032_ref_method.sg` - Methods taking &self, &mut self
- [ ] `tests/033_ref_deref.sg` - Dereferencing *ref

**Success Criteria:**
- Reference types compile
- &self and &mut self handled correctly (already partially working)
- Dereferencing doesn't cause NULL crashes (fixed)

---

### 1.5 Generics 🔴
**Status:** FAILING (not implemented)

**Test Cases:**
- [ ] `tests/040_generic_function.sg` - fn foo<T>(x: T) -> T
- [ ] `tests/041_generic_struct.sg` - struct Container<T>
- [ ] `tests/042_generic_impl.sg` - impl<T> Container<T>
- [ ] `tests/043_generic_bounds.sg` - fn foo<T: Display>(x: T)
- [ ] `tests/044_multiple_params.sg` - <T, U, V>

**Success Criteria:**
- Generic functions monomorphize correctly
- Generic structs generate specialized types
- Trait bounds enforced at compile time

**Expected Failures:**
- Likely not implemented or incomplete
- May need full type system rewrite

---

### 1.6 Modules & Imports ❓
**Status:** UNKNOWN (needs testing)

**Test Cases:**
- [ ] `tests/050_mod_single.sg` - Single module file
- [ ] `tests/051_mod_nested.sg` - Nested modules (mod foo { })
- [ ] `tests/052_invoke_local.sg` - invoke tome::module
- [ ] `tests/053_invoke_external.sg` - invoke other_tome::item
- [ ] `tests/054_invoke_std.sg` - invoke std::collections::HashMap

**Success Criteria:**
- Module declarations work
- invoke resolves paths correctly
- tome:: prefix works for current tome
- External tome resolution works

**Expected Failures:**
- External tome resolution may not work
- std:: imports might be stubbed

---

### 1.7 Evidentiality Markers 🔴
**Status:** FAILING (not enforced)

**Test Cases:**
- [ ] `tests/060_evidence_known.sg` - !T types
- [ ] `tests/061_evidence_uncertain.sg` - ?T types
- [ ] `tests/062_evidence_reported.sg` - ~T types
- [ ] `tests/063_evidence_paradox.sg` - ‽T types
- [ ] `tests/064_evidence_coercion.sg` - Coercion rules

**Success Criteria:**
- Markers parse correctly
- Type system tracks evidentiality
- Coercion rules enforced

**Expected Failures:**
- Markers may parse but not be enforced
- Runtime checks might be missing

---

### 1.8 Arrays & Slices ❓
**Status:** UNKNOWN (needs testing)

**Test Cases:**
- [ ] `tests/070_array_literal.sg` - [1, 2, 3]
- [ ] `tests/071_array_index.sg` - arr[0]
- [ ] `tests/072_array_slice.sg` - &arr[1..3]
- [ ] `tests/073_array_methods.sg` - arr.len(), arr.push()

**Success Criteria:**
- Array literals compile
- Indexing works
- Methods on arrays work

---

### 1.9 Pipe Operators 🔴
**Status:** FAILING (not implemented)

**Test Cases:**
- [ ] `tests/080_pipe_basic.sg` - x|func
- [ ] `tests/081_pipe_chain.sg` - x|f|g|h
- [ ] `tests/082_pipe_morpheme.sg` - x|τ{_ * 2}
- [ ] `tests/083_pipe_method.sg` - x|obj.method

**Success Criteria:**
- | operator desugars correctly
- Morphemes (τ, φ, α, etc.) work
- Chaining preserves types

**Expected Failures:**
- Not implemented
- Morpheme syntax may not parse

---

### 1.10 Result & Option Types ❓
**Status:** UNKNOWN (needs testing)

**Test Cases:**
- [ ] `tests/090_option_some.sg` - Some(value)
- [ ] `tests/091_option_none.sg` - None
- [ ] `tests/092_result_ok.sg` - Ok(value)
- [ ] `tests/093_result_err.sg` - Err(error)
- [ ] `tests/094_question_mark.sg` - expr?

**Success Criteria:**
- Option/Result types work
- Pattern matching on them works
- ? operator desugars correctly

---

## Phase 2: Standard Library Features

### 2.1 String Methods ✅
**Status:** PASSING (fixed via multiline emission)

**Test Cases:**
- [x] `tests/100_string_clone.sg` - s.clone()
- [x] `tests/101_string_is_empty.sg` - s.is_empty()
- [x] `tests/102_string_contains.sg` - s.contains("x")
- [ ] `tests/103_string_split.sg` - s.split(",")
- [ ] `tests/104_string_trim.sg` - s.trim()

---

### 2.2 Vec Methods ❓
**Status:** UNKNOWN

**Test Cases:**
- [ ] `tests/110_vec_new.sg` - Vec::new()
- [ ] `tests/111_vec_push.sg` - v.push(item)
- [ ] `tests/112_vec_pop.sg` - v.pop()
- [ ] `tests/113_vec_len.sg` - v.len()
- [ ] `tests/114_vec_iter.sg` - v.iter()

---

### 2.3 HashMap Methods 🔴
**Status:** FAILING (not implemented)

**Test Cases:**
- [ ] `tests/120_map_new.sg` - HashMap::new()
- [ ] `tests/121_map_insert.sg` - m.insert(k, v)
- [ ] `tests/122_map_get.sg` - m.get(k)
- [ ] `tests/123_map_contains.sg` - m.contains_key(k)

---

## Phase 3: Styx-Specific Requirements

### 3.1 Arcanum Crypto FFI 🔴
**Status:** FAILING (FFI not implemented)

**Test Cases:**
- [ ] `tests/200_ffi_declare.sg` - extern "C" fn ...
- [ ] `tests/201_ffi_call.sg` - Call C function
- [ ] `tests/202_ffi_types.sg` - C-compatible types
- [ ] `tests/203_arcanum_sha3.sg` - Use arcanum::hash::sha3_256

**Expected Failures:**
- FFI likely not implemented
- extern "C" syntax may not parse

---

### 3.2 Complex Type Hierarchies ❓
**Status:** UNKNOWN

**Test Cases:**
- [ ] `tests/210_nested_generics.sg` - Vec<Option<T>>
- [ ] `tests/211_trait_objects.sg` - Box<dyn Trait>
- [ ] `tests/212_associated_types.sg` - trait with type Item
- [ ] `tests/213_where_clauses.sg` - where T: Trait + Other

---

### 3.3 Styx Core Compilation 🔴
**Status:** FAILING (prerequisites incomplete)

**Test Cases:**
- [ ] `tests/300_compile_id.sg` - Compile styx-core/src/id.sigil
- [ ] `tests/301_compile_error.sg` - Compile styx-core/src/error.sigil
- [ ] `tests/302_compile_config.sg` - Compile styx-core/src/config.sigil
- [ ] `tests/303_compile_lib.sg` - Compile full styx-core tome

**Success Criteria:**
- Each file compiles to C
- C code compiles without errors
- All styx-core files compile together
- Can link into library

---

## Test Infrastructure

### Test Directory Structure
```
tests/
├── features/              # Individual feature tests
│   ├── 001_struct_basic.sg
│   ├── 001_struct_basic.expected  # Expected output
│   ├── 002_struct_impl.sg
│   └── ...
├── stdlib/                # Standard library tests
│   ├── 100_string_clone.sg
│   └── ...
├── integration/           # Full program tests
│   ├── 300_compile_id.sg
│   └── ...
└── run_tests.sh          # Test runner script
```

### Test Runner Script
```bash
#!/bin/bash
# tests/run_tests.sh

SIGIL_COMPILER="./build/sigil2"
TEST_DIR="./tests"
PASS=0
FAIL=0

for test_file in "$TEST_DIR"/**/*.sg; do
    basename=$(basename "$test_file" .sg)
    expected="${test_file%.sg}.expected"

    echo "Testing: $basename"

    # Compile Sigil to C
    if ! "$SIGIL_COMPILER" compile "$test_file" -o "/tmp/${basename}.c" 2>/tmp/${basename}.err; then
        echo "  ❌ FAIL: Compilation failed"
        cat "/tmp/${basename}.err"
        ((FAIL++))
        continue
    fi

    # Compile C to binary
    if ! gcc -o "/tmp/${basename}" "/tmp/${basename}.c" -lm 2>/tmp/${basename}_gcc.err; then
        echo "  ❌ FAIL: C compilation failed"
        cat "/tmp/${basename}_gcc.err"
        ((FAIL++))
        continue
    fi

    # Run binary and compare output
    if [ -f "$expected" ]; then
        "/tmp/${basename}" > "/tmp/${basename}.out"
        if diff -q "$expected" "/tmp/${basename}.out" > /dev/null; then
            echo "  ✅ PASS"
            ((PASS++))
        else
            echo "  ❌ FAIL: Output mismatch"
            diff "$expected" "/tmp/${basename}.out"
            ((FAIL++))
        fi
    else
        echo "  ⚠️  WARN: No expected output file"
        ((PASS++))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL
```

---

## TDD Workflow Example

### Example: Implementing Pattern Matching

#### Step 1: Write Failing Test ✍️
```sigil
// tests/features/022_match_basic.sg
enum Color {
    Red,
    Green,
    Blue,
}

fn describe(c: Color) -> str {
    match c {
        Color::Red => "red",
        Color::Green => "green",
        Color::Blue => "blue",
    }
}

fn main() {
    let c = Color::Red;
    print(describe(c));
}
```

```
// tests/features/022_match_basic.expected
red
```

#### Step 2: Run Test (RED) ❌
```bash
./tests/run_tests.sh features/022_match_basic.sg
# Expected: ❌ FAIL: match not implemented
```

#### Step 3: Implement Feature 🔧
```sigil
// src/codegen.sg - Add match codegen
IrOperation::Match { .. } => {
    // Generate C switch statement or if-else chain
    ...
}
```

#### Step 4: Run Test (GREEN) ✅
```bash
./tests/run_tests.sh features/022_match_basic.sg
# Expected: ✅ PASS
```

#### Step 5: Commit 📝
```bash
git add tests/features/022_match_basic.sg src/codegen.sg
git commit -m "feat(codegen): implement basic match expressions

- Add match → C code generation
- Test: tests/features/022_match_basic.sg
- Supports enum matching with exhaustiveness check"
```

---

## Priority Order for Styx Compilation

### P0 - Critical (Must Have)
1. Traits & Methods ✅ DONE
2. Structs & Enums
3. Pattern Matching
4. Generics (at least basic <T>)
5. Module System (invoke/tome)

### P1 - Important (Should Have)
6. FFI (for Arcanum)
7. References & Borrowing
8. Option/Result types
9. Standard collections (Vec, HashMap)

### P2 - Nice to Have
10. Pipe operators
11. Evidentiality enforcement
12. Advanced generics (where clauses, associated types)

---

## Success Metrics

### Milestone 1: Feature Complete
- [ ] All P0 tests passing
- [ ] All P1 tests passing
- [ ] Test coverage >80%

### Milestone 2: Styx Core Compiles
- [ ] styx-core/src/id.sigil compiles
- [ ] styx-core/src/error.sigil compiles
- [ ] All styx-core files compile
- [ ] styx-core links into library

### Milestone 3: Full Styx Build
- [ ] All Styx tomes compile
- [ ] Styx binary builds
- [ ] Styx binary runs
- [ ] Styx passes its own tests

---

## Next Immediate Actions

1. **Create test infrastructure**
   ```bash
   mkdir -p tests/{features,stdlib,integration}
   touch tests/run_tests.sh
   chmod +x tests/run_tests.sh
   ```

2. **Write first batch of tests** (P0 features)
   - 001-004: Structs
   - 010-014: Traits
   - 020-024: Enums & Match
   - 040-044: Generics

3. **Run tests to establish baseline**
   ```bash
   ./tests/run_tests.sh > baseline.txt
   ```

4. **Fix failures one by one** (TDD cycle)
   - Pick failing test
   - Implement feature
   - Verify test passes
   - Commit
   - Repeat

5. **Track progress**
   - Update this document with test results
   - Mark features as ✅ PASSING or 🔴 FAILING
   - Document known issues

---

## Notes

- **Test First, Always**: Never implement without a failing test
- **Small Commits**: Each commit should include test + implementation
- **Document Failures**: When a test fails, document WHY
- **Refactor Fearlessly**: Tests give confidence to refactor
- **Focus on Styx**: Tests should prioritize features Styx needs

---

**Last Updated:** 2026-01-14
**Next Review:** After Phase 1 tests written

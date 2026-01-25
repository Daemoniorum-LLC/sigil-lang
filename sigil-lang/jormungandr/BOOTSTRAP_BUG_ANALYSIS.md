# Bootstrap Bug: Root Cause Analysis

## Bug Description

The Jormungandr self-hosting compiler loses function definitions during compilation, producing empty output files with no user code.

## Root Cause

### The Core Issue

When generating C code for **mutable reference parameters** (`&mut T`), the compiler incorrectly generates:

```c
sigil_struct_set_field(&param, "field", value);  // WRONG
```

Instead of:

```c
sigil_struct_set_field((SigilValue*)param.v.ptr, "field", value);  // CORRECT
```

### Why This Happens

1. Mutable reference parameters are passed as TAG_REF wrappers in C:
   ```c
   sigil_lower_item((SigilValue){ .tag = TAG_REF, .v.ptr = &module }, ...)
   ```

2. Inside the function, `module` is the TAG_REF wrapper, not the actual struct

3. When modifying fields, we must extract `.v.ptr` to get the actual struct pointer

4. The codegen was only checking `if base_code.starts_with("(*")` which doesn't catch mutable ref params

## Affected Functions

### In lower.sg:
- `lower_item(ctx: &mut LoweringContext, module: &mut IrModule, item: !Item)`
  - Line 144: `module.functions.push(ir_fn);`
  - Line 148: `module.types.push(...);`
  - Line 151: `module.types.push(...);`
  - Line 154: `module.types.push(...);`
  - Line 158: `module.constants.push(...);`
  - Line 162: `module.traits.push(...);`
  - Line 203: `module.functions.push(ir_fn);` (in impl block)
  - Line 208: `module.impls.push(...);`

All other functions take `&mut LoweringContext` but those modifications work because the bug only affects struct field assignments.

## The Fix

### Implementation in src/codegen.sg

Added three components:

1. **Track mutable reference parameters** (line 105):
```sigil
current_mut_ref_params: ![String],
```

2. **Populate in emit_function** (lines 2250-2256):
```sigil
// FIX: Collect all mutable reference parameters
self.current_mut_ref_params = [];
for p in func.params.iter() {
    if p.mutable {
        self.current_mut_ref_params.push(p.name.clone());
    }
}
```

3. **Use in struct_set_field generation** (lines 3230-3244 and 4098-4116):
```sigil
// FIX: Check if base is a mutable reference parameter
let mut is_mut_ref_param = false;
let mut i = 0;
while i < self.current_mut_ref_params.len() {
    if self.current_mut_ref_params[i] == base_var {
        is_mut_ref_param = true;
        break;
    }
    i = i + 1;
}
if is_mut_ref_param {
    self.line(format!("sigil_struct_set_field((SigilValue*){}.v.ptr, \"{}\", {});", base_var, field, value_code));
} else {
    self.line(format!("sigil_struct_set_field(&{}, \"{}\", {});", base_var, field, value_code));
}
```

## Bootstrap Chicken-and-Egg Problem

1. sigil1 (handwritten bootstrap) has the bug
2. When sigil1 compiles the fixed codegen.sg, it generates buggy C
3. The resulting binary still has the bug
4. Manual patching of the C file is required to break the cycle

## Manual Patch Locations

In the generated C file from sigil1, replace:
```c
sigil_struct_set_field(&module, "functions", ...)
sigil_struct_set_field(&module, "types", ...)
sigil_struct_set_field(&module, "constants", ...)
sigil_struct_set_field(&module, "traits", ...)
sigil_struct_set_field(&module, "impls", ...)
```

With:
```c
sigil_struct_set_field((SigilValue*)module.v.ptr, "functions", ...)
sigil_struct_set_field((SigilValue*)module.v.ptr, "types", ...)
sigil_struct_set_field((SigilValue*)module.v.ptr, "constants", ...)
sigil_struct_set_field((SigilValue*)module.v.ptr, "traits", ...)
sigil_struct_set_field((SigilValue*)module.v.ptr, "impls", ...)
```

## Status

- ✅ Fix implemented in src/codegen.sg
- ✅ Manual patch applied to generated C
- ✅ Patched binary compiles
- ❓ Testing reveals additional issues (functions still not appearing)

## Next Steps

1. Verify the patched binary can compile itself
2. Check if there are additional codegen issues
3. Test full bootstrap chain: sigil1 → sigil2 → sigil3
4. Verify fixed-point compilation

## Files Modified

- `src/codegen.sg` - Lines 105, 123, 2250-2256, 2307, 3230-3244, 4098-4116

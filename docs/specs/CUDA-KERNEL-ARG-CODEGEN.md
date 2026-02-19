# CUDA Kernel Argument Codegen

**Version:** 0.1.0
**Status:** Draft
**Author:** Claude (Opus 4.5)
**Date:** 2026-02-11

---

## 1. Problem Statement

### 1.1 Current Behavior

When launching CUDA kernels via `Cuda·launch_1d`, the user provides an array of argument values:

```sigil
≔ d_ptr = Cuda·malloc(1024);
≔ n = 64i64;
≔ Δ args = [d_ptr, n];
Cuda·launch_1d(kernel, 1, 64, args.as_ptr() as i64, 2);
```

The current codegen passes this directly to the runtime, which expects `void**` (CUDA Driver API convention).

### 1.2 The Gap

CUDA's `cuLaunchKernel` expects `void** args` where each element is a **pointer to** the argument value:

```c
// What CUDA expects
int* d_ptr = ...;
int n = 64;
void* args[] = { &d_ptr, &n };  // pointers TO variables
cuLaunchKernel(func, ..., args, NULL);
```

But Sigil's array contains the **values themselves**, not pointers to them. The runtime dereferences garbage.

### 1.3 Scope

This spec covers:
- Automatic void** generation in LLVM codegen for `Cuda·launch_1d` and `Cuda·launch_2d`
- Preserving backward compatibility (args still passed as array)

Out of scope:
- Interpreter mode (already works differently)
- New API surface changes

---

## 2. Desired Behavior

### 2.1 User Code (Unchanged)

```sigil
≔ d_ptr = Cuda·malloc(n * 8);
≔ n = 64i64;
≔ Δ args = [d_ptr, n];
≔ ok = Cuda·launch_1d(kernel, 1, 64, args.as_ptr() as i64, 2);
```

### 2.2 Generated Code (Conceptual)

When codegen sees `Cuda·launch_1d(kernel, grid, block, args_ptr, num_args)`:

```llvm
; 1. Read values from user's args array
%user_args = inttoptr i64 %args_ptr to ptr
%arg0_val = load i64, ptr %user_args          ; d_ptr value
%arg1_ptr = getelementptr i64, ptr %user_args, i32 1
%arg1_val = load i64, ptr %arg1_ptr           ; n value

; 2. Allocate stack slots for each arg
%slot0 = alloca i64
%slot1 = alloca i64
store i64 %arg0_val, ptr %slot0
store i64 %arg1_val, ptr %slot1

; 3. Build void** array of pointers to slots
%void_args = alloca [2 x ptr]
store ptr %slot0, ptr %void_args[0]
store ptr %slot1, ptr %void_args[1]

; 4. Call runtime with void**
%result = call i64 @sigil_cuda_launch_kernel_1d(
    i64 %kernel, i64 %grid, i64 %block,
    ptr %void_args, i64 2)
```

### 2.3 Invariants

1. User code syntax unchanged
2. All i64 args handled uniformly (pointers and integers both work)
3. Stack allocation ensures args live through kernel launch
4. No heap allocation required

---

## 3. Implementation Plan

### 3.1 Detection

In `compile_call()` when `func_name` matches `"Cuda::launch_1d"` or `"Cuda·launch_1d"`:
- Extract: kernel handle, grid_x, block_x, args_ptr, num_args
- If num_args > 0: apply void** transformation

### 3.2 Transformation Steps

```
1. Compile args_ptr expression → i64 value
2. Convert to pointer: inttoptr i64 → ptr
3. For i in 0..num_args:
   a. GEP to args[i]
   b. Load value
   c. Alloca stack slot
   d. Store value to slot
4. Alloca [num_args x ptr] for void_args
5. For i in 0..num_args:
   a. Store slot pointer to void_args[i]
6. Call runtime with void_args pointer
```

### 3.3 Edge Cases

| Case | Handling |
|------|----------|
| num_args = 0 | Pass null, skip transformation |
| num_args not literal | Runtime error (must be compile-time known) |
| args_ptr is null | Pass null, skip transformation |

---

## 4. Test Specification

### 4.1 Basic Kernel Launch

```sigil
// Test: kernel receives correct argument values
rite test_kernel_args() -> i32 {
    Cuda·init();

    ≔ kernel_src = "
extern \"C\" __global__ void write_val(long long* out, long long val) {
    if (threadIdx.x == 0) *out = val;
}";

    ≔ kernel = Cuda·compile_kernel(kernel_src, "write_val");
    ≔ d_out = Cuda·malloc(8);
    ≔ test_val = 0xDEADBEEFi64;

    ≔ Δ args = [d_out, test_val];
    Cuda·launch_1d(kernel, 1, 1, args.as_ptr() as i64, 2);
    Cuda·sync();

    ≔ Δ h_result = [0i64; 1];
    Cuda·memcpy_d2h(h_result.as_ptr() as i64, d_out, 8);

    Cuda·free(d_out);
    Cuda·cleanup();

    ⎇ h_result[0] == test_val { ↩ 0; }
    ↩ 1;
}
```

### 4.2 Multiple Arguments

```sigil
// Test: multiple args passed correctly
rite test_multiple_args() -> i32 {
    // Kernel that computes a*b+c and writes to out
    // Verify result matches expected value
}
```

### 4.3 Zero Arguments

```sigil
// Test: launch with no args doesn't crash
rite test_zero_args() -> i32 {
    // Empty kernel, just verify launch succeeds
}
```

---

## 5. Success Criteria

- [ ] Test 4.1 passes (basic kernel with args)
- [ ] Test 4.2 passes (multiple args)
- [ ] Test 4.3 passes (zero args)
- [ ] Existing CUDA tests still pass
- [ ] No heap allocations in generated code

---

## 6. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-11 | Initial spec |

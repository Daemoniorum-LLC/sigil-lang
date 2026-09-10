//! Sigil Nihil Runtime - FFI exports for LLVM-compiled Sigil code
//!
//! This provides CPU implementations of nihil tensor operations.
//! Phase 1: Minimal stubs to unblock stage7 initialization
//! Phase 2: CPU tensor operations
//! Phase 3: Autograd
//! Phase 4: CUDA acceleration

mod tensor;
mod cuda;
mod ops;

pub use tensor::*;
pub use cuda::*;

use std::ffi::c_void;

// Re-export FFI functions
pub use tensor::{
    nihil_tensor_randn,
    nihil_tensor_ones,
    nihil_tensor_zeros,
    nihil_tensor_from_vec,
    nihil_tensor_shape,
    nihil_tensor_ndim,
    nihil_tensor_numel,
    nihil_tensor_free,
    nihil_tensor_clone,
    nihil_tensor_to_vec,
    nihil_tensor_print,
};

pub use cuda::{
    nihil_cuda_current,
    nihil_cuda_device_id,
    nihil_cuda_is_ada,
    nihil_cuda_has_tensor_cores,
    nihil_cuda_free,
};

pub use ops::{
    nihil_tensor_add,
    nihil_tensor_sub,
    nihil_tensor_mul,
    nihil_tensor_div,
    nihil_tensor_neg,
    nihil_tensor_scale,
    nihil_tensor_matmul,
    nihil_tensor_transpose,
    nihil_tensor_reshape,
};

/// Initialize the nihil runtime (called once at program start)
#[no_mangle]
pub extern "C" fn nihil_init() {
    // Currently no-op, but could initialize thread pools, CUDA context, etc.
}

/// Shutdown the nihil runtime
#[no_mangle]
pub extern "C" fn nihil_shutdown() {
    // Currently no-op
}

/// Print debug info about runtime
#[no_mangle]
pub extern "C" fn nihil_debug_info() {
    eprintln!("[nihil] Runtime version: 0.1.0");
    eprintln!("[nihil] Backend: CPU");
    eprintln!("[nihil] Tensor size: {} bytes", std::mem::size_of::<tensor::Tensor>());
}

//! Tensor implementation for nihil runtime
//!
//! Tensor ABI matches what the Sigil LLVM backend expects.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

/// Tensor data type
#[repr(i64)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32 = 0,
    F64 = 1,
    I64 = 2,
    I32 = 3,
    Bool = 4,
}

/// Device type
#[repr(i64)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    Cpu = 0,
    Cuda = 1,
}

/// Tensor struct - matches Sigil's expected layout
#[repr(C)]
pub struct Tensor {
    /// Pointer to contiguous data
    pub data: *mut f32,
    /// Shape array
    pub shape: *mut i64,
    /// Stride array
    pub strides: *mut i64,
    /// Number of dimensions
    pub ndim: i64,
    /// Total number of elements
    pub numel: i64,
    /// Data type (0=f32, 1=f64, 2=i64, ...)
    pub dtype: i64,
    /// Device (0=CPU, 1=CUDA)
    pub device: i64,
    /// Reference count for memory management
    pub refcount: i64,
}

impl Tensor {
    /// Create a new tensor with given shape
    pub fn new(shape: &[i64], dtype: DType, device: Device) -> *mut Tensor {
        let ndim = shape.len() as i64;
        let numel: i64 = shape.iter().product();

        // Allocate data
        let data_layout = Layout::array::<f32>(numel as usize).unwrap();
        let data = unsafe { alloc(data_layout) as *mut f32 };

        // Allocate shape
        let shape_layout = Layout::array::<i64>(ndim as usize).unwrap();
        let shape_ptr = unsafe { alloc(shape_layout) as *mut i64 };
        unsafe {
            ptr::copy_nonoverlapping(shape.as_ptr(), shape_ptr, ndim as usize);
        }

        // Compute strides (row-major order)
        let strides_ptr = unsafe { alloc(shape_layout) as *mut i64 };
        let mut stride = 1i64;
        for i in (0..ndim as usize).rev() {
            unsafe { *strides_ptr.add(i) = stride; }
            stride *= shape[i];
        }

        // Allocate tensor struct
        let tensor = Box::new(Tensor {
            data,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim,
            numel,
            dtype: dtype as i64,
            device: device as i64,
            refcount: 1,
        });

        Box::into_raw(tensor)
    }

    /// Get shape as slice
    pub unsafe fn shape_slice(&self) -> &[i64] {
        std::slice::from_raw_parts(self.shape, self.ndim as usize)
    }

    /// Get data as slice
    pub unsafe fn data_slice(&self) -> &[f32] {
        std::slice::from_raw_parts(self.data, self.numel as usize)
    }

    /// Get data as mutable slice
    pub unsafe fn data_slice_mut(&mut self) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.data, self.numel as usize)
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            if !self.data.is_null() {
                let data_layout = Layout::array::<f32>(self.numel as usize).unwrap();
                dealloc(self.data as *mut u8, data_layout);
            }
            if !self.shape.is_null() {
                let shape_layout = Layout::array::<i64>(self.ndim as usize).unwrap();
                dealloc(self.shape as *mut u8, shape_layout);
            }
            if !self.strides.is_null() {
                let strides_layout = Layout::array::<i64>(self.ndim as usize).unwrap();
                dealloc(self.strides as *mut u8, strides_layout);
            }
        }
    }
}

// ============================================================================
// FFI Functions - Tensor Creation
// ============================================================================

/// Create a tensor filled with random values from N(0,1)
///
/// # Arguments
/// * `shape_ptr` - Pointer to shape array
/// * `ndim` - Number of dimensions
///
/// # Returns
/// Pointer to new Tensor, or null on error
#[no_mangle]
pub extern "C" fn nihil_tensor_randn(shape_ptr: *const i64, ndim: i64) -> *mut Tensor {
    if shape_ptr.is_null() || ndim <= 0 {
        return ptr::null_mut();
    }

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim as usize) };
    let tensor = Tensor::new(shape, DType::F32, Device::Cpu);

    if tensor.is_null() {
        return ptr::null_mut();
    }

    // Fill with random values from N(0,1)
    let mut rng = rand::thread_rng();
    let normal = StandardNormal;
    unsafe {
        let t = &mut *tensor;
        let data = t.data_slice_mut();
        for x in data.iter_mut() {
            *x = normal.sample(&mut rng) as f32;
        }
    }

    tensor
}

/// Create a tensor filled with ones
#[no_mangle]
pub extern "C" fn nihil_tensor_ones(shape_ptr: *const i64, ndim: i64) -> *mut Tensor {
    if shape_ptr.is_null() || ndim <= 0 {
        return ptr::null_mut();
    }

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim as usize) };
    let tensor = Tensor::new(shape, DType::F32, Device::Cpu);

    if tensor.is_null() {
        return ptr::null_mut();
    }

    // Fill with ones
    unsafe {
        let t = &mut *tensor;
        let data = t.data_slice_mut();
        for x in data.iter_mut() {
            *x = 1.0;
        }
    }

    tensor
}

/// Create a tensor filled with zeros
#[no_mangle]
pub extern "C" fn nihil_tensor_zeros(shape_ptr: *const i64, ndim: i64) -> *mut Tensor {
    if shape_ptr.is_null() || ndim <= 0 {
        return ptr::null_mut();
    }

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim as usize) };
    let tensor = Tensor::new(shape, DType::F32, Device::Cpu);

    if tensor.is_null() {
        return ptr::null_mut();
    }

    // Fill with zeros (already zeroed by alloc? be explicit)
    unsafe {
        let t = &mut *tensor;
        let data = t.data_slice_mut();
        for x in data.iter_mut() {
            *x = 0.0;
        }
    }

    tensor
}

/// Create a tensor from a vector of f32 values
#[no_mangle]
pub extern "C" fn nihil_tensor_from_vec(
    data_ptr: *const f32,
    data_len: i64,
    shape_ptr: *const i64,
    ndim: i64,
    _device: i64, // ignored for now, always CPU
) -> *mut Tensor {
    if data_ptr.is_null() || shape_ptr.is_null() || ndim <= 0 {
        return ptr::null_mut();
    }

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim as usize) };
    let expected_numel: i64 = shape.iter().product();

    if data_len != expected_numel {
        eprintln!("[nihil] from_vec: data length {} != shape product {}", data_len, expected_numel);
        return ptr::null_mut();
    }

    let tensor = Tensor::new(shape, DType::F32, Device::Cpu);

    if tensor.is_null() {
        return ptr::null_mut();
    }

    // Copy data
    unsafe {
        let t = &mut *tensor;
        ptr::copy_nonoverlapping(data_ptr, t.data, data_len as usize);
    }

    tensor
}

// ============================================================================
// FFI Functions - Tensor Properties
// ============================================================================

/// Get shape of tensor as pointer to i64 array
/// Returns pointer valid only while tensor is alive
#[no_mangle]
pub extern "C" fn nihil_tensor_shape(tensor: *const Tensor) -> *const i64 {
    if tensor.is_null() {
        return ptr::null();
    }
    unsafe { (*tensor).shape }
}

/// Get number of dimensions
#[no_mangle]
pub extern "C" fn nihil_tensor_ndim(tensor: *const Tensor) -> i64 {
    if tensor.is_null() {
        return 0;
    }
    unsafe { (*tensor).ndim }
}

/// Get total number of elements
#[no_mangle]
pub extern "C" fn nihil_tensor_numel(tensor: *const Tensor) -> i64 {
    if tensor.is_null() {
        return 0;
    }
    unsafe { (*tensor).numel }
}

/// Free a tensor
#[no_mangle]
pub extern "C" fn nihil_tensor_free(tensor: *mut Tensor) {
    if tensor.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(tensor);
    }
}

/// Clone a tensor (deep copy)
#[no_mangle]
pub extern "C" fn nihil_tensor_clone(tensor: *const Tensor) -> *mut Tensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let t = &*tensor;
        let shape = t.shape_slice();
        let new_tensor = Tensor::new(shape, DType::F32, Device::Cpu);

        if new_tensor.is_null() {
            return ptr::null_mut();
        }

        // Copy data
        let new_t = &mut *new_tensor;
        ptr::copy_nonoverlapping(t.data, new_t.data, t.numel as usize);

        new_tensor
    }
}

/// Convert tensor to vector (copies data out)
/// Caller must free the returned pointer with nihil_vec_free
#[no_mangle]
pub extern "C" fn nihil_tensor_to_vec(tensor: *const Tensor, out_len: *mut i64) -> *mut f32 {
    if tensor.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let t = &*tensor;
        let len = t.numel as usize;

        // Allocate output buffer
        let layout = Layout::array::<f32>(len).unwrap();
        let out = alloc(layout) as *mut f32;

        if out.is_null() {
            return ptr::null_mut();
        }

        // Copy data
        ptr::copy_nonoverlapping(t.data, out, len);
        *out_len = t.numel;

        out
    }
}

/// Print tensor info (for debugging)
#[no_mangle]
pub extern "C" fn nihil_tensor_print(tensor: *const Tensor) {
    if tensor.is_null() {
        eprintln!("[nihil] Tensor: null");
        return;
    }

    unsafe {
        let t = &*tensor;
        let shape = t.shape_slice();
        let data = t.data_slice();

        eprintln!("[nihil] Tensor: shape={:?}, numel={}, dtype={}, device={}",
                  shape, t.numel, t.dtype, t.device);

        // Print first few elements
        let preview_len = std::cmp::min(10, data.len());
        eprint!("[nihil]   data=[");
        for (i, x) in data[..preview_len].iter().enumerate() {
            if i > 0 { eprint!(", "); }
            eprint!("{:.4}", x);
        }
        if data.len() > preview_len {
            eprint!(", ...");
        }
        eprintln!("]");
    }
}

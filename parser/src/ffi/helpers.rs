//! FFI Runtime Helpers
//!
//! Provides runtime support for FFI operations including:
//! - String conversion (Sigil <-> C strings)
//! - Pointer manipulation
//! - Memory management for FFI allocations

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// A C string wrapper that owns its memory.
/// Automatically freed when dropped.
pub struct CStringOwned {
    ptr: *mut c_char,
    len: usize,
}

impl CStringOwned {
    /// Create a null pointer (represents empty/null string).
    pub fn null() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }

    /// Get the raw pointer for passing to C functions.
    pub fn as_ptr(&self) -> *const c_char {
        self.ptr
    }

    /// Get a mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_char {
        self.ptr
    }

    /// Check if this is a null pointer.
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    /// Get the length (excluding null terminator).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for CStringOwned {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                // Free the CString memory
                drop(CString::from_raw(self.ptr));
            }
        }
    }
}

// ============================================
// Runtime helper functions (called from JIT)
// ============================================

/// Convert a Sigil string (ptr, len) to a null-terminated C string.
/// Returns a pointer that must be freed with `sigil_cstring_free`.
#[no_mangle]
pub extern "C" fn sigil_string_to_cstring(ptr: *const u8, len: usize) -> *mut c_char {
    if ptr.is_null() || len == 0 {
        // Return empty string, not null
        match CString::new("") {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    } else {
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, len);
            match std::str::from_utf8(slice) {
                Ok(s) => match CString::new(s) {
                    Ok(cstr) => cstr.into_raw(),
                    Err(_) => std::ptr::null_mut(),
                },
                Err(_) => std::ptr::null_mut(),
            }
        }
    }
}

/// Free a C string allocated by `sigil_string_to_cstring`.
#[no_mangle]
pub extern "C" fn sigil_cstring_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

/// Convert a null-terminated C string to Sigil string representation.
/// Returns the length of the string. The pointer remains valid.
#[no_mangle]
pub extern "C" fn sigil_cstring_len(ptr: *const c_char) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { CStr::from_ptr(ptr).to_bytes().len() }
}

/// Copy a C string into a Sigil-managed buffer.
/// Returns the number of bytes copied.
#[no_mangle]
pub extern "C" fn sigil_cstring_copy(src: *const c_char, dst: *mut u8, dst_len: usize) -> usize {
    if src.is_null() || dst.is_null() || dst_len == 0 {
        return 0;
    }

    unsafe {
        let c_str = CStr::from_ptr(src);
        let bytes = c_str.to_bytes();
        let copy_len = bytes.len().min(dst_len);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, copy_len);
        copy_len
    }
}

// ============================================
// Pointer manipulation helpers
// ============================================

/// Create a pointer from an integer address.
#[no_mangle]
pub extern "C" fn sigil_ptr_from_int(addr: i64) -> *mut u8 {
    addr as *mut u8
}

/// Convert a pointer to an integer address.
#[no_mangle]
pub extern "C" fn sigil_ptr_to_int(ptr: *const u8) -> i64 {
    ptr as i64
}

/// Read a byte from a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_read_u8(ptr: *const u8) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { *ptr }
}

/// Write a byte to a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_write_u8(ptr: *mut u8, value: u8) {
    if !ptr.is_null() {
        unsafe {
            *ptr = value;
        }
    }
}

/// Read a 32-bit integer from a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_read_i32(ptr: *const i32) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { *ptr }
}

/// Write a 32-bit integer to a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_write_i32(ptr: *mut i32, value: i32) {
    if !ptr.is_null() {
        unsafe {
            *ptr = value;
        }
    }
}

/// Read a 64-bit integer from a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_read_i64(ptr: *const i64) -> i64 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { *ptr }
}

/// Write a 64-bit integer to a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_write_i64(ptr: *mut i64, value: i64) {
    if !ptr.is_null() {
        unsafe {
            *ptr = value;
        }
    }
}

/// Read a double from a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_read_f64(ptr: *const f64) -> f64 {
    if ptr.is_null() {
        return 0.0;
    }
    unsafe { *ptr }
}

/// Write a double to a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_write_f64(ptr: *mut f64, value: f64) {
    if !ptr.is_null() {
        unsafe {
            *ptr = value;
        }
    }
}

/// Add an offset to a pointer.
#[no_mangle]
pub extern "C" fn sigil_ptr_add(ptr: *const u8, offset: i64) -> *const u8 {
    if ptr.is_null() {
        return std::ptr::null();
    }
    unsafe { ptr.offset(offset as isize) }
}

/// Check if a pointer is null.
#[no_mangle]
pub extern "C" fn sigil_ptr_is_null(ptr: *const u8) -> i64 {
    if ptr.is_null() {
        1
    } else {
        0
    }
}

// ============================================
// Memory allocation helpers
// ============================================

/// Allocate memory of the given size.
/// Returns a pointer to the allocated memory, or null on failure.
#[no_mangle]
pub extern "C" fn sigil_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }

    let layout = match std::alloc::Layout::from_size_align(size, 8) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };

    unsafe { std::alloc::alloc_zeroed(layout) }
}

/// Free memory allocated by `sigil_alloc`.
#[no_mangle]
pub extern "C" fn sigil_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }

    let layout = match std::alloc::Layout::from_size_align(size, 8) {
        Ok(l) => l,
        Err(_) => return,
    };

    unsafe {
        std::alloc::dealloc(ptr, layout);
    }
}

/// Reallocate memory to a new size.
#[no_mangle]
pub extern "C" fn sigil_realloc(ptr: *mut u8, old_size: usize, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return sigil_alloc(new_size);
    }

    if new_size == 0 {
        sigil_free(ptr, old_size);
        return std::ptr::null_mut();
    }

    let old_layout = match std::alloc::Layout::from_size_align(old_size, 8) {
        Ok(l) => l,
        Err(_) => return std::ptr::null_mut(),
    };

    unsafe { std::alloc::realloc(ptr, old_layout, new_size) }
}

/// Copy memory from src to dst.
#[no_mangle]
pub extern "C" fn sigil_memcpy(dst: *mut u8, src: *const u8, size: usize) {
    if dst.is_null() || src.is_null() || size == 0 {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
}

/// Set memory to a value.
#[no_mangle]
pub extern "C" fn sigil_memset(ptr: *mut u8, value: u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    unsafe {
        std::ptr::write_bytes(ptr, value, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_conversion() {
        let rust_str = "Hello, World!";
        let bytes = rust_str.as_bytes();

        let c_str = sigil_string_to_cstring(bytes.as_ptr(), bytes.len());
        assert!(!c_str.is_null());

        let len = sigil_cstring_len(c_str);
        assert_eq!(len, rust_str.len());

        sigil_cstring_free(c_str);
    }

    #[test]
    fn test_pointer_operations() {
        let value: i64 = 42;
        let ptr = &value as *const i64;

        let addr = sigil_ptr_to_int(ptr as *const u8);
        let recovered_ptr = sigil_ptr_from_int(addr) as *const i64;

        let read_value = sigil_ptr_read_i64(recovered_ptr);
        assert_eq!(read_value, 42);
    }

    #[test]
    fn test_memory_allocation() {
        let ptr = sigil_alloc(1024);
        assert!(!ptr.is_null());

        // Write some data
        sigil_memset(ptr, 0xAB, 1024);

        // Verify
        unsafe {
            assert_eq!(*ptr, 0xAB);
            assert_eq!(*ptr.add(512), 0xAB);
        }

        sigil_free(ptr, 1024);
    }

    #[test]
    fn test_ptr_add() {
        let array: [u8; 4] = [1, 2, 3, 4];
        let ptr = array.as_ptr();

        let ptr2 = sigil_ptr_add(ptr, 2);
        assert_eq!(sigil_ptr_read_u8(ptr2), 3);
    }
}

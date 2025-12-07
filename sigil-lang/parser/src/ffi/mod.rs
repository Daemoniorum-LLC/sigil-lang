//! Foreign Function Interface support for Sigil.
//!
//! This module provides C type aliases and FFI utilities for
//! interfacing with C libraries.
//!
//! # Modules
//!
//! - `ctypes`: C type mappings (c_int, c_char, size_t, etc.)
//! - `helpers`: Runtime helpers for string/pointer conversion

pub mod ctypes;
pub mod helpers;

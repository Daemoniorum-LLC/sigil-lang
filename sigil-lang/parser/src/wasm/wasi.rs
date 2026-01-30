//! WASI (WebAssembly System Interface) support.
//!
//! Enables Sigil programs to run outside browsers using WASI-compatible runtimes
//! like Wasmtime, Wasmer, or WasmEdge.
//!
//! # Example
//!
//! ```ignore
//! // Compile to WASI
//! sigil wasm --target wasi program.sg -o program.wasm
//!
//! // Run with Wasmtime
//! wasmtime run program.wasm
//! ```

use wasm_encoder::ValType;

use super::ImportRegistry;

/// WASI target configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WasmTarget {
    /// Browser target with JS runtime imports (default)
    #[default]
    Browser,
    /// WASI target for standalone execution
    Wasi,
}

impl WasmTarget {
    /// Parse target from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "browser" | "web" | "js" => Some(Self::Browser),
            "wasi" | "wasm32-wasi" | "standalone" => Some(Self::Wasi),
            _ => None,
        }
    }

    /// Check if this is the WASI target.
    pub fn is_wasi(&self) -> bool {
        matches!(self, Self::Wasi)
    }

    /// Check if this is the browser target.
    pub fn is_browser(&self) -> bool {
        matches!(self, Self::Browser)
    }
}

/// WASI Preview 1 file descriptor constants.
pub mod fd {
    /// Standard input
    pub const STDIN: i32 = 0;
    /// Standard output
    pub const STDOUT: i32 = 1;
    /// Standard error
    pub const STDERR: i32 = 2;
}

/// WASI error codes.
pub mod errno {
    pub const SUCCESS: i32 = 0;
    pub const BADF: i32 = 8;
    pub const INVAL: i32 = 28;
    pub const NOSYS: i32 = 52;
}

/// Register WASI Preview 1 imports.
pub fn register_wasi_imports(registry: &mut ImportRegistry) {
    use ValType::*;

    // ============================================
    // File Descriptor Operations
    // ============================================

    // fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> errno
    // Write to a file descriptor
    registry.add_import(
        "wasi_snapshot_preview1",
        "fd_write",
        vec![I32, I32, I32, I32],
        vec![I32],
    );

    // fd_read(fd: i32, iovs: i32, iovs_len: i32, nread: i32) -> errno
    // Read from a file descriptor
    registry.add_import(
        "wasi_snapshot_preview1",
        "fd_read",
        vec![I32, I32, I32, I32],
        vec![I32],
    );

    // fd_close(fd: i32) -> errno
    // Close a file descriptor
    registry.add_import("wasi_snapshot_preview1", "fd_close", vec![I32], vec![I32]);

    // fd_seek(fd: i32, offset: i64, whence: i32, newoffset: i32) -> errno
    // Seek within a file
    registry.add_import(
        "wasi_snapshot_preview1",
        "fd_seek",
        vec![I32, I64, I32, I32],
        vec![I32],
    );

    // fd_fdstat_get(fd: i32, buf: i32) -> errno
    // Get file descriptor status
    registry.add_import(
        "wasi_snapshot_preview1",
        "fd_fdstat_get",
        vec![I32, I32],
        vec![I32],
    );

    // fd_prestat_get(fd: i32, buf: i32) -> errno
    // Get prestat (for pre-opened directories)
    registry.add_import(
        "wasi_snapshot_preview1",
        "fd_prestat_get",
        vec![I32, I32],
        vec![I32],
    );

    // fd_prestat_dir_name(fd: i32, path: i32, path_len: i32) -> errno
    // Get pre-opened directory name
    registry.add_import(
        "wasi_snapshot_preview1",
        "fd_prestat_dir_name",
        vec![I32, I32, I32],
        vec![I32],
    );

    // ============================================
    // Path Operations
    // ============================================

    // path_open(fd: i32, dirflags: i32, path: i32, path_len: i32, oflags: i32,
    //           fs_rights_base: i64, fs_rights_inheriting: i64, fdflags: i32, opened_fd: i32) -> errno
    registry.add_import(
        "wasi_snapshot_preview1",
        "path_open",
        vec![I32, I32, I32, I32, I32, I64, I64, I32, I32],
        vec![I32],
    );

    // ============================================
    // Environment & Arguments
    // ============================================

    // args_sizes_get(argc: i32, argv_buf_size: i32) -> errno
    // Get command line argument sizes
    registry.add_import(
        "wasi_snapshot_preview1",
        "args_sizes_get",
        vec![I32, I32],
        vec![I32],
    );

    // args_get(argv: i32, argv_buf: i32) -> errno
    // Get command line arguments
    registry.add_import(
        "wasi_snapshot_preview1",
        "args_get",
        vec![I32, I32],
        vec![I32],
    );

    // environ_sizes_get(environc: i32, environ_buf_size: i32) -> errno
    // Get environment variable sizes
    registry.add_import(
        "wasi_snapshot_preview1",
        "environ_sizes_get",
        vec![I32, I32],
        vec![I32],
    );

    // environ_get(environ: i32, environ_buf: i32) -> errno
    // Get environment variables
    registry.add_import(
        "wasi_snapshot_preview1",
        "environ_get",
        vec![I32, I32],
        vec![I32],
    );

    // ============================================
    // Clock Operations
    // ============================================

    // clock_time_get(clock_id: i32, precision: i64, time: i32) -> errno
    // Get current time
    registry.add_import(
        "wasi_snapshot_preview1",
        "clock_time_get",
        vec![I32, I64, I32],
        vec![I32],
    );

    // clock_res_get(clock_id: i32, resolution: i32) -> errno
    // Get clock resolution
    registry.add_import(
        "wasi_snapshot_preview1",
        "clock_res_get",
        vec![I32, I32],
        vec![I32],
    );

    // ============================================
    // Random
    // ============================================

    // random_get(buf: i32, buf_len: i32) -> errno
    // Get random bytes
    registry.add_import(
        "wasi_snapshot_preview1",
        "random_get",
        vec![I32, I32],
        vec![I32],
    );

    // ============================================
    // Process Control
    // ============================================

    // proc_exit(code: i32) -> !
    // Exit the process
    registry.add_import("wasi_snapshot_preview1", "proc_exit", vec![I32], vec![]);

    // ============================================
    // Sigil WASI Stdlib Aliases
    // ============================================

    // Register aliases for Sigil builtins that map to WASI
    // These will be implemented as wrapper functions in the compiled WASM

    // 'print' builtin - will use fd_write to stdout
    // (implemented as a generated wrapper, not a direct import)
}

/// WASI iovec structure layout.
pub mod iovec {
    /// Offset of the buffer pointer field
    pub const BUF_OFFSET: u32 = 0;
    /// Offset of the buffer length field
    pub const BUF_LEN_OFFSET: u32 = 4;
    /// Size of an iovec structure
    pub const SIZE: u32 = 8;
}

/// WASI clock IDs.
pub mod clock {
    /// Real-time clock (wall clock)
    pub const REALTIME: i32 = 0;
    /// Monotonic clock
    pub const MONOTONIC: i32 = 1;
    /// Process CPU time
    pub const PROCESS_CPUTIME_ID: i32 = 2;
    /// Thread CPU time
    pub const THREAD_CPUTIME_ID: i32 = 3;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_target_from_str() {
        assert_eq!(WasmTarget::from_str("wasi"), Some(WasmTarget::Wasi));
        assert_eq!(WasmTarget::from_str("WASI"), Some(WasmTarget::Wasi));
        assert_eq!(WasmTarget::from_str("browser"), Some(WasmTarget::Browser));
        assert_eq!(WasmTarget::from_str("web"), Some(WasmTarget::Browser));
        assert_eq!(WasmTarget::from_str("invalid"), None);
    }

    #[test]
    fn test_wasm_target_predicates() {
        assert!(WasmTarget::Wasi.is_wasi());
        assert!(!WasmTarget::Wasi.is_browser());
        assert!(WasmTarget::Browser.is_browser());
        assert!(!WasmTarget::Browser.is_wasi());
    }

    #[test]
    fn test_register_wasi_imports() {
        let mut registry = ImportRegistry::empty();
        register_wasi_imports(&mut registry);

        // Check that core WASI functions are registered
        assert!(registry
            .get_func("wasi_snapshot_preview1.fd_write")
            .is_some());
        assert!(registry
            .get_func("wasi_snapshot_preview1.fd_read")
            .is_some());
        assert!(registry
            .get_func("wasi_snapshot_preview1.proc_exit")
            .is_some());
        assert!(registry
            .get_func("wasi_snapshot_preview1.args_get")
            .is_some());
        assert!(registry
            .get_func("wasi_snapshot_preview1.clock_time_get")
            .is_some());
    }

    #[test]
    fn test_fd_constants() {
        assert_eq!(fd::STDIN, 0);
        assert_eq!(fd::STDOUT, 1);
        assert_eq!(fd::STDERR, 2);
    }
}

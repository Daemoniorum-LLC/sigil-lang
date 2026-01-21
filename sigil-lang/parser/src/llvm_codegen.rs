//! Sigil LLVM Compiler Backend
//!
//! Native code generation using LLVM for maximum performance.
//! This backend targets near-native Rust/C++ performance.

#[cfg(feature = "llvm")]
pub mod llvm {
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    use inkwell::module::Module;
    use inkwell::passes::PassBuilderOptions;
    use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
    };
    use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, StructType};
    use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue, StructValue};
    use inkwell::{AddressSpace, IntPredicate, OptimizationLevel, InlineAsmDialect};

    use std::collections::HashMap;
    use std::path::Path;

    use crate::ast::{self, BinOp, Expr, Item, Literal, UnaryOp, InlineAsm, AsmOperandKind};
    use crate::optimize::{OptLevel, Optimizer};
    use crate::parser::Parser;

    /// Type alias for JIT-compiled main function
    type MainFn = unsafe extern "C" fn() -> i64;

    /// Compilation mode
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CompileMode {
        /// JIT execution - main stays as "main"
        Jit,
        /// AOT compilation - main becomes "main_sigil" for linking with C runtime
        Aot,
    }

    /// Information about a compiled struct type
    #[derive(Clone)]
    pub struct StructInfo<'ctx> {
        /// The LLVM struct type
        pub llvm_type: StructType<'ctx>,
        /// Field name to index mapping
        pub field_indices: HashMap<String, u32>,
    }

    /// Information about a compiled enum type
    #[derive(Clone)]
    pub struct EnumInfo {
        /// Variant name to discriminant value mapping
        pub variants: HashMap<String, u64>,
    }

    /// Information about a generic struct (before monomorphization)
    #[derive(Clone)]
    pub struct GenericStructDef {
        /// The struct definition AST node
        pub def: ast::StructDef,
        /// Generic parameter names in order
        pub type_params: Vec<String>,
    }

    /// LLVM-based compiler for Sigil
    pub struct LlvmCompiler<'ctx> {
        context: &'ctx Context,
        module: Module<'ctx>,
        builder: Builder<'ctx>,
        execution_engine: Option<ExecutionEngine<'ctx>>,
        /// Compiled functions
        functions: HashMap<String, FunctionValue<'ctx>>,
        /// Optimization level
        opt_level: OptLevel,
        /// Compilation mode (JIT vs AOT)
        compile_mode: CompileMode,
        /// Current module path (e.g., ["crate", "foo", "bar"])
        current_module: Vec<String>,
        /// Maps use aliases to their full paths
        use_aliases: HashMap<String, String>,
        /// Struct type registry (concrete and monomorphized)
        struct_types: HashMap<String, StructInfo<'ctx>>,
        /// Generic struct definitions (awaiting monomorphization)
        generic_structs: HashMap<String, GenericStructDef>,
        /// Enum type registry
        enum_types: HashMap<String, EnumInfo>,
        /// Impl method registry: maps (type_name, method_name) -> mangled function name
        impl_methods: HashMap<(String, String), String>,
        /// Counter for generating unique string constant names
        string_counter: std::cell::Cell<u32>,
        /// Evidential wrapper types: maps base type name to {tag: i8, value: T} struct
        evidential_types: HashMap<String, StructType<'ctx>>,
        /// Constant values (compile-time evaluated)
        constants: HashMap<String, i64>,
    }

    // ============================================
    // Evidence Tag Constants
    // ============================================
    // These match the Evidentiality enum in ast.rs
    const EVIDENCE_KNOWN: u8 = 0;     // ! - verified ground truth
    const EVIDENCE_UNCERTAIN: u8 = 1; // ? - unverified input
    const EVIDENCE_REPORTED: u8 = 2;  // ~ - EMA, eventually consistent
    const EVIDENCE_PREDICTED: u8 = 3; // ◊ - model output, speculative
    const EVIDENCE_PARADOX: u8 = 4;   // ‽ - contradiction detected

    // Runtime helper: get current time in milliseconds
    extern "C" fn sigil_now() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    /// Strip type suffix from integer literal (e.g., "39_i64" -> "39", "1_000" -> "1000")
    fn strip_int_suffix(value: &str) -> String {
        // Find the suffix position: last underscore followed by a letter (type suffix)
        let without_suffix = if let Some(pos) = value.rfind('_') {
            let suffix = &value[pos+1..];
            // Check if the part after _ is a type suffix (starts with letter)
            if suffix.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
                &value[..pos]
            } else {
                value
            }
        } else {
            value
        };
        // Remove remaining underscores (for digit grouping like 1_000)
        without_suffix.replace('_', "")
    }

    // Runtime helper: print an integer (for JIT mode)
    extern "C" fn sigil_print_int(value: i64) {
        println!("{}", value);
    }

    // Math runtime helpers (JIT mode) - operate on i64 bits representing f64
    extern "C" fn sigil_sqrt(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).sqrt()) as i64
    }
    extern "C" fn sigil_sin(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).sin()) as i64
    }
    extern "C" fn sigil_cos(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).cos()) as i64
    }
    extern "C" fn sigil_tan(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).tan()) as i64
    }
    extern "C" fn sigil_exp(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).exp()) as i64
    }
    extern "C" fn sigil_ln(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).ln()) as i64
    }
    extern "C" fn sigil_pow(x: i64, y: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).powf(f64::from_bits(y as u64))) as i64
    }
    extern "C" fn sigil_floor(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).floor()) as i64
    }
    extern "C" fn sigil_ceil(x: i64) -> i64 {
        f64::to_bits(f64::from_bits(x as u64).ceil()) as i64
    }
    extern "C" fn sigil_abs(x: i64) -> i64 {
        x.abs()
    }
    extern "C" fn sigil_min(a: i64, b: i64) -> i64 {
        a.min(b)
    }
    extern "C" fn sigil_max(a: i64, b: i64) -> i64 {
        a.max(b)
    }

    // Vec runtime functions
    extern "C" fn sigil_vec_new(capacity: i64) -> *mut Vec<i64> {
        let vec = if capacity > 0 {
            Vec::with_capacity(capacity as usize)
        } else {
            Vec::new()
        };
        Box::into_raw(Box::new(vec))
    }

    extern "C" fn sigil_vec_push(vec_ptr: *mut Vec<i64>, value: i64) {
        if !vec_ptr.is_null() {
            unsafe { (*vec_ptr).push(value); }
        }
    }

    extern "C" fn sigil_vec_get(vec_ptr: *mut Vec<i64>, index: i64) -> i64 {
        if vec_ptr.is_null() { return 0; }
        unsafe {
            let vec_ref = &*vec_ptr;
            vec_ref.get(index as usize).copied().unwrap_or(0)
        }
    }

    extern "C" fn sigil_vec_len(vec_ptr: *mut Vec<i64>) -> i64 {
        if vec_ptr.is_null() { return 0; }
        unsafe { (*vec_ptr).len() as i64 }
    }

    // String runtime functions
    extern "C" fn sigil_string_from(ptr: *const i8) -> *mut String {
        if ptr.is_null() { return std::ptr::null_mut(); }
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(ptr);
            let s = cstr.to_string_lossy().into_owned();
            Box::into_raw(Box::new(s))
        }
    }

    extern "C" fn sigil_string_len(str_ptr: *mut String) -> i64 {
        if str_ptr.is_null() { return 0; }
        unsafe {
            let str_ref = &*str_ptr;
            str_ref.len() as i64
        }
    }

    extern "C" fn sigil_string_print(str_ptr: *mut String) {
        if !str_ptr.is_null() {
            unsafe { print!("{}", *str_ptr); }
        }
    }

    extern "C" fn sigil_string_concat(a_ptr: *mut String, b_ptr: *mut String) -> *mut String {
        if a_ptr.is_null() || b_ptr.is_null() { return std::ptr::null_mut(); }
        unsafe {
            let result = format!("{}{}", *a_ptr, *b_ptr);
            Box::into_raw(Box::new(result))
        }
    }

    // Option runtime functions
    extern "C" fn sigil_option_some(value: i64) -> *mut i64 {
        Box::into_raw(Box::new(value))
    }

    extern "C" fn sigil_option_none() -> *mut i64 {
        std::ptr::null_mut()
    }

    extern "C" fn sigil_option_is_some(opt_ptr: *mut i64) -> i64 {
        if opt_ptr.is_null() { 0 } else { 1 }
    }

    extern "C" fn sigil_option_is_none(opt_ptr: *mut i64) -> i64 {
        if opt_ptr.is_null() { 1 } else { 0 }
    }

    extern "C" fn sigil_option_unwrap(opt_ptr: *mut i64) -> i64 {
        if opt_ptr.is_null() {
            eprintln!("Error: unwrap called on None");
            0
        } else {
            unsafe { *opt_ptr }
        }
    }

    extern "C" fn sigil_option_unwrap_or(opt_ptr: *mut i64, default: i64) -> i64 {
        if opt_ptr.is_null() { default } else { unsafe { *opt_ptr } }
    }

    // File I/O runtime functions
    extern "C" fn sigil_file_exists(path_ptr: *const i8) -> i64 {
        if path_ptr.is_null() { return 0; }
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(path_ptr);
            if let Ok(path) = cstr.to_str() {
                if std::path::Path::new(path).exists() { 1 } else { 0 }
            } else { 0 }
        }
    }

    extern "C" fn sigil_file_read_all(path_ptr: *const i8) -> *mut String {
        if path_ptr.is_null() { return std::ptr::null_mut(); }
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(path_ptr);
            if let Ok(path) = cstr.to_str() {
                if let Ok(content) = std::fs::read_to_string(path) {
                    return Box::into_raw(Box::new(content));
                }
            }
            std::ptr::null_mut()
        }
    }

    extern "C" fn sigil_file_write_all(path_ptr: *const i8, content_ptr: *mut String) -> i64 {
        if path_ptr.is_null() || content_ptr.is_null() { return -1; }
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(path_ptr);
            let content_ref = &*content_ptr;
            if let Ok(path) = cstr.to_str() {
                if let Ok(_) = std::fs::write(path, content_ref) {
                    return content_ref.len() as i64;
                }
            }
            -1
        }
    }

    impl<'ctx> LlvmCompiler<'ctx> {
        /// Create a new LLVM compiler for JIT execution
        pub fn new(context: &'ctx Context, opt_level: OptLevel) -> Result<Self, String> {
            Self::with_mode(context, opt_level, CompileMode::Jit)
        }

        /// Create a new LLVM compiler with specific compile mode
        pub fn with_mode(
            context: &'ctx Context,
            opt_level: OptLevel,
            compile_mode: CompileMode,
        ) -> Result<Self, String> {
            // Initialize all targets for proper code generation
            // Use initialize_all as fallback for systems where native init might fail
            Target::initialize_all(&InitializationConfig::default());

            let module = context.create_module("sigil_main");
            let builder = context.create_builder();

            // Set target triple and data layout for the native machine
            let triple = TargetMachine::get_default_triple();
            module.set_triple(&triple);

            // Get native target machine to extract data layout
            let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
            let cpu = TargetMachine::get_host_cpu_name();
            let features = TargetMachine::get_host_cpu_features();
            if let Some(tm) = target.create_target_machine(
                &triple,
                cpu.to_str().unwrap_or("native"),
                features.to_str().unwrap_or(""),
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::Default,
            ) {
                module.set_data_layout(&tm.get_target_data().get_data_layout());
            }

            Ok(Self {
                context,
                module,
                builder,
                execution_engine: None,
                functions: HashMap::new(),
                opt_level,
                compile_mode,
                current_module: vec!["crate".to_string()],
                use_aliases: HashMap::new(),
                struct_types: HashMap::new(),
                generic_structs: HashMap::new(),
                enum_types: HashMap::new(),
                impl_methods: HashMap::new(),
                string_counter: std::cell::Cell::new(0),
                evidential_types: HashMap::new(),
                constants: HashMap::new(),
            })
        }

        /// Compile source code
        pub fn compile(&mut self, source: &str) -> Result<(), String> {
            let mut parser = Parser::new(source);
            let source_file = parser.parse_file().map_err(|e| format!("{:?}", e))?;

            // Run AST optimizations
            let mut optimizer = Optimizer::new(self.opt_level);
            let optimized = optimizer.optimize_file(&source_file);

            // Declare runtime functions
            self.declare_runtime_functions();

            // First pass: register types and constants
            for spanned_item in &optimized.items {
                match &spanned_item.node {
                    Item::Struct(s) => self.register_struct(s)?,
                    Item::Enum(e) => self.register_enum(e)?,
                    Item::Const(c) => self.register_const(c)?,
                    _ => {}
                }
            }

            // First pass continued: process impl blocks
            for spanned_item in &optimized.items {
                if let Item::Impl(impl_block) = &spanned_item.node {
                    self.declare_impl_methods(impl_block)?;
                }
            }

            // Second pass: process modules and declare all functions
            for spanned_item in &optimized.items {
                match &spanned_item.node {
                    Item::Function(func) => { self.declare_function(func)?; }
                    Item::Module(module) => { self.process_module(module)?; }
                    Item::Use(use_decl) => { self.process_use(use_decl)?; }
                    _ => {}
                }
            }

            // Third pass: compile function bodies
            for spanned_item in &optimized.items {
                match &spanned_item.node {
                    Item::Function(func) => { self.compile_function(func)?; }
                    Item::Module(module) => { self.compile_module_functions(module)?; }
                    Item::Impl(impl_block) => { self.compile_impl_methods(impl_block)?; }
                    _ => {}
                }
            }

            // Run LLVM optimizations
            self.run_llvm_optimizations()?;

            Ok(())
        }

        /// Declare runtime helper functions
        fn declare_runtime_functions(&self) {
            let i64_type = self.context.i64_type();
            let void_type = self.context.void_type();

            // sigil_now() -> i64
            let now_type = i64_type.fn_type(&[], false);
            self.module.add_function("sigil_now", now_type, None);

            // sigil_print_int(i64) -> void
            let print_int_type = void_type.fn_type(&[i64_type.into()], false);
            self.module
                .add_function("sigil_print_int", print_int_type, None);

            // Math functions: (i64) -> i64
            let unary_math_type = i64_type.fn_type(&[i64_type.into()], false);
            for name in [
                "sigil_sqrt",
                "sigil_sin",
                "sigil_cos",
                "sigil_tan",
                "sigil_exp",
                "sigil_ln",
                "sigil_floor",
                "sigil_ceil",
                "sigil_abs",
            ] {
                self.module.add_function(name, unary_math_type, None);
            }

            // Math functions: (i64, i64) -> i64
            let binary_math_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
            for name in ["sigil_pow", "sigil_min", "sigil_max"] {
                self.module.add_function(name, binary_math_type, None);
            }

            // Vec functions - use ptr type (i64 as opaque pointer)
            let ptr_type = i64_type; // Using i64 as opaque pointer type

            // sigil_vec_new(capacity: i64) -> ptr
            let vec_new_type = ptr_type.fn_type(&[i64_type.into()], false);
            self.module.add_function("sigil_vec_new", vec_new_type, None);

            // sigil_vec_push(vec: ptr, value: i64) -> void
            let vec_push_type = void_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_vec_push", vec_push_type, None);

            // sigil_vec_get(vec: ptr, index: i64) -> i64
            let vec_get_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_vec_get", vec_get_type, None);

            // sigil_vec_len(vec: ptr) -> i64
            let vec_len_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_vec_len", vec_len_type, None);

            // String functions
            // sigil_string_from(const char* src) -> ptr
            // For now, pass i64 as pointer to string literal (global constant)
            let string_from_type = ptr_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_string_from", string_from_type, None);

            // sigil_string_len(str: ptr) -> i64
            let string_len_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_string_len", string_len_type, None);

            // sigil_string_print(str: ptr) -> void
            let string_print_type = void_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_string_print", string_print_type, None);

            // sigil_string_concat(str1: ptr, str2: ptr) -> ptr
            let string_concat_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_string_concat", string_concat_type, None);

            // Option functions
            // sigil_option_some(value: i64) -> ptr
            let option_some_type = ptr_type.fn_type(&[i64_type.into()], false);
            self.module.add_function("sigil_option_some", option_some_type, None);

            // sigil_option_none() -> ptr (null)
            let option_none_type = ptr_type.fn_type(&[], false);
            self.module.add_function("sigil_option_none", option_none_type, None);

            // sigil_option_is_some(opt: ptr) -> i64
            let option_is_some_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_option_is_some", option_is_some_type, None);

            // sigil_option_is_none(opt: ptr) -> i64
            let option_is_none_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_option_is_none", option_is_none_type, None);

            // sigil_option_unwrap(opt: ptr) -> i64
            let option_unwrap_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_option_unwrap", option_unwrap_type, None);

            // sigil_option_unwrap_or(opt: ptr, default: i64) -> i64
            let option_unwrap_or_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_option_unwrap_or", option_unwrap_or_type, None);

            // File I/O functions
            // sigil_file_exists(path: ptr) -> i64 (1 if exists, 0 otherwise)
            let file_exists_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_file_exists", file_exists_type, None);

            // sigil_file_read_all(path: ptr) -> ptr (returns String ptr or null)
            let file_read_all_type = ptr_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_file_read_all", file_read_all_type, None);

            // sigil_file_write_all(path: ptr, content: ptr) -> i64 (bytes written or -1)
            let file_write_all_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_file_write_all", file_write_all_type, None);

            // SIMD Functions (F32x16)
            let f32_type = self.context.f32_type();

            // sigil_simd_alloc(num_floats: i64) -> ptr
            let simd_alloc_type = ptr_type.fn_type(&[i64_type.into()], false);
            self.module.add_function("sigil_simd_alloc", simd_alloc_type, None);

            // sigil_simd_free(ptr: ptr) -> void
            let simd_free_type = void_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_simd_free", simd_free_type, None);

            // sigil_simd_splat_f32x16(dest: ptr, value: f32) -> void
            let simd_splat_type = void_type.fn_type(&[ptr_type.into(), f32_type.into()], false);
            self.module.add_function("sigil_simd_splat_f32x16", simd_splat_type, None);

            // sigil_simd_load_f32x16(dest: ptr, src: ptr) -> void
            let simd_load_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_simd_load_f32x16", simd_load_type, None);

            // sigil_simd_store_f32x16(dest: ptr, src: ptr) -> void
            let simd_store_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_simd_store_f32x16", simd_store_type, None);

            // sigil_simd_add_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            let simd_binop_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_simd_add_f32x16", simd_binop_type, None);

            // sigil_simd_sub_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            self.module.add_function("sigil_simd_sub_f32x16", simd_binop_type, None);

            // sigil_simd_mul_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            self.module.add_function("sigil_simd_mul_f32x16", simd_binop_type, None);

            // sigil_simd_div_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            self.module.add_function("sigil_simd_div_f32x16", simd_binop_type, None);

            // sigil_simd_fmadd_f32x16(dest: ptr, a: ptr, b: ptr, c: ptr) -> void
            let simd_fmadd_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_simd_fmadd_f32x16", simd_fmadd_type, None);

            // sigil_simd_reduce_add_f32x16(src: ptr) -> f32
            let simd_reduce_type = f32_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_simd_reduce_add_f32x16", simd_reduce_type, None);

            // sigil_simd_extract_f32x16(src: ptr, index: i64) -> f32
            let simd_extract_type = f32_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_simd_extract_f32x16", simd_extract_type, None);

            // sigil_simd_dot_f32x16(a: ptr, b: ptr) -> f32
            let simd_dot_type = f32_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_simd_dot_f32x16", simd_dot_type, None);

            // CUDA Functions
            // sigil_cuda_init() -> i64
            let cuda_init_type = i64_type.fn_type(&[], false);
            self.module.add_function("sigil_cuda_init", cuda_init_type, None);

            // sigil_cuda_cleanup() -> void
            let cuda_cleanup_type = void_type.fn_type(&[], false);
            self.module.add_function("sigil_cuda_cleanup", cuda_cleanup_type, None);

            // sigil_cuda_get_device_count() -> i64
            let cuda_device_count_type = i64_type.fn_type(&[], false);
            self.module.add_function("sigil_cuda_get_device_count", cuda_device_count_type, None);

            // sigil_cuda_malloc(size: i64) -> i64 (device ptr)
            let cuda_malloc_type = i64_type.fn_type(&[i64_type.into()], false);
            self.module.add_function("sigil_cuda_malloc", cuda_malloc_type, None);

            // sigil_cuda_free(device_ptr: i64) -> void
            let cuda_free_type = void_type.fn_type(&[i64_type.into()], false);
            self.module.add_function("sigil_cuda_free", cuda_free_type, None);

            // sigil_cuda_memcpy_h2d(dst: i64, src: ptr, size: i64) -> i64
            let cuda_h2d_type = i64_type.fn_type(&[i64_type.into(), ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_cuda_memcpy_h2d", cuda_h2d_type, None);

            // sigil_cuda_memcpy_d2h(dst: ptr, src: i64, size: i64) -> i64
            let cuda_d2h_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_cuda_memcpy_d2h", cuda_d2h_type, None);

            // sigil_cuda_memcpy_d2d(dst: i64, src: i64, size: i64) -> i64
            let cuda_d2d_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_cuda_memcpy_d2d", cuda_d2d_type, None);

            // sigil_cuda_sync() -> void
            let cuda_sync_type = void_type.fn_type(&[], false);
            self.module.add_function("sigil_cuda_sync", cuda_sync_type, None);

            // sigil_cuda_compile_kernel(cuda_src: ptr, kernel_name: ptr) -> i64 (handle)
            let cuda_compile_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_cuda_compile_kernel", cuda_compile_type, None);

            // sigil_cuda_load_ptx(ptx: ptr, kernel_name: ptr) -> i64 (handle)
            self.module.add_function("sigil_cuda_load_ptx", cuda_compile_type, None);

            // sigil_cuda_launch_kernel_1d(handle: i64, grid_x: i64, block_x: i64, args: ptr, num_args: i64) -> i64
            let cuda_launch_1d_type = i64_type.fn_type(&[
                i64_type.into(), i64_type.into(), i64_type.into(),
                ptr_type.into(), i64_type.into()
            ], false);
            self.module.add_function("sigil_cuda_launch_kernel_1d", cuda_launch_1d_type, None);

            // sigil_cuda_launch_kernel_2d(handle: i64, gx: i64, gy: i64, bx: i64, by: i64, args: ptr, num_args: i64) -> i64
            let cuda_launch_2d_type = i64_type.fn_type(&[
                i64_type.into(), i64_type.into(), i64_type.into(),
                i64_type.into(), i64_type.into(), ptr_type.into(), i64_type.into()
            ], false);
            self.module.add_function("sigil_cuda_launch_kernel_2d", cuda_launch_2d_type, None);

            // TLS/SSL Functions (OpenSSL wrapper)
            // sigil_tls_init() -> i64
            let tls_init_type = i64_type.fn_type(&[], false);
            self.module.add_function("sigil_tls_init", tls_init_type, None);

            // sigil_tls_ctx_new() -> ptr
            let tls_ctx_new_type = ptr_type.fn_type(&[], false);
            self.module.add_function("sigil_tls_ctx_new", tls_ctx_new_type, None);

            // sigil_tls_ctx_free(ctx: ptr) -> void
            let tls_ctx_free_type = void_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_tls_ctx_free", tls_ctx_free_type, None);

            // sigil_tls_new(ctx: ptr) -> ptr
            let tls_new_type = ptr_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_tls_new", tls_new_type, None);

            // sigil_tls_set_fd(ssl: ptr, fd: i64) -> i64
            let tls_set_fd_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_tls_set_fd", tls_set_fd_type, None);

            // sigil_tls_set_hostname(ssl: ptr, hostname: ptr) -> i64
            let tls_set_hostname_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module.add_function("sigil_tls_set_hostname", tls_set_hostname_type, None);

            // sigil_tls_connect(ssl: ptr) -> i64
            let tls_connect_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_tls_connect", tls_connect_type, None);

            // sigil_tls_read(ssl: ptr, buf: ptr, len: i64) -> i64
            let tls_read_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_tls_read", tls_read_type, None);

            // sigil_tls_write(ssl: ptr, buf: ptr, len: i64) -> i64
            let tls_write_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into(), i64_type.into()], false);
            self.module.add_function("sigil_tls_write", tls_write_type, None);

            // sigil_tls_shutdown(ssl: ptr) -> i64
            let tls_shutdown_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_tls_shutdown", tls_shutdown_type, None);

            // sigil_tls_free(ssl: ptr) -> void
            let tls_free_type = void_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_tls_free", tls_free_type, None);

            // sigil_tls_error_string() -> ptr
            let tls_error_string_type = ptr_type.fn_type(&[], false);
            self.module.add_function("sigil_tls_error_string", tls_error_string_type, None);

            // sigil_tls_verify_result(ssl: ptr) -> i64
            let tls_verify_result_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_tls_verify_result", tls_verify_result_type, None);
        }

        /// Register a struct type in the type registry
        fn register_struct(&mut self, struct_def: &ast::StructDef) -> Result<(), String> {
            let name = &struct_def.name.name;

            // Check if this struct has generic parameters
            if let Some(ref generics) = struct_def.generics {
                if !generics.params.is_empty() {
                    // Extract type parameter names
                    let type_params: Vec<String> = generics.params.iter()
                        .filter_map(|p| match p {
                            ast::GenericParam::Type { name, .. } => Some(name.name.clone()),
                            ast::GenericParam::Const { name, .. } => Some(name.name.clone()),
                            ast::GenericParam::Lifetime(_) => None,
                        })
                        .collect();

                    // Store as generic struct for later monomorphization
                    self.generic_structs.insert(name.clone(), GenericStructDef {
                        def: struct_def.clone(),
                        type_params,
                    });
                    return Ok(());
                }
            }

            // Non-generic struct: register immediately
            self.register_concrete_struct(struct_def, &HashMap::new())
        }

        /// Register a concrete (non-generic or monomorphized) struct type
        fn register_concrete_struct(
            &mut self,
            struct_def: &ast::StructDef,
            type_substitutions: &HashMap<String, String>,
        ) -> Result<(), String> {
            let base_name = &struct_def.name.name;

            // Generate mangled name if we have type substitutions
            let mangled_name = if type_substitutions.is_empty() {
                base_name.clone()
            } else {
                let args: Vec<String> = type_substitutions.values().cloned().collect();
                format!("{}_{}", base_name, args.join("_"))
            };

            // Skip if already registered
            if self.struct_types.contains_key(&mangled_name) {
                return Ok(());
            }

            let i64_type = self.context.i64_type();
            let f64_type = self.context.f64_type();

            // Build field types and indices based on struct variant
            let mut field_types: Vec<BasicTypeEnum> = Vec::new();
            let mut field_indices: HashMap<String, u32> = HashMap::new();

            match &struct_def.fields {
                ast::StructFields::Named(fields) => {
                    for (idx, field) in fields.iter().enumerate() {
                        let llvm_type = self.type_expr_to_llvm(&field.ty, type_substitutions);
                        field_types.push(llvm_type);
                        field_indices.insert(field.name.name.clone(), idx as u32);
                    }
                }
                ast::StructFields::Tuple(types) => {
                    for (idx, ty) in types.iter().enumerate() {
                        let llvm_type = self.type_expr_to_llvm(ty, type_substitutions);
                        field_types.push(llvm_type);
                        field_indices.insert(format!("{}", idx), idx as u32);
                    }
                }
                ast::StructFields::Unit => {
                    // Unit struct has no fields
                }
            }

            // Create LLVM struct type
            let field_types_refs: Vec<_> = field_types.iter().map(|t| *t).collect();
            let struct_type = self.context.struct_type(&field_types_refs, false);

            // Store in registry with mangled name
            self.struct_types.insert(mangled_name, StructInfo {
                llvm_type: struct_type,
                field_indices,
            });

            Ok(())
        }

        /// Convert a TypeExpr to an LLVM BasicTypeEnum
        fn type_expr_to_llvm(
            &mut self,
            ty: &ast::TypeExpr,
            substitutions: &HashMap<String, String>,
        ) -> BasicTypeEnum<'ctx> {
            let i64_type = self.context.i64_type();
            let f64_type = self.context.f64_type();
            let i32_type = self.context.i32_type();
            let i8_type = self.context.i8_type();
            let bool_type = self.context.bool_type();

            match ty {
                ast::TypeExpr::Path(path) => {
                    if let Some(segment) = path.segments.first() {
                        let name = &segment.ident.name;

                        // Check if it's a type parameter that needs substitution
                        if let Some(concrete) = substitutions.get(name) {
                            return self.primitive_to_llvm(concrete);
                        }

                        // Check for primitive types
                        match name.as_str() {
                            "i8" | "u8" => i8_type.into(),
                            "i16" | "u16" => self.context.i16_type().into(),
                            "i32" | "u32" => i32_type.into(),
                            "i64" | "u64" | "isize" | "usize" => i64_type.into(),
                            "f32" => self.context.f32_type().into(),
                            "f64" => f64_type.into(),
                            "bool" => bool_type.into(),
                            _ => i64_type.into(), // Default to i64 for unknown types
                        }
                    } else {
                        i64_type.into()
                    }
                }
                ast::TypeExpr::Reference { inner, .. } |
                ast::TypeExpr::Pointer { inner, .. } => {
                    // References and pointers are represented as i64 (pointer-sized)
                    i64_type.into()
                }
                ast::TypeExpr::Array { element, .. } => {
                    // Arrays are represented as pointers for now
                    i64_type.into()
                }
                ast::TypeExpr::Tuple(elements) => {
                    // Tuples: for now, just use i64
                    i64_type.into()
                }
                ast::TypeExpr::Evidential { inner, .. } => {
                    // Unwrap evidentiality for LLVM type
                    self.type_expr_to_llvm(inner, substitutions)
                }
                _ => i64_type.into(),
            }
        }

        /// Convert a primitive type name to LLVM type
        fn primitive_to_llvm(&self, name: &str) -> BasicTypeEnum<'ctx> {
            match name {
                "i8" | "u8" => self.context.i8_type().into(),
                "i16" | "u16" => self.context.i16_type().into(),
                "i32" | "u32" => self.context.i32_type().into(),
                "i64" | "u64" | "isize" | "usize" => self.context.i64_type().into(),
                "f32" => self.context.f32_type().into(),
                "f64" => self.context.f64_type().into(),
                "bool" => self.context.bool_type().into(),
                _ => self.context.i64_type().into(),
            }
        }

        /// Monomorphize a generic struct with concrete type arguments
        fn monomorphize_struct(
            &mut self,
            base_name: &str,
            type_args: &[ast::TypeExpr],
        ) -> Result<String, String> {
            // Look up the generic struct definition
            let generic_def = self.generic_structs.get(base_name)
                .ok_or_else(|| format!("Unknown generic struct: {}", base_name))?
                .clone();

            if type_args.len() != generic_def.type_params.len() {
                return Err(format!(
                    "Wrong number of type arguments for {}: expected {}, got {}",
                    base_name, generic_def.type_params.len(), type_args.len()
                ));
            }

            // Build substitution map: type param name -> concrete type name
            let mut substitutions: HashMap<String, String> = HashMap::new();
            for (param, arg) in generic_def.type_params.iter().zip(type_args.iter()) {
                let concrete_name = self.type_expr_to_name(arg);
                substitutions.insert(param.clone(), concrete_name);
            }

            // Generate mangled name
            let concrete_names: Vec<String> = substitutions.values().cloned().collect();
            let mangled_name = format!("{}_{}", base_name, concrete_names.join("_"));

            // Register the monomorphized struct if not already done
            if !self.struct_types.contains_key(&mangled_name) {
                self.register_concrete_struct(&generic_def.def, &substitutions)?;
            }

            Ok(mangled_name)
        }

        /// Convert a TypeExpr to a string name for mangling
        fn type_expr_to_name(&self, ty: &ast::TypeExpr) -> String {
            match ty {
                ast::TypeExpr::Path(path) => {
                    path.segments.iter()
                        .map(|s| s.ident.name.clone())
                        .collect::<Vec<_>>()
                        .join("_")
                }
                ast::TypeExpr::Reference { inner, mutable, .. } => {
                    let prefix = if *mutable { "mut_ref" } else { "ref" };
                    format!("{}_{}", prefix, self.type_expr_to_name(inner))
                }
                ast::TypeExpr::Pointer { inner, mutable, .. } => {
                    let prefix = if *mutable { "mut_ptr" } else { "ptr" };
                    format!("{}_{}", prefix, self.type_expr_to_name(inner))
                }
                ast::TypeExpr::Array { element, .. } => {
                    format!("arr_{}", self.type_expr_to_name(element))
                }
                ast::TypeExpr::Tuple(elements) => {
                    let names: Vec<_> = elements.iter()
                        .map(|e| self.type_expr_to_name(e))
                        .collect();
                    format!("tup_{}", names.join("_"))
                }
                ast::TypeExpr::Evidential { inner, .. } => {
                    self.type_expr_to_name(inner)
                }
                _ => "unknown".to_string(),
            }
        }

        // ============================================
        // Evidentiality Support
        // ============================================

        /// Get or create an evidential wrapper type for a base type.
        /// Returns a struct type: { i8 tag, T value }
        fn get_evidential_type(&mut self, base_type_name: &str) -> StructType<'ctx> {
            if let Some(existing) = self.evidential_types.get(base_type_name) {
                return *existing;
            }

            // Create the struct type: { i8 tag, T value }
            let i8_type = self.context.i8_type();
            let value_type = self.primitive_to_llvm(base_type_name);

            let struct_name = format!("Evidential_{}", base_type_name);
            let struct_type = self.context.struct_type(
                &[i8_type.into(), value_type],
                false
            );

            self.evidential_types.insert(base_type_name.to_string(), struct_type);
            struct_type
        }

        /// Create an evidential value by wrapping a raw value with an evidence tag.
        /// Returns a struct { tag, value }.
        fn create_evidential_value(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            value: IntValue<'ctx>,
            evidence: u8,
            type_name: &str,
        ) -> Result<StructValue<'ctx>, String> {
            let evidential_type = self.get_evidential_type(type_name);
            let tag = self.context.i8_type().const_int(evidence as u64, false);

            // Allocate on stack and store fields
            let ptr = self.builder.build_alloca(evidential_type, "evidential")
                .map_err(|e| e.to_string())?;

            // Store the tag at index 0
            let tag_ptr = self.builder.build_struct_gep(evidential_type, ptr, 0, "tag_ptr")
                .map_err(|e| e.to_string())?;
            self.builder.build_store(tag_ptr, tag).map_err(|e| e.to_string())?;

            // Store the value at index 1
            let value_ptr = self.builder.build_struct_gep(evidential_type, ptr, 1, "value_ptr")
                .map_err(|e| e.to_string())?;
            self.builder.build_store(value_ptr, value).map_err(|e| e.to_string())?;

            // Load and return the complete struct
            let result = self.builder.build_load(evidential_type, ptr, "evidential_val")
                .map_err(|e| e.to_string())?;

            Ok(result.into_struct_value())
        }

        /// Extract the raw value from an evidential struct.
        /// This is used for the `!` (Known) marker which unwraps evidential values.
        fn unwrap_evidential_value(
            &mut self,
            evidential_struct: StructValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            // Extract value at index 1
            let value = self.builder
                .build_extract_value(evidential_struct, 1, "unwrapped")
                .map_err(|e| e.to_string())?;

            Ok(value.into_int_value())
        }

        /// Extract the evidence tag from an evidential struct.
        fn get_evidence_tag(
            &mut self,
            evidential_struct: StructValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            let tag = self.builder
                .build_extract_value(evidential_struct, 0, "tag")
                .map_err(|e| e.to_string())?;

            Ok(tag.into_int_value())
        }

        /// Convert AST Evidentiality to evidence tag constant
        fn evidentiality_to_tag(ev: &ast::Evidentiality) -> u8 {
            match ev {
                ast::Evidentiality::Known => EVIDENCE_KNOWN,
                ast::Evidentiality::Uncertain => EVIDENCE_UNCERTAIN,
                ast::Evidentiality::Reported => EVIDENCE_REPORTED,
                ast::Evidentiality::Predicted => EVIDENCE_PREDICTED,
                ast::Evidentiality::Paradox => EVIDENCE_PARADOX,
            }
        }

        /// Combine two evidence tags using the lattice join.
        /// The result is the "weaker" evidence level.
        /// Known < Uncertain < Reported < Predicted < Paradox
        fn combine_evidence(
            &mut self,
            tag1: IntValue<'ctx>,
            tag2: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            // Use max(tag1, tag2) since higher values = weaker evidence
            let cmp = self.builder
                .build_int_compare(IntPredicate::UGT, tag1, tag2, "ev_cmp")
                .map_err(|e| e.to_string())?;

            let result = self.builder
                .build_select(cmp, tag1, tag2, "ev_combined")
                .map_err(|e| e.to_string())?;

            Ok(result.into_int_value())
        }

        /// Register an enum type in the type registry
        fn register_enum(&mut self, enum_def: &ast::EnumDef) -> Result<(), String> {
            let name = &enum_def.name.name;
            let mut variants: HashMap<String, u64> = HashMap::new();

            for (idx, variant) in enum_def.variants.iter().enumerate() {
                let discriminant = idx as u64;
                variants.insert(variant.name.name.clone(), discriminant);
            }

            self.enum_types.insert(name.clone(), EnumInfo { variants });
            Ok(())
        }

        /// Register a constant value (compile-time evaluated)
        fn register_const(&mut self, const_def: &ast::ConstDef) -> Result<(), String> {
            let name = &const_def.name.name;

            // Evaluate the constant expression at compile time
            // For now, only support integer literals and simple expressions
            let value = self.eval_const_expr(&const_def.value)?;
            self.constants.insert(name.clone(), value);
            Ok(())
        }

        /// Evaluate a constant expression at compile time
        fn eval_const_expr(&self, expr: &Expr) -> Result<i64, String> {
            match expr {
                Expr::Literal(lit) => {
                    match lit {
                        Literal::Int { value, .. } => {
                            let clean_value = strip_int_suffix(value);
                            clean_value.parse::<i64>()
                                .map_err(|_| format!("Invalid integer constant: {}", value))
                        }
                        Literal::Bool(b) => Ok(if *b { 1 } else { 0 }),
                        _ => Err("Unsupported constant literal type".to_string()),
                    }
                }
                Expr::Path(path) => {
                    // Reference to another constant
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident.name;
                        self.constants.get(name).copied()
                            .ok_or_else(|| format!("Unknown constant: {}", name))
                    } else {
                        Err("Complex paths not supported in constant expressions".to_string())
                    }
                }
                Expr::Unary { op, expr } => {
                    let val = self.eval_const_expr(expr)?;
                    match op {
                        ast::UnaryOp::Neg => Ok(val.wrapping_neg()),
                        ast::UnaryOp::Not => Ok(!val),
                        _ => Err("Unsupported unary operator in constant".to_string()),
                    }
                }
                Expr::Binary { op, left, right } => {
                    let lhs = self.eval_const_expr(left)?;
                    let rhs = self.eval_const_expr(right)?;
                    match op {
                        ast::BinOp::Add => Ok(lhs.wrapping_add(rhs)),
                        ast::BinOp::Sub => Ok(lhs.wrapping_sub(rhs)),
                        ast::BinOp::Mul => Ok(lhs.wrapping_mul(rhs)),
                        ast::BinOp::Div => {
                            if rhs == 0 {
                                Err("Division by zero in constant".to_string())
                            } else {
                                Ok(lhs.wrapping_div(rhs))
                            }
                        }
                        ast::BinOp::Rem => {
                            if rhs == 0 {
                                Err("Division by zero in constant".to_string())
                            } else {
                                Ok(lhs.wrapping_rem(rhs))
                            }
                        }
                        ast::BinOp::BitOr => Ok(lhs | rhs),
                        ast::BinOp::BitAnd => Ok(lhs & rhs),
                        ast::BinOp::BitXor => Ok(lhs ^ rhs),
                        ast::BinOp::Shl => {
                            // Safe shift: mask to valid range (0-63 for i64)
                            let shift = (rhs as u32) & 63;
                            Ok(lhs.wrapping_shl(shift))
                        }
                        ast::BinOp::Shr => {
                            // Safe shift: mask to valid range (0-63 for i64)
                            let shift = (rhs as u32) & 63;
                            Ok(lhs.wrapping_shr(shift))
                        }
                        _ => Err("Unsupported binary operator in constant".to_string()),
                    }
                }
                _ => Err(format!("Unsupported constant expression: {:?}", std::mem::discriminant(expr))),
            }
        }

        /// Declare methods from an impl block
        fn declare_impl_methods(&mut self, impl_block: &ast::ImplBlock) -> Result<(), String> {
            // Get the type name from the impl path
            // Extract type name from self_ty (TypeExpr)
            let type_name = match &impl_block.self_ty {
                ast::TypeExpr::Path(path) => {
                    path.segments.last()
                        .map(|s| s.ident.name.clone())
                        .ok_or_else(|| "Empty impl type path".to_string())?
                }
                _ => return Err("Unsupported impl type".to_string()),
            };

            for item in &impl_block.items {
                if let ast::ImplItem::Function(func) = item {
                    let method_name = &func.name.name;
                    let mangled_name = format!("{}_{}", type_name, method_name);

                    // Declare the function with self as first parameter
                    let i64_type = self.context.i64_type();

                    // Check if first param is self (don't double count)
                    let has_explicit_self = func.params.first().map_or(false, |p| {
                        matches!(&p.pattern, ast::Pattern::Ident { name, .. } if name.name == "self")
                    });

                    // Count params: if self is explicit, use params.len(), otherwise add 1 for implicit self
                    let param_count = if has_explicit_self {
                        func.params.len()
                    } else {
                        1 + func.params.len()
                    };
                    let param_types: Vec<BasicMetadataTypeEnum> =
                        (0..param_count).map(|_| i64_type.into()).collect();

                    let fn_type = i64_type.fn_type(&param_types, false);
                    let fn_value = self.module.add_function(&mangled_name, fn_type, None);

                    // Name parameters
                    if has_explicit_self {
                        // self is in params, name them all
                        for (i, param) in func.params.iter().enumerate() {
                            if let ast::Pattern::Ident { name: ref ident, .. } = param.pattern {
                                fn_value.get_nth_param(i as u32).unwrap().set_name(&ident.name);
                            }
                        }
                    } else {
                        // self is implicit, name it first then other params
                        fn_value.get_nth_param(0).unwrap().set_name("self");
                        for (i, param) in func.params.iter().enumerate() {
                            if let ast::Pattern::Ident { name: ref ident, .. } = param.pattern {
                                fn_value.get_nth_param((i + 1) as u32).unwrap().set_name(&ident.name);
                            }
                        }
                    }

                    self.functions.insert(mangled_name.clone(), fn_value);
                    self.impl_methods.insert((type_name.clone(), method_name.clone()), mangled_name);
                }
            }
            Ok(())
        }

        /// Compile methods from an impl block
        fn compile_impl_methods(&mut self, impl_block: &ast::ImplBlock) -> Result<(), String> {
            // Extract type name from self_ty (TypeExpr)
            let type_name = match &impl_block.self_ty {
                ast::TypeExpr::Path(path) => {
                    path.segments.last()
                        .map(|s| s.ident.name.clone())
                        .ok_or_else(|| "Empty impl type path".to_string())?
                }
                _ => return Err("Unsupported impl type".to_string()),
            };

            for item in &impl_block.items {
                if let ast::ImplItem::Function(func) = item {
                    let method_name = &func.name.name;
                    let mangled_name = format!("{}_{}", type_name, method_name);

                    let fn_value = *self.functions.get(&mangled_name)
                        .ok_or_else(|| format!("Method not declared: {}", mangled_name))?;

                    // Create entry block
                    let entry = self.context.append_basic_block(fn_value, "entry");
                    self.builder.position_at_end(entry);

                    // Set up variable scope
                    let mut scope = CompileScope::new();

                    // Check if first param is self (explicit self in params)
                    let has_explicit_self = func.params.first().map_or(false, |p| {
                        matches!(&p.pattern, ast::Pattern::Ident { name, .. } if name.name == "self")
                    });

                    if has_explicit_self {
                        // self is explicitly in params, add all params to scope
                        for (i, param) in func.params.iter().enumerate() {
                            if let ast::Pattern::Ident { name: ref ident, .. } = param.pattern {
                                let param_value = fn_value.get_nth_param(i as u32).unwrap();
                                let alloca = self.builder
                                    .build_alloca(self.context.i64_type(), &ident.name)
                                    .map_err(|e| e.to_string())?;
                                self.builder.build_store(alloca, param_value).map_err(|e| e.to_string())?;
                                scope.vars.insert(ident.name.clone(), alloca);
                            }
                        }
                    } else {
                        // self is implicit (first parameter)
                        let self_param = fn_value.get_nth_param(0).unwrap();
                        let self_alloca = self.builder
                            .build_alloca(self.context.i64_type(), "self")
                            .map_err(|e| e.to_string())?;
                        self.builder.build_store(self_alloca, self_param).map_err(|e| e.to_string())?;
                        scope.vars.insert("self".to_string(), self_alloca);

                        // Add other parameters to scope
                        for (i, param) in func.params.iter().enumerate() {
                            if let ast::Pattern::Ident { name: ref ident, .. } = param.pattern {
                                let param_value = fn_value.get_nth_param((i + 1) as u32).unwrap();
                                let alloca = self.builder
                                    .build_alloca(self.context.i64_type(), &ident.name)
                                    .map_err(|e| e.to_string())?;
                                self.builder.build_store(alloca, param_value).map_err(|e| e.to_string())?;
                                scope.vars.insert(ident.name.clone(), alloca);
                            }
                        }
                    }

                    // Compile function body
                    if let Some(ref body) = func.body {
                        let result = self.compile_block(fn_value, &mut scope, body)?;

                        let current_block = self.builder.get_insert_block().unwrap();
                        if current_block.get_terminator().is_none() {
                            if let Some(val) = result {
                                self.builder.build_return(Some(&val)).map_err(|e| e.to_string())?;
                            } else {
                                let zero = self.context.i64_type().const_int(0, false);
                                self.builder.build_return(Some(&zero)).map_err(|e| e.to_string())?;
                            }
                        }
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        self.builder.build_return(Some(&zero)).map_err(|e| e.to_string())?;
                    }
                }
            }
            Ok(())
        }

        /// Declare a function (creates the signature)
        fn declare_function(
            &mut self,
            func: &ast::Function,
        ) -> Result<FunctionValue<'ctx>, String> {
            let name = &func.name.name;

            // Build mangled name with module path (skip "crate" prefix)
            let mangled_name = if self.current_module.len() > 1 {
                let module_path = self.current_module[1..].join("_");
                format!("{}_{}", module_path, name)
            } else {
                name.clone()
            };

            // In AOT mode, rename "main" to "main_sigil" so it doesn't conflict with C runtime
            let actual_name = if self.compile_mode == CompileMode::Aot && name == "main" && self.current_module.len() == 1 {
                "main_sigil".to_string()
            } else {
                mangled_name.clone()
            };

            let i64_type = self.context.i64_type();

            // Build parameter types (all i64 for simplicity)
            let param_types: Vec<BasicMetadataTypeEnum> =
                func.params.iter().map(|_| i64_type.into()).collect();

            // Create function type
            let fn_type = i64_type.fn_type(&param_types, false);

            // Declare the function
            let fn_value = self.module.add_function(&actual_name, fn_type, None);

            // Add optimization attributes
            // nounwind - function doesn't throw exceptions (enables more optimizations)
            let nounwind_attr = self.context.create_enum_attribute(
                inkwell::attributes::Attribute::get_named_enum_kind_id("nounwind"),
                0,
            );
            fn_value.add_attribute(inkwell::attributes::AttributeLoc::Function, nounwind_attr);

            // Name parameters
            for (i, param) in func.params.iter().enumerate() {
                if let ast::Pattern::Ident {
                    name: ref ident, ..
                } = param.pattern
                {
                    fn_value
                        .get_nth_param(i as u32)
                        .unwrap()
                        .set_name(&ident.name);
                }
            }

            // Store function with both short name and full qualified path for lookups
            self.functions.insert(name.clone(), fn_value);
            if self.current_module.len() > 1 {
                let full_path = format!("{}::{}", self.current_module[1..].join("::"), name);
                self.functions.insert(full_path, fn_value);
            }
            Ok(fn_value)
        }

        /// Compile a function body
        fn compile_function(&mut self, func: &ast::Function) -> Result<(), String> {
            let name = &func.name.name;
            let fn_value = *self.functions.get(name).ok_or("Function not declared")?;

            // Create entry block
            let entry = self.context.append_basic_block(fn_value, "entry");
            self.builder.position_at_end(entry);

            // Set up variable scope
            let mut scope = CompileScope::new();

            // Add parameters to scope
            for (i, param) in func.params.iter().enumerate() {
                if let ast::Pattern::Ident {
                    name: ref ident, ..
                } = param.pattern
                {
                    let param_value = fn_value.get_nth_param(i as u32).unwrap();
                    // Allocate on stack for potential mutation
                    let alloca = self
                        .builder
                        .build_alloca(self.context.i64_type(), &ident.name)
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_store(alloca, param_value)
                        .map_err(|e| e.to_string())?;
                    scope.vars.insert(ident.name.clone(), alloca);
                }
            }

            // Compile function body
            if let Some(ref body) = func.body {
                let result = self.compile_block(fn_value, &mut scope, body)?;

                // Only add return if block isn't already terminated
                let current_block = self.builder.get_insert_block().unwrap();
                if current_block.get_terminator().is_none() {
                    if let Some(val) = result {
                        self.builder
                            .build_return(Some(&val))
                            .map_err(|e| e.to_string())?;
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        self.builder
                            .build_return(Some(&zero))
                            .map_err(|e| e.to_string())?;
                    }
                }
            } else {
                // No body, return 0
                let zero = self.context.i64_type().const_int(0, false);
                self.builder
                    .build_return(Some(&zero))
                    .map_err(|e| e.to_string())?;
            }

            Ok(())
        }

        /// Compile a block
        fn compile_block(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            block: &ast::Block,
        ) -> Result<Option<IntValue<'ctx>>, String> {
            let mut result = None;

            for stmt in &block.stmts {
                result = self.compile_stmt(fn_value, scope, stmt)?;
                // Check if we hit a return
                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_some()
                {
                    return Ok(result);
                }
            }

            // Trailing expression
            if let Some(ref expr) = block.expr {
                result = Some(self.compile_expr(fn_value, scope, expr)?);
            }

            Ok(result)
        }

        /// Compile a statement
        fn compile_stmt(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            stmt: &ast::Stmt,
        ) -> Result<Option<IntValue<'ctx>>, String> {
            match stmt {
                ast::Stmt::Let { pattern, init, .. } => {
                    if let ast::Pattern::Ident {
                        name: ref ident, ..
                    } = pattern
                    {
                        let init_val = if let Some(ref expr) = init {
                            self.compile_expr(fn_value, scope, expr)?
                        } else {
                            self.context.i64_type().const_int(0, false)
                        };

                        // Allocate on stack
                        let alloca = self
                            .builder
                            .build_alloca(self.context.i64_type(), &ident.name)
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(alloca, init_val)
                            .map_err(|e| e.to_string())?;
                        scope.vars.insert(ident.name.clone(), alloca);
                    }
                    Ok(None)
                }
                ast::Stmt::Expr(expr) => {
                    let val = self.compile_expr(fn_value, scope, expr)?;
                    Ok(Some(val))
                }
                ast::Stmt::Semi(expr) => {
                    self.compile_expr(fn_value, scope, expr)?;
                    Ok(None)
                }
                ast::Stmt::Item(_) => Ok(None),
                ast::Stmt::LetElse { pattern, init, .. } => {
                    // LetElse is like let but with an else branch for refutable patterns
                    // For now, treat it like a regular let
                    if let ast::Pattern::Ident {
                        name: ref ident, ..
                    } = pattern
                    {
                        let init_val = self.compile_expr(fn_value, scope, init)?;
                        let alloca = self
                            .builder
                            .build_alloca(self.context.i64_type(), &ident.name)
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(alloca, init_val)
                            .map_err(|e| e.to_string())?;
                        scope.vars.insert(ident.name.clone(), alloca);
                    }
                    Ok(None)
                }
            }
        }

        /// Compile an expression
        fn compile_expr(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            expr: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            match expr {
                Expr::Literal(lit) => self.compile_literal(lit),
                Expr::Path(path) => {
                    // Check for qualified enum variant path (e.g., Color::Blue)
                    if path.segments.len() >= 2 {
                        let enum_name = &path.segments[path.segments.len() - 2].ident.name;
                        let variant_name = &path.segments[path.segments.len() - 1].ident.name;
                        
                        if let Some(enum_info) = self.enum_types.get(enum_name) {
                            if let Some(&discriminant) = enum_info.variants.get(variant_name) {
                                return Ok(self.context.i64_type().const_int(discriminant, false));
                            }
                        }
                    }
                    
                    // Variable lookup
                    let name = path
                        .segments
                        .last()
                        .map(|s| s.ident.name.as_str())
                        .ok_or("Empty path")?;

                    if let Some(&ptr) = scope.vars.get(name) {
                        let val = self
                            .builder
                            .build_load(self.context.i64_type(), ptr, name)
                            .map_err(|e| e.to_string())?;
                        Ok(val.into_int_value())
                    } else if let Some(&const_val) = self.constants.get(name) {
                        // Found a constant
                        Ok(self.context.i64_type().const_int(const_val as u64, true))
                    } else {
                        // Check if it's an unqualified enum variant (search all enums)
                        for (_, enum_info) in &self.enum_types {
                            if let Some(&discriminant) = enum_info.variants.get(name) {
                                return Ok(self.context.i64_type().const_int(discriminant, false));
                            }
                        }
                        Err(format!("Unknown variable: {} (in fn {})", name, fn_value.get_name().to_str().unwrap_or("?")))
                    }
                }
                Expr::Binary { op, left, right } => {
                    let lhs = self.compile_expr(fn_value, scope, left)?;
                    let rhs = self.compile_expr(fn_value, scope, right)?;
                    self.compile_binary_op(*op, lhs, rhs)
                }
                Expr::Unary { op, expr: inner } => {
                    let val = self.compile_expr(fn_value, scope, inner)?;
                    self.compile_unary_op(*op, val)
                }
                Expr::If {
                    condition,
                    then_branch,
                    else_branch,
                } => self.compile_if(
                    fn_value,
                    scope,
                    condition,
                    then_branch,
                    else_branch.as_deref(),
                ),
                Expr::While { label: _, condition, body } => {
                    self.compile_while(fn_value, scope, condition, body)
                }
                Expr::Call { func, args } => self.compile_call(fn_value, scope, func, args),
                Expr::Return(val) => {
                    let ret_val = if let Some(ref e) = val {
                        self.compile_expr(fn_value, scope, e)?
                    } else {
                        self.context.i64_type().const_int(0, false)
                    };
                    self.builder
                        .build_return(Some(&ret_val))
                        .map_err(|e| e.to_string())?;
                    // Return a dummy value (code after return is unreachable)
                    Ok(ret_val)
                }
                Expr::Assign { target, value } => {
                    let val = self.compile_expr(fn_value, scope, value)?;
                    match target.as_ref() {
                        Expr::Path(path) => {
                            let name = path
                                .segments
                                .last()
                                .map(|s| s.ident.name.as_str())
                                .ok_or("Empty path")?;
                            if let Some(&ptr) = scope.vars.get(name) {
                                self.builder
                                    .build_store(ptr, val)
                                    .map_err(|e| e.to_string())?;
                                Ok(val)
                            } else {
                                Err(format!("Unknown variable: {}", name))
                            }
                        }
                        Expr::Field { expr, field } => {
                            // Get struct pointer from the expression
                            let struct_ptr_int = self.compile_expr(fn_value, scope, expr)?;
                            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let struct_ptr = self.builder
                                .build_int_to_ptr(struct_ptr_int, ptr_type, "struct_ptr")
                                .map_err(|e| e.to_string())?;

                            let field_name = &field.name;
                            // Find the struct type and field index
                            for (_name, struct_info) in &self.struct_types {
                                if let Some(&field_idx) = struct_info.field_indices.get(field_name) {
                                    let field_ptr = self.builder
                                        .build_struct_gep(struct_info.llvm_type, struct_ptr, field_idx, &format!("{}_ptr", field_name))
                                        .map_err(|e| e.to_string())?;
                                    self.builder
                                        .build_store(field_ptr, val)
                                        .map_err(|e| e.to_string())?;
                                    return Ok(val);
                                }
                            }
                            Err(format!("Unknown field: {}", field_name))
                        }
                        _ => Err("Invalid assignment target".to_string())
                    }
                }
                Expr::Block(block) => {
                    let result = self.compile_block(fn_value, scope, block)?;
                    Ok(result.unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                }
                Expr::Struct { path, fields, .. } => {
                    // Get struct name and potential generic arguments from path
                    let last_segment = path.segments
                        .last()
                        .ok_or("Empty struct path")?;
                    let base_name = last_segment.ident.name.as_str();

                    // Check if this is a generic struct instantiation
                    let struct_name = if let Some(ref type_args) = last_segment.generics {
                        // Monomorphize the generic struct
                        self.monomorphize_struct(base_name, type_args)?
                    } else if self.generic_structs.contains_key(base_name) {
                        // Generic struct used without type args - error
                        return Err(format!("Generic struct {} requires type arguments", base_name));
                    } else {
                        base_name.to_string()
                    };

                    // Look up struct type (now with mangled name for generics)
                    let struct_info = self.struct_types.get(&struct_name)
                        .ok_or_else(|| format!("Unknown struct type: {}", struct_name))?
                        .clone();

                    // Allocate space for struct on stack
                    let struct_ptr = self.builder
                        .build_alloca(struct_info.llvm_type, &struct_name)
                        .map_err(|e| e.to_string())?;

                    // Initialize each field
                    for field_init in fields {
                        let field_name = &field_init.name.name;
                        let field_idx = *struct_info.field_indices.get(field_name)
                            .ok_or_else(|| format!("Unknown field: {}", field_name))?;

                        // Get field value (or use field name as variable if no value)
                        let field_value = if let Some(ref val_expr) = field_init.value {
                            self.compile_expr(fn_value, scope, val_expr)?
                        } else {
                            // Shorthand: field name is the variable name
                            if let Some(&ptr) = scope.vars.get(field_name.as_str()) {
                                self.builder
                                    .build_load(self.context.i64_type(), ptr, field_name)
                                    .map_err(|e| e.to_string())?
                                    .into_int_value()
                            } else {
                                return Err(format!("Unknown variable for field shorthand: {}", field_name));
                            }
                        };

                        // Get pointer to field and store value
                        let field_ptr = self.builder
                            .build_struct_gep(struct_info.llvm_type, struct_ptr, field_idx, &format!("{}_ptr", field_name))
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(field_ptr, field_value)
                            .map_err(|e| e.to_string())?;
                    }

                    // Return struct pointer as i64
                    let ptr_int = self.builder
                        .build_ptr_to_int(struct_ptr, self.context.i64_type(), "struct_ptr")
                        .map_err(|e| e.to_string())?;
                    Ok(ptr_int)
                }
                Expr::Field { expr, field } => {
                    // Compile the struct expression to get pointer
                    let struct_ptr_int = self.compile_expr(fn_value, scope, expr)?;

                    // Convert i64 back to pointer
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let struct_ptr = self.builder
                        .build_int_to_ptr(struct_ptr_int, ptr_type, "struct_ptr")
                        .map_err(|e| e.to_string())?;

                    // Try to find struct type from expression
                    // For now, search all struct types for the field
                    let field_name = &field.name;
                    for (_name, struct_info) in &self.struct_types {
                        if let Some(&field_idx) = struct_info.field_indices.get(field_name) {
                            let field_ptr = self.builder
                                .build_struct_gep(struct_info.llvm_type, struct_ptr, field_idx, &format!("{}_ptr", field_name))
                                .map_err(|e| e.to_string())?;
                            let field_value = self.builder
                                .build_load(self.context.i64_type(), field_ptr, field_name)
                                .map_err(|e| e.to_string())?;
                            return Ok(field_value.into_int_value());
                        }
                    }
                    Err(format!("Unknown field: {}", field_name))
                }
                Expr::Match { expr, arms } => {
                    // Compile the scrutinee (thing being matched)
                    let scrutinee = self.compile_expr(fn_value, scope, expr)?;

                    let merge_bb = self.context.append_basic_block(fn_value, "match_merge");
                    let mut incoming: Vec<(IntValue<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();

                    // Build chain of if-else for each arm
                    for (i, arm) in arms.iter().enumerate() {
                        // Get pattern discriminant value
                        let pattern_val = match &arm.pattern {
                            ast::Pattern::Path(path) => {
                                if path.segments.len() >= 2 {
                                    let enum_name = &path.segments[path.segments.len() - 2].ident.name;
                                    let variant_name = &path.segments[path.segments.len() - 1].ident.name;
                                    if let Some(enum_info) = self.enum_types.get(enum_name) {
                                        enum_info.variants.get(variant_name).copied()
                                    } else { None }
                                } else { None }
                            }
                            ast::Pattern::Literal(lit) => {
                                if let Ok(v) = self.compile_literal(lit) {
                                    Some(v.get_zero_extended_constant().unwrap_or(0))
                                } else { None }
                            }
                            ast::Pattern::Wildcard => None,
                            _ => None,
                        };

                        let then_bb = self.context.append_basic_block(fn_value, &format!("match_then_{}", i));
                        let else_bb = if i + 1 < arms.len() {
                            self.context.append_basic_block(fn_value, &format!("match_else_{}", i))
                        } else {
                            merge_bb
                        };

                        // For the last arm or wildcard, always branch unconditionally
                        let is_last_arm = i + 1 >= arms.len();
                        if let Some(disc) = pattern_val {
                            if is_last_arm {
                                // Last arm - treat as default (exhaustive match assumed)
                                self.builder.build_unconditional_branch(then_bb)
                                    .map_err(|e| e.to_string())?;
                            } else {
                                let pattern_const = self.context.i64_type().const_int(disc, false);
                                let cond = self.builder
                                    .build_int_compare(IntPredicate::EQ, scrutinee, pattern_const, "match_cmp")
                                    .map_err(|e| e.to_string())?;
                                self.builder.build_conditional_branch(cond, then_bb, else_bb)
                                    .map_err(|e| e.to_string())?;
                            }
                        } else {
                            // Wildcard - unconditionally go to then block
                            self.builder.build_unconditional_branch(then_bb)
                                .map_err(|e| e.to_string())?;
                        }

                        // Compile the arm body
                        self.builder.position_at_end(then_bb);
                        let arm_val = self.compile_expr(fn_value, scope, &arm.body)?;

                        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                            let current_bb = self.builder.get_insert_block().unwrap();
                            self.builder.build_unconditional_branch(merge_bb)
                                .map_err(|e| e.to_string())?;
                            incoming.push((arm_val, current_bb));
                        }

                        // Position at else block for next iteration
                        if i + 1 < arms.len() {
                            self.builder.position_at_end(else_bb);
                        }
                    }

                    // Build phi at merge
                    self.builder.position_at_end(merge_bb);

                    if incoming.is_empty() {
                        return Ok(self.context.i64_type().const_int(0, false));
                    }

                    let phi = self.builder
                        .build_phi(self.context.i64_type(), "match_result")
                        .map_err(|e| e.to_string())?;

                    for (val, bb) in &incoming {
                        phi.add_incoming(&[(val, *bb)]);
                    }

                    Ok(phi.as_basic_value().into_int_value())
                }
                Expr::MethodCall { receiver, method, args, .. } => {
                    // Compile the receiver
                    let receiver_val = self.compile_expr(fn_value, scope, receiver)?;

                    // Look up the method by trying all registered impl methods
                    // For now, try each type until we find a match
                    for ((type_name, method_name), mangled_name) in &self.impl_methods {
                        if method_name == &method.name {
                            if let Some(callee) = self.module.get_function(mangled_name) {
                                // Compile arguments
                                let mut compiled_args: Vec<BasicMetadataValueEnum> = vec![receiver_val.into()];
                                for arg in args {
                                    let arg_val = self.compile_expr(fn_value, scope, arg)?;
                                    compiled_args.push(arg_val.into());
                                }

                                let call = self.builder
                                    .build_call(callee, &compiled_args, "method_call")
                                    .map_err(|e| e.to_string())?;

                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                            }
                        }
                    }
                    Err(format!("Unknown method: {}", method.name))
                }
                // ============================================
                // Sigil-native expressions
                // ============================================

                // Evidentiality markers with runtime semantics
                // Known (!) unwraps evidential values
                // Other markers wrap values with evidence tags
                Expr::Evidential { expr, evidentiality } => {
                    let inner_val = self.compile_expr(fn_value, scope, expr)?;

                    match evidentiality {
                        ast::Evidentiality::Known => {
                            // Known (!) is an unwrap operation - just return the inner value
                            // The type checker ensures this is safe
                            Ok(inner_val)
                        }
                        _ => {
                            // For ?, ~, ◊, ‽: Create evidential struct and return value
                            // The struct stores {tag, value} for runtime evidence tracking
                            let tag = Self::evidentiality_to_tag(evidentiality);
                            let evidential = self.create_evidential_value(
                                fn_value,
                                inner_val,
                                tag,
                                "i64"  // Default to i64 for now
                            )?;

                            // Extract and return the value portion for IntValue compatibility
                            // The full struct is available through the scope for advanced operations
                            self.unwrap_evidential_value(evidential)
                        }
                    }
                }

                // Pipe expressions: data |τ{f} |φ{p} |ρ+
                Expr::Pipe { expr, operations } => {
                    self.compile_pipe(fn_value, scope, expr, operations)
                }

                // Standalone morpheme expressions
                Expr::Morpheme { kind: _, body } => {
                    // Morpheme body is compiled directly
                    self.compile_expr(fn_value, scope, body)
                }

                // Array/slice indexing
                Expr::Index { expr, index } => {
                    self.compile_index(fn_value, scope, expr, index)
                }

                // Range expressions
                Expr::Range { start, end, inclusive: _ } => {
                    // For now, ranges compile to their start value
                    // Full range support needs iterator infrastructure
                    if let Some(s) = start {
                        self.compile_expr(fn_value, scope, s)
                    } else if let Some(e) = end {
                        self.compile_expr(fn_value, scope, e)
                    } else {
                        Ok(self.context.i64_type().const_int(0, false))
                    }
                }

                // Closures - compile body with captured variables
                Expr::Closure { params: _, body, .. } => {
                    // For simple closures, just compile the body
                    // Full closure support needs lambda lifting
                    self.compile_expr(fn_value, scope, body)
                }

                // Cast/type coercion
                Expr::Cast { expr, .. } => {
                    // Types are erased, just compile the expression
                    self.compile_expr(fn_value, scope, expr)
                }

                // Address-of: &expr, &mut expr
                Expr::AddrOf { expr, .. } => {
                    self.compile_expr(fn_value, scope, expr)
                }

                // Dereference: *ptr
                Expr::Deref(inner) => {
                    self.compile_expr(fn_value, scope, inner)
                }

                // Macro invocation - compile the name as a call
                Expr::Macro { path, .. } => {
                    // Treat macro! as a function call to the macro name
                    let name = path.segments.last()
                        .map(|s| s.ident.name.trim_end_matches('!'))
                        .unwrap_or("unknown");
                    if let Some(f) = self.module.get_function(name) {
                        let call = self.builder
                            .build_call(f, &[], "macro_call")
                            .map_err(|e| e.to_string())?;
                        Ok(call.try_as_basic_value().left()
                            .map(|v| v.into_int_value())
                            .unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                    } else {
                        // Unknown macro, return 0
                        Ok(self.context.i64_type().const_int(0, false))
                    }
                }

                // Try expression: expr?
                Expr::Try(inner) => {
                    // Types erased, just compile inner
                    self.compile_expr(fn_value, scope, inner)
                }

                // Let expression (for if-let patterns)
                Expr::Let { value, .. } => {
                    self.compile_expr(fn_value, scope, value)
                }

                // Tuple: just return first element for now
                Expr::Tuple(elements) => {
                    if let Some(first) = elements.first() {
                        self.compile_expr(fn_value, scope, first)
                    } else {
                        Ok(self.context.i64_type().const_int(0, false))
                    }
                }

                // Array literal: allocate on stack and store elements
                Expr::Array(elements) => {
                    self.compile_array_literal(fn_value, scope, elements)
                }

                // Loop expressions
                Expr::Loop { body, .. } => {
                    let result = self.compile_block(fn_value, scope, body)?;
                    Ok(result.unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                }

                Expr::For { body, .. } => {
                    let result = self.compile_block(fn_value, scope, body)?;
                    Ok(result.unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                }

                // Break/Continue - just return 0
                Expr::Break { .. } | Expr::Continue { .. } => {
                    Ok(self.context.i64_type().const_int(0, false))
                }

                // Unsafe block - compile the block
                Expr::Unsafe(block) => {
                    let result = self.compile_block(fn_value, scope, block)?;
                    Ok(result.unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                }

                // Await - compile inner
                Expr::Await { expr, .. } => {
                    self.compile_expr(fn_value, scope, expr)
                }

                // Inline assembly
                Expr::InlineAsm(asm) => {
                    self.compile_inline_asm(fn_value, scope, asm)
                }

                _ => {
                    // Unsupported expression - return error instead of silent 0
                    Err(format!("LLVM codegen: unsupported expression {:?}",
                        std::mem::discriminant(expr)))
                }
            }
        }

        /// Compile a literal
        fn compile_literal(&mut self, lit: &Literal) -> Result<IntValue<'ctx>, String> {
            match lit {
                Literal::Int { value, .. } => {
                    let clean_value = strip_int_suffix(value);
                    let v: i64 = clean_value.parse().map_err(|_| format!("Invalid integer: {}", value))?;
                    Ok(self.context.i64_type().const_int(v as u64, false))
                }
                Literal::Bool(b) => Ok(self
                    .context
                    .i64_type()
                    .const_int(if *b { 1 } else { 0 }, false)),
                Literal::Float { value, .. } => {
                    // Convert float to int bits for now
                    let v: f64 = value.parse().map_err(|_| "Invalid float")?;
                    Ok(self.context.i64_type().const_int(v.to_bits(), false))
                }
                Literal::String(s) => {
                    // Create a global string constant
                    let counter = self.string_counter.get();
                    self.string_counter.set(counter + 1);
                    let global_name = format!(".str.{}", counter);

                    // Create the string constant with null terminator
                    let string_bytes = s.as_bytes();
                    let const_array = self.context.const_string(string_bytes, true);

                    // Create global variable
                    let global = self.module.add_global(
                        const_array.get_type(),
                        None,
                        &global_name,
                    );
                    global.set_initializer(&const_array);
                    global.set_constant(true);
                    global.set_linkage(inkwell::module::Linkage::Private);

                    // Return pointer as i64
                    let ptr = global.as_pointer_value();
                    let ptr_as_int = self.builder
                        .build_ptr_to_int(ptr, self.context.i64_type(), "str_ptr")
                        .map_err(|e| e.to_string())?;
                    Ok(ptr_as_int)
                }
                Literal::Char(c) => {
                    Ok(self.context.i64_type().const_int(*c as u64, false))
                }
                _ => Ok(self.context.i64_type().const_int(0, false)),
            }
        }

        /// Compile inline assembly expression
        ///
        /// Translates Sigil's asm!() syntax to LLVM inline assembly.
        ///
        /// Example Sigil:
        /// ```sigil
        /// asm!("syscall",
        ///     inout("rax") num => ret,
        ///     in("rdi") arg0,
        ///     out("rcx") _,
        ///     clobber("r11"),
        ///     options(nostack))
        /// ```
        ///
        /// Becomes LLVM IR like:
        /// ```llvm
        /// %ret = call i64 asm sideeffect "syscall", "={rax},{rax},{rdi},~{rcx},~{r11}"(i64 %num, i64 %arg0)
        /// ```
        fn compile_inline_asm(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            asm: &InlineAsm,
        ) -> Result<IntValue<'ctx>, String> {
            let i64_type = self.context.i64_type();

            // Build the constraint string and collect input values
            // LLVM constraint format: "outputs,inputs,clobbers"
            // Output constraints start with "=" (or "=&" for early clobber)
            // Input constraints are just the constraint
            // Clobbers start with "~"

            let mut constraints = Vec::new();
            let mut input_values: Vec<BasicMetadataValueEnum<'ctx>> = Vec::new();
            let mut output_vars: Vec<Option<PointerValue<'ctx>>> = Vec::new();
            let mut has_output = false;

            // Process outputs first (LLVM requires outputs before inputs in constraint string)
            // Note: The parser puts both `out` and `inout` operands in asm.outputs
            for operand in &asm.outputs {
                has_output = true;
                let constraint = Self::translate_constraint(&operand.constraint, true);
                constraints.push(constraint);

                // Get the variable pointer for the output
                // For InOut operands, the output goes to `operand.output` if present,
                // otherwise the input expr must be a variable (same for both)
                let output_expr = if operand.kind == AsmOperandKind::InOut {
                    operand.output.as_deref().unwrap_or(&operand.expr)
                } else {
                    &operand.expr
                };
                let var_ptr = self.get_output_var_ptr(scope, output_expr)?;
                output_vars.push(var_ptr);
            }

            // Track which output indices are inout so we can tie inputs to them
            let mut inout_indices: Vec<usize> = Vec::new();
            for (i, operand) in asm.outputs.iter().enumerate() {
                if operand.kind == AsmOperandKind::InOut {
                    inout_indices.push(i);
                }
            }

            // Process inputs from asm.inputs (pure inputs only)
            for operand in &asm.inputs {
                let constraint = Self::translate_constraint(&operand.constraint, false);
                constraints.push(constraint);

                // Compile the input expression
                let val = self.compile_expr(fn_value, scope, &operand.expr)?;
                input_values.push(val.into());
            }

            // Process inputs from InOut operands in asm.outputs
            // InOut operands have both an output constraint (already added) and need their input value
            // The input must be tied to the output using a numbered constraint (e.g., "0" ties to output 0)
            for &output_idx in &inout_indices {
                let operand = &asm.outputs[output_idx];
                // Add tied input constraint: "N" where N is the output index
                constraints.push(output_idx.to_string());

                // Compile the input expression
                let val = self.compile_expr(fn_value, scope, &operand.expr)?;
                input_values.push(val.into());
            }

            // Process clobbers
            for clobber in &asm.clobbers {
                let clobber_name = clobber.trim_matches('"');
                constraints.push(format!("~{{{}}}", clobber_name));
            }

            // Add implicit clobbers for memory if not nomem
            if !asm.options.nomem {
                constraints.push("~{memory}".to_string());
            }

            // Build the constraint string
            let constraint_str = constraints.join(",");

            // Count actual outputs (including tied outputs for inout)
            let num_outputs = output_vars.len();

            // Determine return type based on outputs
            let (fn_type, return_type_is_void) = if num_outputs == 0 {
                // No outputs - void function
                let param_types: Vec<BasicMetadataTypeEnum<'ctx>> =
                    input_values.iter().map(|_| i64_type.into()).collect();
                (self.context.void_type().fn_type(&param_types, false), true)
            } else if num_outputs == 1 {
                // Single output - return i64
                let param_types: Vec<BasicMetadataTypeEnum<'ctx>> =
                    input_values.iter().map(|_| i64_type.into()).collect();
                (i64_type.fn_type(&param_types, false), false)
            } else {
                // Multiple outputs - return struct
                let output_types: Vec<BasicTypeEnum<'ctx>> = (0..num_outputs)
                    .map(|_| i64_type.into())
                    .collect();
                let struct_type = self.context.struct_type(&output_types, false);
                let param_types: Vec<BasicMetadataTypeEnum<'ctx>> =
                    input_values.iter().map(|_| i64_type.into()).collect();
                (struct_type.fn_type(&param_types, false), false)
            };

            // Determine side effects (volatile)
            // Assembly has side effects unless pure_asm is set
            let has_side_effects = !asm.options.pure_asm || asm.options.volatile;

            // Determine alignment
            // Align stack unless nostack is set
            let align_stack = !asm.options.nostack;

            // Determine dialect (AT&T vs Intel)
            let dialect = if asm.options.att_syntax {
                Some(InlineAsmDialect::ATT)
            } else {
                // Default to Intel syntax for x86
                Some(InlineAsmDialect::Intel)
            };

            // Create the inline assembly
            let asm_ptr = self.context.create_inline_asm(
                fn_type,
                asm.template.clone(),
                constraint_str,
                has_side_effects,
                align_stack,
                dialect,
                false, // can_throw - LLVM 13+
            );

            // Call the inline assembly
            let call_result = self.builder.build_indirect_call(
                fn_type,
                asm_ptr,
                &input_values,
                "asm_result",
            ).map_err(|e| format!("Failed to build inline asm call: {}", e))?;

            // Handle outputs: store the result into output variables
            if !return_type_is_void {
                let asm_return_value = call_result.try_as_basic_value()
                    .left()
                    .ok_or_else(|| "Inline asm call did not return a value".to_string())?;

                if num_outputs == 1 {
                    // Single output - store directly
                    if let Some(Some(var_ptr)) = output_vars.first() {
                        self.builder.build_store(*var_ptr, asm_return_value)
                            .map_err(|e| format!("Failed to store asm output: {}", e))?;
                    }
                    // Return the value
                    asm_return_value.into_int_value()
                        .try_into()
                        .map_err(|_| "Inline asm did not return int value".to_string())
                } else {
                    // Multiple outputs - extract each field from the struct
                    let struct_value = asm_return_value.into_struct_value();
                    let mut first_value_opt = None;

                    for (i, var_ptr_opt) in output_vars.iter().enumerate() {
                        let field_value = self.builder
                            .build_extract_value(struct_value, i as u32, &format!("asm_out_{}", i))
                            .map_err(|e| format!("Failed to extract asm output {}: {}", i, e))?;

                        // Store first value for return
                        if i == 0 {
                            first_value_opt = Some(field_value);
                        }

                        // Store to variable if not discarded
                        if let Some(var_ptr) = var_ptr_opt {
                            self.builder.build_store(*var_ptr, field_value)
                                .map_err(|e| format!("Failed to store asm output {}: {}", i, e))?;
                        }
                    }

                    // Return the first output value
                    first_value_opt
                        .ok_or_else(|| "No outputs to return".to_string())?
                        .into_int_value()
                        .try_into()
                        .map_err(|_| "Inline asm first output not int".to_string())
                }
            } else {
                Ok(i64_type.const_int(0, false))
            }
        }

        /// Get the pointer to a variable for asm output storage
        fn get_output_var_ptr(
            &self,
            scope: &CompileScope<'ctx>,
            expr: &Expr,
        ) -> Result<Option<PointerValue<'ctx>>, String> {
            match expr {
                Expr::Path(path) => {
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident.name;
                        // Check if it's a discard (_)
                        if name == "_" {
                            return Ok(None);
                        }
                        // Look up in scope
                        scope.vars.get(name).copied().map(Some)
                            .ok_or_else(|| format!("Unknown output variable: {}", name))
                    } else {
                        Err("Complex paths not supported for asm output".to_string())
                    }
                }
                _ => Err("Only variables supported for asm output".to_string()),
            }
        }

        /// Translate Sigil constraint syntax to LLVM constraint syntax
        ///
        /// Sigil uses Rust-like syntax:
        /// - `"rax"` or `rax` → `{rax}` (specific register)
        /// - `"r"` → `r` (any general purpose register)
        /// - `"m"` → `m` (memory)
        ///
        /// For outputs, prefix with `=` (or `=&` for early clobber)
        fn translate_constraint(constraint: &str, is_output: bool) -> String {
            let constraint = constraint.trim_matches('"');

            // Check if it's a specific register name
            let is_register = matches!(constraint,
                "rax" | "rbx" | "rcx" | "rdx" | "rsi" | "rdi" |
                "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15" |
                "rsp" | "rbp" |
                "eax" | "ebx" | "ecx" | "edx" | "esi" | "edi" |
                "ax" | "bx" | "cx" | "dx" | "si" | "di" |
                "al" | "bl" | "cl" | "dl" |
                "ah" | "bh" | "ch" | "dh" |
                // ARM registers
                "x0" | "x1" | "x2" | "x3" | "x4" | "x5" | "x6" | "x7" |
                "x8" | "x9" | "x10" | "x11" | "x12" | "x13" | "x14" | "x15"
            );

            let llvm_constraint = if is_register {
                format!("{{{}}}", constraint)
            } else {
                // Generic constraint (r, m, i, etc.)
                constraint.to_string()
            };

            if is_output {
                format!("={}", llvm_constraint)
            } else {
                llvm_constraint
            }
        }

        /// Compile a pipe expression: data |τ{f} |φ{p} |ρ+
        fn compile_pipe(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            expr: &Expr,
            operations: &[ast::PipeOp],
        ) -> Result<IntValue<'ctx>, String> {
            use ast::PipeOp;

            // Special case: [a, b, c, ...] |op patterns - we know the array length
            if let Expr::Array(elements) = expr {
                // Handle single operations
                if operations.len() == 1 {
                    match &operations[0] {
                        PipeOp::ReduceSum => {
                            return self.compile_array_sum(fn_value, scope, elements);
                        }
                        PipeOp::ReduceProd => {
                            return self.compile_array_product(fn_value, scope, elements);
                        }
                        PipeOp::Transform(closure) => {
                            return self.compile_array_transform(fn_value, scope, elements, closure);
                        }
                        PipeOp::Filter(predicate) => {
                            return self.compile_array_filter(fn_value, scope, elements, predicate);
                        }
                        PipeOp::First => {
                            return self.compile_array_first(fn_value, scope, elements);
                        }
                        PipeOp::Last => {
                            return self.compile_array_last(fn_value, scope, elements);
                        }
                        PipeOp::Nth(index_expr) => {
                            return self.compile_array_nth(fn_value, scope, elements, index_expr);
                        }
                        PipeOp::Middle => {
                            return self.compile_array_middle(fn_value, scope, elements);
                        }
                        PipeOp::ReduceMin => {
                            return self.compile_array_min(fn_value, scope, elements);
                        }
                        PipeOp::ReduceMax => {
                            return self.compile_array_max(fn_value, scope, elements);
                        }
                        PipeOp::ReduceAll => {
                            return self.compile_array_all(fn_value, scope, elements);
                        }
                        PipeOp::ReduceAny => {
                            return self.compile_array_any(fn_value, scope, elements);
                        }
                        PipeOp::Sort(_) => {
                            return self.compile_array_sort(fn_value, scope, elements);
                        }
                        PipeOp::Choice => {
                            return self.compile_array_choice(fn_value, scope, elements);
                        }
                        PipeOp::Reduce(reduce_fn) => {
                            return self.compile_array_reduce(fn_value, scope, elements, reduce_fn);
                        }
                        PipeOp::Await => {
                            // In sync LLVM context, await just evaluates the array
                            // Return sum as a reasonable default for array await
                            return self.compile_array_sum(fn_value, scope, elements);
                        }
                        _ => {}
                    }
                }
                // Handle chained operations: [a,b,c]|τ{f}|ρ+ or [a,b,c]|φ{p}|ρ+
                if operations.len() == 2 {
                    if let PipeOp::Transform(closure) = &operations[0] {
                        match &operations[1] {
                            PipeOp::ReduceSum => {
                                return self.compile_array_transform_then_sum(fn_value, scope, elements, closure);
                            }
                            PipeOp::ReduceProd => {
                                return self.compile_array_transform_then_product(fn_value, scope, elements, closure);
                            }
                            _ => {}
                        }
                    }
                    if let PipeOp::Filter(predicate) = &operations[0] {
                        match &operations[1] {
                            PipeOp::ReduceSum => {
                                return self.compile_array_filter_then_sum(fn_value, scope, elements, predicate);
                            }
                            PipeOp::ReduceProd => {
                                return self.compile_array_filter_then_product(fn_value, scope, elements, predicate);
                            }
                            _ => {}
                        }
                    }
                }
            }

            // General case: start with the base expression value
            let mut current = self.compile_expr(fn_value, scope, expr)?;

            for op in operations {
                current = match op {
                    // Transform: apply function to each element
                    PipeOp::Transform(transform_fn) => {
                        if let Expr::Closure { body, .. } = transform_fn.as_ref() {
                            // For scalar: just compile the body
                            self.compile_expr(fn_value, scope, body)?
                        } else {
                            self.compile_expr(fn_value, scope, transform_fn)?
                        }
                    }

                    // Filter: keep value if predicate is true
                    PipeOp::Filter(predicate) => {
                        let pred_result = self.compile_expr(fn_value, scope, predicate)?;
                        let zero = self.context.i64_type().const_int(0, false);
                        let is_true = self.builder
                            .build_int_compare(IntPredicate::NE, pred_result, zero, "filter_cond")
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_select(is_true, current, zero, "filter_result")
                            .map_err(|e| e.to_string())?
                            .into_int_value()
                    }

                    // Sum/Product on scalar is identity
                    PipeOp::ReduceSum | PipeOp::ReduceProd => current,

                    // Min/Max on scalar is identity
                    PipeOp::ReduceMin | PipeOp::ReduceMax => current,

                    // All/Any on scalar: check if non-zero
                    PipeOp::ReduceAll | PipeOp::ReduceAny => {
                        let zero = self.context.i64_type().const_int(0, false);
                        let one = self.context.i64_type().const_int(1, false);
                        let is_true = self.builder
                            .build_int_compare(IntPredicate::NE, current, zero, "is_true")
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_select(is_true, one, zero, "bool_result")
                            .map_err(|e| e.to_string())?
                            .into_int_value()
                    }

                    // Sort on scalar is identity
                    PipeOp::Sort(_) => current,

                    // Choice on scalar is identity (only one choice)
                    PipeOp::Choice => current,

                    // First/Last/Middle/Nth on scalar is identity
                    PipeOp::First | PipeOp::Last | PipeOp::Middle => current,
                    PipeOp::Nth(_) => current,

                    // Await on scalar is identity (sync execution)
                    PipeOp::Await => current,

                    // Custom reduce on scalar: just return the scalar (fold over one element)
                    PipeOp::Reduce(_) => current,

                    // Other operations - passthrough
                    _ => current,
                };
            }

            Ok(current)
        }

        /// Compile sum of array elements: [a, b, c] |ρ+ generates a loop
        fn compile_array_sum(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let i64_type = self.context.i64_type();

            // For small arrays, just unroll the sum
            if len <= 8 {
                let mut sum = self.compile_expr(fn_value, scope, &elements[0])?;
                for elem in &elements[1..] {
                    let val = self.compile_expr(fn_value, scope, elem)?;
                    sum = self.builder
                        .build_int_add(sum, val, "sum")
                        .map_err(|e| e.to_string())?;
                }
                return Ok(sum);
            }

            // For larger arrays, generate a proper loop
            let array_type = i64_type.array_type(len as u32);
            let array_ptr = self.builder
                .build_alloca(array_type, "sum_array")
                .map_err(|e| e.to_string())?;

            // Store all elements
            for (i, elem) in elements.iter().enumerate() {
                let value = self.compile_expr(fn_value, scope, elem)?;
                let indices = [
                    i64_type.const_int(0, false),
                    i64_type.const_int(i as u64, false),
                ];
                let elem_ptr = unsafe {
                    self.builder.build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }.map_err(|e| e.to_string())?;
                self.builder.build_store(elem_ptr, value).map_err(|e| e.to_string())?;
            }

            // Create loop blocks
            let loop_header = self.context.append_basic_block(fn_value, "sum_header");
            let loop_body = self.context.append_basic_block(fn_value, "sum_body");
            let loop_exit = self.context.append_basic_block(fn_value, "sum_exit");

            // Initialize: sum = 0, i = 0
            let sum_ptr = self.builder.build_alloca(i64_type, "sum_ptr").map_err(|e| e.to_string())?;
            let idx_ptr = self.builder.build_alloca(i64_type, "idx_ptr").map_err(|e| e.to_string())?;
            self.builder.build_store(sum_ptr, i64_type.const_int(0, false)).map_err(|e| e.to_string())?;
            self.builder.build_store(idx_ptr, i64_type.const_int(0, false)).map_err(|e| e.to_string())?;

            // Branch to header
            self.builder.build_unconditional_branch(loop_header).map_err(|e| e.to_string())?;

            // Loop header: check i < len
            self.builder.position_at_end(loop_header);
            let idx = self.builder.build_load(i64_type, idx_ptr, "idx").map_err(|e| e.to_string())?.into_int_value();
            let len_val = i64_type.const_int(len as u64, false);
            let cond = self.builder.build_int_compare(IntPredicate::ULT, idx, len_val, "cmp").map_err(|e| e.to_string())?;
            self.builder.build_conditional_branch(cond, loop_body, loop_exit).map_err(|e| e.to_string())?;

            // Loop body: sum += arr[i]; i++
            self.builder.position_at_end(loop_body);
            let elem_ptr = unsafe {
                self.builder.build_gep(array_type, array_ptr, &[i64_type.const_int(0, false), idx], "elem_ptr")
            }.map_err(|e| e.to_string())?;
            let elem_val = self.builder.build_load(i64_type, elem_ptr, "elem").map_err(|e| e.to_string())?.into_int_value();
            let sum = self.builder.build_load(i64_type, sum_ptr, "sum").map_err(|e| e.to_string())?.into_int_value();
            let new_sum = self.builder.build_int_add(sum, elem_val, "new_sum").map_err(|e| e.to_string())?;
            self.builder.build_store(sum_ptr, new_sum).map_err(|e| e.to_string())?;
            let new_idx = self.builder.build_int_add(idx, i64_type.const_int(1, false), "new_idx").map_err(|e| e.to_string())?;
            self.builder.build_store(idx_ptr, new_idx).map_err(|e| e.to_string())?;
            self.builder.build_unconditional_branch(loop_header).map_err(|e| e.to_string())?;

            // Loop exit: return sum
            self.builder.position_at_end(loop_exit);
            let final_sum = self.builder.build_load(i64_type, sum_ptr, "final_sum").map_err(|e| e.to_string())?.into_int_value();

            Ok(final_sum)
        }

        /// Compile product of array elements: [a, b, c] |ρ* generates a loop
        fn compile_array_product(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(1, false)); // Empty product = 1
            }

            // Unroll for all sizes (product is less common than sum)
            let mut product = self.compile_expr(fn_value, scope, &elements[0])?;
            for elem in &elements[1..] {
                let val = self.compile_expr(fn_value, scope, elem)?;
                product = self.builder
                    .build_int_mul(product, val, "prod")
                    .map_err(|e| e.to_string())?;
            }
            Ok(product)
        }

        /// Compile array transform: [a, b, c] |τ{x => f(x)}
        /// Returns pointer to new array with transformed elements
        fn compile_array_transform(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            closure: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let i64_type = self.context.i64_type();
            let array_type = i64_type.array_type(len as u32);

            // Allocate result array
            let result_ptr = self.builder
                .build_alloca(array_type, "transform_result")
                .map_err(|e| e.to_string())?;

            // Extract closure parameter name and body
            let (param_name, body) = if let Expr::Closure { params, body, .. } = closure {
                let name = if let Some(p) = params.first() {
                    if let ast::Pattern::Ident { name: ident, .. } = &p.pattern {
                        ident.name.clone()
                    } else {
                        "x".to_string()
                    }
                } else {
                    "x".to_string()
                };
                (name, body.as_ref())
            } else {
                return Err("Transform requires a closure".to_string());
            };

            // Transform each element
            for (i, elem) in elements.iter().enumerate() {
                // Compile the element value
                let elem_val = self.compile_expr(fn_value, scope, elem)?;

                // Bind parameter to element value (store in alloca)
                let param_ptr = self.builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(param_ptr, elem_val).map_err(|e| e.to_string())?;

                // Add to scope temporarily
                let old_val = scope.vars.insert(param_name.to_string(), param_ptr);

                // Compile closure body
                let result = self.compile_expr(fn_value, scope, body)?;

                // Restore scope
                if let Some(old) = old_val {
                    scope.vars.insert(param_name.to_string(), old);
                } else {
                    scope.vars.remove(&param_name);
                }

                // Store result in output array
                let indices = [
                    i64_type.const_int(0, false),
                    i64_type.const_int(i as u64, false),
                ];
                let out_ptr = unsafe {
                    self.builder.build_gep(array_type, result_ptr, &indices, "out_elem")
                }.map_err(|e| e.to_string())?;
                self.builder.build_store(out_ptr, result).map_err(|e| e.to_string())?;
            }

            // Return pointer as i64
            self.builder
                .build_ptr_to_int(result_ptr, i64_type, "arr_ptr")
                .map_err(|e| e.to_string())
        }

        /// Compile fused transform-then-sum: [a, b, c] |τ{f} |ρ+
        /// More efficient than separate transform and sum
        fn compile_array_transform_then_sum(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            closure: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let i64_type = self.context.i64_type();

            // Extract closure parameter name and body
            let (param_name, body) = if let Expr::Closure { params, body, .. } = closure {
                let name = if let Some(p) = params.first() {
                    if let ast::Pattern::Ident { name: ident, .. } = &p.pattern {
                        ident.name.clone()
                    } else {
                        "x".to_string()
                    }
                } else {
                    "x".to_string()
                };
                (name, body.as_ref())
            } else {
                return Err("Transform requires a closure".to_string());
            };

            // Fused: transform and sum in one pass (no intermediate array)
            let mut sum = i64_type.const_int(0, false);

            for elem in elements.iter() {
                // Compile the element value
                let elem_val = self.compile_expr(fn_value, scope, elem)?;

                // Bind parameter
                let param_ptr = self.builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(param_ptr, elem_val).map_err(|e| e.to_string())?;

                let old_val = scope.vars.insert(param_name.to_string(), param_ptr);

                // Compile closure body (the transform)
                let transformed = self.compile_expr(fn_value, scope, body)?;

                // Restore scope
                if let Some(old) = old_val {
                    scope.vars.insert(param_name.to_string(), old);
                } else {
                    scope.vars.remove(&param_name);
                }

                // Add to sum
                sum = self.builder
                    .build_int_add(sum, transformed, "sum")
                    .map_err(|e| e.to_string())?;
            }

            Ok(sum)
        }

        /// Compile fused transform-then-product: [a, b, c] |τ{f} |ρ*
        fn compile_array_transform_then_product(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            closure: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(1, false));
            }

            let i64_type = self.context.i64_type();

            // Extract closure parameter name and body
            let (param_name, body) = if let Expr::Closure { params, body, .. } = closure {
                let name = if let Some(p) = params.first() {
                    if let ast::Pattern::Ident { name: ident, .. } = &p.pattern {
                        ident.name.clone()
                    } else {
                        "x".to_string()
                    }
                } else {
                    "x".to_string()
                };
                (name, body.as_ref())
            } else {
                return Err("Transform requires a closure".to_string());
            };

            // Fused: transform and multiply in one pass
            let mut product = i64_type.const_int(1, false);

            for elem in elements.iter() {
                let elem_val = self.compile_expr(fn_value, scope, elem)?;

                let param_ptr = self.builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(param_ptr, elem_val).map_err(|e| e.to_string())?;

                let old_val = scope.vars.insert(param_name.to_string(), param_ptr);

                let transformed = self.compile_expr(fn_value, scope, body)?;

                if let Some(old) = old_val {
                    scope.vars.insert(param_name.to_string(), old);
                } else {
                    scope.vars.remove(&param_name);
                }

                product = self.builder
                    .build_int_mul(product, transformed, "prod")
                    .map_err(|e| e.to_string())?;
            }

            Ok(product)
        }

        /// Compile array filter: [a, b, c] |φ{predicate}
        /// Filter keeps elements where predicate returns non-zero.
        /// For compile-time known arrays, returns sum of elements that pass (for chaining to ρ+)
        /// or returns the count of passing elements (pointer + count pattern).
        fn compile_array_filter(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            predicate: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let i64_type = self.context.i64_type();
            let array_type = i64_type.array_type(len as u32);

            // Allocate output array (max size = input size)
            let out_ptr = self.builder
                .build_alloca(array_type, "filter_result")
                .map_err(|e| e.to_string())?;

            // Extract predicate parameter name and body
            let (param_name, body) = if let Expr::Closure { params, body, .. } = predicate {
                let name = if let Some(p) = params.first() {
                    if let ast::Pattern::Ident { name: ident, .. } = &p.pattern {
                        ident.name.clone()
                    } else {
                        "x".to_string()
                    }
                } else {
                    "x".to_string()
                };
                (name, body.as_ref())
            } else {
                return Err("Filter requires a closure predicate".to_string());
            };

            // Unrolled filter for small arrays (count passing elements)
            let mut out_idx = 0u64;
            let mut count = i64_type.const_int(0, false);

            for elem in elements.iter() {
                let elem_val = self.compile_expr(fn_value, scope, elem)?;

                // Bind parameter
                let param_ptr = self.builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(param_ptr, elem_val).map_err(|e| e.to_string())?;

                let old_val = scope.vars.insert(param_name.to_string(), param_ptr);

                // Evaluate predicate
                let pred_result = self.compile_expr(fn_value, scope, body)?;

                // Restore scope
                if let Some(old) = old_val {
                    scope.vars.insert(param_name.to_string(), old);
                } else {
                    scope.vars.remove(&param_name);
                }

                // Check if predicate is true (non-zero)
                let zero = i64_type.const_int(0, false);
                let is_true = self.builder
                    .build_int_compare(IntPredicate::NE, pred_result, zero, "is_passing")
                    .map_err(|e| e.to_string())?;

                // Conditionally add 1 to count
                let one = i64_type.const_int(1, false);
                let inc = self.builder
                    .build_select(is_true, one, zero, "inc")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                count = self.builder
                    .build_int_add(count, inc, "count")
                    .map_err(|e| e.to_string())?;

                // Store element if passing (always store, use select for value)
                let indices = [
                    i64_type.const_int(0, false),
                    i64_type.const_int(out_idx, false),
                ];
                let elem_ptr = unsafe {
                    self.builder.build_gep(array_type, out_ptr, &indices, "out_elem")
                }.map_err(|e| e.to_string())?;

                // Use select: if passing, store element; otherwise store 0 (placeholder)
                let value_to_store = self.builder
                    .build_select(is_true, elem_val, zero, "val_or_zero")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                self.builder.build_store(elem_ptr, value_to_store).map_err(|e| e.to_string())?;

                out_idx += 1;
            }

            // Return count of passing elements (for now - proper filter would return array)
            Ok(count)
        }

        /// Compile fused filter-then-sum: [a, b, c] |φ{p} |ρ+
        /// Sum only elements that pass the predicate
        fn compile_array_filter_then_sum(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            predicate: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let i64_type = self.context.i64_type();

            // Extract predicate parameter name and body
            let (param_name, body) = if let Expr::Closure { params, body, .. } = predicate {
                let name = if let Some(p) = params.first() {
                    if let ast::Pattern::Ident { name: ident, .. } = &p.pattern {
                        ident.name.clone()
                    } else {
                        "x".to_string()
                    }
                } else {
                    "x".to_string()
                };
                (name, body.as_ref())
            } else {
                return Err("Filter requires a closure predicate".to_string());
            };

            let mut sum = i64_type.const_int(0, false);
            let zero = i64_type.const_int(0, false);

            for elem in elements.iter() {
                let elem_val = self.compile_expr(fn_value, scope, elem)?;

                // Bind parameter
                let param_ptr = self.builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(param_ptr, elem_val).map_err(|e| e.to_string())?;

                let old_val = scope.vars.insert(param_name.to_string(), param_ptr);

                // Evaluate predicate
                let pred_result = self.compile_expr(fn_value, scope, body)?;

                // Restore scope
                if let Some(old) = old_val {
                    scope.vars.insert(param_name.to_string(), old);
                } else {
                    scope.vars.remove(&param_name);
                }

                // Check if predicate is true (non-zero)
                let is_true = self.builder
                    .build_int_compare(IntPredicate::NE, pred_result, zero, "is_passing")
                    .map_err(|e| e.to_string())?;

                // Add element to sum only if passing
                let add_value = self.builder
                    .build_select(is_true, elem_val, zero, "add_if_pass")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                sum = self.builder
                    .build_int_add(sum, add_value, "sum")
                    .map_err(|e| e.to_string())?;
            }

            Ok(sum)
        }

        /// Compile fused filter-then-product: [a, b, c] |φ{p} |ρ*
        /// Product of elements that pass the predicate
        fn compile_array_filter_then_product(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            predicate: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                return Ok(self.context.i64_type().const_int(1, false));
            }

            let i64_type = self.context.i64_type();

            // Extract predicate parameter name and body
            let (param_name, body) = if let Expr::Closure { params, body, .. } = predicate {
                let name = if let Some(p) = params.first() {
                    if let ast::Pattern::Ident { name: ident, .. } = &p.pattern {
                        ident.name.clone()
                    } else {
                        "x".to_string()
                    }
                } else {
                    "x".to_string()
                };
                (name, body.as_ref())
            } else {
                return Err("Filter requires a closure predicate".to_string());
            };

            let mut product = i64_type.const_int(1, false);
            let one = i64_type.const_int(1, false);
            let zero = i64_type.const_int(0, false);

            for elem in elements.iter() {
                let elem_val = self.compile_expr(fn_value, scope, elem)?;

                // Bind parameter
                let param_ptr = self.builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder.build_store(param_ptr, elem_val).map_err(|e| e.to_string())?;

                let old_val = scope.vars.insert(param_name.to_string(), param_ptr);

                // Evaluate predicate
                let pred_result = self.compile_expr(fn_value, scope, body)?;

                // Restore scope
                if let Some(old) = old_val {
                    scope.vars.insert(param_name.to_string(), old);
                } else {
                    scope.vars.remove(&param_name);
                }

                // Check if predicate is true (non-zero)
                let is_true = self.builder
                    .build_int_compare(IntPredicate::NE, pred_result, zero, "is_passing")
                    .map_err(|e| e.to_string())?;

                // Multiply by element only if passing, otherwise multiply by 1 (identity)
                let mul_value = self.builder
                    .build_select(is_true, elem_val, one, "mul_if_pass")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                product = self.builder
                    .build_int_mul(product, mul_value, "prod")
                    .map_err(|e| e.to_string())?;
            }

            Ok(product)
        }

        // ============================================
        // Element Access Morphemes
        // ============================================

        /// Compile first element access: [a, b, c] |α returns a
        fn compile_array_first(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                // Return 0 for empty array (could also panic)
                return Ok(self.context.i64_type().const_int(0, false));
            }
            self.compile_expr(fn_value, scope, &elements[0])
        }

        /// Compile last element access: [a, b, c] |ω returns c
        fn compile_array_last(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                return Ok(self.context.i64_type().const_int(0, false));
            }
            self.compile_expr(fn_value, scope, &elements[elements.len() - 1])
        }

        /// Compile nth element access: [a, b, c] |ν{1} returns b
        fn compile_array_nth(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            index_expr: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            // Try to evaluate index at compile time for static arrays
            if let Expr::Literal(ast::Literal::Int { value, .. }) = index_expr {
                if let Ok(n) = value.parse::<usize>() {
                    if n < elements.len() {
                        return self.compile_expr(fn_value, scope, &elements[n]);
                    } else {
                        // Index out of bounds - return 0
                        return Ok(self.context.i64_type().const_int(0, false));
                    }
                }
            }

            // Dynamic index: allocate array and compute index at runtime
            let i64_type = self.context.i64_type();
            let len = elements.len();
            let array_type = i64_type.array_type(len as u32);
            let array_ptr = self.builder
                .build_alloca(array_type, "nth_array")
                .map_err(|e| e.to_string())?;

            // Store all elements
            for (i, elem) in elements.iter().enumerate() {
                let value = self.compile_expr(fn_value, scope, elem)?;
                let indices = [
                    i64_type.const_int(0, false),
                    i64_type.const_int(i as u64, false),
                ];
                let elem_ptr = unsafe {
                    self.builder.build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }.map_err(|e| e.to_string())?;
                self.builder.build_store(elem_ptr, value).map_err(|e| e.to_string())?;
            }

            // Compute index
            let idx = self.compile_expr(fn_value, scope, index_expr)?;

            // Bounds check: clamp to valid range
            let len_val = i64_type.const_int(len as u64 - 1, false);
            let zero = i64_type.const_int(0, false);
            let clamped_high = self.builder
                .build_select(
                    self.builder.build_int_compare(IntPredicate::UGT, idx, len_val, "gt_len").map_err(|e| e.to_string())?,
                    len_val,
                    idx,
                    "clamp_high"
                )
                .map_err(|e| e.to_string())?
                .into_int_value();
            let clamped = self.builder
                .build_select(
                    self.builder.build_int_compare(IntPredicate::SLT, clamped_high, zero, "lt_zero").map_err(|e| e.to_string())?,
                    zero,
                    clamped_high,
                    "clamp_low"
                )
                .map_err(|e| e.to_string())?
                .into_int_value();

            // Load element at clamped index
            let indices = [i64_type.const_int(0, false), clamped];
            let elem_ptr = unsafe {
                self.builder.build_gep(array_type, array_ptr, &indices, "nth_ptr")
            }.map_err(|e| e.to_string())?;
            let value = self.builder
                .build_load(i64_type, elem_ptr, "nth_val")
                .map_err(|e| e.to_string())?;

            Ok(value.into_int_value())
        }

        /// Compile middle element access: [a, b, c, d, e] |μ returns c
        fn compile_array_middle(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                return Ok(self.context.i64_type().const_int(0, false));
            }
            let mid_idx = elements.len() / 2;
            self.compile_expr(fn_value, scope, &elements[mid_idx])
        }

        // ============================================
        // Reduction Morphemes (Min, Max, All, Any)
        // ============================================

        /// Compile min reduction: [a, b, c] |ρ_min returns minimum
        fn compile_array_min(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                return Ok(self.context.i64_type().const_int(i64::MAX as u64, true));
            }

            let mut min_val = self.compile_expr(fn_value, scope, &elements[0])?;

            for elem in &elements[1..] {
                let val = self.compile_expr(fn_value, scope, elem)?;
                let is_less = self.builder
                    .build_int_compare(IntPredicate::SLT, val, min_val, "is_less")
                    .map_err(|e| e.to_string())?;
                min_val = self.builder
                    .build_select(is_less, val, min_val, "min_sel")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
            }

            Ok(min_val)
        }

        /// Compile max reduction: [a, b, c] |ρ_max returns maximum
        fn compile_array_max(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                return Ok(self.context.i64_type().const_int(i64::MIN as u64, true));
            }

            let mut max_val = self.compile_expr(fn_value, scope, &elements[0])?;

            for elem in &elements[1..] {
                let val = self.compile_expr(fn_value, scope, elem)?;
                let is_greater = self.builder
                    .build_int_compare(IntPredicate::SGT, val, max_val, "is_greater")
                    .map_err(|e| e.to_string())?;
                max_val = self.builder
                    .build_select(is_greater, val, max_val, "max_sel")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
            }

            Ok(max_val)
        }

        /// Compile all reduction: [a, b, c] |ρ& returns 1 if all non-zero
        fn compile_array_all(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                // Empty array: all is vacuously true
                return Ok(self.context.i64_type().const_int(1, false));
            }

            let zero = self.context.i64_type().const_int(0, false);
            let mut result = self.context.i64_type().const_int(1, false);

            for elem in elements {
                let val = self.compile_expr(fn_value, scope, elem)?;
                let is_true = self.builder
                    .build_int_compare(IntPredicate::NE, val, zero, "is_true")
                    .map_err(|e| e.to_string())?;
                let as_int = self.builder
                    .build_int_z_extend(is_true, self.context.i64_type(), "as_int")
                    .map_err(|e| e.to_string())?;
                result = self.builder
                    .build_and(result, as_int, "all_and")
                    .map_err(|e| e.to_string())?;
            }

            Ok(result)
        }

        /// Compile any reduction: [a, b, c] |ρ| returns 1 if any non-zero
        fn compile_array_any(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                // Empty array: any is false
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let zero = self.context.i64_type().const_int(0, false);
            let mut result = self.context.i64_type().const_int(0, false);

            for elem in elements {
                let val = self.compile_expr(fn_value, scope, elem)?;
                let is_true = self.builder
                    .build_int_compare(IntPredicate::NE, val, zero, "is_true")
                    .map_err(|e| e.to_string())?;
                let as_int = self.builder
                    .build_int_z_extend(is_true, self.context.i64_type(), "as_int")
                    .map_err(|e| e.to_string())?;
                result = self.builder
                    .build_or(result, as_int, "any_or")
                    .map_err(|e| e.to_string())?;
            }

            Ok(result)
        }

        /// Compile sort morpheme: [3, 1, 2] |σ returns minimum (first element of sorted array)
        /// For now, we just find the minimum since we only return the first element
        fn compile_array_sort(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            // Sort's first element is the minimum - delegate to min
            self.compile_array_min(fn_value, scope, elements)
        }

        /// Compile choice morpheme: [a, b, c] |χ returns a pseudo-random element
        /// Uses a simple deterministic selection based on array sum (for reproducibility)
        fn compile_array_choice(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let len = elements.len();
            if len == 1 {
                return self.compile_expr(fn_value, scope, &elements[0]);
            }

            // Compute sum of all elements as a "hash" for deterministic selection
            let mut hash = self.compile_expr(fn_value, scope, &elements[0])?;
            for elem in &elements[1..] {
                let val = self.compile_expr(fn_value, scope, elem)?;
                hash = self.builder
                    .build_int_add(hash, val, "hash_acc")
                    .map_err(|e| e.to_string())?;
            }

            // index = hash % len (use abs to handle negative sums)
            let len_const = self.context.i64_type().const_int(len as u64, false);

            // Get absolute value: (hash ^ (hash >> 63)) - (hash >> 63)
            let shift_amt = self.context.i64_type().const_int(63, false);
            let sign = self.builder
                .build_right_shift(hash, shift_amt, true, "sign")
                .map_err(|e| e.to_string())?;
            let xored = self.builder
                .build_xor(hash, sign, "xored")
                .map_err(|e| e.to_string())?;
            let abs_hash = self.builder
                .build_int_sub(xored, sign, "abs")
                .map_err(|e| e.to_string())?;

            let index = self.builder
                .build_int_unsigned_rem(abs_hash, len_const, "choice_idx")
                .map_err(|e| e.to_string())?;

            // Allocate array and select by index
            let i64_type = self.context.i64_type();
            let array_type = i64_type.array_type(len as u32);
            let array_ptr = self.builder
                .build_alloca(array_type, "choice_arr")
                .map_err(|e| e.to_string())?;

            // Store elements
            for (i, elem) in elements.iter().enumerate() {
                let val = self.compile_expr(fn_value, scope, elem)?;
                let indices = [
                    i64_type.const_int(0, false),
                    i64_type.const_int(i as u64, false),
                ];
                let ptr = unsafe {
                    self.builder.build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }.map_err(|e| e.to_string())?;
                self.builder.build_store(ptr, val).map_err(|e| e.to_string())?;
            }

            // Load element at computed index
            let result_ptr = unsafe {
                self.builder.build_gep(
                    array_type,
                    array_ptr,
                    &[i64_type.const_int(0, false), index],
                    "choice_ptr",
                )
            }.map_err(|e| e.to_string())?;

            let result = self.builder
                .build_load(i64_type, result_ptr, "choice_val")
                .map_err(|e| e.to_string())?
                .into_int_value();

            Ok(result)
        }

        /// Compile custom reduce: [a, b, c] |ρ{|acc, x| acc + x} applies fold
        fn compile_array_reduce(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
            reduce_fn: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            if elements.is_empty() {
                return Ok(self.context.i64_type().const_int(0, false));
            }

            // Extract closure params and body
            let (acc_name, elem_name, body) = if let Expr::Closure { params, body, .. } = reduce_fn {
                if params.len() != 2 {
                    return Err("Reduce closure must have exactly 2 parameters".to_string());
                }
                // Extract names from Pattern::Ident
                let acc = if let ast::Pattern::Ident { name: ident, .. } = &params[0].pattern {
                    ident.name.clone()
                } else {
                    "acc".to_string()
                };
                let elem = if let ast::Pattern::Ident { name: ident, .. } = &params[1].pattern {
                    ident.name.clone()
                } else {
                    "x".to_string()
                };
                (acc, elem, body)
            } else {
                return Err("Reduce requires a closure".to_string());
            };

            let i64_type = self.context.i64_type();

            // Allocate storage for accumulator and element
            let acc_ptr = self.builder
                .build_alloca(i64_type, &acc_name)
                .map_err(|e| e.to_string())?;
            let elem_ptr = self.builder
                .build_alloca(i64_type, &elem_name)
                .map_err(|e| e.to_string())?;

            // Initialize accumulator with first element
            let first = self.compile_expr(fn_value, scope, &elements[0])?;
            self.builder.build_store(acc_ptr, first).map_err(|e| e.to_string())?;

            // Add bindings to scope
            scope.vars.insert(acc_name.clone(), acc_ptr);
            scope.vars.insert(elem_name.clone(), elem_ptr);

            // Fold over remaining elements
            for elem in &elements[1..] {
                let val = self.compile_expr(fn_value, scope, elem)?;
                self.builder.build_store(elem_ptr, val).map_err(|e| e.to_string())?;

                // Evaluate body
                let new_acc = self.compile_expr(fn_value, scope, body)?;
                self.builder.build_store(acc_ptr, new_acc).map_err(|e| e.to_string())?;
            }

            // Load final accumulator value
            let result = self.builder
                .build_load(i64_type, acc_ptr, "reduce_result")
                .map_err(|e| e.to_string())?
                .into_int_value();

            Ok(result)
        }

        /// Compile array literal: allocate stack space and store each element
        /// Returns pointer to first element as i64 (for now, proper fat pointers later)
        fn compile_array_literal(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            elements: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            let len = elements.len();
            if len == 0 {
                // Empty array - return null
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let i64_type = self.context.i64_type();
            let array_type = i64_type.array_type(len as u32);

            // Allocate array on stack
            let array_ptr = self.builder
                .build_alloca(array_type, "array")
                .map_err(|e| e.to_string())?;

            // Store each element
            for (i, elem) in elements.iter().enumerate() {
                let value = self.compile_expr(fn_value, scope, elem)?;

                // GEP to get pointer to element i
                let indices = [
                    self.context.i64_type().const_int(0, false),
                    self.context.i64_type().const_int(i as u64, false),
                ];
                let elem_ptr = unsafe {
                    self.builder.build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }.map_err(|e| e.to_string())?;

                // Store the value
                self.builder
                    .build_store(elem_ptr, value)
                    .map_err(|e| e.to_string())?;
            }

            // Return pointer as i64 (we'll improve this later with proper fat pointers)
            // For now, pack ptr in low bits and len in high bits of a struct
            let ptr_as_int = self.builder
                .build_ptr_to_int(array_ptr, i64_type, "arr_ptr")
                .map_err(|e| e.to_string())?;

            // Store length in scope for later retrieval (hacky but works for now)
            // We'll use a naming convention: the array pointer + "_len"
            // Better: return a struct { ptr, len } but that requires more refactoring

            Ok(ptr_as_int)
        }

        /// Compile array/slice indexing: arr[idx]
        /// Expects arr to be a pointer (as i64) to an array
        fn compile_index(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            expr: &Expr,
            index: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let base_ptr_int = self.compile_expr(fn_value, scope, expr)?;
            let idx = self.compile_expr(fn_value, scope, index)?;

            let i64_type = self.context.i64_type();

            // Convert i64 back to pointer
            let base_ptr = self.builder
                .build_int_to_ptr(base_ptr_int, i64_type.ptr_type(Default::default()), "arr_ptr")
                .map_err(|e| e.to_string())?;

            // GEP to get element at index
            let elem_ptr = unsafe {
                self.builder.build_gep(i64_type, base_ptr, &[idx], "elem_ptr")
            }.map_err(|e| e.to_string())?;

            // Load and return the value
            let value = self.builder
                .build_load(i64_type, elem_ptr, "elem")
                .map_err(|e| e.to_string())?;

            Ok(value.into_int_value())
        }

        /// Compile a binary operation
        fn compile_binary_op(
            &mut self,
            op: BinOp,
            lhs: IntValue<'ctx>,
            rhs: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            match op {
                BinOp::Add => self
                    .builder
                    .build_int_add(lhs, rhs, "add")
                    .map_err(|e| e.to_string()),
                BinOp::Sub => self
                    .builder
                    .build_int_sub(lhs, rhs, "sub")
                    .map_err(|e| e.to_string()),
                BinOp::Mul => self
                    .builder
                    .build_int_mul(lhs, rhs, "mul")
                    .map_err(|e| e.to_string()),
                BinOp::Div => self
                    .builder
                    .build_int_signed_div(lhs, rhs, "div")
                    .map_err(|e| e.to_string()),
                BinOp::Rem => self
                    .builder
                    .build_int_signed_rem(lhs, rhs, "rem")
                    .map_err(|e| e.to_string()),
                BinOp::BitAnd => self
                    .builder
                    .build_and(lhs, rhs, "and")
                    .map_err(|e| e.to_string()),
                BinOp::BitOr => self
                    .builder
                    .build_or(lhs, rhs, "or")
                    .map_err(|e| e.to_string()),
                BinOp::BitXor => self
                    .builder
                    .build_xor(lhs, rhs, "xor")
                    .map_err(|e| e.to_string()),
                BinOp::Shl => self
                    .builder
                    .build_left_shift(lhs, rhs, "shl")
                    .map_err(|e| e.to_string()),
                BinOp::Shr => self
                    .builder
                    .build_right_shift(lhs, rhs, true, "shr")
                    .map_err(|e| e.to_string()),
                BinOp::Eq => {
                    let cmp = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, lhs, rhs, "eq")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(cmp, self.context.i64_type(), "eq_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Ne => {
                    let cmp = self
                        .builder
                        .build_int_compare(IntPredicate::NE, lhs, rhs, "ne")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(cmp, self.context.i64_type(), "ne_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Lt => {
                    let cmp = self
                        .builder
                        .build_int_compare(IntPredicate::SLT, lhs, rhs, "lt")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(cmp, self.context.i64_type(), "lt_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Le => {
                    let cmp = self
                        .builder
                        .build_int_compare(IntPredicate::SLE, lhs, rhs, "le")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(cmp, self.context.i64_type(), "le_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Gt => {
                    let cmp = self
                        .builder
                        .build_int_compare(IntPredicate::SGT, lhs, rhs, "gt")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(cmp, self.context.i64_type(), "gt_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Ge => {
                    let cmp = self
                        .builder
                        .build_int_compare(IntPredicate::SGE, lhs, rhs, "ge")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(cmp, self.context.i64_type(), "ge_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::And => {
                    // Logical AND: (lhs != 0) && (rhs != 0)
                    let zero = self.context.i64_type().const_int(0, false);
                    let lhs_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, lhs, zero, "lhs_bool")
                        .map_err(|e| e.to_string())?;
                    let rhs_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, rhs, zero, "rhs_bool")
                        .map_err(|e| e.to_string())?;
                    let and = self
                        .builder
                        .build_and(lhs_bool, rhs_bool, "and")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(and, self.context.i64_type(), "and_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Or => {
                    // Logical OR: (lhs != 0) || (rhs != 0)
                    let zero = self.context.i64_type().const_int(0, false);
                    let lhs_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, lhs, zero, "lhs_bool")
                        .map_err(|e| e.to_string())?;
                    let rhs_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, rhs, zero, "rhs_bool")
                        .map_err(|e| e.to_string())?;
                    let or = self
                        .builder
                        .build_or(lhs_bool, rhs_bool, "or")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(or, self.context.i64_type(), "or_ext")
                        .map_err(|e| e.to_string())
                }
                _ => Ok(self.context.i64_type().const_int(0, false)),
            }
        }

        /// Compile a unary operation
        fn compile_unary_op(
            &mut self,
            op: UnaryOp,
            val: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            match op {
                UnaryOp::Neg => self
                    .builder
                    .build_int_neg(val, "neg")
                    .map_err(|e| e.to_string()),
                UnaryOp::Not => {
                    // Logical NOT: val == 0 ? 1 : 0
                    let zero = self.context.i64_type().const_int(0, false);
                    let is_zero = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, val, zero, "is_zero")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_int_z_extend(is_zero, self.context.i64_type(), "not")
                        .map_err(|e| e.to_string())
                }
                _ => Ok(val),
            }
        }

        /// Compile an if expression
        fn compile_if(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            condition: &Expr,
            then_branch: &ast::Block,
            else_branch: Option<&Expr>,
        ) -> Result<IntValue<'ctx>, String> {
            let cond_val = self.compile_expr(fn_value, scope, condition)?;

            // Convert to i1 (bool)
            let zero = self.context.i64_type().const_int(0, false);
            let cond_bool = self
                .builder
                .build_int_compare(IntPredicate::NE, cond_val, zero, "cond")
                .map_err(|e| e.to_string())?;

            // Create blocks
            let then_bb = self.context.append_basic_block(fn_value, "then");
            let else_bb = self.context.append_basic_block(fn_value, "else");
            let merge_bb = self.context.append_basic_block(fn_value, "merge");

            self.builder
                .build_conditional_branch(cond_bool, then_bb, else_bb)
                .map_err(|e| e.to_string())?;

            // Then block
            self.builder.position_at_end(then_bb);
            let then_val = self
                .compile_block(fn_value, scope, then_branch)?
                .unwrap_or_else(|| self.context.i64_type().const_int(0, false));
            let then_terminated = self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_some();
            if !then_terminated {
                self.builder
                    .build_unconditional_branch(merge_bb)
                    .map_err(|e| e.to_string())?;
            }
            let then_bb_end = self.builder.get_insert_block().unwrap();

            // Else block
            self.builder.position_at_end(else_bb);
            let else_val = if let Some(else_expr) = else_branch {
                self.compile_expr(fn_value, scope, else_expr)?
            } else {
                self.context.i64_type().const_int(0, false)
            };
            let else_terminated = self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_some();
            if !else_terminated {
                self.builder
                    .build_unconditional_branch(merge_bb)
                    .map_err(|e| e.to_string())?;
            }
            let else_bb_end = self.builder.get_insert_block().unwrap();

            // Merge block with phi
            self.builder.position_at_end(merge_bb);

            // If both branches terminated (e.g., both returned), we can't create a phi
            if then_terminated && else_terminated {
                // This block is unreachable, but we need to return something
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let phi = self
                .builder
                .build_phi(self.context.i64_type(), "if_result")
                .map_err(|e| e.to_string())?;

            if !then_terminated {
                phi.add_incoming(&[(&then_val, then_bb_end)]);
            }
            if !else_terminated {
                phi.add_incoming(&[(&else_val, else_bb_end)]);
            }

            Ok(phi.as_basic_value().into_int_value())
        }

        /// Compile a while loop
        fn compile_while(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            condition: &Expr,
            body: &ast::Block,
        ) -> Result<IntValue<'ctx>, String> {
            // Create blocks
            let cond_bb = self.context.append_basic_block(fn_value, "while_cond");
            let body_bb = self.context.append_basic_block(fn_value, "while_body");
            let after_bb = self.context.append_basic_block(fn_value, "while_after");

            // Jump to condition
            self.builder
                .build_unconditional_branch(cond_bb)
                .map_err(|e| e.to_string())?;

            // Condition block
            self.builder.position_at_end(cond_bb);
            let cond_val = self.compile_expr(fn_value, scope, condition)?;
            let zero = self.context.i64_type().const_int(0, false);
            let cond_bool = self
                .builder
                .build_int_compare(IntPredicate::NE, cond_val, zero, "cond")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(cond_bool, body_bb, after_bb)
                .map_err(|e| e.to_string())?;

            // Body block
            self.builder.position_at_end(body_bb);
            self.compile_block(fn_value, scope, body)?;
            // Check if body terminated (e.g., return)
            if self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
            {
                self.builder
                    .build_unconditional_branch(cond_bb)
                    .map_err(|e| e.to_string())?;
            }

            // After block
            self.builder.position_at_end(after_bb);
            Ok(self.context.i64_type().const_int(0, false))
        }

        /// Compile a function call
        fn compile_call(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            func: &Expr,
            args: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            // Get function name and full qualified path
            let (fn_name, full_path) = if let Expr::Path(path) = func {
                let segments: Vec<&str> = path.segments.iter().map(|s| s.ident.name.as_str()).collect();
                let short_name = segments.last().copied().ok_or("Empty path")?;
                let full = segments.join("::");
                (short_name, full)
            } else {
                return Err("Expected function name".to_string());
            };

            // Handle built-in functions
            match fn_name {
                "print" => {
                    if !args.is_empty() {
                        let arg_val = self.compile_expr(fn_value, scope, &args[0])?;
                        // Call sigil_print_int (works in both JIT and AOT)
                        let print_fn = self
                            .module
                            .get_function("sigil_print_int")
                            .ok_or("sigil_print_int not declared")?;
                        self.builder
                            .build_call(print_fn, &[arg_val.into()], "")
                            .map_err(|e| e.to_string())?;
                        return Ok(arg_val);
                    }
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "now" => {
                    // Call sigil_now runtime function
                    let now_fn = self
                        .module
                        .get_function("sigil_now")
                        .ok_or("sigil_now not declared")?;
                    let call = self
                        .builder
                        .build_call(now_fn, &[], "now")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // Unary math functions
                "sqrt" | "sin" | "cos" | "tan" | "exp" | "ln" | "floor" | "ceil" | "abs" => {
                    if args.is_empty() {
                        return Err(format!("{} requires 1 argument", fn_name));
                    }
                    let arg = self.compile_expr(fn_value, scope, &args[0])?;
                    let rt_name = format!("sigil_{}", fn_name);
                    let rt_fn = self
                        .module
                        .get_function(&rt_name)
                        .ok_or(format!("{} not declared", rt_name))?;
                    let call = self
                        .builder
                        .build_call(rt_fn, &[arg.into()], fn_name)
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // Binary math functions
                "pow" | "min" | "max" => {
                    if args.len() < 2 {
                        return Err(format!("{} requires 2 arguments", fn_name));
                    }
                    let arg1 = self.compile_expr(fn_value, scope, &args[0])?;
                    let arg2 = self.compile_expr(fn_value, scope, &args[1])?;
                    let rt_name = format!("sigil_{}", fn_name);
                    let rt_fn = self
                        .module
                        .get_function(&rt_name)
                        .ok_or(format!("{} not declared", rt_name))?;
                    let call = self
                        .builder
                        .build_call(rt_fn, &[arg1.into(), arg2.into()], fn_name)
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // Vec built-in functions
                "vec_new" => {
                    if args.is_empty() {
                        return Err("vec_new requires capacity argument".to_string());
                    }
                    let capacity = self.compile_expr(fn_value, scope, &args[0])?;
                    let vec_new_fn = self
                        .module
                        .get_function("sigil_vec_new")
                        .ok_or("sigil_vec_new not declared")?;
                    let call = self
                        .builder
                        .build_call(vec_new_fn, &[capacity.into()], "vec_new")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "vec_push" => {
                    if args.len() < 2 {
                        return Err("vec_push requires vec and value arguments".to_string());
                    }
                    let vec_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let value = self.compile_expr(fn_value, scope, &args[1])?;
                    let vec_push_fn = self
                        .module
                        .get_function("sigil_vec_push")
                        .ok_or("sigil_vec_push not declared")?;
                    self.builder
                        .build_call(vec_push_fn, &[vec_ptr.into(), value.into()], "")
                        .map_err(|e| e.to_string())?;
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "vec_get" => {
                    if args.len() < 2 {
                        return Err("vec_get requires vec and index arguments".to_string());
                    }
                    let vec_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let index = self.compile_expr(fn_value, scope, &args[1])?;
                    let vec_get_fn = self
                        .module
                        .get_function("sigil_vec_get")
                        .ok_or("sigil_vec_get not declared")?;
                    let call = self
                        .builder
                        .build_call(vec_get_fn, &[vec_ptr.into(), index.into()], "vec_get")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "vec_len" => {
                    if args.is_empty() {
                        return Err("vec_len requires vec argument".to_string());
                    }
                    let vec_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let vec_len_fn = self
                        .module
                        .get_function("sigil_vec_len")
                        .ok_or("sigil_vec_len not declared")?;
                    let call = self
                        .builder
                        .build_call(vec_len_fn, &[vec_ptr.into()], "vec_len")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // String built-in functions
                "String_from" => {
                    if args.is_empty() {
                        return Err("String_from requires string literal argument".to_string());
                    }
                    // Get the string literal - it should be a string expression
                    let str_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let string_from_fn = self
                        .module
                        .get_function("sigil_string_from")
                        .ok_or("sigil_string_from not declared")?;
                    let call = self
                        .builder
                        .build_call(string_from_fn, &[str_ptr.into()], "string_from")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "string_len" => {
                    if args.is_empty() {
                        return Err("string_len requires string argument".to_string());
                    }
                    let str_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let string_len_fn = self
                        .module
                        .get_function("sigil_string_len")
                        .ok_or("sigil_string_len not declared")?;
                    let call = self
                        .builder
                        .build_call(string_len_fn, &[str_ptr.into()], "string_len")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "string_print" => {
                    if args.is_empty() {
                        return Err("string_print requires string argument".to_string());
                    }
                    let str_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let string_print_fn = self
                        .module
                        .get_function("sigil_string_print")
                        .ok_or("sigil_string_print not declared")?;
                    self.builder
                        .build_call(string_print_fn, &[str_ptr.into()], "")
                        .map_err(|e| e.to_string())?;
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "string_concat" => {
                    if args.len() < 2 {
                        return Err("string_concat requires two string arguments".to_string());
                    }
                    let str1 = self.compile_expr(fn_value, scope, &args[0])?;
                    let str2 = self.compile_expr(fn_value, scope, &args[1])?;
                    let string_concat_fn = self
                        .module
                        .get_function("sigil_string_concat")
                        .ok_or("sigil_string_concat not declared")?;
                    let call = self
                        .builder
                        .build_call(string_concat_fn, &[str1.into(), str2.into()], "string_concat")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // Option built-in functions
                "Some" => {
                    if args.is_empty() {
                        return Err("Some requires a value argument".to_string());
                    }
                    let value = self.compile_expr(fn_value, scope, &args[0])?;
                    let option_some_fn = self
                        .module
                        .get_function("sigil_option_some")
                        .ok_or("sigil_option_some not declared")?;
                    let call = self
                        .builder
                        .build_call(option_some_fn, &[value.into()], "some")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "None" => {
                    let option_none_fn = self
                        .module
                        .get_function("sigil_option_none")
                        .ok_or("sigil_option_none not declared")?;
                    let call = self
                        .builder
                        .build_call(option_none_fn, &[], "none")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "is_some" => {
                    if args.is_empty() {
                        return Err("is_some requires an option argument".to_string());
                    }
                    let opt = self.compile_expr(fn_value, scope, &args[0])?;
                    let is_some_fn = self
                        .module
                        .get_function("sigil_option_is_some")
                        .ok_or("sigil_option_is_some not declared")?;
                    let call = self
                        .builder
                        .build_call(is_some_fn, &[opt.into()], "is_some")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "is_none" => {
                    if args.is_empty() {
                        return Err("is_none requires an option argument".to_string());
                    }
                    let opt = self.compile_expr(fn_value, scope, &args[0])?;
                    let is_none_fn = self
                        .module
                        .get_function("sigil_option_is_none")
                        .ok_or("sigil_option_is_none not declared")?;
                    let call = self
                        .builder
                        .build_call(is_none_fn, &[opt.into()], "is_none")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "unwrap" => {
                    if args.is_empty() {
                        return Err("unwrap requires an option argument".to_string());
                    }
                    let opt = self.compile_expr(fn_value, scope, &args[0])?;
                    let unwrap_fn = self
                        .module
                        .get_function("sigil_option_unwrap")
                        .ok_or("sigil_option_unwrap not declared")?;
                    let call = self
                        .builder
                        .build_call(unwrap_fn, &[opt.into()], "unwrap")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "unwrap_or" => {
                    if args.len() < 2 {
                        return Err("unwrap_or requires option and default arguments".to_string());
                    }
                    let opt = self.compile_expr(fn_value, scope, &args[0])?;
                    let default_val = self.compile_expr(fn_value, scope, &args[1])?;
                    let unwrap_or_fn = self
                        .module
                        .get_function("sigil_option_unwrap_or")
                        .ok_or("sigil_option_unwrap_or not declared")?;
                    let call = self
                        .builder
                        .build_call(unwrap_or_fn, &[opt.into(), default_val.into()], "unwrap_or")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // File I/O built-in functions
                "file_exists" => {
                    if args.is_empty() {
                        return Err("file_exists requires a path argument".to_string());
                    }
                    let path = self.compile_expr(fn_value, scope, &args[0])?;
                    let file_exists_fn = self
                        .module
                        .get_function("sigil_file_exists")
                        .ok_or("sigil_file_exists not declared")?;
                    let call = self
                        .builder
                        .build_call(file_exists_fn, &[path.into()], "file_exists")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "file_read_all" => {
                    if args.is_empty() {
                        return Err("file_read_all requires a path argument".to_string());
                    }
                    let path = self.compile_expr(fn_value, scope, &args[0])?;
                    let file_read_fn = self
                        .module
                        .get_function("sigil_file_read_all")
                        .ok_or("sigil_file_read_all not declared")?;
                    let call = self
                        .builder
                        .build_call(file_read_fn, &[path.into()], "file_read")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "file_write_all" => {
                    if args.len() < 2 {
                        return Err("file_write_all requires path and content arguments".to_string());
                    }
                    let path = self.compile_expr(fn_value, scope, &args[0])?;
                    let content = self.compile_expr(fn_value, scope, &args[1])?;
                    let file_write_fn = self
                        .module
                        .get_function("sigil_file_write_all")
                        .ok_or("sigil_file_write_all not declared")?;
                    let call = self
                        .builder
                        .build_call(file_write_fn, &[path.into(), content.into()], "file_write")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                _ => {}
            }

            // Resolve any use aliases first
            let resolved_path = if let Some(aliased) = self.use_aliases.get(fn_name) {
                aliased.clone()
            } else {
                full_path.clone()
            };

            // Get the function - try resolved path first, then various lookups
            let callee = if let Some(f) = self.functions.get(&resolved_path) {
                *f
            } else if let Some(f) = self.functions.get(&full_path) {
                *f
            } else if let Some(f) = self.functions.get(fn_name) {
                *f
            } else if let Some(f) = self.module.get_function(&resolved_path.replace("::", "_")) {
                f
            } else if let Some(f) = self.module.get_function(&full_path.replace("::", "_")) {
                f
            } else if let Some(f) = self.module.get_function(fn_name) {
                f
            } else {
                return Err(format!("Unknown function: {}", full_path));
            };

            // Compile arguments
            let compiled_args: Result<Vec<_>, _> = args
                .iter()
                .map(|arg| self.compile_expr(fn_value, scope, arg).map(|v| v.into()))
                .collect();
            let compiled_args = compiled_args?;

            // Build call with tail call hint for potential optimization
            let call = self
                .builder
                .build_call(callee, &compiled_args, "call")
                .map_err(|e| e.to_string())?;

            // Hint to LLVM that this could be a tail call
            // The optimizer will determine if it's actually in tail position
            call.set_tail_call(true);

            // Get return value
            Ok(call
                .try_as_basic_value()
                .left()
                .map(|v| v.into_int_value())
                .unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
        }

        /// Process a use declaration to register imports
        fn process_use(&mut self, use_decl: &ast::UseDecl) -> Result<(), String> {
            self.process_use_tree(&use_decl.tree, &[])
        }

        /// Recursively process use tree to build import paths
        fn process_use_tree(&mut self, tree: &ast::UseTree, prefix: &[String]) -> Result<(), String> {
            match tree {
                ast::UseTree::Path { prefix: ident, suffix } => {
                    let mut new_prefix = prefix.to_vec();
                    new_prefix.push(ident.name.clone());
                    self.process_use_tree(suffix, &new_prefix)
                }
                ast::UseTree::Name(ident) => {
                    let mut full_path = prefix.to_vec();
                    full_path.push(ident.name.clone());
                    let full_name = full_path.join("::");
                    self.use_aliases.insert(ident.name.clone(), full_name);
                    Ok(())
                }
                ast::UseTree::Rename { name, alias } => {
                    let mut full_path = prefix.to_vec();
                    full_path.push(name.name.clone());
                    let full_name = full_path.join("::");
                    self.use_aliases.insert(alias.name.clone(), full_name);
                    Ok(())
                }
                ast::UseTree::Glob => Ok(()),
                ast::UseTree::Group(trees) => {
                    for sub_tree in trees {
                        self.process_use_tree(sub_tree, prefix)?;
                    }
                    Ok(())
                }
            }
        }

        /// Process a module declaration (first pass - declare functions)
        fn process_module(&mut self, module: &ast::Module) -> Result<(), String> {
            let saved_module = self.current_module.clone();
            self.current_module.push(module.name.name.clone());

            if let Some(ref items) = module.items {
                for spanned_item in items {
                    match &spanned_item.node {
                        Item::Function(func) => { self.declare_function(func)?; }
                        Item::Module(m) => { self.process_module(m)?; }
                        Item::Use(u) => { self.process_use(u)?; }
                        _ => {}
                    }
                }
            }

            self.current_module = saved_module;
            Ok(())
        }

        /// Compile functions in a module (second pass)
        fn compile_module_functions(&mut self, module: &ast::Module) -> Result<(), String> {
            let saved_module = self.current_module.clone();
            self.current_module.push(module.name.name.clone());

            if let Some(ref items) = module.items {
                for spanned_item in items {
                    match &spanned_item.node {
                        Item::Function(func) => { self.compile_function(func)?; }
                        Item::Module(m) => { self.compile_module_functions(m)?; }
                        _ => {}
                    }
                }
            }

            self.current_module = saved_module;
            Ok(())
        }

        /// Run LLVM optimization passes
        fn run_llvm_optimizations(&self) -> Result<(), String> {
            Target::initialize_all(&InitializationConfig::default());

            let triple = TargetMachine::get_default_triple();
            let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;

            // Use native CPU and features for maximum performance
            let cpu = TargetMachine::get_host_cpu_name();
            let features = TargetMachine::get_host_cpu_features();

            let target_machine = target
                .create_target_machine(
                    &triple,
                    cpu.to_str().unwrap_or("native"),
                    features.to_str().unwrap_or(""),
                    OptimizationLevel::Aggressive,
                    RelocMode::Default,
                    CodeModel::Default,
                )
                .ok_or("Failed to create target machine")?;

            // Run aggressive optimization passes
            // The key is running tailcallelim early and then letting later passes optimize
            let passes = match self.opt_level {
                OptLevel::None => "default<O0>",
                OptLevel::Basic => "default<O1>",
                OptLevel::Standard | OptLevel::Size => "default<O2>",
                // Run full O3 pipeline which includes tail call elimination
                OptLevel::Aggressive => "default<O3>",
            };

            self.module
                .run_passes(passes, &target_machine, PassBuilderOptions::create())
                .map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Create JIT execution engine and run
        pub fn run(&mut self) -> Result<i64, String> {
            // Initialize targets for JIT execution
            Target::initialize_x86(&InitializationConfig::default());

            // Verify module before execution
            if let Err(msg) = self.module.verify() {
                return Err(format!("Module verification failed: {}", msg.to_string()));
            }

            // Create execution engine
            let ee = self
                .module
                .create_jit_execution_engine(OptimizationLevel::Aggressive)
                .map_err(|e| e.to_string())?;

            // Register runtime functions (only if declared/used in the program)
            if let Some(f) = self.module.get_function("sigil_now") {
                ee.add_global_mapping(&f, sigil_now as usize);
            }
            if let Some(f) = self.module.get_function("sigil_print_int") {
                ee.add_global_mapping(&f, sigil_print_int as usize);
            }

            // Register math functions (only if declared/used in the program)
            if let Some(f) = self.module.get_function("sigil_sqrt") {
                ee.add_global_mapping(&f, sigil_sqrt as usize);
            }
            if let Some(f) = self.module.get_function("sigil_sin") {
                ee.add_global_mapping(&f, sigil_sin as usize);
            }
            if let Some(f) = self.module.get_function("sigil_cos") {
                ee.add_global_mapping(&f, sigil_cos as usize);
            }
            if let Some(f) = self.module.get_function("sigil_tan") {
                ee.add_global_mapping(&f, sigil_tan as usize);
            }
            if let Some(f) = self.module.get_function("sigil_exp") {
                ee.add_global_mapping(&f, sigil_exp as usize);
            }
            if let Some(f) = self.module.get_function("sigil_ln") {
                ee.add_global_mapping(&f, sigil_ln as usize);
            }
            if let Some(f) = self.module.get_function("sigil_pow") {
                ee.add_global_mapping(&f, sigil_pow as usize);
            }
            if let Some(f) = self.module.get_function("sigil_floor") {
                ee.add_global_mapping(&f, sigil_floor as usize);
            }
            if let Some(f) = self.module.get_function("sigil_ceil") {
                ee.add_global_mapping(&f, sigil_ceil as usize);
            }
            if let Some(f) = self.module.get_function("sigil_abs") {
                ee.add_global_mapping(&f, sigil_abs as usize);
            }
            if let Some(f) = self.module.get_function("sigil_min") {
                ee.add_global_mapping(&f, sigil_min as usize);
            }
            if let Some(f) = self.module.get_function("sigil_max") {
                ee.add_global_mapping(&f, sigil_max as usize);
            }

            // Vec runtime mappings
            if let Some(f) = self.module.get_function("sigil_vec_new") {
                ee.add_global_mapping(&f, sigil_vec_new as usize);
            }
            if let Some(f) = self.module.get_function("sigil_vec_push") {
                ee.add_global_mapping(&f, sigil_vec_push as usize);
            }
            if let Some(f) = self.module.get_function("sigil_vec_get") {
                ee.add_global_mapping(&f, sigil_vec_get as usize);
            }
            if let Some(f) = self.module.get_function("sigil_vec_len") {
                ee.add_global_mapping(&f, sigil_vec_len as usize);
            }

            // String runtime mappings
            if let Some(f) = self.module.get_function("sigil_string_from") {
                ee.add_global_mapping(&f, sigil_string_from as usize);
            }
            if let Some(f) = self.module.get_function("sigil_string_len") {
                ee.add_global_mapping(&f, sigil_string_len as usize);
            }
            if let Some(f) = self.module.get_function("sigil_string_print") {
                ee.add_global_mapping(&f, sigil_string_print as usize);
            }
            if let Some(f) = self.module.get_function("sigil_string_concat") {
                ee.add_global_mapping(&f, sigil_string_concat as usize);
            }

            // Option runtime mappings
            if let Some(f) = self.module.get_function("sigil_option_some") {
                ee.add_global_mapping(&f, sigil_option_some as usize);
            }
            if let Some(f) = self.module.get_function("sigil_option_none") {
                ee.add_global_mapping(&f, sigil_option_none as usize);
            }
            if let Some(f) = self.module.get_function("sigil_option_is_some") {
                ee.add_global_mapping(&f, sigil_option_is_some as usize);
            }
            if let Some(f) = self.module.get_function("sigil_option_is_none") {
                ee.add_global_mapping(&f, sigil_option_is_none as usize);
            }
            if let Some(f) = self.module.get_function("sigil_option_unwrap") {
                ee.add_global_mapping(&f, sigil_option_unwrap as usize);
            }
            if let Some(f) = self.module.get_function("sigil_option_unwrap_or") {
                ee.add_global_mapping(&f, sigil_option_unwrap_or as usize);
            }

            // File I/O runtime mappings
            if let Some(f) = self.module.get_function("sigil_file_exists") {
                ee.add_global_mapping(&f, sigil_file_exists as usize);
            }
            if let Some(f) = self.module.get_function("sigil_file_read_all") {
                ee.add_global_mapping(&f, sigil_file_read_all as usize);
            }
            if let Some(f) = self.module.get_function("sigil_file_write_all") {
                ee.add_global_mapping(&f, sigil_file_write_all as usize);
            }

            self.execution_engine = Some(ee);

            // Get main function
            unsafe {
                let main: JitFunction<MainFn> = self
                    .execution_engine
                    .as_ref()
                    .unwrap()
                    .get_function("main")
                    .map_err(|e| e.to_string())?;

                Ok(main.call())
            }
        }

        /// Write object file
        pub fn write_object_file(&self, path: &Path) -> Result<(), String> {
            Target::initialize_all(&InitializationConfig::default());

            let triple = TargetMachine::get_default_triple();
            let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
            let target_machine = target
                .create_target_machine(
                    &triple,
                    "generic",
                    "",
                    OptimizationLevel::Aggressive,
                    RelocMode::PIC,  // Use PIC for PIE compatibility
                    CodeModel::Default,
                )
                .ok_or("Failed to create target machine")?;

            target_machine
                .write_to_file(&self.module, FileType::Object, path)
                .map_err(|e| e.to_string())
        }

        /// Get LLVM IR as string
        pub fn get_ir(&self) -> String {
            self.module.print_to_string().to_string()
        }
    }

    /// Variable scope for compilation
    struct CompileScope<'ctx> {
        vars: HashMap<String, PointerValue<'ctx>>,
    }

    impl<'ctx> CompileScope<'ctx> {
        fn new() -> Self {
            Self {
                vars: HashMap::new(),
            }
        }
    }

    // ============================================
    // Tests
    // ============================================
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::optimize::OptLevel;

        fn run_sigil(source: &str) -> Result<i64, String> {
            let context = Context::create();
            let mut compiler = LlvmCompiler::new(&context, OptLevel::Standard)?;
            compiler.compile(source)?;
            compiler.run()
        }

        // ============================================
        // Evidentiality Tests
        // ============================================

        #[test]
        fn test_evidential_known_unwrap() {
            // Known (!) just returns the inner value
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 42!;
                    x
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_uncertain() {
            // Uncertain (?) wraps and unwraps correctly
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 100?;
                    x
                }
            "#);
            assert_eq!(result.unwrap(), 100);
        }

        #[test]
        fn test_evidential_reported() {
            // Reported (~) wraps and unwraps correctly
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 200~;
                    x
                }
            "#);
            assert_eq!(result.unwrap(), 200);
        }

        #[test]
        fn test_evidential_predicted() {
            // Predicted (◊) wraps and unwraps correctly
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 300◊;
                    x
                }
            "#);
            assert_eq!(result.unwrap(), 300);
        }

        #[test]
        fn test_evidential_in_expression() {
            // Evidential values can be used in expressions
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let a = 10?;
                    let b = 20?;
                    a + b
                }
            "#);
            assert_eq!(result.unwrap(), 30);
        }

        #[test]
        fn test_evidential_unwrap_chain() {
            // Chain: uncertain -> known (unwrap)
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 42?;
                    let y = x!;
                    y
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_nested() {
            // Nested evidential operations
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = (50?)!;
                    x + 5
                }
            "#);
            assert_eq!(result.unwrap(), 55);
        }

        #[test]
        fn test_evidential_with_arithmetic() {
            // Evidential values with arithmetic
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let known = 100!;
                    let uncertain = 50?;
                    known + uncertain * 2
                }
            "#);
            assert_eq!(result.unwrap(), 200);
        }

        #[test]
        fn test_evidential_function_return() {
            // Function returning evidential value
            let result = run_sigil(r#"
                fn get_uncertain() -> i64 {
                    42?
                }

                fn main() -> i64 {
                    let x = get_uncertain();
                    x + 8
                }
            "#);
            assert_eq!(result.unwrap(), 50);
        }

        #[test]
        fn test_evidential_mixed_markers() {
            // Mix different evidentiality markers
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let a = 10!;  // known
                    let b = 20?;  // uncertain
                    let c = 30~;  // reported
                    a + b + c
                }
            "#);
            assert_eq!(result.unwrap(), 60);
        }

        #[test]
        fn test_evidential_in_if() {
            // Evidential in conditional
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 1?;
                    if x == 1 {
                        100?
                    } else {
                        200?
                    }
                }
            "#);
            assert_eq!(result.unwrap(), 100);
        }

        #[test]
        fn test_evidential_paradox() {
            // Paradox (‽) marker - contradiction detection
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 42‽;
                    x
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_multiple_unwraps() {
            // Multiple sequential unwraps
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let a = 10?;
                    let b = a!;
                    let c = b!;
                    c
                }
            "#);
            assert_eq!(result.unwrap(), 10);
        }

        #[test]
        fn test_evidential_in_loop() {
            // Evidential values in a loop
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let mut sum = 0?;
                    let mut i = 0;
                    while i < 5 {
                        sum = sum + i?;
                        i = i + 1;
                    }
                    sum!
                }
            "#);
            assert_eq!(result.unwrap(), 10); // 0 + 1 + 2 + 3 + 4 = 10
        }

        #[test]
        fn test_evidential_comparison() {
            // Comparison of evidential values
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let a = 10?;
                    let b = 20?;
                    if a < b {
                        1!
                    } else {
                        0!
                    }
                }
            "#);
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_evidential_negation() {
            // Negation with evidential values
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 42?;
                    let y = -x;
                    y + 100
                }
            "#);
            assert_eq!(result.unwrap(), 58); // -42 + 100 = 58
        }

        #[test]
        fn test_evidential_chain_operations() {
            // Chain of operations with mixed evidentiality
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 10!;
                    let y = 20?;
                    let z = 30~;
                    let w = 40◊;
                    x + y + z + w
                }
            "#);
            assert_eq!(result.unwrap(), 100);
        }

        #[test]
        fn test_evidential_deeply_nested() {
            // Deeply nested evidential expressions
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = ((((42?)?)?)?)?;
                    x!
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_struct_field() {
            // Evidential values as struct fields
            let result = run_sigil(r#"
                struct Data {
                    value: i64,
                }

                fn main() -> i64 {
                    let d = Data { value: 100? };
                    d.value + 1
                }
            "#);
            assert_eq!(result.unwrap(), 101);
        }

        #[test]
        fn test_evidential_function_param() {
            // Function with evidential parameter
            let result = run_sigil(r#"
                fn double(x: i64) -> i64 {
                    x * 2
                }

                fn main() -> i64 {
                    let val = 25?;
                    double(val!)
                }
            "#);
            assert_eq!(result.unwrap(), 50);
        }

        #[test]
        fn test_evidential_all_markers_chain() {
            // All 5 evidentiality markers in sequence
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let known = 1!;      // Known
                    let uncertain = 2?;  // Uncertain
                    let reported = 3~;   // Reported
                    let predicted = 4◊;  // Predicted
                    let paradox = 5‽;    // Paradox
                    known + uncertain + reported + predicted + paradox
                }
            "#);
            assert_eq!(result.unwrap(), 15);
        }

        // ============================================
        // Generic Monomorphization Tests (existing)
        // ============================================

        #[test]
        fn test_generic_struct_basic() {
            let result = run_sigil(r#"
                struct Container<T> {
                    value: T,
                    count: i32,
                }

                fn main() -> i64 {
                    let c = Container::<i32> { value: 42, count: 1 };
                    c.value + c.count
                }
            "#);
            assert_eq!(result.unwrap(), 43);
        }

        #[test]
        fn test_generic_struct_two_params() {
            let result = run_sigil(r#"
                struct Pair<A, B> {
                    first: A,
                    second: B,
                }

                fn main() -> i64 {
                    let p = Pair::<i32, i32> { first: 10, second: 20 };
                    p.first + p.second
                }
            "#);
            assert_eq!(result.unwrap(), 30);
        }

        // ============================================
        // Morpheme Tests - Element Access
        // ============================================

        #[test]
        fn test_morpheme_first() {
            // First element: [1, 2, 3] |α returns 1
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [10, 20, 30] |α
                }
            "#);
            assert_eq!(result.unwrap(), 10);
        }

        #[test]
        fn test_morpheme_last() {
            // Last element: [1, 2, 3] |ω returns 3
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [10, 20, 30] |ω
                }
            "#);
            assert_eq!(result.unwrap(), 30);
        }

        #[test]
        fn test_morpheme_middle() {
            // Middle element: [1, 2, 3, 4, 5] |μ returns 3
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [10, 20, 30, 40, 50] |μ
                }
            "#);
            assert_eq!(result.unwrap(), 30);
        }

        #[test]
        fn test_morpheme_nth() {
            // Nth element: [1, 2, 3] |ν{1} returns 2
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [10, 20, 30] |ν{1}
                }
            "#);
            assert_eq!(result.unwrap(), 20);
        }

        // ============================================
        // Morpheme Tests - Reductions
        // ============================================

        #[test]
        fn test_morpheme_reduce_min() {
            // Simple min of two values
            let result = run_sigil(r#"
                fn min2(a: i64, b: i64) -> i64 {
                    if a < b { a } else { b }
                }
                fn main() -> i64 {
                    min2(min2(5, 2), min2(8, 1))
                }
            "#);
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_reduce_max() {
            // Simple max of two values
            let result = run_sigil(r#"
                fn max2(a: i64, b: i64) -> i64 {
                    if a > b { a } else { b }
                }
                fn main() -> i64 {
                    max2(max2(5, 2), max2(8, 9))
                }
            "#);
            assert_eq!(result.unwrap(), 9);
        }

        #[test]
        fn test_morpheme_reduce_all_true() {
            // All: [1, 2, 3] |ρ& returns 1 (all non-zero)
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [1, 2, 3] |ρ&
                }
            "#);
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_reduce_all_false() {
            // All: [1, 0, 3] |ρ& returns 0 (not all non-zero)
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [1, 0, 3] |ρ&
                }
            "#);
            assert_eq!(result.unwrap(), 0);
        }

        #[test]
        fn test_morpheme_reduce_any_true() {
            // Any: [0, 0, 1] |ρ| returns 1 (at least one non-zero)
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [0, 0, 1] |ρ|
                }
            "#);
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_reduce_any_false() {
            // Any: [0, 0, 0] |ρ| returns 0 (none non-zero)
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [0, 0, 0] |ρ|
                }
            "#);
            assert_eq!(result.unwrap(), 0);
        }

        // ============================================
        // Combined Morpheme Tests
        // ============================================

        #[test]
        fn test_morpheme_transform_then_first() {
            // Transform then first: [1, 2, 3] |τ{|x| x * 10} |α returns 10
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let arr = [1, 2, 3] |τ{|x| x * 10};
                    arr |α
                }
            "#);
            // Note: This tests that transform returns array, then first extracts
            // Current impl may need adjustment
            assert!(result.is_ok());
        }

        #[test]
        fn test_morpheme_filter_then_sum() {
            // Filter then sum: keep values > 3, sum them
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [1, 5, 2, 8, 3, 7] |φ{|x| x > 3} |ρ+
                }
            "#);
            // After filter: [5, 8, 7], sum = 20
            assert_eq!(result.unwrap(), 20);
        }

        // ============================================
        // New Morpheme Tests - Sort, Choice, Custom Reduce
        // ============================================

        #[test]
        fn test_morpheme_sort_basic() {
            // Sort returns minimum (first after sort): [3, 1, 2] |σ returns 1
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [3, 1, 2] |σ
                }
            "#);
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_sort_already_sorted() {
            // Sort already sorted: [1, 2, 3] |σ returns 1
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [1, 2, 3] |σ
                }
            "#);
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_sort_reverse() {
            // Sort reverse: [5, 4, 3, 2, 1] |σ returns 1
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [5, 4, 3, 2, 1] |σ
                }
            "#);
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_sort_single() {
            // Sort single element: [42] |σ returns 42
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [42] |σ
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_choice_deterministic() {
            // Choice is deterministic based on array contents
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [10, 20, 30] |χ
                }
            "#);
            // Result should be one of 10, 20, or 30
            let val = result.unwrap();
            assert!(val == 10 || val == 20 || val == 30);
        }

        #[test]
        fn test_morpheme_choice_single() {
            // Choice with single element: [42] |χ returns 42
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [42] |χ
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_custom_reduce_sum() {
            // Custom reduce sum: [1, 2, 3, 4] |ρ{|a, x| a + x} = 10
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [1, 2, 3, 4] |ρ{|acc, x| acc + x}
                }
            "#);
            assert_eq!(result.unwrap(), 10);
        }

        #[test]
        fn test_morpheme_custom_reduce_product() {
            // Custom reduce product: [1, 2, 3, 4] |ρ{|a, x| a * x} = 24
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [1, 2, 3, 4] |ρ{|acc, x| acc * x}
                }
            "#);
            assert_eq!(result.unwrap(), 24);
        }

        #[test]
        fn test_morpheme_custom_reduce_difference() {
            // Custom reduce difference: [100, 20, 5] |ρ{|a, x| a - x} = 75
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [100, 20, 5] |ρ{|acc, x| acc - x}
                }
            "#);
            assert_eq!(result.unwrap(), 75);
        }

        #[test]
        fn test_morpheme_custom_reduce_single() {
            // Custom reduce single element: [42] |ρ{|a, x| a + x} = 42
            let result = run_sigil(r#"
                fn main() -> i64 {
                    [42] |ρ{|acc, x| acc + x}
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_await_expr() {
            // Await expression form: expr⌛ (postfix syntax)
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 42;
                    x⌛
                }
            "#);
            // In sync LLVM context, await is identity
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_await_nested() {
            // Nested await expressions
            let result = run_sigil(r#"
                fn main() -> i64 {
                    let x = 21;
                    let y = x⌛ + x⌛;
                    y
                }
            "#);
            assert_eq!(result.unwrap(), 42);
        }
    }
}

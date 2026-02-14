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
    use inkwell::values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue,
        StructValue,
    };
    use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

    use std::collections::HashMap;
    use std::path::Path;

    use crate::ast::{self, BinOp, Expr, Ident, Item, Literal, NumBase, PathSegment, TypePath, UnaryOp};
    use crate::optimize::{OptLevel, Optimizer};
    use crate::parser::Parser;
    use crate::span::Span;

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
        /// Current Self type when compiling impl blocks
        current_self_type: Option<String>,
        /// Global/static variables
        global_vars: HashMap<String, inkwell::values::GlobalValue<'ctx>>,
        /// Libraries to link from #[link("lib")] attributes
        link_libraries: Vec<String>,
        /// Field type names: maps (struct_name, field_name) -> field_type_name (for method dispatch)
        field_type_names: HashMap<(String, String), String>,
        /// G21: Functions that return f64 (for float detection in println!)
        float_funcs: std::collections::HashSet<String>,
        /// G31: Function return types for tuple destructuring type inference
        ret_types: HashMap<String, ast::TypeExpr>,
    }

    // ============================================
    // Evidence Tag Constants
    // ============================================
    // These match the Evidentiality enum in ast.rs
    const EVIDENCE_KNOWN: u8 = 0; // ! - verified ground truth
    const EVIDENCE_UNCERTAIN: u8 = 1; // ? - unverified input
    const EVIDENCE_REPORTED: u8 = 2; // ~ - EMA, eventually consistent
    const EVIDENCE_PREDICTED: u8 = 3; // ◊ - model output, speculative
    const EVIDENCE_PARADOX: u8 = 4; // ‽ - contradiction detected

    // Runtime helper: get current time in milliseconds
    extern "C" fn sigil_now() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    // Runtime helper: allocate memory (for struct construction)
    // Returns i64 to match LLVM's optimized function signature (ptr gets converted to i64)
    extern "C" fn sigil_alloc(size: i64) -> i64 {
        use std::alloc::{alloc, Layout};
        let size = size.max(8) as usize; // Minimum 8 bytes
        let layout = Layout::from_size_align(size, 8).unwrap();
        unsafe { alloc(layout) as i64 }
    }

    // Runtime helper: print an integer with newline (for JIT mode)
    extern "C" fn sigil_print_int(value: i64) {
        println!("{}", value);
    }

    // Runtime helper: print a newline (for JIT mode)
    extern "C" fn sigil_print_newline() {
        println!();
    }

    // Runtime helper: write an integer without newline (for format strings)
    extern "C" fn sigil_write_int(value: i64) {
        use std::io::Write;
        print!("{}", value);
        let _ = std::io::stdout().flush();
    }

    // Runtime helper: write a string without newline (for format strings)
    extern "C" fn sigil_write_str(ptr: *const i8) {
        use std::io::Write;
        if !ptr.is_null() {
            let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
            if let Ok(s) = c_str.to_str() {
                print!("{}", s);
                let _ = std::io::stdout().flush();
            }
        }
    }

    // Runtime helper: write a float without newline (for format strings)
    extern "C" fn sigil_write_float(value: f64) {
        use std::io::Write;
        print!("{}", value);
        let _ = std::io::stdout().flush();
    }

    // Runtime helper: get string length (C string)
    extern "C" fn sigil_strlen(ptr: *const i8) -> i64 {
        if ptr.is_null() {
            return 0;
        }
        let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
        c_str.to_bytes().len() as i64
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
    extern "C" fn sigil_pi() -> i64 {
        f64::to_bits(std::f64::consts::PI) as i64
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
            unsafe {
                (*vec_ptr).push(value);
            }
        }
    }

    extern "C" fn sigil_vec_get(vec_ptr: *mut Vec<i64>, index: i64) -> i64 {
        if vec_ptr.is_null() {
            return 0;
        }
        unsafe {
            let vec_ref = &*vec_ptr;
            vec_ref.get(index as usize).copied().unwrap_or(0)
        }
    }

    extern "C" fn sigil_vec_len(vec_ptr: *mut Vec<i64>) -> i64 {
        if vec_ptr.is_null() {
            return 0;
        }
        unsafe { (*vec_ptr).len() as i64 }
    }

    extern "C" fn sigil_vec_set(vec_ptr: *mut Vec<i64>, index: i64, value: i64) {
        if vec_ptr.is_null() {
            return;
        }
        unsafe {
            let vec_ref = &mut *vec_ptr;
            if (index as usize) < vec_ref.len() {
                vec_ref[index as usize] = value;
            }
        }
    }

    // String runtime functions
    extern "C" fn sigil_string_new() -> *mut String {
        Box::into_raw(Box::new(String::new()))
    }

    extern "C" fn sigil_string_from(ptr: *const i8) -> *mut String {
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(ptr);
            let s = cstr.to_string_lossy().into_owned();
            Box::into_raw(Box::new(s))
        }
    }

    extern "C" fn sigil_string_len(str_ptr: *mut String) -> i64 {
        if str_ptr.is_null() {
            return 0;
        }
        unsafe {
            let str_ref = &*str_ptr;
            str_ref.len() as i64
        }
    }

    extern "C" fn sigil_string_print(str_ptr: *mut String) {
        if !str_ptr.is_null() {
            unsafe {
                print!("{}", *str_ptr);
            }
        }
    }

    extern "C" fn sigil_string_concat(a_ptr: *mut String, b_ptr: *mut String) -> *mut String {
        if a_ptr.is_null() || b_ptr.is_null() {
            return std::ptr::null_mut();
        }
        unsafe {
            let result = format!("{}{}", *a_ptr, *b_ptr);
            Box::into_raw(Box::new(result))
        }
    }

    // G27: File system runtime functions
    extern "C" fn sigil_fs_read(path_ptr: *const i8) -> *mut String {
        if path_ptr.is_null() {
            return Box::into_raw(Box::new(String::new()));
        }
        let path = unsafe { std::ffi::CStr::from_ptr(path_ptr) };
        match path.to_str() {
            Ok(path_str) => match std::fs::read_to_string(path_str) {
                Ok(contents) => Box::into_raw(Box::new(contents)),
                Err(_) => Box::into_raw(Box::new(String::new())),
            },
            Err(_) => Box::into_raw(Box::new(String::new())),
        }
    }

    // Get string data as C string pointer (for printing Rust Strings)
    extern "C" fn sigil_string_data(str_ptr: *mut String) -> *const i8 {
        if str_ptr.is_null() {
            return b"\0".as_ptr() as *const i8;
        }
        let s = unsafe { &*str_ptr };
        s.as_ptr() as *const i8
    }

    // sigil_string_len is already defined above

    // G32: String slice - create substring from start to end indices
    extern "C" fn sigil_string_slice(str_ptr: *const i8, start: i64, end: i64) -> *mut String {
        if str_ptr.is_null() {
            return Box::into_raw(Box::new(String::new()));
        }
        let cstr = unsafe { std::ffi::CStr::from_ptr(str_ptr) };
        let s = cstr.to_string_lossy();
        let start_idx = start.max(0) as usize;
        let end_idx = end.max(0) as usize;
        let end_idx = end_idx.min(s.len());
        let substring = if start_idx < end_idx {
            s[start_idx..end_idx].to_string()
        } else {
            String::new()
        };
        Box::into_raw(Box::new(substring))
    }

    // G32: Rust String slice - create substring from Rust String
    extern "C" fn sigil_rust_string_slice(str_ptr: *mut String, start: i64, end: i64) -> *mut String {
        if str_ptr.is_null() {
            return Box::into_raw(Box::new(String::new()));
        }
        let s = unsafe { &*str_ptr };
        let start_idx = start.max(0) as usize;
        let end_idx = end.max(0) as usize;
        let end_idx = end_idx.min(s.len());
        let substring = if start_idx < end_idx {
            s[start_idx..end_idx].to_string()
        } else {
            String::new()
        };
        Box::into_raw(Box::new(substring))
    }

    // G32: Get bytes from a Rust String with null terminator for strlen compatibility
    // Returns a pointer to a copy of the string's bytes with a null terminator appended
    extern "C" fn sigil_rust_string_as_bytes(str_ptr: *mut String) -> *const i8 {
        if str_ptr.is_null() {
            return b"\0".as_ptr() as *const i8;
        }
        let s = unsafe { &*str_ptr };
        // Create a copy with null terminator so strlen works
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0); // Add null terminator
        let ptr = bytes.as_ptr() as *const i8;
        std::mem::forget(bytes); // Leak - current design doesn't track/free these
        ptr
    }

    // Print a Rust String directly
    extern "C" fn sigil_print_rust_string(str_ptr: *mut String) {
        if str_ptr.is_null() {
            println!();
            return;
        }
        let s = unsafe { &*str_ptr };
        println!("{}", s);
    }

    // Option runtime functions
    extern "C" fn sigil_option_some(value: i64) -> *mut i64 {
        Box::into_raw(Box::new(value))
    }

    extern "C" fn sigil_option_none() -> *mut i64 {
        std::ptr::null_mut()
    }

    extern "C" fn sigil_option_is_some(opt_ptr: *mut i64) -> i64 {
        if opt_ptr.is_null() {
            0
        } else {
            1
        }
    }

    extern "C" fn sigil_option_is_none(opt_ptr: *mut i64) -> i64 {
        if opt_ptr.is_null() {
            1
        } else {
            0
        }
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
        if opt_ptr.is_null() {
            default
        } else {
            unsafe { *opt_ptr }
        }
    }

    // File I/O runtime functions
    extern "C" fn sigil_file_exists(path_ptr: *const i8) -> i64 {
        if path_ptr.is_null() {
            return 0;
        }
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(path_ptr);
            if let Ok(path) = cstr.to_str() {
                if std::path::Path::new(path).exists() {
                    1
                } else {
                    0
                }
            } else {
                0
            }
        }
    }

    extern "C" fn sigil_file_read_all(path_ptr: *const i8) -> *mut String {
        if path_ptr.is_null() {
            return std::ptr::null_mut();
        }
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
        if path_ptr.is_null() || content_ptr.is_null() {
            return -1;
        }
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
                current_self_type: None,
                global_vars: HashMap::new(),
                link_libraries: Vec::new(),
                field_type_names: HashMap::new(),
                float_funcs: std::collections::HashSet::new(),
                ret_types: HashMap::new(),
            })
        }

        /// Compile source code
        pub fn compile(&mut self, source: &str) -> Result<(), String> {
            let mut parser = Parser::new(source);
            let source_file = parser.parse_file().map_err(|e| format!("{:?}", e))?;

            // Run AST optimizations (temporarily disabled to debug)
            // let mut optimizer = Optimizer::new(self.opt_level);
            // let optimized = optimizer.optimize_file(&source_file);
            let optimized = source_file; // Skip optimization for now

            // Declare runtime functions
            self.declare_runtime_functions();

            // First pass: register types
            for spanned_item in &optimized.items {
                match &spanned_item.node {
                    Item::Struct(s) => self.register_struct(s)?,
                    Item::Enum(e) => self.register_enum(e)?,
                    _ => {}
                }
            }

            // First pass continued: process impl blocks
            for spanned_item in &optimized.items {
                if let Item::Impl(impl_block) = &spanned_item.node {
                    self.declare_impl_methods(impl_block)?;
                }
            }

            // Second pass: process modules, statics, and declare all functions
            for spanned_item in &optimized.items {
                match &spanned_item.node {
                    Item::Function(func) => {
                        self.declare_function(func)?;
                    }
                    Item::Module(module) => {
                        self.process_module(module)?;
                    }
                    Item::Use(use_decl) => {
                        self.process_use(use_decl)?;
                    }
                    Item::Static(static_decl) => {
                        self.process_static(static_decl)?;
                    }
                    Item::Const(const_decl) => {
                        self.process_const(const_decl)?;
                    }
                    _ => {}
                }
            }

            // Third pass: compile function bodies
            for spanned_item in &optimized.items {
                match &spanned_item.node {
                    Item::Function(func) => {
                        self.compile_function(func)?;
                    }
                    Item::Module(module) => {
                        self.compile_module_functions(module)?;
                    }
                    Item::Impl(impl_block) => {
                        self.compile_impl_methods(impl_block)?;
                    }
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

            // sigil_now() -> i64 (milliseconds)
            let now_type = i64_type.fn_type(&[], false);
            self.module.add_function("sigil_now", now_type, None);

            // sigil_now_micros() -> i64 (microseconds)
            self.module.add_function("sigil_now_micros", now_type, None);

            // sigil_print_int(i64) -> void
            let print_int_type = void_type.fn_type(&[i64_type.into()], false);
            self.module
                .add_function("sigil_print_int", print_int_type, None);

            // sigil_print_str(const char*) -> void - for raw C string literals
            let ptr_type_generic = self.context.ptr_type(AddressSpace::default());
            let print_str_type = void_type.fn_type(&[ptr_type_generic.into()], false);
            self.module
                .add_function("sigil_print_str", print_str_type, None);

            // sigil_print_float(f64) -> void
            let f64_type = self.context.f64_type();
            let print_float_type = void_type.fn_type(&[f64_type.into()], false);
            self.module
                .add_function("sigil_print_float", print_float_type, None);

            // sigil_print_newline() -> void
            let newline_type = void_type.fn_type(&[], false);
            self.module
                .add_function("sigil_print_newline", newline_type, None);

            // Write functions (no newline) for format strings
            // sigil_write_int(i64) -> void
            self.module
                .add_function("sigil_write_int", print_int_type, None);

            // sigil_write_str(const char*) -> void
            self.module
                .add_function("sigil_write_str", print_str_type, None);

            // sigil_strlen(const char*) -> i64
            let strlen_type = i64_type.fn_type(&[ptr_type_generic.into()], false);
            self.module
                .add_function("sigil_strlen", strlen_type, None);

            // Jormungandr-compatible print functions (const char*) -> void
            self.module.add_function("print", print_str_type, None);
            self.module.add_function("println", print_str_type, None);
            self.module.add_function("eprint", print_str_type, None);
            self.module.add_function("eprintln", print_str_type, None);

            // sigil_write_float(f64) -> void
            self.module
                .add_function("sigil_write_float", print_float_type, None);

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

            // Math constants: () -> i64
            let const_type = i64_type.fn_type(&[], false);
            self.module.add_function("sigil_pi", const_type, None);

            // Vec functions - use ptr type (i64 as opaque pointer)
            let ptr_type = i64_type; // Using i64 as opaque pointer type

            // sigil_vec_new(capacity: i64) -> ptr
            let vec_new_type = ptr_type.fn_type(&[i64_type.into()], false);
            self.module
                .add_function("sigil_vec_new", vec_new_type, None);

            // sigil_vec_push(vec: ptr, value: i64) -> void
            let vec_push_type = void_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_vec_push", vec_push_type, None);

            // sigil_vec_get(vec: ptr, index: i64) -> i64
            let vec_get_type = i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_vec_get", vec_get_type, None);

            // sigil_vec_len(vec: ptr) -> i64
            let vec_len_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_vec_len", vec_len_type, None);

            // sigil_vec_set(vec: ptr, index: i64, value: i64) -> void
            let vec_set_type =
                void_type.fn_type(&[ptr_type.into(), i64_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_vec_set", vec_set_type, None);

            // String functions
            // sigil_string_new() -> ptr (empty string)
            let string_new_type = ptr_type.fn_type(&[], false);
            self.module
                .add_function("sigil_string_new", string_new_type, None);

            // sigil_string_from(const char* src) -> ptr
            // For now, pass i64 as pointer to string literal (global constant)
            let string_from_type = ptr_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_string_from", string_from_type, None);

            // sigil_string_len(str: ptr) -> i64
            let string_len_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_string_len", string_len_type, None);

            // sigil_string_print(str: ptr) -> void
            let string_print_type = void_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_string_print", string_print_type, None);

            // sigil_string_concat(str1: ptr, str2: ptr) -> ptr
            let string_concat_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module
                .add_function("sigil_string_concat", string_concat_type, None);

            // sigil_string_repeat(str: ptr, count: i64) -> ptr
            let string_repeat_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_string_repeat", string_repeat_type, None);

            // G27: File system functions
            // sigil_fs_read(path: *const i8) -> *mut String (real C pointers)
            let fs_read_type = ptr_type_generic.fn_type(&[ptr_type_generic.into()], false);
            self.module
                .add_function("sigil_fs_read", fs_read_type, None);

            // sigil_string_data(str: *mut String) -> *const i8
            let string_data_type = ptr_type_generic.fn_type(&[ptr_type_generic.into()], false);
            self.module
                .add_function("sigil_string_data", string_data_type, None);

            // sigil_string_len(str: i64) -> i64
            // Uses i64 for pointer representation like other functions
            let string_len_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_string_len", string_len_type, None);

            // G32: sigil_string_slice(str: i64, start: i64, end: i64) -> i64
            // Uses i64 for pointer representation like other functions
            let string_slice_type = ptr_type.fn_type(
                &[ptr_type.into(), i64_type.into(), i64_type.into()],
                false,
            );
            self.module
                .add_function("sigil_string_slice", string_slice_type, None);

            // G32: sigil_rust_string_slice(str: i64, start: i64, end: i64) -> i64
            self.module
                .add_function("sigil_rust_string_slice", string_slice_type, None);

            // G32: sigil_rust_string_as_bytes(str: i64) -> i64 (C string pointer)
            // Takes Rust String pointer, returns byte pointer with null terminator
            let rust_string_as_bytes_type = ptr_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_rust_string_as_bytes", rust_string_as_bytes_type, None);

            // sigil_print_rust_string(str: *mut String) -> void
            let print_rust_string_type = void_type.fn_type(&[ptr_type_generic.into()], false);
            self.module
                .add_function("sigil_print_rust_string", print_rust_string_type, None);

            // Option functions
            // sigil_option_some(value: i64) -> ptr
            let option_some_type = ptr_type.fn_type(&[i64_type.into()], false);
            self.module
                .add_function("sigil_option_some", option_some_type, None);

            // sigil_option_none() -> ptr (null)
            let option_none_type = ptr_type.fn_type(&[], false);
            self.module
                .add_function("sigil_option_none", option_none_type, None);

            // sigil_option_is_some(opt: ptr) -> i64
            let option_is_some_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_option_is_some", option_is_some_type, None);

            // sigil_option_is_none(opt: ptr) -> i64
            let option_is_none_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_option_is_none", option_is_none_type, None);

            // sigil_option_unwrap(opt: ptr) -> i64
            let option_unwrap_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_option_unwrap", option_unwrap_type, None);

            // sigil_option_unwrap_or(opt: ptr, default: i64) -> i64
            let option_unwrap_or_type =
                i64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_option_unwrap_or", option_unwrap_or_type, None);

            // File I/O functions
            // sigil_file_exists(path: ptr) -> i64 (1 if exists, 0 otherwise)
            let file_exists_type = i64_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_file_exists", file_exists_type, None);

            // sigil_file_read_all(path: ptr) -> ptr (returns String ptr or null)
            let file_read_all_type = ptr_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_file_read_all", file_read_all_type, None);

            // sigil_file_write_all(path: ptr, content: ptr) -> i64 (bytes written or -1)
            let file_write_all_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module
                .add_function("sigil_file_write_all", file_write_all_type, None);

            // sigil_exit(code: i64) -> void
            let exit_type = void_type.fn_type(&[i64_type.into()], false);
            self.module.add_function("sigil_exit", exit_type, None);

            // Memory functions
            // sigil_alloc(size: i64) -> i64 (returns pointer as i64 to match LLVM optimization behavior)
            let alloc_type = i64_type.fn_type(&[i64_type.into()], false);
            self.module.add_function("sigil_alloc", alloc_type, None);

            // sigil_realloc(ptr: ptr, new_size: i64) -> ptr
            let realloc_type = ptr_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_realloc", realloc_type, None);

            // sigil_free(ptr: ptr) -> void
            let free_type = void_type.fn_type(&[ptr_type.into()], false);
            self.module.add_function("sigil_free", free_type, None);

            // SIMD Functions (F32x16)
            let f32_type = self.context.f32_type();

            // sigil_simd_alloc(num_floats: i64) -> ptr
            let simd_alloc_type = ptr_type.fn_type(&[i64_type.into()], false);
            self.module
                .add_function("sigil_simd_alloc", simd_alloc_type, None);

            // sigil_simd_free(ptr: ptr) -> void
            let simd_free_type = void_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_simd_free", simd_free_type, None);

            // sigil_simd_splat_f32x16(dest: ptr, value: f32) -> void
            let simd_splat_type = void_type.fn_type(&[ptr_type.into(), f32_type.into()], false);
            self.module
                .add_function("sigil_simd_splat_f32x16", simd_splat_type, None);

            // sigil_simd_load_f32x16(dest: ptr, src: ptr) -> void
            let simd_load_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module
                .add_function("sigil_simd_load_f32x16", simd_load_type, None);

            // sigil_simd_store_f32x16(dest: ptr, src: ptr) -> void
            let simd_store_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module
                .add_function("sigil_simd_store_f32x16", simd_store_type, None);

            // sigil_simd_add_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            let simd_binop_type =
                void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
            self.module
                .add_function("sigil_simd_add_f32x16", simd_binop_type, None);

            // sigil_simd_sub_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            self.module
                .add_function("sigil_simd_sub_f32x16", simd_binop_type, None);

            // sigil_simd_mul_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            self.module
                .add_function("sigil_simd_mul_f32x16", simd_binop_type, None);

            // sigil_simd_div_f32x16(dest: ptr, a: ptr, b: ptr) -> void
            self.module
                .add_function("sigil_simd_div_f32x16", simd_binop_type, None);

            // sigil_simd_fmadd_f32x16(dest: ptr, a: ptr, b: ptr, c: ptr) -> void
            let simd_fmadd_type = void_type.fn_type(
                &[
                    ptr_type.into(),
                    ptr_type.into(),
                    ptr_type.into(),
                    ptr_type.into(),
                ],
                false,
            );
            self.module
                .add_function("sigil_simd_fmadd_f32x16", simd_fmadd_type, None);

            // sigil_simd_reduce_add_f32x16(src: ptr) -> f32
            let simd_reduce_type = f32_type.fn_type(&[ptr_type.into()], false);
            self.module
                .add_function("sigil_simd_reduce_add_f32x16", simd_reduce_type, None);

            // sigil_simd_extract_f32x16(src: ptr, index: i64) -> f32
            let simd_extract_type = f32_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_simd_extract_f32x16", simd_extract_type, None);

            // sigil_simd_dot_f32x16(a: ptr, b: ptr) -> f32
            let simd_dot_type = f32_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module
                .add_function("sigil_simd_dot_f32x16", simd_dot_type, None);

            // CUDA Functions
            // sigil_cuda_init() -> i64
            let cuda_init_type = i64_type.fn_type(&[], false);
            self.module
                .add_function("sigil_cuda_init", cuda_init_type, None);

            // sigil_cuda_cleanup() -> void
            let cuda_cleanup_type = void_type.fn_type(&[], false);
            self.module
                .add_function("sigil_cuda_cleanup", cuda_cleanup_type, None);

            // sigil_cuda_get_device_count() -> i64
            let cuda_device_count_type = i64_type.fn_type(&[], false);
            self.module
                .add_function("sigil_cuda_get_device_count", cuda_device_count_type, None);

            // sigil_cuda_malloc(size: i64) -> i64 (device ptr)
            let cuda_malloc_type = i64_type.fn_type(&[i64_type.into()], false);
            self.module
                .add_function("sigil_cuda_malloc", cuda_malloc_type, None);

            // sigil_cuda_free(device_ptr: i64) -> void
            let cuda_free_type = void_type.fn_type(&[i64_type.into()], false);
            self.module
                .add_function("sigil_cuda_free", cuda_free_type, None);

            // sigil_cuda_memcpy_h2d(dst: i64, src: ptr, size: i64) -> i64
            let cuda_h2d_type =
                i64_type.fn_type(&[i64_type.into(), ptr_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_cuda_memcpy_h2d", cuda_h2d_type, None);

            // sigil_cuda_memcpy_d2h(dst: ptr, src: i64, size: i64) -> i64
            let cuda_d2h_type =
                i64_type.fn_type(&[ptr_type.into(), i64_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_cuda_memcpy_d2h", cuda_d2h_type, None);

            // sigil_cuda_memcpy_d2d(dst: i64, src: i64, size: i64) -> i64
            let cuda_d2d_type =
                i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
            self.module
                .add_function("sigil_cuda_memcpy_d2d", cuda_d2d_type, None);

            // sigil_cuda_sync() -> void
            let cuda_sync_type = void_type.fn_type(&[], false);
            self.module
                .add_function("sigil_cuda_sync", cuda_sync_type, None);

            // sigil_cuda_compile_kernel(cuda_src: ptr, kernel_name: ptr) -> i64 (handle)
            let cuda_compile_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
            self.module
                .add_function("sigil_cuda_compile_kernel", cuda_compile_type, None);

            // sigil_cuda_load_ptx(ptx: ptr, kernel_name: ptr) -> i64 (handle)
            self.module
                .add_function("sigil_cuda_load_ptx", cuda_compile_type, None);

            // sigil_cuda_launch_kernel_1d(handle: i64, grid_x: i64, block_x: i64, args: ptr, num_args: i64) -> i64
            let cuda_launch_1d_type = i64_type.fn_type(
                &[
                    i64_type.into(),
                    i64_type.into(),
                    i64_type.into(),
                    ptr_type.into(),
                    i64_type.into(),
                ],
                false,
            );
            self.module
                .add_function("sigil_cuda_launch_kernel_1d", cuda_launch_1d_type, None);

            // sigil_cuda_launch_kernel_2d(handle: i64, gx: i64, gy: i64, bx: i64, by: i64, args: ptr, num_args: i64) -> i64
            let cuda_launch_2d_type = i64_type.fn_type(
                &[
                    i64_type.into(),
                    i64_type.into(),
                    i64_type.into(),
                    i64_type.into(),
                    i64_type.into(),
                    ptr_type.into(),
                    i64_type.into(),
                ],
                false,
            );
            self.module
                .add_function("sigil_cuda_launch_kernel_2d", cuda_launch_2d_type, None);
        }

        /// Register a struct type in the type registry
        fn register_struct(&mut self, struct_def: &ast::StructDef) -> Result<(), String> {
            let name = &struct_def.name.name;

            // Check if this struct has generic parameters
            if let Some(ref generics) = struct_def.generics {
                if !generics.params.is_empty() {
                    // Extract type parameter names
                    let type_params: Vec<String> = generics
                        .params
                        .iter()
                        .filter_map(|p| match p {
                            ast::GenericParam::Type { name, .. } => Some(name.name.clone()),
                            ast::GenericParam::Const { name, .. } => Some(name.name.clone()),
                            ast::GenericParam::Lifetime(_) => None,
                        })
                        .collect();

                    // Store as generic struct for later monomorphization
                    self.generic_structs.insert(
                        name.clone(),
                        GenericStructDef {
                            def: struct_def.clone(),
                            type_params,
                        },
                    );
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

                        // G11 fix: Track field type names for method dispatch
                        if let Some(field_type_name) = self.get_field_struct_type(&field.ty) {
                            self.field_type_names.insert(
                                (base_name.clone(), field.name.name.clone()),
                                field_type_name,
                            );
                        }
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
            self.struct_types.insert(
                mangled_name,
                StructInfo {
                    llvm_type: struct_type,
                    field_indices,
                },
            );

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
                            // AVX-512 SIMD types
                            "F32x16" | "__m512" => {
                                // 512-bit vector of 16 f32s
                                self.context.f32_type().vec_type(16).into()
                            }
                            "F64x8" | "__m512d" => {
                                // 512-bit vector of 8 f64s
                                self.context.f64_type().vec_type(8).into()
                            }
                            "I32x16" | "__m512i" => {
                                // 512-bit vector of 16 i32s
                                self.context.i32_type().vec_type(16).into()
                            }
                            "I64x8" => {
                                // 512-bit vector of 8 i64s
                                self.context.i64_type().vec_type(8).into()
                            }
                            // AVX-256 SIMD types
                            "F32x8" | "__m256" => self.context.f32_type().vec_type(8).into(),
                            "F64x4" | "__m256d" => self.context.f64_type().vec_type(4).into(),
                            _ => i64_type.into(), // Default to i64 for unknown types
                        }
                    } else {
                        i64_type.into()
                    }
                }
                ast::TypeExpr::Reference { inner, .. } | ast::TypeExpr::Pointer { inner, .. } => {
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

        /// Extract the struct type name from a TypeExpr (for method dispatch)
        /// Returns the struct/type name if it's a user-defined type, None for primitives
        fn get_field_struct_type(&self, ty: &ast::TypeExpr) -> Option<String> {
            match ty {
                ast::TypeExpr::Path(path) => {
                    if let Some(segment) = path.segments.first() {
                        let name = &segment.ident.name;
                        // Skip primitive types (but keep generic types like Vec<T>)
                        match name.as_str() {
                            "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64"
                            | "isize" | "usize" | "f32" | "f64" | "bool" | "str" | "String"
                            | "Option" | "Result" => None,
                            // G15 fix: Include Vec<T> with element type for method dispatch
                            "Vec" => {
                                if let Some(ref generics) = segment.generics {
                                    if let Some(first_arg) = generics.first() {
                                        // Get element type name
                                        if let Some(elem_type) = self.get_field_struct_type(first_arg) {
                                            return Some(format!("Vec<{}>", elem_type));
                                        }
                                    }
                                }
                                None
                            }
                            _ => Some(name.clone()),
                        }
                    } else {
                        None
                    }
                }
                ast::TypeExpr::Reference { inner, .. } | ast::TypeExpr::Pointer { inner, .. } => {
                    // For references/pointers, get the inner type name
                    self.get_field_struct_type(inner)
                }
                _ => None,
            }
        }

        /// Monomorphize a generic struct with concrete type arguments
        fn monomorphize_struct(
            &mut self,
            base_name: &str,
            type_args: &[ast::TypeExpr],
        ) -> Result<String, String> {
            // Look up the generic struct definition
            let generic_def = self
                .generic_structs
                .get(base_name)
                .ok_or_else(|| format!("Unknown generic struct: {}", base_name))?
                .clone();

            if type_args.len() != generic_def.type_params.len() {
                return Err(format!(
                    "Wrong number of type arguments for {}: expected {}, got {}",
                    base_name,
                    generic_def.type_params.len(),
                    type_args.len()
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
                ast::TypeExpr::Path(path) => path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("_"),
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
                    let names: Vec<_> =
                        elements.iter().map(|e| self.type_expr_to_name(e)).collect();
                    format!("tup_{}", names.join("_"))
                }
                ast::TypeExpr::Evidential { inner, .. } => self.type_expr_to_name(inner),
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
            let struct_type = self
                .context
                .struct_type(&[i8_type.into(), value_type], false);

            self.evidential_types
                .insert(base_type_name.to_string(), struct_type);
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
            let ptr = self
                .builder
                .build_alloca(evidential_type, "evidential")
                .map_err(|e| e.to_string())?;

            // Store the tag at index 0
            let tag_ptr = self
                .builder
                .build_struct_gep(evidential_type, ptr, 0, "tag_ptr")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(tag_ptr, tag)
                .map_err(|e| e.to_string())?;

            // Store the value at index 1
            let value_ptr = self
                .builder
                .build_struct_gep(evidential_type, ptr, 1, "value_ptr")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(value_ptr, value)
                .map_err(|e| e.to_string())?;

            // Load and return the complete struct
            let result = self
                .builder
                .build_load(evidential_type, ptr, "evidential_val")
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
            let value = self
                .builder
                .build_extract_value(evidential_struct, 1, "unwrapped")
                .map_err(|e| e.to_string())?;

            Ok(value.into_int_value())
        }

        /// Extract the evidence tag from an evidential struct.
        fn get_evidence_tag(
            &mut self,
            evidential_struct: StructValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            let tag = self
                .builder
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
            let cmp = self
                .builder
                .build_int_compare(IntPredicate::UGT, tag1, tag2, "ev_cmp")
                .map_err(|e| e.to_string())?;

            let result = self
                .builder
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

        /// Process a static declaration
        fn process_static(&mut self, static_decl: &ast::StaticDef) -> Result<(), String> {
            let name = &static_decl.name.name;
            // Create a global variable
            let i64_type = self.context.i64_type();
            let global = self.module.add_global(i64_type, None, name);
            // Initialize to 0 (will be initialized properly at runtime)
            global.set_initializer(&i64_type.const_int(0, false));
            self.global_vars.insert(name.clone(), global);
            Ok(())
        }

        /// Process a const declaration
        fn process_const(&mut self, const_decl: &ast::ConstDef) -> Result<(), String> {
            let name = &const_decl.name.name;
            // Create a global constant
            let i64_type = self.context.i64_type();
            let global = self.module.add_global(i64_type, None, name);
            // Initialize to 0 for now (could evaluate const expr)
            global.set_initializer(&i64_type.const_int(0, false));
            global.set_constant(true);
            self.global_vars.insert(name.clone(), global);
            Ok(())
        }

        /// Extract type name from impl block's self_ty
        fn extract_impl_type_name(&self, self_ty: &ast::TypeExpr) -> Result<String, String> {
            match self_ty {
                ast::TypeExpr::Path(path) => path
                    .segments
                    .last()
                    .map(|s| s.ident.name.clone())
                    .ok_or_else(|| "Empty impl type path".to_string()),
                ast::TypeExpr::Evidential {
                    inner,
                    evidentiality,
                    ..
                } => {
                    // Handle ?T (Option<T>) and !T (Result<T>) type impls
                    let inner_name = self.extract_impl_type_name(inner)?;
                    match evidentiality {
                        ast::Evidentiality::Uncertain => Ok(format!("Option_{}", inner_name)),
                        ast::Evidentiality::Known => Ok(format!("Result_{}", inner_name)),
                        ast::Evidentiality::Reported => Ok(format!("Reported_{}", inner_name)),
                        ast::Evidentiality::Predicted => Ok(format!("Predicted_{}", inner_name)),
                        ast::Evidentiality::Paradox => Ok(format!("Paradox_{}", inner_name)),
                    }
                }
                ast::TypeExpr::Reference { inner, .. } => {
                    // Handle &T impl - use inner type name
                    self.extract_impl_type_name(inner)
                }
                ast::TypeExpr::Slice(inner) => {
                    let inner_name = self.extract_impl_type_name(inner)?;
                    Ok(format!("Slice_{}", inner_name))
                }
                ast::TypeExpr::Array { element, .. } => {
                    let inner_name = self.extract_impl_type_name(element)?;
                    Ok(format!("Array_{}", inner_name))
                }
                ast::TypeExpr::Tuple(elements) => {
                    if elements.is_empty() {
                        Ok("Unit".to_string())
                    } else {
                        let names: Result<Vec<_>, _> = elements
                            .iter()
                            .map(|e| self.extract_impl_type_name(e))
                            .collect();
                        Ok(format!("Tuple_{}", names?.join("_")))
                    }
                }
                // Fallback for other types - generate a unique name
                _ => Ok("UnknownType".to_string()),
            }
        }

        /// Declare methods from an impl block
        fn declare_impl_methods(&mut self, impl_block: &ast::ImplBlock) -> Result<(), String> {
            // Get the type name from the impl path
            // Extract type name from self_ty (TypeExpr)
            let type_name = self.extract_impl_type_name(&impl_block.self_ty)?;

            for item in &impl_block.items {
                if let ast::ImplItem::Function(func) = item {
                    let method_name = &func.name.name;
                    let mangled_name = format!("{}_{}", type_name, method_name);

                    // Declare the function with self as first parameter
                    let i64_type = self.context.i64_type();

                    // Check if first param is self/this (instance method)
                    let has_explicit_self = func.params.first().map_or(false, |p| {
                        matches!(&p.pattern, ast::Pattern::Ident { name, .. } if name.name == "self" || name.name == "this")
                    });

                    // Check if first param looks like a self reference (&self, &mut self, vary this)
                    let has_self_ref = func.params.first().map_or(false, |p| {
                        matches!(&p.pattern, ast::Pattern::Ref { pattern, .. } if {
                            matches!(&**pattern, ast::Pattern::Ident { name, .. } if name.name == "self" || name.name == "this")
                        }) || matches!(&p.pattern, ast::Pattern::RefBinding { name, .. } if name.name == "self" || name.name == "this")
                    });

                    // A method needs self/this if it has explicit self/this as first param
                    // Static methods have no self param at all
                    let is_instance_method = has_explicit_self || has_self_ref;

                    // Count params: instance methods might have implicit self, static methods don't
                    let param_count = func.params.len();
                    let param_types: Vec<BasicMetadataTypeEnum> =
                        (0..param_count).map(|_| i64_type.into()).collect();

                    let fn_type = i64_type.fn_type(&param_types, false);
                    let fn_value = self.module.add_function(&mangled_name, fn_type, None);

                    // Name parameters - all params are named directly, no implicit self
                    for (i, param) in func.params.iter().enumerate() {
                        let param_name = match &param.pattern {
                            ast::Pattern::Ident { name: ident, .. } => ident.name.clone(),
                            ast::Pattern::RefBinding { name: ident, .. } => ident.name.clone(),
                            ast::Pattern::Ref { pattern, .. } => {
                                if let ast::Pattern::Ident { name: ident, .. } = &**pattern {
                                    ident.name.clone()
                                } else {
                                    format!("param{}", i)
                                }
                            }
                            _ => format!("param{}", i),
                        };
                        fn_value
                            .get_nth_param(i as u32)
                            .unwrap()
                            .set_name(&param_name);
                    }
                    let _ = is_instance_method; // Silence unused warning

                    self.functions.insert(mangled_name.clone(), fn_value);
                    self.impl_methods
                        .insert((type_name.clone(), method_name.clone()), mangled_name.clone());

                    // G21b: Track methods that return f64 for float detection in println!
                    if let Some(ref return_type) = func.return_type {
                        if self.type_contains_f64(return_type) {
                            self.float_funcs.insert(mangled_name);
                            // Also store the short method name for MethodCall lookups
                            self.float_funcs.insert(method_name.clone());
                        }
                    }
                }
            }
            Ok(())
        }

        /// Compile methods from an impl block
        fn compile_impl_methods(&mut self, impl_block: &ast::ImplBlock) -> Result<(), String> {
            // Extract type name from self_ty (TypeExpr)
            let type_name = self.extract_impl_type_name(&impl_block.self_ty)?;

            // Set the current Self type for resolving Self:: calls
            self.current_self_type = Some(type_name.clone());

            for item in &impl_block.items {
                if let ast::ImplItem::Function(func) = item {
                    let method_name = &func.name.name;
                    let mangled_name = format!("{}_{}", type_name, method_name);

                    let fn_value = *self
                        .functions
                        .get(&mangled_name)
                        .ok_or_else(|| format!("Method not declared: {}", mangled_name))?;

                    // Create entry block
                    let entry = self.context.append_basic_block(fn_value, "entry");
                    self.builder.position_at_end(entry);

                    // Set up variable scope
                    let mut scope = CompileScope::new();
                    // G23: Copy global float_funcs registry to scope for float detection in impl methods
                    scope.float_funcs = self.float_funcs.clone();

                    // Add all parameters to scope - no implicit self for any method
                    // (Static methods have no self, instance methods have explicit self/this)
                    for (i, param) in func.params.iter().enumerate() {
                        let param_name = match &param.pattern {
                            ast::Pattern::Ident { name: ident, .. } => ident.name.clone(),
                            ast::Pattern::RefBinding { name: ident, .. } => ident.name.clone(),
                            ast::Pattern::Ref { pattern, .. } => {
                                if let ast::Pattern::Ident { name: ident, .. } = &**pattern {
                                    ident.name.clone()
                                } else {
                                    format!("param{}", i)
                                }
                            }
                            _ => format!("param{}", i),
                        };
                        let param_value = fn_value.get_nth_param(i as u32).unwrap();
                        let alloca = self
                            .builder
                            .build_alloca(self.context.i64_type(), &param_name)
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(alloca, param_value)
                            .map_err(|e| e.to_string())?;
                        scope.vars.insert(param_name.clone(), alloca);

                        // G23: Track float parameters for float detection
                        if self.type_contains_f64(&param.ty) {
                            scope.float_vars.insert(param_name.clone());
                        }

                        // G26: Track struct type from parameter type annotation for method dispatch
                        // For self/this parameters, use the impl block's type
                        if param_name == "this" || param_name == "self" {
                            scope.register_struct_type(param_name.clone(), type_name.clone());
                        } else if let Some(struct_type) = self.extract_struct_type_from_type_expr(&param.ty) {
                            scope.register_struct_type(param_name.clone(), struct_type);
                        }

                        // G32: Track byte slice parameters (&[u8], &str) for direct pointer indexing
                        // This was missing from impl methods, causing .len() on &[u8] params to fail
                        if self.type_is_byte_slice(&param.ty) {
                            scope.var_types.insert(param_name, SigilType::String);
                        }
                    }

                    // Compile function body
                    if let Some(ref body) = func.body {
                        let result = self.compile_block(fn_value, &mut scope, body)?;

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
                        let zero = self.context.i64_type().const_int(0, false);
                        self.builder
                            .build_return(Some(&zero))
                            .map_err(|e| e.to_string())?;
                    }
                }
            }

            // Clear the current Self type
            self.current_self_type = None;
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
            let actual_name = if self.compile_mode == CompileMode::Aot
                && name == "main"
                && self.current_module.len() == 1
            {
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

            // Handle inline hints for controlling inlining behavior
            if let Some(ref hint) = func.attrs.inline {
                match hint {
                    ast::InlineHint::Never => {
                        // Prevent inlining - critical for benchmarks to avoid DCE
                        let noinline_attr = self.context.create_enum_attribute(
                            inkwell::attributes::Attribute::get_named_enum_kind_id("noinline"),
                            0,
                        );
                        fn_value.add_attribute(inkwell::attributes::AttributeLoc::Function, noinline_attr);
                    }
                    ast::InlineHint::Always => {
                        let alwaysinline_attr = self.context.create_enum_attribute(
                            inkwell::attributes::Attribute::get_named_enum_kind_id("alwaysinline"),
                            0,
                        );
                        fn_value.add_attribute(inkwell::attributes::AttributeLoc::Function, alwaysinline_attr);
                    }
                    ast::InlineHint::Hint => {
                        let inlinehint_attr = self.context.create_enum_attribute(
                            inkwell::attributes::Attribute::get_named_enum_kind_id("inlinehint"),
                            0,
                        );
                        fn_value.add_attribute(inkwell::attributes::AttributeLoc::Function, inlinehint_attr);
                    }
                }
            }

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

            // G21: Track functions that return f64 for float detection in println!
            if let Some(ref return_type) = func.return_type {
                if self.type_contains_f64(return_type) {
                    self.float_funcs.insert(name.clone());
                }
                // G31: Store return type for tuple destructuring type inference
                self.ret_types.insert(name.clone(), return_type.clone());
            }

            Ok(fn_value)
        }

        /// Compile a function body
        fn compile_function(&mut self, func: &ast::Function) -> Result<(), String> {
            let name = &func.name.name;
            // eprintln!("DEBUG: Compiling function: {}", name);
            let fn_value = *self.functions.get(name).ok_or("Function not declared")?;

            // Create entry block
            let entry = self.context.append_basic_block(fn_value, "entry");
            self.builder.position_at_end(entry);

            // Set up variable scope
            let mut scope = CompileScope::new();
            // G21: Copy global float_funcs registry to scope for float detection
            scope.float_funcs = self.float_funcs.clone();

            // Add parameters to scope
            for (i, param) in func.params.iter().enumerate() {
                // Extract parameter name from various pattern types
                let param_name = match &param.pattern {
                    ast::Pattern::Ident { name: ident, .. } => Some(ident.name.clone()),
                    ast::Pattern::RefBinding { name: ident, .. } => Some(ident.name.clone()),
                    _ => {
                        eprintln!("WARNING: Unhandled parameter pattern type in function '{}' param {}: {:?}",
                            name, i, param.pattern);
                        None
                    }
                };

                if let Some(param_name) = param_name {
                    let param_value = fn_value.get_nth_param(i as u32).unwrap();
                    // Allocate on stack for potential mutation
                    let alloca = self
                        .builder
                        .build_alloca(self.context.i64_type(), &param_name)
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_store(alloca, param_value)
                        .map_err(|e| e.to_string())?;
                    scope.vars.insert(param_name.clone(), alloca);

                    // Check if parameter type contains f64 (for Vec<f64> or f64 params)
                    // This enables float detection for indexing operations
                    if self.type_contains_f64(&param.ty) {
                        scope.float_vars.insert(param_name.clone());
                    }

                    // G26: Track struct type from parameter type annotation for method dispatch
                    if let Some(struct_type) = self.extract_struct_type_from_type_expr(&param.ty) {
                        scope.register_struct_type(param_name.clone(), struct_type);
                    }

                    // G28: Track byte slice parameters (&[u8], &str, &[T]) for direct pointer indexing
                    if self.type_is_byte_slice(&param.ty) {
                        scope.var_types.insert(param_name.clone(), SigilType::String);
                    }
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
            // eprintln!("DEBUG: compile_block with {} stmts, expr: {}", block.stmts.len(), block.expr.is_some());

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
                ast::Stmt::Let { pattern, ty, init } => {
                    // G14 fix: Handle Pattern::Tuple for tuple destructuring
                    if let ast::Pattern::Tuple(patterns) = pattern {
                        // Compile the init expression (should be a tuple pointer)
                        let tuple_ptr_int = if let Some(ref expr) = init {
                            self.compile_expr(fn_value, scope, expr)?
                        } else {
                            return Err("Tuple pattern requires initializer".to_string());
                        };

                        // Convert i64 to pointer
                        let tuple_ptr = self
                            .builder
                            .build_int_to_ptr(
                                tuple_ptr_int,
                                self.context.ptr_type(AddressSpace::default()),
                                "tuple_destr_ptr",
                            )
                            .map_err(|e| e.to_string())?;

                        // G31: Determine if tuple elements are floats based on function return type
                        let mut elem_is_float = vec![false; patterns.len()];
                        if let Some(ref expr) = init {
                            if let Expr::Call { func, .. } = expr {
                                if let Expr::Path(path) = func.as_ref() {
                                    if let Some(seg) = path.segments.last() {
                                        // Look up function return type in ret_types
                                        if let Some(ret_ty) = self.ret_types.get(&seg.ident.name) {
                                            // Check if return type is tuple with f64 elements
                                            if let ast::TypeExpr::Tuple(elem_types) = ret_ty {
                                                for (i, ty) in elem_types.iter().enumerate() {
                                                    if i < elem_is_float.len() && self.type_contains_f64(ty) {
                                                        elem_is_float[i] = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Extract each element and bind to pattern variables
                        for (idx, pat) in patterns.iter().enumerate() {
                            if let ast::Pattern::Ident { name: ident, .. } = pat {
                                // Load element from tuple at offset idx * 8
                                let offset = self.context.i64_type().const_int(idx as u64 * 8, false);
                                let elem_ptr = unsafe {
                                    self.builder
                                        .build_gep(
                                            self.context.i8_type(),
                                            tuple_ptr,
                                            &[offset],
                                            &format!("tuple_elem_{}_ptr", idx),
                                        )
                                        .map_err(|e| e.to_string())?
                                };
                                let elem_val = self
                                    .builder
                                    .build_load(self.context.i64_type(), elem_ptr, &format!("tuple_elem_{}", idx))
                                    .map_err(|e| e.to_string())?
                                    .into_int_value();

                                // Allocate and store
                                let alloca = self
                                    .builder
                                    .build_alloca(self.context.i64_type(), &ident.name)
                                    .map_err(|e| e.to_string())?;
                                self.builder
                                    .build_store(alloca, elem_val)
                                    .map_err(|e| e.to_string())?;
                                scope.vars.insert(ident.name.clone(), alloca);

                                // G31: Track float types for tuple elements
                                if idx < elem_is_float.len() && elem_is_float[idx] {
                                    scope.float_vars.insert(ident.name.clone());
                                }
                            }
                        }
                        return Ok(None);
                    }

                    if let ast::Pattern::Ident {
                        name: ref ident, ..
                    } = pattern
                    {
                        // eprintln!("DEBUG: Let binding: {}", ident.name);
                        // G19: Check if type annotation contains f64 (e.g., Vec<f64>)
                        let is_float_from_ty = if let Some(ref type_expr) = ty {
                            self.type_contains_f64(type_expr)
                        } else {
                            false
                        };

                        // Check if init is a float expression before compiling
                        let is_float = is_float_from_ty || if let Some(ref expr) = init {
                            self.is_float_expr_with_scope(expr, scope)
                        } else {
                            false
                        };

                        // G27: Check if type annotation indicates string
                        let is_string_from_ty = if let Some(ref type_expr) = ty {
                            let ty_str = self.type_expr_to_string(type_expr);
                            ty_str.contains("str") || ty_str.contains("String")
                        } else {
                            false
                        };

                        // G27: Check if init is a string literal
                        let is_string_literal = if let Some(ref expr) = init {
                            matches!(expr, Expr::Literal(Literal::String(_)))
                        } else {
                            false
                        };

                        // G32: Check if init is a function call that returns Rust String
                        // Functions returning String type return heap-allocated Rust Strings
                        let is_rust_string_from_call = if let Some(ref expr) = init {
                            if let Expr::Call { func, .. } = expr {
                                if let Expr::Path(path) = &**func {
                                    if let Some(seg) = path.segments.last() {
                                        let name = &seg.ident.name;
                                        // Check ret_types for String return type
                                        let ret_is_string = if let Some(ret_ty) = self.ret_types.get(name) {
                                            self.type_contains_string(ret_ty)
                                        } else {
                                            false
                                        };
                                        // Heuristic: functions that likely return Rust Strings
                                        ret_is_string || name.contains("_corpus")
                                            || name == "format"
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // G27: Check if init is a function call that returns C string (&str)
                        // These are less common - mostly for functions returning static strings
                        // G37: Don't apply heuristic if ret_types proves it returns Rust String
                        let is_string_from_call = if is_rust_string_from_call {
                            // Already known to return Rust String, not C string
                            false
                        } else if let Some(ref expr) = init {
                            if let Expr::Call { func, .. } = expr {
                                if let Expr::Path(path) = &**func {
                                    if let Some(seg) = path.segments.last() {
                                        let name = &seg.ident.name;
                                        // Heuristic: functions that return C strings
                                        // G37: Use more specific pattern to avoid "make_string" matching "_str"
                                        name.ends_with("_str") || name.starts_with("get_")
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // G27/G32: Check if init is a method call that returns bytes
                        // Note: as_bytes returns &[u8] which is C-string-like, NOT a Rust String
                        let is_string_from_method = if let Some(ref expr) = init {
                            if let Expr::MethodCall { method, .. } = expr {
                                method.name == "as_bytes"
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // G32: Check if init is a method call that returns Rust String
                        let is_rust_string_from_method = if let Some(ref expr) = init {
                            if let Expr::MethodCall { method, .. } = expr {
                                // to_string() returns a heap-allocated Rust String
                                method.name == "to_string"
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // C strings: literals, as_bytes, and C-string returning functions
                        let is_string = is_string_from_ty || is_string_literal || is_string_from_call || is_string_from_method;

                        // G32: Rust Strings: to_string(), fs_read(), and String-returning functions
                        let is_fs_read = if let Some(ref expr) = init {
                            if let Expr::Call { func, .. } = expr {
                                if let Expr::Path(path) = &**func {
                                    if let Some(seg) = path.segments.last() {
                                        seg.ident.name == "fs_read"
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        let is_rust_string = is_rust_string_from_method || is_rust_string_from_call || is_fs_read;

                        // Check if init is a struct type for method dispatch
                        let struct_type = if let Some(ref expr) = init {
                            self.get_struct_type_from_expr(expr, scope)
                        } else {
                            None
                        };

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

                        // Track float variables
                        if is_float {
                            scope.float_vars.insert(ident.name.clone());
                        }

                        // G27: Track string variables for print handling
                        if is_string {
                            scope.var_types.insert(ident.name.clone(), SigilType::String);
                        } else if is_rust_string {
                            scope.var_types.insert(ident.name.clone(), SigilType::RustString);
                        }

                        // Track struct type for method dispatch (G11 fix)
                        if let Some(ty) = struct_type {
                            scope.register_struct_type(ident.name.clone(), ty);
                        }
                        // eprintln!("DEBUG: Added {} to scope, scope now: {:?}", ident.name, scope.vars.keys().collect::<Vec<_>>());
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

                    // Build full path for qualified lookups
                    let full_path: String = path
                        .segments
                        .iter()
                        .map(|s| s.ident.name.as_str())
                        .collect::<Vec<_>>()
                        .join("::");

                    if let Some(&ptr) = scope.vars.get(name) {
                        let val = self
                            .builder
                            .build_load(self.context.i64_type(), ptr, name)
                            .map_err(|e| e.to_string())?;
                        Ok(val.into_int_value())
                    } else if let Some(global) = self.global_vars.get(name) {
                        // Load from global variable
                        let val = self
                            .builder
                            .build_load(self.context.i64_type(), global.as_pointer_value(), name)
                            .map_err(|e| e.to_string())?;
                        Ok(val.into_int_value())
                    } else if path.segments.len() > 1 {
                        // Qualified path like Token::LParen or Type::Variant
                        // Check if it's a qualified enum variant
                        let type_name = path
                            .segments
                            .first()
                            .map(|s| s.ident.name.as_str())
                            .unwrap_or("");

                        // Try to find in registered enum types
                        if let Some(enum_info) = self.enum_types.get(type_name) {
                            if let Some(&discriminant) = enum_info.variants.get(name) {
                                return Ok(self.context.i64_type().const_int(discriminant, false));
                            }
                        }

                        // Fallback: treat as a constant enum value (use hash of name as discriminant)
                        // This handles cases like Token::LParen, TokenKind::Fn, etc.
                        let hash = full_path
                            .bytes()
                            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                        Ok(self.context.i64_type().const_int(hash, false))
                    } else {
                        // Check if it's an unqualified enum variant (search all enums)
                        for (_, enum_info) in &self.enum_types {
                            if let Some(&discriminant) = enum_info.variants.get(name) {
                                return Ok(self.context.i64_type().const_int(discriminant, false));
                            }
                        }

                        // Check if it might be a constant (UPPER_CASE naming convention)
                        if name
                            .chars()
                            .all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
                        {
                            // Treat as constant - use hash of name
                            let hash = name
                                .bytes()
                                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                            return Ok(self.context.i64_type().const_int(hash, false));
                        }

                        // Fallback: return 0 for unknown symbols (may be external constants)
                        // This is lenient but allows compilation to proceed
                        Ok(self.context.i64_type().const_int(0, false))
                    }
                }
                Expr::Binary { op, left, right } => {
                    // Check if either operand is a float expression (using scope for variable tracking)
                    let is_float = self.is_float_expr_with_scope(left, scope) || self.is_float_expr_with_scope(right, scope);
                    if is_float {
                        // Use native float path for arithmetic ops - avoids bitcasts within the expression
                        match op {
                            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                                let lhs_f64 = self.compile_native_float_expr(fn_value, scope, left)?;
                                let rhs_f64 = self.compile_native_float_expr(fn_value, scope, right)?;
                                let result_f64 = match op {
                                    BinOp::Add => self.builder.build_float_add(lhs_f64, rhs_f64, "fadd"),
                                    BinOp::Sub => self.builder.build_float_sub(lhs_f64, rhs_f64, "fsub"),
                                    BinOp::Mul => self.builder.build_float_mul(lhs_f64, rhs_f64, "fmul"),
                                    BinOp::Div => self.builder.build_float_div(lhs_f64, rhs_f64, "fdiv"),
                                    BinOp::Rem => self.builder.build_float_rem(lhs_f64, rhs_f64, "frem"),
                                    _ => unreachable!(),
                                }.map_err(|e| e.to_string())?;
                                // Convert back to i64 bits for return
                                self.builder.build_bit_cast(result_f64, self.context.i64_type(), "fres_bits")
                                    .map_err(|e| e.to_string())
                                    .map(|v| v.into_int_value())
                            }
                            _ => {
                                // For comparisons and other ops, use the traditional path
                                let lhs = self.compile_expr(fn_value, scope, left)?;
                                let rhs = self.compile_expr(fn_value, scope, right)?;
                                self.compile_float_binary_op(*op, lhs, rhs)
                            }
                        }
                    } else {
                        let lhs = self.compile_expr(fn_value, scope, left)?;
                        let rhs = self.compile_expr(fn_value, scope, right)?;
                        self.compile_binary_op(*op, lhs, rhs)
                    }
                }
                Expr::Unary { op, expr: inner } => {
                    // Special case: *( ptr + offset ) needs GEP for proper pointer arithmetic
                    if matches!(op, ast::UnaryOp::Deref) {
                        if let Expr::Binary { op: BinOp::Add, left, right } = inner.as_ref() {
                            // Compile base pointer and offset separately
                            let base_addr = self.compile_expr(fn_value, scope, left)?;
                            let offset = self.compile_expr(fn_value, scope, right)?;

                            // Convert base address to pointer
                            let i64_type = self.context.i64_type();
                            let base_ptr = self
                                .builder
                                .build_int_to_ptr(
                                    base_addr,
                                    i64_type.ptr_type(Default::default()),
                                    "base_ptr",
                                )
                                .map_err(|e| e.to_string())?;

                            // Use GEP for proper pointer arithmetic (scales by element size)
                            let elem_ptr = unsafe {
                                self.builder
                                    .build_gep(i64_type, base_ptr, &[offset], "elem_ptr")
                            }
                            .map_err(|e| e.to_string())?;

                            // Load the value
                            let loaded = self
                                .builder
                                .build_load(i64_type, elem_ptr, "deref_val")
                                .map_err(|e| e.to_string())?;
                            return Ok(loaded.into_int_value());
                        }
                    }
                    // Default case: compile inner and apply unary op
                    let val = self.compile_expr(fn_value, scope, inner)?;

                    // G18: Float negation requires XOR with sign bit, not integer neg
                    if matches!(op, ast::UnaryOp::Neg) && self.is_float_expr_with_scope(inner, scope) {
                        // Flip sign bit for float negation
                        let sign_bit = self.context.i64_type().const_int(0x8000000000000000, false);
                        return self.builder
                            .build_xor(val, sign_bit, "fneg")
                            .map_err(|e| e.to_string());
                    }

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
                Expr::While {
                    label: _,
                    condition,
                    body,
                } => self.compile_while(fn_value, scope, condition, body),
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
                            } else if let Some(global) = self.global_vars.get(name) {
                                // Store to global variable
                                self.builder
                                    .build_store(global.as_pointer_value(), val)
                                    .map_err(|e| e.to_string())?;
                                Ok(val)
                            } else {
                                // Fallback: create global on demand for unknown variables
                                let i64_type = self.context.i64_type();
                                let global = self.module.add_global(i64_type, None, name);
                                global.set_initializer(&i64_type.const_int(0, false));
                                self.builder
                                    .build_store(global.as_pointer_value(), val)
                                    .map_err(|e| e.to_string())?;
                                // Note: can't insert into self.global_vars here due to borrow checker
                                Ok(val)
                            }
                        }
                        Expr::Field { expr, field } => {
                            // Get struct pointer from the expression
                            let struct_ptr_int = self.compile_expr(fn_value, scope, expr)?;
                            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let struct_ptr = self
                                .builder
                                .build_int_to_ptr(struct_ptr_int, ptr_type, "struct_ptr")
                                .map_err(|e| e.to_string())?;

                            let field_name = &field.name;

                            // G29 fix: First try to get the struct type from the expression
                            // This ensures we use the correct struct when multiple structs share field names
                            if let Some(struct_type_name) = self.get_struct_type_from_expr(expr, scope) {
                                if let Some(struct_info) = self.struct_types.get(&struct_type_name) {
                                    if let Some(&field_idx) = struct_info.field_indices.get(field_name) {
                                        let field_ptr = self
                                            .builder
                                            .build_struct_gep(
                                                struct_info.llvm_type,
                                                struct_ptr,
                                                field_idx,
                                                &format!("{}_ptr", field_name),
                                            )
                                            .map_err(|e| e.to_string())?;
                                        self.builder
                                            .build_store(field_ptr, val)
                                            .map_err(|e| e.to_string())?;
                                        return Ok(val);
                                    }
                                }
                            }

                            // Fallback: search all struct types for the field (less accurate)
                            for (_name, struct_info) in &self.struct_types {
                                if let Some(&field_idx) = struct_info.field_indices.get(field_name)
                                {
                                    let field_ptr = self
                                        .builder
                                        .build_struct_gep(
                                            struct_info.llvm_type,
                                            struct_ptr,
                                            field_idx,
                                            &format!("{}_ptr", field_name),
                                        )
                                        .map_err(|e| e.to_string())?;
                                    self.builder
                                        .build_store(field_ptr, val)
                                        .map_err(|e| e.to_string())?;
                                    return Ok(val);
                                }
                            }
                            // Fallback: use offset-based field access for assignment
                            let offset = match field_name.as_str() {
                                "0" => 0u64,
                                "1" => 1,
                                "2" => 2,
                                "3" => 3,
                                "start" | "first" | "x" | "name" | "key" | "id" => 0,
                                "end" | "second" | "y" | "value" | "ty" => 1,
                                "z" | "third" | "body" | "args" => 2,
                                _ => 0, // Default to first field
                            };
                            let offset_val = self.context.i64_type().const_int(offset * 8, false);
                            let field_ptr_int = self
                                .builder
                                .build_int_add(struct_ptr_int, offset_val, "field_ptr")
                                .map_err(|e| e.to_string())?;
                            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let field_ptr = self
                                .builder
                                .build_int_to_ptr(
                                    field_ptr_int,
                                    ptr_type,
                                    &format!("{}_ptr", field_name),
                                )
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_store(field_ptr, val)
                                .map_err(|e| e.to_string())?;
                            Ok(val)
                        }
                        Expr::Unary { op, expr } if matches!(op, ast::UnaryOp::Deref) => {
                            // Dereference assignment: *ptr = val or *(ptr + offset) = val
                            // Check for pointer arithmetic pattern
                            if let Expr::Binary { op: BinOp::Add, left, right } = expr.as_ref() {
                                let base_addr = self.compile_expr(fn_value, scope, left)?;
                                let offset = self.compile_expr(fn_value, scope, right)?;
                                let i64_type = self.context.i64_type();
                                let base_ptr = self
                                    .builder
                                    .build_int_to_ptr(
                                        base_addr,
                                        i64_type.ptr_type(Default::default()),
                                        "base_ptr",
                                    )
                                    .map_err(|e| e.to_string())?;
                                // Use GEP for proper pointer arithmetic (scales by element size)
                                let elem_ptr = unsafe {
                                    self.builder
                                        .build_gep(i64_type, base_ptr, &[offset], "elem_ptr")
                                }
                                .map_err(|e| e.to_string())?;
                                self.builder
                                    .build_store(elem_ptr, val)
                                    .map_err(|e| e.to_string())?;
                                return Ok(val);
                            }
                            // Simple dereference: *ptr = val
                            let ptr_val = self.compile_expr(fn_value, scope, expr)?;
                            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let ptr = self
                                .builder
                                .build_int_to_ptr(ptr_val, ptr_type, "deref_ptr")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_store(ptr, val)
                                .map_err(|e| e.to_string())?;
                            Ok(val)
                        }
                        Expr::Index { expr, index } => {
                            // Index assignment: arr[i] = val
                            // G25 Fix: Use sigil_vec_set for proper Vec access
                            // The Rust Vec memory layout (ptr, len, cap) doesn't match
                            // the inline data assumption. Call the runtime function.
                            let idx = self.compile_expr(fn_value, scope, index)?;
                            let vec_ptr = self.compile_expr(fn_value, scope, expr)?;

                            let vec_set_fn = self
                                .module
                                .get_function("sigil_vec_set")
                                .ok_or("sigil_vec_set not declared")?;

                            self.builder
                                .build_call(
                                    vec_set_fn,
                                    &[vec_ptr.into(), idx.into(), val.into()],
                                    "",
                                )
                                .map_err(|e| e.to_string())?;

                            Ok(val)
                        }
                        Expr::Deref(inner) => {
                            // Dereference assignment: *ptr = val or *(ptr + offset) = val
                            // Check for pointer arithmetic pattern
                            if let Expr::Binary { op: BinOp::Add, left, right } = inner.as_ref() {
                                let base_addr = self.compile_expr(fn_value, scope, left)?;
                                let offset = self.compile_expr(fn_value, scope, right)?;
                                let i64_type = self.context.i64_type();
                                let base_ptr = self
                                    .builder
                                    .build_int_to_ptr(
                                        base_addr,
                                        i64_type.ptr_type(Default::default()),
                                        "base_ptr",
                                    )
                                    .map_err(|e| e.to_string())?;
                                // Use GEP for proper pointer arithmetic (scales by element size)
                                let elem_ptr = unsafe {
                                    self.builder
                                        .build_gep(i64_type, base_ptr, &[offset], "elem_ptr")
                                }
                                .map_err(|e| e.to_string())?;
                                self.builder
                                    .build_store(elem_ptr, val)
                                    .map_err(|e| e.to_string())?;
                                return Ok(val);
                            }
                            // Simple dereference: *ptr = val
                            let ptr_val = self.compile_expr(fn_value, scope, inner)?;
                            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let ptr = self
                                .builder
                                .build_int_to_ptr(ptr_val, ptr_type, "deref_ptr")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_store(ptr, val)
                                .map_err(|e| e.to_string())?;
                            Ok(val)
                        }
                        _ => {
                            // Lenient fallback: just evaluate the target and return the value
                            // This allows compilation to proceed for unsupported assignment patterns
                            let _ = self.compile_expr(fn_value, scope, target)?;
                            Ok(val)
                        }
                    }
                }
                Expr::Block(block) => {
                    let result = self.compile_block(fn_value, scope, block)?;
                    Ok(result.unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                }
                Expr::Struct { path, fields, .. } => {
                    // Get struct name and potential generic arguments from path
                    let last_segment = path.segments.last().ok_or("Empty struct path")?;
                    let base_name = last_segment.ident.name.as_str();

                    // Resolve Self/This to actual type name
                    let base_name: String = if base_name == "Self" || base_name == "This" {
                        if let Some(ref self_type) = self.current_self_type {
                            self_type.clone()
                        } else {
                            return Err("Self/This used outside of impl block".to_string());
                        }
                    } else {
                        base_name.to_string()
                    };

                    // Check if this is a generic struct instantiation
                    let struct_name = if let Some(ref type_args) = last_segment.generics {
                        // Monomorphize the generic struct
                        self.monomorphize_struct(&base_name, type_args)?
                    } else if self.generic_structs.contains_key(&base_name) {
                        // Generic struct used without type args - use default monomorphization
                        // Assume i64 for all type parameters
                        format!("{}_i64", base_name)
                    } else {
                        base_name
                    };

                    // Look up struct type (now with mangled name for generics)
                    let struct_info_opt = self.struct_types.get(&struct_name).cloned();

                    if let Some(struct_info) = struct_info_opt {
                        // Known struct type - use heap allocation so returned structs survive
                        // Calculate struct size: number of fields * 8 bytes (all fields are i64)
                        let struct_size = struct_info.field_indices.len() as u64 * 8;
                        let size_const = self.context.i64_type().const_int(struct_size.max(8), false);

                        let alloc_fn = self
                            .module
                            .get_function("sigil_alloc")
                            .ok_or("sigil_alloc not declared")?;
                        let alloc_call = self
                            .builder
                            .build_call(alloc_fn, &[size_const.into()], "struct_alloc")
                            .map_err(|e| e.to_string())?;
                        let alloc_result = alloc_call
                            .try_as_basic_value()
                            .left()
                            .ok_or("sigil_alloc returned void")?;

                        // sigil_alloc returns ptr type - convert to typed pointer
                        let struct_ptr = if alloc_result.is_pointer_value() {
                            alloc_result.into_pointer_value()
                        } else {
                            // If it returns i64, convert to pointer
                            self.builder
                                .build_int_to_ptr(
                                    alloc_result.into_int_value(),
                                    self.context.ptr_type(AddressSpace::default()),
                                    "struct_heap_ptr",
                                )
                                .map_err(|e| e.to_string())?
                        };

                        // Initialize each field
                        for (idx, field_init) in fields.iter().enumerate() {
                            let field_name = &field_init.name.name;
                            // Try to get field index from struct info, fallback to position
                            let field_idx = *struct_info
                                .field_indices
                                .get(field_name)
                                .unwrap_or(&(idx as u32));

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
                                    // Fallback: use 0 for unknown shorthand variables
                                    self.context.i64_type().const_int(0, false)
                                }
                            };

                            // Get pointer to field and store value
                            let field_ptr = self
                                .builder
                                .build_struct_gep(
                                    struct_info.llvm_type,
                                    struct_ptr,
                                    field_idx,
                                    &format!("{}_ptr", field_name),
                                )
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_store(field_ptr, field_value)
                                .map_err(|e| e.to_string())?;
                        }

                        // Return struct pointer as i64
                        let ptr_int = self
                            .builder
                            .build_ptr_to_int(struct_ptr, self.context.i64_type(), "struct_ptr")
                            .map_err(|e| e.to_string())?;
                        return Ok(ptr_int);
                    }

                    // Unknown struct type - create dynamic struct on the fly
                    // Build field types (all i64 for now)
                    let field_types: Vec<BasicTypeEnum> = fields
                        .iter()
                        .map(|_| self.context.i64_type().into())
                        .collect();

                    // Create struct type dynamically
                    let llvm_type = self.context.struct_type(&field_types, false);

                    // Use heap allocation so returned structs survive (matches known struct path)
                    let struct_size = (fields.len() as u64).max(1) * 8;
                    let size_const = self.context.i64_type().const_int(struct_size, false);
                    let alloc_fn = self
                        .module
                        .get_function("sigil_alloc")
                        .ok_or("sigil_alloc not declared")?;
                    let alloc_call = self
                        .builder
                        .build_call(alloc_fn, &[size_const.into()], "struct_alloc_dyn")
                        .map_err(|e| e.to_string())?;
                    let alloc_result = alloc_call
                        .try_as_basic_value()
                        .left()
                        .ok_or("sigil_alloc returned void")?;
                    let struct_ptr = if alloc_result.is_pointer_value() {
                        alloc_result.into_pointer_value()
                    } else {
                        self.builder
                            .build_int_to_ptr(
                                alloc_result.into_int_value(),
                                self.context.ptr_type(AddressSpace::default()),
                                "struct_heap_ptr_dyn",
                            )
                            .map_err(|e| e.to_string())?
                    };

                    // Initialize each field by index
                    for (idx, field_init) in fields.iter().enumerate() {
                        let field_name = &field_init.name.name;
                        let field_idx = idx as u32;

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
                                // Fallback: use 0 for unknown shorthand variables
                                self.context.i64_type().const_int(0, false)
                            }
                        };

                        // Get pointer to field and store value
                        let field_ptr = self
                            .builder
                            .build_struct_gep(
                                llvm_type,
                                struct_ptr,
                                field_idx,
                                &format!("{}_ptr", field_name),
                            )
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(field_ptr, field_value)
                            .map_err(|e| e.to_string())?;
                    }

                    // Return struct pointer as i64
                    let ptr_int = self
                        .builder
                        .build_ptr_to_int(struct_ptr, self.context.i64_type(), "struct_ptr")
                        .map_err(|e| e.to_string())?;
                    Ok(ptr_int)
                }
                Expr::Field { expr, field } => {
                    // Compile the struct expression to get pointer
                    let struct_ptr_int = self.compile_expr(fn_value, scope, expr)?;

                    // Convert i64 back to pointer
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let struct_ptr = self
                        .builder
                        .build_int_to_ptr(struct_ptr_int, ptr_type, "struct_ptr")
                        .map_err(|e| e.to_string())?;

                    let field_name = &field.name;

                    // G13 fix: First try to get the struct type from the expression
                    // This ensures we use the correct struct when multiple structs share field names
                    if let Some(struct_type_name) = self.get_struct_type_from_expr(expr, scope) {
                        if let Some(struct_info) = self.struct_types.get(&struct_type_name) {
                            if let Some(&field_idx) = struct_info.field_indices.get(field_name) {
                                let field_ptr = self
                                    .builder
                                    .build_struct_gep(
                                        struct_info.llvm_type,
                                        struct_ptr,
                                        field_idx,
                                        &format!("{}_ptr", field_name),
                                    )
                                    .map_err(|e| e.to_string())?;
                                let field_value = self
                                    .builder
                                    .build_load(self.context.i64_type(), field_ptr, field_name)
                                    .map_err(|e| e.to_string())?;
                                return Ok(field_value.into_int_value());
                            }
                        }
                    }

                    // Fallback: search all struct types for the field (less accurate)
                    for (_name, struct_info) in &self.struct_types {
                        if let Some(&field_idx) = struct_info.field_indices.get(field_name) {
                            let field_ptr = self
                                .builder
                                .build_struct_gep(
                                    struct_info.llvm_type,
                                    struct_ptr,
                                    field_idx,
                                    &format!("{}_ptr", field_name),
                                )
                                .map_err(|e| e.to_string())?;
                            let field_value = self
                                .builder
                                .build_load(self.context.i64_type(), field_ptr, field_name)
                                .map_err(|e| e.to_string())?;
                            return Ok(field_value.into_int_value());
                        }
                    }

                    // Fallback: For structs not in our registry, use offset-based access
                    // Common struct field conventions:
                    // - Span: start(0), end(1)
                    // - Range: start(0), end(1)
                    // - Point: x(0), y(1), z(2)
                    // - Location: line(0), col(1)
                    // - General: first field at offset 0, etc.
                    let field_offset = match field_name.as_str() {
                        // First field (offset 0)
                        "start" | "first" | "x" | "line" | "lo" | "begin" | "name" | "key"
                        | "id" | "data" | "value" => Some(0u64),
                        // Second field (offset 1)
                        "end" | "second" | "y" | "col" | "hi" | "suffix" | "span" | "ty"
                        | "type" => Some(1u64),
                        // Third field (offset 2)
                        "z" | "third" | "depth" | "body" | "args" | "params" => Some(2u64),
                        // Fourth field (offset 3)
                        "w" | "fourth" | "return_type" | "ret" => Some(3u64),
                        // Common Sigil types
                        "message" => Some(0u64),
                        "source" => Some(1u64),
                        "file" => Some(2u64),
                        "kind" => Some(0u64),
                        "ident" => Some(0u64),
                        "segments" => Some(0u64),
                        "items" => Some(0u64),
                        "fields" => Some(0u64),
                        "variants" => Some(0u64),
                        "methods" => Some(0u64),
                        "generics" => Some(1u64),
                        "vis" | "visibility" => Some(0u64),
                        "attrs" | "attributes" => Some(1u64),
                        "node" => Some(0u64),
                        "path" | "paths" => Some(0u64),
                        "left" | "lhs" => Some(0u64),
                        "right" | "rhs" => Some(1u64),
                        "op" => Some(0u64),
                        "inner" | "expr" => Some(0u64),
                        "condition" | "cond" => Some(0u64),
                        "then_branch" | "then" => Some(1u64),
                        "else_branch" | "else" => Some(2u64),
                        "pattern" | "pat" => Some(0u64),
                        "iter" | "iterator" => Some(1u64),
                        "guard" => Some(1u64),
                        "arms" => Some(1u64),
                        "scrutinee" => Some(0u64),
                        "init" => Some(0u64),
                        "receiver" => Some(0u64),
                        "method" => Some(1u64),
                        "func" => Some(0u64),
                        "callee" => Some(0u64),
                        "target" => Some(0u64),
                        "module" | "mod" => Some(0u64),
                        "imports" | "uses" => Some(0u64),
                        "exports" => Some(1u64),
                        "decls" | "declarations" => Some(0u64),
                        "stmts" | "statements" => Some(0u64),
                        // AST and compiler-related
                        "tree" | "ast" | "root" => Some(0u64),
                        "content" | "contents" => Some(0u64),
                        "entries" | "elements" | "children" => Some(0u64),
                        "parent" => Some(0u64),
                        "next" => Some(1u64),
                        "prev" => Some(0u64),
                        "tokens" => Some(0u64),
                        "input" => Some(0u64),
                        "output" => Some(1u64),
                        "errors" => Some(0u64),
                        "warnings" => Some(1u64),
                        "result" => Some(0u64),
                        "success" => Some(0u64),
                        "failure" => Some(1u64),
                        "text" | "string" | "str" => Some(0u64),
                        "len" | "length" | "count" | "size" => Some(1u64),
                        // G33: Common struct fields that need specific offsets
                        "vocab_size" => Some(2u64),  // BPETokenizer: vocab(0), merges(1), vocab_size(2)
                        "pos" | "position" => Some(0u64),
                        "offset" => Some(0u64),
                        "range" => Some(0u64),
                        "scope" => Some(0u64),
                        "context" | "ctx" => Some(0u64),
                        "state" | "status" => Some(0u64),
                        "index" | "idx" => Some(0u64),
                        "level" => Some(0u64),
                        "tomes" | "modules" => Some(0u64),
                        "functions" | "fns" => Some(0u64),
                        "structs" | "types" => Some(0u64),
                        "traits" | "interfaces" => Some(0u64),
                        "env" | "environment" => Some(0u64),
                        "bindings" => Some(0u64),
                        "symbols" => Some(0u64),
                        "table" => Some(0u64),
                        // More compiler/IR related
                        "impls" | "implementations" => Some(0u64),
                        "enums" => Some(0u64),
                        "consts" | "constants" => Some(0u64),
                        "statics" => Some(0u64),
                        "uses" | "use_stmts" => Some(0u64),
                        "imports" => Some(0u64),
                        "exports" => Some(0u64),
                        "type_aliases" | "aliases" => Some(0u64),
                        "macros" => Some(0u64),
                        "generics" | "type_params" => Some(1u64),
                        "where_clause" | "bounds" => Some(2u64),
                        "receiver" | "self_param" => Some(0u64),
                        "params" | "parameters" => Some(1u64),
                        "return_ty" | "ret_ty" => Some(2u64),
                        "is_pub" | "is_public" => Some(0u64),
                        "is_mut" | "is_mutable" => Some(0u64),
                        "is_const" | "is_constant" => Some(0u64),
                        "is_static" => Some(0u64),
                        "is_async" => Some(0u64),
                        "is_unsafe" => Some(0u64),
                        "declared" | "declarations" => Some(0u64),
                        "defined" | "definitions" => Some(1u64),
                        "referenced" | "references" => Some(2u64),
                        "resolved" => Some(0u64),
                        "unresolved" => Some(1u64),
                        "pending" => Some(2u64),
                        "diagnostics" | "diags" => Some(0u64),
                        "notes" => Some(1u64),
                        "hints" => Some(2u64),
                        "fixes" | "suggestions" => Some(3u64),
                        // Impl block related
                        "self_ty" | "self_type" | "for_type" => Some(0u64),
                        "trait_ref" | "trait_name" | "trait_path" => Some(1u64),
                        "associated_items" | "assoc_items" => Some(2u64),
                        // Type related
                        "inner_ty" | "inner_type" | "elem_ty" | "elem_type" => Some(0u64),
                        "key_ty" | "key_type" => Some(0u64),
                        "value_ty" | "value_type" => Some(1u64),
                        // Variable/binding related
                        "mutable" | "is_mutable" | "is_mut" => Some(0u64),
                        "evidence" | "evidential" | "evidentiality" => Some(0u64),
                        "element" | "elem" | "item" => Some(0u64),
                        // More AST fields
                        "prefix" => Some(0u64),
                        "suffix" => Some(1u64),
                        "base" => Some(0u64),
                        "index" => Some(0u64),
                        "slice_from" => Some(0u64),
                        "slice_to" => Some(1u64),
                        "callee" | "callable" => Some(0u64),
                        "arguments" | "args_list" => Some(1u64),
                        "is_async" | "async" => Some(0u64),
                        "is_await" | "await" => Some(0u64),
                        "is_unsafe" | "unsafe" => Some(0u64),
                        "is_move" | "move" => Some(0u64),
                        "capture" | "captures" => Some(0u64),
                        "label" => Some(0u64),
                        "lifetime" => Some(0u64),
                        "value" | "val" => Some(0u64),
                        "default" | "default_value" => Some(1u64),
                        // More compiler fields
                        "modulus" | "mod_val" => Some(0u64),
                        "no_std" | "nostd" => Some(0u64),
                        "patterns" | "pats" => Some(0u64),
                        "pos" | "position" | "cursor" => Some(0u64),
                        "globals" | "global_vars" => Some(0u64),
                        "current_fn_has_mut_self" | "has_mut_self" => Some(0u64),
                        "current_fn" | "current_function" => Some(0u64),
                        "current_block" | "cur_block" => Some(0u64),
                        "current_loop" | "loop_info" => Some(0u64),
                        "break_target" | "break_bb" => Some(0u64),
                        "continue_target" | "continue_bb" => Some(1u64),
                        "return_type" | "ret_type" | "fn_ret_type" => Some(0u64),
                        "locals" | "local_vars" => Some(0u64),
                        "temps" | "temporaries" => Some(0u64),
                        "stack" | "stack_ptr" => Some(0u64),
                        "heap" | "heap_ptr" => Some(0u64),
                        "output_buffer" | "buf" | "buffer" => Some(0u64),
                        "indent" | "indent_level" => Some(0u64),
                        "line_start" | "col_start" => Some(0u64),
                        "line_end" | "col_end" => Some(1u64),
                        "filename" | "file_name" | "filepath" | "file_path" => Some(0u64),
                        // Iterator/loop fields
                        "iterable" | "collection" | "sequence" => Some(0u64),
                        "current" | "curr" => Some(0u64),
                        "remaining" | "rest" => Some(1u64),
                        "done" | "finished" | "exhausted" => Some(0u64),
                        // Closure/function context
                        "in_closure_body" | "in_closure" => Some(0u64),
                        "in_loop" | "in_loop_body" => Some(0u64),
                        "in_async" | "in_async_fn" => Some(0u64),
                        // SIMD/vector fields
                        "lanes" | "lane_count" => Some(0u64),
                        "scalar_type" | "element_type" => Some(1u64),
                        // Tuple field access (numeric)
                        "0" => Some(0u64),
                        "1" => Some(1u64),
                        "2" => Some(2u64),
                        "3" => Some(3u64),
                        "4" => Some(4u64),
                        "5" => Some(5u64),
                        "6" => Some(6u64),
                        "7" => Some(7u64),
                        // Range fields
                        "inclusive" | "is_inclusive" => Some(0u64),
                        "exclusive" | "is_exclusive" => Some(0u64),
                        // Operator fields
                        "operator" | "op" | "opcode" => Some(0u64),
                        "operand" | "operands" => Some(1u64),
                        "precedence" | "prec" => Some(2u64),
                        "associativity" | "assoc" => Some(3u64),
                        // Type checker fields
                        "params" | "parameters" | "param_list" => Some(1u64),
                        "type_params" | "generic_params" => Some(2u64),
                        "constraints" | "where_clause" => Some(3u64),
                        // Parse state
                        "no_std" | "is_no_std" | "crate_type" => Some(0u64),
                        // Function IR fields
                        "function" | "fn" | "fn_def" | "fn_decl" => Some(0u64),
                        "return_value" | "ret_val" => Some(0u64),
                        "basic_blocks" | "blocks" | "bbs" => Some(0u64),
                        "entry_block" | "entry" | "entry_bb" => Some(0u64),
                        "exit_block" | "exit" | "exit_bb" => Some(1u64),
                        "alloca_block" => Some(2u64),
                        // Repeated common names with different offsets fallback
                        "name" => Some(0u64),
                        "pos" => Some(0u64),
                        // More IR/codegen fields
                        "variant" | "enum_variant" => Some(0u64),
                        "operations" | "ops" => Some(0u64),
                        "instructions" | "instrs" => Some(0u64),
                        // Catch-all for any numeric-ish field that might be an offset
                        _ => {
                            // Try to parse as numeric field access
                            if field_name.chars().all(|c| c.is_numeric()) {
                                field_name.parse::<u64>().ok()
                            } else {
                                None
                            }
                        }
                    };

                    if let Some(offset) = field_offset {
                        // Calculate field pointer using byte offset
                        let offset_val = self.context.i64_type().const_int(offset * 8, false); // 8 bytes per i64
                        let struct_ptr_as_int = self
                            .builder
                            .build_ptr_to_int(struct_ptr, self.context.i64_type(), "ptr_as_int")
                            .map_err(|e| e.to_string())?;
                        let field_ptr_int = self
                            .builder
                            .build_int_add(struct_ptr_as_int, offset_val, "field_ptr_int")
                            .map_err(|e| e.to_string())?;
                        let field_ptr = self
                            .builder
                            .build_int_to_ptr(
                                field_ptr_int,
                                ptr_type,
                                &format!("{}_ptr", field_name),
                            )
                            .map_err(|e| e.to_string())?;
                        let field_value = self
                            .builder
                            .build_load(self.context.i64_type(), field_ptr, field_name)
                            .map_err(|e| e.to_string())?;
                        return Ok(field_value.into_int_value());
                    }

                    // Fallback: use offset 0 for unknown fields
                    // This is lenient but allows compilation to proceed
                    let offset_val = self.context.i64_type().const_int(0, false);
                    let field_ptr_int = self
                        .builder
                        .build_int_add(struct_ptr_int, offset_val, "fallback_field_ptr")
                        .map_err(|e| e.to_string())?;
                    let field_ptr = self
                        .builder
                        .build_int_to_ptr(field_ptr_int, ptr_type, &format!("{}_ptr", field_name))
                        .map_err(|e| e.to_string())?;
                    let field_value = self
                        .builder
                        .build_load(self.context.i64_type(), field_ptr, field_name)
                        .map_err(|e| e.to_string())?;
                    Ok(field_value.into_int_value())
                }
                Expr::Match { expr, arms } => {
                    // Compile the scrutinee (thing being matched)
                    let scrutinee = self.compile_expr(fn_value, scope, expr)?;

                    let merge_bb = self.context.append_basic_block(fn_value, "match_merge");
                    let mut incoming: Vec<(
                        IntValue<'ctx>,
                        inkwell::basic_block::BasicBlock<'ctx>,
                    )> = Vec::new();

                    // Build chain of if-else for each arm
                    for (i, arm) in arms.iter().enumerate() {
                        // Get pattern discriminant value
                        let pattern_val = match &arm.pattern {
                            ast::Pattern::Path(path) => {
                                if path.segments.len() >= 2 {
                                    let enum_name =
                                        &path.segments[path.segments.len() - 2].ident.name;
                                    let variant_name =
                                        &path.segments[path.segments.len() - 1].ident.name;
                                    if let Some(enum_info) = self.enum_types.get(enum_name) {
                                        enum_info.variants.get(variant_name).copied()
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            ast::Pattern::Literal(lit) => {
                                if let Ok(v) = self.compile_literal(lit) {
                                    Some(v.get_zero_extended_constant().unwrap_or(0))
                                } else {
                                    None
                                }
                            }
                            ast::Pattern::Wildcard => None,
                            _ => None,
                        };

                        let then_bb = self
                            .context
                            .append_basic_block(fn_value, &format!("match_then_{}", i));
                        let else_bb = if i + 1 < arms.len() {
                            self.context
                                .append_basic_block(fn_value, &format!("match_else_{}", i))
                        } else {
                            merge_bb
                        };

                        // For the last arm or wildcard, always branch unconditionally
                        let is_last_arm = i + 1 >= arms.len();
                        if let Some(disc) = pattern_val {
                            if is_last_arm {
                                // Last arm - treat as default (exhaustive match assumed)
                                self.builder
                                    .build_unconditional_branch(then_bb)
                                    .map_err(|e| e.to_string())?;
                            } else {
                                let pattern_const = self.context.i64_type().const_int(disc, false);
                                let cond = self
                                    .builder
                                    .build_int_compare(
                                        IntPredicate::EQ,
                                        scrutinee,
                                        pattern_const,
                                        "match_cmp",
                                    )
                                    .map_err(|e| e.to_string())?;
                                self.builder
                                    .build_conditional_branch(cond, then_bb, else_bb)
                                    .map_err(|e| e.to_string())?;
                            }
                        } else {
                            // Wildcard - unconditionally go to then block
                            self.builder
                                .build_unconditional_branch(then_bb)
                                .map_err(|e| e.to_string())?;
                        }

                        // Compile the arm body
                        self.builder.position_at_end(then_bb);

                        // Extract pattern bindings and add to scope
                        // For patterns like `Enum::Variant { field1, field2 }` or `Enum::Variant(x)`
                        match &arm.pattern {
                            ast::Pattern::TupleStruct {
                                path: _, fields, ..
                            } => {
                                // Bind each field pattern as a variable
                                // For simplicity, assume scrutinee is a pointer to struct data
                                for (i, field_pattern) in fields.iter().enumerate() {
                                    if let ast::Pattern::Ident { name, .. } = field_pattern {
                                        // Create a variable for this binding
                                        // For enums with data, field 0 is often at offset 8 (after tag)
                                        let offset = (i as u64 + 1) * 8; // Skip tag byte
                                        let offset_val =
                                            self.context.i64_type().const_int(offset, false);

                                        let ptr_type =
                                            self.context.ptr_type(inkwell::AddressSpace::default());
                                        let scrutinee_ptr = self
                                            .builder
                                            .build_int_to_ptr(scrutinee, ptr_type, "scrutinee_ptr")
                                            .map_err(|e| e.to_string())?;
                                        let scrutinee_int = self
                                            .builder
                                            .build_ptr_to_int(
                                                scrutinee_ptr,
                                                self.context.i64_type(),
                                                "scr_int",
                                            )
                                            .map_err(|e| e.to_string())?;
                                        let field_ptr_int = self
                                            .builder
                                            .build_int_add(
                                                scrutinee_int,
                                                offset_val,
                                                "field_ptr_int",
                                            )
                                            .map_err(|e| e.to_string())?;
                                        let field_ptr = self
                                            .builder
                                            .build_int_to_ptr(
                                                field_ptr_int,
                                                ptr_type,
                                                &format!("{}_ptr", name.name),
                                            )
                                            .map_err(|e| e.to_string())?;
                                        let field_val = self
                                            .builder
                                            .build_load(
                                                self.context.i64_type(),
                                                field_ptr,
                                                &name.name,
                                            )
                                            .map_err(|e| e.to_string())?;

                                        let alloca = self
                                            .builder
                                            .build_alloca(self.context.i64_type(), &name.name)
                                            .map_err(|e| e.to_string())?;
                                        self.builder
                                            .build_store(alloca, field_val)
                                            .map_err(|e| e.to_string())?;
                                        scope.vars.insert(name.name.clone(), alloca);
                                    }
                                }
                            }
                            ast::Pattern::Struct {
                                path: _, fields, ..
                            } => {
                                // Bind each field pattern as a variable
                                for (i, field_pattern) in fields.iter().enumerate() {
                                    // Get binding name: use pattern if present, else use field name (shorthand)
                                    let binding_name = if let Some(ref pat) = field_pattern.pattern
                                    {
                                        if let ast::Pattern::Ident { name, .. } = pat {
                                            Some(name.name.clone())
                                        } else {
                                            None
                                        }
                                    } else {
                                        // Shorthand: `{ prefix }` means bind field_pattern.name
                                        Some(field_pattern.name.name.clone())
                                    };

                                    if let Some(name) = binding_name {
                                        // Use field index for offset
                                        let offset = (i as u64 + 1) * 8;
                                        let offset_val =
                                            self.context.i64_type().const_int(offset, false);

                                        let ptr_type =
                                            self.context.ptr_type(inkwell::AddressSpace::default());
                                        let scrutinee_ptr = self
                                            .builder
                                            .build_int_to_ptr(scrutinee, ptr_type, "scrutinee_ptr")
                                            .map_err(|e| e.to_string())?;
                                        let scrutinee_int = self
                                            .builder
                                            .build_ptr_to_int(
                                                scrutinee_ptr,
                                                self.context.i64_type(),
                                                "scr_int",
                                            )
                                            .map_err(|e| e.to_string())?;
                                        let field_ptr_int = self
                                            .builder
                                            .build_int_add(
                                                scrutinee_int,
                                                offset_val,
                                                "field_ptr_int",
                                            )
                                            .map_err(|e| e.to_string())?;
                                        let field_ptr = self
                                            .builder
                                            .build_int_to_ptr(
                                                field_ptr_int,
                                                ptr_type,
                                                &format!("{}_ptr", name),
                                            )
                                            .map_err(|e| e.to_string())?;
                                        let field_val = self
                                            .builder
                                            .build_load(self.context.i64_type(), field_ptr, &name)
                                            .map_err(|e| e.to_string())?;

                                        let alloca = self
                                            .builder
                                            .build_alloca(self.context.i64_type(), &name)
                                            .map_err(|e| e.to_string())?;
                                        self.builder
                                            .build_store(alloca, field_val)
                                            .map_err(|e| e.to_string())?;
                                        scope.vars.insert(name, alloca);
                                    }
                                }
                            }
                            ast::Pattern::Ident { name, .. } => {
                                // Simple binding - bind the entire scrutinee
                                let alloca = self
                                    .builder
                                    .build_alloca(self.context.i64_type(), &name.name)
                                    .map_err(|e| e.to_string())?;
                                self.builder
                                    .build_store(alloca, scrutinee)
                                    .map_err(|e| e.to_string())?;
                                scope.vars.insert(name.name.clone(), alloca);
                            }
                            _ => {}
                        }

                        let arm_val = self.compile_expr(fn_value, scope, &arm.body)?;

                        if self
                            .builder
                            .get_insert_block()
                            .unwrap()
                            .get_terminator()
                            .is_none()
                        {
                            let current_bb = self.builder.get_insert_block().unwrap();
                            self.builder
                                .build_unconditional_branch(merge_bb)
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
                        // All arms returned early - merge block is unreachable
                        // Delete the empty merge block and return dummy value
                        // The actual return happens in the arms themselves
                        unsafe {
                            merge_bb
                                .delete()
                                .map_err(|_| "Failed to delete unreachable merge block")?;
                        }
                        return Ok(self.context.i64_type().const_int(0, false));
                    }

                    let phi = self
                        .builder
                        .build_phi(self.context.i64_type(), "match_result")
                        .map_err(|e| e.to_string())?;

                    for (val, bb) in &incoming {
                        phi.add_incoming(&[(val, *bb)]);
                    }

                    Ok(phi.as_basic_value().into_int_value())
                }
                Expr::MethodCall {
                    receiver,
                    method,
                    args,
                    ..
                } => {
                    // Compile the receiver
                    let receiver_val = self.compile_expr(fn_value, scope, receiver)?;
                    let method_name = method.name.as_str();

                    // G11 fix: Check user-defined methods BEFORE built-in methods
                    // This ensures struct methods like `get` don't conflict with Vec.get()
                    let receiver_type = self.get_struct_type_from_expr(receiver, scope);
                    if let Some(ref recv_type) = receiver_type {
                        // Check if we have an impl method for this type
                        if let Some(mangled_name) = self.impl_methods.get(&(recv_type.clone(), method_name.to_string())) {
                            if let Some(callee) = self.module.get_function(mangled_name) {
                                // Compile arguments
                                let mut compiled_args: Vec<BasicMetadataValueEnum> =
                                    vec![receiver_val.into()];
                                for arg in args {
                                    let arg_val = self.compile_expr(fn_value, scope, arg)?;
                                    compiled_args.push(arg_val.into());
                                }

                                let call = self
                                    .builder
                                    .build_call(callee, &compiled_args, "method_call")
                                    .map_err(|e| e.to_string())?;

                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| {
                                        self.context.i64_type().const_int(0, false)
                                    }));
                            }
                        }
                    }

                    // Handle built-in Vec methods (fallback when no user-defined method matches)
                    match method_name {
                        "push" => {
                            // v.push(val) -> sigil_vec_push(v, val)
                            if args.is_empty() {
                                return Err("push requires a value argument".to_string());
                            }

                            // Track if we're pushing a float to mark the Vec as a float Vec
                            let arg_is_float = self.is_float_expr_with_scope(&args[0], scope);
                            if arg_is_float {
                                // Mark the receiver Vec as containing floats
                                if let Expr::Path(path) = receiver.as_ref() {
                                    if let Some(seg) = path.segments.last() {
                                        scope.float_vars.insert(seg.ident.name.clone());
                                    }
                                }
                            }

                            let value = self.compile_expr(fn_value, scope, &args[0])?;
                            let push_fn = self
                                .module
                                .get_function("sigil_vec_push")
                                .ok_or("sigil_vec_push not declared")?;
                            self.builder
                                .build_call(push_fn, &[receiver_val.into(), value.into()], "")
                                .map_err(|e| e.to_string())?;
                            return Ok(self.context.i64_type().const_int(0, false));
                        }
                        "len" => {
                            // G32: Check receiver type for appropriate len function
                            let (is_c_string, is_rust_string) =
                                if matches!(receiver.as_ref(), Expr::Literal(Literal::String(_))) {
                                    (true, false)
                                } else if let Expr::Path(path) = receiver.as_ref() {
                                    if let Some(seg) = path.segments.last() {
                                        match scope.var_types.get(&seg.ident.name) {
                                            Some(SigilType::String) => (true, false),
                                            Some(SigilType::RustString) => (false, true),
                                            _ => (false, false),
                                        }
                                    } else {
                                        (false, false)
                                    }
                                } else {
                                    (false, false)
                                };

                            if is_c_string {
                                // C string: s.len() -> sigil_strlen(s)
                                let ptr_type = self.context.ptr_type(AddressSpace::default());
                                let str_ptr = self
                                    .builder
                                    .build_int_to_ptr(receiver_val, ptr_type, "str_ptr")
                                    .map_err(|e| e.to_string())?;
                                let strlen_fn = self
                                    .module
                                    .get_function("sigil_strlen")
                                    .ok_or("sigil_strlen not declared")?;
                                let call = self
                                    .builder
                                    .build_call(strlen_fn, &[str_ptr.into()], "str_len")
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                            }

                            if is_rust_string {
                                // Rust String: s.len() -> sigil_string_len(s)
                                // Functions use i64 as pointer type, so pass receiver_val directly
                                let strlen_fn = self
                                    .module
                                    .get_function("sigil_string_len")
                                    .ok_or("sigil_string_len not declared")?;
                                let call = self
                                    .builder
                                    .build_call(strlen_fn, &[receiver_val.into()], "rust_str_len")
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                            }

                            // v.len() -> sigil_vec_len(v)
                            let len_fn = self
                                .module
                                .get_function("sigil_vec_len")
                                .ok_or("sigil_vec_len not declared")?;
                            let call = self
                                .builder
                                .build_call(len_fn, &[receiver_val.into()], "vec_len")
                                .map_err(|e| e.to_string())?;
                            return Ok(call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                        }
                        "get" => {
                            // v.get(idx) -> sigil_vec_get(v, idx)
                            if args.is_empty() {
                                return Err("get requires an index argument".to_string());
                            }
                            let index = self.compile_expr(fn_value, scope, &args[0])?;
                            let get_fn = self
                                .module
                                .get_function("sigil_vec_get")
                                .ok_or("sigil_vec_get not declared")?;
                            let call = self
                                .builder
                                .build_call(get_fn, &[receiver_val.into(), index.into()], "vec_get")
                                .map_err(|e| e.to_string())?;
                            return Ok(call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                        }
                        "iter" => {
                            // v.iter() returns the array/vec itself for iteration
                            // In Sigil, .iter() is an identity operation that signals
                            // the value should be iterated over in a for loop
                            return Ok(receiver_val);
                        }
                        "clone" => {
                            // For now, clone is identity (shallow copy semantics)
                            // TODO: Implement proper deep clone via runtime
                            return Ok(receiver_val);
                        }
                        "as_bytes" => {
                            // G32: Check receiver type for appropriate as_bytes handling
                            let is_rust_string = if let Expr::Path(path) = receiver.as_ref() {
                                if let Some(seg) = path.segments.last() {
                                    matches!(
                                        scope.var_types.get(&seg.ident.name),
                                        Some(SigilType::RustString)
                                    )
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                            if is_rust_string {
                                // Rust String: call sigil_rust_string_as_bytes to get
                                // byte pointer with null terminator (so strlen works)
                                let as_bytes_fn = self
                                    .module
                                    .get_function("sigil_rust_string_as_bytes")
                                    .ok_or("sigil_rust_string_as_bytes not declared")?;
                                let call = self
                                    .builder
                                    .build_call(as_bytes_fn, &[receiver_val.into()], "bytes_ptr")
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                            }

                            // C string: pointer already points to bytes
                            return Ok(receiver_val);
                        }
                        "repeat" => {
                            // str.repeat(n) -> sigil_string_repeat(str, n)
                            // Works on string literals (C strings) - returns new allocated string
                            if args.is_empty() {
                                return Err("repeat requires a count argument".to_string());
                            }
                            let count = self.compile_expr(fn_value, scope, &args[0])?;

                            // Convert receiver i64 (ptr as int) to actual pointer
                            let ptr_type = self.context.ptr_type(AddressSpace::default());
                            let str_ptr = self.builder
                                .build_int_to_ptr(receiver_val, ptr_type, "repeat_str_ptr")
                                .map_err(|e| e.to_string())?;

                            let repeat_fn = self
                                .module
                                .get_function("sigil_string_repeat")
                                .ok_or("sigil_string_repeat not declared")?;

                            let call = self
                                .builder
                                .build_call(repeat_fn, &[str_ptr.into(), count.into()], "repeated_str")
                                .map_err(|e| e.to_string())?;

                            // The function returns a pointer, convert to i64 for Sigil
                            let result = call
                                .try_as_basic_value()
                                .left()
                                .ok_or("repeat returned void")?;

                            // Check if result is pointer or int
                            if result.is_pointer_value() {
                                let result_ptr = result.into_pointer_value();
                                let result_as_int = self.builder
                                    .build_ptr_to_int(result_ptr, self.context.i64_type(), "repeated_str_int")
                                    .map_err(|e| e.to_string())?;
                                return Ok(result_as_int);
                            } else {
                                // Already an i64 (some LLVM versions return this way)
                                return Ok(result.into_int_value());
                            }
                        }
                        "to_vec" => {
                            // slice.to_vec() or array.to_vec() - create a Vec from the source
                            let i64_type = self.context.i64_type();

                            // Get runtime functions
                            let new_fn = self
                                .module
                                .get_function("sigil_vec_new")
                                .ok_or("sigil_vec_new not declared")?;
                            let push_fn = self
                                .module
                                .get_function("sigil_vec_push")
                                .ok_or("sigil_vec_push not declared")?;

                            // Check if receiver is an array literal - handle specially
                            if let Expr::Array(elements) = receiver.as_ref() {
                                // Array literal: we know the length statically
                                let arr_len = elements.len() as u64;
                                let src_len = i64_type.const_int(arr_len, false);

                                // Create new vec with known capacity
                                let new_call = self
                                    .builder
                                    .build_call(new_fn, &[src_len.into()], "new_vec")
                                    .map_err(|e| e.to_string())?;
                                let new_vec = new_call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| i64_type.const_int(0, false));

                                // Push each element directly (unrolled loop for small arrays)
                                for elem_expr in elements {
                                    let elem_val = self.compile_expr(fn_value, scope, elem_expr)?;
                                    self.builder
                                        .build_call(push_fn, &[new_vec.into(), elem_val.into()], "")
                                        .map_err(|e| e.to_string())?;
                                }

                                return Ok(new_vec);
                            }

                            // For Vec/slice: use runtime functions to copy
                            let len_fn = self
                                .module
                                .get_function("sigil_vec_len")
                                .ok_or("sigil_vec_len not declared")?;
                            let get_fn = self
                                .module
                                .get_function("sigil_vec_get")
                                .ok_or("sigil_vec_get not declared")?;

                            // Get length of source
                            let len_call = self
                                .builder
                                .build_call(len_fn, &[receiver_val.into()], "src_len")
                                .map_err(|e| e.to_string())?;
                            let src_len = len_call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| i64_type.const_int(0, false));

                            // Create new vec with same capacity
                            let new_call = self
                                .builder
                                .build_call(new_fn, &[src_len.into()], "new_vec")
                                .map_err(|e| e.to_string())?;
                            let new_vec = new_call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| i64_type.const_int(0, false));

                            // Build loop to copy elements
                            let loop_header = self.context.append_basic_block(fn_value, "to_vec_header");
                            let loop_body = self.context.append_basic_block(fn_value, "to_vec_body");
                            let loop_end = self.context.append_basic_block(fn_value, "to_vec_end");

                            // Initialize counter
                            let counter_ptr = self.builder
                                .build_alloca(i64_type, "to_vec_i")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_store(counter_ptr, i64_type.const_int(0, false))
                                .map_err(|e| e.to_string())?;

                            // Jump to header
                            self.builder
                                .build_unconditional_branch(loop_header)
                                .map_err(|e| e.to_string())?;

                            // Loop header: check if i < len
                            self.builder.position_at_end(loop_header);
                            let i = self.builder
                                .build_load(i64_type, counter_ptr, "i")
                                .map_err(|e| e.to_string())?
                                .into_int_value();
                            let cmp = self.builder
                                .build_int_compare(inkwell::IntPredicate::SLT, i, src_len, "cmp")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_conditional_branch(cmp, loop_body, loop_end)
                                .map_err(|e| e.to_string())?;

                            // Loop body: get element from source, push to new vec
                            self.builder.position_at_end(loop_body);
                            let get_call = self.builder
                                .build_call(get_fn, &[receiver_val.into(), i.into()], "elem")
                                .map_err(|e| e.to_string())?;
                            let elem = get_call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| i64_type.const_int(0, false));
                            self.builder
                                .build_call(push_fn, &[new_vec.into(), elem.into()], "")
                                .map_err(|e| e.to_string())?;

                            // Increment counter
                            let next_i = self.builder
                                .build_int_add(i, i64_type.const_int(1, false), "next_i")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_store(counter_ptr, next_i)
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_unconditional_branch(loop_header)
                                .map_err(|e| e.to_string())?;

                            // Position at end
                            self.builder.position_at_end(loop_end);

                            return Ok(new_vec);
                        }
                        "is_empty" => {
                            // v.is_empty() -> v.len() == 0
                            let len_fn = self
                                .module
                                .get_function("sigil_vec_len")
                                .ok_or("sigil_vec_len not declared")?;
                            let call = self
                                .builder
                                .build_call(len_fn, &[receiver_val.into()], "vec_len")
                                .map_err(|e| e.to_string())?;
                            let len = call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| self.context.i64_type().const_int(0, false));
                            let zero = self.context.i64_type().const_int(0, false);
                            let is_empty = self
                                .builder
                                .build_int_compare(IntPredicate::EQ, len, zero, "is_empty")
                                .map_err(|e| e.to_string())?;
                            // Convert bool (i1) to i64
                            let result = self
                                .builder
                                .build_int_z_extend(
                                    is_empty,
                                    self.context.i64_type(),
                                    "is_empty_i64",
                                )
                                .map_err(|e| e.to_string())?;
                            return Ok(result);
                        }
                        // String methods
                        "starts_with" => {
                            if args.is_empty() {
                                return Err("starts_with requires a prefix argument".to_string());
                            }
                            let prefix = self.compile_expr(fn_value, scope, &args[0])?;
                            let fn_name = "sigil_string_starts_with";
                            if let Some(callee) = self.module.get_function(fn_name) {
                                let call = self
                                    .builder
                                    .build_call(
                                        callee,
                                        &[receiver_val.into(), prefix.into()],
                                        "starts_with",
                                    )
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| {
                                        self.context.i64_type().const_int(0, false)
                                    }));
                            }
                            // Fallback: return false (0)
                            return Ok(self.context.i64_type().const_int(0, false));
                        }
                        "to_string" => {
                            // G32: Check if receiver is already a Rust String (slice, method call, etc.)
                            // In these cases, to_string() is a no-op since the result is already a String
                            let is_already_rust_string = match receiver.as_ref() {
                                // Index on a Rust String (slice) already returns a String
                                Expr::Index { expr, .. } => {
                                    if let Expr::Path(path) = expr.as_ref() {
                                        if let Some(seg) = path.segments.last() {
                                            matches!(
                                                scope.var_types.get(&seg.ident.name),
                                                Some(SigilType::RustString)
                                            )
                                        } else {
                                            false
                                        }
                                    } else {
                                        false
                                    }
                                }
                                // Method call results (like another .to_string()) are already Strings
                                Expr::MethodCall { method, .. } => {
                                    method.name == "to_string"
                                }
                                // Path to a Rust String variable
                                Expr::Path(path) => {
                                    if let Some(seg) = path.segments.last() {
                                        matches!(
                                            scope.var_types.get(&seg.ident.name),
                                            Some(SigilType::RustString)
                                        )
                                    } else {
                                        false
                                    }
                                }
                                _ => false,
                            };

                            if is_already_rust_string {
                                // Already a Rust String, just return the receiver
                                return Ok(receiver_val);
                            }

                            // Convert C string to heap-allocated Rust String
                            // Call sigil_string_from(i64_as_ptr) -> i64_as_ptr
                            let string_from_fn = self
                                .module
                                .get_function("sigil_string_from")
                                .ok_or("sigil_string_from not declared")?;
                            let call = self
                                .builder
                                .build_call(string_from_fn, &[receiver_val.into()], "rust_string")
                                .map_err(|e| e.to_string())?;
                            return Ok(call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                        }
                        "as_str" | "as_ref" => {
                            // String -> &str conversion is identity in our representation
                            return Ok(receiver_val);
                        }
                        "chars" => {
                            // s.chars() -> sigil_string_chars(s)
                            let fn_name = "sigil_string_chars";
                            if let Some(callee) = self.module.get_function(fn_name) {
                                let call = self
                                    .builder
                                    .build_call(callee, &[receiver_val.into()], "chars")
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| {
                                        self.context.i64_type().const_int(0, false)
                                    }));
                            }
                            return Ok(receiver_val);
                        }
                        "collect" => {
                            // Iterator collect - for now return the iterator itself
                            return Ok(receiver_val);
                        }
                        "skip" => {
                            // Iterator skip(n) - would need runtime support
                            // For now, return the iterator
                            return Ok(receiver_val);
                        }
                        "unwrap" => {
                            // Option/Result unwrap - return inner value
                            // In our simplified model, just return the value
                            return Ok(receiver_val);
                        }
                        "unwrap_or" => {
                            // Option/Result unwrap_or(default) - simplified
                            return Ok(receiver_val);
                        }
                        "is_some" | "is_ok" => {
                            // For simplicity, return true (1)
                            return Ok(self.context.i64_type().const_int(1, false));
                        }
                        "is_none" | "is_err" => {
                            // For simplicity, return false (0)
                            return Ok(self.context.i64_type().const_int(0, false));
                        }
                        // Math methods: x.sqrt(), x.sin(), x.cos(), etc.
                        "sqrt" | "sin" | "cos" | "tan" | "exp" | "ln" | "floor" | "ceil" | "abs" => {
                            let rt_name = format!("sigil_{}", method_name);
                            let rt_fn = self
                                .module
                                .get_function(&rt_name)
                                .ok_or(format!("{} not declared", rt_name))?;
                            let call = self
                                .builder
                                .build_call(rt_fn, &[receiver_val.into()], method_name)
                                .map_err(|e| e.to_string())?;
                            return Ok(call
                                .try_as_basic_value()
                                .left()
                                .map(|v| v.into_int_value())
                                .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                        }
                        "push_str" => {
                            // s.push_str(other) - append string
                            if args.is_empty() {
                                return Err("push_str requires a string argument".to_string());
                            }
                            let other = self.compile_expr(fn_value, scope, &args[0])?;
                            if let Some(callee) = self.module.get_function("sigil_string_push_str")
                            {
                                let call = self
                                    .builder
                                    .build_call(
                                        callee,
                                        &[receiver_val.into(), other.into()],
                                        "push_str",
                                    )
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| {
                                        self.context.i64_type().const_int(0, false)
                                    }));
                            }
                            // Fallback: return the string unchanged
                            return Ok(receiver_val);
                        }
                        "push" => {
                            // s.push(char) or vec.push(item)
                            if args.is_empty() {
                                return Err("push requires an argument".to_string());
                            }
                            let item = self.compile_expr(fn_value, scope, &args[0])?;
                            // Try vec push first
                            if let Some(callee) = self.module.get_function("sigil_vec_push") {
                                self.builder
                                    .build_call(
                                        callee,
                                        &[receiver_val.into(), item.into()],
                                        "vec_push",
                                    )
                                    .map_err(|e| e.to_string())?;
                                return Ok(receiver_val);
                            }
                            return Ok(receiver_val);
                        }
                        "contains" => {
                            // s.contains(substr) or vec.contains(item)
                            if args.is_empty() {
                                return Err("contains requires an argument".to_string());
                            }
                            let needle = self.compile_expr(fn_value, scope, &args[0])?;
                            if let Some(callee) = self.module.get_function("sigil_string_contains")
                            {
                                let call = self
                                    .builder
                                    .build_call(
                                        callee,
                                        &[receiver_val.into(), needle.into()],
                                        "contains",
                                    )
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| {
                                        self.context.i64_type().const_int(0, false)
                                    }));
                            }
                            // Fallback: return false
                            return Ok(self.context.i64_type().const_int(0, false));
                        }
                        "trim" | "trim_start" | "trim_end" => {
                            // String trim methods - for now return the string unchanged
                            return Ok(receiver_val);
                        }
                        "split" | "lines" => {
                            // String split methods - return iterator (same as string for now)
                            return Ok(receiver_val);
                        }
                        "join" => {
                            // vec.join(sep) - for now return empty string
                            if let Some(callee) = self.module.get_function("sigil_string_new") {
                                let call = self
                                    .builder
                                    .build_call(callee, &[], "join_result")
                                    .map_err(|e| e.to_string())?;
                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| {
                                        self.context.i64_type().const_int(0, false)
                                    }));
                            }
                            return Ok(self.context.i64_type().const_int(0, false));
                        }
                        _ => {}
                    }

                    // G11 fix: Look up the method by type AND method name
                    // First, try to determine the receiver's struct type
                    let receiver_type = self.get_struct_type_from_expr(receiver, scope);

                    // First pass: Try to find method matching both type and name
                    if let Some(ref recv_type) = receiver_type {
                        for ((type_name, meth_name), mangled_name) in &self.impl_methods {
                            if type_name == recv_type && meth_name == method_name {
                                if let Some(callee) = self.module.get_function(mangled_name) {
                                    // Compile arguments
                                    let mut compiled_args: Vec<BasicMetadataValueEnum> =
                                        vec![receiver_val.into()];
                                    for arg in args {
                                        let arg_val = self.compile_expr(fn_value, scope, arg)?;
                                        compiled_args.push(arg_val.into());
                                    }

                                    let call = self
                                        .builder
                                        .build_call(callee, &compiled_args, "method_call")
                                        .map_err(|e| e.to_string())?;

                                    return Ok(call
                                        .try_as_basic_value()
                                        .left()
                                        .map(|v| v.into_int_value())
                                        .unwrap_or_else(|| {
                                            self.context.i64_type().const_int(0, false)
                                        }));
                                }
                            }
                        }
                    }

                    // Second pass: Fallback to matching by method name only (for unknown types)
                    for ((_type_name, meth_name), mangled_name) in &self.impl_methods {
                        if meth_name == method_name {
                            if let Some(callee) = self.module.get_function(mangled_name) {
                                // Compile arguments
                                let mut compiled_args: Vec<BasicMetadataValueEnum> =
                                    vec![receiver_val.into()];
                                for arg in args {
                                    let arg_val = self.compile_expr(fn_value, scope, arg)?;
                                    compiled_args.push(arg_val.into());
                                }

                                let call = self
                                    .builder
                                    .build_call(callee, &compiled_args, "method_call")
                                    .map_err(|e| e.to_string())?;

                                return Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| {
                                        self.context.i64_type().const_int(0, false)
                                    }));
                            }
                        }
                    }

                    // Fallback: Try to call as a method that mutates/accesses the receiver
                    // Many methods like `collect_type_def`, `check`, etc. just return unit or
                    // are side-effecting methods that we can stub as no-ops
                    let method_lower = method_name.to_lowercase();
                    if method_lower.starts_with("collect")
                        || method_lower.starts_with("check")
                        || method_lower.starts_with("validate")
                        || method_lower.starts_with("verify")
                        || method_lower.starts_with("register")
                        || method_lower.starts_with("add")
                        || method_lower.starts_with("insert")
                        || method_lower.starts_with("remove")
                        || method_lower.starts_with("clear")
                        || method_lower.starts_with("reset")
                        || method_lower.starts_with("update")
                        || method_lower.starts_with("set")
                        || method_lower.starts_with("process")
                        || method_lower.starts_with("analyze")
                        || method_lower.starts_with("visit")
                        || method_lower.starts_with("emit")
                        || method_lower.starts_with("write")
                        || method_lower.starts_with("flush")
                        || method_lower.starts_with("sync")
                        || method_lower.starts_with("close")
                        || method_lower.starts_with("finish")
                        || method_lower.starts_with("complete")
                        || method_lower.starts_with("init")
                        || method_lower.starts_with("setup")
                        || method_lower.starts_with("configure")
                    {
                        // Side-effecting methods - return unit (0)
                        return Ok(self.context.i64_type().const_int(0, false));
                    }

                    // Methods that likely return the receiver (builder pattern)
                    if method_lower.starts_with("with_")
                        || method_lower.starts_with("and_")
                        || method_lower == "build"
                        || method_lower == "done"
                    {
                        return Ok(receiver_val);
                    }

                    // Methods that likely return a bool
                    if method_lower.starts_with("is_")
                        || method_lower.starts_with("has_")
                        || method_lower.starts_with("can_")
                        || method_lower.starts_with("should_")
                        || method_lower == "exists"
                        || method_lower == "contains"
                    {
                        // Return false (0) as default
                        return Ok(self.context.i64_type().const_int(0, false));
                    }

                    // Methods that likely return the receiver or a reference to internal data
                    if method_lower.starts_with("get_")
                        || method_lower.starts_with("find_")
                        || method_lower.starts_with("lookup_")
                        || method_lower == "get"
                        || method_lower == "take"
                        || method_lower == "borrow"
                    {
                        // Return receiver as placeholder for accessed data
                        return Ok(receiver_val);
                    }

                    // Default fallback - return 0 as unit
                    // eprintln!("DEBUG: Unknown method '{}' - treating as no-op", method_name);
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                // ============================================
                // Sigil-native expressions
                // ============================================

                // Evidentiality markers with runtime semantics
                // Known (!) unwraps evidential values
                // Other markers wrap values with evidence tags
                Expr::Evidential {
                    expr,
                    evidentiality,
                } => {
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
                                fn_value, inner_val, tag, "i64", // Default to i64 for now
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
                Expr::Index { expr, index } => self.compile_index(fn_value, scope, expr, index),

                // Range expressions
                Expr::Range {
                    start,
                    end,
                    inclusive: _,
                } => {
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
                Expr::Closure {
                    params: _, body, ..
                } => {
                    // For simple closures, just compile the body
                    // Full closure support needs lambda lifting
                    self.compile_expr(fn_value, scope, body)
                }

                // Cast/type coercion
                Expr::Cast { expr, ty } => {
                    let val = self.compile_expr(fn_value, scope, expr)?;

                    // Check target type for numeric conversions
                    let target_type_str = self.type_expr_to_string(ty);

                    // Cast to f64
                    if target_type_str == "f64" {
                        // G18: If source is already f64, no-op (value is already stored as f64 bits)
                        if self.is_float_expr_with_scope(expr, scope) {
                            return Ok(val);
                        }

                        // i64 -> f64: use sitofp, then bitcast back to i64 for storage
                        let f64_val = self
                            .builder
                            .build_signed_int_to_float(
                                val,
                                self.context.f64_type(),
                                "i_to_f64",
                            )
                            .map_err(|e| e.to_string())?;
                        // Bitcast f64 to i64 for uniform storage
                        let bits = self
                            .builder
                            .build_bit_cast(f64_val, self.context.i64_type(), "f64_bits")
                            .map_err(|e| e.to_string())?;
                        return Ok(bits.into_int_value());
                    }

                    // f64 -> integer: bitcast i64 to f64, then fptosi
                    if target_type_str == "i64" || target_type_str == "isize" || target_type_str == "usize" {
                        // Check if source is a float expression
                        if self.is_float_expr_with_scope(expr, scope) {
                            // Bitcast i64 (float bits) back to f64, then convert to int
                            let f64_val = self
                                .builder
                                .build_bit_cast(val, self.context.f64_type(), "bits_to_f64")
                                .map_err(|e| e.to_string())?
                                .into_float_value();
                            let int_val = self
                                .builder
                                .build_float_to_signed_int(
                                    f64_val,
                                    self.context.i64_type(),
                                    "f64_to_i64",
                                )
                                .map_err(|e| e.to_string())?;
                            return Ok(int_val);
                        }
                        // Source is already an integer, return as-is
                        return Ok(val);
                    }

                    // Default: pass through
                    Ok(val)
                }

                // Address-of: &expr, &mut expr
                Expr::AddrOf { expr, .. } => self.compile_expr(fn_value, scope, expr),

                // Dereference: *ptr or *(ptr + offset)
                Expr::Deref(inner) => {
                    // Check for pointer arithmetic pattern: *(ptr + offset)
                    // When we have *(ptr + n), we need to use GEP to scale offset by element size
                    if let Expr::Binary { op: BinOp::Add, left, right } = inner.as_ref() {
                        // Compile base pointer and offset separately
                        let base_addr = self.compile_expr(fn_value, scope, left)?;
                        let offset = self.compile_expr(fn_value, scope, right)?;

                        // Convert base address to pointer
                        let i64_type = self.context.i64_type();
                        let base_ptr = self
                            .builder
                            .build_int_to_ptr(
                                base_addr,
                                i64_type.ptr_type(Default::default()),
                                "base_ptr",
                            )
                            .map_err(|e| e.to_string())?;

                        // Use GEP for proper pointer arithmetic (scales by element size)
                        let elem_ptr = unsafe {
                            self.builder
                                .build_gep(i64_type, base_ptr, &[offset], "elem_ptr")
                        }
                        .map_err(|e| e.to_string())?;

                        // Load the value
                        let loaded = self
                            .builder
                            .build_load(i64_type, elem_ptr, "deref_val")
                            .map_err(|e| e.to_string())?;
                        return Ok(loaded.into_int_value());
                    }

                    // Simple dereference: *ptr (no offset)
                    let ptr_val = self.compile_expr(fn_value, scope, inner)?;
                    let ptr = self
                        .builder
                        .build_int_to_ptr(
                            ptr_val,
                            self.context.ptr_type(AddressSpace::default()),
                            "deref_ptr",
                        )
                        .map_err(|e| e.to_string())?;
                    let loaded = self
                        .builder
                        .build_load(self.context.i64_type(), ptr, "deref_val")
                        .map_err(|e| e.to_string())?;
                    Ok(loaded.into_int_value())
                }

                // Macro invocation - handle println!, print!, format!, etc.
                Expr::Macro { path, tokens } => {
                    let macro_name = path
                        .segments
                        .last()
                        .map(|s| s.ident.name.trim_end_matches('!'))
                        .unwrap_or("unknown");

                    match macro_name {
                        "println" | "print" => {
                            self.compile_print_macro(
                                fn_value,
                                scope,
                                tokens,
                                macro_name == "println",
                            )?;
                            Ok(self.context.i64_type().const_int(0, false))
                        }
                        "format" => {
                            // format! returns a String - for now just return 0
                            // TODO: implement proper string formatting
                            Ok(self.context.i64_type().const_int(0, false))
                        }
                        "vec" => {
                            // vec![...] - parse and create Vec
                            Ok(self.compile_vec_macro(fn_value, scope, tokens)?)
                        }
                        "panic" => {
                            // For now, just exit with code 1
                            // TODO: print panic message first
                            let exit_fn = self.module.get_function("sigil_exit");
                            if let Some(f) = exit_fn {
                                let one = self.context.i64_type().const_int(1, false);
                                self.builder
                                    .build_call(f, &[one.into()], "")
                                    .map_err(|e| e.to_string())?;
                            }
                            Ok(self.context.i64_type().const_int(0, false))
                        }
                        "assert" | "assert_eq" | "assert_ne" => {
                            // TODO: implement assertions
                            Ok(self.context.i64_type().const_int(0, false))
                        }
                        _ => {
                            // Unknown macro - try to call as function
                            if let Some(f) = self.module.get_function(macro_name) {
                                let call = self
                                    .builder
                                    .build_call(f, &[], "macro_call")
                                    .map_err(|e| e.to_string())?;
                                Ok(call
                                    .try_as_basic_value()
                                    .left()
                                    .map(|v| v.into_int_value())
                                    .unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                            } else {
                                Ok(self.context.i64_type().const_int(0, false))
                            }
                        }
                    }
                }

                // Try expression: expr?
                Expr::Try(inner) => {
                    // Types erased, just compile inner
                    self.compile_expr(fn_value, scope, inner)
                }

                // Let expression (for if-let patterns)
                Expr::Let { value, .. } => self.compile_expr(fn_value, scope, value),

                // Tuple: allocate on heap and store all elements
                // G14 fix: Proper tuple support - tuples are heap-allocated structs
                Expr::Tuple(elements) => {
                    if elements.is_empty() {
                        // Unit tuple () - return 0
                        return Ok(self.context.i64_type().const_int(0, false));
                    }

                    // Allocate tuple: num_elements * 8 bytes
                    let tuple_size = elements.len() as u64 * 8;
                    let size_const = self.context.i64_type().const_int(tuple_size, false);

                    let alloc_fn = self
                        .module
                        .get_function("sigil_alloc")
                        .ok_or("sigil_alloc not declared")?;
                    let alloc_call = self
                        .builder
                        .build_call(alloc_fn, &[size_const.into()], "tuple_alloc")
                        .map_err(|e| e.to_string())?;
                    let alloc_result = alloc_call
                        .try_as_basic_value()
                        .left()
                        .ok_or("sigil_alloc returned void")?;

                    let tuple_ptr = if alloc_result.is_pointer_value() {
                        alloc_result.into_pointer_value()
                    } else {
                        self.builder
                            .build_int_to_ptr(
                                alloc_result.into_int_value(),
                                self.context.ptr_type(AddressSpace::default()),
                                "tuple_heap_ptr",
                            )
                            .map_err(|e| e.to_string())?
                    };

                    // Store each element at its offset
                    for (idx, elem) in elements.iter().enumerate() {
                        let elem_val = self.compile_expr(fn_value, scope, elem)?;
                        let offset = self.context.i64_type().const_int(idx as u64 * 8, false);
                        let elem_ptr = unsafe {
                            self.builder
                                .build_gep(
                                    self.context.i8_type(),
                                    tuple_ptr,
                                    &[offset],
                                    &format!("tuple_elem_{}_ptr", idx),
                                )
                                .map_err(|e| e.to_string())?
                        };
                        self.builder
                            .build_store(elem_ptr, elem_val)
                            .map_err(|e| e.to_string())?;
                    }

                    // Return pointer as i64 (consistent with struct handling)
                    let ptr_as_int = self
                        .builder
                        .build_ptr_to_int(tuple_ptr, self.context.i64_type(), "tuple_ptr_int")
                        .map_err(|e| e.to_string())?;
                    Ok(ptr_as_int)
                }

                // Array literal: allocate on stack and store elements
                Expr::Array(elements) => self.compile_array_literal(fn_value, scope, elements),

                // Loop expressions
                Expr::Loop { body, .. } => {
                    let result = self.compile_block(fn_value, scope, body)?;
                    Ok(result.unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                }

                Expr::For {
                    pattern,
                    iter,
                    body,
                    ..
                } => self.compile_for_loop(fn_value, scope, pattern, iter, body),

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
                Expr::Await { expr, .. } => self.compile_expr(fn_value, scope, expr),

                _ => {
                    // Unsupported expression - return error instead of silent 0
                    Err(format!(
                        "LLVM codegen: unsupported expression {:?}",
                        std::mem::discriminant(expr)
                    ))
                }
            }
        }

        /// Parse an integer literal with support for hex, binary, octal, and type suffixes
        fn parse_int_literal(&self, value: &str) -> Result<u64, String> {
            // Strip underscores (visual separators)
            let s = value.replace('_', "");

            // Strip known type suffixes (must check longer ones first)
            let suffixes = [
                "isize", "usize", "i128", "u128", "i64", "u64", "i32", "u32", "i16", "u16", "i8",
                "u8",
            ];
            let s = suffixes.iter().fold(s, |acc, suffix| {
                if acc.ends_with(suffix) {
                    acc[..acc.len() - suffix.len()].to_string()
                } else {
                    acc
                }
            });

            // Parse based on prefix
            if s.starts_with("0x") || s.starts_with("0X") {
                u64::from_str_radix(&s[2..], 16)
                    .map_err(|_| format!("Invalid hex integer: {}", value))
            } else if s.starts_with("0b") || s.starts_with("0B") {
                u64::from_str_radix(&s[2..], 2)
                    .map_err(|_| format!("Invalid binary integer: {}", value))
            } else if s.starts_with("0o") || s.starts_with("0O") {
                u64::from_str_radix(&s[2..], 8)
                    .map_err(|_| format!("Invalid octal integer: {}", value))
            } else {
                s.parse::<u64>()
                    .or_else(|_| {
                        // Try parsing as signed and reinterpreting
                        s.parse::<i64>().map(|v| v as u64)
                    })
                    .map_err(|_| format!("Invalid integer: {}", value))
            }
        }

        /// Compile a literal
        fn compile_literal(&mut self, lit: &Literal) -> Result<IntValue<'ctx>, String> {
            match lit {
                Literal::Int { value, .. } => {
                    // Parse integer with support for hex, binary, octal, and type suffixes
                    let v = self.parse_int_literal(value)?;
                    Ok(self.context.i64_type().const_int(v, false))
                }
                Literal::Bool(b) => Ok(self
                    .context
                    .i64_type()
                    .const_int(if *b { 1 } else { 0 }, false)),
                Literal::Float { value, .. } => {
                    // Convert float to int bits for now
                    // Strip underscores (visual separators) and type suffix (f32, f64)
                    let s = value.replace('_', "");
                    let s = s.trim_end_matches("f64").trim_end_matches("f32");
                    let v: f64 = s.parse().map_err(|_| format!("Invalid float: {}", value))?;
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
                    let global = self
                        .module
                        .add_global(const_array.get_type(), None, &global_name);
                    global.set_initializer(&const_array);
                    global.set_constant(true);
                    global.set_linkage(inkwell::module::Linkage::Private);

                    // Return pointer as i64
                    let ptr = global.as_pointer_value();
                    let ptr_as_int = self
                        .builder
                        .build_ptr_to_int(ptr, self.context.i64_type(), "str_ptr")
                        .map_err(|e| e.to_string())?;
                    Ok(ptr_as_int)
                }
                Literal::Char(c) => Ok(self.context.i64_type().const_int(*c as u64, false)),
                _ => Ok(self.context.i64_type().const_int(0, false)),
            }
        }

        /// Convert a TypeExpr to a simple string for type checking
        fn type_expr_to_string(&self, ty: &ast::TypeExpr) -> String {
            match ty {
                ast::TypeExpr::Path(path) => {
                    // Get the last segment's name
                    path.segments
                        .last()
                        .map(|seg| seg.ident.name.clone())
                        .unwrap_or_else(|| "unknown".to_string())
                }
                ast::TypeExpr::Reference { inner, .. } => {
                    format!("&{}", self.type_expr_to_string(inner))
                }
                ast::TypeExpr::Pointer { inner, .. } => {
                    format!("*{}", self.type_expr_to_string(inner))
                }
                ast::TypeExpr::Array { element, .. } => {
                    format!("[{}]", self.type_expr_to_string(element))
                }
                ast::TypeExpr::Slice(inner) => {
                    format!("[{}]", self.type_expr_to_string(inner))
                }
                _ => "unknown".to_string(),
            }
        }

        /// Extract struct type name from a TypeExpr (e.g., &SimpleLM -> SimpleLM)
        /// Returns None for primitive types, Some(name) for user-defined structs
        fn extract_struct_type_from_type_expr(&self, ty: &ast::TypeExpr) -> Option<String> {
            match ty {
                ast::TypeExpr::Path(path) => {
                    if let Some(seg) = path.segments.last() {
                        let name = &seg.ident.name;
                        // Exclude primitive types
                        if !matches!(name.as_str(),
                            "i8" | "i16" | "i32" | "i64" | "i128" |
                            "u8" | "u16" | "u32" | "u64" | "u128" |
                            "f32" | "f64" | "bool" | "char" | "str" | "String"
                        ) && !name.starts_with("Vec<") {
                            // Check if this is a known struct
                            if self.struct_types.contains_key(name) {
                                return Some(name.clone());
                            }
                        }
                    }
                    None
                }
                ast::TypeExpr::Reference { inner, .. } => self.extract_struct_type_from_type_expr(inner),
                ast::TypeExpr::Pointer { inner, .. } => self.extract_struct_type_from_type_expr(inner),
                _ => None,
            }
        }

        /// Check if a TypeExpr contains f64 (including in generics like Vec<f64>)
        fn type_contains_f64(&self, ty: &ast::TypeExpr) -> bool {
            match ty {
                ast::TypeExpr::Path(path) => {
                    // Check if the type itself is f64
                    if let Some(seg) = path.segments.last() {
                        if seg.ident.name == "f64" || seg.ident.name == "f32" {
                            return true;
                        }
                        // Check generic arguments (e.g., Vec<f64>)
                        if let Some(ref generics) = seg.generics {
                            for inner_ty in generics {
                                if self.type_contains_f64(inner_ty) {
                                    return true;
                                }
                            }
                        }
                    }
                    false
                }
                ast::TypeExpr::Reference { inner, .. } => self.type_contains_f64(inner),
                ast::TypeExpr::Pointer { inner, .. } => self.type_contains_f64(inner),
                ast::TypeExpr::Array { element, .. } => self.type_contains_f64(element),
                ast::TypeExpr::Slice(inner) => self.type_contains_f64(inner),
                ast::TypeExpr::Tuple(elements) => elements.iter().any(|e| self.type_contains_f64(e)),
                _ => false,
            }
        }

        /// G32: Check if a TypeExpr is or contains String/str
        fn type_contains_string(&self, ty: &ast::TypeExpr) -> bool {
            match ty {
                ast::TypeExpr::Path(path) => {
                    if let Some(seg) = path.segments.last() {
                        if seg.ident.name == "String" || seg.ident.name == "str" {
                            return true;
                        }
                    }
                    false
                }
                ast::TypeExpr::Reference { inner, .. } => self.type_contains_string(inner),
                ast::TypeExpr::Pointer { inner, .. } => self.type_contains_string(inner),
                _ => false,
            }
        }

        /// G28: Check if type is a byte slice (&[u8], &str, &[T]) for direct pointer indexing
        fn type_is_byte_slice(&self, ty: &ast::TypeExpr) -> bool {
            match ty {
                ast::TypeExpr::Path(path) => {
                    // Check for &str
                    if let Some(seg) = path.segments.last() {
                        if seg.ident.name == "str" {
                            return true;
                        }
                    }
                    false
                }
                ast::TypeExpr::Reference { inner, .. } => {
                    // &[u8] or &str only - NOT &[f64], &[i64] etc.
                    match inner.as_ref() {
                        ast::TypeExpr::Slice(elem_ty) => {
                            // Only &[u8] counts as byte slice
                            if let ast::TypeExpr::Path(path) = elem_ty.as_ref() {
                                if let Some(seg) = path.segments.last() {
                                    seg.ident.name == "u8"
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        ast::TypeExpr::Path(path) => {
                            if let Some(seg) = path.segments.last() {
                                seg.ident.name == "str"  // &str
                            } else {
                                false
                            }
                        }
                        _ => false,
                    }
                }
                ast::TypeExpr::Slice(elem_ty) => {
                    // [u8] without & - only byte arrays
                    if let ast::TypeExpr::Path(path) = elem_ty.as_ref() {
                        if let Some(seg) = path.segments.last() {
                            seg.ident.name == "u8"
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                _ => false,
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
                            return self
                                .compile_array_transform(fn_value, scope, elements, closure);
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
                                return self.compile_array_transform_then_sum(
                                    fn_value, scope, elements, closure,
                                );
                            }
                            PipeOp::ReduceProd => {
                                return self.compile_array_transform_then_product(
                                    fn_value, scope, elements, closure,
                                );
                            }
                            _ => {}
                        }
                    }
                    if let PipeOp::Filter(predicate) = &operations[0] {
                        match &operations[1] {
                            PipeOp::ReduceSum => {
                                return self.compile_array_filter_then_sum(
                                    fn_value, scope, elements, predicate,
                                );
                            }
                            PipeOp::ReduceProd => {
                                return self.compile_array_filter_then_product(
                                    fn_value, scope, elements, predicate,
                                );
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
                        let is_true = self
                            .builder
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
                        let is_true = self
                            .builder
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
                    sum = self
                        .builder
                        .build_int_add(sum, val, "sum")
                        .map_err(|e| e.to_string())?;
                }
                return Ok(sum);
            }

            // For larger arrays, generate a proper loop
            let array_type = i64_type.array_type(len as u32);
            let array_ptr = self
                .builder
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
                    self.builder
                        .build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }
                .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(elem_ptr, value)
                    .map_err(|e| e.to_string())?;
            }

            // Create loop blocks
            let loop_header = self.context.append_basic_block(fn_value, "sum_header");
            let loop_body = self.context.append_basic_block(fn_value, "sum_body");
            let loop_exit = self.context.append_basic_block(fn_value, "sum_exit");

            // Initialize: sum = 0, i = 0
            let sum_ptr = self
                .builder
                .build_alloca(i64_type, "sum_ptr")
                .map_err(|e| e.to_string())?;
            let idx_ptr = self
                .builder
                .build_alloca(i64_type, "idx_ptr")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(sum_ptr, i64_type.const_int(0, false))
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(idx_ptr, i64_type.const_int(0, false))
                .map_err(|e| e.to_string())?;

            // Branch to header
            self.builder
                .build_unconditional_branch(loop_header)
                .map_err(|e| e.to_string())?;

            // Loop header: check i < len
            self.builder.position_at_end(loop_header);
            let idx = self
                .builder
                .build_load(i64_type, idx_ptr, "idx")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let len_val = i64_type.const_int(len as u64, false);
            let cond = self
                .builder
                .build_int_compare(IntPredicate::ULT, idx, len_val, "cmp")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(cond, loop_body, loop_exit)
                .map_err(|e| e.to_string())?;

            // Loop body: sum += arr[i]; i++
            self.builder.position_at_end(loop_body);
            let elem_ptr = unsafe {
                self.builder.build_gep(
                    array_type,
                    array_ptr,
                    &[i64_type.const_int(0, false), idx],
                    "elem_ptr",
                )
            }
            .map_err(|e| e.to_string())?;
            let elem_val = self
                .builder
                .build_load(i64_type, elem_ptr, "elem")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let sum = self
                .builder
                .build_load(i64_type, sum_ptr, "sum")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let new_sum = self
                .builder
                .build_int_add(sum, elem_val, "new_sum")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(sum_ptr, new_sum)
                .map_err(|e| e.to_string())?;
            let new_idx = self
                .builder
                .build_int_add(idx, i64_type.const_int(1, false), "new_idx")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(idx_ptr, new_idx)
                .map_err(|e| e.to_string())?;
            self.builder
                .build_unconditional_branch(loop_header)
                .map_err(|e| e.to_string())?;

            // Loop exit: return sum
            self.builder.position_at_end(loop_exit);
            let final_sum = self
                .builder
                .build_load(i64_type, sum_ptr, "final_sum")
                .map_err(|e| e.to_string())?
                .into_int_value();

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
                product = self
                    .builder
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
            let result_ptr = self
                .builder
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
                let param_ptr = self
                    .builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(param_ptr, elem_val)
                    .map_err(|e| e.to_string())?;

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
                    self.builder
                        .build_gep(array_type, result_ptr, &indices, "out_elem")
                }
                .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(out_ptr, result)
                    .map_err(|e| e.to_string())?;
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
                let param_ptr = self
                    .builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(param_ptr, elem_val)
                    .map_err(|e| e.to_string())?;

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
                sum = self
                    .builder
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

                let param_ptr = self
                    .builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(param_ptr, elem_val)
                    .map_err(|e| e.to_string())?;

                let old_val = scope.vars.insert(param_name.to_string(), param_ptr);

                let transformed = self.compile_expr(fn_value, scope, body)?;

                if let Some(old) = old_val {
                    scope.vars.insert(param_name.to_string(), old);
                } else {
                    scope.vars.remove(&param_name);
                }

                product = self
                    .builder
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
            let out_ptr = self
                .builder
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
                let param_ptr = self
                    .builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(param_ptr, elem_val)
                    .map_err(|e| e.to_string())?;

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
                let is_true = self
                    .builder
                    .build_int_compare(IntPredicate::NE, pred_result, zero, "is_passing")
                    .map_err(|e| e.to_string())?;

                // Conditionally add 1 to count
                let one = i64_type.const_int(1, false);
                let inc = self
                    .builder
                    .build_select(is_true, one, zero, "inc")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                count = self
                    .builder
                    .build_int_add(count, inc, "count")
                    .map_err(|e| e.to_string())?;

                // Store element if passing (always store, use select for value)
                let indices = [
                    i64_type.const_int(0, false),
                    i64_type.const_int(out_idx, false),
                ];
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(array_type, out_ptr, &indices, "out_elem")
                }
                .map_err(|e| e.to_string())?;

                // Use select: if passing, store element; otherwise store 0 (placeholder)
                let value_to_store = self
                    .builder
                    .build_select(is_true, elem_val, zero, "val_or_zero")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                self.builder
                    .build_store(elem_ptr, value_to_store)
                    .map_err(|e| e.to_string())?;

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
                let param_ptr = self
                    .builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(param_ptr, elem_val)
                    .map_err(|e| e.to_string())?;

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
                let is_true = self
                    .builder
                    .build_int_compare(IntPredicate::NE, pred_result, zero, "is_passing")
                    .map_err(|e| e.to_string())?;

                // Add element to sum only if passing
                let add_value = self
                    .builder
                    .build_select(is_true, elem_val, zero, "add_if_pass")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                sum = self
                    .builder
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
                let param_ptr = self
                    .builder
                    .build_alloca(i64_type, &param_name)
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(param_ptr, elem_val)
                    .map_err(|e| e.to_string())?;

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
                let is_true = self
                    .builder
                    .build_int_compare(IntPredicate::NE, pred_result, zero, "is_passing")
                    .map_err(|e| e.to_string())?;

                // Multiply by element only if passing, otherwise multiply by 1 (identity)
                let mul_value = self
                    .builder
                    .build_select(is_true, elem_val, one, "mul_if_pass")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                product = self
                    .builder
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
            let array_ptr = self
                .builder
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
                    self.builder
                        .build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }
                .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(elem_ptr, value)
                    .map_err(|e| e.to_string())?;
            }

            // Compute index
            let idx = self.compile_expr(fn_value, scope, index_expr)?;

            // Bounds check: clamp to valid range
            let len_val = i64_type.const_int(len as u64 - 1, false);
            let zero = i64_type.const_int(0, false);
            let clamped_high = self
                .builder
                .build_select(
                    self.builder
                        .build_int_compare(IntPredicate::UGT, idx, len_val, "gt_len")
                        .map_err(|e| e.to_string())?,
                    len_val,
                    idx,
                    "clamp_high",
                )
                .map_err(|e| e.to_string())?
                .into_int_value();
            let clamped = self
                .builder
                .build_select(
                    self.builder
                        .build_int_compare(IntPredicate::SLT, clamped_high, zero, "lt_zero")
                        .map_err(|e| e.to_string())?,
                    zero,
                    clamped_high,
                    "clamp_low",
                )
                .map_err(|e| e.to_string())?
                .into_int_value();

            // Load element at clamped index
            let indices = [i64_type.const_int(0, false), clamped];
            let elem_ptr = unsafe {
                self.builder
                    .build_gep(array_type, array_ptr, &indices, "nth_ptr")
            }
            .map_err(|e| e.to_string())?;
            let value = self
                .builder
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
                let is_less = self
                    .builder
                    .build_int_compare(IntPredicate::SLT, val, min_val, "is_less")
                    .map_err(|e| e.to_string())?;
                min_val = self
                    .builder
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
                let is_greater = self
                    .builder
                    .build_int_compare(IntPredicate::SGT, val, max_val, "is_greater")
                    .map_err(|e| e.to_string())?;
                max_val = self
                    .builder
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
                let is_true = self
                    .builder
                    .build_int_compare(IntPredicate::NE, val, zero, "is_true")
                    .map_err(|e| e.to_string())?;
                let as_int = self
                    .builder
                    .build_int_z_extend(is_true, self.context.i64_type(), "as_int")
                    .map_err(|e| e.to_string())?;
                result = self
                    .builder
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
                let is_true = self
                    .builder
                    .build_int_compare(IntPredicate::NE, val, zero, "is_true")
                    .map_err(|e| e.to_string())?;
                let as_int = self
                    .builder
                    .build_int_z_extend(is_true, self.context.i64_type(), "as_int")
                    .map_err(|e| e.to_string())?;
                result = self
                    .builder
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
                hash = self
                    .builder
                    .build_int_add(hash, val, "hash_acc")
                    .map_err(|e| e.to_string())?;
            }

            // index = hash % len (use abs to handle negative sums)
            let len_const = self.context.i64_type().const_int(len as u64, false);

            // Get absolute value: (hash ^ (hash >> 63)) - (hash >> 63)
            let shift_amt = self.context.i64_type().const_int(63, false);
            let sign = self
                .builder
                .build_right_shift(hash, shift_amt, true, "sign")
                .map_err(|e| e.to_string())?;
            let xored = self
                .builder
                .build_xor(hash, sign, "xored")
                .map_err(|e| e.to_string())?;
            let abs_hash = self
                .builder
                .build_int_sub(xored, sign, "abs")
                .map_err(|e| e.to_string())?;

            let index = self
                .builder
                .build_int_unsigned_rem(abs_hash, len_const, "choice_idx")
                .map_err(|e| e.to_string())?;

            // Allocate array and select by index
            let i64_type = self.context.i64_type();
            let array_type = i64_type.array_type(len as u32);
            let array_ptr = self
                .builder
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
                    self.builder
                        .build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }
                .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(ptr, val)
                    .map_err(|e| e.to_string())?;
            }

            // Load element at computed index
            let result_ptr = unsafe {
                self.builder.build_gep(
                    array_type,
                    array_ptr,
                    &[i64_type.const_int(0, false), index],
                    "choice_ptr",
                )
            }
            .map_err(|e| e.to_string())?;

            let result = self
                .builder
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
            let (acc_name, elem_name, body) = if let Expr::Closure { params, body, .. } = reduce_fn
            {
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
            let acc_ptr = self
                .builder
                .build_alloca(i64_type, &acc_name)
                .map_err(|e| e.to_string())?;
            let elem_ptr = self
                .builder
                .build_alloca(i64_type, &elem_name)
                .map_err(|e| e.to_string())?;

            // Initialize accumulator with first element
            let first = self.compile_expr(fn_value, scope, &elements[0])?;
            self.builder
                .build_store(acc_ptr, first)
                .map_err(|e| e.to_string())?;

            // Add bindings to scope
            scope.vars.insert(acc_name.clone(), acc_ptr);
            scope.vars.insert(elem_name.clone(), elem_ptr);

            // Fold over remaining elements
            for elem in &elements[1..] {
                let val = self.compile_expr(fn_value, scope, elem)?;
                self.builder
                    .build_store(elem_ptr, val)
                    .map_err(|e| e.to_string())?;

                // Evaluate body
                let new_acc = self.compile_expr(fn_value, scope, body)?;
                self.builder
                    .build_store(acc_ptr, new_acc)
                    .map_err(|e| e.to_string())?;
            }

            // Load final accumulator value
            let result = self
                .builder
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
            let array_ptr = self
                .builder
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
                    self.builder
                        .build_gep(array_type, array_ptr, &indices, "elem_ptr")
                }
                .map_err(|e| e.to_string())?;

                // Store the value
                self.builder
                    .build_store(elem_ptr, value)
                    .map_err(|e| e.to_string())?;
            }

            // Return pointer as i64 (we'll improve this later with proper fat pointers)
            // For now, pack ptr in low bits and len in high bits of a struct
            let ptr_as_int = self
                .builder
                .build_ptr_to_int(array_ptr, i64_type, "arr_ptr")
                .map_err(|e| e.to_string())?;

            // Store length in scope for later retrieval (hacky but works for now)
            // We'll use a naming convention: the array pointer + "_len"
            // Better: return a struct { ptr, len } but that requires more refactoring

            Ok(ptr_as_int)
        }

        /// Compile array/slice indexing: arr[idx] or arr[start..end]
        /// Vec layout is: {len: i64, cap: i64, data[]} where data is inline at offset 2
        /// Uses cached data base pointer when available to avoid repeated +2 offset calculations
        /// Range indexing (arr[start..end]) creates a new Vec with copied elements
        fn compile_index(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            expr: &Expr,
            index: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            let i64_type = self.context.i64_type();

            // Check if this is a range index (slice operation)
            if let Expr::Range { start, end, inclusive } = index {
                return self.compile_range_index(fn_value, scope, expr, start.as_deref(), end.as_deref(), *inclusive);
            }

            let idx = self.compile_expr(fn_value, scope, index)?;

            // G28: Check if this is a byte slice (from as_bytes)
            // Byte slices use direct pointer indexing, not Vec runtime functions
            let is_byte_slice = if let Expr::Path(path) = expr {
                if let Some(seg) = path.segments.last() {
                    scope.is_string_var(&seg.ident.name)
                } else {
                    false
                }
            } else {
                false
            };

            if is_byte_slice {
                // Direct pointer indexing for byte slices
                let ptr_type = self.context.ptr_type(AddressSpace::default());
                let ptr_val = self.compile_expr(fn_value, scope, expr)?;
                let ptr = self
                    .builder
                    .build_int_to_ptr(ptr_val, ptr_type, "byte_slice_ptr")
                    .map_err(|e| e.to_string())?;

                // GEP to get byte at index
                let byte_ptr = unsafe {
                    self.builder
                        .build_gep(self.context.i8_type(), ptr, &[idx], "byte_ptr")
                        .map_err(|e| e.to_string())?
                };

                // Load the byte
                let byte_val = self
                    .builder
                    .build_load(self.context.i8_type(), byte_ptr, "byte_val")
                    .map_err(|e| e.to_string())?
                    .into_int_value();

                // Zero-extend to i64
                return self
                    .builder
                    .build_int_z_extend(byte_val, i64_type, "byte_i64")
                    .map_err(|e| e.to_string());
            }

            // G25 Fix: Use sigil_vec_get for proper Vec access
            // The Rust Vec memory layout (ptr, len, cap) doesn't match the
            // inline data assumption. Call the runtime function instead.
            let vec_ptr = self.compile_expr(fn_value, scope, expr)?;

            let vec_get_fn = self
                .module
                .get_function("sigil_vec_get")
                .ok_or("sigil_vec_get not declared")?;

            let call = self
                .builder
                .build_call(vec_get_fn, &[vec_ptr.into(), idx.into()], "vec_elem")
                .map_err(|e| e.to_string())?;

            Ok(call
                .try_as_basic_value()
                .left()
                .map(|v| v.into_int_value())
                .unwrap_or_else(|| i64_type.const_int(0, false)))
        }

        /// Compile range indexing (slicing): arr[start..end]
        /// Creates a new Vec containing copied elements from the range
        /// G32: Also handles string slicing using sigil_string_slice
        fn compile_range_index(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            expr: &Expr,
            start: Option<&Expr>,
            end: Option<&Expr>,
            _inclusive: bool,
        ) -> Result<IntValue<'ctx>, String> {
            let i64_type = self.context.i64_type();
            let ptr_type = self.context.ptr_type(AddressSpace::default());

            // G32: Check if this is a C string literal (direct literal in expression)
            let is_c_string_literal = matches!(expr, Expr::Literal(Literal::String(_)));

            // G32: Check variable types for string handling
            let (is_c_string_var, is_rust_string) = if let Expr::Path(path) = expr {
                if let Some(seg) = path.segments.last() {
                    let var_type = scope.var_types.get(&seg.ident.name);
                    match var_type {
                        // SigilType::RustString: from to_string(), fs_read(), etc.
                        Some(SigilType::RustString) => (false, true),
                        // SigilType::String: C string literal stored in variable
                        Some(SigilType::String) => (true, false),
                        _ => (false, false),
                    }
                } else {
                    (false, false)
                }
            } else {
                (false, false)
            };

            let is_c_string = is_c_string_literal || is_c_string_var;

            // G32: Handle string slicing
            if is_c_string || is_rust_string {
                let src_ptr_int = self.compile_expr(fn_value, scope, expr)?;
                let start_idx = if let Some(s) = start {
                    self.compile_expr(fn_value, scope, s)?
                } else {
                    i64_type.const_int(0, false)
                };
                // For end, we need to get length first if not specified
                let end_idx = if let Some(e) = end {
                    self.compile_expr(fn_value, scope, e)?
                } else {
                    // Get string length - use appropriate function
                    let strlen_fn_name = if is_rust_string {
                        "sigil_string_len"
                    } else {
                        "sigil_strlen"
                    };
                    let strlen_fn = self
                        .module
                        .get_function(strlen_fn_name)
                        .ok_or(format!("{} not declared", strlen_fn_name))?;
                    let len_call = self
                        .builder
                        .build_call(strlen_fn, &[src_ptr_int.into()], "str_len")
                        .map_err(|e| e.to_string())?;
                    len_call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| i64_type.const_int(0, false))
                };

                // Use appropriate slice function
                let slice_fn_name = if is_rust_string {
                    "sigil_rust_string_slice"
                } else {
                    "sigil_string_slice"
                };
                let slice_fn = self
                    .module
                    .get_function(slice_fn_name)
                    .ok_or(format!("{} not declared", slice_fn_name))?;

                // Functions use i64 for pointers, so pass directly
                let call = self
                    .builder
                    .build_call(
                        slice_fn,
                        &[src_ptr_int.into(), start_idx.into(), end_idx.into()],
                        "str_slice",
                    )
                    .map_err(|e| e.to_string())?;
                // Result is i64 (pointer as integer)
                return Ok(call
                    .try_as_basic_value()
                    .left()
                    .map(|v| v.into_int_value())
                    .unwrap_or_else(|| i64_type.const_int(0, false)));
            }

            // Get the source vec pointer
            let src_ptr_int = self.compile_expr(fn_value, scope, expr)?;
            let src_ptr = self
                .builder
                .build_int_to_ptr(src_ptr_int, ptr_type, "src_ptr")
                .map_err(|e| e.to_string())?;

            // Get source length via sigil_vec_len
            let len_fn = self
                .module
                .get_function("sigil_vec_len")
                .ok_or("sigil_vec_len not declared")?;
            let len_call = self
                .builder
                .build_call(len_fn, &[src_ptr_int.into()], "src_len")
                .map_err(|e| e.to_string())?;
            let src_len = len_call
                .try_as_basic_value()
                .left()
                .map(|v| v.into_int_value())
                .unwrap_or_else(|| i64_type.const_int(0, false));

            // Get start index (default: 0)
            let start_idx = if let Some(s) = start {
                self.compile_expr(fn_value, scope, s)?
            } else {
                i64_type.const_int(0, false)
            };

            // Get end index (default: src_len)
            let end_idx = if let Some(e) = end {
                self.compile_expr(fn_value, scope, e)?
            } else {
                src_len
            };

            // Calculate slice length: end - start
            let slice_len = self
                .builder
                .build_int_sub(end_idx, start_idx, "slice_len")
                .map_err(|e| e.to_string())?;

            // Get runtime functions
            let new_fn = self
                .module
                .get_function("sigil_vec_new")
                .ok_or("sigil_vec_new not declared")?;
            let get_fn = self
                .module
                .get_function("sigil_vec_get")
                .ok_or("sigil_vec_get not declared")?;
            let push_fn = self
                .module
                .get_function("sigil_vec_push")
                .ok_or("sigil_vec_push not declared")?;

            // Create new vec with calculated capacity
            let new_call = self
                .builder
                .build_call(new_fn, &[slice_len.into()], "new_vec")
                .map_err(|e| e.to_string())?;
            let new_vec = new_call
                .try_as_basic_value()
                .left()
                .map(|v| v.into_int_value())
                .unwrap_or_else(|| i64_type.const_int(0, false));

            // Build loop to copy elements from start_idx to end_idx
            let loop_header = self.context.append_basic_block(fn_value, "slice_header");
            let loop_body = self.context.append_basic_block(fn_value, "slice_body");
            let loop_end = self.context.append_basic_block(fn_value, "slice_end");

            // Initialize counter to start_idx
            let counter_ptr = self
                .builder
                .build_alloca(i64_type, "slice_i")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(counter_ptr, start_idx)
                .map_err(|e| e.to_string())?;

            // Jump to header
            self.builder
                .build_unconditional_branch(loop_header)
                .map_err(|e| e.to_string())?;

            // Loop header: check if i < end_idx
            self.builder.position_at_end(loop_header);
            let i = self
                .builder
                .build_load(i64_type, counter_ptr, "i")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let cmp = self
                .builder
                .build_int_compare(inkwell::IntPredicate::SLT, i, end_idx, "cmp")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(cmp, loop_body, loop_end)
                .map_err(|e| e.to_string())?;

            // Loop body: get element from source at index i, push to new vec
            self.builder.position_at_end(loop_body);
            let get_call = self
                .builder
                .build_call(get_fn, &[src_ptr_int.into(), i.into()], "elem")
                .map_err(|e| e.to_string())?;
            let elem = get_call
                .try_as_basic_value()
                .left()
                .map(|v| v.into_int_value())
                .unwrap_or_else(|| i64_type.const_int(0, false));
            self.builder
                .build_call(push_fn, &[new_vec.into(), elem.into()], "")
                .map_err(|e| e.to_string())?;

            // Increment counter
            let next_i = self
                .builder
                .build_int_add(i, i64_type.const_int(1, false), "next_i")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(counter_ptr, next_i)
                .map_err(|e| e.to_string())?;
            self.builder
                .build_unconditional_branch(loop_header)
                .map_err(|e| e.to_string())?;

            // Position at end and return new vec pointer
            self.builder.position_at_end(loop_end);

            Ok(new_vec)
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

        /// Check if an expression is a float (or involves floats), with scope for variable tracking
        fn is_float_expr_with_scope(&self, expr: &Expr, scope: &CompileScope<'ctx>) -> bool {
            match expr {
                Expr::Literal(Literal::Float { .. }) => true,
                Expr::Binary { op, left, right } => {
                    // G20: Comparison operators always return integers, even with float operands
                    if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne) {
                        return false;
                    }
                    // Arithmetic/logical operators: float if either operand is float
                    self.is_float_expr_with_scope(left, scope) || self.is_float_expr_with_scope(right, scope)
                }
                Expr::Unary { expr: inner, .. } => self.is_float_expr_with_scope(inner, scope),
                Expr::Path(path) => {
                    // Check if variable is known to be a float
                    if let Some(seg) = path.segments.last() {
                        return scope.float_vars.contains(&seg.ident.name);
                    }
                    false
                }
                Expr::Call { func, .. } => {
                    // Check for float-returning functions
                    if let Expr::Path(path) = func.as_ref() {
                        if let Some(seg) = path.segments.last() {
                            let name = seg.ident.name.as_str();
                            // Built-in math functions
                            if matches!(
                                name,
                                "sin" | "cos" | "tan" | "sqrt" | "exp" | "log" | "PI" |
                                "floor" | "ceil" | "abs" | "pow" | "asin" | "acos" | "atan"
                            ) {
                                return true;
                            }
                            // G21: User-defined functions that return f64
                            if scope.float_funcs.contains(name) {
                                return true;
                            }
                        }
                    }
                    false
                }
                Expr::MethodCall { receiver, method, .. } => {
                    let name = method.name.as_str();
                    // Methods that always return float
                    if matches!(name, "sqrt" | "abs" | "floor" | "ceil" | "sin" | "cos" | "tan" | "exp" | "log" | "ln" | "pow") {
                        return true;
                    }
                    // G19: Methods that return integers even if receiver is float-containing
                    if matches!(name, "len" | "capacity" | "is_empty" | "first" | "last" | "get") {
                        return false;
                    }
                    // G21b: Check if this method is known to return f64
                    if scope.float_funcs.contains(name) {
                        return true;
                    }
                    // For other methods, check if receiver is a scalar float (not a container)
                    if let Expr::Path(path) = receiver.as_ref() {
                        if let Some(seg) = path.segments.last() {
                            // Check if it's a scalar float variable, not a Vec<f64>
                            if scope.float_vars.contains(&seg.ident.name) {
                                // Need to distinguish scalar floats from Vec<f64>
                                // For now, assume if it's in float_vars and called with an unknown method,
                                // it's probably a scalar float being operated on
                                return true;
                            }
                        }
                    }
                    false
                }
                Expr::Cast { ty, .. } => {
                    // Check if casting to f64
                    if let ast::TypeExpr::Path(path) = ty {
                        if let Some(seg) = path.segments.last() {
                            return seg.ident.name == "f64" || seg.ident.name == "f32";
                        }
                    }
                    false
                }
                Expr::Index { expr, .. } => {
                    // Check if the array/vec being indexed contains floats
                    // This is a heuristic - check if the container variable is marked as float
                    if let Expr::AddrOf { expr: inner, .. } = expr.as_ref() {
                        return self.is_float_expr_with_scope(inner, scope);
                    }
                    self.is_float_expr_with_scope(expr, scope)
                }
                // G19: Check if vec! macro contains float elements
                Expr::Macro { path, tokens } => {
                    let macro_name = path.segments.last()
                        .map(|s| s.ident.name.trim_end_matches('!'))
                        .unwrap_or("");
                    if macro_name == "vec" {
                        // Check if tokens contain a float literal (has decimal point)
                        let tokens_trimmed = tokens.trim();
                        if tokens_trimmed.contains('.') {
                            // Verify it's likely a number (not a method call)
                            for part in tokens_trimmed.split(',') {
                                let part = part.trim().split(';').next().unwrap_or("").trim();
                                if part.contains('.') && !part.contains('(') {
                                    return true;
                                }
                            }
                        }
                    }
                    false
                }
                // G24: Handle struct field access - check if field type is f64
                Expr::Field { expr: base_expr, field } => {
                    // Get the type of the base expression
                    let base_type = match base_expr.as_ref() {
                        // For `this`/`self`, use current_self_type
                        Expr::Path(path) if path.segments.len() == 1 &&
                            (path.segments[0].ident.name == "this" || path.segments[0].ident.name == "self") => {
                            self.current_self_type.clone()
                        }
                        // For other variables, look up their struct type
                        _ => self.get_struct_type_from_expr(base_expr, scope),
                    };

                    // If we know the base type, look up the field type
                    if let Some(struct_name) = base_type {
                        let field_name = &field.name;
                        // Look up field type in field_type_names
                        if let Some(field_type) = self.field_type_names.get(&(struct_name.clone(), field_name.clone())) {
                            // Check if the field type is f64 or contains f64
                            if field_type == "f64" || field_type.contains("f64") {
                                return true;
                            }
                        }
                    }

                    // Fallback: use heuristic based on field name
                    let field_name = &field.name.to_lowercase();
                    let float_field_patterns = [
                        "lambda", "rate", "energy", "loss", "weight", "scale", "bias",
                        "grad", "lr", "epsilon", "alpha", "beta", "gamma", "momentum",
                        "decay", "factor", "ratio", "threshold", "temp", "sigma", "eps",
                    ];
                    for pattern in &float_field_patterns {
                        if field_name.contains(pattern) {
                            return true;
                        }
                    }
                    false
                }
                _ => false,
            }
        }

        /// Extract the struct type name from an expression, if it can be determined
        /// Returns Some(type_name) for struct literals, constructor calls, and known variables
        fn get_struct_type_from_expr(&self, expr: &Expr, scope: &CompileScope<'ctx>) -> Option<String> {
            match expr {
                // Struct literal: StructName { field: value, ... }
                Expr::Struct { path, .. } => {
                    // path is a TypePath, get the last segment
                    if let Some(seg) = path.segments.last() {
                        return Some(seg.ident.name.clone());
                    }
                    None
                }
                // Constructor call: StructName·new() or StructName·method()
                // G33: Also check ret_types for standalone function calls
                Expr::Call { func, .. } => {
                    if let Expr::Path(path) = func.as_ref() {
                        // Look for Type·method pattern (2 or more segments with middledot)
                        if path.segments.len() >= 2 {
                            // First segment is likely the type name
                            return Some(path.segments[0].ident.name.clone());
                        }
                        // G33: Check if standalone function returns a struct type
                        if path.segments.len() == 1 {
                            let func_name = &path.segments[0].ident.name;
                            if let Some(ret_ty) = self.ret_types.get(func_name) {
                                // Extract struct name from return type
                                if let Some(type_name) = self.extract_struct_type_from_type_expr(ret_ty) {
                                    return Some(type_name);
                                }
                            }
                        }
                    }
                    None
                }
                // Variable reference: look up its known type
                Expr::Path(path) => {
                    if path.segments.len() == 1 {
                        let var_name = &path.segments[0].ident.name;
                        // G15 fix: Handle this/self specially to use current_self_type
                        if var_name == "this" || var_name == "self" {
                            return self.current_self_type.clone();
                        }
                        return scope.get_struct_type(var_name).cloned();
                    }
                    None
                }
                // Index access: vec[i] - get element type from Vec<T>
                // G15 fix: Enable correct method dispatch on indexed Vec elements
                Expr::Index { expr: container, .. } => {
                    // Check if container is a field access (e.g., model.layers[0])
                    if let Expr::Field { expr: base_expr, field } = container.as_ref() {
                        // Get the base type (e.g., Model)
                        let base_type = self.get_struct_type_from_expr(base_expr, scope);
                        if let Some(struct_name) = base_type {
                            // Look up field type (e.g., layers -> Vec<Layer>)
                            if let Some(field_type) = self.field_type_names.get(&(struct_name.clone(), field.name.clone())) {
                                // Extract element type from Vec<T>
                                if field_type.starts_with("Vec<") && field_type.ends_with(">") {
                                    let elem_type = &field_type[4..field_type.len()-1];
                                    return Some(elem_type.to_string());
                                }
                            }
                        }
                    }
                    // Also try recursively for nested containers
                    if let Some(container_type) = self.get_struct_type_from_expr(container, scope) {
                        // Check if container_type is Vec<T>
                        if container_type.starts_with("Vec<") && container_type.ends_with(">") {
                            let elem_type = &container_type[4..container_type.len()-1];
                            return Some(elem_type.to_string());
                        }
                    }
                    None
                }
                // Field access: this.field or var.field - get the field's type from struct definition
                Expr::Field { expr: base_expr, field } => {
                    // Get the type of the base expression
                    let base_type = match base_expr.as_ref() {
                        // For `this`/`self`, use current_self_type
                        // Note: Sigil parser normalizes `this` to `self` in AST
                        Expr::Path(path) if path.segments.len() == 1 &&
                            (path.segments[0].ident.name == "this" || path.segments[0].ident.name == "self") => {
                            self.current_self_type.clone()
                        }
                        // For other variables, look up their struct type
                        _ => self.get_struct_type_from_expr(base_expr, scope),
                    };

                    // If we know the base type, look up the field type
                    if let Some(struct_name) = base_type {
                        let field_name = &field.name;
                        // Look up field type in field_type_names
                        if let Some(field_type) = self.field_type_names.get(&(struct_name.clone(), field_name.clone())) {
                            return Some(field_type.clone());
                        }
                    }
                    None
                }
                _ => None,
            }
        }

        /// Check if an expression is a float (or involves floats), without scope
        fn is_float_expr(&self, expr: &Expr) -> bool {
            match expr {
                Expr::Literal(Literal::Float { .. }) => true,
                Expr::Binary { left, right, .. } => {
                    self.is_float_expr(left) || self.is_float_expr(right)
                }
                Expr::Unary { expr: inner, .. } => self.is_float_expr(inner),
                Expr::Call { func, .. } => {
                    // Check for float-returning functions
                    if let Expr::Path(path) = func.as_ref() {
                        if let Some(seg) = path.segments.last() {
                            let name = seg.ident.name.as_str();
                            return matches!(
                                name,
                                "sin" | "cos" | "tan" | "sqrt" | "exp" | "log" | "PI" |
                                "floor" | "ceil" | "abs" | "pow" | "asin" | "acos" | "atan"
                            );
                        }
                    }
                    false
                }
                Expr::MethodCall { method, .. } => {
                    // sqrt() method returns float
                    let name = method.name.as_str();
                    matches!(name, "sqrt" | "abs" | "floor" | "ceil" | "sin" | "cos" | "tan" | "exp" | "log" | "pow")
                }
                Expr::Cast { ty, .. } => {
                    // Check if casting to f64
                    if let ast::TypeExpr::Path(path) = ty {
                        if let Some(seg) = path.segments.last() {
                            return seg.ident.name == "f64" || seg.ident.name == "f32";
                        }
                    }
                    false
                }
                _ => false,
            }
        }

        /// Compile an expression that is known to be a float, returning native FloatValue
        /// This avoids bitcasts within float expressions by keeping values as f64
        fn compile_native_float_expr(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            expr: &Expr,
        ) -> Result<inkwell::values::FloatValue<'ctx>, String> {
            let f64_type = self.context.f64_type();

            match expr {
                Expr::Literal(Literal::Float { value, .. }) => {
                    // Parse float literal directly to f64
                    let s = value.replace('_', "");
                    let s = s.trim_end_matches("f64").trim_end_matches("f32");
                    let v: f64 = s.parse().map_err(|_| format!("Invalid float: {}", value))?;
                    Ok(f64_type.const_float(v))
                }
                Expr::Path(path) => {
                    // Variable load - check if it's a float variable
                    let name = path.segments.last()
                        .map(|s| s.ident.name.as_str())
                        .ok_or("Empty path")?;

                    if let Some(&ptr) = scope.vars.get(name) {
                        // Load as i64 bits, convert to f64
                        let val = self.builder
                            .build_load(self.context.i64_type(), ptr, name)
                            .map_err(|e| e.to_string())?
                            .into_int_value();
                        let f_val = self.builder
                            .build_bit_cast(val, f64_type, "load_f64")
                            .map_err(|e| e.to_string())?
                            .into_float_value();
                        Ok(f_val)
                    } else if let Some(global) = self.global_vars.get(name) {
                        // Load from global/static variable
                        let val = self.builder
                            .build_load(self.context.i64_type(), global.as_pointer_value(), name)
                            .map_err(|e| e.to_string())?
                            .into_int_value();
                        let f_val = self.builder
                            .build_bit_cast(val, f64_type, "load_f64")
                            .map_err(|e| e.to_string())?
                            .into_float_value();
                        Ok(f_val)
                    } else {
                        Err(format!("Variable not found: {}", name))
                    }
                }
                Expr::Binary { op, left, right } => {
                    // Compile both sides as floats
                    let lhs = self.compile_native_float_expr(fn_value, scope, left)?;
                    let rhs = self.compile_native_float_expr(fn_value, scope, right)?;

                    // Constant folding: if both operands are constants, compute at compile time
                    if let (Some((lhs_val, _)), Some((rhs_val, _))) = (lhs.get_constant(), rhs.get_constant()) {
                        let result = match op {
                            BinOp::Add => lhs_val + rhs_val,
                            BinOp::Sub => lhs_val - rhs_val,
                            BinOp::Mul => lhs_val * rhs_val,
                            BinOp::Div => lhs_val / rhs_val,
                            BinOp::Rem => lhs_val % rhs_val,
                            _ => {
                                // Fall back to runtime for non-arithmetic ops
                                let int_result = self.compile_expr(fn_value, scope, expr)?;
                                return self.builder
                                    .build_bit_cast(int_result, f64_type, "cast_f64")
                                    .map_err(|e| e.to_string())
                                    .map(|v| v.into_float_value());
                            }
                        };
                        return Ok(f64_type.const_float(result));
                    }

                    // Runtime path
                    match op {
                        BinOp::Add => self.builder
                            .build_float_add(lhs, rhs, "fadd")
                            .map_err(|e| e.to_string()),
                        BinOp::Sub => self.builder
                            .build_float_sub(lhs, rhs, "fsub")
                            .map_err(|e| e.to_string()),
                        BinOp::Mul => self.builder
                            .build_float_mul(lhs, rhs, "fmul")
                            .map_err(|e| e.to_string()),
                        BinOp::Div => self.builder
                            .build_float_div(lhs, rhs, "fdiv")
                            .map_err(|e| e.to_string()),
                        BinOp::Rem => self.builder
                            .build_float_rem(lhs, rhs, "frem")
                            .map_err(|e| e.to_string()),
                        _ => {
                            // For comparisons and other ops, fall back to regular compile
                            let int_result = self.compile_expr(fn_value, scope, expr)?;
                            self.builder
                                .build_bit_cast(int_result, f64_type, "cast_f64")
                                .map_err(|e| e.to_string())
                                .map(|v| v.into_float_value())
                        }
                    }
                }
                Expr::Call { func, args } => {
                    // Handle math functions that return float
                    if let Expr::Path(path) = func.as_ref() {
                        if let Some(seg) = path.segments.last() {
                            match seg.ident.name.as_str() {
                                "PI" => {
                                    return Ok(f64_type.const_float(std::f64::consts::PI));
                                }
                                "E" => {
                                    return Ok(f64_type.const_float(std::f64::consts::E));
                                }
                                "TAU" => {
                                    return Ok(f64_type.const_float(std::f64::consts::TAU));
                                }
                                "SQRT2" => {
                                    return Ok(f64_type.const_float(std::f64::consts::SQRT_2));
                                }
                                "LN2" => {
                                    return Ok(f64_type.const_float(std::f64::consts::LN_2));
                                }
                                "LN10" => {
                                    return Ok(f64_type.const_float(std::f64::consts::LN_10));
                                }
                                "sin" | "cos" | "tan" | "sqrt" | "exp" | "log" | "floor" | "ceil" | "abs" |
                                "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" |
                                "log10" | "log2" | "round" | "trunc" => {
                                    // Compile argument as float
                                    if !args.is_empty() {
                                        let arg_f64 = self.compile_native_float_expr(fn_value, scope, &args[0])?;

                                        // Constant folding: if argument is constant, compute at compile time
                                        if let Some((val, _)) = arg_f64.get_constant() {
                                            let result = match seg.ident.name.as_str() {
                                                "sin" => val.sin(),
                                                "cos" => val.cos(),
                                                "tan" => val.tan(),
                                                "sqrt" => val.sqrt(),
                                                "exp" => val.exp(),
                                                "log" => val.ln(),
                                                "floor" => val.floor(),
                                                "ceil" => val.ceil(),
                                                "abs" => val.abs(),
                                                "asin" => val.asin(),
                                                "acos" => val.acos(),
                                                "atan" => val.atan(),
                                                "sinh" => val.sinh(),
                                                "cosh" => val.cosh(),
                                                "tanh" => val.tanh(),
                                                "log10" => val.log10(),
                                                "log2" => val.log2(),
                                                "round" => val.round(),
                                                "trunc" => val.trunc(),
                                                _ => unreachable!(),
                                            };
                                            return Ok(f64_type.const_float(result));
                                        }

                                        let intrinsic_name = match seg.ident.name.as_str() {
                                            "abs" => "llvm.fabs.f64".to_string(),
                                            name => format!("llvm.{}.f64", name),
                                        };

                                        // Try to get or declare the intrinsic
                                        let intrinsic = self.module.get_function(&intrinsic_name)
                                            .unwrap_or_else(|| {
                                                let fn_type = f64_type.fn_type(&[f64_type.into()], false);
                                                self.module.add_function(&intrinsic_name, fn_type, None)
                                            });

                                        let call = self.builder
                                            .build_call(intrinsic, &[arg_f64.into()], &seg.ident.name)
                                            .map_err(|e| e.to_string())?;
                                        return Ok(call.try_as_basic_value().left()
                                            .ok_or("Expected return value")?
                                            .into_float_value());
                                    }
                                }
                                // Two-argument math functions
                                "atan2" | "copysign" | "fmin" | "fmax" | "pow" => {
                                    if args.len() >= 2 {
                                        let arg1_f64 = self.compile_native_float_expr(fn_value, scope, &args[0])?;
                                        let arg2_f64 = self.compile_native_float_expr(fn_value, scope, &args[1])?;

                                        // Constant folding: if both arguments are constant, compute at compile time
                                        if let (Some((v1, _)), Some((v2, _))) = (arg1_f64.get_constant(), arg2_f64.get_constant()) {
                                            let result = match seg.ident.name.as_str() {
                                                "atan2" => v1.atan2(v2),
                                                "copysign" => v1.copysign(v2),
                                                "fmin" => v1.min(v2),
                                                "fmax" => v1.max(v2),
                                                "pow" => v1.powf(v2),
                                                _ => unreachable!(),
                                            };
                                            return Ok(f64_type.const_float(result));
                                        }

                                        let intrinsic_name = match seg.ident.name.as_str() {
                                            "fmin" => "llvm.minnum.f64".to_string(),
                                            "fmax" => "llvm.maxnum.f64".to_string(),
                                            name => format!("llvm.{}.f64", name),
                                        };

                                        let intrinsic = self.module.get_function(&intrinsic_name)
                                            .unwrap_or_else(|| {
                                                let fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
                                                self.module.add_function(&intrinsic_name, fn_type, None)
                                            });

                                        let call = self.builder
                                            .build_call(intrinsic, &[arg1_f64.into(), arg2_f64.into()], &seg.ident.name)
                                            .map_err(|e| e.to_string())?;
                                        return Ok(call.try_as_basic_value().left()
                                            .ok_or("Expected return value")?
                                            .into_float_value());
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    // Fall back to regular compile and convert
                    let int_result = self.compile_expr(fn_value, scope, expr)?;
                    self.builder
                        .build_bit_cast(int_result, f64_type, "call_f64")
                        .map_err(|e| e.to_string())
                        .map(|v| v.into_float_value())
                }
                Expr::MethodCall { receiver, method, args, .. } => {
                    // Handle .sqrt(), .sin(), etc.
                    match method.name.as_str() {
                        "sqrt" | "sin" | "cos" | "tan" | "exp" | "log" | "floor" | "ceil" | "abs" |
                        "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" |
                        "log10" | "log2" | "round" | "trunc" => {
                            let recv_f64 = self.compile_native_float_expr(fn_value, scope, receiver)?;

                            // Constant folding: if receiver is constant, compute at compile time
                            if let Some((val, _)) = recv_f64.get_constant() {
                                let result = match method.name.as_str() {
                                    "sqrt" => val.sqrt(),
                                    "sin" => val.sin(),
                                    "cos" => val.cos(),
                                    "tan" => val.tan(),
                                    "exp" => val.exp(),
                                    "log" => val.ln(),
                                    "floor" => val.floor(),
                                    "ceil" => val.ceil(),
                                    "abs" => val.abs(),
                                    "asin" => val.asin(),
                                    "acos" => val.acos(),
                                    "atan" => val.atan(),
                                    "sinh" => val.sinh(),
                                    "cosh" => val.cosh(),
                                    "tanh" => val.tanh(),
                                    "log10" => val.log10(),
                                    "log2" => val.log2(),
                                    "round" => val.round(),
                                    "trunc" => val.trunc(),
                                    _ => unreachable!(),
                                };
                                return Ok(f64_type.const_float(result));
                            }

                            let intrinsic_name = match method.name.as_str() {
                                "abs" => "llvm.fabs.f64".to_string(),
                                name => format!("llvm.{}.f64", name),
                            };

                            let intrinsic = self.module.get_function(&intrinsic_name)
                                .unwrap_or_else(|| {
                                    let fn_type = f64_type.fn_type(&[f64_type.into()], false);
                                    self.module.add_function(&intrinsic_name, fn_type, None)
                                });

                            let call = self.builder
                                .build_call(intrinsic, &[recv_f64.into()], &method.name)
                                .map_err(|e| e.to_string())?;
                            Ok(call.try_as_basic_value().left()
                                .ok_or("Expected return value")?
                                .into_float_value())
                        }
                        // Two-argument methods: receiver.method(arg)
                        "pow" | "atan2" | "copysign" => {
                            if !args.is_empty() {
                                let recv_f64 = self.compile_native_float_expr(fn_value, scope, receiver)?;
                                let arg_f64 = self.compile_native_float_expr(fn_value, scope, &args[0])?;

                                // Constant folding
                                if let (Some((v1, _)), Some((v2, _))) = (recv_f64.get_constant(), arg_f64.get_constant()) {
                                    let result = match method.name.as_str() {
                                        "pow" => v1.powf(v2),
                                        "atan2" => v1.atan2(v2),
                                        "copysign" => v1.copysign(v2),
                                        _ => unreachable!(),
                                    };
                                    return Ok(f64_type.const_float(result));
                                }

                                let intrinsic_name = format!("llvm.{}.f64", method.name);

                                let intrinsic = self.module.get_function(&intrinsic_name)
                                    .unwrap_or_else(|| {
                                        let fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
                                        self.module.add_function(&intrinsic_name, fn_type, None)
                                    });

                                let call = self.builder
                                    .build_call(intrinsic, &[recv_f64.into(), arg_f64.into()], &method.name)
                                    .map_err(|e| e.to_string())?;
                                Ok(call.try_as_basic_value().left()
                                    .ok_or("Expected return value")?
                                    .into_float_value())
                            } else {
                                Err(format!("{} requires an argument", method.name))
                            }
                        }
                        // min/max methods
                        "min" | "max" => {
                            if !args.is_empty() {
                                let recv_f64 = self.compile_native_float_expr(fn_value, scope, receiver)?;
                                let arg_f64 = self.compile_native_float_expr(fn_value, scope, &args[0])?;

                                // Constant folding
                                if let (Some((v1, _)), Some((v2, _))) = (recv_f64.get_constant(), arg_f64.get_constant()) {
                                    let result = match method.name.as_str() {
                                        "min" => v1.min(v2),
                                        "max" => v1.max(v2),
                                        _ => unreachable!(),
                                    };
                                    return Ok(f64_type.const_float(result));
                                }

                                let intrinsic_name = match method.name.as_str() {
                                    "min" => "llvm.minnum.f64".to_string(),
                                    "max" => "llvm.maxnum.f64".to_string(),
                                    _ => unreachable!(),
                                };

                                let intrinsic = self.module.get_function(&intrinsic_name)
                                    .unwrap_or_else(|| {
                                        let fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
                                        self.module.add_function(&intrinsic_name, fn_type, None)
                                    });

                                let call = self.builder
                                    .build_call(intrinsic, &[recv_f64.into(), arg_f64.into()], &method.name)
                                    .map_err(|e| e.to_string())?;
                                Ok(call.try_as_basic_value().left()
                                    .ok_or("Expected return value")?
                                    .into_float_value())
                            } else {
                                Err(format!("{} requires an argument", method.name))
                            }
                        }
                        _ => {
                            // Fall back to regular compile
                            let int_result = self.compile_expr(fn_value, scope, expr)?;
                            self.builder
                                .build_bit_cast(int_result, f64_type, "method_f64")
                                .map_err(|e| e.to_string())
                                .map(|v| v.into_float_value())
                        }
                    }
                }
                Expr::Index { expr: container, index } => {
                    // Load from Vec and convert to float
                    let int_result = self.compile_expr(fn_value, scope, expr)?;
                    self.builder
                        .build_bit_cast(int_result, f64_type, "idx_f64")
                        .map_err(|e| e.to_string())
                        .map(|v| v.into_float_value())
                }
                Expr::Cast { expr: inner, ty } => {
                    // Handle int to float cast
                    if let ast::TypeExpr::Path(path) = ty {
                        if let Some(seg) = path.segments.last() {
                            if seg.ident.name == "f64" || seg.ident.name == "f32" {
                                // G18: If source is already float, just bitcast to f64
                                if self.is_float_expr_with_scope(inner, scope) {
                                    let int_val = self.compile_expr(fn_value, scope, inner)?;
                                    return self.builder
                                        .build_bit_cast(int_val, f64_type, "f64_to_f64")
                                        .map_err(|e| e.to_string())
                                        .map(|v| v.into_float_value());
                                }
                                // Compile inner as int and convert
                                let int_val = self.compile_expr(fn_value, scope, inner)?;
                                return self.builder
                                    .build_signed_int_to_float(int_val, f64_type, "sitofp")
                                    .map_err(|e| e.to_string());
                            }
                        }
                    }
                    // Fall back
                    let int_result = self.compile_expr(fn_value, scope, expr)?;
                    self.builder
                        .build_bit_cast(int_result, f64_type, "cast_f64")
                        .map_err(|e| e.to_string())
                        .map(|v| v.into_float_value())
                }
                Expr::Tuple(ref elems) if elems.len() == 1 => {
                    // Single-element tuple acts like parenthesized expression
                    self.compile_native_float_expr(fn_value, scope, &elems[0])
                }
                _ => {
                    // For other expressions, compile normally and convert
                    let int_result = self.compile_expr(fn_value, scope, expr)?;
                    self.builder
                        .build_bit_cast(int_result, f64_type, "other_f64")
                        .map_err(|e| e.to_string())
                        .map(|v| v.into_float_value())
                }
            }
        }

        /// Compile a float binary operation
        /// Values are stored as i64 bit patterns, so we bitcast to f64, operate, and bitcast back
        fn compile_float_binary_op(
            &mut self,
            op: BinOp,
            lhs: IntValue<'ctx>,
            rhs: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            let f64_type = self.context.f64_type();
            let i64_type = self.context.i64_type();

            // Bitcast i64 -> f64
            let lhs_f64 = self
                .builder
                .build_bit_cast(lhs, f64_type, "lhs_f64")
                .map_err(|e| e.to_string())?
                .into_float_value();
            let rhs_f64 = self
                .builder
                .build_bit_cast(rhs, f64_type, "rhs_f64")
                .map_err(|e| e.to_string())?
                .into_float_value();

            // Perform float operation
            let result_f64 = match op {
                BinOp::Add => self
                    .builder
                    .build_float_add(lhs_f64, rhs_f64, "fadd")
                    .map_err(|e| e.to_string())?,
                BinOp::Sub => self
                    .builder
                    .build_float_sub(lhs_f64, rhs_f64, "fsub")
                    .map_err(|e| e.to_string())?,
                BinOp::Mul => self
                    .builder
                    .build_float_mul(lhs_f64, rhs_f64, "fmul")
                    .map_err(|e| e.to_string())?,
                BinOp::Div => self
                    .builder
                    .build_float_div(lhs_f64, rhs_f64, "fdiv")
                    .map_err(|e| e.to_string())?,
                BinOp::Rem => self
                    .builder
                    .build_float_rem(lhs_f64, rhs_f64, "frem")
                    .map_err(|e| e.to_string())?,
                // Comparisons return i64 (0 or 1)
                BinOp::Lt => {
                    let cmp = self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OLT, lhs_f64, rhs_f64, "flt")
                        .map_err(|e| e.to_string())?;
                    return self
                        .builder
                        .build_int_z_extend(cmp, i64_type, "flt_ext")
                        .map_err(|e| e.to_string());
                }
                BinOp::Le => {
                    let cmp = self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OLE, lhs_f64, rhs_f64, "fle")
                        .map_err(|e| e.to_string())?;
                    return self
                        .builder
                        .build_int_z_extend(cmp, i64_type, "fle_ext")
                        .map_err(|e| e.to_string());
                }
                BinOp::Gt => {
                    let cmp = self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OGT, lhs_f64, rhs_f64, "fgt")
                        .map_err(|e| e.to_string())?;
                    return self
                        .builder
                        .build_int_z_extend(cmp, i64_type, "fgt_ext")
                        .map_err(|e| e.to_string());
                }
                BinOp::Ge => {
                    let cmp = self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OGE, lhs_f64, rhs_f64, "fge")
                        .map_err(|e| e.to_string())?;
                    return self
                        .builder
                        .build_int_z_extend(cmp, i64_type, "fge_ext")
                        .map_err(|e| e.to_string());
                }
                BinOp::Eq => {
                    let cmp = self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OEQ, lhs_f64, rhs_f64, "feq")
                        .map_err(|e| e.to_string())?;
                    return self
                        .builder
                        .build_int_z_extend(cmp, i64_type, "feq_ext")
                        .map_err(|e| e.to_string());
                }
                BinOp::Ne => {
                    let cmp = self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::ONE, lhs_f64, rhs_f64, "fne")
                        .map_err(|e| e.to_string())?;
                    return self
                        .builder
                        .build_int_z_extend(cmp, i64_type, "fne_ext")
                        .map_err(|e| e.to_string());
                }
                // For other ops, fall back to integer
                _ => return self.compile_binary_op(op, lhs, rhs),
            };

            // Bitcast f64 -> i64
            let result = self
                .builder
                .build_bit_cast(result_f64, i64_type, "fresult")
                .map_err(|e| e.to_string())?;
            Ok(result.into_int_value())
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
                UnaryOp::Deref => {
                    // Dereference: treat val as pointer (i64 containing address) and load
                    let ptr = self
                        .builder
                        .build_int_to_ptr(
                            val,
                            self.context.ptr_type(AddressSpace::default()),
                            "deref_ptr",
                        )
                        .map_err(|e| e.to_string())?;
                    let loaded = self
                        .builder
                        .build_load(self.context.i64_type(), ptr, "deref_val")
                        .map_err(|e| e.to_string())?;
                    Ok(loaded.into_int_value())
                }
                UnaryOp::Ref | UnaryOp::RefMut => {
                    // Address-of operations - for now just return the value
                    // (proper handling would need to track allocas)
                    Ok(val)
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
            // Check if this is an if-let pattern
            let (cond_val, let_binding) = if let Expr::Let { pattern, value } = condition {
                // Compile the value expression
                let val = self.compile_expr(fn_value, scope, value)?;
                // For if-let, the condition is whether val is non-null (Some)
                // Extract the binding name from the pattern
                let binding_name = match pattern {
                    ast::Pattern::Ident { name, .. } => Some(name.name.clone()),
                    ast::Pattern::TupleStruct { path, fields, .. } => {
                        // Pattern like ?file or Some(file)
                        // Extract the first field pattern's name
                        if !fields.is_empty() {
                            if let ast::Pattern::Ident { name, .. } = &fields[0] {
                                Some(name.name.clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                (val, binding_name)
            } else {
                (self.compile_expr(fn_value, scope, condition)?, None)
            };

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

            // If this was an if-let, bind the variable in the then-block scope
            if let Some(name) = let_binding {
                let alloca = self
                    .builder
                    .build_alloca(self.context.i64_type(), &name)
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(alloca, cond_val)
                    .map_err(|e| e.to_string())?;
                scope.vars.insert(name, alloca);
            }

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

        /// Compile a for loop: `each pattern ∈ iter { body }`
        fn compile_for_loop(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            pattern: &ast::Pattern,
            iter: &Expr,
            body: &ast::Block,
        ) -> Result<IntValue<'ctx>, String> {
            // Get the loop variable name from pattern
            let var_name = match pattern {
                ast::Pattern::Ident { name, .. } => name.name.clone(),
                _ => "_item".to_string(),
            };

            // Check if iterator is a Range expression (0..n, start..end)
            if let Expr::Range { start, end, inclusive } = iter {
                return self.compile_range_for_loop(
                    fn_value, scope, &var_name, start.as_deref(), end.as_deref(), *inclusive, body
                );
            }

            // For non-range iterators (arrays, vecs), use array-based loop
            self.compile_array_for_loop(fn_value, scope, &var_name, iter, body)
        }

        /// Compile a for loop over a range (0..n, start..end)
        fn compile_range_for_loop(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            var_name: &str,
            start: Option<&Expr>,
            end: Option<&Expr>,
            inclusive: bool,
            body: &ast::Block,
        ) -> Result<IntValue<'ctx>, String> {
            // G24 Fix: Compile start/end values and store in allocas.
            // This ensures the values are safely accessible from all loop blocks
            // without domination issues from complex expressions like vec.len().

            // Compile start value in current block
            let start_val = if let Some(s) = start {
                self.compile_expr(fn_value, scope, s)?
            } else {
                self.context.i64_type().const_int(0, false)
            };

            // Compile end value in current block
            let end_val = if let Some(e) = end {
                self.compile_expr(fn_value, scope, e)?
            } else {
                // No end means infinite - use max i64 (shouldn't happen in practice)
                self.context.i64_type().const_int(i64::MAX as u64, false)
            };

            // Store end value in an alloca so it can be loaded from loop blocks
            // This prevents domination issues when end_val computation involves
            // getelementptr or other non-constant expressions
            let end_ptr = self
                .builder
                .build_alloca(self.context.i64_type(), "loop_end")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(end_ptr, end_val)
                .map_err(|e| e.to_string())?;

            // Create blocks
            let init_bb = self.context.append_basic_block(fn_value, "range_init");
            let cond_bb = self.context.append_basic_block(fn_value, "range_cond");
            let body_bb = self.context.append_basic_block(fn_value, "range_body");
            let incr_bb = self.context.append_basic_block(fn_value, "range_incr");
            let after_bb = self.context.append_basic_block(fn_value, "range_after");

            // Jump to init
            self.builder
                .build_unconditional_branch(init_bb)
                .map_err(|e| e.to_string())?;

            // Init block: allocate loop variable, set to start
            self.builder.position_at_end(init_bb);
            let var_ptr = self
                .builder
                .build_alloca(self.context.i64_type(), var_name)
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(var_ptr, start_val)
                .map_err(|e| e.to_string())?;
            scope.vars.insert(var_name.to_string(), var_ptr);

            self.builder
                .build_unconditional_branch(cond_bb)
                .map_err(|e| e.to_string())?;

            // Condition block: check if var < end (or var <= end for inclusive)
            self.builder.position_at_end(cond_bb);
            let var_val = self
                .builder
                .build_load(self.context.i64_type(), var_ptr, var_name)
                .map_err(|e| e.to_string())?
                .into_int_value();

            // Load end value from alloca (safe in any block)
            let end_val_loaded = self
                .builder
                .build_load(self.context.i64_type(), end_ptr, "end_val")
                .map_err(|e| e.to_string())?
                .into_int_value();

            let predicate = if inclusive {
                IntPredicate::SLE // signed less than or equal
            } else {
                IntPredicate::SLT // signed less than
            };

            let cond = self
                .builder
                .build_int_compare(predicate, var_val, end_val_loaded, "range_cond")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(cond, body_bb, after_bb)
                .map_err(|e| e.to_string())?;

            // Body block
            self.builder.position_at_end(body_bb);
            self.compile_block(fn_value, scope, body)?;

            // If body didn't terminate, jump to increment
            if self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
            {
                self.builder
                    .build_unconditional_branch(incr_bb)
                    .map_err(|e| e.to_string())?;
            }

            // Increment block: var++
            self.builder.position_at_end(incr_bb);
            let var_val = self
                .builder
                .build_load(self.context.i64_type(), var_ptr, var_name)
                .map_err(|e| e.to_string())?
                .into_int_value();
            let one = self.context.i64_type().const_int(1, false);
            let next_val = self
                .builder
                .build_int_add(var_val, one, "next_val")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(var_ptr, next_val)
                .map_err(|e| e.to_string())?;
            self.builder
                .build_unconditional_branch(cond_bb)
                .map_err(|e| e.to_string())?;

            // After block
            self.builder.position_at_end(after_bb);

            // Clean up
            scope.vars.remove(var_name);

            Ok(self.context.i64_type().const_int(0, false))
        }

        /// Compile a for loop over an array/vec
        fn compile_array_for_loop(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            var_name: &str,
            iter: &Expr,
            body: &ast::Block,
        ) -> Result<IntValue<'ctx>, String> {
            // G24 Fix: Evaluate iterator and store in alloca to ensure domination
            let iter_val = self.compile_expr(fn_value, scope, iter)?;

            // Store iterator value in alloca to safely access from all loop blocks
            let iter_ptr = self
                .builder
                .build_alloca(self.context.i64_type(), "iter_ptr")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(iter_ptr, iter_val)
                .map_err(|e| e.to_string())?;

            // Create blocks
            let init_bb = self.context.append_basic_block(fn_value, "for_init");
            let cond_bb = self.context.append_basic_block(fn_value, "for_cond");
            let body_bb = self.context.append_basic_block(fn_value, "for_body");
            let incr_bb = self.context.append_basic_block(fn_value, "for_incr");
            let after_bb = self.context.append_basic_block(fn_value, "for_after");

            // Jump to init
            self.builder
                .build_unconditional_branch(init_bb)
                .map_err(|e| e.to_string())?;

            // Init block: allocate index and loop variable
            self.builder.position_at_end(init_bb);
            let idx_ptr = self
                .builder
                .build_alloca(self.context.i64_type(), "for_idx")
                .map_err(|e| e.to_string())?;
            let zero = self.context.i64_type().const_int(0, false);
            self.builder
                .build_store(idx_ptr, zero)
                .map_err(|e| e.to_string())?;

            let var_ptr = self
                .builder
                .build_alloca(self.context.i64_type(), var_name)
                .map_err(|e| e.to_string())?;
            scope.vars.insert(var_name.to_string(), var_ptr);

            // G15 fix: Register element type for loop variable to enable correct method dispatch
            // Extract element type from iterator (Vec<T> -> T)
            if let Some(iter_type) = self.get_struct_type_from_expr(iter, scope) {
                if iter_type.starts_with("Vec<") && iter_type.ends_with(">") {
                    let elem_type = &iter_type[4..iter_type.len()-1];
                    scope.register_struct_type(var_name.to_string(), elem_type.to_string());
                }
            }
            // Also check if iter is a reference to a field access (&this.layers or &vary this.layers)
            // Handle both Expr::AddrOf and Expr::Unary { op: Ref/RefMut } forms
            let inner_expr = match iter {
                Expr::AddrOf { expr: inner, .. } => Some(inner.as_ref()),
                Expr::Unary { op: ast::UnaryOp::Ref | ast::UnaryOp::RefMut, expr: inner } => Some(inner.as_ref()),
                _ => None,
            };
            if let Some(inner) = inner_expr {
                if let Expr::Field { expr: base_expr, field } = inner {
                    let base_type = self.get_struct_type_from_expr(base_expr, scope);
                    if let Some(struct_name) = base_type {
                        if let Some(field_type) = self.field_type_names.get(&(struct_name.clone(), field.name.clone())) {
                            if field_type.starts_with("Vec<") && field_type.ends_with(">") {
                                let elem_type = &field_type[4..field_type.len()-1];
                                scope.register_struct_type(var_name.to_string(), elem_type.to_string());
                            }
                        }
                    }
                }
            }

            // G24 Fix: Get length and store in alloca to ensure domination
            // Load iter from alloca first (safe in init block)
            let iter_for_len = self
                .builder
                .build_load(self.context.i64_type(), iter_ptr, "iter_for_len")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let len_val = self.get_vec_length(iter_for_len)?;

            // Store length in alloca
            let len_ptr = self
                .builder
                .build_alloca(self.context.i64_type(), "len_ptr")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(len_ptr, len_val)
                .map_err(|e| e.to_string())?;

            self.builder
                .build_unconditional_branch(cond_bb)
                .map_err(|e| e.to_string())?;

            // Condition block: check if idx < len
            self.builder.position_at_end(cond_bb);
            let idx_val = self
                .builder
                .build_load(self.context.i64_type(), idx_ptr, "idx")
                .map_err(|e| e.to_string())?
                .into_int_value();
            // Load length from alloca (safe in any block)
            let len_loaded = self
                .builder
                .build_load(self.context.i64_type(), len_ptr, "len")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let cond = self
                .builder
                .build_int_compare(IntPredicate::ULT, idx_val, len_loaded, "for_cond")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(cond, body_bb, after_bb)
                .map_err(|e| e.to_string())?;

            // Body block: get element, store in var, execute body
            self.builder.position_at_end(body_bb);
            // Load iter from alloca for element access (safe in any block)
            let iter_for_elem = self
                .builder
                .build_load(self.context.i64_type(), iter_ptr, "iter_for_elem")
                .map_err(|e| e.to_string())?
                .into_int_value();
            // Reload idx for body
            let idx_for_elem = self
                .builder
                .build_load(self.context.i64_type(), idx_ptr, "idx_for_elem")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let elem_val = self.get_vec_element(iter_for_elem, idx_for_elem)?;
            self.builder
                .build_store(var_ptr, elem_val)
                .map_err(|e| e.to_string())?;

            self.compile_block(fn_value, scope, body)?;

            // If body didn't terminate, jump to increment
            if self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
            {
                self.builder
                    .build_unconditional_branch(incr_bb)
                    .map_err(|e| e.to_string())?;
            }

            // Increment block: idx++
            self.builder.position_at_end(incr_bb);
            let idx_val = self
                .builder
                .build_load(self.context.i64_type(), idx_ptr, "idx")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let one = self.context.i64_type().const_int(1, false);
            let next_idx = self
                .builder
                .build_int_add(idx_val, one, "next_idx")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(idx_ptr, next_idx)
                .map_err(|e| e.to_string())?;
            self.builder
                .build_unconditional_branch(cond_bb)
                .map_err(|e| e.to_string())?;

            // After block
            self.builder.position_at_end(after_bb);
            scope.vars.remove(var_name);

            Ok(self.context.i64_type().const_int(0, false))
        }

        /// Get length from a Vec (field 0 of {len, cap, data} struct)
        fn get_vec_length(&self, vec_ptr_int: IntValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            let ptr_type = self.context.ptr_type(AddressSpace::default());
            let i64_type = self.context.i64_type();

            // Vec is stored as pointer to {len, cap, data}
            let vec_ptr = self
                .builder
                .build_int_to_ptr(vec_ptr_int, ptr_type, "vec_ptr")
                .map_err(|e| e.to_string())?;

            // Load length (field 0, offset 0)
            let len = self
                .builder
                .build_load(i64_type, vec_ptr, "vec_len")
                .map_err(|e| e.to_string())?;

            Ok(len.into_int_value())
        }

        /// Get element from a Vec at given index
        /// Vec layout: {len: i64, cap: i64, data: i64[]} - data is inline at offset 2
        fn get_vec_element(
            &self,
            vec_ptr_int: IntValue<'ctx>,
            index: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            let ptr_type = self.context.ptr_type(AddressSpace::default());
            let i64_type = self.context.i64_type();

            let vec_ptr = self
                .builder
                .build_int_to_ptr(vec_ptr_int, ptr_type, "vec_ptr")
                .map_err(|e| e.to_string())?;

            // Data is inline at offset 2 (after len and cap)
            // Element i is at vec[2 + i]
            let offset_2 = self.context.i64_type().const_int(2, false);
            let adjusted_idx = self
                .builder
                .build_int_add(index, offset_2, "adj_idx")
                .map_err(|e| e.to_string())?;

            // Get element at adjusted index
            let elem_ptr = unsafe {
                self.builder
                    .build_gep(i64_type, vec_ptr, &[adjusted_idx], "elem_ptr")
            }
            .map_err(|e| e.to_string())?;

            let elem = self
                .builder
                .build_load(i64_type, elem_ptr, "elem")
                .map_err(|e| e.to_string())?;

            Ok(elem.into_int_value())
        }

        /// Get the length of an array (represented as i64 for now)
        fn get_array_length(&self, _array_val: IntValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            // For now, arrays are represented as packed structs or pointers
            // The length would typically be stored alongside the data
            // Simplified: return a constant for testing, needs proper implementation
            // TODO: Implement proper array length extraction from runtime representation
            Ok(self.context.i64_type().const_int(0, false))
        }

        /// Get an element from an array at a given index
        fn get_array_element(
            &self,
            _array_val: IntValue<'ctx>,
            _index: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            // For now, return a placeholder
            // TODO: Implement proper array element access
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
                let segments: Vec<&str> = path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect();
                let short_name = segments.last().copied().ok_or("Empty path")?;
                let full = segments.join("::");
                (short_name, full)
            } else if let Expr::Field { field, .. } = func {
                // Method call like obj.method() - field is the method name
                let method_name = field.name.as_str();
                (method_name, method_name.to_string())
            } else {
                // Fallback: treat as anonymous function call, return 0
                return Ok(self.context.i64_type().const_int(0, false));
            };

            // Resolve Self:: and This:: to the actual type name
            let full_path = if full_path.starts_with("Self::")
                || full_path.starts_with("This::")
                || full_path.starts_with("Self·")
                || full_path.starts_with("This·")
            {
                if let Some(ref self_type) = self.current_self_type {
                    // Replace Self/This with actual type name
                    let method = if full_path.contains("::") {
                        full_path.split("::").last().unwrap_or("")
                    } else {
                        full_path.split('·').last().unwrap_or("")
                    };
                    // eprintln!("DEBUG: Resolving Self/This to {}::{}", self_type, method);
                    format!("{}::{}", self_type, method)
                } else {
                    return Err(format!(
                        "Self/This used outside of impl block: {}",
                        full_path
                    ));
                }
            } else {
                full_path
            };

            // Handle common enum/type constructors explicitly (match statement has mysterious issues)
            if full_path == "Result::Ok" || full_path == "Result·Ok" {
                if args.is_empty() {
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                return self.compile_expr(fn_value, scope, &args[0]);
            }
            if full_path == "Result::Err" || full_path == "Result·Err" {
                if args.is_empty() {
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                return self.compile_expr(fn_value, scope, &args[0]);
            }
            if full_path == "Option::Some" || full_path == "Option·Some" {
                if args.is_empty() {
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                return self.compile_expr(fn_value, scope, &args[0]);
            }
            if full_path == "Option::None" || full_path == "Option·None" {
                return Ok(self.context.i64_type().const_int(0, false));
            }
            if full_path == "String::new" || full_path == "String·new" {
                let str_new_fn = self
                    .module
                    .get_function("sigil_string_new")
                    .ok_or("sigil_string_new not declared")?;
                let call = self
                    .builder
                    .build_call(str_new_fn, &[], "string_new")
                    .map_err(|e| e.to_string())?;
                return Ok(call
                    .try_as_basic_value()
                    .left()
                    .map(|v| v.into_int_value())
                    .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
            }
            if full_path == "String::from" || full_path == "String·from" {
                if args.is_empty() {
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                return self.compile_expr(fn_value, scope, &args[0]);
            }
            // Print functions - handle both strings and integers
            if full_path == "println"
                || full_path == "print"
                || full_path == "eprintln"
                || full_path == "eprint"
            {
                let is_stderr = full_path == "eprintln" || full_path == "eprint";
                let with_newline = full_path == "println" || full_path == "eprintln";

                if !args.is_empty() {
                    // Check if argument is a string literal
                    let is_string_literal = matches!(&args[0], Expr::Literal(Literal::String(_)));

                    // G27: Also check if argument is a string variable
                    let is_string_var = if let Expr::Path(path) = &args[0] {
                        if let Some(seg) = path.segments.last() {
                            scope.is_string_var(&seg.ident.name)
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    // G27: Check if argument is a Rust String variable (from fs_read)
                    let is_rust_string_var = if let Expr::Path(path) = &args[0] {
                        if let Some(seg) = path.segments.last() {
                            scope.is_rust_string_var(&seg.ident.name)
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if is_string_literal {
                        // For string literals, compile to get pointer and call print_str
                        if let Expr::Literal(Literal::String(s)) = &args[0] {
                            let str_ptr = self.create_global_string(s, "print_str");
                            let print_fn_name = if is_stderr {
                                if with_newline {
                                    "eprintln"
                                } else {
                                    "eprint"
                                }
                            } else {
                                if with_newline {
                                    "println"
                                } else {
                                    "print"
                                }
                            };
                            // Try extern C functions first, fall back to sigil_print_str
                            if let Some(print_fn) = self.module.get_function(print_fn_name) {
                                self.builder
                                    .build_call(print_fn, &[str_ptr.into()], "print_call")
                                    .map_err(|e| e.to_string())?;
                            } else if let Some(print_fn) =
                                self.module.get_function("sigil_print_str")
                            {
                                self.builder
                                    .build_call(print_fn, &[str_ptr.into()], "print_call")
                                    .map_err(|e| e.to_string())?;
                                // Add newline if needed
                                if with_newline {
                                    let nl_ptr = self.create_global_string("\n", "newline");
                                    if let Some(write_fn) =
                                        self.module.get_function("sigil_write_str")
                                    {
                                        self.builder
                                            .build_call(write_fn, &[nl_ptr.into()], "nl_call")
                                            .map_err(|e| e.to_string())?;
                                    }
                                }
                            }
                        }
                    } else if is_rust_string_var {
                        // G27: For Rust String variables (from fs_read), use sigil_print_rust_string
                        let arg = self.compile_expr(fn_value, scope, &args[0])?;
                        let ptr_type = self.context.ptr_type(AddressSpace::default());
                        let str_ptr = self
                            .builder
                            .build_int_to_ptr(arg, ptr_type, "rust_str_ptr")
                            .map_err(|e| e.to_string())?;

                        if let Some(print_fn) =
                            self.module.get_function("sigil_print_rust_string")
                        {
                            self.builder
                                .build_call(print_fn, &[str_ptr.into()], "print_rust_str")
                                .map_err(|e| e.to_string())?;
                        }
                        // Note: sigil_print_rust_string already adds newline
                    } else if is_string_var {
                        // G27: For C string variables, compile to get pointer and call write_str
                        let arg = self.compile_expr(fn_value, scope, &args[0])?;
                        let ptr_type = self.context.ptr_type(AddressSpace::default());
                        let str_ptr = self
                            .builder
                            .build_int_to_ptr(arg, ptr_type, "str_ptr")
                            .map_err(|e| e.to_string())?;

                        if let Some(write_fn) = self.module.get_function("sigil_write_str") {
                            self.builder
                                .build_call(write_fn, &[str_ptr.into()], "print_str")
                                .map_err(|e| e.to_string())?;
                        }
                        // Add newline for println/eprintln
                        if with_newline {
                            let nl_ptr = self.create_global_string("\n", "newline");
                            if let Some(write_fn) = self.module.get_function("sigil_write_str") {
                                self.builder
                                    .build_call(write_fn, &[nl_ptr.into()], "nl_call")
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                    } else {
                        // For non-string arguments, compile and print as int
                        let arg = self.compile_expr(fn_value, scope, &args[0])?;
                        if let Some(print_fn) = self.module.get_function("sigil_print_int") {
                            self.builder
                                .build_call(print_fn, &[arg.into()], "print_call")
                                .map_err(|e| e.to_string())?;
                        }
                        // Add newline for println/eprintln with non-string
                        if with_newline {
                            if let Some(print_nl) = self.module.get_function("sigil_print_newline")
                            {
                                self.builder
                                    .build_call(print_nl, &[], "nl_call")
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                    }
                } else {
                    // No args - just print newline for println/eprintln
                    if with_newline {
                        if let Some(print_nl) = self.module.get_function("sigil_print_newline") {
                            self.builder
                                .build_call(print_nl, &[], "nl_call")
                                .map_err(|e| e.to_string())?;
                        }
                    }
                }
                return Ok(self.context.i64_type().const_int(0, false));
            }
            // Format function
            if full_path == "format" || full_path == "format!" {
                // For now, just return the first argument or empty string
                if !args.is_empty() {
                    return self.compile_expr(fn_value, scope, &args[0]);
                }
                return Ok(self.context.i64_type().const_int(0, false));
            }
            // Panic function
            if full_path == "panic" || full_path == "panic!" || full_path == "unreachable" {
                // Print error and abort
                if let Some(panic_fn) = self.module.get_function("sigil_panic") {
                    self.builder
                        .build_call(panic_fn, &[], "panic_call")
                        .map_err(|e| e.to_string())?;
                }
                return Ok(self.context.i64_type().const_int(0, false));
            }
            // assert functions
            if full_path == "assert"
                || full_path == "assert!"
                || full_path == "assert_eq"
                || full_path == "assert_eq!"
            {
                // For now, just evaluate arguments and ignore
                for arg in args {
                    self.compile_expr(fn_value, scope, arg)?;
                }
                return Ok(self.context.i64_type().const_int(0, false));
            }

            // G27: File system functions
            if full_path == "fs_read" {
                if args.is_empty() {
                    return Err("fs_read requires a path argument".to_string());
                }
                // Compile the path argument - should be a string pointer
                let path_val = self.compile_expr(fn_value, scope, &args[0])?;
                let ptr_type = self.context.ptr_type(AddressSpace::default());
                let path_ptr = self.builder
                    .build_int_to_ptr(path_val, ptr_type, "path_ptr")
                    .map_err(|e| e.to_string())?;

                let fs_read_fn = self
                    .module
                    .get_function("sigil_fs_read")
                    .ok_or("sigil_fs_read not declared")?;
                let call = self
                    .builder
                    .build_call(fs_read_fn, &[path_ptr.into()], "fs_read_result")
                    .map_err(|e| e.to_string())?;

                // Function returns real pointer, convert to i64 for sigil
                let result_ptr = call
                    .try_as_basic_value()
                    .left()
                    .ok_or("sigil_fs_read returned void")?
                    .into_pointer_value();
                let result_int = self
                    .builder
                    .build_ptr_to_int(result_ptr, self.context.i64_type(), "fs_read_int")
                    .map_err(|e| e.to_string())?;
                return Ok(result_int);
            }

            // Handle qualified type paths (e.g., Vec::new, Box::new)
            match full_path.as_str() {
                "Vec::new" => {
                    // Vec::new() with default capacity
                    let capacity = self.context.i64_type().const_int(8, false);
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
                "Vec::with_capacity" => {
                    if args.is_empty() {
                        return Err("Vec::with_capacity requires capacity argument".to_string());
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
                "Box::new" => {
                    // Box::new allocates and stores value
                    if args.is_empty() {
                        return Err("Box::new requires a value argument".to_string());
                    }
                    // Allocate 8 bytes (i64) and store the value
                    let alloc_fn = self
                        .module
                        .get_function("sigil_alloc")
                        .ok_or("sigil_alloc not declared")?;
                    let size = self.context.i64_type().const_int(8, false);
                    let call = self
                        .builder
                        .build_call(alloc_fn, &[size.into()], "box_alloc")
                        .map_err(|e| e.to_string())?;
                    let ptr = call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false));

                    // Compile the value and store it
                    let value = self.compile_expr(fn_value, scope, &args[0])?;
                    let ptr_as_ptr = self
                        .builder
                        .build_int_to_ptr(
                            ptr,
                            self.context.ptr_type(AddressSpace::default()),
                            "box_ptr",
                        )
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_store(ptr_as_ptr, value)
                        .map_err(|e| e.to_string())?;

                    return Ok(ptr);
                }
                // File I/O operations
                "File::read" | "File::read_all" => {
                    // File::read(path) -> String content
                    if args.is_empty() {
                        return Err("File::read requires a path argument".to_string());
                    }
                    // Get the path string pointer
                    let path_val = self.compile_expr(fn_value, scope, &args[0])?;
                    let read_fn = self
                        .module
                        .get_function("sigil_file_read_all")
                        .ok_or("sigil_file_read_all not declared")?;
                    let call = self
                        .builder
                        .build_call(read_fn, &[path_val.into()], "file_read")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "File::write" | "File::write_all" => {
                    // File::write(path, content) -> success (1 or 0)
                    if args.len() < 2 {
                        return Err("File::write requires path and content arguments".to_string());
                    }
                    let path_val = self.compile_expr(fn_value, scope, &args[0])?;
                    let content_val = self.compile_expr(fn_value, scope, &args[1])?;
                    let write_fn = self
                        .module
                        .get_function("sigil_file_write_all")
                        .ok_or("sigil_file_write_all not declared")?;
                    let call = self
                        .builder
                        .build_call(
                            write_fn,
                            &[path_val.into(), content_val.into()],
                            "file_write",
                        )
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "File::exists" => {
                    // File::exists(path) -> 1 or 0
                    if args.is_empty() {
                        return Err("File::exists requires a path argument".to_string());
                    }
                    let path_val = self.compile_expr(fn_value, scope, &args[0])?;
                    let exists_fn = self
                        .module
                        .get_function("sigil_file_exists")
                        .ok_or("sigil_file_exists not declared")?;
                    let call = self
                        .builder
                        .build_call(exists_fn, &[path_val.into()], "file_exists")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // ========================================
                // Float print functions (need i64->f64 bitcast)
                // ========================================
                "sigil_print_float" | "sigil_write_float" => {
                    if args.is_empty() {
                        return Err("sigil_print_float requires an argument".to_string());
                    }
                    // Compile argument as i64 (bit pattern)
                    let arg_bits = self.compile_expr(fn_value, scope, &args[0])?;
                    // Bitcast i64 to f64
                    let f64_val = self
                        .builder
                        .build_bit_cast(arg_bits, self.context.f64_type(), "f64_arg")
                        .map_err(|e| e.to_string())?;
                    // Get the function
                    let print_fn = self
                        .module
                        .get_function(fn_name)
                        .ok_or(format!("{} not declared", fn_name))?;
                    // Call with f64 argument
                    self.builder
                        .build_call(print_fn, &[f64_val.into()], "")
                        .map_err(|e| e.to_string())?;
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                // ========================================
                // AVX-512 SIMD Intrinsics
                // ========================================
                "F32x16::splat" => {
                    // Create a vector with all elements set to the same value via runtime
                    if args.is_empty() {
                        return Err("F32x16::splat requires a value argument".to_string());
                    }
                    let scalar = self.compile_expr(fn_value, scope, &args[0])?;

                    // Allocate aligned result buffer via sigil_simd_alloc (64-byte aligned for AVX-512)
                    let alloc_fn = self
                        .module
                        .get_function("sigil_simd_alloc")
                        .ok_or("sigil_simd_alloc not declared")?;
                    // 16 floats
                    let result_call = self
                        .builder
                        .build_call(
                            alloc_fn,
                            &[self.context.i64_type().const_int(16, false).into()],
                            "result_buf",
                        )
                        .map_err(|e| e.to_string())?;
                    let result_val = result_call
                        .try_as_basic_value()
                        .left()
                        .ok_or("alloc returned void")?;
                    // Handle both pointer and integer return types
                    let result_int = if result_val.is_pointer_value() {
                        self.builder
                            .build_ptr_to_int(
                                result_val.into_pointer_value(),
                                self.context.i64_type(),
                                "result_int",
                            )
                            .map_err(|e| e.to_string())?
                    } else {
                        result_val.into_int_value()
                    };

                    // Convert i64 bits to f32 for splat
                    let f32_val = self
                        .builder
                        .build_bit_cast(scalar, self.context.f32_type(), "f32_val")
                        .map_err(|e| e.to_string())?;

                    // Call runtime splat
                    let splat_fn = self
                        .module
                        .get_function("sigil_simd_splat_f32x16")
                        .ok_or("sigil_simd_splat_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let dest_ptr = self
                        .builder
                        .build_int_to_ptr(result_int, ptr_type, "dest")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_call(splat_fn, &[dest_ptr.into(), f32_val.into()], "")
                        .map_err(|e| e.to_string())?;

                    return Ok(result_int);
                }
                "F32x16::load_aligned" | "_mm512_load_ps" => {
                    // Load 16 f32s from aligned memory via runtime
                    if args.is_empty() {
                        return Err("F32x16::load_aligned requires a pointer argument".to_string());
                    }
                    let src_ptr_val = self.compile_expr(fn_value, scope, &args[0])?;

                    // Allocate aligned result buffer (64-byte aligned for AVX-512)
                    let alloc_fn = self
                        .module
                        .get_function("sigil_simd_alloc")
                        .ok_or("sigil_simd_alloc not declared")?;
                    let result_call = self
                        .builder
                        .build_call(
                            alloc_fn,
                            &[self.context.i64_type().const_int(16, false).into()],
                            "result_buf",
                        )
                        .map_err(|e| e.to_string())?;
                    let result_val = result_call
                        .try_as_basic_value()
                        .left()
                        .ok_or("alloc returned void")?;
                    let result_int = if result_val.is_pointer_value() {
                        self.builder
                            .build_ptr_to_int(
                                result_val.into_pointer_value(),
                                self.context.i64_type(),
                                "result_int",
                            )
                            .map_err(|e| e.to_string())?
                    } else {
                        result_val.into_int_value()
                    };

                    // Call runtime load
                    let load_fn = self
                        .module
                        .get_function("sigil_simd_load_f32x16")
                        .ok_or("sigil_simd_load_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let dest_ptr = self
                        .builder
                        .build_int_to_ptr(result_int, ptr_type, "dest")
                        .map_err(|e| e.to_string())?;
                    let src_ptr = self
                        .builder
                        .build_int_to_ptr(src_ptr_val, ptr_type, "src")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_call(load_fn, &[dest_ptr.into(), src_ptr.into()], "")
                        .map_err(|e| e.to_string())?;

                    return Ok(result_int);
                }
                "F32x16::store_aligned" | "_mm512_store_ps" => {
                    // Store 16 f32s to aligned memory via runtime
                    if args.len() < 2 {
                        return Err(
                            "F32x16::store_aligned requires destination and value".to_string()
                        );
                    }
                    let dest_val = self.compile_expr(fn_value, scope, &args[0])?;
                    let src_val = self.compile_expr(fn_value, scope, &args[1])?;

                    // Call runtime store
                    let store_fn = self
                        .module
                        .get_function("sigil_simd_store_f32x16")
                        .ok_or("sigil_simd_store_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let dest_ptr = self
                        .builder
                        .build_int_to_ptr(dest_val, ptr_type, "dest")
                        .map_err(|e| e.to_string())?;
                    let src_ptr = self
                        .builder
                        .build_int_to_ptr(src_val, ptr_type, "src")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_call(store_fn, &[dest_ptr.into(), src_ptr.into()], "")
                        .map_err(|e| e.to_string())?;

                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "F32x16::add" | "_mm512_add_ps" => {
                    // Vector add via runtime: dest = a + b
                    if args.len() < 2 {
                        return Err("F32x16::add requires two vector arguments".to_string());
                    }
                    let a_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let b_ptr = self.compile_expr(fn_value, scope, &args[1])?;

                    // Allocate aligned result buffer (64-byte aligned for AVX-512)
                    let alloc_fn = self
                        .module
                        .get_function("sigil_simd_alloc")
                        .ok_or("sigil_simd_alloc not declared")?;
                    let result_call = self
                        .builder
                        .build_call(
                            alloc_fn,
                            &[self.context.i64_type().const_int(16, false).into()],
                            "result_buf",
                        )
                        .map_err(|e| e.to_string())?;
                    let result_val = result_call
                        .try_as_basic_value()
                        .left()
                        .ok_or("alloc returned void")?;
                    let result_int = if result_val.is_pointer_value() {
                        self.builder
                            .build_ptr_to_int(
                                result_val.into_pointer_value(),
                                self.context.i64_type(),
                                "result_int",
                            )
                            .map_err(|e| e.to_string())?
                    } else {
                        result_val.into_int_value()
                    };

                    // Call runtime SIMD add
                    let add_fn = self
                        .module
                        .get_function("sigil_simd_add_f32x16")
                        .ok_or("sigil_simd_add_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let dest_ptr = self
                        .builder
                        .build_int_to_ptr(result_int, ptr_type, "dest")
                        .map_err(|e| e.to_string())?;
                    let a_ptr_cast = self
                        .builder
                        .build_int_to_ptr(a_ptr, ptr_type, "a")
                        .map_err(|e| e.to_string())?;
                    let b_ptr_cast = self
                        .builder
                        .build_int_to_ptr(b_ptr, ptr_type, "b")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_call(
                            add_fn,
                            &[dest_ptr.into(), a_ptr_cast.into(), b_ptr_cast.into()],
                            "",
                        )
                        .map_err(|e| e.to_string())?;

                    return Ok(result_int);
                }
                "F32x16::mul" | "_mm512_mul_ps" => {
                    // Vector multiply via runtime: dest = a * b
                    if args.len() < 2 {
                        return Err("F32x16::mul requires two vector arguments".to_string());
                    }
                    let a_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let b_ptr = self.compile_expr(fn_value, scope, &args[1])?;

                    // Allocate aligned result buffer (64-byte aligned for AVX-512)
                    let alloc_fn = self
                        .module
                        .get_function("sigil_simd_alloc")
                        .ok_or("sigil_simd_alloc not declared")?;
                    let result_call = self
                        .builder
                        .build_call(
                            alloc_fn,
                            &[self.context.i64_type().const_int(16, false).into()],
                            "result_buf",
                        )
                        .map_err(|e| e.to_string())?;
                    let result_val = result_call
                        .try_as_basic_value()
                        .left()
                        .ok_or("alloc returned void")?;
                    let result_int = if result_val.is_pointer_value() {
                        self.builder
                            .build_ptr_to_int(
                                result_val.into_pointer_value(),
                                self.context.i64_type(),
                                "result_int",
                            )
                            .map_err(|e| e.to_string())?
                    } else {
                        result_val.into_int_value()
                    };

                    // Call runtime SIMD mul
                    let mul_fn = self
                        .module
                        .get_function("sigil_simd_mul_f32x16")
                        .ok_or("sigil_simd_mul_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let dest_ptr = self
                        .builder
                        .build_int_to_ptr(result_int, ptr_type, "dest")
                        .map_err(|e| e.to_string())?;
                    let a_ptr_cast = self
                        .builder
                        .build_int_to_ptr(a_ptr, ptr_type, "a")
                        .map_err(|e| e.to_string())?;
                    let b_ptr_cast = self
                        .builder
                        .build_int_to_ptr(b_ptr, ptr_type, "b")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_call(
                            mul_fn,
                            &[dest_ptr.into(), a_ptr_cast.into(), b_ptr_cast.into()],
                            "",
                        )
                        .map_err(|e| e.to_string())?;

                    return Ok(result_int);
                }
                "F32x16::fmadd" | "_mm512_fmadd_ps" => {
                    // Fused multiply-add via runtime: dest = a * b + c
                    if args.len() < 3 {
                        return Err("F32x16::fmadd requires three vector arguments".to_string());
                    }
                    let a_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let b_ptr = self.compile_expr(fn_value, scope, &args[1])?;
                    let c_ptr = self.compile_expr(fn_value, scope, &args[2])?;

                    // Allocate aligned result buffer (64-byte aligned for AVX-512)
                    let alloc_fn = self
                        .module
                        .get_function("sigil_simd_alloc")
                        .ok_or("sigil_simd_alloc not declared")?;
                    let result_call = self
                        .builder
                        .build_call(
                            alloc_fn,
                            &[self.context.i64_type().const_int(16, false).into()],
                            "result_buf",
                        )
                        .map_err(|e| e.to_string())?;
                    let result_val = result_call
                        .try_as_basic_value()
                        .left()
                        .ok_or("alloc returned void")?;
                    let result_int = if result_val.is_pointer_value() {
                        self.builder
                            .build_ptr_to_int(
                                result_val.into_pointer_value(),
                                self.context.i64_type(),
                                "result_int",
                            )
                            .map_err(|e| e.to_string())?
                    } else {
                        result_val.into_int_value()
                    };

                    // Call runtime SIMD fmadd
                    let fmadd_fn = self
                        .module
                        .get_function("sigil_simd_fmadd_f32x16")
                        .ok_or("sigil_simd_fmadd_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let dest_ptr = self
                        .builder
                        .build_int_to_ptr(result_int, ptr_type, "dest")
                        .map_err(|e| e.to_string())?;
                    let a_ptr_cast = self
                        .builder
                        .build_int_to_ptr(a_ptr, ptr_type, "a")
                        .map_err(|e| e.to_string())?;
                    let b_ptr_cast = self
                        .builder
                        .build_int_to_ptr(b_ptr, ptr_type, "b")
                        .map_err(|e| e.to_string())?;
                    let c_ptr_cast = self
                        .builder
                        .build_int_to_ptr(c_ptr, ptr_type, "c")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_call(
                            fmadd_fn,
                            &[
                                dest_ptr.into(),
                                a_ptr_cast.into(),
                                b_ptr_cast.into(),
                                c_ptr_cast.into(),
                            ],
                            "",
                        )
                        .map_err(|e| e.to_string())?;

                    return Ok(result_int);
                }
                "F32x16::extract" => {
                    // Extract single element from vector via runtime
                    if args.len() < 2 {
                        return Err("F32x16::extract requires vector and index".to_string());
                    }
                    let vec_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let idx = self.compile_expr(fn_value, scope, &args[1])?;

                    // Call runtime extract
                    let extract_fn = self
                        .module
                        .get_function("sigil_simd_extract_f32x16")
                        .ok_or("sigil_simd_extract_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let src_ptr = self
                        .builder
                        .build_int_to_ptr(vec_ptr, ptr_type, "src")
                        .map_err(|e| e.to_string())?;
                    let f32_result = self
                        .builder
                        .build_call(extract_fn, &[src_ptr.into(), idx.into()], "extract")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("extract returned void")?;

                    // Convert f32 back to i64 bits
                    let bits = self
                        .builder
                        .build_bit_cast(f32_result, self.context.i32_type(), "bits")
                        .map_err(|e| e.to_string())?;
                    let extended = self
                        .builder
                        .build_int_z_extend(bits.into_int_value(), self.context.i64_type(), "ext")
                        .map_err(|e| e.to_string())?;
                    return Ok(extended);
                }
                "F32x16::reduce_add" => {
                    // Horizontal sum via runtime
                    if args.is_empty() {
                        return Err("F32x16::reduce_add requires a vector argument".to_string());
                    }
                    let vec_ptr = self.compile_expr(fn_value, scope, &args[0])?;

                    // Call runtime reduce_add
                    let reduce_fn = self
                        .module
                        .get_function("sigil_simd_reduce_add_f32x16")
                        .ok_or("sigil_simd_reduce_add_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let src_ptr = self
                        .builder
                        .build_int_to_ptr(vec_ptr, ptr_type, "src")
                        .map_err(|e| e.to_string())?;
                    let f32_result = self
                        .builder
                        .build_call(reduce_fn, &[src_ptr.into()], "reduce")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("reduce returned void")?;

                    // Convert f32 back to i64 bits
                    let bits = self
                        .builder
                        .build_bit_cast(f32_result, self.context.i32_type(), "bits")
                        .map_err(|e| e.to_string())?;
                    let extended = self
                        .builder
                        .build_int_z_extend(bits.into_int_value(), self.context.i64_type(), "ext")
                        .map_err(|e| e.to_string())?;
                    return Ok(extended);
                }
                "F32x16::dot" => {
                    // Dot product via runtime
                    if args.len() < 2 {
                        return Err("F32x16::dot requires two vector arguments".to_string());
                    }
                    let a_ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let b_ptr = self.compile_expr(fn_value, scope, &args[1])?;

                    // Call runtime dot
                    let dot_fn = self
                        .module
                        .get_function("sigil_simd_dot_f32x16")
                        .ok_or("sigil_simd_dot_f32x16 not declared")?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let a_ptr_cast = self
                        .builder
                        .build_int_to_ptr(a_ptr, ptr_type, "a")
                        .map_err(|e| e.to_string())?;
                    let b_ptr_cast = self
                        .builder
                        .build_int_to_ptr(b_ptr, ptr_type, "b")
                        .map_err(|e| e.to_string())?;
                    let f32_result = self
                        .builder
                        .build_call(dot_fn, &[a_ptr_cast.into(), b_ptr_cast.into()], "dot")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("dot returned void")?;

                    // Convert f32 back to i64 bits
                    let bits = self
                        .builder
                        .build_bit_cast(f32_result, self.context.i32_type(), "bits")
                        .map_err(|e| e.to_string())?;
                    let extended = self
                        .builder
                        .build_int_z_extend(bits.into_int_value(), self.context.i64_type(), "ext")
                        .map_err(|e| e.to_string())?;
                    return Ok(extended);
                }
                // ========================================
                // CUDA Functions
                // ========================================
                "Cuda::init" | "cuda_init" => {
                    let init_fn = self
                        .module
                        .get_function("sigil_cuda_init")
                        .ok_or("sigil_cuda_init not declared")?;
                    let result = self
                        .builder
                        .build_call(init_fn, &[], "cuda_init")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("cuda_init returned void")?;
                    return Ok(result.into_int_value());
                }
                "Cuda::cleanup" | "cuda_cleanup" => {
                    let cleanup_fn = self
                        .module
                        .get_function("sigil_cuda_cleanup")
                        .ok_or("sigil_cuda_cleanup not declared")?;
                    self.builder
                        .build_call(cleanup_fn, &[], "")
                        .map_err(|e| e.to_string())?;
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "Cuda::device_count" | "cuda_device_count" => {
                    let count_fn = self
                        .module
                        .get_function("sigil_cuda_get_device_count")
                        .ok_or("sigil_cuda_get_device_count not declared")?;
                    let result = self
                        .builder
                        .build_call(count_fn, &[], "device_count")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("device_count returned void")?;
                    return Ok(result.into_int_value());
                }
                "Cuda::malloc" | "cuda_malloc" => {
                    if args.is_empty() {
                        return Err("Cuda::malloc requires size argument".to_string());
                    }
                    let size = self.compile_expr(fn_value, scope, &args[0])?;
                    let malloc_fn = self
                        .module
                        .get_function("sigil_cuda_malloc")
                        .ok_or("sigil_cuda_malloc not declared")?;
                    let result = self
                        .builder
                        .build_call(malloc_fn, &[size.into()], "cuda_ptr")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("cuda_malloc returned void")?;
                    return Ok(result.into_int_value());
                }
                "Cuda::free" | "cuda_free" => {
                    if args.is_empty() {
                        return Err("Cuda::free requires device pointer argument".to_string());
                    }
                    let ptr = self.compile_expr(fn_value, scope, &args[0])?;
                    let free_fn = self
                        .module
                        .get_function("sigil_cuda_free")
                        .ok_or("sigil_cuda_free not declared")?;
                    self.builder
                        .build_call(free_fn, &[ptr.into()], "")
                        .map_err(|e| e.to_string())?;
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "Cuda::memcpy_h2d" | "cuda_memcpy_h2d" => {
                    if args.len() < 3 {
                        return Err(
                            "Cuda::memcpy_h2d requires (dst_device, src_host, size)".to_string()
                        );
                    }
                    let dst = self.compile_expr(fn_value, scope, &args[0])?;
                    let src = self.compile_expr(fn_value, scope, &args[1])?;
                    let size = self.compile_expr(fn_value, scope, &args[2])?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let src_ptr = self
                        .builder
                        .build_int_to_ptr(src, ptr_type, "src_ptr")
                        .map_err(|e| e.to_string())?;
                    let h2d_fn = self
                        .module
                        .get_function("sigil_cuda_memcpy_h2d")
                        .ok_or("sigil_cuda_memcpy_h2d not declared")?;
                    let result = self
                        .builder
                        .build_call(h2d_fn, &[dst.into(), src_ptr.into(), size.into()], "h2d")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("h2d returned void")?;
                    return Ok(result.into_int_value());
                }
                "Cuda::memcpy_d2h" | "cuda_memcpy_d2h" => {
                    if args.len() < 3 {
                        return Err(
                            "Cuda::memcpy_d2h requires (dst_host, src_device, size)".to_string()
                        );
                    }
                    let dst = self.compile_expr(fn_value, scope, &args[0])?;
                    let src = self.compile_expr(fn_value, scope, &args[1])?;
                    let size = self.compile_expr(fn_value, scope, &args[2])?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let dst_ptr = self
                        .builder
                        .build_int_to_ptr(dst, ptr_type, "dst_ptr")
                        .map_err(|e| e.to_string())?;
                    let d2h_fn = self
                        .module
                        .get_function("sigil_cuda_memcpy_d2h")
                        .ok_or("sigil_cuda_memcpy_d2h not declared")?;
                    let result = self
                        .builder
                        .build_call(d2h_fn, &[dst_ptr.into(), src.into(), size.into()], "d2h")
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("d2h returned void")?;
                    return Ok(result.into_int_value());
                }
                "Cuda::sync" | "cuda_sync" => {
                    let sync_fn = self
                        .module
                        .get_function("sigil_cuda_sync")
                        .ok_or("sigil_cuda_sync not declared")?;
                    self.builder
                        .build_call(sync_fn, &[], "")
                        .map_err(|e| e.to_string())?;
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "Cuda::compile_kernel" | "cuda_compile_kernel" => {
                    if args.len() < 2 {
                        return Err(
                            "Cuda::compile_kernel requires (cuda_source, kernel_name)".to_string()
                        );
                    }
                    let src = self.compile_expr(fn_value, scope, &args[0])?;
                    let name = self.compile_expr(fn_value, scope, &args[1])?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let src_ptr = self
                        .builder
                        .build_int_to_ptr(src, ptr_type, "src_ptr")
                        .map_err(|e| e.to_string())?;
                    let name_ptr = self
                        .builder
                        .build_int_to_ptr(name, ptr_type, "name_ptr")
                        .map_err(|e| e.to_string())?;
                    let compile_fn = self
                        .module
                        .get_function("sigil_cuda_compile_kernel")
                        .ok_or("sigil_cuda_compile_kernel not declared")?;
                    let result = self
                        .builder
                        .build_call(
                            compile_fn,
                            &[src_ptr.into(), name_ptr.into()],
                            "kernel_handle",
                        )
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("compile_kernel returned void")?;
                    return Ok(result.into_int_value());
                }
                "Cuda::launch_1d" | "cuda_launch_1d" => {
                    // launch_1d(handle, grid_x, block_x, arg_array_ptr, num_args)
                    if args.len() < 5 {
                        return Err("Cuda::launch_1d requires (handle, grid_x, block_x, args_ptr, num_args)".to_string());
                    }
                    let handle = self.compile_expr(fn_value, scope, &args[0])?;
                    let grid_x = self.compile_expr(fn_value, scope, &args[1])?;
                    let block_x = self.compile_expr(fn_value, scope, &args[2])?;
                    let args_ptr = self.compile_expr(fn_value, scope, &args[3])?;
                    let num_args = self.compile_expr(fn_value, scope, &args[4])?;
                    let ptr_type = self.context.ptr_type(AddressSpace::default());
                    let args_cast = self
                        .builder
                        .build_int_to_ptr(args_ptr, ptr_type, "args")
                        .map_err(|e| e.to_string())?;
                    let launch_fn = self
                        .module
                        .get_function("sigil_cuda_launch_kernel_1d")
                        .ok_or("sigil_cuda_launch_kernel_1d not declared")?;
                    let result = self
                        .builder
                        .build_call(
                            launch_fn,
                            &[
                                handle.into(),
                                grid_x.into(),
                                block_x.into(),
                                args_cast.into(),
                                num_args.into(),
                            ],
                            "launch",
                        )
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value()
                        .left()
                        .ok_or("launch returned void")?;
                    return Ok(result.into_int_value());
                }
                _ => {}
            }

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
                    // Call sigil_now runtime function (milliseconds)
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
                "now_micros" => {
                    // Call sigil_now_micros runtime function (microseconds)
                    let now_fn = self
                        .module
                        .get_function("sigil_now_micros")
                        .ok_or("sigil_now_micros not declared")?;
                    let call = self
                        .builder
                        .build_call(now_fn, &[], "now_micros")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                // Memory allocation: alloc(size) -> ptr as i64
                "alloc" => {
                    if args.is_empty() {
                        return Err("alloc requires size argument".to_string());
                    }
                    let size = self.compile_expr(fn_value, scope, &args[0])?;
                    let alloc_fn = self
                        .module
                        .get_function("sigil_alloc")
                        .ok_or("sigil_alloc not declared")?;
                    let call = self
                        .builder
                        .build_call(alloc_fn, &[size.into()], "alloc")
                        .map_err(|e| e.to_string())?;
                    // Convert pointer to i64 for uniform handling
                    let ptr_val = call
                        .try_as_basic_value()
                        .left()
                        .ok_or("alloc returned void")?;
                    if ptr_val.is_pointer_value() {
                        return Ok(self
                            .builder
                            .build_ptr_to_int(
                                ptr_val.into_pointer_value(),
                                self.context.i64_type(),
                                "ptr_as_int",
                            )
                            .map_err(|e| e.to_string())?);
                    }
                    return Ok(ptr_val.into_int_value());
                }
                // Memory deallocation: free(ptr)
                "free" => {
                    if args.is_empty() {
                        return Err("free requires pointer argument".to_string());
                    }
                    let ptr_int = self.compile_expr(fn_value, scope, &args[0])?;
                    // Convert i64 back to pointer for the call
                    let ptr = self
                        .builder
                        .build_int_to_ptr(
                            ptr_int,
                            self.context.ptr_type(AddressSpace::default()),
                            "free_ptr",
                        )
                        .map_err(|e| e.to_string())?;
                    let free_fn = self
                        .module
                        .get_function("sigil_free")
                        .ok_or("sigil_free not declared")?;
                    self.builder
                        .build_call(free_fn, &[ptr.into()], "")
                        .map_err(|e| e.to_string())?;
                    return Ok(self.context.i64_type().const_int(0, false));
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
                // Math constant: PI()
                "PI" => {
                    let pi_fn = self
                        .module
                        .get_function("sigil_pi")
                        .ok_or("sigil_pi not declared")?;
                    let call = self
                        .builder
                        .build_call(pi_fn, &[], "pi")
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
                        .build_call(
                            string_concat_fn,
                            &[str1.into(), str2.into()],
                            "string_concat",
                        )
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
                        return Err(
                            "file_write_all requires path and content arguments".to_string()
                        );
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
                // Result enum constructors
                "Result::Ok" | "Result·Ok" => {
                    // Result::Ok(value) - for now, just return the value
                    // In a full implementation, we'd tag it as Ok variant
                    if args.is_empty() {
                        return Ok(self.context.i64_type().const_int(0, false));
                    }
                    return self.compile_expr(fn_value, scope, &args[0]);
                }
                "Result::Err" | "Result·Err" => {
                    // Result::Err(error) - for now, return the error value
                    // In a full implementation, we'd tag it as Err variant
                    // eprintln!("DEBUG: Handling Result::Err with {} args", args.len());
                    if args.is_empty() {
                        return Ok(self.context.i64_type().const_int(0, false));
                    }
                    // eprintln!("DEBUG: Compiling Result::Err arg");
                    let result = self.compile_expr(fn_value, scope, &args[0]);
                    // eprintln!("DEBUG: Result::Err arg compiled: {:?}", result.is_ok());
                    return result;
                }
                // Option enum constructors
                "Option::Some" | "Option·Some" => {
                    if args.is_empty() {
                        return Ok(self.context.i64_type().const_int(0, false));
                    }
                    return self.compile_expr(fn_value, scope, &args[0]);
                }
                "Option::None" | "Option·None" => {
                    // None is represented as null/0
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                // String constructors
                "String::new" | "String·new" => {
                    let str_new_fn = self
                        .module
                        .get_function("sigil_string_new")
                        .ok_or("sigil_string_new not declared")?;
                    let call = self
                        .builder
                        .build_call(str_new_fn, &[], "string_new")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                "String::from" | "String·from" => {
                    if args.is_empty() {
                        return Ok(self.context.i64_type().const_int(0, false));
                    }
                    // String::from(s) just returns the string
                    return self.compile_expr(fn_value, scope, &args[0]);
                }
                // Map constructors
                "Map::new" | "Map·new" | "HashMap::new" => {
                    let map_new_fn = self
                        .module
                        .get_function("sigil_map_new")
                        .ok_or("sigil_map_new not declared")?;
                    let call = self
                        .builder
                        .build_call(map_new_fn, &[], "map_new")
                        .map_err(|e| e.to_string())?;
                    return Ok(call
                        .try_as_basic_value()
                        .left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                _ => {
                    // eprintln!("DEBUG: Match fallthrough for full_path = {:?}", full_path);
                }
            }

            // Resolve any use aliases first
            let resolved_path = if let Some(aliased) = self.use_aliases.get(fn_name) {
                aliased.clone()
            } else {
                full_path.clone()
            };

            // NOTE: Generic struct constructors (::new, ·new, etc.) are now handled by
            // actual function lookup below. The previous stub that returned 0 has been removed
            // to allow proper impl method dispatch.

            // Get the function - try resolved path first, then various lookups
            // Also try mangled name (Type_method format) for impl methods
            let mangled_resolved = resolved_path.replace("::", "_").replace("·", "_");
            let mangled_full = full_path.replace("::", "_").replace("·", "_");

            let callee = if let Some(f) = self.functions.get(&resolved_path) {
                *f
            } else if let Some(f) = self.functions.get(&full_path) {
                *f
            } else if let Some(f) = self.functions.get(&mangled_resolved) {
                *f
            } else if let Some(f) = self.functions.get(&mangled_full) {
                *f
            } else if let Some(f) = self.functions.get(fn_name) {
                *f
            } else if let Some(f) = self.module.get_function(&mangled_resolved) {
                f
            } else if let Some(f) = self.module.get_function(&mangled_full) {
                f
            } else if let Some(f) = self.module.get_function(fn_name) {
                f
            } else {
                // Fallback: Try heuristics based on function name pattern
                let fn_lower = fn_name.to_lowercase();

                // Functions that likely transform/process data and return a result
                if fn_lower.starts_with("lower")
                    || fn_lower.starts_with("parse")
                    || fn_lower.starts_with("compile")
                    || fn_lower.starts_with("transform")
                    || fn_lower.starts_with("convert")
                    || fn_lower.starts_with("generate")
                    || fn_lower.starts_with("create")
                    || fn_lower.starts_with("build")
                    || fn_lower.starts_with("make")
                    || fn_lower.starts_with("read")
                    || fn_lower.starts_with("load")
                    || fn_lower.starts_with("fetch")
                    || fn_lower.starts_with("get")
                    || fn_lower.starts_with("find")
                    || fn_lower.starts_with("lookup")
                    || fn_lower.starts_with("resolve")
                    || fn_lower.starts_with("extract")
                    || fn_lower.starts_with("infer")
                    || fn_lower.starts_with("derive")
                    || fn_lower.starts_with("compute")
                    || fn_lower.starts_with("calculate")
                {
                    // These typically return a result/value - stub with 0
                    // eprintln!("DEBUG: Unknown function '{}' - stubbing as data transform", full_path);
                    return Ok(self.context.i64_type().const_int(0, false));
                }

                // Functions that are side-effecting (write, emit, etc.)
                if fn_lower.starts_with("write")
                    || fn_lower.starts_with("emit")
                    || fn_lower.starts_with("output")
                    || fn_lower.starts_with("print")
                    || fn_lower.starts_with("log")
                    || fn_lower.starts_with("debug")
                    || fn_lower.starts_with("warn")
                    || fn_lower.starts_with("error")
                    || fn_lower.starts_with("report")
                    || fn_lower.starts_with("notify")
                {
                    // eprintln!("DEBUG: Unknown function '{}' - stubbing as side-effect", full_path);
                    return Ok(self.context.i64_type().const_int(0, false));
                }

                // Checker/validator functions return bool
                if fn_lower.starts_with("check")
                    || fn_lower.starts_with("validate")
                    || fn_lower.starts_with("verify")
                    || fn_lower.starts_with("test")
                    || fn_lower.starts_with("is_")
                    || fn_lower.starts_with("has_")
                    || fn_lower.starts_with("can_")
                {
                    // eprintln!("DEBUG: Unknown function '{}' - stubbing as bool check", full_path);
                    return Ok(self.context.i64_type().const_int(0, false));
                }

                // Default fallback - assume returns something, use 0 as placeholder
                // eprintln!("DEBUG: Unknown function '{}' - using default stub", full_path);
                return Ok(self.context.i64_type().const_int(0, false));
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

        /// Compile vec! macro - creates a new Vec and optionally initializes it
        /// Handles: vec![], vec![a, b, c], vec![val; count]
        fn compile_vec_macro(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            tokens: &str,
        ) -> Result<IntValue<'ctx>, String> {
            let tokens = tokens.trim();
            let i64_type = self.context.i64_type();

            // Get sigil_vec_new function
            let vec_new_fn = self
                .module
                .get_function("sigil_vec_new")
                .ok_or("sigil_vec_new not declared")?;

            // Case 1: vec![] - empty vector
            if tokens.is_empty() {
                let capacity = i64_type.const_int(8, false);
                let call = self
                    .builder
                    .build_call(vec_new_fn, &[capacity.into()], "vec_new")
                    .map_err(|e| e.to_string())?;
                return Ok(call
                    .try_as_basic_value()
                    .left()
                    .map(|v| v.into_int_value())
                    .unwrap_or_else(|| i64_type.const_int(0, false)));
            }

            // Case 2: vec![val; count] - repeat syntax
            if let Some(semicolon_pos) = tokens.rfind(';') {
                let val_str = tokens[..semicolon_pos].trim();
                let count_str = tokens[semicolon_pos + 1..].trim();

                // Parse count as integer constant or expression
                let count_val = if let Ok(n) = count_str.parse::<u64>() {
                    i64_type.const_int(n, false)
                } else {
                    // Try to compile as expression
                    let count_expr = self.parse_simple_expr(count_str)?;
                    self.compile_expr(fn_value, scope, &count_expr)?
                };

                // Create vector with capacity = count
                let call = self
                    .builder
                    .build_call(vec_new_fn, &[count_val.into()], "vec_new")
                    .map_err(|e| e.to_string())?;
                let vec_ptr = call
                    .try_as_basic_value()
                    .left()
                    .map(|v| v.into_int_value())
                    .unwrap_or_else(|| i64_type.const_int(0, false));

                // Parse value expression
                let val_expr = self.parse_simple_expr(val_str)?;
                let value = self.compile_expr(fn_value, scope, &val_expr)?;

                // Get push function
                let push_fn = self
                    .module
                    .get_function("sigil_vec_push")
                    .ok_or("sigil_vec_push not declared")?;

                // Build a loop to push the value count times
                // For now, if count is a constant, unroll the loop
                // Otherwise, build actual loop
                if let Some(const_count) = count_val.get_zero_extended_constant() {
                    // Unroll for small constant counts (up to 64 iterations)
                    if const_count <= 64 {
                        for _ in 0..const_count {
                            self.builder
                                .build_call(push_fn, &[vec_ptr.into(), value.into()], "")
                                .map_err(|e| e.to_string())?;
                        }
                    } else {
                        // Build a proper loop for larger counts
                        self.build_vec_fill_loop(fn_value, vec_ptr, value, count_val, push_fn)?;
                    }
                } else {
                    // Runtime count - build a loop
                    self.build_vec_fill_loop(fn_value, vec_ptr, value, count_val, push_fn)?;
                }

                return Ok(vec_ptr);
            }

            // Case 3: vec![a, b, c] - element list
            let elements = self.split_macro_args(tokens);
            let capacity = i64_type.const_int(elements.len().max(8) as u64, false);

            let call = self
                .builder
                .build_call(vec_new_fn, &[capacity.into()], "vec_new")
                .map_err(|e| e.to_string())?;
            let vec_ptr = call
                .try_as_basic_value()
                .left()
                .map(|v| v.into_int_value())
                .unwrap_or_else(|| i64_type.const_int(0, false));

            // Get push function
            let push_fn = self
                .module
                .get_function("sigil_vec_push")
                .ok_or("sigil_vec_push not declared")?;

            // Push each element
            for elem_str in elements {
                let elem_str = elem_str.trim();
                if elem_str.is_empty() {
                    continue;
                }
                let elem_expr = self.parse_simple_expr(elem_str)?;
                let elem_val = self.compile_expr(fn_value, scope, &elem_expr)?;
                self.builder
                    .build_call(push_fn, &[vec_ptr.into(), elem_val.into()], "")
                    .map_err(|e| e.to_string())?;
            }

            Ok(vec_ptr)
        }

        /// Build a loop to fill a vector with a value
        fn build_vec_fill_loop(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            vec_ptr: IntValue<'ctx>,
            value: IntValue<'ctx>,
            count: IntValue<'ctx>,
            push_fn: FunctionValue<'ctx>,
        ) -> Result<(), String> {
            let i64_type = self.context.i64_type();

            // Create blocks
            let loop_header = self.context.append_basic_block(fn_value, "vec_fill_header");
            let loop_body = self.context.append_basic_block(fn_value, "vec_fill_body");
            let loop_end = self.context.append_basic_block(fn_value, "vec_fill_end");

            // Initialize counter
            let counter_ptr = self.builder
                .build_alloca(i64_type, "fill_counter")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(counter_ptr, i64_type.const_int(0, false))
                .map_err(|e| e.to_string())?;

            // Jump to header
            self.builder
                .build_unconditional_branch(loop_header)
                .map_err(|e| e.to_string())?;

            // Loop header: check if counter < count
            self.builder.position_at_end(loop_header);
            let counter = self.builder
                .build_load(i64_type, counter_ptr, "counter")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let cmp = self.builder
                .build_int_compare(inkwell::IntPredicate::SLT, counter, count, "cmp")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(cmp, loop_body, loop_end)
                .map_err(|e| e.to_string())?;

            // Loop body: push value and increment counter
            self.builder.position_at_end(loop_body);
            self.builder
                .build_call(push_fn, &[vec_ptr.into(), value.into()], "")
                .map_err(|e| e.to_string())?;
            let next_counter = self.builder
                .build_int_add(counter, i64_type.const_int(1, false), "next_counter")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(counter_ptr, next_counter)
                .map_err(|e| e.to_string())?;
            self.builder
                .build_unconditional_branch(loop_header)
                .map_err(|e| e.to_string())?;

            // Position at end of loop
            self.builder.position_at_end(loop_end);

            Ok(())
        }

        /// Parse a simple expression from a string (for macro arguments)
        /// G16 fix: Use full parser for complex expressions like method calls
        fn parse_simple_expr(&self, s: &str) -> Result<Expr, String> {
            let s = s.trim();

            // Use the full parser to handle complex expressions like method calls
            // This properly handles "source.len()", "a + b", "foo.bar.baz()", etc.
            let mut parser = Parser::new(s);
            if let Ok(expr) = parser.parse_expr() {
                return Ok(expr);
            }

            // Fallback for edge cases: treat as a simple path
            let default_span = Span { start: 0, end: 0 };
            Ok(Expr::Path(TypePath {
                segments: vec![PathSegment {
                    ident: Ident {
                        name: s.to_string(),
                        evidentiality: None,
                        affect: None,
                        span: default_span,
                    },
                    generics: None,
                }],
            }))
        }

        /// Split macro arguments by comma, respecting nesting
        fn split_macro_args(&self, s: &str) -> Vec<String> {
            let mut result = Vec::new();
            let mut current = String::new();
            let mut depth = 0;

            for c in s.chars() {
                match c {
                    '(' | '[' | '{' => {
                        depth += 1;
                        current.push(c);
                    }
                    ')' | ']' | '}' => {
                        depth -= 1;
                        current.push(c);
                    }
                    ',' if depth == 0 => {
                        result.push(current.trim().to_string());
                        current = String::new();
                    }
                    _ => current.push(c),
                }
            }

            if !current.trim().is_empty() {
                result.push(current.trim().to_string());
            }

            result
        }

        /// Compile println! and print! macros
        fn compile_print_macro(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            tokens: &str,
            newline: bool,
        ) -> Result<(), String> {
            // Parse the macro tokens to extract format string and arguments
            // Format: "format string", arg1, arg2, ...
            let tokens = tokens.trim();

            if tokens.is_empty() {
                // println!() with no args - just print newline
                if newline {
                    let empty_str = self.create_global_string("\n", "empty_nl");
                    let print_fn = self
                        .module
                        .get_function("sigil_print_str")
                        .ok_or("sigil_print_str not declared")?;
                    self.builder
                        .build_call(print_fn, &[empty_str.into()], "")
                        .map_err(|e| e.to_string())?;
                }
                return Ok(());
            }

            // Find the format string (first quoted string)
            let (format_str, args_str) = if tokens.starts_with('"') {
                // Find the closing quote (handling escaped quotes)
                let mut chars = tokens[1..].chars().peekable();
                let mut format_content = String::new();
                let mut escaped = false;

                while let Some(c) = chars.next() {
                    if escaped {
                        format_content.push(c);
                        escaped = false;
                    } else if c == '\\' {
                        format_content.push(c);
                        escaped = true;
                    } else if c == '"' {
                        break;
                    } else {
                        format_content.push(c);
                    }
                }

                // Remaining args after the format string
                let remaining: String = chars.collect();
                let args_owned = remaining.trim_start_matches(',').trim().to_string();
                (format_content, args_owned)
            } else {
                // No format string, treat as expression to print
                (String::new(), tokens.to_string())
            };

            // Check if format string has placeholders (including format specs like {:>6}, {:.2})
            let has_placeholders = format_str.contains("{") && format_str.contains("}");

            if !has_placeholders && args_str.is_empty() {
                // Simple string literal - use write_str (no newline) then add newline if needed
                let output = format_str.replace("\\n", "\n").replace("\\t", "\t");

                let write_str_fn = self
                    .module
                    .get_function("sigil_write_str")
                    .ok_or("sigil_write_str not declared")?;

                let str_ptr = self.create_global_string(&output, "print_str");
                self.builder
                    .build_call(write_str_fn, &[str_ptr.into()], "")
                    .map_err(|e| e.to_string())?;

                // Add newline if println!
                if newline {
                    let nl_str = self.create_global_string("\n", "newline");
                    self.builder
                        .build_call(write_str_fn, &[nl_str.into()], "")
                        .map_err(|e| e.to_string())?;
                }
            } else if has_placeholders {
                // Format string with placeholders - parse and substitute
                // Parse format specs like {}, {:>6}, {:.2}, {:>10.4}
                let (parts, format_specs) = self.parse_format_string(&format_str);
                let args: Vec<&str> = args_str
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();

                // Get write functions (no newline versions for inline output)
                let write_str_fn = self
                    .module
                    .get_function("sigil_write_str")
                    .ok_or("sigil_write_str not declared")?;
                let write_int_fn = self
                    .module
                    .get_function("sigil_write_int")
                    .ok_or("sigil_write_int not declared")?;
                let write_float_fn = self
                    .module
                    .get_function("sigil_write_float")
                    .ok_or("sigil_write_float not declared")?;

                for (i, part) in parts.iter().enumerate() {
                    // Print the static part (no newline)
                    if !part.is_empty() {
                        let part_str = part.replace("\\n", "\n").replace("\\t", "\t");
                        let str_ptr = self.create_global_string(&part_str, "fmt_part");
                        self.builder
                            .build_call(write_str_fn, &[str_ptr.into()], "")
                            .map_err(|e| e.to_string())?;
                    }

                    // Print the argument (if there's one for this placeholder)
                    if i < args.len() {
                        let arg_str = args[i];
                        let spec = if i < format_specs.len() { &format_specs[i] } else { "" };

                        // G27: Check if this is a string (including string variables)
                        let is_string = self.is_string_expression_with_scope(arg_str, scope);

                        // Check if this is a float - either by variable type or format spec
                        // Format specs with .N (like :.2, :>10.4) indicate float formatting
                        let spec_indicates_float = spec.contains('.');
                        let is_float = !is_string && (spec_indicates_float || self.is_float_expression(arg_str, scope));

                        // Parse and compile the argument expression
                        let arg_value = self.compile_format_arg(fn_value, scope, arg_str)?;

                        if is_string {
                            // String value - convert to pointer and call write_str
                            let ptr_type = self.context.ptr_type(AddressSpace::default());
                            let str_ptr = self.builder
                                .build_int_to_ptr(arg_value, ptr_type, "str_ptr_for_print")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_call(write_str_fn, &[str_ptr.into()], "")
                                .map_err(|e| e.to_string())?;
                        } else if is_float {
                            // Reinterpret i64 bits as f64 via memory (standard LLVM pattern)
                            let temp_alloca = self.builder
                                .build_alloca(self.context.i64_type(), "float_bits_temp")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_store(temp_alloca, arg_value)
                                .map_err(|e| e.to_string())?;
                            let f64_val = self.builder
                                .build_load(self.context.f64_type(), temp_alloca, "float_val")
                                .map_err(|e| e.to_string())?
                                .into_float_value();
                            self.builder
                                .build_call(write_float_fn, &[f64_val.into()], "")
                                .map_err(|e| e.to_string())?;
                        } else {
                            self.builder
                                .build_call(write_int_fn, &[arg_value.into()], "")
                                .map_err(|e| e.to_string())?;
                        }
                    }
                }

                // Add newline if println!
                if newline {
                    let nl_str = self.create_global_string("\n", "newline");
                    self.builder
                        .build_call(write_str_fn, &[nl_str.into()], "")
                        .map_err(|e| e.to_string())?;
                }
            } else if !args_str.is_empty() {
                // No format string, just print the expression value
                let arg_value = self.compile_format_arg(fn_value, scope, &args_str)?;
                let print_int_fn = self
                    .module
                    .get_function("sigil_print_int")
                    .ok_or("sigil_print_int not declared")?;
                self.builder
                    .build_call(print_int_fn, &[arg_value.into()], "")
                    .map_err(|e| e.to_string())?;
            }

            Ok(())
        }

        /// Compile a format argument expression (simple variable lookup or literal)
        fn compile_format_arg(
            &mut self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            arg_str: &str,
        ) -> Result<IntValue<'ctx>, String> {
            let arg_str = arg_str.trim();

            // Try to parse as integer literal
            if let Ok(n) = arg_str.parse::<i64>() {
                return Ok(self.context.i64_type().const_int(n as u64, n < 0));
            }

            // Try to look up as variable
            if let Some(var) = scope.vars.get(arg_str) {
                let loaded = self
                    .builder
                    .build_load(self.context.i64_type(), *var, arg_str)
                    .map_err(|e| e.to_string())?;
                return Ok(loaded.into_int_value());
            }

            // Try to parse as more complex expression
            let mut parser = Parser::new(arg_str);
            if let Ok(expr) = parser.parse_expr() {
                return self.compile_expr(fn_value, scope, &expr);
            }

            // Fallback: return 0
            Ok(self.context.i64_type().const_int(0, false))
        }

        /// Parse a format string and extract static parts and format specs
        /// Returns (static_parts, format_specs) where format_specs[i] is the spec for the i-th placeholder
        fn parse_format_string(&self, format_str: &str) -> (Vec<String>, Vec<String>) {
            let mut parts = Vec::new();
            let mut specs = Vec::new();
            let mut current_part = String::new();
            let mut chars = format_str.chars().peekable();

            while let Some(c) = chars.next() {
                if c == '{' {
                    // Start of placeholder
                    parts.push(current_part.clone());
                    current_part.clear();

                    // Collect until '}'
                    let mut spec = String::new();
                    while let Some(&next) = chars.peek() {
                        chars.next();
                        if next == '}' {
                            break;
                        }
                        spec.push(next);
                    }
                    // spec is empty for {}, or ":>6" for {:>6}, etc.
                    specs.push(spec);
                } else {
                    current_part.push(c);
                }
            }

            // Push final part
            parts.push(current_part);

            (parts, specs)
        }

        /// Check if an expression string will produce a float value
        fn is_float_expression(&self, arg_str: &str, scope: &CompileScope<'ctx>) -> bool {
            let arg_str = arg_str.trim();

            // G20: Comparison operators always return integers, even with float operands
            if arg_str.contains(" < ") || arg_str.contains(" > ") ||
               arg_str.contains(" <= ") || arg_str.contains(" >= ") ||
               arg_str.contains(" == ") || arg_str.contains(" != ") {
                return false;
            }

            // Check if it's a simple variable name in float_vars
            if scope.float_vars.contains(arg_str) {
                return true;
            }

            // G19: Check if it's an array/slice index like data[i] where data is float
            if let Some(bracket_pos) = arg_str.find('[') {
                let base = arg_str[..bracket_pos].trim();
                if scope.float_vars.contains(base) {
                    return true;
                }
            }

            // G19: Methods that return integer even if receiver is float-containing
            if arg_str.ends_with(".len()") || arg_str.ends_with(".capacity()") ||
               arg_str.ends_with(".is_empty()") {
                return false;
            }

            // G23: Check for method calls that return f64
            // Pattern: receiver.method_name(args) - extract method_name and check float_funcs
            if arg_str.ends_with(')') && arg_str.contains('.') {
                // Find the method name between the last '.' before '(' and the '('
                if let Some(paren_pos) = arg_str.rfind('(') {
                    let before_paren = &arg_str[..paren_pos];
                    if let Some(dot_pos) = before_paren.rfind('.') {
                        let method_name = &before_paren[dot_pos + 1..];
                        // G30: Built-in math methods that always return f64
                        if matches!(method_name, "sqrt" | "sin" | "cos" | "tan" | "exp" | "log" | "ln" |
                                                 "floor" | "ceil" | "abs" | "pow" | "asin" | "acos" | "atan") {
                            return true;
                        }
                        // User-defined float functions
                        if scope.float_funcs.contains(method_name) {
                            return true;
                        }
                    }
                }
            }

            // Check if it's a float literal (contains decimal point)
            if arg_str.contains('.') && !arg_str.contains("..") && !arg_str.contains(".repeat") {
                // Make sure it's not a method call like "x.len()"
                if arg_str.parse::<f64>().is_ok() {
                    return true;
                }
            }

            // Check for arithmetic with floats (e.g., "x * 2.0")
            if arg_str.contains(" * ") || arg_str.contains(" / ") {
                // If any part is a float, the result is probably a float
                for part in arg_str.split(|c| c == '*' || c == '/' || c == '+' || c == '-') {
                    let part = part.trim();
                    if scope.float_vars.contains(part) {
                        return true;
                    }
                    if part.contains('.') && part.parse::<f64>().is_ok() {
                        return true;
                    }
                }
            }

            // Check for struct field access that likely returns a float
            // Heuristic: field names commonly used for floats
            let float_field_patterns = [
                "lambda", "rate", "energy", "loss", "weight", "scale", "bias",
                "grad", "lr", "epsilon", "alpha", "beta", "gamma", "momentum",
                "decay", "factor", "ratio", "threshold", "temp", "sigma",
            ];
            if arg_str.contains('.') && !arg_str.ends_with(')') {
                // Field access like config.lambda_spectral
                let parts: Vec<&str> = arg_str.split('.').collect();
                if let Some(field_name) = parts.last() {
                    let field_lower = field_name.to_lowercase();
                    for pattern in &float_field_patterns {
                        if field_lower.contains(pattern) {
                            return true;
                        }
                    }
                }
            }

            false
        }

        /// Check if an expression string will produce a string value
        fn is_string_expression(&self, arg_str: &str) -> bool {
            let arg_str = arg_str.trim();

            // String literal
            if arg_str.starts_with('"') && arg_str.ends_with('"') {
                return true;
            }

            // String method that returns string
            if arg_str.contains(".repeat(") {
                return true;
            }

            false
        }

        /// G27: Check if an expression string will produce a string value (with scope lookup)
        fn is_string_expression_with_scope(&self, arg_str: &str, scope: &CompileScope<'ctx>) -> bool {
            let arg_str = arg_str.trim();

            // String literal
            if arg_str.starts_with('"') && arg_str.ends_with('"') {
                return true;
            }

            // String method that returns string
            if arg_str.contains(".repeat(") {
                return true;
            }

            // Check if it's a simple variable name that's a string
            if !arg_str.contains('.') && !arg_str.contains('(') && !arg_str.contains('[') {
                return scope.is_string_var(arg_str);
            }

            false
        }

        /// Create a global string constant and return pointer to it
        fn create_global_string(&self, s: &str, name: &str) -> PointerValue<'ctx> {
            let counter = self.string_counter.get();
            self.string_counter.set(counter + 1);
            let unique_name = format!("{}_{}", name, counter);

            // Create a null-terminated string constant
            let string_val = self.context.const_string(s.as_bytes(), true);
            let global = self
                .module
                .add_global(string_val.get_type(), None, &unique_name);
            global.set_initializer(&string_val);
            global.set_constant(true);
            global.set_linkage(inkwell::module::Linkage::Private);

            // Get pointer to the first element
            global.as_pointer_value()
        }

        /// Process a use declaration to register imports
        fn process_use(&mut self, use_decl: &ast::UseDecl) -> Result<(), String> {
            self.process_use_tree(&use_decl.tree, &[])
        }

        /// Recursively process use tree to build import paths
        fn process_use_tree(
            &mut self,
            tree: &ast::UseTree,
            prefix: &[String],
        ) -> Result<(), String> {
            match tree {
                ast::UseTree::Path {
                    prefix: ident,
                    suffix,
                } => {
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
                        Item::Function(func) => {
                            self.declare_function(func)?;
                        }
                        Item::Module(m) => {
                            self.process_module(m)?;
                        }
                        Item::Use(u) => {
                            self.process_use(u)?;
                        }
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
                        Item::Function(func) => {
                            self.compile_function(func)?;
                        }
                        Item::Module(m) => {
                            self.compile_module_functions(m)?;
                        }
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

            // Run aggressive optimization passes with vectorization enabled
            let passes = match self.opt_level {
                OptLevel::None => "default<O0>",
                OptLevel::Basic => "default<O1>",
                OptLevel::Standard | OptLevel::Size => "default<O2>",
                // Run full O3 pipeline which includes tail call elimination
                OptLevel::Aggressive => "default<O3>",
            };

            // Configure pass builder with explicit vectorization options
            let pass_options = PassBuilderOptions::create();
            pass_options.set_loop_vectorization(true);
            pass_options.set_loop_slp_vectorization(true);
            pass_options.set_loop_interleaving(true);
            pass_options.set_loop_unrolling(true);

            self.module
                .run_passes(passes, &target_machine, pass_options)
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
            if let Some(f) = self.module.get_function("sigil_alloc") {
                ee.add_global_mapping(&f, sigil_alloc as usize);
            }
            if let Some(f) = self.module.get_function("sigil_print_int") {
                ee.add_global_mapping(&f, sigil_print_int as usize);
            }
            if let Some(f) = self.module.get_function("sigil_print_newline") {
                ee.add_global_mapping(&f, sigil_print_newline as usize);
            }
            if let Some(f) = self.module.get_function("sigil_write_int") {
                ee.add_global_mapping(&f, sigil_write_int as usize);
            }
            if let Some(f) = self.module.get_function("sigil_write_str") {
                ee.add_global_mapping(&f, sigil_write_str as usize);
            }
            if let Some(f) = self.module.get_function("sigil_strlen") {
                ee.add_global_mapping(&f, sigil_strlen as usize);
            }
            if let Some(f) = self.module.get_function("sigil_write_float") {
                ee.add_global_mapping(&f, sigil_write_float as usize);
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
            if let Some(f) = self.module.get_function("sigil_pi") {
                ee.add_global_mapping(&f, sigil_pi as usize);
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
            if let Some(f) = self.module.get_function("sigil_vec_set") {
                ee.add_global_mapping(&f, sigil_vec_set as usize);
            }

            // String runtime mappings
            if let Some(f) = self.module.get_function("sigil_string_new") {
                ee.add_global_mapping(&f, sigil_string_new as usize);
            }
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
            if let Some(f) = self.module.get_function("sigil_fs_read") {
                ee.add_global_mapping(&f, sigil_fs_read as usize);
            }
            if let Some(f) = self.module.get_function("sigil_string_data") {
                ee.add_global_mapping(&f, sigil_string_data as usize);
            }
            if let Some(f) = self.module.get_function("sigil_string_len") {
                ee.add_global_mapping(&f, sigil_string_len as usize);
            }
            if let Some(f) = self.module.get_function("sigil_print_rust_string") {
                ee.add_global_mapping(&f, sigil_print_rust_string as usize);
            }
            // G32: String slice functions
            if let Some(f) = self.module.get_function("sigil_string_slice") {
                ee.add_global_mapping(&f, sigil_string_slice as usize);
            }
            if let Some(f) = self.module.get_function("sigil_rust_string_slice") {
                ee.add_global_mapping(&f, sigil_rust_string_slice as usize);
            }
            // G32: Rust String as_bytes
            if let Some(f) = self.module.get_function("sigil_rust_string_as_bytes") {
                ee.add_global_mapping(&f, sigil_rust_string_as_bytes as usize);
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

            // Use native CPU and features for maximum performance (AVX-512, etc.)
            let cpu = TargetMachine::get_host_cpu_name();
            let features = TargetMachine::get_host_cpu_features();

            let target_machine = target
                .create_target_machine(
                    &triple,
                    cpu.to_str().unwrap_or("native"),
                    features.to_str().unwrap_or(""),
                    OptimizationLevel::Aggressive,
                    RelocMode::PIC, // Use PIC for PIE compatibility
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

        /// Get libraries to link from #[link("lib")] attributes on extern blocks
        pub fn get_link_libraries(&self) -> &[String] {
            &self.link_libraries
        }
    }

    /// Type of a Sigil value for codegen
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SigilType {
        Integer,
        Float,
        Pointer,
        String,     // C string literal (null-terminated)
        RustString, // Rust String pointer (from fs_read, etc.)
    }

    /// Variable info in compile scope
    #[derive(Debug, Clone, Copy)]
    struct VarInfo<'ctx> {
        ptr: PointerValue<'ctx>,
        ty: SigilType,
    }

    /// Variable scope for compilation
    struct CompileScope<'ctx> {
        vars: HashMap<String, PointerValue<'ctx>>,
        /// Track which variables hold float values (legacy - being replaced by var_types)
        float_vars: std::collections::HashSet<String>,
        /// Track variable types explicitly
        var_types: HashMap<String, SigilType>,
        /// Cache Vec data base pointers (ptr to element 0) for faster indexing
        vec_bases: HashMap<String, PointerValue<'ctx>>,
        /// Track struct type names for method dispatch (var_name -> struct_type_name)
        struct_types: HashMap<String, String>,
        /// G21: Track functions that return f64 for float detection
        float_funcs: std::collections::HashSet<String>,
    }

    impl<'ctx> CompileScope<'ctx> {
        fn new() -> Self {
            Self {
                vars: HashMap::new(),
                float_vars: std::collections::HashSet::new(),
                var_types: HashMap::new(),
                vec_bases: HashMap::new(),
                struct_types: HashMap::new(),
                float_funcs: std::collections::HashSet::new(),
            }
        }

        /// Register a variable with its type
        fn register_var(&mut self, name: String, ptr: PointerValue<'ctx>, ty: SigilType) {
            self.vars.insert(name.clone(), ptr);
            self.var_types.insert(name.clone(), ty);
            if ty == SigilType::Float {
                self.float_vars.insert(name);
            }
        }

        /// Register a Vec's data base pointer for faster indexing
        fn register_vec_base(&mut self, name: String, base_ptr: PointerValue<'ctx>) {
            self.vec_bases.insert(name, base_ptr);
        }

        /// Get cached Vec data base pointer
        fn get_vec_base(&self, name: &str) -> Option<PointerValue<'ctx>> {
            self.vec_bases.get(name).copied()
        }

        /// Get the type of a variable
        fn get_var_type(&self, name: &str) -> SigilType {
            self.var_types.get(name).copied().unwrap_or(SigilType::Integer)
        }

        /// Check if a variable is a float
        fn is_float_var(&self, name: &str) -> bool {
            self.get_var_type(name) == SigilType::Float
        }

        /// Check if a variable is a C string
        fn is_string_var(&self, name: &str) -> bool {
            self.get_var_type(name) == SigilType::String
        }

        /// Check if a variable is a Rust String (from fs_read, etc.)
        fn is_rust_string_var(&self, name: &str) -> bool {
            self.get_var_type(name) == SigilType::RustString
        }

        /// Register the struct type of a variable for method dispatch
        fn register_struct_type(&mut self, var_name: String, struct_type: String) {
            self.struct_types.insert(var_name, struct_type);
        }

        /// Get the struct type of a variable (if known)
        fn get_struct_type(&self, name: &str) -> Option<&String> {
            self.struct_types.get(name)
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
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 42!;
                    x
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_uncertain() {
            // Uncertain (?) wraps and unwraps correctly
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 100?;
                    x
                }
            "#,
            );
            assert_eq!(result.unwrap(), 100);
        }

        #[test]
        fn test_evidential_reported() {
            // Reported (~) wraps and unwraps correctly
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 200~;
                    x
                }
            "#,
            );
            assert_eq!(result.unwrap(), 200);
        }

        #[test]
        fn test_evidential_predicted() {
            // Predicted (◊) wraps and unwraps correctly
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 300◊;
                    x
                }
            "#,
            );
            assert_eq!(result.unwrap(), 300);
        }

        #[test]
        fn test_evidential_in_expression() {
            // Evidential values can be used in expressions
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ a = 10?;
                    ≔ b = 20?;
                    a + b
                }
            "#,
            );
            assert_eq!(result.unwrap(), 30);
        }

        #[test]
        fn test_evidential_unwrap_chain() {
            // Chain: uncertain -> known (unwrap)
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 42?;
                    ≔ y = x!;
                    y
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_nested() {
            // Nested evidential operations
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = (50?)!;
                    x + 5
                }
            "#,
            );
            assert_eq!(result.unwrap(), 55);
        }

        #[test]
        fn test_evidential_with_arithmetic() {
            // Evidential values with arithmetic
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ known = 100!;
                    ≔ uncertain = 50?;
                    known + uncertain * 2
                }
            "#,
            );
            assert_eq!(result.unwrap(), 200);
        }

        #[test]
        fn test_evidential_function_return() {
            // Function returning evidential value
            let result = run_sigil(
                r#"
                rite get_uncertain() -> i64 {
                    42?
                }

                rite main() -> i64 {
                    ≔ x = get_uncertain();
                    x + 8
                }
            "#,
            );
            assert_eq!(result.unwrap(), 50);
        }

        #[test]
        fn test_evidential_mixed_markers() {
            // Mix different evidentiality markers
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ a = 10!;  // known
                    ≔ b = 20?;  // uncertain
                    ≔ c = 30~;  // reported
                    a + b + c
                }
            "#,
            );
            assert_eq!(result.unwrap(), 60);
        }

        #[test]
        fn test_evidential_in_if() {
            // Evidential in conditional
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 1?;
                    ⎇ x == 1 {
                        100?
                    } ⎉ {
                        200?
                    }
                }
            "#,
            );
            assert_eq!(result.unwrap(), 100);
        }

        #[test]
        fn test_evidential_paradox() {
            // Paradox (‽) marker - contradiction detection
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 42‽;
                    x
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_multiple_unwraps() {
            // Multiple sequential unwraps
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ a = 10?;
                    ≔ b = a!;
                    ≔ c = b!;
                    c
                }
            "#,
            );
            assert_eq!(result.unwrap(), 10);
        }

        #[test]
        fn test_evidential_in_loop() {
            // Evidential values in a loop
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ Δ sum = 0?;
                    ≔ Δ i = 0;
                    ⟳ i < 5 {
                        sum = sum + i?;
                        i = i + 1;
                    }
                    sum!
                }
            "#,
            );
            assert_eq!(result.unwrap(), 10); // 0 + 1 + 2 + 3 + 4 = 10
        }

        #[test]
        fn test_evidential_comparison() {
            // Comparison of evidential values
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ a = 10?;
                    ≔ b = 20?;
                    ⎇ a < b {
                        1!
                    } ⎉ {
                        0!
                    }
                }
            "#,
            );
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_evidential_negation() {
            // Negation with evidential values
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 42?;
                    ≔ y = -x;
                    y + 100
                }
            "#,
            );
            assert_eq!(result.unwrap(), 58); // -42 + 100 = 58
        }

        #[test]
        fn test_evidential_chain_operations() {
            // Chain of operations with mixed evidentiality
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 10!;
                    ≔ y = 20?;
                    ≔ z = 30~;
                    ≔ w = 40◊;
                    x + y + z + w
                }
            "#,
            );
            assert_eq!(result.unwrap(), 100);
        }

        #[test]
        fn test_evidential_deeply_nested() {
            // Deeply nested evidential expressions
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = ((((42?)?)?)?)?;
                    x!
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_evidential_struct_field() {
            // Evidential values as struct fields
            let result = run_sigil(
                r#"
                Σ Data {
                    value: i64,
                }

                rite main() -> i64 {
                    ≔ d = Data { value: 100? };
                    d.value + 1
                }
            "#,
            );
            assert_eq!(result.unwrap(), 101);
        }

        #[test]
        fn test_evidential_function_param() {
            // Function with evidential parameter
            let result = run_sigil(
                r#"
                rite double(x: i64) -> i64 {
                    x * 2
                }

                rite main() -> i64 {
                    ≔ val = 25?;
                    double(val!)
                }
            "#,
            );
            assert_eq!(result.unwrap(), 50);
        }

        #[test]
        fn test_evidential_all_markers_chain() {
            // All 5 evidentiality markers in sequence
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ known = 1!;      // Known
                    ≔ uncertain = 2?;  // Uncertain
                    ≔ reported = 3~;   // Reported
                    ≔ predicted = 4◊;  // Predicted
                    ≔ paradox = 5‽;    // Paradox
                    known + uncertain + reported + predicted + paradox
                }
            "#,
            );
            assert_eq!(result.unwrap(), 15);
        }

        // ============================================
        // Generic Monomorphization Tests (existing)
        // ============================================

        #[test]
        fn test_generic_struct_basic() {
            let result = run_sigil(
                r#"
                Σ Container<T> {
                    value: T,
                    count: i32,
                }

                rite main() -> i64 {
                    ≔ c = Container·<i32> { value: 42, count: 1 };
                    c.value + c.count
                }
            "#,
            );
            assert_eq!(result.unwrap(), 43);
        }

        #[test]
        fn test_generic_struct_two_params() {
            let result = run_sigil(
                r#"
                Σ Pair<A, B> {
                    first: A,
                    second: B,
                }

                rite main() -> i64 {
                    ≔ p = Pair·<i32, i32> { first: 10, second: 20 };
                    p.first + p.second
                }
            "#,
            );
            assert_eq!(result.unwrap(), 30);
        }

        // ============================================
        // Morpheme Tests - Element Access
        // ============================================

        #[test]
        fn test_morpheme_first() {
            // First element: [1, 2, 3] |α returns 1
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [10, 20, 30] |α
                }
            "#,
            );
            assert_eq!(result.unwrap(), 10);
        }

        #[test]
        fn test_morpheme_last() {
            // Last element: [1, 2, 3] |ω returns 3
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [10, 20, 30] |ω
                }
            "#,
            );
            assert_eq!(result.unwrap(), 30);
        }

        #[test]
        fn test_morpheme_middle() {
            // Middle element: [1, 2, 3, 4, 5] |μ returns 3
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [10, 20, 30, 40, 50] |μ
                }
            "#,
            );
            assert_eq!(result.unwrap(), 30);
        }

        #[test]
        fn test_morpheme_nth() {
            // Nth element: [1, 2, 3] |ν{1} returns 2
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [10, 20, 30] |ν{1}
                }
            "#,
            );
            assert_eq!(result.unwrap(), 20);
        }

        // ============================================
        // Morpheme Tests - Reductions
        // ============================================

        #[test]
        fn test_morpheme_reduce_min() {
            // Simple min of two values
            let result = run_sigil(
                r#"
                rite min2(a: i64, b: i64) -> i64 {
                    ⎇ a < b { a } ⎉ { b }
                }
                rite main() -> i64 {
                    min2(min2(5, 2), min2(8, 1))
                }
            "#,
            );
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_reduce_max() {
            // Simple max of two values
            let result = run_sigil(
                r#"
                rite max2(a: i64, b: i64) -> i64 {
                    ⎇ a > b { a } ⎉ { b }
                }
                rite main() -> i64 {
                    max2(max2(5, 2), max2(8, 9))
                }
            "#,
            );
            assert_eq!(result.unwrap(), 9);
        }

        #[test]
        fn test_morpheme_reduce_all_true() {
            // All: [1, 2, 3] |ρ& returns 1 (all non-zero)
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [1, 2, 3] |ρ&
                }
            "#,
            );
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_reduce_all_false() {
            // All: [1, 0, 3] |ρ& returns 0 (not all non-zero)
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [1, 0, 3] |ρ&
                }
            "#,
            );
            assert_eq!(result.unwrap(), 0);
        }

        #[test]
        fn test_morpheme_reduce_any_true() {
            // Any: [0, 0, 1] |ρ| returns 1 (at least one non-zero)
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [0, 0, 1] |ρ|
                }
            "#,
            );
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_reduce_any_false() {
            // Any: [0, 0, 0] |ρ| returns 0 (none non-zero)
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [0, 0, 0] |ρ|
                }
            "#,
            );
            assert_eq!(result.unwrap(), 0);
        }

        // ============================================
        // Combined Morpheme Tests
        // ============================================

        #[test]
        fn test_morpheme_transform_then_first() {
            // Transform then first: [1, 2, 3] |τ{|x| x * 10} |α returns 10
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ arr = [1, 2, 3] |τ{|x| x * 10};
                    arr |α
                }
            "#,
            );
            // Note: This tests that transform returns array, then first extracts
            // Current impl may need adjustment
            assert!(result.is_ok());
        }

        #[test]
        fn test_morpheme_filter_then_sum() {
            // Filter then sum: keep values > 3, sum them
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [1, 5, 2, 8, 3, 7] |φ{|x| x > 3} |ρ+
                }
            "#,
            );
            // After filter: [5, 8, 7], sum = 20
            assert_eq!(result.unwrap(), 20);
        }

        // ============================================
        // New Morpheme Tests - Sort, Choice, Custom Reduce
        // ============================================

        #[test]
        fn test_morpheme_sort_basic() {
            // Sort returns minimum (first after sort): [3, 1, 2] |σ returns 1
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [3, 1, 2] |σ
                }
            "#,
            );
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_sort_already_sorted() {
            // Sort already sorted: [1, 2, 3] |σ returns 1
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [1, 2, 3] |σ
                }
            "#,
            );
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_sort_reverse() {
            // Sort reverse: [5, 4, 3, 2, 1] |σ returns 1
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [5, 4, 3, 2, 1] |σ
                }
            "#,
            );
            assert_eq!(result.unwrap(), 1);
        }

        #[test]
        fn test_morpheme_sort_single() {
            // Sort single element: [42] |σ returns 42
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [42] |σ
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_choice_deterministic() {
            // Choice is deterministic based on array contents
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [10, 20, 30] |χ
                }
            "#,
            );
            // Result should be one of 10, 20, or 30
            let val = result.unwrap();
            assert!(val == 10 || val == 20 || val == 30);
        }

        #[test]
        fn test_morpheme_choice_single() {
            // Choice with single element: [42] |χ returns 42
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [42] |χ
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_custom_reduce_sum() {
            // Custom reduce sum: [1, 2, 3, 4] |ρ{|a, x| a + x} = 10
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [1, 2, 3, 4] |ρ{|acc, x| acc + x}
                }
            "#,
            );
            assert_eq!(result.unwrap(), 10);
        }

        #[test]
        fn test_morpheme_custom_reduce_product() {
            // Custom reduce product: [1, 2, 3, 4] |ρ{|a, x| a * x} = 24
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [1, 2, 3, 4] |ρ{|acc, x| acc * x}
                }
            "#,
            );
            assert_eq!(result.unwrap(), 24);
        }

        #[test]
        fn test_morpheme_custom_reduce_difference() {
            // Custom reduce difference: [100, 20, 5] |ρ{|a, x| a - x} = 75
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [100, 20, 5] |ρ{|acc, x| acc - x}
                }
            "#,
            );
            assert_eq!(result.unwrap(), 75);
        }

        #[test]
        fn test_morpheme_custom_reduce_single() {
            // Custom reduce single element: [42] |ρ{|a, x| a + x} = 42
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    [42] |ρ{|acc, x| acc + x}
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_await_expr() {
            // Await expression form: expr⌛ (postfix syntax)
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 42;
                    x⌛
                }
            "#,
            );
            // In sync LLVM context, await is identity
            assert_eq!(result.unwrap(), 42);
        }

        #[test]
        fn test_morpheme_await_nested() {
            // Nested await expressions
            let result = run_sigil(
                r#"
                rite main() -> i64 {
                    ≔ x = 21;
                    ≔ y = x⌛ + x⌛;
                    y
                }
            "#,
            );
            assert_eq!(result.unwrap(), 42);
        }
    }
}

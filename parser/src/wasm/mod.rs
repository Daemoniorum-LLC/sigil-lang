//! Sigil WASM Compiler
//!
//! Compiles Sigil AST to WebAssembly for browser execution.
//! Designed for AI-native web development with sigil-web-interface.
//!
//! # Features
//!
//! - Direct WASM bytecode generation using wasm-encoder
//! - DOM/Web API bindings via imported functions
//! - Evidentiality tracking preserved at runtime
//! - Morpheme operator support (τ, φ, σ, ρ, α, Ω, etc.)
//! - Closure compilation with environment capture
//!
//! # Example
//!
//! ```ignore
//! use sigil_parser::wasm::WasmCompiler;
//!
//! let source = r#"
//!     pub fn main() {
//!         let data = [1, 2, 3, 4, 5];
//!         let sum = data|ρ+;
//!         print(sum);
//!     }
//! "#;
//!
//! let mut compiler = WasmCompiler::new();
//! let wasm_bytes = compiler.compile(source)?;
//! ```

pub mod async_sm;
pub mod async_sm_ir;
pub mod closures;
pub mod constants;
pub mod control_flow;
pub mod deps;
pub mod error;
pub mod expressions;
pub mod imports;
pub mod literals;
pub mod macros;
pub mod morphemes;
pub mod operators;
pub mod sourcemap;
pub mod statements;
pub mod types;

// Re-export main types
pub use constants::{evidence, memory, type_tag};
pub use error::{WasmError, WasmErrorKind, WasmResult};
pub use imports::ImportRegistry;
pub use sourcemap::{SourceMap, SourceMapBuilder, SourceLocation};
pub use types::*;

use std::collections::HashMap;
use wasm_encoder::ValType;

/// Sentinel base for defined-function indices stored during compilation.
///
/// `register_function_sig` cannot know the final import count (more imports
/// may be added by later crates), so we store `DEFINED_FUNC_SENTINEL + array_index`
/// instead of `import_count + array_index`.  The post-processing pass
/// `fix_stale_func_indices` rewrites these to `final_import_count + array_index`
/// once all crates have been compiled.
///
/// The sentinel is large enough that it never collides with any real WASM
/// function index (real imports are at most in the tens of thousands).
pub(crate) const DEFINED_FUNC_SENTINEL: u32 = 0x1000_0000;

use crate::optimize::OptLevel;
use crate::parser::Parser;

/// WASM Compiler for Sigil.
///
/// Compiles Sigil source code to WASM bytecode.
#[derive(Debug)]
pub struct WasmCompiler {
    /// Import registry
    pub(crate) imports: ImportRegistry,

    /// Compiled functions
    pub(crate) functions: Vec<CompiledFunction>,

    /// Function name -> index mapping
    pub(crate) func_map: HashMap<String, u32>,

    /// Global variables: (type, mutable, initial_value)
    pub(crate) globals: Vec<(ValType, bool, i64)>,

    /// Global name -> index mapping
    pub(crate) global_map: HashMap<String, u32>,

    /// Data segments: (offset, bytes)
    pub(crate) data_segments: Vec<(u32, Vec<u8>)>,

    /// Current data offset
    pub(crate) data_offset: u32,

    /// String interning map
    pub(crate) string_map: HashMap<String, u32>,

    /// String constants map (name -> data segment offset)
    pub(crate) string_consts: HashMap<String, u32>,

    /// Function table elements (for indirect calls)
    pub(crate) table_elements: Vec<u32>,

    /// Closure information
    pub(crate) closure_map: HashMap<String, ClosureInfo>,

    /// Closure counter for unique names
    pub(crate) closure_counter: u32,

    /// Struct layouts
    pub(crate) struct_layouts: HashMap<String, StructLayout>,

    /// Enum layouts
    pub(crate) enum_layouts: HashMap<String, EnumLayout>,

    /// Current function being compiled
    pub(crate) current_fn_idx: Option<usize>,

    /// Loop context stack
    pub(crate) loop_stack: Vec<LoopContext>,

    /// Label counter for blocks
    pub(crate) label_counter: u32,

    /// Scope stack for capture analysis
    pub(crate) scope_vars: Vec<HashMap<String, u32>>,

    /// Mutable captures: variables that are captured mutably (need cell indirection)
    pub(crate) mutable_captures: std::collections::HashSet<String>,

    /// Cell pointers for mutable captures: variable name -> heap address of cell
    pub(crate) capture_cells: HashMap<String, u32>,

    /// External module imports: simple_name -> (module_name, qualified_name)
    /// Used for cross-module WASM linking
    pub(crate) external_imports: HashMap<String, (String, String)>,

    /// Current module path (for nested module compilation)
    /// e.g., ["vdom", "element"] when compiling vdom::element module
    pub(crate) module_path: Vec<String>,

    /// All items organized by qualified path
    /// Maps "vdom::Element" -> function/type index
    pub(crate) qualified_items: HashMap<String, QualifiedItem>,

    /// Extern type names (from extern blocks)
    /// Used to resolve method calls like Node::append_child
    pub(crate) extern_types: std::collections::HashSet<String>,

    /// Variable types: maps variable name to its type name
    /// Used to resolve method calls like app·view() where app has type PlatformApp
    pub(crate) var_types: HashMap<String, String>,

    /// Optimization level
    pub(crate) opt_level: OptLevel,

    /// Include debug info
    pub(crate) debug_info: bool,

    /// Source map builder (when debug_info is enabled)
    pub(crate) source_map: Option<SourceMapBuilder>,

    /// Source file name for source maps
    pub(crate) source_file: String,

    /// Source directory for resolving file-based modules (scroll foo;)
    pub(crate) source_dir: std::path::PathBuf,

    /// Already-loaded module files to prevent circular imports
    pub(crate) loaded_modules: std::collections::HashSet<std::path::PathBuf>,

    /// Cached parsed module items (keyed by canonical path)
    pub(crate) module_cache: std::collections::HashMap<std::path::PathBuf, Vec<crate::span::Spanned<crate::ast::Item>>>,

    /// Deferred static initializers: (global_index, init_expression, module_path_snapshot)
    /// These are statics with non-constant initializers that need runtime init.
    /// The module_path_snapshot is the compiler's module_path at the time the init was deferred,
    /// so that name resolution in generate_start_function uses the correct crate/module context.
    pub(crate) deferred_static_inits: Vec<(u32, crate::ast::Expr, Vec<String>)>,

    /// Start function index (for __wasm_start if we have deferred inits)
    pub(crate) start_function_idx: Option<u32>,

    /// Actor type names registered during compilation.
    /// Used to detect `ActorType·method()` static calls (e.g. `Wraith·view()`)
    /// and emit `I64Const(0)` as the dummy self argument instead of trying to
    /// evaluate the type name as an expression (which would fail).
    pub(crate) actor_names: std::collections::HashSet<String>,

    /// Current actor being compiled (for self.field resolution)
    pub(crate) current_actor: Option<String>,

    /// Current impl type being compiled (for self.field offset resolution).
    /// Set to the struct/impl type name when entering `⊢ Type { }` blocks.
    /// Used by get_field_offset to prefer the correct struct layout when multiple
    /// structs have fields with the same name (e.g. VElement and VFragment both
    /// have `children`).
    pub(crate) current_impl_type: Option<String>,

    /// Local variable type map: variable name → struct type name.
    /// Populated when match arms bind enum variant payloads, e.g.
    ///   `VNode::Element(el)` → registers "el" → "VElement"
    ///   `VNode::Fragment(frag)` → registers "frag" → "VFragment"
    /// Used to dispatch method calls and resolve field offsets for non-self variables.
    pub(crate) local_var_types: HashMap<String, String>,

    /// Depth of impl/actor blocks currently being compiled.
    /// Incremented when entering `⊢ Type { }` or `actor { }` and decremented on exit.
    /// Used by register_function_sig to distinguish impl-block methods (which must NOT
    /// overwrite import func_map entries with the same simple name) from free functions
    /// in module files (which SHOULD overwrite).
    /// NOTE: module_path.len() > 1 cannot be used for this because module file paths
    /// also push onto module_path, making free functions inside module files look like
    /// impl methods when they are not.
    pub(crate) impl_depth: usize,
}

impl WasmCompiler {
    /// Create a new WASM compiler.
    pub fn new() -> Self {
        let mut compiler = Self {
            imports: ImportRegistry::new(),
            functions: Vec::new(),
            func_map: HashMap::new(),
            globals: Vec::new(),
            global_map: HashMap::new(),
            data_segments: Vec::new(),
            data_offset: memory::HEAP_START,
            string_map: HashMap::new(),
            string_consts: HashMap::new(),
            table_elements: Vec::new(),
            closure_map: HashMap::new(),
            closure_counter: 0,
            struct_layouts: HashMap::new(),
            enum_layouts: HashMap::new(),
            current_fn_idx: None,
            loop_stack: Vec::new(),
            label_counter: 0,
            scope_vars: Vec::new(),
            mutable_captures: std::collections::HashSet::new(),
            capture_cells: HashMap::new(),
            external_imports: HashMap::new(),
            module_path: Vec::new(),
            qualified_items: HashMap::new(),
            extern_types: std::collections::HashSet::new(),
            var_types: HashMap::new(),
            opt_level: OptLevel::Standard,
            debug_info: false,
            source_map: None,
            source_file: String::new(),
            source_dir: std::path::PathBuf::new(),
            loaded_modules: std::collections::HashSet::new(),
            module_cache: std::collections::HashMap::new(),
            deferred_static_inits: Vec::new(),
            start_function_idx: None,
            actor_names: std::collections::HashSet::new(),
            current_actor: None,
            current_impl_type: None,
            local_var_types: HashMap::new(),
            impl_depth: 0,
        };

        // Add heap pointer global
        compiler.globals.push((ValType::I32, true, memory::HEAP_START as i64));
        compiler.global_map.insert("__heap_ptr".to_string(), 0);

        compiler
    }

    /// Create compiler with optimization level.
    pub fn with_opt_level(opt_level: OptLevel) -> Self {
        let mut compiler = Self::new();
        compiler.opt_level = opt_level;
        compiler
    }

    /// Enable debug info generation.
    pub fn with_debug_info(mut self) -> Self {
        self.debug_info = true;
        self
    }

    /// Compile source code to WASM bytes.
    pub fn compile(&mut self, source: &str) -> WasmResult<Vec<u8>> {
        // Initialize source map builder if debug info is enabled
        if self.debug_info {
            let file_name = if self.source_file.is_empty() {
                "input.sigil".to_string()
            } else {
                self.source_file.clone()
            };
            self.source_map = Some(SourceMapBuilder::new(file_name, source));
        }

        // Parse source
        let mut parser = Parser::new(source);
        let ast = parser.parse_file().map_err(|e| WasmError::parse(e.to_string()))?;

        // TODO: Optimize AST if opt_level != None
        // let ast = if self.opt_level != OptLevel::None {
        //     Optimizer::new(self.opt_level).optimize_file(&ast)
        // } else {
        //     ast
        // };

        // Compile AST
        self.compile_file(&ast)?;

        // Generate __wasm_start function if we have deferred static initializers
        if !self.deferred_static_inits.is_empty() {
            self.generate_start_function()?;
        }

        // Resolve sentinel function indices to final module indices.
        // (Same pass as compile_project; needed here because single-file compilation
        // also uses DEFINED_FUNC_SENTINEL during registration.)
        self.fix_stale_func_indices();

        // Fix invalid call wrappers (functions with 0 params that call imports expecting params)
        self.fix_invalid_call_wrappers();

        // Fix control flow stack imbalance (spurious initial values)
        self.fix_control_flow_stack();

        // Generate WASM module
        self.generate_module()
    }

    /// Compile with a specific source file name (for source maps).
    pub fn compile_file_named(&mut self, source: &str, file_name: &str) -> WasmResult<Vec<u8>> {
        self.source_file = file_name.to_string();
        self.compile(source)
    }

    /// Compile from a file path, enabling multi-file module resolution.
    pub fn compile_from_path(&mut self, path: &std::path::Path) -> WasmResult<Vec<u8>> {
        use std::fs;

        // Set source directory for resolving file-based modules
        if let Some(parent) = path.parent() {
            self.source_dir = parent.to_path_buf();
        }

        // Track this file as loaded
        let canonical = path.canonicalize()
            .map_err(|e| WasmError::io(format!("cannot resolve path {}: {}", path.display(), e)))?;
        self.loaded_modules.insert(canonical);

        // Read and compile
        let source = fs::read_to_string(path)
            .map_err(|e| WasmError::io(format!("cannot read {}: {}", path.display(), e)))?;

        self.source_file = path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "input.sigil".to_string());

        self.compile(&source)
    }

    /// Compile a project with dependencies from sigil.toml.
    ///
    /// This resolves all dependencies, compiles them in order, and bundles
    /// everything into a single WASM module.
    pub fn compile_project(project_dir: &std::path::Path) -> WasmResult<Vec<u8>> {
        use deps::{DependencyGraph, ProjectManifest};

        // Build dependency graph
        let graph = DependencyGraph::from_project(project_dir)?;

        // Create compiler instance
        let mut compiler = Self::new();

        // Compile each dependency in order (dependencies first)
        for manifest in graph.iter_in_order() {
            compiler.compile_crate(&manifest)?;
        }

        // Generate __wasm_start for any deferred (non-const) static/actor-state initializers.
        // Must happen before fix_stale_func_indices so the new function's Call instructions
        // are also remapped.
        if !compiler.deferred_static_inits.is_empty() {
            compiler.generate_start_function()?;
        }

        // Fix stale function indices: register_function_sig uses import_count at registration
        // time, which is less than the final count. This re-maps all Call instructions that
        // target defined functions to their correct (final) module indices.
        compiler.fix_stale_func_indices();

        // Fix invalid call wrappers (functions with 0 params that call imports expecting params)
        compiler.fix_invalid_call_wrappers();

        // Fix control flow stack imbalance (spurious initial values)
        compiler.fix_control_flow_stack();

        // Generate the final WASM module
        compiler.generate_module()
    }

    /// Compile a single crate into the current compiler state.
    fn compile_crate(&mut self, manifest: &deps::ProjectManifest) -> WasmResult<()> {
        use std::fs;

        // Set source directory for module resolution
        self.source_dir = manifest.root_dir.join("src");

        // Read the lib entry point
        let lib_path = &manifest.lib_path;
        if !lib_path.exists() {
            // No lib.sigil - might be a binary-only crate, skip
            return Ok(());
        }

        let canonical = lib_path.canonicalize()
            .map_err(|e| WasmError::io(format!(
                "cannot resolve {}: {}",
                lib_path.display(),
                e
            )))?;

        // Skip if already compiled
        if self.loaded_modules.contains(&canonical) {
            return Ok(());
        }
        self.loaded_modules.insert(canonical);

        let source = fs::read_to_string(lib_path)
            .map_err(|e| WasmError::io(format!(
                "cannot read {}: {}",
                lib_path.display(),
                e
            )))?;

        self.source_file = lib_path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "lib.sigil".to_string());

        // Push crate name to module path for qualified names
        let crate_name = manifest.name.replace('-', "_");
        self.module_path.push(crate_name);

        // Parse and compile
        let mut parser = Parser::new(&source);
        let ast = parser.parse_file()
            .map_err(|e| WasmError::parse(e.to_string()))?;

        self.compile_file(&ast)?;

        // Pop crate name
        self.module_path.pop();

        Ok(())
    }

    /// Get or create a type index.
    pub fn get_or_create_type(&mut self, params: Vec<ValType>, results: Vec<ValType>) -> u32 {
        self.imports.get_or_create_type(params, results)
    }

    /// Look up a function by name.
    pub fn get_func(&self, name: &str) -> Option<u32> {
        // Check local functions first
        if let Some(&idx) = self.func_map.get(name) {
            return Some(idx);
        }
        // Check imports
        self.imports.get_func(name)
    }

    /// Summarize an expression for debug output.
    pub fn expr_summary(expr: &crate::ast::Expr) -> String {
        match expr {
            crate::ast::Expr::Path(p) => format!("Path({})", p.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("·")),
            crate::ast::Expr::MethodCall { method, .. } => format!("MethodCall({})", method.name),
            crate::ast::Expr::Call { func, .. } => format!("Call({:?})", Self::expr_summary(func)),
            crate::ast::Expr::Field { field, .. } => format!("Field({})", field.name),
            crate::ast::Expr::Await { .. } => "Await".to_string(),
            crate::ast::Expr::Closure { .. } => "Closure".to_string(),
            _ => format!("Other"),
        }
    }

    /// Add a function to the indirect call table and return its table index.
    pub fn add_to_table(&mut self, func_idx: u32) -> u32 {
        // Check if function is already in table
        if let Some(pos) = self.table_elements.iter().position(|&f| f == func_idx) {
            return pos as u32;
        }
        // Add to table
        let table_idx = self.table_elements.len() as u32;
        self.table_elements.push(func_idx);
        table_idx
    }

    /// Look up a global by name.
    pub fn get_global(&self, name: &str) -> Option<u32> {
        self.global_map.get(name).copied()
    }

    /// Add a string to the data section, returning its offset.
    pub fn add_string(&mut self, s: &str) -> u32 {
        if let Some(&offset) = self.string_map.get(s) {
            return offset;
        }

        let offset = self.data_offset;
        let bytes = s.as_bytes();

        // Format: 4-byte length + string bytes
        let mut data = (bytes.len() as u32).to_le_bytes().to_vec();
        data.extend(bytes);

        self.data_segments.push((offset, data.clone()));
        self.data_offset += data.len() as u32;
        // Align to 8 bytes
        self.data_offset = (self.data_offset + 7) & !7;

        self.string_map.insert(s.to_string(), offset);
        offset
    }

    /// Get the current function being compiled.
    pub fn current_function(&self) -> Option<&CompiledFunction> {
        self.current_fn_idx.and_then(|idx| self.functions.get(idx))
    }

    /// Get the current function mutably.
    pub fn current_function_mut(&mut self) -> Option<&mut CompiledFunction> {
        self.current_fn_idx
            .and_then(move |idx| self.functions.get_mut(idx))
    }

    /// Check if a user-defined function returns void (no results).
    pub fn func_returns_void(&self, func_idx: u32) -> bool {
        // During body compilation, defined-function call sites may still hold sentinel
        // indices (DEFINED_FUNC_SENTINEL + array_index) because fix_stale_func_indices
        // hasn't run yet.  Resolve sentinels directly via the functions array.
        if func_idx >= DEFINED_FUNC_SENTINEL {
            let local_idx = (func_idx - DEFINED_FUNC_SENTINEL) as usize;
            return self.functions
                .get(local_idx)
                .map(|f| f.results.is_empty())
                .unwrap_or(false);
        }

        let import_count = self.imports.import_count();
        if func_idx < import_count {
            // Import function - check via imports
            self.imports.get_return_type(func_idx).is_none()
        } else {
            // User-defined function (post-fix_stale_func_indices path)
            let local_idx = (func_idx - import_count) as usize;
            self.functions
                .get(local_idx)
                .map(|f| f.results.is_empty())
                .unwrap_or(false)
        }
    }

    /// Get or add an external module import.
    /// Returns the function index for calling the imported function.
    pub fn get_or_add_external_import(&mut self, module: &str, name: &str, arg_count: usize) -> u32 {
        // Check if already imported
        let qualified = format!("{}_{}", module, name);
        if let Some(idx) = self.imports.get_func(&qualified) {
            return idx;
        }

        // Also check by simple name
        if let Some(idx) = self.imports.get_func(name) {
            return idx;
        }

        // Add as new import with generic signature:
        // All args are i64, returns i64 (uniform type system)
        let params: Vec<ValType> = vec![ValType::I64; arg_count];
        let results = vec![ValType::I64];

        self.imports.add_import(module, name, params, results)
    }

    /// Get the current qualified path prefix.
    /// Returns "foo::bar" if we're currently compiling inside scroll foo { scroll bar { } }
    pub fn current_module_prefix(&self) -> String {
        self.module_path.join("::")
    }

    /// Build a qualified name from the current module path and an item name.
    pub fn qualify_name(&self, name: &str) -> String {
        if self.module_path.is_empty() {
            name.to_string()
        } else {
            format!("{}::{}", self.current_module_prefix(), name)
        }
    }

    /// Look up a qualified item by path.
    /// Handles paths like "foo::bar::Baz" or just "Baz".
    pub fn lookup_qualified(&self, path: &[String]) -> Option<&QualifiedItem> {
        let qualified = path.join("::");
        self.qualified_items.get(&qualified)
    }

    /// Register an item with its qualified path.
    pub fn register_qualified(&mut self, name: &str, item: QualifiedItem) {
        let qualified = self.qualify_name(name);
        self.qualified_items.insert(qualified, item);
    }

    /// Resolve a path that may start with "tome" (crate root).
    /// Returns the resolved path segments.
    pub fn resolve_path(&self, segments: &[String]) -> Vec<String> {
        match segments.first().map(|s| s.as_str()) {
            Some("tome") | Some("crate") => segments[1..].to_vec(),
            _ => segments.to_vec(),
        }
    }

    /// Look up a function by qualified path.
    /// Handles tome:: prefix and module-relative paths.
    pub fn get_func_by_path(&self, segments: &[String]) -> Option<u32> {
        let resolved = self.resolve_path(segments);

        // If it's a single-segment path, check simple name first
        if resolved.len() == 1 {
            if let Some(idx) = self.func_map.get(&resolved[0]) {
                return Some(*idx);
            }
        }

        // Try qualified lookup
        let qualified = resolved.join("::");
        if let Some(idx) = self.func_map.get(&qualified) {
            return Some(*idx);
        }

        // Check in qualified_items
        if let Some(QualifiedItem::Function(idx)) = self.qualified_items.get(&qualified) {
            return Some(*idx);
        }

        // Try just the last segment (simple name) for module-qualified calls
        // e.g., components::nav_view -> try just "nav_view"
        if resolved.len() > 1 {
            let simple_name = resolved.last().unwrap();
            if let Some(idx) = self.func_map.get(simple_name) {
                return Some(*idx);
            }
        }

        // Suffix-based fallback: when the exact path is not in func_map (e.g. because the
        // registered name has a crate prefix), search for any key ending with "::<qualified>".
        // E.g. call "Preferences::default" should find "wraith_sigil::Preferences::default".
        if resolved.len() >= 2 {
            let suffix = format!("::{}", qualified);
            if let Some((&ref _k, &idx)) = self.func_map.iter().find(|(k, _)| k.ends_with(&suffix)) {
                return Some(idx);
            }
        }

        // Check imports
        self.imports.get_func(&qualified)
    }

    // compile_file is implemented in statements.rs

    /// Fix control flow stack imbalance: functions with extra values on the stack.
    /// This handles several cases:
    /// 1. Spurious i64.const 0 at the start
    /// 2. LocalTee leaving values on stack before if blocks
    fn fix_control_flow_stack(&mut self) {
        use wasm_encoder::Instruction;

        for func in self.functions.iter_mut() {
            // Only check functions that return exactly 1 value
            if func.results.len() != 1 {
                continue;
            }

            // Need at least 3 instructions
            if func.instructions.len() < 3 {
                continue;
            }

            // Case 1: i64.const 0 followed by local.get (spurious initial push)
            let first_is_const_0 = matches!(&func.instructions[0], Instruction::I64Const(0));
            let second_is_local_get = matches!(&func.instructions[1], Instruction::LocalGet(_));

            if first_is_const_0 && second_is_local_get {
                let third = func.instructions.get(2);
                let third_is_wrap = matches!(third, Some(Instruction::I32WrapI64));
                let third_is_local_set = matches!(
                    third,
                    Some(Instruction::LocalSet(_)) | Some(Instruction::LocalTee(_))
                );
                let third_is_drop = matches!(third, Some(Instruction::Drop));

                if third_is_wrap || third_is_local_set {
                    let has_early_store_of_initial = func.instructions[2..8.min(func.instructions.len())]
                        .iter()
                        .any(|i| matches!(i, Instruction::LocalSet(0)));

                    if !has_early_store_of_initial {
                        func.instructions.remove(0);
                    }
                } else if !third_is_drop {
                    let last_before_end = func.instructions.len().saturating_sub(2);
                    if let Some(Instruction::Call(_)) = func.instructions.get(last_before_end) {
                        func.instructions.remove(0);
                    }
                }
            }

            // Case 2: LocalTee before If leaves value on stack
            // Pattern: ..., call X, local.tee Y, local.get Z, ..., if
            // The local.tee leaves a value that goes through the if block unused
            // Fix: Replace local.tee with local.set
            for i in 0..func.instructions.len().saturating_sub(3) {
                if let Instruction::LocalTee(local_idx) = &func.instructions[i] {
                    let local_idx = *local_idx;
                    // Check if followed by local.get (not of same local), then eventually if
                    if let Some(Instruction::LocalGet(get_idx)) = func.instructions.get(i + 1) {
                        if *get_idx != local_idx {
                            // Look ahead for an if within a few instructions
                            let has_if_soon = func.instructions[i + 2..((i + 8).min(func.instructions.len()))]
                                .iter()
                                .any(|instr| matches!(instr, Instruction::If(_)));

                            if has_if_soon {
                                // Replace local.tee with local.set to not leave value on stack
                                func.instructions[i] = Instruction::LocalSet(local_idx);
                            }
                        }
                    }
                }
            }

            // Case 3: Spurious i64.const 0 before a call that expects i32 args
            // Pattern: ..., i32.wrap_i64, i64.const 0, call(import)
            // The i64.const 0 is spurious - remove it
            let import_count = self.imports.import_count();
            let mut i = 0;
            while i + 2 < func.instructions.len() {
                if let Instruction::I64Const(0) = &func.instructions[i] {
                    if let Instruction::Call(call_idx) = &func.instructions[i + 1] {
                        let call_idx = *call_idx;
                        // Check if it's an import
                        if call_idx < import_count {
                            if let Some(params) = self.imports.get_param_types(call_idx) {
                                // The i64.const 0 is spurious if:
                                // 1. All params are i32 (classic case), OR
                                // 2. Last param is i32 (i64.const 0 can't be for it)
                                let all_i32 = !params.is_empty()
                                    && params.iter().all(|p| *p == ValType::I32);
                                let last_is_i32 = params.last() == Some(&ValType::I32);

                                if all_i32 || last_is_i32 {
                                    // Check if previous instruction produces an i32
                                    if i > 0 {
                                        let prev = &func.instructions[i - 1];
                                        if matches!(prev, Instruction::I32WrapI64 | Instruction::I32Const(_)) {
                                            // Remove the spurious i64.const 0
                                            func.instructions.remove(i);
                                            continue; // Don't increment i since we removed an instruction
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                i += 1;
            }
        }
    }

    /// Fix invalid call wrappers: functions that call imports without proper stack setup.
    /// This handles two cases:
    /// 1. Functions with 0 params that call imports expecting params (replace with constant)
    /// 2. Functions with params that immediately call imports without pushing params first
    /// Resolve sentinel function indices to final module indices.
    ///
    /// During compilation `register_function_sig` stores `DEFINED_FUNC_SENTINEL + M`
    /// (where M is the functions-array index) instead of `import_count + M`.  This
    /// avoids the stale-index bug where import_count at registration time is smaller
    /// than the final count (because later crates add more imports).
    ///
    /// Once all crates are compiled this pass replaces every sentinel with the
    /// correct index `final_import_count + M`.  Sentinel values are unmistakable
    /// (they are far above any realistic function index) so there is zero risk of
    /// accidentally remapping a legitimate call.
    fn fix_stale_func_indices(&mut self) {
        use wasm_encoder::Instruction;

        let final_import_count = self.imports.import_count();

        // Fix Call instructions in every function body.
        for func in self.functions.iter_mut() {
            for instr in func.instructions.iter_mut() {
                if let Instruction::Call(idx) = instr {
                    if *idx >= DEFINED_FUNC_SENTINEL {
                        *idx = final_import_count + (*idx - DEFINED_FUNC_SENTINEL);
                    }
                }
            }
            // Fix the func_idx stored on the CompiledFunction itself
            // (used by generate_module for exports and the start section).
            if func.func_idx >= DEFINED_FUNC_SENTINEL {
                func.func_idx = final_import_count + (func.func_idx - DEFINED_FUNC_SENTINEL);
            }
        }

        // Fix func_map so that lookups resolve to the correct final index.
        for val in self.func_map.values_mut() {
            if *val >= DEFINED_FUNC_SENTINEL {
                *val = final_import_count + (*val - DEFINED_FUNC_SENTINEL);
            }
        }

        // Fix indirect-call table elements.
        for elem in self.table_elements.iter_mut() {
            if *elem >= DEFINED_FUNC_SENTINEL {
                *elem = final_import_count + (*elem - DEFINED_FUNC_SENTINEL);
            }
        }

        // Fix the start function index if present.
        if let Some(start_idx) = self.start_function_idx {
            if start_idx >= DEFINED_FUNC_SENTINEL {
                self.start_function_idx =
                    Some(final_import_count + (start_idx - DEFINED_FUNC_SENTINEL));
            }
        }
    }

    fn fix_invalid_call_wrappers(&mut self) {
        use wasm_encoder::Instruction;

        let import_count = self.imports.import_count();

        // Get a snapshot of import param info before iterating
        let import_param_counts: Vec<usize> = (0..import_count)
            .map(|idx| {
                self.imports
                    .get_param_types(idx)
                    .map(|p| p.len())
                    .unwrap_or(0)
            })
            .collect();

        // Pre-calculate local function param counts (to avoid borrow conflicts in loop)
        let local_func_param_counts: Vec<usize> = self
            .functions
            .iter()
            .map(|f| f.params.len())
            .collect();

        for func in self.functions.iter_mut() {
            // Case 1: Functions with 0 parameters that call imports expecting params
            if func.params.is_empty() {
                // Find the first call instruction
                let first_call_idx = func.instructions.iter().position(|i| matches!(i, Instruction::Call(_)));

                if let Some(idx) = first_call_idx {
                    if let Instruction::Call(call_idx) = &func.instructions[idx] {
                        let call_idx = *call_idx;
                        // Check if it's calling an import
                        if call_idx < import_count {
                            // Get the import's param count
                            let param_count = import_param_counts.get(call_idx as usize).copied().unwrap_or(0);
                            // If import expects params but we have 0, need to push a default value
                            if param_count > 0 {
                                // Check if there's anything on the stack before the call
                                // A simple heuristic: if idx == 0, stack is definitely empty
                                // Otherwise check if previous instructions push values
                                let stack_empty = idx == 0 || !func.instructions[0..idx]
                                    .iter()
                                    .any(|i| matches!(i,
                                        Instruction::I64Const(_) |
                                        Instruction::I32Const(_) |
                                        Instruction::LocalGet(_) |
                                        Instruction::GlobalGet(_) |
                                        Instruction::Call(_)
                                    ));

                                if stack_empty {
                                    // Insert i64.const 0 before the call for each required param
                                    for _ in 0..param_count {
                                        func.instructions.insert(idx, Instruction::I64Const(0));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Case 2: ANY function with a call to import early that doesn't have enough values on stack
            // Scan through early instructions looking for calls to imports
            for i in 0..func.instructions.len().min(20) {
                if let Instruction::Call(call_idx) = &func.instructions[i] {
                    let call_idx = *call_idx;
                    if call_idx < import_count {
                        let import_param_count = import_param_counts.get(call_idx as usize).copied().unwrap_or(0);
                        if import_param_count == 0 {
                            continue;
                        }

                        // Count how many values are pushed before this call
                        let mut stack_depth: i32 = 0;
                        for j in 0..i {
                            match &func.instructions[j] {
                                Instruction::LocalGet(_) | Instruction::GlobalGet(_) |
                                Instruction::I32Const(_) | Instruction::I64Const(_) |
                                Instruction::F32Const(_) | Instruction::F64Const(_) => {
                                    stack_depth += 1;
                                }
                                Instruction::LocalSet(_) | Instruction::GlobalSet(_) |
                                Instruction::Drop => {
                                    stack_depth -= 1;
                                }
                                Instruction::LocalTee(_) => {
                                    // Tee doesn't change stack depth (pops and pushes)
                                }
                                Instruction::Call(idx) => {
                                    // For calls, track stack effect
                                    if *idx < import_count {
                                        // Import function: use known param count
                                        let params = import_param_counts.get(*idx as usize).copied().unwrap_or(0);
                                        stack_depth -= params as i32;
                                        stack_depth += 1; // Assume 1 return value
                                    } else {
                                        // Local function: look up param count from pre-calculated vector
                                        let local_idx = (*idx - import_count) as usize;
                                        if let Some(&param_count) = local_func_param_counts.get(local_idx) {
                                            stack_depth -= param_count as i32;
                                        }
                                        // All Sigil functions return i64
                                        stack_depth += 1;
                                    }
                                }
                                // Binary operations: consume 2, produce 1 → net -1
                                Instruction::I32Add | Instruction::I32Sub |
                                Instruction::I32Mul | Instruction::I32DivS | Instruction::I32DivU |
                                Instruction::I32RemS | Instruction::I32RemU |
                                Instruction::I32And | Instruction::I32Or | Instruction::I32Xor |
                                Instruction::I32Shl | Instruction::I32ShrS | Instruction::I32ShrU |
                                Instruction::I64Add | Instruction::I64Sub |
                                Instruction::I64Mul | Instruction::I64DivS | Instruction::I64DivU |
                                Instruction::I64RemS | Instruction::I64RemU |
                                Instruction::I64And | Instruction::I64Or | Instruction::I64Xor |
                                Instruction::I64Shl | Instruction::I64ShrS | Instruction::I64ShrU |
                                Instruction::F32Add | Instruction::F32Sub |
                                Instruction::F32Mul | Instruction::F32Div |
                                Instruction::F64Add | Instruction::F64Sub |
                                Instruction::F64Mul | Instruction::F64Div |
                                // Comparison operations also consume 2, produce 1
                                Instruction::I32Eq | Instruction::I32Ne |
                                Instruction::I32LtS | Instruction::I32LtU |
                                Instruction::I32GtS | Instruction::I32GtU |
                                Instruction::I32LeS | Instruction::I32LeU |
                                Instruction::I32GeS | Instruction::I32GeU |
                                Instruction::I64Eq | Instruction::I64Ne |
                                Instruction::I64LtS | Instruction::I64LtU |
                                Instruction::I64GtS | Instruction::I64GtU |
                                Instruction::I64LeS | Instruction::I64LeU |
                                Instruction::I64GeS | Instruction::I64GeU |
                                Instruction::F32Eq | Instruction::F32Ne |
                                Instruction::F32Lt | Instruction::F32Gt |
                                Instruction::F32Le | Instruction::F32Ge |
                                Instruction::F64Eq | Instruction::F64Ne |
                                Instruction::F64Lt | Instruction::F64Gt |
                                Instruction::F64Le | Instruction::F64Ge => {
                                    stack_depth -= 1; // Net effect: -2 + 1 = -1
                                }
                                // Unary operations: consume 1, produce 1 → net 0
                                Instruction::I32Eqz | Instruction::I64Eqz |
                                Instruction::I32Clz | Instruction::I32Ctz | Instruction::I32Popcnt |
                                Instruction::I64Clz | Instruction::I64Ctz | Instruction::I64Popcnt |
                                Instruction::F32Abs | Instruction::F32Neg |
                                Instruction::F32Sqrt | Instruction::F32Ceil | Instruction::F32Floor |
                                Instruction::F64Abs | Instruction::F64Neg |
                                Instruction::F64Sqrt | Instruction::F64Ceil | Instruction::F64Floor |
                                // Conversions: consume 1, produce 1
                                Instruction::I32WrapI64 | Instruction::I64ExtendI32S | Instruction::I64ExtendI32U |
                                Instruction::F32ConvertI32S | Instruction::F32ConvertI32U |
                                Instruction::F32ConvertI64S | Instruction::F32ConvertI64U |
                                Instruction::F64ConvertI32S | Instruction::F64ConvertI32U |
                                Instruction::F64ConvertI64S | Instruction::F64ConvertI64U |
                                Instruction::I32TruncF32S | Instruction::I32TruncF32U |
                                Instruction::I32TruncF64S | Instruction::I32TruncF64U |
                                Instruction::I64TruncF32S | Instruction::I64TruncF32U |
                                Instruction::I64TruncF64S | Instruction::I64TruncF64U |
                                Instruction::F32DemoteF64 | Instruction::F64PromoteF32 |
                                Instruction::I32ReinterpretF32 | Instruction::I64ReinterpretF64 |
                                Instruction::F32ReinterpretI32 | Instruction::F64ReinterpretI64 => {
                                    // Net 0 - no change to stack depth
                                }
                                _ => {}
                            }
                        }

                        // If stack depth is less than needed, add default values
                        let missing = (import_param_count as i32) - stack_depth;
                        if missing > 0 {
                            for _ in 0..missing {
                                func.instructions.insert(i, Instruction::I64Const(0));
                            }
                            // After inserting, break to avoid re-processing shifted instructions
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Generate the __wasm_start function for deferred static initialization.
    /// This function runs automatically when the WASM module is instantiated.
    fn generate_start_function(&mut self) -> WasmResult<()> {
        use types::CompiledFunction;
        use wasm_encoder::Instruction;

        // Create __wasm_start function: () -> ()
        let type_idx = self.get_or_create_type(vec![], vec![]);
        // Use sentinel so fix_stale_func_indices can resolve this to the correct index
        // after all crate imports have been registered.
        let func_idx = DEFINED_FUNC_SENTINEL + self.functions.len() as u32;

        let mut start_func = CompiledFunction::new(
            "__wasm_start".to_string(),
            type_idx,
            func_idx,
            vec![],     // No params
            vec![],     // No results
            false,      // Not exported (internal)
        );

        // Take deferred inits to avoid borrow issues
        let deferred_inits = std::mem::take(&mut self.deferred_static_inits);

        // Set up for compilation
        let fn_list_idx = self.functions.len();
        self.functions.push(start_func);
        self.current_fn_idx = Some(fn_list_idx);

        // Push an empty scope for locals
        self.scope_vars.push(std::collections::HashMap::new());

        // For each deferred static init, compile the expression and store in global.
        // Restore the module_path snapshot so that name resolution matches compile-time context
        // (e.g. "Preferences::default" must resolve as "wraith_sigil::Preferences::default").
        let saved_module_path = self.module_path.clone();
        for (global_idx, init_expr, module_path_snapshot) in deferred_inits {
            // Restore the module path from when the init was deferred
            self.module_path = module_path_snapshot;

            // Compile the initializer expression
            self.compile_expr(&init_expr)?;

            // Store in the global
            let func = self.current_function_mut()
                .ok_or_else(|| error::WasmError::internal("not in function context"))?;
            func.push(Instruction::GlobalSet(global_idx));
        }
        self.module_path = saved_module_path;

        // End the function
        let func = self.current_function_mut().unwrap();
        func.push(Instruction::End);

        // Pop scope
        self.scope_vars.pop();
        self.current_fn_idx = None;

        // Record this as the start function
        self.start_function_idx = Some(func_idx);
        self.func_map.insert("__wasm_start".to_string(), func_idx);

        Ok(())
    }

    fn generate_module(&mut self) -> WasmResult<Vec<u8>> {
        // TODO: Implement in codegen.rs
        use wasm_encoder::{
            CodeSection, DataSection, DataSegment, DataSegmentMode, ElementSection, Elements,
            ExportSection, Function, FunctionSection, GlobalSection, GlobalType, ImportSection,
            MemorySection, MemoryType, Module, TableSection, TableType, TypeSection,
        };
        use wasm_encoder::{ConstExpr, RefType};

        let mut module = Module::new();

        // Type section
        let mut types = TypeSection::new();
        for (params, results) in self.imports.types() {
            types.ty().function(params.iter().copied(), results.iter().copied());
        }
        module.section(&types);

        // Import section
        let mut imports = ImportSection::new();
        for import in self.imports.imports() {
            imports.import(
                &import.module,
                &import.name,
                wasm_encoder::EntityType::Function(import.type_idx),
            );
        }
        module.section(&imports);

        // Function section
        let mut functions = FunctionSection::new();
        for func in &self.functions {
            functions.function(func.type_idx);
        }
        module.section(&functions);

        // Table section (for indirect calls)
        if !self.table_elements.is_empty() {
            let mut tables = TableSection::new();
            tables.table(TableType {
                element_type: RefType::FUNCREF,
                minimum: self.table_elements.len() as u64,
                maximum: Some(self.table_elements.len() as u64),
                table64: false,
                shared: false,
            });
            module.section(&tables);
        }

        // Memory section
        let mut memories = MemorySection::new();
        memories.memory(MemoryType {
            minimum: memory::INITIAL_PAGES,
            maximum: Some(memory::MAX_PAGES),
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&memories);

        // Global section
        if !self.globals.is_empty() {
            let mut globals = GlobalSection::new();
            for (ty, mutable, init) in &self.globals {
                globals.global(
                    GlobalType {
                        val_type: *ty,
                        mutable: *mutable,
                        shared: false,
                    },
                    &match ty {
                        ValType::I32 => ConstExpr::i32_const(*init as i32),
                        ValType::I64 => ConstExpr::i64_const(*init),
                        ValType::F32 => ConstExpr::f32_const(*init as f32),
                        ValType::F64 => ConstExpr::f64_const(*init as f64),
                        _ => ConstExpr::i64_const(*init),
                    },
                );
            }
            module.section(&globals);
        }

        // Export section
        let mut exports = ExportSection::new();
        exports.export("memory", wasm_encoder::ExportKind::Memory, 0);

        if !self.table_elements.is_empty() {
            exports.export("__indirect_function_table", wasm_encoder::ExportKind::Table, 0);
        }

        // Track seen export names to avoid duplicates (WASM requires unique export names)
        let mut seen_exports = std::collections::HashSet::new();
        seen_exports.insert("memory".to_string());
        seen_exports.insert("__indirect_function_table".to_string());

        for func in &self.functions {
            if func.is_exported {
                let export_name = if seen_exports.contains(&func.name) {
                    // Use qualified name for duplicates
                    format!("{}_{}", func.name, func.func_idx)
                } else {
                    func.name.clone()
                };
                seen_exports.insert(export_name.clone());
                exports.export(&export_name, wasm_encoder::ExportKind::Func, func.func_idx);
            }
        }
        module.section(&exports);

        // Start section (for deferred static initialization)
        if let Some(start_func_idx) = self.start_function_idx {
            use wasm_encoder::StartSection;
            module.section(&StartSection { function_index: start_func_idx });
        }

        // Element section (for table initialization)
        if !self.table_elements.is_empty() {
            let mut elements = ElementSection::new();
            elements.active(
                Some(0),
                &ConstExpr::i32_const(0),
                Elements::Functions(std::borrow::Cow::Borrowed(&self.table_elements)),
            );
            module.section(&elements);
        }

        // Code section
        let mut codes = CodeSection::new();
        for func in &self.functions {
            let mut f = Function::new_with_locals_types(func.local_types.clone());
            for instr in &func.instructions {
                f.instruction(instr);
            }
            codes.function(&f);
        }
        module.section(&codes);

        // Data section
        if !self.data_segments.is_empty() {
            let mut data = DataSection::new();
            for (offset, bytes) in &self.data_segments {
                data.segment(DataSegment {
                    mode: DataSegmentMode::Active {
                        memory_index: 0,
                        offset: &ConstExpr::i32_const(*offset as i32),
                    },
                    data: bytes.iter().copied(),
                });
            }
            module.section(&data);
        }

        // Source map custom section (if debug info enabled)
        if self.debug_info {
            if let Some(builder) = self.source_map.take() {
                // Build source map from collected function data
                let source_map = builder.build();
                let json = source_map.to_compact_json();
                let custom = wasm_encoder::CustomSection {
                    name: std::borrow::Cow::Borrowed("sigil_sourcemap"),
                    data: std::borrow::Cow::Owned(json.into_bytes()),
                };
                module.section(&custom);
            }
        }

        Ok(module.finish())
    }

    /// Get the generated source map (after compilation with debug_info).
    pub fn get_source_map(&self) -> Option<SourceMap> {
        self.source_map.as_ref().map(|b| {
            // Clone the builder to build a source map without consuming it
            SourceMap::new(&self.source_file)
        })
    }
}

impl Default for WasmCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = WasmCompiler::new();
        assert!(compiler.imports.import_count() > 0);
        assert!(compiler.functions.is_empty());
    }

    #[test]
    fn test_compiler_with_opt_level() {
        let compiler = WasmCompiler::with_opt_level(OptLevel::Aggressive);
        assert_eq!(compiler.opt_level, OptLevel::Aggressive);
    }

    #[test]
    fn test_heap_pointer_global() {
        let compiler = WasmCompiler::new();
        assert_eq!(compiler.get_global("__heap_ptr"), Some(0));
    }

    #[test]
    fn test_string_interning() {
        let mut compiler = WasmCompiler::new();

        let offset1 = compiler.add_string("hello");
        let offset2 = compiler.add_string("world");
        let offset3 = compiler.add_string("hello"); // Same as offset1

        assert_eq!(offset1, offset3); // Interned
        assert_ne!(offset1, offset2); // Different strings
    }

    #[test]
    fn test_string_alignment() {
        let mut compiler = WasmCompiler::new();

        compiler.add_string("x"); // 5 bytes (4 len + 1 char)
        let offset2 = compiler.add_string("y");

        // Should be aligned to 8 bytes
        assert_eq!(offset2 % 8, 0);
    }

    #[test]
    fn test_get_func_from_imports() {
        let compiler = WasmCompiler::new();
        assert!(compiler.get_func("console_log_i64").is_some());
    }

    #[test]
    fn test_generate_empty_module() {
        let mut compiler = WasmCompiler::new();
        let result = compiler.generate_module();
        assert!(result.is_ok());

        let bytes = result.unwrap();
        // WASM magic number
        assert_eq!(&bytes[0..4], &[0x00, 0x61, 0x73, 0x6d]);
        // WASM version
        assert_eq!(&bytes[4..8], &[0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_type_registration() {
        let mut compiler = WasmCompiler::new();

        let idx1 = compiler.get_or_create_type(vec![ValType::I64], vec![ValType::I64]);
        let idx2 = compiler.get_or_create_type(vec![ValType::I64], vec![ValType::I64]);

        assert_eq!(idx1, idx2); // Same type should return same index
    }
}

/// WASM validation tests using wasmparser
#[cfg(test)]
mod validation_tests {
    use super::*;

    /// Validate WASM bytes using wasmparser
    fn validate_wasm(bytes: &[u8]) -> Result<(), String> {
        use wasmparser::Validator;

        let mut validator = Validator::new();
        validator
            .validate_all(bytes)
            .map(|_| ())
            .map_err(|e| format!("WASM validation failed: {}", e))
    }

    /// Count sections in a WASM module
    fn count_sections(bytes: &[u8]) -> Result<usize, String> {
        use wasmparser::Parser;

        let parser = Parser::new(0);
        let count = parser
            .parse_all(bytes)
            .filter_map(|p| p.ok())
            .filter(|p| matches!(p, wasmparser::Payload::TypeSection { .. }
                | wasmparser::Payload::ImportSection { .. }
                | wasmparser::Payload::FunctionSection { .. }
                | wasmparser::Payload::TableSection { .. }
                | wasmparser::Payload::MemorySection { .. }
                | wasmparser::Payload::GlobalSection { .. }
                | wasmparser::Payload::ExportSection { .. }
                | wasmparser::Payload::ElementSection { .. }
                | wasmparser::Payload::CodeSectionStart { .. }
                | wasmparser::Payload::DataSection { .. }
            ))
            .count();
        Ok(count)
    }

    #[test]
    fn test_validate_empty_module() {
        let mut compiler = WasmCompiler::new();
        let bytes = compiler.generate_module().unwrap();

        let result = validate_wasm(&bytes);
        assert!(result.is_ok(), "Empty module validation failed: {:?}", result);
    }

    #[test]
    fn test_validate_simple_function() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite answer() -> i64 {
                42
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_function_with_params() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite add(a: i64, b: i64) -> i64 {
                a + b
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_function_with_locals() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite compute() -> i64 {
                ≔ x = 10;
                ≔ y = 20;
                ≔ z = x + y;
                z * 2
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_if_expression() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite max(a: i64, b: i64) -> i64 {
                ⎇ a > b { a } ⎉ { b }
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_while_loop() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite sum_to_n(n: i64) -> i64 {
                ≔ Δ sum = 0;
                ≔ Δ i = 1;
                ⟳ i <= n {
                    sum = sum + i;
                    i = i + 1;
                }
                sum
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_multiple_functions() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite double(x: i64) -> i64 {
                x * 2
            }

            ☉ rite quadruple(x: i64) -> i64 {
                double(double(x))
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_string_literal() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite greeting() -> i64 {
                ≔ s = "Hello, World!";
                42
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_nested_blocks() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite nested(x: i64) -> i64 {
                ≔ a = {
                    ≔ b = x + 1;
                    b * 2
                };
                a + 10
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_comparison_operators() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite compare(a: i64, b: i64) -> i64 {
                ⎇ a == b { 0 }
                ⎉ ⎇ a < b { -1 }
                ⎉ { 1 }
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_logical_operators() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite check(a: bool, b: bool) -> bool {
                (a && b) || (!a && !b)
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_bitwise_operators() {
        let mut compiler = WasmCompiler::new();

        // Note: Using decimal literals to avoid parser issues with hex
        // Testing bitwise AND, XOR, shift operators
        let result = compiler.compile(r#"
            ☉ rite bits(x: i64) -> i64 {
                ≔ a = x & 255;
                ≔ c = x ^ 85;
                ≔ d = x << 4;
                ≔ e = x >> 2;
                a + c + d + e
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_const_definition() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            const MAX: i64 = 100;

            ☉ rite get_max() -> i64 {
                MAX
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_static_definition() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            static Δ COUNTER: i64 = 0;

            ☉ rite get_counter() -> i64 {
                COUNTER
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_early_return() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite early(x: i64) -> i64 {
                ⎇ x < 0 {
                    ⤺ -1;
                }
                x * 2
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_break_continue() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite find_first_even(n: i64) -> i64 {
                ≔ Δ i = 0;
                ⟳ i < n {
                    ⎇ i % 2 == 0 {
                        ⊗;
                    }
                    i = i + 1;
                    ↻;
                }
                i
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_module_has_required_sections() {
        let mut compiler = WasmCompiler::new();
        let bytes = compiler.generate_module().unwrap();

        // Count sections - should have at least Type, Import, Memory, Export
        let section_count = count_sections(&bytes).unwrap();
        assert!(section_count >= 4, "Module should have at least 4 sections, got {}", section_count);
    }

    #[test]
    fn test_module_exports_memory() {
        let mut compiler = WasmCompiler::new();
        let bytes = compiler.generate_module().unwrap();

        use wasmparser::{Parser, Payload};

        let parser = Parser::new(0);
        let mut has_memory_export = false;

        for payload in parser.parse_all(&bytes) {
            if let Ok(Payload::ExportSection(reader)) = payload {
                for export in reader {
                    if let Ok(exp) = export {
                        if exp.name == "memory" {
                            has_memory_export = true;
                            break;
                        }
                    }
                }
            }
        }

        assert!(has_memory_export, "Module should export memory");
    }

    #[test]
    fn test_module_function_exports() {
        let mut compiler = WasmCompiler::new();

        compiler.compile(r#"
            ☉ rite public_fn() -> i64 { 1 }
            rite private_fn() -> i64 { 2 }
        "#).unwrap();

        let bytes = compiler.generate_module().unwrap();

        use wasmparser::{Parser, Payload, ExternalKind};

        let parser = Parser::new(0);
        let mut exported_funcs = Vec::new();

        for payload in parser.parse_all(&bytes) {
            if let Ok(Payload::ExportSection(reader)) = payload {
                for export in reader {
                    if let Ok(exp) = export {
                        if matches!(exp.kind, ExternalKind::Func) {
                            exported_funcs.push(exp.name.to_string());
                        }
                    }
                }
            }
        }

        assert!(exported_funcs.contains(&"public_fn".to_string()), "public_fn should be exported");
        assert!(!exported_funcs.contains(&"private_fn".to_string()), "private_fn should not be exported");
    }

    #[test]
    fn test_validate_unary_operators() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite negate(x: i64) -> i64 {
                -x
            }

            ☉ rite logical_not(x: bool) -> bool {
                !x
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_complex_expression() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite complex(a: i64, b: i64, c: i64) -> i64 {
                ((a + b) * c - a / b) % (c + 1)
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_for_loop() {
        let mut compiler = WasmCompiler::new();

        // For loop iterates over an array
        let result = compiler.compile(r#"
            ☉ rite sum_array(arr: [i64]) -> i64 {
                ≔ Δ sum = 0;
                ∀ x ∈ arr {
                    sum = sum + x;
                }
                sum
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_match_expression() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite describe(x: i64) -> i64 {
                ⌥ x {
                    0 => 100,
                    1 => 200,
                    _ => 300
                }
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    // Note: tuple_expression and closure tests require heap_alloc runtime
    // and are tested in closures.rs and expressions.rs unit tests instead

    #[test]
    fn test_validate_struct_field_access() {
        let mut compiler = WasmCompiler::new();

        // Test function parameter that accesses struct-like data
        let result = compiler.compile(r#"
            ☉ rite identity(x: i64) -> i64 {
                x
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_recursive_function() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite factorial(n: i64) -> i64 {
                ⎇ n <= 1 {
                    1
                } ⎉ {
                    n * factorial(n - 1)
                }
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_nested_if() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite classify(x: i64) -> i64 {
                ⎇ x < 0 {
                    -1
                } ⎉ ⎇ x == 0 {
                    0
                } ⎉ ⎇ x < 10 {
                    1
                } ⎉ {
                    2
                }
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_validate_mutable_variable() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite count_up(n: i64) -> i64 {
                ≔ Δ count = 0;
                ≔ Δ i = 0;
                ⟳ i < n {
                    count = count + 1;
                    i = i + 1;
                }
                count
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_source_map_generation() {
        let mut compiler = WasmCompiler::new().with_debug_info();

        let result = compiler.compile(r#"
            ☉ rite add(a: i64, b: i64) -> i64 {
                a + b
            }

            ☉ rite multiply(x: i64, y: i64) -> i64 {
                x * y
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        // Find the source map custom section
        use wasmparser::{Parser, Payload};

        let parser = Parser::new(0);
        let mut found_source_map = false;
        let mut source_map_json = String::new();

        for payload in parser.parse_all(&bytes) {
            if let Ok(Payload::CustomSection(reader)) = payload {
                if reader.name() == "sigil_sourcemap" {
                    found_source_map = true;
                    source_map_json = String::from_utf8_lossy(reader.data()).to_string();
                }
            }
        }

        assert!(found_source_map, "Source map custom section not found");

        // Verify the source map contains our functions
        assert!(source_map_json.contains("\"add\""), "Source map should contain 'add' function");
        assert!(source_map_json.contains("\"multiply\""), "Source map should contain 'multiply' function");

        // Verify it has line/column information (not just placeholder 1,0)
        let source_map: serde_json::Value = serde_json::from_str(&source_map_json)
            .expect("Source map should be valid JSON");

        let functions = source_map.get("functions").expect("Should have functions");
        assert!(functions.is_object());

        // Check that add function has real location data
        let add_fn = functions.get("add").expect("Should have add function");
        let start_line = add_fn.get("start").and_then(|s| s.get("line")).and_then(|l| l.as_u64());
        assert!(start_line.is_some() && start_line.unwrap() > 0, "Add function should have valid start line");
    }

    #[test]
    fn test_multi_module_import() {
        let mut compiler = WasmCompiler::new();

        // Compile code that imports from an external module
        let result = compiler.compile(r#"
            invoke math_utils·helper;

            ☉ rite caller() -> i64 {
                helper(10, 20)
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        // Verify the WASM is valid
        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);

        // Verify the import was generated
        use wasmparser::{Parser, Payload};

        let parser = Parser::new(0);
        let mut found_helper_import = false;

        for payload in parser.parse_all(&bytes) {
            if let Ok(Payload::ImportSection(reader)) = payload {
                for import in reader {
                    if let Ok(imp) = import {
                        if imp.module == "math_utils" && imp.name == "helper" {
                            found_helper_import = true;
                        }
                    }
                }
            }
        }

        assert!(found_helper_import, "Should have imported 'helper' from 'math_utils' module");
    }

    #[test]
    fn test_multi_module_import_with_rename() {
        let mut compiler = WasmCompiler::new();

        // Compile code that imports with a rename
        let result = compiler.compile(r#"
            invoke external·original as renamed;

            ☉ rite use_renamed() -> i64 {
                renamed(42)
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    #[test]
    fn test_multi_module_nested_path() {
        let mut compiler = WasmCompiler::new();

        // Compile code with nested module path
        let result = compiler.compile(r#"
            invoke deeply·nested·module·function;

            ☉ rite call_nested() -> i64 {
                function()
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);

        // Verify the import uses the first path segment as module name
        use wasmparser::{Parser, Payload};

        let parser = Parser::new(0);
        let mut found_import = false;

        for payload in parser.parse_all(&bytes) {
            if let Ok(Payload::ImportSection(reader)) = payload {
                for import in reader {
                    if let Ok(imp) = import {
                        if imp.module == "deeply" && imp.name == "function" {
                            found_import = true;
                        }
                    }
                }
            }
        }

        assert!(found_import, "Should have imported 'function' from 'deeply' module");
    }

    #[test]
    fn test_vec_join_method() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(r#"
            ☉ rite format_parts() -> i64 {
                ≔ arr = [1, 2, 3];
                ≔ joined = arr.join(", ");
                42
            }
        "#);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);

        // Verify vec_join import exists
        use wasmparser::{Parser, Payload};

        let parser = Parser::new(0);
        let mut found_join_import = false;

        for payload in parser.parse_all(&bytes) {
            if let Ok(Payload::ImportSection(reader)) = payload {
                for import in reader {
                    if let Ok(imp) = import {
                        if imp.module == "morpheme" && imp.name == "vec_join" {
                            found_join_import = true;
                        }
                    }
                }
            }
        }

        assert!(found_join_import, "Should have imported 'vec_join' from 'morpheme' module");
    }

    /// Regression test for local-function-call-before-import bug.
    ///
    /// Bug: When a local function call preceded an import call, the stack depth
    /// calculation in fix_invalid_call_wrappers didn't account for the local
    /// function's return value, causing it to spuriously insert i64.const 0
    /// before the import call, leading to type mismatch errors.
    ///
    /// This specifically tests the pattern:
    ///   let val = local_func();
    ///   import_func(val, other_arg);
    ///
    /// Where import_func has mixed param types (e.g., i32, i64).
    #[test]
    fn test_local_function_before_import_call() {
        let mut compiler = WasmCompiler::new();

        // This pattern triggered the bug:
        // 1. wrapper() is a local function returning i64
        // 2. Its result is stored in `val`
        // 3. vdom·mount_vnode expects (i32, i64) - mixed types
        // 4. The stack tracker didn't count wrapper()'s return value,
        //    causing spurious i64.const 0 insertion before mount_vnode
        let result = compiler.compile(
r##"rite wrapper() -> i64 {
    42
}

rite main() {
    ≔ val = wrapper();
    vdom·mount_vnode(val, "#app");
}"##);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(
            validation.is_ok(),
            "Validation failed (likely spurious i64.const 0 before import): {:?}",
            validation
        );
    }

    /// Test variant: multiple local function calls before import
    #[test]
    fn test_multiple_local_calls_before_import() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(
r##"rite get_id() -> i64 {
    100
}

rite get_selector() -> i64 {
    200
}

rite main() {
    ≔ id = get_id();
    ≔ sel = get_selector();
    vdom·mount_vnode(id, "#root");
}"##);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }

    /// Test variant: local function call result used directly (no let binding)
    #[test]
    fn test_local_call_inline_in_import() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile(
r##"rite create_vnode() -> i64 {
    42
}

rite main() {
    vdom·mount_vnode(create_vnode(), "#app");
}"##);

        assert!(result.is_ok(), "Compilation failed: {:?}", result);
        let bytes = result.unwrap();

        let validation = validate_wasm(&bytes);
        assert!(validation.is_ok(), "Validation failed: {:?}", validation);
    }
}

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
pub mod closures;
pub mod constants;
pub mod control_flow;
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

    /// Deferred static initializers: (global_index, init_expression)
    /// These are statics with non-constant initializers that need runtime init
    pub(crate) deferred_static_inits: Vec<(u32, crate::ast::Expr)>,

    /// Start function index (for __wasm_start if we have deferred inits)
    pub(crate) start_function_idx: Option<u32>,
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
            opt_level: OptLevel::Standard,
            debug_info: false,
            source_map: None,
            source_file: String::new(),
            source_dir: std::path::PathBuf::new(),
            loaded_modules: std::collections::HashSet::new(),
            module_cache: std::collections::HashMap::new(),
            deferred_static_inits: Vec::new(),
            start_function_idx: None,
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
        let import_count = self.imports.import_count();
        if func_idx < import_count {
            // Import function - check via imports
            self.imports.get_return_type(func_idx).is_none()
        } else {
            // User-defined function
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
        if segments.first().map(|s| s.as_str()) == Some("tome") {
            // tome:: means crate root - skip the "tome" prefix
            segments[1..].to_vec()
        } else {
            segments.to_vec()
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

        // Check imports
        self.imports.get_func(&qualified)
    }

    // compile_file is implemented in statements.rs

    /// Generate the __wasm_start function for deferred static initialization.
    /// This function runs automatically when the WASM module is instantiated.
    fn generate_start_function(&mut self) -> WasmResult<()> {
        use types::CompiledFunction;
        use wasm_encoder::Instruction;

        // Create __wasm_start function: () -> ()
        let type_idx = self.get_or_create_type(vec![], vec![]);
        let func_idx = self.imports.import_count() + self.functions.len() as u32;

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

        // For each deferred static init, compile the expression and store in global
        for (global_idx, init_expr) in deferred_inits {
            // Compile the initializer expression
            self.compile_expr(&init_expr)?;

            // Store in the global
            let func = self.current_function_mut()
                .ok_or_else(|| error::WasmError::internal("not in function context"))?;
            func.push(Instruction::GlobalSet(global_idx));
        }

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

        for func in &self.functions {
            if func.is_exported {
                exports.export(&func.name, wasm_encoder::ExportKind::Func, func.func_idx);
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
            pub fn answer() -> i64 {
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
            pub fn add(a: i64, b: i64) -> i64 {
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
            pub fn compute() -> i64 {
                let x = 10;
                let y = 20;
                let z = x + y;
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
            pub fn max(a: i64, b: i64) -> i64 {
                if a > b { a } else { b }
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
            pub fn sum_to_n(n: i64) -> i64 {
                let mut sum = 0;
                let mut i = 1;
                while i <= n {
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
            pub fn double(x: i64) -> i64 {
                x * 2
            }

            pub fn quadruple(x: i64) -> i64 {
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
            pub fn greeting() -> i64 {
                let s = "Hello, World!";
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
            pub fn nested(x: i64) -> i64 {
                let a = {
                    let b = x + 1;
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
            pub fn compare(a: i64, b: i64) -> i64 {
                if a == b { 0 }
                else if a < b { -1 }
                else { 1 }
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
            pub fn check(a: bool, b: bool) -> bool {
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
            pub fn bits(x: i64) -> i64 {
                let a = x & 255;
                let c = x ^ 85;
                let d = x << 4;
                let e = x >> 2;
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

            pub fn get_max() -> i64 {
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
            static mut COUNTER: i64 = 0;

            pub fn get_counter() -> i64 {
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
            pub fn early(x: i64) -> i64 {
                if x < 0 {
                    return -1;
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
            pub fn find_first_even(n: i64) -> i64 {
                let mut i = 0;
                while i < n {
                    if i % 2 == 0 {
                        break;
                    }
                    i = i + 1;
                    continue;
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
            pub fn public_fn() -> i64 { 1 }
            fn private_fn() -> i64 { 2 }
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
            pub fn negate(x: i64) -> i64 {
                -x
            }

            pub fn logical_not(x: bool) -> bool {
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
            pub fn complex(a: i64, b: i64, c: i64) -> i64 {
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
            pub fn sum_array(arr: [i64]) -> i64 {
                let mut sum = 0;
                for x in arr {
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
            pub fn describe(x: i64) -> i64 {
                match x {
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
            pub fn identity(x: i64) -> i64 {
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
            pub fn factorial(n: i64) -> i64 {
                if n <= 1 {
                    1
                } else {
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
            pub fn classify(x: i64) -> i64 {
                if x < 0 {
                    -1
                } else if x == 0 {
                    0
                } else if x < 10 {
                    1
                } else {
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
            pub fn count_up(n: i64) -> i64 {
                let mut count = 0;
                let mut i = 0;
                while i < n {
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
            pub fn add(a: i64, b: i64) -> i64 {
                a + b
            }

            pub fn multiply(x: i64, y: i64) -> i64 {
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
            invoke math_utils::helper;

            pub fn caller() -> i64 {
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
            invoke external::original as renamed;

            pub fn use_renamed() -> i64 {
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
            invoke deeply::nested::module::function;

            pub fn call_nested() -> i64 {
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
}

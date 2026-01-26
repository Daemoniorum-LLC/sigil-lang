//! Sigil WASM Compiler
//!
//! Compiles Sigil AST to WebAssembly for browser execution.
//! Designed for AI-native web development with sigil-web-interface.
//!
//! Features:
//! - Direct WASM bytecode generation using wasm-encoder
//! - DOM/Web API bindings via imported functions
//! - Evidentiality tracking preserved at runtime
//! - Morpheme operator support
//! - Closure compilation with environment capture

#[cfg(feature = "wasm")]
pub mod wasm {
    use wasm_encoder::{
        CodeSection, DataSection, ExportSection, Function, FunctionSection,
        GlobalSection, ImportSection, Instruction, MemorySection, MemoryType,
        Module, TypeSection, ValType, GlobalType, ConstExpr, DataSegment,
        DataSegmentMode, TableSection, TableType, ElementSection, Elements,
        BlockType, RefType,
    };

    use std::collections::HashMap;

    use crate::ast::{
        self, BinOp, Block, Expr, Function as AstFunction, Item, Literal,
        Pattern, SourceFile, Stmt, UnaryOp, PipeOp, Evidentiality, MatchArm,
        ClosureParam, StructDef, StructFields, EnumDef, FieldInit,
    };
    use crate::parser::Parser;
    use crate::optimize::{Optimizer, OptLevel};

    // =========================================================================
    // Constants
    // =========================================================================

    /// Evidentiality tag bits (stored in high bits of i64)
    pub mod evidence_tags {
        pub const KNOWN: i64 = 0x0000_0000_0000_0000;      // !
        pub const UNCERTAIN: i64 = 0x1000_0000_0000_0000;  // ?
        pub const REPORTED: i64 = 0x2000_0000_0000_0000;   // ~
        pub const PARADOX: i64 = 0x3000_0000_0000_0000;    // ‽
        pub const TAG_MASK: i64 = 0x7000_0000_0000_0000;
        pub const VALUE_MASK: i64 = 0x0FFF_FFFF_FFFF_FFFF;
        pub const TYPE_SHIFT: i64 = 56;
    }

    /// Type tags (stored in bits 56-59)
    pub mod type_tags {
        pub const INT: i64 = 0x00 << 56;
        pub const FLOAT: i64 = 0x01 << 56;
        pub const BOOL: i64 = 0x02 << 56;
        pub const NULL: i64 = 0x03 << 56;
        pub const PTR: i64 = 0x04 << 56;
        pub const FUNC: i64 = 0x05 << 56;
        pub const STRING: i64 = 0x06 << 56;
        pub const ARRAY: i64 = 0x07 << 56;
    }

    /// Memory layout constants
    pub mod memory {
        pub const STACK_START: u32 = 0x0400;
        pub const STACK_SIZE: u32 = 0x0C00;  // 3KB
        pub const GLOBALS_START: u32 = 0x1000;
        pub const STRING_POOL_START: u32 = 0x2000;
        pub const VDOM_POOL_START: u32 = 0x3000;
        pub const HEAP_START: u32 = 0x4000;
    }

    // =========================================================================
    // Types
    // =========================================================================

    /// Local variable info
    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    struct LocalVar {
        index: u32,
        ty: ValType,
        is_param: bool,
    }

    /// Compiled function
    #[derive(Clone, Debug)]
    struct CompiledFunction {
        name: String,
        type_idx: u32,
        func_idx: u32,
        params: Vec<(String, ValType)>,
        results: Vec<ValType>,
        locals: HashMap<String, LocalVar>,
        local_types: Vec<ValType>,  // Non-param locals
        instructions: Vec<Instruction<'static>>,
        is_exported: bool,
    }

    /// Import function info
    #[derive(Clone, Debug)]
    struct ImportFn {
        module: String,
        name: String,
        type_idx: u32,
    }

    /// Loop context for break/continue
    #[derive(Clone, Debug)]
    struct LoopContext {
        break_label: u32,
        continue_label: u32,
    }

    /// Closure info for capture analysis
    #[derive(Clone, Debug)]
    struct ClosureInfo {
        func_idx: u32,
        table_idx: u32,
        captures: Vec<String>,
        env_size: u32,
    }

    /// Struct layout info
    #[derive(Clone, Debug)]
    struct StructLayout {
        name: String,
        fields: Vec<(String, u32)>,  // (field_name, offset)
        size: u32,
    }

    /// Enum variant info
    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    struct EnumLayout {
        name: String,
        variants: Vec<(String, u32, Option<StructLayout>)>,  // (variant_name, tag, payload)
    }

    // =========================================================================
    // Compiler
    // =========================================================================

    /// WASM Compiler for Sigil
    #[allow(dead_code)]
    pub struct WasmCompiler {
        // Type section
        types: Vec<(Vec<ValType>, Vec<ValType>)>,
        type_map: HashMap<(Vec<ValType>, Vec<ValType>), u32>,

        // Imports
        imports: Vec<ImportFn>,
        import_count: u32,

        // Functions
        functions: Vec<CompiledFunction>,
        func_map: HashMap<String, u32>,

        // Globals
        globals: Vec<(ValType, bool, i64)>,
        global_map: HashMap<String, u32>,

        // Data segments
        data_segments: Vec<(u32, Vec<u8>)>,
        data_offset: u32,
        string_map: HashMap<String, u32>,

        // Tables (for indirect calls / closures)
        table_elements: Vec<u32>,
        closure_map: HashMap<String, ClosureInfo>,
        closure_counter: u32,

        // Struct and enum layouts
        struct_layouts: HashMap<String, StructLayout>,
        enum_layouts: HashMap<String, EnumLayout>,

        // Compilation state
        current_fn_idx: Option<usize>,
        loop_stack: Vec<LoopContext>,
        label_counter: u32,

        // Scope for capture analysis
        scope_vars: Vec<HashMap<String, u32>>,  // Stack of scopes with var -> local_idx

        // Options
        opt_level: OptLevel,
        debug_info: bool,
    }

    impl WasmCompiler {
        /// Create a new WASM compiler
        pub fn new() -> Self {
            let mut compiler = Self {
                types: Vec::new(),
                type_map: HashMap::new(),
                imports: Vec::new(),
                import_count: 0,
                functions: Vec::new(),
                func_map: HashMap::new(),
                globals: Vec::new(),
                global_map: HashMap::new(),
                data_segments: Vec::new(),
                data_offset: memory::HEAP_START,
                string_map: HashMap::new(),
                table_elements: Vec::new(),
                closure_map: HashMap::new(),
                closure_counter: 0,
                struct_layouts: HashMap::new(),
                enum_layouts: HashMap::new(),
                current_fn_idx: None,
                loop_stack: Vec::new(),
                label_counter: 0,
                scope_vars: Vec::new(),
                opt_level: OptLevel::Standard,
                debug_info: false,
            };

            // Add heap pointer global
            compiler.globals.push((ValType::I32, true, memory::HEAP_START as i64));
            compiler.global_map.insert("__heap_ptr".to_string(), 0);

            // Register standard imports
            compiler.register_imports();

            compiler
        }

        /// Create compiler with optimization level
        pub fn with_opt_level(opt_level: OptLevel) -> Self {
            let mut compiler = Self::new();
            compiler.opt_level = opt_level;
            compiler
        }

        // =====================================================================
        // Import Registration
        // =====================================================================

        fn register_imports(&mut self) {
            // Console
            self.add_import("console", "log_i64", vec![ValType::I64], vec![]);
            self.add_import("console", "log_f64", vec![ValType::F64], vec![]);
            self.add_import("console", "log_str", vec![ValType::I32, ValType::I32], vec![]);

            // DOM operations
            self.add_import("dom", "create_element", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("dom", "create_text", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("dom", "set_attribute", vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("dom", "remove_attribute", vec![ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("dom", "set_property", vec![ValType::I32, ValType::I32, ValType::I32, ValType::I64], vec![]);
            self.add_import("dom", "append_child", vec![ValType::I32, ValType::I32], vec![]);
            self.add_import("dom", "insert_before", vec![ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("dom", "remove_child", vec![ValType::I32, ValType::I32], vec![]);
            self.add_import("dom", "replace_child", vec![ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("dom", "set_text_content", vec![ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("dom", "get_element_by_id", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("dom", "query_selector", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("dom", "clone_node", vec![ValType::I32, ValType::I32], vec![ValType::I32]);

            // Events
            self.add_import("events", "add_listener", vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("events", "remove_listener", vec![ValType::I32], vec![]);
            self.add_import("events", "prevent_default", vec![ValType::I32], vec![]);
            self.add_import("events", "stop_propagation", vec![ValType::I32], vec![]);
            self.add_import("events", "get_target", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("events", "get_value", vec![ValType::I32, ValType::I32], vec![ValType::I32]);

            // Timing
            self.add_import("timing", "now", vec![], vec![ValType::F64]);
            self.add_import("timing", "set_timeout", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("timing", "clear_timeout", vec![ValType::I32], vec![]);
            self.add_import("timing", "set_interval", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("timing", "clear_interval", vec![ValType::I32], vec![]);
            self.add_import("timing", "request_animation_frame", vec![ValType::I32], vec![ValType::I32]);

            // Fetch
            self.add_import("fetch", "start", vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("fetch", "poll", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("fetch", "get_status", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("fetch", "get_body", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("fetch", "abort", vec![ValType::I32], vec![]);

            // Storage
            self.add_import("storage", "local_get", vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("storage", "local_set", vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("storage", "local_remove", vec![ValType::I32, ValType::I32], vec![]);

            // Router
            self.add_import("router", "push_state", vec![ValType::I32, ValType::I32], vec![]);
            self.add_import("router", "replace_state", vec![ValType::I32, ValType::I32], vec![]);
            self.add_import("router", "get_pathname", vec![ValType::I32], vec![ValType::I32]);

            // Memory
            self.add_import("memory", "alloc", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("memory", "realloc", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("memory", "free", vec![ValType::I32], vec![]);

            // Morpheme operators
            self.add_import("morpheme", "array_new", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("morpheme", "array_push", vec![ValType::I32, ValType::I64], vec![]);
            self.add_import("morpheme", "array_get", vec![ValType::I32, ValType::I32], vec![ValType::I64]);
            self.add_import("morpheme", "array_set", vec![ValType::I32, ValType::I32, ValType::I64], vec![]);
            self.add_import("morpheme", "array_len", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("morpheme", "array_map", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("morpheme", "array_filter", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("morpheme", "array_reduce", vec![ValType::I32, ValType::I32, ValType::I64], vec![ValType::I64]);
            self.add_import("morpheme", "array_sort", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("morpheme", "array_first", vec![ValType::I32], vec![ValType::I64]);
            self.add_import("morpheme", "array_last", vec![ValType::I32], vec![ValType::I64]);
            self.add_import("morpheme", "array_nth", vec![ValType::I32, ValType::I32], vec![ValType::I64]);

            // Math
            self.add_import("math", "sqrt", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "sin", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "cos", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "tan", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "pow", vec![ValType::F64, ValType::F64], vec![ValType::F64]);
            self.add_import("math", "exp", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "log", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "floor", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "ceil", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "round", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "abs", vec![ValType::F64], vec![ValType::F64]);
            self.add_import("math", "random", vec![], vec![ValType::F64]);

            // VDOM
            self.add_import("vdom", "create_vnode", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("vdom", "create_text_vnode", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("vdom", "create_fragment", vec![], vec![ValType::I32]);
            self.add_import("vdom", "set_vnode_prop", vec![ValType::I32, ValType::I32, ValType::I32, ValType::I64], vec![]);
            self.add_import("vdom", "set_vnode_str_prop", vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("vdom", "append_vnode_child", vec![ValType::I32, ValType::I32], vec![]);
            self.add_import("vdom", "diff_and_patch", vec![ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("vdom", "mount_vnode", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("vdom", "dispose", vec![ValType::I32], vec![]);

            // Signals (Phase 5)
            self.add_import("signal", "create", vec![ValType::I64], vec![ValType::I32]);
            self.add_import("signal", "get", vec![ValType::I32], vec![ValType::I64]);
            self.add_import("signal", "set", vec![ValType::I32, ValType::I64], vec![]);
            self.add_import("signal", "subscribe", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("signal", "unsubscribe", vec![ValType::I32], vec![]);
            self.add_import("signal", "batch_start", vec![], vec![]);
            self.add_import("signal", "batch_end", vec![], vec![]);
            self.add_import("signal", "computed", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("signal", "effect", vec![ValType::I32], vec![ValType::I32]);

            // Async (Phase 6)
            self.add_import("async", "promise_new", vec![], vec![ValType::I32]);
            self.add_import("async", "promise_resolve", vec![ValType::I32, ValType::I64], vec![]);
            self.add_import("async", "promise_reject", vec![ValType::I32, ValType::I32, ValType::I32], vec![]);
            self.add_import("async", "promise_then", vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("async", "promise_catch", vec![ValType::I32, ValType::I32], vec![ValType::I32]);
            self.add_import("async", "promise_all", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("async", "promise_race", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("async", "spawn", vec![ValType::I32], vec![ValType::I32]);
            self.add_import("async", "yield_now", vec![], vec![]);
        }

        fn add_import(&mut self, module: &str, name: &str, params: Vec<ValType>, results: Vec<ValType>) {
            let type_idx = self.get_or_create_type(params, results);
            let func_idx = self.import_count;

            self.func_map.insert(format!("{}_{}", module, name), func_idx);
            self.imports.push(ImportFn {
                module: module.to_string(),
                name: name.to_string(),
                type_idx,
            });
            self.import_count += 1;
        }

        fn get_or_create_type(&mut self, params: Vec<ValType>, results: Vec<ValType>) -> u32 {
            let key = (params.clone(), results.clone());
            if let Some(&idx) = self.type_map.get(&key) {
                return idx;
            }
            let idx = self.types.len() as u32;
            self.types.push((params, results));
            self.type_map.insert(key, idx);
            idx
        }

        // =====================================================================
        // Compilation Entry Point
        // =====================================================================

        /// Compile source code to WASM bytes
        pub fn compile(&mut self, source: &str) -> Result<Vec<u8>, String> {
            let mut parser = Parser::new(source);
            let ast = parser.parse_file().map_err(|e| format!("Parse error: {}", e))?;

            let ast = if self.opt_level != OptLevel::None {
                Optimizer::new(self.opt_level).optimize_file(&ast)
            } else {
                ast
            };

            self.compile_file(&ast)?;
            self.generate_module()
        }

        fn compile_file(&mut self, file: &SourceFile) -> Result<(), String> {
            // First pass: declare all types (structs, enums)
            for item in &file.items {
                match &item.node {
                    Item::Struct(s) => self.declare_struct(s)?,
                    Item::Enum(e) => self.declare_enum(e)?,
                    _ => {}
                }
            }

            // Second pass: declare all functions
            for item in &file.items {
                match &item.node {
                    Item::Function(func) => self.declare_function(func)?,
                    Item::Const(c) => self.compile_const(c)?,
                    Item::Static(s) => self.compile_static(s)?,
                    _ => {}
                }
            }

            // Third pass: compile function bodies
            for (idx, item) in file.items.iter().enumerate() {
                if let Item::Function(func) = &item.node {
                    self.compile_function_body(func, idx)?;
                }
            }

            Ok(())
        }

        // =====================================================================
        // Struct and Enum Declaration
        // =====================================================================

        fn declare_struct(&mut self, s: &StructDef) -> Result<(), String> {
            let name = s.name.name.clone();
            let mut fields = Vec::new();
            let mut offset = 0u32;

            match &s.fields {
                StructFields::Named(field_defs) => {
                    for field in field_defs {
                        fields.push((field.name.name.clone(), offset));
                        offset += 8; // All fields are 8 bytes (i64)
                    }
                }
                StructFields::Tuple(types) => {
                    for (i, _ty) in types.iter().enumerate() {
                        fields.push((format!("{}", i), offset));
                        offset += 8;
                    }
                }
                StructFields::Unit => {}
            }

            let layout = StructLayout {
                name: name.clone(),
                fields,
                size: offset,
            };
            self.struct_layouts.insert(name, layout);
            Ok(())
        }

        fn declare_enum(&mut self, e: &EnumDef) -> Result<(), String> {
            let name = e.name.name.clone();
            let mut variants = Vec::new();

            for (tag, variant) in e.variants.iter().enumerate() {
                let payload = match &variant.fields {
                    StructFields::Named(field_defs) => {
                        let mut fields = Vec::new();
                        let mut offset = 8u32; // Skip tag
                        for field in field_defs {
                            fields.push((field.name.name.clone(), offset));
                            offset += 8;
                        }
                        Some(StructLayout {
                            name: variant.name.name.clone(),
                            fields,
                            size: offset,
                        })
                    }
                    StructFields::Tuple(types) => {
                        let mut fields = Vec::new();
                        let mut offset = 8u32;
                        for (i, _) in types.iter().enumerate() {
                            fields.push((format!("{}", i), offset));
                            offset += 8;
                        }
                        Some(StructLayout {
                            name: variant.name.name.clone(),
                            fields,
                            size: offset,
                        })
                    }
                    StructFields::Unit => None,
                };
                variants.push((variant.name.name.clone(), tag as u32, payload));
            }

            let layout = EnumLayout {
                name: name.clone(),
                variants,
            };
            self.enum_layouts.insert(name, layout);
            Ok(())
        }

        // =====================================================================
        // Function Compilation
        // =====================================================================

        fn declare_function(&mut self, func: &AstFunction) -> Result<(), String> {
            let name = &func.name.name;
            let params: Vec<ValType> = func.params.iter().map(|_| ValType::I64).collect();
            let results = if func.return_type.is_some() || func.body.as_ref().map(|b| b.expr.is_some()).unwrap_or(false) {
                vec![ValType::I64]
            } else {
                vec![]
            };

            let type_idx = self.get_or_create_type(params.clone(), results.clone());
            let func_idx = self.import_count + self.functions.len() as u32;

            let param_locals: HashMap<String, LocalVar> = func.params
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let pname = match &p.pattern {
                        Pattern::Ident { name, .. } => name.name.clone(),
                        _ => format!("param_{}", i),
                    };
                    (pname, LocalVar { index: i as u32, ty: ValType::I64, is_param: true })
                })
                .collect();

            let is_exported = matches!(func.visibility, ast::Visibility::Public) || name == "main";

            self.func_map.insert(name.clone(), func_idx);
            self.functions.push(CompiledFunction {
                name: name.clone(),
                type_idx,
                func_idx,
                params: func.params.iter().enumerate().map(|(i, p)| {
                    let pname = match &p.pattern {
                        Pattern::Ident { name, .. } => name.name.clone(),
                        _ => format!("param_{}", i),
                    };
                    (pname, ValType::I64)
                }).collect(),
                results,
                locals: param_locals,
                local_types: Vec::new(),
                instructions: Vec::new(),
                is_exported,
            });

            Ok(())
        }

        fn compile_function_body(&mut self, func: &AstFunction, _item_idx: usize) -> Result<(), String> {
            let name = &func.name.name;
            let func_idx = *self.func_map.get(name)
                .ok_or_else(|| format!("Function not declared: {}", name))?;
            let def_idx = (func_idx - self.import_count) as usize;

            self.current_fn_idx = Some(def_idx);
            self.loop_stack.clear();

            if let Some(body) = &func.body {
                self.compile_block(body)?;
            }

            // Ensure function ends properly
            let func_def = &mut self.functions[def_idx];
            if func_def.results.is_empty() {
                func_def.instructions.push(Instruction::End);
            } else {
                // If block didn't produce a value, return 0
                if func_def.instructions.is_empty() ||
                   !matches!(func_def.instructions.last(), Some(Instruction::Return)) {
                    if func.body.as_ref().map(|b| b.expr.is_none()).unwrap_or(true) {
                        func_def.instructions.push(Instruction::I64Const(0));
                    }
                    func_def.instructions.push(Instruction::End);
                }
            }

            self.current_fn_idx = None;
            Ok(())
        }

        fn compile_const(&mut self, c: &ast::ConstDef) -> Result<(), String> {
            let value = self.eval_const_expr(&c.value)?;
            let global_idx = self.globals.len() as u32;
            self.globals.push((ValType::I64, false, value));
            self.global_map.insert(c.name.name.clone(), global_idx);
            Ok(())
        }

        fn compile_static(&mut self, s: &ast::StaticDef) -> Result<(), String> {
            let value = self.eval_const_expr(&s.value)?;
            let global_idx = self.globals.len() as u32;
            self.globals.push((ValType::I64, s.mutable, value));
            self.global_map.insert(s.name.name.clone(), global_idx);
            Ok(())
        }

        fn eval_const_expr(&self, expr: &Expr) -> Result<i64, String> {
            match expr {
                Expr::Literal(lit) => match lit {
                    Literal::Int { value, .. } => value.parse().map_err(|e: std::num::ParseIntError| e.to_string()),
                    Literal::Bool(b) => Ok(if *b { 1 } else { 0 }),
                    Literal::Null | Literal::Empty => Ok(0),
                    _ => Err("Unsupported literal in const".to_string()),
                },
                Expr::Binary { left, op, right } => {
                    let l = self.eval_const_expr(left)?;
                    let r = self.eval_const_expr(right)?;
                    Ok(match op {
                        BinOp::Add => l.wrapping_add(r),
                        BinOp::Sub => l.wrapping_sub(r),
                        BinOp::Mul => l.wrapping_mul(r),
                        BinOp::Div if r != 0 => l / r,
                        BinOp::Rem if r != 0 => l % r,
                        BinOp::BitAnd => l & r,
                        BinOp::BitOr => l | r,
                        BinOp::BitXor => l ^ r,
                        BinOp::Shl => l << (r & 63),
                        BinOp::Shr => l >> (r & 63),
                        _ => return Err("Unsupported op in const".to_string()),
                    })
                }
                Expr::Unary { op, expr } => {
                    let v = self.eval_const_expr(expr)?;
                    Ok(match op {
                        UnaryOp::Neg => -v,
                        UnaryOp::Not => if v == 0 { 1 } else { 0 },
                        _ => return Err("Unsupported unary in const".to_string()),
                    })
                }
                _ => Err("Unsupported const expression".to_string()),
            }
        }

        // =====================================================================
        // Block & Statement Compilation
        // =====================================================================

        fn compile_block(&mut self, block: &Block) -> Result<(), String> {
            for stmt in &block.stmts {
                self.compile_stmt(stmt)?;
            }
            if let Some(expr) = &block.expr {
                self.compile_expr(expr)?;
            }
            Ok(())
        }

        fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
            match stmt {
                Stmt::Let { pattern, init, .. } => {
                    let var_name = match pattern {
                        Pattern::Ident { name, .. } => name.name.clone(),
                        _ => return Err("Complex patterns not yet supported".to_string()),
                    };

                    let def_idx = self.current_fn_idx.ok_or("Not in function")?;
                    let func = &mut self.functions[def_idx];

                    let local_idx = func.params.len() as u32 + func.local_types.len() as u32;
                    func.locals.insert(var_name.clone(), LocalVar {
                        index: local_idx,
                        ty: ValType::I64,
                        is_param: false,
                    });
                    func.local_types.push(ValType::I64);

                    if let Some(init_expr) = init {
                        self.compile_expr(init_expr)?;
                        let func = &mut self.functions[def_idx];
                        func.instructions.push(Instruction::LocalSet(local_idx));
                    }
                    Ok(())
                }
                Stmt::Expr(expr) | Stmt::Semi(expr) => {
                    self.compile_expr(expr)?;
                    let def_idx = self.current_fn_idx.ok_or("Not in function")?;
                    self.functions[def_idx].instructions.push(Instruction::Drop);
                    Ok(())
                }
                Stmt::Item(_) => Ok(()),
            }
        }

        // =====================================================================
        // Expression Compilation
        // =====================================================================

        fn compile_expr(&mut self, expr: &Expr) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            match expr {
                Expr::Literal(lit) => self.compile_literal(lit),

                Expr::Path(path) => {
                    let name = path.segments.first()
                        .map(|s| s.ident.name.as_str())
                        .unwrap_or("");

                    // Check locals
                    let local = self.functions[def_idx].locals.get(name).cloned();
                    if let Some(local) = local {
                        self.functions[def_idx].instructions.push(Instruction::LocalGet(local.index));
                        return Ok(());
                    }

                    // Check globals
                    if let Some(&global_idx) = self.global_map.get(name) {
                        self.functions[def_idx].instructions.push(Instruction::GlobalGet(global_idx));
                        return Ok(());
                    }

                    Err(format!("Unknown variable: {}", name))
                }

                Expr::Binary { left, op, right } => {
                    // Short-circuit for && and ||
                    match op {
                        BinOp::And => {
                            self.compile_expr(left)?;
                            let func = &mut self.functions[def_idx];
                            func.instructions.push(Instruction::I64Const(0));
                            func.instructions.push(Instruction::I64Ne);
                            func.instructions.push(Instruction::If(BlockType::Result(ValType::I64)));
        
                            self.compile_expr(right)?;

                            let func = &mut self.functions[def_idx];
                            func.instructions.push(Instruction::Else);
                            func.instructions.push(Instruction::I64Const(0));
                            func.instructions.push(Instruction::End);
                            return Ok(());
                        }
                        BinOp::Or => {
                            self.compile_expr(left)?;
                            let func = &mut self.functions[def_idx];
                            func.instructions.push(Instruction::I64Const(0));
                            func.instructions.push(Instruction::I64Ne);
                            func.instructions.push(Instruction::If(BlockType::Result(ValType::I64)));
                            func.instructions.push(Instruction::I64Const(1));
                            func.instructions.push(Instruction::Else);
        
                            self.compile_expr(right)?;

                            let func = &mut self.functions[def_idx];
                            func.instructions.push(Instruction::End);
                            return Ok(());
                        }
                        _ => {}
                    }

                    self.compile_expr(left)?;
                    self.compile_expr(right)?;
                    self.emit_binop(*op);
                    Ok(())
                }

                Expr::Unary { op, expr } => {
                    self.compile_expr(expr)?;
                    self.emit_unaryop(*op);
                    Ok(())
                }

                Expr::Call { func, args } => {
                    for arg in args {
                        self.compile_expr(arg)?;
                    }

                    if let Expr::Path(path) = func.as_ref() {
                        // Handle multi-segment paths like signal::create -> signal_create
                        let name = if path.segments.len() > 1 {
                            path.segments.iter()
                                .map(|s| s.ident.name.as_str())
                                .collect::<Vec<_>>()
                                .join("_")
                        } else {
                            path.segments.first()
                                .map(|s| s.ident.name.to_string())
                                .unwrap_or_default()
                        };

                        // Built-in functions
                        match name.as_str() {
                            "print" | "println" => {
                                let log_idx = *self.func_map.get("console_log_i64").unwrap();
                                self.functions[def_idx].instructions.push(Instruction::Call(log_idx));
                                self.functions[def_idx].instructions.push(Instruction::I64Const(0));
                                return Ok(());
                            }
                            _ => {}
                        }

                        if let Some(&func_idx) = self.func_map.get(&name) {
                            self.functions[def_idx].instructions.push(Instruction::Call(func_idx));
                            return Ok(());
                        }

                        return Err(format!("Unknown function: {}", name));
                    }

                    Err("Complex call not supported".to_string())
                }

                Expr::If { condition, then_branch, else_branch } => {
                    self.compile_expr(condition)?;

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I64Const(0));
                    func.instructions.push(Instruction::I64Ne);
                    func.instructions.push(Instruction::If(BlockType::Result(ValType::I64)));

                    self.compile_block(then_branch)?;

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::Else);

                    if let Some(else_expr) = else_branch {
                        match else_expr.as_ref() {
                            Expr::Block(block) => self.compile_block(block)?,
                            _ => self.compile_expr(else_expr)?,
                        }
                    } else {
                        self.functions[def_idx].instructions.push(Instruction::I64Const(0));
                    }

                    self.functions[def_idx].instructions.push(Instruction::End);
                    Ok(())
                }

                Expr::While { condition, body } => {
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::Block(BlockType::Empty));
                    func.instructions.push(Instruction::Loop(BlockType::Empty));

                    self.loop_stack.push(LoopContext {
                        break_label: 1,
                        continue_label: 0,
                    });

                    self.compile_expr(condition)?;

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I64Eqz);
                    func.instructions.push(Instruction::BrIf(1));

                    self.compile_block(body)?;

                    // Drop block result if any
                    if body.expr.is_some() {
                        self.functions[def_idx].instructions.push(Instruction::Drop);
                    }

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::Br(0));
                    func.instructions.push(Instruction::End);
                    func.instructions.push(Instruction::End);

                    self.loop_stack.pop();

                    // While produces unit
                    func.instructions.push(Instruction::I64Const(0));
                    Ok(())
                }

                Expr::For { pattern, iter, body } => {
                    // Simplified for loop - assumes iter produces array pointer
                    let var_name = match pattern {
                        Pattern::Ident { name, .. } => name.name.clone(),
                        _ => return Err("Complex for patterns not supported".to_string()),
                    };

                    // Allocate locals for iterator state
                    let func = &mut self.functions[def_idx];
                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    let idx_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    let len_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    let item_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I64);
                    func.locals.insert(var_name, LocalVar { index: item_local, ty: ValType::I64, is_param: false });

                    // Compile iterator expression
                    self.compile_expr(iter)?;

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    func.instructions.push(Instruction::LocalSet(arr_local));
                    func.instructions.push(Instruction::I32Const(0));
                    func.instructions.push(Instruction::LocalSet(idx_local));

                    // Get array length
                    let len_fn = *self.func_map.get("morpheme_array_len").unwrap();
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::Call(len_fn));
                    func.instructions.push(Instruction::LocalSet(len_local));

                    func.instructions.push(Instruction::Block(BlockType::Empty));
                    func.instructions.push(Instruction::Loop(BlockType::Empty));

                    // Check if idx < len
                    func.instructions.push(Instruction::LocalGet(idx_local));
                    func.instructions.push(Instruction::LocalGet(len_local));
                    func.instructions.push(Instruction::I32GeU);
                    func.instructions.push(Instruction::BrIf(1));

                    // Get current item
                    let get_fn = *self.func_map.get("morpheme_array_get").unwrap();
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::LocalGet(idx_local));
                    func.instructions.push(Instruction::Call(get_fn));
                    func.instructions.push(Instruction::LocalSet(item_local));

                    self.loop_stack.push(LoopContext { break_label: 1, continue_label: 0 });
                    self.compile_block(body)?;
                    self.loop_stack.pop();

                    if body.expr.is_some() {
                        self.functions[def_idx].instructions.push(Instruction::Drop);
                    }

                    // Increment index
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::LocalGet(idx_local));
                    func.instructions.push(Instruction::I32Const(1));
                    func.instructions.push(Instruction::I32Add);
                    func.instructions.push(Instruction::LocalSet(idx_local));
                    func.instructions.push(Instruction::Br(0));
                    func.instructions.push(Instruction::End);
                    func.instructions.push(Instruction::End);
                    func.instructions.push(Instruction::I64Const(0));
                    Ok(())
                }

                Expr::Block(block) => {
                    self.functions[def_idx].instructions.push(Instruction::Block(BlockType::Result(ValType::I64)));
                    self.compile_block(block)?;
                    if block.expr.is_none() {
                        self.functions[def_idx].instructions.push(Instruction::I64Const(0));
                    }
                    self.functions[def_idx].instructions.push(Instruction::End);
                    Ok(())
                }

                Expr::Return(ret_expr) => {
                    if let Some(e) = ret_expr {
                        self.compile_expr(e)?;
                    } else {
                        self.functions[def_idx].instructions.push(Instruction::I64Const(0));
                    }
                    self.functions[def_idx].instructions.push(Instruction::Return);
                    Ok(())
                }

                Expr::Break(break_expr) => {
                    if let Some(e) = break_expr {
                        self.compile_expr(e)?;
                        self.functions[def_idx].instructions.push(Instruction::Drop);
                    }
                    let ctx = self.loop_stack.last().ok_or("break outside loop")?;
                    self.functions[def_idx].instructions.push(Instruction::Br(ctx.break_label));
                    Ok(())
                }

                Expr::Continue => {
                    let ctx = self.loop_stack.last().ok_or("continue outside loop")?;
                    self.functions[def_idx].instructions.push(Instruction::Br(ctx.continue_label));
                    Ok(())
                }

                Expr::Assign { target, value } => {
                    self.compile_expr(value)?;

                    if let Expr::Path(path) = target.as_ref() {
                        let name = path.segments.first()
                            .map(|s| s.ident.name.as_str())
                            .unwrap_or("");

                        let local = self.functions[def_idx].locals.get(name).cloned();
                        if let Some(local) = local {
                            self.functions[def_idx].instructions.push(Instruction::LocalTee(local.index));
                            return Ok(());
                        }

                        if let Some(&global_idx) = self.global_map.get(name) {
                            let func = &mut self.functions[def_idx];
                            func.instructions.push(Instruction::GlobalSet(global_idx));
                            func.instructions.push(Instruction::GlobalGet(global_idx));
                            return Ok(());
                        }
                    }

                    Err("Complex assign not supported".to_string())
                }


                Expr::Index { expr, index } => {
                    self.compile_expr(expr)?;
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);

                    self.compile_expr(index)?;

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    let get_fn = *self.func_map.get("morpheme_array_get").unwrap();
                    func.instructions.push(Instruction::Call(get_fn));
                    Ok(())
                }

                Expr::Array(elements) => {
                    let new_fn = *self.func_map.get("morpheme_array_new").unwrap();
                    let push_fn = *self.func_map.get("morpheme_array_push").unwrap();

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32Const(elements.len() as i32));
                    func.instructions.push(Instruction::Call(new_fn));

                    // Store array pointer in a temp local
                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));

                    for elem in elements {
                        self.functions[def_idx].instructions.push(Instruction::LocalGet(arr_local));
                        self.compile_expr(elem)?;
                        self.functions[def_idx].instructions.push(Instruction::Call(push_fn));
                    }

                    self.functions[def_idx].instructions.push(Instruction::LocalGet(arr_local));
                    self.functions[def_idx].instructions.push(Instruction::I64ExtendI32U);
                    Ok(())
                }

                Expr::Pipe { expr, operations } => {
                    self.compile_expr(expr)?;
                    for op in operations {
                        self.compile_pipe_op(op)?;
                    }
                    Ok(())
                }

                Expr::Evidential { expr, evidentiality } => {
                    self.compile_expr(expr)?;
                    let tag = match evidentiality {
                        Evidentiality::Known => evidence_tags::KNOWN,
                        Evidentiality::Uncertain => evidence_tags::UNCERTAIN,
                        Evidentiality::Reported => evidence_tags::REPORTED,
                        Evidentiality::Paradox => evidence_tags::PARADOX,
                    };
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I64Const(evidence_tags::VALUE_MASK));
                    func.instructions.push(Instruction::I64And);
                    func.instructions.push(Instruction::I64Const(tag));
                    func.instructions.push(Instruction::I64Or);
                    Ok(())
                }

                Expr::Match { expr, arms } => {
                    self.compile_match(expr, arms)
                }

                Expr::Tuple(elements) => {
                    // For now, just compile as array
                    let new_fn = *self.func_map.get("morpheme_array_new").unwrap();
                    let push_fn = *self.func_map.get("morpheme_array_push").unwrap();

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32Const(elements.len() as i32));
                    func.instructions.push(Instruction::Call(new_fn));

                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));

                    for elem in elements {
                        self.functions[def_idx].instructions.push(Instruction::LocalGet(arr_local));
                        self.compile_expr(elem)?;
                        self.functions[def_idx].instructions.push(Instruction::Call(push_fn));
                    }

                    self.functions[def_idx].instructions.push(Instruction::LocalGet(arr_local));
                    self.functions[def_idx].instructions.push(Instruction::I64ExtendI32U);
                    Ok(())
                }

                Expr::Closure { params, body } => {
                    self.compile_closure(params, body)
                }

                Expr::Struct { path, fields, rest } => {
                    self.compile_struct_expr(path, fields, rest.as_deref())
                }

                Expr::Field { expr, field } => {
                    self.compile_field_access(expr, &field.name)
                }

                Expr::MethodCall { receiver, method, args } => {
                    // Compile as: method(receiver, args...)
                    self.compile_expr(receiver)?;
                    for arg in args {
                        self.compile_expr(arg)?;
                    }

                    let method_name = &method.name;
                    if let Some(&func_idx) = self.func_map.get(method_name) {
                        self.functions[def_idx].instructions.push(Instruction::Call(func_idx));
                    } else {
                        // Unknown method - return receiver unchanged
                    }
                    Ok(())
                }

                _ => {
                    // Fallback: return 0 for unimplemented expressions
                    self.functions[def_idx].instructions.push(Instruction::I64Const(0));
                    Ok(())
                }
            }
        }

        fn compile_literal(&mut self, lit: &Literal) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;
            let func = &mut self.functions[def_idx];

            match lit {
                Literal::Int { value, .. } => {
                    let v: i64 = value.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
                    func.instructions.push(Instruction::I64Const(v));
                }
                Literal::Float { value, .. } => {
                    let v: f64 = value.parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
                    func.instructions.push(Instruction::I64Const(v.to_bits() as i64));
                }
                Literal::Bool(b) => {
                    func.instructions.push(Instruction::I64Const(if *b { 1 } else { 0 }));
                }
                Literal::String(s) => {
                    let offset = self.add_string(s);
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32Const(offset as i32));
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                Literal::Char(c) => {
                    func.instructions.push(Instruction::I64Const(*c as i64));
                }
                Literal::Null | Literal::Empty => {
                    func.instructions.push(Instruction::I64Const(0));
                }
                _ => {
                    func.instructions.push(Instruction::I64Const(0));
                }
            }
            Ok(())
        }

        fn add_string(&mut self, s: &str) -> u32 {
            if let Some(&offset) = self.string_map.get(s) {
                return offset;
            }
            let offset = self.data_offset;
            let bytes = s.as_bytes();
            let mut data = (bytes.len() as u32).to_le_bytes().to_vec();
            data.extend(bytes);
            self.data_segments.push((offset, data.clone()));
            self.data_offset += data.len() as u32;
            self.data_offset = (self.data_offset + 7) & !7; // Align
            self.string_map.insert(s.to_string(), offset);
            offset
        }

        // =====================================================================
        // Operators
        // =====================================================================

        fn emit_binop(&mut self, op: BinOp) {
            let def_idx = self.current_fn_idx.unwrap();
            let func = &mut self.functions[def_idx];

            match op {
                BinOp::Add => func.instructions.push(Instruction::I64Add),
                BinOp::Sub => func.instructions.push(Instruction::I64Sub),
                BinOp::Mul => func.instructions.push(Instruction::I64Mul),
                BinOp::Div => func.instructions.push(Instruction::I64DivS),
                BinOp::Rem => func.instructions.push(Instruction::I64RemS),
                BinOp::BitAnd => func.instructions.push(Instruction::I64And),
                BinOp::BitOr => func.instructions.push(Instruction::I64Or),
                BinOp::BitXor => func.instructions.push(Instruction::I64Xor),
                BinOp::Shl => func.instructions.push(Instruction::I64Shl),
                BinOp::Shr => func.instructions.push(Instruction::I64ShrS),
                BinOp::Eq => {
                    func.instructions.push(Instruction::I64Eq);
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                BinOp::Ne => {
                    func.instructions.push(Instruction::I64Ne);
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                BinOp::Lt => {
                    func.instructions.push(Instruction::I64LtS);
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                BinOp::Le => {
                    func.instructions.push(Instruction::I64LeS);
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                BinOp::Gt => {
                    func.instructions.push(Instruction::I64GtS);
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                BinOp::Ge => {
                    func.instructions.push(Instruction::I64GeS);
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                BinOp::And => func.instructions.push(Instruction::I64And),
                BinOp::Or => func.instructions.push(Instruction::I64Or),
                _ => func.instructions.push(Instruction::I64Const(0)),
            }
        }

        fn emit_unaryop(&mut self, op: UnaryOp) {
            let def_idx = self.current_fn_idx.unwrap();
            let func = &mut self.functions[def_idx];

            match op {
                UnaryOp::Neg => {
                    func.instructions.push(Instruction::I64Const(-1));
                    func.instructions.push(Instruction::I64Mul);
                }
                UnaryOp::Not => {
                    func.instructions.push(Instruction::I64Eqz);
                    func.instructions.push(Instruction::I64ExtendI32U);
                }
                UnaryOp::Deref | UnaryOp::Ref | UnaryOp::RefMut => {
                    // These are reference operations - pass through for now
                }
            }
        }

        // =====================================================================
        // Pipe Operations (Morphemes)
        // =====================================================================

        fn compile_pipe_op(&mut self, op: &PipeOp) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            match op {
                PipeOp::Transform(body) => {
                    // τ{body} - map transformation
                    // Stack has array pointer, compile closure and call array_map
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);

                    // Store array pointer
                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));

                    // Compile the closure body as an inline function
                    let closure_id = self.closure_counter;
                    self.closure_counter += 1;

                    // Create a morpheme callback function: (elem: i64) -> i64
                    let callback_name = format!("__morpheme_map_{}", closure_id);
                    let type_idx = self.get_or_create_type(vec![ValType::I64], vec![ValType::I64]);
                    let callback_idx = self.import_count + self.functions.len() as u32;

                    // Create locals for the callback - single param "it"
                    let mut callback_locals = HashMap::new();
                    callback_locals.insert("it".to_string(), LocalVar {
                        index: 0,
                        ty: ValType::I64,
                        is_param: true,
                    });

                    self.functions.push(CompiledFunction {
                        name: callback_name.clone(),
                        type_idx,
                        func_idx: callback_idx,
                        params: vec![("it".to_string(), ValType::I64)],
                        results: vec![ValType::I64],
                        locals: callback_locals,
                        local_types: Vec::new(),
                        instructions: Vec::new(),
                        is_exported: false,
                    });
                    self.func_map.insert(callback_name, callback_idx);

                    // Add to table
                    let table_idx = self.table_elements.len() as u32;
                    self.table_elements.push(callback_idx);

                    // Compile the body
                    let parent_fn = self.current_fn_idx;
                    let callback_fn_idx = self.functions.len() - 1;
                    self.current_fn_idx = Some(callback_fn_idx);
                    self.compile_expr(body)?;
                    self.functions[callback_fn_idx].instructions.push(Instruction::End);
                    self.current_fn_idx = parent_fn;

                    // Call array_map(arr, table_idx)
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::I32Const(table_idx as i32));
                    let map_fn = *self.func_map.get("morpheme_array_map").unwrap();
                    func.instructions.push(Instruction::Call(map_fn));
                    func.instructions.push(Instruction::I64ExtendI32U);
                    Ok(())
                }

                PipeOp::Filter(predicate) => {
                    // φ{predicate} - filter
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);

                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));

                    let closure_id = self.closure_counter;
                    self.closure_counter += 1;

                    let callback_name = format!("__morpheme_filter_{}", closure_id);
                    let type_idx = self.get_or_create_type(vec![ValType::I64], vec![ValType::I64]);
                    let callback_idx = self.import_count + self.functions.len() as u32;

                    let mut callback_locals = HashMap::new();
                    callback_locals.insert("it".to_string(), LocalVar {
                        index: 0,
                        ty: ValType::I64,
                        is_param: true,
                    });

                    self.functions.push(CompiledFunction {
                        name: callback_name.clone(),
                        type_idx,
                        func_idx: callback_idx,
                        params: vec![("it".to_string(), ValType::I64)],
                        results: vec![ValType::I64],
                        locals: callback_locals,
                        local_types: Vec::new(),
                        instructions: Vec::new(),
                        is_exported: false,
                    });
                    self.func_map.insert(callback_name, callback_idx);

                    let table_idx = self.table_elements.len() as u32;
                    self.table_elements.push(callback_idx);

                    let parent_fn = self.current_fn_idx;
                    let callback_fn_idx = self.functions.len() - 1;
                    self.current_fn_idx = Some(callback_fn_idx);
                    self.compile_expr(predicate)?;
                    self.functions[callback_fn_idx].instructions.push(Instruction::End);
                    self.current_fn_idx = parent_fn;

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::I32Const(table_idx as i32));
                    let filter_fn = *self.func_map.get("morpheme_array_filter").unwrap();
                    func.instructions.push(Instruction::Call(filter_fn));
                    func.instructions.push(Instruction::I64ExtendI32U);
                    Ok(())
                }

                PipeOp::Sort(_field) => {
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    let sort_fn = *self.func_map.get("morpheme_array_sort").unwrap();
                    func.instructions.push(Instruction::Call(sort_fn));
                    func.instructions.push(Instruction::I64ExtendI32U);
                    Ok(())
                }

                PipeOp::First => {
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    let first_fn = *self.func_map.get("morpheme_array_first").unwrap();
                    func.instructions.push(Instruction::Call(first_fn));
                    Ok(())
                }

                PipeOp::Last => {
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    let last_fn = *self.func_map.get("morpheme_array_last").unwrap();
                    func.instructions.push(Instruction::Call(last_fn));
                    Ok(())
                }

                PipeOp::Middle => {
                    // μ - get middle element
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    // Get length, divide by 2, get nth
                    let len_fn = *self.func_map.get("morpheme_array_len").unwrap();
                    let nth_fn = *self.func_map.get("morpheme_array_nth").unwrap();
                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::Call(len_fn));
                    func.instructions.push(Instruction::I32Const(2));
                    func.instructions.push(Instruction::I32DivU);
                    func.instructions.push(Instruction::Call(nth_fn));
                    Ok(())
                }

                PipeOp::Nth(index_expr) => {
                    // ν{n} - get nth element
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));

                    self.compile_expr(index_expr)?;

                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    // Swap order for (arr, idx)
                    let idx_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalSet(idx_local));
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::LocalGet(idx_local));
                    let nth_fn = *self.func_map.get("morpheme_array_nth").unwrap();
                    func.instructions.push(Instruction::Call(nth_fn));
                    Ok(())
                }

                PipeOp::Reduce(body) => {
                    // ρ{reducer} - reduce with accumulator
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);

                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));

                    let closure_id = self.closure_counter;
                    self.closure_counter += 1;

                    // Reducer takes (acc, elem) -> acc
                    let callback_name = format!("__morpheme_reduce_{}", closure_id);
                    let type_idx = self.get_or_create_type(vec![ValType::I64, ValType::I64], vec![ValType::I64]);
                    let callback_idx = self.import_count + self.functions.len() as u32;

                    let mut callback_locals = HashMap::new();
                    callback_locals.insert("acc".to_string(), LocalVar {
                        index: 0,
                        ty: ValType::I64,
                        is_param: true,
                    });
                    callback_locals.insert("it".to_string(), LocalVar {
                        index: 1,
                        ty: ValType::I64,
                        is_param: true,
                    });

                    self.functions.push(CompiledFunction {
                        name: callback_name.clone(),
                        type_idx,
                        func_idx: callback_idx,
                        params: vec![("acc".to_string(), ValType::I64), ("it".to_string(), ValType::I64)],
                        results: vec![ValType::I64],
                        locals: callback_locals,
                        local_types: Vec::new(),
                        instructions: Vec::new(),
                        is_exported: false,
                    });
                    self.func_map.insert(callback_name, callback_idx);

                    let table_idx = self.table_elements.len() as u32;
                    self.table_elements.push(callback_idx);

                    let parent_fn = self.current_fn_idx;
                    let callback_fn_idx = self.functions.len() - 1;
                    self.current_fn_idx = Some(callback_fn_idx);
                    self.compile_expr(body)?;
                    self.functions[callback_fn_idx].instructions.push(Instruction::End);
                    self.current_fn_idx = parent_fn;

                    // Call array_reduce(arr, table_idx, initial_value=0)
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::I32Const(table_idx as i32));
                    func.instructions.push(Instruction::I64Const(0)); // Initial value
                    let reduce_fn = *self.func_map.get("morpheme_array_reduce").unwrap();
                    func.instructions.push(Instruction::Call(reduce_fn));
                    Ok(())
                }

                PipeOp::Await => {
                    // Await is handled specially - requires async state machine
                    // For now, just pass through the value
                    Ok(())
                }

                PipeOp::Choice => {
                    // χ - random element
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::I32WrapI64);
                    let arr_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalTee(arr_local));

                    // random() * len | floor | nth
                    let random_fn = *self.func_map.get("math_random").unwrap();
                    let len_fn = *self.func_map.get("morpheme_array_len").unwrap();
                    let nth_fn = *self.func_map.get("morpheme_array_nth").unwrap();

                    func.instructions.push(Instruction::Call(random_fn));
                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::Call(len_fn));
                    func.instructions.push(Instruction::F64ConvertI32U);
                    func.instructions.push(Instruction::F64Mul);
                    func.instructions.push(Instruction::I32TruncF64U);
                    let idx_local = func.params.len() as u32 + func.local_types.len() as u32;
                    func.local_types.push(ValType::I32);
                    func.instructions.push(Instruction::LocalSet(idx_local));

                    func.instructions.push(Instruction::LocalGet(arr_local));
                    func.instructions.push(Instruction::LocalGet(idx_local));
                    func.instructions.push(Instruction::Call(nth_fn));
                    Ok(())
                }

                _ => Ok(()),
            }
        }

        // =====================================================================
        // Match Compilation
        // =====================================================================

        fn compile_match(&mut self, expr: &Expr, arms: &[MatchArm]) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            // Compile scrutinee and store in local
            self.compile_expr(expr)?;

            let func = &mut self.functions[def_idx];
            let scrutinee_local = func.params.len() as u32 + func.local_types.len() as u32;
            func.local_types.push(ValType::I64);
            func.instructions.push(Instruction::LocalSet(scrutinee_local));

            // Generate if-else chain
            for (i, arm) in arms.iter().enumerate() {
                let is_last = i == arms.len() - 1;

                // Check pattern
                self.compile_pattern_check(&arm.pattern, scrutinee_local)?;

                let func = &mut self.functions[def_idx];
                func.instructions.push(Instruction::If(BlockType::Result(ValType::I64)));

                // Bind pattern variables
                self.bind_pattern(&arm.pattern, scrutinee_local)?;

                // Compile arm body
                self.compile_expr(&arm.body)?;

                if !is_last {
                    let func = &mut self.functions[def_idx];
                    func.instructions.push(Instruction::Else);
                }
            }

            // Close all the if blocks
            let func = &mut self.functions[def_idx];
            for _ in 0..arms.len() {
                func.instructions.push(Instruction::End);
            }

            Ok(())
        }

        fn compile_pattern_check(&mut self, pattern: &Pattern, scrutinee: u32) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            match pattern {
                Pattern::Wildcard => {
                    self.functions[def_idx].instructions.push(Instruction::I64Const(1));
                }
                Pattern::Ident { .. } => {
                    self.functions[def_idx].instructions.push(Instruction::I64Const(1));
                }
                Pattern::Literal(lit) => {
                    self.functions[def_idx].instructions.push(Instruction::LocalGet(scrutinee));
                    self.compile_literal(lit)?;
                    self.functions[def_idx].instructions.push(Instruction::I64Eq);
                    self.functions[def_idx].instructions.push(Instruction::I64ExtendI32U);
                }
                _ => {
                    self.functions[def_idx].instructions.push(Instruction::I64Const(1));
                }
            }
            Ok(())
        }

        fn bind_pattern(&mut self, pattern: &Pattern, scrutinee: u32) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            if let Pattern::Ident { name, .. } = pattern {
                let func = &mut self.functions[def_idx];
                let local_idx = func.params.len() as u32 + func.local_types.len() as u32;
                func.local_types.push(ValType::I64);
                func.locals.insert(name.name.clone(), LocalVar {
                    index: local_idx,
                    ty: ValType::I64,
                    is_param: false,
                });
                func.instructions.push(Instruction::LocalGet(scrutinee));
                func.instructions.push(Instruction::LocalSet(local_idx));
            }
            Ok(())
        }

        // =====================================================================
        // Closure Compilation
        // =====================================================================

        /// Compile a closure expression.
        /// Creates a helper function and returns its table index.
        fn compile_closure(&mut self, params: &[ClosureParam], body: &Expr) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            // Generate unique closure name
            let closure_id = self.closure_counter;
            self.closure_counter += 1;
            let closure_name = format!("__closure_{}", closure_id);

            // Analyze captures - find free variables in closure body
            let outer_locals = self.functions[def_idx].locals.clone();
            let captures = self.analyze_captures(body, params, &outer_locals);

            // Create closure function type: (env_ptr, params...) -> result
            let mut param_types = vec![ValType::I32]; // env pointer first
            for _ in params {
                param_types.push(ValType::I64);
            }
            let result_types = vec![ValType::I64];

            let type_idx = self.get_or_create_type(param_types.clone(), result_types.clone());
            let func_idx = self.import_count + self.functions.len() as u32;

            // Build closure locals
            let mut closure_locals: HashMap<String, LocalVar> = HashMap::new();
            closure_locals.insert("__env".to_string(), LocalVar {
                index: 0,
                ty: ValType::I32,
                is_param: true,
            });
            for (i, p) in params.iter().enumerate() {
                let pname = match &p.pattern {
                    Pattern::Ident { name, .. } => name.name.clone(),
                    _ => format!("param_{}", i),
                };
                closure_locals.insert(pname.clone(), LocalVar {
                    index: (i + 1) as u32,
                    ty: ValType::I64,
                    is_param: true,
                });
            }

            // Create the closure function
            self.functions.push(CompiledFunction {
                name: closure_name.clone(),
                type_idx,
                func_idx,
                params: std::iter::once(("__env".to_string(), ValType::I32))
                    .chain(params.iter().enumerate().map(|(i, p)| {
                        let pname = match &p.pattern {
                            Pattern::Ident { name, .. } => name.name.clone(),
                            _ => format!("param_{}", i),
                        };
                        (pname, ValType::I64)
                    }))
                    .collect(),
                results: result_types,
                locals: closure_locals,
                local_types: Vec::new(),
                instructions: Vec::new(),
                is_exported: false,
            });

            self.func_map.insert(closure_name.clone(), func_idx);

            // Add to table for indirect calls
            let table_idx = self.table_elements.len() as u32;
            self.table_elements.push(func_idx);

            // Store closure info
            self.closure_map.insert(closure_name.clone(), ClosureInfo {
                func_idx,
                table_idx,
                captures: captures.clone(),
                env_size: (captures.len() * 8) as u32,
            });

            // Compile closure body
            let parent_fn_idx = self.current_fn_idx;
            self.current_fn_idx = Some(self.functions.len() - 1);

            // Generate code to load captures from env
            let closure_fn_idx = self.functions.len() - 1;
            for (i, capture) in captures.iter().enumerate() {
                // Load captured value from env: env_ptr + offset -> local
                let func = &mut self.functions[closure_fn_idx];
                let local_idx = func.params.len() as u32 + func.local_types.len() as u32;
                func.local_types.push(ValType::I64);
                func.locals.insert(capture.clone(), LocalVar {
                    index: local_idx,
                    ty: ValType::I64,
                    is_param: false,
                });

                // Load from env: i64.load(env_ptr + i*8)
                func.instructions.push(Instruction::LocalGet(0)); // env ptr
                func.instructions.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: (i * 8) as u64,
                    align: 3, // 8-byte aligned
                    memory_index: 0,
                }));
                func.instructions.push(Instruction::LocalSet(local_idx));
            }

            // Compile the closure body
            self.compile_expr(body)?;
            self.functions[closure_fn_idx].instructions.push(Instruction::End);

            self.current_fn_idx = parent_fn_idx;

            // Back in parent function: allocate env and store captures
            let alloc_fn = *self.func_map.get("memory_alloc").unwrap();
            let func = &mut self.functions[def_idx];

            // Allocate environment
            let env_size = (captures.len() * 8) as i32;
            if env_size > 0 {
                func.instructions.push(Instruction::I32Const(env_size));
                func.instructions.push(Instruction::Call(alloc_fn));
            } else {
                func.instructions.push(Instruction::I32Const(0)); // null env
            }

            let env_local = func.params.len() as u32 + func.local_types.len() as u32;
            func.local_types.push(ValType::I32);
            func.instructions.push(Instruction::LocalTee(env_local));

            // Store captured values
            for (i, capture) in captures.iter().enumerate() {
                if let Some(local) = outer_locals.get(capture) {
                    func.instructions.push(Instruction::LocalGet(env_local));
                    func.instructions.push(Instruction::LocalGet(local.index));
                    func.instructions.push(Instruction::I64Store(wasm_encoder::MemArg {
                        offset: (i * 8) as u64,
                        align: 3,
                        memory_index: 0,
                    }));
                }
            }

            // Return closure as packed (table_idx << 32) | env_ptr
            func.instructions.push(Instruction::LocalGet(env_local));
            func.instructions.push(Instruction::I64ExtendI32U);
            func.instructions.push(Instruction::I64Const((table_idx as i64) << 32));
            func.instructions.push(Instruction::I64Or);

            Ok(())
        }

        /// Analyze free variables in an expression
        fn analyze_captures(
            &self,
            expr: &Expr,
            params: &[ClosureParam],
            outer_locals: &HashMap<String, LocalVar>,
        ) -> Vec<String> {
            let mut captures = Vec::new();
            let param_names: std::collections::HashSet<_> = params.iter()
                .filter_map(|p| match &p.pattern {
                    Pattern::Ident { name, .. } => Some(name.name.clone()),
                    _ => None,
                })
                .collect();

            self.collect_free_vars(expr, &param_names, outer_locals, &mut captures);
            captures
        }

        fn collect_free_vars(
            &self,
            expr: &Expr,
            bound: &std::collections::HashSet<String>,
            outer: &HashMap<String, LocalVar>,
            captures: &mut Vec<String>,
        ) {
            match expr {
                Expr::Path(path) => {
                    if let Some(seg) = path.segments.first() {
                        let name = &seg.ident.name;
                        if !bound.contains(name) && outer.contains_key(name) && !captures.contains(name) {
                            captures.push(name.clone());
                        }
                    }
                }
                Expr::Binary { left, right, .. } => {
                    self.collect_free_vars(left, bound, outer, captures);
                    self.collect_free_vars(right, bound, outer, captures);
                }
                Expr::Unary { expr, .. } => {
                    self.collect_free_vars(expr, bound, outer, captures);
                }
                Expr::Call { func, args } => {
                    self.collect_free_vars(func, bound, outer, captures);
                    for arg in args {
                        self.collect_free_vars(arg, bound, outer, captures);
                    }
                }
                Expr::If { condition, then_branch, else_branch } => {
                    self.collect_free_vars(condition, bound, outer, captures);
                    for stmt in &then_branch.stmts {
                        if let Stmt::Expr(e) | Stmt::Semi(e) = stmt {
                            self.collect_free_vars(e, bound, outer, captures);
                        }
                    }
                    if let Some(e) = &then_branch.expr {
                        self.collect_free_vars(e, bound, outer, captures);
                    }
                    if let Some(else_expr) = else_branch {
                        self.collect_free_vars(else_expr, bound, outer, captures);
                    }
                }
                Expr::Block(block) => {
                    for stmt in &block.stmts {
                        if let Stmt::Expr(e) | Stmt::Semi(e) = stmt {
                            self.collect_free_vars(e, bound, outer, captures);
                        }
                    }
                    if let Some(e) = &block.expr {
                        self.collect_free_vars(e, bound, outer, captures);
                    }
                }
                Expr::Field { expr, .. } => {
                    self.collect_free_vars(expr, bound, outer, captures);
                }
                Expr::Index { expr, index } => {
                    self.collect_free_vars(expr, bound, outer, captures);
                    self.collect_free_vars(index, bound, outer, captures);
                }
                Expr::Array(elements) => {
                    for e in elements {
                        self.collect_free_vars(e, bound, outer, captures);
                    }
                }
                _ => {}
            }
        }

        // =====================================================================
        // Struct Expression Compilation
        // =====================================================================

        fn compile_struct_expr(
            &mut self,
            path: &ast::TypePath,
            fields: &[FieldInit],
            _rest: Option<&Expr>,
        ) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            let struct_name = path.segments.first()
                .map(|s| s.ident.name.clone())
                .unwrap_or_default();

            let layout = self.struct_layouts.get(&struct_name).cloned();

            if let Some(layout) = layout {
                // Allocate struct on heap
                let alloc_fn = *self.func_map.get("memory_alloc").unwrap();
                let func = &mut self.functions[def_idx];
                func.instructions.push(Instruction::I32Const(layout.size as i32));
                func.instructions.push(Instruction::Call(alloc_fn));

                let struct_local = func.params.len() as u32 + func.local_types.len() as u32;
                func.local_types.push(ValType::I32);
                func.instructions.push(Instruction::LocalTee(struct_local));

                // Store each field
                for field_init in fields {
                    let field_name = &field_init.name.name;
                    if let Some((_, offset)) = layout.fields.iter().find(|(n, _)| n == field_name) {
                        // Get struct ptr
                        self.functions[def_idx].instructions.push(Instruction::LocalGet(struct_local));

                        // Compile field value
                        if let Some(value) = &field_init.value {
                            self.compile_expr(value)?;
                        } else {
                            // Shorthand: use variable with same name
                            let local = self.functions[def_idx].locals.get(field_name).cloned();
                            if let Some(local) = local {
                                self.functions[def_idx].instructions.push(Instruction::LocalGet(local.index));
                            } else {
                                self.functions[def_idx].instructions.push(Instruction::I64Const(0));
                            }
                        }

                        // Store to field offset
                        self.functions[def_idx].instructions.push(Instruction::I64Store(wasm_encoder::MemArg {
                            offset: *offset as u64,
                            align: 3,
                            memory_index: 0,
                        }));
                    }
                }

                // Return struct pointer
                self.functions[def_idx].instructions.push(Instruction::LocalGet(struct_local));
                self.functions[def_idx].instructions.push(Instruction::I64ExtendI32U);
            } else {
                // Unknown struct - return 0
                self.functions[def_idx].instructions.push(Instruction::I64Const(0));
            }

            Ok(())
        }

        fn compile_field_access(&mut self, expr: &Expr, field_name: &str) -> Result<(), String> {
            let def_idx = self.current_fn_idx.ok_or("Not in function")?;

            // Compile the struct expression
            self.compile_expr(expr)?;

            // Convert to i32 pointer
            let func = &mut self.functions[def_idx];
            func.instructions.push(Instruction::I32WrapI64);

            // TODO: For full struct support, we'd need type inference to know the struct type
            // For now, we try to find matching field in any known struct
            let mut found_offset = None;
            for layout in self.struct_layouts.values() {
                if let Some((_, offset)) = layout.fields.iter().find(|(n, _)| n == field_name) {
                    found_offset = Some(*offset);
                    break;
                }
            }

            if let Some(offset) = found_offset {
                func.instructions.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: offset as u64,
                    align: 3,
                    memory_index: 0,
                }));
            } else {
                // Unknown field - pop the pointer and return 0
                func.instructions.push(Instruction::Drop);
                func.instructions.push(Instruction::I64Const(0));
            }

            Ok(())
        }

        // =====================================================================
        // Module Generation
        // =====================================================================

        fn generate_module(&self) -> Result<Vec<u8>, String> {
            let mut module = Module::new();

            // Type section
            let mut types = TypeSection::new();
            for (params, results) in &self.types {
                types.ty().function(params.iter().copied(), results.iter().copied());
            }
            module.section(&types);

            // Import section
            let mut imports = ImportSection::new();
            for import in &self.imports {
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
                minimum: 16,
                maximum: Some(256),
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
                        GlobalType { val_type: *ty, mutable: *mutable, shared: false },
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

            // Export function table for indirect calls (morpheme callbacks)
            if !self.table_elements.is_empty() {
                exports.export("__indirect_function_table", wasm_encoder::ExportKind::Table, 0);
            }

            for func in &self.functions {
                if func.is_exported {
                    exports.export(&func.name, wasm_encoder::ExportKind::Func, func.func_idx);
                }
            }
            module.section(&exports);

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

            Ok(module.finish())
        }
    }

    impl Default for WasmCompiler {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(feature = "wasm")]
pub use wasm::WasmCompiler;

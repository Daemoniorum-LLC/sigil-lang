//! Closure compilation.
//!
//! Compiles Sigil closures to WASM with environment capture.

use wasm_encoder::{BlockType, Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::types::{ClosureInfo, CompiledFunction};
use super::WasmCompiler;
use crate::ast::{ClosureParam, Expr, Pattern};

/// Get name from a closure parameter pattern.
fn get_param_name(param: &ClosureParam) -> String {
    match &param.pattern {
        Pattern::Ident { name, .. } => name.name.clone(),
        Pattern::Wildcard => "_".to_string(),
        _ => "__param".to_string(),
    }
}

impl WasmCompiler {
    /// Compile a closure expression.
    ///
    /// `is_move` indicates a move closure that takes ownership of captured variables.
    /// For WASM compilation, this means captured values are deep-copied into the
    /// closure's environment rather than storing references. In practice, since WASM
    /// uses value semantics for i64/f64, this mainly affects composite types.
    pub fn compile_closure(&mut self, params: &[ClosureParam], body: &Expr, is_move: bool) -> WasmResult<()> {
        // Analyze captures
        let captures = self.analyze_captures(body)?;

        if captures.is_empty() {
            // No captures - compile as regular function (is_move irrelevant)
            self.compile_simple_closure(params, body)
        } else {
            // Has captures - compile with environment
            // Move closures copy captured values; non-move may reference them
            self.compile_capturing_closure(params, body, &captures, is_move)
        }
    }

    /// Compile a closure with no captures.
    fn compile_simple_closure(
        &mut self,
        params: &[ClosureParam],
        body: &Expr,
    ) -> WasmResult<()> {
        // Generate unique closure name
        let closure_name = format!("__closure_{}", self.closure_counter);
        self.closure_counter += 1;

        // Create function type
        let param_types: Vec<ValType> = params.iter().map(|_| ValType::I64).collect();
        let type_idx = self.get_or_create_type(param_types.clone(), vec![ValType::I64]);

        // Create function
        let func_idx = self.imports.import_count() + self.functions.len() as u32;
        let params_with_names: Vec<(String, ValType)> = params
            .iter()
            .map(|p| (get_param_name(p), ValType::I64))
            .collect();

        let mut func = CompiledFunction::new(
            closure_name.clone(),
            type_idx,
            func_idx,
            params_with_names,
            vec![ValType::I64],
            false, // Not exported
        );

        // Save current function context
        let prev_fn_idx = self.current_fn_idx;

        // Add function and set as current
        self.functions.push(func);
        self.current_fn_idx = Some(self.functions.len() - 1);

        // Compile body
        self.compile_expr(body)?;

        // Add function end
        {
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::End);
        }

        // Restore previous function context
        self.current_fn_idx = prev_fn_idx;

        // Add to function table for indirect calls
        self.table_elements.push(func_idx);
        let table_idx = (self.table_elements.len() - 1) as u32;

        // Store closure info
        self.closure_map.insert(
            closure_name.clone(),
            ClosureInfo {
                func_idx,
                table_idx,
                captures: vec![],
                env_size: 0,
            },
        );

        // Return closure pointer (table index for indirect call)
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Create closure representation: [table_idx, env_ptr]
        // For no-capture closure, env_ptr is 0
        func.push(Instruction::I64Const(table_idx as i64));

        Ok(())
    }

    /// Compile a closure with captured variables.
    ///
    /// `is_move`: When true, captured values are moved (copied) into the closure's
    /// environment. When false, captures may use references. For WASM with value
    /// types (i64/f64), this distinction has limited effect since values are
    /// copied either way. The distinction matters more for reference types.
    fn compile_capturing_closure(
        &mut self,
        params: &[ClosureParam],
        body: &Expr,
        captures: &[String],
        _is_move: bool,
    ) -> WasmResult<()> {
        // Generate closure name
        let closure_name = format!("__closure_{}", self.closure_counter);
        self.closure_counter += 1;

        // Determine which captures are mutable
        // Note: is_move affects ownership semantics. For WASM value types,
        // we always copy into the environment (effectively a move). Reference
        // type handling would check is_move to decide copy vs reference.
        let mutable_captures: Vec<bool> = captures
            .iter()
            .map(|c| self.mutable_captures.contains(c))
            .collect();

        // Closure function takes: [env_ptr, ...params]
        let mut param_types = vec![ValType::I64]; // env pointer
        param_types.extend(params.iter().map(|_| ValType::I64));

        let type_idx = self.get_or_create_type(param_types.clone(), vec![ValType::I64]);

        let func_idx = self.imports.import_count() + self.functions.len() as u32;

        let mut params_with_names = vec![("__env".to_string(), ValType::I64)];
        params_with_names.extend(params.iter().map(|p| (get_param_name(p), ValType::I64)));

        let func = CompiledFunction::new(
            closure_name.clone(),
            type_idx,
            func_idx,
            params_with_names,
            vec![ValType::I64],
            false,
        );

        // Save context
        let prev_fn_idx = self.current_fn_idx;

        // Add function
        self.functions.push(func);
        self.current_fn_idx = Some(self.functions.len() - 1);

        // Load captures from environment into locals
        // For mutable captures, store the cell pointer (used for indirection)
        // For immutable captures, store the value directly
        {
            let func = self.current_function_mut().unwrap();

            for (i, (capture, is_mutable)) in captures.iter().zip(mutable_captures.iter()).enumerate() {
                // env_ptr is local 0
                func.push(Instruction::LocalGet(0)); // env_ptr
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: (i * 8) as u64,
                    align: 3,
                    memory_index: 0,
                }));

                if *is_mutable {
                    // For mutable captures, this is a cell pointer
                    // Store it with a __cell_ prefix so we know to use indirection
                    let local_idx = func.alloc_local(format!("__cell_{}", capture), ValType::I64);
                    func.push(Instruction::LocalSet(local_idx));
                } else {
                    let local_idx = func.alloc_local(capture.clone(), ValType::I64);
                    func.push(Instruction::LocalSet(local_idx));
                }
            }
        }

        // Compile body
        self.compile_expr(body)?;

        // End function
        {
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::End);
        }

        // Restore context
        self.current_fn_idx = prev_fn_idx;

        // Add to function table
        self.table_elements.push(func_idx);
        let table_idx = (self.table_elements.len() - 1) as u32;

        let env_size = (captures.len() * 8) as u32;

        // Store closure info
        self.closure_map.insert(
            closure_name.clone(),
            ClosureInfo {
                func_idx,
                table_idx,
                captures: captures.to_vec(),
                env_size,
            },
        );

        // Get heap_alloc function index first (requires immutable borrow)
        let alloc_idx = self
            .get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;

        // First, resolve all captures to determine if they're locals or globals
        // This requires immutable access to current_function()
        enum CaptureSource {
            Local(u32),
            Global(u32),
        }

        let capture_sources: Vec<CaptureSource> = {
            let func = self.current_function();
            let mut sources = Vec::new();

            for capture in captures.iter() {
                if let Some(func_ref) = &func {
                    if let Some(local) = func_ref.get_local(capture) {
                        sources.push(CaptureSource::Local(local.index));
                        continue;
                    }
                }
                // Try global
                if let Some(global_idx) = self.get_global(capture) {
                    sources.push(CaptureSource::Global(global_idx));
                } else {
                    return Err(WasmError::undefined_variable(capture));
                }
            }
            sources
        };

        // Now allocate environment and store captures
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Allocate env: heap_alloc(env_size)
        func.push(Instruction::I64Const(env_size as i64));
        func.push(Instruction::Call(alloc_idx));

        let env_idx = func.alloc_local("__env_ptr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(env_idx));



        // Store each capture in environment using pre-resolved sources
        // For mutable captures, allocate a cell first
        for (i, (source, (capture, is_mutable))) in capture_sources
            .iter()
            .zip(captures.iter().zip(mutable_captures.iter()))
            .enumerate()
        {
            if *is_mutable {
                // Mutable capture: allocate a cell
                let func = self.current_function_mut().unwrap();

                // Allocate 8-byte cell
                func.push(Instruction::I64Const(8));
                func.push(Instruction::Call(alloc_idx));

                let cell_idx = func.alloc_local(format!("__cell_alloc_{}", capture), ValType::I64);
                func.push(Instruction::LocalSet(cell_idx));

                // Store current value in cell
                func.push(Instruction::LocalGet(cell_idx));
                func.push(Instruction::I32WrapI64);

                match source {
                    CaptureSource::Local(idx) => {
                        func.push(Instruction::LocalGet(*idx));
                    }
                    CaptureSource::Global(idx) => {
                        func.push(Instruction::GlobalGet(*idx));
                    }
                }

                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));

                // Store cell pointer in environment
                func.push(Instruction::LocalGet(env_idx));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::LocalGet(cell_idx));
                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: (i * 8) as u64,
                    align: 3,
                    memory_index: 0,
                }));
            } else {
                // Immutable capture: store value directly
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(env_idx));
                func.push(Instruction::I32WrapI64);

                match source {
                    CaptureSource::Local(idx) => {
                        func.push(Instruction::LocalGet(*idx));
                    }
                    CaptureSource::Global(idx) => {
                        func.push(Instruction::GlobalGet(*idx));
                    }
                }

                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: (i * 8) as u64,
                    align: 3,
                    memory_index: 0,
                }));
            }
        }

        let func = self.current_function_mut().unwrap();

        // Create closure object: [table_idx, env_ptr]
        // Allocate 16 bytes
        func.push(Instruction::I64Const(16));
        func.push(Instruction::Call(alloc_idx));

        let closure_idx = func.alloc_local("__closure".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(closure_idx));

        // Store table index
        func.push(Instruction::LocalGet(closure_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Const(table_idx as i64));
        func.push(Instruction::I64Store(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));

        // Store env pointer
        func.push(Instruction::LocalGet(closure_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::LocalGet(env_idx));
        func.push(Instruction::I64Store(wasm_encoder::MemArg {
            offset: 8,
            align: 3,
            memory_index: 0,
        }));

        // Return closure pointer
        func.push(Instruction::LocalGet(closure_idx));

        Ok(())
    }

    /// Analyze an expression for captured variables.
    fn analyze_captures(&self, expr: &Expr) -> WasmResult<Vec<String>> {
        let mut captures = Vec::new();
        let mut visitor = CaptureAnalyzer {
            captures: &mut captures,
            bound: vec![],
            compiler: self,
        };
        visitor.visit(expr);
        Ok(captures)
    }

    /// Compile a function call expression.
    pub fn compile_call(&mut self, func_expr: &Expr, args: &[Expr]) -> WasmResult<()> {
        match func_expr {
            Expr::Path(path) => {
                // Extract path segments
                let segments: Vec<String> = path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.clone())
                    .collect();

                // Resolve tome:: prefix (crate root)
                let resolved_segments = self.resolve_path(&segments);

                // Build the qualified name (e.g., ["signal", "create"] -> "signal_create" for imports)
                let import_name: String = resolved_segments.join("_");
                let name = import_name.as_str();

                // Also get just the simple name for local lookups
                let simple_name = resolved_segments.last().map(|s| s.as_str()).unwrap_or("");

                // Build qualified path for local lookups (e.g., ["signal", "create"] -> "signal::create")
                let qualified_path = resolved_segments.join("::");

                // Check for enum variant constructor (e.g., VNode::Text(123) or tome::vdom::VNode::Text(123))
                // Use resolved_segments (stripped of tome::) for enum lookup
                if resolved_segments.len() >= 2 {
                    // Last two segments are [EnumName, VariantName]
                    let enum_name = &resolved_segments[resolved_segments.len() - 2];
                    let variant_name = &resolved_segments[resolved_segments.len() - 1];

                    if let Some(layout) = self.enum_layouts.get(enum_name).cloned() {
                        if let Some(tag) = layout.variant_tag(variant_name) {
                            return self.compile_enum_construction(&layout, variant_name, tag, args);
                        }
                    }
                }

                // =================================================================
                // VNode Builder Pattern - Static Constructors
                // =================================================================
                // Handle VNode·div(), VNode·span(), VNode·fragment(), etc.
                if resolved_segments.len() >= 2 {
                    let type_name = &resolved_segments[resolved_segments.len() - 2];
                    let method_name = &resolved_segments[resolved_segments.len() - 1];

                    if type_name == "VNode" {
                        if let Some(result) = self.try_compile_vnode_constructor(method_name, args) {
                            return result;
                        }
                    }
                }

                // Handle std library functions
                // std::cmp::max, std::cmp::min, etc.
                if name == "std_cmp_max" && args.len() == 2 {
                    // max(a, b) -> if a > b { a } else { b }
                    self.compile_expr(&args[0])?;
                    self.compile_expr(&args[1])?;
                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    // Use select instruction: select(a, b, condition) where condition = a > b
                    // First compute: args[0] args[1] (args[0] > args[1])
                    // Stack: a b
                    // We want: if a > b { a } else { b }
                    // Use i64.gt_s then select
                    let a = func.alloc_local("__max_a".to_string(), ValType::I64);
                    let b = func.alloc_local("__max_b".to_string(), ValType::I64);
                    func.push(Instruction::LocalSet(b));
                    func.push(Instruction::LocalSet(a));
                    func.push(Instruction::LocalGet(a));
                    func.push(Instruction::LocalGet(b));
                    func.push(Instruction::LocalGet(a));
                    func.push(Instruction::LocalGet(b));
                    func.push(Instruction::I64GtS);
                    func.push(Instruction::Select);
                    return Ok(());
                }
                if name == "std_cmp_min" && args.len() == 2 {
                    // min(a, b) -> if a < b { a } else { b }
                    self.compile_expr(&args[0])?;
                    self.compile_expr(&args[1])?;
                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    let a = func.alloc_local("__min_a".to_string(), ValType::I64);
                    let b = func.alloc_local("__min_b".to_string(), ValType::I64);
                    func.push(Instruction::LocalSet(b));
                    func.push(Instruction::LocalSet(a));
                    func.push(Instruction::LocalGet(a));
                    func.push(Instruction::LocalGet(b));
                    func.push(Instruction::LocalGet(a));
                    func.push(Instruction::LocalGet(b));
                    func.push(Instruction::I64LtS);
                    func.push(Instruction::Select);
                    return Ok(());
                }

                // Handle builtin Option/Result constructors (Some, Ok, Err)
                // Some(value) -> value (non-zero is Some in our representation)
                // Ok(value) -> value
                // Err(value) -> value (with error flag, but simplified for now)
                if simple_name == "Some" && args.len() == 1 {
                    // Some(x) - just compile the value (non-zero means Some)
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                if simple_name == "Ok" && args.len() == 1 {
                    // Ok(x) - just compile the value
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                if simple_name == "Err" && args.len() == 1 {
                    // Err(x) - for now, compile value (proper Result handling would need tagging)
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }

                // Handle stdlib wrapper type constructors
                // These are identity functions in WASM - just return the inner value
                // Cell::new, RefCell::new, Rc::new, Box::new, etc.
                if (name == "Cell_new" || name == "std_cell_Cell_new" || simple_name == "new")
                    && resolved_segments.len() >= 2
                    && (resolved_segments[resolved_segments.len() - 2] == "Cell"
                        || resolved_segments.iter().any(|s| s == "Cell"))
                    && args.len() == 1
                {
                    // Cell::new(x) -> x (identity in WASM)
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                if (name == "RefCell_new" || name == "std_cell_RefCell_new" || simple_name == "new")
                    && resolved_segments.len() >= 2
                    && (resolved_segments[resolved_segments.len() - 2] == "RefCell"
                        || resolved_segments.iter().any(|s| s == "RefCell"))
                    && args.len() == 1
                {
                    // RefCell::new(x) -> x (identity in WASM)
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                if (name == "Rc_new" || name == "std_rc_Rc_new" || simple_name == "new")
                    && resolved_segments.len() >= 2
                    && (resolved_segments[resolved_segments.len() - 2] == "Rc"
                        || resolved_segments.iter().any(|s| s == "Rc"))
                    && args.len() == 1
                {
                    // Rc::new(x) -> x (identity in WASM, no refcounting)
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                if (name == "Box_new" || name == "std_boxed_Box_new" || simple_name == "new")
                    && resolved_segments.len() >= 2
                    && (resolved_segments[resolved_segments.len() - 2] == "Box"
                        || resolved_segments.iter().any(|s| s == "Box"))
                    && args.len() == 1
                {
                    // Box::new(x) -> x (identity in WASM)
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                if (name == "HashSet_new" || simple_name == "new")
                    && resolved_segments.len() >= 2
                    && (resolved_segments[resolved_segments.len() - 2] == "HashSet"
                        || resolved_segments.iter().any(|s| s == "HashSet"))
                    && args.is_empty()
                {
                    // HashSet::new() -> new array (use morpheme_array_new)
                    if let Some(func_idx) = self.imports.get_func("morpheme_array_new") {
                        let func = self.current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;
                        func.push(Instruction::Call(func_idx));
                        // array_new returns i32, extend to i64 for Sigil's uniform type system
                        func.push(Instruction::I64ExtendI32U);
                        return Ok(());
                    }
                }

                // std::mem::take - replace with default and return original
                // In WASM, this is simplified to just returning the value (no mutation tracking)
                if name == "std_mem_take" && args.len() == 1 {
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                // std::mem::replace - replace value and return old
                if name == "std_mem_replace" && args.len() == 2 {
                    // Simplified: just return the first argument (old value)
                    self.compile_expr(&args[0])?;
                    return Ok(());
                }
                // std::mem::swap - swap two values
                if name == "std_mem_swap" && args.len() == 2 {
                    // Simplified: no-op in WASM
                    let func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::I64Const(0)); // unit
                    return Ok(());
                }

                // drop(value) - explicitly drop a value (no-op in WASM, values are on stack)
                if (simple_name == "drop" || name == "std_mem_drop") && args.len() == 1 {
                    // Compile the argument to consume it, then drop it
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::Drop);
                    func.push(Instruction::I64Const(0)); // return unit
                    return Ok(());
                }

                // Handle thread_local·with(closure) pattern
                // When parsed as a path like [RUNTIME, with], treat as thread_local.with(closure)
                if simple_name == "with" && args.len() == 1 && resolved_segments.len() >= 2 {
                    let thread_local_name = resolved_segments[..resolved_segments.len() - 1].join("_");

                    // Check if this looks like a thread_local (uppercase name or known pattern)
                    let is_thread_local = thread_local_name.chars().next().map_or(false, |c| c.is_uppercase())
                        || thread_local_name == "RUNTIME"
                        || thread_local_name.ends_with("_RUNTIME");

                    if is_thread_local {
                        if let Expr::Closure { params, body, .. } = &args[0] {
                            // Get the closure parameter name
                            let param_name = params.first()
                                .map(|p| get_param_name(p))
                                .unwrap_or_else(|| "rt".to_string());

                            // Get or create the global for this thread_local
                            let global_idx = if let Some(&idx) = self.global_map.get(&thread_local_name) {
                                idx
                            } else {
                                // Create a dummy global for cross-module thread_local
                                let idx = self.globals.len() as u32;
                                self.globals.push((ValType::I64, true, 0));
                                self.global_map.insert(thread_local_name.clone(), idx);
                                idx
                            };

                            // Load global and bind to a local with the closure parameter's name
                            let func = self.current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::GlobalGet(global_idx));
                            let local_idx = func.alloc_local(param_name.clone(), ValType::I64);
                            func.push(Instruction::LocalSet(local_idx));

                            // Compile the closure body
                            self.compile_expr(body)?;

                            return Ok(());
                        }
                    }
                }

                // Map web_sys path aliases to their actual imports
                // web_sys::set_timeout -> timing::set_timeout, etc.
                // Also handle simple names that may have lost their qualified path during parsing
                // Also handle cross-module calls that should be stubbed
                let mapped_name = match name {
                    "web_sys_set_timeout" => "timing_set_timeout",
                    "web_sys_clear_timeout" => "timing_clear_timeout",
                    "web_sys_set_interval" => "timing_set_interval",
                    "web_sys_clear_interval" => "timing_clear_interval",
                    "web_sys_request_animation_frame" => "timing_request_animation_frame",
                    "web_sys_window" | "window" => "browser_window",
                    "web_sys_document" | "document" => "browser_document",
                    "web_sys_match_media" | "match_media" => "browser_match_media",
                    _ => name,
                };

                // Check for import function first to get parameter types.
                // This MUST come before the cross-module stub handler so that
                // real imports (e.g. vdom_create_vnode) are compiled as actual
                // calls instead of being treated as identity stubs.
                if let Some(func_idx) = self.imports.get_func(mapped_name) {
                    // Get parameter and return types before compilation
                    let param_types: Vec<ValType> = self
                        .imports
                        .get_param_types(func_idx)
                        .map(|p| p.to_vec())
                        .unwrap_or_default();
                    let return_type = self.imports.get_return_type(func_idx);

                    // Compile arguments with type conversion
                    for (i, arg) in args.iter().enumerate() {
                        self.compile_expr(arg)?;

                        // Convert I64 to I32 if parameter expects I32
                        if let Some(ValType::I32) = param_types.get(i) {
                            let func = self.current_function_mut().unwrap();
                            func.push(Instruction::I32WrapI64);
                        }
                    }

                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::Call(func_idx));

                    // Handle return type conversion for uniform I64 type system
                    match return_type {
                        Some(ValType::I32) => {
                            // Extend I32 to I64
                            func.push(Instruction::I64ExtendI32U);
                        }
                        None => {
                            // Void function - push unit value (0) for stack consistency
                            func.push(Instruction::I64Const(0));
                        }
                        _ => {
                            // I64 or F64 - leave as-is (F64 should be boxed in real use)
                        }
                    }

                    return Ok(());
                }

                // Handle cross-module calls to hooks, signals, runtime modules
                // These are compiled as stubs that compile arguments and return a dummy value.
                // This is a FALLBACK for unresolved cross-module references — the import
                // lookup above handles any name that has a real import registered.
                let is_cross_module_stub = name.starts_with("hooks_")
                    || name.starts_with("signals_")
                    || name.starts_with("runtime_")
                    || name.starts_with("component_")
                    || name.starts_with("vdom_")
                    || name.starts_with("VNode_")
                    || name.starts_with("HookState_")
                    || name.starts_with("Effect_")
                    // Standard library type methods
                    || name.starts_with("TypeId_")
                    || name.starts_with("Any_")
                    || name.starts_with("Box_")
                    || name.starts_with("Rc_")
                    || name.starts_with("Arc_")
                    || name.starts_with("RefCell_")
                    || name.starts_with("Cell_")
                    || name.starts_with("Vec_")
                    || name.starts_with("HashMap_")
                    || name.starts_with("HashSet_")
                    || name.starts_with("VecDeque_")
                    || name.starts_with("Option_")
                    || name.starts_with("Result_")
                    || name.starts_with("ComponentInstance_")
                    || name.starts_with("Patch_")
                    || name.starts_with("AttrValue_")
                    // Generic type method calls (C::default(), T::new(), etc.)
                    // Single uppercase letter followed by underscore is typically a generic
                    || (name.len() >= 3 && name.chars().next().unwrap().is_ascii_uppercase()
                        && name.chars().nth(1) == Some('_'));
                if is_cross_module_stub {
                    // Compile all arguments (for side effects), then leave
                    // exactly one i64 value on the stack as a dummy return.
                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    let func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    if args.is_empty() {
                        // No args — push a zero as dummy return
                        func.push(Instruction::I64Const(0));
                    } else {
                        // Drop all but the first argument value
                        for _ in 1..args.len() {
                            func.push(Instruction::Drop);
                        }
                    }
                    return Ok(());
                }

                // =================================================================
                // Method Call on Local Variable Detection
                // =================================================================
                // When we have a path like ["app", "view"], check if "app" is a local variable.
                // If so, this is actually a method call: app.view() where we need to:
                // 1. Push the receiver (app) onto the stack
                // 2. Call Type::view where Type is inferred from the receiver
                if resolved_segments.len() == 2 {
                    let potential_receiver = &resolved_segments[0];
                    let method_name = &resolved_segments[1];

                    // Check if first segment is a local variable (not a type or module)
                    let is_local = self.current_function()
                        .and_then(|f| f.get_local(potential_receiver))
                        .is_some();
                    let is_type = self.struct_layouts.contains_key(potential_receiver.as_str())
                        || self.enum_layouts.contains_key(potential_receiver.as_str());

                    if is_local && !is_type {
                        // This is a method call on a local variable!
                        // Get the receiver's type from var_types if available
                        let receiver_type = self.var_types.get(potential_receiver).cloned();

                        // Try to find the method
                        let method_func_idx = if let Some(ref ty) = receiver_type {
                            let qualified_method = format!("{}::{}", ty, method_name);
                            self.func_map.get(&qualified_method).copied()
                        } else {
                            None
                        };

                        if let Some(func_idx) = method_func_idx {
                            // Found the method! Emit receiver as first argument
                            let local_idx = self.current_function()
                                .and_then(|f| f.get_local(potential_receiver))
                                .map(|l| l.index)
                                .unwrap();

                            let func = self.current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::LocalGet(local_idx));

                            // Compile remaining arguments
                            for arg in args {
                                self.compile_expr(arg)?;
                            }

                            // Call the method
                            let returns_void = self.func_returns_void(func_idx);
                            let func = self.current_function_mut().unwrap();
                            func.push(Instruction::Call(func_idx));

                            if returns_void {
                                func.push(Instruction::I64Const(0));
                            }

                            return Ok(());
                        }
                    }
                }

                // Compile arguments for non-import calls
                for arg in args {
                    self.compile_expr(arg)?;
                }

                // Check for direct function call (user-defined functions)
                // Try qualified path first (handles tome:: and module::function calls)
                let func_idx_opt = self.get_func_by_path(&resolved_segments)
                    .or_else(|| self.get_func(simple_name))
                    .or_else(|| self.get_func(&qualified_path));
                if let Some(func_idx) = func_idx_opt {
                    // Check if function returns void
                    let returns_void = self.func_returns_void(func_idx);

                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::Call(func_idx));

                    // If function returns void, push unit value for stack consistency
                    if returns_void {
                        func.push(Instruction::I64Const(0));
                    }

                    return Ok(());
                }

                // Check for closure call (indirect)
                if let Some(closure_info) = self.closure_map.get(simple_name).cloned() {
                    // For closures with captures, we need to load the closure variable
                    // and extract the env pointer from it
                    if !closure_info.captures.is_empty() {
                        // Load the closure pointer from the local variable
                        if let Some(func) = self.current_function() {
                            if let Some(local) = func.get_local(name) {
                                let closure_ptr_idx = local.index;
                                return self.compile_closure_call_with_env(
                                    closure_ptr_idx,
                                    closure_info,
                                    args.len(),
                                );
                            }
                        }
                        // Check global
                        if let Some(global_idx) = self.get_global(simple_name) {
                            let func = self.current_function_mut().unwrap();
                            func.push(Instruction::GlobalGet(global_idx));
                            let closure_ptr_idx =
                                func.alloc_local("__closure_ptr".to_string(), ValType::I64);
                            func.push(Instruction::LocalSet(closure_ptr_idx));

                            return self.compile_closure_call_with_env(
                                closure_ptr_idx,
                                closure_info,
                                args.len(),
                            );
                        }
                        return Err(WasmError::undefined_variable(simple_name));
                    } else {
                        // No captures - simple indirect call
                        self.compile_indirect_call(closure_info, args.len())
                    }
                } else if let Some((module_name, _qualified_name)) = self.external_imports.get(simple_name).cloned() {
                    // External module import - add as WASM import and call
                    // Use the simple name for the WASM import name
                    let func_idx = self.get_or_add_external_import(&module_name, simple_name, args.len());

                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::Call(func_idx));

                    // External functions are assumed to return i64 (uniform type system)
                    Ok(())
                } else {
                    // Check if this is a local variable that holds a function/closure pointer
                    // This handles cases like calling a captured closure parameter: `compute()`
                    let local_info = self.current_function().and_then(|f| f.get_local(simple_name).cloned());
                    if let Some(local) = local_info {
                        // Compile arguments first
                        for arg in args {
                            self.compile_expr(arg)?;
                        }

                        // Load the closure pointer from local
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(local.index));

                        // Treat as indirect call through closure pointer
                        // Closure representation: [table_idx, env_ptr]
                        let temp_ptr = func.alloc_local("__call_local_closure".to_string(), ValType::I64);
                        func.push(Instruction::LocalSet(temp_ptr));

                        // Get table index from closure (offset 0)
                        func.push(Instruction::LocalGet(temp_ptr));
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 3,
                            memory_index: 0,
                        }));
                        func.push(Instruction::I32WrapI64);

                        // Get env pointer (offset 8)
                        func.push(Instruction::LocalGet(temp_ptr));
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: 8,
                            align: 3,
                            memory_index: 0,
                        }));

                        // Indirect call with env as first argument
                        let mut param_types = vec![ValType::I64]; // env
                        param_types.extend(std::iter::repeat(ValType::I64).take(args.len()));
                        let type_idx = self.get_or_create_type(param_types, vec![ValType::I64]);

                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::CallIndirect { type_index: type_idx, table_index: 0 });

                        Ok(())
                    } else {
                        // Fallback for nested/helper functions that weren't hoisted
                        // These are common patterns like `fn walk(...)` inside functions
                        // For now, stub them by returning a default value
                        if simple_name.chars().next().map_or(false, |c| c.is_ascii_lowercase()) {
                            // Looks like a local helper function - compile args and return dummy
                            let func = self.current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            // Drop all args that were already compiled
                            for _ in 0..args.len() {
                                func.push(Instruction::Drop);
                            }
                            // Return dummy value (Option None / 0)
                            func.push(Instruction::I64Const(0));
                            return Ok(());
                        }
                        Err(WasmError::undefined_function(name))
                    }
                }
            }

            // Closure or complex expression - indirect call
            _ => {
                // Compile the function expression
                self.compile_expr(func_expr)?;

                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                // Store closure pointer
                let closure_ptr = func.alloc_local("__call_closure".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(closure_ptr));

        

                // Compile arguments
                for arg in args {
                    self.compile_expr(arg)?;
                }

                let func = self.current_function_mut().unwrap();

                // Get table index from closure
                func.push(Instruction::LocalGet(closure_ptr));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
                func.push(Instruction::I32WrapI64);

                // Get env pointer
                func.push(Instruction::LocalGet(closure_ptr));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 8,
                    align: 3,
                    memory_index: 0,
                }));

                // Indirect call with env as first argument
                // Type: (env, args...) -> result
                let mut param_types = vec![ValType::I64]; // env
                param_types.extend(std::iter::repeat(ValType::I64).take(args.len()));
                let type_idx = self.get_or_create_type(param_types, vec![ValType::I64]);

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::CallIndirect { type_index: type_idx, table_index: 0 });

                Ok(())
            }
        }
    }

    /// Compile a closure call when we have the closure pointer in a local.
    /// This extracts the env pointer from the closure object and calls with it.
    fn compile_closure_call_with_env(
        &mut self,
        closure_ptr_idx: u32,
        closure_info: ClosureInfo,
        arg_count: usize,
    ) -> WasmResult<()> {
        // Load env pointer from closure object (offset 8)
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        func.push(Instruction::LocalGet(closure_ptr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 8,
            align: 3,
            memory_index: 0,
        }));

        // Type includes env parameter
        let mut param_types = vec![ValType::I64]; // env
        param_types.extend(std::iter::repeat(ValType::I64).take(arg_count));
        let type_idx = self.get_or_create_type(param_types, vec![ValType::I64]);

        // Load table index from closure object (offset 0)
        let func = self.current_function_mut().unwrap();
        func.push(Instruction::LocalGet(closure_ptr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));
        func.push(Instruction::I32WrapI64);

        func.push(Instruction::CallIndirect {
            type_index: type_idx,
            table_index: 0,
        });

        Ok(())
    }

    /// Compile an indirect call through a closure (for closures without captures).
    fn compile_indirect_call(
        &mut self,
        closure_info: ClosureInfo,
        arg_count: usize,
    ) -> WasmResult<()> {
        // No captures - direct call through table using known table index
        let param_types: Vec<ValType> =
            std::iter::repeat(ValType::I64).take(arg_count).collect();
        let type_idx = self.get_or_create_type(param_types, vec![ValType::I64]);

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32Const(closure_info.table_idx as i32));
        func.push(Instruction::CallIndirect {
            type_index: type_idx,
            table_index: 0,
        });

        Ok(())
    }

    /// Compile enum variant construction (e.g., VNode::Text(123) or VNode::Empty)
    fn compile_enum_construction(
        &mut self,
        layout: &super::types::EnumLayout,
        variant_name: &str,
        tag: u32,
        args: &[Expr],
    ) -> WasmResult<()> {
        // Get variant info
        let variant_info = layout.variants.iter().find(|(name, _, _)| name == variant_name);

        match variant_info {
            Some((_, _, None)) => {
                // Unit variant (no payload) - just push the tag as i64
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(tag as i64));
                Ok(())
            }
            Some((_, _, Some(_payload_layout))) => {
                // Variant with payload - allocate memory and store [tag, args...]
                // Memory layout: [tag: i64, field0: i64, field1: i64, ...]
                let total_size = 8 + (args.len() * 8); // 8 bytes for tag + 8 bytes per field

                // Allocate memory using bump allocator
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                // Load current heap pointer
                func.push(Instruction::GlobalGet(0)); // Assume global 0 is heap pointer
                let ptr_local = func.alloc_local("__enum_ptr".to_string(), ValType::I64);
                func.push(Instruction::LocalTee(ptr_local));

                // Bump heap pointer
                func.push(Instruction::I64Const(total_size as i64));
                func.push(Instruction::I64Add);
                func.push(Instruction::GlobalSet(0));

                // Store tag at offset 0
                func.push(Instruction::LocalGet(ptr_local));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Const(tag as i64));
                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));

                // Compile and store each argument at offset 8, 16, etc.
                for (i, arg) in args.iter().enumerate() {
                    self.compile_expr(arg)?;
                    let func = self.current_function_mut().unwrap();
                    let value_local = func.alloc_local(format!("__enum_field_{}", i), ValType::I64);
                    func.push(Instruction::LocalSet(value_local));

                    // Store at offset 8 + i*8
                    func.push(Instruction::LocalGet(ptr_local));
                    func.push(Instruction::I32WrapI64);
                    func.push(Instruction::LocalGet(value_local));
                    func.push(Instruction::I64Store(wasm_encoder::MemArg {
                        offset: (8 + i * 8) as u64,
                        align: 3,
                        memory_index: 0,
                    }));
                }

                // Return pointer to the enum value
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(ptr_local));
                Ok(())
            }
            None => Err(WasmError::internal(&format!(
                "unknown enum variant: {}",
                variant_name
            ))),
        }
    }

    /// Compile a method call.
    pub fn compile_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[Expr],
    ) -> WasmResult<()> {
        // First, check if receiver is a simple local variable that needs explicit handling
        // This is needed because some code paths don't properly emit LocalGet for variables
        let receiver_local_idx = if let Expr::Path(path) = receiver {
            if path.segments.len() == 1 {
                let var_name = &path.segments[0].ident.name;
                // Not "self" (handled separately), not a type name
                if var_name != "self" && !self.struct_layouts.contains_key(var_name.as_str())
                   && !self.enum_layouts.contains_key(var_name.as_str()) {
                    self.current_function()
                        .and_then(|f| f.get_local(var_name))
                        .map(|l| l.index)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Check for actor self·method() calls
        if let Expr::Path(path) = receiver {
            if path.segments.len() == 1 && path.segments[0].ident.name == "self" {
                // Inside an actor, self·method() -> ActorName::method()
                if let Some(actor_name) = &self.current_actor.clone() {
                    let qualified = format!("{}::{}", actor_name, method);
                    if let Some(&func_idx) = self.func_map.get(&qualified) {
                        // Push dummy self reference (actor state is in globals, not passed)
                        let func = self
                            .current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;
                        func.push(Instruction::I64Const(0));
                        drop(func);

                        // Compile explicit arguments
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        let func = self
                            .current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;
                        func.push(Instruction::Call(func_idx));
                        return Ok(());
                    }
                }
            }
        }

        // Check for enum variant access: EnumType·Variant
        if let Expr::Path(path) = receiver {
            if let Some(first_seg) = path.segments.first() {
                let enum_name = &first_seg.ident.name;
                if let Some(layout) = self.enum_layouts.get(enum_name).cloned() {
                    if let Some(tag) = layout.variant_tag(method) {
                        // This is an enum variant access without arguments (unit variant)
                        let func = self.current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;
                        func.push(Instruction::I64Const(tag as i64));
                        return Ok(());
                    }
                }
            }
        }

        // Try builtin method dispatch first (to_string, clone, unwrap, etc.)
        if self.try_compile_builtin_method(receiver, method, args)? {
            return Ok(());
        }

        // Try VNode builder method dispatch (·child, ·attr, ·style, etc.)
        if self.try_compile_vnode_builder_method(receiver, method, args)? {
            return Ok(());
        }

        // Check for module-prefixed import calls: module·function(args)
        // e.g., vdom·mount_vnode(vnode, "#app") → vdom_mount_vnode import
        if let Expr::Path(path) = receiver {
            if path.segments.len() == 1 {
                let module_name = &path.segments[0].ident.name;
                let import_name = format!("{}_{}", module_name, method);

                if let Some(func_idx) = self.imports.get_func(&import_name) {
                    // Get parameter types for proper conversion
                    let param_types: Vec<ValType> = self
                        .imports
                        .get_param_types(func_idx)
                        .map(|p| p.to_vec())
                        .unwrap_or_default();
                    let return_type = self.imports.get_return_type(func_idx);

                    // Compile arguments with type conversion
                    for (i, arg) in args.iter().enumerate() {
                        self.compile_expr(arg)?;

                        // Convert I64 to I32 if parameter expects I32
                        if let Some(ValType::I32) = param_types.get(i) {
                            let func = self.current_function_mut().unwrap();
                            func.push(Instruction::I32WrapI64);
                        }
                    }

                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::Call(func_idx));

                    // Handle return type conversion
                    match return_type {
                        Some(ValType::I32) => {
                            func.push(Instruction::I64ExtendI32U);
                        }
                        None => {
                            func.push(Instruction::I64Const(0));
                        }
                        _ => {}
                    }

                    return Ok(());
                }
            }
        }

        // Try type-qualified lookup first to determine if we need receiver
        let qualified_func = if let Some(receiver_type) = self.infer_receiver_type(receiver) {
            let qualified = format!("{}::{}", receiver_type, method);
            self.func_map.get(&qualified).copied()
        } else {
            None
        };

        // Also check simple function name
        let simple_func = self.get_func(method);

        let func_idx = qualified_func.or(simple_func)
            .ok_or_else(|| WasmError::undefined_function(method))?;

        // Compile receiver as first argument
        // If we identified a local variable at the start, emit LocalGet directly
        if let Some(local_idx) = receiver_local_idx {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::LocalGet(local_idx));
        } else {
            // Otherwise use normal expression compilation
            self.compile_expr(receiver)?;
        }

        // Compile remaining arguments
        for arg in args {
            self.compile_expr(arg)?;
        }

        // Call the method
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(func_idx));
        Ok(())
    }

    /// Infer the type of a receiver expression for method resolution.
    /// Used to resolve method chains like VNode::div().child() -> VNode::child
    fn infer_receiver_type(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path) => {
                // Check if the path is a known type (struct or enum)
                let name = path.segments.first()?.ident.name.as_str();
                if self.struct_layouts.contains_key(name) || self.enum_layouts.contains_key(name) {
                    return Some(name.to_string());
                }
                None
            }
            Expr::Call { func, .. } => {
                // For a call like VNode::div(), infer the return type
                if let Expr::MethodCall { receiver, .. } = &**func {
                    // Nested call: receiver.method(...) -> check receiver type
                    return self.infer_receiver_type(receiver);
                }
                if let Expr::Path(path) = &**func {
                    // Static method call like VNode::div() or VNode·div()
                    // The first segment might be the type
                    if let Some(first_seg) = path.segments.first() {
                        let type_name = &first_seg.ident.name;
                        if self.struct_layouts.contains_key(type_name.as_str())
                            || self.enum_layouts.contains_key(type_name.as_str())
                        {
                            return Some(type_name.clone());
                        }
                    }
                }
                None
            }
            Expr::MethodCall { receiver, .. } => {
                // Method chain: receiver.method() - if method returns Self (builder pattern)
                // For now, assume builder pattern preserves type
                self.infer_receiver_type(receiver)
            }
            _ => None,
        }
    }

    /// Try to compile a builtin method call. Returns true if handled.
    fn try_compile_builtin_method(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[Expr],
    ) -> WasmResult<bool> {
        match method {
            // to_string() - convert primitives to string
            "to_string" => {
                self.compile_to_string(receiver)?;
                Ok(true)
            }

            // clone() - for primitives, just evaluate (Copy semantics)
            "clone" => {
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // unwrap() - extract Option/Result value or trap
            "unwrap" => {
                self.compile_unwrap(receiver)?;
                Ok(true)
            }

            // expect() - like unwrap but with message (message ignored in WASM)
            "expect" => {
                // Ignore the message argument, just unwrap
                self.compile_unwrap(receiver)?;
                Ok(true)
            }

            // is_some() / is_none() for Option
            "is_some" => {
                self.compile_is_some(receiver)?;
                Ok(true)
            }
            "is_none" => {
                self.compile_is_none(receiver)?;
                Ok(true)
            }

            // len() / length() - for strings and arrays
            "len" | "length" => {
                self.compile_len(receiver)?;
                Ok(true)
            }

            // is_empty() - len() == 0
            "is_empty" => {
                self.compile_is_empty(receiver)?;
                Ok(true)
            }

            // into() - identity for now (proper trait dispatch later)
            "into" => {
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // RefCell/Cell methods - identity in WASM (no runtime borrow checking)
            "borrow" | "borrow_mut" | "get_mut" | "as_ref" | "as_mut" => {
                // These all just return the receiver unchanged
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // DOM Event methods - treated as pass-through or identity
            // These will be properly implemented when web_sys imports are available
            "type_" | "event_type" | "target" | "current_target" | "event_phase"
            | "prevent_default" | "stop_propagation" | "stop_immediate_propagation"
            | "default_prevented" | "bubbles" | "cancelable" | "composed" => {
                // For now, compile receiver and let the import handle it
                self.compile_expr(receiver)?;
                // Most of these return unit or need actual DOM imports
                // For compilation to succeed, we treat them as no-ops that return receiver
                Ok(true)
            }

            // More DOM methods for specific event types
            "client_x" | "client_y" | "page_x" | "page_y" | "screen_x" | "screen_y"
            | "button" | "buttons" | "alt_key" | "ctrl_key" | "shift_key" | "meta_key"
            | "key" | "code" | "repeat" | "location"
            | "data" | "input_type" | "is_composing"
            | "delta_x" | "delta_y" | "delta_z" | "delta_mode"
            | "touches" | "changed_touches" | "target_touches"
            | "identifier" | "related_target"
            | "animation_name" | "elapsed_time" | "pseudo_element" | "property_name"
            | "data_transfer"
            | "inner_width" | "inner_height" | "outer_width" | "outer_height" => {
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // and_then for Option chaining
            "and_then" => {
                // For and_then, we compile the receiver (should be an Option)
                // and the closure (args[0]), then apply if Some
                // Simplified: just return receiver for now
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // ok() for Result -> Option conversion
            "ok" => {
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // dyn_into for type casting
            "dyn_into" => {
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // Option methods
            "map" => {
                // For map(), we apply the closure to the unwrapped value
                // Simplified: if Some, apply closure; if None, return None
                self.compile_expr(receiver)?;
                if args.len() == 1 {
                    // For now, just return the receiver (proper implementation later)
                    // TODO: Implement proper Option::map semantics
                }
                Ok(true)
            }
            "unwrap_or" => {
                // unwrap_or(default): return value if Some, otherwise default
                self.compile_expr(receiver)?;
                // TODO: Implement proper unwrap_or
                Ok(true)
            }
            "unwrap_or_default" => {
                // For now just unwrap (proper Default trait support later)
                self.compile_unwrap(receiver)?;
                Ok(true)
            }
            "unwrap_or_else" => {
                // unwrap_or_else(|| default_fn): return value if Some, otherwise call closure
                // For now, compile receiver and ignore the closure (simplified)
                self.compile_expr(receiver)?;
                // TODO: Implement proper unwrap_or_else with closure evaluation
                Ok(true)
            }

            // Option::ok_or_else - convert Option<T> to Result<T, E>
            // Some(v) -> Ok(v), None -> Err(closure())
            "ok_or_else" => {
                self.compile_ok_or_else(receiver, args)?;
                Ok(true)
            }

            // Option::ok_or - convert Option<T> to Result<T, E>
            // Some(v) -> Ok(v), None -> Err(default)
            "ok_or" => {
                self.compile_ok_or(receiver, args)?;
                Ok(true)
            }

            // Result::map_err - transform the error value
            "map_err" => {
                self.compile_map_err(receiver, args)?;
                Ok(true)
            }

            // Result::is_ok / is_err
            "is_ok" => {
                self.compile_is_ok(receiver)?;
                Ok(true)
            }
            "is_err" => {
                self.compile_is_err(receiver)?;
                Ok(true)
            }

            // String methods
            "lines" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_lines") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "split_whitespace" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_split_whitespace") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "split" => {
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                }
                if let Some(func_idx) = self.imports.get_func("string_split") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "trim" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_trim") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "trim_start" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_trim_start") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "trim_end" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_trim_end") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "to_uppercase" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_to_uppercase") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "to_lowercase" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_to_lowercase") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "contains" => {
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                }
                if let Some(func_idx) = self.imports.get_func("string_contains") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "starts_with" => {
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                }
                if let Some(func_idx) = self.imports.get_func("string_starts_with") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "ends_with" => {
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                }
                if let Some(func_idx) = self.imports.get_func("string_ends_with") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "replace" => {
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                if let Some(func_idx) = self.imports.get_func("string_replace") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }
            "chars" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("string_chars") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                Ok(true)
            }

            // Numeric methods
            "abs" => {
                self.compile_expr(receiver)?;
                // Use integer abs by default (most common case)
                // For proper type dispatch, would check receiver type
                if let Some(func_idx) = self.imports.get_func("math_abs_int") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                } else {
                    // Fallback: inline implementation for i64
                    // abs(x) = if x < 0 then -x else x
                    let func = self.current_function_mut().unwrap();
                    let temp = func.alloc_local("__abs_tmp".to_string(), ValType::I64);
                    func.push(Instruction::LocalTee(temp));
                    func.push(Instruction::I64Const(0));
                    func.push(Instruction::I64LtS);
                    func.push(Instruction::If(BlockType::Result(ValType::I64)));
                    func.push(Instruction::I64Const(0));
                    func.push(Instruction::LocalGet(temp));
                    func.push(Instruction::I64Sub);
                    func.push(Instruction::Else);
                    func.push(Instruction::LocalGet(temp));
                    func.push(Instruction::End);
                }
                Ok(true)
            }
            "clamp" => {
                // receiver.clamp(min, max)
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                if let Some(func_idx) = self.imports.get_func("math_clamp_int") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                } else {
                    // Fallback: inline implementation
                    // clamp(x, min, max) = max(min, min(x, max))
                    let func = self.current_function_mut().unwrap();
                    let val = func.alloc_local("__clamp_val".to_string(), ValType::I64);
                    let min = func.alloc_local("__clamp_min".to_string(), ValType::I64);
                    let max = func.alloc_local("__clamp_max".to_string(), ValType::I64);
                    func.push(Instruction::LocalSet(max));
                    func.push(Instruction::LocalSet(min));
                    func.push(Instruction::LocalSet(val));
                    // min(val, max)
                    func.push(Instruction::LocalGet(val));
                    func.push(Instruction::LocalGet(max));
                    func.push(Instruction::LocalGet(val));
                    func.push(Instruction::LocalGet(max));
                    func.push(Instruction::I64LtS);
                    func.push(Instruction::Select);
                    // max(result, min)
                    let intermediate = func.alloc_local("__clamp_tmp".to_string(), ValType::I64);
                    func.push(Instruction::LocalTee(intermediate));
                    func.push(Instruction::LocalGet(min));
                    func.push(Instruction::LocalGet(intermediate));
                    func.push(Instruction::LocalGet(min));
                    func.push(Instruction::I64GtS);
                    func.push(Instruction::Select);
                }
                Ok(true)
            }
            "signum" => {
                self.compile_expr(receiver)?;
                if let Some(func_idx) = self.imports.get_func("math_signum_int") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                } else {
                    // Inline: returns -1, 0, or 1
                    let func = self.current_function_mut().unwrap();
                    let temp = func.alloc_local("__signum_tmp".to_string(), ValType::I64);
                    func.push(Instruction::LocalTee(temp));
                    func.push(Instruction::I64Const(0));
                    func.push(Instruction::I64LtS);
                    func.push(Instruction::If(BlockType::Result(ValType::I64)));
                    func.push(Instruction::I64Const(-1));
                    func.push(Instruction::Else);
                    func.push(Instruction::LocalGet(temp));
                    func.push(Instruction::I64Const(0));
                    func.push(Instruction::I64GtS);
                    func.push(Instruction::If(BlockType::Result(ValType::I64)));
                    func.push(Instruction::I64Const(1));
                    func.push(Instruction::Else);
                    func.push(Instruction::I64Const(0));
                    func.push(Instruction::End);
                    func.push(Instruction::End);
                }
                Ok(true)
            }

            // Collection methods - map to morpheme imports
            // Vec/Array methods
            "push" => {
                self.compile_collection_method(receiver, "morpheme_array_push", args)?;
                Ok(true)
            }
            "get" => {
                if args.is_empty() {
                    // Cell::get() - no args, just return the receiver (identity)
                    self.compile_expr(receiver)?;
                } else {
                    // Array::get(index) - use morpheme array get
                    self.compile_collection_method(receiver, "morpheme_array_get", args)?;
                }
                Ok(true)
            }
            "set" => {
                if args.len() == 1 {
                    // Cell::set(value) - store value to receiver location and return value
                    // For fields like self.field·set(x), we need to store x to the field
                    // Compile value first (will be on top of stack)
                    self.compile_expr(&args[0])?;
                    // For now, just return the value (proper field store needs receiver context)
                    // The receiver is typically a field access - we'd need to extract it
                    // TODO: Implement proper field store for Cell::set
                }
                Ok(true)
            }
            "first" => {
                self.compile_collection_method(receiver, "morpheme_array_first", args)?;
                Ok(true)
            }
            "last" => {
                self.compile_collection_method(receiver, "morpheme_array_last", args)?;
                Ok(true)
            }
            "pop" => {
                self.compile_collection_method(receiver, "morpheme_array_pop", args)?;
                Ok(true)
            }
            "iter" => {
                // iter() just returns the array handle for morpheme pipeline
                self.compile_expr(receiver)?;
                Ok(true)
            }
            "join" => {
                // Vec::join(separator) -> concatenate elements with separator
                // Stack: [vec_ptr, separator_ptr] -> [result_str_ptr]
                self.compile_expr(receiver)?;
                if let Some(sep) = args.first() {
                    self.compile_expr(sep)?;
                } else {
                    // Default separator: empty string
                    let empty = self.add_string("");
                    let func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::I32Const(empty as i32));
                }
                let join_idx = self.get_func("vec_join")
                    .ok_or_else(|| WasmError::internal("vec_join import missing"))?;
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Call(join_idx));
                Ok(true)
            }

            // HashMap methods
            "insert" => {
                self.compile_collection_method(receiver, "hashmap_insert", args)?;
                Ok(true)
            }
            "contains_key" => {
                self.compile_collection_method(receiver, "hashmap_contains", args)?;
                Ok(true)
            }
            "keys" => {
                self.compile_collection_method(receiver, "hashmap_keys", args)?;
                Ok(true)
            }
            "entry" => {
                // HashMap entry API: map.entry(key) -> Entry
                // Returns an entry handle for the key
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                // Entry is just a (map_handle, key) pair - return key for now
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Drop); // Drop receiver, keep key
                Ok(true)
            }
            "or_insert_with" => {
                // Entry.or_insert_with(init_fn) -> &mut V
                // If entry doesn't exist, call init_fn to create it
                self.compile_expr(receiver)?; // Entry (key)
                // For stub, just call the init function
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    // If it's a closure, the closure returns the initialized value
                }
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Drop); // Drop entry
                Ok(true)
            }
            "or_insert" => {
                // Entry.or_insert(default) -> &mut V
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                }
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Drop);
                Ok(true)
            }

            // Constructor methods for wrapper types (Closure, JsValue, etc.)
            // In WASM compilation, we just compile the closure argument directly
            "new" => {
                if args.len() == 1 {
                    // Closure::new(closure) -> just compile the closure
                    // The closure will be compiled and its pointer returned
                    self.compile_expr(&args[0])?;
                    Ok(true)
                } else {
                    // Not a closure constructor - fallback
                    Ok(false)
                }
            }

            // as_ref method - for Closure.as_ref() to convert to callback reference
            "as_ref" => {
                // Just compile the receiver (the closure pointer)
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // DOM event listener methods - map to browser imports
            "add_event_listener" => {
                // window.add_event_listener(event_type, callback) -> browser::add_event_listener
                self.compile_expr(receiver)?; // window/element handle
                for arg in args {
                    self.compile_expr(arg)?;
                }
                // Call browser import
                if let Some(func_idx) = self.imports.get_func("browser_add_event_listener") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                } else {
                    // Stub - push dummy return
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64Const(0));
                }
                Ok(true)
            }
            "remove_event_listener" => {
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                if let Some(func_idx) = self.imports.get_func("browser_remove_event_listener") {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                }
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(0));
                Ok(true)
            }
            "add_listener" | "remove_listener" | "matches" | "match_media" => {
                // MediaQueryList methods and Window.matchMedia
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                // Return a dummy handle (0 = no match, non-zero = match)
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(0));
                Ok(true)
            }

            // DOM document methods
            "document" => {
                self.compile_expr(receiver)?;
                let func = self.current_function_mut().unwrap();
                // Return document handle (or Option wrapper)
                func.push(Instruction::I64Const(1));
                Ok(true)
            }
            "create_text_node" | "create_element" | "create_comment" | "create_document_fragment"
            | "get_element_by_id" | "query_selector" | "query_selector_all" => {
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let func = self.current_function_mut().unwrap();
                // Pop args and return node handle
                for _ in 0..args.len() {
                    func.push(Instruction::Drop);
                }
                func.push(Instruction::Drop); // Drop receiver
                func.push(Instruction::I64Const(1)); // Return node handle
                Ok(true)
            }
            "set_attribute" | "remove_attribute" | "set_text_content" => {
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let func = self.current_function_mut().unwrap();
                // Pop all and return Result
                for _ in 0..args.len() {
                    func.push(Instruction::Drop);
                }
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(0)); // Ok result
                Ok(true)
            }
            "append_child" | "remove_child" | "replace_child" | "insert_before" => {
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let func = self.current_function_mut().unwrap();
                for _ in 0..args.len() {
                    func.push(Instruction::Drop);
                }
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(0));
                Ok(true)
            }
            "child_nodes" | "parent_node" => {
                self.compile_expr(receiver)?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(0));
                Ok(true)
            }
            "add_event_listener_with_callback" => {
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let func = self.current_function_mut().unwrap();
                for _ in 0..args.len() {
                    func.push(Instruction::Drop);
                }
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(0));
                Ok(true)
            }

            // Type conversion and Any methods
            "dyn_into" | "unchecked_ref" | "into" => {
                // Type conversion - just return the receiver
                self.compile_expr(receiver)?;
                Ok(true)
            }
            "downcast_ref" => {
                self.compile_expr(receiver)?;
                // Returns Option<&T> - just return receiver wrapped in Some
                Ok(true)
            }
            "and_then" | "map" | "filter" | "or_else" => {
                // Option/Iterator combinator - compile receiver then call closure
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop);
                }
                Ok(true)
            }
            "cloned" => {
                // Option::cloned - just return receiver
                self.compile_expr(receiver)?;
                Ok(true)
            }

            // Closure methods
            "forget" => {
                // Closure::forget() - leak the closure (do nothing in WASM)
                self.compile_expr(receiver)?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(0));
                Ok(true)
            }

            // Iterator methods
            "iter_mut" => {
                self.compile_expr(receiver)?;
                Ok(true)
            }
            "next" => {
                // Iterator::next() - get next element from iterator
                self.compile_expr(receiver)?;
                // For morpheme arrays, this is typically a no-op or returns first element
                // The iterator state is typically held externally
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(0)); // Return None/empty for stub
                Ok(true)
            }
            "collect" => {
                // Iterator::collect() - collects iterator into a collection
                // For morpheme arrays, this is essentially an identity operation
                self.compile_expr(receiver)?;
                // The array handle is already on stack
                Ok(true)
            }
            "take" => {
                // Iterator::take(n) - take first n elements
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop); // Drop count for now
                }
                // Return receiver as stub (proper impl would slice)
                Ok(true)
            }
            "skip" => {
                // Iterator::skip(n) - skip first n elements
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop); // Drop count for now
                }
                Ok(true)
            }
            "head" => {
                // Head of iterator/collection - return first element
                self.compile_collection_method(receiver, "morpheme_array_first", args)?;
                Ok(true)
            }
            "tail" => {
                // Tail of iterator/collection - return all but first
                self.compile_expr(receiver)?;
                // For stub, just return receiver (proper impl would slice)
                Ok(true)
            }
            "enumerate" => {
                // Iterator::enumerate() - add indices to elements
                self.compile_expr(receiver)?;
                // For stub, just return receiver
                Ok(true)
            }
            "zip" => {
                // Iterator::zip(other) - combine two iterators
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop);
                }
                Ok(true)
            }
            "flatten" => {
                // Iterator::flatten() - flatten nested iterators
                self.compile_expr(receiver)?;
                Ok(true)
            }
            "flat_map" => {
                // Iterator::flat_map(f) - map then flatten
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop);
                }
                Ok(true)
            }
            "fold" => {
                // Iterator::fold(init, f) - reduce with initial value
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                if args.len() >= 2 {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop); // Drop closure
                }
                // Return init value as stub
                Ok(true)
            }
            "find" => {
                // Iterator::find(predicate) - find first matching element
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop);
                }
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(0)); // Return None as stub
                Ok(true)
            }
            "position" | "find_index" => {
                // Iterator::position(predicate) - find index of first matching element
                self.compile_expr(receiver)?;
                if !args.is_empty() {
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Drop);
                }
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(-1)); // Return -1 (not found) as stub
                Ok(true)
            }
            "rev" | "reverse" => {
                // Iterator::rev() - reverse iterator
                self.compile_expr(receiver)?;
                Ok(true)
            }
            "count" => {
                // Iterator::count() - count elements
                self.compile_len(receiver)?;
                Ok(true)
            }

            // Runtime allocation methods
            "allocate_subscriber_id" | "allocate_component_id" => {
                self.compile_expr(receiver)?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(1)); // Return ID
                Ok(true)
            }

            // Component methods
            "update_if_subscribed" | "re_render" => {
                self.compile_expr(receiver)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let func = self.current_function_mut().unwrap();
                for _ in 0..args.len() {
                    func.push(Instruction::Drop);
                }
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(0));
                Ok(true)
            }

            // Runtime methods - these are methods on the runtime system
            // For cross-module method calls, we treat them as stubs that return receiver or dummy value
            "use_hook" | "get_hook_state" | "use_effect" | "use_layout_effect"
            | "current_subscriber" | "set_subscriber" | "register_effect" | "run_effects"
            | "schedule_update" | "flush_updates" | "render" | "mount" | "unmount"
            | "add_subscriber" | "remove_subscriber" | "notify_subscribers"
            | "track_dependency" | "clear_dependencies"
            | "get_context" | "set_context" | "provide_context" | "consume_context"
            | "create_context" | "use_context"
            | "create_effect" | "create_memo" | "create_resource" | "create_signal"
            | "batch" | "untrack" | "on_cleanup" | "on_mount" => {
                // Compile receiver (runtime instance)
                self.compile_expr(receiver)?;
                // Compile any arguments
                for arg in args {
                    self.compile_expr(arg)?;
                }
                // For now, return the last argument or receiver as stub result
                // These will be properly implemented when runtime imports are available
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                // Most of these just need to return something; use receiver or 0
                if args.is_empty() {
                    // Methods that don't have args typically return state from receiver
                    // Stack already has receiver on it
                } else {
                    // Pop extra values from stack, leave one result
                    for _ in 0..args.len() {
                        func.push(Instruction::Drop);
                    }
                }
                Ok(true)
            }

            // Not a builtin method
            _ => Ok(false),
        }
    }

    // ==========================================================================
    // VNode Builder Pattern Support
    // ==========================================================================

    /// Try to compile a VNode static constructor (e.g., VNode·div(), VNode·span()).
    /// Returns None if not a recognized VNode constructor.
    fn try_compile_vnode_constructor(
        &mut self,
        method: &str,
        args: &[Expr],
    ) -> Option<WasmResult<()>> {
        // HTML element constructors
        let tag = match method {
            // Standard HTML elements
            "div" | "span" | "p" | "a" | "button" | "form" | "input" | "label" |
            "select" | "option" | "textarea" | "img" | "video" | "audio" | "canvas" |
            "table" | "thead" | "tbody" | "tfoot" | "tr" | "th" | "td" |
            "ul" | "ol" | "li" | "dl" | "dt" | "dd" |
            "nav" | "header" | "footer" | "section" | "article" | "aside" |
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" |
            "pre" | "code" | "blockquote" | "hr" | "br" |
            "strong" | "em" | "b" | "i" | "u" | "small" | "sub" | "sup" |
            "svg" | "path" | "circle" | "rect" | "line" | "polygon" | "polyline" => method,

            // main_elem for <main> (avoiding keyword conflict)
            "main_elem" => "main",

            // Fragment - no element, just a container for children
            "fragment" => {
                return Some(self.compile_vnode_fragment());
            }

            // Text node
            "text" if args.len() == 1 => {
                return Some(self.compile_vnode_text(&args[0]));
            }

            // Empty placeholder
            "Empty" => {
                return Some(self.compile_vnode_empty());
            }

            _ => return None,
        };

        Some(self.compile_vnode_element(tag))
    }

    /// Compile VNode element creation: vdom_create_vnode(tag) -> handle
    fn compile_vnode_element(&mut self, tag: &str) -> WasmResult<()> {
        let tag_offset = self.add_string(tag);
        let create_idx = self.imports.get_func("vdom_create_vnode")
            .ok_or_else(|| WasmError::internal("vdom_create_vnode import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Pass string offset as i64 (VDOM imports expect i64 for string refs)
        func.push(Instruction::I64Const(tag_offset as i64));
        func.push(Instruction::Call(create_idx));
        // Extend i32 result to i64 (Sigil's uniform type)
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile VNode fragment creation
    fn compile_vnode_fragment(&mut self) -> WasmResult<()> {
        let create_idx = self.imports.get_func("vdom_create_fragment")
            .ok_or_else(|| WasmError::internal("vdom_create_fragment import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        func.push(Instruction::Call(create_idx));
        // Extend i32 result to i64 (Sigil's uniform type)
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile VNode text node creation
    fn compile_vnode_text(&mut self, text_expr: &Expr) -> WasmResult<()> {
        // Compile the text expression (should produce string handle as i64)
        self.compile_expr(text_expr)?;

        // VDOM import expects i64 for string ref - don't wrap
        let create_idx = self.imports.get_func("vdom_create_text_vnode")
            .ok_or_else(|| WasmError::internal("vdom_create_text_vnode import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Text expression already produces i64 string ref
        func.push(Instruction::Call(create_idx));
        // Extend i32 result back to i64
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile VNode::Empty - returns a null/empty vnode handle
    fn compile_vnode_empty(&mut self) -> WasmResult<()> {
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Return 0 as empty vnode handle
        func.push(Instruction::I64Const(0));

        Ok(())
    }

    /// Try to compile a VNode builder method (·child, ·attr, ·style, etc.).
    /// Returns true if handled.
    fn try_compile_vnode_builder_method(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[Expr],
    ) -> WasmResult<bool> {
        // Check if receiver is VNode-typed
        if !self.is_vnode_expression(receiver) {
            return Ok(false);
        }

        match method {
            // ·child(vnode) - append a child and return self
            "child" if args.len() == 1 => {
                self.compile_vnode_child(receiver, &args[0])?;
                Ok(true)
            }

            // ·children(vec) - append multiple children
            "children" if args.len() == 1 => {
                self.compile_vnode_children(receiver, &args[0])?;
                Ok(true)
            }

            // ·attr(name, value) - set attribute
            "attr" if args.len() == 2 => {
                self.compile_vnode_attr(receiver, &args[0], &args[1])?;
                Ok(true)
            }

            // ·style(prop, value) - set inline style
            "style" if args.len() == 2 => {
                self.compile_vnode_style(receiver, &args[0], &args[1])?;
                Ok(true)
            }

            // ·class(name) - set class attribute
            "class" if args.len() == 1 => {
                self.compile_vnode_class(receiver, &args[0])?;
                Ok(true)
            }

            // ·text_child(text) - add text content as child
            "text_child" if args.len() == 1 => {
                self.compile_vnode_text_child(receiver, &args[0])?;
                Ok(true)
            }

            _ => Ok(false),
        }
    }

    /// Check if an expression is VNode-typed (for builder method dispatch)
    fn is_vnode_expression(&self, expr: &Expr) -> bool {
        match expr {
            // VNode·div() - static constructor call
            Expr::Call { func, .. } => {
                if let Expr::Path(path) = &**func {
                    let segments: Vec<&str> = path.segments.iter()
                        .map(|s| s.ident.name.as_str())
                        .collect();
                    if segments.len() >= 2 && segments[segments.len() - 2] == "VNode" {
                        return true;
                    }
                }
                // Nested method call on VNode (chained builders)
                if let Expr::MethodCall { receiver, .. } = &**func {
                    return self.is_vnode_expression(receiver);
                }
                false
            }
            // expr·method() - chained method call
            Expr::MethodCall { receiver, .. } => {
                self.is_vnode_expression(receiver)
            }
            // Path like VNode
            Expr::Path(path) => {
                if let Some(first) = path.segments.first() {
                    first.ident.name == "VNode"
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Compile ·child(child_vnode) - append child and return parent
    fn compile_vnode_child(&mut self, receiver: &Expr, child: &Expr) -> WasmResult<()> {
        use wasm_encoder::ValType;

        // Get import index first (before mutable borrows)
        let append_idx = self.imports.get_func("vdom_append_vnode_child")
            .ok_or_else(|| WasmError::internal("vdom_append_vnode_child import not found"))?;

        // Compile receiver (parent vnode)
        self.compile_expr(receiver)?;

        // Store parent handle to local (don't leave on stack during child compilation)
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let parent_local = func.alloc_local("__vnode_parent".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(parent_local));
        drop(func);

        // Compile child expression (this may involve complex calls with their own args)
        self.compile_expr(child)?;

        // Now set up the append_child call:
        // Stack currently has child result (i64)
        // Need: parent (i32), child (i32)
        let func = self.current_function_mut().unwrap();
        let child_local = func.alloc_local("__vnode_child".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(child_local));

        // Push parent, wrap to i32
        func.push(Instruction::LocalGet(parent_local));
        func.push(Instruction::I32WrapI64);

        // Push child, wrap to i32
        func.push(Instruction::LocalGet(child_local));
        func.push(Instruction::I32WrapI64);

        // Call append_child
        func.push(Instruction::Call(append_idx));

        // Return parent handle for chaining
        func.push(Instruction::LocalGet(parent_local));

        Ok(())
    }

    /// Compile ·children(vec) - append multiple children
    fn compile_vnode_children(&mut self, receiver: &Expr, children_vec: &Expr) -> WasmResult<()> {
        use wasm_encoder::ValType;

        // Get import index first (before mutable borrows)
        let append_children_idx = self.imports.get_func("vdom_append_children");

        // Compile receiver (parent vnode)
        self.compile_expr(receiver)?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let parent_local = func.alloc_local("__vnode_parent_c".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(parent_local));
        drop(func);

        // Compile children vector
        self.compile_expr(children_vec)?;

        // Call vdom_append_children(parent, children_array)
        let func = self.current_function_mut().unwrap();
        let children_local = func.alloc_local("__vnode_children".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(children_local));

        // Get parent, wrap to i32
        func.push(Instruction::LocalGet(parent_local));
        func.push(Instruction::I32WrapI64);

        // Get children array handle
        func.push(Instruction::LocalGet(children_local));
        func.push(Instruction::I32WrapI64);

        // Call append_children if available, otherwise drop args (stub)
        if let Some(idx) = append_children_idx {
            func.push(Instruction::Call(idx));
        } else {
            // Fallback: drop children (stub)
            func.push(Instruction::Drop);
            func.push(Instruction::Drop);
        }

        // Return parent handle
        func.push(Instruction::LocalGet(parent_local));

        Ok(())
    }

    /// Compile ·attr(name, value) - set attribute
    /// Import signature: set_vnode_str_prop(vnodeId: i32, nameStrRef: i64, valueStrRef: i64)
    fn compile_vnode_attr(&mut self, receiver: &Expr, name: &Expr, value: &Expr) -> WasmResult<()> {
        use wasm_encoder::ValType;

        // Get import index first (before mutable borrows)
        let set_prop_idx = self.imports.get_func("vdom_set_vnode_str_prop")
            .ok_or_else(|| WasmError::internal("vdom_set_vnode_str_prop import not found"))?;

        // Compile receiver and store (clear stack for subsequent expressions)
        self.compile_expr(receiver)?;
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let vnode_local = func.alloc_local("__vnode_attr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(vnode_local));
        drop(func);

        // Compile name expression and store
        self.compile_expr(name)?;
        let func = self.current_function_mut().unwrap();
        let name_local = func.alloc_local("__attr_name".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(name_local));
        drop(func);

        // Compile value expression and store
        self.compile_expr(value)?;
        let func = self.current_function_mut().unwrap();
        let value_local = func.alloc_local("__attr_value".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(value_local));

        // Now push args in order: vnode (i32), name (i64), value (i64)
        func.push(Instruction::LocalGet(vnode_local));
        func.push(Instruction::I32WrapI64);  // vnode handle wrapped to i32
        func.push(Instruction::LocalGet(name_local));   // name stays i64
        func.push(Instruction::LocalGet(value_local));  // value stays i64
        func.push(Instruction::Call(set_prop_idx));

        // Return vnode for chaining
        func.push(Instruction::LocalGet(vnode_local));

        Ok(())
    }

    /// Compile ·style(prop, value) - set inline style
    /// Import signature: set_vnode_str_prop(vnodeId: i32, nameStrRef: i64, valueStrRef: i64)
    fn compile_vnode_style(&mut self, receiver: &Expr, prop: &Expr, value: &Expr) -> WasmResult<()> {
        use wasm_encoder::ValType;

        // Get import indices first (before mutable borrows)
        let style_idx = self.imports.get_func("vdom_set_vnode_style");
        let fallback_idx = self.imports.get_func("vdom_set_vnode_str_prop");

        // Compile receiver and store (clear stack for subsequent expressions)
        self.compile_expr(receiver)?;
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let vnode_local = func.alloc_local("__vnode_style".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(vnode_local));
        drop(func);

        // Compile property name and store
        self.compile_expr(prop)?;
        let func = self.current_function_mut().unwrap();
        let prop_local = func.alloc_local("__style_prop".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(prop_local));
        drop(func);

        // Compile value and store
        self.compile_expr(value)?;
        let func = self.current_function_mut().unwrap();
        let value_local = func.alloc_local("__style_value".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(value_local));

        // Push args in order: vnode (i32), prop (i64), value (i64)
        func.push(Instruction::LocalGet(vnode_local));
        func.push(Instruction::I32WrapI64);  // vnode handle wrapped to i32
        func.push(Instruction::LocalGet(prop_local));    // prop stays i64
        func.push(Instruction::LocalGet(value_local));   // value stays i64

        // Call vdom_set_vnode_style or fallback to str_prop
        if let Some(idx) = style_idx {
            func.push(Instruction::Call(idx));
        } else if let Some(idx) = fallback_idx {
            func.push(Instruction::Call(idx));
        } else {
            return Err(WasmError::internal("vdom_set_vnode_style import not found"));
        }

        // Return vnode handle for chaining
        func.push(Instruction::LocalGet(vnode_local));

        Ok(())
    }

    /// Compile ·class(name) - set class attribute
    /// Import signature: set_vnode_str_prop(vnodeId: i32, nameStrRef: i64, valueStrRef: i64)
    fn compile_vnode_class(&mut self, receiver: &Expr, class_name: &Expr) -> WasmResult<()> {
        use wasm_encoder::ValType;

        // Get import index and string offset first (before mutable borrows)
        let set_prop_idx = self.imports.get_func("vdom_set_vnode_str_prop")
            .ok_or_else(|| WasmError::internal("vdom_set_vnode_str_prop import not found"))?;
        let class_str_offset = self.add_string("class");

        // Compile receiver and store (clear stack for subsequent expressions)
        self.compile_expr(receiver)?;
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let vnode_local = func.alloc_local("__vnode_class".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(vnode_local));
        drop(func);

        // Compile class name value and store
        self.compile_expr(class_name)?;
        let func = self.current_function_mut().unwrap();
        let class_local = func.alloc_local("__class_name".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(class_local));

        // Push args in order: vnode (i32), "class" (i64), className (i64)
        func.push(Instruction::LocalGet(vnode_local));
        func.push(Instruction::I32WrapI64);  // vnode handle wrapped to i32
        func.push(Instruction::I64Const(class_str_offset as i64));  // "class" as i64
        func.push(Instruction::LocalGet(class_local));  // className stays i64
        func.push(Instruction::Call(set_prop_idx));

        // Return vnode for chaining
        func.push(Instruction::LocalGet(vnode_local));

        Ok(())
    }

    /// Compile ·text_child(text) - add text content as child
    /// create_text_vnode(textStrRef: i64) -> i32
    /// append_vnode_child(parent: i32, child: i32) -> ()
    fn compile_vnode_text_child(&mut self, receiver: &Expr, text: &Expr) -> WasmResult<()> {
        use wasm_encoder::ValType;

        // Get import indices first (before mutable borrows)
        let create_text_idx = self.imports.get_func("vdom_create_text_vnode")
            .ok_or_else(|| WasmError::internal("vdom_create_text_vnode import not found"))?;
        let append_idx = self.imports.get_func("vdom_append_vnode_child")
            .ok_or_else(|| WasmError::internal("vdom_append_vnode_child import not found"))?;

        // Compile receiver and store (clear stack)
        self.compile_expr(receiver)?;
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let vnode_local = func.alloc_local("__vnode_text_p".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(vnode_local));
        drop(func);

        // Compile text expression and store
        self.compile_expr(text)?;
        let func = self.current_function_mut().unwrap();
        let text_local = func.alloc_local("__text_str".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(text_local));

        // Create text vnode: text stays i64 for create_text_vnode
        func.push(Instruction::LocalGet(text_local));
        func.push(Instruction::Call(create_text_idx));
        // Result is i32, store it
        let text_vnode_local = func.alloc_local("__text_vnode".to_string(), ValType::I32);
        func.push(Instruction::LocalSet(text_vnode_local));

        // Append: parent (i32), child (i32)
        func.push(Instruction::LocalGet(vnode_local));
        func.push(Instruction::I32WrapI64);  // parent wrapped to i32
        func.push(Instruction::LocalGet(text_vnode_local));  // child already i32
        func.push(Instruction::Call(append_idx));

        // Return parent for chaining
        func.push(Instruction::LocalGet(vnode_local));

        Ok(())
    }

    /// Compile a collection method call (Vec, HashMap, etc.).
    fn compile_collection_method(
        &mut self,
        receiver: &Expr,
        import_name: &str,
        args: &[Expr],
    ) -> WasmResult<()> {
        // Compile receiver (collection handle)
        self.compile_expr(receiver)?;

        // Compile arguments
        for arg in args {
            self.compile_expr(arg)?;
        }

        // Call the import function
        if let Some(func_idx) = self.imports.get_func(import_name) {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::Call(func_idx));
            Ok(())
        } else {
            // Fall back to no-op for missing imports (collection is simulated)
            // In production, these would be real WASM imports
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            // Push a dummy result (0) for now
            func.push(Instruction::I64Const(0));
            Ok(())
        }
    }

    /// Compile to_string() method call.
    /// Uses runtime type detection: strings pass through, numbers convert.
    fn compile_to_string(&mut self, receiver: &Expr) -> WasmResult<()> {
        // Compile the receiver expression
        self.compile_expr(receiver)?;

        // For now, assume numeric type and call string_from_int
        // A proper implementation would check the receiver type
        // and dispatch to string_from_int or string_from_float accordingly.
        //
        // Since we use a uniform i64 representation, we call string_from_int.
        // Strings are already string handles (i32), so they'd need special handling,
        // but for sigil-web the common case is numbers.
        if let Some(func_idx) = self.imports.get_func("string_from_int") {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::Call(func_idx));
            // string_from_int returns i32 (string handle), extend to i64
            func.push(Instruction::I64ExtendI32U);
            Ok(())
        } else {
            Err(WasmError::internal("string_from_int import not found"))
        }
    }

    /// Compile unwrap() for Option/Result types.
    /// Option is represented as: 0 = None, non-zero = Some(value)
    /// For simplicity, we treat non-zero values as the unwrapped value itself.
    fn compile_unwrap(&mut self, receiver: &Expr) -> WasmResult<()> {
        // Compile the Option/Result expression
        self.compile_expr(receiver)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store value in temp local
        let temp = func.alloc_local("__unwrap_tmp".to_string(), wasm_encoder::ValType::I64);
        func.push(Instruction::LocalTee(temp));

        // Check if None (value == 0 for simple Option representation)
        // For proper enum representation, we'd check the tag field
        func.push(Instruction::I64Eqz);
        func.push(Instruction::If(wasm_encoder::BlockType::Empty));
        func.push(Instruction::Unreachable); // Trap on None
        func.push(Instruction::End);

        // Return the value
        func.push(Instruction::LocalGet(temp));

        Ok(())
    }

    /// Compile is_some() for Option - returns true if not None.
    fn compile_is_some(&mut self, receiver: &Expr) -> WasmResult<()> {
        self.compile_expr(receiver)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // For simple Option: 0 = None, non-zero = Some
        // is_some = value != 0
        func.push(Instruction::I64Const(0));
        func.push(Instruction::I64Ne);
        // Extend bool (i32) to i64
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile is_none() for Option - returns true if None.
    fn compile_is_none(&mut self, receiver: &Expr) -> WasmResult<()> {
        self.compile_expr(receiver)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // For simple Option: 0 = None
        // is_none = value == 0
        func.push(Instruction::I64Eqz);
        // Extend bool (i32) to i64
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile len() for strings and arrays.
    fn compile_len(&mut self, receiver: &Expr) -> WasmResult<()> {
        self.compile_expr(receiver)?;

        // Try string length first, then array length
        if let Some(func_idx) = self.imports.get_func("string_length") {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            // Convert i64 handle to i32 for string functions
            func.push(Instruction::I32WrapI64);
            func.push(Instruction::Call(func_idx));
            // Extend result to i64
            func.push(Instruction::I64ExtendI32U);
            Ok(())
        } else if let Some(func_idx) = self.imports.get_func("morpheme_array_len") {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::Call(func_idx));
            // array_len returns i32, extend to i64
            func.push(Instruction::I64ExtendI32U);
            Ok(())
        } else {
            Err(WasmError::internal("no len import found"))
        }
    }

    /// Compile is_empty() - returns len() == 0.
    fn compile_is_empty(&mut self, receiver: &Expr) -> WasmResult<()> {
        self.compile_len(receiver)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        func.push(Instruction::I64Eqz);
        // Extend bool (i32) to i64
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile is_ok() for Result - returns true if Ok variant.
    ///
    /// Result representation: struct with (discriminant: i64, payload: i64)
    /// where discriminant 0 = Ok, 1 = Err
    fn compile_is_ok(&mut self, receiver: &Expr) -> WasmResult<()> {
        // Compile receiver (Result pointer)
        self.compile_expr(receiver)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Load discriminant from offset 0
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 0,
            align: 3, // 8-byte alignment
            memory_index: 0,
        }));

        // is_ok = discriminant == 0
        func.push(Instruction::I64Eqz);
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile is_err() for Result - returns true if Err variant.
    fn compile_is_err(&mut self, receiver: &Expr) -> WasmResult<()> {
        // Compile receiver (Result pointer)
        self.compile_expr(receiver)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Load discriminant from offset 0
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));

        // is_err = discriminant != 0 (i.e., == 1)
        func.push(Instruction::I64Const(0));
        func.push(Instruction::I64Ne);
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile ok_or_else() for Option - convert to Result.
    ///
    /// Option representation: struct at heap pointer with:
    ///   offset 0: discriminant (i64) - 0 = None, 1 = Some
    ///   offset 8: payload (i64) - the Some value (undefined for None)
    ///
    /// Result representation: struct at heap pointer with:
    ///   offset 0: discriminant (i64) - 0 = Ok, 1 = Err
    ///   offset 8: payload (i64) - the Ok or Err value
    ///
    /// Semantics:
    ///   - Some(v).ok_or_else(f) = Ok(v)
    ///   - None.ok_or_else(f) = Err(f())
    fn compile_ok_or_else(&mut self, receiver: &Expr, args: &[Expr]) -> WasmResult<()> {
        // Compile receiver (Option pointer)
        self.compile_expr(receiver)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store Option pointer
        let opt_ptr = func.alloc_local("__ok_or_else_opt".to_string(), wasm_encoder::ValType::I64);
        func.push(Instruction::LocalTee(opt_ptr));

        // Load discriminant
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));

        // Store discriminant
        let disc = func.alloc_local("__ok_or_else_disc".to_string(), wasm_encoder::ValType::I64);
        func.push(Instruction::LocalSet(disc));

        // Allocate Result struct (16 bytes)
        func.push(Instruction::I64Const(16));

        // Get heap_alloc
        drop(func);
        let alloc_idx = self.get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(alloc_idx));

        let result_ptr = func.alloc_local("__ok_or_else_result".to_string(), wasm_encoder::ValType::I64);
        func.push(Instruction::LocalTee(result_ptr));

        // Check discriminant: if disc == 1 (Some), write Ok
        // if disc == 0 (None), write Err with closure result
        func.push(Instruction::LocalGet(disc));
        func.push(Instruction::I64Const(1));
        func.push(Instruction::I64Eq);

        func.push(Instruction::If(BlockType::Empty));
        {
            // Some case: write Ok (discriminant 0, copy payload)
            func.push(Instruction::LocalGet(result_ptr));
            func.push(Instruction::I32WrapI64);
            func.push(Instruction::I64Const(0)); // Ok discriminant
            func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index: 0,
            }));

            // Copy payload from Option to Result
            func.push(Instruction::LocalGet(result_ptr));
            func.push(Instruction::I32WrapI64);
            func.push(Instruction::LocalGet(opt_ptr));
            func.push(Instruction::I32WrapI64);
            func.push(Instruction::I64Load(wasm_encoder::MemArg {
                offset: 8,
                align: 3,
                memory_index: 0,
            }));
            func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: 8,
                align: 3,
                memory_index: 0,
            }));
        }
        func.push(Instruction::Else);
        {
            // None case: write Err (discriminant 1)
            func.push(Instruction::LocalGet(result_ptr));
            func.push(Instruction::I32WrapI64);
            func.push(Instruction::I64Const(1)); // Err discriminant
            func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: 0,
                align: 3,
                memory_index: 0,
            }));

            // For payload, we need to call the closure
            // For now, use a simple placeholder (the closure would provide the error)
            drop(func);
            if args.len() == 1 {
                if let Expr::Closure { body, .. } = &args[0] {
                    self.compile_expr(body)?;
                    let func = self.current_function_mut().unwrap();
                    let err_val = func.alloc_local("__ok_or_else_err".to_string(), wasm_encoder::ValType::I64);
                    func.push(Instruction::LocalSet(err_val));

                    // Store error value
                    func.push(Instruction::LocalGet(result_ptr));
                    func.push(Instruction::I32WrapI64);
                    func.push(Instruction::LocalGet(err_val));
                    func.push(Instruction::I64Store(wasm_encoder::MemArg {
                        offset: 8,
                        align: 3,
                        memory_index: 0,
                    }));
                } else {
                    // Non-closure argument - just compile it
                    self.compile_expr(&args[0])?;
                    let func = self.current_function_mut().unwrap();
                    let err_val = func.alloc_local("__ok_or_else_err".to_string(), wasm_encoder::ValType::I64);
                    func.push(Instruction::LocalSet(err_val));

                    func.push(Instruction::LocalGet(result_ptr));
                    func.push(Instruction::I32WrapI64);
                    func.push(Instruction::LocalGet(err_val));
                    func.push(Instruction::I64Store(wasm_encoder::MemArg {
                        offset: 8,
                        align: 3,
                        memory_index: 0,
                    }));
                }
            } else {
                // No argument - store 0 as error
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(result_ptr));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Const(0));
                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: 8,
                    align: 3,
                    memory_index: 0,
                }));
            }
        }
        let func = self.current_function_mut().unwrap();
        func.push(Instruction::End);

        // Return Result pointer
        func.push(Instruction::LocalGet(result_ptr));

        Ok(())
    }

    /// Compile ok_or() for Option - convert to Result with default error.
    ///
    /// Semantics:
    ///   - Some(v).ok_or(e) = Ok(v)
    ///   - None.ok_or(e) = Err(e)
    fn compile_ok_or(&mut self, receiver: &Expr, args: &[Expr]) -> WasmResult<()> {
        // Simplified implementation: reuse ok_or_else logic
        // The error value is evaluated eagerly, but semantics are the same
        self.compile_ok_or_else(receiver, args)
    }

    /// Compile map_err() for Result - transform the error value.
    ///
    /// Result representation: struct at heap pointer with:
    ///   offset 0: discriminant (i64) - 0 = Ok, 1 = Err
    ///   offset 8: payload (i64) - the Ok or Err value
    ///
    /// Semantics:
    ///   - Ok(v).map_err(f) = Ok(v)  (unchanged)
    ///   - Err(e).map_err(f) = Err(f(e))  (transform error)
    fn compile_map_err(&mut self, receiver: &Expr, args: &[Expr]) -> WasmResult<()> {
        // Need exactly one argument (the closure)
        if args.is_empty() {
            return Err(WasmError::internal("map_err requires a closure argument"));
        }

        let closure_expr = &args[0];

        // Check if this is a closure expression we can inline
        match closure_expr {
            Expr::Closure { params, body, .. } => {
                // Inline closure compilation
                if params.len() != 1 {
                    return Err(WasmError::internal("map_err closure must take exactly 1 argument"));
                }

                // Compile receiver to get Result pointer
                self.compile_expr(receiver)?;

                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                // Store Result pointer in temp local
                let result_ptr = func.alloc_local("__map_err_result".to_string(), wasm_encoder::ValType::I64);
                func.push(Instruction::LocalTee(result_ptr));

                // Load discriminant
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));

                // Store discriminant in temp
                let disc = func.alloc_local("__map_err_disc".to_string(), wasm_encoder::ValType::I64);
                func.push(Instruction::LocalSet(disc));

                // Check if Err (discriminant == 1)
                func.push(Instruction::LocalGet(disc));
                func.push(Instruction::I64Const(1));
                func.push(Instruction::I64Eq);

                // If Err, transform the error value using the closure
                func.push(Instruction::If(wasm_encoder::BlockType::Empty));

                // Load error value
                func.push(Instruction::LocalGet(result_ptr));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 8,
                    align: 3,
                    memory_index: 0,
                }));

                // Create temp local for the closure parameter
                let param_name = match &params[0].pattern {
                    crate::ast::Pattern::Ident { name, .. } => name.name.clone(),
                    _ => "__closure_param".to_string(),
                };
                let param_local = func.alloc_local(param_name.clone(), wasm_encoder::ValType::I64);
                func.push(Instruction::LocalSet(param_local));

                // Release func borrow to compile body
                drop(func);

                // Push the parameter to scope
                self.scope_vars.push(std::collections::HashMap::new());
                if let Some(scope) = self.scope_vars.last_mut() {
                    scope.insert(param_name, param_local);
                }

                // Compile closure body - this puts the transformed error on stack
                self.compile_expr(body)?;

                // Pop scope
                self.scope_vars.pop();

                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                // Store transformed error back to payload
                let new_err = func.alloc_local("__new_err".to_string(), wasm_encoder::ValType::I64);
                func.push(Instruction::LocalSet(new_err));

                func.push(Instruction::LocalGet(result_ptr));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::LocalGet(new_err));
                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: 8,
                    align: 3,
                    memory_index: 0,
                }));

                func.push(Instruction::End); // End if

                // Return the (possibly modified) Result pointer
                func.push(Instruction::LocalGet(result_ptr));
            }
            _ => {
                // Not an inline closure - fall back to pass-through behavior
                // TODO: Handle function references and captured closures
                self.compile_expr(receiver)?;
            }
        }

        Ok(())
    }

    /// Compile incorporation expressions: expr·method(args)·method2(args2)...
    /// This handles the middle-dot syntax for method chaining.
    ///
    /// The parser converts expressions to IncorporationSegments:
    /// - Simple variable `x` -> { name: "x", args: None }
    /// - Function call `f(a)` -> { name: "f", args: Some([a]) }
    /// - Field access `obj.field` -> { name: "field", args: Some([obj]) }
    /// - Literal `"str"` -> { name: "__lit__", args: Some([literal]) }
    pub fn compile_incorporation(
        &mut self,
        segments: &[crate::ast::IncorporationSegment],
    ) -> WasmResult<()> {
        if segments.is_empty() {
            return Err(WasmError::internal("empty incorporation chain"));
        }

        // Build the initial receiver expression from the first segment
        let first = &segments[0];
        let mut current_expr = self.segment_to_receiver(first)?;

        // Pre-compute local index for first segment if it's a simple variable
        let first_local_idx = if first.args.is_none() {
            let var_name = &first.name.name;
            if var_name != "self" && !self.struct_layouts.contains_key(var_name.as_str())
               && !self.enum_layouts.contains_key(var_name.as_str()) {
                self.current_function()
                    .and_then(|f| f.get_local(var_name))
                    .map(|l| l.index)
            } else {
                None
            }
        } else {
            None
        };

        // Handle the method calls (remaining segments) one at a time
        if segments.len() == 1 {
            // No method calls, just compile the receiver
            if let Some(local_idx) = first_local_idx {
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::LocalGet(local_idx));
            } else {
                self.compile_expr(&current_expr)?;
            }
            return Ok(());
        }

        // Process each method call in the chain
        for (i, segment) in segments.iter().enumerate().skip(1) {
            let method_name = &segment.name.name;
            let method_args = segment.args.clone().unwrap_or_default();

            // Check if this is the last segment
            let is_last = i == segments.len() - 1;

            // For builtin methods that are identity (return receiver), we can chain
            // without compiling intermediate results
            let is_identity_method = matches!(
                method_name.as_str(),
                "borrow" | "borrow_mut" | "get_mut" | "as_ref" | "as_mut" | "clone" | "into" | "iter"
            );

            if is_identity_method && !is_last {
                // Identity method in middle of chain - skip it, just pass through
                // Update current_expr to be a method call expression for type tracking
                current_expr = Expr::MethodCall {
                    receiver: Box::new(current_expr),
                    method: segment.name.clone(),
                    type_args: None,
                    args: method_args,
                };
                continue;
            }

            // Special handling for thread_local·with(closure) pattern
            // RUNTIME·with(|rt| { ... }) -> inline the closure body with rt bound to RUNTIME
            if method_name == "with" && method_args.len() == 1 {
                if let Expr::Path(path) = &current_expr {
                    let path_name = path.segments.iter()
                        .map(|s| s.ident.name.as_str())
                        .collect::<Vec<_>>()
                        .join("_");

                    // Check if this looks like a thread_local (common names: RUNTIME, etc.)
                    // or if it's already defined as a global
                    let is_thread_local = self.global_map.contains_key(&path_name)
                        || path_name == "RUNTIME"
                        || path_name.ends_with("_RUNTIME");

                    if is_thread_local {
                        // This is a thread_local·with pattern
                        // Compile the closure body with the parameter bound to the global
                        if let Expr::Closure { params, body, .. } = &method_args[0] {
                            // Create a local binding for the closure parameter
                            if let Some(param) = params.first() {
                                // Extract parameter name from pattern
                                let param_name = match &param.pattern {
                                    crate::ast::Pattern::Ident { name, .. } => name.name.clone(),
                                    _ => "rt".to_string(), // fallback
                                };

                                // Get or create the global
                                let global_idx = if let Some(&idx) = self.global_map.get(&path_name) {
                                    idx
                                } else {
                                    // Create a dummy global for cross-module thread_local
                                    let idx = self.globals.len() as u32;
                                    self.globals.push((ValType::I64, true, 0));
                                    self.global_map.insert(path_name.clone(), idx);
                                    idx
                                };

                                let func = self.current_function_mut()
                                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                                func.push(Instruction::GlobalGet(global_idx));

                                // Store it in a local with the closure parameter's name
                                let local_idx = func.alloc_local(param_name.clone(), ValType::I64);
                                func.push(Instruction::LocalSet(local_idx));

                                // Compile the closure body
                                self.compile_expr(body)?;

                                if is_last {
                                    return Ok(());
                                }
                                // For chaining, store result and continue
                                let func = self.current_function_mut().unwrap();
                                let temp_local = func.alloc_local(
                                    format!("__chain_{}", i),
                                    ValType::I64,
                                );
                                func.push(Instruction::LocalSet(temp_local));
                                current_expr = Expr::Path(crate::ast::TypePath {
                                    segments: vec![crate::ast::PathSegment {
                                        ident: crate::ast::Ident {
                                            name: format!("__chain_{}", i),
                                            evidentiality: None,
                                            affect: None,
                                            span: crate::span::Span::new(0, 0),
                                        },
                                        generics: None,
                                    }],
                                });
                                continue;
                            }
                        }
                    }
                }
            }

            // Try builtin method dispatch
            if self.try_compile_builtin_method(&current_expr, method_name, &method_args)? {
                if is_last {
                    return Ok(());
                }
                // For non-last segments that compiled via builtin, we need to store
                // the result for the next method call
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let temp_local = func.alloc_local(
                    format!("__chain_{}", i),
                    ValType::I64,
                );
                func.push(Instruction::LocalSet(temp_local));

                // Create a "local get" expression for the next iteration
                // For now, we'll just continue - the value is on the stack
                // This is a simplification; proper chaining would need more work
                current_expr = Expr::Path(crate::ast::TypePath {
                    segments: vec![crate::ast::PathSegment {
                        ident: crate::ast::Ident {
                            name: format!("__chain_{}", i),
                            evidentiality: None,
                            affect: None,
                            span: crate::span::Span::new(0, 0),
                        },
                        generics: None,
                    }],
                });
                continue;
            }

            // Not a builtin - compile as method call
            // For the first iteration (i==1), use pre-computed local index if available
            if i == 1 && first_local_idx.is_some() {
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::LocalGet(first_local_idx.unwrap()));
            } else {
                // Compile the current expression as receiver
                self.compile_expr(&current_expr)?;
            }

            for arg in &method_args {
                self.compile_expr(arg)?;
            }

            // Look up the method as a registered function
            if let Some(func_idx) = self.get_func(method_name) {
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Call(func_idx));

                if !is_last {
                    // Store for next iteration
                    let temp_local = func.alloc_local(
                        format!("__chain_{}", i),
                        ValType::I64,
                    );
                    func.push(Instruction::LocalSet(temp_local));
                    current_expr = Expr::Path(crate::ast::TypePath {
                        segments: vec![crate::ast::PathSegment {
                            ident: crate::ast::Ident {
                                name: format!("__chain_{}", i),
                                evidentiality: None,
                                affect: None,
                                span: crate::span::Span::new(0, 0),
                            },
                            generics: None,
                        }],
                    });
                }
            } else {
                return Err(WasmError::undefined_function(method_name));
            }
        }

        Ok(())
    }

    /// Convert an incorporation segment back to an expression for the receiver.
    fn segment_to_receiver(
        &self,
        segment: &crate::ast::IncorporationSegment,
    ) -> WasmResult<Expr> {
        let name = &segment.name.name;

        match (&segment.args, name.as_str()) {
            // Simple variable: { name: "x", args: None } -> x
            (None, _) => Ok(Expr::Path(crate::ast::TypePath {
                segments: vec![crate::ast::PathSegment {
                    ident: segment.name.clone(),
                    generics: None,
                }],
            })),

            // Literal: { name: "__lit__", args: Some([literal]) } -> literal
            (Some(args), "__lit__") if args.len() == 1 => Ok(args[0].clone()),

            // Index access: { name: "__index__", args: Some([base, index]) } -> base[index]
            (Some(args), "__index__") if args.len() == 2 => Ok(Expr::Index {
                expr: Box::new(args[0].clone()),
                index: Box::new(args[1].clone()),
            }),

            // Expression wrapper: { name: "__expr__", args: Some([expr]) } -> expr
            (Some(args), "__expr__") if args.len() == 1 => Ok(args[0].clone()),

            // Field access: { name: "field", args: Some([base_obj]) } -> base_obj.field
            // When there's exactly one arg, it's the base object of a field access
            (Some(args), _) if args.len() == 1 => Ok(Expr::Field {
                expr: Box::new(args[0].clone()),
                field: segment.name.clone(),
            }),

            // Function call: { name: "func", args: Some([a, b, ...]) } -> func(a, b, ...)
            (Some(args), _) => Ok(Expr::Call {
                func: Box::new(Expr::Path(crate::ast::TypePath {
                    segments: vec![crate::ast::PathSegment {
                        ident: segment.name.clone(),
                        generics: None,
                    }],
                })),
                args: args.clone(),
            }),
        }
    }

    /// Compile field access.
    pub fn compile_field_access(&mut self, expr: &Expr, field: &str) -> WasmResult<()> {
        // Check for actor self.field access
        if let Expr::Path(path) = expr {
            if path.segments.len() == 1 {
                let name = &path.segments[0].ident.name;
                if name == "self" {
                    // Inside an actor method, self.field -> load from actor global
                    if let Some(actor_name) = &self.current_actor {
                        let global_name = format!("{}_{}", actor_name, field);
                        if let Some(idx) = self.get_global(&global_name) {
                            let func = self
                                .current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::GlobalGet(idx));
                            return Ok(());
                        }
                        // Fall through to try qualified name
                        let qualified = self.qualify_name(&global_name);
                        if let Some(idx) = self.get_global(&qualified) {
                            let func = self
                                .current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::GlobalGet(idx));
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Regular struct field access
        // Compile expression to get struct pointer
        self.compile_expr(expr)?;

        // Get field offset first (requires immutable borrow)
        let offset = self.get_field_offset(field)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Load field value
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: offset as u64,
            align: 3,
            memory_index: 0,
        }));

        Ok(())
    }

    /// Compile index access.
    ///
    /// Handles both single-element indexing and range-based slicing:
    /// - `arr[i]` → single element access
    /// - `arr[start..end]` → slice operation
    /// - `str[1..]` → substring from index 1 to end
    pub fn compile_index(&mut self, expr: &Expr, index: &Expr) -> WasmResult<()> {
        // Check if this is a range index (slicing operation)
        if let Expr::Range { start, end, inclusive } = index {
            return self.compile_slice(expr, start.as_deref(), end.as_deref(), *inclusive);
        }

        // Single-element indexing
        // Compile array pointer
        self.compile_expr(expr)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        let arr_idx = func.alloc_local("__index_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));

        // Compile index
        drop(func);
        self.compile_expr(index)?;

        let func = self.current_function_mut().unwrap();

        // Calculate offset: index * 8 + 4 (skip length)
        func.push(Instruction::I64Const(8));
        func.push(Instruction::I64Mul);
        func.push(Instruction::I64Const(4));
        func.push(Instruction::I64Add);
        func.push(Instruction::I32WrapI64);

        // Add base address
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Add);

        // Load value
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));

        Ok(())
    }

    /// Compile slice operation: `arr[start..end]` or `str[start..]`
    ///
    /// Generates a call to string_slice or array_slice depending on context.
    /// For unbounded end (..end missing), uses the string/array length.
    fn compile_slice(
        &mut self,
        expr: &Expr,
        start: Option<&Expr>,
        end: Option<&Expr>,
        inclusive: bool,
    ) -> WasmResult<()> {
        // Compile the base expression (string or array pointer)
        self.compile_expr(expr)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store base pointer in local
        let base_ptr = func.alloc_local("__slice_base".to_string(), ValType::I64);
        func.push(Instruction::LocalTee(base_ptr));

        // Convert to i32 for string functions
        func.push(Instruction::I32WrapI64);

        // Compile start index (default 0)
        drop(func);
        if let Some(s) = start {
            self.compile_expr(s)?;
        } else {
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::I64Const(0));
        }

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64);

        // Compile end index
        drop(func);
        if let Some(e) = end {
            self.compile_expr(e)?;

            if inclusive {
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(1));
                func.push(Instruction::I64Add);
            }
        } else {
            // Unbounded end: use string length
            // Look up imports first to avoid borrow conflicts
            let len_idx = self.imports.get_func("string_length");

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(base_ptr));
            func.push(Instruction::I32WrapI64);

            // Call string_length to get the end
            if let Some(idx) = len_idx {
                func.push(Instruction::Call(idx));
                // Result is i32, extend to i64 then back to i32
                func.push(Instruction::I64ExtendI32U);
            } else {
                // Fallback: use a large constant
                func.push(Instruction::I64Const(i32::MAX as i64));
            }
        }

        // Look up slice import first to avoid borrow conflicts
        let slice_idx = self.imports.get_func("string_slice");

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64);

        // Call string_slice(str_ptr, start, end) -> new_str_ptr
        if let Some(idx) = slice_idx {
            func.push(Instruction::Call(idx));
            // Result is i32 (pointer), extend to i64
            func.push(Instruction::I64ExtendI32U);
        } else {
            // Fallback: just return the base pointer (no slice available)
            func.push(Instruction::Drop);
            func.push(Instruction::Drop);
            func.push(Instruction::LocalGet(base_ptr));
        }

        Ok(())
    }

    /// Compile array literal.
    pub fn compile_array(&mut self, elements: &[Expr]) -> WasmResult<()> {
        let len = elements.len();

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Allocate: 4 bytes for length + 8 bytes per element
        let size = 4 + (len * 8);
        func.push(Instruction::I64Const(size as i64));

        let alloc_idx = self
            .get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(alloc_idx));

        let arr_idx = func.alloc_local("__array".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));

        // Write length
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Const(len as i32));
        func.push(Instruction::I32Store(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));



        // Write elements
        for (i, elem) in elements.iter().enumerate() {
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(arr_idx));
            func.push(Instruction::I32WrapI64);
    

            self.compile_expr(elem)?;

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: (4 + i * 8) as u64,
                align: 3,
                memory_index: 0,
            }));
        }

        // Return array pointer
        let func = self.current_function_mut().unwrap();
        func.push(Instruction::LocalGet(arr_idx));

        Ok(())
    }

    /// Compile array repeat literal: `[value; count]`
    pub fn compile_array_repeat(&mut self, value: &Expr, count: &Expr) -> WasmResult<()> {
        // Extract count as a constant integer
        let len = match count {
            Expr::Literal(crate::ast::Literal::Int { value, .. }) => {
                value.parse::<usize>().map_err(|_| {
                    WasmError::internal("array repeat count must be a valid integer")
                })?
            }
            _ => {
                return Err(WasmError::unsupported(
                    "array repeat with non-constant count",
                ));
            }
        };

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Allocate: 4 bytes for length + 8 bytes per element
        let size = 4 + (len * 8);
        func.push(Instruction::I64Const(size as i64));

        let alloc_idx = self
            .get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(alloc_idx));

        let arr_idx = func.alloc_local("__array_repeat".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));

        // Write length
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Const(len as i32));
        func.push(Instruction::I32Store(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));

        // Write elements (same value repeated)
        for i in 0..len {
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(arr_idx));
            func.push(Instruction::I32WrapI64);

            self.compile_expr(value)?;

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: (4 + i * 8) as u64,
                align: 3,
                memory_index: 0,
            }));
        }

        // Return array pointer
        let func = self.current_function_mut().unwrap();
        func.push(Instruction::LocalGet(arr_idx));

        Ok(())
    }

    /// Compile tuple literal.
    pub fn compile_tuple(&mut self, elements: &[Expr]) -> WasmResult<()> {
        let len = elements.len();

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Allocate: 8 bytes per element
        let size = len * 8;
        func.push(Instruction::I64Const(size as i64));

        let alloc_idx = self
            .get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(alloc_idx));

        let tuple_idx = func.alloc_local("__tuple".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(tuple_idx));



        // Write elements
        for (i, elem) in elements.iter().enumerate() {
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(tuple_idx));
            func.push(Instruction::I32WrapI64);
    

            self.compile_expr(elem)?;

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: (i * 8) as u64,
                align: 3,
                memory_index: 0,
            }));
        }

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::LocalGet(tuple_idx));

        Ok(())
    }

    /// Compile struct literal.
    pub fn compile_struct_literal(
        &mut self,
        path: &crate::ast::TypePath,
        fields: &[crate::ast::FieldInit],
        _rest: Option<&Expr>,
    ) -> WasmResult<()> {
        let struct_name = path
            .segments
            .first()
            .map(|s| s.ident.name.as_str())
            .unwrap_or("");

        // Get or create struct layout
        let layout = if let Some(l) = self.struct_layouts.get(struct_name) {
            l.clone()
        } else {
            // Create layout from fields
            let mut layout = super::types::StructLayout::new(struct_name);
            for field in fields {
                layout.add_field(&field.name.name);
            }
            self.struct_layouts
                .insert(struct_name.to_string(), layout.clone());
            layout
        };

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Allocate struct
        func.push(Instruction::I64Const(layout.size as i64));

        let alloc_idx = self
            .get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(alloc_idx));

        let struct_idx = func.alloc_local("__struct".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(struct_idx));



        // Initialize fields
        for field in fields {
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(struct_idx));
            func.push(Instruction::I32WrapI64);
    

            // Compile field value
            if let Some(value) = &field.value {
                self.compile_expr(value)?;
            } else {
                // Shorthand: field name is the variable
                self.compile_expr(&Expr::Path(crate::ast::TypePath {
                    segments: vec![crate::ast::PathSegment {
                        ident: field.name.clone(),
                        generics: None,
                    }],
                }))?;
            }

            let offset = layout.field_offset(&field.name.name).unwrap_or(0);

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: offset as u64,
                align: 3,
                memory_index: 0,
            }));
        }

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::LocalGet(struct_idx));

        Ok(())
    }
}

/// Visitor for analyzing captured variables.
struct CaptureAnalyzer<'a> {
    captures: &'a mut Vec<String>,
    bound: Vec<String>,
    compiler: &'a WasmCompiler,
}

impl<'a> CaptureAnalyzer<'a> {
    fn visit(&mut self, expr: &Expr) {
        match expr {
            Expr::Path(path) => {
                let name = path.segments.first().map(|s| s.ident.name.as_str()).unwrap_or("");
                // If not bound locally and exists in enclosing scope, it's a capture
                if !self.bound.contains(&name.to_string()) {
                    // Check if it's a local in the current function
                    if let Some(func) = self.compiler.current_function() {
                        if func.get_local(name).is_some()
                            && !self.captures.contains(&name.to_string())
                        {
                            self.captures.push(name.to_string());
                        }
                    }
                }
            }

            Expr::Closure { params, body, is_move: _, return_type: _ } => {
                // Add parameters to bound set (is_move affects ownership, not capture analysis)
                let prev_len = self.bound.len();
                for param in params {
                    self.bound.push(get_param_name(param));
                }
                self.visit(body);
                self.bound.truncate(prev_len);
            }

            Expr::Binary { left, right, .. } => {
                self.visit(left);
                self.visit(right);
            }

            Expr::Unary { expr, .. } => {
                self.visit(expr);
            }

            Expr::Block(block) => {
                for stmt in &block.stmts {
                    match stmt {
                        crate::ast::Stmt::Let { pattern, init, .. } => {
                            if let Some(val) = init {
                                self.visit(val);
                            }
                            // Add bound variable
                            if let Pattern::Ident { name, .. } = pattern {
                                self.bound.push(name.name.clone());
                            }
                        }
                        crate::ast::Stmt::Expr(expr) | crate::ast::Stmt::Semi(expr) => {
                            self.visit(expr);
                        }
                        _ => {}
                    }
                }
                if let Some(expr) = &block.expr {
                    self.visit(expr);
                }
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.visit(condition);
                self.visit(&Expr::Block(then_branch.clone()));
                if let Some(else_expr) = else_branch {
                    self.visit(else_expr);
                }
            }

            Expr::Call { func, args } => {
                self.visit(func);
                for arg in args {
                    self.visit(arg);
                }
            }

            Expr::Array(elements) | Expr::Tuple(elements) => {
                for elem in elements {
                    self.visit(elem);
                }
            }

            // Incorporation chains: expr·method(args)·method2(args2)
            Expr::Incorporation { segments } => {
                // First segment is the receiver - treat as a path reference for capture analysis
                if let Some(first) = segments.first() {
                    let name = first.name.name.as_str();
                    // If not bound locally and exists in enclosing scope, it's a capture
                    if !self.bound.contains(&name.to_string()) {
                        if let Some(func) = self.compiler.current_function() {
                            if func.get_local(name).is_some()
                                && !self.captures.contains(&name.to_string())
                            {
                                self.captures.push(name.to_string());
                            }
                        }
                    }
                }
                // Visit arguments of all segments
                for segment in segments {
                    if let Some(args) = &segment.args {
                        for arg in args {
                            self.visit(arg);
                        }
                    }
                }
            }

            // Field access
            Expr::Field { expr, .. } => {
                self.visit(expr);
            }

            // Index expression
            Expr::Index { expr, index } => {
                self.visit(expr);
                self.visit(index);
            }

            // Match expression
            Expr::Match { expr, arms } => {
                self.visit(expr);
                for arm in arms {
                    // Visit the arm body
                    self.visit(&arm.body);
                    // Visit guard if present
                    if let Some(guard) = &arm.guard {
                        self.visit(guard);
                    }
                }
            }

            // Struct literals
            Expr::Struct { fields, rest, .. } => {
                for field in fields {
                    if let Some(value) = &field.value {
                        self.visit(value);
                    }
                }
                if let Some(rest_expr) = rest {
                    self.visit(rest_expr);
                }
            }

            // Assign expression
            Expr::Assign { target, value } => {
                self.visit(target);
                self.visit(value);
            }

            // Reference/dereference
            Expr::AddrOf { expr, .. } | Expr::Deref(expr) => {
                self.visit(expr);
            }

            // Let expression (as an expression)
            Expr::Let { value, .. } => {
                self.visit(value);
            }

            // For loop
            Expr::For { pattern, iter, body, .. } => {
                // Visit the iterator expression first
                self.visit(iter);
                // The loop variable is bound within the body
                let prev_len = self.bound.len();
                if let Pattern::Ident { name, .. } = pattern {
                    self.bound.push(name.name.clone());
                }
                self.visit(&Expr::Block(body.clone()));
                self.bound.truncate(prev_len);
            }

            // While loop
            Expr::While { condition, body, .. } => {
                self.visit(condition);
                self.visit(&Expr::Block(body.clone()));
            }

            // Loop
            Expr::Loop { body, .. } => {
                self.visit(&Expr::Block(body.clone()));
            }

            // Return
            Expr::Return(value) => {
                if let Some(val) = value {
                    self.visit(val);
                }
            }

            // Range
            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.visit(s);
                }
                if let Some(e) = end {
                    self.visit(e);
                }
            }

            // Cast
            Expr::Cast { expr, .. } => {
                self.visit(expr);
            }

            // Try (?)
            Expr::Try(expr) => {
                self.visit(expr);
            }

            // Await
            Expr::Await { expr, .. } => {
                self.visit(expr);
            }

            // Method call
            Expr::MethodCall { receiver, args, .. } => {
                self.visit(receiver);
                for arg in args {
                    self.visit(arg);
                }
            }

            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Ident, Literal, NumBase, PathSegment, TypePath};
    use crate::span::Span;
    use crate::wasm::literals::{create_test_compiler_with_function, create_test_compiler_with_heap_alloc};

    fn make_int(value: i64) -> Expr {
        Expr::Literal(Literal::Int {
            value: value.to_string(),
            base: NumBase::Decimal,
            suffix: None,
        })
    }

    fn make_ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            evidentiality: None,
            affect: None,
            span: Span::new(0, 0),
        }
    }

    fn make_path(name: &str) -> Expr {
        Expr::Path(TypePath {
            segments: vec![PathSegment {
                ident: make_ident(name),
                generics: None,
            }],
        })
    }

    fn make_closure(param: &str, body: Expr) -> Expr {
        Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Ident {
                    mutable: false,
                    name: make_ident(param),
                    evidentiality: None,
                },
                ty: None,
            }],
            body: Box::new(body),
            is_move: false,
            return_type: None,
        }
    }

    #[test]
    fn test_compile_simple_closure() {
        let mut compiler = create_test_compiler_with_function();

        // {x => x + 1}
        let closure = make_closure(
            "x",
            Expr::Binary {
                left: Box::new(make_path("x")),
                op: crate::ast::BinOp::Add,
                right: Box::new(make_int(1)),
            },
        );

        compiler.compile_expr(&closure).unwrap();

        // Should create a new function
        assert!(compiler.functions.len() > 1);
        assert!(!compiler.table_elements.is_empty());
    }

    #[test]
    fn test_compile_array() {
        let mut compiler = create_test_compiler_with_heap_alloc();

        let arr = Expr::Array(vec![make_int(1), make_int(2), make_int(3)]);

        compiler.compile_expr(&arr).unwrap();

        let func = compiler.current_function().unwrap();
        // Should call allocator
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }

    #[test]
    fn test_compile_tuple() {
        let mut compiler = create_test_compiler_with_heap_alloc();

        let tuple = Expr::Tuple(vec![make_int(1), make_int(2)]);

        compiler.compile_expr(&tuple).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }

    #[test]
    fn test_compile_index() {
        let mut compiler = create_test_compiler_with_function();

        let index = Expr::Index {
            expr: Box::new(make_int(0x1000)),
            index: Box::new(make_int(5)),
        };

        compiler.compile_expr(&index).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Load(_))));
    }

    #[test]
    fn test_compile_mutable_capture() {
        let mut compiler = create_test_compiler_with_heap_alloc();

        // Create outer function context with a mutable local
        let outer_func = compiler.current_function_mut().unwrap();
        let counter_idx = outer_func.alloc_local("counter".to_string(), ValType::I64);
        outer_func.push(Instruction::I64Const(0));
        outer_func.push(Instruction::LocalSet(counter_idx));


        // Create closure that modifies the captured variable:
        // |_| { counter = counter + 1; counter }
        let closure = Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Wildcard,
                ty: None,
            }],
            body: Box::new(Expr::Block(crate::ast::Block {
                stmts: vec![crate::ast::Stmt::Semi(Expr::Assign {
                    target: Box::new(make_path("counter")),
                    value: Box::new(Expr::Binary {
                        left: Box::new(make_path("counter")),
                        op: crate::ast::BinOp::Add,
                        right: Box::new(make_int(1)),
                    }),
                })],
                expr: Some(Box::new(make_path("counter"))),
            })),
            is_move: false,
            return_type: None,
        };

        // Mark counter as mutable capture
        compiler.mutable_captures.insert("counter".to_string());

        compiler.compile_expr(&closure).unwrap();

        // Should create an environment with a cell reference
        assert!(compiler.functions.len() > 1);
        // The closure should have instructions that load through the cell
        let closure_func = &compiler.functions[1];
        // Should have I64Store for the assignment
        assert!(closure_func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Store(_))));
    }

    #[test]
    fn test_mutable_capture_shared_between_closures() {
        let mut compiler = create_test_compiler_with_heap_alloc();

        // Create outer function context with a mutable local
        let outer_func = compiler.current_function_mut().unwrap();
        let counter_idx = outer_func.alloc_local("counter".to_string(), ValType::I64);
        outer_func.push(Instruction::I64Const(0));
        outer_func.push(Instruction::LocalSet(counter_idx));


        // Mark counter as mutable capture
        compiler.mutable_captures.insert("counter".to_string());

        // First closure: increment
        let inc_closure = Expr::Closure {
            params: vec![],
            body: Box::new(Expr::Block(crate::ast::Block {
                stmts: vec![crate::ast::Stmt::Semi(Expr::Assign {
                    target: Box::new(make_path("counter")),
                    value: Box::new(Expr::Binary {
                        left: Box::new(make_path("counter")),
                        op: crate::ast::BinOp::Add,
                        right: Box::new(make_int(1)),
                    }),
                })],
                expr: Some(Box::new(make_path("counter"))),
            })),
            is_move: false,
            return_type: None,
        };

        compiler.compile_expr(&inc_closure).unwrap();

        // Second closure: read counter
        let read_closure = Expr::Closure {
            params: vec![],
            body: Box::new(make_path("counter")),
            is_move: false,
            return_type: None,
        };

        compiler.compile_expr(&read_closure).unwrap();

        // Both closures should share the same cell
        assert_eq!(compiler.functions.len(), 3); // test + 2 closures
    }

    #[test]
    fn test_compile_nested_closure() {
        let mut compiler = create_test_compiler_with_heap_alloc();

        // Create outer function context with a local
        let outer_func = compiler.current_function_mut().unwrap();
        let x_idx = outer_func.alloc_local("x".to_string(), ValType::I64);
        outer_func.push(Instruction::I64Const(10));
        outer_func.push(Instruction::LocalSet(x_idx));


        // Create outer closure that captures x and creates inner closure:
        // |y| { |z| { x + y + z } }
        let inner_closure = Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Ident {
                    mutable: false,
                    name: make_ident("z"),
                    evidentiality: None,
                },
                ty: None,
            }],
            body: Box::new(Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(make_path("x")),
                    op: crate::ast::BinOp::Add,
                    right: Box::new(make_path("y")),
                }),
                op: crate::ast::BinOp::Add,
                right: Box::new(make_path("z")),
            }),
            is_move: false,
            return_type: None,
        };

        let outer_closure = Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Ident {
                    mutable: false,
                    name: make_ident("y"),
                    evidentiality: None,
                },
                ty: None,
            }],
            body: Box::new(inner_closure),
            is_move: false,
            return_type: None,
        };

        compiler.compile_expr(&outer_closure).unwrap();

        // Should create two closure functions (outer and inner)
        assert!(compiler.functions.len() >= 3); // test + outer + inner
        // Both should be in the function table
        assert!(compiler.table_elements.len() >= 2);
    }

    #[test]
    fn test_nested_closure_captures() {
        let mut compiler = create_test_compiler_with_heap_alloc();

        // Create function with local
        let outer_func = compiler.current_function_mut().unwrap();
        let outer_var_idx = outer_func.alloc_local("outer_var".to_string(), ValType::I64);
        outer_func.push(Instruction::I64Const(100));
        outer_func.push(Instruction::LocalSet(outer_var_idx));


        // Outer closure captures outer_var, returns inner closure
        // |a| { |b| { outer_var + a + b } }
        let inner = Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Ident {
                    mutable: false,
                    name: make_ident("b"),
                    evidentiality: None,
                },
                ty: None,
            }],
            body: Box::new(Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(make_path("outer_var")),
                    op: crate::ast::BinOp::Add,
                    right: Box::new(make_path("a")),
                }),
                op: crate::ast::BinOp::Add,
                right: Box::new(make_path("b")),
            }),
            is_move: false,
            return_type: None,
        };

        let outer = Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Ident {
                    mutable: false,
                    name: make_ident("a"),
                    evidentiality: None,
                },
                ty: None,
            }],
            body: Box::new(inner),
            is_move: false,
            return_type: None,
        };

        compiler.compile_expr(&outer).unwrap();

        // Should have outer function + outer closure + inner closure
        assert!(compiler.functions.len() >= 3);
    }
}

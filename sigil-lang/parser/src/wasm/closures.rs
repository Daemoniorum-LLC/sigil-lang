//! Closure compilation.
//!
//! Compiles Sigil closures to WASM with environment capture.

use wasm_encoder::{Instruction, ValType};

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
                        func.push(Instruction::I64Const(0)); // initial capacity hint
                        func.push(Instruction::Call(func_idx));
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

                // Handle cross-module calls to hooks, signals, runtime modules
                // These are compiled as stubs that compile arguments and return a dummy value
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
                    // Compile all arguments
                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    // Return the last argument value or 0 if no args
                    // Most hook/signal functions return their first argument
                    if args.is_empty() {
                        let func = self.current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;
                        func.push(Instruction::I64Const(0));
                    }
                    return Ok(());
                }

                // Check for import function first to get parameter types
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
        // Try builtin method dispatch first (to_string, clone, unwrap, etc.)
        if self.try_compile_builtin_method(receiver, method, args)? {
            return Ok(());
        }

        // Compile receiver as first argument
        self.compile_expr(receiver)?;

        // Compile remaining arguments
        for arg in args {
            self.compile_expr(arg)?;
        }

        // Look up method as function
        if let Some(func_idx) = self.get_func(method) {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::Call(func_idx));
            Ok(())
        } else {
            Err(WasmError::undefined_function(method))
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

        // Handle the method calls (remaining segments) one at a time
        if segments.len() == 1 {
            // No method calls, just compile the receiver
            self.compile_expr(&current_expr)?;
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
            self.compile_expr(&current_expr)?;

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
    pub fn compile_index(&mut self, expr: &Expr, index: &Expr) -> WasmResult<()> {
        // Compile array pointer
        self.compile_expr(expr)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        let arr_idx = func.alloc_local("__index_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));



        // Compile index
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

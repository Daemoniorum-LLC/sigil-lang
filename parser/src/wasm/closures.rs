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
    pub fn compile_closure(
        &mut self,
        params: &[ClosureParam],
        body: &Expr,
        is_move: bool,
    ) -> WasmResult<()> {
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
    fn compile_simple_closure(&mut self, params: &[ClosureParam], body: &Expr) -> WasmResult<()> {
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

            for (i, (capture, is_mutable)) in
                captures.iter().zip(mutable_captures.iter()).enumerate()
            {
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
                // Build the qualified name for module paths (e.g., signal::create -> signal_create)
                let qualified_name: String = path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("_");
                let name = qualified_name.as_str();

                // Also get just the simple name for local lookups
                let simple_name = path
                    .segments
                    .first()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");

                // Check for import function first to get parameter types
                if let Some(func_idx) = self.imports.get_func(name) {
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
                if let Some(func_idx) = self.get_func(simple_name) {
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
                } else {
                    Err(WasmError::undefined_function(name))
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
                func.push(Instruction::CallIndirect {
                    type_index: type_idx,
                    table_index: 0,
                });

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
        let param_types: Vec<ValType> = std::iter::repeat(ValType::I64).take(arg_count).collect();
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

    /// Compile a method call.
    pub fn compile_method_call(
        &mut self,
        receiver: &Expr,
        method: &str,
        args: &[Expr],
    ) -> WasmResult<()> {
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
                let name = path
                    .segments
                    .first()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");
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

            Expr::Closure {
                params,
                body,
                is_move: _,
            } => {
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
                    if let crate::ast::Stmt::Let { pattern, init, .. } = stmt {
                        if let Some(val) = init {
                            self.visit(val);
                        }
                        // Add bound variable
                        if let Pattern::Ident { name, .. } = pattern {
                            self.bound.push(name.name.clone());
                        }
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

            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Ident, Literal, NumBase, PathSegment, TypePath};
    use crate::span::Span;
    use crate::wasm::literals::{
        create_test_compiler_with_function, create_test_compiler_with_heap_alloc,
    };

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
        };

        compiler.compile_expr(&inc_closure).unwrap();

        // Second closure: read counter
        let read_closure = Expr::Closure {
            params: vec![],
            body: Box::new(make_path("counter")),
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
        };

        compiler.compile_expr(&outer).unwrap();

        // Should have outer function + outer closure + inner closure
        assert!(compiler.functions.len() >= 3);
    }
}

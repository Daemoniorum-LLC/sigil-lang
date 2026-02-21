//! Expression compilation.
//!
//! Compiles Sigil expressions to WASM instructions.

use wasm_encoder::{BlockType, Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;
use crate::ast::{Expr, TypePath, UnaryOp};

impl WasmCompiler {
    /// Compile an expression, pushing the result onto the WASM stack.
    pub fn compile_expr(&mut self, expr: &Expr) -> WasmResult<()> {
        match expr {
            // Literals
            Expr::Literal(lit) => self.compile_literal(lit),

            // Variable reference
            Expr::Path(path) => self.compile_path(path),

            // Binary operations
            Expr::Binary { left, op, right } => {
                // Short-circuit evaluation for && and ||
                use crate::ast::BinOp;
                match op {
                    BinOp::And => self.compile_short_circuit_and(left, right),
                    BinOp::Or => self.compile_short_circuit_or(left, right),
                    _ => {
                        // Standard binary: compile operands, then emit operator
                        self.compile_expr(left)?;
                        self.compile_expr(right)?;
                        self.emit_binop(*op)
                    }
                }
            }

            // Unary operations
            Expr::Unary { op, expr } => {
                self.compile_expr(expr)?;
                self.emit_unaryop(*op)
            }

            // Block expression
            Expr::Block(block) => self.compile_block(block),

            // If expression
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => self.compile_if(condition, then_branch, else_branch.as_deref()),

            // While loop
            Expr::While {
                label,
                condition,
                body,
            } => {
                let label_name = label.as_ref().map(|l| l.name.clone());
                self.compile_while(condition, body, label_name)
            }

            // For loop
            Expr::For {
                label,
                pattern,
                iter,
                body,
            } => {
                let label_name = label.as_ref().map(|l| l.name.clone());
                self.compile_for(pattern, iter, body, label_name)
            }

            // Infinite loop
            Expr::Loop { label, body } => {
                let label_name = label.as_ref().map(|l| l.name.clone());
                self.compile_loop(body, label_name)
            }

            // Function call
            Expr::Call { func, args } => self.compile_call(func, args),

            // Method call
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => self.compile_method_call(receiver, &method.name, args),

            // Field access
            Expr::Field { expr, field } => self.compile_field_access(expr, &field.name),

            // Index access
            Expr::Index { expr, index } => self.compile_index(expr, index),

            // Array literal
            Expr::Array(elements) => self.compile_array(elements),

            // Tuple literal
            Expr::Tuple(elements) => self.compile_tuple(elements),

            // Struct literal
            Expr::Struct { path, fields, rest } => {
                self.compile_struct_literal(path, fields, rest.as_deref())
            }

            // Closure
            Expr::Closure {
                params,
                body,
                is_move,
                return_type: _,
            } => self.compile_closure(params, body, *is_move),

            // Pipe expression (morphemes)
            Expr::Pipe { expr, operations } => self.compile_pipe(expr, operations),

            // Morpheme application
            Expr::Morpheme { kind, body } => self.compile_morpheme(*kind, body),

            // Assignment
            Expr::Assign { target, value } => self.compile_assign(target, value),

            // Return
            Expr::Return(value) => self.compile_return(value.as_deref()),

            // Break
            Expr::Break { label, value } => {
                let label_name = label.as_ref().map(|l| l.name.clone());
                self.compile_break(value.as_deref(), label_name)
            }

            // Continue
            Expr::Continue { label } => {
                let label_name = label.as_ref().map(|l| l.name.clone());
                self.compile_continue(label_name)
            }

            // Try operator
            Expr::Try(expr) => self.compile_try(expr),

            // Await
            Expr::Await {
                expr,
                evidentiality,
            } => self.compile_await(expr, *evidentiality),

            // Match
            Expr::Match { expr, arms } => self.compile_match(expr, arms),

            // Range
            Expr::Range {
                start,
                end,
                inclusive,
            } => self.compile_range(start.as_deref(), end.as_deref(), *inclusive),

            // Cast
            Expr::Cast { expr, ty } => self.compile_cast(expr, ty),

            // Let expression (for if-let patterns)
            Expr::Let { pattern, value } => self.compile_let_expr(pattern, value),

            // Evidential marker
            Expr::Evidential {
                expr,
                evidentiality,
            } => self.compile_evidential(expr, *evidentiality),

            // Macro invocation - route to macro compiler
            Expr::Macro { path, tokens } => {
                let macro_name = path.segments
                    .last()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");

                // Try to compile as known macro
                if self.compile_macro(macro_name, tokens)? {
                    Ok(())
                } else {
                    // Unknown macro - check if it's a procedural macro attribute
                    Err(WasmError::unsupported(&format!(
                        "macro '{}!' (procedural macros like #[component] require pre-expansion)",
                        macro_name
                    )))
                }
            }

            // Incorporation chains: expr·method(args)·method2(args2)
            Expr::Incorporation { segments } => self.compile_incorporation(segments),
            // Unsafe blocks are transparent in WASM — just compile the inner block.
            Expr::Unsafe(block) => self.compile_block(block),
            Expr::Deref(_) => Err(WasmError::unsupported("raw pointer dereference")),
            Expr::AddrOf { .. } => Err(WasmError::unsupported("address-of expressions")),
            Expr::InlineAsm(_) => Err(WasmError::unsupported("inline assembly")),
            Expr::VolatileRead { .. } => Err(WasmError::unsupported("volatile read")),
            Expr::VolatileWrite { .. } => Err(WasmError::unsupported("volatile write")),
            Expr::SimdLiteral { .. } => Err(WasmError::unsupported("SIMD literals")),
            Expr::SimdIntrinsic { .. } => Err(WasmError::unsupported("SIMD intrinsics")),
            Expr::SimdShuffle { .. } => Err(WasmError::unsupported("SIMD shuffle")),
            Expr::SimdExtract { .. } => Err(WasmError::unsupported("SIMD extract")),
            Expr::SimdInsert { .. } => Err(WasmError::unsupported("SIMD insert")),
            Expr::AtomicOp { .. } => Err(WasmError::unsupported("atomic operations")),
            Expr::AtomicFence { .. } => Err(WasmError::unsupported("atomic fence")),
            Expr::HttpRequest { .. } => Err(WasmError::unsupported("HTTP requests")),
            Expr::WebSocketConnect { .. } => Err(WasmError::unsupported("WebSocket")),
            Expr::GrpcCall { .. } => Err(WasmError::unsupported("gRPC calls")),
            Expr::SimdSplat { .. } => Err(WasmError::unsupported("SIMD splat")),
            Expr::WebSocketMessage { .. } => Err(WasmError::unsupported("WebSocket message")),
            Expr::KafkaOp { .. } => Err(WasmError::unsupported("Kafka operations")),
            Expr::GraphQLOp { .. } => Err(WasmError::unsupported("GraphQL operations")),
            Expr::ProtocolStream { .. } => Err(WasmError::unsupported("protocol streams")),
            Expr::ArrayRepeat { value, count } => self.compile_array_repeat(value, count),
            Expr::Async { block, .. } => {
                // Stub: compile block synchronously (proper async/await needs JS promise integration)
                self.compile_block(block)
            }
            Expr::LegionFieldVar { .. } => Err(WasmError::unsupported("Legion field variables")),
            Expr::LegionSuperposition { .. } => Err(WasmError::unsupported("Legion superposition")),
            Expr::LegionInterference { .. } => Err(WasmError::unsupported("Legion interference")),
            Expr::LegionResonance { .. } => Err(WasmError::unsupported("Legion resonance")),
            Expr::LegionDistribute { .. } => Err(WasmError::unsupported("Legion distribute")),
            Expr::LegionGather { .. } => Err(WasmError::unsupported("Legion gather")),
            Expr::LegionBroadcast { .. } => Err(WasmError::unsupported("Legion broadcast")),
            Expr::LegionConsensus { .. } => Err(WasmError::unsupported("Legion consensus")),
            Expr::LegionDecay { .. } => Err(WasmError::unsupported("Legion decay")),
            Expr::NamedArg { .. } => Err(WasmError::unsupported("named arguments")),
            Expr::NoGrad(_) => Err(WasmError::unsupported("no_grad blocks")),
            Expr::Attributed { expr, .. } => self.compile_expr(expr),
            Expr::Turbofish { expr, .. } => self.compile_expr(expr),
        }
    }

    /// Compile a path (variable reference).
    fn compile_path(&mut self, path: &TypePath) -> WasmResult<()> {
        let name = path
            .segments
            .first()
            .map(|s| s.ident.name.as_str())
            .unwrap_or("");

        // Handle Option/Result builtins
        match name {
            "None" => {
                // Create Option::None (discriminant 0, no payload)
                // Allocate 16 bytes for Option struct
                let alloc_idx = self.get_func("heap_alloc")
                    .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(16));
                func.push(Instruction::Call(alloc_idx));

                // Store in temp and write discriminant 0
                let ptr = func.alloc_local("__none_ptr".to_string(), ValType::I64);
                func.push(Instruction::LocalTee(ptr));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Const(0)); // None discriminant
                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
                // Return pointer
                func.push(Instruction::LocalGet(ptr));
                return Ok(());
            }
            "true" => {
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(1));
                return Ok(());
            }
            "false" => {
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(0));
                return Ok(());
            }
            _ => {}
        }

        // Check for enum type reference (for method chaining like WebSocketState·Connecting)
        // Return a placeholder value that will be used by the method chain
        if self.enum_layouts.contains_key(name) {
            let func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            // Return 0 as placeholder - variant construction happens in method chain
            func.push(Instruction::I64Const(0));
            return Ok(());
        }

        // VNode type reference (for builder pattern chaining)
        // When VNode is referenced directly, return a placeholder for type-level operations
        if name == "VNode" {
            let func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            // Return 0 as placeholder - actual construction happens via VNode·div() etc.
            func.push(Instruction::I64Const(0));
            return Ok(());
        }

        // Check local variables first
        if let Some(func) = self.current_function() {
            if let Some(local) = func.get_local(name) {
                let index = local.index;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(index));
                return Ok(());
            }

            // Check for mutable capture (cell indirection)
            let cell_name = format!("__cell_{}", name);
            if let Some(cell_local) = func.get_local(&cell_name) {
                let index = cell_local.index;
                let func = self.current_function_mut().unwrap();
                // Load cell pointer, then load value from cell
                func.push(Instruction::LocalGet(index));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
                return Ok(());
            }
        }

        // Check globals
        if let Some(idx) = self.get_global(name) {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::GlobalGet(idx));
            return Ok(());
        }

        // Check for function reference (for function pointers)
        if let Some(_func_idx) = self.get_func(name) {
            // Return function table index for indirect calls
            return Err(WasmError::unsupported("function references"));
        }

        // Handle multi-segment paths like typography·FONT_SANS or api·ConnectionState·Connected
        // Try building qualified names and looking them up
        if path.segments.len() > 1 {
            let segments: Vec<&str> = path.segments.iter()
                .map(|s| s.ident.name.as_str())
                .collect();

            // For 3+ segment paths like api·ConnectionState·Connected:
            // - Last segment is the variant name
            // - Second-to-last is the enum name
            // - Rest are module path
            if segments.len() >= 2 {
                let variant_name = segments[segments.len() - 1];
                let enum_name = segments[segments.len() - 2];

                // Try to find the enum by various qualified paths
                // First: direct enum name (if enum was imported)
                if let Some(layout) = self.enum_layouts.get(enum_name).cloned() {
                    if let Some(tag) = layout.variant_tag(variant_name) {
                        let func = self
                            .current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;
                        func.push(Instruction::I64Const(tag as i64));
                        return Ok(());
                    }
                }

                // Second: module-qualified enum (e.g., api::ConnectionState)
                let enum_qualified = segments[..segments.len()-1].join("::");
                if let Some(layout) = self.enum_layouts.get(&enum_qualified).cloned() {
                    if let Some(tag) = layout.variant_tag(variant_name) {
                        let func = self
                            .current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;
                        func.push(Instruction::I64Const(tag as i64));
                        return Ok(());
                    }
                }

                // Third: try with full current module prefix
                if !self.module_path.is_empty() {
                    let full_enum = format!("{}::{}", self.current_module_prefix(), enum_qualified);
                    if let Some(layout) = self.enum_layouts.get(&full_enum).cloned() {
                        if let Some(tag) = layout.variant_tag(variant_name) {
                            let func = self
                                .current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::I64Const(tag as i64));
                            return Ok(());
                        }
                    }
                }
            }

            // Try full path as global (e.g., typography::FONT_SANS)
            let qualified = segments.join("::");
            if let Some(idx) = self.get_global(&qualified) {
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::GlobalGet(idx));
                return Ok(());
            }

            // Try with current module prefix
            if !self.module_path.is_empty() {
                let full_path = format!("{}::{}", self.current_module_prefix(), qualified);
                if let Some(idx) = self.get_global(&full_path) {
                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::GlobalGet(idx));
                    return Ok(());
                }
            }

            // Try as function call (for static methods like VNode::div)
            if let Some(func_idx) = self.get_func(&qualified) {
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Call(func_idx));
                return Ok(());
            }
        }

        Err(WasmError::undefined_variable(name))
    }

    /// Short-circuit AND (&&): left && right
    fn compile_short_circuit_and(&mut self, left: &Expr, right: &Expr) -> WasmResult<()> {
        // Compile: if left { right } else { 0 }
        self.compile_expr(left)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Convert to i32 for branch
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::If(BlockType::Result(ValType::I64)));

        // Then branch: evaluate right
        self.compile_expr(right)?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Else);
        // Else branch: false (0)
        func.push(Instruction::I64Const(0));
        func.push(Instruction::End);

        Ok(())
    }

    /// Short-circuit OR (||): left || right
    fn compile_short_circuit_or(&mut self, left: &Expr, right: &Expr) -> WasmResult<()> {
        // Compile: if left { 1 } else { right }
        self.compile_expr(left)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Convert to i32 for branch
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::If(BlockType::Result(ValType::I64)));

        // Then branch: true (1)
        func.push(Instruction::I64Const(1));
        func.push(Instruction::Else);

        // Else branch: evaluate right
        self.compile_expr(right)?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::End);

        Ok(())
    }

    /// Compile assignment expression.
    pub fn compile_assign(&mut self, target: &Expr, value: &Expr) -> WasmResult<()> {
        match target {
            // Simple variable assignment
            Expr::Path(path) => {
                let name = path
                    .segments
                    .first()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");

                // Check if it's a local or mutable capture
                let local_index = self
                    .current_function()
                    .and_then(|f| f.get_local(name).map(|l| l.index));
                let cell_name = format!("__cell_{}", name);
                let cell_index = self
                    .current_function()
                    .and_then(|f| f.get_local(&cell_name).map(|l| l.index));

                if let Some(index) = local_index {
                    // Direct local assignment
                    self.compile_expr(value)?;
                    let func = self.current_function_mut().unwrap();
                    // Duplicate the value (assignment is an expression)
                    func.push(Instruction::LocalTee(index));
                    return Ok(());
                }

                if let Some(cell_index) = cell_index {
                    // Mutable capture (cell indirection)
                    let func = self.current_function_mut().unwrap();

                    // Get cell pointer
                    func.push(Instruction::LocalGet(cell_index));
                    func.push(Instruction::I32WrapI64);

                    // Compile the value
                    self.compile_expr(value)?;

                    let func = self.current_function_mut().unwrap();

                    // Store value to cell
                    // Stack is [cell_ptr, value], we need [value, cell_ptr, value]
                    // Store a local copy to return later
                    let value_temp = func.alloc_local("__assign_temp".to_string(), ValType::I64);
                    func.push(Instruction::LocalTee(value_temp));

                    // Now stack is [cell_ptr, value]
                    func.push(Instruction::I64Store(wasm_encoder::MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    }));

                    // Return the value (assignment is an expression)
                    func.push(Instruction::LocalGet(value_temp));

                    return Ok(());
                }

                // Compile the value first for global assignment
                self.compile_expr(value)?;

                // Check if it's a global
                if let Some(idx) = self.get_global(name) {
                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    // Duplicate value, set global, leave value on stack
                    let temp = idx; // Use global's own index as temp for now
                    func.push(Instruction::GlobalSet(temp));
                    func.push(Instruction::GlobalGet(temp));
                    return Ok(());
                }

                Err(WasmError::undefined_variable(name))
            }

            // Field assignment
            Expr::Field { expr, field } => self.compile_field_assign(expr, &field.name, value),

            // Index assignment
            Expr::Index { expr: array, index } => self.compile_index_assign(array, index, value),

            // Dereference assignment: *ptr = value
            Expr::Unary { op: UnaryOp::Deref, expr } => {
                // Compile the pointer expression (the thing being dereferenced)
                self.compile_expr(expr)?;

                // Convert to i32 for memory addressing
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I32WrapI64);

                // Compile the value
                self.compile_expr(value)?;

                // Store at the dereferenced address
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));

                // Assignment is an expression, return unit (0)
                func.push(Instruction::I64Const(0));
                Ok(())
            }

            _ => Err(WasmError::invalid_assignment_target()),
        }
    }

    /// Compile field assignment.
    fn compile_field_assign(&mut self, target: &Expr, field: &str, value: &Expr) -> WasmResult<()> {
        // Check for actor self.field assignment
        if let Expr::Path(path) = target {
            if path.segments.len() == 1 {
                let name = &path.segments[0].ident.name;
                if name == "self" {
                    // Inside an actor method, self.field = value -> store to actor global
                    if let Some(actor_name) = &self.current_actor.clone() {
                        let global_name = format!("{}_{}", actor_name, field);
                        if let Some(idx) = self.get_global(&global_name) {
                            // Compile value
                            self.compile_expr(value)?;
                            let func = self
                                .current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::GlobalSet(idx));
                            // Assignment returns unit
                            func.push(Instruction::I64Const(0));
                            return Ok(());
                        }
                        // Try qualified name
                        let qualified = self.qualify_name(&global_name);
                        if let Some(idx) = self.get_global(&qualified) {
                            // Compile value
                            self.compile_expr(value)?;
                            let func = self
                                .current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::GlobalSet(idx));
                            // Assignment returns unit
                            func.push(Instruction::I64Const(0));
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Regular struct field assignment
        // If target is a known-type local variable, use its type for offset resolution.
        let receiver_struct_type = if let Expr::Path(path) = target {
            if path.segments.len() == 1 {
                let var_name = &path.segments[0].ident.name;
                self.local_var_types.get(var_name.as_str()).cloned()
            } else {
                None
            }
        } else {
            None
        };

        // Get struct pointer (i64), then wrap to i32 for memory addressing
        self.compile_expr(target)?;
        {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I32WrapI64);
        }

        // Get field offset, preferring the known receiver type when available.
        let offset = if let Some(ref type_name) = receiver_struct_type {
            if let Some(layout) = self.struct_layouts.get(type_name.as_str()) {
                layout.field_offset(field).unwrap_or_else(|| self.get_field_offset(field).unwrap_or(0))
            } else {
                self.get_field_offset(field)?
            }
        } else {
            self.get_field_offset(field)?
        };

        // Compile value
        self.compile_expr(value)?;

        // Store to memory: stack is [i32_addr, i64_value]
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        func.push(Instruction::I64Store(wasm_encoder::MemArg {
            offset: offset as u64,
            align: 3, // 8-byte alignment
            memory_index: 0,
        }));

        // Assignment returns unit (0) for now
        func.push(Instruction::I64Const(0));

        Ok(())
    }

    /// Compile index assignment.
    fn compile_index_assign(&mut self, array: &Expr, index: &Expr, value: &Expr) -> WasmResult<()> {
        // Get array pointer
        self.compile_expr(array)?;

        // Get index
        self.compile_expr(index)?;

        // Calculate offset: ptr + (index * 8)
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        func.push(Instruction::I64Const(8));
        func.push(Instruction::I64Mul);
        func.push(Instruction::I64Add);

        // Convert to i32 for memory operations
        func.push(Instruction::I32WrapI64);

        // Compile value
        self.compile_expr(value)?;

        // Store
        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I64Store(wasm_encoder::MemArg {
            offset: 8, // Skip length field
            align: 3,
            memory_index: 0,
        }));

        // Return the value
        self.compile_expr(value)
    }

    /// Get field offset from struct layout.
    ///
    /// Prefers the layout of the struct currently being compiled (`current_impl_type`)
    /// so that identically-named fields in different structs (e.g. `children` exists
    /// in both VElement at offset 32 and VFragment at offset 0) resolve to the correct
    /// offset rather than a random one chosen by HashMap iteration order.
    pub fn get_field_offset(&self, field: &str) -> WasmResult<u32> {
        // 1. Prefer the current impl type's layout (deterministic, context-aware).
        if let Some(impl_type) = &self.current_impl_type {
            if let Some(layout) = self.struct_layouts.get(impl_type.as_str()) {
                if let Some(offset) = layout.field_offset(field) {
                    return Ok(offset);
                }
            }
        }

        // 2. Fall back to searching all layouts (for non-self field accesses).
        for layout in self.struct_layouts.values() {
            if let Some(offset) = layout.field_offset(field) {
                return Ok(offset);
            }
        }

        // Default to 0 if not found (will be fixed with proper type info)
        Ok(0)
    }

    // Placeholder methods for other expression types
    // These will be implemented in their respective modules

    fn compile_return(&mut self, value: Option<&Expr>) -> WasmResult<()> {
        if let Some(expr) = value {
            self.compile_expr(expr)?;
        } else {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I64Const(0));
        }

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Return);
        Ok(())
    }

    fn compile_break(&mut self, value: Option<&Expr>, label: Option<String>) -> WasmResult<()> {
        if let Some(expr) = value {
            self.compile_expr(expr)?;
        }

        // Find the target loop context
        let (break_label, depth_adjustment) = if let Some(label_name) = label {
            // Find the loop with the matching label
            let mut found = None;
            for (i, ctx) in self.loop_stack.iter().rev().enumerate() {
                if ctx.name.as_ref() == Some(&label_name) {
                    // Each nested loop adds 2 to the relative depth (block + loop)
                    found = Some((ctx.break_label + (i as u32 * 2), i));
                    break;
                }
            }
            found.ok_or_else(|| WasmError::undefined_label(&label_name))?
        } else {
            // No label - use innermost loop
            let ctx = self
                .loop_stack
                .last()
                .ok_or_else(|| WasmError::not_in_loop("break"))?;
            (ctx.break_label, 0)
        };

        let _ = depth_adjustment; // Used in calculation above
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        func.push(Instruction::Br(break_label));
        Ok(())
    }

    fn compile_continue(&mut self, label: Option<String>) -> WasmResult<()> {
        // Find the target loop context
        let continue_label = if let Some(label_name) = label {
            // Find the loop with the matching label
            let mut found = None;
            for (i, ctx) in self.loop_stack.iter().rev().enumerate() {
                if ctx.name.as_ref() == Some(&label_name) {
                    // Each nested loop adds 2 to the relative depth (block + loop)
                    found = Some(ctx.continue_label + (i as u32 * 2));
                    break;
                }
            }
            found.ok_or_else(|| WasmError::undefined_label(&label_name))?
        } else {
            // No label - use innermost loop
            let ctx = self
                .loop_stack
                .last()
                .ok_or_else(|| WasmError::not_in_loop("continue"))?;
            ctx.continue_label
        };

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        func.push(Instruction::Br(continue_label));
        Ok(())
    }

    fn compile_try(&mut self, _expr: &Expr) -> WasmResult<()> {
        Err(WasmError::unsupported("try operator (?)"))
    }

    /// Compile an await expression.
    ///
    /// Await expressions suspend the current execution until the promise resolves.
    /// The JS runtime handles the actual suspension/resumption via the await_promise import.
    pub fn compile_await(
        &mut self,
        expr: &Expr,
        evidentiality: Option<crate::ast::Evidentiality>,
    ) -> WasmResult<()> {
        // Compile the expression that returns a Promise (as i32 pointer)
        self.compile_expr(expr)?;

        // Convert i64 to i32 pointer (promise is stored as pointer)
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32WrapI64);

        // Get the await_promise function
        let await_fn = self
            .get_func("async_await_promise")
            .ok_or_else(|| WasmError::internal("async_await_promise not found"))?;
        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(await_fn));

        // Result is already i64

        // Apply evidentiality tag if specified
        if let Some(ev) = evidentiality {
            self.apply_evidentiality_tag(ev)?;
        }

        Ok(())
    }

    /// Apply an evidentiality tag to the value on the stack.
    fn apply_evidentiality_tag(&mut self, ev: crate::ast::Evidentiality) -> WasmResult<()> {
        use crate::ast::Evidentiality;

        // Evidentiality is stored in the high bits of the i64 value
        // See constants.rs for the tag values
        let tag: i64 = match ev {
            Evidentiality::Known => 0x0 << 60,     // ! - locally verified
            Evidentiality::Uncertain => 0x1 << 60, // ? - may be absent
            Evidentiality::Reported => 0x2 << 60,  // ~ - external/untrusted
            Evidentiality::Paradox => 0x3 << 60,   // ‽ - trust boundary
            Evidentiality::Predicted => 0x4 << 60, // ◊ - model output, speculative
            Evidentiality::Chaos => 0x5 << 60,     // ⁂ - intentional randomness, entropic
        };

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Mask off existing tag and apply new one
        func.push(Instruction::I64Const(0x0FFF_FFFF_FFFF_FFFF)); // Value mask
        func.push(Instruction::I64And);
        func.push(Instruction::I64Const(tag));
        func.push(Instruction::I64Or);

        Ok(())
    }

    /// Compile a range expression.
    ///
    /// Range expressions compile to two i64 values on the stack: (start, end).
    /// For unbounded ends, we use -1 as a sentinel (meaning "to end").
    /// Inclusive ranges have end adjusted by +1 at compile time.
    fn compile_range(
        &mut self,
        start: Option<&Expr>,
        end: Option<&Expr>,
        inclusive: bool,
    ) -> WasmResult<()> {
        // Compile start value (default to 0)
        if let Some(s) = start {
            self.compile_expr(s)?;
        } else {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I64Const(0));
        }

        // Compile end value (use -1 as sentinel for "to end")
        if let Some(e) = end {
            self.compile_expr(e)?;

            if inclusive {
                // For inclusive ranges, add 1 to end
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(1));
                func.push(Instruction::I64Add);
            }
        } else {
            // Unbounded end: use -1 sentinel
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I64Const(-1));
        }

        Ok(())
    }

    fn compile_cast(&mut self, expr: &Expr, _ty: &crate::ast::TypeExpr) -> WasmResult<()> {
        // For now, just compile the expression (type casts are mostly no-ops in WASM)
        self.compile_expr(expr)
    }

    /// Compile a let expression (pattern matching expression).
    /// Used in `if let Some(x) = value { ... }` patterns.
    /// Returns 1 (true) if pattern matches, 0 (false) otherwise.
    /// Also binds matched values to locals for use in subsequent code.
    fn compile_let_expr(
        &mut self,
        pattern: &crate::ast::Pattern,
        value: &Expr,
    ) -> WasmResult<()> {
        use crate::ast::Pattern;

        // Compile the value being matched
        self.compile_expr(value)?;

        match pattern {
            // if let Some(x) = value - check Option discriminant
            Pattern::TupleStruct { path, fields, .. } => {
                let type_name = path.segments.last()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");

                match type_name {
                    "Some" => {
                        // Option: discriminant 1 = Some
                        let func = self.current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;

                        // Store Option pointer
                        let opt_ptr = func.alloc_local("__let_opt".to_string(), ValType::I64);
                        func.push(Instruction::LocalTee(opt_ptr));

                        // Load discriminant
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 3,
                            memory_index: 0,
                        }));

                        // Store discriminant for later
                        let disc = func.alloc_local("__let_disc".to_string(), ValType::I64);
                        func.push(Instruction::LocalSet(disc));

                        // If pattern has bindings, extract the value
                        if let Some(first_field) = fields.first() {
                            let binding_name = match first_field {
                                Pattern::Ident { name, .. } => Some(name.name.clone()),
                                Pattern::RefBinding { name, .. } => Some(name.name.clone()),
                                _ => None,
                            };
                            if let Some(bname) = binding_name {
                                // Load payload from Option (offset 8)
                                func.push(Instruction::LocalGet(opt_ptr));
                                func.push(Instruction::I32WrapI64);
                                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                                    offset: 8,
                                    align: 3,
                                    memory_index: 0,
                                }));

                                // Bind to local variable
                                let binding = func.alloc_local(bname, ValType::I64);
                                func.push(Instruction::LocalSet(binding));
                            }
                        }

                        // Return match result: discriminant == 1
                        func.push(Instruction::LocalGet(disc));
                        func.push(Instruction::I64Const(1));
                        func.push(Instruction::I64Eq);
                        func.push(Instruction::I64ExtendI32U);
                    }
                    "Ok" => {
                        // Result::Ok: discriminant 0
                        let func = self.current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;

                        let result_ptr = func.alloc_local("__let_result".to_string(), ValType::I64);
                        func.push(Instruction::LocalTee(result_ptr));
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 3,
                            memory_index: 0,
                        }));

                        let disc = func.alloc_local("__let_disc".to_string(), ValType::I64);
                        func.push(Instruction::LocalSet(disc));

                        if let Some(first_field) = fields.first() {
                            let binding_name = match first_field {
                                Pattern::Ident { name, .. } => Some(name.name.clone()),
                                Pattern::RefBinding { name, .. } => Some(name.name.clone()),
                                _ => None,
                            };
                            if let Some(bname) = binding_name {
                                func.push(Instruction::LocalGet(result_ptr));
                                func.push(Instruction::I32WrapI64);
                                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                                    offset: 8,
                                    align: 3,
                                    memory_index: 0,
                                }));
                                let binding = func.alloc_local(bname, ValType::I64);
                                func.push(Instruction::LocalSet(binding));
                            }
                        }

                        // Ok = discriminant 0
                        func.push(Instruction::LocalGet(disc));
                        func.push(Instruction::I64Eqz);
                        func.push(Instruction::I64ExtendI32U);
                    }
                    "Err" => {
                        // Result::Err: discriminant 1
                        let func = self.current_function_mut()
                            .ok_or_else(|| WasmError::internal("not in function context"))?;

                        let result_ptr = func.alloc_local("__let_result".to_string(), ValType::I64);
                        func.push(Instruction::LocalTee(result_ptr));
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: 0,
                            align: 3,
                            memory_index: 0,
                        }));

                        let disc = func.alloc_local("__let_disc".to_string(), ValType::I64);
                        func.push(Instruction::LocalSet(disc));

                        if let Some(first_field) = fields.first() {
                            let binding_name = match first_field {
                                Pattern::Ident { name, .. } => Some(name.name.clone()),
                                Pattern::RefBinding { name, .. } => Some(name.name.clone()),
                                _ => None,
                            };
                            if let Some(bname) = binding_name {
                                func.push(Instruction::LocalGet(result_ptr));
                                func.push(Instruction::I32WrapI64);
                                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                                    offset: 8,
                                    align: 3,
                                    memory_index: 0,
                                }));
                                let binding = func.alloc_local(bname, ValType::I64);
                                func.push(Instruction::LocalSet(binding));
                            }
                        }

                        // Err = discriminant != 0
                        func.push(Instruction::LocalGet(disc));
                        func.push(Instruction::I64Const(0));
                        func.push(Instruction::I64Ne);
                        func.push(Instruction::I64ExtendI32U);
                    }
                    _ => {
                        // Other enum variants - check enum layouts
                        if let Some(layout) = self.enum_layouts.get(type_name).cloned() {
                            if let Some(tag) = layout.variant_tag(type_name) {
                                let func = self.current_function_mut()
                                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                                func.push(Instruction::I64Const(tag as i64));
                                func.push(Instruction::I64Eq);
                                func.push(Instruction::I64ExtendI32U);
                            } else {
                                // Fallback: always match
                                let func = self.current_function_mut()
                                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                                func.push(Instruction::Drop);
                                func.push(Instruction::I64Const(1));
                            }
                        } else {
                            // Unknown variant, always match
                            let func = self.current_function_mut()
                                .ok_or_else(|| WasmError::internal("not in function context"))?;
                            func.push(Instruction::Drop);
                            func.push(Instruction::I64Const(1));
                        }
                    }
                }
            }
            // Simple binding: let x = value (always matches)
            Pattern::Ident { name, .. } => {
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let local = func.alloc_local(name.name.clone(), ValType::I64);
                func.push(Instruction::LocalSet(local));
                func.push(Instruction::I64Const(1)); // Always matches
            }
            // Wildcard: _ = value (always matches)
            Pattern::Wildcard => {
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Drop);
                func.push(Instruction::I64Const(1));
            }
            _ => {
                // Other patterns not yet supported
                return Err(WasmError::unsupported("complex let patterns"));
            }
        }

        Ok(())
    }

    fn compile_evidential(
        &mut self,
        expr: &Expr,
        _evidentiality: crate::ast::Evidentiality,
    ) -> WasmResult<()> {
        // Compile the expression, evidentiality is metadata
        self.compile_expr(expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Ident, Literal, NumBase, PathSegment};
    use crate::span::Span;
    use crate::wasm::literals::create_test_compiler_with_function;
    use wasm_encoder::ValType;

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

    #[test]
    fn test_compile_literal_expr() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_expr(&make_int(42)).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(42)));
    }

    #[test]
    fn test_compile_binary_add() {
        let mut compiler = create_test_compiler_with_function();

        let expr = Expr::Binary {
            left: Box::new(make_int(10)),
            op: BinOp::Add,
            right: Box::new(make_int(20)),
        };

        compiler.compile_expr(&expr).unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 3);
        assert!(matches!(func.instructions[0], Instruction::I64Const(10)));
        assert!(matches!(func.instructions[1], Instruction::I64Const(20)));
        assert!(matches!(func.instructions[2], Instruction::I64Add));
    }

    #[test]
    fn test_compile_local_variable() {
        let mut compiler = create_test_compiler_with_function();

        // Add a local variable
        {
            let func = compiler.current_function_mut().unwrap();
            func.alloc_local("x".to_string(), ValType::I64);
        }

        compiler.compile_expr(&make_path("x")).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::LocalGet(0))); // index 0 (first local, no params)
    }

    #[test]
    fn test_compile_undefined_variable() {
        let mut compiler = create_test_compiler_with_function();

        let result = compiler.compile_expr(&make_path("undefined_var"));
        assert!(result.is_err());
    }

    // Helper to create a compiler with full imports (for async tests)
    fn create_full_compiler_with_function() -> WasmCompiler {
        let mut compiler = WasmCompiler::new();
        let type_idx = compiler.get_or_create_type(vec![], vec![ValType::I64]);
        let func_idx = compiler.imports.import_count();
        let func = crate::wasm::types::CompiledFunction::new(
            "test".to_string(),
            type_idx,
            func_idx,
            vec![],
            vec![ValType::I64],
            false,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);
        compiler
    }

    #[test]
    fn test_compile_await_expression() {
        let mut compiler = create_full_compiler_with_function();

        // await some_promise - should compile to: compile expr, wrap i32, call await_promise
        let await_expr = Expr::Await {
            expr: Box::new(make_int(123)), // Pretend this is a promise pointer
            evidentiality: None,
        };

        compiler.compile_expr(&await_expr).unwrap();

        let func = compiler.current_function().unwrap();
        // Should have: I64Const(123), I32WrapI64, Call(await_promise)
        assert!(func.instructions.len() >= 3);
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32WrapI64)));
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }

    #[test]
    fn test_compile_await_with_evidentiality() {
        use crate::ast::Evidentiality;

        let mut compiler = create_full_compiler_with_function();

        // await? some_promise - should apply Uncertain evidentiality tag
        let await_expr = Expr::Await {
            expr: Box::new(make_int(456)),
            evidentiality: Some(Evidentiality::Uncertain),
        };

        compiler.compile_expr(&await_expr).unwrap();

        let func = compiler.current_function().unwrap();
        // Should include I64And and I64Or for evidentiality tagging
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64And)));
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Or)));
    }

    #[test]
    fn test_compile_sequential_awaits() {
        let mut compiler = create_full_compiler_with_function();

        // First await
        let await1 = Expr::Await {
            expr: Box::new(make_int(100)),
            evidentiality: None,
        };
        compiler.compile_expr(&await1).unwrap();

        // Get instruction count after first await
        let first_count = compiler.current_function().unwrap().instructions.len();

        // Second await
        let await2 = Expr::Await {
            expr: Box::new(make_int(200)),
            evidentiality: None,
        };
        compiler.compile_expr(&await2).unwrap();

        let func = compiler.current_function().unwrap();
        // Should have more instructions after second await
        assert!(func.instructions.len() > first_count);

        // Should have two Call instructions (for two await_promise calls)
        let call_count = func
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(call_count, 2);
    }

    #[test]
    fn test_compile_short_circuit_and() {
        let mut compiler = create_test_compiler_with_function();

        let expr = Expr::Binary {
            left: Box::new(make_int(1)),
            op: BinOp::And,
            right: Box::new(make_int(2)),
        };

        compiler.compile_expr(&expr).unwrap();

        let func = compiler.current_function().unwrap();
        // Should have if/else structure
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::If(_))));
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Else)));
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::End)));
    }

    #[test]
    fn test_compile_short_circuit_or() {
        let mut compiler = create_test_compiler_with_function();

        let expr = Expr::Binary {
            left: Box::new(make_int(0)),
            op: BinOp::Or,
            right: Box::new(make_int(1)),
        };

        compiler.compile_expr(&expr).unwrap();

        let func = compiler.current_function().unwrap();
        // Should have if/else structure
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::If(_))));
    }

    #[test]
    fn test_compile_return_with_value() {
        let mut compiler = create_test_compiler_with_function();

        let expr = Expr::Return(Some(Box::new(make_int(42))));

        compiler.compile_expr(&expr).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(42)));
        assert!(matches!(func.instructions[1], Instruction::Return));
    }

    #[test]
    fn test_compile_return_no_value() {
        let mut compiler = create_test_compiler_with_function();

        let expr = Expr::Return(None);

        compiler.compile_expr(&expr).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(0)));
        assert!(matches!(func.instructions[1], Instruction::Return));
    }
}

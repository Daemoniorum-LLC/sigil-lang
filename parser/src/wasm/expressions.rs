//! Expression compilation.
//!
//! Compiles Sigil expressions to WASM instructions.

use wasm_encoder::{BlockType, Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;
use crate::ast::{Expr, TypePath};

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
                return_type: _,
                body,
                is_move,
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

            // Unsupported for now
            Expr::Incorporation { .. } => Err(WasmError::unsupported("incorporation expressions")),
            Expr::Macro { .. } => Err(WasmError::unsupported("macro expressions")),
            Expr::Unsafe(_) => Err(WasmError::unsupported("unsafe blocks")),
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
            Expr::ArrayRepeat { .. } => Err(WasmError::unsupported("array repeat [value; count]")),
            Expr::Async { .. } => Err(WasmError::unsupported("async blocks")),
            Expr::LegionFieldVar { .. } => Err(WasmError::unsupported("Legion field variables")),
            Expr::LegionSuperposition { .. } => Err(WasmError::unsupported("Legion superposition")),
            Expr::LegionInterference { .. } => Err(WasmError::unsupported("Legion interference")),
            Expr::LegionResonance { .. } => Err(WasmError::unsupported("Legion resonance")),
            Expr::LegionDistribute { .. } => Err(WasmError::unsupported("Legion distribute")),
            Expr::LegionGather { .. } => Err(WasmError::unsupported("Legion gather")),
            Expr::LegionBroadcast { .. } => Err(WasmError::unsupported("Legion broadcast")),
            Expr::LegionConsensus { .. } => Err(WasmError::unsupported("Legion consensus")),
            Expr::LegionDecay { .. } => Err(WasmError::unsupported("Legion decay")),

            // Named arguments - handled in call compilation
            Expr::NamedArg { .. } => Err(WasmError::unsupported(
                "named arguments outside of function calls",
            )),

            // Template expressions (i18n)
            Expr::Template(_) => Err(WasmError::unsupported("template expressions")),
            Expr::TemplateFragment { .. } => Err(WasmError::unsupported("template fragments")),
        }
    }

    /// Compile a path (variable reference).
    fn compile_path(&mut self, path: &TypePath) -> WasmResult<()> {
        let name = path
            .segments
            .first()
            .map(|s| s.ident.name.as_str())
            .unwrap_or("");

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

            _ => Err(WasmError::invalid_assignment_target()),
        }
    }

    /// Compile field assignment.
    fn compile_field_assign(&mut self, target: &Expr, field: &str, value: &Expr) -> WasmResult<()> {
        // Get struct pointer
        self.compile_expr(target)?;

        // Get field offset (need type info)
        // For now, use a simple offset calculation
        let _offset = self.get_field_offset(field)?;

        // Compile value
        self.compile_expr(value)?;

        // Store to memory
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Stack: [ptr, value] -> need to store value at ptr+offset
        // This is simplified - real implementation needs type info
        func.push(Instruction::I64Store(wasm_encoder::MemArg {
            offset: 0,
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
    pub fn get_field_offset(&self, field: &str) -> WasmResult<u32> {
        // Check all struct layouts for this field
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
    fn compile_await(
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

    fn compile_range(
        &mut self,
        _start: Option<&Expr>,
        _end: Option<&Expr>,
        _inclusive: bool,
    ) -> WasmResult<()> {
        Err(WasmError::unsupported("range expressions"))
    }

    fn compile_cast(&mut self, expr: &Expr, _ty: &crate::ast::TypeExpr) -> WasmResult<()> {
        // For now, just compile the expression (type casts are mostly no-ops in WASM)
        self.compile_expr(expr)
    }

    fn compile_let_expr(
        &mut self,
        _pattern: &crate::ast::Pattern,
        _value: &Expr,
    ) -> WasmResult<()> {
        Err(WasmError::unsupported("let expressions"))
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

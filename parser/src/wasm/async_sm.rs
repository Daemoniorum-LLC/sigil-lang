//! Async State Machine Generation
//!
//! Transforms async functions with multiple await points into state machines.
//! Each await becomes a yield point where the function can suspend and resume.
//!
//! # Strategy
//!
//! An async function like:
//! ```sigil
//! async rite fetch_both(url1: str, url2: str) -> (Data, Data) {
//!     ≔ a = fetch(url1)|await;
//!     ≔ b = fetch(url2)|await;
//!     (a, b)
//! }
//! ```
//!
//! Is transformed into a state machine:
//! ```wasm
//! ;; State 0: Initial - start first fetch
//! ;; State 1: After first await - start second fetch
//! ;; State 2: After second await - return result
//! ```
//!
//! The state machine stores:
//! - Current state number
//! - Saved local variables
//! - Intermediate results
//!
//! When an await is encountered:
//! 1. Save current state and locals to memory
//! 2. Call create_continuation with resumption point
//! 3. Return the promise
//!
//! When resumed:
//! 1. Load state and locals from memory
//! 2. Continue execution from saved state
//!
//! # Compilation Modes
//!
//! Two modes are supported:
//! - **Asyncify mode** (default): Uses `await_promise` import, relies on runtime stack switching
//! - **State machine mode**: Explicit transformation, works on any WASM runtime

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;
use crate::ast::{Block, Expr, Function, Pattern, Stmt};

#[cfg(feature = "wasm")]
use wasm_encoder::{Instruction, ValType};

/// Information about an await point in an async function
#[derive(Debug, Clone)]
pub struct AwaitPoint {
    /// Index of this await point (0-based)
    pub index: usize,
    /// Offset into the state frame for saved locals
    pub frame_offset: u32,
    /// Variables that need to be saved at this point
    pub saved_locals: Vec<String>,
    /// Variables that are live after this await (need to be restored)
    pub live_after: Vec<String>,
}

/// State machine representation of an async function
#[derive(Debug)]
pub struct AsyncStateMachine {
    /// Original function name
    pub name: String,
    /// All await points in order
    pub await_points: Vec<AwaitPoint>,
    /// Size of the state frame in bytes
    pub frame_size: u32,
    /// Number of local variables to save
    pub num_saved_locals: usize,
    /// All local variables declared in the function
    pub all_locals: Vec<String>,
}

/// Frame layout for async state machine:
/// - Offset 0: state number (i32, 4 bytes)
/// - Offset 4: padding (4 bytes for alignment)
/// - Offset 8: saved locals (8 bytes each, i64)
/// - After locals: intermediate values
pub const STATE_OFFSET: u32 = 0;
pub const LOCALS_OFFSET: u32 = 8;

impl WasmCompiler {
    /// Analyze an async function to find await points and build state machine info
    pub fn analyze_async_function(&self, func: &Function) -> Option<AsyncStateMachine> {
        if !func.is_async {
            return None;
        }

        let mut await_points = Vec::new();
        let mut frame_offset = LOCALS_OFFSET;

        // Collect parameter names
        let param_names: Vec<String> = func.params
            .iter()
            .filter_map(|p| {
                if let Pattern::Ident { name, .. } = &p.pattern {
                    Some(name.name.clone())
                } else {
                    None
                }
            })
            .collect();

        // Collect all local variable declarations from body
        let mut all_locals = param_names.clone();
        if let Some(ref body) = func.body {
            self.collect_local_declarations(body, &mut all_locals);
        }

        // Find all await expressions
        if let Some(ref body) = func.body {
            self.find_await_points(body, &mut await_points, &all_locals, &mut frame_offset);
        }

        if await_points.is_empty() {
            // No await points - can use simple async compilation
            return None;
        }

        // Calculate frame size:
        // - 8 bytes for state (i32 state + 4 byte padding)
        // - 8 bytes per local variable
        let num_saved_locals = all_locals.len();
        let frame_size = LOCALS_OFFSET + (num_saved_locals as u32 * 8);

        Some(AsyncStateMachine {
            name: func.name.name.clone(),
            await_points,
            frame_size,
            num_saved_locals,
            all_locals,
        })
    }

    /// Collect all local variable declarations from a block
    fn collect_local_declarations(&self, block: &Block, locals: &mut Vec<String>) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { pattern, .. } | Stmt::LetElse { pattern, .. } => {
                    self.collect_pattern_names(pattern, locals);
                }
                _ => {}
            }
        }
        // Check nested blocks in expressions
        if let Some(expr) = &block.expr {
            self.collect_locals_in_expr(expr, locals);
        }
    }

    /// Collect variable names from a pattern
    fn collect_pattern_names(&self, pattern: &Pattern, names: &mut Vec<String>) {
        match pattern {
            Pattern::Ident { name, .. } => {
                if !names.contains(&name.name) {
                    names.push(name.name.clone());
                }
            }
            Pattern::Tuple(elements) => {
                for elem in elements {
                    self.collect_pattern_names(elem, names);
                }
            }
            Pattern::Struct { fields, .. } => {
                for field in fields {
                    if let Some(ref pat) = field.pattern {
                        self.collect_pattern_names(pat, names);
                    } else {
                        // Field shorthand: `{ name }` binds `name`
                        if !names.contains(&field.name.name) {
                            names.push(field.name.name.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Collect locals declared in nested expressions
    fn collect_locals_in_expr(&self, expr: &Expr, locals: &mut Vec<String>) {
        match expr {
            Expr::Block(block) => {
                self.collect_local_declarations(block, locals);
            }
            Expr::If { then_branch, else_branch, .. } => {
                self.collect_local_declarations(then_branch, locals);
                if let Some(else_expr) = else_branch {
                    self.collect_locals_in_expr(else_expr, locals);
                }
            }
            Expr::Match { arms, .. } => {
                for arm in arms {
                    self.collect_pattern_names(&arm.pattern, locals);
                    self.collect_locals_in_expr(&arm.body, locals);
                }
            }
            Expr::While { body, .. } | Expr::Loop { body, .. } => {
                self.collect_local_declarations(body, locals);
            }
            Expr::For { pattern, body, .. } => {
                self.collect_pattern_names(pattern, locals);
                self.collect_local_declarations(body, locals);
            }
            _ => {}
        }
    }

    /// Recursively find await expressions in a block
    fn find_await_points(
        &self,
        block: &Block,
        points: &mut Vec<AwaitPoint>,
        saved_locals: &[String],
        frame_offset: &mut u32,
    ) {
        for stmt in &block.stmts {
            self.find_await_in_stmt(stmt, points, saved_locals, frame_offset);
        }
        if let Some(expr) = &block.expr {
            self.find_await_in_expr(expr, points, saved_locals, frame_offset);
        }
    }

    fn find_await_in_stmt(
        &self,
        stmt: &Stmt,
        points: &mut Vec<AwaitPoint>,
        saved_locals: &[String],
        frame_offset: &mut u32,
    ) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr(expr) | Stmt::Semi(expr) => {
                self.find_await_in_expr(expr, points, saved_locals, frame_offset);
            }
            Stmt::LetElse { init, else_branch, .. } => {
                self.find_await_in_expr(init, points, saved_locals, frame_offset);
                self.find_await_in_expr(else_branch, points, saved_locals, frame_offset);
            }
            _ => {}
        }
    }

    fn find_await_in_expr(
        &self,
        expr: &Expr,
        points: &mut Vec<AwaitPoint>,
        saved_locals: &[String],
        frame_offset: &mut u32,
    ) {
        match expr {
            Expr::Await { .. } => {
                // All locals are conservatively considered live after await
                // A more sophisticated analysis would compute actual liveness
                points.push(AwaitPoint {
                    index: points.len(),
                    frame_offset: *frame_offset,
                    saved_locals: saved_locals.to_vec(),
                    live_after: saved_locals.to_vec(),
                });
                *frame_offset += (saved_locals.len() as u32 * 8) + 8;
            }
            Expr::Binary { left, right, .. } => {
                self.find_await_in_expr(left, points, saved_locals, frame_offset);
                self.find_await_in_expr(right, points, saved_locals, frame_offset);
            }
            Expr::Call { func, args } => {
                self.find_await_in_expr(func, points, saved_locals, frame_offset);
                for arg in args {
                    self.find_await_in_expr(arg, points, saved_locals, frame_offset);
                }
            }
            Expr::If { condition, then_branch, else_branch } => {
                self.find_await_in_expr(condition, points, saved_locals, frame_offset);
                self.find_await_points(then_branch, points, saved_locals, frame_offset);
                if let Some(else_expr) = else_branch {
                    self.find_await_in_expr(else_expr, points, saved_locals, frame_offset);
                }
            }
            Expr::Block(block) => {
                self.find_await_points(block, points, saved_locals, frame_offset);
            }
            Expr::Pipe { expr, operations } => {
                self.find_await_in_expr(expr, points, saved_locals, frame_offset);
                for op in operations {
                    if let crate::ast::PipeOp::Await { .. } = op {
                        points.push(AwaitPoint {
                            index: points.len(),
                            frame_offset: *frame_offset,
                            live_after: saved_locals.to_vec(),
                            saved_locals: saved_locals.to_vec(),
                        });
                        *frame_offset += (saved_locals.len() as u32 * 8) + 8;
                    }
                }
            }
            Expr::While { condition, body, .. } => {
                self.find_await_in_expr(condition, points, saved_locals, frame_offset);
                self.find_await_points(body, points, saved_locals, frame_offset);
            }
            Expr::Loop { body, .. } => {
                self.find_await_points(body, points, saved_locals, frame_offset);
            }
            Expr::For { iter, body, .. } => {
                self.find_await_in_expr(iter, points, saved_locals, frame_offset);
                self.find_await_points(body, points, saved_locals, frame_offset);
            }
            Expr::Unary { expr, .. } => {
                self.find_await_in_expr(expr, points, saved_locals, frame_offset);
            }
            Expr::Tuple(elements) | Expr::Array(elements) => {
                for elem in elements {
                    self.find_await_in_expr(elem, points, saved_locals, frame_offset);
                }
            }
            Expr::Match { expr, arms } => {
                self.find_await_in_expr(expr, points, saved_locals, frame_offset);
                for arm in arms {
                    self.find_await_in_expr(&arm.body, points, saved_locals, frame_offset);
                }
            }
            _ => {}
        }
    }

    /// Compile an async function as a state machine
    ///
    /// For functions with multiple await points, this generates:
    /// 1. A check for initial call vs resume (based on hidden first parameter)
    /// 2. A state dispatcher at function entry (br_table)
    /// 3. State blocks for each segment between awaits
    /// 4. Save/restore code for locals at each await boundary
    ///
    /// # Runtime Contract
    ///
    /// The runtime must:
    /// - On initial call: pass 0 as the hidden first parameter
    /// - On resume: pass the frame_ptr (from continuation) as first parameter,
    ///   and the resolved value in the second parameter slot
    ///
    /// The function returns either:
    /// - A continuation (promise) if suspended at an await
    /// - The final result if completed
    #[cfg(feature = "wasm")]
    pub fn compile_async_state_machine(
        &mut self,
        func: &Function,
        sm: &AsyncStateMachine,
    ) -> WasmResult<()> {
        // For single await or no awaits, use simple await_promise approach
        if sm.await_points.len() <= 1 {
            return Ok(());
        }

        // Get required imports
        let alloc_idx = self.get_func("alloc")
            .ok_or_else(|| WasmError::internal("alloc import not found"))?;
        let create_continuation_idx = self.get_func("async_create_continuation")
            .ok_or_else(|| WasmError::internal("async_create_continuation import not found"))?;

        let compiled_func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Allocate locals for state machine internals
        let frame_ptr_local = compiled_func.alloc_local("__sm_frame".to_string(), ValType::I32);
        let state_local = compiled_func.alloc_local("__sm_state".to_string(), ValType::I32);
        let resume_value_local = compiled_func.alloc_local("__sm_resume".to_string(), ValType::I64);
        let is_initial_local = compiled_func.alloc_local("__sm_is_initial".to_string(), ValType::I32);

        // Allocate locals for each saved variable (these mirror the frame slots)
        let mut local_indices: Vec<(String, u32)> = Vec::new();
        for local_name in &sm.all_locals {
            let idx = compiled_func.alloc_local(local_name.clone(), ValType::I64);
            local_indices.push((local_name.clone(), idx));
        }

        // === PROLOGUE: Check if initial call or resume ===
        // Convention: First parameter of the original function is repurposed
        // If it's 0, this is initial call; otherwise it's frame_ptr for resume
        //
        // We use the first param slot (local 0) as the frame_ptr indicator
        // Runtime passes: initial call -> 0, resume -> frame_ptr

        // Check if local 0 is 0 (initial call)
        compiled_func.push(Instruction::LocalGet(0)); // First param
        compiled_func.push(Instruction::I32Eqz);
        compiled_func.push(Instruction::LocalSet(is_initial_local));

        // Branch based on initial vs resume
        compiled_func.push(Instruction::LocalGet(is_initial_local));
        compiled_func.push(Instruction::If(wasm_encoder::BlockType::Empty));

        // === INITIAL CALL PATH ===
        // Allocate new frame
        compiled_func.push(Instruction::I32Const(sm.frame_size as i32));
        compiled_func.push(Instruction::Call(alloc_idx));
        compiled_func.push(Instruction::LocalSet(frame_ptr_local));

        // Initialize state to 0
        compiled_func.push(Instruction::LocalGet(frame_ptr_local));
        compiled_func.push(Instruction::I32Const(0));
        compiled_func.push(Instruction::I32Store(wasm_encoder::MemArg {
            offset: STATE_OFFSET as u64,
            align: 2,
            memory_index: 0,
        }));

        // Save original parameters to frame
        for (i, (_, local_idx)) in local_indices.iter().enumerate() {
            // Only save actual params (first N locals match params)
            if i < func.params.len() {
                compiled_func.push(Instruction::LocalGet(frame_ptr_local));
                // Get the original param value (offset by our internal locals)
                compiled_func.push(Instruction::LocalGet(*local_idx));
                compiled_func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: (LOCALS_OFFSET + (i as u32 * 8)) as u64,
                    align: 3,
                    memory_index: 0,
                }));
            }
        }

        compiled_func.push(Instruction::Else);

        // === RESUME PATH ===
        // Use passed frame_ptr (from local 0)
        compiled_func.push(Instruction::LocalGet(0));
        compiled_func.push(Instruction::LocalSet(frame_ptr_local));

        // Get resume value from local 1 (second param on resume)
        compiled_func.push(Instruction::LocalGet(1));
        compiled_func.push(Instruction::LocalSet(resume_value_local));

        // Restore all locals from frame
        for (i, (_, local_idx)) in local_indices.iter().enumerate() {
            compiled_func.push(Instruction::LocalGet(frame_ptr_local));
            compiled_func.push(Instruction::I64Load(wasm_encoder::MemArg {
                offset: (LOCALS_OFFSET + (i as u32 * 8)) as u64,
                align: 3,
                memory_index: 0,
            }));
            compiled_func.push(Instruction::LocalSet(*local_idx));
        }

        compiled_func.push(Instruction::End); // End if/else

        // === Load current state for dispatcher ===
        compiled_func.push(Instruction::LocalGet(frame_ptr_local));
        compiled_func.push(Instruction::I32Load(wasm_encoder::MemArg {
            offset: STATE_OFFSET as u64,
            align: 2,
            memory_index: 0,
        }));
        compiled_func.push(Instruction::LocalSet(state_local));

        // === STATE DISPATCHER ===
        // Generate nested blocks for br_table targets
        let num_states = sm.await_points.len() + 1; // +1 for final state

        // Structure: block $final { block $stateN { ... block $state0 { br_table } } }
        for _i in 0..num_states {
            compiled_func.push(Instruction::Block(wasm_encoder::BlockType::Empty));
        }

        // br_table dispatches to correct state
        compiled_func.push(Instruction::LocalGet(state_local));
        let targets: Vec<u32> = (0..num_states as u32).collect();
        compiled_func.push(Instruction::BrTable(
            targets[..num_states-1].to_vec().into(),
            (num_states - 1) as u32, // default to final state
        ));

        // === STATE BLOCKS ===
        // State 0 is entry, states 1..N are resume points after awaits
        for state_idx in 0..num_states {
            // End the block for this state (br_table jumps here)
            compiled_func.push(Instruction::End);

            if state_idx < sm.await_points.len() {
                let await_point = &sm.await_points[state_idx];

                // For resume states (> 0), the resume_value contains the await result
                // The body compilation needs to bind this to the appropriate variable
                // For now, we just have the framework in place

                // TODO: Here we would emit the body code for this state segment
                // This requires splitting the AST at await boundaries
                // For now, this is a placeholder

                // === AWAIT POINT: Save and suspend ===
                // Save all locals to frame before suspending
                for (i, local_name) in await_point.saved_locals.iter().enumerate() {
                    if let Some((_, local_idx)) = local_indices.iter().find(|(n, _)| n == local_name) {
                        compiled_func.push(Instruction::LocalGet(frame_ptr_local));
                        compiled_func.push(Instruction::LocalGet(*local_idx));
                        compiled_func.push(Instruction::I64Store(wasm_encoder::MemArg {
                            offset: (LOCALS_OFFSET + (i as u32 * 8)) as u64,
                            align: 3,
                            memory_index: 0,
                        }));
                    }
                }

                // Set next state
                compiled_func.push(Instruction::LocalGet(frame_ptr_local));
                compiled_func.push(Instruction::I32Const((state_idx + 1) as i32));
                compiled_func.push(Instruction::I32Store(wasm_encoder::MemArg {
                    offset: STATE_OFFSET as u64,
                    align: 2,
                    memory_index: 0,
                }));

                // Create continuation and return it
                // create_continuation(frame_ptr, next_state) -> continuation_ptr
                compiled_func.push(Instruction::LocalGet(frame_ptr_local));
                compiled_func.push(Instruction::I32Const((state_idx + 1) as i32));
                compiled_func.push(Instruction::Call(create_continuation_idx));

                // Return continuation as i64
                compiled_func.push(Instruction::I64ExtendI32U);
                compiled_func.push(Instruction::Return);
            } else {
                // === FINAL STATE: Complete execution ===
                // The resume_value from the last await is the input to final computation
                // For now, just return the resume value as the result
                compiled_func.push(Instruction::LocalGet(resume_value_local));
            }
        }

        Ok(())
    }

    /// Generate code to save all locals to the state frame
    #[cfg(feature = "wasm")]
    pub fn emit_save_locals(
        &mut self,
        frame_ptr_local: u32,
        locals: &[(String, u32)],
    ) -> WasmResult<()> {
        let compiled_func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        for (i, (_, local_idx)) in locals.iter().enumerate() {
            compiled_func.push(Instruction::LocalGet(frame_ptr_local));
            compiled_func.push(Instruction::LocalGet(*local_idx));
            compiled_func.push(Instruction::I64Store(wasm_encoder::MemArg {
                offset: (LOCALS_OFFSET + (i as u32 * 8)) as u64,
                align: 3,
                memory_index: 0,
            }));
        }

        Ok(())
    }

    /// Generate code to restore all locals from the state frame
    #[cfg(feature = "wasm")]
    pub fn emit_restore_locals(
        &mut self,
        frame_ptr_local: u32,
        locals: &[(String, u32)],
    ) -> WasmResult<()> {
        let compiled_func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        for (i, (_, local_idx)) in locals.iter().enumerate() {
            compiled_func.push(Instruction::LocalGet(frame_ptr_local));
            compiled_func.push(Instruction::I64Load(wasm_encoder::MemArg {
                offset: (LOCALS_OFFSET + (i as u32 * 8)) as u64,
                align: 3,
                memory_index: 0,
            }));
            compiled_func.push(Instruction::LocalSet(*local_idx));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::span::Span;

    fn make_ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            evidentiality: None,
            affect: None,
            span: Span::new(0, 0),
        }
    }

    fn make_type_path(name: &str) -> TypePath {
        TypePath {
            segments: vec![PathSegment {
                ident: make_ident(name),
                generics: None,
            }],
        }
    }

    fn make_param(name: &str) -> Param {
        Param {
            pattern: Pattern::Ident {
                mutable: false,
                name: make_ident(name),
                evidentiality: None,
            },
            ty: TypeExpr::Path(make_type_path("i64")),
        }
    }

    fn make_path_expr(name: &str) -> Expr {
        Expr::Path(make_type_path(name))
    }

    fn make_await_expr() -> Expr {
        Expr::Await {
            expr: Box::new(Expr::Call {
                func: Box::new(make_path_expr("fetch")),
                args: vec![],
            }),
            evidentiality: None,
        }
    }

    fn make_function(
        name: &str,
        params: Vec<Param>,
        body: Block,
        is_async: bool,
    ) -> Function {
        Function {
            doc_comments: vec![],
            visibility: Visibility::Private,
            is_async,
            is_const: false,
            is_unsafe: false,
            attrs: FunctionAttrs::default(),
            name: make_ident(name),
            aspect: None,
            generics: None,
            params,
            return_type: None,
            where_clause: None,
            body: Some(body),
        }
    }

    #[test]
    fn test_analyze_async_no_awaits() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "simple",
            vec![],
            Block {
                stmts: vec![],
                expr: Some(Box::new(Expr::Literal(Literal::Int {
                    value: "42".to_string(),
                    base: NumBase::Decimal,
                    suffix: None,
                }))),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_none()); // No await points
    }

    #[test]
    fn test_analyze_async_single_await() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "fetch_one",
            vec![make_param("url")],
            Block {
                stmts: vec![],
                expr: Some(Box::new(make_await_expr())),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        assert_eq!(sm.await_points.len(), 1);
        assert_eq!(sm.num_saved_locals, 1); // url parameter
    }

    #[test]
    fn test_analyze_async_multiple_awaits() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "fetch_two",
            vec![make_param("url1"), make_param("url2")],
            Block {
                stmts: vec![
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("a"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(make_await_expr()),
                    },
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("b"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(make_await_expr()),
                    },
                ],
                expr: Some(Box::new(Expr::Tuple(vec![
                    make_path_expr("a"),
                    make_path_expr("b"),
                ]))),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        assert_eq!(sm.await_points.len(), 2);
        // Now collects all locals: params (url1, url2) + body declarations (a, b)
        assert_eq!(sm.num_saved_locals, 4);
        assert_eq!(sm.all_locals, vec!["url1", "url2", "a", "b"]);
    }

    #[test]
    fn test_state_machine_frame_size() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "test",
            vec![make_param("a"), make_param("b"), make_param("c")],
            Block {
                stmts: vec![],
                expr: Some(Box::new(make_await_expr())),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        // 3 locals * 8 bytes + 8 bytes for state = 32 bytes
        assert_eq!(sm.frame_size, 32);
    }

    #[test]
    fn test_await_in_loop() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "poll_until_done",
            vec![make_param("handle")],
            Block {
                stmts: vec![],
                expr: Some(Box::new(Expr::While {
                    label: None,
                    condition: Box::new(Expr::Literal(Literal::Bool(true))),
                    body: Block {
                        stmts: vec![Stmt::Expr(make_await_expr())],
                        expr: None,
                    },
                })),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        // Should find await inside while loop
        assert_eq!(sm.await_points.len(), 1);
    }

    #[test]
    fn test_await_in_nested_structures() {
        let compiler = WasmCompiler::new();
        // Test: async function with await in if-else and match
        let func = make_function(
            "complex_async",
            vec![make_param("flag")],
            Block {
                stmts: vec![
                    // let x = if flag { fetch()|await } else { fetch()|await };
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("x"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(Expr::If {
                            condition: Box::new(make_path_expr("flag")),
                            then_branch: Block {
                                stmts: vec![],
                                expr: Some(Box::new(make_await_expr())),
                            },
                            else_branch: Some(Box::new(Expr::Block(Block {
                                stmts: vec![],
                                expr: Some(Box::new(make_await_expr())),
                            }))),
                        }),
                    },
                ],
                expr: Some(Box::new(make_path_expr("x"))),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        // Should find both awaits in if-else branches
        assert_eq!(sm.await_points.len(), 2);
        // flag + x
        assert_eq!(sm.all_locals.len(), 2);
    }

    #[test]
    fn test_state_machine_all_locals_collected() {
        let compiler = WasmCompiler::new();
        // Test that locals declared at different nesting levels are all collected
        let func = make_function(
            "nested_locals",
            vec![make_param("input")],
            Block {
                stmts: vec![
                    // let a = 1;
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("a"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(Expr::Literal(Literal::Int {
                            value: "1".to_string(),
                            base: NumBase::Decimal,
                            suffix: None,
                        })),
                    },
                    // let b = await;
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("b"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(make_await_expr()),
                    },
                    // let c = await;
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("c"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(make_await_expr()),
                    },
                ],
                expr: Some(Box::new(Expr::Tuple(vec![
                    make_path_expr("a"),
                    make_path_expr("b"),
                    make_path_expr("c"),
                ]))),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();

        // Should have 2 await points
        assert_eq!(sm.await_points.len(), 2);

        // Should collect all locals: input (param), a, b, c
        assert_eq!(sm.all_locals.len(), 4);
        assert!(sm.all_locals.contains(&"input".to_string()));
        assert!(sm.all_locals.contains(&"a".to_string()));
        assert!(sm.all_locals.contains(&"b".to_string()));
        assert!(sm.all_locals.contains(&"c".to_string()));

        // Frame size: 8 (state) + 4*8 (locals) = 40 bytes
        assert_eq!(sm.frame_size, 40);
    }
}

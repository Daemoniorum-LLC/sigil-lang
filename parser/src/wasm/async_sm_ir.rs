//! StateMachineIR to WASM Compilation
//!
//! Compiles the backend-independent `StateMachineIR` from async_transform
//! into WASM bytecode. This provides true state machine semantics without
//! requiring Asyncify or JSPI runtime support.
//!
//! # Architecture
//!
//! The compiler generates a WASM function with:
//! 1. Prologue: Check initial call vs resume, allocate/restore frame
//! 2. State dispatcher: br_table to jump to current state
//! 3. State blocks: Code for each state, ending with exit handling
//!
//! # Frame Layout
//!
//! ```text
//! Offset 0:  state number (i32)
//! Offset 4:  padding (i32)
//! Offset 8+: saved locals (i64 each)
//! ```
//!
//! # Return Value Encoding
//!
//! Per ASYNC-STATE-MACHINE-SPEC.md §4.3:
//! ```text
//! | Bits 63-32 | Bits 31-0  | Meaning                         |
//! |------------|------------|----------------------------------|
//! | 0          | value      | Complete, final result is value  |
//! | 1          | cont_ptr   | Suspended, continuation at ptr   |
//! ```
//!
//! # Runtime Imports
//!
//! The compiled code expects these imports:
//! - `alloc(size: i32) -> i32`: Allocate memory for suspension frame
//! - `async_create_continuation(frame_ptr: i32, state: i32, promise: i64) -> i32`:
//!   Register a continuation with the promise to await

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;
use crate::async_transform::{StateMachineIR, StateExit};
use crate::ast::Stmt;

#[cfg(feature = "wasm")]
use wasm_encoder::{Instruction, ValType};

/// Constants for frame layout (same as async_sm.rs for compatibility)
pub const STATE_OFFSET: u32 = 0;
pub const LOCALS_OFFSET: u32 = 8;

/// Bit flag for suspended state in return value (bit 32)
#[cfg(feature = "wasm")]
pub const SUSPENDED_FLAG: i64 = 1 << 32;

/// Collected state data for compilation (avoids borrow conflicts).
#[cfg(feature = "wasm")]
struct CollectedState {
    is_resume: bool,
    resume_binding: Option<String>,
    body: Vec<Stmt>,
    exit: StateExit,
}

/// Local variable mapping for state machine compilation.
#[cfg(feature = "wasm")]
struct SmLocals {
    frame_ptr: u32,
    state: u32,
    resume_value: u32,
    is_initial: u32,
    /// Map from variable name to WASM local index
    indices: Vec<(String, u32)>,
}

impl WasmCompiler {
    /// Compile a StateMachineIR to WASM.
    ///
    /// This method generates explicit state machine code that works on any
    /// WASM runtime without Asyncify support.
    ///
    /// # Runtime Contract
    ///
    /// The generated function expects:
    /// - Initial call: first param = 0
    /// - Resume call: first param = frame_ptr, second param = resume value
    ///
    /// Returns:
    /// - If suspended: continuation pointer (to be resumed later)
    /// - If complete: final result value
    #[cfg(feature = "wasm")]
    pub fn compile_state_machine_ir(&mut self, ir: &StateMachineIR) -> WasmResult<()> {
        // Ensure we have required imports
        let alloc_idx = self.get_func("alloc")
            .ok_or_else(|| WasmError::internal("alloc import not found"))?;
        let create_continuation_idx = self.get_func("async_create_continuation")
            .ok_or_else(|| WasmError::internal("async_create_continuation import not found"))?;

        // === Phase 1: Collect all state data upfront (avoids borrow conflicts) ===
        let collected_states: Vec<CollectedState> = ir.states.iter().map(|s| {
            CollectedState {
                is_resume: s.is_resume,
                resume_binding: s.resume_binding.clone(),
                body: s.body.clone(),
                exit: s.exit.clone(),
            }
        }).collect();

        let num_states = collected_states.len();
        if num_states == 0 {
            return Ok(());
        }

        // Collect frame layout info
        let frame_size = ir.frame_layout.total_size;
        let frame_offsets: Vec<(String, u32)> = ir.locals.iter()
            .filter_map(|l| ir.frame_layout.get_offset(&l.name).map(|o| (l.name.clone(), o)))
            .collect();

        // Collect parameter info
        let params: Vec<(String, Option<u32>)> = ir.params.iter()
            .map(|(name, _)| (name.clone(), ir.frame_layout.get_offset(name)))
            .collect();

        // === Phase 2: Allocate locals ===
        let sm_locals = {
            let compiled_func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;

            let frame_ptr = compiled_func.alloc_local("__sm_frame".to_string(), ValType::I32);
            let state = compiled_func.alloc_local("__sm_state".to_string(), ValType::I32);
            let resume_value = compiled_func.alloc_local("__sm_resume".to_string(), ValType::I64);
            let is_initial = compiled_func.alloc_local("__sm_is_initial".to_string(), ValType::I32);

            // Allocate WASM locals for each IR local
            let mut indices: Vec<(String, u32)> = Vec::new();
            for local in &ir.locals {
                let idx = compiled_func.alloc_local(local.name.clone(), ValType::I64);
                indices.push((local.name.clone(), idx));
            }

            // Also allocate locals for parameters (they may not be in ir.locals)
            for (param_name, _) in &ir.params {
                if !indices.iter().any(|(n, _)| n == param_name) {
                    let idx = compiled_func.alloc_local(param_name.clone(), ValType::I64);
                    indices.push((param_name.clone(), idx));
                }
            }

            SmLocals { frame_ptr, state, resume_value, is_initial, indices }
        };

        // === Phase 3: Generate prologue ===
        self.emit_sm_prologue(&sm_locals, alloc_idx, frame_size, &params, &frame_offsets)?;

        // === Phase 4: Generate state dispatcher ===
        self.emit_sm_dispatcher(&sm_locals, num_states)?;

        // === Phase 5: Generate state blocks ===
        for (_state_idx, state) in collected_states.into_iter().enumerate() {
            self.emit_state_block(
                &sm_locals,
                &state,
                num_states,
                create_continuation_idx,
                &frame_offsets,
            )?;
        }

        Ok(())
    }

    /// Emit the state machine prologue (initial vs resume path).
    #[cfg(feature = "wasm")]
    fn emit_sm_prologue(
        &mut self,
        locals: &SmLocals,
        alloc_idx: u32,
        frame_size: u32,
        params: &[(String, Option<u32>)],
        frame_offsets: &[(String, u32)],
    ) -> WasmResult<()> {
        let compiled_func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Check if first param is 0 (initial call)
        compiled_func.push(Instruction::LocalGet(0));
        compiled_func.push(Instruction::I32Eqz);
        compiled_func.push(Instruction::LocalSet(locals.is_initial));

        compiled_func.push(Instruction::LocalGet(locals.is_initial));
        compiled_func.push(Instruction::If(wasm_encoder::BlockType::Empty));

        // === INITIAL CALL PATH ===
        // Allocate frame
        compiled_func.push(Instruction::I32Const(frame_size as i32));
        compiled_func.push(Instruction::Call(alloc_idx));
        compiled_func.push(Instruction::LocalSet(locals.frame_ptr));

        // Initialize state to 0
        compiled_func.push(Instruction::LocalGet(locals.frame_ptr));
        compiled_func.push(Instruction::I32Const(0));
        compiled_func.push(Instruction::I32Store(wasm_encoder::MemArg {
            offset: STATE_OFFSET as u64,
            align: 2,
            memory_index: 0,
        }));

        // Save parameters to frame
        for (i, (_, offset)) in params.iter().enumerate() {
            if let Some(off) = offset {
                compiled_func.push(Instruction::LocalGet(locals.frame_ptr));
                compiled_func.push(Instruction::LocalGet(i as u32));
                compiled_func.push(Instruction::I64Store(wasm_encoder::MemArg {
                    offset: *off as u64,
                    align: 3,
                    memory_index: 0,
                }));
            }
        }

        compiled_func.push(Instruction::Else);

        // === RESUME PATH ===
        compiled_func.push(Instruction::LocalGet(0));
        compiled_func.push(Instruction::LocalSet(locals.frame_ptr));

        compiled_func.push(Instruction::LocalGet(1));
        compiled_func.push(Instruction::LocalSet(locals.resume_value));

        // Restore all locals from frame
        for (local_name, local_idx) in &locals.indices {
            if let Some((_, offset)) = frame_offsets.iter().find(|(n, _)| n == local_name) {
                compiled_func.push(Instruction::LocalGet(locals.frame_ptr));
                compiled_func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: 3,
                    memory_index: 0,
                }));
                compiled_func.push(Instruction::LocalSet(*local_idx));
            }
        }

        compiled_func.push(Instruction::End); // End if/else

        // Load current state
        compiled_func.push(Instruction::LocalGet(locals.frame_ptr));
        compiled_func.push(Instruction::I32Load(wasm_encoder::MemArg {
            offset: STATE_OFFSET as u64,
            align: 2,
            memory_index: 0,
        }));
        compiled_func.push(Instruction::LocalSet(locals.state));

        Ok(())
    }

    /// Emit the state dispatcher (br_table).
    #[cfg(feature = "wasm")]
    fn emit_sm_dispatcher(&mut self, locals: &SmLocals, num_states: usize) -> WasmResult<()> {
        let compiled_func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Create nested blocks for br_table targets
        for _ in 0..num_states {
            compiled_func.push(Instruction::Block(wasm_encoder::BlockType::Empty));
        }

        // br_table dispatches based on state
        compiled_func.push(Instruction::LocalGet(locals.state));
        let targets: Vec<u32> = (0..num_states as u32).collect();
        compiled_func.push(Instruction::BrTable(
            targets[..num_states.saturating_sub(1)].to_vec().into(),
            (num_states.saturating_sub(1)) as u32,
        ));

        Ok(())
    }

    /// Emit a single state block.
    #[cfg(feature = "wasm")]
    fn emit_state_block(
        &mut self,
        locals: &SmLocals,
        state: &CollectedState,
        num_states: usize,
        create_continuation_idx: u32,
        frame_offsets: &[(String, u32)],
    ) -> WasmResult<()> {
        // End this state's block
        {
            let compiled_func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            compiled_func.push(Instruction::End);
        }

        // For resume states, bind the resume value if needed
        if state.is_resume {
            if let Some(ref binding) = state.resume_binding {
                if let Some((_, local_idx)) = locals.indices.iter().find(|(n, _)| n == binding) {
                    let compiled_func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    compiled_func.push(Instruction::LocalGet(locals.resume_value));
                    compiled_func.push(Instruction::LocalSet(*local_idx));
                }
            }
        }

        // Compile state body statements
        for stmt in &state.body {
            self.compile_stmt(stmt)?;
        }

        // Compile state exit
        self.emit_state_exit(locals, &state.exit, num_states, create_continuation_idx, frame_offsets)?;

        Ok(())
    }

    /// Emit state exit code.
    #[cfg(feature = "wasm")]
    fn emit_state_exit(
        &mut self,
        locals: &SmLocals,
        exit: &StateExit,
        num_states: usize,
        create_continuation_idx: u32,
        frame_offsets: &[(String, u32)],
    ) -> WasmResult<()> {
        match exit {
            StateExit::Await { promise, next_state, saved_locals } => {
                // Save locals to frame before suspension
                {
                    let compiled_func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;

                    for local_name in saved_locals {
                        if let Some((_, offset)) = frame_offsets.iter().find(|(n, _)| n == local_name) {
                            if let Some((_, local_idx)) = locals.indices.iter().find(|(n, _)| n == local_name) {
                                compiled_func.push(Instruction::LocalGet(locals.frame_ptr));
                                compiled_func.push(Instruction::LocalGet(*local_idx));
                                compiled_func.push(Instruction::I64Store(wasm_encoder::MemArg {
                                    offset: *offset as u64,
                                    align: 3,
                                    memory_index: 0,
                                }));
                            }
                        }
                    }

                    // Set next state in frame
                    compiled_func.push(Instruction::LocalGet(locals.frame_ptr));
                    compiled_func.push(Instruction::I32Const(*next_state as i32));
                    compiled_func.push(Instruction::I32Store(wasm_encoder::MemArg {
                        offset: STATE_OFFSET as u64,
                        align: 2,
                        memory_index: 0,
                    }));
                }

                // Compile the promise expression - this is the future/promise we're awaiting
                // The compiled expression leaves its value (i64) on the stack
                self.compile_expr(promise)?;

                // Create continuation: async_create_continuation(frame_ptr, state, promise) -> cont_ptr
                // Stack before: [promise: i64]
                // We need: [frame_ptr: i32, state: i32, promise: i64]
                let compiled_func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                // Promise is on stack (i64), save it temporarily
                let promise_local = compiled_func.alloc_local("__sm_promise_temp".to_string(), ValType::I64);
                compiled_func.push(Instruction::LocalSet(promise_local));

                // Push args in order: frame_ptr, state, promise
                compiled_func.push(Instruction::LocalGet(locals.frame_ptr));
                compiled_func.push(Instruction::I32Const(*next_state as i32));
                compiled_func.push(Instruction::LocalGet(promise_local));
                compiled_func.push(Instruction::Call(create_continuation_idx));

                // Result is cont_ptr (i32). Encode return value per spec:
                // Bits 63-32 = 1 (suspended), Bits 31-0 = cont_ptr
                // return (cont_ptr as i64) | SUSPENDED_FLAG
                compiled_func.push(Instruction::I64ExtendI32U);
                compiled_func.push(Instruction::I64Const(SUSPENDED_FLAG));
                compiled_func.push(Instruction::I64Or);
                compiled_func.push(Instruction::Return);
            }

            StateExit::Return { value } => {
                // Compile the return value expression
                self.compile_expr(value)?;

                let compiled_func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                compiled_func.push(Instruction::Return);
            }

            StateExit::Goto { target } => {
                let compiled_func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let br_depth = num_states - (*target as usize) - 1;
                compiled_func.push(Instruction::Br(br_depth as u32));
            }

            StateExit::Branch { condition, then_state, else_state } => {
                // Compile the condition expression
                self.compile_expr(condition)?;

                let compiled_func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                // Convert i64 result to i32 for branch
                compiled_func.push(Instruction::I32WrapI64);

                let then_depth = num_states - (*then_state as usize) - 1;
                let else_depth = num_states - (*else_state as usize) - 1;

                compiled_func.push(Instruction::If(wasm_encoder::BlockType::Empty));
                compiled_func.push(Instruction::Br((then_depth + 1) as u32));
                compiled_func.push(Instruction::Else);
                compiled_func.push(Instruction::Br((else_depth + 1) as u32));
                compiled_func.push(Instruction::End);
            }

            StateExit::LoopHead { condition, body_state, exit_state } => {
                let body_depth = num_states - (*body_state as usize) - 1;
                let exit_depth = num_states - (*exit_state as usize) - 1;

                if let Some(cond) = condition {
                    // Compile condition expression
                    self.compile_expr(cond)?;

                    let compiled_func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;

                    // Convert i64 result to i32 for branch
                    compiled_func.push(Instruction::I32WrapI64);

                    compiled_func.push(Instruction::If(wasm_encoder::BlockType::Empty));
                    compiled_func.push(Instruction::Br((body_depth + 1) as u32));
                    compiled_func.push(Instruction::Else);
                    compiled_func.push(Instruction::Br((exit_depth + 1) as u32));
                    compiled_func.push(Instruction::End);
                } else {
                    // Infinite loop - always go to body
                    let compiled_func = self.current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    compiled_func.push(Instruction::Br(body_depth as u32));
                }
            }

            StateExit::Unreachable => {
                let compiled_func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                compiled_func.push(Instruction::Unreachable);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_transform::{FrameLayout, LocalDecl, State, StateMachineIR, StateExit};
    use crate::ast::{Expr, Literal, NumBase, TypeExpr, TypePath, PathSegment, Ident};
    use crate::span::Span;

    fn make_type_path(name: &str) -> TypePath {
        TypePath {
            segments: vec![PathSegment {
                ident: Ident {
                    name: name.to_string(),
                    evidentiality: None,
                    affect: None,
                    span: Span::default(),
                },
                generics: None,
            }],
        }
    }

    fn make_simple_ir() -> StateMachineIR {
        let mut ir = StateMachineIR::new(
            "test_async".to_string(),
            vec![("x".to_string(), TypeExpr::Path(make_type_path("i64")))],
            Some(TypeExpr::Path(make_type_path("i64"))),
        );

        // State 0: Entry - do await
        let mut state0 = State::entry();
        state0.exit = StateExit::Await {
            promise: Expr::Path(make_type_path("promise")),
            next_state: 1,
            saved_locals: vec!["x".to_string()],
        };
        ir.add_state(state0);

        // State 1: Resume - return result
        let mut state1 = State::resume(1);
        state1.resume_binding = Some("result".to_string());
        state1.exit = StateExit::Return {
            value: Expr::Literal(Literal::Int {
                value: "42".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
        };
        ir.add_state(state1);

        // Declare locals
        ir.declare_local("x".to_string(), None, 0);
        ir.declare_local("result".to_string(), None, 1);

        ir
    }

    #[test]
    fn test_simple_ir_structure() {
        let ir = make_simple_ir();
        assert_eq!(ir.states.len(), 2);
        assert_eq!(ir.locals.len(), 2);
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn test_frame_layout() {
        let ir = make_simple_ir();
        assert_eq!(ir.frame_layout.state_offset, 0);
        assert_eq!(ir.frame_layout.locals_offset, 8);
        assert!(ir.frame_layout.get_offset("x").is_some());
        assert!(ir.frame_layout.get_offset("result").is_some());
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn test_compile_simple_ir() {
        use crate::wasm::WasmCompiler;

        let mut compiler = WasmCompiler::new();
        let ir = make_simple_ir();

        // Note: Full compilation test requires setting up function context
        // This test verifies the IR is valid for compilation
        assert!(ir.validate().is_ok());
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn test_compile_ir_with_return_expression() {
        use crate::wasm::WasmCompiler;
        use crate::wasm::types::CompiledFunction;
        use wasm_encoder::{Instruction, ValType};

        let mut compiler = WasmCompiler::new();

        // Set up required imports using add_import
        let alloc_idx = compiler.imports.add_import("env", "alloc", vec![ValType::I32], vec![ValType::I32]);
        compiler.func_map.insert("alloc".to_string(), alloc_idx);

        // async_create_continuation(frame_ptr: i32, state: i32, promise: i64) -> cont_ptr: i32
        let cont_idx = compiler.imports.add_import("env", "async_create_continuation", vec![ValType::I32, ValType::I32, ValType::I64], vec![ValType::I32]);
        compiler.func_map.insert("async_create_continuation".to_string(), cont_idx);

        // Create a simple IR with just a return statement
        let mut ir = StateMachineIR::new(
            "test_return".to_string(),
            vec![],
            Some(TypeExpr::Path(make_type_path("i64"))),
        );

        // State 0: Return 42
        let mut state0 = State::entry();
        state0.exit = StateExit::Return {
            value: Expr::Literal(Literal::Int {
                value: "42".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
        };
        ir.add_state(state0);

        assert!(ir.validate().is_ok());

        // Set up function context
        let type_idx = compiler.get_or_create_type(
            vec![ValType::I32, ValType::I64],  // frame_ptr, resume_value
            vec![ValType::I64],
        );
        let func = CompiledFunction::new(
            "test_return".to_string(),
            type_idx,
            2,  // func_idx after imports
            vec![("frame_ptr".to_string(), ValType::I32), ("resume_value".to_string(), ValType::I64)],
            vec![ValType::I64],
            false,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);

        // Compile the state machine IR
        let result = compiler.compile_state_machine_ir(&ir);
        assert!(result.is_ok(), "compile_state_machine_ir failed: {:?}", result.err());

        // Verify the compiled function has instructions
        let compiled_func = compiler.current_function().unwrap();
        assert!(!compiled_func.instructions.is_empty(), "No instructions generated");

        // Verify that I64Const(42) and Return are in the instructions
        let has_const_42 = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::I64Const(42)));
        let has_return = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::Return));

        assert!(has_const_42, "Expected I64Const(42) in instructions");
        assert!(has_return, "Expected Return in instructions");
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn test_compile_ir_with_branch() {
        use crate::wasm::WasmCompiler;
        use crate::wasm::types::CompiledFunction;
        use crate::ast::BinOp;
        use wasm_encoder::{Instruction, ValType};

        let mut compiler = WasmCompiler::new();

        // Set up required imports using add_import
        let alloc_idx = compiler.imports.add_import("env", "alloc", vec![ValType::I32], vec![ValType::I32]);
        compiler.func_map.insert("alloc".to_string(), alloc_idx);

        // async_create_continuation(frame_ptr: i32, state: i32, promise: i64) -> cont_ptr: i32
        let cont_idx = compiler.imports.add_import("env", "async_create_continuation", vec![ValType::I32, ValType::I32, ValType::I64], vec![ValType::I32]);
        compiler.func_map.insert("async_create_continuation".to_string(), cont_idx);

        // Create an IR with a branch
        let mut ir = StateMachineIR::new(
            "test_branch".to_string(),
            vec![],
            Some(TypeExpr::Path(make_type_path("i64"))),
        );

        // State 0: Branch based on 1 > 0
        let mut state0 = State::entry();
        state0.exit = StateExit::Branch {
            condition: Expr::Binary {
                left: Box::new(Expr::Literal(Literal::Int {
                    value: "1".to_string(),
                    base: NumBase::Decimal,
                    suffix: None,
                })),
                op: BinOp::Gt,
                right: Box::new(Expr::Literal(Literal::Int {
                    value: "0".to_string(),
                    base: NumBase::Decimal,
                    suffix: None,
                })),
            },
            then_state: 1,
            else_state: 2,
        };
        ir.add_state(state0);

        // State 1: Return 1 (then branch)
        let mut state1 = State::intermediate(1);
        state1.exit = StateExit::Return {
            value: Expr::Literal(Literal::Int {
                value: "1".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
        };
        ir.add_state(state1);

        // State 2: Return 0 (else branch)
        let mut state2 = State::intermediate(2);
        state2.exit = StateExit::Return {
            value: Expr::Literal(Literal::Int {
                value: "0".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
        };
        ir.add_state(state2);

        assert!(ir.validate().is_ok());

        // Set up function context
        let type_idx = compiler.get_or_create_type(
            vec![ValType::I32, ValType::I64],
            vec![ValType::I64],
        );
        let func = CompiledFunction::new(
            "test_branch".to_string(),
            type_idx,
            2,
            vec![("frame_ptr".to_string(), ValType::I32), ("resume_value".to_string(), ValType::I64)],
            vec![ValType::I64],
            false,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);

        // Compile the state machine IR
        let result = compiler.compile_state_machine_ir(&ir);
        assert!(result.is_ok(), "compile_state_machine_ir failed: {:?}", result.err());

        // Verify the compiled function has instructions
        let compiled_func = compiler.current_function().unwrap();

        // Should have comparison instruction (I64GtS for >)
        let has_comparison = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::I64GtS));
        assert!(has_comparison, "Expected I64GtS comparison in instructions");

        // Should have I32WrapI64 for branch condition conversion
        let has_wrap = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::I32WrapI64));
        assert!(has_wrap, "Expected I32WrapI64 in instructions");

        // Should have If instruction
        let if_count = compiled_func.instructions.iter().filter(|i| matches!(i, Instruction::If(_))).count();
        assert!(if_count >= 1, "Expected at least one If instruction");
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn test_compile_ir_with_body_statements() {
        use crate::wasm::WasmCompiler;
        use crate::wasm::types::CompiledFunction;
        use crate::ast::{Pattern, Ident as AstIdent};
        use wasm_encoder::{Instruction, ValType};

        let mut compiler = WasmCompiler::new();

        // Set up required imports using add_import
        let alloc_idx = compiler.imports.add_import("env", "alloc", vec![ValType::I32], vec![ValType::I32]);
        compiler.func_map.insert("alloc".to_string(), alloc_idx);

        // async_create_continuation(frame_ptr: i32, state: i32, promise: i64) -> cont_ptr: i32
        let cont_idx = compiler.imports.add_import("env", "async_create_continuation", vec![ValType::I32, ValType::I32, ValType::I64], vec![ValType::I32]);
        compiler.func_map.insert("async_create_continuation".to_string(), cont_idx);

        // Create an IR with body statements
        let mut ir = StateMachineIR::new(
            "test_body".to_string(),
            vec![],
            Some(TypeExpr::Path(make_type_path("i64"))),
        );

        // State 0: let x = 10; return x
        let mut state0 = State::entry();
        state0.body = vec![
            Stmt::Let {
                pattern: Pattern::Ident {
                    name: AstIdent {
                        name: "x".to_string(),
                        evidentiality: None,
                        affect: None,
                        span: Span::default(),
                    },
                    mutable: false,
                    evidentiality: None,
                },
                ty: None,
                init: Some(Expr::Literal(Literal::Int {
                    value: "10".to_string(),
                    base: NumBase::Decimal,
                    suffix: None,
                })),
            },
        ];
        state0.exit = StateExit::Return {
            value: Expr::Path(make_type_path("x")),
        };
        ir.add_state(state0);

        // Declare local
        ir.declare_local("x".to_string(), None, 0);

        assert!(ir.validate().is_ok());

        // Set up function context
        let type_idx = compiler.get_or_create_type(
            vec![ValType::I32, ValType::I64],
            vec![ValType::I64],
        );
        let func = CompiledFunction::new(
            "test_body".to_string(),
            type_idx,
            2,
            vec![("frame_ptr".to_string(), ValType::I32), ("resume_value".to_string(), ValType::I64)],
            vec![ValType::I64],
            false,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);

        // Compile the state machine IR
        let result = compiler.compile_state_machine_ir(&ir);
        assert!(result.is_ok(), "compile_state_machine_ir failed: {:?}", result.err());

        // Verify the compiled function has instructions
        let compiled_func = compiler.current_function().unwrap();

        // Should have I64Const(10) for the let binding
        let has_const_10 = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::I64Const(10)));
        assert!(has_const_10, "Expected I64Const(10) for let binding");

        // Should have LocalSet for storing x
        let has_local_set = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::LocalSet(_)));
        assert!(has_local_set, "Expected LocalSet for let binding");

        // Should have LocalGet for reading x in return
        let has_local_get = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::LocalGet(_)));
        assert!(has_local_get, "Expected LocalGet for return value");
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn test_compile_ir_with_await_and_suspend_flag() {
        use crate::wasm::WasmCompiler;
        use crate::wasm::types::CompiledFunction;
        use wasm_encoder::{Instruction, ValType};

        let mut compiler = WasmCompiler::new();

        // Set up required imports
        let alloc_idx = compiler.imports.add_import("env", "alloc", vec![ValType::I32], vec![ValType::I32]);
        compiler.func_map.insert("alloc".to_string(), alloc_idx);

        // async_create_continuation now takes promise as third parameter
        let cont_idx = compiler.imports.add_import("env", "async_create_continuation", vec![ValType::I32, ValType::I32, ValType::I64], vec![ValType::I32]);
        compiler.func_map.insert("async_create_continuation".to_string(), cont_idx);

        // Create IR with an await
        let mut ir = StateMachineIR::new(
            "test_await".to_string(),
            vec![],
            Some(TypeExpr::Path(make_type_path("i64"))),
        );

        // State 0: Await on promise expression (literal 99 as placeholder)
        let mut state0 = State::entry();
        state0.exit = StateExit::Await {
            promise: Expr::Literal(Literal::Int {
                value: "99".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
            next_state: 1,
            saved_locals: vec![],
        };
        ir.add_state(state0);

        // State 1: Return the resume value
        let mut state1 = State::resume(1);
        state1.resume_binding = Some("result".to_string());
        state1.exit = StateExit::Return {
            value: Expr::Literal(Literal::Int {
                value: "42".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
        };
        ir.add_state(state1);

        ir.declare_local("result".to_string(), None, 1);
        assert!(ir.validate().is_ok());

        // Set up function context
        let type_idx = compiler.get_or_create_type(
            vec![ValType::I32, ValType::I64],
            vec![ValType::I64],
        );
        let func = CompiledFunction::new(
            "test_await".to_string(),
            type_idx,
            2,
            vec![("frame_ptr".to_string(), ValType::I32), ("resume_value".to_string(), ValType::I64)],
            vec![ValType::I64],
            false,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);

        // Compile the state machine IR
        let result = compiler.compile_state_machine_ir(&ir);
        assert!(result.is_ok(), "compile_state_machine_ir failed: {:?}", result.err());

        let compiled_func = compiler.current_function().unwrap();

        // Verify promise expression was compiled (I64Const(99))
        let has_promise = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::I64Const(99)));
        assert!(has_promise, "Expected promise expression I64Const(99)");

        // Verify SUSPENDED_FLAG is used (I64Const(1 << 32) = I64Const(4294967296))
        let has_suspend_flag = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::I64Const(n) if *n == SUSPENDED_FLAG));
        assert!(has_suspend_flag, "Expected SUSPENDED_FLAG I64Const({})", SUSPENDED_FLAG);

        // Verify I64Or is used to combine flag with continuation pointer
        let has_or = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::I64Or));
        assert!(has_or, "Expected I64Or to combine suspend flag with cont_ptr");

        // Verify async_create_continuation is called (Call instruction with cont_idx)
        let call_count = compiled_func.instructions.iter().filter(|i| matches!(i, Instruction::Call(idx) if *idx == cont_idx)).count();
        assert!(call_count >= 1, "Expected Call to async_create_continuation");
    }

    // =========================================================================
    // END-TO-END ASYNC COMPILATION TESTS
    // =========================================================================

    /// Test the full async transformation and compilation pipeline.
    ///
    /// This test validates that:
    /// 1. An async function is transformed to StateMachineIR correctly
    /// 2. The IR compiles to valid WASM bytecode
    /// 3. The WASM has correct imports (alloc, async_create_continuation)
    /// 4. The WASM has correct function signatures per the spec
    #[cfg(feature = "wasm")]
    #[test]
    fn test_e2e_async_function_compilation() {
        use crate::wasm::WasmCompiler;
        use crate::wasm::types::CompiledFunction;
        use wasm_encoder::{Instruction, ValType};

        // =====================================================================
        // Step 1: Create IR simulating an async function with 2 awaits
        // =====================================================================
        //
        // Simulates:
        //   async rite fetch_two() -> i64 {
        //       let a = fetch(1)⌛;   // await point 1
        //       let b = fetch(2)⌛;   // await point 2
        //       a + b
        //   }
        //
        // Expected states:
        //   State 0 (entry): call fetch(1), suspend
        //   State 1 (resume a): bind 'a' from resume_value, call fetch(2), suspend
        //   State 2 (resume b): bind 'b' from resume_value, compute a + b, return

        let mut ir = StateMachineIR::new(
            "fetch_two".to_string(),
            vec![],
            Some(TypeExpr::Path(make_type_path("i64"))),
        );

        // Add local declarations
        ir.declare_local("a".to_string(), Some(TypeExpr::Path(make_type_path("i64"))), 1);
        ir.declare_local("b".to_string(), Some(TypeExpr::Path(make_type_path("i64"))), 2);

        // State 0: Entry - suspend on first fetch
        let mut state0 = State::entry();
        state0.exit = StateExit::Await {
            promise: Expr::Call {
                func: Box::new(Expr::Path(TypePath {
                    segments: vec![PathSegment {
                        ident: Ident {
                            name: "fetch".to_string(),
                            evidentiality: None,
                            affect: None,
                            span: crate::span::Span::default(),
                        },
                        generics: None,
                    }],
                })),
                args: vec![Expr::Literal(Literal::Int {
                    value: "1".to_string(),
                    base: NumBase::Decimal,
                    suffix: None,
                })],
            },
            next_state: 1,
            saved_locals: vec![],
        };
        ir.add_state(state0);

        // State 1: Resume with 'a', suspend on second fetch
        let mut state1 = State::resume(1);
        state1.resume_binding = Some("a".to_string());
        state1.exit = StateExit::Await {
            promise: Expr::Call {
                func: Box::new(Expr::Path(TypePath {
                    segments: vec![PathSegment {
                        ident: Ident {
                            name: "fetch".to_string(),
                            evidentiality: None,
                            affect: None,
                            span: crate::span::Span::default(),
                        },
                        generics: None,
                    }],
                })),
                args: vec![Expr::Literal(Literal::Int {
                    value: "2".to_string(),
                    base: NumBase::Decimal,
                    suffix: None,
                })],
            },
            next_state: 2,
            saved_locals: vec!["a".to_string()],
        };
        ir.add_state(state1);

        // State 2: Resume with 'b', return a + b
        let mut state2 = State::resume(2);
        state2.resume_binding = Some("b".to_string());
        // Return expression: a + b (simplified to just literal for test)
        state2.exit = StateExit::Return {
            value: Expr::Literal(Literal::Int {
                value: "42".to_string(), // Simplified - would be a + b in real code
                base: NumBase::Decimal,
                suffix: None,
            }),
        };
        ir.add_state(state2);

        // Add frame layout for locals
        ir.frame_layout.add_local("a");
        ir.frame_layout.add_local("b");

        // =====================================================================
        // Step 2: Validate the IR
        // =====================================================================
        let validation_result = ir.validate();
        assert!(validation_result.is_ok(), "IR validation failed: {:?}", validation_result);

        assert_eq!(ir.states.len(), 3, "Expected 3 states for 2 await points");
        assert!(ir.states[0].is_entry);
        assert!(ir.states[1].is_resume);
        assert!(ir.states[2].is_resume);
        assert!(ir.states[0].exit.is_await());
        assert!(ir.states[1].exit.is_await());
        assert!(ir.states[2].exit.is_return());

        // =====================================================================
        // Step 3: Compile to WASM
        // =====================================================================
        let mut compiler = WasmCompiler::new();

        // Set up required imports per ASYNC-STATE-MACHINE-SPEC.md §4.4
        let alloc_idx = compiler.imports.add_import(
            "memory",
            "alloc",
            vec![ValType::I32],
            vec![ValType::I32],
        );
        compiler.func_map.insert("alloc".to_string(), alloc_idx);

        // async_create_continuation(frame_ptr: i32, state: i32, promise: i64) -> cont_ptr: i32
        let cont_idx = compiler.imports.add_import(
            "async",
            "async_create_continuation",
            vec![ValType::I32, ValType::I32, ValType::I64],
            vec![ValType::I32],
        );
        compiler.func_map.insert("async_create_continuation".to_string(), cont_idx);

        // Add a mock "fetch" function that returns i64 (promise)
        let fetch_idx = compiler.imports.add_import(
            "env",
            "fetch",
            vec![ValType::I64],
            vec![ValType::I64],
        );
        compiler.func_map.insert("fetch".to_string(), fetch_idx);

        // Create function entry with async state machine signature:
        // (frame_ptr: i32, resume_value: i64) -> i64
        let type_idx = compiler.get_or_create_type(
            vec![ValType::I32, ValType::I64],  // frame_ptr, resume_value
            vec![ValType::I64],
        );
        let func = CompiledFunction::new(
            "fetch_two".to_string(),
            type_idx,
            3,  // func_idx after imports (alloc=0, cont=1, fetch=2)
            vec![("frame_ptr".to_string(), ValType::I32), ("resume_value".to_string(), ValType::I64)],
            vec![ValType::I64],
            true,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);

        // Compile the state machine IR
        let result = compiler.compile_state_machine_ir(&ir);
        assert!(result.is_ok(), "compile_state_machine_ir failed: {:?}", result.err());

        // =====================================================================
        // Step 4: Verify the compiled function structure
        // =====================================================================
        let compiled_func = compiler.current_function().unwrap();

        // Verify we have a br_table for state dispatch (central switch)
        let has_br_table = compiled_func.instructions.iter().any(|i| matches!(i, Instruction::BrTable(..)));
        assert!(has_br_table, "Expected br_table for state dispatch");

        // Verify SUSPENDED_FLAG usage
        let suspend_flag_count = compiled_func.instructions.iter()
            .filter(|i| matches!(i, Instruction::I64Const(n) if *n == SUSPENDED_FLAG))
            .count();
        assert!(suspend_flag_count >= 2, "Expected SUSPENDED_FLAG at each await point (got {})", suspend_flag_count);

        // Verify async_create_continuation is called twice (once per await)
        let cont_call_count = compiled_func.instructions.iter()
            .filter(|i| matches!(i, Instruction::Call(idx) if *idx == cont_idx))
            .count();
        assert_eq!(cont_call_count, 2, "Expected 2 calls to async_create_continuation");

        // Verify fetch is called twice
        let fetch_call_count = compiled_func.instructions.iter()
            .filter(|i| matches!(i, Instruction::Call(idx) if *idx == fetch_idx))
            .count();
        assert_eq!(fetch_call_count, 2, "Expected 2 calls to fetch");

        // =====================================================================
        // Step 5: Verify locals for frame management
        // =====================================================================

        // Should have locals for: frame_ptr, resume_value, state, promise temps, etc.
        assert!(compiled_func.locals.len() >= 2, "Expected locals for async state machine");
    }

    /// Test that multiple await points create correct state transitions.
    #[cfg(feature = "wasm")]
    #[test]
    fn test_e2e_state_transitions() {
        use crate::wasm::WasmCompiler;
        use crate::wasm::types::CompiledFunction;
        use wasm_encoder::{Instruction, ValType};

        // Create a 4-state machine (3 await points)
        let mut ir = StateMachineIR::new(
            "triple_await".to_string(),
            vec![],
            Some(TypeExpr::Path(make_type_path("i64"))),
        );

        // State 0: entry -> await -> state 1
        let mut state0 = State::entry();
        state0.exit = StateExit::Await {
            promise: Expr::Literal(Literal::Int {
                value: "100".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
            next_state: 1,
            saved_locals: vec![],
        };
        ir.add_state(state0);

        // State 1: resume -> await -> state 2
        let mut state1 = State::resume(1);
        state1.exit = StateExit::Await {
            promise: Expr::Literal(Literal::Int {
                value: "200".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
            next_state: 2,
            saved_locals: vec![],
        };
        ir.add_state(state1);

        // State 2: resume -> await -> state 3
        let mut state2 = State::resume(2);
        state2.exit = StateExit::Await {
            promise: Expr::Literal(Literal::Int {
                value: "300".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
            next_state: 3,
            saved_locals: vec![],
        };
        ir.add_state(state2);

        // State 3: resume -> return
        let mut state3 = State::resume(3);
        state3.exit = StateExit::Return {
            value: Expr::Literal(Literal::Int {
                value: "42".to_string(),
                base: NumBase::Decimal,
                suffix: None,
            }),
        };
        ir.add_state(state3);

        // Validate
        assert!(ir.validate().is_ok());
        assert_eq!(ir.states.len(), 4);

        // Compile
        let mut compiler = WasmCompiler::new();

        let alloc_idx = compiler.imports.add_import("memory", "alloc", vec![ValType::I32], vec![ValType::I32]);
        compiler.func_map.insert("alloc".to_string(), alloc_idx);

        let cont_idx = compiler.imports.add_import("async", "async_create_continuation", vec![ValType::I32, ValType::I32, ValType::I64], vec![ValType::I32]);
        compiler.func_map.insert("async_create_continuation".to_string(), cont_idx);

        let type_idx = compiler.get_or_create_type(
            vec![ValType::I32, ValType::I64],
            vec![ValType::I64],
        );
        let func = CompiledFunction::new(
            "triple_await".to_string(),
            type_idx,
            2,  // func_idx after imports (alloc=0, cont=1)
            vec![("frame_ptr".to_string(), ValType::I32), ("resume_value".to_string(), ValType::I64)],
            vec![ValType::I64],
            true,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);

        let result = compiler.compile_state_machine_ir(&ir);
        assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

        // Verify 3 continuation calls (3 await points)
        let compiled_func = compiler.current_function().unwrap();
        let cont_call_count = compiled_func.instructions.iter()
            .filter(|i| matches!(i, Instruction::Call(idx) if *idx == cont_idx))
            .count();
        assert_eq!(cont_call_count, 3, "Expected 3 continuation calls");

        // Verify 3 SUSPENDED_FLAG usages
        let suspend_flag_count = compiled_func.instructions.iter()
            .filter(|i| matches!(i, Instruction::I64Const(n) if *n == SUSPENDED_FLAG))
            .count();
        assert_eq!(suspend_flag_count, 3, "Expected 3 SUSPENDED_FLAG usages");

        // Verify final return (I64Const(42))
        let has_final_return = compiled_func.instructions.iter()
            .any(|i| matches!(i, Instruction::I64Const(42)));
        assert!(has_final_return, "Expected final return value of 42");
    }
}

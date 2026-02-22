//! Control flow compilation.
//!
//! Compiles Sigil control flow constructs to WASM instructions.

use wasm_encoder::{BlockType, Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::types::LoopContext;
use super::WasmCompiler;
use crate::ast::{Block, Expr, MatchArm, Pattern, Stmt};

impl WasmCompiler {
    /// Compile a block expression.
    pub fn compile_block(&mut self, block: &Block) -> WasmResult<()> {
        // Push new scope
        self.scope_vars.push(std::collections::HashMap::new());

        // Compile statements
        for stmt in &block.stmts {
            self.compile_stmt(stmt)?;
        }

        // Compile final expression if present
        if let Some(expr) = &block.expr {
            self.compile_expr(expr)?;
        } else {
            // Block returns unit (0)
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I64Const(0));
        }

        // Pop scope
        self.scope_vars.pop();

        Ok(())
    }

    /// Compile an if expression.
    pub fn compile_if(
        &mut self,
        condition: &Expr,
        then_branch: &Block,
        else_branch: Option<&Expr>,
    ) -> WasmResult<()> {
        // Compile condition
        self.compile_expr(condition)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Convert i64 to i32 for branching
        func.push(Instruction::I32WrapI64);

        // Start if block (returns i64)
        func.push(Instruction::If(BlockType::Result(ValType::I64)));



        // Compile then branch
        self.compile_block(then_branch)?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Else);



        // Compile else branch if present
        if let Some(else_expr) = else_branch {
            match else_expr {
                Expr::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    // Else-if chain
                    self.compile_if(condition, then_branch, else_branch.as_deref())?;
                }
                Expr::Block(block) => {
                    self.compile_block(block)?;
                }
                _ => {
                    self.compile_expr(else_expr)?;
                }
            }
        } else {
            // No else branch - return unit
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::I64Const(0));
        }

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::End);

        Ok(())
    }

    /// Compile a while loop.
    pub fn compile_while(&mut self, condition: &Expr, body: &Block, label: Option<String>) -> WasmResult<()> {
        // Structure:
        // block $break           ;; empty result (break jumps here without value)
        //   loop $continue
        //     <condition>
        //     i64.eqz
        //     br_if 1 $break     ;; exit loop if condition is false
        //     <body>
        //     drop               ;; discard body result
        //     br 0 $continue     ;; continue looping
        //   end
        // end
        // i64.const 0            ;; while loop returns unit

        // Reserve labels for future labeled break support
        let _break_label = self.label_counter;
        let _continue_label = self.label_counter + 1;
        self.label_counter += 2;

        // Push loop context
        self.loop_stack.push(LoopContext {
            break_label: 1, // Relative depth to break block
            continue_label: 0, // Relative depth to loop
            name: label,
        });

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Outer block for break - Empty because br_if doesn't carry a value
        func.push(Instruction::Block(BlockType::Empty));

        // Inner loop for continue
        func.push(Instruction::Loop(BlockType::Empty));



        // Compile condition
        self.compile_expr(condition)?;

        let func = self.current_function_mut().unwrap();

        // Check if condition is false (zero) and break
        // I64Eqz returns i32 (1 if zero, 0 if nonzero)
        func.push(Instruction::I64Eqz);
        func.push(Instruction::BrIf(1)); // Break if condition is false



        // Compile body
        self.compile_block(body)?;

        let func = self.current_function_mut().unwrap();

        // Drop body result (statement effect)
        func.push(Instruction::Drop);

        // Continue to next iteration
        func.push(Instruction::Br(0));

        // End loop
        func.push(Instruction::End);

        // End block
        func.push(Instruction::End);

        // Push result (unit) - while loops return unit
        func.push(Instruction::I64Const(0));

        // Pop loop context
        self.loop_stack.pop();

        Ok(())
    }

    /// Compile a for loop.
    pub fn compile_for(&mut self, pattern: &Pattern, iter: &Expr, body: &Block, label: Option<String>) -> WasmResult<()> {
        // Check if iterating over a range (0..n or 0..=n)
        if let Expr::Range { start, end, inclusive } = iter {
            return self.compile_for_range(pattern, start.as_deref(), end.as_deref(), *inclusive, body, label);
        }

        // For arrays: iterate using index
        // Structure:
        // let arr = <iter>
        // let len = array_len(arr)
        // let i = 0
        // block $break
        //   loop $continue
        //     if i >= len: br $break
        //     let <pattern> = arr[i]
        //     <body>
        //     i = i + 1
        //     br $continue
        //   end
        // end

        // Push new scope for loop variables
        self.scope_vars.push(std::collections::HashMap::new());

        // Resolve morpheme imports before mutable borrows
        let array_len_idx = self.imports.get_func("morpheme_array_len")
            .ok_or_else(|| WasmError::internal("morpheme_array_len import not found"))?;
        let array_get_idx = self.imports.get_func("morpheme_array_get")
            .ok_or_else(|| WasmError::internal("morpheme_array_get import not found"))?;

        // Compile iterator expression
        self.compile_expr(iter)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store array handle (i64 JS-table index) in a local
        let arr_idx = func.alloc_local("__for_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));

        // Get array length via morpheme_array_len(arr_i32) -> i32
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::Call(array_len_idx)); // -> i32
        func.push(Instruction::I64ExtendI32U);       // -> i64

        let len_idx = func.alloc_local("__for_len".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(len_idx));

        // Initialize index to 0
        func.push(Instruction::I64Const(0));
        let idx_idx = func.alloc_local("__for_idx".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(idx_idx));

        // Outer block for break - Empty because br_if doesn't carry a value
        func.push(Instruction::Block(BlockType::Empty));

        // Inner loop
        func.push(Instruction::Loop(BlockType::Empty));

        // Push loop context
        self.loop_stack.push(LoopContext {
            break_label: 1,
            continue_label: 0,
            name: label,
        });

        let func = self.current_function_mut().unwrap();

        // Check if i >= len — I64GeU returns i32
        func.push(Instruction::LocalGet(idx_idx));
        func.push(Instruction::LocalGet(len_idx));
        func.push(Instruction::I64GeU);
        func.push(Instruction::BrIf(1)); // Break if done

        // Get current element via morpheme_array_get(arr_i32, idx_i32) -> i64
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::LocalGet(idx_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::Call(array_get_idx)); // -> i64



        // Bind pattern
        self.bind_pattern(pattern)?;

        // Compile body
        self.compile_block(body)?;

        let func = self.current_function_mut().unwrap();

        // Drop body result
        func.push(Instruction::Drop);

        // Increment index: i = i + 1
        func.push(Instruction::LocalGet(idx_idx));
        func.push(Instruction::I64Const(1));
        func.push(Instruction::I64Add);
        func.push(Instruction::LocalSet(idx_idx));

        // Continue
        func.push(Instruction::Br(0));

        // End loop
        func.push(Instruction::End);

        // End block
        func.push(Instruction::End);

        // Result (unit) - for loops return unit
        func.push(Instruction::I64Const(0));

        // Pop loop context and scope
        self.loop_stack.pop();
        self.scope_vars.pop();

        Ok(())
    }

    /// Compile a for loop over a range (start..end or start..=end).
    fn compile_for_range(
        &mut self,
        pattern: &Pattern,
        start: Option<&Expr>,
        end: Option<&Expr>,
        inclusive: bool,
        body: &Block,
        label: Option<String>,
    ) -> WasmResult<()> {
        // Structure:
        // let i = <start> (or 0 if None)
        // let end = <end>
        // block $break
        //   loop $continue
        //     if i >= end (or > for exclusive): br $break
        //     let <pattern> = i
        //     <body>
        //     i = i + 1
        //     br $continue
        //   end
        // end

        // Push new scope for loop variables
        self.scope_vars.push(std::collections::HashMap::new());

        // Compile start value (default to 0)
        if let Some(start_expr) = start {
            self.compile_expr(start_expr)?;
        } else {
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I64Const(0));
        }

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store current value in a local
        let idx_local = func.alloc_local("__range_idx".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(idx_local));

        // Compile end value
        if let Some(end_expr) = end {
            self.compile_expr(end_expr)?;
        } else {
            // No end - this would be an infinite range, which we don't support in for loops
            return Err(WasmError::unsupported("infinite ranges in for loops"));
        }

        let func = self.current_function_mut().unwrap();

        let end_local = func.alloc_local("__range_end".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(end_local));

        // Outer block for break
        func.push(Instruction::Block(BlockType::Empty));

        // Inner loop
        func.push(Instruction::Loop(BlockType::Empty));

        // Push loop context
        self.loop_stack.push(LoopContext {
            break_label: 1,
            continue_label: 0,
            name: label,
        });

        let func = self.current_function_mut().unwrap();

        // Check termination condition
        // For exclusive (..): if i >= end, break
        // For inclusive (..=): if i > end, break
        func.push(Instruction::LocalGet(idx_local));
        func.push(Instruction::LocalGet(end_local));
        if inclusive {
            func.push(Instruction::I64GtS); // i > end
        } else {
            func.push(Instruction::I64GeS); // i >= end
        }
        func.push(Instruction::BrIf(1)); // Break if done

        // Push current value for pattern binding
        func.push(Instruction::LocalGet(idx_local));

        // Bind pattern (e.g., `i` in `for i in 0..n`)
        self.bind_pattern(pattern)?;

        // Compile body
        self.compile_block(body)?;

        let func = self.current_function_mut().unwrap();

        // Drop body result
        func.push(Instruction::Drop);

        // Increment index: i = i + 1
        func.push(Instruction::LocalGet(idx_local));
        func.push(Instruction::I64Const(1));
        func.push(Instruction::I64Add);
        func.push(Instruction::LocalSet(idx_local));

        // Continue
        func.push(Instruction::Br(0));

        // End loop
        func.push(Instruction::End);

        // End block
        func.push(Instruction::End);

        // Result (unit) - for loops return unit
        func.push(Instruction::I64Const(0));

        // Pop loop context and scope
        self.loop_stack.pop();
        self.scope_vars.pop();

        Ok(())
    }

    /// Compile an infinite loop.
    pub fn compile_loop(&mut self, body: &Block, label: Option<String>) -> WasmResult<()> {
        // Structure:
        // block $break
        //   loop $continue
        //     <body>
        //     br 0 $continue
        //   end
        // end

        self.loop_stack.push(LoopContext {
            break_label: 1,
            continue_label: 0,
            name: label,
        });

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Outer block for break (returns the break value)
        func.push(Instruction::Block(BlockType::Result(ValType::I64)));

        // Inner loop
        func.push(Instruction::Loop(BlockType::Empty));



        // Compile body
        self.compile_block(body)?;

        let func = self.current_function_mut().unwrap();

        // Drop body result
        func.push(Instruction::Drop);

        // Continue to next iteration
        func.push(Instruction::Br(0));

        // End loop
        func.push(Instruction::End);

        // This is unreachable unless break is called
        // Need a dummy value for type checking
        func.push(Instruction::I64Const(0));
        func.push(Instruction::End);

        self.loop_stack.pop();

        Ok(())
    }

    /// Compile a match expression.
    pub fn compile_match(&mut self, expr: &Expr, arms: &[MatchArm]) -> WasmResult<()> {
        // Compile the scrutinee
        self.compile_expr(expr)?;

        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store scrutinee in a local for repeated comparison
        let scrutinee_idx = func.alloc_local("__match_scrutinee".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(scrutinee_idx));



        // Compile arms as a chain of if-else
        self.compile_match_arms(scrutinee_idx, arms, 0)
    }

    /// Recursively compile match arms.
    pub fn compile_match_arms(
        &mut self,
        scrutinee_idx: u32,
        arms: &[MatchArm],
        index: usize,
    ) -> WasmResult<()> {
        if index >= arms.len() {
            // No more arms - this shouldn't happen with exhaustive matching
            // Return unit as fallback
            let func = self
                .current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I64Const(0));
            return Ok(());
        }

        let arm = &arms[index];

        // Check if this is a wildcard pattern
        let is_wildcard = matches!(&arm.pattern, Pattern::Wildcard);

        if is_wildcard || index == arms.len() - 1 {
            // Last arm or wildcard - no condition needed
            self.scope_vars.push(std::collections::HashMap::new());

            // Bind pattern variables
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(scrutinee_idx));
    

            self.bind_pattern(&arm.pattern)?;

            // Compile body
            self.compile_expr(&arm.body)?;

            self.scope_vars.pop();
        } else {
            // Compile pattern match check
            self.compile_pattern_check(scrutinee_idx, &arm.pattern)?;

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::I32WrapI64);
            func.push(Instruction::If(BlockType::Result(ValType::I64)));

    

            // Pattern matched - bind and execute body
            self.scope_vars.push(std::collections::HashMap::new());

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(scrutinee_idx));
    

            self.bind_pattern(&arm.pattern)?;

            // Check guard if present
            if let Some(guard) = &arm.guard {
                self.compile_expr(guard)?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::If(BlockType::Result(ValType::I64)));
        

                self.compile_expr(&arm.body)?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Else);
        

                // Guard failed - try next arm
                self.compile_match_arms(scrutinee_idx, arms, index + 1)?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::End);
            } else {
                self.compile_expr(&arm.body)?;
            }

            self.scope_vars.pop();

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::Else);

    

            // Try next arm
            self.compile_match_arms(scrutinee_idx, arms, index + 1)?;

            let func = self.current_function_mut().unwrap();
            func.push(Instruction::End);
        }

        Ok(())
    }

    /// Compile a pattern check (returns 1 if matches, 0 otherwise).
    fn compile_pattern_check(&mut self, scrutinee_idx: u32, pattern: &Pattern) -> WasmResult<()> {
        match pattern {
            Pattern::Wildcard => {
                // Always matches
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(1));
            }

            Pattern::Ident { .. } => {
                // Variable binding always matches
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(1));
            }

            Pattern::Literal(lit) => {
                // Compare with literal
                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalGet(scrutinee_idx));
                }
                self.compile_literal(lit)?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Eq);
                func.push(Instruction::I64ExtendI32U);
            }

            Pattern::Tuple(patterns) => {
                // Allocate temp locals for each element first
                let temp_indices: Vec<u32> = {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64Const(1)); // Start with true

                    let mut indices = Vec::new();
                    for (i, _) in patterns.iter().enumerate() {
                        // Get tuple element
                        func.push(Instruction::LocalGet(scrutinee_idx));
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: (i * 8) as u64,
                            align: 3,
                            memory_index: 0,
                        }));

                        let temp_idx = func.alloc_local(format!("__tuple_{}", i), ValType::I64);
                        func.push(Instruction::LocalSet(temp_idx));
                        indices.push(temp_idx);
                    }
                    indices
                };

                // Now check each pattern
                for (i, pat) in patterns.iter().enumerate() {
                    self.compile_pattern_check(temp_indices[i], pat)?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64And);
                }
            }

            Pattern::Struct { fields, .. } => {
                // Allocate temp locals for each field first
                let field_data: Vec<(u32, Option<Pattern>)> = {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64Const(1));

                    let mut data = Vec::new();
                    for (i, field) in fields.iter().enumerate() {
                        let temp_idx = if field.pattern.is_some() {
                            func.push(Instruction::LocalGet(scrutinee_idx));
                            func.push(Instruction::I32WrapI64);
                            func.push(Instruction::I64Load(wasm_encoder::MemArg {
                                offset: (i * 8) as u64,
                                align: 3,
                                memory_index: 0,
                            }));

                            let temp_idx = func.alloc_local(format!("__struct_{}", i), ValType::I64);
                            func.push(Instruction::LocalSet(temp_idx));
                            temp_idx
                        } else {
                            0 // Placeholder, won't be used
                        };
                        data.push((temp_idx, field.pattern.clone()));
                    }
                    data
                };

                // Now check each pattern
                for (temp_idx, pat_opt) in field_data {
                    if let Some(pat) = pat_opt {
                        self.compile_pattern_check(temp_idx, &pat)?;
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::I64And);
                    }
                }
            }

            Pattern::Or(patterns) => {
                // Check any pattern matches
                if patterns.is_empty() {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64Const(0));
                    return Ok(());
                }

                let temp = {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalGet(scrutinee_idx));
                    let temp = func.alloc_local("__or_temp".to_string(), ValType::I64);
                    func.push(Instruction::LocalSet(temp));
                    temp
                };

                // Start with first pattern
                self.compile_pattern_check(temp, &patterns[0])?;

                // OR with remaining patterns
                for pat in &patterns[1..] {
                    {
                        let func = self.current_function_mut().unwrap();
                        // pattern_check returns i64; I64Eqz: i64→i32 (1 if zero=not matched)
                        func.push(Instruction::I64Eqz);
                        // I64Eqz already returns i32 — If takes i32 condition directly
                        func.push(Instruction::If(BlockType::Result(ValType::I64)));
                    }

                    self.compile_pattern_check(temp, pat)?;

                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Else);
                    func.push(Instruction::I64Const(1));
                    func.push(Instruction::End);
                }
            }

            Pattern::Range { start, end, inclusive } => {
                // Range pattern: check start <= scrutinee <= end (or < end if not inclusive)
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(1)); // Start with true

        

                // Check start bound if present
                if let Some(start_pat) = start {
                    if let Pattern::Literal(lit) = start_pat.as_ref() {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(scrutinee_idx));
                
                        self.compile_literal(lit)?;
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::I64GeS); // scrutinee >= start
                        func.push(Instruction::I64ExtendI32U);
                        func.push(Instruction::I64And);
                    }
                }

                // Check end bound if present
                if let Some(end_pat) = end {
                    if let Pattern::Literal(lit) = end_pat.as_ref() {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(scrutinee_idx));
                
                        self.compile_literal(lit)?;
                        let func = self.current_function_mut().unwrap();
                        if *inclusive {
                            func.push(Instruction::I64LeS); // scrutinee <= end
                        } else {
                            func.push(Instruction::I64LtS); // scrutinee < end
                        }
                        func.push(Instruction::I64ExtendI32U);
                        func.push(Instruction::I64And);
                    }
                }
            }

            Pattern::Path(path) => {
                // Path pattern - typically an enum unit variant like None or Color::Red
                // For unit variants, the tag is stored directly in the value
                // Compare the tag value
                let variant_name = path
                    .segments
                    .last()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");

                // Look up variant tag (enum layout)
                let tag = self.get_enum_variant_tag(variant_name);

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(scrutinee_idx));
                func.push(Instruction::I64Const(tag as i64));
                func.push(Instruction::I64Eq);
                func.push(Instruction::I64ExtendI32U);
            }

            Pattern::TupleStruct { path, fields } => {
                // TupleStruct pattern like Some(x) or Point(x, y)
                // First byte is tag, then fields at offset 8, 16, etc.
                let variant_name = path
                    .segments
                    .last()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");
                let tag = self.get_enum_variant_tag(variant_name);

                // Check tag first
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(scrutinee_idx));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
                func.push(Instruction::I64Const(tag as i64));
                func.push(Instruction::I64Eq);
                func.push(Instruction::I64ExtendI32U);

                // Extract and check each field
                let temp_indices: Vec<u32> = {
                    let func = self.current_function_mut().unwrap();
                    let mut indices = Vec::new();
                    for (i, _) in fields.iter().enumerate() {
                        func.push(Instruction::LocalGet(scrutinee_idx));
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: ((i + 1) * 8) as u64, // Skip tag at offset 0
                            align: 3,
                            memory_index: 0,
                        }));
                        let temp_idx =
                            func.alloc_local(format!("__tuplestruct_{}", i), ValType::I64);
                        func.push(Instruction::LocalSet(temp_idx));
                        indices.push(temp_idx);
                    }
                    indices
                };

                // Check each field pattern
                for (i, pat) in fields.iter().enumerate() {
                    self.compile_pattern_check(temp_indices[i], pat)?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64And);
                }
            }

            Pattern::Slice(patterns) => {
                // Slice pattern like [a, b, c]
                // Array layout: 4-byte length + elements at 8-byte offsets

                // First check length matches
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(scrutinee_idx));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I32Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 2,
                    memory_index: 0,
                }));
                func.push(Instruction::I32Const(patterns.len() as i32));
                func.push(Instruction::I32Eq);
                func.push(Instruction::I64ExtendI32U);

                // Extract and check each element
                let temp_indices: Vec<u32> = {
                    let func = self.current_function_mut().unwrap();
                    let mut indices = Vec::new();
                    for (i, _) in patterns.iter().enumerate() {
                        func.push(Instruction::LocalGet(scrutinee_idx));
                        func.push(Instruction::I32WrapI64);
                        func.push(Instruction::I64Load(wasm_encoder::MemArg {
                            offset: (4 + i * 8) as u64, // Skip 4-byte length
                            align: 3,
                            memory_index: 0,
                        }));
                        let temp_idx = func.alloc_local(format!("__slice_{}", i), ValType::I64);
                        func.push(Instruction::LocalSet(temp_idx));
                        indices.push(temp_idx);
                    }
                    indices
                };

                for (i, pat) in patterns.iter().enumerate() {
                    self.compile_pattern_check(temp_indices[i], pat)?;
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64And);
                }
            }

            Pattern::Rest => {
                // Rest pattern (..) always matches
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(1));
            }

            Pattern::Ref { pattern, .. } => {
                // Reference pattern &x or &mut x - dereference and match inner
                // In WASM, the value is a pointer, so we load through it
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(scrutinee_idx));
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
                let deref_idx = func.alloc_local("__ref_deref".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(deref_idx));
                drop(func);
                self.compile_pattern_check(deref_idx, pattern)?;
            }

            Pattern::RefBinding { .. } => {
                // `ref name` pattern - creates a reference binding
                // Always matches, the binding happens in bind_pattern
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(1));
            }
        }

        Ok(())
    }

    /// Get enum variant tag (simple hash-based for now).
    fn get_enum_variant_tag(&self, variant_name: &str) -> u32 {
        // Check if we have a registered enum layout
        for layout in self.enum_layouts.values() {
            if let Some(tag) = layout.variant_tag(variant_name) {
                return tag;
            }
        }
        // Fallback: simple hash
        variant_name.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32))
    }

    /// Bind pattern variables from value on stack.
    pub(crate) fn bind_pattern(&mut self, pattern: &Pattern) -> WasmResult<()> {
        match pattern {
            Pattern::Wildcard => {
                // Discard value
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Drop);
            }

            Pattern::Ident { name, .. } => {
                // Bind to local variable
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let idx = func.alloc_local(name.name.clone(), ValType::I64);
                func.push(Instruction::LocalSet(idx));

                // Record in scope
                if let Some(scope) = self.scope_vars.last_mut() {
                    scope.insert(name.name.clone(), idx);
                }
            }

            Pattern::Tuple(patterns) => {
                // Store tuple pointer
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let ptr_idx = func.alloc_local("__tuple_ptr".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(ptr_idx));

        

                // Bind each element
                for (i, pat) in patterns.iter().enumerate() {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalGet(ptr_idx));
                    func.push(Instruction::I32WrapI64);
                    func.push(Instruction::I64Load(wasm_encoder::MemArg {
                        offset: (i * 8) as u64,
                        align: 3,
                        memory_index: 0,
                    }));
            

                    self.bind_pattern(pat)?;
                }
            }

            Pattern::Struct { fields, .. } => {
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let ptr_idx = func.alloc_local("__struct_ptr".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(ptr_idx));

        

                for (i, field) in fields.iter().enumerate() {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalGet(ptr_idx));
                    func.push(Instruction::I32WrapI64);
                    func.push(Instruction::I64Load(wasm_encoder::MemArg {
                        offset: (i * 8) as u64,
                        align: 3,
                        memory_index: 0,
                    }));
            

                    if let Some(pat) = &field.pattern {
                        self.bind_pattern(pat)?;
                    } else {
                        // Shorthand: field name becomes binding
                        let func = self.current_function_mut().unwrap();
                        let idx = func.alloc_local(field.name.name.clone(), ValType::I64);
                        func.push(Instruction::LocalSet(idx));
                        if let Some(scope) = self.scope_vars.last_mut() {
                            scope.insert(field.name.name.clone(), idx);
                        }
                    }
                }
            }

            Pattern::Literal(_) => {
                // No binding needed, just discard
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Drop);
            }

            Pattern::Path(_) => {
                // Unit variant - no bindings, just discard the value
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Drop);
            }

            Pattern::TupleStruct { path, fields } => {
                // Store pointer
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let ptr_idx = func.alloc_local("__tuplestruct_ptr".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(ptr_idx));

                // Determine the inner struct type for local_var_types.
                // For e.g. VNode::Element(el): enum="VNode", variant="Element" → inner="VElement"
                let inner_type = if path.segments.len() >= 2 {
                    let enum_name = &path.segments[path.segments.len() - 2].ident.name;
                    let variant_name = &path.segments[path.segments.len() - 1].ident.name;
                    self.enum_layouts.get(enum_name.as_str())
                        .and_then(|layout| layout.variant_inner_type(variant_name))
                        .map(str::to_string)
                } else {
                    None
                };

                // Bind each field (skip tag at offset 0)
                for (i, pat) in fields.iter().enumerate() {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalGet(ptr_idx));
                    func.push(Instruction::I32WrapI64);
                    func.push(Instruction::I64Load(wasm_encoder::MemArg {
                        offset: ((i + 1) * 8) as u64,
                        align: 3,
                        memory_index: 0,
                    }));

                    // If binding a simple ident and we know the inner struct type,
                    // register it for method dispatch and field offset resolution.
                    if let (crate::ast::Pattern::Ident { name, .. }, Some(ref type_name)) =
                        (pat, &inner_type)
                    {
                        if self.struct_layouts.contains_key(type_name.as_str()) {
                            self.local_var_types.insert(name.name.clone(), type_name.clone());
                        }
                    }

                    self.bind_pattern(pat)?;
                }
            }

            Pattern::Slice(patterns) => {
                // Store array pointer
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let ptr_idx = func.alloc_local("__slice_ptr".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(ptr_idx));

        

                // Bind each element (skip 4-byte length)
                for (i, pat) in patterns.iter().enumerate() {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalGet(ptr_idx));
                    func.push(Instruction::I32WrapI64);
                    func.push(Instruction::I64Load(wasm_encoder::MemArg {
                        offset: (4 + i * 8) as u64,
                        align: 3,
                        memory_index: 0,
                    }));
            

                    self.bind_pattern(pat)?;
                }
            }

            Pattern::Or(patterns) => {
                // Bind using first pattern (all should bind same variables)
                if let Some(first) = patterns.first() {
                    self.bind_pattern(first)?;
                } else {
                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::Drop);
                }
            }

            Pattern::Range { .. } | Pattern::Rest => {
                // No bindings for range or rest patterns
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Drop);
            }

            Pattern::Ref { pattern, .. } => {
                // Reference pattern &x - the value on stack is a pointer, we dereference and bind inner
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I32WrapI64);
                func.push(Instruction::I64Load(wasm_encoder::MemArg {
                    offset: 0,
                    align: 3,
                    memory_index: 0,
                }));
                drop(func);
                self.bind_pattern(pattern)?;
            }

            Pattern::RefBinding { name, .. } => {
                // `ref name` pattern - bind a reference (pointer) to the value
                // The value on stack becomes the address bound to `name`
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let idx = func.alloc_local(name.name.clone(), ValType::I64);
                func.push(Instruction::LocalSet(idx));
                if let Some(scope) = self.scope_vars.last_mut() {
                    scope.insert(name.name.clone(), idx);
                }
            }
        }

        Ok(())
    }

    /// Compile a statement (for use in blocks).
    pub fn compile_stmt(&mut self, stmt: &Stmt) -> WasmResult<()> {
        match stmt {
            Stmt::Let {
                pattern, ty, init, ..
            } => {
                // Extract type information from init expression
                // This is used to resolve method calls like app·view()
                let init_type = init.as_ref().and_then(|val| {
                    match val {
                        crate::ast::Expr::Struct { path, .. } => {
                            // Struct initialization: PlatformApp { ... } -> "PlatformApp"
                            path.segments.last().map(|s| s.ident.name.clone())
                        }
                        _ => None,
                    }
                });

                // Record variable type from explicit annotation (e.g. `≔ s: String = ...`)
                // This is needed for methods like `s·length()` to resolve to string_length
                // rather than morpheme_array_len when the receiver type is known.
                if let Some(ref type_annotation) = ty {
                    if let crate::ast::Pattern::Ident { name, .. } = pattern {
                        let type_name = match type_annotation {
                            crate::ast::TypeExpr::Path(path) => {
                                path.segments.last().map(|s| s.ident.name.clone())
                            }
                            crate::ast::TypeExpr::Evidential { inner, .. } => {
                                if let crate::ast::TypeExpr::Path(path) = inner.as_ref() {
                                    path.segments.last().map(|s| s.ident.name.clone())
                                } else {
                                    None
                                }
                            }
                            crate::ast::TypeExpr::Reference { inner, .. } => {
                                if let crate::ast::TypeExpr::Path(path) = inner.as_ref() {
                                    path.segments.last().map(|s| s.ident.name.clone())
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        if let Some(tname) = type_name {
                            self.var_types.entry(name.name.clone()).or_insert(tname);
                        }
                    }
                }

                // Record variable type if we have pattern name and type (from init expr)
                if let Some(ref type_name) = init_type {
                    if let crate::ast::Pattern::Ident { name, .. } = pattern {
                        self.var_types.insert(name.name.clone(), type_name.clone());
                    }
                }

                // Compile init value
                if let Some(val) = init {
                    self.compile_expr(val)?;
                } else {
                    let func = self
                        .current_function_mut()
                        .ok_or_else(|| WasmError::internal("not in function context"))?;
                    func.push(Instruction::I64Const(0));
                }

                // Bind pattern
                self.bind_pattern(pattern)
            }

            Stmt::LetElse { pattern, init, .. } => {
                // Compile init value
                self.compile_expr(init)?;
                // Bind pattern (ignore else for now)
                self.bind_pattern(pattern)
            }

            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                // Drop result (statement effect)
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Drop);
                Ok(())
            }

            Stmt::Semi(expr) => {
                self.compile_expr(expr)?;
                // Drop result (statement with semicolon)
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::Drop);
                Ok(())
            }

            Stmt::Item(_) => {
                // Items in blocks are hoisted - skip for now
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Ident, Literal, NumBase};
    use crate::span::Span;
    use crate::wasm::literals::create_test_compiler_with_function;

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

    fn make_block(expr: Expr) -> Block {
        Block {
            stmts: vec![],
            expr: Some(Box::new(expr)),
        }
    }

    #[test]
    fn test_compile_if_else() {
        let mut compiler = create_test_compiler_with_function();

        compiler
            .compile_if(
                &make_int(1),
                &make_block(make_int(10)),
                Some(&Expr::Block(make_block(make_int(20)))),
            )
            .unwrap();

        let func = compiler.current_function().unwrap();
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
    fn test_compile_if_no_else() {
        let mut compiler = create_test_compiler_with_function();

        compiler
            .compile_if(&make_int(1), &make_block(make_int(10)), None)
            .unwrap();

        let func = compiler.current_function().unwrap();
        // Should still have else branch (returning unit)
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Else)));
    }

    #[test]
    fn test_compile_while() {
        let mut compiler = create_test_compiler_with_function();

        compiler
            .compile_while(&make_int(1), &make_block(make_int(0)), None)
            .unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Loop(_))));
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Block(_))));
    }

    #[test]
    fn test_compile_loop() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_loop(&make_block(make_int(0)), None).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Loop(_))));
    }

    #[test]
    fn test_compile_block_with_statements() {
        let mut compiler = create_test_compiler_with_function();

        let block = Block {
            stmts: vec![Stmt::Semi(make_int(1))],
            expr: Some(Box::new(make_int(42))),
        };

        compiler.compile_block(&block).unwrap();

        let func = compiler.current_function().unwrap();
        // Should have statement value dropped
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Drop)));
    }

    #[test]
    fn test_compile_let_statement() {
        let mut compiler = create_test_compiler_with_function();

        let stmt = Stmt::Let {
            pattern: Pattern::Ident {
                mutable: false,
                name: make_ident("x"),
                evidentiality: None,
            },
            ty: None,
            init: Some(make_int(42)),
        };

        compiler.compile_stmt(&stmt).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalSet(_))));
    }

    #[test]
    fn test_bind_wildcard_pattern() {
        let mut compiler = create_test_compiler_with_function();

        // Push a value
        compiler.compile_expr(&make_int(42)).unwrap();

        // Bind wildcard (should drop)
        compiler.bind_pattern(&Pattern::Wildcard).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Drop)));
    }

    #[test]
    fn test_bind_identifier_pattern() {
        let mut compiler = create_test_compiler_with_function();

        // Push a value
        compiler.compile_expr(&make_int(42)).unwrap();

        let pattern = Pattern::Ident {
            mutable: false,
            name: make_ident("x"),
            evidentiality: None,
        };

        compiler.bind_pattern(&pattern).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalSet(_))));
    }

    #[test]
    fn test_compile_match_wildcard() {
        let mut compiler = create_test_compiler_with_function();

        let arms = vec![MatchArm {
            pattern: Pattern::Wildcard,
            guard: None,
            body: make_int(42),
        }];

        compiler.compile_match(&make_int(0), &arms).unwrap();

        let func = compiler.current_function().unwrap();
        // Should compile without if/else since wildcard always matches
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Const(42))));
    }
}

//! Morpheme and pipe operator compilation.
//!
//! Compiles Sigil morpheme operators (τ, φ, σ, ρ, α, ω, etc.) to WASM instructions.

use wasm_encoder::{BlockType, Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;
use crate::ast::{Expr, MorphemeKind, PipeOp};

impl WasmCompiler {
    /// Compile a pipe expression.
    pub fn compile_pipe(&mut self, expr: &Expr, operations: &[PipeOp]) -> WasmResult<()> {
        // Compile the initial expression
        self.compile_expr(expr)?;

        // Apply each pipe operation
        for op in operations {
            self.compile_pipe_op(op)?;
        }

        Ok(())
    }

    /// Compile a pipe operation.
    fn compile_pipe_op(&mut self, op: &PipeOp) -> WasmResult<()> {
        match op {
            // Transform: τ{f} - map each element
            PipeOp::Transform(func) => self.compile_transform(func),

            // Filter: φ{p} - keep elements where predicate is true
            PipeOp::Filter(predicate) => self.compile_filter(predicate),

            // Sort: σ or σ.field
            PipeOp::Sort(field) => self.compile_sort(field.as_ref()),

            // Reduce: ρ{f}
            PipeOp::Reduce(func) => self.compile_reduce(func),

            // Sum reduction: ρ+
            PipeOp::ReduceSum => self.compile_reduce_sum(),

            // Product reduction: ρ*
            PipeOp::ReduceProd => self.compile_reduce_prod(),

            // Min reduction: ρ_min
            PipeOp::ReduceMin => self.compile_reduce_min(),

            // Max reduction: ρ_max
            PipeOp::ReduceMax => self.compile_reduce_max(),

            // All reduction: ρ&
            PipeOp::ReduceAll => self.compile_reduce_all(),

            // Any reduction: ρ|
            PipeOp::ReduceAny => self.compile_reduce_any(),

            // Concat reduction: ρ++
            PipeOp::ReduceConcat => Err(WasmError::unsupported("concat reduction")),

            // First: α
            PipeOp::First => self.compile_first(),

            // Last: ω
            PipeOp::Last => self.compile_last(),

            // Middle: μ
            PipeOp::Middle => self.compile_middle(),

            // Choice: χ (random)
            PipeOp::Choice => self.compile_choice(),

            // Nth: ν{n}
            PipeOp::Nth(n) => self.compile_nth(n),

            // Next: ξ
            PipeOp::Next => self.compile_next(),

            // Method call in pipe
            PipeOp::Method { name, type_args: _, args } => self.compile_pipe_method(&name.name, args),

            // Await in pipe
            PipeOp::Await => Err(WasmError::unsupported("await in pipe")),

            // Match in pipe
            PipeOp::Match(arms) => {
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                let temp = func.alloc_local("__pipe_match".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(temp));
        
                self.compile_match_arms(temp, arms, 0)
            }

            // Parallel morpheme
            PipeOp::Parallel(inner) => self.compile_parallel(inner),

            // GPU morpheme
            PipeOp::Gpu(_) => Err(WasmError::unsupported("GPU morphemes")),

            // Try/map morpheme
            PipeOp::TryMap(_) => Err(WasmError::unsupported("try-map morpheme")),

            // Named morpheme
            PipeOp::Named { prefix, body } => {
                let name = prefix
                    .iter()
                    .map(|i| i.name.as_str())
                    .collect::<Vec<_>>()
                    .join("·");
                self.compile_named_morpheme(&name, body.as_deref())
            }

            // Protocol operations
            PipeOp::Send(_) => Err(WasmError::unsupported("send operation")),
            PipeOp::Recv => Err(WasmError::unsupported("recv operation")),
            PipeOp::Stream(_) => Err(WasmError::unsupported("stream operation")),
            PipeOp::Connect(_) => Err(WasmError::unsupported("connect operation")),
            PipeOp::Close => Err(WasmError::unsupported("close operation")),
            PipeOp::Header { .. } => Err(WasmError::unsupported("header operation")),
            PipeOp::Body(_) => Err(WasmError::unsupported("body operation")),
            PipeOp::Timeout(_) => Err(WasmError::unsupported("timeout operation")),
            PipeOp::Retry { .. } => Err(WasmError::unsupported("retry operation")),

            // Evidence operations
            PipeOp::Validate { .. } => Err(WasmError::unsupported("validate operation")),
            PipeOp::Assume { .. } => Err(WasmError::unsupported("assume operation")),
            PipeOp::AssertEvidence(_) => Err(WasmError::unsupported("assert evidence")),

            // Scope functions (Kotlin-inspired)
            PipeOp::Also(_) => Err(WasmError::unsupported("also operation")),
            PipeOp::Apply(_) => Err(WasmError::unsupported("apply operation")),
            PipeOp::TakeIf(_) => Err(WasmError::unsupported("take_if operation")),
            PipeOp::TakeUnless(_) => Err(WasmError::unsupported("take_unless operation")),
            PipeOp::Let(_) => Err(WasmError::unsupported("let operation")),

            // Mathematical & APL-inspired operations
            PipeOp::All(_) => Err(WasmError::unsupported("all/forall operation")),
            PipeOp::Any(_) => Err(WasmError::unsupported("any/exists operation")),
            PipeOp::Compose(_) => Err(WasmError::unsupported("compose operation")),
            PipeOp::Zip(_) => Err(WasmError::unsupported("zip operation")),
            PipeOp::Scan(_) => Err(WasmError::unsupported("scan operation")),
            PipeOp::Diff => Err(WasmError::unsupported("diff operation")),
            PipeOp::Gradient(_) => Err(WasmError::unsupported("gradient operation")),
            PipeOp::SortAsc => Err(WasmError::unsupported("sort ascending")),
            PipeOp::SortDesc => Err(WasmError::unsupported("sort descending")),
            PipeOp::Reverse => Err(WasmError::unsupported("reverse operation")),
            PipeOp::Cycle(_) => Err(WasmError::unsupported("cycle operation")),
            PipeOp::Windows(_) => Err(WasmError::unsupported("windows operation")),
            PipeOp::Chunks(_) => Err(WasmError::unsupported("chunks operation")),
            PipeOp::Flatten => Err(WasmError::unsupported("flatten operation")),
            PipeOp::Unique => Err(WasmError::unsupported("unique operation")),
            PipeOp::Enumerate => Err(WasmError::unsupported("enumerate operation")),

            // Function call in pipe
            PipeOp::Call(call_expr) => {
                // Check if this is a morpheme function like σ()
                match call_expr.as_ref() {
                    Expr::Call { func, args } => {
                        // If function is a path like σ, treat as morpheme
                        if let Expr::Path(path) = func.as_ref() {
                            if let Some(seg) = path.segments.first() {
                                let name = &seg.ident.name;
                                match name.as_str() {
                                    "σ" | "Σ" | "sort" | "collect" => {
                                        // Sort/collect operation - Σ often used to materialize/collect
                                        return self.compile_sort(None);
                                    }
                                    "τ" | "map" | "transform" => {
                                        if let Some(body) = args.first() {
                                            return self.compile_transform(body);
                                        }
                                    }
                                    "φ" | "filter" => {
                                        if let Some(pred) = args.first() {
                                            return self.compile_filter(pred);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    // Simple path like σ (no call parens)
                    Expr::Path(path) => {
                        if let Some(seg) = path.segments.first() {
                            let name = &seg.ident.name;
                            match name.as_str() {
                                "σ" | "sort" | "collect" => {
                                    return self.compile_sort(None);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
                Err(WasmError::unsupported("call in pipe"))
            }
        }
    }

    /// Compile a morpheme application.
    pub fn compile_morpheme(&mut self, kind: MorphemeKind, body: &Expr) -> WasmResult<()> {
        match kind {
            MorphemeKind::Transform => self.compile_transform(body),
            MorphemeKind::Filter => self.compile_filter(body),
            MorphemeKind::Sort => {
                // Sort without field - compile body as comparator
                self.compile_sort(None)
            }
            MorphemeKind::Reduce => self.compile_reduce(body),
            MorphemeKind::Lambda => {
                // Lambda is just a closure
                Err(WasmError::unsupported("lambda morpheme"))
            }
            MorphemeKind::Sum => self.compile_reduce_sum(),
            MorphemeKind::Product => self.compile_reduce_prod(),
            MorphemeKind::Middle => self.compile_middle(),
            MorphemeKind::Choice => self.compile_choice(),
            MorphemeKind::Nth => self.compile_nth(body),
            MorphemeKind::Next => self.compile_next(),
            MorphemeKind::First => self.compile_first(),
            MorphemeKind::Last => self.compile_last(),
        }
    }

    /// Compile transform morpheme (τ/map).
    fn compile_transform(&mut self, func: &Expr) -> WasmResult<()> {
        // Structure:
        // 1. Get array from stack
        // 2. Allocate new array of same length
        // 3. Loop: apply func to each element, store in new array
        // 4. Return new array

        let compiler_func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store input array
        let arr_idx = compiler_func.alloc_local("__map_arr".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(arr_idx));

        // Get length
        compiler_func.push(Instruction::LocalGet(arr_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Load(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));
        compiler_func.push(Instruction::I64ExtendI32U);

        let len_idx = compiler_func.alloc_local("__map_len".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(len_idx));

        // Allocate new array: call runtime allocator
        // Size = 4 (length) + len * 8
        compiler_func.push(Instruction::LocalGet(len_idx));
        compiler_func.push(Instruction::I64Const(8));
        compiler_func.push(Instruction::I64Mul);
        compiler_func.push(Instruction::I64Const(4));
        compiler_func.push(Instruction::I64Add);

        // Call allocator
        let alloc_idx = self
            .get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;
        let compiler_func = self.current_function_mut().unwrap();
        compiler_func.push(Instruction::Call(alloc_idx));

        let out_idx = compiler_func.alloc_local("__map_out".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(out_idx));

        // Write length to output array
        compiler_func.push(Instruction::LocalGet(out_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::LocalGet(len_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Store(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));

        // Initialize index
        compiler_func.push(Instruction::I64Const(0));
        let i_idx = compiler_func.alloc_local("__map_i".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(i_idx));

        // Loop
        compiler_func.push(Instruction::Block(BlockType::Empty));
        compiler_func.push(Instruction::Loop(BlockType::Empty));

        // Check: if i >= len, break
        // I64GeU returns i32 which is what BrIf expects
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::LocalGet(len_idx));
        compiler_func.push(Instruction::I64GeU);
        compiler_func.push(Instruction::BrIf(1));

        // Load element: arr[i]
        compiler_func.push(Instruction::LocalGet(arr_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::I64Const(8));
        compiler_func.push(Instruction::I64Mul);
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Add);
        compiler_func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 4,
            align: 3,
            memory_index: 0,
        }));



        // Apply function
        self.compile_apply_to_stack(func)?;

        let compiler_func = self.current_function_mut().unwrap();

        // Store result: out[i] = result
        let result_idx = compiler_func.alloc_local("__map_result".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(result_idx));

        compiler_func.push(Instruction::LocalGet(out_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::I64Const(8));
        compiler_func.push(Instruction::I64Mul);
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Add);
        compiler_func.push(Instruction::LocalGet(result_idx));
        compiler_func.push(Instruction::I64Store(wasm_encoder::MemArg {
            offset: 4,
            align: 3,
            memory_index: 0,
        }));

        // Increment i
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::I64Const(1));
        compiler_func.push(Instruction::I64Add);
        compiler_func.push(Instruction::LocalSet(i_idx));

        // Continue loop
        compiler_func.push(Instruction::Br(0));
        compiler_func.push(Instruction::End);
        compiler_func.push(Instruction::End);

        // Return output array
        compiler_func.push(Instruction::LocalGet(out_idx));

        Ok(())
    }

    /// Compile filter morpheme (φ).
    fn compile_filter(&mut self, predicate: &Expr) -> WasmResult<()> {
        let compiler_func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store input array
        let arr_idx = compiler_func.alloc_local("__filter_arr".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(arr_idx));

        // Get length
        compiler_func.push(Instruction::LocalGet(arr_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Load(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));
        compiler_func.push(Instruction::I64ExtendI32U);

        let len_idx = compiler_func.alloc_local("__filter_len".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(len_idx));

        // Allocate output (worst case: same size)
        compiler_func.push(Instruction::LocalGet(len_idx));
        compiler_func.push(Instruction::I64Const(8));
        compiler_func.push(Instruction::I64Mul);
        compiler_func.push(Instruction::I64Const(4));
        compiler_func.push(Instruction::I64Add);

        let alloc_idx = self
            .get_func("heap_alloc")
            .ok_or_else(|| WasmError::internal("heap_alloc not found"))?;
        let compiler_func = self.current_function_mut().unwrap();
        compiler_func.push(Instruction::Call(alloc_idx));

        let out_idx = compiler_func.alloc_local("__filter_out".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(out_idx));

        // Initialize indices
        compiler_func.push(Instruction::I64Const(0));
        let i_idx = compiler_func.alloc_local("__filter_i".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(i_idx));

        compiler_func.push(Instruction::I64Const(0));
        let out_i = compiler_func.alloc_local("__filter_out_i".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(out_i));

        // Loop
        compiler_func.push(Instruction::Block(BlockType::Empty));
        compiler_func.push(Instruction::Loop(BlockType::Empty));

        // Check: if i >= len, break
        // I64GeU returns i32 which is what BrIf expects
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::LocalGet(len_idx));
        compiler_func.push(Instruction::I64GeU);
        compiler_func.push(Instruction::BrIf(1));

        // Load element
        compiler_func.push(Instruction::LocalGet(arr_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::I64Const(8));
        compiler_func.push(Instruction::I64Mul);
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Add);
        compiler_func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 4,
            align: 3,
            memory_index: 0,
        }));

        // Store element for predicate check
        let elem_idx = compiler_func.alloc_local("__filter_elem".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalTee(elem_idx));



        // Apply predicate
        self.compile_apply_to_stack(predicate)?;

        let compiler_func = self.current_function_mut().unwrap();

        // If predicate is true, store element
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::If(BlockType::Empty));

        // Store element
        compiler_func.push(Instruction::LocalGet(out_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::LocalGet(out_i));
        compiler_func.push(Instruction::I64Const(8));
        compiler_func.push(Instruction::I64Mul);
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Add);
        compiler_func.push(Instruction::LocalGet(elem_idx));
        compiler_func.push(Instruction::I64Store(wasm_encoder::MemArg {
            offset: 4,
            align: 3,
            memory_index: 0,
        }));

        // Increment output index
        compiler_func.push(Instruction::LocalGet(out_i));
        compiler_func.push(Instruction::I64Const(1));
        compiler_func.push(Instruction::I64Add);
        compiler_func.push(Instruction::LocalSet(out_i));

        compiler_func.push(Instruction::End);

        // Increment i
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::I64Const(1));
        compiler_func.push(Instruction::I64Add);
        compiler_func.push(Instruction::LocalSet(i_idx));

        // Continue
        compiler_func.push(Instruction::Br(0));
        compiler_func.push(Instruction::End);
        compiler_func.push(Instruction::End);

        // Write final length
        compiler_func.push(Instruction::LocalGet(out_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::LocalGet(out_i));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Store(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));

        // Return output
        compiler_func.push(Instruction::LocalGet(out_idx));

        Ok(())
    }

    /// Compile sort morpheme (σ).
    fn compile_sort(&mut self, _field: Option<&crate::ast::Ident>) -> WasmResult<()> {
        // Call runtime sort function
        let sort_idx = self
            .get_func("array_sort")
            .ok_or_else(|| WasmError::internal("array_sort not found"))?;

        let func = self.current_function_mut().unwrap();
        // Wrap I64 to I32 for import call
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::Call(sort_idx));
        // Extend I32 result back to I64
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile reduce morpheme (ρ).
    fn compile_reduce(&mut self, reducer: &Expr) -> WasmResult<()> {
        let compiler_func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store input array
        let arr_idx = compiler_func.alloc_local("__reduce_arr".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(arr_idx));

        // Get length
        compiler_func.push(Instruction::LocalGet(arr_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Load(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));
        compiler_func.push(Instruction::I64ExtendI32U);

        let len_idx = compiler_func.alloc_local("__reduce_len".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(len_idx));

        // Initialize accumulator with first element
        compiler_func.push(Instruction::LocalGet(arr_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 4,
            align: 3,
            memory_index: 0,
        }));

        let acc_idx = compiler_func.alloc_local("__reduce_acc".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(acc_idx));

        // Start from index 1
        compiler_func.push(Instruction::I64Const(1));
        let i_idx = compiler_func.alloc_local("__reduce_i".to_string(), ValType::I64);
        compiler_func.push(Instruction::LocalSet(i_idx));

        // Loop
        compiler_func.push(Instruction::Block(BlockType::Empty));
        compiler_func.push(Instruction::Loop(BlockType::Empty));

        // Check: if i >= len, break
        // I64GeU returns i32 which is what BrIf expects
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::LocalGet(len_idx));
        compiler_func.push(Instruction::I64GeU);
        compiler_func.push(Instruction::BrIf(1));

        // Load accumulator
        compiler_func.push(Instruction::LocalGet(acc_idx));

        // Load element
        compiler_func.push(Instruction::LocalGet(arr_idx));
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::I64Const(8));
        compiler_func.push(Instruction::I64Mul);
        compiler_func.push(Instruction::I32WrapI64);
        compiler_func.push(Instruction::I32Add);
        compiler_func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 4,
            align: 3,
            memory_index: 0,
        }));



        // Apply reducer (takes acc, elem on stack, returns new acc)
        self.compile_apply_binary(reducer)?;

        let compiler_func = self.current_function_mut().unwrap();

        // Store new accumulator
        compiler_func.push(Instruction::LocalSet(acc_idx));

        // Increment i
        compiler_func.push(Instruction::LocalGet(i_idx));
        compiler_func.push(Instruction::I64Const(1));
        compiler_func.push(Instruction::I64Add);
        compiler_func.push(Instruction::LocalSet(i_idx));

        // Continue
        compiler_func.push(Instruction::Br(0));
        compiler_func.push(Instruction::End);
        compiler_func.push(Instruction::End);

        // Return accumulator
        compiler_func.push(Instruction::LocalGet(acc_idx));

        Ok(())
    }

    /// Compile sum reduction (ρ+ / Σ).
    fn compile_reduce_sum(&mut self) -> WasmResult<()> {
        // Call runtime sum function: I32 -> I64
        let sum_idx = self
            .get_func("array_sum")
            .ok_or_else(|| WasmError::internal("array_sum not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64); // Convert array ptr to I32
        func.push(Instruction::Call(sum_idx));
        // Return is already I64

        Ok(())
    }

    /// Compile product reduction (ρ* / Π).
    fn compile_reduce_prod(&mut self) -> WasmResult<()> {
        // Call runtime product function: I32 -> I64
        let prod_idx = self
            .get_func("array_product")
            .ok_or_else(|| WasmError::internal("array_product not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64); // Convert array ptr to I32
        func.push(Instruction::Call(prod_idx));
        // Return is already I64

        Ok(())
    }

    /// Compile min reduction.
    fn compile_reduce_min(&mut self) -> WasmResult<()> {
        // Call runtime min function: I32 -> I64
        let min_idx = self
            .get_func("array_min")
            .ok_or_else(|| WasmError::internal("array_min not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64); // Convert array ptr to I32
        func.push(Instruction::Call(min_idx));
        // Return is already I64

        Ok(())
    }

    /// Compile max reduction.
    fn compile_reduce_max(&mut self) -> WasmResult<()> {
        // Call runtime max function: I32 -> I64
        let max_idx = self
            .get_func("array_max")
            .ok_or_else(|| WasmError::internal("array_max not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64); // Convert array ptr to I32
        func.push(Instruction::Call(max_idx));
        // Return is already I64

        Ok(())
    }

    /// Compile all reduction (ρ&).
    fn compile_reduce_all(&mut self) -> WasmResult<()> {
        // Call runtime all function: I32 -> I32
        let all_idx = self
            .get_func("array_all")
            .ok_or_else(|| WasmError::internal("array_all not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64); // Convert array ptr to I32
        func.push(Instruction::Call(all_idx));
        func.push(Instruction::I64ExtendI32U); // Convert result to I64

        Ok(())
    }

    /// Compile any reduction (ρ|).
    fn compile_reduce_any(&mut self) -> WasmResult<()> {
        // Call runtime any function: I32 -> I32
        let any_idx = self
            .get_func("array_any")
            .ok_or_else(|| WasmError::internal("array_any not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::I32WrapI64); // Convert array ptr to I32
        func.push(Instruction::Call(any_idx));
        func.push(Instruction::I64ExtendI32U); // Convert result to I64

        Ok(())
    }

    /// Compile parallel morpheme (∥).
    ///
    /// Wraps another operation to run it in parallel across array elements.
    /// Supported inner operations: Transform, Filter, Reduce
    fn compile_parallel(&mut self, inner: &PipeOp) -> WasmResult<()> {
        match inner {
            PipeOp::Transform(func_expr) => self.compile_parallel_transform(func_expr),
            PipeOp::Filter(pred_expr) => self.compile_parallel_filter(pred_expr),
            PipeOp::Reduce(reducer_expr) => self.compile_parallel_reduce(reducer_expr),
            PipeOp::ReduceSum => self.compile_parallel_reduce_sum(),
            PipeOp::ReduceProd => self.compile_parallel_reduce_prod(),
            _ => Err(WasmError::unsupported(&format!(
                "parallel {:?}",
                std::mem::discriminant(inner)
            ))),
        }
    }

    /// Compile parallel transform (∥{τ{f}}).
    fn compile_parallel_transform(&mut self, func_expr: &Expr) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store input array
        let arr_idx = func.alloc_local("__par_map_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));



        // Compile the function to a table entry
        let func_table_idx = self.compile_closure_to_table(func_expr)?;

        // Get parallel map function
        let par_map_fn = self
            .get_func("morpheme_array_parallel_map")
            .ok_or_else(|| WasmError::internal("morpheme_array_parallel_map not found"))?;

        let func = self.current_function_mut().unwrap();

        // Call array_parallel_map(arr, func_idx) -> new_arr
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Const(func_table_idx as i32));
        func.push(Instruction::Call(par_map_fn));
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile parallel filter (∥{φ{p}}).
    fn compile_parallel_filter(&mut self, pred_expr: &Expr) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store input array
        let arr_idx = func.alloc_local("__par_filter_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));



        // Compile the predicate to a table entry
        let pred_table_idx = self.compile_closure_to_table(pred_expr)?;

        // Get parallel filter function
        let par_filter_fn = self
            .get_func("morpheme_array_parallel_filter")
            .ok_or_else(|| WasmError::internal("morpheme_array_parallel_filter not found"))?;

        let func = self.current_function_mut().unwrap();

        // Call array_parallel_filter(arr, pred_idx) -> new_arr
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Const(pred_table_idx as i32));
        func.push(Instruction::Call(par_filter_fn));
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile parallel reduce (∥{ρ{f}}).
    fn compile_parallel_reduce(&mut self, reducer_expr: &Expr) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store input array
        let arr_idx = func.alloc_local("__par_reduce_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));

        // Get first element as initial value
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 4,
            align: 3,
            memory_index: 0,
        }));

        let init_idx = func.alloc_local("__par_reduce_init".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(init_idx));



        // Compile the reducer to a table entry
        let reducer_table_idx = self.compile_closure_to_table(reducer_expr)?;

        // Get parallel reduce function
        let par_reduce_fn = self
            .get_func("morpheme_array_parallel_reduce")
            .ok_or_else(|| WasmError::internal("morpheme_array_parallel_reduce not found"))?;

        let func = self.current_function_mut().unwrap();

        // Call array_parallel_reduce(arr, reducer_idx, initial) -> result
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Const(reducer_table_idx as i32));
        func.push(Instruction::LocalGet(init_idx));
        func.push(Instruction::Call(par_reduce_fn));

        Ok(())
    }

    /// Compile parallel sum reduction (∥{ρ+}).
    fn compile_parallel_reduce_sum(&mut self) -> WasmResult<()> {
        // For built-in reductions, we can use a pre-registered function
        // For now, fall back to sequential
        self.compile_reduce_sum()
    }

    /// Compile parallel product reduction (∥{ρ*}).
    fn compile_parallel_reduce_prod(&mut self) -> WasmResult<()> {
        // For built-in reductions, fall back to sequential
        self.compile_reduce_prod()
    }

    /// Compile a closure expression to a table entry and return its index.
    fn compile_closure_to_table(&mut self, expr: &Expr) -> WasmResult<u32> {
        match expr {
            Expr::Closure { params, body, is_move: _, return_type: _ } => {
                // Create a new function for this closure (is_move handled in closure compilation)
                let fn_name = format!("__closure_{}", self.functions.len());

                // Determine parameter types (all i64 for now) with names
                let param_types: Vec<(String, ValType)> = params
                    .iter()
                    .map(|p| (get_pattern_name(&p.pattern), ValType::I64))
                    .collect();
                let result_types = vec![ValType::I64];

                // Get just the types for type registration
                let just_types: Vec<ValType> = param_types.iter().map(|(_, t)| *t).collect();
                let type_idx = self.get_or_create_type(just_types, result_types.clone());
                let func_idx = self.imports.import_count() + self.functions.len() as u32;

                let new_func = super::types::CompiledFunction::new(
                    fn_name.clone(),
                    type_idx,
                    func_idx,
                    param_types,
                    result_types,
                    false,
                );

                // Save current function and switch to new one
                let prev_fn_idx = self.current_fn_idx;
                self.functions.push(new_func);
                self.current_fn_idx = Some(self.functions.len() - 1);

                // Compile the body
                self.compile_expr(body)?;

                // Restore previous function
                self.current_fn_idx = prev_fn_idx;

                // Add to function table
                let table_idx = self.add_to_table(func_idx);

                Ok(table_idx)
            }

            Expr::Path(path) => {
                // Reference to existing function
                let name = path
                    .segments
                    .first()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");

                if let Some(func_idx) = self.get_func(name) {
                    let table_idx = self.add_to_table(func_idx);
                    Ok(table_idx)
                } else {
                    Err(WasmError::undefined_function(name))
                }
            }

            _ => Err(WasmError::unsupported("complex closure expression")),
        }
    }

    /// Compile first morpheme (α).
    fn compile_first(&mut self) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Array is on stack, get first element
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 4, // Skip length
            align: 3,
            memory_index: 0,
        }));

        Ok(())
    }

    /// Compile last morpheme (ω).
    fn compile_last(&mut self) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store array pointer
        let arr_idx = func.alloc_local("__last_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));

        // Get length
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Load(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));

        // Calculate offset: (len - 1) * 8 + 4
        func.push(Instruction::I32Const(1));
        func.push(Instruction::I32Sub);
        func.push(Instruction::I32Const(8));
        func.push(Instruction::I32Mul);
        func.push(Instruction::I32Const(4));
        func.push(Instruction::I32Add);

        // Add to base
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

    /// Compile middle morpheme (μ).
    fn compile_middle(&mut self) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        let arr_idx = func.alloc_local("__mid_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));

        // Get length / 2
        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Load(wasm_encoder::MemArg {
            offset: 0,
            align: 2,
            memory_index: 0,
        }));
        func.push(Instruction::I32Const(2));
        func.push(Instruction::I32DivU);

        // Calculate offset
        func.push(Instruction::I32Const(8));
        func.push(Instruction::I32Mul);
        func.push(Instruction::I32Const(4));
        func.push(Instruction::I32Add);

        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Add);

        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));

        Ok(())
    }

    /// Compile choice morpheme (χ - random).
    fn compile_choice(&mut self) -> WasmResult<()> {
        let random_idx = self
            .get_func("array_random_element")
            .ok_or_else(|| WasmError::internal("array_random_element not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(random_idx));

        Ok(())
    }

    /// Compile nth morpheme (ν{n}).
    fn compile_nth(&mut self, n: &Expr) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        let arr_idx = func.alloc_local("__nth_arr".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(arr_idx));



        // Compile index
        self.compile_expr(n)?;

        let func = self.current_function_mut().unwrap();

        // Calculate offset: index * 8 + 4
        func.push(Instruction::I64Const(8));
        func.push(Instruction::I64Mul);
        func.push(Instruction::I64Const(4));
        func.push(Instruction::I64Add);
        func.push(Instruction::I32WrapI64);

        func.push(Instruction::LocalGet(arr_idx));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::I32Add);

        func.push(Instruction::I64Load(wasm_encoder::MemArg {
            offset: 0,
            align: 3,
            memory_index: 0,
        }));

        Ok(())
    }

    /// Compile next morpheme (ξ - iterator next).
    fn compile_next(&mut self) -> WasmResult<()> {
        let next_idx = self
            .get_func("iterator_next")
            .ok_or_else(|| WasmError::internal("iterator_next not found"))?;

        let func = self.current_function_mut().unwrap();
        func.push(Instruction::Call(next_idx));

        Ok(())
    }

    /// Compile a pipe method call.
    fn compile_pipe_method(&mut self, name: &str, args: &[Expr]) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Store receiver
        let recv_idx = func.alloc_local("__pipe_recv".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(recv_idx));



        // Compile arguments
        for arg in args {
            self.compile_expr(arg)?;
        }

        // Look up method as function
        let method_name = name.to_string();
        if let Some(func_idx) = self.get_func(&method_name) {
            // Push receiver back as first argument
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalGet(recv_idx));
            func.push(Instruction::Call(func_idx));
            Ok(())
        } else {
            Err(WasmError::undefined_function(&method_name))
        }
    }

    /// Compile a named morpheme.
    fn compile_named_morpheme(
        &mut self,
        name: &str,
        body: Option<&Expr>,
    ) -> WasmResult<()> {
        match name {
            "map" => {
                if let Some(func) = body {
                    self.compile_transform(func)
                } else {
                    Err(WasmError::missing_morpheme_body("map"))
                }
            }
            "filter" => {
                if let Some(pred) = body {
                    self.compile_filter(pred)
                } else {
                    Err(WasmError::missing_morpheme_body("filter"))
                }
            }
            "reduce" | "fold" => {
                if let Some(func) = body {
                    self.compile_reduce(func)
                } else {
                    Err(WasmError::missing_morpheme_body("reduce"))
                }
            }
            "sort" => self.compile_sort(None),
            "first" => self.compile_first(),
            "last" => self.compile_last(),
            "sum" => self.compile_reduce_sum(),
            "product" => self.compile_reduce_prod(),
            "min" => self.compile_reduce_min(),
            "max" => self.compile_reduce_max(),
            "all" => self.compile_reduce_all(),
            "any" => self.compile_reduce_any(),
            _ => Err(WasmError::unsupported(&format!("named morpheme: {}", name))),
        }
    }

    /// Helper: Apply a unary function to the value on stack.
    fn compile_apply_to_stack(&mut self, func_expr: &Expr) -> WasmResult<()> {
        // Store value, compile function call, apply
        match func_expr {
            Expr::Closure { params, body, is_move: _, return_type: _ } => {
                // Inline closure: bind parameter and compile body (is_move irrelevant for inline)
                if params.len() != 1 {
                    return Err(WasmError::arity_mismatch(1, params.len()));
                }

                // Value is on stack, bind to parameter
                let param_name = get_pattern_name(&params[0].pattern);
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                let idx = func.alloc_local(param_name, ValType::I64);
                func.push(Instruction::LocalSet(idx));

                // Compile body
                self.compile_expr(body)
            }

            Expr::Path(path) => {
                // Function reference - call it
                let name = path.segments.first().map(|s| s.ident.name.as_str()).unwrap_or("");
                if let Some(func_idx) = self.get_func(name) {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                    Ok(())
                } else {
                    Err(WasmError::undefined_function(name))
                }
            }

            // Non-closure expression (e.g., `fibonacci(it)`) - treat as implicit closure with `it` param
            _ => {
                // Bind the stack value to `it`
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                let it_idx = func.alloc_local("it".to_string(), ValType::I64);
                func.push(Instruction::LocalSet(it_idx));

                // Compile the expression (it will reference `it` via local lookup)
                self.compile_expr(func_expr)
            }
        }
    }

    /// Helper: Apply a binary function to two values on stack.
    fn compile_apply_binary(&mut self, func_expr: &Expr) -> WasmResult<()> {
        match func_expr {
            Expr::Closure { params, body, is_move: _, return_type: _ } => {
                // is_move irrelevant for inline binary closure application
                if params.len() != 2 {
                    return Err(WasmError::arity_mismatch(2, params.len()));
                }

                // Bind parameters (reverse order since stack is LIFO)
                let func = self
                    .current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;

                let name2 = get_pattern_name(&params[1].pattern);
                let idx2 = func.alloc_local(name2, ValType::I64);
                func.push(Instruction::LocalSet(idx2));

                let name1 = get_pattern_name(&params[0].pattern);
                let idx1 = func.alloc_local(name1, ValType::I64);
                func.push(Instruction::LocalSet(idx1));

        

                self.compile_expr(body)
            }

            Expr::Path(path) => {
                let name = path.segments.first().map(|s| s.ident.name.as_str()).unwrap_or("");
                if let Some(func_idx) = self.get_func(name) {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::Call(func_idx));
                    Ok(())
                } else {
                    Err(WasmError::undefined_function(name))
                }
            }

            _ => Err(WasmError::unsupported("complex binary function")),
        }
    }
}

/// Get name from a pattern (for closure parameter binding).
fn get_pattern_name(pattern: &crate::ast::Pattern) -> String {
    match pattern {
        crate::ast::Pattern::Ident { name, .. } => name.name.clone(),
        crate::ast::Pattern::Wildcard => "_".to_string(),
        _ => "__param".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Ident, Literal, NumBase, PathSegment, TypePath};
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

    fn make_path(name: &str) -> Expr {
        Expr::Path(TypePath {
            segments: vec![PathSegment {
                ident: make_ident(name),
                generics: None,
            }],
        })
    }

    #[test]
    fn test_compile_first() {
        let mut compiler = create_test_compiler_with_function();

        // Simulate array on stack
        compiler.compile_expr(&make_int(0x1000)).unwrap();

        compiler.compile_first().unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Load(_))));
    }

    #[test]
    fn test_compile_last() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_expr(&make_int(0x1000)).unwrap();

        compiler.compile_last().unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Load(_))));
    }

    #[test]
    fn test_compile_middle() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_expr(&make_int(0x1000)).unwrap();

        compiler.compile_middle().unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32DivU)));
    }

    #[test]
    fn test_compile_nth() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_expr(&make_int(0x1000)).unwrap();

        compiler.compile_nth(&make_int(5)).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Mul)));
    }

    #[test]
    fn test_compile_pipe() {
        let mut compiler = create_test_compiler_with_function();

        // Simulate simple pipe: arr|α
        let pipe_expr = Expr::Pipe {
            expr: Box::new(make_int(0x1000)),
            operations: vec![PipeOp::First],
        };

        compiler.compile_expr(&pipe_expr).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Load(_))));
    }

    #[test]
    fn test_compile_parallel_transform() {
        use crate::ast::{ClosureParam, Pattern};

        let mut compiler = create_test_compiler_with_function();

        // Create parallel transform: arr|∥{τ{|x| x * 2}}
        let closure = Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Ident {
                    name: make_ident("x"),
                    mutable: false,
                    evidentiality: None,
                },
                ty: None,
            }],
            body: Box::new(Expr::Binary {
                left: Box::new(make_path("x")),
                op: crate::ast::BinOp::Mul,
                right: Box::new(make_int(2)),
            }),
            is_move: false,
            return_type: None,
        };

        // Simulate array on stack
        compiler.compile_expr(&make_int(0x1000)).unwrap();

        // Compile parallel transform
        let parallel_op = PipeOp::Parallel(Box::new(PipeOp::Transform(Box::new(closure))));
        compiler.compile_pipe_op(&parallel_op).unwrap();

        let func = compiler.current_function().unwrap();
        // Should call array_parallel_map runtime function
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }

    #[test]
    fn test_compile_parallel_filter() {
        use crate::ast::{ClosureParam, Pattern};

        let mut compiler = create_test_compiler_with_function();

        // Create parallel filter: arr|∥{φ{|x| x > 0}}
        let predicate = Expr::Closure {
            params: vec![ClosureParam {
                pattern: Pattern::Ident {
                    name: make_ident("x"),
                    mutable: false,
                    evidentiality: None,
                },
                ty: None,
            }],
            body: Box::new(Expr::Binary {
                left: Box::new(make_path("x")),
                op: crate::ast::BinOp::Gt,
                right: Box::new(make_int(0)),
            }),
            is_move: false,
            return_type: None,
        };

        // Simulate array on stack
        compiler.compile_expr(&make_int(0x1000)).unwrap();

        // Compile parallel filter
        let parallel_op = PipeOp::Parallel(Box::new(PipeOp::Filter(Box::new(predicate))));
        compiler.compile_pipe_op(&parallel_op).unwrap();

        let func = compiler.current_function().unwrap();
        // Should call array_parallel_filter runtime function
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }

    #[test]
    fn test_compile_parallel_reduce() {
        use crate::ast::{ClosureParam, Pattern};

        let mut compiler = create_test_compiler_with_function();

        // Create parallel reduce: arr|∥{ρ{|a, b| a + b}}
        let reducer = Expr::Closure {
            params: vec![
                ClosureParam {
                    pattern: Pattern::Ident {
                        name: make_ident("a"),
                        mutable: false,
                        evidentiality: None,
                    },
                    ty: None,
                },
                ClosureParam {
                    pattern: Pattern::Ident {
                        name: make_ident("b"),
                        mutable: false,
                        evidentiality: None,
                    },
                    ty: None,
                },
            ],
            body: Box::new(Expr::Binary {
                left: Box::new(make_path("a")),
                op: crate::ast::BinOp::Add,
                right: Box::new(make_path("b")),
            }),
            is_move: false,
            return_type: None,
        };

        // Simulate array on stack
        compiler.compile_expr(&make_int(0x1000)).unwrap();

        // Compile parallel reduce
        let parallel_op = PipeOp::Parallel(Box::new(PipeOp::Reduce(Box::new(reducer))));
        compiler.compile_pipe_op(&parallel_op).unwrap();

        let func = compiler.current_function().unwrap();
        // Should call array_parallel_reduce runtime function
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }

    #[test]
    fn test_compile_parallel_unsupported_inner() {
        let mut compiler = create_test_compiler_with_function();

        // Parallel sort is not supported
        compiler.compile_expr(&make_int(0x1000)).unwrap();

        let parallel_op = PipeOp::Parallel(Box::new(PipeOp::Sort(None)));
        let result = compiler.compile_pipe_op(&parallel_op);

        assert!(result.is_err());
    }
}

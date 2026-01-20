//! Binary and unary operator compilation.
//!
//! Compiles Sigil operators to WASM instructions.

use wasm_encoder::{Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;
use crate::ast::{BinOp, UnaryOp};

impl WasmCompiler {
    /// Emit WASM instructions for a binary operator.
    ///
    /// Assumes both operands are already on the stack.
    pub fn emit_binop(&mut self, op: BinOp) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        match op {
            // Arithmetic
            BinOp::Add => func.push(Instruction::I64Add),
            BinOp::Sub => func.push(Instruction::I64Sub),
            BinOp::Mul => func.push(Instruction::I64Mul),
            BinOp::Div => func.push(Instruction::I64DivS),
            BinOp::Rem => func.push(Instruction::I64RemS),

            // Power operator - calls math::pow
            BinOp::Pow => {
                // Get math_pow function index - lookup before mutable borrow
                let pow_idx = self
                    .get_func("math_pow")
                    .ok_or_else(|| WasmError::internal("math_pow not found"))?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Call(pow_idx));
            }

            // Logical (non-short-circuit versions - see expressions.rs for short-circuit)
            BinOp::And => func.push(Instruction::I64And),
            BinOp::Or => func.push(Instruction::I64Or),

            // Bitwise
            BinOp::BitAnd => func.push(Instruction::I64And),
            BinOp::BitOr => func.push(Instruction::I64Or),
            BinOp::BitXor => func.push(Instruction::I64Xor),
            BinOp::Shl => func.push(Instruction::I64Shl),
            BinOp::Shr => func.push(Instruction::I64ShrS),

            // Comparison - result is i32, extend to i64 for uniform representation
            BinOp::Eq => {
                func.push(Instruction::I64Eq);
                func.push(Instruction::I64ExtendI32U);
            }
            BinOp::Ne => {
                func.push(Instruction::I64Ne);
                func.push(Instruction::I64ExtendI32U);
            }
            BinOp::Lt => {
                func.push(Instruction::I64LtS);
                func.push(Instruction::I64ExtendI32U);
            }
            BinOp::Le => {
                func.push(Instruction::I64LeS);
                func.push(Instruction::I64ExtendI32U);
            }
            BinOp::Gt => {
                func.push(Instruction::I64GtS);
                func.push(Instruction::I64ExtendI32U);
            }
            BinOp::Ge => {
                func.push(Instruction::I64GeS);
                func.push(Instruction::I64ExtendI32U);
            }

            // String concatenation - calls runtime function
            BinOp::Concat => {
                // Both operands are on stack as i64 (string pointers extended)
                // Convert to i32 pointers and call string_concat
                func.push(Instruction::I32WrapI64); // right operand
                let right_local = func.alloc_local("__concat_right".to_string(), ValType::I32);
                func.push(Instruction::LocalSet(right_local));

                func.push(Instruction::I32WrapI64); // left operand

                func.push(Instruction::LocalGet(right_local));

                // Get concat function index
                let concat_idx = self
                    .get_func("string_concat")
                    .ok_or_else(|| WasmError::internal("string_concat not found"))?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Call(concat_idx));
                // Result is i32, extend to i64
                func.push(Instruction::I64ExtendI32U);
            }

            // Matrix/tensor operations - call runtime math functions
            BinOp::MatMul => {
                // Matrix multiplication: A @ B
                // For WASM, this calls a runtime function
                let matmul_idx = self.get_func("math_matmul").ok_or_else(|| {
                    WasmError::internal(
                        "math_matmul not found - matrix operations require runtime support",
                    )
                })?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Call(matmul_idx));
            }

            BinOp::Hadamard => {
                // Hadamard/element-wise product: A ⊙ B
                let hadamard_idx = self.get_func("math_hadamard").ok_or_else(|| {
                    WasmError::internal(
                        "math_hadamard not found - element-wise operations require runtime support",
                    )
                })?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Call(hadamard_idx));
            }

            BinOp::TensorProd => {
                // Tensor/outer product: A ⊗ B
                let tensor_idx = self.get_func("math_tensor_prod").ok_or_else(|| {
                    WasmError::internal(
                        "math_tensor_prod not found - tensor operations require runtime support",
                    )
                })?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Call(tensor_idx));
            }

            BinOp::Convolve => {
                // Convolution/merge: A ⊛ B (holographic shard merging)
                let convolve_idx = self.get_func("math_convolve").ok_or_else(|| {
                    WasmError::internal(
                        "math_convolve not found - convolution operations require runtime support",
                    )
                })?;
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Call(convolve_idx));
            }
        }
        Ok(())
    }

    /// Emit WASM instructions for a unary operator.
    ///
    /// Assumes the operand is already on the stack.
    pub fn emit_unaryop(&mut self, op: UnaryOp) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        match op {
            UnaryOp::Neg => {
                // Negate: multiply by -1
                func.push(Instruction::I64Const(-1));
                func.push(Instruction::I64Mul);
            }
            UnaryOp::Not => {
                // Logical not: eqz then extend
                func.push(Instruction::I64Eqz);
                func.push(Instruction::I64ExtendI32U);
            }
            UnaryOp::Deref | UnaryOp::Ref | UnaryOp::RefMut => {
                // Reference operations - pass through for now
                // In a full implementation, these would involve memory operations
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::literals::create_test_compiler_with_function;
    use wasm_encoder::ValType;

    #[test]
    fn test_emit_add() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Add).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Add));
    }

    #[test]
    fn test_emit_sub() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Sub).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Sub));
    }

    #[test]
    fn test_emit_mul() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Mul).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Mul));
    }

    #[test]
    fn test_emit_div() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Div).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64DivS));
    }

    #[test]
    fn test_emit_rem() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Rem).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64RemS));
    }

    #[test]
    fn test_emit_pow() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Pow).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::Call(_)));
    }

    #[test]
    fn test_emit_and() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::And).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64And));
    }

    #[test]
    fn test_emit_or() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Or).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Or));
    }

    #[test]
    fn test_emit_bitwise_and() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::BitAnd).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64And));
    }

    #[test]
    fn test_emit_bitwise_or() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::BitOr).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Or));
    }

    #[test]
    fn test_emit_bitwise_xor() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::BitXor).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Xor));
    }

    #[test]
    fn test_emit_shl() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Shl).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Shl));
    }

    #[test]
    fn test_emit_shr() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Shr).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64ShrS));
    }

    #[test]
    fn test_emit_eq() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Eq).unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 2);
        assert!(matches!(func.instructions[0], Instruction::I64Eq));
        assert!(matches!(func.instructions[1], Instruction::I64ExtendI32U));
    }

    #[test]
    fn test_emit_ne() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Ne).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Ne));
    }

    #[test]
    fn test_emit_lt() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Lt).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64LtS));
    }

    #[test]
    fn test_emit_le() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Le).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64LeS));
    }

    #[test]
    fn test_emit_gt() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Gt).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64GtS));
    }

    #[test]
    fn test_emit_ge() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_binop(BinOp::Ge).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64GeS));
    }

    #[test]
    fn test_emit_concat() {
        // Concat requires string_concat function, which is registered by default
        let mut compiler = WasmCompiler::new();
        let type_idx = compiler.get_or_create_type(vec![], vec![ValType::I64]);
        let func_idx = compiler.imports.import_count();
        let func = super::super::types::CompiledFunction::new(
            "test".to_string(),
            type_idx,
            func_idx,
            vec![],
            vec![ValType::I64],
            false,
        );
        compiler.functions.push(func);
        compiler.current_fn_idx = Some(0);

        compiler.emit_binop(BinOp::Concat).unwrap();

        let func = compiler.current_function().unwrap();
        // Should have Call instruction for string_concat
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }

    #[test]
    fn test_emit_unary_neg() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_unaryop(UnaryOp::Neg).unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 2);
        assert!(matches!(func.instructions[0], Instruction::I64Const(-1)));
        assert!(matches!(func.instructions[1], Instruction::I64Mul));
    }

    #[test]
    fn test_emit_unary_not() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_unaryop(UnaryOp::Not).unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 2);
        assert!(matches!(func.instructions[0], Instruction::I64Eqz));
        assert!(matches!(func.instructions[1], Instruction::I64ExtendI32U));
    }

    #[test]
    fn test_emit_deref_passthrough() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_unaryop(UnaryOp::Deref).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func.instructions.is_empty()); // No-op for now
    }

    #[test]
    fn test_emit_ref_passthrough() {
        let mut compiler = create_test_compiler_with_function();
        compiler.emit_unaryop(UnaryOp::Ref).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(func.instructions.is_empty()); // No-op for now
    }
}

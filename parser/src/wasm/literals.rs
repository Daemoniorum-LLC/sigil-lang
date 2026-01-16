//! Literal value compilation.
//!
//! Compiles Sigil literal values to WASM instructions.

use wasm_encoder::Instruction;
#[cfg(test)]
use wasm_encoder::ValType;

use super::error::{WasmError, WasmResult};
#[cfg(test)]
use super::types::CompiledFunction;
use super::WasmCompiler;
use crate::ast::{Literal, NumBase};

impl WasmCompiler {
    /// Compile a literal value, pushing the result onto the WASM stack.
    pub fn compile_literal(&mut self, lit: &Literal) -> WasmResult<()> {
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        match lit {
            Literal::Int { value, base, .. } => {
                let v = parse_int(value, *base)?;
                func.push(Instruction::I64Const(v));
            }
            Literal::Float { value, .. } => {
                let v: f64 = value
                    .parse()
                    .map_err(|_| WasmError::parse(format!("invalid float: {}", value)))?;
                // Store float bits as i64 for uniform value representation
                func.push(Instruction::I64Const(v.to_bits() as i64));
            }
            Literal::Bool(b) => {
                func.push(Instruction::I64Const(if *b { 1 } else { 0 }));
            }
            Literal::String(s) | Literal::MultiLineString(s) | Literal::RawString(s) => {
                let offset = self.add_string(s);
                // Get current function again after borrow ends
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I32Const(offset as i32));
                func.push(Instruction::I64ExtendI32U);
            }
            Literal::ByteString(bytes) => {
                // Store byte string in data section
                let offset = self.data_offset;
                let mut data = (bytes.len() as u32).to_le_bytes().to_vec();
                data.extend(bytes);
                self.data_segments.push((offset, data.clone()));
                self.data_offset += data.len() as u32;
                self.data_offset = (self.data_offset + 7) & !7; // Align

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I32Const(offset as i32));
                func.push(Instruction::I64ExtendI32U);
            }
            Literal::SigilStringSql(s) | Literal::SigilStringRoute(s) => {
                let offset = self.add_string(s);
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I32Const(offset as i32));
                func.push(Instruction::I64ExtendI32U);
            }
            Literal::InterpolatedString { .. } => {
                // Interpolated strings require expression compilation
                // For now, return a placeholder
                return Err(WasmError::unsupported("interpolated strings"));
            }
            Literal::Char(c) => {
                func.push(Instruction::I64Const(*c as i64));
            }
            Literal::ByteChar(b) => {
                func.push(Instruction::I64Const(*b as i64));
            }
            Literal::Null | Literal::Empty | Literal::Circle => {
                func.push(Instruction::I64Const(0));
            }
            Literal::Infinity => {
                // Positive infinity as f64 bits
                let inf_bits = f64::INFINITY.to_bits() as i64;
                func.push(Instruction::I64Const(inf_bits));
            }
        }
        Ok(())
    }
}

/// Parse an integer literal with the given base.
fn parse_int(value: &str, base: NumBase) -> WasmResult<i64> {
    let radix = match base {
        NumBase::Decimal => 10,
        NumBase::Binary => 2,
        NumBase::Octal => 8,
        NumBase::Hex => 16,
        NumBase::Vigesimal => 20,
        NumBase::Sexagesimal => 60,
        NumBase::Duodecimal => 12,
        NumBase::Explicit(n) => n as u32,
    };

    // Handle negative numbers
    let (sign, digits) = if value.starts_with('-') {
        (-1i64, &value[1..])
    } else {
        (1, value.as_ref())
    };

    // Remove underscores and parse
    let clean: String = digits.chars().filter(|c| *c != '_').collect();

    i64::from_str_radix(&clean, radix)
        .map(|v| v * sign)
        .map_err(|_| WasmError::parse(format!("invalid integer: {}", value)))
}

/// Helper to create test compilation context.
#[cfg(test)]
pub(crate) fn create_test_compiler_with_function() -> WasmCompiler {
    let mut compiler = WasmCompiler::new();

    // Create a test function
    let type_idx = compiler.get_or_create_type(vec![], vec![ValType::I64]);
    let func_idx = compiler.imports.import_count();

    let func = CompiledFunction::new(
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

/// Helper to create test compiler with heap_alloc function registered.
#[cfg(test)]
pub(crate) fn create_test_compiler_with_heap_alloc() -> WasmCompiler {
    let mut compiler = WasmCompiler::new();

    // Add heap_alloc as import: (i64) -> i64
    let heap_alloc_idx = compiler.imports.add_import(
        "runtime",
        "heap_alloc",
        vec![ValType::I64],
        vec![ValType::I64],
    );

    // Record in func_map with the qualified name that get_func expects
    compiler
        .func_map
        .insert("heap_alloc".to_string(), heap_alloc_idx);

    // Create a test function (func_idx = import_count since import takes earlier indices)
    let test_type = compiler.get_or_create_type(vec![], vec![ValType::I64]);
    let test_func_idx = compiler.imports.import_count();

    let func = CompiledFunction::new(
        "test".to_string(),
        test_type,
        test_func_idx,
        vec![],
        vec![ValType::I64],
        false,
    );

    compiler.functions.push(func);
    compiler.current_fn_idx = Some(0);

    compiler
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::Instruction;

    #[test]
    fn test_compile_int_literal() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Int {
            value: "42".to_string(),
            base: NumBase::Decimal,
            suffix: None,
        };

        compiler.compile_literal(&lit).unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 1);
        assert!(matches!(func.instructions[0], Instruction::I64Const(42)));
    }

    #[test]
    fn test_compile_negative_int() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Int {
            value: "-123".to_string(),
            base: NumBase::Decimal,
            suffix: None,
        };

        compiler.compile_literal(&lit).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(-123)));
    }

    #[test]
    fn test_compile_hex_int() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Int {
            value: "ff".to_string(),
            base: NumBase::Hex,
            suffix: None,
        };

        compiler.compile_literal(&lit).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(255)));
    }

    #[test]
    fn test_compile_binary_int() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Int {
            value: "1010".to_string(),
            base: NumBase::Binary,
            suffix: None,
        };

        compiler.compile_literal(&lit).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(10)));
    }

    #[test]
    fn test_compile_octal_int() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Int {
            value: "77".to_string(),
            base: NumBase::Octal,
            suffix: None,
        };

        compiler.compile_literal(&lit).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(63)));
    }

    #[test]
    fn test_compile_int_with_underscores() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Int {
            value: "1_000_000".to_string(),
            base: NumBase::Decimal,
            suffix: None,
        };

        compiler.compile_literal(&lit).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(
            func.instructions[0],
            Instruction::I64Const(1_000_000)
        ));
    }

    #[test]
    fn test_compile_float_literal() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Float {
            value: "3.14".to_string(),
            suffix: None,
        };

        compiler.compile_literal(&lit).unwrap();

        let func = compiler.current_function().unwrap();
        if let Instruction::I64Const(bits) = func.instructions[0] {
            let f = f64::from_bits(bits as u64);
            assert!((f - 3.14).abs() < 0.0001);
        } else {
            panic!("Expected I64Const");
        }
    }

    #[test]
    fn test_compile_bool_true() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::Bool(true)).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(1)));
    }

    #[test]
    fn test_compile_bool_false() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::Bool(false)).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(0)));
    }

    #[test]
    fn test_compile_null() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::Null).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(0)));
    }

    #[test]
    fn test_compile_empty() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::Empty).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(0)));
    }

    #[test]
    fn test_compile_circle() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::Circle).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(0)));
    }

    #[test]
    fn test_compile_char() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::Char('A')).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(65)));
    }

    #[test]
    fn test_compile_byte_char() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::ByteChar(0x41)).unwrap();

        let func = compiler.current_function().unwrap();
        assert!(matches!(func.instructions[0], Instruction::I64Const(65)));
    }

    #[test]
    fn test_compile_string() {
        let mut compiler = create_test_compiler_with_function();

        compiler
            .compile_literal(&Literal::String("hello".to_string()))
            .unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 2);
        assert!(matches!(func.instructions[0], Instruction::I32Const(_)));
        assert!(matches!(func.instructions[1], Instruction::I64ExtendI32U));
        assert!(!compiler.data_segments.is_empty());
    }

    #[test]
    fn test_compile_multiline_string() {
        let mut compiler = create_test_compiler_with_function();

        compiler
            .compile_literal(&Literal::MultiLineString("line1\nline2".to_string()))
            .unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 2);
    }

    #[test]
    fn test_compile_raw_string() {
        let mut compiler = create_test_compiler_with_function();

        compiler
            .compile_literal(&Literal::RawString(r"raw\nstring".to_string()))
            .unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 2);
    }

    #[test]
    fn test_compile_byte_string() {
        let mut compiler = create_test_compiler_with_function();

        compiler
            .compile_literal(&Literal::ByteString(vec![0x48, 0x69]))
            .unwrap();

        let func = compiler.current_function().unwrap();
        assert_eq!(func.instructions.len(), 2);
        assert!(!compiler.data_segments.is_empty());
    }

    #[test]
    fn test_compile_infinity() {
        let mut compiler = create_test_compiler_with_function();

        compiler.compile_literal(&Literal::Infinity).unwrap();

        let func = compiler.current_function().unwrap();
        if let Instruction::I64Const(bits) = func.instructions[0] {
            let f = f64::from_bits(bits as u64);
            assert!(f.is_infinite() && f.is_sign_positive());
        } else {
            panic!("Expected I64Const");
        }
    }

    #[test]
    fn test_invalid_int_literal() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::Int {
            value: "not_a_number".to_string(),
            base: NumBase::Decimal,
            suffix: None,
        };

        let result = compiler.compile_literal(&lit);
        assert!(result.is_err());
    }

    #[test]
    fn test_not_in_function_context() {
        let mut compiler = WasmCompiler::new();

        let result = compiler.compile_literal(&Literal::Bool(true));
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolated_string_unsupported() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::InterpolatedString { parts: vec![] };

        let result = compiler.compile_literal(&lit);
        assert!(result.is_err());
    }
}

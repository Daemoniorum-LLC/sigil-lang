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
use crate::ast::{InterpolationPart, Literal, NumBase};

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
            Literal::InterpolatedString { parts } => {
                // Drop the func borrow before we start - we'll get it again as needed
                drop(func);

                // Handle empty interpolated string
                if parts.is_empty() {
                    let offset = self.add_string("");
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I32Const(offset as i32));
                    func.push(Instruction::I64ExtendI32U);
                    return Ok(());
                }

                let mut first = true;

                for part in parts {
                    match part {
                        InterpolationPart::Text(s) => {
                            // Add string literal to data section
                            let str_offset = self.add_string(s);
                            let func = self.current_function_mut().unwrap();
                            func.push(Instruction::I32Const(str_offset as i32));

                            if !first {
                                // Concat with accumulator (which is below on stack)
                                // Stack: [acc, new] -> string.concat(acc, new)
                                // But we just pushed new on top, so we need to swap
                                // Actually: concat(str1, str2) takes bottom, top order
                                // So if acc is below and new is on top, that's correct order
                                // But wait - we pushed new AFTER acc, so order is [acc, new]
                                // That's the right order for concat
                                // WRONG: concat expects (str1, str2) where str1 is FIRST pushed
                                // So we need to swap: new is on top, acc is below
                                // Actually looking at WASM call convention: first arg is popped first
                                // So Call(concat) pops [top=str2, next=str1]
                                // We have [acc, new] on stack, str1=acc, str2=new. That's right!
                                let concat_idx =
                                    self.imports.get_func("string_concat").ok_or_else(|| {
                                        WasmError::internal("string_concat not registered")
                                    })?;
                                let func = self.current_function_mut().unwrap();
                                func.push(Instruction::Call(concat_idx));
                            }
                            first = false;
                        }
                        InterpolationPart::Expr(expr) => {
                            // Compile expression - puts i64 value on stack
                            self.compile_expr(expr)?;

                            // Convert to string using string_from_int
                            let from_int_idx =
                                self.imports.get_func("string_from_int").ok_or_else(|| {
                                    WasmError::internal("string_from_int not registered")
                                })?;
                            let func = self.current_function_mut().unwrap();
                            func.push(Instruction::Call(from_int_idx));
                            // Result is i32 string pointer

                            if !first {
                                // Concat with accumulator
                                let concat_idx =
                                    self.imports.get_func("string_concat").ok_or_else(|| {
                                        WasmError::internal("string_concat not registered")
                                    })?;
                                let func = self.current_function_mut().unwrap();
                                func.push(Instruction::Call(concat_idx));
                            }
                            first = false;
                        }
                    }
                }

                // Extend i32 string pointer to i64 for uniform value representation
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64ExtendI32U);
                return Ok(());
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
    fn test_interpolated_string_empty() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::InterpolatedString { parts: vec![] };

        // Empty interpolated string should compile successfully
        let result = compiler.compile_literal(&lit);
        assert!(result.is_ok());

        let func = compiler.current_function().unwrap();
        // Should push string pointer (I32Const) and extend to I64
        assert!(func.instructions.len() >= 2);
    }

    #[test]
    fn test_interpolated_string_text_only() {
        let mut compiler = create_test_compiler_with_function();

        let lit = Literal::InterpolatedString {
            parts: vec![InterpolationPart::Text("hello".to_string())],
        };

        // Single text part should work without concat
        let result = compiler.compile_literal(&lit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_interpolated_string_concat() {
        let mut compiler = create_test_compiler_with_function();

        // Two text parts need string_concat
        let lit = Literal::InterpolatedString {
            parts: vec![
                InterpolationPart::Text("hello ".to_string()),
                InterpolationPart::Text("world".to_string()),
            ],
        };

        // Should succeed - default compiler has string_concat registered
        let result = compiler.compile_literal(&lit);
        assert!(result.is_ok());

        // Should have called string_concat import
        let func = compiler.current_function().unwrap();
        assert!(func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_))));
    }
}

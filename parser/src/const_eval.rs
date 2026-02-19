//! Compile-time Constant Evaluation for Const Generics
//!
//! This module handles the evaluation of compile-time constant expressions
//! for use in const generic parameters. It supports:
//!
//! - Integer literals and arithmetic
//! - Array shape literals like `[4, 8]` → `Shape2<4, 8>`
//! - Constant expressions in type position
//!
//! Part of Phase 4 of the generic monomorphization implementation.

use std::collections::HashMap;

use crate::ast::{Expr, Literal, BinOp, UnaryOp};
use crate::typeck::Type;

/// Result of const evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    /// Integer constant
    Int(i64),
    /// Unsigned integer constant
    UInt(u64),
    /// Float constant (for completeness, though rarely used in const generics)
    Float(f64),
    /// Boolean constant
    Bool(bool),
    /// Array of const values (for shape literals)
    Array(Vec<ConstValue>),
    /// Tuple of const values
    Tuple(Vec<ConstValue>),
}

impl ConstValue {
    /// Convert to i64 if possible
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ConstValue::Int(v) => Some(*v),
            ConstValue::UInt(v) => {
                if *v <= i64::MAX as u64 {
                    Some(*v as i64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Convert to usize if possible (for array dimensions)
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ConstValue::Int(v) if *v >= 0 => Some(*v as usize),
            ConstValue::UInt(v) if *v <= usize::MAX as u64 => Some(*v as usize),
            _ => None,
        }
    }

    /// Convert to Type::ConstGeneric if this is an integer
    pub fn to_const_generic(&self) -> Option<Type> {
        self.as_i64().map(Type::ConstGeneric)
    }
}

/// Error during const evaluation
#[derive(Debug, Clone)]
pub struct ConstEvalError {
    pub message: String,
    pub kind: ConstEvalErrorKind,
}

#[derive(Debug, Clone)]
pub enum ConstEvalErrorKind {
    /// Expression is not a constant
    NotConst,
    /// Division by zero
    DivByZero,
    /// Integer overflow
    Overflow,
    /// Type mismatch in operation
    TypeMismatch,
    /// Unknown variable/constant
    UnknownIdent(String),
    /// Unsupported operation
    Unsupported(String),
}

impl ConstEvalError {
    pub fn not_const(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: ConstEvalErrorKind::NotConst,
        }
    }

    pub fn div_by_zero() -> Self {
        Self {
            message: "division by zero".to_string(),
            kind: ConstEvalErrorKind::DivByZero,
        }
    }

    pub fn overflow() -> Self {
        Self {
            message: "integer overflow".to_string(),
            kind: ConstEvalErrorKind::Overflow,
        }
    }

    pub fn unknown_ident(name: &str) -> Self {
        Self {
            message: format!("unknown constant: {}", name),
            kind: ConstEvalErrorKind::UnknownIdent(name.to_string()),
        }
    }

    pub fn unsupported(what: impl Into<String>) -> Self {
        let msg = what.into();
        Self {
            message: format!("unsupported in const context: {}", msg),
            kind: ConstEvalErrorKind::Unsupported(msg),
        }
    }
}

/// Compile-time constant evaluator
#[derive(Debug, Default)]
pub struct ConstEvaluator {
    /// Named constants (const items)
    constants: HashMap<String, ConstValue>,
}

impl ConstEvaluator {
    /// Create a new const evaluator
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a named constant
    pub fn register_const(&mut self, name: String, value: ConstValue) {
        self.constants.insert(name, value);
    }

    /// Evaluate an expression to a constant value
    pub fn eval(&self, expr: &Expr) -> Result<ConstValue, ConstEvalError> {
        match expr {
            Expr::Literal(lit) => self.eval_literal(lit),
            Expr::Path(path) => {
                // Simple identifier reference
                if path.segments.len() == 1 && path.segments[0].generics.is_none() {
                    self.eval_ident(&path.segments[0].ident.name)
                } else {
                    Err(ConstEvalError::not_const("complex paths are not const-evaluable"))
                }
            }
            Expr::Binary { op, left, right } => self.eval_binary(*op, left, right),
            Expr::Unary { op, expr: inner } => self.eval_unary(*op, inner),
            Expr::Array(elements) => self.eval_array(elements),
            Expr::Tuple(elements) => self.eval_tuple(elements),
            Expr::Cast { expr: inner, .. } => {
                // For const evaluation, we allow casts but warn about potential precision loss
                self.eval(inner)
            }
            _ => Err(ConstEvalError::not_const(format!(
                "expression is not const-evaluable"
            ))),
        }
    }

    /// Evaluate a literal
    fn eval_literal(&self, lit: &Literal) -> Result<ConstValue, ConstEvalError> {
        match lit {
            Literal::Int { value, suffix, .. } => {
                let v = value.parse::<i64>().map_err(|_| {
                    ConstEvalError::overflow()
                })?;
                // Check if suffix indicates unsigned
                if suffix.as_ref().map_or(false, |s| s.starts_with('u')) {
                    Ok(ConstValue::UInt(v as u64))
                } else {
                    Ok(ConstValue::Int(v))
                }
            }
            Literal::Float { value, .. } => {
                let v = value.parse::<f64>().map_err(|_| {
                    ConstEvalError::overflow()
                })?;
                Ok(ConstValue::Float(v))
            }
            Literal::Bool(b) => Ok(ConstValue::Bool(*b)),
            // All other literal types are not const-evaluable
            Literal::Char(_) => Err(ConstEvalError::unsupported("char literals")),
            Literal::ByteChar(_) => Err(ConstEvalError::unsupported("byte char literals")),
            Literal::String(_) => Err(ConstEvalError::unsupported("string literals")),
            Literal::MultiLineString(_) => Err(ConstEvalError::unsupported("multi-line string literals")),
            Literal::RawString(_) => Err(ConstEvalError::unsupported("raw string literals")),
            Literal::ByteString(_) => Err(ConstEvalError::unsupported("byte string literals")),
            Literal::InterpolatedString { .. } => Err(ConstEvalError::unsupported("interpolated string literals")),
            Literal::SigilStringSql(_) => Err(ConstEvalError::unsupported("SQL sigil string literals")),
            Literal::SigilStringRoute(_) => Err(ConstEvalError::unsupported("route sigil string literals")),
            Literal::Null => Err(ConstEvalError::unsupported("null literals")),
            Literal::Empty => Err(ConstEvalError::unsupported("empty set literals")),
            Literal::Infinity => Err(ConstEvalError::unsupported("infinity literals")),
            Literal::Circle => Err(ConstEvalError::unsupported("circle literals")),
        }
    }

    /// Evaluate an identifier (must be a known constant)
    fn eval_ident(&self, name: &str) -> Result<ConstValue, ConstEvalError> {
        self.constants
            .get(name)
            .cloned()
            .ok_or_else(|| ConstEvalError::unknown_ident(name))
    }

    /// Evaluate a binary operation
    fn eval_binary(
        &self,
        op: BinOp,
        left: &Expr,
        right: &Expr,
    ) -> Result<ConstValue, ConstEvalError> {
        let lhs = self.eval(left)?;
        let rhs = self.eval(right)?;

        match (lhs, rhs) {
            (ConstValue::Int(l), ConstValue::Int(r)) => self.eval_int_binary(op, l, r),
            (ConstValue::UInt(l), ConstValue::UInt(r)) => {
                let result = self.eval_uint_binary(op, l, r)?;
                Ok(result)
            }
            (ConstValue::Int(l), ConstValue::UInt(r)) => {
                // Coerce to signed
                self.eval_int_binary(op, l, r as i64)
            }
            (ConstValue::UInt(l), ConstValue::Int(r)) => {
                // Coerce to signed
                self.eval_int_binary(op, l as i64, r)
            }
            (ConstValue::Float(l), ConstValue::Float(r)) => self.eval_float_binary(op, l, r),
            (ConstValue::Bool(l), ConstValue::Bool(r)) => self.eval_bool_binary(op, l, r),
            _ => Err(ConstEvalError {
                message: "type mismatch in binary operation".to_string(),
                kind: ConstEvalErrorKind::TypeMismatch,
            }),
        }
    }

    /// Evaluate integer binary operation
    fn eval_int_binary(&self, op: BinOp, l: i64, r: i64) -> Result<ConstValue, ConstEvalError> {
        let result = match op {
            BinOp::Add => l.checked_add(r).ok_or_else(ConstEvalError::overflow)?,
            BinOp::Sub => l.checked_sub(r).ok_or_else(ConstEvalError::overflow)?,
            BinOp::Mul => l.checked_mul(r).ok_or_else(ConstEvalError::overflow)?,
            BinOp::Div => {
                if r == 0 {
                    return Err(ConstEvalError::div_by_zero());
                }
                l.checked_div(r).ok_or_else(ConstEvalError::overflow)?
            }
            BinOp::Rem => {
                if r == 0 {
                    return Err(ConstEvalError::div_by_zero());
                }
                l.checked_rem(r).ok_or_else(ConstEvalError::overflow)?
            }
            BinOp::BitAnd => l & r,
            BinOp::BitOr => l | r,
            BinOp::BitXor => l ^ r,
            BinOp::Shl => {
                if r < 0 || r >= 64 {
                    return Err(ConstEvalError::overflow());
                }
                l.checked_shl(r as u32).ok_or_else(ConstEvalError::overflow)?
            }
            BinOp::Shr => {
                if r < 0 || r >= 64 {
                    return Err(ConstEvalError::overflow());
                }
                l.checked_shr(r as u32).ok_or_else(ConstEvalError::overflow)?
            }
            // Comparison operators return bool
            BinOp::Eq => return Ok(ConstValue::Bool(l == r)),
            BinOp::Ne => return Ok(ConstValue::Bool(l != r)),
            BinOp::Lt => return Ok(ConstValue::Bool(l < r)),
            BinOp::Le => return Ok(ConstValue::Bool(l <= r)),
            BinOp::Gt => return Ok(ConstValue::Bool(l > r)),
            BinOp::Ge => return Ok(ConstValue::Bool(l >= r)),
            // Logical operators don't apply to integers
            BinOp::And | BinOp::Or => {
                return Err(ConstEvalError::unsupported("logical operators on integers"))
            }
            // Other operators not applicable to integers
            BinOp::Concat | BinOp::MatMul | BinOp::Hadamard | BinOp::TensorProd | BinOp::Convolve | BinOp::Pow => {
                return Err(ConstEvalError::unsupported(format!("{:?} on integers", op)))
            }
        };
        Ok(ConstValue::Int(result))
    }

    /// Evaluate unsigned integer binary operation
    fn eval_uint_binary(&self, op: BinOp, l: u64, r: u64) -> Result<ConstValue, ConstEvalError> {
        let result = match op {
            BinOp::Add => l.checked_add(r).ok_or_else(ConstEvalError::overflow)?,
            BinOp::Sub => l.checked_sub(r).ok_or_else(ConstEvalError::overflow)?,
            BinOp::Mul => l.checked_mul(r).ok_or_else(ConstEvalError::overflow)?,
            BinOp::Div => {
                if r == 0 {
                    return Err(ConstEvalError::div_by_zero());
                }
                l / r
            }
            BinOp::Rem => {
                if r == 0 {
                    return Err(ConstEvalError::div_by_zero());
                }
                l % r
            }
            BinOp::BitAnd => l & r,
            BinOp::BitOr => l | r,
            BinOp::BitXor => l ^ r,
            BinOp::Shl => {
                if r >= 64 {
                    return Err(ConstEvalError::overflow());
                }
                l.checked_shl(r as u32).ok_or_else(ConstEvalError::overflow)?
            }
            BinOp::Shr => {
                if r >= 64 {
                    return Err(ConstEvalError::overflow());
                }
                l.checked_shr(r as u32).ok_or_else(ConstEvalError::overflow)?
            }
            BinOp::Eq => return Ok(ConstValue::Bool(l == r)),
            BinOp::Ne => return Ok(ConstValue::Bool(l != r)),
            BinOp::Lt => return Ok(ConstValue::Bool(l < r)),
            BinOp::Le => return Ok(ConstValue::Bool(l <= r)),
            BinOp::Gt => return Ok(ConstValue::Bool(l > r)),
            BinOp::Ge => return Ok(ConstValue::Bool(l >= r)),
            BinOp::And | BinOp::Or => {
                return Err(ConstEvalError::unsupported("logical operators on integers"))
            }
            BinOp::Concat | BinOp::MatMul | BinOp::Hadamard | BinOp::TensorProd | BinOp::Convolve | BinOp::Pow => {
                return Err(ConstEvalError::unsupported(format!("{:?} on integers", op)))
            }
        };
        Ok(ConstValue::UInt(result))
    }

    /// Evaluate float binary operation
    fn eval_float_binary(&self, op: BinOp, l: f64, r: f64) -> Result<ConstValue, ConstEvalError> {
        let result = match op {
            BinOp::Add => l + r,
            BinOp::Sub => l - r,
            BinOp::Mul => l * r,
            BinOp::Div => l / r,
            BinOp::Rem => l % r,
            BinOp::Eq => return Ok(ConstValue::Bool(l == r)),
            BinOp::Ne => return Ok(ConstValue::Bool(l != r)),
            BinOp::Lt => return Ok(ConstValue::Bool(l < r)),
            BinOp::Le => return Ok(ConstValue::Bool(l <= r)),
            BinOp::Gt => return Ok(ConstValue::Bool(l > r)),
            BinOp::Ge => return Ok(ConstValue::Bool(l >= r)),
            _ => return Err(ConstEvalError::unsupported(format!("{:?} on floats", op))),
        };
        Ok(ConstValue::Float(result))
    }

    /// Evaluate boolean binary operation
    fn eval_bool_binary(&self, op: BinOp, l: bool, r: bool) -> Result<ConstValue, ConstEvalError> {
        let result = match op {
            BinOp::And => l && r,
            BinOp::Or => l || r,
            BinOp::Eq => l == r,
            BinOp::Ne => l != r,
            BinOp::BitAnd => l & r,
            BinOp::BitOr => l | r,
            BinOp::BitXor => l ^ r,
            _ => return Err(ConstEvalError::unsupported(format!("{:?} on booleans", op))),
        };
        Ok(ConstValue::Bool(result))
    }

    /// Evaluate unary operation
    fn eval_unary(&self, op: UnaryOp, inner: &Expr) -> Result<ConstValue, ConstEvalError> {
        let val = self.eval(inner)?;

        match (op, val) {
            (UnaryOp::Neg, ConstValue::Int(v)) => {
                Ok(ConstValue::Int(v.checked_neg().ok_or_else(ConstEvalError::overflow)?))
            }
            (UnaryOp::Neg, ConstValue::Float(v)) => Ok(ConstValue::Float(-v)),
            (UnaryOp::Not, ConstValue::Bool(v)) => Ok(ConstValue::Bool(!v)),
            // In Sigil, `!` on integers is bitwise NOT
            (UnaryOp::Not, ConstValue::Int(v)) => Ok(ConstValue::Int(!v)),
            (UnaryOp::Not, ConstValue::UInt(v)) => Ok(ConstValue::UInt(!v)),
            _ => Err(ConstEvalError::unsupported(format!("{:?} operator", op))),
        }
    }

    /// Evaluate array literal
    fn eval_array(&self, elements: &[Expr]) -> Result<ConstValue, ConstEvalError> {
        let values: Result<Vec<_>, _> = elements.iter().map(|e| self.eval(e)).collect();
        Ok(ConstValue::Array(values?))
    }

    /// Evaluate tuple literal
    fn eval_tuple(&self, elements: &[Expr]) -> Result<ConstValue, ConstEvalError> {
        let values: Result<Vec<_>, _> = elements.iter().map(|e| self.eval(e)).collect();
        Ok(ConstValue::Tuple(values?))
    }

    /// Evaluate an array shape literal like `[4, 8]` and generate a Shape type.
    /// Returns a Named type like `Shape2<4, 8>` with const generic parameters.
    pub fn eval_shape(&self, expr: &Expr) -> Result<Type, ConstEvalError> {
        match expr {
            Expr::Array(elements) => {
                let dims: Result<Vec<_>, _> = elements
                    .iter()
                    .map(|e| {
                        let val = self.eval(e)?;
                        val.as_i64().ok_or_else(|| {
                            ConstEvalError::unsupported("non-integer in shape literal")
                        })
                    })
                    .collect();
                let dims = dims?;

                // Generate Shape{N} type with const generics
                let shape_name = format!("Shape{}", dims.len());
                let generics: Vec<Type> = dims.iter().map(|&d| Type::ConstGeneric(d)).collect();

                Ok(Type::Named {
                    name: shape_name,
                    generics,
                })
            }
            _ => {
                // Not an array literal - try to evaluate as a single dimension
                let val = self.eval(expr)?;
                let dim = val.as_i64().ok_or_else(|| {
                    ConstEvalError::unsupported("non-integer shape value")
                })?;
                Ok(Type::Named {
                    name: "Shape1".to_string(),
                    generics: vec![Type::ConstGeneric(dim)],
                })
            }
        }
    }

    /// Try to evaluate an expression as a const generic type parameter.
    /// Returns Type::ConstGeneric if successful, None if expression is not const.
    pub fn try_as_const_generic(&self, expr: &Expr) -> Option<Type> {
        self.eval(expr).ok().and_then(|v| v.to_const_generic())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Span;

    fn make_int_literal(value: i64) -> Expr {
        use crate::ast::NumBase;
        Expr::Literal(Literal::Int {
            value: value.to_string(),
            base: NumBase::Decimal,
            suffix: None,
        })
    }

    fn make_array(elements: Vec<Expr>) -> Expr {
        Expr::Array(elements)
    }

    #[test]
    fn test_eval_int_literal() {
        let eval = ConstEvaluator::new();
        let expr = make_int_literal(42);
        let result = eval.eval(&expr).unwrap();
        assert_eq!(result, ConstValue::Int(42));
    }

    #[test]
    fn test_eval_binary_add() {
        let eval = ConstEvaluator::new();
        let expr = Expr::Binary {
            op: BinOp::Add,
            left: Box::new(make_int_literal(10)),
            right: Box::new(make_int_literal(32)),
        };
        let result = eval.eval(&expr).unwrap();
        assert_eq!(result, ConstValue::Int(42));
    }

    #[test]
    fn test_eval_binary_mul() {
        let eval = ConstEvaluator::new();
        let expr = Expr::Binary {
            op: BinOp::Mul,
            left: Box::new(make_int_literal(6)),
            right: Box::new(make_int_literal(7)),
        };
        let result = eval.eval(&expr).unwrap();
        assert_eq!(result, ConstValue::Int(42));
    }

    #[test]
    fn test_eval_div_by_zero() {
        let eval = ConstEvaluator::new();
        let expr = Expr::Binary {
            op: BinOp::Div,
            left: Box::new(make_int_literal(42)),
            right: Box::new(make_int_literal(0)),
        };
        let result = eval.eval(&expr);
        assert!(matches!(
            result,
            Err(ConstEvalError { kind: ConstEvalErrorKind::DivByZero, .. })
        ));
    }

    #[test]
    fn test_eval_shape_2d() {
        let eval = ConstEvaluator::new();
        let expr = make_array(vec![make_int_literal(4), make_int_literal(8)]);
        let result = eval.eval_shape(&expr).unwrap();

        assert_eq!(
            result,
            Type::Named {
                name: "Shape2".to_string(),
                generics: vec![Type::ConstGeneric(4), Type::ConstGeneric(8)],
            }
        );
    }

    #[test]
    fn test_eval_shape_3d() {
        let eval = ConstEvaluator::new();
        let expr = make_array(vec![
            make_int_literal(2),
            make_int_literal(3),
            make_int_literal(4),
        ]);
        let result = eval.eval_shape(&expr).unwrap();

        assert_eq!(
            result,
            Type::Named {
                name: "Shape3".to_string(),
                generics: vec![
                    Type::ConstGeneric(2),
                    Type::ConstGeneric(3),
                    Type::ConstGeneric(4),
                ],
            }
        );
    }

    #[test]
    fn test_named_constant() {
        let mut eval = ConstEvaluator::new();
        eval.register_const("BATCH_SIZE".to_string(), ConstValue::Int(32));

        use crate::ast::{TypePath, PathSegment, Ident};
        let expr = Expr::Path(TypePath {
            segments: vec![PathSegment {
                ident: Ident {
                    name: "BATCH_SIZE".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: Span::new(0, 0),
                },
                generics: None,
            }],
        });
        let result = eval.eval(&expr).unwrap();
        assert_eq!(result, ConstValue::Int(32));
    }

    #[test]
    fn test_unknown_constant() {
        let eval = ConstEvaluator::new();
        use crate::ast::{TypePath, PathSegment, Ident};
        let expr = Expr::Path(TypePath {
            segments: vec![PathSegment {
                ident: Ident {
                    name: "UNKNOWN".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: Span::new(0, 0),
                },
                generics: None,
            }],
        });
        let result = eval.eval(&expr);
        assert!(matches!(
            result,
            Err(ConstEvalError { kind: ConstEvalErrorKind::UnknownIdent(_), .. })
        ));
    }

    #[test]
    fn test_complex_expression() {
        let eval = ConstEvaluator::new();
        // (2 + 3) * 4 - 6 = 14
        let add = Expr::Binary {
            op: BinOp::Add,
            left: Box::new(make_int_literal(2)),
            right: Box::new(make_int_literal(3)),
        };
        let mul = Expr::Binary {
            op: BinOp::Mul,
            left: Box::new(add),
            right: Box::new(make_int_literal(4)),
        };
        let sub = Expr::Binary {
            op: BinOp::Sub,
            left: Box::new(mul),
            right: Box::new(make_int_literal(6)),
        };
        let result = eval.eval(&sub).unwrap();
        assert_eq!(result, ConstValue::Int(14));
    }
}

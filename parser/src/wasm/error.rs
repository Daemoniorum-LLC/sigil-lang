//! WASM compilation error types.

use crate::span::Span;
use std::fmt;

/// WASM compilation error with source location.
#[derive(Debug, Clone)]
pub struct WasmError {
    pub kind: WasmErrorKind,
    pub span: Option<Span>,
    pub message: String,
}

/// Categories of WASM compilation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmErrorKind {
    /// Parse error from source
    Parse,
    /// Type mismatch or inference failure
    Type,
    /// Unknown variable or function
    Undefined,
    /// Unsupported language feature
    Unsupported,
    /// Internal compiler error
    Internal,
    /// Invalid WASM generation
    Codegen,
}

impl WasmError {
    pub fn new(kind: WasmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            span: None,
            message: message.into(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(WasmErrorKind::Parse, message)
    }

    pub fn type_error(message: impl Into<String>) -> Self {
        Self::new(WasmErrorKind::Type, message)
    }

    pub fn undefined(name: &str) -> Self {
        Self::new(WasmErrorKind::Undefined, format!("undefined: {}", name))
    }

    pub fn unsupported(feature: &str) -> Self {
        Self::new(
            WasmErrorKind::Unsupported,
            format!("unsupported: {}", feature),
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(WasmErrorKind::Internal, message)
    }

    pub fn codegen(message: impl Into<String>) -> Self {
        Self::new(WasmErrorKind::Codegen, message)
    }

    pub fn undefined_variable(name: &str) -> Self {
        Self::new(
            WasmErrorKind::Undefined,
            format!("undefined variable: {}", name),
        )
    }

    pub fn undefined_function(name: &str) -> Self {
        Self::new(
            WasmErrorKind::Undefined,
            format!("undefined function: {}", name),
        )
    }

    pub fn invalid_assignment_target() -> Self {
        Self::new(WasmErrorKind::Type, "invalid assignment target")
    }

    pub fn not_in_loop(stmt: &str) -> Self {
        Self::new(
            WasmErrorKind::Codegen,
            format!("{} statement not inside loop", stmt),
        )
    }

    pub fn undefined_label(name: &str) -> Self {
        Self::new(
            WasmErrorKind::Undefined,
            format!("undefined loop label: '{}", name),
        )
    }

    pub fn arity_mismatch(expected: usize, got: usize) -> Self {
        Self::new(
            WasmErrorKind::Type,
            format!("expected {} arguments, got {}", expected, got),
        )
    }

    pub fn missing_morpheme_body(morpheme: &str) -> Self {
        Self::new(
            WasmErrorKind::Codegen,
            format!("morpheme '{}' requires a body", morpheme),
        )
    }

    pub fn not_const() -> Self {
        Self::new(WasmErrorKind::Codegen, "expression is not constant")
    }

    pub fn div_by_zero() -> Self {
        Self::new(WasmErrorKind::Codegen, "division by zero")
    }
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind_str = match self.kind {
            WasmErrorKind::Parse => "parse error",
            WasmErrorKind::Type => "type error",
            WasmErrorKind::Undefined => "undefined",
            WasmErrorKind::Unsupported => "unsupported",
            WasmErrorKind::Internal => "internal error",
            WasmErrorKind::Codegen => "codegen error",
        };

        if let Some(span) = &self.span {
            write!(f, "[{}] {}: {}", span, kind_str, self.message)
        } else {
            write!(f, "{}: {}", kind_str, self.message)
        }
    }
}

impl std::error::Error for WasmError {}

/// Result type for WASM compilation.
pub type WasmResult<T> = Result<T, WasmError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = WasmError::undefined("foo");
        assert_eq!(err.kind, WasmErrorKind::Undefined);
        assert!(err.message.contains("foo"));
    }

    #[test]
    fn test_error_display() {
        let err = WasmError::unsupported("async/await");
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported"));
        assert!(msg.contains("async/await"));
    }

    #[test]
    fn test_error_with_span() {
        let span = Span::new(10, 20);
        let err = WasmError::type_error("expected i64").with_span(span);
        assert!(err.span.is_some());
    }

    #[test]
    fn test_parse_error() {
        let err = WasmError::parse("unexpected token");
        assert_eq!(err.kind, WasmErrorKind::Parse);
        assert!(err.message.contains("unexpected token"));
    }

    #[test]
    fn test_type_error() {
        let err = WasmError::type_error("expected i64, got f64");
        assert_eq!(err.kind, WasmErrorKind::Type);
        assert!(err.message.contains("expected i64"));
    }

    #[test]
    fn test_internal_error() {
        let err = WasmError::internal("function not registered");
        assert_eq!(err.kind, WasmErrorKind::Internal);
        assert!(err.message.contains("function not registered"));
    }

    #[test]
    fn test_codegen_error() {
        let err = WasmError::codegen("invalid instruction");
        assert_eq!(err.kind, WasmErrorKind::Codegen);
        assert!(err.message.contains("invalid instruction"));
    }

    #[test]
    fn test_undefined_variable() {
        let err = WasmError::undefined_variable("x");
        assert_eq!(err.kind, WasmErrorKind::Undefined);
        assert!(err.message.contains("undefined variable"));
        assert!(err.message.contains("x"));
    }

    #[test]
    fn test_undefined_function() {
        let err = WasmError::undefined_function("foo");
        assert_eq!(err.kind, WasmErrorKind::Undefined);
        assert!(err.message.contains("undefined function"));
        assert!(err.message.contains("foo"));
    }

    #[test]
    fn test_invalid_assignment_target() {
        let err = WasmError::invalid_assignment_target();
        assert_eq!(err.kind, WasmErrorKind::Type);
        assert!(err.message.contains("assignment target"));
    }

    #[test]
    fn test_not_in_loop() {
        let err = WasmError::not_in_loop("break");
        assert_eq!(err.kind, WasmErrorKind::Codegen);
        assert!(err.message.contains("break"));
        assert!(err.message.contains("not inside loop"));
    }

    #[test]
    fn test_arity_mismatch() {
        let err = WasmError::arity_mismatch(3, 2);
        assert_eq!(err.kind, WasmErrorKind::Type);
        assert!(err.message.contains("3"));
        assert!(err.message.contains("2"));
    }

    #[test]
    fn test_missing_morpheme_body() {
        let err = WasmError::missing_morpheme_body("τ");
        assert_eq!(err.kind, WasmErrorKind::Codegen);
        assert!(err.message.contains("τ"));
        assert!(err.message.contains("requires a body"));
    }

    #[test]
    fn test_not_const() {
        let err = WasmError::not_const();
        assert_eq!(err.kind, WasmErrorKind::Codegen);
        assert!(err.message.contains("not constant"));
    }

    #[test]
    fn test_div_by_zero() {
        let err = WasmError::div_by_zero();
        assert_eq!(err.kind, WasmErrorKind::Codegen);
        assert!(err.message.contains("division by zero"));
    }

    #[test]
    fn test_error_display_with_span() {
        let span = Span::new(10, 20);
        let err = WasmError::type_error("mismatched types").with_span(span);
        let msg = format!("{}", err);
        assert!(msg.contains("type error"));
        assert!(msg.contains("mismatched types"));
        assert!(msg.contains("10")); // span start
    }

    #[test]
    fn test_error_display_all_kinds() {
        let kinds = [
            (WasmError::parse("x"), "parse error"),
            (WasmError::type_error("x"), "type error"),
            (WasmError::undefined("x"), "undefined"),
            (WasmError::unsupported("x"), "unsupported"),
            (WasmError::internal("x"), "internal error"),
            (WasmError::codegen("x"), "codegen error"),
        ];

        for (err, expected_kind) in kinds {
            let msg = format!("{}", err);
            assert!(
                msg.contains(expected_kind),
                "Expected '{}' in '{}'",
                expected_kind,
                msg
            );
        }
    }

    #[test]
    fn test_error_kind_equality() {
        assert_eq!(WasmErrorKind::Parse, WasmErrorKind::Parse);
        assert_ne!(WasmErrorKind::Parse, WasmErrorKind::Type);
        assert_ne!(WasmErrorKind::Internal, WasmErrorKind::Codegen);
    }
}

//! Sigil Parser Library
//!
//! A polysynthetic programming language with evidentiality types.
//!
//! This crate provides:
//! - Lexer and parser for Sigil source code
//! - Tree-walking interpreter for development/debugging
//! - JIT compiler using Cranelift for native performance
//! - Comprehensive optimization passes (O0-O3)
//! - Rich diagnostic reporting with colored output

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod span;
pub mod interpreter;
pub mod typeck;
pub mod stdlib;
pub mod optimize;
pub mod diagnostic;
pub mod ffi;
pub mod ir;

#[cfg(feature = "jit")]
pub mod codegen;

#[cfg(feature = "llvm")]
pub mod llvm_codegen;

#[cfg(feature = "protocol-core")]
pub mod protocol;

pub use lexer::{Token, Lexer};
pub use ast::*;
pub use parser::Parser;
pub use span::Span;
pub use interpreter::{Interpreter, Value, Evidence, RuntimeError, Function};
pub use typeck::{TypeChecker, Type, TypeError, EvidenceLevel};
pub use stdlib::register_stdlib;
pub use optimize::{Optimizer, OptLevel, OptStats, optimize};
pub use diagnostic::{Diagnostic, DiagnosticBuilder, Diagnostics, Severity, FixSuggestion};
pub use ir::{IrEmitter, IrEmitterOptions, IrDocument, emit_ir, emit_ir_json};

#[cfg(feature = "jit")]
pub use codegen::JitCompiler;

#[cfg(feature = "llvm")]
pub use llvm_codegen::llvm::{LlvmCompiler, CompileMode};

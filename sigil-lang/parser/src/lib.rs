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
//! - AI-facing IR for tooling and agent integration

pub mod ast;
pub mod diagnostic;
pub mod ffi;
pub mod interpreter;
pub mod ir;
pub mod lexer;
pub mod lower;
pub mod optimize;
pub mod parser;
pub mod span;
pub mod stdlib;
pub mod typeck;

#[cfg(feature = "jit")]
pub mod codegen;

#[cfg(feature = "llvm")]
pub mod llvm_codegen;

#[cfg(feature = "protocol-core")]
pub mod protocol;

pub use ast::*;
pub use diagnostic::{Diagnostic, DiagnosticBuilder, Diagnostics, FixSuggestion, Severity};
pub use interpreter::{Evidence, Function, Interpreter, RuntimeError, Value};
pub use ir::{IrDumpOptions, IrEvidence, IrFunction, IrModule, IrOperation, IrType};
pub use lexer::{Lexer, Token};
pub use lower::lower_source_file;
pub use optimize::{optimize, OptLevel, OptStats, Optimizer};
pub use parser::Parser;
pub use span::Span;
pub use stdlib::register_stdlib;
pub use typeck::{EvidenceLevel, Type, TypeChecker, TypeError};

#[cfg(feature = "jit")]
pub use codegen::JitCompiler;

#[cfg(feature = "llvm")]
pub use llvm_codegen::llvm::{CompileMode, LlvmCompiler};

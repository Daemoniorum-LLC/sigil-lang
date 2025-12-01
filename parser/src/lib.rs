//! Sigil Parser Library
//!
//! A polysynthetic programming language with evidentiality types.
//!
//! This crate provides:
//! - Lexer and parser for Sigil source code
//! - Tree-walking interpreter for development/debugging
//! - JIT compiler using Cranelift for native performance

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod span;
pub mod interpreter;
pub mod typeck;
pub mod stdlib;

#[cfg(feature = "jit")]
pub mod codegen;

pub use lexer::{Token, Lexer};
pub use ast::*;
pub use parser::Parser;
pub use span::Span;
pub use interpreter::{Interpreter, Value, Evidence, RuntimeError, Function};
pub use typeck::{TypeChecker, Type, TypeError, EvidenceLevel};
pub use stdlib::register_stdlib;

#[cfg(feature = "jit")]
pub use codegen::JitCompiler;

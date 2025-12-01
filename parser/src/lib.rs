//! Sigil Parser Library
//!
//! A polysynthetic programming language with evidentiality types.

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod span;
pub mod interpreter;
pub mod typeck;
pub mod stdlib;

pub use lexer::{Token, Lexer};
pub use ast::*;
pub use parser::Parser;
pub use span::Span;
pub use interpreter::{Interpreter, Value, Evidence, RuntimeError, Function};
pub use typeck::{TypeChecker, Type, TypeError, EvidenceLevel};
pub use stdlib::register_stdlib;

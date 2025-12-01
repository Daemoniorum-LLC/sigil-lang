//! Sigil Parser Library
//!
//! A polysynthetic programming language with evidentiality types.

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod span;
pub mod interpreter;

pub use lexer::{Token, Lexer};
pub use ast::*;
pub use parser::Parser;
pub use span::Span;
pub use interpreter::{Interpreter, Value, Evidence, RuntimeError, Function};

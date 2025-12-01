//! Sigil Parser Library
//!
//! A polysynthetic programming language with evidentiality types.

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod span;

pub use lexer::{Token, Lexer};
pub use ast::*;
pub use parser::Parser;
pub use span::Span;

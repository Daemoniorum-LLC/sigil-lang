//! Sigil WASM Playground - Minimal Interpreter
//!
//! A self-contained Sigil interpreter compiled to WebAssembly.
//! Supports core language features for playground demonstration.

use wasm_bindgen::prelude::*;
use std::collections::HashMap;

mod lexer;
mod parser;
mod interpreter;

use crate::interpreter::Interpreter;
use crate::parser::Parser;

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Execute Sigil code and return JSON result
#[wasm_bindgen]
pub fn execute(source: &str) -> String {
    match run_sigil(source) {
        Ok(result) => format!(
            r#"{{"ok":true,"output":"{}","value":"{}"}}"#,
            escape_json(&result.output),
            escape_json(&result.value)
        ),
        Err(e) => format!(
            r#"{{"ok":false,"error":"{}","phase":"{}"}}"#,
            escape_json(&e.message),
            e.phase
        ),
    }
}

/// Check Sigil code syntax without executing
#[wasm_bindgen]
pub fn check(source: &str) -> String {
    match check_sigil(source) {
        Ok(msg) => format!(r#"{{"ok":true,"output":"{}"}}"#, escape_json(&msg)),
        Err(e) => format!(
            r#"{{"ok":false,"error":"{}","phase":"{}"}}"#,
            escape_json(&e.message),
            e.phase
        ),
    }
}

struct RunResult {
    output: String,
    value: String,
}

struct RunError {
    message: String,
    phase: String,
}

fn run_sigil(source: &str) -> Result<RunResult, RunError> {
    // Parse
    let mut parser = Parser::new(source);
    let ast = parser.parse().map_err(|e| RunError {
        message: e,
        phase: "parse".to_string(),
    })?;

    // Execute
    let mut interp = Interpreter::new();
    let (value, output) = interp.execute(&ast).map_err(|e| RunError {
        message: e,
        phase: "runtime".to_string(),
    })?;

    Ok(RunResult { output, value })
}

fn check_sigil(source: &str) -> Result<String, RunError> {
    let mut parser = Parser::new(source);
    let ast = parser.parse().map_err(|e| RunError {
        message: e,
        phase: "parse".to_string(),
    })?;

    Ok(format!("Parsed {} item(s). Syntax OK.", ast.len()))
}

fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c < ' ' => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result
}

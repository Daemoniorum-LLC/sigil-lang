//! Playground API for browser-based Sigil execution.
//!
//! Provides wasm-bindgen exported functions for parsing, type-checking,
//! and interpreting Sigil source code in a sandboxed browser environment.
//!
//! Build: wasm-pack build --target web --no-default-features --features playground

use wasm_bindgen::prelude::*;

use crate::interpreter::Interpreter;
use crate::parser::Parser;
use crate::stdlib::register_stdlib;
use crate::typeck::TypeChecker;

/// Run Sigil source code and return JSON results.
///
/// Returns JSON: `{ "output": [...], "errors": [...], "elapsed_ms": N }`
#[wasm_bindgen]
pub fn playground_run(source: &str) -> String {
    let start = js_sys::Date::now();

    let mut errors: Vec<String> = Vec::new();

    // Parse
    let mut parser = Parser::new(source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            errors.push(format!("Parse error: {}", e));
            let elapsed = js_sys::Date::now() - start;
            return serde_json::json!({
                "output": [],
                "errors": errors,
                "elapsed_ms": elapsed as u64,
            }).to_string();
        }
    };

    // Type check before execution - evidentiality is enforced at compile time
    let mut type_checker = TypeChecker::new();
    if let Err(type_errors) = type_checker.check_file(&ast) {
        for err in type_errors {
            let mut msg = format!("Type error: {}", err.message);
            for note in &err.notes {
                msg.push_str(&format!("\n  note: {}", note));
            }
            errors.push(msg);
        }
        let elapsed = js_sys::Date::now() - start;
        return serde_json::json!({
            "output": [],
            "errors": errors,
            "elapsed_ms": elapsed as u64,
        }).to_string();
    }

    // Execute with stdlib
    let mut interpreter = Interpreter::new();
    register_stdlib(&mut interpreter);
    interpreter.set_source_code(source.to_string());

    match interpreter.execute(&ast) {
        Ok(value) => {
            let mut output = interpreter.output.clone();
            // If the program returned a non-null value, include it
            if !matches!(value, crate::interpreter::Value::Null) {
                output.push(format!("{}", value));
            }
            let elapsed = js_sys::Date::now() - start;
            serde_json::json!({
                "output": output,
                "errors": errors,
                "elapsed_ms": elapsed as u64,
            }).to_string()
        }
        Err(e) => {
            errors.push(format!("Runtime error: {}", e));
            let output = interpreter.output.clone();
            let elapsed = js_sys::Date::now() - start;
            serde_json::json!({
                "output": output,
                "errors": errors,
                "elapsed_ms": elapsed as u64,
            }).to_string()
        }
    }
}

/// Check Sigil source code for parse and type errors without executing.
///
/// Returns JSON: `{ "diagnostics": [...], "elapsed_ms": N }`
#[wasm_bindgen]
pub fn playground_check(source: &str) -> String {
    let start = js_sys::Date::now();

    let mut diagnostics: Vec<serde_json::Value> = Vec::new();

    // Parse
    let mut parser = Parser::new(source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            diagnostics.push(serde_json::json!({
                "severity": "error",
                "message": format!("Parse error: {}", e),
            }));
            let elapsed = js_sys::Date::now() - start;
            return serde_json::json!({
                "diagnostics": diagnostics,
                "elapsed_ms": elapsed as u64,
            }).to_string();
        }
    };

    // Type check
    let mut type_checker = TypeChecker::new();
    if let Err(type_errors) = type_checker.check_file(&ast) {
        for err in type_errors {
            diagnostics.push(serde_json::json!({
                "severity": "error",
                "message": format!("Type error: {}", err.message),
                "notes": err.notes,
            }));
        }
    }

    let elapsed = js_sys::Date::now() - start;
    serde_json::json!({
        "diagnostics": diagnostics,
        "elapsed_ms": elapsed as u64,
    }).to_string()
}

/// Get Sigil language version info.
#[wasm_bindgen]
pub fn playground_version() -> String {
    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "name": "Sigil Playground",
    }).to_string()
}

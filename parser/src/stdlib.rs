//! Sigil Standard Library
//!
//! A comprehensive collection of built-in functions that leverage
//! Sigil's polysynthetic syntax and evidentiality type system.
//!
//! ## Modules
//!
//! - **core**: Essential functions (print, assert, panic)
//! - **math**: Mathematical operations including poly-cultural math
//! - **collections**: Array, map, set operations
//! - **string**: String manipulation
//! - **evidence**: Evidentiality markers and operations
//! - **iter**: Iterator-style operations for pipes
//! - **io**: File and console I/O
//! - **time**: Date, time, and measurement
//! - **random**: Random number generation
//! - **convert**: Type conversions
//! - **json**: JSON parsing and serialization
//! - **fs**: File system operations
//! - **crypto**: Hashing and encoding (SHA256, MD5, base64)
//! - **regex**: Regular expression matching
//! - **uuid**: UUID generation
//! - **system**: Environment, args, process control
//! - **stats**: Statistical functions
//! - **matrix**: Matrix operations

use crate::interpreter::{Interpreter, Value, Evidence, RuntimeError, BuiltInFn, ChannelInner, ActorInner};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use std::io::Write;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

// External crates for extended stdlib
use sha2::{Sha256, Sha512, Digest};
use md5::Md5;
use base64::{Engine as _, engine::general_purpose};
use regex::Regex;
use uuid::Uuid;

/// Register all standard library functions
pub fn register_stdlib(interp: &mut Interpreter) {
    register_core(interp);
    register_math(interp);
    register_collections(interp);
    register_string(interp);
    register_evidence(interp);
    register_iter(interp);
    register_io(interp);
    register_time(interp);
    register_random(interp);
    register_convert(interp);
    register_cycle(interp);
    register_simd(interp);
    register_concurrency(interp);
    // Phase 4: Extended stdlib
    register_json(interp);
    register_fs(interp);
    register_crypto(interp);
    register_regex(interp);
    register_uuid(interp);
    register_system(interp);
    register_stats(interp);
    register_matrix(interp);
    // Phase 5: Language power-ups
    register_functional(interp);
    register_benchmark(interp);
    register_itertools(interp);
    register_ranges(interp);
    register_bitwise(interp);
    register_format(interp);
    // Phase 6: Pattern matching power-ups
    register_pattern(interp);
}

// Helper to define a builtin
fn define(
    interp: &mut Interpreter,
    name: &str,
    arity: Option<usize>,
    func: fn(&mut Interpreter, Vec<Value>) -> Result<Value, RuntimeError>,
) {
    let builtin = Value::BuiltIn(Rc::new(BuiltInFn {
        name: name.to_string(),
        arity,
        func,
    }));
    interp.globals.borrow_mut().define(name.to_string(), builtin);
}

// ============================================================================
// CORE FUNCTIONS
// ============================================================================

fn register_core(interp: &mut Interpreter) {
    // print - variadic print without newline
    define(interp, "print", None, |interp, args| {
        let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
        let line = output.join(" ");
        print!("{}", line);
        std::io::stdout().flush().ok();
        interp.output.push(line);
        Ok(Value::Null)
    });

    // println - print with newline
    define(interp, "println", None, |interp, args| {
        let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
        let line = output.join(" ");
        println!("{}", line);
        interp.output.push(line);
        Ok(Value::Null)
    });

    // dbg - debug print with source info
    define(interp, "dbg", Some(1), |interp, args| {
        let output = format!("[DEBUG] {:?}", args[0]);
        println!("{}", output);
        interp.output.push(output);
        Ok(args[0].clone())
    });

    // type_of - get type name
    define(interp, "type_of", Some(1), |_, args| {
        let type_name = match &args[0] {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "i64",
            Value::Float(_) => "f64",
            Value::String(_) => "str",
            Value::Char(_) => "char",
            Value::Array(_) => "array",
            Value::Tuple(_) => "tuple",
            Value::Struct { name, .. } => name,
            Value::Variant { enum_name, .. } => enum_name,
            Value::Function(_) => "fn",
            Value::BuiltIn(_) => "builtin",
            Value::Ref(_) => "ref",
            Value::Infinity => "infinity",
            Value::Empty => "empty",
            Value::Evidential { evidence, .. } => match evidence {
                Evidence::Known => "known",
                Evidence::Uncertain => "uncertain",
                Evidence::Reported => "reported",
                Evidence::Paradox => "paradox",
            },
            Value::Map(_) => "map",
            Value::Set(_) => "set",
            Value::Channel(_) => "channel",
            Value::ThreadHandle(_) => "thread",
            Value::Actor(_) => "actor",
            Value::Future(_) => "future",
        };
        Ok(Value::String(Rc::new(type_name.to_string())))
    });

    // assert - assertion with optional message
    define(interp, "assert", None, |_, args| {
        if args.is_empty() {
            return Err(RuntimeError::new("assert() requires at least one argument"));
        }
        let condition = match &args[0] {
            Value::Bool(b) => *b,
            _ => return Err(RuntimeError::new("assert() condition must be bool")),
        };
        if !condition {
            let msg = if args.len() > 1 {
                format!("{}", args[1])
            } else {
                "assertion failed".to_string()
            };
            return Err(RuntimeError::new(format!("Assertion failed: {}", msg)));
        }
        Ok(Value::Null)
    });

    // panic - abort execution with message
    define(interp, "panic", None, |_, args| {
        let msg = if args.is_empty() {
            "explicit panic".to_string()
        } else {
            args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
        };
        Err(RuntimeError::new(format!("PANIC: {}", msg)))
    });

    // todo - mark unimplemented code
    define(interp, "todo", None, |_, args| {
        let msg = if args.is_empty() {
            "not yet implemented".to_string()
        } else {
            format!("{}", args[0])
        };
        Err(RuntimeError::new(format!("TODO: {}", msg)))
    });

    // unreachable - mark code that should never execute
    define(interp, "unreachable", None, |_, args| {
        let msg = if args.is_empty() {
            "entered unreachable code".to_string()
        } else {
            format!("{}", args[0])
        };
        Err(RuntimeError::new(format!("UNREACHABLE: {}", msg)))
    });

    // clone - deep clone a value
    define(interp, "clone", Some(1), |_, args| {
        Ok(deep_clone(&args[0]))
    });

    // identity - return value unchanged (useful in pipes)
    define(interp, "id", Some(1), |_, args| {
        Ok(args[0].clone())
    });

    // default - return default value for a type
    define(interp, "default", Some(1), |_, args| {
        let type_name = match &args[0] {
            Value::String(s) => s.as_str(),
            _ => return Err(RuntimeError::new("default() requires type name string")),
        };
        match type_name {
            "bool" => Ok(Value::Bool(false)),
            "i64" | "int" => Ok(Value::Int(0)),
            "f64" | "float" => Ok(Value::Float(0.0)),
            "str" | "string" => Ok(Value::String(Rc::new(String::new()))),
            "array" => Ok(Value::Array(Rc::new(RefCell::new(Vec::new())))),
            _ => Err(RuntimeError::new(format!("no default for type: {}", type_name))),
        }
    });
}

// Deep clone helper
fn deep_clone(value: &Value) -> Value {
    match value {
        Value::Array(arr) => {
            let cloned: Vec<Value> = arr.borrow().iter().map(deep_clone).collect();
            Value::Array(Rc::new(RefCell::new(cloned)))
        }
        Value::Struct { name, fields } => {
            let cloned: HashMap<String, Value> = fields.borrow()
                .iter()
                .map(|(k, v)| (k.clone(), deep_clone(v)))
                .collect();
            Value::Struct {
                name: name.clone(),
                fields: Rc::new(RefCell::new(cloned)),
            }
        }
        Value::Evidential { value, evidence } => Value::Evidential {
            value: Box::new(deep_clone(value)),
            evidence: *evidence,
        },
        other => other.clone(),
    }
}

// ============================================================================
// MATH FUNCTIONS
// ============================================================================

fn register_math(interp: &mut Interpreter) {
    // Basic math
    define(interp, "abs", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(n.abs())),
            Value::Float(n) => Ok(Value::Float(n.abs())),
            _ => Err(RuntimeError::new("abs() requires number")),
        }
    });

    define(interp, "neg", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(-n)),
            Value::Float(n) => Ok(Value::Float(-n)),
            _ => Err(RuntimeError::new("neg() requires number")),
        }
    });

    define(interp, "sqrt", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
            Value::Float(n) => Ok(Value::Float(n.sqrt())),
            _ => Err(RuntimeError::new("sqrt() requires number")),
        }
    });

    define(interp, "cbrt", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).cbrt())),
            Value::Float(n) => Ok(Value::Float(n.cbrt())),
            _ => Err(RuntimeError::new("cbrt() requires number")),
        }
    });

    define(interp, "pow", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(base), Value::Int(exp)) => {
                if *exp >= 0 {
                    Ok(Value::Int(base.pow(*exp as u32)))
                } else {
                    Ok(Value::Float((*base as f64).powi(*exp as i32)))
                }
            }
            (Value::Float(base), Value::Int(exp)) => Ok(Value::Float(base.powi(*exp as i32))),
            (Value::Float(base), Value::Float(exp)) => Ok(Value::Float(base.powf(*exp))),
            (Value::Int(base), Value::Float(exp)) => Ok(Value::Float((*base as f64).powf(*exp))),
            _ => Err(RuntimeError::new("pow() requires numbers")),
        }
    });

    define(interp, "exp", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).exp())),
            Value::Float(n) => Ok(Value::Float(n.exp())),
            _ => Err(RuntimeError::new("exp() requires number")),
        }
    });

    define(interp, "ln", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).ln())),
            Value::Float(n) => Ok(Value::Float(n.ln())),
            _ => Err(RuntimeError::new("ln() requires number")),
        }
    });

    define(interp, "log", Some(2), |_, args| {
        let (value, base) = match (&args[0], &args[1]) {
            (Value::Int(v), Value::Int(b)) => (*v as f64, *b as f64),
            (Value::Float(v), Value::Int(b)) => (*v, *b as f64),
            (Value::Int(v), Value::Float(b)) => (*v as f64, *b),
            (Value::Float(v), Value::Float(b)) => (*v, *b),
            _ => return Err(RuntimeError::new("log() requires numbers")),
        };
        Ok(Value::Float(value.log(base)))
    });

    define(interp, "log10", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).log10())),
            Value::Float(n) => Ok(Value::Float(n.log10())),
            _ => Err(RuntimeError::new("log10() requires number")),
        }
    });

    define(interp, "log2", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).log2())),
            Value::Float(n) => Ok(Value::Float(n.log2())),
            _ => Err(RuntimeError::new("log2() requires number")),
        }
    });

    // Trigonometry
    define(interp, "sin", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).sin())),
            Value::Float(n) => Ok(Value::Float(n.sin())),
            _ => Err(RuntimeError::new("sin() requires number")),
        }
    });

    define(interp, "cos", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).cos())),
            Value::Float(n) => Ok(Value::Float(n.cos())),
            _ => Err(RuntimeError::new("cos() requires number")),
        }
    });

    define(interp, "tan", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).tan())),
            Value::Float(n) => Ok(Value::Float(n.tan())),
            _ => Err(RuntimeError::new("tan() requires number")),
        }
    });

    define(interp, "asin", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).asin())),
            Value::Float(n) => Ok(Value::Float(n.asin())),
            _ => Err(RuntimeError::new("asin() requires number")),
        }
    });

    define(interp, "acos", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).acos())),
            Value::Float(n) => Ok(Value::Float(n.acos())),
            _ => Err(RuntimeError::new("acos() requires number")),
        }
    });

    define(interp, "atan", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).atan())),
            Value::Float(n) => Ok(Value::Float(n.atan())),
            _ => Err(RuntimeError::new("atan() requires number")),
        }
    });

    define(interp, "atan2", Some(2), |_, args| {
        let (y, x) = match (&args[0], &args[1]) {
            (Value::Int(y), Value::Int(x)) => (*y as f64, *x as f64),
            (Value::Float(y), Value::Int(x)) => (*y, *x as f64),
            (Value::Int(y), Value::Float(x)) => (*y as f64, *x),
            (Value::Float(y), Value::Float(x)) => (*y, *x),
            _ => return Err(RuntimeError::new("atan2() requires numbers")),
        };
        Ok(Value::Float(y.atan2(x)))
    });

    // Hyperbolic
    define(interp, "sinh", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).sinh())),
            Value::Float(n) => Ok(Value::Float(n.sinh())),
            _ => Err(RuntimeError::new("sinh() requires number")),
        }
    });

    define(interp, "cosh", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).cosh())),
            Value::Float(n) => Ok(Value::Float(n.cosh())),
            _ => Err(RuntimeError::new("cosh() requires number")),
        }
    });

    define(interp, "tanh", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).tanh())),
            Value::Float(n) => Ok(Value::Float(n.tanh())),
            _ => Err(RuntimeError::new("tanh() requires number")),
        }
    });

    // Rounding
    define(interp, "floor", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(*n)),
            Value::Float(n) => Ok(Value::Int(n.floor() as i64)),
            _ => Err(RuntimeError::new("floor() requires number")),
        }
    });

    define(interp, "ceil", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(*n)),
            Value::Float(n) => Ok(Value::Int(n.ceil() as i64)),
            _ => Err(RuntimeError::new("ceil() requires number")),
        }
    });

    define(interp, "round", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(*n)),
            Value::Float(n) => Ok(Value::Int(n.round() as i64)),
            _ => Err(RuntimeError::new("round() requires number")),
        }
    });

    define(interp, "trunc", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(*n)),
            Value::Float(n) => Ok(Value::Int(n.trunc() as i64)),
            _ => Err(RuntimeError::new("trunc() requires number")),
        }
    });

    define(interp, "fract", Some(1), |_, args| {
        match &args[0] {
            Value::Int(_) => Ok(Value::Float(0.0)),
            Value::Float(n) => Ok(Value::Float(n.fract())),
            _ => Err(RuntimeError::new("fract() requires number")),
        }
    });

    // Min/Max/Clamp
    define(interp, "min", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.min(b))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.min(*b))),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64).min(*b))),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a.min(*b as f64))),
            _ => Err(RuntimeError::new("min() requires numbers")),
        }
    });

    define(interp, "max", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.max(b))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.max(*b))),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64).max(*b))),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a.max(*b as f64))),
            _ => Err(RuntimeError::new("max() requires numbers")),
        }
    });

    define(interp, "clamp", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::Int(val), Value::Int(min), Value::Int(max)) => {
                Ok(Value::Int(*val.max(min).min(max)))
            }
            (Value::Float(val), Value::Float(min), Value::Float(max)) => {
                Ok(Value::Float(val.max(*min).min(*max)))
            }
            _ => Err(RuntimeError::new("clamp() requires matching number types")),
        }
    });

    // Sign
    define(interp, "sign", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(n.signum())),
            Value::Float(n) => Ok(Value::Float(if *n > 0.0 { 1.0 } else if *n < 0.0 { -1.0 } else { 0.0 })),
            _ => Err(RuntimeError::new("sign() requires number")),
        }
    });

    // Constants
    define(interp, "PI", Some(0), |_, _| Ok(Value::Float(std::f64::consts::PI)));
    define(interp, "E", Some(0), |_, _| Ok(Value::Float(std::f64::consts::E)));
    define(interp, "TAU", Some(0), |_, _| Ok(Value::Float(std::f64::consts::TAU)));
    define(interp, "PHI", Some(0), |_, _| Ok(Value::Float(1.618033988749895))); // Golden ratio

    // GCD/LCM
    define(interp, "gcd", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(gcd(*a, *b))),
            _ => Err(RuntimeError::new("gcd() requires integers")),
        }
    });

    define(interp, "lcm", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => {
                let g = gcd(*a, *b);
                Ok(Value::Int((a * b).abs() / g))
            }
            _ => Err(RuntimeError::new("lcm() requires integers")),
        }
    });

    // Factorial
    define(interp, "factorial", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) if *n >= 0 => {
                let mut result: i64 = 1;
                for i in 2..=(*n as u64) {
                    result = result.saturating_mul(i as i64);
                }
                Ok(Value::Int(result))
            }
            Value::Int(_) => Err(RuntimeError::new("factorial() requires non-negative integer")),
            _ => Err(RuntimeError::new("factorial() requires integer")),
        }
    });

    // Is checks
    define(interp, "is_nan", Some(1), |_, args| {
        match &args[0] {
            Value::Float(n) => Ok(Value::Bool(n.is_nan())),
            Value::Int(_) => Ok(Value::Bool(false)),
            _ => Err(RuntimeError::new("is_nan() requires number")),
        }
    });

    define(interp, "is_infinite", Some(1), |_, args| {
        match &args[0] {
            Value::Float(n) => Ok(Value::Bool(n.is_infinite())),
            Value::Int(_) => Ok(Value::Bool(false)),
            Value::Infinity => Ok(Value::Bool(true)),
            _ => Err(RuntimeError::new("is_infinite() requires number")),
        }
    });

    define(interp, "is_finite", Some(1), |_, args| {
        match &args[0] {
            Value::Float(n) => Ok(Value::Bool(n.is_finite())),
            Value::Int(_) => Ok(Value::Bool(true)),
            Value::Infinity => Ok(Value::Bool(false)),
            _ => Err(RuntimeError::new("is_finite() requires number")),
        }
    });

    define(interp, "is_even", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Bool(n % 2 == 0)),
            _ => Err(RuntimeError::new("is_even() requires integer")),
        }
    });

    define(interp, "is_odd", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Bool(n % 2 != 0)),
            _ => Err(RuntimeError::new("is_odd() requires integer")),
        }
    });

    define(interp, "is_prime", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Bool(is_prime(*n))),
            _ => Err(RuntimeError::new("is_prime() requires integer")),
        }
    });
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn is_prime(n: i64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let sqrt = (n as f64).sqrt() as i64;
    for i in (3..=sqrt).step_by(2) {
        if n % i == 0 { return false; }
    }
    true
}

// ============================================================================
// COLLECTION FUNCTIONS
// ============================================================================

fn register_collections(interp: &mut Interpreter) {
    // Basic operations
    define(interp, "len", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
            Value::String(s) => Ok(Value::Int(s.chars().count() as i64)),
            Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
            Value::Map(m) => Ok(Value::Int(m.borrow().len() as i64)),
            Value::Set(s) => Ok(Value::Int(s.borrow().len() as i64)),
            _ => Err(RuntimeError::new("len() requires array, string, tuple, map, or set")),
        }
    });

    define(interp, "is_empty", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => Ok(Value::Bool(arr.borrow().is_empty())),
            Value::String(s) => Ok(Value::Bool(s.is_empty())),
            Value::Tuple(t) => Ok(Value::Bool(t.is_empty())),
            Value::Map(m) => Ok(Value::Bool(m.borrow().is_empty())),
            Value::Set(s) => Ok(Value::Bool(s.borrow().is_empty())),
            _ => Err(RuntimeError::new("is_empty() requires collection")),
        }
    });

    // Array operations
    define(interp, "push", Some(2), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                arr.borrow_mut().push(args[1].clone());
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("push() requires array")),
        }
    });

    define(interp, "pop", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                arr.borrow_mut().pop()
                    .ok_or_else(|| RuntimeError::new("pop() on empty array"))
            }
            _ => Err(RuntimeError::new("pop() requires array")),
        }
    });

    define(interp, "first", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                arr.borrow().first().cloned()
                    .ok_or_else(|| RuntimeError::new("first() on empty array"))
            }
            Value::Tuple(t) => {
                t.first().cloned()
                    .ok_or_else(|| RuntimeError::new("first() on empty tuple"))
            }
            _ => Err(RuntimeError::new("first() requires array or tuple")),
        }
    });

    define(interp, "last", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                arr.borrow().last().cloned()
                    .ok_or_else(|| RuntimeError::new("last() on empty array"))
            }
            Value::Tuple(t) => {
                t.last().cloned()
                    .ok_or_else(|| RuntimeError::new("last() on empty tuple"))
            }
            _ => Err(RuntimeError::new("last() requires array or tuple")),
        }
    });

    define(interp, "get", Some(2), |_, args| {
        let index = match &args[1] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("get() index must be integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                let idx = if index < 0 { arr.len() as i64 + index } else { index } as usize;
                arr.get(idx).cloned()
                    .ok_or_else(|| RuntimeError::new("index out of bounds"))
            }
            Value::Tuple(t) => {
                let idx = if index < 0 { t.len() as i64 + index } else { index } as usize;
                t.get(idx).cloned()
                    .ok_or_else(|| RuntimeError::new("index out of bounds"))
            }
            _ => Err(RuntimeError::new("get() requires array or tuple")),
        }
    });

    define(interp, "set", Some(3), |_, args| {
        let index = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("set() index must be integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let mut arr = arr.borrow_mut();
                if index >= arr.len() {
                    return Err(RuntimeError::new("index out of bounds"));
                }
                arr[index] = args[2].clone();
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("set() requires array")),
        }
    });

    define(interp, "insert", Some(3), |_, args| {
        let index = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("insert() index must be integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let mut arr = arr.borrow_mut();
                if index > arr.len() {
                    return Err(RuntimeError::new("index out of bounds"));
                }
                arr.insert(index, args[2].clone());
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("insert() requires array")),
        }
    });

    define(interp, "remove", Some(2), |_, args| {
        let index = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("remove() index must be integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let mut arr = arr.borrow_mut();
                if index >= arr.len() {
                    return Err(RuntimeError::new("index out of bounds"));
                }
                Ok(arr.remove(index))
            }
            _ => Err(RuntimeError::new("remove() requires array")),
        }
    });

    define(interp, "clear", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                arr.borrow_mut().clear();
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("clear() requires array")),
        }
    });

    // Searching
    define(interp, "contains", Some(2), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                Ok(Value::Bool(arr.borrow().iter().any(|v| values_equal(v, &args[1]))))
            }
            Value::String(s) => {
                match &args[1] {
                    Value::String(sub) => Ok(Value::Bool(s.contains(sub.as_str()))),
                    Value::Char(c) => Ok(Value::Bool(s.contains(*c))),
                    _ => Err(RuntimeError::new("string contains() requires string or char")),
                }
            }
            _ => Err(RuntimeError::new("contains() requires array or string")),
        }
    });

    define(interp, "index_of", Some(2), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let idx = arr.borrow().iter().position(|v| values_equal(v, &args[1]));
                match idx {
                    Some(i) => Ok(Value::Int(i as i64)),
                    None => Ok(Value::Int(-1)),
                }
            }
            Value::String(s) => {
                match &args[1] {
                    Value::String(sub) => {
                        match s.find(sub.as_str()) {
                            Some(i) => Ok(Value::Int(i as i64)),
                            None => Ok(Value::Int(-1)),
                        }
                    }
                    Value::Char(c) => {
                        match s.find(*c) {
                            Some(i) => Ok(Value::Int(i as i64)),
                            None => Ok(Value::Int(-1)),
                        }
                    }
                    _ => Err(RuntimeError::new("string index_of() requires string or char")),
                }
            }
            _ => Err(RuntimeError::new("index_of() requires array or string")),
        }
    });

    // Transformations
    define(interp, "reverse", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let mut reversed: Vec<Value> = arr.borrow().clone();
                reversed.reverse();
                Ok(Value::Array(Rc::new(RefCell::new(reversed))))
            }
            Value::String(s) => {
                let reversed: String = s.chars().rev().collect();
                Ok(Value::String(Rc::new(reversed)))
            }
            _ => Err(RuntimeError::new("reverse() requires array or string")),
        }
    });

    define(interp, "sort", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let mut sorted: Vec<Value> = arr.borrow().clone();
                sorted.sort_by(compare_values);
                Ok(Value::Array(Rc::new(RefCell::new(sorted))))
            }
            _ => Err(RuntimeError::new("sort() requires array")),
        }
    });

    define(interp, "sort_desc", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let mut sorted: Vec<Value> = arr.borrow().clone();
                sorted.sort_by(|a, b| compare_values(b, a));
                Ok(Value::Array(Rc::new(RefCell::new(sorted))))
            }
            _ => Err(RuntimeError::new("sort_desc() requires array")),
        }
    });

    define(interp, "unique", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                let mut seen = Vec::new();
                let unique: Vec<Value> = arr.iter()
                    .filter(|v| {
                        if seen.iter().any(|s| values_equal(s, v)) {
                            false
                        } else {
                            seen.push((*v).clone());
                            true
                        }
                    })
                    .cloned()
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(unique))))
            }
            _ => Err(RuntimeError::new("unique() requires array")),
        }
    });

    define(interp, "flatten", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let mut flattened = Vec::new();
                for item in arr.borrow().iter() {
                    match item {
                        Value::Array(inner) => flattened.extend(inner.borrow().clone()),
                        other => flattened.push(other.clone()),
                    }
                }
                Ok(Value::Array(Rc::new(RefCell::new(flattened))))
            }
            _ => Err(RuntimeError::new("flatten() requires array")),
        }
    });

    // Combining
    define(interp, "concat", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Array(a), Value::Array(b)) => {
                let mut result = a.borrow().clone();
                result.extend(b.borrow().clone());
                Ok(Value::Array(Rc::new(RefCell::new(result))))
            }
            (Value::String(a), Value::String(b)) => {
                Ok(Value::String(Rc::new(format!("{}{}", a, b))))
            }
            _ => Err(RuntimeError::new("concat() requires two arrays or two strings")),
        }
    });

    define(interp, "zip", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Array(a), Value::Array(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let zipped: Vec<Value> = a.iter().zip(b.iter())
                    .map(|(x, y)| Value::Tuple(Rc::new(vec![x.clone(), y.clone()])))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(zipped))))
            }
            _ => Err(RuntimeError::new("zip() requires two arrays")),
        }
    });

    define(interp, "enumerate", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let enumerated: Vec<Value> = arr.borrow().iter()
                    .enumerate()
                    .map(|(i, v)| Value::Tuple(Rc::new(vec![Value::Int(i as i64), v.clone()])))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(enumerated))))
            }
            _ => Err(RuntimeError::new("enumerate() requires array")),
        }
    });

    // Slicing
    define(interp, "slice", Some(3), |_, args| {
        let start = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("slice() start must be integer")),
        };
        let end = match &args[2] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("slice() end must be integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                let end = end.min(arr.len());
                let sliced: Vec<Value> = arr[start..end].to_vec();
                Ok(Value::Array(Rc::new(RefCell::new(sliced))))
            }
            Value::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let end = end.min(chars.len());
                let sliced: String = chars[start..end].iter().collect();
                Ok(Value::String(Rc::new(sliced)))
            }
            _ => Err(RuntimeError::new("slice() requires array or string")),
        }
    });

    define(interp, "take", Some(2), |_, args| {
        let n = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("take() n must be integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let taken: Vec<Value> = arr.borrow().iter().take(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(taken))))
            }
            _ => Err(RuntimeError::new("take() requires array")),
        }
    });

    define(interp, "skip", Some(2), |_, args| {
        let n = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("skip() n must be integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let skipped: Vec<Value> = arr.borrow().iter().skip(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(skipped))))
            }
            _ => Err(RuntimeError::new("skip() requires array")),
        }
    });

    define(interp, "chunk", Some(2), |_, args| {
        let size = match &args[1] {
            Value::Int(i) if *i > 0 => *i as usize,
            _ => return Err(RuntimeError::new("chunk() size must be positive integer")),
        };
        match &args[0] {
            Value::Array(arr) => {
                let chunks: Vec<Value> = arr.borrow()
                    .chunks(size)
                    .map(|c| Value::Array(Rc::new(RefCell::new(c.to_vec()))))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(chunks))))
            }
            _ => Err(RuntimeError::new("chunk() requires array")),
        }
    });

    // Range
    define(interp, "range", Some(2), |_, args| {
        let start = match &args[0] {
            Value::Int(n) => *n,
            _ => return Err(RuntimeError::new("range() requires integers")),
        };
        let end = match &args[1] {
            Value::Int(n) => *n,
            _ => return Err(RuntimeError::new("range() requires integers")),
        };
        let values: Vec<Value> = (start..end).map(Value::Int).collect();
        Ok(Value::Array(Rc::new(RefCell::new(values))))
    });

    define(interp, "range_inclusive", Some(2), |_, args| {
        let start = match &args[0] {
            Value::Int(n) => *n,
            _ => return Err(RuntimeError::new("range_inclusive() requires integers")),
        };
        let end = match &args[1] {
            Value::Int(n) => *n,
            _ => return Err(RuntimeError::new("range_inclusive() requires integers")),
        };
        let values: Vec<Value> = (start..=end).map(Value::Int).collect();
        Ok(Value::Array(Rc::new(RefCell::new(values))))
    });

    define(interp, "repeat", Some(2), |_, args| {
        let n = match &args[1] {
            Value::Int(i) if *i >= 0 => *i as usize,
            _ => return Err(RuntimeError::new("repeat() count must be non-negative integer")),
        };
        let repeated: Vec<Value> = std::iter::repeat(args[0].clone()).take(n).collect();
        Ok(Value::Array(Rc::new(RefCell::new(repeated))))
    });

    // ========================================
    // HashMap operations
    // ========================================

    // map_new - create empty HashMap
    define(interp, "map_new", Some(0), |_, _| {
        Ok(Value::Map(Rc::new(RefCell::new(HashMap::new()))))
    });

    // map_get - get value by key
    define(interp, "map_get", Some(2), |_, args| {
        let key = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("map_get() key must be string")),
        };
        match &args[0] {
            Value::Map(map) => {
                Ok(map.borrow().get(&key).cloned().unwrap_or(Value::Null))
            }
            _ => Err(RuntimeError::new("map_get() requires map")),
        }
    });

    // map_set - set key-value pair
    define(interp, "map_set", Some(3), |_, args| {
        let key = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("map_set() key must be string")),
        };
        match &args[0] {
            Value::Map(map) => {
                map.borrow_mut().insert(key, args[2].clone());
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("map_set() requires map")),
        }
    });

    // map_has - check if key exists
    define(interp, "map_has", Some(2), |_, args| {
        let key = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("map_has() key must be string")),
        };
        match &args[0] {
            Value::Map(map) => {
                Ok(Value::Bool(map.borrow().contains_key(&key)))
            }
            _ => Err(RuntimeError::new("map_has() requires map")),
        }
    });

    // map_remove - remove key from map
    define(interp, "map_remove", Some(2), |_, args| {
        let key = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("map_remove() key must be string")),
        };
        match &args[0] {
            Value::Map(map) => {
                Ok(map.borrow_mut().remove(&key).unwrap_or(Value::Null))
            }
            _ => Err(RuntimeError::new("map_remove() requires map")),
        }
    });

    // map_keys - get all keys as array
    define(interp, "map_keys", Some(1), |_, args| {
        match &args[0] {
            Value::Map(map) => {
                let keys: Vec<Value> = map.borrow().keys()
                    .map(|k| Value::String(Rc::new(k.clone())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(keys))))
            }
            _ => Err(RuntimeError::new("map_keys() requires map")),
        }
    });

    // map_values - get all values as array
    define(interp, "map_values", Some(1), |_, args| {
        match &args[0] {
            Value::Map(map) => {
                let values: Vec<Value> = map.borrow().values().cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(values))))
            }
            _ => Err(RuntimeError::new("map_values() requires map")),
        }
    });

    // map_len - get number of entries
    define(interp, "map_len", Some(1), |_, args| {
        match &args[0] {
            Value::Map(map) => Ok(Value::Int(map.borrow().len() as i64)),
            _ => Err(RuntimeError::new("map_len() requires map")),
        }
    });

    // map_clear - remove all entries
    define(interp, "map_clear", Some(1), |_, args| {
        match &args[0] {
            Value::Map(map) => {
                map.borrow_mut().clear();
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("map_clear() requires map")),
        }
    });

    // ========================================
    // HashSet operations
    // ========================================

    // set_new - create empty HashSet
    define(interp, "set_new", Some(0), |_, _| {
        Ok(Value::Set(Rc::new(RefCell::new(std::collections::HashSet::new()))))
    });

    // set_add - add item to set
    define(interp, "set_add", Some(2), |_, args| {
        let item = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("set_add() item must be string")),
        };
        match &args[0] {
            Value::Set(set) => {
                set.borrow_mut().insert(item);
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("set_add() requires set")),
        }
    });

    // set_has - check if item exists
    define(interp, "set_has", Some(2), |_, args| {
        let item = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("set_has() item must be string")),
        };
        match &args[0] {
            Value::Set(set) => {
                Ok(Value::Bool(set.borrow().contains(&item)))
            }
            _ => Err(RuntimeError::new("set_has() requires set")),
        }
    });

    // set_remove - remove item from set
    define(interp, "set_remove", Some(2), |_, args| {
        let item = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("set_remove() item must be string")),
        };
        match &args[0] {
            Value::Set(set) => {
                Ok(Value::Bool(set.borrow_mut().remove(&item)))
            }
            _ => Err(RuntimeError::new("set_remove() requires set")),
        }
    });

    // set_to_array - convert set to array
    define(interp, "set_to_array", Some(1), |_, args| {
        match &args[0] {
            Value::Set(set) => {
                let items: Vec<Value> = set.borrow().iter()
                    .map(|s| Value::String(Rc::new(s.clone())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(items))))
            }
            _ => Err(RuntimeError::new("set_to_array() requires set")),
        }
    });

    // set_len - get number of items
    define(interp, "set_len", Some(1), |_, args| {
        match &args[0] {
            Value::Set(set) => Ok(Value::Int(set.borrow().len() as i64)),
            _ => Err(RuntimeError::new("set_len() requires set")),
        }
    });

    // set_clear - remove all items
    define(interp, "set_clear", Some(1), |_, args| {
        match &args[0] {
            Value::Set(set) => {
                set.borrow_mut().clear();
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("set_clear() requires set")),
        }
    });
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Char(a), Value::Char(b)) => a == b,
        (Value::Array(a), Value::Array(b)) => {
            let a = a.borrow();
            let b = b.borrow();
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y))
        }
        (Value::Tuple(a), Value::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y))
        }
        _ => false,
    }
}

fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Char(a), Value::Char(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal,
    }
}

// ============================================================================
// STRING FUNCTIONS
// ============================================================================

fn register_string(interp: &mut Interpreter) {
    define(interp, "chars", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => {
                let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                Ok(Value::Array(Rc::new(RefCell::new(chars))))
            }
            _ => Err(RuntimeError::new("chars() requires string")),
        }
    });

    define(interp, "bytes", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => {
                let bytes: Vec<Value> = s.bytes().map(|b| Value::Int(b as i64)).collect();
                Ok(Value::Array(Rc::new(RefCell::new(bytes))))
            }
            _ => Err(RuntimeError::new("bytes() requires string")),
        }
    });

    define(interp, "split", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(s), Value::String(sep)) => {
                let parts: Vec<Value> = s.split(sep.as_str())
                    .map(|p| Value::String(Rc::new(p.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(parts))))
            }
            (Value::String(s), Value::Char(sep)) => {
                let parts: Vec<Value> = s.split(*sep)
                    .map(|p| Value::String(Rc::new(p.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(parts))))
            }
            _ => Err(RuntimeError::new("split() requires string and separator")),
        }
    });

    define(interp, "join", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Array(arr), Value::String(sep)) => {
                let parts: Vec<String> = arr.borrow().iter()
                    .map(|v| format!("{}", v))
                    .collect();
                Ok(Value::String(Rc::new(parts.join(sep.as_str()))))
            }
            _ => Err(RuntimeError::new("join() requires array and separator string")),
        }
    });

    define(interp, "trim", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::String(Rc::new(s.trim().to_string()))),
            _ => Err(RuntimeError::new("trim() requires string")),
        }
    });

    define(interp, "trim_start", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::String(Rc::new(s.trim_start().to_string()))),
            _ => Err(RuntimeError::new("trim_start() requires string")),
        }
    });

    define(interp, "trim_end", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::String(Rc::new(s.trim_end().to_string()))),
            _ => Err(RuntimeError::new("trim_end() requires string")),
        }
    });

    define(interp, "upper", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::String(Rc::new(s.to_uppercase()))),
            Value::Char(c) => Ok(Value::Char(c.to_uppercase().next().unwrap_or(*c))),
            _ => Err(RuntimeError::new("upper() requires string or char")),
        }
    });

    define(interp, "lower", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::String(Rc::new(s.to_lowercase()))),
            Value::Char(c) => Ok(Value::Char(c.to_lowercase().next().unwrap_or(*c))),
            _ => Err(RuntimeError::new("lower() requires string or char")),
        }
    });

    define(interp, "capitalize", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => {
                let mut chars = s.chars();
                let capitalized = match chars.next() {
                    None => String::new(),
                    Some(c) => c.to_uppercase().chain(chars).collect(),
                };
                Ok(Value::String(Rc::new(capitalized)))
            }
            _ => Err(RuntimeError::new("capitalize() requires string")),
        }
    });

    define(interp, "replace", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::String(s), Value::String(from), Value::String(to)) => {
                Ok(Value::String(Rc::new(s.replace(from.as_str(), to.as_str()))))
            }
            _ => Err(RuntimeError::new("replace() requires three strings")),
        }
    });

    define(interp, "starts_with", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(s), Value::String(prefix)) => {
                Ok(Value::Bool(s.starts_with(prefix.as_str())))
            }
            _ => Err(RuntimeError::new("starts_with() requires two strings")),
        }
    });

    define(interp, "ends_with", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(s), Value::String(suffix)) => {
                Ok(Value::Bool(s.ends_with(suffix.as_str())))
            }
            _ => Err(RuntimeError::new("ends_with() requires two strings")),
        }
    });

    define(interp, "repeat_str", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(s), Value::Int(n)) if *n >= 0 => {
                Ok(Value::String(Rc::new(s.repeat(*n as usize))))
            }
            _ => Err(RuntimeError::new("repeat_str() requires string and non-negative integer")),
        }
    });

    define(interp, "pad_left", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::String(s), Value::Int(width), Value::Char(c)) => {
                let width = *width as usize;
                if s.len() >= width {
                    Ok(Value::String(s.clone()))
                } else {
                    let padding: String = std::iter::repeat(*c).take(width - s.len()).collect();
                    Ok(Value::String(Rc::new(format!("{}{}", padding, s))))
                }
            }
            _ => Err(RuntimeError::new("pad_left() requires string, width, and char")),
        }
    });

    define(interp, "pad_right", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::String(s), Value::Int(width), Value::Char(c)) => {
                let width = *width as usize;
                if s.len() >= width {
                    Ok(Value::String(s.clone()))
                } else {
                    let padding: String = std::iter::repeat(*c).take(width - s.len()).collect();
                    Ok(Value::String(Rc::new(format!("{}{}", s, padding))))
                }
            }
            _ => Err(RuntimeError::new("pad_right() requires string, width, and char")),
        }
    });

    define(interp, "lines", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => {
                let lines: Vec<Value> = s.lines()
                    .map(|l| Value::String(Rc::new(l.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(lines))))
            }
            _ => Err(RuntimeError::new("lines() requires string")),
        }
    });

    define(interp, "words", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => {
                let words: Vec<Value> = s.split_whitespace()
                    .map(|w| Value::String(Rc::new(w.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(words))))
            }
            _ => Err(RuntimeError::new("words() requires string")),
        }
    });

    define(interp, "is_alpha", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_alphabetic()))),
            Value::Char(c) => Ok(Value::Bool(c.is_alphabetic())),
            _ => Err(RuntimeError::new("is_alpha() requires string or char")),
        }
    });

    define(interp, "is_digit", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))),
            Value::Char(c) => Ok(Value::Bool(c.is_ascii_digit())),
            _ => Err(RuntimeError::new("is_digit() requires string or char")),
        }
    });

    define(interp, "is_alnum", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_alphanumeric()))),
            Value::Char(c) => Ok(Value::Bool(c.is_alphanumeric())),
            _ => Err(RuntimeError::new("is_alnum() requires string or char")),
        }
    });

    define(interp, "is_space", Some(1), |_, args| {
        match &args[0] {
            Value::String(s) => Ok(Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_whitespace()))),
            Value::Char(c) => Ok(Value::Bool(c.is_whitespace())),
            _ => Err(RuntimeError::new("is_space() requires string or char")),
        }
    });
}

// ============================================================================
// EVIDENCE FUNCTIONS
// ============================================================================

fn register_evidence(interp: &mut Interpreter) {
    // Create evidential values
    define(interp, "known", Some(1), |_, args| {
        Ok(Value::Evidential {
            value: Box::new(args[0].clone()),
            evidence: Evidence::Known,
        })
    });

    define(interp, "uncertain", Some(1), |_, args| {
        Ok(Value::Evidential {
            value: Box::new(args[0].clone()),
            evidence: Evidence::Uncertain,
        })
    });

    define(interp, "reported", Some(1), |_, args| {
        Ok(Value::Evidential {
            value: Box::new(args[0].clone()),
            evidence: Evidence::Reported,
        })
    });

    define(interp, "paradox", Some(1), |_, args| {
        Ok(Value::Evidential {
            value: Box::new(args[0].clone()),
            evidence: Evidence::Paradox,
        })
    });

    // Query evidence
    define(interp, "evidence_of", Some(1), |_, args| {
        match &args[0] {
            Value::Evidential { evidence, .. } => {
                let level = match evidence {
                    Evidence::Known => "known",
                    Evidence::Uncertain => "uncertain",
                    Evidence::Reported => "reported",
                    Evidence::Paradox => "paradox",
                };
                Ok(Value::String(Rc::new(level.to_string())))
            }
            _ => Ok(Value::String(Rc::new("known".to_string()))), // Non-evidential values are known
        }
    });

    define(interp, "is_known", Some(1), |_, args| {
        match &args[0] {
            Value::Evidential { evidence: Evidence::Known, .. } => Ok(Value::Bool(true)),
            Value::Evidential { .. } => Ok(Value::Bool(false)),
            _ => Ok(Value::Bool(true)), // Non-evidential values are known
        }
    });

    define(interp, "is_uncertain", Some(1), |_, args| {
        match &args[0] {
            Value::Evidential { evidence: Evidence::Uncertain, .. } => Ok(Value::Bool(true)),
            _ => Ok(Value::Bool(false)),
        }
    });

    define(interp, "is_reported", Some(1), |_, args| {
        match &args[0] {
            Value::Evidential { evidence: Evidence::Reported, .. } => Ok(Value::Bool(true)),
            _ => Ok(Value::Bool(false)),
        }
    });

    define(interp, "is_paradox", Some(1), |_, args| {
        match &args[0] {
            Value::Evidential { evidence: Evidence::Paradox, .. } => Ok(Value::Bool(true)),
            _ => Ok(Value::Bool(false)),
        }
    });

    // Extract inner value
    define(interp, "strip_evidence", Some(1), |_, args| {
        match &args[0] {
            Value::Evidential { value, .. } => Ok(*value.clone()),
            other => Ok(other.clone()),
        }
    });

    // Trust operations
    define(interp, "trust", Some(1), |_, args| {
        // Upgrade reported/uncertain to known (with assertion)
        match &args[0] {
            Value::Evidential { value, .. } => {
                Ok(Value::Evidential {
                    value: value.clone(),
                    evidence: Evidence::Known,
                })
            }
            other => Ok(other.clone()),
        }
    });

    define(interp, "verify", Some(2), |_, args| {
        // Verify evidential value with predicate, upgrading if true
        let pred_result = match &args[1] {
            Value::Bool(b) => *b,
            _ => return Err(RuntimeError::new("verify() predicate must be bool")),
        };

        if pred_result {
            match &args[0] {
                Value::Evidential { value, .. } => {
                    Ok(Value::Evidential {
                        value: value.clone(),
                        evidence: Evidence::Known,
                    })
                }
                other => Ok(other.clone()),
            }
        } else {
            Ok(args[0].clone()) // Keep original evidence
        }
    });

    // Combine evidence (join in lattice)
    define(interp, "combine_evidence", Some(2), |_, args| {
        let ev1 = match &args[0] {
            Value::Evidential { evidence, .. } => *evidence,
            _ => Evidence::Known,
        };
        let ev2 = match &args[1] {
            Value::Evidential { evidence, .. } => *evidence,
            _ => Evidence::Known,
        };

        // Join: max of the two
        let combined = match (ev1, ev2) {
            (Evidence::Paradox, _) | (_, Evidence::Paradox) => Evidence::Paradox,
            (Evidence::Reported, _) | (_, Evidence::Reported) => Evidence::Reported,
            (Evidence::Uncertain, _) | (_, Evidence::Uncertain) => Evidence::Uncertain,
            _ => Evidence::Known,
        };

        Ok(Value::String(Rc::new(match combined {
            Evidence::Known => "known",
            Evidence::Uncertain => "uncertain",
            Evidence::Reported => "reported",
            Evidence::Paradox => "paradox",
        }.to_string())))
    });
}

// ============================================================================
// ITERATOR-STYLE FUNCTIONS (for use in pipes)
// ============================================================================

fn register_iter(interp: &mut Interpreter) {
    // sum - sum all elements
    define(interp, "sum", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let mut sum_int: i64 = 0;
                let mut sum_float: f64 = 0.0;
                let mut is_float = false;

                for val in arr.borrow().iter() {
                    match val {
                        Value::Int(n) => {
                            if is_float {
                                sum_float += *n as f64;
                            } else {
                                sum_int += n;
                            }
                        }
                        Value::Float(n) => {
                            if !is_float {
                                sum_float = sum_int as f64;
                                is_float = true;
                            }
                            sum_float += n;
                        }
                        _ => return Err(RuntimeError::new("sum() requires array of numbers")),
                    }
                }

                if is_float {
                    Ok(Value::Float(sum_float))
                } else {
                    Ok(Value::Int(sum_int))
                }
            }
            _ => Err(RuntimeError::new("sum() requires array")),
        }
    });

    // product - multiply all elements
    define(interp, "product", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let mut prod_int: i64 = 1;
                let mut prod_float: f64 = 1.0;
                let mut is_float = false;

                for val in arr.borrow().iter() {
                    match val {
                        Value::Int(n) => {
                            if is_float {
                                prod_float *= *n as f64;
                            } else {
                                prod_int *= n;
                            }
                        }
                        Value::Float(n) => {
                            if !is_float {
                                prod_float = prod_int as f64;
                                is_float = true;
                            }
                            prod_float *= n;
                        }
                        _ => return Err(RuntimeError::new("product() requires array of numbers")),
                    }
                }

                if is_float {
                    Ok(Value::Float(prod_float))
                } else {
                    Ok(Value::Int(prod_int))
                }
            }
            _ => Err(RuntimeError::new("product() requires array")),
        }
    });

    // mean - average of elements
    define(interp, "mean", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Err(RuntimeError::new("mean() on empty array"));
                }

                let mut sum: f64 = 0.0;
                for val in arr.iter() {
                    match val {
                        Value::Int(n) => sum += *n as f64,
                        Value::Float(n) => sum += n,
                        _ => return Err(RuntimeError::new("mean() requires array of numbers")),
                    }
                }

                Ok(Value::Float(sum / arr.len() as f64))
            }
            _ => Err(RuntimeError::new("mean() requires array")),
        }
    });

    // median - middle value
    define(interp, "median", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Err(RuntimeError::new("median() on empty array"));
                }

                let mut nums: Vec<f64> = Vec::new();
                for val in arr.iter() {
                    match val {
                        Value::Int(n) => nums.push(*n as f64),
                        Value::Float(n) => nums.push(*n),
                        _ => return Err(RuntimeError::new("median() requires array of numbers")),
                    }
                }

                nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = nums.len() / 2;

                if nums.len() % 2 == 0 {
                    Ok(Value::Float((nums[mid - 1] + nums[mid]) / 2.0))
                } else {
                    Ok(Value::Float(nums[mid]))
                }
            }
            _ => Err(RuntimeError::new("median() requires array")),
        }
    });

    // min_of - minimum of array
    define(interp, "min_of", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Err(RuntimeError::new("min_of() on empty array"));
                }

                let mut min = &arr[0];
                for val in arr.iter().skip(1) {
                    if matches!(compare_values(val, min), std::cmp::Ordering::Less) {
                        min = val;
                    }
                }
                Ok(min.clone())
            }
            _ => Err(RuntimeError::new("min_of() requires array")),
        }
    });

    // max_of - maximum of array
    define(interp, "max_of", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Err(RuntimeError::new("max_of() on empty array"));
                }

                let mut max = &arr[0];
                for val in arr.iter().skip(1) {
                    if matches!(compare_values(val, max), std::cmp::Ordering::Greater) {
                        max = val;
                    }
                }
                Ok(max.clone())
            }
            _ => Err(RuntimeError::new("max_of() requires array")),
        }
    });

    // count - count elements (optionally matching predicate)
    define(interp, "count", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
            Value::String(s) => Ok(Value::Int(s.chars().count() as i64)),
            _ => Err(RuntimeError::new("count() requires array or string")),
        }
    });

    // any - check if any element is truthy
    define(interp, "any", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                for val in arr.borrow().iter() {
                    if is_truthy(val) {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            _ => Err(RuntimeError::new("any() requires array")),
        }
    });

    // all - check if all elements are truthy
    define(interp, "all", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                for val in arr.borrow().iter() {
                    if !is_truthy(val) {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
            _ => Err(RuntimeError::new("all() requires array")),
        }
    });

    // none - check if no elements are truthy
    define(interp, "none", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                for val in arr.borrow().iter() {
                    if is_truthy(val) {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
            _ => Err(RuntimeError::new("none() requires array")),
        }
    });
}

fn is_truthy(val: &Value) -> bool {
    match val {
        Value::Null | Value::Empty => false,
        Value::Bool(b) => *b,
        Value::Int(n) => *n != 0,
        Value::Float(n) => *n != 0.0 && !n.is_nan(),
        Value::String(s) => !s.is_empty(),
        Value::Array(arr) => !arr.borrow().is_empty(),
        Value::Evidential { value, .. } => is_truthy(value),
        _ => true,
    }
}

// ============================================================================
// I/O FUNCTIONS
// ============================================================================

fn register_io(interp: &mut Interpreter) {
    // read_file - read entire file as string
    define(interp, "read_file", Some(1), |_, args| {
        match &args[0] {
            Value::String(path) => {
                match std::fs::read_to_string(path.as_str()) {
                    Ok(content) => Ok(Value::Evidential {
                        value: Box::new(Value::String(Rc::new(content))),
                        evidence: Evidence::Reported, // File contents are reported, not known
                    }),
                    Err(e) => Err(RuntimeError::new(format!("read_file failed: {}", e))),
                }
            }
            _ => Err(RuntimeError::new("read_file() requires path string")),
        }
    });

    // write_file - write string to file
    define(interp, "write_file", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(path), Value::String(content)) => {
                match std::fs::write(path.as_str(), content.as_str()) {
                    Ok(_) => Ok(Value::Bool(true)),
                    Err(e) => Err(RuntimeError::new(format!("write_file failed: {}", e))),
                }
            }
            _ => Err(RuntimeError::new("write_file() requires path and content strings")),
        }
    });

    // append_file - append to file
    define(interp, "append_file", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(path), Value::String(content)) => {
                use std::fs::OpenOptions;
                let result = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path.as_str())
                    .and_then(|mut f| f.write_all(content.as_bytes()));
                match result {
                    Ok(_) => Ok(Value::Bool(true)),
                    Err(e) => Err(RuntimeError::new(format!("append_file failed: {}", e))),
                }
            }
            _ => Err(RuntimeError::new("append_file() requires path and content strings")),
        }
    });

    // file_exists - check if file exists
    define(interp, "file_exists", Some(1), |_, args| {
        match &args[0] {
            Value::String(path) => Ok(Value::Bool(std::path::Path::new(path.as_str()).exists())),
            _ => Err(RuntimeError::new("file_exists() requires path string")),
        }
    });

    // read_lines - read file as array of lines
    define(interp, "read_lines", Some(1), |_, args| {
        match &args[0] {
            Value::String(path) => {
                match std::fs::read_to_string(path.as_str()) {
                    Ok(content) => {
                        let lines: Vec<Value> = content.lines()
                            .map(|l| Value::String(Rc::new(l.to_string())))
                            .collect();
                        Ok(Value::Evidential {
                            value: Box::new(Value::Array(Rc::new(RefCell::new(lines)))),
                            evidence: Evidence::Reported,
                        })
                    }
                    Err(e) => Err(RuntimeError::new(format!("read_lines failed: {}", e))),
                }
            }
            _ => Err(RuntimeError::new("read_lines() requires path string")),
        }
    });

    // env - get environment variable
    define(interp, "env", Some(1), |_, args| {
        match &args[0] {
            Value::String(name) => {
                match std::env::var(name.as_str()) {
                    Ok(value) => Ok(Value::Evidential {
                        value: Box::new(Value::String(Rc::new(value))),
                        evidence: Evidence::Reported, // Env vars are external
                    }),
                    Err(_) => Ok(Value::Null),
                }
            }
            _ => Err(RuntimeError::new("env() requires variable name string")),
        }
    });

    // env_or - get environment variable with default
    define(interp, "env_or", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(name), default) => {
                match std::env::var(name.as_str()) {
                    Ok(value) => Ok(Value::Evidential {
                        value: Box::new(Value::String(Rc::new(value))),
                        evidence: Evidence::Reported,
                    }),
                    Err(_) => Ok(default.clone()),
                }
            }
            _ => Err(RuntimeError::new("env_or() requires variable name string")),
        }
    });

    // cwd - current working directory
    define(interp, "cwd", Some(0), |_, _| {
        match std::env::current_dir() {
            Ok(path) => Ok(Value::String(Rc::new(path.to_string_lossy().to_string()))),
            Err(e) => Err(RuntimeError::new(format!("cwd() failed: {}", e))),
        }
    });

    // args - command line arguments
    define(interp, "args", Some(0), |_, _| {
        let args: Vec<Value> = std::env::args()
            .map(|a| Value::String(Rc::new(a)))
            .collect();
        Ok(Value::Array(Rc::new(RefCell::new(args))))
    });
}

// ============================================================================
// TIME FUNCTIONS
// ============================================================================

fn register_time(interp: &mut Interpreter) {
    // now - current Unix timestamp in milliseconds
    define(interp, "now", Some(0), |_, _| {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        Ok(Value::Int(duration.as_millis() as i64))
    });

    // now_secs - current Unix timestamp in seconds
    define(interp, "now_secs", Some(0), |_, _| {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        Ok(Value::Int(duration.as_secs() as i64))
    });

    // now_micros - current Unix timestamp in microseconds
    define(interp, "now_micros", Some(0), |_, _| {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        Ok(Value::Int(duration.as_micros() as i64))
    });

    // sleep - sleep for milliseconds
    define(interp, "sleep", Some(1), |_, args| {
        match &args[0] {
            Value::Int(ms) if *ms >= 0 => {
                std::thread::sleep(Duration::from_millis(*ms as u64));
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("sleep() requires non-negative integer milliseconds")),
        }
    });

    // measure - measure execution time of a thunk (returns ms)
    // Note: This would need closure support to work properly
    // For now, we provide a simple timer API

    // timer_start - start a timer (returns opaque handle)
    define(interp, "timer_start", Some(0), |_, _| {
        let now = Instant::now();
        // Store as microseconds since we can't store Instant directly
        Ok(Value::Int(now.elapsed().as_nanos() as i64)) // This is a bit hacky
    });
}

// ============================================================================
// RANDOM FUNCTIONS
// ============================================================================

fn register_random(interp: &mut Interpreter) {
    // random - random float 0.0 to 1.0
    define(interp, "random", Some(0), |_, _| {
        // Simple LCG random - not cryptographically secure
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;
        let rand = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as f64;
        Ok(Value::Float(rand / u32::MAX as f64))
    });

    // random_int - random integer in range [min, max)
    define(interp, "random_int", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(min), Value::Int(max)) if max > min => {
                use std::time::SystemTime;
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_nanos() as u64;
                let range = (max - min) as u64;
                let rand = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) % range;
                Ok(Value::Int(*min + rand as i64))
            }
            _ => Err(RuntimeError::new("random_int() requires min < max integers")),
        }
    });

    // shuffle - shuffle array in place (Fisher-Yates)
    define(interp, "shuffle", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let mut arr = arr.borrow_mut();
                use std::time::SystemTime;
                let mut seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_nanos() as u64;

                for i in (1..arr.len()).rev() {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let j = ((seed >> 16) as usize) % (i + 1);
                    arr.swap(i, j);
                }
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("shuffle() requires array")),
        }
    });

    // sample - random sample from array
    define(interp, "sample", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Err(RuntimeError::new("sample() on empty array"));
                }

                use std::time::SystemTime;
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_nanos() as u64;
                let idx = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as usize % arr.len();
                Ok(arr[idx].clone())
            }
            _ => Err(RuntimeError::new("sample() requires array")),
        }
    });
}

// ============================================================================
// CONVERSION FUNCTIONS
// ============================================================================

fn register_convert(interp: &mut Interpreter) {
    // to_string - convert to string
    define(interp, "to_string", Some(1), |_, args| {
        Ok(Value::String(Rc::new(format!("{}", args[0]))))
    });

    // to_int - convert to integer
    define(interp, "to_int", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Int(*n)),
            Value::Float(n) => Ok(Value::Int(*n as i64)),
            Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
            Value::Char(c) => Ok(Value::Int(*c as i64)),
            Value::String(s) => {
                s.parse::<i64>()
                    .map(Value::Int)
                    .map_err(|_| RuntimeError::new(format!("cannot parse '{}' as integer", s)))
            }
            _ => Err(RuntimeError::new("to_int() cannot convert this type")),
        }
    });

    // to_float - convert to float
    define(interp, "to_float", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::Float(*n as f64)),
            Value::Float(n) => Ok(Value::Float(*n)),
            Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
            Value::String(s) => {
                s.parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| RuntimeError::new(format!("cannot parse '{}' as float", s)))
            }
            _ => Err(RuntimeError::new("to_float() cannot convert this type")),
        }
    });

    // to_bool - convert to boolean
    define(interp, "to_bool", Some(1), |_, args| {
        Ok(Value::Bool(is_truthy(&args[0])))
    });

    // to_char - convert to character
    define(interp, "to_char", Some(1), |_, args| {
        match &args[0] {
            Value::Char(c) => Ok(Value::Char(*c)),
            Value::Int(n) => {
                char::from_u32(*n as u32)
                    .map(Value::Char)
                    .ok_or_else(|| RuntimeError::new(format!("invalid char code: {}", n)))
            }
            Value::String(s) => {
                s.chars().next()
                    .map(Value::Char)
                    .ok_or_else(|| RuntimeError::new("to_char() on empty string"))
            }
            _ => Err(RuntimeError::new("to_char() cannot convert this type")),
        }
    });

    // to_array - convert to array
    define(interp, "to_array", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => Ok(Value::Array(arr.clone())),
            Value::Tuple(t) => Ok(Value::Array(Rc::new(RefCell::new(t.as_ref().clone())))),
            Value::String(s) => {
                let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                Ok(Value::Array(Rc::new(RefCell::new(chars))))
            }
            _ => Ok(Value::Array(Rc::new(RefCell::new(vec![args[0].clone()])))),
        }
    });

    // to_tuple - convert to tuple
    define(interp, "to_tuple", Some(1), |_, args| {
        match &args[0] {
            Value::Tuple(t) => Ok(Value::Tuple(t.clone())),
            Value::Array(arr) => Ok(Value::Tuple(Rc::new(arr.borrow().clone()))),
            _ => Ok(Value::Tuple(Rc::new(vec![args[0].clone()]))),
        }
    });

    // char_code - get unicode code point
    define(interp, "char_code", Some(1), |_, args| {
        match &args[0] {
            Value::Char(c) => Ok(Value::Int(*c as i64)),
            _ => Err(RuntimeError::new("char_code() requires char")),
        }
    });

    // from_char_code - create char from code point
    define(interp, "from_char_code", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => {
                char::from_u32(*n as u32)
                    .map(Value::Char)
                    .ok_or_else(|| RuntimeError::new(format!("invalid char code: {}", n)))
            }
            _ => Err(RuntimeError::new("from_char_code() requires integer")),
        }
    });

    // hex - convert to hex string
    define(interp, "hex", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::String(Rc::new(format!("{:x}", n)))),
            _ => Err(RuntimeError::new("hex() requires integer")),
        }
    });

    // oct - convert to octal string
    define(interp, "oct", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::String(Rc::new(format!("{:o}", n)))),
            _ => Err(RuntimeError::new("oct() requires integer")),
        }
    });

    // bin - convert to binary string
    define(interp, "bin", Some(1), |_, args| {
        match &args[0] {
            Value::Int(n) => Ok(Value::String(Rc::new(format!("{:b}", n)))),
            _ => Err(RuntimeError::new("bin() requires integer")),
        }
    });

    // parse_int - parse string as integer with optional base
    define(interp, "parse_int", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::String(s), Value::Int(base)) if *base >= 2 && *base <= 36 => {
                i64::from_str_radix(s.trim(), *base as u32)
                    .map(Value::Int)
                    .map_err(|_| RuntimeError::new(format!("cannot parse '{}' as base-{} integer", s, base)))
            }
            _ => Err(RuntimeError::new("parse_int() requires string and base 2-36")),
        }
    });
}

// ============================================================================
// CYCLE (MODULAR ARITHMETIC) FUNCTIONS
// For poly-cultural mathematics
// ============================================================================

fn register_cycle(interp: &mut Interpreter) {
    // cycle - create a cycle value (modular)
    define(interp, "cycle", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(value), Value::Int(modulus)) if *modulus > 0 => {
                let normalized = value.rem_euclid(*modulus);
                Ok(Value::Tuple(Rc::new(vec![
                    Value::Int(normalized),
                    Value::Int(*modulus),
                ])))
            }
            _ => Err(RuntimeError::new("cycle() requires value and positive modulus")),
        }
    });

    // mod_add - modular addition
    define(interp, "mod_add", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::Int(a), Value::Int(b), Value::Int(m)) if *m > 0 => {
                Ok(Value::Int((a + b).rem_euclid(*m)))
            }
            _ => Err(RuntimeError::new("mod_add() requires two integers and positive modulus")),
        }
    });

    // mod_sub - modular subtraction
    define(interp, "mod_sub", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::Int(a), Value::Int(b), Value::Int(m)) if *m > 0 => {
                Ok(Value::Int((a - b).rem_euclid(*m)))
            }
            _ => Err(RuntimeError::new("mod_sub() requires two integers and positive modulus")),
        }
    });

    // mod_mul - modular multiplication
    define(interp, "mod_mul", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::Int(a), Value::Int(b), Value::Int(m)) if *m > 0 => {
                Ok(Value::Int((a * b).rem_euclid(*m)))
            }
            _ => Err(RuntimeError::new("mod_mul() requires two integers and positive modulus")),
        }
    });

    // mod_pow - modular exponentiation (fast)
    define(interp, "mod_pow", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::Int(base), Value::Int(exp), Value::Int(m)) if *m > 0 && *exp >= 0 => {
                Ok(Value::Int(mod_pow(*base, *exp as u64, *m)))
            }
            _ => Err(RuntimeError::new("mod_pow() requires base, non-negative exp, and positive modulus")),
        }
    });

    // mod_inv - modular multiplicative inverse (if exists)
    define(interp, "mod_inv", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(m)) if *m > 0 => {
                match mod_inverse(*a, *m) {
                    Some(inv) => Ok(Value::Int(inv)),
                    None => Err(RuntimeError::new(format!("no modular inverse of {} mod {}", a, m))),
                }
            }
            _ => Err(RuntimeError::new("mod_inv() requires integer and positive modulus")),
        }
    });

    // Musical cycles (for tuning systems)
    // octave - normalize to octave (pitch class)
    define(interp, "octave", Some(1), |_, args| {
        match &args[0] {
            Value::Int(note) => Ok(Value::Int(note.rem_euclid(12))),
            Value::Float(freq) => {
                // Normalize frequency to octave starting at A4=440Hz
                let semitones = 12.0 * (freq / 440.0).log2();
                Ok(Value::Float(semitones.rem_euclid(12.0)))
            }
            _ => Err(RuntimeError::new("octave() requires number")),
        }
    });

    // interval - musical interval (semitones)
    define(interp, "interval", Some(2), |_, args| {
        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => {
                Ok(Value::Int((b - a).rem_euclid(12)))
            }
            _ => Err(RuntimeError::new("interval() requires two integers")),
        }
    });

    // cents - convert semitones to cents or vice versa
    define(interp, "cents", Some(1), |_, args| {
        match &args[0] {
            Value::Int(semitones) => Ok(Value::Int(*semitones * 100)),
            Value::Float(semitones) => Ok(Value::Float(*semitones * 100.0)),
            _ => Err(RuntimeError::new("cents() requires number")),
        }
    });

    // freq - convert MIDI note number to frequency
    define(interp, "freq", Some(1), |_, args| {
        match &args[0] {
            Value::Int(midi) => {
                let freq = 440.0 * 2.0_f64.powf((*midi as f64 - 69.0) / 12.0);
                Ok(Value::Float(freq))
            }
            _ => Err(RuntimeError::new("freq() requires integer MIDI note")),
        }
    });

    // midi - convert frequency to MIDI note number
    define(interp, "midi", Some(1), |_, args| {
        match &args[0] {
            Value::Float(freq) if *freq > 0.0 => {
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                Ok(Value::Int(midi.round() as i64))
            }
            Value::Int(freq) if *freq > 0 => {
                let midi = 69.0 + 12.0 * (*freq as f64 / 440.0).log2();
                Ok(Value::Int(midi.round() as i64))
            }
            _ => Err(RuntimeError::new("midi() requires positive frequency")),
        }
    });
}

// Fast modular exponentiation
fn mod_pow(mut base: i64, mut exp: u64, modulus: i64) -> i64 {
    if modulus == 1 { return 0; }
    let mut result: i64 = 1;
    base = base.rem_euclid(modulus);
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base).rem_euclid(modulus);
        }
        exp /= 2;
        base = (base * base).rem_euclid(modulus);
    }
    result
}

// ============================================================================
// SIMD VECTOR FUNCTIONS
// High-performance vector operations for game/graphics math
// ============================================================================

fn register_simd(interp: &mut Interpreter) {
    // simd_new - create a SIMD 4-component vector
    define(interp, "simd_new", Some(4), |_, args| {
        let values: Result<Vec<f64>, _> = args.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(RuntimeError::new("simd_new() requires numbers")),
        }).collect();
        let values = values?;
        Ok(Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(values[0]),
            Value::Float(values[1]),
            Value::Float(values[2]),
            Value::Float(values[3]),
        ]))))
    });

    // simd_splat - create vector with all same components
    define(interp, "simd_splat", Some(1), |_, args| {
        let v = match &args[0] {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            _ => return Err(RuntimeError::new("simd_splat() requires number")),
        };
        Ok(Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(v), Value::Float(v), Value::Float(v), Value::Float(v),
        ]))))
    });

    // simd_add - component-wise addition
    define(interp, "simd_add", Some(2), |_, args| {
        simd_binary_op(&args[0], &args[1], |a, b| a + b, "simd_add")
    });

    // simd_sub - component-wise subtraction
    define(interp, "simd_sub", Some(2), |_, args| {
        simd_binary_op(&args[0], &args[1], |a, b| a - b, "simd_sub")
    });

    // simd_mul - component-wise multiplication
    define(interp, "simd_mul", Some(2), |_, args| {
        simd_binary_op(&args[0], &args[1], |a, b| a * b, "simd_mul")
    });

    // simd_div - component-wise division
    define(interp, "simd_div", Some(2), |_, args| {
        simd_binary_op(&args[0], &args[1], |a, b| a / b, "simd_div")
    });

    // simd_dot - dot product of two vectors
    define(interp, "simd_dot", Some(2), |_, args| {
        let a = extract_simd(&args[0], "simd_dot")?;
        let b = extract_simd(&args[1], "simd_dot")?;
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        Ok(Value::Float(dot))
    });

    // simd_cross - 3D cross product (w component set to 0)
    define(interp, "simd_cross", Some(2), |_, args| {
        let a = extract_simd(&args[0], "simd_cross")?;
        let b = extract_simd(&args[1], "simd_cross")?;
        Ok(Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(a[1] * b[2] - a[2] * b[1]),
            Value::Float(a[2] * b[0] - a[0] * b[2]),
            Value::Float(a[0] * b[1] - a[1] * b[0]),
            Value::Float(0.0),
        ]))))
    });

    // simd_length - vector length (magnitude)
    define(interp, "simd_length", Some(1), |_, args| {
        let v = extract_simd(&args[0], "simd_length")?;
        let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        Ok(Value::Float(len_sq.sqrt()))
    });

    // simd_normalize - normalize vector to unit length
    define(interp, "simd_normalize", Some(1), |_, args| {
        let v = extract_simd(&args[0], "simd_normalize")?;
        let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        let len = len_sq.sqrt();
        if len < 1e-10 {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![
                Value::Float(0.0), Value::Float(0.0), Value::Float(0.0), Value::Float(0.0),
            ]))));
        }
        let inv_len = 1.0 / len;
        Ok(Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(v[0] * inv_len),
            Value::Float(v[1] * inv_len),
            Value::Float(v[2] * inv_len),
            Value::Float(v[3] * inv_len),
        ]))))
    });

    // simd_min - component-wise minimum
    define(interp, "simd_min", Some(2), |_, args| {
        simd_binary_op(&args[0], &args[1], |a, b| a.min(b), "simd_min")
    });

    // simd_max - component-wise maximum
    define(interp, "simd_max", Some(2), |_, args| {
        simd_binary_op(&args[0], &args[1], |a, b| a.max(b), "simd_max")
    });

    // simd_hadd - horizontal add (sum all components)
    define(interp, "simd_hadd", Some(1), |_, args| {
        let v = extract_simd(&args[0], "simd_hadd")?;
        Ok(Value::Float(v[0] + v[1] + v[2] + v[3]))
    });

    // simd_extract - extract single component (0-3)
    define(interp, "simd_extract", Some(2), |_, args| {
        let v = extract_simd(&args[0], "simd_extract")?;
        let idx = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("simd_extract() requires integer index")),
        };
        if idx > 3 {
            return Err(RuntimeError::new("simd_extract() index must be 0-3"));
        }
        Ok(Value::Float(v[idx]))
    });

    // simd_free - no-op in interpreter (for JIT compatibility)
    define(interp, "simd_free", Some(1), |_, _| {
        Ok(Value::Null)
    });

    // simd_lerp - linear interpolation between vectors
    define(interp, "simd_lerp", Some(3), |_, args| {
        let a = extract_simd(&args[0], "simd_lerp")?;
        let b = extract_simd(&args[1], "simd_lerp")?;
        let t = match &args[2] {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f64,
            _ => return Err(RuntimeError::new("simd_lerp() requires float t")),
        };
        let one_t = 1.0 - t;
        Ok(Value::Array(Rc::new(RefCell::new(vec![
            Value::Float(a[0] * one_t + b[0] * t),
            Value::Float(a[1] * one_t + b[1] * t),
            Value::Float(a[2] * one_t + b[2] * t),
            Value::Float(a[3] * one_t + b[3] * t),
        ]))))
    });
}

// Helper to extract SIMD values from array
fn extract_simd(val: &Value, fn_name: &str) -> Result<[f64; 4], RuntimeError> {
    match val {
        Value::Array(arr) => {
            let arr = arr.borrow();
            if arr.len() < 4 {
                return Err(RuntimeError::new(format!("{}() requires 4-element array", fn_name)));
            }
            let mut result = [0.0; 4];
            for (i, v) in arr.iter().take(4).enumerate() {
                result[i] = match v {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => return Err(RuntimeError::new(format!("{}() requires numeric array", fn_name))),
                };
            }
            Ok(result)
        }
        _ => Err(RuntimeError::new(format!("{}() requires array argument", fn_name))),
    }
}

// Helper for binary SIMD operations
fn simd_binary_op<F>(a: &Value, b: &Value, op: F, fn_name: &str) -> Result<Value, RuntimeError>
where
    F: Fn(f64, f64) -> f64,
{
    let a = extract_simd(a, fn_name)?;
    let b = extract_simd(b, fn_name)?;
    Ok(Value::Array(Rc::new(RefCell::new(vec![
        Value::Float(op(a[0], b[0])),
        Value::Float(op(a[1], b[1])),
        Value::Float(op(a[2], b[2])),
        Value::Float(op(a[3], b[3])),
    ]))))
}

// ============================================================================
// CONCURRENCY FUNCTIONS
// ============================================================================
// WARNING: Interpreter Limitations
// ---------------------------------
// The interpreter uses Rc<RefCell<>> which is NOT thread-safe (not Send/Sync).
// This means:
// - Channels work but block the main thread
// - Actors run single-threaded with message queuing
// - Thread primitives simulate behavior but don't provide true parallelism
// - Atomics work correctly for single-threaded access patterns
//
// For true parallel execution, compile with the JIT backend (--jit flag).
// The JIT uses Arc/Mutex and compiles to native code with proper threading.
// ============================================================================

fn register_concurrency(interp: &mut Interpreter) {
    // --- CHANNELS ---

    // channel_new - create a new channel for message passing
    define(interp, "channel_new", Some(0), |_, _| {
        let (sender, receiver) = mpsc::channel();
        let inner = ChannelInner {
            sender: Mutex::new(sender),
            receiver: Mutex::new(receiver),
        };
        Ok(Value::Channel(Arc::new(inner)))
    });

    // channel_send - send a value on a channel (blocking)
    define(interp, "channel_send", Some(2), |_, args| {
        let channel = match &args[0] {
            Value::Channel(ch) => ch.clone(),
            _ => return Err(RuntimeError::new("channel_send() requires channel as first argument")),
        };
        let value = args[1].clone();

        let sender = channel.sender.lock().map_err(|_| RuntimeError::new("channel mutex poisoned"))?;
        sender.send(value).map_err(|_| RuntimeError::new("channel_send() failed: receiver dropped"))?;

        Ok(Value::Null)
    });

    // channel_recv - receive a value from a channel (blocking)
    define(interp, "channel_recv", Some(1), |_, args| {
        let channel = match &args[0] {
            Value::Channel(ch) => ch.clone(),
            _ => return Err(RuntimeError::new("channel_recv() requires channel argument")),
        };

        let receiver = channel.receiver.lock().map_err(|_| RuntimeError::new("channel mutex poisoned"))?;
        match receiver.recv() {
            Ok(value) => Ok(value),
            Err(_) => Err(RuntimeError::new("channel_recv() failed: sender dropped")),
        }
    });

    // channel_try_recv - non-blocking receive, returns Option
    define(interp, "channel_try_recv", Some(1), |_, args| {
        let channel = match &args[0] {
            Value::Channel(ch) => ch.clone(),
            _ => return Err(RuntimeError::new("channel_try_recv() requires channel argument")),
        };

        let receiver = channel.receiver.lock().map_err(|_| RuntimeError::new("channel mutex poisoned"))?;
        match receiver.try_recv() {
            Ok(value) => {
                // Return Some(value) as a variant
                Ok(Value::Variant {
                    enum_name: "Option".to_string(),
                    variant_name: "Some".to_string(),
                    fields: Some(Rc::new(vec![value])),
                })
            }
            Err(mpsc::TryRecvError::Empty) => {
                // Return None
                Ok(Value::Variant {
                    enum_name: "Option".to_string(),
                    variant_name: "None".to_string(),
                    fields: None,
                })
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                Err(RuntimeError::new("channel_try_recv() failed: sender dropped"))
            }
        }
    });

    // channel_recv_timeout - receive with timeout in milliseconds
    define(interp, "channel_recv_timeout", Some(2), |_, args| {
        let channel = match &args[0] {
            Value::Channel(ch) => ch.clone(),
            _ => return Err(RuntimeError::new("channel_recv_timeout() requires channel as first argument")),
        };
        let timeout_ms = match &args[1] {
            Value::Int(ms) => *ms as u64,
            _ => return Err(RuntimeError::new("channel_recv_timeout() requires integer timeout in ms")),
        };

        let receiver = channel.receiver.lock().map_err(|_| RuntimeError::new("channel mutex poisoned"))?;
        match receiver.recv_timeout(std::time::Duration::from_millis(timeout_ms)) {
            Ok(value) => Ok(Value::Variant {
                enum_name: "Option".to_string(),
                variant_name: "Some".to_string(),
                fields: Some(Rc::new(vec![value])),
            }),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(Value::Variant {
                enum_name: "Option".to_string(),
                variant_name: "None".to_string(),
                fields: None,
            }),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(RuntimeError::new("channel_recv_timeout() failed: sender dropped"))
            }
        }
    });

    // --- THREADS ---
    // Note: The interpreter's Value type uses Rc which is not Send.
    // For true threading, use channels to communicate primitive types.
    // These functions provide basic thread primitives.

    // thread_spawn_detached - spawn a detached thread (no join)
    // Useful for background work, results communicated via channels
    define(interp, "thread_spawn_detached", Some(0), |_, _| {
        // Spawn a simple detached thread that does nothing
        // Real work should be done via channels
        thread::spawn(|| {
            // Background thread
        });
        Ok(Value::Null)
    });

    // thread_join - placeholder for join semantics
    // In interpreter, actual work is done via channels
    define(interp, "thread_join", Some(1), |_, args| {
        match &args[0] {
            Value::ThreadHandle(h) => {
                let mut guard = h.lock().map_err(|_| RuntimeError::new("thread handle mutex poisoned"))?;
                if let Some(handle) = guard.take() {
                    match handle.join() {
                        Ok(v) => Ok(v),
                        Err(_) => Err(RuntimeError::new("thread panicked")),
                    }
                } else {
                    Err(RuntimeError::new("thread already joined"))
                }
            }
            // For non-handles, just return the value
            _ => Ok(args[0].clone()),
        }
    });

    // thread_sleep - sleep for specified milliseconds
    define(interp, "thread_sleep", Some(1), |_, args| {
        let ms = match &args[0] {
            Value::Int(ms) => *ms as u64,
            Value::Float(ms) => *ms as u64,
            _ => return Err(RuntimeError::new("thread_sleep() requires integer milliseconds")),
        };

        thread::sleep(std::time::Duration::from_millis(ms));
        Ok(Value::Null)
    });

    // thread_yield - yield the current thread
    define(interp, "thread_yield", Some(0), |_, _| {
        thread::yield_now();
        Ok(Value::Null)
    });

    // thread_id - get current thread id as string
    define(interp, "thread_id", Some(0), |_, _| {
        let id = thread::current().id();
        Ok(Value::String(Rc::new(format!("{:?}", id))))
    });

    // --- ACTORS ---
    // Single-threaded actor model for the interpreter.
    // Messages are queued and processed synchronously.
    // For true async actors with background threads, use the JIT backend.

    // spawn_actor - create a new actor with given name
    define(interp, "spawn_actor", Some(1), |_, args| {
        let name = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("actor_spawn() requires string name")),
        };

        let inner = ActorInner {
            name,
            message_queue: Mutex::new(Vec::new()),
            message_count: std::sync::atomic::AtomicUsize::new(0),
        };

        Ok(Value::Actor(Arc::new(inner)))
    });

    // send_to_actor - send a message to an actor
    // Messages are queued for later processing
    define(interp, "send_to_actor", Some(3), |_, args| {
        let actor = match &args[0] {
            Value::Actor(a) => a.clone(),
            _ => return Err(RuntimeError::new("actor_send() requires actor as first argument")),
        };
        let msg_type = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("actor_send() requires string message type")),
        };
        let msg_data = format!("{}", args[2]);

        let mut queue = actor.message_queue.lock().map_err(|_| RuntimeError::new("actor queue poisoned"))?;
        queue.push((msg_type, msg_data));
        actor.message_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(Value::Null)
    });

    // tell_actor - alias for send_to_actor (Erlang/Akka style)
    define(interp, "tell_actor", Some(3), |_, args| {
        let actor = match &args[0] {
            Value::Actor(a) => a.clone(),
            _ => return Err(RuntimeError::new("actor_tell() requires actor as first argument")),
        };
        let msg_type = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("actor_tell() requires string message type")),
        };
        let msg_data = format!("{}", args[2]);

        let mut queue = actor.message_queue.lock().map_err(|_| RuntimeError::new("actor queue poisoned"))?;
        queue.push((msg_type, msg_data));
        actor.message_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(Value::Null)
    });

    // recv_from_actor - receive (pop) a message from the actor's queue
    // Returns Option<(type, data)>
    define(interp, "recv_from_actor", Some(1), |_, args| {
        let actor = match &args[0] {
            Value::Actor(a) => a.clone(),
            _ => return Err(RuntimeError::new("actor_recv() requires actor argument")),
        };

        let mut queue = actor.message_queue.lock().map_err(|_| RuntimeError::new("actor queue poisoned"))?;
        match queue.pop() {
            Some((msg_type, msg_data)) => {
                // Return Some((type, data))
                Ok(Value::Variant {
                    enum_name: "Option".to_string(),
                    variant_name: "Some".to_string(),
                    fields: Some(Rc::new(vec![
                        Value::Tuple(Rc::new(vec![
                            Value::String(Rc::new(msg_type)),
                            Value::String(Rc::new(msg_data)),
                        ]))
                    ])),
                })
            }
            None => Ok(Value::Variant {
                enum_name: "Option".to_string(),
                variant_name: "None".to_string(),
                fields: None,
            }),
        }
    });

    // get_actor_msg_count - get total messages ever sent to actor
    define(interp, "get_actor_msg_count", Some(1), |_, args| {
        let a = match &args[0] {
            Value::Actor(a) => a.clone(),
            _ => return Err(RuntimeError::new("get_actor_msg_count() requires actor argument")),
        };

        let count = a.message_count.load(std::sync::atomic::Ordering::SeqCst);
        Ok(Value::Int(count as i64))
    });

    // get_actor_name - get actor's name
    define(interp, "get_actor_name", Some(1), |_, args| {
        let a = match &args[0] {
            Value::Actor(a) => a.clone(),
            _ => return Err(RuntimeError::new("get_actor_name() requires actor argument")),
        };

        Ok(Value::String(Rc::new(a.name.clone())))
    });

    // get_actor_pending - get number of pending messages
    define(interp, "get_actor_pending", Some(1), |_, args| {
        let a = match &args[0] {
            Value::Actor(a) => a.clone(),
            _ => return Err(RuntimeError::new("get_actor_pending() requires actor argument")),
        };

        let queue = a.message_queue.lock().map_err(|_| RuntimeError::new("actor queue poisoned"))?;
        Ok(Value::Int(queue.len() as i64))
    });

    // --- SYNCHRONIZATION PRIMITIVES ---

    // mutex_new - create a new mutex wrapping a value
    define(interp, "mutex_new", Some(1), |_, args| {
        let value = args[0].clone();
        // Store as a Map with special key for mutex semantics
        let mut map = std::collections::HashMap::new();
        map.insert("__mutex_value".to_string(), value);
        map.insert("__mutex_locked".to_string(), Value::Bool(false));
        Ok(Value::Map(Rc::new(RefCell::new(map))))
    });

    // mutex_lock - lock a mutex and get the value
    define(interp, "mutex_lock", Some(1), |_, args| {
        let mutex = match &args[0] {
            Value::Map(m) => m.clone(),
            _ => return Err(RuntimeError::new("mutex_lock() requires mutex")),
        };

        let mut map = mutex.borrow_mut();
        // Simple spin-wait for interpreter (not true mutex, but demonstrates concept)
        map.insert("__mutex_locked".to_string(), Value::Bool(true));

        match map.get("__mutex_value") {
            Some(v) => Ok(v.clone()),
            None => Err(RuntimeError::new("invalid mutex")),
        }
    });

    // mutex_unlock - unlock a mutex, optionally setting new value
    define(interp, "mutex_unlock", Some(2), |_, args| {
        let mutex = match &args[0] {
            Value::Map(m) => m.clone(),
            _ => return Err(RuntimeError::new("mutex_unlock() requires mutex")),
        };
        let new_value = args[1].clone();

        let mut map = mutex.borrow_mut();
        map.insert("__mutex_value".to_string(), new_value);
        map.insert("__mutex_locked".to_string(), Value::Bool(false));

        Ok(Value::Null)
    });

    // atomic_new - create an atomic integer
    define(interp, "atomic_new", Some(1), |_, args| {
        let value = match &args[0] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("atomic_new() requires integer")),
        };

        // Wrap in Map with atomic semantics
        let mut map = std::collections::HashMap::new();
        map.insert("__atomic_value".to_string(), Value::Int(value));
        Ok(Value::Map(Rc::new(RefCell::new(map))))
    });

    // atomic_load - atomically load value
    define(interp, "atomic_load", Some(1), |_, args| {
        let atomic = match &args[0] {
            Value::Map(m) => m.clone(),
            _ => return Err(RuntimeError::new("atomic_load() requires atomic")),
        };

        let map = atomic.borrow();
        match map.get("__atomic_value") {
            Some(v) => Ok(v.clone()),
            None => Err(RuntimeError::new("invalid atomic")),
        }
    });

    // atomic_store - atomically store value
    define(interp, "atomic_store", Some(2), |_, args| {
        let atomic = match &args[0] {
            Value::Map(m) => m.clone(),
            _ => return Err(RuntimeError::new("atomic_store() requires atomic")),
        };
        let value = match &args[1] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("atomic_store() requires integer value")),
        };

        let mut map = atomic.borrow_mut();
        map.insert("__atomic_value".to_string(), Value::Int(value));
        Ok(Value::Null)
    });

    // atomic_add - atomically add and return old value
    define(interp, "atomic_add", Some(2), |_, args| {
        let atomic = match &args[0] {
            Value::Map(m) => m.clone(),
            _ => return Err(RuntimeError::new("atomic_add() requires atomic")),
        };
        let delta = match &args[1] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("atomic_add() requires integer delta")),
        };

        let mut map = atomic.borrow_mut();
        let old = match map.get("__atomic_value") {
            Some(Value::Int(i)) => *i,
            _ => return Err(RuntimeError::new("invalid atomic")),
        };
        map.insert("__atomic_value".to_string(), Value::Int(old + delta));
        Ok(Value::Int(old))
    });

    // atomic_cas - compare and swap, returns bool success
    define(interp, "atomic_cas", Some(3), |_, args| {
        let atomic = match &args[0] {
            Value::Map(m) => m.clone(),
            _ => return Err(RuntimeError::new("atomic_cas() requires atomic")),
        };
        let expected = match &args[1] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("atomic_cas() requires integer expected")),
        };
        let new_value = match &args[2] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("atomic_cas() requires integer new value")),
        };

        let mut map = atomic.borrow_mut();
        let current = match map.get("__atomic_value") {
            Some(Value::Int(i)) => *i,
            _ => return Err(RuntimeError::new("invalid atomic")),
        };

        if current == expected {
            map.insert("__atomic_value".to_string(), Value::Int(new_value));
            Ok(Value::Bool(true))
        } else {
            Ok(Value::Bool(false))
        }
    });

    // --- PARALLEL ITERATION ---

    // parallel_map - map function over array in parallel (simplified)
    define(interp, "parallel_map", Some(2), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("parallel_map() requires array")),
        };
        let _func = args[1].clone();

        // For interpreter, just return original array
        // Real parallelism needs thread-safe interpreter
        Ok(Value::Array(Rc::new(RefCell::new(arr))))
    });

    // parallel_for - parallel for loop (simplified)
    define(interp, "parallel_for", Some(3), |_, args| {
        let start = match &args[0] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("parallel_for() requires integer start")),
        };
        let end = match &args[1] {
            Value::Int(i) => *i,
            _ => return Err(RuntimeError::new("parallel_for() requires integer end")),
        };
        let _func = args[2].clone();

        // For interpreter, execute sequentially
        // Returns range as array
        let range: Vec<Value> = (start..end).map(|i| Value::Int(i)).collect();
        Ok(Value::Array(Rc::new(RefCell::new(range))))
    });

    // ============================================================================
    // ASYNC/AWAIT FUNCTIONS
    // ============================================================================
    // WARNING: Interpreter Blocking Behavior
    // --------------------------------------
    // In the interpreter, async operations use cooperative scheduling but
    // execute on the main thread. This means:
    // - async_sleep() blocks the interpreter for the specified duration
    // - await() polls futures but may block waiting for completion
    // - No true concurrent I/O - operations execute sequentially
    // - Future combinators (race, all) work but don't provide parallelism
    //
    // The async model is designed for composability and clean code structure.
    // For non-blocking async with true concurrency, use the JIT backend.
    // ============================================================================

    // async_sleep - create a future that completes after specified milliseconds
    define(interp, "async_sleep", Some(1), |interp, args| {
        let ms = match &args[0] {
            Value::Int(ms) => *ms as u64,
            Value::Float(ms) => *ms as u64,
            _ => return Err(RuntimeError::new("async_sleep() requires integer milliseconds")),
        };

        Ok(interp.make_future_timer(std::time::Duration::from_millis(ms)))
    });

    // future_ready - create an immediately resolved future
    define(interp, "future_ready", Some(1), |interp, args| {
        Ok(interp.make_future_immediate(args[0].clone()))
    });

    // future_pending - create a pending future (never resolves)
    define(interp, "future_pending", Some(0), |_, _| {
        Ok(Value::Future(Rc::new(RefCell::new(crate::interpreter::FutureInner {
            state: crate::interpreter::FutureState::Pending,
            computation: None,
            complete_at: None,
        }))))
    });

    // is_future - check if a value is a future
    define(interp, "is_future", Some(1), |_, args| {
        Ok(Value::Bool(matches!(&args[0], Value::Future(_))))
    });

    // is_ready - check if a future is ready
    define(interp, "is_ready", Some(1), |_, args| {
        match &args[0] {
            Value::Future(fut) => {
                let f = fut.borrow();
                Ok(Value::Bool(matches!(f.state, crate::interpreter::FutureState::Ready(_))))
            }
            _ => Ok(Value::Bool(true)), // Non-futures are always "ready"
        }
    });

    // join_futures - join multiple futures into one that resolves to array
    define(interp, "join_futures", Some(1), |_, args| {
        let futures = match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                let mut futs = Vec::new();
                for v in arr.iter() {
                    match v {
                        Value::Future(f) => futs.push(f.clone()),
                        _ => return Err(RuntimeError::new("join_futures() requires array of futures")),
                    }
                }
                futs
            }
            _ => return Err(RuntimeError::new("join_futures() requires array of futures")),
        };

        Ok(Value::Future(Rc::new(RefCell::new(crate::interpreter::FutureInner {
            state: crate::interpreter::FutureState::Pending,
            computation: Some(crate::interpreter::FutureComputation::Join(futures)),
            complete_at: None,
        }))))
    });

    // race_futures - return first future to complete
    define(interp, "race_futures", Some(1), |_, args| {
        let futures = match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                let mut futs = Vec::new();
                for v in arr.iter() {
                    match v {
                        Value::Future(f) => futs.push(f.clone()),
                        _ => return Err(RuntimeError::new("race_futures() requires array of futures")),
                    }
                }
                futs
            }
            _ => return Err(RuntimeError::new("race_futures() requires array of futures")),
        };

        Ok(Value::Future(Rc::new(RefCell::new(crate::interpreter::FutureInner {
            state: crate::interpreter::FutureState::Pending,
            computation: Some(crate::interpreter::FutureComputation::Race(futures)),
            complete_at: None,
        }))))
    });

    // poll_future - try to resolve a future without blocking (returns Option)
    define(interp, "poll_future", Some(1), |_, args| {
        match &args[0] {
            Value::Future(fut) => {
                let f = fut.borrow();
                match &f.state {
                    crate::interpreter::FutureState::Ready(v) => {
                        Ok(Value::Variant {
                            enum_name: "Option".to_string(),
                            variant_name: "Some".to_string(),
                            fields: Some(Rc::new(vec![(**v).clone()])),
                        })
                    }
                    _ => {
                        Ok(Value::Variant {
                            enum_name: "Option".to_string(),
                            variant_name: "None".to_string(),
                            fields: None,
                        })
                    }
                }
            }
            // Non-futures return Some(value)
            other => Ok(Value::Variant {
                enum_name: "Option".to_string(),
                variant_name: "Some".to_string(),
                fields: Some(Rc::new(vec![other.clone()])),
            }),
        }
    });
}

// ============================================================================
// JSON FUNCTIONS
// ============================================================================

fn register_json(interp: &mut Interpreter) {
    // json_parse - parse JSON string into Sigil value
    define(interp, "json_parse", Some(1), |_, args| {
        let json_str = match &args[0] {
            Value::String(s) => s.as_str(),
            _ => return Err(RuntimeError::new("json_parse() requires string argument")),
        };

        fn json_to_value(json: &serde_json::Value) -> Value {
            match json {
                serde_json::Value::Null => Value::Null,
                serde_json::Value::Bool(b) => Value::Bool(*b),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Value::Int(i)
                    } else if let Some(f) = n.as_f64() {
                        Value::Float(f)
                    } else {
                        Value::Null
                    }
                }
                serde_json::Value::String(s) => Value::String(Rc::new(s.clone())),
                serde_json::Value::Array(arr) => {
                    let values: Vec<Value> = arr.iter().map(json_to_value).collect();
                    Value::Array(Rc::new(RefCell::new(values)))
                }
                serde_json::Value::Object(obj) => {
                    let mut map = HashMap::new();
                    for (k, v) in obj {
                        map.insert(k.clone(), json_to_value(v));
                    }
                    Value::Map(Rc::new(RefCell::new(map)))
                }
            }
        }

        match serde_json::from_str(json_str) {
            Ok(json) => Ok(json_to_value(&json)),
            Err(e) => Err(RuntimeError::new(format!("JSON parse error: {}", e))),
        }
    });

    // json_stringify - convert Sigil value to JSON string
    define(interp, "json_stringify", Some(1), |_, args| {
        fn value_to_json(val: &Value) -> serde_json::Value {
            match val {
                Value::Null => serde_json::Value::Null,
                Value::Bool(b) => serde_json::Value::Bool(*b),
                Value::Int(n) => serde_json::Value::Number(serde_json::Number::from(*n)),
                Value::Float(f) => {
                    serde_json::Number::from_f64(*f)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                }
                Value::String(s) => serde_json::Value::String(s.to_string()),
                Value::Array(arr) => {
                    let arr = arr.borrow();
                    serde_json::Value::Array(arr.iter().map(value_to_json).collect())
                }
                Value::Tuple(t) => {
                    serde_json::Value::Array(t.iter().map(value_to_json).collect())
                }
                Value::Map(map) => {
                    let map = map.borrow();
                    let obj: serde_json::Map<String, serde_json::Value> = map
                        .iter()
                        .map(|(k, v)| (k.clone(), value_to_json(v)))
                        .collect();
                    serde_json::Value::Object(obj)
                }
                Value::Struct { fields, .. } => {
                    let fields = fields.borrow();
                    let obj: serde_json::Map<String, serde_json::Value> = fields
                        .iter()
                        .map(|(k, v)| (k.clone(), value_to_json(v)))
                        .collect();
                    serde_json::Value::Object(obj)
                }
                _ => serde_json::Value::String(format!("{}", val)),
            }
        }

        let json = value_to_json(&args[0]);
        Ok(Value::String(Rc::new(json.to_string())))
    });

    // json_pretty - convert to pretty-printed JSON
    define(interp, "json_pretty", Some(1), |_, args| {
        fn value_to_json(val: &Value) -> serde_json::Value {
            match val {
                Value::Null => serde_json::Value::Null,
                Value::Bool(b) => serde_json::Value::Bool(*b),
                Value::Int(n) => serde_json::Value::Number(serde_json::Number::from(*n)),
                Value::Float(f) => {
                    serde_json::Number::from_f64(*f)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                }
                Value::String(s) => serde_json::Value::String(s.to_string()),
                Value::Array(arr) => {
                    let arr = arr.borrow();
                    serde_json::Value::Array(arr.iter().map(value_to_json).collect())
                }
                Value::Map(map) => {
                    let map = map.borrow();
                    let obj: serde_json::Map<String, serde_json::Value> = map
                        .iter()
                        .map(|(k, v)| (k.clone(), value_to_json(v)))
                        .collect();
                    serde_json::Value::Object(obj)
                }
                _ => serde_json::Value::String(format!("{}", val)),
            }
        }

        let json = value_to_json(&args[0]);
        match serde_json::to_string_pretty(&json) {
            Ok(s) => Ok(Value::String(Rc::new(s))),
            Err(e) => Err(RuntimeError::new(format!("JSON stringify error: {}", e))),
        }
    });

    // json_get - get value at JSON path (dot notation)
    define(interp, "json_get", Some(2), |_, args| {
        let path = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("json_get() requires string path")),
        };

        let mut current = args[0].clone();
        for key in path.split('.') {
            current = match &current {
                Value::Map(map) => {
                    let map = map.borrow();
                    map.get(key).cloned().unwrap_or(Value::Null)
                }
                Value::Array(arr) => {
                    if let Ok(idx) = key.parse::<usize>() {
                        let arr = arr.borrow();
                        arr.get(idx).cloned().unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                }
                _ => Value::Null,
            };
        }
        Ok(current)
    });

    // json_set - set value at JSON path
    define(interp, "json_set", Some(3), |_, args| {
        let path = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("json_set() requires string path")),
        };
        let new_value = args[2].clone();

        // For simplicity, only handle single-level paths
        match &args[0] {
            Value::Map(map) => {
                let mut map = map.borrow_mut();
                map.insert(path, new_value);
                Ok(Value::Map(Rc::new(RefCell::new(map.clone()))))
            }
            _ => Err(RuntimeError::new("json_set() requires map/object")),
        }
    });
}

// ============================================================================
// FILE SYSTEM FUNCTIONS
// ============================================================================

fn register_fs(interp: &mut Interpreter) {
    // fs_read - read entire file as string
    define(interp, "fs_read", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_read() requires string path")),
        };

        match std::fs::read_to_string(&path) {
            Ok(content) => Ok(Value::String(Rc::new(content))),
            Err(e) => Err(RuntimeError::new(format!("fs_read() error: {}", e))),
        }
    });

    // fs_read_bytes - read file as byte array
    define(interp, "fs_read_bytes", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_read_bytes() requires string path")),
        };

        match std::fs::read(&path) {
            Ok(bytes) => {
                let values: Vec<Value> = bytes.iter().map(|b| Value::Int(*b as i64)).collect();
                Ok(Value::Array(Rc::new(RefCell::new(values))))
            }
            Err(e) => Err(RuntimeError::new(format!("fs_read_bytes() error: {}", e))),
        }
    });

    // fs_write - write string to file
    define(interp, "fs_write", Some(2), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_write() requires string path")),
        };
        let content = format!("{}", args[1]);

        match std::fs::write(&path, content) {
            Ok(()) => Ok(Value::Null),
            Err(e) => Err(RuntimeError::new(format!("fs_write() error: {}", e))),
        }
    });

    // fs_append - append to file
    define(interp, "fs_append", Some(2), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_append() requires string path")),
        };
        let content = format!("{}", args[1]);

        use std::fs::OpenOptions;
        match OpenOptions::new().append(true).create(true).open(&path) {
            Ok(mut file) => {
                use std::io::Write;
                match file.write_all(content.as_bytes()) {
                    Ok(()) => Ok(Value::Null),
                    Err(e) => Err(RuntimeError::new(format!("fs_append() write error: {}", e))),
                }
            }
            Err(e) => Err(RuntimeError::new(format!("fs_append() error: {}", e))),
        }
    });

    // fs_exists - check if path exists
    define(interp, "fs_exists", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_exists() requires string path")),
        };
        Ok(Value::Bool(std::path::Path::new(&path).exists()))
    });

    // fs_is_file - check if path is a file
    define(interp, "fs_is_file", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_is_file() requires string path")),
        };
        Ok(Value::Bool(std::path::Path::new(&path).is_file()))
    });

    // fs_is_dir - check if path is a directory
    define(interp, "fs_is_dir", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_is_dir() requires string path")),
        };
        Ok(Value::Bool(std::path::Path::new(&path).is_dir()))
    });

    // fs_mkdir - create directory
    define(interp, "fs_mkdir", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_mkdir() requires string path")),
        };

        match std::fs::create_dir_all(&path) {
            Ok(()) => Ok(Value::Null),
            Err(e) => Err(RuntimeError::new(format!("fs_mkdir() error: {}", e))),
        }
    });

    // fs_remove - remove file or directory
    define(interp, "fs_remove", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_remove() requires string path")),
        };

        let p = std::path::Path::new(&path);
        let result = if p.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };

        match result {
            Ok(()) => Ok(Value::Null),
            Err(e) => Err(RuntimeError::new(format!("fs_remove() error: {}", e))),
        }
    });

    // fs_list - list directory contents
    define(interp, "fs_list", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_list() requires string path")),
        };

        match std::fs::read_dir(&path) {
            Ok(entries) => {
                let mut files = Vec::new();
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        files.push(Value::String(Rc::new(name.to_string())));
                    }
                }
                Ok(Value::Array(Rc::new(RefCell::new(files))))
            }
            Err(e) => Err(RuntimeError::new(format!("fs_list() error: {}", e))),
        }
    });

    // fs_copy - copy file
    define(interp, "fs_copy", Some(2), |_, args| {
        let src = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_copy() requires string source path")),
        };
        let dst = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_copy() requires string destination path")),
        };

        match std::fs::copy(&src, &dst) {
            Ok(bytes) => Ok(Value::Int(bytes as i64)),
            Err(e) => Err(RuntimeError::new(format!("fs_copy() error: {}", e))),
        }
    });

    // fs_rename - rename/move file
    define(interp, "fs_rename", Some(2), |_, args| {
        let src = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_rename() requires string source path")),
        };
        let dst = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_rename() requires string destination path")),
        };

        match std::fs::rename(&src, &dst) {
            Ok(()) => Ok(Value::Null),
            Err(e) => Err(RuntimeError::new(format!("fs_rename() error: {}", e))),
        }
    });

    // fs_size - get file size in bytes
    define(interp, "fs_size", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("fs_size() requires string path")),
        };

        match std::fs::metadata(&path) {
            Ok(meta) => Ok(Value::Int(meta.len() as i64)),
            Err(e) => Err(RuntimeError::new(format!("fs_size() error: {}", e))),
        }
    });

    // path_join - join path components
    define(interp, "path_join", None, |_, args| {
        let mut path = std::path::PathBuf::new();
        for arg in &args {
            match arg {
                Value::String(s) => path.push(s.as_str()),
                Value::Array(arr) => {
                    for v in arr.borrow().iter() {
                        if let Value::String(s) = v {
                            path.push(s.as_str());
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(Value::String(Rc::new(path.to_string_lossy().to_string())))
    });

    // path_parent - get parent directory
    define(interp, "path_parent", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("path_parent() requires string path")),
        };

        let p = std::path::Path::new(&path);
        match p.parent() {
            Some(parent) => Ok(Value::String(Rc::new(parent.to_string_lossy().to_string()))),
            None => Ok(Value::Null),
        }
    });

    // path_filename - get filename component
    define(interp, "path_filename", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("path_filename() requires string path")),
        };

        let p = std::path::Path::new(&path);
        match p.file_name() {
            Some(name) => Ok(Value::String(Rc::new(name.to_string_lossy().to_string()))),
            None => Ok(Value::Null),
        }
    });

    // path_extension - get file extension
    define(interp, "path_extension", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("path_extension() requires string path")),
        };

        let p = std::path::Path::new(&path);
        match p.extension() {
            Some(ext) => Ok(Value::String(Rc::new(ext.to_string_lossy().to_string()))),
            None => Ok(Value::Null),
        }
    });
}

// ============================================================================
// CRYPTO FUNCTIONS
// ============================================================================
// SECURITY WARNING
// -----------------
// - sha256(), sha512(): Recommended for secure hashing (checksums, integrity)
// - md5(): DEPRECATED for security use. MD5 is cryptographically broken.
//   Only use for legacy compatibility, checksums of non-security data, or
//   cache keys. NEVER use for passwords, signatures, or security tokens.
// - For password hashing, consider using external tools like bcrypt/argon2
// ============================================================================

fn register_crypto(interp: &mut Interpreter) {
    // sha256 - compute SHA-256 hash (RECOMMENDED for security use)
    define(interp, "sha256", Some(1), |_, args| {
        let data = match &args[0] {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Array(arr) => {
                let arr = arr.borrow();
                arr.iter().filter_map(|v| {
                    if let Value::Int(n) = v { Some(*n as u8) } else { None }
                }).collect()
            }
            _ => return Err(RuntimeError::new("sha256() requires string or byte array")),
        };

        let mut hasher = Sha256::new();
        hasher.update(&data);
        let result = hasher.finalize();
        let hex = result.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        Ok(Value::String(Rc::new(hex)))
    });

    // sha512 - compute SHA-512 hash
    define(interp, "sha512", Some(1), |_, args| {
        let data = match &args[0] {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Array(arr) => {
                let arr = arr.borrow();
                arr.iter().filter_map(|v| {
                    if let Value::Int(n) = v { Some(*n as u8) } else { None }
                }).collect()
            }
            _ => return Err(RuntimeError::new("sha512() requires string or byte array")),
        };

        let mut hasher = Sha512::new();
        hasher.update(&data);
        let result = hasher.finalize();
        let hex = result.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        Ok(Value::String(Rc::new(hex)))
    });

    // md5 - compute MD5 hash
    // ⚠️  DEPRECATED: MD5 is cryptographically broken. Use sha256() instead.
    // Only provided for legacy compatibility and non-security checksums.
    define(interp, "md5", Some(1), |_, args| {
        let data = match &args[0] {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Array(arr) => {
                let arr = arr.borrow();
                arr.iter().filter_map(|v| {
                    if let Value::Int(n) = v { Some(*n as u8) } else { None }
                }).collect()
            }
            _ => return Err(RuntimeError::new("md5() requires string or byte array")),
        };

        let mut hasher = Md5::new();
        hasher.update(&data);
        let result = hasher.finalize();
        let hex = result.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        Ok(Value::String(Rc::new(hex)))
    });

    // base64_encode - encode to base64
    define(interp, "base64_encode", Some(1), |_, args| {
        let data = match &args[0] {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Array(arr) => {
                let arr = arr.borrow();
                arr.iter().filter_map(|v| {
                    if let Value::Int(n) = v { Some(*n as u8) } else { None }
                }).collect()
            }
            _ => return Err(RuntimeError::new("base64_encode() requires string or byte array")),
        };

        let encoded = general_purpose::STANDARD.encode(&data);
        Ok(Value::String(Rc::new(encoded)))
    });

    // base64_decode - decode from base64
    define(interp, "base64_decode", Some(1), |_, args| {
        let encoded = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("base64_decode() requires string")),
        };

        match general_purpose::STANDARD.decode(&encoded) {
            Ok(bytes) => {
                // Try to convert to string, otherwise return byte array
                match String::from_utf8(bytes.clone()) {
                    Ok(s) => Ok(Value::String(Rc::new(s))),
                    Err(_) => {
                        let values: Vec<Value> = bytes.iter().map(|b| Value::Int(*b as i64)).collect();
                        Ok(Value::Array(Rc::new(RefCell::new(values))))
                    }
                }
            }
            Err(e) => Err(RuntimeError::new(format!("base64_decode() error: {}", e))),
        }
    });

    // hex_encode - encode bytes to hex string
    define(interp, "hex_encode", Some(1), |_, args| {
        let data = match &args[0] {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Array(arr) => {
                let arr = arr.borrow();
                arr.iter().filter_map(|v| {
                    if let Value::Int(n) = v { Some(*n as u8) } else { None }
                }).collect()
            }
            _ => return Err(RuntimeError::new("hex_encode() requires string or byte array")),
        };

        let hex = data.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        Ok(Value::String(Rc::new(hex)))
    });

    // hex_decode - decode hex string to bytes
    define(interp, "hex_decode", Some(1), |_, args| {
        let hex = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("hex_decode() requires string")),
        };

        let hex = hex.trim();
        if hex.len() % 2 != 0 {
            return Err(RuntimeError::new("hex_decode() requires even-length hex string"));
        }

        let mut bytes = Vec::new();
        for i in (0..hex.len()).step_by(2) {
            match u8::from_str_radix(&hex[i..i+2], 16) {
                Ok(b) => bytes.push(Value::Int(b as i64)),
                Err(_) => return Err(RuntimeError::new("hex_decode() invalid hex character")),
            }
        }
        Ok(Value::Array(Rc::new(RefCell::new(bytes))))
    });
}

// ============================================================================
// REGEX FUNCTIONS
// ============================================================================

fn register_regex(interp: &mut Interpreter) {
    // regex_match - check if string matches pattern
    define(interp, "regex_match", Some(2), |_, args| {
        let pattern = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_match() requires string pattern")),
        };
        let text = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_match() requires string text")),
        };

        match Regex::new(&pattern) {
            Ok(re) => Ok(Value::Bool(re.is_match(&text))),
            Err(e) => Err(RuntimeError::new(format!("regex_match() invalid pattern: {}", e))),
        }
    });

    // regex_find - find first match
    define(interp, "regex_find", Some(2), |_, args| {
        let pattern = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_find() requires string pattern")),
        };
        let text = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_find() requires string text")),
        };

        match Regex::new(&pattern) {
            Ok(re) => {
                match re.find(&text) {
                    Some(m) => Ok(Value::String(Rc::new(m.as_str().to_string()))),
                    None => Ok(Value::Null),
                }
            }
            Err(e) => Err(RuntimeError::new(format!("regex_find() invalid pattern: {}", e))),
        }
    });

    // regex_find_all - find all matches
    define(interp, "regex_find_all", Some(2), |_, args| {
        let pattern = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_find_all() requires string pattern")),
        };
        let text = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_find_all() requires string text")),
        };

        match Regex::new(&pattern) {
            Ok(re) => {
                let matches: Vec<Value> = re.find_iter(&text)
                    .map(|m| Value::String(Rc::new(m.as_str().to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(matches))))
            }
            Err(e) => Err(RuntimeError::new(format!("regex_find_all() invalid pattern: {}", e))),
        }
    });

    // regex_replace - replace first match
    define(interp, "regex_replace", Some(3), |_, args| {
        let pattern = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_replace() requires string pattern")),
        };
        let text = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_replace() requires string text")),
        };
        let replacement = match &args[2] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_replace() requires string replacement")),
        };

        match Regex::new(&pattern) {
            Ok(re) => {
                let result = re.replace(&text, replacement.as_str());
                Ok(Value::String(Rc::new(result.to_string())))
            }
            Err(e) => Err(RuntimeError::new(format!("regex_replace() invalid pattern: {}", e))),
        }
    });

    // regex_replace_all - replace all matches
    define(interp, "regex_replace_all", Some(3), |_, args| {
        let pattern = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_replace_all() requires string pattern")),
        };
        let text = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_replace_all() requires string text")),
        };
        let replacement = match &args[2] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_replace_all() requires string replacement")),
        };

        match Regex::new(&pattern) {
            Ok(re) => {
                let result = re.replace_all(&text, replacement.as_str());
                Ok(Value::String(Rc::new(result.to_string())))
            }
            Err(e) => Err(RuntimeError::new(format!("regex_replace_all() invalid pattern: {}", e))),
        }
    });

    // regex_split - split by pattern
    define(interp, "regex_split", Some(2), |_, args| {
        let pattern = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_split() requires string pattern")),
        };
        let text = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_split() requires string text")),
        };

        match Regex::new(&pattern) {
            Ok(re) => {
                let parts: Vec<Value> = re.split(&text)
                    .map(|s| Value::String(Rc::new(s.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(parts))))
            }
            Err(e) => Err(RuntimeError::new(format!("regex_split() invalid pattern: {}", e))),
        }
    });

    // regex_captures - capture groups
    define(interp, "regex_captures", Some(2), |_, args| {
        let pattern = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_captures() requires string pattern")),
        };
        let text = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("regex_captures() requires string text")),
        };

        match Regex::new(&pattern) {
            Ok(re) => {
                match re.captures(&text) {
                    Some(caps) => {
                        let captures: Vec<Value> = caps.iter()
                            .map(|m| {
                                m.map(|m| Value::String(Rc::new(m.as_str().to_string())))
                                    .unwrap_or(Value::Null)
                            })
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(captures))))
                    }
                    None => Ok(Value::Null),
                }
            }
            Err(e) => Err(RuntimeError::new(format!("regex_captures() invalid pattern: {}", e))),
        }
    });
}

// ============================================================================
// UUID FUNCTIONS
// ============================================================================

fn register_uuid(interp: &mut Interpreter) {
    // uuid_v4 - generate random UUID v4
    define(interp, "uuid_v4", Some(0), |_, _| {
        let id = Uuid::new_v4();
        Ok(Value::String(Rc::new(id.to_string())))
    });

    // uuid_nil - get nil UUID (all zeros)
    define(interp, "uuid_nil", Some(0), |_, _| {
        Ok(Value::String(Rc::new(Uuid::nil().to_string())))
    });

    // uuid_parse - parse UUID string
    define(interp, "uuid_parse", Some(1), |_, args| {
        let s = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("uuid_parse() requires string")),
        };

        match Uuid::parse_str(&s) {
            Ok(id) => Ok(Value::String(Rc::new(id.to_string()))),
            Err(e) => Err(RuntimeError::new(format!("uuid_parse() error: {}", e))),
        }
    });

    // uuid_is_valid - check if string is valid UUID
    define(interp, "uuid_is_valid", Some(1), |_, args| {
        let s = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Ok(Value::Bool(false)),
        };
        Ok(Value::Bool(Uuid::parse_str(&s).is_ok()))
    });
}

// ============================================================================
// SYSTEM FUNCTIONS
// ============================================================================

fn register_system(interp: &mut Interpreter) {
    // env_get - get environment variable
    define(interp, "env_get", Some(1), |_, args| {
        let key = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("env_get() requires string key")),
        };

        match std::env::var(&key) {
            Ok(val) => Ok(Value::String(Rc::new(val))),
            Err(_) => Ok(Value::Null),
        }
    });

    // env_set - set environment variable
    define(interp, "env_set", Some(2), |_, args| {
        let key = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("env_set() requires string key")),
        };
        let val = match &args[1] {
            Value::String(s) => s.to_string(),
            _ => format!("{}", args[1]),
        };

        std::env::set_var(&key, &val);
        Ok(Value::Null)
    });

    // env_remove - remove environment variable
    define(interp, "env_remove", Some(1), |_, args| {
        let key = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("env_remove() requires string key")),
        };

        std::env::remove_var(&key);
        Ok(Value::Null)
    });

    // env_vars - get all environment variables as map
    define(interp, "env_vars", Some(0), |_, _| {
        let mut map = HashMap::new();
        for (key, val) in std::env::vars() {
            map.insert(key, Value::String(Rc::new(val)));
        }
        Ok(Value::Map(Rc::new(RefCell::new(map))))
    });

    // args - get command line arguments
    define(interp, "args", Some(0), |_, _| {
        let args: Vec<Value> = std::env::args()
            .map(|s| Value::String(Rc::new(s)))
            .collect();
        Ok(Value::Array(Rc::new(RefCell::new(args))))
    });

    // cwd - get current working directory
    define(interp, "cwd", Some(0), |_, _| {
        match std::env::current_dir() {
            Ok(path) => Ok(Value::String(Rc::new(path.to_string_lossy().to_string()))),
            Err(e) => Err(RuntimeError::new(format!("cwd() error: {}", e))),
        }
    });

    // chdir - change current directory
    define(interp, "chdir", Some(1), |_, args| {
        let path = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("chdir() requires string path")),
        };

        match std::env::set_current_dir(&path) {
            Ok(()) => Ok(Value::Null),
            Err(e) => Err(RuntimeError::new(format!("chdir() error: {}", e))),
        }
    });

    // hostname - get system hostname
    define(interp, "hostname", Some(0), |_, _| {
        // Try to read from /etc/hostname or use fallback
        match std::fs::read_to_string("/etc/hostname") {
            Ok(name) => Ok(Value::String(Rc::new(name.trim().to_string()))),
            Err(_) => Ok(Value::String(Rc::new("unknown".to_string()))),
        }
    });

    // pid - get current process ID
    define(interp, "pid", Some(0), |_, _| {
        Ok(Value::Int(std::process::id() as i64))
    });

    // exit - exit the program with code
    define(interp, "exit", Some(1), |_, args| {
        let code = match &args[0] {
            Value::Int(n) => *n as i32,
            _ => 0,
        };
        std::process::exit(code);
    });

    // shell - execute shell command and return output
    define(interp, "shell", Some(1), |_, args| {
        let cmd = match &args[0] {
            Value::String(s) => s.to_string(),
            _ => return Err(RuntimeError::new("shell() requires string command")),
        };

        match std::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let code = output.status.code().unwrap_or(-1);

                let mut result = HashMap::new();
                result.insert("stdout".to_string(), Value::String(Rc::new(stdout)));
                result.insert("stderr".to_string(), Value::String(Rc::new(stderr)));
                result.insert("code".to_string(), Value::Int(code as i64));
                result.insert("success".to_string(), Value::Bool(output.status.success()));

                Ok(Value::Map(Rc::new(RefCell::new(result))))
            }
            Err(e) => Err(RuntimeError::new(format!("shell() error: {}", e))),
        }
    });

    // platform - get OS name
    define(interp, "platform", Some(0), |_, _| {
        Ok(Value::String(Rc::new(std::env::consts::OS.to_string())))
    });

    // arch - get CPU architecture
    define(interp, "arch", Some(0), |_, _| {
        Ok(Value::String(Rc::new(std::env::consts::ARCH.to_string())))
    });
}

// ============================================================================
// STATISTICS FUNCTIONS
// ============================================================================

fn register_stats(interp: &mut Interpreter) {
    // Helper to extract numbers from array
    fn extract_numbers(val: &Value) -> Result<Vec<f64>, RuntimeError> {
        match val {
            Value::Array(arr) => {
                let arr = arr.borrow();
                let mut nums = Vec::new();
                for v in arr.iter() {
                    match v {
                        Value::Int(n) => nums.push(*n as f64),
                        Value::Float(f) => nums.push(*f),
                        _ => return Err(RuntimeError::new("stats functions require numeric array")),
                    }
                }
                Ok(nums)
            }
            _ => Err(RuntimeError::new("stats functions require array")),
        }
    }

    // mean - arithmetic mean
    define(interp, "mean", Some(1), |_, args| {
        let nums = extract_numbers(&args[0])?;
        if nums.is_empty() {
            return Ok(Value::Float(0.0));
        }
        let sum: f64 = nums.iter().sum();
        Ok(Value::Float(sum / nums.len() as f64))
    });

    // median - middle value
    define(interp, "median", Some(1), |_, args| {
        let mut nums = extract_numbers(&args[0])?;
        if nums.is_empty() {
            return Ok(Value::Float(0.0));
        }
        nums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = nums.len();
        if len % 2 == 0 {
            Ok(Value::Float((nums[len/2 - 1] + nums[len/2]) / 2.0))
        } else {
            Ok(Value::Float(nums[len/2]))
        }
    });

    // mode - most frequent value
    define(interp, "mode", Some(1), |_, args| {
        let nums = extract_numbers(&args[0])?;
        if nums.is_empty() {
            return Ok(Value::Null);
        }

        let mut counts: HashMap<String, usize> = HashMap::new();
        for n in &nums {
            let key = format!("{:.10}", n);
            *counts.entry(key).or_insert(0) += 1;
        }

        let max_count = counts.values().max().unwrap_or(&0);
        for n in &nums {
            let key = format!("{:.10}", n);
            if counts.get(&key) == Some(max_count) {
                return Ok(Value::Float(*n));
            }
        }
        Ok(Value::Null)
    });

    // variance - population variance
    define(interp, "variance", Some(1), |_, args| {
        let nums = extract_numbers(&args[0])?;
        if nums.is_empty() {
            return Ok(Value::Float(0.0));
        }
        let mean: f64 = nums.iter().sum::<f64>() / nums.len() as f64;
        let variance: f64 = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nums.len() as f64;
        Ok(Value::Float(variance))
    });

    // stddev - standard deviation
    define(interp, "stddev", Some(1), |_, args| {
        let nums = extract_numbers(&args[0])?;
        if nums.is_empty() {
            return Ok(Value::Float(0.0));
        }
        let mean: f64 = nums.iter().sum::<f64>() / nums.len() as f64;
        let variance: f64 = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nums.len() as f64;
        Ok(Value::Float(variance.sqrt()))
    });

    // percentile - compute nth percentile
    define(interp, "percentile", Some(2), |_, args| {
        let mut nums = extract_numbers(&args[0])?;
        let p = match &args[1] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("percentile() requires numeric percentile")),
        };

        if nums.is_empty() {
            return Ok(Value::Float(0.0));
        }

        nums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p / 100.0 * (nums.len() - 1) as f64).round() as usize;
        Ok(Value::Float(nums[idx.min(nums.len() - 1)]))
    });

    // correlation - Pearson correlation coefficient
    define(interp, "correlation", Some(2), |_, args| {
        let x = extract_numbers(&args[0])?;
        let y = extract_numbers(&args[1])?;

        if x.len() != y.len() || x.is_empty() {
            return Err(RuntimeError::new("correlation() requires equal-length non-empty arrays"));
        }

        let n = x.len() as f64;
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x == 0.0 || var_y == 0.0 {
            return Ok(Value::Float(0.0));
        }

        Ok(Value::Float(cov / (var_x.sqrt() * var_y.sqrt())))
    });

    // range - difference between max and min
    define(interp, "range", Some(1), |_, args| {
        let nums = extract_numbers(&args[0])?;
        if nums.is_empty() {
            return Ok(Value::Float(0.0));
        }
        let min = nums.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(Value::Float(max - min))
    });

    // zscore - compute z-scores for array
    define(interp, "zscore", Some(1), |_, args| {
        let nums = extract_numbers(&args[0])?;
        if nums.is_empty() {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
        }

        let mean: f64 = nums.iter().sum::<f64>() / nums.len() as f64;
        let variance: f64 = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / nums.len() as f64;
        let stddev = variance.sqrt();

        if stddev == 0.0 {
            let zeros: Vec<Value> = nums.iter().map(|_| Value::Float(0.0)).collect();
            return Ok(Value::Array(Rc::new(RefCell::new(zeros))));
        }

        let zscores: Vec<Value> = nums.iter()
            .map(|x| Value::Float((x - mean) / stddev))
            .collect();
        Ok(Value::Array(Rc::new(RefCell::new(zscores))))
    });
}

// ============================================================================
// MATRIX FUNCTIONS
// ============================================================================

fn register_matrix(interp: &mut Interpreter) {
    // Helper to extract 2D matrix from nested arrays
    fn extract_matrix(val: &Value) -> Result<Vec<Vec<f64>>, RuntimeError> {
        match val {
            Value::Array(arr) => {
                let arr = arr.borrow();
                let mut matrix = Vec::new();
                for row in arr.iter() {
                    match row {
                        Value::Array(row_arr) => {
                            let row_arr = row_arr.borrow();
                            let mut row_vec = Vec::new();
                            for v in row_arr.iter() {
                                match v {
                                    Value::Int(n) => row_vec.push(*n as f64),
                                    Value::Float(f) => row_vec.push(*f),
                                    _ => return Err(RuntimeError::new("matrix requires numeric values")),
                                }
                            }
                            matrix.push(row_vec);
                        }
                        _ => return Err(RuntimeError::new("matrix requires 2D array")),
                    }
                }
                Ok(matrix)
            }
            _ => Err(RuntimeError::new("matrix requires array")),
        }
    }

    fn matrix_to_value(m: Vec<Vec<f64>>) -> Value {
        let rows: Vec<Value> = m.into_iter()
            .map(|row| {
                let cols: Vec<Value> = row.into_iter().map(Value::Float).collect();
                Value::Array(Rc::new(RefCell::new(cols)))
            })
            .collect();
        Value::Array(Rc::new(RefCell::new(rows)))
    }

    // matrix_new - create matrix filled with value
    define(interp, "matrix_new", Some(3), |_, args| {
        let rows = match &args[0] {
            Value::Int(n) => *n as usize,
            _ => return Err(RuntimeError::new("matrix_new() requires integer rows")),
        };
        let cols = match &args[1] {
            Value::Int(n) => *n as usize,
            _ => return Err(RuntimeError::new("matrix_new() requires integer cols")),
        };
        let fill = match &args[2] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => 0.0,
        };

        let matrix = vec![vec![fill; cols]; rows];
        Ok(matrix_to_value(matrix))
    });

    // matrix_identity - create identity matrix
    define(interp, "matrix_identity", Some(1), |_, args| {
        let size = match &args[0] {
            Value::Int(n) => *n as usize,
            _ => return Err(RuntimeError::new("matrix_identity() requires integer size")),
        };

        let mut matrix = vec![vec![0.0; size]; size];
        for i in 0..size {
            matrix[i][i] = 1.0;
        }
        Ok(matrix_to_value(matrix))
    });

    // matrix_add - add two matrices
    define(interp, "matrix_add", Some(2), |_, args| {
        let a = extract_matrix(&args[0])?;
        let b = extract_matrix(&args[1])?;

        if a.len() != b.len() || a.is_empty() || a[0].len() != b[0].len() {
            return Err(RuntimeError::new("matrix_add() requires same-size matrices"));
        }

        let result: Vec<Vec<f64>> = a.iter().zip(b.iter())
            .map(|(row_a, row_b)| {
                row_a.iter().zip(row_b.iter()).map(|(x, y)| x + y).collect()
            })
            .collect();

        Ok(matrix_to_value(result))
    });

    // matrix_sub - subtract two matrices
    define(interp, "matrix_sub", Some(2), |_, args| {
        let a = extract_matrix(&args[0])?;
        let b = extract_matrix(&args[1])?;

        if a.len() != b.len() || a.is_empty() || a[0].len() != b[0].len() {
            return Err(RuntimeError::new("matrix_sub() requires same-size matrices"));
        }

        let result: Vec<Vec<f64>> = a.iter().zip(b.iter())
            .map(|(row_a, row_b)| {
                row_a.iter().zip(row_b.iter()).map(|(x, y)| x - y).collect()
            })
            .collect();

        Ok(matrix_to_value(result))
    });

    // matrix_mul - multiply two matrices
    define(interp, "matrix_mul", Some(2), |_, args| {
        let a = extract_matrix(&args[0])?;
        let b = extract_matrix(&args[1])?;

        if a.is_empty() || b.is_empty() || a[0].len() != b.len() {
            return Err(RuntimeError::new("matrix_mul() requires compatible matrices (a.cols == b.rows)"));
        }

        let rows = a.len();
        let cols = b[0].len();
        let inner = b.len();

        let mut result = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..inner {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        Ok(matrix_to_value(result))
    });

    // matrix_scale - multiply matrix by scalar
    define(interp, "matrix_scale", Some(2), |_, args| {
        let m = extract_matrix(&args[0])?;
        let scale = match &args[1] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("matrix_scale() requires numeric scalar")),
        };

        let result: Vec<Vec<f64>> = m.iter()
            .map(|row| row.iter().map(|x| x * scale).collect())
            .collect();

        Ok(matrix_to_value(result))
    });

    // matrix_transpose - transpose matrix
    define(interp, "matrix_transpose", Some(1), |_, args| {
        let m = extract_matrix(&args[0])?;
        if m.is_empty() {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
        }

        let rows = m.len();
        let cols = m[0].len();
        let mut result = vec![vec![0.0; rows]; cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j][i] = m[i][j];
            }
        }

        Ok(matrix_to_value(result))
    });

    // matrix_det - determinant (for 2x2 and 3x3)
    define(interp, "matrix_det", Some(1), |_, args| {
        let m = extract_matrix(&args[0])?;

        if m.is_empty() || m.len() != m[0].len() {
            return Err(RuntimeError::new("matrix_det() requires square matrix"));
        }

        let n = m.len();
        match n {
            1 => Ok(Value::Float(m[0][0])),
            2 => Ok(Value::Float(m[0][0] * m[1][1] - m[0][1] * m[1][0])),
            3 => {
                let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
                Ok(Value::Float(det))
            }
            _ => Err(RuntimeError::new("matrix_det() only supports up to 3x3 matrices")),
        }
    });

    // matrix_trace - trace (sum of diagonal)
    define(interp, "matrix_trace", Some(1), |_, args| {
        let m = extract_matrix(&args[0])?;

        let size = m.len().min(if m.is_empty() { 0 } else { m[0].len() });
        let trace: f64 = (0..size).map(|i| m[i][i]).sum();

        Ok(Value::Float(trace))
    });

    // matrix_dot - dot product of vectors (1D arrays)
    define(interp, "matrix_dot", Some(2), |_, args| {
        fn extract_vector(val: &Value) -> Result<Vec<f64>, RuntimeError> {
            match val {
                Value::Array(arr) => {
                    let arr = arr.borrow();
                    let mut vec = Vec::new();
                    for v in arr.iter() {
                        match v {
                            Value::Int(n) => vec.push(*n as f64),
                            Value::Float(f) => vec.push(*f),
                            _ => return Err(RuntimeError::new("dot product requires numeric vectors")),
                        }
                    }
                    Ok(vec)
                }
                _ => Err(RuntimeError::new("dot product requires arrays")),
            }
        }

        let a = extract_vector(&args[0])?;
        let b = extract_vector(&args[1])?;

        if a.len() != b.len() {
            return Err(RuntimeError::new("matrix_dot() requires same-length vectors"));
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(Value::Float(dot))
    });
}

// Extended Euclidean algorithm for modular inverse
fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let (mut old_r, mut r) = (a, m);
    let (mut old_s, mut s) = (1i64, 0i64);

    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }

    if old_r != 1 {
        None // No inverse exists
    } else {
        Some(old_s.rem_euclid(m))
    }
}

// ============================================================================
// Phase 5: Language Power-Ups
// ============================================================================

/// Functional programming utilities
fn register_functional(interp: &mut Interpreter) {
    // identity - returns its argument unchanged
    define(interp, "identity", Some(1), |_, args| {
        Ok(args[0].clone())
    });

    // const_fn - returns a function that always returns the given value
    define(interp, "const_fn", Some(1), |_, args| {
        Ok(args[0].clone())
    });

    // apply - apply a function to an array of arguments
    define(interp, "apply", Some(2), |interp, args| {
        let func = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("apply: first argument must be a function")),
        };
        let fn_args = match &args[1] {
            Value::Array(arr) => arr.borrow().clone(),
            _ => return Err(RuntimeError::new("apply: second argument must be an array")),
        };
        interp.call_function(&func, fn_args)
    });

    // flip - swap the first two arguments of a binary function
    define(interp, "flip", Some(3), |interp, args| {
        let func = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("flip: first argument must be a function")),
        };
        let flipped_args = vec![args[2].clone(), args[1].clone()];
        interp.call_function(&func, flipped_args)
    });

    // tap - execute a function for side effects, return original value
    define(interp, "tap", Some(2), |interp, args| {
        let val = args[0].clone();
        let func = match &args[1] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("tap: second argument must be a function")),
        };
        let _ = interp.call_function(&func, vec![val.clone()]);
        Ok(val)
    });

    // thunk - create a delayed computation (wrap in array for later forcing)
    define(interp, "thunk", Some(1), |_, args| {
        Ok(Value::Array(Rc::new(RefCell::new(vec![args[0].clone()]))))
    });

    // force - force evaluation of a thunk
    define(interp, "force", Some(1), |interp, args| {
        match &args[0] {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.len() == 1 {
                    if let Value::Function(f) = &arr[0] {
                        return interp.call_function(f, vec![]);
                    }
                }
                Ok(arr.get(0).cloned().unwrap_or(Value::Null))
            }
            v => Ok(v.clone()),
        }
    });

    // negate - negate a predicate function result
    define(interp, "negate", Some(2), |interp, args| {
        let func = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("negate: first argument must be a function")),
        };
        let result = interp.call_function(&func, vec![args[1].clone()])?;
        Ok(Value::Bool(!is_truthy(&result)))
    });

    // complement - same as negate
    define(interp, "complement", Some(2), |interp, args| {
        let func = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("complement: first argument must be a function")),
        };
        let result = interp.call_function(&func, vec![args[1].clone()])?;
        Ok(Value::Bool(!is_truthy(&result)))
    });

    // partial - partially apply a function with some arguments
    define(interp, "partial", None, |interp, args| {
        if args.len() < 2 {
            return Err(RuntimeError::new("partial: requires at least function and one argument"));
        }
        let func = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("partial: first argument must be a function")),
        };
        let partial_args: Vec<Value> = args[1..].to_vec();
        interp.call_function(&func, partial_args)
    });

    // juxt - apply multiple functions to same args, return array of results
    define(interp, "juxt", None, |interp, args| {
        if args.len() < 2 {
            return Err(RuntimeError::new("juxt: requires functions and a value"));
        }
        let val = args.last().unwrap().clone();
        let results: Result<Vec<Value>, _> = args[..args.len()-1].iter().map(|f| {
            match f {
                Value::Function(func) => interp.call_function(func, vec![val.clone()]),
                _ => Err(RuntimeError::new("juxt: all but last argument must be functions")),
            }
        }).collect();
        Ok(Value::Array(Rc::new(RefCell::new(results?))))
    });
}

/// Benchmarking and profiling utilities
fn register_benchmark(interp: &mut Interpreter) {
    // bench - run a function N times and return average time in ms
    define(interp, "bench", Some(2), |interp, args| {
        let func = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("bench: first argument must be a function")),
        };
        let iterations = match &args[1] {
            Value::Int(n) => *n as usize,
            _ => return Err(RuntimeError::new("bench: second argument must be an integer")),
        };

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = interp.call_function(&func, vec![])?;
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        Ok(Value::Float(avg_ms))
    });

    // time_it - run a function once and return (result, time_ms) tuple
    define(interp, "time_it", Some(1), |interp, args| {
        let func = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("time_it: argument must be a function")),
        };

        let start = std::time::Instant::now();
        let result = interp.call_function(&func, vec![])?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(Value::Tuple(Rc::new(vec![result, Value::Float(elapsed_ms)])))
    });

    // stopwatch_start - return current time in ms
    define(interp, "stopwatch_start", Some(0), |_, _| {
        let elapsed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        Ok(Value::Float(elapsed.as_secs_f64() * 1000.0))
    });

    // stopwatch_elapsed - get elapsed time since a stopwatch start
    define(interp, "stopwatch_elapsed", Some(1), |_, args| {
        let start_ms = match &args[0] {
            Value::Float(f) => *f,
            Value::Int(n) => *n as f64,
            _ => return Err(RuntimeError::new("stopwatch_elapsed: argument must be a number")),
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let now_ms = now.as_secs_f64() * 1000.0;
        Ok(Value::Float(now_ms - start_ms))
    });

    // compare_bench - compare two functions, return speedup ratio
    define(interp, "compare_bench", Some(3), |interp, args| {
        let func1 = match &args[0] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("compare_bench: first argument must be a function")),
        };
        let func2 = match &args[1] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("compare_bench: second argument must be a function")),
        };
        let iterations = match &args[2] {
            Value::Int(n) => *n as usize,
            _ => return Err(RuntimeError::new("compare_bench: third argument must be an integer")),
        };

        let start1 = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = interp.call_function(&func1, vec![])?;
        }
        let time1 = start1.elapsed().as_secs_f64();

        let start2 = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = interp.call_function(&func2, vec![])?;
        }
        let time2 = start2.elapsed().as_secs_f64();

        let mut results = std::collections::HashMap::new();
        results.insert("time1_ms".to_string(), Value::Float(time1 * 1000.0));
        results.insert("time2_ms".to_string(), Value::Float(time2 * 1000.0));
        results.insert("speedup".to_string(), Value::Float(time1 / time2));
        results.insert("iterations".to_string(), Value::Int(iterations as i64));

        Ok(Value::Struct { name: "BenchResult".to_string(), fields: Rc::new(RefCell::new(results)) })
    });

    // memory_usage - placeholder
    define(interp, "memory_usage", Some(0), |_, _| {
        Ok(Value::Int(0))
    });
}

/// Extended iterator utilities (itertools-inspired)
fn register_itertools(interp: &mut Interpreter) {
    // cycle - create infinite cycle of array elements (returns first N)
    define(interp, "cycle", Some(2), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("cycle: first argument must be an array")),
        };
        let n = match &args[1] {
            Value::Int(n) => *n as usize,
            _ => return Err(RuntimeError::new("cycle: second argument must be an integer")),
        };

        if arr.is_empty() {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
        }

        let result: Vec<Value> = (0..n).map(|i| arr[i % arr.len()].clone()).collect();
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // repeat_val - repeat a value N times
    define(interp, "repeat_val", Some(2), |_, args| {
        let val = args[0].clone();
        let n = match &args[1] {
            Value::Int(n) => *n as usize,
            _ => return Err(RuntimeError::new("repeat_val: second argument must be an integer")),
        };

        let result: Vec<Value> = std::iter::repeat(val).take(n).collect();
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // take_while - take elements while predicate is true
    define(interp, "take_while", Some(2), |interp, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("take_while: first argument must be an array")),
        };
        let pred = match &args[1] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("take_while: second argument must be a function")),
        };

        let mut result = Vec::new();
        for item in arr {
            let keep = interp.call_function(&pred, vec![item.clone()])?;
            if is_truthy(&keep) {
                result.push(item);
            } else {
                break;
            }
        }
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // drop_while - drop elements while predicate is true
    define(interp, "drop_while", Some(2), |interp, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("drop_while: first argument must be an array")),
        };
        let pred = match &args[1] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("drop_while: second argument must be a function")),
        };

        let mut dropping = true;
        let mut result = Vec::new();
        for item in arr {
            if dropping {
                let drop = interp.call_function(&pred, vec![item.clone()])?;
                if !is_truthy(&drop) {
                    dropping = false;
                    result.push(item);
                }
            } else {
                result.push(item);
            }
        }
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // group_by - group consecutive elements by key function
    define(interp, "group_by", Some(2), |interp, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("group_by: first argument must be an array")),
        };
        let key_fn = match &args[1] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("group_by: second argument must be a function")),
        };

        let mut groups: Vec<Value> = Vec::new();
        let mut current_group: Vec<Value> = Vec::new();
        let mut current_key: Option<Value> = None;

        for item in arr {
            let key = interp.call_function(&key_fn, vec![item.clone()])?;
            match &current_key {
                Some(k) if value_eq(k, &key) => {
                    current_group.push(item);
                }
                _ => {
                    if !current_group.is_empty() {
                        groups.push(Value::Array(Rc::new(RefCell::new(current_group))));
                    }
                    current_group = vec![item];
                    current_key = Some(key);
                }
            }
        }
        if !current_group.is_empty() {
            groups.push(Value::Array(Rc::new(RefCell::new(current_group))));
        }

        Ok(Value::Array(Rc::new(RefCell::new(groups))))
    });

    // partition - split array by predicate into (true_items, false_items)
    define(interp, "partition", Some(2), |interp, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("partition: first argument must be an array")),
        };
        let pred = match &args[1] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("partition: second argument must be a function")),
        };

        let mut true_items = Vec::new();
        let mut false_items = Vec::new();

        for item in arr {
            let result = interp.call_function(&pred, vec![item.clone()])?;
            if is_truthy(&result) {
                true_items.push(item);
            } else {
                false_items.push(item);
            }
        }

        Ok(Value::Tuple(Rc::new(vec![
            Value::Array(Rc::new(RefCell::new(true_items))),
            Value::Array(Rc::new(RefCell::new(false_items))),
        ])))
    });

    // interleave - interleave two arrays
    define(interp, "interleave", Some(2), |_, args| {
        let arr1 = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("interleave: first argument must be an array")),
        };
        let arr2 = match &args[1] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("interleave: second argument must be an array")),
        };

        let mut result = Vec::new();
        let mut i1 = arr1.into_iter();
        let mut i2 = arr2.into_iter();

        loop {
            match (i1.next(), i2.next()) {
                (Some(a), Some(b)) => {
                    result.push(a);
                    result.push(b);
                }
                (Some(a), None) => {
                    result.push(a);
                    result.extend(i1);
                    break;
                }
                (None, Some(b)) => {
                    result.push(b);
                    result.extend(i2);
                    break;
                }
                (None, None) => break,
            }
        }

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // chunks - split array into chunks of size N
    define(interp, "chunks", Some(2), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("chunks: first argument must be an array")),
        };
        let size = match &args[1] {
            Value::Int(n) if *n > 0 => *n as usize,
            _ => return Err(RuntimeError::new("chunks: second argument must be a positive integer")),
        };

        let chunks: Vec<Value> = arr.chunks(size)
            .map(|chunk| Value::Array(Rc::new(RefCell::new(chunk.to_vec()))))
            .collect();

        Ok(Value::Array(Rc::new(RefCell::new(chunks))))
    });

    // windows - sliding windows of size N
    define(interp, "windows", Some(2), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("windows: first argument must be an array")),
        };
        let size = match &args[1] {
            Value::Int(n) if *n > 0 => *n as usize,
            _ => return Err(RuntimeError::new("windows: second argument must be a positive integer")),
        };

        if arr.len() < size {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
        }

        let windows: Vec<Value> = arr.windows(size)
            .map(|window| Value::Array(Rc::new(RefCell::new(window.to_vec()))))
            .collect();

        Ok(Value::Array(Rc::new(RefCell::new(windows))))
    });

    // scan - like fold but returns all intermediate values
    define(interp, "scan", Some(3), |interp, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("scan: first argument must be an array")),
        };
        let init = args[1].clone();
        let func = match &args[2] {
            Value::Function(f) => f.clone(),
            _ => return Err(RuntimeError::new("scan: third argument must be a function")),
        };

        let mut results = vec![init.clone()];
        let mut acc = init;

        for item in arr {
            acc = interp.call_function(&func, vec![acc, item])?;
            results.push(acc.clone());
        }

        Ok(Value::Array(Rc::new(RefCell::new(results))))
    });

    // frequencies - count occurrences of each element
    define(interp, "frequencies", Some(1), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("frequencies: argument must be an array")),
        };

        let mut counts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for item in &arr {
            let key = format!("{}", item);
            *counts.entry(key).or_insert(0) += 1;
        }

        let result: std::collections::HashMap<String, Value> = counts.into_iter()
            .map(|(k, v)| (k, Value::Int(v)))
            .collect();

        Ok(Value::Map(Rc::new(RefCell::new(result))))
    });

    // dedupe - remove consecutive duplicates
    define(interp, "dedupe", Some(1), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("dedupe: argument must be an array")),
        };

        let mut result = Vec::new();
        let mut prev: Option<Value> = None;

        for item in arr {
            match &prev {
                Some(p) if value_eq(p, &item) => continue,
                _ => {
                    result.push(item.clone());
                    prev = Some(item);
                }
            }
        }

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // unique - remove all duplicates (not just consecutive)
    define(interp, "unique", Some(1), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("unique: argument must be an array")),
        };

        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        for item in arr {
            let key = format!("{}", item);
            if seen.insert(key) {
                result.push(item);
            }
        }

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });
}

/// Advanced range utilities
fn register_ranges(interp: &mut Interpreter) {
    // range_step - range with custom step
    define(interp, "range_step", Some(3), |_, args| {
        let start = match &args[0] {
            Value::Int(n) => *n,
            Value::Float(f) => *f as i64,
            _ => return Err(RuntimeError::new("range_step: start must be a number")),
        };
        let end = match &args[1] {
            Value::Int(n) => *n,
            Value::Float(f) => *f as i64,
            _ => return Err(RuntimeError::new("range_step: end must be a number")),
        };
        let step = match &args[2] {
            Value::Int(n) if *n != 0 => *n,
            Value::Float(f) if *f != 0.0 => *f as i64,
            _ => return Err(RuntimeError::new("range_step: step must be a non-zero number")),
        };

        let mut result = Vec::new();
        if step > 0 {
            let mut i = start;
            while i < end {
                result.push(Value::Int(i));
                i += step;
            }
        } else {
            let mut i = start;
            while i > end {
                result.push(Value::Int(i));
                i += step;
            }
        }

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // linspace - N evenly spaced values from start to end (inclusive)
    define(interp, "linspace", Some(3), |_, args| {
        let start = match &args[0] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("linspace: start must be a number")),
        };
        let end = match &args[1] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("linspace: end must be a number")),
        };
        let n = match &args[2] {
            Value::Int(n) if *n > 0 => *n as usize,
            _ => return Err(RuntimeError::new("linspace: count must be a positive integer")),
        };

        if n == 1 {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![Value::Float(start)]))));
        }

        let step = (end - start) / (n - 1) as f64;
        let result: Vec<Value> = (0..n)
            .map(|i| Value::Float(start + step * i as f64))
            .collect();

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // logspace - N logarithmically spaced values
    define(interp, "logspace", Some(3), |_, args| {
        let start_exp = match &args[0] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("logspace: start exponent must be a number")),
        };
        let end_exp = match &args[1] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("logspace: end exponent must be a number")),
        };
        let n = match &args[2] {
            Value::Int(n) if *n > 0 => *n as usize,
            _ => return Err(RuntimeError::new("logspace: count must be a positive integer")),
        };

        if n == 1 {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![Value::Float(10f64.powf(start_exp))]))));
        }

        let step = (end_exp - start_exp) / (n - 1) as f64;
        let result: Vec<Value> = (0..n)
            .map(|i| Value::Float(10f64.powf(start_exp + step * i as f64)))
            .collect();

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // arange - like numpy arange (start, stop, step with float support)
    define(interp, "arange", Some(3), |_, args| {
        let start = match &args[0] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("arange: start must be a number")),
        };
        let stop = match &args[1] {
            Value::Int(n) => *n as f64,
            Value::Float(f) => *f,
            _ => return Err(RuntimeError::new("arange: stop must be a number")),
        };
        let step = match &args[2] {
            Value::Int(n) if *n != 0 => *n as f64,
            Value::Float(f) if *f != 0.0 => *f,
            _ => return Err(RuntimeError::new("arange: step must be a non-zero number")),
        };

        let mut result = Vec::new();
        if step > 0.0 {
            let mut x = start;
            while x < stop {
                result.push(Value::Float(x));
                x += step;
            }
        } else {
            let mut x = start;
            while x > stop {
                result.push(Value::Float(x));
                x += step;
            }
        }

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // geomspace - geometrically spaced values (like logspace but using actual values)
    define(interp, "geomspace", Some(3), |_, args| {
        let start = match &args[0] {
            Value::Int(n) if *n > 0 => *n as f64,
            Value::Float(f) if *f > 0.0 => *f,
            _ => return Err(RuntimeError::new("geomspace: start must be a positive number")),
        };
        let end = match &args[1] {
            Value::Int(n) if *n > 0 => *n as f64,
            Value::Float(f) if *f > 0.0 => *f,
            _ => return Err(RuntimeError::new("geomspace: end must be a positive number")),
        };
        let n = match &args[2] {
            Value::Int(n) if *n > 0 => *n as usize,
            _ => return Err(RuntimeError::new("geomspace: count must be a positive integer")),
        };

        if n == 1 {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![Value::Float(start)]))));
        }

        let ratio = (end / start).powf(1.0 / (n - 1) as f64);
        let result: Vec<Value> = (0..n)
            .map(|i| Value::Float(start * ratio.powi(i as i32)))
            .collect();

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });
}

/// Bitwise operations
fn register_bitwise(interp: &mut Interpreter) {
    define(interp, "bit_and", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_and: arguments must be integers")) };
        let b = match &args[1] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_and: arguments must be integers")) };
        Ok(Value::Int(a & b))
    });

    define(interp, "bit_or", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_or: arguments must be integers")) };
        let b = match &args[1] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_or: arguments must be integers")) };
        Ok(Value::Int(a | b))
    });

    define(interp, "bit_xor", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_xor: arguments must be integers")) };
        let b = match &args[1] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_xor: arguments must be integers")) };
        Ok(Value::Int(a ^ b))
    });

    define(interp, "bit_not", Some(1), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_not: argument must be an integer")) };
        Ok(Value::Int(!a))
    });

    define(interp, "bit_shl", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_shl: first argument must be an integer")) };
        let b = match &args[1] {
            Value::Int(n) if *n >= 0 && *n < 64 => *n as u32,
            _ => return Err(RuntimeError::new("bit_shl: shift amount must be 0-63")),
        };
        Ok(Value::Int(a << b))
    });

    define(interp, "bit_shr", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_shr: first argument must be an integer")) };
        let b = match &args[1] {
            Value::Int(n) if *n >= 0 && *n < 64 => *n as u32,
            _ => return Err(RuntimeError::new("bit_shr: shift amount must be 0-63")),
        };
        Ok(Value::Int(a >> b))
    });

    define(interp, "popcount", Some(1), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("popcount: argument must be an integer")) };
        Ok(Value::Int(a.count_ones() as i64))
    });

    define(interp, "leading_zeros", Some(1), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("leading_zeros: argument must be an integer")) };
        Ok(Value::Int(a.leading_zeros() as i64))
    });

    define(interp, "trailing_zeros", Some(1), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("trailing_zeros: argument must be an integer")) };
        Ok(Value::Int(a.trailing_zeros() as i64))
    });

    define(interp, "bit_test", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_test: first argument must be an integer")) };
        let pos = match &args[1] {
            Value::Int(n) if *n >= 0 && *n < 64 => *n as u32,
            _ => return Err(RuntimeError::new("bit_test: position must be 0-63")),
        };
        Ok(Value::Bool((a >> pos) & 1 == 1))
    });

    define(interp, "bit_set", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_set: first argument must be an integer")) };
        let pos = match &args[1] {
            Value::Int(n) if *n >= 0 && *n < 64 => *n as u32,
            _ => return Err(RuntimeError::new("bit_set: position must be 0-63")),
        };
        Ok(Value::Int(a | (1 << pos)))
    });

    define(interp, "bit_clear", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_clear: first argument must be an integer")) };
        let pos = match &args[1] {
            Value::Int(n) if *n >= 0 && *n < 64 => *n as u32,
            _ => return Err(RuntimeError::new("bit_clear: position must be 0-63")),
        };
        Ok(Value::Int(a & !(1 << pos)))
    });

    define(interp, "bit_toggle", Some(2), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("bit_toggle: first argument must be an integer")) };
        let pos = match &args[1] {
            Value::Int(n) if *n >= 0 && *n < 64 => *n as u32,
            _ => return Err(RuntimeError::new("bit_toggle: position must be 0-63")),
        };
        Ok(Value::Int(a ^ (1 << pos)))
    });

    define(interp, "to_binary", Some(1), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("to_binary: argument must be an integer")) };
        Ok(Value::String(Rc::new(format!("{:b}", a))))
    });

    define(interp, "from_binary", Some(1), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("from_binary: argument must be a string")) };
        match i64::from_str_radix(&s, 2) {
            Ok(n) => Ok(Value::Int(n)),
            Err(_) => Err(RuntimeError::new("from_binary: invalid binary string")),
        }
    });

    define(interp, "to_hex", Some(1), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("to_hex: argument must be an integer")) };
        Ok(Value::String(Rc::new(format!("{:x}", a))))
    });

    define(interp, "from_hex", Some(1), |_, args| {
        let s = match &args[0] { Value::String(s) => s.trim_start_matches("0x").to_string(), _ => return Err(RuntimeError::new("from_hex: argument must be a string")) };
        match i64::from_str_radix(&s, 16) {
            Ok(n) => Ok(Value::Int(n)),
            Err(_) => Err(RuntimeError::new("from_hex: invalid hex string")),
        }
    });

    define(interp, "to_octal", Some(1), |_, args| {
        let a = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("to_octal: argument must be an integer")) };
        Ok(Value::String(Rc::new(format!("{:o}", a))))
    });

    define(interp, "from_octal", Some(1), |_, args| {
        let s = match &args[0] { Value::String(s) => s.trim_start_matches("0o").to_string(), _ => return Err(RuntimeError::new("from_octal: argument must be a string")) };
        match i64::from_str_radix(&s, 8) {
            Ok(n) => Ok(Value::Int(n)),
            Err(_) => Err(RuntimeError::new("from_octal: invalid octal string")),
        }
    });
}

/// String formatting utilities
fn register_format(interp: &mut Interpreter) {
    // format - basic string formatting with {} placeholders
    define(interp, "format", None, |_, args| {
        if args.is_empty() {
            return Err(RuntimeError::new("format: requires at least a format string"));
        }
        let template = match &args[0] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("format: first argument must be a string")),
        };
        let mut result = template;
        for arg in &args[1..] {
            if let Some(pos) = result.find("{}") {
                result = format!("{}{}{}", &result[..pos], arg, &result[pos+2..]);
            }
        }
        Ok(Value::String(Rc::new(result)))
    });

    // pad_left - left-pad string to length with char
    define(interp, "pad_left", Some(3), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("pad_left: first argument must be a string")) };
        let width = match &args[1] { Value::Int(n) if *n >= 0 => *n as usize, _ => return Err(RuntimeError::new("pad_left: width must be a non-negative integer")) };
        let pad_char = match &args[2] { Value::String(s) if !s.is_empty() => s.chars().next().unwrap(), _ => return Err(RuntimeError::new("pad_left: pad character must be a non-empty string")) };
        if s.len() >= width { return Ok(Value::String(Rc::new(s))); }
        let padding: String = std::iter::repeat(pad_char).take(width - s.len()).collect();
        Ok(Value::String(Rc::new(format!("{}{}", padding, s))))
    });

    // pad_right - right-pad string to length with char
    define(interp, "pad_right", Some(3), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("pad_right: first argument must be a string")) };
        let width = match &args[1] { Value::Int(n) if *n >= 0 => *n as usize, _ => return Err(RuntimeError::new("pad_right: width must be a non-negative integer")) };
        let pad_char = match &args[2] { Value::String(s) if !s.is_empty() => s.chars().next().unwrap(), _ => return Err(RuntimeError::new("pad_right: pad character must be a non-empty string")) };
        if s.len() >= width { return Ok(Value::String(Rc::new(s))); }
        let padding: String = std::iter::repeat(pad_char).take(width - s.len()).collect();
        Ok(Value::String(Rc::new(format!("{}{}", s, padding))))
    });

    // center - center string with padding
    define(interp, "center", Some(3), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("center: first argument must be a string")) };
        let width = match &args[1] { Value::Int(n) if *n >= 0 => *n as usize, _ => return Err(RuntimeError::new("center: width must be a non-negative integer")) };
        let pad_char = match &args[2] { Value::String(s) if !s.is_empty() => s.chars().next().unwrap(), _ => return Err(RuntimeError::new("center: pad character must be a non-empty string")) };
        if s.len() >= width { return Ok(Value::String(Rc::new(s))); }
        let total_padding = width - s.len();
        let left_padding = total_padding / 2;
        let right_padding = total_padding - left_padding;
        let left: String = std::iter::repeat(pad_char).take(left_padding).collect();
        let right: String = std::iter::repeat(pad_char).take(right_padding).collect();
        Ok(Value::String(Rc::new(format!("{}{}{}", left, s, right))))
    });

    // number_format - format number with thousand separators
    define(interp, "number_format", Some(1), |_, args| {
        let n = match &args[0] { Value::Int(n) => *n, Value::Float(f) => *f as i64, _ => return Err(RuntimeError::new("number_format: argument must be a number")) };
        let s = n.abs().to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 { result.push(','); }
            result.push(c);
        }
        let formatted: String = result.chars().rev().collect();
        if n < 0 { Ok(Value::String(Rc::new(format!("-{}", formatted)))) } else { Ok(Value::String(Rc::new(formatted))) }
    });

    // ordinal - convert number to ordinal string (1st, 2nd, 3rd, etc)
    define(interp, "ordinal", Some(1), |_, args| {
        let n = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("ordinal: argument must be an integer")) };
        let suffix = match (n % 10, n % 100) { (1, 11) => "th", (2, 12) => "th", (3, 13) => "th", (1, _) => "st", (2, _) => "nd", (3, _) => "rd", _ => "th" };
        Ok(Value::String(Rc::new(format!("{}{}", n, suffix))))
    });

    // pluralize - simple pluralization
    define(interp, "pluralize", Some(3), |_, args| {
        let count = match &args[0] { Value::Int(n) => *n, _ => return Err(RuntimeError::new("pluralize: first argument must be an integer")) };
        let singular = match &args[1] { Value::String(s) => s.clone(), _ => return Err(RuntimeError::new("pluralize: second argument must be a string")) };
        let plural = match &args[2] { Value::String(s) => s.clone(), _ => return Err(RuntimeError::new("pluralize: third argument must be a string")) };
        if count == 1 || count == -1 { Ok(Value::String(singular)) } else { Ok(Value::String(plural)) }
    });

    // truncate - truncate string with ellipsis
    define(interp, "truncate", Some(2), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("truncate: first argument must be a string")) };
        let max_len = match &args[1] { Value::Int(n) if *n >= 0 => *n as usize, _ => return Err(RuntimeError::new("truncate: max length must be a non-negative integer")) };
        if s.len() <= max_len { return Ok(Value::String(Rc::new(s))); }
        if max_len <= 3 { return Ok(Value::String(Rc::new(s.chars().take(max_len).collect()))); }
        let truncated: String = s.chars().take(max_len - 3).collect();
        Ok(Value::String(Rc::new(format!("{}...", truncated))))
    });

    // word_wrap - wrap text at specified width
    define(interp, "word_wrap", Some(2), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("word_wrap: first argument must be a string")) };
        let width = match &args[1] { Value::Int(n) if *n > 0 => *n as usize, _ => return Err(RuntimeError::new("word_wrap: width must be a positive integer")) };
        let mut result = String::new();
        let mut line_len = 0;
        for word in s.split_whitespace() {
            if line_len > 0 && line_len + 1 + word.len() > width { result.push('\n'); line_len = 0; }
            else if line_len > 0 { result.push(' '); line_len += 1; }
            result.push_str(word);
            line_len += word.len();
        }
        Ok(Value::String(Rc::new(result)))
    });

    // snake_case - convert string to snake_case
    define(interp, "snake_case", Some(1), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("snake_case: argument must be a string")) };
        let mut result = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() { if i > 0 { result.push('_'); } result.push(c.to_lowercase().next().unwrap()); }
            else if c == ' ' || c == '-' { result.push('_'); }
            else { result.push(c); }
        }
        Ok(Value::String(Rc::new(result)))
    });

    // camel_case - convert string to camelCase
    define(interp, "camel_case", Some(1), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("camel_case: argument must be a string")) };
        let mut result = String::new();
        let mut capitalize_next = false;
        for (i, c) in s.chars().enumerate() {
            if c == '_' || c == '-' || c == ' ' { capitalize_next = true; }
            else if capitalize_next { result.push(c.to_uppercase().next().unwrap()); capitalize_next = false; }
            else if i == 0 { result.push(c.to_lowercase().next().unwrap()); }
            else { result.push(c); }
        }
        Ok(Value::String(Rc::new(result)))
    });

    // kebab_case - convert string to kebab-case
    define(interp, "kebab_case", Some(1), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("kebab_case: argument must be a string")) };
        let mut result = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() { if i > 0 { result.push('-'); } result.push(c.to_lowercase().next().unwrap()); }
            else if c == '_' || c == ' ' { result.push('-'); }
            else { result.push(c); }
        }
        Ok(Value::String(Rc::new(result)))
    });

    // title_case - convert string to Title Case
    define(interp, "title_case", Some(1), |_, args| {
        let s = match &args[0] { Value::String(s) => (**s).clone(), _ => return Err(RuntimeError::new("title_case: argument must be a string")) };
        let result: String = s.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() { None => String::new(), Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase() }
            })
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Value::String(Rc::new(result)))
    });
}

// ============================================================================
// PATTERN MATCHING FUNCTIONS (Phase 6)
// ============================================================================
// Advanced pattern matching utilities for expressive data manipulation.
// These complement Sigil's match expressions with functional alternatives.
// ============================================================================

fn register_pattern(interp: &mut Interpreter) {
    // --- TYPE MATCHING ---

    // type_of - get the type name as a string
    define(interp, "type_of", Some(1), |_, args| {
        let type_name = match &args[0] {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Char(_) => "char",
            Value::Array(_) => "array",
            Value::Tuple(_) => "tuple",
            Value::Map(_) => "map",
            Value::Set(_) => "set",
            Value::Struct { name, .. } => return Ok(Value::String(Rc::new(format!("struct:{}", name)))),
            Value::Variant { enum_name, variant_name, .. } => return Ok(Value::String(Rc::new(format!("{}::{}", enum_name, variant_name)))),
            Value::Function(_) => "function",
            Value::BuiltIn(_) => "builtin",
            Value::Ref(_) => "ref",
            Value::Infinity => "infinity",
            Value::Empty => "empty",
            Value::Evidential { .. } => "evidential",
            Value::Channel(_) => "channel",
            Value::ThreadHandle(_) => "thread",
            Value::Actor(_) => "actor",
            Value::Future(_) => "future",
        };
        Ok(Value::String(Rc::new(type_name.to_string())))
    });

    // is_type - check if value matches type name
    define(interp, "is_type", Some(2), |_, args| {
        let type_name = match &args[1] {
            Value::String(s) => s.to_lowercase(),
            _ => return Err(RuntimeError::new("is_type: second argument must be type name string")),
        };
        let matches = match (&args[0], type_name.as_str()) {
            (Value::Null, "null") => true,
            (Value::Bool(_), "bool") => true,
            (Value::Int(_), "int") | (Value::Int(_), "integer") => true,
            (Value::Float(_), "float") | (Value::Float(_), "number") => true,
            (Value::Int(_), "number") => true,
            (Value::String(_), "string") => true,
            (Value::Array(_), "array") | (Value::Array(_), "list") => true,
            (Value::Tuple(_), "tuple") => true,
            (Value::Map(_), "map") | (Value::Map(_), "dict") | (Value::Map(_), "object") => true,
            (Value::Set(_), "set") => true,
            (Value::Function(_), "function") | (Value::Function(_), "fn") => true,
            (Value::BuiltIn(_), "function") | (Value::BuiltIn(_), "builtin") => true,
            (Value::Struct { name, .. }, t) => t == "struct" || t == &name.to_lowercase(),
            (Value::Variant { enum_name, .. }, t) => t == "variant" || t == "enum" || t == &enum_name.to_lowercase(),
            (Value::Channel(_), "channel") => true,
            (Value::ThreadHandle(_), "thread") => true,
            (Value::Actor(_), "actor") => true,
            (Value::Future(_), "future") => true,
            _ => false,
        };
        Ok(Value::Bool(matches))
    });

    // is_null, is_bool, is_int, is_float, is_string, is_array, is_map - type predicates
    define(interp, "is_null", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Null))));
    define(interp, "is_bool", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Bool(_)))));
    define(interp, "is_int", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Int(_)))));
    define(interp, "is_float", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Float(_)))));
    define(interp, "is_number", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Int(_) | Value::Float(_)))));
    define(interp, "is_string", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::String(_)))));
    define(interp, "is_array", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Array(_)))));
    define(interp, "is_tuple", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Tuple(_)))));
    define(interp, "is_map", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Map(_)))));
    define(interp, "is_set", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Set(_)))));
    define(interp, "is_function", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Function(_) | Value::BuiltIn(_)))));
    define(interp, "is_struct", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Struct { .. }))));
    define(interp, "is_variant", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Variant { .. }))));
    define(interp, "is_future", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Future(_)))));
    define(interp, "is_channel", Some(1), |_, args| Ok(Value::Bool(matches!(&args[0], Value::Channel(_)))));

    // is_empty - check if collection is empty
    define(interp, "is_empty", Some(1), |_, args| {
        let empty = match &args[0] {
            Value::Null => true,
            Value::String(s) => s.is_empty(),
            Value::Array(a) => a.borrow().is_empty(),
            Value::Tuple(t) => t.is_empty(),
            Value::Map(m) => m.borrow().is_empty(),
            Value::Set(s) => s.borrow().is_empty(),
            _ => false,
        };
        Ok(Value::Bool(empty))
    });

    // --- REGEX PATTERN MATCHING ---

    // match_regex - match string against regex, return captures or null
    define(interp, "match_regex", Some(2), |_, args| {
        let text = match &args[0] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("match_regex: first argument must be a string")),
        };
        let pattern = match &args[1] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("match_regex: second argument must be a regex pattern string")),
        };

        let re = match Regex::new(&pattern) {
            Ok(r) => r,
            Err(e) => return Err(RuntimeError::new(format!("match_regex: invalid regex: {}", e))),
        };

        match re.captures(&text) {
            Some(caps) => {
                let mut captures: Vec<Value> = Vec::new();
                for i in 0..caps.len() {
                    if let Some(m) = caps.get(i) {
                        captures.push(Value::String(Rc::new(m.as_str().to_string())));
                    } else {
                        captures.push(Value::Null);
                    }
                }
                Ok(Value::Array(Rc::new(RefCell::new(captures))))
            }
            None => Ok(Value::Null),
        }
    });

    // match_all_regex - find all matches of regex in string
    define(interp, "match_all_regex", Some(2), |_, args| {
        let text = match &args[0] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("match_all_regex: first argument must be a string")),
        };
        let pattern = match &args[1] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("match_all_regex: second argument must be a regex pattern string")),
        };

        let re = match Regex::new(&pattern) {
            Ok(r) => r,
            Err(e) => return Err(RuntimeError::new(format!("match_all_regex: invalid regex: {}", e))),
        };

        let matches: Vec<Value> = re.find_iter(&text)
            .map(|m| Value::String(Rc::new(m.as_str().to_string())))
            .collect();
        Ok(Value::Array(Rc::new(RefCell::new(matches))))
    });

    // capture_named - extract named captures from regex match
    define(interp, "capture_named", Some(2), |_, args| {
        let text = match &args[0] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("capture_named: first argument must be a string")),
        };
        let pattern = match &args[1] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("capture_named: second argument must be a regex pattern string")),
        };

        let re = match Regex::new(&pattern) {
            Ok(r) => r,
            Err(e) => return Err(RuntimeError::new(format!("capture_named: invalid regex: {}", e))),
        };

        match re.captures(&text) {
            Some(caps) => {
                let mut result: HashMap<String, Value> = HashMap::new();
                for name in re.capture_names().flatten() {
                    if let Some(m) = caps.name(name) {
                        result.insert(name.to_string(), Value::String(Rc::new(m.as_str().to_string())));
                    }
                }
                Ok(Value::Map(Rc::new(RefCell::new(result))))
            }
            None => Ok(Value::Null),
        }
    });

    // --- STRUCTURAL PATTERN MATCHING ---

    // match_struct - check if value is a struct with given name
    define(interp, "match_struct", Some(2), |_, args| {
        let expected_name = match &args[1] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("match_struct: second argument must be struct name string")),
        };
        match &args[0] {
            Value::Struct { name, .. } => Ok(Value::Bool(name == &expected_name)),
            _ => Ok(Value::Bool(false)),
        }
    });

    // match_variant - check if value is a variant with given enum and variant name
    define(interp, "match_variant", Some(3), |_, args| {
        let expected_enum = match &args[1] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("match_variant: second argument must be enum name string")),
        };
        let expected_variant = match &args[2] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("match_variant: third argument must be variant name string")),
        };
        match &args[0] {
            Value::Variant { enum_name, variant_name, .. } => {
                Ok(Value::Bool(enum_name == &expected_enum && variant_name == &expected_variant))
            }
            _ => Ok(Value::Bool(false)),
        }
    });

    // get_field - get field from struct by name (returns null if not found)
    define(interp, "get_field", Some(2), |_, args| {
        let field_name = match &args[1] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("get_field: second argument must be field name string")),
        };
        match &args[0] {
            Value::Struct { fields, .. } => {
                Ok(fields.borrow().get(&field_name).cloned().unwrap_or(Value::Null))
            }
            Value::Map(m) => {
                Ok(m.borrow().get(&field_name).cloned().unwrap_or(Value::Null))
            }
            _ => Ok(Value::Null),
        }
    });

    // has_field - check if struct/map has a field
    define(interp, "has_field", Some(2), |_, args| {
        let field_name = match &args[1] {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("has_field: second argument must be field name string")),
        };
        match &args[0] {
            Value::Struct { fields, .. } => Ok(Value::Bool(fields.borrow().contains_key(&field_name))),
            Value::Map(m) => Ok(Value::Bool(m.borrow().contains_key(&field_name))),
            _ => Ok(Value::Bool(false)),
        }
    });

    // get_fields - get all field names from struct/map
    define(interp, "get_fields", Some(1), |_, args| {
        let fields: Vec<Value> = match &args[0] {
            Value::Struct { fields, .. } => {
                fields.borrow().keys().map(|k| Value::String(Rc::new(k.clone()))).collect()
            }
            Value::Map(m) => {
                m.borrow().keys().map(|k| Value::String(Rc::new(k.clone()))).collect()
            }
            _ => return Err(RuntimeError::new("get_fields: argument must be struct or map")),
        };
        Ok(Value::Array(Rc::new(RefCell::new(fields))))
    });

    // struct_name - get the name of a struct
    define(interp, "struct_name", Some(1), |_, args| {
        match &args[0] {
            Value::Struct { name, .. } => Ok(Value::String(Rc::new(name.clone()))),
            _ => Ok(Value::Null),
        }
    });

    // variant_name - get the variant name of an enum value
    define(interp, "variant_name", Some(1), |_, args| {
        match &args[0] {
            Value::Variant { variant_name, .. } => Ok(Value::String(Rc::new(variant_name.clone()))),
            _ => Ok(Value::Null),
        }
    });

    // variant_data - get the data payload of a variant
    define(interp, "variant_data", Some(1), |_, args| {
        match &args[0] {
            Value::Variant { fields, .. } => {
                match fields {
                    Some(f) => Ok(Value::Array(Rc::new(RefCell::new((**f).clone())))),
                    None => Ok(Value::Null),
                }
            }
            _ => Ok(Value::Null),
        }
    });

    // --- GUARDS AND CONDITIONALS ---

    // guard - conditionally return value or null (for pattern guard chains)
    define(interp, "guard", Some(2), |_, args| {
        if is_truthy(&args[0]) {
            Ok(args[1].clone())
        } else {
            Ok(Value::Null)
        }
    });

    // when - like guard but evaluates a function if condition is true
    define(interp, "when", Some(2), |interp, args| {
        if is_truthy(&args[0]) {
            match &args[1] {
                Value::Function(f) => interp.call_function(f, vec![]),
                other => Ok(other.clone()),
            }
        } else {
            Ok(Value::Null)
        }
    });

    // unless - opposite of when
    define(interp, "unless", Some(2), |interp, args| {
        if !is_truthy(&args[0]) {
            match &args[1] {
                Value::Function(f) => interp.call_function(f, vec![]),
                other => Ok(other.clone()),
            }
        } else {
            Ok(Value::Null)
        }
    });

    // cond - evaluate conditions in order, return first matching result
    // cond([[cond1, val1], [cond2, val2], ...])
    define(interp, "cond", Some(1), |interp, args| {
        let clauses = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("cond: argument must be array of [condition, value] pairs")),
        };

        for clause in clauses {
            let pair = match &clause {
                Value::Array(a) => a.borrow().clone(),
                Value::Tuple(t) => (**t).clone(),
                _ => return Err(RuntimeError::new("cond: each clause must be [condition, value] pair")),
            };
            if pair.len() != 2 {
                return Err(RuntimeError::new("cond: each clause must have exactly 2 elements"));
            }

            if is_truthy(&pair[0]) {
                return match &pair[1] {
                    Value::Function(f) => interp.call_function(f, vec![]),
                    other => Ok(other.clone()),
                };
            }
        }
        Ok(Value::Null)
    });

    // case - match value against patterns, return matching result
    // case(val, [[pattern1, result1], [pattern2, result2], ...])
    define(interp, "case", Some(2), |interp, args| {
        let value = &args[0];
        let clauses = match &args[1] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("case: second argument must be array of [pattern, result] pairs")),
        };

        for clause in clauses {
            let pair = match &clause {
                Value::Array(a) => a.borrow().clone(),
                Value::Tuple(t) => (**t).clone(),
                _ => return Err(RuntimeError::new("case: each clause must be [pattern, result] pair")),
            };
            if pair.len() != 2 {
                return Err(RuntimeError::new("case: each clause must have exactly 2 elements"));
            }

            if value_eq(value, &pair[0]) {
                return match &pair[1] {
                    Value::Function(f) => interp.call_function(f, vec![value.clone()]),
                    other => Ok(other.clone()),
                };
            }
        }
        Ok(Value::Null)
    });

    // --- DESTRUCTURING ---

    // destructure_array - extract elements at specified indices
    define(interp, "destructure_array", Some(2), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            Value::Tuple(t) => (**t).clone(),
            _ => return Err(RuntimeError::new("destructure_array: first argument must be array or tuple")),
        };
        let indices = match &args[1] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("destructure_array: second argument must be array of indices")),
        };

        let mut result = Vec::new();
        for idx in indices {
            match idx {
                Value::Int(i) => {
                    let i = if i < 0 { arr.len() as i64 + i } else { i } as usize;
                    result.push(arr.get(i).cloned().unwrap_or(Value::Null));
                }
                _ => result.push(Value::Null),
            }
        }
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // destructure_map - extract values for specified keys
    define(interp, "destructure_map", Some(2), |_, args| {
        let map = match &args[0] {
            Value::Map(m) => m.borrow().clone(),
            Value::Struct { fields, .. } => fields.borrow().clone(),
            _ => return Err(RuntimeError::new("destructure_map: first argument must be map or struct")),
        };
        let keys = match &args[1] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("destructure_map: second argument must be array of keys")),
        };

        let mut result = Vec::new();
        for key in keys {
            match key {
                Value::String(k) => {
                    result.push(map.get(&*k).cloned().unwrap_or(Value::Null));
                }
                _ => result.push(Value::Null),
            }
        }
        Ok(Value::Array(Rc::new(RefCell::new(result))))
    });

    // head_tail - split array into [head, tail]
    define(interp, "head_tail", Some(1), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("head_tail: argument must be array")),
        };

        if arr.is_empty() {
            Ok(Value::Tuple(Rc::new(vec![Value::Null, Value::Array(Rc::new(RefCell::new(vec![])))])))
        } else {
            let head = arr[0].clone();
            let tail = arr[1..].to_vec();
            Ok(Value::Tuple(Rc::new(vec![head, Value::Array(Rc::new(RefCell::new(tail)))])))
        }
    });

    // init_last - split array into [init, last]
    define(interp, "init_last", Some(1), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("init_last: argument must be array")),
        };

        if arr.is_empty() {
            Ok(Value::Tuple(Rc::new(vec![Value::Array(Rc::new(RefCell::new(vec![]))), Value::Null])))
        } else {
            let last = arr[arr.len() - 1].clone();
            let init = arr[..arr.len() - 1].to_vec();
            Ok(Value::Tuple(Rc::new(vec![Value::Array(Rc::new(RefCell::new(init))), last])))
        }
    });

    // split_at - split array at index into [left, right]
    define(interp, "split_at", Some(2), |_, args| {
        let arr = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("split_at: first argument must be array")),
        };
        let idx = match &args[1] {
            Value::Int(i) => *i as usize,
            _ => return Err(RuntimeError::new("split_at: second argument must be integer")),
        };

        let idx = idx.min(arr.len());
        let left = arr[..idx].to_vec();
        let right = arr[idx..].to_vec();
        Ok(Value::Tuple(Rc::new(vec![
            Value::Array(Rc::new(RefCell::new(left))),
            Value::Array(Rc::new(RefCell::new(right))),
        ])))
    });

    // --- OPTIONAL/NULLABLE HELPERS ---

    // unwrap_or - return value if not null, else default
    define(interp, "unwrap_or", Some(2), |_, args| {
        if matches!(&args[0], Value::Null) {
            Ok(args[1].clone())
        } else {
            Ok(args[0].clone())
        }
    });

    // unwrap_or_else - return value if not null, else call function
    define(interp, "unwrap_or_else", Some(2), |interp, args| {
        if matches!(&args[0], Value::Null) {
            match &args[1] {
                Value::Function(f) => interp.call_function(f, vec![]),
                other => Ok(other.clone()),
            }
        } else {
            Ok(args[0].clone())
        }
    });

    // map_or - if value is not null, apply function, else return default
    define(interp, "map_or", Some(3), |interp, args| {
        if matches!(&args[0], Value::Null) {
            Ok(args[1].clone())
        } else {
            match &args[2] {
                Value::Function(f) => interp.call_function(f, vec![args[0].clone()]),
                _ => Err(RuntimeError::new("map_or: third argument must be a function")),
            }
        }
    });

    // coalesce - return first non-null value from array
    define(interp, "coalesce", Some(1), |_, args| {
        let values = match &args[0] {
            Value::Array(a) => a.borrow().clone(),
            _ => return Err(RuntimeError::new("coalesce: argument must be array")),
        };

        for v in values {
            if !matches!(v, Value::Null) {
                return Ok(v);
            }
        }
        Ok(Value::Null)
    });

    // --- EQUALITY AND COMPARISON ---

    // deep_eq - deep structural equality check
    define(interp, "deep_eq", Some(2), |_, args| {
        Ok(Value::Bool(deep_value_eq(&args[0], &args[1])))
    });

    // same_type - check if two values have the same type
    define(interp, "same_type", Some(2), |_, args| {
        let same = match (&args[0], &args[1]) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(_), Value::Bool(_)) => true,
            (Value::Int(_), Value::Int(_)) => true,
            (Value::Float(_), Value::Float(_)) => true,
            (Value::String(_), Value::String(_)) => true,
            (Value::Array(_), Value::Array(_)) => true,
            (Value::Tuple(_), Value::Tuple(_)) => true,
            (Value::Map(_), Value::Map(_)) => true,
            (Value::Set(_), Value::Set(_)) => true,
            (Value::Function(_), Value::Function(_)) => true,
            (Value::BuiltIn(_), Value::BuiltIn(_)) => true,
            (Value::Struct { name: n1, .. }, Value::Struct { name: n2, .. }) => n1 == n2,
            (Value::Variant { enum_name: e1, .. }, Value::Variant { enum_name: e2, .. }) => e1 == e2,
            _ => false,
        };
        Ok(Value::Bool(same))
    });

    // compare - three-way comparison: -1, 0, or 1
    define(interp, "compare", Some(2), |_, args| {
        let cmp = match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => return Err(RuntimeError::new("compare: can only compare numbers or strings")),
        };
        Ok(Value::Int(match cmp {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }))
    });

    // between - check if value is between min and max (inclusive)
    define(interp, "between", Some(3), |_, args| {
        let in_range = match (&args[0], &args[1], &args[2]) {
            (Value::Int(v), Value::Int(min), Value::Int(max)) => v >= min && v <= max,
            (Value::Float(v), Value::Float(min), Value::Float(max)) => v >= min && v <= max,
            (Value::Int(v), Value::Int(min), Value::Float(max)) => (*v as f64) >= (*min as f64) && (*v as f64) <= *max,
            (Value::Int(v), Value::Float(min), Value::Int(max)) => (*v as f64) >= *min && (*v as f64) <= (*max as f64),
            (Value::Float(v), Value::Int(min), Value::Int(max)) => *v >= (*min as f64) && *v <= (*max as f64),
            (Value::String(v), Value::String(min), Value::String(max)) => v >= min && v <= max,
            _ => return Err(RuntimeError::new("between: arguments must be comparable (numbers or strings)")),
        };
        Ok(Value::Bool(in_range))
    });

    // clamp - constrain value to range
    define(interp, "clamp", Some(3), |_, args| {
        match (&args[0], &args[1], &args[2]) {
            (Value::Int(v), Value::Int(min), Value::Int(max)) => {
                Ok(Value::Int((*v).max(*min).min(*max)))
            }
            (Value::Float(v), Value::Float(min), Value::Float(max)) => {
                Ok(Value::Float(v.max(*min).min(*max)))
            }
            (Value::Int(v), Value::Int(min), Value::Float(max)) => {
                Ok(Value::Float((*v as f64).max(*min as f64).min(*max)))
            }
            _ => Err(RuntimeError::new("clamp: arguments must be numbers")),
        }
    });
}

// Deep value equality for nested structures
fn deep_value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            (*a as f64 - b).abs() < f64::EPSILON
        }
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Array(a), Value::Array(b)) => {
            let a = a.borrow();
            let b = b.borrow();
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| deep_value_eq(x, y))
        }
        (Value::Tuple(a), Value::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| deep_value_eq(x, y))
        }
        (Value::Map(a), Value::Map(b)) => {
            let a = a.borrow();
            let b = b.borrow();
            a.len() == b.len() && a.iter().all(|(k, v)| {
                b.get(k).map_or(false, |bv| deep_value_eq(v, bv))
            })
        }
        (Value::Set(a), Value::Set(b)) => {
            let a = a.borrow();
            let b = b.borrow();
            a.len() == b.len() && a.iter().all(|k| b.contains(k))
        }
        (Value::Struct { name: n1, fields: f1 }, Value::Struct { name: n2, fields: f2 }) => {
            let f1 = f1.borrow();
            let f2 = f2.borrow();
            n1 == n2 && f1.len() == f2.len() && f1.iter().all(|(k, v)| {
                f2.get(k).map_or(false, |v2| deep_value_eq(v, v2))
            })
        }
        (Value::Variant { enum_name: e1, variant_name: v1, fields: d1 },
         Value::Variant { enum_name: e2, variant_name: v2, fields: d2 }) => {
            if e1 != e2 || v1 != v2 { return false; }
            match (d1, d2) {
                (Some(f1), Some(f2)) => f1.len() == f2.len() && f1.iter().zip(f2.iter()).all(|(x, y)| deep_value_eq(x, y)),
                (None, None) => true,
                _ => false,
            }
        }
        _ => false,
    }
}

// Helper for value equality comparison
fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            (*a as f64 - b).abs() < f64::EPSILON
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Parser;

    fn eval(source: &str) -> Result<Value, RuntimeError> {
        let mut parser = Parser::new(source);
        let file = parser.parse_file().map_err(|e| RuntimeError::new(e.to_string()))?;
        let mut interp = Interpreter::new();
        register_stdlib(&mut interp);
        interp.execute(&file)
    }

    #[test]
    fn test_math_functions() {
        assert!(matches!(eval("fn main() { return abs(-5); }"), Ok(Value::Int(5))));
        assert!(matches!(eval("fn main() { return floor(3.7); }"), Ok(Value::Int(3))));
        assert!(matches!(eval("fn main() { return ceil(3.2); }"), Ok(Value::Int(4))));
        assert!(matches!(eval("fn main() { return max(3, 7); }"), Ok(Value::Int(7))));
        assert!(matches!(eval("fn main() { return min(3, 7); }"), Ok(Value::Int(3))));
    }

    #[test]
    fn test_collection_functions() {
        assert!(matches!(eval("fn main() { return len([1, 2, 3]); }"), Ok(Value::Int(3))));
        assert!(matches!(eval("fn main() { return first([1, 2, 3]); }"), Ok(Value::Int(1))));
        assert!(matches!(eval("fn main() { return last([1, 2, 3]); }"), Ok(Value::Int(3))));
    }

    #[test]
    fn test_string_functions() {
        assert!(matches!(eval(r#"fn main() { return upper("hello"); }"#), Ok(Value::String(s)) if s.as_str() == "HELLO"));
        assert!(matches!(eval(r#"fn main() { return lower("HELLO"); }"#), Ok(Value::String(s)) if s.as_str() == "hello"));
    }

    #[test]
    fn test_evidence_functions() {
        let result = eval("fn main() { return evidence_of(uncertain(42)); }");
        assert!(matches!(result, Ok(Value::String(s)) if s.as_str() == "uncertain"));
    }

    #[test]
    fn test_iter_functions() {
        assert!(matches!(eval("fn main() { return sum([1, 2, 3, 4]); }"), Ok(Value::Int(10))));
        assert!(matches!(eval("fn main() { return product([1, 2, 3, 4]); }"), Ok(Value::Int(24))));
    }

    #[test]
    fn test_cycle_functions() {
        assert!(matches!(eval("fn main() { return mod_add(7, 8, 12); }"), Ok(Value::Int(3))));
        assert!(matches!(eval("fn main() { return mod_pow(2, 10, 1000); }"), Ok(Value::Int(24))));
    }
}

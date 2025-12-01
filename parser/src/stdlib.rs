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

use crate::interpreter::{Interpreter, Value, Evidence, RuntimeError, BuiltInFn};
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use std::io::Write;

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
            _ => Err(RuntimeError::new("len() requires array, string, or tuple")),
        }
    });

    define(interp, "is_empty", Some(1), |_, args| {
        match &args[0] {
            Value::Array(arr) => Ok(Value::Bool(arr.borrow().is_empty())),
            Value::String(s) => Ok(Value::Bool(s.is_empty())),
            Value::Tuple(t) => Ok(Value::Bool(t.is_empty())),
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

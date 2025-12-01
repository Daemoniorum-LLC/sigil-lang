//! Tree-walking interpreter for Sigil.
//!
//! Executes Sigil AST directly for rapid prototyping and REPL.

use crate::ast::*;
use crate::span::Span;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

/// Runtime value in Sigil.
#[derive(Clone)]
pub enum Value {
    /// Null/void
    Null,
    /// Boolean
    Bool(bool),
    /// Integer (64-bit)
    Int(i64),
    /// Float (64-bit)
    Float(f64),
    /// String
    String(Rc<String>),
    /// Character
    Char(char),
    /// Array/list
    Array(Rc<RefCell<Vec<Value>>>),
    /// Tuple
    Tuple(Rc<Vec<Value>>),
    /// Struct instance
    Struct {
        name: String,
        fields: Rc<RefCell<HashMap<String, Value>>>,
    },
    /// Enum variant
    Variant {
        enum_name: String,
        variant_name: String,
        fields: Option<Rc<Vec<Value>>>,
    },
    /// Function/closure
    Function(Rc<Function>),
    /// Built-in function
    BuiltIn(Rc<BuiltInFn>),
    /// Reference to another value
    Ref(Rc<RefCell<Value>>),
    /// Special mathematical values
    Infinity,
    Empty,
    /// Evidence-wrapped value
    Evidential {
        value: Box<Value>,
        evidence: Evidence,
    },
}

/// Evidence level at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    Known,      // !
    Uncertain,  // ?
    Reported,   // ~
    Paradox,    // ‽
}

/// A Sigil function
pub struct Function {
    pub name: Option<String>,
    pub params: Vec<String>,
    pub body: Expr,
    pub closure: Rc<RefCell<Environment>>,
}

/// Built-in function type
pub struct BuiltInFn {
    pub name: String,
    pub arity: Option<usize>, // None = variadic
    pub func: fn(&mut Interpreter, Vec<Value>) -> Result<Value, RuntimeError>,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Char(c) => write!(f, "'{}'", c),
            Value::Array(arr) => {
                let arr = arr.borrow();
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", v)?;
                }
                write!(f, "]")
            }
            Value::Tuple(vals) => {
                write!(f, "(")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", v)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                let fields = fields.borrow();
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {:?}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::Variant { enum_name, variant_name, fields } => {
                write!(f, "{}::{}", enum_name, variant_name)?;
                if let Some(fields) = fields {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{:?}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Function(func) => {
                write!(f, "<fn {}>", func.name.as_deref().unwrap_or("anonymous"))
            }
            Value::BuiltIn(b) => write!(f, "<builtin {}>", b.name),
            Value::Ref(r) => write!(f, "&{:?}", r.borrow()),
            Value::Infinity => write!(f, "∞"),
            Value::Empty => write!(f, "∅"),
            Value::Evidential { value, evidence } => {
                write!(f, "{:?}", value)?;
                match evidence {
                    Evidence::Known => write!(f, "!"),
                    Evidence::Uncertain => write!(f, "?"),
                    Evidence::Reported => write!(f, "~"),
                    Evidence::Paradox => write!(f, "‽"),
                }
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Char(c) => write!(f, "{}", c),
            Value::Array(arr) => {
                let arr = arr.borrow();
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Evidential { value, .. } => write!(f, "{}", value),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Runtime error
#[derive(Debug)]
pub struct RuntimeError {
    pub message: String,
    pub span: Option<Span>,
}

impl RuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
        }
    }

    pub fn with_span(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span: Some(span),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Runtime error: {}", self.message)?;
        if let Some(span) = self.span {
            write!(f, " at {}", span)?;
        }
        Ok(())
    }
}

/// Control flow signals for return/break/continue
#[derive(Debug, Clone)]
pub enum ControlFlow {
    Return(Value),
    Break(Option<Value>),
    Continue,
}

impl From<ControlFlow> for RuntimeError {
    fn from(cf: ControlFlow) -> Self {
        match cf {
            ControlFlow::Return(_) => RuntimeError::new("return outside function"),
            ControlFlow::Break(_) => RuntimeError::new("break outside loop"),
            ControlFlow::Continue => RuntimeError::new("continue outside loop"),
        }
    }
}

/// Result type that can contain control flow
pub type EvalResult = Result<Value, EvalError>;

/// Error type that includes control flow
#[derive(Debug)]
pub enum EvalError {
    Runtime(RuntimeError),
    Control(ControlFlow),
}

impl From<RuntimeError> for EvalError {
    fn from(e: RuntimeError) -> Self {
        EvalError::Runtime(e)
    }
}

impl From<ControlFlow> for EvalError {
    fn from(cf: ControlFlow) -> Self {
        EvalError::Control(cf)
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Runtime(e) => write!(f, "{}", e),
            EvalError::Control(cf) => write!(f, "Unexpected control flow: {:?}", cf),
        }
    }
}

/// Environment for variable bindings
#[derive(Clone)]
pub struct Environment {
    values: HashMap<String, Value>,
    parent: Option<Rc<RefCell<Environment>>>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            parent: None,
        }
    }

    pub fn with_parent(parent: Rc<RefCell<Environment>>) -> Self {
        Self {
            values: HashMap::new(),
            parent: Some(parent),
        }
    }

    pub fn define(&mut self, name: String, value: Value) {
        self.values.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.values.get(name) {
            Some(value.clone())
        } else if let Some(ref parent) = self.parent {
            parent.borrow().get(name)
        } else {
            None
        }
    }

    pub fn set(&mut self, name: &str, value: Value) -> Result<(), RuntimeError> {
        if self.values.contains_key(name) {
            self.values.insert(name.to_string(), value);
            Ok(())
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().set(name, value)
        } else {
            Err(RuntimeError::new(format!("Undefined variable: {}", name)))
        }
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

/// The Sigil interpreter
pub struct Interpreter {
    /// Global environment
    pub globals: Rc<RefCell<Environment>>,
    /// Current environment
    pub environment: Rc<RefCell<Environment>>,
    /// Type definitions
    pub types: HashMap<String, TypeDef>,
    /// Output buffer (for testing)
    pub output: Vec<String>,
    /// Return value from the last return statement (control flow)
    return_value: Option<Value>,
}

/// Type definition for structs/enums
pub enum TypeDef {
    Struct(StructDef),
    Enum(EnumDef),
}

impl Interpreter {
    pub fn new() -> Self {
        let globals = Rc::new(RefCell::new(Environment::new()));
        let environment = globals.clone();

        let mut interp = Self {
            globals: globals.clone(),
            environment,
            types: HashMap::new(),
            return_value: None,
            output: Vec::new(),
        };

        // Register built-in functions
        interp.register_builtins();

        interp
    }

    fn register_builtins(&mut self) {
        // Print function
        self.define_builtin("print", None, |interp, args| {
            let output: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
            let line = output.join(" ");
            println!("{}", line);
            interp.output.push(line);
            Ok(Value::Null)
        });

        // Type checking
        self.define_builtin("type_of", Some(1), |_, args| {
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
                Value::Evidential { .. } => "evidential",
            };
            Ok(Value::String(Rc::new(type_name.to_string())))
        });

        // Array operations
        self.define_builtin("len", Some(1), |_, args| {
            match &args[0] {
                Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
                _ => Err(RuntimeError::new("len() requires array, string, or tuple")),
            }
        });

        self.define_builtin("push", Some(2), |_, args| {
            match &args[0] {
                Value::Array(arr) => {
                    arr.borrow_mut().push(args[1].clone());
                    Ok(Value::Null)
                }
                _ => Err(RuntimeError::new("push() requires array")),
            }
        });

        self.define_builtin("pop", Some(1), |_, args| {
            match &args[0] {
                Value::Array(arr) => {
                    arr.borrow_mut().pop()
                        .ok_or_else(|| RuntimeError::new("pop() on empty array"))
                }
                _ => Err(RuntimeError::new("pop() requires array")),
            }
        });

        // Math functions
        self.define_builtin("abs", Some(1), |_, args| {
            match &args[0] {
                Value::Int(n) => Ok(Value::Int(n.abs())),
                Value::Float(n) => Ok(Value::Float(n.abs())),
                _ => Err(RuntimeError::new("abs() requires number")),
            }
        });

        self.define_builtin("sqrt", Some(1), |_, args| {
            match &args[0] {
                Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
                Value::Float(n) => Ok(Value::Float(n.sqrt())),
                _ => Err(RuntimeError::new("sqrt() requires number")),
            }
        });

        self.define_builtin("sin", Some(1), |_, args| {
            match &args[0] {
                Value::Int(n) => Ok(Value::Float((*n as f64).sin())),
                Value::Float(n) => Ok(Value::Float(n.sin())),
                _ => Err(RuntimeError::new("sin() requires number")),
            }
        });

        self.define_builtin("cos", Some(1), |_, args| {
            match &args[0] {
                Value::Int(n) => Ok(Value::Float((*n as f64).cos())),
                Value::Float(n) => Ok(Value::Float(n.cos())),
                _ => Err(RuntimeError::new("cos() requires number")),
            }
        });

        // Evidence operations
        self.define_builtin("known", Some(1), |_, args| {
            Ok(Value::Evidential {
                value: Box::new(args[0].clone()),
                evidence: Evidence::Known,
            })
        });

        self.define_builtin("uncertain", Some(1), |_, args| {
            Ok(Value::Evidential {
                value: Box::new(args[0].clone()),
                evidence: Evidence::Uncertain,
            })
        });

        self.define_builtin("reported", Some(1), |_, args| {
            Ok(Value::Evidential {
                value: Box::new(args[0].clone()),
                evidence: Evidence::Reported,
            })
        });

        // Range function
        self.define_builtin("range", Some(2), |_, args| {
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
    }

    fn define_builtin(
        &mut self,
        name: &str,
        arity: Option<usize>,
        func: fn(&mut Interpreter, Vec<Value>) -> Result<Value, RuntimeError>,
    ) {
        let builtin = Value::BuiltIn(Rc::new(BuiltInFn {
            name: name.to_string(),
            arity,
            func,
        }));
        self.globals.borrow_mut().define(name.to_string(), builtin);
    }

    /// Execute a source file
    pub fn execute(&mut self, file: &SourceFile) -> Result<Value, RuntimeError> {
        let mut result = Value::Null;

        for item in &file.items {
            result = self.execute_item(&item.node)?;
        }

        // Look for main function and execute it
        let main_fn = self.globals.borrow().get("main").and_then(|v| {
            if let Value::Function(f) = v {
                Some(f.clone())
            } else {
                None
            }
        });
        if let Some(f) = main_fn {
            result = self.call_function(&f, vec![])?;
        }

        Ok(result)
    }

    fn execute_item(&mut self, item: &Item) -> Result<Value, RuntimeError> {
        match item {
            Item::Function(func) => {
                let fn_value = self.create_function(func)?;
                self.globals.borrow_mut().define(func.name.name.clone(), fn_value);
                Ok(Value::Null)
            }
            Item::Struct(s) => {
                self.types.insert(s.name.name.clone(), TypeDef::Struct(s.clone()));
                Ok(Value::Null)
            }
            Item::Enum(e) => {
                self.types.insert(e.name.name.clone(), TypeDef::Enum(e.clone()));
                Ok(Value::Null)
            }
            Item::Const(c) => {
                let value = self.evaluate(&c.value)?;
                self.globals.borrow_mut().define(c.name.name.clone(), value);
                Ok(Value::Null)
            }
            Item::Static(s) => {
                let value = self.evaluate(&s.value)?;
                self.globals.borrow_mut().define(s.name.name.clone(), value);
                Ok(Value::Null)
            }
            _ => Ok(Value::Null), // Skip other items for now
        }
    }

    fn create_function(&self, func: &crate::ast::Function) -> Result<Value, RuntimeError> {
        let params: Vec<String> = func.params.iter().map(|p| {
            match &p.pattern {
                Pattern::Ident { name, .. } => name.name.clone(),
                _ => "_".to_string(),
            }
        }).collect();

        let body = func.body.as_ref()
            .map(|b| Expr::Block(b.clone()))
            .unwrap_or(Expr::Literal(Literal::Bool(false)));

        Ok(Value::Function(Rc::new(Function {
            name: Some(func.name.name.clone()),
            params,
            body,
            closure: self.environment.clone(),
        })))
    }

    /// Evaluate an expression
    pub fn evaluate(&mut self, expr: &Expr) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Literal(lit) => self.eval_literal(lit),
            Expr::Path(path) => self.eval_path(path),
            Expr::Binary { left, op, right } => self.eval_binary(left, op, right),
            Expr::Unary { op, expr } => self.eval_unary(op, expr),
            Expr::Call { func, args } => self.eval_call(func, args),
            Expr::Array(elements) => self.eval_array(elements),
            Expr::Tuple(elements) => self.eval_tuple(elements),
            Expr::Block(block) => self.eval_block(block),
            Expr::If { condition, then_branch, else_branch } => {
                self.eval_if(condition, then_branch, else_branch)
            }
            Expr::Match { expr, arms } => self.eval_match(expr, arms),
            Expr::For { pattern, iter, body } => self.eval_for(pattern, iter, body),
            Expr::While { condition, body } => self.eval_while(condition, body),
            Expr::Loop(body) => self.eval_loop(body),
            Expr::Return(value) => self.eval_return(value),
            Expr::Break(value) => self.eval_break(value),
            Expr::Continue => Err(RuntimeError::new("continue outside loop")),
            Expr::Index { expr, index } => self.eval_index(expr, index),
            Expr::Field { expr, field } => self.eval_field(expr, field),
            Expr::MethodCall { receiver, method, args } => {
                self.eval_method_call(receiver, method, args)
            }
            Expr::Pipe { expr, operations } => self.eval_pipe(expr, operations),
            Expr::Closure { params, body } => self.eval_closure(params, body),
            Expr::Struct { path, fields, rest } => self.eval_struct_literal(path, fields, rest),
            Expr::Evidential { expr, evidentiality } => self.eval_evidential(expr, evidentiality),
            Expr::Range { start, end, inclusive } => self.eval_range(start, end, *inclusive),
            _ => Err(RuntimeError::new(format!("Unsupported expression: {:?}", expr))),
        }
    }

    fn eval_literal(&self, lit: &Literal) -> Result<Value, RuntimeError> {
        match lit {
            Literal::Int { value, base, .. } => {
                let n = self.parse_int(value, base)?;
                Ok(Value::Int(n))
            }
            Literal::Float { value, .. } => {
                let n: f64 = value.parse().map_err(|_| {
                    RuntimeError::new(format!("Invalid float: {}", value))
                })?;
                Ok(Value::Float(n))
            }
            Literal::String(s) => Ok(Value::String(Rc::new(s.clone()))),
            Literal::Char(c) => Ok(Value::Char(*c)),
            Literal::Bool(b) => Ok(Value::Bool(*b)),
            Literal::Empty => Ok(Value::Empty),
            Literal::Infinity => Ok(Value::Infinity),
            Literal::Circle => Ok(Value::Int(0)), // ◯ = zero
        }
    }

    fn parse_int(&self, value: &str, base: &NumBase) -> Result<i64, RuntimeError> {
        let (radix, prefix_len) = match base {
            NumBase::Binary => (2, 2),      // 0b
            NumBase::Octal => (8, 2),       // 0o
            NumBase::Decimal => (10, 0),
            NumBase::Hex => (16, 2),        // 0x
            NumBase::Vigesimal => (20, 2),  // 0v
            NumBase::Sexagesimal => (60, 2), // 0s
            NumBase::Duodecimal => (12, 2), // 0z
            NumBase::Explicit(b) => (*b as u32, 0),
        };

        let clean = value[prefix_len..].replace('_', "");
        i64::from_str_radix(&clean, radix).map_err(|_| {
            RuntimeError::new(format!("Invalid integer: {}", value))
        })
    }

    fn eval_path(&self, path: &TypePath) -> Result<Value, RuntimeError> {
        if path.segments.len() == 1 {
            let name = &path.segments[0].ident.name;
            self.environment.borrow().get(name)
                .ok_or_else(|| RuntimeError::new(format!("Undefined variable: {}", name)))
        } else {
            // Multi-segment path (module::item)
            Err(RuntimeError::new("Module paths not yet supported"))
        }
    }

    fn eval_binary(&mut self, left: &Expr, op: &BinOp, right: &Expr) -> Result<Value, RuntimeError> {
        let lhs = self.evaluate(left)?;

        // Short-circuit for && and ||
        match op {
            BinOp::And => {
                if !self.is_truthy(&lhs) {
                    return Ok(Value::Bool(false));
                }
                let rhs = self.evaluate(right)?;
                return Ok(Value::Bool(self.is_truthy(&rhs)));
            }
            BinOp::Or => {
                if self.is_truthy(&lhs) {
                    return Ok(Value::Bool(true));
                }
                let rhs = self.evaluate(right)?;
                return Ok(Value::Bool(self.is_truthy(&rhs)));
            }
            _ => {}
        }

        let rhs = self.evaluate(right)?;

        match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => self.int_binary_op(a, b, op),
            (Value::Float(a), Value::Float(b)) => self.float_binary_op(a, b, op),
            (Value::Int(a), Value::Float(b)) => self.float_binary_op(a as f64, b, op),
            (Value::Float(a), Value::Int(b)) => self.float_binary_op(a, b as f64, op),
            (Value::String(a), Value::String(b)) => {
                match op {
                    BinOp::Concat => Ok(Value::String(Rc::new(format!("{}{}", a, b)))),
                    BinOp::Eq => Ok(Value::Bool(*a == *b)),
                    BinOp::Ne => Ok(Value::Bool(*a != *b)),
                    _ => Err(RuntimeError::new("Invalid string operation")),
                }
            }
            (Value::Bool(a), Value::Bool(b)) => {
                match op {
                    BinOp::Eq => Ok(Value::Bool(a == b)),
                    BinOp::Ne => Ok(Value::Bool(a != b)),
                    _ => Err(RuntimeError::new("Invalid boolean operation")),
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                match op {
                    BinOp::Concat => {
                        let mut result = a.borrow().clone();
                        result.extend(b.borrow().iter().cloned());
                        Ok(Value::Array(Rc::new(RefCell::new(result))))
                    }
                    _ => Err(RuntimeError::new("Invalid array operation")),
                }
            }
            _ => Err(RuntimeError::new("Type mismatch in binary operation")),
        }
    }

    fn int_binary_op(&self, a: i64, b: i64, op: &BinOp) -> Result<Value, RuntimeError> {
        Ok(match op {
            BinOp::Add => Value::Int(a + b),
            BinOp::Sub => Value::Int(a - b),
            BinOp::Mul => Value::Int(a * b),
            BinOp::Div => {
                if b == 0 {
                    return Err(RuntimeError::new("Division by zero"));
                }
                Value::Int(a / b)
            }
            BinOp::Rem => {
                if b == 0 {
                    return Err(RuntimeError::new("Division by zero"));
                }
                Value::Int(a % b)
            }
            BinOp::Pow => Value::Int(a.pow(b as u32)),
            BinOp::Eq => Value::Bool(a == b),
            BinOp::Ne => Value::Bool(a != b),
            BinOp::Lt => Value::Bool(a < b),
            BinOp::Le => Value::Bool(a <= b),
            BinOp::Gt => Value::Bool(a > b),
            BinOp::Ge => Value::Bool(a >= b),
            BinOp::BitAnd => Value::Int(a & b),
            BinOp::BitOr => Value::Int(a | b),
            BinOp::BitXor => Value::Int(a ^ b),
            BinOp::Shl => Value::Int(a << b),
            BinOp::Shr => Value::Int(a >> b),
            _ => return Err(RuntimeError::new("Invalid integer operation")),
        })
    }

    fn float_binary_op(&self, a: f64, b: f64, op: &BinOp) -> Result<Value, RuntimeError> {
        Ok(match op {
            BinOp::Add => Value::Float(a + b),
            BinOp::Sub => Value::Float(a - b),
            BinOp::Mul => Value::Float(a * b),
            BinOp::Div => Value::Float(a / b),
            BinOp::Rem => Value::Float(a % b),
            BinOp::Pow => Value::Float(a.powf(b)),
            BinOp::Eq => Value::Bool((a - b).abs() < f64::EPSILON),
            BinOp::Ne => Value::Bool((a - b).abs() >= f64::EPSILON),
            BinOp::Lt => Value::Bool(a < b),
            BinOp::Le => Value::Bool(a <= b),
            BinOp::Gt => Value::Bool(a > b),
            BinOp::Ge => Value::Bool(a >= b),
            _ => return Err(RuntimeError::new("Invalid float operation")),
        })
    }

    fn eval_unary(&mut self, op: &UnaryOp, expr: &Expr) -> Result<Value, RuntimeError> {
        let val = self.evaluate(expr)?;
        match (op, val) {
            (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
            (UnaryOp::Neg, Value::Float(n)) => Ok(Value::Float(-n)),
            (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            (UnaryOp::Not, Value::Int(n)) => Ok(Value::Int(!n)),
            (UnaryOp::Ref, val) => Ok(Value::Ref(Rc::new(RefCell::new(val)))),
            (UnaryOp::Deref, Value::Ref(r)) => Ok(r.borrow().clone()),
            _ => Err(RuntimeError::new("Invalid unary operation")),
        }
    }

    fn eval_call(&mut self, func_expr: &Expr, args: &[Expr]) -> Result<Value, RuntimeError> {
        let func = self.evaluate(func_expr)?;
        let arg_values: Vec<Value> = args.iter()
            .map(|a| self.evaluate(a))
            .collect::<Result<_, _>>()?;

        match func {
            Value::Function(f) => self.call_function(&f, arg_values),
            Value::BuiltIn(b) => self.call_builtin(&b, arg_values),
            _ => Err(RuntimeError::new("Cannot call non-function")),
        }
    }

    pub fn call_function(&mut self, func: &Function, args: Vec<Value>) -> Result<Value, RuntimeError> {
        if args.len() != func.params.len() {
            return Err(RuntimeError::new(format!(
                "Expected {} arguments, got {}",
                func.params.len(),
                args.len()
            )));
        }

        // Create new environment for function
        let env = Rc::new(RefCell::new(Environment::with_parent(func.closure.clone())));

        // Bind parameters
        for (param, value) in func.params.iter().zip(args) {
            env.borrow_mut().define(param.clone(), value);
        }

        // Execute function body
        let prev_env = self.environment.clone();
        self.environment = env;

        let result = match self.evaluate(&func.body) {
            Ok(val) => Ok(val),
            Err(e) if e.message == "return" => {
                // Extract return value from stored location
                Ok(self.return_value.take().unwrap_or(Value::Null))
            }
            Err(e) => Err(e),
        };

        self.environment = prev_env;
        result
    }

    fn call_builtin(&mut self, builtin: &BuiltInFn, args: Vec<Value>) -> Result<Value, RuntimeError> {
        if let Some(arity) = builtin.arity {
            if args.len() != arity {
                return Err(RuntimeError::new(format!(
                    "{}() expects {} arguments, got {}",
                    builtin.name, arity, args.len()
                )));
            }
        }
        (builtin.func)(self, args)
    }

    fn eval_array(&mut self, elements: &[Expr]) -> Result<Value, RuntimeError> {
        let values: Vec<Value> = elements.iter()
            .map(|e| self.evaluate(e))
            .collect::<Result<_, _>>()?;
        Ok(Value::Array(Rc::new(RefCell::new(values))))
    }

    fn eval_tuple(&mut self, elements: &[Expr]) -> Result<Value, RuntimeError> {
        let values: Vec<Value> = elements.iter()
            .map(|e| self.evaluate(e))
            .collect::<Result<_, _>>()?;
        Ok(Value::Tuple(Rc::new(values)))
    }

    fn eval_block(&mut self, block: &Block) -> Result<Value, RuntimeError> {
        let env = Rc::new(RefCell::new(Environment::with_parent(self.environment.clone())));
        let prev_env = self.environment.clone();
        self.environment = env;

        let mut result = Value::Null;

        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { pattern, init, .. } => {
                    let value = match init {
                        Some(expr) => self.evaluate(expr)?,
                        None => Value::Null,
                    };
                    self.bind_pattern(pattern, value)?;
                }
                Stmt::Expr(expr) => {
                    result = self.evaluate(expr)?;
                }
                Stmt::Semi(expr) => {
                    self.evaluate(expr)?;
                    result = Value::Null;
                }
                Stmt::Item(item) => {
                    self.execute_item(item)?;
                }
            }
        }

        if let Some(expr) = &block.expr {
            result = self.evaluate(expr)?;
        }

        self.environment = prev_env;
        Ok(result)
    }

    fn bind_pattern(&mut self, pattern: &Pattern, value: Value) -> Result<(), RuntimeError> {
        match pattern {
            Pattern::Ident { name, .. } => {
                self.environment.borrow_mut().define(name.name.clone(), value);
                Ok(())
            }
            Pattern::Tuple(patterns) => {
                if let Value::Tuple(values) = value {
                    if patterns.len() != values.len() {
                        return Err(RuntimeError::new("Tuple pattern size mismatch"));
                    }
                    for (p, v) in patterns.iter().zip(values.iter()) {
                        self.bind_pattern(p, v.clone())?;
                    }
                    Ok(())
                } else {
                    Err(RuntimeError::new("Expected tuple"))
                }
            }
            Pattern::Wildcard => Ok(()),
            _ => Err(RuntimeError::new("Unsupported pattern")),
        }
    }

    fn eval_if(
        &mut self,
        condition: &Expr,
        then_branch: &Block,
        else_branch: &Option<Box<Expr>>,
    ) -> Result<Value, RuntimeError> {
        let cond = self.evaluate(condition)?;
        if self.is_truthy(&cond) {
            self.eval_block(then_branch)
        } else if let Some(else_expr) = else_branch {
            self.evaluate(else_expr)
        } else {
            Ok(Value::Null)
        }
    }

    fn eval_match(&mut self, expr: &Expr, arms: &[MatchArm]) -> Result<Value, RuntimeError> {
        let value = self.evaluate(expr)?;

        for arm in arms {
            if self.pattern_matches(&arm.pattern, &value)? {
                // Check guard if present
                if let Some(guard) = &arm.guard {
                    let guard_val = self.evaluate(guard)?;
                    if !self.is_truthy(&guard_val) {
                        continue;
                    }
                }

                // Bind pattern variables and evaluate body
                let env = Rc::new(RefCell::new(Environment::with_parent(self.environment.clone())));
                let prev_env = self.environment.clone();
                self.environment = env;

                self.bind_pattern(&arm.pattern, value)?;
                let result = self.evaluate(&arm.body);

                self.environment = prev_env;
                return result;
            }
        }

        Err(RuntimeError::new("No matching pattern"))
    }

    fn pattern_matches(&self, pattern: &Pattern, value: &Value) -> Result<bool, RuntimeError> {
        match (pattern, value) {
            (Pattern::Wildcard, _) => Ok(true),
            (Pattern::Ident { .. }, _) => Ok(true),
            (Pattern::Literal(lit), val) => {
                let lit_val = self.eval_literal(lit)?;
                Ok(self.values_equal(&lit_val, val))
            }
            (Pattern::Tuple(patterns), Value::Tuple(values)) => {
                if patterns.len() != values.len() {
                    return Ok(false);
                }
                for (p, v) in patterns.iter().zip(values.iter()) {
                    if !self.pattern_matches(p, v)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            _ => false,
        }
    }

    fn eval_for(&mut self, pattern: &Pattern, iter: &Expr, body: &Block) -> Result<Value, RuntimeError> {
        let iterable = self.evaluate(iter)?;
        let items = match iterable {
            Value::Array(arr) => arr.borrow().clone(),
            Value::Tuple(t) => (*t).clone(),
            _ => return Err(RuntimeError::new("Cannot iterate over non-iterable")),
        };

        let mut result = Value::Null;
        for item in items {
            let env = Rc::new(RefCell::new(Environment::with_parent(self.environment.clone())));
            let prev_env = self.environment.clone();
            self.environment = env;

            self.bind_pattern(pattern, item)?;

            match self.eval_block(body) {
                Ok(val) => result = val,
                Err(e) if e.message == "break" => {
                    self.environment = prev_env;
                    break;
                }
                Err(e) if e.message == "continue" => {
                    self.environment = prev_env;
                    continue;
                }
                Err(e) => {
                    self.environment = prev_env;
                    return Err(e);
                }
            }

            self.environment = prev_env;
        }

        Ok(result)
    }

    fn eval_while(&mut self, condition: &Expr, body: &Block) -> Result<Value, RuntimeError> {
        let mut result = Value::Null;
        loop {
            let cond = self.evaluate(condition)?;
            if !self.is_truthy(&cond) {
                break;
            }

            match self.eval_block(body) {
                Ok(val) => result = val,
                Err(e) if e.message == "break" => break,
                Err(e) if e.message == "continue" => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(result)
    }

    fn eval_loop(&mut self, body: &Block) -> Result<Value, RuntimeError> {
        loop {
            match self.eval_block(body) {
                Ok(_) => {}
                Err(e) if e.message == "break" => break,
                Err(e) if e.message == "continue" => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(Value::Null)
    }

    fn eval_return(&mut self, value: &Option<Box<Expr>>) -> Result<Value, RuntimeError> {
        let val = match value {
            Some(expr) => self.evaluate(expr)?,
            None => Value::Null,
        };
        // Store return value for call_function to retrieve
        self.return_value = Some(val);
        Err(RuntimeError::new("return"))
    }

    fn eval_break(&mut self, value: &Option<Box<Expr>>) -> Result<Value, RuntimeError> {
        Err(RuntimeError::new("break"))
    }

    fn eval_index(&mut self, expr: &Expr, index: &Expr) -> Result<Value, RuntimeError> {
        let collection = self.evaluate(expr)?;
        let idx = self.evaluate(index)?;

        match (collection, idx) {
            (Value::Array(arr), Value::Int(i)) => {
                let arr = arr.borrow();
                let i = if i < 0 { arr.len() as i64 + i } else { i } as usize;
                arr.get(i).cloned()
                    .ok_or_else(|| RuntimeError::new("Index out of bounds"))
            }
            (Value::Tuple(t), Value::Int(i)) => {
                let i = if i < 0 { t.len() as i64 + i } else { i } as usize;
                t.get(i).cloned()
                    .ok_or_else(|| RuntimeError::new("Index out of bounds"))
            }
            (Value::String(s), Value::Int(i)) => {
                let i = if i < 0 { s.len() as i64 + i } else { i } as usize;
                s.chars().nth(i)
                    .map(Value::Char)
                    .ok_or_else(|| RuntimeError::new("Index out of bounds"))
            }
            _ => Err(RuntimeError::new("Cannot index")),
        }
    }

    fn eval_field(&mut self, expr: &Expr, field: &Ident) -> Result<Value, RuntimeError> {
        let value = self.evaluate(expr)?;
        match value {
            Value::Struct { fields, .. } => {
                fields.borrow().get(&field.name).cloned()
                    .ok_or_else(|| RuntimeError::new(format!("Unknown field: {}", field.name)))
            }
            Value::Tuple(t) => {
                // Tuple field access like .0, .1
                let idx: usize = field.name.parse()
                    .map_err(|_| RuntimeError::new("Invalid tuple index"))?;
                t.get(idx).cloned()
                    .ok_or_else(|| RuntimeError::new("Tuple index out of bounds"))
            }
            _ => Err(RuntimeError::new("Cannot access field on non-struct")),
        }
    }

    fn eval_method_call(
        &mut self,
        receiver: &Expr,
        method: &Ident,
        args: &[Expr],
    ) -> Result<Value, RuntimeError> {
        let recv = self.evaluate(receiver)?;
        let arg_values: Vec<Value> = args.iter()
            .map(|a| self.evaluate(a))
            .collect::<Result<_, _>>()?;

        // Built-in methods
        match (&recv, method.name.as_str()) {
            (Value::Array(arr), "len") => Ok(Value::Int(arr.borrow().len() as i64)),
            (Value::Array(arr), "push") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("push expects 1 argument"));
                }
                arr.borrow_mut().push(arg_values[0].clone());
                Ok(Value::Null)
            }
            (Value::Array(arr), "pop") => {
                arr.borrow_mut().pop()
                    .ok_or_else(|| RuntimeError::new("pop on empty array"))
            }
            (Value::Array(arr), "reverse") => {
                let mut v = arr.borrow().clone();
                v.reverse();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
            (Value::String(s), "chars") => {
                let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                Ok(Value::Array(Rc::new(RefCell::new(chars))))
            }
            (Value::String(s), "contains") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("contains expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::String(sub) => Ok(Value::Bool(s.contains(sub.as_str()))),
                    _ => Err(RuntimeError::new("contains expects string")),
                }
            }
            _ => Err(RuntimeError::new(format!("Unknown method: {}", method.name))),
        }
    }

    fn eval_pipe(&mut self, expr: &Expr, operations: &[PipeOp]) -> Result<Value, RuntimeError> {
        let mut value = self.evaluate(expr)?;

        for op in operations {
            value = self.apply_pipe_op(value, op)?;
        }

        Ok(value)
    }

    fn apply_pipe_op(&mut self, value: Value, op: &PipeOp) -> Result<Value, RuntimeError> {
        match op {
            PipeOp::Transform(body) => {
                // τ{f} - map over collection or apply to single value
                match value {
                    Value::Array(arr) => {
                        let results: Vec<Value> = arr.borrow().iter()
                            .map(|item| {
                                self.environment.borrow_mut().define("_".to_string(), item.clone());
                                self.evaluate(body)
                            })
                            .collect::<Result<_, _>>()?;
                        Ok(Value::Array(Rc::new(RefCell::new(results))))
                    }
                    single => {
                        self.environment.borrow_mut().define("_".to_string(), single);
                        self.evaluate(body)
                    }
                }
            }
            PipeOp::Filter(predicate) => {
                // φ{p} - filter collection
                match value {
                    Value::Array(arr) => {
                        let results: Vec<Value> = arr.borrow().iter()
                            .filter_map(|item| {
                                self.environment.borrow_mut().define("_".to_string(), item.clone());
                                match self.evaluate(predicate) {
                                    Ok(v) if self.is_truthy(&v) => Some(Ok(item.clone())),
                                    Ok(_) => None,
                                    Err(e) => Some(Err(e)),
                                }
                            })
                            .collect::<Result<_, _>>()?;
                        Ok(Value::Array(Rc::new(RefCell::new(results))))
                    }
                    _ => Err(RuntimeError::new("Filter requires array")),
                }
            }
            PipeOp::Sort(field) => {
                // σ - sort collection
                match value {
                    Value::Array(arr) => {
                        let mut v = arr.borrow().clone();
                        v.sort_by(|a, b| self.compare_values(a, b, field));
                        Ok(Value::Array(Rc::new(RefCell::new(v))))
                    }
                    _ => Err(RuntimeError::new("Sort requires array")),
                }
            }
            PipeOp::Reduce(body) => {
                // ρ{f} - reduce collection
                match value {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        if arr.is_empty() {
                            return Err(RuntimeError::new("Cannot reduce empty array"));
                        }
                        let mut acc = arr[0].clone();
                        for item in arr.iter().skip(1) {
                            self.environment.borrow_mut().define("acc".to_string(), acc);
                            self.environment.borrow_mut().define("_".to_string(), item.clone());
                            acc = self.evaluate(body)?;
                        }
                        Ok(acc)
                    }
                    _ => Err(RuntimeError::new("Reduce requires array")),
                }
            }
            PipeOp::Method { name, args } => {
                let arg_values: Vec<Value> = args.iter()
                    .map(|a| self.evaluate(a))
                    .collect::<Result<_, _>>()?;

                // Check for built-in pipe methods
                match name.name.as_str() {
                    "collect" => Ok(value), // Already collected
                    "sum" | "Σ" => self.sum_values(value),
                    "product" | "Π" => self.product_values(value),
                    "len" => {
                        match &value {
                            Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                            Value::String(s) => Ok(Value::Int(s.len() as i64)),
                            _ => Err(RuntimeError::new("len requires array or string")),
                        }
                    }
                    "reverse" => {
                        match value {
                            Value::Array(arr) => {
                                let mut v = arr.borrow().clone();
                                v.reverse();
                                Ok(Value::Array(Rc::new(RefCell::new(v))))
                            }
                            _ => Err(RuntimeError::new("reverse requires array")),
                        }
                    }
                    "first" => {
                        match &value {
                            Value::Array(arr) => arr.borrow().first().cloned()
                                .ok_or_else(|| RuntimeError::new("first on empty array")),
                            _ => Err(RuntimeError::new("first requires array")),
                        }
                    }
                    "last" => {
                        match &value {
                            Value::Array(arr) => arr.borrow().last().cloned()
                                .ok_or_else(|| RuntimeError::new("last on empty array")),
                            _ => Err(RuntimeError::new("last requires array")),
                        }
                    }
                    "take" => {
                        if arg_values.len() != 1 {
                            return Err(RuntimeError::new("take requires 1 argument"));
                        }
                        let n = match &arg_values[0] {
                            Value::Int(n) => *n as usize,
                            _ => return Err(RuntimeError::new("take requires integer")),
                        };
                        match value {
                            Value::Array(arr) => {
                                let v: Vec<Value> = arr.borrow().iter().take(n).cloned().collect();
                                Ok(Value::Array(Rc::new(RefCell::new(v))))
                            }
                            _ => Err(RuntimeError::new("take requires array")),
                        }
                    }
                    "skip" => {
                        if arg_values.len() != 1 {
                            return Err(RuntimeError::new("skip requires 1 argument"));
                        }
                        let n = match &arg_values[0] {
                            Value::Int(n) => *n as usize,
                            _ => return Err(RuntimeError::new("skip requires integer")),
                        };
                        match value {
                            Value::Array(arr) => {
                                let v: Vec<Value> = arr.borrow().iter().skip(n).cloned().collect();
                                Ok(Value::Array(Rc::new(RefCell::new(v))))
                            }
                            _ => Err(RuntimeError::new("skip requires array")),
                        }
                    }
                    _ => Err(RuntimeError::new(format!("Unknown pipe method: {}", name.name))),
                }
            }
            PipeOp::Await => {
                // For now, just return the value (no async yet)
                Ok(value)
            }
            PipeOp::Named { prefix, body } => {
                // Named morpheme like ·map{f}
                let method_name = prefix.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join("·");
                match method_name.as_str() {
                    "map" => {
                        if let Some(body) = body {
                            match value {
                                Value::Array(arr) => {
                                    let results: Vec<Value> = arr.borrow().iter()
                                        .map(|item| {
                                            self.environment.borrow_mut().define("_".to_string(), item.clone());
                                            self.evaluate(body)
                                        })
                                        .collect::<Result<_, _>>()?;
                                    Ok(Value::Array(Rc::new(RefCell::new(results))))
                                }
                                _ => Err(RuntimeError::new("map requires array")),
                            }
                        } else {
                            Ok(value)
                        }
                    }
                    "filter" => {
                        if let Some(body) = body {
                            match value {
                                Value::Array(arr) => {
                                    let results: Vec<Value> = arr.borrow().iter()
                                        .filter_map(|item| {
                                            self.environment.borrow_mut().define("_".to_string(), item.clone());
                                            match self.evaluate(body) {
                                                Ok(v) if self.is_truthy(&v) => Some(Ok(item.clone())),
                                                Ok(_) => None,
                                                Err(e) => Some(Err(e)),
                                            }
                                        })
                                        .collect::<Result<_, _>>()?;
                                    Ok(Value::Array(Rc::new(RefCell::new(results))))
                                }
                                _ => Err(RuntimeError::new("filter requires array")),
                            }
                        } else {
                            Ok(value)
                        }
                    }
                    _ => Err(RuntimeError::new(format!("Unknown named morpheme: {}", method_name))),
                }
            }
        }
    }

    fn sum_values(&self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Ok(Value::Int(0));
                }
                let mut sum = match &arr[0] {
                    Value::Int(_) => Value::Int(0),
                    Value::Float(_) => Value::Float(0.0),
                    _ => return Err(RuntimeError::new("Cannot sum non-numeric array")),
                };
                for item in arr.iter() {
                    sum = match (&sum, item) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                        (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
                        (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
                        (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
                        _ => return Err(RuntimeError::new("Cannot sum non-numeric values")),
                    };
                }
                Ok(sum)
            }
            _ => Err(RuntimeError::new("sum requires array")),
        }
    }

    fn product_values(&self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Ok(Value::Int(1));
                }
                let mut prod = match &arr[0] {
                    Value::Int(_) => Value::Int(1),
                    Value::Float(_) => Value::Float(1.0),
                    _ => return Err(RuntimeError::new("Cannot multiply non-numeric array")),
                };
                for item in arr.iter() {
                    prod = match (&prod, item) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
                        (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
                        (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
                        (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
                        _ => return Err(RuntimeError::new("Cannot multiply non-numeric values")),
                    };
                }
                Ok(prod)
            }
            _ => Err(RuntimeError::new("product requires array")),
        }
    }

    fn compare_values(&self, a: &Value, b: &Value, _field: &Option<Ident>) -> std::cmp::Ordering {
        // Simple comparison for now
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        }
    }

    fn eval_closure(&mut self, params: &[ClosureParam], body: &Expr) -> Result<Value, RuntimeError> {
        let param_names: Vec<String> = params.iter().map(|p| {
            match &p.pattern {
                Pattern::Ident { name, .. } => name.name.clone(),
                _ => "_".to_string(),
            }
        }).collect();

        Ok(Value::Function(Rc::new(Function {
            name: None,
            params: param_names,
            body: body.clone(),
            closure: self.environment.clone(),
        })))
    }

    fn eval_struct_literal(
        &mut self,
        path: &TypePath,
        fields: &[FieldInit],
        _rest: &Option<Box<Expr>>,
    ) -> Result<Value, RuntimeError> {
        let name = path.segments.iter()
            .map(|s| s.ident.name.as_str())
            .collect::<Vec<_>>()
            .join("::");

        let mut field_values = HashMap::new();
        for field in fields {
            let value = match &field.value {
                Some(expr) => self.evaluate(expr)?,
                None => self.environment.borrow().get(&field.name.name)
                    .ok_or_else(|| RuntimeError::new(format!("Unknown variable: {}", field.name.name)))?,
            };
            field_values.insert(field.name.name.clone(), value);
        }

        Ok(Value::Struct {
            name,
            fields: Rc::new(RefCell::new(field_values)),
        })
    }

    fn eval_evidential(&mut self, expr: &Expr, ev: &Evidentiality) -> Result<Value, RuntimeError> {
        let value = self.evaluate(expr)?;
        let evidence = match ev {
            Evidentiality::Known => Evidence::Known,
            Evidentiality::Uncertain => Evidence::Uncertain,
            Evidentiality::Reported => Evidence::Reported,
            Evidentiality::Paradox => Evidence::Paradox,
        };
        Ok(Value::Evidential {
            value: Box::new(value),
            evidence,
        })
    }

    fn eval_range(
        &mut self,
        start: &Option<Box<Expr>>,
        end: &Option<Box<Expr>>,
        inclusive: bool,
    ) -> Result<Value, RuntimeError> {
        let start_val = match start {
            Some(e) => match self.evaluate(e)? {
                Value::Int(n) => n,
                _ => return Err(RuntimeError::new("Range requires integer bounds")),
            },
            None => 0,
        };

        let end_val = match end {
            Some(e) => match self.evaluate(e)? {
                Value::Int(n) => n,
                _ => return Err(RuntimeError::new("Range requires integer bounds")),
            },
            None => return Err(RuntimeError::new("Range requires end bound")),
        };

        let values: Vec<Value> = if inclusive {
            (start_val..=end_val).map(Value::Int).collect()
        } else {
            (start_val..end_val).map(Value::Int).collect()
        };

        Ok(Value::Array(Rc::new(RefCell::new(values))))
    }

    fn is_truthy(&self, value: &Value) -> bool {
        match value {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(n) => *n != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Array(arr) => !arr.borrow().is_empty(),
            Value::Empty => false,
            Value::Evidential { value, .. } => self.is_truthy(value),
            _ => true,
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Parser;

    fn run(source: &str) -> Result<Value, RuntimeError> {
        let mut parser = Parser::new(source);
        let file = parser.parse_file().map_err(|e| RuntimeError::new(e.to_string()))?;
        let mut interp = Interpreter::new();
        interp.execute(&file)
    }

    #[test]
    fn test_arithmetic() {
        assert!(matches!(run("fn main() { return 2 + 3; }"), Ok(Value::Int(5))));
        assert!(matches!(run("fn main() { return 10 - 4; }"), Ok(Value::Int(6))));
        assert!(matches!(run("fn main() { return 3 * 4; }"), Ok(Value::Int(12))));
        assert!(matches!(run("fn main() { return 15 / 3; }"), Ok(Value::Int(5))));
        assert!(matches!(run("fn main() { return 2 ** 10; }"), Ok(Value::Int(1024))));
    }

    #[test]
    fn test_variables() {
        assert!(matches!(run("fn main() { let x = 42; return x; }"), Ok(Value::Int(42))));
    }

    #[test]
    fn test_conditionals() {
        assert!(matches!(
            run("fn main() { if true { return 1; } else { return 2; } }"),
            Ok(Value::Int(1))
        ));
        assert!(matches!(
            run("fn main() { if false { return 1; } else { return 2; } }"),
            Ok(Value::Int(2))
        ));
    }

    #[test]
    fn test_arrays() {
        assert!(matches!(run("fn main() { return [1, 2, 3][1]; }"), Ok(Value::Int(2))));
    }

    #[test]
    fn test_functions() {
        let result = run("
            fn double(x: i64) -> i64 { return x * 2; }
            fn main() { return double(21); }
        ");
        assert!(matches!(result, Ok(Value::Int(42))));
    }

    #[test]
    fn test_pipe_transform() {
        let result = run("fn main() { return [1, 2, 3]|τ{_ * 2}|sum; }");
        assert!(matches!(result, Ok(Value::Int(12))));
    }

    #[test]
    fn test_pipe_filter() {
        let result = run("fn main() { return [1, 2, 3, 4, 5]|φ{_ > 2}|sum; }");
        assert!(matches!(result, Ok(Value::Int(12)))); // 3 + 4 + 5
    }
}

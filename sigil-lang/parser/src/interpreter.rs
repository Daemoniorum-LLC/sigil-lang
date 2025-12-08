//! Tree-walking interpreter for Sigil.
//!
//! Executes Sigil AST directly for rapid prototyping and REPL.

use crate::ast::*;
use crate::span::Span;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

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
    /// Affect-wrapped value (sentiment, emotion, sarcasm, etc.)
    Affective {
        value: Box<Value>,
        affect: RuntimeAffect,
    },
    /// HashMap
    Map(Rc<RefCell<HashMap<String, Value>>>),
    /// HashSet (stores keys only, values are unit)
    Set(Rc<RefCell<std::collections::HashSet<String>>>),
    /// Channel for message passing (sender, receiver)
    Channel(Arc<ChannelInner>),
    /// Thread handle
    ThreadHandle(Arc<Mutex<Option<JoinHandle<Value>>>>),
    /// Actor (mailbox + state)
    Actor(Arc<ActorInner>),
    /// Future - represents an async computation
    Future(Rc<RefCell<FutureInner>>),
}

/// Future state for async computations
#[derive(Clone)]
pub enum FutureState {
    /// Not yet started
    Pending,
    /// Currently executing
    Running,
    /// Completed with value
    Ready(Box<Value>),
    /// Failed with error
    Failed(String),
}

/// Inner future representation
pub struct FutureInner {
    /// Current state
    pub state: FutureState,
    /// The computation to run (if pending)
    pub computation: Option<FutureComputation>,
    /// Completion time for timer futures
    pub complete_at: Option<std::time::Instant>,
}

impl Clone for FutureInner {
    fn clone(&self) -> Self {
        FutureInner {
            state: self.state.clone(),
            computation: self.computation.clone(),
            complete_at: self.complete_at,
        }
    }
}

/// Types of future computations
#[derive(Clone)]
pub enum FutureComputation {
    /// Immediate value (already resolved)
    Immediate(Box<Value>),
    /// Timer - completes after duration
    Timer(std::time::Duration),
    /// Lazy computation - function + captured args
    Lazy {
        func: Rc<Function>,
        args: Vec<Value>,
    },
    /// Join multiple futures
    Join(Vec<Rc<RefCell<FutureInner>>>),
    /// Race multiple futures (first to complete wins)
    Race(Vec<Rc<RefCell<FutureInner>>>),
}

/// Inner channel state - wraps mpsc channel
pub struct ChannelInner {
    pub sender: Mutex<mpsc::Sender<Value>>,
    pub receiver: Mutex<mpsc::Receiver<Value>>,
}

impl Clone for ChannelInner {
    fn clone(&self) -> Self {
        // Channels can't really be cloned - create a dummy
        // This is for the Clone requirement on Value
        panic!("Channels cannot be cloned directly - use channel_clone()")
    }
}

/// Inner actor state - single-threaded for interpreter (Value contains Rc)
/// For true async actors, use the JIT backend
pub struct ActorInner {
    pub name: String,
    pub message_queue: Mutex<Vec<(String, String)>>, // (msg_type, serialized_data)
    pub message_count: std::sync::atomic::AtomicUsize,
}

/// Evidence level at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    Known,     // !
    Uncertain, // ?
    Reported,  // ~
    Paradox,   // ‽
}

/// Runtime affect markers for sentiment and emotion tracking
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeAffect {
    pub sentiment: Option<RuntimeSentiment>,
    pub sarcasm: bool, // ⸮
    pub intensity: Option<RuntimeIntensity>,
    pub formality: Option<RuntimeFormality>,
    pub emotion: Option<RuntimeEmotion>,
    pub confidence: Option<RuntimeConfidence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeSentiment {
    Positive, // ⊕
    Negative, // ⊖
    Neutral,  // ⊜
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeIntensity {
    Up,   // ↑
    Down, // ↓
    Max,  // ⇈
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFormality {
    Formal,   // ♔
    Informal, // ♟
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEmotion {
    Joy,      // ☺
    Sadness,  // ☹
    Anger,    // ⚡
    Fear,     // ❄
    Surprise, // ✦
    Love,     // ♡
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeConfidence {
    High,   // ◉
    Medium, // ◎
    Low,    // ○
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
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, "]")
            }
            Value::Tuple(vals) => {
                write!(f, "(")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                let fields = fields.borrow();
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {:?}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::Variant {
                enum_name,
                variant_name,
                fields,
            } => {
                write!(f, "{}::{}", enum_name, variant_name)?;
                if let Some(fields) = fields {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
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
            Value::Map(map) => {
                let map = map.borrow();
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}: {:?}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Set(set) => {
                let set = set.borrow();
                write!(f, "Set{{")?;
                for (i, k) in set.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", k)?;
                }
                write!(f, "}}")
            }
            Value::Channel(_) => write!(f, "<channel>"),
            Value::ThreadHandle(_) => write!(f, "<thread>"),
            Value::Actor(actor) => write!(f, "<actor {}>", actor.name),
            Value::Future(fut) => {
                let fut = fut.borrow();
                match &fut.state {
                    FutureState::Pending => write!(f, "<future pending>"),
                    FutureState::Running => write!(f, "<future running>"),
                    FutureState::Ready(v) => write!(f, "<future ready: {:?}>", v),
                    FutureState::Failed(e) => write!(f, "<future failed: {}>", e),
                }
            }
            Value::Affective { value, affect } => {
                write!(f, "{:?}", value)?;
                if let Some(s) = &affect.sentiment {
                    match s {
                        RuntimeSentiment::Positive => write!(f, "⊕")?,
                        RuntimeSentiment::Negative => write!(f, "⊖")?,
                        RuntimeSentiment::Neutral => write!(f, "⊜")?,
                    }
                }
                if affect.sarcasm {
                    write!(f, "⸮")?;
                }
                if let Some(i) = &affect.intensity {
                    match i {
                        RuntimeIntensity::Up => write!(f, "↑")?,
                        RuntimeIntensity::Down => write!(f, "↓")?,
                        RuntimeIntensity::Max => write!(f, "⇈")?,
                    }
                }
                if let Some(fo) = &affect.formality {
                    match fo {
                        RuntimeFormality::Formal => write!(f, "♔")?,
                        RuntimeFormality::Informal => write!(f, "♟")?,
                    }
                }
                if let Some(e) = &affect.emotion {
                    match e {
                        RuntimeEmotion::Joy => write!(f, "☺")?,
                        RuntimeEmotion::Sadness => write!(f, "☹")?,
                        RuntimeEmotion::Anger => write!(f, "⚡")?,
                        RuntimeEmotion::Fear => write!(f, "❄")?,
                        RuntimeEmotion::Surprise => write!(f, "✦")?,
                        RuntimeEmotion::Love => write!(f, "♡")?,
                    }
                }
                if let Some(c) = &affect.confidence {
                    match c {
                        RuntimeConfidence::High => write!(f, "◉")?,
                        RuntimeConfidence::Medium => write!(f, "◎")?,
                        RuntimeConfidence::Low => write!(f, "○")?,
                    }
                }
                Ok(())
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
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Evidential { value, .. } => write!(f, "{}", value),
            Value::Affective { value, affect } => {
                // Display affect markers as suffix symbols
                let mut suffix = String::new();
                if let Some(sent) = &affect.sentiment {
                    suffix.push(match sent {
                        RuntimeSentiment::Positive => '⊕',
                        RuntimeSentiment::Negative => '⊖',
                        RuntimeSentiment::Neutral => '⊜',
                    });
                }
                if affect.sarcasm {
                    suffix.push('⸮');
                }
                if let Some(int) = &affect.intensity {
                    suffix.push(match int {
                        RuntimeIntensity::Up => '↑',
                        RuntimeIntensity::Down => '↓',
                        RuntimeIntensity::Max => '⇈',
                    });
                }
                if let Some(form) = &affect.formality {
                    suffix.push(match form {
                        RuntimeFormality::Formal => '♔',
                        RuntimeFormality::Informal => '♟',
                    });
                }
                if let Some(emo) = &affect.emotion {
                    suffix.push(match emo {
                        RuntimeEmotion::Joy => '☺',
                        RuntimeEmotion::Sadness => '☹',
                        RuntimeEmotion::Anger => '⚡',
                        RuntimeEmotion::Fear => '❄',
                        RuntimeEmotion::Surprise => '✦',
                        RuntimeEmotion::Love => '♡',
                    });
                }
                if let Some(conf) = &affect.confidence {
                    suffix.push(match conf {
                        RuntimeConfidence::High => '◉',
                        RuntimeConfidence::Medium => '◎',
                        RuntimeConfidence::Low => '○',
                    });
                }
                write!(f, "{}{}", value, suffix)
            }
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
    /// Variant constructors: qualified_name -> (enum_name, variant_name, arity)
    pub variant_constructors: HashMap<String, (String, String, usize)>,
    /// Structs with #[derive(Default)]
    pub default_structs: HashMap<String, StructDef>,
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
            variant_constructors: HashMap::new(),
            default_structs: HashMap::new(),
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
                Value::Affective { .. } => "affective",
                Value::Map(_) => "map",
                Value::Set(_) => "set",
                Value::Channel(_) => "channel",
                Value::ThreadHandle(_) => "thread",
                Value::Actor(_) => "actor",
                Value::Future(_) => "future",
            };
            Ok(Value::String(Rc::new(type_name.to_string())))
        });

        // Array operations
        self.define_builtin("len", Some(1), |_, args| match &args[0] {
            Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
            Value::String(s) => Ok(Value::Int(s.len() as i64)),
            Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
            _ => Err(RuntimeError::new("len() requires array, string, or tuple")),
        });

        self.define_builtin("push", Some(2), |_, args| match &args[0] {
            Value::Array(arr) => {
                arr.borrow_mut().push(args[1].clone());
                Ok(Value::Null)
            }
            _ => Err(RuntimeError::new("push() requires array")),
        });

        self.define_builtin("pop", Some(1), |_, args| match &args[0] {
            Value::Array(arr) => arr
                .borrow_mut()
                .pop()
                .ok_or_else(|| RuntimeError::new("pop() on empty array")),
            _ => Err(RuntimeError::new("pop() requires array")),
        });

        // Math functions
        self.define_builtin("abs", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Int(n.abs())),
            Value::Float(n) => Ok(Value::Float(n.abs())),
            _ => Err(RuntimeError::new("abs() requires number")),
        });

        self.define_builtin("sqrt", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
            Value::Float(n) => Ok(Value::Float(n.sqrt())),
            _ => Err(RuntimeError::new("sqrt() requires number")),
        });

        self.define_builtin("sin", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).sin())),
            Value::Float(n) => Ok(Value::Float(n.sin())),
            _ => Err(RuntimeError::new("sin() requires number")),
        });

        self.define_builtin("cos", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).cos())),
            Value::Float(n) => Ok(Value::Float(n.cos())),
            _ => Err(RuntimeError::new("cos() requires number")),
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

        // Box::new - just return the value (Sigil is GC'd)
        self.globals.borrow_mut().define(
            "Box·new".to_string(),
            Value::BuiltIn(Rc::new(BuiltInFn {
                name: "Box·new".to_string(),
                arity: Some(1),
                func: |_, args| Ok(args[0].clone()),
            })),
        );

        // Map::new - create empty map
        self.globals.borrow_mut().define(
            "Map·new".to_string(),
            Value::BuiltIn(Rc::new(BuiltInFn {
                name: "Map·new".to_string(),
                arity: Some(0),
                func: |_, _| Ok(Value::Map(Rc::new(RefCell::new(HashMap::new())))),
            })),
        );

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

        // Look for main function and execute it (only if it takes no args)
        let main_fn = self.globals.borrow().get("main").and_then(|v| {
            if let Value::Function(f) = v {
                Some(f.clone())
            } else {
                None
            }
        });
        if let Some(f) = main_fn {
            // Only auto-call main if it takes no arguments
            // If main expects args, caller should call it explicitly via call_function_by_name
            if f.params.is_empty() {
                result = self.call_function(&f, vec![])?;
            }
        }

        Ok(result)
    }

    fn execute_item(&mut self, item: &Item) -> Result<Value, RuntimeError> {
        match item {
            Item::Function(func) => {
                let fn_value = self.create_function(func)?;
                self.globals
                    .borrow_mut()
                    .define(func.name.name.clone(), fn_value);
                Ok(Value::Null)
            }
            Item::Struct(s) => {
                self.types
                    .insert(s.name.name.clone(), TypeDef::Struct(s.clone()));

                // Check for #[derive(Default)] attribute and store for later lookup
                let has_default = s.attrs.derives.iter().any(|d| matches!(d, DeriveTrait::Default));
                if has_default {
                    self.default_structs.insert(s.name.name.clone(), s.clone());
                }

                Ok(Value::Null)
            }
            Item::Enum(e) => {
                self.types
                    .insert(e.name.name.clone(), TypeDef::Enum(e.clone()));

                // Register variant constructors as EnumName·VariantName
                // Store them in a lookup table that the variant_constructor builtin can use
                let enum_name = e.name.name.clone();
                for variant in &e.variants {
                    let variant_name = variant.name.name.clone();
                    let qualified_name = format!("{}·{}", enum_name, variant_name);

                    let arity = match &variant.fields {
                        crate::ast::StructFields::Unit => 0,
                        crate::ast::StructFields::Tuple(types) => types.len(),
                        crate::ast::StructFields::Named(fields) => fields.len(),
                    };

                    // Store variant info for later lookup
                    self.variant_constructors.insert(qualified_name.clone(), (enum_name.clone(), variant_name.clone(), arity));
                }
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
            Item::ExternBlock(extern_block) => {
                // Register extern functions as builtins
                for item in &extern_block.items {
                    if let ExternItem::Function(func) = item {
                        let name = func.name.name.clone();
                        // Register emulated FFI functions
                        match name.as_str() {
                            "sigil_read_file" => {
                                self.define_builtin("sigil_read_file", Some(2), |_, args| {
                                    // args[0] = path pointer (we'll use string), args[1] = len
                                    let path = match &args[0] {
                                        Value::String(s) => (**s).clone(),
                                        _ => return Err(RuntimeError::new("sigil_read_file expects string path")),
                                    };
                                    match std::fs::read_to_string(&path) {
                                        Ok(content) => {
                                            // Store content in a global for sigil_file_len to access
                                            Ok(Value::String(Rc::new(content)))
                                        }
                                        Err(_) => Ok(Value::Null),
                                    }
                                });
                            }
                            "sigil_file_len" => {
                                self.define_builtin("sigil_file_len", Some(0), |_, _| {
                                    // This is a placeholder - in real usage, would track last read
                                    Ok(Value::Int(0))
                                });
                            }
                            "sigil_write_file" => {
                                self.define_builtin("sigil_write_file", Some(4), |_, args| {
                                    let path = match &args[0] {
                                        Value::String(s) => (**s).clone(),
                                        _ => return Err(RuntimeError::new("sigil_write_file expects string path")),
                                    };
                                    let content = match &args[2] {
                                        Value::String(s) => (**s).clone(),
                                        _ => return Err(RuntimeError::new("sigil_write_file expects string content")),
                                    };
                                    match std::fs::write(&path, &content) {
                                        Ok(_) => Ok(Value::Bool(true)),
                                        Err(_) => Ok(Value::Bool(false)),
                                    }
                                });
                            }
                            "write" => {
                                self.define_builtin("write", Some(3), |_, args| {
                                    // write(fd, buf, count)
                                    let fd = match &args[0] {
                                        Value::Int(n) => *n,
                                        _ => 1,
                                    };
                                    let content = match &args[1] {
                                        Value::String(s) => (**s).clone(),
                                        _ => format!("{}", args[1]),
                                    };
                                    if fd == 1 {
                                        print!("{}", content);
                                    } else if fd == 2 {
                                        eprint!("{}", content);
                                    }
                                    Ok(Value::Int(content.len() as i64))
                                });
                            }
                            _ => {
                                // Unknown extern function - register a no-op
                            }
                        }
                    }
                }
                Ok(Value::Null)
            }
            Item::Impl(impl_block) => {
                // Extract type name from self_ty
                let type_name = match &impl_block.self_ty {
                    TypeExpr::Path(path) => {
                        path.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::")
                    }
                    _ => return Ok(Value::Null), // Can't handle complex types
                };

                // Register each method with qualified name TypeName·method
                for impl_item in &impl_block.items {
                    if let ImplItem::Function(func) = impl_item {
                        let fn_value = self.create_function(func)?;
                        let qualified_name = format!("{}·{}", type_name, func.name.name);
                        // Debug: track Lexer method registration
                        if type_name == "Lexer" && func.name.name.contains("keyword") {
                            eprintln!("DEBUG registering: {}", qualified_name);
                        }
                        self.globals.borrow_mut().define(qualified_name, fn_value);
                    }
                }
                Ok(Value::Null)
            }
            _ => Ok(Value::Null), // Skip other items for now
        }
    }

    fn create_function(&self, func: &crate::ast::Function) -> Result<Value, RuntimeError> {
        let params: Vec<String> = func
            .params
            .iter()
            .map(|p| match &p.pattern {
                Pattern::Ident { name, .. } => name.name.clone(),
                _ => "_".to_string(),
            })
            .collect();

        let body = func
            .body
            .as_ref()
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
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => self.eval_if(condition, then_branch, else_branch),
            Expr::Match { expr, arms } => self.eval_match(expr, arms),
            Expr::For {
                pattern,
                iter,
                body,
            } => self.eval_for(pattern, iter, body),
            Expr::While { condition, body } => self.eval_while(condition, body),
            Expr::Loop(body) => self.eval_loop(body),
            Expr::Return(value) => self.eval_return(value),
            Expr::Break(value) => self.eval_break(value),
            Expr::Continue => Err(RuntimeError::new("continue outside loop")),
            Expr::Index { expr, index } => self.eval_index(expr, index),
            Expr::Field { expr, field } => self.eval_field(expr, field),
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => self.eval_method_call(receiver, method, args),
            // Polysynthetic incorporation: path·file·read·string
            // Each segment is a method/function that transforms the value
            Expr::Incorporation { segments } => self.eval_incorporation(segments),
            Expr::Pipe { expr, operations } => self.eval_pipe(expr, operations),
            Expr::Closure { params, body } => self.eval_closure(params, body),
            Expr::Struct { path, fields, rest } => self.eval_struct_literal(path, fields, rest),
            Expr::Evidential {
                expr,
                evidentiality,
            } => self.eval_evidential(expr, evidentiality),
            Expr::Range {
                start,
                end,
                inclusive,
            } => self.eval_range(start, end, *inclusive),
            Expr::Assign { target, value } => self.eval_assign(target, value),
            Expr::Let { pattern, value } => {
                // Let expression (for if-let, while-let patterns)
                // Evaluate the value and bind it to the pattern
                let val = self.evaluate(value)?;
                self.bind_pattern(pattern, val.clone())?;
                Ok(val)
            }
            Expr::Await {
                expr: inner,
                evidentiality,
            } => {
                let value = self.evaluate(inner)?;
                let awaited = self.await_value(value)?;
                // Handle evidentiality marker semantics
                match evidentiality {
                    Some(Evidentiality::Uncertain) => {
                        // ⌛? - propagate error like Try
                        self.unwrap_result_or_option(awaited, true, false)
                    }
                    Some(Evidentiality::Known) => {
                        // ⌛! - expect success, panic on error
                        self.unwrap_result_or_option(awaited, true, true)
                    }
                    Some(Evidentiality::Reported) | Some(Evidentiality::Paradox) => {
                        // ⌛~ or ⌛‽ - mark as external/reported, unwrap if Result/Option
                        self.unwrap_result_or_option(awaited, false, false)
                    }
                    None => Ok(awaited),
                }
            }
            // Macro invocations: format!(...), println!(...), etc.
            Expr::Macro { path, tokens } => {
                let macro_name = path.segments.last()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");

                match macro_name {
                    "format" => self.eval_format_macro(tokens),
                    "println" => {
                        let formatted = self.eval_format_macro(tokens)?;
                        if let Value::String(s) = formatted {
                            println!("{}", s);
                        }
                        Ok(Value::Null)
                    }
                    "eprintln" => {
                        let formatted = self.eval_format_macro(tokens)?;
                        if let Value::String(s) = formatted {
                            eprintln!("{}", s);
                        }
                        Ok(Value::Null)
                    }
                    "print" => {
                        let formatted = self.eval_format_macro(tokens)?;
                        if let Value::String(s) = formatted {
                            print!("{}", s);
                        }
                        Ok(Value::Null)
                    }
                    "eprint" => {
                        let formatted = self.eval_format_macro(tokens)?;
                        if let Value::String(s) = formatted {
                            eprint!("{}", s);
                        }
                        Ok(Value::Null)
                    }
                    "vec" => {
                        // vec![a, b, c] - parse elements and create array
                        self.eval_vec_macro(tokens)
                    }
                    "panic" => {
                        let formatted = self.eval_format_macro(tokens)?;
                        let msg = if let Value::String(s) = formatted {
                            s.to_string()
                        } else {
                            "panic!".to_string()
                        };
                        Err(RuntimeError::new(format!("panic: {}", msg)))
                    }
                    "assert" => {
                        // Simple assert - just evaluate the expression
                        let condition = self.eval_format_macro(tokens)?;
                        if self.is_truthy(&condition) {
                            Ok(Value::Null)
                        } else {
                            Err(RuntimeError::new("assertion failed"))
                        }
                    }
                    _ => {
                        // Unknown macro - return tokens as string for debugging
                        Ok(Value::String(Rc::new(tokens.clone())))
                    }
                }
            }
            // Unsafe block - just evaluate the block normally
            Expr::Unsafe(block) => self.eval_block(block),
            // Try expression: expr?
            Expr::Try(inner) => {
                let value = self.evaluate(inner)?;
                // If Result::Err or None, propagate the error
                // If Result::Ok or Some, unwrap the value
                match &value {
                    Value::Variant { enum_name, variant_name, fields } => {
                        match (enum_name.as_str(), variant_name.as_str()) {
                            ("Result", "Ok") => {
                                if let Some(f) = fields {
                                    Ok(f.first().cloned().unwrap_or(Value::Null))
                                } else {
                                    Ok(Value::Null)
                                }
                            }
                            ("Result", "Err") => {
                                let err_msg = if let Some(f) = fields {
                                    let first = f.first().cloned().unwrap_or(Value::Null);
                                    // Try to get more detail from the error
                                    match &first {
                                        Value::Struct { name, fields: sf } => {
                                            let field_str = sf.borrow().iter()
                                                .map(|(k, v)| format!("{}: {}", k, v))
                                                .collect::<Vec<_>>()
                                                .join(", ");
                                            format!("{} {{ {} }}", name, field_str)
                                        }
                                        Value::Variant { enum_name: en, variant_name: vn, fields: vf } => {
                                            let vf_str = vf.as_ref().map(|vs|
                                                vs.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(", ")
                                            ).unwrap_or_default();
                                            format!("{}::{} {{ {} }}", en, vn, vf_str)
                                        }
                                        _ => format!("{}", first)
                                    }
                                } else {
                                    "error".to_string()
                                };
                                Err(RuntimeError::new(format!("try failed: {}", err_msg)))
                            }
                            ("Option", "Some") => {
                                if let Some(f) = fields {
                                    Ok(f.first().cloned().unwrap_or(Value::Null))
                                } else {
                                    Ok(Value::Null)
                                }
                            }
                            ("Option", "None") => {
                                Err(RuntimeError::new("try failed: None"))
                            }
                            _ => Ok(value), // Not a Result/Option, pass through
                        }
                    }
                    _ => Ok(value), // Not a variant, pass through
                }
            }
            _ => Err(RuntimeError::new(format!(
                "Unsupported expression: {:?}",
                expr
            ))),
        }
    }

    fn eval_assign(&mut self, target: &Expr, value: &Expr) -> Result<Value, RuntimeError> {
        let val = self.evaluate(value)?;

        match target {
            Expr::Path(path) if path.segments.len() == 1 => {
                let name = &path.segments[0].ident.name;
                self.environment.borrow_mut().set(name, val.clone())?;
                Ok(val)
            }
            Expr::Index { expr, index } => {
                // Array/map index assignment
                let idx = self.evaluate(index)?;
                let idx = match idx {
                    Value::Int(i) => i as usize,
                    _ => return Err(RuntimeError::new("Index must be an integer")),
                };

                // Get the array and modify it
                if let Expr::Path(path) = expr.as_ref() {
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident.name;
                        let current = self.environment.borrow().get(name).ok_or_else(|| {
                            RuntimeError::new(format!("Undefined variable: {}", name))
                        })?;

                        if let Value::Array(arr) = current {
                            let borrowed = arr.borrow();
                            let mut new_arr = borrowed.clone();
                            drop(borrowed);
                            if idx < new_arr.len() {
                                new_arr[idx] = val.clone();
                                self.environment
                                    .borrow_mut()
                                    .set(name, Value::Array(Rc::new(RefCell::new(new_arr))))?;
                                return Ok(val);
                            }
                        }
                    }
                }
                Err(RuntimeError::new("Invalid index assignment target"))
            }
            Expr::Field { expr, field } => {
                // Field assignment: struct.field = value
                // Need to find the variable and update its field
                match expr.as_ref() {
                    Expr::Path(path) if path.segments.len() == 1 => {
                        let var_name = &path.segments[0].ident.name;
                        let current = self.environment.borrow().get(var_name).ok_or_else(|| {
                            RuntimeError::new(format!("Undefined variable: {}", var_name))
                        })?;

                        match current {
                            Value::Struct { fields, .. } => {
                                fields.borrow_mut().insert(field.name.clone(), val.clone());
                                Ok(val)
                            }
                            Value::Ref(r) => {
                                let mut borrowed = r.borrow_mut();
                                if let Value::Struct { fields, .. } = &mut *borrowed {
                                    fields.borrow_mut().insert(field.name.clone(), val.clone());
                                    Ok(val)
                                } else {
                                    Err(RuntimeError::new("Cannot assign field on non-struct ref"))
                                }
                            }
                            _ => Err(RuntimeError::new("Cannot assign field on non-struct")),
                        }
                    }
                    _ => {
                        // For now, just evaluate and try to update (won't persist for non-path exprs)
                        let struct_val = self.evaluate(expr)?;
                        match struct_val {
                            Value::Struct { fields, .. } => {
                                fields.borrow_mut().insert(field.name.clone(), val.clone());
                                Ok(val)
                            }
                            Value::Ref(r) => {
                                let mut borrowed = r.borrow_mut();
                                if let Value::Struct { fields, .. } = &mut *borrowed {
                                    fields.borrow_mut().insert(field.name.clone(), val.clone());
                                    Ok(val)
                                } else {
                                    Err(RuntimeError::new("Cannot assign field on non-struct"))
                                }
                            }
                            _ => Err(RuntimeError::new("Cannot assign field on non-struct")),
                        }
                    }
                }
            }
            _ => Err(RuntimeError::new("Invalid assignment target")),
        }
    }

    fn eval_literal(&mut self, lit: &Literal) -> Result<Value, RuntimeError> {
        match lit {
            Literal::Int { value, base, .. } => {
                let n = self.parse_int(value, base)?;
                Ok(Value::Int(n))
            }
            Literal::Float { value, .. } => {
                let n: f64 = value
                    .parse()
                    .map_err(|_| RuntimeError::new(format!("Invalid float: {}", value)))?;
                Ok(Value::Float(n))
            }
            Literal::String(s) => Ok(Value::String(Rc::new(s.clone()))),
            Literal::MultiLineString(s) => Ok(Value::String(Rc::new(s.clone()))),
            Literal::RawString(s) => Ok(Value::String(Rc::new(s.clone()))),
            Literal::ByteString(bytes) => {
                // Convert byte array to an array of integers
                let arr: Vec<Value> = bytes.iter().map(|&b| Value::Int(b as i64)).collect();
                Ok(Value::Array(Rc::new(RefCell::new(arr))))
            }
            Literal::InterpolatedString { parts } => {
                // Evaluate each part and concatenate, tracking evidentiality
                let mut result = String::new();
                let mut combined_evidence: Option<Evidence> = None;

                for part in parts {
                    match part {
                        InterpolationPart::Text(s) => result.push_str(s),
                        InterpolationPart::Expr(expr) => {
                            let value = self.evaluate(expr)?;

                            // Track explicit evidentiality
                            combined_evidence = Self::combine_evidence(
                                combined_evidence,
                                Self::extract_evidence(&value),
                            );

                            // Track affect-derived evidentiality (sarcasm, confidence)
                            if let Some(affect) = Self::extract_affect(&value) {
                                combined_evidence = Self::combine_evidence(
                                    combined_evidence,
                                    Self::affect_to_evidence(affect),
                                );
                            }

                            // Use the fully unwrapped value for display
                            let display_value = Self::unwrap_value(&value);
                            result.push_str(&format!("{}", display_value));
                        }
                    }
                }

                // Wrap result with evidentiality if any interpolated values were evidential
                let string_value = Value::String(Rc::new(result));
                match combined_evidence {
                    Some(evidence) => Ok(Value::Evidential {
                        value: Box::new(string_value),
                        evidence,
                    }),
                    None => Ok(string_value),
                }
            }
            Literal::SigilStringSql(s) => {
                // SQL sigil string - for now just return as string
                // Future: could add SQL validation or templating
                Ok(Value::String(Rc::new(s.clone())))
            }
            Literal::SigilStringRoute(s) => {
                // Route sigil string - for now just return as string
                // Future: could add route validation or templating
                Ok(Value::String(Rc::new(s.clone())))
            }
            Literal::Char(c) => Ok(Value::Char(*c)),
            Literal::ByteChar(b) => Ok(Value::Int(*b as i64)),
            Literal::Bool(b) => Ok(Value::Bool(*b)),
            Literal::Null => Ok(Value::Null),
            Literal::Empty => Ok(Value::Empty),
            Literal::Infinity => Ok(Value::Infinity),
            Literal::Circle => Ok(Value::Int(0)), // ◯ = zero
        }
    }

    fn parse_int(&self, value: &str, base: &NumBase) -> Result<i64, RuntimeError> {
        let (radix, prefix_len) = match base {
            NumBase::Binary => (2, 2), // 0b
            NumBase::Octal => (8, 2),  // 0o
            NumBase::Decimal => (10, 0),
            NumBase::Hex => (16, 2),         // 0x
            NumBase::Vigesimal => (20, 2),   // 0v
            NumBase::Sexagesimal => (60, 2), // 0s
            NumBase::Duodecimal => (12, 2),  // 0z
            NumBase::Explicit(b) => (*b as u32, 0),
        };

        let clean = value[prefix_len..].replace('_', "");
        i64::from_str_radix(&clean, radix)
            .map_err(|_| RuntimeError::new(format!("Invalid integer: {}", value)))
    }

    fn eval_path(&self, path: &TypePath) -> Result<Value, RuntimeError> {
        if path.segments.len() == 1 {
            let name = &path.segments[0].ident.name;
            // Handle wildcard "_" - return Null instead of looking it up
            if name == "_" {
                return Ok(Value::Null);
            }
            self.environment
                .borrow()
                .get(name)
                .ok_or_else(|| RuntimeError::new(format!("Undefined variable: {}", name)))
        } else {
            // Multi-segment path (module::item or Type·method)
            // Try full qualified name first (joined with ·)
            let full_name = path
                .segments
                .iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("·");

            if let Some(val) = self.environment.borrow().get(&full_name) {
                return Ok(val);
            }

            // Try looking up the last segment (for Math·sqrt -> sqrt)
            let last_name = &path.segments.last().unwrap().ident.name;
            if let Some(val) = self.environment.borrow().get(last_name) {
                return Ok(val);
            }

            // Check for enum variant syntax (EnumName::Variant)
            if path.segments.len() == 2 {
                let type_name = &path.segments[0].ident.name;
                let variant_name = &path.segments[1].ident.name;

                // Check if this is an enum variant
                if let Some(TypeDef::Enum(enum_def)) = self.types.get(type_name) {
                    for variant in &enum_def.variants {
                        if &variant.name.name == variant_name {
                            // Return a variant constructor or unit variant
                            if matches!(variant.fields, crate::ast::StructFields::Unit) {
                                return Ok(Value::Variant {
                                    enum_name: type_name.clone(),
                                    variant_name: variant_name.clone(),
                                    fields: None,
                                });
                            }
                        }
                    }
                }
            }

            // Check for variant constructor in variant_constructors table
            if let Some((enum_name, variant_name, arity)) = self.variant_constructors.get(&full_name).cloned() {
                // Return a special marker that eval_call can recognize
                // For unit variants, return the variant directly
                if arity == 0 {
                    return Ok(Value::Variant {
                        enum_name,
                        variant_name,
                        fields: None,
                    });
                }
                // For variants with fields, we need to return something callable
                // We'll use a special builtin-like marker
                // Actually, let's just let eval_call handle it via call_function_by_name
            }

            Err(RuntimeError::new(format!(
                "Undefined: {} (tried {} and {})",
                full_name, full_name, last_name
            )))
        }
    }

    fn eval_binary(
        &mut self,
        left: &Expr,
        op: &BinOp,
        right: &Expr,
    ) -> Result<Value, RuntimeError> {
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

        // Unwrap all wrappers (evidential/affective/ref) for binary operations
        let lhs = Self::unwrap_all(&lhs);
        let rhs = Self::unwrap_all(&rhs);

        match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => self.int_binary_op(a, b, op),
            (Value::Float(a), Value::Float(b)) => self.float_binary_op(a, b, op),
            (Value::Int(a), Value::Float(b)) => self.float_binary_op(a as f64, b, op),
            (Value::Float(a), Value::Int(b)) => self.float_binary_op(a, b as f64, op),
            (Value::String(a), Value::String(b)) => match op {
                BinOp::Add | BinOp::Concat => Ok(Value::String(Rc::new(format!("{}{}", a, b)))),
                BinOp::Eq => Ok(Value::Bool(*a == *b)),
                BinOp::Ne => Ok(Value::Bool(*a != *b)),
                _ => Err(RuntimeError::new("Invalid string operation")),
            },
            (Value::Bool(a), Value::Bool(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(a == b)),
                BinOp::Ne => Ok(Value::Bool(a != b)),
                _ => Err(RuntimeError::new("Invalid boolean operation")),
            },
            (Value::Array(a), Value::Array(b)) => match op {
                BinOp::Concat => {
                    let mut result = a.borrow().clone();
                    result.extend(b.borrow().iter().cloned());
                    Ok(Value::Array(Rc::new(RefCell::new(result))))
                }
                BinOp::Eq => Ok(Value::Bool(Rc::ptr_eq(&a, &b))),
                BinOp::Ne => Ok(Value::Bool(!Rc::ptr_eq(&a, &b))),
                _ => Err(RuntimeError::new("Invalid array operation")),
            },
            // Null equality
            (Value::Null, Value::Null) => match op {
                BinOp::Eq => Ok(Value::Bool(true)),
                BinOp::Ne => Ok(Value::Bool(false)),
                _ => Err(RuntimeError::new("Invalid null operation")),
            },
            (Value::Null, _) | (_, Value::Null) => match op {
                BinOp::Eq => Ok(Value::Bool(false)),
                BinOp::Ne => Ok(Value::Bool(true)),
                _ => Err(RuntimeError::new("Invalid null operation")),
            },
            // Char comparisons
            (Value::Char(a), Value::Char(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(a == b)),
                BinOp::Ne => Ok(Value::Bool(a != b)),
                BinOp::Lt => Ok(Value::Bool(a < b)),
                BinOp::Le => Ok(Value::Bool(a <= b)),
                BinOp::Gt => Ok(Value::Bool(a > b)),
                BinOp::Ge => Ok(Value::Bool(a >= b)),
                _ => Err(RuntimeError::new("Invalid char operation")),
            },
            // String and char operations
            (Value::String(a), Value::Char(b)) => match op {
                BinOp::Add | BinOp::Concat => Ok(Value::String(Rc::new(format!("{}{}", a, b)))),
                _ => Err(RuntimeError::new("Invalid string/char operation")),
            },
            (Value::Char(a), Value::String(b)) => match op {
                BinOp::Add | BinOp::Concat => Ok(Value::String(Rc::new(format!("{}{}", a, b)))),
                _ => Err(RuntimeError::new("Invalid char/string operation")),
            },
            // Variant equality
            (Value::Variant { enum_name: e1, variant_name: v1, fields: f1 },
             Value::Variant { enum_name: e2, variant_name: v2, fields: f2 }) => match op {
                BinOp::Eq => {
                    let eq = e1 == e2 && v1 == v2 && match (f1, f2) {
                        (None, None) => true,
                        (Some(a), Some(b)) => Rc::ptr_eq(&a, &b),
                        _ => false,
                    };
                    Ok(Value::Bool(eq))
                }
                BinOp::Ne => {
                    let eq = e1 == e2 && v1 == v2 && match (f1, f2) {
                        (None, None) => true,
                        (Some(a), Some(b)) => Rc::ptr_eq(&a, &b),
                        _ => false,
                    };
                    Ok(Value::Bool(!eq))
                }
                _ => Err(RuntimeError::new("Invalid variant operation")),
            },
            // Struct equality (by reference)
            (Value::Struct { name: n1, fields: f1 }, Value::Struct { name: n2, fields: f2 }) => match op {
                BinOp::Eq => Ok(Value::Bool(n1 == n2 && Rc::ptr_eq(&f1, &f2))),
                BinOp::Ne => Ok(Value::Bool(n1 != n2 || !Rc::ptr_eq(&f1, &f2))),
                _ => Err(RuntimeError::new("Invalid struct operation")),
            },
            (l, r) => Err(RuntimeError::new(format!(
                "Type mismatch in binary operation: {:?} {:?} {:?}",
                l, op, r
            ))),
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
        match (op, &val) {
            (UnaryOp::Neg, Value::Int(n)) => Ok(Value::Int(-n)),
            (UnaryOp::Neg, Value::Float(n)) => Ok(Value::Float(-n)),
            (UnaryOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
            (UnaryOp::Not, Value::Int(n)) => Ok(Value::Int(!n)),
            (UnaryOp::Ref, _) => Ok(Value::Ref(Rc::new(RefCell::new(val)))),
            (UnaryOp::RefMut, _) => Ok(Value::Ref(Rc::new(RefCell::new(val)))),
            (UnaryOp::Deref, Value::Ref(r)) => Ok(r.borrow().clone()),
            (UnaryOp::Deref, other) => {
                // Try to unwrap evidential/affective wrappers and deref
                let unwrapped = Self::unwrap_all(&val);
                if let Value::Ref(r) = &unwrapped {
                    return Ok(r.borrow().clone());
                }
                // For non-ref types in interpreted code, just return the value as-is
                // (dereferencing a copy type in Sigil is a no-op)
                Ok(unwrapped)
            }
            _ => Err(RuntimeError::new(format!("Invalid unary {:?} on {:?}", op, std::mem::discriminant(&val)))),
        }
    }

    fn eval_call(&mut self, func_expr: &Expr, args: &[Expr]) -> Result<Value, RuntimeError> {
        // Check if func_expr is a path that might be a variant constructor
        if let Expr::Path(path) = func_expr {
            let qualified_name = path.segments.iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("·");

            // Check for TypeName·default pattern
            if qualified_name.ends_with("·default") && args.is_empty() {
                let type_name = qualified_name.strip_suffix("·default").unwrap();
                if let Some(struct_def) = self.default_structs.get(type_name).cloned() {
                    let mut fields = HashMap::new();
                    if let StructFields::Named(field_defs) = &struct_def.fields {
                        for field in field_defs {
                            let default_val = if let Some(default_expr) = &field.default {
                                self.evaluate(default_expr)?
                            } else {
                                // No default - use null for optional/uncertain types
                                Value::Null
                            };
                            fields.insert(field.name.name.clone(), default_val);
                        }
                    }
                    return Ok(Value::Struct {
                        name: type_name.to_string(),
                        fields: Rc::new(RefCell::new(fields)),
                    });
                }
            }

            // Check variant constructors
            if let Some((enum_name, variant_name, arity)) = self.variant_constructors.get(&qualified_name).cloned() {
                let arg_values: Vec<Value> = args
                    .iter()
                    .map(|a| self.evaluate(a))
                    .collect::<Result<_, _>>()?;

                if arg_values.len() != arity {
                    return Err(RuntimeError::new(format!(
                        "{} expects {} arguments, got {}",
                        qualified_name, arity, arg_values.len()
                    )));
                }

                if arity == 0 {
                    return Ok(Value::Variant {
                        enum_name,
                        variant_name,
                        fields: None,
                    });
                } else {
                    return Ok(Value::Variant {
                        enum_name,
                        variant_name,
                        fields: Some(Rc::new(arg_values)),
                    });
                }
            }
        }

        let func = self.evaluate(func_expr)?;
        let arg_values: Vec<Value> = args
            .iter()
            .map(|a| self.evaluate(a))
            .collect::<Result<_, _>>()?;

        match func {
            Value::Function(f) => self.call_function(&f, arg_values),
            Value::BuiltIn(b) => self.call_builtin(&b, arg_values),
            _ => Err(RuntimeError::new("Cannot call non-function")),
        }
    }

    pub fn call_function(
        &mut self,
        func: &Function,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        if args.len() != func.params.len() {
            return Err(RuntimeError::new(format!(
                "Expected {} arguments, got {}",
                func.params.len(),
                args.len()
            )));
        }

        // Debug: trace calls to keyword_or_ident
        if func.params.iter().any(|p| p == "name") {
            for arg in &args {
                let unwrapped = Self::unwrap_all(arg);
                if let Value::String(s) = &unwrapped {
                    if s.len() <= 10 {
                        eprintln!("DEBUG call_function(name='{}')", s);
                    }
                }
            }
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

    fn call_builtin(
        &mut self,
        builtin: &BuiltInFn,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        if let Some(arity) = builtin.arity {
            if args.len() != arity {
                return Err(RuntimeError::new(format!(
                    "{}() expects {} arguments, got {}",
                    builtin.name,
                    arity,
                    args.len()
                )));
            }
        }
        (builtin.func)(self, args)
    }

    /// Await a value - if it's a future, resolve it; otherwise return as-is
    pub fn await_value(&mut self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Future(fut) => {
                let mut fut_inner = fut.borrow_mut();
                self.poll_future(&mut fut_inner)
            }
            // Non-futures return immediately
            other => Ok(other),
        }
    }

    /// Unwrap a Result or Option value with configurable error handling
    /// - propagate_errors: if true, return error on Err/None; if false, just unwrap
    /// - panic_on_error: if true, panic instead of returning error
    fn unwrap_result_or_option(
        &self,
        value: Value,
        propagate_errors: bool,
        panic_on_error: bool,
    ) -> Result<Value, RuntimeError> {
        // First, determine what kind of value we have and extract any inner value
        let (is_ok_or_some, is_err, is_none, inner_val) = match &value {
            Value::Struct { name, fields } if name == "Ok" || name == "Some" => {
                let borrowed = fields.borrow();
                let inner = borrowed.get("0").or(borrowed.get("value")).cloned();
                (true, false, false, inner)
            }
            Value::Struct { name, fields } if name == "Err" => {
                let borrowed = fields.borrow();
                let inner = borrowed.get("0").or(borrowed.get("value")).cloned();
                (false, true, false, inner)
            }
            Value::Struct { name, .. } if name == "None" => (false, false, true, None),
            _ => return Ok(value),
        };

        if is_ok_or_some {
            Ok(inner_val.unwrap_or(value))
        } else if is_err {
            let msg = format!("Error: {:?}", inner_val);
            if panic_on_error {
                panic!("{}", msg);
            } else if propagate_errors {
                Err(RuntimeError::new(msg))
            } else {
                Ok(inner_val.unwrap_or(value))
            }
        } else if is_none {
            if panic_on_error {
                panic!("Unwrapped None");
            } else if propagate_errors {
                Err(RuntimeError::new("Unwrapped None".to_string()))
            } else {
                Ok(value)
            }
        } else {
            Ok(value)
        }
    }

    /// Poll a future to completion
    fn poll_future(&mut self, fut: &mut FutureInner) -> Result<Value, RuntimeError> {
        // Check if already resolved
        match &fut.state {
            FutureState::Ready(v) => return Ok((**v).clone()),
            FutureState::Failed(e) => return Err(RuntimeError::new(e.clone())),
            _ => {}
        }

        // Check if it's a timer future
        if let Some(complete_at) = fut.complete_at {
            if std::time::Instant::now() >= complete_at {
                fut.state = FutureState::Ready(Box::new(Value::Null));
                return Ok(Value::Null);
            } else {
                // Timer not complete - in interpreter, we just sleep
                let remaining = complete_at - std::time::Instant::now();
                std::thread::sleep(remaining);
                fut.state = FutureState::Ready(Box::new(Value::Null));
                return Ok(Value::Null);
            }
        }

        // Execute computation if pending
        if let Some(computation) = fut.computation.take() {
            fut.state = FutureState::Running;

            match computation {
                FutureComputation::Immediate(v) => {
                    fut.state = FutureState::Ready(v.clone());
                    Ok((*v).clone())
                }
                FutureComputation::Timer(duration) => {
                    // Sleep for the duration
                    std::thread::sleep(duration);
                    fut.state = FutureState::Ready(Box::new(Value::Null));
                    Ok(Value::Null)
                }
                FutureComputation::Lazy { func, args } => {
                    // Execute the function
                    match self.call_function(&func, args) {
                        Ok(result) => {
                            fut.state = FutureState::Ready(Box::new(result.clone()));
                            Ok(result)
                        }
                        Err(e) => {
                            fut.state = FutureState::Failed(e.message.clone());
                            Err(e)
                        }
                    }
                }
                FutureComputation::Join(futures) => {
                    // Await all futures and collect results
                    let mut results = Vec::new();
                    for f in futures {
                        let mut f_inner = f.borrow_mut();
                        results.push(self.poll_future(&mut f_inner)?);
                    }
                    let result = Value::Array(Rc::new(RefCell::new(results)));
                    fut.state = FutureState::Ready(Box::new(result.clone()));
                    Ok(result)
                }
                FutureComputation::Race(futures) => {
                    // Return first completed future
                    // In interpreter, just poll in order
                    for f in futures {
                        let f_inner = f.borrow_mut();
                        if matches!(f_inner.state, FutureState::Ready(_)) {
                            if let FutureState::Ready(v) = &f_inner.state {
                                fut.state = FutureState::Ready(v.clone());
                                return Ok((**v).clone());
                            }
                        }
                    }
                    // None ready, poll first one
                    Err(RuntimeError::new("No futures ready in race"))
                }
            }
        } else {
            // No computation - return current state
            match &fut.state {
                FutureState::Ready(v) => Ok((**v).clone()),
                FutureState::Failed(e) => Err(RuntimeError::new(e.clone())),
                _ => Err(RuntimeError::new("Future has no computation")),
            }
        }
    }

    /// Create a new future from a value
    pub fn make_future_immediate(&self, value: Value) -> Value {
        Value::Future(Rc::new(RefCell::new(FutureInner {
            state: FutureState::Ready(Box::new(value)),
            computation: None,
            complete_at: None,
        })))
    }

    /// Create a pending future with lazy computation
    pub fn make_future_lazy(&self, func: Rc<Function>, args: Vec<Value>) -> Value {
        Value::Future(Rc::new(RefCell::new(FutureInner {
            state: FutureState::Pending,
            computation: Some(FutureComputation::Lazy { func, args }),
            complete_at: None,
        })))
    }

    /// Create a timer future
    pub fn make_future_timer(&self, duration: std::time::Duration) -> Value {
        Value::Future(Rc::new(RefCell::new(FutureInner {
            state: FutureState::Pending,
            computation: Some(FutureComputation::Timer(duration)),
            complete_at: Some(std::time::Instant::now() + duration),
        })))
    }

    fn eval_array(&mut self, elements: &[Expr]) -> Result<Value, RuntimeError> {
        let values: Vec<Value> = elements
            .iter()
            .map(|e| self.evaluate(e))
            .collect::<Result<_, _>>()?;
        Ok(Value::Array(Rc::new(RefCell::new(values))))
    }

    fn eval_tuple(&mut self, elements: &[Expr]) -> Result<Value, RuntimeError> {
        let values: Vec<Value> = elements
            .iter()
            .map(|e| self.evaluate(e))
            .collect::<Result<_, _>>()?;
        Ok(Value::Tuple(Rc::new(values)))
    }

    fn eval_block(&mut self, block: &Block) -> Result<Value, RuntimeError> {
        let env = Rc::new(RefCell::new(Environment::with_parent(
            self.environment.clone(),
        )));
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
                Stmt::LetElse { pattern, init, else_branch, .. } => {
                    let value = self.evaluate(init)?;
                    // Try to bind pattern, if it fails, execute else branch
                    if self.bind_pattern(pattern, value.clone()).is_err() {
                        return self.evaluate(else_branch);
                    }
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
                // Don't bind "_" - it's a wildcard in identifier form
                if name.name != "_" {
                    self.environment
                        .borrow_mut()
                        .define(name.name.clone(), value);
                }
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
            Pattern::Struct { fields, .. } => {
                // Unwrap any wrappers first
                let unwrapped = Self::unwrap_all(&value);
                // Bind each field from the struct
                if let Value::Struct { fields: struct_fields, .. } = &unwrapped {
                    for field_pat in fields {
                        let field_name = &field_pat.name.name;
                        if let Some(field_val) = struct_fields.borrow().get(field_name) {
                            if let Some(pat) = &field_pat.pattern {
                                self.bind_pattern(pat, field_val.clone())?;
                            } else {
                                // Shorthand: foo: foo - bind to same name
                                self.environment.borrow_mut().define(field_name.clone(), field_val.clone());
                            }
                        }
                    }
                    Ok(())
                } else {
                    eprintln!("DEBUG struct pattern bind: expected struct but got {:?}", std::mem::discriminant(&unwrapped));
                    Err(RuntimeError::new("Expected struct value for struct pattern"))
                }
            }
            Pattern::Path(_path) => {
                // Path patterns like Result::Ok - unit variant patterns
                // Don't bind anything
                Ok(())
            }
            Pattern::TupleStruct { path, fields } => {
                // Enum variant with fields: Result::Ok(value)
                if let Value::Variant { variant_name, fields: variant_fields, .. } = &value {
                    let pattern_variant = path.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("");
                    if pattern_variant == variant_name {
                        // Unwrap fields and bind
                        if let Some(inner_fields) = variant_fields {
                            if fields.len() == 1 && inner_fields.len() == 1 {
                                self.bind_pattern(&fields[0], inner_fields[0].clone())?;
                            } else {
                                for (pat, val) in fields.iter().zip(inner_fields.iter()) {
                                    self.bind_pattern(pat, val.clone())?;
                                }
                            }
                        }
                    }
                    Ok(())
                } else {
                    // Maybe it's a regular tuple being matched
                    if let Value::Tuple(tuple_vals) = &value {
                        for (pat, val) in fields.iter().zip(tuple_vals.iter()) {
                            self.bind_pattern(pat, val.clone())?;
                        }
                        Ok(())
                    } else {
                        Err(RuntimeError::new("Expected variant or tuple for tuple struct pattern"))
                    }
                }
            }
            Pattern::Literal(_) => {
                // Literal patterns don't bind anything, just match
                Ok(())
            }
            Pattern::Rest => {
                // Rest pattern .. - just ignores rest of values
                Ok(())
            }
            Pattern::Range { .. } => {
                // Range patterns like 'a'..='z' don't bind anything
                Ok(())
            }
            Pattern::Or(_) => {
                // Or patterns like 'a' | 'b' | 'c' don't bind anything
                // (If they had identifiers, we'd need to ensure all arms bind the same names)
                Ok(())
            }
            _ => Err(RuntimeError::new(format!("Unsupported pattern: {:?}", pattern))),
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

        // Debug all string matches to find keyword_or_ident
        let unwrapped = Self::unwrap_all(&value);
        if let Value::String(s) = &unwrapped {
            if s.len() <= 10 {
                eprintln!("DEBUG eval_match: string='{}', arms={}", s, arms.len());
            }
        }

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
                let env = Rc::new(RefCell::new(Environment::with_parent(
                    self.environment.clone(),
                )));
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

    fn pattern_matches(&mut self, pattern: &Pattern, value: &Value) -> Result<bool, RuntimeError> {
        // Unwrap evidential/affective/ref wrappers from value
        let value = Self::unwrap_all(value);

        // Debug string pattern matching
        if let Value::String(s) = &value {
            if **s == "fn" {
                eprintln!("DEBUG pattern_matches: value='fn', pattern={:?}", pattern);
            }
        }

        match (pattern, &value) {
            (Pattern::Wildcard, _) => Ok(true),
            // Pattern::Ident with evidentiality - ?g matches Some/non-null, !g matches Known values
            (Pattern::Ident { evidentiality: Some(Evidentiality::Uncertain), .. }, val) => {
                // ?g pattern should match non-null values (i.e., Option::Some)
                match val {
                    Value::Null => Ok(false),
                    Value::Variant { variant_name, .. } if variant_name == "None" => Ok(false),
                    _ => Ok(true),
                }
            }
            (Pattern::Ident { .. }, _) => Ok(true),
            (Pattern::Literal(lit), val) => {
                let lit_val = self.eval_literal(lit)?;
                let result = self.values_equal(&lit_val, val);
                if let Value::String(s) = &value {
                    if **s == "fn" {
                        eprintln!("DEBUG literal match: pattern lit={:?}, result={}", lit, result);
                    }
                }
                Ok(result)
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
            // Path pattern - matches unit enum variants like CompileMode::Compile
            (Pattern::Path(path), Value::Variant { variant_name, fields, .. }) => {
                let pattern_variant = path.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("");
                // Match if variant name matches and has no fields
                Ok(pattern_variant == variant_name && fields.is_none())
            }
            // TupleStruct pattern - matches enum variants with data like Result::Ok(x)
            (Pattern::TupleStruct { path, fields: pat_fields }, Value::Variant { variant_name, fields, .. }) => {
                let pattern_variant = path.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("");
                if pattern_variant != variant_name {
                    return Ok(false);
                }
                // Match field patterns
                if let Some(variant_fields) = fields {
                    if pat_fields.len() != variant_fields.len() {
                        return Ok(false);
                    }
                    for (p, v) in pat_fields.iter().zip(variant_fields.iter()) {
                        if !self.pattern_matches(p, v)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                } else {
                    // Variant has no fields but pattern expects some
                    Ok(pat_fields.is_empty())
                }
            }
            // Struct pattern - matches struct values
            (Pattern::Struct { path, fields: pat_fields, rest }, Value::Struct { name: struct_name, fields: struct_fields }) => {
                let pattern_name = path.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::");
                if pattern_name != *struct_name {
                    return Ok(false);
                }
                // Check each field in the pattern
                let borrowed = struct_fields.borrow();
                for field_pat in pat_fields {
                    let field_name = &field_pat.name.name;
                    if let Some(field_val) = borrowed.get(field_name) {
                        if let Some(sub_pat) = &field_pat.pattern {
                            if !self.pattern_matches(sub_pat, field_val)? {
                                return Ok(false);
                            }
                        }
                        // If no sub-pattern, any value matches
                    } else if !rest {
                        // Field not found and no rest pattern
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            // Or pattern - match if any sub-pattern matches
            (Pattern::Or(patterns), val) => {
                for p in patterns {
                    if self.pattern_matches(p, val)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            // Rest pattern always matches
            (Pattern::Rest, _) => Ok(true),
            // Range pattern: 'a'..='z' or 0..=9
            (Pattern::Range { start, end, inclusive }, val) => {
                // Helper to extract char from pattern
                let extract_char = |pat: &Option<Box<Pattern>>| -> Option<char> {
                    match pat {
                        Some(p) => match p.as_ref() {
                            Pattern::Literal(Literal::Char(c)) => Some(*c),
                            _ => None,
                        },
                        None => None,
                    }
                };
                // Helper to extract int from pattern
                let extract_int = |pat: &Option<Box<Pattern>>| -> Option<i64> {
                    match pat {
                        Some(p) => match p.as_ref() {
                            Pattern::Literal(Literal::Int { value, .. }) => value.parse().ok(),
                            _ => None,
                        },
                        None => None,
                    }
                };

                match val {
                    Value::Char(c) => {
                        let start_val = extract_char(start);
                        let end_val = extract_char(end);
                        let in_range = match (start_val, end_val, *inclusive) {
                            (Some(s), Some(e), true) => *c >= s && *c <= e,
                            (Some(s), Some(e), false) => *c >= s && *c < e,
                            (Some(s), None, _) => *c >= s,
                            (None, Some(e), true) => *c <= e,
                            (None, Some(e), false) => *c < e,
                            (None, None, _) => true,
                        };
                        Ok(in_range)
                    }
                    Value::Int(i) => {
                        let start_val = extract_int(start);
                        let end_val = extract_int(end);
                        let in_range = match (start_val, end_val, *inclusive) {
                            (Some(s), Some(e), true) => *i >= s && *i <= e,
                            (Some(s), Some(e), false) => *i >= s && *i < e,
                            (Some(s), None, _) => *i >= s,
                            (None, Some(e), true) => *i <= e,
                            (None, Some(e), false) => *i < e,
                            (None, None, _) => true,
                        };
                        Ok(in_range)
                    }
                    _ => Ok(false),
                }
            }
            // Literal matching against string or char
            (Pattern::Literal(Literal::String(s)), Value::String(vs)) => {
                Ok(s == vs.as_str())
            }
            (Pattern::Literal(Literal::Char(c)), Value::Char(vc)) => {
                Ok(c == vc)
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
            (Value::String(a), Value::String(b)) => {
                let result = **a == **b;
                // Debug ALL short string comparisons
                if a.len() <= 5 && b.len() <= 5 {
                    eprintln!("DEBUG values_equal: '{}' == '{}' -> {}", a, b, result);
                }
                result
            }
            (Value::Char(a), Value::Char(b)) => a == b,
            _ => false,
        }
    }

    fn eval_for(
        &mut self,
        pattern: &Pattern,
        iter: &Expr,
        body: &Block,
    ) -> Result<Value, RuntimeError> {
        let iterable_raw = self.evaluate(iter)?;
        let iterable = Self::unwrap_all(&iterable_raw);
        let items = match iterable {
            Value::Array(arr) => arr.borrow().clone(),
            Value::Tuple(t) => (*t).clone(),
            Value::String(s) => s.chars().map(Value::Char).collect(),
            Value::Map(m) => {
                // Iterate over key-value pairs as tuples
                m.borrow()
                    .iter()
                    .map(|(k, v)| {
                        Value::Tuple(Rc::new(vec![
                            Value::String(Rc::new(k.clone())),
                            v.clone(),
                        ]))
                    })
                    .collect()
            }
            Value::Variant { fields: Some(f), .. } => (*f).clone(),
            _ => return Err(RuntimeError::new(format!("Cannot iterate over non-iterable: {:?}", iterable_raw))),
        };

        let mut result = Value::Null;
        for item in items {
            let env = Rc::new(RefCell::new(Environment::with_parent(
                self.environment.clone(),
            )));
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

    fn eval_break(&mut self, _value: &Option<Box<Expr>>) -> Result<Value, RuntimeError> {
        // TODO: break with value for loop expressions
        Err(RuntimeError::new("break"))
    }

    fn eval_index(&mut self, expr: &Expr, index: &Expr) -> Result<Value, RuntimeError> {
        let collection = self.evaluate(expr)?;
        let idx = self.evaluate(index)?;

        match (collection, idx) {
            (Value::Array(arr), Value::Int(i)) => {
                let arr = arr.borrow();
                let i = if i < 0 { arr.len() as i64 + i } else { i } as usize;
                arr.get(i)
                    .cloned()
                    .ok_or_else(|| RuntimeError::new("Index out of bounds"))
            }
            (Value::Tuple(t), Value::Int(i)) => {
                let i = if i < 0 { t.len() as i64 + i } else { i } as usize;
                t.get(i)
                    .cloned()
                    .ok_or_else(|| RuntimeError::new("Index out of bounds"))
            }
            (Value::String(s), Value::Int(i)) => {
                let i = if i < 0 { s.len() as i64 + i } else { i } as usize;
                s.chars()
                    .nth(i)
                    .map(Value::Char)
                    .ok_or_else(|| RuntimeError::new("Index out of bounds"))
            }
            _ => Err(RuntimeError::new("Cannot index")),
        }
    }

    fn eval_field(&mut self, expr: &Expr, field: &Ident) -> Result<Value, RuntimeError> {
        let value = self.evaluate(expr)?;
        // Helper to get field from a value
        fn get_field(val: &Value, field_name: &str) -> Result<Value, RuntimeError> {
            match val {
                Value::Struct { name, fields } => {
                    let field_val = fields.borrow().get(field_name).cloned();
                    if field_val.is_none() && field_name == "path" {
                        eprintln!("DEBUG Unknown field 'path': struct={}, available={:?}", name, fields.borrow().keys().collect::<Vec<_>>());
                    }
                    field_val.ok_or_else(|| RuntimeError::new(format!("Unknown field: {}", field_name)))
                }
                Value::Tuple(t) => {
                    // Tuple field access like .0, .1
                    let idx: usize = field_name
                        .parse()
                        .map_err(|_| RuntimeError::new("Invalid tuple index"))?;
                    t.get(idx)
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("Tuple index out of bounds"))
                }
                Value::Ref(r) => {
                    // Dereference and access field on inner value
                    get_field(&r.borrow(), field_name)
                }
                Value::Evidential { value, .. } => {
                    // Unwrap evidential wrapper and access field
                    get_field(value, field_name)
                }
                Value::Affective { value, .. } => {
                    // Unwrap affective wrapper and access field
                    get_field(value, field_name)
                }
                other => {
                    eprintln!("DEBUG Cannot access field '{}' on: {:?}", field_name, std::mem::discriminant(other));
                    Err(RuntimeError::new("Cannot access field on non-struct"))
                }
            }
        }
        get_field(&value, &field.name)
    }

    fn eval_method_call(
        &mut self,
        receiver: &Expr,
        method: &Ident,
        args: &[Expr],
    ) -> Result<Value, RuntimeError> {
        // Special handling for String::push/push_str - needs to mutate the variable
        if (method.name == "push" || method.name == "push_str") && args.len() == 1 {
            let recv_val = self.evaluate(receiver)?;
            let recv_unwrapped = Self::unwrap_value(&recv_val).clone();
            if let Value::String(s) = &recv_unwrapped {
                let arg = self.evaluate(&args[0])?;
                let arg_unwrapped = Self::unwrap_value(&arg).clone();
                let new_s = match arg_unwrapped {
                    Value::Char(c) => {
                        let mut new_str = (**s).clone();
                        new_str.push(c);
                        new_str
                    }
                    Value::String(ref add_s) => {
                        let mut new_str = (**s).clone();
                        new_str.push_str(add_s);
                        new_str
                    }
                    _ => return Err(RuntimeError::new("push expects char or string argument")),
                };
                let new_val = Value::String(Rc::new(new_s));
                // Update the receiver variable if it's a simple identifier
                if let Expr::Path(path) = receiver {
                    if path.segments.len() == 1 {
                        let var_name = &path.segments[0].ident.name;
                        self.environment.borrow_mut().set(var_name, new_val)?;
                        return Ok(Value::Null);
                    }
                }
                // Handle field access like self.output.push(c)
                if let Expr::Field { expr: base_expr, field: field_ident } = receiver {
                    let base = self.evaluate(base_expr)?;
                    if let Value::Struct { fields, .. } = base {
                        fields.borrow_mut().insert(field_ident.name.clone(), new_val);
                        return Ok(Value::Null);
                    }
                }
                // Fallback: can't mutate, just return the new string
                return Ok(new_val);
            }
        }

        let recv_raw = self.evaluate(receiver)?;
        // Unwrap evidential/affective wrappers for method dispatch
        let recv = Self::unwrap_value(&recv_raw).clone();

        // Debug: trace ALL method calls to find Lexer
        static METHOD_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = METHOD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 500 {
            let recv_type = match &recv {
                Value::Struct { name, .. } => format!("Struct({})", name),
                Value::String(s) => format!("String('{}')", if s.len() <= 20 { s.as_str() } else { "<long>" }),
                Value::Ref(r) => format!("Ref({:?})", std::mem::discriminant(&*r.borrow())),
                other => format!("{:?}", std::mem::discriminant(other)),
            };
            if recv_type.contains("Lexer") || method.name.contains("keyword") || method.name.contains("lex") {
                eprintln!("DEBUG method #{}: {}.{}()", count, recv_type, method.name);
            }
        }
        let arg_values: Vec<Value> = args
            .iter()
            .map(|a| self.evaluate(a))
            .collect::<Result<_, _>>()?;

        // Debug: Trace cloned/clone method calls
        if method.name == "cloned" || method.name == "clone" {
            let recv_type = match &recv {
                Value::Struct { name, .. } => format!("Struct({})", name),
                Value::Variant { enum_name, variant_name, .. } => format!("Variant({}::{})", enum_name, variant_name),
                Value::String(_) => "String".to_string(),
                Value::Ref(r) => format!("Ref({:?})", std::mem::discriminant(&*r.borrow())),
                Value::Null => "Null".to_string(),
                other => format!("{:?}", std::mem::discriminant(other)),
            };
            eprintln!("DEBUG {}: recv_type={}", method.name, recv_type);
        }

        // Debug: Trace ALL as_str calls
        if method.name == "as_str" {
            let recv_unwrapped = Self::unwrap_all(&recv);
            if let Value::String(s) = &recv_unwrapped {
                eprintln!("DEBUG as_str CALL: recv='{}' len={}", s, s.len());
            } else {
                eprintln!("DEBUG as_str CALL: recv={:?} (not string)", recv_unwrapped);
            }
        }

        // Debug: trace keyword_or_ident method calls
        if method.name == "keyword_or_ident" {
            let recv_type = match &recv {
                Value::Struct { name, .. } => format!("Struct({})", name),
                Value::String(_) => "String".to_string(),
                Value::Ref(r) => format!("Ref({})", match &*r.borrow() {
                    Value::Struct { name, .. } => format!("Struct({})", name),
                    other => format!("{:?}", std::mem::discriminant(other)),
                }),
                other => format!("{:?}", std::mem::discriminant(other)),
            };
            eprintln!("DEBUG keyword_or_ident: recv_type={}", recv_type);
        }

        // Debug: Find "fn" as method argument
        for arg in &arg_values {
            let unwrapped = Self::unwrap_all(arg);
            if let Value::String(s) = &unwrapped {
                if **s == "fn" {
                    let recv_type = match &recv {
                        Value::Struct { name, .. } => format!("Struct({})", name),
                        Value::String(_) => "String".to_string(),
                        Value::Ref(_) => "Ref".to_string(),
                        other => format!("{:?}", std::mem::discriminant(other)),
                    };
                    eprintln!("DEBUG method call with 'fn': method={}, recv_type={}", method.name, recv_type);
                }
            }
        }

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
            (Value::Array(arr), "pop") => arr
                .borrow_mut()
                .pop()
                .ok_or_else(|| RuntimeError::new("pop on empty array")),
            (Value::Array(arr), "reverse") => {
                let mut v = arr.borrow().clone();
                v.reverse();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "first") => arr
                .borrow()
                .first()
                .cloned()
                .ok_or_else(|| RuntimeError::new("empty array")),
            (Value::Array(arr), "last") => arr
                .borrow()
                .last()
                .cloned()
                .ok_or_else(|| RuntimeError::new("empty array")),
            (Value::Array(arr), "iter") => {
                // iter() on an array just returns the array - iteration happens in for loops
                Ok(Value::Array(arr.clone()))
            }
            (Value::Array(arr), "zip") => {
                // zip with another array to create array of tuples
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("zip expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::Array(other) => {
                        let a = arr.borrow();
                        let b = other.borrow();
                        let zipped: Vec<Value> = a.iter()
                            .zip(b.iter())
                            .map(|(x, y)| Value::Tuple(Rc::new(vec![x.clone(), y.clone()])))
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(zipped))))
                    }
                    _ => Err(RuntimeError::new("zip expects array argument")),
                }
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
            (Value::String(s), "as_str") => {
                if s.len() <= 10 { eprintln!("DEBUG as_str: '{}'", s); }
                Ok(Value::String(s.clone()))
            }
            (Value::String(s), "to_string") => Ok(Value::String(s.clone())),
            (Value::String(s), "starts_with") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("starts_with expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::String(prefix) => Ok(Value::Bool(s.starts_with(prefix.as_str()))),
                    _ => Err(RuntimeError::new("starts_with expects string")),
                }
            }
            (Value::String(s), "ends_with") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("ends_with expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::String(suffix) => Ok(Value::Bool(s.ends_with(suffix.as_str()))),
                    _ => Err(RuntimeError::new("ends_with expects string")),
                }
            }
            (Value::String(s), "is_empty") => Ok(Value::Bool(s.is_empty())),
            (Value::String(s), "clone") => Ok(Value::String(Rc::new((**s).clone()))),
            (Value::String(s), "as_ptr") => {
                // Return the string itself - FFI emulation doesn't need real pointers
                Ok(Value::String(s.clone()))
            }
            (Value::String(_), "is_null") => Ok(Value::Bool(false)),
            (Value::Null, "is_null") => Ok(Value::Bool(true)),
            (Value::String(s), "char_at") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("char_at expects 1 argument"));
                }
                let idx = match &arg_values[0] {
                    Value::Int(i) => *i as usize,
                    _ => return Err(RuntimeError::new("char_at expects integer index")),
                };
                match s.chars().nth(idx) {
                    Some(c) => Ok(Value::Char(c)),
                    None => Err(RuntimeError::new("char_at index out of bounds")),
                }
            }
            (Value::String(s), "chars") => {
                let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                Ok(Value::Array(Rc::new(RefCell::new(chars))))
            }
            (Value::String(s), "bytes") => {
                let bytes: Vec<Value> = s.bytes().map(|b| Value::Int(b as i64)).collect();
                Ok(Value::Array(Rc::new(RefCell::new(bytes))))
            }
            (Value::String(s), "split") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("split expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::String(sep) => {
                        let parts: Vec<Value> = s.split(sep.as_str())
                            .map(|p| Value::String(Rc::new(p.to_string())))
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(parts))))
                    }
                    Value::Char(sep) => {
                        let parts: Vec<Value> = s.split(*sep)
                            .map(|p| Value::String(Rc::new(p.to_string())))
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(parts))))
                    }
                    _ => Err(RuntimeError::new("split expects string or char separator")),
                }
            }
            // Char methods
            (Value::Char(c), "len_utf8") => Ok(Value::Int(c.len_utf8() as i64)),
            (Value::Char(c), "is_alphabetic") => Ok(Value::Bool(c.is_alphabetic())),
            (Value::Char(c), "is_alphanumeric") => Ok(Value::Bool(c.is_alphanumeric())),
            (Value::Char(c), "is_ascii_alphanumeric") => Ok(Value::Bool(c.is_ascii_alphanumeric())),
            (Value::Char(c), "is_ascii_alphabetic") => Ok(Value::Bool(c.is_ascii_alphabetic())),
            (Value::Char(c), "is_ascii_digit") => Ok(Value::Bool(c.is_ascii_digit())),
            (Value::Char(c), "is_ascii_hexdigit") => Ok(Value::Bool(c.is_ascii_hexdigit())),
            (Value::Char(c), "is_ascii") => Ok(Value::Bool(c.is_ascii())),
            (Value::Char(c), "is_digit") => {
                let radix = if arg_values.is_empty() { 10 } else {
                    match &arg_values[0] {
                        Value::Int(n) => *n as u32,
                        _ => 10,
                    }
                };
                Ok(Value::Bool(c.is_digit(radix)))
            }
            (Value::Char(c), "is_numeric") => Ok(Value::Bool(c.is_numeric())),
            (Value::Char(c), "is_whitespace") => Ok(Value::Bool(c.is_whitespace())),
            (Value::Char(c), "is_uppercase") => Ok(Value::Bool(c.is_uppercase())),
            (Value::Char(c), "is_lowercase") => Ok(Value::Bool(c.is_lowercase())),
            (Value::Char(c), "to_uppercase") => {
                let upper: String = c.to_uppercase().collect();
                Ok(Value::String(Rc::new(upper)))
            }
            (Value::Char(c), "to_lowercase") => {
                let lower: String = c.to_lowercase().collect();
                Ok(Value::String(Rc::new(lower)))
            }
            (Value::Char(c), "to_string") => Ok(Value::String(Rc::new(c.to_string()))),
            (Value::Char(c), "to_digit") => {
                let radix = if arg_values.is_empty() { 10 } else {
                    match &arg_values[0] {
                        Value::Int(n) => *n as u32,
                        _ => 10,
                    }
                };
                match c.to_digit(radix) {
                    Some(d) => Ok(Value::Int(d as i64)),
                    None => Ok(Value::Null),
                }
            }
            (Value::String(s), "upper") | (Value::String(s), "uppercase") => {
                Ok(Value::String(Rc::new(s.to_uppercase())))
            }
            (Value::String(s), "lower") | (Value::String(s), "lowercase") => {
                Ok(Value::String(Rc::new(s.to_lowercase())))
            }
            (Value::String(s), "trim") => Ok(Value::String(Rc::new(s.trim().to_string()))),
            (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
            (Value::String(s), "is_empty") => Ok(Value::Bool(s.is_empty())),
            (Value::String(s), "first") => s
                .chars()
                .next()
                .map(Value::Char)
                .ok_or_else(|| RuntimeError::new("empty string")),
            (Value::String(s), "last") => s
                .chars()
                .last()
                .map(Value::Char)
                .ok_or_else(|| RuntimeError::new("empty string")),
            (Value::Array(arr), "is_empty") => Ok(Value::Bool(arr.borrow().is_empty())),
            (Value::Array(arr), "clone") => Ok(Value::Array(Rc::new(RefCell::new(arr.borrow().clone())))),
            // Map methods
            (Value::Map(m), "insert") => {
                if arg_values.len() != 2 {
                    return Err(RuntimeError::new("insert expects 2 arguments"));
                }
                let key = match &arg_values[0] {
                    Value::String(s) => (**s).clone(),
                    _ => format!("{}", arg_values[0]),
                };
                m.borrow_mut().insert(key, arg_values[1].clone());
                Ok(Value::Null)
            }
            (Value::Map(m), "get") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("get expects 1 argument"));
                }
                let key = match &arg_values[0] {
                    Value::String(s) => (**s).clone(),
                    _ => format!("{}", arg_values[0]),
                };
                Ok(m.borrow().get(&key).cloned().unwrap_or(Value::Null))
            }
            (Value::Map(m), "contains_key") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("contains_key expects 1 argument"));
                }
                let key = match &arg_values[0] {
                    Value::String(s) => (**s).clone(),
                    _ => format!("{}", arg_values[0]),
                };
                Ok(Value::Bool(m.borrow().contains_key(&key)))
            }
            (Value::Map(m), "len") => Ok(Value::Int(m.borrow().len() as i64)),
            (Value::Map(m), "is_empty") => Ok(Value::Bool(m.borrow().is_empty())),
            (Value::Map(m), "keys") => {
                let keys: Vec<Value> = m.borrow().keys()
                    .map(|k| Value::String(Rc::new(k.clone())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(keys))))
            }
            (Value::Map(m), "values") => {
                let values: Vec<Value> = m.borrow().values().cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(values))))
            }
            // Ref methods
            (Value::Ref(r), "cloned") => {
                // Clone the inner value
                Ok(r.borrow().clone())
            }
            (Value::Ref(r), "borrow") => {
                // Return a reference to the inner value
                Ok(recv.clone())
            }
            (Value::Ref(r), "borrow_mut") => {
                // Return a reference to the inner value (mutable in place)
                Ok(recv.clone())
            }
            // Forward method calls on Ref to inner value (struct method lookup)
            (Value::Ref(r), _) => {
                // Dereference and look up method on inner struct
                let inner = r.borrow().clone();
                if let Value::Struct { name, fields: _ } = &inner {
                    // Try struct method lookup with the inner struct
                    let qualified_name = format!("{}·{}", name, method.name);
                    let func = self.globals.borrow().get(&qualified_name).map(|v| v.clone());
                    if let Some(func) = func {
                        if let Value::Function(f) = func {
                            // Pass the Ref as the receiver (for &mut self methods)
                            let mut all_args = vec![recv.clone()];
                            all_args.extend(arg_values.clone());
                            return self.call_function(&f, all_args);
                        } else if let Value::BuiltIn(b) = func {
                            let mut all_args = vec![recv.clone()];
                            all_args.extend(arg_values.clone());
                            return (b.func)(self, all_args);
                        }
                    }
                    return Err(RuntimeError::new(format!(
                        "Unknown method: {} on &{}",
                        method.name, name
                    )));
                }
                Err(RuntimeError::new(format!(
                    "Cannot call method {} on Ref to non-struct",
                    method.name
                )))
            }
            // Try struct method lookup: StructName·method
            (Value::Struct { name, fields }, _) => {
                // Built-in struct methods
                if method.name == "clone" {
                    eprintln!("DEBUG struct clone: name={}", name);
                    // Clone the struct value
                    return Ok(recv.clone());
                }

                let qualified_name = format!("{}·{}", name, method.name);

                // Debug: track Lexer method calls
                if name == "Lexer" {
                    // Print all args for lex_ident_or_keyword
                    if method.name == "lex_ident_or_keyword" {
                        for (i, arg) in arg_values.iter().enumerate() {
                            let unwrapped = Self::unwrap_all(arg);
                            if let Value::Char(c) = &unwrapped {
                                eprintln!("DEBUG Lexer·lex_ident_or_keyword arg[{}]='{}'", i, c);
                            }
                        }
                    }
                    eprintln!("DEBUG Lexer method call: {}", qualified_name);
                }
                // Check if arg is "fn" string
                for arg in &arg_values {
                    let unwrapped = Self::unwrap_all(arg);
                    if let Value::String(s) = &unwrapped {
                        if **s == "fn" {
                            eprintln!("DEBUG struct method with 'fn': {} recv_name={}", method.name, name);
                        }
                    }
                }

                let func = self.globals.borrow().get(&qualified_name).map(|v| v.clone());
                if let Some(func) = func {
                    if let Value::Function(f) = func {
                        // Call with self as first argument
                        let mut all_args = vec![recv.clone()];
                        all_args.extend(arg_values.clone());
                        return self.call_function(&f, all_args);
                    } else if let Value::BuiltIn(b) = func {
                        let mut all_args = vec![recv.clone()];
                        all_args.extend(arg_values.clone());
                        return (b.func)(self, all_args);
                    }
                }

                // If struct name is "Self", try to find the method by searching all types
                if name == "Self" {
                    // Get field names to match struct type
                    let field_names: Vec<String> = fields.borrow().keys().cloned().collect();

                    // Search through registered types to find a matching struct
                    for (type_name, type_def) in &self.types {
                        if let TypeDef::Struct(struct_def) = type_def {
                            // Check if field names match
                            let def_fields: Vec<String> = match &struct_def.fields {
                                crate::ast::StructFields::Named(fs) => fs.iter().map(|f| f.name.name.clone()).collect(),
                                _ => continue,
                            };

                            // Rough match - if we have fields that exist in the definition
                            let matches = field_names.iter().all(|f| def_fields.contains(f));
                            if matches {
                                let qualified_name = format!("{}·{}", type_name, method.name);
                                let func = self.globals.borrow().get(&qualified_name).map(|v| v.clone());
                                if let Some(func) = func {
                                    if let Value::Function(f) = func {
                                        let mut all_args = vec![recv.clone()];
                                        all_args.extend(arg_values.clone());
                                        return self.call_function(&f, all_args);
                                    } else if let Value::BuiltIn(b) = func {
                                        let mut all_args = vec![recv.clone()];
                                        all_args.extend(arg_values.clone());
                                        return (b.func)(self, all_args);
                                    }
                                }
                            }
                        }
                    }
                }

                Err(RuntimeError::new(format!(
                    "Unknown method: {} on {}",
                    method.name, name
                )))
            }
            // Try variant method lookup: EnumName·method
            (Value::Variant { enum_name, variant_name, fields }, _) => {
                // Built-in Option methods
                if enum_name == "Option" {
                    match method.name.as_str() {
                        "cloned" => {
                            // cloned() on Option<&T> returns Option<T>
                            // In our interpreter, just clone the value
                            return Ok(recv.clone());
                        }
                        "is_some" => {
                            return Ok(Value::Bool(variant_name == "Some"));
                        }
                        "is_none" => {
                            return Ok(Value::Bool(variant_name == "None"));
                        }
                        "unwrap" => {
                            if variant_name == "Some" {
                                if let Some(f) = fields {
                                    return Ok(f.first().cloned().unwrap_or(Value::Null));
                                }
                            }
                            return Err(RuntimeError::new("unwrap on None"));
                        }
                        "unwrap_or" => {
                            if variant_name == "Some" {
                                if let Some(f) = fields {
                                    return Ok(f.first().cloned().unwrap_or(Value::Null));
                                }
                            }
                            return Ok(arg_values.first().cloned().unwrap_or(Value::Null));
                        }
                        "map" => {
                            // Option::map takes a closure
                            if variant_name == "Some" {
                                if let Some(f) = fields {
                                    if let Some(inner) = f.first() {
                                        if let Some(Value::Function(func)) = arg_values.first() {
                                            let result = self.call_function(func, vec![inner.clone()])?;
                                            return Ok(Value::Variant {
                                                enum_name: "Option".to_string(),
                                                variant_name: "Some".to_string(),
                                                fields: Some(Rc::new(vec![result])),
                                            });
                                        }
                                    }
                                }
                            }
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
                        }
                        _ => {}
                    }
                }
                // Built-in Result methods
                if enum_name == "Result" {
                    match method.name.as_str() {
                        "is_ok" => {
                            return Ok(Value::Bool(variant_name == "Ok"));
                        }
                        "is_err" => {
                            return Ok(Value::Bool(variant_name == "Err"));
                        }
                        "unwrap" => {
                            if variant_name == "Ok" {
                                if let Some(f) = fields {
                                    return Ok(f.first().cloned().unwrap_or(Value::Null));
                                }
                            }
                            return Err(RuntimeError::new("unwrap on Err"));
                        }
                        "unwrap_or" => {
                            if variant_name == "Ok" {
                                if let Some(f) = fields {
                                    return Ok(f.first().cloned().unwrap_or(Value::Null));
                                }
                            }
                            return Ok(arg_values.first().cloned().unwrap_or(Value::Null));
                        }
                        _ => {}
                    }
                }
                // Built-in clone method for all variants
                if method.name == "clone" {
                    return Ok(recv.clone());
                }

                let qualified_name = format!("{}·{}", enum_name, method.name);
                let func = self.globals.borrow().get(&qualified_name).map(|v| v.clone());
                if let Some(func) = func {
                    if let Value::Function(f) = func {
                        let mut all_args = vec![recv.clone()];
                        all_args.extend(arg_values.clone());
                        return self.call_function(&f, all_args);
                    } else if let Value::BuiltIn(b) = func {
                        let mut all_args = vec![recv.clone()];
                        all_args.extend(arg_values.clone());
                        return (b.func)(self, all_args);
                    }
                }
                Err(RuntimeError::new(format!(
                    "Unknown method: {} on {}",
                    method.name, enum_name
                )))
            }
            _ => {
                // Debug: what type is failing method lookup
                let recv_type = match &recv {
                    Value::String(s) => format!("String(len={})", s.len()),
                    Value::Array(arr) => format!("Array(len={})", arr.borrow().len()),
                    Value::Struct { name, .. } => format!("Struct({})", name),
                    Value::Variant { enum_name, variant_name, .. } => format!("Variant({}::{})", enum_name, variant_name),
                    Value::Ref(r) => format!("Ref({:?})", std::mem::discriminant(&*r.borrow())),
                    other => format!("{:?}", std::mem::discriminant(other)),
                };
                eprintln!("DEBUG Unknown method '{}' on recv_type={}", method.name, recv_type);
                Err(RuntimeError::new(format!(
                    "Unknown method: {}",
                    method.name
                )))
            }
        }
    }

    /// Evaluate polysynthetic incorporation: path·file·read·string
    /// The first segment provides the initial value, subsequent segments are method-like transformations
    fn eval_incorporation(
        &mut self,
        segments: &[IncorporationSegment],
    ) -> Result<Value, RuntimeError> {
        if segments.is_empty() {
            return Err(RuntimeError::new("empty incorporation chain"));
        }

        // First segment: get initial value (variable lookup or function call)
        let first = &segments[0];
        let mut value = if let Some(args) = &first.args {
            // First segment is a function call: func(args)·next·...
            let arg_values: Vec<Value> = args
                .iter()
                .map(|a| self.evaluate(a))
                .collect::<Result<_, _>>()?;
            self.call_function_by_name(&first.name.name, arg_values)?
        } else {
            // First segment is a variable: var·next·...
            self.environment
                .borrow()
                .get(&first.name.name)
                .ok_or_else(|| RuntimeError::new(format!("undefined: {}", first.name.name)))?
        };

        // Process remaining segments as method-like calls
        for segment in segments.iter().skip(1) {
            let arg_values: Vec<Value> = segment
                .args
                .as_ref()
                .map(|args| {
                    args.iter()
                        .map(|a| self.evaluate(a))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();

            // Try to call as a method on the value
            value = self.call_incorporation_method(&value, &segment.name.name, arg_values)?;
        }

        Ok(value)
    }

    /// Call a method in an incorporation chain
    /// This looks up the segment name as a method or stdlib function
    fn call_incorporation_method(
        &mut self,
        receiver: &Value,
        method_name: &str,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        // First try as a method on the receiver value
        match (receiver, method_name) {
            // String methods
            (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
            (Value::String(s), "upper") | (Value::String(s), "uppercase") => {
                Ok(Value::String(Rc::new(s.to_uppercase())))
            }
            (Value::String(s), "lower") | (Value::String(s), "lowercase") => {
                Ok(Value::String(Rc::new(s.to_lowercase())))
            }
            (Value::String(s), "trim") => Ok(Value::String(Rc::new(s.trim().to_string()))),
            (Value::String(s), "chars") => {
                let chars: Vec<Value> = s
                    .chars()
                    .map(|c| Value::String(Rc::new(c.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(chars))))
            }
            (Value::String(s), "lines") => {
                let lines: Vec<Value> = s
                    .lines()
                    .map(|l| Value::String(Rc::new(l.to_string())))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(lines))))
            }
            (Value::String(s), "bytes") => {
                let bytes: Vec<Value> = s.bytes().map(|b| Value::Int(b as i64)).collect();
                Ok(Value::Array(Rc::new(RefCell::new(bytes))))
            }
            (Value::String(s), "parse_int") | (Value::String(s), "to_int") => s
                .parse::<i64>()
                .map(Value::Int)
                .map_err(|_| RuntimeError::new(format!("cannot parse '{}' as int", s))),
            (Value::String(s), "parse_float") | (Value::String(s), "to_float") => s
                .parse::<f64>()
                .map(Value::Float)
                .map_err(|_| RuntimeError::new(format!("cannot parse '{}' as float", s))),
            (Value::String(s), "as_str") => {
                if s.len() <= 10 { eprintln!("DEBUG as_str: '{}'", s); }
                Ok(Value::String(s.clone()))
            }
            (Value::String(s), "to_string") => Ok(Value::String(s.clone())),
            (Value::String(s), "starts_with") => {
                if args.len() != 1 {
                    return Err(RuntimeError::new("starts_with expects 1 argument"));
                }
                match &args[0] {
                    Value::String(prefix) => Ok(Value::Bool(s.starts_with(prefix.as_str()))),
                    _ => Err(RuntimeError::new("starts_with expects string")),
                }
            }
            (Value::String(s), "ends_with") => {
                if args.len() != 1 {
                    return Err(RuntimeError::new("ends_with expects 1 argument"));
                }
                match &args[0] {
                    Value::String(suffix) => Ok(Value::Bool(s.ends_with(suffix.as_str()))),
                    _ => Err(RuntimeError::new("ends_with expects string")),
                }
            }
            (Value::String(s), "is_empty") => Ok(Value::Bool(s.is_empty())),
            (Value::String(s), "clone") => Ok(Value::String(Rc::new((**s).clone()))),
            (Value::String(s), "first") => s
                .chars()
                .next()
                .map(Value::Char)
                .ok_or_else(|| RuntimeError::new("empty string")),
            (Value::String(s), "last") => s
                .chars()
                .last()
                .map(Value::Char)
                .ok_or_else(|| RuntimeError::new("empty string")),

            // Array methods
            (Value::Array(arr), "len") => Ok(Value::Int(arr.borrow().len() as i64)),
            (Value::Array(arr), "first") => arr
                .borrow()
                .first()
                .cloned()
                .ok_or_else(|| RuntimeError::new("empty array")),
            (Value::Array(arr), "last") => arr
                .borrow()
                .last()
                .cloned()
                .ok_or_else(|| RuntimeError::new("empty array")),
            (Value::Array(arr), "reverse") | (Value::Array(arr), "rev") => {
                let mut v = arr.borrow().clone();
                v.reverse();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "join") => {
                let sep = args
                    .first()
                    .map(|v| match v {
                        Value::String(s) => s.to_string(),
                        _ => "".to_string(),
                    })
                    .unwrap_or_default();
                let joined = arr
                    .borrow()
                    .iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join(&sep);
                Ok(Value::String(Rc::new(joined)))
            }
            (Value::Array(arr), "sum") => {
                let mut sum = 0i64;
                for v in arr.borrow().iter() {
                    match v {
                        Value::Int(i) => sum += i,
                        Value::Float(f) => return Ok(Value::Float(sum as f64 + f)),
                        _ => {}
                    }
                }
                Ok(Value::Int(sum))
            }

            // Number methods
            (Value::Int(n), "abs") => Ok(Value::Int(n.abs())),
            (Value::Float(n), "abs") => Ok(Value::Float(n.abs())),
            (Value::Int(n), "to_string") | (Value::Int(n), "string") => {
                Ok(Value::String(Rc::new(n.to_string())))
            }
            (Value::Float(n), "to_string") | (Value::Float(n), "string") => {
                Ok(Value::String(Rc::new(n.to_string())))
            }
            (Value::Int(n), "to_float") | (Value::Int(n), "float") => Ok(Value::Float(*n as f64)),
            (Value::Float(n), "to_int") | (Value::Float(n), "int") => Ok(Value::Int(*n as i64)),

            // Map/Struct field access
            (Value::Map(map), field) => map
                .borrow()
                .get(field)
                .cloned()
                .ok_or_else(|| RuntimeError::new(format!("no field '{}' in map", field))),
            (Value::Struct { fields, .. }, field) => fields
                .borrow()
                .get(field)
                .cloned()
                .ok_or_else(|| RuntimeError::new(format!("no field '{}' in struct", field))),

            // Try stdlib function with receiver as first arg
            _ => {
                let mut all_args = vec![receiver.clone()];
                all_args.extend(args);
                self.call_function_by_name(method_name, all_args)
            }
        }
    }

    /// Call a function by name from the environment
    pub fn call_function_by_name(
        &mut self,
        name: &str,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        // Get the function value from environment (clone to avoid borrow issues)
        let func_value = self.environment.borrow().get(name);

        match func_value {
            Some(Value::Function(f)) => self.call_function(&f, args),
            Some(Value::BuiltIn(b)) => self.call_builtin(&b, args),
            Some(_) => Err(RuntimeError::new(format!("{} is not a function", name))),
            None => {
                // Check for variant constructor
                if let Some((enum_name, variant_name, arity)) = self.variant_constructors.get(name).cloned() {
                    if arity == 0 && args.is_empty() {
                        return Ok(Value::Variant {
                            enum_name,
                            variant_name,
                            fields: None,
                        });
                    } else if args.len() == arity {
                        return Ok(Value::Variant {
                            enum_name,
                            variant_name,
                            fields: Some(Rc::new(args)),
                        });
                    } else {
                        return Err(RuntimeError::new(format!(
                            "{} expects {} arguments, got {}",
                            name, arity, args.len()
                        )));
                    }
                }
                Err(RuntimeError::new(format!("undefined function: {}", name)))
            }
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
                        let results: Vec<Value> = arr
                            .borrow()
                            .iter()
                            .map(|item| {
                                self.environment
                                    .borrow_mut()
                                    .define("_".to_string(), item.clone());
                                self.evaluate(body)
                            })
                            .collect::<Result<_, _>>()?;
                        Ok(Value::Array(Rc::new(RefCell::new(results))))
                    }
                    single => {
                        self.environment
                            .borrow_mut()
                            .define("_".to_string(), single);
                        self.evaluate(body)
                    }
                }
            }
            PipeOp::Filter(predicate) => {
                // φ{p} - filter collection
                match value {
                    Value::Array(arr) => {
                        let results: Vec<Value> = arr
                            .borrow()
                            .iter()
                            .filter_map(|item| {
                                self.environment
                                    .borrow_mut()
                                    .define("_".to_string(), item.clone());
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
                            self.environment
                                .borrow_mut()
                                .define("_".to_string(), item.clone());
                            acc = self.evaluate(body)?;
                        }
                        Ok(acc)
                    }
                    _ => Err(RuntimeError::new("Reduce requires array")),
                }
            }
            PipeOp::ReduceSum => {
                // ρ+ or ρ_sum - sum all elements
                self.sum_values(value)
            }
            PipeOp::ReduceProd => {
                // ρ* or ρ_prod - multiply all elements
                self.product_values(value)
            }
            PipeOp::ReduceMin => {
                // ρ_min - find minimum element
                self.min_values(value)
            }
            PipeOp::ReduceMax => {
                // ρ_max - find maximum element
                self.max_values(value)
            }
            PipeOp::ReduceConcat => {
                // ρ++ or ρ_cat - concatenate strings/arrays
                self.concat_values(value)
            }
            PipeOp::ReduceAll => {
                // ρ& or ρ_all - logical AND (all true)
                self.all_values(value)
            }
            PipeOp::ReduceAny => {
                // ρ| or ρ_any - logical OR (any true)
                self.any_values(value)
            }
            PipeOp::Match(arms) => {
                // |match{ Pattern => expr, ... } - pattern matching in pipe
                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &value)? {
                        // Create new scope for pattern bindings
                        let prev_env = self.environment.clone();
                        self.environment =
                            Rc::new(RefCell::new(Environment::with_parent(prev_env.clone())));

                        // Bind pattern variables
                        self.bind_pattern(&arm.pattern, value.clone())?;

                        // Also bind _ to the piped value for convenient access
                        self.environment
                            .borrow_mut()
                            .define("_".to_string(), value.clone());

                        // Check guard if present
                        let guard_passes = if let Some(guard) = &arm.guard {
                            matches!(self.evaluate(guard)?, Value::Bool(true))
                        } else {
                            true
                        };

                        if guard_passes {
                            let result = self.evaluate(&arm.body)?;
                            self.environment = prev_env;
                            return Ok(result);
                        }

                        // Guard failed, restore environment and try next arm
                        self.environment = prev_env;
                    }
                }
                Err(RuntimeError::new("No pattern matched in pipe match"))
            }
            PipeOp::TryMap(mapper) => {
                // |? or |?{mapper} - unwrap Result/Option or transform error
                match &value {
                    // Handle Result-like values (struct with ok/err fields)
                    Value::Struct { name, fields } if name == "Ok" || name.ends_with("::Ok") => {
                        // Extract the inner value from Ok
                        let fields = fields.borrow();
                        fields
                            .get("0")
                            .or_else(|| fields.get("value"))
                            .cloned()
                            .ok_or_else(|| RuntimeError::new("Ok variant has no value"))
                    }
                    Value::Struct { name, fields } if name == "Err" || name.ends_with("::Err") => {
                        // Transform error if mapper provided, otherwise propagate
                        let fields = fields.borrow();
                        let err_val = fields
                            .get("0")
                            .or_else(|| fields.get("error"))
                            .cloned()
                            .unwrap_or(Value::Null);
                        if let Some(mapper_expr) = mapper {
                            // Apply mapper to error
                            let prev_env = self.environment.clone();
                            self.environment =
                                Rc::new(RefCell::new(Environment::with_parent(prev_env.clone())));
                            self.environment
                                .borrow_mut()
                                .define("_".to_string(), err_val);
                            let mapped = self.evaluate(mapper_expr)?;
                            self.environment = prev_env;
                            Err(RuntimeError::new(format!("Error: {:?}", mapped)))
                        } else {
                            Err(RuntimeError::new(format!("Error: {:?}", err_val)))
                        }
                    }
                    // Handle Option-like values
                    Value::Struct { name, fields }
                        if name == "Some" || name.ends_with("::Some") =>
                    {
                        let fields = fields.borrow();
                        fields
                            .get("0")
                            .or_else(|| fields.get("value"))
                            .cloned()
                            .ok_or_else(|| RuntimeError::new("Some variant has no value"))
                    }
                    Value::Struct { name, .. } if name == "None" || name.ends_with("::None") => {
                        Err(RuntimeError::new("Unwrapped None value"))
                    }
                    Value::Null => Err(RuntimeError::new("Unwrapped null value")),
                    // Pass through non-Result/Option values unchanged
                    _ => Ok(value),
                }
            }
            PipeOp::Method { name, args } => {
                let arg_values: Vec<Value> = args
                    .iter()
                    .map(|a| self.evaluate(a))
                    .collect::<Result<_, _>>()?;

                // Check for built-in pipe methods
                match name.name.as_str() {
                    "collect" => Ok(value), // Already collected
                    "sum" | "Σ" => self.sum_values(value),
                    "product" | "Π" => self.product_values(value),
                    "len" => match &value {
                        Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                        Value::String(s) => Ok(Value::Int(s.len() as i64)),
                        _ => Err(RuntimeError::new("len requires array or string")),
                    },
                    "reverse" => match value {
                        Value::Array(arr) => {
                            let mut v = arr.borrow().clone();
                            v.reverse();
                            Ok(Value::Array(Rc::new(RefCell::new(v))))
                        }
                        _ => Err(RuntimeError::new("reverse requires array")),
                    },
                    "first" => match &value {
                        Value::Array(arr) => arr
                            .borrow()
                            .first()
                            .cloned()
                            .ok_or_else(|| RuntimeError::new("first on empty array")),
                        _ => Err(RuntimeError::new("first requires array")),
                    },
                    "last" => match &value {
                        Value::Array(arr) => arr
                            .borrow()
                            .last()
                            .cloned()
                            .ok_or_else(|| RuntimeError::new("last on empty array")),
                        _ => Err(RuntimeError::new("last requires array")),
                    },
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
                    "join" => {
                        // Join array elements with a separator string
                        let separator = if arg_values.is_empty() {
                            String::new()
                        } else {
                            match &arg_values[0] {
                                Value::String(s) => (**s).clone(),
                                _ => return Err(RuntimeError::new("join separator must be string")),
                            }
                        };
                        match value {
                            Value::Array(arr) => {
                                let parts: Vec<String> = arr.borrow().iter()
                                    .map(|v| format!("{}", Self::unwrap_all(v)))
                                    .collect();
                                Ok(Value::String(Rc::new(parts.join(&separator))))
                            }
                            _ => Err(RuntimeError::new("join requires array")),
                        }
                    }
                    _ => Err(RuntimeError::new(format!(
                        "Unknown pipe method: {}",
                        name.name
                    ))),
                }
            }
            PipeOp::Await => {
                // Await a future - resolve it to a value
                self.await_value(value)
            }
            // New access morphemes
            PipeOp::First => {
                // α - first element
                match &value {
                    Value::Array(arr) => arr
                        .borrow()
                        .first()
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("first (α) on empty array")),
                    Value::Tuple(t) => t
                        .first()
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("first (α) on empty tuple")),
                    _ => Err(RuntimeError::new("first (α) requires array or tuple")),
                }
            }
            PipeOp::Last => {
                // ω - last element
                match &value {
                    Value::Array(arr) => arr
                        .borrow()
                        .last()
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("last (ω) on empty array")),
                    Value::Tuple(t) => t
                        .last()
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("last (ω) on empty tuple")),
                    _ => Err(RuntimeError::new("last (ω) requires array or tuple")),
                }
            }
            PipeOp::Middle => {
                // μ - middle/median element
                match &value {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        if arr.is_empty() {
                            return Err(RuntimeError::new("middle (μ) on empty array"));
                        }
                        let mid = arr.len() / 2;
                        Ok(arr[mid].clone())
                    }
                    Value::Tuple(t) => {
                        if t.is_empty() {
                            return Err(RuntimeError::new("middle (μ) on empty tuple"));
                        }
                        let mid = t.len() / 2;
                        Ok(t[mid].clone())
                    }
                    _ => Err(RuntimeError::new("middle (μ) requires array or tuple")),
                }
            }
            PipeOp::Choice => {
                // χ - random element
                use std::time::{SystemTime, UNIX_EPOCH};
                match &value {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        if arr.is_empty() {
                            return Err(RuntimeError::new("choice (χ) on empty array"));
                        }
                        let seed = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or(std::time::Duration::ZERO)
                            .as_nanos() as u64;
                        let idx = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) >> 16)
                            as usize
                            % arr.len();
                        Ok(arr[idx].clone())
                    }
                    Value::Tuple(t) => {
                        if t.is_empty() {
                            return Err(RuntimeError::new("choice (χ) on empty tuple"));
                        }
                        let seed = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or(std::time::Duration::ZERO)
                            .as_nanos() as u64;
                        let idx = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) >> 16)
                            as usize
                            % t.len();
                        Ok(t[idx].clone())
                    }
                    _ => Err(RuntimeError::new("choice (χ) requires array or tuple")),
                }
            }
            PipeOp::Nth(index_expr) => {
                // ν{n} - nth element
                let index = match self.evaluate(index_expr)? {
                    Value::Int(n) => n,
                    _ => return Err(RuntimeError::new("nth (ν) index must be integer")),
                };
                match &value {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        if index < 0 || index as usize >= arr.len() {
                            return Err(RuntimeError::new("nth (ν) index out of bounds"));
                        }
                        Ok(arr[index as usize].clone())
                    }
                    Value::Tuple(t) => {
                        if index < 0 || index as usize >= t.len() {
                            return Err(RuntimeError::new("nth (ν) index out of bounds"));
                        }
                        Ok(t[index as usize].clone())
                    }
                    _ => Err(RuntimeError::new("nth (ν) requires array or tuple")),
                }
            }
            PipeOp::Next => {
                // ξ - next element (for iterators, currently just returns first)
                // In a full implementation, this would advance an iterator
                match &value {
                    Value::Array(arr) => arr
                        .borrow()
                        .first()
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("next (ξ) on empty array")),
                    Value::Tuple(t) => t
                        .first()
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("next (ξ) on empty tuple")),
                    _ => Err(RuntimeError::new("next (ξ) requires array or tuple")),
                }
            }
            PipeOp::Named { prefix, body } => {
                // Named morpheme like ·map{f}
                let method_name = prefix
                    .iter()
                    .map(|i| i.name.as_str())
                    .collect::<Vec<_>>()
                    .join("·");
                match method_name.as_str() {
                    "map" => {
                        if let Some(body) = body {
                            match value {
                                Value::Array(arr) => {
                                    let results: Vec<Value> = arr
                                        .borrow()
                                        .iter()
                                        .map(|item| {
                                            self.environment
                                                .borrow_mut()
                                                .define("_".to_string(), item.clone());
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
                                    let results: Vec<Value> = arr
                                        .borrow()
                                        .iter()
                                        .filter_map(|item| {
                                            self.environment
                                                .borrow_mut()
                                                .define("_".to_string(), item.clone());
                                            match self.evaluate(body) {
                                                Ok(v) if self.is_truthy(&v) => {
                                                    Some(Ok(item.clone()))
                                                }
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
                    _ => Err(RuntimeError::new(format!(
                        "Unknown named morpheme: {}",
                        method_name
                    ))),
                }
            }
            PipeOp::Parallel(inner_op) => {
                // ∥ - parallel execution of the inner operation
                // For arrays, execute the operation in parallel using threads
                match value {
                    Value::Array(arr) => {
                        use std::sync::{Arc, Mutex};

                        let arr_ref = arr.borrow();
                        let len = arr_ref.len();
                        if len == 0 {
                            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
                        }

                        // For Transform operations, parallelize across elements
                        match inner_op.as_ref() {
                            PipeOp::Transform(body) => {
                                // Determine number of threads (use available parallelism)
                                let num_threads = std::thread::available_parallelism()
                                    .map(|p| p.get())
                                    .unwrap_or(4)
                                    .min(len);

                                // For future parallel implementation
                                let _chunk_size = (len + num_threads - 1) / num_threads;
                                let _results = Arc::new(Mutex::new(vec![Value::Null; len]));
                                let items: Vec<Value> = arr_ref.clone();
                                drop(arr_ref);

                                // Clone the body expression for each thread (for future use)
                                let _body_str = format!("{:?}", body);

                                // For now, fall back to sequential since full parallelization
                                // requires thread-safe evaluation context
                                // In production, this would use Rayon or a work-stealing scheduler
                                let mut result_vec = Vec::with_capacity(len);
                                for item in items.iter() {
                                    self.environment
                                        .borrow_mut()
                                        .define("_".to_string(), item.clone());
                                    result_vec.push(self.evaluate(body)?);
                                }
                                Ok(Value::Array(Rc::new(RefCell::new(result_vec))))
                            }
                            PipeOp::Filter(predicate) => {
                                // Parallel filter - evaluate predicate in parallel
                                let items: Vec<Value> = arr_ref.clone();
                                drop(arr_ref);

                                let mut result_vec = Vec::new();
                                for item in items.iter() {
                                    self.environment
                                        .borrow_mut()
                                        .define("_".to_string(), item.clone());
                                    let pred_result = self.evaluate(predicate)?;
                                    if self.is_truthy(&pred_result) {
                                        result_vec.push(item.clone());
                                    }
                                }
                                Ok(Value::Array(Rc::new(RefCell::new(result_vec))))
                            }
                            _ => {
                                // For other operations, just apply them normally
                                drop(arr_ref);
                                self.apply_pipe_op(Value::Array(arr), inner_op)
                            }
                        }
                    }
                    _ => {
                        // For non-arrays, just apply the inner operation
                        self.apply_pipe_op(value, inner_op)
                    }
                }
            }
            PipeOp::Gpu(inner_op) => {
                // ⊛ - GPU compute shader execution
                // This is a placeholder that falls back to CPU execution
                // In production, this would:
                // 1. Generate SPIR-V/WGSL compute shader
                // 2. Submit to GPU via wgpu/vulkan
                // 3. Read back results
                match value {
                    Value::Array(arr) => {
                        // For now, emit a hint that GPU execution would occur
                        // and fall back to CPU
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[GPU] Would execute {:?} on GPU, falling back to CPU",
                            inner_op
                        );

                        self.apply_pipe_op(Value::Array(arr), inner_op)
                    }
                    _ => self.apply_pipe_op(value, inner_op),
                }
            }

            // ==========================================
            // Protocol Operations - Sigil-native networking
            // All protocol results are wrapped with Reported evidentiality
            // since network data comes from external sources ("hearsay")
            // ==========================================
            PipeOp::Send(data_expr) => {
                // |send{data} or |⇒{data} - Send data over a connection
                // The value should be a connection object
                let data = self.evaluate(data_expr)?;

                // Create a protocol response with Reported evidentiality
                // In production, this would actually send data over the network
                let response = self.protocol_send(&value, &data)?;

                // Wrap in Reported evidentiality - network responses are hearsay
                Ok(self.wrap_reported(response))
            }

            PipeOp::Recv => {
                // |recv or |⇐ - Receive data from a connection
                // The value should be a connection object

                // In production, this would actually receive data from the network
                let response = self.protocol_recv(&value)?;

                // Wrap in Reported evidentiality - network data is hearsay
                Ok(self.wrap_reported(response))
            }

            PipeOp::Stream(handler_expr) => {
                // |stream{handler} or |≋{handler} - Stream data with a handler
                let handler = self.evaluate(handler_expr)?;

                // Create a streaming iterator over network data
                // Each element will be wrapped in Reported evidentiality
                let stream = self.protocol_stream(&value, &handler)?;
                Ok(stream)
            }

            PipeOp::Connect(config_expr) => {
                // |connect or |connect{config} or |⊸{config} - Establish connection
                let config = match config_expr {
                    Some(expr) => Some(self.evaluate(expr)?),
                    None => None,
                };

                // Create a connection object
                let connection = self.protocol_connect(&value, config.as_ref())?;
                Ok(connection)
            }

            PipeOp::Close => {
                // |close or |⊗ - Close connection gracefully
                self.protocol_close(&value)?;
                Ok(Value::Null)
            }

            PipeOp::Header {
                name,
                value: value_expr,
            } => {
                // |header{name, value} - Add/set header on request
                let header_name = self.evaluate(name)?;
                let header_value = self.evaluate(value_expr)?;

                // Add header to the request builder
                self.protocol_add_header(value, &header_name, &header_value)
            }

            PipeOp::Body(data_expr) => {
                // |body{data} - Set request body
                let body_data = self.evaluate(data_expr)?;

                // Set body on the request builder
                self.protocol_set_body(value, &body_data)
            }

            PipeOp::Timeout(ms_expr) => {
                // |timeout{ms} or |⏱{ms} - Set operation timeout
                let ms = self.evaluate(ms_expr)?;

                // Set timeout on the request/connection
                self.protocol_set_timeout(value, &ms)
            }

            PipeOp::Retry { count, strategy } => {
                // |retry{count} or |retry{count, strategy} - Set retry policy
                let retry_count = self.evaluate(count)?;
                let retry_strategy = match strategy {
                    Some(s) => Some(self.evaluate(s)?),
                    None => None,
                };

                // Set retry policy on the request
                self.protocol_set_retry(value, &retry_count, retry_strategy.as_ref())
            }

            // ==========================================
            // Evidence Promotion Operations
            // ==========================================
            PipeOp::Validate {
                predicate,
                target_evidence,
            } => {
                // |validate!{predicate} - validate and promote evidence
                // Execute the predicate with the current value
                let predicate_result = match predicate.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        if let Some(param) = params.first() {
                            let param_name = match &param.pattern {
                                Pattern::Ident { name, .. } => name.name.clone(),
                                _ => "it".to_string(),
                            };
                            self.environment
                                .borrow_mut()
                                .define(param_name, value.clone());
                        }
                        self.evaluate(body)?
                    }
                    _ => self.evaluate(predicate)?,
                };

                // Check if validation passed
                match predicate_result {
                    Value::Bool(true) => {
                        // Validation passed: promote evidence
                        let target_ev = match target_evidence {
                            Evidentiality::Known => Evidence::Known,
                            Evidentiality::Uncertain => Evidence::Uncertain,
                            Evidentiality::Reported => Evidence::Reported,
                            Evidentiality::Paradox => Evidence::Paradox,
                        };
                        let inner = match value {
                            Value::Evidential { value: v, .. } => *v,
                            v => v,
                        };
                        Ok(Value::Evidential {
                            value: Box::new(inner),
                            evidence: target_ev,
                        })
                    }
                    Value::Bool(false) => Err(RuntimeError::new(
                        "validation failed: predicate returned false",
                    )),
                    _ => Err(RuntimeError::new("validation predicate must return bool")),
                }
            }

            PipeOp::Assume {
                reason,
                target_evidence,
            } => {
                // |assume!("reason") - explicitly assume evidence (with audit trail)
                let reason_str: Rc<String> = if let Some(r) = reason {
                    match self.evaluate(r)? {
                        Value::String(s) => s,
                        _ => Rc::new("<no reason>".to_string()),
                    }
                } else {
                    Rc::new("<no reason>".to_string())
                };

                // Log the assumption for audit purposes
                #[cfg(debug_assertions)]
                eprintln!(
                    "[AUDIT] Evidence assumption: {} - reason: {}",
                    match target_evidence {
                        Evidentiality::Known => "!",
                        Evidentiality::Uncertain => "?",
                        Evidentiality::Reported => "~",
                        Evidentiality::Paradox => "‽",
                    },
                    reason_str
                );

                let target_ev = match target_evidence {
                    Evidentiality::Known => Evidence::Known,
                    Evidentiality::Uncertain => Evidence::Uncertain,
                    Evidentiality::Reported => Evidence::Reported,
                    Evidentiality::Paradox => Evidence::Paradox,
                };

                let inner = match value {
                    Value::Evidential { value: v, .. } => *v,
                    v => v,
                };

                Ok(Value::Evidential {
                    value: Box::new(inner),
                    evidence: target_ev,
                })
            }

            PipeOp::AssertEvidence(expected) => {
                // |assert_evidence!{!} - assert evidence level
                let actual_evidence = match &value {
                    Value::Evidential { evidence, .. } => evidence.clone(),
                    _ => Evidence::Known,
                };

                let expected_ev = match expected {
                    Evidentiality::Known => Evidence::Known,
                    Evidentiality::Uncertain => Evidence::Uncertain,
                    Evidentiality::Reported => Evidence::Reported,
                    Evidentiality::Paradox => Evidence::Paradox,
                };

                // Check if actual satisfies expected
                let satisfies = match (&actual_evidence, &expected_ev) {
                    (Evidence::Known, _) => true,
                    (
                        Evidence::Uncertain,
                        Evidence::Uncertain | Evidence::Reported | Evidence::Paradox,
                    ) => true,
                    (Evidence::Reported, Evidence::Reported | Evidence::Paradox) => true,
                    (Evidence::Paradox, Evidence::Paradox) => true,
                    _ => false,
                };

                if satisfies {
                    Ok(value)
                } else {
                    Err(RuntimeError::new(format!(
                        "evidence assertion failed: expected {:?}, found {:?}",
                        expected_ev, actual_evidence
                    )))
                }
            }

            // ==========================================
            // Scope Functions (Kotlin-inspired)
            // ==========================================
            PipeOp::Also(func) => {
                // |also{f} - execute side effect, return original value
                // Execute the function with the value for side effects
                match func.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        if let Some(param) = params.first() {
                            let param_name = match &param.pattern {
                                Pattern::Ident { name, .. } => name.name.clone(),
                                _ => "it".to_string(),
                            };
                            self.environment
                                .borrow_mut()
                                .define(param_name, value.clone());
                        }
                        // Execute for side effects, ignore result
                        let _ = self.evaluate(body);
                    }
                    _ => {
                        // Call as function with value as argument
                        let _ = self.evaluate(func);
                    }
                }
                // Return original value unchanged
                Ok(value)
            }

            PipeOp::Apply(func) => {
                // |apply{block} - mutate value in place, return modified value
                // The closure receives the value and can modify it
                match func.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        if let Some(param) = params.first() {
                            let param_name = match &param.pattern {
                                Pattern::Ident { name, .. } => name.name.clone(),
                                _ => "it".to_string(),
                            };
                            self.environment
                                .borrow_mut()
                                .define(param_name, value.clone());
                        }
                        // Execute the body - mutations happen via the bound variable
                        let _ = self.evaluate(body);
                    }
                    _ => {
                        let _ = self.evaluate(func);
                    }
                }
                // Return the (potentially modified) value
                Ok(value)
            }

            PipeOp::TakeIf(predicate) => {
                // |take_if{p} - return Some(value) if predicate true, None otherwise
                let predicate_result = match predicate.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        if let Some(param) = params.first() {
                            let param_name = match &param.pattern {
                                Pattern::Ident { name, .. } => name.name.clone(),
                                _ => "it".to_string(),
                            };
                            self.environment
                                .borrow_mut()
                                .define(param_name, value.clone());
                        }
                        self.evaluate(body)?
                    }
                    _ => self.evaluate(predicate)?,
                };

                match predicate_result {
                    Value::Bool(true) => Ok(Value::Variant {
                        enum_name: "Option".to_string(),
                        variant_name: "Some".to_string(),
                        fields: Some(Rc::new(vec![value])),
                    }),
                    Value::Bool(false) => Ok(Value::Variant {
                        enum_name: "Option".to_string(),
                        variant_name: "None".to_string(),
                        fields: None,
                    }),
                    _ => Err(RuntimeError::new("take_if predicate must return bool")),
                }
            }

            PipeOp::TakeUnless(predicate) => {
                // |take_unless{p} - return Some(value) if predicate false, None otherwise
                let predicate_result = match predicate.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        if let Some(param) = params.first() {
                            let param_name = match &param.pattern {
                                Pattern::Ident { name, .. } => name.name.clone(),
                                _ => "it".to_string(),
                            };
                            self.environment
                                .borrow_mut()
                                .define(param_name, value.clone());
                        }
                        self.evaluate(body)?
                    }
                    _ => self.evaluate(predicate)?,
                };

                match predicate_result {
                    Value::Bool(false) => Ok(Value::Variant {
                        enum_name: "Option".to_string(),
                        variant_name: "Some".to_string(),
                        fields: Some(Rc::new(vec![value])),
                    }),
                    Value::Bool(true) => Ok(Value::Variant {
                        enum_name: "Option".to_string(),
                        variant_name: "None".to_string(),
                        fields: None,
                    }),
                    _ => Err(RuntimeError::new("take_unless predicate must return bool")),
                }
            }

            PipeOp::Let(func) => {
                // |let{f} - transform value (alias for map/transform)
                match func.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        if let Some(param) = params.first() {
                            let param_name = match &param.pattern {
                                Pattern::Ident { name, .. } => name.name.clone(),
                                _ => "it".to_string(),
                            };
                            self.environment
                                .borrow_mut()
                                .define(param_name, value.clone());
                        }
                        self.evaluate(body)
                    }
                    _ => self.evaluate(func),
                }
            }

            // ==========================================
            // Mathematical & APL-Inspired Operations
            // ==========================================
            PipeOp::All(pred) => {
                // |∀{p} - check if ALL elements satisfy predicate
                match value {
                    Value::Array(arr) => {
                        for elem in arr.borrow().iter() {
                            self.environment
                                .borrow_mut()
                                .define("_".to_string(), elem.clone());
                            let result = self.evaluate(pred)?;
                            if !self.is_truthy(&result) {
                                return Ok(Value::Bool(false));
                            }
                        }
                        Ok(Value::Bool(true))
                    }
                    _ => Err(RuntimeError::new("All requires array")),
                }
            }

            PipeOp::Any(pred) => {
                // |∃{p} - check if ANY element satisfies predicate
                match value {
                    Value::Array(arr) => {
                        for elem in arr.borrow().iter() {
                            self.environment
                                .borrow_mut()
                                .define("_".to_string(), elem.clone());
                            let result = self.evaluate(pred)?;
                            if self.is_truthy(&result) {
                                return Ok(Value::Bool(true));
                            }
                        }
                        Ok(Value::Bool(false))
                    }
                    _ => Err(RuntimeError::new("Any requires array")),
                }
            }

            PipeOp::Compose(f) => {
                // |∘{f} - function composition / apply function
                self.environment.borrow_mut().define("_".to_string(), value);
                self.evaluate(f)
            }

            PipeOp::Zip(other_expr) => {
                // |⋈{other} - zip with another collection
                let other = self.evaluate(other_expr)?;
                match (value, other) {
                    (Value::Array(arr1), Value::Array(arr2)) => {
                        let zipped: Vec<Value> = arr1
                            .borrow()
                            .iter()
                            .zip(arr2.borrow().iter())
                            .map(|(a, b)| Value::Tuple(Rc::new(vec![a.clone(), b.clone()])))
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(zipped))))
                    }
                    _ => Err(RuntimeError::new("Zip requires two arrays")),
                }
            }

            PipeOp::Scan(f) => {
                // |∫{f} - cumulative fold (scan)
                match value {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        if arr.is_empty() {
                            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
                        }
                        let mut results = vec![arr[0].clone()];
                        let mut acc = arr[0].clone();
                        for elem in arr.iter().skip(1) {
                            self.environment
                                .borrow_mut()
                                .define("acc".to_string(), acc.clone());
                            self.environment
                                .borrow_mut()
                                .define("_".to_string(), elem.clone());
                            acc = self.evaluate(f)?;
                            results.push(acc.clone());
                        }
                        Ok(Value::Array(Rc::new(RefCell::new(results))))
                    }
                    _ => Err(RuntimeError::new("Scan requires array")),
                }
            }

            PipeOp::Diff => {
                // |∂ - differences between adjacent elements
                match value {
                    Value::Array(arr) => {
                        let arr = arr.borrow();
                        if arr.len() < 2 {
                            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
                        }
                        let mut diffs = Vec::new();
                        for i in 1..arr.len() {
                            let diff = self.subtract_values(&arr[i], &arr[i - 1])?;
                            diffs.push(diff);
                        }
                        Ok(Value::Array(Rc::new(RefCell::new(diffs))))
                    }
                    _ => Err(RuntimeError::new("Diff requires array")),
                }
            }

            PipeOp::Gradient(var_expr) => {
                // |∇{var} - automatic differentiation
                // For now, just a placeholder - real autodiff requires tape recording
                let _ = var_expr;
                Ok(Value::Float(0.0)) // TODO: Implement real autodiff
            }

            PipeOp::SortAsc => {
                // |⍋ - sort ascending
                match value {
                    Value::Array(arr) => {
                        let mut v = arr.borrow().clone();
                        v.sort_by(|a, b| self.compare_values(a, b, &None));
                        Ok(Value::Array(Rc::new(RefCell::new(v))))
                    }
                    _ => Err(RuntimeError::new("SortAsc requires array")),
                }
            }

            PipeOp::SortDesc => {
                // |⍒ - sort descending
                match value {
                    Value::Array(arr) => {
                        let mut v = arr.borrow().clone();
                        v.sort_by(|a, b| self.compare_values(b, a, &None));
                        Ok(Value::Array(Rc::new(RefCell::new(v))))
                    }
                    _ => Err(RuntimeError::new("SortDesc requires array")),
                }
            }

            PipeOp::Reverse => {
                // |⌽ - reverse collection
                match value {
                    Value::Array(arr) => {
                        let mut v = arr.borrow().clone();
                        v.reverse();
                        Ok(Value::Array(Rc::new(RefCell::new(v))))
                    }
                    _ => Err(RuntimeError::new("Reverse requires array")),
                }
            }

            PipeOp::Cycle(n_expr) => {
                // |↻{n} - repeat collection n times
                match value {
                    Value::Array(arr) => {
                        let n_val = self.evaluate(n_expr)?;
                        let n = match n_val {
                            Value::Int(i) => i as usize,
                            _ => return Err(RuntimeError::new("Cycle count must be integer")),
                        };
                        let arr = arr.borrow();
                        let cycled: Vec<Value> =
                            arr.iter().cloned().cycle().take(arr.len() * n).collect();
                        Ok(Value::Array(Rc::new(RefCell::new(cycled))))
                    }
                    _ => Err(RuntimeError::new("Cycle requires array")),
                }
            }

            PipeOp::Windows(n_expr) => {
                // |⌺{n} - sliding windows
                match value {
                    Value::Array(arr) => {
                        let n_val = self.evaluate(n_expr)?;
                        let n = match n_val {
                            Value::Int(i) => i as usize,
                            _ => return Err(RuntimeError::new("Window size must be integer")),
                        };
                        let arr = arr.borrow();
                        let windows: Vec<Value> = arr
                            .windows(n)
                            .map(|w| Value::Array(Rc::new(RefCell::new(w.to_vec()))))
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(windows))))
                    }
                    _ => Err(RuntimeError::new("Windows requires array")),
                }
            }

            PipeOp::Chunks(n_expr) => {
                // |⊞{n} - split into chunks
                match value {
                    Value::Array(arr) => {
                        let n_val = self.evaluate(n_expr)?;
                        let n = match n_val {
                            Value::Int(i) => i as usize,
                            _ => return Err(RuntimeError::new("Chunk size must be integer")),
                        };
                        let arr = arr.borrow();
                        let chunks: Vec<Value> = arr
                            .chunks(n)
                            .map(|c| Value::Array(Rc::new(RefCell::new(c.to_vec()))))
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(chunks))))
                    }
                    _ => Err(RuntimeError::new("Chunks requires array")),
                }
            }

            PipeOp::Flatten => {
                // |⋳ - flatten nested collection
                match value {
                    Value::Array(arr) => {
                        let mut flat = Vec::new();
                        for elem in arr.borrow().iter() {
                            match elem {
                                Value::Array(inner) => {
                                    flat.extend(inner.borrow().iter().cloned());
                                }
                                other => flat.push(other.clone()),
                            }
                        }
                        Ok(Value::Array(Rc::new(RefCell::new(flat))))
                    }
                    _ => Err(RuntimeError::new("Flatten requires array")),
                }
            }

            PipeOp::Unique => {
                // |∪ - remove duplicates
                match value {
                    Value::Array(arr) => {
                        let mut seen = std::collections::HashSet::new();
                        let mut unique = Vec::new();
                        for elem in arr.borrow().iter() {
                            let key = format!("{:?}", elem);
                            if seen.insert(key) {
                                unique.push(elem.clone());
                            }
                        }
                        Ok(Value::Array(Rc::new(RefCell::new(unique))))
                    }
                    _ => Err(RuntimeError::new("Unique requires array")),
                }
            }

            PipeOp::Enumerate => {
                // |⍳ - pair with indices
                match value {
                    Value::Array(arr) => {
                        let enumerated: Vec<Value> = arr
                            .borrow()
                            .iter()
                            .enumerate()
                            .map(|(i, v)| {
                                Value::Tuple(Rc::new(vec![Value::Int(i as i64), v.clone()]))
                            })
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(enumerated))))
                    }
                    _ => Err(RuntimeError::new("Enumerate requires array")),
                }
            }
        }
    }

    // ==========================================
    // Protocol Helper Methods
    // ==========================================

    /// Wrap a value in Reported evidentiality
    /// Network data is "hearsay" - it comes from external sources we can't verify
    fn wrap_reported(&self, value: Value) -> Value {
        Value::Evidential {
            value: Box::new(value),
            evidence: Evidence::Reported,
        }
    }

    /// Send data over a protocol connection
    fn protocol_send(&mut self, connection: &Value, data: &Value) -> Result<Value, RuntimeError> {
        // Extract connection info and send data
        match connection {
            Value::Map(obj) => {
                let obj = obj.borrow();
                if let Some(Value::String(protocol)) = obj.get("__protocol__") {
                    match protocol.as_str() {
                        "http" | "https" => {
                            // For HTTP, "send" means execute the request
                            // The data becomes the body
                            #[cfg(debug_assertions)]
                            eprintln!("[HTTP] Would send request with body: {:?}", data);
                            Ok(Value::Map(Rc::new(RefCell::new({
                                let mut response = HashMap::new();
                                response.insert("status".to_string(), Value::Int(200));
                                response.insert("body".to_string(), data.clone());
                                response.insert(
                                    "__protocol__".to_string(),
                                    Value::String(Rc::new("http_response".to_string())),
                                );
                                response
                            }))))
                        }
                        "ws" | "wss" => {
                            // For WebSocket, send a message
                            #[cfg(debug_assertions)]
                            eprintln!("[WebSocket] Would send message: {:?}", data);
                            Ok(Value::Bool(true)) // Message sent successfully
                        }
                        "grpc" => {
                            // For gRPC, send the request message
                            #[cfg(debug_assertions)]
                            eprintln!("[gRPC] Would send message: {:?}", data);
                            Ok(Value::Map(Rc::new(RefCell::new({
                                let mut response = HashMap::new();
                                response.insert("status".to_string(), Value::Int(0)); // OK
                                response.insert("message".to_string(), data.clone());
                                response.insert(
                                    "__protocol__".to_string(),
                                    Value::String(Rc::new("grpc_response".to_string())),
                                );
                                response
                            }))))
                        }
                        "kafka" => {
                            // For Kafka, produce a message
                            #[cfg(debug_assertions)]
                            eprintln!("[Kafka] Would produce message: {:?}", data);
                            Ok(Value::Map(Rc::new(RefCell::new({
                                let mut result = HashMap::new();
                                result.insert("partition".to_string(), Value::Int(0));
                                result.insert("offset".to_string(), Value::Int(42));
                                result
                            }))))
                        }
                        _ => Err(RuntimeError::new(format!("Unknown protocol: {}", protocol))),
                    }
                } else {
                    Err(RuntimeError::new(
                        "Connection object missing __protocol__ field",
                    ))
                }
            }
            _ => Err(RuntimeError::new("send requires a connection object")),
        }
    }

    /// Receive data from a protocol connection
    fn protocol_recv(&mut self, connection: &Value) -> Result<Value, RuntimeError> {
        match connection {
            Value::Map(obj) => {
                let obj = obj.borrow();
                if let Some(Value::String(protocol)) = obj.get("__protocol__") {
                    match protocol.as_str() {
                        "ws" | "wss" => {
                            // For WebSocket, receive a message
                            #[cfg(debug_assertions)]
                            eprintln!("[WebSocket] Would receive message");
                            Ok(Value::String(Rc::new("received message".to_string())))
                        }
                        "kafka" => {
                            // For Kafka, consume a message
                            #[cfg(debug_assertions)]
                            eprintln!("[Kafka] Would consume message");
                            Ok(Value::Map(Rc::new(RefCell::new({
                                let mut msg = HashMap::new();
                                msg.insert("key".to_string(), Value::Null);
                                msg.insert(
                                    "value".to_string(),
                                    Value::String(Rc::new("consumed message".to_string())),
                                );
                                msg.insert("partition".to_string(), Value::Int(0));
                                msg.insert("offset".to_string(), Value::Int(100));
                                msg
                            }))))
                        }
                        "grpc" => {
                            // For gRPC streaming, receive next message
                            #[cfg(debug_assertions)]
                            eprintln!("[gRPC] Would receive stream message");
                            Ok(Value::Map(Rc::new(RefCell::new({
                                let mut msg = HashMap::new();
                                msg.insert(
                                    "data".to_string(),
                                    Value::String(Rc::new("stream data".to_string())),
                                );
                                msg
                            }))))
                        }
                        _ => Err(RuntimeError::new(format!(
                            "recv not supported for protocol: {}",
                            protocol
                        ))),
                    }
                } else {
                    Err(RuntimeError::new(
                        "Connection object missing __protocol__ field",
                    ))
                }
            }
            _ => Err(RuntimeError::new("recv requires a connection object")),
        }
    }

    /// Create a streaming iterator over protocol data
    fn protocol_stream(
        &mut self,
        connection: &Value,
        _handler: &Value,
    ) -> Result<Value, RuntimeError> {
        // Create a lazy stream that yields values with Reported evidentiality
        match connection {
            Value::Map(obj) => {
                let obj = obj.borrow();
                if let Some(Value::String(protocol)) = obj.get("__protocol__") {
                    #[cfg(debug_assertions)]
                    eprintln!("[{}] Would create stream", protocol);

                    // Return a stream object that can be iterated
                    Ok(Value::Map(Rc::new(RefCell::new({
                        let mut stream = HashMap::new();
                        stream.insert(
                            "__type__".to_string(),
                            Value::String(Rc::new("Stream".to_string())),
                        );
                        stream.insert("__protocol__".to_string(), Value::String(protocol.clone()));
                        stream.insert(
                            "__evidentiality__".to_string(),
                            Value::String(Rc::new("reported".to_string())),
                        );
                        stream
                    }))))
                } else {
                    Err(RuntimeError::new(
                        "Connection object missing __protocol__ field",
                    ))
                }
            }
            _ => Err(RuntimeError::new("stream requires a connection object")),
        }
    }

    /// Establish a protocol connection
    fn protocol_connect(
        &mut self,
        target: &Value,
        _config: Option<&Value>,
    ) -> Result<Value, RuntimeError> {
        match target {
            Value::String(url) => {
                // Parse URL to determine protocol
                let protocol = if url.starts_with("wss://") || url.starts_with("ws://") {
                    if url.starts_with("wss://") {
                        "wss"
                    } else {
                        "ws"
                    }
                } else if url.starts_with("https://") || url.starts_with("http://") {
                    if url.starts_with("https://") {
                        "https"
                    } else {
                        "http"
                    }
                } else if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                    "grpc"
                } else if url.starts_with("kafka://") {
                    "kafka"
                } else if url.starts_with("amqp://") || url.starts_with("amqps://") {
                    "amqp"
                } else {
                    "unknown"
                };

                #[cfg(debug_assertions)]
                eprintln!("[{}] Would connect to: {}", protocol, url);

                // Return a connection object
                Ok(Value::Map(Rc::new(RefCell::new({
                    let mut conn = HashMap::new();
                    conn.insert(
                        "__protocol__".to_string(),
                        Value::String(Rc::new(protocol.to_string())),
                    );
                    conn.insert("url".to_string(), Value::String(url.clone()));
                    conn.insert("connected".to_string(), Value::Bool(true));
                    conn
                }))))
            }
            Value::Map(obj) => {
                // Already a connection config object
                let mut conn = obj.borrow().clone();
                conn.insert("connected".to_string(), Value::Bool(true));
                Ok(Value::Map(Rc::new(RefCell::new(conn))))
            }
            _ => Err(RuntimeError::new(
                "connect requires URL string or config object",
            )),
        }
    }

    /// Close a protocol connection
    fn protocol_close(&mut self, connection: &Value) -> Result<(), RuntimeError> {
        match connection {
            Value::Map(obj) => {
                let mut obj = obj.borrow_mut();
                if let Some(Value::String(protocol)) = obj.get("__protocol__").cloned() {
                    #[cfg(debug_assertions)]
                    eprintln!("[{}] Would close connection", protocol);
                    obj.insert("connected".to_string(), Value::Bool(false));
                    Ok(())
                } else {
                    Err(RuntimeError::new(
                        "Connection object missing __protocol__ field",
                    ))
                }
            }
            _ => Err(RuntimeError::new("close requires a connection object")),
        }
    }

    /// Add a header to a protocol request
    fn protocol_add_header(
        &mut self,
        mut request: Value,
        name: &Value,
        header_value: &Value,
    ) -> Result<Value, RuntimeError> {
        let name_str = match name {
            Value::String(s) => (**s).clone(),
            _ => return Err(RuntimeError::new("Header name must be a string")),
        };
        let value_str = match header_value {
            Value::String(s) => (**s).clone(),
            Value::Int(i) => i.to_string(),
            _ => return Err(RuntimeError::new("Header value must be string or int")),
        };

        match &mut request {
            Value::Map(obj) => {
                let mut obj = obj.borrow_mut();

                // Get or create headers map
                let headers = obj
                    .entry("headers".to_string())
                    .or_insert_with(|| Value::Map(Rc::new(RefCell::new(HashMap::new()))));

                if let Value::Map(headers_obj) = headers {
                    headers_obj
                        .borrow_mut()
                        .insert(name_str, Value::String(Rc::new(value_str)));
                }
                drop(obj);
                Ok(request)
            }
            _ => Err(RuntimeError::new("header requires a request object")),
        }
    }

    /// Set the body of a protocol request
    fn protocol_set_body(
        &mut self,
        mut request: Value,
        body: &Value,
    ) -> Result<Value, RuntimeError> {
        match &mut request {
            Value::Map(obj) => {
                obj.borrow_mut().insert("body".to_string(), body.clone());
                Ok(request)
            }
            _ => Err(RuntimeError::new("body requires a request object")),
        }
    }

    /// Set the timeout for a protocol operation
    fn protocol_set_timeout(
        &mut self,
        mut request: Value,
        ms: &Value,
    ) -> Result<Value, RuntimeError> {
        let timeout_ms = match ms {
            Value::Int(n) => *n,
            Value::Float(f) => *f as i64,
            _ => return Err(RuntimeError::new("Timeout must be a number (milliseconds)")),
        };

        match &mut request {
            Value::Map(obj) => {
                obj.borrow_mut()
                    .insert("timeout_ms".to_string(), Value::Int(timeout_ms));
                Ok(request)
            }
            _ => Err(RuntimeError::new("timeout requires a request object")),
        }
    }

    /// Set the retry policy for a protocol operation
    fn protocol_set_retry(
        &mut self,
        mut request: Value,
        count: &Value,
        strategy: Option<&Value>,
    ) -> Result<Value, RuntimeError> {
        let retry_count = match count {
            Value::Int(n) => *n,
            _ => return Err(RuntimeError::new("Retry count must be an integer")),
        };

        match &mut request {
            Value::Map(obj) => {
                let mut obj = obj.borrow_mut();
                obj.insert("retry_count".to_string(), Value::Int(retry_count));
                if let Some(strat) = strategy {
                    obj.insert("retry_strategy".to_string(), strat.clone());
                }
                drop(obj);
                Ok(request)
            }
            _ => Err(RuntimeError::new("retry requires a request object")),
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

    fn min_values(&self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Err(RuntimeError::new("Cannot find min of empty array"));
                }
                let mut min = arr[0].clone();
                for item in arr.iter().skip(1) {
                    min = match (&min, item) {
                        (Value::Int(a), Value::Int(b)) => {
                            if *b < *a {
                                Value::Int(*b)
                            } else {
                                Value::Int(*a)
                            }
                        }
                        (Value::Float(a), Value::Float(b)) => {
                            if *b < *a {
                                Value::Float(*b)
                            } else {
                                Value::Float(*a)
                            }
                        }
                        (Value::Int(a), Value::Float(b)) => {
                            let af = *a as f64;
                            if *b < af {
                                Value::Float(*b)
                            } else {
                                Value::Float(af)
                            }
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            let bf = *b as f64;
                            if bf < *a {
                                Value::Float(bf)
                            } else {
                                Value::Float(*a)
                            }
                        }
                        _ => {
                            return Err(RuntimeError::new("Cannot find min of non-numeric values"))
                        }
                    };
                }
                Ok(min)
            }
            _ => Err(RuntimeError::new("min requires array")),
        }
    }

    fn max_values(&self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Err(RuntimeError::new("Cannot find max of empty array"));
                }
                let mut max = arr[0].clone();
                for item in arr.iter().skip(1) {
                    max = match (&max, item) {
                        (Value::Int(a), Value::Int(b)) => {
                            if *b > *a {
                                Value::Int(*b)
                            } else {
                                Value::Int(*a)
                            }
                        }
                        (Value::Float(a), Value::Float(b)) => {
                            if *b > *a {
                                Value::Float(*b)
                            } else {
                                Value::Float(*a)
                            }
                        }
                        (Value::Int(a), Value::Float(b)) => {
                            let af = *a as f64;
                            if *b > af {
                                Value::Float(*b)
                            } else {
                                Value::Float(af)
                            }
                        }
                        (Value::Float(a), Value::Int(b)) => {
                            let bf = *b as f64;
                            if bf > *a {
                                Value::Float(bf)
                            } else {
                                Value::Float(*a)
                            }
                        }
                        _ => {
                            return Err(RuntimeError::new("Cannot find max of non-numeric values"))
                        }
                    };
                }
                Ok(max)
            }
            _ => Err(RuntimeError::new("max requires array")),
        }
    }

    fn concat_values(&self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let arr = arr.borrow();
                if arr.is_empty() {
                    return Ok(Value::String(Rc::new(String::new())));
                }
                // Determine if we're concatenating strings or arrays
                match &arr[0] {
                    Value::String(_) => {
                        let mut result = String::new();
                        for item in arr.iter() {
                            if let Value::String(s) = item {
                                result.push_str(s);
                            } else {
                                return Err(RuntimeError::new(
                                    "concat requires all elements to be strings",
                                ));
                            }
                        }
                        Ok(Value::String(Rc::new(result)))
                    }
                    Value::Array(_) => {
                        let mut result = Vec::new();
                        for item in arr.iter() {
                            if let Value::Array(inner) = item {
                                result.extend(inner.borrow().iter().cloned());
                            } else {
                                return Err(RuntimeError::new(
                                    "concat requires all elements to be arrays",
                                ));
                            }
                        }
                        Ok(Value::Array(Rc::new(RefCell::new(result))))
                    }
                    _ => Err(RuntimeError::new("concat requires strings or arrays")),
                }
            }
            _ => Err(RuntimeError::new("concat requires array")),
        }
    }

    fn all_values(&self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let arr = arr.borrow();
                for item in arr.iter() {
                    match item {
                        Value::Bool(b) => {
                            if !*b {
                                return Ok(Value::Bool(false));
                            }
                        }
                        _ => return Err(RuntimeError::new("all requires array of booleans")),
                    }
                }
                Ok(Value::Bool(true))
            }
            _ => Err(RuntimeError::new("all requires array")),
        }
    }

    fn any_values(&self, value: Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let arr = arr.borrow();
                for item in arr.iter() {
                    match item {
                        Value::Bool(b) => {
                            if *b {
                                return Ok(Value::Bool(true));
                            }
                        }
                        _ => return Err(RuntimeError::new("any requires array of booleans")),
                    }
                }
                Ok(Value::Bool(false))
            }
            _ => Err(RuntimeError::new("any requires array")),
        }
    }

    fn compare_values(&self, a: &Value, b: &Value, _field: &Option<Ident>) -> std::cmp::Ordering {
        // Simple comparison for now
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Subtract two values (for diff operation)
    fn subtract_values(&self, a: &Value, b: &Value) -> Result<Value, RuntimeError> {
        match (a, b) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
            _ => Err(RuntimeError::new(format!(
                "Cannot subtract {:?} from {:?}",
                b, a
            ))),
        }
    }

    fn eval_closure(
        &mut self,
        params: &[ClosureParam],
        body: &Expr,
    ) -> Result<Value, RuntimeError> {
        let param_names: Vec<String> = params
            .iter()
            .map(|p| match &p.pattern {
                Pattern::Ident { name, .. } => name.name.clone(),
                _ => "_".to_string(),
            })
            .collect();

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
        let name = path
            .segments
            .iter()
            .map(|s| s.ident.name.as_str())
            .collect::<Vec<_>>()
            .join("::");

        let mut field_values = HashMap::new();
        for field in fields {
            let value = match &field.value {
                Some(expr) => self.evaluate(expr)?,
                None => self
                    .environment
                    .borrow()
                    .get(&field.name.name)
                    .ok_or_else(|| {
                        RuntimeError::new(format!("Unknown variable: {}", field.name.name))
                    })?,
            };
            field_values.insert(field.name.name.clone(), value);
        }

        Ok(Value::Struct {
            name,
            fields: Rc::new(RefCell::new(field_values)),
        })
    }

    /// Extract evidentiality from a value (recursively unwraps Evidential wrapper)
    fn extract_evidence(value: &Value) -> Option<Evidence> {
        match value {
            Value::Evidential { evidence, .. } => Some(*evidence),
            _ => None,
        }
    }

    /// Extract affect from a value
    fn extract_affect(value: &Value) -> Option<&RuntimeAffect> {
        match value {
            Value::Affective { affect, .. } => Some(affect),
            _ => None,
        }
    }

    /// Derive evidence from affect markers.
    /// Sarcasm implies uncertainty (meaning is inverted).
    /// Confidence directly maps to evidence levels.
    fn affect_to_evidence(affect: &RuntimeAffect) -> Option<Evidence> {
        // Sarcasm indicates the literal meaning shouldn't be trusted
        if affect.sarcasm {
            return Some(Evidence::Uncertain);
        }

        // Confidence maps directly to evidence
        match affect.confidence {
            Some(RuntimeConfidence::High) => Some(Evidence::Known),
            Some(RuntimeConfidence::Low) => Some(Evidence::Uncertain),
            Some(RuntimeConfidence::Medium) | None => None,
        }
    }

    /// Combine two evidence levels, returning the "worst" (most uncertain) one.
    /// Order: Known < Uncertain < Reported < Paradox
    fn combine_evidence(a: Option<Evidence>, b: Option<Evidence>) -> Option<Evidence> {
        match (a, b) {
            (None, None) => None,
            (Some(e), None) | (None, Some(e)) => Some(e),
            (Some(a), Some(b)) => {
                let rank = |e: Evidence| match e {
                    Evidence::Known => 0,
                    Evidence::Uncertain => 1,
                    Evidence::Reported => 2,
                    Evidence::Paradox => 3,
                };
                if rank(a) >= rank(b) {
                    Some(a)
                } else {
                    Some(b)
                }
            }
        }
    }

    /// Unwrap an evidential value to get the inner value for display
    fn unwrap_evidential(value: &Value) -> &Value {
        match value {
            Value::Evidential { value: inner, .. } => Self::unwrap_evidential(inner),
            _ => value,
        }
    }

    /// Unwrap an affective value to get the inner value
    fn unwrap_affective(value: &Value) -> &Value {
        match value {
            Value::Affective { value: inner, .. } => Self::unwrap_affective(inner),
            _ => value,
        }
    }

    /// Unwrap both evidential and affective wrappers
    fn unwrap_value(value: &Value) -> &Value {
        match value {
            Value::Evidential { value: inner, .. } => Self::unwrap_value(inner),
            Value::Affective { value: inner, .. } => Self::unwrap_value(inner),
            _ => value,
        }
    }

    /// Unwrap all wrappers including Ref for deep value access
    fn unwrap_all(value: &Value) -> Value {
        match value {
            Value::Evidential { value: inner, .. } => Self::unwrap_all(inner),
            Value::Affective { value: inner, .. } => Self::unwrap_all(inner),
            Value::Ref(r) => Self::unwrap_all(&r.borrow()),
            _ => value.clone(),
        }
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

    /// Evaluate format! macro - parse format string and arguments
    fn eval_format_macro(&mut self, tokens: &str) -> Result<Value, RuntimeError> {
        // Token string looks like: "\"format string\" , arg1 , arg2"
        // We need to parse this properly

        // Find the format string (first quoted string)
        let tokens = tokens.trim();
        if !tokens.starts_with('"') {
            // No format string - just return the tokens as-is
            return Ok(Value::String(Rc::new(tokens.to_string())));
        }

        // Find the end of the format string
        let mut in_escape = false;
        let mut format_end = 1;
        for (i, c) in tokens[1..].char_indices() {
            if in_escape {
                in_escape = false;
            } else if c == '\\' {
                in_escape = true;
            } else if c == '"' {
                format_end = i + 2; // +1 for starting quote, +1 for this quote
                break;
            }
        }

        let format_str = &tokens[1..format_end-1]; // Remove quotes
        let args_str = if format_end < tokens.len() {
            tokens[format_end..].trim_start_matches(',').trim()
        } else {
            ""
        };

        // Parse and evaluate arguments
        let mut arg_values: Vec<String> = Vec::new();
        if !args_str.is_empty() {
            // Split by comma, but respect parentheses/brackets
            let mut depth = 0;
            let mut current_arg = String::new();
            for c in args_str.chars() {
                match c {
                    '(' | '[' | '{' => {
                        depth += 1;
                        current_arg.push(c);
                    }
                    ')' | ']' | '}' => {
                        depth -= 1;
                        current_arg.push(c);
                    }
                    ',' if depth == 0 => {
                        let arg = current_arg.trim().to_string();
                        if !arg.is_empty() {
                            // Parse and evaluate the argument expression
                            let mut parser = crate::parser::Parser::new(&arg);
                            match parser.parse_expr() {
                                Ok(expr) => {
                                    match self.evaluate(&expr) {
                                        Ok(val) => arg_values.push(self.format_value(&val)),
                                        Err(_) => arg_values.push(arg),
                                    }
                                }
                                Err(_) => arg_values.push(arg),
                            }
                        }
                        current_arg.clear();
                    }
                    _ => current_arg.push(c),
                }
            }
            // Don't forget the last argument
            let arg = current_arg.trim().to_string();
            if !arg.is_empty() {
                let mut parser = crate::parser::Parser::new(&arg);
                match parser.parse_expr() {
                    Ok(expr) => {
                        match self.evaluate(&expr) {
                            Ok(val) => arg_values.push(self.format_value(&val)),
                            Err(_) => arg_values.push(arg),
                        }
                    }
                    Err(_) => arg_values.push(arg),
                }
            }
        }

        // Format the string by replacing {} and {:?} with arguments
        let mut result = String::new();
        let mut arg_idx = 0;
        let mut chars = format_str.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' {
                if chars.peek() == Some(&'{') {
                    // Escaped {{ -> {
                    chars.next();
                    result.push('{');
                } else {
                    // Consume until }
                    let mut placeholder = String::new();
                    while let Some(pc) = chars.next() {
                        if pc == '}' {
                            break;
                        }
                        placeholder.push(pc);
                    }
                    // Insert argument value
                    if arg_idx < arg_values.len() {
                        result.push_str(&arg_values[arg_idx]);
                        arg_idx += 1;
                    } else {
                        result.push_str(&format!("{{{}}}", placeholder));
                    }
                }
            } else if c == '}' {
                if chars.peek() == Some(&'}') {
                    // Escaped }} -> }
                    chars.next();
                    result.push('}');
                } else {
                    result.push('}');
                }
            } else if c == '\\' {
                // Handle escape sequences
                if let Some(next) = chars.next() {
                    match next {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        'r' => result.push('\r'),
                        '\\' => result.push('\\'),
                        '"' => result.push('"'),
                        _ => {
                            result.push('\\');
                            result.push(next);
                        }
                    }
                }
            } else {
                result.push(c);
            }
        }

        Ok(Value::String(Rc::new(result)))
    }

    /// Format a value for display in format!
    fn format_value(&self, value: &Value) -> String {
        match value {
            Value::String(s) => s.to_string(),
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Char(c) => c.to_string(),
            Value::Null => "null".to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.borrow().iter().map(|v| self.format_value(v)).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Tuple(items) => {
                let formatted: Vec<String> = items.iter().map(|v| self.format_value(v)).collect();
                format!("({})", formatted.join(", "))
            }
            Value::Struct { name, fields } => {
                let field_strs: Vec<String> = fields.borrow().iter()
                    .map(|(k, v)| format!("{}: {}", k, self.format_value(v)))
                    .collect();
                format!("{} {{ {} }}", name, field_strs.join(", "))
            }
            Value::Variant { enum_name, variant_name, fields } => {
                match fields {
                    Some(f) if !f.is_empty() => {
                        let formatted: Vec<String> = f.iter().map(|v| self.format_value(v)).collect();
                        format!("{}::{}({})", enum_name, variant_name, formatted.join(", "))
                    }
                    _ => format!("{}::{}", enum_name, variant_name),
                }
            }
            Value::Evidential { value: inner, evidence } => {
                format!("{:?}{}", evidence, self.format_value(inner))
            }
            Value::Ref(r) => self.format_value(&r.borrow()),
            _ => format!("{:?}", value),
        }
    }

    /// Evaluate vec! macro
    fn eval_vec_macro(&mut self, tokens: &str) -> Result<Value, RuntimeError> {
        // Parse comma-separated elements
        let mut elements = Vec::new();
        let mut depth = 0;
        let mut current = String::new();

        for c in tokens.chars() {
            match c {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(c);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    let elem = current.trim().to_string();
                    if !elem.is_empty() {
                        let mut parser = crate::parser::Parser::new(&elem);
                        if let Ok(expr) = parser.parse_expr() {
                            elements.push(self.evaluate(&expr)?);
                        }
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        // Last element
        let elem = current.trim().to_string();
        if !elem.is_empty() {
            let mut parser = crate::parser::Parser::new(&elem);
            if let Ok(expr) = parser.parse_expr() {
                elements.push(self.evaluate(&expr)?);
            }
        }

        Ok(Value::Array(Rc::new(RefCell::new(elements))))
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
        let file = parser
            .parse_file()
            .map_err(|e| RuntimeError::new(e.to_string()))?;
        let mut interp = Interpreter::new();
        interp.execute(&file)
    }

    #[test]
    fn test_arithmetic() {
        assert!(matches!(
            run("fn main() { return 2 + 3; }"),
            Ok(Value::Int(5))
        ));
        assert!(matches!(
            run("fn main() { return 10 - 4; }"),
            Ok(Value::Int(6))
        ));
        assert!(matches!(
            run("fn main() { return 3 * 4; }"),
            Ok(Value::Int(12))
        ));
        assert!(matches!(
            run("fn main() { return 15 / 3; }"),
            Ok(Value::Int(5))
        ));
        assert!(matches!(
            run("fn main() { return 2 ** 10; }"),
            Ok(Value::Int(1024))
        ));
    }

    #[test]
    fn test_variables() {
        assert!(matches!(
            run("fn main() { let x = 42; return x; }"),
            Ok(Value::Int(42))
        ));
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
        assert!(matches!(
            run("fn main() { return [1, 2, 3][1]; }"),
            Ok(Value::Int(2))
        ));
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

    #[test]
    fn test_interpolation_evidentiality_propagation() {
        // Test that evidentiality propagates through string interpolation
        // When an evidential value is interpolated, the result string should carry that evidentiality
        let result = run(r#"
            fn main() {
                let rep = reported(42);

                // Interpolating a reported value should make the string reported
                let s = f"Value: {rep}";
                return s;
            }
        "#);

        match result {
            Ok(Value::Evidential {
                evidence: Evidence::Reported,
                value,
            }) => {
                // The inner value should be a string
                assert!(matches!(*value, Value::String(_)));
            }
            Ok(other) => panic!("Expected Evidential Reported, got {:?}", other),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_interpolation_worst_evidence_wins() {
        // When multiple evidential values are interpolated, the worst evidence level wins
        let result = run(r#"
            fn main() {
                let k = known(1);         // Known is best
                let u = uncertain(2);     // Uncertain is worse

                // Combining known and uncertain should yield uncertain
                let s = f"{k} and {u}";
                return s;
            }
        "#);

        match result {
            Ok(Value::Evidential {
                evidence: Evidence::Uncertain,
                ..
            }) => (),
            Ok(other) => panic!("Expected Evidential Uncertain, got {:?}", other),
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    #[test]
    fn test_interpolation_no_evidential_plain_string() {
        // When no evidential values are interpolated, the result is a plain string
        let result = run(r#"
            fn main() {
                let x = 42;
                let s = f"Value: {x}";
                return s;
            }
        "#);

        match result {
            Ok(Value::String(s)) => {
                assert_eq!(*s, "Value: 42");
            }
            Ok(other) => panic!("Expected plain String, got {:?}", other),
            Err(e) => panic!("Error: {:?}", e),
        }
    }
}

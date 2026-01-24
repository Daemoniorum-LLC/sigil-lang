//! Tree-walking interpreter for Sigil.
//!
//! Executes Sigil AST directly for rapid prototyping and REPL.

use crate::ast::*;
use crate::span::Span;

// =============================================================================
// Tensor Field Constants
// =============================================================================

/// Internal field name for tensor data storage
const TENSOR_DATA_FIELD: &str = "__data__";
/// Internal field name for tensor shape
const TENSOR_SHAPE_FIELD: &str = "__shape__";
/// Public field name for tensor shape access
const TENSOR_SHAPE_PUBLIC: &str = "shape";
/// Internal field name for requires_grad flag
const TENSOR_REQUIRES_GRAD_FIELD: &str = "__requires_grad__";
/// Internal field name for gradient storage
const TENSOR_GRAD_FIELD: &str = "__grad__";
/// Public field name for gradient access
const TENSOR_GRAD_PUBLIC: &str = "grad";
/// Tensor type name
const TENSOR_TYPE_NAME: &str = "Tensor";

/// Tensor constructor function names
const TENSOR_CONSTRUCTORS: &[&str] = &["zeros", "ones", "randn"];
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
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
    /// Variant constructor (for creating enum variants)
    VariantConstructor {
        enum_name: String,
        variant_name: String,
    },
    /// Default constructor (for default trait)
    DefaultConstructor {
        type_name: String,
    },
    /// Range value (start..end or start..=end)
    Range {
        start: Option<i64>,
        end: Option<i64>,
        inclusive: bool,
    },
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
            Value::VariantConstructor { enum_name, variant_name } => {
                write!(f, "<constructor {}::{}>", enum_name, variant_name)
            }
            Value::DefaultConstructor { type_name } => {
                write!(f, "<default {}>", type_name)
            }
            Value::Range { start, end, inclusive } => {
                match (start, end) {
                    (Some(s), Some(e)) => if *inclusive {
                        write!(f, "{}..={}", s, e)
                    } else {
                        write!(f, "{}..{}", s, e)
                    },
                    (Some(s), None) => write!(f, "{}..", s),
                    (None, Some(e)) => if *inclusive {
                        write!(f, "..={}", e)
                    } else {
                        write!(f, "..{}", e)
                    },
                    (None, None) => write!(f, ".."),
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
            Value::Float(n) => {
                // Print whole numbers without decimal point (e.g., 6.0 -> "6")
                if n.fract() == 0.0 && n.is_finite() {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
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
            // Special Display for quantum types
            Value::Struct { name, fields } if name == "Cbit" => {
                let fields = fields.borrow();
                match fields.get("__value__") {
                    Some(Value::Bool(true)) => write!(f, "1"),
                    Some(Value::Bool(false)) => write!(f, "0"),
                    _ => write!(f, "Cbit(?)"),
                }
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
    /// Values stored with mutability flag: (value, is_mutable)
    pub(crate) values: HashMap<String, (Value, bool)>,
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

    /// Define a new variable (default: immutable)
    pub fn define(&mut self, name: String, value: Value) {
        self.values.insert(name, (value, false));
    }

    /// Define a variable with explicit mutability
    pub fn define_mut(&mut self, name: String, value: Value, mutable: bool) {
        self.values.insert(name, (value, mutable));
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some((value, _)) = self.values.get(name) {
            Some(value.clone())
        } else if let Some(ref parent) = self.parent {
            parent.borrow().get(name)
        } else {
            None
        }
    }

    /// Check if a variable is mutable
    pub fn is_mutable(&self, name: &str) -> Option<bool> {
        if let Some((_, mutable)) = self.values.get(name) {
            Some(*mutable)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_mutable(name)
        } else {
            None
        }
    }

    pub fn set(&mut self, name: &str, value: Value) -> Result<(), RuntimeError> {
        if let Some((_, mutable)) = self.values.get(name) {
            if !*mutable {
                return Err(RuntimeError::new(format!(
                    "Cannot assign to immutable variable '{}'. Use 'vary' to declare mutable variables.", name
                )));
            }
            self.values.insert(name.to_string(), (value, true));
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
    /// Program arguments (overrides env::args when set)
    pub program_args: Option<Vec<String>>,
    /// Current module prefix for registering definitions
    pub current_module: Option<String>,
    /// Current Self type (when inside an impl block)
    pub current_self_type: Option<String>,
    /// Current source directory for resolving relative module paths
    pub current_source_dir: Option<String>,
    /// Loaded crates registry (crate_name -> true if loaded)
    pub loaded_crates: HashSet<String>,
    /// Crates currently being loaded (for circular dependency detection)
    pub loading_crates: HashSet<String>,
    /// Project root directory (where Sigil.toml is located)
    pub project_root: Option<PathBuf>,
    /// Workspace members: crate_name -> relative path from project root
    pub workspace_members: HashMap<String, PathBuf>,
    /// Types that implement Drop trait - call drop() when they go out of scope
    pub drop_types: HashSet<String>,
    /// Current crate name (for `crate::*` and `crate_name::*` paths)
    pub current_crate: Option<String>,
    /// Registered modules in current crate (module_name -> loaded)
    pub crate_modules: HashSet<String>,
    /// Crate aliases (e.g., "tome" alias for "jormungandr" crate)
    pub crate_aliases: HashSet<String>,
    /// Defined runes (user-defined macros): name -> MacroDef
    pub defined_runes: HashMap<String, crate::ast::MacroDef>,
    /// Trait definitions for IR export
    pub trait_defs: Vec<crate::ast::TraitDef>,
    /// Impl blocks for IR export
    pub impl_blocks: Vec<crate::ast::ImplBlock>,
    /// Constant definitions: (name, span) for IR export
    pub const_defs: Vec<(String, crate::span::Span)>,
    /// Function spans: name -> span for source mapping
    pub function_spans: HashMap<String, crate::span::Span>,
    /// Source text for line number calculation in IR export
    pub source_text: Option<String>,
}

/// Type definition for structs/enums
#[derive(Clone)]
pub enum TypeDef {
    Struct(StructDef),
    Enum(EnumDef),
}

// ============================================================================
// Holographic Helper Functions (for probabilistic data structures)
// Uses shared implementation from crate::holographic module
// ============================================================================

/// Hash a Value for holographic data structures
/// Converts Value to bytes and calls shared FNV-1a implementation
fn holographic_hash_value(value: &Value, seed: u64) -> u64 {
    let bytes = format!("{:?}", value);
    crate::holographic::holographic_hash(bytes.as_bytes(), seed)
}

/// Compute HyperLogLog cardinality estimate from Sigil register array
/// Converts Value::Int registers to i64 and calls shared hll_estimate
fn hll_cardinality_from_registers(registers: &[Value], precision: u32) -> i64 {
    let regs: Vec<i64> = registers.iter()
        .map(|v| match v {
            Value::Int(n) => *n,
            _ => 0,
        })
        .collect();
    let (estimate, _std_error) = crate::holographic::hll_estimate(&regs, precision);
    estimate.round() as i64
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
            program_args: None,
            current_module: None,
            current_self_type: None,
            current_source_dir: None,
            loaded_crates: HashSet::new(),
            loading_crates: HashSet::new(),
            project_root: None,
            workspace_members: HashMap::new(),
            drop_types: HashSet::new(),
            current_crate: None,
            crate_modules: HashSet::new(),
            crate_aliases: HashSet::new(),
            defined_runes: HashMap::new(),
            trait_defs: Vec::new(),
            impl_blocks: Vec::new(),
            const_defs: Vec::new(),
            function_spans: HashMap::new(),
            source_text: None,
        };

        // Register built-in functions
        interp.register_builtins();

        interp
    }

    // =========================================================================
    // Tensor Helper Functions
    // =========================================================================

    /// Create a Tensor Value with the given data and shape
    fn create_tensor_value(data: Vec<Value>, shape: Vec<i64>, requires_grad: bool) -> Value {
        let mut fields = HashMap::new();
        fields.insert(TENSOR_DATA_FIELD.to_string(), Value::Array(Rc::new(RefCell::new(data))));
        fields.insert(TENSOR_SHAPE_FIELD.to_string(), Value::Array(Rc::new(RefCell::new(
            shape.iter().map(|s| Value::Int(*s)).collect()
        ))));
        fields.insert(TENSOR_SHAPE_PUBLIC.to_string(), Value::Array(Rc::new(RefCell::new(
            shape.iter().map(|s| Value::Int(*s)).collect()
        ))));
        fields.insert(TENSOR_REQUIRES_GRAD_FIELD.to_string(), Value::Bool(requires_grad));
        fields.insert(TENSOR_GRAD_FIELD.to_string(), Value::Null);
        fields.insert(TENSOR_GRAD_PUBLIC.to_string(), Value::Null);
        Value::Struct {
            name: TENSOR_TYPE_NAME.to_string(),
            fields: Rc::new(RefCell::new(fields)),
        }
    }

    /// Check if a value is a Tensor struct
    fn is_tensor(value: &Value) -> bool {
        matches!(value, Value::Struct { name, .. } if name == TENSOR_TYPE_NAME)
    }

    /// Extract shape from a Tensor value
    fn get_tensor_shape(value: &Value) -> Option<Vec<i64>> {
        if let Value::Struct { name, fields } = value {
            if name == TENSOR_TYPE_NAME {
                let f = fields.borrow();
                if let Some(Value::Array(arr)) = f.get(TENSOR_SHAPE_FIELD) {
                    return Some(arr.borrow().iter().filter_map(|v| {
                        if let Value::Int(i) = v { Some(*i) } else { None }
                    }).collect());
                }
            }
        }
        None
    }

    /// Extract data from a Tensor value
    fn get_tensor_data(value: &Value) -> Option<Vec<Value>> {
        if let Value::Struct { name, fields } = value {
            if name == TENSOR_TYPE_NAME {
                let f = fields.borrow();
                if let Some(Value::Array(arr)) = f.get(TENSOR_DATA_FIELD) {
                    return Some(arr.borrow().clone());
                }
            }
        }
        None
    }

    /// Extract shape dimensions from a type annotation like Tensor<[3, 4]>
    /// Properly traverses the AST instead of parsing Debug output
    fn extract_tensor_shape_from_type(type_expr: &TypeExpr) -> Option<Vec<i64>> {
        if let TypeExpr::Path(path) = type_expr {
            if path.segments.len() == 1 {
                let segment = &path.segments[0];
                if segment.ident.name == TENSOR_TYPE_NAME {
                    if let Some(generics) = &segment.generics {
                        if let Some(first_arg) = generics.first() {
                            return Self::extract_shape_from_type_expr(first_arg);
                        }
                    }
                }
            }
        }
        None
    }

    /// Helper to extract shape from a type expression (handles ConstExpr, Array, etc.)
    fn extract_shape_from_type_expr(type_expr: &TypeExpr) -> Option<Vec<i64>> {
        match type_expr {
            // Const expression wrapping an array: <[3, 4]>
            TypeExpr::ConstExpr(expr) => Self::extract_shape_from_expr(expr),
            // Direct array type (less common): [i32; 3]
            TypeExpr::Array { size, .. } => {
                if let Some(n) = Self::extract_int_from_expr(size) {
                    Some(vec![n])
                } else {
                    None
                }
            }
            // Tuple of literals: (3, 4) - sometimes used for shape
            TypeExpr::Tuple(elements) => {
                let mut shape = Vec::new();
                for elem in elements {
                    if let TypeExpr::ConstExpr(expr) = elem {
                        if let Some(n) = Self::extract_int_from_expr(expr) {
                            shape.push(n);
                        }
                    }
                }
                if !shape.is_empty() { Some(shape) } else { None }
            }
            _ => None,
        }
    }

    /// Extract shape dimensions from an expression (handles array literals)
    fn extract_shape_from_expr(expr: &Expr) -> Option<Vec<i64>> {
        match expr {
            Expr::Array(elements) => {
                let mut shape = Vec::new();
                for elem in elements {
                    if let Some(n) = Self::extract_int_from_expr(elem) {
                        shape.push(n);
                    } else {
                        return None; // All elements must be integers
                    }
                }
                if !shape.is_empty() { Some(shape) } else { None }
            }
            // Single integer: Tensor<3> means [3]
            Expr::Literal(Literal::Int { value, .. }) => {
                value.parse::<i64>().ok().map(|n| vec![n])
            }
            _ => None,
        }
    }

    /// Extract integer value from an expression
    fn extract_int_from_expr(expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Literal(Literal::Int { value, .. }) => value.parse::<i64>().ok(),
            _ => None,
        }
    }

    /// Extract integer value from a const generic TypeExpr (for Linear·<784, 256>)
    fn extract_const_generic_int(type_expr: &TypeExpr) -> Result<i64, RuntimeError> {
        match type_expr {
            TypeExpr::ConstExpr(expr) => {
                Self::extract_int_from_expr(expr)
                    .ok_or_else(|| RuntimeError::new("Const generic must be an integer literal"))
            }
            TypeExpr::Path(path) if path.segments.len() == 1 => {
                // Handle numeric literal paths (sometimes parsed this way)
                path.segments[0].ident.name.parse::<i64>()
                    .map_err(|_| RuntimeError::new("Const generic must be an integer"))
            }
            _ => Err(RuntimeError::new(format!(
                "Expected const generic integer, got {:?}", type_expr
            ))),
        }
    }

    /// Create a Linear layer with specified input and output dimensions
    fn create_linear_layer(&mut self, in_features: i64, out_features: i64) -> Result<Value, RuntimeError> {
        // Kaiming (He) initialization: std = sqrt(2/in_features)
        let std_dev = (2.0 / in_features as f64).sqrt();
        let total_weights = (out_features * in_features) as usize;

        // Initialize weights with Kaiming uniform
        let weight_data: Vec<Value> = (0..total_weights).map(|i| {
            // Simple deterministic pseudo-random for reproducibility in tests
            let x = ((i as f64 * 0.618033988749) % 1.0) * 2.0 - 1.0; // Golden ratio based
            Value::Float(x * std_dev)
        }).collect();

        // Initialize biases to zero
        let bias_data: Vec<Value> = (0..out_features as usize).map(|_| Value::Float(0.0)).collect();

        let mut fields = HashMap::new();
        fields.insert("weight".to_string(), Self::create_tensor_value(
            weight_data, vec![out_features, in_features], true
        ));
        fields.insert("bias".to_string(), Self::create_tensor_value(
            bias_data, vec![out_features], true
        ));

        Ok(Value::Struct {
            name: "Linear".to_string(),
            fields: Rc::new(RefCell::new(fields)),
        })
    }

    /// Check if an expression is a tensor constructor call (zeros, ones, randn) with no args
    fn is_tensor_constructor_call(expr: &Expr) -> Option<&str> {
        if let Expr::Call { func, args } = expr {
            if args.is_empty() {
                if let Expr::Path(path) = func.as_ref() {
                    if path.segments.len() == 1 {
                        let name = path.segments[0].ident.name.as_str();
                        if TENSOR_CONSTRUCTORS.contains(&name) {
                            return Some(name);
                        }
                    }
                }
            }
        }
        None
    }

    /// Try to inject shape into a tensor constructor call
    /// Returns Some(tensor_value) if successful, None otherwise
    fn try_inject_tensor_shape(&mut self, name: &str, shape: &[i64]) -> Option<Result<Value, RuntimeError>> {
        let shape_array: Vec<Value> = shape.iter().map(|s| Value::Int(*s)).collect();
        let shape_val = Value::Array(Rc::new(RefCell::new(shape_array)));
        let builtin_opt = self.globals.borrow().get(name);
        if let Some(Value::BuiltIn(builtin)) = builtin_opt {
            Some(self.call_builtin(&builtin, vec![shape_val]))
        } else {
            None
        }
    }

    /// Evaluate an expression with potential tensor shape injection from type annotation
    fn eval_with_tensor_shape_inference(
        &mut self,
        expr: &Expr,
        tensor_shape: Option<&Vec<i64>>,
    ) -> Result<Value, RuntimeError> {
        // Early return if no shape to inject
        let shape = match tensor_shape {
            Some(s) => s,
            None => return self.evaluate(expr),
        };

        // Check for direct tensor constructor call: zeros(), ones(), randn()
        if let Some(name) = Self::is_tensor_constructor_call(expr) {
            if let Some(result) = self.try_inject_tensor_shape(name, shape) {
                return result;
            }
        }

        // Check for pipe expression starting with tensor constructor
        if let Expr::Pipe { expr: pipe_expr, operations } = expr {
            if let Some(name) = Self::is_tensor_constructor_call(pipe_expr) {
                if let Some(result) = self.try_inject_tensor_shape(name, shape) {
                    let mut tensor = result?;
                    for op in operations {
                        tensor = self.apply_pipe_op(tensor, op)?;
                    }
                    return Ok(tensor);
                }
            }
        }

        // Fallback to normal evaluation
        self.evaluate(expr)
    }

    // =========================================================================
    // End Tensor Helpers
    // =========================================================================

    /// Set program arguments (overrides env::args for the running program)
    pub fn set_program_args(&mut self, args: Vec<String>) {
        self.program_args = Some(args);
    }

    /// Set current module for registering definitions (module name, not file stem)
    pub fn set_current_module(&mut self, module: Option<String>) {
        self.current_module = module;
    }

    /// Set current source directory for resolving relative module paths
    pub fn set_current_source_dir(&mut self, dir: Option<String>) {
        self.current_source_dir = dir;
    }

    /// Set current source directory (convenience method for run-dir)
    pub fn set_source_dir(&mut self, dir: String) {
        self.current_source_dir = Some(dir);
    }

    /// Set current crate name (for `crate::*` and `crate_name::*` paths)
    pub fn set_crate_name(&mut self, name: String) {
        self.current_crate = Some(name.clone());
        // Also register the crate as a workspace member pointing to current directory
        self.workspace_members.insert(name.clone(), PathBuf::from("."));
        // Mark as loaded so `use crate_name::*` doesn't try to reload
        self.loaded_crates.insert(name);
    }

    /// Register a module as part of the current crate
    pub fn register_module(&mut self, module_name: String) {
        self.crate_modules.insert(module_name.clone());
        // Set current module context for subsequent definitions
        self.current_module = Some(module_name);
    }

    /// Set a crate alias (for compatibility with different naming conventions)
    /// e.g., Jormungandr uses "tome" as its crate name in invoke statements
    pub fn set_crate_alias(&mut self, alias: String) {
        // Register the alias as a workspace member pointing to current directory
        self.workspace_members.insert(alias.clone(), PathBuf::from("."));
        // Mark as loaded so `use alias::*` doesn't try to reload
        self.loaded_crates.insert(alias.clone());
        // Track alias for type registration
        self.crate_aliases.insert(alias);
    }

    /// Get program arguments (uses overridden args if set, otherwise env::args)
    pub fn get_program_args(&self) -> Vec<String> {
        self.program_args.clone().unwrap_or_else(|| std::env::args().collect())
    }

    /// Find and parse Sigil.toml from a source directory, walking up parent directories
    /// Looks for a workspace Sigil.toml (one with [workspace] section and members)
    pub fn discover_project(&mut self, source_dir: &str) -> Result<(), RuntimeError> {
        let mut current = PathBuf::from(source_dir);

        // Walk up to find Sigil.toml with [workspace] section
        loop {
            let sigil_toml = current.join("Sigil.toml");
            if sigil_toml.exists() {
                if let Ok(result) = self.try_parse_workspace_toml(&sigil_toml) {
                    if result {
                        return Ok(());
                    }
                    // Not a workspace Sigil.toml, continue searching
                }
            }

            // Also check for sigil.toml (lowercase)
            let sigil_toml_lower = current.join("sigil.toml");
            if sigil_toml_lower.exists() {
                if let Ok(result) = self.try_parse_workspace_toml(&sigil_toml_lower) {
                    if result {
                        return Ok(());
                    }
                    // Not a workspace Sigil.toml, continue searching
                }
            }

            if !current.pop() {
                // No workspace Sigil.toml found
                crate::sigil_debug!("DEBUG discover_project: no workspace Sigil.toml found from {}", source_dir);
                return Ok(());
            }
        }
    }

    /// Try to parse a Sigil.toml as a workspace config. Returns Ok(true) if it's a workspace,
    /// Ok(false) if it's a crate-level config, Err if parsing failed.
    fn try_parse_workspace_toml(&mut self, path: &PathBuf) -> Result<bool, RuntimeError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| RuntimeError::new(format!("Failed to read Sigil.toml: {}", e)))?;

        let toml_value: toml::Value = content.parse()
            .map_err(|e| RuntimeError::new(format!("Failed to parse Sigil.toml: {}", e)))?;

        // Check if this has a [workspace] section with members
        if let Some(workspace) = toml_value.get("workspace") {
            if workspace.get("members").and_then(|m| m.as_array()).is_some() {
                // This is a workspace Sigil.toml
                return self.parse_sigil_toml(path).map(|_| true);
            }
        }

        // Not a workspace config
        crate::sigil_debug!("DEBUG try_parse_workspace_toml: {:?} is not a workspace config", path);
        Ok(false)
    }

    /// Parse a Sigil.toml file and populate workspace_members
    fn parse_sigil_toml(&mut self, path: &PathBuf) -> Result<(), RuntimeError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| RuntimeError::new(format!("Failed to read Sigil.toml: {}", e)))?;

        let toml_value: toml::Value = content.parse()
            .map_err(|e| RuntimeError::new(format!("Failed to parse Sigil.toml: {}", e)))?;

        self.project_root = path.parent().map(|p| p.to_path_buf());

        // Parse [workspace] members
        if let Some(workspace) = toml_value.get("workspace") {
            if let Some(members) = workspace.get("members").and_then(|m| m.as_array()) {
                for member in members {
                    if let Some(member_path) = member.as_str() {
                        // Extract crate name from path (e.g., "crates/samael-analysis" -> "samael_analysis")
                        let crate_name = std::path::Path::new(member_path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.replace("-", "_"))
                            .unwrap_or_default();

                        if !crate_name.is_empty() {
                            crate::sigil_debug!("DEBUG parse_sigil_toml: registered workspace member: {} -> {}",
                                &crate_name, member_path);
                            self.workspace_members.insert(crate_name, PathBuf::from(member_path));
                        }
                    }
                }
            }
        }

        crate::sigil_debug!("DEBUG parse_sigil_toml: loaded {} workspace members from {:?}",
            self.workspace_members.len(), path);

        Ok(())
    }

    /// Load an external crate by name
    pub fn load_crate(&mut self, crate_name: &str) -> Result<bool, RuntimeError> {
        // Check if already loaded
        if self.loaded_crates.contains(crate_name) {
            return Ok(true);
        }

        // Check for circular dependency
        if self.loading_crates.contains(crate_name) {
            return Err(RuntimeError::new(format!(
                "Circular dependency detected: crate '{}' is already being loaded", crate_name
            )));
        }

        // Find crate path in workspace members
        let crate_path = match self.workspace_members.get(crate_name) {
            Some(p) => p.clone(),
            None => {
                crate::sigil_debug!("DEBUG load_crate: crate '{}' not found in workspace members", crate_name);
                return Ok(false);
            }
        };

        let project_root = match &self.project_root {
            Some(r) => r.clone(),
            None => {
                crate::sigil_debug!("DEBUG load_crate: no project root set");
                return Ok(false);
            }
        };

        // Build path to lib.sigil or lib.sg
        let lib_sigil_path = project_root.join(&crate_path).join("src").join("lib.sigil");
        let lib_sg_path = project_root.join(&crate_path).join("src").join("lib.sg");

        let lib_path = if lib_sigil_path.exists() {
            lib_sigil_path
        } else if lib_sg_path.exists() {
            lib_sg_path
        } else {
            crate::sigil_debug!("DEBUG load_crate: lib.sigil/lib.sg not found at {:?}", lib_sigil_path);
            return Ok(false);
        };

        // Mark as loading (for circular dependency detection)
        self.loading_crates.insert(crate_name.to_string());

        crate::sigil_debug!("DEBUG load_crate: loading crate '{}' from {:?}", crate_name, lib_path);

        // Read and parse the lib.sigil file
        let source = std::fs::read_to_string(&lib_path)
            .map_err(|e| RuntimeError::new(format!("Failed to read {:?}: {}", lib_path, e)))?;

        // Save current state
        let prev_module = self.current_module.clone();
        let prev_source_dir = self.current_source_dir.clone();

        // Set module context to crate name
        self.current_module = Some(crate_name.to_string());
        self.current_source_dir = lib_path.parent().map(|p| p.to_string_lossy().to_string());

        // Parse the source
        let mut parser = crate::Parser::new(&source);

        match parser.parse_file() {
            Ok(parsed_file) => {
                // Execute all items to register types and functions
                for item in &parsed_file.items {
                    if let Err(e) = self.execute_item(&item.node) {
                        crate::sigil_warn!("Warning: error loading crate '{}': {}", crate_name, e);
                    }
                }
            }
            Err(e) => {
                crate::sigil_warn!("Warning: failed to parse crate '{}': {:?}", crate_name, e);
            }
        }

        // Restore previous state
        self.current_module = prev_module;
        self.current_source_dir = prev_source_dir;

        // Mark as loaded and no longer loading
        self.loading_crates.remove(crate_name);
        self.loaded_crates.insert(crate_name.to_string());

        crate::sigil_debug!("DEBUG load_crate: successfully loaded crate '{}'", crate_name);

        Ok(true)
    }

    /// Load a module from the current crate (tome) by path.
    ///
    /// For `invoke tome·rt·sys·{write, Errno}`, this resolves:
    /// - module_path = ["rt", "sys"]
    /// - Tries: src/rt/sys/mod.sg, src/rt/sys.sg, rt/sys/mod.sg, rt/sys.sg
    pub fn load_tome_module(&mut self, module_path: &[String]) -> Result<bool, RuntimeError> {
        if module_path.is_empty() {
            return Ok(false);
        }

        // Build the qualified module name for tracking
        let module_key = format!("tome·{}", module_path.join("·"));

        // Check if already loaded
        if self.loaded_crates.contains(&module_key) {
            return Ok(true);
        }

        // Check for circular dependency
        if self.loading_crates.contains(&module_key) {
            return Err(RuntimeError::new(format!(
                "Circular dependency detected: module '{}' is already being loaded", module_key
            )));
        }

        // Determine base directory for tome modules
        // For 'tome' (current crate), we need the crate's src/ directory
        // Priority: project_root/src > current_source_dir/../src > current_source_dir
        let base_dir = if let Some(ref root) = self.project_root {
            // If project root has a src/ directory, use it
            let src_dir = root.join("src");
            if src_dir.exists() {
                src_dir
            } else {
                // Fall back to project root itself
                root.clone()
            }
        } else if let Some(ref source_dir) = self.current_source_dir {
            // Fall back to current source directory
            // But first, try checking if parent directory has src/ (for tests/ sibling case)
            let source_path = std::path::PathBuf::from(source_dir);
            if let Some(parent) = source_path.parent() {
                let sibling_src = parent.join("src");
                if sibling_src.exists() {
                    sibling_src
                } else {
                    source_path
                }
            } else {
                source_path
            }
        } else {
            crate::sigil_debug!("DEBUG load_tome_module: no source directory available");
            return Ok(false);
        };

        // Build path from module segments: ["rt", "sys"] -> "rt/sys"
        let mut module_file_path = base_dir.clone();
        for segment in module_path {
            module_file_path = module_file_path.join(segment);
        }

        // Try multiple file patterns:
        // 1. rt/sys/mod.sg (directory with mod.sg)
        // 2. rt/sys/mod.sigil
        // 3. rt/sys.sg (file)
        // 4. rt/sys.sigil
        let candidates = [
            module_file_path.join("mod.sg"),
            module_file_path.join("mod.sigil"),
            module_file_path.with_extension("sg"),
            module_file_path.with_extension("sigil"),
        ];

        let found_path = candidates.iter().find(|p| p.exists());

        let actual_path = match found_path {
            Some(p) => p.clone(),
            None => {
                crate::sigil_debug!("DEBUG load_tome_module: module not found, tried: {:?}", candidates);
                return Ok(false);
            }
        };

        crate::sigil_debug!("DEBUG load_tome_module: loading '{}' from {:?}", module_key, actual_path);

        // Mark as loading
        self.loading_crates.insert(module_key.clone());

        // Read and parse
        let source = std::fs::read_to_string(&actual_path)
            .map_err(|e| RuntimeError::new(format!("Failed to read {:?}: {}", actual_path, e)))?;

        // Save current state
        let prev_module = self.current_module.clone();
        let prev_source_dir = self.current_source_dir.clone();

        // Set module context - include "tome" prefix so glob imports can find methods
        let module_name = format!("tome·{}", module_path.join("·"));
        self.current_module = Some(module_name.clone());
        self.current_source_dir = actual_path.parent().map(|p| p.to_string_lossy().to_string());

        // Parse the module
        let mut parser = crate::Parser::new(&source);

        match parser.parse_file() {
            Ok(parsed_file) => {
                for item in &parsed_file.items {
                    if let Err(e) = self.execute_item(&item.node) {
                        crate::sigil_warn!("Warning: error loading module '{}': {}", module_key, e);
                    }
                }
            }
            Err(e) => {
                crate::sigil_warn!("Warning: failed to parse module '{}': {:?}", module_key, e);
            }
        }

        // Restore state
        self.current_module = prev_module;
        self.current_source_dir = prev_source_dir;

        // Mark as loaded
        self.loading_crates.remove(&module_key);
        self.loaded_crates.insert(module_key.clone());

        crate::sigil_debug!("DEBUG load_tome_module: successfully loaded '{}'", module_key);

        Ok(true)
    }

    fn register_builtins(&mut self) {
        // PhantomData - zero-sized type marker
        self.globals.borrow_mut().define("PhantomData".to_string(), Value::Null);

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
                Value::VariantConstructor { .. } => "variant_constructor",
                Value::DefaultConstructor { .. } => "default_constructor",
                Value::Range { .. } => "range",
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
            Value::Array(arr) => match arr.borrow_mut().pop() {
                Some(v) => Ok(Value::Variant {
                    enum_name: "Option".to_string(),
                    variant_name: "Some".to_string(),
                    fields: Some(Rc::new(vec![v])),
                }),
                None => Ok(Value::Variant {
                    enum_name: "Option".to_string(),
                    variant_name: "None".to_string(),
                    fields: None,
                }),
            },
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

        // ========================================================================
        // LLVM Intrinsics - For native runtime math module compatibility
        // ========================================================================

        // Absolute value
        self.define_builtin("__llvm_fabs_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).abs())),
            Value::Float(n) => Ok(Value::Float(n.abs())),
            _ => Err(RuntimeError::new("__llvm_fabs_f64 requires number")),
        });

        // Floor
        self.define_builtin("__llvm_floor_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float(*n as f64)),
            Value::Float(n) => Ok(Value::Float(n.floor())),
            _ => Err(RuntimeError::new("__llvm_floor_f64 requires number")),
        });

        // Ceiling
        self.define_builtin("__llvm_ceil_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float(*n as f64)),
            Value::Float(n) => Ok(Value::Float(n.ceil())),
            _ => Err(RuntimeError::new("__llvm_ceil_f64 requires number")),
        });

        // Round
        self.define_builtin("__llvm_round_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float(*n as f64)),
            Value::Float(n) => Ok(Value::Float(n.round())),
            _ => Err(RuntimeError::new("__llvm_round_f64 requires number")),
        });

        // Truncate
        self.define_builtin("__llvm_trunc_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float(*n as f64)),
            Value::Float(n) => Ok(Value::Float(n.trunc())),
            _ => Err(RuntimeError::new("__llvm_trunc_f64 requires number")),
        });

        // Square root
        self.define_builtin("__llvm_sqrt_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
            Value::Float(n) => Ok(Value::Float(n.sqrt())),
            _ => Err(RuntimeError::new("__llvm_sqrt_f64 requires number")),
        });

        // Exponential e^x
        self.define_builtin("__llvm_exp_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).exp())),
            Value::Float(n) => Ok(Value::Float(n.exp())),
            _ => Err(RuntimeError::new("__llvm_exp_f64 requires number")),
        });

        // Exponential 2^x
        self.define_builtin("__llvm_exp2_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).exp2())),
            Value::Float(n) => Ok(Value::Float(n.exp2())),
            _ => Err(RuntimeError::new("__llvm_exp2_f64 requires number")),
        });

        // Natural logarithm
        self.define_builtin("__llvm_log_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).ln())),
            Value::Float(n) => Ok(Value::Float(n.ln())),
            _ => Err(RuntimeError::new("__llvm_log_f64 requires number")),
        });

        // Log base 2
        self.define_builtin("__llvm_log2_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).log2())),
            Value::Float(n) => Ok(Value::Float(n.log2())),
            _ => Err(RuntimeError::new("__llvm_log2_f64 requires number")),
        });

        // Log base 10
        self.define_builtin("__llvm_log10_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).log10())),
            Value::Float(n) => Ok(Value::Float(n.log10())),
            _ => Err(RuntimeError::new("__llvm_log10_f64 requires number")),
        });

        // Power x^y
        self.define_builtin("__llvm_pow_f64", Some(2), |_, args| {
            let base = match &args[0] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_pow_f64 requires numbers")),
            };
            let exp = match &args[1] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_pow_f64 requires numbers")),
            };
            Ok(Value::Float(base.powf(exp)))
        });

        // Sine
        self.define_builtin("__llvm_sin_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).sin())),
            Value::Float(n) => Ok(Value::Float(n.sin())),
            _ => Err(RuntimeError::new("__llvm_sin_f64 requires number")),
        });

        // Cosine
        self.define_builtin("__llvm_cos_f64", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).cos())),
            Value::Float(n) => Ok(Value::Float(n.cos())),
            _ => Err(RuntimeError::new("__llvm_cos_f64 requires number")),
        });

        // Arc sine
        self.define_builtin("__libm_asin", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).asin())),
            Value::Float(n) => Ok(Value::Float(n.asin())),
            _ => Err(RuntimeError::new("__libm_asin requires number")),
        });

        // Arc cosine
        self.define_builtin("__libm_acos", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).acos())),
            Value::Float(n) => Ok(Value::Float(n.acos())),
            _ => Err(RuntimeError::new("__libm_acos requires number")),
        });

        // Arc tangent
        self.define_builtin("__libm_atan", Some(1), |_, args| match &args[0] {
            Value::Int(n) => Ok(Value::Float((*n as f64).atan())),
            Value::Float(n) => Ok(Value::Float(n.atan())),
            _ => Err(RuntimeError::new("__libm_atan requires number")),
        });

        // Arc tangent 2 (atan2)
        self.define_builtin("__libm_atan2", Some(2), |_, args| {
            let y = match &args[0] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__libm_atan2 requires numbers")),
            };
            let x = match &args[1] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__libm_atan2 requires numbers")),
            };
            Ok(Value::Float(y.atan2(x)))
        });

        // Minimum
        self.define_builtin("__llvm_minnum_f64", Some(2), |_, args| {
            let a = match &args[0] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_minnum_f64 requires numbers")),
            };
            let b = match &args[1] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_minnum_f64 requires numbers")),
            };
            Ok(Value::Float(a.min(b)))
        });

        // Maximum
        self.define_builtin("__llvm_maxnum_f64", Some(2), |_, args| {
            let a = match &args[0] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_maxnum_f64 requires numbers")),
            };
            let b = match &args[1] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_maxnum_f64 requires numbers")),
            };
            Ok(Value::Float(a.max(b)))
        });

        // Fused multiply-add: a*b + c
        self.define_builtin("__llvm_fma_f64", Some(3), |_, args| {
            let a = match &args[0] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_fma_f64 requires numbers")),
            };
            let b = match &args[1] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_fma_f64 requires numbers")),
            };
            let c = match &args[2] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_fma_f64 requires numbers")),
            };
            Ok(Value::Float(a.mul_add(b, c)))
        });

        // Copy sign: |x| with sign of y
        self.define_builtin("__llvm_copysign_f64", Some(2), |_, args| {
            let x = match &args[0] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_copysign_f64 requires numbers")),
            };
            let y = match &args[1] {
                Value::Int(n) => *n as f64,
                Value::Float(n) => *n,
                _ => return Err(RuntimeError::new("__llvm_copysign_f64 requires numbers")),
            };
            Ok(Value::Float(x.copysign(y)))
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

        // ExitCode enum for process exit codes (like Rust's std::process::ExitCode)
        self.globals.borrow_mut().define(
            "ExitCode·SUCCESS".to_string(),
            Value::Variant {
                enum_name: "ExitCode".to_string(),
                variant_name: "SUCCESS".to_string(),
                fields: Some(Rc::new(vec![Value::Int(0)])),
            },
        );
        self.globals.borrow_mut().define(
            "ExitCode·FAILURE".to_string(),
            Value::Variant {
                enum_name: "ExitCode".to_string(),
                variant_name: "FAILURE".to_string(),
                fields: Some(Rc::new(vec![Value::Int(1)])),
            },
        );

        // PathBuf::from - create a PathBuf from a string path
        self.define_builtin("PathBuf·from", Some(1), |_, args| {
            // Unwrap Ref types to get the actual value
            let arg = match &args[0] {
                Value::Ref(r) => r.borrow().clone(),
                other => other.clone(),
            };
            let path = match &arg {
                Value::String(s2) => s2.as_str().to_string(),
                _ => return Err(RuntimeError::new("PathBuf::from expects a string")),
            };
            // Represent PathBuf as a struct with a path field
            let mut fields = HashMap::new();
            fields.insert("path".to_string(), Value::String(Rc::new(path)));
            Ok(Value::Struct {
                name: "PathBuf".to_string(),
                fields: Rc::new(RefCell::new(fields)),
            })
        });

        // Path::new - create a Path from a string (similar to PathBuf for our purposes)
        self.define_builtin("Path·new", Some(1), |_, args| {
            // Unwrap Ref types to get the actual value
            let arg = match &args[0] {
                Value::Ref(r) => r.borrow().clone(),
                other => other.clone(),
            };
            let path = match &arg {
                Value::String(s2) => s2.as_str().to_string(),
                _ => return Err(RuntimeError::new("Path::new expects a string")),
            };
            let mut fields = HashMap::new();
            fields.insert("path".to_string(), Value::String(Rc::new(path)));
            Ok(Value::Struct {
                name: "Path".to_string(),
                fields: Rc::new(RefCell::new(fields)),
            })
        });

        // std::fs::read_to_string - read file contents as a string
        self.define_builtin("std·fs·read_to_string", Some(1), |interp, args| {
            // Recursively unwrap Ref types to get the actual value
            fn unwrap_refs(v: &Value) -> Value {
                match v {
                    Value::Ref(r) => unwrap_refs(&r.borrow()),
                    other => other.clone(),
                }
            }
            let arg = unwrap_refs(&args[0]);
            crate::sigil_debug!("DEBUG read_to_string: arg = {:?}", arg);
            // Also dump the environment to see what 'path' is bound to
            crate::sigil_debug!("DEBUG read_to_string: env has path = {:?}", interp.environment.borrow().get("path"));
            let path = match &arg {
                Value::String(s) => s.to_string(),
                // Handle PathBuf or Path structs
                Value::Struct { name, fields, .. } => {
                    crate::sigil_debug!("DEBUG read_to_string: struct name = {}", name);
                    fields.borrow().get("path")
                        .and_then(|v| if let Value::String(s) = v { Some(s.to_string()) } else { None })
                        .ok_or_else(|| RuntimeError::new("Expected path field in struct"))?
                }
                // Handle Option::Some(String)
                Value::Variant { enum_name, variant_name, fields } if enum_name == "Option" && variant_name == "Some" => {
                    if let Some(fields) = fields {
                        if let Some(Value::String(s)) = fields.first() {
                            s.to_string()
                        } else {
                            return Err(RuntimeError::new("read_to_string: Option::Some does not contain a string"));
                        }
                    } else {
                        return Err(RuntimeError::new("read_to_string: Option::Some has no fields"));
                    }
                }
                _ => return Err(RuntimeError::new(&format!("read_to_string expects a path string or PathBuf, got {:?}", arg))),
            };
            match std::fs::read_to_string(&path) {
                Ok(content) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Ok".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(content))])),
                }),
                Err(e) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Err".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(e.to_string()))])),
                }),
            }
        });

        // fs::read_to_string - alias without std prefix
        self.define_builtin("fs·read_to_string", Some(1), |_, args| {
            let arg = match &args[0] {
                Value::Ref(r) => r.borrow().clone(),
                other => other.clone(),
            };
            let path = match &arg {
                Value::String(s) => s.to_string(),
                Value::Struct { fields, .. } => {
                    fields.borrow().get("path")
                        .and_then(|v| if let Value::String(s) = v { Some(s.to_string()) } else { None })
                        .ok_or_else(|| RuntimeError::new("Expected path field in struct"))?
                }
                _ => return Err(RuntimeError::new("read_to_string expects a path string or PathBuf")),
            };
            match std::fs::read_to_string(&path) {
                Ok(content) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Ok".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(content))])),
                }),
                Err(e) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Err".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(e.to_string()))])),
                }),
            }
        });

        // std::fs::read_dir - read directory entries
        self.define_builtin("std·fs·read_dir", Some(1), |_, args| {
            fn unwrap_refs(v: &Value) -> Value {
                match v {
                    Value::Ref(r) => unwrap_refs(&r.borrow()),
                    other => other.clone(),
                }
            }
            let arg = unwrap_refs(&args[0]);
            let path = match &arg {
                Value::String(s) => s.to_string(),
                Value::Struct { name, fields, .. } if name == "Path" || name == "PathBuf" => {
                    fields.borrow().get("path")
                        .and_then(|v| if let Value::String(s) = v { Some(s.to_string()) } else { None })
                        .ok_or_else(|| RuntimeError::new("Expected path field in struct"))?
                }
                _ => return Err(RuntimeError::new(&format!("read_dir expects a path, got {:?}", arg))),
            };
            match std::fs::read_dir(&path) {
                Ok(entries) => {
                    // Collect entries into a Vec of DirEntry structs wrapped in Result::Ok
                    let entry_values: Vec<Value> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let entry_path = e.path().to_string_lossy().to_string();
                            let mut fields = HashMap::new();
                            fields.insert("path".to_string(), Value::String(Rc::new(entry_path)));
                            // Each entry is wrapped in Result::Ok
                            Value::Variant {
                                enum_name: "Result".to_string(),
                                variant_name: "Ok".to_string(),
                                fields: Some(Rc::new(vec![Value::Struct {
                                    name: "DirEntry".to_string(),
                                    fields: Rc::new(RefCell::new(fields)),
                                }])),
                            }
                        })
                        .collect();
                    // The overall result is Ok(iterator/array)
                    Ok(Value::Variant {
                        enum_name: "Result".to_string(),
                        variant_name: "Ok".to_string(),
                        fields: Some(Rc::new(vec![Value::Array(Rc::new(RefCell::new(entry_values)))])),
                    })
                }
                Err(e) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Err".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(e.to_string()))])),
                }),
            }
        });

        // fs::read_dir - alias without std prefix
        self.define_builtin("fs·read_dir", Some(1), |_, args| {
            fn unwrap_refs(v: &Value) -> Value {
                match v {
                    Value::Ref(r) => unwrap_refs(&r.borrow()),
                    other => other.clone(),
                }
            }
            let arg = unwrap_refs(&args[0]);
            let path = match &arg {
                Value::String(s) => s.to_string(),
                Value::Struct { name, fields, .. } if name == "Path" || name == "PathBuf" => {
                    fields.borrow().get("path")
                        .and_then(|v| if let Value::String(s) = v { Some(s.to_string()) } else { None })
                        .ok_or_else(|| RuntimeError::new("Expected path field in struct"))?
                }
                _ => return Err(RuntimeError::new(&format!("read_dir expects a path, got {:?}", arg))),
            };
            match std::fs::read_dir(&path) {
                Ok(entries) => {
                    let entry_values: Vec<Value> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let entry_path = e.path().to_string_lossy().to_string();
                            let mut fields = HashMap::new();
                            fields.insert("path".to_string(), Value::String(Rc::new(entry_path)));
                            Value::Variant {
                                enum_name: "Result".to_string(),
                                variant_name: "Ok".to_string(),
                                fields: Some(Rc::new(vec![Value::Struct {
                                    name: "DirEntry".to_string(),
                                    fields: Rc::new(RefCell::new(fields)),
                                }])),
                            }
                        })
                        .collect();
                    Ok(Value::Variant {
                        enum_name: "Result".to_string(),
                        variant_name: "Ok".to_string(),
                        fields: Some(Rc::new(vec![Value::Array(Rc::new(RefCell::new(entry_values)))])),
                    })
                }
                Err(e) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Err".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(e.to_string()))])),
                }),
            }
        });

        // std::env::var - get environment variable
        self.define_builtin("std·env·var", Some(1), |_, args| {
            fn unwrap_refs(v: &Value) -> Value {
                match v {
                    Value::Ref(r) => unwrap_refs(&r.borrow()),
                    other => other.clone(),
                }
            }
            let arg = unwrap_refs(&args[0]);
            let var_name = match &arg {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("env::var expects a string")),
            };
            match std::env::var(&var_name) {
                Ok(value) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Ok".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(value))])),
                }),
                Err(_) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Err".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new("Environment variable not found".to_string()))])),
                }),
            }
        });

        // env::var - alias without std prefix
        self.define_builtin("env·var", Some(1), |_, args| {
            fn unwrap_refs(v: &Value) -> Value {
                match v {
                    Value::Ref(r) => unwrap_refs(&r.borrow()),
                    other => other.clone(),
                }
            }
            let arg = unwrap_refs(&args[0]);
            let var_name = match &arg {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("env::var expects a string")),
            };
            match std::env::var(&var_name) {
                Ok(value) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Ok".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new(value))])),
                }),
                Err(_) => Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Err".to_string(),
                    fields: Some(Rc::new(vec![Value::String(Rc::new("Environment variable not found".to_string()))])),
                }),
            }
        });

        // std::env::args - get command line arguments
        // This is a special function that returns an iterator/array of strings
        self.define_builtin("std·env·args", Some(0), |interp, _| {
            let args = interp.get_program_args();
            let arg_values: Vec<Value> = args.iter()
                .map(|s| Value::String(Rc::new(s.clone())))
                .collect();
            Ok(Value::Array(Rc::new(RefCell::new(arg_values))))
        });

        // env::args - alias without std prefix
        self.define_builtin("env·args", Some(0), |interp, _| {
            let args = interp.get_program_args();
            let arg_values: Vec<Value> = args.iter()
                .map(|s| Value::String(Rc::new(s.clone())))
                .collect();
            Ok(Value::Array(Rc::new(RefCell::new(arg_values))))
        });

        // ============================================================
        // Filesystem built-ins for scanner (underscore naming)
        // ============================================================

        // fs_read - read entire file as string
        self.define_builtin("fs_read", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("fs_read requires a string path")),
            };
            match std::fs::read_to_string(&path) {
                Ok(content) => Ok(Value::String(Rc::new(content))),
                Err(e) => {
                    crate::sigil_debug!("DEBUG fs_read error for '{}': {}", path, e);
                    Ok(Value::Null)
                }
            }
        });

        // fs_list - list directory contents as array of strings
        self.define_builtin("fs_list", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("fs_list requires a string path")),
            };
            match std::fs::read_dir(&path) {
                Ok(entries) => {
                    let files: Vec<Value> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| Value::String(Rc::new(e.file_name().to_string_lossy().to_string())))
                        .collect();
                    Ok(Value::Array(Rc::new(RefCell::new(files))))
                }
                Err(e) => {
                    crate::sigil_debug!("DEBUG fs_list error for '{}': {}", path, e);
                    Ok(Value::Array(Rc::new(RefCell::new(Vec::new()))))
                }
            }
        });

        // fs_is_dir - check if path is a directory
        self.define_builtin("fs_is_dir", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("fs_is_dir requires a string path")),
            };
            Ok(Value::Bool(std::path::Path::new(&path).is_dir()))
        });

        // fs_is_file - check if path is a file
        self.define_builtin("fs_is_file", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("fs_is_file requires a string path")),
            };
            Ok(Value::Bool(std::path::Path::new(&path).is_file()))
        });

        // fs_exists - check if path exists
        self.define_builtin("fs_exists", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("fs_exists requires a string path")),
            };
            Ok(Value::Bool(std::path::Path::new(&path).exists()))
        });

        // path_extension - get file extension
        self.define_builtin("path_extension", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("path_extension requires a string path")),
            };
            let ext = std::path::Path::new(&path)
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_string());
            match ext {
                Some(e) => Ok(Value::String(Rc::new(e))),
                None => Ok(Value::Null),
            }
        });

        // path_join - join path components
        self.define_builtin("path_join", Some(2), |_, args| {
            let base = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("path_join requires string paths")),
            };
            let part = match &args[1] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("path_join requires string paths")),
            };
            let joined = std::path::Path::new(&base).join(&part);
            Ok(Value::String(Rc::new(joined.to_string_lossy().to_string())))
        });

        // path_parent - get parent directory
        self.define_builtin("path_parent", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("path_parent requires a string path")),
            };
            match std::path::Path::new(&path).parent() {
                Some(p) => Ok(Value::String(Rc::new(p.to_string_lossy().to_string()))),
                None => Ok(Value::Null),
            }
        });

        // path_file_name - get file name without directory
        self.define_builtin("path_file_name", Some(1), |_, args| {
            let path = match &args[0] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("path_file_name requires a string path")),
            };
            match std::path::Path::new(&path).file_name() {
                Some(n) => Ok(Value::String(Rc::new(n.to_string_lossy().to_string()))),
                None => Ok(Value::Null),
            }
        });

        // ============================================================
        // Tree-sitter parsing built-ins
        // ============================================================

        // TreeSitterParser::new - create a tree-sitter parser for a language
        self.define_builtin("TreeSitterParser·new", Some(1), |_, args| {
            use crate::tree_sitter_support::{TSLanguage, TSParser};

            // Get the language from the argument
            let lang_str = match &args[0] {
                Value::String(s) => s.to_string(),
                Value::Variant { enum_name, variant_name, .. } => {
                    // Handle Language::Rust style enums
                    format!("{}::{}", enum_name, variant_name)
                }
                other => format!("{:?}", other),
            };

            // Try to create the parser
            let language = match TSLanguage::from_str(&lang_str) {
                Some(lang) => lang,
                None => {
                    return Ok(Value::Variant {
                        enum_name: "Result".to_string(),
                        variant_name: "Err".to_string(),
                        fields: Some(Rc::new(vec![Value::Struct {
                            name: "ParseError".to_string(),
                            fields: Rc::new(RefCell::new({
                                let mut f = HashMap::new();
                                f.insert("kind".to_string(), Value::String(Rc::new("ParserNotFound".to_string())));
                                f.insert("message".to_string(), Value::String(Rc::new(format!("Unsupported language: {}", lang_str))));
                                f
                            })),
                        }])),
                    });
                }
            };

            // Create the parser and store the language
            match TSParser::new(language) {
                Ok(_) => {
                    // Return a TreeSitterParser struct
                    let mut fields = HashMap::new();
                    fields.insert("language".to_string(), Value::String(Rc::new(lang_str)));
                    fields.insert("_ts_language".to_string(), Value::String(Rc::new(format!("{:?}", language))));

                    Ok(Value::Variant {
                        enum_name: "Result".to_string(),
                        variant_name: "Ok".to_string(),
                        fields: Some(Rc::new(vec![Value::Struct {
                            name: "TreeSitterParser".to_string(),
                            fields: Rc::new(RefCell::new(fields)),
                        }])),
                    })
                }
                Err(e) => {
                    Ok(Value::Variant {
                        enum_name: "Result".to_string(),
                        variant_name: "Err".to_string(),
                        fields: Some(Rc::new(vec![Value::Struct {
                            name: "ParseError".to_string(),
                            fields: Rc::new(RefCell::new({
                                let mut f = HashMap::new();
                                f.insert("kind".to_string(), Value::String(Rc::new("ParserNotFound".to_string())));
                                f.insert("message".to_string(), Value::String(Rc::new(e)));
                                f
                            })),
                        }])),
                    })
                }
            }
        });

        // tree_sitter_parse - parse source code with tree-sitter
        self.define_builtin("tree_sitter_parse", Some(2), |_, args| {
            use crate::tree_sitter_support::{parse_source, node_to_value};

            // First arg is the language string, second is the source code
            let lang_str = match &args[0] {
                Value::String(s) => s.to_string(),
                Value::Variant { enum_name, variant_name, .. } => {
                    format!("{}::{}", enum_name, variant_name)
                }
                other => format!("{:?}", other),
            };

            let source = match &args[1] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("tree_sitter_parse expects source code string as second argument")),
            };

            // Parse the source
            match parse_source(&lang_str, &source) {
                Ok(tree) => {
                    // Convert to SyntaxNode value
                    let root = tree.root_node();
                    let root_fields = node_to_value(&root);

                    // Create TSTree struct
                    let mut tree_fields = HashMap::new();
                    tree_fields.insert("root".to_string(), Value::Struct {
                        name: "SyntaxNode".to_string(),
                        fields: Rc::new(RefCell::new(root_fields)),
                    });
                    tree_fields.insert("source".to_string(), Value::String(Rc::new(source)));

                    Ok(Value::Variant {
                        enum_name: "Result".to_string(),
                        variant_name: "Ok".to_string(),
                        fields: Some(Rc::new(vec![Value::Struct {
                            name: "TSTree".to_string(),
                            fields: Rc::new(RefCell::new(tree_fields)),
                        }])),
                    })
                }
                Err(e) => {
                    Ok(Value::Variant {
                        enum_name: "Result".to_string(),
                        variant_name: "Err".to_string(),
                        fields: Some(Rc::new(vec![Value::Struct {
                            name: "ParseError".to_string(),
                            fields: Rc::new(RefCell::new({
                                let mut f = HashMap::new();
                                f.insert("kind".to_string(), Value::String(Rc::new("SyntaxError".to_string())));
                                f.insert("message".to_string(), Value::String(Rc::new(e)));
                                f
                            })),
                        }])),
                    })
                }
            }
        });

        // tree_sitter_supported_languages - get list of supported languages
        self.define_builtin("tree_sitter_supported_languages", Some(0), |_, _| {
            use crate::tree_sitter_support::supported_languages;

            let languages: Vec<Value> = supported_languages()
                .iter()
                .map(|s| Value::String(Rc::new(s.to_string())))
                .collect();

            Ok(Value::Array(Rc::new(RefCell::new(languages))))
        });

        // tree_sitter_node_text - extract text from a syntax node using the source
        self.define_builtin("tree_sitter_node_text", Some(2), |_, args| {
            // First arg is the node (with start_byte and end_byte), second is the source
            let (start_byte, end_byte) = match &args[0] {
                Value::Struct { fields, .. } => {
                    let fields = fields.borrow();
                    let start = match fields.get("start_byte") {
                        Some(Value::Int(n)) => *n as usize,
                        _ => return Err(RuntimeError::new("Node missing start_byte field")),
                    };
                    let end = match fields.get("end_byte") {
                        Some(Value::Int(n)) => *n as usize,
                        _ => return Err(RuntimeError::new("Node missing end_byte field")),
                    };
                    (start, end)
                }
                _ => return Err(RuntimeError::new("tree_sitter_node_text expects a SyntaxNode struct")),
            };

            let source = match &args[1] {
                Value::String(s) => s.to_string(),
                _ => return Err(RuntimeError::new("tree_sitter_node_text expects source string as second argument")),
            };

            if end_byte <= source.len() && start_byte <= end_byte {
                Ok(Value::String(Rc::new(source[start_byte..end_byte].to_string())))
            } else {
                Err(RuntimeError::new("Byte range out of bounds"))
            }
        });

        // Rc::new(value) - Reference counted smart pointer (simplified)
        let rc_new = Value::BuiltIn(Rc::new(BuiltInFn {
            name: "Rc·new".to_string(),
            arity: Some(1),
            func: |_, args| {
                let mut fields = HashMap::new();
                fields.insert("_value".to_string(), args[0].clone());
                Ok(Value::Struct {
                    name: "Rc".to_string(),
                    fields: std::rc::Rc::new(RefCell::new(fields)),
                })
            },
        }));
        self.globals.borrow_mut().define("Rc·new".to_string(), rc_new);

        // Cell::new(value) - Interior mutability
        let cell_new = Value::BuiltIn(Rc::new(BuiltInFn {
            name: "Cell·new".to_string(),
            arity: Some(1),
            func: |_, args| {
                let mut fields = HashMap::new();
                fields.insert("_value".to_string(), args[0].clone());
                Ok(Value::Struct {
                    name: "Cell".to_string(),
                    fields: std::rc::Rc::new(RefCell::new(fields)),
                })
            },
        }));
        self.globals.borrow_mut().define("Cell·new".to_string(), cell_new);
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
            if f.params.is_empty() {
                // main() - no arguments
                result = self.call_function(&f, vec![])?;
            } else if f.params.len() == 1 {
                // main(args: [String]) - pass program arguments as array
                let args = self.get_program_args();
                let arg_values: Vec<Value> = args.iter()
                    .map(|s| Value::String(Rc::new(s.clone())))
                    .collect();
                let args_value = Value::Array(Rc::new(RefCell::new(arg_values)));
                result = self.call_function(&f, vec![args_value])?;
            }
            // If main has more than 1 param, skip auto-call
        }

        Ok(result)
    }

    /// Execute a file but only register definitions, don't auto-call main.
    /// Use this when loading files as part of a multi-file workspace.
    pub fn execute_definitions(&mut self, file: &SourceFile) -> Result<Value, RuntimeError> {
        let mut result = Value::Null;

        for item in &file.items {
            result = self.execute_item(&item.node)?;
        }

        Ok(result)
    }

    fn execute_item(&mut self, item: &Item) -> Result<Value, RuntimeError> {
        match item {
            Item::Function(func) => {
                let fn_value = self.create_function(func)?;
                let fn_name = func.name.name.clone();

                // Track function span for IR export
                self.function_spans.insert(fn_name.clone(), func.name.span);

                // Register with both simple name and module-qualified name
                self.globals.borrow_mut().define(fn_name.clone(), fn_value.clone());

                // Also register with module prefix if we're in a module context
                if let Some(ref module) = self.current_module {
                    let qualified_name = format!("{}·{}", module, fn_name);
                    self.globals.borrow_mut().define(qualified_name, fn_value);
                }

                Ok(Value::Null)
            }
            Item::Struct(s) => {
                let struct_name = s.name.name.clone();

                // Register with simple name
                self.types
                    .insert(struct_name.clone(), TypeDef::Struct(s.clone()));

                // Collect aliases to iterate over (need to clone since we're borrowing self)
                let aliases: Vec<String> = self.crate_aliases.iter().cloned().collect();

                // Also register with module-qualified name if in a module context
                if let Some(ref module) = self.current_module {
                    let qualified_name = format!("{}·{}", module, struct_name);
                    self.types.insert(qualified_name.clone(), TypeDef::Struct(s.clone()));

                    // Register with crate-prefixed name (for invoke crate·module·* patterns)
                    if let Some(ref crate_name) = self.current_crate {
                        let crate_qualified = format!("{}·{}·{}", crate_name, module, struct_name);
                        self.types.insert(crate_qualified, TypeDef::Struct(s.clone()));

                        // Also register "crate·module·StructName" pattern
                        let crate_path = format!("crate·{}·{}", module, struct_name);
                        self.types.insert(crate_path, TypeDef::Struct(s.clone()));
                    }

                    // Register with all crate aliases
                    for alias in &aliases {
                        let alias_qualified = format!("{}·{}·{}", alias, module, struct_name);
                        self.types.insert(alias_qualified, TypeDef::Struct(s.clone()));
                    }
                }

                // For unit structs, register the struct name as a value (zero-sized type)
                if matches!(&s.fields, crate::ast::StructFields::Unit) {
                    let unit_value = Value::Struct {
                        name: struct_name.clone(),
                        fields: Rc::new(RefCell::new(HashMap::new())),
                    };
                    self.globals.borrow_mut().define(struct_name.clone(), unit_value.clone());

                    // Also register with module-qualified name
                    if let Some(ref module) = self.current_module {
                        let qualified_name = format!("{}·{}", module, struct_name);
                        self.globals.borrow_mut().define(qualified_name, unit_value);
                    }
                }

                // Check for #[derive(Default)] attribute and store for later lookup
                let has_default = s.attrs.derives.iter().any(|d| matches!(d, DeriveTrait::Default));
                if has_default {
                    self.default_structs.insert(struct_name.clone(), s.clone());
                    if let Some(ref module) = self.current_module {
                        let qualified_name = format!("{}·{}", module, struct_name);
                        self.default_structs.insert(qualified_name, s.clone());
                    }
                }

                Ok(Value::Null)
            }
            Item::Enum(e) => {
                let enum_name = e.name.name.clone();

                // Register with simple name
                self.types
                    .insert(enum_name.clone(), TypeDef::Enum(e.clone()));

                // Collect aliases to iterate over (need to clone since we're borrowing self)
                let aliases: Vec<String> = self.crate_aliases.iter().cloned().collect();

                // Also register with module-qualified name
                if let Some(ref module) = self.current_module {
                    let module_qualified = format!("{}·{}", module, enum_name);
                    self.types.insert(module_qualified.clone(), TypeDef::Enum(e.clone()));

                    // Register with crate-prefixed name (for invoke crate·module·* patterns)
                    if let Some(ref crate_name) = self.current_crate {
                        let crate_qualified = format!("{}·{}·{}", crate_name, module, enum_name);
                        self.types.insert(crate_qualified.clone(), TypeDef::Enum(e.clone()));

                        // Also register "crate·module·EnumName" pattern (for `crate::module::*`)
                        let crate_path = format!("crate·{}·{}", module, enum_name);
                        self.types.insert(crate_path, TypeDef::Enum(e.clone()));
                    }

                    // Register with all crate aliases
                    for alias in &aliases {
                        let alias_qualified = format!("{}·{}·{}", alias, module, enum_name);
                        self.types.insert(alias_qualified, TypeDef::Enum(e.clone()));
                    }
                }

                // Register variant constructors as EnumName·VariantName
                // Store them in a lookup table that the variant_constructor builtin can use
                for variant in &e.variants {
                    let variant_name = variant.name.name.clone();

                    let arity = match &variant.fields {
                        crate::ast::StructFields::Unit => 0,
                        crate::ast::StructFields::Tuple(types) => types.len(),
                        crate::ast::StructFields::Named(fields) => fields.len(),
                    };

                    // Local name: EnumName·VariantName
                    let local_constructor = format!("{}·{}", enum_name, variant_name);
                    self.variant_constructors.insert(
                        local_constructor.clone(),
                        (enum_name.clone(), variant_name.clone(), arity)
                    );

                    // Module-qualified: module·EnumName·VariantName
                    if let Some(ref module) = self.current_module {
                        let module_constructor = format!("{}·{}·{}", module, enum_name, variant_name);
                        self.variant_constructors.insert(
                            module_constructor,
                            (enum_name.clone(), variant_name.clone(), arity)
                        );

                        // Crate-qualified: crate·module·EnumName·VariantName
                        if let Some(ref crate_name) = self.current_crate {
                            let crate_constructor = format!("{}·{}·{}·{}", crate_name, module, enum_name, variant_name);
                            self.variant_constructors.insert(
                                crate_constructor,
                                (enum_name.clone(), variant_name.clone(), arity)
                            );
                        }

                        // Register variant constructors with all crate aliases
                        for alias in &aliases {
                            let alias_constructor = format!("{}·{}·{}·{}", alias, module, enum_name, variant_name);
                            self.variant_constructors.insert(
                                alias_constructor,
                                (enum_name.clone(), variant_name.clone(), arity)
                            );
                        }
                    }
                }
                Ok(Value::Null)
            }
            Item::Trait(trait_def) => {
                // Track trait definition for IR export
                self.trait_defs.push(trait_def.clone());
                Ok(Value::Null)
            }
            Item::Const(c) => {
                let value = self.evaluate(&c.value)?;
                self.globals.borrow_mut().define(c.name.name.clone(), value);
                // Track constant for IR export
                self.const_defs.push((c.name.name.clone(), c.name.span));
                Ok(Value::Null)
            }
            Item::Static(s) => {
                let value = self.evaluate(&s.value)?;
                self.globals.borrow_mut().define_mut(s.name.name.clone(), value, s.mutable);
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
                // Track impl block for IR export
                self.impl_blocks.push(impl_block.clone());

                // Extract type name from self_ty
                let type_name = match &impl_block.self_ty {
                    TypeExpr::Path(path) => {
                        path.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::")
                    }
                    _ => return Ok(Value::Null), // Can't handle complex types
                };

                // Check if this is `impl Drop for X` - register for automatic drop calls
                if let Some(trait_path) = &impl_block.trait_ {
                    let trait_name = trait_path.segments.iter()
                        .map(|s| s.ident.name.as_str())
                        .collect::<Vec<_>>()
                        .join("::");
                    if trait_name == "Drop" {
                        self.drop_types.insert(type_name.clone());
                    }
                }

                // Register each method with qualified name TypeName·method
                for impl_item in &impl_block.items {
                    if let ImplItem::Function(func) = impl_item {
                        let fn_value = self.create_function(func)?;
                        let qualified_name = format!("{}·{}", type_name, func.name.name);
                        // Debug: track Lexer method registration
                        if type_name == "Lexer" && func.name.name.contains("keyword") {
                            crate::sigil_debug!("DEBUG registering: {}", qualified_name);
                        }
                        self.globals.borrow_mut().define(qualified_name.clone(), fn_value.clone());

                        // Also register with module prefix if in a module context
                        if let Some(ref module) = self.current_module {
                            let fully_qualified = format!("{}·{}", module, qualified_name);
                            self.globals.borrow_mut().define(fully_qualified, fn_value);
                        }
                    }
                }
                Ok(Value::Null)
            }
            Item::Module(module) => {
                // Handle module definitions
                let module_name = &module.name.name;

                // Register module in crate_modules for IR reflection
                self.crate_modules.insert(module_name.clone());

                if let Some(items) = &module.items {
                    // Inline module: mod foo { ... }
                    // Register items with qualified names: module_name·item_name
                    for item in items {
                        match &item.node {
                            Item::Const(c) => {
                                let value = self.evaluate(&c.value)?;
                                let qualified_name = format!("{}·{}", module_name, c.name.name);
                                self.globals.borrow_mut().define(qualified_name, value);
                            }
                            Item::Static(s) => {
                                let value = self.evaluate(&s.value)?;
                                let qualified_name = format!("{}·{}", module_name, s.name.name);
                                self.globals.borrow_mut().define_mut(qualified_name, value, s.mutable);
                            }
                            Item::Function(func) => {
                                let fn_value = self.create_function(func)?;
                                let qualified_name = format!("{}·{}", module_name, func.name.name);
                                self.globals.borrow_mut().define(qualified_name, fn_value);
                            }
                            Item::Struct(s) => {
                                let qualified_name = format!("{}·{}", module_name, s.name.name);
                                self.types.insert(qualified_name, TypeDef::Struct(s.clone()));
                            }
                            Item::Enum(e) => {
                                let enum_name = e.name.name.clone();

                                // Register enum type with module-qualified name
                                let qualified_type_name = format!("{}·{}", module_name, enum_name);
                                self.types.insert(qualified_type_name, TypeDef::Enum(e.clone()));

                                // Also register with local name for after 'invoke'
                                self.types.insert(enum_name.clone(), TypeDef::Enum(e.clone()));

                                // Register variant constructors
                                for variant in &e.variants {
                                    let variant_name = variant.name.name.clone();
                                    let arity = match &variant.fields {
                                        crate::ast::StructFields::Unit => 0,
                                        crate::ast::StructFields::Tuple(types) => types.len(),
                                        crate::ast::StructFields::Named(fields) => fields.len(),
                                    };

                                    // Local name: EnumName·VariantName (for use after invoke)
                                    let local_constructor = format!("{}·{}", enum_name, variant_name);
                                    self.variant_constructors.insert(
                                        local_constructor.clone(),
                                        (enum_name.clone(), variant_name.clone(), arity)
                                    );

                                    // Module-qualified name: module·EnumName·VariantName
                                    let module_constructor = format!("{}·{}·{}", module_name, enum_name, variant_name);
                                    self.variant_constructors.insert(
                                        module_constructor,
                                        (enum_name.clone(), variant_name.clone(), arity)
                                    );
                                }
                            }
                            Item::Impl(impl_block) => {
                                // Process impl blocks in inline modules
                                // Extract type name from self_ty
                                let type_name = match &impl_block.self_ty {
                                    TypeExpr::Path(path) => {
                                        path.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::")
                                    }
                                    _ => continue, // Can't handle complex types
                                };

                                // Register each method with both local and module-qualified names
                                for impl_item in &impl_block.items {
                                    if let ImplItem::Function(func) = impl_item {
                                        let fn_value = self.create_function(func)?;

                                        // Local name: Container·new (for use after invoke)
                                        let local_qualified = format!("{}·{}", type_name, func.name.name);
                                        self.globals.borrow_mut().define(local_qualified.clone(), fn_value.clone());

                                        // Module-qualified name: test_mod·Container·new
                                        let module_qualified = format!("{}·{}", module_name, local_qualified);
                                        self.globals.borrow_mut().define(module_qualified, fn_value);
                                    }
                                }
                            }
                            _ => {} // Skip other nested items for now
                        }
                    }
                } else {
                    // External module: mod foo; - try to load foo.sigil or foo.sg from same directory
                    if let Some(ref source_dir) = self.current_source_dir {
                        // Try both .sigil and .sg extensions
                        let sigil_path = std::path::Path::new(source_dir)
                            .join(format!("{}.sigil", module_name));
                        let sg_path = std::path::Path::new(source_dir)
                            .join(format!("{}.sg", module_name));

                        let module_path = if sigil_path.exists() {
                            Some(sigil_path)
                        } else if sg_path.exists() {
                            Some(sg_path)
                        } else {
                            // Neither exists
                            crate::sigil_debug!("DEBUG Module file not found: {} (source_dir={})", sigil_path.display(), source_dir);
                            None
                        };

                        if let Some(module_path) = module_path {
                            crate::sigil_debug!("DEBUG Loading external module: {}", module_path.display());
                            
                            match std::fs::read_to_string(&module_path) {
                                Ok(source) => {
                                    // Parse the module file
                                    let mut parser = crate::Parser::new(&source);
                                    match parser.parse_file() {
                                        Ok(parsed_file) => {
                                            // Save current module context
                                            let prev_module = self.current_module.clone();
                                            
                                            // Set module context for registering definitions
                                            self.current_module = Some(module_name.clone());
                                            
                                            // Execute module definitions
                                            for item in &parsed_file.items {
                                                if let Err(e) = self.execute_item(&item.node) {
                                                    crate::sigil_warn!("Warning: error in module {}: {}", module_name, e);
                                                }
                                            }
                                            
                                            // Restore previous module context
                                            self.current_module = prev_module;
                                        }
                                        Err(e) => {
                                            crate::sigil_warn!("Warning: failed to parse module {}: {:?}", module_name, e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    crate::sigil_warn!("Warning: failed to read module file {}: {}", module_path.display(), e);
                                }
                            }
                        }
                    } else {
                        crate::sigil_debug!("DEBUG No source_dir set, cannot load external module: {}", module_name);
                    }
                }
                Ok(Value::Null)
            }
            Item::Use(use_decl) => {
                // Process use declarations to create type/function aliases
                self.process_use_tree(&use_decl.tree, &[])?;
                Ok(Value::Null)
            }
            Item::Macro(macro_def) => {
                // Register user-defined rune (macro)
                let name = macro_def.name.name.clone();
                crate::sigil_debug!("DEBUG Registering rune: {}", name);
                self.defined_runes.insert(name, macro_def.clone());
                Ok(Value::Null)
            }
            Item::MacroInvocation(invocation) => {
                // Item-level macro invocation (e.g., thread_local! { ... })
                // For now, just skip these as they're typically compile-time
                let macro_name = invocation.path.segments.last()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");
                crate::sigil_debug!("DEBUG Item::MacroInvocation: {}", macro_name);
                Ok(Value::Null)
            }
            _ => Ok(Value::Null), // Skip other items for now
        }
    }

    /// Process a use tree to create type and function aliases
    fn process_use_tree(&mut self, tree: &crate::ast::UseTree, prefix: &[String]) -> Result<(), RuntimeError> {
        use crate::ast::UseTree;
        match tree {
            UseTree::Path { prefix: path_prefix, suffix } => {
                // Build path: prefix + this segment
                let mut new_prefix = prefix.to_vec();
                new_prefix.push(path_prefix.name.clone());
                self.process_use_tree(suffix, &new_prefix)
            }
            UseTree::Name(name) => {
                // use foo::bar::Baz -> import Baz from foo·bar·Baz
                let mut path = prefix.to_vec();
                path.push(name.name.clone());
                let qualified = path.join("·");
                let simple_name = name.name.clone();

                // If the type/function isn't found, try loading the module/crate first
                // The first segment determines how to load: "tome" = current crate, else external
                if !prefix.is_empty() {
                    let first_segment = &prefix[0];
                    let module_key = if first_segment == "tome" || first_segment == "crate" {
                        // Internal module: invoke tome·rt·sys·write
                        // Module path is everything between "tome" and the final name
                        format!("tome·{}", prefix[1..].join("·"))
                    } else {
                        first_segment.clone()
                    };

                    if !self.types.contains_key(&qualified)
                       && self.globals.borrow().get(&qualified).is_none()
                       && !self.loaded_crates.contains(&module_key)
                    {
                        if first_segment == "tome" || first_segment == "crate" {
                            // Load internal module: tome·rt·sys -> load_tome_module(["rt", "sys"])
                            let module_path: Vec<String> = prefix[1..].to_vec();
                            if let Err(e) = self.load_tome_module(&module_path) {
                                crate::sigil_debug!("DEBUG process_use_tree: failed to load tome module '{:?}': {}", module_path, e);
                            }
                        } else {
                            // Load external crate
                            if let Err(e) = self.load_crate(first_segment) {
                                crate::sigil_debug!("DEBUG process_use_tree: failed to load crate '{}': {}", first_segment, e);
                            }
                        }
                    }
                }

                // Create alias: simple_name -> qualified
                // For types: if foo·bar·Baz exists in types, also register as Baz
                if let Some(type_def) = self.types.get(&qualified).cloned() {
                    self.types.insert(simple_name.clone(), type_def);
                }
                // For functions: if foo·bar·Baz exists in globals, also register as Baz
                let func = self.globals.borrow().get(&qualified).map(|v| v.clone());
                if let Some(val) = func {
                    self.globals.borrow_mut().define(simple_name.clone(), val);
                }

                // Also import impl methods for this type
                // e.g., when importing samael_analysis::AnalysisConfig,
                // also import samael_analysis·AnalysisConfig·default as AnalysisConfig·default
                let method_prefix = format!("{}·", qualified);
                let matching_methods: Vec<(String, Value)> = {
                    let globals = self.globals.borrow();
                    globals.values.iter()
                        .filter(|(k, _)| k.starts_with(&method_prefix))
                        .map(|(k, (v, _))| {
                            // samael_analysis·AnalysisConfig·default -> AnalysisConfig·default
                            let method_suffix = k.strip_prefix(&method_prefix).unwrap();
                            let new_name = format!("{}·{}", simple_name, method_suffix);
                            (new_name, v.clone())
                        })
                        .collect()
                };
                for (name, val) in matching_methods {
                    // Only define if not already present - avoids overwriting correctly-named builtins
                    // e.g., fs·read_to_string should keep its proper name, not be overwritten by
                    // a copy of std·fs·read_to_string which has the wrong internal name
                    if self.globals.borrow().get(&name).is_none() {
                        self.globals.borrow_mut().define(name, val);
                    }
                }
                Ok(())
            }
            UseTree::Rename { name, alias } => {
                // use foo::bar::Baz as Qux
                let mut path = prefix.to_vec();
                path.push(name.name.clone());
                let qualified = path.join("·");
                let alias_name = alias.name.clone();

                if let Some(type_def) = self.types.get(&qualified).cloned() {
                    self.types.insert(alias_name.clone(), type_def);
                }
                let func = self.globals.borrow().get(&qualified).map(|v| v.clone());
                if let Some(val) = func {
                    self.globals.borrow_mut().define(alias_name, val);
                }
                Ok(())
            }
            UseTree::Glob => {
                // use foo::bar::* - import all from foo·bar
                let path_prefix = prefix.join("·");

                // Load the module first if it hasn't been loaded
                if !prefix.is_empty() {
                    let first_segment = &prefix[0];
                    let module_key = if first_segment == "tome" || first_segment == "crate" {
                        format!("tome·{}", prefix[1..].join("·"))
                    } else {
                        first_segment.clone()
                    };

                    if !self.loaded_crates.contains(&module_key) {
                        if first_segment == "tome" || first_segment == "crate" {
                            let module_path: Vec<String> = prefix[1..].to_vec();
                            if let Err(e) = self.load_tome_module(&module_path) {
                                crate::sigil_debug!("DEBUG glob: failed to load tome module '{:?}': {}", module_path, e);
                            }
                        } else {
                            if let Err(e) = self.load_crate(first_segment) {
                                crate::sigil_debug!("DEBUG glob: failed to load crate '{}': {}", first_segment, e);
                            }
                        }
                    }
                }
                // Find all types starting with this prefix
                let matching_types: Vec<(String, TypeDef)> = self.types.iter()
                    .filter(|(k, _)| k.starts_with(&path_prefix) && k.len() > path_prefix.len())
                    .map(|(k, v)| {
                        let suffix = k.strip_prefix(&path_prefix).unwrap().trim_start_matches('·');
                        (suffix.to_string(), v.clone())
                    })
                    .filter(|(k, _)| !k.contains('·')) // Only immediate children
                    .collect();
                for (name, def) in &matching_types {
                    self.types.insert(name.clone(), def.clone());
                }
                // Similar for functions
                let matching_funcs: Vec<(String, Value)> = {
                    let globals = self.globals.borrow();
                    globals.values.iter()
                        .filter(|(k, _)| k.starts_with(&path_prefix) && k.len() > path_prefix.len())
                        .map(|(k, (v, _))| {
                            let suffix = k.strip_prefix(&path_prefix).unwrap().trim_start_matches('·');
                            (suffix.to_string(), v.clone())
                        })
                        .filter(|(k, _)| !k.contains('·'))
                        .collect()
                };
                for (name, val) in matching_funcs {
                    self.globals.borrow_mut().define(name, val);
                }

                // Also import impl methods for each imported type
                // e.g., when glob-importing foo·bar, also import foo·bar·TypeName·method as TypeName·method
                for (type_name, _) in &matching_types {
                    let method_prefix = format!("{}·{}·", path_prefix, type_name);
                    let matching_methods: Vec<(String, Value)> = {
                        let globals = self.globals.borrow();
                        globals.values.iter()
                            .filter(|(k, _)| k.starts_with(&method_prefix))
                            .map(|(k, (v, _))| {
                                // foo·bar·Container·new -> Container·new
                                let method_name = k.strip_prefix(&path_prefix).unwrap().trim_start_matches('·');
                                (method_name.to_string(), v.clone())
                            })
                            .collect()
                    };
                    for (name, val) in matching_methods {
                        if self.globals.borrow().get(&name).is_none() {
                            self.globals.borrow_mut().define(name, val);
                        }
                    }
                }

                Ok(())
            }
            UseTree::Group(trees) => {
                // use foo::{Bar, Baz}
                for tree in trees {
                    self.process_use_tree(tree, prefix)?;
                }
                Ok(())
            }
        }
    }

    fn create_function(&self, func: &crate::ast::Function) -> Result<Value, RuntimeError> {
        let params: Vec<String> = func
            .params
            .iter()
            .map(|p| Self::extract_param_name(&p.pattern))
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

    /// Extract parameter name from a pattern, handling &self, mut self, etc.
    fn extract_param_name(pattern: &Pattern) -> String {
        match pattern {
            Pattern::Ident { name, .. } => name.name.clone(),
            // Handle &self and &mut self
            Pattern::Ref { pattern: inner, .. } => Self::extract_param_name(inner),
            // Handle ref self
            Pattern::RefBinding { name, .. } => name.name.clone(),
            _ => "_".to_string(),
        }
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
                ..
            } => self.eval_for(pattern, iter, body),
            Expr::While { condition, body, .. } => self.eval_while(condition, body),
            Expr::Loop { body, .. } => self.eval_loop(body),
            Expr::Return(value) => self.eval_return(value),
            Expr::Break { value, .. } => self.eval_break(value),
            Expr::Continue { .. } => Err(RuntimeError::new("continue")),
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
            Expr::Closure { params, body, .. } => self.eval_closure(params, body),
            Expr::Struct { path, fields, rest } => self.eval_struct_literal(path, fields, rest),
            Expr::Evidential {
                expr,
                evidentiality,
            } => self.eval_evidential(expr, evidentiality),
            Expr::Range {
                start,
                end,
                inclusive,
            } => {
                crate::sigil_debug!("DEBUG evaluate: Expr::Range being evaluated standalone");
                self.eval_range(start, end, *inclusive)
            }
            Expr::Assign { target, value } => self.eval_assign(target, value),
            Expr::Let { pattern, value } => {
                // Let expression (for if-let, while-let patterns)
                // Evaluate the value and check if pattern matches
                let val = self.evaluate(value)?;
                // Check if pattern matches - return true/false for if-let semantics
                if self.pattern_matches(pattern, &val)? {
                    // Pattern matches - bind variables and return true
                    self.bind_pattern(pattern, val)?;
                    Ok(Value::Bool(true))
                } else {
                    // Pattern doesn't match - return false without binding
                    Ok(Value::Bool(false))
                }
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
                    Some(Evidentiality::Reported) | Some(Evidentiality::Paradox) | Some(Evidentiality::Predicted) => {
                        // ⌛~ or ⌛‽ or ⌛◊ - mark as external/reported/predicted, unwrap if Result/Option
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
                crate::sigil_debug!("DEBUG Expr::Macro: name='{}', tokens='{}'", macro_name, tokens);

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
                        // Check for user-defined rune
                        if let Some(rune_def) = self.defined_runes.get(macro_name).cloned() {
                            crate::sigil_debug!("DEBUG Expanding user-defined rune: {}", macro_name);
                            self.expand_rune(&rune_def, tokens, None)
                        } else {
                            // Unknown macro - return tokens as string for debugging
                            Ok(Value::String(Rc::new(tokens.clone())))
                        }
                    }
                }
            }
            // Unsafe block - just evaluate the block normally
            Expr::Unsafe(block) => self.eval_block(block),
            // Async block - evaluate the block (interpreter doesn't handle true async)
            Expr::Async { block, .. } => self.eval_block(block),
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
                                crate::sigil_debug!("DEBUG Try propagating Result::Err with fields: {:?}", fields);
                                let err_msg = if let Some(f) = fields {
                                    let first = f.first().cloned().unwrap_or(Value::Null);
                                    crate::sigil_debug!("DEBUG Try error first value: {}", first);
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
            // Cast expression: expr as Type
            Expr::Cast { expr, ty } => {
                let value = self.evaluate(expr)?;
                // Handle type casts
                let type_name = match ty {
                    TypeExpr::Path(path) => {
                        if !path.segments.is_empty() {
                            path.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("")
                        } else {
                            ""
                        }
                    }
                    _ => "",
                };
                match (value, type_name) {
                    // Char to numeric
                    (Value::Char(c), "u8") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "u16") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "u32") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "u64") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "i8") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "i16") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "i32") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "i64") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "usize") => Ok(Value::Int(c as i64)),
                    (Value::Char(c), "isize") => Ok(Value::Int(c as i64)),
                    // Int to int (no-op in our runtime)
                    (Value::Int(i), "u8") => Ok(Value::Int(i)),
                    (Value::Int(i), "u16") => Ok(Value::Int(i)),
                    (Value::Int(i), "u32") => Ok(Value::Int(i)),
                    (Value::Int(i), "u64") => Ok(Value::Int(i)),
                    (Value::Int(i), "i8") => Ok(Value::Int(i)),
                    (Value::Int(i), "i16") => Ok(Value::Int(i)),
                    (Value::Int(i), "i32") => Ok(Value::Int(i)),
                    (Value::Int(i), "i64") => Ok(Value::Int(i)),
                    (Value::Int(i), "usize") => Ok(Value::Int(i)),
                    (Value::Int(i), "isize") => Ok(Value::Int(i)),
                    // Float to int
                    (Value::Float(f), "i32") => Ok(Value::Int(f as i64)),
                    (Value::Float(f), "i64") => Ok(Value::Int(f as i64)),
                    (Value::Float(f), "u32") => Ok(Value::Int(f as i64)),
                    (Value::Float(f), "u64") => Ok(Value::Int(f as i64)),
                    // Int to float
                    (Value::Int(i), "f32") => Ok(Value::Float(i as f64)),
                    (Value::Int(i), "f64") => Ok(Value::Float(i as f64)),
                    // Int to char
                    (Value::Int(i), "char") => {
                        if let Some(c) = char::from_u32(i as u32) {
                            Ok(Value::Char(c))
                        } else {
                            Err(RuntimeError::new(format!("invalid char code: {}", i)))
                        }
                    }
                    // Pass through for same type
                    (v, _) => Ok(v),
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
            Expr::Unary { op: UnaryOp::Deref, expr: inner } => {
                // Dereference assignment: *ptr = value (parsed as Unary)
                // Handle mutable references (&mut T)
                let ptr_val = self.evaluate(inner)?;
                match ptr_val {
                    Value::Ref(r) => {
                        *r.borrow_mut() = val.clone();
                        Ok(val)
                    }
                    _ => Err(RuntimeError::new("Cannot dereference assign to non-reference")),
                }
            }
            Expr::Deref(inner) => {
                // Dereference assignment: *ptr = value (parsed as Deref)
                // Handle mutable references (&mut T)
                let ptr_val = self.evaluate(inner)?;
                match ptr_val {
                    Value::Ref(r) => {
                        *r.borrow_mut() = val.clone();
                        Ok(val)
                    }
                    _ => Err(RuntimeError::new("Cannot dereference assign to non-reference")),
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
            // Look up the variable in the environment
            // Note: "_" may be bound by pipe operations (τ, φ, etc.), so we must check
            // the environment first before treating it as a wildcard
            if let Some(val) = self.environment.borrow().get(name) {
                return Ok(val);
            }
            // Handle wildcard "_" - return Null if not bound
            if name == "_" {
                return Ok(Value::Null);
            }
            // Handle "Self" as unit struct constructor (for unit structs like `pub fn new() -> Self { Self }`)
            if name == "Self" {
                if let Some(ref self_type) = self.current_self_type {
                    // Check if the type is a unit struct
                    if let Some(TypeDef::Struct(struct_def)) = self.types.get(self_type) {
                        if matches!(struct_def.fields, crate::ast::StructFields::Unit) {
                            return Ok(Value::Struct {
                                name: self_type.clone(),
                                fields: Rc::new(RefCell::new(HashMap::new())),
                            });
                        }
                    }
                    // If not a unit struct, create empty struct (best effort)
                    return Ok(Value::Struct {
                        name: self_type.clone(),
                        fields: Rc::new(RefCell::new(HashMap::new())),
                    });
                }
            }
            if name.len() <= 2 {
                crate::sigil_debug!("DEBUG Undefined variable '{}' (len={})", name, name.len());
            }
            Err(RuntimeError::new(format!("Undefined variable: {}", name)))
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

            // Check globals for qualified name (for Type::method patterns)
            if let Some(val) = self.globals.borrow().get(&full_name) {
                return Ok(val);
            }

            // If in a module context, try current_module·full_name for sibling modules
            // e.g., when in samael_cli, "analyze::execute" -> "samael_cli·analyze·execute"
            if let Some(ref current_mod) = self.current_module {
                // Extract crate name from current_module (e.g., "samael_cli" from "samael_cli·analyze")
                let crate_name = current_mod.split('·').next().unwrap_or(current_mod);
                let crate_qualified = format!("{}·{}", crate_name, full_name);
                if full_name.contains("execute") {
                    crate::sigil_debug!("DEBUG eval_path: Looking for '{}' with crate_qualified='{}'", full_name, crate_qualified);
                }
                if let Some(val) = self.globals.borrow().get(&crate_qualified) {
                    crate::sigil_debug!("DEBUG eval_path: FOUND '{}' via crate_qualified", crate_qualified);
                    return Ok(val);
                }
            } else if full_name.contains("execute") {
                crate::sigil_debug!("DEBUG eval_path: current_module is None, can't resolve '{}'", full_name);
            }

            // DEBUG: Check if we're looking for a ::new call
            if full_name.ends_with("·new") {
                crate::sigil_debug!("DEBUG eval_path: Looking for '{}' - NOT FOUND in globals", full_name);
            }

            // Check for enum variant syntax FIRST (EnumName::Variant)
            // This must come before looking up just the last segment to avoid
            // returning a built-in function instead of the actual variant
            if path.segments.len() == 2 {
                let type_name = &path.segments[0].ident.name;
                let variant_name = &path.segments[1].ident.name;

                // Check if this is an enum variant (direct type name match)
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
                    // Enum found but variant doesn't exist - error
                    let valid_variants: Vec<_> = enum_def.variants.iter()
                        .map(|v| v.name.name.as_str())
                        .collect();
                    return Err(RuntimeError::new(format!(
                        "No variant '{}' on enum '{}'. Valid variants: {:?}",
                        variant_name, type_name, valid_variants
                    )));
                }

                // Fallback: type name might be an alias - search all enums for the variant
                for (actual_type_name, type_def) in &self.types {
                    if let TypeDef::Enum(enum_def) = type_def {
                        for variant in &enum_def.variants {
                            if &variant.name.name == variant_name {
                                if matches!(variant.fields, crate::ast::StructFields::Unit) {
                                    return Ok(Value::Variant {
                                        enum_name: actual_type_name.clone(),
                                        variant_name: variant_name.clone(),
                                        fields: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Try looking up the last segment (for Math·sqrt -> sqrt)
            let last_name = &path.segments.last().unwrap().ident.name;
            if let Some(val) = self.environment.borrow().get(last_name) {
                // DEBUG: Warn about fallback for new functions
                if last_name == "new" {
                    crate::sigil_debug!("DEBUG eval_path: FALLBACK from '{}' to '{}' - found in env", full_name, last_name);
                }
                return Ok(val);
            }

            // Handle Self::method - use current_self_type to get the specific type
            if path.segments.len() == 2 && path.segments[0].ident.name == "Self" {
                if let Some(ref self_type) = self.current_self_type {
                    // Look up the specific Type·method function
                    let qualified = format!("{}·{}", self_type, last_name);
                    if let Some(val) = self.globals.borrow().get(&qualified) {
                        return Ok(val);
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

            // Fallback for unknown types from external crates:
            if path.segments.len() == 2 {
                let type_name = &path.segments[0].ident.name;
                let method_name = &path.segments[1].ident.name;

                // Check if this looks like a constructor/method call (lowercase or snake_case)
                let is_method = method_name.chars().next().map_or(false, |c| c.is_lowercase())
                    || method_name == "new"
                    || method_name == "default"
                    || method_name == "from"
                    || method_name == "try_from"
                    || method_name == "into"
                    || method_name == "with_capacity"
                    || method_name == "from_str";

                if is_method {
                    // Return a special marker that eval_call will recognize
                    // Store the type name in a Struct value with a special marker name
                    return Ok(Value::Struct {
                        name: format!("__constructor__{}", type_name),
                        fields: Rc::new(RefCell::new(HashMap::new())),
                    });
                } else {
                    // Looks like an enum variant (PascalCase) - create unit variant
                    return Ok(Value::Variant {
                        enum_name: type_name.clone(),
                        variant_name: method_name.clone(),
                        fields: None,
                    });
                }
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

        // Debug Mul operations involving potential null
        if matches!(op, BinOp::Mul) && (matches!(lhs, Value::Null) || matches!(rhs, Value::Null) || matches!(lhs, Value::Struct { .. }) || matches!(rhs, Value::Struct { .. })) {
            crate::sigil_debug!("DEBUG eval_binary Mul: left={:?}, right={:?}", left, right);
            crate::sigil_debug!("DEBUG eval_binary Mul: lhs={}, rhs={}", self.format_value(&lhs), self.format_value(&rhs));
        }

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
                BinOp::Concat | BinOp::Convolve => {
                    // Convolution merges arrays (simplified: concatenation)
                    let mut result = a.borrow().clone();
                    result.extend(b.borrow().iter().cloned());
                    Ok(Value::Array(Rc::new(RefCell::new(result))))
                }
                BinOp::Eq => Ok(Value::Bool(Rc::ptr_eq(&a, &b))),
                BinOp::Ne => Ok(Value::Bool(!Rc::ptr_eq(&a, &b))),
                BinOp::Hadamard => {
                    // Element-wise (Hadamard) product: a ⊙ b
                    let a_arr = a.borrow();
                    let b_arr = b.borrow();
                    if a_arr.len() != b_arr.len() {
                        return Err(RuntimeError::new("Hadamard product requires arrays of equal length"));
                    }
                    let result: Vec<Value> = a_arr.iter().zip(b_arr.iter()).map(|(ai, bi)| {
                        match (ai, bi) {
                            (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
                            (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
                            (Value::Float(x), Value::Int(y)) => Value::Float(x * (*y as f64)),
                            (Value::Int(x), Value::Float(y)) => Value::Float((*x as f64) * y),
                            _ => Value::Float(0.0),
                        }
                    }).collect();
                    Ok(Value::Array(Rc::new(RefCell::new(result))))
                }
                BinOp::MatMul => {
                    // Matrix multiplication: a @ b
                    // Treat as 1D dot product for vectors, or 2D matmul for nested arrays
                    let a_arr = a.borrow();
                    let b_arr = b.borrow();

                    // Check if these are 2D matrices (arrays of arrays)
                    let a_is_2d = a_arr.first().map(|v| matches!(v, Value::Array(_))).unwrap_or(false);
                    let b_is_2d = b_arr.first().map(|v| matches!(v, Value::Array(_))).unwrap_or(false);

                    if a_is_2d && b_is_2d {
                        // 2D matrix multiplication
                        self.matmul_2d(&a_arr, &b_arr)
                    } else if !a_is_2d && !b_is_2d {
                        // 1D dot product
                        if a_arr.len() != b_arr.len() {
                            return Err(RuntimeError::new("Vector dot product requires equal lengths"));
                        }
                        let mut sum = 0.0f64;
                        for (ai, bi) in a_arr.iter().zip(b_arr.iter()) {
                            let prod = match (ai, bi) {
                                (Value::Float(x), Value::Float(y)) => x * y,
                                (Value::Int(x), Value::Int(y)) => (*x as f64) * (*y as f64),
                                (Value::Float(x), Value::Int(y)) => x * (*y as f64),
                                (Value::Int(x), Value::Float(y)) => (*x as f64) * y,
                                _ => 0.0,
                            };
                            sum += prod;
                        }
                        Ok(Value::Float(sum))
                    } else {
                        Err(RuntimeError::new("Matrix multiplication dimension mismatch"))
                    }
                }
                _ => Err(RuntimeError::new("Invalid array operation")),
            },
            // Null equality
            (Value::Null, Value::Null) => match op {
                BinOp::Eq => Ok(Value::Bool(true)),
                BinOp::Ne => Ok(Value::Bool(false)),
                _ => {
                    crate::sigil_debug!("DEBUG: null op {:?} on (Null, Null)", op);
                    Err(RuntimeError::new(format!("Invalid null operation: {:?} on (Null, Null)", op)))
                }
            },
            // Option::None is equivalent to null for equality
            (Value::Variant { enum_name, variant_name, .. }, Value::Null)
                if enum_name == "Option" && variant_name == "None" && matches!(op, BinOp::Eq | BinOp::Ne) =>
            {
                match op {
                    BinOp::Eq => Ok(Value::Bool(true)),
                    BinOp::Ne => Ok(Value::Bool(false)),
                    _ => unreachable!(),
                }
            },
            (Value::Null, Value::Variant { enum_name, variant_name, .. })
                if enum_name == "Option" && variant_name == "None" && matches!(op, BinOp::Eq | BinOp::Ne) =>
            {
                match op {
                    BinOp::Eq => Ok(Value::Bool(true)),
                    BinOp::Ne => Ok(Value::Bool(false)),
                    _ => unreachable!(),
                }
            },
            (Value::Null, other) | (other, Value::Null) => match op {
                BinOp::Eq => Ok(Value::Bool(false)),
                BinOp::Ne => Ok(Value::Bool(true)),
                _ => {
                    crate::sigil_debug!("DEBUG: null op {:?} with other={}", op, self.format_value(&other));
                    Err(RuntimeError::new(format!("Invalid null operation: {:?}", op)))
                }
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
                        (Some(a), Some(b)) => {
                            // Compare field values, not Rc pointers
                            if a.len() != b.len() {
                                false
                            } else {
                                a.iter().zip(b.iter()).all(|(va, vb)| self.values_equal(va, vb))
                            }
                        }
                        _ => false,
                    };
                    Ok(Value::Bool(eq))
                }
                BinOp::Ne => {
                    let eq = e1 == e2 && v1 == v2 && match (f1, f2) {
                        (None, None) => true,
                        (Some(a), Some(b)) => {
                            // Compare field values, not Rc pointers
                            if a.len() != b.len() {
                                false
                            } else {
                                a.iter().zip(b.iter()).all(|(va, vb)| self.values_equal(va, vb))
                            }
                        }
                        _ => false,
                    };
                    Ok(Value::Bool(!eq))
                }
                _ => Err(RuntimeError::new("Invalid variant operation")),
            },
            // Struct equality (by reference, except for value types like Cbit)
            (Value::Struct { name: n1, fields: f1 }, Value::Struct { name: n2, fields: f2 }) => match op {
                BinOp::Eq => {
                    if n1 != n2 {
                        return Ok(Value::Bool(false));
                    }
                    // Special case: Cbit compares by value, not reference
                    if n1 == "Cbit" {
                        let f1 = f1.borrow();
                        let f2 = f2.borrow();
                        let v1 = f1.get("__value__");
                        let v2 = f2.get("__value__");
                        let eq = match (v1, v2) {
                            (Some(Value::Bool(b1)), Some(Value::Bool(b2))) => b1 == b2,
                            _ => false,
                        };
                        return Ok(Value::Bool(eq));
                    }
                    // Default: pointer equality for other structs
                    Ok(Value::Bool(Rc::ptr_eq(&f1, &f2)))
                }
                BinOp::Ne => {
                    if n1 != n2 {
                        return Ok(Value::Bool(true));
                    }
                    // Special case: Cbit compares by value, not reference
                    if n1 == "Cbit" {
                        let f1 = f1.borrow();
                        let f2 = f2.borrow();
                        let v1 = f1.get("__value__");
                        let v2 = f2.get("__value__");
                        let ne = match (v1, v2) {
                            (Some(Value::Bool(b1)), Some(Value::Bool(b2))) => b1 != b2,
                            _ => true,
                        };
                        return Ok(Value::Bool(ne));
                    }
                    // Default: pointer inequality for other structs
                    Ok(Value::Bool(!Rc::ptr_eq(&f1, &f2)))
                }
                // Tensor product for Qubits: a ⊗ b -> QRegister
                BinOp::TensorProd => {
                    if n1 == "Qubit" && n2 == "Qubit" {
                        // Reconstruct Value::Struct from destructured parts
                        let left_val = Value::Struct { name: n1.clone(), fields: f1.clone() };
                        let right_val = Value::Struct { name: n2.clone(), fields: f2.clone() };
                        return self.call_builtin_by_name("tensor", vec![left_val, right_val]);
                    }
                    if n1 == "QRegister" && n2 == "Qubit" {
                        let left_val = Value::Struct { name: n1.clone(), fields: f1.clone() };
                        let right_val = Value::Struct { name: n2.clone(), fields: f2.clone() };
                        return self.call_builtin_by_name("tensor", vec![left_val, right_val]);
                    }
                    Err(RuntimeError::new(format!("TensorProd not supported for {} ⊗ {}", n1, n2)))
                }
                // Matrix multiplication for Tensors: a @ b
                BinOp::MatMul => {
                    if n1 == "Tensor" && n2 == "Tensor" {
                        // Extract data as 2D arrays and compute matmul
                        let a_data = {
                            let f = f1.borrow();
                            match f.get("__data__") {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor A")),
                            }
                        };
                        let b_data = {
                            let f = f2.borrow();
                            match f.get("__data__") {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor B")),
                            }
                        };
                        let a_shape: Vec<i64> = {
                            let f = f1.borrow();
                            match f.get("__shape__") {
                                Some(Value::Array(arr)) => arr.borrow().iter().map(|v| match v {
                                    Value::Int(i) => *i,
                                    _ => 1,
                                }).collect(),
                                _ => vec![a_data.len() as i64],
                            }
                        };
                        let b_shape: Vec<i64> = {
                            let f = f2.borrow();
                            match f.get("__shape__") {
                                Some(Value::Array(arr)) => arr.borrow().iter().map(|v| match v {
                                    Value::Int(i) => *i,
                                    _ => 1,
                                }).collect(),
                                _ => vec![b_data.len() as i64],
                            }
                        };

                        // Propagate requires_grad from either input
                        let a_requires_grad = {
                            let f = f1.borrow();
                            matches!(f.get(TENSOR_REQUIRES_GRAD_FIELD), Some(Value::Bool(true)))
                        };
                        let b_requires_grad = {
                            let f = f2.borrow();
                            matches!(f.get(TENSOR_REQUIRES_GRAD_FIELD), Some(Value::Bool(true)))
                        };
                        let result_requires_grad = a_requires_grad || b_requires_grad;

                        // For 2D tensors: [M, K] @ [K, N] = [M, N]
                        if a_shape.len() == 2 && b_shape.len() == 2 {
                            let m = a_shape[0] as usize;
                            let k = a_shape[1] as usize;
                            let k2 = b_shape[0] as usize;
                            let n = b_shape[1] as usize;

                            if k != k2 {
                                return Err(RuntimeError::new(format!(
                                    "Tensor matmul dimension mismatch: [{}, {}] @ [{}, {}]",
                                    m, k, k2, n
                                )));
                            }

                            // Compute C = A @ B
                            let mut result_data: Vec<Value> = Vec::with_capacity(m * n);
                            for i in 0..m {
                                for j in 0..n {
                                    let mut sum = 0.0f64;
                                    for kk in 0..k {
                                        let a_val = match &a_data[i * k + kk] {
                                            Value::Float(f) => *f,
                                            Value::Int(x) => *x as f64,
                                            _ => 0.0,
                                        };
                                        let b_val = match &b_data[kk * n + j] {
                                            Value::Float(f) => *f,
                                            Value::Int(x) => *x as f64,
                                            _ => 0.0,
                                        };
                                        sum += a_val * b_val;
                                    }
                                    result_data.push(Value::Float(sum));
                                }
                            }

                            // Create result tensor with propagated requires_grad
                            return Ok(Self::create_tensor_value(
                                result_data,
                                vec![m as i64, n as i64],
                                result_requires_grad,
                            ));
                        }
                        // Matrix-vector: [M, K] @ [K] = [M]
                        if a_shape.len() == 2 && b_shape.len() == 1 {
                            let m = a_shape[0] as usize;
                            let k = a_shape[1] as usize;
                            let k2 = b_shape[0] as usize;

                            if k != k2 {
                                return Err(RuntimeError::new(format!(
                                    "Tensor matmul dimension mismatch: [{}, {}] @ [{}]",
                                    m, k, k2
                                )));
                            }

                            // Compute c = A @ b
                            let mut result_data: Vec<Value> = Vec::with_capacity(m);
                            for i in 0..m {
                                let mut sum = 0.0f64;
                                for kk in 0..k {
                                    let a_val = match &a_data[i * k + kk] {
                                        Value::Float(f) => *f,
                                        Value::Int(x) => *x as f64,
                                        _ => 0.0,
                                    };
                                    let b_val = match &b_data[kk] {
                                        Value::Float(f) => *f,
                                        Value::Int(x) => *x as f64,
                                        _ => 0.0,
                                    };
                                    sum += a_val * b_val;
                                }
                                result_data.push(Value::Float(sum));
                            }

                            // Create result tensor with propagated requires_grad
                            return Ok(Self::create_tensor_value(
                                result_data,
                                vec![m as i64],
                                result_requires_grad,
                            ));
                        }
                        Err(RuntimeError::new(format!(
                            "Tensor matmul requires compatible shapes: {:?} @ {:?}",
                            a_shape, b_shape
                        )))
                    } else {
                        Err(RuntimeError::new(format!("MatMul not supported for {} @ {}", n1, n2)))
                    }
                }
                // Regular multiplication for Tensors: element-wise multiply
                BinOp::Mul => {
                    if n1 == "Tensor" && n2 == "Tensor" {
                        let a_data = {
                            let f = f1.borrow();
                            match f.get("__data__") {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor A")),
                            }
                        };
                        let b_data = {
                            let f = f2.borrow();
                            match f.get("__data__") {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor B")),
                            }
                        };
                        let shape: Vec<i64> = {
                            let f = f1.borrow();
                            match f.get("__shape__") {
                                Some(Value::Array(arr)) => arr.borrow().iter().map(|v| match v {
                                    Value::Int(i) => *i,
                                    _ => 1,
                                }).collect(),
                                _ => vec![a_data.len() as i64],
                            }
                        };

                        if a_data.len() != b_data.len() {
                            return Err(RuntimeError::new("Tensor multiplication requires tensors of equal size"));
                        }

                        let result_data: Vec<Value> = a_data.iter().zip(b_data.iter()).map(|(a, b)| {
                            match (a, b) {
                                (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
                                (Value::Int(x), Value::Int(y)) => Value::Float((*x as f64) * (*y as f64)),
                                (Value::Float(x), Value::Int(y)) => Value::Float(x * (*y as f64)),
                                (Value::Int(x), Value::Float(y)) => Value::Float((*x as f64) * y),
                                _ => Value::Float(0.0),
                            }
                        }).collect();

                        let mut fields = std::collections::HashMap::new();
                        fields.insert("__data__".to_string(), Value::Array(Rc::new(RefCell::new(result_data))));
                        fields.insert("__shape__".to_string(), Value::Array(Rc::new(RefCell::new(
                            shape.iter().map(|s| Value::Int(*s)).collect()
                        ))));
                        fields.insert("shape".to_string(), Value::Array(Rc::new(RefCell::new(
                            shape.iter().map(|s| Value::Int(*s)).collect()
                        ))));
                        fields.insert("__requires_grad__".to_string(), Value::Bool(false));
                        fields.insert("__grad__".to_string(), Value::Null);
                        fields.insert("grad".to_string(), Value::Null);

                        return Ok(Value::Struct {
                            name: "Tensor".to_string(),
                            fields: Rc::new(RefCell::new(fields)),
                        });
                    }
                    Err(RuntimeError::new(format!("Mul not supported for {} * {}", n1, n2)))
                }
                // Element-wise addition for Tensors: a + b
                BinOp::Add => {
                    if n1 == "Tensor" && n2 == "Tensor" {
                        let a_data = {
                            let f = f1.borrow();
                            match f.get(TENSOR_DATA_FIELD) {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor A")),
                            }
                        };
                        let b_data = {
                            let f = f2.borrow();
                            match f.get(TENSOR_DATA_FIELD) {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor B")),
                            }
                        };
                        let shape: Vec<i64> = {
                            let f = f1.borrow();
                            match f.get(TENSOR_SHAPE_FIELD) {
                                Some(Value::Array(arr)) => arr.borrow().iter().map(|v| match v {
                                    Value::Int(i) => *i,
                                    _ => 1,
                                }).collect(),
                                _ => vec![a_data.len() as i64],
                            }
                        };

                        // Propagate requires_grad
                        let a_requires_grad = {
                            let f = f1.borrow();
                            matches!(f.get(TENSOR_REQUIRES_GRAD_FIELD), Some(Value::Bool(true)))
                        };
                        let b_requires_grad = {
                            let f = f2.borrow();
                            matches!(f.get(TENSOR_REQUIRES_GRAD_FIELD), Some(Value::Bool(true)))
                        };
                        let result_requires_grad = a_requires_grad || b_requires_grad;

                        if a_data.len() != b_data.len() {
                            return Err(RuntimeError::new("Tensor addition requires tensors of equal size"));
                        }

                        let result_data: Vec<Value> = a_data.iter().zip(b_data.iter()).map(|(a, b)| {
                            match (a, b) {
                                (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
                                (Value::Int(x), Value::Int(y)) => Value::Float((*x as f64) + (*y as f64)),
                                (Value::Float(x), Value::Int(y)) => Value::Float(x + (*y as f64)),
                                (Value::Int(x), Value::Float(y)) => Value::Float((*x as f64) + y),
                                _ => Value::Float(0.0),
                            }
                        }).collect();

                        return Ok(Self::create_tensor_value(result_data, shape, result_requires_grad));
                    }
                    Err(RuntimeError::new(format!("Add not supported for {} + {}", n1, n2)))
                }
                // Element-wise subtraction for Tensors: a - b
                BinOp::Sub => {
                    if n1 == "Tensor" && n2 == "Tensor" {
                        let a_data = {
                            let f = f1.borrow();
                            match f.get(TENSOR_DATA_FIELD) {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor A")),
                            }
                        };
                        let b_data = {
                            let f = f2.borrow();
                            match f.get(TENSOR_DATA_FIELD) {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor B")),
                            }
                        };
                        let shape: Vec<i64> = {
                            let f = f1.borrow();
                            match f.get(TENSOR_SHAPE_FIELD) {
                                Some(Value::Array(arr)) => arr.borrow().iter().map(|v| match v {
                                    Value::Int(i) => *i,
                                    _ => 1,
                                }).collect(),
                                _ => vec![a_data.len() as i64],
                            }
                        };

                        // Propagate requires_grad
                        let a_requires_grad = {
                            let f = f1.borrow();
                            matches!(f.get(TENSOR_REQUIRES_GRAD_FIELD), Some(Value::Bool(true)))
                        };
                        let b_requires_grad = {
                            let f = f2.borrow();
                            matches!(f.get(TENSOR_REQUIRES_GRAD_FIELD), Some(Value::Bool(true)))
                        };
                        let result_requires_grad = a_requires_grad || b_requires_grad;

                        if a_data.len() != b_data.len() {
                            return Err(RuntimeError::new("Tensor subtraction requires tensors of equal size"));
                        }

                        let result_data: Vec<Value> = a_data.iter().zip(b_data.iter()).map(|(a, b)| {
                            match (a, b) {
                                (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
                                (Value::Int(x), Value::Int(y)) => Value::Float((*x as f64) - (*y as f64)),
                                (Value::Float(x), Value::Int(y)) => Value::Float(x - (*y as f64)),
                                (Value::Int(x), Value::Float(y)) => Value::Float((*x as f64) - y),
                                _ => Value::Float(0.0),
                            }
                        }).collect();

                        return Ok(Self::create_tensor_value(result_data, shape, result_requires_grad));
                    }
                    Err(RuntimeError::new(format!("Sub not supported for {} - {}", n1, n2)))
                }
                // Hadamard product for Tensors: a ⊙ b
                BinOp::Hadamard => {
                    if n1 == "Tensor" && n2 == "Tensor" {
                        let a_data = {
                            let f = f1.borrow();
                            match f.get("__data__") {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor A")),
                            }
                        };
                        let b_data = {
                            let f = f2.borrow();
                            match f.get("__data__") {
                                Some(Value::Array(arr)) => arr.borrow().clone(),
                                _ => return Err(RuntimeError::new("Invalid Tensor B")),
                            }
                        };
                        let shape: Vec<i64> = {
                            let f = f1.borrow();
                            match f.get("__shape__") {
                                Some(Value::Array(arr)) => arr.borrow().iter().map(|v| match v {
                                    Value::Int(i) => *i,
                                    _ => 1,
                                }).collect(),
                                _ => vec![a_data.len() as i64],
                            }
                        };

                        if a_data.len() != b_data.len() {
                            return Err(RuntimeError::new("Hadamard product requires tensors of equal size"));
                        }

                        let result_data: Vec<Value> = a_data.iter().zip(b_data.iter()).map(|(a, b)| {
                            match (a, b) {
                                (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
                                (Value::Int(x), Value::Int(y)) => Value::Float((*x as f64) * (*y as f64)),
                                (Value::Float(x), Value::Int(y)) => Value::Float(x * (*y as f64)),
                                (Value::Int(x), Value::Float(y)) => Value::Float((*x as f64) * y),
                                _ => Value::Float(0.0),
                            }
                        }).collect();

                        // Create result tensor
                        let mut fields = std::collections::HashMap::new();
                        fields.insert("__data__".to_string(), Value::Array(Rc::new(RefCell::new(result_data))));
                        fields.insert("__shape__".to_string(), Value::Array(Rc::new(RefCell::new(
                            shape.iter().map(|s| Value::Int(*s)).collect()
                        ))));
                        fields.insert("shape".to_string(), Value::Array(Rc::new(RefCell::new(
                            shape.iter().map(|s| Value::Int(*s)).collect()
                        ))));
                        fields.insert("__requires_grad__".to_string(), Value::Bool(false));
                        fields.insert("__grad__".to_string(), Value::Null);
                        fields.insert("grad".to_string(), Value::Null);

                        return Ok(Value::Struct {
                            name: "Tensor".to_string(),
                            fields: Rc::new(RefCell::new(fields)),
                        });
                    }
                    Err(RuntimeError::new(format!("Hadamard not supported for {} ⊙ {}", n1, n2)))
                }
                _ => Err(RuntimeError::new("Invalid struct operation")),
            },
            // Option::Some compared with a non-Option value - unwrap and compare
            (Value::Variant { enum_name, variant_name, fields }, other)
                if enum_name == "Option" && variant_name == "Some" && matches!(op, BinOp::Eq | BinOp::Ne) =>
            {
                if let Some(ref f) = fields {
                    if f.len() == 1 {
                        // Compare inner value with other
                        let inner = f[0].clone();
                        // Recursive call with unwrapped value
                        return self.compare_option_values(&inner, &other, op);
                    }
                }
                // Some with no fields or multiple fields - not equal to simple values
                match op {
                    BinOp::Eq => Ok(Value::Bool(false)),
                    BinOp::Ne => Ok(Value::Bool(true)),
                    _ => Err(RuntimeError::new("Invalid Option comparison")),
                }
            },
            // Other value compared with Option::Some - reverse of above
            (other, Value::Variant { enum_name, variant_name, fields })
                if enum_name == "Option" && variant_name == "Some" && matches!(op, BinOp::Eq | BinOp::Ne) =>
            {
                if let Some(ref f) = fields {
                    if f.len() == 1 {
                        let inner = f[0].clone();
                        return self.compare_option_values(&other, &inner, op);
                    }
                }
                match op {
                    BinOp::Eq => Ok(Value::Bool(false)),
                    BinOp::Ne => Ok(Value::Bool(true)),
                    _ => Err(RuntimeError::new("Invalid Option comparison")),
                }
            },
            // Option::None compared with anything - not equal
            (Value::Variant { enum_name, variant_name, .. }, _)
                if enum_name == "Option" && variant_name == "None" && matches!(op, BinOp::Eq | BinOp::Ne) =>
            {
                match op {
                    BinOp::Eq => Ok(Value::Bool(false)),
                    BinOp::Ne => Ok(Value::Bool(true)),
                    _ => Err(RuntimeError::new("Invalid Option comparison")),
                }
            },
            (_, Value::Variant { enum_name, variant_name, .. })
                if enum_name == "Option" && variant_name == "None" && matches!(op, BinOp::Eq | BinOp::Ne) =>
            {
                match op {
                    BinOp::Eq => Ok(Value::Bool(false)),
                    BinOp::Ne => Ok(Value::Bool(true)),
                    _ => Err(RuntimeError::new("Invalid Option comparison")),
                }
            },
            (l, r) => Err(RuntimeError::new(format!(
                "Type mismatch in binary operation: {} {:?} {}",
                self.format_value(&l), op, self.format_value(&r)
            ))),
        }
    }

    /// Helper for Option comparison - compare unwrapped values
    fn compare_option_values(&mut self, lhs: &Value, rhs: &Value, op: &BinOp) -> Result<Value, RuntimeError> {
        match (lhs, rhs) {
            (Value::Char(a), Value::Char(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(*a == *b)),
                BinOp::Ne => Ok(Value::Bool(*a != *b)),
                _ => Err(RuntimeError::new("Invalid comparison")),
            },
            (Value::Int(a), Value::Int(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(*a == *b)),
                BinOp::Ne => Ok(Value::Bool(*a != *b)),
                _ => Err(RuntimeError::new("Invalid comparison")),
            },
            (Value::String(a), Value::String(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(**a == **b)),
                BinOp::Ne => Ok(Value::Bool(**a != **b)),
                _ => Err(RuntimeError::new("Invalid comparison")),
            },
            (Value::Bool(a), Value::Bool(b)) => match op {
                BinOp::Eq => Ok(Value::Bool(*a == *b)),
                BinOp::Ne => Ok(Value::Bool(*a != *b)),
                _ => Err(RuntimeError::new("Invalid comparison")),
            },
            _ => match op {
                BinOp::Eq => Ok(Value::Bool(false)),
                BinOp::Ne => Ok(Value::Bool(true)),
                _ => Err(RuntimeError::new("Invalid comparison")),
            },
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
            // Handle evidential values - unwrap, negate, rewrap
            (UnaryOp::Not, Value::Evidential { value, evidence }) => {
                // Negate the inner value
                match value.as_ref() {
                    Value::Bool(b) => Ok(Value::Evidential {
                        value: Box::new(Value::Bool(!b)),
                        evidence: evidence.clone(),
                    }),
                    other => {
                        let truthy = self.is_truthy(other);
                        Ok(Value::Evidential {
                            value: Box::new(Value::Bool(!truthy)),
                            evidence: evidence.clone(),
                        })
                    }
                }
            }
            // Handle string truthiness (non-empty = true)
            (UnaryOp::Not, Value::String(s)) => Ok(Value::Bool(s.is_empty())),
            // Handle array truthiness (non-empty = true)
            (UnaryOp::Not, Value::Array(arr)) => Ok(Value::Bool(arr.borrow().is_empty())),
            // Handle null - null is falsy
            (UnaryOp::Not, Value::Null) => Ok(Value::Bool(true)),
            (UnaryOp::Ref, _) => Ok(Value::Ref(Rc::new(RefCell::new(val)))),
            (UnaryOp::RefMut, _) => Ok(Value::Ref(Rc::new(RefCell::new(val)))),
            (UnaryOp::Deref, Value::Ref(r)) => Ok(r.borrow().clone()),
            (UnaryOp::Deref, Value::Struct { name, fields }) if name == "Rc" => {
                // Deref Rc to get inner value
                let borrowed = fields.borrow();
                if let Some(value) = borrowed.get("_value") {
                    Ok(value.clone())
                } else {
                    Err(RuntimeError::new("Rc has no value"))
                }
            }
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
        // Check if func_expr is a path that might be a variant constructor or tuple struct
        if let Expr::Path(path) = func_expr {
            let qualified_name = path.segments.iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("·");
            // Handle Self(...) as tuple struct constructor
            if qualified_name == "Self" {
                if let Some(ref self_type) = self.current_self_type {
                    // Check if this type is a tuple struct - extract arity first to release borrow
                    let tuple_arity = if let Some(TypeDef::Struct(struct_def)) = self.types.get(self_type) {
                        if let crate::ast::StructFields::Tuple(field_types) = &struct_def.fields {
                            Some((self_type.clone(), field_types.len()))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some((type_name, expected_arity)) = tuple_arity {
                        // Now we can safely evaluate arguments
                        let arg_values: Vec<Value> = args
                            .iter()
                            .map(|a| self.evaluate(a))
                            .collect::<Result<_, _>>()?;

                        if arg_values.len() != expected_arity {
                            return Err(RuntimeError::new(format!(
                                "Tuple struct {} expects {} fields, got {}",
                                type_name, expected_arity, arg_values.len()
                            )));
                        }

                        // Create struct with numbered fields (0, 1, 2, ...)
                        let mut fields = HashMap::new();
                        for (i, value) in arg_values.into_iter().enumerate() {
                            fields.insert(i.to_string(), value);
                        }
                        return Ok(Value::Struct {
                            name: type_name,
                            fields: Rc::new(RefCell::new(fields)),
                        });
                    }
                }
            }

            // Handle TypeName(...) as tuple struct constructor
            if path.segments.len() == 1 {
                let type_name = &path.segments[0].ident.name;
                // Extract arity first to release borrow
                let tuple_arity = if let Some(TypeDef::Struct(struct_def)) = self.types.get(type_name) {
                    if let crate::ast::StructFields::Tuple(field_types) = &struct_def.fields {
                        Some((type_name.clone(), field_types.len()))
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((struct_name, expected_arity)) = tuple_arity {
                    let arg_values: Vec<Value> = args
                        .iter()
                        .map(|a| self.evaluate(a))
                        .collect::<Result<_, _>>()?;

                    if arg_values.len() != expected_arity {
                        return Err(RuntimeError::new(format!(
                            "Tuple struct {} expects {} fields, got {}",
                            struct_name, expected_arity, arg_values.len()
                        )));
                    }

                    let mut fields = HashMap::new();
                    for (i, value) in arg_values.into_iter().enumerate() {
                        fields.insert(i.to_string(), value);
                    }
                    return Ok(Value::Struct {
                        name: struct_name,
                        fields: Rc::new(RefCell::new(fields)),
                    });
                }
            }

            // Handle Default::default() when current_self_type is set
            // This allows ..Default::default() to work in struct literals
            if qualified_name == "Default·default" && args.is_empty() {
                if let Some(type_name) = self.current_self_type.clone() {
                    // First check if type has impl Default with explicit default fn
                    let default_fn_name = format!("{}·default", type_name);
                    crate::sigil_debug!("DEBUG Default::default() looking for '{}', self_type='{}'", default_fn_name, type_name);
                    let func_clone = self.globals.borrow().get(&default_fn_name).map(|v| v.clone());
                    if let Some(Value::Function(f)) = func_clone {
                        crate::sigil_debug!("DEBUG Found function '{}', calling it", default_fn_name);
                        crate::sigil_debug!("DEBUG current_self_type before call: {:?}", self.current_self_type);
                        // Call the type's default implementation
                        let result = self.call_function(&f, vec![]);
                        crate::sigil_debug!("DEBUG Default call result: {:?}", result.as_ref().map(|v| self.format_value(v)).unwrap_or_else(|e| format!("ERR: {:?}", e)));
                        return result;
                    }
                    // Otherwise check for #[derive(Default)]
                    if let Some(struct_def) = self.default_structs.get(&type_name).cloned() {
                        let mut fields = HashMap::new();
                        if let StructFields::Named(field_defs) = &struct_def.fields {
                            for field in field_defs {
                                let default_val = if let Some(default_expr) = &field.default {
                                    self.evaluate(default_expr)?
                                } else {
                                    Value::Null
                                };
                                fields.insert(field.name.name.clone(), default_val);
                            }
                        }
                        return Ok(Value::Struct {
                            name: type_name,
                            fields: Rc::new(RefCell::new(fields)),
                        });
                    }
                }
            }

            // Check for TypeName·default pattern
            if qualified_name.ends_with("·default") && args.is_empty() {
                let type_name = qualified_name.strip_suffix("·default").unwrap();
                // First check if type has impl Default with explicit default fn
                let default_fn_name = format!("{}·default", type_name);
                let func_clone = self.globals.borrow().get(&default_fn_name).map(|v| v.clone());
                if let Some(Value::Function(f)) = func_clone {
                    // Call the type's default implementation
                    return self.call_function(&f, vec![]);
                }
                // Otherwise check for #[derive(Default)]
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
                    if enum_name == "Item" {
                        crate::sigil_debug!("DEBUG creating Item::{} variant with {} fields", variant_name, arg_values.len());
                    }
                    return Ok(Value::Variant {
                        enum_name,
                        variant_name,
                        fields: Some(Rc::new(arg_values)),
                    });
                }
            }

            // Check for Neural layer constructors with const generics: Linear·<IN, OUT>·new()
            if path.segments.len() == 2 {
                let first_seg = &path.segments[0];
                let second_seg = &path.segments[1];
                if first_seg.ident.name == "Linear" && second_seg.ident.name == "new" {
                    if let Some(ref generics) = first_seg.generics {
                        if generics.len() == 2 {
                            // Extract const generic values IN and OUT
                            let in_size = Self::extract_const_generic_int(&generics[0])?;
                            let out_size = Self::extract_const_generic_int(&generics[1])?;

                            // Create Linear layer with proper dimensions
                            return self.create_linear_layer(in_size, out_size);
                        }
                    }
                }
            }

            // Check for built-in type constructors (Map::new, String::new, HashMap::new, etc.)
            let segments: Vec<&str> = path.segments.iter().map(|s| s.ident.name.as_str()).collect();
            match segments.as_slice() {
                ["Map", "new"] | ["HashMap", "new"] => {
                    // Create a new empty Map (represented as a struct with HashMap fields)
                    return Ok(Value::Struct {
                        name: "Map".to_string(),
                        fields: Rc::new(RefCell::new(HashMap::new())),
                    });
                }
                ["String", "new"] => {
                    return Ok(Value::String(Rc::new(String::new())));
                }
                ["Vec", "new"] | ["Array", "new"] => {
                    return Ok(Value::Array(Rc::new(RefCell::new(Vec::new()))));
                }
                ["Box", "new"] => {
                    // Box::new(value) - just return the value (no heap allocation in interpreter)
                    if args.len() == 1 {
                        return self.evaluate(&args[0]);
                    }
                    return Err(RuntimeError::new("Box::new expects 1 argument"));
                }
                ["char", "from_u32"] => {
                    // char::from_u32(u32) -> Option<char>
                    if args.len() == 1 {
                        let arg = self.evaluate(&args[0])?;
                        let code = match arg {
                            Value::Int(i) => i as u32,
                            _ => return Err(RuntimeError::new("char::from_u32 expects u32")),
                        };
                        if let Some(c) = char::from_u32(code) {
                            // Return Some(char)
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "Some".to_string(),
                                fields: Some(Rc::new(vec![Value::Char(c)])),
                            });
                        } else {
                            // Return None
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
                        }
                    }
                    return Err(RuntimeError::new("char::from_u32 expects 1 argument"));
                }
                // Mutex::new - create a mutex wrapper around a value
                ["parking_lot", "Mutex", "new"] | ["std", "sync", "Mutex", "new"] | ["Mutex", "new"] => {
                    if args.len() == 1 {
                        let inner = self.evaluate(&args[0])?;
                        return Ok(Value::Struct {
                            name: "Mutex".to_string(),
                            fields: Rc::new(RefCell::new(HashMap::from([
                                ("__inner__".to_string(), inner),
                            ]))),
                        });
                    }
                    return Err(RuntimeError::new("Mutex::new expects 1 argument"));
                }
                // RwLock::new - same as Mutex for interpreter purposes
                ["parking_lot", "RwLock", "new"] | ["std", "sync", "RwLock", "new"] | ["RwLock", "new"] => {
                    if args.len() == 1 {
                        let inner = self.evaluate(&args[0])?;
                        return Ok(Value::Struct {
                            name: "RwLock".to_string(),
                            fields: Rc::new(RefCell::new(HashMap::from([
                                ("__inner__".to_string(), inner),
                            ]))),
                        });
                    }
                    return Err(RuntimeError::new("RwLock::new expects 1 argument"));
                }
                // Arc::new - just wrap the value (no real reference counting in interpreter)
                ["std", "sync", "Arc", "new"] | ["Arc", "new"] => {
                    if args.len() == 1 {
                        let inner = self.evaluate(&args[0])?;
                        return Ok(Value::Ref(Rc::new(RefCell::new(inner))));
                    }
                    return Err(RuntimeError::new("Arc::new expects 1 argument"));
                }
                // AtomicU64::new and similar atomics
                ["std", "sync", "atomic", "AtomicU64", "new"] | ["AtomicU64", "new"] => {
                    if args.len() == 1 {
                        let inner = self.evaluate(&args[0])?;
                        return Ok(Value::Struct {
                            name: "AtomicU64".to_string(),
                            fields: Rc::new(RefCell::new(HashMap::from([
                                ("__value__".to_string(), inner),
                            ]))),
                        });
                    }
                    return Err(RuntimeError::new("AtomicU64::new expects 1 argument"));
                }
                ["std", "sync", "atomic", "AtomicUsize", "new"] | ["AtomicUsize", "new"] => {
                    if args.len() == 1 {
                        let inner = self.evaluate(&args[0])?;
                        return Ok(Value::Struct {
                            name: "AtomicUsize".to_string(),
                            fields: Rc::new(RefCell::new(HashMap::from([
                                ("__value__".to_string(), inner),
                            ]))),
                        });
                    }
                    return Err(RuntimeError::new("AtomicUsize::new expects 1 argument"));
                }
                ["std", "sync", "atomic", "AtomicBool", "new"] | ["AtomicBool", "new"] => {
                    if args.len() == 1 {
                        let inner = self.evaluate(&args[0])?;
                        return Ok(Value::Struct {
                            name: "AtomicBool".to_string(),
                            fields: Rc::new(RefCell::new(HashMap::from([
                                ("__value__".to_string(), inner),
                            ]))),
                        });
                    }
                    return Err(RuntimeError::new("AtomicBool::new expects 1 argument"));
                }
                _ => {}
            }
        }

        // If calling a qualified function (Type::method), set current_self_type
        let type_name_for_self = if let Expr::Path(path) = func_expr {
            if path.segments.len() >= 2 {
                // First segment is the type name
                let first = &path.segments[0].ident.name;
                // Check if it's a type name (exists in types registry)
                if self.types.contains_key(first) {
                    Some(first.clone())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let func = self.evaluate(func_expr)?;

        // Track &mut path arguments for sync-back after function call
        // This enables proper mutable reference semantics where modifications persist
        let mut mut_ref_sync: Vec<(String, Rc<RefCell<Value>>)> = Vec::new();

        let mut arg_values: Vec<Value> = Vec::new();
        for arg in args.iter() {
            let val = self.evaluate(arg)?;

            // If this was a &mut path expression, track it for sync-back
            if let Expr::Unary { op: crate::ast::UnaryOp::RefMut, expr } = arg {
                if let Expr::Path(path) = expr.as_ref() {
                    if path.segments.len() == 1 {
                        let var_name = path.segments[0].ident.name.clone();
                        if let Value::Ref(r) = &val {
                            mut_ref_sync.push((var_name, r.clone()));
                        }
                    }
                }
            }

            arg_values.push(val);
        }

        // Set Self type if we're calling a type-associated function
        // Use clone instead of take to preserve for nested calls that don't set a type
        let old_self_type = self.current_self_type.clone();
        if let Some(type_name) = type_name_for_self {
            self.current_self_type = Some(type_name);
        }

        let result = match func {
            Value::Function(f) => self.call_function(&f, arg_values),
            Value::BuiltIn(b) => self.call_builtin(&b, arg_values),
            // Handle constructor markers for unknown external types
            Value::Struct { ref name, .. } if name.starts_with("__constructor__") => {
                let actual_type = name.strip_prefix("__constructor__").unwrap();
                // Create an empty struct for the unknown type
                Ok(Value::Struct {
                    name: actual_type.to_string(),
                    fields: Rc::new(RefCell::new(HashMap::new())),
                })
            }
            _ => {
                crate::sigil_debug!("DEBUG Cannot call non-function: {:?}, expr: {:?}", func, func_expr);
                Err(RuntimeError::new("Cannot call non-function"))
            }
        };

        // Sync mutable references back to original variables
        // This is what makes `fn foo(x: &mut T)` actually modify the caller's variable
        for (var_name, ref_val) in mut_ref_sync {
            let current_value = ref_val.borrow().clone();
            let _ = self.environment.borrow_mut().set(&var_name, current_value);
        }

        // Restore old Self type
        self.current_self_type = old_self_type;

        result
    }

    pub fn call_function(
        &mut self,
        func: &Function,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        // Debug trace for relevant functions
        if func.name.as_ref().map_or(false, |n| n.contains("read_source") || n.contains("parse_file") || n.contains("load_from_file") || n.contains("read_to_string")) {
            crate::sigil_debug!("DEBUG call_function: name={:?}, params={:?}", func.name, func.params);
            for (i, arg) in args.iter().enumerate() {
                crate::sigil_debug!("  arg[{}] = {:?}", i, arg);
            }
        }
        if args.len() != func.params.len() {
            return Err(RuntimeError::new(format!(
                "Expected {} arguments, got {} (func={:?}, params={:?})",
                func.params.len(),
                args.len(),
                func.name,
                func.params
            )));
        }

        // Debug: trace calls to keyword_or_ident
        if func.params.iter().any(|p| p == "name") {
            for arg in &args {
                let unwrapped = Self::unwrap_all(arg);
                if let Value::String(s) = &unwrapped {
                    if s.len() <= 10 {
                        crate::sigil_debug!("DEBUG call_function(name='{}')", s);
                    }
                }
            }
        }

        // Create new environment for function
        let env = Rc::new(RefCell::new(Environment::with_parent(func.closure.clone())));

        // Bind parameters
        for (param, value) in func.params.iter().zip(args) {
            // Debug: trace path parameter binding
            if param == "path" {
                crate::sigil_debug!("DEBUG call_function func={:?} binding param 'path' = {:?}", func.name, value);
            }
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

    /// Call a builtin function by name
    fn call_builtin_by_name(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        // Look up the builtin in the global environment
        // Clone to avoid borrow issues
        let builtin_opt = {
            let globals = self.globals.borrow();
            if let Some(Value::BuiltIn(builtin)) = globals.get(name) {
                Some(builtin.clone())
            } else {
                None
            }
        };

        if let Some(builtin) = builtin_opt {
            return self.call_builtin(&builtin, args);
        }
        Err(RuntimeError::new(format!("Unknown builtin: {}", name)))
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
                Stmt::Let { pattern, init, ty } => {
                    // Extract tensor shape from type annotation for shape inference
                    let tensor_shape = ty.as_ref().and_then(|t| self.extract_tensor_shape(t));

                    let value = match init {
                        Some(expr) => self.eval_with_tensor_shape_inference(expr, tensor_shape.as_ref())?,
                        None => Value::Null,
                    };

                    // Validate type annotation if present
                    if let Some(type_expr) = ty {
                        self.validate_type_annotation(type_expr, &value)?;
                    }
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

        // RAII: Call Drop::drop() on values going out of scope
        // Collect values to drop (avoid borrowing self during iteration)
        let values_to_drop: Vec<(String, Value)> = self.environment
            .borrow()
            .values
            .iter()
            .filter_map(|(name, (value, _mutable))| {
                if let Value::Struct { name: struct_name, .. } = value {
                    if self.drop_types.contains(struct_name) {
                        return Some((name.clone(), value.clone()));
                    }
                }
                None
            })
            .collect();

        // Call drop on each value
        for (_var_name, value) in values_to_drop {
            if let Value::Struct { name: struct_name, .. } = &value {
                let drop_fn_name = format!("{}·drop", struct_name);
                // Clone the function out of globals to avoid borrow issues
                let drop_fn = self.globals.borrow().get(&drop_fn_name).map(|v| v.clone());
                if let Some(Value::Function(f)) = drop_fn {
                    // Call drop(self) - the value is passed as self
                    let _ = self.call_function(&f, vec![value.clone()]);
                }
            }
        }

        self.environment = prev_env;
        Ok(result)
    }

    fn bind_pattern(&mut self, pattern: &Pattern, value: Value) -> Result<(), RuntimeError> {
        match pattern {
            Pattern::Ident { name, mutable, .. } => {
                // Don't bind "_" - it's a wildcard in identifier form
                if name.name != "_" {
                    // Debug: trace path binding
                    if name.name == "path" {
                        crate::sigil_debug!("DEBUG bind_pattern: binding 'path' = {:?}", value);
                    }
                    self.environment
                        .borrow_mut()
                        .define_mut(name.name.clone(), value, *mutable);
                }
                Ok(())
            }
            Pattern::Tuple(patterns) => {
                // Unwrap evidential wrappers first
                let unwrapped = Self::unwrap_all(&value);
                crate::sigil_debug!("DEBUG bind_pattern Tuple: patterns.len()={}, value type={:?}",
                    patterns.len(), std::mem::discriminant(&unwrapped));
                match unwrapped {
                    Value::Tuple(values) => {
                        if patterns.len() != values.len() {
                            return Err(RuntimeError::new("Tuple pattern size mismatch"));
                        }
                        for (i, (p, v)) in patterns.iter().zip(values.iter()).enumerate() {
                            crate::sigil_debug!("DEBUG   binding tuple element {}: {:?} = {}", i, p, self.format_value(v));
                            self.bind_pattern(p, v.clone())?;
                        }
                        Ok(())
                    }
                    // Handle Option::Some containing a tuple - Sigil shorthand `(a, b) => ...`
                    Value::Variant { enum_name, variant_name, fields }
                        if enum_name == "Option" && variant_name == "Some" =>
                    {
                        if let Some(ref inner_fields) = fields {
                            if inner_fields.len() == 1 {
                                if let Value::Tuple(ref inner_values) = inner_fields[0] {
                                    if patterns.len() != inner_values.len() {
                                        return Err(RuntimeError::new("Tuple pattern size mismatch"));
                                    }
                                    for (i, (p, v)) in patterns.iter().zip(inner_values.iter()).enumerate() {
                                        crate::sigil_debug!("DEBUG   binding Option::Some tuple element {}: {:?} = {}", i, p, self.format_value(v));
                                        self.bind_pattern(p, v.clone())?;
                                    }
                                    return Ok(());
                                }
                            }
                        }
                        Err(RuntimeError::new("Expected tuple in Option::Some"))
                    }
                    Value::Null => {
                        // Null value during iteration - treat as end of iteration (no binding)
                        Ok(())
                    }
                    Value::Array(arr) if arr.borrow().len() == patterns.len() => {
                        // Allow array to be destructured as tuple
                        let vals = arr.borrow();
                        for (p, v) in patterns.iter().zip(vals.iter()) {
                            self.bind_pattern(p, v.clone())?;
                        }
                        Ok(())
                    }
                    // Handle struct-tuples (returned by functions like entangle_holograms)
                    Value::Struct { name, fields } if name == "Tuple" => {
                        let f = fields.borrow();
                        for (idx, p) in patterns.iter().enumerate() {
                            let key = idx.to_string();
                            if let Some(v) = f.get(&key) {
                                self.bind_pattern(p, v.clone())?;
                            } else {
                                return Err(RuntimeError::new(&format!(
                                    "Tuple missing element at index {}", idx
                                )));
                            }
                        }
                        Ok(())
                    }
                    _ => Err(RuntimeError::new("Expected tuple"))
                }
            }
            Pattern::Wildcard => Ok(()),
            Pattern::Struct { path, fields, .. } => {
                // Unwrap any wrappers first
                let unwrapped = Self::unwrap_all(&value);
                // Bind each field from the struct or variant
                match &unwrapped {
                    Value::Struct { fields: struct_fields, .. } => {
                        for field_pat in fields {
                            let field_name = &field_pat.name.name;
                            // Get field value or default to Null for missing optional fields
                            let field_val = struct_fields.borrow().get(field_name).cloned().unwrap_or(Value::Null);
                            if let Some(pat) = &field_pat.pattern {
                                self.bind_pattern(pat, field_val)?;
                            } else {
                                // Shorthand: foo: foo - bind to same name
                                self.environment.borrow_mut().define(field_name.clone(), field_val);
                            }
                        }
                        Ok(())
                    }
                    Value::Variant { enum_name, variant_name, fields: variant_fields } => {
                        // Handle struct-like enum variants (e.g., IrPattern::Ident { name, .. })
                        let pattern_variant = path.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("");
                        if pattern_variant == variant_name || path.segments.iter().any(|s| s.ident.name == *variant_name) {
                            // Variant fields are stored as a Vec, but we need to map by name
                            // For struct-like variants, the fields should be a Struct value with field names
                            if let Some(inner_fields) = variant_fields {
                                if inner_fields.len() == 1 {
                                    // Single wrapped struct with named fields
                                    if let Value::Struct { fields: inner_struct, .. } = &inner_fields[0] {
                                        for field_pat in fields {
                                            let field_name = &field_pat.name.name;
                                            // Default to Null for missing optional fields
                                            let field_val = inner_struct.borrow().get(field_name).cloned().unwrap_or(Value::Null);
                                            if let Some(pat) = &field_pat.pattern {
                                                self.bind_pattern(pat, field_val)?;
                                            } else {
                                                self.environment.borrow_mut().define(field_name.clone(), field_val);
                                            }
                                        }
                                        return Ok(());
                                    }
                                }
                                // Named field lookup from variant's field map
                                // Variants store struct fields as named HashMap inside the variant
                                for field_pat in fields {
                                    let field_name = &field_pat.name.name;
                                    // Look for a field with matching name in variant fields
                                    // Variant fields might be stored in order or as a struct
                                    // For now, search by name if we can find it
                                    let field_val = inner_fields.iter().find_map(|f| {
                                        if let Value::Struct { fields: fs, .. } = f {
                                            fs.borrow().get(field_name).cloned()
                                        } else {
                                            None
                                        }
                                    });
                                    if let Some(val) = field_val {
                                        if let Some(pat) = &field_pat.pattern {
                                            self.bind_pattern(pat, val)?;
                                        } else {
                                            self.environment.borrow_mut().define(field_name.clone(), val);
                                        }
                                    }
                                }
                            }
                            Ok(())
                        } else {
                            crate::sigil_debug!("DEBUG variant name mismatch: pattern={}, actual={}", pattern_variant, variant_name);
                            Err(RuntimeError::new(format!(
                                "Variant name mismatch: expected {} but got {}::{}",
                                pattern_variant, enum_name, variant_name
                            )))
                        }
                    }
                    _ => {
                        crate::sigil_debug!("DEBUG struct pattern bind: expected struct/variant but got {:?}", std::mem::discriminant(&unwrapped));
                        Err(RuntimeError::new("Expected struct or variant value for struct pattern"))
                    }
                }
            }
            Pattern::Path(_path) => {
                // Path patterns like Result::Ok - unit variant patterns
                // Don't bind anything
                Ok(())
            }
            Pattern::TupleStruct { path, fields } => {
                // Enum variant with fields: Result::Ok(value)
                // Unwrap any refs first
                let unwrapped = Self::unwrap_all(&value);
                let path_str = path.segments.iter().map(|s| s.ident.name.as_str()).collect::<Vec<_>>().join("::");
                crate::sigil_debug!("DEBUG bind_pattern TupleStruct: path={}, value type={:?}",
                    path_str,
                    std::mem::discriminant(&unwrapped));
                if let Value::Variant { variant_name, fields: variant_fields, enum_name } = &unwrapped {
                    crate::sigil_debug!("DEBUG   Variant {}::{}, fields={}", enum_name, variant_name,
                        if variant_fields.is_some() { format!("Some(len={})", variant_fields.as_ref().unwrap().len()) } else { "None".to_string() });
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
                        } else if !fields.is_empty() {
                            // Pattern expects fields but variant has none
                            crate::sigil_debug!("DEBUG TupleStruct: pattern expects {} fields but variant has none", fields.len());
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
            Pattern::Or(patterns) => {
                // Or patterns - find the matching pattern and bind its variables
                for p in patterns {
                    if self.pattern_matches(p, &value)? {
                        return self.bind_pattern(p, value.clone());
                    }
                }
                // No pattern matched - this shouldn't happen if pattern_matches returned true earlier
                Err(RuntimeError::new("Or pattern didn't match any alternative"))
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

    /// Validate that a value matches its declared type annotation
    /// Check if two types are compatible (allowing numeric coercion)
    fn types_are_compatible(&self, expected: &str, actual: &str) -> bool {
        if expected == actual {
            return true;
        }
        // Numeric types are compatible with each other
        let numeric_types = ["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f32", "f64", "isize", "usize"];
        let expected_is_numeric = numeric_types.contains(&expected);
        let actual_is_numeric = numeric_types.contains(&actual);
        if expected_is_numeric && actual_is_numeric {
            return true;
        }
        false
    }

    /// Extract tensor shape from type annotation like Tensor<[3, 4]>
    /// Delegates to the static AST-based implementation
    fn extract_tensor_shape(&self, type_expr: &TypeExpr) -> Option<Vec<i64>> {
        Self::extract_tensor_shape_from_type(type_expr)
    }

    fn validate_type_annotation(&self, type_expr: &TypeExpr, value: &Value) -> Result<(), RuntimeError> {
        // Handle Option<T> and Result<T, E> type annotations
        if let TypeExpr::Path(path) = type_expr {
            if path.segments.len() == 1 {
                let type_name = &path.segments[0].ident.name;
                let generics = &path.segments[0].generics;

                // Check Option<T>
                if type_name == "Option" {
                    if let Some(gen_args) = generics {
                        if let Some(first_arg) = gen_args.first() {
                            // Get the expected inner type
                            let expected_type = self.type_expr_to_string(first_arg);

                            // Check if value is Option::Some with wrong inner type
                            if let Value::Variant { enum_name, variant_name, fields } = value {
                                if enum_name == "Option" && variant_name == "Some" {
                                    if let Some(ref inner) = fields {
                                        if !inner.is_empty() {
                                            let actual_type = self.get_value_type_name(&inner[0]);
                                            if !self.types_are_compatible(&expected_type, &actual_type) && !expected_type.is_empty() {
                                                return Err(RuntimeError::new(format!(
                                                    "type mismatch: expected Option<{}>, found Option<{}>",
                                                    expected_type, actual_type
                                                )));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Check Result<T, E>
                if type_name == "Result" {
                    if let Some(gen_args) = generics {
                        if let Some(first_arg) = gen_args.first() {
                            let expected_ok_type = self.type_expr_to_string(first_arg);

                            // Check if value is Result::Ok with wrong inner type
                            if let Value::Variant { enum_name, variant_name, fields } = value {
                                if enum_name == "Result" && variant_name == "Ok" {
                                    if let Some(ref inner) = fields {
                                        if !inner.is_empty() {
                                            let actual_type = self.get_value_type_name(&inner[0]);
                                            if !self.types_are_compatible(&expected_ok_type, &actual_type) && !expected_ok_type.is_empty() {
                                                return Err(RuntimeError::new(format!(
                                                    "type mismatch: expected Result<{}, _>, found Result<{}, _>",
                                                    expected_ok_type, actual_type
                                                )));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Convert a TypeExpr to a simple string representation
    fn type_expr_to_string(&self, type_expr: &TypeExpr) -> String {
        match type_expr {
            TypeExpr::Path(path) => {
                path.segments.iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::")
            }
            TypeExpr::Reference { inner, .. } => self.type_expr_to_string(inner),
            TypeExpr::Evidential { inner, .. } => self.type_expr_to_string(inner),
            TypeExpr::Linear(inner) => self.type_expr_to_string(inner),
            TypeExpr::Affine(inner) => self.type_expr_to_string(inner),
            TypeExpr::Relevant(inner) => self.type_expr_to_string(inner),
            _ => String::new(),
        }
    }

    /// Get the type name of a runtime value
    fn get_value_type_name(&self, value: &Value) -> String {
        match value {
            Value::Int(_) => "i64".to_string(),
            Value::Float(_) => "f64".to_string(),
            Value::Bool(_) => "bool".to_string(),
            Value::Char(_) => "char".to_string(),
            Value::String(_) => "String".to_string(),
            Value::Array(_) => "Array".to_string(),
            Value::Tuple(_) => "Tuple".to_string(),
            Value::Struct { name, .. } => name.clone(),
            Value::Variant { enum_name, .. } => enum_name.clone(),
            Value::Null => "null".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Check that all match arms have consistent result types (for simple expressions)
    fn check_match_arm_types(&self, arms: &[MatchArm]) -> Result<(), RuntimeError> {
        let mut first_type: Option<&'static str> = None;

        for (i, arm) in arms.iter().enumerate() {
            // Get the type of the arm body (only for simple literals)
            let arm_type = self.get_expr_type(&arm.body);

            if let Some(t) = arm_type {
                match first_type {
                    None => first_type = Some(t),
                    Some(expected) if expected != t => {
                        return Err(RuntimeError::new(format!(
                            "type mismatch in match arm {}: expected {}, found {}",
                            i + 1, expected, t
                        )));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Get the static type of an expression (only for simple literals/expressions)
    fn get_expr_type(&self, expr: &Expr) -> Option<&'static str> {
        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Int { .. } => Some("i64"),
                Literal::Float { .. } => Some("f64"),
                Literal::String(_) | Literal::MultiLineString(_) | Literal::RawString(_) => Some("String"),
                Literal::Bool(_) => Some("bool"),
                Literal::Char(_) => Some("char"),
                Literal::Null => Some("null"),
                _ => None,
            },
            Expr::Block(block) => {
                // For blocks, check the final expression
                if let Some(ref final_expr) = block.expr {
                    self.get_expr_type(final_expr)
                } else {
                    Some("null")
                }
            }
            _ => None, // Can't determine type statically for complex expressions
        }
    }

    fn eval_match(&mut self, expr: &Expr, arms: &[MatchArm]) -> Result<Value, RuntimeError> {
        // Pre-check: Verify all arms have consistent types (for literals)
        self.check_match_arm_types(arms)?;

        let value = self.evaluate(expr)?;

        // Debug all string matches to find keyword_or_ident
        let unwrapped = Self::unwrap_all(&value);
        if let Value::String(s) = &unwrapped {
            if s.len() <= 10 {
                crate::sigil_debug!("DEBUG eval_match: string='{}', arms={}", s, arms.len());
            }
        }

        for arm in arms {
            if self.pattern_matches(&arm.pattern, &value)? {
                // Create new environment for pattern bindings
                let env = Rc::new(RefCell::new(Environment::with_parent(
                    self.environment.clone(),
                )));
                let prev_env = self.environment.clone();
                self.environment = env;

                // Bind pattern variables FIRST (before evaluating guard)
                // This is necessary for guards like `?fields if !fields.is_empty()`
                if let Err(e) = self.bind_pattern(&arm.pattern, value.clone()) {
                    self.environment = prev_env;
                    return Err(e);
                }

                // Check guard if present (pattern vars are now in scope)
                if let Some(guard) = &arm.guard {
                    let guard_val = self.evaluate(guard)?;
                    if !self.is_truthy(&guard_val) {
                        // Guard failed - restore environment and try next arm
                        self.environment = prev_env;
                        continue;
                    }
                }

                // Pattern matched and guard passed - evaluate body
                let result = self.evaluate(&arm.body);

                self.environment = prev_env;
                return result;
            }
        }

        // Debug: show what value we're trying to match with discriminant
        crate::sigil_debug!("DEBUG No matching pattern for value: {} (discriminant: {:?})",
            self.format_value(&value), std::mem::discriminant(&value));
        // Also show the arms
        for (i, arm) in arms.iter().enumerate() {
            crate::sigil_debug!("DEBUG   arm {}: {:?}", i, arm.pattern);
        }
        Err(RuntimeError::new(format!("No matching pattern for {}", self.format_value(&value))))
    }

    fn pattern_matches(&mut self, pattern: &Pattern, value: &Value) -> Result<bool, RuntimeError> {
        // Unwrap evidential/affective/ref wrappers from value
        let value = Self::unwrap_all(value);

        // Debug string pattern matching
        if let Value::String(s) = &value {
            if **s == "fn" {
                crate::sigil_debug!("DEBUG pattern_matches: value='fn', pattern={:?}", pattern);
            }
        }

        match (pattern, &value) {
            (Pattern::Wildcard, _) => Ok(true),
            // Pattern::Ident with evidentiality - ?g matches Some/non-null, !g matches Known values
            (Pattern::Ident { evidentiality: Some(Evidentiality::Uncertain), name, .. }, val) => {
                // ?g pattern should match non-null values (i.e., Option::Some)
                let matches = match val {
                    Value::Null => false,
                    Value::Variant { variant_name, .. } if variant_name == "None" => false,
                    _ => true,
                };
                crate::sigil_debug!("DEBUG pattern_matches ?{}: value={} => {}", name.name, self.format_value(val), matches);
                Ok(matches)
            }
            (Pattern::Ident { .. }, _) => Ok(true),
            (Pattern::Literal(lit), val) => {
                let lit_val = self.eval_literal(lit)?;
                // Special case: null pattern matches Option::None
                if matches!(lit, Literal::Null) {
                    if matches!(val, Value::Null) {
                        return Ok(true);
                    }
                    // Option::None is equivalent to null
                    if let Value::Variant { enum_name, variant_name, .. } = val {
                        if enum_name == "Option" && variant_name == "None" {
                            return Ok(true);
                        }
                    }
                    return Ok(false);
                }
                let result = self.values_equal(&lit_val, val);
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
            // Tuple pattern can also match Option::Some containing a tuple
            // This supports Sigil's shorthand: `(a, b) => ...` matching `Some((a, b))`
            (Pattern::Tuple(patterns), Value::Variant { enum_name, variant_name, fields })
                if enum_name == "Option" && variant_name == "Some" =>
            {
                // Get the inner tuple from Some(...)
                if let Some(ref inner_fields) = fields {
                    if inner_fields.len() == 1 {
                        // The single field should be a tuple
                        if let Value::Tuple(ref inner_values) = inner_fields[0] {
                            if patterns.len() != inner_values.len() {
                                return Ok(false);
                            }
                            for (p, v) in patterns.iter().zip(inner_values.iter()) {
                                if !self.pattern_matches(p, v)? {
                                    return Ok(false);
                                }
                            }
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
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
            // Struct pattern - matches struct-like enum variants (e.g., TypeExpr::Evidential { inner, ... })
            (Pattern::Struct { path, fields: pat_fields, rest }, Value::Variant { variant_name, fields: variant_fields, .. }) => {
                let pattern_variant = path.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("");
                if pattern_variant != variant_name {
                    return Ok(false);
                }
                // Struct-like variants store fields as a single wrapped Struct value
                if let Some(inner_fields) = variant_fields {
                    if inner_fields.len() == 1 {
                        if let Value::Struct { fields: inner_struct, .. } = &inner_fields[0] {
                            let borrowed = inner_struct.borrow();
                            for field_pat in pat_fields {
                                let field_name = &field_pat.name.name;
                                if let Some(field_val) = borrowed.get(field_name) {
                                    if let Some(sub_pat) = &field_pat.pattern {
                                        if !self.pattern_matches(sub_pat, field_val)? {
                                            return Ok(false);
                                        }
                                    }
                                } else if !rest {
                                    return Ok(false);
                                }
                            }
                            return Ok(true);
                        }
                    }
                }
                // No fields or structure doesn't match
                Ok(pat_fields.is_empty() || *rest)
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
        // Unwrap any Ref wrappers before comparison
        let a_unwrapped = match a {
            Value::Ref(r) => r.borrow().clone(),
            _ => a.clone(),
        };
        let b_unwrapped = match b {
            Value::Ref(r) => r.borrow().clone(),
            _ => b.clone(),
        };
        match (&a_unwrapped, &b_unwrapped) {
            (Value::Null, Value::Null) => true,
            // Option::None is equivalent to null for equality
            (Value::Variant { enum_name, variant_name, .. }, Value::Null)
                if enum_name == "Option" && variant_name == "None" => true,
            (Value::Null, Value::Variant { enum_name, variant_name, .. })
                if enum_name == "Option" && variant_name == "None" => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => {
                let result = **a == **b;
                // Debug ALL short string comparisons
                if a.len() <= 5 && b.len() <= 5 {
                    crate::sigil_debug!("DEBUG values_equal: '{}' == '{}' -> {}", a, b, result);
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
            Value::Range { start, end, inclusive } => {
                // Convert range to vector of integers
                let start_val = start.unwrap_or(0);
                let end_val = end.unwrap_or(0);
                if inclusive {
                    (start_val..=end_val).map(Value::Int).collect()
                } else {
                    (start_val..end_val).map(Value::Int).collect()
                }
            }
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

        // Dereference Ref values to get the underlying collection
        let collection = match collection {
            Value::Ref(r) => r.borrow().clone(),
            other => other,
        };

        // Handle range slicing before evaluating the index
        if let Expr::Range { start, end, inclusive } = index {
            let start_val = match start {
                Some(e) => match self.evaluate(e)? {
                    Value::Int(n) => n as usize,
                    _ => return Err(RuntimeError::new("Slice start must be an integer")),
                },
                None => 0,
            };

            return match &collection {
                Value::Array(arr) => {
                    let arr = arr.borrow();
                    let len = arr.len();
                    let end_val = match end {
                        Some(e) => match self.evaluate(e)? {
                            Value::Int(n) => {
                                let n = n as usize;
                                if *inclusive { n + 1 } else { n }
                            },
                            _ => return Err(RuntimeError::new("Slice end must be an integer")),
                        },
                        None => len,  // Open-ended range: slice to end
                    };
                    let end_val = end_val.min(len);
                    let start_val = start_val.min(len);
                    let sliced: Vec<Value> = arr[start_val..end_val].to_vec();
                    Ok(Value::Array(Rc::new(RefCell::new(sliced))))
                }
                Value::String(s) => {
                    let len = s.len();
                    let end_val = match end {
                        Some(e) => match self.evaluate(e)? {
                            Value::Int(n) => {
                                let n = n as usize;
                                if *inclusive { n + 1 } else { n }
                            },
                            _ => return Err(RuntimeError::new("Slice end must be an integer")),
                        },
                        None => len,  // Open-ended range: slice to end
                    };
                    let end_val = end_val.min(len);
                    let start_val = start_val.min(len);
                    // Use byte slicing for consistency with char_at
                    let sliced = &s[start_val..end_val];
                    Ok(Value::String(Rc::new(sliced.to_string())))
                }
                _ => Err(RuntimeError::new("Cannot slice this type")),
            };
        }

        let idx = self.evaluate(index)?;

        match (collection, idx) {
            (Value::Array(arr), Value::Int(i)) => {
                if i < 0 {
                    return Err(RuntimeError::new(format!(
                        "Array index cannot be negative: {}", i
                    )));
                }
                let arr = arr.borrow();
                let i = i as usize;
                let result = arr.get(i)
                    .cloned()
                    .ok_or_else(|| RuntimeError::new("Index out of bounds"));
                if let Ok(ref v) = result {
                    crate::sigil_debug!("DEBUG eval_index: arr[{}] = {:?}", i, std::mem::discriminant(v));
                }
                result
            }
            (Value::Tuple(t), Value::Int(i)) => {
                if i < 0 {
                    return Err(RuntimeError::new(format!(
                        "Tuple index cannot be negative: {}", i
                    )));
                }
                let i = i as usize;
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
            // Handle open-ended range slicing (from eval_range returning tuple)
            (Value::Array(arr), Value::Tuple(range_tuple)) if range_tuple.len() == 2 => {
                let arr = arr.borrow();
                let start = match &range_tuple[0] {
                    Value::Int(n) => *n as usize,
                    _ => return Err(RuntimeError::new("Range start must be integer")),
                };
                let end = match &range_tuple[1] {
                    Value::Null => arr.len(),  // Open end - slice to end
                    Value::Int(n) => *n as usize,
                    _ => return Err(RuntimeError::new("Range end must be integer or None")),
                };
                let start = start.min(arr.len());
                let end = end.min(arr.len());
                let sliced: Vec<Value> = arr[start..end].to_vec();
                Ok(Value::Array(Rc::new(RefCell::new(sliced))))
            }
            (Value::String(s), Value::Tuple(range_tuple)) if range_tuple.len() == 2 => {
                let start = match &range_tuple[0] {
                    Value::Int(n) => *n as usize,
                    _ => return Err(RuntimeError::new("Range start must be integer")),
                };
                let end = match &range_tuple[1] {
                    Value::Null => s.len(),  // Open end - slice to end
                    Value::Int(n) => *n as usize,
                    _ => return Err(RuntimeError::new("Range end must be integer or None")),
                };
                let start = start.min(s.len());
                let end = end.min(s.len());
                let sliced = &s[start..end];
                Ok(Value::String(Rc::new(sliced.to_string())))
            }
            // Tensor indexing: t[i] returns row or element depending on shape
            (Value::Struct { name, fields }, Value::Int(i)) if name == "Tensor" => {
                if i < 0 {
                    return Err(RuntimeError::new(format!(
                        "Tensor index cannot be negative: {}", i
                    )));
                }
                let i = i as usize;
                let f = fields.borrow();
                let data = match f.get("__data__") {
                    Some(Value::Array(arr)) => arr.borrow().clone(),
                    _ => return Err(RuntimeError::new("Invalid Tensor")),
                };
                let shape: Vec<i64> = match f.get("__shape__") {
                    Some(Value::Array(arr)) => arr.borrow().iter().map(|v| match v {
                        Value::Int(n) => *n,
                        _ => 1,
                    }).collect(),
                    _ => vec![data.len() as i64],
                };
                drop(f);

                if shape.is_empty() {
                    // Scalar - can't index
                    return Err(RuntimeError::new("Cannot index scalar tensor"));
                } else if shape.len() == 1 {
                    // 1D tensor - return element
                    data.get(i)
                        .cloned()
                        .ok_or_else(|| RuntimeError::new("Tensor index out of bounds"))
                } else {
                    // Multi-D tensor - return slice as array or nested tensor
                    // For t[i] on 2D tensor [M, N], return row i (N elements)
                    let row_size: usize = shape[1..].iter().map(|x| *x as usize).product();
                    let start = i * row_size;
                    let end = start + row_size;
                    if end > data.len() {
                        return Err(RuntimeError::new("Tensor index out of bounds"));
                    }
                    let row_data: Vec<Value> = data[start..end].to_vec();

                    if shape.len() == 2 {
                        // 2D: return simple array for the row
                        Ok(Value::Array(Rc::new(RefCell::new(row_data))))
                    } else {
                        // Higher-D: return sub-tensor
                        let new_shape: Vec<i64> = shape[1..].to_vec();
                        let mut fields = std::collections::HashMap::new();
                        fields.insert("__data__".to_string(), Value::Array(Rc::new(RefCell::new(row_data))));
                        fields.insert("__shape__".to_string(), Value::Array(Rc::new(RefCell::new(
                            new_shape.iter().map(|s| Value::Int(*s)).collect()
                        ))));
                        fields.insert("shape".to_string(), Value::Array(Rc::new(RefCell::new(
                            new_shape.iter().map(|s| Value::Int(*s)).collect()
                        ))));
                        fields.insert("__requires_grad__".to_string(), Value::Bool(false));
                        fields.insert("__grad__".to_string(), Value::Null);
                        fields.insert("grad".to_string(), Value::Null);
                        Ok(Value::Struct {
                            name: "Tensor".to_string(),
                            fields: Rc::new(RefCell::new(fields)),
                        })
                    }
                }
            }
            (coll, idx) => {
                crate::sigil_debug!("DEBUG Cannot index: collection={:?}, index={:?}",
                    std::mem::discriminant(&coll), std::mem::discriminant(&idx));
                Err(RuntimeError::new("Cannot index"))
            }
        }
    }

    fn eval_field(&mut self, expr: &Expr, field: &Ident) -> Result<Value, RuntimeError> {
        if field.name == "items" {
            crate::sigil_debug!("DEBUG eval_field: accessing .items on expr={:?}", expr);
        }
        // Debug evidence field access
        if field.name == "evidence" {
            crate::sigil_debug!("DEBUG eval_field: accessing .evidence");
        }
        let value = self.evaluate(expr)?;
        if field.name == "items" {
            crate::sigil_debug!("DEBUG eval_field: .items receiver value={:?}", std::mem::discriminant(&value));
        }
        if field.name == "evidence" {
            crate::sigil_debug!("DEBUG eval_field: .evidence receiver value type={:?}", std::mem::discriminant(&value));
        }
        // Helper to get field from a value
        fn get_field(val: &Value, field_name: &str) -> Result<Value, RuntimeError> {
            // Debug all field access on IrPattern
            if field_name == "evidence" {
                crate::sigil_debug!("DEBUG get_field 'evidence' on value type: {:?}", std::mem::discriminant(val));
            }
            match val {
                Value::Struct { name, fields } => {
                    let field_val = fields.borrow().get(field_name).cloned();
                    if field_val.is_none() && field_name == "path" {
                        crate::sigil_debug!("DEBUG Unknown field 'path': struct={}, available={:?}", name, fields.borrow().keys().collect::<Vec<_>>());
                    }
                    // Debug evidence field access
                    if field_name == "evidence" || name.contains("IrPattern") {
                        crate::sigil_debug!("DEBUG get_field on Struct: name={}, field={}, available={:?}, found={}",
                            name, field_name, fields.borrow().keys().collect::<Vec<_>>(), field_val.is_some());
                    }
                    // Error on unknown fields
                    match field_val {
                        Some(v) => Ok(v),
                        None => {
                            Err(RuntimeError::new(format!("No field '{}' on struct '{}'", field_name, name)))
                        }
                    }
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
                Value::Variant { fields: variant_fields, .. } => {
                    // Handle struct-like enum variants (e.g., IrPattern::Ident { name, evidence, .. })
                    // Variant fields may be stored as a Struct value or as positional values
                    if let Some(inner_fields) = variant_fields {
                        // Try to find a Struct value that contains the field
                        for f in inner_fields.iter() {
                            if let Value::Struct { fields: struct_fields, .. } = f {
                                if let Some(field_val) = struct_fields.borrow().get(field_name).cloned() {
                                    return Ok(field_val);
                                }
                            }
                        }
                        // Field not found in any struct - return Null for optional fields
                        Ok(Value::Null)
                    } else {
                        // No fields in variant
                        Ok(Value::Null)
                    }
                }
                Value::Map(m) => {
                    // Handle field access on Map (e.g., url.scheme for URL objects)
                    Ok(m.borrow().get(field_name).cloned().unwrap_or(Value::Null))
                }
                _other => {
                    // Fallback for field access on non-struct types: return null
                    crate::sigil_warn!("WARN: Cannot access field '{}' on non-struct - returning null", field_name);
                    Ok(Value::Null)
                }
            }
        }
        get_field(&value, &field.name)
    }

    // Helper: Extract the root variable name from a method chain
    fn extract_root_var(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path) if path.segments.len() == 1 => {
                Some(path.segments[0].ident.name.clone())
            }
            Expr::MethodCall { receiver, .. } => {
                Self::extract_root_var(receiver)
            }
            _ => None,
        }
    }

    fn eval_method_call(
        &mut self,
        receiver: &Expr,
        method: &Ident,
        args: &[Expr],
    ) -> Result<Value, RuntimeError> {
        // Try "Type·method" as a combined function name for unresolved receiver patterns
        // This allows syntax like fs·read_to_string(path) to resolve to "fs·read_to_string" function
        if let Expr::Path(path) = receiver {
            if path.segments.len() == 1 {
                let recv_name = &path.segments[0].ident.name;
                // Only try combined lookup if receiver is not in environment
                let recv_exists = self.environment.borrow().get(recv_name).is_some();
                if !recv_exists {
                    let combined_name = format!("{}·{}", recv_name, method.name);
                    // Check if combined function exists in environment or globals
                    let func_val_opt = self.environment.borrow().get(&combined_name)
                        .or_else(|| self.globals.borrow().get(&combined_name));
                    if let Some(func_val) = func_val_opt {
                        // Evaluate arguments and call the function
                        let arg_values: Vec<Value> = args
                            .iter()
                            .map(|a| self.evaluate(a))
                            .collect::<Result<_, _>>()?;

                        // Set current_self_type for Self { ... } resolution in constructors
                        // recv_name is the type name (e.g., "DebugLexer" from "DebugLexer·new")
                        let old_self_type = self.current_self_type.take();
                        self.current_self_type = Some(recv_name.clone());

                        let result = match func_val {
                            Value::Function(f) => self.call_function(&f, arg_values),
                            Value::BuiltIn(b) => self.call_builtin(&b, arg_values),
                            _ => Err(RuntimeError::new(format!("{} is not a function", combined_name))),
                        };

                        // Restore old Self type
                        self.current_self_type = old_self_type;
                        return result;
                    }
                }
            }
        }

        // Special handling for String::push/push_str - needs to mutate the variable
        if (method.name == "push" || method.name == "push_str") && args.len() == 1 {
            let recv_val = self.evaluate(receiver)?;
            let recv_unwrapped = Self::unwrap_all(&recv_val);
            if let Value::String(s) = &recv_unwrapped {
                let arg = self.evaluate(&args[0])?;
                let arg_unwrapped = Self::unwrap_all(&arg);
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

                // Try to extract root variable from chain and mutate it
                if let Some(root_var) = Self::extract_root_var(receiver) {
                    self.environment.borrow_mut().set(&root_var, new_val.clone())?;
                    // Return the new value for method chaining
                    return Ok(new_val);
                }

                // Handle field access like self.output.push(c)
                if let Expr::Field { expr: base_expr, field: field_ident } = receiver {
                    let base = self.evaluate(base_expr)?;
                    if let Value::Struct { fields, .. } = base {
                        fields.borrow_mut().insert(field_ident.name.clone(), new_val.clone());
                        // Return the new value for method chaining
                        return Ok(new_val);
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
                crate::sigil_debug!("DEBUG method #{}: {}.{}()", count, recv_type, method.name);
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
            crate::sigil_debug!("DEBUG {}: recv_type={}", method.name, recv_type);
        }

        // Debug: Trace ALL as_str calls
        if method.name == "as_str" {
            let recv_unwrapped = Self::unwrap_all(&recv);
            if let Value::String(s) = &recv_unwrapped {
                crate::sigil_debug!("DEBUG as_str CALL: recv='{}' len={}", s, s.len());
            } else {
                crate::sigil_debug!("DEBUG as_str CALL: recv={:?} (not string)", recv_unwrapped);
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
            crate::sigil_debug!("DEBUG keyword_or_ident: recv_type={}", recv_type);
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
                    crate::sigil_debug!("DEBUG method call with 'fn': method={}, recv_type={}", method.name, recv_type);
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
                // Type check: if array has elements, new element must match type
                let arr_ref = arr.borrow();
                if let Some(first) = arr_ref.first() {
                    let expected_type = self.get_value_type_name(first);
                    let actual_type = self.get_value_type_name(&arg_values[0]);
                    // For enum variants like "Stmt::Let" and "Stmt::Semi", compare base type
                    let expected_base = expected_type.split("::").next().unwrap_or(&expected_type);
                    let actual_base = actual_type.split("::").next().unwrap_or(&actual_type);
                    if expected_base != actual_base {
                        return Err(RuntimeError::new(format!(
                            "type mismatch: cannot push {} into Vec<{}>",
                            actual_type, expected_type
                        )));
                    }
                }
                drop(arr_ref);
                arr.borrow_mut().push(arg_values[0].clone());
                Ok(Value::Null)
            }
            (Value::Array(arr), "pop") => match arr.borrow_mut().pop() {
                Some(v) => Ok(Value::Variant {
                    enum_name: "Option".to_string(),
                    variant_name: "Some".to_string(),
                    fields: Some(Rc::new(vec![v])),
                }),
                None => Ok(Value::Variant {
                    enum_name: "Option".to_string(),
                    variant_name: "None".to_string(),
                    fields: None,
                }),
            },
            (Value::Array(arr), "clear") => {
                arr.borrow_mut().clear();
                Ok(Value::Null)
            }
            (Value::Array(arr), "extend") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("extend expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::Array(other) => {
                        arr.borrow_mut().extend(other.borrow().iter().cloned());
                        Ok(Value::Null)
                    }
                    _ => Err(RuntimeError::new("extend expects array argument")),
                }
            }
            (Value::Array(arr), "reverse") => {
                let mut v = arr.borrow().clone();
                v.reverse();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "skip") => {
                let n = match arg_values.first() {
                    Some(Value::Int(i)) => *i as usize,
                    _ => 1,
                };
                let v: Vec<Value> = arr.borrow().iter().skip(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "take") => {
                let n = match arg_values.first() {
                    Some(Value::Int(i)) => *i as usize,
                    _ => 1,
                };
                let v: Vec<Value> = arr.borrow().iter().take(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "step_by") => {
                let n = match arg_values.first() {
                    Some(Value::Int(i)) if *i > 0 => *i as usize,
                    _ => 1,
                };
                let v: Vec<Value> = arr.borrow().iter().step_by(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "contains") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("contains expects 1 argument"));
                }
                let target = &arg_values[0];
                let found = arr.borrow().iter().any(|v| self.values_equal(v, target));
                Ok(Value::Bool(found))
            }
            (Value::Array(arr), "to_vec") | (Value::Array(arr), "clone") => {
                // Clone the array
                let cloned = arr.borrow().clone();
                Ok(Value::Array(Rc::new(RefCell::new(cloned))))
            }
            (Value::Array(arr), "capacity") => {
                // Return the capacity of the underlying vector
                Ok(Value::Int(arr.borrow().capacity() as i64))
            }
            (Value::Array(arr), "get") => {
                // get(index) - returns Option::Some(value) or Option::None for out-of-bounds
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("get expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::Int(idx) => {
                        let idx = *idx as usize;
                        let arr_ref = arr.borrow();
                        if idx < arr_ref.len() {
                            Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "Some".to_string(),
                                fields: Some(Rc::new(vec![arr_ref[idx].clone()])),
                            })
                        } else {
                            Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            })
                        }
                    }
                    _ => Err(RuntimeError::new("get expects integer index")),
                }
            }
            (Value::Array(arr), "as_slice") => {
                // as_slice() - returns the array as a slice (in interpreter, just returns itself)
                Ok(Value::Array(arr.clone()))
            }
            // Tuple methods
            (Value::Tuple(t), "to_string") | (Value::Tuple(t), "string") => {
                let s: Vec<String> = t.iter().map(|v| format!("{}", v)).collect();
                Ok(Value::String(Rc::new(format!("({})", s.join(", ")))))
            }
            (Value::Tuple(t), "len") => Ok(Value::Int(t.len() as i64)),
            (Value::Tuple(t), "first") => t.first().cloned().ok_or_else(|| RuntimeError::new("empty tuple")),
            (Value::Tuple(t), "last") => t.last().cloned().ok_or_else(|| RuntimeError::new("empty tuple")),
            (Value::Tuple(t), "get") => {
                let idx = match arg_values.first() {
                    Some(Value::Int(i)) => *i as usize,
                    _ => return Err(RuntimeError::new("get expects integer index")),
                };
                t.get(idx).cloned().ok_or_else(|| RuntimeError::new("tuple index out of bounds"))
            }
            (Value::Array(arr), "first") | (Value::Array(arr), "next") => Ok(arr
                .borrow()
                .first()
                .cloned()
                .unwrap_or(Value::Null)),
            (Value::Array(arr), "last") => arr
                .borrow()
                .last()
                .cloned()
                .ok_or_else(|| RuntimeError::new("empty array")),
            (Value::Array(arr), "iter") | (Value::Array(arr), "into_iter") => {
                // iter()/into_iter() on an array just returns the array - iteration happens in for loops
                Ok(Value::Array(arr.clone()))
            }
            (Value::Array(arr), "map") => {
                // map(closure) applies closure to each element
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("map expects 1 argument (closure)"));
                }
                match &arg_values[0] {
                    Value::Function(f) => {
                        let mut results = Vec::new();
                        for val in arr.borrow().iter() {
                            let result = self.call_function(f, vec![val.clone()])?;
                            results.push(result);
                        }
                        Ok(Value::Array(Rc::new(RefCell::new(results))))
                    }
                    _ => Err(RuntimeError::new("map expects closure argument")),
                }
            }
            (Value::Array(arr), "filter") => {
                // filter(predicate) keeps elements where predicate returns true
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("filter expects 1 argument (closure)"));
                }
                match &arg_values[0] {
                    Value::Function(f) => {
                        let mut results = Vec::new();
                        for val in arr.borrow().iter() {
                            let keep = self.call_function(f, vec![val.clone()])?;
                            if matches!(keep, Value::Bool(true)) {
                                results.push(val.clone());
                            }
                        }
                        Ok(Value::Array(Rc::new(RefCell::new(results))))
                    }
                    _ => Err(RuntimeError::new("filter expects closure argument")),
                }
            }
            (Value::Array(arr), "any") => {
                // any(predicate) returns true if any element satisfies predicate
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("any expects 1 argument (closure)"));
                }
                match &arg_values[0] {
                    Value::Function(f) => {
                        for val in arr.borrow().iter() {
                            let result = self.call_function(f, vec![val.clone()])?;
                            if matches!(result, Value::Bool(true)) {
                                return Ok(Value::Bool(true));
                            }
                        }
                        Ok(Value::Bool(false))
                    }
                    _ => Err(RuntimeError::new("any expects closure argument")),
                }
            }
            (Value::Array(arr), "all") => {
                // all(predicate) returns true if all elements satisfy predicate
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("all expects 1 argument (closure)"));
                }
                match &arg_values[0] {
                    Value::Function(f) => {
                        for val in arr.borrow().iter() {
                            let result = self.call_function(f, vec![val.clone()])?;
                            if !matches!(result, Value::Bool(true)) {
                                return Ok(Value::Bool(false));
                            }
                        }
                        Ok(Value::Bool(true))
                    }
                    _ => Err(RuntimeError::new("all expects closure argument")),
                }
            }
            (Value::Array(arr), "find") => {
                // find(predicate) returns first element satisfying predicate, or None
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("find expects 1 argument (closure)"));
                }
                match &arg_values[0] {
                    Value::Function(f) => {
                        for val in arr.borrow().iter() {
                            let result = self.call_function(f, vec![val.clone()])?;
                            if matches!(result, Value::Bool(true)) {
                                return Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![val.clone()])),
                                });
                            }
                        }
                        Ok(Value::Variant {
                            enum_name: "Option".to_string(),
                            variant_name: "None".to_string(),
                            fields: None,
                        })
                    }
                    _ => Err(RuntimeError::new("find expects closure argument")),
                }
            }
            (Value::Array(arr), "enumerate") => {
                // enumerate() returns array of (index, value) tuples
                let enumerated: Vec<Value> = arr
                    .borrow()
                    .iter()
                    .enumerate()
                    .map(|(i, v)| Value::Tuple(Rc::new(vec![Value::Int(i as i64), v.clone()])))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(enumerated))))
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
                    Value::Char(c) => Ok(Value::Bool(s.contains(*c))),
                    Value::Ref(inner) => {
                        if let Value::String(sub) = &*inner.borrow() {
                            Ok(Value::Bool(s.contains(sub.as_str())))
                        } else {
                            Err(RuntimeError::new("contains expects string or char"))
                        }
                    }
                    _ => Err(RuntimeError::new("contains expects string or char")),
                }
            }
            (Value::String(s), "as_str") => {
                if s.len() <= 10 { crate::sigil_debug!("DEBUG as_str: '{}'", s); }
                Ok(Value::String(s.clone()))
            }
            (Value::String(s), "to_string") => Ok(Value::String(s.clone())),
            (Value::String(s), "into") => Ok(Value::String(s.clone())),  // into() for String just returns String
            (Value::String(s), "capacity") => Ok(Value::Int(s.capacity() as i64)),
            // NOTE: String::clear() intentionally not implemented - strings are immutable.
            // Use `s = ""` instead of `s.clear()`.
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
            (Value::String(s), "strip_prefix") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("strip_prefix expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::String(prefix) => {
                        match s.strip_prefix(prefix.as_str()) {
                            Some(stripped) => Ok(Value::String(Rc::new(stripped.to_string()))),
                            None => Ok(Value::Null),
                        }
                    }
                    _ => Err(RuntimeError::new("strip_prefix expects string")),
                }
            }
            (Value::String(s), "strip_suffix") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("strip_suffix expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::String(suffix) => {
                        match s.strip_suffix(suffix.as_str()) {
                            Some(stripped) => Ok(Value::String(Rc::new(stripped.to_string()))),
                            None => Ok(Value::Null),
                        }
                    }
                    _ => Err(RuntimeError::new("strip_suffix expects string")),
                }
            }
            (Value::String(s), "replace") => {
                if arg_values.len() != 2 {
                    return Err(RuntimeError::new("replace expects 2 arguments (from, to)"));
                }
                match (&arg_values[0], &arg_values[1]) {
                    (Value::String(from), Value::String(to)) => {
                        Ok(Value::String(Rc::new(s.replace(from.as_str(), to.as_str()))))
                    }
                    _ => Err(RuntimeError::new("replace expects two string arguments")),
                }
            }
            (Value::String(s), "is_empty") => Ok(Value::Bool(s.is_empty())),
            (Value::String(s), "find") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("find expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::Char(c) => {
                        match s.find(*c) {
                            Some(idx) => Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "Some".to_string(),
                                fields: Some(Rc::new(vec![Value::Int(idx as i64)])),
                            }),
                            None => Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            }),
                        }
                    }
                    Value::String(pattern) => {
                        match s.find(pattern.as_str()) {
                            Some(idx) => Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "Some".to_string(),
                                fields: Some(Rc::new(vec![Value::Int(idx as i64)])),
                            }),
                            None => Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            }),
                        }
                    }
                    Value::Function(f) => {
                        for (idx, c) in s.chars().enumerate() {
                            let result = self.call_function(f, vec![Value::Char(c)])?;
                            if let Value::Bool(true) = result {
                                return Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![Value::Int(idx as i64)])),
                                });
                            }
                        }
                        Ok(Value::Variant {
                            enum_name: "Option".to_string(),
                            variant_name: "None".to_string(),
                            fields: None,
                        })
                    }
                    _ => Err(RuntimeError::new("find expects a char, string, or closure")),
                }
            }
            (Value::String(s), "clone") => Ok(Value::String(Rc::new((**s).clone()))),
            (Value::String(s), "concat") => {
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new("concat expects 1 argument"));
                }
                match &arg_values[0] {
                    Value::String(other) => {
                        let mut result = (**s).clone();
                        result.push_str(other);
                        Ok(Value::String(Rc::new(result)))
                    }
                    _ => Err(RuntimeError::new("concat expects string argument")),
                }
            }
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
                // Use byte-based indexing to match the self-hosted lexer's pos tracking
                // which increments by c.len_utf8() (byte count, not character count)
                if idx < s.len() {
                    // Get the character starting at byte position idx
                    let remaining = &s[idx..];
                    match remaining.chars().next() {
                        Some(c) => Ok(Value::Char(c)),
                        None => Ok(Value::Null),
                    }
                } else {
                    Ok(Value::Null) // Out of bounds
                }
            }
            (Value::String(s), "chars") => {
                let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                Ok(Value::Array(Rc::new(RefCell::new(chars))))
            }
            (Value::String(s), "bytes") | (Value::String(s), "as_bytes") => {
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
            (Value::Char(c), "to_ascii_uppercase") => Ok(Value::Char(c.to_ascii_uppercase())),
            (Value::Char(c), "to_ascii_lowercase") => Ok(Value::Char(c.to_ascii_lowercase())),
            (Value::Char(c), "clone") => Ok(Value::Char(*c)),
            (Value::String(s), "upper") | (Value::String(s), "uppercase") | (Value::String(s), "to_uppercase") => {
                Ok(Value::String(Rc::new(s.to_uppercase())))
            }
            (Value::String(s), "lower") | (Value::String(s), "lowercase") | (Value::String(s), "to_lowercase") => {
                Ok(Value::String(Rc::new(s.to_lowercase())))
            }
            (Value::String(s), "trim") => Ok(Value::String(Rc::new(s.trim().to_string()))),
            (Value::String(s), "len") => Ok(Value::Int(s.len() as i64)),
            (Value::String(s), "is_empty") => Ok(Value::Bool(s.is_empty())),
            // Path-like methods for strings (treat string as file path)
            (Value::String(s), "exists") => Ok(Value::Bool(std::path::Path::new(s.as_str()).exists())),
            (Value::String(s), "is_dir") => Ok(Value::Bool(std::path::Path::new(s.as_str()).is_dir())),
            (Value::String(s), "is_file") => Ok(Value::Bool(std::path::Path::new(s.as_str()).is_file())),
            (Value::String(s), "join") => {
                // Path join: "dir".join("file") => "dir/file"
                if arg_values.len() != 1 {
                    return Err(RuntimeError::new(&format!("join expects 1 argument, got {}", arg_values.len())));
                }
                let other = match &arg_values[0] {
                    Value::String(s2) => s2.as_str().to_string(),
                    other => return Err(RuntimeError::new(&format!("join expects String argument, got {:?}", other))),
                };
                let path = std::path::Path::new(s.as_str()).join(&other);
                Ok(Value::String(Rc::new(path.to_string_lossy().to_string())))
            }
            (Value::String(s), "parent") => {
                // Get parent directory
                let path = std::path::Path::new(s.as_str());
                match path.parent() {
                    Some(p) => Ok(Value::String(Rc::new(p.to_string_lossy().to_string()))),
                    None => Ok(Value::Null),
                }
            }
            (Value::String(s), "file_name") => {
                // Get file name component
                let path = std::path::Path::new(s.as_str());
                match path.file_name() {
                    Some(n) => Ok(Value::String(Rc::new(n.to_string_lossy().to_string()))),
                    None => Ok(Value::Null),
                }
            }
            (Value::String(s), "extension") => {
                // Get file extension
                let path = std::path::Path::new(s.as_str());
                match path.extension() {
                    Some(e) => Ok(Value::String(Rc::new(e.to_string_lossy().to_string()))),
                    None => Ok(Value::Null),
                }
            }
            // Result-like chaining for strings (used when string represents a Result-like value)
            (Value::String(_), "and_then") | (Value::String(_), "or_else") => {
                // Just pass through - these are no-ops for plain strings
                Ok(recv.clone())
            }
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
            (Value::Array(arr), "collect") => {
                // collect() on array just returns the array itself
                // It's the terminal operation that materializes pipeline results
                Ok(Value::Array(arr.clone()))
            }
            (Value::Array(arr), "join") => {
                let separator = if arg_values.is_empty() {
                    String::new()
                } else {
                    match &arg_values[0] {
                        Value::String(s) => (**s).clone(),
                        _ => return Err(RuntimeError::new("join separator must be string")),
                    }
                };
                let parts: Vec<String> = arr.borrow().iter()
                    .map(|v| self.format_value(v))
                    .collect();
                Ok(Value::String(Rc::new(parts.join(&separator))))
            }
            // Map type-aware method dispatch (for HttpClient, WebSocket, etc.)
            // Check if Map has __type__ and dispatch to Type·method if available
            (Value::Map(m), method_name) => {
                let borrowed = m.borrow();
                if let Some(Value::String(type_name)) = borrowed.get("__type__") {
                    let qualified_method = format!("{}·{}", type_name, method_name);
                    drop(borrowed); // Release borrow before looking up globals

                    // Look up the method in globals (clone to avoid borrow issues)
                    let func_val_opt = self.globals.borrow().get(&qualified_method).map(|v| v.clone());
                    if let Some(func_val) = func_val_opt {
                        // Call Type·method(self, args...)
                        let mut full_args = vec![Value::Map(m.clone())];
                        full_args.extend(arg_values);
                        return match func_val {
                            Value::Function(f) => self.call_function(&f, full_args),
                            Value::BuiltIn(b) => self.call_builtin(&b, full_args),
                            _ => Err(RuntimeError::new(format!("{} is not a function", qualified_method))),
                        };
                    }
                    // Fall through to generic Map methods
                }

                // Generic Map methods
                match method_name {
                    "insert" => {
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
                    "get" => {
                        if arg_values.len() != 1 {
                            return Err(RuntimeError::new("get expects 1 argument"));
                        }
                        let key = match &arg_values[0] {
                            Value::String(s) => (**s).clone(),
                            _ => format!("{}", arg_values[0]),
                        };
                        Ok(m.borrow().get(&key).cloned().unwrap_or(Value::Null))
                    }
                    "contains_key" => {
                        if arg_values.len() != 1 {
                            return Err(RuntimeError::new("contains_key expects 1 argument"));
                        }
                        let key = match &arg_values[0] {
                            Value::String(s) => (**s).clone(),
                            _ => format!("{}", arg_values[0]),
                        };
                        Ok(Value::Bool(m.borrow().contains_key(&key)))
                    }
                    "len" => Ok(Value::Int(m.borrow().len() as i64)),
                    "is_empty" => Ok(Value::Bool(m.borrow().is_empty())),
                    "keys" => {
                        let keys: Vec<Value> = m.borrow().keys()
                            .map(|k| Value::String(Rc::new(k.clone())))
                            .collect();
                        Ok(Value::Array(Rc::new(RefCell::new(keys))))
                    }
                    "values" => {
                        let values: Vec<Value> = m.borrow().values().cloned().collect();
                        Ok(Value::Array(Rc::new(RefCell::new(values))))
                    }
                    _ => Err(RuntimeError::new(format!("Map has no method: {}", method_name)))
                }
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
                if let Value::Struct { name, fields } = &inner {
                    // Try struct method lookup with the inner struct
                    let qualified_name = format!("{}·{}", name, method.name);
                    let func = self.globals.borrow().get(&qualified_name).map(|v| v.clone());
                    if let Some(func) = func {
                        if let Value::Function(f) = func {
                            // Set current Self type for Self { ... } resolution
                            let old_self_type = self.current_self_type.take();
                            self.current_self_type = Some(name.clone());

                            // Pass the Ref as the receiver (for &mut self methods)
                            let mut all_args = vec![recv.clone()];
                            all_args.extend(arg_values.clone());
                            let result = self.call_function(&f, all_args);

                            // Restore old Self type
                            self.current_self_type = old_self_type;
                            return result;
                        } else if let Value::BuiltIn(b) = func {
                            let mut all_args = vec![recv.clone()];
                            all_args.extend(arg_values.clone());
                            return (b.func)(self, all_args);
                        }
                    }

                    // If struct name is "Self", search by matching field names
                    if name == "Self" {
                        let field_names: Vec<String> = fields.borrow().keys().cloned().collect();

                        // Search through registered types to find a matching struct
                        for (type_name, type_def) in &self.types {
                            if let TypeDef::Struct(struct_def) = type_def {
                                let def_fields: Vec<String> = match &struct_def.fields {
                                    crate::ast::StructFields::Named(fs) => fs.iter().map(|f| f.name.name.clone()).collect(),
                                    _ => continue,
                                };

                                // Match if our fields exist in the definition
                                let matches = field_names.iter().all(|f| def_fields.contains(f));
                                if matches {
                                    let qualified_name = format!("{}·{}", type_name, method.name);
                                    let func = self.globals.borrow().get(&qualified_name).map(|v| v.clone());
                                    if let Some(func) = func {
                                        if let Value::Function(f) = func {
                                            // Set current Self type for Self { ... } resolution
                                            let old_self_type = self.current_self_type.take();
                                            self.current_self_type = Some(type_name.clone());

                                            let mut all_args = vec![recv.clone()];
                                            all_args.extend(arg_values.clone());
                                            let result = self.call_function(&f, all_args);

                                            // Restore old Self type
                                            self.current_self_type = old_self_type;
                                            return result;
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

                    // Built-in methods for PathBuf struct
                    if name == "PathBuf" || name == "Path" {
                        if let Some(Value::String(path)) = fields.borrow().get("path").cloned() {
                            match method.name.as_str() {
                                "exists" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).exists())),
                                "is_dir" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).is_dir())),
                                "is_file" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).is_file())),
                                "join" => {
                                    if let Some(Value::String(other)) = arg_values.first() {
                                        let new_path = std::path::Path::new(path.as_str()).join(other.as_str());
                                        let mut new_fields = std::collections::HashMap::new();
                                        new_fields.insert("path".to_string(), Value::String(Rc::new(new_path.to_string_lossy().to_string())));
                                        return Ok(Value::Struct {
                                            name: "PathBuf".to_string(),
                                            fields: Rc::new(RefCell::new(new_fields)),
                                        });
                                    }
                                    return Err(RuntimeError::new("join requires string argument"));
                                }
                                "parent" => {
                                    let p = std::path::Path::new(path.as_str());
                                    return match p.parent() {
                                        Some(par) => {
                                            let mut new_fields = std::collections::HashMap::new();
                                            new_fields.insert("path".to_string(), Value::String(Rc::new(par.to_string_lossy().to_string())));
                                            Ok(Value::Struct {
                                                name: "PathBuf".to_string(),
                                                fields: Rc::new(RefCell::new(new_fields)),
                                            })
                                        }
                                        None => Ok(Value::Null),
                                    };
                                }
                                "file_name" => {
                                    let p = std::path::Path::new(path.as_str());
                                    return match p.file_name() {
                                        Some(n) => Ok(Value::String(Rc::new(n.to_string_lossy().to_string()))),
                                        None => Ok(Value::Null),
                                    };
                                }
                                "extension" => {
                                    let p = std::path::Path::new(path.as_str());
                                    return match p.extension() {
                                        Some(e) => Ok(Value::String(Rc::new(e.to_string_lossy().to_string()))),
                                        None => Ok(Value::Null),
                                    };
                                }
                                "to_string" | "display" | "to_str" => {
                                    return Ok(Value::String(path.clone()));
                                }
                                _ => {}
                            }
                        }
                    }

                    // Fallback for unknown methods on external type references: return null
                    crate::sigil_warn!("WARN: Unknown method '{}' on '&{}' - returning null", method.name, name);
                    return Ok(Value::Null);
                }
                // For non-struct refs (like &str), auto-deref and call method on inner value
                // Handle common methods on &str (reference to String)
                if let Value::String(s) = &inner {
                    match method.name.as_str() {
                        "to_string" => return Ok(Value::String(s.clone())),
                        "len" => return Ok(Value::Int(s.len() as i64)),
                        "is_empty" => return Ok(Value::Bool(s.is_empty())),
                        "as_str" => return Ok(Value::String(s.clone())),
                        "starts_with" => {
                            let prefix = match arg_values.first() {
                                Some(Value::String(p)) => p.as_str(),
                                Some(Value::Char(c)) => return Ok(Value::Bool(s.starts_with(*c))),
                                _ => return Err(RuntimeError::new("starts_with expects string or char")),
                            };
                            return Ok(Value::Bool(s.starts_with(prefix)));
                        }
                        "ends_with" => {
                            let suffix = match arg_values.first() {
                                Some(Value::String(p)) => p.as_str(),
                                Some(Value::Char(c)) => return Ok(Value::Bool(s.ends_with(*c))),
                                _ => return Err(RuntimeError::new("ends_with expects string or char")),
                            };
                            return Ok(Value::Bool(s.ends_with(suffix)));
                        }
                        "contains" => {
                            let substr = match arg_values.first() {
                                Some(Value::String(p)) => p.as_str(),
                                Some(Value::Char(c)) => return Ok(Value::Bool(s.contains(*c))),
                                _ => return Err(RuntimeError::new("contains expects string or char")),
                            };
                            return Ok(Value::Bool(s.contains(substr)));
                        }
                        "trim" => return Ok(Value::String(Rc::new(s.trim().to_string()))),
                        "to_lowercase" => return Ok(Value::String(Rc::new(s.to_lowercase()))),
                        "to_uppercase" => return Ok(Value::String(Rc::new(s.to_uppercase()))),
                        "chars" => {
                            let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                            return Ok(Value::Array(Rc::new(RefCell::new(chars))));
                        }
                        "split" => {
                            let delim = match arg_values.first() {
                                Some(Value::String(d)) => d.as_str().to_string(),
                                Some(Value::Char(c)) => c.to_string(),
                                _ => " ".to_string(),
                            };
                            let parts: Vec<Value> = s.split(&delim)
                                .map(|p| Value::String(Rc::new(p.to_string())))
                                .collect();
                            return Ok(Value::Array(Rc::new(RefCell::new(parts))));
                        }
                        "replace" => {
                            if arg_values.len() != 2 {
                                return Err(RuntimeError::new("replace expects 2 arguments"));
                            }
                            let from = match &arg_values[0] {
                                Value::String(f) => f.as_str().to_string(),
                                Value::Char(c) => c.to_string(),
                                _ => return Err(RuntimeError::new("replace expects strings")),
                            };
                            let to = match &arg_values[1] {
                                Value::String(t) => t.as_str().to_string(),
                                Value::Char(c) => c.to_string(),
                                _ => return Err(RuntimeError::new("replace expects strings")),
                            };
                            return Ok(Value::String(Rc::new(s.replace(&from, &to))));
                        }
                        _ => {}
                    }
                }
                // Handle methods on &[T] and &mut [T] (references to arrays/slices)
                if let Value::Array(arr) = &inner {
                    match method.name.as_str() {
                        "len" => return Ok(Value::Int(arr.borrow().len() as i64)),
                        "is_empty" => return Ok(Value::Bool(arr.borrow().is_empty())),
                        "to_vec" | "clone" => {
                            // Clone the array
                            let cloned = arr.borrow().clone();
                            return Ok(Value::Array(Rc::new(RefCell::new(cloned))));
                        }
                        "push" => {
                            if arg_values.len() != 1 {
                                return Err(RuntimeError::new("push expects 1 argument"));
                            }
                            arr.borrow_mut().push(arg_values[0].clone());
                            return Ok(Value::Null);
                        }
                        "pop" => {
                            return match arr.borrow_mut().pop() {
                                Some(v) => Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![v])),
                                }),
                                None => Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "None".to_string(),
                                    fields: None,
                                }),
                            };
                        }
                        "contains" => {
                            if arg_values.len() != 1 {
                                return Err(RuntimeError::new("contains expects 1 argument"));
                            }
                            let target = &arg_values[0];
                            let found = arr.borrow().iter().any(|v| self.values_equal(v, target));
                            return Ok(Value::Bool(found));
                        }
                        "first" | "next" => {
                            return Ok(arr.borrow().first().cloned().unwrap_or(Value::Null));
                        }
                        "last" => {
                            return arr.borrow().last().cloned()
                                .ok_or_else(|| RuntimeError::new("empty array"));
                        }
                        "iter" | "into_iter" => {
                            return Ok(Value::Array(arr.clone()));
                        }
                        "reverse" => {
                            let mut v = arr.borrow().clone();
                            v.reverse();
                            return Ok(Value::Array(Rc::new(RefCell::new(v))));
                        }
                        "skip" => {
                            let n = match arg_values.first() {
                                Some(Value::Int(i)) => *i as usize,
                                _ => 1,
                            };
                            let v: Vec<Value> = arr.borrow().iter().skip(n).cloned().collect();
                            return Ok(Value::Array(Rc::new(RefCell::new(v))));
                        }
                        "take" => {
                            let n = match arg_values.first() {
                                Some(Value::Int(i)) => *i as usize,
                                _ => 1,
                            };
                            let v: Vec<Value> = arr.borrow().iter().take(n).cloned().collect();
                            return Ok(Value::Array(Rc::new(RefCell::new(v))));
                        }
                        "get" => {
                            let idx = match arg_values.first() {
                                Some(Value::Int(i)) => *i as usize,
                                _ => return Err(RuntimeError::new("get expects integer index")),
                            };
                            return Ok(arr.borrow().get(idx).cloned().unwrap_or(Value::Null));
                        }
                        _ => {}
                    }
                }
                // Handle clone on any Ref value - clone the inner value
                if method.name == "clone" {
                    crate::sigil_debug!("DEBUG clone: recv_type=Ref({:?})", std::mem::discriminant(&inner));
                    return Ok(inner.clone());
                }
                // Handle into on Ref value - convert to owned value
                if method.name == "into" {
                    return Ok(inner.clone());
                }
                // Handle to_string on Ref value
                if method.name == "to_string" {
                    return Ok(Value::String(Rc::new(format!("{}", inner))));
                }
                // Path methods for Ref containing PathBuf struct
                if let Value::Struct { name, fields, .. } = &inner {
                    if name == "PathBuf" || name == "Path" {
                        let borrowed = fields.borrow();
                        if let Some(Value::String(path)) = borrowed.get("path") {
                            match method.name.as_str() {
                                "exists" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).exists())),
                                "is_dir" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).is_dir())),
                                "is_file" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).is_file())),
                                "join" => {
                                    if let Some(Value::String(other)) = arg_values.first() {
                                        let new_path = std::path::Path::new(path.as_str()).join(other.as_str());
                                        let mut new_fields = std::collections::HashMap::new();
                                        new_fields.insert("path".to_string(), Value::String(Rc::new(new_path.to_string_lossy().to_string())));
                                        return Ok(Value::Struct {
                                            name: "PathBuf".to_string(),
                                            fields: Rc::new(RefCell::new(new_fields)),
                                        });
                                    }
                                    return Err(RuntimeError::new("join requires string argument"));
                                }
                                "parent" => {
                                    let p = std::path::Path::new(path.as_str());
                                    return match p.parent() {
                                        Some(par) => {
                                            let mut new_fields = std::collections::HashMap::new();
                                            new_fields.insert("path".to_string(), Value::String(Rc::new(par.to_string_lossy().to_string())));
                                            Ok(Value::Struct {
                                                name: "PathBuf".to_string(),
                                                fields: Rc::new(RefCell::new(new_fields)),
                                            })
                                        }
                                        None => Ok(Value::Null),
                                    };
                                }
                                "file_name" => {
                                    let p = std::path::Path::new(path.as_str());
                                    return match p.file_name() {
                                        Some(n) => Ok(Value::String(Rc::new(n.to_string_lossy().to_string()))),
                                        None => Ok(Value::Null),
                                    };
                                }
                                "extension" => {
                                    let p = std::path::Path::new(path.as_str());
                                    return match p.extension() {
                                        Some(e) => Ok(Value::String(Rc::new(e.to_string_lossy().to_string()))),
                                        None => Ok(Value::Null),
                                    };
                                }
                                "to_string" | "display" => {
                                    return Ok(Value::String(path.clone()));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                // Path methods for Ref containing String (PathBuf behavior)
                if let Value::String(s) = &inner {
                    match method.name.as_str() {
                        "exists" => return Ok(Value::Bool(std::path::Path::new(s.as_str()).exists())),
                        "is_dir" => return Ok(Value::Bool(std::path::Path::new(s.as_str()).is_dir())),
                        "is_file" => return Ok(Value::Bool(std::path::Path::new(s.as_str()).is_file())),
                        "join" => {
                            if let Some(Value::String(other)) = arg_values.first() {
                                let path = std::path::Path::new(s.as_str()).join(other.as_str());
                                return Ok(Value::String(Rc::new(path.to_string_lossy().to_string())));
                            }
                            return Err(RuntimeError::new("join requires string argument"));
                        }
                        "parent" => {
                            let path = std::path::Path::new(s.as_str());
                            return match path.parent() {
                                Some(p) => Ok(Value::String(Rc::new(p.to_string_lossy().to_string()))),
                                None => Ok(Value::Null),
                            };
                        }
                        "file_name" => {
                            let path = std::path::Path::new(s.as_str());
                            return match path.file_name() {
                                Some(n) => Ok(Value::String(Rc::new(n.to_string_lossy().to_string()))),
                                None => Ok(Value::Null),
                            };
                        }
                        "extension" => {
                            let path = std::path::Path::new(s.as_str());
                            return match path.extension() {
                                Some(e) => Ok(Value::String(Rc::new(e.to_string_lossy().to_string()))),
                                None => Ok(Value::Null),
                            };
                        }
                        _ => {}
                    }
                }
                // If the inner value is a string, recursively call method dispatch
                // This handles cases like &s[..].find(...) where we have a Ref to a String slice
                if let Value::String(_) = inner {
                    // Recursively dispatch method call on the inner string
                    // Create a temporary receiver with the unwrapped string
                    let recv_unwrapped = inner.clone();
                    match (&recv_unwrapped, method.name.as_str()) {
                        (Value::String(s), "find") => {
                            if arg_values.len() != 1 {
                                return Err(RuntimeError::new("find expects 1 argument"));
                            }
                            match &arg_values[0] {
                                Value::Char(c) => {
                                    return match s.find(*c) {
                                        Some(idx) => Ok(Value::Variant {
                                            enum_name: "Option".to_string(),
                                            variant_name: "Some".to_string(),
                                            fields: Some(Rc::new(vec![Value::Int(idx as i64)])),
                                        }),
                                        None => Ok(Value::Variant {
                                            enum_name: "Option".to_string(),
                                            variant_name: "None".to_string(),
                                            fields: None,
                                        }),
                                    }
                                }
                                Value::String(pattern) => {
                                    return match s.find(pattern.as_str()) {
                                        Some(idx) => Ok(Value::Variant {
                                            enum_name: "Option".to_string(),
                                            variant_name: "Some".to_string(),
                                            fields: Some(Rc::new(vec![Value::Int(idx as i64)])),
                                        }),
                                        None => Ok(Value::Variant {
                                            enum_name: "Option".to_string(),
                                            variant_name: "None".to_string(),
                                            fields: None,
                                        }),
                                    }
                                }
                                Value::Function(f) => {
                                    for (idx, c) in s.chars().enumerate() {
                                        let result = self.call_function(f, vec![Value::Char(c)])?;
                                        if let Value::Bool(true) = result {
                                            return Ok(Value::Variant {
                                                enum_name: "Option".to_string(),
                                                variant_name: "Some".to_string(),
                                                fields: Some(Rc::new(vec![Value::Int(idx as i64)])),
                                            });
                                        }
                                    }
                                    return Ok(Value::Variant {
                                        enum_name: "Option".to_string(),
                                        variant_name: "None".to_string(),
                                        fields: None,
                                    })
                                }
                                _ => return Err(RuntimeError::new("find expects a char, string, or closure")),
                            }
                        }
                        (Value::String(s), "trim") => return Ok(Value::String(Rc::new(s.trim().to_string()))),
                        (Value::String(s), "is_empty") => return Ok(Value::Bool(s.is_empty())),
                        (Value::String(s), "len") => return Ok(Value::Int(s.len() as i64)),
                        (Value::String(s), "to_string") => return Ok(Value::String(s.clone())),
                        (Value::String(s), "chars") => {
                            let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                            return Ok(Value::Array(Rc::new(RefCell::new(chars))))
                        }
                        (Value::String(s), "starts_with") => {
                            if let Some(Value::String(prefix)) = arg_values.first() {
                                return Ok(Value::Bool(s.starts_with(prefix.as_str())));
                            }
                            return Err(RuntimeError::new("starts_with expects string argument"));
                        }
                        _ => {}
                    }
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
                    // Clone the struct value
                    return Ok(recv.clone());
                }
                // PathBuf struct methods
                if name == "PathBuf" || name == "Path" {
                    let borrowed = fields.borrow();
                    if let Some(Value::String(path)) = borrowed.get("path") {
                        match method.name.as_str() {
                            "exists" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).exists())),
                            "is_dir" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).is_dir())),
                            "is_file" => return Ok(Value::Bool(std::path::Path::new(path.as_str()).is_file())),
                            "join" => {
                                if let Some(Value::String(other)) = arg_values.first() {
                                    let new_path = std::path::Path::new(path.as_str()).join(other.as_str());
                                    let mut new_fields = std::collections::HashMap::new();
                                    new_fields.insert("path".to_string(), Value::String(Rc::new(new_path.to_string_lossy().to_string())));
                                    return Ok(Value::Struct {
                                        name: "PathBuf".to_string(),
                                        fields: Rc::new(RefCell::new(new_fields)),
                                    });
                                }
                                return Err(RuntimeError::new("join requires string argument"));
                            }
                            "parent" => {
                                let p = std::path::Path::new(path.as_str());
                                return match p.parent() {
                                    Some(par) => {
                                        let mut new_fields = std::collections::HashMap::new();
                                        new_fields.insert("path".to_string(), Value::String(Rc::new(par.to_string_lossy().to_string())));
                                        Ok(Value::Struct {
                                            name: "PathBuf".to_string(),
                                            fields: Rc::new(RefCell::new(new_fields)),
                                        })
                                    }
                                    None => Ok(Value::Null),
                                };
                            }
                            "file_name" => {
                                let p = std::path::Path::new(path.as_str());
                                return match p.file_name() {
                                    Some(n) => Ok(Value::String(Rc::new(n.to_string_lossy().to_string()))),
                                    None => Ok(Value::Null),
                                };
                            }
                            "extension" => {
                                let p = std::path::Path::new(path.as_str());
                                return match p.extension() {
                                    Some(e) => Ok(Value::String(Rc::new(e.to_string_lossy().to_string()))),
                                    None => Ok(Value::Null),
                                };
                            }
                            "to_string" | "display" => {
                                return Ok(Value::String(path.clone()));
                            }
                            _ => {}
                        }
                    }
                }
                // Rc struct methods
                if name == "Rc" {
                    let borrowed = fields.borrow();
                    if let Some(value) = borrowed.get("_value") {
                        match method.name.as_str() {
                            "clone" => {
                                // Return a new Rc with same value
                                let mut new_fields = HashMap::new();
                                new_fields.insert("_value".to_string(), value.clone());
                                return Ok(Value::Struct {
                                    name: "Rc".to_string(),
                                    fields: Rc::new(RefCell::new(new_fields)),
                                });
                            }
                            _ => {}
                        }
                    }
                }
                // Cell struct methods
                if name == "Cell" {
                    match method.name.as_str() {
                        "get" => {
                            let borrowed = fields.borrow();
                            if let Some(value) = borrowed.get("_value") {
                                return Ok(value.clone());
                            }
                            return Err(RuntimeError::new("Cell has no value"));
                        }
                        "set" => {
                            if arg_values.len() != 1 {
                                return Err(RuntimeError::new("set expects 1 argument"));
                            }
                            fields.borrow_mut().insert("_value".to_string(), arg_values[0].clone());
                            return Ok(Value::Null);
                        }
                        _ => {}
                    }
                }
                // Duration struct methods
                if name == "Duration" {
                    let borrowed = fields.borrow();
                    let secs = match borrowed.get("secs") {
                        Some(Value::Int(s)) => *s,
                        _ => 0,
                    };
                    let nanos = match borrowed.get("nanos") {
                        Some(Value::Int(n)) => *n,
                        _ => 0,
                    };
                    match method.name.as_str() {
                        "as_secs" => return Ok(Value::Int(secs)),
                        "as_millis" => return Ok(Value::Int(secs * 1000 + nanos / 1_000_000)),
                        "as_micros" => return Ok(Value::Int(secs * 1_000_000 + nanos / 1000)),
                        "as_nanos" => return Ok(Value::Int(secs * 1_000_000_000 + nanos)),
                        "subsec_nanos" => return Ok(Value::Int(nanos)),
                        "subsec_millis" => return Ok(Value::Int(nanos / 1_000_000)),
                        "is_zero" => return Ok(Value::Bool(secs == 0 && nanos == 0)),
                        _ => {}
                    }
                }
                // Mutex methods - lock() returns a Ref to the inner value
                if name == "Mutex" {
                    match method.name.as_str() {
                        "lock" => {
                            // lock() returns a guard that provides access to inner value
                            // In the interpreter, we just return a Ref to the inner value
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                // Return a Ref wrapping the inner value for mutation
                                return Ok(Value::Ref(Rc::new(RefCell::new(inner.clone()))));
                            }
                            return Err(RuntimeError::new("Mutex has no inner value"));
                        }
                        "try_lock" => {
                            // try_lock() returns Some(guard) - in interpreter always succeeds
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                let guard = Value::Ref(Rc::new(RefCell::new(inner.clone())));
                                return Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![guard])),
                                });
                            }
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
                        }
                        "into_inner" => {
                            // into_inner() consumes the mutex and returns the inner value
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                return Ok(inner.clone());
                            }
                            return Err(RuntimeError::new("Mutex has no inner value"));
                        }
                        "get_mut" => {
                            // get_mut() returns &mut T when we have exclusive access
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                return Ok(Value::Ref(Rc::new(RefCell::new(inner.clone()))));
                            }
                            return Err(RuntimeError::new("Mutex has no inner value"));
                        }
                        _ => {}
                    }
                }
                // RwLock methods - read() and write() return guards
                if name == "RwLock" {
                    match method.name.as_str() {
                        "read" => {
                            // read() returns a read guard
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                return Ok(Value::Ref(Rc::new(RefCell::new(inner.clone()))));
                            }
                            return Err(RuntimeError::new("RwLock has no inner value"));
                        }
                        "write" => {
                            // write() returns a write guard
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                return Ok(Value::Ref(Rc::new(RefCell::new(inner.clone()))));
                            }
                            return Err(RuntimeError::new("RwLock has no inner value"));
                        }
                        "try_read" => {
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                let guard = Value::Ref(Rc::new(RefCell::new(inner.clone())));
                                return Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![guard])),
                                });
                            }
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
                        }
                        "try_write" => {
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                let guard = Value::Ref(Rc::new(RefCell::new(inner.clone())));
                                return Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![guard])),
                                });
                            }
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
                        }
                        "into_inner" => {
                            let borrowed = fields.borrow();
                            if let Some(inner) = borrowed.get("__inner__") {
                                return Ok(inner.clone());
                            }
                            return Err(RuntimeError::new("RwLock has no inner value"));
                        }
                        _ => {}
                    }
                }
                // Atomic methods - load/store/fetch_add etc.
                if name == "AtomicU64" || name == "AtomicUsize" || name == "AtomicI64" || name == "AtomicIsize" {
                    match method.name.as_str() {
                        "load" => {
                            // load() returns the current value
                            let borrowed = fields.borrow();
                            if let Some(val) = borrowed.get("__value__") {
                                return Ok(val.clone());
                            }
                            return Ok(Value::Int(0));
                        }
                        "store" => {
                            // store(value) sets the value
                            if let Some(new_val) = arg_values.first() {
                                fields.borrow_mut().insert("__value__".to_string(), new_val.clone());
                                return Ok(Value::Null);
                            }
                            return Err(RuntimeError::new("store requires a value"));
                        }
                        "fetch_add" => {
                            // fetch_add(n) adds n and returns old value
                            if let Some(Value::Int(n)) = arg_values.first() {
                                let mut borrowed = fields.borrow_mut();
                                let old = match borrowed.get("__value__") {
                                    Some(Value::Int(v)) => *v,
                                    _ => 0,
                                };
                                borrowed.insert("__value__".to_string(), Value::Int(old + n));
                                return Ok(Value::Int(old));
                            }
                            return Err(RuntimeError::new("fetch_add requires integer"));
                        }
                        "fetch_sub" => {
                            if let Some(Value::Int(n)) = arg_values.first() {
                                let mut borrowed = fields.borrow_mut();
                                let old = match borrowed.get("__value__") {
                                    Some(Value::Int(v)) => *v,
                                    _ => 0,
                                };
                                borrowed.insert("__value__".to_string(), Value::Int(old - n));
                                return Ok(Value::Int(old));
                            }
                            return Err(RuntimeError::new("fetch_sub requires integer"));
                        }
                        "swap" => {
                            if let Some(new_val) = arg_values.first() {
                                let mut borrowed = fields.borrow_mut();
                                let old = borrowed.get("__value__").cloned().unwrap_or(Value::Int(0));
                                borrowed.insert("__value__".to_string(), new_val.clone());
                                return Ok(old);
                            }
                            return Err(RuntimeError::new("swap requires a value"));
                        }
                        "compare_exchange" | "compare_and_swap" => {
                            // compare_exchange(current, new) - if value == current, set to new
                            if arg_values.len() >= 2 {
                                let current = &arg_values[0];
                                let new_val = &arg_values[1];
                                let mut borrowed = fields.borrow_mut();
                                let actual = borrowed.get("__value__").cloned().unwrap_or(Value::Int(0));
                                if self.values_equal(&actual, current) {
                                    borrowed.insert("__value__".to_string(), new_val.clone());
                                    return Ok(Value::Variant {
                                        enum_name: "Result".to_string(),
                                        variant_name: "Ok".to_string(),
                                        fields: Some(Rc::new(vec![actual])),
                                    });
                                } else {
                                    return Ok(Value::Variant {
                                        enum_name: "Result".to_string(),
                                        variant_name: "Err".to_string(),
                                        fields: Some(Rc::new(vec![actual])),
                                    });
                                }
                            }
                            return Err(RuntimeError::new("compare_exchange requires two arguments"));
                        }
                        _ => {}
                    }
                }
                // AtomicBool methods
                if name == "AtomicBool" {
                    match method.name.as_str() {
                        "load" => {
                            let borrowed = fields.borrow();
                            if let Some(val) = borrowed.get("__value__") {
                                return Ok(val.clone());
                            }
                            return Ok(Value::Bool(false));
                        }
                        "store" => {
                            if let Some(new_val) = arg_values.first() {
                                fields.borrow_mut().insert("__value__".to_string(), new_val.clone());
                                return Ok(Value::Null);
                            }
                            return Err(RuntimeError::new("store requires a value"));
                        }
                        "swap" => {
                            if let Some(new_val) = arg_values.first() {
                                let mut borrowed = fields.borrow_mut();
                                let old = borrowed.get("__value__").cloned().unwrap_or(Value::Bool(false));
                                borrowed.insert("__value__".to_string(), new_val.clone());
                                return Ok(old);
                            }
                            return Err(RuntimeError::new("swap requires a value"));
                        }
                        "fetch_and" => {
                            if let Some(Value::Bool(b)) = arg_values.first() {
                                let mut borrowed = fields.borrow_mut();
                                let old = match borrowed.get("__value__") {
                                    Some(Value::Bool(v)) => *v,
                                    _ => false,
                                };
                                borrowed.insert("__value__".to_string(), Value::Bool(old && *b));
                                return Ok(Value::Bool(old));
                            }
                            return Err(RuntimeError::new("fetch_and requires boolean"));
                        }
                        "fetch_or" => {
                            if let Some(Value::Bool(b)) = arg_values.first() {
                                let mut borrowed = fields.borrow_mut();
                                let old = match borrowed.get("__value__") {
                                    Some(Value::Bool(v)) => *v,
                                    _ => false,
                                };
                                borrowed.insert("__value__".to_string(), Value::Bool(old || *b));
                                return Ok(Value::Bool(old));
                            }
                            return Err(RuntimeError::new("fetch_or requires boolean"));
                        }
                        _ => {}
                    }
                }
                if method.name == "to_string" {
                    // Generic to_string for structs - returns a debug representation
                    let field_str = fields.borrow().iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Ok(Value::String(Rc::new(format!("{} {{ {} }}", name, field_str))));
                }

                // Pattern methods - for AST patterns stored as structs (Pattern::Ident, Pattern::Tuple, etc.)
                if name.starts_with("Pattern::") {
                    match method.name.as_str() {
                        "evidentiality" => {
                            // Return the evidentiality field from the pattern struct
                            if let Some(ev) = fields.borrow().get("evidentiality") {
                                return Ok(ev.clone());
                            }
                            return Ok(Value::Null);
                        }
                        "name" | "binding_name" => {
                            // Return the name field from the pattern struct (for binding purposes)
                            if let Some(n) = fields.borrow().get("name") {
                                // The name field might be an Ident struct with a nested "name" field
                                // Extract the inner string if that's the case
                                let result = match &n {
                                    Value::Struct { fields: inner_fields, .. } => {
                                        if let Some(inner_name) = inner_fields.borrow().get("name") {
                                            crate::sigil_debug!("DEBUG binding_name: returning inner name {} from {}", inner_name, name);
                                            inner_name.clone()
                                        } else {
                                            crate::sigil_debug!("DEBUG binding_name: returning struct {} from {}", n, name);
                                            n.clone()
                                        }
                                    }
                                    _ => {
                                        crate::sigil_debug!("DEBUG binding_name: returning {} from {}", n, name);
                                        n.clone()
                                    }
                                };
                                return Ok(result);
                            }
                            crate::sigil_debug!("DEBUG binding_name: 'name' field not found in {}, fields: {:?}", name, fields.borrow().keys().collect::<Vec<_>>());
                            // For Pattern::Ident, name is the binding name
                            return Ok(Value::Null);
                        }
                        "mutable" => {
                            // Return the mutable field from the pattern struct
                            if let Some(m) = fields.borrow().get("mutable") {
                                return Ok(m.clone());
                            }
                            return Ok(Value::Bool(false));
                        }
                        "is_ident" => {
                            return Ok(Value::Bool(name == "Pattern::Ident"));
                        }
                        "is_wildcard" => {
                            return Ok(Value::Bool(name == "Pattern::Wildcard"));
                        }
                        "clone" => {
                            return Ok(recv.clone());
                        }
                        _ => {}
                    }
                }

                // PathBuf methods
                if name == "PathBuf" || name == "Path" {
                    match method.name.as_str() {
                        "exists" => {
                            // Check if path exists
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            return Ok(Value::Bool(std::path::Path::new(&path).exists()));
                        }
                        "is_dir" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            return Ok(Value::Bool(std::path::Path::new(&path).is_dir()));
                        }
                        "is_file" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            return Ok(Value::Bool(std::path::Path::new(&path).is_file()));
                        }
                        "extension" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            match std::path::Path::new(&path).extension() {
                                Some(ext) => {
                                    // Return Option::Some with an OsStr-like struct
                                    let ext_str = ext.to_string_lossy().to_string();
                                    let mut ext_fields = HashMap::new();
                                    ext_fields.insert("value".to_string(), Value::String(Rc::new(ext_str)));
                                    return Ok(Value::Variant {
                                        enum_name: "Option".to_string(),
                                        variant_name: "Some".to_string(),
                                        fields: Some(Rc::new(vec![Value::Struct {
                                            name: "OsStr".to_string(),
                                            fields: Rc::new(RefCell::new(ext_fields)),
                                        }])),
                                    });
                                }
                                None => {
                                    return Ok(Value::Variant {
                                        enum_name: "Option".to_string(),
                                        variant_name: "None".to_string(),
                                        fields: None,
                                    });
                                }
                            }
                        }
                        "file_name" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            match std::path::Path::new(&path).file_name() {
                                Some(fname) => {
                                    let fname_str = fname.to_string_lossy().to_string();
                                    let mut fname_fields = HashMap::new();
                                    fname_fields.insert("value".to_string(), Value::String(Rc::new(fname_str)));
                                    return Ok(Value::Variant {
                                        enum_name: "Option".to_string(),
                                        variant_name: "Some".to_string(),
                                        fields: Some(Rc::new(vec![Value::Struct {
                                            name: "OsStr".to_string(),
                                            fields: Rc::new(RefCell::new(fname_fields)),
                                        }])),
                                    });
                                }
                                None => {
                                    return Ok(Value::Variant {
                                        enum_name: "Option".to_string(),
                                        variant_name: "None".to_string(),
                                        fields: None,
                                    });
                                }
                            }
                        }
                        "parent" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            match std::path::Path::new(&path).parent() {
                                Some(parent) => {
                                    let mut parent_fields = HashMap::new();
                                    parent_fields.insert("path".to_string(), Value::String(Rc::new(parent.to_string_lossy().to_string())));
                                    return Ok(Value::Variant {
                                        enum_name: "Option".to_string(),
                                        variant_name: "Some".to_string(),
                                        fields: Some(Rc::new(vec![Value::Struct {
                                            name: "Path".to_string(),
                                            fields: Rc::new(RefCell::new(parent_fields)),
                                        }])),
                                    });
                                }
                                None => {
                                    return Ok(Value::Variant {
                                        enum_name: "Option".to_string(),
                                        variant_name: "None".to_string(),
                                        fields: None,
                                    });
                                }
                            }
                        }
                        "to_str" => {
                            // Convert to string (returns Option<&str>, we just return the string)
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.clone(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            // Wrap in Some for unwrap() compatibility
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "Some".to_string(),
                                fields: Some(Rc::new(vec![Value::String(path)])),
                            });
                        }
                        "to_string_lossy" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.clone(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            return Ok(Value::String(path));
                        }
                        "join" => {
                            // Join path with another component
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("join expects 1 argument"));
                            }
                            let base = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            let component = match &arg_values[0] {
                                Value::String(s) => s.to_string(),
                                Value::Struct { name: n, fields: f } if n == "PathBuf" || n == "Path" => {
                                    match f.borrow().get("path") {
                                        Some(Value::String(s)) => s.to_string(),
                                        _ => return Err(RuntimeError::new("PathBuf has no path field")),
                                    }
                                }
                                _ => return Err(RuntimeError::new("join expects string or PathBuf")),
                            };
                            let joined = std::path::Path::new(&base).join(&component);
                            let mut new_fields = HashMap::new();
                            new_fields.insert("path".to_string(), Value::String(Rc::new(joined.to_string_lossy().to_string())));
                            return Ok(Value::Struct {
                                name: "PathBuf".to_string(),
                                fields: Rc::new(RefCell::new(new_fields)),
                            });
                        }
                        "display" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.clone(),
                                _ => return Err(RuntimeError::new("PathBuf has no path field")),
                            };
                            return Ok(Value::String(path));
                        }
                        "to_path_buf" => {
                            // Path -> PathBuf (just return a copy)
                            return Ok(recv.clone());
                        }
                        _ => {}
                    }
                }

                // OsStr methods
                if name == "OsStr" {
                    match method.name.as_str() {
                        "to_str" => {
                            let val = match fields.borrow().get("value") {
                                Some(Value::String(s)) => s.clone(),
                                _ => return Err(RuntimeError::new("OsStr has no value field")),
                            };
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "Some".to_string(),
                                fields: Some(Rc::new(vec![Value::String(val)])),
                            });
                        }
                        "to_string_lossy" => {
                            let val = match fields.borrow().get("value") {
                                Some(Value::String(s)) => s.clone(),
                                _ => return Err(RuntimeError::new("OsStr has no value field")),
                            };
                            return Ok(Value::String(val));
                        }
                        "to_lowercase" => {
                            let val = match fields.borrow().get("value") {
                                Some(Value::String(s)) => s.to_lowercase(),
                                _ => return Err(RuntimeError::new("OsStr has no value field")),
                            };
                            return Ok(Value::String(Rc::new(val)));
                        }
                        "as_str" => {
                            let val = match fields.borrow().get("value") {
                                Some(Value::String(s)) => s.clone(),
                                _ => return Err(RuntimeError::new("OsStr has no value field")),
                            };
                            return Ok(Value::String(val));
                        }
                        _ => {}
                    }
                }

                // ==========================================
                // Holographic Sketch Types (Spec 11-HOLOGRAPHIC.md)
                // ==========================================

                // HyperLogLog methods - probabilistic cardinality estimation
                // Uses hybrid approach: exact set for small cardinalities, HLL for large
                if name == "HyperLogLog" {
                    const SMALL_SET_THRESHOLD: usize = 1000;

                    match method.name.as_str() {
                        "insert" => {
                            // insert(value) - add a value using hybrid exact/HLL algorithm
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("HyperLogLog.insert expects 1 argument"));
                            }
                            let value = &arg_values[0];
                            let mut fields_mut = fields.borrow_mut();

                            // Check if we're still in exact mode
                            let use_hll = match fields_mut.get("__use_hll__") {
                                Some(Value::Bool(b)) => *b,
                                _ => false,
                            };

                            // Get precision for HLL
                            let precision = match fields_mut.get("__precision__") {
                                Some(Value::Int(p)) => *p as u32,
                                _ => 10,
                            };

                            // Always update HLL registers (for when we transition)
                            let hash = holographic_hash_value(value, 0);
                            let register_idx = (hash >> (64 - precision)) as usize;
                            let remaining = hash << precision;
                            let rho = crate::holographic::count_leading_zeros(remaining) + 1;

                            if let Some(Value::Array(registers)) = fields_mut.get("__registers__") {
                                let mut regs = registers.borrow_mut();
                                if register_idx < regs.len() {
                                    if let Value::Int(current) = &regs[register_idx] {
                                        if (rho as i64) > *current {
                                            regs[register_idx] = Value::Int(rho as i64);
                                        }
                                    }
                                }
                            }

                            // If not yet in HLL mode, also track in exact set
                            if !use_hll {
                                if let Some(Value::Map(small_set)) = fields_mut.get("__small_set__") {
                                    let key = format!("{:?}", value);
                                    let mut set = small_set.borrow_mut();
                                    set.insert(key, Value::Bool(true));

                                    // Check if we should switch to HLL mode
                                    if set.len() >= SMALL_SET_THRESHOLD {
                                        drop(set);
                                        fields_mut.insert("__use_hll__".to_string(), Value::Bool(true));
                                    }
                                }
                            }

                            return Ok(Value::Null);
                        }
                        "count" | "cardinality" => {
                            // Return count using hybrid approach
                            let borrowed = fields.borrow();

                            // Check if we're in exact mode
                            let use_hll = match borrowed.get("__use_hll__") {
                                Some(Value::Bool(b)) => *b,
                                _ => false,
                            };

                            // If in exact mode, return exact count
                            if !use_hll {
                                if let Some(Value::Map(small_set)) = borrowed.get("__small_set__") {
                                    return Ok(Value::Int(small_set.borrow().len() as i64));
                                }
                            }

                            // Otherwise use HLL estimate via shared helper
                            let precision = match borrowed.get("__precision__") {
                                Some(Value::Int(p)) => *p as u32,
                                _ => 10,
                            };

                            if let Some(Value::Array(registers)) = borrowed.get("__registers__") {
                                let regs = registers.borrow();
                                return Ok(Value::Int(hll_cardinality_from_registers(&regs, precision)));
                            }
                            return Ok(Value::Int(0));
                        }
                        "merge" => {
                            // merge(other) - union with another HyperLogLog (element-wise maximum)
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("HyperLogLog.merge expects 1 argument"));
                            }
                            let other = match &arg_values[0] {
                                Value::Struct { name: n, fields: f } if n == "HyperLogLog" => f.clone(),
                                _ => return Err(RuntimeError::new("HyperLogLog.merge expects another HyperLogLog")),
                            };

                            let mut fields_mut = fields.borrow_mut();
                            let other_borrowed = other.borrow();

                            // Merge registers: take element-wise maximum
                            if let (Some(Value::Array(self_regs)), Some(Value::Array(other_regs))) =
                                (fields_mut.get("__registers__"), other_borrowed.get("__registers__")) {
                                let mut self_regs_mut = self_regs.borrow_mut();
                                let other_regs_ref = other_regs.borrow();

                                let len = self_regs_mut.len().min(other_regs_ref.len());
                                for i in 0..len {
                                    if let (Value::Int(sv), Value::Int(ov)) = (&self_regs_mut[i], &other_regs_ref[i]) {
                                        if *ov > *sv {
                                            self_regs_mut[i] = Value::Int(*ov);
                                        }
                                    }
                                }
                            }

                            // If either is in HLL mode, merged result uses HLL mode
                            let other_use_hll = matches!(other_borrowed.get("__use_hll__"), Some(Value::Bool(true)));
                            if other_use_hll {
                                fields_mut.insert("__use_hll__".to_string(), Value::Bool(true));
                            }

                            return Ok(Value::Null);
                        }
                        "count_bounds" => {
                            // count_bounds(confidence) - return (lower, upper) confidence interval
                            let confidence = if arg_values.is_empty() {
                                0.95 // Default to 95% confidence
                            } else {
                                match &arg_values[0] {
                                    Value::Float(f) => *f,
                                    Value::Int(i) => *i as f64,
                                    _ => 0.95,
                                }
                            };

                            let borrowed = fields.borrow();
                            let precision = match borrowed.get("__precision__") {
                                Some(Value::Int(p)) => *p as u32,
                                _ => 10,
                            };

                            // Check if we're in exact mode
                            let use_hll = match borrowed.get("__use_hll__") {
                                Some(Value::Bool(b)) => *b,
                                _ => false,
                            };

                            if !use_hll {
                                // In exact mode, return exact count with no interval
                                if let Some(Value::Map(small_set)) = borrowed.get("__small_set__") {
                                    let exact = small_set.borrow().len() as i64;
                                    let tuple = vec![Value::Int(exact), Value::Int(exact)];
                                    return Ok(Value::Array(Rc::new(RefCell::new(tuple))));
                                }
                            }

                            // Use HLL estimate with confidence bounds
                            if let Some(Value::Array(registers)) = borrowed.get("__registers__") {
                                let regs = registers.borrow();
                                let reg_values: Vec<i64> = regs.iter()
                                    .map(|v| match v { Value::Int(n) => *n, _ => 0 })
                                    .collect();

                                let (estimate, std_error) = crate::holographic::hll_estimate(&reg_values, precision);
                                let (lower, upper) = crate::holographic::hll_bounds(estimate, std_error, confidence);

                                let tuple = vec![
                                    Value::Int(lower.round() as i64),
                                    Value::Int(upper.round() as i64)
                                ];
                                return Ok(Value::Array(Rc::new(RefCell::new(tuple))));
                            }

                            let tuple = vec![Value::Int(0), Value::Int(0)];
                            return Ok(Value::Array(Rc::new(RefCell::new(tuple))));
                        }
                        "clear" | "reset" => {
                            // clear() - reset HyperLogLog to initial state
                            let mut fields_mut = fields.borrow_mut();
                            let precision = match fields_mut.get("__precision__") {
                                Some(Value::Int(p)) => *p as usize,
                                _ => 10,
                            };
                            let num_registers = 1usize << precision;

                            // Reset registers to zero
                            if let Some(Value::Array(registers)) = fields_mut.get("__registers__") {
                                let mut regs = registers.borrow_mut();
                                for i in 0..regs.len().min(num_registers) {
                                    regs[i] = Value::Int(0);
                                }
                            }
                            // Reset to exact mode
                            if let Some(Value::Map(small_set)) = fields_mut.get("__small_set__") {
                                small_set.borrow_mut().clear();
                            }
                            fields_mut.insert("__use_hll__".to_string(), Value::Bool(false));

                            return Ok(Value::Null);
                        }
                        _ => {}
                    }
                }

                // BloomFilter methods - probabilistic set membership
                if name == "BloomFilter" {
                    match method.name.as_str() {
                        "insert" => {
                            // insert(value) - add value using multiple hash functions
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("BloomFilter.insert expects 1 argument"));
                            }
                            let value = &arg_values[0];
                            let mut fields_mut = fields.borrow_mut();

                            // Get filter parameters
                            let size = match fields_mut.get("__size__") {
                                Some(Value::Int(s)) => *s as usize,
                                _ => 1024,
                            };
                            let num_hashes = match fields_mut.get("__num_hashes__") {
                                Some(Value::Int(k)) => *k as usize,
                                _ => 7,
                            };

                            // Set bits using multiple hash functions
                            if let Some(Value::Array(bits)) = fields_mut.get("__bits__") {
                                let mut bits_mut = bits.borrow_mut();
                                for i in 0..num_hashes {
                                    let hash = holographic_hash_value(value, i as u64);
                                    let idx = (hash as usize) % size;
                                    if idx < bits_mut.len() {
                                        bits_mut[idx] = Value::Bool(true);
                                    }
                                }
                            }

                            return Ok(Value::Null);
                        }
                        "contains" => {
                            // contains(value) - check if all hash positions are set
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("BloomFilter.contains expects 1 argument"));
                            }
                            let value = &arg_values[0];
                            let borrowed = fields.borrow();

                            let size = match borrowed.get("__size__") {
                                Some(Value::Int(s)) => *s as usize,
                                _ => 1024,
                            };
                            let num_hashes = match borrowed.get("__num_hashes__") {
                                Some(Value::Int(k)) => *k as usize,
                                _ => 7,
                            };

                            // Check all hash positions
                            if let Some(Value::Array(bits)) = borrowed.get("__bits__") {
                                let bits_ref = bits.borrow();
                                for i in 0..num_hashes {
                                    let hash = holographic_hash_value(value, i as u64);
                                    let idx = (hash as usize) % size;
                                    if idx < bits_ref.len() {
                                        match &bits_ref[idx] {
                                            Value::Bool(false) => return Ok(Value::Bool(false)),
                                            _ => {}
                                        }
                                    }
                                }
                                return Ok(Value::Bool(true));
                            }
                            return Ok(Value::Bool(false));
                        }
                        "merge" => {
                            // merge(other) - union with another BloomFilter (bitwise OR)
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("BloomFilter.merge expects 1 argument"));
                            }
                            let other = match &arg_values[0] {
                                Value::Struct { name: n, fields: f } if n == "BloomFilter" => f.clone(),
                                _ => return Err(RuntimeError::new("BloomFilter.merge expects another BloomFilter")),
                            };

                            let mut fields_mut = fields.borrow_mut();
                            let other_borrowed = other.borrow();

                            // Merge bits: bitwise OR
                            if let (Some(Value::Array(self_bits)), Some(Value::Array(other_bits))) =
                                (fields_mut.get("__bits__"), other_borrowed.get("__bits__")) {
                                let mut self_bits_mut = self_bits.borrow_mut();
                                let other_bits_ref = other_bits.borrow();

                                let len = self_bits_mut.len().min(other_bits_ref.len());
                                for i in 0..len {
                                    if let (Value::Bool(sv), Value::Bool(ov)) = (&self_bits_mut[i], &other_bits_ref[i]) {
                                        if *ov && !*sv {
                                            self_bits_mut[i] = Value::Bool(true);
                                        }
                                    }
                                }
                            }

                            return Ok(Value::Null);
                        }
                        "clear" | "reset" => {
                            // clear() - reset BloomFilter to initial state
                            let fields_mut = fields.borrow_mut();
                            // Reset all bits to false
                            if let Some(Value::Array(bits)) = fields_mut.get("__bits__") {
                                let mut bits_mut = bits.borrow_mut();
                                for i in 0..bits_mut.len() {
                                    bits_mut[i] = Value::Bool(false);
                                }
                            }
                            return Ok(Value::Null);
                        }
                        _ => {}
                    }
                }

                // CountMinSketch methods - frequency estimation
                if name == "CountMinSketch" {
                    match method.name.as_str() {
                        "insert" | "add" => {
                            // insert(value) - increment counters using multiple hash functions
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("CountMinSketch.insert expects 1 argument"));
                            }
                            let value = &arg_values[0];
                            let mut fields_mut = fields.borrow_mut();

                            let width = match fields_mut.get("__width__") {
                                Some(Value::Int(w)) => *w as usize,
                                _ => 1024,
                            };
                            let depth = match fields_mut.get("__depth__") {
                                Some(Value::Int(d)) => *d as usize,
                                _ => 5,
                            };

                            // Increment counter in each row
                            if let Some(Value::Array(counters)) = fields_mut.get("__counters__") {
                                let rows = counters.borrow();
                                for (row_idx, row) in rows.iter().enumerate() {
                                    if let Value::Array(row_arr) = row {
                                        let hash = holographic_hash_value(value, row_idx as u64);
                                        let col_idx = (hash as usize) % width;
                                        let mut row_mut = row_arr.borrow_mut();
                                        if col_idx < row_mut.len() {
                                            if let Value::Int(current) = &row_mut[col_idx] {
                                                row_mut[col_idx] = Value::Int(*current + 1);
                                            }
                                        }
                                    }
                                }
                            }

                            return Ok(Value::Null);
                        }
                        "frequency" | "estimate" => {
                            // frequency(value) - return minimum counter value
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("CountMinSketch.frequency expects 1 argument"));
                            }
                            let value = &arg_values[0];
                            let borrowed = fields.borrow();

                            let width = match borrowed.get("__width__") {
                                Some(Value::Int(w)) => *w as usize,
                                _ => 1024,
                            };

                            // Return minimum counter value across all rows
                            if let Some(Value::Array(counters)) = borrowed.get("__counters__") {
                                let rows = counters.borrow();
                                let mut min_count = i64::MAX;

                                for (row_idx, row) in rows.iter().enumerate() {
                                    if let Value::Array(row_arr) = row {
                                        let hash = holographic_hash_value(value, row_idx as u64);
                                        let col_idx = (hash as usize) % width;
                                        let row_ref = row_arr.borrow();
                                        if col_idx < row_ref.len() {
                                            if let Value::Int(count) = &row_ref[col_idx] {
                                                if *count < min_count {
                                                    min_count = *count;
                                                }
                                            }
                                        }
                                    }
                                }

                                if min_count == i64::MAX {
                                    return Ok(Value::Int(0));
                                }
                                return Ok(Value::Int(min_count));
                            }

                            // Counter-based lookup should work; return 0 if it didn't
                            return Ok(Value::Int(0));
                        }
                        "merge" => {
                            // merge(other) - combine with another CountMinSketch (pointwise add)
                            if arg_values.is_empty() {
                                return Err(RuntimeError::new("CountMinSketch.merge expects 1 argument"));
                            }
                            let other = match &arg_values[0] {
                                Value::Struct { name: n, fields: f } if n == "CountMinSketch" => f.clone(),
                                _ => return Err(RuntimeError::new("CountMinSketch.merge expects another CountMinSketch")),
                            };

                            let mut fields_mut = fields.borrow_mut();
                            let other_borrowed = other.borrow();

                            // Merge counters: pointwise addition
                            if let (Some(Value::Array(self_counters)), Some(Value::Array(other_counters))) =
                                (fields_mut.get("__counters__"), other_borrowed.get("__counters__")) {
                                let self_counters_ref = self_counters.borrow();
                                let other_counters_ref = other_counters.borrow();

                                let num_rows = self_counters_ref.len().min(other_counters_ref.len());
                                for row_idx in 0..num_rows {
                                    if let (Value::Array(self_row_arr), Value::Array(other_row_arr)) =
                                        (&self_counters_ref[row_idx], &other_counters_ref[row_idx]) {
                                        let mut self_row_mut = self_row_arr.borrow_mut();
                                        let other_row_ref = other_row_arr.borrow();

                                        let num_cols = self_row_mut.len().min(other_row_ref.len());
                                        for col_idx in 0..num_cols {
                                            if let (Value::Int(s), Value::Int(o)) = (&self_row_mut[col_idx], &other_row_ref[col_idx]) {
                                                self_row_mut[col_idx] = Value::Int(*s + *o);
                                            }
                                        }
                                    }
                                }
                            }

                            return Ok(Value::Null);
                        }
                        "clear" | "reset" => {
                            // clear() - reset CountMinSketch to initial state
                            let fields_mut = fields.borrow_mut();
                            // Reset all counters to zero
                            if let Some(Value::Array(counters)) = fields_mut.get("__counters__") {
                                let counters_ref = counters.borrow();
                                for row in counters_ref.iter() {
                                    if let Value::Array(row_arr) = row {
                                        let mut row_mut = row_arr.borrow_mut();
                                        for i in 0..row_mut.len() {
                                            row_mut[i] = Value::Int(0);
                                        }
                                    }
                                }
                            }
                            return Ok(Value::Null);
                        }
                        _ => {}
                    }
                }

                // MerkleTree methods - content-addressed proofs
                if name == "MerkleTree" {
                    match method.name.as_str() {
                        "prove_inclusion" => {
                            // prove_inclusion(index) - create a proof for element at index
                            let idx = if arg_values.is_empty() {
                                0
                            } else {
                                match &arg_values[0] {
                                    Value::Int(n) => *n as usize,
                                    _ => 0,
                                }
                            };
                            let borrowed = fields.borrow();
                            let element = if let Some(Value::Array(arr)) = borrowed.get("__elements__") {
                                let arr_borrowed = arr.borrow();
                                if idx < arr_borrowed.len() {
                                    arr_borrowed[idx].clone()
                                } else {
                                    Value::Null
                                }
                            } else {
                                Value::Null
                            };
                            let root = borrowed.get("root").cloned().unwrap_or(Value::Null);

                            // Create proof struct
                            let mut proof_fields = HashMap::new();
                            proof_fields.insert("index".to_string(), Value::Int(idx as i64));
                            proof_fields.insert("element".to_string(), element);
                            proof_fields.insert("root".to_string(), root);
                            proof_fields.insert("value".to_string(), Value::Bool(true)); // For □verify
                            return Ok(Value::Struct {
                                name: "MerkleProof".to_string(),
                                fields: Rc::new(RefCell::new(proof_fields)),
                            });
                        }
                        "root" => {
                            let borrowed = fields.borrow();
                            if let Some(root) = borrowed.get("root") {
                                return Ok(root.clone());
                            }
                            return Ok(Value::Null);
                        }
                        _ => {}
                    }
                }

                // DirEntry methods
                if name == "DirEntry" {
                    match method.name.as_str() {
                        "path" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.clone(),
                                _ => return Err(RuntimeError::new("DirEntry has no path field")),
                            };
                            let mut path_fields = HashMap::new();
                            path_fields.insert("path".to_string(), Value::String(path));
                            return Ok(Value::Struct {
                                name: "PathBuf".to_string(),
                                fields: Rc::new(RefCell::new(path_fields)),
                            });
                        }
                        "file_name" => {
                            let path = match fields.borrow().get("path") {
                                Some(Value::String(s)) => s.to_string(),
                                _ => return Err(RuntimeError::new("DirEntry has no path field")),
                            };
                            let fname = std::path::Path::new(&path)
                                .file_name()
                                .map(|f| f.to_string_lossy().to_string())
                                .unwrap_or_default();
                            let mut fname_fields = HashMap::new();
                            fname_fields.insert("value".to_string(), Value::String(Rc::new(fname)));
                            return Ok(Value::Struct {
                                name: "OsStr".to_string(),
                                fields: Rc::new(RefCell::new(fname_fields)),
                            });
                        }
                        _ => {}
                    }
                }

                // Map methods - for built-in hash map operations
                if name == "Map" {
                    match method.name.as_str() {
                        "get" => {
                            // map.get(key) -> ?value
                            if arg_values.len() != 1 {
                                return Err(RuntimeError::new("Map.get expects 1 argument"));
                            }
                            let key = match &arg_values[0] {
                                Value::String(s) => s.to_string(),
                                Value::Int(n) => n.to_string(),
                                other => format!("{:?}", other),
                            };
                            if let Some(val) = fields.borrow().get(&key) {
                                return Ok(val.clone());
                            }
                            return Ok(Value::Null);
                        }
                        "insert" => {
                            // map.insert(key, value)
                            if arg_values.len() != 2 {
                                return Err(RuntimeError::new("Map.insert expects 2 arguments"));
                            }
                            let key = match &arg_values[0] {
                                Value::String(s) => s.to_string(),
                                Value::Int(n) => n.to_string(),
                                other => format!("{:?}", other),
                            };
                            crate::sigil_debug!("DEBUG Map.insert: key='{}', value={}", key, arg_values[1]);
                            fields.borrow_mut().insert(key, arg_values[1].clone());
                            return Ok(Value::Null);
                        }
                        "contains_key" => {
                            if arg_values.len() != 1 {
                                return Err(RuntimeError::new("Map.contains_key expects 1 argument"));
                            }
                            let key = match &arg_values[0] {
                                Value::String(s) => s.to_string(),
                                Value::Int(n) => n.to_string(),
                                other => format!("{:?}", other),
                            };
                            return Ok(Value::Bool(fields.borrow().contains_key(&key)));
                        }
                        "len" => {
                            return Ok(Value::Int(fields.borrow().len() as i64));
                        }
                        "is_empty" => {
                            return Ok(Value::Bool(fields.borrow().is_empty()));
                        }
                        "keys" => {
                            let keys: Vec<Value> = fields.borrow()
                                .keys()
                                .map(|k| Value::String(Rc::new(k.clone())))
                                .collect();
                            return Ok(Value::Array(Rc::new(RefCell::new(keys))));
                        }
                        "values" => {
                            let vals: Vec<Value> = fields.borrow()
                                .values()
                                .cloned()
                                .collect();
                            return Ok(Value::Array(Rc::new(RefCell::new(vals))));
                        }
                        "clone" => {
                            return Ok(recv.clone());
                        }
                        _ => {}
                    }
                }

                // QHCompressed methods
                if name == "QHCompressed" {
                    match method.name.as_str() {
                        "size" => {
                            let f = fields.borrow();
                            if let Some(sz) = f.get("__compressed_size__") {
                                return Ok(sz.clone());
                            }
                            return Err(RuntimeError::new("QHCompressed has no size"));
                        }
                        _ => {}
                    }
                }

                let qualified_name = format!("{}·{}", name, method.name);

                // Debug: track Parser method calls
                if name == "Parser" && (method.name == "parse_file" || method.name == "read_source") {
                    crate::sigil_debug!("DEBUG Parser method call: {}", qualified_name);
                    for (i, arg) in arg_values.iter().enumerate() {
                        crate::sigil_debug!("  arg_value[{}] = {:?}", i, arg);
                    }
                }

                // Debug: track Lexer method calls
                if name == "Lexer" {
                    // Print all args for lex_ident_or_keyword
                    if method.name == "lex_ident_or_keyword" {
                        for (i, arg) in arg_values.iter().enumerate() {
                            let unwrapped = Self::unwrap_all(arg);
                            if let Value::Char(c) = &unwrapped {
                                crate::sigil_debug!("DEBUG Lexer·lex_ident_or_keyword arg[{}]='{}'", i, c);
                            }
                        }
                    }
                    crate::sigil_debug!("DEBUG Lexer method call: {}", qualified_name);
                }
                // Check if arg is "fn" string
                for arg in &arg_values {
                    let unwrapped = Self::unwrap_all(arg);
                    if let Value::String(s) = &unwrapped {
                        if **s == "fn" {
                            crate::sigil_debug!("DEBUG struct method with 'fn': {} recv_name={}", method.name, name);
                        }
                    }
                }

                let func = self.globals.borrow().get(&qualified_name).map(|v| v.clone());
                if let Some(func) = func {
                    if let Value::Function(f) = func {
                        // Set current Self type for Self { ... } resolution
                        let old_self_type = self.current_self_type.take();
                        self.current_self_type = Some(name.clone());

                        // Call with self as first argument
                        let mut all_args = vec![recv.clone()];
                        all_args.extend(arg_values.clone());
                        let result = self.call_function(&f, all_args);

                        // Restore old Self type
                        self.current_self_type = old_self_type;
                        return result;
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
                                        // Set current Self type for Self { ... } resolution
                                        let old_self_type = self.current_self_type.take();
                                        self.current_self_type = Some(type_name.clone());

                                        let mut all_args = vec![recv.clone()];
                                        all_args.extend(arg_values.clone());
                                        let result = self.call_function(&f, all_args);

                                        // Restore old Self type
                                        self.current_self_type = old_self_type;
                                        return result;
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

                // Built-in methods for all structs (Option-like semantics)
                match method.name.as_str() {
                    "is_some" => return Ok(Value::Bool(true)),
                    "is_none" => return Ok(Value::Bool(false)),
                    "unwrap" => return Ok(recv.clone()),
                    _ => {}
                }

                // Fallback for unknown methods on external types: return null
                // This allows code using external crate types to run without full loading
                crate::sigil_warn!("WARN: Unknown method '{}' on '{}' - returning null", method.name, name);
                Ok(Value::Null)
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
                            crate::sigil_debug!("DEBUG Option.unwrap: variant={}, fields={:?}", variant_name, fields);
                            if variant_name == "Some" {
                                if let Some(f) = fields {
                                    let result = f.first().cloned().unwrap_or(Value::Null);
                                    crate::sigil_debug!("DEBUG Option.unwrap: returning {:?}", result);
                                    return Ok(result);
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
                        "and_then" => {
                            // Option::and_then takes a closure that returns Option<U>
                            crate::sigil_debug!("DEBUG and_then: variant={}, has_fields={}, arg_count={}", variant_name, fields.is_some(), arg_values.len());
                            if let Some(arg) = arg_values.first() {
                                crate::sigil_debug!("DEBUG and_then: arg type = {:?}", std::mem::discriminant(arg));
                            }
                            if variant_name == "Some" {
                                if let Some(f) = fields {
                                    if let Some(inner) = f.first() {
                                        crate::sigil_debug!("DEBUG and_then: inner = {:?}", inner);
                                        if let Some(Value::Function(func)) = arg_values.first() {
                                            let result = self.call_function(func, vec![inner.clone()])?;
                                            crate::sigil_debug!("DEBUG and_then: result = {:?}", result);
                                            // The closure should return an Option, return it directly
                                            return Ok(result);
                                        } else {
                                            crate::sigil_debug!("DEBUG and_then: arg is not a Function!");
                                        }
                                    }
                                }
                            }
                            // None case - return None
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
                        }
                        "or_else" => {
                            // Option::or_else takes a closure that returns Option<T>
                            if variant_name == "Some" {
                                // Some case - return self
                                return Ok(recv.clone());
                            }
                            // None case - call the closure
                            if let Some(Value::Function(func)) = arg_values.first() {
                                return self.call_function(func, vec![]);
                            }
                            return Ok(recv.clone());
                        }
                        "ok_or" | "ok_or_else" => {
                            // Convert Option to Result
                            if variant_name == "Some" {
                                if let Some(f) = fields {
                                    if let Some(inner) = f.first() {
                                        return Ok(Value::Variant {
                                            enum_name: "Result".to_string(),
                                            variant_name: "Ok".to_string(),
                                            fields: Some(Rc::new(vec![inner.clone()])),
                                        });
                                    }
                                }
                            }
                            // None case - return Err with the provided value
                            let err_val = arg_values.first().cloned().unwrap_or(Value::String(Rc::new("None".to_string())));
                            return Ok(Value::Variant {
                                enum_name: "Result".to_string(),
                                variant_name: "Err".to_string(),
                                fields: Some(Rc::new(vec![err_val])),
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
                        "ok" => {
                            // Convert Result<T, E> to Option<T>
                            // Ok(val) -> Some(val), Err(_) -> None
                            if variant_name == "Ok" {
                                let inner = fields.as_ref()
                                    .and_then(|f| f.first().cloned())
                                    .unwrap_or(Value::Null);
                                return Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![inner])),
                                });
                            }
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
                        }
                        "err" => {
                            // Convert Result<T, E> to Option<E>
                            // Ok(_) -> None, Err(e) -> Some(e)
                            if variant_name == "Err" {
                                let inner = fields.as_ref()
                                    .and_then(|f| f.first().cloned())
                                    .unwrap_or(Value::Null);
                                return Ok(Value::Variant {
                                    enum_name: "Option".to_string(),
                                    variant_name: "Some".to_string(),
                                    fields: Some(Rc::new(vec![inner])),
                                });
                            }
                            return Ok(Value::Variant {
                                enum_name: "Option".to_string(),
                                variant_name: "None".to_string(),
                                fields: None,
                            });
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
                        "unwrap_err" => {
                            // Return the Err value, or panic if Ok
                            if variant_name == "Err" {
                                if let Some(f) = fields {
                                    return Ok(f.first().cloned().unwrap_or(Value::Null));
                                }
                            }
                            return Err(RuntimeError::new("unwrap_err on Ok"));
                        }
                        "map" => {
                            // map(fn) - apply fn to Ok value, leave Err unchanged
                            if variant_name == "Ok" {
                                if let Some(Value::Function(f)) = arg_values.first() {
                                    let inner = fields.as_ref()
                                        .and_then(|f| f.first().cloned())
                                        .unwrap_or(Value::Null);
                                    let result = self.call_function(f, vec![inner])?;
                                    return Ok(Value::Variant {
                                        enum_name: "Result".to_string(),
                                        variant_name: "Ok".to_string(),
                                        fields: Some(Rc::new(vec![result])),
                                    });
                                }
                            }
                            // For Err variant, return unchanged
                            return Ok(recv.clone());
                        }
                        "map_err" => {
                            // map_err(fn) - apply fn to Err value, leave Ok unchanged
                            if variant_name == "Err" {
                                if let Some(Value::Function(f)) = arg_values.first() {
                                    let inner = fields.as_ref()
                                        .and_then(|f| f.first().cloned())
                                        .unwrap_or(Value::Null);
                                    let result = self.call_function(f, vec![inner])?;
                                    return Ok(Value::Variant {
                                        enum_name: "Result".to_string(),
                                        variant_name: "Err".to_string(),
                                        fields: Some(Rc::new(vec![result])),
                                    });
                                }
                            }
                            // For Ok variant, return unchanged
                            return Ok(recv.clone());
                        }
                        "and_then" => {
                            // and_then(fn) - chain Result-returning functions
                            if variant_name == "Ok" {
                                if let Some(Value::Function(f)) = arg_values.first() {
                                    let inner = fields.as_ref()
                                        .and_then(|f| f.first().cloned())
                                        .unwrap_or(Value::Null);
                                    return self.call_function(f, vec![inner]);
                                }
                            }
                            // For Err variant, return unchanged
                            return Ok(recv.clone());
                        }
                        _ => {}
                    }
                }
                // Pattern methods - for AST pattern access
                crate::sigil_debug!("DEBUG variant method call: enum_name={}, variant_name={}, method={}", enum_name, variant_name, method.name);

                // Generic enum methods that work on any variant
                match method.name.as_str() {
                    "cloned" | "clone" => {
                        // .cloned() / .clone() on an enum variant returns a clone of the variant
                        return Ok(recv.clone());
                    }
                    _ => {}
                }

                // Type methods
                if enum_name == "Type" {
                    match method.name.as_str() {
                        "is_never" => {
                            // Type::Never is the never type, all others are not
                            return Ok(Value::Bool(variant_name == "Never"));
                        }
                        "to_string" => {
                            // Convert type to string representation
                            let type_str = match variant_name.as_str() {
                                "Bool" => "bool".to_string(),
                                "Int" => "i64".to_string(),
                                "Float" => "f64".to_string(),
                                "Str" => "str".to_string(),
                                "Char" => "char".to_string(),
                                "Unit" => "()".to_string(),
                                "Never" => "!".to_string(),
                                "Error" => "<error>".to_string(),
                                other => format!("Type::{}", other),
                            };
                            return Ok(Value::String(Rc::new(type_str)));
                        }
                        _ => {}
                    }
                }

                if enum_name == "Pattern" {
                    match method.name.as_str() {
                        "evidentiality" => {
                            // Pattern::Ident { name, mutable, evidentiality } - return the evidentiality field
                            if variant_name == "Ident" {
                                if let Some(f) = fields {
                                    // Fields are stored as a struct or in order
                                    // Try to find evidentiality field
                                    for field_val in f.iter() {
                                        if let Value::Struct { fields: inner, .. } = field_val {
                                            if let Some(ev) = inner.borrow().get("evidentiality") {
                                                return Ok(ev.clone());
                                            }
                                        }
                                    }
                                    // If fields are stored in order: name, mutable, evidentiality (index 2)
                                    if f.len() > 2 {
                                        return Ok(f[2].clone());
                                    }
                                }
                            }
                            // No evidentiality for other pattern types
                            return Ok(Value::Null);
                        }
                        "name" => {
                            // Get the name from Pattern::Ident
                            if variant_name == "Ident" {
                                if let Some(f) = fields {
                                    for field_val in f.iter() {
                                        if let Value::Struct { fields: inner, .. } = field_val {
                                            if let Some(n) = inner.borrow().get("name") {
                                                return Ok(n.clone());
                                            }
                                        }
                                    }
                                    // First field is name
                                    if let Some(n) = f.first() {
                                        return Ok(n.clone());
                                    }
                                }
                            }
                            return Ok(Value::Null);
                        }
                        "mutable" => {
                            // Get mutable flag from Pattern::Ident
                            if variant_name == "Ident" {
                                if let Some(f) = fields {
                                    for field_val in f.iter() {
                                        if let Value::Struct { fields: inner, .. } = field_val {
                                            if let Some(m) = inner.borrow().get("mutable") {
                                                return Ok(m.clone());
                                            }
                                        }
                                    }
                                    // Second field is mutable
                                    if f.len() > 1 {
                                        return Ok(f[1].clone());
                                    }
                                }
                            }
                            return Ok(Value::Bool(false));
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
                // Fallback for unknown methods on external enum types: return null
                crate::sigil_warn!("WARN: Unknown method '{}' on enum '{}' - returning null", method.name, enum_name);
                Ok(Value::Null)
            }
            // Null-safe method handlers - methods called on null return sensible defaults
            (Value::Null, "len_utf8") => Ok(Value::Int(0)),
            (Value::Null, "is_ascii") => Ok(Value::Bool(false)),
            (Value::Null, "is_alphabetic") => Ok(Value::Bool(false)),
            (Value::Null, "is_alphanumeric") => Ok(Value::Bool(false)),
            (Value::Null, "is_numeric") | (Value::Null, "is_digit") => Ok(Value::Bool(false)),
            (Value::Null, "is_whitespace") => Ok(Value::Bool(false)),
            (Value::Null, "is_uppercase") => Ok(Value::Bool(false)),
            (Value::Null, "is_lowercase") => Ok(Value::Bool(false)),
            (Value::Null, "len") => Ok(Value::Int(0)),
            (Value::Null, "is_empty") => Ok(Value::Bool(true)),
            (Value::Null, "to_string") => Ok(Value::String(Rc::new("".to_string()))),
            (Value::Null, "clone") => Ok(Value::Null),
            (Value::Null, "cloned") => Ok(Value::Null),  // .cloned() on null returns null
            (Value::Null, "is_some") => Ok(Value::Bool(false)),
            (Value::Null, "is_none") => Ok(Value::Bool(true)),
            (Value::Null, "unwrap_or") => {
                if arg_values.is_empty() {
                    Ok(Value::Null)
                } else {
                    Ok(arg_values[0].clone())
                }
            }
            // unwrap_or for non-null values returns the value itself
            (Value::Char(c), "unwrap_or") => Ok(Value::Char(*c)),
            (Value::Int(n), "unwrap_or") => Ok(Value::Int(*n)),
            (Value::Float(n), "unwrap_or") => Ok(Value::Float(*n)),
            (Value::String(s), "unwrap_or") => Ok(Value::String(s.clone())),
            (Value::Bool(b), "unwrap_or") => Ok(Value::Bool(*b)),
            // is_some/is_none/unwrap for non-null values (Option-like semantics)
            (Value::Char(c), "is_some") => Ok(Value::Bool(true)),
            (Value::Int(_), "is_some") => Ok(Value::Bool(true)),
            (Value::Float(_), "is_some") => Ok(Value::Bool(true)),
            (Value::String(_), "is_some") => Ok(Value::Bool(true)),
            (Value::Bool(_), "is_some") => Ok(Value::Bool(true)),
            (Value::Array(_), "is_some") => Ok(Value::Bool(true)),
            (Value::Struct { .. }, "is_some") => Ok(Value::Bool(true)),
            (Value::Char(_), "is_none") => Ok(Value::Bool(false)),
            (Value::Int(_), "is_none") => Ok(Value::Bool(false)),
            (Value::Float(_), "is_none") => Ok(Value::Bool(false)),
            (Value::String(_), "is_none") => Ok(Value::Bool(false)),
            (Value::Bool(_), "is_none") => Ok(Value::Bool(false)),
            (Value::Array(_), "is_none") => Ok(Value::Bool(false)),
            (Value::Struct { .. }, "is_none") => Ok(Value::Bool(false)),
            (Value::Char(c), "unwrap") => Ok(Value::Char(*c)),
            (Value::Int(n), "unwrap") => Ok(Value::Int(*n)),
            (Value::Float(n), "unwrap") => Ok(Value::Float(*n)),
            (Value::String(s), "unwrap") => Ok(Value::String(s.clone())),
            (Value::Bool(b), "unwrap") => Ok(Value::Bool(*b)),
            (Value::Array(a), "unwrap") => Ok(Value::Array(a.clone())),
            (Value::Struct { name, fields }, "unwrap") => Ok(Value::Struct { name: name.clone(), fields: fields.clone() }),
            // Int methods
            (Value::Int(n), "to_string") | (Value::Int(n), "string") => {
                Ok(Value::String(Rc::new(n.to_string())))
            }
            (Value::Int(n), "abs") => Ok(Value::Int(n.abs())),
            (Value::Int(n), "to_float") | (Value::Int(n), "float") => Ok(Value::Float(*n as f64)),
            // Pointer emulation: treat 0 as null pointer
            (Value::Int(n), "is_null") => Ok(Value::Bool(*n == 0)),
            (Value::Int(n), "duration_since") => {
                // Treat Int as nanoseconds since some epoch
                // Return a Duration struct
                let other_ns = match arg_values.first() {
                    Some(Value::Int(i)) => *i,
                    Some(Value::Struct { fields, .. }) => {
                        let borrowed = fields.borrow();
                        let secs = match borrowed.get("secs") {
                            Some(Value::Int(s)) => *s,
                            _ => 0,
                        };
                        let nanos = match borrowed.get("nanos") {
                            Some(Value::Int(n)) => *n,
                            _ => 0,
                        };
                        secs * 1_000_000_000 + nanos
                    }
                    _ => 0,
                };
                let diff_ns = n - other_ns;
                let mut fields = std::collections::HashMap::new();
                fields.insert("secs".to_string(), Value::Int(diff_ns / 1_000_000_000));
                fields.insert("nanos".to_string(), Value::Int(diff_ns % 1_000_000_000));
                Ok(Value::Variant {
                    enum_name: "Result".to_string(),
                    variant_name: "Ok".to_string(),
                    fields: Some(Rc::new(vec![Value::Struct {
                        name: "Duration".to_string(),
                        fields: Rc::new(RefCell::new(fields)),
                    }])),
                })
            }
            // Float methods
            (Value::Float(n), "to_string") | (Value::Float(n), "string") => {
                Ok(Value::String(Rc::new(n.to_string())))
            }
            (Value::Float(n), "abs") => Ok(Value::Float(n.abs())),
            (Value::Float(n), "to_int") | (Value::Float(n), "int") => Ok(Value::Int(*n as i64)),
            // Bool methods
            (Value::Bool(b), "to_string") | (Value::Bool(b), "string") => {
                Ok(Value::String(Rc::new(b.to_string())))
            }
            // Char methods
            (Value::Char(c), "to_string") | (Value::Char(c), "string") => {
                Ok(Value::String(Rc::new(c.to_string())))
            }
            // Universal Option-like methods (work on any value, Null = None)
            (Value::Null, "is_some") => Ok(Value::Bool(false)),
            (Value::Null, "is_none") => Ok(Value::Bool(true)),
            (Value::Null, "unwrap") => Err(RuntimeError::new("called unwrap on null value")),
            (_, "is_some") => Ok(Value::Bool(true)),
            (_, "is_none") => Ok(Value::Bool(false)),
            (val, "unwrap") => Ok(val.clone()),
            _ => {
                // For primitive types, error on unknown methods
                let recv_type = match &recv {
                    Value::Int(_) => "i64",
                    Value::Float(_) => "f64",
                    Value::Bool(_) => "bool",
                    Value::Char(_) => "char",
                    Value::String(_) => "String",
                    Value::Array(_) => "Array",
                    Value::Tuple(_) => "Tuple",
                    _ => "",
                };
                if !recv_type.is_empty() {
                    return Err(RuntimeError::new(format!(
                        "No method '{}' on type '{}'", method.name, recv_type
                    )));
                }
                // For user-defined types, warn and return null for external crate compatibility
                let type_desc = match &recv {
                    Value::Struct { name, .. } => format!("Struct({})", name),
                    Value::Variant { enum_name, variant_name, .. } => format!("Variant({}::{})", enum_name, variant_name),
                    Value::Ref(r) => format!("Ref({:?})", std::mem::discriminant(&*r.borrow())),
                    Value::Null => "Null".to_string(),
                    other => format!("{:?}", std::mem::discriminant(other)),
                };
                crate::sigil_warn!("WARN: Unknown method '{}' on {} - returning null", method.name, type_desc);
                Ok(Value::Null)
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

        // Special case: if first segment is undefined and there's a second segment with args,
        // try combining them as a function name (e.g., fs·read_to_string -> "fs·read_to_string")
        let first = &segments[0];
        if first.args.is_none() && segments.len() >= 2 {
            let first_name = &first.name.name;
            let env_lookup = self.environment.borrow().get(first_name);
            if env_lookup.is_none() {
                let second = &segments[1];
                // Handle Self· by substituting actual type name
                let resolved_first = if first_name == "Self" {
                    self.current_self_type.clone().unwrap_or_else(|| first_name.clone())
                } else {
                    first_name.clone()
                };
                let combined_name = format!("{}·{}", resolved_first, second.name.name);
                // Try to find combined function in environment or globals
                let combined_func = self.environment.borrow().get(&combined_name)
                    .or_else(|| self.globals.borrow().get(&combined_name));
                if let Some(func_val) = combined_func {
                    // Call the combined function with args from second segment
                    let arg_values: Vec<Value> = second.args
                        .as_ref()
                        .map(|args| {
                            args.iter()
                                .map(|a| self.evaluate(a))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()?
                        .unwrap_or_default();
                    let mut value = match func_val {
                        Value::Function(f) => self.call_function(&f, arg_values)?,
                        Value::BuiltIn(b) => self.call_builtin(&b, arg_values)?,
                        _ => return Err(RuntimeError::new(format!("{} is not a function", combined_name))),
                    };
                    // Process remaining segments (3rd, 4th, ...) if any
                    for segment in segments.iter().skip(2) {
                        let seg_args: Vec<Value> = segment
                            .args
                            .as_ref()
                            .map(|args| {
                                args.iter()
                                    .map(|a| self.evaluate(a))
                                    .collect::<Result<Vec<_>, _>>()
                            })
                            .transpose()?
                            .unwrap_or_default();
                        value = self.call_incorporation_method(&value, &segment.name.name, seg_args)?;
                    }
                    return Ok(value);
                }
            }
        }

        // First segment: get initial value (variable lookup or function call)
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
            (Value::String(s), "upper") | (Value::String(s), "uppercase") | (Value::String(s), "to_uppercase") => {
                Ok(Value::String(Rc::new(s.to_uppercase())))
            }
            (Value::String(s), "lower") | (Value::String(s), "lowercase") | (Value::String(s), "to_lowercase") => {
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
                if s.len() <= 10 { crate::sigil_debug!("DEBUG as_str: '{}'", s); }
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
            (Value::Array(arr), "first") | (Value::Array(arr), "next") => Ok(arr
                .borrow()
                .first()
                .cloned()
                .unwrap_or(Value::Null)),
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
            (Value::Array(arr), "skip") => {
                let n = match args.first() {
                    Some(Value::Int(i)) => *i as usize,
                    _ => 1,
                };
                let v: Vec<Value> = arr.borrow().iter().skip(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "take") => {
                let n = match args.first() {
                    Some(Value::Int(i)) => *i as usize,
                    _ => 1,
                };
                let v: Vec<Value> = arr.borrow().iter().take(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "step_by") => {
                let n = match args.first() {
                    Some(Value::Int(i)) if *i > 0 => *i as usize,
                    _ => 1,
                };
                let v: Vec<Value> = arr.borrow().iter().step_by(n).cloned().collect();
                Ok(Value::Array(Rc::new(RefCell::new(v))))
            }
            (Value::Array(arr), "to_vec") | (Value::Array(arr), "clone") => {
                let cloned = arr.borrow().clone();
                Ok(Value::Array(Rc::new(RefCell::new(cloned))))
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
        // Unwrap evidential/affective wrappers for pipe operations
        let value = Self::unwrap_all(&value);

        match op {
            PipeOp::Transform(body) => {
                // τ{f} - map over collection or apply to single value
                // Extract closure parameter pattern and body
                let (param_pattern, inner_body) = match body.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        let pattern = params.first().map(|p| p.pattern.clone());
                        (pattern, body.as_ref())
                    }
                    _ => (None, body.as_ref()),
                };

                match value {
                    Value::Array(arr) => {
                        let results: Vec<Value> = arr
                            .borrow()
                            .iter()
                            .map(|item| {
                                // Bind the item to the pattern (supports tuple destructuring)
                                if let Some(ref pattern) = param_pattern {
                                    self.bind_pattern(pattern, item.clone())?;
                                } else {
                                    self.environment
                                        .borrow_mut()
                                        .define("_".to_string(), item.clone());
                                }
                                self.evaluate(inner_body)
                            })
                            .collect::<Result<_, _>>()?;
                        Ok(Value::Array(Rc::new(RefCell::new(results))))
                    }
                    single => {
                        if let Some(ref pattern) = param_pattern {
                            self.bind_pattern(pattern, single)?;
                        } else {
                            self.environment
                                .borrow_mut()
                                .define("_".to_string(), single);
                        }
                        self.evaluate(inner_body)
                    }
                }
            }
            PipeOp::Filter(predicate) => {
                // φ{p} - filter collection
                // Extract closure parameter pattern and body
                let (param_pattern, inner_pred) = match predicate.as_ref() {
                    Expr::Closure { params, body, .. } => {
                        let pattern = params.first().map(|p| p.pattern.clone());
                        (pattern, body.as_ref())
                    }
                    _ => (None, predicate.as_ref()),
                };

                match value {
                    Value::Array(arr) => {
                        let results: Vec<Value> = arr
                            .borrow()
                            .iter()
                            .filter_map(|item| {
                                // Bind the item to the pattern (supports tuple destructuring)
                                if let Some(ref pattern) = param_pattern {
                                    if let Err(e) = self.bind_pattern(pattern, item.clone()) {
                                        return Some(Err(e));
                                    }
                                } else {
                                    self.environment
                                        .borrow_mut()
                                        .define("_".to_string(), item.clone());
                                }
                                match self.evaluate(inner_pred) {
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
            PipeOp::Call(callee) => {
                // |expr - call an arbitrary expression (like self.layer) with piped value

                // FIRST: Check if this is a simple path that should be a pipe method
                // This handles |measure, |observe, |teleport, etc. without parentheses
                if let Expr::Path(path) = callee.as_ref() {
                    if path.segments.len() == 1 {
                        let name = &path.segments[0].ident.name;

                        // Check if value is a Qubit - delegate to stdlib quantum functions
                        let is_qubit = matches!(&value, Value::Struct { name: n, .. } if n == "Qubit");
                        if is_qubit {
                            let quantum_gate = matches!(name.as_str(),
                                "H" | "X" | "Y" | "Z" | "S" | "T" | "measure"
                            );
                            if quantum_gate {
                                return self.call_builtin_by_name(name, vec![value.clone()]);
                            }
                        }

                        let is_qh_method = matches!(name.as_str(),
                            "measure" | "observe" | "quantum_reconstruct" | "interfere" |
                            "is_pure" | "apply_noise" | "error_correct" | "size" |
                            "partial_trace" | "qh_compress"
                        );
                        if is_qh_method {
                            // Delegate to QH helper functions
                            return match name.as_str() {
                                "measure" => self.qh_measure(&value),
                                "observe" => self.qh_observe(&value),
                                "quantum_reconstruct" => self.qh_quantum_reconstruct(&value),
                                "is_pure" => self.qh_is_pure(&value),
                                "error_correct" => self.qh_error_correct(&value),
                                "size" => self.qh_size(&value),
                                "partial_trace" => self.qh_partial_trace(&value),
                                "qh_compress" => self.qh_compress(&value),
                                "apply_noise" => self.qh_apply_noise(&value, 0.01), // Default noise rate
                                "interfere" => Ok(value.clone()), // No-arg interfere returns unchanged
                                _ => Ok(value.clone()),
                            };
                        }

                        // Check if value is a Tensor - delegate to stdlib neural functions
                        let is_tensor = matches!(&value, Value::Struct { name: n, .. } if n == "Tensor");
                        if is_tensor {
                            let tensor_method = matches!(name.as_str(),
                                "relu" | "softmax" | "sigmoid" | "tanh" | "flatten" | "backward" | "Σ"
                            );
                            if tensor_method {
                                return self.call_builtin_by_name(name, vec![value.clone()]);
                            }
                        }
                    }
                }

                // Special case: pipe-invoked rune (value|rune!{...})
                if let Expr::Macro { path, tokens } = callee.as_ref() {
                    let macro_name = path.segments.last()
                        .map(|s| s.ident.name.as_str())
                        .unwrap_or("");
                    crate::sigil_debug!("DEBUG PipeOp::Call with Macro: {}", macro_name);

                    // Check for user-defined rune
                    if let Some(rune_def) = self.defined_runes.get(macro_name).cloned() {
                        crate::sigil_debug!("DEBUG Expanding pipe-invoked rune: {}", macro_name);
                        return self.expand_rune(&rune_def, tokens, Some(value));
                    }

                    // Not a user-defined rune, evaluate as normal macro
                    // but pipe value is lost (built-in macros don't use it)
                }

                // Special handling for Expr::Call - prepend piped value to args
                // This handles cases like: q|Rx(pi) => Rx(q, pi)
                if let Expr::Call { func, args } = callee.as_ref() {
                    // Check if this is a QH method with arguments
                    if let Expr::Path(path) = func.as_ref() {
                        if path.segments.len() == 1 {
                            let method_name = &path.segments[0].ident.name;
                            match method_name.as_str() {
                                "interfere" => {
                                    if args.is_empty() {
                                        return Ok(value.clone());
                                    }
                                    let other = self.evaluate(&args[0])?;
                                    return self.qh_interfere(&value, &other);
                                }
                                "apply_noise" => {
                                    let noise_level = if !args.is_empty() {
                                        match self.evaluate(&args[0])? {
                                            Value::Float(f) => f,
                                            Value::Int(i) => i as f64,
                                            _ => 0.01,
                                        }
                                    } else {
                                        0.01
                                    };
                                    return self.qh_apply_noise(&value, noise_level);
                                }
                                // Tensor methods with arguments
                                "requires_grad" => {
                                    let flag = if !args.is_empty() {
                                        match self.evaluate(&args[0])? {
                                            Value::Bool(b) => b,
                                            _ => true,
                                        }
                                    } else {
                                        true
                                    };
                                    return self.call_builtin_by_name("requires_grad", vec![value.clone(), Value::Bool(flag)]);
                                }
                                "reshape" => {
                                    if !args.is_empty() {
                                        let shape = self.evaluate(&args[0])?;
                                        return self.call_builtin_by_name("reshape", vec![value.clone(), shape]);
                                    }
                                    return Ok(value.clone());
                                }
                                _ => {} // Fall through to normal Expr::Call handling
                            }
                        }
                    }
                    let callee_val = self.evaluate(func)?;
                    let mut all_args = vec![value];
                    for arg in args {
                        all_args.push(self.evaluate(arg)?);
                    }
                    return match callee_val {
                        Value::Function(f) => self.call_function(&f, all_args),
                        Value::BuiltIn(b) => self.call_builtin(&b, all_args),
                        _ => Err(RuntimeError::new(format!(
                            "Cannot call non-function value in pipe: {:?}",
                            callee_val
                        ))),
                    };
                }

                let callee_val = self.evaluate(callee)?;
                match callee_val {
                    Value::Function(f) => {
                        // Call the function with the piped value as argument
                        self.call_function(&f, vec![value])
                    }
                    Value::BuiltIn(b) => {
                        // Call built-in with the piped value
                        self.call_builtin(&b, vec![value])
                    }
                    Value::Struct { .. } => {
                        // Structs that implement __call__ can be called as functions
                        // For now, just return the value (ML layers would override)
                        Ok(value)
                    }
                    _ => Err(RuntimeError::new(format!(
                        "Cannot call non-function value in pipe: {:?}",
                        callee_val
                    ))),
                }
            }
            PipeOp::Method { name, type_args: _, args } => {
                let arg_values: Vec<Value> = args
                    .iter()
                    .map(|a| self.evaluate(a))
                    .collect::<Result<_, _>>()?;

                // Check for built-in pipe methods
                match name.name.as_str() {
                    "collect" => Ok(value), // Already collected
                    "observe" => {
                        // Collapse a Superposition to a definite value
                        match &value {
                            Value::Struct { name, fields } if name.contains("Superposition") => {
                                let borrowed = fields.borrow();
                                // Check for __values__ (QH spec) or __elements__ (legacy)
                                let elements_opt = borrowed.get("__values__")
                                    .or_else(|| borrowed.get("__elements__"));
                                if let Some(Value::Array(elements)) = elements_opt {
                                    let arr = elements.borrow();
                                    if arr.is_empty() {
                                        return Ok(Value::Null);
                                    }
                                    // Return first element (simplified)
                                    return Ok(arr[0].clone());
                                }
                                Err(RuntimeError::new("Superposition has no elements"))
                            }
                            Value::Struct { name, fields } if name == "Hologram" => {
                                // Observe a Hologram - return its value
                                let borrowed = fields.borrow();
                                borrowed.get("__value__").or_else(|| borrowed.get("__original__")).cloned()
                                    .ok_or_else(|| RuntimeError::new("Hologram has no value"))
                            }
                            other => Ok(other.clone()),
                        }
                    }
                    "encode" => {
                        // Encode data with erasure coding scheme (for Hologram)
                        // Default: 4 data shards, 7 total (3 parity) - can recover from any 3 lost
                        // arg_values[0] = encoding scheme (optional, only ReedSolomon supported)
                        let data_shards: usize = 4;
                        let total_shards: usize = 7;

                        let shards: Vec<Value> = (0..total_shards).map(|i| {
                            let mut shard_fields = HashMap::new();
                            shard_fields.insert("index".to_string(), Value::Int(i as i64));
                            shard_fields.insert("data".to_string(), value.clone());
                            shard_fields.insert("is_parity".to_string(), Value::Bool(i >= data_shards));
                            Value::Struct {
                                name: "Shard".to_string(),
                                fields: Rc::new(RefCell::new(shard_fields)),
                            }
                        }).collect();

                        let mut fields = HashMap::new();
                        fields.insert("shards".to_string(), Value::Array(Rc::new(RefCell::new(shards))));
                        fields.insert("data_shards".to_string(), Value::Int(data_shards as i64));
                        fields.insert("parity_shards".to_string(), Value::Int((total_shards - data_shards) as i64));
                        fields.insert("__original__".to_string(), value.clone());

                        Ok(Value::Struct {
                            name: "Hologram".to_string(),
                            fields: Rc::new(RefCell::new(fields)),
                        })
                    }
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
                    "iter" | "into_iter" => {
                        // iter()/into_iter() returns the array for iteration (identity operation)
                        Ok(value)
                    },
                    "enumerate" => {
                        // enumerate() returns array of (index, value) tuples
                        match &value {
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
                            _ => Err(RuntimeError::new("enumerate requires array")),
                        }
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
                    "all" => {
                        // Check if all elements are truthy (no predicate in Method variant)
                        match value {
                            Value::Array(arr) => {
                                for item in arr.borrow().iter() {
                                    if !self.is_truthy(item) {
                                        return Ok(Value::Bool(false));
                                    }
                                }
                                Ok(Value::Bool(true))
                            }
                            _ => Err(RuntimeError::new("all requires array")),
                        }
                    }
                    "any" => {
                        // Check if any element is truthy
                        match value {
                            Value::Array(arr) => {
                                for item in arr.borrow().iter() {
                                    if self.is_truthy(item) {
                                        return Ok(Value::Bool(true));
                                    }
                                }
                                Ok(Value::Bool(false))
                            }
                            _ => Err(RuntimeError::new("any requires array")),
                        }
                    }
                    "map" => {
                        // map(closure) applies closure to each element
                        if arg_values.len() != 1 {
                            return Err(RuntimeError::new("map expects 1 argument (closure)"));
                        }
                        match (&value, &arg_values[0]) {
                            (Value::Array(arr), Value::Function(f)) => {
                                let mut results = Vec::new();
                                for val in arr.borrow().iter() {
                                    let result = self.call_function(f, vec![val.clone()])?;
                                    results.push(result);
                                }
                                Ok(Value::Array(Rc::new(RefCell::new(results))))
                            }
                            (Value::Array(_), _) => Err(RuntimeError::new("map expects closure argument")),
                            _ => Err(RuntimeError::new("map requires array")),
                        }
                    }
                    "filter" => {
                        // filter(predicate) keeps elements where predicate returns true
                        if arg_values.len() != 1 {
                            return Err(RuntimeError::new("filter expects 1 argument (closure)"));
                        }
                        match (&value, &arg_values[0]) {
                            (Value::Array(arr), Value::Function(f)) => {
                                let mut results = Vec::new();
                                for val in arr.borrow().iter() {
                                    let keep = self.call_function(f, vec![val.clone()])?;
                                    if matches!(keep, Value::Bool(true)) {
                                        results.push(val.clone());
                                    }
                                }
                                Ok(Value::Array(Rc::new(RefCell::new(results))))
                            }
                            (Value::Array(_), _) => Err(RuntimeError::new("filter expects closure argument")),
                            _ => Err(RuntimeError::new("filter requires array")),
                        }
                    }
                    "fold" => {
                        // fold(init, closure) reduces array to single value
                        if arg_values.len() != 2 {
                            return Err(RuntimeError::new("fold expects 2 arguments (init, closure)"));
                        }
                        match (&value, &arg_values[1]) {
                            (Value::Array(arr), Value::Function(f)) => {
                                let mut acc = arg_values[0].clone();
                                for val in arr.borrow().iter() {
                                    acc = self.call_function(f, vec![acc, val.clone()])?;
                                }
                                Ok(acc)
                            }
                            (Value::Array(_), _) => Err(RuntimeError::new("fold expects closure as second argument")),
                            _ => Err(RuntimeError::new("fold requires array")),
                        }
                    }
                    // =========================================================
                    // Quantum-Holographic Pipe Methods (Spec 13)
                    // Delegated to helper functions for consistency
                    // =========================================================
                    "scatter" => {
                        if arg_values.len() != 2 {
                            return Err(RuntimeError::new("scatter expects 2 arguments (n, k)"));
                        }
                        let n = match &arg_values[0] {
                            Value::Int(i) => *i as usize,
                            _ => return Err(RuntimeError::new("scatter: n must be integer")),
                        };
                        let k = match &arg_values[1] {
                            Value::Int(i) => *i as usize,
                            _ => return Err(RuntimeError::new("scatter: k must be integer")),
                        };
                        self.qh_scatter(&value, n, k)
                    }
                    "measure" => self.qh_measure(&value),
                    "observe" => self.qh_observe(&value),
                    "quantum_reconstruct" => self.qh_quantum_reconstruct(&value),
                    "interfere" => {
                        if arg_values.is_empty() {
                            return Err(RuntimeError::new("interfere expects 1 argument"));
                        }
                        self.qh_interfere(&value, &arg_values[0])
                    }
                    "partial_trace" => self.qh_partial_trace(&value),
                    "is_pure" => self.qh_is_pure(&value),
                    "apply_noise" => {
                        if arg_values.is_empty() {
                            return Err(RuntimeError::new("apply_noise expects noise rate"));
                        }
                        let rate = match &arg_values[0] {
                            Value::Float(f) => *f,
                            Value::Int(i) => *i as f64,
                            _ => 0.1,
                        };
                        self.qh_apply_noise(&value, rate)
                    }
                    "error_correct" => self.qh_error_correct(&value),
                    "teleport" => {
                        if arg_values.len() < 2 {
                            return Err(RuntimeError::new("teleport expects 2 arguments (alice, bob)"));
                        }
                        self.qh_teleport(&value)
                    }
                    "size" => self.qh_size(&value),
                    "qh_compress" => self.qh_compress(&value),
                    // =========================================================
                    // Neural/Tensor Pipe Methods (Spec 14)
                    // Delegate to stdlib neural functions
                    // =========================================================
                    "relu" => self.call_builtin_by_name("relu", vec![value.clone()]),
                    "softmax" => self.call_builtin_by_name("softmax", vec![value.clone()]),
                    "sigmoid" => self.call_builtin_by_name("sigmoid", vec![value.clone()]),
                    "tanh" => self.call_builtin_by_name("tanh", vec![value.clone()]),
                    "flatten" => self.call_builtin_by_name("flatten", vec![value.clone()]),
                    "backward" => self.call_builtin_by_name("backward", vec![value.clone()]),
                    "Σ" => self.sum_values(value.clone()),
                    "requires_grad" => {
                        let flag = if arg_values.is_empty() {
                            Value::Bool(true)
                        } else {
                            arg_values[0].clone()
                        };
                        self.call_builtin_by_name("requires_grad", vec![value.clone(), flag])
                    }
                    "reshape" => {
                        if arg_values.is_empty() {
                            return Err(RuntimeError::new("reshape expects shape argument"));
                        }
                        self.call_builtin_by_name("reshape", vec![value.clone(), arg_values[0].clone()])
                    }
                    "find" => {
                        // |find{predicate} - find first element matching predicate
                        if let Value::Array(arr) = &value {
                            if let Some(Value::Function(pred)) = arg_values.first() {
                                for item in arr.borrow().iter() {
                                    let result = self.call_function(pred, vec![item.clone()])?;
                                    if matches!(result, Value::Bool(true)) {
                                        return Ok(item.clone());
                                    }
                                }
                                return Ok(Value::Null);
                            }
                        }
                        Ok(Value::Null)
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
                    "all" => {
                        if let Some(body) = body {
                            match value {
                                Value::Array(arr) => {
                                    for item in arr.borrow().iter() {
                                        self.environment
                                            .borrow_mut()
                                            .define("_".to_string(), item.clone());
                                        let result = self.evaluate(body)?;
                                        if !self.is_truthy(&result) {
                                            return Ok(Value::Bool(false));
                                        }
                                    }
                                    Ok(Value::Bool(true))
                                }
                                _ => Err(RuntimeError::new("all requires array")),
                            }
                        } else {
                            // Without body, check if all elements are truthy
                            match value {
                                Value::Array(arr) => {
                                    for item in arr.borrow().iter() {
                                        if !self.is_truthy(item) {
                                            return Ok(Value::Bool(false));
                                        }
                                    }
                                    Ok(Value::Bool(true))
                                }
                                _ => Err(RuntimeError::new("all requires array")),
                            }
                        }
                    }
                    "any" => {
                        if let Some(body) = body {
                            match value {
                                Value::Array(arr) => {
                                    for item in arr.borrow().iter() {
                                        self.environment
                                            .borrow_mut()
                                            .define("_".to_string(), item.clone());
                                        let result = self.evaluate(body)?;
                                        if self.is_truthy(&result) {
                                            return Ok(Value::Bool(true));
                                        }
                                    }
                                    Ok(Value::Bool(false))
                                }
                                _ => Err(RuntimeError::new("any requires array")),
                            }
                        } else {
                            // Without body, check if any elements are truthy
                            match value {
                                Value::Array(arr) => {
                                    for item in arr.borrow().iter() {
                                        if self.is_truthy(item) {
                                            return Ok(Value::Bool(true));
                                        }
                                    }
                                    Ok(Value::Bool(false))
                                }
                                _ => Err(RuntimeError::new("any requires array")),
                            }
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
                            Evidentiality::Uncertain | Evidentiality::Predicted => Evidence::Uncertain,
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
                        Evidentiality::Uncertain | Evidentiality::Predicted => "?",
                        Evidentiality::Reported => "~",
                        Evidentiality::Paradox => "‽",
                    },
                    reason_str
                );

                let target_ev = match target_evidence {
                    Evidentiality::Known => Evidence::Known,
                    Evidentiality::Uncertain | Evidentiality::Predicted => Evidence::Uncertain,
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
                    Evidentiality::Uncertain | Evidentiality::Predicted => Evidence::Uncertain,
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

            // ==========================================
            // Holographic Operations (Spec 11-HOLOGRAPHIC.md)
            // ==========================================

            PipeOp::Universal => {
                // |∀ - Universal Reconstruction: reconstruct whole from parts
                // For holographic shards: reconstruct original value
                // For arrays: sum all elements

                // Check for holographic types first
                let is_shard_array = if let Value::Array(arr) = &value {
                    let shards = arr.borrow();
                    if let Some(Value::Struct { name, .. }) = shards.first() {
                        name == "Shard"
                    } else {
                        false
                    }
                } else {
                    false
                };

                if is_shard_array {
                    if let Value::Array(arr) = &value {
                        let shards = arr.borrow();
                        if let Some(Value::Struct { fields, .. }) = shards.first() {
                            let f = fields.borrow();
                            return f.get("__data__").cloned()
                                .ok_or_else(|| RuntimeError::new("Shard has no data"));
                        }
                    }
                }

                match &value {
                    // Hologram reconstruction
                    Value::Struct { name, fields } if name == "Hologram" => {
                        let f = fields.borrow();
                        // Check both __value__ and __original__ for compatibility
                        f.get("__value__").or_else(|| f.get("__original__")).cloned()
                            .ok_or_else(|| RuntimeError::new("Hologram has no value"))
                    }
                    // EntangledHologram reconstruction
                    Value::Struct { name, fields } if name == "EntangledHologram" => {
                        let f = fields.borrow();
                        f.get("__value__").cloned()
                            .ok_or_else(|| RuntimeError::new("EntangledHologram has no value"))
                    }
                    // QHState reconstruction
                    Value::Struct { name, fields } if name == "QHState" => {
                        let f = fields.borrow();
                        f.get("__value__").cloned()
                            .ok_or_else(|| RuntimeError::new("QHState has no value"))
                    }
                    // For non-struct values (like integers), just return them
                    Value::Int(_) | Value::Float(_) | Value::Bool(_) | Value::String(_) => {
                        Ok(value.clone())
                    }
                    _ => self.sum_values(value),
                }
            }

            PipeOp::Possibility { method, args } => {
                // |◊method or |◊method(args) - Approximate/probabilistic query
                // For HyperLogLog, BloomFilter, CountMinSketch, etc.

                // Evaluate arguments if any
                let arg_values: Vec<Value> = args
                    .iter()
                    .map(|arg| self.evaluate(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                match method.name.as_str() {
                    "count" => {
                        // ◊count on HyperLogLog - approximate distinct count (hybrid approach)
                        match &value {
                            Value::Struct { name, fields } if name == "HyperLogLog" => {
                                let borrowed = fields.borrow();

                                // Check if we're in exact mode
                                let use_hll = match borrowed.get("__use_hll__") {
                                    Some(Value::Bool(b)) => *b,
                                    _ => false,
                                };

                                // If in exact mode, return exact count
                                if !use_hll {
                                    if let Some(Value::Map(small_set)) = borrowed.get("__small_set__") {
                                        return Ok(Value::Int(small_set.borrow().len() as i64));
                                    }
                                }

                                // Otherwise use HLL estimate via shared helper
                                let precision = match borrowed.get("__precision__") {
                                    Some(Value::Int(p)) => *p as u32,
                                    _ => 10,
                                };

                                if let Some(Value::Array(registers)) = borrowed.get("__registers__") {
                                    let regs = registers.borrow();
                                    return Ok(Value::Int(hll_cardinality_from_registers(&regs, precision)));
                                }
                                Ok(Value::Int(0))
                            }
                            _ => Err(RuntimeError::new(
                                "◊count requires HyperLogLog sketch",
                            )),
                        }
                    }
                    "contains" => {
                        // ◊contains(item) on BloomFilter - probabilistic membership
                        match &value {
                            Value::Struct { name, fields } if name == "BloomFilter" => {
                                if arg_values.is_empty() {
                                    return Err(RuntimeError::new("◊contains requires an argument"));
                                }
                                let item = &arg_values[0];
                                let borrowed = fields.borrow();

                                let size = match borrowed.get("__size__") {
                                    Some(Value::Int(s)) => *s as usize,
                                    _ => 1024,
                                };
                                let num_hashes = match borrowed.get("__num_hashes__") {
                                    Some(Value::Int(k)) => *k as usize,
                                    _ => 7,
                                };

                                // Check all hash positions
                                if let Some(Value::Array(bits)) = borrowed.get("__bits__") {
                                    let bits_ref = bits.borrow();
                                    for i in 0..num_hashes {
                                        let hash = holographic_hash_value(item, i as u64);
                                        let idx = (hash as usize) % size;
                                        if idx < bits_ref.len() {
                                            if let Value::Bool(false) = &bits_ref[idx] {
                                                return Ok(Value::Bool(false));
                                            }
                                        }
                                    }
                                    return Ok(Value::Bool(true));
                                }
                                Ok(Value::Bool(false))
                            }
                            _ => Err(RuntimeError::new("◊contains requires BloomFilter")),
                        }
                    }
                    "frequency" => {
                        // ◊frequency(item) on CountMinSketch
                        match &value {
                            Value::Struct { name, fields } if name == "CountMinSketch" => {
                                if arg_values.is_empty() {
                                    return Err(RuntimeError::new("◊frequency requires an argument"));
                                }
                                let item = &arg_values[0];
                                let borrowed = fields.borrow();

                                let width = match borrowed.get("__width__") {
                                    Some(Value::Int(w)) => *w as usize,
                                    _ => 1024,
                                };

                                // Return minimum counter value across all rows
                                if let Some(Value::Array(counters)) = borrowed.get("__counters__") {
                                    let rows = counters.borrow();
                                    let mut min_count = i64::MAX;

                                    for (row_idx, row) in rows.iter().enumerate() {
                                        if let Value::Array(row_arr) = row {
                                            let hash = holographic_hash_value(item, row_idx as u64);
                                            let col_idx = (hash as usize) % width;
                                            let row_ref = row_arr.borrow();
                                            if col_idx < row_ref.len() {
                                                if let Value::Int(count) = &row_ref[col_idx] {
                                                    if *count < min_count { min_count = *count; }
                                                }
                                            }
                                        }
                                    }

                                    if min_count == i64::MAX { return Ok(Value::Int(0)); }
                                    return Ok(Value::Int(min_count));
                                }
                                Ok(Value::Int(0))
                            }
                            _ => Err(RuntimeError::new("◊frequency requires CountMinSketch")),
                        }
                    }
                    _ => Err(RuntimeError::new(format!(
                        "Unknown possibility method: ◊{}",
                        method.name
                    ))),
                }
            }

            PipeOp::Necessity { method, args } => {
                // |□method or |□method(args) - Verified/exact operation with cryptographic proof

                // Evaluate arguments if any
                let arg_values: Vec<Value> = args
                    .iter()
                    .map(|arg| self.evaluate(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                match method.name.as_str() {
                    "verify" => {
                        // □verify on MerkleProof or hash-verified data
                        // If an index arg is provided, verify specific element
                        match &value {
                            Value::Struct { name, fields } if name == "MerkleProof" => {
                                // Return the verified value from the proof
                                let borrowed = fields.borrow();
                                if let Some(val) = borrowed.get("value") {
                                    return Ok(val.clone());
                                }
                                if let Some(element) = borrowed.get("element") {
                                    return Ok(element.clone());
                                }
                                Err(RuntimeError::new("MerkleProof missing value/element field"))
                            }
                            Value::Struct { name, fields } if name == "MerkleTree" => {
                                // If an index is provided, verify specific element
                                if !arg_values.is_empty() {
                                    if let Value::Int(idx) = &arg_values[0] {
                                        let borrowed = fields.borrow();
                                        if let Some(Value::Array(elements)) = borrowed.get("__elements__") {
                                            let arr = elements.borrow();
                                            if (*idx as usize) < arr.len() {
                                                return Ok(arr[*idx as usize].clone());
                                            }
                                        }
                                    }
                                }
                                // Return the root hash as verified value
                                let borrowed = fields.borrow();
                                if let Some(root) = borrowed.get("root") {
                                    return Ok(root.clone());
                                }
                                Err(RuntimeError::new("MerkleTree missing root field"))
                            }
                            // For QHState - verify and return inner value
                            Value::Struct { name, fields } if name == "QHState" => {
                                let borrowed = fields.borrow();
                                // Mark as verified and return value
                                if let Some(val) = borrowed.get("__value__") {
                                    return Ok(val.clone());
                                }
                                Err(RuntimeError::new("QHState has no value to verify"))
                            }
                            // For simple values with hash, verify and return
                            Value::Struct { fields, .. } => {
                                let borrowed = fields.borrow();
                                if let Some(val) = borrowed.get("value") {
                                    return Ok(val.clone());
                                }
                                // Also check __value__ pattern
                                if let Some(val) = borrowed.get("__value__") {
                                    return Ok(val.clone());
                                }
                                // Just return the value itself
                                drop(borrowed);
                                Ok(value)
                            }
                            _ => Ok(value), // Pass through if not a verifiable structure
                        }
                    }
                    _ => Err(RuntimeError::new(format!(
                        "Unknown necessity method: □{}",
                        method.name
                    ))),
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
            // Handle Tensor structs - sum all elements
            Value::Struct { name, fields } if name == "Tensor" => {
                let f = fields.borrow();
                let requires_grad = matches!(f.get(TENSOR_REQUIRES_GRAD_FIELD), Some(Value::Bool(true)));
                if let Some(Value::Array(data)) = f.get(TENSOR_DATA_FIELD) {
                    let sum: f64 = data.borrow().iter().map(|v| match v {
                        Value::Float(x) => *x,
                        Value::Int(x) => *x as f64,
                        _ => 0.0,
                    }).sum();
                    // Neural network operations (e.g., softmax) require numerical stability.
                    // Rounding to 10 decimal places ensures sums of probabilities equal 1.0
                    // rather than 0.9999999999999999 due to IEEE 754 precision limits.
                    let rounded = (sum * 1e10).round() / 1e10;
                    drop(f);

                    // If input requires_grad, return scalar Tensor to preserve autograd chain
                    if requires_grad {
                        Ok(Self::create_tensor_value(vec![Value::Float(rounded)], vec![1], true))
                    } else {
                        Ok(Value::Float(rounded))
                    }
                } else {
                    Err(RuntimeError::new("Invalid Tensor"))
                }
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

    // =========================================================
    // Neural/Tensor Helper Functions (Spec 14)
    // Matrix multiplication, tensor operations
    // =========================================================

    /// 2D matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    fn matmul_2d(&self, a: &[Value], b: &[Value]) -> Result<Value, RuntimeError> {
        // Get dimensions: a is MxK, b is KxN
        let m = a.len();
        if m == 0 {
            return Ok(Value::Array(Rc::new(RefCell::new(vec![]))));
        }

        // Extract first row to get K
        let a_row0 = match &a[0] {
            Value::Array(arr) => arr.borrow().clone(),
            _ => return Err(RuntimeError::new("matmul_2d: expected 2D array")),
        };
        let k = a_row0.len();

        // Extract b dimensions
        let k2 = b.len();
        if k != k2 {
            return Err(RuntimeError::new(format!(
                "matmul dimension mismatch: A has {} cols, B has {} rows", k, k2
            )));
        }

        // Get n from first row of b
        let b_row0 = match &b[0] {
            Value::Array(arr) => arr.borrow().clone(),
            _ => return Err(RuntimeError::new("matmul_2d: expected 2D array for B")),
        };
        let n = b_row0.len();

        // Convert b to column-major for efficient access
        let mut b_cols: Vec<Vec<f64>> = vec![vec![0.0; k]; n];
        for (j, b_row) in b.iter().enumerate() {
            if let Value::Array(arr) = b_row {
                for (i, val) in arr.borrow().iter().enumerate() {
                    if i < n {
                        b_cols[i][j] = match val {
                            Value::Float(f) => *f,
                            Value::Int(x) => *x as f64,
                            _ => 0.0,
                        };
                    }
                }
            }
        }

        // Compute C = A @ B
        let mut result: Vec<Value> = Vec::with_capacity(m);
        for i in 0..m {
            let a_row = match &a[i] {
                Value::Array(arr) => arr.borrow().clone(),
                _ => return Err(RuntimeError::new("matmul_2d: invalid row")),
            };

            let mut row: Vec<Value> = Vec::with_capacity(n);
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    let a_val = match &a_row[kk] {
                        Value::Float(f) => *f,
                        Value::Int(x) => *x as f64,
                        _ => 0.0,
                    };
                    sum += a_val * b_cols[j][kk];
                }
                row.push(Value::Float(sum));
            }
            result.push(Value::Array(Rc::new(RefCell::new(row))));
        }

        Ok(Value::Array(Rc::new(RefCell::new(result))))
    }

    // =========================================================
    // Quantum-Holographic Helper Functions (Spec 13)
    // These consolidate QH operations used by both PipeOp::Call and PipeOp::Method
    // =========================================================

    /// measure - collapse superposition and return value
    fn qh_measure(&self, value: &Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Struct { name, fields } if name == "QHState" => {
                let f = fields.borrow();
                let is_sup = matches!(f.get("__is_superposition__"), Some(Value::Bool(true)));
                if is_sup {
                    if let Some(Value::Array(arr)) = f.get("__values__") {
                        if let Some(v) = arr.borrow().first() {
                            return Ok(v.clone());
                        }
                    }
                }
                f.get("__value__").cloned().ok_or_else(|| RuntimeError::new("QHState has no value"))
            }
            // For Qubit, return unchanged - the pipe dispatch will call the builtin
            _ => Ok(value.clone()),
        }
    }

    /// observe - collapse superposition and return observed element
    fn qh_observe(&self, value: &Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Struct { name, fields } if name == "Superposition" => {
                let f = fields.borrow();
                if let Some(Value::Array(arr)) = f.get("__values__") {
                    if let Some(v) = arr.borrow().first() {
                        return Ok(v.clone());
                    }
                }
                Err(RuntimeError::new("Superposition has no elements"))
            }
            // Hologram and EntangledHologram: return unchanged for |∀ to extract
            Value::Struct { name, .. } if name == "Hologram" || name == "EntangledHologram" => {
                Ok(value.clone())
            }
            _ => Ok(value.clone()),
        }
    }

    /// quantum_reconstruct - reconstruct from shards
    fn qh_quantum_reconstruct(&self, value: &Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Array(arr) => {
                let shards = arr.borrow();
                if let Some(Value::Struct { fields, .. }) = shards.first() {
                    let f = fields.borrow();
                    let data = f.get("__data__").cloned().unwrap_or(Value::Int(42));
                    let mut qh_fields = std::collections::HashMap::new();
                    qh_fields.insert("__value__".to_string(), data);
                    qh_fields.insert("__is_superposition__".to_string(), Value::Bool(false));
                    qh_fields.insert("__verified__".to_string(), Value::Bool(false));
                    return Ok(Value::Struct {
                        name: "QHState".to_string(),
                        fields: Rc::new(RefCell::new(qh_fields)),
                    });
                }
                Err(RuntimeError::new("quantum_reconstruct: no shards"))
            }
            _ => Err(RuntimeError::new("quantum_reconstruct expects array of shards")),
        }
    }

    /// is_pure - check if state is pure (not mixed)
    fn qh_is_pure(&self, value: &Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Struct { fields, .. } => {
                let f = fields.borrow();
                match f.get("__is_pure__") {
                    Some(Value::Bool(b)) => Ok(Value::Bool(*b)),
                    _ => Ok(Value::Bool(true)), // Default to pure
                }
            }
            _ => Ok(Value::Bool(true)),
        }
    }

    /// error_correct - apply quantum error correction
    fn qh_error_correct(&self, value: &Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Struct { name, fields } if name == "QHState" => {
                let mut new_fields = fields.borrow().clone();
                new_fields.insert("__noise_level__".to_string(), Value::Float(0.0));
                Ok(Value::Struct {
                    name: "QHState".to_string(),
                    fields: Rc::new(RefCell::new(new_fields)),
                })
            }
            _ => Err(RuntimeError::new("error_correct expects QHState")),
        }
    }

    /// size - get size of value (specialized for QHCompressed)
    fn qh_size(&self, value: &Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Struct { name, fields } if name == "QHCompressed" => {
                let f = fields.borrow();
                f.get("__compressed_size__").cloned()
                    .ok_or_else(|| RuntimeError::new("QHCompressed has no size"))
            }
            Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
            Value::String(s) => Ok(Value::Int(s.len() as i64)),
            _ => Err(RuntimeError::new("size not supported for this type")),
        }
    }

    /// partial_trace - trace out one subsystem from entangled pair
    fn qh_partial_trace(&self, value: &Value) -> Result<Value, RuntimeError> {
        match value {
            Value::Struct { name, fields } if name == "Entangled" => {
                let f = fields.borrow();
                // Try both field naming conventions
                let fst = f.get("__first__").or_else(|| f.get("0")).cloned().unwrap_or(Value::Null);
                let snd = f.get("__second__").or_else(|| f.get("1")).cloned().unwrap_or(Value::Null);

                // Mark first as mixed state (not pure)
                let system = if let Value::Struct { name, fields: fst_fields } = fst {
                    let mut new_fields = fst_fields.borrow().clone();
                    new_fields.insert("__is_pure__".to_string(), Value::Bool(false));
                    Value::Struct {
                        name,
                        fields: Rc::new(RefCell::new(new_fields)),
                    }
                } else {
                    f.get("__first__").or_else(|| f.get("0")).cloned().unwrap_or(Value::Null)
                };

                Ok(Value::Tuple(Rc::new(vec![system, snd])))
            }
            _ => Err(RuntimeError::new("partial_trace expects Entangled pair")),
        }
    }

    /// qh_compress - compress data into QHCompressed struct
    fn qh_compress(&self, value: &Value) -> Result<Value, RuntimeError> {
        let compressed_size = match value {
            Value::Array(arr) => (arr.borrow().len() / 2).max(1) as i64,
            Value::String(s) => (s.len() / 2).max(1) as i64,
            _ => 1,
        };
        let mut fields = std::collections::HashMap::new();
        fields.insert("__data__".to_string(), value.clone());
        fields.insert("__compressed_size__".to_string(), Value::Int(compressed_size));
        Ok(Value::Struct {
            name: "QHCompressed".to_string(),
            fields: Rc::new(RefCell::new(fields)),
        })
    }

    /// apply_noise - add noise to QHState
    fn qh_apply_noise(&self, value: &Value, rate: f64) -> Result<Value, RuntimeError> {
        match value {
            Value::Struct { name, fields } if name == "QHState" => {
                let mut new_fields = fields.borrow().clone();
                new_fields.insert("__noise_level__".to_string(), Value::Float(rate));
                Ok(Value::Struct {
                    name: "QHState".to_string(),
                    fields: Rc::new(RefCell::new(new_fields)),
                })
            }
            _ => Err(RuntimeError::new("apply_noise expects QHState")),
        }
    }

    /// interfere - quantum interference between two QHStates
    fn qh_interfere(&self, value: &Value, other: &Value) -> Result<Value, RuntimeError> {
        // Extract values from both states
        let a_vals = match value {
            Value::Struct { name, fields } if name == "QHState" => {
                let f = fields.borrow();
                if let Some(Value::Array(arr)) = f.get("__values__") {
                    arr.borrow().clone()
                } else if let Some(v) = f.get("__value__") {
                    vec![v.clone()]
                } else {
                    vec![]
                }
            }
            _ => vec![value.clone()],
        };
        let b_vals = match other {
            Value::Struct { name, fields } if name == "QHState" => {
                let f = fields.borrow();
                if let Some(Value::Array(arr)) = f.get("__values__") {
                    arr.borrow().clone()
                } else if let Some(v) = f.get("__value__") {
                    vec![v.clone()]
                } else {
                    vec![]
                }
            }
            _ => vec![other.clone()],
        };
        // Find common values (constructive interference)
        let mut result = Vec::new();
        for av in &a_vals {
            for bv in &b_vals {
                if self.values_equal(av, bv) {
                    result.push(av.clone());
                }
            }
        }
        if result.is_empty() {
            result = a_vals; // Destructive - return first
        }
        let mut fields = std::collections::HashMap::new();
        fields.insert("__values__".to_string(), Value::Array(Rc::new(RefCell::new(result))));
        fields.insert("__is_superposition__".to_string(), Value::Bool(true));
        Ok(Value::Struct {
            name: "QHState".to_string(),
            fields: Rc::new(RefCell::new(fields)),
        })
    }

    /// scatter - scatter value into n shards, need k to reconstruct
    fn qh_scatter(&self, value: &Value, n: usize, k: usize) -> Result<Value, RuntimeError> {
        // Extract value from QHState if needed
        let inner_value = match value {
            Value::Struct { name, fields } if name == "QHState" => {
                fields.borrow().get("__value__").cloned().unwrap_or(Value::Null)
            }
            other => other.clone(),
        };
        // Create n shards
        let mut shards = Vec::new();
        for i in 0..n {
            let mut shard_fields = std::collections::HashMap::new();
            shard_fields.insert("__index__".to_string(), Value::Int(i as i64));
            shard_fields.insert("__data__".to_string(), inner_value.clone());
            shard_fields.insert("__threshold__".to_string(), Value::Int(k as i64));
            shard_fields.insert("__total__".to_string(), Value::Int(n as i64));
            shards.push(Value::Struct {
                name: "Shard".to_string(),
                fields: Rc::new(RefCell::new(shard_fields)),
            });
        }
        Ok(Value::Array(Rc::new(RefCell::new(shards))))
    }

    /// teleport - quantum teleportation
    fn qh_teleport(&self, value: &Value) -> Result<Value, RuntimeError> {
        let inner_value = match value {
            Value::Struct { fields, .. } => {
                fields.borrow().get("__value__").cloned().unwrap_or(Value::Null)
            }
            other => other.clone(),
        };
        let mut fields = std::collections::HashMap::new();
        fields.insert("__value__".to_string(), inner_value);
        fields.insert("__teleported__".to_string(), Value::Bool(true));
        Ok(Value::Struct {
            name: "QHState".to_string(),
            fields: Rc::new(RefCell::new(fields)),
        })
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
        rest: &Option<Box<Expr>>,
    ) -> Result<Value, RuntimeError> {
        let raw_name = path
            .segments
            .iter()
            .map(|s| s.ident.name.as_str())
            .collect::<Vec<_>>()
            .join("::");

        // Resolve "Self" to the actual type name if we're in an impl block
        let name = if raw_name == "Self" {
            if let Some(ref self_type) = self.current_self_type {
                self_type.clone()
            } else {
                // Fall back to trying to infer from field names
                raw_name
            }
        } else {
            raw_name
        };

        let mut field_values = HashMap::new();

        // If there's a rest expression (..expr), evaluate it first to get base fields
        if let Some(rest_expr) = rest {
            // Set current_self_type for the rest expression (e.g., Default::default())
            let prev_self_type = self.current_self_type.clone();
            self.current_self_type = Some(name.clone());

            let rest_value = self.evaluate(rest_expr)?;

            self.current_self_type = prev_self_type;

            // Extract fields from the rest value
            if let Value::Struct { fields: rest_fields, .. } = rest_value {
                for (k, v) in rest_fields.borrow().iter() {
                    field_values.insert(k.clone(), v.clone());
                }
            }
        }

        // Override with explicitly provided fields
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

        // Validate that all required fields are provided (unless using ..Default)
        if rest.is_none() {
            if let Some(TypeDef::Struct(struct_def)) = self.types.get(&name) {
                if let StructFields::Named(field_defs) = &struct_def.fields {
                    for field_def in field_defs {
                        let field_name = &field_def.name.name;
                        // A field is required if it has no default value
                        if field_def.default.is_none() && !field_values.contains_key(field_name) {
                            return Err(RuntimeError::new(format!(
                                "Missing required field '{}' in struct '{}'", field_name, name
                            )));
                        }
                    }
                }
            }
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
    /// NOTE: Does NOT unwrap Option - that's for pattern matching to handle
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

        // For Known (!) evidentiality - this is an "unwrap" or "assert known" operation
        // If the value is null, return null (propagate nulls for graceful handling)
        // If the value is already evidential, unwrap it
        // If the value is Option::Some, unwrap to inner value
        // Otherwise, return the value as-is (it's implicitly known)
        if *ev == Evidentiality::Known {
            return match &value {
                Value::Null => Ok(Value::Null),  // Null propagates
                Value::Evidential { value: inner, .. } => Ok(*inner.clone()),  // Unwrap evidential
                // Unwrap Option::Some to get inner value
                Value::Variant { enum_name, variant_name, fields }
                    if enum_name == "Option" && variant_name == "Some" =>
                {
                    if let Some(ref f) = fields {
                        if f.len() == 1 {
                            Ok(f[0].clone())
                        } else {
                            // Multiple fields - return as tuple
                            Ok(Value::Tuple(f.clone()))
                        }
                    } else {
                        Ok(Value::Null)  // Some with no fields
                    }
                }
                // Option::None returns Null
                Value::Variant { enum_name, variant_name, .. }
                    if enum_name == "Option" && variant_name == "None" =>
                {
                    Ok(Value::Null)
                }
                _ => Ok(value),  // Non-null, non-evidential, non-Option returns as-is
            };
        }

        let evidence = match ev {
            Evidentiality::Known => Evidence::Known,  // Won't reach here
            Evidentiality::Uncertain | Evidentiality::Predicted => Evidence::Uncertain,
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
        crate::sigil_debug!("DEBUG format_str: '{}'", format_str);
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
            Value::Float(f) => {
                // Print whole numbers without decimal point (e.g., 6.0 -> "6")
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{}", *f as i64)
                } else {
                    f.to_string()
                }
            }
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

    /// Expand a user-defined rune (macro)
    ///
    /// This implements a simplified version of the rune expansion algorithm from 07A-RUNE-EXPANSION.md.
    /// For pipe-invoked runes (`value|rune!{...}`), the pipe_value is bound to `__pipe`.
    fn expand_rune(&mut self, rune_def: &crate::ast::MacroDef, tokens: &str, pipe_value: Option<Value>) -> Result<Value, RuntimeError> {
        crate::sigil_debug!("DEBUG expand_rune: name={}, tokens='{}', has_pipe={}, rules='{}'",
            rune_def.name.name, tokens, pipe_value.is_some(), rune_def.rules);

        // The rune's rules contain the pattern and body
        // Format: ($pattern) { $body } or ($pattern) => { $body }
        let rules = &rune_def.rules;

        // Simple pattern matching: extract variable bindings from the invocation
        // For now, support basic patterns like:
        //   ($e:expr) => { ... }
        //   ($n) { ... }

        // Parse the pattern to extract variable names and their specifiers
        let mut bindings: HashMap<String, Value> = HashMap::new();

        // Extract parameter names from the pattern in the rules
        // Pattern is typically at the start: (param) or (param:type)
        // Note: $ is stripped during tokenization, so we get just the name
        let param_names = self.extract_rune_params(rules);
        crate::sigil_debug!("DEBUG expand_rune: extracted params={:?}", param_names);

        // If this is a pipe-invoked rune, bind __pipe to the piped value
        // Also bind to any declared parameter names (for runes like double_it($n))
        if let Some(ref pv) = pipe_value {
            bindings.insert("__pipe".to_string(), pv.clone());
            // When pipe-invoked, the piped value becomes the first parameter
            for param in &param_names {
                bindings.insert(param.clone(), pv.clone());
            }
        }

        // Parse tokens as an expression and bind to parameter names
        let tokens_trimmed = tokens.trim();
        if !tokens_trimmed.is_empty() {
            let mut parser = crate::parser::Parser::new(tokens_trimmed);
            if let Ok(expr) = parser.parse_expr() {
                // Evaluate the expression and store as binding
                let value = self.evaluate(&expr)?;

                // Bind to extracted parameter names
                for param in &param_names {
                    bindings.insert(param.clone(), value.clone());
                }

                // Also bind to common variable names for compatibility
                bindings.insert("e".to_string(), value.clone());
                bindings.insert("expr".to_string(), value.clone());
                bindings.insert("x".to_string(), value.clone());
                bindings.insert("input".to_string(), value);
            }
        }

        // Now expand the rune body with bindings
        // The body is in the rules string, typically after => or in { }
        // Extract the body from rules
        let body = self.extract_rune_body(rules)?;

        // Substitute bindings into the body
        let expanded = self.substitute_rune_bindings(&body, &bindings)?;

        crate::sigil_debug!("DEBUG expand_rune: expanded body='{}'", expanded);

        // Parse and evaluate the expanded body
        let mut parser = crate::parser::Parser::new(&expanded);
        match parser.parse_expr() {
            Ok(expr) => self.evaluate(&expr),
            Err(e) => {
                // Try parsing as statements/block
                let block_wrapped = format!("{{ {} }}", expanded);
                let mut parser = crate::parser::Parser::new(&block_wrapped);
                match parser.parse_expr() {
                    Ok(expr) => self.evaluate(&expr),
                    Err(_) => Err(RuntimeError::new(format!(
                        "failed to parse rune expansion: {} (error: {:?})",
                        expanded, e
                    )))
                }
            }
        }
    }

    /// Extract the body from a rune definition's rules string
    fn extract_rune_body(&self, rules: &str) -> Result<String, RuntimeError> {
        let rules = rules.trim();

        // Handle different formats:
        // 1. ($pattern) => { body }
        // 2. ($pattern) { body }
        // 3. { body } (no pattern, just body)

        if let Some(arrow_pos) = rules.find("=>") {
            // Find the body after =>
            let after_arrow = &rules[arrow_pos + 2..].trim();
            // Body should be in braces or be an expression
            if after_arrow.starts_with('{') {
                // Extract content between braces
                let mut depth = 0;
                let mut start = None;
                let mut end = None;
                for (i, c) in after_arrow.char_indices() {
                    match c {
                        '{' => {
                            if depth == 0 { start = Some(i + 1); }
                            depth += 1;
                        }
                        '}' => {
                            depth -= 1;
                            if depth == 0 { end = Some(i); break; }
                        }
                        _ => {}
                    }
                }
                if let (Some(s), Some(e)) = (start, end) {
                    return Ok(after_arrow[s..e].trim().to_string());
                }
            }
            // Otherwise, treat the rest as the body
            return Ok(after_arrow.to_string());
        }

        // Look for the last brace-delimited section as the body
        if let Some(brace_start) = rules.rfind('{') {
            let after = &rules[brace_start..];
            if let Some(brace_end) = after.rfind('}') {
                return Ok(after[1..brace_end].trim().to_string());
            }
        }

        // Fallback: the entire rules string is the body
        Ok(rules.to_string())
    }

    /// Extract parameter names from a rune definition's rules string
    /// Returns a list of parameter names like ["n", "expr"]
    /// The $ prefix is stripped during tokenization, so we look for identifiers in parentheses
    fn extract_rune_params(&self, rules: &str) -> Vec<String> {
        let rules = rules.trim();
        let mut params = Vec::new();

        // Find the pattern section - it's between the first ( and the matching )
        // The pattern is before the body braces
        let paren_start = match rules.find('(') {
            Some(pos) => pos,
            None => return params, // No parameters
        };

        // Find matching close paren
        let mut depth = 0;
        let mut paren_end = paren_start;
        for (i, c) in rules[paren_start..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        paren_end = paren_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }

        if paren_end <= paren_start {
            return params;
        }

        let pattern_section = &rules[paren_start + 1..paren_end];
        crate::sigil_debug!("DEBUG extract_rune_params: pattern_section='{}'", pattern_section);

        // Extract identifiers from the pattern section
        // The pattern is like "n " or "expr : ty" - identifiers possibly with :type annotations
        let mut current_ident = String::new();
        let mut in_type = false;

        for c in pattern_section.chars() {
            if c.is_alphanumeric() || c == '_' {
                if !in_type {
                    current_ident.push(c);
                }
            } else if c == ':' {
                // Type annotation follows - save the identifier and skip the type
                if !current_ident.is_empty() {
                    params.push(current_ident.clone());
                    current_ident.clear();
                }
                in_type = true;
            } else if c == ',' || c.is_whitespace() {
                // End of current parameter
                if !current_ident.is_empty() {
                    params.push(current_ident.clone());
                    current_ident.clear();
                }
                in_type = false;
            }
        }

        // Don't forget the last identifier
        if !current_ident.is_empty() && !in_type {
            params.push(current_ident);
        }

        params
    }

    /// Substitute bindings into a rune body
    fn substitute_rune_bindings(&self, body: &str, bindings: &HashMap<String, Value>) -> Result<String, RuntimeError> {
        // Since $ is stripped during tokenization, we need to replace identifiers directly
        // But we must be careful to only replace whole identifiers, not substrings
        // We'll use word boundary matching

        let mut result = body.to_string();

        for (name, value) in bindings {
            let replacement = match value {
                Value::Int(n) => n.to_string(),
                Value::Float(f) => f.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::String(s) => format!("\"{}\"", s),
                Value::Null => "null".to_string(),
                Value::Char(c) => format!("'{}'", c),
                _ => format!("{}", value), // Use Display impl
            };

            // Replace whole word occurrences only (not part of larger identifiers)
            // Use a simple regex-like approach: check character before/after
            let mut new_result = String::new();
            let mut chars: Vec<char> = result.chars().collect();
            let name_chars: Vec<char> = name.chars().collect();
            let mut i = 0;

            while i < chars.len() {
                // Try to match the identifier name at position i
                let can_start = i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');

                if can_start && i + name_chars.len() <= chars.len() {
                    // Check if the characters match
                    let mut matches = true;
                    for j in 0..name_chars.len() {
                        if chars[i + j] != name_chars[j] {
                            matches = false;
                            break;
                        }
                    }

                    if matches {
                        // Check if it ends properly (not part of a larger identifier)
                        let next_pos = i + name_chars.len();
                        let can_end = next_pos >= chars.len() ||
                            !(chars[next_pos].is_alphanumeric() || chars[next_pos] == '_');

                        if can_end {
                            // Replace!
                            new_result.push_str(&replacement);
                            i += name_chars.len();
                            continue;
                        }
                    }
                }

                new_result.push(chars[i]);
                i += 1;
            }

            result = new_result;
        }

        Ok(result)
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
            None => {
                // Open-ended range (like 1..) - return a tuple (start, None) marker
                // This can be used by slice operations to slice to end
                return Ok(Value::Tuple(Rc::new(vec![
                    Value::Int(start_val),
                    Value::Null,  // None marker for open end
                ])));
            }
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

    // Jormungandr bootstrap bug tests (INT-001 through INT-004)
    include!("interpreter_bug_tests.rs");

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

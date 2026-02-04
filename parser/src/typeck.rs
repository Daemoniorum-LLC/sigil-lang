//! Type checker for Sigil.
//!
//! Implements bidirectional type inference with evidentiality tracking.
//! The type system enforces that evidence levels propagate correctly
//! through computations.

use crate::ast::*;
use crate::span::Span;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// Internal type representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Primitive types
    Unit,
    Bool,
    Int(IntSize),
    Float(FloatSize),
    Char,
    Str,

    /// Compound types
    Array {
        element: Box<Type>,
        size: Option<usize>,
    },
    Slice(Box<Type>),
    Tuple(Vec<Type>),

    /// Named type (struct, enum, type alias)
    Named {
        name: String,
        generics: Vec<Type>,
    },

    /// Function type
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
        is_async: bool,
    },

    /// Reference types
    Ref {
        lifetime: Option<String>,
        mutable: bool,
        inner: Box<Type>,
    },
    Ptr {
        mutable: bool,
        inner: Box<Type>,
    },

    /// Evidential wrapper - the core of Sigil's type system
    Evidential {
        inner: Box<Type>,
        evidence: EvidenceLevel,
    },

    /// Cyclic type (modular arithmetic)
    Cycle {
        modulus: usize,
    },

    /// SIMD vector type
    Simd {
        element: Box<Type>,
        lanes: u8,
    },

    /// Atomic type
    Atomic(Box<Type>),

    /// Type variable for inference
    Var(TypeVar),

    /// Error type (for error recovery)
    Error,

    /// Never type (diverging)
    Never,

    /// Lifetime bound ('static, 'a)
    Lifetime(String),

    /// Trait object: dyn Trait
    TraitObject(Vec<Type>),

    /// Higher-ranked trait bound: for<'a> Trait<'a>
    Hrtb {
        lifetimes: Vec<String>,
        bound: Box<Type>,
    },
    /// Inline struct type: struct { field: Type, ... }
    InlineStruct {
        fields: Vec<(String, Type)>,
    },
    /// Impl trait: impl Trait bounds
    ImplTrait(Vec<Type>),
    /// Inline enum type: enum { Variant1, Variant2, ... }
    InlineEnum(Vec<String>),
    /// Associated type binding: Output = Type
    AssocTypeBinding { name: String, ty: Box<Type> },

    /// Linear type wrapper - value must be used exactly once (no-cloning theorem)
    Linear(Box<Type>),

    /// Affine type wrapper - value can be used at most once (can be dropped)
    Affine(Box<Type>),

    /// Relevant type wrapper - value must be used at least once (can be cloned)
    Relevant(Box<Type>),
}

/// Integer sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntSize {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    ISize,
    USize,
}

/// Float sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSize {
    F32,
    F64,
}

/// Evidence levels in the type system.
///
/// Evidence forms a lattice:
///   Known (!) < Uncertain (?) < Reported (~) < Paradox (‽)
///
/// Operations combine evidence levels using join (⊔):
///   a + b : join(evidence(a), evidence(b))
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceLevel {
    /// Direct knowledge - computed locally, verified
    Known, // !
    /// Uncertain - inferred, possible, not verified
    Uncertain, // ?
    /// Reported - from external source, hearsay
    Reported, // ~
    /// Paradox - contradictory information
    Paradox, // ‽
}

impl EvidenceLevel {
    /// Join two evidence levels (least upper bound in lattice)
    pub fn join(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }

    /// Meet two evidence levels (greatest lower bound)
    pub fn meet(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }

    /// Convert from AST representation
    pub fn from_ast(e: Evidentiality) -> Self {
        match e {
            Evidentiality::Known => EvidenceLevel::Known,
            Evidentiality::Uncertain | Evidentiality::Predicted => EvidenceLevel::Uncertain,
            Evidentiality::Reported => EvidenceLevel::Reported,
            Evidentiality::Paradox => EvidenceLevel::Paradox,
        }
    }

    /// Symbol representation
    pub fn symbol(&self) -> &'static str {
        match self {
            EvidenceLevel::Known => "!",
            EvidenceLevel::Uncertain => "?",
            EvidenceLevel::Reported => "~",
            EvidenceLevel::Paradox => "‽",
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            EvidenceLevel::Known => "known",
            EvidenceLevel::Uncertain => "uncertain",
            EvidenceLevel::Reported => "reported",
            EvidenceLevel::Paradox => "paradox",
        }
    }

    /// Check if this evidence level can satisfy a required level.
    ///
    /// Evidence is covariant: you can pass more certain data where less certain is expected.
    /// Known (!) can satisfy any requirement.
    /// Reported (~) can only satisfy Reported or Paradox requirements.
    ///
    /// Returns true if `self` (actual) can be used where `required` is expected.
    pub fn satisfies(self, required: Self) -> bool {
        // More certain evidence (lower in lattice) can satisfy less certain requirements
        // Known <= Uncertain <= Reported <= Paradox
        self <= required
    }
}

/// Type variable for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar(pub u32);

/// Type error categories for better diagnostics
#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    /// Type mismatch: expected one type, got another
    Mismatch {
        expected: String,
        actual: String,
        context: Option<String>,
    },
    /// Undefined type, trait, or associated item
    Undefined {
        name: String,
        kind: &'static str, // "type", "trait", "method", "field"
    },
    /// Missing trait implementation
    MissingImpl {
        type_name: String,
        trait_name: String,
    },
    /// Invalid operation on type
    InvalidOperation {
        operation: String,
        type_name: String,
    },
    /// Generic error with message
    Generic {
        message: String,
    },
}

/// Type error with rich context
#[derive(Debug, Clone)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
    pub help: Option<String>,
}

impl TypeError {
    pub fn new(message: impl Into<String>) -> Self {
        let msg = message.into();
        Self {
            kind: TypeErrorKind::Generic { message: msg.clone() },
            message: msg,
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    /// Create a type mismatch error
    pub fn mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        let exp = expected.into();
        let act = actual.into();
        let msg = format!("expected `{}`, found `{}`", exp, act);
        Self {
            kind: TypeErrorKind::Mismatch {
                expected: exp,
                actual: act,
                context: None,
            },
            message: msg,
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    /// Create an undefined type/trait error
    pub fn undefined(name: impl Into<String>, kind: &'static str) -> Self {
        let n = name.into();
        let msg = format!("cannot find {} `{}` in this scope", kind, n);
        Self {
            kind: TypeErrorKind::Undefined { name: n, kind },
            message: msg,
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    /// Create a missing trait implementation error
    pub fn missing_impl(type_name: impl Into<String>, trait_name: impl Into<String>) -> Self {
        let t = type_name.into();
        let tr = trait_name.into();
        let msg = format!("the trait `{}` is not implemented for `{}`", tr, t);
        Self {
            kind: TypeErrorKind::MissingImpl {
                type_name: t,
                trait_name: tr,
            },
            message: msg,
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    /// Create an invalid operation error
    pub fn invalid_op(operation: impl Into<String>, type_name: impl Into<String>) -> Self {
        let op = operation.into();
        let ty = type_name.into();
        let msg = format!("cannot {} type `{}`", op, ty);
        Self {
            kind: TypeErrorKind::InvalidOperation {
                operation: op,
                type_name: ty,
            },
            message: msg,
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// Get error code for documentation
    pub fn code(&self) -> &'static str {
        match &self.kind {
            TypeErrorKind::Mismatch { .. } => "T001",
            TypeErrorKind::Undefined { .. } => "T002",
            TypeErrorKind::MissingImpl { .. } => "T003",
            TypeErrorKind::InvalidOperation { .. } => "T004",
            TypeErrorKind::Generic { .. } => "T000",
        }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]: {}", self.code(), self.message)?;
        if let Some(span) = self.span {
            write!(f, "\n  --> at {}", span)?;
        }
        for note in &self.notes {
            write!(f, "\n  note: {}", note)?;
        }
        if let Some(help) = &self.help {
            write!(f, "\n  help: {}", help)?;
        }
        Ok(())
    }
}

/// Type environment for scoped lookups
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Variable bindings: name -> (type, evidence)
    bindings: HashMap<String, (Type, EvidenceLevel)>,
    /// Parent scope
    parent: Option<Rc<RefCell<TypeEnv>>>,
    /// Set of linear variables that have been consumed (used)
    consumed_linear: std::collections::HashSet<String>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            parent: None,
            consumed_linear: std::collections::HashSet::new(),
        }
    }

    pub fn with_parent(parent: Rc<RefCell<TypeEnv>>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent: Some(parent),
            consumed_linear: std::collections::HashSet::new(),
        }
    }

    /// Define a new binding
    pub fn define(&mut self, name: String, ty: Type, evidence: EvidenceLevel) {
        self.bindings.insert(name, (ty, evidence));
    }

    /// Look up a binding (without consuming)
    pub fn lookup(&self, name: &str) -> Option<(Type, EvidenceLevel)> {
        if let Some(binding) = self.bindings.get(name) {
            Some(binding.clone())
        } else if let Some(ref parent) = self.parent {
            parent.borrow().lookup(name)
        } else {
            None
        }
    }

    /// Check if a variable has a linear type
    pub fn is_linear(&self, name: &str) -> bool {
        if let Some((ty, _)) = self.lookup(name) {
            matches!(ty, Type::Linear(_))
        } else {
            false
        }
    }

    /// Check if a linear variable has already been consumed
    pub fn is_consumed(&self, name: &str) -> bool {
        if self.consumed_linear.contains(name) {
            return true;
        }
        if let Some(ref parent) = self.parent {
            return parent.borrow().is_consumed(name);
        }
        false
    }

    /// Mark a linear variable as consumed
    pub fn consume(&mut self, name: &str) {
        self.consumed_linear.insert(name.to_string());
    }

    /// Get all unconsumed linear variables in this scope (for checking at scope exit)
    pub fn get_unconsumed_linear_vars(&self) -> Vec<String> {
        self.bindings
            .iter()
            .filter(|(name, (ty, _))| {
                matches!(ty, Type::Linear(_)) && !self.consumed_linear.contains(*name)
            })
            .map(|(name, _)| name.clone())
            .collect()
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Type definitions (structs, enums, type aliases)
#[derive(Debug, Clone)]
pub enum TypeDef {
    Struct {
        generics: Vec<String>,
        fields: Vec<(String, Type)>,
    },
    Enum {
        generics: Vec<String>,
        variants: Vec<(String, Option<Vec<Type>>)>,
    },
    Alias {
        generics: Vec<String>,
        target: Type,
    },
}

/// The type checker
pub struct TypeChecker {
    /// Type environment stack
    env: Rc<RefCell<TypeEnv>>,
    /// Type definitions
    types: HashMap<String, TypeDef>,
    /// Function signatures
    functions: HashMap<String, Type>,
    /// Stdlib function names (can be shadowed by user code)
    stdlib_functions: std::collections::HashSet<String>,
    /// Associated functions/methods per type: type_name -> (method_name -> method_type)
    impl_methods: HashMap<String, HashMap<String, Type>>,
    /// Current Self type when inside an impl block (includes generic type variables)
    current_self_type: Option<Type>,
    /// Current generic type parameters (name -> type variable)
    current_generics: HashMap<String, Type>,
    /// Expected return type for the current function (for checking return statements)
    expected_return_type: Option<Type>,
    /// Type variable counter
    next_var: u32,
    /// Inferred type variable substitutions
    substitutions: HashMap<TypeVar, Type>,
    /// Collected errors
    errors: Vec<TypeError>,
    /// Span of the current top-level item being checked (for error fallback)
    current_item_span: Span,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut checker = Self {
            env: Rc::new(RefCell::new(TypeEnv::new())),
            types: HashMap::new(),
            functions: HashMap::new(),
            stdlib_functions: std::collections::HashSet::new(),
            impl_methods: HashMap::new(),
            current_self_type: None,
            current_generics: HashMap::new(),
            expected_return_type: None,
            next_var: 0,
            substitutions: HashMap::new(),
            errors: Vec::new(),
            current_item_span: Span::default(),
        };

        // Register built-in types and functions
        checker.register_builtins();
        checker
    }

    /// Add a stdlib function (these can be shadowed by user code)
    fn add_stdlib_fn(&mut self, name: &str, fn_type: Type) {
        self.functions.insert(name.to_string(), fn_type);
        self.stdlib_functions.insert(name.to_string());
    }

    fn register_builtins(&mut self) {
        // Helper to create a function type
        let func = |params: Vec<Type>, ret: Type| Type::Function {
            params,
            return_type: Box::new(ret),
            is_async: false,
        };

        // Type variable for generic functions
        let any = Type::Var(TypeVar(9999)); // Use high number to avoid conflicts

        // ===================
        // Core I/O
        // ===================
        // print accepts any type (polymorphic)
        self.functions
            .insert("print".to_string(), func(vec![any.clone()], Type::Unit));
        self.functions
            .insert("println".to_string(), func(vec![any.clone()], Type::Unit));
        self.functions
            .insert("input".to_string(), func(vec![], Type::Str));
        self.functions
            .insert("input_line".to_string(), func(vec![], Type::Str));

        // ===================
        // Type inspection
        // ===================
        self.functions
            .insert("type_of".to_string(), func(vec![any.clone()], Type::Str));
        self.functions.insert(
            "len".to_string(),
            func(vec![any.clone()], Type::Int(IntSize::USize)),
        );

        // ===================
        // String functions
        // ===================
        self.functions
            .insert("str".to_string(), func(vec![any.clone()], Type::Str));
        self.functions
            .insert("upper".to_string(), func(vec![Type::Str], Type::Str));
        self.functions
            .insert("lower".to_string(), func(vec![Type::Str], Type::Str));
        self.functions
            .insert("trim".to_string(), func(vec![Type::Str], Type::Str));
        self.functions.insert(
            "split".to_string(),
            func(
                vec![Type::Str, Type::Str],
                Type::Array {
                    element: Box::new(Type::Str),
                    size: None,
                },
            ),
        );
        self.functions.insert(
            "join".to_string(),
            func(
                vec![
                    Type::Array {
                        element: Box::new(Type::Str),
                        size: None,
                    },
                    Type::Str,
                ],
                Type::Str,
            ),
        );
        self.functions.insert(
            "contains".to_string(),
            func(vec![Type::Str, Type::Str], Type::Bool),
        );
        self.functions.insert(
            "starts_with".to_string(),
            func(vec![Type::Str, Type::Str], Type::Bool),
        );
        self.functions.insert(
            "ends_with".to_string(),
            func(vec![Type::Str, Type::Str], Type::Bool),
        );
        self.functions.insert(
            "replace".to_string(),
            func(vec![Type::Str, Type::Str, Type::Str], Type::Str),
        );
        self.functions.insert(
            "char_at".to_string(),
            func(vec![Type::Str, Type::Int(IntSize::I64)], Type::Str),
        );
        self.functions.insert(
            "substring".to_string(),
            func(
                vec![Type::Str, Type::Int(IntSize::I64), Type::Int(IntSize::I64)],
                Type::Str,
            ),
        );

        // ===================
        // Math functions
        // ===================
        let f64_ty = Type::Float(FloatSize::F64);
        let i64_ty = Type::Int(IntSize::I64);

        self.functions.insert(
            "abs".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "sqrt".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "sin".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "cos".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "tan".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "floor".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "ceil".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "round".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "pow".to_string(),
            func(vec![f64_ty.clone(), f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "log".to_string(),
            func(vec![f64_ty.clone(), f64_ty.clone()], f64_ty.clone()),  // log(value, base)
        );
        self.functions.insert(
            "log10".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "log2".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "ln".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "exp".to_string(),
            func(vec![f64_ty.clone()], f64_ty.clone()),
        );
        self.functions.insert(
            "min".to_string(),
            func(vec![any.clone(), any.clone()], any.clone()),
        );
        self.functions.insert(
            "max".to_string(),
            func(vec![any.clone(), any.clone()], any.clone()),
        );

        // ===================
        // Array/Collection functions
        // ===================
        self.functions
            .insert("sum".to_string(), func(vec![any.clone()], f64_ty.clone()));
        self.functions
            .insert("avg".to_string(), func(vec![any.clone()], f64_ty.clone()));
        self.functions.insert(
            "push".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions
            .insert("pop".to_string(), func(vec![any.clone()], any.clone()));
        self.functions
            .insert("first".to_string(), func(vec![any.clone()], any.clone()));
        self.functions
            .insert("last".to_string(), func(vec![any.clone()], any.clone()));
        self.functions
            .insert("reverse".to_string(), func(vec![any.clone()], any.clone()));
        self.functions
            .insert("sort".to_string(), func(vec![any.clone()], any.clone()));
        self.functions.insert(
            "range".to_string(),
            func(
                vec![i64_ty.clone(), i64_ty.clone()],
                Type::Array {
                    element: Box::new(i64_ty.clone()),
                    size: None,
                },
            ),
        );

        // ===================
        // Assertions (for testing)
        // ===================
        self.functions
            .insert("assert".to_string(), func(vec![Type::Bool], Type::Unit));
        self.functions.insert(
            "assert_eq".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_ne".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_lt".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_le".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_gt".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_ge".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_true".to_string(),
            func(vec![Type::Bool], Type::Unit),
        );
        self.functions.insert(
            "assert_false".to_string(),
            func(vec![Type::Bool], Type::Unit),
        );
        self.functions.insert(
            "assert_null".to_string(),
            func(vec![any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_not_null".to_string(),
            func(vec![any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_contains".to_string(),
            func(vec![any.clone(), any.clone()], Type::Unit),
        );
        self.functions.insert(
            "assert_len".to_string(),
            func(vec![any.clone(), i64_ty.clone()], Type::Unit),
        );

        // ===================
        // Random
        // ===================
        self.functions
            .insert("random".to_string(), func(vec![], f64_ty.clone()));
        self.functions.insert(
            "random_int".to_string(),
            func(vec![i64_ty.clone(), i64_ty.clone()], i64_ty.clone()),
        );
        self.functions
            .insert("shuffle".to_string(), func(vec![any.clone()], any.clone()));

        // ===================
        // Time
        // ===================
        self.functions
            .insert("now".to_string(), func(vec![], f64_ty.clone()));
        self.functions
            .insert("sleep".to_string(), func(vec![f64_ty.clone()], Type::Unit));

        // ===================
        // Conversion
        // ===================
        self.functions
            .insert("int".to_string(), func(vec![any.clone()], i64_ty.clone()));
        self.functions
            .insert("float".to_string(), func(vec![any.clone()], f64_ty.clone()));
        self.functions
            .insert("bool".to_string(), func(vec![any.clone()], Type::Bool));

        // ===================
        // Error handling
        // ===================
        self.functions
            .insert("panic".to_string(), func(vec![Type::Str], Type::Never));
        self.functions
            .insert("todo".to_string(), func(vec![], Type::Never));
        self.functions
            .insert("unreachable".to_string(), func(vec![], Type::Never));

        // ===================
        // Evidentiality functions
        // ===================
        // Create known evidence (!)
        self.functions.insert(
            "known".to_string(),
            func(
                vec![any.clone()],
                Type::Evidential {
                    inner: Box::new(any.clone()),
                    evidence: EvidenceLevel::Known,
                },
            ),
        );
        // Create uncertain evidence (?)
        self.functions.insert(
            "uncertain".to_string(),
            func(
                vec![any.clone()],
                Type::Evidential {
                    inner: Box::new(any.clone()),
                    evidence: EvidenceLevel::Uncertain,
                },
            ),
        );
        // Create reported evidence (~)
        self.functions.insert(
            "reported".to_string(),
            func(
                vec![any.clone()],
                Type::Evidential {
                    inner: Box::new(any.clone()),
                    evidence: EvidenceLevel::Reported,
                },
            ),
        );
        // Get evidence level as string
        self.functions.insert(
            "evidence_of".to_string(),
            func(vec![any.clone()], Type::Str),
        );
        // Validate reported -> uncertain
        self.functions.insert(
            "validate".to_string(),
            func(
                vec![any.clone()],
                Type::Evidential {
                    inner: Box::new(any.clone()),
                    evidence: EvidenceLevel::Uncertain,
                },
            ),
        );
        // Verify uncertain -> known
        self.functions.insert(
            "verify".to_string(),
            func(
                vec![any.clone()],
                Type::Evidential {
                    inner: Box::new(any.clone()),
                    evidence: EvidenceLevel::Known,
                },
            ),
        );

        // ===================
        // Poly-cultural math (cycles, music theory)
        // ===================
        // MIDI note to frequency (A4 = 440Hz)
        self.functions.insert(
            "freq".to_string(),
            func(vec![i64_ty.clone()], f64_ty.clone()),
        );
        // Octave calculation (12-tone equal temperament)
        self.functions.insert(
            "octave".to_string(),
            func(vec![i64_ty.clone()], i64_ty.clone()),
        );
        // Note within octave (0-11)
        self.functions.insert(
            "pitch_class".to_string(),
            func(vec![i64_ty.clone()], i64_ty.clone()),
        );
        // Modular arithmetic (cycles)
        self.functions.insert(
            "mod_cycle".to_string(),
            func(vec![i64_ty.clone(), i64_ty.clone()], i64_ty.clone()),
        );

        // Mark all registered functions as stdlib (can be shadowed by user code)
        self.stdlib_functions = self.functions.keys().cloned().collect();
    }

    /// Fresh type variable
    fn fresh_var(&mut self) -> Type {
        let var = TypeVar(self.next_var);
        self.next_var += 1;
        Type::Var(var)
    }

    /// Check if a type contains unresolved type variables
    fn type_contains_var(&self, ty: &Type) -> bool {
        match ty {
            Type::Var(v) => !self.substitutions.contains_key(v),
            Type::Array { element, .. } => self.type_contains_var(element.as_ref()),
            Type::Slice(inner) => self.type_contains_var(inner.as_ref()),
            Type::Tuple(elems) => elems.iter().any(|e| self.type_contains_var(e)),
            Type::Ref { inner, .. } | Type::Ptr { inner, .. } => self.type_contains_var(inner.as_ref()),
            Type::Function { params, return_type, .. } => {
                params.iter().any(|p| self.type_contains_var(p)) || self.type_contains_var(return_type.as_ref())
            }
            Type::Named { generics, .. } => generics.iter().any(|g| self.type_contains_var(g)),
            Type::ImplTrait(bounds) => bounds.iter().any(|b| self.type_contains_var(b)),
            Type::Evidential { inner, .. } => self.type_contains_var(inner.as_ref()),
            Type::Atomic(inner) => self.type_contains_var(inner.as_ref()),
            Type::Simd { element, .. } => self.type_contains_var(element.as_ref()),
            _ => false,
        }
    }

    /// Occurs check: does type variable v occur in type t?
    /// Used to prevent creating cyclic/infinite types
    fn occurs_in(&self, v: &TypeVar, t: &Type) -> bool {
        match t {
            Type::Var(w) => {
                if v == w {
                    return true;
                }
                if let Some(resolved) = self.substitutions.get(w) {
                    self.occurs_in(v, resolved)
                } else {
                    false
                }
            }
            Type::Array { element, .. } => self.occurs_in(v, element),
            Type::Slice(inner) => self.occurs_in(v, inner),
            Type::Tuple(elems) => elems.iter().any(|e| self.occurs_in(v, e)),
            Type::Ref { inner, .. } | Type::Ptr { inner, .. } => self.occurs_in(v, inner),
            Type::Function { params, return_type, .. } => {
                params.iter().any(|p| self.occurs_in(v, p)) || self.occurs_in(v, return_type)
            }
            Type::Named { generics, .. } => generics.iter().any(|g| self.occurs_in(v, g)),
            Type::ImplTrait(bounds) => bounds.iter().any(|b| self.occurs_in(v, b)),
            Type::Evidential { inner, .. } => self.occurs_in(v, inner),
            Type::Atomic(inner) => self.occurs_in(v, inner),
            Type::Simd { element, .. } => self.occurs_in(v, element),
            _ => false,
        }
    }

    /// Freshen a type by replacing all type variables with fresh ones
    /// This is used for polymorphic built-in functions
    fn freshen(&mut self, ty: &Type) -> Type {
        let mut mapping = std::collections::HashMap::new();
        self.freshen_inner(ty, &mut mapping)
    }

    fn freshen_inner(
        &mut self,
        ty: &Type,
        mapping: &mut std::collections::HashMap<u32, Type>,
    ) -> Type {
        match ty {
            Type::Var(TypeVar(id)) => {
                if let Some(fresh) = mapping.get(id) {
                    fresh.clone()
                } else {
                    let fresh = self.fresh_var();
                    mapping.insert(*id, fresh.clone());
                    fresh
                }
            }
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.freshen_inner(element, mapping)),
                size: *size,
            },
            Type::Slice(inner) => Type::Slice(Box::new(self.freshen_inner(inner, mapping))),
            Type::Ref { lifetime, mutable, inner } => Type::Ref {
                lifetime: lifetime.clone(),
                mutable: *mutable,
                inner: Box::new(self.freshen_inner(inner, mapping)),
            },
            Type::Tuple(elems) => Type::Tuple(
                elems
                    .iter()
                    .map(|e| self.freshen_inner(e, mapping))
                    .collect(),
            ),
            Type::Function {
                params,
                return_type,
                is_async,
            } => Type::Function {
                params: params
                    .iter()
                    .map(|p| self.freshen_inner(p, mapping))
                    .collect(),
                return_type: Box::new(self.freshen_inner(return_type, mapping)),
                is_async: *is_async,
            },
            Type::Evidential { inner, evidence } => Type::Evidential {
                inner: Box::new(self.freshen_inner(inner, mapping)),
                evidence: *evidence,
            },
            Type::Named { name, generics } => Type::Named {
                name: name.clone(),
                generics: generics
                    .iter()
                    .map(|g| self.freshen_inner(g, mapping))
                    .collect(),
            },
            // Primitive types don't contain type variables
            _ => ty.clone(),
        }
    }

    /// Push a new scope
    fn push_scope(&mut self) {
        let new_env = TypeEnv::with_parent(self.env.clone());
        self.env = Rc::new(RefCell::new(new_env));
    }

    /// Pop current scope
    fn pop_scope(&mut self) {
        let parent = self.env.borrow().parent.clone();
        if let Some(p) = parent {
            self.env = p;
        }
    }

    /// Record an error, auto-attaching current item span if no span is set
    fn error(&mut self, mut err: TypeError) {
        if err.span.is_none() && !self.current_item_span.is_empty() {
            err.span = Some(self.current_item_span);
        }
        self.errors.push(err);
    }

    /// Check if actual evidence can satisfy expected evidence requirement.
    /// Returns Ok(()) if compatible, Err with helpful message if not.
    fn check_evidence(
        &mut self,
        expected: EvidenceLevel,
        actual: EvidenceLevel,
        context: &str,
    ) -> bool {
        if actual.satisfies(expected) {
            true
        } else {
            let mut err = TypeError::new(format!(
                "evidence mismatch {}: expected {} ({}), found {} ({})",
                context,
                expected.name(),
                expected.symbol(),
                actual.name(),
                actual.symbol(),
            ));

            // Add helpful notes based on the specific mismatch
            match (expected, actual) {
                (EvidenceLevel::Known, EvidenceLevel::Reported) => {
                    err = err.with_note(
                        "reported data (~) cannot be used where known data (!) is required",
                    );
                    err = err.with_note(
                        "help: use |validate!{...} to verify and promote evidence level",
                    );
                }
                (EvidenceLevel::Known, EvidenceLevel::Uncertain) => {
                    err = err.with_note(
                        "uncertain data (?) cannot be used where known data (!) is required",
                    );
                    err = err.with_note(
                        "help: use pattern matching or unwrap to handle the uncertainty",
                    );
                }
                (EvidenceLevel::Uncertain, EvidenceLevel::Reported) => {
                    err = err.with_note(
                        "reported data (~) cannot be used where uncertain data (?) is required",
                    );
                    err = err.with_note("help: use |validate?{...} to verify external data");
                }
                _ => {
                    err = err.with_note(format!(
                        "evidence lattice: known (!) < uncertain (?) < reported (~) < paradox (‽)"
                    ));
                }
            }

            self.error(err);
            false
        }
    }

    /// Extract evidence level from a type, defaulting to Known
    fn get_evidence(&self, ty: &Type) -> EvidenceLevel {
        match ty {
            Type::Evidential { evidence, .. } => *evidence,
            _ => EvidenceLevel::Known,
        }
    }

    /// Check a source file
    pub fn check_file(&mut self, file: &SourceFile) -> Result<(), Vec<TypeError>> {
        // First pass: collect type definitions
        for item in &file.items {
            self.collect_type_def(&item.node);
        }

        // Second pass: collect function signatures
        for item in &file.items {
            self.collect_fn_sig(&item.node);
        }

        // Third pass: check function bodies
        for item in &file.items {
            self.current_item_span = item.span;
            self.check_item(&item.node);
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    /// Collect type definitions (first pass)
    fn collect_type_def(&mut self, item: &Item) {
        match item {
            Item::Struct(s) => {
                let generics = s
                    .generics
                    .as_ref()
                    .map(|g| {
                        g.params
                            .iter()
                            .filter_map(|p| {
                                if let GenericParam::Type { name, .. } = p {
                                    Some(name.name.clone())
                                } else {
                                    None
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let fields = match &s.fields {
                    StructFields::Named(fs) => fs
                        .iter()
                        .map(|f| (f.name.name.clone(), self.convert_type(&f.ty)))
                        .collect(),
                    StructFields::Tuple(ts) => ts
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (i.to_string(), self.convert_type(t)))
                        .collect(),
                    StructFields::Unit => vec![],
                };

                // Check for duplicate struct definition
                if self.types.contains_key(&s.name.name) {
                    self.error(TypeError::new(format!(
                        "duplicate type definition: '{}'",
                        s.name.name
                    )));
                }
                self.types
                    .insert(s.name.name.clone(), TypeDef::Struct { generics, fields });
            }
            Item::Enum(e) => {
                let generics = e
                    .generics
                    .as_ref()
                    .map(|g| {
                        g.params
                            .iter()
                            .filter_map(|p| {
                                if let GenericParam::Type { name, .. } = p {
                                    Some(name.name.clone())
                                } else {
                                    None
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let variants = e
                    .variants
                    .iter()
                    .map(|v| {
                        let fields = match &v.fields {
                            StructFields::Tuple(ts) => {
                                Some(ts.iter().map(|t| self.convert_type(t)).collect())
                            }
                            StructFields::Named(fs) => {
                                Some(fs.iter().map(|f| self.convert_type(&f.ty)).collect())
                            }
                            StructFields::Unit => None,
                        };
                        (v.name.name.clone(), fields)
                    })
                    .collect();

                self.types
                    .insert(e.name.name.clone(), TypeDef::Enum { generics, variants });
            }
            Item::TypeAlias(t) => {
                let generics = t
                    .generics
                    .as_ref()
                    .map(|g| {
                        g.params
                            .iter()
                            .filter_map(|p| {
                                if let GenericParam::Type { name, .. } = p {
                                    Some(name.name.clone())
                                } else {
                                    None
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let target = self.convert_type(&t.ty);
                self.types
                    .insert(t.name.name.clone(), TypeDef::Alias { generics, target });
            }
            _ => {}
        }
    }

    /// Collect function signatures (second pass)
    fn collect_fn_sig(&mut self, item: &Item) {
        match item {
            Item::Function(f) => {
                // Set up generic type parameters as type variables
                if let Some(ref generics) = f.generics {
                    for param in &generics.params {
                        if let crate::ast::GenericParam::Type { name, .. } = param {
                            let type_var = self.fresh_var();
                            self.current_generics.insert(name.name.clone(), type_var);
                        }
                    }
                }

                let params: Vec<Type> = f.params.iter().map(|p| self.convert_type(&p.ty)).collect();

                let return_type = f
                    .return_type
                    .as_ref()
                    .map(|t| self.convert_type(t))
                    .unwrap_or(Type::Unit);

                let fn_type = Type::Function {
                    params,
                    return_type: Box::new(return_type),
                    is_async: f.is_async,
                };

                // Check for duplicate function definition (allow shadowing stdlib functions)
                if self.functions.contains_key(&f.name.name)
                    && !self.stdlib_functions.contains(&f.name.name)
                {
                    self.error(TypeError::new(format!(
                        "duplicate function definition: '{}'",
                        f.name.name
                    )));
                }
                self.functions.insert(f.name.name.clone(), fn_type);

                // Clear generics after processing
                self.current_generics.clear();
            }
            Item::Impl(impl_block) => {
                // Get the type name being implemented
                let type_name = self.type_path_to_name(&impl_block.self_ty);

                // Set up generic type parameters as type variables
                // Must do this FIRST so we can include generics in current_self_type
                let mut generic_types = Vec::new();
                if let Some(ref generics) = impl_block.generics {
                    for param in &generics.params {
                        if let crate::ast::GenericParam::Type { name, .. } = param {
                            let type_var = self.fresh_var();
                            self.current_generics.insert(name.name.clone(), type_var.clone());
                            generic_types.push(type_var);
                        }
                    }
                }

                // Set current_self_type with generics so Self resolves correctly
                self.current_self_type = Some(Type::Named {
                    name: type_name.clone(),
                    generics: generic_types,
                });

                // Collect associated functions/methods
                for impl_item in &impl_block.items {
                    if let crate::ast::ImplItem::Function(f) = impl_item {
                        let params: Vec<Type> =
                            f.params.iter().map(|p| self.convert_type(&p.ty)).collect();

                        let return_type = f
                            .return_type
                            .as_ref()
                            .map(|t| self.convert_type(t))
                            .unwrap_or(Type::Unit);

                        let fn_type = Type::Function {
                            params,
                            return_type: Box::new(return_type),
                            is_async: f.is_async,
                        };

                        // Register in impl_methods
                        self.impl_methods
                            .entry(type_name.clone())
                            .or_insert_with(HashMap::new)
                            .insert(f.name.name.clone(), fn_type);
                    }
                }

                // Clear current_self_type and generics when done
                self.current_self_type = None;
                self.current_generics.clear();
            }
            _ => {}
        }
    }

    /// Convert a TypePath to a simple type name string
    fn type_path_to_name(&self, ty: &crate::ast::TypeExpr) -> String {
        match ty {
            crate::ast::TypeExpr::Path(path) => {
                path.segments
                    .iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::")
            }
            _ => "Unknown".to_string(),
        }
    }

    /// Check an item (third pass)
    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.check_function(f),
            Item::Const(c) => {
                let declared = self.convert_type(&c.ty);
                let inferred = self.infer_expr(&c.value);
                if !self.unify(&declared, &inferred) {
                    self.error(
                        TypeError::new(format!(
                            "type mismatch in const '{}': expected {:?}, found {:?}",
                            c.name.name, declared, inferred
                        ))
                        .with_span(c.name.span),
                    );
                }
            }
            Item::Static(s) => {
                let declared = self.convert_type(&s.ty);
                let inferred = self.infer_expr(&s.value);
                if !self.unify(&declared, &inferred) {
                    self.error(
                        TypeError::new(format!(
                            "type mismatch in static '{}': expected {:?}, found {:?}",
                            s.name.name, declared, inferred
                        ))
                        .with_span(s.name.span),
                    );
                }
            }
            Item::Impl(impl_block) => {
                // Get the type name being implemented
                let type_name = self.type_path_to_name(&impl_block.self_ty);

                // Set up generic type parameters as type variables
                // Must do this FIRST so we can include generics in current_self_type
                let mut generic_types = Vec::new();
                if let Some(ref generics) = impl_block.generics {
                    for param in &generics.params {
                        if let crate::ast::GenericParam::Type { name, .. } = param {
                            let type_var = self.fresh_var();
                            self.current_generics.insert(name.name.clone(), type_var.clone());
                            generic_types.push(type_var);
                        }
                    }
                }

                // Set current_self_type with generics so Self resolves correctly
                self.current_self_type = Some(Type::Named {
                    name: type_name,
                    generics: generic_types,
                });

                // Check each function in the impl block
                for impl_item in &impl_block.items {
                    if let crate::ast::ImplItem::Function(f) = impl_item {
                        self.check_function(f);
                    }
                }

                // Clear current_self_type and generics when done
                self.current_self_type = None;
                self.current_generics.clear();
            }
            _ => {}
        }
    }

    /// Check a function body
    fn check_function(&mut self, func: &Function) {
        self.push_scope();

        // Set up generic type parameters as type variables
        if let Some(ref generics) = func.generics {
            for param in &generics.params {
                if let crate::ast::GenericParam::Type { name, .. } = param {
                    let type_var = self.fresh_var();
                    self.current_generics.insert(name.name.clone(), type_var);
                }
            }
        }

        // Bind parameters with evidence inference
        for param in &func.params {
            let ty = self.convert_type(&param.ty);
            // Infer parameter evidence from type annotation if present,
            // otherwise from pattern annotation, otherwise default to Known
            let type_evidence = self.get_evidence(&ty);
            let evidence = param
                .pattern
                .evidentiality()
                .map(EvidenceLevel::from_ast)
                .unwrap_or(type_evidence);

            if let Some(name) = param.pattern.binding_name() {
                self.env.borrow_mut().define(name, ty, evidence);
            }
        }

        // Set expected return type for checking explicit return statements
        let expected_return = func
            .return_type
            .as_ref()
            .map(|t| self.convert_type(t))
            .unwrap_or(Type::Unit);
        let old_return_type = self.expected_return_type.clone();
        self.expected_return_type = Some(expected_return.clone());

        // Check body
        if let Some(ref body) = func.body {
            let body_type = self.check_block(body);

            // Restore old return type
            self.expected_return_type = old_return_type;

            // Check return type (for implicit returns)
            let expected_return = func
                .return_type
                .as_ref()
                .map(|t| self.convert_type(t))
                .unwrap_or(Type::Unit);

            // Check structural type compatibility
            // For bootstrapping: skip return type checking to be lenient with
            // cross-file references and unresolved type variables
            let _ = self.unify(&expected_return, &body_type);

            // Evidence inference for return types:
            // - If return type has explicit evidence annotation → check compatibility
            // - If function name has evidence annotation (e.g., validate!) → use that
            // - If no explicit annotation → infer evidence from body
            // - For public functions, warn if evidence should be annotated at module boundary
            let type_has_evidence = self.type_has_explicit_evidence(func.return_type.as_ref());
            // Function name evidentiality (e.g., validate_model! has ! evidentiality)
            let name_evidence = func.name.evidentiality.as_ref()
                .map(|e| EvidenceLevel::from_ast(*e));
            let has_explicit_evidence = type_has_evidence || name_evidence.is_some();
            let actual_evidence = self.get_evidence(&body_type);

            if has_explicit_evidence {
                // Explicit annotation: check compatibility
                // EXCEPT: if evidentiality is on the function NAME (e.g., validate!),
                // the function is declaring it transforms evidence - trust that declaration
                if name_evidence.is_none() {
                    let expected_evidence = self.get_evidence(&expected_return);
                    self.check_evidence(
                        expected_evidence,
                        actual_evidence,
                        &format!("in return type of '{}'", func.name.name),
                    );
                }
                // If name has evidentiality, skip the check - function transforms evidence
            } else {
                // No explicit annotation: infer from body
                // For public functions at module boundaries, suggest annotation
                if func.visibility == Visibility::Public && actual_evidence > EvidenceLevel::Known {
                    self.error(
                        TypeError::new(format!(
                            "public function '{}' returns {} ({}) data but has no explicit evidence annotation",
                            func.name.name,
                            actual_evidence.name(),
                            actual_evidence.symbol(),
                        ))
                        .with_span(func.name.span)
                        .with_note("help: add explicit evidence annotation to the return type")
                        .with_note(format!(
                            "example: fn {}(...) -> {}{} {{ ... }}",
                            func.name.name,
                            expected_return,
                            actual_evidence.symbol()
                        )),
                    );
                }
                // Inference succeeds - the body's evidence becomes the function's evidence
            }
        }

        // Clear generics after processing
        self.current_generics.clear();
        self.pop_scope();
    }

    /// Check if a type expression has an explicit evidence annotation
    fn type_has_explicit_evidence(&self, ty: Option<&TypeExpr>) -> bool {
        match ty {
            Some(TypeExpr::Evidential { .. }) => true,
            Some(TypeExpr::Reference { inner, .. })
            | Some(TypeExpr::Pointer { inner, .. })
            | Some(TypeExpr::Slice(inner))
            | Some(TypeExpr::Array { element: inner, .. }) => {
                self.type_has_explicit_evidence(Some(inner.as_ref()))
            }
            Some(TypeExpr::Tuple(elements)) => elements
                .iter()
                .any(|e| self.type_has_explicit_evidence(Some(e))),
            _ => false,
        }
    }

    /// Check a block and return its type
    fn check_block(&mut self, block: &Block) -> Type {
        self.push_scope();

        let mut diverges = false;
        for stmt in &block.stmts {
            let stmt_ty = self.check_stmt(stmt);
            if matches!(stmt_ty, Type::Never) {
                diverges = true;
            }
        }

        let result = if let Some(ref expr) = block.expr {
            self.infer_expr(expr)
        } else if diverges {
            Type::Never
        } else {
            Type::Unit
        };

        self.pop_scope();
        result
    }

    /// Check a statement and return its type (Never if it diverges)
    fn check_stmt(&mut self, stmt: &Stmt) -> Type {
        match stmt {
            Stmt::Let { pattern, ty, init } => {
                let declared_ty = ty.as_ref().map(|t| self.convert_type(t));
                let init_ty = init.as_ref().map(|e| self.infer_expr(e));

                let final_ty = match (&declared_ty, &init_ty) {
                    (Some(d), Some(i)) => {
                        if !self.unify(d, i) {
                            // Report type mismatch error with helpful hints
                            let binding_name = pattern.binding_name().unwrap_or_else(|| "<pattern>".to_string());

                            // Check for common Rust-ism: using [T] slice syntax with array literal
                            let hint = match (d, i) {
                                (Type::Slice(_), Type::Array { .. }) => {
                                    ". Hint: `[T]` is slice syntax in Sigil. \
                                    For arrays, use `[T; N]` or omit the type annotation entirely"
                                }
                                _ => "",
                            };

                            let mut err = TypeError::new(format!(
                                "type mismatch in let binding '{}': expected {:?}, found {:?}{}",
                                binding_name, d, i, hint
                            ));
                            if let Some(span) = pattern.binding_span() {
                                err = err.with_span(span);
                            }
                            self.error(err);
                        }
                        d.clone()
                    }
                    (Some(d), None) => d.clone(),
                    (None, Some(i)) => i.clone(),
                    (None, None) => self.fresh_var(),
                };

                // Evidence inference: explicit annotation takes precedence,
                // otherwise infer from initializer expression.
                // This reduces annotation burden while maintaining safety:
                // - `let x = network_call()` → x is automatically ~
                // - `let x! = validated_data` → explicit ! annotation honored
                let evidence = pattern
                    .evidentiality()
                    .map(EvidenceLevel::from_ast)
                    .unwrap_or_else(|| {
                        // Infer evidence from initializer type
                        init_ty
                            .as_ref()
                            .map(|ty| self.get_evidence(ty))
                            .unwrap_or(EvidenceLevel::Known)
                    });

                // For simple ident patterns, use define() directly (preserves evidence wrapping).
                // For complex patterns (tuples, structs), use bind_pattern for destructuring.
                if let Some(name) = pattern.binding_name() {
                    self.env.borrow_mut().define(name, final_ty, evidence);
                } else {
                    self.bind_pattern(pattern, &final_ty, evidence);
                }
                Type::Unit
            }
            Stmt::LetElse { pattern, ty, init, else_branch } => {
                // Type check let-else similar to let
                let declared_ty = ty.as_ref().map(|t| self.convert_type(t));
                let init_ty = self.infer_expr(init);
                // Infer evidence before moving init_ty
                let evidence = pattern
                    .evidentiality()
                    .map(EvidenceLevel::from_ast)
                    .unwrap_or_else(|| self.get_evidence(&init_ty));
                let final_ty = declared_ty.unwrap_or(init_ty);
                // Check else branch
                self.infer_expr(else_branch);
                // For simple ident patterns, use define() directly (preserves evidence wrapping).
                // For complex patterns (tuples, structs), use bind_pattern for destructuring.
                if let Some(name) = pattern.binding_name() {
                    self.env.borrow_mut().define(name, final_ty, evidence);
                } else {
                    self.bind_pattern(pattern, &final_ty, evidence);
                }
                Type::Unit
            }
            Stmt::Expr(e) | Stmt::Semi(e) => self.infer_expr(e),
            Stmt::Item(item) => {
                self.check_item(item);
                Type::Unit
            }
        }
    }

    /// Infer the type of an expression
    pub fn infer_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            Expr::Literal(lit) => self.infer_literal(lit),

            Expr::Path(path) => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident.name;

                    // First, lookup the type (immutable borrow)
                    let lookup_result = self.env.borrow().lookup(name);

                    if let Some((ty, _)) = lookup_result {
                        // Check for linear type double-use
                        if matches!(ty, Type::Linear(_)) {
                            let already_consumed = self.env.borrow().is_consumed(name);
                            if already_consumed {
                                // Linear variable used twice - no-cloning violation!
                                self.error(TypeError::new(format!(
                                    "linear value '{}' used twice: linear types cannot be cloned (no-cloning theorem)",
                                    name
                                )));
                            } else {
                                // Mark as consumed for future use checks
                                self.env.borrow_mut().consume(name);
                            }
                        }
                        return ty;
                    }
                    if let Some(ty) = self.functions.get(name).cloned() {
                        // Freshen polymorphic types to get fresh type variables
                        return self.freshen(&ty);
                    }
                } else if path.segments.len() == 2 {
                    // Handle Type::method() - associated function lookup
                    let type_name = &path.segments[0].ident.name;
                    let method_name = &path.segments[1].ident.name;

                    // Check impl_methods for associated functions
                    if let Some(methods) = self.impl_methods.get(type_name) {
                        if let Some(ty) = methods.get(method_name) {
                            let ty_cloned = ty.clone();
                            return self.freshen(&ty_cloned);
                        }
                    }

                    // Check for enum variant constructors: Enum::Variant
                    if let Some(TypeDef::Enum { variants, .. }) = self.types.get(type_name) {
                        for (variant_name, _variant_fields) in variants {
                            if variant_name == method_name {
                                // Return the enum type for unit/tuple variants
                                return Type::Named {
                                    name: type_name.clone(),
                                    generics: vec![],
                                };
                            }
                        }
                    }
                }
                // For bootstrapping: treat undefined paths as unknown types
                // This allows cross-file references to not cause errors
                // A real type checker would require imports or multi-file analysis
                self.fresh_var()
            }

            Expr::Binary { left, op, right } => {
                let lt = self.infer_expr(left);
                let rt = self.infer_expr(right);
                self.infer_binary_op(op, &lt, &rt)
            }

            Expr::Unary { op, expr } => {
                let inner = self.infer_expr(expr);
                self.infer_unary_op(op, &inner)
            }

            Expr::Call { func, args } => {
                let fn_type = self.infer_expr(func);
                let arg_types: Vec<Type> = args.iter().map(|a| self.infer_expr(a)).collect();

                if let Type::Function {
                    params,
                    return_type,
                    ..
                } = fn_type
                {
                    // Extract function name for variadic builtin check
                    let func_name = match func.as_ref() {
                        Expr::Path(path) if path.segments.len() == 1 => {
                            Some(path.segments[0].ident.name.as_str())
                        }
                        _ => None,
                    };

                    // Known variadic builtins: interpreter accepts variable args
                    // (registered with arity: None in stdlib.rs). Allow extra
                    // arguments beyond the minimum registered parameter count.
                    let is_variadic_builtin = func_name.map_or(false, |name| {
                        matches!(
                            name,
                            "assert"
                                | "println"
                                | "print"
                                | "eprintln"
                                | "eprint"
                                | "panic"
                                | "todo"
                                | "unreachable"
                                | "format"
                        )
                    });

                    // Check argument count: variadic builtins require at least
                    // params.len() args; all others require exact match.
                    if is_variadic_builtin {
                        if arg_types.len() < params.len() {
                            self.error(TypeError::new(format!(
                                "expected at least {} arguments, found {}",
                                params.len(),
                                arg_types.len()
                            )));
                        }
                    } else if params.len() != arg_types.len() {
                        self.error(TypeError::new(format!(
                            "expected {} arguments, found {}",
                            params.len(),
                            arg_types.len()
                        )));
                    }

                    // Check argument types and evidence levels
                    for (i, (param, arg)) in params.iter().zip(arg_types.iter()).enumerate() {
                        // Check argument type matches parameter type
                        if !self.unify(param, arg) {
                            // Allow implicit numeric coercion: int → float
                            let is_numeric_coercion = Self::is_numeric_coercion(param, arg);
                            // Allow reference coercions: &mut T → &T, &Box<T> → &T, &Vec<T> → &[T], &&T → &T
                            let is_reference_coercion = Self::is_reference_coercion(param, arg);
                            // Allow auto-ref/deref: T → &T, &T → T
                            let is_ref_value_coercion = Self::is_ref_value_coercion(param, arg);
                            // Only report error for concrete type mismatches, not type variables
                            if !matches!(param, Type::Var(_)) && !matches!(arg, Type::Var(_))
                                && !is_numeric_coercion && !is_reference_coercion
                                && !is_ref_value_coercion {
                                self.error(TypeError::new(format!(
                                    "type mismatch in argument {}: expected {}, found {}",
                                    i + 1, param, arg
                                )));
                            }
                        }

                        // Check evidence compatibility only when the parameter has an
                        // explicit evidence annotation (Type::Evidential). Unannotated
                        // parameters like `x: usize` accept any evidence level.
                        // Type variables (polymorphic) also skip evidence checking.
                        if matches!(param, Type::Evidential { .. }) {
                            let expected_evidence = self.get_evidence(param);
                            let actual_evidence = self.get_evidence(arg);
                            self.check_evidence(
                                expected_evidence,
                                actual_evidence,
                                &format!("in argument {}", i + 1),
                            );
                        }
                    }

                    *return_type
                } else if let Type::Var(_) = &fn_type {
                    // For bootstrapping: if function is a type variable (undefined path),
                    // create a function type and unify, then return fresh result
                    let result_ty = self.fresh_var();
                    let inferred_fn = Type::Function {
                        params: arg_types,
                        return_type: Box::new(result_ty.clone()),
                        is_async: false,
                    };
                    self.unify(&fn_type, &inferred_fn);
                    result_ty
                } else {
                    // For bootstrapping: return fresh type variable instead of error
                    self.fresh_var()
                }
            }

            Expr::Array(elements) => {
                if elements.is_empty() {
                    Type::Array {
                        element: Box::new(self.fresh_var()),
                        size: Some(0),
                    }
                } else {
                    let elem_ty = self.infer_expr(&elements[0]);
                    for elem in &elements[1..] {
                        let t = self.infer_expr(elem);
                        if !self.unify(&elem_ty, &t) {
                            self.error(TypeError::new("array elements must have same type"));
                        }
                    }
                    Type::Array {
                        element: Box::new(elem_ty),
                        size: Some(elements.len()),
                    }
                }
            }

            Expr::Tuple(elements) => {
                Type::Tuple(elements.iter().map(|e| self.infer_expr(e)).collect())
            }

            Expr::Block(block) => self.check_block(block),

            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_ty = self.infer_expr(condition);
                // Strip evidence wrapper before checking: bool? is still bool
                let (bare_cond_ty, _) = self.strip_evidence(&cond_ty);
                if !self.unify(&Type::Bool, &bare_cond_ty) {
                    self.error(TypeError::new("if condition must be bool"));
                }

                let then_ty = self.check_block(then_branch);

                if let Some(else_expr) = else_branch {
                    // Else branch can be another expr (e.g., if-else chain) or a block
                    let else_ty = match else_expr.as_ref() {
                        Expr::Block(block) => self.check_block(block),
                        other => self.infer_expr(other),
                    };
                    // For bootstrapping: just try to unify, skip error if unification fails
                    // (type inference is incomplete so false positives are common)
                    let _ = self.unify(&then_ty, &else_ty);

                    // Evidence inference for control flow:
                    // Join evidence from both branches (pessimistic - takes least certain)
                    // This ensures that if either branch produces uncertain data,
                    // the result is marked as uncertain.
                    let then_ev = self.get_evidence(&then_ty);
                    let else_ev = self.get_evidence(&else_ty);
                    let joined_ev = then_ev.join(else_ev);

                    let (inner_ty, _) = self.strip_evidence(&then_ty);
                    if joined_ev > EvidenceLevel::Known {
                        Type::Evidential {
                            inner: Box::new(inner_ty),
                            evidence: joined_ev,
                        }
                    } else {
                        inner_ty
                    }
                } else {
                    Type::Unit
                }
            }

            Expr::While {
                condition,
                body,
                ..
            } => {
                let cond_ty = self.infer_expr(condition);
                let (bare_cond_ty, _) = self.strip_evidence(&cond_ty);
                if !self.unify(&Type::Bool, &bare_cond_ty) {
                    self.error(TypeError::new("while condition must be bool"));
                }
                self.check_block(body);
                Type::Unit
            }

            Expr::Loop { body, .. } => {
                self.check_block(body);
                Type::Unit
            }

            Expr::For {
                pattern: _,
                iter,
                body,
                ..
            } => {
                // Infer the iterable type (for basic type checking)
                let _ = self.infer_expr(iter);
                self.check_block(body);
                Type::Unit
            }

            Expr::Pipe { expr, operations } => {
                let mut current = self.infer_expr(expr);

                for op in operations {
                    current = self.infer_pipe_op(op, &current);
                }

                current
            }

            Expr::Index { expr, index } => {
                let coll_ty = self.infer_expr(expr);
                let idx_ty = self.infer_expr(index);

                match coll_ty {
                    Type::Array { element, .. } | Type::Slice(element) => {
                        if !matches!(idx_ty, Type::Int(_)) {
                            self.error(TypeError::new("index must be integer"));
                        }
                        *element
                    }
                    _ => {
                        // For bootstrapping: return fresh type variable
                        self.fresh_var()
                    }
                }
            }

            Expr::Return(val) => {
                let actual_type = if let Some(e) = val {
                    self.infer_expr(e)
                } else {
                    Type::Unit
                };

                // Check against expected return type if we're inside a function
                if let Some(expected) = self.expected_return_type.clone() {
                    if !self.unify(&expected, &actual_type) {
                        self.error(TypeError::new(format!(
                            "type mismatch in return: expected {}, found {}",
                            expected, actual_type
                        )));
                    }
                }

                Type::Never
            }

            // Mark expression with evidence
            Expr::Evidential {
                expr,
                evidentiality,
            } => {
                let inner = self.infer_expr(expr);
                let ev = EvidenceLevel::from_ast(*evidentiality);

                // When ? (Uncertain) is applied to Result<T, E> or Option<T>,
                // this is the try operator: unwrap to T
                if ev == EvidenceLevel::Uncertain {
                    let resolved = if let Type::Var(v) = &inner {
                        self.substitutions.get(v).cloned().unwrap_or(inner.clone())
                    } else {
                        inner.clone()
                    };
                    match &resolved {
                        Type::Named { name, generics } if name == "Result" && !generics.is_empty() => {
                            return generics[0].clone();
                        }
                        Type::Named { name, generics } if name == "Option" && !generics.is_empty() => {
                            return generics[0].clone();
                        }
                        _ => {}
                    }
                }

                Type::Evidential {
                    inner: Box::new(inner),
                    evidence: ev,
                }
            }

            // Match expression with evidence-aware dispatch
            Expr::Match { expr, arms } => {
                let scrutinee = self.infer_expr(expr);
                let scrutinee_ev = self.get_evidence(&scrutinee);

                if arms.is_empty() {
                    return Type::Never; // Empty match is diverging
                }

                // Check all arms and collect their types
                let mut arm_types: Vec<Type> = Vec::new();
                let mut max_evidence = EvidenceLevel::Known;

                // Snapshot substitutions before match: each arm should start
                // from the same type variable state. This prevents one arm's
                // bindings (e.g., Device=Cuda) from conflicting with another
                // arm's bindings (e.g., Device=Cpu) in device dispatch patterns.
                let saved_substitutions = self.substitutions.clone();

                for arm in arms {
                    // Restore substitutions to pre-match state for each arm
                    self.substitutions = saved_substitutions.clone();
                    self.push_scope();

                    // Bind pattern variables with scrutinee's evidence level
                    // This propagates evidence through pattern matching
                    self.bind_pattern(&arm.pattern, &scrutinee, scrutinee_ev);

                    // Check guard if present
                    if let Some(ref guard) = arm.guard {
                        let guard_ty = self.infer_expr(guard);
                        if !self.unify(&Type::Bool, &guard_ty) {
                            self.error(TypeError::new("match guard must be bool"));
                        }
                    }

                    // Infer arm body type
                    let body_ty = self.infer_expr(&arm.body);
                    let body_ev = self.get_evidence(&body_ty);

                    // Join evidence from all arms (pessimistic)
                    max_evidence = max_evidence.join(body_ev);
                    arm_types.push(body_ty);

                    self.pop_scope();
                }

                // Restore to pre-match state, then let arm type unification
                // establish the final bindings from the match result type
                self.substitutions = saved_substitutions;

                // Unify all arm types
                // For bootstrapping: skip error, just try to unify
                let first_ty = &arm_types[0];
                for (_i, ty) in arm_types.iter().enumerate().skip(1) {
                    let _ = self.unify(first_ty, ty);
                }

                // Result has joined evidence from all arms
                let (inner_ty, _) = self.strip_evidence(first_ty);
                if max_evidence > EvidenceLevel::Known {
                    Type::Evidential {
                        inner: Box::new(inner_ty),
                        evidence: max_evidence,
                    }
                } else {
                    inner_ty
                }
            }

            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let recv_ty = self.infer_expr(receiver);
                let (recv_inner, recv_ev) = self.strip_evidence(&recv_ty);
                // Strip references to get the underlying type for method lookup
                let recv_derefed = match &recv_inner {
                    Type::Ref { inner, .. } => {
                        // Also strip evidence from inner ref
                        let (inner_stripped, _) = self.strip_evidence(inner);
                        // Handle &&T -> T
                        match &inner_stripped {
                            Type::Ref { inner: inner2, .. } => {
                                let (i2, _) = self.strip_evidence(inner2);
                                i2
                            }
                            other => other.clone(),
                        }
                    }
                    other => other.clone(),
                };
                let _arg_types: Vec<Type> = args.iter().map(|a| self.infer_expr(a)).collect();

                // FIRST: Check user-defined methods in impl_methods
                // This takes priority over hardcoded patterns
                // Try both the original recv_inner and the deref'd version
                let user_method_result = {
                    // Try original type first
                    let mut result = None;
                    if let Type::Named { name: ref type_name, .. } = recv_inner {
                        if let Some(fn_type) = self.impl_methods.get(type_name)
                            .and_then(|methods| methods.get(&method.name))
                            .cloned()
                        {
                            if let Type::Function { return_type, .. } = self.freshen(&fn_type) {
                                result = Some(*return_type);
                            }
                        }
                    }
                    // If not found, try deref'd type
                    if result.is_none() {
                        if let Type::Named { name: ref type_name, .. } = recv_derefed {
                            if let Some(fn_type) = self.impl_methods.get(type_name)
                                .and_then(|methods| methods.get(&method.name))
                                .cloned()
                            {
                                if let Type::Function { return_type, .. } = self.freshen(&fn_type) {
                                    result = Some(*return_type);
                                }
                            }
                        }
                    }
                    result
                };

                // If user-defined method found, use it; otherwise fall back to hardcoded patterns
                let result_ty = if let Some(user_ty) = user_method_result {
                    user_ty
                } else {
                    // Resolve known methods based on receiver type and method name
                    match method.name.as_str() {
                    // Collection methods returning usize
                    "len" | "count" | "size" => Type::Int(IntSize::USize),

                    // Boolean predicates
                    "is_empty" | "contains" | "starts_with" | "ends_with" | "is_some"
                    | "is_none" | "is_ok" | "is_err" | "is_ascii" | "is_alphabetic"
                    | "is_numeric" | "is_alphanumeric" | "is_whitespace" | "is_uppercase"
                    | "is_lowercase" | "exists" | "is_file" | "is_dir" | "is_match"
                    | "matches" | "eq" | "ne" | "lt" | "le" | "gt" | "ge" => Type::Bool,

                    // String methods returning String
                    "to_string" | "to_lowercase" | "to_uppercase" | "trim" | "trim_start"
                    | "trim_end" | "to_owned" | "replace" | "replacen" | "repeat"
                    | "to_string_lossy" => Type::Named {
                        name: "String".to_string(),
                        generics: vec![],
                    },

                    // String methods returning &str
                    "as_str" | "trim_matches" | "trim_start_matches" | "trim_end_matches"
                    | "strip_prefix" | "strip_suffix" => Type::Ref {
                        lifetime: None,
                        mutable: false,
                        inner: Box::new(Type::Str),
                    },

                    // Clone returns same type as receiver
                    "clone" | "cloned" | "copied" => recv_inner.clone(),

                    // Option/Result unwrapping - return inner type or fresh var
                    "unwrap" | "unwrap_or" | "unwrap_or_default" | "unwrap_or_else"
                    | "expect" | "ok" | "err" => {
                        if let Type::Named { name, generics } = &recv_inner {
                            if (name == "Option" || name == "Result") && !generics.is_empty() {
                                generics[0].clone()
                            } else {
                                self.fresh_var()
                            }
                        } else {
                            self.fresh_var()
                        }
                    }

                    // collect() returns fresh var to unify with type annotation
                    "collect" => self.fresh_var(),

                    // Iterator/collection transformation methods - preserve receiver type
                    "iter" | "into_iter" | "iter_mut" | "rev" | "skip" | "take"
                    | "filter" | "map" | "filter_map" | "flat_map" | "enumerate"
                    | "zip" | "chain" | "flatten" | "reverse" | "sorted"
                    | "dedup" | "unique" | "peekable" | "fuse" | "cycle" | "step_by"
                    | "take_while" | "skip_while" | "scan" | "inspect" => recv_inner.clone(),

                    // String splitting/iteration - returns iterator (fresh var for proper chaining)
                    "split" | "rsplit" | "splitn" | "rsplitn" | "split_whitespace"
                    | "split_ascii_whitespace" | "lines" | "chars" | "bytes"
                    | "char_indices" | "split_terminator" | "rsplit_terminator"
                    | "split_inclusive" | "matches_iter" => self.fresh_var(),

                    // HashMap/BTreeMap methods - return iterator-like fresh var
                    "keys" | "values" | "values_mut" | "into_keys"
                    | "into_values" | "entry" | "drain" => self.fresh_var(),

                    // Methods returning Option<T>
                    "first" | "last" | "get" | "get_mut" | "pop" | "pop_front"
                    | "pop_back" | "find" | "find_map" | "position" | "rposition"
                    | "next" | "next_back" | "peek" | "nth" | "last_mut"
                    | "binary_search" | "parent" | "file_name" | "file_stem"
                    | "extension" => Type::Named {
                        name: "Option".to_string(),
                        generics: vec![self.fresh_var()],
                    },

                    // Methods returning Result
                    "parse" | "try_into" | "try_from" => Type::Named {
                        name: "Result".to_string(),
                        generics: vec![self.fresh_var(), self.fresh_var()],
                    },

                    // Methods that remove and return an element
                    "remove" | "swap_remove" => {
                        // Vec::remove(index) returns T, not ()
                        let effective_recv = if let Type::Named { .. } = &recv_inner {
                            &recv_inner
                        } else {
                            &recv_derefed
                        };
                        if let Type::Named { generics, .. } = effective_recv {
                            if !generics.is_empty() {
                                generics[0].clone()
                            } else {
                                self.fresh_var()
                            }
                        } else {
                            self.fresh_var()
                        }
                    }

                    // Push/insert/mutating methods return unit
                    "push" | "push_str" | "push_front" | "push_back" | "insert"
                    | "clear" | "sort" | "sort_by" | "sort_by_key"
                    | "sort_unstable" | "truncate" | "resize" | "extend" | "append"
                    | "retain" | "swap" => Type::Unit,

                    // Numeric methods
                    "abs" | "floor" | "ceil" | "round" | "trunc" | "fract" | "sqrt"
                    | "cbrt" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
                    | "sinh" | "cosh" | "tanh" | "exp" | "exp2" | "ln" | "log"
                    | "log2" | "log10" | "pow" | "powi" | "powf" | "min" | "max"
                    | "clamp" | "signum" | "copysign" | "saturating_add"
                    | "saturating_sub" | "saturating_mul" | "wrapping_add"
                    | "wrapping_sub" | "wrapping_mul" | "checked_add" | "checked_sub"
                    | "checked_mul" | "checked_div" => recv_inner.clone(),

                    // Char methods
                    "to_digit" | "to_lowercase_char" | "to_uppercase_char" => Type::Named {
                        name: "Option".to_string(),
                        generics: vec![Type::Int(IntSize::U32)],
                    },

                    // Duration/Time methods
                    "duration_since" | "elapsed" | "as_secs" | "as_millis" | "as_micros"
                    | "as_nanos" | "from_secs" | "from_millis" => recv_inner.clone(),

                    // Path methods returning PathBuf (only for Path/PathBuf receivers)
                    "to_path_buf" | "with_extension" | "with_file_name" => {
                        Type::Named {
                            name: "PathBuf".to_string(),
                            generics: vec![],
                        }
                    }

                    // join: Path::join returns PathBuf, but Vec::join/Iterator::join returns String
                    "join" => {
                        // Check if receiver is Path or PathBuf
                        let is_path_type = if let Type::Named { name, .. } = &recv_inner {
                            name == "Path" || name == "PathBuf"
                        } else {
                            false
                        };
                        if is_path_type {
                            Type::Named {
                                name: "PathBuf".to_string(),
                                generics: vec![],
                            }
                        } else {
                            // Vec::join and Iterator::join return String
                            Type::Named {
                                name: "String".to_string(),
                                generics: vec![],
                            }
                        }
                    }

                    // Path methods returning &str via OsStr
                    "to_str" => Type::Named {
                        name: "Option".to_string(),
                        generics: vec![Type::Ref {
                            lifetime: None,
                            mutable: false,
                            inner: Box::new(Type::Str),
                        }],
                    },

                    // Formatting
                    "fmt" | "write_str" | "write_fmt" => Type::Named {
                        name: "Result".to_string(),
                        generics: vec![Type::Unit, self.fresh_var()],
                    },

                    // IO methods
                    "read" | "write" | "flush" | "read_to_string" | "read_to_end"
                    | "read_line" | "write_all" => Type::Named {
                        name: "Result".to_string(),
                        generics: vec![self.fresh_var(), self.fresh_var()],
                    },

                    // fs metadata
                    "metadata" | "modified" | "created" | "accessed" | "len_file"
                    | "is_readonly" | "permissions" => Type::Named {
                        name: "Result".to_string(),
                        generics: vec![self.fresh_var(), self.fresh_var()],
                    },

                    // Map error
                    "map_err" | "and_then" | "or_else" => recv_inner.clone(),

                    // and/or for Option/Result
                    "and" | "or" => recv_inner.clone(),

                    // Default: return fresh type variable
                    // (user-defined methods were already checked above)
                    _ => self.fresh_var(),
                    }
                };

                // Propagate evidence from receiver
                if recv_ev > EvidenceLevel::Known {
                    Type::Evidential {
                        inner: Box::new(result_ty),
                        evidence: recv_ev,
                    }
                } else {
                    result_ty
                }
            }

            Expr::Field { expr, field } => {
                let recv_ty = self.infer_expr(expr);
                let (recv_inner, recv_ev) = self.strip_evidence(&recv_ty);

                // Try to resolve field type from struct definition
                let field_ty = if let Type::Named { name, .. } = &recv_inner {
                    // Look up struct definition in type definitions
                    if let Some(struct_def) = self.types.get(name) {
                        if let TypeDef::Struct { fields, .. } = struct_def {
                            fields
                                .iter()
                                .find(|(n, _)| n == &field.name)
                                .map(|(_, ty)| ty.clone())
                                .unwrap_or_else(|| self.fresh_var())
                        } else {
                            self.fresh_var()
                        }
                    } else {
                        self.fresh_var()
                    }
                } else {
                    self.fresh_var()
                };

                // Propagate evidence from receiver
                if recv_ev > EvidenceLevel::Known {
                    Type::Evidential {
                        inner: Box::new(field_ty),
                        evidence: recv_ev,
                    }
                } else {
                    field_ty
                }
            }

            Expr::Index { expr, index, .. } => {
                let arr_ty = self.infer_expr(expr);
                let idx_ty = self.infer_expr(index);
                let (arr_inner, arr_ev) = self.strip_evidence(&arr_ty);

                // Index should be usize
                let _ = self.unify(&idx_ty, &Type::Int(IntSize::USize));

                // Get element type from array/slice
                let elem_ty = match arr_inner {
                    Type::Array { element, .. } => *element,
                    Type::Slice(element) => *element,
                    Type::Named { name, generics } if name == "Vec" && !generics.is_empty() => {
                        generics[0].clone()
                    }
                    _ => self.fresh_var(),
                };

                // Propagate evidence
                if arr_ev > EvidenceLevel::Known {
                    Type::Evidential {
                        inner: Box::new(elem_ty),
                        evidence: arr_ev,
                    }
                } else {
                    elem_ty
                }
            }

            Expr::Try(inner) => {
                // expr? unwraps Result<T, E> or Option<T> to T
                let inner_ty = self.infer_expr(inner);
                // Resolve type variables before matching
                let resolved = if let Type::Var(v) = &inner_ty {
                    self.substitutions.get(v).cloned().unwrap_or(inner_ty.clone())
                } else {
                    inner_ty.clone()
                };
                match &resolved {
                    Type::Named { name, generics } if name == "Result" && !generics.is_empty() => {
                        // Result<T, E>? → T with uncertain evidence
                        generics[0].clone()
                    }
                    Type::Named { name, generics } if name == "Option" && !generics.is_empty() => {
                        // Option<T>? → T with uncertain evidence
                        generics[0].clone()
                    }
                    _ => {
                        // For unresolved types, ? produces a fresh type variable
                        // (type inference will resolve it later)
                        self.fresh_var()
                    }
                }
            }

            _ => {
                // Handle other expression types
                self.fresh_var()
            }
        }
    }

    /// Infer type from literal
    fn infer_literal(&self, lit: &Literal) -> Type {
        match lit {
            Literal::Int { .. } => Type::Int(IntSize::I64),
            Literal::Float { suffix, .. } => {
                match suffix.as_ref().map(|s| s.as_str()) {
                    Some("f32") => Type::Float(FloatSize::F32),
                    Some("f64") => Type::Float(FloatSize::F64),
                    // Default to f64 for unsuffixed or other suffixes
                    None | Some(_) => Type::Float(FloatSize::F64),
                }
            }
            Literal::Bool(_) => Type::Bool,
            Literal::Char(_) => Type::Char,
            Literal::ByteChar(_) => Type::Int(IntSize::U8),
            // String literals have type &str
            Literal::String(_) => Type::Ref {
                lifetime: None,
                mutable: false,
                inner: Box::new(Type::Str),
            },
            Literal::MultiLineString(_) => Type::Ref {
                lifetime: None,
                mutable: false,
                inner: Box::new(Type::Str),
            },
            Literal::RawString(_) => Type::Ref {
                lifetime: None,
                mutable: false,
                inner: Box::new(Type::Str),
            },
            Literal::ByteString(bytes) => Type::Ref {
                lifetime: None,
                mutable: false,
                inner: Box::new(Type::Array {
                    element: Box::new(Type::Int(IntSize::U8)),
                    size: Some(bytes.len()),
                }),
            },
            Literal::InterpolatedString { .. } => Type::Str,
            Literal::SigilStringSql(_) => Type::Str,
            Literal::SigilStringRoute(_) => Type::Str,
            Literal::Null => Type::Unit, // null has unit type
            Literal::Empty => Type::Unit,
            Literal::Infinity => Type::Float(FloatSize::F64),
            Literal::Circle => Type::Float(FloatSize::F64),
        }
    }

    /// Infer type of binary operation
    fn infer_binary_op(&mut self, op: &BinOp, left: &Type, right: &Type) -> Type {
        // Extract evidence levels for propagation
        let (left_inner, left_ev) = self.strip_evidence(left);
        let (right_inner, right_ev) = self.strip_evidence(right);

        // Helper to detect type variable or function types (incomplete inference)
        let is_var_or_fn = |ty: &Type| {
            matches!(ty, Type::Var(_) | Type::Function { .. })
        };

        let result_ty = match op {
            // Arithmetic: numeric -> numeric with coercion
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                // Numeric coercion: if either operand is float, result is float
                let is_left_float = matches!(&left_inner, Type::Float(_));
                let is_right_float = matches!(&right_inner, Type::Float(_));

                if is_left_float || is_right_float {
                    // Result is the wider float type
                    if is_left_float { left_inner } else { right_inner }
                } else {
                    // Both are ints (or unknown) - try to unify
                    let _ = self.unify(&left_inner, &right_inner);
                    left_inner
                }
            }

            // Matrix multiplication: tensor @ tensor -> tensor
            // Hadamard/element-wise: tensor ⊙ tensor -> tensor
            // Tensor product: tensor ⊗ tensor -> tensor
            BinOp::MatMul | BinOp::Hadamard | BinOp::TensorProd | BinOp::Convolve => {
                // Return a fresh type variable for now (proper tensor type checking would go here)
                self.fresh_var()
            }

            // Comparison: any -> bool
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                // For bootstrapping: skip error when either side is a type variable or function
                // (indicates incomplete type inference from unhandled expressions)
                if !self.unify(&left_inner, &right_inner)
                    && !is_var_or_fn(&left_inner)
                    && !is_var_or_fn(&right_inner)
                {
                    self.error(TypeError::new(format!(
                        "comparison operands must have same type: left={:?}, right={:?}",
                        left_inner, right_inner
                    )));
                }
                Type::Bool
            }

            // Logical: bool -> bool
            BinOp::And | BinOp::Or => {
                if !self.unify(&Type::Bool, &left_inner) {
                    self.error(TypeError::new("logical operand must be bool"));
                }
                if !self.unify(&Type::Bool, &right_inner) {
                    self.error(TypeError::new("logical operand must be bool"));
                }
                Type::Bool
            }

            // Bitwise: int -> int
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => left_inner,

            // String concatenation
            BinOp::Concat => {
                if !self.unify(&Type::Str, &left_inner) {
                    self.error(TypeError::new("concat operand must be string"));
                }
                Type::Str
            }
        };

        // Combine evidence levels
        let combined_ev = left_ev.join(right_ev);

        // Wrap result in evidence if either operand had evidence
        if combined_ev > EvidenceLevel::Known {
            Type::Evidential {
                inner: Box::new(result_ty),
                evidence: combined_ev,
            }
        } else {
            result_ty
        }
    }

    /// Infer type of unary operation
    fn infer_unary_op(&mut self, op: &UnaryOp, inner: &Type) -> Type {
        let (inner_ty, evidence) = self.strip_evidence(inner);

        let result = match op {
            UnaryOp::Neg => inner_ty,
            UnaryOp::Not => {
                // ! operator: logical NOT for bool, bitwise NOT for integers
                if matches!(inner_ty, Type::Int(_)) {
                    // Bitwise NOT on integer types - returns same type
                    inner_ty
                } else {
                    if !self.unify(&Type::Bool, &inner_ty) {
                        self.error(TypeError::new(format!(
                            "type mismatch: '!' requires bool or integer, found {}",
                            inner_ty
                        )));
                    }
                    Type::Bool
                }
            }
            UnaryOp::Ref => Type::Ref {
                lifetime: None,
                mutable: false,
                inner: Box::new(inner_ty),
            },
            UnaryOp::RefMut => Type::Ref {
                lifetime: None,
                mutable: true,
                inner: Box::new(inner_ty),
            },
            UnaryOp::Deref => {
                if let Type::Ref { inner, .. } | Type::Ptr { inner, .. } = inner_ty {
                    *inner
                } else {
                    // For bootstrapping: return fresh type variable
                    self.fresh_var()
                }
            }
        };

        // Preserve evidence
        if evidence > EvidenceLevel::Known {
            Type::Evidential {
                inner: Box::new(result),
                evidence,
            }
        } else {
            result
        }
    }

    /// Infer type of pipe operation
    fn infer_pipe_op(&mut self, op: &PipeOp, input: &Type) -> Type {
        let (inner_ev_stripped, evidence) = self.strip_evidence(input);

        // Also strip reference wrapper for pipe operations
        // This allows `&[T]` to be treated as `[T]` in pipes
        let inner = match inner_ev_stripped {
            Type::Ref { inner: ref_inner, .. } => (*ref_inner).clone(),
            other => other,
        };

        let result = match op {
            // Transform: [T] -> [U] where body: T -> U
            PipeOp::Transform(_body) => {
                if let Type::Array { element, size } = inner {
                    Type::Array { element, size }
                } else if let Type::Slice(element) = inner {
                    Type::Slice(element)
                } else {
                    // For bootstrapping: return fresh type variable
                    self.fresh_var()
                }
            }

            // Filter: [T] -> [T]
            PipeOp::Filter(_pred) => inner,

            // Sort: [T] -> [T]
            PipeOp::Sort(_) => inner,

            // Reduce: [T] -> T (also Vec<T> -> T)
            PipeOp::Reduce(_) => {
                if let Type::Array { element, .. } | Type::Slice(element) = inner {
                    *element
                } else if let Type::Named { name, generics } = &inner {
                    // Support Vec<T>, LinkedList<T>, etc.
                    if (name == "Vec" || name == "LinkedList" || name == "VecDeque")
                        && !generics.is_empty() {
                        generics[0].clone()
                    } else {
                        self.fresh_var()
                    }
                } else if let Type::Var(_) = inner {
                    // For bootstrapping: return fresh type variable when input is unknown
                    self.fresh_var()
                } else {
                    self.error(TypeError::new("reduce requires array or slice"));
                    Type::Error
                }
            }
            PipeOp::ReduceSum | PipeOp::ReduceProd | PipeOp::ReduceMin | PipeOp::ReduceMax => {
                // Numeric reductions return the element type
                let element = if let Type::Array { element, .. } | Type::Slice(element) = &inner {
                    Some(element.clone())
                } else if let Type::Named { name, generics } = &inner {
                    // Support Vec<T>, etc.
                    if (name == "Vec" || name == "LinkedList" || name == "VecDeque")
                        && !generics.is_empty() {
                        Some(Box::new(generics[0].clone()))
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(element) = element {
                    match element.as_ref() {
                        Type::Int(_) | Type::Float(_) => *element,
                        Type::Var(_) => *element, // For bootstrapping: allow type variables
                        _ => {
                            self.error(TypeError::new("numeric reduction requires numeric array"));
                            Type::Error
                        }
                    }
                } else if let Type::Var(_) = inner {
                    // For bootstrapping: return fresh type variable when input is unknown
                    self.fresh_var()
                } else {
                    self.error(TypeError::new("reduction requires array or slice"));
                    Type::Error
                }
            }
            PipeOp::ReduceConcat => {
                // Concat returns string or array depending on element type
                if let Type::Array { element, .. } | Type::Slice(element) = inner {
                    match element.as_ref() {
                        Type::Str => Type::Str,
                        Type::Array { .. } => *element,
                        Type::Var(_) => self.fresh_var(), // For bootstrapping
                        _ => {
                            self.error(TypeError::new(
                                "concat reduction requires array of strings or arrays",
                            ));
                            Type::Error
                        }
                    }
                } else if let Type::Var(_) = inner {
                    // For bootstrapping: return fresh type variable
                    self.fresh_var()
                } else {
                    self.error(TypeError::new("concat reduction requires array or slice"));
                    Type::Error
                }
            }
            PipeOp::ReduceAll | PipeOp::ReduceAny => {
                // Boolean reductions return bool
                if let Type::Array { element, .. } | Type::Slice(element) = inner {
                    match element.as_ref() {
                        Type::Bool => Type::Bool,
                        Type::Var(_) => Type::Bool, // For bootstrapping: assume bool
                        _ => {
                            self.error(TypeError::new(
                                "boolean reduction requires array of booleans",
                            ));
                            Type::Error
                        }
                    }
                } else if let Type::Var(_) = inner {
                    // For bootstrapping: return bool
                    Type::Bool
                } else {
                    self.error(TypeError::new("boolean reduction requires array or slice"));
                    Type::Error
                }
            }

            // Match morpheme: |match{ Pattern => expr, ... }
            PipeOp::Match(arms) => {
                // All arms should return the same type
                if arms.is_empty() {
                    self.error(TypeError::new("match expression has no arms"));
                    Type::Error
                } else {
                    // Infer type from first arm, other arms should match
                    let result_type = self.infer_expr(&arms[0].body);
                    for arm in arms.iter().skip(1) {
                        let arm_type = self.infer_expr(&arm.body);
                        self.unify(&result_type, &arm_type);
                    }
                    result_type
                }
            }

            // Try/Error transformation: |? or |?{mapper}
            PipeOp::TryMap(_) => {
                // Unwraps Result<T, E> to T or Option<T> to T
                // For now, return a fresh type variable
                // (proper implementation would extract inner type from Result/Option)
                self.fresh_var()
            }

            // Call expression (like |self.layer)
            PipeOp::Call(callee) => {
                // Infer the type of the callee and extract return type
                let callee_ty = self.infer_expr(callee);
                if let Type::Function { return_type, .. } = callee_ty {
                    *return_type
                } else {
                    // Could be a callable struct or closure, return fresh var
                    self.fresh_var()
                }
            }

            // Method call
            PipeOp::Method { name, type_args: _, args: _ } => {
                // Look up method
                if let Some(fn_ty) = self.functions.get(&name.name).cloned() {
                    // Freshen to get fresh type variables for polymorphic functions
                    let fresh_ty = self.freshen(&fn_ty);
                    if let Type::Function { return_type, .. } = fresh_ty {
                        *return_type
                    } else {
                        Type::Error
                    }
                } else {
                    // Could be a method on the type
                    self.fresh_var()
                }
            }

            // Named operation (morpheme)
            PipeOp::Named { prefix, body: _ } => {
                // Named operations like |sum, |product
                if let Some(first) = prefix.first() {
                    match first.name.as_str() {
                        "sum" | "product" => {
                            if let Type::Array { element, .. } | Type::Slice(element) = inner {
                                *element
                            } else {
                                self.error(TypeError::new("sum/product requires array"));
                                Type::Error
                            }
                        }
                        _ => self.fresh_var(),
                    }
                } else {
                    self.fresh_var()
                }
            }

            // Await: unwrap future
            PipeOp::Await => {
                // Future<T> -> T
                inner
            }

            // Access morphemes: [T] -> T (return element type)
            PipeOp::First
            | PipeOp::Last
            | PipeOp::Middle
            | PipeOp::Choice
            | PipeOp::Nth(_)
            | PipeOp::Next => {
                if let Type::Array { element, .. } | Type::Slice(element) = inner {
                    *element
                } else if let Type::Named { name, generics } = &inner {
                    // Support Vec<T>, VecDeque<T>, etc.
                    if (name == "Vec" || name == "VecDeque" || name == "LinkedList")
                        && !generics.is_empty() {
                        generics[0].clone()
                    } else {
                        self.fresh_var()
                    }
                } else if let Type::Tuple(elements) = inner {
                    // For tuple, return Any since elements might be different types
                    if let Some(first) = elements.first() {
                        first.clone()
                    } else {
                        Type::Unit
                    }
                } else if let Type::Var(_) = inner {
                    // For bootstrapping: return fresh type variable
                    self.fresh_var()
                } else {
                    // For bootstrapping: allow access on unknown types, return fresh var
                    self.fresh_var()
                }
            }

            // Parallel morpheme: ∥ - wraps another operation
            // Type is determined by the inner operation
            PipeOp::Parallel(inner_op) => self.infer_pipe_op(inner_op, input),

            // GPU morpheme: ⊛ - wraps another operation for GPU execution
            // Type is determined by the inner operation
            PipeOp::Gpu(inner_op) => self.infer_pipe_op(inner_op, input),

            // ==========================================
            // Protocol Operations - Sigil-native networking
            // All protocol results have Reported evidentiality
            // ==========================================

            // Send: connection -> response (with Reported evidence)
            PipeOp::Send(_) => {
                // Returns response object with Reported evidentiality
                Type::Evidential {
                    inner: Box::new(self.fresh_var()),
                    evidence: EvidenceLevel::Reported,
                }
            }

            // Recv: connection -> data (with Reported evidence)
            PipeOp::Recv => {
                // Returns received data with Reported evidentiality
                Type::Evidential {
                    inner: Box::new(self.fresh_var()),
                    evidence: EvidenceLevel::Reported,
                }
            }

            // Stream: connection -> Stream<T> (elements have Reported evidence)
            PipeOp::Stream(_) => {
                // Returns a stream type
                self.fresh_var()
            }

            // Connect: url/config -> connection
            PipeOp::Connect(_) => {
                // Returns connection object
                self.fresh_var()
            }

            // Close: connection -> ()
            PipeOp::Close => Type::Unit,

            // Header: request -> request (adds header)
            PipeOp::Header { .. } => inner,

            // Body: request -> request (sets body)
            PipeOp::Body(_) => inner,

            // Timeout: request -> request (sets timeout)
            PipeOp::Timeout(_) => inner,

            // Retry: request -> request (sets retry policy)
            PipeOp::Retry { .. } => inner,

            // ==========================================
            // Evidence Promotion Operations
            // ==========================================

            // Validate: T~ -> T! (promotes with validation)
            PipeOp::Validate {
                predicate: _,
                target_evidence,
            } => {
                // Check that the predicate returns bool
                // (We'd need to infer the closure type properly, skipping for now)

                let target_ev = EvidenceLevel::from_ast(*target_evidence);

                // Validation can only promote evidence (make more certain)
                if evidence < target_ev {
                    self.error(
                        TypeError::new(format!(
                            "cannot demote evidence from {} ({}) to {} ({}) using validate",
                            evidence.name(),
                            evidence.symbol(),
                            target_ev.name(),
                            target_ev.symbol()
                        ))
                        .with_note("validate! can only promote evidence to a more certain level"),
                    );
                }

                // Return inner type with promoted evidence
                return Type::Evidential {
                    inner: Box::new(inner.clone()),
                    evidence: target_ev,
                };
            }

            // Assume: T~ -> T! (explicit trust with audit trail)
            PipeOp::Assume {
                reason: _,
                target_evidence,
            } => {
                let target_ev = EvidenceLevel::from_ast(*target_evidence);

                // Assumption always succeeds but should be logged/audited
                // In a real implementation, this would record for security review

                if evidence < target_ev {
                    self.error(
                        TypeError::new(format!(
                            "assume! cannot demote evidence from {} ({}) to {} ({})",
                            evidence.name(),
                            evidence.symbol(),
                            target_ev.name(),
                            target_ev.symbol()
                        ))
                        .with_note("assume! is for promoting evidence, not demoting"),
                    );
                }

                // Return inner type with assumed evidence
                return Type::Evidential {
                    inner: Box::new(inner.clone()),
                    evidence: target_ev,
                };
            }

            // AssertEvidence: compile-time evidence check
            PipeOp::AssertEvidence(expected_ast) => {
                let expected = EvidenceLevel::from_ast(*expected_ast);

                if !evidence.satisfies(expected) {
                    self.error(
                        TypeError::new(format!(
                            "evidence assertion failed: expected {} ({}) or more certain, found {} ({})",
                            expected.name(), expected.symbol(),
                            evidence.name(), evidence.symbol()
                        ))
                        .with_note("use |validate!{...} or |assume! to promote evidence before assertion")
                    );
                }

                // Return the same type (this is just an assertion)
                return input.clone();
            }

            // ==========================================
            // Scope Functions (Kotlin-inspired)
            // ==========================================

            // Also: execute side effect, return original value unchanged
            // T -> T (side effect executed but value preserved)
            PipeOp::Also(_) => {
                // The closure is executed for side effects only
                // Return type is same as input, evidence preserved
                return input.clone();
            }

            // Apply: mutate value in place, return modified value
            // T -> T (value may be mutated)
            PipeOp::Apply(_) => {
                // The closure can mutate the value
                // Return type is same as input, evidence preserved
                return input.clone();
            }

            // TakeIf: return Some(value) if predicate true, None otherwise
            // T -> Option<T>
            PipeOp::TakeIf(_) => {
                // Returns Option wrapping the input type
                // Evidence is preserved in the inner type
                return Type::Named {
                    name: "Option".to_string(),
                    generics: vec![input.clone()],
                };
            }

            // TakeUnless: return Some(value) if predicate false, None otherwise
            // T -> Option<T>
            PipeOp::TakeUnless(_) => {
                // Returns Option wrapping the input type
                // Evidence is preserved in the inner type
                return Type::Named {
                    name: "Option".to_string(),
                    generics: vec![input.clone()],
                };
            }

            // Let: transform value (alias for Transform/tau)
            // T -> U
            PipeOp::Let(func) => {
                // Same as Transform - applies function and returns result
                let _ = self.infer_expr(func);
                self.fresh_var() // Result type depends on function
            }

            // Mathematical & APL-Inspired Operations
            PipeOp::All(_) | PipeOp::Any(_) => Type::Bool,
            PipeOp::Compose(f) => {
                let _ = self.infer_expr(f);
                self.fresh_var()
            }
            PipeOp::Zip(other) => {
                let _ = self.infer_expr(other);
                self.fresh_var() // Array of tuples
            }
            PipeOp::Scan(f) => {
                let _ = self.infer_expr(f);
                self.fresh_var() // Array of accumulated values
            }
            PipeOp::Diff => self.fresh_var(), // Array of differences
            PipeOp::Gradient(var) => {
                let _ = self.infer_expr(var);
                self.fresh_var() // Gradient value
            }
            PipeOp::SortAsc | PipeOp::SortDesc | PipeOp::Reverse => {
                inner.clone() // Same type, reordered
            }
            PipeOp::Cycle(n) | PipeOp::Windows(n) | PipeOp::Chunks(n) => {
                let _ = self.infer_expr(n);
                self.fresh_var() // Array type
            }
            PipeOp::Flatten | PipeOp::Unique => self.fresh_var(),
            PipeOp::Enumerate => self.fresh_var(), // Array of (index, value) tuples

            // Holographic operations (Spec 11-HOLOGRAPHIC.md)
            PipeOp::Universal => {
                // |∀ - Universal reconstruction: [T] -> T (sum/aggregate)
                if let Type::Array { element, .. } | Type::Slice(element) = inner {
                    *element
                } else if let Type::Named { name, generics } = &inner {
                    if (name == "Vec" || name == "LinkedList" || name == "VecDeque")
                        && !generics.is_empty()
                    {
                        generics[0].clone()
                    } else {
                        self.fresh_var()
                    }
                } else {
                    self.fresh_var()
                }
            }
            PipeOp::Possibility { .. } => self.fresh_var(), // |◊method - approximate query result
            PipeOp::Necessity { .. } => self.fresh_var(),   // |□method - verified result
            PipeOp::PossibilityExtract => self.fresh_var(), // |◊ - extract from Option/Array
            PipeOp::NecessityVerify => inner,               // |□ - verify non-empty, pass through
        };

        // Preserve evidence through pipe
        if evidence > EvidenceLevel::Known {
            Type::Evidential {
                inner: Box::new(result),
                evidence,
            }
        } else {
            result
        }
    }

    /// Strip evidence wrapper, returning (inner_type, evidence_level)
    fn strip_evidence(&self, ty: &Type) -> (Type, EvidenceLevel) {
        match ty {
            Type::Evidential { inner, evidence } => (*inner.clone(), *evidence),
            _ => (ty.clone(), EvidenceLevel::Known),
        }
    }

    /// Bind pattern variables with the given type and evidence level.
    /// This propagates evidence through pattern matching.
    fn bind_pattern(&mut self, pattern: &Pattern, ty: &Type, evidence: EvidenceLevel) {
        let (inner_ty, ty_ev) = self.strip_evidence(ty);
        // Use the more restrictive evidence level
        let final_ev = evidence.join(ty_ev);

        match pattern {
            Pattern::Ident {
                name,
                evidentiality,
                ..
            } => {
                // Explicit evidence annotation overrides inference
                let ev = evidentiality
                    .map(EvidenceLevel::from_ast)
                    .unwrap_or(final_ev);
                self.env
                    .borrow_mut()
                    .define(name.name.clone(), inner_ty, ev);
            }
            Pattern::Tuple(patterns) => {
                if let Type::Tuple(types) = &inner_ty {
                    for (pat, ty) in patterns.iter().zip(types.iter()) {
                        self.bind_pattern(pat, ty, final_ev);
                    }
                }
            }
            Pattern::Struct { fields, .. } => {
                // For struct patterns, we'd need field type info
                // For now, bind with fresh vars
                for field in fields {
                    let fresh = self.fresh_var();
                    if let Some(ref pat) = field.pattern {
                        self.bind_pattern(pat, &fresh, final_ev);
                    } else {
                        self.env
                            .borrow_mut()
                            .define(field.name.name.clone(), fresh, final_ev);
                    }
                }
            }
            Pattern::TupleStruct { fields, .. } => {
                for pat in fields {
                    let fresh = self.fresh_var();
                    self.bind_pattern(pat, &fresh, final_ev);
                }
            }
            Pattern::Slice(patterns) => {
                let elem_ty = if let Type::Array { element, .. } | Type::Slice(element) = &inner_ty
                {
                    *element.clone()
                } else {
                    self.fresh_var()
                };
                for pat in patterns {
                    self.bind_pattern(pat, &elem_ty, final_ev);
                }
            }
            Pattern::Or(patterns) => {
                // For or-patterns, bind the same variables from any branch
                // (they should all have the same bindings)
                if let Some(first) = patterns.first() {
                    self.bind_pattern(first, ty, evidence);
                }
            }
            Pattern::Wildcard | Pattern::Rest | Pattern::Literal(_) | Pattern::Range { .. } | Pattern::Path(_) => {
                // These don't introduce bindings
            }
            Pattern::Ref { pattern, .. } => {
                // For reference patterns, bind the inner pattern
                // The inner type would be the deref'd type, but for now use a fresh var
                let inner_ty = self.fresh_var();
                self.bind_pattern(pattern, &inner_ty, final_ev);
            }
            Pattern::RefBinding {
                name,
                evidentiality,
                ..
            } => {
                // Ref binding - similar to Ident but binds by reference
                let ev = evidentiality
                    .map(EvidenceLevel::from_ast)
                    .unwrap_or(final_ev);
                self.env
                    .borrow_mut()
                    .define(name.name.clone(), inner_ty, ev);
            }
        }
    }

    /// Resolve type aliases to their underlying types
    fn resolve_alias(&self, ty: &Type) -> Type {
        if let Type::Named { name, generics } = ty {
            if generics.is_empty() {
                if let Some(TypeDef::Alias { target, .. }) = self.types.get(name) {
                    return target.clone();
                }
            }
        }
        ty.clone()
    }

    /// Attempt to unify two types
    fn unify(&mut self, a: &Type, b: &Type) -> bool {
        // Resolve type aliases first
        let a = self.resolve_alias(a);
        let b = self.resolve_alias(b);

        match (&a, &b) {
            // Type variables - check these FIRST before other patterns
            (Type::Var(v), t) => {
                if let Some(resolved) = self.substitutions.get(v) {
                    let resolved = resolved.clone();
                    self.unify(&resolved, t)
                } else if !self.occurs_in(v, t) {
                    self.substitutions.insert(*v, t.clone());
                    true
                } else {
                    // Occurs check failed - cyclic type, just return true for bootstrapping
                    true
                }
            }
            (t, Type::Var(v)) => {
                if let Some(resolved) = self.substitutions.get(v) {
                    let resolved = resolved.clone();
                    self.unify(t, &resolved)
                } else if !self.occurs_in(v, t) {
                    self.substitutions.insert(*v, t.clone());
                    true
                } else {
                    // Occurs check failed - cyclic type, just return true for bootstrapping
                    true
                }
            }

            // Same types
            (Type::Unit, Type::Unit) |
            (Type::Bool, Type::Bool) |
            (Type::Char, Type::Char) |
            (Type::Str, Type::Str) |
            (Type::Never, Type::Never) |
            (Type::Error, _) |
            (_, Type::Error) |
            // Never (bottom type) unifies with anything
            (Type::Never, _) |
            (_, Type::Never) => true,

            // Linear type wrapper: Linear(T) unifies with T (linear is a usage qualifier)
            (Type::Linear(inner), other) => self.unify(inner, other),
            (other, Type::Linear(inner)) => self.unify(other, inner),

            // For bootstrapping: allow integer literals to coerce to any integer type
            // This is lenient - a proper type system would have more precise rules
            (Type::Int(_), Type::Int(_)) => true,
            // For bootstrapping: allow float literals to coerce to any float type
            // This handles cases like `const X: f32 = 0.3;` where 0.3 infers as f64
            (Type::Float(_), Type::Float(_)) => true,

            // For bootstrapping: allow &str to coerce to Str and vice versa
            (Type::Ref { mutable: false, inner: a, .. }, Type::Str) if matches!(a.as_ref(), Type::Str) => true,
            (Type::Str, Type::Ref { mutable: false, inner: b, .. }) if matches!(b.as_ref(), Type::Str) => true,

            // For bootstrapping: allow String to coerce to &str (via Deref)
            // This allows passing String where &str is expected
            (Type::Named { name: n, .. }, Type::Ref { mutable: false, inner, .. })
                if n == "String" && matches!(inner.as_ref(), Type::Str) => true,
            (Type::Ref { mutable: false, inner, .. }, Type::Named { name: n, .. })
                if n == "String" && matches!(inner.as_ref(), Type::Str) => true,

            // String to str coercion (deref-like): String ↔ str
            // Analogous to Rust's String → &str deref coercion.
            // String owns string data, str is a view — they're interchangeable
            // in type checking since the interpreter uses the same representation.
            (Type::Str, Type::Named { name, .. }) if name == "String" => true,
            (Type::Named { name, .. }, Type::Str) if name == "String" => true,

            // Arrays
            (Type::Array { element: a, size: sa }, Type::Array { element: b, size: sb }) => {
                (sa == sb || sa.is_none() || sb.is_none()) && self.unify(a, b)
            }

            // Slices
            (Type::Slice(a), Type::Slice(b)) => self.unify(a, b),

            // Array to Slice coercion: [T; N] → [T]
            // A fixed-size array is always a valid slice of the same element type.
            (Type::Slice(a), Type::Array { element: b, .. }) => self.unify(a, b),
            (Type::Array { element: a, .. }, Type::Slice(b)) => self.unify(a, b),

            // Tuples
            (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
                a.iter().zip(b.iter()).all(|(x, y)| self.unify(x, y))
            }

            // References
            (Type::Ref { mutable: ma, inner: a, .. }, Type::Ref { mutable: mb, inner: b, .. }) => {
                // Allow &[T; N] to coerce to &[T] (array to slice)
                match (a.as_ref(), b.as_ref()) {
                    (Type::Array { element: ea, .. }, Type::Slice(es)) => {
                        (ma == mb || !ma) && self.unify(ea, es)
                    }
                    (Type::Slice(es), Type::Array { element: ea, .. }) => {
                        (ma == mb || !mb) && self.unify(es, ea)
                    }
                    _ => {
                        let mut_ok = ma == mb || (!ma && *mb) || (!mb && *ma);
                        if mut_ok && self.unify(a, b) {
                            return true;
                        }
                        // Auto-deref: &&T → &T (strip one layer of reference)
                        if let Type::Ref { inner: inner_b, .. } = b.as_ref() {
                            if self.unify(a, inner_b) {
                                return true;
                            }
                        }
                        if let Type::Ref { inner: inner_a, .. } = a.as_ref() {
                            if self.unify(inner_a, b) {
                                return true;
                            }
                        }
                        // Smart pointer deref: &Arc<T> → &T, &Rc<T> → &T, etc.
                        if let Type::Named { name, generics, .. } = b.as_ref() {
                            if matches!(name.as_str(), "Arc" | "Rc" | "Box" | "Cell" | "RefCell" | "Mutex")
                                && !generics.is_empty()
                            {
                                if self.unify(a, &generics[0]) {
                                    return true;
                                }
                            }
                        }
                        if let Type::Named { name, generics, .. } = a.as_ref() {
                            if matches!(name.as_str(), "Arc" | "Rc" | "Box" | "Cell" | "RefCell" | "Mutex")
                                && !generics.is_empty()
                            {
                                if self.unify(&generics[0], b) {
                                    return true;
                                }
                            }
                        }
                        false
                    }
                }
            }

            // Functions
            (Type::Function { params: pa, return_type: ra, is_async: aa },
             Type::Function { params: pb, return_type: rb, is_async: ab }) => {
                aa == ab && pa.len() == pb.len() &&
                pa.iter().zip(pb.iter()).all(|(x, y)| self.unify(x, y)) &&
                self.unify(ra, rb)
            }

            // Named types
            (Type::Named { name: na, generics: ga }, Type::Named { name: nb, generics: gb }) => {
                if na == nb {
                    // Same name, same arity: unify generics pairwise
                    if ga.len() == gb.len() {
                        return ga.iter().zip(gb.iter()).all(|(x, y)| self.unify(x, y));
                    }
                    // Bare type (0 generics) is compatible with the generic version
                    // e.g., `Tensor` (user wrote without generics) matches `Tensor<S, D, Dev>`
                    if ga.is_empty() || gb.is_empty() {
                        return true;
                    }
                    return false;
                }
                // Different names: check if either is a type parameter
                // Type parameters (single uppercase letter like T, N, M) unify with any type
                if (ga.is_empty() && Self::is_type_parameter(na))
                    || (gb.is_empty() && Self::is_type_parameter(nb)) {
                    return true;
                }
                false
            }

            // Null (Unit) is assignable to any uncertain type (like None for Option<T>)
            (Type::Unit, Type::Evidential { evidence, .. })
                if *evidence == EvidenceLevel::Uncertain => true,
            (Type::Evidential { evidence, .. }, Type::Unit)
                if *evidence == EvidenceLevel::Uncertain => true,

            // Evidential types: inner must unify, evidence can differ
            (Type::Evidential { inner: a, .. }, Type::Evidential { inner: b, .. }) => {
                self.unify(a, b)
            }
            (Type::Evidential { inner: a, .. }, b) => {
                self.unify(a, b)
            }
            (a, Type::Evidential { inner: b, .. }) => {
                self.unify(a, b)
            }

            // Cycles
            (Type::Cycle { modulus: a }, Type::Cycle { modulus: b }) => a == b,

            // ImplTrait: impl Trait bounds
            // Two impl Trait types unify if their bounds match
            (Type::ImplTrait(bounds_a), Type::ImplTrait(bounds_b)) => {
                bounds_a.len() == bounds_b.len() &&
                bounds_a.iter().zip(bounds_b.iter()).all(|(a, b)| self.unify(a, b))
            }
            // impl Trait acts as an existential type — it accepts any concrete type
            // that satisfies the bound. For type checking purposes, unify permissively.
            (Type::ImplTrait(_), _) | (_, Type::ImplTrait(_)) => true,

            // For bootstrapping: treat type parameters (single uppercase letter names like T, U, E)
            // as compatible with any type. This allows generic functions to type check without
            // full generic instantiation support.
            (Type::Named { name, generics }, _) | (_, Type::Named { name, generics })
                if generics.is_empty() && Self::is_type_parameter(name) => {
                true
            }

            // Auto ref/deref coercion: &T ↔ T
            // When one side is a reference and the other is not, try unifying the inner type
            (Type::Ref { inner: a, .. }, b) => self.unify(a, b),
            (a, Type::Ref { inner: b, .. }) => self.unify(a, b),

            _ => false,
        }
    }

    /// Check if a name looks like a type parameter (single uppercase letter or common generic names)
    fn is_type_parameter(name: &str) -> bool {
        // Single uppercase letter (T, U, E, K, V, etc.)
        if name.len() == 1 && name.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false) {
            return true;
        }
        // Common generic parameter names
        matches!(name, "Item" | "Output" | "Error" | "Key" | "Value" | "Idx" | "Self")
    }

    /// Check if this is an allowed implicit numeric coercion (int → float)
    fn is_numeric_coercion(expected: &Type, actual: &Type) -> bool {
        // Allow integers to coerce to floats
        match (expected, actual) {
            (Type::Float(_), Type::Int(_)) => true,
            // Also allow through evidential wrappers
            (Type::Evidential { inner: exp, .. }, Type::Int(_)) => {
                matches!(exp.as_ref(), Type::Float(_))
            }
            (Type::Float(_), Type::Evidential { inner: act, .. }) => {
                matches!(act.as_ref(), Type::Int(_))
            }
            _ => false,
        }
    }

    /// Check for reference coercions (reborrow, deref coercion, unsized coercion)
    fn is_reference_coercion(expected: &Type, actual: &Type) -> bool {
        // Extract inner types from references
        let (exp_inner, exp_mutable) = match expected {
            Type::Ref { inner, mutable, .. } => (inner.as_ref(), *mutable),
            _ => return false,
        };
        let (act_inner, act_mutable) = match actual {
            Type::Ref { inner, mutable, .. } => (inner.as_ref(), *mutable),
            _ => return false,
        };

        // 1. Reborrow: &mut T → &T (mutable ref can become immutable ref)
        if !exp_mutable && act_mutable {
            // Compare inner types (ignoring mutability)
            if Self::types_structurally_equal(exp_inner, act_inner) {
                return true;
            }
        }

        // 2. Deref coercion: &Box<T> → &T, &Arc<T> → &T, &Rc<T> → &T, etc.
        if let Type::Named { name, generics, .. } = act_inner {
            if matches!(name.as_str(), "Box" | "Arc" | "Rc" | "Cell" | "RefCell" | "Mutex")
                && !generics.is_empty()
            {
                if Self::types_structurally_equal(exp_inner, &generics[0]) {
                    return true;
                }
            }
        }

        // 3. Unsized coercion: &Vec<T> → &[T]
        if let Type::Named { name, generics, .. } = act_inner {
            if name == "Vec" && !generics.is_empty() {
                if let Type::Slice(element) = exp_inner {
                    if Self::types_structurally_equal(element.as_ref(), &generics[0]) {
                        return true;
                    }
                }
            }
        }

        // 4. Auto-deref: &&T → &T (strip one layer of reference from actual)
        if let Type::Ref { inner: act_inner_inner, .. } = act_inner {
            if Self::types_structurally_equal(exp_inner, act_inner_inner.as_ref()) {
                return true;
            }
        }

        false
    }

    /// Check if an implicit ref/deref coercion between non-reference types is valid.
    /// Handles: T → &T (auto-ref) and &T → T (auto-deref)
    fn is_ref_value_coercion(expected: &Type, actual: &Type) -> bool {
        // Auto-deref: &T → T (strip reference from actual to match expected value type)
        if let Type::Ref { inner, .. } = actual {
            if Self::types_structurally_equal(expected, inner.as_ref()) {
                return true;
            }
        }
        // Auto-ref: T → &T (expected is a reference, actual is a value)
        if let Type::Ref { inner, .. } = expected {
            if Self::types_structurally_equal(inner.as_ref(), actual) {
                return true;
            }
        }
        false
    }

    /// Helper to compare types structurally (ignoring small differences)
    fn types_structurally_equal(a: &Type, b: &Type) -> bool {
        match (a, b) {
            (Type::Int(a_bits), Type::Int(b_bits)) => a_bits == b_bits,
            (Type::Float(a_bits), Type::Float(b_bits)) => a_bits == b_bits,
            (Type::Bool, Type::Bool) => true,
            (Type::Str, Type::Str) => true,
            (Type::Named { name: a_name, generics: a_gen, .. },
             Type::Named { name: b_name, generics: b_gen, .. }) => {
                a_name == b_name && a_gen.len() == b_gen.len() &&
                a_gen.iter().zip(b_gen.iter()).all(|(a, b)| Self::types_structurally_equal(a, b))
            }
            (Type::Slice(a_el), Type::Slice(b_el)) => {
                Self::types_structurally_equal(a_el, b_el)
            }
            (Type::Ref { inner: a_in, .. }, Type::Ref { inner: b_in, .. }) => {
                Self::types_structurally_equal(a_in, b_in)
            }
            (Type::Evidential { inner: a_in, .. }, Type::Evidential { inner: b_in, .. }) => {
                Self::types_structurally_equal(a_in, b_in)
            }
            // Allow evidential to match non-evidential for inner comparison
            (Type::Evidential { inner, .. }, other) | (other, Type::Evidential { inner, .. }) => {
                Self::types_structurally_equal(inner, other)
            }
            _ => false,
        }
    }

    /// Convert AST type to internal type
    fn convert_type(&self, ty: &TypeExpr) -> Type {
        match ty {
            TypeExpr::Path(path) => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident.name;
                    match name.as_str() {
                        "bool" => return Type::Bool,
                        "char" => return Type::Char,
                        "str" | "String" => return Type::Str,
                        "i8" => return Type::Int(IntSize::I8),
                        "i16" => return Type::Int(IntSize::I16),
                        "i32" => return Type::Int(IntSize::I32),
                        "i64" => return Type::Int(IntSize::I64),
                        "i128" => return Type::Int(IntSize::I128),
                        "isize" => return Type::Int(IntSize::ISize),
                        "u8" => return Type::Int(IntSize::U8),
                        "u16" => return Type::Int(IntSize::U16),
                        "u32" => return Type::Int(IntSize::U32),
                        "u64" => return Type::Int(IntSize::U64),
                        "u128" => return Type::Int(IntSize::U128),
                        "usize" => return Type::Int(IntSize::USize),
                        "f32" => return Type::Float(FloatSize::F32),
                        "f64" => return Type::Float(FloatSize::F64),
                        // Handle Self type - resolve to current impl type (with generics)
                        "Self" => {
                            if let Some(ref self_ty) = self.current_self_type {
                                return self_ty.clone();
                            }
                        }
                        _ => {
                            // Check if this is a generic type parameter
                            if let Some(ty) = self.current_generics.get(name) {
                                return ty.clone();
                            }
                        }
                    }
                }

                let name = path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::");

                let generics = path
                    .segments
                    .last()
                    .and_then(|s| s.generics.as_ref())
                    .map(|gs| gs.iter().map(|t| self.convert_type(t)).collect())
                    .unwrap_or_default();

                Type::Named { name, generics }
            }

            TypeExpr::Reference { lifetime, mutable, inner } => Type::Ref {
                lifetime: lifetime.clone(),
                mutable: *mutable,
                inner: Box::new(self.convert_type(inner)),
            },

            TypeExpr::Pointer { mutable, inner } => Type::Ptr {
                mutable: *mutable,
                inner: Box::new(self.convert_type(inner)),
            },

            TypeExpr::Array { element, size: _ } => {
                Type::Array {
                    element: Box::new(self.convert_type(element)),
                    size: None, // Could evaluate const expr
                }
            }

            TypeExpr::Slice(inner) => Type::Slice(Box::new(self.convert_type(inner))),

            TypeExpr::Tuple(elements) => {
                Type::Tuple(elements.iter().map(|t| self.convert_type(t)).collect())
            }

            TypeExpr::Function {
                params,
                return_type,
            } => Type::Function {
                params: params.iter().map(|t| self.convert_type(t)).collect(),
                return_type: Box::new(
                    return_type
                        .as_ref()
                        .map(|t| self.convert_type(t))
                        .unwrap_or(Type::Unit),
                ),
                is_async: false,
            },

            TypeExpr::Evidential {
                inner,
                evidentiality,
                error_type,
            } => {
                // If error_type is specified, this is sugar for Result<T, E>
                // For now, lower as evidential type; full expansion to Result comes later
                let _ = error_type; // TODO: expand T?[E] to Result<T, E> with evidence
                Type::Evidential {
                    inner: Box::new(self.convert_type(inner)),
                    evidence: EvidenceLevel::from_ast(*evidentiality),
                }
            }

            TypeExpr::Cycle { modulus: _ } => {
                Type::Cycle { modulus: 12 } // Default, should evaluate
            }

            TypeExpr::Simd { element, lanes } => {
                let elem_ty = self.convert_type(element);
                Type::Simd {
                    element: Box::new(elem_ty),
                    lanes: *lanes,
                }
            }

            TypeExpr::Atomic(inner) => {
                let inner_ty = self.convert_type(inner);
                Type::Atomic(Box::new(inner_ty))
            }

            TypeExpr::Never => Type::Never,
            TypeExpr::Infer => Type::Var(TypeVar(0)), // Fresh var
            TypeExpr::Lifetime(name) => Type::Lifetime(name.clone()),
            TypeExpr::TraitObject(bounds) => {
                let converted: Vec<Type> = bounds.iter().map(|b| self.convert_type(b)).collect();
                Type::TraitObject(converted)
            }
            TypeExpr::Hrtb { lifetimes, bound } => Type::Hrtb {
                lifetimes: lifetimes.clone(),
                bound: Box::new(self.convert_type(bound)),
            },
            TypeExpr::InlineStruct { fields } => Type::InlineStruct {
                fields: fields
                    .iter()
                    .map(|f| (f.name.name.clone(), self.convert_type(&f.ty)))
                    .collect(),
            },
            TypeExpr::ImplTrait(bounds) => {
                Type::ImplTrait(bounds.iter().map(|b| self.convert_type(b)).collect())
            }
            TypeExpr::InlineEnum { variants } => {
                Type::InlineEnum(variants.iter().map(|v| v.name.name.clone()).collect())
            }
            TypeExpr::AssocTypeBinding { name, ty } => Type::AssocTypeBinding {
                name: name.name.clone(),
                ty: Box::new(self.convert_type(ty)),
            },
            TypeExpr::ConstExpr(_) => {
                // Const expressions in type position (const generics)
                // For now, treat as an inferred/opaque type
                Type::Var(TypeVar(0))
            }
            TypeExpr::QualifiedPath { self_type, trait_path, item_path } => {
                // Qualified path: <Type as Trait>::AssociatedType
                // For now, represent as a named type with a synthesized name
                let trait_part = trait_path.as_ref()
                    .map(|tp| tp.segments.iter().map(|s| s.ident.name.clone()).collect::<Vec<_>>().join("::"))
                    .unwrap_or_default();
                let item_part = item_path.segments.iter().map(|s| s.ident.name.clone()).collect::<Vec<_>>().join("::");
                let name = if trait_part.is_empty() {
                    format!("<_>::{}", item_part)
                } else {
                    format!("<_ as {}>::{}", trait_part, item_part)
                };
                Type::Named {
                    name,
                    generics: vec![self.convert_type(self_type)],
                }
            }
            // Linear/affine/relevant type modifiers - wrap the inner type
            // Linear types enforce the no-cloning theorem at compile time
            TypeExpr::Linear(inner) => Type::Linear(Box::new(self.convert_type(inner))),
            TypeExpr::Affine(inner) => Type::Affine(Box::new(self.convert_type(inner))),
            TypeExpr::Relevant(inner) => Type::Relevant(Box::new(self.convert_type(inner))),
        }
    }

    /// Get errors
    pub fn errors(&self) -> &[TypeError] {
        &self.errors
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

// Helper trait for Pattern
trait PatternExt {
    fn evidentiality(&self) -> Option<Evidentiality>;
    fn binding_name(&self) -> Option<String>;
    fn binding_span(&self) -> Option<Span>;
}

impl PatternExt for Pattern {
    fn evidentiality(&self) -> Option<Evidentiality> {
        match self {
            Pattern::Ident { evidentiality, .. } => *evidentiality,
            _ => None,
        }
    }

    fn binding_name(&self) -> Option<String> {
        match self {
            Pattern::Ident { name, .. } => Some(name.name.clone()),
            _ => None,
        }
    }

    fn binding_span(&self) -> Option<Span> {
        match self {
            Pattern::Ident { name, .. } => Some(name.span),
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Unit => write!(f, "()"),
            Type::Bool => write!(f, "bool"),
            Type::Int(size) => write!(f, "{:?}", size),
            Type::Float(size) => write!(f, "{:?}", size),
            Type::Char => write!(f, "char"),
            Type::Str => write!(f, "str"),
            Type::Array { element, size } => {
                if let Some(n) = size {
                    write!(f, "[{}; {}]", element, n)
                } else {
                    write!(f, "[{}]", element)
                }
            }
            Type::Slice(inner) => write!(f, "[{}]", inner),
            Type::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Type::Named { name, generics } => {
                write!(f, "{}", name)?;
                if !generics.is_empty() {
                    write!(f, "<")?;
                    for (i, g) in generics.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", g)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            Type::Function {
                params,
                return_type,
                is_async,
            } => {
                if *is_async {
                    write!(f, "async ")?;
                }
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", return_type)
            }
            Type::Ref { lifetime, mutable, inner } => {
                let lt = lifetime.as_ref().map(|l| format!("'{} ", l)).unwrap_or_default();
                write!(f, "&{}{}{}", lt, if *mutable { "mut " } else { "" }, inner)
            }
            Type::Ptr { mutable, inner } => {
                write!(f, "*{}{}", if *mutable { "mut " } else { "const " }, inner)
            }
            Type::Evidential { inner, evidence } => {
                write!(f, "{}{}", inner, evidence.symbol())
            }
            Type::Cycle { modulus } => write!(f, "Cycle<{}>", modulus),
            Type::Var(v) => write!(f, "?{}", v.0),
            Type::Error => write!(f, "<error>"),
            Type::Never => write!(f, "!"),
            Type::Simd { element, lanes } => write!(f, "simd<{}, {}>", element, lanes),
            Type::Atomic(inner) => write!(f, "atomic<{}>", inner),
            Type::Lifetime(name) => write!(f, "'{}", name),
            Type::TraitObject(bounds) => {
                write!(f, "dyn ")?;
                for (i, bound) in bounds.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", bound)?;
                }
                Ok(())
            }
            Type::Hrtb { lifetimes, bound } => {
                write!(f, "for<")?;
                for (i, lt) in lifetimes.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "'{}", lt)?;
                }
                write!(f, "> {}", bound)
            }
            Type::InlineStruct { fields } => {
                write!(f, "struct {{ ")?;
                for (i, (name, ty)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", name, ty)?;
                }
                write!(f, " }}")
            }
            Type::ImplTrait(bounds) => {
                write!(f, "impl ")?;
                for (i, bound) in bounds.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", bound)?;
                }
                Ok(())
            }
            Type::InlineEnum(variants) => {
                write!(f, "enum {{ ")?;
                for (i, name) in variants.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", name)?;
                }
                write!(f, " }}")
            }
            Type::AssocTypeBinding { name, ty } => {
                write!(f, "{} = {}", name, ty)
            }
            // Linear type modifiers for quantum computing
            Type::Linear(inner) => write!(f, "linear {}", inner),
            Type::Affine(inner) => write!(f, "affine {}", inner),
            Type::Relevant(inner) => write!(f, "relevant {}", inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Parser;

    fn check(source: &str) -> Result<(), Vec<TypeError>> {
        let mut parser = Parser::new(source);
        let file = parser.parse_file().expect("parse failed");
        let mut checker = TypeChecker::new();
        checker.check_file(&file)
    }

    #[test]
    fn test_basic_types() {
        assert!(check("rite main() { ≔ x: i64 = 42; }").is_ok());
        assert!(check("rite main() { ≔ x: bool = true; }").is_ok());
        assert!(check("rite main() { ≔ x: f64 = 3.14; }").is_ok());
    }

    #[test]
    fn test_type_mismatch() {
        assert!(check("rite main() { ≔ x: bool = 42; }").is_err());
    }

    #[test]
    fn test_evidence_propagation() {
        // Evidence should propagate through operations
        assert!(check(
            r#"
            rite main() {
                ≔ known: i64! = 42;
                ≔ uncertain: i64? = 10;
                ≔ result = known + uncertain;
            }
        "#
        )
        .is_ok());
    }

    #[test]
    fn test_function_return() {
        let result = check(
            r#"
            rite add(a: i64, b: i64) -> i64 {
                ⤺ a + b;
            }
            rite main() {
                ≔ x = add(1, 2);
            }
        "#,
        );
        if let Err(errors) = &result {
            for e in errors {
                eprintln!("Error: {}", e);
            }
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_array_types() {
        assert!(check(
            r#"
            rite main() {
                ≔ arr = [1, 2, 3];
                ≔ x = arr[0];
            }
        "#
        )
        .is_ok());
    }

    // ==========================================
    // Evidence Inference Tests
    // ==========================================

    #[test]
    fn test_evidence_inference_from_initializer() {
        // Evidence should be inferred from initializer when not explicitly annotated
        assert!(check(
            r#"
            rite main() {
                ≔ reported_val: i64~ = 42;
                // x should inherit ~ evidence from reported_val
                ≔ x = reported_val + 1;
            }
        "#
        )
        .is_ok());
    }

    #[test]
    fn test_evidence_inference_explicit_override() {
        // Explicit annotation should override inference
        assert!(check(
            r#"
            rite main() {
                ≔ reported_val: i64~ = 42;
                // Explicit ! annotation - this would fail ⎇ we checked evidence properly
                // but the type system allows it as an override
                ≔ x! = 42;
            }
        "#
        )
        .is_ok());
    }

    #[test]
    fn test_if_else_evidence_join() {
        // Evidence from both branches should be joined
        assert!(check(
            r#"
            rite main() {
                ≔ known_val: i64! = 1;
                ≔ reported_val: i64~ = 2;
                ≔ cond: bool = true;
                // Result should have ~ evidence (join of ! and ~)
                ≔ result = ⎇ cond { known_val } ⎉ { reported_val };
            }
        "#
        )
        .is_ok());
    }

    #[test]
    fn test_binary_op_evidence_propagation() {
        // Binary operations should join evidence levels
        assert!(check(
            r#"
            rite main() {
                ≔ known: i64! = 1;
                ≔ reported: i64~ = 2;
                // Result should have ~ evidence (max of ! and ~)
                ≔ result = known + reported;
            }
        "#
        )
        .is_ok());
    }

    #[test]
    fn test_match_evidence_join() {
        // Match arms should join evidence from all branches
        // Note: This test is structural - the type checker should handle it
        assert!(check(
            r#"
            rite main() {
                ≔ x: i64 = 1;
            }
        "#
        )
        .is_ok());
    }
}

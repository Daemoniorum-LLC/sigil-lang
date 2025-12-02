//! Type checker for Sigil.
//!
//! Implements bidirectional type inference with evidentiality tracking.
//! The type system enforces that evidence levels propagate correctly
//! through computations.

use crate::ast::*;
use crate::span::Span;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

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
    Array { element: Box<Type>, size: Option<usize> },
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
    Ref { mutable: bool, inner: Box<Type> },
    Ptr { mutable: bool, inner: Box<Type> },

    /// Evidential wrapper - the core of Sigil's type system
    Evidential {
        inner: Box<Type>,
        evidence: EvidenceLevel,
    },

    /// Cyclic type (modular arithmetic)
    Cycle { modulus: usize },

    /// SIMD vector type
    Simd { element: Box<Type>, lanes: u8 },

    /// Atomic type
    Atomic(Box<Type>),

    /// Type variable for inference
    Var(TypeVar),

    /// Error type (for error recovery)
    Error,

    /// Never type (diverging)
    Never,
}

/// Integer sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntSize {
    I8, I16, I32, I64, I128,
    U8, U16, U32, U64, U128,
    ISize, USize,
}

/// Float sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSize {
    F32, F64,
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
    Known,      // !
    /// Uncertain - inferred, possible, not verified
    Uncertain,  // ?
    /// Reported - from external source, hearsay
    Reported,   // ~
    /// Paradox - contradictory information
    Paradox,    // ‽
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
            Evidentiality::Uncertain => EvidenceLevel::Uncertain,
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
}

/// Type variable for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar(pub u32);

/// Type error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
}

impl TypeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            notes: Vec::new(),
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
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(span) = self.span {
            write!(f, " at {}", span)?;
        }
        for note in &self.notes {
            write!(f, "\n  note: {}", note)?;
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
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            parent: None,
        }
    }

    pub fn with_parent(parent: Rc<RefCell<TypeEnv>>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent: Some(parent),
        }
    }

    /// Define a new binding
    pub fn define(&mut self, name: String, ty: Type, evidence: EvidenceLevel) {
        self.bindings.insert(name, (ty, evidence));
    }

    /// Look up a binding
    pub fn lookup(&self, name: &str) -> Option<(Type, EvidenceLevel)> {
        if let Some(binding) = self.bindings.get(name) {
            Some(binding.clone())
        } else if let Some(ref parent) = self.parent {
            parent.borrow().lookup(name)
        } else {
            None
        }
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
    /// Type variable counter
    next_var: u32,
    /// Inferred type variable substitutions
    substitutions: HashMap<TypeVar, Type>,
    /// Collected errors
    errors: Vec<TypeError>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut checker = Self {
            env: Rc::new(RefCell::new(TypeEnv::new())),
            types: HashMap::new(),
            functions: HashMap::new(),
            next_var: 0,
            substitutions: HashMap::new(),
            errors: Vec::new(),
        };

        // Register built-in types and functions
        checker.register_builtins();
        checker
    }

    fn register_builtins(&mut self) {
        // Built-in functions
        self.functions.insert("print".to_string(), Type::Function {
            params: vec![Type::Str], // variadic in practice
            return_type: Box::new(Type::Unit),
            is_async: false,
        });

        self.functions.insert("len".to_string(), Type::Function {
            params: vec![Type::Var(TypeVar(0))], // generic
            return_type: Box::new(Type::Int(IntSize::USize)),
            is_async: false,
        });

        self.functions.insert("type_of".to_string(), Type::Function {
            params: vec![Type::Var(TypeVar(0))],
            return_type: Box::new(Type::Str),
            is_async: false,
        });
    }

    /// Fresh type variable
    fn fresh_var(&mut self) -> Type {
        let var = TypeVar(self.next_var);
        self.next_var += 1;
        Type::Var(var)
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

    /// Record an error
    fn error(&mut self, err: TypeError) {
        self.errors.push(err);
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
                let generics = s.generics.as_ref()
                    .map(|g| g.params.iter().filter_map(|p| {
                        if let GenericParam::Type { name, .. } = p {
                            Some(name.name.clone())
                        } else {
                            None
                        }
                    }).collect())
                    .unwrap_or_default();

                let fields = match &s.fields {
                    StructFields::Named(fs) => fs.iter()
                        .map(|f| (f.name.name.clone(), self.convert_type(&f.ty)))
                        .collect(),
                    StructFields::Tuple(ts) => ts.iter()
                        .enumerate()
                        .map(|(i, t)| (i.to_string(), self.convert_type(t)))
                        .collect(),
                    StructFields::Unit => vec![],
                };

                self.types.insert(s.name.name.clone(), TypeDef::Struct { generics, fields });
            }
            Item::Enum(e) => {
                let generics = e.generics.as_ref()
                    .map(|g| g.params.iter().filter_map(|p| {
                        if let GenericParam::Type { name, .. } = p {
                            Some(name.name.clone())
                        } else {
                            None
                        }
                    }).collect())
                    .unwrap_or_default();

                let variants = e.variants.iter()
                    .map(|v| {
                        let fields = match &v.fields {
                            StructFields::Tuple(ts) => Some(ts.iter()
                                .map(|t| self.convert_type(t))
                                .collect()),
                            StructFields::Named(fs) => Some(fs.iter()
                                .map(|f| self.convert_type(&f.ty))
                                .collect()),
                            StructFields::Unit => None,
                        };
                        (v.name.name.clone(), fields)
                    })
                    .collect();

                self.types.insert(e.name.name.clone(), TypeDef::Enum { generics, variants });
            }
            Item::TypeAlias(t) => {
                let generics = t.generics.as_ref()
                    .map(|g| g.params.iter().filter_map(|p| {
                        if let GenericParam::Type { name, .. } = p {
                            Some(name.name.clone())
                        } else {
                            None
                        }
                    }).collect())
                    .unwrap_or_default();

                let target = self.convert_type(&t.ty);
                self.types.insert(t.name.name.clone(), TypeDef::Alias { generics, target });
            }
            _ => {}
        }
    }

    /// Collect function signatures (second pass)
    fn collect_fn_sig(&mut self, item: &Item) {
        if let Item::Function(f) = item {
            let params: Vec<Type> = f.params.iter()
                .map(|p| self.convert_type(&p.ty))
                .collect();

            let return_type = f.return_type.as_ref()
                .map(|t| self.convert_type(t))
                .unwrap_or(Type::Unit);

            let fn_type = Type::Function {
                params,
                return_type: Box::new(return_type),
                is_async: f.is_async,
            };

            self.functions.insert(f.name.name.clone(), fn_type);
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
                    self.error(TypeError::new(format!(
                        "type mismatch in const '{}': expected {:?}, found {:?}",
                        c.name.name, declared, inferred
                    )).with_span(c.name.span));
                }
            }
            Item::Static(s) => {
                let declared = self.convert_type(&s.ty);
                let inferred = self.infer_expr(&s.value);
                if !self.unify(&declared, &inferred) {
                    self.error(TypeError::new(format!(
                        "type mismatch in static '{}': expected {:?}, found {:?}",
                        s.name.name, declared, inferred
                    )).with_span(s.name.span));
                }
            }
            _ => {}
        }
    }

    /// Check a function body
    fn check_function(&mut self, func: &Function) {
        self.push_scope();

        // Bind parameters
        for param in &func.params {
            let ty = self.convert_type(&param.ty);
            let evidence = param.pattern
                .evidentiality()
                .map(EvidenceLevel::from_ast)
                .unwrap_or(EvidenceLevel::Known);

            if let Some(name) = param.pattern.binding_name() {
                self.env.borrow_mut().define(name, ty, evidence);
            }
        }

        // Check body
        if let Some(ref body) = func.body {
            let body_type = self.check_block(body);

            // Check return type
            let expected_return = func.return_type.as_ref()
                .map(|t| self.convert_type(t))
                .unwrap_or(Type::Unit);

            if !self.unify(&expected_return, &body_type) {
                self.error(TypeError::new(format!(
                    "return type mismatch in '{}': expected {:?}, found {:?}",
                    func.name.name, expected_return, body_type
                )).with_span(func.name.span));
            }
        }

        self.pop_scope();
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

                let final_ty = match (declared_ty, init_ty) {
                    (Some(d), Some(i)) => {
                        if !self.unify(&d, &i) {
                            self.error(TypeError::new(format!(
                                "type mismatch in let binding: expected {:?}, found {:?}",
                                d, i
                            )));
                        }
                        d
                    }
                    (Some(d), None) => d,
                    (None, Some(i)) => i,
                    (None, None) => self.fresh_var(),
                };

                let evidence = pattern
                    .evidentiality()
                    .map(EvidenceLevel::from_ast)
                    .unwrap_or(EvidenceLevel::Known);

                if let Some(name) = pattern.binding_name() {
                    self.env.borrow_mut().define(name, final_ty, evidence);
                }
                Type::Unit
            }
            Stmt::Expr(e) | Stmt::Semi(e) => {
                self.infer_expr(e)
            }
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
                    if let Some((ty, _)) = self.env.borrow().lookup(name) {
                        return ty;
                    }
                    if let Some(ty) = self.functions.get(name) {
                        return ty.clone();
                    }
                }
                self.error(TypeError::new(format!("undefined: {:?}", path)));
                Type::Error
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
                let arg_types: Vec<Type> = args.iter()
                    .map(|a| self.infer_expr(a))
                    .collect();

                if let Type::Function { params, return_type, .. } = fn_type {
                    // Check argument count
                    if params.len() != arg_types.len() {
                        self.error(TypeError::new(format!(
                            "expected {} arguments, found {}",
                            params.len(), arg_types.len()
                        )));
                    }

                    // Check argument types
                    for (param, arg) in params.iter().zip(arg_types.iter()) {
                        if !self.unify(param, arg) {
                            self.error(TypeError::new(format!(
                                "argument type mismatch: expected {:?}, found {:?}",
                                param, arg
                            )));
                        }
                    }

                    *return_type
                } else {
                    self.error(TypeError::new("cannot call non-function"));
                    Type::Error
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

            Expr::If { condition, then_branch, else_branch } => {
                let cond_ty = self.infer_expr(condition);
                if !self.unify(&Type::Bool, &cond_ty) {
                    self.error(TypeError::new("if condition must be bool"));
                }

                let then_ty = self.check_block(then_branch);

                if let Some(else_expr) = else_branch {
                    // Else branch can be another expr (e.g., if-else chain) or a block
                    let else_ty = match else_expr.as_ref() {
                        Expr::Block(block) => self.check_block(block),
                        other => self.infer_expr(other),
                    };
                    if !self.unify(&then_ty, &else_ty) {
                        self.error(TypeError::new("if branches must have same type"));
                    }
                    then_ty
                } else {
                    Type::Unit
                }
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
                        self.error(TypeError::new("cannot index non-array"));
                        Type::Error
                    }
                }
            }

            Expr::Return(val) => {
                if let Some(e) = val {
                    self.infer_expr(e);
                }
                Type::Never
            }

            // Mark expression with evidence
            Expr::Evidential { expr, evidentiality } => {
                let inner = self.infer_expr(expr);
                Type::Evidential {
                    inner: Box::new(inner),
                    evidence: EvidenceLevel::from_ast(*evidentiality),
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
            Literal::Float { .. } => Type::Float(FloatSize::F64),
            Literal::Bool(_) => Type::Bool,
            Literal::Char(_) => Type::Char,
            Literal::String(_) => Type::Str,
            Literal::Null => Type::Unit,  // null has unit type
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

        let result_ty = match op {
            // Arithmetic: numeric -> numeric
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div |
            BinOp::Rem | BinOp::Pow => {
                if !self.unify(&left_inner, &right_inner) {
                    self.error(TypeError::new("arithmetic operands must have same type"));
                }
                left_inner
            }

            // Comparison: any -> bool
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le |
            BinOp::Gt | BinOp::Ge => {
                if !self.unify(&left_inner, &right_inner) {
                    self.error(TypeError::new("comparison operands must have same type"));
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
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor |
            BinOp::Shl | BinOp::Shr => {
                left_inner
            }

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
                if !self.unify(&Type::Bool, &inner_ty) {
                    self.error(TypeError::new("! operand must be bool"));
                }
                Type::Bool
            }
            UnaryOp::Ref => Type::Ref { mutable: false, inner: Box::new(inner_ty) },
            UnaryOp::RefMut => Type::Ref { mutable: true, inner: Box::new(inner_ty) },
            UnaryOp::Deref => {
                if let Type::Ref { inner, .. } | Type::Ptr { inner, .. } = inner_ty {
                    *inner
                } else {
                    self.error(TypeError::new("cannot dereference non-pointer"));
                    Type::Error
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
        let (inner, evidence) = self.strip_evidence(input);

        let result = match op {
            // Transform: [T] -> [U] where body: T -> U
            PipeOp::Transform(_body) => {
                if let Type::Array { element, size } = inner {
                    Type::Array { element, size }
                } else if let Type::Slice(element) = inner {
                    Type::Slice(element)
                } else {
                    self.error(TypeError::new("transform requires array or slice"));
                    Type::Error
                }
            }

            // Filter: [T] -> [T]
            PipeOp::Filter(_pred) => inner,

            // Sort: [T] -> [T]
            PipeOp::Sort(_) => inner,

            // Reduce: [T] -> T
            PipeOp::Reduce(_) => {
                if let Type::Array { element, .. } | Type::Slice(element) = inner {
                    *element
                } else {
                    self.error(TypeError::new("reduce requires array or slice"));
                    Type::Error
                }
            }

            // Method call
            PipeOp::Method { name, args: _ } => {
                // Look up method
                if let Some(fn_ty) = self.functions.get(&name.name) {
                    if let Type::Function { return_type, .. } = fn_ty {
                        *return_type.clone()
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
            PipeOp::First | PipeOp::Last | PipeOp::Middle |
            PipeOp::Choice | PipeOp::Nth(_) | PipeOp::Next => {
                if let Type::Array { element, .. } | Type::Slice(element) = inner {
                    *element
                } else if let Type::Tuple(elements) = inner {
                    // For tuple, return Any since elements might be different types
                    if let Some(first) = elements.first() {
                        first.clone()
                    } else {
                        Type::Unit
                    }
                } else {
                    self.error(TypeError::new("access morpheme requires array, slice, or tuple"));
                    Type::Error
                }
            }
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

    /// Attempt to unify two types
    fn unify(&mut self, a: &Type, b: &Type) -> bool {
        match (a, b) {
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

            (Type::Int(a), Type::Int(b)) => a == b,
            (Type::Float(a), Type::Float(b)) => a == b,

            // Arrays
            (Type::Array { element: a, size: sa }, Type::Array { element: b, size: sb }) => {
                (sa == sb || sa.is_none() || sb.is_none()) && self.unify(a, b)
            }

            // Slices
            (Type::Slice(a), Type::Slice(b)) => self.unify(a, b),

            // Tuples
            (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
                a.iter().zip(b.iter()).all(|(x, y)| self.unify(x, y))
            }

            // References
            (Type::Ref { mutable: ma, inner: a }, Type::Ref { mutable: mb, inner: b }) => {
                ma == mb && self.unify(a, b)
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
                na == nb && ga.len() == gb.len() &&
                ga.iter().zip(gb.iter()).all(|(x, y)| self.unify(x, y))
            }

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

            // Type variables
            (Type::Var(v), t) => {
                if let Some(resolved) = self.substitutions.get(v) {
                    let resolved = resolved.clone();
                    self.unify(&resolved, t)
                } else {
                    self.substitutions.insert(*v, t.clone());
                    true
                }
            }
            (t, Type::Var(v)) => {
                if let Some(resolved) = self.substitutions.get(v) {
                    let resolved = resolved.clone();
                    self.unify(t, &resolved)
                } else {
                    self.substitutions.insert(*v, t.clone());
                    true
                }
            }

            // Cycles
            (Type::Cycle { modulus: a }, Type::Cycle { modulus: b }) => a == b,

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
                        _ => {}
                    }
                }

                let name = path.segments.iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::");

                let generics = path.segments.last()
                    .and_then(|s| s.generics.as_ref())
                    .map(|gs| gs.iter().map(|t| self.convert_type(t)).collect())
                    .unwrap_or_default();

                Type::Named { name, generics }
            }

            TypeExpr::Reference { mutable, inner } => {
                Type::Ref {
                    mutable: *mutable,
                    inner: Box::new(self.convert_type(inner)),
                }
            }

            TypeExpr::Pointer { mutable, inner } => {
                Type::Ptr {
                    mutable: *mutable,
                    inner: Box::new(self.convert_type(inner)),
                }
            }

            TypeExpr::Array { element, size: _ } => {
                Type::Array {
                    element: Box::new(self.convert_type(element)),
                    size: None, // Could evaluate const expr
                }
            }

            TypeExpr::Slice(inner) => {
                Type::Slice(Box::new(self.convert_type(inner)))
            }

            TypeExpr::Tuple(elements) => {
                Type::Tuple(elements.iter().map(|t| self.convert_type(t)).collect())
            }

            TypeExpr::Function { params, return_type } => {
                Type::Function {
                    params: params.iter().map(|t| self.convert_type(t)).collect(),
                    return_type: Box::new(
                        return_type.as_ref()
                            .map(|t| self.convert_type(t))
                            .unwrap_or(Type::Unit)
                    ),
                    is_async: false,
                }
            }

            TypeExpr::Evidential { inner, evidentiality } => {
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
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Type::Named { name, generics } => {
                write!(f, "{}", name)?;
                if !generics.is_empty() {
                    write!(f, "<")?;
                    for (i, g) in generics.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", g)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            Type::Function { params, return_type, is_async } => {
                if *is_async { write!(f, "async ")?; }
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", return_type)
            }
            Type::Ref { mutable, inner } => {
                write!(f, "&{}{}", if *mutable { "mut " } else { "" }, inner)
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
        assert!(check("fn main() { let x: i64 = 42; }").is_ok());
        assert!(check("fn main() { let x: bool = true; }").is_ok());
        assert!(check("fn main() { let x: f64 = 3.14; }").is_ok());
    }

    #[test]
    fn test_type_mismatch() {
        assert!(check("fn main() { let x: bool = 42; }").is_err());
    }

    #[test]
    fn test_evidence_propagation() {
        // Evidence should propagate through operations
        assert!(check(r#"
            fn main() {
                let known: i64! = 42;
                let uncertain: i64? = 10;
                let result = known + uncertain;
            }
        "#).is_ok());
    }

    #[test]
    fn test_function_return() {
        let result = check(r#"
            fn add(a: i64, b: i64) -> i64 {
                return a + b;
            }
            fn main() {
                let x = add(1, 2);
            }
        "#);
        if let Err(errors) = &result {
            for e in errors {
                eprintln!("Error: {}", e);
            }
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_array_types() {
        assert!(check(r#"
            fn main() {
                let arr = [1, 2, 3];
                let x = arr[0];
            }
        "#).is_ok());
    }
}

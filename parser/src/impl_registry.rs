//! Impl Registry for Generic Monomorphization
//!
//! This module provides infrastructure for tracking generic impl blocks
//! and resolving methods with concrete type bindings.
//!
//! Part of Phase 2 of the generic monomorphization implementation.

use std::collections::HashMap;

use crate::ast::{self, Block, GenericParam, ImplBlock, WhereClause};
use crate::typeck::Type;

/// Type bindings from pattern matching: generic name -> concrete type
pub type TypeBindings = HashMap<String, Type>;

/// Registry of all impl blocks across crates
#[derive(Debug, Clone, Default)]
pub struct ImplRegistry {
    /// Generic impl definitions: impl<S, D, Dev> Tensor<S, D, Dev> { ... }
    generic_impls: Vec<GenericImpl>,

    /// Concrete (non-generic) impl definitions
    concrete_impls: Vec<ConcreteImpl>,
}

/// A generic impl block with its methods
#[derive(Debug, Clone)]
pub struct GenericImpl {
    /// Source crate name
    pub crate_name: String,
    /// Impl-level generic parameters: <S: Shape, D: DType, Dev: Device>
    pub generics: Vec<GenericParamInfo>,
    /// The Self type pattern: Tensor<S, D, Dev>
    pub self_type: TypePattern,
    /// Where clause bounds (if any)
    pub where_clauses: Vec<WhereBound>,
    /// Methods defined in this impl
    pub methods: HashMap<String, MethodDef>,
}

/// A concrete (non-generic) impl block
#[derive(Debug, Clone)]
pub struct ConcreteImpl {
    /// Source crate name
    pub crate_name: String,
    /// The concrete Self type
    pub self_type: Type,
    /// Methods defined in this impl
    pub methods: HashMap<String, MethodDef>,
}

/// Simplified generic parameter info
#[derive(Debug, Clone)]
pub struct GenericParamInfo {
    pub name: String,
    pub bounds: Vec<String>,  // Trait bounds as strings for now
    pub is_const: bool,
    /// Default type for this generic parameter (e.g., `D: DType = f32`)
    pub default: Option<TypePattern>,
}

/// A method definition with its generic signature
#[derive(Debug, Clone)]
pub struct MethodDef {
    pub name: String,
    /// Method-level generic parameters (in addition to impl-level)
    pub generics: Vec<GenericParamInfo>,
    /// Parameter names and type patterns
    pub params: Vec<(String, TypePattern)>,
    /// Return type pattern
    pub return_type: TypePattern,
    /// AST of method body (if available)
    pub body: Option<Block>,
    /// Whether this is a static method (no self parameter)
    pub is_static: bool,
}

/// Type pattern for matching (may contain generic params)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypePattern {
    /// A concrete type (no generic parameters)
    Concrete(Type),
    /// A generic parameter name like "S" or "T"
    Generic(String),
    /// A parameterized type like Tensor<S, D, Dev>
    Parameterized {
        name: String,
        params: Vec<TypePattern>,
    },
    /// Reference pattern: &T or &mut T
    Reference {
        mutable: bool,
        inner: Box<TypePattern>,
    },
    /// Array pattern: [T; N]
    Array {
        element: Box<TypePattern>,
        size: Option<usize>,
    },
    /// Slice pattern: [T]
    Slice(Box<TypePattern>),
    /// Tuple pattern: (A, B, C)
    Tuple(Vec<TypePattern>),
    /// Unit type
    Unit,
}

/// A where clause bound
#[derive(Debug, Clone)]
pub struct WhereBound {
    pub type_param: String,
    pub bounds: Vec<String>,
}

impl ImplRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a generic impl block from AST
    pub fn register_generic_impl(
        &mut self,
        crate_name: &str,
        impl_block: &ImplBlock,
    ) {
        let generics = Self::extract_generic_params(impl_block.generics.as_ref());
        let self_type = Self::type_expr_to_pattern(&impl_block.self_ty, &generics);
        let where_clauses = Self::extract_where_clauses(impl_block.where_clause.as_ref());

        eprintln!("[IMPL-REG] register_generic_impl: crate='{}', self_type={:?}, generics={:?}, items={}",
            crate_name, self_type, generics, impl_block.items.len());

        let mut methods = HashMap::new();
        for (i, item) in impl_block.items.iter().enumerate() {
            match item {
                ast::ImplItem::Function(func) => {
                    let method_def = Self::function_to_method_def(func, &generics);
                    eprintln!("[IMPL-REG] item[{}] = Function '{}': is_static={}", i, func.name.name, method_def.is_static);
                    methods.insert(func.name.name.clone(), method_def);
                }
                ast::ImplItem::Type(t) => {
                    eprintln!("[IMPL-REG] item[{}] = Type '{}'", i, t.name.name);
                }
                ast::ImplItem::Const(c) => {
                    eprintln!("[IMPL-REG] item[{}] = Const '{}'", i, c.name.name);
                }
            }
        }

        // Only register if there are generic parameters
        if !generics.is_empty() {
            eprintln!("[IMPL-REG] -> registering as GENERIC impl");
            self.generic_impls.push(GenericImpl {
                crate_name: crate_name.to_string(),
                generics,
                self_type,
                where_clauses,
                methods,
            });
        } else {
            // Concrete impl
            eprintln!("[IMPL-REG] -> registering as CONCRETE impl (no generics)");
            let concrete_type = Self::pattern_to_concrete_type(&self_type);
            if let Some(ty) = concrete_type {
                self.concrete_impls.push(ConcreteImpl {
                    crate_name: crate_name.to_string(),
                    self_type: ty,
                    methods,
                });
            }
        }
    }

    /// Find a method for a concrete receiver type
    pub fn resolve_method(
        &self,
        receiver_type: &Type,
        method_name: &str,
    ) -> Option<(MethodDef, TypeBindings)> {
        // First, check concrete impls for exact match
        for impl_def in &self.concrete_impls {
            if &impl_def.self_type == receiver_type {
                if let Some(method) = impl_def.methods.get(method_name) {
                    return Some((method.clone(), TypeBindings::new()));
                }
            }
        }

        // Then, check generic impls
        for impl_def in &self.generic_impls {
            if let Some(bindings) = self.match_type(&impl_def.self_type, receiver_type) {
                if let Some(method) = impl_def.methods.get(method_name) {
                    // TODO: Verify where clauses are satisfied
                    return Some((method.clone(), bindings));
                }
            }
        }

        None
    }

    /// Find a static method by type name and method name
    /// This is for static method calls like TensorLayout::contiguous()
    /// which don't have a receiver type
    ///
    /// G67: Returns (MethodDef, impl_generics) where impl_generics are the
    /// impl-level generic parameters (e.g., `S` in `impl<S> Wrapper<S>`)
    pub fn resolve_static_method(
        &self,
        type_name: &str,
        method_name: &str,
    ) -> Option<(MethodDef, Vec<GenericParamInfo>)> {
        eprintln!("[IMPL-DEBUG] resolve_static_method('{}', '{}') - {} concrete, {} generic impls",
            type_name, method_name, self.concrete_impls.len(), self.generic_impls.len());
        // First check concrete impls for exact match (no impl-level generics)
        for impl_def in &self.concrete_impls {
            if let Type::Named { name, .. } = &impl_def.self_type {
                if name == type_name {
                    if let Some(method) = impl_def.methods.get(method_name) {
                        if method.is_static {
                            return Some((method.clone(), vec![]));
                        }
                    }
                }
            }
        }

        // Then check generic impls (return impl-level generics)
        for impl_def in &self.generic_impls {
            eprintln!("[IMPL-DEBUG] Checking generic impl: {:?}", impl_def.self_type);
            if let TypePattern::Parameterized { name, .. } = &impl_def.self_type {
                eprintln!("[IMPL-DEBUG] Generic impl name='{}', looking for='{}'", name, type_name);
                if name == type_name {
                    eprintln!("[IMPL-DEBUG] Found type match, methods: {:?}", impl_def.methods.keys().collect::<Vec<_>>());
                    if let Some(method) = impl_def.methods.get(method_name) {
                        eprintln!("[IMPL-DEBUG] Found method '{}', is_static={}", method_name, method.is_static);
                        if method.is_static {
                            return Some((method.clone(), impl_def.generics.clone()));
                        }
                    }
                }
            }
            // Also check non-parameterized patterns (concrete type in generic impl)
            if let TypePattern::Concrete(Type::Named { name, .. }) = &impl_def.self_type {
                if name == type_name {
                    if let Some(method) = impl_def.methods.get(method_name) {
                        if method.is_static {
                            return Some((method.clone(), impl_def.generics.clone()));
                        }
                    }
                }
            }
        }

        None
    }

    /// Match a type pattern against a concrete type, producing bindings
    pub fn match_type(
        &self,
        pattern: &TypePattern,
        concrete: &Type,
    ) -> Option<TypeBindings> {
        let mut bindings = TypeBindings::new();
        self.match_recursive(pattern, concrete, &mut bindings)?;
        Some(bindings)
    }

    fn match_recursive(
        &self,
        pattern: &TypePattern,
        concrete: &Type,
        bindings: &mut TypeBindings,
    ) -> Option<()> {
        match (pattern, concrete) {
            // Generic parameter - bind it
            (TypePattern::Generic(name), ty) => {
                if let Some(existing) = bindings.get(name) {
                    // Already bound - must match
                    if existing == ty { Some(()) } else { None }
                } else {
                    bindings.insert(name.clone(), ty.clone());
                    Some(())
                }
            }

            // Parameterized type - match name and recurse on params
            (TypePattern::Parameterized { name: pn, params: pp },
             Type::Named { name: cn, generics: cg }) => {
                if pn != cn || pp.len() != cg.len() {
                    return None;
                }
                for (p, c) in pp.iter().zip(cg.iter()) {
                    self.match_recursive(p, c, bindings)?;
                }
                Some(())
            }

            // Concrete type must match exactly
            (TypePattern::Concrete(p), c) => {
                if p == c { Some(()) } else { None }
            }

            // Reference patterns
            (TypePattern::Reference { mutable: pm, inner: pi },
             Type::Ref { mutable: cm, inner: ci, .. }) => {
                if pm != cm {
                    return None;
                }
                self.match_recursive(pi, ci, bindings)
            }

            // Array patterns
            (TypePattern::Array { element: pe, size: ps },
             Type::Array { element: ce, size: cs }) => {
                if ps != cs {
                    return None;
                }
                self.match_recursive(pe, ce, bindings)
            }

            // Slice patterns
            (TypePattern::Slice(pi), Type::Slice(ci)) => {
                self.match_recursive(pi, ci, bindings)
            }

            // Tuple patterns
            (TypePattern::Tuple(pp), Type::Tuple(cp)) => {
                if pp.len() != cp.len() {
                    return None;
                }
                for (p, c) in pp.iter().zip(cp.iter()) {
                    self.match_recursive(p, c, bindings)?;
                }
                Some(())
            }

            // Unit
            (TypePattern::Unit, Type::Unit) => Some(()),

            // Named type without generics can match a parameterized pattern with 0 params
            (TypePattern::Parameterized { name: pn, params: pp },
             Type::Named { name: cn, generics: cg }) if pp.is_empty() && cg.is_empty() => {
                if pn == cn { Some(()) } else { None }
            }

            _ => None,
        }
    }

    /// Extract generic parameter info from AST Generics
    fn extract_generic_params(generics: Option<&ast::Generics>) -> Vec<GenericParamInfo> {
        let Some(generics) = generics else {
            return Vec::new();
        };

        generics.params.iter().map(|param| {
            match param {
                GenericParam::Type { name, bounds, default, .. } => {
                    // Convert the default TypeExpr to TypePattern if present
                    // Note: We pass empty generic_names since defaults are typically concrete types
                    let default_pattern = default.as_ref().map(|d| {
                        Self::type_expr_to_pattern(d, &[])
                    });
                    GenericParamInfo {
                        name: name.name.clone(),
                        bounds: bounds.iter().map(|b| Self::type_expr_to_string(b)).collect(),
                        is_const: false,
                        default: default_pattern,
                    }
                },
                GenericParam::Const { name, .. } => GenericParamInfo {
                    name: name.name.clone(),
                    bounds: Vec::new(),
                    is_const: true,
                    default: None,
                },
                GenericParam::Lifetime(lt) => GenericParamInfo {
                    name: lt.clone(),
                    bounds: Vec::new(),
                    is_const: false,
                    default: None,
                },
            }
        }).collect()
    }

    /// Convert AST TypeExpr to TypePattern
    fn type_expr_to_pattern(
        ty: &ast::TypeExpr,
        generic_names: &[GenericParamInfo],
    ) -> TypePattern {
        match ty {
            ast::TypeExpr::Path(path) => {
                if path.segments.len() == 1 {
                    let seg = &path.segments[0];
                    let name = &seg.ident.name;

                    // Check if this is a generic parameter
                    if generic_names.iter().any(|g| &g.name == name) {
                        return TypePattern::Generic(name.clone());
                    }

                    // Check for primitive types - return as Concrete
                    if let Some(prim) = Self::primitive_type(name) {
                        return TypePattern::Concrete(prim);
                    }

                    // Named type with optional generics
                    if let Some(ref type_args) = seg.generics {
                        if !type_args.is_empty() {
                            return TypePattern::Parameterized {
                                name: name.clone(),
                                params: type_args.iter()
                                    .map(|a| Self::type_expr_to_pattern(a, generic_names))
                                    .collect(),
                            };
                        }
                    }

                    // Simple named type (no generics)
                    TypePattern::Parameterized {
                        name: name.clone(),
                        params: Vec::new(),
                    }
                } else {
                    // Multi-segment path - use last segment
                    let seg = path.segments.last().unwrap();
                    let params = seg.generics.as_ref()
                        .map(|gs| gs.iter().map(|g| Self::type_expr_to_pattern(g, generic_names)).collect())
                        .unwrap_or_default();

                    TypePattern::Parameterized {
                        name: seg.ident.name.clone(),
                        params,
                    }
                }
            }
            ast::TypeExpr::Reference { inner, mutable, .. } => {
                TypePattern::Reference {
                    mutable: *mutable,
                    inner: Box::new(Self::type_expr_to_pattern(inner, generic_names)),
                }
            }
            ast::TypeExpr::Array { element, .. } => {
                TypePattern::Array {
                    element: Box::new(Self::type_expr_to_pattern(element, generic_names)),
                    size: None,  // TODO: Extract const size
                }
            }
            ast::TypeExpr::Slice(inner) => {
                TypePattern::Slice(Box::new(Self::type_expr_to_pattern(inner, generic_names)))
            }
            ast::TypeExpr::Tuple(elements) => {
                if elements.is_empty() {
                    TypePattern::Unit
                } else {
                    TypePattern::Tuple(
                        elements.iter()
                            .map(|e| Self::type_expr_to_pattern(e, generic_names))
                            .collect()
                    )
                }
            }
            _ => {
                // Fallback for unsupported patterns
                TypePattern::Parameterized {
                    name: "unknown".to_string(),
                    params: Vec::new(),
                }
            }
        }
    }

    /// Check if a name corresponds to a primitive type
    fn primitive_type(name: &str) -> Option<Type> {
        use crate::typeck::{IntSize, FloatSize};
        match name {
            "bool" => Some(Type::Bool),
            "char" => Some(Type::Char),
            "str" | "String" => Some(Type::Str),
            "i8" => Some(Type::Int(IntSize::I8)),
            "i16" => Some(Type::Int(IntSize::I16)),
            "i32" => Some(Type::Int(IntSize::I32)),
            "i64" => Some(Type::Int(IntSize::I64)),
            "i128" => Some(Type::Int(IntSize::I128)),
            "isize" => Some(Type::Int(IntSize::ISize)),
            "u8" => Some(Type::Int(IntSize::U8)),
            "u16" => Some(Type::Int(IntSize::U16)),
            "u32" => Some(Type::Int(IntSize::U32)),
            "u64" => Some(Type::Int(IntSize::U64)),
            "u128" => Some(Type::Int(IntSize::U128)),
            "usize" => Some(Type::Int(IntSize::USize)),
            "f32" => Some(Type::Float(FloatSize::F32)),
            "f64" => Some(Type::Float(FloatSize::F64)),
            "()" => Some(Type::Unit),
            _ => None,
        }
    }

    /// Convert TypeExpr to string (for bounds)
    fn type_expr_to_string(ty: &ast::TypeExpr) -> String {
        match ty {
            ast::TypeExpr::Path(path) => {
                path.segments.iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::")
            }
            _ => "unknown".to_string(),
        }
    }

    /// Extract where clauses from AST
    fn extract_where_clauses(where_clause: Option<&WhereClause>) -> Vec<WhereBound> {
        let Some(wc) = where_clause else {
            return Vec::new();
        };

        wc.predicates.iter().map(|pred| {
            let type_param = Self::type_expr_to_string(&pred.ty);
            let bound_strs: Vec<String> = pred.bounds.iter()
                .map(|b| Self::type_expr_to_string(b))
                .collect();
            WhereBound {
                type_param,
                bounds: bound_strs,
            }
        }).collect()
    }

    /// Convert a function to MethodDef
    fn function_to_method_def(
        func: &ast::Function,
        impl_generics: &[GenericParamInfo],
    ) -> MethodDef {
        // Combine impl-level and method-level generics for pattern resolution
        let method_generics = Self::extract_generic_params(func.generics.as_ref());
        let all_generics: Vec<GenericParamInfo> = impl_generics.iter()
            .chain(method_generics.iter())
            .cloned()
            .collect();

        // Check if first param is self or this (Sigil uses "this" for self reference)
        // Also handles reference patterns like &this or &mut this
        let extract_receiver_name = |pat: &ast::Pattern| -> Option<String> {
            match pat {
                ast::Pattern::Ident { name, .. } => Some(name.name.clone()),
                ast::Pattern::Ref { pattern, .. } => {
                    // &this or &mut this
                    match pattern.as_ref() {
                        ast::Pattern::Ident { name, .. } => Some(name.name.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        let first_param_name = func.params.first().and_then(|p| extract_receiver_name(&p.pattern));
        let is_receiver = |name: &Option<String>| -> bool {
            matches!(name.as_deref(), Some("self" | "this"))
        };
        let is_static = !is_receiver(&first_param_name);

        // Extract parameters (skip self/this for instance methods)
        let params: Vec<(String, TypePattern)> = func.params.iter()
            .filter(|p| {
                let name = extract_receiver_name(&p.pattern);
                !is_receiver(&name)
            })
            .map(|p| {
                let name = match &p.pattern {
                    ast::Pattern::Ident { name, .. } => name.name.clone(),
                    _ => "_".to_string(),
                };
                let pattern = Self::type_expr_to_pattern(&p.ty, &all_generics);
                (name, pattern)
            })
            .collect();

        // Return type
        let return_type = func.return_type.as_ref()
            .map(|rt| Self::type_expr_to_pattern(rt, &all_generics))
            .unwrap_or(TypePattern::Unit);

        MethodDef {
            name: func.name.name.clone(),
            generics: method_generics,
            params,
            return_type,
            body: func.body.clone(),
            is_static,
        }
    }

    /// Try to convert a TypePattern to a concrete Type (only works for non-generic patterns)
    fn pattern_to_concrete_type(pattern: &TypePattern) -> Option<Type> {
        match pattern {
            TypePattern::Concrete(ty) => Some(ty.clone()),
            TypePattern::Generic(_) => None,  // Can't convert generic to concrete
            TypePattern::Parameterized { name, params } => {
                if params.iter().any(|p| matches!(p, TypePattern::Generic(_))) {
                    return None;  // Has generic params
                }
                let concrete_params: Option<Vec<Type>> = params.iter()
                    .map(Self::pattern_to_concrete_type)
                    .collect();
                concrete_params.map(|generics| Type::Named {
                    name: name.clone(),
                    generics,
                })
            }
            TypePattern::Reference { mutable, inner } => {
                Self::pattern_to_concrete_type(inner).map(|inner_ty| Type::Ref {
                    lifetime: None,
                    mutable: *mutable,
                    inner: Box::new(inner_ty),
                })
            }
            TypePattern::Array { element, size } => {
                Self::pattern_to_concrete_type(element).map(|elem_ty| Type::Array {
                    element: Box::new(elem_ty),
                    size: *size,
                })
            }
            TypePattern::Slice(inner) => {
                Self::pattern_to_concrete_type(inner).map(|inner_ty| Type::Slice(Box::new(inner_ty)))
            }
            TypePattern::Tuple(elements) => {
                let concrete: Option<Vec<Type>> = elements.iter()
                    .map(Self::pattern_to_concrete_type)
                    .collect();
                concrete.map(Type::Tuple)
            }
            TypePattern::Unit => Some(Type::Unit),
        }
    }

    /// Get number of registered generic impls (for debugging)
    pub fn generic_impl_count(&self) -> usize {
        self.generic_impls.len()
    }

    /// Get number of registered concrete impls (for debugging)
    pub fn concrete_impl_count(&self) -> usize {
        self.concrete_impls.len()
    }

    /// List all registered impl types (for debugging)
    pub fn list_impls(&self) -> Vec<String> {
        let mut result = Vec::new();
        for impl_def in &self.generic_impls {
            result.push(format!("generic: {:?}", impl_def.self_type));
        }
        for impl_def in &self.concrete_impls {
            result.push(format!("concrete: {:?}", impl_def.self_type));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_simple_generic() {
        let registry = ImplRegistry::new();

        // Pattern: T
        let pattern = TypePattern::Generic("T".to_string());

        // Concrete: i64
        let concrete = Type::Int(crate::typeck::IntSize::I64);

        let bindings = registry.match_type(&pattern, &concrete);
        assert!(bindings.is_some());
        let bindings = bindings.unwrap();
        assert_eq!(bindings.get("T"), Some(&concrete));
    }

    #[test]
    fn test_match_parameterized() {
        let registry = ImplRegistry::new();

        // Pattern: Vec<T>
        let pattern = TypePattern::Parameterized {
            name: "Vec".to_string(),
            params: vec![TypePattern::Generic("T".to_string())],
        };

        // Concrete: Vec<i32>
        let concrete = Type::Named {
            name: "Vec".to_string(),
            generics: vec![Type::Int(crate::typeck::IntSize::I32)],
        };

        let bindings = registry.match_type(&pattern, &concrete);
        assert!(bindings.is_some());
        let bindings = bindings.unwrap();
        assert_eq!(
            bindings.get("T"),
            Some(&Type::Int(crate::typeck::IntSize::I32))
        );
    }

    #[test]
    fn test_match_multiple_generics() {
        let registry = ImplRegistry::new();

        // Pattern: Tensor<S, D, Dev>
        let pattern = TypePattern::Parameterized {
            name: "Tensor".to_string(),
            params: vec![
                TypePattern::Generic("S".to_string()),
                TypePattern::Generic("D".to_string()),
                TypePattern::Generic("Dev".to_string()),
            ],
        };

        // Concrete: Tensor<Shape2, f32, Cuda>
        let concrete = Type::Named {
            name: "Tensor".to_string(),
            generics: vec![
                Type::Named { name: "Shape2".to_string(), generics: vec![] },
                Type::Float(crate::typeck::FloatSize::F32),
                Type::Named { name: "Cuda".to_string(), generics: vec![] },
            ],
        };

        let bindings = registry.match_type(&pattern, &concrete);
        assert!(bindings.is_some());
        let bindings = bindings.unwrap();

        assert_eq!(
            bindings.get("S"),
            Some(&Type::Named { name: "Shape2".to_string(), generics: vec![] })
        );
        assert_eq!(
            bindings.get("D"),
            Some(&Type::Float(crate::typeck::FloatSize::F32))
        );
        assert_eq!(
            bindings.get("Dev"),
            Some(&Type::Named { name: "Cuda".to_string(), generics: vec![] })
        );
    }

    #[test]
    fn test_no_match_different_name() {
        let registry = ImplRegistry::new();

        // Pattern: Vec<T>
        let pattern = TypePattern::Parameterized {
            name: "Vec".to_string(),
            params: vec![TypePattern::Generic("T".to_string())],
        };

        // Concrete: HashMap<i32> (different name)
        let concrete = Type::Named {
            name: "HashMap".to_string(),
            generics: vec![Type::Int(crate::typeck::IntSize::I32)],
        };

        let bindings = registry.match_type(&pattern, &concrete);
        assert!(bindings.is_none());
    }

    #[test]
    fn test_no_match_different_arity() {
        let registry = ImplRegistry::new();

        // Pattern: Pair<A, B>
        let pattern = TypePattern::Parameterized {
            name: "Pair".to_string(),
            params: vec![
                TypePattern::Generic("A".to_string()),
                TypePattern::Generic("B".to_string()),
            ],
        };

        // Concrete: Pair<i32> (only 1 param)
        let concrete = Type::Named {
            name: "Pair".to_string(),
            generics: vec![Type::Int(crate::typeck::IntSize::I32)],
        };

        let bindings = registry.match_type(&pattern, &concrete);
        assert!(bindings.is_none());
    }
}

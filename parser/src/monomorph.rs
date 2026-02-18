//! Monomorphization Engine for Generic Methods
//!
//! This module handles the compilation of generic methods with concrete type parameters.
//! When a generic method like `Tensor<S, D, Dev>::matmul` is called with concrete types,
//! the monomorphizer generates a specialized version of that method.
//!
//! Part of Phase 3 of the generic monomorphization implementation.

use std::collections::{HashMap, VecDeque};

use crate::impl_registry::{MethodDef, TypeBindings, TypePattern};
use crate::typeck::{Type, IntSize, FloatSize};

/// Key for uniquely identifying a monomorphized method instance
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MonomorphKey {
    /// Source crate/impl identifier
    pub impl_id: String,
    /// Method name
    pub method_name: String,
    /// Concrete type bindings (sorted for consistent hashing)
    pub bindings: Vec<(String, Type)>,
}

impl MonomorphKey {
    /// Create a new monomorph key from method info and bindings
    pub fn new(impl_id: &str, method_name: &str, bindings: &TypeBindings) -> Self {
        let mut sorted_bindings: Vec<(String, Type)> = bindings
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        sorted_bindings.sort_by(|a, b| a.0.cmp(&b.0));

        Self {
            impl_id: impl_id.to_string(),
            method_name: method_name.to_string(),
            bindings: sorted_bindings,
        }
    }

    /// Generate unique symbol name for this instantiation
    pub fn mangle(&self) -> String {
        let mut name = format!("{}_{}", self.impl_id, self.method_name);

        for (param, ty) in &self.bindings {
            name.push_str("__");
            name.push_str(param);
            name.push('_');
            name.push_str(&Self::mangle_type(ty));
        }

        // Sanitize for LLVM symbol names
        name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
    }

    /// Mangle a type into a string suitable for symbol names
    fn mangle_type(ty: &Type) -> String {
        match ty {
            Type::Unit => "unit".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Char => "char".to_string(),
            Type::Str => "str".to_string(),
            Type::Int(size) => match size {
                IntSize::I8 => "i8".to_string(),
                IntSize::I16 => "i16".to_string(),
                IntSize::I32 => "i32".to_string(),
                IntSize::I64 => "i64".to_string(),
                IntSize::I128 => "i128".to_string(),
                IntSize::ISize => "isize".to_string(),
                IntSize::U8 => "u8".to_string(),
                IntSize::U16 => "u16".to_string(),
                IntSize::U32 => "u32".to_string(),
                IntSize::U64 => "u64".to_string(),
                IntSize::U128 => "u128".to_string(),
                IntSize::USize => "usize".to_string(),
            },
            Type::Float(size) => match size {
                FloatSize::F32 => "f32".to_string(),
                FloatSize::F64 => "f64".to_string(),
            },
            Type::Named { name, generics } => {
                if generics.is_empty() {
                    name.clone()
                } else {
                    let params: Vec<String> = generics.iter()
                        .map(Self::mangle_type)
                        .collect();
                    format!("{}__{}", name, params.join("_"))
                }
            }
            Type::Array { element, size } => {
                format!("arr_{}_{}",
                    Self::mangle_type(element),
                    size.unwrap_or(0))
            }
            Type::Slice(inner) => {
                format!("slice_{}", Self::mangle_type(inner))
            }
            Type::Ref { inner, mutable, .. } => {
                let prefix = if *mutable { "refmut" } else { "ref" };
                format!("{}_{}", prefix, Self::mangle_type(inner))
            }
            Type::Ptr { inner, mutable } => {
                let prefix = if *mutable { "ptrmut" } else { "ptr" };
                format!("{}_{}", prefix, Self::mangle_type(inner))
            }
            Type::Tuple(elements) => {
                if elements.is_empty() {
                    "unit".to_string()
                } else {
                    let parts: Vec<String> = elements.iter()
                        .map(Self::mangle_type)
                        .collect();
                    format!("tup_{}", parts.join("_"))
                }
            }
            Type::Function { params, return_type, .. } => {
                let param_parts: Vec<String> = params.iter()
                    .map(Self::mangle_type)
                    .collect();
                format!("fn_{}__ret_{}",
                    param_parts.join("_"),
                    Self::mangle_type(return_type))
            }
            Type::ConstGeneric(value) => format!("c{}", value),
            _ => "unknown".to_string(),
        }
    }
}

/// A pending monomorphization request
#[derive(Debug, Clone)]
pub struct MonomorphRequest {
    pub key: MonomorphKey,
    pub method: MethodDef,
    pub bindings: TypeBindings,
}

/// Cache of compiled generic instantiations
#[derive(Debug, Default)]
pub struct MonomorphCache {
    /// Compiled instances: key -> mangled function name
    /// We store names rather than FunctionValue to avoid lifetime issues
    instances: HashMap<MonomorphKey, String>,

    /// Pending compilation requests
    pending: VecDeque<MonomorphRequest>,

    /// Statistics for debugging
    cache_hits: usize,
    cache_misses: usize,
}

impl MonomorphCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if an instance is already compiled
    pub fn get(&self, key: &MonomorphKey) -> Option<&String> {
        self.instances.get(key)
    }

    /// Register a compiled instance
    pub fn insert(&mut self, key: MonomorphKey, fn_name: String) {
        self.instances.insert(key, fn_name);
    }

    /// Check if we have a cached instance, returning the mangled name
    pub fn get_or_request(
        &mut self,
        method: &MethodDef,
        bindings: &TypeBindings,
        impl_id: &str,
    ) -> Result<GetOrRequest, String> {
        let key = MonomorphKey::new(impl_id, &method.name, bindings);

        if let Some(fn_name) = self.instances.get(&key) {
            self.cache_hits += 1;
            return Ok(GetOrRequest::Cached(fn_name.clone()));
        }

        self.cache_misses += 1;
        let mangled_name = key.mangle();

        // Add to pending queue
        self.pending.push_back(MonomorphRequest {
            key: key.clone(),
            method: method.clone(),
            bindings: bindings.clone(),
        });

        // Pre-register to avoid duplicate requests
        self.instances.insert(key, mangled_name.clone());

        Ok(GetOrRequest::NeedsCompilation(mangled_name))
    }

    /// Get next pending compilation request
    pub fn pop_pending(&mut self) -> Option<MonomorphRequest> {
        self.pending.pop_front()
    }

    /// Check if there are pending compilations
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get number of cached instances
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.cache_hits, self.cache_misses)
    }
}

/// Result of get_or_request
#[derive(Debug, Clone)]
pub enum GetOrRequest {
    /// Already compiled, here's the function name
    Cached(String),
    /// Needs compilation, here's the mangled name to use
    NeedsCompilation(String),
}

/// Substitute type patterns with concrete types from bindings
pub fn substitute_pattern(pattern: &TypePattern, bindings: &TypeBindings) -> Type {
    match pattern {
        TypePattern::Concrete(ty) => ty.clone(),
        TypePattern::Generic(name) => {
            bindings.get(name).cloned().unwrap_or_else(|| {
                // If not found, return as a named type (might be a concrete type we don't know about)
                Type::Named {
                    name: name.clone(),
                    generics: vec![],
                }
            })
        }
        TypePattern::Parameterized { name, params } => {
            Type::Named {
                name: name.clone(),
                generics: params.iter()
                    .map(|p| substitute_pattern(p, bindings))
                    .collect(),
            }
        }
        TypePattern::Reference { mutable, inner } => {
            Type::Ref {
                lifetime: None,
                mutable: *mutable,
                inner: Box::new(substitute_pattern(inner, bindings)),
            }
        }
        TypePattern::Array { element, size } => {
            Type::Array {
                element: Box::new(substitute_pattern(element, bindings)),
                size: *size,
            }
        }
        TypePattern::Slice(inner) => {
            Type::Slice(Box::new(substitute_pattern(inner, bindings)))
        }
        TypePattern::Tuple(elements) => {
            Type::Tuple(elements.iter()
                .map(|e| substitute_pattern(e, bindings))
                .collect())
        }
        TypePattern::Unit => Type::Unit,
    }
}

/// Substitute types in a method signature, producing concrete parameter and return types
pub fn substitute_method_signature(
    method: &MethodDef,
    bindings: &TypeBindings,
) -> (Vec<(String, Type)>, Type) {
    let concrete_params: Vec<(String, Type)> = method.params.iter()
        .map(|(name, ty_pat)| {
            (name.clone(), substitute_pattern(ty_pat, bindings))
        })
        .collect();

    let concrete_ret = substitute_pattern(&method.return_type, bindings);

    (concrete_params, concrete_ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mangle_simple() {
        let mut bindings = TypeBindings::new();
        bindings.insert("T".to_string(), Type::Int(IntSize::I64));

        let key = MonomorphKey::new("Vec", "push", &bindings);
        let mangled = key.mangle();

        assert!(mangled.contains("Vec"));
        assert!(mangled.contains("push"));
        assert!(mangled.contains("i64"));
    }

    #[test]
    fn test_mangle_multiple_generics() {
        let mut bindings = TypeBindings::new();
        bindings.insert("S".to_string(), Type::Named {
            name: "Shape2".to_string(),
            generics: vec![]
        });
        bindings.insert("D".to_string(), Type::Float(FloatSize::F32));
        bindings.insert("Dev".to_string(), Type::Named {
            name: "Cuda".to_string(),
            generics: vec![]
        });

        let key = MonomorphKey::new("Tensor", "matmul", &bindings);
        let mangled = key.mangle();

        assert!(mangled.contains("Tensor"));
        assert!(mangled.contains("matmul"));
        assert!(mangled.contains("Shape2"));
        assert!(mangled.contains("f32"));
        assert!(mangled.contains("Cuda"));
    }

    #[test]
    fn test_cache_hit_miss() {
        let mut cache = MonomorphCache::new();

        let method = MethodDef {
            name: "test".to_string(),
            generics: vec![],
            params: vec![],
            return_type: TypePattern::Unit,
            body: None,
            is_static: true,
        };

        let mut bindings = TypeBindings::new();
        bindings.insert("T".to_string(), Type::Int(IntSize::I32));

        // First request should be a miss
        let result = cache.get_or_request(&method, &bindings, "TestImpl").unwrap();
        assert!(matches!(result, GetOrRequest::NeedsCompilation(_)));

        // Second request should be a hit
        let result = cache.get_or_request(&method, &bindings, "TestImpl").unwrap();
        assert!(matches!(result, GetOrRequest::Cached(_)));

        let (hits, misses) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_substitute_pattern() {
        let mut bindings = TypeBindings::new();
        bindings.insert("T".to_string(), Type::Float(FloatSize::F64));

        // Test generic substitution
        let pattern = TypePattern::Generic("T".to_string());
        let result = substitute_pattern(&pattern, &bindings);
        assert_eq!(result, Type::Float(FloatSize::F64));

        // Test parameterized substitution
        let pattern = TypePattern::Parameterized {
            name: "Vec".to_string(),
            params: vec![TypePattern::Generic("T".to_string())],
        };
        let result = substitute_pattern(&pattern, &bindings);
        assert_eq!(result, Type::Named {
            name: "Vec".to_string(),
            generics: vec![Type::Float(FloatSize::F64)],
        });
    }

    #[test]
    fn test_consistent_key_ordering() {
        // Keys should be equal regardless of insertion order
        let mut bindings1 = TypeBindings::new();
        bindings1.insert("A".to_string(), Type::Int(IntSize::I32));
        bindings1.insert("B".to_string(), Type::Int(IntSize::I64));

        let mut bindings2 = TypeBindings::new();
        bindings2.insert("B".to_string(), Type::Int(IntSize::I64));
        bindings2.insert("A".to_string(), Type::Int(IntSize::I32));

        let key1 = MonomorphKey::new("Test", "method", &bindings1);
        let key2 = MonomorphKey::new("Test", "method", &bindings2);

        assert_eq!(key1, key2);
        assert_eq!(key1.mangle(), key2.mangle());
    }
}

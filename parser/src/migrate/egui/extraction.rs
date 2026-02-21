//! egui AST extraction using `syn`.
//!
//! Walks a Rust source file and collects:
//! - `pub struct` definitions → state field candidates
//! - `impl` blocks → method inventory
//! - Method bodies → detected egui UI patterns and ambiguities
//!
//! The visitor deliberately stays shallow: it identifies patterns for the
//! pattern library to classify rather than trying to fully interpret Rust semantics.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use syn::{
    visit::{self, Visit},
    Fields, ImplItem, ImplItemFn, ItemImpl, ItemStruct, Type,
    Expr, ExprMethodCall,
};

use super::patterns::{AmbiguityKind, classify_method_call};

// =============================================================================
// Top-level extraction result
// =============================================================================

/// Complete extraction from a single Rust/egui source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiExtraction {
    pub file: FileInfo,
    pub structs: Vec<StructExtraction>,
    pub impls: Vec<ImplExtraction>,
}

/// File metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: PathBuf,
    pub relative_path: String,
    /// Raw source text (used as `source.code` in the spec).
    pub source: String,
}

// =============================================================================
// Struct extraction
// =============================================================================

/// A `pub struct` (or private struct) found in the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructExtraction {
    pub name: String,
    pub is_pub: bool,
    pub fields: Vec<FieldExtraction>,
}

/// A single struct field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldExtraction {
    pub name: String,
    /// Stringified type, e.g. `"Option<String>"`, `"Vec<NotificationEntry>"`.
    pub field_type: String,
    pub is_pub: bool,
}

// =============================================================================
// Impl extraction
// =============================================================================

/// An `impl Foo { ... }` block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplExtraction {
    pub type_name: String,
    pub methods: Vec<MethodExtraction>,
}

/// A single method inside an impl block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodExtraction {
    pub name: String,
    pub is_pub: bool,
    /// True for `fn show(…)`, `fn render(…)`, `fn ui(…)`, `fn view(…)`.
    pub is_view: bool,
    /// True if the first param is `&mut self`.
    pub takes_mut_self: bool,
    /// Stringified parameter types (excluding self).
    pub params: Vec<String>,
    /// Detected egui call patterns in the method body.
    pub body_patterns: Vec<DetectedPattern>,
    /// Ambiguities found in the method body.
    pub ambiguities: Vec<DetectedAmbiguity>,
}

// =============================================================================
// Detected patterns
// =============================================================================

/// An egui call pattern detected in a method body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Canonical name from the pattern library, e.g. `"label"`, `"button"`, `"horizontal"`.
    pub kind: String,
    /// Argument text snippets (up to 2).
    pub args: Vec<String>,
    /// Approximate line number (0 if unavailable).
    pub line: u32,
}

/// An egui call that could not be automatically mapped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAmbiguity {
    pub kind: AmbiguityKind,
    /// Approximate line number (0 if unavailable).
    pub line: u32,
    /// Short source snippet for context.
    pub snippet: String,
}

// =============================================================================
// Entry point
// =============================================================================

/// Parse Rust source text directly and return its extraction.
///
/// Used in tests to avoid file I/O and race conditions.
pub fn extract_source(source: &str, virtual_path: &Path) -> Result<EguiExtraction, String> {
    let syntax = syn::parse_file(source)
        .map_err(|e| format!("Parse error in {:?}: {}", virtual_path, e))?;

    let file_info = FileInfo {
        path: virtual_path.to_path_buf(),
        relative_path: virtual_path.to_string_lossy().into_owned(),
        source: source.to_string(),
    };

    let mut visitor = EguiVisitor::new();
    visitor.visit_file(&syntax);

    Ok(EguiExtraction {
        file: file_info,
        structs: visitor.structs,
        impls: visitor.impls,
    })
}

/// Parse a Rust file and return its extraction.
pub fn extract_file(path: &Path, source_root: &Path) -> Result<EguiExtraction, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {:?}: {}", path, e))?;

    let syntax = syn::parse_file(&source)
        .map_err(|e| format!("Parse error in {:?}: {}", path, e))?;

    let relative_path = path
        .strip_prefix(source_root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();

    let file_info = FileInfo {
        path: path.to_path_buf(),
        relative_path,
        source,
    };

    let mut visitor = EguiVisitor::new();
    visitor.visit_file(&syntax);

    Ok(EguiExtraction {
        file: file_info,
        structs: visitor.structs,
        impls: visitor.impls,
    })
}

// =============================================================================
// syn Visitor
// =============================================================================

struct EguiVisitor {
    structs: Vec<StructExtraction>,
    impls: Vec<ImplExtraction>,
}

impl EguiVisitor {
    fn new() -> Self {
        Self {
            structs: Vec::new(),
            impls: Vec::new(),
        }
    }
}

impl<'ast> Visit<'ast> for EguiVisitor {
    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        let is_pub = matches!(node.vis, syn::Visibility::Public(_));
        let name = node.ident.to_string();

        let fields = match &node.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .map(|f| FieldExtraction {
                    name: f
                        .ident
                        .as_ref()
                        .map(|i| i.to_string())
                        .unwrap_or_default(),
                    field_type: type_to_string(&f.ty),
                    is_pub: matches!(f.vis, syn::Visibility::Public(_)),
                })
                .collect(),
            _ => vec![],
        };

        self.structs.push(StructExtraction { name, is_pub, fields });

        visit::visit_item_struct(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        // Only interested in plain `impl Foo { }` (not trait impls).
        if node.trait_.is_some() {
            visit::visit_item_impl(self, node);
            return;
        }

        let type_name = match node.self_ty.as_ref() {
            Type::Path(tp) => tp
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default(),
            _ => return,
        };

        let mut methods = Vec::new();

        for item in &node.items {
            if let ImplItem::Fn(method) = item {
                methods.push(extract_method(method));
            }
        }

        if !methods.is_empty() {
            self.impls.push(ImplExtraction { type_name, methods });
        }

        visit::visit_item_impl(self, node);
    }
}

// =============================================================================
// Method extraction helpers
// =============================================================================

fn extract_method(method: &ImplItemFn) -> MethodExtraction {
    let name = method.sig.ident.to_string();
    let is_pub = matches!(method.vis, syn::Visibility::Public(_));

    let view_names = ["show", "render", "ui", "view", "draw"];
    let is_view = view_names.contains(&name.as_str());

    // Check for &mut self
    let takes_mut_self = method.sig.inputs.iter().any(|arg| {
        matches!(arg, syn::FnArg::Receiver(r) if r.mutability.is_some())
    });

    // Collect non-self parameter types
    let params: Vec<String> = method
        .sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            syn::FnArg::Typed(pt) => Some(type_to_string(&pt.ty)),
            _ => None,
        })
        .collect();

    // Walk body for egui patterns
    let mut body_visitor = BodyVisitor::default();
    body_visitor.visit_block(&method.block);

    MethodExtraction {
        name,
        is_pub,
        is_view,
        takes_mut_self,
        params,
        body_patterns: body_visitor.patterns,
        ambiguities: body_visitor.ambiguities,
    }
}

// =============================================================================
// Body visitor — walks method bodies for egui calls
// =============================================================================

#[derive(Default)]
struct BodyVisitor {
    patterns: Vec<DetectedPattern>,
    ambiguities: Vec<DetectedAmbiguity>,
}

impl<'ast> Visit<'ast> for BodyVisitor {
    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method_name = node.method.to_string();

        // Check if the receiver chain touches `ui` or `ctx`
        if is_egui_receiver(&node.receiver) {
            let args: Vec<String> = node.args.iter().take(2).map(expr_to_string).collect();

            match classify_method_call(&method_name, &args) {
                Ok(kind) => {
                    self.patterns.push(DetectedPattern {
                        kind,
                        args,
                        line: 0, // proc_macro2 spans lack line info outside proc macros
                    });
                }
                Err(ambiguity) => {
                    self.ambiguities.push(DetectedAmbiguity {
                        kind: ambiguity,
                        line: 0,
                        snippet: format!(".{}({})", method_name, args.join(", ")),
                    });
                }
            }
        }

        // Always recurse into sub-expressions (nested closures, etc.)
        visit::visit_expr_method_call(self, node);
    }
}

/// Returns true if the expression chain involves a known egui receiver (`ui`, `ctx`, `painter`).
fn is_egui_receiver(expr: &Expr) -> bool {
    match expr {
        Expr::Path(p) => {
            let s = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            matches!(s.as_str(), "ui" | "ctx" | "painter")
        }
        Expr::MethodCall(mc) => is_egui_receiver(&mc.receiver),
        Expr::Reference(r) => is_egui_receiver(&r.expr),
        Expr::Field(f) => {
            // `self.some_widget`
            if let Expr::Path(p) = f.base.as_ref() {
                let s = p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
                return s == "self";
            }
            false
        }
        _ => false,
    }
}

// =============================================================================
// Type / expr stringification utilities
// =============================================================================

/// Convert a `syn::Type` to a human-readable string.
pub fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::Path(tp) => {
            let segments: Vec<String> = tp
                .path
                .segments
                .iter()
                .map(|s| {
                    let args = match &s.arguments {
                        syn::PathArguments::AngleBracketed(ab) => {
                            let inner: Vec<String> = ab
                                .args
                                .iter()
                                .map(|a| match a {
                                    syn::GenericArgument::Type(t) => type_to_string(t),
                                    _ => "_".to_string(),
                                })
                                .collect();
                            if inner.is_empty() {
                                String::new()
                            } else {
                                format!("<{}>", inner.join(", "))
                            }
                        }
                        _ => String::new(),
                    };
                    format!("{}{}", s.ident, args)
                })
                .collect();
            segments.join("::")
        }
        Type::Reference(r) => {
            let mutability = if r.mutability.is_some() { "&mut " } else { "&" };
            format!("{}{}", mutability, type_to_string(&r.elem))
        }
        Type::Tuple(t) if t.elems.is_empty() => "()".to_string(),
        Type::Tuple(t) => {
            let inner: Vec<String> = t.elems.iter().map(type_to_string).collect();
            format!("({})", inner.join(", "))
        }
        _ => "_".to_string(),
    }
}

/// Convert a `syn::Expr` to a short string snippet (for pattern args / snippets).
fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Lit(l) => match &l.lit {
            syn::Lit::Str(s) => format!("\"{}\"", s.value()),
            syn::Lit::Int(i) => i.to_string(),
            syn::Lit::Float(f) => f.to_string(),
            syn::Lit::Bool(b) => b.value.to_string(),
            _ => "…".to_string(),
        },
        Expr::Path(p) => p
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        Expr::MethodCall(mc) => format!("{}.{}(…)", expr_to_string(&mc.receiver), mc.method),
        Expr::Reference(r) => format!("&{}", expr_to_string(&r.expr)),
        Expr::Unary(u) => match u.op {
            syn::UnOp::Deref(_) => format!("*{}", expr_to_string(&u.expr)),
            syn::UnOp::Not(_) => format!("!{}", expr_to_string(&u.expr)),
            _ => "…".to_string(),
        },
        _ => "…".to_string(),
    }
}

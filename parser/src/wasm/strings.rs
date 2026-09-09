//! Which values are strings.
//!
//! Every value in this backend is an i64, so a string handle and the number 7
//! are the same thing to the code generator. It never asked which it had:
//! `format!` sent every argument through `string.from_int`, so
//! `format!("<{}>", tag)` produced `<16384>` — the decimal of the pointer;
//! `to_string()` did the same; and `+` compiled to `i64.add`, so `"a" + "b"`
//! was the sum of two addresses. All three are correct in the interpreter, so
//! the same program meant different things in the two backends, and the
//! WebAssembly one produced garbage without failing.
//!
//! This is the static approximation the backend can actually make from the AST
//! it is given: declared parameter, field and return types, plus the
//! expressions whose result type is known by construction. It is deliberately
//! conservative — an expression it cannot prove is a string is treated as a
//! number, which is the behaviour that was there before.

use crate::ast::{BinOp, Expr, Literal, TypeExpr, UnaryOp};

use super::WasmCompiler;

/// Method names that return a string whatever their receiver is.
const STRING_METHODS: &[&str] = &[
    "to_string",
    "to_uppercase",
    "to_lowercase",
    "trim",
    "trim_start",
    "trim_end",
    "replace",
    "join",
    "concat",
    "repeat",
    "to_fixed",
    "char_at",
    "escape_html",
    "escape_attr",
];

/// The name a declared type resolves to, looking through references and
/// evidentiality markers: `&Elem!` is `Elem`.
pub fn named_type(ty: &TypeExpr) -> Option<String> {
    match ty {
        TypeExpr::Path(path) => path.segments.last().map(|s| s.ident.name.clone()),
        TypeExpr::Reference { inner, .. }
        | TypeExpr::Evidential { inner, .. }
        | TypeExpr::Atomic(inner) => named_type(inner),
        _ => None,
    }
}

/// Whether a declared type is a string.
pub fn type_is_string(ty: &TypeExpr) -> bool {
    match ty {
        TypeExpr::Path(path) => matches!(
            path.segments.last().map(|s| s.ident.name.as_str()),
            Some("String") | Some("str")
        ),
        TypeExpr::Reference { inner, .. }
        | TypeExpr::Evidential { inner, .. }
        | TypeExpr::Atomic(inner) => type_is_string(inner),
        _ => false,
    }
}

impl WasmCompiler {
    /// Whether this expression is statically known to produce a string.
    pub(crate) fn is_string_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::String(_)) => true,

            // A local the current function bound from a string, or a parameter
            // declared `&str` / `String`.
            Expr::Path(path) if path.segments.len() == 1 => {
                self.string_locals.contains(&path.segments[0].ident.name)
            }

            // `self.field`, where the struct declares the field as a string.
            Expr::Field { expr: base, field } => self
                .receiver_struct_name(base)
                .and_then(|s| self.string_fields.get(&s))
                .is_some_and(|fields| fields.contains(&field.name)),

            Expr::MethodCall { method, .. } => STRING_METHODS.contains(&method.name.as_str()),

            Expr::Call { func, .. } => match &**func {
                Expr::Path(path) => path
                    .segments
                    .last()
                    .is_some_and(|s| self.string_returning.contains(&s.ident.name)),
                _ => false,
            },

            // `format!(…)` and `concat!(…)`.
            Expr::Macro { path, .. } => matches!(
                path.segments.last().map(|s| s.ident.name.as_str()),
                Some("format") | Some("concat")
            ),

            // Concatenation is a string if either side is.
            Expr::Binary { left, op, right } if matches!(op, BinOp::Add | BinOp::Concat) => {
                self.is_string_expr(left) || self.is_string_expr(right)
            }

            // Transparent wrappers.
            Expr::AddrOf { expr: inner, .. }
            | Expr::Attributed { expr: inner, .. }
            | Expr::Turbofish { expr: inner, .. }
            | Expr::Evidential { expr: inner, .. } => self.is_string_expr(inner),
            Expr::Unary { op: UnaryOp::Deref, expr: inner } => self.is_string_expr(inner),

            // Both arms of an `if` decide together; one is enough, since a
            // program whose arms disagree is already ill-typed.
            Expr::If { then_branch, else_branch, .. } => {
                self.block_is_string(then_branch)
                    || else_branch.as_deref().is_some_and(|e| self.is_string_expr(e))
            }
            Expr::Block(block) => self.block_is_string(block),

            _ => false,
        }
    }

    fn block_is_string(&self, block: &crate::ast::Block) -> bool {
        block
            .expr
            .as_deref()
            .is_some_and(|e| self.is_string_expr(e))
    }

    /// The struct name a receiver expression refers to, for field lookups.
    fn receiver_struct_name(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path) if path.segments.len() == 1 => {
                let name = &path.segments[0].ident.name;
                if name == "self" {
                    self.current_actor
                        .clone()
                        .or_else(|| self.module_path.last().cloned())
                } else {
                    self.infer_receiver_type(expr)
                }
            }
            _ => self.infer_receiver_type(expr),
        }
    }

    /// Record that a name holds a string for the rest of this function.
    pub(crate) fn note_string_local(&mut self, name: &str, value: &Expr) {
        if self.is_string_expr(value) {
            self.string_locals.insert(name.to_string());
        } else {
            // A rebinding to a non-string has to clear the old answer.
            self.string_locals.remove(name);
        }
    }
}

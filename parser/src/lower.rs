//! AST to IR Lowering Pass
//!
//! Converts the parsed AST into the AI-facing IR format.
//! This pass:
//! - Generates unique IDs for all definitions
//! - Infers types where needed (or uses placeholders)
//! - Propagates evidentiality through expressions
//! - Normalizes pipeline operations

use crate::ast::{self, SourceFile};
use crate::ir::*;
use std::collections::HashMap;

/// Counter for generating unique IDs
struct IdGenerator {
    counters: HashMap<String, usize>,
}

impl IdGenerator {
    fn new() -> Self {
        Self {
            counters: HashMap::new(),
        }
    }

    fn next(&mut self, prefix: &str) -> String {
        let count = self.counters.entry(prefix.to_string()).or_insert(0);
        *count += 1;
        format!("{}_{:03}", prefix, count)
    }
}

/// Lowering context
pub struct LoweringContext {
    ids: IdGenerator,
    /// Current scope for variable resolution
    scope: Vec<HashMap<String, String>>,
    /// Inferred types for expressions (placeholder for now)
    type_cache: HashMap<String, IrType>,
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            ids: IdGenerator::new(),
            scope: vec![HashMap::new()],
            type_cache: HashMap::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scope.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scope.pop();
    }

    fn bind_var(&mut self, name: &str) -> String {
        let id = self.ids.next("var");
        if let Some(scope) = self.scope.last_mut() {
            scope.insert(name.to_string(), id.clone());
        }
        id
    }

    fn lookup_var(&self, name: &str) -> Option<String> {
        for scope in self.scope.iter().rev() {
            if let Some(id) = scope.get(name) {
                return Some(id.clone());
            }
        }
        None
    }
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower a complete source file to IR
pub fn lower_source_file(source: &str, ast: &SourceFile) -> IrModule {
    let mut ctx = LoweringContext::new();
    let mut module = IrModule::new(source.to_string());

    for item in &ast.items {
        lower_item(&mut ctx, &mut module, &item.node);
    }

    module
}

fn lower_item(ctx: &mut LoweringContext, module: &mut IrModule, item: &ast::Item) {
    match item {
        ast::Item::Function(f) => {
            if let Some(ir_fn) = lower_function(ctx, f) {
                module.functions.push(ir_fn);
            }
        }
        ast::Item::Struct(s) => {
            module.types.push(lower_struct_def(ctx, s));
        }
        ast::Item::Enum(e) => {
            module.types.push(lower_enum_def(ctx, e));
        }
        ast::Item::Trait(t) => {
            module.traits.push(lower_trait_def(ctx, t));
        }
        ast::Item::Impl(i) => {
            module.impls.push(lower_impl_block(ctx, i));
        }
        ast::Item::TypeAlias(t) => {
            module.types.push(lower_type_alias(ctx, t));
        }
        ast::Item::Const(c) => {
            module.constants.push(lower_const_def(ctx, c));
        }
        ast::Item::Module(m) => {
            if let Some(items) = &m.items {
                for item in items {
                    lower_item(ctx, module, &item.node);
                }
            }
        }
        ast::Item::Static(_)
        | ast::Item::Actor(_)
        | ast::Item::Use(_)
        | ast::Item::ExternBlock(_) => {
            // TODO: Handle these items
        }
    }
}

fn lower_function(ctx: &mut LoweringContext, f: &ast::Function) -> Option<IrFunction> {
    let id = ctx.ids.next("fn");
    ctx.push_scope();

    let params: Vec<IrParam> = f.params.iter().map(|p| lower_param(ctx, p)).collect();

    let return_type = f
        .return_type
        .as_ref()
        .map(|t| lower_type_expr(t))
        .unwrap_or(IrType::Unit);

    let body = f.body.as_ref().map(|b| lower_block(ctx, b));

    let mut attributes = Vec::new();
    if f.is_async {
        attributes.push("async".to_string());
    }
    if f.attrs.inline.is_some() {
        attributes.push("inline".to_string());
    }
    if f.attrs.naked {
        attributes.push("naked".to_string());
    }
    if f.attrs.no_mangle {
        attributes.push("no_mangle".to_string());
    }

    ctx.pop_scope();

    Some(IrFunction {
        name: f.name.name.clone(),
        id,
        visibility: lower_visibility(f.visibility),
        generics: lower_generics(&f.generics),
        params,
        return_type,
        body,
        attributes,
        is_async: f.is_async,
        span: None,
    })
}

fn lower_param(ctx: &mut LoweringContext, p: &ast::Param) -> IrParam {
    let name = extract_pattern_name(&p.pattern);
    let evidence = extract_pattern_evidence(&p.pattern);
    ctx.bind_var(&name);

    IrParam {
        name,
        ty: lower_type_expr(&p.ty),
        evidence,
    }
}

fn extract_pattern_name(p: &ast::Pattern) -> String {
    match p {
        ast::Pattern::Ident { name, .. } => name.name.clone(),
        ast::Pattern::Tuple(pats) if !pats.is_empty() => extract_pattern_name(&pats[0]),
        _ => "_".to_string(),
    }
}

fn extract_pattern_evidence(p: &ast::Pattern) -> IrEvidence {
    match p {
        ast::Pattern::Ident { evidentiality, .. } => evidentiality
            .map(lower_evidentiality)
            .unwrap_or(IrEvidence::Known),
        _ => IrEvidence::Known,
    }
}

fn lower_visibility(v: ast::Visibility) -> IrVisibility {
    match v {
        ast::Visibility::Public => IrVisibility::Public,
        ast::Visibility::Private => IrVisibility::Private,
        ast::Visibility::Crate => IrVisibility::Crate,
        ast::Visibility::Super => IrVisibility::Private,
    }
}

fn lower_evidentiality(e: ast::Evidentiality) -> IrEvidence {
    match e {
        ast::Evidentiality::Known => IrEvidence::Known,
        ast::Evidentiality::Uncertain => IrEvidence::Uncertain,
        ast::Evidentiality::Reported => IrEvidence::Reported,
        ast::Evidentiality::Paradox => IrEvidence::Paradox,
    }
}

fn lower_generics(g: &Option<ast::Generics>) -> Vec<IrGenericParam> {
    g.as_ref()
        .map(|g| {
            g.params
                .iter()
                .filter_map(|p| match p {
                    ast::GenericParam::Type { name, bounds, .. } => Some(IrGenericParam {
                        name: name.name.clone(),
                        bounds: bounds.iter().map(type_expr_to_string).collect(),
                    }),
                    ast::GenericParam::Const { name, .. } => Some(IrGenericParam {
                        name: name.name.clone(),
                        bounds: vec!["const".to_string()],
                    }),
                    ast::GenericParam::Lifetime(l) => Some(IrGenericParam {
                        name: l.clone(),
                        bounds: vec!["lifetime".to_string()],
                    }),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn type_expr_to_string(t: &ast::TypeExpr) -> String {
    match t {
        ast::TypeExpr::Path(p) => p
            .segments
            .iter()
            .map(|s| s.ident.name.clone())
            .collect::<Vec<_>>()
            .join("::"),
        _ => "?".to_string(),
    }
}

fn lower_type_expr(t: &ast::TypeExpr) -> IrType {
    match t {
        ast::TypeExpr::Path(p) => {
            let name = p
                .segments
                .iter()
                .map(|s| s.ident.name.clone())
                .collect::<Vec<_>>()
                .join("::");

            // Check for primitive types
            match name.as_str() {
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" | "f32" | "f64" | "bool" | "char" | "str" => {
                    IrType::Primitive { name }
                }
                "()" => IrType::Unit,
                "!" | "never" => IrType::Never,
                _ => {
                    let generics = p
                        .segments
                        .last()
                        .and_then(|s| s.generics.as_ref())
                        .map(|gs| gs.iter().map(lower_type_expr).collect())
                        .unwrap_or_default();

                    IrType::Named { name, generics }
                }
            }
        }
        ast::TypeExpr::Reference { mutable, inner } => IrType::Reference {
            mutable: *mutable,
            inner: Box::new(lower_type_expr(inner)),
        },
        ast::TypeExpr::Pointer { mutable, inner } => IrType::Pointer {
            mutable: *mutable,
            inner: Box::new(lower_type_expr(inner)),
        },
        ast::TypeExpr::Array { element, size: _ } => IrType::Array {
            element: Box::new(lower_type_expr(element)),
            size: None, // Would need const eval
        },
        ast::TypeExpr::Slice(inner) => IrType::Slice {
            element: Box::new(lower_type_expr(inner)),
        },
        ast::TypeExpr::Tuple(elements) => IrType::Tuple {
            elements: elements.iter().map(lower_type_expr).collect(),
        },
        ast::TypeExpr::Function {
            params,
            return_type,
        } => IrType::Function {
            params: params.iter().map(lower_type_expr).collect(),
            return_type: Box::new(
                return_type
                    .as_ref()
                    .map(|r| lower_type_expr(r))
                    .unwrap_or(IrType::Unit),
            ),
            is_async: false,
        },
        ast::TypeExpr::Evidential {
            inner,
            evidentiality,
        } => IrType::Evidential {
            inner: Box::new(lower_type_expr(inner)),
            evidence: lower_evidentiality(*evidentiality),
        },
        ast::TypeExpr::Cycle { .. } => IrType::Cycle { modulus: 0 },
        ast::TypeExpr::Simd { element, lanes } => IrType::Simd {
            element: Box::new(lower_type_expr(element)),
            lanes: *lanes as usize,
        },
        ast::TypeExpr::Atomic(inner) => IrType::Atomic {
            inner: Box::new(lower_type_expr(inner)),
        },
        ast::TypeExpr::Never => IrType::Never,
        ast::TypeExpr::Infer => IrType::Infer,
    }
}

fn lower_block(ctx: &mut LoweringContext, block: &ast::Block) -> IrOperation {
    ctx.push_scope();

    let mut statements: Vec<IrOperation> = block
        .stmts
        .iter()
        .filter_map(|s| lower_stmt(ctx, s))
        .collect();

    if let Some(expr) = &block.expr {
        statements.push(lower_expr(ctx, expr));
    }

    ctx.pop_scope();

    IrOperation::Block {
        statements,
        ty: IrType::Infer,
        evidence: IrEvidence::Known,
    }
}

fn lower_stmt(ctx: &mut LoweringContext, stmt: &ast::Stmt) -> Option<IrOperation> {
    match stmt {
        ast::Stmt::Let { pattern, ty, init } => {
            let ir_pattern = lower_pattern(ctx, pattern);
            let type_annotation = ty.as_ref().map(lower_type_expr);
            let init_expr = init
                .as_ref()
                .map(|e| Box::new(lower_expr(ctx, e)))
                .unwrap_or_else(|| {
                    Box::new(IrOperation::Literal {
                        variant: LiteralVariant::Null,
                        value: serde_json::Value::Null,
                        ty: IrType::Unit,
                        evidence: IrEvidence::Known,
                    })
                });

            Some(IrOperation::Let {
                pattern: ir_pattern,
                type_annotation,
                init: init_expr,
                evidence: IrEvidence::Known,
            })
        }
        ast::Stmt::Expr(e) => Some(lower_expr(ctx, e)),
        ast::Stmt::Semi(e) => Some(lower_expr(ctx, e)),
        ast::Stmt::Item(_) => None, // Handled at module level
    }
}

fn lower_pattern(ctx: &mut LoweringContext, pattern: &ast::Pattern) -> IrPattern {
    match pattern {
        ast::Pattern::Ident {
            mutable,
            name,
            evidentiality,
        } => {
            ctx.bind_var(&name.name);
            IrPattern::Ident {
                name: name.name.clone(),
                mutable: *mutable,
                evidence: evidentiality.map(lower_evidentiality),
            }
        }
        ast::Pattern::Tuple(pats) => IrPattern::Tuple {
            elements: pats.iter().map(|p| lower_pattern(ctx, p)).collect(),
        },
        ast::Pattern::Struct { path, fields, rest } => IrPattern::Struct {
            path: path
                .segments
                .iter()
                .map(|s| s.ident.name.clone())
                .collect::<Vec<_>>()
                .join("::"),
            fields: fields
                .iter()
                .map(|f| {
                    (
                        f.name.name.clone(),
                        f.pattern.as_ref().map(|p| lower_pattern(ctx, p)).unwrap_or(
                            IrPattern::Ident {
                                name: f.name.name.clone(),
                                mutable: false,
                                evidence: None,
                            },
                        ),
                    )
                })
                .collect(),
            rest: *rest,
        },
        ast::Pattern::TupleStruct { path, fields } => IrPattern::TupleStruct {
            path: path
                .segments
                .iter()
                .map(|s| s.ident.name.clone())
                .collect::<Vec<_>>()
                .join("::"),
            fields: fields.iter().map(|p| lower_pattern(ctx, p)).collect(),
        },
        ast::Pattern::Slice(pats) => IrPattern::Slice {
            elements: pats.iter().map(|p| lower_pattern(ctx, p)).collect(),
        },
        ast::Pattern::Or(pats) => IrPattern::Or {
            patterns: pats.iter().map(|p| lower_pattern(ctx, p)).collect(),
        },
        ast::Pattern::Literal(lit) => IrPattern::Literal {
            value: lower_literal_value(lit),
        },
        ast::Pattern::Range {
            start,
            end,
            inclusive,
        } => IrPattern::Range {
            start: start.as_ref().map(|p| Box::new(lower_pattern(ctx, p))),
            end: end.as_ref().map(|p| Box::new(lower_pattern(ctx, p))),
            inclusive: *inclusive,
        },
        ast::Pattern::Wildcard | ast::Pattern::Rest => IrPattern::Wildcard,
    }
}

fn lower_expr(ctx: &mut LoweringContext, expr: &ast::Expr) -> IrOperation {
    match expr {
        ast::Expr::Literal(lit) => lower_literal(lit),

        ast::Expr::Path(path) => {
            let name = path
                .segments
                .iter()
                .map(|s| s.ident.name.clone())
                .collect::<Vec<_>>()
                .join("::");
            let id = ctx.lookup_var(&name).unwrap_or_else(|| name.clone());

            IrOperation::Var {
                name,
                id,
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Binary { left, op, right } => {
            let left_ir = lower_expr(ctx, left);
            let right_ir = lower_expr(ctx, right);
            let left_ev = get_operation_evidence(&left_ir);
            let right_ev = get_operation_evidence(&right_ir);

            IrOperation::Binary {
                operator: lower_binop(*op),
                left: Box::new(left_ir),
                right: Box::new(right_ir),
                ty: IrType::Infer,
                evidence: left_ev.join(right_ev),
            }
        }

        ast::Expr::Unary { op, expr: inner } => {
            let inner_ir = lower_expr(ctx, inner);
            let evidence = get_operation_evidence(&inner_ir);

            IrOperation::Unary {
                operator: lower_unaryop(*op),
                operand: Box::new(inner_ir),
                ty: IrType::Infer,
                evidence,
            }
        }

        ast::Expr::Call { func, args } => {
            let func_name = match func.as_ref() {
                ast::Expr::Path(p) => p
                    .segments
                    .iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("::"),
                _ => "anonymous".to_string(),
            };

            let args_ir: Vec<IrOperation> = args.iter().map(|a| lower_expr(ctx, a)).collect();
            let evidence = args_ir
                .iter()
                .map(get_operation_evidence)
                .fold(IrEvidence::Known, |acc, e| acc.join(e));

            IrOperation::Call {
                function: func_name.clone(),
                function_id: format!("fn_{}", func_name),
                args: args_ir,
                type_args: vec![],
                ty: IrType::Infer,
                evidence,
            }
        }

        ast::Expr::MethodCall {
            receiver,
            method,
            args,
        } => {
            let receiver_ir = lower_expr(ctx, receiver);
            let args_ir: Vec<IrOperation> = args.iter().map(|a| lower_expr(ctx, a)).collect();
            let evidence = std::iter::once(get_operation_evidence(&receiver_ir))
                .chain(args_ir.iter().map(get_operation_evidence))
                .fold(IrEvidence::Known, |acc, e| acc.join(e));

            IrOperation::MethodCall {
                receiver: Box::new(receiver_ir),
                method: method.name.clone(),
                args: args_ir,
                type_args: vec![],
                ty: IrType::Infer,
                evidence,
            }
        }

        ast::Expr::Pipe { expr, operations } => lower_pipeline(ctx, expr, operations),

        ast::Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_ir = lower_expr(ctx, condition);
            let then_ir = lower_block(ctx, then_branch);
            let else_ir = else_branch.as_ref().map(|e| Box::new(lower_expr(ctx, e)));

            let evidence = get_operation_evidence(&cond_ir)
                .join(get_operation_evidence(&then_ir))
                .join(
                    else_ir
                        .as_ref()
                        .map(|e| get_operation_evidence(e))
                        .unwrap_or(IrEvidence::Known),
                );

            IrOperation::If {
                condition: Box::new(cond_ir),
                then_branch: Box::new(then_ir),
                else_branch: else_ir,
                ty: IrType::Infer,
                evidence,
            }
        }

        ast::Expr::Match {
            expr: scrutinee,
            arms,
        } => {
            let scrutinee_ir = lower_expr(ctx, scrutinee);
            let arms_ir: Vec<IrMatchArm> = arms
                .iter()
                .map(|arm| {
                    ctx.push_scope();
                    let pattern = lower_pattern(ctx, &arm.pattern);
                    let guard = arm.guard.as_ref().map(|g| lower_expr(ctx, g));
                    let body = lower_expr(ctx, &arm.body);
                    ctx.pop_scope();
                    IrMatchArm {
                        pattern,
                        guard,
                        body,
                    }
                })
                .collect();

            IrOperation::Match {
                scrutinee: Box::new(scrutinee_ir),
                arms: arms_ir,
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Loop(block) => IrOperation::Loop {
            variant: LoopVariant::Infinite,
            condition: None,
            iterator: None,
            body: Box::new(lower_block(ctx, block)),
            ty: IrType::Never,
            evidence: IrEvidence::Known,
        },

        ast::Expr::While { condition, body } => IrOperation::Loop {
            variant: LoopVariant::While,
            condition: Some(Box::new(lower_expr(ctx, condition))),
            iterator: None,
            body: Box::new(lower_block(ctx, body)),
            ty: IrType::Unit,
            evidence: IrEvidence::Known,
        },

        ast::Expr::For {
            pattern,
            iter,
            body,
        } => {
            ctx.push_scope();
            let pat = lower_pattern(ctx, pattern);
            let iter_ir = lower_expr(ctx, iter);
            let body_ir = lower_block(ctx, body);
            ctx.pop_scope();

            IrOperation::Loop {
                variant: LoopVariant::For,
                condition: None,
                iterator: Some(IrForIterator {
                    pattern: pat,
                    iterable: Box::new(iter_ir),
                }),
                body: Box::new(body_ir),
                ty: IrType::Unit,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Closure { params, body } => {
            ctx.push_scope();
            let params_ir: Vec<IrParam> = params
                .iter()
                .map(|p| {
                    let name = extract_pattern_name(&p.pattern);
                    ctx.bind_var(&name);
                    IrParam {
                        name,
                        ty: p.ty.as_ref().map(lower_type_expr).unwrap_or(IrType::Infer),
                        evidence: IrEvidence::Known,
                    }
                })
                .collect();
            let body_ir = lower_expr(ctx, body);
            ctx.pop_scope();

            IrOperation::Closure {
                params: params_ir,
                body: Box::new(body_ir),
                captures: vec![],
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Block(block) => lower_block(ctx, block),

        ast::Expr::Array(elements) => {
            let elements_ir: Vec<IrOperation> =
                elements.iter().map(|e| lower_expr(ctx, e)).collect();
            IrOperation::Array {
                elements: elements_ir,
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Tuple(elements) => {
            let elements_ir: Vec<IrOperation> =
                elements.iter().map(|e| lower_expr(ctx, e)).collect();
            IrOperation::Tuple {
                elements: elements_ir,
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Struct { path, fields, rest } => {
            let name = path
                .segments
                .iter()
                .map(|s| s.ident.name.clone())
                .collect::<Vec<_>>()
                .join("::");

            let fields_ir: Vec<(String, IrOperation)> = fields
                .iter()
                .map(|f| {
                    let value = f
                        .value
                        .as_ref()
                        .map(|v| lower_expr(ctx, v))
                        .unwrap_or_else(|| IrOperation::Var {
                            name: f.name.name.clone(),
                            id: ctx.lookup_var(&f.name.name).unwrap_or_default(),
                            ty: IrType::Infer,
                            evidence: IrEvidence::Known,
                        });
                    (f.name.name.clone(), value)
                })
                .collect();

            IrOperation::StructInit {
                name,
                fields: fields_ir,
                rest: rest.as_ref().map(|r| Box::new(lower_expr(ctx, r))),
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Field { expr: inner, field } => IrOperation::Field {
            expr: Box::new(lower_expr(ctx, inner)),
            field: field.name.clone(),
            ty: IrType::Infer,
            evidence: IrEvidence::Known,
        },

        ast::Expr::Index { expr: inner, index } => IrOperation::Index {
            expr: Box::new(lower_expr(ctx, inner)),
            index: Box::new(lower_expr(ctx, index)),
            ty: IrType::Infer,
            evidence: IrEvidence::Known,
        },

        ast::Expr::Assign { target, value } => IrOperation::Assign {
            target: Box::new(lower_expr(ctx, target)),
            value: Box::new(lower_expr(ctx, value)),
            evidence: IrEvidence::Known,
        },

        ast::Expr::Return(value) => IrOperation::Return {
            value: value.as_ref().map(|v| Box::new(lower_expr(ctx, v))),
            evidence: IrEvidence::Known,
        },

        ast::Expr::Break(value) => IrOperation::Break {
            value: value.as_ref().map(|v| Box::new(lower_expr(ctx, v))),
            evidence: IrEvidence::Known,
        },

        ast::Expr::Continue => IrOperation::Continue {
            evidence: IrEvidence::Known,
        },

        ast::Expr::Await(inner) => {
            let inner_ir = lower_expr(ctx, inner);
            let evidence = get_operation_evidence(&inner_ir);
            IrOperation::Await {
                expr: Box::new(inner_ir),
                ty: IrType::Infer,
                evidence,
            }
        }

        ast::Expr::Try(inner) => {
            let inner_ir = lower_expr(ctx, inner);
            IrOperation::Try {
                expr: Box::new(inner_ir),
                ty: IrType::Infer,
                evidence: IrEvidence::Uncertain, // Try always produces uncertain
            }
        }

        ast::Expr::Unsafe(block) => IrOperation::Unsafe {
            body: Box::new(lower_block(ctx, block)),
            ty: IrType::Infer,
            evidence: IrEvidence::Paradox, // Unsafe always produces paradox
        },

        ast::Expr::Cast { expr: inner, ty } => {
            let inner_ir = lower_expr(ctx, inner);
            let evidence = get_operation_evidence(&inner_ir);
            IrOperation::Cast {
                expr: Box::new(inner_ir),
                target_type: lower_type_expr(ty),
                ty: lower_type_expr(ty),
                evidence,
            }
        }

        ast::Expr::Evidential {
            expr: inner,
            evidentiality,
        } => {
            let inner_ir = lower_expr(ctx, inner);
            let from_evidence = get_operation_evidence(&inner_ir);
            let to_evidence = lower_evidentiality(*evidentiality);

            IrOperation::EvidenceCoerce {
                operation: EvidenceOp::Mark,
                expr: Box::new(inner_ir),
                from_evidence,
                to_evidence,
                ty: IrType::Infer,
            }
        }

        ast::Expr::Morpheme { kind, body } => {
            let body_ir = lower_expr(ctx, body);
            IrOperation::Morpheme {
                morpheme: lower_morpheme_kind(*kind),
                symbol: morpheme_symbol(*kind).to_string(),
                input: Box::new(IrOperation::Var {
                    name: "_".to_string(),
                    id: "implicit_input".to_string(),
                    ty: IrType::Infer,
                    evidence: IrEvidence::Known,
                }),
                body: Some(Box::new(body_ir)),
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        ast::Expr::Incorporation { segments } => {
            let segs: Vec<IncorporationSegment> = segments
                .iter()
                .map(|s| {
                    if s.args.is_some() {
                        IncorporationSegment::Verb {
                            name: s.name.name.clone(),
                        }
                    } else {
                        IncorporationSegment::Noun {
                            name: s.name.name.clone(),
                        }
                    }
                })
                .collect();

            // Collect all args from all segments
            let mut args: Vec<IrOperation> = Vec::new();
            for seg in segments {
                if let Some(ref seg_args) = seg.args {
                    for a in seg_args {
                        args.push(lower_expr(ctx, a));
                    }
                }
            }

            IrOperation::Incorporation {
                segments: segs,
                args,
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            }
        }

        // Protocol operations - all yield Reported evidence
        ast::Expr::HttpRequest {
            method,
            url,
            headers,
            body,
            timeout,
        } => IrOperation::HttpRequest {
            method: lower_http_method(*method),
            url: Box::new(lower_expr(ctx, url)),
            headers: if headers.is_empty() {
                None
            } else {
                Some(Box::new(IrOperation::Array {
                    elements: headers
                        .iter()
                        .map(|(k, v)| IrOperation::Tuple {
                            elements: vec![lower_expr(ctx, k), lower_expr(ctx, v)],
                            ty: IrType::Infer,
                            evidence: IrEvidence::Known,
                        })
                        .collect(),
                    ty: IrType::Infer,
                    evidence: IrEvidence::Known,
                }))
            },
            body: body.as_ref().map(|b| Box::new(lower_expr(ctx, b))),
            timeout: timeout.as_ref().map(|t| Box::new(lower_expr(ctx, t))),
            ty: IrType::Infer,
            evidence: IrEvidence::Reported,
        },

        ast::Expr::GrpcCall {
            service,
            method,
            message,
            metadata,
            timeout,
        } => IrOperation::GrpcCall {
            service: expr_to_string(service),
            method: expr_to_string(method),
            message: Box::new(message.as_ref().map(|m| lower_expr(ctx, m)).unwrap_or(
                IrOperation::Literal {
                    variant: LiteralVariant::Null,
                    value: serde_json::Value::Null,
                    ty: IrType::Unit,
                    evidence: IrEvidence::Known,
                },
            )),
            metadata: if metadata.is_empty() {
                None
            } else {
                Some(Box::new(IrOperation::Array {
                    elements: metadata
                        .iter()
                        .map(|(k, v)| IrOperation::Tuple {
                            elements: vec![lower_expr(ctx, k), lower_expr(ctx, v)],
                            ty: IrType::Infer,
                            evidence: IrEvidence::Known,
                        })
                        .collect(),
                    ty: IrType::Infer,
                    evidence: IrEvidence::Known,
                }))
            },
            timeout: timeout.as_ref().map(|t| Box::new(lower_expr(ctx, t))),
            ty: IrType::Infer,
            evidence: IrEvidence::Reported,
        },

        // Handle remaining expression types with placeholders
        _ => IrOperation::Literal {
            variant: LiteralVariant::Null,
            value: serde_json::json!({"unhandled": format!("{:?}", std::mem::discriminant(expr))}),
            ty: IrType::Infer,
            evidence: IrEvidence::Known,
        },
    }
}

fn lower_pipeline(
    ctx: &mut LoweringContext,
    input: &ast::Expr,
    operations: &[ast::PipeOp],
) -> IrOperation {
    let input_ir = lower_expr(ctx, input);
    let mut evidence = get_operation_evidence(&input_ir);

    let steps: Vec<IrPipelineStep> = operations
        .iter()
        .map(|op| {
            let step = lower_pipe_op(ctx, op);
            // Update evidence based on operation
            evidence = evidence.join(pipe_op_evidence(op));
            step
        })
        .collect();

    IrOperation::Pipeline {
        input: Box::new(input_ir),
        steps,
        ty: IrType::Infer,
        evidence,
    }
}

fn lower_pipe_op(ctx: &mut LoweringContext, op: &ast::PipeOp) -> IrPipelineStep {
    match op {
        ast::PipeOp::Transform(body) => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Transform,
            symbol: "τ".to_string(),
            body: Some(Box::new(lower_expr(ctx, body))),
        },
        ast::PipeOp::Filter(body) => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Filter,
            symbol: "φ".to_string(),
            body: Some(Box::new(lower_expr(ctx, body))),
        },
        ast::PipeOp::Sort(field) => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Sort,
            symbol: "σ".to_string(),
            body: field.as_ref().map(|f| {
                Box::new(IrOperation::Var {
                    name: f.name.clone(),
                    id: f.name.clone(),
                    ty: IrType::Infer,
                    evidence: IrEvidence::Known,
                })
            }),
        },
        ast::PipeOp::Reduce(body) => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Reduce,
            symbol: "ρ".to_string(),
            body: Some(Box::new(lower_expr(ctx, body))),
        },
        ast::PipeOp::First => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::First,
            symbol: "α".to_string(),
            body: None,
        },
        ast::PipeOp::Last => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Last,
            symbol: "ω".to_string(),
            body: None,
        },
        ast::PipeOp::Middle => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Middle,
            symbol: "μ".to_string(),
            body: None,
        },
        ast::PipeOp::Choice => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Choice,
            symbol: "χ".to_string(),
            body: None,
        },
        ast::PipeOp::Nth(n) => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Nth,
            symbol: "ν".to_string(),
            body: Some(Box::new(lower_expr(ctx, n))),
        },
        ast::PipeOp::Next => IrPipelineStep::Morpheme {
            morpheme: MorphemeKind::Next,
            symbol: "ξ".to_string(),
            body: None,
        },
        ast::PipeOp::Method { name, args } => IrPipelineStep::Method {
            name: name.name.clone(),
            args: args.iter().map(|a| lower_expr(ctx, a)).collect(),
        },
        ast::PipeOp::Await => IrPipelineStep::Await,
        ast::PipeOp::Named { prefix, body } => {
            let fn_name = prefix
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>()
                .join("·");
            IrPipelineStep::Call {
                function: fn_name,
                args: body
                    .as_ref()
                    .map(|b| vec![lower_expr(ctx, b)])
                    .unwrap_or_default(),
            }
        }
        // Protocol operations in pipeline
        ast::PipeOp::Send(data) => IrPipelineStep::Protocol {
            operation: ProtocolOp::Send,
            config: Some(Box::new(lower_expr(ctx, data))),
        },
        ast::PipeOp::Recv => IrPipelineStep::Protocol {
            operation: ProtocolOp::Recv,
            config: None,
        },
        ast::PipeOp::Stream(handler) => IrPipelineStep::Protocol {
            operation: ProtocolOp::Stream,
            config: Some(Box::new(lower_expr(ctx, handler))),
        },
        ast::PipeOp::Connect(config) => IrPipelineStep::Protocol {
            operation: ProtocolOp::Connect,
            config: config.as_ref().map(|c| Box::new(lower_expr(ctx, c))),
        },
        ast::PipeOp::Close => IrPipelineStep::Protocol {
            operation: ProtocolOp::Close,
            config: None,
        },
        ast::PipeOp::Timeout(ms) => IrPipelineStep::Protocol {
            operation: ProtocolOp::Timeout,
            config: Some(Box::new(lower_expr(ctx, ms))),
        },
        ast::PipeOp::Retry { count, strategy } => IrPipelineStep::Protocol {
            operation: ProtocolOp::Retry,
            config: Some(Box::new(IrOperation::Tuple {
                elements: vec![
                    lower_expr(ctx, count),
                    strategy
                        .as_ref()
                        .map(|s| lower_expr(ctx, s))
                        .unwrap_or(IrOperation::Literal {
                            variant: LiteralVariant::String,
                            value: serde_json::json!("exponential"),
                            ty: IrType::Primitive {
                                name: "str".to_string(),
                            },
                            evidence: IrEvidence::Known,
                        }),
                ],
                ty: IrType::Infer,
                evidence: IrEvidence::Known,
            })),
        },
        _ => IrPipelineStep::Identity,
    }
}

fn pipe_op_evidence(op: &ast::PipeOp) -> IrEvidence {
    match op {
        // Protocol operations always produce Reported evidence
        ast::PipeOp::Send(_)
        | ast::PipeOp::Recv
        | ast::PipeOp::Stream(_)
        | ast::PipeOp::Connect(_)
        | ast::PipeOp::Close => IrEvidence::Reported,
        _ => IrEvidence::Known,
    }
}

fn lower_literal(lit: &ast::Literal) -> IrOperation {
    match lit {
        ast::Literal::Int {
            value,
            base,
            suffix,
        } => IrOperation::Literal {
            variant: LiteralVariant::Int,
            value: serde_json::json!({
                "value": value,
                "base": format!("{:?}", base),
                "suffix": suffix
            }),
            ty: suffix
                .as_ref()
                .map(|s| IrType::Primitive { name: s.clone() })
                .unwrap_or(IrType::Primitive {
                    name: "i64".to_string(),
                }),
            evidence: IrEvidence::Known,
        },
        ast::Literal::Float { value, suffix } => IrOperation::Literal {
            variant: LiteralVariant::Float,
            value: serde_json::json!({"value": value, "suffix": suffix}),
            ty: suffix
                .as_ref()
                .map(|s| IrType::Primitive { name: s.clone() })
                .unwrap_or(IrType::Primitive {
                    name: "f64".to_string(),
                }),
            evidence: IrEvidence::Known,
        },
        ast::Literal::String(s) | ast::Literal::MultiLineString(s) | ast::Literal::RawString(s) => {
            IrOperation::Literal {
                variant: LiteralVariant::String,
                value: serde_json::Value::String(s.clone()),
                ty: IrType::Primitive {
                    name: "str".to_string(),
                },
                evidence: IrEvidence::Known,
            }
        }
        ast::Literal::Char(c) => IrOperation::Literal {
            variant: LiteralVariant::Char,
            value: serde_json::Value::String(c.to_string()),
            ty: IrType::Primitive {
                name: "char".to_string(),
            },
            evidence: IrEvidence::Known,
        },
        ast::Literal::Bool(b) => IrOperation::Literal {
            variant: LiteralVariant::Bool,
            value: serde_json::Value::Bool(*b),
            ty: IrType::Primitive {
                name: "bool".to_string(),
            },
            evidence: IrEvidence::Known,
        },
        ast::Literal::Null | ast::Literal::Empty => IrOperation::Literal {
            variant: LiteralVariant::Null,
            value: serde_json::Value::Null,
            ty: IrType::Unit,
            evidence: IrEvidence::Known,
        },
        _ => IrOperation::Literal {
            variant: LiteralVariant::Null,
            value: serde_json::Value::Null,
            ty: IrType::Infer,
            evidence: IrEvidence::Known,
        },
    }
}

fn lower_literal_value(lit: &ast::Literal) -> serde_json::Value {
    match lit {
        ast::Literal::Int { value, .. } => serde_json::json!(value),
        ast::Literal::Float { value, .. } => serde_json::json!(value),
        ast::Literal::String(s) => serde_json::Value::String(s.clone()),
        ast::Literal::Bool(b) => serde_json::Value::Bool(*b),
        ast::Literal::Char(c) => serde_json::Value::String(c.to_string()),
        ast::Literal::Null => serde_json::Value::Null,
        _ => serde_json::Value::Null,
    }
}

fn lower_binop(op: ast::BinOp) -> BinaryOp {
    match op {
        ast::BinOp::Add => BinaryOp::Add,
        ast::BinOp::Sub => BinaryOp::Sub,
        ast::BinOp::Mul => BinaryOp::Mul,
        ast::BinOp::Div => BinaryOp::Div,
        ast::BinOp::Rem => BinaryOp::Rem,
        ast::BinOp::Pow => BinaryOp::Pow,
        ast::BinOp::And => BinaryOp::And,
        ast::BinOp::Or => BinaryOp::Or,
        ast::BinOp::BitAnd => BinaryOp::BitAnd,
        ast::BinOp::BitOr => BinaryOp::BitOr,
        ast::BinOp::BitXor => BinaryOp::BitXor,
        ast::BinOp::Shl => BinaryOp::Shl,
        ast::BinOp::Shr => BinaryOp::Shr,
        ast::BinOp::Eq => BinaryOp::Eq,
        ast::BinOp::Ne => BinaryOp::Ne,
        ast::BinOp::Lt => BinaryOp::Lt,
        ast::BinOp::Le => BinaryOp::Le,
        ast::BinOp::Gt => BinaryOp::Gt,
        ast::BinOp::Ge => BinaryOp::Ge,
        ast::BinOp::Concat => BinaryOp::Concat,
    }
}

fn lower_unaryop(op: ast::UnaryOp) -> UnaryOp {
    match op {
        ast::UnaryOp::Neg => UnaryOp::Neg,
        ast::UnaryOp::Not => UnaryOp::Not,
        ast::UnaryOp::Deref => UnaryOp::Deref,
        ast::UnaryOp::Ref => UnaryOp::Ref,
        ast::UnaryOp::RefMut => UnaryOp::RefMut,
    }
}

fn lower_morpheme_kind(kind: ast::MorphemeKind) -> MorphemeKind {
    match kind {
        ast::MorphemeKind::Transform => MorphemeKind::Transform,
        ast::MorphemeKind::Filter => MorphemeKind::Filter,
        ast::MorphemeKind::Sort => MorphemeKind::Sort,
        ast::MorphemeKind::Reduce => MorphemeKind::Reduce,
        ast::MorphemeKind::Lambda => MorphemeKind::Lambda,
        ast::MorphemeKind::Sum => MorphemeKind::Sum,
        ast::MorphemeKind::Product => MorphemeKind::Product,
        ast::MorphemeKind::First => MorphemeKind::First,
        ast::MorphemeKind::Last => MorphemeKind::Last,
        ast::MorphemeKind::Middle => MorphemeKind::Middle,
        ast::MorphemeKind::Choice => MorphemeKind::Choice,
        ast::MorphemeKind::Nth => MorphemeKind::Nth,
        ast::MorphemeKind::Next => MorphemeKind::Next,
    }
}

fn morpheme_symbol(kind: ast::MorphemeKind) -> &'static str {
    match kind {
        ast::MorphemeKind::Transform => "τ",
        ast::MorphemeKind::Filter => "φ",
        ast::MorphemeKind::Sort => "σ",
        ast::MorphemeKind::Reduce => "ρ",
        ast::MorphemeKind::Lambda => "λ",
        ast::MorphemeKind::Sum => "Σ",
        ast::MorphemeKind::Product => "Π",
        ast::MorphemeKind::First => "α",
        ast::MorphemeKind::Last => "ω",
        ast::MorphemeKind::Middle => "μ",
        ast::MorphemeKind::Choice => "χ",
        ast::MorphemeKind::Nth => "ν",
        ast::MorphemeKind::Next => "ξ",
    }
}

fn lower_http_method(method: ast::HttpMethod) -> HttpMethod {
    match method {
        ast::HttpMethod::Get => HttpMethod::Get,
        ast::HttpMethod::Post => HttpMethod::Post,
        ast::HttpMethod::Put => HttpMethod::Put,
        ast::HttpMethod::Delete => HttpMethod::Delete,
        ast::HttpMethod::Patch => HttpMethod::Patch,
        ast::HttpMethod::Head => HttpMethod::Head,
        ast::HttpMethod::Options => HttpMethod::Options,
        ast::HttpMethod::Connect => HttpMethod::Connect,
        ast::HttpMethod::Trace => HttpMethod::Trace,
    }
}

fn get_operation_evidence(op: &IrOperation) -> IrEvidence {
    match op {
        IrOperation::Literal { evidence, .. }
        | IrOperation::Var { evidence, .. }
        | IrOperation::Let { evidence, .. }
        | IrOperation::Binary { evidence, .. }
        | IrOperation::Unary { evidence, .. }
        | IrOperation::Call { evidence, .. }
        | IrOperation::MethodCall { evidence, .. }
        | IrOperation::Closure { evidence, .. }
        | IrOperation::If { evidence, .. }
        | IrOperation::Match { evidence, .. }
        | IrOperation::Loop { evidence, .. }
        | IrOperation::Block { evidence, .. }
        | IrOperation::Pipeline { evidence, .. }
        | IrOperation::Morpheme { evidence, .. }
        | IrOperation::Fork { evidence, .. }
        | IrOperation::Identity { evidence, .. }
        | IrOperation::Array { evidence, .. }
        | IrOperation::Tuple { evidence, .. }
        | IrOperation::StructInit { evidence, .. }
        | IrOperation::Field { evidence, .. }
        | IrOperation::Index { evidence, .. }
        | IrOperation::Assign { evidence, .. }
        | IrOperation::Break { evidence, .. }
        | IrOperation::Continue { evidence }
        | IrOperation::Return { evidence, .. }
        | IrOperation::Incorporation { evidence, .. }
        | IrOperation::Affect { evidence, .. }
        | IrOperation::HttpRequest { evidence, .. }
        | IrOperation::GrpcCall { evidence, .. }
        | IrOperation::WebSocket { evidence, .. }
        | IrOperation::KafkaOp { evidence, .. }
        | IrOperation::Await { evidence, .. }
        | IrOperation::Unsafe { evidence, .. }
        | IrOperation::Cast { evidence, .. }
        | IrOperation::Try { evidence, .. } => *evidence,
        IrOperation::EvidenceCoerce { to_evidence, .. } => *to_evidence,
    }
}

fn expr_to_string(e: &ast::Expr) -> String {
    match e {
        ast::Expr::Path(p) => p
            .segments
            .iter()
            .map(|s| s.ident.name.clone())
            .collect::<Vec<_>>()
            .join("::"),
        ast::Expr::Literal(ast::Literal::String(s)) => s.clone(),
        _ => "dynamic".to_string(),
    }
}

// === Type definitions lowering ===

fn lower_struct_def(_ctx: &mut LoweringContext, s: &ast::StructDef) -> IrTypeDef {
    let fields = match &s.fields {
        ast::StructFields::Named(fields) => fields
            .iter()
            .map(|f| IrField {
                name: f.name.name.clone(),
                ty: lower_type_expr(&f.ty),
                visibility: lower_visibility(f.visibility),
            })
            .collect(),
        ast::StructFields::Tuple(types) => types
            .iter()
            .enumerate()
            .map(|(i, t)| IrField {
                name: format!("{}", i),
                ty: lower_type_expr(t),
                visibility: IrVisibility::Public,
            })
            .collect(),
        ast::StructFields::Unit => vec![],
    };

    IrTypeDef::Struct {
        name: s.name.name.clone(),
        generics: lower_generics(&s.generics),
        fields,
        span: None,
    }
}

fn lower_enum_def(_ctx: &mut LoweringContext, e: &ast::EnumDef) -> IrTypeDef {
    let variants = e
        .variants
        .iter()
        .map(|v| {
            let fields = match &v.fields {
                ast::StructFields::Named(fields) => Some(
                    fields
                        .iter()
                        .map(|f| IrField {
                            name: f.name.name.clone(),
                            ty: lower_type_expr(&f.ty),
                            visibility: lower_visibility(f.visibility),
                        })
                        .collect(),
                ),
                ast::StructFields::Tuple(types) => Some(
                    types
                        .iter()
                        .enumerate()
                        .map(|(i, t)| IrField {
                            name: format!("{}", i),
                            ty: lower_type_expr(t),
                            visibility: IrVisibility::Public,
                        })
                        .collect(),
                ),
                ast::StructFields::Unit => None,
            };

            IrVariant {
                name: v.name.name.clone(),
                fields,
                discriminant: None,
            }
        })
        .collect();

    IrTypeDef::Enum {
        name: e.name.name.clone(),
        generics: lower_generics(&e.generics),
        variants,
        span: None,
    }
}

fn lower_type_alias(_ctx: &mut LoweringContext, t: &ast::TypeAlias) -> IrTypeDef {
    IrTypeDef::TypeAlias {
        name: t.name.name.clone(),
        generics: lower_generics(&t.generics),
        target: lower_type_expr(&t.ty),
        span: None,
    }
}

fn lower_trait_def(ctx: &mut LoweringContext, t: &ast::TraitDef) -> IrTraitDef {
    let methods = t
        .items
        .iter()
        .filter_map(|item| match item {
            ast::TraitItem::Function(f) => lower_function(ctx, f),
            _ => None,
        })
        .collect();

    IrTraitDef {
        name: t.name.name.clone(),
        generics: lower_generics(&t.generics),
        super_traits: t.supertraits.iter().map(type_expr_to_string).collect(),
        methods,
        span: None,
    }
}

fn lower_impl_block(ctx: &mut LoweringContext, i: &ast::ImplBlock) -> IrImplBlock {
    let methods = i
        .items
        .iter()
        .filter_map(|item| match item {
            ast::ImplItem::Function(f) => lower_function(ctx, f),
            _ => None,
        })
        .collect();

    IrImplBlock {
        trait_name: i.trait_.as_ref().map(|t| {
            t.segments
                .iter()
                .map(|s| s.ident.name.clone())
                .collect::<Vec<_>>()
                .join("::")
        }),
        target_type: lower_type_expr(&i.self_ty),
        generics: lower_generics(&i.generics),
        methods,
        span: None,
    }
}

fn lower_const_def(ctx: &mut LoweringContext, c: &ast::ConstDef) -> IrConstant {
    IrConstant {
        name: c.name.name.clone(),
        ty: lower_type_expr(&c.ty),
        value: lower_expr(ctx, &c.value),
        visibility: lower_visibility(c.visibility),
        span: None,
    }
}

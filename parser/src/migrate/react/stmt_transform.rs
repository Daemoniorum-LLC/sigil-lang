//! JavaScript statements to Sigil statements.
//!
//! S30: handler bodies were reduced to a flat list of state mutations, so a
//! handler that wrote one field on success and another on failure emitted both,
//! unconditionally, in source order. The branches were gone before the generator
//! saw them. The same gap left 206 of 346 helper bodies untranslated — a helper
//! whose body is one `return` had an expression the migrator could transform,
//! and anything with a statement in it did not.
//!
//! Both need the same thing, which is this: a walk over the statement AST that
//! keeps the shape. Everything it cannot translate becomes a `// React:` comment
//! carrying the original source, so an untranslated statement is visible rather
//! than silently dropped or silently reordered.

use swc_common::{FileName, FilePathMapping, SourceMap, Spanned, sync::Lrc};
use swc_ecma_ast::*;
use swc_ecma_parser::{Parser, StringInput, Syntax, TsSyntax};

use super::ast_transform::{transform_expression, TransformConfig};

/// The result of translating a function body.
#[derive(Debug, Clone)]
pub struct StatementResult {
    /// Sigil statements, one per line, without surrounding braces.
    pub code: String,
    /// Whether every statement translated. False when anything became a
    /// `// React:` comment.
    pub complete: bool,
    /// What could not be translated, for the caller to report.
    pub warnings: Vec<String>,
}

/// Translate a JavaScript function body to Sigil statements.
///
/// `code` may be a braced block (`{ … }`) or the expression of a concise arrow
/// (`x => expr`), which becomes a single tail expression.
pub fn transform_statements(code: &str, config: &TransformConfig, indent: &str) -> StatementResult {
    let src = code.trim();
    if src.is_empty() {
        return StatementResult { code: String::new(), complete: true, warnings: vec![] };
    }

    // A concise arrow body is an expression, not a block.
    if !src.starts_with('{') {
        let r = transform_expression(src, config);
        return StatementResult {
            code: format!("{}{}", indent, r.code),
            complete: r.complete,
            warnings: r.warnings,
        };
    }

    match parse_block(src) {
        Ok(block) => {
            let mut t = StmtTransformer { config, src, warnings: vec![], locals: vec![] };
            let code = t.block_body(&block.stmts, indent);
            StatementResult { code, complete: t.warnings.is_empty(), warnings: t.warnings }
        }
        Err(e) => StatementResult {
            code: format!("{}// React: {}", indent, one_line(src)),
            complete: false,
            warnings: vec![format!("could not parse body: {}", e)],
        },
    }
}

fn parse_block(code: &str) -> Result<BlockStmt, String> {
    let cm: Lrc<SourceMap> = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    // `parse_stmt` on a braced block yields a block statement.
    let fm = cm.new_source_file(FileName::Anon.into(), code.to_string());
    let mut parser = Parser::new(
        Syntax::Typescript(TsSyntax { tsx: true, ..Default::default() }),
        StringInput::from(&*fm),
        None,
    );
    match parser.parse_stmt() {
        Ok(Stmt::Block(b)) => Ok(b),
        Ok(other) => Ok(BlockStmt { span: Default::default(), stmts: vec![other], ctxt: Default::default() }),
        Err(e) => Err(format!("{:?}", e)),
    }
}

/// Collapse source to one line so it can live in a `//` comment.
fn one_line(s: &str) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 140 {
        let cut: String = flat.chars().take(137).collect();
        format!("{}…", cut)
    } else {
        flat
    }
}

struct StmtTransformer<'a> {
    config: &'a TransformConfig,
    /// The body's own source, for the `// React:` comments.
    src: &'a str,
    warnings: Vec<String>,
    /// Names bound by `let`/`const` in this body, so a later reference is not
    /// mistaken for a state field and prefixed with `self.`.
    locals: Vec<String>,
}

impl<'a> StmtTransformer<'a> {
    /// Transform an expression with the locals this body has bound so far in
    /// scope.
    fn expr(&mut self, e: &Expr) -> String {
        let mut config = self.config.clone();
        config.locals.extend(self.locals.iter().cloned());
        let r = transform_expression(&slice(self.src, e.span()), &config);
        if !r.complete {
            self.warnings.extend(r.warnings);
        }
        r.code
    }

    /// `p.then(cb).catch(cb).finally(cb)` as Sigil statements, or `None` when
    /// the expression is not a promise chain.
    ///
    /// The awaited value is bound only when a `then` callback names a
    /// parameter, so the common zero-argument `then(() => { … })` stays a bare
    /// `await` followed by the callback's own body. `catch` is marked rather
    /// than translated — Sigil has no exceptions, so the handler cannot run —
    /// and `finally` is emitted, because it does.
    fn promise_chain(&mut self, expr: &Expr, indent: &str) -> Option<String> {
        // Peel `.then(..)` / `.catch(..)` / `.finally(..)` off the outside in.
        let mut links: Vec<(String, Option<Expr>)> = Vec::new();
        let mut cursor = expr;
        loop {
            let Expr::Call(call) = cursor else { break };
            let Callee::Expr(callee) = &call.callee else { break };
            let Expr::Member(member) = callee.as_ref() else { break };
            let MemberProp::Ident(ident) = &member.prop else { break };
            let name = ident.sym.to_string();
            if !matches!(name.as_str(), "then" | "catch" | "finally") {
                break;
            }
            let cb = call.args.first().map(|a| (*a.expr).clone());
            links.push((name, cb));
            cursor = &member.obj;
        }
        if links.is_empty() {
            return None;
        }
        links.reverse();

        // Does any `then` callback name the resolved value?
        let bound = links.iter().find_map(|(name, cb)| {
            if name != "then" {
                return None;
            }
            cb.as_ref().and_then(callback_param)
        });

        let mut out: Vec<String> = Vec::new();
        let base = self.expr(cursor);
        match &bound {
            Some(param) => {
                out.push(format!("{}≔ {} = {}.await;", indent, param, base));
                self.locals.push(param.clone());
            }
            None => out.push(format!("{}{}.await;", indent, base)),
        }

        for (name, cb) in &links {
            match (name.as_str(), cb) {
                ("catch", Some(cb)) => {
                    self.warnings.push("promise catch not translated".to_string());
                    out.push(format!(
                        "{}// React: .catch({}) — Sigil has no exceptions; this did not run.",
                        indent,
                        one_line(&slice(self.src, cb.span()))
                    ));
                }
                ("catch", None) => {}
                (_, Some(cb)) => match callback_body(cb) {
                    Some(body) => out.push(self.callback_statements(&body, indent)),
                    None => {
                        self.warnings.push(format!("promise {} not translated", name));
                        out.push(format!(
                            "{}// React: .{}({})",
                            indent,
                            name,
                            one_line(&slice(self.src, cb.span()))
                        ));
                    }
                },
                (_, None) => {}
            }
        }
        Some(out.join("\n"))
    }

    /// A promise callback's body, translated in this transformer's scope.
    fn callback_statements(&mut self, body: &BlockStmtOrExpr, indent: &str) -> String {
        match body {
            BlockStmtOrExpr::BlockStmt(block) => {
                let stmts = block.stmts.clone();
                self.block_body(&stmts, indent)
            }
            BlockStmtOrExpr::Expr(e) => {
                let code = self.expr(e);
                let trimmed = code.trim();
                if trimmed.starts_with('{') && trimmed.ends_with('}') {
                    let inner = trimmed[1..trimmed.len() - 1].trim();
                    let inner = inner.strip_suffix(';').unwrap_or(inner);
                    format!("{}{};", indent, inner)
                } else {
                    format!("{}{};", indent, code)
                }
            }
        }
    }

    fn block_body(&mut self, stmts: &[Stmt], indent: &str) -> String {
        let mut out: Vec<String> = Vec::new();
        for stmt in stmts {
            let line = self.stmt(stmt, indent);
            if !line.trim().is_empty() {
                out.push(line);
            }
        }
        out.join("\n")
    }

    fn nested(&mut self, stmt: &Stmt, indent: &str) -> String {
        match stmt {
            Stmt::Block(b) => self.block_body(&b.stmts, indent),
            other => self.stmt(other, indent),
        }
    }

    fn stmt(&mut self, stmt: &Stmt, indent: &str) -> String {
        let inner = format!("{}    ", indent);
        match stmt {
            Stmt::Block(b) => self.block_body(&b.stmts, indent),
            Stmt::Empty(_) => String::new(),

            Stmt::Expr(e) => {
                // `f().then(cb).catch(cb)` is a promise chain, and Sigil spells
                // that `.await` plus the callback's own statements. Left to the
                // expression transform it emitted calls to `then` and `catch`,
                // which are not functions in Sigil and failed the WASM build.
                if let Some(unrolled) = self.promise_chain(&e.expr, indent) {
                    return unrolled;
                }

                // `setBusy(x)` comes back from the expression transform as the
                // block `{ self.busy = x; }`, which is an expression there and a
                // statement here. Left wrapped, a block after an `⎇` with no
                // `⎉` reads as that `if`'s else-arm.
                let code = self.expr(&e.expr);
                let trimmed = code.trim();
                if trimmed.starts_with('{') && trimmed.ends_with('}') {
                    let inner = trimmed[1..trimmed.len() - 1].trim().to_string();
                    let inner = inner.strip_suffix(';').unwrap_or(&inner).to_string();
                    format!("{}{};", indent, inner)
                } else {
                    format!("{}{};", indent, code)
                }
            }

            Stmt::Return(r) => match &r.arg {
                Some(arg) => format!("{}⤺ {};", indent, self.expr(arg)),
                None => format!("{}⤺ ∅;", indent),
            },

            Stmt::Decl(Decl::Var(var)) => self.var_decl(var, indent),

            Stmt::If(if_stmt) => {
                let cond = self.expr(&if_stmt.test);
                let then = self.nested(&if_stmt.cons, &inner);
                match &if_stmt.alt {
                    None => format!("{}⎇ {} {{\n{}\n{}}}", indent, cond, then, indent),
                    Some(alt) => {
                        // `else if` chains stay chains rather than nesting.
                        let is_else_if = matches!(**alt, Stmt::If(_));
                        if is_else_if {
                            let rest = self.stmt(alt, indent);
                            let rest = rest.trim_start().to_string();
                            format!(
                                "{}⎇ {} {{\n{}\n{}}} ⎉ {}",
                                indent, cond, then, indent, rest
                            )
                        } else {
                            let other = self.nested(alt, &inner);
                            format!(
                                "{}⎇ {} {{\n{}\n{}}} ⎉ {{\n{}\n{}}}",
                                indent, cond, then, indent, other, indent
                            )
                        }
                    }
                }
            }

            Stmt::ForOf(for_of) => {
                let binding = self.for_head(&for_of.left);
                let iterable = self.expr(&for_of.right);
                let body = self.nested(&for_of.body, &inner);
                format!("{}∀ {} ∈ {} {{\n{}\n{}}}", indent, binding, iterable, body, indent)
            }

            Stmt::While(w) => {
                let cond = self.expr(&w.test);
                let body = self.nested(&w.body, &inner);
                format!("{}⟳ {} {{\n{}\n{}}}", indent, cond, body, indent)
            }

            Stmt::Switch(sw) => self.switch(sw, indent),

            // `try`/`catch`/`finally`. Sigil has no exceptions, so the guarded
            // body and the `finally` both run and the `catch` is recorded
            // rather than invented — running a catch block unconditionally is
            // exactly the reordering S30 was about.
            Stmt::Try(t) => {
                let mut parts = vec![self.block_body(&t.block.stmts, indent)];
                if let Some(handler) = &t.handler {
                    self.warnings.push("catch block not translated".to_string());
                    parts.push(format!(
                        "{}// React: catch {{ {} }} — Sigil has no exceptions; this did not run.",
                        indent,
                        one_line(&slice(self.src, handler.body.span))
                    ));
                }
                if let Some(finalizer) = &t.finalizer {
                    parts.push(self.block_body(&finalizer.stmts, indent));
                }
                parts.retain(|p| !p.trim().is_empty());
                parts.join("\n")
            }

            Stmt::Throw(t) => {
                self.warnings.push("throw not translated".to_string());
                format!(
                    "{}// React: throw {} — Sigil has no exceptions.",
                    indent,
                    one_line(&slice(self.src, t.arg.span()))
                )
            }

            Stmt::Break(_) | Stmt::Continue(_) => {
                self.warnings.push("break/continue not translated".to_string());
                format!("{}// React: break/continue — Sigil has neither.", indent)
            }

            other => {
                self.warnings.push(format!("statement not translated: {:?}", kind_of(other)));
                format!("{}// React: {}", indent, one_line(&slice(self.src, other.span())))
            }
        }
    }

    fn var_decl(&mut self, var: &VarDecl, indent: &str) -> String {
        let mut lines = Vec::new();
        for d in &var.decls {
            let name = match &d.name {
                Pat::Ident(id) => id.id.sym.to_string(),
                other => {
                    // Destructuring has no single name to bind.
                    self.warnings.push("destructuring declaration not translated".to_string());
                    lines.push(format!(
                        "{}// React: {} — destructuring.",
                        indent,
                        one_line(&slice(self.src, other.span()))
                    ));
                    continue;
                }
            };
            let snake = to_snake(&name);
            let value = match &d.init {
                Some(init) => self.expr(init),
                None => "∅".to_string(),
            };
            // `Δ` because a JS `let` is reassigned often enough that declaring
            // it immutable turns a translated body into a compile error.
            let mutable = if var.kind == VarDeclKind::Const { "" } else { "Δ " };
            lines.push(format!("{}≔ {}{} = {};", indent, mutable, snake, value));
            self.locals.push(name);
        }
        lines.join("\n")
    }

    fn for_head(&mut self, left: &ForHead) -> String {
        match left {
            ForHead::VarDecl(var) => match var.decls.first().map(|d| &d.name) {
                Some(Pat::Ident(id)) => {
                    let name = id.id.sym.to_string();
                    let snake = to_snake(&name);
                    self.locals.push(name);
                    snake
                }
                _ => {
                    self.warnings.push("for-of binding not translated".to_string());
                    "_".to_string()
                }
            },
            ForHead::Pat(p) => match &**p {
                Pat::Ident(id) => {
                    let name = id.id.sym.to_string();
                    let snake = to_snake(&name);
                    self.locals.push(name);
                    snake
                }
                _ => {
                    self.warnings.push("for-of binding not translated".to_string());
                    "_".to_string()
                }
            },
            _ => {
                self.warnings.push("for-of head not translated".to_string());
                "_".to_string()
            }
        }
    }

    fn switch(&mut self, sw: &SwitchStmt, indent: &str) -> String {
        let disc = self.expr(&sw.discriminant);
        let inner = format!("{}    ", indent);
        let body_indent = format!("{}    ", inner);
        let mut arms = Vec::new();
        for case in &sw.cases {
            let pattern = match &case.test {
                Some(t) => self.expr(t),
                None => "_".to_string(),
            };
            let body = self.block_body(&case.cons, &body_indent);
            arms.push(format!("{}{} => {{\n{}\n{}}}", inner, pattern, body, inner));
        }
        // A `⌥` has to be exhaustive; JS `switch` without a `default` is not.
        if !sw.cases.iter().any(|c| c.test.is_none()) {
            arms.push(format!("{}_ => {{ }}", inner));
        }
        format!("{}⌥ {} {{\n{}\n{}}}", indent, disc, arms.join("\n"), indent)
    }
}

fn kind_of(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::For(_) => "for",
        Stmt::ForIn(_) => "for-in",
        Stmt::DoWhile(_) => "do-while",
        Stmt::Labeled(_) => "labeled",
        Stmt::With(_) => "with",
        Stmt::Debugger(_) => "debugger",
        Stmt::Decl(_) => "declaration",
        _ => "statement",
    }
}

fn to_snake(name: &str) -> String {
    super::ast_transform::to_snake_case(name)
}

// The original source, sliced by span. A fresh `SourceMap` numbers the file
// from `BytePos(1)`, so a span's `lo`/`hi` are offsets into `code` plus one.
pub(crate) fn slice(src: &str, span: swc_common::Span) -> String {
    let lo = (span.lo.0 as usize).saturating_sub(1);
    let hi = (span.hi.0 as usize).saturating_sub(1);
    if lo <= hi && hi <= src.len() && src.is_char_boundary(lo) && src.is_char_boundary(hi) {
        src[lo..hi].to_string()
    } else {
        "<source unavailable>".to_string()
    }
}

/// The braced body of a function, given its whole source.
///
/// Extraction keeps a helper's full source (`function f(a) { … }`, `const f =
/// (a) => { … }`), and what the statement transform needs is the block. Parsing
/// is the only way to find it that a nested brace or a brace inside a string
/// does not break.
pub fn body_of_function(source: &str) -> Option<String> {
    let src = source.trim();
    let cm: Lrc<SourceMap> = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    let fm = cm.new_source_file(FileName::Anon.into(), src.to_string());
    let mut parser = Parser::new(
        Syntax::Typescript(TsSyntax { tsx: true, ..Default::default() }),
        StringInput::from(&*fm),
        None,
    );
    let module = parser.parse_module().ok()?;

    fn from_expr(e: &Expr, src: &str) -> Option<String> {
        match e {
            Expr::Arrow(a) => match &*a.body {
                BlockStmtOrExpr::BlockStmt(b) => Some(slice(src, b.span)),
                BlockStmtOrExpr::Expr(inner) => Some(slice(src, inner.span())),
            },
            Expr::Fn(f) => f.function.body.as_ref().map(|b| slice(src, b.span)),
            _ => None,
        }
    }

    for item in &module.body {
        match item {
            ModuleItem::Stmt(Stmt::Decl(Decl::Fn(f))) => {
                if let Some(b) = &f.function.body {
                    return Some(slice(src, b.span));
                }
            }
            ModuleItem::Stmt(Stmt::Decl(Decl::Var(v))) => {
                if let Some(init) = v.decls.first().and_then(|d| d.init.as_ref()) {
                    if let Some(b) = from_expr(init, src) {
                        return Some(b);
                    }
                }
            }
            ModuleItem::Stmt(Stmt::Expr(e)) => {
                if let Some(b) = from_expr(&e.expr, src) {
                    return Some(b);
                }
            }
            ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(d)) => match &d.decl {
                Decl::Fn(f) => {
                    if let Some(b) = &f.function.body {
                        return Some(slice(src, b.span));
                    }
                }
                Decl::Var(v) => {
                    if let Some(init) = v.decls.first().and_then(|d| d.init.as_ref()) {
                        if let Some(b) = from_expr(init, src) {
                            return Some(b);
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    None
}

/// The single parameter name of an arrow or function callback, if it has one.
fn callback_param(cb: &Expr) -> Option<String> {
    let pats: Vec<Pat> = match cb {
        Expr::Arrow(a) => a.params.clone(),
        Expr::Fn(f) => f.function.params.iter().map(|p| p.pat.clone()).collect(),
        _ => return None,
    };
    match pats.first() {
        Some(Pat::Ident(id)) => Some(super::ast_transform::to_snake_case(&id.id.sym)),
        _ => None,
    }
}

/// The body of an arrow or function callback.
fn callback_body(cb: &Expr) -> Option<BlockStmtOrExpr> {
    match cb {
        Expr::Arrow(a) => Some((*a.body).clone()),
        Expr::Fn(f) => f
            .function
            .body
            .clone()
            .map(|b| BlockStmtOrExpr::BlockStmt(b)),
        _ => None,
    }
}

/// The callback a `useCallback(fn, deps)` or `useMemo(fn, deps)` wraps.
///
/// Returns its parameter names (snake-cased) and its body source. `useCallback`
/// yields a function, so the local becomes a Sigil closure; `useMemo` yields a
/// value, so only a concise arrow — whose body IS the value — is taken.
/// Everything else stays bound to `∅` with the React beside it, as before.
pub struct HookCallback {
    pub params: Vec<String>,
    /// The body source: a braced block, or the expression of a concise arrow.
    pub body: String,
    pub is_block: bool,
}

pub fn hook_callback(source: &str, hooks: &[&str]) -> Option<HookCallback> {
    let src = source.trim();
    let cm: Lrc<SourceMap> = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    let fm = cm.new_source_file(FileName::Anon.into(), src.to_string());
    let mut parser = Parser::new(
        Syntax::Typescript(TsSyntax { tsx: true, ..Default::default() }),
        StringInput::from(&*fm),
        None,
    );
    let module = parser.parse_module().ok()?;

    let expr = module.body.iter().find_map(|item| match item {
        ModuleItem::Stmt(Stmt::Expr(e)) => Some(&*e.expr),
        _ => None,
    })?;
    let Expr::Call(call) = expr else { return None };
    let Callee::Expr(callee) = &call.callee else { return None };
    let Expr::Ident(name) = callee.as_ref() else { return None };
    if !hooks.contains(&name.sym.as_ref()) {
        return None;
    }
    let cb = call.args.first()?;
    if cb.spread.is_some() {
        return None;
    }
    match &*cb.expr {
        Expr::Arrow(a) => {
            let mut params = Vec::new();
            for p in &a.params {
                match p {
                    Pat::Ident(id) => params.push(to_snake(&id.id.sym)),
                    // A destructured parameter has no single name to bind.
                    _ => return None,
                }
            }
            match &*a.body {
                BlockStmtOrExpr::BlockStmt(b) => Some(HookCallback {
                    params,
                    body: slice(src, b.span),
                    is_block: true,
                }),
                BlockStmtOrExpr::Expr(inner) => Some(HookCallback {
                    params,
                    body: slice(src, inner.span()),
                    is_block: false,
                }),
            }
        }
        _ => None,
    }
}

/// A local bound directly to a function: `const flush = (text) => { … }`.
///
/// The extractor unwraps `useCallback(fn, deps)` to `fn`, so this is what most
/// component-level callbacks actually look like by the time the generator sees
/// them — `hook_callback` alone matched almost none of them.
pub fn function_local(source: &str) -> Option<HookCallback> {
    let src = source.trim();
    let cm: Lrc<SourceMap> = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    let fm = cm.new_source_file(FileName::Anon.into(), src.to_string());
    let mut parser = Parser::new(
        Syntax::Typescript(TsSyntax { tsx: true, ..Default::default() }),
        StringInput::from(&*fm),
        None,
    );
    let module = parser.parse_module().ok()?;
    let expr = module.body.iter().find_map(|item| match item {
        ModuleItem::Stmt(Stmt::Expr(e)) => Some(&*e.expr),
        _ => None,
    })?;
    let Expr::Arrow(a) = expr else { return None };
    let mut params = Vec::new();
    for p in &a.params {
        match p {
            Pat::Ident(id) => params.push(to_snake(&id.id.sym)),
            _ => return None,
        }
    }
    match &*a.body {
        BlockStmtOrExpr::BlockStmt(b) => {
            Some(HookCallback { params, body: slice(src, b.span), is_block: true })
        }
        BlockStmtOrExpr::Expr(inner) => {
            Some(HookCallback { params, body: slice(src, inner.span()), is_block: false })
        }
    }
}

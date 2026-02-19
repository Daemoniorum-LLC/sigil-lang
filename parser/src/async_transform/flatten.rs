//! Await Expression Flattening
//!
//! This pass transforms complex expressions containing await points into sequences
//! of simple let-bindings. It runs BEFORE the state machine transformation, ensuring
//! the state machine only needs to handle simple `≔ x = expr⌛` patterns.
//!
//! ## Transformation Examples
//!
//! ```sigil
//! // Input: await nested in binary expression
//! ≔ x = foo() + bar()⌛
//!
//! // Output: hoisted to simple form
//! ≔ __pre_0 = foo()        // Saved before await
//! ≔ __await_0 = bar()⌛    // Suspension point
//! ≔ x = __pre_0 + __await_0
//! ```
//!
//! See: docs/specs/AWAIT-EXPRESSION-FLATTENING-SPEC.md

use crate::ast::*;
use crate::span::Span;

/// Error type for flattening validation
#[derive(Debug, Clone)]
pub struct FlattenError {
    pub message: String,
    pub hint: Option<String>,
    pub span: Span,
}

impl FlattenError {
    pub fn await_in_closure(span: Span) -> Self {
        Self {
            message: "await inside closure requires `async ||` syntax".to_string(),
            hint: Some(
                "Use `async || { ... }` for an async closure, \
                 or extract the await outside: `≔ x = expr⌛; || x`"
                    .to_string(),
            ),
            span,
        }
    }
}

/// Result type for flattening operations
pub type FlattenResult<T> = Result<T, FlattenError>;

/// Context for the flattening pass
#[derive(Debug, Default)]
pub struct FlattenContext {
    /// Counter for await temporaries (__await_N)
    await_counter: u32,
    /// Counter for pre-await temporaries (__pre_N)
    pre_counter: u32,
    // NOTE: The spec (§4.3) reserves __sc_N for short-circuit temporaries, but the
    // current implementation transforms short-circuit operators directly to if/else
    // without needing intermediate bindings. The counter is omitted since it's unused.
    // If future optimizations require short-circuit temporaries (e.g., for CSE or
    // complex condition caching), this counter can be added back.
}

impl FlattenContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a fresh await temporary name
    pub fn fresh_await_temp(&mut self) -> String {
        let n = self.await_counter;
        self.await_counter += 1;
        format!("__await_{}", n)
    }

    /// Generate a fresh pre-await temporary name
    pub fn fresh_pre_temp(&mut self) -> String {
        let n = self.pre_counter;
        self.pre_counter += 1;
        format!("__pre_{}", n)
    }
}

// ============================================
// Helper Functions
// ============================================

/// Create an Ident with default span and no markers
fn make_ident(name: String) -> Ident {
    Ident {
        name,
        evidentiality: None,
        affect: None,
        span: Span::default(),
    }
}

/// Create a Path expression from a simple identifier
fn make_path_expr(name: String) -> Expr {
    Expr::Path(TypePath {
        segments: vec![PathSegment {
            ident: make_ident(name),
            generics: None,
        }],
    })
}

/// Create a simple identifier pattern
fn make_ident_pattern(name: String) -> Pattern {
    Pattern::Ident {
        mutable: false,
        name: make_ident(name),
        evidentiality: None,
    }
}

/// Create a block with statements and optional trailing expression
fn make_block(stmts: Vec<Stmt>, expr: Option<Expr>) -> Block {
    Block {
        stmts,
        expr: expr.map(Box::new),
    }
}

/// Create a block with just a trailing expression
fn make_block_with_expr(expr: Expr) -> Block {
    Block {
        stmts: vec![],
        expr: Some(Box::new(expr)),
    }
}

// ============================================
// Await Detection
// ============================================

/// Check if an expression contains an await
pub fn contains_await(expr: &Expr) -> bool {
    match expr {
        Expr::Await { .. } => true,
        Expr::Literal(_) => false,
        Expr::Path(_) => false,
        Expr::Binary { left, right, .. } => contains_await(left) || contains_await(right),
        Expr::Unary { expr, .. } => contains_await(expr),
        Expr::Pipe { expr, operations } => {
            contains_await(expr) || operations.iter().any(pipe_op_contains_await)
        }
        Expr::Call { func, args } => {
            contains_await(func) || args.iter().any(contains_await)
        }
        Expr::MethodCall { receiver, args, .. } => {
            contains_await(receiver) || args.iter().any(contains_await)
        }
        Expr::Field { expr, .. } => contains_await(expr),
        Expr::Index { expr, index } => contains_await(expr) || contains_await(index),
        Expr::Array(elements) => elements.iter().any(contains_await),
        Expr::ArrayRepeat { value, count } => contains_await(value) || contains_await(count),
        Expr::Tuple(elements) => elements.iter().any(contains_await),
        Expr::Struct { fields, rest, .. } => {
            fields.iter().any(|f| f.value.as_ref().map_or(false, contains_await))
                || rest.as_ref().map_or(false, |r| contains_await(r))
        }
        Expr::Block(block) => block_contains_await(block),
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            contains_await(condition)
                || block_contains_await(then_branch)
                || else_branch.as_ref().map_or(false, |e| contains_await(e))
        }
        Expr::Match { expr, arms } => {
            contains_await(expr)
                || arms.iter().any(|arm| {
                    arm.guard.as_ref().map_or(false, contains_await) || contains_await(&arm.body)
                })
        }
        Expr::Loop { body, .. } => block_contains_await(body),
        Expr::While { condition, body, .. } => {
            contains_await(condition) || block_contains_await(body)
        }
        Expr::For { iter, body, .. } => contains_await(iter) || block_contains_await(body),
        Expr::Closure { body, .. } => contains_await(body),
        Expr::Try(inner) => contains_await(inner),
        Expr::Return(opt) => opt.as_ref().map_or(false, |e| contains_await(e)),
        Expr::Break { value, .. } => value.as_ref().map_or(false, |e| contains_await(e)),
        Expr::Continue { .. } => false,
        Expr::Range { start, end, .. } => {
            start.as_ref().map_or(false, |e| contains_await(e))
                || end.as_ref().map_or(false, |e| contains_await(e))
        }
        Expr::Macro { .. } => false,
        Expr::Evidential { expr, .. } => contains_await(expr),
        Expr::Attributed { expr, .. } => contains_await(expr),
        Expr::Assign { target, value } => contains_await(target) || contains_await(value),
        Expr::Let { value, .. } => contains_await(value),
        Expr::Unsafe(block) => block_contains_await(block),
        Expr::Async { block, .. } => block_contains_await(block),
        Expr::NoGrad(block) => block_contains_await(block),
        Expr::Deref(inner) => contains_await(inner),
        Expr::AddrOf { expr, .. } => contains_await(expr),
        Expr::Cast { expr, .. } => contains_await(expr),
        Expr::Turbofish { expr, .. } => contains_await(expr),
        Expr::InlineAsm(asm) => {
            asm.inputs.iter().any(|op| contains_await(&op.expr))
                || asm.outputs.iter().any(|op| contains_await(&op.expr))
        }
        Expr::VolatileRead { ptr, .. } => contains_await(ptr),
        Expr::VolatileWrite { ptr, value, .. } => contains_await(ptr) || contains_await(value),
        Expr::SimdLiteral { elements, .. } => elements.iter().any(contains_await),
        Expr::SimdIntrinsic { args, .. } => args.iter().any(contains_await),
        Expr::SimdShuffle { a, b, .. } => contains_await(a) || contains_await(b),
        Expr::SimdSplat { value, .. } => contains_await(value),
        Expr::SimdExtract { vector, .. } => contains_await(vector),
        Expr::SimdInsert { vector, value, .. } => contains_await(vector) || contains_await(value),
        Expr::AtomicOp {
            ptr,
            value,
            expected,
            ..
        } => {
            contains_await(ptr)
                || value.as_ref().map_or(false, |v| contains_await(v))
                || expected.as_ref().map_or(false, |e| contains_await(e))
        }
        Expr::AtomicFence { .. } => false,
        Expr::HttpRequest {
            url,
            headers,
            body,
            timeout,
            ..
        } => {
            contains_await(url)
                || headers.iter().any(|(k, v)| contains_await(k) || contains_await(v))
                || body.as_ref().map_or(false, |b| contains_await(b))
                || timeout.as_ref().map_or(false, |t| contains_await(t))
        }
        Expr::GrpcCall {
            service,
            method,
            message,
            metadata,
            timeout,
        } => {
            contains_await(service)
                || contains_await(method)
                || message.as_ref().map_or(false, |m| contains_await(m))
                || metadata.iter().any(|(k, v)| contains_await(k) || contains_await(v))
                || timeout.as_ref().map_or(false, |t| contains_await(t))
        }
        Expr::WebSocketConnect {
            url,
            protocols,
            headers,
        } => {
            contains_await(url)
                || protocols.iter().any(contains_await)
                || headers.iter().any(|(k, v)| contains_await(k) || contains_await(v))
        }
        Expr::Incorporation { segments } => segments
            .iter()
            .any(|seg| seg.args.as_ref().map_or(false, |a| a.iter().any(contains_await))),
        Expr::Morpheme { body, .. } => contains_await(body),

        // Protocol expressions
        Expr::WebSocketMessage { data, .. } => contains_await(data),
        Expr::KafkaOp {
            topic,
            payload,
            key,
            partition,
            ..
        } => {
            contains_await(topic)
                || payload.as_ref().map_or(false, |p| contains_await(p))
                || key.as_ref().map_or(false, |k| contains_await(k))
                || partition.as_ref().map_or(false, |p| contains_await(p))
        }
        Expr::GraphQLOp {
            document,
            variables,
            operation_name,
            ..
        } => {
            contains_await(document)
                || variables.as_ref().map_or(false, |v| contains_await(v))
                || operation_name.as_ref().map_or(false, |n| contains_await(n))
        }
        Expr::ProtocolStream { source, config, .. } => {
            contains_await(source) || config.as_ref().map_or(false, |c| contains_await(c))
        }

        // Legion expressions
        Expr::LegionFieldVar { .. } => false,
        Expr::LegionSuperposition { field, pattern } => {
            contains_await(field) || contains_await(pattern)
        }
        Expr::LegionInterference { query, field } => {
            contains_await(query) || contains_await(field)
        }
        Expr::LegionResonance { expr } => contains_await(expr),
        Expr::LegionDistribute { task, count } => contains_await(task) || contains_await(count),
        Expr::LegionGather { fragments } => contains_await(fragments),
        Expr::LegionBroadcast { signal, target } => {
            contains_await(signal) || contains_await(target)
        }
        Expr::LegionConsensus { contributions } => contains_await(contributions),
        Expr::LegionDecay { field, rate } => contains_await(field) || contains_await(rate),

        // Named argument
        Expr::NamedArg { value, .. } => contains_await(value),
    }
}

/// Check if a block contains an await
fn block_contains_await(block: &Block) -> bool {
    block.stmts.iter().any(stmt_contains_await)
        || block.expr.as_ref().map_or(false, |e| contains_await(e))
}

/// Check if a statement contains an await
fn stmt_contains_await(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Let { init, .. } => init.as_ref().map_or(false, contains_await),
        Stmt::LetElse { init, else_branch, .. } => {
            contains_await(init) || contains_await(else_branch)
        }
        Stmt::Expr(expr) | Stmt::Semi(expr) => contains_await(expr),
        Stmt::Item(_) => false,
    }
}

/// Check if a pipe operation contains an await
fn pipe_op_contains_await(op: &PipeOp) -> bool {
    match op {
        PipeOp::Await => true,
        PipeOp::Transform(expr)
        | PipeOp::Filter(expr)
        | PipeOp::SortBy(expr)
        | PipeOp::Reduce(expr)
        | PipeOp::Nth(expr)
        | PipeOp::Call(expr)
        | PipeOp::Send(expr)
        | PipeOp::Stream(expr)
        | PipeOp::Body(expr)
        | PipeOp::Timeout(expr)
        | PipeOp::Also(expr)
        | PipeOp::Apply(expr)
        | PipeOp::TakeIf(expr)
        | PipeOp::TakeUnless(expr)
        | PipeOp::Let(expr)
        | PipeOp::All(expr)
        | PipeOp::Any(expr)
        | PipeOp::Compose(expr)
        | PipeOp::Zip(expr)
        | PipeOp::Scan(expr)
        | PipeOp::Gradient(expr)
        | PipeOp::Cycle(expr)
        | PipeOp::Windows(expr)
        | PipeOp::Chunks(expr) => contains_await(expr),
        PipeOp::ReduceWithInit(init, acc) => contains_await(init) || contains_await(acc),
        PipeOp::Method { args, .. } => args.iter().any(contains_await),
        PipeOp::Match(arms) => arms.iter().any(|arm| {
            arm.guard.as_ref().map_or(false, contains_await) || contains_await(&arm.body)
        }),
        PipeOp::TryMap(opt) => opt.as_ref().map_or(false, |e| contains_await(e)),
        PipeOp::Named { body, .. } => body.as_ref().map_or(false, |e| contains_await(e)),
        PipeOp::Validate { predicate, .. } => contains_await(predicate),
        PipeOp::Assume { reason, .. } => reason.as_ref().map_or(false, |r| contains_await(r)),
        PipeOp::Parallel(inner) | PipeOp::Gpu(inner) => pipe_op_contains_await(inner),
        PipeOp::Header { name, value } => contains_await(name) || contains_await(value),
        PipeOp::Retry { count, strategy } => {
            contains_await(count) || strategy.as_ref().map_or(false, |s| contains_await(s))
        }
        PipeOp::Connect(opt) => opt.as_ref().map_or(false, |e| contains_await(e)),
        PipeOp::Possibility { args, .. } | PipeOp::Necessity { args, .. } => {
            args.iter().any(contains_await)
        }
        // Simple ops with no subexpressions
        PipeOp::Sort(_)
        | PipeOp::ReduceSum
        | PipeOp::ReduceProd
        | PipeOp::ReduceMin
        | PipeOp::ReduceMax
        | PipeOp::ReduceConcat
        | PipeOp::ReduceAll
        | PipeOp::ReduceAny
        | PipeOp::Middle
        | PipeOp::Choice
        | PipeOp::Next
        | PipeOp::First
        | PipeOp::Last
        | PipeOp::Recv
        | PipeOp::Close
        | PipeOp::AssertEvidence(_)
        | PipeOp::PossibilityExtract
        | PipeOp::NecessityVerify
        | PipeOp::Diff
        | PipeOp::SortAsc
        | PipeOp::SortDesc
        | PipeOp::Reverse
        | PipeOp::Flatten
        | PipeOp::Unique
        | PipeOp::Enumerate
        | PipeOp::Universal => false,
    }
}

// ============================================
// Simple Expression Check
// ============================================

/// Check if an expression is "simple" (doesn't need saving across await)
///
/// Simple expressions are:
/// - Identifiers (already variables)
/// - Literals (can be re-evaluated cheaply)
/// - Paths (constants)
pub fn is_simple(expr: &Expr) -> bool {
    match expr {
        Expr::Path(_) => true,
        Expr::Literal(_) => true,
        _ => false,
    }
}

// ============================================
// Closure Validation
// ============================================

/// Validate that no await appears inside non-async closures
pub fn validate_no_await_in_closure(expr: &Expr) -> FlattenResult<()> {
    match expr {
        // Closures in Sigil don't have is_async flag - they're sync by default
        // Async closures would be `async || { ... }` which parses as Async { block }
        Expr::Closure { body, .. } => {
            // Check if body contains await
            if contains_await(body) {
                return Err(FlattenError::await_in_closure(Span::default()));
            }
            // Recursively check the body (but the above already catches it)
            validate_no_await_in_closure(body)
        }
        // Async blocks are fine - they can contain await
        Expr::Async { .. } => Ok(()),
        // Recursively check all other expressions
        Expr::Binary { left, right, .. } => {
            validate_no_await_in_closure(left)?;
            validate_no_await_in_closure(right)
        }
        Expr::Unary { expr, .. } => validate_no_await_in_closure(expr),
        Expr::Call { func, args } => {
            validate_no_await_in_closure(func)?;
            for arg in args {
                validate_no_await_in_closure(arg)?;
            }
            Ok(())
        }
        Expr::MethodCall { receiver, args, .. } => {
            validate_no_await_in_closure(receiver)?;
            for arg in args {
                validate_no_await_in_closure(arg)?;
            }
            Ok(())
        }
        Expr::Field { expr, .. } => validate_no_await_in_closure(expr),
        Expr::Index { expr, index } => {
            validate_no_await_in_closure(expr)?;
            validate_no_await_in_closure(index)
        }
        Expr::Array(elements) => {
            for e in elements {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::Tuple(elements) => {
            for e in elements {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::Struct { fields, rest, .. } => {
            for f in fields {
                if let Some(ref v) = f.value {
                    validate_no_await_in_closure(v)?;
                }
            }
            if let Some(r) = rest {
                validate_no_await_in_closure(r)?;
            }
            Ok(())
        }
        Expr::Block(block) => validate_block_no_await_in_closure(block),
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_no_await_in_closure(condition)?;
            validate_block_no_await_in_closure(then_branch)?;
            if let Some(e) = else_branch {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::Match { expr, arms } => {
            validate_no_await_in_closure(expr)?;
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    validate_no_await_in_closure(guard)?;
                }
                validate_no_await_in_closure(&arm.body)?;
            }
            Ok(())
        }
        Expr::Loop { body, .. } => validate_block_no_await_in_closure(body),
        Expr::While { condition, body, .. } => {
            validate_no_await_in_closure(condition)?;
            validate_block_no_await_in_closure(body)
        }
        Expr::For { iter, body, .. } => {
            validate_no_await_in_closure(iter)?;
            validate_block_no_await_in_closure(body)
        }
        Expr::Await { expr, .. } => validate_no_await_in_closure(expr),
        Expr::Try(inner) => validate_no_await_in_closure(inner),
        Expr::Return(opt) => {
            if let Some(e) = opt {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::Break { value, .. } => {
            if let Some(e) = value {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::Pipe { expr, operations } => {
            validate_no_await_in_closure(expr)?;
            for op in operations {
                validate_pipe_op_no_await_in_closure(op)?;
            }
            Ok(())
        }
        // Simple expressions - no subexpressions to check
        Expr::Literal(_)
        | Expr::Path(_)
        | Expr::Continue { .. }
        | Expr::Macro { .. }
        | Expr::AtomicFence { .. } => Ok(()),
        // Other expressions with subexpressions
        Expr::Range { start, end, .. } => {
            if let Some(s) = start {
                validate_no_await_in_closure(s)?;
            }
            if let Some(e) = end {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::Evidential { expr, .. } => validate_no_await_in_closure(expr),
        Expr::Attributed { expr, .. } => validate_no_await_in_closure(expr),
        Expr::Assign { target, value } => {
            validate_no_await_in_closure(target)?;
            validate_no_await_in_closure(value)
        }
        Expr::Let { value, .. } => validate_no_await_in_closure(value),
        Expr::Unsafe(block) => validate_block_no_await_in_closure(block),
        Expr::NoGrad(block) => validate_block_no_await_in_closure(block),
        Expr::Deref(inner) => validate_no_await_in_closure(inner),
        Expr::AddrOf { expr, .. } => validate_no_await_in_closure(expr),
        Expr::Cast { expr, .. } => validate_no_await_in_closure(expr),
        Expr::Turbofish { expr, .. } => validate_no_await_in_closure(expr),
        Expr::ArrayRepeat { value, count } => {
            validate_no_await_in_closure(value)?;
            validate_no_await_in_closure(count)
        }
        Expr::InlineAsm(asm) => {
            for op in &asm.inputs {
                validate_no_await_in_closure(&op.expr)?;
            }
            for op in &asm.outputs {
                validate_no_await_in_closure(&op.expr)?;
            }
            Ok(())
        }
        Expr::VolatileRead { ptr, .. } => validate_no_await_in_closure(ptr),
        Expr::VolatileWrite { ptr, value, .. } => {
            validate_no_await_in_closure(ptr)?;
            validate_no_await_in_closure(value)
        }
        Expr::SimdLiteral { elements, .. } => {
            for e in elements {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::SimdIntrinsic { args, .. } => {
            for a in args {
                validate_no_await_in_closure(a)?;
            }
            Ok(())
        }
        Expr::SimdShuffle { a, b, .. } => {
            validate_no_await_in_closure(a)?;
            validate_no_await_in_closure(b)
        }
        Expr::SimdSplat { value, .. } => validate_no_await_in_closure(value),
        Expr::SimdExtract { vector, .. } => validate_no_await_in_closure(vector),
        Expr::SimdInsert { vector, value, .. } => {
            validate_no_await_in_closure(vector)?;
            validate_no_await_in_closure(value)
        }
        Expr::AtomicOp {
            ptr,
            value,
            expected,
            ..
        } => {
            validate_no_await_in_closure(ptr)?;
            if let Some(v) = value {
                validate_no_await_in_closure(v)?;
            }
            if let Some(e) = expected {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Expr::HttpRequest {
            url,
            headers,
            body,
            timeout,
            ..
        } => {
            validate_no_await_in_closure(url)?;
            for (k, v) in headers {
                validate_no_await_in_closure(k)?;
                validate_no_await_in_closure(v)?;
            }
            if let Some(b) = body {
                validate_no_await_in_closure(b)?;
            }
            if let Some(t) = timeout {
                validate_no_await_in_closure(t)?;
            }
            Ok(())
        }
        Expr::GrpcCall {
            service,
            method,
            message,
            metadata,
            timeout,
        } => {
            validate_no_await_in_closure(service)?;
            validate_no_await_in_closure(method)?;
            if let Some(m) = message {
                validate_no_await_in_closure(m)?;
            }
            for (k, v) in metadata {
                validate_no_await_in_closure(k)?;
                validate_no_await_in_closure(v)?;
            }
            if let Some(t) = timeout {
                validate_no_await_in_closure(t)?;
            }
            Ok(())
        }
        Expr::WebSocketConnect {
            url,
            protocols,
            headers,
        } => {
            validate_no_await_in_closure(url)?;
            for p in protocols {
                validate_no_await_in_closure(p)?;
            }
            for (k, v) in headers {
                validate_no_await_in_closure(k)?;
                validate_no_await_in_closure(v)?;
            }
            Ok(())
        }
        Expr::Incorporation { segments } => {
            for seg in segments {
                if let Some(args) = &seg.args {
                    for a in args {
                        validate_no_await_in_closure(a)?;
                    }
                }
            }
            Ok(())
        }
        Expr::Morpheme { body, .. } => validate_no_await_in_closure(body),

        // Protocol expressions
        Expr::WebSocketMessage { data, .. } => validate_no_await_in_closure(data),
        Expr::KafkaOp {
            topic,
            payload,
            key,
            partition,
            ..
        } => {
            validate_no_await_in_closure(topic)?;
            if let Some(p) = payload {
                validate_no_await_in_closure(p)?;
            }
            if let Some(k) = key {
                validate_no_await_in_closure(k)?;
            }
            if let Some(p) = partition {
                validate_no_await_in_closure(p)?;
            }
            Ok(())
        }
        Expr::GraphQLOp {
            document,
            variables,
            operation_name,
            ..
        } => {
            validate_no_await_in_closure(document)?;
            if let Some(v) = variables {
                validate_no_await_in_closure(v)?;
            }
            if let Some(n) = operation_name {
                validate_no_await_in_closure(n)?;
            }
            Ok(())
        }
        Expr::ProtocolStream { source, config, .. } => {
            validate_no_await_in_closure(source)?;
            if let Some(c) = config {
                validate_no_await_in_closure(c)?;
            }
            Ok(())
        }

        // Legion expressions
        Expr::LegionFieldVar { .. } => Ok(()),
        Expr::LegionSuperposition { field, pattern } => {
            validate_no_await_in_closure(field)?;
            validate_no_await_in_closure(pattern)
        }
        Expr::LegionInterference { query, field } => {
            validate_no_await_in_closure(query)?;
            validate_no_await_in_closure(field)
        }
        Expr::LegionResonance { expr } => validate_no_await_in_closure(expr),
        Expr::LegionDistribute { task, count } => {
            validate_no_await_in_closure(task)?;
            validate_no_await_in_closure(count)
        }
        Expr::LegionGather { fragments } => validate_no_await_in_closure(fragments),
        Expr::LegionBroadcast { signal, target } => {
            validate_no_await_in_closure(signal)?;
            validate_no_await_in_closure(target)
        }
        Expr::LegionConsensus { contributions } => validate_no_await_in_closure(contributions),
        Expr::LegionDecay { field, rate } => {
            validate_no_await_in_closure(field)?;
            validate_no_await_in_closure(rate)
        }

        // Named argument
        Expr::NamedArg { value, .. } => validate_no_await_in_closure(value),
    }
}

fn validate_block_no_await_in_closure(block: &Block) -> FlattenResult<()> {
    for stmt in &block.stmts {
        validate_stmt_no_await_in_closure(stmt)?;
    }
    if let Some(expr) = &block.expr {
        validate_no_await_in_closure(expr)?;
    }
    Ok(())
}

fn validate_stmt_no_await_in_closure(stmt: &Stmt) -> FlattenResult<()> {
    match stmt {
        Stmt::Let { init, .. } => {
            if let Some(e) = init {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        Stmt::LetElse { init, else_branch, .. } => {
            validate_no_await_in_closure(init)?;
            validate_no_await_in_closure(else_branch)
        }
        Stmt::Expr(e) | Stmt::Semi(e) => validate_no_await_in_closure(e),
        Stmt::Item(_) => Ok(()),
    }
}

fn validate_pipe_op_no_await_in_closure(op: &PipeOp) -> FlattenResult<()> {
    match op {
        PipeOp::Transform(e)
        | PipeOp::Filter(e)
        | PipeOp::SortBy(e)
        | PipeOp::Reduce(e)
        | PipeOp::Nth(e)
        | PipeOp::Call(e)
        | PipeOp::Send(e)
        | PipeOp::Stream(e)
        | PipeOp::Body(e)
        | PipeOp::Timeout(e)
        | PipeOp::Also(e)
        | PipeOp::Apply(e)
        | PipeOp::TakeIf(e)
        | PipeOp::TakeUnless(e)
        | PipeOp::Let(e)
        | PipeOp::All(e)
        | PipeOp::Any(e)
        | PipeOp::Compose(e)
        | PipeOp::Zip(e)
        | PipeOp::Scan(e)
        | PipeOp::Gradient(e)
        | PipeOp::Cycle(e)
        | PipeOp::Windows(e)
        | PipeOp::Chunks(e) => validate_no_await_in_closure(e),
        PipeOp::ReduceWithInit(init, acc) => {
            validate_no_await_in_closure(init)?;
            validate_no_await_in_closure(acc)
        }
        PipeOp::Method { args, .. } => {
            for a in args {
                validate_no_await_in_closure(a)?;
            }
            Ok(())
        }
        PipeOp::Match(arms) => {
            for arm in arms {
                if let Some(g) = &arm.guard {
                    validate_no_await_in_closure(g)?;
                }
                validate_no_await_in_closure(&arm.body)?;
            }
            Ok(())
        }
        PipeOp::TryMap(opt) => {
            if let Some(e) = opt {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        PipeOp::Named { body, .. } => {
            if let Some(e) = body {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        PipeOp::Validate { predicate, .. } => validate_no_await_in_closure(predicate),
        PipeOp::Assume { reason, .. } => {
            if let Some(r) = reason {
                validate_no_await_in_closure(r)?;
            }
            Ok(())
        }
        PipeOp::Parallel(inner) | PipeOp::Gpu(inner) => validate_pipe_op_no_await_in_closure(inner),
        PipeOp::Header { name, value } => {
            validate_no_await_in_closure(name)?;
            validate_no_await_in_closure(value)
        }
        PipeOp::Retry { count, strategy } => {
            validate_no_await_in_closure(count)?;
            if let Some(s) = strategy {
                validate_no_await_in_closure(s)?;
            }
            Ok(())
        }
        PipeOp::Connect(opt) => {
            if let Some(e) = opt {
                validate_no_await_in_closure(e)?;
            }
            Ok(())
        }
        PipeOp::Possibility { args, .. } | PipeOp::Necessity { args, .. } => {
            for a in args {
                validate_no_await_in_closure(a)?;
            }
            Ok(())
        }
        // Simple ops
        _ => Ok(()),
    }
}

// ============================================
// Expression Flattening
// ============================================

/// Result of flattening an expression
pub struct FlattenedExpr {
    /// Statements to prepend (hoisted let bindings)
    pub hoisted: Vec<Stmt>,
    /// The transformed expression
    pub expr: Expr,
}

impl FlattenedExpr {
    fn simple(expr: Expr) -> Self {
        Self {
            hoisted: vec![],
            expr,
        }
    }

    fn with_hoisted(hoisted: Vec<Stmt>, expr: Expr) -> Self {
        Self { hoisted, expr }
    }
}

/// Flatten an expression, hoisting nested awaits to preceding let-bindings
pub fn flatten_expr(ctx: &mut FlattenContext, expr: Expr) -> FlattenResult<FlattenedExpr> {
    // If no await, return unchanged
    if !contains_await(&expr) {
        return Ok(FlattenedExpr::simple(expr));
    }

    match expr {
        // Direct await - flatten inner if needed
        Expr::Await { expr: inner, evidentiality } => {
            let flattened_inner = flatten_expr(ctx, *inner)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened_inner.hoisted,
                Expr::Await {
                    expr: Box::new(flattened_inner.expr),
                    evidentiality,
                },
            ))
        }

        // Binary expression
        Expr::Binary { left, op, right } => flatten_binary(ctx, *left, op, *right),

        // Call expression
        Expr::Call { func, args } => flatten_call(ctx, *func, args),

        // Method call
        Expr::MethodCall {
            receiver,
            method,
            type_args,
            args,
        } => flatten_method_call(ctx, *receiver, method, type_args, args),

        // Unary expression
        Expr::Unary { op, expr: inner } => {
            let flattened = flatten_expr(ctx, *inner)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::Unary {
                    op,
                    expr: Box::new(flattened.expr),
                },
            ))
        }

        // Index expression
        Expr::Index { expr: arr, index } => flatten_index(ctx, *arr, *index),

        // Field access
        Expr::Field { expr: inner, field } => {
            let flattened = flatten_expr(ctx, *inner)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::Field {
                    expr: Box::new(flattened.expr),
                    field,
                },
            ))
        }

        // Match - flatten scrutinee
        Expr::Match { expr: scrutinee, arms } => {
            let flattened_scrutinee = flatten_expr(ctx, *scrutinee)?;
            // Note: we don't flatten inside match arms - that's handled by state machine
            Ok(FlattenedExpr::with_hoisted(
                flattened_scrutinee.hoisted,
                Expr::Match {
                    expr: Box::new(flattened_scrutinee.expr),
                    arms,
                },
            ))
        }

        // If expression - flatten condition
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let flattened_cond = flatten_expr(ctx, *condition)?;
            // Note: we don't flatten inside branches - that's handled by state machine
            Ok(FlattenedExpr::with_hoisted(
                flattened_cond.hoisted,
                Expr::If {
                    condition: Box::new(flattened_cond.expr),
                    then_branch,
                    else_branch,
                },
            ))
        }

        // While loop - flatten condition
        Expr::While {
            label,
            condition,
            body,
        } => {
            let flattened_cond = flatten_expr(ctx, *condition)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened_cond.hoisted,
                Expr::While {
                    label,
                    condition: Box::new(flattened_cond.expr),
                    body,
                },
            ))
        }

        // For loop - flatten iterator
        Expr::For {
            label,
            pattern,
            iter,
            body,
        } => {
            let flattened_iter = flatten_expr(ctx, *iter)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened_iter.hoisted,
                Expr::For {
                    label,
                    pattern,
                    iter: Box::new(flattened_iter.expr),
                    body,
                },
            ))
        }

        // Try expression
        Expr::Try(inner) => {
            let flattened = flatten_expr(ctx, *inner)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::Try(Box::new(flattened.expr)),
            ))
        }

        // Array literal
        Expr::Array(elements) => flatten_array(ctx, elements),

        // Tuple literal
        Expr::Tuple(elements) => flatten_tuple(ctx, elements),

        // Struct literal
        Expr::Struct { path, fields, rest } => flatten_struct(ctx, path, fields, rest),

        // Block - flatten statements within
        Expr::Block(block) => {
            let flattened_block = flatten_block(ctx, block)?;
            Ok(FlattenedExpr::simple(Expr::Block(flattened_block)))
        }

        // Pipe expression
        Expr::Pipe { expr: inner, operations } => flatten_pipe(ctx, *inner, operations),

        // Return with value
        Expr::Return(Some(value)) => {
            let flattened = flatten_expr(ctx, *value)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::Return(Some(Box::new(flattened.expr))),
            ))
        }

        // Break with value
        Expr::Break { label, value: Some(v) } => {
            let flattened = flatten_expr(ctx, *v)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::Break {
                    label,
                    value: Some(Box::new(flattened.expr)),
                },
            ))
        }

        // Assign
        Expr::Assign { target, value } => {
            let flattened_target = flatten_expr(ctx, *target)?;
            let flattened_value = flatten_expr(ctx, *value)?;
            let mut hoisted = flattened_target.hoisted;
            hoisted.extend(flattened_value.hoisted);
            Ok(FlattenedExpr::with_hoisted(
                hoisted,
                Expr::Assign {
                    target: Box::new(flattened_target.expr),
                    value: Box::new(flattened_value.expr),
                },
            ))
        }

        // Range
        Expr::Range { start, end, inclusive } => {
            let mut hoisted = vec![];
            let new_start = if let Some(s) = start {
                let f = flatten_expr(ctx, *s)?;
                hoisted.extend(f.hoisted);
                Some(Box::new(f.expr))
            } else {
                None
            };
            let new_end = if let Some(e) = end {
                let f = flatten_expr(ctx, *e)?;
                hoisted.extend(f.hoisted);
                Some(Box::new(f.expr))
            } else {
                None
            };
            Ok(FlattenedExpr::with_hoisted(
                hoisted,
                Expr::Range {
                    start: new_start,
                    end: new_end,
                    inclusive,
                },
            ))
        }

        // Cast
        Expr::Cast { expr: inner, ty } => {
            let flattened = flatten_expr(ctx, *inner)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::Cast {
                    expr: Box::new(flattened.expr),
                    ty,
                },
            ))
        }

        // Deref
        Expr::Deref(inner) => {
            let flattened = flatten_expr(ctx, *inner)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::Deref(Box::new(flattened.expr)),
            ))
        }

        // AddrOf
        Expr::AddrOf { mutable, expr: inner } => {
            let flattened = flatten_expr(ctx, *inner)?;
            Ok(FlattenedExpr::with_hoisted(
                flattened.hoisted,
                Expr::AddrOf {
                    mutable,
                    expr: Box::new(flattened.expr),
                },
            ))
        }

        // Expressions that shouldn't have await in flattening context or
        // where await is handled specially
        other => Ok(FlattenedExpr::simple(other)),
    }
}

/// Flatten a binary expression
fn flatten_binary(
    ctx: &mut FlattenContext,
    left: Expr,
    op: BinOp,
    right: Expr,
) -> FlattenResult<FlattenedExpr> {
    // Special case: short-circuit operators
    if matches!(op, BinOp::Or | BinOp::And) {
        return flatten_short_circuit(ctx, left, op, right);
    }

    let left_has_await = contains_await(&left);
    let right_has_await = contains_await(&right);

    let mut hoisted = vec![];

    // Flatten left side
    let flattened_left = flatten_expr(ctx, left)?;
    hoisted.extend(flattened_left.hoisted);
    let mut left_expr = flattened_left.expr;

    // If right has await, left's value must be saved (unless simple)
    if right_has_await && !is_simple(&left_expr) {
        let temp = ctx.fresh_pre_temp();
        hoisted.push(Stmt::Let {
            pattern: make_ident_pattern(temp.clone()),
            ty: None,
            init: Some(left_expr),
        });
        left_expr = make_path_expr(temp);
    }

    // Flatten right side
    let flattened_right = flatten_expr(ctx, right)?;
    hoisted.extend(flattened_right.hoisted);

    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::Binary {
            left: Box::new(left_expr),
            op,
            right: Box::new(flattened_right.expr),
        },
    ))
}

/// Flatten short-circuit operators (|| and &&)
///
/// These are transformed to if/else to preserve short-circuit semantics:
/// - `a || b⌛` → `if a { true } else { b⌛ }`
/// - `a && b⌛` → `if a { b⌛ } else { false }`
fn flatten_short_circuit(
    ctx: &mut FlattenContext,
    left: Expr,
    op: BinOp,
    right: Expr,
) -> FlattenResult<FlattenedExpr> {
    let left_has_await = contains_await(&left);
    let right_has_await = contains_await(&right);

    // If neither side has await, no transformation needed
    if !left_has_await && !right_has_await {
        return Ok(FlattenedExpr::simple(Expr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }));
    }

    let mut hoisted = vec![];

    // Flatten left side (may contain await)
    let flattened_left = flatten_expr(ctx, left)?;
    hoisted.extend(flattened_left.hoisted);

    // Transform to if/else
    let result = match op {
        BinOp::Or => {
            // a || b → if a { true } else { b }
            Expr::If {
                condition: Box::new(flattened_left.expr),
                then_branch: make_block_with_expr(Expr::Literal(Literal::Bool(true))),
                else_branch: Some(Box::new(right)),
            }
        }
        BinOp::And => {
            // a && b → if a { b } else { false }
            Expr::If {
                condition: Box::new(flattened_left.expr),
                then_branch: make_block_with_expr(right),
                else_branch: Some(Box::new(Expr::Literal(Literal::Bool(false)))),
            }
        }
        _ => unreachable!(),
    };

    Ok(FlattenedExpr::with_hoisted(hoisted, result))
}

/// Flatten a call expression
fn flatten_call(
    ctx: &mut FlattenContext,
    func: Expr,
    args: Vec<Expr>,
) -> FlattenResult<FlattenedExpr> {
    let mut hoisted = vec![];

    // Check if any argument has await
    let any_await = args.iter().any(contains_await);

    // Flatten function expression
    let flattened_func = flatten_expr(ctx, func)?;
    hoisted.extend(flattened_func.hoisted);
    let mut func_expr = flattened_func.expr;

    // If any arg has await, save func if complex
    if any_await && !is_simple(&func_expr) {
        let temp = ctx.fresh_pre_temp();
        hoisted.push(Stmt::Let {
            pattern: make_ident_pattern(temp.clone()),
            ty: None,
            init: Some(func_expr),
        });
        func_expr = make_path_expr(temp);
    }

    // Process arguments left-to-right
    let mut new_args = Vec::with_capacity(args.len());
    for (i, arg) in args.into_iter().enumerate() {
        let flattened_arg = flatten_expr(ctx, arg)?;
        hoisted.extend(flattened_arg.hoisted);
        let mut arg_expr = flattened_arg.expr;

        // If later args have await, save this one (unless simple)
        // Note: we need to check remaining args, but we've already moved them
        // So we use a different approach: check if arg_expr is not simple and
        // the original expression position had await in subsequent positions
        // For simplicity, we save all non-simple args if any arg has await
        if any_await && !is_simple(&arg_expr) {
            let temp = ctx.fresh_pre_temp();
            hoisted.push(Stmt::Let {
                pattern: make_ident_pattern(temp.clone()),
                ty: None,
                init: Some(arg_expr),
            });
            arg_expr = make_path_expr(temp);
        }

        new_args.push(arg_expr);
    }

    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::Call {
            func: Box::new(func_expr),
            args: new_args,
        },
    ))
}

/// Flatten a method call expression
fn flatten_method_call(
    ctx: &mut FlattenContext,
    receiver: Expr,
    method: Ident,
    type_args: Option<Vec<TypeExpr>>,
    args: Vec<Expr>,
) -> FlattenResult<FlattenedExpr> {
    let mut hoisted = vec![];

    // Check if any argument has await
    let any_await = args.iter().any(contains_await);

    // Flatten receiver
    let flattened_recv = flatten_expr(ctx, receiver)?;
    hoisted.extend(flattened_recv.hoisted);
    let mut recv_expr = flattened_recv.expr;

    // If any arg has await, save receiver if complex
    if any_await && !is_simple(&recv_expr) {
        let temp = ctx.fresh_pre_temp();
        hoisted.push(Stmt::Let {
            pattern: make_ident_pattern(temp.clone()),
            ty: None,
            init: Some(recv_expr),
        });
        recv_expr = make_path_expr(temp);
    }

    // Process arguments
    let mut new_args = Vec::with_capacity(args.len());
    for arg in args {
        let flattened_arg = flatten_expr(ctx, arg)?;
        hoisted.extend(flattened_arg.hoisted);
        let mut arg_expr = flattened_arg.expr;

        if any_await && !is_simple(&arg_expr) {
            let temp = ctx.fresh_pre_temp();
            hoisted.push(Stmt::Let {
                pattern: make_ident_pattern(temp.clone()),
                ty: None,
                init: Some(arg_expr),
            });
            arg_expr = make_path_expr(temp);
        }

        new_args.push(arg_expr);
    }

    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::MethodCall {
            receiver: Box::new(recv_expr),
            method,
            type_args,
            args: new_args,
        },
    ))
}

/// Flatten an index expression
fn flatten_index(
    ctx: &mut FlattenContext,
    arr: Expr,
    index: Expr,
) -> FlattenResult<FlattenedExpr> {
    let mut hoisted = vec![];

    let index_has_await = contains_await(&index);

    // Flatten array expression
    let flattened_arr = flatten_expr(ctx, arr)?;
    hoisted.extend(flattened_arr.hoisted);
    let mut arr_expr = flattened_arr.expr;

    // If index has await, save array if complex
    if index_has_await && !is_simple(&arr_expr) {
        let temp = ctx.fresh_pre_temp();
        hoisted.push(Stmt::Let {
            pattern: make_ident_pattern(temp.clone()),
            ty: None,
            init: Some(arr_expr),
        });
        arr_expr = make_path_expr(temp);
    }

    // Flatten index
    let flattened_index = flatten_expr(ctx, index)?;
    hoisted.extend(flattened_index.hoisted);

    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::Index {
            expr: Box::new(arr_expr),
            index: Box::new(flattened_index.expr),
        },
    ))
}

/// Flatten an array literal
fn flatten_array(ctx: &mut FlattenContext, elements: Vec<Expr>) -> FlattenResult<FlattenedExpr> {
    let mut hoisted = vec![];
    let any_await = elements.iter().any(contains_await);

    let mut new_elements = Vec::with_capacity(elements.len());
    for elem in elements {
        let flattened = flatten_expr(ctx, elem)?;
        hoisted.extend(flattened.hoisted);
        let mut elem_expr = flattened.expr;

        if any_await && !is_simple(&elem_expr) {
            let temp = ctx.fresh_pre_temp();
            hoisted.push(Stmt::Let {
                pattern: make_ident_pattern(temp.clone()),
                ty: None,
                init: Some(elem_expr),
            });
            elem_expr = make_path_expr(temp);
        }

        new_elements.push(elem_expr);
    }

    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::Array(new_elements),
    ))
}

/// Flatten a tuple literal
fn flatten_tuple(ctx: &mut FlattenContext, elements: Vec<Expr>) -> FlattenResult<FlattenedExpr> {
    let mut hoisted = vec![];
    let any_await = elements.iter().any(contains_await);

    let mut new_elements = Vec::with_capacity(elements.len());
    for elem in elements {
        let flattened = flatten_expr(ctx, elem)?;
        hoisted.extend(flattened.hoisted);
        let mut elem_expr = flattened.expr;

        if any_await && !is_simple(&elem_expr) {
            let temp = ctx.fresh_pre_temp();
            hoisted.push(Stmt::Let {
                pattern: make_ident_pattern(temp.clone()),
                ty: None,
                init: Some(elem_expr),
            });
            elem_expr = make_path_expr(temp);
        }

        new_elements.push(elem_expr);
    }

    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::Tuple(new_elements),
    ))
}

/// Flatten a struct literal
fn flatten_struct(
    ctx: &mut FlattenContext,
    path: TypePath,
    fields: Vec<FieldInit>,
    rest: Option<Box<Expr>>,
) -> FlattenResult<FlattenedExpr> {
    let mut hoisted = vec![];
    let any_await =
        fields.iter().any(|f| f.value.as_ref().map_or(false, contains_await))
            || rest.as_ref().map_or(false, |r| contains_await(r));

    let mut new_fields = Vec::with_capacity(fields.len());
    for field in fields {
        // FieldInit.value is Option<Expr> - None means shorthand `{ name }` syntax
        let new_value = if let Some(value) = field.value {
            let flattened = flatten_expr(ctx, value)?;
            hoisted.extend(flattened.hoisted);
            let mut value_expr = flattened.expr;

            if any_await && !is_simple(&value_expr) {
                let temp = ctx.fresh_pre_temp();
                hoisted.push(Stmt::Let {
                    pattern: make_ident_pattern(temp.clone()),
                    ty: None,
                    init: Some(value_expr),
                });
                value_expr = make_path_expr(temp);
            }

            Some(value_expr)
        } else {
            None
        };

        new_fields.push(FieldInit {
            name: field.name,
            value: new_value,
        });
    }

    let new_rest = if let Some(r) = rest {
        let flattened = flatten_expr(ctx, *r)?;
        hoisted.extend(flattened.hoisted);
        Some(Box::new(flattened.expr))
    } else {
        None
    };

    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::Struct {
            path,
            fields: new_fields,
            rest: new_rest,
        },
    ))
}

/// Flatten a pipe expression
fn flatten_pipe(
    ctx: &mut FlattenContext,
    expr: Expr,
    operations: Vec<PipeOp>,
) -> FlattenResult<FlattenedExpr> {
    let mut hoisted = vec![];

    // Flatten the initial expression
    let flattened_expr = flatten_expr(ctx, expr)?;
    hoisted.extend(flattened_expr.hoisted);

    // For now, we don't flatten inside pipe operations
    // The state machine handles await in pipes
    Ok(FlattenedExpr::with_hoisted(
        hoisted,
        Expr::Pipe {
            expr: Box::new(flattened_expr.expr),
            operations,
        },
    ))
}

// ============================================
// Statement and Block Flattening
// ============================================

/// Flatten a statement
pub fn flatten_stmt(ctx: &mut FlattenContext, stmt: Stmt) -> FlattenResult<Vec<Stmt>> {
    match stmt {
        Stmt::Let { pattern, ty, init } => {
            if let Some(init_expr) = init {
                if contains_await(&init_expr) {
                    let flattened = flatten_expr(ctx, init_expr)?;
                    let mut result = flattened.hoisted;
                    result.push(Stmt::Let {
                        pattern,
                        ty,
                        init: Some(flattened.expr),
                    });
                    return Ok(result);
                }
                // No await, return unchanged
                Ok(vec![Stmt::Let {
                    pattern,
                    ty,
                    init: Some(init_expr),
                }])
            } else {
                // No init expression
                Ok(vec![Stmt::Let { pattern, ty, init: None }])
            }
        }

        Stmt::LetElse {
            pattern,
            ty,
            init,
            else_branch,
        } => {
            if contains_await(&init) {
                let flattened = flatten_expr(ctx, init)?;
                let mut result = flattened.hoisted;
                result.push(Stmt::LetElse {
                    pattern,
                    ty,
                    init: flattened.expr,
                    else_branch,
                });
                return Ok(result);
            }
            Ok(vec![Stmt::LetElse {
                pattern,
                ty,
                init,
                else_branch,
            }])
        }

        Stmt::Expr(expr) => {
            if contains_await(&expr) {
                let flattened = flatten_expr(ctx, expr)?;
                let mut result = flattened.hoisted;
                result.push(Stmt::Expr(flattened.expr));
                return Ok(result);
            }
            Ok(vec![Stmt::Expr(expr)])
        }

        Stmt::Semi(expr) => {
            if contains_await(&expr) {
                let flattened = flatten_expr(ctx, expr)?;
                let mut result = flattened.hoisted;
                result.push(Stmt::Semi(flattened.expr));
                return Ok(result);
            }
            Ok(vec![Stmt::Semi(expr)])
        }

        Stmt::Item(item) => Ok(vec![Stmt::Item(item)]),
    }
}

/// Flatten a block
pub fn flatten_block(ctx: &mut FlattenContext, block: Block) -> FlattenResult<Block> {
    let mut new_stmts = vec![];

    for stmt in block.stmts {
        let flattened = flatten_stmt(ctx, stmt)?;
        new_stmts.extend(flattened);
    }

    let new_expr = if let Some(expr) = block.expr {
        if contains_await(&expr) {
            let flattened = flatten_expr(ctx, *expr)?;
            new_stmts.extend(flattened.hoisted);
            Some(Box::new(flattened.expr))
        } else {
            Some(expr)
        }
    } else {
        None
    };

    Ok(Block {
        stmts: new_stmts,
        expr: new_expr,
    })
}

/// Flatten a function body
pub fn flatten_function(func: &mut Function) -> FlattenResult<()> {
    // Only process async functions
    if !func.is_async {
        return Ok(());
    }

    // Function.body is Option<Block>
    let Some(ref body) = func.body else {
        return Ok(());
    };

    // Validate no await in closures
    for stmt in &body.stmts {
        match stmt {
            Stmt::Expr(e) | Stmt::Semi(e) => validate_no_await_in_closure(e)?,
            Stmt::Let { init: Some(ref e), .. } => validate_no_await_in_closure(e)?,
            Stmt::LetElse { init, else_branch, .. } => {
                validate_no_await_in_closure(init)?;
                validate_no_await_in_closure(else_branch)?;
            }
            _ => {}
        }
    }
    if let Some(ref expr) = body.expr {
        validate_no_await_in_closure(expr)?;
    }

    // Flatten the body - take ownership of the body
    // SAFETY: func.body is guaranteed to be Some by the guard at line 1694
    let body = func.body.take().expect("func.body guaranteed Some by earlier guard");
    let mut ctx = FlattenContext::new();
    func.body = Some(flatten_block(&mut ctx, body)?);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a simple await expression
    fn make_await(inner: Expr) -> Expr {
        Expr::Await {
            expr: Box::new(inner),
            evidentiality: None,
        }
    }

    // Helper to create a simple call expression
    fn make_call(name: &str) -> Expr {
        Expr::Call {
            func: Box::new(make_path_expr(name.to_string())),
            args: vec![],
        }
    }

    #[test]
    fn test_contains_await_simple() {
        assert!(!contains_await(&Expr::Literal(Literal::Int {
            value: "42".to_string(),
            base: NumBase::Decimal,
            suffix: None,
        })));

        assert!(contains_await(&make_await(make_call("foo"))));
    }

    #[test]
    fn test_contains_await_binary() {
        let expr = Expr::Binary {
            left: Box::new(make_call("foo")),
            op: BinOp::Add,
            right: Box::new(make_await(make_call("bar"))),
        };
        assert!(contains_await(&expr));
    }

    #[test]
    fn test_is_simple() {
        assert!(is_simple(&make_path_expr("x".to_string())));
        assert!(is_simple(&Expr::Literal(Literal::Int {
            value: "42".to_string(),
            base: NumBase::Decimal,
            suffix: None,
        })));
        assert!(!is_simple(&make_call("foo")));
    }

    #[test]
    fn test_flatten_direct_await() {
        let mut ctx = FlattenContext::new();
        let expr = make_await(make_call("foo"));
        let result = flatten_expr(&mut ctx, expr).unwrap();

        assert!(result.hoisted.is_empty());
        assert!(matches!(result.expr, Expr::Await { .. }));
    }

    #[test]
    fn test_flatten_binary_with_await() {
        let mut ctx = FlattenContext::new();
        let expr = Expr::Binary {
            left: Box::new(make_call("foo")),
            op: BinOp::Add,
            right: Box::new(make_await(make_call("bar"))),
        };
        let result = flatten_expr(&mut ctx, expr).unwrap();

        // foo() should be hoisted to __pre_0
        assert_eq!(result.hoisted.len(), 1);
        assert!(matches!(result.expr, Expr::Binary { .. }));
    }

    #[test]
    fn test_flatten_short_circuit_or() {
        let mut ctx = FlattenContext::new();
        let expr = Expr::Binary {
            left: Box::new(make_call("check")),
            op: BinOp::Or,
            right: Box::new(make_await(make_call("fetch"))),
        };
        let result = flatten_expr(&mut ctx, expr).unwrap();

        // Should be transformed to if/else
        assert!(matches!(result.expr, Expr::If { .. }));
    }

    #[test]
    fn test_flatten_short_circuit_and() {
        let mut ctx = FlattenContext::new();
        let expr = Expr::Binary {
            left: Box::new(make_call("validate")),
            op: BinOp::And,
            right: Box::new(make_await(make_call("submit"))),
        };
        let result = flatten_expr(&mut ctx, expr).unwrap();

        // Should be transformed to if/else
        assert!(matches!(result.expr, Expr::If { .. }));
    }

    #[test]
    fn test_await_in_closure_error() {
        let expr = Expr::Closure {
            params: vec![],
            return_type: None,
            body: Box::new(make_await(make_call("foo"))),
            is_move: false,
        };

        let result = validate_no_await_in_closure(&expr);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("async ||"));
    }
}

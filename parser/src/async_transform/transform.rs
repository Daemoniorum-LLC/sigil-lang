//! Async function transformation
//!
//! Transforms async function AST into StateMachineIR.

use crate::ast::{Block, Expr, Function, Pattern, Stmt, TypeExpr};
use super::ir::*;

/// Context for transforming an if expression with awaits.
#[derive(Debug, Clone)]
enum IfContext {
    /// If is the trailing expression (return value of block)
    Trailing,
    /// If is in statement position (result discarded)
    Statement,
    /// If is in let binding position: `let name = if ... {}`
    LetBinding {
        name: Option<String>,
        ty: Option<TypeExpr>,
    },
}

/// Error during async transformation.
#[derive(Debug, Clone)]
pub struct TransformError {
    pub message: String,
    pub kind: TransformErrorKind,
}

#[derive(Debug, Clone)]
pub enum TransformErrorKind {
    /// Function has no body
    NoBody,
    /// Unsupported expression containing await
    UnsupportedAwaitContext,
    /// Internal error
    Internal,
}

impl TransformError {
    pub fn no_body(name: &str) -> Self {
        Self {
            message: format!("Async function '{}' has no body", name),
            kind: TransformErrorKind::NoBody,
        }
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: TransformErrorKind::UnsupportedAwaitContext,
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: TransformErrorKind::Internal,
        }
    }
}

/// Result type for transformation operations.
pub type TransformResult<T> = Result<T, TransformError>;

/// Transform an async function into a state machine IR.
///
/// This is the main entry point for the async transformation.
pub fn transform_async_function(func: &Function) -> TransformResult<StateMachineIR> {
    let transformer = AsyncTransformer::new(func)?;
    transformer.transform()
}

/// Check if a function needs state machine transformation.
///
/// Returns true if the function has multiple await points.
pub fn needs_state_machine(func: &Function) -> bool {
    if !func.is_async {
        return false;
    }

    let Some(body) = &func.body else {
        return false;
    };

    count_awaits_in_block(body) > 1
}

/// Transform an async function with flattening pre-pass.
///
/// This runs the expression flattening pass before state machine transformation,
/// which simplifies complex await expressions into simple let-bindings.
pub fn transform_with_flatten(func: &Function) -> TransformResult<StateMachineIR> {
    use super::flatten::flatten_function;

    // Clone the function so we can mutate it
    let mut flattened = func.clone();

    // Flatten complex await expressions (modifies in place)
    flatten_function(&mut flattened).map_err(|e| {
        TransformError::internal(format!("Flattening failed: {}", e.message))
    })?;

    // Then transform to state machine
    transform_async_function(&flattened)
}

/// Count await points in a block.
fn count_awaits_in_block(block: &Block) -> usize {
    let mut count = 0;
    for stmt in &block.stmts {
        count += count_awaits_in_stmt(stmt);
    }
    if let Some(expr) = &block.expr {
        count += count_awaits_in_expr(expr);
    }
    count
}

/// Count await points in a statement.
fn count_awaits_in_stmt(stmt: &Stmt) -> usize {
    match stmt {
        Stmt::Let { init, .. } => {
            init.as_ref().map_or(0, count_awaits_in_expr)
        }
        Stmt::LetElse { init, else_branch, .. } => {
            count_awaits_in_expr(init) + count_awaits_in_expr(else_branch)
        }
        Stmt::Expr(e) | Stmt::Semi(e) => count_awaits_in_expr(e),
        Stmt::Item(_) => 0,
    }
}

/// Count await points in an expression.
fn count_awaits_in_expr(expr: &Expr) -> usize {
    match expr {
        Expr::Await { .. } => 1,
        Expr::Binary { left, right, .. } => {
            count_awaits_in_expr(left) + count_awaits_in_expr(right)
        }
        Expr::Unary { expr, .. } => count_awaits_in_expr(expr),
        Expr::Call { func, args, .. } => {
            count_awaits_in_expr(func)
                + args.iter().map(count_awaits_in_expr).sum::<usize>()
        }
        Expr::If { condition, then_branch, else_branch, .. } => {
            count_awaits_in_expr(condition)
                + count_awaits_in_block(then_branch)
                + else_branch.as_ref().map_or(0, |e| count_awaits_in_expr(e))
        }
        Expr::Block(block) => count_awaits_in_block(block),
        Expr::Tuple(elements) | Expr::Array(elements) => {
            elements.iter().map(count_awaits_in_expr).sum()
        }
        Expr::Match { expr, arms } => {
            count_awaits_in_expr(expr)
                + arms.iter().map(|arm| count_awaits_in_expr(&arm.body)).sum::<usize>()
        }
        Expr::While { condition, body, .. } => {
            count_awaits_in_expr(condition) + count_awaits_in_block(body)
        }
        Expr::Loop { body, .. } => {
            count_awaits_in_block(body)
        }
        Expr::For { iter, body, .. } => {
            count_awaits_in_expr(iter) + count_awaits_in_block(body)
        }
        Expr::Return(value) => {
            value.as_ref().map_or(0, |e| count_awaits_in_expr(e))
        }
        Expr::Try(inner) => count_awaits_in_expr(inner),
        Expr::Index { expr, index, .. } => {
            count_awaits_in_expr(expr) + count_awaits_in_expr(index)
        }
        Expr::Field { expr, .. } => count_awaits_in_expr(expr),
        Expr::MethodCall { receiver, args, .. } => {
            count_awaits_in_expr(receiver)
                + args.iter().map(count_awaits_in_expr).sum::<usize>()
        }
        _ => 0,
    }
}

/// Extract the name from a pattern if it's a simple identifier.
fn pattern_to_name(pattern: &Pattern) -> Option<String> {
    match pattern {
        Pattern::Ident { name, .. } => Some(name.name.clone()),
        _ => None,
    }
}

/// Context for tracking loop state during transformation.
#[derive(Debug, Clone)]
struct LoopContext {
    /// State index of the loop head (where condition is checked)
    head_state: u32,
    /// State index after the loop exits
    exit_state: u32,
    /// Optional label for labeled break/continue
    label: Option<String>,
    /// Optional binding name for break values (for loops-as-expressions)
    #[allow(dead_code)]
    break_value_binding: Option<String>,
}

/// The async function transformer.
struct AsyncTransformer<'a> {
    func: &'a Function,
    ir: StateMachineIR,
    current_state_idx: u32,
    /// Locals that are currently live (defined and may be used later)
    live_locals: Vec<String>,
    /// Stack of enclosing loops for break/continue handling
    loop_stack: Vec<LoopContext>,
    /// Counter for generating unique synthetic local names
    synthetic_counter: u32,
}

impl<'a> AsyncTransformer<'a> {
    fn new(func: &'a Function) -> TransformResult<Self> {
        if func.body.is_none() {
            return Err(TransformError::no_body(&func.name.name));
        }

        let params: Vec<(String, _)> = func.params.iter()
            .filter_map(|p| {
                pattern_to_name(&p.pattern).map(|name| (name, p.ty.clone()))
            })
            .collect();

        let result_type = func.return_type.clone();

        Ok(Self {
            func,
            ir: StateMachineIR::new(func.name.name.clone(), params, result_type),
            current_state_idx: 0,
            live_locals: Vec::new(),
            loop_stack: Vec::new(),
            synthetic_counter: 0,
        })
    }

    fn transform(mut self) -> TransformResult<StateMachineIR> {
        // Create entry state
        let entry = State::entry();
        self.ir.add_state(entry);
        self.current_state_idx = 0;

        // Add parameters as locals
        for param in &self.func.params {
            if let Some(name) = pattern_to_name(&param.pattern) {
                self.ir.declare_local(name.clone(), Some(param.ty.clone()), 0);
                self.live_locals.push(name);
            }
        }

        // Transform the body
        let body = self.func.body.as_ref().unwrap();
        self.transform_block(body)?;

        // Frame layout is already computed incrementally in declare_local()
        // No need to rebuild it here.

        // Validate before returning
        self.ir.validate().map_err(|errs| {
            TransformError::internal(format!("Validation failed: {:?}", errs))
        })?;

        Ok(self.ir)
    }

    fn transform_block(&mut self, block: &Block) -> TransformResult<()> {
        for stmt in &block.stmts {
            self.transform_stmt(stmt)?;
        }

        // Handle trailing expression
        if let Some(expr) = &block.expr {
            self.transform_trailing_expr(expr)?;
        } else {
            // No trailing expression - ensure current state has an exit
            // This handles cases like `if cond { await } else { await };` with no trailing expr
            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
            if matches!(state.exit, StateExit::Unreachable) {
                state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
            }
        }

        Ok(())
    }

    fn transform_stmt(&mut self, stmt: &Stmt) -> TransformResult<()> {
        match stmt {
            Stmt::Let { pattern, ty, init } => {
                let name = pattern_to_name(pattern);

                if let Some(init_expr) = init {
                    // Check if init contains an await
                    if let Some(await_expr) = extract_direct_await(init_expr) {
                        // This is `let x = expr|await` - create suspension point
                        self.emit_await(await_expr, name.clone())?;

                        // Declare the local in the new state
                        if let Some(n) = name {
                            self.ir.declare_local(n.clone(), ty.clone(), self.current_state_idx);
                            self.live_locals.push(n);
                        }
                    } else if let Expr::If { condition, then_branch, else_branch } = init_expr {
                        // Check if the if expression has awaits
                        let then_awaits = count_awaits_in_block(then_branch);
                        let else_awaits = else_branch.as_ref().map_or(0, |e| count_awaits_in_expr(e));

                        if then_awaits > 0 || else_awaits > 0 {
                            // Transform if expression with binding
                            self.transform_if_expr(
                                condition,
                                then_branch,
                                else_branch.as_deref(),
                                IfContext::LetBinding { name: name.clone(), ty: ty.clone() },
                            )?;

                            // The binding is already declared by transform_if_expr
                            if let Some(n) = &name {
                                self.live_locals.push(n.clone());
                            }
                        } else {
                            // No await in if, add as-is
                            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                            state.body.push(stmt.clone());

                            if let Some(n) = name {
                                self.ir.declare_local(n.clone(), ty.clone(), self.current_state_idx);
                                self.live_locals.push(n);
                            }
                        }
                    } else if count_awaits_in_expr(init_expr) > 0 {
                        // Await nested in other expression - not yet supported
                        return Err(TransformError::unsupported(
                            "Await nested in expression not yet supported"
                        ));
                    } else {
                        // No await, just add to current state's body
                        let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                        state.body.push(stmt.clone());

                        if let Some(n) = name {
                            self.ir.declare_local(n.clone(), ty.clone(), self.current_state_idx);
                            self.live_locals.push(n);
                        }
                    }
                } else {
                    // No init, just declare
                    let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                    state.body.push(stmt.clone());

                    if let Some(n) = name {
                        self.ir.declare_local(n.clone(), ty.clone(), self.current_state_idx);
                        self.live_locals.push(n);
                    }
                }
            }

            Stmt::Expr(expr) | Stmt::Semi(expr) => {
                if let Expr::If { condition, then_branch, else_branch } = expr {
                    let then_awaits = count_awaits_in_block(then_branch);
                    let else_awaits = else_branch.as_ref().map_or(0, |e| count_awaits_in_expr(e));

                    if then_awaits > 0 || else_awaits > 0 {
                        // If in statement position with awaits
                        self.transform_if_expr(
                            condition,
                            then_branch,
                            else_branch.as_deref(),
                            IfContext::Statement,
                        )?;
                        return Ok(());
                    }
                }

                // Handle while loops with awaits
                if let Expr::While { label, condition, body } = expr {
                    if count_awaits_in_block(body) > 0 {
                        self.transform_while_expr(label.as_ref(), condition, body)?;
                        return Ok(());
                    }
                }

                // Handle infinite loops with awaits
                if let Expr::Loop { label, body } = expr {
                    if count_awaits_in_block(body) > 0 {
                        self.transform_loop_expr(label.as_ref(), body)?;
                        return Ok(());
                    }
                }

                // For loops with await - not yet supported
                if let Expr::For { body, .. } = expr {
                    if count_awaits_in_block(body) > 0 {
                        return Err(TransformError::unsupported(
                            "For loops with await not yet supported (requires iterator desugaring)"
                        ));
                    }
                }

                if count_awaits_in_expr(expr) > 0 {
                    // Expression with await
                    if let Some(await_expr) = extract_direct_await(expr) {
                        // Await in statement position (result discarded)
                        self.emit_await(await_expr, None)?;
                    } else {
                        return Err(TransformError::unsupported(
                            "Await nested in expression statement not yet supported"
                        ));
                    }
                } else {
                    let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                    state.body.push(stmt.clone());
                }
            }

            Stmt::LetElse { init, else_branch, .. } => {
                // Check for await in let-else (not yet supported)
                if count_awaits_in_expr(init) > 0 || count_awaits_in_expr(else_branch) > 0 {
                    return Err(TransformError::unsupported(
                        "Await in let-else expression not yet supported"
                    ));
                }
                // No await, just add to body
                let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                state.body.push(stmt.clone());
            }

            Stmt::Item(_) => {
                // Items are compile-time, just add to body
                let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                state.body.push(stmt.clone());
            }
        }

        Ok(())
    }

    /// Well-known name for the resume value in trailing await position.
    const RESUME_VALUE_BINDING: &'static str = "__resume_value";

    fn transform_trailing_expr(&mut self, expr: &Expr) -> TransformResult<()> {
        // Handle return expressions
        if let Expr::Return(value) = expr {
            let return_value = value.as_ref()
                .map(|e| (**e).clone())
                .unwrap_or(Expr::Tuple(vec![]));

            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
            state.exit = StateExit::Return { value: return_value };
            return Ok(());
        }

        // Handle if expression with awaits in trailing position
        if let Expr::If { condition, then_branch, else_branch } = expr {
            let then_awaits = count_awaits_in_block(then_branch);
            let else_awaits = else_branch.as_ref().map_or(0, |e| count_awaits_in_expr(e));

            if then_awaits > 0 || else_awaits > 0 {
                // Transform if as trailing expression (each branch returns)
                return self.transform_if_expr(
                    condition,
                    then_branch,
                    else_branch.as_deref(),
                    IfContext::Trailing,
                );
            }
        }

        // Handle while loop in trailing position
        if let Expr::While { label, condition, body } = expr {
            if count_awaits_in_block(body) > 0 {
                self.transform_while_expr(label.as_ref(), condition, body)?;
                // After while, set return for the exit state
                let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
                return Ok(());
            }
        }

        // Handle loop in trailing position
        if let Expr::Loop { label, body } = expr {
            if count_awaits_in_block(body) > 0 {
                self.transform_loop_expr(label.as_ref(), body)?;
                // After loop, the exit state should have a return
                // Note: for infinite loops without break, exit state may be unreachable
                let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                if matches!(state.exit, StateExit::Unreachable) {
                    state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
                }
                return Ok(());
            }
        }

        // Check for await in trailing expr
        if let Some(await_expr) = extract_direct_await(expr) {
            // Await as the final expression - bind result to well-known name
            let binding_name = Self::RESUME_VALUE_BINDING.to_string();
            self.emit_await(await_expr, Some(binding_name.clone()))?;

            // Declare the local for the resume value
            self.ir.declare_local(binding_name.clone(), None, self.current_state_idx);

            // The resume state returns the bound value
            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
            state.exit = StateExit::Return {
                value: Expr::Path(crate::ast::TypePath {
                    segments: vec![crate::ast::PathSegment {
                        ident: crate::ast::Ident {
                            name: binding_name,
                            evidentiality: None,
                            affect: None,
                            span: crate::span::Span::default(),
                        },
                        generics: None,
                    }],
                }),
            };
            return Ok(());
        }

        // No await, set as return value
        let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
        state.exit = StateExit::Return { value: expr.clone() };

        Ok(())
    }

    /// Transform an if expression that contains await points.
    ///
    /// This creates a Branch exit in the current state, then transforms
    /// each branch into its own sequence of states.
    fn transform_if_expr(
        &mut self,
        condition: &Expr,
        then_branch: &Block,
        else_branch: Option<&Expr>,
        context: IfContext,
    ) -> TransformResult<()> {
        // Check for await in condition (not yet supported)
        if count_awaits_in_expr(condition) > 0 {
            return Err(TransformError::unsupported(
                "Await in if condition not yet supported"
            ));
        }

        // Save current context
        let saved_live_locals = self.live_locals.clone();
        let branch_state_idx = self.current_state_idx;

        // Allocate state indices for then and else branches
        let then_state_idx = self.ir.next_state_idx();

        // Create placeholder then state
        let then_state = State::intermediate(then_state_idx);
        self.ir.add_state(then_state);

        // For else branch, we need to know its state index
        let else_state_idx = self.ir.next_state_idx();

        // Create placeholder else state
        let else_state = State::intermediate(else_state_idx);
        self.ir.add_state(else_state);

        // Set the branch exit
        {
            let state = self.ir.get_state_mut(branch_state_idx).unwrap();
            state.exit = StateExit::Branch {
                condition: condition.clone(),
                then_state: then_state_idx,
                else_state: else_state_idx,
            };
        }

        // Determine if we need a join state (for LetBinding and Statement contexts)
        let needs_join = matches!(context, IfContext::LetBinding { .. } | IfContext::Statement);
        let binding_name = match &context {
            IfContext::LetBinding { name, .. } => name.clone(),
            _ => None,
        };
        let binding_ty = match &context {
            IfContext::LetBinding { ty, .. } => ty.clone(),
            _ => None,
        };

        // ========================================
        // Transform THEN branch
        // ========================================
        self.current_state_idx = then_state_idx;
        self.live_locals = saved_live_locals.clone();

        // Transform the then block
        self.transform_branch_block(then_branch, &context, binding_name.clone(), binding_ty.clone())?;

        let then_end_state = self.current_state_idx;
        let then_ended_with_exit = {
            let state = self.ir.get_state(then_end_state).unwrap();
            !matches!(state.exit, StateExit::Unreachable)
        };

        // ========================================
        // Transform ELSE branch
        // ========================================
        self.current_state_idx = else_state_idx;
        self.live_locals = saved_live_locals.clone();

        if let Some(else_expr) = else_branch {
            // else_expr is typically Expr::Block for else { ... }
            match else_expr {
                Expr::Block(block) => {
                    self.transform_branch_block(block, &context, binding_name.clone(), binding_ty.clone())?;
                }
                Expr::If { condition: cond, then_branch: tb, else_branch: eb } => {
                    // else if ... - recursive
                    self.transform_if_expr(cond, tb, eb.as_deref(), context.clone())?;
                }
                _ => {
                    // Single expression else
                    self.transform_trailing_expr(else_expr)?;
                }
            }
        } else {
            // No else branch - return unit
            let state = self.ir.get_state_mut(else_state_idx).unwrap();
            match &context {
                IfContext::Trailing => {
                    state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
                }
                IfContext::Statement | IfContext::LetBinding { .. } => {
                    // Will be patched to Goto join state below
                    state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
                }
            }
        }

        let else_end_state = self.current_state_idx;
        let else_ended_with_exit = {
            let state = self.ir.get_state(else_end_state).unwrap();
            !matches!(state.exit, StateExit::Unreachable)
        };

        // ========================================
        // Create JOIN state if needed
        // ========================================
        if needs_join {
            let join_state_idx = self.ir.next_state_idx();
            let mut join_state = State::intermediate(join_state_idx);
            join_state.exit = StateExit::Unreachable; // Will be set by caller or set to Return
            self.ir.add_state(join_state);

            // Patch then branch to goto join (if it didn't already exit)
            if then_ended_with_exit {
                let then_state = self.ir.get_state_mut(then_end_state).unwrap();
                // Check if the exit is Return - convert to Goto join
                if let StateExit::Return { .. } = &then_state.exit {
                    then_state.exit = StateExit::Goto { target: join_state_idx };
                }
            }

            // Patch else branch to goto join (if it didn't already exit)
            if else_ended_with_exit {
                let else_state = self.ir.get_state_mut(else_end_state).unwrap();
                if let StateExit::Return { .. } = &else_state.exit {
                    else_state.exit = StateExit::Goto { target: join_state_idx };
                }
            }

            // Note: Local binding is declared in each branch state where it's assigned,
            // not in the join state. This handles the case where different branches
            // may have different declaration points (e.g., await vs non-await).
            // The binding is already added to live_locals by transform_if_expr's caller.

            // Update current state to join
            self.current_state_idx = join_state_idx;
            self.live_locals = saved_live_locals;
        } else {
            // Trailing context - branches already have Return exits
            // Restore to arbitrary state (caller should be done)
            self.live_locals = saved_live_locals;
        }

        Ok(())
    }

    /// Transform a block within a branch of an if expression.
    fn transform_branch_block(
        &mut self,
        block: &Block,
        context: &IfContext,
        binding: Option<String>,
        binding_ty: Option<TypeExpr>,
    ) -> TransformResult<()> {
        // Transform statements in the block
        for stmt in &block.stmts {
            self.transform_stmt(stmt)?;
        }

        // Handle trailing expression based on context
        if let Some(expr) = &block.expr {
            match context {
                IfContext::Trailing => {
                    // Branch should return its value
                    self.transform_trailing_expr(expr)?;
                }
                IfContext::Statement => {
                    // Result discarded - just execute and return unit
                    if count_awaits_in_expr(expr) > 0 {
                        if let Some(await_expr) = extract_direct_await(expr) {
                            self.emit_await(await_expr, None)?;
                        } else if let Expr::If { condition, then_branch, else_branch } = expr.as_ref() {
                            self.transform_if_expr(condition, then_branch, else_branch.as_deref(), IfContext::Statement)?;
                        } else {
                            return Err(TransformError::unsupported(
                                "Await nested in expression not yet supported"
                            ));
                        }
                    }
                    let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                    state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
                }
                IfContext::LetBinding { name, .. } => {
                    // Branch value is bound to `name`
                    if let Some(await_expr) = extract_direct_await(expr) {
                        // Await as branch result - bind to the let name
                        self.emit_await(await_expr, binding.clone())?;

                        // Declare the local in the resume state (where it's first assigned)
                        if let Some(ref bind_name) = binding {
                            self.ir.declare_local_if_new(
                                bind_name.clone(),
                                binding_ty.clone(),
                                self.current_state_idx,
                            );
                        }

                        let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                        // The value is in the binding via resume_binding, just goto join
                        state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
                    } else if count_awaits_in_expr(expr) > 0 {
                        if let Expr::If { condition, then_branch, else_branch } = expr.as_ref() {
                            self.transform_if_expr(
                                condition,
                                then_branch,
                                else_branch.as_deref(),
                                IfContext::LetBinding { name: name.clone(), ty: binding_ty.clone() },
                            )?;
                        } else {
                            return Err(TransformError::unsupported(
                                "Await nested in expression not yet supported"
                            ));
                        }
                    } else {
                        // No await - create a let statement to assign the value
                        if let Some(ref bind_name) = binding {
                            // Create a synthetic let statement: let <name> = <expr>;
                            let let_stmt = Stmt::Let {
                                pattern: Pattern::Ident {
                                    mutable: false,
                                    name: crate::ast::Ident {
                                        name: bind_name.clone(),
                                        evidentiality: None,
                                        affect: None,
                                        span: crate::span::Span::default(),
                                    },
                                    evidentiality: None,
                                },
                                ty: binding_ty.clone(),
                                init: Some(expr.as_ref().clone()),
                            };

                            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                            state.body.push(let_stmt);

                            // Declare the local in this state
                            self.ir.declare_local_if_new(
                                bind_name.clone(),
                                binding_ty.clone(),
                                self.current_state_idx,
                            );
                        }

                        let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
                        state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
                    }
                }
            }
        } else {
            // No trailing expression - return unit
            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
            state.exit = StateExit::Return { value: Expr::Tuple(vec![]) };
        }

        Ok(())
    }

    /// Emit an await point, creating a new state for the continuation.
    fn emit_await(&mut self, promise: &Expr, binding: Option<String>) -> TransformResult<()> {
        let next_state_idx = self.ir.next_state_idx();

        // Compute locals to save (all currently live locals)
        let saved_locals = self.live_locals.clone();

        // Set current state's exit to Await
        {
            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
            state.exit = StateExit::Await {
                promise: promise.clone(),
                next_state: next_state_idx,
                saved_locals,
            };
        }

        // Create the resume state
        let mut resume_state = State::resume(next_state_idx);
        resume_state.resume_binding = binding;
        self.ir.add_state(resume_state);

        // Update current state
        self.current_state_idx = next_state_idx;

        Ok(())
    }

    /// Transform a while loop with await points in its body.
    ///
    /// Creates a LoopHead state that checks the condition, with transitions
    /// to the body or to the exit state.
    fn transform_while_expr(
        &mut self,
        label: Option<&crate::ast::Ident>,
        condition: &Expr,
        body: &Block,
    ) -> TransformResult<()> {
        // Check for await in condition (not yet supported)
        if count_awaits_in_expr(condition) > 0 {
            return Err(TransformError::unsupported(
                "Await in while condition not yet supported"
            ));
        }

        // Save current state for the Goto to loop head
        let pre_loop_state_idx = self.current_state_idx;

        // Create loop head state
        let loop_head_idx = self.ir.next_state_idx();
        let loop_head = State::intermediate(loop_head_idx);
        self.ir.add_state(loop_head);

        // Create body start state
        let body_state_idx = self.ir.next_state_idx();
        let body_state = State::intermediate(body_state_idx);
        self.ir.add_state(body_state);

        // Create exit state (after the loop)
        let exit_state_idx = self.ir.next_state_idx();
        let exit_state = State::intermediate(exit_state_idx);
        self.ir.add_state(exit_state);

        // Set pre-loop state to Goto loop head
        {
            let state = self.ir.get_state_mut(pre_loop_state_idx).unwrap();
            state.exit = StateExit::Goto { target: loop_head_idx };
        }

        // Set loop head's exit to LoopHead
        {
            let state = self.ir.get_state_mut(loop_head_idx).unwrap();
            state.exit = StateExit::LoopHead {
                condition: Some(condition.clone()),
                body_state: body_state_idx,
                exit_state: exit_state_idx,
            };
        }

        // Push loop context for break/continue handling
        self.loop_stack.push(LoopContext {
            head_state: loop_head_idx,
            exit_state: exit_state_idx,
            label: label.map(|l| l.name.clone()),
            break_value_binding: None,
        });

        // Transform the loop body
        self.current_state_idx = body_state_idx;
        let saved_live_locals = self.live_locals.clone();

        self.transform_loop_body(body, loop_head_idx)?;

        // Pop loop context
        self.loop_stack.pop();

        // Restore live locals and continue after the loop
        self.live_locals = saved_live_locals;
        self.current_state_idx = exit_state_idx;

        Ok(())
    }

    /// Transform an infinite loop (loop { ... }) with await points.
    fn transform_loop_expr(
        &mut self,
        label: Option<&crate::ast::Ident>,
        body: &Block,
    ) -> TransformResult<()> {
        // Save current state for the Goto to loop head
        let pre_loop_state_idx = self.current_state_idx;

        // Create loop head state
        let loop_head_idx = self.ir.next_state_idx();
        let loop_head = State::intermediate(loop_head_idx);
        self.ir.add_state(loop_head);

        // Create body start state
        let body_state_idx = self.ir.next_state_idx();
        let body_state = State::intermediate(body_state_idx);
        self.ir.add_state(body_state);

        // Create exit state (after the loop - reachable via break)
        let exit_state_idx = self.ir.next_state_idx();
        let exit_state = State::intermediate(exit_state_idx);
        self.ir.add_state(exit_state);

        // Set pre-loop state to Goto loop head
        {
            let state = self.ir.get_state_mut(pre_loop_state_idx).unwrap();
            state.exit = StateExit::Goto { target: loop_head_idx };
        }

        // Set loop head's exit to LoopHead with no condition (infinite loop)
        {
            let state = self.ir.get_state_mut(loop_head_idx).unwrap();
            state.exit = StateExit::LoopHead {
                condition: None,
                body_state: body_state_idx,
                exit_state: exit_state_idx,
            };
        }

        // Push loop context for break/continue handling
        self.loop_stack.push(LoopContext {
            head_state: loop_head_idx,
            exit_state: exit_state_idx,
            label: label.map(|l| l.name.clone()),
            break_value_binding: None,
        });

        // Transform the loop body
        self.current_state_idx = body_state_idx;
        let saved_live_locals = self.live_locals.clone();

        self.transform_loop_body(body, loop_head_idx)?;

        // Pop loop context
        self.loop_stack.pop();

        // Restore live locals and continue after the loop
        self.live_locals = saved_live_locals;
        self.current_state_idx = exit_state_idx;

        Ok(())
    }

    /// Transform the body of a loop, ensuring it returns to the loop head.
    fn transform_loop_body(&mut self, body: &Block, loop_head_idx: u32) -> TransformResult<()> {
        // Transform statements in the body
        for stmt in &body.stmts {
            self.transform_stmt(stmt)?;
        }

        // Handle trailing expression
        if let Some(expr) = &body.expr {
            // Check for break expression
            if let Expr::Break { label, value } = expr.as_ref() {
                return self.handle_break(label.as_ref(), value.as_deref());
            }

            // Check for continue expression
            if let Expr::Continue { label } = expr.as_ref() {
                return self.handle_continue(label.as_ref());
            }

            // Other trailing expression - evaluate and loop back
            if count_awaits_in_expr(expr) > 0 {
                if let Some(await_expr) = extract_direct_await(expr) {
                    self.emit_await(await_expr, None)?;
                } else {
                    return Err(TransformError::unsupported(
                        "Await nested in loop trailing expression not yet supported"
                    ));
                }
            }
        }

        // End of loop body - go back to loop head
        let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
        if matches!(state.exit, StateExit::Unreachable) {
            state.exit = StateExit::Goto { target: loop_head_idx };
        }

        Ok(())
    }

    /// Handle a break expression.
    fn handle_break(&mut self, label: Option<&crate::ast::Ident>, value: Option<&Expr>) -> TransformResult<()> {
        // Find the target loop
        let loop_ctx = if let Some(label) = label {
            self.loop_stack.iter().rev()
                .find(|ctx| ctx.label.as_ref() == Some(&label.name))
                .ok_or_else(|| TransformError::unsupported(
                    format!("No loop found with label '{}'", label.name)
                ))?
        } else {
            self.loop_stack.last()
                .ok_or_else(|| TransformError::unsupported("break outside of loop"))?
        };

        let exit_state = loop_ctx.exit_state;

        // Handle break with value by creating a synthetic local
        if let Some(value_expr) = value {
            // Generate a unique synthetic local name for the break value
            let synthetic_name = format!("__break_value_{}", self.synthetic_counter);
            self.synthetic_counter += 1;

            // Declare the synthetic local
            self.ir.declare_local(synthetic_name.clone(), None, self.current_state_idx);
            self.live_locals.push(synthetic_name.clone());

            // Create a let statement to assign the value
            let let_stmt = Stmt::Let {
                pattern: Pattern::Ident {
                    mutable: false,
                    name: crate::ast::Ident {
                        name: synthetic_name.clone(),
                        evidentiality: None,
                        affect: None,
                        span: crate::span::Span::default(),
                    },
                    evidentiality: None,
                },
                ty: None,
                init: Some(value_expr.clone()),
            };

            // Add the let statement to the current state
            let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
            state.body.push(let_stmt);
        }

        // Set current state to Goto exit
        let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
        state.exit = StateExit::Goto { target: exit_state };

        Ok(())
    }

    /// Handle a continue expression.
    fn handle_continue(&mut self, label: Option<&crate::ast::Ident>) -> TransformResult<()> {
        // Find the target loop
        let loop_ctx = if let Some(label) = label {
            self.loop_stack.iter().rev()
                .find(|ctx| ctx.label.as_ref() == Some(&label.name))
                .ok_or_else(|| TransformError::unsupported(
                    format!("No loop found with label '{}'", label.name)
                ))?
        } else {
            self.loop_stack.last()
                .ok_or_else(|| TransformError::unsupported("continue outside of loop"))?
        };

        let head_state = loop_ctx.head_state;

        // Set current state to Goto loop head
        let state = self.ir.get_state_mut(self.current_state_idx).unwrap();
        state.exit = StateExit::Goto { target: head_state };

        Ok(())
    }
}

/// Extract a direct await expression (i.e., `expr|await` or `expr⌛`).
fn extract_direct_await(expr: &Expr) -> Option<&Expr> {
    match expr {
        Expr::Await { expr: inner, .. } => Some(inner.as_ref()),
        _ => None,
    }
}

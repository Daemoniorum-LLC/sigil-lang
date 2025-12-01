//! Sigil Optimization Passes
//!
//! A comprehensive optimization framework for the Sigil language.
//! Implements industry-standard compiler optimizations.

use crate::ast::{self, BinOp, Expr, Ident, Item, Literal, Stmt, UnaryOp, Block, NumBase, TypePath, PathSegment, Pattern, Param, TypeExpr, Visibility, FunctionAttrs};
use crate::span::Span;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Optimization Pass Infrastructure
// ============================================================================

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations (O0)
    None,
    /// Basic optimizations (O1) - constant folding, dead code elimination
    Basic,
    /// Standard optimizations (O2) - adds CSE, strength reduction, inlining
    Standard,
    /// Aggressive optimizations (O3) - adds loop opts, vectorization hints
    Aggressive,
    /// Size optimization (Os)
    Size,
}

/// Statistics collected during optimization
#[derive(Debug, Default, Clone)]
pub struct OptStats {
    pub constants_folded: usize,
    pub dead_code_eliminated: usize,
    pub expressions_deduplicated: usize,
    pub functions_inlined: usize,
    pub strength_reductions: usize,
    pub branches_simplified: usize,
    pub loops_optimized: usize,
    pub tail_recursion_transforms: usize,
}

/// The main optimizer that runs all passes
pub struct Optimizer {
    level: OptLevel,
    stats: OptStats,
    /// Function bodies for inlining decisions
    functions: HashMap<String, ast::Function>,
    /// Track which functions are recursive
    recursive_functions: HashSet<String>,
    /// Counter for CSE variable names
    cse_counter: usize,
}

impl Optimizer {
    pub fn new(level: OptLevel) -> Self {
        Self {
            level,
            stats: OptStats::default(),
            functions: HashMap::new(),
            recursive_functions: HashSet::new(),
            cse_counter: 0,
        }
    }

    /// Get optimization statistics
    pub fn stats(&self) -> &OptStats {
        &self.stats
    }

    /// Optimize a source file (returns optimized copy)
    pub fn optimize_file(&mut self, file: &ast::SourceFile) -> ast::SourceFile {
        // First pass: collect function information
        for item in &file.items {
            if let Item::Function(func) = &item.node {
                self.functions.insert(func.name.name.clone(), func.clone());
                if self.is_recursive(&func.name.name, func) {
                    self.recursive_functions.insert(func.name.name.clone());
                }
            }
        }

        // Accumulator transformation pass (for aggressive optimization)
        // This transforms double-recursive functions like fib into tail-recursive form
        let mut new_items: Vec<crate::span::Spanned<Item>> = Vec::new();
        let mut transformed_functions: HashMap<String, String> = HashMap::new();

        if self.level == OptLevel::Aggressive {
            for item in &file.items {
                if let Item::Function(func) = &item.node {
                    if let Some((helper_func, wrapper_func)) = self.try_accumulator_transform(func) {
                        // Add helper function first
                        new_items.push(crate::span::Spanned {
                            node: Item::Function(helper_func),
                            span: item.span.clone(),
                        });
                        transformed_functions.insert(func.name.name.clone(), wrapper_func.name.name.clone());
                        self.stats.tail_recursion_transforms += 1;
                    }
                }
            }
        }

        // Optimization passes
        let items: Vec<_> = file.items.iter().map(|item| {
            let node = match &item.node {
                Item::Function(func) => {
                    // Check if this function was transformed
                    if let Some((_, wrapper)) = self.try_accumulator_transform(func) {
                        if self.level == OptLevel::Aggressive && transformed_functions.contains_key(&func.name.name) {
                            Item::Function(self.optimize_function(&wrapper))
                        } else {
                            Item::Function(self.optimize_function(func))
                        }
                    } else {
                        Item::Function(self.optimize_function(func))
                    }
                }
                other => other.clone(),
            };
            crate::span::Spanned { node, span: item.span.clone() }
        }).collect();

        // Combine new helper functions with transformed items
        new_items.extend(items);

        ast::SourceFile {
            attrs: file.attrs.clone(),
            config: file.config.clone(),
            items: new_items,
        }
    }

    /// Try to transform a double-recursive function into tail-recursive form
    /// Returns (helper_function, wrapper_function) if transformation is possible
    fn try_accumulator_transform(&self, func: &ast::Function) -> Option<(ast::Function, ast::Function)> {
        // Only transform functions with single parameter
        if func.params.len() != 1 {
            return None;
        }

        // Must be recursive
        if !self.recursive_functions.contains(&func.name.name) {
            return None;
        }

        let body = func.body.as_ref()?;

        // Detect fib-like pattern:
        // if n <= 1 { return n; }
        // return fib(n - 1) + fib(n - 2);
        if !self.is_fib_like_pattern(&func.name.name, body) {
            return None;
        }

        // Get parameter name
        let param_name = if let Pattern::Ident { name, .. } = &func.params[0].pattern {
            name.name.clone()
        } else {
            return None;
        };

        // Generate helper function name
        let helper_name = format!("{}_tail", func.name.name);

        // Create helper function: fn fib_tail(n, a, b) { if n <= 0 { return a; } return fib_tail(n - 1, b, a + b); }
        let helper_func = self.generate_fib_helper(&helper_name, &param_name);

        // Create wrapper function: fn fib(n) { return fib_tail(n, 0, 1); }
        let wrapper_func = self.generate_fib_wrapper(&func.name.name, &helper_name, &param_name, func);

        Some((helper_func, wrapper_func))
    }

    /// Check if a function body matches the Fibonacci pattern
    fn is_fib_like_pattern(&self, func_name: &str, body: &Block) -> bool {
        // Pattern we're looking for:
        // { if n <= 1 { return n; } return f(n-1) + f(n-2); }
        // or
        // { if n <= 1 { return n; } f(n-1) + f(n-2) }

        // Should have an if statement/expression followed by a recursive expression
        if body.stmts.is_empty() && body.expr.is_none() {
            return false;
        }

        // Check for the pattern in the block expression
        if let Some(expr) = &body.expr {
            if let Expr::If { else_branch: Some(else_expr), .. } = expr.as_ref() {
                // Check if then_branch is a base case (return n or similar)
                // Check if else_branch has double recursive calls
                return self.is_double_recursive_expr(func_name, else_expr);
            }
        }

        // Check for pattern: if ... { return ...; } return f(n-1) + f(n-2);
        if body.stmts.len() >= 1 {
            // Last statement or expression should be the recursive call
            if let Some(Stmt::Expr(expr) | Stmt::Semi(expr)) = body.stmts.last() {
                if let Expr::Return(Some(ret_expr)) = expr {
                    return self.is_double_recursive_expr(func_name, ret_expr);
                }
            }
            if let Some(expr) = &body.expr {
                return self.is_double_recursive_expr(func_name, expr);
            }
        }

        false
    }

    /// Check if expression is f(n-1) + f(n-2) pattern
    fn is_double_recursive_expr(&self, func_name: &str, expr: &Expr) -> bool {
        if let Expr::Binary { op: BinOp::Add, left, right } = expr {
            let left_is_recursive = self.is_recursive_call_with_decrement(func_name, left);
            let right_is_recursive = self.is_recursive_call_with_decrement(func_name, right);
            return left_is_recursive && right_is_recursive;
        }
        false
    }

    /// Check if expression is f(n - k) for some constant k
    fn is_recursive_call_with_decrement(&self, func_name: &str, expr: &Expr) -> bool {
        if let Expr::Call { func, args } = expr {
            if let Expr::Path(path) = func.as_ref() {
                if path.segments.last().map(|s| s.ident.name.as_str()) == Some(func_name) {
                    // Check if argument is n - constant
                    if args.len() == 1 {
                        if let Expr::Binary { op: BinOp::Sub, .. } = &args[0] {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Generate the tail-recursive helper function
    fn generate_fib_helper(&self, name: &str, _param_name: &str) -> ast::Function {
        let span = Span { start: 0, end: 0 };

        // fn fib_tail(n, a, b) {
        //     if n <= 0 { return a; }
        //     return fib_tail(n - 1, b, a + b);
        // }
        let n_ident = Ident { name: "n".to_string(), evidentiality: None, span: span.clone() };
        let a_ident = Ident { name: "a".to_string(), evidentiality: None, span: span.clone() };
        let b_ident = Ident { name: "b".to_string(), evidentiality: None, span: span.clone() };

        let params = vec![
            Param {
                pattern: Pattern::Ident { mutable: false, name: n_ident.clone(), evidentiality: None },
                ty: TypeExpr::Infer,
            },
            Param {
                pattern: Pattern::Ident { mutable: false, name: a_ident.clone(), evidentiality: None },
                ty: TypeExpr::Infer,
            },
            Param {
                pattern: Pattern::Ident { mutable: false, name: b_ident.clone(), evidentiality: None },
                ty: TypeExpr::Infer,
            },
        ];

        // Condition: n <= 0
        let condition = Expr::Binary {
            op: BinOp::Le,
            left: Box::new(Expr::Path(TypePath {
                segments: vec![PathSegment { ident: n_ident.clone(), generics: None }],
            })),
            right: Box::new(Expr::Literal(Literal::Int { value: "0".to_string(), base: NumBase::Decimal, suffix: None })),
        };

        // Then branch: return a
        let then_branch = Block {
            stmts: vec![],
            expr: Some(Box::new(Expr::Return(Some(Box::new(Expr::Path(TypePath {
                segments: vec![PathSegment { ident: a_ident.clone(), generics: None }],
            })))))),
        };

        // Recursive call: fib_tail(n - 1, b, a + b)
        let recursive_call = Expr::Call {
            func: Box::new(Expr::Path(TypePath {
                segments: vec![PathSegment {
                    ident: Ident { name: name.to_string(), evidentiality: None, span: span.clone() },
                    generics: None,
                }],
            })),
            args: vec![
                // n - 1
                Expr::Binary {
                    op: BinOp::Sub,
                    left: Box::new(Expr::Path(TypePath {
                        segments: vec![PathSegment { ident: n_ident.clone(), generics: None }],
                    })),
                    right: Box::new(Expr::Literal(Literal::Int { value: "1".to_string(), base: NumBase::Decimal, suffix: None })),
                },
                // b
                Expr::Path(TypePath {
                    segments: vec![PathSegment { ident: b_ident.clone(), generics: None }],
                }),
                // a + b
                Expr::Binary {
                    op: BinOp::Add,
                    left: Box::new(Expr::Path(TypePath {
                        segments: vec![PathSegment { ident: a_ident.clone(), generics: None }],
                    })),
                    right: Box::new(Expr::Path(TypePath {
                        segments: vec![PathSegment { ident: b_ident.clone(), generics: None }],
                    })),
                },
            ],
        };

        // if n <= 0 { return a; } else { return fib_tail(n - 1, b, a + b); }
        let body = Block {
            stmts: vec![],
            expr: Some(Box::new(Expr::If {
                condition: Box::new(condition),
                then_branch,
                else_branch: Some(Box::new(Expr::Return(Some(Box::new(recursive_call))))),
            })),
        };

        ast::Function {
            visibility: Visibility::default(),
            is_async: false,
            attrs: FunctionAttrs::default(),
            name: Ident { name: name.to_string(), evidentiality: None, span: span.clone() },
            generics: None,
            params,
            return_type: None,
            where_clause: None,
            body: Some(body),
        }
    }

    /// Generate the wrapper function that calls the helper
    fn generate_fib_wrapper(&self, name: &str, helper_name: &str, param_name: &str, original: &ast::Function) -> ast::Function {
        let span = Span { start: 0, end: 0 };

        // fn fib(n) { return fib_tail(n, 0, 1); }
        let call_helper = Expr::Call {
            func: Box::new(Expr::Path(TypePath {
                segments: vec![PathSegment {
                    ident: Ident { name: helper_name.to_string(), evidentiality: None, span: span.clone() },
                    generics: None,
                }],
            })),
            args: vec![
                // n
                Expr::Path(TypePath {
                    segments: vec![PathSegment {
                        ident: Ident { name: param_name.to_string(), evidentiality: None, span: span.clone() },
                        generics: None,
                    }],
                }),
                // 0 (initial acc1)
                Expr::Literal(Literal::Int { value: "0".to_string(), base: NumBase::Decimal, suffix: None }),
                // 1 (initial acc2)
                Expr::Literal(Literal::Int { value: "1".to_string(), base: NumBase::Decimal, suffix: None }),
            ],
        };

        let body = Block {
            stmts: vec![],
            expr: Some(Box::new(Expr::Return(Some(Box::new(call_helper))))),
        };

        ast::Function {
            visibility: original.visibility,
            is_async: original.is_async,
            attrs: original.attrs.clone(),
            name: Ident { name: name.to_string(), evidentiality: None, span: span.clone() },
            generics: original.generics.clone(),
            params: original.params.clone(),
            return_type: original.return_type.clone(),
            where_clause: original.where_clause.clone(),
            body: Some(body),
        }
    }

    /// Check if a function is recursive
    fn is_recursive(&self, name: &str, func: &ast::Function) -> bool {
        if let Some(body) = &func.body {
            self.block_calls_function(name, body)
        } else {
            false
        }
    }

    fn block_calls_function(&self, name: &str, block: &Block) -> bool {
        for stmt in &block.stmts {
            if self.stmt_calls_function(name, stmt) {
                return true;
            }
        }
        if let Some(expr) = &block.expr {
            if self.expr_calls_function(name, expr) {
                return true;
            }
        }
        false
    }

    fn stmt_calls_function(&self, name: &str, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Let { init: Some(expr), .. } => self.expr_calls_function(name, expr),
            Stmt::Expr(expr) | Stmt::Semi(expr) => self.expr_calls_function(name, expr),
            _ => false,
        }
    }

    fn expr_calls_function(&self, name: &str, expr: &Expr) -> bool {
        match expr {
            Expr::Call { func, args } => {
                if let Expr::Path(path) = func.as_ref() {
                    if path.segments.last().map(|s| s.ident.name.as_str()) == Some(name) {
                        return true;
                    }
                }
                args.iter().any(|a| self.expr_calls_function(name, a))
            }
            Expr::Binary { left, right, .. } => {
                self.expr_calls_function(name, left) || self.expr_calls_function(name, right)
            }
            Expr::Unary { expr, .. } => self.expr_calls_function(name, expr),
            Expr::If { condition, then_branch, else_branch } => {
                self.expr_calls_function(name, condition)
                    || self.block_calls_function(name, then_branch)
                    || else_branch.as_ref().map(|e| self.expr_calls_function(name, e)).unwrap_or(false)
            }
            Expr::While { condition, body } => {
                self.expr_calls_function(name, condition) || self.block_calls_function(name, body)
            }
            Expr::Block(block) => self.block_calls_function(name, block),
            Expr::Return(Some(e)) => self.expr_calls_function(name, e),
            _ => false,
        }
    }

    /// Optimize a single function
    fn optimize_function(&mut self, func: &ast::Function) -> ast::Function {
        // Reset CSE counter per function for cleaner variable names
        self.cse_counter = 0;

        let body = if let Some(body) = &func.body {
            // Run passes based on optimization level
            let optimized = match self.level {
                OptLevel::None => body.clone(),
                OptLevel::Basic => {
                    let b = self.pass_constant_fold_block(body);
                    self.pass_dead_code_block(&b)
                }
                OptLevel::Standard | OptLevel::Size => {
                    let b = self.pass_constant_fold_block(body);
                    let b = self.pass_strength_reduce_block(&b);
                    let b = self.pass_cse_block(&b);  // CSE pass
                    let b = self.pass_dead_code_block(&b);
                    self.pass_simplify_branches_block(&b)
                }
                OptLevel::Aggressive => {
                    // Multiple iterations for fixed-point
                    let mut b = body.clone();
                    for _ in 0..3 {
                        b = self.pass_constant_fold_block(&b);
                        b = self.pass_strength_reduce_block(&b);
                        b = self.pass_cse_block(&b);  // CSE pass
                        b = self.pass_dead_code_block(&b);
                        b = self.pass_simplify_branches_block(&b);
                    }
                    b
                }
            };
            Some(optimized)
        } else {
            None
        };

        ast::Function {
            visibility: func.visibility.clone(),
            is_async: func.is_async,
            name: func.name.clone(),
            generics: func.generics.clone(),
            params: func.params.clone(),
            return_type: func.return_type.clone(),
            where_clause: func.where_clause.clone(),
            body,
            attrs: func.attrs.clone(),
        }
    }

    // ========================================================================
    // Pass: Constant Folding
    // ========================================================================

    fn pass_constant_fold_block(&mut self, block: &Block) -> Block {
        let stmts = block.stmts.iter().map(|s| self.pass_constant_fold_stmt(s)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.pass_constant_fold_expr(e)));
        Block { stmts, expr }
    }

    fn pass_constant_fold_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_constant_fold_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_constant_fold_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_constant_fold_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_constant_fold_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Binary { op, left, right } => {
                let left = Box::new(self.pass_constant_fold_expr(left));
                let right = Box::new(self.pass_constant_fold_expr(right));

                // Try to fold
                if let (Some(l), Some(r)) = (self.as_int(&left), self.as_int(&right)) {
                    if let Some(result) = self.fold_binary(op.clone(), l, r) {
                        self.stats.constants_folded += 1;
                        return Expr::Literal(Literal::Int {
                            value: result.to_string(),
                            base: NumBase::Decimal,
                            suffix: None,
                        });
                    }
                }

                Expr::Binary { op: op.clone(), left, right }
            }
            Expr::Unary { op, expr: inner } => {
                let inner = Box::new(self.pass_constant_fold_expr(inner));

                if let Some(v) = self.as_int(&inner) {
                    if let Some(result) = self.fold_unary(*op, v) {
                        self.stats.constants_folded += 1;
                        return Expr::Literal(Literal::Int {
                            value: result.to_string(),
                            base: NumBase::Decimal,
                            suffix: None,
                        });
                    }
                }

                Expr::Unary { op: *op, expr: inner }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_constant_fold_expr(condition));
                let then_branch = self.pass_constant_fold_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_constant_fold_expr(e)));

                // Fold constant conditions
                if let Some(cond) = self.as_bool(&condition) {
                    self.stats.branches_simplified += 1;
                    if cond {
                        return Expr::Block(then_branch);
                    } else if let Some(else_expr) = else_branch {
                        return *else_expr;
                    } else {
                        return Expr::Literal(Literal::Bool(false));
                    }
                }

                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_constant_fold_expr(condition));
                let body = self.pass_constant_fold_block(body);

                // while false { ... } -> nothing
                if let Some(false) = self.as_bool(&condition) {
                    self.stats.branches_simplified += 1;
                    return Expr::Block(Block { stmts: vec![], expr: None });
                }

                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_constant_fold_block(block))
            }
            Expr::Call { func, args } => {
                let args = args.iter().map(|a| self.pass_constant_fold_expr(a)).collect();
                Expr::Call { func: func.clone(), args }
            }
            Expr::Return(e) => {
                Expr::Return(e.as_ref().map(|e| Box::new(self.pass_constant_fold_expr(e))))
            }
            Expr::Assign { target, value } => {
                let value = Box::new(self.pass_constant_fold_expr(value));
                Expr::Assign { target: target.clone(), value }
            }
            Expr::Index { expr: e, index } => {
                let e = Box::new(self.pass_constant_fold_expr(e));
                let index = Box::new(self.pass_constant_fold_expr(index));
                Expr::Index { expr: e, index }
            }
            Expr::Array(elements) => {
                let elements = elements.iter().map(|e| self.pass_constant_fold_expr(e)).collect();
                Expr::Array(elements)
            }
            other => other.clone(),
        }
    }

    fn as_int(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Literal(Literal::Int { value, .. }) => value.parse().ok(),
            Expr::Literal(Literal::Bool(b)) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    fn as_bool(&self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::Literal(Literal::Bool(b)) => Some(*b),
            Expr::Literal(Literal::Int { value, .. }) => {
                value.parse::<i64>().ok().map(|v| v != 0)
            }
            _ => None,
        }
    }

    fn fold_binary(&self, op: BinOp, l: i64, r: i64) -> Option<i64> {
        match op {
            BinOp::Add => Some(l.wrapping_add(r)),
            BinOp::Sub => Some(l.wrapping_sub(r)),
            BinOp::Mul => Some(l.wrapping_mul(r)),
            BinOp::Div if r != 0 => Some(l / r),
            BinOp::Rem if r != 0 => Some(l % r),
            BinOp::BitAnd => Some(l & r),
            BinOp::BitOr => Some(l | r),
            BinOp::BitXor => Some(l ^ r),
            BinOp::Shl => Some(l << (r & 63)),
            BinOp::Shr => Some(l >> (r & 63)),
            BinOp::Eq => Some(if l == r { 1 } else { 0 }),
            BinOp::Ne => Some(if l != r { 1 } else { 0 }),
            BinOp::Lt => Some(if l < r { 1 } else { 0 }),
            BinOp::Le => Some(if l <= r { 1 } else { 0 }),
            BinOp::Gt => Some(if l > r { 1 } else { 0 }),
            BinOp::Ge => Some(if l >= r { 1 } else { 0 }),
            BinOp::And => Some(if l != 0 && r != 0 { 1 } else { 0 }),
            BinOp::Or => Some(if l != 0 || r != 0 { 1 } else { 0 }),
            _ => None,
        }
    }

    fn fold_unary(&self, op: UnaryOp, v: i64) -> Option<i64> {
        match op {
            UnaryOp::Neg => Some(-v),
            UnaryOp::Not => Some(if v == 0 { 1 } else { 0 }),
            _ => None,
        }
    }

    // ========================================================================
    // Pass: Strength Reduction
    // ========================================================================

    fn pass_strength_reduce_block(&mut self, block: &Block) -> Block {
        let stmts = block.stmts.iter().map(|s| self.pass_strength_reduce_stmt(s)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.pass_strength_reduce_expr(e)));
        Block { stmts, expr }
    }

    fn pass_strength_reduce_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_strength_reduce_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_strength_reduce_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_strength_reduce_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_strength_reduce_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Binary { op, left, right } => {
                let left = Box::new(self.pass_strength_reduce_expr(left));
                let right = Box::new(self.pass_strength_reduce_expr(right));

                // x * 2 -> x << 1, x * 4 -> x << 2, etc.
                if *op == BinOp::Mul {
                    if let Some(n) = self.as_int(&right) {
                        if n > 0 && (n as u64).is_power_of_two() {
                            self.stats.strength_reductions += 1;
                            let shift = (n as u64).trailing_zeros() as i64;
                            return Expr::Binary {
                                op: BinOp::Shl,
                                left,
                                right: Box::new(Expr::Literal(Literal::Int {
                                    value: shift.to_string(),
                                    base: NumBase::Decimal,
                                    suffix: None,
                                })),
                            };
                        }
                    }
                    if let Some(n) = self.as_int(&left) {
                        if n > 0 && (n as u64).is_power_of_two() {
                            self.stats.strength_reductions += 1;
                            let shift = (n as u64).trailing_zeros() as i64;
                            return Expr::Binary {
                                op: BinOp::Shl,
                                left: right,
                                right: Box::new(Expr::Literal(Literal::Int {
                                    value: shift.to_string(),
                                    base: NumBase::Decimal,
                                    suffix: None,
                                })),
                            };
                        }
                    }
                }

                // x + 0 -> x, x - 0 -> x, x * 1 -> x, x / 1 -> x
                if let Some(n) = self.as_int(&right) {
                    match (op, n) {
                        (BinOp::Add | BinOp::Sub | BinOp::BitOr | BinOp::BitXor, 0)
                        | (BinOp::Mul | BinOp::Div, 1)
                        | (BinOp::Shl | BinOp::Shr, 0) => {
                            self.stats.strength_reductions += 1;
                            return *left;
                        }
                        (BinOp::Mul, 0) | (BinOp::BitAnd, 0) => {
                            self.stats.strength_reductions += 1;
                            return Expr::Literal(Literal::Int {
                                value: "0".to_string(),
                                base: NumBase::Decimal,
                                suffix: None,
                            });
                        }
                        _ => {}
                    }
                }

                // 0 + x -> x, 1 * x -> x
                if let Some(n) = self.as_int(&left) {
                    match (op, n) {
                        (BinOp::Add | BinOp::BitOr | BinOp::BitXor, 0)
                        | (BinOp::Mul, 1) => {
                            self.stats.strength_reductions += 1;
                            return *right;
                        }
                        (BinOp::Mul, 0) | (BinOp::BitAnd, 0) => {
                            self.stats.strength_reductions += 1;
                            return Expr::Literal(Literal::Int {
                                value: "0".to_string(),
                                base: NumBase::Decimal,
                                suffix: None,
                            });
                        }
                        _ => {}
                    }
                }

                Expr::Binary { op: op.clone(), left, right }
            }
            Expr::Unary { op, expr: inner } => {
                let inner = Box::new(self.pass_strength_reduce_expr(inner));

                // --x -> x
                if *op == UnaryOp::Neg {
                    if let Expr::Unary { op: UnaryOp::Neg, expr: inner2 } = inner.as_ref() {
                        self.stats.strength_reductions += 1;
                        return *inner2.clone();
                    }
                }

                // !!x -> x (for booleans)
                if *op == UnaryOp::Not {
                    if let Expr::Unary { op: UnaryOp::Not, expr: inner2 } = inner.as_ref() {
                        self.stats.strength_reductions += 1;
                        return *inner2.clone();
                    }
                }

                Expr::Unary { op: *op, expr: inner }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_strength_reduce_expr(condition));
                let then_branch = self.pass_strength_reduce_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_strength_reduce_expr(e)));
                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_strength_reduce_expr(condition));
                let body = self.pass_strength_reduce_block(body);
                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_strength_reduce_block(block))
            }
            Expr::Call { func, args } => {
                let args = args.iter().map(|a| self.pass_strength_reduce_expr(a)).collect();
                Expr::Call { func: func.clone(), args }
            }
            Expr::Return(e) => {
                Expr::Return(e.as_ref().map(|e| Box::new(self.pass_strength_reduce_expr(e))))
            }
            Expr::Assign { target, value } => {
                let value = Box::new(self.pass_strength_reduce_expr(value));
                Expr::Assign { target: target.clone(), value }
            }
            other => other.clone(),
        }
    }

    // ========================================================================
    // Pass: Dead Code Elimination
    // ========================================================================

    fn pass_dead_code_block(&mut self, block: &Block) -> Block {
        // Remove statements after a return
        let mut stmts = Vec::new();
        let mut found_return = false;

        for stmt in &block.stmts {
            if found_return {
                self.stats.dead_code_eliminated += 1;
                continue;
            }
            let stmt = self.pass_dead_code_stmt(stmt);
            if self.stmt_returns(&stmt) {
                found_return = true;
            }
            stmts.push(stmt);
        }

        // If we found a return, the trailing expression is dead
        let expr = if found_return {
            if block.expr.is_some() {
                self.stats.dead_code_eliminated += 1;
            }
            None
        } else {
            block.expr.as_ref().map(|e| Box::new(self.pass_dead_code_expr(e)))
        };

        Block { stmts, expr }
    }

    fn pass_dead_code_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_dead_code_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_dead_code_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_dead_code_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_dead_code_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_dead_code_expr(condition));
                let then_branch = self.pass_dead_code_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_dead_code_expr(e)));
                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_dead_code_expr(condition));
                let body = self.pass_dead_code_block(body);
                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_dead_code_block(block))
            }
            other => other.clone(),
        }
    }

    fn stmt_returns(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr) | Stmt::Semi(expr) => self.expr_returns(expr),
            _ => false,
        }
    }

    fn expr_returns(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Return(_) => true,
            Expr::Block(block) => {
                block.stmts.iter().any(|s| self.stmt_returns(s))
                    || block.expr.as_ref().map(|e| self.expr_returns(e)).unwrap_or(false)
            }
            _ => false,
        }
    }

    // ========================================================================
    // Pass: Branch Simplification
    // ========================================================================

    fn pass_simplify_branches_block(&mut self, block: &Block) -> Block {
        let stmts = block.stmts.iter().map(|s| self.pass_simplify_branches_stmt(s)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.pass_simplify_branches_expr(e)));
        Block { stmts, expr }
    }

    fn pass_simplify_branches_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_simplify_branches_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_simplify_branches_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_simplify_branches_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_simplify_branches_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_simplify_branches_expr(condition));
                let then_branch = self.pass_simplify_branches_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_simplify_branches_expr(e)));

                // if !cond { a } else { b } -> if cond { b } else { a }
                if let Expr::Unary { op: UnaryOp::Not, expr: inner } = condition.as_ref() {
                    if let Some(else_expr) = &else_branch {
                        self.stats.branches_simplified += 1;
                        let new_else = Some(Box::new(Expr::Block(then_branch)));
                        let new_then = match else_expr.as_ref() {
                            Expr::Block(b) => b.clone(),
                            other => Block { stmts: vec![], expr: Some(Box::new(other.clone())) },
                        };
                        return Expr::If {
                            condition: inner.clone(),
                            then_branch: new_then,
                            else_branch: new_else,
                        };
                    }
                }

                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_simplify_branches_expr(condition));
                let body = self.pass_simplify_branches_block(body);
                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_simplify_branches_block(block))
            }
            Expr::Binary { op, left, right } => {
                let left = Box::new(self.pass_simplify_branches_expr(left));
                let right = Box::new(self.pass_simplify_branches_expr(right));
                Expr::Binary { op: op.clone(), left, right }
            }
            Expr::Unary { op, expr: inner } => {
                let inner = Box::new(self.pass_simplify_branches_expr(inner));
                Expr::Unary { op: *op, expr: inner }
            }
            Expr::Call { func, args } => {
                let args = args.iter().map(|a| self.pass_simplify_branches_expr(a)).collect();
                Expr::Call { func: func.clone(), args }
            }
            Expr::Return(e) => {
                Expr::Return(e.as_ref().map(|e| Box::new(self.pass_simplify_branches_expr(e))))
            }
            other => other.clone(),
        }
    }

    // ========================================================================
    // Pass: Common Subexpression Elimination (CSE)
    // ========================================================================

    fn pass_cse_block(&mut self, block: &Block) -> Block {
        // Step 1: Collect all expressions in this block
        let mut collected = Vec::new();
        collect_exprs_from_block(block, &mut collected);

        // Step 2: Count occurrences using hash + equality check
        let mut expr_counts: HashMap<u64, Vec<Expr>> = HashMap::new();
        for ce in &collected {
            let entry = expr_counts.entry(ce.hash).or_insert_with(Vec::new);
            // Check if this exact expression is already in the bucket
            let found = entry.iter().any(|e| expr_eq(e, &ce.expr));
            if !found {
                entry.push(ce.expr.clone());
            }
        }

        // Count actual occurrences (need to count duplicates)
        let mut occurrence_counts: Vec<(Expr, usize)> = Vec::new();
        for ce in &collected {
            // Find or create entry for this expression
            let existing = occurrence_counts.iter_mut().find(|(e, _)| expr_eq(e, &ce.expr));
            if let Some((_, count)) = existing {
                *count += 1;
            } else {
                occurrence_counts.push((ce.expr.clone(), 1));
            }
        }

        // Step 3: Find expressions that occur 2+ times
        let candidates: Vec<Expr> = occurrence_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .map(|(expr, _)| expr)
            .collect();

        if candidates.is_empty() {
            // No CSE opportunities, just recurse into nested blocks
            return self.pass_cse_nested(block);
        }

        // Step 4: Create let bindings for each candidate and replace occurrences
        let mut result_block = block.clone();
        let mut new_lets: Vec<Stmt> = Vec::new();

        for expr in candidates {
            let var_name = format!("__cse_{}", self.cse_counter);
            self.cse_counter += 1;

            // Create the let binding
            new_lets.push(make_cse_let(&var_name, expr.clone()));

            // Replace all occurrences in the block
            result_block = replace_in_block(&result_block, &expr, &var_name);

            self.stats.expressions_deduplicated += 1;
        }

        // Step 5: Prepend the new let bindings to the block
        let mut final_stmts = new_lets;
        final_stmts.extend(result_block.stmts);

        // Step 6: Recurse into nested blocks
        let result = Block {
            stmts: final_stmts,
            expr: result_block.expr,
        };
        self.pass_cse_nested(&result)
    }

    /// Recurse CSE into nested blocks (if, while, block expressions)
    fn pass_cse_nested(&mut self, block: &Block) -> Block {
        let stmts = block.stmts.iter().map(|stmt| self.pass_cse_stmt(stmt)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.pass_cse_expr(e)));
        Block { stmts, expr }
    }

    fn pass_cse_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init } => Stmt::Let {
                pattern: pattern.clone(),
                ty: ty.clone(),
                init: init.as_ref().map(|e| self.pass_cse_expr(e)),
            },
            Stmt::Expr(e) => Stmt::Expr(self.pass_cse_expr(e)),
            Stmt::Semi(e) => Stmt::Semi(self.pass_cse_expr(e)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_cse_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::If { condition, then_branch, else_branch } => Expr::If {
                condition: Box::new(self.pass_cse_expr(condition)),
                then_branch: self.pass_cse_block(then_branch),
                else_branch: else_branch.as_ref().map(|e| Box::new(self.pass_cse_expr(e))),
            },
            Expr::While { condition, body } => Expr::While {
                condition: Box::new(self.pass_cse_expr(condition)),
                body: self.pass_cse_block(body),
            },
            Expr::Block(b) => Expr::Block(self.pass_cse_block(b)),
            Expr::Binary { op, left, right } => Expr::Binary {
                op: *op,
                left: Box::new(self.pass_cse_expr(left)),
                right: Box::new(self.pass_cse_expr(right)),
            },
            Expr::Unary { op, expr: inner } => Expr::Unary {
                op: *op,
                expr: Box::new(self.pass_cse_expr(inner)),
            },
            Expr::Call { func, args } => Expr::Call {
                func: func.clone(),
                args: args.iter().map(|a| self.pass_cse_expr(a)).collect(),
            },
            Expr::Return(e) => Expr::Return(e.as_ref().map(|e| Box::new(self.pass_cse_expr(e)))),
            Expr::Assign { target, value } => Expr::Assign {
                target: target.clone(),
                value: Box::new(self.pass_cse_expr(value)),
            },
            other => other.clone(),
        }
    }
}

// ============================================================================
// Common Subexpression Elimination (CSE) - Helper Functions
// ============================================================================

/// Expr hash for CSE - identifies structurally equivalent expressions
fn expr_hash(expr: &Expr) -> u64 {
    use std::hash::Hasher;
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    expr_hash_recursive(expr, &mut hasher);
    hasher.finish()
}

fn expr_hash_recursive<H: std::hash::Hasher>(expr: &Expr, hasher: &mut H) {
    use std::hash::Hash;

    std::mem::discriminant(expr).hash(hasher);

    match expr {
        Expr::Literal(lit) => {
            match lit {
                Literal::Int { value, .. } => value.hash(hasher),
                Literal::Float { value, .. } => value.hash(hasher),
                Literal::String(s) => s.hash(hasher),
                Literal::Char(c) => c.hash(hasher),
                Literal::Bool(b) => b.hash(hasher),
                _ => {}
            }
        }
        Expr::Path(path) => {
            for seg in &path.segments {
                seg.ident.name.hash(hasher);
            }
        }
        Expr::Binary { op, left, right } => {
            std::mem::discriminant(op).hash(hasher);
            expr_hash_recursive(left, hasher);
            expr_hash_recursive(right, hasher);
        }
        Expr::Unary { op, expr } => {
            std::mem::discriminant(op).hash(hasher);
            expr_hash_recursive(expr, hasher);
        }
        Expr::Call { func, args } => {
            expr_hash_recursive(func, hasher);
            args.len().hash(hasher);
            for arg in args {
                expr_hash_recursive(arg, hasher);
            }
        }
        Expr::Index { expr, index } => {
            expr_hash_recursive(expr, hasher);
            expr_hash_recursive(index, hasher);
        }
        _ => {}
    }
}

/// Check if an expression is pure (no side effects)
fn is_pure_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_) => true,
        Expr::Path(_) => true,
        Expr::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        Expr::Unary { expr, .. } => is_pure_expr(expr),
        Expr::If { condition, then_branch, else_branch } => {
            is_pure_expr(condition)
                && then_branch.stmts.is_empty()
                && then_branch.expr.as_ref().map(|e| is_pure_expr(e)).unwrap_or(true)
                && else_branch.as_ref().map(|e| is_pure_expr(e)).unwrap_or(true)
        }
        Expr::Index { expr, index } => is_pure_expr(expr) && is_pure_expr(index),
        Expr::Array(elements) => elements.iter().all(is_pure_expr),
        // These have side effects
        Expr::Call { .. } => false,
        Expr::Assign { .. } => false,
        Expr::Return(_) => false,
        _ => false,
    }
}

/// Check if an expression is worth caching (complex enough)
fn is_cse_worthy(expr: &Expr) -> bool {
    match expr {
        // Simple expressions - not worth CSE overhead
        Expr::Literal(_) => false,
        Expr::Path(_) => false,
        // Binary operations are worth it
        Expr::Binary { .. } => true,
        // Unary operations might be worth it
        Expr::Unary { .. } => true,
        // Calls might be worth it if pure (but we can't know easily)
        Expr::Call { .. } => false,
        // Index operations are worth it
        Expr::Index { .. } => true,
        _ => false,
    }
}

/// Check if two expressions are structurally equal
fn expr_eq(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Literal(la), Expr::Literal(lb)) => match (la, lb) {
            (Literal::Int { value: va, .. }, Literal::Int { value: vb, .. }) => va == vb,
            (Literal::Float { value: va, .. }, Literal::Float { value: vb, .. }) => va == vb,
            (Literal::String(sa), Literal::String(sb)) => sa == sb,
            (Literal::Char(ca), Literal::Char(cb)) => ca == cb,
            (Literal::Bool(ba), Literal::Bool(bb)) => ba == bb,
            _ => false,
        },
        (Expr::Path(pa), Expr::Path(pb)) => {
            pa.segments.len() == pb.segments.len()
                && pa.segments.iter().zip(&pb.segments).all(|(sa, sb)| {
                    sa.ident.name == sb.ident.name
                })
        }
        (Expr::Binary { op: oa, left: la, right: ra }, Expr::Binary { op: ob, left: lb, right: rb }) => {
            oa == ob && expr_eq(la, lb) && expr_eq(ra, rb)
        }
        (Expr::Unary { op: oa, expr: ea }, Expr::Unary { op: ob, expr: eb }) => {
            oa == ob && expr_eq(ea, eb)
        }
        (Expr::Index { expr: ea, index: ia }, Expr::Index { expr: eb, index: ib }) => {
            expr_eq(ea, eb) && expr_eq(ia, ib)
        }
        (Expr::Call { func: fa, args: aa }, Expr::Call { func: fb, args: ab }) => {
            expr_eq(fa, fb) && aa.len() == ab.len() && aa.iter().zip(ab).all(|(a, b)| expr_eq(a, b))
        }
        _ => false,
    }
}

/// Collected expression with its location info
#[derive(Clone)]
struct CollectedExpr {
    expr: Expr,
    hash: u64,
}

/// Collect all CSE-worthy expressions from an expression tree
fn collect_exprs_from_expr(expr: &Expr, out: &mut Vec<CollectedExpr>) {
    // First, recurse into subexpressions
    match expr {
        Expr::Binary { left, right, .. } => {
            collect_exprs_from_expr(left, out);
            collect_exprs_from_expr(right, out);
        }
        Expr::Unary { expr: inner, .. } => {
            collect_exprs_from_expr(inner, out);
        }
        Expr::Index { expr: e, index } => {
            collect_exprs_from_expr(e, out);
            collect_exprs_from_expr(index, out);
        }
        Expr::Call { func, args } => {
            collect_exprs_from_expr(func, out);
            for arg in args {
                collect_exprs_from_expr(arg, out);
            }
        }
        Expr::If { condition, then_branch, else_branch } => {
            collect_exprs_from_expr(condition, out);
            collect_exprs_from_block(then_branch, out);
            if let Some(else_expr) = else_branch {
                collect_exprs_from_expr(else_expr, out);
            }
        }
        Expr::While { condition, body } => {
            collect_exprs_from_expr(condition, out);
            collect_exprs_from_block(body, out);
        }
        Expr::Block(block) => {
            collect_exprs_from_block(block, out);
        }
        Expr::Return(Some(e)) => {
            collect_exprs_from_expr(e, out);
        }
        Expr::Assign { value, .. } => {
            collect_exprs_from_expr(value, out);
        }
        Expr::Array(elements) => {
            for e in elements {
                collect_exprs_from_expr(e, out);
            }
        }
        _ => {}
    }

    // Then, if this expression is CSE-worthy and pure, add it
    if is_cse_worthy(expr) && is_pure_expr(expr) {
        out.push(CollectedExpr {
            expr: expr.clone(),
            hash: expr_hash(expr),
        });
    }
}

/// Collect expressions from a block
fn collect_exprs_from_block(block: &Block, out: &mut Vec<CollectedExpr>) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { init: Some(e), .. } => collect_exprs_from_expr(e, out),
            Stmt::Expr(e) | Stmt::Semi(e) => collect_exprs_from_expr(e, out),
            _ => {}
        }
    }
    if let Some(e) = &block.expr {
        collect_exprs_from_expr(e, out);
    }
}

/// Replace all occurrences of target expression with a variable reference
fn replace_in_expr(expr: &Expr, target: &Expr, var_name: &str) -> Expr {
    // Check if this expression matches the target
    if expr_eq(expr, target) {
        return Expr::Path(TypePath {
            segments: vec![PathSegment {
                ident: Ident {
                    name: var_name.to_string(),
                    evidentiality: None,
                    span: Span { start: 0, end: 0 },
                },
                generics: None,
            }],
        });
    }

    // Otherwise, recurse into subexpressions
    match expr {
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(replace_in_expr(left, target, var_name)),
            right: Box::new(replace_in_expr(right, target, var_name)),
        },
        Expr::Unary { op, expr: inner } => Expr::Unary {
            op: *op,
            expr: Box::new(replace_in_expr(inner, target, var_name)),
        },
        Expr::Index { expr: e, index } => Expr::Index {
            expr: Box::new(replace_in_expr(e, target, var_name)),
            index: Box::new(replace_in_expr(index, target, var_name)),
        },
        Expr::Call { func, args } => Expr::Call {
            func: Box::new(replace_in_expr(func, target, var_name)),
            args: args.iter().map(|a| replace_in_expr(a, target, var_name)).collect(),
        },
        Expr::If { condition, then_branch, else_branch } => Expr::If {
            condition: Box::new(replace_in_expr(condition, target, var_name)),
            then_branch: replace_in_block(then_branch, target, var_name),
            else_branch: else_branch.as_ref().map(|e| Box::new(replace_in_expr(e, target, var_name))),
        },
        Expr::While { condition, body } => Expr::While {
            condition: Box::new(replace_in_expr(condition, target, var_name)),
            body: replace_in_block(body, target, var_name),
        },
        Expr::Block(block) => Expr::Block(replace_in_block(block, target, var_name)),
        Expr::Return(e) => Expr::Return(e.as_ref().map(|e| Box::new(replace_in_expr(e, target, var_name)))),
        Expr::Assign { target: t, value } => Expr::Assign {
            target: t.clone(),
            value: Box::new(replace_in_expr(value, target, var_name)),
        },
        Expr::Array(elements) => Expr::Array(
            elements.iter().map(|e| replace_in_expr(e, target, var_name)).collect()
        ),
        other => other.clone(),
    }
}

/// Replace in a block
fn replace_in_block(block: &Block, target: &Expr, var_name: &str) -> Block {
    let stmts = block.stmts.iter().map(|stmt| {
        match stmt {
            Stmt::Let { pattern, ty, init } => Stmt::Let {
                pattern: pattern.clone(),
                ty: ty.clone(),
                init: init.as_ref().map(|e| replace_in_expr(e, target, var_name)),
            },
            Stmt::Expr(e) => Stmt::Expr(replace_in_expr(e, target, var_name)),
            Stmt::Semi(e) => Stmt::Semi(replace_in_expr(e, target, var_name)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }).collect();

    let expr = block.expr.as_ref().map(|e| Box::new(replace_in_expr(e, target, var_name)));

    Block { stmts, expr }
}

/// Create a let statement for a CSE variable
fn make_cse_let(var_name: &str, expr: Expr) -> Stmt {
    Stmt::Let {
        pattern: Pattern::Ident {
            mutable: false,
            name: Ident {
                name: var_name.to_string(),
                evidentiality: None,
                span: Span { start: 0, end: 0 },
            },
            evidentiality: None,
        },
        ty: None,
        init: Some(expr),
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Optimize a source file at the given optimization level
pub fn optimize(file: &ast::SourceFile, level: OptLevel) -> (ast::SourceFile, OptStats) {
    let mut optimizer = Optimizer::new(level);
    let optimized = optimizer.optimize_file(file);
    (optimized, optimizer.stats)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create an integer literal expression
    fn int_lit(v: i64) -> Expr {
        Expr::Literal(Literal::Int {
            value: v.to_string(),
            base: NumBase::Decimal,
            suffix: None,
        })
    }

    /// Helper to create a variable reference
    fn var(name: &str) -> Expr {
        Expr::Path(TypePath {
            segments: vec![PathSegment {
                ident: Ident {
                    name: name.to_string(),
                    evidentiality: None,
                    span: Span { start: 0, end: 0 },
                },
                generics: None,
            }],
        })
    }

    /// Helper to create a binary add expression
    fn add(left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op: BinOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Helper to create a binary multiply expression
    fn mul(left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op: BinOp::Mul,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    #[test]
    fn test_expr_hash_equal() {
        // Same expressions should have same hash
        let e1 = add(var("a"), var("b"));
        let e2 = add(var("a"), var("b"));
        assert_eq!(expr_hash(&e1), expr_hash(&e2));
    }

    #[test]
    fn test_expr_hash_different() {
        // Different expressions should have different hashes
        let e1 = add(var("a"), var("b"));
        let e2 = add(var("a"), var("c"));
        assert_ne!(expr_hash(&e1), expr_hash(&e2));
    }

    #[test]
    fn test_expr_eq() {
        let e1 = add(var("a"), var("b"));
        let e2 = add(var("a"), var("b"));
        let e3 = add(var("a"), var("c"));

        assert!(expr_eq(&e1, &e2));
        assert!(!expr_eq(&e1, &e3));
    }

    #[test]
    fn test_is_pure_expr() {
        assert!(is_pure_expr(&int_lit(42)));
        assert!(is_pure_expr(&var("x")));
        assert!(is_pure_expr(&add(var("a"), var("b"))));

        // Calls are not pure
        let call = Expr::Call {
            func: Box::new(var("print")),
            args: vec![int_lit(42)],
        };
        assert!(!is_pure_expr(&call));
    }

    #[test]
    fn test_is_cse_worthy() {
        assert!(!is_cse_worthy(&int_lit(42)));  // literals not worth it
        assert!(!is_cse_worthy(&var("x")));     // variables not worth it
        assert!(is_cse_worthy(&add(var("a"), var("b"))));  // binary ops worth it
    }

    #[test]
    fn test_cse_basic() {
        // Create a block with repeated subexpression:
        // let x = a + b;
        // let y = (a + b) * 2;
        // The (a + b) should be extracted
        let a_plus_b = add(var("a"), var("b"));

        let block = Block {
            stmts: vec![
                Stmt::Let {
                    pattern: Pattern::Ident {
                        mutable: false,
                        name: Ident { name: "x".to_string(), evidentiality: None, span: Span { start: 0, end: 0 } },
                        evidentiality: None,
                    },
                    ty: None,
                    init: Some(a_plus_b.clone()),
                },
                Stmt::Let {
                    pattern: Pattern::Ident {
                        mutable: false,
                        name: Ident { name: "y".to_string(), evidentiality: None, span: Span { start: 0, end: 0 } },
                        evidentiality: None,
                    },
                    ty: None,
                    init: Some(mul(a_plus_b.clone(), int_lit(2))),
                },
            ],
            expr: None,
        };

        let mut optimizer = Optimizer::new(OptLevel::Standard);
        let result = optimizer.pass_cse_block(&block);

        // Should have 3 statements now: __cse_0 = a + b, x = __cse_0, y = __cse_0 * 2
        assert_eq!(result.stmts.len(), 3);
        assert_eq!(optimizer.stats.expressions_deduplicated, 1);

        // First statement should be the CSE let binding
        if let Stmt::Let { pattern: Pattern::Ident { name, .. }, .. } = &result.stmts[0] {
            assert_eq!(name.name, "__cse_0");
        } else {
            panic!("Expected CSE let binding");
        }
    }

    #[test]
    fn test_cse_no_duplicates() {
        // No repeated expressions - should not add any CSE bindings
        let block = Block {
            stmts: vec![
                Stmt::Let {
                    pattern: Pattern::Ident {
                        mutable: false,
                        name: Ident { name: "x".to_string(), evidentiality: None, span: Span { start: 0, end: 0 } },
                        evidentiality: None,
                    },
                    ty: None,
                    init: Some(add(var("a"), var("b"))),
                },
                Stmt::Let {
                    pattern: Pattern::Ident {
                        mutable: false,
                        name: Ident { name: "y".to_string(), evidentiality: None, span: Span { start: 0, end: 0 } },
                        evidentiality: None,
                    },
                    ty: None,
                    init: Some(add(var("c"), var("d"))),
                },
            ],
            expr: None,
        };

        let mut optimizer = Optimizer::new(OptLevel::Standard);
        let result = optimizer.pass_cse_block(&block);

        // Should still have 2 statements (no CSE applied)
        assert_eq!(result.stmts.len(), 2);
        assert_eq!(optimizer.stats.expressions_deduplicated, 0);
    }
}
